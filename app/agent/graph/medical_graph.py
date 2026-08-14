"""End-to-end medical consultation graph."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from app.agent.dynamic_router import route as router_route
from app.agent.recursive_feedback import validate_and_refine
from app.agent.specialist_agents import run_specialist_agent
from app.config import get_settings


class MedicalGraphState(TypedDict):
    user_query: str
    patient_id: Optional[str]
    session_id: Optional[str]
    conversation_history: List[Dict[str, str]]
    pre_routed: Optional[Dict[str, Any]]
    routed_department: str
    intent_distribution: Dict[str, float]
    reasoning: str
    confidence: float
    low_confidence: bool
    human_review_required: bool
    risk_level: str
    retrieved_docs: List[Dict[str, Any]]
    rag_context: str
    response: str
    refined_response: str
    feedback_applied: bool
    recursion_depth: int
    evidence_score: Optional[float]
    contradictions: List[str]
    agent_used: str
    error: Optional[str]


def create_medical_graph():
    graph = StateGraph(MedicalGraphState)
    graph.add_node("route", routing_node)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("generate", generation_node)
    graph.add_node("validate", validation_node)
    graph.add_edge("route", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "validate")
    graph.add_edge("validate", END)
    graph.set_entry_point("route")
    return graph.compile()


_medical_graph = None


def get_medical_graph():
    global _medical_graph
    if _medical_graph is None:
        _medical_graph = create_medical_graph()
    return _medical_graph


def routing_node(state: MedicalGraphState) -> MedicalGraphState:
    pre_routed = state.get("pre_routed")
    if pre_routed:
        # chat 端点已先算过路由，直接复用，避免同一请求打两次 LLM 分类
        for key in (
            "routed_department", "intent_distribution", "reasoning", "confidence",
            "low_confidence", "human_review_required", "risk_level",
        ):
            state[key] = pre_routed.get(key, state.get(key))
        state["error"] = None
        return state
    try:
        result = router_route(state["user_query"], state.get("patient_id"))
        for key in (
            "routed_department", "intent_distribution", "reasoning", "confidence",
            "low_confidence", "human_review_required", "risk_level",
        ):
            state[key] = result.get(key, state.get(key))
        state["error"] = None
    except Exception as exc:
        state.update(
            {
                "routed_department": "全科",
                "intent_distribution": {"全科": 1.0},
                "reasoning": "路由服务不可用，已降级到全科辅助模式",
                "confidence": 0.0,
                "low_confidence": True,
                "human_review_required": True,
                "risk_level": "unknown",
                "error": str(exc),
            }
        )
    return state


def retrieval_node(state: MedicalGraphState) -> MedicalGraphState:
    try:
        from app.service.medical_rag import get_medical_rag_service

        results, context = get_medical_rag_service().retrieve_and_build_context(
            state["user_query"], state.get("routed_department")
        )
        state["retrieved_docs"] = results
        state["rag_context"] = context
        state["error"] = None
    except Exception as exc:
        state["retrieved_docs"] = []
        state["rag_context"] = ""
        state["error"] = str(exc)
    return state


def generation_node(state: MedicalGraphState) -> MedicalGraphState:
    try:
        response = run_specialist_agent(
            state["user_query"],
            state.get("routed_department", "全科"),
            state.get("rag_context", ""),
            state.get("conversation_history", []),
            human_review_required=state.get("human_review_required", False),
        )
        state["response"] = response
        state["refined_response"] = response
        state["agent_used"] = f"{state.get('routed_department', '全科')}SpecialistAgent"
        state["error"] = None
    except Exception as exc:
        state["response"] = "当前模型服务不可用，暂时无法生成回答。请稍后重试。"
        state["refined_response"] = state["response"]
        state["agent_used"] = "FallbackSafetyAgent"
        state["error"] = str(exc)
    return state


def validation_node(state: MedicalGraphState) -> MedicalGraphState:
    result = validate_and_refine(
        response=state.get("response", ""),
        conversation_history=state.get("conversation_history", []),
        max_recursion=get_settings().MAX_RECURSION,
        evidence=state.get("retrieved_docs", []),
    )
    state["refined_response"] = result["refined_response"]
    state["feedback_applied"] = result["feedback_applied"]
    state["recursion_depth"] = result["recursion_depth"]
    state["evidence_score"] = result.get("evidence_score")
    state["contradictions"] = result.get("contradictions", [])
    return state


async def run_medical_query(
    user_query: str,
    patient_id: Optional[str] = None,
    session_id: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    pre_routed: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    initial_state: MedicalGraphState = {
        "user_query": user_query,
        "patient_id": patient_id,
        "session_id": session_id,
        "conversation_history": conversation_history or [],
        "pre_routed": pre_routed,
        "routed_department": "全科",
        "intent_distribution": {},
        "reasoning": "",
        "confidence": 0.0,
        "low_confidence": False,
        "human_review_required": False,
        "risk_level": "normal",
        "retrieved_docs": [],
        "rag_context": "",
        "response": "",
        "refined_response": "",
        "feedback_applied": False,
        "recursion_depth": 0,
        "evidence_score": None,
        "contradictions": [],
        "agent_used": "",
        "error": None,
    }
    result = await asyncio.to_thread(get_medical_graph().invoke, initial_state)
    return {
        "response": result["refined_response"],
        "department": result["routed_department"],
        "agent_used": result["agent_used"],
        "intent_distribution": result["intent_distribution"],
        "reasoning": result["reasoning"],
        "confidence": result["confidence"],
        "low_confidence": result["low_confidence"],
        "human_review_required": result["human_review_required"],
        "risk_level": result["risk_level"],
        "retrieved_docs": result["retrieved_docs"],
        "feedback_applied": result["feedback_applied"],
        "recursion_depth": result["recursion_depth"],
        "evidence_score": result["evidence_score"],
        "contradictions": result["contradictions"],
        "error": result["error"],
    }
