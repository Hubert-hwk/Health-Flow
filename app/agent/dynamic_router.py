"""Triage controller for routing medical-assistance requests."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from app.config import get_settings
from app.data.neo4j_client import get_neo4j_client
from app.model.llm import get_llm_client


DEPARTMENTS = ("内分泌科", "心内科", "消化科", "呼吸科", "全科")
DEPT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "内分泌科": ("血糖", "糖尿病", "甲状腺", "代谢", "胰岛素", "糖化血红蛋白", "尿酸"),
    "心内科": ("血压", "心脏", "血脂", "冠心病", "心电图", "心率", "心肌", "胸痛"),
    "消化科": ("胃", "肠", "胃肠", "肝", "胆", "腹痛", "腹泻", "便秘", "消化", "胰腺"),
    "呼吸科": ("咳嗽", "咽痛", "肺", "气管", "呼吸", "胸闷", "气短", "流感"),
    "全科": ("体检", "健康", "咨询", "报告", "建议", "异常", "怎么办"),
}
DEPARTMENT_ALIASES = {
    "endocrinology": "内分泌科",
    "cardiology": "心内科",
    "gastroenterology": "消化科",
    "respiratory": "呼吸科",
    "general": "全科",
    "内分泌": "内分泌科",
    "心血管": "心内科",
    "消化内科": "消化科",
    "呼吸内科": "呼吸科",
}
HIGH_RISK_TERMS = ("危急值", "胸痛", "呼吸困难", "意识不清", "昏厥", "大出血")


class RouterState(TypedDict):
    user_query: str
    patient_id: Optional[str]
    intent_distribution: Dict[str, float]
    routed_department: str
    reasoning: str
    confidence: float
    related_symptoms: List[Dict[str, Any]]
    low_confidence: bool
    human_review_required: bool
    risk_level: str
    specialist_skills: List[str]


def _normalise_distribution(values: Dict[str, Any]) -> Dict[str, float]:
    clean = {dept: 0.0 for dept in DEPARTMENTS}
    for key, value in values.items():
        department = DEPARTMENT_ALIASES.get(str(key), str(key))
        if department not in clean:
            continue
        try:
            clean[department] = max(0.0, float(value))
        except (TypeError, ValueError):
            continue
    total = sum(clean.values())
    if total <= 0:
        return {dept: (1.0 if dept == "全科" else 0.0) for dept in DEPARTMENTS}
    return {dept: score / total for dept, score in clean.items()}


def _keyword_distribution(query: str) -> Dict[str, float]:
    scores = {dept: 0.0 for dept in DEPARTMENTS}
    for department, keywords in DEPT_KEYWORDS.items():
        scores[department] = float(sum(query.count(keyword) for keyword in keywords))
    if sum(scores.values()) == 0:
        scores["全科"] = 1.0
    return _normalise_distribution(scores)


def _llm_distribution(query: str) -> Optional[Dict[str, float]]:
    prompt = (
        "将用户医疗辅助问题路由到一个或多个科室。只输出 JSON，键可使用 "
        "endocrinology/cardiology/gastroenterology/respiratory/general，值为 0 到 1 的概率。\n"
        f"用户问题：{query}"
    )
    try:
        response = get_llm_client().chat_with_json(
            messages=[
                {"role": "system", "content": "你是医疗分诊控制器，不做诊断。"},
                {"role": "user", "content": prompt},
            ],
            json_schema={"type": "object"},
        )
        if isinstance(response, dict):
            return _normalise_distribution(response)
    except Exception:
        return None
    return None


def calculate_intent_distribution(state: RouterState) -> RouterState:
    query = state["user_query"]
    keyword_dist = _keyword_distribution(query)
    keyword_hits = sum(query.count(keyword) for words in DEPT_KEYWORDS.values() for keyword in words)

    # Keyword evidence is deterministic and preferred for explicit medical terms.
    # LLM classification is used only for ambiguous queries.
    distribution = keyword_dist if keyword_hits else (_llm_distribution(query) or keyword_dist)
    state["intent_distribution"] = distribution
    return state


def generate_reasoning(state: RouterState) -> RouterState:
    distribution = _normalise_distribution(state["intent_distribution"])
    ranked = sorted(distribution.items(), key=lambda item: item[1], reverse=True)
    department, confidence = ranked[0]
    threshold = get_settings().ROUTER_CONFIDENCE_THRESHOLD
    margin = confidence - ranked[1][1] if len(ranked) > 1 else confidence
    risk = "high" if any(term in state["user_query"] for term in HIGH_RISK_TERMS) else "normal"
    low_confidence = confidence < threshold or margin < 0.15
    human_review = low_confidence or risk == "high"

    reasons = [f"分析结果：主路由为 {department}（置信度 {confidence:.1%}）"]
    if low_confidence:
        reasons.append("置信度或主次科室差值不足，建议人工复核")
    if risk == "high":
        reasons.append("检测到潜在高风险词，优先人工/急诊分流")

    state["intent_distribution"] = distribution
    state["routed_department"] = department if not (risk == "high" and low_confidence) else "全科"
    state["confidence"] = confidence
    state["reasoning"] = "；".join(reasons)
    state["low_confidence"] = low_confidence
    state["human_review_required"] = human_review
    state["risk_level"] = risk
    state["specialist_skills"] = [f"{department}报告解读", "证据引用", "医疗安全校验"]
    return state


def query_knowledge_graph(state: RouterState) -> RouterState:
    """Attach optional graph hints without making routing depend on Neo4j."""
    client = get_neo4j_client()
    symptoms: list[dict[str, Any]] = []
    try:
        # The graph client owns Cypher and can return an empty list when the
        # optional service is unavailable.  The router remains usable.
        for entity in (state["routed_department"], state["user_query"]):
            for item in client.query_by_entity(entity, limit=5):
                symptoms.append(item)
    except Exception:
        symptoms = []
    state["related_symptoms"] = symptoms[:10]
    return state


def create_router_graph():
    graph = StateGraph(RouterState)
    graph.add_node("calculate_intent", calculate_intent_distribution)
    graph.add_node("generate_reasoning", generate_reasoning)
    graph.add_node("query_kg", query_knowledge_graph)
    graph.add_edge("calculate_intent", "generate_reasoning")
    graph.add_edge("generate_reasoning", "query_kg")
    graph.add_edge("query_kg", END)
    graph.set_entry_point("calculate_intent")
    return graph.compile()


_router_graph = None


def get_router_graph():
    global _router_graph
    if _router_graph is None:
        _router_graph = create_router_graph()
    return _router_graph


def route(user_query: str, patient_id: Optional[str] = None) -> Dict[str, Any]:
    initial_state: RouterState = {
        "user_query": user_query,
        "patient_id": patient_id,
        "intent_distribution": {},
        "routed_department": "全科",
        "reasoning": "",
        "confidence": 0.0,
        "related_symptoms": [],
        "low_confidence": False,
        "human_review_required": False,
        "risk_level": "normal",
        "specialist_skills": [],
    }
    result = get_router_graph().invoke(initial_state)
    return {
        key: result.get(key)
        for key in (
            "routed_department",
            "intent_distribution",
            "confidence",
            "reasoning",
            "related_symptoms",
            "low_confidence",
            "human_review_required",
            "risk_level",
            "specialist_skills",
        )
    }
