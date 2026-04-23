"""Medical graph - LangGraph for multi-agent medical consultation."""

from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END

from app.agent.dynamic_router import route as router_route
from app.agent.recursive_feedback import validate_and_refine


class MedicalGraphState(TypedDict):
    """State for the medical consultation graph."""

    user_query: str
    patient_id: Optional[str]
    session_id: Optional[str]

    # Router outputs
    routed_department: str
    intent_distribution: Dict[str, float]
    reasoning: str
    confidence: float

    # RAG outputs
    retrieved_docs: List[Dict[str, Any]]
    rag_context: str

    # Agent response
    response: str
    refined_response: str

    # Meta
    feedback_applied: bool
    recursion_depth: int
    error: Optional[str]


def create_medical_graph():
    """
    创建完整的医疗问诊图。

    整合动态路由、RAG检索、Agent响应、反馈修正。
    """
    graph = StateGraph(MedicalGraphState)

    # 添加节点
    graph.add_node("route", routing_node)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("generate", generation_node)
    graph.add_node("validate", validation_node)

    # 定义流程
    graph.add_edge("route", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "validate")

    # 设置入口和结束
    graph.set_entry_point("route")

    # 条件边：如果验证发现问题，返回generate重新生成
    graph.add_conditional_edges(
        "validate",
        should_regenerate,
        {
            "generate": "generate",
            END: END
        }
    )

    return graph.compile()


def routing_node(state: MedicalGraphState) -> MedicalGraphState:
    """路由节点 - 确定科室和意图。"""
    try:
        result = router_route(
            user_query=state["user_query"],
            patient_id=state.get("patient_id")
        )

        state["routed_department"] = result["routed_department"]
        state["intent_distribution"] = result["intent_distribution"]
        state["reasoning"] = result["reasoning"]
        state["confidence"] = result["confidence"]
        state["error"] = None

    except Exception as e:
        state["routed_department"] = "全科"
        state["intent_distribution"] = {"全科": 1.0}
        state["reasoning"] = "路由服务暂时不可用"
        state["confidence"] = 0.0
        state["error"] = str(e)

    return state


def retrieval_node(state: MedicalGraphState) -> MedicalGraphState:
    """检索节点 - RAG检索相关文档。"""
    try:
        from app.service.medical_rag import get_medical_rag_service

        rag_service = get_medical_rag_service()
        query = state["user_query"]
        department = state.get("routed_department")

        # 执行混合检索
        results, context = rag_service.retrieve_and_build_context(query, department)

        state["retrieved_docs"] = results
        state["rag_context"] = context

    except Exception as e:
        # RAG服务不可用时返回空
        state["retrieved_docs"] = []
        state["rag_context"] = ""
        state["error"] = str(e)

    return state


def generation_node(state: MedicalGraphState) -> MedicalGraphState:
    """生成节点 - Agent生成响应。"""
    try:
        from app.model.llm import get_llm_client

        llm_client = get_llm_client()
        rag_context = state.get("rag_context", "")
        department = state.get("routed_department", "全科")

        # 构建系统提示
        system_prompt = f"""你是一个专业的医疗辅助助手。请根据提供的信息，为用户提供健康咨询。

科室: {department}

{rag_context if rag_context else '无相关参考信息'}

请遵循以下原则:
1. 仅提供健康建议，不能给出具体用药剂量
2. 不能替代医生进行诊断
3. 危急值需明确警示
4. 提示"仅供参考，请咨询专业医生"
5. 不能基于单一指标做出疾病判断

回答应该专业、清晰、易懂。"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": state["user_query"]}
        ]

        response = llm_client.chat(messages)
        state["response"] = response
        state["refined_response"] = response

    except Exception as e:
        state["response"] = f"抱歉，服务暂时不可用。请稍后再试。"
        state["refined_response"] = state["response"]
        state["error"] = str(e)

    return state


def validation_node(state: MedicalGraphState) -> MedicalGraphState:
    """验证节点 - 反馈验证与修正。"""
    try:
        from app.agent.recursive_feedback import validate_and_refine

        # 获取会话历史（如果有session_id）
        conversation_history = []
        session_id = state.get("session_id")
        if session_id:
            from app.agent.consistency_manager import get_consistency_manager
            cm = get_consistency_manager()
            session = cm.get_session(session_id)
            if session:
                # 从会话中获取历史消息（需要上层传入）
                conversation_history = []

        feedback_result = validate_and_refine(
            response=state["response"],
            conversation_history=conversation_history,
            max_recursion=3
        )

        state["refined_response"] = feedback_result["refined_response"]
        state["feedback_applied"] = feedback_result["feedback_applied"]
        state["recursion_depth"] = feedback_result["recursion_depth"]

        # 如果反馈检测到矛盾，标记状态
        if feedback_result.get("contradictions"):
            state["error"] = f"检测到矛盾: {feedback_result['contradictions']}"

    except Exception as e:
        state["feedback_applied"] = False
        state["recursion_depth"] = 0
        state["error"] = str(e)

    return state


def should_regenerate(state: MedicalGraphState) -> str:
    """
    判断是否需要重新生成。

    如果反馈检测到问题且未超过最大递归深度，返回"generate"重新生成。
    """
    if state.get("recursion_depth", 0) >= 3:
        return END

    if not state.get("feedback_applied", False):
        return END

    # 如果需要重新生成（反馈检测到矛盾），返回generate
    return "generate"


async def run_medical_query(
    user_query: str,
    patient_id: Optional[str] = None,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    运行医疗问诊流程。

    Args:
        user_query: 用户问题
        patient_id: 患者ID
        session_id: 会话ID

    Returns:
        问诊结果
    """
    graph = create_medical_graph()

    initial_state: MedicalGraphState = {
        "user_query": user_query,
        "patient_id": patient_id,
        "session_id": session_id,
        "routed_department": "全科",
        "intent_distribution": {},
        "reasoning": "",
        "confidence": 0.0,
        "retrieved_docs": [],
        "rag_context": "",
        "response": "",
        "refined_response": "",
        "feedback_applied": False,
        "recursion_depth": 0,
        "error": None,
    }

    result = graph.invoke(initial_state)

    return {
        "response": result["refined_response"],
        "department": result["routed_department"],
        "intent_distribution": result["intent_distribution"],
        "reasoning": result["reasoning"],
        "confidence": result["confidence"],
        "feedback_applied": result["feedback_applied"],
        "error": result["error"],
    }
