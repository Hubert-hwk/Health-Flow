"""LangGraph RecursiveFeedback Agent.

递归式反馈机制，对多轮问诊进行逻辑一致性校验，提升长上下文推演的输出稳定性。
"""

from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END

from app.model.llm import get_llm_client


class FeedbackState(TypedDict):
    """Feedback agent state."""

    original_response: str
    conversation_history: List[Dict[str, str]]
    current_response: str
    contradictions: List[str]
    recursion_depth: int
    max_recursion: int
    is_consistent: bool
    refined_response: str


# 医疗一致性规则
CONSISTENCY_RULES = [
    {
        "type": "value_contradiction",
        "description": "同一指标前后描述不一致",
        "example": "前轮说血糖正常，后轮说血糖偏高"
    },
    {
        "type": "range_contradiction",
        "description": "指标值超出合理区间",
        "example": "血糖值为-5.0"
    },
    {
        "type": "logic_contradiction",
        "description": "因果关系颠倒或冲突",
        "example": "先说需要用药，后说无需治疗"
    }
]


def detect_contradictions(state: FeedbackState) -> FeedbackState:
    """
    检测当前回答与历史上下文的矛盾。

    使用LLM分析是否存在逻辑矛盾。
    """
    llm_client = get_llm_client()
    current_response = state["current_response"]
    history = state["conversation_history"]

    if not history:
        state["contradictions"] = []
        state["is_consistent"] = True
        return state

    # 构建历史对话摘要
    history_summary = "\n".join([
        f"{msg['role']}: {msg['content']}"
        for msg in history[-5:]  # 只看最近5轮
    ])

    # LLM判断是否存在矛盾
    prompt = f"""请分析当前回答与历史对话是否存在矛盾。

历史对话:
{history_summary}

当前回答:
{current_response}

请以JSON格式输出:
{{
    "has_contradiction": true/false,
    "contradictions": [
        "矛盾1的具体描述",
        "矛盾2的具体描述"
    ],
    "reasoning": "判断理由"
}}
"""

    try:
        response = llm_client.chat_with_json(
            messages=[
                {"role": "system", "content": "你是一个医疗一致性分析助手。"},
                {"role": "user", "content": prompt}
            ]
        )

        if isinstance(response, dict):
            state["is_consistent"] = not response.get("has_contradiction", False)
            state["contradictions"] = response.get("contradictions", [])
        else:
            state["is_consistent"] = True
            state["contradictions"] = []

    except Exception:
        state["is_consistent"] = True
        state["contradictions"] = []

    return state


def refine_response(state: FeedbackState) -> FeedbackState:
    """
    当检测到矛盾时，触发修正流程。

    重新检索信息，生成修正后的回答。
    """
    llm_client = get_llm_client()
    current_response = state["current_response"]
    contradictions = state["contradictions"]
    history = state["conversation_history"]

    if state["is_consistent"] or not contradictions:
        state["refined_response"] = current_response
        return state

    # 如果已达最大递归深度，返回原回答但标注问题
    if state["recursion_depth"] >= state["max_recursion"]:
        state["refined_response"] = current_response + "\n\n[注意：检测到潜在逻辑问题，建议咨询医生确认]"
        return state

    # 构建修正提示
    contradiction_desc = "\n".join([f"- {c}" for c in contradictions])

    history_summary = "\n".join([
        f"{msg['role']}: {msg['content']}"
        for msg in history[-5:]
    ])

    prompt = f"""请修正以下回答中的逻辑矛盾问题。

原始回答:
{current_response}

检测到的矛盾:
{contradiction_desc}

历史对话:
{history_summary}

请生成修正后的回答，确保:
1. 解决上述矛盾
2. 保持医学逻辑正确
3. 回答完整且一致

直接输出修正后的回答，不需要解释。"""

    try:
        refined = llm_client.chat(
            messages=[
                {"role": "system", "content": "你是一个医疗助手。"},
                {"role": "user", "content": prompt}
            ]
        )
        state["refined_response"] = refined
    except Exception:
        state["refined_response"] = current_response

    state["recursion_depth"] += 1
    return state


def should_continue(state: FeedbackState) -> str:
    """
    判断是否需要继续递归修正。

    如果修正后仍存在矛盾，且未超过最大递归深度，继续修正。
    """
    if state["is_consistent"]:
        return END

    if state["recursion_depth"] >= state["max_recursion"]:
        return END

    # 重新检测矛盾
    llm_client = get_llm_client()
    refined = state["refined_response"]
    history = state["conversation_history"]

    history_summary = "\n".join([
        f"{msg['role']}: {msg['content']}"
        for msg in history[-5:]
    ])

    prompt = f"""分析修正后的回答与历史对话是否一致。

历史对话:
{history_summary}

修正后回答:
{refined}

输出JSON:
{{
    "has_contradiction": true/false
}}
"""

    try:
        response = llm_client.chat_with_json(
            messages=[
                {"role": "system", "content": "你是一个医疗一致性分析助手。"},
                {"role": "user", "content": prompt}
            ]
        )

        if isinstance(response, dict) and response.get("has_contradiction", False):
            return "refine"
        else:
            return END
    except Exception:
        return END


def create_feedback_graph() -> StateGraph:
    """
    创建递归反馈图。

    流程:
    1. 检测矛盾
    2. 如果有矛盾，修正回答
    3. 重新检测
    4. 最多递归max_recursion次
    """
    graph = StateGraph(FeedbackState)

    # 添加节点
    graph.add_node("detect", detect_contradictions)
    graph.add_node("refine", refine_response)

    # 添加边
    graph.add_conditional_edges(
        "detect",
        should_continue,
        {
            END: END,
            "refine": "refine"
        }
    )
    graph.add_edge("refine", "detect")

    # 设置入口点
    graph.set_entry_point("detect")

    return graph.compile()


# 全局反馈图实例
_feedback_graph = None


def get_feedback_graph() -> StateGraph:
    """获取反馈图实例。"""
    global _feedback_graph
    if _feedback_graph is None:
        _feedback_graph = create_feedback_graph()
    return _feedback_graph


def validate_and_refine(
    response: str,
    conversation_history: List[Dict[str, str]],
    max_recursion: int = 3
) -> Dict[str, Any]:
    """
    验证并修正回答。

    Args:
        response: 原始回答
        conversation_history: 对话历史
        max_recursion: 最大递归深度

    Returns:
        验证结果，包含修正后的回答、矛盾列表、递归深度等
    """
    graph = get_feedback_graph()

    initial_state: FeedbackState = {
        "original_response": response,
        "conversation_history": conversation_history,
        "current_response": response,
        "contradictions": [],
        "recursion_depth": 0,
        "max_recursion": max_recursion,
        "is_consistent": False,
        "refined_response": response,
    }

    result = graph.invoke(initial_state)

    return {
        "original_response": result["original_response"],
        "refined_response": result["refined_response"],
        "contradictions": result["contradictions"],
        "recursion_depth": result["recursion_depth"],
        "is_consistent": result["is_consistent"],
        "feedback_applied": result["recursion_depth"] > 0,
    }
