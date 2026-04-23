"""LangGraph DynamicRouter Agent.

基于意图先验分布的动态路由算法，实现不同专科Agent的精准分发。
"""

from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END

from app.model.llm import get_llm_client
from app.data.neo4j_client import get_neo4j_client


class RouterState(TypedDict):
    """Router agent state."""

    user_query: str
    patient_id: Optional[str]
    intent_distribution: Dict[str, float]
    routed_department: str
    reasoning: str
    confidence: float
    related_symptoms: List[Dict[str, Any]]


# 科室关键词映射
DEPT_KEYWORDS = {
    "内分泌科": ["血糖", "甲状腺", "代谢", "糖尿病", "甲亢", "甲减", "激素", "空腹", "餐后", "糖化血红蛋白"],
    "心内科": ["血压", "心脏", "血脂", "冠心病", "心电图", "心率", "心肌", "心衰", "动脉", "支架"],
    "消化科": ["胃肠", "肝胆", "胃痛", "腹胀", "腹泻", "便秘", "消化", "肝脏", "胆囊", "胰腺"],
    "呼吸科": ["咳嗽", "哮喘", "肺炎", "支气管", "肺", "呼吸道", "胸闷", "气短", "流感", "新冠"],
    "全科": ["体检", "健康", "咨询", "建议", "一般", "普通"]
}


def calculate_intent_distribution(state: RouterState) -> RouterState:
    """
    计算意图分布。

    基于关键词匹配和LLM推理，计算用户query属于各科室的概率分布。
    """
    query = state["user_query"]
    llm_client = get_llm_client()

    # 1. 关键词初步匹配
    keyword_scores = {}
    for dept, keywords in DEPT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in query)
        keyword_scores[dept] = score

    # 2. 如果有关键词匹配，使用关键词得分
    if sum(keyword_scores.values()) > 0:
        total = sum(keyword_scores.values())
        intent_dist = {dept: score / total for dept, score in keyword_scores.items()}
    else:
        # 3. 无关键词匹配时，使用LLM推理
        prompt = f"""根据用户query，判断其最可能属于哪个医疗科室。

用户query: {query}

可选科室: 内分泌科, 心内科, 消化科, 呼吸科, 全科

请以JSON格式输出，格式如下:
{{
    "内分泌科": 0.x,
    "心内科": 0.x,
    "消化科": 0.x,
    "呼吸科": 0.x,
    "全科": 0.x
}}
确保概率之和为1.0。"""

        try:
            response = llm_client.chat_with_json(
                messages=[
                    {"role": "system", "content": "你是一个医疗科室分类助手。"},
                    {"role": "user", "content": prompt}
                ]
            )
            # 解析LLM返回的JSON
            if isinstance(response, dict):
                # 确保所有科室都存在
                for dept in DEPT_KEYWORDS.keys():
                    if dept not in response:
                        response[dept] = 0.0
                intent_dist = response
            else:
                intent_dist = {dept: 0.2 for dept in DEPT_KEYWORDS.keys()}
        except Exception:
            intent_dist = {dept: 0.2 for dept in DEPT_KEYWORDS.keys()}

    # 4. 更新状态
    state["intent_distribution"] = intent_dist
    return state


def generate_reasoning(state: RouterState) -> RouterState:
    """
    生成路由推理过程。

    基于意图分布，生成推理说明并确定最终路由科室。
    """
    intent_dist = state["intent_distribution"]

    # 找出概率最高的科室
    routed_dept = max(intent_dist, key=intent_dist.get)
    confidence = intent_dist[routed_dept]

    # 生成推理说明
    sorted_intents = sorted(intent_dist.items(), key=lambda x: x[1], reverse=True)
    reasoning_parts = [f"根据分析，您的症状最可能涉及以下科室："]
    for dept, prob in sorted_intents:
        if prob > 0.05:
            reasoning_parts.append(f"- {dept}: {prob:.1%}")

    state["routed_department"] = routed_dept
    state["confidence"] = confidence
    state["reasoning"] = "\n".join(reasoning_parts)

    return state


def query_knowledge_graph(state: RouterState) -> RouterState:
    """
    查询知识图谱，获取相关症状信息。

    根据路由科室，查询相关症状和疾病信息。
    """
    neo4j_client = get_neo4j_client()
    routed_dept = state["routed_department"]

    # 根据科室查询相关症状
    try:
        # 查询该科室的常见症状
        query = """
        MATCH (d:Department {name: $dept})<-[:BELONGS_TO]-(s:Symptom)
        RETURN s.name AS symptom, s.description AS description
        LIMIT 10
        """
        with neo4j_client.driver.session(database=neo4j_client.database) as session:
            result = session.run(query, dept=routed_dept)
            symptoms = [{"name": r["symptom"], "description": r["description"]} for r in result]
    except Exception:
        symptoms = []

    state["related_symptoms"] = symptoms
    return state


def create_router_graph() -> StateGraph:
    """
    创建动态路由图。

    流程:
    1. 计算意图分布
    2. 生成推理说明
    3. 查询知识图谱
    4. 返回路由结果
    """
    # 定义节点
    nodes = {
        "calculate_intent": calculate_intent_distribution,
        "generate_reasoning": generate_reasoning,
        "query_kg": query_knowledge_graph,
    }

    # 定义边
    edges = [
        ("calculate_intent", "generate_reasoning"),
        ("generate_reasoning", "query_kg"),
        ("query_kg", END),
    ]

    # 创建图
    graph = StateGraph(RouterState)
    for name, func in nodes.items():
        graph.add_node(name, func)

    # 添加边
    for from_node, to_node in edges:
        graph.add_edge(from_node, to_node)

    # 设置入口点
    graph.set_entry_point("calculate_intent")

    return graph.compile()


# 全局路由器实例
_router_graph = None


def get_router_graph() -> StateGraph:
    """获取路由器图实例。"""
    global _router_graph
    if _router_graph is None:
        _router_graph = create_router_graph()
    return _router_graph


def route(user_query: str, patient_id: Optional[str] = None) -> Dict[str, Any]:
    """
    执行路由。

    Args:
        user_query: 用户query
        patient_id: 患者ID

    Returns:
        路由结果，包含意图分布、路由科室、推理说明等
    """
    graph = get_router_graph()

    initial_state: RouterState = {
        "user_query": user_query,
        "patient_id": patient_id,
        "intent_distribution": {},
        "routed_department": "全科",
        "reasoning": "",
        "confidence": 0.0,
        "related_symptoms": [],
    }

    result = graph.invoke(initial_state)

    return {
        "routed_department": result["routed_department"],
        "intent_distribution": result["intent_distribution"],
        "confidence": result["confidence"],
        "reasoning": result["reasoning"],
        "related_symptoms": result["related_symptoms"],
    }
