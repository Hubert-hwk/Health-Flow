"""Chat API endpoints.

提供智能问诊、流式问诊、意图识别与科室路由等接口。
"""

from typing import Optional, AsyncIterator
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.data import get_db
from app.data.models import ChatSession, ChatMessage, RoutingLog
from app.schema.chat import (
    ChatRequest,
    ChatResponse,
    RoutingRequest,
    RoutingResponse,
    SafetyCheckResult,
    FeedbackInfo,
)
from app.agent.dynamic_router import route as router_route
from app.agent.recursive_feedback import validate_and_refine
from app.agent.consistency_manager import get_consistency_manager
from app.service.medical_rag import get_medical_rag_service


router = APIRouter()


# 医疗安全红线检查
SAFETY_RED_LINES = [
    {
        "rule": "红线1",
        "description": "不能给出具体用药剂量建议",
        "keywords": ["剂量", "每天", "每次", "服用", "口服", "mg", "ml"]
    },
    {
        "rule": "红线2",
        "description": "不能替代医生进行诊断",
        "keywords": ["诊断", "确诊", "你就是", "你就是"]
    },
    {
        "rule": "红线3",
        "description": "危急值需明确警示",
        "keywords": ["危急", "危险", "立即", "紧急"]
    },
    {
        "rule": "红线4",
        "description": "必须提示仅供参考",
        "keywords": []
    },
    {
        "rule": "红线5",
        "description": "不能基于单一指标判断",
        "keywords": ["只能", "仅凭", "单一指标"]
    }
]


def check_safety(content: str) -> SafetyCheckResult:
    """
    检查内容是否触及医疗安全红线。

    Args:
        content: 待检查的回复内容

    Returns:
        安全检查结果
    """
    warnings = []
    has_red_line = False

    for rule in SAFETY_RED_LINES:
        # 规则4：必须提示仅供参考
        if rule["rule"] == "红线4":
            if "仅供参考" not in content and "咨询医生" not in content:
                warnings.append("建议添加'仅供参考，请咨询专业医生'提示")
            continue

        # 其他规则：检查关键词
        for keyword in rule["keywords"]:
            if keyword in content:
                warnings.append(f"触及{rule['description']}")
                has_red_line = True
                break

    return SafetyCheckResult(
        passed=not has_red_line,
        warnings=warnings[:3],  # 最多返回3条警告
        红旗标记=has_red_line
    )


def build_rag_context(user_query: str, department: str) -> str:
    """
    构建RAG上下文。

    Args:
        user_query: 用户问题
        department: 科室

    Returns:
        上下文字符串
    """
    try:
        rag_service = get_medical_rag_service()
        return rag_service.build_context(user_query, department)
    except Exception:
        return ""


async def generate_response(
    user_query: str,
    patient_id: Optional[str],
    session_id: Optional[str],
    department: str,
    conversation_history: list
) -> tuple[str, bool, int]:
    """
    生成回复。

    Args:
        user_query: 用户问题
        patient_id: 患者ID
        session_id: 会话ID
        department: 科室
        conversation_history: 历史对话

    Returns:
        (回复内容, 是否应用了反馈, 递归深度)
    """
    # 构建RAG上下文
    rag_context = build_rag_context(user_query, department)

    # 构建系统提示
    system_prompt = f"""你是一个专业的医疗辅助助手。请根据提供的信息，为用户提供健康咨询。

科室: {department}

{rag_context}

请遵循以下原则:
1. 仅提供健康建议，不能给出具体用药剂量
2. 不能替代医生进行诊断
3. 危急值需明确警示
4. 提示"仅供参考，请咨询专业医生"
5. 不能基于单一指标做出疾病判断

回答应该专业、清晰、易懂。"""

    # 构建消息
    messages = [{"role": "system", "content": system_prompt}]
    for msg in conversation_history[-5:]:
        messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
    messages.append({"role": "user", "content": user_query})

    # 调用LLM
    from app.model.llm import get_llm_client
    llm_client = get_llm_client()

    try:
        response = llm_client.chat(messages)
    except Exception as e:
        response = f"抱歉，服务暂时不可用。请稍后再试。错误: str(e)"

    # 应用反馈验证
    feedback_result = validate_and_refine(
        response=response,
        conversation_history=conversation_history,
        max_recursion=3
    )

    return (
        feedback_result["refined_response"],
        feedback_result["feedback_applied"],
        feedback_result["recursion_depth"]
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: Session = Depends(get_db)
):
    """
    智能问诊接口。

    Args:
        request: 问诊请求
        db: 数据库会话

    Returns:
        问诊响应
    """
    # 1. 路由决策
    routing_result = router_route(
        user_query=request.message,
        patient_id=request.patient_id
    )

    routed_department = routing_result["routed_department"]
    intent_distribution = routing_result["intent_distribution"]
    confidence = routing_result["confidence"]
    reasoning = routing_result["reasoning"]

    # 2. 获取或创建会话
    if request.session_id:
        session = db.query(ChatSession).filter(
            ChatSession.id == int(request.session_id.replace("sess_", ""))
        ).first()
    else:
        session = ChatSession(
            patient_id=request.patient_id or "anonymous",
            current_department=routed_department
        )
        db.add(session)
        db.commit()
        db.refresh(session)

    # 3. 获取对话历史
    messages = db.query(ChatMessage).filter(
        ChatMessage.session_id == session.id
    ).order_by(ChatMessage.created_at).all()

    conversation_history = [
        {"role": msg.role, "content": msg.content}
        for msg in messages
    ]

    # 4. 生成回复
    response_content, feedback_applied, recursion_depth = await generate_response(
        user_query=request.message,
        patient_id=request.patient_id,
        session_id=request.session_id,
        department=routed_department,
        conversation_history=conversation_history
    )

    # 5. 安全检查
    safety_result = check_safety(response_content)

    # 6. 保存用户消息
    user_msg = ChatMessage(
        session_id=session.id,
        role="user",
        content=request.message
    )
    db.add(user_msg)

    # 7. 保存助手回复
    assistant_msg = ChatMessage(
        session_id=session.id,
        role="assistant",
        content=response_content,
        safety_check_result="PASS" if safety_result.passed else "WARNING"
    )
    db.add(assistant_msg)

    # 8. 保存路由日志
    routing_log = RoutingLog(
        session_id=session.id,
        user_query=request.message,
        intent_distribution=intent_distribution,
        routed_department=routed_department,
        confidence=str(confidence)
    )
    db.add(routing_log)

    db.commit()

    # 9. 返回响应
    return ChatResponse(
        reply=response_content,
        department=routed_department,
        agent_used=f"{routed_department}Agent",
        intent_distribution=intent_distribution,
        referenced_metrics=[],
        references=[],
        safety_check=safety_result,
        feedback_info=FeedbackInfo(
            recursion_depth=recursion_depth,
            consistency_check="PASS" if not feedback_applied else "REFINED"
        )
    )


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    流式问诊接口（SSE）。

    Args:
        request: 问诊请求

    Returns:
        SSE流式响应
    """
    async def event_generator() -> AsyncIterator[str]:
        # 1. 路由决策
        routing_result = router_route(
            user_query=request.message,
            patient_id=request.patient_id
        )

        routed_department = routing_result["routed_department"]

        # 发送路由信息
        yield f"data: {routed_department}\n\n"

        # 2. 获取RAG上下文
        rag_context = build_rag_context(request.message, routed_department)

        # 3. 构建提示
        system_prompt = f"""你是一个专业的医疗辅助助手。请根据提供的信息，为用户提供健康咨询。

科室: {routed_department}

{rag_context}

请遵循以下原则:
1. 仅提供健康建议，不能给出具体用药剂量
2. 不能替代医生进行诊断
3. 危急值需明确警示
4. 提示"仅供参考，请咨询专业医生"

回答应该专业、清晰、易懂。直接回答，不要说"好的"之类的废话。"""

        messages = [{"role": "system", "content": system_prompt}]
        messages.append({"role": "user", "content": request.message})

        # 4. 流式调用LLM
        from app.model.llm import get_llm_client
        llm_client = get_llm_client()

        try:
            response = llm_client.chat(messages)
            # 模拟流式输出
            for chunk in response.split():
                yield f"data: {chunk}\n\n"
        except Exception as e:
            yield f"data: 抱歉，服务暂时不可用。\n\n"

        # 发送完成标记
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.post("/routing", response_model=RoutingResponse)
async def routing(request: RoutingRequest):
    """
    意图识别与科室路由接口。

    Args:
        request: 路由请求

    Returns:
        路由结果
    """
    result = router_route(
        user_query=request.query,
        patient_id=request.patient_id
    )

    return RoutingResponse(
        routed_department=result["routed_department"],
        intent_distribution=result["intent_distribution"],
        confidence=result["confidence"],
        reasoning=result["reasoning"]
    )


@router.get("/safety/check")
async def safety_check(content: str):
    """
    安全红线检查接口。

    Args:
        content: 待检查的内容

    Returns:
        检查结果
    """
    result = check_safety(content)
    return {
        "passed": result.passed,
        "warnings": result.warnings,
        "红旗标记": result.红旗标记
    }
