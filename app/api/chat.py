"""Chat, routing and safety endpoints."""

from __future__ import annotations

import json
from types import GeneratorType
from typing import AsyncIterator, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.agent.dynamic_router import route as router_route
from app.agent.graph.medical_graph import run_medical_query
from app.config import get_settings
from app.data import get_db
from app.data.models import ChatMessage, ChatSession, RoutingLog
from app.schema.chat import (
    ChatRequest,
    ChatResponse,
    FeedbackInfo,
    ReferenceItem,
    RoutingRequest,
    RoutingResponse,
    SafetyCheckResult,
)
from app.service.medical_rag import get_medical_rag_service
from app.service.safety_guard import check_response, enforce_boundary


router = APIRouter()


def db_dependency():
    """Stable FastAPI dependency wrapper that also keeps test overrides simple."""
    from app.data import get_db as current_get_db

    value = current_get_db()
    if isinstance(value, GeneratorType):
        yield from value
    else:
        yield value


def check_safety(content: str) -> SafetyCheckResult:
    """Backward-compatible public wrapper used by the safety endpoint."""
    return check_response(content)


def build_rag_context(user_query: str, department: str) -> str:
    try:
        return get_medical_rag_service().build_context(user_query, department)
    except Exception:
        return ""


async def generate_response(
    user_query: str,
    patient_id: Optional[str],
    session_id: Optional[str],
    department: str,
    conversation_history: list,
) -> tuple[str, bool, int]:
    """Compatibility helper; the graph is the single generation path."""
    result = await run_medical_query(
        user_query,
        patient_id=patient_id,
        session_id=session_id,
        conversation_history=conversation_history,
    )
    return result["response"], result["feedback_applied"], result["recursion_depth"]


def _session_from_request(db: Session, request: ChatRequest, department: str) -> ChatSession:
    if request.session_id:
        raw_id = request.session_id.removeprefix("sess_")
        try:
            session_id = int(raw_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="session_id 格式错误") from exc
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="会话不存在")
        session.current_department = department
        return session

    session = ChatSession(
        patient_id=request.patient_id or "anonymous",
        current_department=department,
    )
    db.add(session)
    db.flush()
    return session


def _history(db: Session, session_id: int, include_history: bool) -> list[dict[str, str]]:
    if not include_history:
        return []
    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at)
        .all()
    )
    return [{"role": item.role, "content": item.content} for item in messages]


def _references(results: list[dict]) -> list[ReferenceItem]:
    values: list[ReferenceItem] = []
    for item in results:
        values.append(
            ReferenceItem(
                type=str(item.get("source", "unknown")),
                source_id=str(item.get("source_id", "")) or None,
                content=str(item.get("content") or item.get("description") or item.get("name") or "")[:800],
                score=float(item["score"]) if item.get("score") is not None else None,
                path=list(item.get("path") or []),
            )
        )
    return values


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, db: Session = Depends(db_dependency)):
    routing = router_route(request.message, request.patient_id)
    department = routing["routed_department"]
    session = _session_from_request(db, request, department)
    history = _history(db, session.id, request.include_history)

    result = await run_medical_query(
        request.message,
        patient_id=request.patient_id,
        session_id=f"sess_{session.id}",
        conversation_history=history,
    )
    safe_reply, safety = enforce_boundary(result["response"])

    db.add(ChatMessage(session_id=session.id, role="user", content=request.message))
    db.add(
        ChatMessage(
            session_id=session.id,
            role="assistant",
            content=safe_reply,
            safety_check_result="PASS" if safety.passed else "BLOCKED",
        )
    )
    db.add(
        RoutingLog(
            session_id=session.id,
            user_query=request.message,
            intent_distribution=routing["intent_distribution"],
            routed_department=department,
            confidence=str(routing["confidence"]),
        )
    )
    db.commit()

    return ChatResponse(
        reply=safe_reply,
        department=result["department"],
        agent_used=result["agent_used"],
        intent_distribution=result["intent_distribution"],
        references=_references(result.get("retrieved_docs", [])),
        safety_check=safety,
        feedback_info=FeedbackInfo(
            recursion_depth=result["recursion_depth"],
            consistency_check="PASS" if not result["contradictions"] else "REVIEW",
            evidence_score=result.get("evidence_score"),
            contradictions=result.get("contradictions", []),
        ),
    )


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    async def event_generator() -> AsyncIterator[str]:
        routing = router_route(request.message, request.patient_id)
        yield f"data: {json.dumps({'type': 'route', **routing}, ensure_ascii=False)}\n\n"
        result = await run_medical_query(
            request.message,
            patient_id=request.patient_id,
            session_id=request.session_id,
            conversation_history=[],
        )
        safe_reply, safety = enforce_boundary(result["response"])
        for chunk in (safe_reply[index : index + 80] for index in range(0, len(safe_reply), 80)):
            yield f"data: {json.dumps({'type': 'delta', 'content': chunk}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'safety_check': safety.model_dump(by_alias=True)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@router.post("/routing", response_model=RoutingResponse)
async def routing(request: RoutingRequest):
    result = router_route(request.query, request.patient_id)
    return RoutingResponse(
        routed_department=result["routed_department"],
        intent_distribution=result["intent_distribution"],
        confidence=result["confidence"],
        reasoning=result["reasoning"],
        low_confidence=result.get("low_confidence", False),
        human_review_required=result.get("human_review_required", False),
    )


@router.get("/safety/check")
async def safety_check(content: str):
    result = check_safety(content)
    return {
        "passed": result.passed,
        "warnings": result.warnings,
        "red_flag": result.red_flag,
        "红旗标记": result.red_flag,
        "critical": result.critical,
    }
