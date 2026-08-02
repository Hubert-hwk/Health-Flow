"""Conversation state and medical-entity tracking for Self-Correction."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.model.llm import get_llm_client


class MedicalEntity(BaseModel):
    name: str
    type: str
    value: Optional[str] = None
    unit: Optional[str] = None
    first_mentioned_at: int = 0
    last_mentioned_at: int = 0


class SessionContext(BaseModel):
    session_id: str
    patient_id: str
    current_department: Optional[str] = None
    entities: Dict[str, MedicalEntity] = Field(default_factory=dict)
    messages: List[Dict[str, str]] = Field(default_factory=list)
    message_count: int = 0
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated_at: datetime = Field(default_factory=datetime.now)


class ConsistencyManager:
    def __init__(self, max_context_messages: int = 10) -> None:
        self.max_context_messages = max_context_messages
        self._sessions: Dict[str, SessionContext] = {}
        self._llm_client = None

    @property
    def llm_client(self):
        if self._llm_client is None:
            self._llm_client = get_llm_client()
        return self._llm_client

    def get_or_create_session(self, session_id: str, patient_id: str = "unknown") -> SessionContext:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionContext(session_id=session_id, patient_id=patient_id)
        elif patient_id != "unknown":
            self._sessions[session_id].patient_id = patient_id
        return self._sessions[session_id]

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        referenced_metrics: Optional[List[str]] = None,
        patient_id: Optional[str] = None,
    ) -> SessionContext:
        session = self.get_or_create_session(session_id, patient_id or "unknown")
        session.message_count += 1
        session.last_updated_at = datetime.now()
        session.messages.append({"role": role, "content": content})
        session.messages = session.messages[-self.max_context_messages :]
        self._add_explicit_entities(session, referenced_metrics or [])
        self._extract_entities(session, content)
        return session

    def _add_explicit_entities(self, session: SessionContext, names: List[str]) -> None:
        for name in names:
            if not name:
                continue
            entity = session.entities.get(name)
            if entity:
                entity.last_mentioned_at = session.message_count
            else:
                session.entities[name] = MedicalEntity(
                    name=name,
                    type="metric",
                    first_mentioned_at=session.message_count,
                    last_mentioned_at=session.message_count,
                )

    def _extract_entities(self, session: SessionContext, content: str) -> None:
        if not content.strip():
            return
        try:
            result = self.llm_client.chat_with_json(
                messages=[
                    {"role": "system", "content": "提取明确出现的医疗指标、症状或疾病，只输出 JSON 数组。"},
                    {
                        "role": "user",
                        "content": f"文本：{content}\n格式：[{{\"name\":\"血糖\",\"type\":\"metric\"}}]",
                    },
                ],
                json_schema={"type": "array"},
            )
            values = result if isinstance(result, list) else result.get("entities", []) if isinstance(result, dict) else []
        except Exception:
            values = []
        for value in values:
            if not isinstance(value, dict) or not value.get("name"):
                continue
            name = str(value["name"])
            entity = session.entities.get(name)
            if entity:
                entity.last_mentioned_at = session.message_count
                entity.value = value.get("value") or entity.value
                entity.unit = value.get("unit") or entity.unit
            else:
                session.entities[name] = MedicalEntity(
                    name=name,
                    type=str(value.get("type", "unknown")),
                    value=value.get("value"),
                    unit=value.get("unit"),
                    first_mentioned_at=session.message_count,
                    last_mentioned_at=session.message_count,
                )

    def get_history(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
        session = self._sessions.get(session_id)
        if not session:
            return []
        return session.messages[-(limit or self.max_context_messages) :]

    def get_context_summary(self, session_id: str, include_history: bool = True) -> str:
        session = self._sessions.get(session_id)
        if not session:
            return ""
        entities = []
        for entity in session.entities.values():
            value = f"：{entity.value} {entity.unit or ''}" if entity.value else ""
            entities.append(f"- {entity.name}（{entity.type}）{value}")
        lines = [
            f"当前科室：{session.current_department or '未确定'}",
            "已追踪实体：\n" + ("\n".join(entities) if entities else "无"),
            f"对话轮数：{session.message_count}",
        ]
        if include_history:
            lines.append("最近对话：\n" + "\n".join(
                f"{item['role']}: {item['content']}" for item in self.get_history(session_id)
            ))
        return "\n".join(lines)

    def get_active_entities(self, session_id: str, lookback: int = 3) -> List[MedicalEntity]:
        session = self._sessions.get(session_id)
        if not session:
            return []
        current = session.message_count
        return sorted(
            [item for item in session.entities.values() if current - item.last_mentioned_at <= lookback],
            key=lambda item: item.last_mentioned_at,
            reverse=True,
        )

    def update_department(self, session_id: str, department: str) -> None:
        session = self._sessions.get(session_id)
        if session:
            session.current_department = department

    def clear_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def get_session(self, session_id: str) -> Optional[SessionContext]:
        return self._sessions.get(session_id)


_consistency_manager: ConsistencyManager | None = None


def get_consistency_manager() -> ConsistencyManager:
    global _consistency_manager
    if _consistency_manager is None:
        _consistency_manager = ConsistencyManager()
    return _consistency_manager
