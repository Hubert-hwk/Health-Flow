"""ConsistencyManager - 多轮问诊一致性管理器。

负责会话状态跟踪、上下文窗口管理、医疗实体追踪。
"""

from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from app.model.llm import get_llm_client


class MedicalEntity(BaseModel):
    """医疗实体。"""

    name: str = Field(..., description="实体名称")
    type: str = Field(..., description="实体类型：metric/symptom/disease/drug")
    value: Optional[str] = Field(None, description="实体值（如指标值）")
    unit: Optional[str] = Field(None, description="单位")
    first_mentioned_at: int = Field(0, description="首次提到的消息索引")
    last_mentioned_at: int = Field(0, description="最后提到的消息索引")


class SessionContext(BaseModel):
    """会话上下文。"""

    session_id: str
    patient_id: str
    current_department: Optional[str] = None
    entities: Dict[str, MedicalEntity] = Field(default_factory=dict)
    message_count: int = 0
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated_at: datetime = Field(default_factory=datetime.now)


class ConsistencyManager:
    """
    多轮问诊一致性管理器。

    核心职责:
    1. 会话状态跟踪
    2. 医疗实体追踪
    3. 上下文窗口管理
    4. 关键信息提取与持久化
    """

    def __init__(self, max_context_messages: int = 10):
        """
        初始化一致性管理器。

        Args:
            max_context_messages: 保留的最大消息数量
        """
        self.max_context_messages = max_context_messages
        self._sessions: Dict[str, SessionContext] = {}
        self._llm_client = None

    @property
    def llm_client(self):
        """懒加载LLM客户端。"""
        if self._llm_client is None:
            self._llm_client = get_llm_client()
        return self._llm_client

    def get_or_create_session(
        self,
        session_id: str,
        patient_id: str
    ) -> SessionContext:
        """
        获取或创建会话上下文。

        Args:
            session_id: 会话ID
            patient_id: 患者ID

        Returns:
            会话上下文
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionContext(
                session_id=session_id,
                patient_id=patient_id
            )
        return self._sessions[session_id]

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        referenced_metrics: Optional[List[str]] = None
    ) -> SessionContext:
        """
        添加消息到会话。

        Args:
            session_id: 会话ID
            role: 角色 (user/assistant/system)
            content: 消息内容
            referenced_metrics: 引用的指标列表

        Returns:
            更新后的会话上下文
        """
        session = self.get_or_create_session(session_id, session_id.split("_")[0] if "_" in session_id else "unknown")

        session.message_count += 1
        session.last_updated_at = datetime.now()

        # 提取并追踪医疗实体
        self._extract_entities(session, content, referenced_metrics)

        return session

    def _extract_entities(
        self,
        session: SessionContext,
        content: str,
        referenced_metrics: Optional[List[str]] = None
    ) -> None:
        """
        从消息内容中提取医疗实体。

        Args:
            session: 会话上下文
            content: 消息内容
            referenced_metrics: 已知的指标列表
        """
        # 如果有已知的指标，添加到追踪列表
        if referenced_metrics:
            for metric_name in referenced_metrics:
                if metric_name not in session.entities:
                    session.entities[metric_name] = MedicalEntity(
                        name=metric_name,
                        type="metric",
                        first_mentioned_at=session.message_count,
                        last_mentioned_at=session.message_count,
                    )
                else:
                    session.entities[metric_name].last_mentioned_at = session.message_count

        # 使用LLM提取更多实体
        prompt = f"""从以下文本中提取医疗实体（指标、症状、疾病、药品）。

文本:
{content}

以JSON数组格式输出，示例:
[
    {{"name": "空腹血糖", "type": "metric", "value": "6.5", "unit": "mmol/L"}},
    {{"name": "多饮", "type": "symptom"}}
]

只提取明确提到的实体，不要推测。"""

        try:
            response = self.llm_client.chat_with_json(
                messages=[
                    {"role": "system", "content": "你是一个医疗实体提取助手。"},
                    {"role": "user", "content": prompt}
                ]
            )

            if isinstance(response, list):
                for entity_data in response:
                    name = entity_data.get("name")
                    if name and name not in session.entities:
                        session.entities[name] = MedicalEntity(
                            name=name,
                            type=entity_data.get("type", "unknown"),
                            value=entity_data.get("value"),
                            unit=entity_data.get("unit"),
                            first_mentioned_at=session.message_count,
                            last_mentioned_at=session.message_count,
                        )
                    elif name:
                        session.entities[name].last_mentioned_at = session.message_count

        except Exception:
            pass  # LLM提取失败，使用已知指标即可

    def get_context_summary(
        self,
        session_id: str,
        include_history: bool = True
    ) -> str:
        """
        获取上下文摘要。

        Args:
            session_id: 会话ID
            include_history: 是否包含历史消息摘要

        Returns:
            上下文摘要字符串
        """
        if session_id not in self._sessions:
            return ""

        session = self._sessions[session_id]

        # 构建实体追踪摘要
        entity_parts = []
        for entity in session.entities.values():
            if entity.type == "metric":
                value_info = f"{entity.value} {entity.unit}" if entity.value else ""
                entity_parts.append(f"- {entity.name}: {value_info}")
            else:
                entity_parts.append(f"- {entity.name} ({entity.type})")

        entities_summary = "\n".join(entity_parts) if entity_parts else "无"

        summary = f"""## 当前科室: {session.current_department or '未确定'}
## 已追踪实体:
{entities_summary}
## 对话轮数: {session.message_count}"""

        return summary

    def get_active_entities(
        self,
        session_id: str,
        lookback: int = 3
    ) -> List[MedicalEntity]:
        """
        获取最近活跃的实体。

        Args:
            session_id: 会话ID
            lookback: 向前查看的消息数量

        Returns:
            最近活跃的实体列表
        """
        if session_id not in self._sessions:
            return []

        session = self._sessions[session_id]
        current_msg = session.message_count

        active_entities = [
            entity for entity in session.entities.values()
            if current_msg - entity.last_mentioned_at <= lookback
        ]

        return sorted(active_entities, key=lambda e: e.last_mentioned_at, reverse=True)

    def update_department(self, session_id: str, department: str) -> None:
        """
        更新会话科室。

        Args:
            session_id: 会话ID
            department: 科室名称
        """
        if session_id in self._sessions:
            self._sessions[session_id].current_department = department

    def clear_session(self, session_id: str) -> None:
        """
        清除会话上下文。

        Args:
            session_id: 会话ID
        """
        if session_id in self._sessions:
            del self._sessions[session_id]

    def get_session(self, session_id: str) -> Optional[SessionContext]:
        """
        获取会话上下文。

        Args:
            session_id: 会话ID

        Returns:
            会话上下文，如果不存在则返回None
        """
        return self._sessions.get(session_id)


# 全局实例
_consistency_manager: ConsistencyManager | None = None


def get_consistency_manager() -> ConsistencyManager:
    """获取一致性管理器实例。"""
    global _consistency_manager
    if _consistency_manager is None:
        _consistency_manager = ConsistencyManager()
    return _consistency_manager
