"""Chat related schemas."""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Chat message schema."""

    role: str = Field(..., description="角色，user/assistant/system")
    content: str = Field(..., description="消息内容")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)
    referenced_metrics: Optional[List[str]] = Field(default_factory=list)
    safety_check_result: Optional[str] = Field(None)


class ChatRequest(BaseModel):
    """Chat request schema."""

    session_id: Optional[str] = Field(None, description="会话ID")
    message: str = Field(..., description="用户消息")
    patient_id: Optional[str] = Field(None, description="患者ID")
    include_history: bool = Field(True, description="是否包含历史上下文")


class IntentDistribution(BaseModel):
    """Intent distribution for routing."""

    内分泌科: float = Field(0.0, ge=0, le=1)
    心内科: float = Field(0.0, ge=0, le=1)
    消化科: float = Field(0.0, ge=0, le=1)
    呼吸科: float = Field(0.0, ge=0, le=1)
    全科: float = Field(0.0, ge=0, le=1)

    def get_routed_department(self) -> str:
        """Get the department with highest probability."""
        return max(self.model_dump(), key=self.model_dump().get)


class ReferenceItem(BaseModel):
    """Reference item in chat response."""

    type: str = Field(..., description="来源类型，kg/report/literature")
    content: str = Field(..., description="引用内容")


class SafetyCheckResult(BaseModel):
    """Safety check result."""

    passed: bool = Field(..., description="是否通过")
    warnings: List[str] = Field(default_factory=list, description="警告信息")
    红旗标记: bool = Field(False, description="是否触发红旗")


class FeedbackInfo(BaseModel):
    """Feedback loop information."""

    recursion_depth: int = Field(0, ge=0, le=3)
    consistency_check: str = Field("PASS", description="一致性检查结果")


class ChatResponse(BaseModel):
    """Chat response schema."""

    reply: str = Field(..., description="助手回复")
    department: str = Field(..., description="路由科室")
    agent_used: str = Field(..., description="使用的Agent")
    intent_distribution: Optional[Dict[str, float]] = Field(None, description="意图分布")
    referenced_metrics: List[str] = Field(default_factory=list, description="引用的指标")
    references: List[ReferenceItem] = Field(default_factory=list, description="引用来源")
    safety_check: SafetyCheckResult
    feedback_info: FeedbackInfo


class ChatStreamRequest(ChatRequest):
    """Chat streaming request."""
    pass


class RoutingRequest(BaseModel):
    """Routing request schema."""

    query: str = Field(..., description="用户query")
    patient_id: Optional[str] = Field(None, description="患者ID")


class RoutingResponse(BaseModel):
    """Routing response schema."""

    routed_department: str = Field(..., description="路由科室")
    intent_distribution: Dict[str, float]
    confidence: float = Field(..., ge=0, le=1)
    reasoning: str = Field(..., description="路由推理过程")
