"""Schemas used by chat, routing and safety APIs."""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    role: str = Field(..., description="user/assistant/system")
    content: str
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)
    referenced_metrics: List[str] = Field(default_factory=list)
    safety_check_result: Optional[str] = None


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str = Field(..., min_length=1, max_length=4000)
    patient_id: Optional[str] = None
    include_history: bool = True


class IntentDistribution(BaseModel):
    endocrinology: float = Field(0.0, ge=0, le=1)
    cardiology: float = Field(0.0, ge=0, le=1)
    gastroenterology: float = Field(0.0, ge=0, le=1)
    respiratory: float = Field(0.0, ge=0, le=1)
    general: float = Field(0.0, ge=0, le=1)

    def get_routed_department(self) -> str:
        return max(self.model_dump(), key=self.model_dump().get)


class ReferenceItem(BaseModel):
    type: str = Field(..., description="evidence type: vector, graph or report")
    content: str
    source_id: Optional[str] = None
    score: Optional[float] = None
    path: List[str] = Field(default_factory=list)


class SafetyCheckResult(BaseModel):
    passed: bool
    warnings: List[str] = Field(default_factory=list)
    # 保持纯英文字段名：FastAPI 默认按别名序列化，中文别名会导致
    # 前端拿到「红旗标记」而不是 red_flag，各接口返回不一致。
    red_flag: bool = False
    critical: bool = False


class FeedbackInfo(BaseModel):
    recursion_depth: int = Field(0, ge=0, le=5)
    consistency_check: str = "PASS"
    evidence_score: Optional[float] = Field(None, ge=0, le=1)
    contradictions: List[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    reply: str
    department: str
    agent_used: str
    # 服务端分配的会话 ID（sess_<int>），客户端应保存并在后续轮次回传
    session_id: Optional[str] = None
    intent_distribution: Optional[Dict[str, float]] = None
    referenced_metrics: List[str] = Field(default_factory=list)
    references: List[ReferenceItem] = Field(default_factory=list)
    safety_check: SafetyCheckResult
    feedback_info: FeedbackInfo


class ChatStreamRequest(ChatRequest):
    pass


class RoutingRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    patient_id: Optional[str] = None


class RoutingResponse(BaseModel):
    routed_department: str
    intent_distribution: Dict[str, float]
    confidence: float = Field(..., ge=0, le=1)
    reasoning: str
    low_confidence: bool = False
    human_review_required: bool = False
