"""SQLAlchemy models for MySQL."""

from datetime import datetime
from sqlalchemy import (
    Column,
    Integer,
    BigInteger,
    String,
    Text,
    DateTime,
    ForeignKey,
    JSON,
)
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class MedicalReport(Base):
    """Medical report model."""

    __tablename__ = "medical_reports"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    patient_id = Column(String(64), nullable=False, index=True)
    report_type = Column(String(32))
    file_url = Column(String(512))
    parsed_content = Column(JSON)
    exam_date = Column(DateTime)
    department = Column(String(64))
    created_at = Column(DateTime, default=datetime.now)

    # Relationships
    metrics = relationship("MetricRecord", back_populates="report", cascade="all, delete-orphan")


class MetricRecord(Base):
    """Metric record model."""

    __tablename__ = "metric_records"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    report_id = Column(BigInteger, ForeignKey("medical_reports.id"), nullable=False)
    metric_name = Column(String(128))
    metric_value = Column(String(64))
    unit = Column(String(32))
    reference_range = Column(String(64))
    trend = Column(String(16))
    abnormal_flag = Column(String(8))
    bbox = Column(JSON)

    # Relationships
    report = relationship("MedicalReport", back_populates="metrics")


class ChatSession(Base):
    """Chat session model."""

    __tablename__ = "chat_sessions"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    patient_id = Column(String(64), nullable=False, index=True)
    current_department = Column(String(64))
    agent_type = Column(String(64))
    conversation_summary = Column(Text)
    created_at = Column(DateTime, default=datetime.now)

    # Relationships
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")
    routing_logs = relationship("RoutingLog", back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    """Chat message model."""

    __tablename__ = "chat_messages"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    session_id = Column(BigInteger, ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(String(16))  # user/assistant/system
    content = Column(Text)
    referenced_metrics = Column(JSON)
    safety_check_result = Column(String(16))
    created_at = Column(DateTime, default=datetime.now)

    # Relationships
    session = relationship("ChatSession", back_populates="messages")


class RoutingLog(Base):
    """Routing log model."""

    __tablename__ = "routing_logs"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    session_id = Column(BigInteger, ForeignKey("chat_sessions.id"), nullable=False)
    user_query = Column(Text)
    intent_distribution = Column(JSON)
    routed_department = Column(String(64))
    confidence = Column(String(8))
    created_at = Column(DateTime, default=datetime.now)

    # Relationships
    session = relationship("ChatSession", back_populates="routing_logs")
