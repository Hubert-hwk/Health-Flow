"""Tests for MySQL data access layer."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.data.models import Base, MedicalReport, ChatSession, ChatMessage


@pytest.fixture
def test_db():
    """Create a test database."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()


def test_create_medical_report(test_db):
    """Test creating a medical report."""
    report = MedicalReport(
        patient_id="P001",
        report_type="体检",
        department="内分泌科"
    )
    test_db.add(report)
    test_db.commit()
    test_db.refresh(report)

    assert report.id is not None
    assert report.patient_id == "P001"
    assert report.report_type == "体检"
    assert report.department == "内分泌科"


def test_create_chat_session(test_db):
    """Test creating a chat session."""
    session = ChatSession(
        patient_id="P001",
        current_department="内分泌科"
    )
    test_db.add(session)
    test_db.commit()
    test_db.refresh(session)

    assert session.id is not None
    assert session.patient_id == "P001"
    assert session.conversation_summary is None  # default


def test_chat_session_with_messages(test_db):
    """Test chat session with messages."""
    session = ChatSession(
        patient_id="P001",
        current_department="内分泌科"
    )
    test_db.add(session)
    test_db.commit()

    msg1 = ChatMessage(
        session_id=session.id,
        role="user",
        content="我空腹血糖有点高"
    )
    msg2 = ChatMessage(
        session_id=session.id,
        role="assistant",
        content="您的空腹血糖为6.5mmol/L..."
    )
    test_db.add_all([msg1, msg2])
    test_db.commit()

    # Refresh session to load messages
    test_db.refresh(session)
    assert len(session.messages) == 2
    assert session.messages[0].role == "user"
    assert session.messages[1].role == "assistant"
