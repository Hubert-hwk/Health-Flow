"""Tests for ConsistencyManager."""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime


class MockLLMClient:
    """Mock LLM client for testing."""

    def chat_with_json(self, messages, **kwargs):
        return [
            {"name": "空腹血糖", "type": "metric", "value": "6.5", "unit": "mmol/L"},
            {"name": "多饮", "type": "symptom"}
        ]


@pytest.fixture
def mock_deps():
    """Mock dependencies."""
    with patch('app.agent.consistency_manager.get_llm_client', return_value=MockLLMClient()):
        yield


def test_consistency_manager_init():
    """Test ConsistencyManager initialization."""
    from app.agent.consistency_manager import ConsistencyManager

    manager = ConsistencyManager()
    assert manager.max_context_messages == 10


def test_consistency_manager_custom_max_messages():
    """Test ConsistencyManager with custom max messages."""
    from app.agent.consistency_manager import ConsistencyManager

    manager = ConsistencyManager(max_context_messages=5)
    assert manager.max_context_messages == 5


def test_get_or_create_session():
    """Test session creation."""
    from app.agent.consistency_manager import ConsistencyManager

    manager = ConsistencyManager()
    session = manager.get_or_create_session("sess_123", "P001")

    assert session.session_id == "sess_123"
    assert session.patient_id == "P001"
    assert session.message_count == 0
    assert session.entities == {}


def test_get_or_create_session_existing():
    """Test getting existing session."""
    from app.agent.consistency_manager import ConsistencyManager

    manager = ConsistencyManager()
    session1 = manager.get_or_create_session("sess_123", "P001")
    session2 = manager.get_or_create_session("sess_123", "P001")

    assert session1 is session2


def test_add_message(mock_deps):
    """Test adding a message."""
    from app.agent.consistency_manager import ConsistencyManager

    manager = ConsistencyManager()
    session = manager.add_message(
        session_id="sess_123",
        role="user",
        content="我空腹血糖有点高"
    )

    assert session.message_count == 1


def test_add_message_with_metrics(mock_deps):
    """Test adding message with referenced metrics."""
    from app.agent.consistency_manager import ConsistencyManager

    manager = ConsistencyManager()
    session = manager.add_message(
        session_id="sess_123",
        role="user",
        content="我空腹血糖有点高",
        referenced_metrics=["空腹血糖"]
    )

    assert session.message_count == 1
    assert "空腹血糖" in session.entities


def test_get_context_summary_empty():
    """Test context summary for empty session."""
    from app.agent.consistency_manager import ConsistencyManager

    manager = ConsistencyManager()
    summary = manager.get_context_summary("sess_unknown")

    assert summary == ""


def test_get_context_summary_with_entities(mock_deps):
    """Test context summary with entities."""
    from app.agent.consistency_manager import ConsistencyManager

    manager = ConsistencyManager()
    manager.add_message(
        session_id="sess_123",
        role="user",
        content="我空腹血糖有点高",
        referenced_metrics=["空腹血糖"]
    )

    summary = manager.get_context_summary("sess_123")

    assert "空腹血糖" in summary
    assert "对话轮数" in summary


def test_get_active_entities(mock_deps):
    """Test getting active entities."""
    from app.agent.consistency_manager import ConsistencyManager

    manager = ConsistencyManager()

    # Add messages with different metrics
    manager.add_message(
        session_id="sess_123",
        role="user",
        content="我空腹血糖有点高",
        referenced_metrics=["空腹血糖"]
    )
    manager.add_message(
        session_id="sess_123",
        role="assistant",
        content="您的空腹血糖偏高。"
    )
    manager.add_message(
        session_id="sess_123",
        role="user",
        content="那我现在怎么办"
    )

    active = manager.get_active_entities("sess_123", lookback=3)

    # Should have at least the metric entities
    assert len(active) >= 1


def test_update_department():
    """Test updating department."""
    from app.agent.consistency_manager import ConsistencyManager

    manager = ConsistencyManager()
    manager.get_or_create_session("sess_123", "P001")

    manager.update_department("sess_123", "内分泌科")

    session = manager.get_session("sess_123")
    assert session.current_department == "内分泌科"


def test_clear_session(mock_deps):
    """Test clearing session."""
    from app.agent.consistency_manager import ConsistencyManager

    manager = ConsistencyManager()
    manager.add_message(
        session_id="sess_123",
        role="user",
        content="test"
    )

    assert manager.get_session("sess_123") is not None

    manager.clear_session("sess_123")

    assert manager.get_session("sess_123") is None


def test_medical_entity_model():
    """Test MedicalEntity model."""
    from app.agent.consistency_manager import MedicalEntity

    entity = MedicalEntity(
        name="空腹血糖",
        type="metric",
        value="6.5",
        unit="mmol/L"
    )

    assert entity.name == "空腹血糖"
    assert entity.type == "metric"
    assert entity.value == "6.5"
    assert entity.unit == "mmol/L"


def test_session_context_model():
    """Test SessionContext model."""
    from app.agent.consistency_manager import SessionContext

    session = SessionContext(
        session_id="sess_123",
        patient_id="P001",
        current_department="内分泌科"
    )

    assert session.session_id == "sess_123"
    assert session.patient_id == "P001"
    assert session.current_department == "内分泌科"
    assert session.entities == {}
    assert session.message_count == 0
