"""Tests for Chat API."""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


class MockLLMClient:
    """Mock LLM client."""

    def chat(self, messages, **kwargs):
        return "根据您的描述，空腹血糖6.5mmol/L超过正常上限6.1，建议咨询内分泌科医生。"

    def chat_with_json(self, messages, **kwargs):
        return {
            "内分泌科": 0.7,
            "心内科": 0.1,
            "消化科": 0.1,
            "呼吸科": 0.05,
            "全科": 0.05
        }


class MockRAGService:
    """Mock RAG service."""

    def build_context(self, query, department=None):
        return "## 参考信息:\n空腹血糖正常范围3.9-6.1mmol/L"


@pytest.fixture
def client():
    """Create test client with mocked dependencies."""
    with patch('app.model.llm.get_llm_client', return_value=MockLLMClient()), \
         patch('app.service.medical_rag.get_medical_rag_service', return_value=MockRAGService()), \
         patch('app.agent.dynamic_router.get_llm_client', return_value=MockLLMClient()), \
         patch('app.agent.recursive_feedback.get_llm_client', return_value=MockLLMClient()), \
         patch('app.data.get_db') as mock_db:
        # Mock database
        mock_session = MagicMock()
        mock_db.return_value = mock_session
        mock_db.query.return_value.filter.return_value.first.return_value = None

        from app.main import app
        yield TestClient(app)


def test_chat_endpoint_basic(client):
    """Test basic chat endpoint."""
    with patch('app.api.chat.router_route') as mock_route, \
         patch('app.data.get_db') as mock_db:

        mock_route.return_value = {
            "routed_department": "内分泌科",
            "intent_distribution": {"内分泌科": 0.7, "其他": 0.3},
            "confidence": 0.7,
            "reasoning": "根据您的症状"
        }

        mock_session = MagicMock()
        mock_db.return_value = mock_session

        response = client.post("/api/health/chat", json={
            "message": "我空腹血糖有点高"
        })

        assert response.status_code == 200
        data = response.json()
        assert "reply" in data
        assert "department" in data


def test_routing_endpoint(client):
    """Test routing endpoint."""
    with patch('app.api.chat.router_route') as mock_route:
        mock_route.return_value = {
            "routed_department": "内分泌科",
            "intent_distribution": {"内分泌科": 0.8, "其他": 0.2},
            "confidence": 0.8,
            "reasoning": "根据分析"
        }

        response = client.post("/api/health/routing", json={
            "query": "我空腹血糖有点高"
        })

        assert response.status_code == 200
        data = response.json()
        assert "routed_department" in data
        assert "intent_distribution" in data


def test_safety_check_endpoint(client):
    """Test safety check endpoint."""
    response = client.get("/api/health/safety/check?content=您的空腹血糖偏高，建议服用二甲双胍500mg")

    assert response.status_code == 200
    data = response.json()
    assert "passed" in data
    assert "warnings" in data
    assert data["red_flag"] is True  # Contains dosage recommendation


def test_safety_check_pass(client):
    """Test safety check passes for safe content."""
    response = client.get("/api/health/safety/check?content=您的空腹血糖偏高，建议咨询医生")

    assert response.status_code == 200
    data = response.json()
    assert data["passed"] is True


def test_chat_request_validation(client):
    """Test chat request validation."""
    # Empty message should fail with 422 (min_length=1)
    response = client.post("/api/health/chat", json={
        "message": ""
    })
    assert response.status_code == 422


def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_root_endpoint(client):
    """Test root endpoint."""
    response = client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert "HealthFlow" in data["message"]
