"""Tests for DynamicRouter Agent."""

import pytest
from unittest.mock import MagicMock, patch


class MockLLMClient:
    """Mock LLM client for testing."""

    def chat(self, messages, **kwargs):
        return "mock response"

    def chat_with_json(self, messages, **kwargs):
        return {
            "内分泌科": 0.7,
            "心内科": 0.1,
            "消化科": 0.1,
            "呼吸科": 0.05,
            "全科": 0.05
        }


class MockNeo4jClient:
    """Mock Neo4j client for testing."""

    def __init__(self):
        self.driver = None
        self.database = "neo4j"


@pytest.fixture
def mock_deps():
    """Mock dependencies."""
    with patch('app.agent.dynamic_router.get_llm_client', return_value=MockLLMClient()), \
         patch('app.agent.dynamic_router.get_neo4j_client', return_value=MockNeo4jClient()):
        yield


def test_calculate_intent_distribution_keyword_match(mock_deps):
    """Test intent distribution with keyword matching."""
    from app.agent.dynamic_router import calculate_intent_distribution, RouterState

    state: RouterState = {
        "user_query": "我最近血糖有点高，空腹6.5，而且有点胸闷",
        "patient_id": "P001",
        "intent_distribution": {},
        "routed_department": "",
        "reasoning": "",
        "confidence": 0.0,
        "related_symptoms": []
    }

    result = calculate_intent_distribution(state)

    assert "intent_distribution" in result
    assert result["intent_distribution"]["内分泌科"] > 0


def test_calculate_intent_distribution_no_match(mock_deps):
    """Test intent distribution with no keyword match."""
    from app.agent.dynamic_router import calculate_intent_distribution, RouterState

    state: RouterState = {
        "user_query": "今天天气不错",
        "patient_id": "P001",
        "intent_distribution": {},
        "routed_department": "",
        "reasoning": "",
        "confidence": 0.0,
        "related_symptoms": []
    }

    result = calculate_intent_distribution(state)

    assert "intent_distribution" in result
    # Should fall back to LLM or equal distribution
    assert sum(result["intent_distribution"].values()) == pytest.approx(1.0)


def test_generate_reasoning(mock_deps):
    """Test reasoning generation."""
    from app.agent.dynamic_router import generate_reasoning, RouterState

    state: RouterState = {
        "user_query": "血糖高怎么办",
        "patient_id": "P001",
        "intent_distribution": {"内分泌科": 0.7, "心内科": 0.1, "全科": 0.2},
        "routed_department": "",
        "reasoning": "",
        "confidence": 0.0,
        "related_symptoms": []
    }

    result = generate_reasoning(state)

    assert result["routed_department"] == "内分泌科"
    assert result["confidence"] == 0.7
    assert "推理" in result["reasoning"] or "分析" in result["reasoning"]


def test_query_knowledge_graph(mock_deps):
    """Test knowledge graph query."""
    from app.agent.dynamic_router import query_knowledge_graph, RouterState

    state: RouterState = {
        "user_query": "血糖高",
        "patient_id": "P001",
        "intent_distribution": {},
        "routed_department": "内分泌科",
        "reasoning": "",
        "confidence": 0.7,
        "related_symptoms": []
    }

    # Neo4j returns empty in mock, so should handle gracefully
    result = query_knowledge_graph(state)

    assert "related_symptoms" in result
    assert isinstance(result["related_symptoms"], list)


def test_route_function(mock_deps):
    """Test the main route function."""
    from app.agent.dynamic_router import route

    result = route(
        user_query="我空腹血糖有点高，6.5左右",
        patient_id="P001"
    )

    assert "routed_department" in result
    assert "intent_distribution" in result
    assert "confidence" in result
    assert "reasoning" in result
    assert result["routed_department"] in ["内分泌科", "心内科", "消化科", "呼吸科", "全科"]


def test_route_with_medical_keywords():
    """Test routing with various medical keywords."""
    from app.agent.dynamic_router import DEPT_KEYWORDS

    # Verify department keywords are defined
    assert "内分泌科" in DEPT_KEYWORDS
    assert "心内科" in DEPT_KEYWORDS
    assert "消化科" in DEPT_KEYWORDS
    assert "呼吸科" in DEPT_KEYWORDS
    assert "全科" in DEPT_KEYWORDS

    # Verify keywords are meaningful
    assert "血糖" in DEPT_KEYWORDS["内分泌科"]
    assert "血压" in DEPT_KEYWORDS["心内科"]
    assert "胃肠" in DEPT_KEYWORDS["消化科"]
    assert "咳嗽" in DEPT_KEYWORDS["呼吸科"]
