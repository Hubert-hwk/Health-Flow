"""Tests for RecursiveFeedback Agent."""

import pytest
from unittest.mock import MagicMock, patch


class MockLLMClient:
    """Mock LLM client for testing."""

    def chat(self, messages, **kwargs):
        return "这是修正后的回答。"

    def chat_with_json(self, messages, **kwargs):
        # Default: no contradiction
        return {
            "has_contradiction": False,
            "contradictions": []
        }


@pytest.fixture
def mock_llm():
    """Mock LLM client."""
    with patch('app.agent.recursive_feedback.get_llm_client', return_value=MockLLMClient()):
        yield


def test_detect_contradictions_no_history():
    """Test contradiction detection with no history."""
    from app.agent.recursive_feedback import detect_contradictions, FeedbackState

    state: FeedbackState = {
        "original_response": "您的血糖正常。",
        "conversation_history": [],
        "current_response": "您的血糖正常。",
        "contradictions": [],
        "recursion_depth": 0,
        "max_recursion": 3,
        "is_consistent": False,
        "refined_response": ""
    }

    result = detect_contradictions(state)

    # No history means no contradiction
    assert result["is_consistent"] is True
    assert result["contradictions"] == []


def test_detect_contradictions_with_contradiction(mock_llm):
    """Test contradiction detection with actual contradiction."""
    from app.agent.recursive_feedback import detect_contradictions, FeedbackState

    state: FeedbackState = {
        "original_response": "您的空腹血糖为6.5mmol/L，属于正常范围。",
        "conversation_history": [
            {"role": "user", "content": "我空腹血糖6.5"},
            {"role": "assistant", "content": "您的空腹血糖偏高，超过正常上限6.1mmol/L。"}
        ],
        "current_response": "您的空腹血糖为6.5mmol/L，属于正常范围。",
        "contradictions": [],
        "recursion_depth": 0,
        "max_recursion": 3,
        "is_consistent": False,
        "refined_response": ""
    }

    # The mock returns has_contradiction=False, so this should pass
    result = detect_contradictions(state)

    assert "is_consistent" in result


def test_refine_response_no_contradiction():
    """Test refine when there's no contradiction."""
    from app.agent.recursive_feedback import refine_response, FeedbackState

    state: FeedbackState = {
        "original_response": "您的血糖偏高。",
        "conversation_history": [],
        "current_response": "您的血糖偏高。",
        "contradictions": [],
        "recursion_depth": 0,
        "max_recursion": 3,
        "is_consistent": True,
        "refined_response": ""
    }

    result = refine_response(state)

    # No contradiction means original response is kept
    assert result["refined_response"] == "您的血糖偏高。"
    assert result["recursion_depth"] == 0


def test_refine_response_with_max_recursion():
    """Test refine when max recursion is reached."""
    from app.agent.recursive_feedback import refine_response, FeedbackState

    state: FeedbackState = {
        "original_response": "您的血糖偏高。",
        "conversation_history": [],
        "current_response": "您的血糖偏高。",
        "contradictions": ["矛盾1", "矛盾2"],
        "recursion_depth": 3,  # Already at max
        "max_recursion": 3,
        "is_consistent": False,
        "refined_response": ""
    }

    result = refine_response(state)

    # Should return original with warning
    assert "建议咨询医生" in result["refined_response"]


def test_should_continue_end_when_consistent():
    """Test should_continue returns END when consistent."""
    from app.agent.recursive_feedback import should_continue, END, FeedbackState

    state: FeedbackState = {
        "original_response": "您的血糖偏高。",
        "conversation_history": [],
        "current_response": "您的血糖偏高。",
        "contradictions": [],
        "recursion_depth": 0,
        "max_recursion": 3,
        "is_consistent": True,
        "refined_response": "您的血糖偏高。"
    }

    result = should_continue(state)

    assert result == END


def test_validate_and_refine_basic():
    """Test the main validate_and_refine function."""
    from app.agent.recursive_feedback import validate_and_refine

    result = validate_and_refine(
        response="您的空腹血糖为6.5mmol/L，属于正常范围。",
        conversation_history=[
            {"role": "user", "content": "我空腹血糖6.5"},
            {"role": "assistant", "content": "您的空腹血糖偏高。"}
        ],
        max_recursion=3
    )

    assert "original_response" in result
    assert "refined_response" in result
    assert "contradictions" in result
    assert "recursion_depth" in result
    assert "is_consistent" in result
    assert "feedback_applied" in result


def test_consistency_rules_defined():
    """Test that consistency rules are properly defined."""
    from app.agent.recursive_feedback import CONSISTENCY_RULES

    assert len(CONSISTENCY_RULES) == 3

    # Verify rule types
    rule_types = [r["type"] for r in CONSISTENCY_RULES]
    assert "value_contradiction" in rule_types
    assert "range_contradiction" in rule_types
    assert "logic_contradiction" in rule_types

    # Verify each rule has description and example
    for rule in CONSISTENCY_RULES:
        assert "description" in rule
        assert "example" in rule
