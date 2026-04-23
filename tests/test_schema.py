"""Tests for schema validation."""

import pytest
from pydantic import ValidationError
from app.schema.report import MedicalReportCreate, MetricRecord
from app.schema.chat import ChatMessage, ChatRequest, ChatResponse
from app.schema.train import DataAugmentRequest, FinetuneRequest


def test_metric_record_validation():
    """Test MetricRecord validation."""
    metric = MetricRecord(
        metric_name="空腹血糖",
        metric_value="6.5",
        unit="mmol/L",
        reference_range="3.9-6.1"
    )
    assert metric.metric_name == "空腹血糖"
    assert metric.metric_value == "6.5"


def test_metric_record_with_bbox():
    """Test MetricRecord with bbox."""
    metric = MetricRecord(
        metric_name="空腹血糖",
        metric_value="6.5",
        bbox=[120.0, 340.0, 280.0, 360.0]
    )
    assert metric.bbox == [120.0, 340.0, 280.0, 360.0]


def test_chat_request_validation():
    """Test ChatRequest validation."""
    request = ChatRequest(
        session_id="sess_123",
        message="我空腹血糖有点高",
        patient_id="P001",
        include_history=True
    )
    assert request.session_id == "sess_123"
    assert request.message == "我空腹血糖有点高"


def test_chat_request_without_optional():
    """Test ChatRequest without optional fields."""
    request = ChatRequest(
        message="我空腹血糖有点高"
    )
    assert request.session_id is None
    assert request.include_history is True  # default


def test_chat_response_structure():
    """Test ChatResponse has required fields."""
    from app.schema.chat import SafetyCheckResult, FeedbackInfo

    response = ChatResponse(
        reply="这是一条回复",
        department="内分泌科",
        agent_used="EndocrinolAgent",
        safety_check=SafetyCheckResult(passed=True),
        feedback_info=FeedbackInfo()
    )
    assert response.reply == "这是一条回复"
    assert response.department == "内分泌科"


def test_finetune_request_defaults():
    """Test FinetuneRequest default values."""
    request = FinetuneRequest(
        model_name="qwen-vl-plus",
        dataset_path="/data/train.json",
        output_dir="/models/output"
    )
    assert request.method == "qlora"
    assert request.lora_r == 64
    assert request.lora_alpha == 16
    assert request.learning_rate == 2e-4
    assert request.num_epochs == 3
    assert request.batch_size == 4


def test_data_augment_request():
    """Test DataAugmentRequest validation."""
    request = DataAugmentRequest(
        source="pmc",
        target_size=8000,
        categories=["体检报告解读", "指标异常问询"]
    )
    assert request.source == "pmc"
    assert request.target_size == 8000
    assert len(request.categories) == 2
