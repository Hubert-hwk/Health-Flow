"""Tests for Train API."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch


@pytest.fixture
def client():
    """Create test client."""
    from app.main import app
    yield TestClient(app)


def test_trigger_augment_endpoint(client):
    """Test triggering data augmentation."""
    response = client.post("/api/health/train/augment", json={
        "source": "pmc",
        "target_size": 8000,
        "categories": ["体检报告解读"]
    })

    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert data["status"] in ["QUEUED", "STARTING"]


def test_get_augment_status_not_found(client):
    """Test getting status of non-existent task."""
    response = client.get("/api/health/train/augment/nonexistent-task-id")

    assert response.status_code == 404


def test_trigger_finetune_endpoint(client):
    """Test triggering fine-tuning."""
    response = client.post("/api/health/train/finetune", json={
        "model_name": "qwen-vl-plus",
        "dataset_path": "/data/train.json",
        "output_dir": "/models/output",
        "method": "qlora",
        "lora_r": 64,
        "lora_alpha": 16,
        "learning_rate": 0.0002,
        "num_epochs": 3,
        "batch_size": 4
    })

    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data


def test_trigger_dpo_endpoint(client):
    """Test triggering DPO training."""
    response = client.post("/api/health/train/dpo", json={
        "model_name": "qwen-vl-plus",
        "dataset_path": "/data/preference.json",
        "output_dir": "/models/dpo",
        "beta": 0.1,
        "num_epochs": 3,
        "batch_size": 4
    })

    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data


def test_list_tasks_endpoint(client):
    """Test listing all tasks."""
    response = client.get("/api/health/train/tasks")

    assert response.status_code == 200
    data = response.json()
    assert "tasks" in data
    assert "count" in data


def test_cancel_task_endpoint(client):
    """Test cancelling a task."""
    # First create a task
    response = client.post("/api/health/train/augment", json={
        "source": "pmc",
        "target_size": 1000
    })
    task_id = response.json()["task_id"]

    # Cancel it
    response = client.delete(f"/api/health/train/task/{task_id}")

    assert response.status_code == 200
    data = response.json()
    assert "取消" in data["message"] or "CANCELLED" in data["message"]


def test_cancel_nonexistent_task(client):
    """Test cancelling non-existent task."""
    response = client.delete("/api/health/train/task/nonexistent-task-id")

    assert response.status_code == 404


def test_finetune_request_defaults(client):
    """Test finetune request with defaults."""
    response = client.post("/api/health/train/finetune", json={
        "model_name": "qwen-vl-plus",
        "dataset_path": "/data/train.json",
        "output_dir": "/models/output"
    })

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "QUEUED"
