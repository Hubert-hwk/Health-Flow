"""Tests for LLM client."""

import pytest
from app.model.llm import LLMClient, vLLMClient, VLMClient, get_llm_client


def test_llm_client_init():
    """Test LLMClient initialization."""
    client = LLMClient()
    assert client.model == "qwen-vl-plus"


def test_vllm_client_init():
    """Test vLLMClient initialization."""
    client = vLLMClient()
    assert client.model == "qwen-vl-plus"
    assert "localhost:8000" in client.api_base
    assert client.temperature == 0.7
    assert client.max_tokens == 2048


def test_vllm_client_custom_params():
    """Test vLLMClient with custom parameters."""
    client = vLLMClient(
        model="custom-model",
        api_base="http://custom:8000/v1",
        temperature=0.5,
        max_tokens=1024
    )
    assert client.model == "custom-model"
    assert client.api_base == "http://custom:8000/v1"
    assert client.temperature == 0.5
    assert client.max_tokens == 1024


def test_get_llm_client_returns_vllm_client():
    """Test get_llm_client returns vLLMClient instance."""
    client = get_llm_client()
    assert isinstance(client, vLLMClient)


def test_vlm_client_inheritance():
    """Test VLMClient inherits from vLLMClient."""
    client = VLMClient()
    assert isinstance(client, vLLMClient)
    assert client.model == "qwen-vl-plus"
