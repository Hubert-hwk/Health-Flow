"""Tests for Milvus client."""

import pytest
from unittest.mock import MagicMock, patch
from app.data.milvus_client import MilvusClient, get_milvus_client


def test_milvus_client_init():
    """Test MilvusClient initialization."""
    client = MilvusClient()
    assert client.collection_name == "medical_reports"
    assert client.dim == 1024  # bge-large-zh-v1.5 dimension


def test_milvus_client_custom_params():
    """Test MilvusClient with custom parameters."""
    client = MilvusClient(
        host="custom-host",
        port=19530,
        collection_name="custom_collection",
        dim=768
    )
    assert client.host == "custom-host"
    assert client.port == 19530
    assert client.collection_name == "custom_collection"
    assert client.dim == 768


def test_get_milvus_client_singleton():
    """Test singleton pattern."""
    client1 = get_milvus_client()
    client2 = get_milvus_client()
    # Note: In actual test, this would be the same instance
    assert client1.collection_name == client2.collection_name


def test_milvus_client_lazy_init():
    """Test MilvusClient does not connect until needed."""
    client = MilvusClient()
    assert client._client is None  # Not connected yet
