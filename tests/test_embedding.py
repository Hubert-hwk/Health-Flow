"""Tests for Embedding client."""

import pytest
from app.model.embedding import EmbeddingClient, get_embedding_client


def test_embedding_client_init():
    """Test EmbeddingClient initialization."""
    client = EmbeddingClient()
    assert client.model_name == "BAAI/bge-large-zh-v1.5"
    assert client.device == "cpu"
    assert client.normalize is True
    assert client.dimension == 1024


def test_embedding_client_custom_params():
    """Test EmbeddingClient with custom parameters."""
    client = EmbeddingClient(
        model_name="custom-model",
        device="cuda",
        normalize=False
    )
    assert client.model_name == "custom-model"
    assert client.device == "cuda"
    assert client.normalize is False


def test_get_embedding_client_singleton():
    """Test singleton pattern."""
    client1 = get_embedding_client()
    client2 = get_embedding_client()
    assert client1.model_name == client2.model_name


def test_embed_returns_correct_dimension():
    """Test embed returns correct dimension."""
    client = EmbeddingClient()
    text = "空腹血糖6.5mmol/L，超过正常范围"
    embedding = client.embed(text)
    assert isinstance(embedding, list)
    assert len(embedding) == 1024


def test_embed_batch_returns_correct_shape():
    """Test embed_batch returns correct shape."""
    client = EmbeddingClient()
    texts = ["文本1", "文本2", "文本3"]
    embeddings = client.embed_batch(texts)
    assert isinstance(embeddings, list)
    assert len(embeddings) == 3
    assert all(len(emb) == 1024 for emb in embeddings)


def test_compute_similarity():
    """Test similarity computation."""
    client = EmbeddingClient()
    text1 = "空腹血糖偏高"
    text2 = "血糖指标超标"
    similarity = client.compute_similarity(text1, text2)
    assert isinstance(similarity, float)
    assert 0.0 <= similarity <= 1.0


def test_compute_similarity_identical():
    """Test similarity of identical texts."""
    client = EmbeddingClient()
    text = "空腹血糖6.5mmol/L"
    similarity = client.compute_similarity(text, text)
    assert abs(similarity - 1.0) < 0.01  # Should be very close to 1.0
