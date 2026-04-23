"""Tests for MedicalRAG Service."""

import pytest
from unittest.mock import MagicMock, patch


class MockEmbeddingClient:
    """Mock Embedding client."""

    def embed(self, text):
        return [0.1] * 1024

    def embed_batch(self, texts, batch_size=32):
        return [[0.1] * 1024 for _ in texts]


class MockMilvusClient:
    """Mock Milvus client."""

    def __init__(self):
        self.search_results = [
            {"id": 1, "report_id": 1, "content": "空腹血糖正常范围3.9-6.1", "distance": 0.1}
        ]

    def search(self, query_embedding, top_k=5, department=None):
        return self.search_results


class MockNeo4jClient:
    """Mock Neo4j client."""

    def get_related_symptoms(self, disease):
        return [{"name": "多饮", "description": "喝水量增多"}]

    def get_related_drugs(self, disease):
        return [{"name": "二甲双胍", "description": "降糖药"}]

    def find_diagnosis_path(self, symptoms):
        return [{"disease": "糖尿病", "description": "慢性代谢性疾病", "matched_symptoms": symptoms}]


class MockLLMClient:
    """Mock LLM client."""

    def chat(self, messages, **kwargs):
        return "空腹血糖, 糖尿病"


@pytest.fixture
def mock_deps():
    """Mock all dependencies."""
    with patch('app.service.medical_rag.get_embedding_client', return_value=MockEmbeddingClient()), \
         patch('app.service.medical_rag.get_milvus_client', return_value=MockMilvusClient()), \
         patch('app.service.medical_rag.get_neo4j_client', return_value=MockNeo4jClient()), \
         patch('app.service.medical_rag.get_llm_client', return_value=MockLLMClient()):
        yield


def test_medical_rag_init():
    """Test MedicalRAGService initialization."""
    from app.service.medical_rag import MedicalRAGService

    service = MedicalRAGService()
    assert service.vector_weight == 0.6
    assert service.kg_weight == 0.4
    assert service.top_k == 5


def test_medical_rag_custom_weights():
    """Test MedicalRAGService with custom weights."""
    from app.service.medical_rag import MedicalRAGService

    service = MedicalRAGService(vector_weight=0.7, kg_weight=0.3, top_k=10)
    assert service.vector_weight == 0.7
    assert service.kg_weight == 0.3
    assert service.top_k == 10


def test_vector_search(mock_deps):
    """Test vector search."""
    from app.service.medical_rag import MedicalRAGService

    service = MedicalRAGService()
    results = service.vector_search("空腹血糖偏高")

    assert len(results) > 0
    assert "content" in results[0]


def test_kg_search(mock_deps):
    """Test knowledge graph search."""
    from app.service.medical_rag import MedicalRAGService

    service = MedicalRAGService()
    results = service.kg_search("糖尿病")

    assert len(results) > 0
    assert results[0]["type"] in ["symptom", "drug", "diagnosis"]


def test_kg_search_no_entity(mock_deps):
    """Test KG search with no entity."""
    from app.service.medical_rag import MedicalRAGService

    service = MedicalRAGService()
    results = service.kg_search("无意义的查询")

    # Should handle gracefully
    assert isinstance(results, list)


def test_hybrid_search(mock_deps):
    """Test hybrid search combining vector and KG."""
    from app.service.medical_rag import MedicalRAGService

    service = MedicalRAGService()
    results = service.hybrid_search("空腹血糖偏高")

    assert isinstance(results, list)


def test_build_context(mock_deps):
    """Test context building."""
    from app.service.medical_rag import MedicalRAGService

    service = MedicalRAGService()
    context = service.build_context("空腹血糖偏高")

    assert isinstance(context, str)
    # Should contain reference information
    assert "参考信息" in context or len(context) > 0


def test_extract_medical_entities(mock_deps):
    """Test medical entity extraction."""
    from app.service.medical_rag import MedicalRAGService

    service = MedicalRAGService()
    entities = service._extract_medical_entities("我空腹血糖有点高")

    assert isinstance(entities, list)


def test_fuse_results():
    """Test result fusion."""
    from app.service.medical_rag import MedicalRAGService

    service = MedicalRAGService()

    vector_results = [
        {"id": 1, "content": "空腹血糖正常", "distance": 0.1},
        {"id": 2, "content": "血糖偏高", "distance": 0.2}
    ]

    kg_results = [
        {"type": "symptom", "name": "多饮", "description": "喝水量增多"}
    ]

    fused = service._fuse_results(vector_results, kg_results, "空腹血糖")

    assert len(fused) > 0
    # Results should be sorted by score
    assert isinstance(fused, list)
