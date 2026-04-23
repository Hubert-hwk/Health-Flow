"""Tests for KG API."""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


class MockNeo4jClient:
    """Mock Neo4j client."""

    def __init__(self):
        self.uri = "bolt://localhost:7687"
        self.database = "neo4j"
        self._driver = None

    def connect(self):
        return True

    def query_by_entity(self, entity, limit=10):
        return [
            {"entity": entity, "entity_type": ["Disease"], "relation": "HAS_SYMPTOM", "related_entity": "多饮"}
        ]

    def get_related_symptoms(self, disease):
        return [{"name": "多饮", "description": "喝水量增多"}]

    def get_related_drugs(self, disease):
        return [{"name": "二甲双胍", "description": "降糖药"}]

    def get_related_examinations(self, disease):
        return [{"name": "空腹血糖", "description": "血糖检测"}]

    def get_department(self, symptom):
        return "内分泌科"

    def find_diagnosis_path(self, symptoms):
        return [{"disease": "糖尿病", "description": "慢性代谢性疾病", "matched_symptoms": symptoms, "symptom_count": 2}]


@pytest.fixture
def client():
    """Create test client with mocked dependencies."""
    with patch('app.data.neo4j_client.get_neo4j_client', return_value=MockNeo4jClient()):
        from app.main import app
        yield TestClient(app)


def test_kg_query_endpoint(client):
    """Test KG general query endpoint."""
    response = client.post("/api/health/kg/query?entity=糖尿病&limit=10")

    assert response.status_code == 200
    data = response.json()
    assert "entity" in data
    assert "results" in data


def test_kg_symptoms_endpoint(client):
    """Test KG symptoms query endpoint."""
    response = client.get("/api/health/kg/symptoms/糖尿病")

    assert response.status_code == 200
    data = response.json()
    assert data["disease"] == "糖尿病"
    assert "symptoms" in data


def test_kg_drugs_endpoint(client):
    """Test KG drugs query endpoint."""
    response = client.get("/api/health/kg/drugs/糖尿病")

    assert response.status_code == 200
    data = response.json()
    assert data["disease"] == "糖尿病"
    assert "drugs" in data


def test_kg_examinations_endpoint(client):
    """Test KG examinations query endpoint."""
    response = client.get("/api/health/kg/examinations/糖尿病")

    assert response.status_code == 200
    data = response.json()
    assert data["disease"] == "糖尿病"
    assert "examinations" in data


def test_kg_department_endpoint(client):
    """Test KG department query endpoint."""
    response = client.get("/api/health/kg/department/多饮")

    assert response.status_code == 200
    data = response.json()
    assert data["symptom"] == "多饮"
    assert data["department"] == "内分泌科"


def test_kg_diagnosis_endpoint(client):
    """Test KG diagnosis endpoint."""
    response = client.post("/api/health/kg/diagnosis?symptoms=多饮&symptoms=多尿")

    assert response.status_code == 200
    data = response.json()
    assert "input_symptoms" in data
    assert "possible_diagnoses" in data


def test_kg_diagnosis_requires_min_symptoms(client):
    """Test that diagnosis requires at least 2 symptoms."""
    response = client.post("/api/health/kg/diagnosis?symptoms=多饮")

    # Should fail with 400
    assert response.status_code == 400


def test_kg_health_endpoint(client):
    """Test KG health check endpoint."""
    response = client.get("/api/health/kg/health")

    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["service"] == "neo4j"
