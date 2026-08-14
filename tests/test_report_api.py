"""Tests for Report API."""

import io

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from unittest.mock import MagicMock, patch

from app.data.models import Base

class MockVisionService:
    """Mock VisionEncoder service."""

    def parse(self, content, filename):
        from app.service.vision_encoder import ParsedReport
        return ParsedReport(
            report_type="text_pdf",
            raw_text="空腹血糖: 6.5 mmol/L",
            metrics=[],
            page_count=1,
            success=True
        )


class MockEmbeddingClient:
    """Mock Embedding client."""

    def embed(self, text):
        return [0.1] * 1024


class MockMilvusClient:
    """Mock Milvus client."""

    def insert(self, report_ids, texts, embeddings, departments=None):
        return report_ids

    def flush(self):
        pass


@pytest.fixture
def client():
    """Create test client with mocked dependencies."""
    with patch('app.data.mysql_client.get_mysql_client') as mock_mysql, \
         patch('app.service.vision_encoder.get_vision_encoder_service', return_value=MockVisionService()), \
         patch('app.model.embedding.get_embedding_client', return_value=MockEmbeddingClient()), \
         patch('app.data.milvus_client.get_milvus_client', return_value=MockMilvusClient()):

        # Mock MySQL
        mock_client = MagicMock()
        mock_mysql.return_value = mock_client

        from app.main import app
        yield TestClient(app)


def test_upload_report_endpoint(client):
    """Test report upload endpoint against a real in-memory SQLite database."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)

    def override_get_db():
        with SessionLocal() as session:
            yield session

    with patch('app.api.report.get_vision_encoder_service', return_value=MockVisionService()), \
         patch('app.api.report.get_embedding_client', return_value=MockEmbeddingClient()), \
         patch('app.api.report.get_milvus_client', return_value=MockMilvusClient()), \
         patch('app.data.get_db', override_get_db):

        fake_pdf = b'%PDF-1.4 fake pdf content'

        response = client.post(
            "/api/health/report/upload",
            data={"patient_id": "P001", "department": "内分泌科"},
            files={"file": ("test.pdf", io.BytesIO(fake_pdf), "application/pdf")}
        )

        assert response.status_code == 200, response.text
        data = response.json()
        assert data["id"] is not None
        assert data["patient_id"] == "P001"
        assert data["department"] == "内分泌科"
        assert data["report_type"] == "体检"
        assert isinstance(data["metrics"], list)


def test_get_report_endpoint_not_found(client):
    """Test getting non-existent report."""
    with patch('app.data.get_db') as mock_db:
        mock_session = MagicMock()
        mock_db.return_value = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = None

        response = client.get("/api/health/report/999")

        # Should return 404
        assert response.status_code == 404


def test_list_reports_endpoint(client):
    """Test listing reports."""
    with patch('app.data.get_db') as mock_db:
        mock_session = MagicMock()
        mock_db.return_value = mock_session
        mock_session.query.return_value.filter.return_value.order_by.return_value.offset.return_value.limit.return_value.all.return_value = []

        response = client.get("/api/health/reports")

        # Should return list (possibly empty)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


def test_delete_report_not_found(client):
    """Test deleting non-existent report."""
    with patch('app.data.get_db') as mock_db:
        mock_session = MagicMock()
        mock_db.return_value = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = None

        response = client.delete("/api/health/report/999")

        assert response.status_code == 404
