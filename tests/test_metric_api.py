"""Tests for Metric API."""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from datetime import datetime


class MockMySQLClient:
    """Mock MySQL client."""

    def __init__(self):
        self.engine = None
        self.SessionLocal = MagicMock()

    def get_session(self):
        mock_session = MagicMock()
        return mock_session


@pytest.fixture
def client():
    """Create test client with mocked dependencies."""
    with patch('app.data.mysql_client.get_mysql_client', return_value=MockMySQLClient()):
        from app.main import app
        yield TestClient(app)


def test_metric_trend_endpoint_no_data(client):
    """Test metric trend with no data."""
    with patch('app.api.metric.get_mysql_client') as mock_mysql:
        mock_client = MagicMock()
        mock_mysql.return_value = mock_client

        # Mock empty result
        mock_session = MagicMock()
        mock_client.get_session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_client.get_session.return_value.__exit__ = MagicMock(return_value=None)

        mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = []
        mock_session.query.return_value.filter.return_value.in_.return_value.all.return_value = []

        response = client.get("/api/health/metric/trend?patient_id=P001&metric_name=空腹血糖")

        assert response.status_code == 200
        data = response.json()
        assert data["patient_id"] == "P001"
        assert data["metric_name"] == "空腹血糖"


def test_search_metrics_endpoint(client):
    """Test searching metrics."""
    with patch('app.api.metric.get_mysql_client') as mock_mysql:
        mock_client = MagicMock()
        mock_mysql.return_value = mock_client

        mock_session = MagicMock()
        mock_client.get_session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_client.get_session.return_value.__exit__ = MagicMock(return_value=None)

        mock_session.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []
        mock_session.query.return_value.filter.return_value.in_.return_value.limit.return_value.all.return_value = []

        response = client.get("/api/health/metric/search?patient_id=P001&keyword=血糖")

        assert response.status_code == 200
        data = response.json()
        assert "metrics" in data


def test_get_anomalies_endpoint(client):
    """Test getting anomalies."""
    with patch('app.api.metric.get_mysql_client') as mock_mysql:
        mock_client = MagicMock()
        mock_mysql.return_value = mock_client

        mock_session = MagicMock()
        mock_client.get_session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_client.get_session.return_value.__exit__ = MagicMock(return_value=None)

        mock_session.query.return_value.filter.return_value.all.return_value = []

        response = client.get("/api/health/metric/anomalies?patient_id=P001")

        assert response.status_code == 200
        data = response.json()
        assert "anomalies" in data
        assert "summary" in data


def test_metric_trend_with_days_parameter(client):
    """Test metric trend with custom days parameter."""
    with patch('app.api.metric.get_mysql_client') as mock_mysql:
        mock_client = MagicMock()
        mock_mysql.return_value = mock_client

        mock_session = MagicMock()
        mock_client.get_session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_client.get_session.return_value.__exit__ = MagicMock(return_value=None)

        mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = []
        mock_session.query.return_value.filter.return_value.in_.return_value.all.return_value = []

        response = client.get("/api/health/metric/trend?patient_id=P001&metric_name=空腹血糖&days=30")

        assert response.status_code == 200


def test_search_metrics_abnormal_only(client):
    """Test searching metrics with abnormal only filter."""
    with patch('app.api.metric.get_mysql_client') as mock_mysql:
        mock_client = MagicMock()
        mock_mysql.return_value = mock_client

        mock_session = MagicMock()
        mock_client.get_session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_client.get_session.return_value.__exit__ = MagicMock(return_value=None)

        mock_session.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []
        mock_session.query.return_value.filter.return_value.in_.return_value.filter.return_value.limit.return_value.all.return_value = []

        response = client.get("/api/health/metric/search?patient_id=P001&abnormal_only=true")

        assert response.status_code == 200
