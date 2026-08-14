"""Tests for Metric API (real in-memory SQLite, seeded data)."""

from contextlib import contextmanager
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.data.models import Base, MedicalReport, MetricRecord


class _FakeDB:
    """Minimal stand-in for MySQLClient backed by an in-memory engine."""

    def __init__(self, engine):
        self.SessionLocal = sessionmaker(bind=engine)

    @contextmanager
    def get_session(self):
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        finally:
            session.close()


@pytest.fixture
def client():
    """Create test client with a real in-memory SQLite database.

    接口是同步 def（在线程池中执行），因此内存库需 StaticPool +
    check_same_thread=False，保证多线程共享同一个连接。
    """
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    fake_db = _FakeDB(engine)

    now = datetime.now()
    with fake_db.get_session() as db:
        report1 = MedicalReport(
            patient_id="P001",
            report_type="体检",
            department="内分泌科",
            created_at=now - timedelta(days=25),
            exam_date=now - timedelta(days=25),
        )
        report2 = MedicalReport(
            patient_id="P001",
            report_type="体检",
            department="内分泌科",
            created_at=now - timedelta(days=5),
            exam_date=now - timedelta(days=5),
        )
        db.add_all([report1, report2])
        db.flush()
        db.add_all(
            [
                MetricRecord(report_id=report1.id, metric_name="空腹血糖", metric_value="6.5",
                             unit="mmol/L", reference_range="3.9-6.1", abnormal_flag="H"),
                MetricRecord(report_id=report1.id, metric_name="血压", metric_value="160/100",
                             unit="mmHg", reference_range="<140/90", abnormal_flag="H"),
                MetricRecord(report_id=report2.id, metric_name="空腹血糖", metric_value="5.2",
                             unit="mmol/L", reference_range="3.9-6.1", abnormal_flag="N"),
            ]
        )
        db.commit()

    with patch('app.api.metric.get_mysql_client', return_value=fake_db):
        from app.main import app
        yield TestClient(app)


def test_metric_trend_with_data(client):
    """Trend for 空腹血糖 should return two ordered data points + statistics."""
    response = client.get("/api/health/metric/trend?patient_id=P001&metric_name=空腹血糖&days=90")

    assert response.status_code == 200
    data = response.json()
    assert data["patient_id"] == "P001"
    assert data["metric_name"] == "空腹血糖"
    assert len(data["data_points"]) == 2
    # 按日期升序：先 6.5（30 天前），后 5.2（5 天前）
    assert data["data_points"][0]["value"] == "6.5"
    assert data["data_points"][1]["value"] == "5.2"
    assert data["statistics"]["count"] == 2
    assert data["statistics"]["average"] == pytest.approx(5.85, abs=0.01)
    assert data["statistics"]["overall_trend"] in ("↑ 上升", "↓ 下降", "→ 稳定", "未知")


def test_metric_trend_no_data(client):
    """Trend for a patient without reports should return empty data points."""
    response = client.get("/api/health/metric/trend?patient_id=P999&metric_name=空腹血糖")

    assert response.status_code == 200
    data = response.json()
    assert data["data_points"] == []
    assert data["summary"] == "暂无数据"


def test_search_metrics_endpoint(client):
    """Searching 血糖 should return the seeded 空腹血糖 rows with exam dates."""
    response = client.get("/api/health/metric/search?patient_id=P001&keyword=血糖")

    assert response.status_code == 200
    data = response.json()
    names = [item["metric_name"] for item in data["metrics"]]
    assert "空腹血糖" in names
    assert all(item["report_id"] is not None for item in data["metrics"])


def test_get_anomalies_endpoint(client):
    """Anomalies should include the flagged 空腹血糖 H and 血压 H rows."""
    response = client.get("/api/health/metric/anomalies?patient_id=P001")

    assert response.status_code == 200
    data = response.json()
    assert data["anomalies"]
    flagged = {(item["metric_name"], item["abnormal_flag"]) for item in data["anomalies"]}
    assert ("空腹血糖", "H") in flagged
    assert ("血压", "H") in flagged
    assert int(data["summary"].split("发现")[1].split("项")[0]) == len(data["anomalies"])


def test_search_metrics_abnormal_only(client):
    """abnormal_only=true should return only H/L flagged metrics."""
    response = client.get("/api/health/metric/search?patient_id=P001&abnormal_only=true")

    assert response.status_code == 200
    data = response.json()
    assert data["metrics"]
    assert all(item["abnormal_flag"] in ("H", "L") for item in data["metrics"])
