"""Tests for configuration management."""

import pytest
from app.config import Settings, get_settings


def test_settings_defaults():
    """Test default settings values."""
    settings = Settings()
    assert settings.MYSQL_HOST == "localhost"
    assert settings.MYSQL_PORT == 3306
    assert settings.MILVUS_HOST == "localhost"
    assert settings.MILVUS_PORT == 19530
    assert settings.NEO4J_URI == "bolt://localhost:7687"
    assert settings.VLLM_HOST == "localhost"
    assert settings.VLLM_PORT == 8000
    assert settings.API_PORT == 8080


def test_settings_from_env(monkeypatch):
    """Test settings loaded from environment variables."""
    monkeypatch.setenv("MYSQL_HOST", "test-mysql")
    monkeypatch.setenv("MYSQL_PORT", "3307")
    settings = Settings()
    assert settings.MYSQL_HOST == "test-mysql"
    assert settings.MYSQL_PORT == 3307


def test_get_settings_singleton():
    """Test that get_settings returns singleton."""
    settings1 = get_settings()
    settings2 = get_settings()
    assert settings1 is settings2


def test_mysql_url_property():
    """Test MySQL URL generation."""
    settings = Settings(
        MYSQL_HOST="db.example.com",
        MYSQL_PORT=3306,
        MYSQL_USER="admin",
        MYSQL_PASSWORD="secret",
        MYSQL_DATABASE="testdb"
    )
    expected = "mysql+pymysql://admin:secret@db.example.com:3306/testdb"
    assert settings.mysql_url == expected
