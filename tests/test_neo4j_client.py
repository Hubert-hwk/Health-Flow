"""Tests for Neo4j client."""

import pytest
from app.data.neo4j_client import Neo4jClient, get_neo4j_client


def test_neo4j_client_init():
    """Test Neo4jClient initialization."""
    client = Neo4jClient()
    assert client.database == "neo4j"
    assert client.uri == "bolt://localhost:7687"


def test_neo4j_client_custom_params():
    """Test Neo4jClient with custom parameters."""
    client = Neo4jClient(
        uri="bolt://custom:7687",
        user="custom_user",
        password="custom_pass",
        database="test_db"
    )
    assert client.uri == "bolt://custom:7687"
    assert client.user == "custom_user"
    assert client.database == "test_db"


def test_get_neo4j_client_singleton():
    """Test singleton pattern."""
    client1 = get_neo4j_client()
    client2 = get_neo4j_client()
    assert client1.database == client2.database


def test_neo4j_client_lazy_init():
    """Test Neo4jClient does not connect until needed."""
    client = Neo4jClient()
    assert client._driver is None  # Not connected yet
