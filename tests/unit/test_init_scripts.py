"""Tests for database initialization scripts (Milvus, Neo4j)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# Milvus/Neo4j SDK 未安装时跳过本模块（这些测试通过 patch 字符串目标依赖 SDK 可导入）。
pytest.importorskip("pymilvus")
pytest.importorskip("neo4j")


class TestInitMilvus:
    """Tests for Milvus initialization."""

    def test_create_medical_kb_collection(self) -> None:
        from scripts.init_milvus import create_medical_kb_collection

        mock_client = MagicMock()

        with patch("pymilvus.utility") as mock_utility:
            mock_utility.has_collection.return_value = False
            create_medical_kb_collection(mock_client)

        mock_client.create_collection.assert_called_once()
        call_kwargs = mock_client.create_collection.call_args.kwargs
        assert call_kwargs["metric_type"] == "COSINE"
        assert call_kwargs["index_type"] == "HNSW"

    def test_create_medical_entities_collection(self) -> None:
        from scripts.init_milvus import create_medical_entities_collection

        mock_client = MagicMock()

        with patch("pymilvus.utility") as mock_utility:
            mock_utility.has_collection.return_value = False
            create_medical_entities_collection(mock_client)

        mock_client.create_collection.assert_called_once()
        call_kwargs = mock_client.create_collection.call_args.kwargs
        assert call_kwargs["vector_field_name"] == "embedding"

    def test_create_reports_collection(self) -> None:
        from scripts.init_milvus import create_reports_collection

        mock_client = MagicMock()

        with patch("pymilvus.utility") as mock_utility:
            mock_utility.has_collection.return_value = False
            create_reports_collection(mock_client)

        mock_client.create_collection.assert_called_once()
        call_kwargs = mock_client.create_collection.call_args.kwargs
        assert call_kwargs["vector_field_name"] == "embedding"

    def test_init_milvus_calls_all_create_functions(self) -> None:
        """Test that init_milvus creates all three collections."""
        from scripts.init_milvus import (
            create_medical_entities_collection,
            create_medical_kb_collection,
            create_reports_collection,
            init_milvus,
        )

        mock_client = MagicMock()

        with patch("pymilvus.MilvusClient", return_value=mock_client), \
             patch("pymilvus.connections"), \
             patch("scripts.init_milvus.create_medical_kb_collection") as mock_kb, \
             patch("scripts.init_milvus.create_medical_entities_collection") as mock_entities, \
             patch("scripts.init_milvus.create_reports_collection") as mock_reports:
            init_milvus()

        mock_kb.assert_called_once_with(mock_client)
        mock_entities.assert_called_once_with(mock_client)
        mock_reports.assert_called_once_with(mock_client)


class TestInitNeo4j:
    """Tests for Neo4j initialization."""

    def test_create_constraints(self) -> None:
        from scripts.init_neo4j import create_constraints

        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=None)

        create_constraints(mock_driver)

        assert mock_session.run.call_count == 7

    def test_create_indexes(self) -> None:
        from scripts.init_neo4j import create_indexes

        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=None)

        create_indexes(mock_driver)

        assert mock_session.run.call_count == 5

    def test_load_medical_ontology(self) -> None:
        from scripts.init_neo4j import load_medical_ontology

        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=None)

        load_medical_ontology(mock_driver)

        assert mock_session.run.call_count >= 20

    @patch("scripts.init_neo4j.load_medical_ontology")
    @patch("scripts.init_neo4j.create_indexes")
    @patch("scripts.init_neo4j.create_constraints")
    @patch("neo4j.GraphDatabase")
    def test_init_neo4j_full_flow(
        self, mock_gdb_cls, mock_constraints, mock_indexes, mock_ontology
    ) -> None:
        from scripts.init_neo4j import init_neo4j

        mock_driver = MagicMock()
        mock_gdb_cls.driver.return_value = mock_driver
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=None)

        init_neo4j()

        mock_constraints.assert_called_once_with(mock_driver)
        mock_indexes.assert_called_once_with(mock_driver)
        mock_ontology.assert_called_once_with(mock_driver)
