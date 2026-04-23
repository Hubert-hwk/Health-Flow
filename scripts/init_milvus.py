"""
Milvus vector database initialization script.
Creates collections and indexes for the HealthFlow knowledge base.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pymilvus import MilvusClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = "19530"
DIM_EMBEDDING = 1024  # matches the embedding model output dimension


def create_medical_kb_collection(client: "MilvusClient") -> None:
    """Create the medical knowledge base collection with optimized HNSW index."""
    from pymilvus import CollectionSchema, DataType, FieldSchema, utility

    collection_name = "medical_kb"
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
        FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="embedding", dtype=DataType.FLOAT16_VECTOR, dim=DIM_EMBEDDING),
        FieldSchema(name="metadata", dtype=DataType.JSON),
    ]
    schema = CollectionSchema(
        fields=fields,
        description="Medical knowledge base with disease, symptom, treatment, drug information",
    )

    if utility.has_collection(collection_name):
        logger.info(f"Collection '{collection_name}' already exists, dropping and recreating")
        client.drop_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        vector_field_name="embedding",
        metric_type="COSINE",
        index_type="HNSW",
        params={"M": 16, "efConstruction": 256},
    )
    logger.info(f"Collection '{collection_name}' created with HNSW index")


def create_medical_entities_collection(client: "MilvusClient") -> None:
    """Create the medical entities collection for entity-level retrieval."""
    from pymilvus import CollectionSchema, DataType, FieldSchema, utility

    collection_name = "medical_entities"
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
        FieldSchema(name="entity_type", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(name="embedding", dtype=DataType.FLOAT16_VECTOR, dim=DIM_EMBEDDING),
        FieldSchema(name="metadata", dtype=DataType.JSON),
    ]
    schema = CollectionSchema(
        fields=fields,
        description="Medical entity registry for entity-level RAG retrieval",
    )

    if utility.has_collection(collection_name):
        logger.info(f"Collection '{collection_name}' already exists, dropping and recreating")
        client.drop_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        vector_field_name="embedding",
        metric_type="COSINE",
        index_type="HNSW",
        params={"M": 16, "efConstruction": 256},
    )
    logger.info(f"Collection '{collection_name}' created with HNSW index")


def create_reports_collection(client: "MilvusClient") -> None:
    """Create the medical reports collection for report content retrieval."""
    from pymilvus import CollectionSchema, DataType, FieldSchema, utility

    collection_name = "medical_reports"
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
        FieldSchema(name="report_type", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="patient_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="embedding", dtype=DataType.FLOAT16_VECTOR, dim=DIM_EMBEDDING),
        FieldSchema(name="metadata", dtype=DataType.JSON),
    ]
    schema = CollectionSchema(
        fields=fields,
        description="Medical reports vector store for similarity search",
    )

    if utility.has_collection(collection_name):
        logger.info(f"Collection '{collection_name}' already exists, dropping and recreating")
        client.drop_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        vector_field_name="embedding",
        metric_type="COSINE",
        index_type="HNSW",
        params={"M": 16, "efConstruction": 256},
    )
    logger.info(f"Collection '{collection_name}' created with HNSW index")


def init_milvus(host: str = MILVUS_HOST, port: str = MILVUS_PORT) -> "MilvusClient":
    """Connect to Milvus and initialise all collections."""
    from pymilvus import MilvusClient, connections

    logger.info(f"Connecting to Milvus at {host}:{port}")
    connections.connect(host=host, port=port, alias="default")
    client = MilvusClient(uri=f"http://{host}:{port}")

    create_medical_kb_collection(client)
    create_medical_entities_collection(client)
    create_reports_collection(client)

    logger.info("Milvus initialisation complete")
    return client


if __name__ == "__main__":
    init_milvus()
