"""Initialise the optional Milvus collections used by HealthFlow.

The application only requires ``medical_reports`` at runtime.  The other two
collections are kept here as explicit knowledge-base/entity boundaries so a
deployment can initialise the full retrieval topology in one command.
"""

from __future__ import annotations

import argparse
from typing import Any

from app.config import get_settings


EMBEDDING_DIM = 1024


def _schema(collection_name: str) -> Any:
    """Build a compatible schema without importing the SDK at module import."""

    from pymilvus import CollectionSchema, DataType, FieldSchema

    primary = FieldSchema(
        name="id", dtype=DataType.INT64, is_primary=True, auto_id=False
    )
    fields = [primary]
    if collection_name == "medical_reports":
        fields.extend(
            [
                FieldSchema(name="report_id", dtype=DataType.INT64),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(
                    name="embedding",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=EMBEDDING_DIM,
                ),
                FieldSchema(name="department", dtype=DataType.VARCHAR, max_length=64),
            ]
        )
    else:
        fields.extend(
            [
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(
                    name="embedding",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=EMBEDDING_DIM,
                ),
            ]
        )
    return CollectionSchema(fields=fields, description=f"HealthFlow {collection_name}")


def _create_collection(client: Any, collection_name: str) -> None:
    """Create one collection if it is absent.

    ``utility.has_collection`` is used when available for compatibility with
    older pymilvus deployments; the client call itself remains easy to mock in
    CI and in local development.
    """

    from pymilvus import utility

    try:
        if utility.has_collection(collection_name):
            return
    except Exception:
        # A remote utility connection may not be configured yet.  Let the
        # MilvusClient report the real connection error instead.
        pass

    client.create_collection(
        collection_name=collection_name,
        schema=_schema(collection_name),
        vector_field_name="embedding",
        metric_type="COSINE",
        index_type="HNSW",
        params={"M": 16, "efConstruction": 256},
    )


def create_medical_kb_collection(client: Any) -> None:
    _create_collection(client, "medical_kb")


def create_medical_entities_collection(client: Any) -> None:
    _create_collection(client, "medical_entities")


def create_reports_collection(client: Any) -> None:
    _create_collection(client, "medical_reports")


def init_milvus(drop_existing: bool = False) -> Any:
    """Connect to Milvus and initialise all HealthFlow collections."""

    from pymilvus import MilvusClient, connections

    settings = get_settings()
    connections.connect(host=settings.MILVUS_HOST, port=str(settings.MILVUS_PORT))
    client = MilvusClient(uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}")

    if drop_existing:
        for collection_name in ("medical_kb", "medical_entities", "medical_reports"):
            try:
                if client.has_collection(collection_name):
                    client.drop_collection(collection_name)
            except Exception:
                # Older clients do not expose has_collection; creation below
                # will still provide the actionable server-side error.
                pass

    create_medical_kb_collection(client)
    create_medical_entities_collection(client)
    create_reports_collection(client)
    return client


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drop-existing", action="store_true")
    args = parser.parse_args()
    init_milvus(args.drop_existing)
