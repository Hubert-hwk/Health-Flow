"""Milvus client with an explicit optional-service boundary."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.config import get_settings


class MilvusClient:
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection_name: str = "medical_reports",
        dim: int = 1024,
    ) -> None:
        settings = get_settings()
        self.host = host or settings.MILVUS_HOST
        self.port = port or settings.MILVUS_PORT
        self.collection_name = collection_name
        self.dim = dim
        self._client = None
        self._collection_ready = False
        self.last_error: Optional[str] = None

    @property
    def client(self):
        if self._client is None:
            try:
                from pymilvus import MilvusClient as SDKMilvusClient

                self._client = SDKMilvusClient(uri=f"http://{self.host}:{self.port}")
            except Exception as exc:
                self.last_error = str(exc)
                self._client = None
        return self._client

    @property
    def available(self) -> bool:
        return self.client is not None

    def connect(self):
        return self.client

    def ensure_collection(self) -> bool:
        if not self.client:
            return False
        if self._collection_ready:
            return True
        try:
            collections = self.client.list_collections()
            if self.collection_name not in collections:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    dimension=self.dim,
                    primary_field_name="id",
                    vector_field_name="embedding",
                    metric_type="COSINE",
                    auto_id=False,
                    enable_dynamic_field=True,
                )
            self._collection_ready = True
            return True
        except Exception as exc:
            self.last_error = str(exc)
            return False

    def create_collection(self, drop_existing: bool = False):
        if not self.client:
            return False
        try:
            collections = self.client.list_collections()
            if drop_existing and self.collection_name in collections:
                self.client.drop_collection(self.collection_name)
                collections = []
            self._collection_ready = False
            return self.ensure_collection()
        except Exception as exc:
            self.last_error = str(exc)
            return False

    def insert(
        self,
        report_ids: List[int],
        texts: List[str],
        embeddings: List[List[float]],
        departments: Optional[List[str]] = None,
    ) -> List[int]:
        if not self.ensure_collection():
            return []
        if not (len(report_ids) == len(texts) == len(embeddings)):
            raise ValueError("report_ids、texts 和 embeddings 长度必须一致")
        data = []
        for index, (report_id, text, embedding) in enumerate(zip(report_ids, texts, embeddings)):
            data.append(
                {
                    "id": int(report_id),
                    "report_id": int(report_id),
                    "content": text,
                    "embedding": embedding,
                    "department": departments[index] if departments and index < len(departments) else "",
                }
            )
        try:
            result = self.client.insert(collection_name=self.collection_name, data=data)
            return list(result.get("ids", report_ids)) if isinstance(result, dict) else report_ids
        except Exception as exc:
            self.last_error = str(exc)
            return []

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        department: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not self.ensure_collection():
            return []
        filter_expr = None
        if department:
            escaped = department.replace('"', '\\"')
            filter_expr = f'department == "{escaped}"'
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                data=[query_embedding],
                limit=max(1, min(top_k, 100)),
                filter=filter_expr,
                output_fields=["report_id", "content", "department"],
            )
            hits = results[0] if results else []
            parsed: list[dict[str, Any]] = []
            for hit in hits:
                entity = hit.get("entity", hit) if hasattr(hit, "get") else hit
                parsed.append(
                    {
                        "id": hit.get("id") if hasattr(hit, "get") else None,
                        "report_id": entity.get("report_id") if hasattr(entity, "get") else None,
                        "content": entity.get("content", "") if hasattr(entity, "get") else "",
                        "department": entity.get("department", "") if hasattr(entity, "get") else "",
                        "distance": hit.get("distance", 0.0) if hasattr(hit, "get") else 0.0,
                    }
                )
            return parsed
        except Exception as exc:
            self.last_error = str(exc)
            return []

    def delete_by_report_id(self, report_id: int) -> bool:
        if not self.ensure_collection():
            return False
        try:
            self.client.delete(
                collection_name=self.collection_name,
                filter=f"report_id == {int(report_id)}",
            )
            return True
        except Exception as exc:
            self.last_error = str(exc)
            return False

    def flush(self) -> None:
        # MilvusClient commits inserts synchronously; older SDKs may expose flush.
        if self.client and hasattr(self.client, "flush"):
            try:
                self.client.flush(self.collection_name)
            except Exception as exc:
                self.last_error = str(exc)

    def close(self) -> None:
        self._client = None
        self._collection_ready = False


_milvus_client: MilvusClient | None = None


def get_milvus_client() -> MilvusClient:
    global _milvus_client
    if _milvus_client is None:
        _milvus_client = MilvusClient()
    return _milvus_client
