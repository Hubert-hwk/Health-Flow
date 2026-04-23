"""Milvus vector database client."""

from typing import List, Optional, Dict, Any

from app.config import get_settings


class MilvusClient:
    """Milvus vector database client."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection_name: str = "medical_reports",
        dim: int = 1024,
    ):
        """
        Initialize Milvus client.

        Args:
            host: Milvus host
            port: Milvus port
            collection_name: Collection name
            dim: Embedding dimension
        """
        settings = get_settings()
        self.host = host or settings.MILVUS_HOST
        self.port = port or settings.MILVUS_PORT
        self.collection_name = collection_name
        self.dim = dim
        self._client = None
        self._collection = None

    @property
    def client(self):
        """Get Milvus client (lazy initialization)."""
        if self._client is None:
            try:
                from pymilvus import MilvusClient
                self._client = MilvusClient(uri=f"http://{self.host}:{self.port}")
            except ImportError:
                # For testing without actual Milvus
                self._client = None
        return self._client

    @property
    def collection(self):
        """Get Milvus collection."""
        if self._collection is None and self.client:
            self._collection = self.client.get_collection(self.collection_name)
        return self._collection

    def connect(self):
        """Connect to Milvus."""
        return self.client

    def create_collection(self, drop_existing: bool = False):
        """
        Create collection with schema.

        Args:
            drop_existing: Whether to drop existing collection
        """
        if not self.client:
            return

        try:
            if drop_existing and self.collection_name in self.client.list_collections():
                self.client.drop_collection(self.collection_name)

            if self.collection_name not in self.client.list_collections():
                self.client.create_collection(
                    collection_name=self.collection_name,
                    dimension=self.dim,
                    primary_field_name="id",
                    vector_field_name="embedding",
                    metric_type="COSINE",
                    index_type="AUTOINDEX",
                )
        except Exception:
            # Collection might already exist
            pass

    def insert(
        self,
        report_ids: List[int],
        texts: List[str],
        embeddings: List[List[float]],
        departments: Optional[List[str]] = None,
    ) -> List[int]:
        """
        Insert vectors into collection.

        Args:
            report_ids: Report IDs
            texts: Text content
            embeddings: Embedding vectors
            departments: Department labels

        Returns:
            List of inserted IDs
        """
        if not self.client:
            return []

        data = [
            {"id": rid, "report_id": rid, "content": text, "embedding": emb}
            for rid, text, emb in zip(report_ids, texts, embeddings)
        ]
        if departments:
            for i, d in enumerate(data):
                data[i]["department"] = departments[i] if i < len(departments) else None

        try:
            result = self.client.insert(
                collection_name=self.collection_name,
                data=data,
            )
            return result.get("ids", [])
        except Exception:
            return []

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        department: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search similar vectors.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results
            department: Optional department filter

        Returns:
            List of search results
        """
        if not self.client:
            return []

        filter_expr = f'department == "{department}"' if department else None

        try:
            results = self.client.search(
                collection_name=self.collection_name,
                data=[query_embedding],
                limit=top_k,
                filter=filter_expr,
                output_fields=["id", "report_id", "content", "department"],
            )

            return [
                {
                    "id": hit["id"],
                    "report_id": hit["report_id"],
                    "content": hit["content"],
                    "department": hit["department"],
                    "distance": hit["distance"],
                }
                for hit in results[0]
            ]
        except Exception:
            return []

    def delete_by_report_id(self, report_id: int) -> bool:
        """
        Delete vectors by report ID.

        Args:
            report_id: Report ID

        Returns:
            Success status
        """
        if not self.client:
            return False

        try:
            self.client.delete(
                collection_name=self.collection_name,
                filter=f"report_id == {report_id}",
            )
            return True
        except Exception:
            return False

    def flush(self):
        """Flush collection to persist data."""
        if self.client:
            try:
                self.client.flush(self.collection_name)
            except Exception:
                pass

    def close(self):
        """Close client connection."""
        if self._client:
            self._client = None
            self._collection = None


# Global instance
_milvus_client: MilvusClient | None = None


def get_milvus_client() -> MilvusClient:
    """Get Milvus client singleton."""
    global _milvus_client
    if _milvus_client is None:
        _milvus_client = MilvusClient()
    return _milvus_client
