"""Embedding client with deterministic offline fallback."""

from __future__ import annotations

import hashlib
from typing import List, Optional

import numpy as np

from app.config import get_settings


class EmbeddingClient:
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        normalize: bool = True,
    ) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.device = device or "cpu"
        self.normalize = normalize
        self._model = None
        self._model_load_failed = False
        self._dimension = 1024
        self._offline = get_settings().EMBEDDING_OFFLINE

    @property
    def model(self):
        if self._model is None and not self._model_load_failed and not self._offline:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name, device=self.device)
            except Exception:
                # Offline/dev environments must remain usable without silently
                # making every request retry a network model download.
                self._model_load_failed = True
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    def _fallback_embedding(self, text: str) -> List[float]:
        """Stable hashed embedding for local tests and service degradation.

        It is not a semantic replacement for BGE; production retrieval should
        expose the model health and use the configured embedding model.
        """
        values: list[float] = []
        seed = text.encode("utf-8")
        for index in range(self._dimension):
            digest = hashlib.blake2b(seed + index.to_bytes(4, "little"), digest_size=4).digest()
            values.append((int.from_bytes(digest, "little") / 2**31) - 1.0)
        if self.normalize:
            norm = float(np.linalg.norm(values))
            if norm:
                values = [value / norm for value in values]
        return values

    def embed(self, text: str) -> List[float]:
        if self.model is None:
            return self._fallback_embedding(text)
        embedding = self.model.encode(
            text,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        return embedding.tolist()

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        if self.model is None:
            return [self._fallback_embedding(text) for text in texts]
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def compute_similarity(self, text1: str, text2: str) -> float:
        if text1 == text2:
            return 1.0
        emb1 = np.array(self.embed(text1))
        emb2 = np.array(self.embed(text2))
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.clip(np.dot(emb1, emb2) / (norm1 * norm2), 0.0, 1.0))


_embedding_client: EmbeddingClient | None = None


def get_embedding_client() -> EmbeddingClient:
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = EmbeddingClient()
    return _embedding_client
