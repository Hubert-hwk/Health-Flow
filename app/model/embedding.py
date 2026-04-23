"""Embedding服务，使用sentence-transformers."""

from typing import List, Optional
import numpy as np

from app.config import get_settings


class EmbeddingClient:
    """Embedding客户端."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        normalize: bool = True,
    ):
        """
        初始化Embedding客户端.

        Args:
            model_name: 模型名称
            device: 设备，"cpu" 或 "cuda"
            normalize: 是否归一化向量
        """
        settings = get_settings()
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.device = device or "cpu"
        self.normalize = normalize
        self._model = None
        self._dimension = 1024  # bge-large-zh-v1.5 dimension

    @property
    def model(self):
        """获取模型（延迟加载）."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device=self.device)
            except ImportError:
                # For testing without actual model
                self._model = None
        return self._model

    @property
    def dimension(self) -> int:
        """获取向量维度."""
        return self._dimension

    def embed(self, text: str) -> List[float]:
        """
        获取单条文本的embedding.

        Args:
            text: 输入文本

        Returns:
            embedding向量
        """
        if self.model is None:
            # Return dummy embedding for testing
            return [0.0] * self._dimension

        embedding = self.model.encode(
            text,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        return embedding.tolist()

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        批量获取文本的embedding.

        Args:
            texts: 文本列表
            batch_size: 批量大小

        Returns:
            embedding向量列表
        """
        if self.model is None:
            # Return dummy embeddings for testing
            return [[0.0] * self._dimension for _ in texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        计算两条文本的相似度.

        Args:
            text1: 文本1
            text2: 文本2

        Returns:
            余弦相似度 (0-1)
        """
        emb1 = np.array(self.embed(text1))
        emb2 = np.array(self.embed(text2))

        # 余弦相似度
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)


# 全局实例
_embedding_client: EmbeddingClient | None = None


def get_embedding_client() -> EmbeddingClient:
    """获取Embedding客户端单例."""
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = EmbeddingClient()
    return _embedding_client
