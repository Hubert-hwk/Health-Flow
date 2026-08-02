"""Hybrid medical retrieval with source-aware evidence formatting."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from app.data.milvus_client import get_milvus_client
from app.data.neo4j_client import get_neo4j_client
from app.model.embedding import get_embedding_client
from app.model.llm import get_llm_client


MEDICAL_ENTITY_TERMS = (
    "血糖", "糖尿病", "糖化血红蛋白", "血压", "血脂", "尿酸", "甲状腺", "心脏",
    "胸痛", "咳嗽", "呼吸", "胃", "肝", "胆", "腹痛", "便秘", "体检报告",
)


class MedicalRAGService:
    """Dense retrieval + constrained graph expansion + reciprocal-rank fusion."""

    def __init__(self, vector_weight: float = 0.6, kg_weight: float = 0.4, top_k: int = 5) -> None:
        total = max(vector_weight + kg_weight, 1e-6)
        self.vector_weight = vector_weight / total
        self.kg_weight = kg_weight / total
        self.top_k = top_k
        self._embedding_client = None
        self._milvus_client = None
        self._neo4j_client = None
        self._llm_client = None

    @property
    def embedding_client(self):
        if self._embedding_client is None:
            self._embedding_client = get_embedding_client()
        return self._embedding_client

    @property
    def milvus_client(self):
        if self._milvus_client is None:
            self._milvus_client = get_milvus_client()
        return self._milvus_client

    @property
    def neo4j_client(self):
        if self._neo4j_client is None:
            self._neo4j_client = get_neo4j_client()
        return self._neo4j_client

    @property
    def llm_client(self):
        if self._llm_client is None:
            self._llm_client = get_llm_client()
        return self._llm_client

    def vector_search(
        self, query: str, top_k: Optional[int] = None, department: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        try:
            embedding = self.embedding_client.embed(query)
            values = self.milvus_client.search(embedding, top_k or self.top_k, department)
        except Exception:
            return []
        results: list[dict[str, Any]] = []
        for index, item in enumerate(values):
            item = dict(item)
            item.setdefault("source", "vector")
            item.setdefault("source_id", f"V{index + 1}")
            item.setdefault("content", "")
            item.setdefault("score", self._distance_to_score(item.get("distance"), index, len(values)))
            results.append(item)
        return results

    def kg_search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        entities = self._extract_medical_entities(query)
        results: list[dict[str, Any]] = []
        for entity in entities[:5]:
            try:
                if hasattr(self.neo4j_client, "query_by_entity"):
                    graph_items = self.neo4j_client.query_by_entity(entity, limit=5)
                else:
                    graph_items = []
                for item in graph_items:
                    result = dict(item)
                    result.update(
                        {
                            "source": "graph",
                            "source_id": result.get("source_id") or f"G-{entity}",
                            "entity": entity,
                            "content": result.get("description") or self._format_graph_item(result),
                            "path": result.get("path", []),
                        }
                    )
                    results.append(result)

                # Keep the domain-specific methods as a compatibility path for
                # older Neo4j adapters and small test doubles.
                for item in getattr(self.neo4j_client, "get_related_symptoms", lambda _: [])(entity):
                    results.append({
                        "type": "symptom", "name": item.get("name", ""),
                        "description": item.get("description", ""), "entity": entity,
                        "source": "graph", "source_id": f"G-symptom-{entity}",
                    })
                for item in getattr(self.neo4j_client, "get_related_drugs", lambda _: [])(entity):
                    results.append({
                        "type": "drug", "name": item.get("name", ""),
                        "description": item.get("description", ""), "entity": entity,
                        "source": "graph", "source_id": f"G-drug-{entity}",
                    })
                for item in getattr(self.neo4j_client, "find_diagnosis_path", lambda _: [])([entity]):
                    results.append({
                        "type": "diagnosis", "name": item.get("disease", ""),
                        "description": item.get("description", ""), "entity": entity,
                        "matched_symptoms": item.get("matched_symptoms", []),
                        "source": "graph", "source_id": f"G-diagnosis-{entity}",
                    })
            except Exception:
                continue
        return self._deduplicate(results)[: top_k or self.top_k]

    def hybrid_search(
        self, query: str, top_k: Optional[int] = None, department: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        k = top_k or self.top_k
        vector_results = self.vector_search(query, k * 2, department)
        graph_results = self.kg_search(query, k * 2)
        return self._fuse_results(vector_results, graph_results)[:k]

    def _fuse_results(
        self,
        vector_results: List[Dict[str, Any]],
        graph_results: List[Dict[str, Any]],
        query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        fused: dict[str, dict[str, Any]] = {}

        def add(items: list[dict[str, Any]], weight: float, prefix: str) -> None:
            size = max(1, len(items))
            for rank, item in enumerate(items, start=1):
                key = str(item.get("source_id") or item.get("id") or item.get("content", "")[:120])
                key = f"{prefix}:{key}"
                contribution = weight / (30 + rank) + weight * float(item.get("score", 0.0)) * 0.1
                if key in fused:
                    fused[key]["score"] += contribution
                else:
                    fused[key] = {**item, "score": contribution}

        add(vector_results, self.vector_weight, "V")
        add(graph_results, self.kg_weight, "G")
        return sorted(fused.values(), key=lambda item: item["score"], reverse=True)

    def _extract_medical_entities(self, query: str) -> list[str]:
        entities = [term for term in MEDICAL_ENTITY_TERMS if term in query]
        if entities:
            return entities
        try:
            response = self.llm_client.chat(
                messages=[
                    {"role": "system", "content": "提取医疗实体，只返回逗号分隔的实体名。"},
                    {"role": "user", "content": query},
                ]
            )
            return [item.strip() for item in re.split(r"[,，、\n]", response) if item.strip()][:5]
        except Exception:
            return []

    @staticmethod
    def _distance_to_score(distance: Any, rank: int, size: int) -> float:
        try:
            value = float(distance)
        except (TypeError, ValueError):
            return max(0.0, 1.0 - rank / max(1, size))
        return max(0.0, min(1.0, value if 0 <= value <= 1 else 1.0 / (1.0 + value)))

    @staticmethod
    def _format_graph_item(item: dict[str, Any]) -> str:
        entity = item.get("entity", "")
        related = item.get("related_entity", "")
        relation = item.get("relation", "related_to")
        return f"{entity} -[{relation}]-> {related}".strip()

    @staticmethod
    def _deduplicate(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for item in items:
            key = str(item.get("source_id") or item.get("content", "")[:160])
            if key not in seen:
                seen.add(key)
                unique.append(item)
        return unique

    def retrieve_and_build_context(
        self, query: str, department: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], str]:
        results = self.hybrid_search(query, department=department)
        return results, self.build_context_from_results(results)

    def build_context(self, query: str, department: Optional[str] = None) -> str:
        return self.retrieve_and_build_context(query, department)[1]

    def build_context_from_results(self, results: List[Dict[str, Any]]) -> str:
        if not results:
            return ""
        lines = ["## 可引用医疗证据（仅用于辅助回答）"]
        for index, result in enumerate(results, start=1):
            source_id = str(result.get("source_id") or f"S{index}")
            content = str(result.get("content") or result.get("description") or result.get("name") or "")
            path = result.get("path") or []
            path_text = f"；路径：{' -> '.join(path)}" if path else ""
            lines.append(f"[{source_id}] {content[:800]}{path_text}")
        return "\n".join(lines)


_medical_rag_service: MedicalRAGService | None = None


def get_medical_rag_service() -> MedicalRAGService:
    global _medical_rag_service
    if _medical_rag_service is None:
        _medical_rag_service = MedicalRAGService()
    return _medical_rag_service
