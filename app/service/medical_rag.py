"""MedicalRAG Service - 医疗混合RAG服务.

向量检索 + 知识图谱混合召回。
"""

from typing import List, Dict, Any, Optional, Tuple

from app.model.embedding import get_embedding_client
from app.model.llm import get_llm_client
from app.data.milvus_client import get_milvus_client
from app.data.neo4j_client import get_neo4j_client


class MedicalRAGService:
    """
    医疗混合RAG服务。

    核心职责:
    1. 向量检索召回
    2. 知识图谱增强召回
    3. 混合融合与重排序
    4. 检索过滤
    """

    def __init__(
        self,
        vector_weight: float = 0.6,
        kg_weight: float = 0.4,
        top_k: int = 5
    ):
        """
        初始化MedicalRAG服务。

        Args:
            vector_weight: 向量检索权重
            kg_weight: 知识图谱权重
            top_k: 返回结果数量
        """
        self.vector_weight = vector_weight
        self.kg_weight = kg_weight
        self.top_k = top_k

        self._embedding_client = None
        self._milvus_client = None
        self._neo4j_client = None
        self._llm_client = None

    @property
    def embedding_client(self):
        """获取Embedding客户端。"""
        if self._embedding_client is None:
            self._embedding_client = get_embedding_client()
        return self._embedding_client

    @property
    def milvus_client(self):
        """获取Milvus客户端。"""
        if self._milvus_client is None:
            self._milvus_client = get_milvus_client()
        return self._milvus_client

    @property
    def neo4j_client(self):
        """获取Neo4j客户端。"""
        if self._neo4j_client is None:
            self._neo4j_client = get_neo4j_client()
        return self._neo4j_client

    @property
    def llm_client(self):
        """获取LLM客户端。"""
        if self._llm_client is None:
            self._llm_client = get_llm_client()
        return self._llm_client

    def vector_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        department: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        向量检索。

        Args:
            query: 查询文本
            top_k: 返回数量
            department: 科室过滤

        Returns:
            检索结果列表
        """
        # 生成查询向量
        query_embedding = self.embedding_client.embed(query)

        # 执行搜索
        results = self.milvus_client.search(
            query_embedding=query_embedding,
            top_k=top_k or self.top_k,
            department=department
        )

        return results

    def kg_search(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        知识图谱检索。

        从query中提取医学实体，查询相关知识。

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            知识图谱结果列表
        """
        # 提取医学实体
        entities = self._extract_medical_entities(query)

        if not entities:
            return []

        results = []

        # 查询每个实体的相关知识
        for entity in entities[:3]:  # 最多处理3个实体
            # 查询相关症状
            symptoms = self.neo4j_client.get_related_symptoms(entity)
            for symptom in symptoms:
                results.append({
                    "type": "symptom",
                    "entity": entity,
                    "name": symptom["name"],
                    "description": symptom.get("description", ""),
                    "source": "knowledge_graph"
                })

            # 查询相关药品
            drugs = self.neo4j_client.get_related_drugs(entity)
            for drug in drugs:
                results.append({
                    "type": "drug",
                    "entity": entity,
                    "name": drug["name"],
                    "description": drug.get("description", ""),
                    "source": "knowledge_graph"
                })

            # 查询诊断路径
            diagnosis_paths = self.neo4j_client.find_diagnosis_path([entity])
            for path in diagnosis_paths:
                results.append({
                    "type": "diagnosis",
                    "entity": entity,
                    "name": path["disease"],
                    "description": path.get("description", ""),
                    "matched_symptoms": path.get("matched_symptoms", []),
                    "source": "knowledge_graph"
                })

        # 去重并限制数量
        seen = set()
        unique_results = []
        for r in results:
            key = f"{r['type']}:{r['name']}"
            if key not in seen:
                seen.add(key)
                unique_results.append(r)

        return unique_results[:top_k or self.top_k]

    def hybrid_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        department: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        混合搜索 - 向量 + 知识图谱。

        Args:
            query: 查询文本
            top_k: 返回数量
            department: 科室过滤

        Returns:
            混合检索结果列表
        """
        k = top_k or self.top_k

        # 并行执行向量搜索和KG搜索
        vector_results = self.vector_search(query, k * 2, department)
        kg_results = self.kg_search(query, k * 2)

        # 融合结果
        fused_results = self._fuse_results(vector_results, kg_results, query)

        return fused_results[:k]

    def _fuse_results(
        self,
        vector_results: List[Dict[str, Any]],
        kg_results: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """
        融合向量和KG的检索结果。

        Args:
            vector_results: 向量检索结果
            kg_results: KG检索结果
            query: 原始查询

        Returns:
            融合后的结果列表
        """
        # 为每个结果分配分数
        scored_results: Dict[str, Tuple[float, Dict[str, Any]]] = {}

        # 向量结果打分 (0-1)
        for i, result in enumerate(vector_results):
            key = f"vector:{result.get('content', '')[:50]}"
            # 排名得分
            rank_score = 1.0 - (i / len(vector_results)) if vector_results else 0
            # 距离得分（Milvus返回的是distance）
            distance_score = 1.0 - result.get("distance", 0.5)
            vector_score = self.vector_weight * (rank_score * 0.4 + distance_score * 0.6)
            scored_results[key] = (vector_score, result)

        # KG结果打分
        for i, result in enumerate(kg_results):
            key = f"kg:{result.get('name', '')[:50]}"
            rank_score = 1.0 - (i / len(kg_results)) if kg_results else 0
            kg_score = self.kg_weight * rank_score

            if key in scored_results:
                old_score, old_result = scored_results[key]
                scored_results[key] = (old_score + kg_score, old_result)
            else:
                scored_results[key] = (kg_score, result)

        # 按分数排序
        sorted_results = sorted(
            scored_results.values(),
            key=lambda x: x[0],
            reverse=True
        )

        return [r[1] for r in sorted_results]

    def _extract_medical_entities(self, query: str) -> List[str]:
        """
        从查询中提取医学实体。

        Args:
            query: 查询文本

        Returns:
            实体名称列表
        """
        prompt = f"""从以下文本中提取医学实体（疾病、症状、指标名称）。

文本:
{query}

只输出实体名称，用逗号分隔。例如: 空腹血糖, 糖尿病, 多饮

只输出实体名称，不要其他内容。"""

        try:
            response = self.llm_client.chat(
                messages=[
                    {"role": "system", "content": "你是一个医疗实体提取助手。"},
                    {"role": "user", "content": prompt}
                ]
            )

            # 解析响应
            entities = [e.strip() for e in response.split(",") if e.strip()]
            return entities

        except Exception:
            return []

    def build_context(
        self,
        query: str,
        department: Optional[str] = None
    ) -> str:
        """
        构建RAG上下文。

        Args:
            query: 查询文本
            department: 科室

        Returns:
            上下文字符串
        """
        results = self.hybrid_search(query, department=department)

        if not results:
            return ""

        context_parts = ["## 参考信息:\n"]

        for i, result in enumerate(results, 1):
            source = result.get("source", "unknown")
            if source == "knowledge_graph":
                context_parts.append(
                    f"{i}. [{result.get('type', 'knowledge')}] {result.get('name', '')}"
                )
                if result.get("description"):
                    context_parts.append(f"   {result['description']}")
            else:
                context_parts.append(
                    f"{i}. [文档] {result.get('content', '')[:200]}"
                )

        return "\n".join(context_parts)

    def retrieve_and_build_context(
        self,
        query: str,
        department: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        检索并构建上下文。

        Args:
            query: 查询文本
            department: 科室

        Returns:
            (原始检索结果, 上下文字符串)
        """
        results = self.hybrid_search(query, department=department)
        context = self.build_context_from_results(results)

        return results, context

    def build_context_from_results(
        self,
        results: List[Dict[str, Any]]
    ) -> str:
        """
        从检索结果构建上下文字符串。

        Args:
            results: 检索结果

        Returns:
            上下文字符串
        """
        if not results:
            return ""

        context_parts = ["## 参考信息:\n"]

        for i, result in enumerate(results, 1):
            source = result.get("source", "unknown")
            if source == "knowledge_graph":
                context_parts.append(
                    f"{i}. [{result.get('type', 'knowledge')}] {result.get('name', '')}"
                )
                if result.get("description"):
                    context_parts.append(f"   描述: {result['description']}")
            else:
                content = result.get("content", "")
                context_parts.append(f"{i}. [文档] {content[:200]}")

        return "\n".join(context_parts)


# 全局实例
_medical_rag_service: MedicalRAGService | None = None


def get_medical_rag_service() -> MedicalRAGService:
    """获取MedicalRAG服务实例。"""
    global _medical_rag_service
    if _medical_rag_service is None:
        _medical_rag_service = MedicalRAGService()
    return _medical_rag_service
