"""Knowledge Graph API endpoints.

提供知识图谱查询接口。
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, Path

router = APIRouter()


def _get_neo4j_client():
    # Resolve lazily so tests/deployments can replace the optional connector.
    from app.data.neo4j_client import get_neo4j_client

    return get_neo4j_client()


@router.post("/kg/query")
def query_knowledge_graph(
    entity: str = Query(..., description="实体名称"),
    limit: int = Query(10, description="返回结果数量")
):
    """
    通用知识图谱查询。

    根据实体名称查询相关信息。

    Args:
        entity: 实体名称
        limit: 返回结果数量

    Returns:
        实体相关信息
    """
    neo4j_client = _get_neo4j_client()

    try:
        results = neo4j_client.query_by_entity(entity, limit=limit)
        return {
            "entity": entity,
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


@router.get("/kg/symptoms/{disease}")
def get_disease_symptoms(
    disease: str = Path(..., description="疾病名称")
):
    """
    查询疾病相关症状。

    Args:
        disease: 疾病名称

    Returns:
        相关症状列表
    """
    neo4j_client = _get_neo4j_client()

    try:
        symptoms = neo4j_client.get_related_symptoms(disease)
        return {
            "disease": disease,
            "symptoms": symptoms,
            "count": len(symptoms)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


@router.get("/kg/drugs/{disease}")
def get_disease_drugs(
    disease: str = Path(..., description="疾病名称")
):
    """
    查询疾病相关药品。

    Args:
        disease: 疾病名称

    Returns:
        相关药品列表
    """
    neo4j_client = _get_neo4j_client()

    try:
        drugs = neo4j_client.get_related_drugs(disease)
        return {
            "disease": disease,
            "drugs": drugs,
            "count": len(drugs)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


@router.get("/kg/examinations/{disease}")
def get_disease_examinations(
    disease: str = Path(..., description="疾病名称")
):
    """
    查询疾病相关检查项目。

    Args:
        disease: 疾病名称

    Returns:
        相关检查项目列表
    """
    neo4j_client = _get_neo4j_client()

    try:
        examinations = neo4j_client.get_related_examinations(disease)
        return {
            "disease": disease,
            "examinations": examinations,
            "count": len(examinations)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


@router.get("/kg/department/{symptom}")
def get_symptom_department(
    symptom: str = Path(..., description="症状名称")
):
    """
    查询症状所属科室。

    Args:
        symptom: 症状名称

    Returns:
        所属科室
    """
    neo4j_client = _get_neo4j_client()

    try:
        department = neo4j_client.get_department(symptom)
        return {
            "symptom": symptom,
            "department": department,
            "source": "knowledge_graph"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


@router.post("/kg/diagnosis")
def find_diagnosis(
    symptoms: List[str] = Query(..., description="症状列表")
):
    """
    根据症状查找可能的疾病。

    Args:
        symptoms: 症状列表

    Returns:
        可能的疾病列表
    """
    neo4j_client = _get_neo4j_client()

    if not symptoms:
        raise HTTPException(status_code=400, detail="症状列表不能为空")

    if len(symptoms) < 2:
        raise HTTPException(status_code=400, detail="至少需要2个症状才能进行诊断推理")

    try:
        diagnosis_paths = neo4j_client.find_diagnosis_path(symptoms)
        return {
            "input_symptoms": symptoms,
            "possible_diagnoses": diagnosis_paths,
            "count": len(diagnosis_paths)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"诊断推理失败: {str(e)}")


@router.get("/kg/health")
def kg_health_check():
    """
    知识图谱健康检查。

    Returns:
        连接状态
    """
    neo4j_client = _get_neo4j_client()

    try:
        connected = neo4j_client.connect()
        return {
            "status": "healthy" if connected else "disconnected",
            "service": "neo4j",
            "uri": neo4j_client.uri
        }
    except Exception as e:
        return {
            "status": "error",
            "service": "neo4j",
            "error": str(e)
        }
