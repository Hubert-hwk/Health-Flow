"""Report API endpoints.

提供体检报告上传、详情查询、指标列表等接口。
"""

from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
import json

from app.data import get_db
from app.data.models import MedicalReport as ReportModel, MetricRecord as MetricModel
from app.schema.report import (
    MedicalReportCreate,
    MedicalReportResponse,
    MetricRecord,
)
from app.service.vision_encoder import get_vision_encoder_service
from app.model.embedding import get_embedding_client
from app.data.milvus_client import get_milvus_client


router = APIRouter()


@router.post("/report/upload", response_model=MedicalReportResponse)
async def upload_report(
    patient_id: str,
    report_type: Optional[str] = None,
    department: Optional[str] = None,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    上传体检报告。

    支持PDF和图片文件，自动检测类型并解析。

    Args:
        patient_id: 患者ID
        report_type: 报告类型（体检/化验/影像）
        department: 科室
        file: 上传的文件（PDF/图片）
        db: 数据库会话

    Returns:
        解析后的报告信息
    """
    # 读取文件内容
    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="文件内容为空")

    filename = file.filename or "unknown"

    # 解析报告
    vision_service = get_vision_encoder_service()
    parsed = vision_service.parse(content, filename)

    if not parsed.success:
        raise HTTPException(
            status_code=500,
            detail=f"报告解析失败: {parsed.error}"
        )

    # 创建报告记录
    report = ReportModel(
        patient_id=patient_id,
        report_type=report_type or "体检",
        department=department,
        parsed_content=json.dumps({
            "report_type": parsed.report_type,
            "raw_text": parsed.raw_text,
            "page_count": parsed.page_count
        }, ensure_ascii=False),
        exam_date=datetime.now()
    )
    db.add(report)
    db.commit()
    db.refresh(report)

    # 保存指标记录
    for metric_data in parsed.metrics:
        metric = MetricModel(
            report_id=report.id,
            metric_name=metric_data.metric_name,
            metric_value=metric_data.metric_value,
            unit=metric_data.unit,
            reference_range=metric_data.reference_range,
            trend=metric_data.trend,
            abnormal_flag=metric_data.abnormal_flag,
            bbox=json.dumps(metric_data.bbox) if metric_data.bbox else None
        )
        db.add(metric)

    db.commit()

    # 向量索引（异步处理会更好，这里同步处理）
    try:
        embedding_client = get_embedding_client()
        milvus_client = get_milvus_client()

        # 准备向量数据
        texts = [parsed.raw_text]
        embeddings = [embedding_client.embed(parsed.raw_text)]

        # 插入Milvus
        milvus_client.insert(
            report_ids=[report.id],
            texts=texts,
            embeddings=embeddings,
            departments=[department] if department else None
        )
        milvus_client.flush()
    except Exception as e:
        # 向量索引失败不影响主流程
        print(f"向量索引失败: {e}")

    # 返回响应
    return MedicalReportResponse(
        id=report.id,
        patient_id=report.patient_id,
        report_type=report.report_type,
        exam_date=report.exam_date,
        department=report.department,
        metrics=parsed.metrics,
        created_at=report.created_at
    )


@router.get("/report/{report_id}", response_model=MedicalReportResponse)
async def get_report(
    report_id: int,
    db: Session = Depends(get_db)
):
    """
    获取报告详情。

    Args:
        report_id: 报告ID
        db: 数据库会话

    Returns:
        报告详情
    """
    report = db.query(ReportModel).filter(ReportModel.id == report_id).first()

    if not report:
        raise HTTPException(status_code=404, detail="报告不存在")

    # 获取指标
    metrics = db.query(MetricModel).filter(MetricModel.report_id == report_id).all()

    metric_records = [
        MetricRecord(
            metric_name=m.metric_name,
            metric_value=m.metric_value,
            unit=m.unit,
            reference_range=m.reference_range,
            trend=m.trend,
            abnormal_flag=m.abnormal_flag,
            bbox=json.loads(m.bbox) if m.bbox else None
        )
        for m in metrics
    ]

    return MedicalReportResponse(
        id=report.id,
        patient_id=report.patient_id,
        report_type=report.report_type,
        exam_date=report.exam_date,
        department=report.department,
        metrics=metric_records,
        created_at=report.created_at
    )


@router.get("/report/{report_id}/metrics", response_model=List[MetricRecord])
async def get_report_metrics(
    report_id: int,
    db: Session = Depends(get_db)
):
    """
    获取报告的指标列表。

    Args:
        report_id: 报告ID
        db: 数据库会话

    Returns:
        指标列表
    """
    # 检查报告是否存在
    report = db.query(ReportModel).filter(ReportModel.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="报告不存在")

    # 获取指标
    metrics = db.query(MetricModel).filter(MetricModel.report_id == report_id).all()

    return [
        MetricRecord(
            metric_name=m.metric_name,
            metric_value=m.metric_value,
            unit=m.unit,
            reference_range=m.reference_range,
            trend=m.trend,
            abnormal_flag=m.abnormal_flag,
            bbox=json.loads(m.bbox) if m.bbox else None
        )
        for m in metrics
    ]


@router.get("/reports", response_model=List[MedicalReportResponse])
async def list_reports(
    patient_id: Optional[str] = None,
    department: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """
    获取报告列表。

    Args:
        patient_id: 患者ID过滤
        department: 科室过滤
        limit: 返回数量
        offset: 偏移量
        db: 数据库会话

    Returns:
        报告列表
    """
    query = db.query(ReportModel)

    if patient_id:
        query = query.filter(ReportModel.patient_id == patient_id)

    if department:
        query = query.filter(ReportModel.department == department)

    reports = query.order_by(ReportModel.created_at.desc()).offset(offset).limit(limit).all()

    result = []
    for report in reports:
        metrics = db.query(MetricModel).filter(MetricModel.report_id == report.id).all()

        metric_records = [
            MetricRecord(
                metric_name=m.metric_name,
                metric_value=m.metric_value,
                unit=m.unit,
                reference_range=m.reference_range,
                trend=m.trend,
                abnormal_flag=m.abnormal_flag,
                bbox=json.loads(m.bbox) if m.bbox else None
            )
            for m in metrics
        ]

        result.append(MedicalReportResponse(
            id=report.id,
            patient_id=report.patient_id,
            report_type=report.report_type,
            exam_date=report.exam_date,
            department=report.department,
            metrics=metric_records,
            created_at=report.created_at
        ))

    return result


@router.delete("/report/{report_id}")
async def delete_report(
    report_id: int,
    db: Session = Depends(get_db)
):
    """
    删除报告。

    Args:
        report_id: 报告ID
        db: 数据库会话

    Returns:
        删除结果
    """
    report = db.query(ReportModel).filter(ReportModel.id == report_id).first()

    if not report:
        raise HTTPException(status_code=404, detail="报告不存在")

    # 删除Milvus中的向量
    try:
        milvus_client = get_milvus_client()
        milvus_client.delete_by_report_id(report_id)
    except Exception:
        pass  # 向量删除失败不影响主流程

    # 删除指标
    db.query(MetricModel).filter(MetricModel.report_id == report_id).delete()

    # 删除报告
    db.delete(report)
    db.commit()

    return {"message": "报告已删除", "report_id": report_id}
