"""Report upload, parsing and metric endpoints."""

from __future__ import annotations

import json
from datetime import datetime
from types import GeneratorType
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.config import get_settings
from app.data import get_db
from app.data.milvus_client import get_milvus_client
from app.data.models import MedicalReport as ReportModel, MetricRecord as MetricModel
from app.model.embedding import get_embedding_client
from app.schema.report import MedicalReportResponse, MetricRecord
from app.service.vision_encoder import get_vision_encoder_service


router = APIRouter()
ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".gif", ".bmp"}


def db_dependency():
    from app.data import get_db as current_get_db

    value = current_get_db()
    if isinstance(value, GeneratorType):
        yield from value
    else:
        yield value


def _metric_response(metric: MetricModel) -> MetricRecord:
    def load_json(value):
        if value is None or isinstance(value, (list, dict)):
            return value
        try:
            return json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return None

    return MetricRecord(
        metric_name=metric.metric_name or "未命名指标",
        metric_value=metric.metric_value or "",
        unit=metric.unit,
        reference_range=metric.reference_range,
        trend=metric.trend,
        abnormal_flag=metric.abnormal_flag,
        bbox=load_json(metric.bbox),
        bbox_normalized=load_json(metric.bbox_normalized),
        page_number=metric.page_number,
        evidence_text=metric.evidence_text,
        source_id=metric.source_id,
    )


@router.post("/report/upload", response_model=MedicalReportResponse)
async def upload_report(
    patient_id: str,
    report_type: Optional[str] = None,
    department: Optional[str] = None,
    file: UploadFile = File(...),
    db: Session = Depends(db_dependency),
):
    filename = file.filename or "unknown"
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="仅支持 PDF 或常见图片格式")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="文件内容为空")
    if len(content) > get_settings().MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="文件超过大小限制")

    parsed = get_vision_encoder_service().parse(content, filename)
    if not parsed.success:
        raise HTTPException(status_code=422, detail=f"报告解析失败：{parsed.error}")

    report = ReportModel(
        patient_id=patient_id,
        report_type=report_type or "体检",
        department=department,
        parsed_content={
            "report_type": parsed.report_type,
            "raw_text": parsed.raw_text,
            "page_count": parsed.page_count,
            "metric_count": len(parsed.metrics),
        },
        exam_date=datetime.now(),
    )
    db.add(report)
    db.flush()
    if report.id is None:
        raise HTTPException(status_code=500, detail="报告写入数据库失败")

    for item in parsed.metrics:
        db.add(
            MetricModel(
                report_id=report.id,
                metric_name=item.metric_name,
                metric_value=item.metric_value,
                unit=item.unit,
                reference_range=item.reference_range,
                trend=item.trend,
                abnormal_flag=item.abnormal_flag,
                bbox=item.bbox,
                bbox_normalized=item.bbox_normalized,
                page_number=item.page_number,
                evidence_text=item.evidence_text,
                source_id=item.source_id,
            )
        )
    db.commit()
    db.refresh(report)

    # Indexing is best-effort because Milvus is an optional service.  The SQL
    # report remains the source of truth when the vector service is offline.
    if parsed.raw_text.strip():
        try:
            vector = get_embedding_client().embed(parsed.raw_text)
            client = get_milvus_client()
            client.insert(
                report_ids=[report.id],
                texts=[parsed.raw_text],
                embeddings=[vector],
                departments=[department] if department else None,
            )
            client.flush()
        except Exception:
            pass

    return MedicalReportResponse(
        id=report.id,
        patient_id=report.patient_id,
        report_type=report.report_type,
        exam_date=report.exam_date,
        department=report.department,
        metrics=parsed.metrics,
        created_at=report.created_at,
    )


def _report_response(report: ReportModel, metrics: list[MetricModel]) -> MedicalReportResponse:
    return MedicalReportResponse(
        id=report.id,
        patient_id=report.patient_id,
        report_type=report.report_type,
        exam_date=report.exam_date,
        department=report.department,
        metrics=[_metric_response(metric) for metric in metrics],
        created_at=report.created_at,
    )


@router.get("/report/{report_id}", response_model=MedicalReportResponse)
async def get_report(report_id: int, db: Session = Depends(db_dependency)):
    report = db.query(ReportModel).filter(ReportModel.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="报告不存在")
    metrics = db.query(MetricModel).filter(MetricModel.report_id == report_id).all()
    return _report_response(report, metrics)


@router.get("/report/{report_id}/metrics", response_model=list[MetricRecord])
async def get_report_metrics(report_id: int, db: Session = Depends(db_dependency)):
    if not db.query(ReportModel).filter(ReportModel.id == report_id).first():
        raise HTTPException(status_code=404, detail="报告不存在")
    return [
        _metric_response(metric)
        for metric in db.query(MetricModel).filter(MetricModel.report_id == report_id).all()
    ]


@router.get("/reports", response_model=list[MedicalReportResponse])
async def list_reports(
    patient_id: Optional[str] = None,
    department: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    db: Session = Depends(db_dependency),
):
    limit = max(1, min(limit, 100))
    query = db.query(ReportModel)
    if patient_id:
        query = query.filter(ReportModel.patient_id == patient_id)
    if department:
        query = query.filter(ReportModel.department == department)
    reports = query.order_by(ReportModel.created_at.desc()).offset(max(0, offset)).limit(limit).all()
    return [
        _report_response(
            report,
            db.query(MetricModel).filter(MetricModel.report_id == report.id).all(),
        )
        for report in reports
    ]


@router.delete("/report/{report_id}")
async def delete_report(report_id: int, db: Session = Depends(db_dependency)):
    report = db.query(ReportModel).filter(ReportModel.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="报告不存在")
    try:
        get_milvus_client().delete_by_report_id(report_id)
    except Exception:
        pass
    db.delete(report)
    db.commit()
    return {"message": "报告已删除", "report_id": report_id}
