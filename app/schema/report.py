"""Schemas for report parsing and coordinate-aware metric extraction."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class MetricRecord(BaseModel):
    metric_name: str = Field(..., min_length=1)
    metric_value: str
    unit: Optional[str] = None
    reference_range: Optional[str] = None
    trend: Optional[str] = None
    abnormal_flag: Optional[str] = None
    bbox: Optional[List[float]] = Field(None, min_length=4, max_length=4)
    bbox_normalized: Optional[List[float]] = Field(None, min_length=4, max_length=4)
    page_number: Optional[int] = Field(None, ge=1)
    evidence_text: Optional[str] = None
    source_id: Optional[str] = None


class MedicalReportCreate(BaseModel):
    patient_id: str
    report_type: Optional[str] = None
    file_url: Optional[str] = None
    parsed_content: Optional[dict] = None
    exam_date: Optional[datetime] = None
    department: Optional[str] = None
    metrics: List[MetricRecord] = Field(default_factory=list)


class MedicalReport(MedicalReportCreate):
    id: int
    created_at: datetime = Field(default_factory=datetime.now)

    model_config = {"from_attributes": True}

class MedicalReportResponse(BaseModel):
    id: int
    patient_id: str
    report_type: Optional[str]
    exam_date: Optional[datetime]
    department: Optional[str]
    metrics: List[MetricRecord]
    created_at: datetime

    model_config = {"from_attributes": True}
