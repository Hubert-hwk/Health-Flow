"""Report related schemas."""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class MetricRecord(BaseModel):
    """Medical metric record."""

    metric_name: str = Field(..., description="指标名称，如空腹血糖")
    metric_value: str = Field(..., description="指标值")
    unit: Optional[str] = Field(None, description="单位，如mmol/L")
    reference_range: Optional[str] = Field(None, description="参考范围，如3.9-6.1")
    trend: Optional[str] = Field(None, description="趋势，↑/↓/-")
    abnormal_flag: Optional[str] = Field(None, description="异常标记，H/L/N")
    bbox: Optional[List[float]] = Field(None, description="坐标BBOX [x1,y1,x2,y2]")


class MedicalReportCreate(BaseModel):
    """Schema for creating a medical report."""

    patient_id: str = Field(..., description="患者ID")
    report_type: Optional[str] = Field(None, description="报告类型")
    file_url: Optional[str] = Field(None, description="文件URL")
    parsed_content: Optional[dict] = Field(None, description="解析后的内容")
    exam_date: Optional[datetime] = Field(None, description="检查日期")
    department: Optional[str] = Field(None, description="科室")
    metrics: Optional[List[MetricRecord]] = Field(default_factory=list, description="指标列表")


class MedicalReport(MedicalReportCreate):
    """Schema for medical report with ID."""

    id: int = Field(..., description="报告ID")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")

    model_config = {"from_attributes": True}


class MedicalReportResponse(BaseModel):
    """Schema for medical report API response."""

    id: int
    patient_id: str
    report_type: Optional[str]
    exam_date: Optional[datetime]
    department: Optional[str]
    metrics: List[MetricRecord]
    created_at: datetime

    model_config = {"from_attributes": True}
