"""Metric API endpoints.

提供指标查询、趋势分析等接口。
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import json

from app.data.mysql_client import get_mysql_client
from app.data.models import MedicalReport as ReportModel, MetricRecord as MetricModel
from app.schema.report import MetricRecord


router = APIRouter()


@router.get("/metric/trend")
def get_metric_trend(
    patient_id: str,
    metric_name: str = Query(..., description="指标名称，如空腹血糖"),
    days: int = Query(90, ge=1, description="查询天数范围")
):
    """
    获取指标趋势分析。

    Args:
        patient_id: 患者ID
        metric_name: 指标名称
        days: 查询天数范围

    Returns:
        指标趋势数据
    """
    # 获取数据库连接
    mysql_client = get_mysql_client()

    with mysql_client.get_session() as db:
        # 计算日期范围
        start_date = datetime.now() - timedelta(days=days)

        # 查询该患者的报告
        reports = db.query(ReportModel).filter(
            ReportModel.patient_id == patient_id,
            ReportModel.created_at >= start_date
        ).order_by(ReportModel.created_at).all()

        if not reports:
            return {
                "patient_id": patient_id,
                "metric_name": metric_name,
                "data_points": [],
                "summary": "暂无数据"
            }

        report_ids = [r.id for r in reports]

        # 查询该指标
        metrics = db.query(MetricModel).filter(
            MetricModel.report_id.in_(report_ids),
            MetricModel.metric_name == metric_name
        ).all()

        # 构建趋势数据
        metric_by_report = {m.report_id: m for m in metrics}
        data_points = []

        for report in reports:
            if report.id in metric_by_report:
                m = metric_by_report[report.id]
                # 解析参考范围
                reference_range = m.reference_range
                is_abnormal = False
                if reference_range and m.metric_value:
                    try:
                        # 简单判断是否超出参考范围
                        ref_values = reference_range.replace(" ", "").split("-")
                        if len(ref_values) == 2:
                            low = float(ref_values[0])
                            high = float(ref_values[1])
                            value = float(m.metric_value)
                            is_abnormal = value < low or value > high
                    except (ValueError, AttributeError):
                        pass

                data_points.append({
                    "report_id": report.id,
                    "exam_date": report.exam_date.isoformat() if report.exam_date else report.created_at.isoformat(),
                    "value": m.metric_value,
                    "unit": m.unit,
                    "reference_range": m.reference_range,
                    "trend": m.trend,
                    "abnormal_flag": m.abnormal_flag or ("H" if is_abnormal else "N")
                })

        # 计算统计信息
        values = []
        for dp in data_points:
            try:
                values.append(float(dp["value"]))
            except (ValueError, TypeError):
                pass

        if values:
            avg_value = sum(values) / len(values)
            min_value = min(values)
            max_value = max(values)

            # 判断趋势
            if len(values) >= 2:
                if values[-1] > values[0] * 1.1:
                    overall_trend = "↑ 上升"
                elif values[-1] < values[0] * 0.9:
                    overall_trend = "↓ 下降"
                else:
                    overall_trend = "→ 稳定"
            else:
                overall_trend = "→ 稳定"
        else:
            avg_value = min_value = max_value = None
            overall_trend = "未知"

        return {
            "patient_id": patient_id,
            "metric_name": metric_name,
            "data_points": data_points,
            "statistics": {
                "count": len(values),
                "average": round(avg_value, 2) if avg_value is not None else None,
                "min": round(min_value, 2) if min_value is not None else None,
                "max": round(max_value, 2) if max_value is not None else None,
                "overall_trend": overall_trend
            }
        }


@router.get("/metric/search")
def search_metrics(
    patient_id: str,
    keyword: Optional[str] = None,
    department: Optional[str] = None,
    abnormal_only: bool = False,
    limit: int = Query(50, ge=1, le=200)
):
    """
    搜索指标记录。

    Args:
        patient_id: 患者ID
        keyword: 搜索关键词（指标名称）
        department: 科室过滤
        abnormal_only: 只返回异常指标
        limit: 返回数量

    Returns:
        指标列表
    """
    mysql_client = get_mysql_client()

    with mysql_client.get_session() as db:
        # 构建查询
        query = db.query(ReportModel).filter(ReportModel.patient_id == patient_id)

        if department:
            query = query.filter(ReportModel.department == department)

        reports = query.order_by(ReportModel.created_at.desc()).limit(limit * 2).all()

        if not reports:
            return {"metrics": []}

        report_ids = [r.id for r in reports]

        # 查询指标
        metric_query = db.query(MetricModel).filter(MetricModel.report_id.in_(report_ids))

        if keyword:
            metric_query = metric_query.filter(MetricModel.metric_name.contains(keyword))

        if abnormal_only:
            metric_query = metric_query.filter(MetricModel.abnormal_flag.in_(["H", "L"]))

        metrics = metric_query.limit(limit).all()

        # 构建报告ID到日期的映射
        report_dates = {r.id: (r.exam_date or r.created_at) for r in reports}

        result = []
        for m in metrics:
            # 判断是否异常
            is_abnormal = m.abnormal_flag in ["H", "L"]
            if not is_abnormal and m.reference_range and m.metric_value:
                try:
                    ref_values = m.reference_range.replace(" ", "").split("-")
                    if len(ref_values) == 2:
                        low = float(ref_values[0])
                        high = float(ref_values[1])
                        value = float(m.metric_value)
                        is_abnormal = value < low or value > high
                except (ValueError, AttributeError):
                    pass

            result.append({
                "metric_name": m.metric_name,
                "metric_value": m.metric_value,
                "unit": m.unit,
                "reference_range": m.reference_range,
                "trend": m.trend,
                "abnormal_flag": m.abnormal_flag or ("H" if is_abnormal else "N"),
                "report_id": m.report_id,
                "exam_date": report_dates.get(m.report_id).isoformat() if report_dates.get(m.report_id) else None
            })

        return {"metrics": result}


@router.get("/metric/anomalies")
def get_anomalies(
    patient_id: str,
    days: int = Query(30, ge=1, description="查询天数范围")
):
    """
    获取患者异常指标汇总。

    Args:
        patient_id: 患者ID
        days: 查询天数范围

    Returns:
        异常指标列表
    """
    mysql_client = get_mysql_client()

    with mysql_client.get_session() as db:
        start_date = datetime.now() - timedelta(days=days)

        # 查询该患者最近的报告
        reports = db.query(ReportModel).filter(
            ReportModel.patient_id == patient_id,
            ReportModel.created_at >= start_date
        ).all()

        if not reports:
            return {"anomalies": [], "summary": "暂无数据"}

        report_ids = [r.id for r in reports]

        # 查询所有指标
        metrics = db.query(MetricModel).filter(
            MetricModel.report_id.in_(report_ids)
        ).all()

        # 筛选异常指标
        anomalies = []
        for m in metrics:
            is_abnormal = m.abnormal_flag in ["H", "L"]

            # 如果标记了异常
            if is_abnormal:
                anomalies.append({
                    "metric_name": m.metric_name,
                    "metric_value": m.metric_value,
                    "unit": m.unit,
                    "reference_range": m.reference_range,
                    "abnormal_flag": m.abnormal_flag,
                    "report_id": m.report_id,
                    "severity": "高" if m.abnormal_flag == "H" else "低"
                })
                continue

            # 自动判断异常
            if m.reference_range and m.metric_value:
                try:
                    ref_values = m.reference_range.replace(" ", "").split("-")
                    if len(ref_values) == 2:
                        low = float(ref_values[0])
                        high = float(ref_values[1])
                        value = float(m.metric_value)

                        if value < low:
                            anomalies.append({
                                "metric_name": m.metric_name,
                                "metric_value": m.metric_value,
                                "unit": m.unit,
                                "reference_range": m.reference_range,
                                "abnormal_flag": "L",
                                "report_id": m.report_id,
                                "severity": "中",
                                "note": f"低于参考范围下限{low}"
                            })
                        elif value > high:
                            anomalies.append({
                                "metric_name": m.metric_name,
                                "metric_value": m.metric_value,
                                "unit": m.unit,
                                "reference_range": m.reference_range,
                                "abnormal_flag": "H",
                                "report_id": m.report_id,
                                "severity": "高",
                                "note": f"高于参考范围上限{high}"
                            })
                except (ValueError, AttributeError):
                    pass

        # 按严重程度排序
        severity_order = {"高": 0, "中": 1, "低": 2}
        anomalies.sort(key=lambda x: severity_order.get(x.get("severity", "低"), 2))

        # 按指标分组
        metric_groups = {}
        for a in anomalies:
            name = a["metric_name"]
            if name not in metric_groups:
                metric_groups[name] = []
            metric_groups[name].append(a)

        return {
            "anomalies": anomalies,
            "summary": f"发现{len(anomalies)}项异常指标，{len(metric_groups)}项不同指标"
        }
