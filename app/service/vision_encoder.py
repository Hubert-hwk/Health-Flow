"""PDF/image parsing with coordinate-aware metric extraction."""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from app.model.llm import get_llm_client, get_vlm_client
from app.schema.report import MetricRecord


@dataclass
class ParsedReport:
    report_type: str
    raw_text: str
    metrics: List[MetricRecord]
    page_count: int
    success: bool
    error: Optional[str] = None


class VisionEncoderService:
    """Route text PDFs, scanned PDFs and images through the suitable parser."""

    def __init__(self) -> None:
        self._vlm_client = None
        self._llm_client = None

    @property
    def vlm_client(self):
        if self._vlm_client is None:
            self._vlm_client = get_vlm_client()
        return self._vlm_client

    @property
    def llm_client(self):
        if self._llm_client is None:
            self._llm_client = get_llm_client()
        return self._llm_client

    def detect_pdf_type(self, pdf_bytes: bytes) -> Tuple[str, int]:
        try:
            import pdfplumber

            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                page_count = len(pdf.pages)
                text_pages = sum(
                    1
                    for page in pdf.pages
                    if (page.extract_text() or "").strip()
                )
                return ("text_pdf" if text_pages >= 1 else "scanned_pdf", page_count)
        except ImportError:
            return "scanned_pdf", 0
        except Exception as exc:
            return "unknown", 0

    def parse_text_pdf(self, pdf_bytes: bytes) -> ParsedReport:
        try:
            import pdfplumber

            pages: list[str] = []
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                page_count = len(pdf.pages)
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    tables = page.extract_tables() or []
                    rows = [
                        " | ".join(str(cell or "") for cell in row)
                        for table in tables
                        for row in table
                        if row
                    ]
                    pages.append("\n".join(item for item in [text, *rows] if item))

            raw_text = "\n\n".join(page for page in pages if page)
            return ParsedReport(
                report_type="text_pdf",
                raw_text=raw_text,
                metrics=self._extract_metrics_from_text(raw_text),
                page_count=page_count,
                success=True,
            )
        except Exception as exc:
            return ParsedReport("text_pdf", "", [], 0, False, str(exc))

    def parse_scanned_pdf(self, pdf_bytes: bytes) -> ParsedReport:
        images = self._render_pdf_to_images(pdf_bytes)
        if not images:
            return ParsedReport("scanned_pdf", "", [], 0, False, "无法渲染 PDF 页面，请安装 PyMuPDF 和 Pillow")

        metrics: list[MetricRecord] = []
        texts: list[str] = []
        errors: list[str] = []
        for page_number, image_bytes in enumerate(images, start=1):
            parsed = self._parse_image_with_vlm(image_bytes, "image/png", page_number)
            texts.append(parsed[0])
            metrics.extend(parsed[1])
            if parsed[2]:
                errors.append(f"page {page_number}: {parsed[2]}")

        return ParsedReport(
            report_type="scanned_pdf",
            raw_text="\n\n".join(texts),
            metrics=metrics,
            page_count=len(images),
            success=bool(texts or metrics) and not errors,
            error="; ".join(errors) if errors else None,
        )

    def parse_image_report(self, image_bytes: bytes, mime_type: str = "image/png") -> ParsedReport:
        text, metrics, error = self._parse_image_with_vlm(image_bytes, mime_type, 1)
        return ParsedReport("image", text, metrics, 1, error is None, error)

    def parse(self, content: bytes, filename: str) -> ParsedReport:
        lower = filename.lower()
        if lower.endswith(".pdf"):
            pdf_type, _ = self.detect_pdf_type(content)
            return self.parse_text_pdf(content) if pdf_type == "text_pdf" else self.parse_scanned_pdf(content)
        if lower.endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp")):
            return self.parse_image_report(content, self._get_mime_type(lower))
        return ParsedReport("unknown", "", [], 0, False, f"不支持的文件类型：{filename}")

    def _parse_image_with_vlm(
        self, image_bytes: bytes, mime_type: str, page_number: int
    ) -> tuple[str, list[MetricRecord], Optional[str]]:
        image_base64 = base64.b64encode(image_bytes).decode("ascii")
        width, height = self._image_size(image_bytes)
        prompt = """
解析体检报告页面，输出严格 JSON，不要输出 Markdown。必须尽量保留版面证据。
每个 metric 必须包含 metric_name、metric_value；如果能定位，请返回页面像素坐标 bbox
[x1,y1,x2,y2]、归一化坐标 bbox_normalized [0,0,1000,1000]、evidence_text。
JSON 格式：
{"text_summary":"页面摘要","metrics":[
 {"metric_name":"空腹血糖","metric_value":"6.5","unit":"mmol/L",
  "reference_range":"3.9-6.1","abnormal_flag":"H","bbox":[0,0,0,0],
  "bbox_normalized":[0,0,0,0],"evidence_text":"原文片段"}
]}
无法确认的坐标返回 null，禁止猜测坐标。
""".strip()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_base64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        try:
            response = self.vlm_client.chat_with_image(messages)
            payload = self._parse_json_response(response)
            metrics = [
                self._metric_from_payload(item, page_number, width, height, index)
                for index, item in enumerate(payload.get("metrics", []), start=1)
            ]
            return str(payload.get("text_summary", "")), [item for item in metrics if item], None
        except Exception as exc:
            return "", [], str(exc)

    def _metric_from_payload(
        self,
        data: dict[str, Any],
        page_number: int,
        width: Optional[int],
        height: Optional[int],
        index: int,
    ) -> Optional[MetricRecord]:
        name = str(data.get("metric_name", "")).strip()
        value = str(data.get("metric_value", "")).strip()
        if not name or not value:
            return None

        bbox = self._clean_bbox(data.get("bbox"))
        normalized = self._clean_bbox(data.get("bbox_normalized"))
        if normalized is None and bbox and width and height:
            normalized = self.normalize_bbox(bbox, width, height)
        if bbox is None and normalized and width and height:
            bbox = self.denormalize_bbox(normalized, width, height)

        return MetricRecord(
            metric_name=name,
            metric_value=value,
            unit=data.get("unit"),
            reference_range=data.get("reference_range"),
            trend=data.get("trend"),
            abnormal_flag=data.get("abnormal_flag"),
            bbox=bbox,
            bbox_normalized=normalized,
            page_number=page_number,
            evidence_text=data.get("evidence_text"),
            source_id=str(data.get("source_id") or f"p{page_number}-m{index}"),
        )

    @staticmethod
    def _clean_bbox(value: Any) -> Optional[list[float]]:
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            return None
        try:
            values = [float(item) for item in value]
        except (TypeError, ValueError):
            return None
        x1, y1, x2, y2 = values
        if x2 < x1 or y2 < y1:
            return None
        return values

    @staticmethod
    def normalize_bbox(bbox: list[float], width: int, height: int) -> list[float]:
        if width <= 0 or height <= 0:
            return [0.0, 0.0, 0.0, 0.0]
        x1, y1, x2, y2 = bbox
        return [
            round(max(0.0, min(1000.0, x1 / width * 1000)), 2),
            round(max(0.0, min(1000.0, y1 / height * 1000)), 2),
            round(max(0.0, min(1000.0, x2 / width * 1000)), 2),
            round(max(0.0, min(1000.0, y2 / height * 1000)), 2),
        ]

    @staticmethod
    def denormalize_bbox(bbox: list[float], width: int, height: int) -> list[float]:
        x1, y1, x2, y2 = bbox
        return [x1 / 1000 * width, y1 / 1000 * height, x2 / 1000 * width, y2 / 1000 * height]

    @staticmethod
    def _parse_json_response(response: Any) -> dict[str, Any]:
        if isinstance(response, dict):
            return response
        text = str(response or "").strip().replace("```json", "").replace("```", "")
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("VLM 未返回 JSON")
        payload = json.loads(text[start : end + 1])
        if not isinstance(payload, dict):
            raise ValueError("VLM JSON 顶层不是对象")
        return payload

    @staticmethod
    def _image_size(image_bytes: bytes) -> tuple[Optional[int], Optional[int]]:
        try:
            from PIL import Image

            with Image.open(io.BytesIO(image_bytes)) as image:
                return image.width, image.height
        except Exception:
            return None, None

    def _render_pdf_to_images(self, pdf_bytes: bytes, dpi: int = 144) -> List[bytes]:
        try:
            import fitz

            document = fitz.open(stream=pdf_bytes, filetype="pdf")
            matrix = fitz.Matrix(dpi / 72, dpi / 72)
            images = [page.get_pixmap(matrix=matrix).tobytes("png") for page in document]
            document.close()
            return images
        except Exception:
            return []

    @staticmethod
    def _get_mime_type(filename: str) -> str:
        return {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
        }.get(next((ext for ext in (".jpg", ".jpeg", ".png", ".gif", ".bmp") if filename.endswith(ext)), ".png"), "image/png")

    def _extract_metrics_from_text(self, text: str) -> List[MetricRecord]:
        if not text.strip():
            return []
        prompt = (
            "从体检报告文本中提取医学指标，只输出 JSON："
            '{"metrics":[{"metric_name":"","metric_value":"","unit":"",'
            '"reference_range":"","abnormal_flag":"","evidence_text":""}]}\n'
            f"文本：{text[:12000]}"
        )
        try:
            result = self.llm_client.chat_with_json(
                messages=[
                    {"role": "system", "content": "你是医疗报告结构化抽取器，不做诊断。"},
                    {"role": "user", "content": prompt},
                ],
                json_schema={"type": "object"},
            )
            values = result.get("metrics", []) if isinstance(result, dict) else []
            records: list[MetricRecord] = []
            for index, item in enumerate(values, start=1):
                metric = self._metric_from_payload(item, 1, None, None, index)
                if metric:
                    records.append(metric)
            return records
        except Exception:
            return []


_vision_encoder_service: VisionEncoderService | None = None


def get_vision_encoder_service() -> VisionEncoderService:
    global _vision_encoder_service
    if _vision_encoder_service is None:
        _vision_encoder_service = VisionEncoderService()
    return _vision_encoder_service
