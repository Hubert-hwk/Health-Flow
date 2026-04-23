"""VisionEncoder Service - PDF/图像解析服务.

支持文本型PDF、扫描件PDF、以及纯图像报告的解析。
"""

import io
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from app.model.llm import get_vlm_client, get_llm_client
from app.schema.report import MetricRecord


@dataclass
class ParsedReport:
    """解析后的报告。"""

    report_type: str  # "text_pdf", "scanned_pdf", "image"
    raw_text: str
    metrics: List[MetricRecord]
    page_count: int
    success: bool
    error: Optional[str] = None


class VisionEncoderService:
    """
    视觉编码服务。

    核心职责:
    1. PDF类型检测（文本型 vs 扫描件）
    2. 文本型PDF快速提取
    3. 扫描件PDF渲染 + VLM解析
    4. 图像报告VLM解析
    """

    def __init__(self):
        self._vlm_client = None
        self._llm_client = None

    @property
    def vlm_client(self):
        """获取VLM客户端。"""
        if self._vlm_client is None:
            self._vlm_client = get_vlm_client()
        return self._vlm_client

    @property
    def llm_client(self):
        """获取LLM客户端。"""
        if self._llm_client is None:
            self._llm_client = get_llm_client()
        return self._llm_client

    def detect_pdf_type(self, pdf_bytes: bytes) -> Tuple[str, int]:
        """
        检测PDF类型。

        Args:
            pdf_bytes: PDF文件字节

        Returns:
            (pdf类型, 页数)
            类型: "text_pdf" 或 "scanned_pdf"
        """
        try:
            import pdfplumber

            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                page_count = len(pdf.pages)

                # 检查前3页是否包含文本
                text_pages = 0
                for page in pdf.pages[:3]:
                    text = page.extract_text()
                    if text and len(text.strip()) > 50:
                        text_pages += 1

                # 如果超过一半的页面有文本，判定为文本型PDF
                if text_pages >= 2:
                    return "text_pdf", page_count
                else:
                    return "scanned_pdf", page_count

        except ImportError:
            # pdfplumber不可用，假设为扫描件
            return "scanned_pdf", 1
        except Exception:
            return "unknown", 0

    def parse_text_pdf(self, pdf_bytes: bytes) -> ParsedReport:
        """
        解析文本型PDF。

        Args:
            pdf_bytes: PDF文件字节

        Returns:
            解析结果
        """
        try:
            import pdfplumber

            all_text = []
            all_metrics = []

            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for page in pdf.pages:
                    # 提取文本
                    text = page.extract_text()
                    if text:
                        all_text.append(text)

                    # 提取表格
                    tables = page.extract_tables()
                    for table in tables:
                        for row in table:
                            if row:
                                row_text = " | ".join([str(cell) if cell else "" for cell in row])
                                all_text.append(row_text)

            # 合并所有文本
            raw_text = "\n\n".join(all_text)

            # 提取指标
            metrics = self._extract_metrics_from_text(raw_text)

            return ParsedReport(
                report_type="text_pdf",
                raw_text=raw_text,
                metrics=metrics,
                page_count=len(pdf.pages) if 'pdf' in locals() else 1,
                success=True
            )

        except Exception as e:
            return ParsedReport(
                report_type="text_pdf",
                raw_text="",
                metrics=[],
                page_count=0,
                success=False,
                error=str(e)
            )

    def parse_scanned_pdf(self, pdf_bytes: bytes) -> ParsedReport:
        """
        解析扫描件PDF。

        将PDF页面渲染为图像，然后使用VLM解析。

        Args:
            pdf_bytes: PDF文件字节

        Returns:
            解析结果
        """
        try:
            from PIL import Image
            import base64

            # 渲染PDF页面
            images = self._render_pdf_to_images(pdf_bytes)

            if not images:
                return ParsedReport(
                    report_type="scanned_pdf",
                    raw_text="",
                    metrics=[],
                    page_count=0,
                    success=False,
                    error="无法渲染PDF页面"
                )

            # 使用VLM解析每页
            all_metrics = []
            all_text_parts = []

            for i, image_bytes in enumerate(images):
                # 将图像转换为base64
                image_base64 = base64.b64encode(image_bytes).decode("utf-8")

                # 构建VLM消息
                messages = [{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                        },
                        {
                            "type": "text",
                            "text": """请解析这张体检报告图像，提取所有医学指标。

请以JSON格式输出:
{
    "text_summary": "页面内容的文字总结",
    "metrics": [
        {"metric_name": "指标名", "metric_value": "指标值", "unit": "单位", "reference_range": "参考范围"}
    ]
}

只输出JSON，不要其他内容。"""
                        }
                    ]
                }]

                try:
                    response = self.vlm_client.chat_with_image(messages)

                    # 解析VLM响应
                    import json
                    if isinstance(response, str):
                        # 尝试提取JSON
                        start = response.find("{")
                        end = response.rfind("}") + 1
                        if start != -1 and end != 0:
                            parsed = json.loads(response[start:end])
                            all_text_parts.append(parsed.get("text_summary", ""))

                            for m in parsed.get("metrics", []):
                                all_metrics.append(MetricRecord(
                                    metric_name=m.get("metric_name", ""),
                                    metric_value=m.get("metric_value", ""),
                                    unit=m.get("unit"),
                                    reference_range=m.get("reference_range")
                                ))
                except Exception as e:
                    all_text_parts.append(f"[第{i+1}页解析失败: {str(e)}]")

            raw_text = "\n\n".join(all_text_parts)

            return ParsedReport(
                report_type="scanned_pdf",
                raw_text=raw_text,
                metrics=all_metrics,
                page_count=len(images),
                success=True
            )

        except Exception as e:
            return ParsedReport(
                report_type="scanned_pdf",
                raw_text="",
                metrics=[],
                page_count=0,
                success=False,
                error=str(e)
            )

    def parse_image_report(self, image_bytes: bytes, mime_type: str = "image/png") -> ParsedReport:
        """
        解析图像型报告（直接上传的图片）。

        Args:
            image_bytes: 图像字节
            mime_type: MIME类型

        Returns:
            解析结果
        """
        try:
            import base64
            import json

            # 将图像转换为base64
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")

            # 确定data URL前缀
            if mime_type == "image/jpeg":
                data_prefix = "data:image/jpeg;base64,"
            elif mime_type == "image/png":
                data_prefix = "data:image/png;base64,"
            else:
                data_prefix = "data:image/png;base64,"

            # 构建VLM消息
            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"{data_prefix}{image_base64}"}
                    },
                    {
                        "type": "text",
                        "text": """请解析这张体检报告图像，提取所有医学指标。

请以JSON格式输出:
{
    "text_summary": "图像内容的文字总结",
    "metrics": [
        {"metric_name": "指标名", "metric_value": "指标值", "unit": "单位", "reference_range": "参考范围"}
    ]
}

只输出JSON，不要其他内容。"""
                    }
                ]
            }]

            response = self.vlm_client.chat_with_image(messages)

            # 解析响应
            if isinstance(response, str):
                start = response.find("{")
                end = response.rfind("}") + 1
                if start != -1 and end != 0:
                    parsed = json.loads(response[start:end])
                    text_summary = parsed.get("text_summary", "")
                    metrics_data = parsed.get("metrics", [])

                    metrics = [
                        MetricRecord(
                            metric_name=m.get("metric_name", ""),
                            metric_value=m.get("metric_value", ""),
                            unit=m.get("unit"),
                            reference_range=m.get("reference_range")
                        )
                        for m in metrics_data
                    ]

                    return ParsedReport(
                        report_type="image",
                        raw_text=text_summary,
                        metrics=metrics,
                        page_count=1,
                        success=True
                    )

            return ParsedReport(
                report_type="image",
                raw_text=response if isinstance(response, str) else "",
                metrics=[],
                page_count=1,
                success=True
            )

        except Exception as e:
            return ParsedReport(
                report_type="image",
                raw_text="",
                metrics=[],
                page_count=1,
                success=False,
                error=str(e)
            )

    def parse(self, content: bytes, filename: str) -> ParsedReport:
        """
        统一解析入口。

        根据文件类型自动选择解析方法。

        Args:
            content: 文件字节
            filename: 文件名

        Returns:
            解析结果
        """
        filename_lower = filename.lower()

        # 根据扩展名判断类型
        if filename_lower.endswith(".pdf"):
            pdf_type, _ = self.detect_pdf_type(content)
            if pdf_type == "text_pdf":
                return self.parse_text_pdf(content)
            else:
                return self.parse_scanned_pdf(content)

        elif filename_lower.endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp")):
            return self.parse_image_report(content, self._get_mime_type(filename_lower))

        else:
            return ParsedReport(
                report_type="unknown",
                raw_text="",
                metrics=[],
                page_count=0,
                success=False,
                error=f"不支持的文件类型: {filename}"
            )

    def _render_pdf_to_images(self, pdf_bytes: bytes, dpi: int = 72) -> List[bytes]:
        """
        将PDF页面渲染为图像。

        Args:
            pdf_bytes: PDF字节
            dpi: 渲染DPI

        Returns:
            图像字节列表
        """
        try:
            from PIL import Image
            import fitz  # pymupdf

            images = []
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")

            for page in doc:
                # 渲染页面
                mat = fitz.Matrix(dpi / 72, dpi / 72)
                pix = page.get_pixmap(matrix=mat)

                # 转换为PNG
                img_bytes = pix.tobytes("png")
                images.append(img_bytes)

            doc.close()
            return images

        except ImportError:
            # pymupdf不可用
            return []
        except Exception:
            return []

    def _get_mime_type(self, filename: str) -> str:
        """获取MIME类型。"""
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".bmp": "image/bmp"
        }
        for ext, mime in mime_types.items():
            if filename.endswith(ext):
                return mime
        return "image/png"

    def _extract_metrics_from_text(self, text: str) -> List[MetricRecord]:
        """
        从文本中提取指标。

        使用LLM进行结构化提取。

        Args:
            text: 文本内容

        Returns:
            指标列表
        """
        prompt = f"""从以下体检报告文本中提取所有医学指标。

文本:
{text[:3000]}

请以JSON格式输出:
{{
    "metrics": [
        {{"metric_name": "指标名", "metric_value": "指标值", "unit": "单位", "reference_range": "参考范围"}}
    ]
}}

如果没有找到指标，返回空数组: {{"metrics": []}}
只输出JSON。"""

        try:
            response = self.llm_client.chat_with_json(
                messages=[
                    {"role": "system", "content": "你是一个医疗指标提取助手。"},
                    {"role": "user", "content": prompt}
                ]
            )

            if isinstance(response, dict):
                metrics_data = response.get("metrics", [])
                return [
                    MetricRecord(
                        metric_name=m.get("metric_name", ""),
                        metric_value=m.get("metric_value", ""),
                        unit=m.get("unit"),
                        reference_range=m.get("reference_range")
                    )
                    for m in metrics_data
                ]

        except Exception:
            pass

        return []


# 全局实例
_vision_encoder_service: VisionEncoderService | None = None


def get_vision_encoder_service() -> VisionEncoderService:
    """获取VisionEncoder服务实例。"""
    global _vision_encoder_service
    if _vision_encoder_service is None:
        _vision_encoder_service = VisionEncoderService()
    return _vision_encoder_service
