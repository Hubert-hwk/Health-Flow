"""Tests for VisionEncoder Service."""

import pytest
from unittest.mock import MagicMock, patch
import io


class MockVLMClient:
    """Mock VLM client."""

    def chat_with_image(self, messages, **kwargs):
        return '{"text_summary": "空腹血糖6.5mmol/L", "metrics": [{"metric_name": "空腹血糖", "metric_value": "6.5", "unit": "mmol/L", "reference_range": "3.9-6.1"}]}'


class MockLLMClient:
    """Mock LLM client."""

    def chat_with_json(self, messages, **kwargs):
        return {
            "metrics": [
                {"metric_name": "空腹血糖", "metric_value": "6.5", "unit": "mmol/L", "reference_range": "3.9-6.1"}
            ]
        }


@pytest.fixture
def mock_deps():
    """Mock dependencies."""
    with patch('app.service.vision_encoder.get_vlm_client', return_value=MockVLMClient()), \
         patch('app.service.vision_encoder.get_llm_client', return_value=MockLLMClient()):
        yield


def test_vision_encoder_init():
    """Test VisionEncoderService initialization."""
    from app.service.vision_encoder import VisionEncoderService

    service = VisionEncoderService()
    assert service is not None


def test_parse_image_report(mock_deps):
    """Test parsing image report."""
    from app.service.vision_encoder import VisionEncoderService

    service = VisionEncoderService()

    # Create a small PNG image (1x1 pixel)
    img_bytes = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'

    result = service.parse_image_report(img_bytes, "image/png")

    assert result.report_type == "image"
    assert result.success is True
    assert result.page_count == 1


def test_parse_unsupported_file(mock_deps):
    """Test parsing unsupported file type."""
    from app.service.vision_encoder import VisionEncoderService

    service = VisionEncoderService()
    result = service.parse(b"some content", "unknown.xyz")

    assert result.report_type == "unknown"
    assert result.success is False
    assert "不支持" in result.error


def test_get_mime_type():
    """Test MIME type detection."""
    from app.service.vision_encoder import VisionEncoderService

    service = VisionEncoderService()

    assert service._get_mime_type("test.jpg") == "image/jpeg"
    assert service._get_mime_type("test.jpeg") == "image/jpeg"
    assert service._get_mime_type("test.png") == "image/png"
    assert service._get_mime_type("test.gif") == "image/gif"
    assert service._get_mime_type("test.bmp") == "image/bmp"
    assert service._get_mime_type("test.xyz") == "image/png"  # default


def test_parsed_report_dataclass():
    """Test ParsedReport dataclass."""
    from app.service.vision_encoder import ParsedReport

    report = ParsedReport(
        report_type="text_pdf",
        raw_text="Test content",
        metrics=[],
        page_count=1,
        success=True
    )

    assert report.report_type == "text_pdf"
    assert report.raw_text == "Test content"
    assert report.success is True
    assert report.error is None


def test_parse_with_filename_jpg(mock_deps):
    """Test parsing JPEG file."""
    from app.service.vision_encoder import VisionEncoderService

    service = VisionEncoderService()

    # Minimal JPEG data
    jpeg_data = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\x27 ,#\x1c\x1c(7teleservices5!=17==11teleservices1x;x8teleservices\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x01\x01\x01\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfb\xd5v\xff\xd9'

    result = service.parse(jpeg_data, "test.jpg")

    # Should attempt to parse as image
    assert result.report_type == "image"


def test_render_pdf_to_images_fallback():
    """Test PDF rendering fallback when pymupdf not available."""
    from app.service.vision_encoder import VisionEncoderService

    service = VisionEncoderService()

    # With no pymupdf, should return empty list
    result = service._render_pdf_to_images(b"fake pdf content")
    assert isinstance(result, list)
