"""Deterministic safety guardrails for the medical-assistance boundary."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.schema.chat import SafetyCheckResult


DISCLAIMER = "仅供参考，请咨询专业医生。"


@dataclass(frozen=True)
class SafetyRule:
    name: str
    description: str
    patterns: tuple[str, ...]
    blocking: bool = True


RULES = (
    SafetyRule(
        "dosage",
        "不得给出具体用药剂量或服用频次",
        (r"(?:^|[^0-9A-Za-z])\d+(?:\.\d+)?\s*(?:mg|g|ml)\b", r"每日\s*\d+", r"每次\s*\d+"),
    ),
    SafetyRule(
        "diagnosis",
        "不得替代医生作出明确诊断",
        ("确诊", "就是", "一定是", "你患有", "诊断为"),
    ),
    SafetyRule(
        "single_metric",
        "不得仅依据单一指标判断疾病",
        ("仅凭", "只凭", "单凭", "单一指标", "仅根据这个指标"),
    ),
)

EMERGENCY_TERMS = (
    "危急值", "胸痛", "呼吸困难", "意识不清", "昏厥", "大出血", "严重过敏", "急性胸痛"
)
EMERGENCY_ACTION = ("立即就医", "急诊", "拨打120", "尽快就医")


def _matches(text: str, pattern: str) -> bool:
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def check_response(content: str, *, require_disclaimer: bool = True) -> SafetyCheckResult:
    """Return a structured result; ``red_flag`` means the output must be blocked."""

    text = content or ""
    warnings: list[str] = []
    blocking = False

    for rule in RULES:
        if any(_matches(text, pattern) for pattern in rule.patterns):
            warnings.append(f"触发安全规则：{rule.description}")
            blocking = blocking or rule.blocking

    if require_disclaimer and DISCLAIMER not in text and "咨询专业医生" not in text:
        warnings.append("缺少医疗辅助免责声明")

    has_emergency = any(term in text for term in EMERGENCY_TERMS)
    if has_emergency and not any(action in text for action in EMERGENCY_ACTION):
        warnings.append("检测到潜在危急症状，但未给出及时就医提示")
        blocking = True

    return SafetyCheckResult(
        passed=not blocking,
        warnings=warnings[:5],
        red_flag=blocking,
        critical=blocking and has_emergency,
    )


def enforce_boundary(content: str) -> tuple[str, SafetyCheckResult]:
    """Never return a blocking model response verbatim."""

    result = check_response(content)
    if result.red_flag:
        safe_text = (
            "这个问题涉及需要专业判断的医疗风险，我不能据此做出诊断或提供具体用药方案。"
            "建议携带完整报告、既往病史和用药记录，尽快咨询专业医生；如出现胸痛、呼吸困难、"
            f"意识不清或其他危急症状，请立即前往急诊。{DISCLAIMER}"
        )
        return safe_text, result

    if DISCLAIMER not in content and "咨询专业医生" not in content:
        content = f"{content.rstrip()}\n\n{DISCLAIMER}"
    return content, result
