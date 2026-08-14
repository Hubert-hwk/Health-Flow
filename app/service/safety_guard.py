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


# 剂量规则要求"用药语境"（数量词/动词 + 单位），避免把化验值
# （如 血红蛋白145g/L、24h尿蛋白0.5g）误判为剂量。单位后紧跟
# "/"（g/L、mg/dL、g/24h）的视为化验单位，不命中。
# 注意：这里用普通字符串拼接而非 rf 字符串，避免 {0,12} 被当成占位符。
_DOSE_QTY = "(?:每天|每日|每次|一天|一日|早晚|睡前|饭后|餐前|餐后|空腹|一次|一片|一粒|一支|一针|三餐)"
_DOSE_VERB = "(?:服用|口服|吃|用药|注射|静注|静脉|舌下|含服|滴注|外用|涂抹|吸入|点滴)"
_DOSE_UNIT = "(?:毫克|克|毫升|mg|g|ml|片|粒|支|单位|ug|μg)"
_CN_NUM = "[一二两三四五六七八九十半]"
_NAME_CHARS = "[一-龥A-Za-z0-9·\\-（）()]"

RULES = (
    SafetyRule(
        "dosage",
        "不得给出具体用药剂量或服用频次",
        (
            # 数量词 +（可选药名）+ 阿拉伯数字剂量，如「每天吃二甲双胍500mg」「每次1片」
            _DOSE_QTY + _NAME_CHARS + "{0,12}\\s*\\d+(?:\\.\\d+)?\\s*" + _DOSE_UNIT + "(?!\\s*[/%])",
            # 动词 +（可选药名）+ 阿拉伯数字剂量，如「直接吃硝苯地平缓释片20mg」
            _DOSE_VERB + _NAME_CHARS + "{0,12}\\s*\\d+(?:\\.\\d+)?\\s*" + _DOSE_UNIT + "(?!\\s*[/%])",
            # 中文数字剂量/频次，如「每次一片」「每天三次」「早晚各半片」
            _DOSE_QTY + "\\s*(?:服用|口服|吃|用|各)?\\s*" + _CN_NUM + "+\\s*(?:" + _DOSE_UNIT + "|次)(?!\\s*[/%])",
            # 明确的服药频次缩写（tid/bid/qd/qid）
            r"\b(?:tid|bid|qd|qid|prn)\b",
        ),
    ),
    SafetyRule(
        "diagnosis",
        "不得替代医生作出明确诊断",
        (
            "确诊",
            "你患有",
            "诊断为",
            "一定是",
            "绝对是",
            "肯定是",
            # 「就是」只在与疾病名词搭配时拦截，避免误伤
            # 「血糖偏高就是需要控制饮食」这类正常表述
            r"就是[一-龥]{0,10}(?:病|癌|症|炎|综合征)",
            r"(?:你|您)就是(?:得了)?[一-龥]{0,8}(?:病|癌|症|炎|综合征)",
        ),
    ),
    SafetyRule(
        "single_metric",
        "不得仅依据单一指标判断疾病",
        ("仅凭", "只凭", "单凭", "单一指标", "仅根据这个指标"),
    ),
)

# 触发后需检查 30 字窗口内是否带否定/谨慎措辞；若带则不算违规
# （如「仅凭空腹血糖6.5无法确诊糖尿病」是合规的安全表述）。
_SINGLE_METRIC_NEGATION = ("无法", "不能", "不代表", "不等于", "不是", "不可", "需", "需要", "建议", "应", "进一步", "完善")

EMERGENCY_TERMS = (
    "危急值", "胸痛", "呼吸困难", "意识不清", "昏厥", "大出血", "严重过敏", "急性胸痛"
)
EMERGENCY_ACTION = (
    "立即就医", "急诊", "拨打120", "尽快就医", "去医院", "到医院", "就医",
    "看医生", "就诊", "急救",
)


def _matches(text: str, pattern: str) -> bool:
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


# 诊断类触发词附近出现否定/谨慎措辞时不视为越界诊断，
# 如「无法确诊糖尿病」「不能诊断为高血压」是合规的安全表述。
_DIAGNOSIS_NEGATION = ("无法", "不能", "不代表", "不等于", "不是", "不可", "难以", "未", "排除", "需", "需要", "建议", "应", "进一步", "完善", "复查", "检查")


def _diagnosis_blocks(text: str) -> bool:
    rule = next(rule for rule in RULES if rule.name == "diagnosis")
    for pattern in rule.patterns:
        start = 0
        while True:
            match = re.search(pattern, text[start:], flags=re.IGNORECASE)
            if not match:
                break
            pos = start + match.start()
            window = text[max(0, pos - 12) : pos + len(match.group(0)) + 12]
            if not any(neg in window for neg in _DIAGNOSIS_NEGATION):
                return True
            start = pos + len(match.group(0))
    return False


def _single_metric_blocks(text: str) -> bool:
    """仅凭/只凭等触发词，若紧跟否定或建议性措辞则不算单一指标下结论。"""
    for trigger in ("仅凭", "只凭", "单凭", "单一指标", "仅根据这个指标"):
        start = 0
        while True:
            pos = text.find(trigger, start)
            if pos < 0:
                break
            window = text[pos : pos + len(trigger) + 30]
            if not any(neg in window for neg in _SINGLE_METRIC_NEGATION):
                return True
            start = pos + len(trigger)
    return False


def check_response(content: str, *, require_disclaimer: bool = True) -> SafetyCheckResult:
    """Return a structured result; ``red_flag`` means the output must be blocked."""

    text = content or ""
    warnings: list[str] = []
    blocking = False

    for rule in RULES:
        if rule.name == "single_metric":
            if _single_metric_blocks(text):
                warnings.append(f"触发安全规则：{rule.description}")
                blocking = blocking or rule.blocking
            continue
        if rule.name == "diagnosis":
            if _diagnosis_blocks(text):
                warnings.append(f"触发安全规则：{rule.description}")
                blocking = blocking or rule.blocking
            continue
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

    # 免责声明由本函数负责补齐，因此检查时不再把「缺少免责声明」记为警告
    result = check_response(content, require_disclaimer=False)
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
