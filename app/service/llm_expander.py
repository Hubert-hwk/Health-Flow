"""LLMExpander - 使用 LLM 批量生成多样化的 SFT 指令-响应对."""

import json
import re
import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field

from app.model.llm import get_minimax_client

logger = logging.getLogger(__name__)


@dataclass
class ExpansionResult:
    """单次扩展结果."""

    instruction: str
    input: str
    output: str
    category: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
            "category": self.category,
            **self.metadata,
        }


@dataclass
class LLMExpanderConfig:
    """LLM扩展配置."""

    model: str = "MiniMax-M2.7"
    temperature: float = 0.8
    max_tokens: int = 1024
    batch_size: int = 3  # 每次请求生成多少条（小一点更稳定）
    max_retries: int = 3


class LLMExpander:
    """
    使用 LLM 批量生成多样化的指令-响应对。

    核心流程：
    1. 接收种子模板（含占位符）
    2. 为每个模板构造生成提示
    3. 调用 MiniMax API 批量生成多样化变体
    4. 解析并返回结构化结果
    """

    # ========== 各类别系统提示 ==========

    SYSTEM_PROMPTS = {
        "体检报告解读": """你是一个专业的医疗助手，擅长解读体检报告指标。
请根据给定的模板，生成多条多样化的体检报告解读数据。
要求：
- 保持医学知识的准确性
- 使用不同的表述方式
- 包含适当的健康建议
- 必须添加"仅供参考，请咨询专业医生"等免责提示
- 输出格式为JSON数组""",
        "指标异常问询": """你是一个专业的医疗助手，擅长回答指标异常相关问题。
请根据给定的模板，生成多条多样化的指标异常问询数据。
要求：
- 饮食建议要具体但不过度
- 不能提及具体用药剂量
- 不能替代医生诊断
- 必须添加"仅供参考，请咨询专业医生"等免责提示
- 输出格式为JSON数组""",
        "科室分诊建议": """你是一个专业的医疗分诊助手，擅长根据症状推荐就诊科室。
请根据给定的模板，生成多条多样化的科室分诊建议数据。
要求：
- 科室推荐要准确
- 解释要简洁明了
- 包含就诊建议
- 输出格式为JSON数组""",
        "医疗安全问答": """你是一个严格的医疗安全审核助手。
请根据给定的safe回答，生成一条对应的unsafe回答（包含具体剂量、替代诊断等不安全内容）。
要求：
- safe回答必须符合医疗安全规范
- unsafe回答必须明确违反安全红线（如具体剂量、替代诊断）
- 两者问题相同，但回答完全不同
- 输出格式为JSON，包含safe和unsafe两个字段""",
    }

    # ========== 生成提示模板 ==========

    EXPANSION_PROMPTS = {
        "体检报告解读": """请为以下模板生成{batch_size}条多样化的变体。

模板：
{template}

要求：
1. 替换{placeholder}为不同的医学指标数值（注意在合理范围内）
2. 使用不同的问法表达同一意思
3. 输出必须是有医疗意义的内容
4. 每条数据包含：instruction、input、output

请直接输出JSON数组，不要包含其他内容。""",

        "指标异常问询": """请为以下模板生成{batch_size}条多样化的变体。

模板：
{template}

要求：
1. 替换{placeholder}为不同的指标和数值
2. 饮食建议要实用、具体
3. 不能提及任何具体用药剂量
4. 每条数据包含：instruction、input、output

请直接输出JSON数组，不要包含其他内容。""",

        "科室分诊建议": """请为以下模板生成{batch_size}条多样化的变体。

模板：
{template}

要求：
1. 替换{symptom}为不同的症状描述
2. 症状要常见且描述清晰
3. 科室推荐要准确
4. 每条数据包含：instruction、input、output

请直接输出JSON数组，不要包含其他内容。""",

        "医疗安全问答": """请为以下safe回答生成对应的unsafe回答。

Safe回答：
{safe_output}

要求：
- 生成一条完全相同的instruction和input
- 但unsafe_output必须包含：具体剂量（如"每天500mg"）、替代诊断（如"你就是癌症"）
- 这用于DPO训练，教会模型识别不安全回答

请直接输出JSON，格式：{{"instruction": "...", "input": "...", "unsafe_output": "..."}}，不要包含其他内容。""",
    }

    def __init__(self, config: Optional[LLMExpanderConfig] = None):
        self.config = config or LLMExpanderConfig()
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = get_minimax_client()
        return self._client

    def set_progress_callback(self, callback: Callable[[float, str], None]):
        """设置进度回调。"""
        self._progress_callback = callback

    def _report_progress(self, progress: float, status: str):
        if hasattr(self, "_progress_callback"):
            self._progress_callback(progress, status)
        else:
            logger.info(f"[{progress:.0%}] {status}")

    def _call_llm(self, messages: List[Dict[str, str]], retries: int = 0) -> str:
        """调用 LLM，返回文本内容。"""
        import time
        try:
            return self.client.chat(messages)
        except Exception as e:
            if retries < self.config.max_retries:
                wait_time = (retries + 1) * 2  # 指数退避: 2s, 4s, 6s
                logger.warning(f"LLM调用失败，{wait_time}秒后重试 {retries + 1}/{self.config.max_retries}: {e}")
                time.sleep(wait_time)
                return self._call_llm(messages, retries + 1)
            logger.error(f"LLM调用最终失败: {e}")
            return ""

    def _parse_json_array(self, text: str, category: str) -> List[Dict[str, Any]]:
        """解析 LLM 返回的 JSON 数组。"""
        # 去除 <think>...</think> 标签（MiniMax 模型可能返回推理过程）
        text = re.sub(r"<think>[\s\S]*?</think>", "", text)

        # 尝试提取 ```json ... ``` 包裹的内容
        match = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text)
        if match:
            text = match.group(1)
        else:
            # 尝试直接找 JSON 数组
            start = text.find("[")
            end = text.rfind("]") + 1
            if start != -1 and end > start:
                text = text[start:end]

        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return [data]
        except json.JSONDecodeError as e:
            logger.warning(f"JSON解析失败: {e}, 原始文本: {text[:200]}")
            # 尝试修复常见转义问题后重试
            try:
                fixed = text.replace('\\/', '/').replace('\\n', '\n')
                data = json.loads(fixed)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    return [data]
            except:
                pass
        except Exception as e:
            logger.warning(f"JSON解析异常: {e}")

        return []

    def _parse_json_object(self, text: str) -> Dict[str, Any]:
        """解析 LLM 返回的 JSON 对象。"""
        # 去除 <think>...</think> 标签
        text = re.sub(r"<think>[\s\S]*?</think>", "", text)

        match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
        if match:
            text = match.group(1)
        else:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                text = text[start:end]

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON对象解析失败: {e}")
            return {}

    def expand_examination(self, template: str, placeholders: Dict[str, List[str]], count: int) -> List[ExpansionResult]:
        """
        扩展体检报告解读类数据。

        Args:
            template: 模板字符串
            placeholders: 占位符字典，如 {"metric": ["空腹血糖", "血压"], "value": ["6.5", "7.8"]}
            count: 需要生成的总数

        Returns:
            扩展结果列表
        """
        results = []
        batches = (count + self.config.batch_size - 1) // self.config.batch_size

        for batch_idx in range(batches):
            batch_count = min(self.config.batch_size, count - batch_idx * self.config.batch_size)
            self._report_progress(
                batch_idx / batches,
                f"体检报告解读: 生成第 {batch_idx + 1}/{batches} 批 ({batch_count}条)"
            )

            prompt = self.EXPANSION_PROMPTS["体检报告解读"].format(
                template=template,
                placeholder=str(placeholders),
                batch_size=batch_count,
            )

            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPTS["体检报告解读"]},
                {"role": "user", "content": prompt},
            ]

            text = self._call_llm(messages)
            if not text:
                continue

            items = self._parse_json_array(text, "体检报告解读")
            for item in items:
                if "instruction" in item and "output" in item:
                    results.append(ExpansionResult(
                        instruction=item["instruction"],
                        input=item.get("input", ""),
                        output=item["output"],
                        category="体检报告解读",
                        metadata={"source": "llm_expanded"},
                    ))

        return results

    def expand_metric_query(self, template: str, placeholders: Dict[str, List[str]], count: int) -> List[ExpansionResult]:
        """扩展指标异常问询类数据。"""
        results = []
        batches = (count + self.config.batch_size - 1) // self.config.batch_size

        for batch_idx in range(batches):
            batch_count = min(self.config.batch_size, count - batch_idx * self.config.batch_size)
            self._report_progress(
                batch_idx / batches,
                f"指标异常问询: 生成第 {batch_idx + 1}/{batches} 批 ({batch_count}条)"
            )

            prompt = self.EXPANSION_PROMPTS["指标异常问询"].format(
                template=template,
                placeholder=str(placeholders),
                batch_size=batch_count,
            )

            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPTS["指标异常问询"]},
                {"role": "user", "content": prompt},
            ]

            text = self._call_llm(messages)
            if not text:
                continue

            items = self._parse_json_array(text, "指标异常问询")
            for item in items:
                if "instruction" in item and "output" in item:
                    results.append(ExpansionResult(
                        instruction=item["instruction"],
                        input=item.get("input", ""),
                        output=item["output"],
                        category="指标异常问询",
                        metadata={"source": "llm_expanded"},
                    ))

        return results

    def expand_triage(self, template: str, symptoms: List[str], count: int) -> List[ExpansionResult]:
        """扩展科室分诊建议类数据。"""
        results = []
        batches = (count + self.config.batch_size - 1) // self.config.batch_size

        for batch_idx in range(batches):
            batch_count = min(self.config.batch_size, count - batch_idx * self.config.batch_size)
            self._report_progress(
                batch_idx / batches,
                f"科室分诊: 生成第 {batch_idx + 1}/{batches} 批 ({batch_count}条)"
            )

            prompt = self.EXPANSION_PROMPTS["科室分诊建议"].format(
                template=template,
                symptom=str(symptoms[:batch_count]),
                batch_size=batch_count,
            )

            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPTS["科室分诊建议"]},
                {"role": "user", "content": prompt},
            ]

            text = self._call_llm(messages)
            if not text:
                continue

            items = self._parse_json_array(text, "科室分诊建议")
            for item in items:
                if "instruction" in item and "output" in item:
                    results.append(ExpansionResult(
                        instruction=item["instruction"],
                        input=item.get("input", ""),
                        output=item["output"],
                        category="科室分诊建议",
                        metadata={"source": "llm_expanded"},
                    ))

        return results

    def expand_safety_pair(self, safe_outputs: List[str], count: int) -> List[ExpansionResult]:
        """
        扩展医疗安全问答类数据（生成 safe/unsafe 对）。

        Args:
            safe_outputs: safe 回答列表
            count: 需要生成的总数

        Returns:
            扩展结果列表，包含 safe 和 unsafe 样本
        """
        results = []
        for i, safe_output in enumerate(safe_outputs[:count]):
            self._report_progress(
                i / count,
                f"医疗安全问答: 生成第 {i + 1}/{count} 条"
            )

            prompt = self.EXPANSION_PROMPTS["医疗安全问答"].format(
                safe_output=safe_output,
            )

            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPTS["医疗安全问答"]},
                {"role": "user", "content": prompt},
            ]

            text = self._call_llm(messages)
            if not text:
                continue

            data = self._parse_json_object(text)
            if "instruction" in data and "unsafe_output" in data:
                # Safe sample
                results.append(ExpansionResult(
                    instruction=data["instruction"],
                    input=data.get("input", ""),
                    output=safe_output,
                    category="医疗安全问答",
                    metadata={"source": "llm_expanded", "is_dpo_negative": False},
                ))
                # Unsafe sample (DPO negative)
                results.append(ExpansionResult(
                    instruction=data["instruction"],
                    input=data.get("input", ""),
                    output=data["unsafe_output"],
                    category="医疗安全问答",
                    metadata={"source": "llm_expanded", "is_dpo_negative": True},
                ))

        return results


def expand_category(
    category: str,
    templates: List[Dict[str, Any]],
    total_count: int,
    expander: Optional[LLMExpander] = None,
) -> List[ExpansionResult]:
    """
    通用扩展函数，根据类别调用对应扩展方法。

    Args:
        category: 类别名称
        templates: 模板列表
        total_count: 需要生成的总数量
        expander: LLMExpander 实例

    Returns:
        扩展结果列表
    """
    if expander is None:
        expander = LLMExpander()

    results = []
    per_template = total_count // len(templates)

    if category == "体检报告解读":
        for tmpl in templates:
            results.extend(
                expander.expand_examination(
                    template=tmpl["template"],
                    placeholders=tmpl.get("placeholders", {}),
                    count=per_template,
                )
            )

    elif category == "指标异常问询":
        for tmpl in templates:
            results.extend(
                expander.expand_metric_query(
                    template=tmpl["template"],
                    placeholders=tmpl.get("placeholders", {}),
                    count=per_template,
                )
            )

    elif category == "科室分诊建议":
        for tmpl in templates:
            results.extend(
                expander.expand_triage(
                    template=tmpl["template"],
                    symptoms=tmpl.get("symptoms", []),
                    count=per_template,
                )
            )

    elif category == "医疗安全问答":
        for tmpl in templates:
            results.extend(
                expander.expand_safety_pair(
                    safe_outputs=tmpl.get("safe_outputs", []),
                    count=per_template,
                )
            )

    return results
