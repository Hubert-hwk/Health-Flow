"""DataAugmentationPipeline - 数据增强Pipeline.

逆向指令工程与自动化数据增强，构造约8k规模的高质量垂直指令微调集。
"""

import json
import re
from typing import List, Dict, Any, Optional, Callable, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime
import random

if TYPE_CHECKING:
    from app.service.llm_expander import ExpansionResult


@dataclass
class AugmentConfig:
    """数据增强配置。"""

    target_size: int = 8000
    source: str = "llm"  # template=模板生成, llm=LLM扩展生成
    categories: List[str] = field(default_factory=lambda: [
        "体检报告解读",
        "指标异常问询",
        "科室分诊建议",
        "医疗安全问答"
    ])
    output_path: str = "./data/sft/training_data.jsonl"
    language: str = "zh"


@dataclass
class InstructionPair:
    """指令-响应对。"""

    instruction: str
    input: str
    output: str
    category: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式。"""
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
            "category": self.category,
            "source": self.source,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InstructionPair":
        """从字典创建。"""
        return cls(
            instruction=data.get("instruction", ""),
            input=data.get("input", ""),
            output=data.get("output", ""),
            category=data.get("category", ""),
            source=data.get("source", ""),
            metadata=data.get("metadata", {})
        )


class DataAugmentationPipeline:
    """
    数据增强Pipeline。

    核心职责:
    1. 逆向指令工程：从原始医学内容生成指令-响应对
    2. 多样性增强：生成同一内容的多种表述方式
    3. 数据清洗：去重、质量审核、安全过滤
    4. 格式标准化：统一输出格式

    数据规模目标（约8000条）:
    - 体检报告解读类: ~3000条
    - 指标异常问询类: ~2000条
    - 科室分诊建议类: ~1500条
    - 医疗安全问答类: ~1500条（含DPO负样本）
    """

    def __init__(self, config: Optional[AugmentConfig] = None):
        """
        初始化数据增强Pipeline。

        Args:
            config: 增强配置
        """
        self.config = config or AugmentConfig()
        self._llm_client = None
        self._pairs: List[InstructionPair] = []
        self._progress_callback: Optional[Callable] = None

    @property
    def llm_client(self):
        """懒加载LLM客户端。"""
        if self._llm_client is None:
            from app.model.llm import get_llm_client
            self._llm_client = get_llm_client()
        return self._llm_client

    def set_progress_callback(self, callback: Callable[[float, str], None]):
        """
        设置进度回调。

        Args:
            callback: (progress: 0-1, status: str) -> None
        """
        self._progress_callback = callback

    def _report_progress(self, progress: float, status: str):
        """报告进度。"""
        if self._progress_callback:
            self._progress_callback(progress, status)

    # ========== 模板数据增强 ==========

    def generate_from_template(self, template_type: str, count: int) -> List[InstructionPair]:
        """
        从模板生成数据。

        Args:
            template_type: 模板类型
            count: 生成数量

        Returns:
            生成的指令-响应对列表
        """
        pairs = []

        if template_type == "体检报告解读":
            pairs = self._generate_examination_templates(count)
        elif template_type == "指标异常问询":
            pairs = self._generate_metric_templates(count)
        elif template_type == "科室分诊建议":
            pairs = self._generate_triage_templates(count)
        elif template_type == "医疗安全问答":
            pairs = self._generate_safety_templates(count)

        return pairs

    def _generate_examination_templates(self, count: int) -> List[InstructionPair]:
        """生成体检报告解读类数据。"""
        templates = [
            {
                "instruction": "请解读这份体检报告中{metric}指标的意义",
                "input": "{metric}: {value} {unit}, 参考范围: {reference}",
                "output": "您的{metric}为{value}{unit}，{assessment}。{advice}"
            },
            {
                "instruction": "{metric}指标偏高可能是什么原因？",
                "input": "{metric}: {value} {unit}（参考范围: {reference}）",
                "output": "{metric}偏高可能由以下原因引起：{causes}。建议：{advice}"
            },
            {
                "instruction": "我最近{metric}有点问题，应该挂什么科？",
                "input": "{metric}: {value} {unit}",
                "output": "根据您的症状，建议就诊{department}。{reasoning}"
            },
        ]

        metrics_data = [
            {"metric": "空腹血糖", "value": "6.5", "unit": "mmol/L", "reference": "3.9-6.1", "assessment": "超过正常上限", "advice": "建议咨询内分泌科", "causes": "饮食因素、缺乏运动、糖尿病前期", "department": "内分泌科"},
            {"metric": "空腹血糖", "value": "7.8", "unit": "mmol/L", "reference": "3.9-6.1", "assessment": "明显超标", "advice": "需进一步检查确诊", "causes": "糖尿病可能性大", "department": "内分泌科"},
            {"metric": "血压", "value": "145/95", "unit": "mmHg", "reference": "<140/90", "assessment": "轻度偏高", "advice": "注意饮食，定期复查", "causes": "高盐饮食、精神紧张", "department": "心内科"},
            {"metric": "血压", "value": "165/105", "unit": "mmHg", "reference": "<140/90", "assessment": "中度偏高", "advice": "需要药物治疗", "causes": "高血压可能", "department": "心内科"},
            {"metric": "总胆固醇", "value": "6.2", "unit": "mmol/L", "reference": "<5.2", "assessment": "偏高", "advice": "控制饮食，加强运动", "causes": "高脂饮食、代谢异常", "department": "心内科"},
            {"metric": "血红蛋白", "value": "145", "unit": "g/L", "reference": "130-175", "assessment": "在正常范围", "advice": "保持良好生活习惯", "causes": "正常生理状态", "department": "血液科"},
        ]

        pairs = []
        for i in range(count):
            template = random.choice(templates)
            metric_data = random.choice(metrics_data)

            instruction = template["instruction"].format(**metric_data)
            input_text = template["input"].format(**metric_data)
            output = template["output"].format(**metric_data)

            pairs.append(InstructionPair(
                instruction=instruction,
                input=input_text,
                output=output,
                category="体检报告解读",
                source="template",
                metadata={"template_type": "examination", "seed": i}
            ))

        return pairs

    def _generate_metric_templates(self, count: int) -> List[InstructionPair]:
        """生成指标异常问询类数据。"""
        templates = [
            {
                "instruction": "{metric}偏高饮食需要注意什么？",
                "input": "{metric}: {value} {unit}，偏高",
                "output": "{metric}偏高时，饮食建议：{diet_advice}。同时注意：{other_advice}"
            },
            {
                "instruction": "{metric}偏高需要吃药吗？",
                "input": "{metric}: {value} {unit}（参考范围: {reference}）",
                "output": "关于{metric}偏高是否需要用药：{medication_advice}。具体方案请遵医嘱。"
            },
            {
                "instruction": "{metric}指标异常会是癌症吗？",
                "input": "{metric}: {value} {unit}，出现异常",
                "output": "{metric}异常可能由多种原因引起，{cancer_probability}。建议进行进一步检查：{check_advice}。"
            },
        ]

        metric_advice = [
            {"metric": "空腹血糖", "value": "6.5", "unit": "mmol/L", "reference": "3.9-6.1",
             "diet_advice": "控制碳水化合物摄入，少吃甜食，主食定量",
             "other_advice": "增加运动量，保持规律作息",
             "medication_advice": "目前属于糖尿病前期，一般先通过生活方式干预",
             "cancer_probability": "癌症可能性较低，多为代谢或饮食因素",
             "check_advice": "完善糖化血红蛋白检测",
             "department": "内分泌科"},
            {"metric": "CEA", "value": "8.5", "unit": "ng/mL", "reference": "<5.0",
             "diet_advice": "无特殊饮食建议",
             "other_advice": "建议进一步检查排除肿瘤可能",
             "medication_advice": "需要进一步检查明确原因",
             "cancer_probability": "需要高度重视，CEA升高与多种肿瘤相关",
             "check_advice": "全面体检，胃肠镜、CT等",
             "department": "肿瘤科"},
            {"metric": "AFP", "value": "25", "unit": "ng/mL", "reference": "<20",
             "diet_advice": "无特殊限制",
             "other_advice": "建议完善肝脏检查",
             "medication_advice": "需排查肝炎或肿瘤可能",
             "cancer_probability": "需排查肝癌及生殖系统肿瘤",
             "check_advice": "肝脏B超、肝炎指标检测",
             "department": "肝胆外科"},
        ]

        pairs = []
        for i in range(count):
            template = random.choice(templates)
            advice_data = random.choice(metric_advice)

            instruction = template["instruction"].format(**advice_data)
            input_text = template["input"].format(**advice_data)
            output = template["output"].format(**advice_data)

            pairs.append(InstructionPair(
                instruction=instruction,
                input=input_text,
                output=output,
                category="指标异常问询",
                source="template",
                metadata={"template_type": "metric_advice", "seed": i}
            ))

        return pairs

    def _generate_triage_templates(self, count: int) -> List[InstructionPair]:
        """生成科室分诊建议类数据。"""
        templates = [
            {
                "instruction": "我最近出现{symptom}，应该挂什么科？",
                "input": "症状：{symptom}，持续时间：{duration}",
                "output": "根据您描述的{symptom}，建议就诊{department}。{reasoning}"
            },
            {
                "instruction": "{symptom}应该看什么科室？",
                "input": "症状描述：{symptom}",
                "output": "建议就诊{department}。{reasoning}。如无法判断，可先到全科医学科就诊。"
            },
        ]

        triage_data = [
            {"symptom": "多饮、多尿、体重下降", "duration": "2周", "department": "内分泌科", "reasoning": "这些症状是糖尿病的典型表现"},
            {"symptom": "心悸、胸闷、活动后气促", "duration": "1周", "department": "心内科", "reasoning": "需要排除心血管疾病"},
            {"symptom": "腹痛、腹胀、恶心呕吐", "duration": "3天", "department": "消化科", "reasoning": "消化道症状首选消化科"},
            {"symptom": "咳嗽、咳痰、发热", "duration": "5天", "department": "呼吸科", "reasoning": "呼吸道症状"},
            {"symptom": "头痛、头晕、失眠", "duration": "1周", "department": "神经内科", "reasoning": "神经系统症状需要专科评估"},
            {"symptom": "尿频、尿急、尿痛", "duration": "2天", "department": "泌尿外科", "reasoning": "泌尿道感染可能"},
            {"symptom": "关节疼痛、肿胀", "duration": "1月", "department": "风湿免疫科", "reasoning": "需排查自身免疫性疾病"},
            {"symptom": "皮疹、瘙痒、过敏", "duration": "3天", "department": "皮肤科", "reasoning": "皮肤病变"},
            {"symptom": "视力下降、眼前黑影", "duration": "1周", "department": "眼科", "reasoning": "眼科症状需要专科检查"},
            {"symptom": "一般健康咨询", "duration": "无", "department": "全科医学科", "reasoning": "全科可进行初步评估和分诊"},
        ]

        pairs = []
        for i in range(count):
            template = random.choice(templates)
            triage = random.choice(triage_data)

            instruction = template["instruction"].format(**triage)
            input_text = template["input"].format(**triage)
            output = template["output"].format(**triage)

            pairs.append(InstructionPair(
                instruction=instruction,
                input=input_text,
                output=output,
                category="科室分诊建议",
                source="template",
                metadata={"template_type": "triage", "seed": i}
            ))

        return pairs

    def _generate_safety_templates(self, count: int) -> List[InstructionPair]:
        """生成医疗安全问答类数据（含DPO负样本）。"""
        templates = [
            {
                "instruction": "空腹血糖6.5，应该怎么吃药？",
                "input": "空腹血糖6.5mmol/L",
                "safe_output": "您的空腹血糖为6.5mmol/L，超过正常上限6.1，属于空腹血糖受损。建议：1. 咨询内分泌科医生进行进一步评估；2. 可考虑完善糖化血红蛋白检测；3. 注意饮食控制和适量运动。具体用药方案请遵医嘱。",
                "unsafe_output": "您的空腹血糖偏高，建议每天服用二甲双胍500mg，tid，餐后服用，配合阿卡波糖50mg一起吃。",
                "is_safe": True
            },
            {
                "instruction": "我是不是得了糖尿病？",
                "input": "空腹血糖6.5mmol/L",
                "safe_output": "仅凭空腹血糖6.5mmol/L无法确诊糖尿病。空腹血糖受损（IFG）的诊断标准是6.1-7.0mmol/L，糖尿病诊断需要空腹血糖≥7.0mmol/L或OGTT 2h血糖≥11.1mmol/L。建议完善糖化血红蛋白检测，并咨询内分泌科医生。",
                "unsafe_output": "是的，您的空腹血糖6.5已经是糖尿病了，需要立即开始服药治疗。",
                "is_safe": True
            },
            {
                "instruction": "血压160/100很危险吗？",
                "input": "血压160/100mmHg",
                "safe_output": "您的血压属于2级高血压（160/100mmHg），需要引起重视。建议：1. 尽快心内科就诊；2. 规律监测血压；3. 低盐低脂饮食；4. 避免情绪激动。具体治疗方案请遵医嘱。",
                "unsafe_output": "这个血压不危险，吃点卡托普利就行，每天3次，每次25mg。",
                "is_safe": True
            },
        ]

        safety_data = []
        for template in templates:
            # Safe sample
            safety_data.append(InstructionPair(
                instruction=template["instruction"],
                input=template["input"],
                output=template["safe_output"],
                category="医疗安全问答",
                source="template",
                metadata={"is_safe": True, "template_type": "safety"}
            ))
            # Unsafe sample (for DPO training)
            safety_data.append(InstructionPair(
                instruction=template["instruction"],
                input=template["input"],
                output=template["unsafe_output"],
                category="医疗安全问答",
                source="template",
                metadata={"is_safe": False, "is_dpo_negative": True, "template_type": "safety"}
            ))

        # Fill to requested count
        pairs = []
        for i in range(count):
            pairs.append(random.choice(safety_data))

        return pairs

    # ========== 数据处理 ==========

    def filter_by_safety(self, pairs: List[InstructionPair]) -> List[InstructionPair]:
        """
        安全过滤。

        过滤掉包含以下内容的样本:
        - 具体用药剂量
        - 替代医生诊断
        - 不当医疗建议

        Args:
            pairs: 原始数据对

        Returns:
            过滤后的数据对
        """
        danger_patterns = [
            r'\d+\s*mg',  # 具体剂量
            r'\d+\s*ml',
            r'每天\d+次',
            r'每次\d+片',
            r'你就是.*病',
            r'确诊.*癌症',
        ]

        filtered = []
        for pair in pairs:
            text = pair.instruction + pair.input + pair.output

            is_safe = True
            for pattern in danger_patterns:
                if re.search(pattern, text):
                    # 检查是否已被标记为unsafe且用于DPO训练
                    if pair.metadata.get("is_dpo_negative"):
                        continue
                    is_safe = False
                    break

            if is_safe:
                filtered.append(pair)

        return filtered

    def deduplicate(self, pairs: List[InstructionPair]) -> List[InstructionPair]:
        """
        去重。

        基于instruction的相似度去重。

        Args:
            pairs: 原始数据对

        Returns:
            去重后的数据对
        """
        seen = set()
        unique_pairs = []

        for pair in pairs:
            # 使用instruction的前50字符作为去重键
            key = pair.instruction[:50].lower().strip()

            if key not in seen:
                seen.add(key)
                unique_pairs.append(pair)

        return unique_pairs

    def add_diversity_variants(
        self,
        pairs: List[InstructionPair],
        variants_per_sample: int = 2
    ) -> List[InstructionPair]:
        """
        添加多样性变体。

        通过同义词替换、句式变换等方式增加数据多样性。

        Args:
            pairs: 原始数据对
            variants_per_sample: 每个样本生成的变体数量

        Returns:
            增强后的数据对
        """
        synonyms = {
            "偏高": ["偏高", "升高", "超标", "偏高一些"],
            "偏低": ["偏低", "降低", "偏低一些"],
            "正常": ["正常", "在正常范围", "没有问题"],
            "建议": ["建议", "推荐", "提倡"],
            "检查": ["检查", "检测", "化验"],
        }

        new_pairs = []

        for pair in pairs:
            for _ in range(variants_per_sample):
                new_instruction = pair.instruction
                new_output = pair.output

                # 简单同义词替换
                for word, alternatives in synonyms.items():
                    if word in new_instruction:
                        new_instruction = new_instruction.replace(
                            word,
                            random.choice(alternatives)
                        )
                    if word in new_output:
                        new_output = new_output.replace(
                            word,
                            random.choice(alternatives)
                        )

                if new_instruction != pair.instruction or new_output != pair.output:
                    new_pairs.append(InstructionPair(
                        instruction=new_instruction,
                        input=pair.input,
                        output=new_output,
                        category=pair.category,
                        source=pair.source,
                        metadata={**pair.metadata, "is_variant": True}
                    ))

        return pairs + new_pairs

    # ========== 主流程 ==========

    def run(self) -> List[InstructionPair]:
        """
        运行完整的数据增强流程。

        Returns:
            增强后的数据集
        """
        self._report_progress(0.0, "开始数据增强...")

        all_pairs = []
        target_per_category = self.config.target_size // len(self.config.categories)

        # 根据配置选择生成方式
        if self.config.source == "llm":
            # LLM 扩展生成（使用 MiniMax API）
            from app.service.llm_expander import LLMExpander
            expander = LLMExpander()
            all_pairs = self._run_llm_expansion(target_per_category, expander)
        else:
            # 模板生成（纯本地，无 API 调用）
            for i, category in enumerate(self.config.categories):
                self._report_progress(
                    i / len(self.config.categories),
                    f"生成{category}类数据..."
                )
                pairs = self.generate_from_template(category, target_per_category)
                all_pairs.extend(pairs)

            self._report_progress(0.6, "数据去重...")
            all_pairs = self.deduplicate(all_pairs)

            self._report_progress(0.7, "安全过滤...")
            all_pairs = self.filter_by_safety(all_pairs)

            self._report_progress(0.8, "添加多样性变体...")
            all_pairs = self.add_diversity_variants(all_pairs, variants_per_sample=1)

        self._report_progress(0.95, "格式化输出...")
        self._report_progress(1.0, f"数据增强完成，共{len(all_pairs)}条数据")

        self._pairs = all_pairs
        return all_pairs

    def _run_llm_expansion(self, per_category: int, expander) -> List[InstructionPair]:
        """
        使用 LLM 扩展生成数据。

        Args:
            per_category: 每类生成数量
            expander: LLMExpander 实例

        Returns:
            生成的指令-响应对列表
        """
        all_pairs = []

        # 各类别的模板配置
        category_configs = {
            "体检报告解读": [
                {
                    "template": "请解读这份体检报告中{metric}指标的意义。\n{metric}: {value} {unit}, 参考范围: {reference}",
                    "placeholders": {
                        "metric": ["空腹血糖", "总胆固醇", "甘油三酯", "血压", "血红蛋白", "白细胞", "血小板", "谷丙转氨酶", "谷草转氨酶", "肌酐", "尿素氮", "尿酸"],
                        "value": ["6.5", "5.8", "7.2", "6.0", "5.5", "8.0", "5.0", "5.2", "4.8", "6.3", "5.9", "7.5", "8.5"],
                        "unit": ["mmol/L", "mmol/L", "mmol/L", "g/L", "g/L", "10^9/L", "10^9/L", "U/L", "U/L", "μmol/L", "mmol/L", "μmol/L"],
                        "reference": ["3.9-6.1", "<5.2", "<1.7", "<140/90", "130-175", "4-10", "100-300", "<40", "<40", "44-133", "2.6-7.5", "208-428"],
                    },
                },
            ],
            "指标异常问询": [
                {
                    "template": "{metric}偏高饮食需要注意什么？",
                    "placeholders": {
                        "metric": ["空腹血糖", "总胆固醇", "甘油三酯", "血压", "尿酸", "同型半胱氨酸"],
                    },
                },
            ],
            "科室分诊建议": [
                {
                    "template": "我最近出现{symptom}，应该挂什么科？",
                    "symptoms": [
                        "多饮、多尿、体重下降", "心悸、胸闷、活动后气促",
                        "腹痛、腹胀、恶心呕吐", "咳嗽、咳痰、发热",
                        "头痛、头晕、失眠", "尿频、尿急、尿痛",
                        "关节疼痛、肿胀", "皮疹、瘙痒、过敏",
                        "视力下降、眼前黑影", "一般健康咨询",
                        "胸痛、压榨感", "腹胀、乏力、食欲不振",
                        "体重明显下降", "颈部肿块",
                    ],
                },
            ],
            "医疗安全问答": [
                {
                    "safe_outputs": [
                        "您的空腹血糖为6.5mmol/L，超过正常上限6.1，属于空腹血糖受损。建议：1. 咨询内分泌科医生进行进一步评估；2. 可考虑完善糖化血红蛋白检测；3. 注意饮食控制和适量运动。具体用药方案请遵医嘱。",
                        "您的血压属于2级高血压（160/100mmHg），需要引起重视。建议：1. 尽快心内科就诊；2. 规律监测血压；3. 低盐低脂饮食；4. 避免情绪激动。具体治疗方案请遵医嘱。",
                        "您的CEA为8.5ng/mL，偏高（参考值<5.0）。建议：1. 进一步完善检查排除肿瘤可能；2. 胃肠镜、CT等检查；3. 定期复查。具体诊断请遵医嘱。",
                    ],
                },
            ],
        }

        categories = list(category_configs.keys())
        for i, category in enumerate(categories):
            self._report_progress(i / len(categories), f"LLM生成: {category}...")

            config = category_configs[category]
            per_template = per_category // len(config)

            if category == "体检报告解读":
                for tmpl in config:
                    pairs = expander.expand_examination(
                        template=tmpl["template"],
                        placeholders=tmpl["placeholders"],
                        count=per_template,
                    )
                    all_pairs.extend([self._expansion_to_pair(p) for p in pairs])

            elif category == "指标异常问询":
                for tmpl in config:
                    pairs = expander.expand_metric_query(
                        template=tmpl["template"],
                        placeholders=tmpl["placeholders"],
                        count=per_template,
                    )
                    all_pairs.extend([self._expansion_to_pair(p) for p in pairs])

            elif category == "科室分诊建议":
                for tmpl in config:
                    pairs = expander.expand_triage(
                        template=tmpl["template"],
                        symptoms=tmpl["symptoms"],
                        count=per_template,
                    )
                    all_pairs.extend([self._expansion_to_pair(p) for p in pairs])

            elif category == "医疗安全问答":
                for tmpl in config:
                    pairs = expander.expand_safety_pair(
                        safe_outputs=tmpl["safe_outputs"],
                        count=per_template,
                    )
                    all_pairs.extend([self._expansion_to_pair(p) for p in pairs])

        # 安全过滤（仅对非DPO负样本）
        self._report_progress(0.9, "安全过滤...")
        all_pairs = self.filter_by_safety(all_pairs)

        # 去重
        self._report_progress(0.95, "数据去重...")
        all_pairs = self.deduplicate(all_pairs)

        return all_pairs

    def _expansion_to_pair(self, exp: "ExpansionResult") -> "InstructionPair":
        """将 LLMExpander 的 ExpansionResult 转换为 InstructionPair。"""
        return InstructionPair(
            instruction=exp.instruction,
            input=exp.input,
            output=exp.output,
            category=exp.category,
            source="llm",
            metadata=exp.metadata,
        )

    def save(self, path: Optional[str] = None) -> str:
        """
        保存数据集到文件。

        Args:
            path: 保存路径，默认使用配置的output_path

        Returns:
            保存的文件路径
        """
        save_path = path or self.config.output_path

        # 确保目录存在
        import os
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        # 输出为 JSONL 格式
        with open(save_path, 'w', encoding='utf-8') as f:
            for pair in self._pairs:
                line = json.dumps(pair.to_dict(), ensure_ascii=False)
                f.write(line + "\n")

        return save_path

    def load(self, path: str) -> List[InstructionPair]:
        """
        从文件加载数据集（支持 JSONL 和 JSON 数组格式）。

        Args:
            path: 数据文件路径

        Returns:
            加载的数据集
        """
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                self._pairs = []
                return self._pairs

            # 尝试 JSONL 格式（每行一个 JSON 对象）
            if content.startswith("{"):
                lines = content.split("\n")
                data = [json.loads(line) for line in lines if line.strip()]
            else:
                # 尝试 JSON 数组格式
                data = json.loads(content)
                if isinstance(data, list):
                    pass
                else:
                    data = [data]

        self._pairs = [InstructionPair.from_dict(d) for d in data]
        return self._pairs

    def get_stats(self) -> Dict[str, Any]:
        """
        获取数据集统计信息。

        Returns:
            统计信息字典
        """
        if not self._pairs:
            return {"total": 0}

        stats = {
            "total": len(self._pairs),
            "by_category": {},
            "by_source": {},
            "safe_ratio": 0.0,
        }

        for pair in self._pairs:
            # 按类别统计
            stats["by_category"][pair.category] = \
                stats["by_category"].get(pair.category, 0) + 1

            # 按来源统计
            stats["by_source"][pair.source] = \
                stats["by_source"].get(pair.source, 0) + 1

        # 计算安全比例
        safe_count = sum(1 for p in self._pairs if p.metadata.get("is_safe", True))
        stats["safe_ratio"] = safe_count / len(self._pairs) if self._pairs else 0.0

        return stats
