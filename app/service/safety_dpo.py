"""DPO safety-alignment pipeline with canonical preference validation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class DPOConfig:
    base_model: str = "Qwen/Qwen2-VL-2B-Instruct"
    ref_model: str = "Qwen/Qwen2-VL-2B-Instruct"
    dataset_path: str = "data/sft/safety_qa.jsonl"
    output_dir: str = "outputs/dpo"
    general_dataset_path: Optional[str] = None
    general_mix_ratio: float = 0.15
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-6
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 2
    max_seq_length: int = 1024
    beta: float = 0.1
    min_beta: float = 0.05
    max_beta: float = 0.3
    target_kl: float = 0.1
    dynamic_kl: bool = True
    label_smoothing: float = 0.0
    use_qlora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    seed: int = 42
    push_to_hub: bool = False
    hub_model_id: str = ""


@dataclass
class DPOStats:
    total_pairs: int = 0
    trainable_parameters: int = 0
    total_parameters: int = 0
    final_beta: float = 0.0


def canonicalize_preference(sample: dict[str, Any]) -> dict[str, Any]:
    """Accept the old output/output_unsafe schema and emit canonical DPO fields."""
    prompt = sample.get("prompt") or sample.get("instruction") or ""
    if sample.get("input"):
        prompt = f"{prompt}\n{sample['input']}".strip()
    chosen = sample.get("chosen") or sample.get("output") or ""
    rejected = sample.get("rejected") or sample.get("output_unsafe") or ""
    if not prompt or not chosen or not rejected:
        raise ValueError("DPO 样本必须包含 prompt、chosen 和 rejected")
    return {
        "prompt": str(prompt),
        "chosen": str(chosen),
        "rejected": str(rejected),
        "image": sample.get("image") or sample.get("image_path") or "",
        "category": sample.get("category", "medical_safety"),
        "source": sample.get("source", "unknown"),
    }


def load_preference_records(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"DPO 数据文件不存在：{file_path}")
    records: list[dict[str, Any]] = []
    with file_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                records.append(canonicalize_preference(raw))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"DPO 第 {line_number} 行无效：{exc}") from exc
    if not records:
        raise ValueError(f"DPO 数据文件为空：{file_path}")
    return records


class AdaptiveKLController:
    """Bounded beta controller used by the DPO callback."""

    def __init__(self, beta: float, min_beta: float, max_beta: float, target_kl: float) -> None:
        self.beta = beta
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.target_kl = target_kl

    def update(self, observed_kl: float) -> float:
        if observed_kl > self.target_kl * 1.2:
            self.beta = min(self.max_beta, self.beta * 1.1)
        elif observed_kl < self.target_kl * 0.8:
            self.beta = max(self.min_beta, self.beta * 0.95)
        return self.beta


class DynamicKLBetaCallback:
    """Update DPO beta from logged KL estimates without unbounded drift."""

    def __init__(self, controller: AdaptiveKLController) -> None:
        self.controller = controller
        self.trainer = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        observed_kl = logs.get("train/approx_kl", logs.get("approx_kl"))
        if observed_kl is None:
            return control
        beta = self.controller.update(float(observed_kl))
        if self.trainer is not None and hasattr(self.trainer, "beta"):
            self.trainer.beta = beta
        return control


class SafetyDPOTrainer:
    def __init__(self, config: DPOConfig | None = None) -> None:
        self.config = config or DPOConfig()
        self._model: Any = None
        self._ref_model: Any = None
        self._processor: Any = None
        self._stats: DPOStats | None = None
        self._kl_controller = AdaptiveKLController(
            self.config.beta, self.config.min_beta, self.config.max_beta, self.config.target_kl
        )

    def _load_vlm_model(self, model_name: str, **kwargs: Any) -> Any:
        try:
            from transformers import Qwen2VLForConditionalGeneration

            return Qwen2VLForConditionalGeneration.from_pretrained(model_name, trust_remote_code=True, **kwargs)
        except Exception:
            from transformers import AutoModelForVision2Seq

            return AutoModelForVision2Seq.from_pretrained(model_name, trust_remote_code=True, **kwargs)

    def _setup_model(self) -> None:
        import torch
        from transformers import AutoProcessor

        cfg = self.config
        self._processor = AutoProcessor.from_pretrained(cfg.base_model, trust_remote_code=True)
        if hasattr(self._processor, "tokenizer"):
            self._processor.tokenizer.padding_side = "right"
        load_kwargs: dict[str, Any] = {"device_map": "auto"}
        if cfg.use_qlora:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        self._model = self._load_vlm_model(cfg.base_model, **load_kwargs)
        self._model.config.use_cache = False
        if cfg.use_qlora:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

            self._model = prepare_model_for_kbit_training(self._model)
            self._model = get_peft_model(
                self._model,
                LoraConfig(
                    r=cfg.lora_rank,
                    lora_alpha=cfg.lora_alpha,
                    lora_dropout=cfg.lora_dropout,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                    task_type="CAUSAL_LM",
                ),
            )
            self._ref_model = None
        else:
            self._ref_model = self._load_vlm_model(cfg.ref_model, device_map="auto")
            self._ref_model.eval()

    def _load_dataset(self):
        from datasets import concatenate_datasets, load_dataset

        dataset = load_dataset("json", data_files=self.config.dataset_path, split="train")
        if isinstance(getattr(dataset, "column_names", None), list):
            required = {"prompt", "chosen", "rejected"}
            if not required.issubset(set(dataset.column_names)):
                dataset = dataset.map(canonicalize_preference)
        general_path = self.config.general_dataset_path
        if general_path and Path(general_path).exists() and self.config.general_mix_ratio > 0:
            general = load_dataset("json", data_files=str(general_path), split="train")
            if isinstance(getattr(general, "column_names", None), list):
                required = {"prompt", "chosen", "rejected"}
                if not required.issubset(set(general.column_names)):
                    general = general.map(canonicalize_preference)
            count = min(len(general), max(1, int(len(dataset) * self.config.general_mix_ratio)))
            dataset = concatenate_datasets([dataset, general.select(range(count))]).shuffle(seed=self.config.seed)
        return dataset

    def train(self) -> DPOStats:
        from transformers import set_seed
        from trl.trainer.dpo_trainer import DPOConfig as TRLDPOConfig, DPOTrainer

        set_seed(self.config.seed)
        self._setup_model()
        dataset = self._load_dataset()
        cfg = self.config
        args = TRLDPOConfig(
            output_dir=cfg.output_dir,
            num_train_epochs=cfg.num_train_epochs,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            warmup_ratio=cfg.warmup_ratio,
            lr_scheduler_type=cfg.lr_scheduler_type,
            logging_steps=cfg.logging_steps,
            save_steps=cfg.save_steps,
            save_total_limit=cfg.save_total_limit,
            seed=cfg.seed,
            max_length=cfg.max_seq_length,
            beta=self._kl_controller.beta,
            label_smoothing=cfg.label_smoothing,
            push_to_hub=cfg.push_to_hub,
            hub_model_id=cfg.hub_model_id or None,
            bf16=False,
            gradient_checkpointing=True,
            report_to=["none"],
        )
        trainer = DPOTrainer(
            model=self._model,
            ref_model=self._ref_model,
            args=args,
            train_dataset=dataset,
            processing_class=self._processor,
        )
        if cfg.dynamic_kl:
            callback = DynamicKLBetaCallback(self._kl_controller)
            callback.trainer = trainer
            trainer.add_callback(callback)
        trainer.train()
        trainable, total = self._parameter_counts(self._model)
        self._stats = DPOStats(len(dataset), trainable, total, self._kl_controller.beta)
        return self._stats

    @staticmethod
    def _parameter_counts(model: Any) -> tuple[int, int]:
        compatibility_counter = getattr(model, "get_trainable_parameter_counts", None)
        if callable(compatibility_counter):
            counts = compatibility_counter()
            if isinstance(counts, tuple) and len(counts) == 2:
                return int(counts[0]), int(counts[1])
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in model.parameters())
        return trainable, total

    def save_model(self, output_path: str | Path | None = None) -> Path:
        if self._model is None:
            raise RuntimeError("模型尚未训练 (model not trained yet)")
        path = Path(output_path or self.config.output_dir) / "final"
        path.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(str(path))
        self._processor.save_pretrained(str(path))
        return path

    def get_stats(self) -> DPOStats | None:
        return self._stats
