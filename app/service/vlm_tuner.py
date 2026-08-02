"""Coordinate-aware VLM SFT/QLoRA training utilities."""

from __future__ import annotations

import json
import inspect
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Pre-load the concrete SFT module when available.  This keeps TRL's lazy
# module compatible with environments that replace transformers classes while
# constructing a trainer in tests; failures are deferred until ``train``.
try:  # pragma: no cover - availability depends on the optional training stack
    import trl.trainer.sft_trainer  # noqa: F401
except Exception:
    pass


@dataclass
class VLMTunerConfig:
    base_model: str = "Qwen/Qwen2-VL-2B-Instruct"
    dataset_path: str = "data/sft/training_data.jsonl"
    output_dir: str = "outputs/vlm/sft"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 2
    max_seq_length: int = 1024
    use_qlora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    train_projector: bool = True
    seed: int = 42
    disable_gradient_checkpointing: bool = False
    remove_unused_columns: bool = False
    group_by_length: bool = False
    dataloader_num_workers: int = 0
    push_to_hub: bool = False
    hub_model_id: str = ""
    hub_token: str = ""


@dataclass
class VLMTunerStats:
    total_samples: int = 0
    trainable_parameters: int = 0
    total_parameters: int = 0
    estimated_qlora_params: int = 0
    peak_gpu_memory_gb: float = 0.0


def coordinate_prefix(sample: dict[str, Any]) -> str:
    """Serialize OCR/page coordinates as stable text prefixes for SFT."""
    values = sample.get("metrics") or sample.get("regions") or []
    prefixes: list[str] = []
    for item in values:
        if not isinstance(item, dict):
            continue
        normalized = item.get("bbox_normalized") or item.get("bbox")
        if not isinstance(normalized, (list, tuple)) or len(normalized) != 4:
            continue
        try:
            bbox = ",".join(f"{float(value):.2f}" for value in normalized)
        except (TypeError, ValueError):
            continue
        page = item.get("page_number") or item.get("page") or 1
        text = item.get("evidence_text") or item.get("text") or item.get("metric_name") or "region"
        prefixes.append(f"[PAGE={page}][BBOX={bbox}] {text}")
    return "\n".join(prefixes)


def build_sft_text(sample: dict[str, Any]) -> str:
    instruction = str(sample.get("instruction") or sample.get("prompt") or "").strip()
    user_input = str(sample.get("input") or "").strip()
    output = str(sample.get("output") or sample.get("answer") or "").strip()
    prefix = coordinate_prefix(sample)
    sections = [part for part in (prefix, instruction, user_input, output) if part]
    return "\n".join(sections)


class VLMTuner:
    def __init__(self, config: VLMTunerConfig | None = None) -> None:
        self.config = config or VLMTunerConfig()
        self._model: Any = None
        self._processor: Any = None
        self._stats: VLMTunerStats | None = None

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
                    target_modules=cfg.lora_target_modules,
                    task_type="CAUSAL_LM",
                ),
            )
        if cfg.train_projector:
            for name, parameter in self._model.named_parameters():
                if any(token in name.lower() for token in ("projector", "visual.merger", "multi_modal_projector")):
                    parameter.requires_grad = True

    def _load_dataset(self):
        from datasets import load_dataset

        dataset = load_dataset("json", data_files=self.config.dataset_path, split="train")
        if isinstance(getattr(dataset, "column_names", None), list):
            return dataset.map(lambda sample: {"text": build_sft_text(sample)})
        return dataset

    def train(self) -> VLMTunerStats:
        from transformers import TrainingArguments, set_seed
        # Import from the concrete module so both current TRL and test doubles
        # patched at ``trl.trainer.sft_trainer.SFTTrainer`` are respected.
        from trl.trainer.sft_trainer import SFTTrainer

        set_seed(self.config.seed)
        self._setup_model()
        dataset = self._load_dataset()
        cfg = self.config
        args = TrainingArguments(
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
            remove_unused_columns=cfg.remove_unused_columns,
            group_by_length=cfg.group_by_length,
            dataloader_num_workers=cfg.dataloader_num_workers,
            push_to_hub=cfg.push_to_hub,
            hub_model_id=cfg.hub_model_id or None,
            hub_token=cfg.hub_token or None,
            bf16=False,
            gradient_checkpointing=not cfg.disable_gradient_checkpointing,
            report_to=["none"],
        )
        trainer_kwargs: dict[str, Any] = {
            "model": self._model,
            "args": args,
            "train_dataset": dataset,
            "processing_class": self._processor,
        }
        supported = (
            {"dataset_text_field", "max_seq_length"}
            if not inspect.isclass(SFTTrainer)
            else inspect.signature(SFTTrainer.__init__).parameters
        )
        if "dataset_text_field" in supported:
            trainer_kwargs["dataset_text_field"] = "text"
        if "max_seq_length" in supported:
            trainer_kwargs["max_seq_length"] = cfg.max_seq_length

        trainer = SFTTrainer(
            **trainer_kwargs,
        )
        trainer.train()
        trainable, total = self._parameter_counts(self._model)
        lora_params = sum(
            parameter.numel() for name, parameter in self._model.named_parameters() if "lora_" in name
        )
        self._stats = VLMTunerStats(len(dataset), trainable, total, lora_params)
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

    def get_stats(self) -> VLMTunerStats | None:
        return self._stats
