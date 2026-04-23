"""
VLM Fine-tuning Service (SFT / QLoRA)
Supports supervised fine-tuning and parameter-efficient fine-tuning with LoRA/QLoRA.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    from transformers import TrainingArguments

logger = logging.getLogger(__name__)


@dataclass
class VLMTunerConfig:
    """SFT / QLoRA configuration."""

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
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    seed: int = 42
    disable_gradient_checkpointing: bool = False
    remove_unused_columns: bool = False
    group_by_length: bool = False
    label_names: list[str] = field(default_factory=lambda: ["labels"])
    dataloader_num_workers: int = 0
    push_to_hub: bool = False
    hub_model_id: str = ""
    hub_token: str = ""


@dataclass
class VLMTunerStats:
    """Training statistics."""

    total_samples: int = 0
    trainable_parameters: int = 0
    total_parameters: int = 0
    estimated_qlora_params: int = 0
    peak_gpu_memory_gb: float = 0.0


class VLMTuner:
    """SFT / QLoRA fine-tuner for Vision-Language Models."""

    def __init__(self, config: VLMTunerConfig | None = None) -> None:
        self.config = config or VLMTunerConfig()
        self._model: Any = None
        self._processor: Any = None
        self._tokenizer: Any = None
        self._stats: VLMTunerStats | None = None

    def _setup_model(self) -> None:
        """Load model, processor, and tokenizer."""
        import torch
        from transformers import AutoProcessor, AutoTokenizer

        cfg = self.config
        logger.info(f"Loading base model: {cfg.base_model}")

        self._processor = AutoProcessor.from_pretrained(
            cfg.base_model, trust_remote_code=True
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            cfg.base_model, trust_remote_code=True
        )
        self._tokenizer.padding_side = "right"
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        if cfg.use_qlora:
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self._model = self._load_vlm_model(
                cfg.base_model,
                quantization_config=bnb_config,
                device_map="auto",
            )
        else:
            self._model = self._load_vlm_model(
                cfg.base_model,
                device_map="auto",
            )

        self._model.config.use_cache = False

        # Apply LoRA adapters
        if cfg.use_qlora:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

            self._model = prepare_model_for_kbit_training(self._model)
            lora_cfg = LoraConfig(
                r=cfg.lora_rank,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules=cfg.lora_target_modules,
                task_type="SEQ_2_SEQ_LM",
            )
            self._model = get_peft_model(self._model, lora_cfg)

        self._model.print_trainable_parameters()
        logger.info("Model initialisation complete")

    def _load_vlm_model(self, model_name: str, **kwargs: Any) -> Any:
        """Load VLM model — tries Qwen2-VL then generic VLM."""
        try:
            # Qwen2-VL models
            from transformers import Qwen2VLForConditionalGeneration

            return Qwen2VLForConditionalGeneration.from_pretrained(
                model_name, trust_remote_code=True, **kwargs
            )
        except Exception:
            pass
        # Fallback: use model-specific auto class
        from transformers import AutoModelForVision2Seq

        return AutoModelForVision2Seq.from_pretrained(
            model_name, trust_remote_code=True, **kwargs
        )

    def _load_dataset(self) -> Any:
        """Load and format SFT dataset from JSONL."""
        import datasets

        cfg = self.config
        dataset = datasets.load_dataset("json", data_files=cfg.dataset_path, split="train")

        def format_sample(sample: dict[str, Any]) -> dict[str, Any]:
            text = sample.get("instruction", "") + "\n" + sample.get("output", "")
            return {"text": text, "image": sample.get("image_path", "")}

        dataset = dataset.map(format_sample)
        return dataset

    def train(self) -> VLMTunerStats:
        """Run SFT / QLoRA training."""
        import torch
        from transformers import TrainingArguments, set_seed
        from trl import SFTTrainer

        cfg = self.config
        set_seed(cfg.seed)

        self._setup_model()
        dataset = self._load_dataset()

        training_args = TrainingArguments(
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
            label_names=cfg.label_names,
            dataloader_num_workers=cfg.dataloader_num_workers,
            push_to_hub=cfg.push_to_hub,
            hub_model_id=cfg.hub_model_id if cfg.hub_model_id else None,
            hub_token=cfg.hub_token if cfg.hub_token else None,
            bf16=True,
            gradient_checkpointing=not cfg.disable_gradient_checkpointing,
            optim="paged_adamw_8bit",
            report_to=["none"],
        )

        trainer = SFTTrainer(
            model=self._model,
            args=training_args,
            train_dataset=dataset,
            processing_class=self._processor,
        )

        logger.info("Starting training ...")
        trainer.train()

        # Collect stats
        stats = VLMTunerStats()
        stats.total_samples = len(dataset)
        trainable, total = self._model.get_trainable_parameter_counts()
        stats.trainable_parameters = trainable
        stats.total_parameters = total
        if cfg.use_qlora:
            stats.estimated_qlora_params = (
                cfg.lora_rank
                * sum(
                    sum(p.numel() for p in self._model.parameters() if "lora_" in n)
                    for n, _ in self._model.named_parameters()
                )
                // cfg.lora_rank
            )
        self._stats = stats
        logger.info(f"Training complete. Stats: {stats}")
        return stats

    def save_model(self, output_path: str | Path | None = None) -> Path:
        """Save the fine-tuned model and adapter."""
        if self._model is None:
            raise RuntimeError("Model not trained yet")
        path = Path(output_path or self.config.output_dir) / "final"
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(str(path))
        self._processor.save_pretrained(str(path))
        logger.info(f"Model saved to {path}")
        return path

    def get_stats(self) -> VLMTunerStats | None:
        """Return training statistics."""
        return self._stats


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to JSON config")
    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            cfg_dict = json.load(f)
        cfg = VLMTunerConfig(**cfg_dict)
    else:
        cfg = VLMTunerConfig()

    tuner = VLMTuner(cfg)
    tuner.train()
    tuner.save_model()
