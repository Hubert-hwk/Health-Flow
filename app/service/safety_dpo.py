"""
Safety DPO (Direct Preference Optimization) Training.
Uses paired preference data (chosen / rejected responses) to align VLM safety behaviour.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class DPOConfig:
    """DPO training configuration."""

    base_model: str = "Qwen/Qwen2-VL-2B-Instruct"
    ref_model: str = "Qwen/Qwen2-VL-2B-Instruct"
    dataset_path: str = "data/dpo/preference_data.jsonl"
    output_dir: str = "outputs/dpo"
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
    label_smoothing: float = 0.0
    seed: int = 42
    push_to_hub: bool = False
    hub_model_id: str = ""


@dataclass
class DPOStats:
    """DPO training statistics."""

    total_pairs: int = 0
    trainable_parameters: int = 0
    total_parameters: int = 0


class SafetyDPOTrainer:
    """Direct Preference Optimization for safety alignment."""

    def __init__(self, config: DPOConfig | None = None) -> None:
        self.config = config or DPOConfig()
        self._model: Any = None
        self._ref_model: Any = None
        self._processor: Any = None
        self._stats: DPOStats | None = None

    def _load_vlm_model(self, model_name: str, **kwargs: Any) -> Any:
        """Load VLM model — tries Qwen2-VL then generic VLM."""
        try:
            from transformers import Qwen2VLForConditionalGeneration

            return Qwen2VLForConditionalGeneration.from_pretrained(
                model_name, trust_remote_code=True, **kwargs
            )
        except Exception:
            pass
        from transformers import AutoModelForVision2Seq

        return AutoModelForVision2Seq.from_pretrained(
            model_name, trust_remote_code=True, **kwargs
        )

    def _setup_model(self) -> None:
        """Load policy model and reference (SFT) model."""
        from transformers import AutoProcessor

        cfg = self.config
        logger.info(f"Loading policy model: {cfg.base_model}")

        self._processor = AutoProcessor.from_pretrained(
            cfg.base_model, trust_remote_code=True
        )
        self._processor.tokenizer.padding_side = "right"

        self._model = self._load_vlm_model(cfg.base_model, device_map="auto")
        self._model.config.use_cache = False

        logger.info(f"Loading reference model: {cfg.ref_model}")
        self._ref_model = self._load_vlm_model(cfg.ref_model, device_map="auto")
        self._ref_model.eval()

        self._model.print_trainable_parameters()

    def _load_dataset(self) -> Any:
        """Load preference dataset (JSONL with chosen / rejected fields)."""
        import datasets

        cfg = self.config
        dataset = datasets.load_dataset("json", data_files=cfg.dataset_path, split="train")

        def format_sample(sample: dict[str, Any]) -> dict[str, Any]:
            return {
                "prompt": sample.get("instruction", ""),
                "chosen": sample.get("chosen", ""),
                "rejected": sample.get("rejected", ""),
                "image": sample.get("image_path", ""),
            }

        return dataset.map(format_sample)

    def train(self) -> DPOStats:
        """Run DPO training."""
        from transformers import set_seed
        from trl import DPOConfig, DPOTrainer

        cfg = self.config
        set_seed(cfg.seed)

        self._setup_model()
        dataset = self._load_dataset()

        training_args = DPOConfig(
            output_dir=cfg.output_dir,
            num_train_epochs=cfg.num_train_epochs,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            lr_scheduler_type=cfg.lr_scheduler_type,
            logging_steps=cfg.logging_steps,
            save_steps=cfg.save_steps,
            save_total_limit=cfg.save_total_limit,
            seed=cfg.seed,
            max_length=cfg.max_seq_length,
            beta=cfg.beta,
            label_smoothing=cfg.label_smoothing,
            push_to_hub=cfg.push_to_hub,
            hub_model_id=cfg.hub_model_id if cfg.hub_model_id else None,
            bf16=True,
            gradient_checkpointing=True,
            optim="paged_adamw_8bit",
            report_to=["none"],
        )

        trainer = DPOTrainer(
            model=self._model,
            ref_model=self._ref_model,
            args=training_args,
            train_dataset=dataset,
            processing_class=self._processor,
        )

        logger.info("Starting DPO training ...")
        trainer.train()

        stats = DPOStats()
        stats.total_pairs = len(dataset)
        trainable, total = self._model.get_trainable_parameter_counts()
        stats.trainable_parameters = trainable
        stats.total_parameters = total
        self._stats = stats
        logger.info(f"DPO training complete. Stats: {stats}")
        return stats

    def save_model(self, output_path: str | Path | None = None) -> Path:
        """Save the DPO-adapted model."""
        if self._model is None:
            raise RuntimeError("Model not trained yet")
        path = Path(output_path or self.config.output_dir) / "final"
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(str(path))
        self._processor.save_pretrained(str(path))
        logger.info(f"DPO model saved to {path}")
        return path

    def get_stats(self) -> DPOStats | None:
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
        cfg = DPOConfig(**cfg_dict)
    else:
        cfg = DPOConfig()

    trainer = SafetyDPOTrainer(cfg)
    trainer.train()
    trainer.save_model()
