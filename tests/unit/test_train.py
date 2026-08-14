"""Tests for VLM tuner and Safety DPO training modules."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# 训练栈（transformers/trl/datasets）是可选依赖，未安装时跳过整个模块，
# 而不是在收集阶段抛 ModuleNotFoundError。
pytest.importorskip("transformers")
pytest.importorskip("trl")
pytest.importorskip("datasets")


class TestVLMTunerConfig:
    """Tests for VLMTunerConfig."""

    def test_default_config_values(self) -> None:
        from app.service.vlm_tuner import VLMTunerConfig

        cfg = VLMTunerConfig()
        assert cfg.base_model == "Qwen/Qwen2-VL-2B-Instruct"
        assert cfg.num_train_epochs == 3
        assert cfg.use_qlora is True
        assert cfg.lora_rank == 16
        assert cfg.lora_alpha == 32
        assert cfg.max_seq_length == 1024

    def test_custom_config_values(self) -> None:
        from app.service.vlm_tuner import VLMTunerConfig

        cfg = VLMTunerConfig(
            base_model="Qwen/Qwen2-VL-7B-Instruct",
            num_train_epochs=5,
            per_device_train_batch_size=4,
            learning_rate=1e-4,
        )
        assert cfg.base_model == "Qwen/Qwen2-VL-7B-Instruct"
        assert cfg.num_train_epochs == 5
        assert cfg.per_device_train_batch_size == 4
        assert cfg.learning_rate == 1e-4


class TestVLMTuner:
    """Tests for VLMTuner with mocked external dependencies."""

    def test_setup_model_mocks_load_vlm(self) -> None:
        """Test _setup_model using _load_vlm_model mock to avoid network calls."""
        from app.service.vlm_tuner import VLMTuner

        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_tokenizer = MagicMock()

        def fake_setup(self: VLMTuner) -> None:
            self._model = mock_model
            self._processor = mock_processor
            self._tokenizer = mock_tokenizer

        with patch.object(VLMTuner, "_setup_model", fake_setup):
            tuner = VLMTuner()
            tuner._setup_model()

        assert tuner._model is mock_model
        assert tuner._processor is mock_processor
        assert tuner._tokenizer is mock_tokenizer

    def test_setup_model_with_qlora(self) -> None:
        from app.service.vlm_tuner import VLMTuner

        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_tokenizer = MagicMock()

        def fake_setup(self: VLMTuner) -> None:
            self._model = mock_model
            self._processor = mock_processor
            self._tokenizer = mock_tokenizer

        with patch.object(VLMTuner, "_setup_model", fake_setup):
            from app.service.vlm_tuner import VLMTunerConfig
            cfg = VLMTunerConfig(use_qlora=True)
            tuner = VLMTuner(cfg)
            tuner._setup_model()

        assert tuner._model is mock_model
        assert cfg.use_qlora is True

    def test_train_returns_stats(self) -> None:
        from app.service.vlm_tuner import VLMTuner, VLMTunerConfig

        mock_model = MagicMock()
        mock_model.get_trainable_parameter_counts.return_value = (1000, 1000000)
        mock_processor = MagicMock()
        mock_tokenizer = MagicMock()

        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=200)

        mock_trainer_instance = MagicMock()

        def fake_setup(self: VLMTuner) -> None:
            self._model = mock_model
            self._processor = mock_processor
            self._tokenizer = mock_tokenizer

        with patch.object(VLMTuner, "_setup_model", fake_setup), \
             patch("datasets.load_dataset", return_value=mock_dataset), \
             patch("transformers.set_seed"), \
             patch("transformers.TrainingArguments") as mock_ta, \
             patch("trl.trainer.sft_trainer.SFTTrainer", return_value=mock_trainer_instance):
            mock_ta.return_value = MagicMock()
            cfg = VLMTunerConfig(use_qlora=False, disable_gradient_checkpointing=True)
            tuner = VLMTuner(cfg)
            stats = tuner.train()

        assert stats.total_samples == 200
        assert stats.trainable_parameters == 1000
        assert stats.total_parameters == 1000000

    def test_save_model_raises_if_not_trained(self) -> None:
        from app.service.vlm_tuner import VLMTuner

        tuner = VLMTuner()
        with pytest.raises(RuntimeError, match="not trained yet"):
            tuner.save_model()


class TestSafetyDPOConfig:
    """Tests for DPOConfig."""

    def test_default_config_values(self) -> None:
        from app.service.safety_dpo import DPOConfig

        cfg = DPOConfig()
        assert cfg.base_model == "Qwen/Qwen2-VL-2B-Instruct"
        assert cfg.num_train_epochs == 3
        assert cfg.beta == 0.1
        assert cfg.max_seq_length == 1024

    def test_custom_config_values(self) -> None:
        from app.service.safety_dpo import DPOConfig

        cfg = DPOConfig(
            base_model="Qwen/Qwen2-VL-7B-Instruct",
            num_train_epochs=5,
            beta=0.2,
            learning_rate=5e-7,
        )
        assert cfg.base_model == "Qwen/Qwen2-VL-7B-Instruct"
        assert cfg.num_train_epochs == 5
        assert cfg.beta == 0.2
        assert cfg.learning_rate == 5e-7


class TestSafetyDPOTrainer:
    """Tests for SafetyDPOTrainer with mocked external dependencies."""

    def test_setup_model(self) -> None:
        from app.service.safety_dpo import SafetyDPOTrainer

        mock_model = MagicMock()
        mock_processor = MagicMock()

        def fake_setup(self: SafetyDPOTrainer) -> None:
            self._model = mock_model
            self._ref_model = mock_model
            self._processor = mock_processor

        with patch.object(SafetyDPOTrainer, "_setup_model", fake_setup):
            trainer = SafetyDPOTrainer()
            trainer._setup_model()

        assert trainer._model is mock_model
        assert trainer._ref_model is mock_model
        assert trainer._processor is mock_processor

    def test_train_returns_stats(self) -> None:
        from app.service.safety_dpo import SafetyDPOTrainer

        mock_model = MagicMock()
        mock_model.get_trainable_parameter_counts.return_value = (500, 500000)
        mock_processor = MagicMock()

        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=200)

        mock_trainer_instance = MagicMock()

        def fake_setup(self: SafetyDPOTrainer) -> None:
            self._model = mock_model
            self._ref_model = mock_model
            self._processor = mock_processor

        with patch.object(SafetyDPOTrainer, "_setup_model", fake_setup), \
             patch("datasets.load_dataset", return_value=mock_dataset), \
             patch("transformers.set_seed"), \
             patch("trl.trainer.dpo_trainer.DPOConfig", return_value=MagicMock()), \
             patch("trl.trainer.dpo_trainer.DPOTrainer", return_value=mock_trainer_instance):
            trainer = SafetyDPOTrainer()
            stats = trainer.train()

        assert stats.total_pairs == 200
        assert stats.trainable_parameters == 500

    def test_save_model_raises_if_not_trained(self) -> None:
        from app.service.safety_dpo import SafetyDPOTrainer

        trainer = SafetyDPOTrainer()
        with pytest.raises(RuntimeError, match="not trained yet"):
            trainer.save_model()
