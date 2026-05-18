"""Integration test: run a few training steps and verify loss decreases."""

import pytest
import torch

from accelerate import Accelerator

from ret.config import ExperimentConfig, LLMConfig, DataConfig, EncoderConfig, PredictorConfig, TrainingConfig, LoggingConfig
from ret.training.trainer import JEPATrainer
from ret.utils.seeding import seed_everything


def _make_synthetic_batch(B: int, T: int, vocab_size: int = 50304) -> dict[str, torch.Tensor]:
    return {"input_ids": torch.randint(0, vocab_size, (B, T))}


def _make_padded_batch(B: int, T: int, vocab_size: int = 50304) -> dict[str, torch.Tensor]:
    """Synthetic batch with variable-length sequences (some positions padded)."""
    input_ids = torch.randint(0, vocab_size, (B, T))
    attention_mask = torch.ones(B, T, dtype=torch.long)
    # Make the last portion padding for half the batch
    for i in range(B // 2, B):
        pad_start = T // 2 + i  # variable padding per sample
        attention_mask[i, min(pad_start, T):] = 0
    return {"input_ids": input_ids, "attention_mask": attention_mask}


@pytest.fixture
def config():
    return ExperimentConfig(
        llm=LLMConfig(name="EleutherAI/pythia-160m", layer_index=6, hidden_dim=768),
        data=DataConfig(chunk_length=128),
        encoder=EncoderConfig(type="mlp", z_dim=32, mlp_hidden_dims=[256, 128]),
        predictor=PredictorConfig(mode="autonomous", hidden_dim=128, num_layers=2),
        training=TrainingConfig(
            batch_size=4, max_steps=10, lr=1e-3, warmup_steps=2,
            eval_every=100, checkpoint_every=100, grad_clip=1.0,
        ),
        logging=LoggingConfig(output_dir="/tmp/ret_test"),
    )


def test_training_step_loss_decreases(config):
    """Run 10 steps on synthetic data, verify loss decreases and no NaN."""
    seed_everything(42)
    trainer = JEPATrainer(config, accelerator=Accelerator(split_batches=True))

    losses = []
    for step in range(10):
        batch = _make_synthetic_batch(4, 128)
        metrics = trainer.train_step(batch)
        loss = metrics["train/total_loss"]
        assert not torch.isnan(torch.tensor(loss)), f"NaN loss at step {step}"
        losses.append(loss)

    # Loss should generally decrease (first vs last 3 steps average)
    avg_first = sum(losses[:3]) / 3
    avg_last = sum(losses[-3:]) / 3
    assert avg_last < avg_first, f"Loss did not decrease: {avg_first:.4f} -> {avg_last:.4f}"


def test_training_step_controlled(config):
    """Test controlled predictor mode."""
    config.predictor.mode = "controlled"
    config.predictor.control_dim = 16
    seed_everything(42)
    trainer = JEPATrainer(config, accelerator=Accelerator(split_batches=True))

    batch = _make_synthetic_batch(4, 128)
    metrics = trainer.train_step(batch)
    assert not torch.isnan(torch.tensor(metrics["train/total_loss"]))
    assert metrics["train/pred_loss"] > 0


def test_training_step_transformer_encoder(config):
    """Test with transformer encoder."""
    config.encoder.type = "transformer"
    config.encoder.z_dim = 32
    config.encoder.num_heads = 4
    seed_everything(42)
    trainer = JEPATrainer(config, accelerator=Accelerator(split_batches=True))

    batch = _make_synthetic_batch(4, 128)
    metrics = trainer.train_step(batch)
    assert not torch.isnan(torch.tensor(metrics["train/total_loss"]))


def test_eval_step(config):
    """Test evaluation step produces valid metrics."""
    seed_everything(42)
    trainer = JEPATrainer(config, accelerator=Accelerator(split_batches=True))

    batch = _make_synthetic_batch(4, 128)
    metrics = trainer.eval_step(batch)
    assert "val/pred_loss" in metrics
    assert metrics["val/pred_loss"] > 0


def test_training_step_sparse_encoder(config):
    """Test training with sparse MLP encoder produces valid metrics."""
    config.encoder.type = "mlp"
    config.encoder.z_regularization = "sparse"
    config.encoder.lambda_sparse = 1e-3
    seed_everything(42)
    trainer = JEPATrainer(config, accelerator=Accelerator(split_batches=True))

    batch = _make_synthetic_batch(4, 128)
    metrics = trainer.train_step(batch)
    assert not torch.isnan(torch.tensor(metrics["train/total_loss"]))
    assert "encoder/sparse_loss" in metrics
    assert metrics["encoder/sparse_loss"] >= 0


def test_training_step_vq_encoder(config):
    """Test training with VQ encoder produces valid metrics."""
    config.encoder.type = "mlp"
    config.encoder.z_regularization = "vq"
    config.encoder.codebook_size = 64
    config.encoder.commitment_weight = 0.25
    seed_everything(42)
    trainer = JEPATrainer(config, accelerator=Accelerator(split_batches=True))

    batch = _make_synthetic_batch(4, 128)
    metrics = trainer.train_step(batch)
    assert not torch.isnan(torch.tensor(metrics["train/total_loss"]))
    assert "encoder/vq_loss" in metrics
    assert "encoder/codebook_perplexity" in metrics
    assert metrics["encoder/codebook_perplexity"] > 0


def test_training_step_transformer_sparse(config):
    """Test training with transformer encoder + sparse wrapper."""
    config.encoder.type = "transformer"
    config.encoder.z_dim = 32
    config.encoder.num_heads = 4
    config.encoder.z_regularization = "sparse"
    config.encoder.lambda_sparse = 1e-3
    seed_everything(42)
    trainer = JEPATrainer(config, accelerator=Accelerator(split_batches=True))

    batch = _make_synthetic_batch(4, 128)
    metrics = trainer.train_step(batch)
    assert not torch.isnan(torch.tensor(metrics["train/total_loss"]))
    assert "encoder/sparse_loss" in metrics


def test_training_step_transformer_vq(config):
    """Test training with transformer encoder + VQ wrapper."""
    config.encoder.type = "transformer"
    config.encoder.z_dim = 32
    config.encoder.num_heads = 4
    config.encoder.z_regularization = "vq"
    config.encoder.codebook_size = 64
    config.encoder.commitment_weight = 0.25
    seed_everything(42)
    trainer = JEPATrainer(config, accelerator=Accelerator(split_batches=True))

    batch = _make_synthetic_batch(4, 128)
    metrics = trainer.train_step(batch)
    assert not torch.isnan(torch.tensor(metrics["train/total_loss"]))
    assert "encoder/vq_loss" in metrics
    assert "encoder/codebook_perplexity" in metrics


def test_training_step_with_attention_mask(config):
    """Test training step with variable-length padded sequences."""
    seed_everything(42)
    trainer = JEPATrainer(config, accelerator=Accelerator(split_batches=True))

    batch = _make_padded_batch(4, 128)
    metrics = trainer.train_step(batch)
    assert not torch.isnan(torch.tensor(metrics["train/total_loss"]))
    assert metrics["train/pred_loss"] > 0


def test_training_step_with_attention_mask_transformer(config):
    """Test training step with attention_mask on transformer encoder."""
    config.encoder.type = "transformer"
    config.encoder.z_dim = 32
    config.encoder.num_heads = 4
    seed_everything(42)
    trainer = JEPATrainer(config, accelerator=Accelerator(split_batches=True))

    batch = _make_padded_batch(4, 128)
    metrics = trainer.train_step(batch)
    assert not torch.isnan(torch.tensor(metrics["train/total_loss"]))
    assert metrics["train/pred_loss"] > 0


def test_eval_step_with_attention_mask(config):
    """Test eval step with variable-length padded sequences."""
    seed_everything(42)
    trainer = JEPATrainer(config, accelerator=Accelerator(split_batches=True))

    batch = _make_padded_batch(4, 128)
    metrics = trainer.eval_step(batch)
    assert "val/pred_loss" in metrics
    assert metrics["val/pred_loss"] > 0


# ---------------------------------------------------------------------------
# Cached hidden states mode
# ---------------------------------------------------------------------------

def _make_cached_batch(B: int, T: int, d_h: int = 768) -> dict[str, torch.Tensor]:
    """Synthetic batch simulating cached hidden states."""
    return {
        "input_ids": torch.randint(0, 50304, (B, T)),
        "attention_mask": torch.ones(B, T, dtype=torch.long),
        "hidden_states": torch.randn(B, T, d_h),
    }


def _make_cached_padded_batch(B: int, T: int, d_h: int = 768) -> dict[str, torch.Tensor]:
    """Synthetic cached batch with variable lengths."""
    batch = _make_cached_batch(B, T, d_h)
    for i in range(B // 2, B):
        pad_start = T // 2 + i
        batch["attention_mask"][i, min(pad_start, T):] = 0
        batch["hidden_states"][i, min(pad_start, T):] = 0
    return batch


@pytest.fixture
def cached_config():
    return ExperimentConfig(
        llm=LLMConfig(name="EleutherAI/pythia-160m", layer_index=6, hidden_dim=768),
        data=DataConfig(mode="cached", hf_dataset=None, cached_hidden_states_dir="/tmp/fake_cache"),
        encoder=EncoderConfig(type="mlp", z_dim=32, mlp_hidden_dims=[256, 128]),
        predictor=PredictorConfig(mode="autonomous", hidden_dim=128, num_layers=2),
        training=TrainingConfig(
            batch_size=4, max_steps=10, lr=1e-3, warmup_steps=2,
            eval_every=100, checkpoint_every=100, grad_clip=1.0,
        ),
        logging=LoggingConfig(output_dir="/tmp/ret_test_cached"),
    )


def test_training_step_cached_mode(cached_config):
    """Cached mode: loss decreases without loading LLM."""
    seed_everything(42)
    trainer = JEPATrainer(cached_config, accelerator=Accelerator(split_batches=True))
    assert trainer.extractor is None  # No LLM loaded

    losses = []
    for step in range(10):
        batch = _make_cached_batch(4, 128)
        metrics = trainer.train_step(batch)
        loss = metrics["train/total_loss"]
        assert not torch.isnan(torch.tensor(loss)), f"NaN loss at step {step}"
        losses.append(loss)

    avg_first = sum(losses[:3]) / 3
    avg_last = sum(losses[-3:]) / 3
    assert avg_last < avg_first, f"Loss did not decrease: {avg_first:.4f} -> {avg_last:.4f}"


def test_eval_step_cached_mode(cached_config):
    """Eval step works with cached hidden states."""
    seed_everything(42)
    trainer = JEPATrainer(cached_config, accelerator=Accelerator(split_batches=True))

    batch = _make_cached_batch(4, 128)
    metrics = trainer.eval_step(batch)
    assert "val/pred_loss" in metrics
    assert metrics["val/pred_loss"] > 0


def test_training_step_cached_with_mask(cached_config):
    """Cached mode with variable-length padded sequences."""
    seed_everything(42)
    trainer = JEPATrainer(cached_config, accelerator=Accelerator(split_batches=True))

    batch = _make_cached_padded_batch(4, 128)
    metrics = trainer.train_step(batch)
    assert not torch.isnan(torch.tensor(metrics["train/total_loss"]))
    assert metrics["train/pred_loss"] > 0
