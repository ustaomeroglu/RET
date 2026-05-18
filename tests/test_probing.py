"""Tests for the probing experiment infrastructure."""

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# test_build_prompt
# ---------------------------------------------------------------------------

def test_build_prompt_no_chat_template():
    """Base models (Pythia) have no chat_template — should return plain user text."""
    from transformers import AutoTokenizer
    from evaluation.probing.model_utils import build_prompt

    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    prompt = build_prompt("Is the sky blue?", tok, system_text="You are a helper.")
    # Pythia has no chat_template, so build_prompt should return raw user text
    assert prompt == "Is the sky blue?"


def test_build_prompt_with_chat_template():
    """Tokenizers with a chat_template should use apply_chat_template."""
    from unittest.mock import MagicMock
    from evaluation.probing.model_utils import build_prompt

    tok = MagicMock()
    tok.chat_template = "some_template"
    tok.apply_chat_template.return_value = "<formatted>Is the sky blue?</formatted>"

    result = build_prompt("Is the sky blue?", tok, system_text="System.")
    tok.apply_chat_template.assert_called_once()
    assert result == "<formatted>Is the sky blue?</formatted>"


# ---------------------------------------------------------------------------
# test_probes
# ---------------------------------------------------------------------------

def test_linear_probe_binary():
    """LinearProbe on well-separated synthetic binary data."""
    from evaluation.probing.baselines.linear_probe import LinearProbe

    rng = np.random.default_rng(42)
    N, D = 200, 16
    X = np.vstack([rng.normal(-2, 0.5, (N // 2, D)),
                   rng.normal(2, 0.5, (N // 2, D))]).astype(np.float32)
    y = np.array([0] * (N // 2) + [1] * (N // 2))

    probe = LinearProbe(task_type="binary", lr=1e-2, max_epochs=200, device="cpu")
    probe.fit(X, y)
    preds = probe.predict(X)
    assert set(preds.tolist()).issubset({0, 1})
    acc = (preds == y).mean()
    assert acc > 0.9, f"Expected >90% accuracy on separable data, got {acc:.2%}"


def test_dom_probe_binary():
    """DOMProbe on well-separated synthetic binary data — should be near-perfect."""
    from evaluation.probing.baselines.dom_probe import DOMProbe

    rng = np.random.default_rng(42)
    N, D = 200, 16
    X = np.vstack([rng.normal(-3, 0.3, (N // 2, D)),
                   rng.normal(3, 0.3, (N // 2, D))]).astype(np.float32)
    y = np.array([0] * (N // 2) + [1] * (N // 2))

    probe = DOMProbe(task_type="binary", device="cpu")
    probe.fit(X, y)
    preds = probe.predict(X)
    acc = (preds == y).mean()
    assert acc > 0.95, f"DOM should be near-perfect on separated data, got {acc:.2%}"


def test_dom_probe_multiclass():
    """DOMProbe on 3-class synthetic data with non-degenerate centers."""
    from evaluation.probing.baselines.dom_probe import DOMProbe

    rng = np.random.default_rng(42)
    N_per_class, D = 100, 16
    # Use orthogonal directions so no class centroid sits at the global mean
    c0 = np.zeros(D); c0[0] = 5
    c1 = np.zeros(D); c1[1] = 5
    c2 = np.zeros(D); c2[2] = 5
    X = np.vstack([rng.normal(c0, 0.3, (N_per_class, D)),
                   rng.normal(c1, 0.3, (N_per_class, D)),
                   rng.normal(c2, 0.3, (N_per_class, D))]).astype(np.float32)
    y = np.array([0] * N_per_class + [1] * N_per_class + [2] * N_per_class)

    probe = DOMProbe(task_type="multiclass", device="cpu")
    probe.fit(X, y)
    preds = probe.predict(X)
    acc = (preds == y).mean()
    assert acc > 0.9, f"Expected >90% on separated 3-class data, got {acc:.2%}"


def test_attention_probe_binary():
    """AttentionProbe on synthetic dict-format data."""
    from evaluation.probing.baselines.attention_probe import AttentionProbe

    rng = np.random.default_rng(42)
    N, S, D = 50, 10, 8
    hs = rng.standard_normal((N, S, D)).astype(np.float32)
    # Make class 1 have higher mean hidden states
    hs[:25] -= 2.0
    hs[25:] += 2.0
    mask = np.ones((N, S), dtype=np.float32)
    X = {"hidden_states": hs, "attention_mask": mask}
    y = np.array([0] * 25 + [1] * 25)

    probe = AttentionProbe(task_type="binary", lr=1e-2, max_epochs=100, device="cpu")
    probe.fit(X, y)
    preds = probe.predict(X)
    assert preds.shape == (N,)
    assert set(preds.tolist()).issubset({0, 1})


# ---------------------------------------------------------------------------
# test_evaluation
# ---------------------------------------------------------------------------

def test_compute_metrics():
    """Verify accuracy, F1, and confusion matrix on known data."""
    from evaluation.probing.evaluation import compute_metrics

    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 0])
    m = compute_metrics(y_true, y_pred)
    assert m["accuracy"] == pytest.approx(3 / 5)
    assert m["n_samples"] == 5
    cm = np.array(m["confusion_matrix"])
    assert cm.shape == (2, 2)
    # TN=1, FP=1, FN=1, TP=2
    assert cm[0, 0] == 1  # TN
    assert cm[0, 1] == 1  # FP
    assert cm[1, 0] == 1  # FN
    assert cm[1, 1] == 2  # TP


def test_concat_X_different_seq_lengths():
    """_concat_X should pad dicts with different sequence lengths."""
    from evaluation.probing.evaluation import _concat_X

    D = 4
    x1 = {
        "hidden_states": np.ones((3, 5, D), dtype=np.float32),
        "attention_mask": np.ones((3, 5), dtype=np.float32),
    }
    x2 = {
        "hidden_states": np.ones((2, 10, D), dtype=np.float32) * 2,
        "attention_mask": np.ones((2, 10), dtype=np.float32),
    }
    result = _concat_X([x1, x2])
    assert result["hidden_states"].shape == (5, 10, D)
    assert result["attention_mask"].shape == (5, 10)
    # x1 should be left-padded: first 5 columns zero, last 5 columns = 1
    assert result["hidden_states"][0, 0, 0] == 0.0  # padding
    assert result["hidden_states"][0, 5, 0] == 1.0  # original data
    # x2 should be unchanged
    assert result["hidden_states"][3, 0, 0] == 2.0


# ---------------------------------------------------------------------------
# test_extract_yes_no
# ---------------------------------------------------------------------------

def test_extract_yes_no():
    """Test yes/no extraction edge cases."""
    from evaluation.probing.model_utils import extract_yes_no

    assert extract_yes_no("Yes") == "yes"
    assert extract_yes_no("no") == "no"
    assert extract_yes_no("") == "no"
    assert extract_yes_no("yesterday") == "no"  # should NOT match
    assert extract_yes_no("novel") == "no"       # should NOT match
    assert extract_yes_no("yes no yes") == "yes"  # last match
    assert extract_yes_no("No, not really") == "no"
    assert extract_yes_no("I think yes but also no") == "no"  # last match


# ---------------------------------------------------------------------------
# test_needed_modes
# ---------------------------------------------------------------------------

def test_needed_modes():
    """Verify mode set computation for different probe/hs_type combos."""
    from evaluation.probing.experiment_utils import needed_modes
    from evaluation.probing.config import ModelConfig, ProbeConfig

    class MockCfg:
        def __init__(self, probe_type, hs_type):
            self.probe = ProbeConfig(probe_type=probe_type)
            self.model = ModelConfig(hidden_state_type=hs_type)

    # All probes, all hs modes
    modes = needed_modes(MockCfg("all", "all"))
    assert "average" in modes
    assert "assistant_token" in modes
    assert "all_tokens" in modes

    # Linear only, average only
    modes = needed_modes(MockCfg("linear", "average"))
    assert modes == {"average"}

    # Attention only
    modes = needed_modes(MockCfg("attention", "all"))
    assert modes == {"all_tokens"}

    # DOM, assistant_token
    modes = needed_modes(MockCfg("dom", "assistant_token"))
    assert modes == {"assistant_token"}


# ---------------------------------------------------------------------------
# test_get_layer_from architectures
# ---------------------------------------------------------------------------

def test_get_layer_from_gptneox():
    """Verify _get_layer_from supports GPTNeoX (Pythia) architecture."""
    from evaluation.probing.model_utils import _get_layer_from
    from unittest.mock import MagicMock

    # Mock a GPTNeoX model
    model = MagicMock()
    model.gpt_neox.layers = [MagicMock(name=f"layer_{i}") for i in range(12)]
    # Remove other attrs to ensure gpt_neox branch is taken
    del model.model
    del model.transformer

    layer = _get_layer_from(model, 6)
    assert layer is model.gpt_neox.layers[6]

    # Test layer -1 (last layer)
    layer = _get_layer_from(model, -1)
    assert layer is model.gpt_neox.layers[11]
