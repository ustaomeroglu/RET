"""Tests for hidden state extraction."""

import pytest
import torch
import torch.nn as nn

from ret.config import LLMConfig, llm_short_name
from ret.extraction.hooks import HiddenStateExtractor, _find_layers_and_embed


@pytest.fixture(scope="module")
def extractor():
    config = LLMConfig(name="EleutherAI/pythia-160m", layer_index=6, hidden_dim=768)
    ext = HiddenStateExtractor(config, device="cuda")
    yield ext
    ext.cleanup()


def test_extract_shape(extractor):
    """Verify hook captures correct shape (B, T, 768)."""
    B, T = 2, 64
    input_ids = torch.randint(0, 50304, (B, T), device="cuda")
    h = extractor.extract(input_ids)
    assert h.shape == (B, T, 768)
    assert h.dtype == torch.float32


def test_extract_no_gradients(extractor):
    """Verify no gradients flow to frozen LLM."""
    input_ids = torch.randint(0, 50304, (2, 64), device="cuda")
    h = extractor.extract(input_ids)
    assert not h.requires_grad
    for p in extractor.model.parameters():
        assert not p.requires_grad
        assert p.grad is None


def test_token_embeddings_shape(extractor):
    """Verify token embeddings shape."""
    B, T = 2, 64
    input_ids = torch.randint(0, 50304, (B, T), device="cuda")
    emb = extractor.get_token_embeddings(input_ids)
    assert emb.shape == (B, T, 768)
    assert emb.dtype == torch.float32


def test_extract_deterministic(extractor):
    """Verify extraction is deterministic."""
    input_ids = torch.randint(0, 50304, (2, 32), device="cuda")
    h1 = extractor.extract(input_ids)
    h2 = extractor.extract(input_ids)
    assert torch.allclose(h1, h2)


# ---------------------------------------------------------------------------
# Architecture detection tests (no GPU required)
# ---------------------------------------------------------------------------

class _MockGPTNeoXModel:
    """Minimal mock for GPTNeoX architecture detection."""
    def __init__(self):
        self.gpt_neox = type("GptNeox", (), {
            "layers": nn.ModuleList([nn.Linear(10, 10)]),
            "embed_in": nn.Embedding(100, 10),
        })()


class _MockLlamaModel:
    """Minimal mock for LLaMA/GPT-OSS architecture detection."""
    def __init__(self):
        self.model = type("LlamaModel", (), {
            "layers": nn.ModuleList([nn.Linear(10, 10)]),
            "embed_tokens": nn.Embedding(100, 10),
        })()


class _MockGPT2Model:
    """Minimal mock for GPT-2 architecture detection."""
    def __init__(self):
        self.transformer = type("Transformer", (), {
            "h": nn.ModuleList([nn.Linear(10, 10)]),
            "wte": nn.Embedding(100, 10),
        })()


def test_find_layers_gptneox():
    mock = _MockGPTNeoXModel()
    layers, embed = _find_layers_and_embed(mock)
    assert isinstance(layers, nn.ModuleList)
    assert isinstance(embed, nn.Embedding)


def test_find_layers_llama():
    mock = _MockLlamaModel()
    layers, embed = _find_layers_and_embed(mock)
    assert isinstance(layers, nn.ModuleList)
    assert isinstance(embed, nn.Embedding)


def test_find_layers_gpt2():
    mock = _MockGPT2Model()
    layers, embed = _find_layers_and_embed(mock)
    assert isinstance(layers, nn.ModuleList)
    assert isinstance(embed, nn.Embedding)


def test_find_layers_unknown_raises():
    """Unknown architecture should raise ValueError."""
    mock = type("Unknown", (), {})()
    with pytest.raises(ValueError, match="Cannot auto-detect"):
        _find_layers_and_embed(mock)


# ---------------------------------------------------------------------------
# llm_short_name tests
# ---------------------------------------------------------------------------

def test_llm_short_name_pythia():
    cfg = LLMConfig(name="EleutherAI/pythia-160m")
    assert llm_short_name(cfg) == "Pythia160M"


def test_llm_short_name_gpt_oss():
    cfg = LLMConfig(name="openai/gpt-oss-20b")
    assert llm_short_name(cfg) == "GptOss20B"


def test_llm_short_name_no_slash():
    cfg = LLMConfig(name="my-model-7b")
    assert llm_short_name(cfg) == "MyModel7B"


def test_llm_short_name_decimal():
    cfg = LLMConfig(name="EleutherAI/pythia-1.4b")
    assert llm_short_name(cfg) == "Pythia1.4B"


def test_find_layers_model_model_embed_in():
    """Test the fallback embed_in path under model.model."""
    class _MockModelWithEmbedIn:
        def __init__(self):
            self.model = type("Inner", (), {
                "layers": nn.ModuleList([nn.Linear(10, 10)]),
                "embed_in": nn.Embedding(100, 10),
            })()
    mock = _MockModelWithEmbedIn()
    layers, embed = _find_layers_and_embed(mock)
    assert isinstance(layers, nn.ModuleList)
    assert isinstance(embed, nn.Embedding)
