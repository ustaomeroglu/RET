from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from ret.config import LLMConfig

# Ensure custom model types (e.g. gpt_rope) are registered with HF's
# auto-class machinery before any AutoModelForCausalLM.from_pretrained call.
try:
    from LocalLLMs import gpt_rope  # noqa: F401
except ImportError:
    pass


class _EarlyExitException(Exception):
    """Raised to short-circuit the LLM forward pass after the target layer."""
    pass


def _find_layers_and_embed(model: nn.Module) -> tuple[nn.ModuleList, nn.Module]:
    """Auto-detect the transformer layer stack and embedding module.

    Supported architectures:
        - GPTNeoX (Pythia): model.gpt_neox.layers / model.gpt_neox.embed_in
        - LLaMA/Mistral/GPT-OSS: model.model.layers / model.model.embed_tokens
        - GPT-2/GPT-J: model.transformer.h / model.transformer.wte
    """
    # GPTNeoX (Pythia family)
    if hasattr(model, "gpt_neox"):
        return model.gpt_neox.layers, model.gpt_neox.embed_in

    # LLaMA, Mistral, GPT-OSS, and similar
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        inner = model.model
        if hasattr(inner, "embed_tokens"):
            return inner.layers, inner.embed_tokens
        if hasattr(inner, "embed_in"):
            return inner.layers, inner.embed_in

    # GPT-2, GPT-J
    if hasattr(model, "transformer"):
        t = model.transformer
        layers = getattr(t, "h", None)
        embed = getattr(t, "wte", None)
        if layers is not None and embed is not None:
            return layers, embed

    raise ValueError(
        f"Cannot auto-detect layers and embeddings for model type "
        f"{type(model).__name__}. Supported: GPTNeoX, LLaMA/Mistral/GPT-OSS, GPT-2/GPT-J."
    )


class HiddenStateExtractor(nn.Module):
    """Extracts hidden states from a frozen LLM via forward hooks.

    Model-agnostic: auto-detects architecture (GPTNeoX, LLaMA, GPT-OSS, GPT-2, etc.).
    Supports single-GPU (.to(device)) and multi-GPU (device_map='auto').

    Supports both single-layer extraction (``layer_index`` is an int) and
    multi-layer extraction (``layer_index`` is a list of ints or ``"all"``).
    In multi-layer mode, hidden states from all listed layers are concatenated
    along the feature dimension.
    """

    def __init__(self, config: LLMConfig, device: str = "cuda"):
        super().__init__()
        self.config = config
        self.device = device

        dtype = getattr(torch, config.dtype)

        load_kwargs: dict = dict(torch_dtype=dtype)
        if config.trust_remote_code:
            load_kwargs["trust_remote_code"] = True

        if config.device_map is not None:
            load_kwargs["device_map"] = config.device_map
            self.model = AutoModelForCausalLM.from_pretrained(
                config.name, **load_kwargs
            )
            self._multi_gpu = True
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                config.name, **load_kwargs
            ).to(device)
            self._multi_gpu = False

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        tokenizer_kwargs: dict = {}
        if config.trust_remote_code:
            tokenizer_kwargs["trust_remote_code"] = True
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.name, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Auto-detect architecture
        self._layers, self._embed = _find_layers_and_embed(self.model)

        # Resolve layer_index: int, list[int], or "all"
        li = config.layer_index
        if li == "all":
            self._layer_indices = list(range(len(self._layers)))
        elif isinstance(li, list):
            self._layer_indices = sorted(li)
        else:
            self._layer_indices = [int(li)]
        self._multi_layer = len(self._layer_indices) > 1

        if self._multi_layer:
            self._hidden_states_dict: dict[int, torch.Tensor] = {}
            self._hook_handles: list = []
            self._register_multi_hooks()
        else:
            self._hidden_states: torch.Tensor | None = None
            self._register_hook(self._layer_indices[0])

    @property
    def num_extract_layers(self) -> int:
        """Number of layers being extracted."""
        return len(self._layer_indices)

    @property
    def input_device(self) -> torch.device:
        """Device where input_ids should be placed."""
        if self._multi_gpu:
            try:
                return next(self._embed.parameters()).device
            except StopIteration:
                return torch.device(self.device)
        return torch.device(self.device)

    def _register_hook(self, layer_index: int) -> None:
        """Register a single-layer extraction hook (original behaviour)."""
        layer = self._layers[layer_index]
        total_layers = len(self._layers)
        self._use_early_exit = layer_index < total_layers - 1

        def hook_fn(module: nn.Module, input: tuple, output: tuple | torch.Tensor) -> None:
            if isinstance(output, tuple):
                self._hidden_states = output[0]
            else:
                self._hidden_states = output
            if self._use_early_exit:
                raise _EarlyExitException

        self._hook_handle = layer.register_forward_hook(hook_fn)

    def _register_multi_hooks(self) -> None:
        """Register hooks on multiple layers for concatenated extraction."""
        max_idx = max(self._layer_indices)
        total_layers = len(self._layers)
        self._use_early_exit = max_idx < total_layers - 1

        for idx in self._layer_indices:
            layer = self._layers[idx]

            def make_hook(layer_idx: int):
                def hook_fn(module: nn.Module, input: tuple, output: tuple | torch.Tensor) -> None:
                    h = output[0] if isinstance(output, tuple) else output
                    self._hidden_states_dict[layer_idx] = h
                    if layer_idx == max_idx and self._use_early_exit:
                        raise _EarlyExitException
                return hook_fn

            handle = layer.register_forward_hook(make_hook(idx))
            self._hook_handles.append(handle)

    @torch.no_grad()
    def extract(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Extract hidden states from the target layer(s).

        Args:
            input_ids: (B, T) token IDs
            attention_mask: optional (B, T) mask (1 = valid, 0 = padding).

        Returns:
            Single-layer: (B, T, d_h) in float32.
            Multi-layer:  (B, T, num_layers * d_h) in float32 (concatenated).
        """
        input_ids = input_ids.to(self.input_device)
        kwargs: dict = {"input_ids": input_ids}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask.to(self.input_device)
        try:
            self.model(**kwargs)
        except _EarlyExitException:
            pass

        if self._multi_layer:
            h_list = [
                self._hidden_states_dict[idx].to(self.device).float()
                for idx in self._layer_indices
            ]
            self._hidden_states_dict.clear()
            return torch.cat(h_list, dim=-1)  # (B, T, num_layers * d_h)
        else:
            assert self._hidden_states is not None
            h = self._hidden_states.to(self.device).float()
            self._hidden_states = None
            return h

    @torch.no_grad()
    def get_token_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get raw token embeddings (input to the transformer).

        Args:
            input_ids: (B, T) token IDs

        Returns:
            Token embeddings (B, T, d_h) in float32 on self.device
        """
        input_ids = input_ids.to(self.input_device)
        embeddings = self._embed(input_ids)
        return embeddings.to(self.device).float()

    def cleanup(self) -> None:
        if self._multi_layer:
            for handle in self._hook_handles:
                handle.remove()
        else:
            self._hook_handle.remove()
