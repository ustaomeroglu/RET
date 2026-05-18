"""
Model utilities: loading, prompt formatting, hidden-state extraction, generation.

Design notes
------------
* We use HuggingFace transformers for everything (no vLLM).
* Prompt formatting is model-agnostic: uses tokenizer.apply_chat_template() when
  available (chat models like gpt-oss, LLaMA-chat), falls back to plain text for
  base models (Pythia, GPT-2).
* Hidden-state extraction registers a single forward hook on the target layer and
  collects the residual-stream tensor.  Three pooling modes are supported:
    - "average"        : mean over all input-token positions
    - "assistant_token": the hidden state at the very last input token
    - "all_tokens"     : full sequence (for AttentionProbe)
* Architecture-agnostic layer access supports GPTNeoX (Pythia), LLaMA/Mistral/GPT-OSS,
  and GPT-2/GPT-J.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Multi-GPU runner
# ---------------------------------------------------------------------------

class MultiGPURunner:
    """Holds N independent model copies, one per GPU.

    Prompts are split evenly across copies and processed in parallel via
    threads (CUDA releases the GIL, so true concurrency is possible).
    Hooks register on each copy's real layer so every copy's output is
    captured correctly — no DataParallel batch-split/gather confusion.
    """

    def __init__(self, copies: list):
        self.copies   = copies          # [model_on_cuda_0, model_on_cuda_1, ...]
        self.n_gpus   = len(copies)

    def parameters(self):
        """Expose first copy's parameters (e.g. for device resolution)."""
        return self.copies[0].parameters()

    @property
    def module(self):
        """Compatibility shim: callers that do model.module.generate()."""
        return self.copies[0]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    model_id: str,
    device: str = "cuda",
    n_gpus: int = 0,
) -> Tuple[AutoTokenizer, "AutoModelForCausalLM | MultiGPURunner"]:
    """Load the model for inference.

    Parameters
    ----------
    n_gpus : number of GPUs to use.
             0 = use all available GPUs.
             1 = single-GPU on cuda:0.
             >1 = load a FULL model copy on each of the first n_gpus GPUs
                  (true data-parallel: one full copy per GPU, batches split
                  across copies via threads — no layer-splitting).
    """
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    logger.info("Loading tokenizer from %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    n_available = torch.cuda.device_count()
    if n_gpus <= 0 or n_gpus > n_available:
        n_gpus = n_available
    logger.info("Loading model from %s (dtype=%s, n_gpus=%d)",
                model_id, dtype, n_gpus)

    if n_gpus == 0:
        # CPU-only mode
        logger.info("  No GPUs available; loading on CPU (dtype=float32)")
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
        model.eval()
    elif n_gpus == 1:
        logger.info("  Loading copy on cuda:0 ...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map={"": 0})
        model.eval()
        logger.info("Single-GPU mode (cuda:0)")
    else:
        copies = []
        for g in range(n_gpus):
            logger.info("  Loading copy on cuda:%d ...", g)
            m = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=dtype, device_map={"": g})
            m.eval()
            copies.append(m)
        model = MultiGPURunner(copies)
        logger.info("MultiGPURunner ready: %d GPUs", n_gpus)

    return tokenizer, model


def free_model(model) -> None:
    """Delete model weights from GPU and empty the CUDA cache.

    Call this after all extraction / generation is complete so that probe
    training has access to the full GPU memory.  Hidden states should already
    be cached on CPU before calling this.

    Note: callers should also set their own reference to None after calling
    this function to allow garbage collection.
    """
    import gc

    if isinstance(model, MultiGPURunner):
        for g, copy in enumerate(model.copies):
            del copy
            logger.info("  Freed model copy on cuda:%d", g)
        model.copies.clear()
    else:
        del model

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    logger.info("Model freed from GPU; CUDA cache emptied.")


# ---------------------------------------------------------------------------
# Model-agnostic prompt formatting
# ---------------------------------------------------------------------------

import re as _re

_HARMONY_SYSTEM_BLOCK_RE = _re.compile(
    r'<\|start\|>system<\|message\|>.*?<\|end\|>',
    _re.DOTALL,
)


def strip_harmony_system_block(text: str) -> str:
    """Remove the ``<|start|>system<|message|>…<|end|>`` block (GPT identity,
    knowledge cutoff, reasoning level, valid channels) that the GPT-OSS
    tokenizer's chat template injects automatically.

    The developer and user blocks are preserved unchanged.
    """
    return _HARMONY_SYSTEM_BLOCK_RE.sub('', text, count=1)


def build_prompt(
    user_text: str,
    tokenizer: AutoTokenizer,
    system_text: str = (
        "You are a helpful assistant. "
        "Do not think step by step or add any reasoning. "
        "Give only the direct answer, nothing else."
    ),
    add_generation_prompt: bool = True,
    include_gpt_info: bool = True,
    **template_kwargs,
) -> str:
    """Return a fully-formatted prompt string, auto-detecting chat template.

    Chat models (gpt-oss, LLaMA-chat, etc.): uses tokenizer.apply_chat_template()
    Base models (Pythia, GPT-2, etc.): returns plain user text (base models do
    not understand role structure, so we skip the system message wrapper).

    Parameters
    ----------
    include_gpt_info : bool
        If *False* (default) the GPT-OSS ``<|start|>system…<|end|>`` block
        (ChatGPT identity, knowledge cutoff, reasoning effort, valid channels)
        is stripped, keeping only the developer-instructions and user blocks.
        Set to *True* to preserve the full harmony prompt including the
        system preamble.

    Raises
    ------
    RuntimeError
        If the tokenizer has a ``chat_template`` but ``apply_chat_template``
        fails (e.g. template does not support the 'system' role).
    """
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]

    # Try the tokenizer's chat template first
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                **template_kwargs,
            )
        except Exception as e:
            raise RuntimeError(
                f"apply_chat_template failed: {e}. Check that the tokenizer's "
                f"chat_template supports the 'system' role."
            ) from e
        if not include_gpt_info:
            prompt = strip_harmony_system_block(prompt)
        return prompt

    # Fallback for base models: system_text cannot be used
    logger.warning(
        "Tokenizer has no chat_template — system_text will be ignored. "
        "Only the raw user_text is returned. Model: %s",
        type(tokenizer).__name__,
    )
    return user_text


# ---------------------------------------------------------------------------
# Token boundary detection
# ---------------------------------------------------------------------------

def compute_system_prefix_len(
    tokenizer: AutoTokenizer,
    system_text: str = (
        "You are a helpful assistant. "
        "Do not think step by step or add any reasoning. "
        "Give only the direct answer, nothing else."
    ),
    include_gpt_info: bool = True,
) -> int:
    """Return the number of tokens in the system-prompt prefix.

    This count includes all role-formatting tokens up to and including the
    last formatting token *before* the first user-content token.  It is the
    same for every sample that shares the same system prompt.

    When ``include_gpt_info=False`` the GPT-OSS system block is stripped
    before counting, so only the developer-instructions prefix is measured.

    For base models (no chat template) returns 0.
    """
    if not (hasattr(tokenizer, 'chat_template') and tokenizer.chat_template):
        return 0

    sys_only = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_text}],
        tokenize=False,
        add_generation_prompt=False,
    )
    user_stub = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_text},
         {"role": "user",   "content": ""}],
        tokenize=False,
        add_generation_prompt=False,
    )

    if not include_gpt_info:
        sys_only = strip_harmony_system_block(sys_only)
        user_stub = strip_harmony_system_block(user_stub)

    prefix_ids = tokenizer.encode(user_stub, add_special_tokens=False)
    user_ids = tokenizer.encode("", add_special_tokens=False)
    prefix_len = len(prefix_ids) - len(user_ids)

    sys_ids = tokenizer.encode(sys_only, add_special_tokens=False)
    prefix_len = max(prefix_len, len(sys_ids))

    logger.info("System prefix length: %d tokens (include_gpt_info=%s, system_text=%r...)",
                prefix_len, include_gpt_info, system_text[:50])
    return prefix_len


def make_user_token_mask(
    attention_mask: "np.ndarray",
    system_prefix_len: int,
) -> "np.ndarray":
    """Build a mask selecting only user-input tokens (excluding system prefix).

    Works correctly with left-padded sequences: the system prefix is the
    first ``system_prefix_len`` real (non-padding) tokens.

    Parameters
    ----------
    attention_mask : [N, S]  float/int array (1 = real token, 0 = padding)
    system_prefix_len : number of system-prefix tokens to exclude

    Returns
    -------
    user_mask : [N, S]  same shape, 1 only for user-input tokens
    """
    assert attention_mask.ndim == 2, \
        f"attention_mask must be 2D [N,S], got {attention_mask.shape}"
    assert isinstance(system_prefix_len, int), \
        f"system_prefix_len must be int, got {type(system_prefix_len)}"

    if system_prefix_len <= 0:
        return attention_mask.copy()

    cumsum = np.cumsum(attention_mask.astype(np.float32), axis=1)
    user_mask = attention_mask.copy().astype(np.float32)
    user_mask[cumsum <= system_prefix_len] = 0.0

    # Output must never have MORE real tokens than input
    assert np.all(user_mask.sum(axis=1) <= attention_mask.sum(axis=1) + 1e-6), \
        "user_mask has more real tokens than original mask"

    return user_mask


# ---------------------------------------------------------------------------
# Hidden-state extraction
# ---------------------------------------------------------------------------

class _HiddenStateHook:
    """Forward hook that captures the output of a single transformer layer."""
    def __init__(self):
        self.output: Optional[torch.Tensor] = None

    def __call__(self, module, module_input, module_output):
        # Layer output is (hidden_states, ...) for most architectures
        if isinstance(module_output, tuple):
            self.output = module_output[0]
        else:
            self.output = module_output


def _unwrap(model) -> AutoModelForCausalLM:
    """Return the underlying HF model (single copy) regardless of wrapper."""
    if isinstance(model, MultiGPURunner):
        return model.copies[0]
    if isinstance(model, torch.nn.DataParallel):
        return model.module
    return model


def _get_layer_from(m: AutoModelForCausalLM, layer_idx: int) -> torch.nn.Module:
    """Retrieve transformer layer from a *single* HF model instance.

    Supports:
        - GPTNeoX (Pythia): model.gpt_neox.layers
        - LLaMA/Mistral/GPT-OSS: model.model.layers
        - GPT-2/GPT-J: model.transformer.h
    """
    if hasattr(m, "gpt_neox") and hasattr(m.gpt_neox, "layers"):
        layers = m.gpt_neox.layers
    elif hasattr(m, "model") and hasattr(m.model, "layers"):
        layers = m.model.layers
    elif hasattr(m, "transformer") and hasattr(m.transformer, "h"):
        layers = m.transformer.h
    else:
        raise ValueError(
            "Cannot locate transformer layers. "
            "Expected model.gpt_neox.layers, model.model.layers, or model.transformer.h."
        )
    if layer_idx == -1:
        layer_idx = len(layers) - 1
    return layers[layer_idx]


def _get_layer(model, layer_idx: int) -> torch.nn.Module:
    """Retrieve transformer layer; unwraps MultiGPURunner/DataParallel."""
    return _get_layer_from(_unwrap(model), layer_idx)


# ---------------------------------------------------------------------------
# Core single-GPU batch extraction helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def _extract_batch_on_model(
    prompts: List[str],
    m: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    layer_idx: int,
    modes: List[str],
    batch_size: int,
    sample_accumulator=None,
) -> dict:
    """Run full extraction on a single model instance (single GPU).

    Returns a partial result dict (same structure as extract_hidden_states_multi_mode)
    but only for the given subset of prompts.

    Note: tokenizer.padding_side must be set to "left" BEFORE calling this
    function (done in extract_hidden_states_multi_mode for thread safety).
    """
    target_layer = _get_layer_from(m, layer_idx)
    device       = next(m.parameters()).device
    all_tokens_needed = "all_tokens" in modes

    acc: dict = {mode: [] for mode in modes}
    if all_tokens_needed:
        acc["all_tokens_mask"] = []

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        enc = tokenizer(batch_prompts, return_tensors="pt",
                        padding=True, truncation=True)
        input_ids      = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        hook   = _HiddenStateHook()
        handle = target_layer.register_forward_hook(hook)
        try:
            m(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            handle.remove()

        # Log samples AFTER the forward pass so the tokenizer is no longer
        # borrowed by the batch-encode call above (fast tokenizer re-entrancy).
        if sample_accumulator is not None and not sample_accumulator.is_full:
            sample_accumulator.add_batch(input_ids, attention_mask)

        hidden = hook.output.float()   # [B, S, D]

        if "average" in modes:
            mask_f = attention_mask.unsqueeze(-1).float()
            acc["average"].append(
                ((hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)).cpu().numpy()
            )

        if "assistant_token" in modes:
            seq_len = (attention_mask.sum(dim=1) - 1).clamp(min=0)
            bidx    = torch.arange(hidden.size(0), device=device)
            acc["assistant_token"].append(hidden[bidx, seq_len].cpu().numpy())

        if all_tokens_needed:
            acc["all_tokens"].append(hidden.cpu().numpy())
            acc["all_tokens_mask"].append(attention_mask.cpu().numpy())

    # Build per-mode outputs
    partial: dict = {}
    for mode in ["average", "assistant_token"]:
        if mode in modes:
            partial[mode] = np.concatenate(acc[mode], axis=0)

    if all_tokens_needed:
        all_h = acc["all_tokens"]
        all_m = acc["all_tokens_mask"]
        max_S = max(h.shape[1] for h in all_h)
        D, N  = all_h[0].shape[2], len(prompts)
        ph = np.zeros((N, max_S, D), dtype=np.float32)
        pm = np.zeros((N, max_S),    dtype=np.float32)
        row = 0
        for h, mv in zip(all_h, all_m):
            B, S, _ = h.shape
            ph[row:row+B, max_S-S:, :] = h
            pm[row:row+B, max_S-S:]    = mv
            row += B
        partial["all_tokens"] = {"hidden_states": ph, "attention_mask": pm}

    return partial


def _merge_partial_results(partials: List[dict], global_indices: List[List[int]],
                           n_total: int) -> dict:
    """Merge per-GPU partial results back into global arrays in original order.

    Parameters
    ----------
    partials       : list of partial dicts, one per GPU
    global_indices : global_indices[g] = list of original prompt indices GPU g handled
    n_total        : total number of prompts
    """
    modes = list(partials[0].keys())
    result: dict = {}

    for mode in modes:
        if mode == "all_tokens":
            # dict with hidden_states [N_g, S, D] and attention_mask [N_g, S]
            # Find global max_S
            all_S = [p["all_tokens"]["hidden_states"].shape[1] for p in partials]
            max_S = max(all_S)
            D     = partials[0]["all_tokens"]["hidden_states"].shape[2]
            gh    = np.zeros((n_total, max_S, D), dtype=np.float32)
            gm    = np.zeros((n_total, max_S),    dtype=np.float32)
            for partial, idxs in zip(partials, global_indices):
                h = partial["all_tokens"]["hidden_states"]  # [N_g, S_g, D]
                m = partial["all_tokens"]["attention_mask"] # [N_g, S_g]
                S_g = h.shape[1]
                for local_i, global_i in enumerate(idxs):
                    # right-align (left-padded) into global buffer
                    gh[global_i, max_S-S_g:, :] = h[local_i]
                    gm[global_i, max_S-S_g:]    = m[local_i]
            result["all_tokens"] = {"hidden_states": gh, "attention_mask": gm}
        else:
            # ndarray [N_g, D]
            D    = partials[0][mode].shape[1]
            garr = np.zeros((n_total, D), dtype=np.float32)
            for partial, idxs in zip(partials, global_indices):
                for local_i, global_i in enumerate(idxs):
                    garr[global_i] = partial[mode][local_i]
            result[mode] = garr

    return result


@torch.no_grad()
def extract_hidden_states(
    prompts: List[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    layer: int,
    mode: Literal["average", "assistant_token"],
    batch_size: int = 8,
) -> np.ndarray:
    """Extract per-sample hidden states from the specified layer.

    Parameters
    ----------
    prompts : list[str]
        Already-formatted prompt strings (output of ``build_prompt``).
    model : AutoModelForCausalLM
        Must be a single model, NOT a MultiGPURunner.
    tokenizer : AutoTokenizer
    layer : int
        0-indexed transformer layer index. -1 = last layer.
    mode : "average" | "assistant_token"
        Pooling strategy.
    batch_size : int

    Returns
    -------
    np.ndarray of shape [N, hidden_dim]
    """
    if isinstance(model, MultiGPURunner):
        raise ValueError(
            "extract_hidden_states does not support MultiGPURunner. "
            "Use extract_hidden_states_multi_mode instead."
        )

    target_layer = _get_layer(model, layer)
    device = next(model.parameters()).device

    all_hidden: List[np.ndarray] = []

    tokenizer.padding_side = "left"
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]

        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        hook = _HiddenStateHook()
        handle = target_layer.register_forward_hook(hook)
        try:
            model(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            handle.remove()

        # hook.output : [batch, seq_len, hidden_dim]
        hidden = hook.output.float()  # cast to fp32 for probe training stability

        if mode == "average":
            mask = attention_mask.unsqueeze(-1).float()  # [B, T, 1]
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            pooled = summed / counts  # [B, hidden_dim]

        elif mode == "assistant_token":
            seq_lengths = (attention_mask.sum(dim=1) - 1).clamp(min=0)  # [B]
            batch_idx = torch.arange(hidden.size(0), device=device)
            pooled = hidden[batch_idx, seq_lengths]  # [B, hidden_dim]

        else:
            raise ValueError(f"Unknown hidden_state mode: {mode!r}")

        all_hidden.append(pooled.cpu().numpy())

    return np.concatenate(all_hidden, axis=0)  # [N, hidden_dim]


@torch.no_grad()
def extract_all_token_hidden_states(
    prompts: List[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    layer: int,
    batch_size: int = 8,
) -> dict:
    """Extract all token hidden states from user-start to assistant token.

    Parameters
    ----------
    prompts : fully-formatted prompt strings (output of build_prompt).
    model : Must be a single model, NOT a MultiGPURunner.
    layer : transformer layer index.
    batch_size : forward-pass batch size.

    Returns
    -------
    dict with:
        "hidden_states" : np.ndarray [N, S_max, D]  (left-padded)
        "attention_mask": np.ndarray [N, S_max]      (1 = valid token)
    """
    if isinstance(model, MultiGPURunner):
        raise ValueError(
            "extract_all_token_hidden_states does not support MultiGPURunner. "
            "Use extract_hidden_states_multi_mode instead."
        )

    target_layer = _get_layer(model, layer)
    device = next(model.parameters()).device

    all_hidden: List[np.ndarray] = []
    all_masks:  List[np.ndarray] = []

    tokenizer.padding_side = "left"
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]

        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids    = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        hook = _HiddenStateHook()
        handle = target_layer.register_forward_hook(hook)
        try:
            model(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            handle.remove()

        hidden = hook.output.float()
        all_hidden.append(hidden.cpu().numpy())
        all_masks.append(attention_mask.cpu().numpy())

    max_S = max(h.shape[1] for h in all_hidden)
    D     = all_hidden[0].shape[2]

    padded_h = np.zeros((len(prompts), max_S, D), dtype=np.float32)
    padded_m = np.zeros((len(prompts), max_S),    dtype=np.float32)

    row = 0
    for h, m in zip(all_hidden, all_masks):
        B, S, _ = h.shape
        padded_h[row:row+B, max_S-S:, :] = h
        padded_m[row:row+B, max_S-S:]    = m
        row += B

    return {"hidden_states": padded_h, "attention_mask": padded_m}

@torch.no_grad()
def extract_hidden_states_multi_mode(
    prompts: List[str],
    model,
    tokenizer: AutoTokenizer,
    layer: int,
    modes: List[str],
    batch_size: int = 8,
    sample_accumulator=None,
) -> dict:
    """Extract hidden states for ALL requested modes in a SINGLE forward pass.

    Dispatches to MultiGPURunner (one full model copy per GPU, prompts split
    across copies in parallel threads) or falls back to single-GPU extraction.

    Parameters
    ----------
    sample_accumulator : optional SampleAccumulator
        If provided, collects batched input_ids/attention_mask for detailed
        prompt logging.

    Returns
    -------
    dict:
        "average"        -> np.ndarray [N, D]
        "assistant_token"-> np.ndarray [N, D]
        "all_tokens"     -> dict{"hidden_states": [N,S,D], "attention_mask": [N,S]}
    """
    # Set padding_side ONCE before any threads are launched (thread safety)
    tokenizer.padding_side = "left"

    if isinstance(model, MultiGPURunner):
        n        = model.n_gpus
        N        = len(prompts)
        gpu_idxs = [list(range(g, N, n)) for g in range(n)]
        gpu_chunks = [[prompts[i] for i in idxs] for idxs in gpu_idxs]

        import copy
        tok_copies = [copy.deepcopy(tokenizer) for _ in range(n)]
        for tc in tok_copies:
            tc.padding_side = "left"

        def _run_gpu(g):
            if not gpu_chunks[g]:
                return {}
            acc = sample_accumulator if g == 0 else None
            return _extract_batch_on_model(
                gpu_chunks[g], model.copies[g], tok_copies[g], layer, modes,
                batch_size, sample_accumulator=acc,
            )

        with ThreadPoolExecutor(max_workers=n) as ex:
            partials = list(ex.map(_run_gpu, range(n)))

        non_empty = [(p, idxs) for p, idxs in zip(partials, gpu_idxs) if idxs]
        return _merge_partial_results(
            [p for p, _ in non_empty],
            [idxs for _, idxs in non_empty],
            N,
        )
    else:
        m = _unwrap(model)
        return _extract_batch_on_model(
            prompts, m, tokenizer, layer, modes, batch_size,
            sample_accumulator=sample_accumulator,
        )


def generate_answers(
    prompts: List[str],
    model,
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 64,
    stop_strings: Optional[List[str]] = None,
    batch_size: int = 8,
) -> List[str]:
    """Generate continuations for a list of already-formatted prompts.

    With MultiGPURunner, prompts are split across GPU copies and generated
    in parallel threads; outputs are merged back in original order.
    """
    # Set padding_side ONCE before any threads (thread safety)
    tokenizer.padding_side = "left"

    def _gen_on(m, chunk):
        """Generate on a single HF model instance."""
        dev = next(m.parameters()).device
        outputs = []
        for s in range(0, len(chunk), batch_size):
            bp = chunk[s : s + batch_size]
            enc = tokenizer(bp, return_tensors="pt", padding=True, truncation=True)
            ids  = enc["input_ids"].to(dev)
            mask = enc["attention_mask"].to(dev)
            ilen = ids.shape[1]
            out = m.generate(ids, attention_mask=mask,
                             max_new_tokens=max_new_tokens,
                             pad_token_id=tokenizer.pad_token_id,
                             do_sample=False)
            for seq in out:
                text = tokenizer.decode(seq[ilen:], skip_special_tokens=True).strip()
                if stop_strings:
                    for ss in stop_strings:
                        idx = text.find(ss)
                        if idx != -1:
                            text = text[:idx].strip()
                outputs.append(text)
        return outputs

    if isinstance(model, MultiGPURunner):
        n          = model.n_gpus
        N          = len(prompts)
        gpu_idxs   = [list(range(g, N, n)) for g in range(n)]
        gpu_chunks  = [[prompts[i] for i in idxs] for idxs in gpu_idxs]

        with ThreadPoolExecutor(max_workers=n) as ex:
            futures = {ex.submit(_gen_on, model.copies[g], gpu_chunks[g]): g
                       for g in range(n) if gpu_chunks[g]}
            per_gpu: dict = {}
            for future in as_completed(futures):
                per_gpu[futures[future]] = future.result()

        # Reassemble in original order
        all_outputs = [""] * N
        for g, idxs in enumerate(gpu_idxs):
            for local_i, global_i in enumerate(idxs):
                all_outputs[global_i] = per_gpu[g][local_i]
        return all_outputs
    else:
        return _gen_on(_unwrap(model), prompts)



# ---------------------------------------------------------------------------
# Yes / No extraction
# ---------------------------------------------------------------------------

def extract_yes_no(text: str) -> str:
    """Return 'yes' if the last whole-word match is 'yes'; otherwise return 'no'.

    Case-insensitive. Matches 'yes' and 'no' as whole words using regex word
    boundaries so substrings like 'not', 'novel', 'yesterday' do NOT match.
    If neither token is found, returns 'no'.
    """
    import re

    matches = list(re.finditer(r"\b(?:yes|no)\b", text, flags=re.IGNORECASE))
    if not matches:
        return "no"

    last = matches[-1].group(0).lower()
    return "yes" if last == "yes" else "no"
