"""
Multi-turn conversation generation with hidden-state extraction.

Two-pass approach per turn:
  1. model.generate() produces the response text (batched across conversations)
  2. Forward pass on prompt+response extracts hidden states at target layers

Batching strategy: conversations are batched by turn number. All conversations
in a batch generate turn 1 together, then turn 2 together, etc. This maximizes
GPU utilization since turns within a single conversation are sequential.

Reuses _HiddenStateHook and _get_layer_from from evaluation/probing/model_utils.py.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation.datasets.sycophancy.data import ConversationSeed
from evaluation.datasets.sycophancy.prompts import build_turn_messages, format_messages
from evaluation.probing.model_utils import (
    _HiddenStateHook,
    _get_layer_from,
    _unwrap,
    free_model,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TurnResult:
    turn: int
    user_message: str
    assistant_response: str
    hidden_states: Dict[int, np.ndarray] = field(default_factory=dict)  # layer -> [S, D]
    boundaries: Dict[str, np.ndarray] = field(default_factory=dict)  # token_ids, role_ids, turn_ids
    prompt_token_count: int = 0
    response_token_count: int = 0


@dataclass
class ConversationResult:
    seed: ConversationSeed
    turns: List[TurnResult] = field(default_factory=list)
    model_id: str = ""


# ---------------------------------------------------------------------------
# Model loading (sharded for large models)
# ---------------------------------------------------------------------------

def load_model_sharded(
    model_id: str,
    dtype: torch.dtype = None,
) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Load model with device_map='auto' for cross-GPU sharding.

    Unlike MultiGPURunner (which makes full copies), this splits layers
    across GPUs — necessary for 72B models that don't fit on one GPU.
    """
    if dtype is None:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    logger.info("Loading tokenizer from %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Two different paths inside this module tokenize the *same* text:
    #   _extract_hidden_states -> tokenizer(prompt_text)          # adds special tokens
    #   compute_token_boundaries -> tokenizer.encode(..., add_special_tokens=False)
    # If they disagree on token count (e.g. tokenizer silently prepends a BOS
    # via add_special_tokens=True), MemmapWriter writes mismatched-length
    # hidden_states.bin vs token/role/turn arrays and MemmapReader refuses
    # to open the cache afterwards. Reject such tokenizers up front instead
    # of producing a corrupted cache. The paper's sycophancy experiments use
    # Qwen2.5/Qwen3.5 only (no BOS), so this check is satisfied for the
    # supported model set.
    _probe = "Hello world."
    _n_call = len(tokenizer(_probe)["input_ids"])
    _n_encode = len(tokenizer.encode(_probe, add_special_tokens=False))
    if _n_call != _n_encode:
        raise NotImplementedError(
            f"Tokenizer for {model_id!r} silently adds "
            f"{_n_call - _n_encode} special token(s) when called as "
            f"tokenizer(text). The sycophancy generation pipeline assumes "
            f"that tokenizer(text) and tokenizer.encode(text, "
            f"add_special_tokens=False) agree (true for Qwen2.5 / Qwen3.5 "
            f"because bos_token_id is None). Models with a BOS token "
            f"(Llama family, Gemma, GPT-2 derivatives, ...) would silently "
            f"corrupt the hidden-state cache with off-by-one token boundaries. "
            f"Either use a supported model or fix the two tokenize call sites "
            f"in evaluation/datasets/sycophancy/generation.py to agree."
        )

    logger.info("Loading model from %s (dtype=%s, device_map=auto)", model_id, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map="auto",
    )
    model.eval()
    logger.info("Model loaded across devices: %s", getattr(model, "hf_device_map", "single"))
    return tokenizer, model


# ---------------------------------------------------------------------------
# Hidden-state extraction (single forward pass with hooks)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _extract_hidden_states(
    prompt_text: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    layers: List[int],
) -> Tuple[Dict[int, np.ndarray], int]:
    """Run a forward pass and extract hidden states at specified layers.

    Returns
    -------
    hidden_states : dict[layer_idx -> ndarray [S, D]]
    seq_len : int (number of tokens in the input)
    """
    raw_model = _unwrap(model)

    enc = tokenizer(prompt_text, return_tensors="pt", truncation=True)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    first_device = next(raw_model.parameters()).device
    input_ids = input_ids.to(first_device)
    attention_mask = attention_mask.to(first_device)

    hooks: List[_HiddenStateHook] = []
    handles = []
    for layer_idx in layers:
        # _get_layer_from handles negative indices natively (Python list).
        layer_module = _get_layer_from(raw_model, layer_idx)
        hook = _HiddenStateHook()
        handle = layer_module.register_forward_hook(hook)
        hooks.append(hook)
        handles.append(handle)

    try:
        raw_model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        for handle in handles:
            handle.remove()

    # Key result dict by the original layer arg so callers (e.g. the
    # MemmapWriter registry in run_benchmark.py) can match on the same key
    # they passed in. Don't translate -1 here — that previously caused a
    # silent key mismatch where layer_-1_activations_cache/ ended up empty.
    result = {}
    for hook, layer_idx in zip(hooks, layers):
        hs = hook.output.float().squeeze(0).cpu().numpy()  # [S, D]
        result[layer_idx] = hs

    return result, input_ids.shape[1]


# ---------------------------------------------------------------------------
# Token boundary computation
# ---------------------------------------------------------------------------

# Role labels for per-token annotation
ROLE_SYSTEM = 0
ROLE_USER = 1
ROLE_ASSISTANT = 2

def compute_token_boundaries(
    seed: ConversationSeed,
    prior_responses: List[str],
    num_turns: int,
    model_id: str,
    tokenizer: AutoTokenizer,
    final_response: str,
    **chat_template_kwargs,
) -> Dict[str, np.ndarray]:
    """Compute per-token role and turn annotations for the full conversation.

    Builds the message list incrementally, tokenizing at each step to find
    where each role segment starts/ends.

    Returns
    -------
    dict with:
        "token_ids"  : int32 [S]  — token IDs of the full sequence
        "role_ids"   : int8  [S]  — 0=system, 1=user, 2=assistant per token
        "turn_ids"   : int8  [S]  — 0=system, 1..num_turns for conversation turns
    """
    from evaluation.datasets.sycophancy.prompts import (
        get_system_prompt, _format_user_message,
    )

    def _tok_len(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    # Build the full text (same as extraction pass). MUST forward
    # chat_template_kwargs so the boundary computation uses the exact same
    # chat template variant (e.g. enable_thinking=False) as the generation
    # pipeline — otherwise Qwen3.5 no_thinking produces a 2-token discrepancy
    # per conversation because the default template injects
    # `<think>\n\n</think>\n\n` vs the no_thinking variant.
    messages = build_turn_messages(seed, prior_responses, num_turns, model_id=model_id)
    full_prompt = format_messages(messages, tokenizer, **chat_template_kwargs)
    full_text = full_prompt + final_response
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    S = len(full_ids)

    role_ids = np.zeros(S, dtype=np.int8)
    turn_ids = np.zeros(S, dtype=np.int8)

    # Build incrementally to find boundaries
    # Some chat templates (e.g. Qwen3.5) require at least one user message,
    # so we combine the system prompt with the first user message in step 1.
    sys_msg = {"role": "system", "content": get_system_prompt(seed.scenario, seed, model_id)}
    prev_len = 0

    # Steps 1..num_turns: user + assistant pairs
    cur_msgs = [sys_msg]
    for turn in range(1, num_turns + 1):
        # Add user message
        if turn == 1:
            user_content = _format_user_message(seed.scenario, seed.initial_question)
        else:
            idx = turn - 2
            user_content = seed.followup_messages[idx] if idx < len(seed.followup_messages) else ""
        cur_msgs.append({"role": "user", "content": user_content})

        cur_text = tokenizer.apply_chat_template(cur_msgs, tokenize=False, add_generation_prompt=False, **chat_template_kwargs)
        cur_len = _tok_len(cur_text)
        role_ids[prev_len:min(cur_len, S)] = ROLE_USER
        turn_ids[prev_len:min(cur_len, S)] = turn
        prev_len = cur_len

        # Add assistant message
        if turn < num_turns:
            asst_content = prior_responses[turn - 1]
        else:
            # Final turn: use the actual response
            asst_content = final_response

        cur_msgs.append({"role": "assistant", "content": asst_content})
        cur_text = tokenizer.apply_chat_template(cur_msgs, tokenize=False, add_generation_prompt=False, **chat_template_kwargs)
        cur_len = _tok_len(cur_text)
        role_ids[prev_len:min(cur_len, S)] = ROLE_ASSISTANT
        turn_ids[prev_len:min(cur_len, S)] = turn
        prev_len = cur_len

    # Any remaining tokens (from add_generation_prompt scaffolding)
    # stay as whatever the last role was — typically assistant for the final response

    return {
        "token_ids": np.array(full_ids, dtype=np.int32),
        "role_ids": role_ids,
        "turn_ids": turn_ids,
    }


# ---------------------------------------------------------------------------
# Batched generation for one turn across multiple conversations
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from a response."""
    return _THINK_RE.sub("", text).strip()


def _get_generation_suffix(tokenizer, chat_template_kwargs: dict) -> str:
    """Return the text the chat template appends after '<|im_start|>assistant\\n'.

    For thinking-enabled Qwen3.5 this is '<think>\\n' — the model's generated
    tokens start *after* this, so we need to prepend it to the decoded response
    to get the complete output.
    """
    dummy = [{"role": "user", "content": "x"}]
    with_prompt = tokenizer.apply_chat_template(
        dummy, tokenize=False, add_generation_prompt=True, **chat_template_kwargs,
    )
    without_prompt = tokenizer.apply_chat_template(
        dummy, tokenize=False, add_generation_prompt=False, **chat_template_kwargs,
    )
    # The generation prompt is everything after the last message
    gen_prompt = with_prompt[len(without_prompt):]
    # Strip the role header (e.g. '<|im_start|>assistant\n')
    # We want only what comes after it (e.g. '<think>\n')
    role_header = "<|im_start|>assistant\n"
    if gen_prompt.startswith(role_header):
        return gen_prompt[len(role_header):]
    return ""


@torch.no_grad()
def _generate_batch(
    prompts: List[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 2048,
) -> List[Tuple[str, int, int]]:
    """Generate responses for a batch of prompts.

    Returns list of (response_text, prompt_token_count, response_token_count).
    """
    raw_model = _unwrap(model)
    first_device = next(raw_model.parameters()).device

    enc = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True,
    )
    input_ids = enc["input_ids"].to(first_device)
    attention_mask = enc["attention_mask"].to(first_device)

    output_ids = raw_model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False,
    )

    results = []
    prompt_len = input_ids.shape[1]
    for i in range(len(prompts)):
        gen_ids = output_ids[i, prompt_len:]
        # Strip padding from generated tokens
        if tokenizer.pad_token_id is not None:
            gen_ids = gen_ids[gen_ids != tokenizer.pad_token_id]
        response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        # Per-sample prompt length (count non-pad tokens)
        sample_prompt_len = int(attention_mask[i].sum().item())
        results.append((response, sample_prompt_len, len(gen_ids)))

    return results


# ---------------------------------------------------------------------------
# Batched multi-turn runner
# ---------------------------------------------------------------------------

def run_batched(
    seeds: List[ConversationSeed],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    layers: List[int],
    model_id: str = "",
    max_new_tokens: int = 2048,
    num_turns: int = 5,
    batch_size: int = 8,
    extract_hidden_states: bool = True,
    chat_template_kwargs: Optional[Dict] = None,
) -> List[ConversationResult]:
    """Run multiple conversations batched by turn number.

    All conversations generate turn 1 together, then turn 2, etc.
    Hidden states are extracted per-sample (unbatched) since sequence
    lengths vary after generation.
    """
    chat_template_kwargs = chat_template_kwargs or {}
    gen_suffix = _get_generation_suffix(tokenizer, chat_template_kwargs)
    if gen_suffix:
        logger.info("  Generation suffix from chat template: %r", gen_suffix)
    tokenizer.padding_side = "left"
    n = len(seeds)

    # Initialize results and per-conversation response histories
    results = [ConversationResult(seed=s, model_id=model_id) for s in seeds]
    prior_responses: List[List[str]] = [[] for _ in range(n)]

    for turn in range(1, num_turns + 1):
        logger.info("  Turn %d/%d for %d conversations", turn, num_turns, n)

        # Build prompts for all conversations at this turn
        all_prompts = []
        for i, seed in enumerate(seeds):
            messages = build_turn_messages(
                seed, prior_responses[i], turn, model_id=model_id,
            )
            all_prompts.append(format_messages(messages, tokenizer, **chat_template_kwargs))

        # Generate in batches
        all_gen_results: List[Tuple[str, int, int]] = []
        for batch_start in range(0, n, batch_size):
            batch_prompts = all_prompts[batch_start:batch_start + batch_size]
            batch_results = _generate_batch(
                batch_prompts, model, tokenizer, max_new_tokens,
            )
            all_gen_results.extend(batch_results)

        # Store turn results (no hidden states yet)
        is_final_turn = (turn == num_turns)
        for i, (response, prompt_len, resp_len) in enumerate(all_gen_results):
            if turn == 1:
                user_msg = seeds[i].initial_question
            else:
                idx = turn - 2
                user_msg = seeds[i].followup_messages[idx] if idx < len(seeds[i].followup_messages) else ""

            # Prepend any text the chat template added after the role header
            # (e.g. '<think>\n' for thinking-enabled Qwen3.5)
            full_response = gen_suffix + response if gen_suffix else response

            results[i].turns.append(TurnResult(
                turn=turn,
                user_message=user_msg,
                assistant_response=full_response,
                prompt_token_count=prompt_len,
                response_token_count=resp_len,
            ))
            # Strip thinking from prior responses: Qwen3.5 was trained with
            # thinking stripped from earlier turns (only current turn thinks)
            prior_responses[i].append(strip_thinking(full_response))

        logger.info("    Turn %d done. Sample response length: %d tokens",
                     turn, all_gen_results[0][2])

    # Extract hidden states only on the final turn (full conversation in context)
    # Also compute per-token boundary annotations (role_ids, turn_ids)
    if extract_hidden_states and layers:
        logger.info("  Extracting hidden states + boundaries for %d conversations...", n)
        for i in range(n):
            final_prompt = all_prompts[i]
            final_response = results[i].turns[-1].assistant_response
            # Strip the generation suffix (e.g. '<think>\n\n</think>\n\n' for
            # Qwen3.5 no_thinking) from the stored response before concatenating
            # with the prompt — the prompt already includes it via
            # add_generation_prompt=True, so including it again would create a
            # duplicate (double <think> block) that the model never saw.
            response_for_extraction = final_response
            if gen_suffix and final_response.startswith(gen_suffix):
                response_for_extraction = final_response[len(gen_suffix):]
            full_text = final_prompt + response_for_extraction
            hs_dict, _ = _extract_hidden_states(full_text, model, tokenizer, layers)
            results[i].turns[-1].hidden_states = hs_dict

            # Compute token boundaries (uses the same response_for_extraction
            # so hidden_states and boundaries have matching token counts)
            boundaries = compute_token_boundaries(
                seeds[i], prior_responses[i], num_turns,
                model_id, tokenizer, response_for_extraction,
                **chat_template_kwargs,
            )
            results[i].turns[-1].boundaries = boundaries
        logger.info("  Hidden state extraction + boundaries done.")

    return results


# ---------------------------------------------------------------------------
# Single conversation (kept for testing / small runs)
# ---------------------------------------------------------------------------

def run_conversation(
    seed: ConversationSeed,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    layers: List[int],
    model_id: str = "",
    max_new_tokens: int = 2048,
    num_turns: int = 5,
    extract_hidden_states: bool = True,
    chat_template_kwargs: Optional[Dict] = None,
) -> ConversationResult:
    """Run a single multi-turn conversation (unbatched)."""
    results = run_batched(
        [seed], model, tokenizer, layers,
        model_id=model_id,
        max_new_tokens=max_new_tokens,
        num_turns=num_turns,
        batch_size=1,
        extract_hidden_states=extract_hidden_states,
        chat_template_kwargs=chat_template_kwargs,
    )
    return results[0]
