"""Async Claude client + three prompt templates for interp score.

Three roles:
    describe_from_codes  — given a naming table + per-token code sequence,
                           describe what's happening over time (≤N words).
    describe_from_text   — given the raw text, describe what's happening over
                           time (≤N words).
    judge_similarity     — given the two descriptions, score similarity 1–10
                           + short rationale (JSON output).

All three calls use temperature=0 with bounded retry/backoff. Pattern mirrors
``evaluation/datasets/sycophancy/judge.py``.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-20250514"
_MAX_RETRIES = 5
_RETRY_BASE = 2.0


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_CODES_SYS = (
    "You are describing the mental-state trajectory of a language model as "
    "it processes a chat prompt containing a math problem and generates its "
    "response. You will receive a naming table defining mental-state groups "
    "and clusters, followed by a per-token sequence of state codes in the "
    "form 'G<group>.C<cluster>'. Using ONLY the naming table and the "
    "sequence, describe the model's mental state and how it evolves from "
    "the first token to the last. Output ONLY the description, no preamble, "
    "no bullets. Stay within the word budget."
)

_SAE_CODES_SYS = (
    "You are describing the mental-state trajectory of a language model as "
    "it processes a chat prompt containing a math problem and generates its "
    "response, based on its SAE latent activations. You will receive a "
    "naming table mapping SAE latent IDs to the token strings that most "
    "strongly activate them, followed by a per-token sequence where each "
    "token is represented by its top-K active latent IDs. Using ONLY this "
    "information, describe the model's mental state and how it evolves from "
    "the first token to the last. Output ONLY the description, no preamble, "
    "no bullets. Stay within the word budget."
)

_TEXT_SYS = (
    "You are describing the mental-state trajectory of a language model as "
    "it processes a chat prompt containing a math problem and generates its "
    "response. Given the full text from first input token to last output "
    "token, describe the model's mental state and how it evolves over time. "
    "Narrate the trajectory as if to someone who cannot see the page — "
    "describe what the model is doing cognitively, not what is written. "
    "Output ONLY the description, no preamble, no bullets. Stay within the "
    "word budget."
)


# ----- afterassistant variants -----
# When the analysis is restricted to the assistant's generated response only
# (scaffolding / user prompt stripped), these prompts replace the full-chat
# versions. The framing matches: we're describing only the model's response.

_CODES_SYS_AFTER = (
    "You are describing the mental-state trajectory of a language model "
    "while it generates a response to a math problem. You will receive a "
    "naming table defining mental-state groups and clusters, followed by a "
    "sequence of state codes in the form 'G<group>.C<cluster>' covering "
    "only the tokens of the model's generated response. Using ONLY the "
    "naming table and the sequence, describe the model's mental state and "
    "how it evolves from the first response token to the last. Output ONLY "
    "the description, no preamble, no bullets. Stay within the word budget."
)

_SAE_CODES_SYS_AFTER = (
    "You are describing the mental-state trajectory of a language model "
    "while it generates a response to a math problem, based on its SAE "
    "latent activations. You will receive a naming table mapping SAE latent "
    "IDs to the token strings that most strongly activate them, followed "
    "by a sequence covering only the tokens of the model's generated "
    "response. Using ONLY this information, describe the model's mental "
    "state and how it evolves from the first response token to the last. "
    "Output ONLY the description, no preamble, no bullets. Stay within the "
    "word budget."
)

_TEXT_SYS_AFTER = (
    "You are describing the mental-state trajectory of a language model "
    "while it generates a response to a math problem. Given the text of "
    "the model's response (from the first generated token to the last), "
    "describe the model's mental state and how it evolves over time. "
    "Narrate the trajectory as if to someone who cannot see the page — "
    "describe what the model is doing cognitively, not what is written. "
    "Output ONLY the description, no preamble, no bullets. Stay within "
    "the word budget."
)

_JUDGE_SYS = (
    "You are comparing two short descriptions of the same underlying math-"
    "problem text. Rate how closely they describe the same content on a 1–10 "
    "scale:\n"
    "10 = near-identical content coverage\n"
    " 8 = strong overlap, minor differences\n"
    " 6 = moderate overlap, some mismatched detail\n"
    " 4 = weak overlap, mostly different focus\n"
    " 2 = very little overlap\n"
    " 1 = unrelated\n"
    "Output STRICT JSON only with keys 'score' (integer 1–10) and 'rationale' "
    "(one short sentence). No other text."
)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

@dataclass
class MISLLM:
    """Async Anthropic client wrapper with bounded concurrency."""
    client: object
    model: str
    semaphore: asyncio.Semaphore

    @classmethod
    def create(cls, model: str = DEFAULT_MODEL, max_concurrent: int = 5) -> "MISLLM":
        from evaluation.llm_client import create_anthropic_client

        return cls(
            client=create_anthropic_client(),
            model=model,
            semaphore=asyncio.Semaphore(max_concurrent),
        )

    async def _call(self, system: str, user: str, max_tokens: int) -> str:
        from evaluation.llm_client import call_with_retry
        async with self.semaphore:
            return await call_with_retry(
                self.client, system, user,
                model=self.model, max_tokens=max_tokens,
                max_retries=_MAX_RETRIES, backoff_base=_RETRY_BASE,
            )


# ---------------------------------------------------------------------------
# Prompt builders + role wrappers
# ---------------------------------------------------------------------------

def build_codes_prompt(
    naming_table: str,
    sequence: str,
    word_limit: int,
    is_sae: bool = False,
    variant: str = "full",
) -> tuple[str, str]:
    if variant == "afterassistant":
        sys_prompt = _SAE_CODES_SYS_AFTER if is_sae else _CODES_SYS_AFTER
    else:
        sys_prompt = _SAE_CODES_SYS if is_sae else _CODES_SYS
    user = (
        f"{naming_table}\n\n"
        f"{sequence}\n\n"
        f"Describe what is happening in ≤{word_limit} words."
    )
    return sys_prompt, user


def build_text_prompt(
    text: str,
    word_limit: int,
    variant: str = "full",
) -> tuple[str, str]:
    sys_prompt = _TEXT_SYS_AFTER if variant == "afterassistant" else _TEXT_SYS
    return sys_prompt, (
        f"# Text\n\n{text}\n\nDescribe what is happening in ≤{word_limit} words."
    )


def build_judge_prompt(desc_codes: str, desc_text: str) -> tuple[str, str]:
    user = (
        f"# Description A (derived from state codes)\n{desc_codes}\n\n"
        f"# Description B (derived from text)\n{desc_text}\n\n"
        "Score their similarity. Respond with STRICT JSON only."
    )
    return _JUDGE_SYS, user


async def describe_from_codes(
    llm: MISLLM,
    naming_table: str,
    sequence: str,
    word_limit: int = 80,
    is_sae: bool = False,
    max_tokens: int = 400,
    variant: str = "full",
) -> str:
    sys_p, user_p = build_codes_prompt(
        naming_table, sequence, word_limit, is_sae=is_sae, variant=variant,
    )
    return await llm._call(sys_p, user_p, max_tokens)


async def describe_from_text(
    llm: MISLLM,
    text: str,
    word_limit: int = 80,
    max_tokens: int = 400,
    variant: str = "full",
) -> str:
    sys_p, user_p = build_text_prompt(text, word_limit, variant=variant)
    return await llm._call(sys_p, user_p, max_tokens)


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

_SCORE_RE = re.compile(r'"score"\s*:\s*([0-9]+)')
_RATIONALE_RE = re.compile(r'"rationale"\s*:\s*"([^"]*)"')


def _parse_judge_json(raw: str) -> Dict[str, object]:
    """Best-effort parse: prefer strict JSON, fall back to regex."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*", "", cleaned)
        cleaned = cleaned.rstrip("`").strip()
    try:
        data = json.loads(cleaned)
        score = int(data["score"])
        rationale = str(data.get("rationale", ""))
        return {"score": max(1, min(10, score)), "rationale": rationale, "raw": raw}
    except Exception:
        pass

    m = _SCORE_RE.search(raw)
    if m:
        score = max(1, min(10, int(m.group(1))))
    else:
        logger.warning("Judge: could not parse score from %r — defaulting to 0", raw)
        score = 0

    r = _RATIONALE_RE.search(raw)
    rationale = r.group(1) if r else ""
    return {"score": score, "rationale": rationale, "raw": raw}


async def judge_similarity(
    llm: MISLLM,
    desc_codes: str,
    desc_text: str,
    max_tokens: int = 200,
) -> Dict[str, object]:
    sys_p, user_p = build_judge_prompt(desc_codes, desc_text)
    raw = await llm._call(sys_p, user_p, max_tokens)
    return _parse_judge_json(raw)
