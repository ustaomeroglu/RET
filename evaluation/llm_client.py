"""Shared async Anthropic client + retry-with-backoff helper.

Used by both ``evaluation/datasets/sycophancy/judge.py`` (judge prompts) and
``evaluation/mis/llm.py`` (interp-score prompts) so that
client construction, API-key validation, and retry/backoff are defined in
exactly one place.
"""
from __future__ import annotations

import asyncio
import logging
import os

logger = logging.getLogger(__name__)


def create_anthropic_client():
    """Return a fresh ``AsyncAnthropic`` client.

    Raises ``RuntimeError`` if ``ANTHROPIC_API_KEY`` is unset.
    """
    import anthropic  # local import: optional dep

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set")
    return anthropic.AsyncAnthropic(api_key=api_key)


async def call_with_retry(
    client,
    system: str,
    user: str,
    *,
    model: str,
    max_tokens: int,
    temperature: float = 0.0,
    max_retries: int = 5,
    backoff_base: int = 2,
    backoff_mult: int = 2,
) -> str:
    """Call ``client.messages.create`` with exponential-backoff retries.

    Returns the assistant's text content (stripped). Backoff sleep is
    ``backoff_base ** attempt * backoff_mult`` seconds.
    """
    for attempt in range(max_retries):
        try:
            message = await client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return message.content[0].text.strip()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = backoff_base ** attempt * backoff_mult
            logger.warning(
                "Anthropic API error (attempt %d/%d): %s — retrying in %ds",
                attempt + 1, max_retries, e, wait,
            )
            await asyncio.sleep(wait)
    raise RuntimeError("Unreachable: exhausted retries without returning or raising")
