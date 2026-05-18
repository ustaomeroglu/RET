"""Custom token-by-token generation loop with steering hook."""
from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation.unsupervised_steering.hooks import SteeringHook

logger = logging.getLogger(__name__)


def _sample_token(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
) -> torch.Tensor:
    """Sample a single token from logits with temperature and nucleus sampling.

    Args:
        logits: (1, vocab) raw logits
        temperature: sampling temperature (0 = greedy)
        top_p: nucleus probability mass

    Returns:
        (1, 1) token id tensor
    """
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)

    probs = F.softmax(logits / temperature, dim=-1)

    # Nucleus (top-p) filtering
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
    cumsum = sorted_probs.cumsum(dim=-1)
    mask = cumsum - sorted_probs > top_p
    sorted_probs[mask] = 0.0
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)

    # Sample from filtered distribution
    token_pos = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_idx.gather(-1, token_pos)


class SteeredGenerator:
    """Token-by-token generation with a steering hook on a target layer."""

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        hook: SteeringHook,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.hook = hook

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> tuple[str, list[dict]]:
        """Generate text with optional unsupervised steering.

        Args:
            prompt: input text
            max_new_tokens: maximum tokens to generate
            temperature: sampling temperature
            top_p: nucleus probability mass

        Returns:
            (generated_text, per_token_diagnostics)
        """
        device = next(self.model.parameters()).device

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)
        logger.info("Prompt: %d tokens", input_ids.shape[1])

        # Phase 1: Process prompt (hook passively collects h_history)
        self.hook.enabled = False
        outputs = self.model(input_ids, use_cache=True)
        past_kv = outputs.past_key_values
        logger.info(
            "Prompt processed, h_history has %d entries", len(self.hook.h_history),
        )

        # Compute z features for prompt tokens (for HTML visualization)
        self.hook.compute_prompt_features()
        # Store decoded prompt tokens in prompt diagnostics
        prompt_ids = input_ids[0].tolist()
        for i, d in enumerate(self.hook.prompt_diagnostics):
            d["token_id"] = int(prompt_ids[i])
            d["token"] = self.tokenizer.decode([prompt_ids[i]])

        # Phase 2: Token-by-token generation (hook steers when rules fire)
        self.hook.enabled = True
        next_token = input_ids[:, -1:]  # seed with last prompt token
        generated_ids: list[int] = []

        for step in range(max_new_tokens):
            outputs = self.model(
                next_token,
                past_key_values=past_kv,
                use_cache=True,
            )
            # Hook has intercepted & possibly modified h_t at target layer
            logits = outputs.logits[:, -1, :]  # (1, vocab)
            next_token = _sample_token(logits, temperature, top_p)
            token_id = next_token.item()
            generated_ids.append(token_id)
            past_kv = outputs.past_key_values

            # Update diagnostics with token text and id
            if self.hook.diagnostics and step < len(self.hook.diagnostics):
                self.hook.diagnostics[step]["token_id"] = int(token_id)
                self.hook.diagnostics[step]["token"] = self.tokenizer.decode(
                    [token_id]
                )

            if token_id == self.tokenizer.eos_token_id:
                break

        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        logger.info("Generated %d tokens", len(generated_ids))
        return text, self.hook.diagnostics
