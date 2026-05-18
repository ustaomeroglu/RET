"""Prompt logging utility.

Saves the first N prompts (and optional generations) fed to the frozen LLM
as a JSONL file in the output directory, capturing exactly what the model sees.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import torch


class PromptLogger:
    """Accumulates the first *max_entries* prompts and writes them to JSONL.

    Two main entry points:

    * ``log_forward_pass_batch`` — decodes ``input_ids`` tensors back to text
      (for concat-and-chunk or document-mode pipelines).
    * ``log_prompts`` — accepts raw prompt strings (optionally with
      generated text for generation-mode scripts).

    Call ``save()`` once at the end to flush to disk.  All ``log_*`` methods
    are no-ops after *max_entries* have been recorded, so the overhead during
    long training runs is negligible.
    """

    def __init__(
        self,
        output_dir: str | Path,
        filename: str = "prompt_log.jsonl",
        max_entries: int = 50,
    ):
        self.output_dir = Path(output_dir)
        self.filename = filename
        self.max_entries = max_entries
        self._entries: list[dict] = []
        self._count = 0
        self._saved = False

    @property
    def is_full(self) -> bool:
        return self._count >= self.max_entries

    # ------------------------------------------------------------------
    # Logging from input_ids tensors (decode back to text)
    # ------------------------------------------------------------------

    def log_forward_pass_batch(
        self,
        input_ids: torch.Tensor,
        tokenizer,
        attention_mask: Optional[torch.Tensor] = None,
        source: str = "concat_chunk",
    ) -> None:
        """Log a batch of ``input_ids`` by decoding each sequence to text.

        Parameters
        ----------
        input_ids : (B, T) token IDs.
        tokenizer : HuggingFace tokenizer used for decoding.
        attention_mask : optional (B, T) mask (1 = valid, 0 = padding).
            When provided, ``num_tokens`` reports the valid count.
        source : tag stored in the JSONL (e.g. ``"concat_chunk"``, ``"document"``).
        """
        if self.is_full:
            return
        for i in range(input_ids.shape[0]):
            if self.is_full:
                break
            ids = input_ids[i]
            if attention_mask is not None:
                num_tokens = int(attention_mask[i].sum().item())
                ids = ids[:num_tokens]  # truncate padding before decoding
            else:
                num_tokens = int(ids.shape[0])
            text = tokenizer.decode(ids, skip_special_tokens=False)
            self._entries.append({
                "index": self._count,
                "mode": "forward_pass",
                "source": source,
                "prompt_text": text,
                "num_tokens": num_tokens,
            })
            self._count += 1

    # ------------------------------------------------------------------
    # Logging from raw prompt strings
    # ------------------------------------------------------------------

    def log_prompts(
        self,
        prompts: List[str],
        tokenizer=None,
        source: str = "individual",
        generations: Optional[List[str]] = None,
    ) -> None:
        """Log raw prompt strings (and optional generations).

        Parameters
        ----------
        prompts : list of fully-formatted prompt strings.
        tokenizer : optional; used to count tokens.  If ``None``,
            ``num_tokens`` is set to the character length as a fallback.
        source : tag stored in the JSONL.
        generations : if provided (same length as *prompts*), each entry
            is saved with ``mode="generation"`` including the generated text.
        """
        if self.is_full:
            return
        for idx, prompt in enumerate(prompts):
            if self.is_full:
                break
            if tokenizer is not None:
                n_tok = len(tokenizer.encode(prompt, add_special_tokens=False))
            else:
                n_tok = len(prompt)

            if generations is not None:
                gen = generations[idx]
                if tokenizer is not None:
                    n_gen = len(tokenizer.encode(gen, add_special_tokens=False))
                else:
                    n_gen = len(gen)
                self._entries.append({
                    "index": self._count,
                    "mode": "generation",
                    "source": source,
                    "prompt_text": prompt,
                    "generation_text": gen,
                    "num_tokens_prompt": n_tok,
                    "num_tokens_generation": n_gen,
                })
            else:
                self._entries.append({
                    "index": self._count,
                    "mode": "forward_pass",
                    "source": source,
                    "prompt_text": prompt,
                    "num_tokens": n_tok,
                })
            self._count += 1

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> Optional[Path]:
        """Write accumulated entries to JSONL.  Idempotent."""
        if self._saved or not self._entries:
            return None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / self.filename
        with open(path, "w") as f:
            for entry in self._entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._saved = True
        return path
