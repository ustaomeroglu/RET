"""
Prompt inspection utility for probing / RET experiments.

Saves detailed token-level breakdowns of the first N samples as a JSONL file.
Captures: raw prompt text, every token (id + decoded string), special-token
flags, system / user / assistant boundaries, generation tokens (when present),
attention masks, and per-sample metadata.

All writes are **immediate** — each sample is flushed to disk as it arrives,
so you can inspect the file while training / extraction is still running.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _token_detail(token_id: int, tokenizer) -> Dict[str, Any]:
    """Return a dict describing a single token."""
    decoded = tokenizer.decode([token_id])
    tok_str = tokenizer.convert_ids_to_tokens([token_id])[0]
    is_special = token_id in tokenizer.all_special_ids
    return {
        "id": token_id,
        "token": tok_str,
        "decoded": decoded,
        "is_special": is_special,
    }


def _identify_boundaries(
    token_ids: List[int],
    tokenizer,
    system_text: str,
    include_gpt_info: bool = True,
) -> Dict[str, Any]:
    """Identify system / user / assistant-start regions in a token sequence.

    When ``include_gpt_info=False`` the GPT-OSS system block is stripped from
    the reference strings before computing boundaries, mirroring the prompts
    that were actually fed to the model.
    """
    from evaluation.probing.model_utils import strip_harmony_system_block

    info: Dict[str, Any] = {}

    if not (hasattr(tokenizer, "chat_template") and tokenizer.chat_template):
        info["has_chat_template"] = False
        return info

    info["has_chat_template"] = True
    info["include_gpt_info"] = include_gpt_info

    sys_only_str = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_text}],
        tokenize=False, add_generation_prompt=False,
    )
    if not include_gpt_info:
        sys_only_str = strip_harmony_system_block(sys_only_str)
    sys_only_ids = tokenizer.encode(sys_only_str, add_special_tokens=False)
    info["system_prefix_len"] = len(sys_only_ids)

    sys_user_stub = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_text},
         {"role": "user", "content": ""}],
        tokenize=False, add_generation_prompt=False,
    )
    if not include_gpt_info:
        sys_user_stub = strip_harmony_system_block(sys_user_stub)
    sys_user_stub_ids = tokenizer.encode(sys_user_stub, add_special_tokens=False)
    info["user_content_start"] = len(sys_user_stub_ids)

    spl = info["system_prefix_len"]
    labels = ["developer_instructions" if i < spl else "user_or_assistant"
              for i in range(len(token_ids))]
    info["position_labels"] = labels

    return info


def _special_ids_dict(tokenizer) -> Dict[str, Any]:
    return {
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
    }


# ---------------------------------------------------------------------------
# Entry-point 1: from prompt strings (experiment runners, non-cached)
# ---------------------------------------------------------------------------

def save_prompt_samples(
    output_path: str,
    prompts: List[str],
    tokenizer,
    system_text: str = "",
    labels=None,
    generations: Optional[List[str]] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,
    max_samples: int = 100,
    source: str = "unknown",
    include_gpt_info: bool = True,
) -> str:
    """Save the first *max_samples* prompts with full token-level detail.

    Writes are **streamed** — each sample is flushed immediately.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    n = min(len(prompts), max_samples)
    special_ids = _special_ids_dict(tokenizer)

    with open(output_path, "w") as f:
        for i in range(n):
            prompt = prompts[i]
            ids = tokenizer.encode(prompt, add_special_tokens=False)

            tokens_detail = [_token_detail(tid, tokenizer) for tid in ids]
            boundaries = (
                _identify_boundaries(ids, tokenizer, system_text,
                                     include_gpt_info=include_gpt_info)
                if system_text else {}
            )

            entry: Dict[str, Any] = {
                "index": i,
                "source": source,
                "prompt_text": prompt,
                "num_tokens": len(ids),
                "token_ids": ids,
                "tokens": tokens_detail,
                "boundaries": boundaries,
                "special_token_ids": special_ids,
            }

            if labels is not None:
                try:
                    lbl = labels[i]
                    entry["label"] = int(lbl) if hasattr(lbl, "item") else lbl
                except (IndexError, KeyError):
                    pass

            if generations is not None and i < len(generations):
                gen_text = generations[i]
                gen_ids = tokenizer.encode(gen_text, add_special_tokens=False)
                entry["generation_text"] = gen_text
                entry["generation_num_tokens"] = len(gen_ids)
                entry["generation_token_ids"] = gen_ids
                entry["generation_tokens"] = [
                    _token_detail(tid, tokenizer) for tid in gen_ids
                ]

            if metadata is not None and i < len(metadata):
                entry["metadata"] = metadata[i]

            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
            f.flush()

    logger.info("Saved %d detailed prompt samples to %s", n, output_path)
    return output_path


# ---------------------------------------------------------------------------
# Entry-point 2: from batched forward-pass tensors
# ---------------------------------------------------------------------------

def save_forward_pass_samples(
    output_path: str,
    input_ids,
    attention_mask,
    tokenizer,
    system_text: str = "",
    max_samples: int = 100,
    source: str = "forward_pass",
) -> str:
    """Save token-level detail from batched ``input_ids`` / ``attention_mask``.

    Writes are **streamed** — each sample is flushed immediately.
    """
    import torch
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.cpu()
        attention_mask = attention_mask.cpu()

    special_ids = _special_ids_dict(tokenizer)
    n = min(int(input_ids.shape[0]), max_samples)

    with open(output_path, "w") as f:
        for i in range(n):
            ids_row = input_ids[i].tolist()
            mask_row = attention_mask[i].tolist()
            real_len = int(sum(mask_row))

            real_ids = [tid for tid, m in zip(ids_row, mask_row) if m]
            tokens_detail = [_token_detail(tid, tokenizer) for tid in real_ids]
            full_text = tokenizer.decode(real_ids, skip_special_tokens=False)

            boundaries = (
                _identify_boundaries(real_ids, tokenizer, system_text)
                if system_text else {}
            )

            entry: Dict[str, Any] = {
                "index": i,
                "source": source,
                "full_decoded_text": full_text,
                "num_real_tokens": real_len,
                "total_seq_len": len(ids_row),
                "num_padding_tokens": len(ids_row) - real_len,
                "real_token_ids": real_ids,
                "tokens": tokens_detail,
                "boundaries": boundaries,
                "special_token_ids": special_ids,
                "attention_mask": mask_row,
            }
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
            f.flush()

    logger.info("Saved %d forward-pass samples to %s", n, output_path)
    return output_path


# ---------------------------------------------------------------------------
# Streaming accumulator — writes each sample to disk as it arrives
# ---------------------------------------------------------------------------

class SampleAccumulator:
    """Collects input_ids / attention_mask across batches.

    ``add_batch`` only stashes raw integer lists — it never touches the
    tokenizer, so it is safe to call from threads that share a fast
    tokenizer.  All tokenizer work (decode, token detail, boundaries)
    happens in ``close()``, which writes the JSONL and flushes to disk.
    """

    def __init__(
        self,
        output_path: str,
        tokenizer,
        system_text: str = "",
        max_samples: int = 100,
        source: str = "forward_pass",
        include_gpt_info: bool = True,
    ):
        self.output_path = output_path
        self.tokenizer = tokenizer
        self.system_text = system_text
        self.max_samples = max_samples
        self.source = source
        self.include_gpt_info = include_gpt_info
        self._rows: list = []  # list of (ids_list, mask_list)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @property
    def is_full(self) -> bool:
        return len(self._rows) >= self.max_samples

    def add_batch(self, input_ids, attention_mask) -> None:
        """Stash raw token ids + mask. No tokenizer calls (thread-safe)."""
        import torch
        if self.is_full:
            return

        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.cpu()
            attention_mask = attention_mask.cpu()

        B = int(input_ids.shape[0])
        for j in range(B):
            if self.is_full:
                break
            self._rows.append((
                input_ids[j].tolist(),
                attention_mask[j].tolist(),
            ))

    def close(self) -> None:
        """Decode all stashed samples and write the JSONL file."""
        if not self._rows:
            return

        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        special_ids = _special_ids_dict(self.tokenizer)

        with open(self.output_path, "w") as f:
            for idx, (ids_row, mask_row) in enumerate(self._rows):
                real_len = int(sum(mask_row))
                real_ids = [tid for tid, m in zip(ids_row, mask_row) if m]
                tokens_detail = [_token_detail(tid, self.tokenizer)
                                 for tid in real_ids]
                full_text = self.tokenizer.decode(
                    real_ids, skip_special_tokens=False)

                boundaries = (
                    _identify_boundaries(
                        real_ids, self.tokenizer, self.system_text,
                        include_gpt_info=self.include_gpt_info)
                    if self.system_text else {}
                )

                entry = {
                    "index": idx,
                    "source": self.source,
                    "full_decoded_text": full_text,
                    "num_real_tokens": real_len,
                    "total_seq_len": len(ids_row),
                    "num_padding_tokens": len(ids_row) - real_len,
                    "real_token_ids": real_ids,
                    "tokens": tokens_detail,
                    "boundaries": boundaries,
                    "special_token_ids": special_ids,
                    "attention_mask": mask_row,
                }
                f.write(
                    json.dumps(entry, ensure_ascii=False, default=str) + "\n"
                )
                f.flush()

        logger.info("SampleAccumulator: %d samples saved to %s",
                    len(self._rows), self.output_path)
        self._rows.clear()


# ---------------------------------------------------------------------------
# Entry-point 3: RET training data inspection (reads Arrow dataset directly)
# ---------------------------------------------------------------------------

def save_ret_training_samples(
    data_dir: str,
    output_path: str,
    llm_name: str,
    system_message: str,
    scaffolding: str = "chatify",
    max_samples: int = 100,
    max_seq_length: int = 512,
    trust_remote_code: bool = True,
    include_gpt_info: bool = True,
) -> str:
    """Inspect RET training data by loading the Arrow dataset and tokenizer.

    This is a standalone function that does NOT require the trainer or
    ``scripts/train.py`` — it reads the prepared Arrow datasets directly
    and saves the first *max_samples* samples with full token-level detail.

    Call this from ``train_ret.py`` BEFORE launching ``scripts/train.py``.
    """
    from pathlib import Path
    from datasets import load_from_disk
    from transformers import AutoTokenizer

    tok_kwargs = {}
    if trust_remote_code:
        tok_kwargs["trust_remote_code"] = True
    tokenizer = AutoTokenizer.from_pretrained(llm_name, **tok_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    train_dir = Path(data_dir) / "train"
    if not train_dir.exists():
        logger.warning("No train/ directory at %s — skipping RET sample logging", data_dir)
        return output_path

    dataset = load_from_disk(str(train_dir))
    n = min(len(dataset), max_samples)
    special_ids = _special_ids_dict(tokenizer)

    has_chat = hasattr(tokenizer, "chat_template") and tokenizer.chat_template

    with open(output_path, "w") as f:
        for i in range(n):
            row = dataset[i]
            raw_text = row.get("text", "")

            if scaffolding == "chatify" and has_chat:
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": raw_text},
                ]
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False)
                if not include_gpt_info:
                    from evaluation.probing.model_utils import strip_harmony_system_block
                    formatted = strip_harmony_system_block(formatted)
            elif scaffolding == "proper_chat" and "messages" in row:
                messages = row["messages"]
                if system_message:
                    messages = [{"role": "system", "content": system_message}] + messages
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False)
                if not include_gpt_info:
                    from evaluation.probing.model_utils import strip_harmony_system_block
                    formatted = strip_harmony_system_block(formatted)
            else:
                formatted = raw_text

            ids = tokenizer.encode(
                formatted, add_special_tokens=False,
                truncation=True, max_length=max_seq_length,
            )
            tokens_detail = [_token_detail(tid, tokenizer) for tid in ids]
            boundaries = (
                _identify_boundaries(ids, tokenizer, system_message,
                                     include_gpt_info=include_gpt_info)
                if system_message and has_chat else {}
            )

            entry = {
                "index": i,
                "source": "ret_training_data",
                "raw_text": raw_text,
                "formatted_prompt": formatted,
                "num_tokens": len(ids),
                "truncated": len(ids) == max_seq_length,
                "token_ids": ids,
                "tokens": tokens_detail,
                "boundaries": boundaries,
                "special_token_ids": special_ids,
            }

            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
            f.flush()

    logger.info("Saved %d RET training samples to %s", n, output_path)
    return output_path
