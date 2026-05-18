from __future__ import annotations

import json
import logging

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers import AutoTokenizer

from pathlib import Path

from ret.config import DataConfig, LLMConfig
from evaluation.datasets.corpus_source import load_corpus

logger = logging.getLogger(__name__)


def maybe_apply_chat_template(
    text_or_messages: str | list[dict],
    tokenizer,
    data_config: DataConfig,
) -> str:
    """Apply chat template scaffolding based on ``data_config.scaffolding``.

    Parameters
    ----------
    text_or_messages
        For ``scaffolding='raw'`` or ``'chatify'``: a plain text string.
        For ``scaffolding='proper_chat'``: a list of ``{role, content}`` dicts.

    Raises
    ------
    ValueError
        If a chat scaffolding mode is used but the tokenizer has no
        ``chat_template``.
    RuntimeError
        If ``apply_chat_template`` raises (e.g. template does not support
        the 'system' role).
    """
    if data_config.scaffolding == "raw":
        return text_or_messages

    # chatify or proper_chat — both need a chat template
    if not (hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None):
        raise ValueError(
            f"data.scaffolding='{data_config.scaffolding}' but tokenizer "
            f"({type(tokenizer).__name__}) has no chat_template. "
            f"Set scaffolding='raw' for base models."
        )

    if data_config.scaffolding == "chatify":
        # Wrap entire text as a single user message
        messages = [
            {"role": "system", "content": data_config.system_message},
            {"role": "user", "content": text_or_messages},
        ]
    elif data_config.scaffolding == "proper_chat":
        # Use structured messages directly, prepend system message
        messages = [
            {"role": "system", "content": data_config.system_message},
            *text_or_messages,
        ]
    else:
        raise ValueError(f"Unknown scaffolding mode: {data_config.scaffolding!r}")

    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            **data_config.chat_template_kwargs,
        )
    except Exception as e:
        raise RuntimeError(
            f"apply_chat_template failed: {e}. Check that the tokenizer's "
            f"chat_template supports the required roles."
        ) from e


def _compute_scaffold_tokens(data_config: DataConfig, tokenizer):
    """Pre-compute scaffold prefix/suffix token arrays for chunked+chatify mode.

    Returns (prefix_ids, suffix_ids) as numpy int64 arrays, or (None, None)
    if scaffolding is 'raw'. Only used in chunked mode with 'chatify'.
    (proper_chat is forbidden in chunked mode — validated in config.)
    """
    if data_config.scaffolding == "raw":
        return None, None

    # Validate tokenizer has chat_template
    if not (hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None):
        raise ValueError(
            f"data.scaffolding='{data_config.scaffolding}' but tokenizer "
            f"({type(tokenizer).__name__}) has no chat_template. "
            f"Set scaffolding='raw' for base models."
        )

    # Use a unique marker to split the scaffolded text into prefix and suffix
    marker = "\x00SCAFFOLD_MARKER\x00"
    scaffolded = maybe_apply_chat_template(marker, tokenizer, data_config)
    parts = scaffolded.split(marker, 1)
    prefix_ids = np.array(
        tokenizer.encode(parts[0], add_special_tokens=False), dtype=np.int64,
    )
    suffix_ids = np.array(
        tokenizer.encode(parts[1], add_special_tokens=False) if len(parts) > 1 else [],
        dtype=np.int64,
    )
    logger.info(
        "Chat scaffold: prefix=%d tokens, suffix=%d tokens, "
        "effective seq_length=%d (chunk_length=%d)",
        len(prefix_ids), len(suffix_ids),
        len(prefix_ids) + data_config.chunk_length + len(suffix_ids),
        data_config.chunk_length,
    )
    return prefix_ids, suffix_ids


class TokenChunkDataset(Dataset):
    """Concatenate-and-chunk dataset for language modeling.

    Loads a text dataset, tokenizes everything, concatenates all tokens into
    a flat array, then slices into non-overlapping chunks of fixed length.
    If scaffolding='chatify', each chunk is wrapped in chat template tokens at
    retrieval time (scaffold-per-chunk, not per-document).
    """

    def __init__(
        self,
        data_config: DataConfig,
        llm_config: LLMConfig,
        split: str | None = "train",
        max_chunks: int | None = None,
    ):
        self.chunk_length = data_config.chunk_length

        tok_kwargs: dict = {}
        if llm_config.trust_remote_code:
            tok_kwargs["trust_remote_code"] = True
        tokenizer = AutoTokenizer.from_pretrained(llm_config.name, **tok_kwargs)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dataset = load_corpus(
            hf_dataset=data_config.hf_dataset,
            local_dir=data_config.local_dir,
        )
        text_field = data_config.text_field

        # Tokenize raw text — NO scaffolding here (applied per-chunk later)
        all_tokens: list[int] = []
        target_total = (max_chunks or 10_000_000) * data_config.chunk_length
        for example in dataset:
            text = example[text_field]
            if not isinstance(text, str) or not text.strip():
                continue
            tokens = tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend(tokens)
            if len(all_tokens) >= target_total:
                break

        # Chunk into non-overlapping segments
        n_chunks = len(all_tokens) // data_config.chunk_length
        usable = n_chunks * data_config.chunk_length
        all_tokens = all_tokens[:usable]
        self.tokens = np.array(all_tokens, dtype=np.int64).reshape(n_chunks, data_config.chunk_length)

        # Pre-compute scaffold prefix/suffix tokens
        self.scaffold_prefix, self.scaffold_suffix = _compute_scaffold_tokens(
            data_config, tokenizer,
        )

        # Train/val split (skip if split is None)
        if split is not None:
            val_size = max(1, int(n_chunks * data_config.val_fraction))
            if split == "val":
                self.tokens = self.tokens[-val_size:]
            else:
                self.tokens = self.tokens[:-val_size]

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        chunk = self.tokens[idx]
        if self.scaffold_prefix is not None:
            ids = np.concatenate([self.scaffold_prefix, chunk, self.scaffold_suffix])
            return {"input_ids": torch.from_numpy(ids.copy())}
        return {"input_ids": torch.from_numpy(chunk.copy())}


class StreamingTokenChunkDataset(IterableDataset):
    """Streaming concatenate-and-chunk dataset.

    Lazily tokenizes from a HuggingFace streaming dataset and yields
    fixed-length chunks. No __len__ — runs until the stream exhausts.
    If scaffolding='chatify', each chunk is wrapped in chat template tokens.
    """

    def __init__(self, data_config: DataConfig, llm_config: LLMConfig, seed: int = 42):
        self.data_config = data_config
        self.llm_config = llm_config
        self.chunk_length = data_config.chunk_length
        self.seed = seed

    def __iter__(self):
        tok_kwargs: dict = {}
        if self.llm_config.trust_remote_code:
            tok_kwargs["trust_remote_code"] = True
        tokenizer = AutoTokenizer.from_pretrained(self.llm_config.name, **tok_kwargs)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dataset = load_corpus(
            hf_dataset=self.data_config.hf_dataset,
            local_dir=self.data_config.local_dir,
        )
        if self.data_config.hf_dataset is not None:
            dataset = dataset.shuffle(
                buffer_size=self.data_config.shuffle_buffer_size,
                seed=self.seed,
            )
        else:
            dataset = dataset.shuffle(seed=self.seed)

        # Pre-compute scaffold tokens
        scaffold_prefix, scaffold_suffix = _compute_scaffold_tokens(
            self.data_config, tokenizer,
        )

        text_field = self.data_config.text_field
        buffer: list[int] = []
        for example in dataset:
            text = example[text_field]
            if not isinstance(text, str) or not text.strip():
                continue
            # Tokenize raw text — NO scaffolding here
            tokens = tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(tokens)
            while len(buffer) >= self.chunk_length:
                chunk = buffer[: self.chunk_length]
                buffer = buffer[self.chunk_length :]
                if scaffold_prefix is not None:
                    chunk = scaffold_prefix.tolist() + chunk + scaffold_suffix.tolist()
                yield {"input_ids": torch.tensor(chunk, dtype=torch.long)}


def build_dataloader(
    data_config: DataConfig,
    llm_config: LLMConfig,
    split: str = "train",
    max_chunks: int | None = None,
    batch_size: int = 32,
    shuffle: bool | None = None,
    seed: int = 42,
) -> DataLoader:
    """Build a DataLoader for token chunks (chunked/HF mode)."""
    if split == "train":
        # Always stream for training
        dataset = StreamingTokenChunkDataset(data_config, llm_config, seed=seed)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )

    # Val: eagerly load a small number of chunks with proper train/val split
    dataset = TokenChunkDataset(
        data_config, llm_config, split="val", max_chunks=data_config.val_chunks,
    )

    if shuffle is None:
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=data_config.num_workers,
        pin_memory=True,
        drop_last=False,
    )


class PaddedPromptDataset(Dataset):
    """Dataset of tokenized prompts stored as variable-length tensors.

    Unlike TokenChunkDataset which concatenates and chunks text into fixed-length
    segments, this dataset preserves individual prompt boundaries with padding.
    Suitable for probing experiments where each prompt must remain separate.
    """

    def __init__(self, prompts: list[str], tokenizer, max_length: int = 512):
        self.samples: list[torch.Tensor] = []
        for prompt in prompts:
            tokens = tokenizer.encode(
                prompt, add_special_tokens=False,
                truncation=True, max_length=max_length,
            )
            self.samples.append(torch.tensor(tokens, dtype=torch.long))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {"input_ids": self.samples[idx]}


def collate_padded(batch: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    """Pad a batch of variable-length token sequences (right-padding)."""
    ids = [item["input_ids"] for item in batch]
    max_len = max(len(x) for x in ids)
    padded = torch.full((len(ids), max_len), pad_id, dtype=torch.long)
    masks = torch.zeros(len(ids), max_len, dtype=torch.long)
    for i, x in enumerate(ids):
        padded[i, : len(x)] = x
        masks[i, : len(x)] = 1
    return {"input_ids": padded, "attention_mask": masks}


def build_padded_dataloader(
    data_config: DataConfig,
    llm_config: LLMConfig,
    split: str = "train",
    batch_size: int = 32,
    shuffle: bool | None = None,
    seed: int = 42,
    drop_last: bool | None = None,
) -> DataLoader:
    """Build a DataLoader for variable-length document sequences with padding.

    Loads a pre-saved Arrow dataset (via data_config.local_dir/{split}/), applies
    chat template wrapping per-document, tokenizes with truncation at
    data_config.max_seq_length, and returns a padded DataLoader with attention_mask.
    """
    tok_kwargs: dict = {}
    if llm_config.trust_remote_code:
        tok_kwargs["trust_remote_code"] = True
    tokenizer = AutoTokenizer.from_pretrained(llm_config.name, **tok_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    # Load Arrow dataset from disk
    split_dir = Path(data_config.local_dir) / split
    dataset = load_corpus(local_dir=str(split_dir))

    text_field = data_config.text_field
    max_length = data_config.max_seq_length or 2048

    # Validate messages column exists for proper_chat
    if data_config.scaffolding == "proper_chat":
        if data_config.messages_field not in dataset.column_names:
            raise ValueError(
                f"scaffolding='proper_chat' requires a '{data_config.messages_field}' "
                f"column in the dataset at {split_dir}, but found columns: "
                f"{dataset.column_names}. Either set messages_field to the correct "
                f"column name, or use scaffolding='chatify' instead."
            )

    # Build prompts with scaffolding (per-document)
    prompts = []
    for row in dataset:
        if data_config.scaffolding == "proper_chat":
            messages = row[data_config.messages_field]
            if not messages or not isinstance(messages, list):
                raise ValueError(
                    f"scaffolding='proper_chat' but row has invalid "
                    f"'{data_config.messages_field}': {type(messages)}. "
                    f"Expected a non-empty list of {{role, content}} dicts."
                )
            for msg in messages:
                if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                    raise ValueError(
                        f"scaffolding='proper_chat' but message is malformed: "
                        f"{msg}. Each message must have 'role' and 'content' keys."
                    )
            formatted = maybe_apply_chat_template(messages, tokenizer, data_config)
        else:
            text = row[text_field]
            if not isinstance(text, str) or not text.strip():
                continue
            formatted = maybe_apply_chat_template(text, tokenizer, data_config)
        prompts.append(formatted)

    ds = PaddedPromptDataset(prompts, tokenizer, max_length=max_length)

    if shuffle is None:
        shuffle = split == "train"
    if drop_last is None:
        drop_last = split == "train"

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda b: collate_padded(b, pad_id),
        num_workers=data_config.num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )


# ---------------------------------------------------------------------------
# Cached hidden states (pre-extracted, no LLM needed)
# ---------------------------------------------------------------------------

def _validate_cache_metadata(metadata: dict, llm_config: LLMConfig) -> None:
    """Validate that cached hidden states match the LLM config."""
    if metadata["model_name"] != llm_config.name:
        raise ValueError(
            f"Cache model mismatch: cached='{metadata['model_name']}', "
            f"config='{llm_config.name}'"
        )
    if metadata["layer_index"] != llm_config.layer_index:
        raise ValueError(
            f"Cache layer mismatch: cached={metadata['layer_index']}, "
            f"config={llm_config.layer_index}"
        )
    if metadata["hidden_dim"] != llm_config.hidden_dim:
        raise ValueError(
            f"Cache hidden_dim mismatch: cached={metadata['hidden_dim']}, "
            f"config={llm_config.hidden_dim}"
        )


class CachedHiddenStateDataset(Dataset):
    """Dataset that reads pre-extracted hidden states from memmap files.

    Each item is a variable-length document with hidden states and input_ids.
    Collation (via ``collate_cached``) handles right-padding and mask creation.

    Expects the cache directory layout produced by ``scripts/cache_hidden_states.py``.
    """

    def __init__(self, split_dir: str | Path, llm_config: LLMConfig | None = None,
                 max_seq_length: int | None = None):
        split_dir = Path(split_dir)
        self.max_seq_length = max_seq_length

        with open(split_dir / "metadata.json") as f:
            self.metadata = json.load(f)

        self.offsets = np.load(split_dir / "offsets.npy")   # (num_docs,) int64
        self.lengths = np.load(split_dir / "lengths.npy")   # (num_docs,) int32
        self.hidden_dim = self.metadata["hidden_dim"]

        if llm_config is not None:
            _validate_cache_metadata(self.metadata, llm_config)

        self.h_mmap = np.memmap(
            split_dir / "hidden_states.bin",
            dtype=self.metadata["dtype"], mode="r",
            shape=(self.metadata["total_tokens"], self.hidden_dim),
        )
        self.ids_mmap = np.memmap(
            split_dir / "input_ids.bin", dtype=np.int32, mode="r",
            shape=(self.metadata["total_tokens"],),
        )

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, idx: int) -> dict:
        off = int(self.offsets[idx])
        length = int(self.lengths[idx])

        if self.max_seq_length is not None:
            length = min(length, self.max_seq_length)

        h = self.h_mmap[off: off + length]     # (L, D) float16/32
        ids = self.ids_mmap[off: off + length]  # (L,) int32

        return {
            "hidden_states": torch.from_numpy(h.astype(np.float32).copy()),
            "input_ids": torch.from_numpy(ids.astype(np.int64).copy()),
            "length": length,
        }


def collate_cached(batch: list[dict], pad_id: int = 0) -> dict[str, torch.Tensor]:
    """Collate variable-length cached hidden state samples with right-padding."""
    max_len = max(item["length"] for item in batch)
    B = len(batch)
    hidden_dim = batch[0]["hidden_states"].shape[-1]

    padded_h = torch.zeros(B, max_len, hidden_dim)
    padded_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    masks = torch.zeros(B, max_len, dtype=torch.long)

    # Optional fields (rewards, reward_mask)
    has_rewards = "rewards" in batch[0]
    if has_rewards:
        padded_rewards = torch.zeros(B, max_len)
        padded_reward_mask = torch.zeros(B, max_len)

    for i, item in enumerate(batch):
        L = item["length"]
        padded_h[i, :L] = item["hidden_states"]
        padded_ids[i, :L] = item["input_ids"]
        masks[i, :L] = 1
        if has_rewards:
            padded_rewards[i, :L] = item["rewards"]
            padded_reward_mask[i, :L] = item["reward_mask"]

    result = {
        "hidden_states": padded_h,
        "input_ids": padded_ids,
        "attention_mask": masks,
    }
    if has_rewards:
        result["rewards"] = padded_rewards
        result["reward_mask"] = padded_reward_mask
    return result


def build_cached_dataloader(
    data_config: DataConfig,
    llm_config: LLMConfig,
    split: str = "train",
    batch_size: int = 32,
    shuffle: bool | None = None,
    seed: int = 42,
    drop_last: bool | None = None,
) -> DataLoader:
    """Build a DataLoader for cached hidden states (no LLM needed).

    Supports two layouts:
    1. **Split layout** (default): ``cached_hidden_states_dir/{split}/`` with
       per-split memmap files (offsets.npy, lengths.npy, etc.).
    2. **Unified layout**: ``cached_hidden_states_dir/`` contains
       ``conv_index.jsonl`` + ``*.bin``. Requires ``split_indices_path``
       in data_config. Uses ``SycophancyMemmapDataset``.
    """
    cache_dir = Path(data_config.cached_hidden_states_dir)

    # Detect unified format (conv_index.jsonl present at cache root)
    unified = (cache_dir / "conv_index.jsonl").exists()

    if unified:
        if data_config.split_indices_path is None:
            raise ValueError(
                "Unified memmap format detected (conv_index.jsonl exists) but "
                "data.split_indices_path is not set. Provide the path to "
                "split_indices.json."
            )
        return build_sycophancy_cached_dataloader(
            cache_dir=cache_dir,
            split_indices_path=data_config.split_indices_path,
            split=split,
            batch_size=batch_size,
            shuffle=shuffle,
            max_seq_length=data_config.max_seq_length,
            num_workers=data_config.num_workers,
            seed=seed,
            judge_results_path=data_config.judge_results_path,
            sycophancy_mode=data_config.sycophancy_mode,
            drop_last=drop_last,
            truncate_at_first_cave=getattr(data_config, "truncate_at_first_cave", False),
        )

    # Fall through to split layout
    split_dir = cache_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Cached hidden states split directory not found: {split_dir}"
        )

    dataset = CachedHiddenStateDataset(
        split_dir, llm_config=llm_config,
        max_seq_length=data_config.max_seq_length,
    )

    # Get pad_token_id from tokenizer (fast — loads only tokenizer, not model)
    from transformers import AutoTokenizer
    tok_kwargs: dict = {}
    if llm_config.trust_remote_code:
        tok_kwargs["trust_remote_code"] = True
    tokenizer = AutoTokenizer.from_pretrained(llm_config.name, **tok_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    logger.info(
        "Cached dataset (%s): %d docs, %d tokens, hidden_dim=%d, pad_id=%d",
        split, len(dataset), dataset.metadata["total_tokens"], dataset.hidden_dim,
        pad_id,
    )

    if shuffle is None:
        shuffle = split == "train"
    if drop_last is None:
        drop_last = split == "train"

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_cached(batch, pad_id),
        num_workers=data_config.num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )


# ---------------------------------------------------------------------------
# Sycophancy Memmap Dataset (adapter for unified memmap format)
# ---------------------------------------------------------------------------

class SycophancyMemmapDataset(Dataset):
    """Dataset adapter for the sycophancy unified memmap cache.

    Reads from the unified memmap format (conv_index.jsonl + *.bin files)
    and presents the same ``__getitem__`` interface as
    ``CachedHiddenStateDataset``, so it works with ``collate_cached``.

    Splits are applied at read time: the dataset receives a list of
    conversation indices (from ``split_indices.json``) and only serves
    those conversations.

    If ``judge_results`` is provided, each item also includes:
      - ``rewards``:     [L] float32 — per-token reward (sycophancy label for that turn)
      - ``reward_mask``: [L] float32 — 1 for assistant tokens in turns 2-5, 0 elsewhere
    """

    def __init__(
        self,
        cache_dir: str | Path,
        conv_indices: list[int] | None = None,
        max_seq_length: int | None = None,
        judge_results: dict[str, list[int]] | None = None,
        sycophancy_mode: str | None = None,
        truncate_at_first_cave: bool = False,
    ):
        from evaluation.datasets.sycophancy.storage import MemmapReader

        if judge_results is not None and sycophancy_mode is None:
            raise ValueError(
                "sycophancy_mode is required when judge_results is provided. "
                "Set it explicitly in your YAML config (e.g. sycophancy_mode: false_presupposition)."
            )

        # FP default: truncate each conversation at the end of the first sycophantic
        # turn. Post-cave turns are noise (model won't un-cave) and for 35B are
        # marked with judge sentinel -1 which must not leak into training.
        if sycophancy_mode == "false_presupposition" and not truncate_at_first_cave:
            truncate_at_first_cave = True

        cache_dir = Path(cache_dir)
        self.reader = MemmapReader(str(cache_dir))
        self.max_seq_length = max_seq_length
        self.metadata = self.reader.metadata
        self.hidden_dim = self.reader.hidden_dim
        self.judge_results = judge_results
        self.sycophancy_mode = sycophancy_mode

        # If conv_indices given, serve only those conversations
        if conv_indices is not None:
            self._indices = list(conv_indices)
        else:
            self._indices = list(range(len(self.reader)))

        # For debate with judge results: drop conversations where the model's
        # turn-1 reply did not defend the assigned argument (stance-overlap filter).
        if (judge_results is not None and sycophancy_mode == "debate"):
            from evaluation.datasets.sycophancy.labels import compute_sycophancy_labels
            kept = []
            n_dropped = 0
            for ci in self._indices:
                cid = self.reader.get_index_record(ci)["id"]
                if cid not in judge_results:
                    kept.append(ci)
                    continue
                if compute_sycophancy_labels(judge_results[cid], mode="debate") is None:
                    n_dropped += 1
                else:
                    kept.append(ci)
            if n_dropped > 0:
                logger.info(
                    "SycophancyMemmapDataset: stance-overlap filter dropped %d/%d "
                    "conversations (turn-1 reply did not defend the assigned argument)",
                    n_dropped, len(self._indices),
                )
            self._indices = kept

        # Pre-compute per-conversation truncation lengths for first-cave
        # truncation.  Each conversation is clipped at the end of the first
        # turn where judgment[t]==0.  Conversations that never cave are
        # left at their full length (not stored in the dict).
        self._truncation_lengths: dict[int, int] = {}
        if truncate_at_first_cave and judge_results is not None:
            n_truncated = 0
            for ci in self._indices:
                rec = self.reader.get_index_record(ci)
                cid = rec["id"]
                if cid not in judge_results:
                    continue
                tj = judge_results[cid]
                first_cave = next((i for i, j in enumerate(tj) if j == 0), None)
                if first_cave is None:
                    continue  # never caves — keep full length
                off = rec["offset"]
                length = rec["length"]
                turn_ids = np.array(self.reader._turn[off:off + length])
                cave_turn_id = first_cave + 1  # 0-indexed judgment → 1-indexed turn_id
                trunc_len = int((turn_ids <= cave_turn_id).sum())
                self._truncation_lengths[ci] = trunc_len
                n_truncated += 1
            if n_truncated > 0:
                logger.info(
                    "SycophancyMemmapDataset: truncating %d/%d conversations "
                    "at first cave turn",
                    n_truncated, len(self._indices),
                )

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict:
        real_idx = self._indices[idx]
        rec = self.reader.get_index_record(real_idx)
        off = rec["offset"]
        length = rec["length"]

        # Apply first-cave truncation (before max_seq_length)
        if real_idx in self._truncation_lengths:
            length = self._truncation_lengths[real_idx]

        if self.max_seq_length is not None:
            length = min(length, self.max_seq_length)

        h = self.reader.get_hidden_states_slice(off, length)
        ids = self.reader.get_token_ids_slice(off, length)

        item = {
            "hidden_states": torch.from_numpy(np.array(h, dtype=np.float32)),
            "input_ids": torch.from_numpy(np.array(ids, dtype=np.int64)),
            "length": length,
        }

        # Add per-token rewards and reward mask if judge results available
        if self.judge_results is not None:
            conv_id = rec["id"]
            role_ids = np.array(self.reader._role[off:off + length])
            turn_ids = np.array(self.reader._turn[off:off + length])

            rewards = np.zeros(length, dtype=np.float32)
            reward_mask = np.zeros(length, dtype=np.float32)

            if conv_id in self.judge_results:
                from evaluation.datasets.sycophancy.labels import compute_sycophancy_labels
                judgments = self.judge_results[conv_id]
                labels = compute_sycophancy_labels(judgments, mode=self.sycophancy_mode)

                if labels is not None:
                    start_turn = 1 if self.sycophancy_mode == "false_presupposition" else 2
                    for k in range(start_turn, len(labels) + 1):
                        mask = (role_ids == 2) & (turn_ids == k)
                        reward_mask[mask] = 1.0
                        rewards[mask] = float(labels[k - 1])

            item["rewards"] = torch.from_numpy(rewards)
            item["reward_mask"] = torch.from_numpy(reward_mask)

        return item


def build_sycophancy_cached_dataloader(
    cache_dir: str | Path,
    split_indices_path: str | Path,
    split: str = "train",
    batch_size: int = 32,
    shuffle: bool | None = None,
    max_seq_length: int | None = None,
    num_workers: int = 4,
    seed: int = 42,
    judge_results_path: str | Path | None = None,
    sycophancy_mode: str | None = None,
    drop_last: bool | None = None,
    truncate_at_first_cave: bool = False,
) -> DataLoader:
    """Build a DataLoader for the sycophancy unified memmap format.

    Parameters
    ----------
    cache_dir            : path to layer_XX_activations_cache/
    split_indices_path   : path to split_indices.json (conversation-level splits)
    split                : "train" or "val" (val+test combined for RET)
    batch_size           : batch size
    max_seq_length       : optional max tokens per document
    judge_results_path   : path to judge_results.jsonl; when provided, each batch
                           includes per-token ``rewards`` and ``reward_mask``.
    """
    split_path = Path(split_indices_path)
    if not split_path.exists():
        raise FileNotFoundError(f"Split indices not found: {split_path}")

    with open(split_path) as f:
        split_info = json.load(f)

    if split == "train":
        conv_indices = split_info["train"]
    elif split == "val":
        conv_indices = split_info["val"] + split_info.get("test", [])
    else:
        conv_indices = split_info.get(split, [])

    # Load judge results if path provided
    judge_results = None
    if judge_results_path is not None:
        judge_results_path = Path(judge_results_path)
        if judge_results_path.exists():
            from evaluation.datasets.sycophancy.labels import load_judge_results
            judge_results = load_judge_results(str(judge_results_path))
            logger.info("Loaded %d judge results for reward supervision", len(judge_results))

    dataset = SycophancyMemmapDataset(
        cache_dir, conv_indices=conv_indices,
        max_seq_length=max_seq_length,
        judge_results=judge_results,
        sycophancy_mode=sycophancy_mode,
        truncate_at_first_cave=truncate_at_first_cave,
    )

    if shuffle is None:
        shuffle = split == "train"
    if drop_last is None:
        drop_last = split == "train"

    logger.info(
        "SycophancyMemmapDataset (%s): %d conversations, hidden_dim=%d",
        split, len(dataset), dataset.hidden_dim,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_cached(batch, pad_id=0),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )
