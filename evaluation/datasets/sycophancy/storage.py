"""
Save/load conversations, hidden states, and judge results for SYCON-Bench.

All cached artifacts live under data/sycophancy/{scenario}/{model_slug}/:

    data/sycophancy/debate/Qwen2.5-14B-Instruct/
        conversations.jsonl                     # generated conversations (text)
        layer_23_activations_cache/             # hidden states for layer 23
            metadata.json                       # schema + totals
            conv_index.jsonl                    # per-conversation offsets
            hidden_states.bin                   # [total_tokens, D] float32
            token_ids.bin                       # [total_tokens] int32
            role_ids.bin                        # [total_tokens] int8
            turn_ids.bin                        # [total_tokens] int8
        judge_results.jsonl                     # judge binary labels
        metrics.json                            # ToF/NoF
        sample_prompts.txt                      # 5 formatted prompt examples
"""
from __future__ import annotations

import gzip
import json
import logging
import os
import tempfile
from typing import Dict, List, Optional, Set

import numpy as np
import torch

from evaluation.datasets.sycophancy.generation import ConversationResult

logger = logging.getLogger(__name__)

MEMMAP_FORMAT_VERSION = 2


# ---------------------------------------------------------------------------
# MemmapWriter — incremental append-only writer
# ---------------------------------------------------------------------------

class MemmapWriter:
    """Append-only writer for the unified memmap hidden-state cache.

    Creates / resumes a cache directory with the schema::

        cache_dir/
          metadata.json       # {model_name, layer_index, hidden_dim, dtype, ...}
          conv_index.jsonl    # one line per conversation
          hidden_states.bin   # flat [total_tokens, hidden_dim] float32
          token_ids.bin       # flat [total_tokens] int32
          role_ids.bin        # flat [total_tokens] int8
          turn_ids.bin        # flat [total_tokens] int8

    On init, existing ``conv_index.jsonl`` is parsed for resumption.
    If binary files are larger than expected (partial crash), they are
    truncated to the last consistent state.
    """

    def __init__(
        self,
        cache_dir: str,
        model_name: str,
        layer_index: int,
        hidden_dim: int,
        chunk_id: Optional[str] = None,
    ):
        # When chunk_id is set, write to {cache_dir}/chunk_{chunk_id}/ so
        # parallel jobs can each write to their own subdirectory without
        # racing on the shared binary files. Use merge_chunks(...) to
        # consolidate into the canonical layout afterward.
        if chunk_id is not None:
            cache_dir = os.path.join(cache_dir, f"chunk_{chunk_id}")
        self.cache_dir = cache_dir
        self.model_name = model_name
        self.layer_index = layer_index
        self.hidden_dim = hidden_dim

        os.makedirs(cache_dir, exist_ok=True)

        self._hs_path = os.path.join(cache_dir, "hidden_states.bin")
        self._tok_path = os.path.join(cache_dir, "token_ids.bin")
        self._role_path = os.path.join(cache_dir, "role_ids.bin")
        self._turn_path = os.path.join(cache_dir, "turn_ids.bin")
        self._index_path = os.path.join(cache_dir, "conv_index.jsonl")
        self._meta_path = os.path.join(cache_dir, "metadata.json")

        # Parse existing index for resumption
        self._completed: Dict[str, dict] = {}  # id -> {offset, length, num_turns}
        self._total_tokens = 0
        self._num_conversations = 0

        if os.path.exists(self._index_path):
            with open(self._index_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    self._completed[rec["id"]] = rec
                    self._total_tokens = rec["offset"] + rec["length"]
                    self._num_conversations += 1
            logger.info("MemmapWriter: resuming %s — %d conversations, %d tokens",
                        cache_dir, self._num_conversations, self._total_tokens)

        # Crash recovery: truncate binary files to consistent state
        self._recover_if_needed()

    def _recover_if_needed(self):
        """Truncate binary files if they are larger than expected."""
        expected = {
            self._hs_path: self._total_tokens * self.hidden_dim * 4,  # float32
            self._tok_path: self._total_tokens * 4,                    # int32
            self._role_path: self._total_tokens * 1,                   # int8
            self._turn_path: self._total_tokens * 1,                   # int8
        }
        for path, expected_size in expected.items():
            if os.path.exists(path):
                actual = os.path.getsize(path)
                if actual > expected_size:
                    logger.warning("Truncating %s: %d -> %d bytes (crash recovery)",
                                   path, actual, expected_size)
                    with open(path, "r+b") as f:
                        f.truncate(expected_size)
                elif actual < expected_size:
                    raise RuntimeError(
                        f"Binary file {path} is smaller than expected "
                        f"({actual} < {expected_size}). Data may be corrupted."
                    )

    @property
    def completed_ids(self) -> Set[str]:
        return set(self._completed.keys())

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    @property
    def num_conversations(self) -> int:
        return self._num_conversations

    def append_conversation(
        self,
        conv_id: str,
        hidden_states: np.ndarray,
        token_ids: np.ndarray,
        role_ids: np.ndarray,
        turn_ids: np.ndarray,
        num_turns: int,
    ) -> None:
        """Append one conversation to the cache.

        Parameters
        ----------
        conv_id       : conversation string ID (e.g. "debate_000")
        hidden_states : [S, D] float32
        token_ids     : [S] int32
        role_ids      : [S] int8
        turn_ids      : [S] int8
        num_turns     : number of dialogue turns
        """
        if conv_id in self._completed:
            return  # already written

        S = hidden_states.shape[0]
        assert hidden_states.shape == (S, self.hidden_dim), \
            f"Expected shape ({S}, {self.hidden_dim}), got {hidden_states.shape}"

        # Append to binary files
        with open(self._hs_path, "ab") as f:
            f.write(hidden_states.astype(np.float32).tobytes())
        with open(self._tok_path, "ab") as f:
            f.write(token_ids.astype(np.int32).tobytes())
        with open(self._role_path, "ab") as f:
            f.write(role_ids.astype(np.int8).tobytes())
        with open(self._turn_path, "ab") as f:
            f.write(turn_ids.astype(np.int8).tobytes())

        # Append index record
        record = {
            "id": conv_id,
            "offset": self._total_tokens,
            "length": S,
            "num_turns": num_turns,
        }
        with open(self._index_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        self._completed[conv_id] = record
        self._total_tokens += S
        self._num_conversations += 1

    def flush_metadata(self) -> None:
        """Write metadata.json atomically (write-to-temp then rename)."""
        meta = {
            "model_name": self.model_name,
            "layer_index": self.layer_index,
            "hidden_dim": self.hidden_dim,
            "dtype": "float32",
            "num_conversations": self._num_conversations,
            "total_tokens": self._total_tokens,
            "version": MEMMAP_FORMAT_VERSION,
        }
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=self.cache_dir, suffix=".tmp", prefix="metadata_"
        )
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(meta, f, indent=2)
            os.replace(tmp_path, self._meta_path)
        except BaseException:
            os.unlink(tmp_path)
            raise


# ---------------------------------------------------------------------------
# MemmapReader — read-only access
# ---------------------------------------------------------------------------

class MemmapReader:
    """Read-only access to the unified memmap hidden-state cache.

    Opens all memmap files once and provides conversation-level access
    by ID or by sequential index.
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir

        with open(os.path.join(cache_dir, "metadata.json")) as f:
            self.metadata = json.load(f)

        self.hidden_dim = self.metadata["hidden_dim"]
        total_tokens = self.metadata["total_tokens"]

        # Parse conversation index
        self._index: List[dict] = []
        self._id_to_idx: Dict[str, int] = {}
        index_path = os.path.join(cache_dir, "conv_index.jsonl")
        with open(index_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                self._id_to_idx[rec["id"]] = len(self._index)
                self._index.append(rec)

        # Open memmap files read-only
        self._hs = np.memmap(
            os.path.join(cache_dir, "hidden_states.bin"),
            dtype=np.float32, mode="r",
            shape=(total_tokens, self.hidden_dim),
        )
        self._tok = np.memmap(
            os.path.join(cache_dir, "token_ids.bin"),
            dtype=np.int32, mode="r",
            shape=(total_tokens,),
        )
        self._role = np.memmap(
            os.path.join(cache_dir, "role_ids.bin"),
            dtype=np.int8, mode="r",
            shape=(total_tokens,),
        )
        self._turn = np.memmap(
            os.path.join(cache_dir, "turn_ids.bin"),
            dtype=np.int8, mode="r",
            shape=(total_tokens,),
        )

    def __len__(self) -> int:
        return len(self._index)

    @property
    def conversation_ids(self) -> List[str]:
        return [rec["id"] for rec in self._index]

    def has_conversation(self, conv_id: str) -> bool:
        return conv_id in self._id_to_idx

    def get_conversation(self, conv_id: str) -> dict:
        """Load one conversation by ID.

        Returns dict with keys: hidden_states [S,D], token_ids [S],
        role_ids [S], turn_ids [S], num_turns.
        """
        idx = self._id_to_idx[conv_id]
        return self.get_by_index(idx)

    def get_by_index(self, idx: int) -> dict:
        """Load one conversation by sequential index."""
        rec = self._index[idx]
        off = rec["offset"]
        length = rec["length"]
        return {
            "hidden_states": np.array(self._hs[off:off + length]),   # copy from memmap
            "token_ids": np.array(self._tok[off:off + length]),
            "role_ids": np.array(self._role[off:off + length]),
            "turn_ids": np.array(self._turn[off:off + length]),
            "num_turns": rec["num_turns"],
            "id": rec["id"],
        }

    def get_hidden_states_slice(self, offset: int, length: int) -> np.ndarray:
        """Direct memmap slice — returns a view (zero-copy) for RET training."""
        return self._hs[offset:offset + length]

    def get_token_ids_slice(self, offset: int, length: int) -> np.ndarray:
        """Direct memmap slice for token IDs."""
        return self._tok[offset:offset + length]

    def get_index_record(self, idx: int) -> dict:
        """Return the index record for conversation at position idx."""
        return self._index[idx]


# ---------------------------------------------------------------------------
# High-level save/load functions
# ---------------------------------------------------------------------------

def _model_slug(model_id: str) -> str:
    """Convert 'Qwen/Qwen2.5-14B-Instruct' -> 'Qwen2.5-14B-Instruct'."""
    return model_id.split("/")[-1]


def get_output_dir(data_dir: str, model_id: str, mode: str = "") -> str:
    """Build output directory: {data_dir}/{model_slug}[_{mode}]/"""
    slug = _model_slug(model_id)
    if mode:
        slug = f"{slug}_{mode}"
    return os.path.join(data_dir, slug)


def save_conversation(
    result: ConversationResult,
    output_dir: str,
    writers: Optional[Dict[int, MemmapWriter]] = None,
) -> None:
    """Append conversation text to JSONL and save hidden states via MemmapWriter.

    Parameters
    ----------
    result     : ConversationResult from the generation pipeline
    output_dir : directory containing conversations.jsonl
    writers    : dict mapping layer_index -> MemmapWriter instance.
                 Must be provided when hidden states are available.
    """
    os.makedirs(output_dir, exist_ok=True)

    # -- Text data (JSONL) --
    record = {
        "id": result.seed.id,
        "scenario": result.seed.scenario,
        "model_id": result.model_id,
        "metadata": result.seed.metadata,
        "turns": [
            {
                "turn": t.turn,
                "user_message": t.user_message,
                "assistant_response": t.assistant_response,
                "prompt_token_count": t.prompt_token_count,
                "response_token_count": t.response_token_count,
            }
            for t in result.turns
        ],
    }
    jsonl_path = os.path.join(output_dir, "conversations.jsonl")
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # -- Hidden states via MemmapWriter --
    final_turn = result.turns[-1]
    if writers is None or not final_turn.hidden_states:
        return

    for layer_idx, hs in final_turn.hidden_states.items():
        if layer_idx not in writers:
            continue
        writer = writers[layer_idx]

        boundaries = final_turn.boundaries or {}
        token_ids = boundaries.get("token_ids", np.zeros(hs.shape[0], dtype=np.int32))
        role_ids = boundaries.get("role_ids", np.zeros(hs.shape[0], dtype=np.int8))
        turn_ids = boundaries.get("turn_ids", np.zeros(hs.shape[0], dtype=np.int8))

        writer.append_conversation(
            conv_id=result.seed.id,
            hidden_states=hs,
            token_ids=token_ids,
            role_ids=role_ids,
            turn_ids=turn_ids,
            num_turns=len(result.turns),
        )


def save_sample_prompts(prompts: List[str], output_dir: str, n: int = 5) -> None:
    """Save the first n formatted prompt strings so we can verify chat templates."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "sample_prompts.txt")
    with open(path, "w", encoding="utf-8") as f:
        for i, prompt in enumerate(prompts[:n]):
            f.write(f"{'='*80}\n")
            f.write(f"SAMPLE {i+1}\n")
            f.write(f"{'='*80}\n")
            f.write(prompt)
            f.write(f"\n\n")
    logger.info("Saved %d sample prompts to %s", min(n, len(prompts)), path)


def open_jsonl(path: str):
    """Open a JSONL path for reading as text, transparently handling gzip.

    Accepts either an explicit ``*.gz`` path or a plain path; if the plain
    path is absent but ``<path>.gz`` exists, the gzipped file is read instead.
    This lets large conversation dumps ship gzipped while existing call sites
    and CLI examples keep passing ``conversations.jsonl``.
    """
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    if not os.path.exists(path) and os.path.exists(path + ".gz"):
        return gzip.open(path + ".gz", "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def load_conversations(jsonl_path: str) -> List[dict]:
    """Read all conversations from a JSONL file (gzip-transparent)."""
    records = []
    with open_jsonl(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def merge_chunks(cache_dir: str, delete_chunks: bool = False) -> dict:
    """Merge per-chunk subdirectories into the canonical cache layout.

    When ``MemmapWriter`` is used with ``chunk_id`` (one writer per parallel
    job), each job writes to ``{cache_dir}/chunk_{id}/``. This function
    stream-concatenates those subdirs into the canonical files directly under
    ``{cache_dir}/`` (``hidden_states.bin``, ``token_ids.bin``, ``role_ids.bin``,
    ``turn_ids.bin``, ``conv_index.jsonl``, ``metadata.json``), recomputing
    offsets so the merged ``conv_index.jsonl`` is consistent with the merged
    binary files.

    Duplicate conversation IDs across chunks (e.g. from re-runs that retried
    the same range) are skipped: the first occurrence wins, with a warning.

    Parameters
    ----------
    cache_dir : str
        The canonical cache directory containing one or more ``chunk_*/``
        subdirectories.
    delete_chunks : bool
        If True, remove each ``chunk_*/`` subdirectory after a successful
        merge. Default False (keeps the chunks until the user verifies).

    Returns
    -------
    dict
        ``{"num_conversations": int, "total_tokens": int, "chunks_merged": int}``
    """
    import glob
    import shutil

    chunk_dirs = sorted(glob.glob(os.path.join(cache_dir, "chunk_*")))
    if not chunk_dirs:
        raise FileNotFoundError(
            f"No chunk_* subdirectories found under {cache_dir}"
        )

    logger.info("merge_chunks: found %d chunk subdirs in %s",
                len(chunk_dirs), cache_dir)

    # Validate that all chunks share the same schema
    chunk_metas = []
    for cd in chunk_dirs:
        meta_path = os.path.join(cd, "metadata.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Chunk missing metadata.json: {cd}")
        with open(meta_path) as f:
            chunk_metas.append(json.load(f))

    ref = chunk_metas[0]
    for cm, cd in zip(chunk_metas[1:], chunk_dirs[1:]):
        for k in ("model_name", "layer_index", "hidden_dim", "dtype", "version"):
            if cm.get(k) != ref.get(k):
                raise ValueError(
                    f"Chunk schema mismatch for key {k!r}: "
                    f"{cd}={cm.get(k)!r} vs {chunk_dirs[0]}={ref.get(k)!r}"
                )

    hidden_dim = ref["hidden_dim"]
    if ref["dtype"] != "float32":
        raise NotImplementedError(
            f"merge_chunks currently assumes float32 hidden states, got {ref['dtype']}"
        )

    # Output paths (canonical layout, directly under cache_dir)
    out_hs = os.path.join(cache_dir, "hidden_states.bin")
    out_tok = os.path.join(cache_dir, "token_ids.bin")
    out_role = os.path.join(cache_dir, "role_ids.bin")
    out_turn = os.path.join(cache_dir, "turn_ids.bin")
    out_index = os.path.join(cache_dir, "conv_index.jsonl")
    out_meta = os.path.join(cache_dir, "metadata.json")

    # Refuse to overwrite an existing canonical layout in this dir.
    for p in (out_hs, out_tok, out_role, out_turn, out_index):
        if os.path.exists(p):
            raise FileExistsError(
                f"Refusing to overwrite existing canonical file {p}. "
                f"Move it aside (e.g. .corrupted) before merging."
            )

    READ_CHUNK = 256 * 1024 * 1024  # 256 MB

    seen_ids: Set[str] = set()
    duplicate_ids: List[str] = []
    global_offset_tokens = 0
    n_convs = 0

    with open(out_hs, "wb") as f_hs, \
         open(out_tok, "wb") as f_tok, \
         open(out_role, "wb") as f_role, \
         open(out_turn, "wb") as f_turn, \
         open(out_index, "w") as f_idx:

        for cd, cm in zip(chunk_dirs, chunk_metas):
            # Load this chunk's index
            chunk_records = []
            with open(os.path.join(cd, "conv_index.jsonl")) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        chunk_records.append(json.loads(line))

            chunk_total_tokens = cm["total_tokens"]
            # Sanity-check binary file sizes
            for path, item_bytes in [
                (os.path.join(cd, "hidden_states.bin"), hidden_dim * 4),
                (os.path.join(cd, "token_ids.bin"), 4),
                (os.path.join(cd, "role_ids.bin"), 1),
                (os.path.join(cd, "turn_ids.bin"), 1),
            ]:
                expected = chunk_total_tokens * item_bytes
                actual = os.path.getsize(path)
                if actual != expected:
                    raise RuntimeError(
                        f"Chunk binary size mismatch in {cd}: "
                        f"{path} is {actual} bytes, expected {expected}"
                    )

            # Build a sorted-by-offset view so we can copy ranges in order.
            # Skip duplicates (first chunk wins).
            keep = []
            for rec in chunk_records:
                if rec["id"] in seen_ids:
                    duplicate_ids.append(rec["id"])
                    continue
                seen_ids.add(rec["id"])
                keep.append(rec)
            keep.sort(key=lambda r: r["offset"])

            # Stream the binary files for this chunk and emit only the byte
            # ranges belonging to kept records. For chunks with NO duplicates
            # (the common case), this is just one big copy of the whole file.
            if len(keep) == len(chunk_records):
                # Fast path: stream-copy the entire chunk's binaries.
                _stream_copy(os.path.join(cd, "hidden_states.bin"), f_hs, READ_CHUNK)
                _stream_copy(os.path.join(cd, "token_ids.bin"), f_tok, READ_CHUNK)
                _stream_copy(os.path.join(cd, "role_ids.bin"), f_role, READ_CHUNK)
                _stream_copy(os.path.join(cd, "turn_ids.bin"), f_turn, READ_CHUNK)
                # Emit index records with shifted offsets, in original order.
                for rec in chunk_records:
                    new_rec = dict(rec)
                    new_rec["offset"] = rec["offset"] + global_offset_tokens
                    f_idx.write(json.dumps(new_rec) + "\n")
                    n_convs += 1
                global_offset_tokens += chunk_total_tokens
            else:
                # Slow path: copy only kept ranges, recompute offsets densely.
                with open(os.path.join(cd, "hidden_states.bin"), "rb") as src_hs, \
                     open(os.path.join(cd, "token_ids.bin"), "rb") as src_tok, \
                     open(os.path.join(cd, "role_ids.bin"), "rb") as src_role, \
                     open(os.path.join(cd, "turn_ids.bin"), "rb") as src_turn:
                    for rec in keep:
                        L = rec["length"]
                        src_hs.seek(rec["offset"] * hidden_dim * 4)
                        src_tok.seek(rec["offset"] * 4)
                        src_role.seek(rec["offset"] * 1)
                        src_turn.seek(rec["offset"] * 1)
                        f_hs.write(src_hs.read(L * hidden_dim * 4))
                        f_tok.write(src_tok.read(L * 4))
                        f_role.write(src_role.read(L * 1))
                        f_turn.write(src_turn.read(L * 1))
                        new_rec = dict(rec)
                        new_rec["offset"] = global_offset_tokens
                        f_idx.write(json.dumps(new_rec) + "\n")
                        n_convs += 1
                        global_offset_tokens += L

    if duplicate_ids:
        logger.warning(
            "merge_chunks: skipped %d duplicate conversation IDs (first occurrence kept). "
            "First few: %s",
            len(duplicate_ids), duplicate_ids[:5],
        )

    # Write merged metadata.json (atomic via tmp + rename)
    merged_meta = {
        "model_name": ref["model_name"],
        "layer_index": ref["layer_index"],
        "hidden_dim": hidden_dim,
        "dtype": "float32",
        "num_conversations": n_convs,
        "total_tokens": global_offset_tokens,
        "version": MEMMAP_FORMAT_VERSION,
    }
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=cache_dir, suffix=".tmp", prefix="metadata_"
    )
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(merged_meta, f, indent=2)
        os.replace(tmp_path, out_meta)
    except BaseException:
        os.unlink(tmp_path)
        raise

    logger.info(
        "merge_chunks: merged %d chunks -> %d conversations, %d tokens",
        len(chunk_dirs), n_convs, global_offset_tokens,
    )

    if delete_chunks:
        for cd in chunk_dirs:
            shutil.rmtree(cd)
        logger.info("merge_chunks: deleted %d chunk subdirs", len(chunk_dirs))

    return {
        "num_conversations": n_convs,
        "total_tokens": global_offset_tokens,
        "chunks_merged": len(chunk_dirs),
    }


def _stream_copy(src_path: str, dst_file, chunk_bytes: int) -> None:
    """Copy a file's contents to an open writable file in chunks."""
    with open(src_path, "rb") as src:
        while True:
            buf = src.read(chunk_bytes)
            if not buf:
                break
            dst_file.write(buf)


def get_completed_ids(
    output_dir: str,
    layers: Optional[List[int]] = None,
    chunk_id: Optional[str] = None,
) -> Set[str]:
    """Return IDs of conversations already saved (for resumption).

    Checks conv_index.jsonl in layer cache directories. If multiple layers
    are specified, returns the intersection (only IDs present in all layers).

    When ``layers`` is provided, the layer cache is the AUTHORITATIVE source
    of truth for "already done" — we never fall back to conversations.jsonl
    in that case, even if no layer cache files exist yet. The fallback to
    conversations.jsonl is only used when ``layers`` is None (text-only
    runs without hidden-state extraction). Reason: conversations.jsonl can
    contain records that don't have corresponding hidden states (e.g. when
    re-extracting activations from a previously generated text run, or when
    a chunked job is starting fresh against a pre-existing JSONL).

    When ``chunk_id`` is set, looks inside the per-chunk subdirectory
    ``layer_{L}_activations_cache/chunk_{chunk_id}/`` so resume only sees
    this chunk's progress (each chunk handles a non-overlapping range).
    Returns an empty set if the chunk subdirectory does not yet exist.
    """
    if layers:
        sets = []
        for layer in layers:
            layer_dir = os.path.join(output_dir, f"layer_{layer}_activations_cache")
            if chunk_id is not None:
                layer_dir = os.path.join(layer_dir, f"chunk_{chunk_id}")
            index_path = os.path.join(layer_dir, "conv_index.jsonl")
            if os.path.exists(index_path):
                ids = set()
                with open(index_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            ids.add(json.loads(line)["id"])
                sets.append(ids)
            else:
                # Authoritative: missing layer cache means nothing extracted
                # for this layer. Return empty (no fallback to JSONL).
                sets.append(set())
        return sets[0].intersection(*sets[1:]) if len(sets) > 1 else sets[0]

    # Fallback (only when layers is None — text-only runs)
    jsonl_path = os.path.join(output_dir, "conversations.jsonl")
    if not os.path.exists(jsonl_path) and not os.path.exists(jsonl_path + ".gz"):
        return set()
    ids = set()
    with open_jsonl(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["id"])
    return ids
