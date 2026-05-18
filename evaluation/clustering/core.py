"""Core module for streaming clustering analysis in the RET project.

Provides:
  - K-Means wrapper functions (create, save, load, predict_with_scores)
  - StreamingPreprocessor: two-phase (accumulate -> transform) preprocessing
  - apply_window_and_stride: causal moving average + stride on token features
  - extract_features: shared LLM forward pass returning h and/or z on CPU
  - Statistics functions: transition matrix, dwell times, cluster usage, entropy
  - Discovery utilities: discover_combos, find_config_yaml
  - I/O helpers: save_matrix_csv
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans


# ---------------------------------------------------------------------------
# 1. K-Means wrapper functions
# ---------------------------------------------------------------------------


def create_kmeans(K: int, seed: int, batch_size: int = 65536) -> MiniBatchKMeans:
    """Create a MiniBatchKMeans instance.

    Args:
        K: number of clusters
        seed: random state for reproducibility
        batch_size: mini-batch size used during partial_fit

    Returns:
        Unfitted MiniBatchKMeans instance.
    """
    return MiniBatchKMeans(n_clusters=K, random_state=seed, batch_size=batch_size, n_init=3)


def save_kmeans(model: MiniBatchKMeans, path: Path) -> None:
    """Save k-means model: pickle + centers.npy for inspection.

    Args:
        model: fitted MiniBatchKMeans instance
        path: directory to save into (created if missing)
    """
    import joblib

    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path / "kmeans_model.pkl")
    np.save(path / "centers.npy", model.cluster_centers_)


def load_kmeans(path: Path) -> MiniBatchKMeans:
    """Load k-means model from pickle.

    Args:
        path: directory containing kmeans_model.pkl

    Returns:
        Fitted MiniBatchKMeans instance.
    """
    import joblib

    return joblib.load(path / "kmeans_model.pkl")


def predict_with_scores(
    model: MiniBatchKMeans,
    data: np.ndarray,
    raw: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict cluster labels and compute similarity/distance scores.

    When ``raw=False`` (default), uses cosine similarity (data must be
    L2-normalized).  When ``raw=True``, uses Euclidean distance (matching
    KMeans training) and returns negative distance as the score so that
    higher values still mean "closer".

    Args:
        model: fitted MiniBatchKMeans
        data: (N, D) numpy array (L2-normalized when raw=False)
        raw: if True, use Euclidean distance instead of cosine similarity

    Returns:
        labels: (N,) int array of cluster assignments
        top1_score: (N,) cosine similarity (raw=False) or negative
                    Euclidean distance (raw=True) to assigned center
        margin: (N,) gap between top-1 and top-2 scores
    """
    centers = model.cluster_centers_

    if raw:
        # Euclidean distance based (consistent with KMeans training)
        dists = np.linalg.norm(
            data[:, None, :] - centers[None, :, :], axis=2,
        )  # (N, K)
        labels = dists.argmin(axis=1)
        top1_score = -dists[np.arange(len(labels)), labels]
        dists_copy = dists.copy()
        dists_copy[np.arange(len(labels)), labels] = np.inf
        top2_score = -dists_copy.min(axis=1)
        margin = top1_score - top2_score  # always >= 0
        return labels, top1_score, margin

    # Cosine similarity based (original behaviour)
    centers_norm = centers / np.linalg.norm(centers, axis=1, keepdims=True).clip(min=1e-8)
    sims = data @ centers_norm.T  # (N, K)
    labels = sims.argmax(axis=1)
    top1_cos = sims[np.arange(len(labels)), labels]
    sims_copy = sims.copy()
    sims_copy[np.arange(len(labels)), labels] = -np.inf
    top2_cos = sims_copy.max(axis=1)
    margin = top1_cos - top2_cos
    return labels, top1_cos, margin


# ---------------------------------------------------------------------------
# 2. StreamingPreprocessor
# ---------------------------------------------------------------------------


class StreamingPreprocessor:
    """Two-phase preprocessor: accumulate stats (pass 1), then transform (pass 2).

    Supported variants:
      - "h": center + L2-normalize raw LLM hidden states
      - "z": center + L2-normalize encoder output (same pipeline as "h")
      - Variant with ``_centered`` suffix: center only, no L2 normalization.
        Saves mu.npy and a ``centered_mode.flag`` sentinel.
      - Variant with ``_raw`` suffix: features returned as-is (no centering, no
        normalization). Saves a ``raw_mode.flag`` sentinel instead of mu.npy.

    Args:
        variant: feature variant name. Suffix ``_raw`` skips all preprocessing;
                 suffix ``_centered`` does mean-subtraction only.
        window_size: causal moving-average window size (passed through for reference)
        stride: stride applied to token sequences (passed through for reference)
    """

    def __init__(
        self,
        variant: str,
        window_size: int,
        stride: int,
    ):
        self.variant = variant
        self.window_size = window_size
        self.stride = stride
        self.raw = variant.endswith("_raw")
        self.centered_only = variant.endswith("_centered")
        self._sum: np.ndarray | None = None  # running sum for mean
        self._count: int = 0
        self._fitted: bool = False

    def accumulate(self, features: np.ndarray) -> None:
        """Phase 1: accumulate running mean.

        When raw=True this is a no-op; finalize_fit() will mark the preprocessor
        as fitted without computing a mean.

        Args:
            features: (N, D) numpy array of windowed/strided features
        """
        if self.raw:
            return
        if features.shape[0] == 0:
            return
        if self._sum is None:
            self._sum = np.zeros(features.shape[1], dtype=np.float64)
        self._sum += features.sum(axis=0).astype(np.float64)
        self._count += features.shape[0]

    def finalize_fit(self) -> None:
        """Transition from phase 1 to phase 2.

        In normal mode, computes and stores the global mean from accumulated
        statistics.  In raw mode, simply marks the preprocessor as fitted.

        Raises:
            RuntimeError: if no data was accumulated before calling this method
                          (normal mode only).
        """
        if self.raw:
            self.mu = None
            self._fitted = True
            return
        if self._count == 0:
            raise RuntimeError("No data accumulated")
        self.mu = (self._sum / self._count).astype(np.float32)
        self._fitted = True

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Phase 2: center -> L2-normalize (or pass through in raw mode).

        Args:
            features: (N, D) numpy array

        Returns:
            (N, D) numpy array — L2-normalized in normal mode, mean-subtracted only
            in centered mode, unchanged in raw mode.
        """
        assert self._fitted, "Call finalize_fit() before transform()"
        if self.raw:
            return features.astype(np.float32)
        centered = features - self.mu
        if self.centered_only:
            return centered.astype(np.float32)
        norms = np.linalg.norm(centered, axis=1, keepdims=True).clip(min=1e-8)
        return (centered / norms).astype(np.float32)

    def save(self, dir_path: Path) -> None:
        """Persist preprocessor state to disk.

        In raw mode a sentinel file ``raw_mode.flag`` is written instead of
        ``mu.npy`` so that load() can detect the mode automatically.
        In centered-only mode both ``mu.npy`` and a ``centered_mode.flag``
        sentinel are written.

        Args:
            dir_path: directory to save into (created if missing)
        """
        dir_path.mkdir(parents=True, exist_ok=True)
        if self.raw:
            (dir_path / "raw_mode.flag").touch()
        else:
            np.save(dir_path / "mu.npy", self.mu)
            if self.centered_only:
                (dir_path / "centered_mode.flag").touch()

    @classmethod
    def load(
        cls,
        dir_path: Path,
        variant: str,
        window_size: int,
        stride: int,
    ) -> StreamingPreprocessor:
        """Load a previously saved StreamingPreprocessor from disk.

        Detects raw mode automatically by checking for the presence of
        ``raw_mode.flag`` in dir_path.  Detects centered-only mode by checking
        for ``centered_mode.flag``.

        Args:
            dir_path: directory containing either mu.npy (normal/centered mode)
                      or raw_mode.flag (raw mode)
            variant: feature variant used when the preprocessor was created
            window_size: window size used when the preprocessor was created
            stride: stride used when the preprocessor was created

        Returns:
            Fitted StreamingPreprocessor ready for transform().
        """
        raw = variant.endswith("_raw") or (dir_path / "raw_mode.flag").exists()
        centered_only = (
            variant.endswith("_centered") or (dir_path / "centered_mode.flag").exists()
        )
        obj = cls(variant, window_size, stride)
        obj.raw = raw  # override in case sentinel exists but variant lacks suffix
        obj.centered_only = centered_only and not raw
        if raw:
            obj.mu = None
            obj._fitted = True
        else:
            obj.mu = np.load(dir_path / "mu.npy")
            obj._fitted = True
        return obj


# ---------------------------------------------------------------------------
# 3. apply_window_and_stride
# ---------------------------------------------------------------------------


def apply_window_and_stride(
    features: torch.Tensor,  # (B, T, D)
    mask: torch.Tensor | None,  # (B, T)
    window_size: int,
    stride: int,
) -> tuple[np.ndarray, list[int]]:
    """Apply causal moving average + stride to per-token features.

    For window_size == 1 the features are returned as-is (modulo stride).
    For window_size > 1 a causal moving average is computed via cumsum so
    that position t only sees tokens t-window_size+1 .. t.

    Padding tokens indicated by mask == 0 are excluded from the output.
    The windowed average uses only valid (non-padding) positions.

    Args:
        features: (B, T, D) float tensor on any device
        mask: (B, T) binary tensor where 1 = valid, 0 = padding; if None,
              all positions are treated as valid
        window_size: causal averaging window length (>= 1)
        stride: step size when subsampling the time dimension (>= 1)

    Returns:
        windowed: (N_total, D) numpy array, flattened across the batch dimension
        doc_lengths: list of int, number of valid windowed positions per document
    """
    B, T, D = features.shape
    device = features.device

    if mask is None:
        mask = torch.ones(B, T, dtype=torch.float32, device=device)
    else:
        mask = mask.float().to(device)

    if window_size == 1:
        # No averaging: just select valid positions at the given stride
        averaged = features  # (B, T, D)
    else:
        # Causal cumsum-based moving average.
        # Mask out padding positions in the feature sum.
        mask_3d = mask.unsqueeze(-1)  # (B, T, 1)
        masked_features = features * mask_3d  # zero-out padding

        cumsum_feat = torch.cumsum(masked_features, dim=1)  # (B, T, D)
        cumsum_mask = torch.cumsum(mask_3d, dim=1)  # (B, T, 1) — count of valid tokens

        # For position t, sum over [t - window_size + 1 .. t]
        # = cumsum[t] - cumsum[t - window_size]  (with boundary clamping)
        shift = window_size - 1
        # Pad cumsum at the front with zeros so index (t - window_size) works at boundaries
        # cumsum_feat: (B, T, D) -> prepend `shift` zero frames -> (B, T+shift, D)
        pad_feat = torch.zeros(B, shift, D, dtype=features.dtype, device=device)
        pad_mask = torch.zeros(B, shift, 1, dtype=features.dtype, device=device)
        cum_f_padded = torch.cat([pad_feat, cumsum_feat], dim=1)   # (B, T+shift, D)
        cum_m_padded = torch.cat([pad_mask, cumsum_mask], dim=1)   # (B, T+shift, 1)

        # Window sum at position t uses indices t+shift (cumsum[t]) - t (cumsum[t-w])
        window_sum = cum_f_padded[:, shift:, :] - cum_f_padded[:, :T, :]  # (B, T, D)
        window_cnt = cum_m_padded[:, shift:, :] - cum_m_padded[:, :T, :]  # (B, T, 1)

        averaged = window_sum / window_cnt.clamp(min=1e-8)  # (B, T, D)

    # Apply stride: collect strided time indices
    time_indices = torch.arange(0, T, stride, device=device)  # (S,)
    strided = averaged[:, time_indices, :]     # (B, S, D)
    mask_strided = mask[:, time_indices]       # (B, S)  — float

    # Flatten valid positions across the batch
    results: list[np.ndarray] = []
    doc_lengths: list[int] = []

    strided_np = strided.detach().cpu().numpy()
    mask_np = mask_strided.detach().cpu().numpy().astype(bool)

    for b in range(B):
        valid = mask_np[b]  # (S,)
        selected = strided_np[b][valid]  # (n_valid, D)
        doc_lengths.append(int(valid.sum()))
        results.append(selected)

    if results:
        windowed = np.concatenate(results, axis=0)  # (N_total, D)
    else:
        windowed = np.empty((0, D), dtype=np.float32)

    return windowed, doc_lengths


# ---------------------------------------------------------------------------
# 4. extract_features
# ---------------------------------------------------------------------------


def extract_features(
    batch: dict,
    extractor,  # HiddenStateExtractor
    encoder: torch.nn.Module | None,
    need_h: bool,
    need_z: bool,
    device: str,
) -> dict[str, torch.Tensor]:
    """Extract h and/or z from one batch via a shared LLM forward pass.

    A single LLM forward pass is performed regardless of whether both h and z
    are requested, keeping GPU overhead minimal.

    For VQ encoders that store pre-quantization activations in aux["z_pre"],
    z is taken from aux["z_pre"] instead of the quantized output so that
    clustering operates in the continuous representation space.

    Args:
        batch: dict with keys "input_ids" and optionally "attention_mask"
        extractor: HiddenStateExtractor instance (frozen LLM)
        encoder: optional encoder nn.Module; required when need_z is True
        need_h: whether to include raw hidden states in the output
        need_z: whether to include encoder output in the output
        device: torch device string (e.g. "cuda", "cpu")

    Returns:
        dict with keys "h" and/or "z", each as a (B, T, D) float32 tensor
        moved to CPU.

    Raises:
        ValueError: if need_z is True but encoder is None
    """
    if need_z and encoder is None:
        raise ValueError("need_z=True but no encoder was provided")

    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        if "hidden_states" in batch:
            # Cached mode: hidden states come from the dataloader
            h = batch["hidden_states"].to(device)
        else:
            input_ids = batch["input_ids"].to(device)
            h = extractor.extract(input_ids, attention_mask=attention_mask)  # (B, T, d_h)

    out: dict[str, torch.Tensor] = {}

    if need_h:
        out["h"] = h.cpu()

    if need_z:
        assert encoder is not None
        with torch.no_grad():
            z, aux = encoder(h, attention_mask=attention_mask)
        # Prefer pre-quantization activations for VQ encoders
        z = aux.get("z_pre", z)
        out["z"] = z.cpu()

    return out


# ---------------------------------------------------------------------------
# 4b. PCA projection helper
# ---------------------------------------------------------------------------


def apply_pca_projection(
    h: torch.Tensor,
    pca_components: np.ndarray,
    pca_mean: np.ndarray,
) -> torch.Tensor:
    """Project hidden states via PCA: (B, T, d_h) -> (B, T, pca_dim).

    Args:
        h: (B, T, d_h) float tensor on any device
        pca_components: (pca_dim, d_h) numpy array of principal components
        pca_mean: (d_h,) numpy array of per-feature mean

    Returns:
        (B, T, pca_dim) float tensor on the same device as h
    """
    comp = torch.from_numpy(pca_components).to(dtype=h.dtype, device=h.device)
    mean = torch.from_numpy(pca_mean).to(dtype=h.dtype, device=h.device)
    return (h - mean) @ comp.T


# ---------------------------------------------------------------------------
# 5. Statistics functions
# ---------------------------------------------------------------------------


def compute_transition_matrix(trajectories: list[np.ndarray], K: int) -> np.ndarray:
    """Compute K x K row-normalized transition probability matrix.

    Each consecutive pair (trajectories[i][t], trajectories[i][t+1]) contributes
    one count to the transition matrix. Rows are normalized to sum to 1 (uniform
    prior for rows with no observed transitions).

    Args:
        trajectories: list of 1-D integer arrays, each holding a sequence of
                      cluster label assignments for one document
        K: total number of clusters

    Returns:
        (K, K) float64 array where entry [i, j] is P(next=j | current=i)
    """
    counts = np.zeros((K, K), dtype=np.float64)
    for traj in trajectories:
        if len(traj) < 2:
            continue
        src = traj[:-1]
        dst = traj[1:]
        for s, d in zip(src, dst):
            counts[s, d] += 1.0
    # Row-normalize; rows with zero sum get uniform distribution
    row_sums = counts.sum(axis=1, keepdims=True)
    zero_rows = (row_sums == 0).squeeze(1)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    transition = counts / row_sums
    transition[zero_rows] = 1.0 / K
    return transition


def compute_transition_matrix_weighted(
    trajectories: list[np.ndarray],
    margins: list[np.ndarray],
    K: int,
    tau: float = 0.1,
    pseudocount: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute margin-weighted, pseudocount-smoothed transition matrix.

    Instead of counting every transition equally, each transition is weighted
    by the confidence (margin) at both endpoints:
        w_t = min(1, m_t / tau) * min(1, m_{t+1} / tau)
        count(i -> j) += w_t

    This downweights boundary-jitter transitions where the assignment is
    ambiguous (low margin).

    Args:
        trajectories: list of 1-D int arrays of cluster label sequences.
        margins: list of 1-D float arrays of cosine margins, aligned with
                 trajectories (same length per document).
        K: total number of clusters.
        tau: margin saturation threshold.  Margins >= tau contribute full
             weight; margins < tau contribute proportionally.
        pseudocount: added to every cell before row-normalization to prevent
                     zero-probability edges.

    Returns:
        P: (K, K) row-normalized transition probability matrix.
        raw_counts: (K, K) raw weighted count matrix (before pseudocount
                    and normalization), useful for macro aggregation.
    """
    raw_counts = np.zeros((K, K), dtype=np.float64)
    for traj, marg in zip(trajectories, margins):
        n = min(len(traj), len(marg))
        if n < 2:
            continue
        for t in range(n - 1):
            w = min(1.0, marg[t] / tau) * min(1.0, marg[t + 1] / tau)
            raw_counts[int(traj[t]), int(traj[t + 1])] += w

    smoothed = raw_counts + pseudocount
    row_sums = smoothed.sum(axis=1, keepdims=True)
    zero_rows = (row_sums == 0).squeeze(1)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    P = smoothed / row_sums
    P[zero_rows] = 1.0 / K
    return P, raw_counts


def prune_transition_matrix(
    P: np.ndarray,
    top_r: int = 3,
    min_prob: float = 0.03,
    keep_self_loops: bool = True,
) -> np.ndarray:
    """Prune a transition matrix to keep only significant edges.

    For each row i, keeps edges that satisfy EITHER:
      - the edge is among the top-r outgoing edges by probability, OR
      - P[i, j] >= min_prob
    Self-loops (i -> i) are always kept when keep_self_loops is True.

    Pruned entries are set to 0.  The matrix is NOT re-normalized (edge
    widths encode the original probability for graph drawing).

    Args:
        P: (K, K) row-normalized transition matrix.
        top_r: keep top-r outgoing edges per node.
        min_prob: keep edges with P[i, j] >= this threshold.
        keep_self_loops: always retain diagonal entries.

    Returns:
        (K, K) array with pruned entries zeroed out.
    """
    K = P.shape[0]
    P_pruned = np.zeros_like(P)
    for i in range(K):
        row = P[i]
        # Top-r indices
        top_indices = set(np.argsort(-row)[:top_r].tolist())
        # Indices above threshold
        above_thresh = set(np.where(row >= min_prob)[0].tolist())
        keep = top_indices | above_thresh
        if keep_self_loops:
            keep.add(i)
        for j in keep:
            P_pruned[i, j] = row[j]
    return P_pruned


def compute_dwell_times(trajectories: list[np.ndarray], K: int) -> dict:
    """Compute dwell time statistics.

    A dwell run is a maximal contiguous subsequence of identical cluster labels.
    Its length is the dwell time.

    Args:
        trajectories: list of 1-D integer arrays of cluster assignments
        K: total number of clusters

    Returns:
        dict with keys:
          "all_dwells": (M,) int array of all run lengths across every trajectory
          "per_cluster_dwells": list of K arrays, one per cluster
          "mean_dwell_per_cluster": (K,) float array of mean dwell per cluster
                                    (NaN for clusters never assigned)
    """
    per_cluster: list[list[int]] = [[] for _ in range(K)]

    for traj in trajectories:
        if len(traj) == 0:
            continue
        # Run-length encode
        prev = traj[0]
        run_len = 1
        for label in traj[1:]:
            if label == prev:
                run_len += 1
            else:
                per_cluster[int(prev)].append(run_len)
                prev = label
                run_len = 1
        per_cluster[int(prev)].append(run_len)

    per_cluster_arrays = [np.array(d, dtype=np.int64) if d else np.array([], dtype=np.int64) for d in per_cluster]

    all_dwells_list: list[int] = []
    for arr in per_cluster_arrays:
        all_dwells_list.extend(arr.tolist())
    all_dwells = np.array(all_dwells_list, dtype=np.int64)

    mean_dwell = np.array(
        [arr.mean() if len(arr) > 0 else np.nan for arr in per_cluster_arrays],
        dtype=np.float64,
    )

    return {
        "all_dwells": all_dwells,
        "per_cluster_dwells": per_cluster_arrays,
        "mean_dwell_per_cluster": mean_dwell,
    }


def compute_cluster_usage(trajectories: list[np.ndarray], K: int) -> np.ndarray:
    """Compute fraction of total assigned tokens in each cluster.

    Args:
        trajectories: list of 1-D integer arrays of cluster assignments
        K: total number of clusters

    Returns:
        (K,) float64 array summing to 1; uniform (1/K) if no tokens at all
    """
    counts = np.zeros(K, dtype=np.float64)
    for traj in trajectories:
        for label in traj:
            counts[int(label)] += 1.0
    total = counts.sum()
    if total == 0:
        return np.full(K, 1.0 / K, dtype=np.float64)
    return counts / total


def compute_trajectory_entropy(trajectories: list[np.ndarray], K: int) -> dict:
    """Compute entropy statistics over cluster label sequences.

    Four entropy measures are computed:
      - mean_per_doc_entropy: mean Shannon entropy of the empirical cluster
        distribution within each document (averaged over documents)
      - global_entropy: Shannon entropy of the global cluster usage distribution
      - max_entropy: log2(K), the maximum possible entropy for K clusters
      - transition_entropy: mean row entropy of the transition probability matrix
        (i.e., average uncertainty about the next state given the current state)

    Args:
        trajectories: list of 1-D integer arrays of cluster assignments
        K: total number of clusters

    Returns:
        dict with keys: "mean_per_doc_entropy", "global_entropy",
                        "max_entropy", "transition_entropy"
    """

    def _entropy(probs: np.ndarray) -> float:
        """Shannon entropy in bits (log2) for a probability distribution."""
        p = probs[probs > 0]
        return float(-np.sum(p * np.log2(p)))

    # Per-document entropy
    per_doc_entropies: list[float] = []
    for traj in trajectories:
        if len(traj) == 0:
            continue
        counts = np.bincount(traj, minlength=K).astype(np.float64)
        total = counts.sum()
        if total == 0:
            continue
        probs = counts / total
        per_doc_entropies.append(_entropy(probs))

    mean_per_doc = float(np.mean(per_doc_entropies)) if per_doc_entropies else 0.0

    # Global entropy from overall cluster usage
    global_usage = compute_cluster_usage(trajectories, K)
    global_ent = _entropy(global_usage)

    # Max entropy
    max_ent = float(np.log2(K)) if K > 1 else 0.0

    # Transition entropy: mean row entropy of the row-normalized transition matrix
    T_mat = compute_transition_matrix(trajectories, K)
    row_entropies = np.array([_entropy(row) for row in T_mat])
    transition_ent = float(row_entropies.mean())

    return {
        "mean_per_doc_entropy": mean_per_doc,
        "global_entropy": global_ent,
        "max_entropy": max_ent,
        "transition_entropy": transition_ent,
    }


# ---------------------------------------------------------------------------
# 6. Hierarchical grouping and macro transitions
# ---------------------------------------------------------------------------


def compute_hybrid_hierarchy(
    centers: np.ndarray,
    P: np.ndarray,
    alpha: float = 0.5,
    K_coarse: int | None = None,
    linkage_method: str = "average",
) -> tuple[np.ndarray, np.ndarray, int]:
    """Compute hybrid geometric+dynamic hierarchy over micro-clusters.

    Combines two distance signals:
        D = alpha * D_geo + (1 - alpha) * D_dyn

    where D_geo is Euclidean distance between L2-normalised centroids and
    D_dyn is the symmetric Jensen-Shannon divergence between transition rows.
    Both are normalised to mean 1 before mixing so that alpha controls the
    true balance.

    Args:
        centers: (K, D) cluster center vectors.
        P: (K, K) row-normalised transition matrix.
        alpha: mixing weight.  1.0 = pure geometric, 0.0 = pure behavioural.
        K_coarse: number of macro-groups.  If None, auto-selects as
                  max(3, min(K // 3, 12)).
        linkage_method: scipy linkage method ("average", "complete", "ward").

    Returns:
        group_labels: (K,) int array mapping micro cluster i -> macro group
                      (0-indexed).
        linkage_matrix: (K-1, 4) scipy linkage matrix.
        K_coarse_used: actual number of macro groups produced.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist, squareform

    K = centers.shape[0]

    # Auto K_coarse
    if K_coarse is None:
        K_coarse = max(3, min(K // 3, 12))
    K_coarse = min(K_coarse, K)  # can't have more groups than clusters

    # --- Geometric distance ---
    norms = np.linalg.norm(centers, axis=1, keepdims=True).clip(min=1e-8)
    normed = centers / norms
    d_geo = pdist(normed, metric="euclidean")

    # --- Behavioural distance (JS divergence between transition rows) ---
    eps = 1e-12
    P_safe = np.clip(P, eps, None)
    # Re-normalise rows after clipping
    P_safe = P_safe / P_safe.sum(axis=1, keepdims=True)

    def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
        m = 0.5 * (p + q)
        kl_pm = float(np.sum(p * np.log2(p / m)))
        kl_qm = float(np.sum(q * np.log2(q / m)))
        return 0.5 * (kl_pm + kl_qm)

    d_dyn_mat = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        for j in range(i + 1, K):
            d_dyn_mat[i, j] = _js_divergence(P_safe[i], P_safe[j])
            d_dyn_mat[j, i] = d_dyn_mat[i, j]
    d_dyn = squareform(d_dyn_mat, checks=False)

    # --- Normalise both to mean 1 before mixing ---
    mean_geo = d_geo.mean() if d_geo.mean() > 0 else 1.0
    mean_dyn = d_dyn.mean() if d_dyn.mean() > 0 else 1.0
    d_geo_normed = d_geo / mean_geo
    d_dyn_normed = d_dyn / mean_dyn

    # --- Mix ---
    d_mixed = alpha * d_geo_normed + (1.0 - alpha) * d_dyn_normed

    # --- Agglomerative clustering ---
    Z = linkage(d_mixed, method=linkage_method)
    labels_1indexed = fcluster(Z, t=K_coarse, criterion="maxclust")
    group_labels = labels_1indexed - 1  # 0-indexed

    K_coarse_used = int(group_labels.max() + 1)
    return group_labels.astype(np.int64), Z, K_coarse_used


def compute_macro_transition_matrix(
    raw_counts: np.ndarray,
    group_labels: np.ndarray,
    K_coarse: int,
    pseudocount: float = 1e-3,
) -> np.ndarray:
    """Aggregate micro raw counts into a macro transition matrix.

    N_macro[A, B] = sum_{i in A, j in B} raw_counts[i, j]

    Then add pseudocount and row-normalise.

    Args:
        raw_counts: (K, K) raw weighted count matrix from micro transitions.
        group_labels: (K,) int array mapping micro cluster -> macro group
                      (0-indexed).
        K_coarse: number of macro groups.
        pseudocount: smoothing constant.

    Returns:
        (K_coarse, K_coarse) row-normalised macro transition matrix.
    """
    K = raw_counts.shape[0]
    macro_counts = np.zeros((K_coarse, K_coarse), dtype=np.float64)
    for i in range(K):
        for j in range(K):
            macro_counts[group_labels[i], group_labels[j]] += raw_counts[i, j]

    smoothed = macro_counts + pseudocount
    row_sums = smoothed.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return smoothed / row_sums


# ---------------------------------------------------------------------------
# Discovery utilities
# ---------------------------------------------------------------------------


def discover_combos(
    centers_dir: Path,
    variants=None,
    K_list=None,
    window_sizes=None,
    seeds=None,
    flat_variant: str | None = None,
) -> list[tuple[str, int, int, int]]:
    """Discover available (variant, w, K, seed) combos from directory structure.

    Expected layout (nested, default)::

        centers_dir/
          {variant}/w{w}/
            mu.npy
            K{K}/seed{seed}/
              centers.npy
              kmeans_model.pkl

    Expected layout (flat, when ``flat_variant`` is set)::

        centers_dir/
          w{w}/
            mu.npy
            K{K}/seed{seed}/
              centers.npy
              kmeans_model.pkl

    Returns list of tuples (variant, w, K, seed).
    """
    combos: list[tuple[str, int, int, int]] = []

    def _scan_w_dirs(base_dir: Path, variant: str) -> None:
        """Scan w{W}/K{K}/seed{S}/ structure under *base_dir* for *variant*."""
        for w_dir in sorted(base_dir.iterdir()):
            if not w_dir.is_dir() or not w_dir.name.startswith("w"):
                continue
            try:
                w = int(w_dir.name[1:])
            except ValueError:
                continue
            if window_sizes and w not in window_sizes:
                continue
            for K_dir in sorted(w_dir.iterdir()):
                if not K_dir.is_dir() or not K_dir.name.startswith("K"):
                    continue
                try:
                    K = int(K_dir.name[1:])
                except ValueError:
                    continue
                if K_list and K not in K_list:
                    continue
                for seed_dir in sorted(K_dir.iterdir()):
                    if not seed_dir.is_dir() or not seed_dir.name.startswith("seed"):
                        continue
                    try:
                        s = int(seed_dir.name[4:])
                    except ValueError:
                        continue
                    if seeds and s not in seeds:
                        continue
                    if (seed_dir / "kmeans_model.pkl").exists():
                        combos.append((variant, w, K, s))

    if flat_variant is not None:
        # Flat layout: w{W}/ dirs are directly under centers_dir
        _scan_w_dirs(centers_dir, flat_variant)
    else:
        # Nested layout: {variant}/w{W}/ dirs
        for variant_dir in sorted(centers_dir.iterdir()):
            if not variant_dir.is_dir():
                continue
            variant = variant_dir.name
            if variants and variant not in variants:
                continue
            _scan_w_dirs(variant_dir, variant)

    return combos


def find_config_yaml(centers_dir: Path, combos: list[tuple[str, int, int, int]], flat: bool = False) -> str | None:
    """Search combo seed dirs for a config.json with a 'config_yaml_path' key."""
    for variant, w, K, seed in combos:
        if flat:
            cfg_json = centers_dir / f"w{w}" / f"K{K}" / f"seed{seed}" / "config.json"
        else:
            cfg_json = centers_dir / variant / f"w{w}" / f"K{K}" / f"seed{seed}" / "config.json"
        if cfg_json.exists():
            try:
                with open(cfg_json) as f:
                    data = json.load(f)
                yaml_path = data.get("config_yaml_path") or data.get("config_yaml")
                if yaml_path and Path(yaml_path).exists():
                    return str(yaml_path)
            except (json.JSONDecodeError, OSError):
                continue
    return None


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def save_matrix_csv(
    matrix: np.ndarray, path: Path, row_prefix: str = "", col_prefix: str = "",
) -> None:
    """Save a 2-D array as a labelled CSV file."""
    n_rows, n_cols = matrix.shape
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = [""] + [f"{col_prefix}{j}" for j in range(n_cols)]
        writer.writerow(header)
        for i in range(n_rows):
            row = [f"{row_prefix}{i}"] + [f"{matrix[i, j]:.4f}" for j in range(n_cols)]
            writer.writerow(row)
