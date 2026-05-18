"""
Experiment: sycophancy — per-token probing

Probes pre-extracted hidden states from multi-turn SYCON-Bench debates
to predict whether the model gives a sycophantic reply at each turn.

Each assistant-token position in turns 2-5 is one sample. Turn 1 is
the anchor (always "not sycophantic") and is skipped.

Input modes for pooled probes (linear, dom):
  h_t                  — h[t]                          [D]
  running_mean_global  — mean(h_0, ..., h_t)           [D]
  running_mean_turn    — mean(h_{turn_start}, ..., h_t) [D]
  z_t                  — RET-encoded z at position t    [z_dim]

Input mode for sequence probes (attention, transformer):
  causal_prefix — h[0:t+1]  [t+1, D]  (variable length)

No LLM loading needed — hidden states are pre-extracted in memmap format.
"""
from __future__ import annotations

import logging
import os
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluation.datasets.sycophancy.config import SycophancyProbingConfig
from evaluation.probing.sycophancy.probe_data import (
    load_judge_results,
    load_canonical_splits,
    build_token_index,
    materialize_pooled_features,
    compute_relative_positions,
    position_stratified_metrics,
    plot_position_accuracy_curve,
    ConversationProbingDataset,
    TurnSegmentDataset,
    collate_conversations,
    compute_sycophancy_labels,
)
from evaluation.datasets.sycophancy.storage import MemmapReader
from evaluation.probing.evaluation import (
    compute_metrics, save_confusion_matrix, print_confusion_matrix,
)
from evaluation.probing.experiment_utils import (
    build_probe_list_for_task,
    probe_output_dir,
    save_probe_results,
    save_training_curves,
    save_experiment_summary,
)

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "sycophancy"

POOLED_INPUT_MODES = ["h_t", "running_mean_global", "running_mean_turn"]
SEQUENCE_INPUT_MODES = ["causal_prefix"]
POOLED_PROBE_NAMES = {"linear", "dom"}
SEQUENCE_PROBE_NAMES = {"attention", "transformer", "transformer_no_proj"}


# ---------------------------------------------------------------------------
# RET encoding helper
# ---------------------------------------------------------------------------

def _build_ret_z_cache(cfg, reader, conv_ids):
    """Encode all conversations through the RET encoder once.

    Returns {conv_id: z_array [S, z_dim]} for each conversation.
    """
    from evaluation.probing.ret_utils import load_ret_encoder, free_ret_encoder

    encoder, ret_cfg = load_ret_encoder(cfg.ret.checkpoint_path, device=cfg.ret.device)
    z_dim = ret_cfg.encoder.z_dim
    logger.info("RET encoding %d conversations (z_dim=%d)...", len(conv_ids), z_dim)

    z_cache = {}
    for i, conv_id in enumerate(conv_ids):
        conv = reader.get_conversation(conv_id)
        h = conv["hidden_states"]  # [S, D]
        h_t = torch.tensor(h, dtype=torch.float32).unsqueeze(0).to(cfg.ret.device)
        mask = torch.ones(1, h.shape[0], dtype=torch.float32, device=cfg.ret.device)
        with torch.no_grad():
            z, _ = encoder(h_t, attention_mask=mask)
        z_cache[conv_id] = z.squeeze(0).cpu().numpy()  # [S, z_dim]

        if (i + 1) % 500 == 0:
            logger.info("  Encoded %d/%d conversations", i + 1, len(conv_ids))

    free_ret_encoder(encoder)
    logger.info("RET encoding complete: %d conversations cached", len(z_cache))
    return z_cache


def _build_ret_z_turn_cache(cfg, reader, conv_ids, judge):
    """Encode each turn's assistant tokens separately through the RET encoder.

    Returns {conv_id: {turn_k: z_array [S_turn, z_dim]}} — one z per turn,
    encoding only within-turn context (no cross-turn information).
    """
    from evaluation.probing.ret_utils import load_ret_encoder, free_ret_encoder

    encoder, ret_cfg = load_ret_encoder(cfg.ret.checkpoint_path, device=cfg.ret.device)
    z_dim = ret_cfg.encoder.z_dim
    logger.info("RET turn-encoding %d conversations (z_dim=%d)...", len(conv_ids), z_dim)

    z_turn_cache = {}
    for i, conv_id in enumerate(conv_ids):
        conv = reader.get_conversation(conv_id)
        h = conv["hidden_states"]
        role_ids = conv["role_ids"]
        turn_ids = conv["turn_ids"]

        turns = {}
        for k in range(2, 6):
            mask = (role_ids == 2) & (turn_ids == k)
            positions = np.where(mask)[0]
            if len(positions) == 0:
                continue
            h_turn = h[positions[0]:positions[-1] + 1]  # [S_turn, D]
            with torch.no_grad():
                h_t = torch.tensor(h_turn, dtype=torch.float32).unsqueeze(0).to(cfg.ret.device)
                m = torch.ones(1, h_turn.shape[0], dtype=torch.float32, device=cfg.ret.device)
                z, _ = encoder(h_t, attention_mask=m)
            turns[k] = z.squeeze(0).cpu().numpy()  # [S_turn, z_dim]
        z_turn_cache[conv_id] = turns

        if (i + 1) % 500 == 0:
            logger.info("  Turn-encoded %d/%d conversations", i + 1, len(conv_ids))

    free_ret_encoder(encoder)
    logger.info("RET turn-encoding complete: %d conversations", len(z_turn_cache))
    return z_turn_cache


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _train_pooled_probe(probe, train_loader, val_loader=None):
    """Train a pooled probe from DataLoaders."""
    probe_type = type(probe).__name__

    if hasattr(probe, "fit_from_dataloader"):
        if "DOM" in probe_type:
            probe.fit_from_dataloader(train_loader)
        else:
            probe.fit_from_dataloader(train_loader, val_loader=val_loader)
    else:
        raise RuntimeError(f"Probe {probe_type} does not support fit_from_dataloader")

    return probe


def _eval_probe_on_loader(probe, loader):
    """Evaluate a trained probe on a DataLoader. Returns metrics dict."""
    all_preds = []
    all_labels = []
    for xb, yb in loader:
        if isinstance(xb, torch.Tensor):
            xb_np = xb.numpy()
        else:
            xb_np = xb
        preds = probe.predict(xb_np)
        all_preds.append(preds)
        if isinstance(yb, torch.Tensor):
            yb = yb.numpy()
        all_labels.append(yb)

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    return compute_metrics(y_true, y_pred)


def _train_sequence_probe_conv(probe, train_loader, val_loader=None, max_epochs=50, lr=1e-3, l2=1e-4, device="cuda"):
    """Train a sequence probe from conversation-level DataLoaders.

    Each batch yields {hidden_states [B,S,D], labels [B,S], supervision_mask [B,S], attention_mask [B,S]}.
    The probe outputs per-position logits; loss is masked to supervised positions only.
    """
    from torch.optim import AdamW

    # Infer D from first batch and build params if needed
    first_batch = next(iter(train_loader))
    D = first_batch["hidden_states"].shape[2]
    if hasattr(probe, "_fitted") and not probe._fitted:
        if hasattr(probe, "_build_params"):
            probe._build_params(D)
        elif hasattr(probe, "_build"):
            probe._build(D)
        probe._fitted = True
    probe.to(device)

    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
    optimizer = AdamW(probe.parameters(), lr=lr, weight_decay=l2)

    probe.train()
    for epoch in range(max_epochs):
        total_loss = 0.0
        total_correct = 0
        total_supervised = 0

        for batch in train_loader:
            h = batch["hidden_states"].to(device)
            y = batch["labels"].to(device)
            sup = batch["supervision_mask"].to(device)
            mask = batch["attention_mask"].to(device)

            optimizer.zero_grad()
            logits = probe._forward(h, mask, per_position=True)  # [B, S]

            # Masked loss: only on supervised positions
            per_pos_loss = loss_fn(logits, y)  # [B, S]
            n_sup = sup.sum().clamp(min=1)
            loss = (per_pos_loss * sup).sum() / n_sup
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                preds = (logits >= 0).float()
                total_correct += ((preds == y) * sup).sum().item()
                total_supervised += sup.sum().item()
                total_loss += loss.item()

        acc = total_correct / max(total_supervised, 1)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info("  Epoch %d/%d  loss=%.4f  acc=%.3f",
                        epoch + 1, max_epochs, total_loss / max(len(train_loader), 1), acc)

    probe.eval()
    return probe


def _eval_sequence_probe_conv(probe, loader, device="cuda"):
    """Evaluate a sequence probe on conversation-level DataLoader.

    Returns metrics dict including position_stratified if rel_positions
    are available in the batch.
    """
    all_preds = []
    all_labels = []
    all_rel_pos = []

    with torch.no_grad():
        for batch in loader:
            h = batch["hidden_states"].to(device)
            y = batch["labels"]
            sup = batch["supervision_mask"]
            mask = batch["attention_mask"].to(device)
            rel = batch.get("rel_positions")

            logits = probe._forward(h, mask, per_position=True)  # [B, S]
            preds = (logits >= 0).float().cpu()

            # Extract predictions at supervised positions
            for b in range(h.shape[0]):
                sup_pos = (sup[b] == 1).nonzero(as_tuple=True)[0]
                all_preds.append(preds[b, sup_pos].numpy().astype(int))
                all_labels.append(y[b, sup_pos].numpy().astype(int))
                if rel is not None:
                    all_rel_pos.append(rel[b, sup_pos].numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    metrics = compute_metrics(y_true, y_pred)

    if all_rel_pos:
        rel_positions = np.concatenate(all_rel_pos)
        pos_metrics = position_stratified_metrics(y_true, y_pred, rel_positions)
        metrics["position_stratified"] = pos_metrics

    return metrics


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

class SycophancyRunner:
    """Per-token sycophancy probing (paper Section 5).

    A concrete, standalone pipeline — intentionally NOT folded into
    ``run_probing``: its per-token, variable-length causal-prefix probing
    is structurally different from the prompt-level path.
    """

    def run(self, cfg: SycophancyProbingConfig) -> Dict:
        os.makedirs(cfg.output_dir, exist_ok=True)

        # Reproducibility: set torch/np seeds once at runner start.
        # Affects probe init, DataLoader shuffle, sklearn-side randomness, and
        # token-subsampling (passed explicitly to build_token_index below).
        seed = int(getattr(cfg, "seed", 42))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        logger.info("SycophancyRunner seed = %d", seed)

        # Resolve paths
        cache_dir = os.path.join(cfg.data_dir, f"layer_{cfg.layer}_activations_cache")
        judge_path = os.path.join(cfg.data_dir, "judge_results.jsonl")

        if not os.path.isdir(cache_dir):
            raise FileNotFoundError(f"Activations cache not found: {cache_dir}")
        if not os.path.isfile(judge_path):
            raise FileNotFoundError(f"Judge results not found: {judge_path}")

        # Open reader and load judge results
        reader = MemmapReader(cache_dir)
        judge = load_judge_results(judge_path)
        logger.info("Loaded %d judge results, %d conversations in cache",
                    len(judge), len(reader))

        # Load canonical splits (conversation-level, filtered to judged)
        splits = load_canonical_splits(cfg.data_dir, reader, judge,
                                        split_file=cfg.split_file)

        if cfg.max_conversations is not None:
            # Only limit training set; keep val/test full for fair comparison
            splits["train"] = splits["train"][:cfg.max_conversations]

        # Build per-token indices for each split
        logger.info(
            "Sycophancy mode: %s (truncate_on=%s, user_context_only=%s)",
            cfg.mode, cfg.truncate_on, getattr(cfg, "user_context_only", False),
        )
        token_indices = {}
        for name, conv_ids in splits.items():
            token_indices[name] = build_token_index(
                reader, conv_ids, judge,
                max_tokens_per_turn=cfg.max_tokens_per_turn,
                seed=seed,
                mode=cfg.mode,
                truncate_on=cfg.truncate_on,
                user_context_only=getattr(cfg, "user_context_only", False),
            )
            n_pos = sum(1 for t in token_indices[name] if t[-1] == 1)
            n_tot = len(token_indices[name])
            logger.info("  %s: %d token-samples (%d conversations), %d sycophantic (%.1f%%)",
                        name, n_tot, len(conv_ids), n_pos,
                        100 * n_pos / n_tot if n_tot > 0 else 0)

        # PCA projection (if configured)
        pca_active = cfg.pca.pca_dir is not None and cfg.pca.pca_dim is not None
        pca_components, pca_mean = None, None
        if pca_active:
            os.makedirs(cfg.pca.pca_dir, exist_ok=True)
            npz_path = os.path.join(cfg.pca.pca_dir, "pca_components.npz")
            if os.path.exists(npz_path):
                saved = np.load(npz_path)
                pca_components = saved["components"].astype(np.float32)
                pca_mean = saved["mean"].astype(np.float32)
                logger.info("Loaded PCA: %d -> %d dims (explains %.1f%%)",
                            pca_components.shape[1], pca_components.shape[0],
                            100 * saved["explained_variance_ratio"].sum())
            else:
                # Fit PCA on train h_t features
                logger.info("Fitting PCA (n_components=%d) on train h_t...", cfg.pca.pca_dim)
                X_pca, _ = materialize_pooled_features(
                    reader, token_indices["train"], mode="h_t")
                from sklearn.decomposition import PCA
                pca = PCA(n_components=cfg.pca.pca_dim)
                pca.fit(X_pca)
                pca_components = pca.components_.astype(np.float32)
                pca_mean = pca.mean_.astype(np.float32)
                np.savez(npz_path, components=pca_components, mean=pca_mean,
                         explained_variance_ratio=pca.explained_variance_ratio_)
                logger.info("Saved PCA to %s (explains %.1f%%)",
                            npz_path, 100 * pca.explained_variance_ratio_.sum())
                del X_pca

        # RET encoding (if configured)
        ret_active = cfg.ret.checkpoint_path is not None
        z_cache = None
        z_turn_cache = None
        if ret_active:
            all_conv_ids = []
            for conv_ids in splits.values():
                all_conv_ids.extend(conv_ids)
            all_conv_ids = list(set(all_conv_ids))
            z_cache = _build_ret_z_cache(cfg, reader, all_conv_ids)
            z_turn_cache = _build_ret_z_turn_cache(cfg, reader, all_conv_ids, judge)

        # Build probe list
        probes = build_probe_list_for_task(cfg, task_type="binary", n_classes=2)

        # Determine which input modes to run
        if ret_active:
            probes = [(n, f) for n, f in probes if n in POOLED_PROBE_NAMES]
            logger.info("RET active: running pooled probes only: %s",
                        [n for n, _ in probes])

        batch_size = cfg.probe.probe_batch_size
        all_results: Dict = {}

        for probe_name, probe_factory in probes:
            logger.info("=" * 60)
            logger.info("Probe: %s", probe_name)
            logger.info("=" * 60)

            probe_dir = probe_output_dir(cfg.output_dir, probe_name, EXPERIMENT_NAME)
            os.makedirs(probe_dir, exist_ok=True)
            cm_dir = os.path.join(probe_dir, "confusion_matrices")
            os.makedirs(cm_dir, exist_ok=True)
            all_results[probe_name] = {}

            if probe_name in POOLED_PROBE_NAMES:
                # Determine modes: pooled h modes + optionally z_t / z_t_turn.
                # For user_context_only we still run all 3 pooling variants on h
                # so the comparison can include (a) just the assistant-marker
                # token [h_t]; (b) cumulative mean over ALL prior tokens
                # including system + prior turns + this turn's user message
                # [running_mean_global]; (c) mean within the current turn
                # [running_mean_turn, averages the `<|im_start|>` + `assistant`
                # marker pair at positions[0..1]].
                # For RET we drop z_t_turn under user_context_only (degenerate
                # at offset 1 of the per-turn encoding).
                user_ctx_only_flag = getattr(cfg, "user_context_only", False)
                if ret_active:
                    modes = ["z_t"] if user_ctx_only_flag else ["z_t", "z_t_turn"]
                else:
                    modes = list(POOLED_INPUT_MODES)

                for mode in modes:
                    logger.info("--- Mode: %s ---", mode)

                    # Materialize features into arrays (fast: one pass over convs)
                    X_train, y_train = materialize_pooled_features(
                        reader, token_indices["train"], mode=mode,
                        z_cache=z_cache, z_turn_cache=z_turn_cache,
                        pca_components=pca_components, pca_mean=pca_mean)
                    X_val, y_val = materialize_pooled_features(
                        reader, token_indices["val"], mode=mode,
                        z_cache=z_cache, z_turn_cache=z_turn_cache,
                        pca_components=pca_components, pca_mean=pca_mean)
                    X_test, y_test = materialize_pooled_features(
                        reader, token_indices["test"], mode=mode,
                        z_cache=z_cache, z_turn_cache=z_turn_cache,
                        pca_components=pca_components, pca_mean=pca_mean)

                    # Train
                    probe = probe_factory()
                    xv = X_val if len(y_val) > 0 else None
                    yv = y_val if len(y_val) > 0 else None
                    probe.fit(X_train, y_train, X_val=xv, y_val=yv)

                    # Save probe weights
                    if hasattr(probe, "save"):
                        save_path = os.path.join(probe_dir, f"model_{mode}.pt")
                        probe.save(save_path)
                        logger.info("Saved probe weights: %s", save_path)

                    # Evaluate on val and test with position-stratified metrics
                    split_metrics = {}
                    for split_name, X_eval, y_eval, split_idx in [
                        ("val", X_val, y_val, token_indices["val"]),
                        ("test", X_test, y_test, token_indices["test"]),
                    ]:
                        if len(y_eval) == 0:
                            logger.info("  Skipping %s evaluation (empty split)", split_name)
                            continue
                        y_pred = probe.predict(X_eval)
                        metrics = compute_metrics(y_eval, y_pred)

                        # Position-stratified evaluation
                        rel_pos = compute_relative_positions(split_idx)
                        pos_metrics = position_stratified_metrics(y_eval, y_pred, rel_pos)
                        metrics["position_stratified"] = pos_metrics

                        split_metrics[split_name] = metrics

                        cm = np.array(metrics["confusion_matrix"])
                        classes = metrics["cm_classes"]
                        print_confusion_matrix(cm, classes,
                                               title=f"{probe_name}/{mode} ({split_name})")
                        cm_path = os.path.join(cm_dir, f"{mode}_{split_name}.png")
                        save_confusion_matrix(cm, classes, cm_path)

                    # Attach training history
                    if hasattr(probe, "history_") and "val" in split_metrics:
                        split_metrics["val"]["training_history"] = probe.history_

                    key = f"sycophancy/{mode}"
                    all_results[probe_name][key] = split_metrics

                    val_acc = split_metrics.get("val", {}).get("accuracy", float("nan"))
                    test_acc = split_metrics.get("test", {}).get("accuracy", float("nan"))
                    test_bal_auc = (split_metrics.get("test", {})
                                    .get("position_stratified", {})
                                    .get("bal_auc", float("nan")))
                    logger.info("  %s/%s: val_acc=%.3f  test_acc=%.3f  test_bal_auc=%.3f",
                                probe_name, mode, val_acc, test_acc, test_bal_auc)

                    # Free memory before next mode
                    del X_train, X_val, X_test, y_train, y_val, y_test

            elif probe_name in SEQUENCE_PROBE_NAMES:
                # Run two variants: full_sequence (full conversation context)
                # and turn_segment (turn-only context, fair comparison to running_mean_turn).
                # In user_context_only mode, skip turn_segment (it would only have
                # one token per segment, degenerate).
                user_ctx_only = getattr(cfg, "user_context_only", False)
                seq_modes = ["full_sequence"] if user_ctx_only else ["full_sequence", "turn_segment"]
                for seq_mode in seq_modes:
                    logger.info("--- Mode: %s ---", seq_mode)

                    if seq_mode == "full_sequence":
                        train_ds = ConversationProbingDataset(
                            reader, splits["train"], judge, mode=cfg.mode,
                            truncate_on=cfg.truncate_on,
                            user_context_only=user_ctx_only)
                        test_ds = ConversationProbingDataset(
                            reader, splits["test"], judge, mode=cfg.mode,
                            truncate_on=cfg.truncate_on,
                            user_context_only=user_ctx_only)
                        bs = min(batch_size, 8)
                    else:  # turn_segment
                        train_ds = TurnSegmentDataset(
                            reader, splits["train"], judge, mode=cfg.mode,
                            truncate_on=cfg.truncate_on)
                        test_ds = TurnSegmentDataset(
                            reader, splits["test"], judge, mode=cfg.mode,
                            truncate_on=cfg.truncate_on)
                        bs = min(batch_size, 32)  # turn segments are shorter

                    train_loader = DataLoader(
                        train_ds, batch_size=bs,
                        shuffle=True, num_workers=0,
                        collate_fn=collate_conversations)
                    test_loader = DataLoader(
                        test_ds, batch_size=bs,
                        shuffle=False, num_workers=0,
                        collate_fn=collate_conversations)

                    probe = probe_factory()
                    _train_sequence_probe_conv(
                        probe, train_loader,
                        max_epochs=cfg.probe.max_epochs,
                        lr=cfg.probe.lr, l2=cfg.probe.l2,
                        device=cfg.model.device,
                    )

                    if hasattr(probe, "save"):
                        save_path = os.path.join(probe_dir, f"model_{seq_mode}.pt")
                        probe.save(save_path)

                    split_metrics = {}
                    for split_name, loader, n_items in [
                        ("test", test_loader, len(test_ds)),
                    ]:
                        if n_items == 0:
                            logger.info("  Skipping %s evaluation (empty)", split_name)
                            continue
                        metrics = _eval_sequence_probe_conv(probe, loader, device=cfg.model.device)
                        split_metrics[split_name] = metrics

                        cm = np.array(metrics["confusion_matrix"])
                        classes = metrics["cm_classes"]
                        print_confusion_matrix(cm, classes,
                                               title=f"{probe_name}/{seq_mode} ({split_name})")
                        cm_path = os.path.join(cm_dir, f"{seq_mode}_{split_name}.png")
                        save_confusion_matrix(cm, classes, cm_path)

                    key = f"sycophancy/{seq_mode}"
                    all_results[probe_name][key] = split_metrics

                    test_acc = split_metrics.get("test", {}).get("accuracy", float("nan"))
                    test_bal = split_metrics.get("test", {}).get("balanced_accuracy", float("nan"))
                    test_bal_auc = (split_metrics.get("test", {})
                                    .get("position_stratified", {})
                                    .get("bal_auc", float("nan")))
                    logger.info("  %s/%s: test_acc=%.3f  test_bal_acc=%.3f  test_bal_auc=%.3f",
                                probe_name, seq_mode, test_acc, test_bal, test_bal_auc)

        # Save results and summary
        for probe_name, probe_results in all_results.items():
            pdir = probe_output_dir(cfg.output_dir, probe_name, EXPERIMENT_NAME)
            save_probe_results(pdir, probe_name, probe_results)
            save_training_curves(pdir, probe_results, probe_name)

        save_experiment_summary(cfg.output_dir, all_results, EXPERIMENT_NAME)

        # Plot position-accuracy curves (test split, all probes+modes)
        test_curves = {}
        for probe_name, probe_results in all_results.items():
            for key, split_metrics in probe_results.items():
                if not isinstance(split_metrics, dict):
                    continue
                test_m = split_metrics.get("test", {})
                pos_data = test_m.get("position_stratified")
                if pos_data:
                    curve_name = f"{probe_name}/{key.split('/')[-1]}"
                    test_curves[curve_name] = pos_data

        if test_curves:
            plot_path = os.path.join(cfg.output_dir, "position_accuracy_curve.png")
            plot_position_accuracy_curve(test_curves, plot_path,
                                         title="Sycophancy Detection: Accuracy vs. Position in Turn")

        # Print summary
        print("\n" + "=" * 70)
        print("  SYCOPHANCY PROBE SUMMARY (per-token)")
        print("=" * 70)
        for probe_name, probe_results in all_results.items():
            for key, split_metrics in probe_results.items():
                if not isinstance(split_metrics, dict):
                    continue
                for split_name in ("val", "test"):
                    m = split_metrics.get(split_name)
                    if not m or "accuracy" not in m:
                        continue
                    bal_auc = m.get("position_stratified", {}).get("bal_auc", 0)
                    print(f"  {probe_name:12s} | {key:30s} | {split_name:4s} | "
                          f"acc={m['accuracy']:.3f}  "
                          f"bal_acc={m['balanced_accuracy']:.3f}  "
                          f"bal_auc={bal_auc:.3f}")
        print()

        return all_results


_runner = SycophancyRunner()


def run(cfg: SycophancyProbingConfig) -> Dict:
    return _runner.run(cfg)
