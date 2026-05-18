"""Assign cluster labels and generate trajectory analysis.

Reads clustering_meta.json from centers_dir (produced by train_centers.py)
to auto-discover the variant, config, checkpoint, and PCA paths.

Usage:
    python scripts/clustering/test_assign.py outputs/.../ret_jepa_auth/clustering
    python scripts/clustering/test_assign.py outputs/.../baselines/h/clustering --split val
    python scripts/clustering/test_assign.py outputs/.../baselines/h_pca128/clustering --max-docs 100
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from evaluation.baselines.sae import encode_sae
from evaluation.clustering.config import (
    build_dataloader,
    load_config_with_overlay,
)
from evaluation.clustering.core import (
    save_matrix_csv,
    discover_combos,
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assign clusters to data splits and generate trajectory analysis."
    )
    parser.add_argument(
        "centers_dir",
        type=str,
        help="Root output directory from train_centers.py (must contain clustering_meta.json)",
    )

    # Filtering
    parser.add_argument("--K-list", type=int, nargs="+", default=None)
    parser.add_argument("--window-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)

    # Data
    parser.add_argument(
        "--split", type=str, default="train",
        help="Split for the main pass (cluster naming, transition matrix, dwell "
             "stats). Default 'train' so naming/stats never touch the held-out "
             "test split. The second pass always loads test for "
             "example_test_trajectories/ HTMLs.",
    )
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--num-timeline-docs", type=int, default=5)
    parser.add_argument("--stride", type=int, default=1)

    # State machine diagram parameters
    parser.add_argument("--tau", type=float, default=0.1,
                        help="Margin saturation threshold for weighted transitions (default: 0.1)")
    parser.add_argument("--pseudocount", type=float, default=1e-3,
                        help="Pseudocount for transition smoothing (default: 1e-3)")
    parser.add_argument("--prune-top-r", type=int, default=3,
                        help="Keep top-r outgoing micro edges per node (default: 3)")
    parser.add_argument("--prune-min-prob", type=float, default=0.03,
                        help="Keep micro edges with P >= this threshold (default: 0.03)")
    parser.add_argument("--macro-top-r", type=int, default=3,
                        help="Keep top-r outgoing macro edges per group (default: 3)")
    parser.add_argument("--macro-min-prob", type=float, default=0.03,
                        help="Keep macro edges with P >= this threshold (default: 0.03)")
    parser.add_argument("--alpha", type=float, nargs="+", default=[1.0],
                        help="Hierarchy mixing weight(s): 1=geometric, 0=behavioral (default: [ 1.0])")
    parser.add_argument("--K-coarse", type=int, default=None,
                        help="Number of macro groups (default: auto = max(3, min(K//3, 12)))")
    parser.add_argument("--linkage-method", type=str, default="average",
                        choices=["average", "complete", "ward"],
                        help="Agglomerative linkage method (default: average)")
    parser.add_argument("--skip-state-machine", action="store_true",
                        help="Skip state machine diagram generation")
    parser.add_argument("--num-test-docs", type=int, default=None,
                        help="Number of test-split docs for colored text examples "
                             "(default: same as --num-timeline-docs)")
    parser.add_argument("--naming-context-tokens", type=int, default=20,
                        help="Context tokens per snippet in v2 naming prompt (default: 20)")
    parser.add_argument("--naming-proto-count", type=int, default=100,
                        help="Max PROTOTYPICAL snippets per cluster in naming (default: 100)")
    parser.add_argument("--naming-typical-count", type=int, default=50,
                        help="Max TYPICAL snippets per cluster in naming (default: 50)")
    parser.add_argument("--naming-reference-file", type=str, default=None,
                        help="Path to reference labels file for v2 naming "
                             "(one 'G0: name' or 'C5: name' per line)")

    # Output
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)

    args = parser.parse_args()

    centers_dir = Path(args.centers_dir).resolve()
    if not centers_dir.is_dir():
        print(f"ERROR: centers_dir does not exist: {centers_dir}", file=sys.stderr)
        sys.exit(1)

    # Load clustering metadata saved by train_centers.py
    meta_path = centers_dir / "clustering_meta.json"
    if not meta_path.exists():
        print(f"ERROR: clustering_meta.json not found in {centers_dir}. "
              f"Was this directory created by the updated train_centers.py?", file=sys.stderr)
        sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)

    variant = meta["variant"]
    config_yaml = meta["config_yaml_path"]
    checkpoint = meta.get("checkpoint_path")
    pca_dir = meta.get("pca_dir")
    sae_checkpoint = meta.get("sae_checkpoint")

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------
    from ret.utils.seeding import seed_everything

    seed_everything(args.seed)

    # ------------------------------------------------------------------
    # 1. Discover combos (always flat layout)
    # ------------------------------------------------------------------
    combos = discover_combos(
        centers_dir,
        K_list=args.K_list,
        window_sizes=args.window_sizes,
        seeds=args.seeds,
        flat_variant=variant,
    )
    if not combos:
        print("No (variant, w, K, seed) combos found in", centers_dir, file=sys.stderr)
        sys.exit(1)

    print(f"Discovered {len(combos)} combo(s):")
    for combo in combos:
        print(f"  variant={combo[0]}  w={combo[1]}  K={combo[2]}  seed={combo[3]}")

    # Unique axes
    unique_variants = [variant]
    unique_windows = sorted({c[1] for c in combos})
    unique_Ks = sorted({c[2] for c in combos})
    unique_seeds = sorted({c[3] for c in combos})

    # ------------------------------------------------------------------
    # 2. Load config
    # ------------------------------------------------------------------
    config = load_config_with_overlay(config_yaml)

    print(f"Config loaded (variant={variant}, LLM: {config.llm.name}, layer {config.llm.layer_index})")

    # ------------------------------------------------------------------
    # 3. Determine which feature types are needed
    # ------------------------------------------------------------------
    need_z = variant.startswith("z")
    need_h = variant.startswith("h") or variant == "sae"

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    # ------------------------------------------------------------------
    # 4. Build encoder (if needed)
    # ------------------------------------------------------------------
    encoder = None
    if need_z:
        if checkpoint is None:
            print("ERROR: z variant requires checkpoint but none in clustering_meta.json", file=sys.stderr)
            sys.exit(1)
        from ret.models.encoders import build_encoder

        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        li = config.llm.layer_index
        if li == "all":
            dp_key = "down_proj.0.weight"
            concat_dim = ckpt["encoder"][dp_key].shape[1] if dp_key in ckpt["encoder"] else None
        elif isinstance(li, list) and len(li) > 1:
            concat_dim = len(li) * config.llm.hidden_dim
        else:
            concat_dim = None
        encoder = build_encoder(config.llm.hidden_dim, config.encoder, concat_dim=concat_dim).to(device)
        encoder.load_state_dict(ckpt["encoder"])
        encoder.eval()
        print(f"Encoder loaded from {checkpoint}")

    # ------------------------------------------------------------------
    # 5. Build hidden-state extractor (skip in cached mode)
    # ------------------------------------------------------------------
    is_cached = config.data.mode == "cached"
    extractor = None
    if not is_cached:
        from ret.extraction.hooks import HiddenStateExtractor
        extractor = HiddenStateExtractor(config.llm, device=device)
        print("HiddenStateExtractor initialised.")
    else:
        print("Cached mode: skipping HiddenStateExtractor (hidden states from dataloader)")

    # ------------------------------------------------------------------
    # 5b. Load SAE model (if sae variant present)
    # ------------------------------------------------------------------
    sae_model = None
    if "sae" == variant:
        from evaluation.probing.sae_utils import load_sae_encoder
        # Prefer the SAE source recorded in clustering_meta.json (e.g. andyrdt path);
        # if absent (legacy meta), fall back to default Goodfire L15.
        sae_ckpt = sae_checkpoint if sae_checkpoint and Path(sae_checkpoint).exists() else None
        sae_model, sae_cfg = load_sae_encoder(checkpoint_path=sae_ckpt, device=device)
        print(f"SAE loaded: d_in={sae_cfg['d_in']}, d_sae={sae_cfg['d_sae']}, k={sae_cfg['k']}, source={sae_cfg['source']}")

    # ------------------------------------------------------------------
    # 6. Load preprocessors per (variant, w)
    # ------------------------------------------------------------------
    from evaluation.clustering.core import StreamingPreprocessor

    preprocessors: dict[tuple[str, int], StreamingPreprocessor] = {}
    for variant in unique_variants:
        for w in unique_windows:
            # Check that at least one combo for this (variant, w) exists
            if not any(c[0] == variant and c[1] == w for c in combos):
                continue
            w_dir = centers_dir / f"w{w}"
            preprocessors[(variant, w)] = StreamingPreprocessor.load(
                w_dir, variant, w, args.stride
            )
            print(f"  Preprocessor loaded for ({variant}, w={w})")

    # ------------------------------------------------------------------
    # 6b. Load PCA components (if h_pca or sae variant present)
    # ------------------------------------------------------------------
    pca_components: dict[str, np.ndarray] = {}
    pca_mean: dict[str, np.ndarray] = {}
    pca_variants = [v for v in unique_variants if v in ("h_pca", "sae")]
    if pca_variants:
        from evaluation.clustering.core import apply_pca_projection

        for pca_var in pca_variants:
            # Determine PCA load path: pca_dir from metadata > centers_dir
            if pca_dir:
                pca_path = Path(pca_dir) / "pca_components.npz"
            else:
                pca_path = centers_dir / "pca_components.npz"
            if not pca_path.exists():
                print(f"ERROR: PCA components not found at {pca_path}", file=sys.stderr)
                sys.exit(1)
            pca_data = np.load(pca_path)
            pca_components[pca_var] = pca_data["components"].astype(np.float32)
            pca_mean[pca_var] = pca_data["mean"].astype(np.float32)
            print(f"  PCA components loaded for '{pca_var}': {pca_components[pca_var].shape[0]} dims from {pca_path}")

    # ------------------------------------------------------------------
    # 7. Load k-means models
    # ------------------------------------------------------------------
    from evaluation.clustering.core import load_kmeans

    kmeans_models: dict[tuple[str, int, int, int], object] = {}
    for variant, w, K, seed in combos:
        seed_dir = centers_dir / f"w{w}" / f"K{K}" / f"seed{seed}"
        kmeans_models[(variant, w, K, seed)] = load_kmeans(seed_dir)
        print(f"  K-means loaded for ({variant}, w={w}, K={K}, seed={seed})")

    # ------------------------------------------------------------------
    # 8–10. Loop over requested splits
    # ------------------------------------------------------------------
    from evaluation.clustering.core import (
        apply_window_and_stride,
        extract_features,
        predict_with_scores,
    )

    if extractor is not None:
        tokenizer = extractor.tokenizer
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.llm.name)

    from evaluation.clustering.core import (
        compute_cluster_usage,
        compute_dwell_times,
        compute_hybrid_hierarchy,
        compute_macro_transition_matrix,
        compute_trajectory_entropy,
        compute_transition_matrix,
        compute_transition_matrix_weighted,
        prune_transition_matrix,
    )
    from evaluation.clustering.plotting.plots import (
        plot_center_similarity,
        plot_cluster_usage,
        plot_cosine_margin_histogram,
        plot_dendrogram,
        plot_dwell_histogram,
        plot_timeline_strips,
        plot_transition_heatmap,
        plot_center_scatter_2d,
    )
    from evaluation.clustering.plotting.state_machine import (
        plot_state_machine_exit,
        plot_state_machine_hierarchical,
        plot_state_machine_micro,
    )
    from evaluation.clustering.plotting.naming import (
        write_cluster_naming_details,
        write_cluster_naming_input,
        write_cluster_naming_overview,
        write_cluster_group_naming_v2,
        _parse_naming_file,
    )
    from evaluation.clustering.plotting.html import render_interactive_trajectory_html

    # Determine output root
    if args.output_dir is not None:
        output_root = Path(args.output_dir)
    else:
        output_root = centers_dir

    num_timeline_docs = args.num_timeline_docs

    split = args.split

    split_loader = build_dataloader(config, split=split, batch_size=args.batch_size, seed=args.seed, shuffle=False, drop_last=False)
    print(f"Dataloader built (split={split}, batch_size={args.batch_size}, cached={is_cached})")

    # Lazy tokenizer load (after dataloader warms HF cache)
    if tokenizer is None:
        from transformers import AutoTokenizer
        tok_kwargs = {}
        if config.llm.trust_remote_code:
            tok_kwargs["trust_remote_code"] = True
        tokenizer = AutoTokenizer.from_pretrained(config.llm.name, **tok_kwargs)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # --------------------------------------------------------------
    # Stream through split
    # --------------------------------------------------------------
    all_trajectories: dict[tuple[str, int, int, int], list[np.ndarray]] = {
        combo: [] for combo in combos
    }
    all_cos: dict[tuple[str, int, int, int], list[np.ndarray]] = {
        combo: [] for combo in combos
    }
    all_margins: dict[tuple[str, int, int, int], list[np.ndarray]] = {
        combo: [] for combo in combos
    }
    all_tokens: list[list[str]] = []
    all_token_ids: list[list[int]] = []

    doc_count = 0

    for batch in split_loader:
        features = extract_features(
            batch, extractor, encoder, need_h, need_z, device
        )
        mask = batch.get("attention_mask")
        B = batch["input_ids"].shape[0]

        for b in range(B):
            ids_b = batch["input_ids"][b]
            if mask is not None:
                valid = mask[b].bool()
                ids_valid = ids_b[valid].tolist()
            else:
                ids_valid = ids_b.tolist()
            all_tokens.append([tokenizer.decode([tid]) for tid in ids_valid])
            all_token_ids.append(ids_valid)

        for variant in unique_variants:
            raw_key = "z" if variant.startswith("z") else "h"
            raw_feat = features[raw_key]

            if variant == "sae":
                raw_feat = encode_sae(sae_model, raw_feat)
            if variant in ("h_pca", "sae"):
                raw_feat = apply_pca_projection(raw_feat, pca_components[variant], pca_mean[variant])

            for w in unique_windows:
                if (variant, w) not in preprocessors:
                    continue

                windowed, doc_lens = apply_window_and_stride(raw_feat, mask, w, args.stride)
                if windowed.shape[0] == 0:
                    continue
                preprocessed = preprocessors[(variant, w)].transform(windowed)

                for combo in combos:
                    if combo[0] != variant or combo[1] != w:
                        continue
                    _, _, K, seed = combo
                    km = kmeans_models[combo]
                    _pp = preprocessors[(variant, w)]
                    labels, top1_cos, margin = predict_with_scores(km, preprocessed, raw=_pp.raw or _pp.centered_only)

                    offset = 0
                    for doc_len in doc_lens:
                        all_trajectories[combo].append(labels[offset : offset + doc_len])
                        all_cos[combo].append(top1_cos[offset : offset + doc_len])
                        all_margins[combo].append(margin[offset : offset + doc_len])
                        offset += doc_len

        doc_count += B
        print(f"  [{split}] Assigned {doc_count} docs so far...")

        if args.max_docs is not None and doc_count >= args.max_docs:
            break

    print(f"[{split}] Streaming complete. Total documents processed: {doc_count}")

    # --------------------------------------------------------------
    # Per-combo: compute stats, save data, generate plots
    # --------------------------------------------------------------
    for combo in combos:
        variant, w, K, seed = combo
        trajs = all_trajectories[combo]

        if not trajs:
            print(f"  WARNING: no trajectories for {combo}, skipping.")
            continue

        trans_matrix = compute_transition_matrix(trajs, K)
        dwell_stats = compute_dwell_times(trajs, K)
        usage = compute_cluster_usage(trajs, K)
        entropy_stats = compute_trajectory_entropy(trajs, K)

        combo_dir = output_root / f"w{w}" / f"K{K}" / f"seed{seed}" / "analysis"
        combo_dir.mkdir(parents=True, exist_ok=True)

        np.save(combo_dir / "transition_matrix.npy", trans_matrix)
        np.save(combo_dir / "cluster_usage.npy", usage)

        traj_dict = {f"doc_{i}": t for i, t in enumerate(trajs)}
        cos_dict = {f"cos_{i}": c for i, c in enumerate(all_cos[combo])}
        margin_dict = {f"margin_{i}": m for i, m in enumerate(all_margins[combo])}
        np.savez(
            combo_dir / "trajectories.npz",
            **traj_dict,
            **cos_dict,
            **margin_dict,
        )

        # ----- Human-readable JSON companions -----

        trans_readable = {
            "description": "Row-normalized transition matrix P[i][j] = P(next=j | current=i)",
            "K": K,
            "matrix": {
                str(i): {str(j): round(float(trans_matrix[i, j]), 6) for j in range(K)}
                for i in range(K)
            },
        }
        with open(combo_dir / "transition_matrix.json", "w") as f:
            json.dump(trans_readable, f, indent=2)

        total_tokens = sum(len(t) for t in trajs)
        usage_readable = {
            "description": "Fraction (and count) of tokens assigned to each cluster",
            "K": K,
            "total_tokens": total_tokens,
            "clusters": {
                str(k): {
                    "fraction": round(float(usage[k]), 6),
                    "count": int(round(usage[k] * total_tokens)),
                }
                for k in range(K)
            },
        }
        with open(combo_dir / "cluster_usage.json", "w") as f:
            json.dump(usage_readable, f, indent=2)

        dwell_readable = {
            "description": "Dwell time statistics (run-lengths of consecutive same-cluster tokens)",
            "K": K,
            "overall": {
                "mean": round(float(dwell_stats["all_dwells"].mean()), 3)
                if len(dwell_stats["all_dwells"]) > 0 else None,
                "median": round(float(np.median(dwell_stats["all_dwells"])), 3)
                if len(dwell_stats["all_dwells"]) > 0 else None,
                "max": int(dwell_stats["all_dwells"].max())
                if len(dwell_stats["all_dwells"]) > 0 else None,
                "total_runs": len(dwell_stats["all_dwells"]),
            },
            "per_cluster": {
                str(k): {
                    "mean": round(float(dwell_stats["per_cluster_dwells"][k].mean()), 3)
                    if len(dwell_stats["per_cluster_dwells"][k]) > 0 else None,
                    "median": round(float(np.median(dwell_stats["per_cluster_dwells"][k])), 3)
                    if len(dwell_stats["per_cluster_dwells"][k]) > 0 else None,
                    "max": int(dwell_stats["per_cluster_dwells"][k].max())
                    if len(dwell_stats["per_cluster_dwells"][k]) > 0 else None,
                    "num_runs": len(dwell_stats["per_cluster_dwells"][k]),
                }
                for k in range(K)
            },
        }
        with open(combo_dir / "dwell_times.json", "w") as f:
            json.dump(dwell_readable, f, indent=2)

        with open(combo_dir / "trajectories.jsonl", "w") as f:
            for doc_i, traj_arr in enumerate(trajs):
                doc_entry = {
                    "doc": doc_i,
                    "length": len(traj_arr),
                    "labels": traj_arr.tolist(),
                    "top1_cos": [round(float(v), 4) for v in all_cos[combo][doc_i]]
                    if doc_i < len(all_cos[combo]) else [],
                    "margin": [round(float(v), 4) for v in all_margins[combo][doc_i]]
                    if doc_i < len(all_margins[combo]) else [],
                }
                f.write(json.dumps(doc_entry) + "\n")

        all_margins_flat = np.concatenate(all_margins[combo]) if all_margins[combo] else np.array([])

        results = {
            "variant": variant,
            "window_size": w,
            "K": K,
            "seed": seed,
            "split": split,
            "num_documents": len(trajs),
            "num_assigned_tokens": total_tokens,
            **entropy_stats,
            "mean_dwell_time": float(dwell_stats["all_dwells"].mean())
            if len(dwell_stats["all_dwells"]) > 0
            else 0,
            "median_dwell_time": float(np.median(dwell_stats["all_dwells"]))
            if len(dwell_stats["all_dwells"]) > 0
            else 0,
            "mean_cosine_margin": float(all_margins_flat.mean())
            if len(all_margins_flat) > 0
            else 0,
        }
        with open(combo_dir / "analysis_results.json", "w") as f:
            json.dump(results, f, indent=2)

        tag = f"{variant} W={w} K={K} seed={seed} [{split}]"
        print(f"  Generating plots for {tag} ...")

        plot_transition_heatmap(
            trans_matrix,
            combo_dir / "transition_heatmap.png",
            title=tag,
        )
        plot_dwell_histogram(
            dwell_stats,
            combo_dir / "dwell_histogram.png",
            title=tag,
        )
        plot_timeline_strips(
            trajs,
            K,
            combo_dir / "timeline_strips.png",
            num_docs=num_timeline_docs,
            title=tag,
        )
        plot_cluster_usage(
            usage,
            combo_dir / "cluster_usage.png",
            title=tag,
        )
        plot_cosine_margin_histogram(
            all_margins_flat,
            combo_dir / "cosine_margin.png",
            title=tag,
        )
        plot_center_similarity(
            kmeans_models[combo].cluster_centers_,
            combo_dir / "center_similarity.png",
            title=tag,
        )

        # ----- State machine diagram -----
        # all_group_labels and all_sm_kwargs track per-alpha results for test HTML rendering
        all_group_labels: dict[float, np.ndarray] = {}
        all_sm_kwargs: dict[float, dict] = {}
        tau_dir = None  # will be set inside the block below

        if not args.skip_state_machine:
            # --- tau-dependent level (alpha-independent) ---
            tau_tag = f"weighted_tau{args.tau}"
            if args.K_coarse is not None:
                tau_tag += f"_Kc{args.K_coarse}"
            tau_dir = combo_dir / tau_tag
            tau_dir.mkdir(parents=True, exist_ok=True)

            print(f"  Generating state machine diagrams for {tag} -> {tau_dir}")
            combo_margins = all_margins[combo]

            # A: Margin-weighted transitions + pruning
            P_weighted, raw_counts = compute_transition_matrix_weighted(
                trajs, combo_margins, K,
                tau=args.tau, pseudocount=args.pseudocount,
            )
            P_pruned = prune_transition_matrix(
                P_weighted, top_r=args.prune_top_r,
                min_prob=args.prune_min_prob,
            )

            # Micro exit matrix: P_exit_micro[i,j] = P[i,j] / (1 - P[i,i]) for j!=i
            P_exit_micro = np.zeros_like(P_weighted)
            for i in range(K):
                off_diag = 1.0 - P_weighted[i, i]
                if off_diag > 1e-12:
                    for j in range(K):
                        if j != i:
                            P_exit_micro[i, j] = P_weighted[i, j] / off_diag

            # Save tau-level artifacts
            np.save(tau_dir / "transition_matrix_weighted.npy", P_weighted)
            np.save(tau_dir / "transition_matrix_pruned.npy", P_pruned)
            np.save(tau_dir / "transition_matrix_micro_exit.npy", P_exit_micro)
            save_matrix_csv(P_weighted, tau_dir / "transition_matrix_weighted.csv",
                             row_prefix="c", col_prefix="c")
            save_matrix_csv(P_exit_micro, tau_dir / "transition_matrix_micro_exit.csv",
                             row_prefix="c", col_prefix="c")

            plot_transition_heatmap(
                P_weighted,
                tau_dir / "transition_heatmap_weighted.png",
                title=f"{tag} (margin-weighted)",
            )
            plot_state_machine_micro(
                P_pruned, usage,
                tau_dir / "state_machine_micro.png",
                title=f"{tag} — State Machine (micro)",
            )

            # --- no_grouping: naming ---
            no_grp_dir = tau_dir / "no_grouping"
            no_grp_naming_dir = no_grp_dir / "cluster_naming"
            no_grp_naming_dir.mkdir(parents=True, exist_ok=True)
            write_cluster_naming_input(
                tokens_per_doc=all_tokens[:len(trajs)],
                trajs_per_doc=trajs,
                K=K,
                usage=usage,
                dwell_stats=dwell_stats,
                out_path=no_grp_naming_dir / "cluster_naming_input.txt",
                variant=variant,
                window_size=w,
                seed=seed,
                num_docs=len(trajs),
                cos_per_doc=all_cos[combo],
                margin_per_doc=all_margins[combo],
            )

            # --- Loop over alpha values ---
            centers = kmeans_models[combo].cluster_centers_

            # Load reference labels once (shared across all alpha values)
            reference_labels = None
            if args.naming_reference_file:
                reference_labels = {}
                with open(args.naming_reference_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if ":" in line:
                            key, name = line.split(":", 1)
                            reference_labels[key.strip()] = name.strip()

            for alpha in args.alpha:
                grp_dir = tau_dir / f"grouped_alpha{alpha}"
                grp_dir.mkdir(parents=True, exist_ok=True)

                # B: Hierarchy
                group_labels, linkage_mat, K_coarse_used = compute_hybrid_hierarchy(
                    centers, P_weighted,
                    alpha=alpha,
                    K_coarse=args.K_coarse,
                    linkage_method=args.linkage_method,
                )
                all_group_labels[alpha] = group_labels

                np.save(grp_dir / "group_labels.npy", group_labels)
                hierarchy_info = {
                    "alpha": alpha,
                    "tau": args.tau,
                    "pseudocount": args.pseudocount,
                    "prune_top_r": args.prune_top_r,
                    "prune_min_prob": args.prune_min_prob,
                    "K_coarse": K_coarse_used,
                    "linkage_method": args.linkage_method,
                    "groups": {
                        str(g): [int(i) for i in range(K) if group_labels[i] == g]
                        for g in range(K_coarse_used)
                    },
                }
                with open(grp_dir / "hierarchy.json", "w") as f:
                    json.dump(hierarchy_info, f, indent=2)

                plot_dendrogram(
                    linkage_mat, K_coarse_used,
                    grp_dir / "dendrogram.png",
                    title=f"{tag} — Dendrogram (alpha={alpha})",
                )

                # C: Macro/micro state machine
                P_macro = compute_macro_transition_matrix(
                    raw_counts, group_labels, K_coarse_used,
                    pseudocount=args.pseudocount,
                )
                np.save(grp_dir / "transition_matrix_macro.npy", P_macro)
                save_matrix_csv(P_macro, grp_dir / "transition_matrix_macro.csv",
                                 row_prefix="g", col_prefix="g")

                # Exit matrix: P(j | leave i) = P[i,j] / (1 - P[i,i]) for j != i
                P_exit = np.zeros_like(P_macro)
                for g in range(K_coarse_used):
                    off_diag_sum = 1.0 - P_macro[g, g]
                    if off_diag_sum > 1e-12:
                        for j in range(K_coarse_used):
                            if j != g:
                                P_exit[g, j] = P_macro[g, j] / off_diag_sum
                save_matrix_csv(P_exit, grp_dir / "transition_matrix_macro_exit.csv",
                                 row_prefix="g", col_prefix="g")

                plot_state_machine_hierarchical(
                    P_pruned, P_macro, group_labels, usage, K_coarse_used,
                    grp_dir / "state_machine_hierarchical.png",
                    title=f"{tag} — Hierarchical State Machine (alpha={alpha})",
                    macro_top_r=args.macro_top_r,
                    macro_min_prob=args.macro_min_prob,
                )

                plot_state_machine_exit(
                    P_exit_micro, P_exit, group_labels, usage, K_coarse_used,
                    grp_dir / "state_machine_exit.png",
                    title=f"{tag} — Exit Probabilities (alpha={alpha})",
                    micro_top_r=args.prune_top_r,
                    micro_min_prob=args.prune_min_prob,
                    macro_top_r=args.macro_top_r,
                    macro_min_prob=args.macro_min_prob,
                    P_self_loop=np.diag(P_weighted),
                )

                # Group naming
                naming_dir = grp_dir / "cluster_naming"
                write_cluster_group_naming_v2(
                    tokens_per_doc=all_tokens[:len(trajs)],
                    trajs_per_doc=trajs,
                    K=K,
                    usage=usage,
                    dwell_stats=dwell_stats,
                    out_dir=naming_dir,
                    variant=variant,
                    window_size=w,
                    seed=seed,
                    num_docs=len(trajs),
                    group_labels=group_labels,
                    K_coarse=K_coarse_used,
                    cos_per_doc=all_cos[combo],
                    margin_per_doc=all_margins[combo],
                    context_tokens=args.naming_context_tokens,
                    proto_count=args.naming_proto_count,
                    typical_count=args.naming_typical_count,
                    reference_labels=reference_labels,
                )

                # Store sm_kwargs for test HTML rendering
                all_sm_kwargs[alpha] = dict(
                    P_exit_micro=P_exit_micro,
                    P_exit=P_exit,
                    usage=usage,
                    K_coarse=K_coarse_used,
                    P_self_loop=np.diag(P_weighted),
                    micro_top_r=args.prune_top_r,
                    micro_min_prob=args.prune_min_prob,
                    macro_top_r=args.macro_top_r,
                    macro_min_prob=args.macro_min_prob,
                )

                # MDS scatter of cluster centers colored by group
                plot_center_scatter_2d(
                    centers,
                    grp_dir / "center_scatter_2d.png",
                    group_labels=group_labels,
                    cluster_names=reference_labels if reference_labels else None,
                    title=f"{tag} — Groups (alpha={alpha})",
                )

                print(f"    alpha={alpha}: grouped artifacts saved to {grp_dir}")

            print(f"  State machine diagrams saved to {tau_dir}")

        print(
            f"  Done: {combo_dir}  "
            f"(n_docs={len(trajs)}, n_tokens={results['num_assigned_tokens']})"
        )

    # ------------------------------------------------------------------
    # 11. Second pass: test-split colored text HTML examples
    # ------------------------------------------------------------------
    if not args.skip_state_machine:
        num_test_docs = args.num_test_docs if args.num_test_docs is not None else num_timeline_docs
        print(f"\n--- Second pass: generating {num_test_docs} test-split colored text examples ---")

        test_loader = build_dataloader(config, split="test", batch_size=args.batch_size, seed=args.seed, shuffle=False, drop_last=False)

        test_tokens: list[list[str]] = []
        test_token_ids: list[list[int]] = []
        test_trajs: dict[tuple[str, int, int, int], list[np.ndarray]] = {
            combo: [] for combo in combos
        }
        test_cos: dict[tuple[str, int, int, int], list[np.ndarray]] = {
            combo: [] for combo in combos
        }
        test_margins: dict[tuple[str, int, int, int], list[np.ndarray]] = {
            combo: [] for combo in combos
        }
        test_features: dict[tuple[str, int], list[np.ndarray]] = {
            vw: [] for vw in preprocessors
        }

        test_doc_count = 0
        for batch in test_loader:
            features = extract_features(
                batch, extractor, encoder, need_h, need_z, device
            )
            mask = batch.get("attention_mask")
            B = batch["input_ids"].shape[0]

            for b in range(B):
                ids_b = batch["input_ids"][b]
                if mask is not None:
                    valid = mask[b].bool()
                    ids_valid = ids_b[valid].tolist()
                else:
                    ids_valid = ids_b.tolist()
                test_tokens.append([tokenizer.decode([tid]) for tid in ids_valid])
                test_token_ids.append(ids_valid)

            for variant in unique_variants:
                raw_key = "z" if variant.startswith("z") else "h"
                raw_feat = features[raw_key]

                if variant == "sae":
                    raw_feat = encode_sae(sae_model, raw_feat)
                if variant in ("h_pca", "sae"):
                    raw_feat = apply_pca_projection(raw_feat, pca_components[variant], pca_mean[variant])

                for w in unique_windows:
                    if (variant, w) not in preprocessors:
                        continue
                    windowed, doc_lens = apply_window_and_stride(raw_feat, mask, w, args.stride)
                    if windowed.shape[0] == 0:
                        continue
                    preprocessed = preprocessors[(variant, w)].transform(windowed)

                    # Retain per-doc features for test HTMLs
                    feat_offset = 0
                    for doc_len in doc_lens:
                        if len(test_features[(variant, w)]) < num_test_docs:
                            test_features[(variant, w)].append(
                                preprocessed[feat_offset : feat_offset + doc_len]
                            )
                        feat_offset += doc_len

                    for combo in combos:
                        if combo[0] != variant or combo[1] != w:
                            continue
                        _, _, K, seed = combo
                        km = kmeans_models[combo]
                        _pp = preprocessors[(variant, w)]
                        labels, top1_cos, margin = predict_with_scores(km, preprocessed, raw=_pp.raw or _pp.centered_only)

                        offset = 0
                        for doc_len in doc_lens:
                            test_trajs[combo].append(labels[offset : offset + doc_len])
                            test_cos[combo].append(top1_cos[offset : offset + doc_len])
                            test_margins[combo].append(margin[offset : offset + doc_len])
                            offset += doc_len

            test_doc_count += B
            if test_doc_count >= num_test_docs:
                break

        print(f"  Test pass complete: {test_doc_count} docs")

        # Generate test colored text HTMLs per combo
        for combo in combos:
            variant, w, K, seed = combo
            t_trajs = test_trajs[combo]
            if not t_trajs:
                continue

            tag = f"{variant} W={w} K={K} seed={seed}"
            combo_dir_test = output_root / f"w{w}" / f"K{K}" / f"seed{seed}" / "analysis"
            tau_tag = f"weighted_tau{args.tau}"
            if args.K_coarse is not None:
                tau_tag += f"_Kc{args.K_coarse}"
            _tau_dir = combo_dir_test / tau_tag

            vw_key = (variant, w)
            centers = kmeans_models[combo].cluster_centers_

            # --- no_grouping test HTMLs ---
            no_grp_test_dir = _tau_dir / "no_grouping" / "example_test_trajectories"
            no_grp_test_dir.mkdir(parents=True, exist_ok=True)
            if vw_key in test_features:
                for doc_idx in range(min(num_test_docs, len(t_trajs))):
                    if doc_idx < len(test_features[vw_key]) and doc_idx < len(test_tokens):
                        render_interactive_trajectory_html(
                            tokens=test_tokens[doc_idx],
                            labels=t_trajs[doc_idx],
                            features=test_features[vw_key][doc_idx],
                            K=K,
                            centers=centers,
                            out_path=no_grp_test_dir / f"interactive_doc_{doc_idx}.html",
                            title=f"{tag} — test doc {doc_idx} (no grouping)",
                            stride=args.stride,
                            window_size=w,
                            top1_cos=test_cos[combo][doc_idx] if doc_idx < len(test_cos[combo]) else None,
                            margin=test_margins[combo][doc_idx] if doc_idx < len(test_margins[combo]) else None,
                            group_labels=None,
                            cluster_names=None,
                            group_names=None,
                            token_ids=test_token_ids[doc_idx],
                            tokenizer=tokenizer,
                        )
            print(f"  No-grouping test HTMLs saved to {no_grp_test_dir}")

            # --- grouped test HTMLs (per alpha) ---
            for alpha in args.alpha:
                grp_test_dir = _tau_dir / f"grouped_alpha{alpha}" / "example_test_trajectories"
                grp_test_dir.mkdir(parents=True, exist_ok=True)

                # Load group_labels from disk (saved during state machine pass)
                _grp_dir = _tau_dir / f"grouped_alpha{alpha}"
                gl_path = _grp_dir / "group_labels.npy"
                test_gl = np.load(gl_path) if gl_path.exists() else None

                # Load state machine data for interactive HTML
                _test_sm_kwargs = {}
                if test_gl is not None:
                    _pe_path = _tau_dir / "transition_matrix_micro_exit.npy"
                    _pw_path = _tau_dir / "transition_matrix_weighted.npy"
                    _pm_path = _grp_dir / "transition_matrix_macro.npy"
                    _usage_path = combo_dir_test / "cluster_usage.npy"
                    if _pe_path.exists() and _pw_path.exists() and _pm_path.exists() and _usage_path.exists():
                        _P_exit_micro = np.load(_pe_path)
                        _P_weighted = np.load(_pw_path)
                        _P_macro = np.load(_pm_path)
                        _usage = np.load(_usage_path)
                        _K_coarse_test = int(test_gl.max()) + 1
                        _P_exit_test = np.zeros_like(_P_macro)
                        for _g in range(_K_coarse_test):
                            _off = 1.0 - _P_macro[_g, _g]
                            if _off > 1e-12:
                                for _j in range(_K_coarse_test):
                                    if _j != _g:
                                        _P_exit_test[_g, _j] = _P_macro[_g, _j] / _off
                        _test_sm_kwargs = dict(
                            P_exit_micro=_P_exit_micro,
                            P_exit=_P_exit_test,
                            usage=_usage,
                            K_coarse=_K_coarse_test,
                            P_self_loop=np.diag(_P_weighted),
                            micro_top_r=args.prune_top_r,
                            micro_min_prob=args.prune_min_prob,
                            macro_top_r=args.macro_top_r,
                            macro_min_prob=args.macro_min_prob,
                        )

                # Auto-detect naming file for this alpha
                _test_cn, _test_gn = None, None
                _test_naming_dir = _grp_dir / "cluster_naming"
                if _test_naming_dir.exists():
                    _tn_cands = [
                        f for f in _test_naming_dir.iterdir()
                        if f.suffix == ".txt" and f.name != "group_cluster_naming.txt"
                    ]
                    if _tn_cands:
                        _test_cn, _test_gn = _parse_naming_file(_tn_cands[0])

                if vw_key in test_features:
                    for doc_idx in range(min(num_test_docs, len(t_trajs))):
                        if doc_idx < len(test_features[vw_key]) and doc_idx < len(test_tokens):
                            render_interactive_trajectory_html(
                                tokens=test_tokens[doc_idx],
                                labels=t_trajs[doc_idx],
                                features=test_features[vw_key][doc_idx],
                                K=K,
                                centers=centers,
                                out_path=grp_test_dir / f"interactive_doc_{doc_idx}.html",
                                title=f"{tag} — test doc {doc_idx} (alpha={alpha})",
                                stride=args.stride,
                                window_size=w,
                                top1_cos=test_cos[combo][doc_idx] if doc_idx < len(test_cos[combo]) else None,
                                margin=test_margins[combo][doc_idx] if doc_idx < len(test_margins[combo]) else None,
                                group_labels=test_gl,
                                cluster_names=_test_cn,
                                group_names=_test_gn,
                                token_ids=test_token_ids[doc_idx],
                                tokenizer=tokenizer,
                                **_test_sm_kwargs,
                            )
                print(f"  Grouped (alpha={alpha}) test HTMLs saved to {grp_test_dir}")

    # ------------------------------------------------------------------
    # 12. Cross-combo: elbow plots per (variant, w)
    # ------------------------------------------------------------------
    from evaluation.clustering.plotting.plots import plot_elbow

    for variant in unique_variants:
        for w in unique_windows:
            inertias_by_K: dict[int, list[float]] = {}
            for K in unique_Ks:
                inertias: list[float] = []
                for seed in unique_seeds:
                    mf = centers_dir / f"w{w}" / f"K{K}" / f"seed{seed}" / "metrics.json"
                    if mf.exists():
                        try:
                            with open(mf) as f:
                                metrics = json.load(f)
                            inertias.append(float(metrics.get("inertia", 0)))
                        except (json.JSONDecodeError, OSError, ValueError):
                            pass
                if inertias:
                    inertias_by_K[K] = inertias

            if inertias_by_K:
                elbow_dir = output_root / f"w{w}"
                elbow_dir.mkdir(parents=True, exist_ok=True)
                plot_elbow(
                    inertias_by_K,
                    elbow_dir / "elbow_K.png",
                    title=f"{variant} W={w}",
                )
                print(f"  Elbow plot saved: {elbow_dir / 'elbow_K.png'}")

    # ------------------------------------------------------------------
    # 12. Cleanup
    # ------------------------------------------------------------------
    if extractor is not None:
        extractor.cleanup()
    print("Done.")


if __name__ == "__main__":
    main()
