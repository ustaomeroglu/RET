"""Streaming clustering analysis for RET representations."""
from evaluation.clustering.core import (
    StreamingPreprocessor,
    find_config_yaml,
    save_matrix_csv,
    apply_pca_projection,
    apply_window_and_stride,
    compute_cluster_usage,
    compute_dwell_times,
    compute_hybrid_hierarchy,
    compute_macro_transition_matrix,
    compute_trajectory_entropy,
    compute_transition_matrix,
    compute_transition_matrix_weighted,
    create_kmeans,
    discover_combos,
    extract_features,
    load_kmeans,
    predict_with_scores,
    prune_transition_matrix,
    save_kmeans,
)
from evaluation.clustering.config import (
    build_dataloader,
    load_config_with_overlay,
)
from evaluation.clustering.plotting.plots import plot_center_scatter_2d
