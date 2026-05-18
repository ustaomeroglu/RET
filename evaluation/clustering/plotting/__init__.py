"""Clustering plotting subpackage.

Functions are split into thematic submodules:
- ``plots``         — basic 2D matplotlib plots (heatmaps, histograms, scatter, …)
- ``state_machine`` — state-machine SVG rendering (micro / hierarchical / exit)
- ``naming``        — cluster-naming I/O (LLM-driven naming pipeline)
- ``html``          — interactive trajectory HTML rendering
- ``colors``        — small color and dim/alpha utilities

The full public API is re-exported here for one-line imports, but new code is
encouraged to import from the specific submodule.
"""
from evaluation.clustering.plotting.plots import (
    plot_transition_heatmap,
    plot_dwell_histogram,
    plot_timeline_strips,
    plot_cluster_usage,
    plot_elbow,
    plot_cosine_margin_histogram,
    plot_center_similarity,
    plot_center_scatter_2d,
    plot_dendrogram,
    plot_umap_trajectories,
)
from evaluation.clustering.plotting.state_machine import (
    plot_state_machine_micro,
    plot_state_machine_hierarchical,
    plot_state_machine_exit,
)
from evaluation.clustering.plotting.naming import (
    write_cluster_naming_input,
    write_cluster_naming_overview,
    write_cluster_naming_details,
    write_cluster_group_naming,
    write_cluster_group_naming_v2,
    _parse_naming_file,
)
from evaluation.clustering.plotting.html import (
    render_colored_text_html,
    render_interactive_trajectory_html,
    repair_utf8_split_tokens,
)
from evaluation.clustering.plotting.colors import (
    _cluster_colors,
    _group_colors,
    _margin_to_alpha,
    _text_color_for_bg,
)
