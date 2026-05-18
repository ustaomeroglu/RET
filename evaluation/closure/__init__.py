"""Closure subpackage: predictor builders, training loops, config/data helpers, sweep utilities."""
from evaluation.closure.predictors import (
    build_closure_predictor,
    build_token_projector,
    compute_loss,
    compute_r2,
    compute_r2_streaming,
    compute_per_element_loss,
)
from evaluation.closure.config import (
    ensure_list,
    load_config_with_overlay,
    closure_data_config,
    closure_data_tag,
    closure_prompt_source,
    build_closure_dataloader,
    make_train_iter,
)
from evaluation.closure.training_combined import (
    train_all_variants,
    train_all_variants_model,
    train_all_variants_sae,
)
from evaluation.closure.training import (
    train_closure_predictor,
    train_closure_predictor_with_projector,
)
from evaluation.closure.data import (
    collect_val_data,
    extract_sequence_data,
    encode_in_chunks,
)
from evaluation.closure.io import (
    save_closure_predictors,
    load_closure_predictors,
)
from evaluation.closure.eval import evaluate_per_step

__all__ = [
    # predictors
    "build_closure_predictor",
    "build_token_projector",
    "compute_loss",
    "compute_r2",
    "compute_r2_streaming",
    "compute_per_element_loss",
    # config
    "ensure_list",
    "load_config_with_overlay",
    "closure_data_config",
    "closure_data_tag",
    "closure_prompt_source",
    "build_closure_dataloader",
    "make_train_iter",
    # training (combined)
    "train_all_variants",
    "train_all_variants_model",
    "train_all_variants_sae",
    # training (basic)
    "train_closure_predictor",
    "train_closure_predictor_with_projector",
    # data
    "collect_val_data",
    "extract_sequence_data",
    "encode_in_chunks",
    # io
    "save_closure_predictors",
    "load_closure_predictors",
    # eval
    "evaluate_per_step",
]
