"""Helpers for personal MOSI/MOSEI examples (synthetic batches, metrics, CSVs)."""

from .metrics import (
    binary_acc_f1,
    regression_metrics,
    split_uniform_5,
    split_uniform_7,
    summarize_predictions,
)
from .paths import MODEL_DIR, REPO_ROOT, RESULTS_DIR, ensure_model_on_path
from .results_tables import load_result_csv, mark_best_rows, repo_result_tables
from .synthetic import FeatureSpec, SyntheticBatch, ablate_modalities, make_batch

__all__ = [
    "FeatureSpec",
    "MODEL_DIR",
    "REPO_ROOT",
    "RESULTS_DIR",
    "SyntheticBatch",
    "ablate_modalities",
    "binary_acc_f1",
    "ensure_model_on_path",
    "load_result_csv",
    "make_batch",
    "mark_best_rows",
    "regression_metrics",
    "repo_result_tables",
    "split_uniform_5",
    "split_uniform_7",
    "summarize_predictions",
]
