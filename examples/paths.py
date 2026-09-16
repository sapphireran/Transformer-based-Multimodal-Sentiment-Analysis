"""Repo-relative paths used by every example script."""

from __future__ import annotations

import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLES_DIR.parent
MODEL_DIR = REPO_ROOT / "model"
RESULTS_DIR = MODEL_DIR / "results"
MOSI_RESULTS_DIR = MODEL_DIR / "mosi_test"
DOCS_DIR = REPO_ROOT / "docs"
OUTPUT_DIR = EXAMPLES_DIR / "output"
CONFIG_DIR = EXAMPLES_DIR / "configs"


def ensure_model_on_path() -> Path:
    """Make ``import models`` and ``import train_and_test`` work like the scripts."""
    model = str(MODEL_DIR)
    if model not in sys.path:
        sys.path.insert(0, model)
    data_dir = str(MODEL_DIR / "data")
    if data_dir not in sys.path:
        sys.path.insert(0, data_dir)
    return MODEL_DIR


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR
