"""Path helpers shared by the example scripts."""

from __future__ import annotations

import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLES_DIR.parent
MODEL_DIR = REPO_ROOT / "model"
RESULTS_DIR = MODEL_DIR / "results"
OUTPUT_DIR = EXAMPLES_DIR / "output"


def add_model_to_path() -> Path:
    """Put ``model/`` on ``sys.path`` so ``import models`` works as in the scripts."""
    model_dir = str(MODEL_DIR)
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    return MODEL_DIR


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR
