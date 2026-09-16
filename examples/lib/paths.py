"""Filesystem anchors so examples work from any cwd."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "model"
RESULTS_DIR = MODEL_DIR / "results"
MOSI_TEST_DIR = MODEL_DIR / "mosi_test"
EXAMPLES_DIR = REPO_ROOT / "examples"
DOCS_DIR = REPO_ROOT / "docs"


def ensure_model_on_path() -> Path:
    """Put `model/` on sys.path so `import models` matches the training scripts."""
    root = str(MODEL_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)
    return MODEL_DIR
