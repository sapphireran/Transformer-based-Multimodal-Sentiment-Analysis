"""Put ``model/`` on ``sys.path`` so examples can ``import models``."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "model"
RESULTS_DIR = MODEL_DIR / "results"
MOSI_TEST_DIR = MODEL_DIR / "mosi_test"

if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))
