"""Put ``model/`` and the repo root on ``sys.path`` for every test."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "model"
for path in (ROOT, MODEL):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)
