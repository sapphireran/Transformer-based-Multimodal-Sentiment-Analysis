"""Put ``examples/`` and ``model/`` on ``sys.path`` for every test."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "examples", ROOT / "model"):
    text = str(extra)
    if text not in sys.path:
        sys.path.insert(0, text)
