"""Make ``model/`` and ``examples/`` importable the same way the walkthroughs are."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
MODEL = ROOT / "model"

for path in (EXAMPLES, MODEL):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)
