"""Put the repo root and ``model/`` on ``sys.path``.

The training scripts assume they are launched from ``model/``. The
examples live one directory up and import both ``examples.*`` and
``models``. Call :func:`ensure_import_path` before those imports when
a file is executed as a script (``python examples/run_gmtm_demo.py``).
``python -m examples.run_gmtm_demo`` from the repo root also works
once this helper has run.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "model"


def ensure_import_path() -> Path:
    for path in (ROOT, MODEL):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    return ROOT
