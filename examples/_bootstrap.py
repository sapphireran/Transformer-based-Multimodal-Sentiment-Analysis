"""Put the repo root and ``model/`` on ``sys.path``.

Import this module first from every example script:

    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from examples._bootstrap import ensure_output_dir
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "model"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

for path in (REPO_ROOT, MODEL_DIR):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR
