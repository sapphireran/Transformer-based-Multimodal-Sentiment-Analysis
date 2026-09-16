import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
for path in (str(ROOT), str(EXAMPLES)):
    if path not in sys.path:
        sys.path.insert(0, path)
