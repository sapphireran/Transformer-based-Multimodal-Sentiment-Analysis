"""Run every example script in order. Exit nonzero if any step fails."""

from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parent
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

MODULES = [
    "synthetic_data",
    "packed_vs_padded",
    "encoder_shapes",
    "fusion_walkthrough",
    "metrics_demo",
    "result_tables",
    "gmtm_toy_train",
]


def main() -> int:
    failed = []
    for name in MODULES:
        print("=" * 72)
        print(f"running examples/{name}.py")
        print("=" * 72)
        try:
            mod = importlib.import_module(name)
            mod.main()
        except Exception:
            traceback.print_exc()
            failed.append(name)
        print()
    if failed:
        print("FAILED:", ", ".join(failed))
        return 1
    print("All example scripts finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
