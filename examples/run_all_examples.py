"""Run every personal example in order.

    python examples/run_all_examples.py
"""

from __future__ import annotations

import sys
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parent
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))


def main() -> None:
    from fusion_forward import run_all
    from gmtm_forward import run_suite
    from metrics_demo import run as run_metrics
    from summarize_results import run as run_summarize

    print("=" * 72)
    print("1 / 4  fusion_forward")
    print("=" * 72)
    run_all()

    print()
    print("=" * 72)
    print("2 / 4  gmtm_forward")
    print("=" * 72)
    run_suite()

    print()
    print("=" * 72)
    print("3 / 4  metrics_demo")
    print("=" * 72)
    run_metrics()

    print()
    print("=" * 72)
    print("4 / 4  summarize_results")
    print("=" * 72)
    run_summarize()

    print()
    print("All personal examples finished.")


if __name__ == "__main__":
    main()
