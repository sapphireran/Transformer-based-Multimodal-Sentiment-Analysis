"""Run every personal example and the unit checks.

Intended as the one command in the root README:

    python examples/run_all.py
"""

from __future__ import annotations

import sys
import traceback
import unittest


def _run(name: str, fn) -> int:
    print()
    print("=" * 72)
    print(name)
    print("=" * 72)
    try:
        code = fn()
    except Exception:
        traceback.print_exc()
        return 1
    return int(code or 0)


def main() -> int:
    from demo_fusion import main as fusion_main
    from demo_gmtm import main as gmtm_main
    from demo_metrics import main as metrics_main
    from inspect_shapes import main as shapes_main
    from demo_train_toy import main as train_main
    from print_logged_results import main as results_main

    status = 0
    status |= _run("print_logged_results", results_main)
    status |= _run("inspect_shapes", shapes_main)
    status |= _run("demo_fusion", fusion_main)
    status |= _run("demo_gmtm", gmtm_main)
    status |= _run("demo_metrics", metrics_main)
    status |= _run("demo_train_toy", train_main)

    print()
    print("=" * 72)
    print("unit tests")
    print("=" * 72)
    suite = unittest.defaultTestLoader.loadTestsFromName("test_examples")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        status = 1

    print()
    if status == 0:
        print("all personal examples and tests passed")
    else:
        print("one or more example steps failed", file=sys.stderr)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
