"""Run every example module's ``main()`` in a stable order."""

from __future__ import annotations

import traceback
from typing import Callable

from examples import (
    ablation_zeroing,
    attention_pooling,
    encoder_stack,
    forward_gmtm,
    fusion_shapes,
    inspect_pickle_schema,
    metrics_walkthrough,
    positional_embeddings,
    results_tables,
    synthetic_data,
    tiny_train_loop,
)

STEPS: list[tuple[str, Callable[[], int]]] = [
    ("synthetic_data", synthetic_data.main),
    ("inspect_pickle_schema", inspect_pickle_schema.main),
    ("metrics_walkthrough", metrics_walkthrough.main),
    ("results_tables", results_tables.main),
    ("positional_embeddings", positional_embeddings.main),
    ("attention_pooling", attention_pooling.main),
    ("encoder_stack", encoder_stack.main),
    ("fusion_shapes", fusion_shapes.main),
    ("forward_gmtm", forward_gmtm.main),
    ("ablation_zeroing", ablation_zeroing.main),
    ("tiny_train_loop", tiny_train_loop.main),
]


def main() -> int:
    failed: list[str] = []
    for name, fn in STEPS:
        print("=" * 72)
        print(name)
        print("=" * 72)
        try:
            code = fn()
        except Exception:
            traceback.print_exc()
            failed.append(name)
            continue
        if code:
            failed.append(name)
        print()
    if failed:
        print("FAILED:", ", ".join(failed))
        return 1
    print(f"All {len(STEPS)} example modules finished with exit code 0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
