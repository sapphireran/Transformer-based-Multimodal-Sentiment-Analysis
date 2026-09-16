#!/usr/bin/env python3
"""Run every example script and fail if any of them raises."""

from __future__ import annotations

import json
import traceback

import evaluate_metrics
import fusion_forward
import gmtm_forward
import gmtm_toy_train
import packed_vs_padded
import summarize_results


def main() -> int:
    jobs = [
        ("fusion_forward", fusion_forward.run),
        ("gmtm_forward", gmtm_forward.run),
        ("gmtm_toy_train", gmtm_toy_train.run),
        ("evaluate_metrics", evaluate_metrics.run),
        ("packed_vs_padded", packed_vs_padded.run),
        ("summarize_results", summarize_results.run),
    ]
    results = []
    failed = False
    for name, fn in jobs:
        try:
            payload = fn()
            results.append({"name": name, "ok": True, "payload": payload})
            print(f"[ok] {name}")
        except Exception as exc:  # noqa: BLE001 — want the full example traceback
            failed = True
            results.append({"name": name, "ok": False, "error": str(exc)})
            print(f"[fail] {name}: {exc}")
            traceback.print_exc()
    print(json.dumps({"failed": failed, "n": len(results)}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
