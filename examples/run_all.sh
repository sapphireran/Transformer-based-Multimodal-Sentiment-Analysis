#!/usr/bin/env bash
# Run every CPU walkthrough from the repository root.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 examples/read_result_tables.py
python3 examples/compare_tables.py
python3 examples/metrics_demo.py
python3 examples/fusion_shapes.py
python3 examples/gmtm_forward.py
python3 examples/multiframework_demo.py
python3 examples/dataset_collate_demo.py
python3 examples/ablation_zero_mask.py

echo
echo "All examples finished."
