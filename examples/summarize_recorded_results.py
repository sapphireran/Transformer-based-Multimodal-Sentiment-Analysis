#!/usr/bin/env python3
"""Print every committed MOSI/MOSEI result table and mark the best cells.

No GPU, no pickle, no network. Stars (*) mark the winning value in each
metric column (min MAE, max Acc / Corr / F1).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lib.results_tables import format_table, load_result_csv, repo_result_tables


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-duplicates",
        action="store_true",
        help="also print the CSV copies sitting next to the training scripts",
    )
    args = parser.parse_args(argv)

    printed = 0
    for title, path in repo_result_tables(include_duplicates=args.include_duplicates):
        payload = load_result_csv(path)
        print(format_table(title, payload))
        print()
        printed += 1

    print(f"printed {printed} tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
