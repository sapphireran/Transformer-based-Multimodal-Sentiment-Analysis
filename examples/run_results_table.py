#!/usr/bin/env python3
"""Print the published CSVs as markdown and emit a ranking summary."""

from __future__ import annotations

from paths import ensure_output_dir
from results_lib import best_by_mae, markdown_table, published_tables

TITLES = {
    "mosei-bert-fusion": "MOSEI + BERT fusion sweep",
    "mosei-glove-fusion": "MOSEI + GloVe fusion sweep",
    "mosei-bert-ablation": "MOSEI + BERT GMTM ablation",
    "mosei-glove-ablation": "MOSEI + GloVe GMTM ablation",
    "mosi-bert-fusion": "MOSI transfer + BERT",
    "mosi-glove-fusion": "MOSI transfer + GloVe",
    "mosi-bert-ablation": "MOSI transfer + BERT GMTM ablation",
    "mosi-glove-ablation": "MOSI transfer + GloVe GMTM ablation",
}


def main() -> int:
    tables = published_tables()
    chunks = ["# Published result tables", ""]
    print("Best MAE per setting")
    print("--------------------")
    for key, rows in tables.items():
        winner = best_by_mae(rows)
        line = f"{key:24s}  {winner.method:24s}  MAE={winner.mae:.4f}"
        print(line)
        chunks.append(markdown_table(rows, TITLES[key]))
        chunks.append("")
        chunks.append(f"_Best MAE: **{winner.method}** ({winner.mae:.4f})_")
        chunks.append("")

    # Cross-check a few documented facts so a CSV edit cannot silently drift.
    bert = {r.method: r for r in tables["mosei-bert-fusion"]}
    assert bert["TransformerLate"].mae < bert["ConcatEarly"].mae
    assert bert["LMF"].mae < bert["ConcatLate"].mae

    gmtm = {r.method: r for r in tables["mosei-bert-ablation"]}
    assert gmtm["text+audio+visual"].mae <= gmtm["text"].mae
    assert gmtm["text"].mae < gmtm["audio"].mae

    out = ensure_output_dir() / "published_tables.md"
    out.write_text("\n".join(chunks))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
