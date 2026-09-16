#!/usr/bin/env python3
"""Compute deltas across the recorded CSVs (BERT vs GloVe, text-only vs full)."""

from __future__ import annotations

import csv
from pathlib import Path

from common import MOSI_RESULTS_DIR, RESULTS_DIR


def _load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _f(row: dict[str, str], key: str) -> float:
    # MOSI BERT uses Acc5; others use ACC5. Acc2 is ACC2 everywhere.
    aliases = {key, key.upper(), key.lower(), "Acc5" if key.upper() == "ACC5" else key}
    for candidate in aliases:
        if candidate in row and row[candidate] != "":
            return float(row[candidate])
    raise KeyError(f"{key} not in {list(row)}")


def _by_name(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    out = {}
    for row in rows:
        name = row.get("Fusion Method") or next(iter(row.values()))
        out[name] = row
    return out


def _delta(a: dict[str, str], b: dict[str, str], metric: str) -> float:
    return _f(a, metric) - _f(b, metric)


def main() -> None:
    bert = _by_name(_load(RESULTS_DIR / "main_results.csv"))
    glove = _by_name(_load(RESULTS_DIR / "glove_results.csv"))
    abl_bert = _by_name(_load(RESULTS_DIR / "ablation_results.csv"))
    abl_glove = _by_name(_load(RESULTS_DIR / "ablation_glove_results.csv"))
    mosi_bert = _by_name(_load(MOSI_RESULTS_DIR / "mosi_bert_results.csv"))

    print("MOSEI baselines: BERT MAE minus GloVe MAE (negative => BERT better)\n")
    print(f"{'fusion':<24} {'BERT MAE':>10} {'GloVe MAE':>10} {'Δ MAE':>10}")
    for name in bert:
        if name not in glove:
            continue
        d = _delta(bert[name], glove[name], "MAE")
        print(f"{name:<24} {_f(bert[name], 'MAE'):10.4f} {_f(glove[name], 'MAE'):10.4f} {d:10.4f}")

    print("\nGMTM ablation (BERT): each row minus text-only\n")
    text = abl_bert["['text']"]
    print(f"{'modalities':<32} {'MAE':>8} {'ΔMAE vs text':>14} {'Corr':>8} {'ΔCorr vs text':>14}")
    for name, row in abl_bert.items():
        print(
            f"{name:<32} {_f(row, 'MAE'):8.4f} {_delta(row, text, 'MAE'):14.4f} "
            f"{_f(row, 'Corr'):8.4f} {_delta(row, text, 'Corr'):14.4f}"
        )

    print("\nGMTM ablation (GloVe): text+audio is the known regression\n")
    g_text = abl_glove["['text']"]
    g_ta = abl_glove["['text', 'audio']"]
    g_full = abl_glove["['text', 'audio', 'visual']"]
    print(
        f"  text MAE { _f(g_text, 'MAE'):.4f}  →  text+audio {_f(g_ta, 'MAE'):.4f}  "
        f"(+{_delta(g_ta, g_text, 'MAE'):.4f})"
    )
    print(
        f"  full trio MAE {_f(g_full, 'MAE'):.4f}  vs text {_delta(g_full, g_text, 'MAE'):+.4f}"
    )

    print("\nIn-domain GMTM full (BERT) vs best BERT baseline (TransformerLate)")
    gmtm = abl_bert["['text', 'audio', 'visual']"]
    late = bert["TransformerLate"]
    print(f"  MAE  { _f(gmtm, 'MAE'):.4f} vs {_f(late, 'MAE'):.4f}  ({_delta(gmtm, late, 'MAE'):+.4f})")
    print(f"  Corr { _f(gmtm, 'Corr'):.4f} vs {_f(late, 'Corr'):.4f}  ({_delta(gmtm, late, 'Corr'):+.4f})")
    print(f"  F1   { _f(gmtm, 'F1'):.4f} vs {_f(late, 'F1'):.4f}  ({_delta(gmtm, late, 'F1'):+.4f})")

    print("\nTransfer gap: same fusion, MOSI-merged MAE minus MOSEI MAE (BERT)")
    print(f"{'fusion':<24} {'MOSEI':>8} {'MOSI':>8} {'gap':>8}")
    name_map = {"ConcatLate": "Concat"}  # MOSI script uses a shorter name
    for name, row in bert.items():
        mosi_name = name_map.get(name, name)
        if mosi_name not in mosi_bert:
            continue
        gap = _f(mosi_bert[mosi_name], "MAE") - _f(row, "MAE")
        print(f"{name:<24} {_f(row, 'MAE'):8.4f} {_f(mosi_bert[mosi_name], 'MAE'):8.4f} {gap:8.4f}")
    if "GatedMultiTransfomer" in mosi_bert:
        gap = _f(mosi_bert["GatedMultiTransfomer"], "MAE") - _f(gmtm, "MAE")
        print(f"{'GMTM full':<24} {_f(gmtm, 'MAE'):8.4f} {_f(mosi_bert['GatedMultiTransfomer'], 'MAE'):8.4f} {gap:8.4f}")

    print(
        "\nTakeaway: BERT beats GloVe on every baseline MAE; GMTM's in-domain"
        " gain over TransformerLate is real but small; MOSI-merged transfer"
        " adds ~0.3–0.4 MAE for every method."
    )


if __name__ == "__main__":
    main()
