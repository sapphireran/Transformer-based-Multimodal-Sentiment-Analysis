"""Load the published CSVs and normalize fusion / ablation names."""

from __future__ import annotations

import ast
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

try:
    from .paths import MOSI_RESULTS_DIR, RESULTS_DIR
except ImportError:  # running as a script next to this file
    from paths import MOSI_RESULTS_DIR, RESULTS_DIR

METRIC_KEYS = ("MAE", "ACC7", "ACC5", "ACC2", "Corr", "F1")


@dataclass(frozen=True)
class ResultRow:
    setting: str
    method: str
    mae: float
    acc7: float
    acc5: float
    acc2: float
    corr: float
    f1: float

    def as_dict(self) -> dict:
        return {
            "setting": self.setting,
            "method": self.method,
            "MAE": self.mae,
            "ACC7": self.acc7,
            "ACC5": self.acc5,
            "ACC2": self.acc2,
            "Corr": self.corr,
            "F1": self.f1,
        }


def normalize_method(raw: str) -> str:
    """Turn ``\"['text', 'audio']\"`` or ``Concat`` into a stable label."""
    text = raw.strip()
    if text.startswith("[") or (text.startswith('"') and "[" in text):
        try:
            parsed = ast.literal_eval(text.strip('"'))
        except (ValueError, SyntaxError):
            parsed = None
        if isinstance(parsed, (list, tuple)):
            return "+".join(str(x) for x in parsed)
    aliases = {
        "Concat": "ConcatLate",
        "GatedMultiTransfomer": "GMTM",
        "GatedMultiTransformer": "GMTM",
        "LowRankTensorFusion": "LMF",
        "TensorFusion": "TFN",
    }
    return aliases.get(text, text)


def _header_index(header: Sequence[str], *candidates: str) -> int:
    lowered = [h.strip() for h in header]
    for name in candidates:
        if name in lowered:
            return lowered.index(name)
    raise KeyError(f"none of {candidates} in {header}")


def load_csv(path: Path, setting: str) -> List[ResultRow]:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        i_method = 0
        i_mae = _header_index(header, "MAE")
        i_acc7 = _header_index(header, "ACC7", "Acc7")
        i_acc5 = _header_index(header, "ACC5", "Acc5")
        i_acc2 = _header_index(header, "ACC2", "Acc2")
        i_corr = _header_index(header, "Corr")
        i_f1 = _header_index(header, "F1")
        rows = []
        for parts in reader:
            if not parts or not parts[0].strip():
                continue
            rows.append(
                ResultRow(
                    setting=setting,
                    method=normalize_method(parts[i_method]),
                    mae=float(parts[i_mae]),
                    acc7=float(parts[i_acc7]),
                    acc5=float(parts[i_acc5]),
                    acc2=float(parts[i_acc2]),
                    corr=float(parts[i_corr]),
                    f1=float(parts[i_f1]),
                )
            )
        return rows


def published_tables() -> dict[str, List[ResultRow]]:
    mapping = {
        "mosei-bert-fusion": RESULTS_DIR / "main_results.csv",
        "mosei-glove-fusion": RESULTS_DIR / "glove_results.csv",
        "mosei-bert-ablation": RESULTS_DIR / "ablation_results.csv",
        "mosei-glove-ablation": RESULTS_DIR / "ablation_glove_results.csv",
        "mosi-bert-fusion": MOSI_RESULTS_DIR / "mosi_bert_results.csv",
        "mosi-glove-fusion": MOSI_RESULTS_DIR / "mosi_glove_results.csv",
        "mosi-bert-ablation": MOSI_RESULTS_DIR / "ablation_mosi_results.csv",
        "mosi-glove-ablation": MOSI_RESULTS_DIR / "ablation_mosi_glove_results.csv",
    }
    return {name: load_csv(path, name) for name, path in mapping.items()}


def markdown_table(rows: Sequence[ResultRow], title: str | None = None) -> str:
    lines = []
    if title:
        lines.append(f"### {title}")
        lines.append("")
    lines.append("| Method | MAE ↓ | Acc-7 | Acc-5 | Acc-2 | Corr | F1 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            f"| {row.method} | {row.mae:.4f} | {row.acc7:.4f} | {row.acc5:.4f} "
            f"| {row.acc2:.4f} | {row.corr:.4f} | {row.f1:.4f} |"
        )
    return "\n".join(lines)


def best_by_mae(rows: Iterable[ResultRow]) -> ResultRow:
    return min(rows, key=lambda r: r.mae)
