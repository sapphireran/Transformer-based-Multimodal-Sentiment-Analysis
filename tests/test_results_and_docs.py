from pathlib import Path

from examples.common import DOCS_DIR, REPO_ROOT, RESULTS_DIR
from examples.results_tables import summarize

REQUIRED_DOCS = [
    "architecture.md",
    "datasets.md",
    "metrics.md",
    "experiments.md",
    "training.md",
    "hyperparameters.md",
    "reproduction.md",
    "module_reference.md",
    "repository_map.md",
    "glossary.md",
    "README.md",
]


def test_docs_exist_and_are_substantial():
    for name in REQUIRED_DOCS:
        path = DOCS_DIR / name
        assert path.is_file(), name
        text = path.read_text(encoding="utf-8")
        assert len(text) > 400, f"{name} looks too short"


def test_readme_mentions_gmtm_and_examples():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "Gated Multi-Transformer" in readme or "GMTM" in readme
    assert "examples/" in readme
    assert "0.5640" in readme


def test_results_csv_winners_match_docs():
    winners = {title: row for title, row in summarize()}
    assert abs(float(winners["MOSEI BERT GMTM ablation"]["MAE"]) - 0.5640) < 1e-9
    assert winners["MOSEI BERT baselines"]["Fusion Method"] == "TransformerLate"
    assert winners["MOSEI GloVe baselines"]["Fusion Method"] == "LowRankTensorFusion"
    assert winners["MOSI GloVe transfer"]["Fusion Method"] == "GatedMultiTransfomer"


def test_result_csvs_present():
    for name in (
        "main_results.csv",
        "glove_results.csv",
        "ablation_results.csv",
        "ablation_glove_results.csv",
    ):
        assert (RESULTS_DIR / name).is_file()


def test_examples_readme_lists_runner():
    text = (REPO_ROOT / "examples" / "README.md").read_text(encoding="utf-8")
    assert "python -m examples.run_all" in text
    assert Path(REPO_ROOT / "examples" / "run_all.py").is_file()
