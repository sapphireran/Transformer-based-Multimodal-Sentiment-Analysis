"""Repository path helpers so examples work from any current directory."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "model"
EXAMPLES_DIR = REPO_ROOT / "examples"
DOCS_DIR = REPO_ROOT / "docs"

# Canonical copies used by summarize_results.py. MOSI tables live only under mosi_test/.
RESULT_CSV_PATHS = {
    "mosei_bert_fusion": MODEL_DIR / "results" / "main_results.csv",
    "mosei_glove_fusion": MODEL_DIR / "results" / "glove_results.csv",
    "mosei_bert_gmtm": MODEL_DIR / "results" / "ablation_results.csv",
    "mosei_glove_gmtm": MODEL_DIR / "results" / "ablation_glove_results.csv",
    "mosi_bert_fusion": MODEL_DIR / "mosi_test" / "mosi_bert_results.csv",
    "mosi_glove_fusion": MODEL_DIR / "mosi_test" / "mosi_glove_results.csv",
    "mosi_bert_gmtm": MODEL_DIR / "mosi_test" / "ablation_mosi_results.csv",
    "mosi_glove_gmtm": MODEL_DIR / "mosi_test" / "ablation_mosi_glove_results.csv",
}

FEATURE_WIDTHS = {
    "bert": {"visual": 35, "audio": 74, "text": 768},
    "glove": {"visual": 35, "audio": 74, "text": 300},
}

PADDED_SEQ_LEN = 50
SENTIMENT_LOW = -3.0
SENTIMENT_HIGH = 3.0
