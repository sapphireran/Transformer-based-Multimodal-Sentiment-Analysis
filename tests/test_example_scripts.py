"""CLI entry points used in examples/README.md should exit 0."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.gmtm_forward_pass import main as gmtm_main
from examples.metrics_walkthrough import main as metrics_main
from examples.model_inventory import main as inventory_main
from examples.summarize_recorded_results import main as summarize_main


def test_summarize_results(capsys):
    assert summarize_main([]) == 0
    out = capsys.readouterr().out
    assert "printed 8 tables" in out
    assert "TransformerLate" in out


def test_metrics_walkthrough(capsys):
    assert metrics_main() == 0
    assert "summarize_predictions" in capsys.readouterr().out


def test_gmtm_cli_text_only(capsys):
    assert gmtm_main(["--ablate", "text", "--batch-size", "2", "--seed", "0"]) == 0
    assert "keep=['text']" in capsys.readouterr().out


def test_model_inventory(capsys):
    assert inventory_main([]) == 0
    out = capsys.readouterr().out
    assert "GMTM BERT" in out
    assert "TensorFusion" in out
