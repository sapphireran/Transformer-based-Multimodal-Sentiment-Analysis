from summarize_results import TABLES, load_table, run, summarize_table


def test_all_recorded_tables_exist():
    missing = [name for name, path in TABLES.items() if not path.exists()]
    assert missing == []


def test_mosei_bert_fusion_ranks_transformer_late_first():
    rows = load_table(TABLES["mosei_bert_fusion"])
    summary = summarize_table(rows)
    assert summary["n"] == 6
    assert summary["best_mae"]["name"] == "TransformerLate"
    assert summary["best_mae"]["MAE"] == 0.5846
    assert summary["best_acc7"]["name"] == "TransformerLate"


def test_mosei_bert_ablation_text_beats_non_text():
    rows = load_table(TABLES["mosei_bert_gmtm_ablation"])
    by_name = {row["name"]: row for row in rows}
    assert by_name["['text']"]["MAE"] < by_name["['audio']"]["MAE"]
    assert by_name["['text']"]["MAE"] < by_name["['visual']"]["MAE"]


def test_run_covers_every_table():
    report = run()
    assert set(report) == set(TABLES)
    for payload in report.values():
        assert payload["n"] >= 6
        assert "ranking_mae" in payload
