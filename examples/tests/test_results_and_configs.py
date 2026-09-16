import yaml

from examples.paths import CONFIG_DIR, REPO_ROOT
from examples.results_lib import best_by_mae, normalize_method, published_tables


def test_normalize_method_aliases_and_lists():
    assert normalize_method("Concat") == "ConcatLate"
    assert normalize_method("GatedMultiTransfomer") == "GMTM"
    assert normalize_method("LowRankTensorFusion") == "LMF"
    assert normalize_method("['text', 'audio']") == "text+audio"
    assert normalize_method("['text']") == "text"
    assert normalize_method("TransformerLate") == "TransformerLate"


def test_published_csvs_and_documented_ranking():
    tables = published_tables()
    assert set(tables) >= {
        "mosei-bert-fusion",
        "mosei-glove-fusion",
        "mosei-bert-ablation",
        "mosi-bert-fusion",
    }
    bert = {r.method: r for r in tables["mosei-bert-fusion"]}
    assert best_by_mae(tables["mosei-bert-fusion"]).method == "TransformerLate"
    assert bert["TransformerLate"].mae < bert["LMF"].mae < bert["ConcatEarly"].mae

    ablate = {r.method: r for r in tables["mosei-bert-ablation"]}
    assert ablate["text+audio+visual"].mae <= ablate["text"].mae < ablate["audio"].mae

    mosi_glove_best = best_by_mae(tables["mosi-glove-fusion"])
    assert mosi_glove_best.method == "GMTM"


def test_yaml_configs_parse_and_mention_feature_widths():
    expected = {
        "bert_mosei.yaml": 768,
        "glove_mosei.yaml": 300,
        "gmtm.yaml": 64,
        "mosi_transfer.yaml": 50,  # not used; just ensure file loads
    }
    for name in expected:
        path = CONFIG_DIR / name
        assert path.is_file(), path
        data = yaml.safe_load(path.read_text())
        assert isinstance(data, dict)
        assert "name" in data

    bert = yaml.safe_load((CONFIG_DIR / "bert_mosei.yaml").read_text())
    assert bert["data"]["feature_dims"]["text"] == 768
    assert bert["data"]["feature_dims"]["visual"] == 35
    glove = yaml.safe_load((CONFIG_DIR / "glove_mosei.yaml").read_text())
    assert glove["data"]["feature_dims"]["text"] == 300
    gmtm = yaml.safe_load((CONFIG_DIR / "gmtm.yaml").read_text())
    assert gmtm["hparams"]["embed_dim"] == 64
    assert gmtm["hparams"]["num_heads"] == 4
    assert gmtm["n_modalities"] == 3


def test_docs_index_links_exist():
    docs = REPO_ROOT / "docs"
    listed = [
        "README.md",
        "architecture.md",
        "fusion-methods.md",
        "datasets.md",
        "training.md",
        "evaluation.md",
        "hyperparameters.md",
        "ablation-study.md",
        "results.md",
        "reproducing-experiments.md",
        "repository-map.md",
    ]
    for name in listed:
        assert (docs / name).is_file(), name
    readme = (REPO_ROOT / "README.md").read_text()
    assert "GatedMultiTransfomer" in readme or "GMTM" in readme
    assert "examples/" in readme
