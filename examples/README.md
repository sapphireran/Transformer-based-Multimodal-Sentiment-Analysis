# Personal examples

These scripts exercise the research code **without** CMU-MOSI / CMU-MOSEI
pickles or a GPU. They are the companion to [`docs/`](../docs/README.md).

Run them from anywhere; they pin `sys.path` to the repo root.

| Script | Needs | What it shows |
| --- | --- | --- |
| `summarize_results.py` | NumPy / stdlib | Ranked reprint of every checked-in CSV |
| `metric_walkthrough.py` | NumPy | Acc7 edges, MAE / Acc2 hand checks |
| `tiny_affect_dataset.py` | NumPy (torch optional) | Pickle schema + a tiny Dataset / collate |
| `forward_pass_demo.py` | PyTorch | Finite CPU forwards of GMTM and the fusion blocks |
| `fusion_shape_walkthrough.py` | PyTorch | Rank contract for each module |
| `toy_train_gmtm.py` | PyTorch | GMTM overfits a synthetic linear mixture |
| `run_all.py` | the above | One entry point for the suite |

```bash
# no torch
python examples/summarize_results.py
python examples/metric_walkthrough.py
python examples/tiny_affect_dataset.py
python examples/run_all.py --skip-torch

# with torch
pip install -r requirements.txt
python examples/forward_pass_demo.py
python examples/fusion_shape_walkthrough.py
python examples/toy_train_gmtm.py --steps 25
python examples/run_all.py
```

`tiny_affect_dataset.py --write-pkl /tmp/tiny_mosei_bert.pkl` writes a
nested dict with the same keys the real loader expects
(`train|valid|test` → `vision`, `audio`, `text`, `labels`, `id`).

Helpers live in `examples/lib/`:

- `metrics.py` — NumPy port of `single_test` scoring
- `synthetic.py` — BERT / GloVe-shaped batches and the toy mixture
- `repo.py` — paths to the CSV logs
- `model_import.py` — loads `model/models.py` without `cd model`

The original experiment runners under `model/` are unchanged. They still
want real pickles, checkpoints, and `.cuda()`.
