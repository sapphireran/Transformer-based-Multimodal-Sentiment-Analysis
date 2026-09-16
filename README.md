# Transformer-based Multimodal Sentiment Analysis

Personal research code for utterance-level sentiment on **CMU-MOSI** and
**CMU-MOSEI**. The repo compares six classical fusion methods with a gated
cross-modal transformer (`GatedMultiTransfomerModel`, GMTM) under BERT and
GloVe text features.

This is a personal project. There is no product, service, or company
codebase here.

## What is in the box

- Fusion modules and GMTM in [`model/models.py`](model/models.py)
- Train / eval loop in [`model/train_and_test.py`](model/train_and_test.py)
- MOSEI sweeps, GMTM ablations, and MOSI transfer scripts under `model/`
- Checked-in CSV logs of those runs
- Long-form notes in [`docs/`](docs/README.md)
- CPU synthetic examples in [`examples/`](examples/README.md) that do **not**
  need the MOSI / MOSEI pickles

## Quick start (examples, no dataset)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python examples/summarize_results.py      # reprint every CSV
python examples/metric_walkthrough.py     # Acc7 edges + hand checks
python examples/forward_pass_demo.py      # GMTM + fusion forwards on CPU
python examples/toy_train_gmtm.py         # short synthetic overfit
python examples/run_all.py                # whole example suite
```

`summarize_results.py` and `metric_walkthrough.py` only need NumPy. The
forward and training demos need PyTorch.

## Quick start (real MOSEI / MOSI)

1. Place the aligned pickles (not in git):

   - `model/data/MOSEI/mosei_raw_bert.pkl`
   - `model/data/MOSEI/mosei_raw_glove.pkl`
   - `model/data/MOSI/mosi_raw_bert.pkl`
   - `model/data/MOSI/mosi_raw_glove.pkl`

2. `cd model` — every original script resolves data and checkpoints from
   that directory.

3. Uncomment the `train(...)` call in the script you want, then run it.
   Several files are currently in **test-only** mode (`torch.load` a
   checkpoint). See [docs/experiments.md](docs/experiments.md).

The launch scripts call `.cuda()`. `train()` / `test()` themselves will
use CPU if CUDA is missing, but the `.cuda()` lines in the runners will
not.

## Feature widths

| Stream | Width |
| --- | --- |
| Visual (Facet 42) | 35 |
| Audio (COVAREP) | 74 |
| Text BERT | 768 |
| Text GloVe 840B | 300 |
| Time (padded) | 50 |
| Label | scalar ≈ `[-3, 3]` |

Modality order inside a batch is always **vision, audio, text**. Ablations
zero unused streams; they do not shrink the model. Details:
[docs/datasets.md](docs/datasets.md).

## Headline numbers (from checked-in CSVs)

MOSEI BERT classical sweep — best row **TransformerLate**:

| Method | MAE | Acc7 | Acc2 | Corr | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ConcatEarly | 0.6178 | 0.4424 | 0.8084 | 0.6652 | 0.8475 |
| LowRankTensorFusion | 0.5970 | 0.4585 | 0.8305 | 0.6925 | 0.8638 |
| TransformerLate | **0.5846** | **0.4675** | **0.8393** | **0.7041** | **0.8699** |

MOSEI BERT GMTM, full trio: **MAE 0.5640**, Acc7 0.4827, Acc2 0.8429,
Corr 0.7255, F1 0.8777. Text-only GMTM is already 0.5687, so most of that
gain is a stronger text path.

MOSEI GloVe classical sweep — best row **LowRankTensorFusion** (MAE 0.6174).
Trimodal GMTM is 0.6251.

Full tables, MOSI transfer, and caveats (merged MOSI splits, uniform Acc7,
commented train loops): [docs/results.md](docs/results.md) and
[docs/personal_lab_notes.md](docs/personal_lab_notes.md).

## Documentation map

| Doc | Contents |
| --- | --- |
| [docs/setup.md](docs/setup.md) | Environment, paths, checkpoints |
| [docs/architecture.md](docs/architecture.md) | GMTM and fusion internals |
| [docs/datasets.md](docs/datasets.md) | Pickle schema and loaders |
| [docs/metrics.md](docs/metrics.md) | MAE, uniform Acc7/5, Acc2, F1 |
| [docs/experiments.md](docs/experiments.md) | What each script runs |
| [docs/results.md](docs/results.md) | Every CSV, transcribed |
| [docs/code_map.md](docs/code_map.md) | File-by-file index |
| [examples/README.md](examples/README.md) | Synthetic demos |

## Layout

```
docs/                 long-form personal notes
examples/             CPU demos + result printers
model/models.py       networks
model/train_*.py      original experiment runners
model/data/           loaders, SDK notebooks, pickle paths
model/results/        CSV copies + plot notebook
model/mosi_test/      MOSEI → MOSI scoring
```

## License

MIT. See [LICENSE](LICENSE).
