# Getting started

## Environment

Python 3.10+ is enough for the examples. The original training scripts were run
with PyTorch 2.x and CUDA.

```bash
python3 -m pip install -r requirements.txt
```

`torch` CPU wheels are sufficient for `examples/` and `tests/`. A GPU is only
required if you uncomment `train(...)` in the `model/train_*.py` scripts — those
call `.cuda()` on every encoder, fusion module, and batch.

Optional extras, only if you rebuild features from the CMU SDK:

- [`CMU-MultimodalSDK`](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK) (`mmsdk`)
- `torchtext` if you swap in a GloVe vocab builder of your own
- `glove.840B.300d.txt` for the GloVe path (see `model/data/readme.md`)

## Repository layout the scripts expect

Training scripts are written to be launched **from `model/`**:

```bash
cd model
python train_main_bert.py
```

They add `os.getcwd()` to `sys.path` and then import:

```text
from train_and_test import train, test
from models import ...
from data.get_dataloader import get_dataloader
```

Relative pickle paths are therefore also relative to `model/`:

```text
data/MOSEI/mosei_raw_bert.pkl
data/MOSEI/mosei_raw_glove.pkl
data/MOSI/mosi_raw_bert.pkl
data/MOSI/mosi_raw_glove.pkl
```

Examples do the opposite: they run from the **repo root** and insert `model/`
onto `sys.path` themselves. You do not need to `cd model` for those.

## Device assumptions

| Code path | Device |
| --- | --- |
| `examples/*`, `tests/*` | CPU (`torch.device("cpu")`) |
| `model/train_*.py`, `model/mosi_test/*.py` | Hard-coded `.cuda()` |
| `LowRankTensorFusion` ones-vector | CUDA if `torch.cuda.is_available()`, else CPU |

If you want to fine-tune on a CPU-only laptop, replace `.cuda()` with
`.to(device)` and pass `device` through. That change is out of scope for the
docs/examples work.

## Running the walkthroughs

```bash
python3 examples/read_result_tables.py   # no torch tensors, just the CSVs
python3 examples/gmtm_forward.py         # GMTM on a 4 x 16 synthetic clip
python3 examples/fusion_shapes.py        # every fusion module, printed shapes
python3 examples/metrics_demo.py         # Acc-7 / Acc-5 / Acc-2 / F1 on toy scores
python3 examples/multiframework_demo.py  # encoders → fusion → head
python3 examples/dataset_collate_demo.py # Affectdataset + _process_1 / _process_2
bash examples/run_all.sh
python3 -m pytest -q
```

## What you will not find in git

- `*.pkl` feature dumps
- `*.pt` trained weights
- Raw `.csd` computational sequences
- `glove.840B.300d.txt`

Checkpoint readmes under `model/checkpoints/` only record the intended
filenames.

## Typical first reading order

1. This page
2. [`architecture.md`](architecture.md) — what GMTM actually computes
3. [`data-pipeline.md`](data-pipeline.md) — what a batch looks like
4. [`metrics.md`](metrics.md) — how a scalar in `[-3, 3]` becomes Acc-7
5. [`examples/gmtm_forward.py`](../examples/gmtm_forward.py) — run it
6. [`results.md`](results.md) — what the recorded tables say
