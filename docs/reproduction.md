# Reproduction

This is a personal research tree. Weights and raw `.csd` / `.pkl` files are
**not** in git (see `.gitignore`). The committed CSVs are the record of the
runs that were actually logged.

## Environment

```bash
python -m pip install -r requirements.txt
# GPU: install a CUDA build of torch that matches your driver
```

Python 3.10+ is enough for the examples. The original notebooks were run on
a Windows path (`F:\MOSEI\...`); loaders here use relative paths from
`model/`.

Optional extras, only if you rebuild features from MMSDK:

- [`CMU-MultimodalSDK`](https://github.com/A2Zadeh/CMU-MultimodalSDK) (`mmsdk`)
- `torchtext` (imported by `get_dataloader.py` but unused by the pickle path)
- GloVe `glove.840B.300d.txt`

## Directory layout the scripts expect

```
model/
  data/
    MOSEI/mosei_raw_bert.pkl
    MOSEI/mosei_raw_glove.pkl
    MOSI/mosi_raw_bert.pkl
    MOSI/mosi_raw_glove.pkl
  checkpoints/
    ConcatEarly.pt
    ConcatLate.pt
    ...
    glove_ConcatEarly.pt
    ablation/model_text+audio+visual.pt
    ablation/model_glove_text+audio+visual.pt
```

Working directory for every `train_*.py` is **`model/`** so the relative
`data/...` and `checkpoints/...` paths resolve.

## Rebuild pickles (optional)

1. Download CMU-MOSEI / CMU-MOSI computational sequences into
   `model/data/MOSEI/cmumosei/` and `model/data/MOSI/cmumosi/`.
   Names are listed in [`model/data/readme.md`](../model/data/readme.md).
2. Open `model/data/MOSEI/get_dataset.ipynb` or
   `model/data/MOSI/get_dataset.ipynb`.
3. Alignment strategy in `get_mosei.py`: average high-rate vision/audio onto
   word intervals (`dataset.align(text_field, collapse_functions=[avg])`),
   then align to labeled segments without collapsing.

The notebooks hard-code local Windows paths; change those to your checkout.

## Re-run MOSEI baselines (BERT)

```bash
cd model
# set epochs back to something like 20–50 before treating numbers as real
python train_main_bert.py
```

Writes `checkpoints/{Fusion}.pt` and `main_results.csv` in `model/`.
Copy the CSV into `model/results/` if you want it next to the committed
table.

GloVe: uncomment `train(...)` in `train_main_glove.py`, then

```bash
python train_main_glove.py
```

## Re-run GMTM + ablations

```bash
cd model
# uncomment the train(...) block in train_GMTM_bert.py
# restore the 7-subset modality_combinations list if you want a full ablation
python train_GMTM_bert.py
python train_GMTM_glove.py
```

## MOSI transfer

Requires the MOSEI checkpoints above.

```bash
cd model/mosi_test
python train_mosi_bert.py
python mult_bert_mosi.py
```

`train_mosi_glove.py` currently imports MultiBench's
`training_structures.Supervised_Learning`. Prefer
`mult_glove_mosi.py` / a local `from train_and_test import test` if you do
not have MultiBench installed.

## Known script mismatches

These are present in the committed Python and are documented so a re-run
does not surprise you:

1. **`test()` extra kwargs.** `train_GMTM_glove.py` and several MOSI files
   call `test(..., dataset='mosei', no_robust=True)` but
   `train_and_test.test` only accepts
   `(model, test_dataloaders_all, is_packed, criterion, input_to_float)`.
   Those calls `TypeError` unless you drop the extra keywords.
2. **`train()` commented out** in GMTM BERT / both GloVe trainers — the
   files are eval-only until you restore the fit call.
3. **`epochs=1`** in `train_main_bert.py` will not reproduce
   `main_results.csv`.
4. **Whole-model `torch.save` / `torch.load`.** Loading a `.pt` from a
   different `models.py` revision can fail on attribute mismatches.
5. **MOSI loader merges splits.** Paper-style MOSI test-only numbers will
   differ.
6. **`plt.show()`** inside `single_test` opens a GUI figure on every eval.
   Headless machines should switch the backend
   (`MPLBACKEND=Agg`) or use `examples/eval_metrics.py`.
7. **`Linear` name clash** in `models.py`: a class and then a function
   share the name. Import order inside that file is what the transformers
   rely on; do not rename one without the other.

## What you *can* reproduce without data

From the repo root, with only `requirements.txt`:

```bash
python -m examples.run_all
python -m pytest tests/ -q
```

That checks:

- synthetic tensors match the loader ranks
- every fusion module accepts those ranks and returns a finite tensor
- GMTM forward is `[B, 1]`
- metric helpers match the documented Acc-2 / Acc-7 rules
- a two-epoch CPU train loop reduces L1 on a toy batch
- the committed CSVs parse and the “best row” helpers agree with
  [`experiments.md`](experiments.md)

Those tests do **not** re-fit MOSEI.
