# Reproduction notes

This page lists what a future you (or anyone cloning the personal repo)
needs to recreate the CSV tables. The git tree alone is **not** enough:
pickles and checkpoints were never committed.

## You already have

- Model definitions (`model/models.py`)
- Train / test loop (`model/train_and_test.py`)
- Experiment scripts and MOSI transfer scripts
- Result CSVs (the recorded numbers)
- These docs and the CPU `examples/` suite

## You still need

| Artifact | Why | Typical location |
| --- | --- | --- |
| CMU-MOSI / MOSEI `.csd` files | raw features + labels | `model/data/MOSI/cmumosi`, `model/data/MOSEI/cmumosei` |
| BERT tokenizer / weights | rebuild `*_bert.pkl` | Hugging Face cache, used by the notebooks |
| `glove.840B.300d.txt` | rebuild `*_glove.pkl` | path noted in `model/data/readme.md` |
| Processed pickles | what the scripts actually load | `mosei_raw_bert.pkl`, etc. |
| `checkpoints/*.pt` | skipped `train()` paths | see [repo_map.md](repo_map.md) |
| A GPU | the scripts call `.cuda()` | CUDA 11/12 + matching torch |

`examples/run_all.py` does **not** need any of the above.

## Environment

```bash
python -m pip install -r requirements.txt
# plus, for pickle rebuilds:
# pip install h5py torchtext
# pip install mmsdk   # CMU Multimodal SDK, install from their repo
```

Run experiment scripts with the working directory set so the `data/` and
`checkpoints/` relative paths resolve:

```bash
cd model
python train_main_bert.py
```

MOSI scripts:

```bash
cd model/mosi_test
python train_mosi_bert.py
```

## Recreate MOSEI BERT fusion numbers

1. Build `data/MOSEI/mosei_raw_bert.pkl` with `model/data/MOSEI/get_dataset.ipynb`.
2. In `train_main_bert.py`, set `total_epochs` to something real (the
   checked-in value is `1`). Keep AdamW `1e-4`, L1, early stop.
3. Run the six-method loop. It writes `main_results.csv`.
4. Compare columns, not exact floats — seeds are not pinned.

`train()` does not set `torch.manual_seed`. Expect jitter of a few points
in the third decimal of MAE.

## Recreate GMTM ablations

1. Uncomment `train(...)` in `train_GMTM_bert.py` / `train_GMTM_glove.py`.
2. Restore the full `modality_combinations` list in the BERT script if you
   want all seven rows.
3. Confirm `get_ablation_dataloader` still zeros unused streams rather than
   shrinking the model.
4. Write the CSV (GloVe script already does; BERT’s writer is commented).

## Recreate MOSI transfer numbers

1. Keep the MOSEI checkpoints from the runs above.
2. Build `data/MOSI/mosi_raw_bert.pkl` (and GloVe if needed).
3. Run `model/mosi_test/*.py`.
4. Remember `get_mosi_dataloader` **merges** MOSI splits. If you change that
   to test-only, start a new CSV instead of overwriting the old one.

`train_mosi_glove.py` imports `training_structures.Supervised_Learning`,
which is not in this repo. Point it at `train_and_test` before running.

## Seeds, versions, and hardware

Not recorded in the CSVs:

- Python / torch / CUDA versions
- GPU model
- random seeds
- exact BERT checkpoint name (base vs large, cased, etc.)
- whether `z_norm` was on (loader default is `False`)

The feature widths (`35 / 74 / 768` and `35 / 74 / 300`) are the strongest
reproducibility constraint. If your pickle has Facet-43 or COVAREP-74 with
a different column subset, every hardcoded `in_dim` will break.

## Sanity checks before a long train

```bash
python examples/run_all.py
```

Then, with a pickle present, in a Python shell from `model/`:

```python
from data.get_dataloader import get_dataloader
tr, va, te = get_dataloader(
    "data/MOSEI/mosei_raw_bert.pkl",
    batch_size=4,
    data_type="mosei",
    num_workers=0,
)
batch = next(iter(tr))
print(type(batch), len(batch))
```

Packed batches are a 4-tuple `(features, lengths, index, label)`.
Max-padded batches are `(vision, audio, text, label)`.

Overfit `ConcatLate` on one batch: if MAE does not drop, the label tensor
is the wrong slice (`collate` takes `labels[:, 0]` when `C > 1`).

## What “done” looks like

A reproduction that matches this repo’s *procedure* will:

- train with L1 on packed or padded loaders as specified per method
- report MAE / Acc7 / Acc5 / Acc2 / Corr / F1 from `single_test`
- keep GMTM at `n_modalities=3` with zeroed streams for ablations
- not claim MOSI merged-split scores as official MOSI test accuracy

A reproduction that matches this repo’s *floats* would also need the
original pickles, seeds, and hardware — those are not in git.
