# Reproduction and extension

This is a personal checklist for rerunning or extending the MOSI / MOSEI
sweeps without rediscovering the packed-vs-padded traps.

## Environment

```bash
python -m pip install -r requirements.txt
# GPU optional. examples/ are CPU-only.
```

Confirm imports from the `model/` directory:

```bash
cd model
python -c "from models import GatedMultiTransfomerModel; from train_and_test import train, test"
```

The custom transformer stack imports `LayerNorm` from `torch.nn` and
`pack_padded_sequence` from `torch.nn.utils.rnn`. No extra CUDA build is
required beyond a normal PyTorch wheel.

## Data you must supply

Git does not contain pickles. Place:

```
model/data/MOSEI/mosei_raw_bert.pkl
model/data/MOSEI/mosei_raw_glove.pkl
model/data/MOSI/mosi_raw_bert.pkl
model/data/MOSI/mosi_raw_glove.pkl
```

Each pickle is a dict with `train` / `valid` / `test` and keys
`vision`, `audio`, `text`, `labels` (see [datasets.md](datasets.md)).

To only test that **code paths** work, skip pickles and run
`examples/05_mini_training.py`.

## Retrain a fusion baseline on MOSEI BERT

`train_main_bert.py` already loops the six names in `fusion_methods`.
From `model/`:

```bash
python train_main_bert.py
```

It will:

1. Build packed dataloaders (`*_OT`) and max-pad dataloaders (`*_TE`).
2. For each fusion, construct encoders / fusion / head (BERT widths).
3. `train(..., total_epochs=1, save=checkpoints/{name}.pt)`.
4. `test(...)` and append a row to `main_results.csv`.

Raise `total_epochs` before a serious run. The original table is not a
1-epoch result.

`TransformerEarly` **must** use the max-pad loaders. The `if fusion ==
'TransformerEarly'` branch already does this.

## Retrain GMTM

In `train_GMTM_bert.py` uncomment:

```python
train(encoders, fusion, head, traindata, validdata, 50,
      optimtype=torch.optim.AdamW, early_stop=True, is_packed=False,
      lr=1e-4, save=f"checkpoints/ablation/model_{'+'.join(modalities)}.pt",
      weight_decay=0.01, objective=torch.nn.L1Loss())
```

Uncomment the modality combinations you want. Names in the save path are
joined with `+` (`model_text+audio+visual.pt`). That string is also what
the MOSI transfer scripts load.

GloVe counterpart: `train_GMTM_glove.py`, `input_dims = [35, 74, 300]`,
save prefix `model_glove_`.

## Evaluate an existing checkpoint

```python
model = torch.load("checkpoints/TransformerLate.pt", map_location="cuda")
test(model=model, test_dataloaders_all=testdata,
     is_packed=True, criterion=torch.nn.L1Loss())
```

`is_packed` is `False` only for TransformerEarly and GMTM.

MOSI transfer: run scripts in `model/mosi_test/` after the MOSEI
checkpoints exist. Those scripts do not train.

## Adding a new fusion module

1. Implement `forward(self, modalities: List[Tensor]) -> Tensor` in
   `models.py`.
2. Add a branch in `run_experiment` (both BERT and GloVe trainers) that
   sets `encoders`, `fusion`, `head`, and the packed flag.
3. Add the name to `fusion_methods`.
4. Decide Identity-vs-LSTM/GRU encoders. If you need variable-length
   RNNs, use `has_padding=True` and `_process_1`.
5. Run `examples/02_fusion_forward.py` against a **synthetic** batch first
   so a shape error shows up before a 16k-clip epoch.

A minimal fusion for experiments:

```python
class MeanFusion(nn.Module):
    def forward(self, modalities):
        stacked = torch.stack([m.flatten(1) for m in modalities], dim=0)
        return stacked.mean(0)
```

Pair it with the ConcatLate LSTM encoders so the head still sees a
`[B, H]` vector.

## Adding a modality

The loader always emits three tensors. To add a fourth stream you need to
change `Affectdataset.__getitem__`, both collate functions, the ablation
index map, every `input_dims = [35, 74, 768]` literal, and GMTM’s
`n_modalities`. There is no registry. Prefer experimenting with **zeroed**
slots (already supported) before widening the tuple.

## Seeds and nondeterminism

The training scripts do not set `torch.manual_seed`. Packed LSTM +
`cudnn` with `enabled=False` only on the packed forward (see `train()`)
means two runs will not match bit-for-bit. For a personal paper table,
average a few seeds or freeze:

```python
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
```

## What not to “fix” while reproducing

These are recorded quirks, not accidental leftovers in the docs:

- `GatedMultiTransfomerModel` spelling.
- `EarlyFusionTransformer.self.linear` is unused.
- BERT early-fusion head is `MLP(64, …)` on a 32-d encoder.
- GloVe `LateFusionTransformer(in_dim=1792)` does not equal 64+128+512.
- `test()` runs the inner evaluation twice.
- MOSI loaders merge all splits.
- `get_mosei.py` is a MOSI SDK scratch file with a local Windows path.

Change them only in a dedicated experiment branch so old `.pt` files
remain loadable.
