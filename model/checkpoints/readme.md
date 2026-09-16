# Checkpoints — main fusion bake-off

Weights are local only. This file is the naming contract
`train_main_*.py` and `mosi_test/train_mosi_*.py` already use.

Format: `torch.save(model, path)` of a full `MultiFramework`
(`encoders + fusion + head`). Loading needs the same class names in
`models.py` / `train_and_test.py`. Do not rename `MultiFramework`,
`ConcatLate`, `LateFusionTransformer`, etc. before a re-eval.

## BERT (MOSEI) — `train_main_bert.py`

| File | Fusion |
| ---- | ------ |
| `ConcatEarly.pt` | early concat + LSTM |
| `ConcatLate.pt` | per-modality LSTM + late concat |
| `LowRankTensorFusion.pt` | LMF rank 32 |
| `TensorFusion.pt` | full TFN |
| `TransformerEarly.pt` | joint transformer, max-pad loader |
| `TransformerLate.pt` | per-modality TransformerSeq + late transformer |

Logged table: `../main_results.csv`. Write-up:
[`../../notes/01-mosei-bert-fusion.md`](../../notes/01-mosei-bert-fusion.md).

MOSI transfer loads these exact names from `mosi_test/` via
`../checkpoints/{Fusion}.pt`. Late concat is *called* `Concat` in that
script’s method list but still loads `ConcatLate.pt`? **Check before
re-running:** `train_mosi_bert.py` uses `../checkpoints/{fusion_method}.pt`
and the method string `'Concat'`, so it expects `Concat.pt`, not
`ConcatLate.pt`. If I only have `ConcatLate.pt` on disk, symlink or
rename. I am recording the inconsistency here so I do not debug it
twice. See also [`../../docs/repo-map.md`](../../docs/repo-map.md).

## GloVe (MOSEI) — `train_main_glove.py`

Prefix `glove_`:

```text
glove_ConcatEarly.pt
glove_ConcatLate.pt
glove_LowRankTensorFusion.pt
glove_TensorFusion.pt
glove_TransformerEarly.pt
glove_TransformerLate.pt
```

`train_mosi_glove.py` loads `glove_{fusion_method}.pt` with method
string `'Concat'` → `glove_Concat.pt`. Same naming trap as BERT.

Logged table: `../glove_results.csv`. Write-up:
[`../../notes/02-mosei-glove-fusion.md`](../../notes/02-mosei-glove-fusion.md).

## Ablation / GMTM weights

See [`ablation/readme.md`](ablation/readme.md).
