# Model storage — main experiments

This folder holds `torch.save(model, …)` pickles from the fusion
sweeps. Weights are gitignored (`*.pt`).

| File pattern | Script |
| --- | --- |
| `{Fusion}.pt` | `train_main_bert.py` |
| `glove_{Fusion}.pt` | `train_main_glove.py` |

`Fusion` is one of `ConcatEarly`, `ConcatLate`, `LowRankTensorFusion`,
`TensorFusion`, `TransformerEarly`, `TransformerLate`.

Ablation weights live in [`ablation/`](ablation/). How they are
produced: [docs/reproducing.md](../../docs/reproducing.md).
