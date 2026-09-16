# Reproducibility

This page lists what you need to recreate the CSV tables, and what
this checkout deliberately does **not** contain.

## Present in git

- Model definitions (`model/models.py`)
- Training / test loop (`model/train_and_test.py`)
- Data loaders and SDK helper scripts
- Result CSVs under `model/`, `model/results/`, and `model/mosi_test/`
- Documentation and synthetic examples added in this branch

## Not in git (and not required for examples)

| Artifact | Why it is missing | How to get it |
| --- | --- | --- |
| `*.pkl` feature caches | Size + dataset license | Build with the notebooks after downloading MOSI/MOSEI |
| `*.csd` SDK sequences | Same | CMU Multimodal SDK / official release |
| `glove.840B.300d.txt` | ~5 GB | Stanford GloVe site |
| `checkpoints/**/*.pt` | Large binary dumps | Retrain; directories are placeholders |
| Random seeds | Never pinned in the scripts | Add `torch.manual_seed` before a serious rerun |

`.gitignore` now excludes those binaries so a local retrain does not
accidentally commit them.

## Software

See [`requirements.txt`](../requirements.txt). The original runs used
PyTorch with CUDA. Examples and tests use CPU PyTorch.

Python 3.10+ is enough. The cloud workspace this branch was edited in
is 3.12.

Optional, only for rebuilding pickles:

- `mmsdk` (CMU Multimodal SDK)
- `h5py`
- `torchtext` (legacy GloVe helper imports)

## Hardware notes

TFN's 128000-D MLP and GMTM's 3×3 transformer grid want a GPU with
enough RAM for batch 32 × seq 50 × BERT 768. The synthetic example
shrinks `seq_len` to 16, `embed_dim` to 16, and `layers` to 1 so a
CPU finish is realistic.

`LowRankTensorFusion.forward` hard-codes the ones-vector device as
`cuda:0` if `torch.cuda.is_available()` else `cpu`. Mixed-device bugs
show up if you move only part of the graph.

## Suggested rerun

```text
cd model
# 1. point filepath at your pickle
# 2. raise total_epochs in train_main_bert.py (it is 1 right now)
# 3. python train_main_bert.py
# 4. python train_GMTM_bert.py
# 5. cd mosi_test && python train_mosi_bert.py
```

Write the new CSVs next to the old ones and diff with
`python examples/inspect_results.py`. Column order is
`Fusion Method, MAE, ACC7, Acc5, ACC2, Corr, F1`.

## Seeds, nondeterminism

`train()` does not call `torch.use_deterministic_algorithms`. CuDNN
is even toggled off for packed forwards. Treat the CSVs as a single
run, not a mean ± std. If you publish a comparison, average a few
seeds and log them.

## License reminder

Code is MIT (`LICENSE`). MOSI, MOSEI, GloVe, and BERT weights are
separate licenses — do not vendor them into this personal repo.
