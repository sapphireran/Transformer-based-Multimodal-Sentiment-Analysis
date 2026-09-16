# Reproducing experiments

This is a personal checklist, not a one-click pipeline. The published CSVs were produced on a CUDA box with local pickles that are **not** in git.

## 0. Environment

```bash
python -m pip install -r requirements.txt
# optional, only to rebuild pickles from CMU .csd files:
# python -m pip install h5py torchtext
# pip install git+https://github.com/A2Zadeh/CMU-MultimodalSDK
```

Confirm:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The train scripts call `.cuda()` on every module. CPU will throw unless you edit those lines. The [examples](../examples/README.md) are CPU-only and do not need this step.

## 1. Download computational sequences

Using the CMU Multimodal SDK (names match `model/data/readme.md`):

```python
from mmsdk import mmdatasdk as md

md.mmdataset(md.cmu_mosei.highlevel, "model/data/MOSEI/cmumosei")
md.mmdataset(md.cmu_mosei.labels,    "model/data/MOSEI/cmumosei")
md.mmdataset(md.cmu_mosi.highlevel,  "model/data/MOSI/cmumosi")
md.mmdataset(md.cmu_mosi.labels,     "model/data/MOSI/cmumosi")
```

You want at least:

```text
CMU_MOSEI_COVAREP.csd
CMU_MOSEI_Labels.csd
CMU_MOSEI_TimestampedWords.csd
CMU_MOSEI_VisualFacet42.csd
CMU_MOSI_COVAREP.csd
CMU_MOSI_Opinion_Labels.csd
CMU_MOSI_TimestampedWords.csd
CMU_MOSI_Visual_Facet_42.csd
```

GloVe runs also need `glove.840B.300d.txt` (Common Crawl, 300-d) next to the data folder. BERT runs need a local Hugging Face / torch BERT checkpoint the notebooks already know how to call.

## 2. Align and pickle

1. Point `MOSI_PATH` in `model/data/MOSEI/get_mosei.py` at your `cmumosi` / `cmumosei` directory (the committed path is a Windows `F:\...` leftover).
2. Run the align-to-words + align-to-labels steps (script or `get_dataset.ipynb`).
3. Embed words with BERT or GloVe.
4. Write:

```text
model/data/MOSEI/mosei_raw_bert.pkl
model/data/MOSEI/mosei_raw_glove.pkl
model/data/MOSI/mosi_raw_bert.pkl
model/data/MOSI/mosi_raw_glove.pkl
```

Each pickle must be a dict of `train` / `valid` / `test`, each with `vision`, `audio`, `text`, `labels`. See [datasets](datasets.md).

Sanity-check without training:

```python
from data.get_dataloader import get_dataloader
tr, va, te = get_dataloader("data/MOSEI/mosei_raw_bert.pkl", batch_size=4, max_pad=True, num_workers=0)
v, a, t, y = next(iter(tr))
assert v.shape[-1] == 35 and a.shape[-1] == 74 and t.shape[-1] == 768
```

## 3. Train MOSEI BERT baselines

Edit `train_main_bert.py`:

- set `total_epochs` to 20–50, not 1
- keep `early_stop=True`

```bash
cd model
python train_main_bert.py
```

Writes `checkpoints/{ConcatEarly,ConcatLate,LowRankTensorFusion,TensorFusion,TransformerEarly,TransformerLate}.pt` and `main_results.csv`.

## 4. Train MOSEI BERT GMTM ablations

In `train_GMTM_bert.py`:

- uncomment the seven modality rows
- uncomment `train(...)`
- keep `max_pad=True`, `is_packed=False`

```bash
python train_GMTM_bert.py
```

Writes `checkpoints/ablation/model_<mod+mod>.pt` and (if you uncomment the CSV block) `ablation_results.csv`.

## 5. GloVe twins

Same two scripts with `mosei_raw_glove.pkl` and input dim 300:

- uncomment `train(...)` in `train_main_glove.py` / `train_GMTM_glove.py`
- delete `dataset=` / `no_robust=` from the `test(...)` calls
- fix `LateFusionTransformer(in_dim=...)` to `64+128+512=704` if you train that variant

## 6. MOSI transfer

With MOSEI checkpoints already on disk:

```bash
cd model/mosi_test
python train_mosi_bert.py
python mult_bert_mosi.py
```

These load `../checkpoints/*.pt` and score the **merged** MOSI pickle. They do not train.

`train_mosi_glove.py` still imports MultiBench (`training_structures.Supervised_Learning`). Either install that stack or retarget the import to `train_and_test` as the BERT file already does.

## 7. Plots

`model/results/plot.ipynb` reads the four MOSEI CSVs, renames the last ablation row to `GatedMultiTransformer`, and concatenates it onto the main tables. Copy fresh CSVs into `model/results/` before re-running the notebook.

## 8. What “reproduced” means here

A rerun matches this repo if:

- MAE / Corr on MOSEI BERT GMTM full-trio are within a small noise band of **0.5640 / 0.7255**
- the same ranking holds: GMTM ≤ late transformer ≤ LRTF ≤ the concats
- text-only stays close to the full model; audio-only / visual-only stay near MAE 0.82

A rerun does **not** need to match Acc-7 to four decimals. Equal-width binning is jumpy.

## 9. If you only want to exercise the code

Skip steps 1–8 and run:

```bash
python examples/run_all_examples.py
python -m pytest tests/test_examples.py -q
```

That path uses synthetic `[B, 50, D]` tensors and the published CSVs. It does not claim to reproduce the tables.
