# Reproducing a run

This is the personal checklist I want next time I open the repo on a
new machine. The synthetic examples do **not** replace a full MOSEI
train; they only prove the modules still import and step.

## 0. Environment

```bash
python3 -m pip install -r requirements.txt
# GPU train: install the CUDA build of torch for your driver
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

Working directory for the training scripts is **`model/`**, because
they look up `data/MOSEI/...` and `checkpoints/` as relative paths.

```bash
cd model
```

## 1. Data

1. Download CMU-MOSI / CMU-MOSEI computational sequences from the
   [CMU Multimodal SDK](https://github.com/A2Zadeh/CMU-MultimodalSDK).
2. Put the `.csd` files where [`model/data/readme.md`](../model/data/readme.md)
   says.
3. For GloVe runs, place `glove.840B.300d.txt` as noted in that file.
4. Produce the four pickles (or copy them from the machine that already
   has them):

   - `data/MOSEI/mosei_raw_bert.pkl`
   - `data/MOSEI/mosei_raw_glove.pkl`
   - `data/MOSI/mosi_raw_bert.pkl`
   - `data/MOSI/mosi_raw_glove.pkl`

   [`get_mosei.py`](../model/data/MOSEI/get_mosei.py) and the notebooks
   under `data/MOSI/` / `data/MOSEI/` are the starting point. They still
   contain local Windows paths (`F:\MOSEI\...`) that you will need to
   edit.

5. Sanity-check a pickle:

   ```python
   import pickle
   data = pickle.load(open("data/MOSEI/mosei_raw_bert.pkl", "rb"))
   for split in ("train", "valid", "test"):
       print(split, {k: data[split][k].shape for k in data[split]})
   ```

   You want `vision[..., 35]`, `audio[..., 74]`, `text[..., 768]`.

## 2. Smoke the modules without data

From the repo root (not `model/`):

```bash
python3 examples/fusion_forward.py
python3 examples/gmtm_forward.py
python3 examples/gmtm_toy_train.py
python3 examples/evaluate_metrics.py
python3 examples/packed_vs_padded.py
python3 -m pytest tests -q
```

If these fail, fix imports / torch first. A broken toy loop will also
be a broken full train.

## 3. Uncomment train, then run one fusion

The sweep scripts mix "train" and "test existing checkpoint" states.
Before a real run:

1. Open the script.
2. Confirm `filepath` points at a pickle that exists.
3. Uncomment `train(...)`.
4. Set `total_epochs` back to something like 20–50, not `1`.
5. Confirm `save=` lands under `checkpoints/` or
   `checkpoints/ablation/`.
6. Run **one** fusion method first (`fusion_methods = ['ConcatLate']`)
   so a shape error does not waste a night.

```bash
cd model
python3 train_main_bert.py
```

`torch.save(model, path)` pickles the whole `MultiFramework`. Load with
the same code revision; `torch.load` without `weights_only=True` is
what the scripts use.

## 4. Ablations

`train_GMTM_bert.py` / `train_GMTM_glove.py` iterate a list of modality
sets. Zeros fill unused streams (see [data-pipeline.md](data-pipeline.md)).
Start with `['text', 'audio', 'visual']` only, then open the rest of
the combinations.

Checkpoint names use `'+'.join(modalities)`, e.g.
`checkpoints/ablation/model_text+audio+visual.pt`.

## 5. MOSI transfer

After MOSEI checkpoints exist:

```bash
cd model/mosi_test
python3 train_mosi_bert.py
```

Those scripts assume `../checkpoints/{Fusion}.pt`. They currently call
`test()` with extra keywords (`dataset=`, `no_robust=`) that
`train_and_test.test` does **not** accept. If you revive them, either
drop those kwargs or add them to `test`. The synthetic suite does not
import those scripts for that reason.

## 6. Logging

Each sweep appends
`[name, MAE, Acc7, Acc5, Acc2, Corr, F1]` and writes a CSV. Keep new
tables under `model/results/` with a date in the filename if you do
not want to overwrite the recorded ones.

`single_test` will try to `plt.show()` a confusion matrix. On a
headless box set `MPLBACKEND=Agg` or comment that block out.

## 7. Things that have bitten me

- Running a script from the repo root instead of `model/` → missing
  pickle / checkpoint paths.
- `is_packed` mismatch (TransformerEarly wants padded batches).
- GloVe `LateFusionTransformer(in_dim=1792)` vs actual concat width.
- BERT `TransformerEarly` head `MLP(64, …)` vs fusion width 32.
- MOSI loader merging all splits — do not compare that Acc7 to a paper
  that uses official test only.
- `memory_profiler` import at module level in `train_and_test.py`:
  install it or the import fails before you can train.

## 8. What "done" looks like for a new experiment

- [ ] One CSV with all fusion rows, same seed / epoch budget.
- [ ] Checkpoints named after the method.
- [ ] A short note in `docs/experiments.md` (or a new dated markdown)
      with the command line and GPU.
- [ ] Acc7 / Acc5 computed with `split_uniform_*`, not a different
      binning.
