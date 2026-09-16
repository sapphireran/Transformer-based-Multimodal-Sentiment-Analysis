# Reproduction notes

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For the example suite and unit tests only:

```bash
pip install -r requirements-dev.txt
pytest tests/test_metrics.py tests/test_examples.py -q
python examples/run_all.py
```

GPU is required for the original `*.cuda()` training scripts. The example
scripts default to CPU.

## Data you must fetch yourself

1. CMU-MOSI and CMU-MOSEI computational sequences from the
   [CMU-MultimodalSDK](https://github.com/A2Zadeh/CMU-MultimodalSDK).
2. Optional [GloVe 840B 300d](https://nlp.stanford.edu/projects/glove/) if
   you rebuild GloVe pickles.
3. BERT embeddings, if you rebuild the BERT pickles rather than reusing a
   previously exported `mosei_raw_bert.pkl`.

Place files as described in `model/data/readme.md`. Nothing in git
contains those binaries.

## Train on MOSEI

```bash
cd model
python train_main_bert.py          # six fusion baselines, BERT
python train_main_glove.py         # six fusion baselines, GloVe
python train_GMTM_bert.py          # gated multi-transformer, BERT
python train_GMTM_glove.py         # gated multi-transformer, GloVe
```

Uncomment the `train(...)` call if the script is in eval-only mode. Output
CSVs are written next to the script (`main_results.csv`,
`ablation_results.csv`, …). Copies under `model/results/` are the snapshot
used for plotting.

## Transfer to MOSI

```bash
cd model/mosi_test
python train_mosi_bert.py
python train_mosi_glove.py
python mult_bert_mosi.py
python mult_glove_mosi.py
```

These load `../checkpoints/*.pt` trained on MOSEI. `get_mosi_dataloader`
scores the **union** of MOSI train/valid/test.

## Known sharp edges

* Scripts assume a CUDA device and call `.cuda()` at construction time.
* `torch.load` is used without `weights_only=True` (full module pickle).
* `LowRankTensorFusion` hard-codes the ones-vector device to `cuda:0`
  when a GPU is visible.
* `train_GMTM_glove.py` currently forwards `robust_test` /
  `dataset` / `no_robust` into helpers that do not accept those keywords
  in `train_and_test.py`. Prefer `train_GMTM_bert.py` as the GMTM
  reference, or delete the extra kwargs before a GloVe re-run.
* `models.py` defines both a `Linear` class and a later `Linear`
  factory function; the factory wins. Import `CustomLinear` if you need
  a module.
* Acc-7 / Acc-5 in this repo are **uniform bins**, not the integer-label
  7-class scheme used in some older MOSI papers.
* `train_main_glove.py` builds `TransformerSeq` widths 64+128+512 = 704
  but constructs `LateFusionTransformer(in_dim=1792)` (BERT's 64+128+1024
  is 1216 and does match). A GloVe Transformer-late **retrain** from
  scratch will fail on that forward until `in_dim` is 704; the committed
  GloVe CSV row was produced from a checkpoint that already existed.
* Packed vs padded collate is easy to mix up. See
  [batch_layout.md](batch_layout.md) and `examples/packed_vs_padded.py`.

## CPU smoke test (no dataset)

The examples do not replace a full training run. They do confirm:

* GMTM and the fusion modules accept the documented tensor shapes
* the metric helpers match the original `digitize` edges
* a short AdamW + L1 loop on synthetic labels descends

```bash
python examples/run_all.py
```
