# Training and evaluation scripts

Run every training script from the `model/` directory so the relative pickle
and checkpoint paths resolve.

```bash
cd model
python train_main_bert.py
python train_GMTM_bert.py
```

## Optimizer and loop

`train()` in `model/train_and_test.py` is the shared supervised loop:

* optimizer: AdamW in the published scripts (`optimtype=torch.optim.AdamW`)
* loss: `L1Loss` on the scalar sentiment
* learning rate: `1e-4`, weight decay `0.01`
* grad clip: `8`
* early stop: 7 epochs of no validation-loss improvement when enabled
* checkpoint: full `torch.save(model, path)` of the `MultiFramework` wrapper
* optional `memory_profiler` peak-RSS tracking via `all_in_one_train`

`is_packed=True` feeds `[modalities, lengths]` into recurrent encoders that
call `pack_padded_sequence`. Transformer-early and GMTM use stacked padded
batches (`max_pad=True`, `is_packed=False`).

## Script map

| Script | Data | What it trains / tests |
| --- | --- | --- |
| `train_main_bert.py` | MOSEI BERT | Concat / TFN / LMF / early+late Transformer |
| `train_main_glove.py` | MOSEI GloVe | same fusion zoo, GloVe widths |
| `train_GMTM_bert.py` | MOSEI BERT | GMTM, optional modality ablation |
| `train_GMTM_glove.py` | MOSEI GloVe | GMTM ablation over all 7 subsets |
| `mosi_test/train_mosi_bert.py` | MOSI BERT | load MOSEI checkpoints, score MOSI |
| `mosi_test/train_mosi_glove.py` | MOSI GloVe | same for GloVe checkpoints |
| `mosi_test/mult_bert_mosi.py` | MOSI BERT | GMTM transfer |
| `mosi_test/mult_glove_mosi.py` | MOSI GloVe | GMTM transfer |

Several scripts have the `train(...)` call commented out and only run
`torch.load` + `test`. That is intentional for reproduction after the
`.pt` files exist under `model/checkpoints/`. Uncomment the `train(...)`
block to fit a new seed.

## Checkpoint layout

```
model/checkpoints/
  ConcatEarly.pt
  ConcatLate.pt
  LowRankTensorFusion.pt
  TensorFusion.pt
  TransformerEarly.pt
  TransformerLate.pt
  glove_*.pt
  ablation/
    model_text+audio+visual.pt
    model_glove_text+audio+visual.pt
    ...
```

The directories currently hold only placeholder readmes; the weight files
are local artifacts (see `.gitignore`).

## BERT vs GloVe encoder widths

Copied from the training scripts so a re-run does not have to reverse-engineer
hidden sizes:

**BERT (text = 768)**

| Fusion | Encoders | Head / fusion extras |
| --- | --- | --- |
| ConcatEarly | 3 × Identity | LSTM(877 → 1024) + MLP |
| ConcatLate | LSTM 35→64, 74→256, 768→1024 | MLP(1344) |
| LMF | GRU+Linear → 32 / 64 / 256 | LMF rank 32, out 256 |
| TFN | GRU+Linear → 19 / 39 / 159 | MLP(128000) |
| TransformerEarly | Identity | EarlyFusionTransformer(877) |
| TransformerLate | TransformerSeq 64 / 128 / 1024 | LateFusionTransformer(1216) |
| GMTM | Identity | `GatedMultiTransfomerModel(3, [35,74,768])` |

**GloVe (text = 300)**

| Fusion | Encoders | Head / fusion extras |
| --- | --- | --- |
| ConcatEarly | 3 × Identity | LSTM(409 → 512) + MLP |
| ConcatLate | LSTM 35→64, 74→256, 300→512 | MLP(832) |
| LMF | GRU+Linear → 32 / 64 / 128 | LMF rank 32, out 128 |
| TFN | GRU+Linear → 19 / 39 / 79 | MLP(64000) |
| TransformerEarly | Identity | EarlyFusionTransformer(409) |
| TransformerLate | TransformerSeq 64 / 128 / 512 | LateFusionTransformer(1792) |
| GMTM | Identity | `GatedMultiTransfomerModel(3, [35,74,300])` |

## Ablation protocol

`modality_combinations` in the GMTM scripts iterates

```
[text], [audio], [visual],
[text, audio], [text, visual], [audio, visual],
[text, audio, visual]
```

Dropped slots are zero tensors, not removed modules, so parameter count
stays constant.

## What the examples cover instead

Full MOSEI training needs the pickles, a GPU, and tens of minutes per
fusion method. `examples/train_toy_gmtm.py` runs the same GMTM graph on
synthetic batches so the forward / backward path can be checked on CPU.
