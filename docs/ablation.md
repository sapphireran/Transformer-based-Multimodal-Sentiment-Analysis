# Ablation study

GMTM is always a **three-slot** network. An ablation run zeros the slots you
drop (`get_ablation_dataloader`) instead of deleting encoders. That answers
“does this stream still help when the architecture and parameter count stay
fixed?” rather than “does a smaller network on fewer streams work?”.

Scripts:

- `model/train_GMTM_bert.py` — BERT text, currently set to the full triple
- `model/train_GMTM_glove.py` — GloVe text, all seven combinations
- `model/mosi_test/mult_*_mosi.py` — same checkpoints scored on MOSI

Tables: `model/results/ablation_results.csv` (MOSEI BERT),
`ablation_glove_results.csv` (MOSEI GloVe), and the MOSI twins under
`model/mosi_test/`.

## Protocol

For each `modalities` list in

```text
[text]
[audio]
[visual]
[text, audio]
[text, visual]
[audio, visual]
[text, audio, visual]
```

1. Build max-padded loaders with unused streams replaced by zeros of the
   original shape.
2. Construct `GatedMultiTransfomerModel(3, [35, 74, F_text], HParams)`.
3. Train with L1 (when the `train(...)` call is enabled) or load
   `checkpoints/ablation/model_{'+'.join(modalities)}.pt`.
4. Score with `single_test`.

Filename tokens use `+`, e.g. `model_text+audio+visual.pt` and
`model_glove_text+visual.pt`.

## MOSEI BERT (recorded)

| Modalities | MAE | Acc7 | Acc5 | Acc2 | Corr | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| text | 0.5687 | 0.4728 | 0.5516 | 0.8404 | 0.7202 | 0.8741 |
| audio | 0.8306 | 0.4127 | 0.4134 | 0.6252 | 0.1124 | 0.7552 |
| visual | 0.8217 | 0.4012 | 0.4127 | 0.6293 | 0.2061 | 0.7693 |
| text+audio | 0.5659 | 0.4806 | 0.5596 | 0.8409 | 0.7227 | 0.8731 |
| text+visual | 0.5667 | 0.4838 | 0.5538 | 0.8324 | 0.7143 | 0.8697 |
| audio+visual | 0.8220 | 0.3951 | 0.4149 | 0.6274 | 0.2231 | 0.7670 |
| text+audio+visual | **0.5640** | 0.4827 | 0.5561 | **0.8429** | **0.7255** | **0.8777** |

### What this says

**Text is the task.** Text-only is already within `0.005` MAE of the full
model and within `0.006` Corr. Adding audio and visual is a small but
consistent nudge on MAE / Acc2 / Corr / F1.

**Non-text streams are weak alone.** Audio-only correlation `0.11` and
visual-only `0.21` mean those features, under this preprocessing and this
network, do not carry much sentiment by themselves. Acc7 near `0.40` for
those rows is close to dumping mass in the center bins.

**Audio+visual without text** does not recover. MAE stays `0.822`, Corr
`0.22` — about the same as visual-only. The cross-modal grid cannot invent
text-like signal from Facet + COVAREP here.

**Text+audio vs text+visual.** Mixed results: text+audio wins Corr and Acc2;
text+visual wins Acc7. Neither pair beats the full triple on MAE or F1.

## MOSEI GloVe (recorded)

| Modalities | MAE | Acc7 | Acc5 | Acc2 | Corr | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| text | 0.6616 | 0.4220 | 0.5091 | 0.8118 | 0.6557 | 0.8509 |
| audio | 0.8232 | 0.4127 | 0.4134 | 0.6362 | 0.2067 | 0.7668 |
| visual | 0.8139 | 0.3977 | 0.4185 | 0.6480 | 0.2256 | 0.7675 |
| text+audio | 0.7200 | 0.3503 | 0.4550 | 0.7760 | 0.5503 | 0.8186 |
| text+visual | 0.6612 | 0.4145 | 0.5207 | 0.7980 | 0.6459 | 0.8466 |
| audio+visual | 0.8141 | 0.3958 | 0.4205 | 0.6489 | 0.2345 | 0.7678 |
| text+audio+visual | **0.6251** | **0.4467** | **0.5317** | 0.8079 | **0.6714** | 0.8474 |

GloVe text is weaker than BERT text (text-only MAE `0.66` vs `0.57`). The
full trio **does** pick up a clearer gain over text-only (`0.625` vs
`0.662` MAE, `0.671` vs `0.656` Corr). Non-text-only rows look similar to
the BERT ablation.

`text+audio` is a **regression** in this GloVe table (MAE `0.720`, Acc7
`0.350`). That is worth treating as an unstable run rather than a finding
that audio hurts GloVe. Re-seed before concluding.

## MOSI transfer (same checkpoints)

MOSI loaders merge all splits (see [datasets.md](datasets.md)). Use these as
transfer diagnostics.

BERT GMTM on MOSI (`ablation_mosi_results.csv`):

| Modalities | MAE | Acc2 | Corr |
| --- | ---: | ---: | ---: |
| text | 0.9363 | 0.7667 | 0.6482 |
| audio | 1.3852 | 0.4509 | −0.1256 |
| visual | 1.3741 | 0.5165 | −0.0278 |
| text+audio | 0.9349 | 0.7786 | 0.6494 |
| text+visual | **0.9044** | **0.7796** | **0.6623** |
| audio+visual | 1.3711 | 0.5208 | 0.0226 |
| full | 0.9493 | 0.7638 | 0.6403 |

On MOSI, **text+visual** beats the full triple. The extra audio stream may
be overfitting MOSEI acoustics that do not transfer. Audio-only correlation
goes negative — the MOSEI audio head is not a MOSI audio head.

GloVe GMTM on MOSI (`ablation_mosi_glove_results.csv`) is the one setting
where the full triple is best (`MAE 0.9748`, `Corr 0.6177`), and it also
beats every GloVe *fusion-sweep* row on MOSI (those MAEs are `1.08–1.19`).
GMTM is doing more of the work when the text encoder is GloVe.

## How to read this as a modeling choice

1. If you only have compute for one BERT model, train **text+audio+visual
   GMTM** or **TransformerLate**. The ablation says you will not win big
   by dropping a stream, and you might lose a little.
2. If you are debugging labels, a **text-only** GMTM (zeros elsewhere) is
   enough to see whether the pickle is sane.
3. Do not quote audio-only / visual-only Acc2 as “the model understands
   faces/voice.” Those rows are weak regressions with a majority-ish binary
   score.
4. Zeroing is not the same as retraining a 1-stream transformer. A follow-up
   personal experiment would shrink `n_modalities` and drop the unused
   `trans[i][j]` modules.

## Re-running

Uncomment `train(...)` in `train_GMTM_glove.py` (20 epochs, early stop) or
the BERT twin (50 epochs). Write the CSV from the loop that is already
appended in the GloVe script. Keep the zeroing protocol if you want rows
that sit next to the existing table.
