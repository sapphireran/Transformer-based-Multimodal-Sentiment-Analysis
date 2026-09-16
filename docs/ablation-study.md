# Ablation study

GMTM is always constructed with three input streams. Ablation means
**zeroing** a stream, not shrinking the network. See
[datasets.md](datasets.md#ablation-zero-out).

The seven subsets:

```
text
audio
visual
text + audio
text + visual
audio + visual
text + audio + visual
```

Scripts: `train_GMTM_bert.py`, `train_GMTM_glove.py`. MOSI transfer of
those checkpoints: `mosi_test/mult_bert_mosi.py`,
`mosi_test/mult_glove_mosi.py` (and the ablation CSVs they wrote).

## MOSEI + BERT GMTM

From [`model/results/ablation_results.csv`](../model/results/ablation_results.csv).

| Modalities | MAE ↓ | Acc-7 | Acc-5 | Acc-2 | Corr | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| text | 0.5687 | 0.4728 | 0.5516 | 0.8404 | 0.7202 | 0.8741 |
| audio | 0.8306 | 0.4127 | 0.4134 | 0.6252 | 0.1124 | 0.7552 |
| visual | 0.8217 | 0.4012 | 0.4127 | 0.6293 | 0.2061 | 0.7693 |
| text+audio | 0.5659 | 0.4806 | 0.5596 | 0.8409 | 0.7227 | 0.8731 |
| text+visual | 0.5667 | 0.4838 | 0.5538 | 0.8324 | 0.7143 | 0.8697 |
| audio+visual | 0.8220 | 0.3951 | 0.4149 | 0.6274 | 0.2231 | 0.7670 |
| **text+audio+visual** | **0.5640** | 0.4827 | 0.5561 | **0.8429** | **0.7255** | **0.8777** |

Reading:

- Text-only is already better than every BERT fusion method in
  `main_results.csv` except it is slightly behind full GMTM. The extra
  two modalities buy about 0.005 MAE and a small F1 bump.
- Audio-only and vision-only sit near 0.82–0.83 MAE with correlation
  0.11–0.21. They are weak unimodal regressors on this feature set.
- Audio+visual without text does **not** recover: MAE 0.8220, essentially
  the worse of the two unimodal non-text runs.
- Adding audio to text helps a little more than adding vision to text on
  MAE / Corr; adding vision helps Acc-7 slightly more.

So the honest summary is: **GMTM is a text model that can use the other
two streams**, not a balanced three-way voter.

## MOSEI + GloVe GMTM

From [`model/results/ablation_glove_results.csv`](../model/results/ablation_glove_results.csv).

| Modalities | MAE ↓ | Acc-7 | Acc-5 | Acc-2 | Corr | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| text | 0.6616 | 0.4220 | 0.5091 | 0.8118 | 0.6557 | 0.8509 |
| audio | 0.8232 | 0.4127 | 0.4134 | 0.6362 | 0.2067 | 0.7668 |
| visual | 0.8139 | 0.3977 | 0.4185 | 0.6480 | 0.2256 | 0.7675 |
| text+audio | 0.7200 | 0.3503 | 0.4550 | 0.7760 | 0.5503 | 0.8186 |
| text+visual | 0.6612 | 0.4145 | 0.5207 | 0.7980 | 0.6459 | 0.8466 |
| audio+visual | 0.8141 | 0.3958 | 0.4205 | 0.6489 | 0.2345 | 0.7678 |
| **text+audio+visual** | **0.6251** | **0.4467** | **0.5317** | 0.8079 | **0.6714** | 0.8474 |

Different story from BERT:

- Full GMTM **does** beat text-only by a clear MAE margin (0.625 vs
  0.662). Non-text cues matter more when the text encoder is GloVe.
- `text+audio` is *worse* than text-only (MAE 0.720, Acc-7 0.350). That
  pair overfit or fought the weaker word vectors; it is the ugly row in
  this table.
- Audio / visual unimodal numbers are in the same band as BERT (~0.81–0.82
  MAE) — those streams never saw the text embedding, so they should match.

## MOSI transfer, BERT GMTM subsets

From [`model/mosi_test/ablation_mosi_results.csv`](../model/mosi_test/ablation_mosi_results.csv).
MOSEI-trained weights, MOSI evaluation pool (train+valid+test concat).

| Modalities | MAE ↓ | Acc-7 | Acc-5 | Acc-2 | Corr | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| text | 0.9363 | 0.3170 | 0.4127 | 0.7667 | 0.6482 | 0.7559 |
| audio | 1.3852 | 0.1681 | 0.1842 | 0.4509 | −0.1256 | 0.4532 |
| visual | 1.3741 | 0.1663 | 0.1837 | 0.5165 | −0.0278 | 0.6800 |
| text+audio | 0.9349 | 0.3055 | 0.3995 | 0.7786 | 0.6494 | 0.7715 |
| **text+visual** | **0.9044** | **0.3257** | 0.4104 | **0.7796** | **0.6623** | **0.7767** |
| audio+visual | 1.3711 | 0.1654 | 0.1846 | 0.5208 | 0.0226 | 0.6807 |
| text+audio+visual | 0.9493 | 0.3078 | 0.3981 | 0.7638 | 0.6403 | 0.7626 |

On MOSI, **dropping audio** (text+visual) is the best GMTM subset. Full
three-modality GMTM is slightly *worse* than text-only. Audio-only
correlation goes negative — the COVAREP statistics did not transfer.

## MOSI transfer, GloVe GMTM subsets

From [`model/mosi_test/ablation_mosi_glove_results.csv`](../model/mosi_test/ablation_mosi_glove_results.csv).

| Modalities | MAE ↓ | Acc-7 | Acc-5 | Acc-2 | Corr | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| text | 1.0085 | 0.3014 | 0.3903 | 0.7317 | 0.5959 | 0.7343 |
| audio | 1.3799 | 0.1681 | 0.1842 | 0.4696 | −0.0720 | 0.5513 |
| visual | 1.3692 | 0.1667 | 0.1832 | 0.5036 | 0.0409 | 0.6177 |
| text+audio | 1.2863 | 0.1745 | 0.2400 | 0.6033 | 0.3524 | 0.5141 |
| text+visual | 1.0057 | 0.2739 | 0.3729 | 0.7235 | 0.5967 | 0.7446 |
| audio+visual | 1.3734 | 0.1672 | 0.1837 | 0.4998 | −0.0042 | 0.6024 |
| **text+audio+visual** | **0.9748** | **0.3152** | **0.3995** | **0.7355** | **0.6177** | 0.7381 |

Here the full model *does* win MAE / Acc-7 / Corr, recovering the
MOSEI-GloVe pattern: weaker text leaves room for the other streams, but
`text+audio` is again toxic.

## How to rerun

```bash
cd model
# uncomment train() in train_GMTM_bert.py if you need new weights
python train_GMTM_bert.py
```

`train_GMTM_bert.py` currently loops only
`['text', 'audio', 'visual']` (the other six combinations are commented).
`train_GMTM_glove.py` still loops all seven.

Checkpoint names use `'+'.join(modalities)`:

```
checkpoints/ablation/model_text+audio+visual.pt
checkpoints/ablation/model_glove_text.pt
```

## What this ablation does *not* tell you

- It does not compare GMTM to ConcatLate under the same zero-out protocol.
  The fusion sweep always sees three live modalities.
- It does not test modality dropout at train time
  (`HParams.modality_dropout` is unused).
- It does not re-tune `embed_dim` per subset. A dedicated text-only
  transformer might beat “GMTM with two zero streams.”
