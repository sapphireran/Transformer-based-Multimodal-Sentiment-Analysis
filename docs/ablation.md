# Modality ablation

The GMTM scripts keep a **fixed 3-stream graph** and zero out unused modalities. That is different from training a smaller model with fewer `n_modalities`.

## Combinations

`train_GMTM_glove.py` (and the MOSI GloVe tester) iterate:

```
['text']
['audio']
['visual']
['text', 'audio']
['text', 'visual']
['audio', 'visual']
['text', 'audio', 'visual']
```

`train_GMTM_bert.py` in this checkout only has the full triple uncommented; the CSV `model/results/ablation_results.csv` still contains all seven rows from an earlier run.

## How zeros are injected

`filter_modalities_list` in `get_dataloader.py` maps names to batch slots:

```
visual → index 0, shape (50, 35)
audio  → index 1, shape (50, 74)
text   → index 2, shape (50, 768) or (50, 300)
```

A dropped slot becomes `torch.zeros(shape)`. The clip id / label column is kept. GMTM still builds a 3×3 cross-attention grid, so a zeroed stream still has parameters and still produces a projected sequence. The ablation therefore answers:

> If this trained 3-input model is given silence / blank face / no text at test (or train) time, how much does it lose?

It does **not** answer “what is the best text-only transformer we could train from scratch.”

## Reading the MOSEI BERT ablation

From `model/results/ablation_results.csv`:

| Inputs | MAE | Acc-2 | Corr |
| --- | --- | --- | --- |
| text | 0.569 | 0.840 | 0.720 |
| audio | 0.831 | 0.625 | 0.112 |
| visual | 0.822 | 0.629 | 0.206 |
| text+audio | 0.566 | 0.841 | 0.723 |
| text+visual | 0.567 | 0.832 | 0.714 |
| audio+visual | 0.822 | 0.627 | 0.223 |
| all three | **0.564** | **0.843** | **0.726** |

Text is necessary. Non-text pairs look like the weaker unimodal rows. Adding audio and visual to text is a small but consistent MAE / Corr gain.

## Reading the GloVe ablation

`ablation_glove_results.csv` is noisier: `text+audio` is **worse** than text-only (MAE 0.720 vs 0.662). The full triple is the best GloVe GMTM row (MAE 0.625). That is a reminder that zero-mask ablation + a shared grid can hurt when a weak stream injects noise.

## MOSI transfer

`mosi_test/ablation_mosi_*.csv` score MOSEI-trained GMTM weights on merged MOSI. Text still dominates. Audio-only correlation on MOSI BERT is **negative** (−0.13). Do not treat unimodal audio/visual GMTM as a usable MOSI sentiment model.

## Reproducing an ablation cell

```bash
cd model
# after pickles exist and train() is uncommented
python train_GMTM_glove.py
```

Or, without data, run the zero-mask demo:

```bash
python examples/gmtm_forward_pass.py --ablate text
```

That builds a synthetic batch, zeros non-text streams the same way the dataloader would, and prints output shape plus a few predictions.
