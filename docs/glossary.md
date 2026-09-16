# Glossary

Terms as they are used in this personal MOSI / MOSEI project. When a paper
uses the same word differently (especially Acc-7), the definition here wins
for numbers stored under `model/results/`.

| Term | Meaning in this repo |
| --- | --- |
| **Acc-2** | Binary accuracy on clips with gold `y ≠ 0`. Both gold and prediction are positive iff the value is `> 0`. Implemented in `eval_affect`. Neutral clips still affect MAE / Corr / Acc-5 / Acc-7. |
| **Acc-5 / Acc-7** | Accuracy after cutting the interval [−3, 3] into 5 or 7 *equal* bins via `np.digitize`. Not the non-uniform “has5 / has7” scheme some MOSI papers publish. |
| **Aligned** | `Affectdataset(aligned=True)` finds the first non-zero **text** frame and slices vision, audio, and text from that index so the three streams share a start. |
| **Attention pooling** | `Linear(F → 1)` over time, softmax, weighted sum. GMTM uses this after concatenating gated modality sequences. |
| **BERT pickle** | Aligned text features of width 768 (`mosei_raw_bert.pkl`, `mosi_raw_bert.pkl`). |
| **COVAREP** | 74-D acoustic features (pitch, formants, glottal, etc.) stored as the `audio` array. |
| **Cross-modal encoder** | A `TransformerEncoder` inside GMTM whose query comes from modality `i` and whose key/value come from modality `j`. There is one for every ordered pair `(i, j)`. |
| **Early fusion** | Concatenate raw sequences on the feature axis, then one temporal model (`ConcatEarly`, `EarlyFusionTransformer`). |
| **FACET 4.2** | 35-D facial action / expression features stored as the `vision` array. |
| **Gate** | Per-modality `σ(W h) ⊙ h` applied after the softmax mix of cross-attended sequences. This is the “gated” in GMTM. |
| **GMTM** | Gated Multi-Transformer Model: project each stream, pairwise cross-modal transformers, softmax modality weights, sigmoid gate, attention pool, MLP head. |
| **GloVe pickle** | Aligned text features of width 300 from `glove.840B.300d` (`mosei_raw_glove.pkl`). |
| **Identity encoder** | `nn.Module` that returns its input. GMTM scripts put `Identity` in every encoder slot and in the head so the fusion module owns the graph. |
| **Late fusion** | Encode each modality first, then merge the codes (`ConcatLate`, `LateFusionTransformer`, TFN, LRTF). |
| **LRTF** | Low-rank tensor fusion. Scripts use rank 32 and a homogeneous `+1` coordinate per modality. |
| **MAE** | Mean absolute error. Training objective (`L1Loss`) and a test column. Lower is better. |
| **max_pad** | Truncate and zero-pad every sequence to length 50 (`_process_2`). Required by GMTM and early transformer. |
| **MOSEI** | CMU multimodal opinion / sentiment set used for *training* in this repo. |
| **MOSI** | Smaller CMU set. Scripts under `model/mosi_test/` score MOSEI-trained weights on a **merged** MOSI train+valid+test loader. |
| **MultiFramework** | Container in `train_and_test.py`: `encoders + fusion + head`. This is what `torch.save` writes to `checkpoints/*.pt`. |
| **Packed batch** | Variable-length tensors plus lengths, so LSTM/GRU can call `pack_padded_sequence`. `is_packed=True` in `train()` / `test()`. |
| **Pearson Corr** | `scipy.stats.pearsonr` between gold and prediction on the full evaluation set (neutrals included). |
| **TFN** | Full outer-product tensor fusion. Output width is `Π (F_i + 1)`; BERT head input is 128000. |
| **Unimodal zeroing** | Ablation style used by `get_ablation_dataloader`: keep three input slots, replace unused modalities with `torch.zeros(50, F)`. GMTM is never rebuilt with `n_modalities=1`. |
| **Uniform bin** | Equal-width partition of [−3, 3]. Step is `6/7` for Acc-7 and `6/5` for Acc-5. |

Related long-form notes: [metrics.md](metrics.md), [architecture.md](architecture.md),
[datasets.md](datasets.md).
