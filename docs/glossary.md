# Glossary

Terms as they are used **in this repository**, which is not always the same
as a paper’s wording.

| Term | Meaning here |
| --- | --- |
| Acc2 | Binary accuracy after dropping gold zeros and thresholding scores at 0 |
| Acc5 / Acc7 | Accuracy on uniform-width bins of `[-3, 3]` (5 or 7 bins) |
| Aligned | Visual / audio cropped using the first nonzero text frame |
| BERT text | 768-D word-aligned embeddings in `*_bert.pkl` |
| COVAREP | Acoustic feature set; 74-D in these pickles |
| Early fusion | Concatenate raw (or lightly projected) features before a shared model |
| Facet | Visual facial feature set; 35-D in these pickles |
| F1 | Binary F1 on the same pair as Acc2 (`sklearn`, `average='binary'`) |
| Fusion sweep | The six methods in `train_main_*.py` |
| GMTM | `GatedMultiTransfomerModel` — pairwise transformers + gates + pooling |
| GloVe text | 300-D word-aligned embeddings in `*_glove.pkl` |
| Homogeneous 1 | Extra constant feature prepended before TFN / LMF products |
| `is_packed` | `MultiFramework` / `train()` flag: encoders receive `[tensor, lengths]` |
| Late fusion | Encode each modality, then combine vectors or encoded sequences |
| LMF | `LowRankTensorFusion` — factorized tensor fusion |
| `max_pad` | Collate clips/pads every utterance to `max_seq_len` (50) |
| MOSEI | CMU multimodal opinion / emotion / sentiment dataset (larger) |
| MOSI | CMU multimodal sentiment dataset (smaller, used here for transfer) |
| `MultiFramework` | `encoders + fusion + head` container |
| Packed batch | `_process_1` output: padded to batch-max length plus length tensors |
| TFN | `TensorFusion` — full outer-product tensor fusion |
| Transfer | Score a MOSEI-trained checkpoint on MOSI loaders |
| Uniform split | Equal-width bins on `[-3, 3]`, not integer rounding |
| `z_norm` | Per-utterance, per-feature z-score over time |

## File-name crumbs

| Token | Refers to |
| --- | --- |
| `OT` in `traindata_OT` | “ordinary / packed” loaders (name is just a local variable) |
| `TE` in `traindata_TE` | Transformer-early / max-padded loaders |
| `Concat` in MOSI CSVs | Late concat (`ConcatLate`) |
| `model_text+audio.pt` | GMTM BERT trained with visual zeroed |
| `glove_TransformerLate.pt` | GloVe fusion-sweep checkpoint |

## Shape crumbs

| Symbol | Typical value |
| --- | --- |
| `B` | 32 (train) or 4–8 in examples |
| `T` | ≤ 50 |
| `F_v, F_a, F_t` | 35, 74, 768 or 300 |
| GMTM `embed_dim` | 64 |
| Early / late transformer `embed_dim` | 32 |
