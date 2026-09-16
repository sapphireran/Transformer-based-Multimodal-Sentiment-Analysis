# Glossary

Short personal glossary for names that appear in the scripts and CSVs.

| Term | Meaning in this repo |
| --- | --- |
| **GMTM** | `GatedMultiTransfomerModel` — pairwise cross-modal transformers + softmax modality weights + sigmoid gates + attention pooling. The class name is missing an "o" (`Transfomer`); it is kept as spelled in `models.py`. |
| **MulT** | Tsai et al., *Multimodal Transformer*. GMTM's pairwise `query=i, key/value=j` stacks are in that family. |
| **TFN** | Tensor Fusion Network. Implemented as `TensorFusion`. |
| **LRTF** | Low-rank multimodal fusion. Implemented as `LowRankTensorFusion`. |
| **FACET** | Commercial facial-action feature set; 35-D `vision` stream. |
| **COVAREP** | Open acoustic feature set; 74-D `audio` stream. `-inf` values are zeroed. |
| **GloVe 840B** | 300-D Common-Crawl word vectors. File is *not* stored in git. |
| **BERT text** | 768-D word-aligned language features in the `*_bert.pkl` files. |
| **MOSEI** | CMU multimodal opinion-level corpus (~23k). Main training set here. |
| **MOSI** | Smaller predecessor (~2.2k). Used as transfer / pooled eval. |
| **Aligned** | Visual / audio cropped from the first non-zero *text* frame so the three streams share a word timeline. |
| **`max_pad`** | Truncate/pad every clip to 50 steps; required by GMTM and TransformerEarly. |
| **Packed** | `is_packed=True`: LSTM/GRU see `[sequence, lengths]` via `pack_padded_sequence`. |
| **Acc-7 / Acc-5** | Accuracy after cutting `[-3, 3]` into 7 or 5 **equal-width** bins. Not rounded MOSI integers. |
| **Acc-2** | Accuracy of `sign(y)` vs `sign(ŷ)` after dropping `y == 0`. |
| **F1** | `sklearn` binary F1 on that same non-neutral split (positive = `y > 0`). |
| **Corr** | Pearson `r` between raw `y` and `ŷ`. |
| **MAE** | Mean absolute error; also the training loss. |
| **Early fusion** | Merge raw (or lightly projected) sequences, then one temporal model. |
| **Late fusion** | Unimodal temporal models, then merge vectors. |
| **Modality ablation** | Zero one or two streams inside a still-3-input GMTM. |
| **Pooled MOSI** | `get_mosi_dataloader` concatenates train+valid+test. |
| **`MultiFramework`** | Thin wrapper: encode each stream, fuse, head. |
| **`Identity`** | No-op module; used when GMTM already includes the head. |
| **`HParams`** | Small class of GMTM knobs in the `train_GMTM_*.py` scripts. |
| **Uniform bins** | `split_uniform_7` / `split_uniform_5` in `train_and_test.py`. |

Typo watch: CSV header `GatedMultiTransfomer`, class
`GatedMultiTransfomerModel`, filename `train_GMTM_*.py`. They all
refer to the same model.
