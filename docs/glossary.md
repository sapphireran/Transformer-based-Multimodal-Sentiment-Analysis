# Glossary

| term | meaning in this repo |
| --- | --- |
| Acc-2 | Binary accuracy after mapping scores with `> 0`, usually dropping true zeros |
| Acc-5 / Acc-7 | Accuracy on equal-width bins of `[-3, 3]` |
| Affectdataset | PyTorch `Dataset` wrapping a MOSI/MOSEI pickle |
| BERT pack | Language width 768 |
| COVAREP | Acoustic feature set, 74-d, `-inf` voicing flags zeroed on load |
| CSD | CMU SDK computational-sequence file |
| Facet 4.2 | Visual face features, 35-d |
| F1 | Binary F1 on the same split as Acc-2 |
| GMTM | `GatedMultiTransfomerModel` — pairwise transformers + mix + sigmoid gate |
| GloVe pack | Language width 300 (`glove.840B.300d`) |
| LMF | Low-rank multimodal fusion (`LowRankTensorFusion`) |
| MAE | Mean absolute error; also the training loss |
| max-pad | Collate that crops/pads every clip to length 50 (`_process_2`) |
| MOSI | CMU small movie-review sentiment set; used here as transfer |
| MOSEI | CMU larger in-the-wild sentiment set; main tables |
| MulT | Tsai et al. Multimodal Transformer; GMTM's pairwise attention is in that family |
| packed | Collate that keeps per-clip lengths for `pack_padded_sequence` (`_process_1`) |
| TFN | Tensor Fusion Network (full outer product) |
| toy pack | Synthetic widths 8 / 12 / 16 used by the examples lab |
