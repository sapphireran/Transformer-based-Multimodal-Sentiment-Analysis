# References

Personal reading list for the modules in this repo. This is not an
official implementation of any of these papers.

## Datasets

- Amir Zadeh, Rowan Zellers, Eli Pincus, Louis-Philippe Morency.
  *MOSI: Multimodal Corpus of Sentiment Intensity and Subjectivity
  Analysis in Online Opinion Videos.* IEEE Intelligent Systems, 2016.
- Amir Zadeh, Paul Pu Liang, Soujanya Poria, Erik Cambria, Louis-Philippe
  Morency. *Multimodal Language Analysis in the Wild: CMU-MOSEI Dataset
  and Interpretable Dynamic Fusion Graph.* ACL 2018.

Feature sources used in the pickles:

- FACET 4.2 (visual action units / facial features)
- COVAREP (acoustic)
- GloVe 840B 300-d (Pennington, Socher, Manning, EMNLP 2014)
- BERT (Devlin et al., NAACL 2019) — aligned word / segment embeddings
  in `mosei_raw_bert.pkl` / `mosi_raw_bert.pkl`

SDK: [CMU-MultimodalSDK](https://github.com/A2Zadeh/CMU-MultimodalSDK).

## Fusion methods re-implemented here

- **TFN** — Amir Zadeh, Minghai Chen, Soujanya Poria, Erik Cambria,
  Louis-Philippe Morency. *Tensor Fusion Network for Multimodal Sentiment
  Analysis.* EMNLP 2017.
  Code comment in `models.py` points at
  [Justin1904/TensorFusionNetworks](https://github.com/Justin1904/TensorFusionNetworks).
- **LMF** — Zhun Liu, Ying Shen, Varun Bharadhwaj Lakshminarasimhan,
  Paul Pu Liang, Amir Zadeh, Louis-Philippe Morency.
  *Efficient Low-rank Multimodal Fusion with Modality-Specific Factors.*
  ACL 2018.
  Comment points at
  [Justin1904/Low-rank-Multimodal-Fusion](https://github.com/Justin1904/Low-rank-Multimodal-Fusion).
- **MulT-style cross-modal attention** — Yao-Hung Hubert Tsai, Shaojie Bai,
  Paul Pu Liang, J. Zico Kolter, Louis-Philippe Morency, Ruslan
  Salakhutdinov. *Multimodal Transformer for Unaligned Multimodal Language
  Sequences.* ACL 2019.
  The custom `TransformerEncoder` (sinusoidal positions, `x_in_k` / `x_in_v`)
  follows that pattern. It is not a guaranteed faithful port — see
  [quirks](quirks.md).

Early / late concat and the two `*FusionTransformer` wrappers are standard
baselines, not tied to a single paper.

## Metrics

MOSI/MOSEI papers typically report MAE, Pearson correlation, binary
accuracy / F1 (often excluding zeros), and 7-class accuracy. This repo’s
Acc7/Acc5 use **equal-width** bins on `[-3, 3]`
([metrics](metrics.md)), which may not match a paper that uses
`round(y)` into seven integers. Always check the bin rule before quoting
both numbers in the same sentence.

## Related personal code

Some MOSI scripts still mention `training_structures.Supervised_Learning`,
which is the trainer layout from
[MultiBench](https://github.com/pliang279/MultiBench) / MultiZoo. This
tree vendors a simplified copy as `train_and_test.py` instead.
