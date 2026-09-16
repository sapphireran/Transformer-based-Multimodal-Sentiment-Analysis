# Experiments and result tables

Numbers below are copied from the CSVs already in this repository. They
were produced by the training / eval scripts under `model/`, not by the
CPU examples lab. The examples never overwrite these files.

## Protocol in one paragraph

MOSEI is the main benchmark. Language is either BERT-768 or GloVe-300;
visual is Facet 4.2 (35-d); audio is COVAREP (74-d). Models train with
AdamW, L1 loss, `lr=1e-4`. Checkpoints are full `MultiFramework` objects
under `model/checkpoints/`. MOSI numbers come from **those same MOSEI
checkpoints** scored on a loader that concatenates every MOSI split
(`get_mosi_dataloader`). Ablation tables keep three input slots and zero
the dropped streams.

## MOSEI · BERT language

Source: `model/results/main_results.csv` and
`model/results/ablation_results.csv`.

| fusion | MAE ↓ | Acc-7 ↑ | Acc-5 ↑ | Acc-2 ↑ | Corr ↑ | F1 ↑ |
| --- | --- | --- | --- | --- | --- | --- |
| ConcatEarly | 0.6178 | 0.4424 | 0.5221 | 0.8084 | 0.6652 | 0.8475 |
| ConcatLate | 0.6148 | 0.4508 | 0.5162 | 0.8084 | 0.6637 | 0.8460 |
| LowRankTensorFusion | 0.5970 | 0.4585 | 0.5409 | 0.8305 | 0.6925 | 0.8638 |
| TensorFusion | 0.6022 | 0.4445 | 0.5256 | 0.8244 | 0.6776 | 0.8626 |
| TransformerEarly | 0.6057 | 0.4514 | 0.5327 | 0.8206 | 0.6788 | 0.8604 |
| TransformerLate | 0.5846 | 0.4675 | 0.5460 | 0.8393 | 0.7041 | 0.8699 |
| GMTM (text+audio+visual) | **0.5640** | 0.4827 | 0.5561 | **0.8429** | **0.7255** | **0.8777** |

GMTM is the best row on every reported column except Acc-7 / Acc-5, where
it is essentially tied with the text+audio and text+visual ablations
(those two slightly win Acc-7 / Acc-5; trimodal wins MAE / Acc-2 / Corr /
F1).

### GMTM modality ablation, MOSEI BERT

| modalities | MAE | Acc-7 | Acc-5 | Acc-2 | Corr | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| text | 0.5687 | 0.4728 | 0.5516 | 0.8404 | 0.7202 | 0.8741 |
| audio | 0.8306 | 0.4127 | 0.4134 | 0.6252 | 0.1124 | 0.7552 |
| visual | 0.8217 | 0.4012 | 0.4127 | 0.6293 | 0.2061 | 0.7693 |
| text+audio | 0.5659 | 0.4806 | 0.5596 | 0.8409 | 0.7227 | 0.8731 |
| text+visual | 0.5667 | 0.4838 | 0.5538 | 0.8324 | 0.7143 | 0.8697 |
| audio+visual | 0.8220 | 0.3951 | 0.4149 | 0.6274 | 0.2231 | 0.7670 |
| text+audio+visual | 0.5640 | 0.4827 | 0.5561 | 0.8429 | 0.7255 | 0.8777 |

Language dominates. Non-text pairs collapse toward the audio-only /
visual-only band (MAE ≈ 0.82, Corr ≈ 0.22). Adding audio or visual to
text is a small but consistent MAE drop (0.5687 → 0.5640). That is the
whole point of keeping COVAREP and Facet in the graph: they are weak
alone and slightly useful as gated context.

## MOSEI · GloVe language

Source: `model/results/glove_results.csv` and
`model/results/ablation_glove_results.csv`.

| fusion | MAE ↓ | Acc-7 ↑ | Acc-5 ↑ | Acc-2 ↑ | Corr ↑ | F1 ↑ |
| --- | --- | --- | --- | --- | --- | --- |
| ConcatEarly | 0.6782 | 0.4123 | 0.4919 | 0.7664 | 0.5942 | 0.8206 |
| ConcatLate | 0.6329 | 0.4373 | 0.5187 | 0.7878 | 0.6528 | 0.8325 |
| LowRankTensorFusion | **0.6174** | 0.4458 | 0.5287 | **0.8107** | 0.6704 | **0.8523** |
| TensorFusion | 0.6337 | 0.4336 | 0.5141 | 0.8038 | 0.6435 | 0.8457 |
| TransformerEarly | 0.6649 | 0.4207 | 0.5020 | 0.7821 | 0.6063 | 0.8373 |
| TransformerLate | 0.6287 | 0.4398 | 0.5076 | 0.7975 | 0.6397 | 0.8425 |
| GMTM (text+audio+visual) | 0.6251 | **0.4467** | **0.5317** | 0.8079 | **0.6714** | 0.8474 |

Switching BERT → GloVe costs the gated model about 0.06 MAE. LMF is the
best GloVe *baseline* on MAE; GMTM still leads Acc-7 / Acc-5 / Corr.
Text-only GMTM (0.6616 MAE) is worse than trimodal GloVe GMTM (0.6251),
so non-text streams help more when language is bag-of-words than when it
is BERT.

GloVe ablation oddity: `text+audio` MAE 0.7200 is *worse* than text-only
0.6616. The gated mix can hurt when a weak stream is not fully suppressed.
The trimodal row recovers, which is why the sigmoid gate is part of the
story rather than a guaranteed win on every subset.

## MOSI transfer · BERT

Source: `model/mosi_test/mosi_bert_results.csv` and
`model/mosi_test/ablation_mosi_results.csv`. These score MOSEI-trained
weights on the merged MOSI loader.

| fusion | MAE ↓ | Acc-7 ↑ | Acc-5 ↑ | Acc-2 ↑ | Corr ↑ | F1 ↑ |
| --- | --- | --- | --- | --- | --- | --- |
| ConcatEarly | 0.9448 | 0.2839 | 0.3574 | 0.7604 | 0.6284 | 0.7704 |
| Concat (late) | 1.0056 | 0.2757 | 0.3574 | 0.7339 | 0.5726 | 0.7287 |
| LowRankTensorFusion | 0.9525 | 0.3156 | 0.4063 | 0.7676 | 0.6149 | 0.7626 |
| TensorFusion | 0.9974 | 0.2616 | 0.3514 | 0.7705 | 0.6352 | 0.7733 |
| TransformerEarly | 0.9534 | 0.2868 | 0.3610 | 0.7681 | 0.6384 | 0.7782 |
| TransformerLate | **0.8986** | **0.3230** | 0.3894 | **0.7959** | **0.6748** | **0.7855** |
| GMTM | 0.9493 | 0.3078 | 0.3981 | 0.7638 | 0.6403 | 0.7626 |

On this transfer dump, late transformers beat GMTM. MOSI is smaller and
more domain-shifted from MOSEI's broader topics; a heavy cross-modal
stack that overfits MOSEI language can lag a simpler late encoder. The
BERT MOSI ablation still shows language carrying the task (text MAE
0.9363 vs audio 1.3852).

## MOSI transfer · GloVe

Source: `model/mosi_test/mosi_glove_results.csv` and
`model/mosi_test/ablation_mosi_glove_results.csv`.

| fusion | MAE ↓ | Acc-7 ↑ | Acc-5 ↑ | Acc-2 ↑ | Corr ↑ | F1 ↑ |
| --- | --- | --- | --- | --- | --- | --- |
| ConcatEarly | 1.1702 | 0.1932 | 0.2458 | 0.6869 | 0.4774 | 0.6840 |
| Concat (late) | 1.0797 | 0.2267 | 0.2943 | 0.7064 | 0.5520 | 0.7139 |
| LowRankTensorFusion | 1.0825 | 0.2332 | 0.3046 | 0.7336 | 0.5559 | 0.7317 |
| TensorFusion | 1.1693 | 0.2158 | 0.2849 | 0.6737 | 0.4958 | 0.6418 |
| TransformerEarly | 1.1941 | 0.1951 | 0.2483 | 0.6938 | 0.4822 | 0.6909 |
| TransformerLate | 1.0855 | 0.2263 | 0.3097 | 0.7245 | 0.5303 | 0.7271 |
| GMTM | **0.9748** | **0.3152** | **0.3995** | **0.7355** | **0.6177** | **0.7381** |

Here the gated model *does* win every column. GloVe transfer is a
weaker language prior, so pairwise cross-modal attention has more room
to help. The GloVe MOSI ablation agrees: trimodal 0.9748 vs text 1.0085
vs audio/visual ≈ 1.37.

## How to read the four-quadrant picture

1. **BERT + in-domain MOSEI:** GMTM edges out late transformers; text is
   already strong, non-text is a small gated bonus.
2. **GloVe + in-domain MOSEI:** LMF is a very strong baseline; GMTM is
   competitive, not dominant on MAE.
3. **BERT + MOSI transfer:** late transformers generalise better.
4. **GloVe + MOSI transfer:** GMTM generalises better.

That pattern is why the repo keeps both language stacks and both
datasets instead of collapsing to a single "best model" table.

## Training-script caveats that affect these numbers

* `train_main_bert.py` currently sets `total_epochs=1`. The CSV above is
  from an earlier longer run; re-running the file as committed will not
  match the table. See `docs/reproduction.md`.
* GMTM train loops in `train_GMTM_*.py` have the `train(...)` call
  commented out and only load checkpoints.
* `train_main_glove.py` / `train_GMTM_glove.py` / several MOSI scripts
  pass `dataset=` and `no_robust=` into `test()`, which the local
  `train_and_test.test` does not accept. Those files are snapshots of an
  older MultiBench-style signature.
* MOSI BERT eval uses fusion name `Concat` for late concat; MOSEI uses
  `ConcatLate`. Same module, different CSV label.
