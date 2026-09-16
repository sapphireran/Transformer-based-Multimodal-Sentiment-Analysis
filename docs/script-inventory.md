# Script inventory

Personal study map of every Python entry point. Paths are from the
repository root.

## Core library

| path | role |
| --- | --- |
| `model/models.py` | Encoders, fusion modules, GMTM, positional embeddings |
| `model/train_and_test.py` | `MultiFramework`, `train`, `test`, `single_test` |
| `model/metrics.py` | Dataset-free MAE / Acc-7 / Acc-2 / F1 (examples + tests) |
| `model/data/get_dataloader.py` | MOSI/MOSEI Dataset, packed/max-pad collate, ablations |

`models.py` defines **two** things named `Linear`: a `nn.Module` subclass
and, later, a factory function. After import, `Linear` is the function.
Call the class by editing the file or by using `nn.Linear`.

## MOSEI training / eval

| path | language | what it does as committed |
| --- | --- | --- |
| `model/train_main_bert.py` | BERT | trains six baselines for 1 epoch, writes `main_results.csv` |
| `model/train_main_glove.py` | GloVe | loads `glove_*.pt`, extra `test()` kwargs (currently incompatible) |
| `model/train_GMTM_bert.py` | BERT | loads trimodal GMTM checkpoint, train() commented out |
| `model/train_GMTM_glove.py` | GloVe | walks seven modality subsets, extra `test()` kwargs |

## MOSI transfer eval

| path | language | what it does as committed |
| --- | --- | --- |
| `model/mosi_test/train_mosi_bert.py` | BERT | loads MOSEI checkpoints, merged MOSI loader |
| `model/mosi_test/train_mosi_glove.py` | GloVe | imports missing MultiBench `training_structures` |
| `model/mosi_test/mult_bert_mosi.py` | BERT | GMTM transfer, train() absent |
| `model/mosi_test/mult_glove_mosi.py` | GloVe | GMTM transfer for GloVe |

## Data conversion (not run by examples)

| path | notes |
| --- | --- |
| `model/data/MOSEI/get_mosei.py` | SDK align + HDF5 dump; Windows path; file name says MOSEI, constants say MOSI |
| `model/data/MOSEI/get_dataset.ipynb` | BERT / GloVe pickle export notebook |
| `model/data/MOSI/get_dataset.ipynb` | MOSI counterpart |
| `model/data/analysis_dataset.ipynb` | exploratory plots |
| `model/results/plot.ipynb` | result figures |

## Result CSVs

Canonical copies live in `model/results/`. Duplicates at `model/*.csv`
are the same BERT/GloVe MOSEI tables. MOSI tables live only under
`model/mosi_test/`.

## Examples lab

| path | role |
| --- | --- |
| `examples/msa_lab/synthetic.py` | correlated `[B,T,F]` generator |
| `examples/msa_lab/fusion_zoo.py` | CPU concat / TFN / LMF / transformer / GMTM |
| `examples/msa_lab/collate.py` | packed vs max-pad |
| `examples/msa_lab/toy_train.py` | short GMTM overfit |
| `examples/01_*.py` … `06_*.py` | runnable walkthroughs |
| `scripts/run_examples.py` | runs 01–06 in order |

## Tests

| path | covers |
| --- | --- |
| `tests/test_metrics.py` | uniform bins, binary F1, CSV columns |
| `tests/test_synthetic.py` | shapes, ablation zeros, length masks |
| `tests/test_fusion_shapes.py` | zoo output ranks |
| `tests/test_gmtm_train.py` | MAE drop on one batch |
| `tests/test_collate.py` | packed lengths vs dense pad |
