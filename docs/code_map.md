# Code map

Personal index of every tracked file under `model/` and the documentation
/ example tree added beside it.

## Training and models

| File | What it is |
| --- | --- |
| `model/models.py` | All neural modules: GMTM, TFN, LMF, concat, transformer encoders, LSTM/GRU/MLP |
| `model/train_and_test.py` | `MultiFramework`, `train`, `test`, `single_test`, Acc7/Acc5 helpers |
| `model/train_main_bert.py` | MOSEI BERT sweep over six fusion names |
| `model/train_main_glove.py` | Same sweep, GloVe widths; train loop commented |
| `model/train_GMTM_bert.py` | GMTM BERT; currently test-only on the full trio |
| `model/train_GMTM_glove.py` | GMTM GloVe ablation over seven modality subsets |

## Data

| File | What it is |
| --- | --- |
| `model/data/get_dataloader.py` | Pickle loaders, `Affectdataset`, ablation zero-masking, MOSI merge |
| `model/data/MOSEI/get_mosei.py` | CMU-MultimodalSDK align + HDF5 dump (Windows paths) |
| `model/data/MOSEI/get_dataset.ipynb` | Interactive SDK session |
| `model/data/MOSI/get_dataset.ipynb` | Interactive SDK session |
| `model/data/analysis_dataset.ipynb` | Pickle introspection + label histograms |
| `model/data/readme.md` | Feature filename checklist |
| `model/data/glove.840B.300d.txt` | Empty placeholder for the GloVe file |

## Results and checkpoints

| File | What it is |
| --- | --- |
| `model/main_results.csv` | MOSEI BERT classical sweep |
| `model/glove_results.csv` | MOSEI GloVe classical sweep |
| `model/ablation_results.csv` | MOSEI BERT GMTM subsets |
| `model/ablation_glove_results.csv` | MOSEI GloVe GMTM subsets |
| `model/results/*` | Copies of the four CSVs plus `plot.ipynb` |
| `model/checkpoints/readme.md` | "put main sweep weights here" |
| `model/checkpoints/ablation/readme.md` | "put GMTM weights here" |

## MOSI transfer

| File | What it is |
| --- | --- |
| `model/mosi_test/train_mosi_bert.py` | Score MOSEI BERT fusion ckpts on MOSI |
| `model/mosi_test/train_mosi_glove.py` | Same for GloVe; broken MultiBench import |
| `model/mosi_test/mult_bert_mosi.py` | Score GMTM BERT ckpts on MOSI |
| `model/mosi_test/mult_glove_mosi.py` | Score GMTM GloVe ckpts on MOSI |
| `model/mosi_test/mosi_bert_results.csv` | Transfer table, BERT |
| `model/mosi_test/mosi_glove_results.csv` | Transfer table, GloVe |
| `model/mosi_test/ablation_mosi_results.csv` | GMTM BERT subsets on MOSI |
| `model/mosi_test/ablation_mosi_glove_results.csv` | GMTM GloVe subsets on MOSI |

## Documentation and examples (this branch)

| Path | What it is |
| --- | --- |
| `docs/` | Architecture, data, metrics, experiments, results, lab notes |
| `examples/lib/synthetic.py` | BERT/GloVe-shaped random batches + a learnable toy mixture |
| `examples/lib/metrics.py` | NumPy port of MAE / Corr / Acc7 / Acc5 / Acc2 / F1 |
| `examples/lib/repo.py` | Path helpers so examples run from any cwd |
| `examples/forward_pass_demo.py` | CPU forward of GMTM + classical fusions |
| `examples/fusion_shape_walkthrough.py` | Rank contract for each module |
| `examples/toy_train_gmtm.py` | Short GMTM overfit on synthetic labels |
| `examples/tiny_affect_dataset.py` | Minimal pickle + Dataset matching `Affectdataset` keys |
| `examples/summarize_results.py` | Ranked reprint of every CSV |
| `examples/metric_walkthrough.py` | Bin edges and hand-checked metric cases |
| `examples/run_all.py` | Runs the example suite and writes a log |
| `examples/README.md` | How to run the demos |
| `tests/test_docs_examples.py` | Assertions around metrics, shapes, and CSV parsing |

## Name collisions and leftovers worth knowing

- `models.py` defines `class Linear` and later `def Linear(...)`. The
  function wins. Use `CustomLinear` if you need the class.
- `GatedMultiTransfomerModel` is misspelled in the class name (one "o" in
  Transformer) and that spelling is what the CSVs use as well
  (`GatedMultiTransfomer`).
- `train_and_test.test` advertises `dataset=` / `no_robust=` in some call
  sites, but the function signature only accepts
  `model, test_dataloaders_all, is_packed, criterion, input_to_float`.
  Extra keywords would TypeError if those scripts are run as-is.
- `train_mosi_glove.py` imports `training_structures.Supervised_Learning`,
  which is not in this repository.
