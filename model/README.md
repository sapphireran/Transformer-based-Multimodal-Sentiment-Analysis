# `model/` — training code and personal run logs

This directory is the original project root from the first upload. Scripts
assume you `cd model` so `data/` and `checkpoints/` resolve as relative
paths.

| Path | Role |
| --- | --- |
| [`models.py`](models.py) | Encoders, fusion, GMTM |
| [`train_and_test.py`](train_and_test.py) | `MultiFramework`, `train`, `test` |
| [`train_main_bert.py`](train_main_bert.py) | MOSEI BERT fusion sweep |
| [`train_main_glove.py`](train_main_glove.py) | MOSEI GloVe fusion sweep |
| [`train_GMTM_bert.py`](train_GMTM_bert.py) | GMTM BERT ablation |
| [`train_GMTM_glove.py`](train_GMTM_glove.py) | GMTM GloVe ablation |
| [`data/`](data) | Loaders + alignment scratch |
| [`checkpoints/`](checkpoints) | Where `.pt` files are written (gitignored) |
| [`results/`](results) | Checked-in CSV tables |
| [`mosi_test/`](mosi_test) | MOSEI → MOSI transfer eval |

Longer notes:

- Architecture: [`../docs/architecture.md`](../docs/architecture.md)
- How to rerun: [`../docs/reproduction.md`](../docs/reproduction.md)
- File-by-file: [`../docs/code-map.md`](../docs/code-map.md)

Dataset-free smoke tests live in [`../examples/`](../examples), not here.
They import `models.py` with `sys.path` pointed at this folder.
