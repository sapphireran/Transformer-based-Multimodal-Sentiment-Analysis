# Setup (personal workstation notes)

These notes match how the training scripts are written today, not an idealized
package layout. The original experiment scripts assume you are inside `model/`
and that PyTorch can see a CUDA device. The new examples under `examples/` are
CPU-friendly and do not require the raw datasets.

## Repository layout

```
.
├── README.md
├── docs/                         # this documentation
├── examples/                     # synthetic demos that do not need MOSI/MOSEI
├── model/
│   ├── models.py                 # fusion modules + GatedMultiTransfomerModel
│   ├── train_and_test.py         # MultiFramework, train(), test()
│   ├── train_main_bert.py        # MOSEI BERT fusion sweep
│   ├── train_main_glove.py       # MOSEI GloVe fusion sweep
│   ├── train_GMTM_bert.py        # GMTM BERT (train commented; test path live)
│   ├── train_GMTM_glove.py       # GMTM GloVe ablation over modality subsets
│   ├── data/get_dataloader.py    # pickle → DataLoader
│   ├── data/MOSEI/               # expected mosei_raw_{bert,glove}.pkl
│   ├── data/MOSI/                # expected mosi_raw_{bert,glove}.pkl
│   ├── checkpoints/              # *.pt from main sweeps
│   ├── checkpoints/ablation/     # GMTM *.pt
│   ├── results/                  # copied CSVs + plot.ipynb
│   └── mosi_test/                # MOSEI → MOSI transfer scripts + CSVs
└── requirements.txt
```

Working directory for the original scripts is **`model/`**, not the repo root.
Relative paths such as `data/MOSEI/mosei_raw_bert.pkl` and
`checkpoints/ConcatEarly.pt` are resolved from there.

## Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Minimum useful packages for reading results and running synthetic examples:

- `torch`
- `numpy`
- `scipy` (Pearson correlation in `train_and_test.py`)
- `scikit-learn` (accuracy / F1)
- `tqdm`
- `matplotlib` (confusion matrix inside `single_test`)
- `memory-profiler` (imported by `train_and_test.py` even if you only call `test`)

Rebuilding pickles from CMU-MultimodalSDK `.csd` files additionally needs
`mmsdk` and `h5py`. `model/data/get_dataloader.py` also imports `torchtext`
even though the current pickle path does not use it; install it only if you
import that module as-is.

## Data files the training scripts expect

None of these pickles are in git (they are large). The loader will raise
`FileNotFoundError` until you place them:

| Path relative to `model/` | Text encoder | Typical text width |
| --- | --- | --- |
| `data/MOSEI/mosei_raw_bert.pkl` | BERT | 768 |
| `data/MOSEI/mosei_raw_glove.pkl` | GloVe 840B | 300 |
| `data/MOSI/mosi_raw_bert.pkl` | BERT | 768 |
| `data/MOSI/mosi_raw_glove.pkl` | GloVe 840B | 300 |

Raw SDK files, if you rebuild pickles yourself:

| Corpus | Computational sequences |
| --- | --- |
| MOSEI | `CMU_MOSEI_COVAREP.csd`, `CMU_MOSEI_Labels.csd`, `CMU_MOSEI_TimestampedWords.csd`, `CMU_MOSEI_VisualFacet42.csd` |
| MOSI | `CMU_MOSI_COVAREP.csd`, `CMU_MOSI_Opinion_Labels.csd`, `CMU_MOSI_TimestampedWords.csd`, `CMU_MOSI_Visual_Facet_42.csd` |

`model/data/readme.md` lists the same filenames. GloVe vectors, if you rebuild
text features from words, are `glove.840B.300d.txt`.

See [datasets.md](datasets.md) for the pickle schema.

## Hardware assumptions in the original scripts

Almost every training script calls `.cuda()` on encoders, fusion, and heads.
`LowRankTensorFusion` picks `cuda:0` when `torch.cuda.is_available()` is true,
otherwise CPU. The synthetic examples force CPU so they run on this environment.

`train()` and `test()` already fall back to CPU when CUDA is missing. The
blocking issue for a CUDA-less machine is the `.cuda()` calls in the launch
scripts, not the training loop itself.

## Checkpoints

| Location | Contents |
| --- | --- |
| `model/checkpoints/` | `{FusionMethod}.pt` and `glove_{FusionMethod}.pt` |
| `model/checkpoints/ablation/` | `model_{text+audio+visual}.pt` and `model_glove_{...}.pt` |

The checkpoint readmes are only location markers; weights are not committed.

## Running the documentation examples

From the repository root:

```bash
python examples/summarize_results.py
python examples/metric_walkthrough.py
python examples/forward_pass_demo.py
python examples/fusion_shape_walkthrough.py
python examples/toy_train_gmtm.py
python examples/run_all.py
```

`summarize_results.py` and `metric_walkthrough.py` need only the standard
library plus NumPy. The forward / training demos need PyTorch.
