# Reproduction notes

Personal environment notes for this repo. The examples path is the
only thing that is meant to work on a CPU-only checkout.

## Two install tiers

```bash
# architecture + metrics examples (CPU is enough)
python3 -m pip install -r requirements.txt

# real MOSI/MOSEI training (GPU strongly recommended)
python3 -m pip install -r requirements-train.txt
```

`requirements-train.txt` pulls `memory-profiler` (imported at module
level by `train_and_test.py`), `h5py`, and `torchtext`. The Multimodal
SDK is a separate git install; you only need it to rebuild pickles.

PyTorch CUDA wheels are **not** pinned. The original runs used a
single visible GPU and `.cuda()` on every module. There is no
`device` CLI flag.

## Directory assumptions

From `model/`:

```
data/MOSEI/mosei_raw_bert.pkl
data/MOSEI/mosei_raw_glove.pkl
checkpoints/{ConcatEarly,ConcatLate,LowRankTensorFusion,
             TensorFusion,TransformerEarly,TransformerLate}.pt
checkpoints/ablation/model_{text,audio,visual,...}.pt
```

From `model/mosi_test/`:

```
../data/MOSI/mosi_raw_bert.pkl
../data/MOSI/mosi_raw_glove.pkl
../checkpoints/*.pt
```

None of those binaries are in git. Without them, `train_*.py` fails
at `pickle.load` / `torch.load`. Use `examples/run_all.py` instead.

## Python import path

Scripts do:

```python
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
```

and then `from models import ...` / `from data.get_dataloader import ...`.
That only works if cwd is `model/` (or `model/mosi_test/` for the
transfer scripts, which also reach `../` checkpoints).

The examples add `model/` onto `sys.path` themselves so they can be
launched from the repo root.

## Loading old checkpoints

Logged runs used `torch.save(model, path)` (entire `nn.Module`).
On PyTorch ≥ 2.6:

```python
model = torch.load(path, map_location="cpu", weights_only=False)
```

The class graph must still be importable (`models.GatedMultiTransfomerModel`,
`train_and_test.MultiFramework`, …). Moving those classes to another
module will break old `.pt` files.

## Known footguns

1. **`memory_profiler` import is unconditional** in `train_and_test.py`.
   A train-tier install is required even if you only want `test()`.
2. **`plt.show()`** inside `single_test` opens a GUI window per eval.
   On a headless box set `MPLBACKEND=Agg` or expect the function to
   hang / dump a file depending on matplotlib's default.
3. **`all_in_one_test` runs the test twice.** Inference-time prints
   are inflated; metrics are computed twice (same numbers).
4. **MOSI loader merges splits.** Do not submit those CSVs as
   official MOSI test.
5. **`train_GMTM_*.py` training is commented.** Uncomment the `train(...)`
   block or the script will only `torch.load` a missing file.
6. **`train_main_bert.py` uses 1 epoch** in the checked-in copy.
7. **Windows paths** in `get_mosei.py` (`F:\MOSEI\...`) are leftover
   from the original machine. Point `MOSI_PATH` at your `.csd` folder.
8. **GloVe file in git is empty.** `model/data/glove.840B.300d.txt` is
   a 0-byte placeholder. Download the real 840B vectors if you rebuild
   GloVe pickles.
9. **Random seeds are not set.** The CSVs are single-run snapshots.
10. **`Linear` is defined twice** in `models.py` (class, then function).
    The function wins. Anything that did `models.Linear(in, out)` as a
    *class* after that line is actually calling the factory.

## CPU example sanity command

```bash
python3 examples/run_all.py
```

Expected: all five demos print `ok` / a small table, process exits 0.
GMTM uses `embed_dim=16`, `layers=1`, so this should finish in seconds
on CPU.

## GPU training sanity (when pickles exist)

```bash
cd model
# uncomment train(...) in train_GMTM_bert.py first
python3 train_GMTM_bert.py
```

You should see tqdm epoch bars, "Saving Best Model", then the metric
block from `single_test` (and a confusion-matrix figure).

## Data license

MOSI and MOSEI are **not** redistributed here. Get them from CMU and
respect their terms. The pickles you build from FACET / COVAREP / BERT
are derived data — keep them local.
