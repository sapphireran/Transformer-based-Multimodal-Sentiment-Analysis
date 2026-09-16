# Datasets and features

The training code expects **pre-aligned pickles**, not raw video. Those
pickles are not in git (they are large and redistributed by CMU). The
examples under `examples/` synthesise tensors with the same shapes so the
model code can be exercised without a download.

## CMU-MOSI and CMU-MOSEI

| | MOSI | MOSEI |
| --- | --- | --- |
| What | YouTube movie-review clips | YouTube opinion clips, broader topics |
| Label | opinion intensity in `[-3, 3]` | same Likert-style score |
| Typical split | ~1.3k / 0.3k / 0.7k train/valid/test | ~16k / 1.9k / 4.7k |
| Role in this repo | transfer / extra test (`model/mosi_test/`) | main tables (`model/results/`) |

Official SDK and downloads: [CMU Multimodal SDK](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK).
`model/data/MOSEI/get_mosei.py` and the notebooks under `model/data/` show
one local conversion path (CSD → aligned HDF5 / pickle). Paths in those
files are machine-specific (including a Windows `F:\...` path) and are
not used by the examples lab.

## Feature streams

`model/data/readme.md` lists the CSD names this project was built from.

| stream | extractor | dim | pickle key |
| --- | --- | --- | --- |
| visual | Facet 4.2 facial action units / emotions | 35 | `vision` |
| acoustic | COVAREP (voicing, glottal, spectral) | 74 | `audio` |
| language (BERT) | contextual word / sentence embeddings | 768 | `text` |
| language (GloVe) | `glove.840B.300d` word vectors | 300 | `text` |

COVAREP can contain `-inf` for unvoiced frames. `Affectdataset` rewrites
those to `0.0` on load. Optional `z_norm=True` standardises each clip per
feature channel; the main scripts leave it off.

## Expected pickle layout

`get_dataloader(filepath)` does `pickle.load` and assumes:

```text
{
  "train": {"vision": ndarray, "audio": ndarray, "text": ndarray, "labels": ndarray},
  "valid": { ... },
  "test":  { ... },
}
```

Each array is `[N, T, F]` except labels, which are `[N, 1]` or `[N, 1, 1]`.
`drop_entry` removes rows whose language tensor is all zeros.

Place files at:

```text
model/data/MOSEI/mosei_raw_bert.pkl
model/data/MOSEI/mosei_raw_glove.pkl
model/data/MOSI/mosi_raw_bert.pkl
model/data/MOSI/mosi_raw_glove.pkl
```

The empty `model/data/glove.840B.300d.txt` is a path placeholder from the
original upload, not the 2 GB GloVe file.

## Dataloader variants

| function | returns | notes |
| --- | --- | --- |
| `get_dataloader` | train, valid, test | standard three-way split |
| `get_ablation_dataloader` | train, valid, test | zeros dropped modalities |
| `get_mosi_dataloader` | **test only** | concatenates MOSI train+valid+test into one loader |
| `get_ablation_mosi_dataloader` | **test only** | same merge, then zero dropped streams |

`get_mosi_dataloader` is what `model/mosi_test/train_mosi_bert.py` uses.
That means the MOSI CSVs are **not** a standard held-out MOSI test: they
score MOSEI-trained checkpoints on **all** MOSI clips. Treat those tables
as a transfer dump, not as a MOSI leaderboard number.

### Packed vs max-pad

`max_pad=False` (default) → `_process_1`:

* `pad_sequence` per modality;
* returns `(inputs, lengths, indices, labels)`;
* LSTM/GRU encoders get `has_padding=True`.

`max_pad=True` → `_process_2`:

* every stream cropped / padded to `max_seq_len` (50);
* returns `(vision, audio, text, labels)` with no length vector;
* required by `TransformerEarly` and GMTM.

`train_main_bert.py` therefore builds **two** dataloader triples from the
same pickle: `traindata_OT` (packed) and `traindata_TE` (max-pad).

### Ablation zeroing

Missing modalities are **not** removed from the batch. They are replaced
with `torch.zeros((50, F))` so GMTM can keep three encoders and a fixed
`n_features` list. `examples/06_ablation_zero_modalities.py` repeats that
trick on synthetic tensors.

## Labels

Scores live in `[-3, 3]`. Evaluation then derives:

* regression: MAE, MSE, Pearson;
* 7-way / 5-way accuracy from **equal-width** bins of that interval;
* binary accuracy / F1 after dropping true zeros.

See `docs/evaluation.md`. This binning is a property of *this* repo; it is
not automatically the same as every CMU-MOSEI paper.

## What the examples use instead

`examples/msa_lab/synthetic.py` draws a latent score `y ~ U[-3, 3]`, then
builds each modality as

```text
X_m = (y/3) · sin(2π (t + φ_m)) · p_m + σ ε
```

with a per-modality phase `φ_m` and a random projection `p_m`. Shapes
match the named packs `toy`, `mosei_bert`, `mosei_glove`, `mosi_bert`,
`mosi_glove`. Noise `σ` is 0.25 by default and 0.05 for the toy overfit.
