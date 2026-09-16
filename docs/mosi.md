# MOSI next

CMU-MOSI is the smaller, older sibling of MOSEI. This repo already **scores**
MOSEI-trained checkpoints on MOSI. It does **not** yet train GMTM on MOSI's own
train fold. That is the next personal experiment this document is for.

## What already exists

| Piece | Location | Status |
| --- | --- | --- |
| MOSI pickle builder | `model/data/MOSI/get_dataset.ipynb` | Writes `mosi_raw_bert.pkl` / `mosi_raw_glove.pkl` |
| Merged MOSI loader | `get_mosi_dataloader`, `get_ablation_mosi_dataloader` | Used for transfer only |
| Fusion transfer | `model/mosi_test/train_mosi_bert.py`, `train_mosi_glove.py` | Test-only, needs MOSEI `.pt` |
| GMTM transfer | `model/mosi_test/mult_bert_mosi.py`, `mult_glove_mosi.py` | Test-only |
| Transfer tables | `model/mosi_test/*results.csv` | Checked in; see [`experiments.md`](experiments.md) |

There is no `train_mosi_gmtm.py` that calls `get_dataloader` on MOSI and writes a
MOSI-native checkpoint. `examples/03_gmtm_tiny_train.py` is the CPU-sized
rehearsal of that loop.

## Transfer protocol (current)

1. Train on MOSEI (`train_GMTM_bert.py` / `train_main_bert.py`).
2. Point a MOSI script at `../data/MOSI/mosi_raw_bert.pkl`.
3. `get_mosi_dataloader` concatenates MOSI train+valid+test.
4. Report MAE / Acc7 / Acc5 / Acc2 / Corr / F1 on that pool.

This answers a domain-shift question (YouTube movie reviews → a different
review set) and **inflates** the MOSI sample count. Do not cite those CSVs as
"MOSI test Acc2".

## What the transfer tables say

Three facts should drive the next run (numbers in [`experiments.md`](experiments.md)):

1. **BERT GMTM wins MOSEI and loses the MOSI transfer to TransformerLate**
   (`0.5640` vs `0.5846` MAE on MOSEI; `0.9493` vs `0.8986` on merged MOSI).
2. **On MOSI BERT transfer, text+visual (`0.9044`) beats the full trio (`0.9493`)**.
   Audio is anti-correlated when used alone (`Corr = -0.1256`).
3. **GloVe GMTM is the best MOSI transfer model** (`0.9748` MAE) even though
   LMF wins GloVe MOSEI. The full trio helps GloVe MOSI; `text+audio` does not.

So "just use the MOSEI-best checkpoint" is the wrong MOSI strategy.

## Next experiment: train on MOSI

Goal: a MOSI-native GMTM number that is comparable to published MOSI papers
**and** to the transfer rows above.

### Recommended first run

```
data:          model/data/MOSI/mosi_raw_bert.pkl
loader:        get_dataloader(..., data_type='mosi', max_pad=True, num_workers=0)
model:         GatedMultiTransfomerModel(3, [35, 74, 768], HParams)
HParams:       same as MOSEI (embed_dim=64, layers=4, heads=4) to start
loss:          L1Loss
optim:         AdamW, lr=1e-4, weight_decay=0.01
early_stop:    True, on MOSI valid L1
save:          model/checkpoints/mosi/gmtm_bert_tav.pt
report:        official test fold only (not the merged loader)
```

Then immediately run the same recipe with `modalities=['text', 'visual']`
because transfer already says audio may hurt.

### Second run (GloVe)

Repeat with `[35, 74, 300]` and `mosi_raw_glove.pkl`. On transfer, GloVe GMTM
wanted all three modalities; check whether that still holds when the model
is allowed to overfit MOSI audio.

### Third run (protocol variants)

Keep the checkpoint fixed and publish **two** test rows:

| Row | Loader | Why |
| --- | --- | --- |
| MOSI test | `get_dataloader` → `test` | Comparable to papers |
| MOSI all | `get_mosi_dataloader` | Comparable to the existing transfer CSVs |

### Hyperparameters to sweep once the loop works

MOSI is ~10× smaller than MOSEI. The MOSEI GMTM (`layers=4`, `embed_dim=64`,
nine pairwise encoders) is easy to overfit.

| Knob | Start | Try next |
| --- | --- | --- |
| `layers` | 4 | 2 |
| `embed_dim` | 64 | 32 |
| `out_dropout` / `embed_dropout` | 0.1 / 0.2 | 0.3 / 0.3 |
| `lr` | 1e-4 | 5e-5 |
| `weight_decay` | 0.01 | 0.05 |
| batch size | 32 | 16 |
| modality dropout | unused in `forward` | wire `HParams.modality_dropout` or keep using ablation zeroes |

`HParams.modality_dropout` and `use_text_transformer` are declared in the
trainers and **never read** by `GatedMultiTransfomerModel`. Do not assume
turning them on in `HParams` changes the graph.

## Script sketch

A MOSI-native trainer should look like `train_GMTM_bert.py` with three edits:

1. `filepath = 'data/MOSI/mosi_raw_bert.pkl'` and `data_type='mosi'`.
2. Use `get_dataloader` / `get_ablation_dataloader`, **not** `get_mosi_dataloader`.
3. Write `mosi_native_bert_results.csv` (test fold) and optionally a second
   merged-loader CSV for comparison with `mosi_bert_results.csv`.

Until that file exists, rehearse the shapes with:

```bash
python examples/01_synthetic_mosi_dataset.py
python examples/03_gmtm_tiny_train.py --steps 20
```

The tiny trainer uses the same `Identity → GMTM → Identity` stack and the
same L1 + Acc2/F1 helpers.

## Bugs to fix before a real MOSI sweep

Logged here so they do not get rediscovered mid-run:

1. `train_mosi_bert.py` calls `test(..., dataset='mosi')` but `test()` has no
   `dataset` argument.
2. `train_mosi_glove.py` imports MultiBench `training_structures.Supervised_Learning`.
3. Checkpoint name `Concat` vs `ConcatLate`.
4. `get_mosei.py` is a MOSI HDF5 exporter with a hardcoded Windows path;
   trainers do not use it.
5. Confusion-matrix `plt.show()` in `single_test` blocks headless runs.
   `model/metrics.py` takes `plot_confusion=False`.

## File checklist when you add a MOSI-native result

- [ ] New script under `model/` or `model/mosi_test/` that trains on the MOSI train fold
- [ ] Checkpoint directory `model/checkpoints/mosi/`
- [ ] CSV with the same six metric columns
- [ ] One extra row (or a note) stating **test-fold vs merged**
- [ ] A line in [`experiments.md`](experiments.md)

## Related

- Dataset files: [`datasets.md`](datasets.md)
- Architecture: [`architecture.md`](architecture.md)
- Full tables: [`experiments.md`](experiments.md)
