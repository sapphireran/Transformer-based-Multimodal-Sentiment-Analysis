# Repo map

Personal inventory of what each path is for. Written so I can open this repo six months
later and not have to re-derive the wiring from `import` statements.

## Top level

| Path | Role |
| ---- | ---- |
| `README.md` | Project brief + headline tables |
| `LICENSE` | MIT, copyright 2024 pang990801 |
| `docs/` | Durable explanations (architecture, metrics, data, setup) |
| `notes/` | Dated-style lab notes. Opinions belong here, not in the training scripts |
| `model/` | All runnable experiment code |

There is no `requirements.txt` in the original tree. The import set I actually used is
listed in [`setup.md`](setup.md).

## Training and eval

| File | What it does | Notes I keep forgetting |
| ---- | ------------ | ------------------------ |
| `model/models.py` | Building blocks: LSTM/GRU/MLP, concat, TFN, LMF, early/late transformers, GMTM | `Linear` is defined twice (class then factory). GMTM still has commented residual / `trans_mems` paths |
| `model/train_and_test.py` | `MultiFramework`, `train`, `single_test`, `eval_affect` | Saves the **whole** `nn.Module` with `torch.save(model, path)`, not a state dict. `single_test` pops a matplotlib confusion matrix |
| `model/train_main_bert.py` | Six fusion methods on MOSEI BERT | `input_dims = [35, 74, 768]`. Packed sequences except TransformerEarly (`max_pad=True`) |
| `model/train_main_glove.py` | Same six methods on MOSEI GloVe | `text_dim = 300`. Train loop commented; loads `checkpoints/glove_{method}.pt` |
| `model/train_GMTM_bert.py` | GMTM on MOSEI BERT | Ablation list currently only `['text','audio','visual']`; other combos commented. Train commented |
| `model/train_GMTM_glove.py` | GMTM on MOSEI GloVe | Full 7-way modality grid. Train commented. Writes `ablation_glove_results.csv` |

## Data

| File | What it does |
| ---- | ------------ |
| `model/data/get_dataloader.py` | `Affectdataset`, packed (`_process_1`) vs stacked (`_process_2`), ablation zero-fill, MOSI merge loader |
| `model/data/readme.md` | Expected `.csd` / GloVe filenames |
| `model/data/MOSEI/get_mosei.py` | CMU-SDK alignment sketch. Path is a Windows drive (`F:\...`) and the field names say MOSI — leftover from the first download pass |
| `model/data/MOSEI/get_dataset.ipynb` | How I built `mosei_raw_*.pkl` |
| `model/data/MOSI/get_dataset.ipynb` | Same for MOSI |
| `model/data/analysis_dataset.ipynb` | Shape / key inspection. Train split on BERT MOSEI had **16327** utterance ids when I last ran it |
| `model/data/glove.840B.300d.txt` | Placeholder in git (0 bytes). Real file stays local |

## MOSI transfer folder

These scripts **do not train on MOSI**. They load MOSEI checkpoints and score MOSI.

| File | Checkpoint pattern | Output CSV |
| ---- | ------------------ | ---------- |
| `mosi_test/train_mosi_bert.py` | `../checkpoints/{Fusion}.pt` | `mosi_bert_results.csv` |
| `mosi_test/train_mosi_glove.py` | `../checkpoints/glove_{Fusion}.pt` | `mosi_glove_results.csv` |
| `mosi_test/mult_bert_mosi.py` | `../checkpoints/ablation/model_{mods}.pt` | `ablation_mosi_results.csv` (write currently commented) |
| `mosi_test/mult_glove_mosi.py` | `../checkpoints/ablation/model_glove_{mods}.pt` | `ablation_mosi_glove_results.csv` |

`train_mosi_glove.py` still imports `training_structures.Supervised_Learning`. That is a
leftover MultiBench path. The BERT MOSI script already uses the local `train_and_test`.
If I re-run GloVe MOSI transfer, switch the import first.

## Results that are checked in

Canonical copies I treat as the lab record:

```text
model/main_results.csv
model/glove_results.csv
model/ablation_results.csv
model/ablation_glove_results.csv
model/results/          # duplicates of the four above + plot.ipynb
model/mosi_test/*.csv
```

`plot.ipynb` is large (~1 MB) because it embeds figures. I have not cleaned the outputs.

## Checkpoints

```text
model/checkpoints/readme.md
model/checkpoints/{Fusion}.pt            # not in git
model/checkpoints/glove_{Fusion}.pt      # not in git
model/checkpoints/ablation/readme.md
model/checkpoints/ablation/model_{text+audio+visual}.pt
model/checkpoints/ablation/model_glove_{...}.pt
```

`torch.save(model, ...)` means the pickle is tied to the class definitions in
`models.py` / `train_and_test.py`. Renaming `MultiFramework` or GMTM will break loads.

## Naming quirks I keep tripping on

- Class name is `GatedMultiTransfomerModel` (missing `o` in Transformer). Filenames say GMTM.
- MOSI late-concat is logged as `Concat` in `mosi_bert_results.csv`, `ConcatLate` on MOSEI.
- GMTM hyperparams set `use_text_transformer` and `modality_dropout` but the forward pass
  does not read them. Dead knobs.
- `get_mosi_dataloader` always merges splits. Name sounds like a normal loader; it is not.
