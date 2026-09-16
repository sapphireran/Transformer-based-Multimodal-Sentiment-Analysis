# Local setup (personal machine)

I did the original runs on a CUDA box with the CMU Multimodal SDK already installed.
This file is the reconstruction checklist, not a one-click installer.

## Python packages that the scripts import

From reading the tree (no lockfile was committed):

```text
torch
numpy
scikit-learn
scipy
tqdm
matplotlib
memory_profiler          # train_and_test.all_in_one_train
h5py                     # get_mosei.py + notebooks
torchtext                # imported in get_dataloader.py; not on the hot path
mmsdk                    # CMU-MultimodalSDK, only for rebuilding .csd → pickle
```

I have not pinned versions here on purpose. The checkpoints were saved as full modules
(`torch.save(model, path)`), so the **PyTorch minor version that wrote the pickle** is
the one that will load it cleanly.

## Directory layout I expect on disk

Working directory for the training scripts is `model/`.

```text
model/
  data/
    glove.840B.300d.txt          # real 840B file; git has a 0-byte stub
    MOSEI/
      mosei_raw_bert.pkl
      mosei_raw_glove.pkl
      cmumosei/                  # optional, only if rebuilding from SDK
        CMU_MOSEI_COVAREP.csd
        CMU_MOSEI_Labels.csd
        CMU_MOSEI_TimestampedWords.csd
        CMU_MOSEI_VisualFacet42.csd
    MOSI/
      mosi_raw_bert.pkl
      mosi_raw_glove.pkl
      cmumosi/
        CMU_MOSI_COVAREP.csd
        CMU_MOSI_Opinion_Labels.csd
        CMU_MOSI_TimestampedWords.csd
        CMU_MOSI_Visual_Facet_42.csd
  checkpoints/
    ConcatEarly.pt
    ConcatLate.pt
    LowRankTensorFusion.pt
    TensorFusion.pt
    TransformerEarly.pt
    TransformerLate.pt
    glove_ConcatEarly.pt
    ...
    ablation/
      model_text.pt
      model_audio.pt
      model_visual.pt
      model_text+audio.pt
      model_text+visual.pt
      model_audio+visual.pt
      model_text+audio+visual.pt
      model_glove_text.pt
      ...
```

None of the `.pkl` / `.pt` / `.csd` / GloVe files are meant to be committed.

## Dataset sources

- CMU-MOSI / CMU-MOSEI via the [CMU Multimodal SDK](https://github.com/A2Zadeh/CMU-MultimodalSDK).
- BERT features: the notebooks under `model/data/MOSEI` and `model/data/MOSI` (I used
  whatever HuggingFace BERT produced 768-d token sequences, then aligned to 50 steps).
- GloVe: `glove.840B.300d` from Stanford NLP.

`model/data/MOSEI/get_mosei.py` still has a hardcoded Windows path
`F:\MOSEI\model\data\MOSEI\cmumosi`. Treat it as a scratch script, not the current
pipeline.

## GPU assumptions

Almost every `run_experiment` does `.cuda()` with no CPU fallback. The fusion modules
in `LowRankTensorFusion` do pick `cuda:0` if available, but the training scripts will
still crash on CPU because encoders are constructed with `.cuda()`.

Batch size in every current script: **32**. Sequence cap: **50**.

## Commands I actually use

```bash
cd model

# BERT fusion bake-off (writes main_results.csv)
python train_main_bert.py

# GloVe bake-off — eval only unless you uncomment train()
python train_main_glove.py

# GMTM BERT / GloVe
python train_GMTM_bert.py
python train_GMTM_glove.py

# Transfer: run from model/mosi_test so the relative checkpoint paths resolve
cd mosi_test
python train_mosi_bert.py
python train_mosi_glove.py
python mult_bert_mosi.py
python mult_glove_mosi.py
```

## Things that will bite a future me

1. **Full-module pickles.** Do not refactor class names before a re-eval.
2. **`train_main_bert.py` epochs = 1.** Bump this before claiming a new table.
3. **Commented `train()` in GMTM and GloVe scripts.** Easy to think a run trained when it only tested.
4. **MOSI loader merges splits.** See [`data-pipeline.md`](data-pipeline.md).
5. **`train_mosi_glove.py` MultiBench import.** Fix before re-running.
6. **`single_test` calls `plt.show()`.** Headless boxes need `MPLBACKEND=Agg` or the
   confusion-matrix block commented out.
7. **`memory_profiler` wraps the entire train function.** Fine on a workstation; noisy in a shared job.

## What I am not putting in this repo

- Company code, private datasets, or anything that is not MOSI/MOSEI + this personal stack.
- HuggingFace cache, GloVe vectors, or trained weights.
- A Docker image. If I ever need one, it belongs in a separate personal env, not here.
