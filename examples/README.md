# Examples lab

These scripts run on CPU with **no MOSI / MOSEI pickles, no GPU, and no
checkpoints**. They reuse the fusion modules in `model/models.py` and the
metric helpers in `model/metrics.py`.

Install the personal study stack first:

```bash
pip install -r requirements.txt
```

Then from the repository root:

```bash
python examples/01_synthetic_batch.py
python examples/02_fusion_forward.py
python examples/03_gated_transformer_toy.py
python examples/04_metrics_walkthrough.py
python examples/05_packed_vs_padded.py
python examples/06_ablation_zero_modalities.py
```

Or run the same list through `python scripts/run_examples.py`.

| Script | What it shows |
| --- | --- |
| `01_synthetic_batch.py` | `[B, T, F]` tensors for the toy / BERT / GloVe dimension packs |
| `02_fusion_forward.py` | Output shapes for concat, TFN, LMF, transformer fusion, GMTM |
| `03_gated_transformer_toy.py` | A few AdamW steps that must reduce MAE on one batch |
| `04_metrics_walkthrough.py` | Uniform Acc-7 bins plus MAE / Acc-2 / F1 on oracle vs noise |
| `05_packed_vs_padded.py` | `_process_1` packed lengths vs `_process_2` max-pad  |
| `06_ablation_zero_modalities.py` | Zeroing dropped streams the way the ablation loader does |

The helper package is `examples/msa_lab/`. Tests under `tests/` import it
directly. None of these numbers should be compared to
`model/results/*.csv`; those tables come from the real CMU features.
