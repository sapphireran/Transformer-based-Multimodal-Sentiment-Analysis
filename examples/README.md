# Examples

CPU walkthroughs for the personal GMTM / multimodal-sentiment tree. None of
these scripts download MOSI/MOSEI, load a `.pt` checkpoint, or call `.cuda()`.

Run them from the **repository root** after `pip install -r requirements.txt`.

| Script | What it shows |
| --- | --- |
| `read_result_tables.py` | Pretty-print the recorded CSVs (no model import) |
| `compare_tables.py` | BERT vs GloVe MAE, text-only vs full GMTM, MOSI transfer gap |
| `gmtm_forward.py` | Step-by-step GMTM tensor shapes on a 4×16 clip |
| `fusion_shapes.py` | Concat / TFN / LMF / transformer fusion output ranks |
| `metrics_demo.py` | Uniform Acc-7 / Acc-5 bins and polarity F1 |
| `multiframework_demo.py` | `encoders → fusion → head` with Identity+GMTM and late concat |
| `dataset_collate_demo.py` | In-memory `Affectdataset` + `_process_1` / `_process_2` |
| `ablation_zero_mask.py` | Same GMTM weights, unused streams zeroed |
| `run_all.sh` | Runs every script above in order |

```bash
python3 examples/read_result_tables.py
python3 examples/gmtm_forward.py
bash examples/run_all.sh
python3 -m pytest -q
```

Helpers live in `common.py`: MOSEI feature widths, a seeded dummy batch, and
`sys.path` setup so `import models` resolves to `model/models.py` the same way
the original training scripts do after `cd model`.
