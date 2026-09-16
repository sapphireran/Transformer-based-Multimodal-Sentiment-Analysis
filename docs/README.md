# Documentation

| Page | Contents |
| --- | --- |
| [architecture.md](architecture.md) | Fusion zoo + gated multi-transformer graph |
| [related_work.md](related_work.md) | TFN, LMF, concat, Transformers, vs MulT |
| [datasets.md](datasets.md) | MOSI / MOSEI features, pickle schema, ablation zeros |
| [batch_layout.md](batch_layout.md) | Packed variable-length vs padded `[B,T,F]` cubes |
| [training.md](training.md) | Scripts, optimizer, BERT vs GloVe hidden sizes |
| [hyperparameters.md](hyperparameters.md) | Values frozen in the committed CSVs |
| [metrics.md](metrics.md) | MAE, uniform Acc-7/5, binary Acc-2 / F1 |
| [results.md](results.md) | Committed CSV tables |
| [interpreting_results.md](interpreting_results.md) | Text dominance, MOSI transfer caveats |
| [reproduction.md](reproduction.md) | Env, data you must fetch, known sharp edges |

Runnable stand-ins that skip the CMU pickles live in
[`../examples/`](../examples/).
