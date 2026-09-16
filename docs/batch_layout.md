# Batch layouts: packed vs padded

The fusion scripts do not all see the same tensor layout. Recurrent late
fusion wants **variable-length packed sequences**. GMTM and the early
Transformer want a **stacked `[B, T, F]` cube** with a fixed `T`. Getting
this wrong is the usual reason a re-run crashes in `pack_padded_sequence`
or in `Conv1d`.

## What `Affectdataset` returns

For a single clip, with `flatten_time_series=False`:

| `max_pad` | Tuple |
| --- | --- |
| `False` | `(vision, audio, text, index, label)` |
| `True` | `(vision, audio, text, label)` with each stream sliced / padded to `max_pad_num` (50) |

Vision / audio / text are `[T_i, F]` float tensors. Labels are `[1, 1]` or
`[1]`. Empty-text clips are already gone (`drop_entry`).

## Collate `_process_1` — packed / variable length

Used when `max_pad=False`. Pads each modality independently to the longest
clip **in that batch** and also returns lengths:

```
processed_input          # list of 3 tensors, each [B, T_max, F]
processed_input_lengths  # list of 3 length vectors, each [B]
indices                  # [B, 1]
labels                   # [B, 1]
```

`train(..., is_packed=True)` then feeds
`[[vision, audio, text], lengths]` into `MultiFramework`. Encoders that
set `has_padding=True` (`LSTM`, `GRUWithLinear`) call
`pack_padded_sequence`.

Scripts: `ConcatEarly`, `ConcatLate`, `LowRankTensorFusion`,
`TensorFusion`, `TransformerLate` in `train_main_*.py`.

## Collate `_process_2` — stacked / fixed `T`

Used when `max_pad=True`. Every clip is already length 50, so collate is
a `torch.stack`:

```
vision, audio, text, labels
# [B, 50, 35], [B, 50, 74], [B, 50, 768|300], [B, 1]
```

`train(..., is_packed=False)` feeds the three tensors as a list.
`Conv1d` / `TransformerEncoder` / GMTM all assume this cube.

Scripts: `TransformerEarly`, every `train_GMTM_*.py` run, and the MOSI
transfer paths that load those checkpoints.

## Why the split exists

| Consumer | Needs |
| --- | --- |
| `pack_padded_sequence` | true lengths, no fake padded steps inside the recurrent core |
| `nn.Conv1d` + Transformer | a dense `[B, F, T]` / `[T, B, F]` layout |
| GMTM pairwise attention | the same `T` on every modality so `trans[i][j]` can attend |

Zeroing a modality for ablation (`get_ablation_dataloader`) happens
**before** collate: the dropped stream is a `[50, F]` zero tensor, so
`_process_2` still stacks three cubes.

## CPU stand-in

`examples/packed_vs_padded.py` builds variable-length toy clips and prints
both collate results so you can see `T_max` vs the fixed-50 cube without
loading a pickle.
