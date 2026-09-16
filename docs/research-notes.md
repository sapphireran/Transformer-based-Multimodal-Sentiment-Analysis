# Personal research notes

Working notes on what the logged tables actually say. These are not
claims about the official MOSI/MOSEI leaderboard. The Acc-7 protocol
here is uniform bins, MOSI numbers are pooled transfer, and F1 is
binary-positive rather than weighted. See [`metrics.md`](metrics.md)
and [`datasets.md`](datasets.md).

## 1. Language is the load-bearing stream

On MOSEI BERT, text-only GMTM is already at MAE **0.5687** / Corr
**0.7202**. Audio-only and visual-only sit around MAE 0.82–0.83 with
Corr 0.11–0.21. Audio+visual without text does not beat visual-only.

That matches the usual MOSI/MOSEI story: FACET + COVAREP add a bit
once language is present, and they are weak sentiment predictors
alone. It is also why a "multimodal win" of 0.0047 MAE (text →
text+audio+visual) is real in the table but small.

If a new idea does not beat **text-only GMTM**, it is not beating the
relevant baseline.

## 2. GMTM helps most when language is already strong

MOSEI BERT, full GMTM vs best transformer baseline (TransformerLate):

| | MAE | Acc-2 | Corr | F1 |
| --- | ---: | ---: | ---: | ---: |
| TransformerLate | 0.5846 | 0.8393 | 0.7041 | 0.8699 |
| GMTM T+A+V | 0.5640 | 0.8429 | 0.7255 | 0.8777 |

That is the cleanest win in the repo: same pickle, same metrics,
GMTM also beats text-only on MAE / Corr / F1.

On MOSEI GloVe the picture flips. Low-rank tensor fusion is the best
*baseline* (MAE 0.6174) and full GMTM is **worse** (0.6251). Pairwise
transformers appear to need the 768-D BERT space more than they need
the 300-D GloVe space — or the GloVe GMTM run simply did not land in
as good a basin (one seed, training loop partly commented).

Do not summarize this project as "GMTM always wins." Summarize it as
"GMTM won the BERT MOSEI comparison I actually logged."

## 3. Cross-modal attention can hurt

Two rows are warnings, not rounding error:

- MOSEI GloVe **text+audio** MAE 0.7200 vs text-only 0.6616.
- MOSI GloVe **text+audio** MAE 1.2863 vs text-only 1.0085.

Zeroing vision and leaving a noisy COVAREP stream is enough to drag
the gated fusion off the text solution. The learned `modal_weights`
and sigmoid gates *can* mute a stream, but they do not reliably do so
on these runs.

The ablation method itself is soft: dropped modalities are zero
tensors that still go through LayerNorm + ReLU + nine pairwise
encoders. A true two-modality GMTM (graph rebuilt without the third
tower) might look different. I have not logged that variant.

## 4. MOSI transfer is a different question

The `mosi_test/` CSVs load MOSEI checkpoints and score a **concatenated
MOSI train+valid+test**. Absolute MOSI MAE ~0.90–1.19 is not comparable
to papers that train on MOSI and report the official test split.

Relative order on that transfer:

- BERT: TransformerLate (0.8986) beats GMTM (0.9493). Text+visual GMTM
  (0.9044) is the best *GMTM* row and still trails TransformerLate.
- GloVe: GMTM (0.9748) beats every other logged fusion by a wide
  margin (next MAE is 1.0797).

So GMTM's MOSI-transfer story is embedding-dependent in the opposite
way from MOSEI: weaker language features, bigger GMTM gap. That is
interesting and also easy to overfit a narrative to. I would want a
second seed and the official MOSI split before writing it up.

Audio-only MOSI BERT Corr is **-0.13**. The model is anti-correlated
with the label. Treat audio-only / visual-only MOSI rows as "this
stream does not transfer," not as a fusion comparison.

## 5. Early vs late, concat vs tensor

On MOSEI BERT the late transformer beats early transformer, and both
tensor fusions beat both concats. The ordering is:

```
TransformerLate < LRTF < TFN < TransformerEarly < ConcatLate < ConcatEarly
```

(MAE, lower is better).

On MOSEI GloVe:

```
LRTF < TransformerLate < ConcatLate < TFN < TransformerEarly < ConcatEarly
```

Late fusion is not universally better; it is better when the unimodal
encoders have enough text width to do something useful before the
merge. GloVe ConcatEarly (joint 409-D LSTM) is the weakest BERT-or-GloVe
baseline I logged.

TFN's 128k-D head is a lot of parameters for a small MAE gap over
LRTF. If I were trimming the baseline set, I would keep LRTF and drop
full TFN.

## 6. Things the code does that a paper reviewer would flag

- Acc-7 / Acc-5 are equal-width bins, not rounded MOSI integers.
- Acc-2 / F1 drop neutrals and use binary F1.
- MOSI loader concatenates all splits.
- `train_main_bert.py` is checked in with `total_epochs=1`.
- GMTM `train(...)` is commented; scripts assume checkpoints exist.
- `EarlyFusionTransformer` head width (64) does not match the module
  output (32). The logged TransformerEarly row may not be reproducible
  from the current file without fixing that.
- `TransformerEncoderLayer(batch_first=True)` is then fed `[T, B, D]`.
- `trans_mems` and `self.alpha` are unused.
- `all_in_one_test` runs the test loop twice.
- `torch.save(model)` rather than `state_dict`.
- Ablations zero a stream instead of removing a tower.

I am leaving those as notes rather than silent fixes so the logged
CSVs stay attached to the code that produced them. The examples under
`examples/` use a tiny GMTM and the metric helpers, not the broken
head-width pairing.

## 7. What I would run next (personal)

1. Official MOSI test split, model trained on MOSI train (not MOSEI
   transfer).
2. A real two-tower GMTM for text+visual, vs zero-masked three-tower.
3. Weighted F1 + rounded Acc-7 next to the uniform-bin numbers, same
   predictions, so the tables can be cited next to prior work.
4. Two extra seeds on MOSEI BERT full GMTM vs TransformerLate. The
   0.02 MAE gap is large enough that it should survive that, but I
   have not shown it.
5. Swap FACET/COVAREP for stronger vision/audio encoders (even frozen
   ones). The current non-text streams may simply be weak features,
   not proof that non-text cannot help.

## 8. One-sentence takeaway

**With BERT features on MOSEI, a gated pairwise transformer is the
best fusion I logged; with GloVe, low-rank tensor fusion is; language
dominates every ablation; audio can actively hurt when vision is
dropped.**
