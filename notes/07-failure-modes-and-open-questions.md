# 07 — Failure modes and open questions

Things that are wrong, sharp, or still unanswered. Personal list. I am
not fixing code in this notes pass.

## Bugs and mismatches I have already tripped on

### 1. `train_main_bert.py` says 1 epoch

The BERT fusion CSV does not look like a 1-epoch run. Either the file was
edited after the experiment or I am wrong about how fast LMF / late
transformer fit. Open: restore the epoch count from memory (20?) and
accept that this git copy is not the recipe.

### 2. GloVe `LateFusionTransformer(in_dim=1792)` vs 704-d concat

Retrain from the current file will not match the checkpoint graph.
Eval via `torch.load` is fine. Fix `in_dim=704` before any GloVe late
transformer retrain.

### 3. Early transformer `batch_first=True` + `(S, B, D)` permute

`EarlyFusionTransformer` and `LateFusionTransformer` construct
`TransformerEncoderLayer(..., batch_first=True)` then permute to
sequence-first and call the encoder. PyTorch will treat the **batch**
dimension as time. This may be “the model I actually trained.” I cannot
unscramble it without a retrain. Open: one controlled A/B.

### 4. `TransformerEncoderLayer.apply_sublayer` looks like a double residual

```text
return norm(x + dropout(x))
```

after `apply_attention` already replaced `x` with attn output. That is
not the standard `x + dropout(attn(LN(x)))`. Combined with
`normalize_before=True` and a second LN, the custom encoder in GMTM is
not a textbook Pre-LN block. I do not know if this is hurting.

### 5. MOSI loader merges all splits

Note 04. Any MOSI number in this repo is in-corpus, not held-out.

### 6. `train_mosi_glove.py` imports MultiBench

`from training_structures.Supervised_Learning import train, test`
will fail in a clean clone. BERT MOSI script already uses local
`train_and_test`.

### 7. `get_mosei.py` is a MOSI Windows scratch script

Path `F:\MOSEI\model\data\MOSEI\cmumosi`, field names `CMU_MOSI_*`.
Do not run it expecting MOSEI.

### 8. `Linear` defined twice in `models.py`

A class `Linear` and later a factory function `Linear`. The factory
wins. Anything that did `Linear(a, b)` after the factory is an
`nn.Linear` with Xavier. The class is dead. Confusing only.

### 9. Confusion matrix blocks eval

`plt.show()` inside `single_test`. Headless hang.

### 10. GMTM dead knobs still in the optimizer

`trans_mems`, `alpha`, unused HParams. `trans_mems` still allocate
GPU memory.

### 11. Early transformer BERT head `MLP(64, 64, 1)` on a 32-d fusion

Either I changed `embed_dim` or the head is wrong. `torch.load` again
saves me on eval. Retrain will crash or silently mis-init depending on
how I construct the module before load.

### 12. `eval_affect` vs the unused `prede` list

`single_test` builds a −1/0/1 tensor and never scores it. Acc-2 uses
raw `> 0` on the regression output. Fine, but the extra tensor made me
think I had a three-class metric. I do not.

## Experimental failures (not bugs — results)

### GloVe T+A collapse

MOSEI GMTM T+A MAE 0.7200 vs text 0.6616. MOSI GloVe T+A MAE 1.2863 vs
text 1.0085. Reproduced across datasets, same pairing. Highest priority
mystery. See note 03 and 04.

### Audio-only negative Corr on MOSI

−0.1256 (BERT) / −0.0720 (GloVe). Audio as trained on MOSEI is
anti-aligned with MOSI labels. Do not fuse audio in transfer without a
new adapter.

### BERT T+A+V worse than T+V on MOSI

0.9493 vs 0.9044 MAE. Audio is a liability under shift even when BERT
text is present.

### Tiny BERT trimodal leftover

0.0047 MAE. Could be seed noise. Do not build a paper on it.

## Open questions I actually want answers to

1. **Is the BERT leftover real at 3 seeds?** Text-only vs T+V vs T+A+V,
   same GMTM, drop slots for real (no zero-fill). If leftover < 0.01 MAE
   on the mean, I stop calling this a trimodal problem.

2. **Does LMF still beat GMTM on GloVe at 3 seeds?** Note 02 says yes
   on one run. If yes again, GMTM is a BERT-only toy.

3. **What do `softmax(modal_weights)` and mean gates look like?** I have
   never printed them. If audio weight stays ~1/3 on the BERT trimodal
   model, the gate story is fiction.

4. **Uniform Acc-7 vs rounded Acc-7?** Re-score every CSV with
   `round(clip(y,-3,3))` so I can glance at a paper table without lying.

5. **Real MOSI test split, same checkpoints.** Keep note 04 as appendix.

6. **Padding mask on GMTM.** Cheap experiment. If MOSI A/V unimodal
   Corr goes from negative to ~0, the zero tail was part of the story.

7. **Frozen BERT text + tiny audio adapter.** If audio cannot help a
   frozen text tower, I should spend time on COVAREP / alignment quality,
   not on another fusion block.

8. **Sarcasm / neutral band.** I never sliced MAE by `|y| < 0.5` vs
   `|y| > 2`. The leftover might live only in one band.

9. **Word-align vs utterance-level pooling.** I accepted the SDK
   word-align recipe. I have no idea how much of “audio is weak” is
   COVAREP-at-word-rate being a bad object.

10. **Does `batch_first` scramble explain TransformerEarly’s mid-pack
    finish?** One A/B would retire a lot of architecture talk.

## What I am not going to do

- I am not porting this into a company repo or a product recsys stack.
- I am not chasing SOTA on MOSEI with a 12-layer unimodal text model.
  That contest is not this project.
- I am not adding more fusion blocks (MCTN, MISA, Self-MM, dmd, etc.)
  until questions 1–3 are answered with the models I already have.
- I am not committing weights, pickles, or GloVe.

## Reproduction checklist if something looks “too good”

1. Which CSV? Which split? Packed or max-pad?
2. Acc-7 definition?
3. Was `train()` actually called?
4. Is this `torch.load` of an old class graph?
5. One seed?

If I cannot answer those, I do not quote the float.
