import numpy as np
import torch

from examples.synthetic_affect import (
    AUDIO_DIM,
    BERT_DIM,
    GLOVE_DIM,
    VISUAL_DIM,
    AffectShapes,
    _projected_score,
    batch_from_split,
    make_pickle_dict,
    make_split_arrays,
    zero_unused_modalities,
)


def test_bert_and_glove_widths():
    bert = make_split_arrays(4, AffectShapes.for_embedding("bert", max_len=10), seed=1)
    glove = make_split_arrays(4, AffectShapes.for_embedding("glove", max_len=10), seed=1)
    assert bert["vision"].shape == (4, 10, VISUAL_DIM)
    assert bert["audio"].shape == (4, 10, AUDIO_DIM)
    assert bert["text"].shape == (4, 10, BERT_DIM)
    assert glove["text"].shape == (4, 10, GLOVE_DIM)
    assert bert["labels"].shape == (4, 1)
    assert bert["labels"].min() >= -3.0
    assert bert["labels"].max() <= 3.0


def test_pickle_dict_has_three_splits():
    data = make_pickle_dict(n_train=8, n_valid=4, n_test=4, seed=2)
    assert set(data) == {"train", "valid", "test"}
    assert data["train"]["vision"].shape[0] == 8
    assert data["valid"]["audio"].shape[0] == 4


def test_batch_and_zero_out():
    split = make_split_arrays(48, AffectShapes.for_embedding("bert", max_len=8), seed=3)
    vision, audio, text, labels = batch_from_split(split, batch_size=3, offset=2)
    assert vision.shape[0] == 3
    assert labels.shape == (3, 1)
    z_v, z_a, z_t = zero_unused_modalities(vision, audio, text, keep=["text"])
    assert torch.count_nonzero(z_v) == 0
    assert torch.count_nonzero(z_a) == 0
    assert torch.equal(z_t, text)
    # labels are a projection of text plus weak leaks; the hidden text
    # direction must dominate the visual one
    y = split["labels"].reshape(-1)
    text_score = _projected_score(split["text"], split["_text_dir"])
    vis_score = _projected_score(split["vision"], split["_vision_dir"])
    r_text = float(np.corrcoef(text_score, y)[0, 1])
    r_vis = float(np.corrcoef(vis_score, y)[0, 1])
    assert r_text > 0.8
    assert r_text > r_vis
