from examples.common import BERT_LAYOUT, GLOVE_LAYOUT
from examples.inspect_pickle_schema import validate_blob
from examples.synthetic_data import make_batch, make_pickle_dict, make_unimodal_batch


def test_bert_batch_ranks():
    batch = make_batch(batch_size=5, seq_len=9, layout=BERT_LAYOUT, seed=1)
    assert batch.vision.shape == (5, 9, 35)
    assert batch.audio.shape == (5, 9, 74)
    assert batch.text.shape == (5, 9, 768)
    assert batch.labels.shape == (5, 1)
    assert batch.labels.min() >= -3.0 and batch.labels.max() <= 3.0


def test_glove_text_width():
    batch = make_batch(batch_size=2, layout=GLOVE_LAYOUT, seed=2)
    assert batch.text.shape[-1] == 300
    assert batch.layout.concat_dim == 409


def test_unimodal_zeroing_matches_loader_contract():
    full = make_batch(batch_size=3, seq_len=6, seed=4)
    text_only = make_unimodal_batch(["text"], full)
    assert text_only.text.abs().sum() > 0
    assert text_only.audio.abs().sum() == 0
    assert text_only.vision.abs().sum() == 0
    assert text_only.labels.equal(full.labels)


def test_pickle_dict_schema():
    blob = make_pickle_dict(n_train=6, n_valid=3, n_test=2)
    assert validate_blob(blob) == []
    assert blob["train"]["vision"].shape == (6, 50, 35)
    assert len(blob["test"]["id"]) == 2
