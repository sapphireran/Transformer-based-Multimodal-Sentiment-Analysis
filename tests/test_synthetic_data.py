from synthetic_data import (
    BERT_DIMS,
    FeatureSpec,
    collate_padded,
    collate_packed,
    make_dataset,
    make_split,
    random_padded_batch,
    random_variable_lengths,
    zero_ablate,
)


def test_split_ranks_and_label_range():
    spec = FeatureSpec()
    split = make_split(10, spec, seed=1)
    assert split["vision"].shape == (10, spec.seq_len, spec.visual)
    assert split["audio"].shape == (10, spec.seq_len, spec.audio)
    assert split["text"].shape == (10, spec.seq_len, spec.text)
    assert split["labels"].shape == (10, 1)
    assert split["labels"].min() >= -3.0
    assert split["labels"].max() <= 3.0


def test_dataset_keys():
    data = make_dataset(n_train=4, n_valid=3, n_test=2, seed=2)
    assert set(data) == {"train", "valid", "test"}
    assert data["train"]["vision"].shape[0] == 4
    assert data["valid"]["vision"].shape[0] == 3
    assert data["test"]["vision"].shape[0] == 2


def test_official_bert_widths_documented():
    assert BERT_DIMS == {"visual": 35, "audio": 74, "text": 768}


def test_padded_batch_device_cpu():
    streams, labels = random_padded_batch(batch_size=5, seed=4)
    assert len(streams) == 3
    assert streams[0].shape[0] == 5
    assert labels.shape == (5, 1)
    assert streams[0].device.type == "cpu"


def test_packed_collate_pads_to_max_length():
    rows = random_variable_lengths(batch_size=6, min_len=4, seed=8)
    lengths = [row[0].size(0) for row in rows]
    features, feat_lengths, indices, labels = collate_packed(rows)
    assert feat_lengths[0].tolist() == lengths
    assert features[0].shape[1] == max(lengths)
    assert indices.shape == (6, 1)
    assert labels.shape == (6, 1)


def test_padded_collate_fixed_time():
    rows = random_variable_lengths(batch_size=4, min_len=4, seed=8)
    vision, audio, text, labels = collate_padded(rows, max_len=12)
    assert vision.shape == (4, 12, 8)
    assert audio.shape == (4, 12, 10)
    assert text.shape == (4, 12, 16)
    assert labels.shape == (4, 1)


def test_zero_ablate_keeps_text_only():
    streams, _ = random_padded_batch(batch_size=3, seed=1)
    ablated = zero_ablate(streams, keep=("text",))
    assert ablated[0].abs().sum().item() == 0.0
    assert ablated[1].abs().sum().item() == 0.0
    assert ablated[2].abs().sum().item() > 0.0
