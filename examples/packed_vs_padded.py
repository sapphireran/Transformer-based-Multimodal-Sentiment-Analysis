"""Show the two collate layouts used by train() / MultiFramework.

Packed  (_process_1, max_pad=False)
    model([[vision, audio, text], [len_v, len_a, len_t]])
    LSTM / GRU set has_padding=True

Padded  (_process_2, max_pad=True)
    model([vision, audio, text])
    TransformerEarly and GMTM

The real collate lives in model/data/get_dataloader.py. This script only
reproduces the tensor nesting so you can see why mixing is_packed with the
wrong loader crashes.
"""

from __future__ import annotations

from common import MAX_SEQ_LEN
from synthetic_data import make_corpus


def show_packed(corpus) -> None:
    features, lengths, labels = corpus.packed()
    print("packed batch (is_packed=True)")
    print(f"  type(features)     = {type(features).__name__}  len={len(features)}")
    for name, feat, leng in zip(("visual", "audio", "text"), features, lengths):
        print(f"  {name:8s}  tensor {tuple(feat.shape)}  lengths {leng.tolist()}")
    print(f"  labels             {tuple(labels.shape)}")
    print("  train() call        model([[x.to(dev) for x in batch[0]], batch[1]])")


def show_padded(corpus) -> None:
    vision, audio, text, labels = corpus.padded(MAX_SEQ_LEN)
    print("max-padded batch (is_packed=False, max_pad=True)")
    print(f"  visual {tuple(vision.shape)}")
    print(f"  audio  {tuple(audio.shape)}")
    print(f"  text   {tuple(text.shape)}")
    print(f"  labels {tuple(labels.shape)}")
    print("  train() call        model([x.to(dev) for x in batch[:-1]])")


def main() -> None:
    corpus = make_corpus(n=4, text="bert", min_len=6, max_len=18, seed=3)
    print(f"raw lengths: {corpus.lengths.tolist()}")
    print()
    show_packed(corpus)
    print()
    show_padded(corpus)
    print()
    print("Mismatch to avoid:")
    print("  packed loader + is_packed=False  -> Identity/LSTM gets a list")
    print("  padded loader + is_packed=True   -> train() indexes batch[0] as a list of modalities")


if __name__ == "__main__":
    main()
