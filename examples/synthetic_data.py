"""Build MOSI/MOSEI-shaped tensors without downloading the datasets.

The real loaders read a pickle with keys vision / audio / text / labels.
This module produces the same ranks so fusion and GMTM examples can run
offline. Labels are drawn from a clipped Gaussian on [-3, 3], the interval
used by Acc5 / Acc7 binning.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from common import MAX_SEQ_LEN, device, feature_dims


@dataclass
class SyntheticCorpus:
    """Variable-length utterances plus a max-padded view of the same data."""

    vision: list[torch.Tensor]
    audio: list[torch.Tensor]
    text: list[torch.Tensor]
    labels: torch.Tensor
    lengths: torch.Tensor
    text_name: str

    def padded(self, max_len: int = MAX_SEQ_LEN) -> tuple[torch.Tensor, ...]:
        vis, aud, txt = [], [], []
        for v, a, t in zip(self.vision, self.audio, self.text):
            vis.append(_crop_or_pad(v, max_len))
            aud.append(_crop_or_pad(a, max_len))
            txt.append(_crop_or_pad(t, max_len))
        return torch.stack(vis), torch.stack(aud), torch.stack(txt), self.labels

    def packed(self) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        """Pad to the max length *in this corpus* and return length vectors."""
        max_len = int(self.lengths.max().item())
        vis, aud, txt = [], [], []
        for v, a, t in zip(self.vision, self.audio, self.text):
            vis.append(_crop_or_pad(v, max_len))
            aud.append(_crop_or_pad(a, max_len))
            txt.append(_crop_or_pad(t, max_len))
        features = [torch.stack(vis), torch.stack(aud), torch.stack(txt)]
        lengths = [self.lengths.clone(), self.lengths.clone(), self.lengths.clone()]
        return features, lengths, self.labels

    def zero_modalities(self, keep: set[str], max_len: int = MAX_SEQ_LEN):
        """Ablation protocol: keep named streams, zero the others."""
        v, a, t, y = self.padded(max_len)
        if "visual" not in keep:
            v = torch.zeros_like(v)
        if "audio" not in keep:
            a = torch.zeros_like(a)
        if "text" not in keep:
            t = torch.zeros_like(t)
        return v, a, t, y


def _crop_or_pad(seq: torch.Tensor, max_len: int) -> torch.Tensor:
    t, f = seq.shape
    if t >= max_len:
        return seq[:max_len]
    out = seq.new_zeros(max_len, f)
    out[:t] = seq
    return out


def make_corpus(
    n: int = 8,
    text: str = "bert",
    min_len: int = 8,
    max_len: int = 40,
    seed: int = 0,
    dev: torch.device | None = None,
) -> SyntheticCorpus:
    """Random utterances with a weak linear relationship from text to label.

    The last-timestep mean of the text stream is scaled into [-3, 3] and
    mixed with noise. That gives GMTM / late fusion something to overfit
    without pretending the noise is a real dataset.
    """
    if dev is None:
        dev = device()
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    fv, fa, ft = feature_dims(text)

    lengths = torch.randint(min_len, max_len + 1, (n,), generator=g)
    vision, audio, text_seqs = [], [], []
    raw_scores = []
    for length in lengths.tolist():
        v = torch.randn(length, fv, generator=g)
        a = torch.randn(length, fa, generator=g)
        t = torch.randn(length, ft, generator=g)
        # Plant a detectable signal in the text stream.
        score = t.mean() * 4.0
        t = t + 0.15 * score
        vision.append(v.to(dev))
        audio.append(a.to(dev))
        text_seqs.append(t.to(dev))
        raw_scores.append(score)

    labels = torch.stack(raw_scores).to(dev).unsqueeze(1)
    labels = labels + 0.25 * torch.randn(n, 1, generator=g).to(dev)
    labels = labels.clamp(-3.0, 3.0)
    return SyntheticCorpus(
        vision=vision,
        audio=audio,
        text=text_seqs,
        labels=labels,
        lengths=lengths.to(dev),
        text_name=text,
    )


def describe_corpus(corpus: SyntheticCorpus) -> str:
    v, a, t, y = corpus.padded()
    lines = [
        f"text encoder : {corpus.text_name}",
        f"utterances   : {len(corpus.vision)}",
        f"lengths      : {corpus.lengths.tolist()}",
        f"padded vision: {tuple(v.shape)}",
        f"padded audio : {tuple(a.shape)}",
        f"padded text  : {tuple(t.shape)}",
        f"labels       : {tuple(y.shape)}  min={y.min().item():.3f}  max={y.max().item():.3f}",
    ]
    return "\n".join(lines)


def main() -> None:
    print("=== BERT-shaped synthetic corpus ===")
    bert = make_corpus(n=6, text="bert", seed=1)
    print(describe_corpus(bert))
    print()
    print("=== GloVe-shaped synthetic corpus ===")
    glove = make_corpus(n=6, text="glove", seed=2)
    print(describe_corpus(glove))
    print()
    print("=== Ablation zeroing (keep text only) ===")
    v, a, t, _ = bert.zero_modalities({"text"})
    print(f"visual abs-sum={v.abs().sum().item():.4f}  (expect 0)")
    print(f"audio  abs-sum={a.abs().sum().item():.4f}  (expect 0)")
    print(f"text   abs-sum={t.abs().sum().item():.4f}  (expect > 0)")


if __name__ == "__main__":
    main()
