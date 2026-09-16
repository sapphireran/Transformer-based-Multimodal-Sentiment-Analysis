"""Shared path setup, devices, and hyper-parameter bundles for examples."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "model"
RESULTS_DIR = MODEL_DIR / "results"
MOSI_RESULTS_DIR = MODEL_DIR / "mosi_test"
DOCS_DIR = REPO_ROOT / "docs"

for path in (MODEL_DIR, REPO_ROOT):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


def cpu_device() -> torch.device:
    """Examples always stay on CPU so they run without a GPU."""
    return torch.device("cpu")


def seed_everything(seed: int = 2024) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class FeatureLayout:
    """Ranks the real loaders emit after ``max_pad=True``."""

    visual: int = 35
    audio: int = 74
    text: int = 768
    seq_len: int = 50
    name: str = "bert"

    @property
    def as_list(self) -> list[int]:
        return [self.visual, self.audio, self.text]

    @property
    def concat_dim(self) -> int:
        return self.visual + self.audio + self.text


BERT_LAYOUT = FeatureLayout(text=768, name="bert")
GLOVE_LAYOUT = FeatureLayout(text=300, name="glove")


class TinyGMTMParams:
    """Small enough for a laptop CPU. Training scripts use 64 / 4 / 4."""

    num_heads = 2
    layers = 1
    attn_dropout = 0.0
    attn_dropout_modalities = [0.0, 0.0, 0.0]
    relu_dropout = 0.0
    res_dropout = 0.0
    out_dropout = 0.0
    embed_dropout = 0.0
    embed_dim = 16
    attn_mask = False
    output_dim = 1
    all_steps = False
    modality_dropout = 0.0
    use_text_transformer = True


class TrainScriptParams:
    """Mirrors ``HParams`` in ``train_GMTM_bert.py`` / ``train_GMTM_glove.py``."""

    num_heads = 4
    layers = 4
    attn_dropout = 0.1
    attn_dropout_modalities = [0, 0, 0.1]
    relu_dropout = 0.1
    res_dropout = 0.1
    out_dropout = 0.1
    embed_dropout = 0.2
    embed_dim = 64
    attn_mask = True
    output_dim = 1
    all_steps = False
    modality_dropout = 0.2
    use_text_transformer = True


def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def finite(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all().item())


def format_shape(t: torch.Tensor) -> str:
    return "x".join(str(s) for s in t.shape)


def print_kv(rows: Iterable[tuple[str, object]]) -> None:
    width = max(len(k) for k, _ in rows)
    for key, value in rows:
        print(f"  {key:<{width}}  {value}")
