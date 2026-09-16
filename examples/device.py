"""Single place to pick CPU vs CUDA for the examples."""

from __future__ import annotations

import torch


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def describe_device(device: torch.device | None = None) -> str:
    device = device or get_device()
    extra = ""
    if device.type == "cuda":
        extra = f" ({torch.cuda.get_device_name(device)})"
    return f"{device}{extra}"
