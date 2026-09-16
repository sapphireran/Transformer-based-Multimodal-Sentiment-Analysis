"""Import model.models without requiring the caller to sit in model/."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

from .repo import MODEL_DIR


def load_models() -> ModuleType:
    model_dir = str(MODEL_DIR)
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    return importlib.import_module("models")
