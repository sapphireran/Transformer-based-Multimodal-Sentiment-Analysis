"""Dataset-free lab helpers for this personal multimodal sentiment repo."""

from .collate import collate_max_pad, collate_variable_length
from .fusion_zoo import build_fusion_zoo, describe_fusion_shapes
from .synthetic import (
    FEATURE_DIMS,
    SentimentBatch,
    make_sentiment_batch,
    zero_modalities,
)
from .toy_train import ToyTrainResult, train_gated_transformer

__all__ = [
    "FEATURE_DIMS",
    "SentimentBatch",
    "build_fusion_zoo",
    "collate_max_pad",
    "collate_variable_length",
    "describe_fusion_shapes",
    "make_sentiment_batch",
    "train_gated_transformer",
    "ToyTrainResult",
    "zero_modalities",
]
