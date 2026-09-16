"""Shared MOSI / MOSEI evaluation helpers.

The training loop in ``train_and_test.py`` used to inline these functions.
They live here so examples and tests can score predictions without importing
``memory_profiler``, constructing a ``MultiFramework``, or calling ``plt.show``.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Union

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score

ArrayLike = Union[np.ndarray, Iterable[float]]


def _as_1d(values: ArrayLike) -> np.ndarray:
    """Detach/copy a prediction or label tensor-like object to a 1-D ndarray."""
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    array = np.asarray(values, dtype=np.float64)
    return array.reshape(-1)


def eval_affect(truths: ArrayLike, results: ArrayLike, exclude_zero: bool = True):
    """Binary F1 and accuracy on the sign of the score.

    Neutral gold labels (exactly 0) are dropped when ``exclude_zero`` is true,
    matching the usual MOSI / MOSEI Acc2 protocol used in ``single_test``.
    """
    test_preds = _as_1d(results)
    test_truth = _as_1d(truths)

    if exclude_zero:
        keep = test_truth != 0
    else:
        keep = np.ones_like(test_truth, dtype=bool)

    if not np.any(keep):
        return 0.0, 0.0

    binary_truth = test_truth[keep] > 0
    binary_preds = test_preds[keep] > 0
    f1 = f1_score(binary_truth, binary_preds, average="binary", zero_division=0)
    accuracy = accuracy_score(binary_truth, binary_preds)
    return float(f1), float(accuracy)


def split_uniform_7(data: ArrayLike) -> np.ndarray:
    """Map ``[-3, 3]`` onto 7 equal-width bins with 1-based indices."""
    data = _as_1d(data)
    step = 6.0 / 7.0
    edges = [-3.0 + i * step for i in range(8)]
    categories = np.digitize(data, edges, right=False)
    return np.clip(categories, 1, 7)


def split_uniform_5(data: ArrayLike) -> np.ndarray:
    """Map ``[-3, 3]`` onto 5 equal-width bins with 1-based indices."""
    data = _as_1d(data)
    step = 6.0 / 5.0
    edges = [-3.0 + i * step for i in range(6)]
    categories = np.digitize(data, edges, right=False)
    return np.clip(categories, 1, 5)


def split_round_7(data: ArrayLike) -> np.ndarray:
    """Alternate Acc7 used in some MOSI papers: round and shift into ``{0..6}``.

    Included so examples can show that this repo's published Acc7 is *not*
    this function. Trainers do not call it.
    """
    clipped = np.clip(_as_1d(data), -3.0, 3.0)
    return np.rint(clipped).astype(np.int64) + 3


def regression_metrics(truths: ArrayLike, preds: ArrayLike) -> Dict[str, float]:
    """MAE, MSE, and Pearson r. Pearson r is 0.0 when a side has zero variance."""
    y = _as_1d(truths)
    y_hat = _as_1d(preds)
    if y.size == 0:
        return {"MAE": 0.0, "MSE": 0.0, "Corr": 0.0}
    mae = float(np.mean(np.abs(y - y_hat)))
    mse = float(np.mean((y - y_hat) ** 2))
    if y.size < 2 or np.allclose(y, y[0]) or np.allclose(y_hat, y_hat[0]):
        corr = 0.0
    else:
        corr, _ = pearsonr(y, y_hat)
        corr = float(corr)
    return {"MAE": mae, "MSE": mse, "Corr": corr}


def compute_affect_metrics(
    truths: ArrayLike,
    preds: ArrayLike,
    exclude_zero: bool = True,
    plot_confusion: bool = False,
    confusion_path: Optional[str] = None,
) -> Dict[str, float]:
    """Full ledger used by the trainers, without requiring a DataLoader.

    When ``plot_confusion`` is true a 7-way matrix is written to
    ``confusion_path`` (or shown interactively if that path is omitted).
    """
    y = _as_1d(truths)
    y_hat = _as_1d(preds)
    reg = regression_metrics(y, y_hat)
    pred_7 = split_uniform_7(y_hat)
    true_7 = split_uniform_7(y)
    pred_5 = split_uniform_5(y_hat)
    true_5 = split_uniform_5(y)
    f1, acc2 = eval_affect(y, y_hat, exclude_zero=exclude_zero)

    if plot_confusion:
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

        matrix = confusion_matrix(true_7, pred_7, labels=list(range(1, 8)))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=matrix, display_labels=list(range(1, 8))
        )
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix (uniform Acc7)")
        if confusion_path:
            plt.savefig(confusion_path, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    return {
        "MAE": reg["MAE"],
        "MSE": reg["MSE"],
        "Corr": reg["Corr"],
        "Acc7_uniform": float(accuracy_score(true_7, pred_7)),
        "Acc5_uniform": float(accuracy_score(true_5, pred_5)),
        "Acc2": acc2,
        "F1": f1,
    }


def format_metrics_row(name: str, metrics: Dict[str, float], digits: int = 4) -> str:
    """One Markdown table row: name plus the six CSV columns."""
    order = ("MAE", "Acc7_uniform", "Acc5_uniform", "Acc2", "Corr", "F1")
    cells = [name] + [f"{metrics[key]:.{digits}f}" for key in order]
    return "| " + " | ".join(cells) + " |"
