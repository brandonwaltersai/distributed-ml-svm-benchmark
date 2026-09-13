from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score


@dataclass(frozen=True)
class BinaryMetrics:
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def evaluate_binary(y_true: Iterable[int], y_pred: Iterable[int]) -> BinaryMetrics:
    """Return a consistent metric set for imbalanced binary classification."""
    return BinaryMetrics(
        accuracy=accuracy_score(y_true, y_pred),
        balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
    )
