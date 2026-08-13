"""Predictor interface (design.md D12).

Implementations: MockPredictor (group 3, no model), BaselinePredictor,
SVMPredictor and RobertaPredictor (group 7, real artifacts). The API depends on this
Protocol via api/dependencies.py so tests can swap in the mock without
loading any model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class PredictionResult:
    predicted_class: str
    confidence: float
    logits: np.ndarray  # shape (n_classes,) — feeds OODDetector.score()
    model_version: str
    score_type: str = "probability"  # probability | relative_margin


@runtime_checkable
class Predictor(Protocol):
    name: str  # "baseline" | "svm" | "roberta"
    version: str  # static, e.g. "roberta-v1" — distinct from a per-call result
    macro_f1: float  # static, from evaluation on the held-out test set

    def predict(self, text: str) -> PredictionResult: ...

    def is_ready(self) -> bool: ...
