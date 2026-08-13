"""Fixed-logit predictor for tests and for developing the API/Docker/monitoring
layers before real model artifacts are wired in (design.md D12 step 3)."""

from __future__ import annotations

import numpy as np

from aegis.models.labels import id_to_label
from aegis.serving.base import PredictionResult, Predictor

_FIXED_LOGITS: dict[str, np.ndarray] = {
    "roberta": np.array([0.2, 6.5, 0.1, 0.3]),  # confidently "Sports"
    "baseline": np.array([0.1, 0.2, 0.3, 5.0]),  # confidently "Sci/Tech"
}


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


class MockPredictor:
    def __init__(
        self,
        name: str,
        label_names: list[str] | None = None,
        macro_f1: float = 0.95,
        version: str | None = None,
        ready: bool = True,
    ) -> None:
        self.name = name
        self.version = version or f"{name}-mock-v0"
        self.macro_f1 = macro_f1
        self._ready = ready
        self._label_names = label_names or ["World", "Sports", "Business", "Sci/Tech"]

    def is_ready(self) -> bool:
        return self._ready

    def predict(self, text: str) -> PredictionResult:
        logits = _FIXED_LOGITS.get(self.name, np.array([1.0, 0.0, 0.0, 0.0])).copy()
        probs = _softmax(logits)
        idx = int(np.argmax(probs))
        return PredictionResult(
            predicted_class=id_to_label(idx, self._label_names),
            confidence=float(probs[idx]),
            logits=logits,
            model_version=self.version,
        )


def mock_predictor_factory() -> dict[str, Predictor]:
    return {
        "baseline": MockPredictor("baseline", macro_f1=0.9259),
        "roberta": MockPredictor("roberta", macro_f1=0.9517),
    }
