"""TF-IDF + LinearSVC predictor used as the original research baseline.

LinearSVC exposes decision margins rather than calibrated probabilities.  We
softmax those margins only to provide a bounded, comparable UI signal; the
result is deliberately treated as a confidence proxy and this branch does not
participate in OOD detection.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np

from aegis.config import get_label_names
from aegis.data.preprocess import clean_text_tfidf
from aegis.models.labels import id_to_label
from aegis.serving.base import PredictionResult

logger = logging.getLogger("aegis.serving.svm")


class SVMPredictor:
    name = "svm"

    def __init__(self, baseline_dir: Path, model_comparison_path: Path | None = None) -> None:
        self._label_names = get_label_names()
        self.vectorizer = joblib.load(baseline_dir / "svm_tfidf_vectorizer.joblib")
        self.model = joblib.load(baseline_dir / "svm_model.joblib")

        self.version = "svm-linearsvc-v1"
        self.macro_f1 = self._load_macro_f1(
            model_comparison_path or baseline_dir.parent / "model_comparison.json"
        )
        self._ready = True

    @staticmethod
    def _load_macro_f1(path: Path) -> float:
        if not path.exists():
            logger.warning("model_comparison.json not found at %s — macro_f1 defaults to 0.0", path)
            return 0.0
        rows = json.loads(path.read_text(encoding="utf-8"))
        row = next((r for r in rows if r["model"] == "TF-IDF + Linear SVM"), None)
        return float(row["test_macro_f1"]) if row else 0.0

    def is_ready(self) -> bool:
        return self._ready

    def predict(self, text: str) -> PredictionResult:
        cleaned = clean_text_tfidf(text)
        x = self.vectorizer.transform([cleaned])
        margins = np.asarray(self.model.decision_function(x)[0], dtype=float)
        proxy = _softmax(margins)
        idx = int(np.argmax(margins))
        return PredictionResult(
            predicted_class=id_to_label(idx, self._label_names),
            confidence=float(proxy[idx]),
            logits=margins,
            model_version=self.version,
            score_type="relative_margin",
        )


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max()
    exp = np.exp(shifted)
    return exp / exp.sum()
