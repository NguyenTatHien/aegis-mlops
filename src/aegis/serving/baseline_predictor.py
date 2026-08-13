"""Real baseline predictor: TF-IDF + LogisticRegression (design.md D3 —
LogisticRegression, not the notebook's LinearSVC, specifically so
predict_proba exists for both `confidence` and entropy-based OOD)."""

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

logger = logging.getLogger("aegis.serving.baseline")


class BaselinePredictor:
    name = "baseline"

    def __init__(self, baseline_dir: Path) -> None:
        self._label_names = get_label_names()

        self.vectorizer = joblib.load(baseline_dir / "logreg_tfidf_vectorizer.joblib")
        self.model = joblib.load(baseline_dir / "logreg_model.joblib")

        self.version = "baseline-logreg-v1"
        self.macro_f1 = self._load_macro_f1(baseline_dir / "baseline_results.json")
        self._ready = True

    @staticmethod
    def _load_macro_f1(path: Path) -> float:
        if not path.exists():
            logger.warning("baseline_results.json not found at %s — macro_f1 defaults to 0.0", path)
            return 0.0
        report = json.loads(path.read_text(encoding="utf-8"))
        return float(report.get("test_macro_f1", 0.0))

    def is_ready(self) -> bool:
        return self._ready

    def predict(self, text: str) -> PredictionResult:
        cleaned = clean_text_tfidf(text)
        x = self.vectorizer.transform([cleaned])
        probs = self.model.predict_proba(x)[0]
        idx = int(np.argmax(probs))
        return PredictionResult(
            predicted_class=id_to_label(idx, self._label_names),
            confidence=float(probs[idx]),
            logits=probs,  # entropy detector consumes probabilities, not raw logits
            model_version=self.version,
        )
