"""Real RoBERTa predictor.

Two contract points enforced by design.md: text goes in raw — passthrough(),
never clean_text_tfidf() (D6, train/serve skew) — and max_len is read from
ood_config.json, never hardcoded or taken from tokenizer_config.json's
model_max_length=512 (D8, that value doesn't match what the model was
trained/calibrated at, 128).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from aegis.config import get_label_names, get_max_len
from aegis.data.preprocess import passthrough
from aegis.models.labels import id_to_label
from aegis.serving.base import PredictionResult

logger = logging.getLogger("aegis.serving.roberta")

torch.set_num_threads(4)


class RobertaPredictor:
    name = "roberta"

    def __init__(self, model_dir: Path, model_comparison_path: Path | None = None) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._label_names = get_label_names()
        self._max_len = get_max_len()

        # model_dir is always a local directory (content/aegis_artifacts/roberta_final),
        # never a Hub repo id — bandit's B615 (unpinned Hub download) is a false
        # positive here since transformers resolves local paths without any
        # network call.
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)  # nosec B615
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)  # nosec B615
        self.model.eval()

        self.version = "roberta-final"
        self.macro_f1 = self._load_macro_f1(
            model_comparison_path or model_dir.parent / "model_comparison.json"
        )
        self._ready = True

    @staticmethod
    def _load_macro_f1(path: Path) -> float:
        if not path.exists():
            logger.warning("model_comparison.json not found at %s — macro_f1 defaults to 0.0", path)
            return 0.0
        rows = json.loads(path.read_text(encoding="utf-8"))
        row = next((r for r in rows if r["model"] == "RoBERTa-base"), None)
        return float(row["test_macro_f1"]) if row else 0.0

    def is_ready(self) -> bool:
        return self._ready

    def predict(self, text: str) -> PredictionResult:
        # .strip() only — NOT clean_text_tfidf(). Trimming incidental
        # leading/trailing whitespace is input hygiene (a client's
        # copy-paste artifact shouldn't change the label); lowercasing or
        # stripping punctuation/digits would be the train/serve skew this
        # predictor exists to avoid (design.md D6).
        raw_text = passthrough(text).strip()
        encoded = self.tokenizer(
            raw_text, truncation=True, max_length=self._max_len, padding=False, return_tensors="pt"
        )
        with torch.no_grad():
            logits = self.model(**encoded).logits[0].numpy()

        probs = _softmax(logits)
        idx = int(np.argmax(probs))
        return PredictionResult(
            predicted_class=id_to_label(idx, self._label_names),
            confidence=float(probs[idx]),
            logits=logits,
            model_version=self.version,
        )


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()
