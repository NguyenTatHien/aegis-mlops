"""Task 7.10 — model validation floors on the real held-out test set.
Sample-based (not full 7.6k) to keep this runnable in CI's scheduled
model-validation workflow without taking forever on CPU."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import f1_score

pytestmark = pytest.mark.model

ROBERTA_DIR = Path("content/aegis_artifacts/roberta_final")
SAMPLE_SIZE = 300
LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


def _require_roberta() -> None:
    if not (ROBERTA_DIR / "model.safetensors").exists():
        pytest.skip("roberta_final artifacts not present")


@pytest.fixture(scope="module")
def sample_predictions():
    _require_roberta()
    from aegis.data.loader import load_ag_news
    from aegis.serving.roberta_predictor import RobertaPredictor

    raw = load_ag_news()
    rng = np.random.default_rng(42)
    idx = rng.choice(len(raw["test"]), size=SAMPLE_SIZE, replace=False)
    texts = [raw["test"]["text"][int(i)] for i in idx]
    labels = [raw["test"]["label"][int(i)] for i in idx]

    predictor = RobertaPredictor(ROBERTA_DIR)
    preds = [LABEL_NAMES.index(predictor.predict(t).predicted_class) for t in texts]
    return np.array(labels), np.array(preds)


def test_macro_f1_floor(sample_predictions) -> None:
    y_true, y_pred = sample_predictions
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    assert macro_f1 >= 0.90


def test_per_class_f1_floor(sample_predictions) -> None:
    y_true, y_pred = sample_predictions
    per_class = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2, 3])
    for cls, f1 in zip(LABEL_NAMES, per_class, strict=True):
        assert f1 >= 0.80, f"{cls} F1 {f1:.3f} below floor 0.80"
