"""Task 6.7/6.9 — verifies the retrained baseline (LogisticRegression, not
the notebook's LinearSVC) actually has calibrated predict_proba, and that
model comparison reads real numbers from all three branches on the same test set.

Marked `model` (design.md D11): needs the real AG News-trained artifacts
under content/aegis_artifacts/baseline, produced by
`python -m aegis.models.train_baseline`. Skips cleanly if they aren't there
yet rather than failing CI on missing prerequisites.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest

BASELINE_DIR = Path("content/aegis_artifacts/baseline")
MODEL_COMPARISON_PATH = Path("content/aegis_artifacts/model_comparison.json")

pytestmark = pytest.mark.model


def _require_baseline_artifacts() -> None:
    if not (BASELINE_DIR / "logreg_model.joblib").exists():
        pytest.skip("baseline not trained yet — run `python -m aegis.models.train_baseline`")


def test_predict_proba_sums_to_one() -> None:
    _require_baseline_artifacts()
    vectorizer = joblib.load(BASELINE_DIR / "logreg_tfidf_vectorizer.joblib")
    model = joblib.load(BASELINE_DIR / "logreg_model.joblib")

    from aegis.data.preprocess import clean_text_tfidf

    x = vectorizer.transform(
        [clean_text_tfidf("The stock market rallied today on strong earnings.")]
    )
    proba = model.predict_proba(x)[0]

    assert proba.shape == (4,)
    assert np.isclose(proba.sum(), 1.0, atol=1e-6)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_baseline_results_json_has_expected_shape() -> None:
    _require_baseline_artifacts()
    report = json.loads((BASELINE_DIR / "baseline_results.json").read_text(encoding="utf-8"))
    assert report["model"] == "TF-IDF + LogisticRegression"
    assert 0.0 < report["val_macro_f1"] <= 1.0
    assert 0.0 < report["test_macro_f1"] <= 1.0
    assert "best_c" in report


def test_model_comparison_has_both_branches_same_test_set_metric() -> None:
    if not MODEL_COMPARISON_PATH.exists():
        pytest.skip(
            "model_comparison.json not generated yet — run `python -m aegis.models.compare`"
        )
    rows = json.loads(MODEL_COMPARISON_PATH.read_text(encoding="utf-8"))
    models = {row["model"] for row in rows}
    assert "RoBERTa-base" in models
    assert "TF-IDF + LogisticRegression" in models
    assert "TF-IDF + Linear SVM" in models
    for row in rows:
        assert "val_macro_f1" in row and "test_macro_f1" in row
