"""Exercises train_baseline()/save_baseline_artifacts() on a tiny synthetic
corpus — no network, no AG News download. Distinct from
tests/model/test_baseline_training.py, which validates the real trained
artifacts on the full dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aegis.models.train_baseline import save_baseline_artifacts, train_baseline

LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]

_CLASS_TEXTS = {
    0: ["government elections policy diplomacy", "president parliament treaty nation"],
    1: ["football match championship goal", "basketball team score playoff"],
    2: ["stock market earnings shares", "company profit revenue investors"],
    3: ["software computer chip processor", "internet technology device release"],
}


def _synthetic_corpus(n_per_class: int = 10) -> tuple[list[str], np.ndarray]:
    texts, labels = [], []
    for cls, phrases in _CLASS_TEXTS.items():
        for i in range(n_per_class):
            texts.append(f"{phrases[i % len(phrases)]} sample {i}")
            labels.append(cls)
    return texts, np.array(labels)


@pytest.mark.unit
def test_train_baseline_returns_valid_result() -> None:
    train_texts, train_labels = _synthetic_corpus(20)
    val_texts, val_labels = _synthetic_corpus(5)
    test_texts, test_labels = _synthetic_corpus(5)

    result = train_baseline(
        train_texts,
        train_labels,
        val_texts,
        val_labels,
        test_texts,
        test_labels,
        LABEL_NAMES,
        c_grid=(0.1, 1.0),
        seed=42,
    )

    assert 0.0 <= result["val_macro_f1"] <= 1.0
    assert 0.0 <= result["test_macro_f1"] <= 1.0
    assert result["best_c"] in (0.1, 1.0)
    assert set(result["cv_results"]["C"]) == {0.1, 1.0}


@pytest.mark.unit
def test_train_baseline_model_has_predict_proba() -> None:
    train_texts, train_labels = _synthetic_corpus(20)
    val_texts, val_labels = _synthetic_corpus(5)
    test_texts, test_labels = _synthetic_corpus(5)

    result = train_baseline(
        train_texts,
        train_labels,
        val_texts,
        val_labels,
        test_texts,
        test_labels,
        LABEL_NAMES,
        c_grid=(1.0,),
    )
    x = result["vectorizer"].transform(["football match championship"])
    proba = result["model"].predict_proba(x)[0]
    assert proba.shape == (4,)
    assert np.isclose(proba.sum(), 1.0, atol=1e-6)


@pytest.mark.unit
def test_save_baseline_artifacts_writes_expected_files(tmp_path: Path) -> None:
    train_texts, train_labels = _synthetic_corpus(20)
    val_texts, val_labels = _synthetic_corpus(5)
    test_texts, test_labels = _synthetic_corpus(5)

    result = train_baseline(
        train_texts,
        train_labels,
        val_texts,
        val_labels,
        test_texts,
        test_labels,
        LABEL_NAMES,
        c_grid=(1.0,),
    )
    save_baseline_artifacts(result, tmp_path)

    assert (tmp_path / "logreg_tfidf_vectorizer.joblib").exists()
    assert (tmp_path / "logreg_model.joblib").exists()
    assert (tmp_path / "baseline_results.json").exists()
