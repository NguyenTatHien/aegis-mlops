from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aegis.eval.evaluate import evaluate_predictions

LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


@pytest.mark.unit
def test_evaluate_predictions_perfect_score(tmp_path: Path) -> None:
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    y_pred = y_true.copy()
    result = evaluate_predictions(y_true, y_pred, LABEL_NAMES, tmp_path, prefix="test")
    assert result["macro_f1"] == pytest.approx(1.0)


@pytest.mark.unit
def test_evaluate_predictions_writes_report_json(tmp_path: Path) -> None:
    y_true = np.array([0, 1, 2, 3])
    y_pred = np.array([0, 1, 3, 2])
    evaluate_predictions(y_true, y_pred, LABEL_NAMES, tmp_path, prefix="test")
    assert (tmp_path / "test_classification_report.json").exists()


@pytest.mark.unit
def test_evaluate_predictions_writes_confusion_matrix_png(tmp_path: Path) -> None:
    y_true = np.array([0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 3])
    evaluate_predictions(y_true, y_pred, LABEL_NAMES, tmp_path, prefix="baseline")
    assert (tmp_path / "baseline_confusion_matrix.png").exists()


@pytest.mark.unit
def test_evaluate_predictions_imperfect_macro_f1_below_one(tmp_path: Path) -> None:
    y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    y_pred = np.array([0, 1, 1, 1, 2, 2, 3, 0])
    result = evaluate_predictions(y_true, y_pred, LABEL_NAMES, tmp_path, prefix="test")
    assert 0.0 < result["macro_f1"] < 1.0
