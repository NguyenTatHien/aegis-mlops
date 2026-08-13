from __future__ import annotations

import json
from pathlib import Path

import pytest

from aegis.models.compare import (
    build_model_comparison,
    load_existing_roberta_metrics,
    save_model_comparison,
)


@pytest.mark.unit
def test_build_model_comparison_has_all_three_models() -> None:
    baseline_result = {"val_macro_f1": 0.9266, "test_macro_f1": 0.9249}
    svm_result = {"val_macro_f1": 0.9237, "test_macro_f1": 0.9259}
    comparison = build_model_comparison(
        baseline_result,
        roberta_val_f1=0.9511,
        roberta_test_f1=0.9517,
        svm_result=svm_result,
    )

    models = {row["model"] for row in comparison}
    assert "RoBERTa-base" in models
    assert "TF-IDF + LogisticRegression" in models
    assert "TF-IDF + Linear SVM" in models
    for row in comparison:
        assert "val_macro_f1" in row and "test_macro_f1" in row


@pytest.mark.unit
def test_load_existing_roberta_metrics_reads_correct_row(tmp_path: Path) -> None:
    path = tmp_path / "model_comparison.json"
    path.write_text(
        json.dumps(
            [
                {"model": "TF-IDF + Linear SVM", "val_macro_f1": 0.92, "test_macro_f1": 0.93},
                {"model": "RoBERTa-base", "val_macro_f1": 0.9511, "test_macro_f1": 0.9517},
            ]
        ),
        encoding="utf-8",
    )
    val_f1, test_f1 = load_existing_roberta_metrics(path)
    assert val_f1 == pytest.approx(0.9511)
    assert test_f1 == pytest.approx(0.9517)


@pytest.mark.unit
def test_save_model_comparison_writes_valid_json(tmp_path: Path) -> None:
    comparison = [{"model": "A", "val_macro_f1": 0.9, "test_macro_f1": 0.91}]
    path = tmp_path / "out" / "model_comparison.json"
    save_model_comparison(comparison, path)

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == comparison
