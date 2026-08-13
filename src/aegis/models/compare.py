"""Regenerates the three-model comparison on the same held-out test set."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_model_comparison(
    baseline_result: dict[str, Any],
    roberta_val_f1: float,
    roberta_test_f1: float,
    svm_result: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    comparison = [
        {
            "model": "TF-IDF + LogisticRegression",
            "val_macro_f1": baseline_result["val_macro_f1"],
            "test_macro_f1": baseline_result["test_macro_f1"],
        },
    ]
    if svm_result is not None:
        comparison.append(
            {
                "model": "TF-IDF + Linear SVM",
                "val_macro_f1": svm_result["val_macro_f1"],
                "test_macro_f1": svm_result["test_macro_f1"],
            }
        )
    comparison.append(
        {
            "model": "RoBERTa-base",
            "val_macro_f1": roberta_val_f1,
            "test_macro_f1": roberta_test_f1,
        }
    )
    return comparison


def load_existing_roberta_metrics(model_comparison_path: Path) -> tuple[float, float]:
    """Reads the RoBERTa row out of the existing model_comparison.json —
    RoBERTa isn't retrained by this module (design.md Non-Goals)."""
    data = json.loads(model_comparison_path.read_text(encoding="utf-8"))
    roberta_row = next(row for row in data if row["model"] == "RoBERTa-base")
    return roberta_row["val_macro_f1"], roberta_row["test_macro_f1"]


def save_model_comparison(comparison: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")


if __name__ == "__main__":
    baseline_results_path = Path("content/aegis_artifacts/baseline/baseline_results.json")
    svm_results_path = Path("content/aegis_artifacts/baseline/svm_results.json")
    model_comparison_path = Path("content/aegis_artifacts/model_comparison.json")

    baseline_result = json.loads(baseline_results_path.read_text(encoding="utf-8"))
    svm_result = json.loads(svm_results_path.read_text(encoding="utf-8"))
    roberta_val_f1, roberta_test_f1 = load_existing_roberta_metrics(model_comparison_path)

    comparison = build_model_comparison(
        baseline_result, roberta_val_f1, roberta_test_f1, svm_result
    )
    save_model_comparison(comparison, model_comparison_path)
    print(json.dumps(comparison, indent=2))
