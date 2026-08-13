"""Regenerates model_comparison.json on the same test set / same metrics for
both branches (spec: ml-pipeline "Model comparison trên cùng test set")."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_model_comparison(
    baseline_result: dict[str, Any], roberta_val_f1: float, roberta_test_f1: float
) -> list[dict[str, Any]]:
    return [
        {
            "model": "TF-IDF + LogisticRegression",
            "val_macro_f1": baseline_result["val_macro_f1"],
            "test_macro_f1": baseline_result["test_macro_f1"],
        },
        {
            "model": "RoBERTa-base",
            "val_macro_f1": roberta_val_f1,
            "test_macro_f1": roberta_test_f1,
        },
    ]


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
    model_comparison_path = Path("content/aegis_artifacts/model_comparison.json")

    baseline_result = json.loads(baseline_results_path.read_text(encoding="utf-8"))
    roberta_val_f1, roberta_test_f1 = load_existing_roberta_metrics(model_comparison_path)

    comparison = build_model_comparison(baseline_result, roberta_val_f1, roberta_test_f1)
    save_model_comparison(comparison, model_comparison_path)
    print(json.dumps(comparison, indent=2))
