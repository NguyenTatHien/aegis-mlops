"""Task 7.14 — regression test: predictions must not silently drift from
tests/fixtures/golden_predictions.json (generated once from the real model
via a throwaway script, not hand-written)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.model

ROBERTA_DIR = Path("content/aegis_artifacts/roberta_final")
GOLDEN_PATH = Path("tests/fixtures/golden_predictions.json")


@pytest.fixture(scope="module")
def predictor():
    if not (ROBERTA_DIR / "model.safetensors").exists():
        pytest.skip("roberta_final artifacts not present")
    from aegis.serving.roberta_predictor import RobertaPredictor

    return RobertaPredictor(ROBERTA_DIR)


@pytest.fixture(scope="module")
def golden_cases() -> list[dict]:
    if not GOLDEN_PATH.exists():
        pytest.skip("golden_predictions.json not generated yet")
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def test_golden_predictions_class_unchanged(predictor, golden_cases: list[dict]) -> None:
    for case in golden_cases:
        result = predictor.predict(case["text"])
        assert result.predicted_class == case["predicted_class"], (
            f"regression: {case['text'][:50]!r} was {case['predicted_class']}, now {result.predicted_class}"
        )


def test_golden_predictions_confidence_within_tolerance(
    predictor, golden_cases: list[dict]
) -> None:
    for case in golden_cases:
        result = predictor.predict(case["text"])
        assert result.confidence == pytest.approx(case["confidence"], abs=0.02), (
            f"regression: {case['text'][:50]!r} confidence drifted from {case['confidence']} to {result.confidence}"
        )
