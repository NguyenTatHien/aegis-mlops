"""Task 7.13 — invariance, directional (golden set), determinism."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.model

ROBERTA_DIR = Path("content/aegis_artifacts/roberta_final")

GOLDEN_SET = [
    ("The national football team secured a dramatic victory in extra time last night.", "Sports"),
    ("The championship final drew record crowds to the stadium this weekend.", "Sports"),
    ("Stocks rallied on Tuesday after strong quarterly earnings from major banks.", "Business"),
    ("The central bank raised interest rates to combat rising inflation.", "Business"),
    ("Scientists unveiled a new processor chip with record-breaking performance.", "Sci/Tech"),
    ("The tech company released a major software update for its flagship device.", "Sci/Tech"),
    ("World leaders gathered at the summit to discuss the ongoing diplomatic crisis.", "World"),
    ("The United Nations issued a statement condemning the recent conflict.", "World"),
]


@pytest.fixture(scope="module")
def predictor():
    if not (ROBERTA_DIR / "model.safetensors").exists():
        pytest.skip("roberta_final artifacts not present")
    from aegis.serving.roberta_predictor import RobertaPredictor

    return RobertaPredictor(ROBERTA_DIR)


def test_invariant_to_surrounding_whitespace(predictor) -> None:
    text = "The team won the championship last night."
    base = predictor.predict(text)
    padded = predictor.predict(f"   \n{text}\t  ")
    assert base.predicted_class == padded.predicted_class


def test_deterministic_across_repeated_calls(predictor) -> None:
    text = "Quarterly earnings beat analyst estimates by a wide margin."
    results = [predictor.predict(text) for _ in range(5)]
    classes = {r.predicted_class for r in results}
    confidences = {round(r.confidence, 6) for r in results}
    assert len(classes) == 1
    assert len(confidences) == 1


@pytest.mark.parametrize("text,expected", GOLDEN_SET)
def test_golden_set_directional(predictor, text: str, expected: str) -> None:
    result = predictor.predict(text)
    assert result.predicted_class == expected
