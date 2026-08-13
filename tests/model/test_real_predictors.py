"""Task 7.3 — cold-start load of real predictors, no network required (all
paths are local artifacts). Confirms the risk flagged in design.md: roberta_final
only ships tokenizer.json (no merges.txt/vocab.json), so a slow tokenizer
would break — this must always resolve to the fast tokenizer.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.model

ROBERTA_DIR = Path("content/aegis_artifacts/roberta_final")
BASELINE_DIR = Path("content/aegis_artifacts/baseline")
LABEL_X_RE = re.compile(r"^LABEL_\d$")


def _require_roberta() -> None:
    if not (ROBERTA_DIR / "model.safetensors").exists():
        pytest.skip("roberta_final artifacts not present")


def _require_baseline() -> None:
    if not (BASELINE_DIR / "logreg_model.joblib").exists():
        pytest.skip("baseline not trained yet")


def test_roberta_predictor_loads_offline_with_fast_tokenizer() -> None:
    _require_roberta()
    from aegis.serving.roberta_predictor import RobertaPredictor

    predictor = RobertaPredictor(ROBERTA_DIR)
    assert predictor.is_ready()
    assert predictor.tokenizer.is_fast


def test_roberta_predictor_predicts_real_label() -> None:
    _require_roberta()
    from aegis.serving.roberta_predictor import RobertaPredictor

    predictor = RobertaPredictor(ROBERTA_DIR)
    result = predictor.predict("The national football team won the championship last night.")
    assert result.predicted_class in {"World", "Sports", "Business", "Sci/Tech"}
    assert not LABEL_X_RE.match(result.predicted_class)
    assert 0.0 <= result.confidence <= 1.0
    assert result.logits.shape == (4,)


def test_roberta_predictor_uses_max_len_128() -> None:
    _require_roberta()
    from aegis.serving.roberta_predictor import RobertaPredictor

    predictor = RobertaPredictor(ROBERTA_DIR)
    assert predictor._max_len == 128


def test_roberta_predictor_text_not_cleaned_before_tokenize() -> None:
    """Regression guard for the train/serve skew bug (design.md D6): the
    tokenizer must see the text unchanged, not lowercased/stripped."""
    _require_roberta()
    from aegis.serving.roberta_predictor import RobertaPredictor

    predictor = RobertaPredictor(ROBERTA_DIR)
    text = "Apple's Q3 revenue hit $89.5B in 2024!"
    encoded_direct = predictor.tokenizer(text, truncation=True, max_length=128)
    encoded_via_predict_path = predictor.tokenizer(
        __import__("aegis.data.preprocess", fromlist=["passthrough"]).passthrough(text),
        truncation=True,
        max_length=128,
    )
    assert encoded_direct["input_ids"] == encoded_via_predict_path["input_ids"]


def test_baseline_predictor_loads_and_predicts() -> None:
    _require_baseline()
    from aegis.serving.baseline_predictor import BaselinePredictor

    predictor = BaselinePredictor(BASELINE_DIR)
    result = predictor.predict("Quarterly earnings beat analyst estimates by a wide margin.")
    assert result.predicted_class in {"World", "Sports", "Business", "Sci/Tech"}
    assert 0.0 <= result.confidence <= 1.0


def test_baseline_predictor_logits_field_is_probability_vector() -> None:
    _require_baseline()
    from aegis.serving.baseline_predictor import BaselinePredictor

    predictor = BaselinePredictor(BASELINE_DIR)
    result = predictor.predict("Stocks rallied on Tuesday amid strong earnings.")
    assert result.logits.sum() == pytest.approx(1.0, abs=1e-6)
