from __future__ import annotations

import pytest
from pydantic import ValidationError

from aegis.api.schemas import (
    BatchPredictRequest,
    ModelInfoResponse,
    OODResult,
    PredictRequest,
    PredictResponse,
)


@pytest.mark.unit
def test_predict_request_accepts_valid_text() -> None:
    req = PredictRequest(text="Apple reported strong Q3 earnings.")
    assert req.text.startswith("Apple")


@pytest.mark.unit
def test_predict_request_rejects_empty_text() -> None:
    with pytest.raises(ValidationError):
        PredictRequest(text="")


@pytest.mark.unit
def test_predict_request_rejects_whitespace_only_text() -> None:
    with pytest.raises(ValidationError):
        PredictRequest(text="   \n\t  ")


@pytest.mark.unit
def test_predict_request_rejects_missing_field() -> None:
    with pytest.raises(ValidationError):
        PredictRequest.model_validate({})


@pytest.mark.unit
def test_predict_response_accepts_null_ood() -> None:
    resp = PredictResponse(
        predicted_class="Sports",
        confidence=0.94,
        model="roberta",
        ood=None,
        latency_ms=58.2,
        model_version="roberta-v1",
    )
    assert resp.ood is None


@pytest.mark.unit
def test_predict_response_accepts_ood_result() -> None:
    resp = PredictResponse(
        predicted_class="World",
        confidence=0.81,
        model="roberta",
        ood=OODResult(is_ood=False, score=0.01, method="energy", threshold=-5.67),
        latency_ms=61.0,
        model_version="roberta-v1",
    )
    assert resp.ood is not None
    assert resp.ood.method == "energy"


@pytest.mark.unit
def test_predict_response_accepts_svm_model() -> None:
    resp = PredictResponse(
        predicted_class="Business",
        confidence=0.74,
        model="svm",
        ood=None,
        latency_ms=2.1,
        model_version="svm-linearsvc-v1",
    )
    assert resp.model == "svm"
    assert resp.score_type == "probability"
    assert resp.ood is None


@pytest.mark.unit
def test_predict_response_accepts_svm_relative_margin() -> None:
    resp = PredictResponse(
        predicted_class="Sports",
        confidence=0.79,
        score_type="relative_margin",
        model="svm",
        ood=None,
        latency_ms=1.0,
        model_version="svm-linearsvc-v1",
    )
    assert resp.score_type == "relative_margin"


@pytest.mark.unit
def test_predict_response_rejects_unknown_model_name() -> None:
    with pytest.raises(ValidationError):
        PredictResponse(
            predicted_class="World",
            confidence=0.81,
            model="gpt4",  # type: ignore[arg-type]
            latency_ms=1.0,
            model_version="x",
        )


@pytest.mark.unit
def test_batch_predict_request_rejects_empty_list() -> None:
    with pytest.raises(ValidationError):
        BatchPredictRequest(texts=[])


@pytest.mark.unit
def test_batch_predict_request_rejects_blank_entry() -> None:
    with pytest.raises(ValidationError):
        BatchPredictRequest(texts=["valid text", "   "])


@pytest.mark.unit
def test_model_info_response_ood_fields_optional_when_disabled() -> None:
    info = ModelInfoResponse(
        model_name="roberta",
        model_version="roberta-v1",
        macro_f1=0.9517,
        ood_enabled=False,
        max_len=128,
        label_names=["World", "Sports", "Business", "Sci/Tech"],
    )
    assert info.ood_method is None
    assert info.ood_threshold is None
