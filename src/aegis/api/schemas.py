"""Pydantic v2 request/response models.

PredictResponse.ood is Optional and defaults to None on purpose (design.md
D4/D13): flipping OOD_ENABLED on/off must never be a breaking schema change.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

ModelName = Literal["baseline", "svm", "roberta"]
ScoreType = Literal["probability", "relative_margin"]


class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
        examples=["The national football team won the championship last night."],
    )

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("text must not be empty or whitespace-only")
        return stripped


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=100)

    @field_validator("texts")
    @classmethod
    def each_non_blank(cls, v: list[str]) -> list[str]:
        for t in v:
            if not t.strip():
                raise ValueError("texts must not contain empty or whitespace-only entries")
        return v


class OODResult(BaseModel):
    is_ood: bool
    score: float
    method: str
    threshold: float


class PredictResponse(BaseModel):
    predicted_class: str
    confidence: float
    score_type: ScoreType = "probability"
    model: ModelName
    ood: OODResult | None = None
    latency_ms: float
    model_version: str


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]


class ErrorResponse(BaseModel):
    error: str
    detail: str
    request_id: str


class ModelInfoResponse(BaseModel):
    model_name: ModelName
    model_version: str
    macro_f1: float
    ood_enabled: bool
    ood_method: str | None = None
    ood_threshold: float | None = None
    max_len: int
    label_names: list[str]


class ExplainRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10_000)
    model: ModelName = "roberta"


class ExplainResponse(BaseModel):
    predicted_class: str
    token_weights: list[tuple[str, float]]
