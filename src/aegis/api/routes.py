from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from aegis.api.dependencies import get_metrics, get_ood_detector, get_predictor
from aegis.api.metrics import Metrics
from aegis.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    ErrorResponse,
    ExplainRequest,
    ExplainResponse,
    ModelInfoResponse,
    OODResult,
    PredictRequest,
    PredictResponse,
)
from aegis.config import get_label_names, get_max_len, get_settings
from aegis.ood.base import OODDetector
from aegis.serving.base import Predictor

router = APIRouter()


def _run_predict(
    text: str, predictor: Predictor, detector: OODDetector, metrics: Metrics, endpoint: str
) -> PredictResponse:
    request_start = time.perf_counter()

    inference_start = time.perf_counter()
    result = predictor.predict(text)
    inference_seconds = time.perf_counter() - inference_start

    ood_result: OODResult | None = None
    ood_is_ood: bool | None = None
    ood_score_value: float | None = None
    ood_method = detector.method if detector.enabled else None
    if detector.enabled:
        score = detector.score(result.logits)
        is_ood = detector.is_ood(score)
        ood_result = OODResult(
            is_ood=is_ood,
            score=score,
            method=detector.method,
            threshold=detector.threshold,  # type: ignore[attr-defined]
        )
        ood_is_ood, ood_score_value = is_ood, score

    latency_seconds = time.perf_counter() - request_start

    metrics.record_prediction(
        endpoint=endpoint,
        model=predictor.name,
        status_code=200,
        predicted_class=result.predicted_class,
        confidence=result.confidence,
        latency_seconds=latency_seconds,
        inference_seconds=inference_seconds,
        text_word_count=len(text.split()),
        ood_method=ood_method,
        ood_is_ood=ood_is_ood,
        ood_score_value=ood_score_value,
    )

    return PredictResponse(
        predicted_class=result.predicted_class,
        confidence=result.confidence,
        model=predictor.name,  # type: ignore[arg-type]
        ood=ood_result,
        latency_ms=latency_seconds * 1000,
        model_version=result.model_version,
    )


@router.post(
    "/v1/predict",
    response_model=PredictResponse,
    summary="Classify a news text",
    description=(
        "Runs the selected model branch (baseline TF-IDF+LogReg or RoBERTa) and returns "
        "the predicted class with a confidence score. When OOD_ENABLED, also flags whether "
        "the text falls outside the AG News training domain."
    ),
    responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def predict(
    body: PredictRequest,
    predictor: Predictor = Depends(get_predictor),
    detector: OODDetector = Depends(get_ood_detector),
    metrics: Metrics = Depends(get_metrics),
) -> PredictResponse:
    return _run_predict(body.text, predictor, detector, metrics, endpoint="/v1/predict")


@router.post(
    "/v1/predict/batch",
    response_model=BatchPredictResponse,
    summary="Classify a batch of news texts",
    description="Same as /v1/predict, applied to a list of texts in one call. Size capped by API_MAX_BATCH_SIZE.",
    responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def predict_batch(
    body: BatchPredictRequest,
    predictor: Predictor = Depends(get_predictor),
    detector: OODDetector = Depends(get_ood_detector),
    metrics: Metrics = Depends(get_metrics),
) -> BatchPredictResponse:
    settings = get_settings()
    if len(body.texts) > settings.api_max_batch_size:
        raise HTTPException(
            status_code=422,
            detail=f"batch size {len(body.texts)} exceeds limit of {settings.api_max_batch_size}",
        )
    results = [
        _run_predict(text, predictor, detector, metrics, endpoint="/v1/predict/batch")
        for text in body.texts
    ]
    return BatchPredictResponse(results=results)


@router.get("/health", summary="Liveness probe — always 200 once the process is up")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get(
    "/ready",
    summary="Readiness probe — 200 only once models are loaded",
    responses={503: {"model": ErrorResponse}},
)
async def ready(request: Request) -> dict[str, str]:
    if not getattr(request.app.state, "ready", False):
        raise HTTPException(status_code=503, detail="models not loaded yet")
    return {"status": "ready"}


@router.get(
    "/v1/model/info",
    response_model=ModelInfoResponse,
    summary="Metadata for the serving model",
    description="Model version, evaluation macro-F1, and the OOD configuration currently in effect.",
)
async def model_info(
    predictor: Predictor = Depends(get_predictor),
    detector: OODDetector = Depends(get_ood_detector),
) -> ModelInfoResponse:
    return ModelInfoResponse(
        model_name=predictor.name,  # type: ignore[arg-type]
        model_version=predictor.version,
        macro_f1=predictor.macro_f1,
        ood_enabled=detector.enabled,
        ood_method=detector.method if detector.enabled else None,
        ood_threshold=detector.threshold if detector.enabled else None,  # type: ignore[attr-defined]
        max_len=get_max_len(),
        label_names=get_label_names(),
    )


@router.post(
    "/v1/explain",
    summary="Token-level explanation (Responsible AI) — reserved, not implemented yet",
    responses={501: {"model": ErrorResponse}},
)
async def explain(body: ExplainRequest) -> ExplainResponse:
    raise HTTPException(status_code=501, detail="explain endpoint not implemented yet")


@router.get("/metrics", summary="Prometheus exposition endpoint")
async def metrics_endpoint(metrics: Metrics = Depends(get_metrics)) -> Response:
    payload, content_type = metrics.exposition()
    return Response(content=payload, media_type=content_type)
