"""Dependency wiring. get_predictor / get_ood_detector are what make routes
testable with MockPredictor — override them (or pass a predictor_factory to
create_app) instead of monkeypatching internals."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import Depends, HTTPException, Query, Request

from aegis.api.metrics import Metrics
from aegis.api.schemas import ModelName
from aegis.config import get_ood_config, get_settings
from aegis.ood.base import OODDetector
from aegis.ood.detector import build_detector
from aegis.serving.base import Predictor

PredictorFactory = Callable[[], dict[str, Predictor]]

# roberta_final's logits suit MSP/Energy; baseline's LogisticRegression only
# has predict_proba, so it pairs with the entropy detector (design.md D4).
_METHOD_BY_MODEL: dict[str, str] = {"roberta": "energy", "baseline": "entropy"}


def default_predictor_factory() -> dict[str, Predictor]:
    """Loads real predictors per settings.model_source. Imports are inside
    the function body (not module-level) so importing this module never
    requires torch/transformers to be installed — only invoking this
    function does, which only happens at app startup, not at test-collection
    time when tests supply their own MockPredictor-based factory."""
    settings = get_settings()
    if settings.model_source == "registry":
        from aegis.serving.mlflow_predictor import load_from_registry

        return load_from_registry(settings)

    if settings.model_source == "mock":
        # Lightweight bring-up path for Docker/Compose/CI runs that need a
        # working container without a 498MB model or torch even installed —
        # not a shortcut around "local"/"registry", a documented third mode.
        from aegis.serving.mock_predictor import mock_predictor_factory

        return mock_predictor_factory()

    from aegis.serving.baseline_predictor import BaselinePredictor
    from aegis.serving.roberta_predictor import RobertaPredictor

    return {
        "baseline": BaselinePredictor(settings.baseline_dir),
        "roberta": RobertaPredictor(settings.roberta_model_dir),
    }


def get_predictor_registry(request: Request) -> dict[str, Predictor]:
    return request.app.state.predictors


def get_predictor(
    model: ModelName = Query("roberta", description="Which model branch to serve the request"),
    registry: dict[str, Predictor] = Depends(get_predictor_registry),
) -> Predictor:
    predictor = registry.get(model)
    if predictor is None or not predictor.is_ready():
        raise HTTPException(status_code=503, detail=f"model '{model}' is not ready")
    return predictor


def get_ood_detector(predictor: Predictor = Depends(get_predictor)) -> OODDetector:
    settings = get_settings()
    method = _METHOD_BY_MODEL.get(predictor.name, "energy")
    return build_detector(method, get_ood_config(), ood_enabled=settings.ood_enabled)


def get_metrics(request: Request) -> Metrics:
    return request.app.state.metrics
