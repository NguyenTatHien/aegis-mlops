from __future__ import annotations

import pytest

import aegis.config as config_module
from aegis.api.dependencies import default_predictor_factory
from aegis.serving.mock_predictor import MockPredictor


@pytest.mark.unit
def test_default_predictor_factory_mock_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_SOURCE", "mock")
    config_module.get_settings.cache_clear()
    try:
        predictors = default_predictor_factory()
        assert set(predictors) == {"baseline", "svm", "roberta"}
        assert isinstance(predictors["baseline"], MockPredictor)
        assert isinstance(predictors["svm"], MockPredictor)
        assert isinstance(predictors["roberta"], MockPredictor)
    finally:
        config_module.get_settings.cache_clear()


@pytest.mark.unit
def test_default_predictor_factory_registry_source_calls_load_from_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registry mode routes to aegis.serving.mlflow_predictor.load_from_registry
    (task 8.7) rather than silently falling back to local/mock. Patches that
    function directly instead of pointing at a real (or fake) MLflow server —
    even a fast-failing connection involves mlflow's REST client retry/backoff,
    which has no place in the unit tier (real MLflow registry behavior is
    covered by tests/integration/test_mlflow_integration.py instead)."""
    monkeypatch.setenv("MODEL_SOURCE", "registry")
    config_module.get_settings.cache_clear()

    import aegis.serving.mlflow_predictor as mlflow_predictor_module

    sentinel = RuntimeError("sentinel: load_from_registry was called")

    def _raise(_settings: object) -> dict:
        raise sentinel

    monkeypatch.setattr(mlflow_predictor_module, "load_from_registry", _raise)

    try:
        with pytest.raises(RuntimeError, match="sentinel"):
            default_predictor_factory()
    finally:
        config_module.get_settings.cache_clear()
