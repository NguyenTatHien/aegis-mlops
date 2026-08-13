"""Task 8.9 — loading a model from the MLflow Model Registry. Needs a live
MLflow server with a registered model (task 8.6), so this is effectively a
`model`-tier concern even though it lives under tests/integration for
directory conventions — skips cleanly if MLflow isn't reachable or nothing
is registered yet, rather than failing the fast CI job.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration

MLFLOW_TRACKING_URI = "http://127.0.0.1:5001"


def _require_mlflow_reachable() -> None:
    import httpx

    try:
        resp = httpx.get(f"{MLFLOW_TRACKING_URI}/health", timeout=3)
    except httpx.TransportError:
        pytest.skip("MLflow tracking server not reachable")
    if resp.status_code != 200:
        pytest.skip("MLflow tracking server not healthy")


def _require_registered_model(name: str) -> None:
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()
    try:
        versions = client.get_latest_versions(name)
    except Exception:
        pytest.skip(f"model '{name}' not registered yet")
    if not versions:
        pytest.skip(f"model '{name}' has no versions registered yet")


def test_load_baseline_model_from_registry() -> None:
    _require_mlflow_reachable()
    _require_registered_model("aegis-baseline")

    import mlflow.sklearn

    model = mlflow.sklearn.load_model("models:/aegis-baseline/Production")
    assert hasattr(model, "predict_proba")


def test_registry_has_baseline_experiment_runs() -> None:
    _require_mlflow_reachable()

    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name("aegis-baseline")
    if experiment is None:
        pytest.skip("aegis-baseline experiment not created yet")

    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) >= 7, "expected at least 7 runs (one per swept C value + parent)"
