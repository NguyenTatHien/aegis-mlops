from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

_LABEL_X_RE = re.compile(r"^LABEL_\d$")


@pytest.mark.integration
def test_predict_default_model_is_roberta(client: TestClient) -> None:
    resp = client.post("/v1/predict", json={"text": "The team won the championship last night."})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "roberta"
    assert body["predicted_class"] in {"World", "Sports", "Business", "Sci/Tech"}
    assert not _LABEL_X_RE.match(body["predicted_class"])
    assert 0.0 <= body["confidence"] <= 1.0
    assert "latency_ms" in body
    assert "model_version" in body


@pytest.mark.integration
def test_predict_baseline_branch(client: TestClient) -> None:
    resp = client.post(
        "/v1/predict?model=baseline", json={"text": "Quarterly earnings beat estimates."}
    )
    assert resp.status_code == 200
    assert resp.json()["model"] == "baseline"


@pytest.mark.integration
def test_predict_invalid_model_name_rejected(client: TestClient) -> None:
    resp = client.post("/v1/predict?model=gpt4", json={"text": "hello"})
    assert resp.status_code == 422


@pytest.mark.integration
def test_predict_ood_field_null_when_disabled(client: TestClient) -> None:
    resp = client.post("/v1/predict", json={"text": "Stocks rallied on Tuesday."})
    assert resp.status_code == 200
    assert resp.json()["ood"] is None


@pytest.mark.integration
def test_predict_ood_field_populated_when_enabled(client_ood_enabled: TestClient) -> None:
    resp = client_ood_enabled.post("/v1/predict", json={"text": "Stocks rallied on Tuesday."})
    assert resp.status_code == 200
    ood = resp.json()["ood"]
    assert ood is not None
    assert set(ood) == {"is_ood", "score", "method", "threshold"}
    assert ood["method"] == "energy"


@pytest.mark.integration
def test_predict_batch_preserves_order(client: TestClient) -> None:
    texts = ["Team wins the cup.", "Markets close higher.", "New chip unveiled."]
    resp = client.post("/v1/predict/batch", json={"texts": texts})
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) == 3


@pytest.mark.integration
def test_health_always_ok(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.integration
def test_ready_ok_when_models_loaded(client: TestClient) -> None:
    resp = client.get("/ready")
    assert resp.status_code == 200


@pytest.mark.integration
def test_ready_503_when_startup_failed(client_not_ready: TestClient) -> None:
    resp = client_not_ready.get("/ready")
    assert resp.status_code == 503


@pytest.mark.integration
def test_health_ok_even_when_not_ready(client_not_ready: TestClient) -> None:
    resp = client_not_ready.get("/health")
    assert resp.status_code == 200


@pytest.mark.integration
def test_predict_503_when_startup_failed(client_not_ready: TestClient) -> None:
    resp = client_not_ready.post("/v1/predict", json={"text": "hello there"})
    assert resp.status_code == 503


@pytest.mark.integration
def test_model_info_contains_required_fields(client: TestClient) -> None:
    resp = client.get("/v1/model/info")
    assert resp.status_code == 200
    body = resp.json()
    for key in ["model_name", "model_version", "macro_f1", "ood_enabled", "max_len", "label_names"]:
        assert key in body
    assert body["label_names"] == ["World", "Sports", "Business", "Sci/Tech"]
    assert body["max_len"] == 128


@pytest.mark.integration
def test_model_info_ood_disabled_by_default(client: TestClient) -> None:
    resp = client.get("/v1/model/info")
    body = resp.json()
    assert body["ood_enabled"] is False
    assert body["ood_method"] is None
    assert body["ood_threshold"] is None


@pytest.mark.integration
def test_model_info_ood_enabled_reflects_state(client_ood_enabled: TestClient) -> None:
    resp = client_ood_enabled.get("/v1/model/info")
    body = resp.json()
    assert body["ood_enabled"] is True
    assert body["ood_method"] == "energy"
    assert body["ood_threshold"] == pytest.approx(-4.313697454571079)


@pytest.mark.integration
def test_explain_returns_501_not_implemented(client: TestClient) -> None:
    resp = client.post("/v1/explain", json={"text": "hello there", "model": "roberta"})
    assert resp.status_code == 501
