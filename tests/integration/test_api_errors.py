from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.mark.integration
def test_predict_empty_text_returns_422(client: TestClient) -> None:
    resp = client.post("/v1/predict", json={"text": ""})
    assert resp.status_code == 422
    body = resp.json()
    assert "request_id" in body
    assert "detail" in body


@pytest.mark.integration
def test_predict_whitespace_only_text_returns_422(client: TestClient) -> None:
    resp = client.post("/v1/predict", json={"text": "   \n\t  "})
    assert resp.status_code == 422


@pytest.mark.integration
def test_predict_missing_field_returns_422(client: TestClient) -> None:
    resp = client.post("/v1/predict", json={})
    assert resp.status_code == 422


@pytest.mark.integration
def test_predict_batch_over_limit_returns_422(client: TestClient) -> None:
    texts = [f"text number {i}" for i in range(200)]
    resp = client.post("/v1/predict/batch", json={"texts": texts})
    assert resp.status_code == 422


@pytest.mark.integration
def test_predict_batch_empty_list_returns_422(client: TestClient) -> None:
    resp = client.post("/v1/predict/batch", json={"texts": []})
    assert resp.status_code == 422


@pytest.mark.integration
def test_error_response_never_leaks_stack_trace(client: TestClient) -> None:
    resp = client.post("/v1/predict", json={"text": ""})
    body = resp.json()
    assert "Traceback" not in body["detail"]
    assert 'File "' not in body["detail"]


@pytest.mark.integration
def test_response_has_request_id_header(client: TestClient) -> None:
    resp = client.get("/health")
    assert "x-request-id" in {k.lower() for k in resp.headers}


@pytest.mark.integration
def test_different_requests_get_different_request_ids(client: TestClient) -> None:
    r1 = client.get("/health")
    r2 = client.get("/health")
    assert r1.headers["x-request-id"] != r2.headers["x-request-id"]
