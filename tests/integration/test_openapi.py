from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.mark.integration
def test_openapi_json_is_valid(client: TestClient) -> None:
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    spec = resp.json()
    assert "paths" in spec
    for expected in [
        "/v1/predict",
        "/v1/predict/batch",
        "/health",
        "/ready",
        "/v1/model/info",
        "/v1/explain",
    ]:
        assert expected in spec["paths"]


@pytest.mark.integration
def test_every_path_has_non_empty_summary(client: TestClient) -> None:
    spec = client.get("/openapi.json").json()
    for path, methods in spec["paths"].items():
        for method, details in methods.items():
            if method.lower() not in {"get", "post", "put", "delete", "patch"}:
                continue
            assert details.get("summary", "").strip() != "", (
                f"{method.upper()} {path} has no summary"
            )


@pytest.mark.integration
def test_docs_page_available(client: TestClient) -> None:
    resp = client.get("/docs")
    assert resp.status_code == 200
