from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

_EXTERNAL_REF_RE = re.compile(r'(?:src|href)\s*=\s*["\'](https?:)?//', re.IGNORECASE)


@pytest.mark.integration
def test_root_serves_html(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "<title>" in resp.text


@pytest.mark.integration
def test_root_has_no_external_script_or_link_references(client: TestClient) -> None:
    resp = client.get("/")
    assert not _EXTERNAL_REF_RE.search(resp.text)


@pytest.mark.integration
def test_static_mount_does_not_shadow_api_routes(client: TestClient) -> None:
    assert client.get("/health").status_code == 200
    assert client.get("/v1/model/info").status_code == 200
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200
    assert client.get("/metrics").status_code == 200


@pytest.mark.integration
def test_page_does_not_hardcode_label_names(client: TestClient) -> None:
    resp = client.get("/")
    for label in ["World", "Sports", "Business", "Sci/Tech"]:
        assert label not in resp.text


@pytest.mark.integration
def test_page_fetches_model_info_and_predict_endpoints() -> None:
    from pathlib import Path

    html = (
        Path(__file__).resolve().parents[2] / "src" / "aegis" / "api" / "static" / "index.html"
    ).read_text(encoding="utf-8")
    assert "/v1/model/info" in html
    assert "/v1/predict" in html
