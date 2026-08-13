"""Task 5.13 — Grafana dashboard/datasource must be committed as code, not
click-configured, so a clean `docker compose up` shows the dashboard
without manual import (containerized-deployment spec)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PROVISIONING_DIR = REPO_ROOT / "monitoring" / "grafana" / "provisioning"


@pytest.mark.integration
def test_datasource_file_exists_and_parses() -> None:
    path = PROVISIONING_DIR / "datasources" / "prometheus.yml"
    assert path.exists()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["datasources"][0]["type"] == "prometheus"
    assert data["datasources"][0]["uid"] == "prometheus"


@pytest.mark.integration
def test_dashboard_provider_file_exists_and_parses() -> None:
    path = PROVISIONING_DIR / "dashboards" / "dashboards.yml"
    assert path.exists()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["providers"][0]["type"] == "file"


@pytest.mark.integration
def test_dashboard_json_is_valid() -> None:
    path = PROVISIONING_DIR / "dashboards" / "aegis.json"
    assert path.exists()
    dashboard = json.loads(path.read_text(encoding="utf-8"))
    assert dashboard["title"]
    assert isinstance(dashboard["panels"], list) and len(dashboard["panels"]) > 0


@pytest.mark.integration
def test_dashboard_has_system_row_panels() -> None:
    dashboard = json.loads(
        (PROVISIONING_DIR / "dashboards" / "aegis.json").read_text(encoding="utf-8")
    )
    titles = [p["title"] for p in dashboard["panels"]]
    for expected in [
        "Requests per second",
        "p95 Latency by model (SLO: 500ms)",
        "Error rate",
        "API up",
    ]:
        assert expected in titles


@pytest.mark.integration
def test_dashboard_has_ml_row_panels() -> None:
    dashboard = json.loads(
        (PROVISIONING_DIR / "dashboards" / "aegis.json").read_text(encoding="utf-8")
    )
    titles = [p["title"] for p in dashboard["panels"]]
    for expected in [
        "Predictions by class",
        "Confidence distribution",
        "OOD rate (with alert threshold)",
    ]:
        assert expected in titles


@pytest.mark.integration
def test_dashboard_latency_panel_split_by_model_label() -> None:
    dashboard = json.loads(
        (PROVISIONING_DIR / "dashboards" / "aegis.json").read_text(encoding="utf-8")
    )
    panel = next(
        p for p in dashboard["panels"] if p["title"] == "p95 Latency by model (SLO: 500ms)"
    )
    assert "by (le, model)" in panel["targets"][0]["expr"]


@pytest.mark.integration
def test_dashboard_has_explicit_ood_status_panel() -> None:
    dashboard = json.loads(
        (PROVISIONING_DIR / "dashboards" / "aegis.json").read_text(encoding="utf-8")
    )
    panel = next(p for p in dashboard["panels"] if p["title"] == "OOD detection status")
    mappings = panel["fieldConfig"]["defaults"]["mappings"][0]["options"]
    assert mappings["0"]["text"] == "DISABLED"
    assert mappings["1"]["text"] == "ENABLED"
