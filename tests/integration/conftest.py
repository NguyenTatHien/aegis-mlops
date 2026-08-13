from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import aegis.config as config_module
from aegis.api.main import create_app
from aegis.serving.mock_predictor import mock_predictor_factory


def _clear_config_cache() -> None:
    config_module.get_settings.cache_clear()
    config_module.get_ood_config.cache_clear()


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Hermetic: forces OOD_ENABLED=false regardless of the ambient .env,
    so this fixture's behavior doesn't depend on what a developer happened
    to leave in their local .env (see design.md D5 — that value changes
    once the team recalibrates thresholds)."""
    monkeypatch.setenv("OOD_ENABLED", "false")
    _clear_config_cache()
    app = create_app(predictor_factory=mock_predictor_factory)
    with TestClient(app) as c:
        yield c
    _clear_config_cache()


@pytest.fixture
def client_ood_enabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[TestClient]:
    """Same MockPredictor, but with a real (temp-file) ood_config.json and
    OOD_ENABLED=true — exercises the 'OOD bật' scenarios in
    specs/ood-detection/spec.md without touching the real artifacts dir."""
    config_path = tmp_path / "ood_config.json"
    config_path.write_text(
        json.dumps(
            {
                "msp_threshold": 0.0973578714028861,
                "energy_threshold": -4.313697454571079,
                "energy_temperature": 1.0,
                "label_names": ["World", "Sports", "Business", "Sci/Tech"],
                "max_len": 128,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OOD_ENABLED", "true")
    monkeypatch.setenv("OOD_CONFIG_PATH", str(config_path))
    _clear_config_cache()
    app = create_app(predictor_factory=mock_predictor_factory)
    with TestClient(app) as c:
        yield c
    _clear_config_cache()


@pytest.fixture
def client_not_ready(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("OOD_ENABLED", "false")
    _clear_config_cache()

    def _empty_factory() -> dict:
        raise RuntimeError("simulated startup failure")

    app = create_app(predictor_factory=_empty_factory)
    with TestClient(app) as c:
        yield c
    _clear_config_cache()
