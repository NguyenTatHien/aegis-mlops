"""Task 5.12 — full-stack smoke test through docker compose.

Marked `slow`: it builds/starts real containers (~1-2GB image download on
first run) and is excluded from the default `pytest -m "unit or integration"`
CI job (design.md D11). Run explicitly with `pytest -m slow` or `make
test-compose-smoke` once Docker is available.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILE = REPO_ROOT / "docker-compose.yml"


def _compose(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FILE), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )


def _wait_healthy(service: str, timeout_s: float = 180.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        result = _compose("ps", "--format", "json", service)
        if '"Health":"healthy"' in result.stdout or '"health":"healthy"' in result.stdout.lower():
            return True
        time.sleep(3)
    return False


def _get_with_retry(url: str, *, attempts: int = 5, **kwargs) -> httpx.Response:
    """A container reporting `healthy` doesn't guarantee the host-side port
    forward is immediately stable — observed on Docker Desktop/Windows as a
    transient ReadError/ConnectionReset on the first request or two right
    after healthcheck passes. Retry briefly instead of failing the whole
    smoke test on that race."""
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return httpx.get(url, **kwargs)
        except httpx.TransportError as exc:
            last_exc = exc
            time.sleep(1.5 * (attempt + 1))
    assert last_exc is not None
    raise last_exc


def _post_with_retry(url: str, *, attempts: int = 5, **kwargs) -> httpx.Response:
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return httpx.post(url, **kwargs)
        except httpx.TransportError as exc:
            last_exc = exc
            time.sleep(1.5 * (attempt + 1))
    assert last_exc is not None
    raise last_exc


@pytest.mark.slow
def test_full_stack_smoke() -> None:
    up = _compose("up", "-d", "--build")
    try:
        assert up.returncode == 0, up.stderr

        for service in ["mlflow", "prometheus", "grafana", "api"]:
            healthy = _wait_healthy(service)
            if not healthy:
                logs = _compose("logs", service)
                pytest.fail(
                    f"{service} never became healthy.\n--- logs ---\n{logs.stdout}\n{logs.stderr}"
                )

        base = "http://127.0.0.1:8000"
        assert _get_with_retry(f"{base}/", timeout=5).status_code == 200
        assert _get_with_retry(f"{base}/health", timeout=5).status_code == 200
        assert _get_with_retry(f"{base}/ready", timeout=5).status_code == 200

        predict_resp = _post_with_retry(
            f"{base}/v1/predict", json={"text": "The team won the championship."}, timeout=10
        )
        assert predict_resp.status_code == 200
        assert predict_resp.json()["predicted_class"] in {"World", "Sports", "Business", "Sci/Tech"}

        metrics_resp = _get_with_retry(f"{base}/metrics", timeout=5)
        assert metrics_resp.status_code == 200
        assert "prediction_requests_total" in metrics_resp.text
    finally:
        all_logs = _compose("logs")
        (REPO_ROOT / "tests" / "fixtures" / "last_compose_smoke.log").write_text(
            all_logs.stdout + all_logs.stderr, encoding="utf-8"
        )
        _compose("down", "-v")
