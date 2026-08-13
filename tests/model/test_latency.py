"""Task 7.12 — latency budget on CPU, matching the committed success metric
(MLOps.docx System Level: latency < 500ms/request)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.model

ROBERTA_DIR = Path("content/aegis_artifacts/roberta_final")
BASELINE_DIR = Path("content/aegis_artifacts/baseline")
SAMPLE_TEXT = "The national football team secured a dramatic victory in extra time last night."
N_RUNS = 20


def test_roberta_p95_latency_under_500ms() -> None:
    if not (ROBERTA_DIR / "model.safetensors").exists():
        pytest.skip("roberta_final artifacts not present")
    from aegis.serving.roberta_predictor import RobertaPredictor

    predictor = RobertaPredictor(ROBERTA_DIR)
    predictor.predict(SAMPLE_TEXT)  # warm up

    latencies = []
    for _ in range(N_RUNS):
        start = time.perf_counter()
        predictor.predict(SAMPLE_TEXT)
        latencies.append(time.perf_counter() - start)

    p95 = float(np.percentile(latencies, 95))
    assert p95 < 0.5, f"p95 latency {p95 * 1000:.1f}ms exceeds 500ms budget"


def test_baseline_p95_latency_under_500ms() -> None:
    if not (BASELINE_DIR / "logreg_model.joblib").exists():
        pytest.skip("baseline not trained yet")
    from aegis.serving.baseline_predictor import BaselinePredictor

    predictor = BaselinePredictor(BASELINE_DIR)
    predictor.predict(SAMPLE_TEXT)

    latencies = []
    for _ in range(N_RUNS):
        start = time.perf_counter()
        predictor.predict(SAMPLE_TEXT)
        latencies.append(time.perf_counter() - start)

    p95 = float(np.percentile(latencies, 95))
    assert p95 < 0.5
