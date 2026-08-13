"""Prometheus instrumentation.

Bundled into a Metrics object created fresh per app instance (create_metrics(),
stored on app.state) rather than module-level globals — module-level Counter
objects would share a single CollectorRegistry across every TestClient built
in a pytest session, making "increments by exactly N" assertions leak between
tests. Each create_app() call gets its own isolated registry instead.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

_KNOWN_MODELS = ("baseline", "roberta")
_KNOWN_OOD_METHODS = ("msp", "energy", "entropy")


class _RollingOODRate:
    """In-process approximation of a rolling OOD rate, exposed as a Gauge for
    Grafana convenience. Alerting itself uses PromQL rate() over the
    counters (monitoring/prometheus/alerts.yml), which survives restarts;
    this gauge does not and is not meant to."""

    def __init__(self, window_seconds: float = 300.0) -> None:
        self._window_seconds = window_seconds
        self._events: dict[str, deque[tuple[float, bool]]] = {}

    def record(self, model: str, is_ood: bool) -> float:
        now = time.time()
        dq = self._events.setdefault(model, deque())
        dq.append((now, is_ood))
        cutoff = now - self._window_seconds
        while dq and dq[0][0] < cutoff:
            dq.popleft()
        if not dq:
            return 0.0
        return sum(1 for _, flag in dq if flag) / len(dq)


@dataclass
class Metrics:
    registry: CollectorRegistry
    prediction_requests_total: Counter
    prediction_latency_seconds: Histogram
    http_request_errors_total: Counter
    inference_duration_seconds: Histogram
    predictions_by_class_total: Counter
    prediction_confidence: Histogram
    ood_detected_total: Counter
    ood_score: Histogram
    ood_rate: Gauge
    input_text_length_words: Histogram
    model_info: Gauge
    ood_enabled: Gauge
    _rolling: _RollingOODRate = field(default_factory=_RollingOODRate)

    def record_prediction(
        self,
        *,
        endpoint: str,
        model: str,
        status_code: int,
        predicted_class: str,
        confidence: float,
        latency_seconds: float,
        inference_seconds: float,
        text_word_count: int,
        ood_method: str | None,
        ood_is_ood: bool | None,
        ood_score_value: float | None,
    ) -> None:
        self.prediction_requests_total.labels(
            endpoint=endpoint, status_code=str(status_code), model=model
        ).inc()
        self.prediction_latency_seconds.labels(model=model).observe(latency_seconds)
        self.inference_duration_seconds.labels(model=model).observe(inference_seconds)
        self.input_text_length_words.labels(model=model).observe(text_word_count)
        self.predictions_by_class_total.labels(predicted_class=predicted_class, model=model).inc()
        self.prediction_confidence.labels(model=model).observe(confidence)

        if ood_method is not None:
            if ood_is_ood:
                self.ood_detected_total.labels(method=ood_method, model=model).inc()
            if ood_score_value is not None:
                self.ood_score.labels(method=ood_method, model=model).observe(ood_score_value)
            self.ood_rate.labels(model=model).set(self._rolling.record(model, bool(ood_is_ood)))

    def record_http_error(self, *, endpoint: str, error_type: str) -> None:
        self.http_request_errors_total.labels(endpoint=endpoint, error_type=error_type).inc()

    def set_model_info(self, *, model: str, version: str, macro_f1: float) -> None:
        self.model_info.labels(model=model, version=version, macro_f1=f"{macro_f1:.4f}").set(1)

    def set_ood_enabled(self, *, model: str, enabled: bool) -> None:
        """Lets Grafana render an explicit ENABLED/DISABLED panel instead of
        inferring OOD status from absence of ood_detected_total activity —
        the spec requires the dashboard state this outright, not imply it."""
        self.ood_enabled.labels(model=model).set(1.0 if enabled else 0.0)

    def exposition(self) -> tuple[bytes, str]:
        return generate_latest(self.registry), CONTENT_TYPE_LATEST


def create_metrics() -> Metrics:
    registry = CollectorRegistry()

    metrics = Metrics(
        registry=registry,
        prediction_requests_total=Counter(
            "prediction_requests_total",
            "Total prediction requests",
            ["endpoint", "status_code", "model"],
            registry=registry,
        ),
        prediction_latency_seconds=Histogram(
            "prediction_latency_seconds",
            "End-to-end prediction request latency (SLO: p95 < 0.5s)",
            ["model"],
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
            registry=registry,
        ),
        http_request_errors_total=Counter(
            "http_request_errors_total",
            "Total HTTP errors",
            ["endpoint", "error_type"],
            registry=registry,
        ),
        inference_duration_seconds=Histogram(
            "inference_duration_seconds",
            "Time spent inside predictor.predict() only, excludes OOD scoring and serialization",
            ["model"],
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
            registry=registry,
        ),
        predictions_by_class_total=Counter(
            "predictions_by_class_total",
            "Predictions per class",
            ["predicted_class", "model"],
            registry=registry,
        ),
        prediction_confidence=Histogram(
            "prediction_confidence",
            "Predicted-class confidence distribution",
            ["model"],
            buckets=(0.25, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
            registry=registry,
        ),
        ood_detected_total=Counter(
            "ood_detected_total",
            "Requests flagged as out-of-domain",
            ["method", "model"],
            registry=registry,
        ),
        ood_score=Histogram(
            "ood_score",
            "OOD anomaly score distribution",
            ["method", "model"],
            registry=registry,
        ),
        ood_rate=Gauge(
            "ood_rate",
            "Rolling ~5-minute OOD rate (process-local approximation; alerting uses PromQL rate() instead)",
            ["model"],
            registry=registry,
        ),
        input_text_length_words=Histogram(
            "input_text_length_words",
            "Word count of input text",
            ["model"],
            buckets=(5, 10, 20, 40, 80, 160, 320),
            registry=registry,
        ),
        model_info=Gauge(
            "model_info",
            "Static model metadata; value is always 1, info carried in labels",
            ["model", "version", "macro_f1"],
            registry=registry,
        ),
        ood_enabled=Gauge(
            "ood_enabled",
            "1 if OOD detection is enabled for this model, else 0",
            ["model"],
            registry=registry,
        ),
    )

    # Pre-touch every known OOD label combination so ood_detected_total /
    # ood_score / ood_rate appear at 0 in /metrics immediately at startup,
    # even while OOD_ENABLED=false (spec: observability "Metrics OOD tồn
    # tại kể cả khi tắt"). prometheus_client only exports a labeled child
    # once .labels(...) has been called on it at least once.
    for model in _KNOWN_MODELS:
        for method in _KNOWN_OOD_METHODS:
            metrics.ood_detected_total.labels(method=method, model=model)
            metrics.ood_score.labels(method=method, model=model)
        metrics.ood_rate.labels(model=model).set(0.0)
        metrics.ood_enabled.labels(model=model).set(0.0)

    return metrics
