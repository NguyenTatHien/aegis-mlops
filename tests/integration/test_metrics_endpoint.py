from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families


def _metric_value(text: str, name: str, labels: dict[str, str]) -> float | None:
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if sample.name == name and all(sample.labels.get(k) == v for k, v in labels.items()):
                return sample.value
    return None


@pytest.mark.integration
def test_metrics_endpoint_returns_prometheus_format(client: TestClient) -> None:
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]


@pytest.mark.integration
def test_prediction_requests_total_increments_by_three(client: TestClient) -> None:
    for _ in range(3):
        client.post("/v1/predict", json={"text": "Stocks rallied on Tuesday."})
    text = client.get("/metrics").text
    value = _metric_value(
        text,
        "prediction_requests_total",
        {"endpoint": "/v1/predict", "model": "roberta", "status_code": "200"},
    )
    assert value == 3.0


@pytest.mark.integration
def test_predictions_by_class_total_tracks_predicted_class(client: TestClient) -> None:
    client.post("/v1/predict", json={"text": "Team wins the championship."})
    text = client.get("/metrics").text
    # MockPredictor for roberta always predicts Sports (see mock_predictor.py fixed logits)
    value = _metric_value(
        text, "predictions_by_class_total", {"predicted_class": "Sports", "model": "roberta"}
    )
    assert value is not None and value >= 1.0


@pytest.mark.integration
def test_ood_metrics_registered_at_zero_when_disabled(client: TestClient) -> None:
    text = client.get("/metrics").text
    value = _metric_value(text, "ood_detected_total", {"method": "energy", "model": "roberta"})
    assert value == 0.0


@pytest.mark.integration
def test_svm_metrics_are_registered_at_startup(client: TestClient) -> None:
    text = client.get("/metrics").text
    value = _metric_value(text, "ood_enabled", {"model": "svm"})
    assert value == 0.0


@pytest.mark.integration
def test_svm_margin_is_not_recorded_as_probability_confidence(client: TestClient) -> None:
    client.post("/v1/predict?model=svm", json={"text": "Team wins the championship."})
    text = client.get("/metrics").text
    margin_count = _metric_value(text, "prediction_relative_margin_count", {"model": "svm"})
    confidence_count = _metric_value(text, "prediction_confidence_count", {"model": "svm"})
    assert margin_count == 1.0
    assert confidence_count is None


@pytest.mark.integration
def test_latency_histogram_has_500ms_bucket() -> None:
    from aegis.api.metrics import create_metrics

    metrics = create_metrics()
    assert 0.5 in metrics.prediction_latency_seconds._upper_bounds  # type: ignore[attr-defined]


@pytest.mark.integration
def test_http_request_errors_total_increments_on_validation_error(client: TestClient) -> None:
    client.post("/v1/predict", json={"text": ""})
    text = client.get("/metrics").text
    value = _metric_value(
        text,
        "http_request_errors_total",
        {"endpoint": "/v1/predict", "error_type": "validation_error"},
    )
    assert value == 1.0


@pytest.mark.integration
def test_model_info_metric_present_after_startup(client: TestClient) -> None:
    text = client.get("/metrics").text
    assert "model_info{" in text
    assert 'model="roberta"' in text
