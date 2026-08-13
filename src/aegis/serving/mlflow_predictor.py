"""MODEL_SOURCE=registry predictor loading (task 8.7).

Baseline is loaded as a real MLflow Model Registry pipeline (vectorizer +
classifier bundled — see train_baseline.py's log_to_mlflow). RoBERTa's
498MB checkpoint stays on the local artifact volume rather than being
stored as an MLflow artifact blob (design.md D9: MLflow tracks lineage/
metadata for RoBERTa via scripts/backfill_mlflow.py; the weights themselves
are the same volume-mounted files MODEL_SOURCE=local uses) — this mirrors
common practice for large model binaries and keeps this class project's
MLflow server from having to serve a 498MB artifact download on every
container start.
"""

from __future__ import annotations

import logging

import numpy as np

from aegis.config import Settings, get_label_names
from aegis.data.preprocess import clean_text_tfidf
from aegis.models.labels import id_to_label
from aegis.serving.base import PredictionResult, Predictor

logger = logging.getLogger("aegis.serving.mlflow")


class MLflowBaselinePredictor:
    name = "baseline"

    def __init__(self, model_uri: str, tracking_uri: str) -> None:
        import mlflow
        import mlflow.sklearn

        mlflow.set_tracking_uri(tracking_uri)
        self._label_names = get_label_names()
        self.pipeline = mlflow.sklearn.load_model(model_uri)
        self.version = model_uri
        self.macro_f1 = self._load_macro_f1(model_uri)
        self._ready = True

    @staticmethod
    def _load_macro_f1(model_uri: str) -> float:
        import mlflow

        try:
            model_name = model_uri.split("/")[1] if model_uri.startswith("models:/") else None
            stage_or_version = model_uri.split("/")[2] if model_uri.startswith("models:/") else None
            client = mlflow.MlflowClient()
            versions = client.search_model_versions(f"name='{model_name}'")
            for v in versions:
                if v.current_stage == stage_or_version or v.version == stage_or_version:
                    run = client.get_run(v.run_id)
                    return float(run.data.metrics.get("test_macro_f1", 0.0))
        except Exception as exc:  # pragma: no cover — best-effort metadata lookup
            logger.warning("could not resolve macro_f1 from MLflow run metadata: %s", exc)
        return 0.0

    def is_ready(self) -> bool:
        return self._ready

    def predict(self, text: str) -> PredictionResult:
        cleaned = clean_text_tfidf(text)
        probs = np.asarray(self.pipeline.predict_proba([cleaned])[0])
        idx = int(np.argmax(probs))
        return PredictionResult(
            predicted_class=id_to_label(idx, self._label_names),
            confidence=float(probs[idx]),
            logits=probs,
            model_version=self.version,
        )


def load_from_registry(settings: Settings) -> dict[str, Predictor]:
    from aegis.serving.roberta_predictor import RobertaPredictor
    from aegis.serving.svm_predictor import SVMPredictor

    return {
        "baseline": MLflowBaselinePredictor(
            settings.mlflow_baseline_model_uri, settings.mlflow_tracking_uri
        ),
        # The original LinearSVC artifact shares the local TF-IDF vectorizer;
        # like RoBERTa weights, it remains on the mounted artifact volume.
        "svm": SVMPredictor(settings.baseline_dir),
        "roberta": RobertaPredictor(settings.roberta_model_dir),
    }
