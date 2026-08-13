"""Central runtime configuration.

Two distinct sources, kept deliberately separate:
- ``Settings``      — operational config from environment variables (.env).
- ``OODConfig``     — model-serving contract (label names, max_len, OOD
  thresholds) loaded from ``ood_config.json``. This file is the single
  source of truth for label names and max_len — never read them from
  ``roberta_final/config.json`` (its id2label is LABEL_0..LABEL_3, and its
  model_max_length of 512 does not match what the model was trained and
  OOD-calibrated at, which is 128).
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger("aegis.config")

# Fallback only used if ood_config.json is entirely missing, so the API can
# still serve classification (without OOD) instead of failing to start.
_FALLBACK_LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
_FALLBACK_MAX_LEN = 128


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model_source: str = Field(default="local", alias="MODEL_SOURCE")  # local | registry
    model_default: str = Field(default="roberta", alias="MODEL_DEFAULT")  # baseline | svm | roberta
    ood_enabled: bool = Field(default=False, alias="OOD_ENABLED")

    artifacts_dir: Path = Field(default=Path("artifacts"), alias="ARTIFACTS_DIR")
    roberta_model_dir: Path = Field(
        default=Path("artifacts/roberta_final"), alias="ROBERTA_MODEL_DIR"
    )
    baseline_dir: Path = Field(default=Path("artifacts/baseline"), alias="BASELINE_DIR")
    ood_config_path: Path = Field(
        default=Path("artifacts/ood_config.json"), alias="OOD_CONFIG_PATH"
    )

    mlflow_tracking_uri: str = Field(default="http://mlflow:5000", alias="MLFLOW_TRACKING_URI")
    mlflow_baseline_model_uri: str = Field(
        default="models:/aegis-baseline/Production", alias="MLFLOW_BASELINE_MODEL_URI"
    )
    mlflow_roberta_model_uri: str = Field(
        default="models:/aegis-roberta/Production", alias="MLFLOW_ROBERTA_MODEL_URI"
    )

    api_max_text_length: int = Field(default=10_000, alias="API_MAX_TEXT_LENGTH")
    api_max_batch_size: int = Field(default=32, alias="API_MAX_BATCH_SIZE")
    api_title: str = Field(default="Aegis News Routing API", alias="API_TITLE")
    api_version: str = Field(default="v1", alias="API_VERSION")


class OODConfig(BaseModel):
    """Mirrors content/aegis_artifacts/ood_config.json, plus fields added by
    scripts/recalibrate_ood.py (task 7.7) — optional so the pre-recalibration
    file (msp_threshold/energy_threshold/energy_temperature/label_names/max_len
    only) still loads."""

    msp_threshold: float
    energy_threshold: float
    energy_temperature: float = 1.0
    entropy_threshold: float | None = None  # baseline-only; added once entropy OOD is calibrated
    label_names: list[str]
    max_len: int

    target_fpr: float | None = None
    measured_fpr: dict[str, float] | None = None
    measured_recall: dict[str, float] | None = None
    calibrated_at: str | None = None


def load_ood_config(path: Path) -> OODConfig | None:
    """Load ood_config.json. Returns None (never raises) if the file is
    missing or malformed — callers fall back to NullOODDetector and a
    hardcoded label list so the classification path keeps working."""
    if not path.exists():
        logger.warning("ood_config.json not found at %s — OOD detection disabled", path)
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return OODConfig.model_validate(data)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("ood_config.json at %s is invalid (%s) — OOD detection disabled", path, exc)
        return None


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_ood_config() -> OODConfig | None:
    return load_ood_config(get_settings().ood_config_path)


def get_label_names() -> list[str]:
    cfg = get_ood_config()
    return cfg.label_names if cfg is not None else list(_FALLBACK_LABEL_NAMES)


def get_max_len() -> int:
    cfg = get_ood_config()
    return cfg.max_len if cfg is not None else _FALLBACK_MAX_LEN
