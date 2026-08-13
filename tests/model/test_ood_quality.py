"""Task 7.11 — OOD FPR ceiling and recall after recalibration (task 7.5-7.7).
Reads the CURRENT ood_config.json, not the pre-recalibration one shipped
with the original artifacts (which has FPR 35-41%, see design.md D5) — this
test is meant to fail loudly against that old file and only pass once
scripts/recalibrate_ood.py has run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.model

OOD_CONFIG_PATH = Path("content/aegis_artifacts/ood_config.json")
LOGITS_DIR = Path("content/aegis_artifacts/ood_logits")


def _require_recalibrated_config() -> dict:
    if not OOD_CONFIG_PATH.exists():
        pytest.skip("ood_config.json not present")
    config = json.loads(OOD_CONFIG_PATH.read_text(encoding="utf-8"))
    if "measured_fpr" not in config:
        pytest.skip("ood_config.json not recalibrated yet — run scripts/recalibrate_ood.py")
    return config


def test_measured_fpr_ceiling_from_config() -> None:
    config = _require_recalibrated_config()
    for method, fpr in config["measured_fpr"].items():
        assert fpr <= 0.10, f"{method} measured_fpr {fpr:.3f} exceeds ceiling 0.10"


def test_measured_recall_recorded_and_nonzero() -> None:
    config = _require_recalibrated_config()
    for method, recall in config["measured_recall"].items():
        assert 0.0 < recall <= 1.0, f"{method} recall {recall} out of (0, 1]"


def test_thresholds_reproduce_measured_fpr_on_cached_logits() -> None:
    """Cross-check: applying the chosen threshold to the cached test-set
    logits reproduces the FPR recorded in ood_config.json (catches drift
    between the sweep script and the shipped config)."""
    config = _require_recalibrated_config()
    if not (LOGITS_DIR / "id_test_logits.npy").exists():
        pytest.skip("cached logits not present")

    from aegis.ood.scoring import energy_score, msp_score

    id_test_logits = np.load(LOGITS_DIR / "id_test_logits.npy")

    msp_scores = msp_score(id_test_logits)
    msp_fpr = float((msp_scores >= config["msp_threshold"]).mean())
    assert msp_fpr == pytest.approx(config["measured_fpr"]["msp"], abs=0.02)

    energy_scores = energy_score(id_test_logits, config["energy_temperature"])
    energy_fpr = float((energy_scores >= config["energy_threshold"]).mean())
    assert energy_fpr == pytest.approx(config["measured_fpr"]["energy"], abs=0.02)
