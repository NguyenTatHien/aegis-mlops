from __future__ import annotations

import json
from pathlib import Path

import pytest

FAKE_OOD_CONFIG = {
    "msp_threshold": 0.0016560554504394531,
    "energy_threshold": -5.673770427703857,
    "energy_temperature": 1.0,
    "label_names": ["World", "Sports", "Business", "Sci/Tech"],
    "max_len": 128,
}


@pytest.fixture
def ood_config_file(tmp_path: Path) -> Path:
    path = tmp_path / "ood_config.json"
    path.write_text(json.dumps(FAKE_OOD_CONFIG), encoding="utf-8")
    return path


@pytest.fixture
def recalibrated_ood_config_file(tmp_path: Path) -> Path:
    data = {
        **FAKE_OOD_CONFIG,
        "target_fpr": 0.05,
        "measured_fpr": {"msp": 0.048, "energy": 0.041},
        "measured_recall": {"msp": 0.44, "energy": 0.47},
        "calibrated_at": "2026-08-13T00:00:00Z",
    }
    path = tmp_path / "ood_config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path
