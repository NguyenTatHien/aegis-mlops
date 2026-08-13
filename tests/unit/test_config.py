"""Contract tests for src/aegis/config.py.

These pin down design.md D7 (label names come only from ood_config.json,
never from roberta_final/config.json's LABEL_0..3) and D8 (max_len=128 is
part of the serving contract, not a free config value).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aegis.config import get_max_len, load_ood_config


@pytest.mark.unit
def test_max_len_matches_training(ood_config_file: Path) -> None:
    cfg = load_ood_config(ood_config_file)
    assert cfg is not None
    assert cfg.max_len == 128


@pytest.mark.unit
def test_label_names_four_entries_in_ag_news_order(ood_config_file: Path) -> None:
    cfg = load_ood_config(ood_config_file)
    assert cfg is not None
    assert cfg.label_names == ["World", "Sports", "Business", "Sci/Tech"]


@pytest.mark.unit
def test_thresholds_not_hardcoded_reflect_file_contents(tmp_path: Path) -> None:
    custom = {
        "msp_threshold": 0.5,
        "energy_threshold": -1.0,
        "energy_temperature": 2.0,
        "label_names": ["World", "Sports", "Business", "Sci/Tech"],
        "max_len": 128,
    }
    path = tmp_path / "ood_config.json"
    path.write_text(json.dumps(custom), encoding="utf-8")

    cfg = load_ood_config(path)
    assert cfg is not None
    assert cfg.msp_threshold == 0.5
    assert cfg.energy_threshold == -1.0
    assert cfg.energy_temperature == 2.0


@pytest.mark.unit
def test_missing_config_file_returns_none_not_raise(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.json"
    assert load_ood_config(missing) is None


@pytest.mark.unit
def test_malformed_config_file_returns_none_not_raise(tmp_path: Path) -> None:
    path = tmp_path / "ood_config.json"
    path.write_text("{not valid json", encoding="utf-8")
    assert load_ood_config(path) is None


@pytest.mark.unit
def test_recalibrated_config_carries_measured_metrics(recalibrated_ood_config_file: Path) -> None:
    cfg = load_ood_config(recalibrated_ood_config_file)
    assert cfg is not None
    assert cfg.target_fpr == 0.05
    assert cfg.measured_fpr is not None
    assert cfg.measured_fpr["energy"] < 0.05


@pytest.mark.unit
def test_get_max_len_fallback_when_config_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    import aegis.config as config_module

    config_module.get_ood_config.cache_clear()
    monkeypatch.setattr(config_module, "get_ood_config", lambda: None)
    assert get_max_len() == 128
