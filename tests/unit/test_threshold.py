from __future__ import annotations

from pathlib import Path

import pytest

from aegis.config import load_ood_config
from aegis.ood.detector import EnergyDetector, MSPDetector, NullOODDetector, build_detector


@pytest.mark.unit
def test_is_ood_true_above_threshold() -> None:
    detector = MSPDetector(threshold=0.5)
    assert detector.is_ood(0.6) is True


@pytest.mark.unit
def test_is_ood_false_below_threshold() -> None:
    detector = MSPDetector(threshold=0.5)
    assert detector.is_ood(0.4) is False


@pytest.mark.unit
def test_is_ood_boundary_at_exact_threshold_counts_as_ood() -> None:
    detector = EnergyDetector(threshold=-3.0)
    assert detector.is_ood(-3.0) is True


@pytest.mark.unit
def test_build_detector_msp_reads_threshold_from_config(ood_config_file: Path) -> None:
    config = load_ood_config(ood_config_file)
    detector = build_detector("msp", config, ood_enabled=True)
    assert isinstance(detector, MSPDetector)
    assert detector.threshold == pytest.approx(0.0016560554504394531)


@pytest.mark.unit
def test_build_detector_energy_reads_threshold_and_temperature(ood_config_file: Path) -> None:
    config = load_ood_config(ood_config_file)
    detector = build_detector("energy", config, ood_enabled=True)
    assert isinstance(detector, EnergyDetector)
    assert detector.threshold == pytest.approx(-5.673770427703857)
    assert detector.temperature == pytest.approx(1.0)


@pytest.mark.unit
def test_build_detector_falls_back_to_null_when_flag_off(ood_config_file: Path) -> None:
    config = load_ood_config(ood_config_file)
    detector = build_detector("energy", config, ood_enabled=False)
    assert isinstance(detector, NullOODDetector)


@pytest.mark.unit
def test_build_detector_falls_back_to_null_when_config_missing(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.json"
    config = load_ood_config(missing)
    detector = build_detector("energy", config, ood_enabled=True)
    assert isinstance(detector, NullOODDetector)


@pytest.mark.unit
def test_build_detector_entropy_without_calibrated_threshold_is_null(ood_config_file: Path) -> None:
    config = load_ood_config(ood_config_file)
    detector = build_detector("entropy", config, ood_enabled=True)
    assert isinstance(detector, NullOODDetector)


@pytest.mark.unit
def test_build_detector_unknown_method_raises(ood_config_file: Path) -> None:
    config = load_ood_config(ood_config_file)
    with pytest.raises(ValueError):
        build_detector("mahalanobis", config, ood_enabled=True)
