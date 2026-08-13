"""Parametrized across every OODDetector implementation, including Null.

The point: adding a new detector that doesn't satisfy the shared contract
(method/enabled attrs, score()->float, is_ood()->bool) fails here first,
before it ever reaches the API layer.
"""

from __future__ import annotations

import numpy as np
import pytest

from aegis.ood.base import OODDetector
from aegis.ood.detector import EnergyDetector, EntropyDetector, MSPDetector, NullOODDetector

FAKE_LOGITS = np.array([[2.1, -0.3, 0.5, -1.2]])
FAKE_PROBS = np.array([[0.4, 0.3, 0.2, 0.1]])

DETECTORS = [
    MSPDetector(threshold=0.5),
    EnergyDetector(threshold=-3.0),
    EntropyDetector(threshold=0.5),
    NullOODDetector(),
]


@pytest.mark.unit
@pytest.mark.parametrize("detector", DETECTORS, ids=lambda d: d.method)
def test_satisfies_protocol(detector: OODDetector) -> None:
    assert isinstance(detector, OODDetector)


@pytest.mark.unit
@pytest.mark.parametrize("detector", DETECTORS, ids=lambda d: d.method)
def test_score_returns_float_without_side_effects(detector: OODDetector) -> None:
    inputs = FAKE_PROBS if detector.method == "entropy" else FAKE_LOGITS
    score = detector.score(inputs)
    assert isinstance(score, float)


@pytest.mark.unit
@pytest.mark.parametrize("detector", DETECTORS, ids=lambda d: d.method)
def test_is_ood_returns_bool(detector: OODDetector) -> None:
    inputs = FAKE_PROBS if detector.method == "entropy" else FAKE_LOGITS
    score = detector.score(inputs)
    assert isinstance(detector.is_ood(score), bool)


@pytest.mark.unit
def test_null_detector_never_flags_ood() -> None:
    detector = NullOODDetector()
    assert detector.enabled is False
    assert detector.is_ood(detector.score(FAKE_LOGITS)) is False
    assert detector.is_ood(999.0) is False


@pytest.mark.unit
@pytest.mark.parametrize(
    "detector",
    [MSPDetector(threshold=0.5), EnergyDetector(threshold=-3.0), EntropyDetector(threshold=0.5)],
)
def test_real_detectors_are_enabled(detector: OODDetector) -> None:
    assert detector.enabled is True
