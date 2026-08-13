from __future__ import annotations

import numpy as np
import pytest

from aegis.ood.scoring import energy_score, entropy_score, msp_score


@pytest.mark.unit
def test_msp_score_uniform_logits_is_075() -> None:
    logits = np.array([[0.0, 0.0, 0.0, 0.0]])
    assert msp_score(logits)[0] == pytest.approx(0.75, abs=1e-6)


@pytest.mark.unit
def test_msp_score_confident_logits_near_zero() -> None:
    logits = np.array([[10.0, 0.0, 0.0, 0.0]])
    assert msp_score(logits)[0] < 0.01


@pytest.mark.unit
def test_energy_score_matches_negative_logsumexp_at_t1() -> None:
    logits = np.array([[1.0, 2.0, 3.0, 4.0]])
    expected = -np.log(np.exp([1.0, 2.0, 3.0, 4.0]).sum())
    assert energy_score(logits, temperature=1.0)[0] == pytest.approx(expected, abs=1e-6)


@pytest.mark.unit
def test_energy_score_monotonic_with_confidence() -> None:
    confident = energy_score(np.array([[10.0, 0.0, 0.0, 0.0]]))[0]
    unsure = energy_score(np.array([[1.0, 0.0, 0.0, 0.0]]))[0]
    assert confident < unsure


@pytest.mark.unit
def test_entropy_score_certain_distribution_is_zero() -> None:
    probs = np.array([[1.0, 0.0, 0.0, 0.0]])
    assert entropy_score(probs)[0] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.unit
def test_entropy_score_uniform_distribution_is_one() -> None:
    probs = np.array([[0.25, 0.25, 0.25, 0.25]])
    assert entropy_score(probs)[0] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.unit
def test_scoring_accepts_1d_input() -> None:
    logits_1d = np.array([2.1, -0.3, 0.5, -1.2])
    logits_2d = logits_1d.reshape(1, -1)
    assert msp_score(logits_1d)[0] == pytest.approx(msp_score(logits_2d)[0], abs=1e-9)
