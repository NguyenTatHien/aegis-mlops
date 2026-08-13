"""OODDetector implementations (design.md D4).

Every implementation here is a pure function of logits: no tokenization, no
model loading, no file reads at scoring time. Thresholds and temperature are
injected at construction from aegis.config.OODConfig, via build_detector().
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aegis.config import OODConfig
from aegis.ood.scoring import energy_score, entropy_score, msp_score


@dataclass
class MSPDetector:
    threshold: float
    method: str = "msp"
    enabled: bool = True

    def score(self, logits: np.ndarray) -> float:
        return float(msp_score(logits)[0])

    def is_ood(self, score: float) -> bool:
        return score >= self.threshold


@dataclass
class EnergyDetector:
    threshold: float
    temperature: float = 1.0
    method: str = "energy"
    enabled: bool = True

    def score(self, logits: np.ndarray) -> float:
        return float(energy_score(logits, self.temperature)[0])

    def is_ood(self, score: float) -> bool:
        return score >= self.threshold


@dataclass
class EntropyDetector:
    """For the baseline branch, which has predict_proba but no logits."""

    threshold: float
    method: str = "entropy"
    enabled: bool = True

    def score(self, probs: np.ndarray) -> float:
        return float(entropy_score(probs)[0])

    def is_ood(self, score: float) -> bool:
        return score >= self.threshold


@dataclass
class NullOODDetector:
    """Used when OOD_ENABLED=false or ood_config.json is missing/invalid.
    Never flags anything OOD — callers must check .enabled to decide whether
    to surface a result at all (PredictResponse.ood stays None)."""

    method: str = "none"
    enabled: bool = False

    def score(self, logits: np.ndarray) -> float:
        return 0.0

    def is_ood(self, score: float) -> bool:
        return False


def build_detector(
    method: str, config: OODConfig | None, *, ood_enabled: bool
) -> MSPDetector | EnergyDetector | EntropyDetector | NullOODDetector:
    """Factory used by api/dependencies.py. Falls back to NullOODDetector
    whenever the feature flag is off or the required threshold is absent —
    this is what lets OOD_ENABLED flip without touching the API schema."""
    if not ood_enabled or config is None:
        return NullOODDetector()

    if method == "msp":
        return MSPDetector(threshold=config.msp_threshold)
    if method == "energy":
        return EnergyDetector(
            threshold=config.energy_threshold, temperature=config.energy_temperature
        )
    if method == "entropy":
        if config.entropy_threshold is None:
            return NullOODDetector()
        return EntropyDetector(threshold=config.entropy_threshold)

    raise ValueError(f"unknown OOD method: {method!r}")
