"""OODDetector interface (design.md D4).

A detector is a pure function of logits: it does not tokenize, does not load
a model, does not read files at scoring time. That constraint is what lets
tests/unit/test_ood_scoring.py and test_ood_interface.py run against fake
numpy arrays without any of torch/transformers installed.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class OODDetector(Protocol):
    method: str
    enabled: bool

    def score(self, logits: np.ndarray) -> float:
        """Anomaly score for a single sample's logits, shape (1, n_classes)
        or (n_classes,). Higher score = more likely out-of-domain."""
        ...

    def is_ood(self, score: float) -> bool:
        """Whether the given score crosses this detector's threshold."""
        ...
