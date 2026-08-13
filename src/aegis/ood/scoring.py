"""Pure scoring functions (design.md D4).

Ported from notebooks/aegis_ag_news_training.ipynb cells 50 and 56.
All functions accept shape (n, k) or (k,) and return shape (n,) or scalar-like
(n,) with n=1 — callers (ood/detector.py) squeeze to a Python float.
"""

from __future__ import annotations

import numpy as np


def msp_score(logits: np.ndarray) -> np.ndarray:
    """1 - max(softmax(logits)). Higher = more anomalous."""
    logits = np.atleast_2d(logits).astype(np.float64)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=-1, keepdims=True)
    return 1.0 - probs.max(axis=-1)


def energy_score(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """-T * logsumexp(logits / T). Higher (less negative) = more anomalous."""
    scaled = np.atleast_2d(logits).astype(np.float64) / temperature
    m = scaled.max(axis=-1, keepdims=True)
    lse = m[..., 0] + np.log(np.exp(scaled - m).sum(axis=-1))
    return -temperature * lse


def entropy_score(probs: np.ndarray) -> np.ndarray:
    """Normalized Shannon entropy of a probability distribution, in [0, 1].
    0 = fully confident, 1 = uniform over classes. Takes probabilities
    (e.g. LogisticRegression.predict_proba), not logits."""
    probs = np.atleast_2d(probs).astype(np.float64)
    n_classes = probs.shape[-1]
    eps = 1e-12
    entropy = -(probs * np.log(probs + eps)).sum(axis=-1)
    return entropy / np.log(n_classes)
