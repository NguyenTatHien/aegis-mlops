"""90/10 stratified train/validation split. AG News ships only train/test,
so validation is carved out of train (notebooks/aegis_ag_news_training.ipynb
cell 15) — indices are written to disk so the exact split is reproducible,
not just "reproducible in expectation" via a fixed seed."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def make_split(
    labels: np.ndarray, seed: int = 42, val_size: float = 0.10
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        idx, test_size=val_size, stratify=labels, random_state=seed
    )
    return train_idx, val_idx


def save_split_indices(train_idx: np.ndarray, val_idx: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, train_idx=train_idx, val_idx=val_idx)


def load_split_indices(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["train_idx"], data["val_idx"]
