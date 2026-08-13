"""Dataset versioning: hash + row counts + class distribution, so a silent
dataset swap (e.g. a HF Hub update) is caught instead of quietly changing
what "the model" was evaluated against."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def compute_dataset_hash(texts: list[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_dataset_card(
    train_texts: list[str],
    train_labels: list[int],
    test_texts: list[str],
    test_labels: list[int],
    label_names: list[str],
) -> dict[str, Any]:
    train_counts = Counter(train_labels)
    return {
        "sha256": compute_dataset_hash(train_texts + test_texts),
        "n_train": len(train_texts),
        "n_test": len(test_texts),
        "class_counts": {label_names[i]: train_counts.get(i, 0) for i in range(len(label_names))},
    }


def write_dataset_card(card: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(card, indent=2), encoding="utf-8")


def check_dataset_card(card: dict[str, Any], path: Path) -> bool | None:
    """Returns True if hash matches the recorded card, False if it doesn't,
    None if there is no prior card to compare against."""
    if not path.exists():
        return None
    previous = json.loads(path.read_text(encoding="utf-8"))
    return previous.get("sha256") == card.get("sha256")
