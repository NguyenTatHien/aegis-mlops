"""AG News ingestion. Wraps datasets.load_dataset with an offline retry so a
second run works without network once the HF cache is populated.

Every load_dataset() call pins an explicit `revision` commit SHA — bandit's
B615 check requires the literal inline (not a module-level constant it can't
resolve via data-flow), which is also the more honest form: the SHA a reader
sees next to the call is the one actually being fetched.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("aegis.data.loader")

DATASET_NAME = "fancyzhx/ag_news"
OOD_SPAM_DATASET = "ucirvine/sms_spam"
OOD_HATE_DATASET = "cardiffnlp/tweet_eval"


def load_ag_news(cache_dir: str | Path | None = None) -> Any:
    from datasets import load_dataset

    kwargs: dict[str, Any] = {"cache_dir": str(cache_dir)} if cache_dir else {}
    try:
        return load_dataset(
            DATASET_NAME, revision="eb185aade064a813bc0b7f42de02595523103ca4", **kwargs
        )
    except (OSError, ConnectionError) as exc:
        logger.warning("network unavailable (%s) — retrying against local HF cache only", exc)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        return load_dataset(
            DATASET_NAME, revision="eb185aade064a813bc0b7f42de02595523103ca4", **kwargs
        )


def prepare_ag_news_splits(seed: int = 42, val_size: float = 0.10) -> dict[str, Any]:
    """Single source of truth for the train/val/test split — used by both
    train_baseline.py and scripts/collect_logits.py so OOD calibration runs
    against the exact same validation set the baseline was tuned on."""
    from aegis.data.split import make_split

    raw = load_ag_news()
    label_names = raw["train"].features["label"].names
    train_labels_full = np.array(raw["train"]["label"])
    train_idx, val_idx = make_split(train_labels_full, seed=seed, val_size=val_size)

    return {
        "train_texts": [raw["train"]["text"][i] for i in train_idx],
        "train_labels": train_labels_full[train_idx],
        "val_texts": [raw["train"]["text"][i] for i in val_idx],
        "val_labels": train_labels_full[val_idx],
        "test_texts": list(raw["test"]["text"]),
        "test_labels": np.array(raw["test"]["label"]),
        "label_names": label_names,
    }


def load_ood_proxy_texts(seed: int = 42) -> tuple[list[str], list[str]]:
    """sms_spam (spam/ad proxy) + tweet_eval/hate (toxic proxy), matching
    notebooks/aegis_ag_news_training.ipynb cell 47. Split 50/50 into OOD
    val/test so thresholds are calibrated on one half and evaluated on the
    other, same discipline as the ID split."""
    from datasets import load_dataset
    from sklearn.model_selection import train_test_split

    sms = load_dataset(OOD_SPAM_DATASET, revision="cae486f927c250fe1d4a5b55f11357964ed1646c")[
        "train"
    ]
    spam_texts = [t for t, label in zip(sms["sms"], sms["label"], strict=True) if label == 1]

    tweet_hate = load_dataset(
        OOD_HATE_DATASET, "hate", revision="b3a375baf0f409c77e6bc7aa35102b7b3534f8be"
    )
    hate_texts = [
        t
        for t, label in zip(tweet_hate["train"]["text"], tweet_hate["train"]["label"], strict=True)
        if label == 1
    ]

    all_ood_texts = spam_texts + hate_texts
    ood_val_texts, ood_test_texts = train_test_split(
        all_ood_texts, test_size=0.5, random_state=seed
    )
    return ood_val_texts, ood_test_texts


if __name__ == "__main__":
    raw = load_ag_news()
    print(raw)
    print("Classes:", raw["train"].features["label"].names)
