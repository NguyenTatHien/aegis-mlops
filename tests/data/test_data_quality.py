"""Data quality tests (spec: automated-testing "Data quality test").

Split into two groups: synthetic-fixture tests exercise the validation
logic itself (marked `unit` — pure functions, no network, no model — even
though the module they test lives under data/) and real-dataset tests
exercise the real AG News data pulled by tests/data/conftest.py's
ag_news_raw fixture (marked `data`, network-bound on first run, excluded
from the default PR CI job per design.md D11).
"""

from __future__ import annotations

import numpy as np
import pytest

from aegis.data.split import make_split
from aegis.data.validate import (
    validate_class_balance,
    validate_label_domain,
    validate_no_duplicates,
    validate_no_leakage,
    validate_no_nulls,
    validate_text_length,
)
from aegis.data.versioning import build_dataset_card, check_dataset_card, compute_dataset_hash

# --- synthetic-fixture: exercises validate.py's logic, no network needed ---


@pytest.mark.unit
def test_label_domain_flags_out_of_range() -> None:
    problems = validate_label_domain([0, 1, 2, 3, 4], n_classes=4)
    assert problems


@pytest.mark.unit
def test_label_domain_passes_valid_range() -> None:
    assert validate_label_domain([0, 1, 2, 3, 0, 1], n_classes=4) == []


@pytest.mark.unit
def test_class_balance_flags_skew() -> None:
    skewed = [0] * 90 + [1] * 5 + [2] * 3 + [3] * 2
    assert validate_class_balance(skewed, n_classes=4, tolerance=0.05)


@pytest.mark.unit
def test_class_balance_passes_even_split() -> None:
    even = [i % 4 for i in range(400)]
    assert validate_class_balance(even, n_classes=4, tolerance=0.01) == []


@pytest.mark.unit
def test_no_nulls_flags_blank_text() -> None:
    assert validate_no_nulls(["hello", "", "world", "   "])


@pytest.mark.unit
def test_no_nulls_passes_clean_texts() -> None:
    assert validate_no_nulls(["hello world", "another text"]) == []


@pytest.mark.unit
def test_text_length_flags_too_short_and_too_long() -> None:
    problems = validate_text_length(
        ["a", "a normal length sentence here", "x " * 400], min_words=3, max_words=200
    )
    assert len(problems) == 2


@pytest.mark.unit
def test_no_duplicates_flags_high_dup_ratio() -> None:
    texts = ["same text"] * 10 + ["unique one"]
    assert validate_no_duplicates(texts, max_dup_ratio=0.05)


@pytest.mark.unit
def test_no_leakage_flags_overlap() -> None:
    train = ["text a", "text b", "text c"]
    test = ["text b", "text d"]
    problems = validate_no_leakage(train, test)
    assert problems and "1" in problems[0]


@pytest.mark.unit
def test_no_leakage_passes_disjoint_sets() -> None:
    assert validate_no_leakage(["a", "b"], ["c", "d"]) == []


@pytest.mark.unit
def test_dataset_hash_deterministic() -> None:
    texts = ["one", "two", "three"]
    assert compute_dataset_hash(texts) == compute_dataset_hash(list(texts))


@pytest.mark.unit
def test_dataset_hash_changes_with_content() -> None:
    assert compute_dataset_hash(["a"]) != compute_dataset_hash(["b"])


@pytest.mark.unit
def test_check_dataset_card_no_prior_card_returns_none(tmp_path) -> None:
    card = build_dataset_card(["a"], [0], ["b"], [1], ["World", "Sports", "Business", "Sci/Tech"])
    assert check_dataset_card(card, tmp_path / "missing.json") is None


@pytest.mark.unit
def test_check_dataset_card_detects_mismatch(tmp_path) -> None:
    from aegis.data.versioning import write_dataset_card

    label_names = ["World", "Sports", "Business", "Sci/Tech"]
    card_v1 = build_dataset_card(["a", "b"], [0, 1], ["c"], [2], label_names)
    path = tmp_path / "dataset_card.json"
    write_dataset_card(card_v1, path)

    card_v2 = build_dataset_card(["a", "b", "extra"], [0, 1, 2], ["c"], [2], label_names)
    assert check_dataset_card(card_v2, path) is False


@pytest.mark.unit
def test_split_is_reproducible_with_same_seed() -> None:
    labels = np.array([i % 4 for i in range(400)])
    train_idx_1, val_idx_1 = make_split(labels, seed=42)
    train_idx_2, val_idx_2 = make_split(labels, seed=42)
    assert np.array_equal(train_idx_1, train_idx_2)
    assert np.array_equal(val_idx_1, val_idx_2)


@pytest.mark.unit
def test_split_keeps_class_ratio() -> None:
    labels = np.array([i % 4 for i in range(4000)])
    train_idx, val_idx = make_split(labels, seed=42, val_size=0.10)
    for split_labels in (labels[train_idx], labels[val_idx]):
        for cls in range(4):
            share = (split_labels == cls).mean()
            assert 0.24 <= share <= 0.26


# --- real dataset: exercises the actual AG News data, network-bound ---


@pytest.mark.data
def test_real_ag_news_label_domain(ag_news_raw) -> None:
    labels = ag_news_raw["train"]["label"]
    assert validate_label_domain(labels, n_classes=4) == []


@pytest.mark.data
def test_real_ag_news_class_balance(ag_news_raw) -> None:
    labels = ag_news_raw["train"]["label"]
    assert validate_class_balance(labels, n_classes=4, tolerance=0.01) == []


@pytest.mark.data
def test_real_ag_news_no_nulls(ag_news_raw) -> None:
    texts = ag_news_raw["train"]["text"]
    assert validate_no_nulls(texts) == []


@pytest.mark.data
def test_real_ag_news_expected_sizes(ag_news_raw) -> None:
    assert len(ag_news_raw["train"]) == 120_000
    assert len(ag_news_raw["test"]) == 7_600


@pytest.mark.data
def test_real_ag_news_no_train_test_leakage(ag_news_raw) -> None:
    train_texts = ag_news_raw["train"]["text"][:2000]
    test_texts = ag_news_raw["test"]["text"][:2000]
    assert validate_no_leakage(train_texts, test_texts) == []
