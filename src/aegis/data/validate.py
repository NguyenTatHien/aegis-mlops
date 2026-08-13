"""Data quality checks. Each function returns a list of human-readable
problem strings — empty list means the check passed. Kept as pure functions
over plain lists/arrays so tests/data/test_data_quality.py can exercise them
against both synthetic fixtures and the real AG News data."""

from __future__ import annotations

from collections import Counter


def validate_label_domain(labels: list[int], n_classes: int = 4) -> list[str]:
    bad = sorted({label for label in labels if not (0 <= label < n_classes)})
    return [f"label {value} outside expected domain [0, {n_classes})" for value in bad]


def validate_class_balance(
    labels: list[int], n_classes: int = 4, tolerance: float = 0.05
) -> list[str]:
    if not labels:
        return ["no labels to validate"]
    counts = Counter(labels)
    total = len(labels)
    expected = 1.0 / n_classes
    problems = []
    for cls in range(n_classes):
        share = counts.get(cls, 0) / total
        if abs(share - expected) > tolerance:
            problems.append(
                f"class {cls} share {share:.3f} deviates from expected {expected:.3f} by more than {tolerance}"
            )
    return problems


def validate_no_nulls(texts: list[str]) -> list[str]:
    n_blank = sum(1 for t in texts if not t or not t.strip())
    return [f"{n_blank} blank or whitespace-only texts found"] if n_blank else []


def validate_text_length(texts: list[str], min_words: int = 1, max_words: int = 300) -> list[str]:
    problems = []
    for i, t in enumerate(texts):
        n_words = len(t.split())
        if n_words < min_words:
            problems.append(f"text[{i}] has {n_words} words, below minimum {min_words}")
        elif n_words > max_words:
            problems.append(f"text[{i}] has {n_words} words, above maximum {max_words}")
    return problems


def validate_no_duplicates(texts: list[str], max_dup_ratio: float = 0.05) -> list[str]:
    if not texts:
        return []
    counts = Counter(texts)
    n_dup = sum(c - 1 for c in counts.values() if c > 1)
    ratio = n_dup / len(texts)
    return (
        [f"duplicate ratio {ratio:.3f} exceeds max {max_dup_ratio}"]
        if ratio > max_dup_ratio
        else []
    )


def validate_no_leakage(train_texts: list[str], test_texts: list[str]) -> list[str]:
    overlap = set(train_texts) & set(test_texts)
    return [f"{len(overlap)} texts appear in both train and test"] if overlap else []
