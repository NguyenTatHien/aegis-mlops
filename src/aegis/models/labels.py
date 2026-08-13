"""Index <-> label name mapping.

Single source of truth is ood_config.json's label_names (design.md D7) —
never roberta_final/config.json, whose id2label is LABEL_0..LABEL_3.
Callers pass label_names explicitly (sourced from aegis.config.get_label_names())
rather than this module reaching into config itself, so it stays a pure
function testable without any file on disk.
"""

from __future__ import annotations

_LABEL_PATTERN_PREFIX = "LABEL_"


def id_to_label(index: int, label_names: list[str]) -> str:
    if not 0 <= index < len(label_names):
        raise ValueError(f"label index {index} out of range for {len(label_names)} labels")
    return label_names[index]


def label_to_id(label: str, label_names: list[str]) -> int:
    return label_names.index(label)


def is_raw_label_placeholder(label: str) -> bool:
    """True for the HuggingFace default id2label placeholders (LABEL_0..3)
    that must never reach an API response."""
    return label.startswith(_LABEL_PATTERN_PREFIX) and label[len(_LABEL_PATTERN_PREFIX) :].isdigit()
