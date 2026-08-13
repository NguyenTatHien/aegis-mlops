from __future__ import annotations

import re

import pytest

from aegis.models.labels import id_to_label, is_raw_label_placeholder, label_to_id

LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
_LABEL_X_RE = re.compile(r"^LABEL_\d$")


@pytest.mark.unit
@pytest.mark.parametrize(
    "index,expected",
    [(0, "World"), (1, "Sports"), (2, "Business"), (3, "Sci/Tech")],
)
def test_id_to_label_ag_news_order(index: int, expected: str) -> None:
    assert id_to_label(index, LABEL_NAMES) == expected


@pytest.mark.unit
def test_id_to_label_out_of_range_raises() -> None:
    with pytest.raises(ValueError):
        id_to_label(4, LABEL_NAMES)


@pytest.mark.unit
def test_label_to_id_roundtrip() -> None:
    for i, name in enumerate(LABEL_NAMES):
        assert label_to_id(name, LABEL_NAMES) == i


@pytest.mark.unit
def test_no_label_ever_matches_placeholder_pattern() -> None:
    for name in LABEL_NAMES:
        assert not _LABEL_X_RE.match(name)


@pytest.mark.unit
@pytest.mark.parametrize("placeholder", ["LABEL_0", "LABEL_1", "LABEL_2", "LABEL_3"])
def test_is_raw_label_placeholder_detects_hf_default(placeholder: str) -> None:
    assert is_raw_label_placeholder(placeholder) is True


@pytest.mark.unit
@pytest.mark.parametrize("real_label", LABEL_NAMES)
def test_is_raw_label_placeholder_false_for_real_labels(real_label: str) -> None:
    assert is_raw_label_placeholder(real_label) is False
