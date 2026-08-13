from __future__ import annotations

import pytest

from aegis.data.preprocess import clean_text_tfidf, passthrough


@pytest.mark.unit
def test_clean_text_tfidf_lowercases_strips_digits_and_punctuation() -> None:
    out = clean_text_tfidf("Apple's Q3 revenue hit $89.5B in 2024!")
    assert out == out.lower()
    assert not any(ch.isdigit() for ch in out)
    assert "$" not in out and "!" not in out and "'" not in out


@pytest.mark.unit
def test_clean_text_tfidf_removes_urls() -> None:
    out = clean_text_tfidf("Read more at https://example.com/news for details")
    assert "http" not in out and "example" not in out


@pytest.mark.unit
def test_clean_text_tfidf_collapses_whitespace() -> None:
    out = clean_text_tfidf("too    many     spaces")
    assert "  " not in out


@pytest.mark.unit
def test_passthrough_is_identity() -> None:
    text = "Apple's Q3 revenue hit $89.5B in 2024!"
    assert passthrough(text) == text


@pytest.mark.unit
def test_passthrough_preserves_case_digits_and_punctuation() -> None:
    text = "The S&P 500 rose 2.3% on Tuesday."
    out = passthrough(text)
    assert out == text
    assert any(ch.isupper() for ch in out)
    assert any(ch.isdigit() for ch in out)
