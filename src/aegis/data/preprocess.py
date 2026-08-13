"""Two preprocessing paths, kept strictly separate (design.md D6).

clean_text_tfidf() ported verbatim from notebooks/aegis_ag_news_training.ipynb
cell 17 (Phase 2 — TF-IDF baseline). It MUST only ever be called on the
baseline branch.

passthrough() exists so callers have an explicit, testable no-op for the
RoBERTa branch instead of skipping preprocessing implicitly — RoBERTa was
trained on raw text (cell 31 tokenizes dataset["train"]["text"] directly),
so lowercasing/stripping digits here would be train/serve skew.
"""

from __future__ import annotations

import re

_URL_RE = re.compile(r"http\S+|www\S+")
_DIGIT_RE = re.compile(r"\d+")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")
_MULTI_SPACE_RE = re.compile(r"\s+")


def clean_text_tfidf(text: str) -> str:
    text = text.lower()
    text = _URL_RE.sub("", text)
    text = _DIGIT_RE.sub("", text)
    text = _NON_ALPHA_RE.sub("", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def passthrough(text: str) -> str:
    return text
