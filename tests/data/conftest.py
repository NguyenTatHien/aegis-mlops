from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def ag_news_raw():
    """Loads the real AG News dataset once per test session. Network-bound
    on first run; the `data` marker keeps this out of the default PR CI job
    (design.md D11) — run explicitly with `pytest -m data`."""
    from aegis.data.loader import load_ag_news

    return load_ag_news()
