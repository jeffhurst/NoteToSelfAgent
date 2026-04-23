from __future__ import annotations

import pytest

from app.url_utils import sanitize_ollama_base_url


def test_sanitize_ollama_base_url_strips_trailing_punctuation() -> None:
    assert sanitize_ollama_base_url("http://192.168.1.165:11434)") == "http://192.168.1.165:11434"


def test_sanitize_ollama_base_url_invalid_raises() -> None:
    with pytest.raises(ValueError):
        sanitize_ollama_base_url("not-a-url")
