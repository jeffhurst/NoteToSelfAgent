from __future__ import annotations

from urllib.parse import urlparse


def sanitize_ollama_base_url(raw_url: str) -> str:
    """Normalize likely human input mistakes in Ollama base URL values."""
    cleaned = raw_url.strip().strip('"\'').rstrip("),.;")
    parsed = urlparse(cleaned)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            "OLLAMA_BASE_URL must be a valid http(s) base URL, for example http://127.0.0.1:11434"
        )
    return cleaned.rstrip("/")
