from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass
class OllamaClient:
    base_url: str
    model: str
    temperature: float = 0.2
    timeout: float = 60.0

    def chat_json(self, *, system_prompt: str, user_prompt: str) -> str:
        """Call Ollama /api/chat and return the raw response content text."""
        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {"temperature": self.temperature},
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()

        message = data.get("message", {})
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError(f"Unexpected Ollama response format: {data}")
        return content
