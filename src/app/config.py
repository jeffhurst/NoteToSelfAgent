from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class AppConfig:
    """Runtime configuration loaded from environment variables."""

    ollama_base_url: str
    ollama_model: str
    workspace_root: Path
    prompt_file: Path
    notes_dir: Path
    loop_delay_seconds: float



def load_config() -> AppConfig:
    """Load app config with defaults suitable for local development."""
    load_dotenv()
    workspace_root = Path(os.getenv("WORKSPACE_ROOT", os.getcwd())).resolve()
    prompt_file = Path(os.getenv("PROMPT_FILE", workspace_root / "prompt.txt"))
    notes_dir = Path(os.getenv("NOTES_DIR", workspace_root / "notes"))

    return AppConfig(
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/"),
        ollama_model=os.getenv("OLLAMA_MODEL", "gemma4:e4b"),
        workspace_root=workspace_root,
        prompt_file=prompt_file,
        notes_dir=notes_dir,
        loop_delay_seconds=float(os.getenv("LOOP_DELAY_SECONDS", "5")),
    )
