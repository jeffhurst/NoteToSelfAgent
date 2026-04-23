from __future__ import annotations

from datetime import datetime
from pathlib import Path


TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M-%S_%f"


def ensure_notes_dir(notes_dir: Path) -> None:
    notes_dir.mkdir(parents=True, exist_ok=True)


def make_timestamped_note_filename(now: datetime | None = None) -> str:
    now = now or datetime.now()
    return f"{now.strftime(TIMESTAMP_FORMAT)}.txt"


def get_latest_note_file(notes_dir: Path) -> Path | None:
    if not notes_dir.exists():
        return None
    txt_files = [p for p in notes_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"]
    if not txt_files:
        return None
    return sorted(txt_files, key=lambda p: p.name)[-1]


def select_input_file(prompt_file: Path) -> Path:
    if not prompt_file.exists():
        raise FileNotFoundError(
            f"Prompt file not found at {prompt_file}. Create prompt.txt before running the agent."
        )
    return prompt_file


def read_utf8_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def write_note_file(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
