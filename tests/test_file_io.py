from __future__ import annotations

from datetime import datetime

import pytest

from app.file_io import get_latest_note_file, make_timestamped_note_filename, select_input_file


def test_timestamp_filename_windows_safe() -> None:
    name = make_timestamped_note_filename(datetime(2026, 4, 22, 19, 45, 10, 123456))
    assert name == "2026-04-22_19-45-10_123456.txt"
    assert ":" not in name


def test_select_input_file_uses_prompt_when_no_notes(tmp_path) -> None:
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("seed", encoding="utf-8")
    notes_dir = tmp_path / "notes"
    selected = select_input_file(prompt, notes_dir)
    assert selected == prompt


def test_select_input_file_uses_latest_note(tmp_path) -> None:
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("seed", encoding="utf-8")
    notes = tmp_path / "notes"
    notes.mkdir()
    old = notes / "2026-04-22_10-00-00_000001.txt"
    new = notes / "2026-04-22_10-00-00_000002.txt"
    old.write_text("old", encoding="utf-8")
    new.write_text("new", encoding="utf-8")
    selected = select_input_file(prompt, notes)
    assert selected == new


def test_get_latest_note_file_none_when_missing(tmp_path) -> None:
    assert get_latest_note_file(tmp_path / "notes") is None


def test_select_input_file_missing_prompt_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        select_input_file(tmp_path / "prompt.txt", tmp_path / "notes")
