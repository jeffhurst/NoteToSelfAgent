from __future__ import annotations

import pytest

from app.tools import list_text_files, read_text_file, resolve_workspace_path, search_text_files


def test_resolve_workspace_path_blocks_escape(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with pytest.raises(ValueError):
        resolve_workspace_path(workspace, "../outside.txt")


def test_resolve_workspace_path_requires_txt(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with pytest.raises(ValueError):
        resolve_workspace_path(workspace, "notes.md")


def test_list_text_files_excludes_directories(tmp_path) -> None:
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "secret.txt").write_text("x", encoding="utf-8")
    (tmp_path / "dir").mkdir()
    (tmp_path / "dir" / "b.txt").write_text("b", encoding="utf-8")

    result = list_text_files(str(tmp_path))
    assert result == ["a.txt", "dir/b.txt"]


def test_read_text_file_reads_under_workspace(tmp_path) -> None:
    target = tmp_path / "hello.txt"
    target.write_text("hello", encoding="utf-8")
    assert read_text_file("hello.txt", str(tmp_path)) == "hello"


def test_search_text_files_returns_matches(tmp_path) -> None:
    (tmp_path / "one.txt").write_text("alpha\nneedles here\nomega\n", encoding="utf-8")
    (tmp_path / "two.txt").write_text("Needles again\n", encoding="utf-8")

    result = search_text_files("needles", str(tmp_path), max_results=10)
    assert len(result) == 2
    assert result[0]["file_path"] == "one.txt"
    assert result[0]["line_number"] == 2
    assert "needles" in result[0]["line_text"].lower()
