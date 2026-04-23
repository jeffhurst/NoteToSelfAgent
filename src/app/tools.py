from __future__ import annotations

from pathlib import Path
from typing import Any

EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
}


def resolve_workspace_path(workspace_root: Path, user_path: str) -> Path:
    """Resolve user-supplied path and enforce workspace boundary."""
    root = workspace_root.resolve()
    candidate = (root / user_path).resolve()
    if root not in candidate.parents and candidate != root:
        raise ValueError(f"Path escapes workspace: {user_path}")
    if candidate.suffix.lower() != ".txt":
        raise ValueError("Only .txt files are allowed")
    return candidate


def list_text_files(root_dir: str) -> list[str]:
    root = Path(root_dir).resolve()
    results: list[str] = []
    for path in root.rglob("*.txt"):
        if any(part in EXCLUDED_DIRS for part in path.parts):
            continue
        try:
            rel = path.resolve().relative_to(root)
        except ValueError:
            continue
        results.append(rel.as_posix())
    return sorted(set(results))


def read_text_file(path: str, workspace_root: str) -> str:
    root = Path(workspace_root)
    resolved = resolve_workspace_path(root, path)
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(f"Text file not found: {path}")
    try:
        return resolved.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return resolved.read_text(encoding="utf-8", errors="replace")


def search_text_files(query: str, root_dir: str, max_results: int = 10) -> list[dict[str, Any]]:
    root = Path(root_dir).resolve()
    q = query.lower().strip()
    if not q:
        return []

    matches: list[dict[str, Any]] = []
    for rel_path in list_text_files(str(root)):
        file_path = (root / rel_path).resolve()
        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        for idx, line in enumerate(lines, start=1):
            if q in line.lower():
                start = max(0, idx - 2)
                end = min(len(lines), idx + 1)
                context = "\n".join(lines[start:end])
                matches.append(
                    {
                        "file_path": rel_path,
                        "line_number": idx,
                        "line_text": line,
                        "context_preview": context,
                    }
                )
    matches.sort(key=lambda m: (m["file_path"], m["line_number"], m["line_text"]))
    return matches[:max_results]


def format_tool_result(tool_name: str, result: Any) -> str:
    if tool_name == "list_text_files":
        files = result if isinstance(result, list) else []
        return "\n".join(files) if files else "No text files found."
    if tool_name == "read_text_file":
        return str(result)
    if tool_name == "search_text_files":
        rows = result if isinstance(result, list) else []
        if not rows:
            return "No matches found."
        formatted = []
        for row in rows:
            formatted.append(
                f"{row['file_path']}:{row['line_number']} | {row['line_text']}\nContext:\n{row['context_preview']}"
            )
        return "\n\n".join(formatted)
    return str(result)
