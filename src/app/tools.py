from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import httpx

EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
}

MAX_TOOL_COMMAND_OUTPUT = 8000


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


def list_workspace_files(root_dir: str, glob_pattern: str = "*") -> list[str]:
    root = Path(root_dir).resolve()
    results: list[str] = []
    for path in root.rglob(glob_pattern):
        if not path.is_file():
            continue
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


def write_text_file(path: str, content: str, workspace_root: str) -> str:
    root = Path(workspace_root)
    resolved = resolve_workspace_path(root, path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} characters to {path}"


def append_text_file(path: str, content: str, workspace_root: str) -> str:
    root = Path(workspace_root)
    resolved = resolve_workspace_path(root, path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("a", encoding="utf-8") as handle:
        handle.write(content)
    return f"Appended {len(content)} characters to {path}"


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


def run_shell_command(command: str, workspace_root: str, timeout_seconds: int = 20) -> str:
    root = Path(workspace_root).resolve()
    completed = subprocess.run(
        command,
        cwd=str(root),
        shell=True,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    stdout = (completed.stdout or "")[:MAX_TOOL_COMMAND_OUTPUT]
    stderr = (completed.stderr or "")[:MAX_TOOL_COMMAND_OUTPUT]
    return (
        f"exit_code={completed.returncode}\n"
        f"stdout:\n{stdout}\n"
        f"stderr:\n{stderr}"
    )


def web_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    q = query.strip()
    if not q:
        return []
    url = f"https://duckduckgo.com/html/?q={quote_plus(q)}"
    response = httpx.get(url, timeout=15.0, follow_redirects=True)
    response.raise_for_status()
    html = response.text
    rows: list[dict[str, str]] = []
    marker = 'class="result__a"'
    for chunk in html.split(marker)[1:]:
        href_key = 'href="'
        href_start = chunk.find(href_key)
        if href_start < 0:
            continue
        href_start += len(href_key)
        href_end = chunk.find('"', href_start)
        if href_end < 0:
            continue
        href = chunk[href_start:href_end]
        title_end = chunk.find("</a>")
        if title_end < 0:
            continue
        title = chunk[chunk.find(">") + 1 : title_end]
        title = " ".join(title.replace("\n", " ").split())
        rows.append({"title": title, "url": href})
        if len(rows) >= max_results:
            break
    return rows


def format_tool_result(tool_name: str, result: Any) -> str:
    if tool_name == "list_text_files":
        files = result if isinstance(result, list) else []
        return "\n".join(files) if files else "No text files found."
    if tool_name == "list_workspace_files":
        files = result if isinstance(result, list) else []
        return "\n".join(files) if files else "No files found."
    if tool_name == "read_text_file":
        return str(result)
    if tool_name in {"write_text_file", "append_text_file", "run_shell_command"}:
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
    if tool_name == "web_search":
        rows = result if isinstance(result, list) else []
        if not rows:
            return "No search results."
        return "\n".join(f"- {row['title']}: {row['url']}" for row in rows)
    return str(result)
