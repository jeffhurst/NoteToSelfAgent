from __future__ import annotations

from typing import TypedDict


class AgentState(TypedDict):
    workspace_root: str
    prompt_file: str
    notes_dir: str
    input_file_path: str
    input_text: str
    action: str | None
    reason: str | None
    note_text: str | None
    tool_name: str | None
    tool_input: str | None
    tool_result: str | None
    output_file_path: str | None
    cycle_started_at: str
    error: str | None
