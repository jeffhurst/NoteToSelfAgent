from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

SUPPORTED_TOOL_NAMES = {
    "list_text_files",
    "list_workspace_files",
    "read_text_file",
    "search_text_files",
    "write_text_file",
    "append_text_file",
    "run_shell_command",
    "web_search",
}

TOOLS_REQUIRING_NONEMPTY_INPUT = {
    "read_text_file",
    "search_text_files",
    "list_workspace_files",
    "write_text_file",
    "append_text_file",
    "run_shell_command",
    "web_search",
}


class DecisionOutput(BaseModel):
    action: Literal["note", "tool"]
    reason: str = Field(min_length=1)
    note_text: str | None = None
    tool_name: str | None = None
    tool_input: str | None = None

    @model_validator(mode="after")
    def validate_action_payload(self) -> "DecisionOutput":
        if self.action == "note":
            if not self.note_text or not self.note_text.strip():
                raise ValueError("note_text must be provided when action=note")
        if self.action == "tool":
            if not self.tool_name or self.tool_name not in SUPPORTED_TOOL_NAMES:
                raise ValueError(f"tool_name must be one of {sorted(SUPPORTED_TOOL_NAMES)}")
            if (
                self.tool_name in TOOLS_REQUIRING_NONEMPTY_INPUT
                and (not self.tool_input or not self.tool_input.strip())
            ):
                raise ValueError("tool_input must be provided when action=tool")
        return self


class SynthesisOutput(BaseModel):
    note_text: str = Field(min_length=1)


class TimestampsOutput(BaseModel):
    run_summary: str = Field(min_length=1)
    next_goal: str = Field(min_length=1)
