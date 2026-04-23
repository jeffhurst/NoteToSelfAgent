from __future__ import annotations

import pytest

from app.schemas import DecisionOutput


def test_tool_decision_allows_empty_input_for_list_text_files() -> None:
    parsed = DecisionOutput.model_validate(
        {
            "action": "tool",
            "reason": "inspect files first",
            "note_text": "",
            "tool_name": "list_text_files",
            "tool_input": "",
        }
    )
    assert parsed.tool_name == "list_text_files"
    assert parsed.tool_input == ""


def test_tool_decision_requires_input_for_read_text_file() -> None:
    with pytest.raises(ValueError, match="tool_input must be provided when action=tool"):
        DecisionOutput.model_validate(
            {
                "action": "tool",
                "reason": "need file contents",
                "note_text": "",
                "tool_name": "read_text_file",
                "tool_input": "",
            }
        )


def test_tool_decision_allows_web_search_with_input() -> None:
    parsed = DecisionOutput.model_validate(
        {
            "action": "tool",
            "reason": "search online docs",
            "note_text": "",
            "tool_name": "web_search",
            "tool_input": "langgraph stategraph example",
        }
    )
    assert parsed.tool_name == "web_search"
