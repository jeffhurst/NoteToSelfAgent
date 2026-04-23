from __future__ import annotations

from app.graph import build_cycle_graph


class FakeClient:
    def chat_json(self, system_prompt: str, user_prompt: str) -> str:
        return (
            '{"action":"note","reason":"record memory",'
            '"note_text":"remember this","tool_name":"","tool_input":""}'
        )


def test_note_action_appends_note_text_to_prompt_file(tmp_path) -> None:
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("seed instructions\n", encoding="utf-8")
    notes_dir = tmp_path / "notes"

    graph = build_cycle_graph(FakeClient())
    final_state = graph.invoke(
        {
            "workspace_root": str(tmp_path),
            "prompt_file": str(prompt_file),
            "notes_dir": str(notes_dir),
            "input_file_path": "",
            "input_text": "",
            "action": None,
            "reason": None,
            "note_text": None,
            "tool_name": None,
            "tool_input": None,
            "tool_result": None,
            "output_file_path": None,
            "cycle_started_at": "2026-04-23T00:00:00",
            "error": None,
        }
    )

    assert final_state["action"] == "note"
    assert "remember this" in prompt_file.read_text(encoding="utf-8")
