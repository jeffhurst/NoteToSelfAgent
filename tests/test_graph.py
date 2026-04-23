from __future__ import annotations

from app.graph import build_cycle_graph


class FakeClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat_json(self, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        if self.calls == 1:
            return (
                '{"action":"note","reason":"record memory",'
                '"note_text":"remember this","tool_name":"","tool_input":""}'
            )
        return (
            '{"run_summary":"created a short note","next_goal":"continue from latest context"}'
        )


class FakeToolClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat_json(self, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        if self.calls == 1:
            return (
                '{"action":"tool","reason":"need context","note_text":"",'
                '"tool_name":"list_text_files","tool_input":""}'
            )
        return '{"note_text":"captured from tool"}'


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
            "timestamps_file_path": None,
            "cycle_started_at": "2026-04-23T00:00:00",
            "error": None,
        }
    )

    assert final_state["action"] == "note"
    assert prompt_file.read_text(encoding="utf-8") == "seed instructions\n"
    timestamps_file = notes_dir / "timestamps.txt"
    assert timestamps_file.exists()
    assert "Run Summary:" in timestamps_file.read_text(encoding="utf-8")


def test_tool_action_appends_final_note_text_to_prompt_file(tmp_path) -> None:
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("seed instructions\n", encoding="utf-8")
    notes_dir = tmp_path / "notes"

    graph = build_cycle_graph(FakeToolClient())
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
            "timestamps_file_path": None,
            "cycle_started_at": "2026-04-23T00:00:00",
            "error": None,
        }
    )

    assert final_state["action"] == "tool"
    assert "captured from tool" not in prompt_file.read_text(encoding="utf-8")


def test_input_includes_latest_timestamps_file(tmp_path) -> None:
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("base prompt", encoding="utf-8")
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    (notes_dir / "timestamps.txt").write_text("Run Summary: previous\nNext Goal: continue", encoding="utf-8")
    client = FakeClient()
    graph = build_cycle_graph(client)

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
            "timestamps_file_path": None,
            "cycle_started_at": "2026-04-23T00:00:00",
            "error": None,
        }
    )
    assert "Latest Timestamps" in final_state["input_text"]
