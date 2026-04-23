from __future__ import annotations

import traceback
from datetime import datetime
import time

from app.config import AppConfig
from app.file_io import ensure_notes_dir, make_timestamped_note_filename, write_note_file
from app.graph import build_cycle_graph
from app.ollama_client import OllamaClient
from app.state import AgentState


def _write_error_note(config: AppConfig, cycle_num: int, exc: Exception) -> str:
    """Write a deterministic timestamped note when a cycle crashes."""
    ensure_notes_dir(config.notes_dir)
    output_path = config.notes_dir / make_timestamped_note_filename()
    content = "\n".join(
        [
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}",
            "Input Source: cycle_error",
            "Action: note",
            "",
            "Note to Future Self:",
            f"Cycle {cycle_num} failed before normal completion.",
            f"Error Type: {type(exc).__name__}",
            f"Error Message: {exc}",
            "",
            "Traceback:",
            traceback.format_exc(limit=20),
        ]
    ).strip() + "\n"
    write_note_file(output_path, content)
    return str(output_path)



def run_forever(config: AppConfig) -> None:
    """Run the autonomous loop forever, one LangGraph invocation per cycle."""
    client = OllamaClient(base_url=config.ollama_base_url, model=config.ollama_model)
    cycle_graph = build_cycle_graph(client)

    cycle_num = 1
    while True:
        started = datetime.now().isoformat()
        print("=" * 80, flush=True)
        print(f"Cycle {cycle_num} started at {started}", flush=True)
        state: AgentState = {
            "workspace_root": str(config.workspace_root),
            "prompt_file": str(config.prompt_file),
            "notes_dir": str(config.notes_dir),
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
            "cycle_started_at": started,
            "error": None,
        }

        try:
            final_state = cycle_graph.invoke(state)
            print(f"Cycle {cycle_num} completed. Output: {final_state.get('output_file_path')}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"Cycle {cycle_num} failed: {exc}", flush=True)
            error_note_path = _write_error_note(config, cycle_num, exc)
            print(f"Wrote fallback error note: {error_note_path}", flush=True)

        print(f"Sleeping {config.loop_delay_seconds} seconds before next cycle...", flush=True)
        time.sleep(config.loop_delay_seconds)
        cycle_num += 1
