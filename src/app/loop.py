from __future__ import annotations

import time
from datetime import datetime

from app.config import AppConfig
from app.graph import build_cycle_graph
from app.ollama_client import OllamaClient
from app.state import AgentState


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
            "cycle_started_at": started,
            "error": None,
        }

        try:
            final_state = cycle_graph.invoke(state)
            print(f"Cycle {cycle_num} completed. Output: {final_state.get('output_file_path')}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"Cycle {cycle_num} failed: {exc}", flush=True)

        print(f"Sleeping {config.loop_delay_seconds} seconds before next cycle...", flush=True)
        time.sleep(config.loop_delay_seconds)
        cycle_num += 1
