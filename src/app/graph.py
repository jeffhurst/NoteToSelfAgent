from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from app.file_io import (
    append_utf8_text,
    ensure_notes_dir,
    make_timestamped_note_filename,
    read_utf8_text,
    select_input_file,
    write_note_file,
)
from app.ollama_client import OllamaClient
from app.prompts import (
    build_corrective_json_prompt,
    build_decision_system_prompt,
    build_decision_user_prompt,
    build_synthesis_system_prompt,
    build_synthesis_user_prompt,
)
from app.schemas import DecisionOutput, SynthesisOutput
from app.state import AgentState
from app.tools import format_tool_result, list_text_files, read_text_file, search_text_files


def _log(msg: str) -> None:
    print(msg, flush=True)


def _extract_json(raw: str) -> dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start : end + 1])
        raise


def _parse_with_retry(
    client: OllamaClient,
    raw_text: str,
    schema_hint: str,
    parser,
    node_name: str,
) -> Any:
    try:
        return parser(raw_text)
    except Exception as first_err:  # noqa: BLE001
        _log(f"[{datetime.now().isoformat()}] {node_name} parse failed. Retrying once. Error: {first_err}")
        corrective = client.chat_json(
            system_prompt="Return valid JSON only.",
            user_prompt=build_corrective_json_prompt(raw_text, schema_hint),
        )
        _log(f"[{datetime.now().isoformat()}] {node_name} raw llm output retry:\n{corrective}")
        return parser(corrective)


def build_cycle_graph(client: OllamaClient):
    graph = StateGraph(AgentState)

    def select_input_file_node(state: AgentState) -> AgentState:
        prompt_file = Path(state["prompt_file"]) 
        notes_dir = Path(state["notes_dir"])
        ensure_notes_dir(notes_dir)
        selected = select_input_file(prompt_file=prompt_file)
        _log(f"Selected input file: {selected}")
        return {**state, "input_file_path": str(selected)}

    def read_input_text_node(state: AgentState) -> AgentState:
        input_path = Path(state["input_file_path"])
        text = read_utf8_text(input_path)
        preview = text[:300].replace("\n", " ")
        _log(f"Input preview: {preview}")
        return {**state, "input_text": text}

    def decide_action_node(state: AgentState) -> AgentState:
        raw = client.chat_json(
            system_prompt=build_decision_system_prompt(),
            user_prompt=build_decision_user_prompt(state["input_text"]),
        )
        _log(f"[{datetime.now().isoformat()}] decide_action raw llm output:\n{raw}")

        def parser(txt: str) -> DecisionOutput:
            obj = _extract_json(txt)
            return DecisionOutput.model_validate(obj)

        try:
            decision = _parse_with_retry(
                client,
                raw,
                "{action,note_text,tool_name,tool_input,reason}",
                parser,
                "decide_action",
            )
            _log(
                "Parsed decision: "
                f"action={decision.action}, reason={decision.reason}, "
                f"tool_name={decision.tool_name}, tool_input={decision.tool_input}"
            )
            return {
                **state,
                "action": decision.action,
                "reason": decision.reason,
                "note_text": decision.note_text,
                "tool_name": decision.tool_name,
                "tool_input": decision.tool_input,
            }
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            fallback_note = (
                "Deterministic fallback note due to decision parse failure. "
                f"Error: {exc}. Raw excerpt: {raw[:500]}"
            )
            _log(f"Decision parse failed after retry; using fallback note. Error: {exc}")
            return {
                **state,
                "action": "note",
                "reason": "fallback_parse_failure",
                "note_text": fallback_note,
                "error": str(exc),
            }

    def route_after_decision(state: AgentState) -> str:
        return "run_tool" if state.get("action") == "tool" else "write_note_file"

    def run_tool_node(state: AgentState) -> AgentState:
        workspace_root = state["workspace_root"]
        tool_name = state.get("tool_name")
        tool_input = state.get("tool_input") or ""
        if not tool_name:
            raise ValueError("tool_name was not set for run_tool")

        if tool_name == "list_text_files":
            result = list_text_files(workspace_root)
        elif tool_name == "read_text_file":
            result = read_text_file(tool_input, workspace_root)
        elif tool_name == "search_text_files":
            result = search_text_files(tool_input, workspace_root, max_results=10)
        else:
            raise ValueError(f"Unsupported tool_name: {tool_name}")

        formatted = format_tool_result(tool_name, result)
        _log(f"Tool output ({tool_name}):\n{formatted}")
        return {**state, "tool_result": formatted}

    def synthesize_note_node(state: AgentState) -> AgentState:
        raw = client.chat_json(
            system_prompt=build_synthesis_system_prompt(),
            user_prompt=build_synthesis_user_prompt(
                state["input_text"],
                state.get("tool_name") or "",
                state.get("tool_input") or "",
                state.get("tool_result") or "",
            ),
        )
        _log(f"[{datetime.now().isoformat()}] synthesize_note_from_tool_result raw llm output:\n{raw}")

        def parser(txt: str) -> SynthesisOutput:
            obj = _extract_json(txt)
            return SynthesisOutput.model_validate(obj)

        try:
            synthesis = _parse_with_retry(
                client,
                raw,
                "{note_text}",
                parser,
                "synthesize_note_from_tool_result",
            )
            _log(f"Final note text: {synthesis.note_text}")
            return {**state, "note_text": synthesis.note_text}
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            fallback_note = (
                "Deterministic fallback note due to synthesis parse failure. "
                f"Error: {exc}. Raw excerpt: {raw[:500]}"
            )
            _log(f"Synthesis parse failed after retry; using fallback note. Error: {exc}")
            return {**state, "note_text": fallback_note, "error": str(exc)}

    def write_note_file_node(state: AgentState) -> AgentState:
        notes_dir = Path(state["notes_dir"])
        ensure_notes_dir(notes_dir)
        out_name = make_timestamped_note_filename()
        out_path = notes_dir / out_name

        action = state.get("action") or "note"
        lines = [
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}",
            f"Input Source: {Path(state['input_file_path']).as_posix()}",
            f"Action: {action}",
        ]
        if action == "tool":
            lines.extend(
                [
                    f"Tool Name: {state.get('tool_name') or ''}",
                    f"Tool Input: {state.get('tool_input') or ''}",
                    "",
                    "Tool Result:",
                    state.get("tool_result") or "",
                ]
            )
        lines.extend(["", "Note to Future Self:", state.get("note_text") or ""])

        if state.get("error"):
            lines.extend(["", "Error:", state["error"]])

        content = "\n".join(lines).strip() + "\n"
        write_note_file(out_path, content)

        note_text = (state.get("note_text") or "").strip()
        if note_text:
            prompt_path = Path(state["prompt_file"])
            append_utf8_text(prompt_path, f"\n{note_text}\n")
            _log(f"Appended note_text to prompt file: {prompt_path}")

        _log(f"Output file written: {out_path}")
        return {**state, "output_file_path": str(out_path)}

    graph.add_node("select_input_file", select_input_file_node)
    graph.add_node("read_input_text", read_input_text_node)
    graph.add_node("decide_action", decide_action_node)
    graph.add_node("run_tool", run_tool_node)
    graph.add_node("synthesize_note_from_tool_result", synthesize_note_node)
    graph.add_node("write_note_file", write_note_file_node)

    graph.add_edge(START, "select_input_file")
    graph.add_edge("select_input_file", "read_input_text")
    graph.add_edge("read_input_text", "decide_action")
    graph.add_conditional_edges("decide_action", route_after_decision)
    graph.add_edge("run_tool", "synthesize_note_from_tool_result")
    graph.add_edge("synthesize_note_from_tool_result", "write_note_file")
    graph.add_edge("write_note_file", END)

    return graph.compile()
