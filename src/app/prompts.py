from __future__ import annotations


def build_decision_system_prompt() -> str:
    return (
        "You are an autonomous local note/tool loop agent. "
        "Decide exactly one action: 'note' or 'tool'. "
        "Available tools: list_text_files, list_workspace_files, read_text_file, search_text_files, write_text_file, append_text_file, run_shell_command, web_search. "
        "Tool input format rules: "
        "list_text_files -> empty string allowed. "
        "list_workspace_files -> glob pattern string (example: '**/*.py'). "
        "read_text_file -> relative .txt path string. "
        "search_text_files -> query string. "
        "write_text_file / append_text_file -> JSON string with keys {path, content}. "
        "run_shell_command -> shell command string. "
        "web_search -> search query string. "
        "If a tool is needed to gather relevant information, choose 'tool', otherwise choose 'note'. "
        "When choosing tool, pick exactly one tool. "
        "Only use write/append on .txt files inside workspace. "
        "Return ONLY valid JSON with keys: action, reason, note_text, tool_name, tool_input. "
        "Keep reason short."
    )


def build_decision_user_prompt(input_text: str) -> str:
    return (
        "Current input text:\n"
        f"{input_text}\n\n"
        "Return only JSON."
    )


def build_corrective_json_prompt(raw_text: str, schema_hint: str) -> str:
    return (
        "Your previous response was invalid for the required JSON schema. "
        f"Schema: {schema_hint}. "
        "Rewrite your answer as valid JSON only.\n"
        f"Invalid response:\n{raw_text}"
    )


def build_synthesis_system_prompt() -> str:
    return (
        "You are writing a concise note to your future self. "
        "Use the original input and tool result. "
        "Return ONLY valid JSON with key: note_text."
    )


def build_timestamps_system_prompt() -> str:
    return (
        "Summarize this cycle for future context. "
        "Return ONLY valid JSON with keys: run_summary and next_goal. "
        "Both fields must be very short and concise."
    )


def build_timestamps_user_prompt(
    input_text: str,
    action: str,
    reason: str,
    tool_name: str,
    tool_input: str,
    tool_result: str,
    note_text: str,
    error: str,
) -> str:
    return (
        f"Input text:\n{input_text}\n\n"
        f"Action: {action}\n"
        f"Reason: {reason}\n"
        f"Tool name: {tool_name}\n"
        f"Tool input: {tool_input}\n"
        f"Tool result:\n{tool_result}\n\n"
        f"Final note:\n{note_text}\n\n"
        f"Error:\n{error}\n\n"
        "Return only JSON."
    )


def build_synthesis_user_prompt(input_text: str, tool_name: str, tool_input: str, tool_result: str) -> str:
    return (
        "Original input:\n"
        f"{input_text}\n\n"
        f"Tool name: {tool_name}\n"
        f"Tool input: {tool_input}\n"
        f"Tool result:\n{tool_result}\n\n"
        "Return only JSON."
    )
