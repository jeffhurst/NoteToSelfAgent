from __future__ import annotations


def build_decision_system_prompt() -> str:
    return (
        "You are an autonomous local note/tool loop agent. "
        "Decide exactly one action: 'note' or 'tool'. "
        "Available tools: list_text_files, read_text_file, search_text_files. "
        "If a tool is needed to gather relevant information, choose 'tool', otherwise choose 'note'. "
        "When choosing tool, pick exactly one tool. "
        "For list_text_files, tool_input may be empty. "
        "For read_text_file and search_text_files, tool_input must be non-empty. "
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


def build_synthesis_user_prompt(input_text: str, tool_name: str, tool_input: str, tool_result: str) -> str:
    return (
        "Original input:\n"
        f"{input_text}\n\n"
        f"Tool name: {tool_name}\n"
        f"Tool input: {tool_input}\n"
        f"Tool result:\n{tool_result}\n\n"
        "Return only JSON."
    )
