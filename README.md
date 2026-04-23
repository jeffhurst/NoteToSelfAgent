# NoteToSelfAgent

A complete local autonomous note/tool loop agent built with:
- **LangGraph Graph API** for per-cycle orchestration
- **Ollama** over HTTP (`/api/chat`) for LLM decisions and synthesis
- **Python** with typed modules, validation, and tests

## What it does

Each cycle:
1. Selects input (`prompt.txt` plus latest `notes/timestamps.txt` when present)
2. Reads input text
3. Asks LLM to choose either:
   - `note` (directly write next note), or
   - `tool` (run one tool, then synthesize next note)
4. Writes exactly one **new timestamped** note file in `notes/`
5. Writes/updates `notes/timestamps.txt` with a concise run summary and next goal

The loop then sleeps and repeats forever.

## Architecture

- **Outer loop**: `while True` in `app.loop.run_forever`
- **Per-cycle graph**: LangGraph in `app.graph.build_cycle_graph`

Flow:

```text
START
  ↓
select_input_file
  ↓
read_input_text
  ↓
decide_action
  ├── note → write_note_file → END
  └── tool → run_tool → synthesize_note_from_tool_result → write_note_file → END
```

## Project layout

```text
.
├── prompt.txt
├── pyproject.toml
├── README.md
├── src/
│   └── app/
│       ├── __init__.py
│       ├── config.py
│       ├── file_io.py
│       ├── graph.py
│       ├── loop.py
│       ├── main.py
│       ├── ollama_client.py
│       ├── prompts.py
│       ├── schemas.py
│       ├── state.py
│       └── tools.py
└── tests/
    ├── test_file_io.py
    └── test_tools.py
```

`notes/` is created at runtime.

## Requirements

- Python 3.11+
- Local Ollama server running
- Model available locally, default: `gemma4:e4b`

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e .[dev]
```

## Run

```bash
python -m app.main
```

## Configuration

Environment variables:

- `OLLAMA_BASE_URL` (default `http://127.0.0.1:11434`; common trailing punctuation like `)` is sanitized)
- `OLLAMA_MODEL` (default `gemma4:e4b`)
- `WORKSPACE_ROOT` (default current working directory)
- `PROMPT_FILE` (default `<workspace_root>/prompt.txt`)
- `NOTES_DIR` (default `<workspace_root>/notes`)
- `LOOP_DELAY_SECONDS` (default `5`)

## Tools exposed to the model

- `list_text_files`
- `list_workspace_files`
- `read_text_file`
- `search_text_files`
- `write_text_file`
- `append_text_file`
- `run_shell_command`
- `web_search`

Safety:
- path resolution constrained to workspace root
- `.txt` only for read/search
- excludes `.git`, `.venv`, `venv`, `__pycache__`, `.pytest_cache`, `node_modules`

## Output files

Each cycle writes one new filename like:

`YYYY-MM-DD_HH-MM-SS_ffffff.txt`

Example:

`2026-04-22_19-45-10_123456.txt`

Format includes timestamp, input source, action, optional tool info/result, and final note text.

## Logging behavior

Console output includes:
- cycle number
- selected input file
- input preview
- raw LLM output before parsing
- parsed decision
- tool output summary
- final note preview
- written output file
- sleep delay

## Tests

Run deterministic tests:

```bash
pytest
```

Covered:
- timestamp filename safety
- input file selection rules
- latest note detection helper behavior
- path safety
- recursive `.txt` listing with exclusions
- text search result structure

## Error resilience

- If a cycle crashes (for example network issues or malformed runtime config), the app writes a timestamped fallback error note into `notes/` and continues looping.
- This guarantees each cycle attempt still leaves a traceable note artifact for future cycles.
