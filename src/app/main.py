from __future__ import annotations

from app.config import load_config
from app.loop import run_forever


def main() -> None:
    config = load_config()
    print(
        "Starting NoteToSelfAgent with "
        f"OLLAMA_BASE_URL={config.ollama_base_url}, "
        f"OLLAMA_MODEL={config.ollama_model}, "
        f"WORKSPACE_ROOT={config.workspace_root}",
        flush=True,
    )
    run_forever(config)


if __name__ == "__main__":
    main()
