"""Session persistence — save and restore conversation history."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

# Default directory for session files
SESSION_DIR = os.environ.get("AGENT_SESSION_DIR", os.path.expanduser("~/.coding-agent/sessions"))


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_session(messages: list[dict], session_id: str | None = None,
                 session_dir: str = SESSION_DIR) -> str:
    """Save messages to a JSONL session file. Returns the session file path."""
    _ensure_dir(session_dir)
    if session_id is None:
        session_id = f"session-{int(time.time())}"

    filepath = os.path.join(session_dir, f"{session_id}.jsonl")

    with open(filepath, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg, default=str) + "\n")

    return filepath


def load_session(session_id: str, session_dir: str = SESSION_DIR) -> list[dict]:
    """Load messages from a session file. Returns empty list if not found."""
    # Allow passing a full path or just an ID
    if os.path.isfile(session_id):
        filepath = session_id
    else:
        filepath = os.path.join(session_dir, f"{session_id}.jsonl")

    if not os.path.isfile(filepath):
        return []

    messages = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                messages.append(json.loads(line))
    return messages


def list_sessions(session_dir: str = SESSION_DIR, limit: int = 20) -> list[dict]:
    """List recent sessions, newest first."""
    if not os.path.isdir(session_dir):
        return []

    sessions = []
    for name in os.listdir(session_dir):
        if not name.endswith(".jsonl"):
            continue
        filepath = os.path.join(session_dir, name)
        stat = os.stat(filepath)
        session_id = name[:-6]  # strip .jsonl

        # Read first message to get a preview
        preview = ""
        try:
            with open(filepath) as f:
                first_line = f.readline().strip()
                if first_line:
                    msg = json.loads(first_line)
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        preview = content[:80]
        except (json.JSONDecodeError, OSError):
            pass

        sessions.append({
            "id": session_id,
            "path": filepath,
            "modified": stat.st_mtime,
            "size": stat.st_size,
            "preview": preview,
        })

    sessions.sort(key=lambda s: s["modified"], reverse=True)
    return sessions[:limit]


def auto_save(messages: list[dict], session_id: str,
              session_dir: str = SESSION_DIR) -> None:
    """Save session, silently ignoring errors (for use in REPL auto-save)."""
    try:
        save_session(messages, session_id=session_id, session_dir=session_dir)
    except OSError:
        pass
