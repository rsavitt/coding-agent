"""Tests for session persistence."""

from __future__ import annotations

import json
import os
import tempfile

import pytest
from session import auto_save, list_sessions, load_session, save_session


@pytest.fixture
def session_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestSaveAndLoad:
    def test_save_and_load_roundtrip(self, session_dir):
        messages = [
            {"role": "user", "content": "Fix the bug"},
            {"role": "assistant", "content": "I'll look at the code."},
        ]
        path = save_session(messages, session_id="test-1", session_dir=session_dir)
        assert os.path.isfile(path)

        loaded = load_session("test-1", session_dir=session_dir)
        assert loaded == messages

    def test_save_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = os.path.join(tmpdir, "a", "b", "c")
            save_session([{"role": "user", "content": "hi"}],
                         session_id="test", session_dir=nested)
            assert os.path.isfile(os.path.join(nested, "test.jsonl"))

    def test_load_nonexistent_returns_empty(self, session_dir):
        loaded = load_session("nonexistent", session_dir=session_dir)
        assert loaded == []

    def test_load_by_full_path(self, session_dir):
        messages = [{"role": "user", "content": "hello"}]
        path = save_session(messages, session_id="test-path", session_dir=session_dir)
        loaded = load_session(path)
        assert loaded == messages

    def test_saves_as_jsonl(self, session_dir):
        messages = [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
            {"role": "user", "content": "three"},
        ]
        path = save_session(messages, session_id="jsonl-test", session_dir=session_dir)

        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 3
        for line in lines:
            json.loads(line)  # should not raise

    def test_auto_generated_session_id(self, session_dir):
        path = save_session([{"role": "user", "content": "hi"}],
                            session_dir=session_dir)
        assert "session-" in path
        assert path.endswith(".jsonl")

    def test_content_blocks_serialize(self, session_dir):
        """Messages with content block lists (tool use) should serialize."""
        messages = [
            {"role": "assistant", "content": [
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "id": "t1", "name": "read_file",
                 "input": {"path": "foo.py"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1",
                 "content": "file contents"},
            ]},
        ]
        save_session(messages, session_id="blocks", session_dir=session_dir)
        loaded = load_session("blocks", session_dir=session_dir)
        assert loaded == messages


class TestListSessions:
    def test_lists_saved_sessions(self, session_dir):
        save_session([{"role": "user", "content": "first"}],
                     session_id="s1", session_dir=session_dir)
        save_session([{"role": "user", "content": "second"}],
                     session_id="s2", session_dir=session_dir)

        sessions = list_sessions(session_dir=session_dir)
        assert len(sessions) == 2
        ids = {s["id"] for s in sessions}
        assert ids == {"s1", "s2"}

    def test_includes_preview(self, session_dir):
        save_session([{"role": "user", "content": "Fix the authentication bug"}],
                     session_id="s1", session_dir=session_dir)

        sessions = list_sessions(session_dir=session_dir)
        assert sessions[0]["preview"] == "Fix the authentication bug"

    def test_empty_directory(self, session_dir):
        sessions = list_sessions(session_dir=session_dir)
        assert sessions == []

    def test_nonexistent_directory(self):
        sessions = list_sessions(session_dir="/nonexistent/path/xyz123")
        assert sessions == []

    def test_limit(self, session_dir):
        for i in range(10):
            save_session([{"role": "user", "content": f"session {i}"}],
                         session_id=f"s{i}", session_dir=session_dir)

        sessions = list_sessions(session_dir=session_dir, limit=3)
        assert len(sessions) == 3

    def test_sorted_newest_first(self, session_dir):
        import time
        save_session([{"role": "user", "content": "older"}],
                     session_id="older", session_dir=session_dir)
        time.sleep(0.05)  # ensure different mtime
        save_session([{"role": "user", "content": "newer"}],
                     session_id="newer", session_dir=session_dir)

        sessions = list_sessions(session_dir=session_dir)
        assert sessions[0]["id"] == "newer"


class TestAutoSave:
    def test_auto_save_works(self, session_dir):
        messages = [{"role": "user", "content": "hello"}]
        auto_save(messages, "auto-test", session_dir=session_dir)

        loaded = load_session("auto-test", session_dir=session_dir)
        assert loaded == messages

    def test_auto_save_ignores_errors(self):
        """auto_save should not raise on write errors."""
        # Try to save to an invalid path — should not raise
        auto_save([{"role": "user", "content": "hi"}],
                  "test", session_dir="/proc/nonexistent/impossible")
