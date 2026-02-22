"""Tests for debug logging utilities."""

from __future__ import annotations

import io
import sys

import debug as debug_mod
from debug import (
    _summarize_content,
    debug_log,
    debug_request,
    debug_response,
    debug_timer,
    set_debug,
)
from providers import Response, ToolCall


class TestDebugLog:
    def setup_method(self):
        self._orig = debug_mod.DEBUG
        set_debug(False)

    def teardown_method(self):
        debug_mod.DEBUG = self._orig

    def test_silent_when_disabled(self, capsys):
        set_debug(False)
        debug_log("should not appear")
        assert capsys.readouterr().err == ""

    def test_prints_when_enabled(self, capsys):
        set_debug(True)
        debug_log("hello debug")
        assert "hello debug" in capsys.readouterr().err

    def test_prefix(self, capsys):
        set_debug(True)
        debug_log("test msg")
        assert "[DEBUG]" in capsys.readouterr().err


class TestDebugRequest:
    def setup_method(self):
        self._orig = debug_mod.DEBUG
        set_debug(True)

    def teardown_method(self):
        debug_mod.DEBUG = self._orig

    def test_logs_model_and_counts(self, capsys):
        messages = [{"role": "user", "content": "hi"}]
        tools = [{"name": "t1"}, {"name": "t2"}]
        debug_request(messages, tools, "test-model", "system prompt")
        err = capsys.readouterr().err
        assert "test-model" in err
        assert "1 messages" in err
        assert "2 tools" in err

    def test_logs_message_roles(self, capsys):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        debug_request(messages, [], "", "")
        err = capsys.readouterr().err
        assert "user" in err
        assert "assistant" in err

    def test_noop_when_disabled(self, capsys):
        set_debug(False)
        debug_request([{"role": "user", "content": "hi"}], [], "", "")
        assert capsys.readouterr().err == ""


class TestDebugResponse:
    def setup_method(self):
        self._orig = debug_mod.DEBUG
        set_debug(True)

    def teardown_method(self):
        debug_mod.DEBUG = self._orig

    def test_logs_token_counts(self, capsys):
        resp = Response(text="hello", input_tokens=100, output_tokens=50, stop_reason="end_turn")
        debug_response(resp)
        err = capsys.readouterr().err
        assert "100" in err
        assert "50" in err

    def test_logs_tool_calls(self, capsys):
        resp = Response(
            text="", tool_calls=[ToolCall(id="t1", name="read_file", arguments={"path": "foo.py"})],
            input_tokens=0, output_tokens=0,
        )
        debug_response(resp)
        err = capsys.readouterr().err
        assert "read_file" in err
        assert "foo.py" in err


class TestDebugTimer:
    def setup_method(self):
        self._orig = debug_mod.DEBUG
        set_debug(True)

    def teardown_method(self):
        debug_mod.DEBUG = self._orig

    def test_logs_elapsed_time(self, capsys):
        with debug_timer("test_op"):
            pass
        err = capsys.readouterr().err
        assert "test_op" in err
        assert "s" in err  # seconds

    def test_noop_when_disabled(self, capsys):
        set_debug(False)
        with debug_timer("test_op"):
            pass
        assert capsys.readouterr().err == ""


class TestSummarizeContent:
    def test_string_content(self):
        assert _summarize_content("hello") == "hello"

    def test_long_string_truncated(self):
        result = _summarize_content("x" * 300)
        assert len(result) < 250
        assert result.endswith("...")

    def test_list_content_blocks(self):
        content = [
            {"type": "text", "text": "hello world"},
            {"type": "tool_use", "name": "read_file", "id": "t1", "input": {}},
        ]
        result = _summarize_content(content)
        assert "text(" in result
        assert "tool_use(read_file)" in result

    def test_tool_result_blocks(self):
        content = [
            {"type": "tool_result", "tool_use_id": "t1", "content": "file contents here"},
        ]
        result = _summarize_content(content)
        assert "tool_result(" in result
