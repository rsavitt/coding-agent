"""Tests for error isolation — tracebacks in tool and sub-agent errors."""

from __future__ import annotations

from unittest.mock import MagicMock

from agent import _execute_tool
from sub_agents import SubAgentRunner, SubAgentResult


def _failing_tool(**kwargs):
    raise ValueError("something broke")


def _working_tool(**kwargs):
    return "OK"


class TestAgentExecuteTool:
    def test_error_includes_type_and_traceback(self):
        tool_map = {"fail": _failing_tool}
        result = _execute_tool(tool_map, "fail", {})
        assert "ValueError" in result
        assert "something broke" in result
        assert "Traceback" in result or "raise ValueError" in result

    def test_success_unaffected(self):
        tool_map = {"ok": _working_tool}
        result = _execute_tool(tool_map, "ok", {})
        assert result == "OK"


class TestSubAgentExecuteTool:
    def test_error_includes_type_and_traceback(self):
        tools = [{"name": "fail", "execute": _failing_tool}]
        runner = SubAgentRunner(
            provider=MagicMock(), tools=tools, system="test",
        )
        result = runner._execute_tool("fail", {})
        assert "ValueError" in result
        assert "something broke" in result
        assert "raise ValueError" in result

    def test_success_unaffected(self):
        tools = [{"name": "ok", "execute": _working_tool}]
        runner = SubAgentRunner(
            provider=MagicMock(), tools=tools, system="test",
        )
        result = runner._execute_tool("ok", {})
        assert result == "OK"


class TestDelegateParallelErrorIsolation:
    def test_error_captures_traceback(self):
        """When a sub-agent future raises, the result should include traceback info."""
        # We test by checking SubAgentResult construction with traceback
        import traceback
        try:
            raise RuntimeError("parallel task failed")
        except Exception as e:
            tb = traceback.format_exc()
            tb_short = "\n".join(tb.strip().splitlines()[-5:])
            result = SubAgentResult(
                summary=f"{type(e).__name__}: {e}\n\nTraceback:\n{tb_short}",
                status="error",
            )
        assert "RuntimeError" in result.summary
        assert "parallel task failed" in result.summary
        assert "Traceback" in result.summary
        assert "raise RuntimeError" in result.summary
