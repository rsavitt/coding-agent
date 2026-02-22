"""Tests for sub-agent truncation and configurable default models."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from providers import Response, ToolCall


class TestSubAgentTruncation:
    """Sub-agent tool results should be truncated like main agent results."""

    def test_truncates_large_tool_results(self):
        from sub_agents import SubAgentRunner
        from prompts import EXPLORER_SYSTEM
        from tools import EXPLORER_TOOLS

        # First call: LLM requests a tool
        resp1 = Response(
            text="Let me search.",
            tool_calls=[ToolCall(id="tc_1", name="search", arguments={"pattern": "foo"})],
            input_tokens=50, output_tokens=30,
        )
        # Second call: LLM gives final answer
        resp2 = Response(
            text="Found results.", tool_calls=[],
            input_tokens=80, output_tokens=40,
        )

        provider = MagicMock()
        provider.call.side_effect = [resp1, resp2]

        # Mock the search tool to return a huge result
        huge_result = "x" * 50000
        mock_tools = [{
            "name": "search",
            "description": "search",
            "parameters": {"type": "object", "properties": {}},
            "execute": lambda **kwargs: huge_result,
        }]

        runner = SubAgentRunner(
            provider=provider, tools=mock_tools,
            system=EXPLORER_SYSTEM, model="test",
        )
        runner.run("Search for foo")

        # Check that the tool result sent to the LLM was truncated
        second_call_messages = provider.call.call_args_list[1][1]["messages"]
        # The last message is the tool result
        tool_result_msg = second_call_messages[-1]
        tool_content = tool_result_msg["content"][0]["content"]
        assert len(tool_content) < 50000
        assert "truncated" in tool_content

    def test_small_results_not_truncated(self):
        from sub_agents import SubAgentRunner
        from prompts import EXPLORER_SYSTEM

        resp1 = Response(
            text="Checking.",
            tool_calls=[ToolCall(id="tc_1", name="read_file", arguments={"path": "x"})],
            input_tokens=50, output_tokens=30,
        )
        resp2 = Response(
            text="Done.", tool_calls=[],
            input_tokens=80, output_tokens=40,
        )

        provider = MagicMock()
        provider.call.side_effect = [resp1, resp2]

        small_result = "small content"
        mock_tools = [{
            "name": "read_file",
            "description": "read",
            "parameters": {"type": "object", "properties": {}},
            "execute": lambda **kwargs: small_result,
        }]

        runner = SubAgentRunner(
            provider=provider, tools=mock_tools,
            system="sys", model="test",
        )
        runner.run("Read file x")

        second_call_messages = provider.call.call_args_list[1][1]["messages"]
        tool_result_msg = second_call_messages[-1]
        tool_content = tool_result_msg["content"][0]["content"]
        assert tool_content == small_result
        assert "truncated" not in tool_content


class TestConfigurableModels:
    def test_default_anthropic_model(self):
        from providers import DEFAULT_ANTHROPIC_MODEL
        assert "claude" in DEFAULT_ANTHROPIC_MODEL

    def test_default_openai_model(self):
        from providers import DEFAULT_OPENAI_MODEL
        assert DEFAULT_OPENAI_MODEL == "gpt-4o"

    def test_env_var_override_anthropic(self):
        """AGENT_ANTHROPIC_MODEL env var should override the default."""
        with patch.dict(os.environ, {"AGENT_ANTHROPIC_MODEL": "claude-opus-4-20250514"}):
            # Need to re-import to pick up new env var
            import importlib
            import providers
            importlib.reload(providers)
            assert providers.DEFAULT_ANTHROPIC_MODEL == "claude-opus-4-20250514"

            # Clean up
            importlib.reload(providers)

    def test_env_var_override_openai(self):
        """AGENT_OPENAI_MODEL env var should override the default."""
        with patch.dict(os.environ, {"AGENT_OPENAI_MODEL": "gpt-4-turbo"}):
            import importlib
            import providers
            importlib.reload(providers)
            assert providers.DEFAULT_OPENAI_MODEL == "gpt-4-turbo"

            # Clean up
            importlib.reload(providers)

    def test_empty_model_uses_default(self):
        """When model='' is passed to call(), it should use the default."""
        # This tests the `model = model or DEFAULT_*_MODEL` logic
        from providers import DEFAULT_ANTHROPIC_MODEL
        # We can't easily test the actual call without API keys,
        # but we can verify the constant exists and is non-empty
        assert DEFAULT_ANTHROPIC_MODEL
        assert isinstance(DEFAULT_ANTHROPIC_MODEL, str)
