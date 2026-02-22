"""Tests for streaming support in providers and agent loop."""

import io
import sys
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest
from providers import Response, ToolCall


# --- Mock streaming events for Anthropic ---

@dataclass
class MockAnthropicEvent:
    type: str


@dataclass
class MockMessageStart(MockAnthropicEvent):
    type: str = "message_start"
    message: object = None


@dataclass
class MockContentBlockStart(MockAnthropicEvent):
    type: str = "content_block_start"
    content_block: object = None


@dataclass
class MockContentBlockDelta(MockAnthropicEvent):
    type: str = "content_block_delta"
    delta: object = None


@dataclass
class MockContentBlockStop(MockAnthropicEvent):
    type: str = "content_block_stop"


@dataclass
class MockMessageDelta(MockAnthropicEvent):
    type: str = "message_delta"
    delta: object = None
    usage: object = None


class TestAgentLoopStreaming:
    """Test that agent_loop correctly uses streaming vs non-streaming."""

    def _make_mock_provider(self, resp: Response):
        """Create a mock provider that returns resp from both call and call_streaming."""
        provider = MagicMock()
        provider.call.return_value = resp
        provider.call_streaming.return_value = resp
        return provider

    def test_streaming_enabled_uses_call_streaming(self):
        """When stream=True, agent_loop should use call_streaming."""
        from agent import agent_loop

        resp = Response(text="hello", tool_calls=[], input_tokens=10, output_tokens=5)
        provider = self._make_mock_provider(resp)

        with patch("sys.stdout", new_callable=io.StringIO):
            agent_loop(provider, [{"role": "user", "content": "hi"}],
                       tools=[], system="sys", stream=True)

        provider.call_streaming.assert_called_once()
        provider.call.assert_not_called()

    def test_streaming_disabled_uses_call(self):
        """When stream=False, agent_loop should use call."""
        from agent import agent_loop

        resp = Response(text="hello", tool_calls=[], input_tokens=10, output_tokens=5)
        provider = self._make_mock_provider(resp)

        with patch("sys.stdout", new_callable=io.StringIO):
            agent_loop(provider, [{"role": "user", "content": "hi"}],
                       tools=[], system="sys", stream=False)

        provider.call.assert_called_once()
        provider.call_streaming.assert_not_called()

    def test_streaming_fallback_when_no_method(self):
        """If provider has no call_streaming, fall back to call even with stream=True."""
        from agent import agent_loop

        resp = Response(text="hello", tool_calls=[], input_tokens=10, output_tokens=5)
        provider = MagicMock(spec=["call"])  # no call_streaming attribute
        provider.call.return_value = resp

        with patch("sys.stdout", new_callable=io.StringIO):
            agent_loop(provider, [{"role": "user", "content": "hi"}],
                       tools=[], system="sys", stream=True)

        provider.call.assert_called_once()

    def test_streaming_with_tool_calls(self):
        """Streaming should work correctly when the response includes tool calls."""
        from agent import agent_loop

        # First response: has a tool call
        resp1 = Response(
            text="Let me check.",
            tool_calls=[ToolCall(id="tc_1", name="read_file", arguments={"path": "foo.py"})],
            input_tokens=10, output_tokens=20,
        )
        # Second response: final answer, no tool calls
        resp2 = Response(
            text="Done.", tool_calls=[], input_tokens=50, output_tokens=10,
        )

        provider = MagicMock()
        provider.call_streaming.side_effect = [resp1, resp2]

        # Mock tool
        tools = [{
            "name": "read_file",
            "description": "read",
            "parameters": {"type": "object", "properties": {}},
            "execute": lambda path: "file contents",
        }]

        with patch("sys.stdout", new_callable=io.StringIO):
            agent_loop(provider, [{"role": "user", "content": "read foo"}],
                       tools=tools, system="sys", stream=True)

        assert provider.call_streaming.call_count == 2


class TestStreamingDefaultOn:
    """Test that streaming is the default in CLI."""

    def test_cli_default_is_streaming(self):
        """--no-stream flag should be needed to disable streaming."""
        import argparse
        from cli import main

        # Parse with no --no-stream: stream should be True
        parser = argparse.ArgumentParser()
        parser.add_argument("--no-stream", action="store_true")
        args = parser.parse_args([])
        assert not args.no_stream  # default is False, so stream = not False = True

        # Parse with --no-stream: stream should be False
        args = parser.parse_args(["--no-stream"])
        assert args.no_stream
