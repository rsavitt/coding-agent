"""Tests for OpenAI message conversion robustness."""

from __future__ import annotations

from providers import _to_openai_messages


class TestPlainStringMessages:
    def test_string_content_passes_through(self):
        msgs = [{"role": "user", "content": "hello"}]
        result = _to_openai_messages(msgs)
        assert result == [{"role": "user", "content": "hello"}]

    def test_multiple_string_messages(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        result = _to_openai_messages(msgs)
        assert result == msgs


class TestToolResultBlocks:
    def test_basic_tool_results(self):
        msgs = [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "result text"},
        ]}]
        result = _to_openai_messages(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "t1"
        assert result[0]["content"] == "result text"

    def test_tool_result_with_list_content(self):
        """Tool result content can be a list of content blocks."""
        msgs = [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": [
                {"type": "text", "text": "line 1"},
                {"type": "text", "text": "line 2"},
            ]},
        ]}]
        result = _to_openai_messages(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert "line 1" in result[0]["content"]
        assert "line 2" in result[0]["content"]

    def test_multiple_tool_results(self):
        msgs = [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "r1"},
            {"type": "tool_result", "tool_use_id": "t2", "content": "r2"},
        ]}]
        result = _to_openai_messages(msgs)
        assert len(result) == 2
        assert result[0]["tool_call_id"] == "t1"
        assert result[1]["tool_call_id"] == "t2"


class TestAssistantToolUse:
    def test_text_and_tool_use(self):
        msgs = [{"role": "assistant", "content": [
            {"type": "text", "text": "Let me check."},
            {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "foo.py"}},
        ]}]
        result = _to_openai_messages(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "Let me check."
        assert len(result[0]["tool_calls"]) == 1
        assert result[0]["tool_calls"][0]["function"]["name"] == "read_file"

    def test_text_only_assistant(self):
        msgs = [{"role": "assistant", "content": [
            {"type": "text", "text": "All done."},
        ]}]
        result = _to_openai_messages(msgs)
        assert result[0]["content"] == "All done."
        assert "tool_calls" not in result[0]


class TestImageBlocks:
    def test_image_content_block(self):
        msgs = [{"role": "user", "content": [
            {"type": "image", "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "iVBORw0KGgo=",
            }},
        ]}]
        result = _to_openai_messages(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        content = result[0]["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "image_url"
        assert "data:image/png;base64,iVBORw0KGgo=" in content[0]["image_url"]["url"]

    def test_mixed_text_and_image(self):
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image", "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": "/9j/4AAQ",
            }},
        ]}]
        result = _to_openai_messages(msgs)
        assert len(result) == 1
        content = result[0]["content"]
        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "What's in this image?"
        assert content[1]["type"] == "image_url"


class TestEdgeCases:
    def test_empty_content_list(self):
        msgs = [{"role": "user", "content": []}]
        result = _to_openai_messages(msgs)
        assert result == []

    def test_mixed_tool_result_and_text_in_same_message(self):
        """User message with both tool_result and text blocks."""
        msgs = [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "result here"},
            {"type": "text", "text": "Also, please check this."},
        ]}]
        result = _to_openai_messages(msgs)
        # Should produce a tool message AND a user message
        tool_msgs = [m for m in result if m["role"] == "tool"]
        user_msgs = [m for m in result if m["role"] == "user"]
        assert len(tool_msgs) == 1
        assert len(user_msgs) == 1
        assert tool_msgs[0]["content"] == "result here"

    def test_fallback_for_unknown_content_type(self):
        msgs = [{"role": "user", "content": {"unexpected": "dict"}}]
        result = _to_openai_messages(msgs)
        # Falls through to fallback
        assert result == msgs
