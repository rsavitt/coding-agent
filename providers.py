"""LLM provider abstraction — Anthropic and OpenAI-compatible APIs."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class Response:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str = ""


class AnthropicProvider:
    def __init__(self, api_key: str | None = None):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])

    def call(self, messages: list[dict], tools: list[dict], system: str = "",
             model: str = "claude-sonnet-4-20250514", max_tokens: int = 8192) -> Response:
        kwargs = dict(model=model, max_tokens=max_tokens, messages=messages)
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [_to_anthropic_tool(t) for t in tools]
        resp = self.client.messages.create(**kwargs)

        text_parts, tool_calls = [], []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=block.input))

        return Response(
            text="\n".join(text_parts),
            tool_calls=tool_calls,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            stop_reason=resp.stop_reason,
        )

    def call_streaming(self, messages: list[dict], tools: list[dict], system: str = "",
                       model: str = "claude-sonnet-4-20250514", max_tokens: int = 8192) -> Response:
        """Stream response, printing text tokens as they arrive."""
        kwargs = dict(model=model, max_tokens=max_tokens, messages=messages)
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [_to_anthropic_tool(t) for t in tools]

        text_parts = []
        tool_calls = []
        input_tokens = output_tokens = 0
        stop_reason = ""

        # Track tool_use blocks being built incrementally
        current_tool_id = None
        current_tool_name = None
        current_tool_json = ""

        with self.client.messages.stream(**kwargs) as stream:
            for event in stream:
                if event.type == "message_start":
                    input_tokens = event.message.usage.input_tokens
                elif event.type == "message_delta":
                    output_tokens = event.usage.output_tokens
                    stop_reason = event.delta.stop_reason or ""
                elif event.type == "content_block_start":
                    block = event.content_block
                    if block.type == "text":
                        pass  # text deltas come in content_block_delta
                    elif block.type == "tool_use":
                        current_tool_id = block.id
                        current_tool_name = block.name
                        current_tool_json = ""
                elif event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        sys.stdout.write(delta.text)
                        sys.stdout.flush()
                        text_parts.append(delta.text)
                    elif delta.type == "input_json_delta":
                        current_tool_json += delta.partial_json
                elif event.type == "content_block_stop":
                    if current_tool_id is not None:
                        tool_calls.append(ToolCall(
                            id=current_tool_id,
                            name=current_tool_name,
                            arguments=json.loads(current_tool_json) if current_tool_json else {},
                        ))
                        current_tool_id = None
                        current_tool_name = None
                        current_tool_json = ""

        full_text = "".join(text_parts)
        if full_text:
            sys.stdout.write("\n")
            sys.stdout.flush()

        return Response(
            text=full_text,
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason=stop_reason,
        )


class OpenAIProvider:
    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        import openai
        kwargs = {"api_key": api_key or os.environ["OPENAI_API_KEY"]}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = openai.OpenAI(**kwargs)

    def call(self, messages: list[dict], tools: list[dict], system: str = "",
             model: str = "gpt-4o", max_tokens: int = 8192) -> Response:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(_to_openai_messages(messages))

        kwargs = dict(model=model, max_tokens=max_tokens, messages=msgs)
        if tools:
            kwargs["tools"] = [_to_openai_tool(t) for t in tools]
        resp = self.client.chat.completions.create(**kwargs)

        choice = resp.choices[0]
        text = choice.message.content or ""
        tool_calls = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id, name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                ))

        usage = resp.usage
        return Response(
            text=text, tool_calls=tool_calls,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            stop_reason=choice.finish_reason,
        )

    def call_streaming(self, messages: list[dict], tools: list[dict], system: str = "",
                       model: str = "gpt-4o", max_tokens: int = 8192) -> Response:
        """Stream response, printing text tokens as they arrive."""
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(_to_openai_messages(messages))

        kwargs = dict(model=model, max_tokens=max_tokens, messages=msgs,
                      stream=True, stream_options={"include_usage": True})
        if tools:
            kwargs["tools"] = [_to_openai_tool(t) for t in tools]

        text_parts = []
        # tool_calls_by_index accumulates partial tool call data
        tool_calls_by_index: dict[int, dict] = {}
        input_tokens = output_tokens = 0
        stop_reason = ""

        stream = self.client.chat.completions.create(**kwargs)
        for chunk in stream:
            if chunk.usage:
                input_tokens = chunk.usage.prompt_tokens
                output_tokens = chunk.usage.completion_tokens

            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            if choice.finish_reason:
                stop_reason = choice.finish_reason

            delta = choice.delta
            if delta is None:
                continue

            # Stream text
            if delta.content:
                sys.stdout.write(delta.content)
                sys.stdout.flush()
                text_parts.append(delta.content)

            # Accumulate tool calls
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_by_index:
                        tool_calls_by_index[idx] = {
                            "id": "", "name": "", "arguments": "",
                        }
                    entry = tool_calls_by_index[idx]
                    if tc_delta.id:
                        entry["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            entry["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            entry["arguments"] += tc_delta.function.arguments

        full_text = "".join(text_parts)
        if full_text:
            sys.stdout.write("\n")
            sys.stdout.flush()

        # Build final tool calls
        tool_calls = []
        for idx in sorted(tool_calls_by_index):
            entry = tool_calls_by_index[idx]
            tool_calls.append(ToolCall(
                id=entry["id"],
                name=entry["name"],
                arguments=json.loads(entry["arguments"]) if entry["arguments"] else {},
            ))

        return Response(
            text=full_text, tool_calls=tool_calls,
            input_tokens=input_tokens, output_tokens=output_tokens,
            stop_reason=stop_reason,
        )


def auto_detect_provider() -> AnthropicProvider | OpenAIProvider:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicProvider()
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAIProvider()
    raise RuntimeError("Set ANTHROPIC_API_KEY or OPENAI_API_KEY")


def _to_anthropic_tool(tool: dict) -> dict:
    return {
        "name": tool["name"],
        "description": tool["description"],
        "input_schema": tool["parameters"],
    }


def _to_openai_tool(tool: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["parameters"],
        },
    }


def _to_openai_messages(messages: list[dict]) -> list[dict]:
    """Convert Anthropic-style messages (content blocks) to OpenAI format."""
    result = []
    for msg in messages:
        content = msg["content"]

        # Plain string content — pass through
        if isinstance(content, str):
            result.append(msg)
            continue

        # List of content blocks — need translation
        if isinstance(content, list):
            # Check if it's tool results (user role with tool_result blocks)
            if content and isinstance(content[0], dict) and content[0].get("type") == "tool_result":
                for block in content:
                    result.append({
                        "role": "tool",
                        "tool_call_id": block["tool_use_id"],
                        "content": block["content"],
                    })
                continue

            # Assistant message with tool_use blocks
            if msg["role"] == "assistant":
                text_parts = []
                tool_calls = []
                for block in content:
                    if block.get("type") == "text":
                        text_parts.append(block["text"])
                    elif block.get("type") == "tool_use":
                        tool_calls.append({
                            "id": block["id"],
                            "type": "function",
                            "function": {
                                "name": block["name"],
                                "arguments": json.dumps(block["input"]),
                            },
                        })
                oai_msg = {"role": "assistant", "content": "\n".join(text_parts) or None}
                if tool_calls:
                    oai_msg["tool_calls"] = tool_calls
                result.append(oai_msg)
                continue

        # Fallback
        result.append(msg)
    return result
