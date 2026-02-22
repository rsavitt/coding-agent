"""LLM provider abstraction — Anthropic and OpenAI-compatible APIs."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


# Default models — override via environment variables
DEFAULT_ANTHROPIC_MODEL = os.environ.get("AGENT_ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
DEFAULT_OPENAI_MODEL = os.environ.get("AGENT_OPENAI_MODEL", "gpt-4o")


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
             model: str = "", max_tokens: int = 8192) -> Response:
        model = model or DEFAULT_ANTHROPIC_MODEL
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


class OpenAIProvider:
    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        import openai
        kwargs = {"api_key": api_key or os.environ["OPENAI_API_KEY"]}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = openai.OpenAI(**kwargs)

    def call(self, messages: list[dict], tools: list[dict], system: str = "",
             model: str = "", max_tokens: int = 8192) -> Response:
        model = model or DEFAULT_OPENAI_MODEL
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
            import json
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
                        import json
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
