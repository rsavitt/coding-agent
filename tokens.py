"""Token estimation utilities — no external tokenizer dependencies."""

from __future__ import annotations

import json

# Average characters per token across GPT/Claude models.
# Conservative estimate (slightly over-counts, which is safer for budgeting).
_CHARS_PER_TOKEN = 3.5


def estimate_tokens(text: str) -> int:
    """Estimate token count from text using character-based heuristic.

    Uses ~3.5 chars/token which slightly over-estimates (safer for budgeting).
    For exact counts, use a model-specific tokenizer.
    """
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def estimate_tool_tokens(tools: list[dict]) -> int:
    """Estimate token overhead from tool/function definitions.

    Tool schemas are serialized as JSON in the API request and consume
    input tokens. This estimates that overhead so it can be factored
    into context budget decisions.
    """
    if not tools:
        return 0

    total = 0
    for tool in tools:
        # Build the schema portion that gets sent to the API
        schema = {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters", {}),
        }
        total += estimate_tokens(json.dumps(schema))

    # Add per-tool overhead for formatting/delimiters (roughly 10 tokens each)
    total += len(tools) * 10

    return total


def estimate_system_tokens(system: str) -> int:
    """Estimate tokens used by the system prompt."""
    if not system:
        return 0
    return estimate_tokens(system)
