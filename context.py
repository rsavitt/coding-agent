"""Context window management — summarize old turns to stay within token limits."""

from __future__ import annotations

import sys

# When input tokens exceed this, trigger compaction
COMPACT_THRESHOLD = 120_000

# Number of recent message pairs (user+assistant) to keep intact
KEEP_RECENT_PAIRS = 6

# System prompt for the summarizer call
_SUMMARIZER_SYSTEM = (
    "You are a conversation summarizer. Condense the provided conversation history "
    "into a concise summary that preserves: (1) what the user asked for, (2) what "
    "files were read/edited/created, (3) key decisions made, (4) current state of "
    "the task. Be factual and specific. Include file paths and function names."
)


def maybe_compact(messages: list[dict], input_tokens: int, provider,
                  model: str = "", threshold: int = COMPACT_THRESHOLD) -> int:
    """Compact messages if input_tokens exceeds threshold.

    Modifies messages in-place. Returns the number of messages removed
    (0 if no compaction happened).
    """
    if input_tokens < threshold:
        return 0

    # Need at least a few message pairs beyond what we keep to make compaction worthwhile
    # Each "pair" is typically 2 messages (assistant + user/tool_result), but the first
    # message is always user. So minimum messages = 1 (first) + KEEP_RECENT_PAIRS*2 + some to summarize
    min_messages = 1 + KEEP_RECENT_PAIRS * 2 + 4  # need at least 2 pairs to summarize
    if len(messages) < min_messages:
        return 0

    # Split: first message | middle (to summarize) | recent (to keep)
    keep_tail = KEEP_RECENT_PAIRS * 2  # each pair = assistant + user messages
    first_msg = messages[0]  # original user request
    middle = messages[1:-keep_tail]
    recent = messages[-keep_tail:]

    if len(middle) < 4:
        return 0  # not enough to summarize

    # Build the conversation text to summarize
    summary_text = _messages_to_text(middle)

    # Ask the provider to summarize
    summary = _call_summarizer(provider, summary_text, model)

    # Replace messages in-place: [first_msg, summary_msg, ...recent]
    summary_msg = {
        "role": "user",
        "content": (
            f"[The following is a summary of earlier conversation that was compacted "
            f"to save context space. {len(middle)} messages were summarized.]\n\n"
            f"{summary}"
        ),
    }

    removed_count = len(middle)
    messages.clear()
    messages.append(first_msg)
    messages.append(summary_msg)
    messages.extend(recent)

    print(
        f"\033[33m[context compacted: {removed_count} old messages summarized, "
        f"{len(messages)} messages remaining]\033[0m",
        file=sys.stderr,
    )
    return removed_count


def _call_summarizer(provider, conversation_text: str, model: str) -> str:
    """Use the provider to summarize conversation text."""
    call_kwargs = {"model": model} if model else {}
    resp = provider.call(
        messages=[{"role": "user", "content": (
            "Summarize this conversation history concisely:\n\n" + conversation_text
        )}],
        tools=[],
        system=_SUMMARIZER_SYSTEM,
        max_tokens=2048,
        **call_kwargs,
    )
    return resp.text


def _messages_to_text(messages: list[dict]) -> str:
    """Convert a list of messages to readable text for summarization."""
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if isinstance(content, str):
            parts.append(f"[{role}]: {content}")
        elif isinstance(content, list):
            # Content blocks (tool use / tool results)
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(f"[{role}]: {block['text']}")
                    elif block.get("type") == "tool_use":
                        args_summary = ", ".join(
                            f"{k}={repr(v)[:100]}" for k, v in block.get("input", {}).items()
                        )
                        parts.append(f"[{role} tool_use]: {block['name']}({args_summary})")
                    elif block.get("type") == "tool_result":
                        result_preview = str(block.get("content", ""))[:500]
                        parts.append(f"[tool_result]: {result_preview}")

    return "\n".join(parts)
