"""Debug logging utilities — enabled via --debug CLI flag."""

from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager

DEBUG = False


def set_debug(enabled: bool) -> None:
    global DEBUG
    DEBUG = enabled


def debug_log(msg: str) -> None:
    """Print a debug message to stderr if debug mode is on."""
    if not DEBUG:
        return
    print(f"\033[90m[DEBUG] {msg}\033[0m", file=sys.stderr)


def debug_request(messages: list[dict], tools: list[dict], model: str, system: str) -> None:
    """Log a summary of the API request."""
    if not DEBUG:
        return
    from tokens import estimate_tool_tokens
    tool_tokens = estimate_tool_tokens(tools)
    debug_log(f"model={model} | {len(messages)} messages | {len(tools)} tools (~{tool_tokens} tokens)")
    if system:
        debug_log(f"system: {system[:100]}{'...' if len(system) > 100 else ''}")
    for i, msg in enumerate(messages):
        content = msg.get("content", "")
        role = msg.get("role", "?")
        summary = _summarize_content(content)
        debug_log(f"  msg[{i}] {role}: {summary}")


def debug_response(resp) -> None:
    """Log a summary of the API response."""
    if not DEBUG:
        return
    text_preview = resp.text[:150] + "..." if len(resp.text) > 150 else resp.text
    debug_log(
        f"response: {resp.input_tokens} in + {resp.output_tokens} out | "
        f"{len(resp.tool_calls)} tool calls | stop={resp.stop_reason}"
    )
    if resp.text:
        debug_log(f"  text: {text_preview}")
    for tc in resp.tool_calls:
        args_str = json.dumps(tc.arguments)
        if len(args_str) > 200:
            args_str = args_str[:200] + "..."
        debug_log(f"  tool: {tc.name}({args_str})")


@contextmanager
def debug_timer(label: str):
    """Context manager that logs elapsed time for an operation."""
    if not DEBUG:
        yield
        return
    start = time.time()
    yield
    elapsed = time.time() - start
    debug_log(f"{label}: {elapsed:.2f}s")


def _summarize_content(content) -> str:
    """Create a short summary of message content."""
    if isinstance(content, str):
        if len(content) > 200:
            return content[:200] + "..."
        return content

    if isinstance(content, list):
        parts = []
        for block in content[:3]:  # show first 3 blocks
            if isinstance(block, dict):
                btype = block.get("type", "?")
                if btype == "text":
                    text = block.get("text", "")
                    parts.append(f"text({len(text)} chars)")
                elif btype == "tool_use":
                    parts.append(f"tool_use({block.get('name', '?')})")
                elif btype == "tool_result":
                    result_content = block.get("content", "")
                    if isinstance(result_content, str):
                        size = len(result_content)
                    else:
                        size = len(str(result_content))
                    parts.append(f"tool_result({size} chars)")
                else:
                    parts.append(btype)
        suffix = f" +{len(content) - 3} more" if len(content) > 3 else ""
        return f"[{', '.join(parts)}{suffix}]"

    return str(content)[:200]
