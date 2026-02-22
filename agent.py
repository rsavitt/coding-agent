"""Core agent loop — tool execution, context management, permission checks."""

from __future__ import annotations

import re
import sys

from context import maybe_compact
from debug import debug_log, debug_request, debug_response, debug_timer
from providers import Response
from tokens import estimate_system_tokens, estimate_tool_tokens

# Commands considered safe to run without user confirmation
SAFE_BASH_PREFIXES = (
    "ls", "cat", "head", "tail", "find", "grep", "rg", "wc", "file", "which",
    "git status", "git diff", "git log", "git branch", "git show",
    "python -m pytest", "pytest", "python -m mypy", "mypy",
    "ruff", "black --check", "npm test", "npm run", "cargo test",
)

# Shell metacharacters that chain or nest commands
_SHELL_OPERATORS = re.compile(r";|&&|\|\||`|\$\(")
# Pipe is allowed (e.g. "grep foo | head") but each segment is checked


def agent_loop(provider, messages: list[dict], tools: list[dict], system: str,
               model: str = "", max_turns: int = 100, stream: bool = True) -> None:
    """Run the agent loop: call LLM, execute tools, repeat until done."""
    tool_map = {t["name"]: t["execute"] for t in tools}
    total_in, total_out = 0, 0
    use_streaming = stream and hasattr(provider, "call_streaming")

    # Estimate fixed token overhead from tool schemas and system prompt
    tool_overhead = estimate_tool_tokens(tools)
    system_overhead = estimate_system_tokens(system)
    fixed_overhead = tool_overhead + system_overhead
    debug_log(f"fixed token overhead: ~{fixed_overhead:,} (tools: ~{tool_overhead:,}, system: ~{system_overhead:,})")

    for turn in range(1, max_turns + 1):
        # Compact context if approaching token limits (account for fixed overhead)
        maybe_compact(messages, total_in + fixed_overhead, provider, model=model)

        call_kwargs = {"model": model} if model else {}
        debug_request(messages, tools, model, system)
        if use_streaming:
            # call_streaming prints text tokens as they arrive
            resp = provider.call_streaming(messages=messages, tools=tools, system=system, **call_kwargs)
        else:
            resp = provider.call(messages=messages, tools=tools, system=system, **call_kwargs)
        debug_response(resp)
        total_in += resp.input_tokens
        total_out += resp.output_tokens

        # Print text (only needed in non-streaming mode; streaming already printed it)
        if not use_streaming and resp.text:
            print(resp.text)

        # If no tool calls, we're done
        if not resp.tool_calls:
            _print_usage(total_in, total_out, turn, fixed_overhead)
            return

        # Build assistant message
        assistant_content = _build_content(resp)
        messages.append({"role": "assistant", "content": assistant_content})

        # Execute each tool call
        tool_results = []
        for tc in resp.tool_calls:
            # Permission check for bash
            if tc.name == "bash" and not _is_safe_bash(tc.arguments.get("command", "")):
                if not _confirm_bash(tc.arguments["command"]):
                    tool_results.append(_tool_result(tc.id, "User denied this command."))
                    continue

            _print_tool_call(tc.name, tc.arguments)
            with debug_timer(f"tool:{tc.name}"):
                result = _execute_tool(tool_map, tc.name, tc.arguments)
            tool_results.append(_tool_result(tc.id, result))

        messages.append({"role": "user", "content": tool_results})

    print("\033[33m[max turns reached]\033[0m")
    _print_usage(total_in, total_out, max_turns, fixed_overhead)


def _execute_tool(tool_map: dict, name: str, arguments: dict) -> str:
    fn = tool_map.get(name)
    if not fn:
        return f"Error: unknown tool '{name}'"
    try:
        result = fn(**arguments)
        if len(result) > 30000:
            result = result[:15000] + "\n\n... (truncated) ...\n\n" + result[-15000:]
        return result
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        tb_short = "\n".join(tb.strip().splitlines()[-3:])
        return f"Error: {type(e).__name__}: {e}\n{tb_short}"


def _is_safe_bash(command: str) -> bool:
    """Check if a bash command is safe to run without confirmation.

    Splits on shell operators (;, &&, ||, backticks, $()) and pipes,
    then verifies every segment starts with a safe prefix. A single
    unsafe segment makes the whole command unsafe.
    """
    cmd = command.strip()
    if not cmd:
        return False

    # Reject commands containing subshells or command substitution outright —
    # these can nest arbitrarily and are not safe to parse with regex.
    if _SHELL_OPERATORS.search(cmd):
        return False

    # Split on pipes and check each segment
    segments = cmd.split("|")
    return all(_segment_is_safe(seg.strip()) for seg in segments)


def _segment_is_safe(segment: str) -> bool:
    """Check if a single command segment (no pipes/chains) is safe."""
    if not segment:
        return False
    return any(segment.startswith(p) for p in SAFE_BASH_PREFIXES)


def _confirm_bash(command: str) -> bool:
    print(f"\033[33m  bash: {command}\033[0m")
    try:
        answer = input("\033[33m  Allow? [y/N] \033[0m").strip().lower()
        return answer in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


def _build_content(resp: Response) -> list[dict]:
    content = []
    if resp.text:
        content.append({"type": "text", "text": resp.text})
    for tc in resp.tool_calls:
        content.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments})
    return content


def _tool_result(tool_use_id: str, content: str) -> dict:
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}


def _print_tool_call(name: str, args: dict) -> None:
    if name == "bash":
        print(f"\033[36m  > {args.get('command', '')}\033[0m")
    elif name == "read_file":
        print(f"\033[36m  > read {args.get('path', '')}\033[0m")
    elif name == "edit_file":
        print(f"\033[36m  > edit {args.get('path', '')}\033[0m")
    elif name == "write_file":
        print(f"\033[36m  > write {args.get('path', '')}\033[0m")
    elif name == "search":
        print(f"\033[36m  > search '{args.get('pattern', '')}' in {args.get('path', '.')}\033[0m")
    elif name == "list_files":
        print(f"\033[36m  > list '{args.get('pattern', '**/*')}' in {args.get('path', '.')}\033[0m")
    elif name == "list_directory":
        print(f"\033[36m  > ls {args.get('path', '.')}\033[0m")
    elif name in ("delegate", "delegate_parallel"):
        print(f"\033[36m  > {name}...\033[0m")
    else:
        print(f"\033[36m  > {name}\033[0m")


def _print_usage(input_tokens: int, output_tokens: int, turns: int,
                 fixed_overhead: int = 0) -> None:
    overhead_str = f" (~{fixed_overhead:,} fixed)" if fixed_overhead else ""
    print(f"\033[90m[{turns} turns | {input_tokens:,} in{overhead_str} + {output_tokens:,} out tokens]\033[0m",
          file=sys.stderr)
