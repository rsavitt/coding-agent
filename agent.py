"""Core agent loop — tool execution, context management, permission checks."""

from __future__ import annotations

import sys

from providers import Response

# Commands considered safe to run without user confirmation
SAFE_BASH_PREFIXES = (
    "ls", "cat", "head", "tail", "find", "grep", "rg", "wc", "file", "which",
    "git status", "git diff", "git log", "git branch", "git show",
    "python -m pytest", "pytest", "python -m mypy", "mypy",
    "ruff", "black --check", "npm test", "npm run", "cargo test",
)


def agent_loop(provider, messages: list[dict], tools: list[dict], system: str,
               model: str = "", max_turns: int = 100) -> None:
    """Run the agent loop: call LLM, execute tools, repeat until done."""
    tool_map = {t["name"]: t["execute"] for t in tools}
    total_in, total_out = 0, 0

    for turn in range(1, max_turns + 1):
        call_kwargs = {"model": model} if model else {}
        resp = provider.call(messages=messages, tools=tools, system=system, **call_kwargs)
        total_in += resp.input_tokens
        total_out += resp.output_tokens

        # Print any text
        if resp.text:
            print(resp.text)

        # If no tool calls, we're done
        if not resp.tool_calls:
            _print_usage(total_in, total_out, turn)
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
            result = _execute_tool(tool_map, tc.name, tc.arguments)
            tool_results.append(_tool_result(tc.id, result))

        messages.append({"role": "user", "content": tool_results})

        # Context budget warning
        if total_in > 150_000:
            print(f"\033[33m[context: {total_in:,} input tokens used]\033[0m")

    print("\033[33m[max turns reached]\033[0m")
    _print_usage(total_in, total_out, max_turns)


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
        return f"Error: {type(e).__name__}: {e}"


def _is_safe_bash(command: str) -> bool:
    cmd = command.strip()
    return any(cmd.startswith(p) for p in SAFE_BASH_PREFIXES)


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
    elif name in ("delegate", "delegate_parallel"):
        print(f"\033[36m  > {name}...\033[0m")
    else:
        print(f"\033[36m  > {name}\033[0m")


def _print_usage(input_tokens: int, output_tokens: int, turns: int) -> None:
    print(f"\033[90m[{turns} turns | {input_tokens:,} in + {output_tokens:,} out tokens]\033[0m",
          file=sys.stderr)
