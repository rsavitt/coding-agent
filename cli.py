#!/usr/bin/env python3
"""CLI entry point — REPL and one-shot modes."""

from __future__ import annotations

import argparse
import sys
import time

from agent import agent_loop
from prompts import MAIN_AGENT_SYSTEM
from providers import AnthropicProvider, OpenAIProvider, auto_detect_provider
from session import auto_save, list_sessions, load_session
from sub_agents import get_delegation_tools
from tools import TOOLS


def main():
    parser = argparse.ArgumentParser(description="Coding agent")
    parser.add_argument("prompt", nargs="?", help="One-shot prompt (omit for REPL)")
    parser.add_argument("--provider", choices=["anthropic", "openai"], default=None)
    parser.add_argument("--model", default=None, help="Model override")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    parser.add_argument("--resume", default=None, metavar="SESSION_ID",
                        help="Resume a previous session by ID or path")
    parser.add_argument("--history", action="store_true", help="List recent sessions")
    args = parser.parse_args()

    # List sessions and exit
    if args.history:
        _show_history()
        return

    # Init provider
    if args.provider == "anthropic":
        provider = AnthropicProvider()
    elif args.provider == "openai":
        provider = OpenAIProvider(base_url=args.base_url)
    else:
        provider = auto_detect_provider()

    model = args.model or ""
    stream = not args.no_stream

    # Assemble tools: base tools + delegation tools
    all_tools = TOOLS + get_delegation_tools(provider, model)

    if args.prompt:
        _one_shot(provider, all_tools, model, args.prompt, stream)
    else:
        _repl(provider, all_tools, model, stream, resume=args.resume)


def _one_shot(provider, tools, model, prompt, stream):
    messages = [{"role": "user", "content": prompt}]
    agent_loop(provider, messages, tools, system=MAIN_AGENT_SYSTEM, model=model, stream=stream)


def _repl(provider, tools, model, stream, resume=None):
    session_id = f"session-{int(time.time())}"

    # Resume previous session
    if resume:
        messages = load_session(resume)
        if messages:
            session_id = resume.replace(".jsonl", "").split("/")[-1]
            print(f"\033[1mCoding Agent\033[0m (resumed {session_id}, {len(messages)} messages)")
        else:
            print(f"\033[33mSession '{resume}' not found, starting fresh.\033[0m")
            messages = []
            print(f"\033[1mCoding Agent\033[0m (type 'quit' to exit)")
    else:
        messages = []
        print(f"\033[1mCoding Agent\033[0m (type 'quit' to exit)")

    print(f"\033[90m[session: {session_id}]\033[0m", file=sys.stderr)

    while True:
        try:
            user_input = input("\n\033[32m> \033[0m").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break

        messages.append({"role": "user", "content": user_input})
        print()
        agent_loop(provider, messages, tools, system=MAIN_AGENT_SYSTEM, model=model, stream=stream)

        # Auto-save after each interaction
        auto_save(messages, session_id)

    # Final save
    if messages:
        auto_save(messages, session_id)
        print(f"\033[90m[session saved: {session_id}]\033[0m", file=sys.stderr)


def _show_history():
    sessions = list_sessions()
    if not sessions:
        print("No saved sessions.")
        return
    print(f"{'ID':<30} {'Modified':<20} {'Preview'}")
    print("-" * 80)
    for s in sessions:
        mod = time.strftime("%Y-%m-%d %H:%M", time.localtime(s["modified"]))
        preview = s["preview"][:40] + "..." if len(s["preview"]) > 40 else s["preview"]
        print(f"{s['id']:<30} {mod:<20} {preview}")


if __name__ == "__main__":
    main()
