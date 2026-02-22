#!/usr/bin/env python3
"""CLI entry point — REPL and one-shot modes."""

from __future__ import annotations

import argparse
import sys

from agent import agent_loop
from prompts import MAIN_AGENT_SYSTEM
from providers import AnthropicProvider, OpenAIProvider, auto_detect_provider
from sub_agents import get_delegation_tools
from tools import TOOLS


def main():
    parser = argparse.ArgumentParser(description="Coding agent")
    parser.add_argument("prompt", nargs="?", help="One-shot prompt (omit for REPL)")
    parser.add_argument("--provider", choices=["anthropic", "openai"], default=None)
    parser.add_argument("--model", default=None, help="Model override")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL")
    args = parser.parse_args()

    # Init provider
    if args.provider == "anthropic":
        provider = AnthropicProvider()
    elif args.provider == "openai":
        provider = OpenAIProvider(base_url=args.base_url)
    else:
        provider = auto_detect_provider()

    model = args.model or ""

    # Assemble tools: base tools + delegation tools
    all_tools = TOOLS + get_delegation_tools(provider, model)

    if args.prompt:
        _one_shot(provider, all_tools, model, args.prompt)
    else:
        _repl(provider, all_tools, model)


def _one_shot(provider, tools, model, prompt):
    messages = [{"role": "user", "content": prompt}]
    agent_loop(provider, messages, tools, system=MAIN_AGENT_SYSTEM, model=model)


def _repl(provider, tools, model):
    print("\033[1mCoding Agent\033[0m (type 'quit' to exit)")
    messages = []

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
        agent_loop(provider, messages, tools, system=MAIN_AGENT_SYSTEM, model=model)


if __name__ == "__main__":
    main()
