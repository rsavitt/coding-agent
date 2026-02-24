#!/usr/bin/env python3
"""Creative self-improvement loop — 6-phase iteration with scoped sub-agents.

Phases:
  1. EXPLORE   — map unexplored files/subsystems (read-only)
  2. HYPOTHESIZE — generate improvement candidates across 7 axes (read-only)
  3. IMPLEMENT  — apply the chosen hypothesis (write access)
  4. RED-TEAM   — try to break the implementation (read-only + tests)
  5. AUDIT      — grade the iteration (read-only)
  6. REFLECT    — update exploration map, propose next seed (read-only)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from providers import AnthropicProvider, OpenAIProvider, auto_detect_provider
from sub_agents import SubAgentRunner, SubAgentResult
from tools import EXPLORER_TOOLS, CODER_TOOLS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMPROVEMENT_AXES = [
    "security",
    "performance",
    "architecture",
    "test_coverage",
    "ux_cli",
    "prompt_engineering",
    "observability",
]

RUNS_DIR = Path("runs")

# ---------------------------------------------------------------------------
# Phase system prompts
# ---------------------------------------------------------------------------

EXPLORE_SYSTEM = """\
You are an EXPLORER agent. Your job is to map unexplored parts of a codebase.

You will be given a list of files already examined. Focus on files and subsystems
NOT yet covered. Look at directory structure, read key files, and identify areas
that could benefit from improvement.

Rules:
- Do NOT modify any files.
- Use read_file, search, list_files, list_directory to explore.
- Only use bash for non-destructive commands (ls, wc, etc.). Do NOT write/edit.
- Return your findings as a JSON block:

```json
{
  "files_examined": ["path1", "path2"],
  "subsystems_found": [
    {"name": "...", "files": ["..."], "summary": "..."}
  ],
  "improvement_opportunities": [
    {"area": "...", "file": "...", "description": "...", "axis": "one of: security|performance|architecture|test_coverage|ux_cli|prompt_engineering|observability"}
  ]
}
```
"""

HYPOTHESIZE_SYSTEM = """\
You are a STRATEGIST agent. Given exploration findings, generate improvement
hypotheses across multiple axes and pick the best one.

IMPORTANT: You MUST consider ALL 7 improvement axes, not just error handling:
  1. security — input validation, injection, secrets, sandboxing
  2. performance — caching, batching, unnecessary work, algorithmic complexity
  3. architecture — separation of concerns, coupling, missing abstractions
  4. test_coverage — untested paths, missing edge cases, flaky tests
  5. ux_cli — help text, error messages, progress feedback, flags
  6. prompt_engineering — system prompt clarity, tool descriptions, few-shot examples
  7. observability — logging, metrics, tracing, debug output

Do NOT default to error-handling improvements. Aim for diversity.

Rules:
- Do NOT modify any files. Read-only exploration is allowed.
- Generate 3-5 hypotheses across DIFFERENT axes.
- Rank each by (estimated_value: 1-10) minus (estimated_risk: 1-10).
- Select the winner with the highest net score.
- Return as JSON:

```json
{
  "hypotheses": [
    {
      "id": "H1",
      "title": "...",
      "axis": "...",
      "description": "...",
      "files_to_change": ["..."],
      "estimated_value": 8,
      "estimated_risk": 2,
      "net_score": 6
    }
  ],
  "selected": "H1",
  "rationale": "..."
}
```
"""

IMPLEMENT_SYSTEM = """\
You are a CODER agent. Implement a specific improvement hypothesis.

Rules:
- Read files before editing.
- Use edit_file for surgical changes, write_file for new files.
- Run the test suite BEFORE making changes to establish baseline.
- Make the changes.
- Run the test suite AFTER changes to verify no regressions.
- Return as JSON:

```json
{
  "hypothesis_id": "H1",
  "files_changed": ["..."],
  "changes_summary": "...",
  "tests_before": {"passed": 0, "failed": 0, "errors": 0},
  "tests_after": {"passed": 0, "failed": 0, "errors": 0},
  "success": true
}
```
"""

RED_TEAM_SYSTEM = """\
You are an ADVERSARY agent. Your job is to try to break a recent code change.

Think like an attacker or a hostile user. Try 3-5 different attack strategies:
- Edge cases and boundary conditions
- Malformed or adversarial inputs
- Concurrency / race conditions (conceptual analysis)
- Dependency on external state that could change
- Regression in existing functionality

Rules:
- Do NOT modify source files.
- You CAN run tests and write temporary test scripts using bash.
- For each attack, describe the strategy, what you tried, and the result.
- Return as JSON:

```json
{
  "attacks": [
    {
      "strategy": "...",
      "description": "...",
      "commands_run": ["..."],
      "result": "pass|fail|partial",
      "details": "..."
    }
  ],
  "verdict": "holds|broken|partial",
  "critical_findings": ["..."]
}
```
"""

AUDIT_SYSTEM = """\
You are an AUDITOR agent. Grade a completed iteration of self-improvement.

Review the exploration findings, the hypothesis selected, the implementation,
and the red-team results. Then score the iteration.

Rules:
- Do NOT modify any files.
- You may read files to verify claims.
- Score each dimension 1-10:
  - correctness: Does the change work as intended?
  - robustness: Does it survive red-team attacks?
  - value: Is this a meaningful improvement?
  - scope: Was the change appropriately sized?
  - tests: Were tests adequate?
- Return as JSON:

```json
{
  "scores": {
    "correctness": 8,
    "robustness": 7,
    "value": 6,
    "scope": 8,
    "tests": 5
  },
  "overall_score": 6.8,
  "strengths": ["..."],
  "weaknesses": ["..."],
  "recommendations": ["..."]
}
```
"""

REFLECT_SYSTEM = """\
You are a STRATEGIST agent in reflection mode. Review the completed iteration
and plan the next one.

Rules:
- Do NOT modify any files.
- Analyze what was accomplished, what the audit found, and what red-team revealed.
- Update the exploration map: which areas are now well-covered?
- Suggest 2-3 seed ideas for the next iteration, preferring under-explored axes.
- Score the trajectory: is quality improving over iterations?
- Return as JSON:

```json
{
  "iteration_assessment": "...",
  "newly_covered_areas": ["..."],
  "axes_histogram_update": {"axis": "+1 or +0"},
  "next_iteration_seeds": [
    {"title": "...", "axis": "...", "rationale": "..."}
  ],
  "trajectory_score": 7,
  "trajectory_trend": "improving|stable|declining"
}
```
"""

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PhaseContext:
    """Accumulates outputs across the 6 phases of one iteration."""
    iteration: int = 0
    explore: dict = field(default_factory=dict)
    hypotheses: dict = field(default_factory=dict)
    implement: dict = field(default_factory=dict)
    red_team: dict = field(default_factory=dict)
    audit: dict = field(default_factory=dict)
    reflect: dict = field(default_factory=dict)
    phase_results: dict = field(default_factory=dict)  # phase_name -> SubAgentResult


@dataclass
class Manifest:
    """Persistent state across all iterations."""
    iterations_completed: int = 0
    files_examined: list = field(default_factory=list)
    audit_score_trend: list = field(default_factory=list)
    red_team_breaks: int = 0
    axes_touched: dict = field(default_factory=lambda: {a: 0 for a in IMPROVEMENT_AXES})
    capability_delta_log: list = field(default_factory=list)
    test_count_trend: list = field(default_factory=list)
    token_cost_per_iteration: list = field(default_factory=list)
    next_seeds: list = field(default_factory=list)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> Manifest:
        if not path.exists():
            return cls()
        with open(path) as f:
            data = json.load(f)
        m = cls()
        for k, v in data.items():
            if hasattr(m, k):
                setattr(m, k, v)
        return m


# ---------------------------------------------------------------------------
# JSON extraction from LLM output
# ---------------------------------------------------------------------------

def extract_json(text: str) -> dict:
    """Extract JSON from LLM output — try fenced code blocks first, then raw scan."""
    if not text:
        return {}

    # Try fenced code blocks: ```json ... ``` or ``` ... ```
    fence_pattern = r"```(?:json)?\s*\n(.*?)\n\s*```"
    matches = re.findall(fence_pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    # Try raw JSON scan: find first { ... } block
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    start = None
                    continue

    return {}


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------

def run_phase(
    provider,
    model: str,
    phase_name: str,
    system_prompt: str,
    task_prompt: str,
    tools: list[dict],
    max_turns: int = 15,
    max_tokens: int = 100_000,
) -> tuple[dict, SubAgentResult]:
    """Run a single phase as a scoped sub-agent. Returns (parsed_json, raw_result)."""
    print(f"\n{'='*60}")
    print(f"  Phase: {phase_name.upper()}")
    print(f"{'='*60}")

    runner = SubAgentRunner(
        provider=provider,
        tools=tools,
        system=system_prompt,
        model=model,
        max_turns=max_turns,
        max_tokens=max_tokens,
    )

    t0 = time.time()
    result = runner.run(task_prompt)
    elapsed = time.time() - t0

    print(f"  Status: {result.status} | Turns: {result.turns_used} | "
          f"Tokens: {result.input_tokens + result.output_tokens:,} | "
          f"Time: {elapsed:.1f}s")

    parsed = extract_json(result.summary) or {}
    return parsed, result


def build_explore_task(manifest: Manifest) -> str:
    already = manifest.files_examined[:200]  # cap to avoid prompt bloat
    seeds = manifest.next_seeds[-3:] if manifest.next_seeds else []
    return f"""\
Explore the codebase at the current working directory.

Files already examined (skip these):
{json.dumps(already, indent=2) if already else "(none — first iteration)"}

{"Suggested focus areas from previous iteration:" if seeds else ""}
{json.dumps(seeds, indent=2) if seeds else ""}

Map out subsystems, identify files worth examining, and find improvement opportunities
across all 7 axes: security, performance, architecture, test_coverage, ux_cli,
prompt_engineering, observability.
"""


def build_hypothesize_task(ctx: PhaseContext, manifest: Manifest) -> str:
    return f"""\
Based on the exploration findings below, generate 3-5 improvement hypotheses
across DIFFERENT axes. Do NOT repeat the same axis if possible.

Exploration findings:
{json.dumps(ctx.explore, indent=2)}

Axes already well-covered in previous iterations:
{json.dumps(manifest.axes_touched, indent=2)}

Previous audit scores: {manifest.audit_score_trend}

Pick the hypothesis with the best value-to-risk ratio that also explores an
under-represented axis.
"""


def build_implement_task(ctx: PhaseContext) -> str:
    selected_id = ctx.hypotheses.get("selected", "H1")
    hypotheses = ctx.hypotheses.get("hypotheses", [])
    selected = next((h for h in hypotheses if h.get("id") == selected_id), hypotheses[0] if hypotheses else {})
    return f"""\
Implement the following improvement hypothesis:

{json.dumps(selected, indent=2)}

Rationale for selection: {ctx.hypotheses.get("rationale", "N/A")}

Steps:
1. Run the test suite to establish a baseline count and pass rate.
2. Read the files that need changing.
3. Make the changes.
4. Run the test suite again to verify no regressions.
5. Report the results.
"""


def build_red_team_task(ctx: PhaseContext) -> str:
    return f"""\
A code change was just made. Try to break it.

What was changed:
{json.dumps(ctx.implement, indent=2)}

Hypothesis that was implemented:
{json.dumps(ctx.hypotheses.get("selected", "unknown"))}

Try 3-5 different attack strategies. You can run tests and write temporary scripts,
but do NOT modify the source files. Focus on:
- Edge cases and malformed inputs
- Paths the implementer may have missed
- Interactions with existing code
- Whether the tests actually cover the change
"""


def build_audit_task(ctx: PhaseContext) -> str:
    return f"""\
Audit this completed iteration of self-improvement.

Exploration findings:
{json.dumps(ctx.explore, indent=2)}

Hypothesis selected:
{json.dumps(ctx.hypotheses, indent=2)}

Implementation results:
{json.dumps(ctx.implement, indent=2)}

Red-team results:
{json.dumps(ctx.red_team, indent=2)}

Score the iteration on: correctness, robustness, value, scope, tests (each 1-10).
"""


def build_reflect_task(ctx: PhaseContext, manifest: Manifest) -> str:
    return f"""\
Reflect on the completed iteration and plan the next one.

Iteration {ctx.iteration} summary:
- Audit scores: {json.dumps(ctx.audit.get("scores", {}), indent=2)}
- Overall: {ctx.audit.get("overall_score", "N/A")}
- Red-team verdict: {ctx.red_team.get("verdict", "N/A")}
- Axis used: {_get_axis(ctx)}
- Audit strengths: {ctx.audit.get("strengths", [])}
- Audit weaknesses: {ctx.audit.get("weaknesses", [])}

Historical axes touched: {json.dumps(manifest.axes_touched, indent=2)}
Historical audit scores: {manifest.audit_score_trend}

Identify which areas are now well-covered and suggest 2-3 seeds for the next
iteration, preferring under-explored axes.
"""


def _get_axis(ctx: PhaseContext) -> str:
    """Extract the axis from the selected hypothesis."""
    selected_id = ctx.hypotheses.get("selected", "")
    for h in ctx.hypotheses.get("hypotheses", []):
        if h.get("id") == selected_id:
            return h.get("axis", "unknown")
    return "unknown"


# ---------------------------------------------------------------------------
# Iteration runner
# ---------------------------------------------------------------------------

def run_iteration(
    provider,
    model: str,
    manifest: Manifest,
    iteration_num: int,
    max_turns: int = 15,
    max_phase_tokens: int = 100_000,
) -> PhaseContext:
    """Run all 6 phases for one iteration."""
    ctx = PhaseContext(iteration=iteration_num)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    iter_dir = RUNS_DIR / f"{ts}_iter{iteration_num}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    total_in, total_out = 0, 0

    mt = max_turns
    mpt = max_phase_tokens

    # Phase 1: EXPLORE
    task = build_explore_task(manifest)
    ctx.explore, result = run_phase(provider, model, "explore", EXPLORE_SYSTEM, task, EXPLORER_TOOLS, max_turns=mt, max_tokens=mpt)
    ctx.phase_results["explore"] = result
    total_in += result.input_tokens
    total_out += result.output_tokens
    _save_phase(iter_dir / "explore.json", ctx.explore, result)

    # Update manifest with newly examined files
    for f in ctx.explore.get("files_examined", []):
        if f not in manifest.files_examined:
            manifest.files_examined.append(f)

    # Phase 2: HYPOTHESIZE
    task = build_hypothesize_task(ctx, manifest)
    ctx.hypotheses, result = run_phase(provider, model, "hypothesize", HYPOTHESIZE_SYSTEM, task, EXPLORER_TOOLS, max_turns=mt, max_tokens=mpt)
    ctx.phase_results["hypothesize"] = result
    total_in += result.input_tokens
    total_out += result.output_tokens
    _save_phase(iter_dir / "hypotheses.json", ctx.hypotheses, result)

    if not ctx.hypotheses.get("hypotheses"):
        print("\n  [WARN] No hypotheses generated. Skipping remaining phases.")
        _save_iteration_summary(iter_dir, ctx, total_in, total_out, aborted=True)
        return ctx

    # Phase 3: IMPLEMENT (write access — gets extra turns)
    impl_turns = min(mt + 10, 25) if mt < 25 else 25
    task = build_implement_task(ctx)
    ctx.implement, result = run_phase(provider, model, "implement", IMPLEMENT_SYSTEM, task, CODER_TOOLS, max_turns=impl_turns, max_tokens=mpt)
    ctx.phase_results["implement"] = result
    total_in += result.input_tokens
    total_out += result.output_tokens
    _save_phase(iter_dir / "implement.json", ctx.implement, result)

    # Phase 4: RED-TEAM (read-only + bash for tests)
    task = build_red_team_task(ctx)
    ctx.red_team, result = run_phase(provider, model, "red_team", RED_TEAM_SYSTEM, task, EXPLORER_TOOLS, max_turns=mt, max_tokens=mpt)
    ctx.phase_results["red_team"] = result
    total_in += result.input_tokens
    total_out += result.output_tokens
    _save_phase(iter_dir / "red_team.json", ctx.red_team, result)

    # Phase 5: AUDIT
    audit_turns = min(mt, 10)
    task = build_audit_task(ctx)
    ctx.audit, result = run_phase(provider, model, "audit", AUDIT_SYSTEM, task, EXPLORER_TOOLS, max_turns=audit_turns, max_tokens=mpt)
    ctx.phase_results["audit"] = result
    total_in += result.input_tokens
    total_out += result.output_tokens
    _save_phase(iter_dir / "audit.json", ctx.audit, result)

    # Phase 6: REFLECT
    reflect_turns = min(mt, 10)
    task = build_reflect_task(ctx, manifest)
    ctx.reflect, result = run_phase(provider, model, "reflect", REFLECT_SYSTEM, task, EXPLORER_TOOLS, max_turns=reflect_turns, max_tokens=mpt)
    ctx.phase_results["reflect"] = result
    total_in += result.input_tokens
    total_out += result.output_tokens
    _save_phase(iter_dir / "reflect.json", ctx.reflect, result)

    # Update manifest
    _update_manifest(manifest, ctx, total_in, total_out)
    manifest.save(RUNS_DIR / "manifest.json")

    # Save iteration summary
    _save_iteration_summary(iter_dir, ctx, total_in, total_out)

    return ctx


def _save_phase(path: Path, parsed: dict, result: SubAgentResult):
    """Save phase output to JSON file."""
    data = {
        "parsed": parsed,
        "status": result.status,
        "turns_used": result.turns_used,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "raw_summary": result.summary[:5000],  # cap for disk space
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _update_manifest(manifest: Manifest, ctx: PhaseContext, total_in: int, total_out: int):
    """Update manifest with iteration results."""
    manifest.iterations_completed += 1

    # Audit score
    overall = ctx.audit.get("overall_score")
    if overall is not None:
        manifest.audit_score_trend.append(overall)

    # Red-team breaks
    verdict = ctx.red_team.get("verdict", "")
    if verdict in ("broken", "partial"):
        manifest.red_team_breaks += 1

    # Axis touched
    axis = _get_axis(ctx)
    if axis in manifest.axes_touched:
        manifest.axes_touched[axis] += 1

    # Capability delta
    changes = ctx.implement.get("changes_summary", "")
    if changes:
        manifest.capability_delta_log.append({
            "iteration": ctx.iteration,
            "axis": axis,
            "summary": changes[:200],
        })

    # Test count
    tests_after = ctx.implement.get("tests_after", {})
    total_tests = tests_after.get("passed", 0) + tests_after.get("failed", 0) + tests_after.get("errors", 0)
    if total_tests > 0:
        manifest.test_count_trend.append(total_tests)

    # Token cost
    manifest.token_cost_per_iteration.append(total_in + total_out)

    # Next seeds from reflect
    seeds = ctx.reflect.get("next_iteration_seeds", [])
    manifest.next_seeds = seeds


def _save_iteration_summary(iter_dir: Path, ctx: PhaseContext, total_in: int, total_out: int, aborted: bool = False):
    """Save a rolled-up iteration summary."""
    summary = {
        "iteration": ctx.iteration,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "aborted": aborted,
        "axis": _get_axis(ctx),
        "audit_overall": ctx.audit.get("overall_score"),
        "audit_scores": ctx.audit.get("scores", {}),
        "red_team_verdict": ctx.red_team.get("verdict"),
        "files_changed": ctx.implement.get("files_changed", []),
        "total_tokens": total_in + total_out,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
    }
    with open(iter_dir / "iteration_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


# ---------------------------------------------------------------------------
# Human checkpoint
# ---------------------------------------------------------------------------

def print_iteration_summary(ctx: PhaseContext, manifest: Manifest):
    """Print a human-readable summary after an iteration."""
    print(f"\n{'='*60}")
    print(f"  ITERATION {ctx.iteration} COMPLETE")
    print(f"{'='*60}")

    axis = _get_axis(ctx)
    print(f"  Axis:        {axis}")
    print(f"  Audit score: {ctx.audit.get('overall_score', 'N/A')}")
    print(f"  Red-team:    {ctx.red_team.get('verdict', 'N/A')}")
    print(f"  Files changed: {ctx.implement.get('files_changed', [])}")

    tests_after = ctx.implement.get("tests_after", {})
    if tests_after:
        print(f"  Tests: {tests_after.get('passed', '?')} passed, "
              f"{tests_after.get('failed', '?')} failed, "
              f"{tests_after.get('errors', '?')} errors")

    print(f"\n  Cumulative stats:")
    print(f"    Iterations: {manifest.iterations_completed}")
    print(f"    Audit trend: {manifest.audit_score_trend}")
    print(f"    Red-team breaks: {manifest.red_team_breaks}")
    print(f"    Axes histogram: {json.dumps(manifest.axes_touched)}")

    strengths = ctx.audit.get("strengths", [])
    weaknesses = ctx.audit.get("weaknesses", [])
    if strengths:
        print(f"\n  Strengths: {', '.join(strengths[:3])}")
    if weaknesses:
        print(f"  Weaknesses: {', '.join(weaknesses[:3])}")

    seeds = ctx.reflect.get("next_iteration_seeds", [])
    if seeds:
        print(f"\n  Next seeds:")
        for s in seeds[:3]:
            print(f"    - [{s.get('axis', '?')}] {s.get('title', '?')}")


def human_checkpoint() -> str:
    """Wait for human input. Returns 'c' (continue), 's' (skip), 'q' (quit)."""
    print(f"\n{'─'*60}")
    print("  [c] Continue to next iteration")
    print("  [s] Skip (save and exit)")
    print("  [q] Quit (discard remaining)")
    print(f"{'─'*60}")
    while True:
        try:
            choice = input("  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "q"
        if choice in ("c", "s", "q"):
            return choice
        print("  Please enter c, s, or q.")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Creative self-improvement loop for a coding agent"
    )
    parser.add_argument("--max-iterations", type=int, default=10,
                        help="Maximum number of iterations (default: 10)")
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="Run without human checkpoints between iterations")
    parser.add_argument("--model", default="",
                        help="Model override (e.g. claude-opus-4-6, llama3.1:8b)")
    parser.add_argument("--provider", choices=["anthropic", "openai", "ollama"],
                        default=None, help="LLM provider (default: auto-detect)")
    parser.add_argument("--base-url", default=None,
                        help="OpenAI-compatible base URL (e.g. http://localhost:11434/v1)")
    parser.add_argument("--max-turns", type=int, default=None,
                        help="Max turns per phase (default: 15, lower for local models)")
    parser.add_argument("--max-phase-tokens", type=int, default=None,
                        help="Token budget per phase (default: 100000, lower for local models)")
    args = parser.parse_args()

    # Init provider
    if args.provider == "ollama":
        base_url = args.base_url or "http://localhost:11434/v1"
        provider = OpenAIProvider(api_key="ollama", base_url=base_url)
        if not args.model:
            args.model = "qwen2.5:14b"  # best tool-calling model available
    elif args.provider == "openai":
        provider = OpenAIProvider(base_url=args.base_url)
    elif args.provider == "anthropic":
        provider = AnthropicProvider()
    else:
        if args.base_url:
            provider = OpenAIProvider(base_url=args.base_url)
        else:
            provider = auto_detect_provider()
    model = args.model

    # Budget defaults — smaller for local models
    is_local = args.provider == "ollama" or args.base_url
    max_turns = args.max_turns or (8 if is_local else 15)
    max_phase_tokens = args.max_phase_tokens or (20_000 if is_local else 100_000)

    # Load or create manifest
    manifest_path = RUNS_DIR / "manifest.json"
    manifest = Manifest.load(manifest_path)

    print(f"Self-improvement loop starting")
    print(f"  Max iterations: {args.max_iterations}")
    print(f"  Checkpoint: {'off' if args.no_checkpoint else 'on'}")
    print(f"  Model: {model or '(provider default)'}")
    print(f"  Provider: {args.provider or 'auto'}")
    print(f"  Max turns/phase: {max_turns}")
    print(f"  Token budget/phase: {max_phase_tokens:,}")
    print(f"  Prior iterations: {manifest.iterations_completed}")
    print(f"  Runs dir: {RUNS_DIR.resolve()}")

    for i in range(1, args.max_iterations + 1):
        iteration_num = manifest.iterations_completed + 1
        print(f"\n{'#'*60}")
        print(f"  STARTING ITERATION {iteration_num}")
        print(f"{'#'*60}")

        try:
            ctx = run_iteration(provider, model, manifest, iteration_num,
                                max_turns=max_turns, max_phase_tokens=max_phase_tokens)
        except KeyboardInterrupt:
            print("\n\nInterrupted. Saving manifest...")
            manifest.save(manifest_path)
            sys.exit(1)
        except Exception as e:
            print(f"\n  [ERROR] Iteration failed: {e}")
            import traceback
            traceback.print_exc()
            manifest.save(manifest_path)
            continue

        print_iteration_summary(ctx, manifest)

        if not args.no_checkpoint and i < args.max_iterations:
            choice = human_checkpoint()
            if choice == "q":
                print("  Quitting.")
                break
            elif choice == "s":
                print("  Saving and exiting.")
                break

    manifest.save(manifest_path)
    print(f"\nDone. {manifest.iterations_completed} total iterations.")
    print(f"Manifest saved to {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
