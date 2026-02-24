# Local LLM Benchmark: Agentic Self-Improvement Loop

**Date**: 2026-02-23
**Setup**: 6-phase self-improvement loop (`self_improve.py`) running against a coding agent codebase (~20 files, Python). Each phase dispatches a scoped sub-agent via `SubAgentRunner` with phase-specific tools and system prompts. Local models served by Ollama on Apple Silicon.

## Models Tested

| Model | Parameters | Quantization | Context | Disk |
|-------|-----------|--------------|---------|------|
| Claude Sonnet (API) | undisclosed | native | 200k | API |
| qwen2.5:14b | 14.8B | Q4_K_M | 128k | 8.5 GB |
| glm-4.7-flash:q8_0 | 29.9B | Q8_0 | 128k | 30 GB |
| llama3.1:8b | 8.0B | Q4_K_M | 128k | 4.7 GB |
| mistral:7b | 7.2B | Q4_K_M | 32k | 4.1 GB |

## Task Structure

Each iteration runs 6 phases sequentially. Only the IMPLEMENT phase has write access; all others are read-only.

| Phase | Role | Purpose |
|-------|------|---------|
| EXPLORE | Explorer | Map unexplored files, find improvement opportunities |
| HYPOTHESIZE | Strategist | Generate 3-5 candidates across 7 axes, pick winner |
| IMPLEMENT | Coder | Apply the change, run tests before/after |
| RED-TEAM | Adversary | Try 3-5 attack strategies to break the change |
| AUDIT | Auditor | Score iteration on correctness/robustness/value/scope/tests |
| REFLECT | Strategist | Update exploration map, propose next-iteration seeds |

Budget per phase: 100k tokens / 15 turns (API), 20k tokens / 8 turns (local).

## Results

### Headline Numbers

| Metric | Claude Sonnet | qwen2.5:14b | glm-4.7-flash | llama3.1:8b | mistral:7b |
|--------|:------------:|:-----------:|:-------------:|:-----------:|:----------:|
| Phases completed | 6/6 | 6/6 | 6/6 | 6/6 | 2/6 |
| Total tokens | 670,003 | 31,714 | 115,073 | 10,088 | 5,689 |
| Wall time | ~110 min | ~10 min | ~40 min | ~2 min | ~1 min |
| Avg turns/phase | 12 | 2 | 4 | 1 | 1.5 |
| JSON parse success | 6/6 | 5/6 | 2/6 | 3/6 | 0/6 |
| Axis chosen | observability | architecture | i18n (invented) | architecture | N/A |
| Audit score | 3.4/10 | 6.6/10 | N/A | N/A | N/A |
| Red-team verdict | broken | N/A | N/A | broken | N/A |

### Capability Breakdown

#### Tool Use Quality

How well each model called tools (read_file, search, bash, edit_file) during agentic loops:

| Model | Calls tools | Correct args | Multi-turn chains | Writes code |
|-------|:-----------:|:------------:|:-----------------:|:-----------:|
| Claude Sonnet | Yes | Yes | Yes (10-15 turns) | Yes |
| qwen2.5:14b | Yes | Yes | Yes (1-3 turns) | Attempted |
| glm-4.7-flash | Yes | Mostly | Yes (5-8 turns) | Attempted |
| llama3.1:8b | Confused | Malformed | No | No |
| mistral:7b | Barely | Yes (1 call) | No | No |

- **Claude**: Exhaustive exploration — read 9+ files per phase, ran tests, made surgical edits.
- **qwen2.5:14b**: Competent tool use with correct argument schemas. Explored directories, read files, produced clean JSON. Best cost/quality ratio.
- **glm-4.7-flash**: Active tool user (hit budget limits on 3 phases) but verbose, burning tokens on long prose between tool calls.
- **llama3.1:8b**: Returned tool-call-shaped JSON as its text response instead of actually invoking tools. Classic small-model confusion between "describe doing X" and "do X."
- **mistral:7b**: Made one `list_directory` call in EXPLORE, then switched to pure prose for everything else.

#### Structured Output Compliance

Each phase asks for a specific JSON schema. Parse success rate:

| Model | explore | hypothesize | implement | red_team | audit | reflect |
|-------|:-------:|:-----------:|:---------:|:--------:|:-----:|:-------:|
| Claude Sonnet | Yes | Yes | Yes | Yes | Yes | Yes |
| qwen2.5:14b | Yes | Yes | Yes | No | Yes | Yes |
| glm-4.7-flash | No | Yes | No | No | No | Yes |
| llama3.1:8b | No | Yes | No | Yes | No | Yes |
| mistral:7b | No | No | - | - | - | - |

Failure modes:
- **glm-4.7-flash**: Generates structured content but buries it in verbose prose, or uses non-standard JSON formatting that the regex extractor misses.
- **llama3.1:8b**: Emits tool-call dicts (`{"name": "bash", "parameters": {...}}`) where content JSON is expected.
- **mistral:7b**: Ignores JSON formatting instructions entirely. Writes good prose analysis but zero fenced code blocks.

#### Hypothesis Diversity

The HYPOTHESIZE prompt enforces 7 improvement axes and says "do not default to error-handling." Results:

| Model | Hypotheses generated | Distinct axes | Invented axes |
|-------|:-------------------:|:-------------:|:-------------:|
| Claude Sonnet | 5 | 5 (obs, perf, ux, test, sec) | No |
| qwen2.5:14b | 4+ | 4 | No |
| glm-4.7-flash | 4 | 4 (i18n, docs, perf, sec) | Yes: i18n, docs |
| llama3.1:8b | 1 | 1 (architecture) | No |
| mistral:7b | 3 | 3 (ux, perf, obs) | No |

- GLM was the most creative, inventing "internationalization" and "documentation" as axes outside the prescribed 7.
- Mistral generated diverse hypotheses in prose but couldn't format them as JSON, so the loop couldn't proceed.

#### Self-Awareness / Reflection Quality

The REFLECT phase asks the model to honestly assess the iteration and plan ahead:

| Model | Self-critical | Actionable seeds | Trajectory awareness |
|-------|:------------:|:----------------:|:-------------------:|
| Claude Sonnet | Yes ("incomplete implementation") | 3, diverse | Yes |
| qwen2.5:14b | Moderate | 2, reasonable | Partial |
| glm-4.7-flash | Excellent ("absence of data is a signal") | 3, well-reasoned | Yes |
| llama3.1:8b | No | 3, generic | No |
| mistral:7b | N/A (aborted) | N/A | N/A |

GLM's reflection was the standout for local models: "the summary is sparse...no actual scores, strengths, or weaknesses are reported. This suggests either an incomplete audit or that no substantive findings were generated...the absence of data is a signal to iterate with more rigor." This is genuine metacognition about its own iteration's shortcomings.

## Failure Taxonomy

### 1. Tool-Call Confusion (llama3.1:8b)
The model returns a dict that looks like a tool call (`{"name": "bash", "parameters": {"command": "ls"}}`) as its text response, instead of actually invoking the tool. The sub-agent framework sees no tool_calls in the API response and treats the turn as "completed." The model thinks it acted; it didn't.

### 2. JSON Schema Blindness (mistral:7b)
The model understands the task and produces good analysis, but ignores the explicit JSON formatting requirement. It writes paragraphs where the prompt asks for fenced ```json blocks. The downstream parser gets nothing, and the loop aborts because it can't extract a hypothesis to implement.

### 3. Verbose Token Burn (glm-4.7-flash)
The model writes extensive reasoning between tool calls, consuming its token budget on prose instead of tool use. It hit `budget_exceeded` on 3/6 phases. The analysis is often good but gets truncated. At 30B Q8, each token is also ~3x slower to generate than the 14B models.

### 4. Hallucinated Execution (llama3.1:8b)
The RED-TEAM phase claimed to run attack scripts and report results, but the "commands_run" were never actually executed. The model fabricated plausible-looking test outputs. This is the most dangerous failure mode: it looks like the phase succeeded.

## Recommendations

### For local agentic workloads:
- **qwen2.5:14b** is the clear winner for cost/quality. Clean tool calls, reliable JSON, fast inference.
- **glm-4.7-flash** is worth trying if you have the VRAM and patience. Its reasoning is strong but needs a more tolerant JSON parser (e.g., LLM-based extraction fallback).
- **7B models are not viable** for multi-turn agentic loops with tool use. They can do single-turn generation but break down when asked to alternate between tool calls and structured output.

### For the framework:
- Add a JSON extraction fallback that re-prompts the model: "Your response did not contain valid JSON. Please output ONLY the JSON block." This would likely recover glm-4.7-flash and mistral:7b failures.
- Consider reducing system prompt length for smaller models. The current phase prompts are ~300-500 tokens each, which is a significant fraction of a 7B model's effective reasoning budget.
- Track tool-call-vs-text confusion as a metric. If a model's "text" response contains `{"name":` patterns, flag it as a tool-call confusion and retry.

## Raw Data

All phase JSONs, manifests, and iteration summaries are in `runs/` subdirectories:
```
runs/
  manifest.json
  <timestamp>_iter1/
    explore.json
    hypotheses.json
    implement.json
    red_team.json
    audit.json
    reflect.json
    iteration_summary.json
```
