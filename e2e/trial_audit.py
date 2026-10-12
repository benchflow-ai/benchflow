"""Audit of the trial folders a scenario leaves (standard library only).

`run.py` calls `audit_job_tree` after a scenario's steps pass; any problem
fails the scenario. A scenario that corrupts its jobs on purpose sets
`audit = false`. Each problem names the trial, the file, and what is wrong:

- complete JSON: BenchFlow's JSON and JSON-lines files parse (a last line
  without a newline is allowed; a cut one is not);
- ATIF: `trainer/atif.json` follows the ATIF model (no unknown keys,
  sequential step ids, agent-only fields on agent steps, observation results
  naming a tool call of their step, unique embedded trajectory ids);
- pairing: every tool call has exactly one result;
- tokens: ATIF's totals equal the trial's `agent_result` (prompt tokens with
  or without the cached ones), and `total_tokens` is the sum of its parts;
- prompt once: no document starts with the same user turn twice;
- tool output once: no tool call's content repeats its earlier output;
- keys: no known key pattern in any file the trial wrote;
- size: no derived file holds a string over 1 MiB without its cut marker.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

JSON_FILES = ["result.json", "config.json", "timing.json", "prompts.json", "custody.json", "trainer/atif.json"]
JSONL_FILES = ["trajectory/acp_trajectory.jsonl", "trainer/verifiers.jsonl", "trainer/adp.jsonl"]
DERIVED = ["trainer/atif.json", "trainer/adp.jsonl", "trainer/verifiers.jsonl", "agent/model_calls.jsonl"]
STRING_CAP = 1 << 20
CUT_MARKER = "[BenchFlow cut this text: "

TRAJ_KEYS = {"schema_version", "session_id", "trajectory_id", "agent", "steps", "notes", "final_metrics",
             "continued_trajectory_ref", "extra", "subagent_trajectories"}
AGENT_KEYS = {"name", "version", "model_name", "tool_definitions", "extra"}
STEP_KEYS = {"step_id", "timestamp", "source", "model_name", "reasoning_effort", "message", "reasoning_content",
             "tool_calls", "observation", "metrics", "is_copied_context", "llm_call_count", "extra"}
TOOL_KEYS = {"tool_call_id", "function_name", "arguments", "extra"}
RESULT_KEYS = {"source_call_id", "content", "subagent_trajectory_ref", "extra"}
METRIC_KEYS = {"total_prompt_tokens", "total_completion_tokens", "total_cached_tokens", "total_cost_usd",
               "total_steps", "extra"}
AGENT_ONLY = ("model_name", "reasoning_effort", "reasoning_content", "tool_calls", "metrics")

KEY_PATTERNS = [
    ("an Anthropic key", re.compile(rb"sk-ant-(?:api|oat|admin)\d\d-[A-Za-z0-9_\-]{20,}")),
    ("an OpenAI-style key", re.compile(rb"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_\-]{32,}")),
    ("a GitHub token", re.compile(rb"\b(?:ghp|gho|ghs|ghu|ghr)_[A-Za-z0-9]{36}\b|\bgithub_pat_[A-Za-z0-9_]{60,}")),
    ("a Google API key", re.compile(rb"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("an AWS access key", re.compile(rb"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("a Hugging Face token", re.compile(rb"\bhf_[A-Za-z0-9]{30,}\b")),
    ("a Daytona key", re.compile(rb"\bdtn_[A-Za-z0-9]{32,}\b")),
    ("a private key", re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----\s*[A-Za-z0-9+/=\r\n]{40,}")),
    ("a JWT", re.compile(rb"\beyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}")),
    ("a signed URL", re.compile(rb"[?&]sig=[A-Za-z0-9%+/=_\-]{20,}")),
]
# Placeholders tests and tasks use on purpose, judged on the match and the
# few bytes just before it (a real key beside a redacted one still counts).
FAKE = re.compile(rb"(?i)(x{6,}|fake|dummy|example|placeholder|test[-_]?key|your[-_]|0{12,})")
SKIP_DIRS = {"sessions", "node_modules", ".git"}
SCAN_MAX = 64 << 20


def trial_dirs(root: Path):
    """Trial folders under `root`: a result.json beside a config.json."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        if "result.json" in filenames and "config.json" in filenames:
            yield Path(dirpath)


def _jsonl(path: Path):
    """The parsed lines, or raise ValueError naming the first bad line."""
    data = path.read_bytes()
    lines = data.split(b"\n")
    out = []
    for i, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except ValueError as e:
            raise ValueError(f"line {i} is not JSON ({e})") from None
    return out


def _atif_rules(doc, where="atif") -> list[str]:
    errs = []
    if not isinstance(doc, dict):
        return [f"{where} is not an object"]
    if set(doc) - TRAJ_KEYS:
        errs.append(f"{where}: unknown keys {sorted(set(doc) - TRAJ_KEYS)}")
    agent = doc.get("agent")
    if not isinstance(agent, dict) or not isinstance(agent.get("name"), str) or not isinstance(agent.get("version"), str):
        errs.append(f"{where}.agent needs a name and a version")
    elif set(agent) - AGENT_KEYS:
        errs.append(f"{where}.agent: unknown keys {sorted(set(agent) - AGENT_KEYS)}")
    steps = doc.get("steps")
    if not isinstance(steps, list) or not steps:
        return errs + [f"{where}.steps is missing or empty"]
    for i, step in enumerate(steps):
        at = f"{where}.steps[{i}]"
        if not isinstance(step, dict):
            errs.append(f"{at} is not an object")
            continue
        if set(step) - STEP_KEYS:
            errs.append(f"{at}: unknown keys {sorted(set(step) - STEP_KEYS)}")
        if step.get("step_id") != i + 1:
            errs.append(f"{at}.step_id is {step.get('step_id')!r}, expected {i + 1}")
        if step.get("source") not in ("system", "user", "agent"):
            errs.append(f"{at}.source is {step.get('source')!r}")
        if "message" not in step:
            errs.append(f"{at}.message is missing")
        if step.get("source") != "agent":
            errs += [f"{at}.{k} is for agent steps only" for k in AGENT_ONLY if step.get(k) is not None]
        ids = set()
        for j, call in enumerate(step.get("tool_calls") or []):
            if not isinstance(call, dict) or set(call) - TOOL_KEYS or not isinstance(call.get("tool_call_id"), str) \
                    or not isinstance(call.get("function_name"), str) or not isinstance(call.get("arguments"), dict):
                errs.append(f"{at}.tool_calls[{j}] does not follow the model")
            else:
                ids.add(call["tool_call_id"])
        obs = step.get("observation")
        if obs is not None:
            for j, res in enumerate(obs.get("results") or [] if isinstance(obs, dict) else []):
                if not isinstance(res, dict) or set(res) - RESULT_KEYS:
                    errs.append(f"{at}.observation.results[{j}] does not follow the model")
                elif res.get("source_call_id") is not None and res["source_call_id"] not in ids:
                    errs.append(f"{at}.observation.results[{j}] names no tool call of its step")
    fm = doc.get("final_metrics")
    if isinstance(fm, dict):
        if set(fm) - METRIC_KEYS:
            errs.append(f"{where}.final_metrics: unknown keys {sorted(set(fm) - METRIC_KEYS)}")
        if isinstance(fm.get("total_steps"), int) and fm["total_steps"] != len(steps):
            errs.append(f"{where}.final_metrics.total_steps is {fm['total_steps']}, the document has {len(steps)}")
    seen = set()
    for j, sub in enumerate(doc.get("subagent_trajectories") or []):
        tid = sub.get("trajectory_id") if isinstance(sub, dict) else None
        if tid is None or tid in seen:
            errs.append(f"{where}.subagent_trajectories[{j}] needs a unique trajectory_id")
        seen.add(tid)
        errs += _atif_rules(sub, f"{where}.subagent_trajectories[{j}]")
    return errs


def _pairing(doc) -> list[str]:
    errs = []
    for step in doc.get("steps") or []:
        results = ((step.get("observation") or {}).get("results")) or []
        for call in step.get("tool_calls") or []:
            n = sum(1 for r in results if r.get("source_call_id") == call.get("tool_call_id"))
            if n != 1:
                errs.append(f"step {step.get('step_id')}: tool call {call.get('tool_call_id')} has {n} results")
    return errs


def _longest(value) -> str:
    if isinstance(value, str):
        return value
    items = value.values() if isinstance(value, dict) else value if isinstance(value, list) else []
    return max((_longest(v) for v in items), key=len, default="")


def _repeated_output(event) -> bool:
    texts = []
    for item in event.get("content") or []:
        inner = item.get("content") if isinstance(item, dict) else None
        text = (inner or {}).get("text") if isinstance(inner, dict) else (item.get("text") if isinstance(item, dict) else None)
        if isinstance(text, str) and len(text) >= 64:
            texts.append(text)
    return any(texts[i].startswith(texts[j]) for i in range(1, len(texts)) for j in range(i))


def audit_trial(trial: Path) -> list[str]:
    """Problems with one trial folder (empty when it is sound)."""
    problems = []
    docs = {}
    for rel in JSON_FILES:
        p = trial / rel
        if p.is_file():
            try:
                docs[rel] = json.loads(p.read_bytes())
            except ValueError as e:
                problems.append(f"{rel} is not complete JSON ({e})")
    for rel in JSONL_FILES + ["agent/model_calls.jsonl"]:
        p = trial / rel
        if p.is_file():
            try:
                docs[rel] = _jsonl(p)
            except ValueError as e:
                problems.append(f"{rel}: {e}")
    result = docs.get("result.json") if isinstance(docs.get("result.json"), dict) else {}
    atif = docs.get("trainer/atif.json")
    if isinstance(atif, dict):
        problems += [f"trainer/atif.json: {e}" for e in _atif_rules(atif)]
        problems += [f"trainer/atif.json: {e}" for e in _pairing(atif)]
        steps = atif.get("steps") or []
        if len(steps) >= 2 and steps[0].get("source") == steps[1].get("source") == "user" \
                and steps[0].get("message") == steps[1].get("message"):
            problems.append("trainer/atif.json starts with the same user turn twice")
        fm, ar = atif.get("final_metrics") or {}, result.get("agent_result") or {}
        cached = (ar.get("n_cache_read_tokens") or 0) + (ar.get("n_cache_creation_tokens") or 0)
        if "total_prompt_tokens" in fm and "n_input_tokens" in ar \
                and fm["total_prompt_tokens"] not in (ar["n_input_tokens"], ar["n_input_tokens"] + cached):
            problems.append(f"trainer/atif.json: total_prompt_tokens {fm['total_prompt_tokens']}, "
                            f"result.json n_input_tokens {ar['n_input_tokens']}")
        if "total_completion_tokens" in fm and "n_output_tokens" in ar and fm["total_completion_tokens"] != ar["n_output_tokens"]:
            problems.append(f"trainer/atif.json: total_completion_tokens {fm['total_completion_tokens']}, "
                            f"result.json n_output_tokens {ar['n_output_tokens']}")
    ar = result.get("agent_result") or {}
    parts = ("n_input_tokens", "n_output_tokens", "n_cache_read_tokens", "n_cache_creation_tokens")
    if all(isinstance(ar.get(k), int) for k in parts + ("total_tokens",)) and sum(ar[k] for k in parts) != ar["total_tokens"]:
        problems.append("result.json: total_tokens is not the sum of its parts")
    verifiers = docs.get("trainer/verifiers.jsonl") or []
    for rec in verifiers[:1]:
        prompt = rec.get("prompt") or []
        if len(prompt) >= 2 and prompt[0] == prompt[1]:
            problems.append("trainer/verifiers.jsonl starts with the same user turn twice")
    for i, event in enumerate(docs.get("trajectory/acp_trajectory.jsonl") or [], 1):
        if isinstance(event, dict) and event.get("type") == "tool_call" and _repeated_output(event):
            problems.append(f"trajectory/acp_trajectory.jsonl line {i}: the tool call's output is stored twice")
            break
    for rel in DERIVED:
        if rel in docs:
            text = _longest(docs[rel])
            if len(text.encode()) > STRING_CAP + 4096 and CUT_MARKER not in text[-300:]:
                problems.append(f"{rel} holds a {len(text) >> 20} MiB string without the cut marker")
    for dirpath, dirnames, filenames in os.walk(trial):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            p = Path(dirpath) / name
            if p.is_symlink() or not p.is_file():
                continue
            data = p.open("rb").read(SCAN_MAX)
            for what, pattern in KEY_PATTERNS:
                for m in pattern.finditer(data):
                    near = data[max(0, m.start() - 12): m.end()]
                    if not FAKE.search(near):
                        problems.append(f"{p.relative_to(trial)} holds {what} at byte {m.start()}")
                        break
    return problems


def audit_job_tree(root: Path) -> list[str]:
    """Problems of every trial folder under `root`, each with its trial."""
    out = []
    for trial in sorted(trial_dirs(root)):
        out += [f"{trial.relative_to(root)}: {p}" for p in audit_trial(trial)]
    return out
