"""Read a Python or Rust trial (or job) folder into one canonical record.

This is the differential-test harness's pinned reader. It imports nothing
from ``benchflow`` (a test checks it): the engines' own readers
(``load_trial``, ``bench eval inspect``, ``bench eval compare``) are code
under test and change this round, so a change in them can neither hide a
diff here nor invent one. Standard library only.

A record holds what both engines must agree on: the outcome, the reward,
the refusal as {family, code, paths}, the failure family, the parsed
verifier files, counts, two hashes of the recorded prompt, timing mapped to
setup/agent/verify/total, and the job's summary counts. Everything left out
on purpose is named in ``DROP`` with the reason; ``raw_view`` keeps the raw
key set and values of ``result.json`` and ``timing.json`` so that a hole in
this mapping cannot hide an added, removed or changed field.

    python3 -I tools/dt/normalise.py <trial, job or jobs folder>
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import sys
import tomllib
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

SCHEMA = "dt.record/1"
CATEGORIES_PATH = Path(__file__).with_name("categories.toml")

# Fields left out of the canonical record on purpose, each with its reason.
# Keys are dotted paths in result.json ("*" matches one segment) or file
# names (a whole file the reader never opens).
DROP: dict[str, str] = {
    "started_at": "a timestamp: differs on every run",
    "finished_at": "a timestamp: differs on every run",
    "rollout_name": "the trial id's random suffix (<task>__<8 hex>): differs on every run",
    "sandbox_id": "the backend's id for the sandbox: differs on every run",
    "timing": "result.json's copy of timing.json; timing is mapped from timing.json (setup/agent/verify/total)",
    "timing.*": "result.json's copy of timing.json (see timing)",
    "trajectory_source": "how the trajectory was captured (acp, scraped, partial); Python only, src/benchflow/rollout/__init__.py",
    "trajectory_summary.trajectory_source": "the same, inside trajectory_summary",
    "skill_mode": 'Rust hard-codes "no-skill" (integ/p1:crates/benchflow-run/src/record.rs:556); the record keeps Python\'s value only',
    "usage_limit_info": "a Python diagnostics block (src/benchflow/diagnostics.py, to_result_fields); Rust has none",
    "integration_failure_info": "a Python diagnostics block (src/benchflow/diagnostics.py)",
    "idle_timeout_info": "a Python diagnostics block (src/benchflow/diagnostics.py)",
    "agent_timeout_info": "a Python diagnostics block (src/benchflow/diagnostics.py)",
    "sandbox_startup_info": "a Python diagnostics block (src/benchflow/diagnostics.py)",
    "transport_error_info": "a Python diagnostics block (src/benchflow/diagnostics.py)",
    "verifier_timeout_info": "a Python diagnostics block (src/benchflow/diagnostics.py)",
    "api_error_info": "a Python diagnostics block (src/benchflow/diagnostics.py)",
    "suspected_api_error_info": "a Python diagnostics block (src/benchflow/diagnostics.py)",
    "scenes": "Python's echo of the run's scene and role layout (src/benchflow/scenes.py); Rust has none",
    "loop": "Python's echo of the loop strategy (src/benchflow/loop_strategies.py); Rust has none",
    "defenses": "Rust's per-defense states (integ/p1:crates/benchflow-run/src/outcome.rs); Python records none",
    "custody": "Rust's verifier custody record (custody.json); Python records none",
    "engine": "the engine's own name and version: differs between arms by design",
    "engine.*": "the engine's own name and version",
    "config.json": "the run's settings echo; it holds host paths and differs by design between engines",
    "results.jsonl": "a second copy of result.json rows; result.json is read",
    "lock.json": "Rust's job lock: host paths and fingerprints",
    "evaluation.json": "the job's settings echo: host paths",
    "custody.json": "Rust's verifier custody file (see custody)",
    "image.log": "the image build log: timestamps and layer ids",
    "summary.json:job_name": "the job folder's timestamp name",
    "summary.json:elapsed_sec": "wall time (timing)",
    "summary.json:*_time_sec": "wall time (timing)",
}

# result.json paths the canonical record is built from. The raw value diff
# skips them (the normalised diff already compares what they mean); every
# other path that is not in DROP is compared raw. A path is listed only when
# the record compares its whole meaning: of rewards only the
# scalar reward (other reward fields stay raw), of a Rust problem its code and
# the free text (message, detail, next: run paths and build logs), compared
# through the code and the refusal paths, so its owner, phase and transient
# flag stay raw. The error texts hold run paths and ids too: the record
# compares their kind (error_kind, verifier_error_kind) and categories.
COVERED = (
    "rewards",
    "rewards.reward",
    "reward",
    "error",
    "verifier_error",
    "n_prompts",
    "n_tool_calls",
    "agent_result.n_tool_calls",
    "agent_result.n_prompts",
    "agent_result.n_input_tokens",
    "agent_result.n_output_tokens",
    "agent_result.n_cache_read_tokens",
    "agent_result.n_cache_creation_tokens",
    "agent_result.total_tokens",
    "trajectory_summary",
    "trajectory_summary.*",
    "task_digest",
    "task_name",
    "agent",
    "status",
    "passed",
    "problem",
    "problem.code",
    "problem.message",
    "problem.detail",
    "problem.next",
)

# Maps whose keys are data (event names, tool-call statuses), not fields:
# a key there coming or going is a value change, compared in the record.
DATA_KEYED = (
    "trajectory_summary.event_type_counts",
    "trajectory_summary.tool_call_status_counts",
)

SCRIPTED_AGENTS = ("oracle", "nop")
SCRIPTED_PREFIXES = ("script:", "probe:")

# Python's refusal header (src/benchflow/task/runtime_capabilities.py,
# UnsupportedTaskFeatureError) and its per-feature line,
# UnsupportedTaskFeature.format(): "<path>: <reason> (sandbox=<sandbox>)".
# The recorded error is describe_exception(e), so a runtime refusal starts
# with the exception's name; an agent's own error text (an ACP error is
# "ACP error <code>: <the agent's message>") can hold the header anywhere
# else, and is never read as a refusal.
PY_REFUSAL_MARKER = "task uses parsed runtime features that benchflow cannot execute"
PY_REFUSAL_PREFIX = "UnsupportedTaskFeatureError: "
# error_category values a recorded refusal can carry: DT1's own category,
# and what classify_error gave it before (none, or "other").
PY_REFUSAL_CATEGORIES = (None, "other", "unsupported_task_feature")
PY_REFUSAL_LINE = re.compile(
    r"^- (?P<path>[^\s:][^:]*): (?P<reason>.+) \(sandbox=(?P<sandbox>[^()\s]+)\)$"
)
# Rust's R401 message (integ/p1:crates/benchflow-run/src/trial.rs:216-223):
# "<task> asks for what this runner cannot do yet: <field>; <field>".
RUST_REFUSAL = re.compile(
    r"^(?P<task>.+?) asks for what this runner cannot do yet: (?P<fields>.+)$", re.S
)
# The planner's refusal as evaluation.py logs it (src/benchflow/evaluation.py
# :1766-1776, TaskMdError's text from taskmd/materialize.py:62-69).
SKIPPING = re.compile(r"Skipping task '(?P<task>[^']+)': (?P<head>.*)$")
SKIPPING_REASON = re.compile(r"^\s+- (?P<field>[^:]+): (?P<detail>.*)$")

# Python's phases and T0's spans that do not overlap (tools/perf/README.md,
# "The spans they read"): setup is everything before the agent's own work.
PY_SETUP_KEYS = ("environment_setup", "install_agent", "agent_setup")
# Rust's phases (integ/p1:crates/benchflow-run/src/record.rs, timing): image,
# sandbox_start and setup are inside environment_setup.
RUST_SETUP_KEYS = ("environment_setup",)
STDOUT_SCRUB = (
    # Download progress a verifier's own package install prints (pip:
    # "1.3/1.3 MB 22.8 MB/s eta 0:00:00").
    (re.compile(r"\b\d+(?:\.\d+)?\s?[kMG]?i?B/s\b"), "<rate>"),
    (re.compile(r"\beta \d+:\d{2}:\d{2}\b"), "eta <t>"),
    # Object addresses in tracebacks ("<function exists at 0x73adef837d80>").
    (re.compile(r"\b0x[0-9a-f]{6,16}\b"), "0x<addr>"),
    # Clock times: `ls -l` ("Oct 16 08:33") and ISO stamps.
    (
        re.compile(
            r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) +\d{1,2} +\d{2}:\d{2}\b"
        ),
        "<date>",
    ),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?\S*"), "<date>"),
    # Test runners print durations ("1 passed in 0.12s", "Ran 3 tests in 0.001s").
    (re.compile(r"\b\d+(?:\.\d+)?\s?(?:s|ms|seconds?)\b"), "<t>"),
)


class NormaliseError(ValueError):
    """A folder the reader cannot turn into a record. Never read as a diff."""


class RefusalFormatError(NormaliseError):
    """A refusal whose text no longer has the shape this reader pins.

    Raised loudly, so a changed UnsupportedTaskFeature.format line (or Rust
    message) fails the run instead of reading as an accepted refusal with no
    paths.
    """


def load_categories(path: Path = CATEGORIES_PATH) -> dict[str, Any]:
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise NormaliseError(f"{path}: not readable JSON ({exc})") from None


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return None


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_scripted(agent: str | None) -> bool:
    agent = agent or ""
    return agent in SCRIPTED_AGENTS or agent.startswith(SCRIPTED_PREFIXES)


def python_outcome(
    reward: float | None, error: str | None, verifier_error: str | None
) -> str:
    """A pinned copy of classify_result's precedence.

    src/benchflow/_utils/scoring.py:429-435 at fc2b9e9a (BASE_T0). Python's
    result.json stores no outcome, so it is recomputed here from the reward;
    a change to classify_result in the code under test shows as
    ``reader_drift`` and ``summary_change``, never as a changed record.
    ``verifier_errored`` is this harness's ``unscored``.
    """
    if reward is not None:
        return "passed" if reward == 1.0 else "failed"
    if error:
        return "errored"
    if verifier_error:
        return "unscored"
    return "errored"


def rust_outcome(status: str | None, passed: Any, reward: float | None) -> str:
    """Rust's outcome from ``status`` (integ/p1:crates/benchflow-run/src/
    outcome.rs:166-176, record.rs:541-548)."""
    if status == "scored":
        if passed is None:
            return "passed" if reward == 1.0 else "failed"
        return "passed" if passed else "failed"
    if status in ("unscored", "errored", "skipped"):
        return status
    raise NormaliseError(f"unknown Rust status {status!r}")


def _family(categories: Mapping[str, Any], table: str, key: str | None) -> str | None:
    if key is None:
        return None
    engine, _, name = table.partition(".")
    mapping = categories.get(engine, {}).get(name, {})
    if key not in mapping:
        raise NormaliseError(
            f"{table} {key!r} is not in categories.toml: place it in a family first"
        )
    return mapping[key]


def is_python_refusal(error: str | None, error_category: str | None) -> bool:
    """Whether a Python result.json error is BenchFlow's runtime refusal.

    Only from its own envelope: the error opens with the header (after the
    exception's name, or bare), and its category is not some other
    failure's. The header anywhere else (inside an agent's ACP error
    message, say) is the agent's text, not a refusal.
    """
    if not error:
        return False
    first = error.split("\n", 1)[0].lower().removeprefix(PY_REFUSAL_PREFIX.lower())
    return (
        first.startswith(PY_REFUSAL_MARKER) and error_category in PY_REFUSAL_CATEGORIES
    )


def parse_python_refusal(error: str) -> list[dict[str, str]]:
    """The features of an UnsupportedTaskFeatureError's recorded text."""
    lines = error.splitlines()
    head = 0 if lines and PY_REFUSAL_MARKER in lines[0].lower() else None
    if head is None:
        raise RefusalFormatError(f"no refusal header in {error[:120]!r}")
    features = []
    for line in lines[head + 1 :]:
        if not line.strip():
            continue
        match = PY_REFUSAL_LINE.match(line)
        if match is None:
            raise RefusalFormatError(
                f"refusal line {line!r} does not match UnsupportedTaskFeature.format "
                '("- <path>: <reason> (sandbox=<sandbox>)"): the format changed, '
                "so update normalise.py and the ledger together"
            )
        features.append(match.groupdict())
    if not features:
        raise RefusalFormatError(
            f"refusal header with no feature lines: {error[:200]!r}"
        )
    return features


def _split_top_level(text: str, sep: str = "; ") -> list[str]:
    parts, depth, start, i = [], 0, 0, 0
    while i < len(text):
        ch = text[i]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        elif depth == 0 and text.startswith(sep, i):
            parts.append(text[start:i])
            i += len(sep)
            start = i
            continue
        i += 1
    parts.append(text[start:])
    return [p.strip() for p in parts if p.strip()]


def _field_path(head: str) -> str:
    """``[sandbox] clock`` -> ``sandbox.clock``; ``environment/docker-compose.yaml
    services.main.working_dir`` -> ``services.main.working_dir``."""
    head = head.strip()
    table = re.match(r"^\[\[?([^\]]+)\]\]?\s*(.*)$", head)
    if table:
        section, rest = table.group(1).strip(), table.group(2).strip()
        return f"{section}.{rest.split()[0]}" if rest else section
    return head.split()[-1] if head else head


def parse_rust_refusal(message: str) -> list[dict[str, str]]:
    match = RUST_REFUSAL.match(message.strip())
    if match is None:
        raise RefusalFormatError(
            f'R401 message {message[:160]!r} does not match "<task> asks for what '
            'this runner cannot do yet: <fields>"'
        )
    features = []
    for field in _split_top_level(match.group("fields")):
        head, _, reason = field.partition(" (")
        features.append({"path": _field_path(head), "reason": reason.rstrip(")")})
    return features


def _verifier(trial_dir: Path) -> dict[str, Any]:
    """The verifier files, as src/benchflow/jobs.py:107-127 reads them."""
    vdir = trial_dir / "verifier"
    reward = _read_text(vdir / "reward.txt")
    out: dict[str, Any] = {
        "reward_text": reward.strip() if reward is not None else None,
        "reward_json": _read_json(vdir / "reward.json"),
    }
    for name, file in (("stdout", "test-stdout.txt"), ("stderr", "test-stderr.txt")):
        text = _read_text(vdir / file)
        if text is not None:
            for pattern, repl in STDOUT_SCRUB:
                text = pattern.sub(repl, text)
        out[f"{name}_sha256"] = _sha(text) if text is not None else None
    ctrf = _read_json(vdir / "ctrf.json")
    if isinstance(ctrf, dict):
        tests = (ctrf.get("results") or {}).get("tests") or []
        out["ctrf"] = sorted(
            [str(t.get("name")), str(t.get("status"))]
            for t in tests
            if isinstance(t, dict)
        )
    else:
        out["ctrf"] = None
    return out


def _prompts(trial_dir: Path) -> dict[str, Any] | None:
    prompts = _read_json(trial_dir / "prompts.json")
    if prompts is None:
        return None
    if not isinstance(prompts, list):
        raise NormaliseError(f"{trial_dir}/prompts.json is not a list")
    texts = ["" if p is None else str(p) for p in prompts]
    return {
        "n": len(texts),
        "raw_sha256": _sha("\x00".join(texts)),
        "stripped_sha256": _sha("\x00".join(t.strip() for t in texts)),
    }


def _num(value: Any) -> float | None:
    return (
        float(value)
        if isinstance(value, int | float) and not isinstance(value, bool)
        else None
    )


def map_timing(
    timing: Mapping[str, Any] | None, engine: str
) -> dict[str, float | None]:
    """setup/agent/verify/total from timing.json (None for a phase not written)."""
    timing = timing or {}
    keys = PY_SETUP_KEYS if engine == "python" else RUST_SETUP_KEYS
    setup_parts = [_num(timing.get(k)) for k in keys]
    setup_known = [p for p in setup_parts if p is not None]
    return {
        "setup": round(sum(setup_known), 3) if setup_known else None,
        "agent": _num(timing.get("agent_execution")),
        "verify": _num(timing.get("verifier")),
        "total": _num(timing.get("total")),
    }


_RUST_CODE_HEAD = re.compile(r"^\[(R\d{3})\]")
_KIND_HEAD = re.compile(r"^([A-Za-z_][\w. -]{0,59}?):")


def error_kind(text: str | None) -> str | None:
    """What kind of error a recorded error text is, without its run-specific
    rest (paths, ids, build logs): Rust's "[R101] ..." is "R101", Python's
    "RuntimeError: ..." is "RuntimeError", "reward out of range: ..." is
    "reward out of range"; text with no such head is "text". None for no
    error."""
    if not text:
        return None
    first = text.strip().split("\n", 1)[0]
    code = _RUST_CODE_HEAD.match(first)
    if code:
        return code.group(1)
    head = _KIND_HEAD.match(first)
    return head.group(1).strip() if head else "text"


def _reward(data: Mapping[str, Any]) -> float | None:
    rewards = data.get("rewards")
    if isinstance(rewards, dict):
        return _num(rewards.get("reward"))
    return _num(data.get("reward"))


def engine_of(data: Mapping[str, Any]) -> str:
    engine = data.get("engine")
    if isinstance(engine, dict) and engine.get("name") == "benchflow-rs":
        return "rust"
    return "rust" if "status" in data and "problem" in data else "python"


def _counts(data: Mapping[str, Any], agent: str | None) -> dict[str, Any]:
    counts: dict[str, Any] = {
        "n_prompts": data.get("n_prompts"),
        "n_tool_calls": data.get("n_tool_calls"),
    }
    if is_scripted(agent):
        # Model agents' tokens and step counts vary run to run; a scripted
        # agent's are fixed, so a change there is a change in the engine.
        result = data.get("agent_result") or {}
        for key in (
            "n_input_tokens",
            "n_output_tokens",
            "n_cache_read_tokens",
            "n_cache_creation_tokens",
            "total_tokens",
        ):
            counts[key] = result.get(key)
        summary = data.get("trajectory_summary")
        if isinstance(summary, dict):
            counts["trajectory_summary"] = {
                k: v for k, v in sorted(summary.items()) if k != "trajectory_source"
            }
        else:
            counts["trajectory_summary"] = None
    return counts


def normalise_trial(
    trial_dir: str | Path, *, categories: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """One trial folder (holding result.json) as a canonical record."""
    trial_dir = Path(trial_dir)
    categories = categories if categories is not None else load_categories()
    data = _read_json(trial_dir / "result.json")
    if not isinstance(data, dict):
        raise NormaliseError(f"{trial_dir}: no result.json")
    engine = engine_of(data)
    agent = data.get("agent") or None
    reward = _reward(data)
    error = data.get("error") or None
    verifier_error = data.get("verifier_error") or None
    refusal = None
    code = None
    if engine == "python":
        outcome = python_outcome(reward, error, verifier_error)
        if error and is_python_refusal(error, data.get("error_category")):
            spec = categories["python"]["runtime_refusal"]
            features = parse_python_refusal(error)
            code = spec["code"]
            refusal = {
                "family": spec["family"],
                "code": code,
                "paths": sorted(f["path"] for f in features),
            }
            family = spec["family"]
        elif error:
            family = _family(
                categories,
                "python.error_category",
                data.get("error_category") or "other",
            )
        elif verifier_error and reward is None:
            family = _family(
                categories,
                "python.verifier_error_category",
                data.get("verifier_error_category") or "verifier_other",
            )
        else:
            family = None
    else:
        problem = data.get("problem") or None
        outcome = rust_outcome(data.get("status"), data.get("passed"), reward)
        code = problem.get("code") if isinstance(problem, dict) else None
        family = _family(categories, "rust.code", code) if code else None
        if code == "R401" and isinstance(problem, dict):
            features = parse_rust_refusal(str(problem.get("message") or ""))
            refusal = {
                "family": family,
                "code": code,
                "paths": sorted(f["path"] for f in features),
            }
    return {
        "schema": SCHEMA,
        "engine": engine,
        "task": data.get("task_name") or data.get("task_id") or "",
        "agent": agent,
        "outcome": outcome,
        "reward": reward,
        "family": family,
        "code": code,
        "refusal": refusal,
        # The failure's evidence beyond its family: the text's kind and the
        # recorded categories (the reader's error_category is checked
        # against the latter).
        "error_kind": error_kind(error),
        "verifier_error_kind": error_kind(verifier_error),
        "error_category": data.get("error_category"),
        "verifier_error_category": data.get("verifier_error_category"),
        "verifier": _verifier(trial_dir),
        "counts": _counts(data, agent),
        "prompt": _prompts(trial_dir),
        "timing": map_timing(_read_json(trial_dir / "timing.json"), engine),
        "task_digest": data.get("task_digest"),
        "skill_mode": data.get("skill_mode") if engine == "python" else None,
    }


def _dropped(path: str) -> bool:
    """Whether a result.json leaf is in DROP (a named field drops its subtree)."""
    for pattern in DROP:
        if ":" in pattern or "/" in pattern or pattern.endswith((".json", ".log")):
            continue
        if _matches(pattern, path) or path.startswith(f"{pattern}."):
            return True
    return False


def _matches(pattern: str, path: str) -> bool:
    p_parts, k_parts = pattern.split("."), path.split(".")
    if p_parts[-1] == "*":
        return len(k_parts) > len(p_parts) - 1 and all(
            a == b
            for a, b in zip(p_parts[:-1], k_parts[: len(p_parts) - 1], strict=True)
        )
    return p_parts == k_parts


def covered(path: str) -> bool:
    return any(_matches(pattern, path) for pattern in COVERED)


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """Dotted paths to leaves; a list is one leaf (its JSON)."""
    if isinstance(value, dict) and value:
        out: dict[str, Any] = {}
        for key, sub in value.items():
            out.update(flatten(sub, f"{prefix}.{key}" if prefix else str(key)))
        return out
    return {prefix: value}


def raw_view(trial_dir: str | Path) -> dict[str, Any]:
    """result.json's leaves (DROP left out) and both files' key sets and bytes."""
    trial_dir = Path(trial_dir)
    out: dict[str, Any] = {}
    for name in ("result.json", "timing.json"):
        path = trial_dir / name
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            out[name] = None
            continue
        data = json.loads(raw)
        leaves = flatten(data) if isinstance(data, dict) else {"": data}
        out[name] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            # The key set leaves out what DROP drops and the data-keyed maps.
            "keys": sorted(
                k
                for k in leaves
                if name != "result.json"
                or not (_dropped(k) or k.startswith(DATA_KEYED))
            ),
            "values": (
                {k: v for k, v in sorted(leaves.items()) if not _dropped(k)}
                if name == "result.json"
                else None
            ),
        }
    return out


# summary.json's scoring fields (top-level keys, fnmatch): the score, its
# ratios, the mean reward with its denominators (P4: mean_reward_n,
# mean_reward_scored_only*) and pass@k, compared by value.
SUMMARY_SCORING = (
    "score",
    "score_ratio",
    "score_interval",
    "score_excl_errors",
    "score_excl_errors_ratio",
    "mean_reward*",
    "pass_at_k*",
    "solve_rates",
)
# The ones both engines write with one meaning (a function of the counts):
# the cross-engine diff compares these only.
SUMMARY_SCORING_CROSS = ("score_ratio", "score_excl_errors_ratio")


def _summary_dropped(key: str) -> bool:
    return any(
        pattern.startswith("summary.json:")
        and fnmatch.fnmatchcase(key, pattern.split(":", 1)[1])
        for pattern in DROP
    )


def summary_counts(
    job_dir: str | Path, categories: Mapping[str, Any] | None = None
) -> dict[str, Any] | None:
    """summary.json's counts, raw and as families, its scoring fields, and
    its key set (DROP left out) for the raw fallback."""
    categories = categories if categories is not None else load_categories()
    path = Path(job_dir) / "summary.json"
    data = _read_json(path)
    if not isinstance(data, dict):
        if find_trials(job_dir):
            raise NormaliseError(f"{path}: completed job requires an object summary")
        return None
    counts = {
        key: data.get(key)
        for key in (
            "total",
            "passed",
            "failed",
            "errored",
            "verifier_errored",
            "unscored",
        )
    }
    families: Counter[str] = Counter()
    if (
        isinstance(data.get("engine"), dict)
        and data["engine"].get("name") == "benchflow-rs"
    ):
        for cause in data.get("failures_by_cause") or []:
            fam = _family(categories, "rust.code", cause.get("code"))
            families[str(fam)] += len(cause.get("trials") or [])
        raw = {
            # Each cause's owner too: who must act is part of the record.
            "failures_by_cause": sorted(
                [str(c.get("code")), str(c.get("owner")), len(c.get("trials") or [])]
                for c in data.get("failures_by_cause") or []
            )
        }
    else:
        for name, table in (
            ("error_categories", "python.error_category"),
            ("verifier_error_categories", "python.verifier_error_category"),
        ):
            for key, n in (data.get(name) or {}).items():
                families[str(_family(categories, table, key))] += int(n)
        raw = {
            "error_categories": data.get("error_categories") or {},
            "verifier_error_categories": data.get("verifier_error_categories") or {},
        }
    scoring = {
        key: value
        for key, value in sorted(
            flatten(
                {
                    k: v
                    for k, v in data.items()
                    if any(fnmatch.fnmatchcase(k, p) for p in SUMMARY_SCORING)
                }
            ).items()
        )
        if key
    }
    return {
        "counts": counts,
        "raw": raw,
        "families": dict(sorted(families.items())),
        "scoring": scoring,
        "keys": sorted(k for k in data if not _summary_dropped(k)),
    }


def find_trials(path: str | Path) -> list[Path]:
    """Trial folders under ``path``: itself, its children, or its grandchildren."""
    path = Path(path)
    if (path / "result.json").is_file():
        return [path]
    for depth in (1, 2):
        found = sorted(
            p.parent for p in path.glob("/".join(["*"] * depth) + "/result.json")
        )
        if found:
            return found
    return []


def final_attempt(trials: Iterable[Path]) -> Path:
    """The attempt a retried trial ended on: the one started last.

    Both engines retry some failures in a new trial folder (Python 0.8:
    RetryConfig, max_retries 2, src/benchflow/evaluation.py; Rust: a
    transient problem, lock.json "retries"); the job's summary counts the
    last attempt, and so does the harness.
    """
    trials = list(trials)
    if not trials:
        raise NormaliseError("no trial folder")

    def started(trial: Path) -> tuple[str, str]:
        data = _read_json(trial / "result.json") or {}
        return (str(data.get("started_at") or ""), trial.name)

    return max(trials, key=started)


def normalise(
    path: str | Path, *, categories: Mapping[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Every trial under ``path`` as a record, with its job's summary counts."""
    categories = categories if categories is not None else load_categories()
    records = []
    for trial in find_trials(path):
        record = normalise_trial(trial, categories=categories)
        record["summary"] = summary_counts(trial.parent, categories)
        records.append(record)
    return records


def refused_at_load_record(
    task: str,
    agent: str,
    log_text: str,
    *,
    categories: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """A requested Python cell with no trial folder: refused at load when the
    run's log names it in a "Skipping task" line, else None (a missing trial
    with no reason is the runner's error, never a refusal)."""
    categories = categories if categories is not None else load_categories()
    lines = log_text.splitlines()
    for i, line in enumerate(lines):
        match = SKIPPING.search(line)
        if match is None or match.group("task") != task:
            continue
        paths = []
        for follow in lines[i + 1 :]:
            reason = SKIPPING_REASON.match(follow)
            if reason is None:
                break
            paths.append(_field_path(reason.group("field")))
        spec = categories["python"]["refused_at_load"]
        return {
            "schema": SCHEMA,
            "engine": "python",
            "task": task,
            "agent": agent,
            "outcome": "refused_at_load",
            "reward": None,
            "family": spec["family"],
            "code": spec["code"],
            "refusal": {
                "family": spec["family"],
                "code": spec["code"],
                "paths": sorted(paths),
            },
            "error_kind": None,
            "verifier_error_kind": None,
            "error_category": None,
            "verifier_error_category": None,
            "verifier": None,
            "counts": None,
            "prompt": None,
            "timing": {"setup": None, "agent": None, "verify": None, "total": None},
            "task_digest": None,
            "skill_mode": None,
            "summary": None,
        }
    return None


def reader_projection(document: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Per trial, what ``bench eval inspect --json`` (src/benchflow/cli/
    results.py:95) says, including both error-ownership channels. A job
    document lists its trials; a trial document is one."""
    trials = document.get("trials")
    rows = trials if isinstance(trials, list) else [document]
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        task = str(row.get("task_name"))
        if task in out:
            # Each cell has one final trial. A dict must not silently turn
            # two exported rows into one apparently correct population.
            raise ValueError(f"duplicate reader trial for task {task!r}")
        outcome = row.get("score_outcome")
        out[task] = {
            "outcome": "unscored" if outcome == "verifier_errored" else outcome,
            "reward": _num(row.get("reward")),
            "error_category": row.get("error_category"),
            "verifier_error_category": row.get("verifier_error_category"),
        }
    return out


def _jsonable(records: Iterable[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(r, sort_keys=True) for r in records)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("path", type=Path)
    parser.add_argument("--raw", action="store_true", help="print raw_view too")
    args = parser.parse_args(argv)
    try:
        records = normalise(args.path)
        if args.raw:
            for record, trial in zip(records, find_trials(args.path), strict=True):
                record["raw"] = raw_view(trial)
    except NormaliseError as exc:
        print(f"normalise: {exc}", file=sys.stderr)
        return 2
    print(_jsonable(records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
