"""The differential-test harness's pinned reader (round PY8, target DT1).

The fixtures in tests/fixtures/dt/ are minimised trial folders from the
engine bench (../engine-bench-1015/raw/trials/trials-jobs.tar.gz), kept as
JSON documents that map each file's path to its content: py-0.7.9, py-0.8
and Rust 0.8.2 ``hello-split`` and ``sidecar`` oracle trials, py-0.8's
``snapshot-hook`` refusal and Rust's ``acp_smoke`` R401. Host paths were
replaced by /tmp/dt/... when they were cut. The reader must turn each into
the canonical record below, and must import nothing from benchflow: the
engines' own readers are code under test.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

import pytest

from tools.dt import normalise

FIXTURES = Path(__file__).parent / "fixtures" / "dt"
HELLO = "Write the number 42 to `/app/answer.txt`."
SIDECAR = (
    "A service named `api` answers HTTP on port 8000 beside your container. "
    "Fetch `http://api:8000/token` and write the response body, exactly, to "
    "`/app/token.txt`."
)
SNAPSHOT = "Write the word `ready` to `/app/status.txt`, with no trailing newline."


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def materialise(name: str, root: Path) -> Path:
    """Write a fixture's files under ``root/<job>`` and return the job folder."""
    doc = json.loads((FIXTURES / f"{name}.json").read_text())
    job = root / doc["job_name"]
    for rel, content in doc["files"].items():
        path = job / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if rel.endswith(".json"):
            path.write_text(json.dumps(content, indent=2))
        else:
            path.write_text(content)
    return job


def _scripted_counts(prompts: int, steps: int, events: dict) -> dict:
    return {
        "n_prompts": prompts,
        "n_tool_calls": 0,
        "n_input_tokens": 0,
        "n_output_tokens": 0,
        "n_cache_read_tokens": 0,
        "n_cache_creation_tokens": 0,
        "total_tokens": 0,
        "trajectory_summary": {
            "agent_message_steps": 0,
            "agent_thought_steps": 0,
            "event_type_counts": events,
            "other_steps": steps,
            "partial_trajectory": False,
            "steps": steps,
            "tool_call_status_counts": {},
            "tool_call_steps": 0,
            "user_message_steps": 0,
        },
    }


def _verifier(reward_text: str | None) -> dict:
    return {
        "reward_text": reward_text,
        "reward_json": None,
        "stdout_sha256": _sha("") if reward_text is not None else None,
        "stderr_sha256": None,
        "ctrf": None,
    }


def _prompt(text: str | None, raw_suffix: str = "") -> dict:
    texts = [] if text is None else [text + raw_suffix]
    return {
        "n": len(texts),
        "raw_sha256": _sha("\x00".join(texts)),
        "stripped_sha256": _sha("\x00".join(t.strip() for t in texts)),
    }


def _summary(
    passed: int, errored: int, *, rust: bool = False, cats=None, unscored=0
) -> dict:
    counts = {
        "total": 1,
        "passed": passed,
        "failed": 0,
        "errored": errored,
        "verifier_errored": 0,
        "unscored": unscored,
    }
    if rust:
        causes = [["R401", "you", 1]] if errored else []
        return {
            "counts": counts,
            "raw": {"failures_by_cause": causes},
            "families": {"refusal": 1} if errored else {},
        }
    return {
        "counts": counts,
        "raw": {"error_categories": cats or {}, "verifier_error_categories": {}},
        "families": {"other": 1} if cats else {},
    }


EXPECTED = {
    "py-0.7.9__hello-split": {
        "engine": "python",
        "task": "hello-split",
        "agent": "oracle",
        "outcome": "passed",
        "reward": 1.0,
        "family": None,
        "code": None,
        "refusal": None,
        "verifier": _verifier("1"),
        "counts": _scripted_counts(1, 1, {"oracle": 1}),
        "prompt": _prompt(HELLO),
        # 0.7.9 writes no agent phase (../engine-bench-1015/trials.md).
        "timing": {"setup": 1.6, "agent": None, "verify": 2.7, "total": 12.6},
        "task_digest": "sha256:86745845afde974b9a0eadf4c14a8e250c02d984699d044077380a7a51c227e3",
        "skill_mode": "no-skill",
        "summary": _summary(1, 0, unscored=None),
    },
    "py-0.8__hello-split": {
        "engine": "python",
        "task": "hello-split",
        "agent": "oracle",
        "outcome": "passed",
        "reward": 1.0,
        "family": None,
        "code": None,
        "refusal": None,
        "verifier": _verifier("1"),
        "counts": _scripted_counts(1, 1, {"oracle": 1}),
        "prompt": _prompt(HELLO),
        "timing": {"setup": 1.6, "agent": 0.2, "verify": 3.6, "total": 12.5},
        "task_digest": "sha256:86745845afde974b9a0eadf4c14a8e250c02d984699d044077380a7a51c227e3",
        "skill_mode": "no-skill",
        "summary": _summary(1, 0),
    },
    "rust-0.8.2__hello-split": {
        "engine": "rust",
        "task": "hello-split",
        "agent": "oracle",
        "outcome": "passed",
        "reward": 1.0,
        "family": None,
        "code": None,
        "refusal": None,
        "verifier": _verifier("1"),
        "counts": _scripted_counts(1, 1, {"oracle": 1}),
        # Rust records instruction.md as it is; Python strips it
        # (src/benchflow/rollout/_setup.py:86): the raw hash differs, the
        # stripped one does not.
        "prompt": _prompt(HELLO, "\n"),
        "timing": {
            "setup": 2.156,
            "agent": 0.414630847,
            "verify": 0.932773792,
            "total": 3.6301356,
        },
        "task_digest": "sha256:a7af16f1c4b915ccc6c13d20ff0d96d79a10ff6acadcc3e88e33049e7fc7cfdb",
        "skill_mode": None,
        "summary": _summary(1, 0, rust=True),
    },
    "py-0.7.9__sidecar": {
        "engine": "python",
        "task": "sidecar",
        "agent": "oracle",
        "outcome": "passed",
        "reward": 1.0,
        "family": None,
        "code": None,
        "refusal": None,
        "verifier": _verifier("1"),
        "counts": _scripted_counts(1, 1, {"oracle": 1}),
        "prompt": _prompt(SIDECAR),
        "timing": {"setup": 3.5, "agent": None, "verify": 3.0, "total": 20.3},
        "task_digest": "sha256:7c79252011d76b50e93690027c7070562bee26ef547c36d4fe800bb5a8b9b50e",
        "skill_mode": "no-skill",
        "summary": _summary(1, 0, unscored=None),
    },
    "py-0.8__sidecar": {
        "engine": "python",
        "task": "sidecar",
        "agent": "oracle",
        "outcome": "passed",
        "reward": 1.0,
        "family": None,
        "code": None,
        "refusal": None,
        "verifier": _verifier("1"),
        "counts": _scripted_counts(1, 1, {"oracle": 1}),
        "prompt": _prompt(SIDECAR),
        "timing": {"setup": 3.3, "agent": 0.4, "verify": 3.8, "total": 19.9},
        "task_digest": "sha256:7c79252011d76b50e93690027c7070562bee26ef547c36d4fe800bb5a8b9b50e",
        "skill_mode": "no-skill",
        "summary": _summary(1, 0),
    },
    "rust-0.8.2__sidecar": {
        "engine": "rust",
        "task": "sidecar",
        "agent": "oracle",
        "outcome": "passed",
        "reward": 1.0,
        "family": None,
        "code": None,
        "refusal": None,
        "verifier": _verifier("1"),
        "counts": _scripted_counts(1, 1, {"oracle": 1}),
        "prompt": _prompt(SIDECAR, "\n"),
        "timing": {
            "setup": 8.385,
            "agent": 0.641138708,
            "verify": 1.008232767,
            "total": 10.403252909,
        },
        "task_digest": "sha256:a751e60eb5f3757ca4355123140e9300be2523fb9e0e927e867b9ca0092b0e8d",
        "skill_mode": None,
        "summary": _summary(1, 0, rust=True),
    },
    "py-0.8__snapshot-hook": {
        "engine": "python",
        "task": "snapshot-hook",
        "agent": "oracle",
        # No reward and an agent error: classify_result's "errored".
        "outcome": "errored",
        "reward": None,
        "family": "refusal",
        "code": "R401",
        "refusal": {"family": "refusal", "code": "R401", "paths": ["verifier.collect"]},
        "error_kind": "UnsupportedTaskFeatureError",
        "error_category": "other",
        "verifier": _verifier(None),
        "counts": _scripted_counts(1, 0, {}),
        "prompt": _prompt(SNAPSHOT),
        "timing": {"setup": None, "agent": None, "verify": None, "total": 0.0},
        "task_digest": "sha256:527e5bed873ec7d66cb1aa084f350e80270824d6c974ab9bbeb6c407eec79fdb",
        "skill_mode": "no-skill",
        "summary": _summary(0, 1, cats={"other": 1}),
    },
    "rust-0.8.2__acp_smoke": {
        "engine": "rust",
        "task": "acp_smoke",
        "agent": "oracle",
        "outcome": "errored",
        "reward": None,
        "family": "refusal",
        "code": "R401",
        "refusal": {
            "family": "refusal",
            "code": "R401",
            "paths": ["services.main.working_dir"],
        },
        "error_kind": "R401",
        "error_category": "unsupported_task_feature",
        "verifier": _verifier(None),
        "counts": {
            "n_prompts": 0,
            "n_tool_calls": 0,
            "n_input_tokens": 0,
            "n_output_tokens": 0,
            "n_cache_read_tokens": 0,
            "n_cache_creation_tokens": 0,
            "total_tokens": 0,
            "trajectory_summary": None,
        },
        "prompt": _prompt(None),
        "timing": {"setup": 0.0, "agent": 0.0, "verify": 0.0, "total": 0.00008594},
        "task_digest": "sha256:c268b4093a9ad98026d0dda86f46da5c34a741e0c07d2027d2c610fd2df76745",
        "skill_mode": None,
        "summary": _summary(0, 1, rust=True),
    },
}


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_fixture_folders_read_into_the_canonical_record(tmp_path, name):
    job = materialise(name, tmp_path)

    (record,) = normalise.normalise(job)

    # The summary's scoring fields and key set are checked below.
    scoring = record["summary"].pop("scoring")
    keys = record["summary"].pop("keys")
    assert isinstance(scoring, dict) and "job_name" not in keys
    assert not any(k.endswith("_time_sec") for k in keys) and "elapsed_sec" not in keys
    assert record == {
        "schema": normalise.SCHEMA,
        "error_kind": None,
        "verifier_error_kind": None,
        "error_category": None,
        "verifier_error_category": None,
        **EXPECTED[name],
    }


def test_python_outcome_is_recomputed_with_classify_results_precedence():
    """A pinned copy of src/benchflow/_utils/scoring.py:429-435 (fc2b9e9a)."""
    assert normalise.python_outcome(1.0, None, None) == "passed"
    assert normalise.python_outcome(0.75, None, None) == "failed"
    # An explicit reward wins over an error string.
    assert normalise.python_outcome(0.0, "warning", None) == "failed"
    assert normalise.python_outcome(None, "boom", "also") == "errored"
    assert normalise.python_outcome(None, None, "verifier failed") == "unscored"
    assert normalise.python_outcome(None, None, None) == "errored"


def test_a_changed_refusal_format_line_fails_loudly(tmp_path):
    """UnsupportedTaskFeature.format (runtime_capabilities.py:61-62) is pinned:
    a refusal whose lines no longer match is an error, never an accepted
    refusal with no paths."""
    job = materialise("py-0.8__snapshot-hook", tmp_path)
    result = next(job.glob("*/result.json"))
    data = json.loads(result.read_text())
    data["error"] = data["error"].replace(" (sandbox=docker)", " [docker]")
    result.write_text(json.dumps(data))

    with pytest.raises(normalise.RefusalFormatError, match="format"):
        normalise.normalise(job)


def test_a_changed_rust_refusal_message_fails_loudly(tmp_path):
    job = materialise("rust-0.8.2__acp_smoke", tmp_path)
    result = next(job.glob("*/result.json"))
    data = json.loads(result.read_text())
    data["problem"]["message"] = "acp_smoke is refused: services.main.working_dir"
    result.write_text(json.dumps(data))

    with pytest.raises(normalise.RefusalFormatError):
        normalise.normalise(job)


def test_an_unknown_category_is_an_error_not_other(tmp_path):
    job = materialise("py-0.8__snapshot-hook", tmp_path)
    result = next(job.glob("*/result.json"))
    data = json.loads(result.read_text())
    data["error"] = "SomethingNew: broke"
    data["error_category"] = "brand_new_category"
    result.write_text(json.dumps(data))

    with pytest.raises(normalise.NormaliseError, match=r"categories\.toml"):
        normalise.normalise(job)


def test_a_planner_refusal_with_no_trial_folder_is_refused_at_load():
    """Task.md packages the planner refuses (TaskMdError, taskmd/plan.py:931-936
    -> taskmd/materialize.py:133-138) write no trial folder; evaluation.py
    :1766-1776 logs "Skipping task" and goes on."""
    log = (
        "2026-10-16 08:00:00 WARNING Skipping task 'net-blocklist': net-blocklist: "
        "BenchFlow cannot honor these task.md draft 2 fields yet, so it refuses the task\n"
        "  - [agent] network: {'block': ['blocked.bench.invalid', '127.0.0.2/32']} differs "
        "from [sandbox] network ('open'); BenchFlow enforces the sandbox's network\n"
        "Job finished: 0 trials\n"
    )

    record = normalise.refused_at_load_record("net-blocklist", "oracle", log)

    assert record is not None
    assert record["outcome"] == "refused_at_load"
    assert record["refusal"] == {
        "family": "refusal",
        "code": "R401",
        "paths": ["agent.network"],
    }
    assert record["family"] == "refusal"
    assert normalise.refused_at_load_record("hello", "oracle", log) is None
    assert (
        normalise.refused_at_load_record("net-blocklist", "oracle", "no such line")
        is None
    )


def test_rust_refusal_fields_split_at_the_top_level_only():
    features = normalise.parse_rust_refusal(
        "codes asks for what this runner cannot do yet: [sandbox] clock (a fixed "
        "clock; not yet); environment/docker-compose.yaml services.main.working_dir "
        "(service main is the agent's container)"
    )
    assert [f["path"] for f in features] == [
        "sandbox.clock",
        "services.main.working_dir",
    ]


def test_raw_view_keeps_keys_and_drops_only_the_drop_table(tmp_path):
    job = materialise("py-0.8__hello-split", tmp_path)
    (trial,) = normalise.find_trials(job)

    raw = normalise.raw_view(trial)

    keys = raw["result.json"]["keys"]
    assert "rewards.reward" in keys and "agent_result.cost_usd" in keys
    # What DROP drops and the data-keyed maps are not in the key set.
    assert "started_at" not in keys and "timing.total" not in keys
    assert not any(k.startswith("trajectory_summary.event_type_counts") for k in keys)
    values = raw["result.json"]["values"]
    assert "started_at" not in values and "rollout_name" not in values
    assert values["agent_result.usage_source"] == "unavailable"
    assert raw["timing.json"]["keys"] == [
        "agent_execution",
        "environment_setup",
        "total",
        "verifier",
    ]


def test_every_drop_entry_names_its_reason():
    for field, reason in normalise.DROP.items():
        assert reason.strip(), field


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_the_reader_and_the_comparer_import_nothing_from_benchflow():
    """Followed through every tools.* module they import."""
    root = Path(__file__).parents[1]
    todo = [
        root / "tools" / "dt" / "normalise.py",
        root / "tools" / "dt" / "compare.py",
    ]
    seen: set[Path] = set()
    while todo:
        path = todo.pop()
        if path in seen:
            continue
        seen.add(path)
        for name in _imports(path):
            assert name.split(".")[0] != "benchflow", f"{path.name} imports {name}"
            if name.startswith("tools."):
                module = root / Path(*name.split("."))
                todo.append(
                    module.with_suffix(".py")
                    if module.with_suffix(".py").exists()
                    else module / "__init__.py"
                )
            if name == "tools.dt" or name == "tools.perf":
                todo.append(root / Path(*name.split(".")) / "__init__.py")
    assert root / "tools" / "perf" / "summarise.py" in seen


KEY_PATTERNS = re.compile(
    r"sk-[A-Za-z0-9_-]{16,}|AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{20,}|xox[abpr]-[A-Za-z0-9-]{10,}"
    r"|AIza[0-9A-Za-z_-]{30,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|Bearer [A-Za-z0-9._-]{20,}"
    r"|dtn_[A-Za-z0-9]{20,}|/home/|/Users/"
)


def test_fixtures_hold_no_key_and_no_home_path():
    files = sorted(FIXTURES.iterdir())
    assert files and all(f.suffix == ".json" for f in files)
    for path in files:
        assert not KEY_PATTERNS.search(path.read_text()), path.name


def test_verifier_stdout_is_hashed_without_run_to_run_noise(tmp_path):
    """Seen in the A/A run (results/DT1): a verifier's own pip install prints
    download rates, pytest prints object addresses and durations, `ls -l`
    prints clock times. None of it says anything about the engine."""

    def stdout_hash(text: str, name: str) -> str:
        trial = tmp_path / name
        (trial / "verifier").mkdir(parents=True)
        (trial / "verifier" / "test-stdout.txt").write_text(text)
        return normalise._verifier(trial)["stdout_sha256"]

    a = (
        "   1.3/1.3 MB 22.8 MB/s eta 0:00:00\n"
        "drwxr-xr-x 1 root root 4096 Oct 16 08:33 ..\n"
        "self = <test_outputs.T object at 0x76de09d67050>\n"
        "1 failed in 0.12s\n"
    )
    b = (
        "   1.3/1.3 MB 56.6 MB/s eta 0:00:01\n"
        "drwxr-xr-x 1 root root 4096 Oct 16 08:34 ..\n"
        "self = <test_outputs.T object at 0x763211eaecc0>\n"
        "1 failed in 0.31s\n"
    )
    assert stdout_hash(a, "a") == stdout_hash(b, "b")
    assert stdout_hash(a, "c") != stdout_hash(a.replace("1 failed", "1 passed"), "d")
