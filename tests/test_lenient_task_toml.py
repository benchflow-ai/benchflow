"""Lenient-but-safe task.toml on the run path.

Harbor lets its task schema grow, so BenchFlow does not reject *every* unknown
task.toml key (that stopped whole Harbor tasks on one new upstream field). But
leniency must not fail open: a typo of a known key, or an unknown key in a
table that decides isolation, network, timeouts, resources or grading, is
refused rather than silently dropped to a default. Genuinely-unknown keys
outside those tables — and known newer-Harbor keys BenchFlow does not enforce —
still load with a warning. ``TaskConfig.model_validate_toml`` and ``task.md``
stay strict. The run path (``Task`` / ``TaskRuntimeView``) and
``bench tasks check`` agree on what they refuse.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from typer.testing import CliRunner

from benchflow.task import Task, TaskConfig
from benchflow.task.imports import (
    LENIENT_OPT_OUT_ENV,
    TaskConfigKeyError,
    classify_unknown_key,
    refused_unknown_keys,
)
from benchflow.task.runtime_capabilities import validate_task_runtime_support
from benchflow.task.runtime_view import TaskRuntimeView

# Harbor's examples/tasks/hello-world/task.toml (harbor-framework/harbor
# 6cb9ff31, Apache-2.0), verbatim.
HARBOR_HELLO_WORLD = """\
schema_version = "1.4"

[task]
name = "harbor/hello-world"
version = "1.0.0"
authors = []
keywords = []

[metadata]
author_name = "Alex Shaw"
author_email = "author@example.com"
difficulty = "easy"
category = "programming"
tags = [ "trivial",]

[verifier]
timeout_sec = 120.0

[agent]
timeout_sec = 120.0

[environment]
build_timeout_sec = 600.0
cpus = 1
memory_mb = 2048
storage_mb = 10240
gpus = 0
mcp_servers = []

[verifier.env]

[solution.env]
"""

# A newer-Harbor top-level informational key: unknown, but outside every
# decision table, so it loads leniently. (BenchFlow already models the whole
# current Harbor sensitive surface, so this forward-compat shape — a new field
# outside the decision tables — is the realistic "should still load" case.)
TOP = 'future_top_level = "from a newer Harbor schema"\n'
WITH_SAFE_EXTRAS = TOP + HARBOR_HELLO_WORLD


def _task(root: Path, toml: str, *, name: str = "hello-world") -> Path:
    d = root / name
    (d / "environment").mkdir(parents=True)
    (d / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (d / "tests").mkdir()
    (d / "tests" / "test.sh").write_text(
        "#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n"
    )
    (d / "solution").mkdir()
    (d / "solution" / "solve.sh").write_text("#!/bin/bash\necho hi\n")
    (d / "instruction.md").write_text("Write hello.txt.\n")
    (d / "task.toml").write_text(toml)
    return d


# ── Safe unknown keys still load ──────────────────────────────────────────


def test_safe_unknown_keys_load_with_a_warning(tmp_path: Path, caplog) -> None:
    d = _task(tmp_path, WITH_SAFE_EXTRAS)
    with caplog.at_level(logging.WARNING):
        task = Task(d)
    assert task.name == "harbor/hello-world"
    assert task.config.sandbox.memory_mb == 2048
    assert task.config.ignored_keys == ("future_top_level",)
    text = caplog.text
    assert "future_top_level" in text
    assert "ignored" in text
    # Not a runtime issue on its own.
    assert validate_task_runtime_support(task.config, sandbox="docker") == []


def test_known_ignorable_key_in_a_sensitive_table_still_loads(
    tmp_path: Path, monkeypatch
) -> None:
    """The escape hatch for a newer-Harbor key BenchFlow ignores by choice.

    The allowlist is empty today (BenchFlow models the whole current Harbor
    sensitive surface), so exercise the mechanism with a seeded entry: a key on
    the list loads even though it sits in a decision table.
    """
    import benchflow.task.imports as imports_mod

    monkeypatch.setitem(
        imports_mod.KNOWN_IGNORABLE_KEYS, "sandbox.future_harbor_knob", "test"
    )
    assert classify_unknown_key("sandbox.future_harbor_knob") is None
    toml = HARBOR_HELLO_WORLD.replace(
        "[environment]\n", "[environment]\nfuture_harbor_knob = true\n"
    )
    task = Task(_task(tmp_path, toml))
    assert "sandbox.future_harbor_knob" in task.config.ignored_keys


# ── Typos are refused with a did-you-mean ─────────────────────────────────


@pytest.mark.parametrize(
    "bad, good",
    [
        ("cpus = 1", "cpuss = 1"),  # sandbox typo
    ],
)
def test_sandbox_typo_is_refused(tmp_path: Path, bad: str, good: str) -> None:
    d = _task(tmp_path, HARBOR_HELLO_WORLD.replace(bad, good))
    with pytest.raises(TaskConfigKeyError, match="did you mean 'cpus'"):
        Task(d)


def test_allow_internet_typo_is_refused_not_defaulted(tmp_path: Path) -> None:
    # The headline fail-open bug: 'allow_internett' silently ran on a public
    # network. It must now be refused with a did-you-mean.
    toml = HARBOR_HELLO_WORLD.replace(
        "[environment]\n", "[environment]\nallow_internett = false\n"
    )
    with pytest.raises(TaskConfigKeyError, match="did you mean 'allow_internet'"):
        Task(_task(tmp_path, toml))


def test_verifier_timeout_typo_is_refused_not_defaulted(tmp_path: Path) -> None:
    toml = HARBOR_HELLO_WORLD.replace(
        "timeout_sec = 120.0\n\n[agent]", "timout_sec = 1200.0\n\n[agent]"
    )
    with pytest.raises(TaskConfigKeyError, match="did you mean 'timeout_sec'"):
        Task(_task(tmp_path, toml))


def test_a_mistyped_table_name_is_refused(tmp_path: Path) -> None:
    toml = HARBOR_HELLO_WORLD.replace("[verifier]\n", "[verifer]\n")
    with pytest.raises(TaskConfigKeyError, match="did you mean 'verifier'"):
        Task(_task(tmp_path, toml))


# ── Unknown keys in decision tables are refused ───────────────────────────


def test_unknown_key_in_a_decision_table_is_refused(tmp_path: Path) -> None:
    toml = HARBOR_HELLO_WORLD + "\n[agent.future_knob]\nenabled = true\n"
    with pytest.raises(TaskConfigKeyError, match=r"\[agent\] table, which controls"):
        Task(_task(tmp_path, toml))


def test_unknown_sandbox_key_is_refused(tmp_path: Path) -> None:
    toml = HARBOR_HELLO_WORLD.replace(
        "[environment]\n", "[environment]\nturbo_mode = true\n"
    )
    with pytest.raises(TaskConfigKeyError, match=r"\[sandbox\] table, which controls"):
        Task(_task(tmp_path, toml))


# ── The opt-out restores full leniency ────────────────────────────────────


def test_opt_out_env_restores_full_leniency(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(LENIENT_OPT_OUT_ENV, "1")
    toml = HARBOR_HELLO_WORLD.replace(
        "[environment]\n", "[environment]\nallow_internett = false\n"
    )
    task = Task(_task(tmp_path, toml))  # would otherwise refuse the typo
    assert "sandbox.allow_internett" in task.config.ignored_keys
    # The typo did NOT flip the real field: it defaulted, which is exactly why
    # the opt-out is a deliberate, documented escape hatch.
    assert task.config.sandbox.allow_internet is True
    assert refused_unknown_keys(("sandbox.allow_internett",)) == []


# ── Strictness of the native parser is unchanged ──────────────────────────


def test_the_native_parser_stays_strict() -> None:
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        TaskConfig.model_validate_toml(WITH_SAFE_EXTRAS)


def test_a_wrong_type_still_fails(tmp_path: Path) -> None:
    d = _task(tmp_path, HARBOR_HELLO_WORLD.replace("cpus = 1", 'cpus = "one"'))
    with pytest.raises(ValueError):
        Task(d)


# ── Harbor verifier.collect keeps its own runtime refusal ─────────────────


def test_verifier_collect_is_refused_at_run_time_not_at_load(tmp_path: Path) -> None:
    toml = HARBOR_HELLO_WORLD + (
        '\n[[verifier.collect]]\nservice = "main"\ncommand = "cat /tmp/x"\n'
    )
    # Loads (its semantic refusal is the capability check's, with a clear reason).
    task = Task(_task(tmp_path, toml))
    issues = validate_task_runtime_support(task.config, sandbox="daytona")
    assert [i.path for i in issues] == ["verifier.collect"]
    assert "not executed" in issues[0].reason


# ── Run path and TaskRuntimeView agree ────────────────────────────────────


def test_runtime_view_reads_the_same_config(tmp_path: Path) -> None:
    view = TaskRuntimeView.from_task_dir(_task(tmp_path, WITH_SAFE_EXTRAS))
    assert view.config.ignored_keys == ("future_top_level",)


def test_runtime_view_refuses_the_same_typo(tmp_path: Path) -> None:
    toml = HARBOR_HELLO_WORLD.replace(
        "[environment]\n", "[environment]\nallow_internett = false\n"
    )
    with pytest.raises(TaskConfigKeyError):
        TaskRuntimeView.from_task_dir(_task(tmp_path, toml))


def test_the_warning_is_logged_once_per_file(tmp_path: Path, caplog) -> None:
    """A run loads a task more than once; the warning was printed each time."""
    d = _task(tmp_path, WITH_SAFE_EXTRAS)
    with caplog.at_level(logging.WARNING):
        Task(d)
        Task(d)
        TaskRuntimeView.from_task_dir(d)
    assert caplog.text.count("future_top_level") == 1


# ── bench tasks check agrees with the run path ────────────────────────────


def test_tasks_check_warns_but_passes_on_safe_keys(tmp_path: Path) -> None:
    from benchflow._utils.task_authoring import check_task, check_task_warnings
    from benchflow.cli.main import app

    d = _task(tmp_path, WITH_SAFE_EXTRAS)
    assert check_task(d) == []
    warnings = check_task_warnings(d)
    assert any("future_top_level" in w and "ignored" in w for w in warnings)
    result = CliRunner().invoke(app, ["tasks", "check", str(d)])
    assert result.exit_code == 0, result.output
    assert "warning" in result.output.lower() and "future_top_level" in result.output


def test_tasks_check_fails_on_a_typo(tmp_path: Path) -> None:
    from benchflow._utils.task_authoring import check_task

    toml = HARBOR_HELLO_WORLD.replace(
        "[environment]\n", "[environment]\nallow_internett = false\n"
    )
    d = _task(tmp_path, toml)
    issues = check_task(d)
    assert any("did you mean 'allow_internet'" in i for i in issues)


def test_tasks_check_names_the_unhonoured_key(tmp_path: Path) -> None:
    from benchflow._utils.task_authoring import check_task, check_task_warnings

    toml = HARBOR_HELLO_WORLD + '\n[[verifier.collect]]\ncommand = "true"\n'
    d = _task(tmp_path, toml)
    assert any(
        "verifier.collect" in w and "refused" in w for w in check_task_warnings(d)
    )
    issues = check_task(d, sandbox_type="docker")
    assert any("verifier.collect" in i for i in issues)
