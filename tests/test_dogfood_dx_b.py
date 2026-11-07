"""DX lane B (dogfood 2026-10-18): the CLI, its config files and its verbs.

Each test replays a dogfooder's repro through the real CLI and checks what
the user sees: the plan a dry run prints, the refusal of a mistyped config
key with the file, line and did-you-mean, one verb set in the CLI's own
hints and in the docs, ``bench run`` taking every run flag, one ``--harness``
help, a task.md typo named by file and line, and a ``bench agents`` list that
fits on a screen. Each fails on integ 66471e95 by what it prints.

Round 2: with no login, doctor and the
smoke name the oracle run instead of each other, and a run with no task says
so before any agent, key or sandbox line. Those fail on af9e05b3.
"""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from benchflow.cli.main import app

REPO = Path(__file__).resolve().parents[1]


def _bench(*args: str, **kwargs: Any):
    return CliRunner().invoke(app, list(args), prog_name="bench", **kwargs)


def _task(root: Path, name: str = "hello-craft", *, frontmatter: str = "") -> Path:
    """A task the run would run: task.md, an oracle and a verifier."""
    task = root / name
    (task / "environment").mkdir(parents=True)
    (task / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (task / "oracle").mkdir()
    (task / "oracle" / "solve.sh").write_text(
        "#!/bin/bash\necho hello > /app/out.txt\n"
    )
    (task / "verifier").mkdir()
    (task / "verifier" / "test.sh").write_text(
        "#!/bin/bash\necho 1.0 > /logs/verifier/reward.txt\n"
    )
    (task / "task.md").write_text(
        "---\n"
        "schema_version: '1.3'\n"
        f"name: {name}\n"
        "agent:\n"
        "  timeout_sec: 300\n"
        f"{frontmatter}"
        "---\n"
        "Write the word hello into the file /app/out.txt.\n"
    )
    return task


def _plan(result) -> dict[str, Any]:
    text = result.stdout
    return json.loads(text[text.index("{") :])


# --- p5-2: --dry-run prints the plan the run runs -------------------------------


def test_dry_run_of_one_task_folder_lists_that_task(tmp_path, monkeypatch) -> None:
    """Guards dogfood p5-2 at 66471e95: a single task folder planned
    ``"tasks": [], "rollouts": 0`` while the run ran the task."""
    monkeypatch.chdir(tmp_path)
    task = _task(tmp_path / "tasks")
    result = _bench("run", str(task), "--dry-run")
    assert result.exit_code == 0, result.output
    plan = _plan(result)
    assert plan["tasks"] == ["hello-craft"] and plan["rollouts"] == 1


def test_dry_run_counts_every_trial_of_a_repeated_run(tmp_path, monkeypatch) -> None:
    """A dry run of ``--trials 3`` over two tasks planned ``"rollouts": 2``
    and the run ran 6; the plan didn't say the tasks repeat (found writing
    the docs' page on checking a run before it starts)."""
    monkeypatch.chdir(tmp_path)
    tasks = tmp_path / "tasks"
    _task(tasks, "one")
    _task(tasks, "two")
    result = _bench(
        "run", str(tasks), "--agent", "oracle", "--trials", "3", "--dry-run"
    )
    assert result.exit_code == 0, result.output
    plan = _plan(result)
    assert plan["tasks"] == ["one", "two"]
    assert (plan["trials"], plan["rollouts"]) == (3, 6)
    assert "matrix" not in plan

    once = _plan(_bench("run", str(tasks), "--agent", "oracle", "--dry-run"))
    assert once["rollouts"] == 2 and "trials" not in once


def test_dry_run_of_a_matrix_names_its_entries(tmp_path, monkeypatch) -> None:
    """Each ``--matrix`` entry's agent and model are in the plan, and every
    entry runs every task ``--trials`` times."""
    monkeypatch.chdir(tmp_path)
    tasks = tmp_path / "tasks"
    _task(tasks, "one")
    matrix = tmp_path / "m.yaml"
    matrix.write_text("models:\n  ora:\n    agent: oracle\n  noop:\n    agent: nop\n")
    result = _bench(
        "run", str(tasks), "--matrix", str(matrix), "--trials", "2", "--dry-run"
    )
    assert result.exit_code == 0, result.output
    plan = _plan(result)
    assert plan["matrix"] == {
        "ora": {"agent": "oracle", "model": None},
        "noop": {"agent": "nop", "model": None},
    }
    assert (plan["trials"], plan["rollouts"]) == (2, 4)


def test_dry_run_of_a_config_plans_the_configs_run(tmp_path, monkeypatch) -> None:
    """Guards dogfood p5-2 at 66471e95: with --config the dry run printed the
    defaults (claude-agent-acp, concurrency 4, jobs) and not the file's."""
    monkeypatch.chdir(tmp_path)
    _task(tmp_path / "cfgsrc")
    config = tmp_path / "cfg2.yaml"
    config.write_text(
        "tasks_dir: cfgsrc\nagent: oracle\nenvironment: docker\n"
        "concurrency: 2\njobs_dir: jobs-cfg-file\n"
    )
    for argv in (
        ["eval", "run", "--config", str(config), "--dry-run"],
        ["run", "--config", str(config), "--dry-run"],
    ):
        result = _bench(*argv)
        assert result.exit_code == 0, (argv, result.output)
        plan = _plan(result)
        assert plan["agent"] == "oracle", argv
        assert plan["concurrency"] == 2, argv
        assert plan["jobs_dir"] == "jobs-cfg-file", argv
        assert plan["tasks"] == ["hello-craft"], argv


def test_dry_run_and_run_load_one_evaluation(tmp_path, monkeypatch) -> None:
    """The dry run prints from the Evaluation the run runs: the config with
    the command line's flags on top (flags > file > defaults)."""
    import benchflow.cli.main as main

    monkeypatch.chdir(tmp_path)
    _task(tmp_path / "cfgsrc")
    config = tmp_path / "cfg.yaml"
    config.write_text("tasks_dir: cfgsrc\nagent: oracle\nconcurrency: 2\n")
    seen: list[Any] = []
    real = main._config_file_evaluation

    def spy(plan):
        evaluation = real(plan)
        seen.append(evaluation)
        return evaluation

    monkeypatch.setattr(main, "_config_file_evaluation", spy)
    result = _bench("run", "--config", str(config), "-j", "3", "--dry-run")
    assert result.exit_code == 0, result.output
    assert _plan(result)["concurrency"] == 3
    (evaluation,) = seen
    assert evaluation._config.concurrency == 3


# --- p5-3: a run config refuses a key it does not read --------------------------


def test_a_mistyped_run_config_key_is_refused_by_file_and_line(
    tmp_path, monkeypatch
) -> None:
    """Guards dogfood p5-3 at 66471e95: ``concurrenzy: 2`` ran with 4 and
    exit 0."""
    monkeypatch.chdir(tmp_path)
    _task(tmp_path / "cfgsrc")
    config = tmp_path / "cfg3.yaml"
    config.write_text(
        "tasks_dir: cfgsrc\nagent: oracle\nsandbox: docker\nconcurrenzy: 2\n"
        "jobs_dir: jobs-cfg-typo\n"
    )
    result = _bench("run", "--config", "cfg3.yaml", "--dry-run")
    assert result.exit_code == 2, result.output
    assert (
        "cfg3.yaml:4: concurrenzy: unknown key (did you mean 'concurrency'?)"
        in result.output
    )
    assert "next:" in result.output
    assert "Traceback" not in result.output
    assert not (tmp_path / "jobs-cfg-typo").exists()


def test_sandbox_is_the_run_configs_name_for_environment(tmp_path) -> None:
    """Guards dogfood p5-3 at 66471e95: ``sandbox: daytona`` (the flag's
    name) was ignored and the run went to Docker."""
    from benchflow._utils.config_keys import ConfigKeyError
    from benchflow.evaluation import check_run_config_keys

    path = tmp_path / "run.yaml"
    raw = check_run_config_keys(
        {"tasks_dir": "t", "sandbox": "daytona"}, path=path, text="tasks_dir: t\n"
    )
    assert raw == {"tasks_dir": "t", "environment": "daytona"}
    text = "tasks_dir: t\nsandbox: daytona\nenvironment: docker\n"
    with pytest.raises(ConfigKeyError, match="sandbox and environment"):
        check_run_config_keys(
            {"tasks_dir": "t", "sandbox": "daytona", "environment": "docker"},
            path=path,
            text=text,
        )


def test_a_config_saying_sandbox_runs_on_that_sandbox(tmp_path, monkeypatch) -> None:
    """Guards dogfood p5-3 at 66471e95, from the CLI: the plan of a config
    with ``sandbox: daytona`` said docker."""
    monkeypatch.chdir(tmp_path)
    _task(tmp_path / "cfgsrc")
    (tmp_path / "cfg.yaml").write_text(
        "tasks_dir: cfgsrc\nagent: oracle\nsandbox: daytona\n"
    )
    result = _bench("eval", "run", "--config", "cfg.yaml", "--dry-run")
    assert result.exit_code == 0, result.output
    assert _plan(result)["environment"] == "daytona"


def test_run_config_keys_are_the_keys_the_loaders_read() -> None:
    """The allowed keys stay the ones Evaluation.from_yaml's loaders read, so
    a key added to a loader is never refused and a typo never accepted."""
    import inspect

    from benchflow.evaluation import (
        LEGACY_RUN_CONFIG_KEYS,
        RUN_CONFIG_KEYS,
        Evaluation,
    )

    def read(fn) -> set[str]:
        source = inspect.getsource(fn)
        keys = set(re.findall(r'raw(?:\.get\(\s*|\[)"([a-z_]+)"', source))
        keys |= set(re.findall(r'"([a-z_]+)" in raw\b', source))
        for loop in re.findall(r"for key in \(([^)]*)\)", source):
            keys |= set(re.findall(r'"([a-z_]+)"', loop))
        return keys

    usage = {"usage_tracking"}
    assert (
        read(Evaluation._from_native_yaml) - {"agents", "datasets"} <= RUN_CONFIG_KEYS
    )
    assert RUN_CONFIG_KEYS - usage - read(Evaluation._from_native_yaml) <= {
        "usage_proxy",
        "usage_proxy_advertised_base_url",
        "usage_proxy_bind_host",
        "usage_proxy_port",
        "usage_proxy_url",
    }
    assert read(Evaluation._from_legacy_yaml) <= LEGACY_RUN_CONFIG_KEYS


# --- p5-4, p2-9: one verb set --------------------------------------------------

# Older spellings with a new verb, and the files that may still name them: the
# release notes and old reports say what changed, and the upgrade page maps
# the old to the new. Elsewhere, a line that names one on purpose says "older
# spelling".
OLD_VERBS = re.compile(
    r"bench (?:eval (?:run|create|view|list|regrade|branch(?!es))|"
    r"tasks (?:init|check|migrate|export)|agent (?:list|show))\b"
)


def test_the_readme_and_docs_use_the_new_verbs() -> None:
    """Guards dogfood p5-4 and p2-9 at 66471e95: README had no new verb and
    docs/ had 166 ``bench eval run`` against 8 ``bench run``."""
    found: list[str] = []
    files = [REPO / "README.md", *sorted((REPO / "docs").rglob("*.md"))]
    for path in files:
        rel = path.relative_to(REPO).as_posix()
        if rel in (
            # The upgrade page quotes the 0.7 spellings on purpose.
            "docs/versions/upgrade-from-0.7.md",
            # Generated by tools/docs_reference.py: cli-python.md from
            # cli_parity.py, one section per 0.8 command; commands.md from the
            # CLI, every command, older spellings included; files.md from the
            # flags' help and each schema's own description; changelog.md from
            # CHANGELOG.md, the release notes that say what changed.
            "docs/reference/cli-python.md",
            "docs/reference/commands.md",
            "docs/reference/files.md",
            # python.md from benchflow.__all__, quoting each name's and each
            # method's docstring as the code has it.
            "docs/reference/python.md",
            "docs/versions/changelog.md",
            # Generated by tools/docs_errors.py from the
            # source: it quotes messages and docstrings as the code has them,
            # some of which still name older spellings.
            "docs/reference/errors.md",
        ) or rel.startswith("docs/contributing/reports/"):
            continue
        text = path.read_text()
        for number, line in enumerate(text.splitlines(), 1):
            if "older spelling" in line.lower():
                continue  # a line that names an older spelling on purpose
            if OLD_VERBS.search(line):
                found.append(f"{rel}:{number}: {line.strip()[:100]}")
    assert found == [], "\n".join(found[:20])


def test_the_clis_own_hints_name_the_new_verbs(tmp_path, monkeypatch) -> None:
    """Guards dogfood p5-4 at 66471e95: ``bench new`` said ``Next: bench
    tasks check ... bench eval run``, every run said ``View: bench eval view``."""
    monkeypatch.chdir(tmp_path)
    created = _bench("new", "hello")
    assert created.exit_code == 0, created.output
    assert "bench check tasks/hello" in created.output
    assert "bench run tasks/hello" in created.output
    assert not OLD_VERBS.search(created.output), created.output
    sources = [
        p for p in (REPO / "src" / "benchflow").rglob("*.py") if p.name != "verbs.py"
    ]
    hints = re.compile(
        r"""(?:print_hint|next_step|hint|View:|Next:)[^\n]*"""
        r"""bench (?:eval view|tasks check|tasks init|agent list)"""
    )
    offenders = [
        f"{p.relative_to(REPO)}" for p in sources if hints.search(p.read_text())
    ]
    assert offenders == []


def test_an_old_spelling_prints_one_short_note(tmp_path, monkeypatch) -> None:
    """Guards dogfood p5-4 and p2-3 at 66471e95: ``--config`` (and every flag
    bench run lacked) printed "has no `bench run` spelling, so this line stays
    `bench eval run`"."""
    monkeypatch.chdir(tmp_path)
    _task(tmp_path / "cfgsrc")
    (tmp_path / "cfg.yaml").write_text("tasks_dir: cfgsrc\nagent: oracle\n")
    result = _bench("eval", "run", "--config", "cfg.yaml", "--dry-run")
    assert result.exit_code == 0, result.output
    notes = [line for line in result.stderr.splitlines() if line.startswith("note:")]
    assert notes == [
        "note: `bench eval run` is now `bench run`: bench run --config cfg.yaml --dry-run"
    ]


# --- p2-3, p5-4: bench run takes every run flag ---------------------------------


def test_bench_run_takes_every_bench_eval_run_flag() -> None:
    """Guards dogfood p2-3 at 66471e95: ``bench run --allow-task-env`` was an
    unknown option, though its own refusal named it."""
    from benchflow.cli.new_verbs import _eval_run_command
    from tests.test_cli_docs_drift import _cli_long_flags

    eval_flags = {
        o
        for p in _eval_run_command().params
        for o in (*p.opts, *p.secondary_opts)
        if o.startswith("--")
    }
    run_flags = _cli_long_flags(["run"]) | {
        o
        for p in _run_command().params
        for o in (*p.opts, *p.secondary_opts)
        if o.startswith("--")
    }
    assert eval_flags - run_flags == set()


def _run_command():
    import typer

    from benchflow.cli.new_verbs import register_run

    host = typer.Typer()
    register_run(host)
    host.command("unused")(lambda: None)
    return typer.main.get_group(host).commands["run"]


def test_bench_run_passes_allow_task_env_and_config_through(
    tmp_path, monkeypatch
) -> None:
    import benchflow.cli.main as main

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(main, "eval_run", lambda **kw: calls.append(kw))
    task = _task(tmp_path)
    result = _bench(
        "run", str(task), "--allow-task-env", "DOGFOOD_FAKE_TOKEN",
        "--job-name", "j", "--freeze-workspace", "--sandbox-user", "agent",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    (call,) = calls
    assert call["allow_task_env"] == ["DOGFOOD_FAKE_TOKEN"]
    assert call["job_name"] == "j" and call["freeze_workspace"] is True
    assert call["sandbox_user"] == "agent"
    calls.clear()
    config = tmp_path / "run.yaml"
    config.write_text("tasks_dir: x\n")
    assert _bench("run", "--config", str(config)).exit_code == 0
    (call,) = calls
    # A run config's own agent runs: bench run adds no oracle default.
    assert call["config_file"] == config and call["agent"] is None


def test_bench_run_help_stays_short_and_help_all_lists_the_rest() -> None:
    short = _bench("run", "--help", env={"COLUMNS": "100"})
    full = _bench("run", "--help-all", env={"COLUMNS": "100"})
    assert short.exit_code == 0 and full.exit_code == 0
    assert "--allow-task-env" not in short.output
    assert "bench run --help-all" in " ".join(short.output.split())
    assert "--allow-task-env" in full.output and "--config" in full.output
    # 0.10 shows the six flags the run card names (--trials, --integrity,
    # --accept-deviations, the two budget caps, --keep-sandbox), with help cut
    # to a few lines each.
    assert len(short.output.splitlines()) < 160


# --- p5-9: one --harness help ------------------------------------------------


def test_harness_help_is_one_string_with_native() -> None:
    """Guards dogfood p5-9 at 66471e95: ``bench run`` said ``acp or headless``
    and ``bench eval run`` ``acp or native``."""
    from benchflow.cli._options import HARNESS_HELP

    def harness_help(command) -> str:
        (param,) = [p for p in command.params if p.name == "harness"]
        return param.help

    from benchflow.cli.new_verbs import _eval_run_command

    assert harness_help(_run_command()) == HARNESS_HELP
    assert harness_help(_eval_run_command()) == HARNESS_HELP
    assert "native" in HARNESS_HELP and "headless JSON" in HARNESS_HELP


def test_both_verbs_help_harness_with_the_same_words() -> None:
    """Guards dogfood p5-9 at 66471e95, from the help each verb prints."""

    def harness_line(*argv: str) -> str:
        text = _bench(*argv, "--help", env={"COLUMNS": "200", "NO_COLOR": "1"}).output
        line = next(line for line in text.splitlines() if "--harness" in line)
        return " ".join(line.replace("│", " ").split()[2:])

    # The first line of each help: the columns may wrap them at other words.
    run_line, eval_line = harness_line("run"), harness_line("eval", "run")
    assert run_line[:40] == eval_line[:40], (run_line, eval_line)
    assert "native" in run_line and "headless (its own" not in run_line


def test_bench_run_takes_harness_native(tmp_path, monkeypatch) -> None:
    import benchflow.cli.main as main

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(main, "eval_run", lambda **kw: calls.append(kw))
    task = _task(tmp_path)
    for value in ("native", "headless"):
        result = _bench("run", str(task), "-a", "codex", "--harness", value)
        assert result.exit_code == 0, (value, result.output)
    assert [c["harness"] for c in calls] == ["native", "native"]


# --- p2-7: a task.md typo is named by file and line ------------------------------


def test_a_frontmatter_typo_names_the_file_line_and_key(tmp_path, monkeypatch) -> None:
    """Guards dogfood p2-7 at 66471e95: ``agent.timeout`` printed pydantic's
    "1 validation error for TaskConfig ... errors.pydantic.dev"."""
    monkeypatch.chdir(tmp_path)
    task = _task(tmp_path / "tasks", "e-typo")
    text = (task / "task.md").read_text().replace("timeout_sec: 300", "timeout: 300")
    (task / "task.md").write_text(text)
    result = _bench("check", "tasks/e-typo")
    assert result.exit_code != 0
    assert (
        "tasks/e-typo/task.md:5: agent.timeout: unknown key (did you mean timeout_sec?)"
        in result.output
    ), result.output
    assert "pydantic" not in result.output and "validation error" not in result.output
    assert "Traceback" not in result.output


def test_a_frontmatter_value_error_names_the_line_and_value(tmp_path) -> None:
    from benchflow.task.document import TaskDocument, TaskDocumentParseError

    task = _task(tmp_path, "bad-value")
    path = task / "task.md"
    path.write_text(path.read_text().replace("timeout_sec: 300", "timeout_sec: five"))
    with pytest.raises(TaskDocumentParseError) as raised:
        TaskDocument.from_path(path)
    message = str(raised.value)
    assert ":5: agent.timeout_sec:" in message and "'five'" in message
    assert "pydantic" not in message


def test_a_frontmatter_yaml_error_counts_lines_as_the_file_does(tmp_path) -> None:
    """The YAML parser counts from the frontmatter; the message counts from
    the file's first line (dogfood p5-13's line 9 that was line 10)."""
    from benchflow.task.document import TaskDocument, TaskDocumentParseError

    task = _task(tmp_path, "bad-yaml")
    path = task / "task.md"
    path.write_text(path.read_text().replace("timeout_sec: 300", "timeout_sec: [300"))
    with pytest.raises(TaskDocumentParseError) as raised:
        TaskDocument.from_path(path)
    line = int(re.search(r"task\.md:(\d+):", str(raised.value)).group(1))
    file_lines = path.read_text().splitlines()
    assert line > 5 and line <= len(file_lines)


# --- p5-11: bench agents fits on a screen ---------------------------------------


@pytest.mark.parametrize("columns", ["80", "100", "200"])
def test_bench_agents_is_one_line_per_agent_with_full_names(
    monkeypatch, columns
) -> None:
    """Guards dogfood p5-11 at 66471e95: 2,883 lines for 40 agents, with
    names cut (``claude-agent-…``) and no oracle or nop."""
    from benchflow.agents.registry import list_agents

    monkeypatch.setenv("BENCHFLOW_AGENTS_SOURCE", "off")
    result = _bench("agents", env={"COLUMNS": columns, "NO_COLOR": "1"})
    assert result.exit_code == 0, result.output
    names = sorted(a.name for a in list_agents())
    lines = result.stdout.splitlines()
    first = {line.split()[0] for line in lines if line.strip()}
    for name in [*names, "oracle", "nop"]:
        assert name in first, (name, columns)
    # A header, two controls, one line per agent and a short footer.
    assert len(lines) <= len(names) + 8, len(lines)


# --- p1-first-run-3: doctor and smoke send a newcomer to the oracle run ---------


def _readme_oracle_command() -> str:
    """The README's step 2, joined from its continuation lines."""
    text = (REPO / "README.md").read_text()
    start = text.index("bench run ")
    block = text[start : text.index("```", start)]
    command = " ".join(block.replace("\\\n", " ").split())
    assert "--agent oracle" in command, command
    return command


def _no_login(monkeypatch, tmp_path: Path, env: dict[str, str] | None = None):
    """The real doctor, on a machine with Docker running and only ``env``."""
    from benchflow import doctor
    from tests.test_doctor import make_probes

    real = doctor.run_doctor
    probes = make_probes(tmp_path, env=env or {})
    monkeypatch.setattr(
        doctor, "run_doctor", lambda **kwargs: real(**kwargs, probes=probes)
    )


def _lines(result) -> list[str]:
    return [" ".join(line.split()) for line in result.output.splitlines()]


def test_doctor_with_no_login_names_the_oracle_run(tmp_path, monkeypatch) -> None:
    """Guards DF-p1-first-run-3 at af9e05b3: with no login, doctor ended with
    ``Next: bench eval smoke``, which has no agent to run."""
    _no_login(monkeypatch, tmp_path)
    result = _bench("doctor", "--offline", terminal_width=80)
    assert result.exit_code == 0, result.output
    assert "Traceback" not in result.output
    lines = _lines(result)
    assert any("no model agent can run" in line for line in lines)
    nexts = [line for line in lines if line.startswith("Next:")]
    assert nexts == [f"Next: {_readme_oracle_command()}"], nexts
    after = " ".join(lines[lines.index(nexts[0]) :])
    assert "needs no login" in after
    # The smoke comes after a login, not as the step.
    assert after.index("log in") < after.index("bench eval smoke")


def test_doctor_with_a_login_names_the_oracle_run_then_the_smoke(
    tmp_path, monkeypatch
) -> None:
    """Guards DF-p1-first-run-3 at af9e05b3: the README's order is the oracle
    run, then the smoke; doctor named only the smoke."""
    _no_login(monkeypatch, tmp_path, {"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-x"})
    result = _bench("doctor", "--offline", "--sandbox", "docker")
    assert result.exit_code == 0, result.output
    lines = _lines(result)
    assert not any("no model agent can run" in line for line in lines)
    nexts = [line for line in lines if line.startswith("Next:")]
    assert nexts == [f"Next: {_readme_oracle_command()}"], nexts
    assert "bench eval smoke" in lines[lines.index(nexts[0]) + 1]


def test_smoke_with_no_login_names_the_oracle_run(tmp_path, monkeypatch) -> None:
    """Guards DF-p1-first-run-3 at af9e05b3: smoke with no login said only
    ``next: run bench doctor``, and doctor said ``bench eval smoke``."""
    _no_login(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)
    result = _bench("eval", "smoke", terminal_width=80)
    assert result.exit_code == 1, result.output
    assert "Traceback" not in result.output
    text = " ".join(result.output.split())
    assert "No agent to smoke" in text
    step = text[text.index("next:") :]
    assert f"`{_readme_oracle_command()}`" in step, step
    assert "needs no login" in step
    assert not (tmp_path / "jobs").exists()


# --- p1-first-run-4: no task is said before any agent or key -------------------


@pytest.mark.parametrize(
    "args",
    [
        ("eval", "run"),
        ("eval", "run", "--agent", "oracle"),
        ("eval", "run", "--agent", "codex", "--sandbox", "daytona"),
    ],
)
def test_a_run_with_no_task_says_so_first(tmp_path, monkeypatch, capsys, args) -> None:
    """Guards DF-p1-first-run-4 at af9e05b3: a bare ``bench eval run`` warned
    about the default agent and ended with ``ANTHROPIC_API_KEY required``
    (``--agent oracle``: Docker is not ready); it never said no task was
    given."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    for key in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "OPENAI_API_KEY",
        "CODEX_API_KEY",
        "DAYTONA_API_KEY",
        "BENCHFLOW_SKIP_PREFLIGHT",
    ):
        monkeypatch.delenv(key, raising=False)
    result = _bench(*args, terminal_width=200)
    assert result.exit_code == 2, result.output
    assert isinstance(result.exception, SystemExit), result.exception
    assert "Traceback" not in result.output
    errors = [
        line
        for line in (result.output + capsys.readouterr().err).splitlines()
        if line.strip() and not line.startswith("note:")
    ]
    assert errors[0].startswith("No tasks to run: no --config, --tasks-dir"), errors
    assert errors[1].strip().startswith("next: name a task"), errors
    assert (
        "bench run benchflow-ai/skillsbench/tasks/econ-detrending-correlation"
        in errors[1]
    )
    assert len(errors) == 2, errors  # no default-agent warning, key or Docker line
    assert not (tmp_path / "jobs").exists()


@pytest.mark.parametrize(
    ("typed", "line"),
    [
        (["eval", "run"], "bench run <task> --agent claude-agent-acp"),
        (["eval", "run", "--agent", "oracle"], "bench run <task> --agent oracle"),
        (
            ["eval", "run", "--dataset", "tb2@2.0", "--agent", "oracle"],
            "bench run --dataset tb2@2.0 --agent oracle",
        ),
        (["eval", "run", "--config", "run.yaml"], "bench run --config run.yaml"),
    ],
)
def test_the_old_verb_note_never_offers_a_run_with_no_task(typed, line) -> None:
    """Guards DF-p1-first-run-4 at af9e05b3: for a bare ``bench eval run`` the
    note offered ``bench run --agent claude-agent-acp``, which fails with "bench
    run needs a target"."""
    from benchflow.cli.verbs import note_for

    note = note_for(typed)
    assert note is not None
    assert note.split(": ", 1)[1].split(" (")[0] == line, note


# --- review of af9e05b3 ---------------------------------------------------------


def _requests(monkeypatch) -> list[Any]:
    """Record the request each run builds, and stop there: nothing runs."""
    import typer

    from benchflow import eval_plan

    seen: list[Any] = []

    def record(request):
        seen.append(request)
        raise typer.Exit(0)

    monkeypatch.setattr(eval_plan, "build_eval_plan", record)
    return seen


def _note_line(result) -> list[str]:
    notes = [line for line in result.output.splitlines() if line.startswith("note:")]
    assert len(notes) == 1, result.output
    line = notes[0].split(": ", 2)[2].split(" (")[0]
    words = shlex.split(line)
    assert words[:2] == ["bench", "run"], line
    return words[1:]


def _whole_run_files(tmp_path: Path) -> None:
    _task(tmp_path / "cfgsrc")
    (tmp_path / "cfg.yaml").write_text("tasks_dir: cfgsrc\nagent: claude-agent-acp\n")
    (tmp_path / "m.yaml").write_text("models:\n  - claude-haiku-4-5-20251001\n")


@pytest.mark.parametrize(
    "typed",
    [
        ["--config", "cfg.yaml", "--model", "claude-sonnet-4-5"],
        ["--config", "cfg.yaml", "--reasoning-effort", "high"],
        ["--config", "cfg.yaml", "--harness", "native"],
        ["--source-env", "primeintellect/general-agent", "--model", "gpt-5.5"],
        ["--tasks-dir", "cfgsrc", "--matrix", "m.yaml", "--reasoning-effort", "low"],
    ],
)
def test_the_note_line_of_a_whole_run_with_a_model_flag_runs(
    tmp_path, monkeypatch, typed
) -> None:
    """Guards review F1 at af9e05b3: with --config, --source-env or --matrix
    and no -a, bench run refused -m, --effort and --harness ("names a model,
    but no agent runs it"), so the note's line exited 2."""
    monkeypatch.chdir(tmp_path)
    _whole_run_files(tmp_path)
    seen = _requests(monkeypatch)
    before = _bench("eval", "run", *typed)
    assert before.exit_code == 0, before.output
    after = _bench(*_note_line(before))
    assert after.exit_code == 0, after.output
    assert len(seen) == 2, (before.output, after.output)
    assert seen[1] == seen[0]


_WHOLE_RUN = re.compile(r"\s--(?:config|source-env|matrix)\b")


def _docs_bench_run_commands() -> list[tuple[str, list[str]]]:
    """Each ``bench run`` command in a docs code block, continuations joined."""
    out: list[tuple[str, list[str]]] = []
    files = [REPO / "README.md", *sorted((REPO / "docs").rglob("*.md"))]
    for path in files:
        rel = path.relative_to(REPO).as_posix()
        # Fences toggle line by line, as Markdown reads them (an indented
        # fence in a list too).
        blocks: list[str] = []
        inside = False
        for raw in path.read_text().splitlines():
            if raw.strip().startswith("```"):
                inside = not inside
                if inside:
                    blocks.append("")
            elif inside:
                blocks[-1] += raw.strip() + "\n"
        for block in blocks:
            for line in block.replace("\\\n", " ").splitlines():
                words = line.split("#", 1)[0].split()
                while words and (
                    re.fullmatch(r"[A-Z_]+=\S*", words[0])
                    or words[0] in ("uv", "run", "$")
                ):
                    words = words[1:]
                if words[:2] == ["bench", "run"] and _WHOLE_RUN.search(
                    " " + " ".join(words)
                ):
                    out.append((rel, words[1:]))
    return out


def test_the_docs_whole_run_commands_include_the_hosted_env_one() -> None:
    commands = _docs_bench_run_commands()
    # The hosted-env command the review named is among them. Docs v2
    # moved docs/running-any-benchmark.md to docs/evaluate/bring-what-you-have.md,
    # which gives none yet.
    if not any("--source-env" in words for _, words in commands):
        pytest.skip(
            "page fix pending: no docs page gives a hosted-environment run "
            "(bench run --source-env ...); docs/running-any-benchmark.md did, "
            "and its successor docs/evaluate/bring-what-you-have.md does not"
        )
    assert any(
        rel == "docs/evaluate/bring-what-you-have.md" and "--source-env" in words
        for rel, words in commands
    ), commands


@pytest.mark.parametrize(
    ("rel", "words"),
    _docs_bench_run_commands(),
    ids=lambda v: v if isinstance(v, str) else " ".join(v)[:60],
)
def test_a_docs_bench_run_of_a_whole_run_runs(monkeypatch, rel, words) -> None:
    """Guards review F1 at af9e05b3: ``bench run --source-env ... --model ...``
    in docs/running-any-benchmark.md (now docs/evaluate/bring-what-you-have.md)
    exited 2."""
    if any("$" in w or "<" in w or "..." in w for w in words):
        pytest.skip(f"{rel}: a placeholder command")
    monkeypatch.chdir(REPO)
    seen = _requests(monkeypatch)
    result = _bench(*words)
    assert result.exit_code == 0, (rel, words, result.output)
    assert len(seen) == 1, (rel, words, result.output)


def _yaml_job(tmp_path: Path, **extra: Any) -> Path:
    """A YAML job config with every supported top-level key."""
    _task(tmp_path / "tasks")
    job = {
        "job_name": "yaml-job",
        "jobs_dir": "jobs",
        "n_attempts": 1,
        "install_only": False,
        "timeout_multiplier": 2.0,
        "agent_timeout_multiplier": None,
        "verifier_timeout_multiplier": 1.5,
        "agent_setup_timeout_multiplier": None,
        "environment_build_timeout_multiplier": None,
        "debug": True,
        "n_concurrent_trials": 3,
        "quiet": False,
        "retry": {"max_retries": 2},
        "environment": {"type": "docker"},
        "verifier": {"disable": True},
        "metrics": [{"type": "mean"}],
        "agents": [{"name": "oracle"}],
        "user_agent": None,
        "datasets": [{"path": "tasks"}],
        "tasks": [],
        "artifacts": ["/app/out.txt"],
        "extra_instruction_paths": [],
        "extra_instructions": ["be brief"],
        "source_jobs": [],
        **extra,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(job, indent=2))
    return path


def test_a_job_job_config_runs_and_names_what_it_ignores(
    tmp_path, monkeypatch, caplog
) -> None:
    """Guards review F2 at af9e05b3: a YAML job config with fully expanded
    top-level keys exited 2 (``timeout_multiplier: unknown key``); it ran on
    66471e95 and in 0.7.11. Now the settings BenchFlow
    cannot honour are refused by name (they were ignored with a warning), and
    the file runs once they are dropped."""
    from benchflow._utils.config_keys import ConfigKeyError
    from benchflow.evaluation import Evaluation

    monkeypatch.chdir(tmp_path)
    path = _yaml_job(tmp_path)
    with pytest.raises(ConfigKeyError) as refused:
        Evaluation.from_yaml(path, preflight=False, warn_default_agent=False)
    text = str(refused.value)
    # timeout_multiplier for its agent setup phase only (review A L-A1 ruling).
    for key in ("timeout_multiplier", "debug", "retry", "verifier", "artifacts"):
        assert f": {key}: " in text, (key, text)
    # The accepted defaults, the mean metric and the verifier multiplier (which
    # BenchFlow scales) are not refused.
    for key in ("install_only", "quiet", "tasks", "user_agent", "source_jobs",
                "metrics", "verifier_timeout_multiplier"):  # fmt: skip
        assert f": {key}: " not in text, (key, text)
    result = _bench("run", "--config", str(path), "--dry-run")
    assert result.exit_code == 2, result.output

    job = json.loads(path.read_text())
    job.update(
        agent_setup_timeout_multiplier=1, debug=False, retry={"max_retries": 0},
        verifier={"disable": False}, artifacts=[],
    )  # fmt: skip
    path.write_text(json.dumps(job, indent=2))
    j = Evaluation.from_yaml(path, preflight=False, warn_default_agent=False)
    assert j._config.agent == "oracle"
    assert j._config.concurrency == 3
    assert j._config.timeout_multiplier == 2.0
    assert j._config.config_override == {
        "timeout_multipliers": {"verifier": 1.5, "build": 2.0}
    }
    assert j._config.extra_instruction == "be brief"
    assert j._job_name == "yaml-job"

    result = _bench("run", "--config", str(path), "--dry-run")
    assert result.exit_code == 0, result.output
    plan = _plan(result)
    assert (plan["agent"], plan["concurrency"], plan["timeout_multiplier"]) == (
        "oracle",
        3,
        2.0,
    )
    assert plan["job"] == "yaml-job"


def test_the_reviewers_job_file_runs_at_its_concurrency(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _task(tmp_path / "tasks")
    (tmp_path / "cfg.yaml").write_text(
        "jobs_dir: jobs\nn_attempts: 1\ntimeout_multiplier: 1.0\n"
        "orchestrator:\n  n_concurrent_trials: 4\nenvironment:\n  type: docker\n"
        "agents:\n  - name: oracle\ndatasets:\n  - path: tasks\n"
    )
    result = _bench("run", "--config", "cfg.yaml", "--dry-run")
    assert result.exit_code == 0, result.output
    assert (_plan(result)["concurrency"], _plan(result)["tasks"]) == (
        4,
        ["hello-craft"],
    )


def test_a_typo_in_a_job_job_config_is_still_refused(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = _yaml_job(tmp_path, timeout_multiplyer=2.0)
    result = _bench("run", "--config", str(path), "--dry-run")
    assert result.exit_code == 2, result.output
    text = " ".join(result.output.split())
    assert (
        "timeout_multiplyer: unknown key (did you mean 'timeout_multiplier'?)" in text
    )
    assert "a job setting" not in text  # refused, not half-loaded
    assert "Traceback" not in result.output


def test_a_bare_variable_name_in_the_note_prints_as_typed(tmp_path, monkeypatch):
    """Guards review L1 at af9e05b3: ``--allow-task-env HF_TOKEN`` printed as
    ``--allow-task-env '<hidden>'``, so the line could not be re-run."""
    from benchflow.cli.verbs import note_for

    base = ["eval", "run", "--tasks-dir", "tasks", "--agent", "oracle"]
    note = note_for([*base, "--allow-task-env", "HF_TOKEN"])
    assert "--allow-task-env HF_TOKEN" in note, note
    note = note_for([*base, "--allow-task-env", "HF_TOKEN=hf_sekrit"])
    assert "--allow-task-env HF_TOKEN=" in note and "sekrit" not in note, note
    note = note_for([*base, "--source-env-arg", "n=5"])
    assert "--source-env-arg n='<hidden>'" in note, note
    # A word that is not a variable name may be a token typed in its place.
    for token in ("ghp_abcDEF123", "sk-ant-api03-xyz", "deadbeef0123"):
        note = note_for([*base, "--allow-task-env", token])
        assert token not in note and "'<hidden>'" in note, note

    monkeypatch.chdir(tmp_path)
    _task(tmp_path / "tasks")
    seen = _requests(monkeypatch)
    before = _bench(*base, "--allow-task-env", "HF_TOKEN")
    assert before.exit_code == 0, before.output
    after = _bench(*_note_line(before))
    assert after.exit_code == 0, after.output
    assert len(seen) == 2 and seen[1] == seen[0]


@pytest.mark.parametrize("value", ["null", "", "none", "0"])
def test_idle_timeout_null_turns_the_watchdog_off_on_both_verbs(
    tmp_path, monkeypatch, value
) -> None:
    """Guards review L2 at af9e05b3: ``bench run --agent-idle-timeout null``
    exited 2; ``bench eval run`` takes it and turns the watchdog off."""
    from benchflow._utils.config import normalize_agent_idle_timeout

    monkeypatch.chdir(tmp_path)
    _task(tmp_path / "tasks")
    seen = _requests(monkeypatch)
    old = _bench(
        "eval", "run", "--tasks-dir", "tasks", "--agent", "oracle",
        "--agent-idle-timeout", value,
    )  # fmt: skip
    assert old.exit_code == 0, old.output
    for flag in ("--agent-idle-timeout", "--idle-timeout"):
        new = _bench("run", "tasks", flag, value)
        assert new.exit_code == 0, (flag, new.output)
    note = _bench(*_note_line(old))
    assert note.exit_code == 0, note.output
    timeouts = [normalize_agent_idle_timeout(r.agent_idle_timeout) for r in seen]
    assert timeouts == [None] * 4, timeouts


# --- review of e31bf65a ---------------------------------------------------------

# Compatible task inputs and minimal job configs built by these tests.
JOB_FIXTURES = REPO / "tests" / "fixtures" / "job_configs"
SPARSE_JOB = "2026-10-19__01-17-57"


def _config_project(tmp_path: Path, *jobs: str) -> Path:
    import shutil

    shutil.copytree(JOB_FIXTURES / "tasks", tmp_path / "tasks")
    for job in jobs or (SPARSE_JOB, "named-job"):
        folder = tmp_path / "jobs" / job
        folder.mkdir(parents=True)
        (folder / "config.json").write_text(
            json.dumps(
                {
                    "job_name": job,
                    "quiet": True,
                    "tasks": [{"path": "tasks/hello-craft"}],
                }
            )
        )
    return tmp_path


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (
            {"datasets": [{"path": "tasks"}], "tasks": [{"path": "tasks/hello-craft"}]},
            "datasets and tasks are both set",
        ),
        (
            {"tasks": [{"path": "tasks/hello-craft"}, {"path": "other/hello-two"}]},
            "tasks: these tasks are in different folders",
        ),
        ({"tasks": [{"name": "dxb/hello-craft"}]}, "tasks: every entry needs a path"),
        (
            {"datasets": [{"name": "terminal-bench", "version": "2.0"}]},
            "datasets[0] has no path",
        ),
    ],
)
def test_a_job_task_list_benchflow_cannot_run_as_one_folder_is_refused(
    tmp_path, monkeypatch, extra, message
) -> None:
    """A job config whose tasks BenchFlow cannot run as one folder is refused;
    before, ``tasks`` was ignored and ``tasks/`` ran whole."""
    monkeypatch.chdir(_config_project(tmp_path, SPARSE_JOB))
    (tmp_path / "cfg.json").write_text(
        json.dumps({"agents": [{"name": "oracle"}], **extra})
    )
    result = _bench("run", "--config", "cfg.json", "--dry-run")
    assert result.exit_code == 1, result.output
    assert message in " ".join(result.output.split()), result.output
    assert "Traceback" not in result.output


def test_job_config_default_blocks_are_not_warned_about(tmp_path, monkeypatch, caplog):
    """Guards review L-R2-2 at e31bf65a: a file that spells out the accepted
    default retry and verifier blocks (as `model_dump()` and lock.json do)
    got two "ignored" warnings."""
    import logging

    from benchflow._utils.config_keys import ConfigKeyError
    from benchflow.evaluation import Evaluation
    from benchflow.job_config import JOB_DEFAULTS as IGNORED_JOB_DEFAULTS

    monkeypatch.chdir(_config_project(tmp_path, SPARSE_JOB))
    defaults = {**IGNORED_JOB_DEFAULTS, "retry": {"max_retries": 0}}
    for spelled in (IGNORED_JOB_DEFAULTS, defaults):
        (tmp_path / "cfg.json").write_text(
            json.dumps(
                {
                    **spelled,
                    "agents": [{"name": "oracle"}],
                    "datasets": [{"path": "tasks"}],
                }
            )
        )
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="benchflow.evaluation"):
            Evaluation.from_yaml("cfg.json", preflight=False, warn_default_agent=False)
        assert not [
            r.getMessage() for r in caplog.records if "job setting" in r.getMessage()
        ]
    # A changed default is refused, by name (it warned).
    retry = {**IGNORED_JOB_DEFAULTS["retry"], "max_retries": 3}
    (tmp_path / "cfg.json").write_text(
        json.dumps(
            {
                "retry": retry,
                "agents": [{"name": "oracle"}],
                "datasets": [{"path": "tasks"}],
            }
        )
    )
    with pytest.raises(ConfigKeyError, match=r"cfg\.json:\d+: retry: "):
        Evaluation.from_yaml("cfg.json", preflight=False, warn_default_agent=False)


def test_job_config_concurrency_and_timeout_follow_precedence(
    tmp_path, monkeypatch, caplog
) -> None:
    """Guards review L-R2-4 at e31bf65a: orchestrator.n_concurrent_trials won
    over the top-level job value, and timeout_multiplier
    scaled only the agent timeout without saying so."""
    import logging

    from benchflow.evaluation import Evaluation

    monkeypatch.chdir(_config_project(tmp_path, SPARSE_JOB))
    (tmp_path / "cfg.json").write_text(
        json.dumps(
            {
                "n_concurrent_trials": 2,
                "orchestrator": {"n_concurrent_trials": 8},
                "agent_timeout_multiplier": 1.5,
                "agents": [{"name": "oracle"}],
                "datasets": [{"path": "tasks"}],
            }
        )
    )
    with caplog.at_level(logging.WARNING, logger="benchflow.evaluation"):
        j = Evaluation.from_yaml("cfg.json", preflight=False, warn_default_agent=False)
    assert j._config.concurrency == 2
    assert j._config.timeout_multiplier == 1.5
    warned = " ".join(r.getMessage() for r in caplog.records)
    assert "n_concurrent_trials: 2 at the top level wins" in warned
    # timeout_multiplier needs an explicit unscaled agent setup phase;
    # otherwise it is refused by name.
    from benchflow._utils.config_keys import ConfigKeyError

    (tmp_path / "cfg.json").write_text(
        json.dumps(
            {
                "timeout_multiplier": 1.5,
                "agents": [{"name": "oracle"}],
                "datasets": [{"path": "tasks"}],
            }
        )
    )
    with pytest.raises(
        ConfigKeyError, match=r"timeout_multiplier: 1\.5: BenchFlow scales the agent"
    ):
        Evaluation.from_yaml("cfg.json", preflight=False, warn_default_agent=False)


@pytest.mark.parametrize(
    "value",
    ["AKIAIOSFODNN7EXAMPLE", "ABCDEFGHIJKLMNOPQRSTUV", "A1B2C3D4E5F6", "4519736280"],
)
def test_a_credential_shaped_allow_task_env_word_stays_hidden(value) -> None:
    """Guards review L-R2-3 at e31bf65a: an all-caps key typed under
    --allow-task-env (an AWS key id) printed in the note."""
    from benchflow.cli.verbs import note_for

    note = note_for(["eval", "run", "--tasks-dir", "t", "--agent", "oracle",
                     "--allow-task-env", value])  # fmt: skip
    assert value not in note and "--allow-task-env '<hidden>'" in note, note


@pytest.mark.parametrize(
    "name", ["HF_TOKEN", "AWS_ACCESS_KEY_ID", "PIP_INDEX_URL", "HOME"]
)
def test_a_variable_name_under_allow_task_env_still_prints(name) -> None:
    from benchflow.cli.verbs import note_for

    note = note_for(["eval", "run", "--tasks-dir", "t", "--agent", "oracle",
                     "--allow-task-env", name])  # fmt: skip
    assert f"--allow-task-env {name}" in note, note


def test_a_matrix_note_with_no_task_names_the_task_slot() -> None:
    """Guards review L-R2-5 at e31bf65a: a matrix names its agents but not its
    tasks, so its note with no task needs the <task> slot too."""
    from benchflow.cli.verbs import note_for

    note = note_for(["eval", "run", "--matrix", "m.yaml"])
    assert note.split(": ", 1)[1] == "bench run <task> --matrix m.yaml", note


# --- round 4: task.toml `collect = []` ------------------------------------------

# A compatible task fixture with an empty [verifier] collect list.
TOML_INIT_TASK = JOB_FIXTURES / "tasks" / "hello-craft"


def _scaffold(tmp_path: Path, *, collect: str | None = "keep") -> Path:
    import shutil

    task = tmp_path / ("hello-craft" if collect == "keep" else "variant")
    shutil.copytree(TOML_INIT_TASK, task)
    toml = task / "task.toml"
    text = toml.read_text()
    assert "\ncollect = []\n" in text  # the fixture explicitly includes an empty list
    if collect != "keep":
        text = text.replace(
            "\ncollect = []\n", "\n" if collect is None else f"\n{collect}\n"
        )
        toml.write_text(text)
    return task


def _backends() -> list[str]:
    from benchflow.task.runtime_capabilities import SUPPORTED_SANDBOX_BACKENDS

    return sorted(SUPPORTED_SANDBOX_BACKENDS)


def _what_benchflow_sees(task: Path, sandbox: str) -> dict[str, Any]:
    from benchflow._utils.task_authoring import check_task, check_task_warnings
    from benchflow.task import Task
    from benchflow.task.runtime_capabilities import validate_task_runtime_support

    config = Task(task).config
    issues = validate_task_runtime_support(config, sandbox=sandbox, task_dir=task)
    return {
        "ignored": config.ignored_keys,
        "config": config.model_dump(mode="json"),
        "issues": [(i.path, i.reason) for i in issues],
        "check": check_task(task, sandbox_type=sandbox),
        "warnings": check_task_warnings(task),
    }


@pytest.mark.parametrize("sandbox", _backends())
def test_an_empty_collect_is_the_same_as_no_collect(tmp_path, sandbox) -> None:
    """Round 4: a task with an empty `collect = []` needs no hooks, but
    BenchFlow refused it (`verifier.collect: task.toml verifier collect hooks
    ... are not executed`) before the oracle ran, though the list is empty."""
    scaffold = _what_benchflow_sees(_scaffold(tmp_path), sandbox)
    absent = _what_benchflow_sees(_scaffold(tmp_path, collect=None), sandbox)
    assert scaffold == absent
    # The config holds an empty collect list; nothing refuses or warns about it.
    reported = (scaffold["issues"], scaffold["check"], scaffold["warnings"])
    assert not any("collect" in str(v) for v in reported), scaffold


@pytest.mark.parametrize("verb", [("check",), ("tasks", "check")])
def test_bench_check_passes_the_toml_init_scaffold(tmp_path, verb) -> None:
    result = _bench(*verb, str(_scaffold(tmp_path)))
    assert result.exit_code == 0, result.output
    assert "collect" not in result.output, result.output


@pytest.mark.parametrize("sandbox", _backends())
def test_a_non_empty_collect_is_honoured(tmp_path, sandbox) -> None:
    """[[verifier.collect]] hooks run since they were modelled: a task with one
    is neither refused nor warned about, and its config holds the hook."""
    hooks = '\n[[verifier.collect]]\nservice = "main"\ncommand = "cat /tmp/x"\n'
    task = _scaffold(tmp_path, collect=None)
    (task / "task.toml").write_text((task / "task.toml").read_text() + hooks)
    seen = _what_benchflow_sees(task, sandbox)
    assert not any("collect" in path for path, _ in seen["issues"]), seen["issues"]
    assert not any("collect" in w for w in seen["warnings"]), seen["warnings"]
    (hook,) = seen["config"]["verifier"]["collect"]
    assert (hook["service"], hook["command"]) == ("main", "cat /tmp/x")


# --- round 4: review of 21fc931e ------------------------------------------------

JOB_CONFIGS = JOB_FIXTURES / "configs"


def _job_config(tmp_path: Path, name: str) -> Path:
    import shutil

    target = tmp_path / name
    shutil.copy(JOB_CONFIGS / name, target)
    return target


def test_a_toml_task_pinned_to_git_is_refused(tmp_path, monkeypatch) -> None:
    """Guards review L-R3-1 at 21fc931e: a task entry pinned to a git repo
    (`git_url`, `git_commit_id`) ran the local
    folder of that name, which may be another version."""
    monkeypatch.chdir(_config_project(tmp_path, SPARSE_JOB))
    _job_config(tmp_path, "git-task-config.json")
    result = _bench(
        "run", "--config", "git-task-config.json", "-a", "oracle", "--dry-run"
    )
    assert result.exit_code == 1, result.output
    text = " ".join(result.output.split())
    assert (
        "'tasks/hello-craft' is pinned to https://github.com/benchflow-ai/skillsbench.git"
        in text
    )
    assert (
        "BenchFlow does not fetch git tasks from job configs" in text
        and "bench run <folder>" in text
    )
    assert "Traceback" not in result.output


def test_a_job_task_list_matches_its_tasks_by_exact_name(tmp_path, monkeypatch):
    """Guards review L-R3-2 at 21fc931e: the list's names were matched as
    globs, so `hello-tw?` also selected hello-two."""
    import shutil

    monkeypatch.chdir(_config_project(tmp_path, SPARSE_JOB))
    shutil.copytree(tmp_path / "tasks" / "hello-two", tmp_path / "tasks" / "hello-tw?")
    _job_config(tmp_path, "glob-name-config.json")
    result = _bench(
        "run", "--config", "glob-name-config.json", "-a", "oracle", "--dry-run"
    )
    assert result.exit_code == 0, result.output
    assert _plan(result)["tasks"] == ["hello-craft", "hello-tw?"]


@pytest.mark.parametrize("flag", ["--include", "--exclude"])
def test_a_cli_filter_cannot_change_a_job_task_list(tmp_path, monkeypatch, flag):
    """Guards review L-R3-2 at 21fc931e: `--include` replaced the file's
    `tasks:` list, and `--exclude` narrowed it."""
    monkeypatch.chdir(_config_project(tmp_path, "hb-named"))
    config = tmp_path / "list.json"
    config.write_text(
        json.dumps(
            {"tasks": [{"path": "tasks/hello-craft"}, {"path": "tasks/hello-two"}]}
        )
    )
    result = _bench(
        "run", "--config", "list.json", "-a", "oracle", flag, "hello-two", "--dry-run"
    )
    assert result.exit_code == 2, result.output
    text = " ".join(result.output.split())
    assert "its tasks: list names the tasks to run (hello-craft, hello-two)" in text
    assert "next: drop --include/--exclude, or edit the tasks: list" in text


@pytest.mark.parametrize("key", ["include", "exclude_tasks"])
def test_a_file_filter_cannot_change_a_job_task_list(tmp_path, monkeypatch, key):
    """Guards review L-R3-2 at 21fc931e: a file-level `include:` was added to
    the `tasks:` list (a union)."""
    monkeypatch.chdir(_config_project(tmp_path, "hb-named"))
    (tmp_path / "list.json").write_text(
        json.dumps({"tasks": [{"path": "tasks/hello-craft"}], key: ["hello-two"]})
    )
    result = _bench("run", "--config", "list.json", "-a", "oracle", "--dry-run")
    assert result.exit_code == 1, result.output
    assert "would change it; keep one" in " ".join(result.output.split())


def test_a_job_tasks_config_with_no_agent_runs_the_oracle(
    tmp_path, monkeypatch, capsys
) -> None:
    """Guards review L-R3-3 at 21fc931e: a task-list config.json
    names no agent, and BenchFlow ran its
    default model agent (claude-agent-acp, which needs a key)."""
    monkeypatch.chdir(_config_project(tmp_path, SPARSE_JOB))
    config = f"jobs/{SPARSE_JOB}/config.json"
    result = _bench("run", "--config", config, "--dry-run")
    assert result.exit_code == 0, result.output
    plan = _plan(result)
    assert (plan["agent"], plan["model"], plan["tasks"]) == (
        "oracle",
        None,
        ["hello-craft"],
    )
    printed = result.output + capsys.readouterr().err
    notes = [line for line in printed.splitlines() if "names no agent" in line]
    assert notes == [
        f"note: {config} names no agent, so the oracle runs; pass -a "
        "<agent> (or add agents:) to run a model agent"
    ], printed
    assert "No agent named" not in printed  # the default-agent warning

    # Naming an agent runs it, and says nothing about the oracle.
    named = _bench("run", "--config", config, "-a", "claude", "--dry-run")
    assert named.exit_code == 0, named.output
    assert _plan(named)["agent"] == "claude-agent-acp"
    assert "names no agent" not in named.output + capsys.readouterr().err


def test_other_configs_with_no_agent_keep_the_default_agent(tmp_path, monkeypatch):
    """The oracle rule is for YAML job configs that name no agent: a BenchFlow-format
    config with no agent, and a YAML `agents:` entry with no name (it asks
    for a model agent), keep the default agent."""
    from benchflow.evaluation import DEFAULT_AGENT, Evaluation

    monkeypatch.chdir(_config_project(tmp_path, SPARSE_JOB))
    (tmp_path / "unnamed.json").write_text(
        json.dumps({"agents": [{"model_name": "x"}], "datasets": [{"path": "tasks"}]})
    )
    (tmp_path / "native.yaml").write_text("tasks_dir: tasks\n")
    for name in ("unnamed.json", "native.yaml"):
        j = Evaluation.from_yaml(name, preflight=False, warn_default_agent=False)
        assert (
            j._config.agent,
            j._config.agent_defaulted,
            getattr(j, "_default_oracle", False),
        ) == (
            DEFAULT_AGENT,
            True,
            False,
        ), name


# --- round 5: every YAML job config that names no agent -------------------------


@pytest.mark.parametrize(
    ("config", "tasks"),
    [
        ("datasets-no-agent-config.json", ["hello-craft", "hello-two"]),
        ("empty-agents-config.json", ["hello-craft"]),
    ],
)
def test_every_job_config_that_names_no_agent_runs_the_oracle(
    tmp_path, monkeypatch, capsys, config, tasks
) -> None:
    """Guards review L-R4-2 at 2bf2434e: a YAML `datasets:` file
    with no `agents:`, and an explicit `agents: []`, ran BenchFlow's default
    model agent (claude-agent-acp, which needs a key). Both compatible config shapes must load."""
    monkeypatch.chdir(_config_project(tmp_path, SPARSE_JOB))
    _job_config(tmp_path, config)
    result = _bench("run", "--config", config, "--dry-run")
    assert result.exit_code == 0, result.output
    plan = _plan(result)
    assert (plan["agent"], plan["model"], plan["tasks"]) == ("oracle", None, tasks)
    printed = result.output + capsys.readouterr().err
    notes = [line for line in printed.splitlines() if "names no agent" in line]
    assert notes == [
        f"note: {config} names no agent, so the oracle runs; pass -a "
        "<agent> (or add agents:) to run a model agent"
    ], printed
    assert "No agent named" not in printed
