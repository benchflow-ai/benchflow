"""bench's verb set, the BenchFlow 0.8 spellings, and next steps.

Rust's ``bench`` 0.8.2 and this ``bench`` share one verb set. These tests run
the real CLI: ``bench --help``'s groups, the one note each 0.8 spelling prints
on stderr (never on stdout), the flags its note renames, the map a 0.8 group
typed alone prints, the next step of a mistyped word, an unknown command or
an unknown flag, ``bench run``'s oracle default, and the parity with Rust
0.8.2's commands. They read the CLI from outside, so each fails on a ``bench``
without the verb set by what it prints, not by an import.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shlex
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from benchflow.cli.main import app
from tests.test_python_sdk_load_job import _job, _trial

# Rust bench 0.8.2's commands: every "### bench <command>" heading of
# docs/rust/cli.md at bench-rs 518009b0, in its order.
RUST_082_COMMANDS = (
    "new",
    "check",
    "test",
    "run",
    "results",
    "view",
    "regrade",
    "branch",
    "shell",
    "handoff",
    "doctor",
    "cache",
    "cache list",
    "cache clean",
    "agents",
    "agents check",
    "import",
    "export",
    "explain",
    "schema",
    "completions",
)

# `bench --help`'s groups: Rust's three, with the verbs Python has, then More.
HELP_GROUPS = {
    "Tasks": ["new", "check", "import"],
    "Runs": ["run", "branch", "results", "view", "regrade", "export"],
    "This machine": ["doctor", "agents", "explain", "cache"],
}


def _bench(*args: str, **kwargs: Any):
    return CliRunner().invoke(app, list(args), prog_name="bench", **kwargs)


def _flat(text: str) -> str:
    """Text without Rich's box characters, whitespace collapsed."""
    return " ".join(re.sub(r"[│╭╮╰╯─]", " ", text).split())


def _panels(text: str) -> dict[str, list[str]]:
    panels: dict[str, list[str]] = {}
    current = None
    for line in text.splitlines():
        title = re.match(r"^\s*╭─ (.+?) ─", line)
        if title:
            current = title.group(1)
            panels[current] = []
            continue
        row = re.match(r"^\s*│ ([a-z][\w-]*)\s", line)
        if row and current is not None:
            panels[current].append(row.group(1))
    return panels


def _task(root: Path, name: str = "hello") -> Path:
    """A task the run runs (its dry run lists it): task.md, an image, an
    oracle and a verifier."""
    task = root / name
    (task / "environment").mkdir(parents=True)
    (task / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (task / "oracle").mkdir()
    (task / "oracle" / "solve.sh").write_text("#!/bin/bash\necho hello\n")
    (task / "verifier").mkdir()
    (task / "verifier" / "test.sh").write_text(
        "#!/bin/bash\necho 1.0 > /logs/verifier/reward.txt\n"
    )
    (task / "task.md").write_text("---\nname: hello\n---\nSay hello.\n")
    return task


# --- bench --help --------------------------------------------------------------


def test_help_lists_the_verbs_in_rusts_groups_then_more() -> None:
    result = _bench("--help", terminal_width=200)
    assert result.exit_code == 0, result.output
    panels = _panels(result.output)
    assert list(panels) == ["Options", "Tasks", "Runs", "This machine", "More"]
    for group, verbs in HELP_GROUPS.items():
        assert panels[group] == verbs, (group, panels[group])
    assert {"eval", "tasks", "agent"} <= set(panels["More"])


@pytest.mark.parametrize(
    "verb", [verb for verbs in HELP_GROUPS.values() for verb in verbs]
)
def test_each_verb_prints_its_help(verb) -> None:
    result = _bench(verb, "--help")
    assert result.exit_code == 0, result.output
    assert f"Usage: bench {verb}" in result.output


def test_embodied_is_not_listed_and_still_works() -> None:
    listing = _bench("--help", terminal_width=200).output
    assert "embodied" not in listing
    for args in (["embodied", "--help"], ["embodied", "check-spec", "--help"]):
        result = _bench(*args)
        assert result.exit_code == 0, result.output
        assert "Usage: bench embodied" in result.output


# --- each 0.8 spelling runs and prints one note on stderr -----------------------


def _one_note(result, new: str) -> str:
    notes = [line for line in result.stderr.splitlines() if line.startswith("note:")]
    assert len(notes) == 1, result.stderr
    assert f"is now `bench {new}`" in notes[0], notes[0]
    assert "note:" not in result.stdout, result.stdout
    return notes[0]


def test_eval_run_runs_and_its_note_stays_off_stdout(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks"
    _task(tasks)
    for spelling in ("run", "create"):
        result = _bench(
            "eval",
            spelling,
            "--tasks-dir",
            str(tasks),
            "--agent",
            "oracle",
            "--dry-run",
        )
        assert result.exit_code == 0, result.output
        _one_note(result, "run")
        # stdout is the plan, as before: JSON only.
        assert json.loads(result.stdout)["agent"] == "oracle"


def test_eval_list_inspect_and_compare_note_bench_results(tmp_path: Path) -> None:
    a = _job(tmp_path, "a", {"t1": 1.0, "t2": 0.0})
    b = _job(tmp_path, "b", {"t1": 1.0, "t2": 1.0})
    listed = _bench("eval", "list", str(tmp_path))
    assert listed.exit_code == 0 and "Jobs" in listed.stdout
    _one_note(listed, "results")
    inspected = _bench("eval", "inspect", str(a), "--json")
    assert inspected.exit_code == 0, inspected.output
    assert json.loads(inspected.stdout)["kind"] == "benchflow.job"
    assert "bench results " in _one_note(inspected, "results")
    compared = _bench("eval", "compare", str(a), str(b), "--json")
    assert compared.exit_code == 0, compared.output
    assert json.loads(compared.stdout)["kind"] == "benchflow.comparison"
    _one_note(compared, "results")


def test_eval_view_regrade_and_branch_note_their_verbs(tmp_path, monkeypatch) -> None:
    from benchflow.trajectories import viewer

    served: list[str] = []
    monkeypatch.setattr(viewer, "serve", lambda path, *a, **k: served.append(path))
    view = _bench("eval", "view", str(tmp_path), "--port", "8999")
    assert view.exit_code == 0 and served == [str(tmp_path)], view.output
    assert "same arguments" in _one_note(view, "view")
    missing = str(tmp_path / "no-such-job")
    _one_note(_bench("eval", "regrade", missing), "regrade")
    _one_note(
        _bench("eval", "branch", "--tasks-dir", missing, "--agent", "oracle"), "branch"
    )


def test_tasks_spellings_run_and_note_their_verbs(tmp_path: Path) -> None:
    created = _bench("tasks", "init", "fresh-task", "--dir", str(tmp_path))
    assert created.exit_code == 0, created.output
    assert (tmp_path / "fresh-task" / "task.md").is_file()
    _one_note(created, "new")
    task = str(tmp_path / "fresh-task")
    _one_note(_bench("tasks", "check", task), "check")
    _one_note(_bench("tasks", "migrate", task), "import")
    _one_note(_bench("tasks", "export", task, "--report-only"), "export")


def test_agent_list_and_show_note_bench_agents(tmp_path, monkeypatch) -> None:
    # An empty local catalog: no network.
    (tmp_path / "agents").mkdir()
    monkeypatch.setenv("BENCHFLOW_AGENTS_SOURCE", str(tmp_path / "agents"))
    listed = _bench("agent", "list")
    assert listed.exit_code == 0 and "Registered Agents" in listed.stdout
    _one_note(listed, "agents")
    shown = _bench("agent", "show", "codex")
    assert shown.exit_code == 0 and "codex-acp" in shown.stdout
    _one_note(shown, "agents")


def test_help_of_a_0_8_spelling_prints_no_note() -> None:
    result = _bench("eval", "run", "--help")
    assert result.exit_code == 0
    assert "note:" not in result.output


# --- flags translate as Rust's coming-from-0.8.md table --------------------------


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (["--job-name", "j"], "--job j"),
        (["--n-tasks", "2"], "--limit 2"),
        (["--reasoning-effort", "high"], "--effort high"),
        (["--freeze-workspace"], "--save-outputs"),
        (["--agent-idle-timeout", "30"], "--idle-timeout 30s"),
        (["--harness", "native"], "--harness native"),
        (["--skill-mode", "with-skill"], "--task-skills"),
        (
            ["--sandbox", "docker", "--concurrency", "3"],
            "--sandbox docker --concurrency 3",
        ),
        (["--include", "a*", "--exclude", "b"], "--include 'a*' --exclude b"),
        (["--jobs-dir", "out", "--fresh"], "--jobs-dir out --fresh"),
        (["--fail-under", "0.5"], "--fail-under 0.5"),
        (["--timeout-multiplier", "2"], "--timeout-multiplier 2"),
        (["--extra-instruction", "be brief"], "--extra-instruction 'be brief'"),
    ],
    ids=lambda v: v[0] if isinstance(v, list) else None,
)
def test_eval_run_flags_are_renamed_in_its_note(tmp_path, old, new) -> None:
    tasks = tmp_path / "tasks"
    _task(tasks)
    result = _bench(
        "eval", "run", "--tasks-dir", str(tasks), "--agent", "oracle", *old, "--dry-run"
    )
    note = _one_note(result, "run")
    assert f"bench run {tasks} --agent oracle {new} --dry-run" in note, note


def test_eval_run_note_names_the_default_agent_and_hides_env_values(tmp_path) -> None:
    tasks = tmp_path / "tasks"
    _task(tasks)
    result = _bench(
        "eval",
        "run",
        "--tasks-dir",
        str(tasks),
        "--agent-env",
        "MY_API_KEY=sk-not-a-real-value-1234567890",
        "--skill-mode",
        "no-skill",
        "--dry-run",
    )
    note = _one_note(result, "run")
    assert "sk-not-a-real-value" not in result.output
    assert '--env MY_API_KEY="$MY_API_KEY"' in note
    # 0.8 ran claude-agent-acp without --agent; bench run would run the oracle.
    assert "--agent claude-agent-acp" in note
    assert "--skill-mode" not in note and "--task-skills" not in note


def test_eval_run_source_repo_becomes_a_target() -> None:
    result = _bench(
        "eval",
        "run",
        "--source-repo",
        "org/repo",
        "--source-path",
        "tasks",
        "--source-ref",
        "main",
        "--agent",
        "oracle",
        "--dry-run",
    )
    assert "bench run org/repo/tasks@main --agent oracle --dry-run" in _one_note(
        result, "run"
    )


def test_every_eval_run_flag_has_its_bench_run_line(tmp_path) -> None:
    """Since DX-b every bench eval run flag is a bench run flag: the note
    names the whole line, and that line plans the same run."""
    tasks = tmp_path / "tasks"
    _task(tasks)
    result = _bench(
        "eval", "run", "--tasks-dir", str(tasks), "--agent", "oracle",
        "--max-tokens", "100", "--sandbox-user", "agent", "--dry-run",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    note = _one_note(result, "run")
    assert "--max-tokens 100 --sandbox-user agent --dry-run" in note
    line = note.split(": ", 2)[2]
    after = _bench(*shlex.split(line)[1:])
    assert after.exit_code == 0, after.output
    assert json.loads(after.stdout) == json.loads(result.stdout)


def test_eval_inspect_flags_without_a_results_spelling_keep_the_old_line(tmp_path):
    a = _job(tmp_path, "a", {"t1": 1.0})
    out = tmp_path / "out.json"
    result = _bench("eval", "inspect", str(a), "--json", "--out", str(out))
    assert result.exit_code == 0 and out.is_file(), result.output
    note = _one_note(result, "results")
    assert "(--out only as `bench eval inspect`)" in note


def test_the_new_line_plans_the_same_run(tmp_path) -> None:
    """The line a note names, run with --dry-run, plans what the old one did."""
    tasks = tmp_path / "tasks"
    _task(tasks)
    old = [
        "eval", "run", "--tasks-dir", str(tasks), "--agent", "oracle",
        "--n-tasks", "1", "--include", "hel*", "--concurrency", "2",
        "--jobs-dir", str(tmp_path / "jobs"), "--extra-instruction", "be brief",
        "--timeout-multiplier", "2", "--dry-run",
    ]  # fmt: skip
    before = _bench(*old)
    assert before.exit_code == 0, before.output
    line = _one_note(before, "run").split(": ", 2)[2]
    words = shlex.split(line)
    assert words[:2] == ["bench", "run"]
    after = _bench(*words[1:])
    assert after.exit_code == 0, after.output
    assert json.loads(after.stdout) == json.loads(before.stdout)


_CASES = ("case-a", "case-b", "case-c")


@pytest.mark.parametrize(
    ("names", "flag", "pattern", "selected"),
    [
        (_CASES, "--exclude", "case-[a,b]", ["case-c"]),
        (_CASES, "--include", "case-[a,b]", ["case-a", "case-b"]),
        (_CASES, "--exclude", "case-[!a,b]", ["case-a", "case-b"]),
        (_CASES, "--include", "case-[!a,b]", ["case-c"]),
        (_CASES, "--exclude", "case-[]a,b]", ["case-c"]),
        (_CASES, "--include", "case-[]a,b]", ["case-a", "case-b"]),
    ],
)
def test_the_translated_glob_selects_the_same_task_names(
    tmp_path, names, flag, pattern, selected
) -> None:
    """Guards the fix and D1-RC2-A-1, the task filters 90135403 translated."""
    tasks = tmp_path / "tasks"
    for name in names:
        _task(tasks, name)
    before = _bench(
        "eval", "run", "--tasks-dir", str(tasks), "--agent", "oracle",
        flag, pattern, "--dry-run",
    )  # fmt: skip
    assert before.exit_code == 0, before.output
    before_plan = json.loads(before.stdout)
    assert before_plan["tasks"] == selected
    line = _one_note(before, "run").split(": ", 2)[2]
    words = shlex.split(line)
    assert words[:2] == ["bench", "run"]
    after = _bench(*words[1:])
    assert after.exit_code == 0, after.output
    assert json.loads(after.stdout) == before_plan


@pytest.mark.parametrize(
    ("names", "flag", "pattern", "selected"),
    [
        # 0.8 takes a value whole and also matches it as an exact name, which
        # no bench run spelling keeps: the escaped a[,]b would select both.
        (("a,b", "a[,]b", "c"), "--include", "a,b", ["a,b"]),
        (("a,b", "a[,]b", "c"), "--exclude", "a,b", ["a[,]b", "c"]),
        (("[a,b", "[[]a[,]b", "c"), "--include", "[a,b", ["[a,b"]),
        (("[a,b", "[[]a[,]b", "c"), "--exclude", "[a,b", ["[[]a[,]b", "c"]),
        (("a ", "a[ ]", "c"), "--include", "a ", ["a "]),
        (("a ", "a[ ]", "c"), "--exclude", "a ", ["a[ ]", "c"]),
        (("a", "b", "c"), "--exclude", "a,b", ["a", "b", "c"]),
        (_CASES, "--exclude", "case-a,case-b", list(_CASES)),
        (("x-1", "y", "x-1,y"), "--include", "x-*,y", ["x-1,y"]),
        ((*_CASES, " case-a"), "--include", " case-a", [" case-a"]),
    ],
)
def test_a_filter_bench_run_would_split_keeps_the_old_line(
    tmp_path, names, flag, pattern, selected
) -> None:
    """An escaped filter changed which names it selects."""
    tasks = tmp_path / "tasks"
    for name in names:
        _task(tasks, name)
    before = _bench(
        "eval", "run", "--tasks-dir", str(tasks), "--agent", "oracle",
        flag, pattern, "--dry-run",
    )  # fmt: skip
    assert before.exit_code == 0, before.output
    assert sorted(json.loads(before.stdout)["tasks"]) == sorted(selected)
    note = _one_note(before, "run")
    assert f"({flag} {shlex.quote(pattern)} only as `bench eval run`)" in note


def test_an_empty_task_filter_keeps_the_old_line(tmp_path) -> None:
    """0.8 selects no task for --include ''; bench run would drop the filter."""
    tasks = tmp_path / "tasks"
    _task(tasks)
    result = _bench(
        "eval", "run", "--tasks-dir", str(tasks), "--agent", "oracle",
        "--include", "", "--dry-run",
    )  # fmt: skip
    note = _one_note(result, "run")
    assert "(--include '' only as `bench eval run`)" in note


# --- 0.8 groups alone, mistyped words, unknown commands and flags --------------


@pytest.mark.parametrize(
    ("group", "rows"),
    [
        (
            "eval",
            [
                "eval run, create",
                "bench run",
                "eval list, inspect, compare",
                "bench results",
            ],
        ),
        ("tasks", ["tasks init", "bench new", "tasks migrate", "bench import"]),
        ("agent", ["agent list, show", "bench agents"]),
    ],
)
def test_a_0_8_group_alone_prints_its_map(group, rows) -> None:
    result = _bench(group)
    assert result.exit_code == 2
    assert result.stdout == ""
    assert f"`bench {group}` was BenchFlow 0.8's" in result.stderr
    for row in rows:
        assert row in result.stderr
    assert _bench(group, "--help").exit_code == 0


@pytest.mark.parametrize(
    ("word", "step"),
    [
        ("compare", "`bench results <job> <job>` compares two jobs"),
        ("list", "`bench results` lists jobs"),
        ("ls", "`bench results` lists jobs"),
        ("status", "`bench results` lists jobs"),
        ("open", "`bench view <job>`"),
        ("init", "`bench new <name>` creates a task"),
        ("version", "`bench --version`"),
        ("update", "uv tool install --python 3.12 --upgrade benchflow"),
        ("upgrade", "uv tool install --python 3.12 --upgrade benchflow"),
        ("uninstall", "uv tool uninstall benchflow"),
        ("remove", "uv tool uninstall benchflow"),
        ("login", "`bench agents <name>`"),
        ("clean", "`bench sandbox cleanup`"),
    ],
)
def test_a_mistyped_word_gets_a_next_step(word, step) -> None:
    result = _bench(word)
    assert result.exit_code == 2
    text = _flat(result.output)
    assert f"bench has no `{word}` command" in text
    assert "next: " in text and step in text


@pytest.mark.parametrize(
    ("args", "step"),
    [
        (["frobnicate"], "next: `bench --help` lists the commands"),
        (["eval", "frobnicate"], "next: `bench eval --help` lists its commands"),
        (["tasks", "frobnicate"], "next: `bench tasks --help` lists its commands"),
        (["--frobnicate"], "next: `bench --help` lists its flags"),
        (["run", "--frobnicate"], "next: `bench run --help-all` lists its flags"),
        (
            ["eval", "run", "--frobnicate"],
            "next: `bench eval run --help` lists its flags",
        ),
        (["results", "--frobnicate"], "next: `bench results --help` lists its flags"),
    ],
)
def test_an_unknown_command_or_flag_exits_2_with_a_next_step(args, step) -> None:
    result = _bench(*args)
    assert result.exit_code == 2, result.output
    assert step in _flat(result.output)


# --- bench run: the oracle without -a -----------------------------------


@pytest.fixture
def eval_run_calls(monkeypatch) -> list[dict[str, Any]]:
    import benchflow.cli.main as main

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(main, "eval_run", lambda **kw: calls.append(kw))
    return calls


def test_bench_run_without_an_agent_runs_the_oracle(tmp_path, eval_run_calls) -> None:
    task = _task(tmp_path)
    result = _bench("run", str(task))
    assert result.exit_code == 0, result.output
    (call,) = eval_run_calls
    assert call["agent"] == "oracle" and call["tasks_dir"] == task
    assert call["skill_mode"] == "no-skill"


@pytest.mark.parametrize(
    "agent", ["", " ", "\t\n"], ids=["empty", "space", "whitespace"]
)
@pytest.mark.parametrize("flags", [[], ["-m", "gpt-5.5"]], ids=["alone", "model"])
def test_bench_run_rejects_a_blank_agent_before_dispatch(
    tmp_path, eval_run_calls, agent, flags
) -> None:
    """Guards the blank-agent oracle fallback from 90135403."""
    task = _task(tmp_path)
    result = _bench("run", str(task), "-a", agent, *flags)
    assert result.exit_code == 2, result.output
    assert eval_run_calls == []
    text = _flat(result.output)
    assert "--agent must not be empty or whitespace-only" in text
    assert "next: pass -a <agent>" in text


@pytest.mark.parametrize("agent", [" codex ", "\tcodex\n"], ids=["space", "tab"])
def test_bench_run_strips_a_padded_agent(tmp_path, eval_run_calls, agent) -> None:
    """A padded name is the registry name, not a raw ACP command."""
    task = _task(tmp_path)
    result = _bench("run", str(task), "-a", agent)
    assert result.exit_code == 0, result.output
    (call,) = eval_run_calls
    assert call["agent"] == "codex"


@pytest.mark.parametrize(
    ("flags", "named"),
    [
        (["-m", "gpt-5"], "-m gpt-5 names a model"),
        (["--model", "gpt-5"], "-m gpt-5 names a model"),
        (["-e", "high"], "--effort high is for model agents"),
        (["--harness", "headless"], "--harness headless is for model agents"),
    ],
)
def test_bench_run_refuses_model_flags_without_an_agent(
    tmp_path, eval_run_calls, flags, named
) -> None:
    task = _task(tmp_path)
    result = _bench("run", str(task), *flags)
    assert result.exit_code == 2, result.output
    assert eval_run_calls == []
    text = _flat(result.output)
    assert named in text and "without -a, bench run runs the oracle" in text
    assert "next: add -a <agent>" in text
    allowed = _bench("run", str(task), "-a", "codex", *flags)
    assert allowed.exit_code == 0, allowed.output


@pytest.mark.parametrize("flag", ["--include", "--exclude"])
@pytest.mark.parametrize(
    ("values", "patterns"),
    [
        ([" case-[a,b], case-c ", "case-d"], ["case-[a,b]", "case-c", "case-d"]),
        (["case-[[],case-c"], ["case-[[]", "case-c"]),
        (["case-[a,b"], ["case-[a", "b"]),
        ([",case-a,,", " "], ["case-a"]),
    ],
)
def test_bench_run_splits_only_commas_outside_glob_brackets(
    tmp_path, eval_run_calls, flag, values, patterns
) -> None:
    """Guards the glob-bracket fix to 90135403 while keeping comma lists."""
    task = _task(tmp_path)
    args = [arg for value in values for arg in (flag, value)]
    result = _bench("run", str(task), *args)
    assert result.exit_code == 0, result.output
    (call,) = eval_run_calls
    assert call[flag.removeprefix("--")] == patterns


def test_bench_run_takes_rusts_flags(tmp_path, eval_run_calls) -> None:
    task = _task(tmp_path)
    result = _bench(
        "run", str(task), "-a", "codex", "-m", "gpt-5", "-e", "high",
        "--harness", "headless", "--env", "K=V", "--task-skills",
        "-i", "a*,b", "-x", "c", "-n", "2", "-s", "docker", "-j", "3",
        "--jobs-dir", "jd", "--timeout-multiplier", "2", "--idle-timeout", "10m",
        "--fail-under", "0.5", "--save-outputs", "--extra-instruction", "hi",
        "--job", "j", "--fresh", "--dry-run", "-q", "--verifier-sandbox", "separate",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    (call,) = eval_run_calls
    assert call == {
        "tasks_dir": task,
        "dataset": None,
        "registry": None,
        "agent": "codex",
        "model": "gpt-5",
        "reasoning_effort": "high",
        "harness": "native",
        "agent_env": ["K=V"],
        "skill_mode": "with-skill",
        "include": ["a*", "b"],
        "exclude": ["c"],
        "n_tasks": 2,
        "environment": "docker",
        "concurrency": 3,
        "warm_pool": None,
        "jobs_dir": "jd",
        "timeout_multiplier": 2.0,
        "agent_idle_timeout": "600",
        "fail_under": 0.5,
        "freeze_workspace": True,
        "extra_instruction": "hi",
        "verifier_sandbox": "separate",
        "job_name": "j",
        "fresh": True,
        "dry_run": True,
        "quiet": True,
    }


def test_bench_run_reads_a_repository_target(tmp_path, eval_run_calls) -> None:
    for target in (
        "org/repo/tasks@main",
        "https://github.com/org/repo/tree/main/tasks",
    ):
        assert _bench("run", target).exit_code == 0
    assert [
        (c["source_repo"], c["source_path"], c["source_ref"]) for c in eval_run_calls
    ] == [("org/repo", "tasks", "main")] * 2


def test_bench_run_refuses_what_it_cannot_run(tmp_path, eval_run_calls) -> None:
    task = _task(tmp_path)
    for args in (
        [],
        [str(task), str(task)],
        [str(task), "--dataset", "x@1"],
        ["./no-such-folder"],
        [str(task), "--idle-timeout", "soon"],
        [str(task), "-a", "codex", "--harness", "fast"],
    ):
        result = _bench("run", *args)
        assert result.exit_code == 2, (args, result.output)
        assert "next:" in result.output, args
    assert eval_run_calls == []


# --- bench results and bench agents -------------------------------------------


def test_bench_results_lists_reads_or_compares_by_how_many_jobs(tmp_path) -> None:
    a = _job(tmp_path, "a", {"t1": 1.0, "t2": 0.0})
    b = _job(tmp_path, "b", {"t1": 1.0, "t2": 1.0})
    listed = _bench("results", str(tmp_path))
    assert listed.exit_code == 0 and "Jobs" in listed.stdout, listed.output
    one = _bench("results", str(a), "--json")
    assert json.loads(one.stdout)["kind"] == "benchflow.job"
    two = _bench("results", str(a), str(b), "--json")
    assert json.loads(two.stdout)["kind"] == "benchflow.comparison"
    three = _bench("results", str(a), str(b), str(a))
    assert three.exit_code == 2 and "next:" in three.output


def test_bench_results_lists_a_folder_of_unfinished_jobs(tmp_path) -> None:
    # Jobs that were interrupted (or still run) have no summary.json yet. Their
    # folder is still a folder of jobs: listed, as `bench eval list` lists it,
    # never read as one job whose pass rate pools both agents' trials.
    flat = tmp_path / "flat"
    _trial(flat / "a", "t1", agent="codex-acp", reward=1.0)
    _trial(flat / "b", "t1", agent="claude-agent-acp", reward=0.0)
    nested = tmp_path / "nested"
    _trial(nested / "exp-a" / "2026-01-01__12-00-00", "t1", reward=1.0)
    nested_trial = _trial(nested / "exp-b" / "2026-01-01__12-00-00", "t1", reward=0.0)
    # A trial writes a results.jsonl of its own; that is not an export.
    for trial in (flat / "a" / "t1__abcd1234", nested_trial):
        (trial / "results.jsonl").write_text(_ROWS)
    for folder, names in ((flat, ("a", "b")), (nested, ("exp-a", "exp-b"))):
        listed = _bench("results", str(folder))
        assert listed.exit_code == 0, listed.output
        assert "Jobs" in listed.stdout and "passed" not in listed.stdout
        assert all(name in listed.stdout for name in names), listed.stdout
        refused = _bench("results", str(folder), "--json")
        assert refused.exit_code == 2 and "next:" in refused.output
    # One unfinished job, its trials right under it, is still read as a job.
    one = _bench("results", str(flat / "a"), "--json")
    assert json.loads(one.stdout)["kind"] == "benchflow.job", one.output


_ROWS = "".join(
    json.dumps({"info": {"task_name": task}, "reward": reward}) + "\n"
    for task, reward in (("t1", 1.0), ("t2", 0.0))
)


def test_bench_results_reads_an_export_as_one_job(tmp_path) -> None:
    # A vf-eval or hosted-env output dir, or a copied-out results file, keeps
    # its rows in results.jsonl and has no trial folders: `bench results` reads
    # it as `bench eval inspect` does, never lists it as an empty folder of jobs.
    top = tmp_path / "top"
    top.mkdir()
    (top / "results.jsonl").write_text(_ROWS)
    deep = tmp_path / "deep" / "evals" / "env--model" / "run-1"
    deep.mkdir(parents=True)
    (deep / "results.jsonl").write_text(_ROWS)
    for folder in (top, tmp_path / "deep"):
        read = _bench("results", str(folder), "--json")
        assert read.exit_code == 0, read.output
        doc = json.loads(read.stdout)
        assert doc["kind"] == "benchflow.job", read.output
        assert sorted(t["task_name"] for t in doc["trials"]) == ["t1", "t2"]
        shown = _bench("results", str(folder))
        assert shown.exit_code == 0 and "Jobs" not in shown.stdout
        assert "1/2" in _flat(shown.stdout), shown.stdout
    # Six levels down, as vf-eval writes it from an environments root.
    envs = tmp_path / "envs"
    rows = envs / "environments" / "foo" / "outputs" / "evals" / "foo--m" / "run-1"
    rows.mkdir(parents=True)
    (rows / "results.jsonl").write_text(_ROWS)
    read = _bench("results", str(envs), "--json")
    assert read.exit_code == 0 and json.loads(read.stdout)["kind"] == "benchflow.job"


def test_bench_results_lists_finished_jobs_whose_folders_have_no_record(
    tmp_path,
) -> None:
    # A finished job writes a job-level results.jsonl next to its trials; that
    # is not an export. Finished jobs under folders with no job record of their
    # own (copied out of their jobs folder, from `bench branch`, or three levels
    # down) are listed, never pooled into one job. So is a folder of such jobs
    # that also holds one copied export.
    archive = tmp_path / "archive"
    runs = tmp_path / "runs"
    for root, path, agent, reward in (
        (archive, ("exp-a",), "codex-acp", 1.0),
        (archive, ("exp-b",), "claude-agent-acp", 0.0),
        (runs, ("d1", "jobs"), "codex-acp", 1.0),
        (runs, ("d2", "jobs"), "claude-agent-acp", 0.0),
    ):
        job = root.joinpath(*path, "2026-01-01__12-00-00")
        _trial(job, "t1", agent=agent, reward=reward)
        for record in ("evaluation.json", "summary.json"):
            (job / record).write_text(json.dumps({"total": 1}))
        (job / "results.jsonl").write_text(_ROWS)
        if root is runs:
            (job.parent / "summary.json").write_text(json.dumps({"total": 1}))
    mixed = tmp_path / "mixed"
    _trial(mixed / "a", "t1", agent="codex-acp", reward=1.0)
    _trial(mixed / "b", "t1", agent="claude-agent-acp", reward=0.0)
    (mixed / "vf").mkdir()
    (mixed / "vf" / "results.jsonl").write_text(_ROWS)
    for folder, names in (
        (archive, ("exp-a", "exp-b")),
        (runs, ("d1", "d2")),
        (mixed, ("a", "b")),
    ):
        listed = _bench("results", str(folder))
        assert listed.exit_code == 0, listed.output
        assert "Jobs" in listed.stdout and "passed" not in listed.stdout
        assert all(name in listed.stdout for name in names), listed.stdout
        refused = _bench("results", str(folder), "--json")
        assert refused.exit_code == 2 and "next:" in refused.output, refused.output


def test_bench_agents_lists_or_shows(tmp_path, monkeypatch) -> None:
    (tmp_path / "agents").mkdir()
    monkeypatch.setenv("BENCHFLOW_AGENTS_SOURCE", str(tmp_path / "agents"))
    listed = _bench("agents")
    assert listed.exit_code == 0 and "Registered Agents" in listed.stdout
    shown = _bench("agents", "codex")
    assert shown.exit_code == 0 and "codex-acp" in shown.stdout
    assert "note:" not in listed.output + shown.output


# --- parity with Rust 0.8.2 ---------------------------------------------------


def _not_ported() -> dict[str, str]:
    """cli/verbs.py's list of Rust commands not ported, with why (none if absent)."""
    if importlib.util.find_spec("benchflow.cli.verbs") is None:
        return {}
    from benchflow.cli.verbs import NOT_PORTED

    return NOT_PORTED


def _is_command(words: list[str]) -> bool:
    """Whether `bench <words>` names a command (not a command with an argument)."""
    import click
    import typer.main
    from typer.core import TyperGroup

    command: click.Command = typer.main.get_command(app)
    for word in words:
        if not isinstance(command, TyperGroup):
            return False
        sub = command.get_command(typer.Context(command), word)
        if sub is None:
            return False
        command = sub
    return _bench(*words, "--help").exit_code == 0


@pytest.mark.parametrize("command", RUST_082_COMMANDS)
def test_each_rust_082_command_is_here_or_listed_with_a_reason(command) -> None:
    here = _is_command(command.split())
    reason = _not_ported().get(command, "")
    assert here or reason.strip(), (
        f"Rust 0.8.2's `bench {command}` is not a Python command and cli/verbs.py's "
        "NOT_PORTED does not say why"
    )
    assert not (here and reason), (
        f"`bench {command}` is a Python command now: drop it from NOT_PORTED"
    )


def test_not_ported_lists_only_rust_commands() -> None:
    assert set(_not_ported()) <= set(RUST_082_COMMANDS)


# --- typer 0.26 and later run on their own copy of click ----------------------


@pytest.fixture
def foreign_click(monkeypatch) -> None:
    """click as the CLI sees it next to typer 0.26 or later (a fresh install).

    Those typers carry their own click: the groups, contexts and usage errors
    of a typer app are not click's classes, and click's context stack stays
    empty. The lock pins a typer that runs on click, so this swaps the verb
    modules' ``click`` for one whose classes and context are not typer's: code
    that names click's classes where it needs typer's fails here as it does
    after ``uv tool install benchflow``.
    """
    import types

    import click

    import benchflow.cli._lazy as lazy
    import benchflow.cli.main as main
    import benchflow.cli.new_verbs as new_verbs

    class UsageError(Exception):
        pass

    class NoSuchOption(UsageError):
        pass

    class Group:
        pass

    def get_current_context(silent: bool = False) -> None:
        if not silent:
            raise RuntimeError("There is no active click context.")

    foreign = types.ModuleType("click")
    vars(foreign).update(vars(click))
    vars(foreign).update(
        UsageError=UsageError,
        NoSuchOption=NoSuchOption,
        Group=Group,
        get_current_context=get_current_context,
    )
    for module in (lazy, main, new_verbs):
        monkeypatch.setattr(module, "click", foreign)


@pytest.mark.usefixtures("foreign_click")
def test_the_verbs_run_on_a_typer_with_its_own_click(tmp_path, monkeypatch) -> None:
    (tmp_path / "agents").mkdir()
    monkeypatch.setenv("BENCHFLOW_AGENTS_SOURCE", str(tmp_path / "agents"))
    job = _job(tmp_path / "jobs", "a", {"t1": 1.0})
    for args, shown in (
        (["doctor", "--help"], "Usage: bench doctor"),
        (["new", "--help"], "Usage: bench new"),
        (["eval", "inspect", str(job), "--json"], '"benchflow.job"'),
        (["results", str(tmp_path / "jobs")], "Jobs"),
        (["results", str(job), "--json"], '"benchflow.job"'),
        (["results", str(job), str(job), "--json"], '"benchflow.comparison"'),
        (["agents"], "Registered Agents"),
        (["agents", "codex"], "codex-acp"),
    ):
        result = _bench(*args)
        assert result.exit_code == 0, (args, result.output, result.exception)
        assert shown in result.stdout, (args, result.stdout)
    for args, step in (
        (["compare"], "`bench results <job> <job>` compares two jobs"),
        (["eval", "view", "--bogus"], "next: `bench eval view --help` lists its flags"),
        (["run", "--model", "x"], "next:"),
    ):
        result = _bench(*args)
        assert result.exit_code == 2, (args, result.output, result.exception)
        assert step in _flat(result.output), (args, result.output)
