"""The command line's craft and output: dogfood p1 (a first-time user) and p5
(CLI craft against 0.7.11) on the 0.8.3 release candidate.

Each test names the finding it checks (``DFR-p1-4`` is p1's finding 4).
"""

from __future__ import annotations

import ast
import asyncio
import io
import logging
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import click
import pytest
from rich.console import Console
from typer.testing import CliRunner

from benchflow.cli.main import app

HELLO = Path(__file__).parent / "examples" / "hello-world-task"
SRC = Path(__file__).resolve().parents[1] / "src" / "benchflow"
runner = CliRunner()


def _flat(text: str) -> str:
    return " ".join(click.unstyle(text).split())


def _squash(text: str) -> str:
    """Text with no whitespace and no box lines: a phrase Rich wrapped in a
    help panel still matches."""
    return re.sub(r"[\s─│╭╮╯╰]", "", click.unstyle(text))


@pytest.fixture
def info_root():
    """The root logger at INFO, as the CLI sets it, restored afterwards."""
    root = logging.getLogger()
    level = root.level
    root.setLevel(logging.INFO)
    yield root
    root.setLevel(level)


# --- output noise -----------------------------------------------------------


def test_the_cli_logs_no_http_request_lines_at_its_default_level():
    """DFR-p1-4: httpx's "HTTP Request:" lines showed at INFO, the CLI's level."""
    code = (
        "import logging, benchflow.cli.main; "
        "print(logging.getLogger('httpx').getEffectiveLevel())"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    assert out.split()[-1] == str(logging.WARNING)


def test_the_verifier_runtime_fetch_logs_no_http_request_line(
    info_root, monkeypatch, caplog
):
    """DFR-p1-4, Python API: a run at INFO printed one line per PyPI fetch."""
    import httpx

    from benchflow.sandbox import _verifier_runtime

    monkeypatch.setattr(logging.getLogger("httpx"), "level", logging.NOTSET)

    def fake_get(url, **_kwargs):
        logging.getLogger("httpx").info('HTTP Request: GET %s "HTTP/1.1 200 OK"', url)
        return SimpleNamespace(status_code=200)

    monkeypatch.setattr(httpx, "get", fake_get)
    caplog.set_level(logging.DEBUG, logger="benchflow")
    _verifier_runtime._http_get("https://pypi.org/pypi/pytest/8.4.2/json")
    assert not [r for r in caplog.records if "HTTP Request" in r.getMessage()]


_ID = re.compile(r"PY\d-[A-Z0-9]")


def _one_line_strings(tree: ast.AST) -> list[ast.Constant]:
    """String constants that are no docstring and no multi-line script text."""
    docs = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(
            node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        )
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docs
        and "\n" not in node.value
    ]


def test_no_message_carries_an_internal_ticket_id():
    """DFR-p1-5: "" and the like reached run logs and errors."""
    found = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in _one_line_strings(tree):
            if _ID.search(node.value):
                found.append(f"{path.relative_to(SRC)}:{node.lineno}: {node.value}")
    assert not found, "\n".join(found)


@pytest.mark.parametrize(
    ("module", "start"),
    [
        ("sandbox/lockdown.py", "Protected runtime ready in "),
        ("sandbox/lockdown.py", "Verifier hardening: "),
        ("task/verifier_core.py", "No marker for the pytest plugin guard"),
        ("sandbox/_verifier_runtime.py", "verifier runtime (%s): %d wheels"),
    ],
)
def test_jargon_lines_are_debug_lines(module, start):
    """DFR-p1-5: protected-runtime and plugin-guard lines printed on every run."""
    tree = ast.parse((SRC / module).read_text(encoding="utf-8"))
    levels = [
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and str(node.args[0].value).startswith(start)
    ]
    assert levels == ["debug"]


def test_a_task_image_without_python_is_a_trial_warning():
    """DFR-p1-5: the hardening note goes to result.json's warnings, not the log."""
    tree = ast.parse((SRC / "sandbox/lockdown.py").read_text(encoding="utf-8"))
    (harden,) = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_harden_steps"
    ]
    called = {
        node.func.id
        for node in ast.walk(harden)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "report_trial_warning" in called


def _fake_rollout(task: str):
    from benchflow.rollout import Rollout

    rollout = Rollout.__new__(Rollout)
    rollout._config = SimpleNamespace(task_path=Path("tasks") / task, purpose="task")
    rollout._rollout_dir = None
    return rollout


def test_a_trials_log_lines_start_with_its_task():
    """DFR-p1-9: at -j 4 "Running verifier..." printed four times, no task."""
    code = """
import asyncio, logging
from pathlib import Path
from types import SimpleNamespace
import benchflow.cli.main
from benchflow.rollout import Rollout

def rollout(task):
    r = Rollout.__new__(Rollout)
    r._config = SimpleNamespace(task_path=Path("tasks") / task, purpose="task")
    r._rollout_dir = None
    async def steps():
        logging.getLogger("benchflow.x").info("Running verifier...")
        await asyncio.sleep(0)
        logging.getLogger("benchflow.x").warning("Oracle solve.sh exited with rc=127")
    r._run_with_scoring = steps
    return r

async def main():
    await asyncio.gather(rollout("fix-git").run(), rollout("regex-log").run())
    logging.getLogger("benchflow.x").info("Job complete")

asyncio.run(main())
"""
    err = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stderr.splitlines()
    assert "[fix-git] Running verifier..." in err
    assert "[regex-log] Running verifier..." in err
    assert "[regex-log] Oracle solve.sh exited with rc=127" in err
    assert "Job complete" in err  # a job's own line has no task


def test_a_prepared_sandboxs_lines_name_its_task():
    """DFR-p1-9: the warm pool prepares a trial's sandbox in a task of its own,
    whose "Setting up sandbox user: agent" printed with no task."""
    code = """
import asyncio, logging
from pathlib import Path
from types import SimpleNamespace
import benchflow.cli.main
from benchflow.rollout import Rollout

r = Rollout.__new__(Rollout)
r._config = SimpleNamespace(task_path=Path("tasks") / "kv-store-grpc", purpose="task")
r._prestart = None
r._phase = "created"
async def steps():
    logging.getLogger("benchflow.x").info("Setting up sandbox user: agent")
r._prestart_steps = steps
r._note_prestart_done = lambda task: None

async def main():
    await r.begin_prestart()

asyncio.run(main())
"""
    err = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stderr.splitlines()
    assert "[kv-store-grpc] Setting up sandbox user: agent" in err


def test_a_warning_replayed_after_the_dashboard_names_its_task(monkeypatch):
    """DFR-p5-16: "Oracle solve.sh exited with rc=127" came after the last frame
    with no task."""
    from benchflow.cli import _live_progress

    out = io.StringIO()
    monkeypatch.setattr(
        _live_progress, "console", Console(file=out, width=200, color_system=None)
    )
    rollout = _fake_rollout("git-multibranch")

    async def steps():
        logging.getLogger("benchflow.x").warning("Oracle solve.sh exited with rc=127")

    rollout._run_with_scoring = steps
    with _live_progress.quiet_root_logging():
        asyncio.run(rollout.run())
    assert "[git-multibranch] Oracle solve.sh exited with rc=127" in out.getvalue()


def test_the_eta_does_not_grow_while_the_last_tasks_run(monkeypatch):
    """DFR-p5-15: at 3/4 the ETA went 0:09, 0:10, ... 0:37 as the last task ran."""
    from benchflow.cli import _live_progress
    from benchflow.cli._live_progress import LiveEvalProgress

    clock = [1000.0]
    monkeypatch.setattr(_live_progress.time, "monotonic", lambda: clock[0])
    live = LiveEvalProgress(
        Console(file=io.StringIO()),
        label="tb2",
        agent="oracle",
        model=None,
        sandbox="docker",
    )
    live.on_plan(4, 0, 4)
    for name in ("a", "b", "c", "d"):
        live.on_task_start(name)
    clock[0] += 10
    for name in ("a", "b", "c"):
        live.on_result(name, SimpleNamespace(score_outcome="passed"))
    clock[0] += 70

    record = Console(file=io.StringIO(), width=200, record=True, color_system=None)
    record.print(live)
    text = record.export_text()
    assert "1 left, running 1:20" in text
    assert "ETA" not in text


def test_quiet_prints_less_than_no_flag(info_root, monkeypatch, caplog, tmp_path):
    """DFR-p5-2: -q swapped the dashboard for every log line of the run."""
    import benchflow.cli.main as cli_main

    # -q sets these for the process; the test gives them back.
    monkeypatch.setenv("BENCHFLOW_NO_PROGRESS", "0")
    monkeypatch.setenv("BENCHFLOW_PROGRESS", "auto")

    def fake_run(*_args, **_kwargs):
        logging.getLogger("benchflow.x").info("Running verifier...")
        logging.getLogger("benchflow.x").warning("a warning")

    monkeypatch.setattr(cli_main, "run_batch_eval", fake_run)
    result = runner.invoke(
        app, ["run", str(HELLO), "-q", "--jobs-dir", str(tmp_path / "jobs")]
    )
    assert result.exit_code == 0, result.output
    messages = [r.getMessage() for r in caplog.records]
    assert "a warning" in messages
    assert "Running verifier..." not in messages


# --- help and flags ---------------------------------------------------------


def test_run_help_shows_the_flags_the_readme_uses():
    """DFR-p5-1: --source-repo and --source-path were only in --help-all."""
    out = runner.invoke(app, ["run", "--help"], terminal_width=200).output
    for flag in ("--source-repo", "--source-path", "--config"):
        assert flag in out, flag
    full = runner.invoke(app, ["run", "--help-all"], terminal_width=200).output
    assert full.count("--source-repo") == out.count("--source-repo")


@pytest.mark.parametrize(
    ("command", "internal"),
    [
        (["eval", "run"], "S axis"),
        (["eval", "run"], "S-axis"),
        (["eval", "run"], "C-axis"),
        (["eval", "run"], "T/A/M/S/R"),
        (["cache", "list"], "Rust's bench"),
        (["eval", "continue"], "Tests use"),
        (["skills", "eval"], "benchflow skills eval ./my-skill"),
        (["branch"], "tests/examples"),
        (["tasks", "generate"], "``"),
        (["tasks", "generate"], "Examples::"),
        (["tasks", "generate"], "task.toml"),
    ],
)
def test_help_pages_carry_no_internal_text(command, internal):
    """DFR-p5-12: internal names, test models and repo paths in help pages."""
    result = runner.invoke(app, [*command, "--help"], terminal_width=200)
    assert result.exit_code == 0, result.output
    assert _squash(internal) not in _squash(result.output)


def test_cache_only_help_states_the_real_default():
    """DFR-p5-21: it read as if build-cache alone were the default."""
    out = _flat(runner.invoke(app, ["cache", "list", "--help"]).output)
    assert "default: images and build-cache" in out
    assert "build-cache (the default)" not in out


def test_eval_help_marks_the_old_spellings():
    """DFR-p5-11: `bench eval --help` listed run, list, view ... as current."""
    out = _squash(runner.invoke(app, ["eval", "--help"]).output)
    for old, new in (
        ("run", "run"),
        ("list", "results"),
        ("view", "view"),
        ("regrade", "regrade"),
        ("branch", "branch"),
        ("inspect", "results"),
        ("compare", "results"),
    ):
        assert _squash(f"(old name of `bench {new}`)") in out, old
    assert out.count(_squash("(old name of")) == 7  # smoke, metrics, resume ... are not


def test_eval_run_takes_bench_runs_short_flags_and_durations(tmp_path):
    """DFR-p5-11: `bench eval run` had no -a/-s/-j/-i/-x/-n and only seconds."""
    result = runner.invoke(
        app,
        [
            "eval",
            "run",
            "--tasks-dir",
            str(HELLO),
            "-a",
            "oracle",
            "-s",
            "docker",
            "-j",
            "2",
            "-i",
            "hello*",
            "-n",
            "1",
            "--idle-timeout",
            "10m",
            "--jobs-dir",
            str(tmp_path / "jobs"),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    # -a counts as naming the agent: the note adds no default agent.
    assert "--agent claude-agent-acp" not in result.stderr


def test_eval_run_note_is_skipped_when_the_line_does_not_parse():
    """DFR-p5-20: the note kept the typo and added --agent claude-agent-acp."""
    bad = runner.invoke(
        app, ["eval", "run", "--tasks-dir", str(HELLO), "--agnet", "oracle"]
    )
    assert bad.exit_code == 2
    assert "note:" not in bad.stderr
    assert "No such option" in bad.stderr
    good = runner.invoke(
        app,
        ["eval", "run", "--tasks-dir", str(HELLO), "--agent", "oracle", "--dry-run"],
    )
    assert "note: `bench eval run` is now `bench run`" in good.stderr


def test_check_runtime_capability_defaults_to_docker():
    """DFR-p5-14: the help gave docker as the default, the check required it."""
    result = runner.invoke(app, ["check", str(HELLO), "--level", "runtime-capability"])
    assert "requires --sandbox" not in _flat(result.output)
    assert result.exit_code == 0, result.output


# --- exit codes and next steps ----------------------------------------------


@pytest.mark.parametrize(
    "args",
    [
        ["run", str(HELLO), "-j", "0"],
        ["run", str(HELLO), "--fail-under", "2"],
        ["run", str(HELLO), "-s", "dokcer"],
        ["run", str(HELLO), "-a", "claud"],
        ["run", "-d", "terminal-bench"],
        ["eval", "run", "--tasks-dir", "no/such/tasks", "--agent", "oracle"],
        ["eval", "metrics", "no/such/jobs"],
        ["eval", "list", "no/such/jobs"],
        ["check", "no/such/task"],
        ["view", "no/such/job"],
    ],
)
def test_usage_mistakes_exit_2_with_a_next_step(args, tmp_path, monkeypatch):
    """DFR-p5-8: the same kind of mistake exited 1 or 2."""
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, args)
    assert result.exit_code == 2, result.output
    assert "next:" in result.output


def test_a_flag_value_click_refuses_has_a_next_step():
    """DFR-p5-9: `-j abc` printed Click's error with no next step."""
    result = runner.invoke(app, ["run", str(HELLO), "-j", "abc"], prog_name="bench")
    assert result.exit_code == 2
    assert _squash("next: `bench run --help-all` says what each flag takes") in (
        _squash(result.output)
    )


def test_an_unknown_agents_next_step_is_about_agents():
    """DFR-p5-9: it said `bench run --help` lists the flags."""
    result = runner.invoke(app, ["run", str(HELLO), "-a", "claud"])
    assert "next: pass -a with a name `bench agents` lists" in _flat(result.output)
    assert "lists the flags" not in result.output


def test_unknown_agent_detail_exits_2(monkeypatch):
    """DFR-p5-8: `bench agents claud` exited 1."""
    import benchflow.agents.registry as registry

    def unknown(name):
        raise KeyError(f"Unknown agent: {name!r}.")

    monkeypatch.setattr(registry, "resolve_agent", unknown)
    result = runner.invoke(app, ["agents", "claud"])
    assert result.exit_code == 2, result.output


def test_an_include_that_matches_no_task_exits_2(monkeypatch, tmp_path):
    """DFR-p5-8: `-i fix-gti` exited 1."""
    import benchflow.evaluation as evaluation

    class Refusing:
        def __init__(self, *args, **kwargs):
            pass

        async def run(self):
            raise evaluation.EmptyTaskSelectionError(
                "No tasks selected after include/exclude filtering (x). --include "
                "'fix-gti' matches no task (did you mean 'fix-git'?). Refusing to "
                "publish an empty 0/0 summary."
            )

    monkeypatch.setattr(evaluation, "Evaluation", Refusing)
    result = runner.invoke(
        app, ["run", str(HELLO), "-i", "fix-gti", "--jobs-dir", str(tmp_path / "j")]
    )
    assert result.exit_code == 2, result.output


def test_a_dataset_include_that_matches_no_task_exits_2(monkeypatch, tmp_path):
    """DFR-p5-8: `-d terminal-bench@2.0 -i fix-gti` exited 1; its next step
    pointed at a snapshot path, where `bench hub show` lists the tasks."""
    import benchflow._utils.dataset_registry as dataset_registry

    resolved = SimpleNamespace(
        spec="terminal-bench@2.0",
        bench_version=None,
        unpinned_tasks=[],
        task_names={"fix-git", "regex-log"},
        provenance={"resolved_sha": "69671fba"},
        tasks_dir=tmp_path,
    )
    monkeypatch.setattr(
        dataset_registry, "resolve_dataset", lambda spec, registry=None: resolved
    )
    result = runner.invoke(
        app,
        [
            "run",
            "-d",
            "terminal-bench@2.0",
            "-i",
            "fix-gti",
            "--jobs-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 2, result.output
    out = _flat(result.output)
    assert "did you mean 'fix-git'?" in out
    assert "next: `bench hub show terminal-bench@2.0` lists the dataset's tasks" in out


def test_a_missing_sandbox_extra_exits_1_and_names_the_key(monkeypatch, tmp_path):
    """DFR-p5-9 / p1: its next step was `bench run --help`; Daytona's key unnamed."""
    import benchflow.eval_plan as eval_plan

    monkeypatch.setattr(eval_plan, "sandbox_sdk_missing", lambda name: True)
    result = runner.invoke(
        app, ["run", str(HELLO), "-s", "daytona", "--jobs-dir", str(tmp_path / "j")]
    )
    assert result.exit_code == 1, result.output
    out = _flat(result.output)
    assert "DAYTONA_API_KEY" in out
    assert "lists the flags" not in out


def test_a_held_job_lock_says_what_to_do(tmp_path):
    """DFR-p5-9: it said `bench check` validates the tasks."""
    from benchflow._utils.job_lock import JobLock

    job = tmp_path / "jobs" / "2026-10-22__12-05-01"
    job.mkdir(parents=True)
    held = JobLock.acquire(job / ".lock")
    try:
        with pytest.raises(RuntimeError) as refused:
            JobLock.acquire(job / ".lock")
    finally:
        held.close()
    assert "--fresh" in str(getattr(refused.value, "hint", ""))


def test_cache_errors_end_with_a_next_step(monkeypatch):
    """DFR-p5-9: `bench cache` printed `error: ...` with plain print."""
    usage = runner.invoke(app, ["cache", "list", "--only", "nope"])
    assert usage.exit_code == 2
    assert "next: `bench cache --help`" in _flat(usage.output)

    import benchflow.cli.cache as cache

    def broken(kinds, notes):
        raise RuntimeError("daemon unreachable")

    monkeypatch.setattr(cache, "_docker_items", broken)
    failed = runner.invoke(app, ["cache", "list", "-s", "docker"])
    assert failed.exit_code == 1
    assert "next: `bench doctor`" in _flat(failed.output)


# --- hangs and pointers -----------------------------------------------------


def test_an_offline_fetch_says_what_failed_and_what_to_do(monkeypatch, capsys):
    """DFR-p5-10: no network showed git's raw line, then Python's repr."""
    from benchflow._utils import benchmark_repos

    said = (
        "fatal: unable to access 'https://github.com/benchflow-ai/skillsbench.git/': "
        "Could not resolve host: github.com\n"
    )
    monkeypatch.setattr(
        benchmark_repos.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=128, stdout="", stderr=said),
    )
    with pytest.raises(subprocess.CalledProcessError) as failed:
        benchmark_repos._git_quiet(Path("/nonexistent"), "read-tree", "-mu", "HEAD")
    assert str(failed.value).startswith("could not reach GitHub (no network?)")
    assert "Could not resolve host: github.com" in str(failed.value)
    assert "network" in str(getattr(failed.value, "hint", ""))
    assert "returned non-zero exit status" not in str(failed.value)
    assert capsys.readouterr().err == ""


def test_help_and_doctor_point_at_the_real_pages():
    """DFR-p1-12: both pointed at a moved stub page, doctor by a repo path."""
    out = _squash(runner.invoke(app, ["--help"]).output)
    assert "docs/start/quick-start.md" in out
    assert "docs/getting-started.md" not in out
    doctor = (SRC / "doctor.py").read_text(encoding="utf-8")
    assert "see docs/getting-started.md" not in doctor
    assert "blob/main/docs/agents/auth.md" in doctor


# --- defaults and text --------------------------------------------------------


def test_the_contributor_prompt_names_no_event():
    """DFR-p5-7: `bench traj setup --prompt` shipped a hackathon's text."""
    out = runner.invoke(app, ["traj", "setup", "--prompt"]).output.lower()
    for event in ("hackathon", "prize", "72 hours", "re:agent"):
        assert event not in out


def test_review_runs_its_default_agent_without_a_model_flag(monkeypatch, tmp_path):
    """DFR-p5-18: `bench review <job>`: opencode has no registry default model."""
    import benchflow.review as review

    seen = {}

    async def fake_run_reviews(path, **kwargs):
        seen.update(kwargs)
        raise review.ReviewRunError("stop here")

    monkeypatch.setattr(review, "run_reviews", fake_run_reviews)
    # The reviewer's key is checked first (round 2, M1).
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    runner.invoke(app, ["review", str(tmp_path)])
    assert seen["agent"] == "opencode"
    assert seen["model"] == "gemini/gemini-3.5-flash-lite"


def test_sandbox_says_daytona_and_how_a_tool_install_adds_it(monkeypatch):
    """DFR-p5-17: "Local sandbox lifecycle", and a source-checkout hint."""
    import benchflow.cli.sandbox as sandbox

    out = _flat(runner.invoke(app, ["sandbox", "--help"]).output)
    assert "Local sandbox" not in out and "Daytona" in out
    monkeypatch.setattr(sandbox, "_daytona_sdk_available", lambda: False)
    listed = _flat(runner.invoke(app, ["sandbox", "list"], terminal_width=300).output)
    assert "uv tool install --python 3.12 --upgrade 'benchflow[sandbox-daytona]'" in (
        listed
    )


_REGISTRY = [
    {
        "name": "terminal-bench",
        "version": "2.0",
        "description": "Terminal-Bench 2.0",
        "tasks": [{"name": "regex-log"}, {"name": "fix-git"}],
    }
]


@pytest.fixture
def registry(monkeypatch):
    import benchflow.hub.registry as hub_registry

    monkeypatch.setattr(hub_registry, "load_hub_registry", lambda src=None: _REGISTRY)


def test_hub_list_defaults_to_benchflows_datasets(registry):
    """DFR-p1-20: "Environment" column, and a default that needs the prime CLI."""
    out = runner.invoke(app, ["hub", "list"], terminal_width=200)
    assert out.exit_code == 0, out.output
    assert "Dataset" in out.output and "Environment" not in out.output
    assert "terminal-bench" in out.output


@pytest.mark.parametrize(
    "args",
    [
        ["terminal-bench@2.0"],
        ["benchflow:terminal-bench@2.0"],
        ["terminal-bench", "--version", "2.0"],
        ["terminal-bench@2.0", "--provider", "benchflow"],
    ],
)
def test_hub_show_lists_a_datasets_tasks(registry, args):
    """DFR-p1-6: `bench hub show terminal-bench@2.0` said "prime CLI is not
    installed"; `hub show` had no --provider."""
    out = runner.invoke(app, ["hub", "show", *args], terminal_width=200)
    assert out.exit_code == 0, out.output
    text = out.output
    assert "2 task(s)" in text
    assert text.index("fix-git") < text.index("regex-log")
    assert "bench run -d terminal-bench@2.0 -i fix-git" in _flat(text)


def test_hub_show_names_the_versions_of_an_unknown_one(registry):
    out = runner.invoke(app, ["hub", "show", "terminal-bench@9.9"])
    assert out.exit_code == 2
    assert "its versions there: 2.0" in _flat(out.output)
    assert "next:" in out.output


def test_hub_show_keeps_hosted_environments(monkeypatch):
    import benchflow.hosted_env as hosted

    monkeypatch.setattr(hosted, "prime_env_info", lambda ref: "META: general-agent")
    out = runner.invoke(app, ["hub", "show", "primeintellect/general-agent"])
    assert out.exit_code == 0, out.output
    assert "META: general-agent" in out.output
