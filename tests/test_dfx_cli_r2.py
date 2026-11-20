"""The command line's craft and output, round 2: the review of DFX-CLI round 1
(M1-M4, L1-L5). Each test names the finding it checks.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

import click
import pytest
from typer.testing import CliRunner

from benchflow.cli.main import app

HELLO = Path(__file__).parent / "examples" / "hello-world-task"
TRACES = str(Path(__file__).parent / "examples" / "traces" / "minimal-claude.jsonl")
runner = CliRunner()
OFFLINE_NEXT = "check this machine's network (or its proxy), then run the command again"
NO_ROOT_PYTHON = "the task's image has no root-owned python3"


def _flat(text: str) -> str:
    return " ".join(click.unstyle(text).split())


@pytest.fixture
def info_root():
    """The root logger at INFO, as a host process may set it, restored afterwards."""
    root = logging.getLogger()
    level = root.level
    root.setLevel(logging.INFO)
    yield root
    root.setLevel(level)


# --- M1: bench review with no reviewer key ------------------------------------


def test_review_without_its_key_prints_one_error_and_no_traceback(
    monkeypatch, tmp_path
):
    """M1: `bench review <job>` with no GEMINI_API_KEY logged a traceback and an
    empty table before its error line."""
    import benchflow.review as review

    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_USE_VERTEXAI"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    started = []

    async def fake_run_reviews(path, **kwargs):
        started.append(path)
        raise AssertionError("no review starts without the reviewer's key")

    monkeypatch.setattr(review, "run_reviews", fake_run_reviews)
    job = tmp_path / "jobs" / "j"
    job.mkdir(parents=True)
    result = runner.invoke(app, ["review", str(job)])
    assert result.exit_code == 1, result.output
    out = _flat(result.output)
    assert "GEMINI_API_KEY" in out
    assert "next:" in out
    assert "Traceback" not in result.output
    assert "Review:" not in out
    assert started == []


def test_the_review_runner_logs_a_user_error_without_its_traceback(
    monkeypatch, tmp_path, caplog
):
    """M1, Python API: the runner logged MissingCredentialError's traceback."""
    from benchflow.errors import MissingCredentialError
    from benchflow.review import runner as review_runner

    def no_key(*args, **kwargs):
        raise MissingCredentialError("GEMINI_API_KEY required for model 'm'")

    monkeypatch.setattr(review_runner, "assemble_review_task", no_key)
    rollout = tmp_path / "rollout"
    rollout.mkdir()
    rubric = SimpleNamespace(contract="v1", criteria=[])
    config = SimpleNamespace(
        image="i",
        timeout_sec=1,
        open_network=False,
        memory_mb=2048,
        storage_mb=None,
        environment="docker",
        pdf_tools=False,
        submitted_file_rule=False,
        criterion_evidence=False,
    )
    caplog.set_level(logging.ERROR, logger="benchflow")
    trial = asyncio.run(
        review_runner.run_review(
            rollout,
            None,
            rubric,
            tmp_path / "rubric.json",
            config,
            tmp_path / "out",
            deterministic_pass=True,
        )
    )
    assert "GEMINI_API_KEY" in (trial.error or "")
    [record] = [r for r in caplog.records if "Review failed for" in r.getMessage()]
    assert "GEMINI_API_KEY" in record.getMessage()
    assert not record.exc_info


# --- M2: offline fetches on every path ----------------------------------------


def _offline_clone_error():
    import subprocess
    from unittest.mock import patch

    from benchflow._utils import benchmark_repos

    command = [
        "git",
        "clone",
        "https://github.com/laude-institute/terminal-bench-2.git",
    ]
    failure = subprocess.CalledProcessError(
        128, command, stderr="fatal: Could not resolve host: github.com\n"
    )
    with patch.object(benchmark_repos.subprocess, "run", side_effect=failure):
        try:
            benchmark_repos._run_clone(command, "laude-institute", "terminal-bench-2")
        except benchmark_repos.GitCloneError as error:
            return error
    raise AssertionError("the failed clone must raise")


def test_an_offline_dataset_fetch_is_an_error_with_its_next_step(monkeypatch, tmp_path):
    """M2: offline, `-d terminal-bench@2.0` crashed as "unexpected error
    (GitCloneError) ... a bug in BenchFlow"."""
    import benchflow._utils.dataset_registry as dataset_registry

    def offline(spec, registry=None):
        raise _offline_clone_error()

    monkeypatch.setattr(dataset_registry, "resolve_dataset", offline)
    result = runner.invoke(
        app,
        [
            "run",
            "-d",
            "terminal-bench@2.0",
            "-i",
            "fix-git",
            "--jobs-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 1, result.output
    out = _flat(result.output)
    assert "could not reach GitHub (no network?)" in out
    assert f"next: {OFFLINE_NEXT}" in out
    assert "bug in BenchFlow" not in out


def test_git_failures_are_expected_errors(monkeypatch):
    """Guards M2 and N3 after 66495284a: only identified remote failures are expected."""
    import subprocess

    from benchflow._utils import benchmark_repos
    from benchflow.cli._errors import is_expected

    assert is_expected(_offline_clone_error())
    monkeypatch.setattr(
        benchmark_repos.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=128, stderr="fatal: Could not resolve host: github.com\n"
        ),
    )
    with pytest.raises(subprocess.CalledProcessError) as caught:
        benchmark_repos._git_quiet(Path("cache"), "fetch", "origin", "HEAD")
    assert is_expected(caught.value)
    assert caught.value.hint == OFFLINE_NEXT


def test_a_run_configs_offline_source_repo_says_what_to_do(monkeypatch, tmp_path):
    """M2: a run config's source.repo clone failure escaped the config's handler."""
    from benchflow._utils import benchmark_repos

    def offline(**kwargs):
        raise _offline_clone_error()

    monkeypatch.setattr(benchmark_repos, "resolve_source_with_metadata", offline)
    config = tmp_path / "run.yaml"
    config.write_text("source:\n  repo: laude-institute/terminal-bench-2\n")
    result = runner.invoke(
        app,
        ["run", "--config", str(config), "-a", "oracle", "--jobs-dir", str(tmp_path)],
    )
    assert result.exit_code == 1, result.output
    out = _flat(result.output)
    assert "Failed to fetch source repo" in out
    assert f"next: {OFFLINE_NEXT}" in out


def test_an_unreachable_registry_url_says_the_network(monkeypatch):
    """M2: `--registry <URL>` offline said only "Could not read ..."."""
    import benchflow._utils.dataset_registry as dataset_registry

    def unreachable(url, timeout=None):
        raise URLError("[Errno -3] Temporary failure in name resolution")

    monkeypatch.setattr(dataset_registry, "urlopen", unreachable)
    with pytest.raises(dataset_registry.DatasetResolutionError) as refused:
        dataset_registry.load_registry("https://example.invalid/registry.json")
    assert refused.value.hint == OFFLINE_NEXT

    def missing(url, timeout=None):
        raise HTTPError(url, 404, "Not Found", {}, io.BytesIO())

    monkeypatch.setattr(dataset_registry, "urlopen", missing)
    with pytest.raises(dataset_registry.DatasetResolutionError) as not_found:
        dataset_registry.load_registry("https://example.invalid/registry.json")
    assert not_found.value.hint is None


def test_hub_check_offline_says_the_network(monkeypatch):
    """M2: `bench hub check --registry <URL>` offline told the user to pass a
    registry file."""
    import benchflow.hub.registry as hub_registry

    def unreachable(*args, **kwargs):
        raise URLError("[Errno -3] Temporary failure in name resolution")

    monkeypatch.setattr(hub_registry, "check_registry_tasks", unreachable)
    result = runner.invoke(
        app, ["hub", "check", "--registry", "https://example.invalid/r.json"]
    )
    assert result.exit_code == 1, result.output
    assert f"next: {OFFLINE_NEXT}" in _flat(result.output)


# --- M3: the hardening note on every verifier path -----------------------------


def _scoring_rollout(monkeypatch, tmp_path):
    """A Rollout with only what its scoring paths read, its steps stubbed."""
    import benchflow.rollout as rollout_mod
    from benchflow._utils.trial_warnings import report_trial_warning
    from benchflow.rollout import Rollout

    async def nothing(*args, **kwargs):
        return None

    async def verify_reporting(*args, **kwargs):
        report_trial_warning(NO_ROOT_PYTHON)
        return {"reward": 1.0}, None

    monkeypatch.setattr(rollout_mod, "_probe_integrity_access", nothing)
    monkeypatch.setattr(rollout_mod, "capture_terminal_workspace", nothing)
    monkeypatch.setattr(rollout_mod, "collect_rollout_artifacts", nothing)
    monkeypatch.setattr(rollout_mod, "separate_verifier_requested", lambda c: True)
    monkeypatch.setattr(rollout_mod, "run_separate_verifier", verify_reporting)
    monkeypatch.setattr(Rollout, "_scrape_trajectory_if_missing", nothing)
    monkeypatch.setattr(Rollout, "_discard_verifier_warmup", nothing)
    monkeypatch.setattr(Rollout, "_verify_separate_overlap", verify_reporting)
    rollout = Rollout.__new__(Rollout)
    rollout._config = SimpleNamespace(task_path=tmp_path / "regex-log")
    rollout._task = SimpleNamespace(config=None)
    return rollout


@pytest.mark.parametrize("order", [None, "prestart"])
def test_a_separate_verifier_keeps_the_hardening_note(monkeypatch, tmp_path, order):
    """M3: with --verifier-sandbox separate the "no root-owned python3" note
    reached neither the console nor result.json."""
    from benchflow.rollout import Rollout

    monkeypatch.setattr(Rollout, "_separate_verifier_order", lambda self: order)
    rollout = _scoring_rollout(monkeypatch, tmp_path)
    assert asyncio.run(Rollout._score_finished_solver(rollout)) == {"reward": 1.0}
    assert vars(rollout).get("_warnings") == [NO_ROOT_PYTHON]


def test_a_verifier_recovery_keeps_the_hardening_note(monkeypatch, tmp_path):
    """M3: a verifier-only recovery ran outside any warnings sink too."""
    from benchflow._utils.trial_warnings import report_trial_warning
    from benchflow.rollout import Rollout, _verifier_recovery

    async def nothing(*args, **kwargs):
        return None

    async def recover(rollout):
        report_trial_warning(NO_ROOT_PYTHON)
        return {"reward": 1.0}, None

    monkeypatch.setattr(_verifier_recovery, "needs_verifier_recovery", lambda e: True)
    monkeypatch.setattr(
        _verifier_recovery, "recovery_ineligible_reason", lambda r: None
    )
    monkeypatch.setattr(_verifier_recovery, "recover_verifier", recover)
    monkeypatch.setattr(_verifier_recovery, "release_recovery_lease", nothing)
    (tmp_path / "solver.json").write_text("{}")
    rollout = Rollout.__new__(Rollout)
    rollout._rollout_dir = tmp_path
    rollout._review_plan = None
    rollout._phase = "cleaned"
    result = SimpleNamespace(rollout_name="regex-log", verifier_error="verifier_wedge")
    rollout._build_result = lambda **kwargs: result
    monkeypatch.setattr(Rollout, "_require_rollout_dir", lambda self: tmp_path)
    asyncio.run(Rollout._finish_scoring_locked(rollout, result))
    assert vars(rollout).get("_warnings") == [NO_ROOT_PYTHON]


# --- M4: exit codes -------------------------------------------------------------


@pytest.mark.parametrize(
    "args",
    [
        ["review", "no/such/job"],
        ["regrade", "no/such/job"],
        ["import", "no/such/task"],
        ["tasks", "normalize", "no/such/task"],
        ["export", "no/such/task", "out"],
        ["tasks", "digest", "no/such/task"],
        ["sandbox", "create", "no/such/task"],
        ["eval", "continue", "no/such/run"],
        ["eval", "continue-batch", "no/such/runs"],
        ["skills", "eval", "no/such/skill"],
        ["train", "validate", "no/such/rows.jsonl"],
        ["traj", "upload", "no/such/traj.jsonl", "--dry-run"],
    ],
)
def test_a_path_that_is_not_there_exits_2(args, tmp_path, monkeypatch):
    """M4: cli.md says every missing path exits 2; these exited 1."""
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, args)
    assert result.exit_code == 2, result.output
    assert "next:" in result.output


@pytest.mark.parametrize(
    "args",
    [
        ["review", str(HELLO), "--passing", "--failing"],
        ["hub", "list", "--provider", "nope"],
        ["hub", "list", "--owner", "someone"],
        ["skills", "eval", str(HELLO), "--sandbox", "dokcer"],
        ["skills", "validate", str(HELLO), "--fail-on", "nope"],
        ["tasks", "generate", "--from-local", "--outcome", "nope"],
        ["run", str(HELLO), "--agent-env", "NOEQUALS"],
        ["run", str(HELLO), "--allow-task-env", "A=B"],
        ["run", str(HELLO), "--separate-verifier-include", "relative/path"],
        ["eval", "run", "--agent", "oracle"],
        ["check", str(HELLO), "--verifier-sandbox", "nope"],
        ["export", str(HELLO), "out", "--target", "nope"],
        ["export", str(HELLO)],
        ["tasks", "generate"],
        ["tasks", "generate", "--from-local", "--from-file", "x.jsonl"],
        ["tasks", "generate", "--from-file", TRACES, "--task-format", "nope"],
        [
            "tasks",
            "generate",
            "--from-file",
            TRACES,
            "--output",
            str(HELLO / "task.md"),
        ],
        ["tasks", "generate", "--from-file", "no/such/traces.jsonl"],
        ["tasks", "generate", "--from-file", str(HELLO)],
        ["train", "convert", str(HELLO), "-o", "x.jsonl", "--format", "nope"],
        ["train", "convert", str(HELLO), "-o", "x.jsonl", "--group-by", "nope"],
        ["train", "convert", str(HELLO), "-o", "x.jsonl", "--include-oracle"],
        [
            "train",
            "convert",
            str(HELLO),
            "-o",
            "x.jsonl",
            "--format",
            "branch-tree",
            "--subagent-rows",
        ],
        ["embodied", "report", "no/such/job"],
        ["embodied", "export", "no/such/job", "-o", "x.jsonl"],
    ],
)
def test_a_bad_value_exits_2(args, tmp_path, monkeypatch):
    """M4: cli.md says a bad value, a missing source or an option that does
    not apply exits 2; these exited 1."""
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, args)
    assert result.exit_code == 2, result.output
    assert "next:" in result.output


def test_the_exit_code_docs_say_what_the_cli_does():
    """M4: analysing-runs.md said an unknown agent exits 1. Docs v2:
    analysing-runs.md is docs/evaluate/read-results.md, and the exit codes
    are on docs/scale/ci.md (a bad value, such as an unknown agent, exits 2)."""
    docs = Path(__file__).resolve().parents[1] / "docs"
    results = (docs / "evaluate" / "read-results.md").read_text(encoding="utf-8")
    ci = (docs / "scale" / "ci.md").read_text(encoding="utf-8")
    assert "a config error (an unknown agent" not in results + ci
    assert (
        "| 2 | the command was wrong: a bad flag or value, refused before "
        "anything ran |" in ci
    )


# --- L1-L5 ------------------------------------------------------------------------


def test_the_identity_switch_refusal_carries_no_review_code():
    """L1: the refusal printed "(K1-R15-2)"."""
    from benchflow.task.identity_switch import refusal

    text = refusal("daytona")
    assert "K1-R15" not in text
    assert "survive into grading" in text


def test_hub_show_prints_a_datasets_entry_as_json(monkeypatch):
    """L2: `bench hub show terminal-bench@2.0 --json` said "No such option"."""
    import benchflow.hub.registry as hub_registry

    entry = {
        "name": "terminal-bench",
        "version": "2.0",
        "tasks": [{"name": "regex-log"}, {"name": "fix-git"}],
    }
    monkeypatch.setattr(hub_registry, "load_hub_registry", lambda src=None: [entry])
    result = runner.invoke(app, ["hub", "show", "terminal-bench@2.0", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == entry

    hosted = runner.invoke(app, ["hub", "show", "primeintellect/x", "--json"])
    assert hosted.exit_code == 2, hosted.output


def test_a_missing_contributor_id_without_a_terminal_exits_2(tmp_path, monkeypatch):
    """L3: it exited 1, though a missing digest or path exits 2."""
    from benchflow.cli import traj

    monkeypatch.setattr(traj, "_infer_github_id", lambda: "")
    monkeypatch.setattr(traj, "_infer_email", lambda: "")
    with pytest.raises(ValueError) as refused:
        traj._resolve_contributor(None, None)
    assert getattr(refused.value, "exit_code", 1) == 2


def test_an_api_run_leaves_the_hosts_httpx_lines_alone(info_root, monkeypatch, caplog):
    """L4: one BenchFlow fetch set httpx to WARNING for the rest of the process."""
    import httpx

    from benchflow.sandbox import _verifier_runtime

    http = logging.getLogger("httpx")
    monkeypatch.setattr(http, "level", logging.NOTSET)
    monkeypatch.setattr(http, "filters", list(http.filters))

    def fake_get(url, **_kwargs):
        http.info('HTTP Request: GET %s "HTTP/1.1 200 OK"', url)
        return SimpleNamespace(status_code=200)

    monkeypatch.setattr(httpx, "get", fake_get)
    caplog.set_level(logging.INFO)
    _verifier_runtime._http_get("https://pypi.org/pypi/pytest/8.4.2/json")
    assert not [r for r in caplog.records if "pypi.org" in r.getMessage()]
    assert http.level == logging.NOTSET
    http.info('HTTP Request: GET %s "HTTP/1.1 200 OK"', "https://host.example/")
    assert [r for r in caplog.records if "host.example" in r.getMessage()]


def test_every_line_of_a_trials_message_names_its_task():
    """L5: the second line of a multi-line warning had no task."""
    from benchflow._utils.log_context import TaskPrefixFormatter, trial_log_context

    record = logging.LogRecord(
        "benchflow", logging.WARNING, __file__, 1, "probe failed\nstderr: x", (), None
    )
    with trial_log_context("overfull-hbox"):
        text = TaskPrefixFormatter("%(message)s").format(record)
    assert text.splitlines() == [
        "[overfull-hbox] probe failed",
        "[overfull-hbox] stderr: x",
    ]


def test_plugin_guard_lines_are_debug_lines():
    """L5: "the plugin guard reaches ... through PYTHONPATH only" and "Verifier
    uv/pip state stays where it was" printed on TB2 runs."""
    import ast

    src = Path(__file__).resolve().parents[1] / "src" / "benchflow"
    tree = ast.parse((src / "sandbox" / "lockdown.py").read_text(encoding="utf-8"))
    levels = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if not (
            isinstance(node.func.value, ast.Name) and node.func.value.id == "logger"
        ):
            continue
        first = node.args[0] if node.args else None
        if isinstance(first, ast.Constant) and str(first.value).startswith(
            "Verifier uv/pip state stays where it was"
        ):
            levels.append(("uv/pip state", node.func.attr))
        if isinstance(first, ast.Name) and first.id == "line":
            levels.append(("guard line", node.func.attr))
    assert sorted(levels) == [("guard line", "debug"), ("uv/pip state", "debug")]
