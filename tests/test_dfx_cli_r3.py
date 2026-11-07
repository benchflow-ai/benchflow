"""Guard the DFX-CLI round 3 fixes to the CLI behavior at commit 66495284a."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from benchflow.cli._errors import is_expected, run_cli
from benchflow.cli.main import app

HELLO = Path(__file__).parent / "examples" / "hello-world-task"
OFFLINE_NEXT = "check this machine's network (or its proxy), then run the command again"


@pytest.fixture(autouse=True)
def isolated_cli(monkeypatch, tmp_path):
    from benchflow.agents import remote_manifests

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BENCHFLOW_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(
        remote_manifests, "autoload_remote_manifest_agents", lambda: None
    )


def _run(capsys, words):
    with pytest.raises(SystemExit) as exited:
        run_cli(app, words)
    captured = capsys.readouterr()
    return exited.value.code, " ".join((captured.out + captured.err).split())


def _fetch_command(monkeypatch, tmp_path, source):
    from benchflow.cli import traj

    monkeypatch.setattr(traj, "_local_skill_md", lambda: None)
    if source == "hf":
        # An empty hub listing takes the real HTTP API fallback, without
        # needing the optional Hub client or any external request.
        monkeypatch.setitem(
            sys.modules,
            "huggingface_hub",
            SimpleNamespace(
                list_repo_files=lambda *a, **k: [],
                hf_hub_download=lambda *a, **k: None,
            ),
        )
        return ["tasks", "generate", "--from-hf", "some/dataset", "--output", "gen"]
    return ["traj", "setup", *(["--yes"] if source == "setup-yes" else [])]


@pytest.mark.parametrize("source", ["setup", "setup-yes", "hf"])
@pytest.mark.parametrize(
    "error_type", [httpx.ConnectError, httpx.ReadTimeout, httpx.ProxyError]
)
def test_offline_download_is_actionable(
    monkeypatch, tmp_path, capsys, source, error_type
):
    """Guards N1 at 66495284a: an offline fetch was reported as a BenchFlow bug."""
    words = _fetch_command(monkeypatch, tmp_path, source)

    def offline(*args, **kwargs):
        raise error_type("network unavailable")

    monkeypatch.setattr(httpx, "get", offline)
    code, output = _run(capsys, words)
    assert code == 1, output
    assert f"next: {OFFLINE_NEXT}" in output
    assert (
        "some/dataset" in output
        if source == "hf"
        else "trajectory-upload skill" in output
    )
    assert "Traceback" not in output
    assert "bug in BenchFlow" not in output
    assert not (tmp_path / "logs").exists()
    assert not list(tmp_path.rglob("*.jsonl"))
    assert not list(tmp_path.rglob("SKILL.md"))


@pytest.mark.parametrize("source", ["setup", "setup-yes", "hf"])
def test_control_download_programming_error_keeps_diagnostics(
    monkeypatch, tmp_path, capsys, source
):
    """Guards N1's 66495284a contract: programming errors still get crash diagnostics."""
    words = _fetch_command(monkeypatch, tmp_path, source)

    def broken(*args, **kwargs):
        raise TypeError("broken request construction")

    monkeypatch.setattr(httpx, "get", broken)
    code, output = _run(capsys, words)
    assert code == 1
    assert "bug in BenchFlow" in output
    assert "Traceback" in output
    assert "TypeError: broken request construction" in output
    assert OFFLINE_NEXT not in output
    assert list((tmp_path / "logs").glob("*"))


@pytest.mark.parametrize(
    "format_name", ["prime-sft", "trl-sft", "branch-tree", "verl-sft"]
)
@pytest.mark.parametrize("existing_output", [False, True])
def test_missing_conversion_job_writes_nothing(
    tmp_path, capsys, format_name, existing_output
):
    """Guards N2 at 66495284a: a mistyped job could overwrite output and report success."""
    output = tmp_path / "x.jsonl"
    if existing_output:
        output.write_text("keep this data\n")
    code, message = _run(
        capsys,
        [
            "train",
            "convert",
            "jobs/nope",
            "-o",
            str(output),
            "--format",
            format_name,
            "--manifest",
            "manifest.json",
        ],
    )
    assert code == 2, message
    assert "jobs/nope" in message and "next:" in message
    assert "Converted" not in message
    assert (
        output.read_text() == "keep this data\n"
        if existing_output
        else not output.exists()
    )
    assert not (tmp_path / "manifest.json").exists()


@pytest.mark.parametrize(
    "words",
    [
        ["review", "job", "--rubric", "nope.json"],
        ["review", "job", "--prompt", "nope.md"],
        ["review", "job", "--tasks-root", "nope"],
        ["run", "-a", "oracle", "-d", "terminal-bench@2.0", "--registry", "nope.json"],
        ["hub", "check", "--registry", "nope.json"],
        [
            "train",
            "convert",
            "job",
            "-o",
            "out.jsonl",
            "--canonical-selection",
            "nope.json",
        ],
    ],
)
def test_missing_input_option_is_usage_error(monkeypatch, tmp_path, capsys, words):
    """Guards N2 at 66495284a: missing input paths in options escaped the argument sweep."""
    from benchflow import runtime

    monkeypatch.setattr(runtime, "check_credentials", lambda *a, **k: None)
    (tmp_path / "job").mkdir()
    code, message = _run(capsys, words)
    assert code == 2, message
    assert "nope" in message and "next:" in message
    assert "Reviewing" not in message
    assert "Traceback" not in message
    assert not (tmp_path / "out.jsonl").exists()
    assert not (tmp_path / "jobs").exists()


@pytest.mark.parametrize(
    "words",
    [
        ["review", "job", "-a", "claud"],
        ["review", "job", "--sandbox", "dokcer"],
        ["check", str(HELLO), "--sandbox", "dokcer", "--level", "runtime-capability"],
        ["check", str(HELLO), "--sandbox", "dokcer", "--level", "schema"],
        ["check", str(HELLO), "--agent", "claud"],
    ],
)
def test_unknown_option_name_is_usage_error(monkeypatch, tmp_path, capsys, words):
    """Guards N2 at 66495284a: unknown names are CLI input errors before review/check."""
    from benchflow import runtime

    monkeypatch.setattr(runtime, "check_credentials", lambda *a, **k: None)
    (tmp_path / "job").mkdir()
    code, message = _run(capsys, words)
    assert code == 2, message
    assert "next:" in message
    assert "claud" in message or "dokcer" in message
    assert "Reviewing" not in message
    assert "errors.pydantic.dev" not in message
    assert "Traceback" not in message


def test_missing_registry_is_rejected_before_preflight(monkeypatch, capsys):
    """Guards N2 at 66495284a: a missing registry must not reach machine/login checks."""
    from benchflow.cli import doctor

    started = []

    def preflight(*args, **kwargs):
        started.append(True)
        raise RuntimeError("machine unavailable")

    monkeypatch.setattr(doctor, "start_docker_preflight", preflight)
    code, message = _run(
        capsys, ["run", "-d", "terminal-bench@2.0", "--registry", "nope.json"]
    )
    assert code == 2, message
    assert "nope.json" in message and "next:" in message
    assert started == []


@pytest.mark.parametrize("command", [["results"], ["eval", "compare"]])
@pytest.mark.parametrize("missing_index", [0, 1])
def test_control_missing_path_among_multiple_positionals(
    tmp_path, capsys, command, missing_index
):
    """Guards N2's 66495284a contract for each position in a two-job command."""
    trial = tmp_path / "trial"
    trial.mkdir()
    (trial / "result.json").write_text(
        json.dumps({"task_name": "hello", "rewards": {"reward": 1.0}})
    )
    paths = [str(trial), str(trial)]
    paths[missing_index] = "jobs/nope"
    code, message = _run(capsys, [*command, *paths])
    assert code == 2, message
    assert "jobs/nope" in message and "next:" in message


@pytest.mark.parametrize("operation", ["read-tree", "clone"])
@pytest.mark.parametrize("source", ["dataset", "source-repo", "config"])
def test_internal_git_failure_keeps_crash_diagnostics(
    monkeypatch, tmp_path, capsys, operation, source
):
    """Guards N3 at 66495284a: internal git failures must not be classified as user mistakes."""
    from benchflow._utils import benchmark_repos, dataset_registry

    empty = tmp_path / "empty-cache"
    subprocess.run(["git", "init", "--quiet", str(empty)], check=True)

    def fail(*args, **kwargs):
        if operation == "read-tree":
            return benchmark_repos._git_quiet(empty, "read-tree", "-mu", "HEAD")

        def broken_clone(cmd, **kwargs):
            raise subprocess.CalledProcessError(
                128, cmd, stderr="fatal: bad config in cache\n"
            )

        with monkeypatch.context() as patch:
            patch.setattr(benchmark_repos.subprocess, "run", broken_clone)
            benchmark_repos._run_clone(
                ["git", "clone", "https://github.com/a/b.git"], "a", "b"
            )

    monkeypatch.setattr(dataset_registry, "resolve_dataset", fail)
    monkeypatch.setattr(benchmark_repos, "resolve_source_with_metadata", fail)
    if source == "dataset":
        words = ["-d", "some@1"]
    elif source == "source-repo":
        words = ["--source-repo", "a/b"]
    else:
        config = tmp_path / "run.yaml"
        config.write_text("source:\n  repo: a/b\n")
        words = ["--config", str(config)]
    code, message = _run(capsys, ["run", "-a", "oracle", *words])
    assert code == 1, message
    assert "bug in BenchFlow" in message
    assert "Traceback" in message
    assert list((tmp_path / "logs").glob("*"))
    assert OFFLINE_NEXT not in message


@pytest.mark.parametrize("operation", ["fetch", "clone"])
def test_missing_remote_ref_has_an_actionable_hint(monkeypatch, operation):
    """Guards N3 at 66495284a: a missing requested remote ref is expected, with a next step."""
    from benchflow._utils import benchmark_repos

    def missing(cmd, **kwargs):
        if operation == "clone":
            raise subprocess.CalledProcessError(
                128,
                cmd,
                stderr="fatal: Remote branch typo not found in upstream origin\n",
            )
        return SimpleNamespace(
            returncode=128, stderr="fatal: couldn't find remote ref typo\n"
        )

    monkeypatch.setattr(benchmark_repos.subprocess, "run", missing)
    with pytest.raises((subprocess.CalledProcessError, RuntimeError)) as caught:
        if operation == "clone":
            benchmark_repos._run_clone(
                ["git", "clone", "--branch", "typo", "https://github.com/a/b.git"],
                "a",
                "b",
            )
        else:
            benchmark_repos._git_quiet(Path("cache"), "fetch", "origin", "typo")
    assert is_expected(caught.value)
    assert "source ref" in (getattr(caught.value, "hint", None) or "")
    assert getattr(caught.value, "hint", None) != OFFLINE_NEXT


@pytest.mark.parametrize(
    "operation", ["read-tree", "checkout", "sparse-checkout", "fetch", "clone"]
)
def test_unknown_git_failure_is_not_expected(monkeypatch, operation):
    """Guards N3 at 66495284a for every cache mutation and clone error class."""
    from benchflow._utils import benchmark_repos

    def broken(cmd, **kwargs):
        if operation == "clone":
            raise subprocess.CalledProcessError(
                128, cmd, stderr="fatal: bad cache state\n"
            )
        return SimpleNamespace(returncode=128, stderr="fatal: bad cache state\n")

    monkeypatch.setattr(benchmark_repos.subprocess, "run", broken)
    with pytest.raises((subprocess.CalledProcessError, RuntimeError)) as caught:
        if operation == "clone":
            benchmark_repos._run_clone(
                ["git", "clone", "https://github.com/a/b.git"], "a", "b"
            )
        else:
            benchmark_repos._git_quiet(Path("cache"), operation)
    assert not is_expected(caught.value)
    assert getattr(caught.value, "hint", None) is None
