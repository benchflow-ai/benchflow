"""Guard R3-M1/R3-L1 at a3bf6cd275: catalog ordering and discovery inputs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

import benchflow
from benchflow.cli.main import app

MODEL = "gemini/gemini-3.5-flash-lite"
MANIFEST = f'''contract_version = "1.0"
name = "review-fixture"
install_cmd = "false"
launch_cmd = "false --acp"
requires_env = ["GEMINI_API_KEY"]
default_model = "{MODEL}"
'''
ENTRY = """
import json
import sys
import benchflow.review as review
from benchflow.agents.env import resolve_agent_env
from benchflow.cli._errors import run_cli
from benchflow.cli.main import app

async def stop_before_review(path, **kwargs):
    resolved = resolve_agent_env(kwargs["agent"], kwargs["model"], kwargs["agent_env"])
    print("REVIEW_BOUNDARY " + json.dumps({
        "agent": kwargs["agent"], "model": kwargs["model"],
        "explicit_key_wins": resolved.get("GEMINI_API_KEY") == "synthetic-explicit",
    }))
    raise SystemExit(0)

review.run_reviews = stop_before_review
run_cli(app, sys.argv[1:])
"""


@pytest.fixture
def isolated_process(tmp_path):
    # Each invocation imports a fresh registry and reads only synthetic config.
    # The real credential check runs, but the review runner cannot start work.
    env = {key: os.environ[key] for key in ("PATH", "SYSTEMROOT") if key in os.environ}
    env.update(
        HOME=str(tmp_path / "home"),
        XDG_CACHE_HOME=str(tmp_path / "home/.cache"),
        BENCHFLOW_DOTENV_PATH=str(tmp_path / ".env"),
        BENCHFLOW_LOG_DIR=str(tmp_path / "logs"),
        BENCHFLOW_AGENTS_SOURCE="off",
        BENCHFLOW_SKIP_UPDATE_CHECK="1",
        NO_COLOR="1",
        COLUMNS="160",
        PYTHONPATH=str(Path(benchflow.__file__).resolve().parent.parent),
    )
    (tmp_path / "job").mkdir()
    manifest = tmp_path / "agents/review-fixture/manifest.toml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(MANIFEST)
    (tmp_path / ".env").write_text(f"BENCHFLOW_AGENTS_DIR={manifest.parent.parent}\n")

    def invoke(words, extra_env=None):
        return subprocess.run(
            [sys.executable, "-c", ENTRY, *words],
            cwd=tmp_path,
            env={**env, **(extra_env or {})},
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=30,
        )

    return invoke


@pytest.mark.parametrize("exported", [False, True], ids=["dotenv", "exported"])
@pytest.mark.parametrize(
    "explicit_model", [False, True], ids=["default-model", "model"]
)
def test_review_catalog_reaches_missing_credential(
    isolated_process, tmp_path, exported, explicit_model
):
    """Guards R3-M1 at a3bf6cd275: .env catalog overrides source=off before lookup."""
    words = ["review", "job", "-a", "review-fixture"]
    if explicit_model:
        words += ["-m", MODEL]
    env = {"BENCHFLOW_AGENTS_DIR": str(tmp_path / "agents")} if exported else {}
    result = isolated_process(words, env)
    assert result.returncode == 1, result.stdout + result.stderr
    assert f"GEMINI_API_KEY required for model '{MODEL}'" in result.stderr
    assert "next: `bench doctor`" in result.stderr
    assert result.stdout == ""
    assert "Unknown agent" not in result.stderr
    assert "Traceback" not in result.stderr
    assert not list((tmp_path / "job").iterdir())
    assert not (tmp_path / "logs").exists()


@pytest.mark.parametrize(
    "options", [[], ["-a", "acp/opencode"], ["-a", "oh", "-m", MODEL]]
)
def test_review_default_and_alias_keep_clean_credential_error(
    isolated_process, options
):
    """Guards R3-M1 at a3bf6cd275: default models, protocol names and aliases still resolve."""
    result = isolated_process(["review", "job", *options])
    assert result.returncode == 1, result.stdout + result.stderr
    assert f"GEMINI_API_KEY required for model '{MODEL}'" in result.stderr
    assert "next: `bench doctor`" in result.stderr
    assert result.stdout == ""
    assert "Traceback" not in result.stderr


def test_unknown_reviewer_uses_selected_catalog(isolated_process):
    """Guards R3-M1 at a3bf6cd275: a genuine unknown still exits 2 after catalog load."""
    result = isolated_process(["review", "job", "-a", "absent-reviewer-zzzz"])
    assert result.returncode == 2, result.stdout + result.stderr
    assert "Unknown agent" in result.stderr
    assert "review-fixture" in result.stderr
    assert "next: pass -a" in result.stderr
    assert "GEMINI_API_KEY" not in result.stderr
    assert result.stdout == ""


def test_explicit_reviewer_key_wins_over_dotenv_and_process(isolated_process, tmp_path):
    """Guards R3-M1 at a3bf6cd275: catalog loading preserves explicit key precedence."""
    with (tmp_path / ".env").open("a") as stream:
        stream.write("GEMINI_API_KEY=synthetic-dotenv\n")
    result = isolated_process(
        [
            "review",
            "job",
            "-a",
            "review-fixture",
            "-m",
            MODEL,
            "--agent-env",
            "GEMINI_API_KEY=synthetic-explicit",
        ],
        {"GEMINI_API_KEY": "synthetic-process"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    boundary = next(
        line
        for line in result.stdout.splitlines()
        if line.startswith("REVIEW_BOUNDARY ")
    )
    assert json.loads(boundary.removeprefix("REVIEW_BOUNDARY ")) == {
        "agent": "review-fixture",
        "model": MODEL,
        "explicit_key_wins": True,
    }
    assert not list((tmp_path / "job").iterdir())


@pytest.mark.parametrize("flag", ["--rubric", "--prompt", "--tasks-root"])
def test_review_input_validation_stays_before_catalog(isolated_process, tmp_path, flag):
    """Guards R3-M1 at a3bf6cd275: unrelated input errors still precede environment load."""
    (tmp_path / ".env").write_bytes(b"\xff")
    result = isolated_process(["review", "job", flag, "no-such-input"])
    assert result.returncode == 2, result.stdout + result.stderr
    assert f"next: pass an existing path to {flag}" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize(
    "command, option", [("projects", "--projects-dir"), ("skills", "--dir")]
)
@pytest.mark.parametrize("kind", ["missing", "file"])
def test_explicit_discovery_directory_is_usage_error(
    isolated_process, tmp_path, command, option, kind
):
    """Guards R3-L1 at a3bf6cd275: missing supplied discovery paths are not empty searches."""
    target = tmp_path / "discovery-input"
    if kind == "file":
        target.write_text("keep this input\n")
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep this output\n")
    words = (
        ["tasks", "generate", "--from-local", "--output", str(output), "--dry-run"]
        if command == "projects"
        else ["skills", "list"]
    )
    result = isolated_process([*words, option, str(target)])
    assert result.returncode == 2, result.stdout + result.stderr
    assert "discovery-input" in result.stderr
    assert "next:" in result.stderr and option in result.stderr
    assert "--from-hf" not in result.stderr
    assert "Traceback" not in result.stderr
    assert result.stdout == ""
    assert list(output.iterdir()) == [sentinel]
    assert sentinel.read_text() == "keep this output\n"
    if kind == "file":
        assert target.read_text() == "keep this input\n"
    else:
        assert not target.exists()


@pytest.mark.parametrize(
    "command, code, message",
    [
        ("projects", 1, "No traces found."),
        ("skills", 0, "No skills found."),
    ],
)
@pytest.mark.parametrize("explicit", [False, True], ids=["omitted", "empty"])
def test_empty_or_omitted_discovery_keeps_its_result(
    isolated_process, tmp_path, command, code, message, explicit
):
    """Guards R3-L1 at a3bf6cd275: optional defaults and existing empty directories stay valid."""
    words = (
        ["tasks", "generate", "--from-local", "--dry-run"]
        if command == "projects"
        else ["skills", "list"]
    )
    if explicit:
        (tmp_path / "empty").mkdir()
        words += ["--projects-dir" if command == "projects" else "--dir", "empty"]
    result = isolated_process(words)
    assert result.returncode == code, result.stdout + result.stderr
    assert message in result.stdout + result.stderr
    assert "Not a directory" not in result.stderr


def test_existing_discovery_directories_work(isolated_process, tmp_path):
    """Guards R3-L1 at a3bf6cd275: real local traces and skills still list successfully."""
    skill = tmp_path / "skills/fixture/SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text(
        "---\nname: fixture\ndescription: Synthetic skill\n---\nDo a fixture task.\n"
    )
    result = isolated_process(["skills", "list", "--dir", "skills"])
    assert result.returncode == 0, result.stdout + result.stderr
    assert "fixture" in result.stdout and "Discovered Skills" in result.stdout

    session = tmp_path / "projects/project/session.jsonl"
    session.parent.mkdir(parents=True)
    session.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {
                    "type": "user",
                    "sessionId": "fixture",
                    "message": {"role": "user", "content": "Write note.txt"},
                },
                {
                    "type": "assistant",
                    "sessionId": "fixture",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "name": "Write",
                                "input": {
                                    "file_path": "note.txt",
                                    "content": "fixture",
                                },
                            },
                        ],
                    },
                },
            ]
        )
        + "\n"
    )
    for source in (
        ["--from-local", "--projects-dir", "projects"],
        ["--from-file", str(session), "--projects-dir", "unused-missing"],
    ):
        result = isolated_process(["tasks", "generate", *source, "--dry-run"])
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Parsed Traces (1)" in result.stdout
        assert not (tmp_path / "tasks").exists()


def test_hf_source_ignores_unused_projects_directory(monkeypatch, tmp_path):
    """Guards R3-L1 at a3bf6cd275: --projects-dir is not an input to the HF source."""
    from benchflow.cli import trace_import

    called = []

    def load_hf(*args):
        called.append(args)
        return []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(trace_import, "_load_hf", load_hf)
    result = CliRunner().invoke(
        app,
        [
            "tasks",
            "generate",
            "--from-hf",
            "fixture/dataset",
            "--projects-dir",
            "unused-missing",
            "--dry-run",
        ],
    )
    assert result.exit_code == 1, result.output
    assert called == [("fixture/dataset", "auto", "train", 100)]
    assert "No traces found." in result.output
    assert "Not a directory" not in result.output
