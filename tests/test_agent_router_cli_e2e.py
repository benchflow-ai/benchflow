"""End-to-end CLI integration tests for the ``bench agent`` adoption router.

These tests drive the *real* registered ``bench agent create|run|verify``
commands through ``typer``'s ``CliRunner`` against an on-disk benchmarks tree in
a tmp dir — no fake exec/report layer on the create/verify path. They pin
behavior (CLI exit codes, verdict strings, on-disk file contents) so a mutation
to the router's logic shows up as a failing assertion rather than passing by
coincidence.

The unit suite in ``test_agent_router.py`` covers the pure functions; this file
covers the wired-up CLI surface: the scaffold actually lands on disk and the
generated Python compiles + YAML parses; create refuses on re-run; bad slugs are
rejected by the real CLI; a realistic parity record flips the verify verdict and
exit code; and the live (non-dry-run) ``run`` path fails closed on missing
credentials without ever spawning ``codex``.
"""

from __future__ import annotations

import json
import py_compile
from pathlib import Path

import click
import pytest
import yaml
from click.testing import Result
from typer.testing import CliRunner

import benchflow.agent_router as agent_router
from benchflow.cli.main import app

# Every generated file the scaffold must land for slug ``my-bench``. Runner file
# and job-yaml names embed the slug, so this set also locks the naming scheme.
_EXPECTED_SCAFFOLD_FILES = {
    "__init__.py",
    "benchflow.py",
    "main.py",
    "parity_test.py",
    "run_my_bench.py",
    "my-bench.yaml",
    "benchmark.yaml",
    "parity_experiment.json",
    "README.md",
}


def _run(runner: CliRunner, *args: str) -> Result:
    return runner.invoke(app, list(args))


# ── create: real scaffold lands, compiles, and parses ─────────────────


def test_cli_create_writes_full_scaffold_that_compiles_and_parses(
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    result = _run(
        runner, "agent", "create", "my-bench", "--benchmarks-dir", str(tmp_path)
    )
    assert result.exit_code == 0, result.output

    target = tmp_path / "my-bench"
    on_disk = {p.name for p in target.iterdir()}
    assert on_disk == _EXPECTED_SCAFFOLD_FILES

    # Every generated Python module must byte-compile (catches template typos).
    for rel in _EXPECTED_SCAFFOLD_FILES:
        if rel.endswith(".py"):
            py_compile.compile(str(target / rel), doraise=True)

    # The converter module exposes the documented entry points with no leftover
    # template tokens — the slug substitution actually ran.
    converter = (target / "benchflow.py").read_text()
    assert "def convert(" in converter
    assert "def convert_all(" in converter
    assert "{{NAME}}" not in converter and "{{TITLE}}" not in converter

    # Both YAML files parse and carry the substituted slug in their key fields.
    descriptor = yaml.safe_load((target / "benchmark.yaml").read_text())
    assert descriptor["name"] == "my-bench"
    job = yaml.safe_load((target / "my-bench.yaml").read_text())
    assert job["tasks_dir"] == "benchmarks/my-bench/tasks"

    # The generated job yaml must load through the same Evaluation.from_yaml the
    # runner uses, with no ValueError from model resolution. A safe_load alone
    # never exercises effective_model, so we assert the *resolved* model equals
    # the default agent's default — the agent/model pair in the template has to
    # be self-consistent or the runner crashes at job-load time before any task
    # runs (regression guard for the codex-acp + empty-model scaffold).
    from benchflow.evaluation import DEFAULT_AGENT, DEFAULT_MODEL, Evaluation

    evaluation = Evaluation.from_yaml(str(target / "my-bench.yaml"))
    assert job["agent"] == DEFAULT_AGENT
    assert evaluation._config.agent == DEFAULT_AGENT
    assert evaluation._config.model == DEFAULT_MODEL

    # The scaffolded parity record is valid JSON in the template state.
    parity = json.loads((target / "parity_experiment.json").read_text())
    assert parity["benchmark"] == "my-bench"
    assert parity["status"] == "template"


def test_cli_create_refuses_existing_benchmark_with_nonzero_exit(
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    first = _run(
        runner, "agent", "create", "my-bench", "--benchmarks-dir", str(tmp_path)
    )
    assert first.exit_code == 0, first.output
    sentinel = tmp_path / "my-bench" / "README.md"
    original = sentinel.read_text()

    again = _run(
        runner, "agent", "create", "my-bench", "--benchmarks-dir", str(tmp_path)
    )
    assert again.exit_code == 1
    assert "already exists" in click.unstyle(again.output)
    # Fail-closed refusal: the existing scaffold is left untouched.
    assert sentinel.read_text() == original


@pytest.mark.parametrize(
    "bad_name",
    [
        "MyBench",  # uppercase
        "1bench",  # leading digit
        "a/b",  # path separator
    ],
)
def test_cli_create_rejects_invalid_slug_without_writing(
    tmp_path: Path, bad_name: str
) -> None:
    runner = CliRunner()
    result = _run(
        runner, "agent", "create", bad_name, "--benchmarks-dir", str(tmp_path)
    )
    assert result.exit_code == 1
    assert "invalid benchmark name" in click.unstyle(result.output)
    # Nothing was written for the rejected slug.
    assert list(tmp_path.iterdir()) == []


# ── verify: realistic parity records flip verdict and exit code ───────


def _confirmed_parity_record() -> dict:
    """A realistic 'parity-confirmed' record (programbench-shaped).

    Every compared criterion agrees (deterministic floor) and every
    legacy-vs-converted reward delta is zero (statistical layer), so the gate
    must return ``parity-confirmed``.
    """
    return {
        "experiment": "side-by-side-parity",
        "benchmark": "my-bench",
        "status": "parity-confirmed",
        "judge_model": "gemini-3.1-flash-lite-preview",
        "conversion_parity": {
            "tasks": [
                {
                    "task_id": "abishekvashok__cmatrix.5c082c6",
                    "criteria_results": [
                        {
                            "criterion_id": "C-001",
                            "original_verdict": "pass",
                            "adapted_verdict": "pass",
                            "agreement": True,
                        },
                        {
                            "criterion_id": "C-002",
                            "original_verdict": "fail",
                            "adapted_verdict": "fail",
                            "agreement": True,
                        },
                    ],
                },
            ],
        },
        "agent_parity": {
            "results": [
                {
                    "task_id": "ajeetdsouza__zoxide.67ca1bc",
                    "programbench": {"reward": 1.0},
                    "benchflow": {"reward": 1.0},
                },
                {
                    "task_id": "anordal__shellharden.6a6ffd4",
                    "programbench": {"reward": 0.9992},
                    "benchflow": {"reward": 0.9992},
                },
            ],
        },
    }


def _divergent_parity_record() -> dict:
    """A record where the converted verdict diverges from the original."""
    return {
        "experiment": "side-by-side-parity",
        "benchmark": "my-bench",
        "conversion_parity": {
            "tasks": [
                {
                    "task_id": "abishekvashok__cmatrix.5c082c6",
                    "criteria_results": [
                        {
                            "criterion_id": "C-001",
                            "original_verdict": "pass",
                            "adapted_verdict": "fail",
                            "agreement": False,
                        },
                    ],
                },
            ],
        },
    }


def test_cli_create_then_verify_confirmed_record_exits_zero(tmp_path: Path) -> None:
    runner = CliRunner()
    create = _run(
        runner, "agent", "create", "my-bench", "--benchmarks-dir", str(tmp_path)
    )
    assert create.exit_code == 0, create.output

    parity_file = tmp_path / "my-bench" / "parity_experiment.json"
    parity_file.write_text(json.dumps(_confirmed_parity_record(), indent=2))

    verify = _run(
        runner, "agent", "verify", "my-bench", "--benchmarks-dir", str(tmp_path)
    )
    assert verify.exit_code == 0, verify.output
    out = click.unstyle(verify.output)
    assert "Verdict:" in out
    assert "parity-confirmed" in out
    # The confirmed path reports full criterion agreement, not a divergence.
    assert "2/2 criteria agree" in out
    assert "parity-divergent" not in out


def test_cli_verify_divergent_record_exits_nonzero_and_writes_issue_draft(
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    create = _run(
        runner, "agent", "create", "my-bench", "--benchmarks-dir", str(tmp_path)
    )
    assert create.exit_code == 0, create.output

    parity_file = tmp_path / "my-bench" / "parity_experiment.json"
    parity_file.write_text(json.dumps(_divergent_parity_record(), indent=2))

    issue_out = tmp_path / "divergence-issue.md"
    verify = _run(
        runner,
        "agent",
        "verify",
        "my-bench",
        "--benchmarks-dir",
        str(tmp_path),
        "--issue-out",
        str(issue_out),
    )
    assert verify.exit_code == 1
    assert "parity-divergent" in click.unstyle(verify.output)

    # --issue-out writes a non-empty draft naming the failing criterion, and the
    # draft states it was not auto-filed.
    assert issue_out.exists()
    issue = issue_out.read_text()
    assert issue.strip()
    assert "parity-divergent" in issue
    assert "original=pass converted=fail" in issue
    assert "NOT been filed" in issue


def test_cli_verify_fresh_scaffold_is_insufficient_evidence(tmp_path: Path) -> None:
    runner = CliRunner()
    create = _run(
        runner, "agent", "create", "my-bench", "--benchmarks-dir", str(tmp_path)
    )
    assert create.exit_code == 0, create.output

    # A fresh scaffold ships a template parity record with no comparisons, so the
    # gate withholds confidence and exits non-zero down the support path.
    verify = _run(
        runner, "agent", "verify", "my-bench", "--benchmarks-dir", str(tmp_path)
    )
    assert verify.exit_code == 1
    out = click.unstyle(verify.output)
    assert "insufficient-evidence" in out
    # A fresh scaffold has not diverged — it simply has no recorded parity data —
    # so the divergence "could not be closed / open an issue" draft must NOT be
    # dumped; the gate points the author at parity_test.py instead.
    assert "NOT been filed" not in out
    assert "parity_test.py" in out


# ── run: dry-run prints the command; live path fails closed ───────────


def test_cli_run_dry_run_prints_codex_command_with_context_markers(
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    result = _run(
        runner,
        "agent",
        "run",
        "github.com/foo/bar",
        "--name",
        "my-bench",
        "--dry-run",
    )
    assert result.exit_code == 0, result.output
    # Inspect pure stdout: the dry-run command prints to stdout, while the
    # `bench agent run` deprecation notice goes to stderr. Using stdout keeps the
    # verbatim/soft-wrap assertion below independent of whether an earlier test
    # in the process already emitted (and deduped) that once-per-process notice.
    out = click.unstyle(result.stdout)
    # The constructed (but un-launched) codex command carries the source and the
    # adoption context — the prompt the live run would have used.
    assert "codex" in out
    assert "exec" in out
    assert "github.com/foo/bar" in out
    assert "Benchmark adoption" in out
    assert "benchmarks/my-bench/" in out
    # The command must be printed verbatim (soft_wrap): console-width hard
    # wrapping would insert newlines mid-token (e.g. inside the --cd path),
    # making the dry-run output non-copy-pasteable. The command head up to the
    # prompt's first newline always exceeds 80 columns, so under hard wrapping
    # this first physical line would be split wherever the repo path puts it.
    assert out.splitlines()[0].endswith("'# Benchmark adoption: my-bench")


def test_cli_run_live_path_fails_closed_without_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The non-dry-run path refuses to launch when no codex credentials exist.

    We force a credential-free host (no API-key env vars, auth file pointed at a
    missing path) and replace the real subprocess exec with a tripwire, so the
    test proves the fail-closed gate fires *before* anything is spawned —
    independent of whatever credentials the host machine actually has.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CODEX_API_KEY", raising=False)
    monkeypatch.setattr(
        agent_router, "_default_codex_auth_file", lambda: tmp_path / "absent-auth.json"
    )

    def _tripwire(*_args: object, **_kwargs: object) -> int:
        raise AssertionError("codex was spawned despite missing credentials")

    monkeypatch.setattr(agent_router, "_subprocess_exec", _tripwire)

    result = _run(
        CliRunner(), "agent", "run", "github.com/foo/bar", "--name", "my-bench"
    )
    assert result.exit_code == 1
    out = click.unstyle(result.output)
    # The precise key/login guidance, unwrapped across rich's line-wrapping.
    assert "codex needs credentials to launch" in out
    assert "OPENAI_API_KEY" in out
    assert "CODEX_API_KEY" in out


# ── error handling: bad user paths fail-fast cleanly, never a traceback ──


def test_cli_create_bad_benchmarks_dir_emits_clean_error_not_traceback(
    tmp_path: Path,
) -> None:
    """A --benchmarks-dir that is a regular file makes create_benchmark's mkdir
    raise NotADirectoryError; the handler must surface a one-line message and
    exit 1, not let the OSError escape into a Rich traceback (shared by the
    Guards PR #789 (CLI error-handling hardening).
    `agent create` / `adopt init` / `eval adopt init` aliases)."""
    runner = CliRunner()
    not_a_dir = tmp_path / "regular_file"
    not_a_dir.write_text("x")

    result = _run(
        runner, "agent", "create", "okbench", "--benchmarks-dir", str(not_a_dir)
    )

    assert result.exit_code == 1
    # The OSError must be caught, not propagated as the result exception.
    assert not isinstance(result.exception, OSError), result.exception
    assert "could not scaffold benchmark" in click.unstyle(result.output)


def test_cli_verify_unwritable_issue_out_emits_clean_error_not_traceback(
    tmp_path: Path,
) -> None:
    """A --issue-out whose parent directory does not exist must fail-fast with a
    clean message after the divergence verdict, not dump a FileNotFoundError
    Guards PR #789 (CLI error-handling hardening).
    traceback from the unguarded write_text."""
    runner = CliRunner()
    create = _run(
        runner, "agent", "create", "my-bench", "--benchmarks-dir", str(tmp_path)
    )
    assert create.exit_code == 0, create.output
    parity_file = tmp_path / "my-bench" / "parity_experiment.json"
    parity_file.write_text(json.dumps(_divergent_parity_record(), indent=2))

    missing_parent = tmp_path / "no_such_dir" / "issue.md"
    verify = _run(
        runner,
        "agent",
        "verify",
        "my-bench",
        "--benchmarks-dir",
        str(tmp_path),
        "--issue-out",
        str(missing_parent),
    )

    assert verify.exit_code == 1
    assert not isinstance(verify.exception, OSError), verify.exception
    assert "cannot write issue draft" in click.unstyle(verify.output)
