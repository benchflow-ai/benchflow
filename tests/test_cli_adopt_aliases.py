"""Benchmark adoption is canonically the single ``bench eval adopt`` command.

Adoption lives under ``eval`` because ``eval`` is the universal benchmark entry
point (``eval create`` runs a benchmark; ``eval adopt`` makes a foreign one
runnable). ``bench eval adopt`` is one multi-mode command: ``<source>`` to
scaffold+convert, ``<name> --scaffold-only`` to only scaffold, and
``<name> --verify`` to run the parity gate. Two prior spellings stay as hidden
deprecated aliases for one release (removed in 0.7), each emitting a one-line
stderr notice that points at the new canonical shape:

* the top-level ``bench adopt init|convert|verify`` (the 0.6-dev intermediate
  name), and
* the original ``bench agent create|run|verify``.

``--hub`` likewise remains the deprecated spelling of ``environment list
--provider`` (hosted browsing moved to ``bench hub env list``). These tests pin
the canonical surface and every back-compat path so the renames stay additive.
"""

from __future__ import annotations

import json
import re

import click
from typer.main import get_command
from typer.testing import CliRunner

import benchflow.cli._shared as shared
from benchflow.agent_router import create_benchmark
from benchflow.cli.main import app

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _panel_command_names(out: str) -> set[str]:
    """Command names from Rich help panel rows (``│ <name>  <desc>``) only.

    The top-level tagline prose contains the word "adopt" (``run, author, and
    adopt agent benchmarks``), so a naive substring check would false-positive;
    parse the command-panel rows the way the docs-drift guard does.
    """
    names: set[str] = set()
    for line in _ANSI_RE.sub("", out).splitlines():
        m = re.match(r"^\s*│\s+([A-Za-z][\w-]*)\s", line)
        if m:
            names.add(m.group(1))
    return names


def _eval_adopt_command() -> click.Command:
    """The ``bench eval adopt`` Click command, read from the registered command
    tree rather than parsed from ``--help`` output — Rich wraps the help under a
    narrow/no-TTY CI terminal, so a flag like ``--verify`` is not a reliable
    substring of the rendered text (cf. the #750 help-parse fix)."""
    cli = get_command(app)
    return cli.commands["eval"].commands["adopt"]  # type: ignore[attr-defined]


def test_eval_adopt_is_single_command_with_modes() -> None:
    adopt = _eval_adopt_command()
    # One multi-mode command, NOT a subgroup with init/convert/verify subcommands.
    assert not isinstance(adopt, click.Group), (
        "`bench eval adopt` should be a single command, not a subgroup"
    )
    opts: set[str] = set()
    for param in adopt.params:
        opts.update(param.opts)
    for flag in ("--verify", "--scaffold-only", "--name", "--dry-run"):
        assert flag in opts, f"`bench eval adopt` missing {flag!r}: {sorted(opts)}"


def test_eval_help_shows_adopt_command() -> None:
    cli = get_command(app)
    assert "adopt" in cli.commands["eval"].commands  # type: ignore[attr-defined]


def test_agent_help_shows_management_hides_adoption() -> None:
    res = runner.invoke(app, ["agent", "--help"])
    assert res.exit_code == 0
    assert "list" in res.output and "show" in res.output
    # adoption verbs are hidden on the agent group now
    for verb in ("create", "run", "verify"):
        assert verb not in res.output, f"deprecated {verb!r} still shown in agent help"


def test_adopt_is_hidden_from_top_level_help() -> None:
    # The intermediate top-level `bench adopt` is now a hidden deprecated alias:
    # absent from the command panels (the tagline prose still says "adopt").
    res = runner.invoke(app, ["--help"], terminal_width=200)
    assert res.exit_code == 0
    assert "adopt" not in _panel_command_names(res.output), (
        f"deprecated top-level adopt still shown as a command: {res.output}"
    )


def test_eval_adopt_scaffold_only_is_canonical_and_silent(tmp_path) -> None:
    shared._DEPRECATION_WARNED.clear()
    res = runner.invoke(
        app,
        [
            "eval",
            "adopt",
            "canonical-bench",
            "--scaffold-only",
            "--benchmarks-dir",
            str(tmp_path),
        ],
    )
    assert res.exit_code == 0
    assert "deprecation" not in res.stderr


# ── the three canonical modes of the single command ───────────────────


def test_eval_adopt_convert_dry_run_prints_codex_command() -> None:
    """Convert mode (default): `bench eval adopt <source> --dry-run` prints the
    codex launch command without running it."""
    shared._DEPRECATION_WARNED.clear()
    res = runner.invoke(
        app,
        ["eval", "adopt", "github.com/foo/bar", "--name", "my-bench", "--dry-run"],
    )
    assert res.exit_code == 0, res.output
    out = click.unstyle(res.stdout)
    assert "codex" in out
    assert "exec" in out
    assert "benchmarks/my-bench/" in out
    # canonical command — no deprecation notice
    assert "deprecation" not in res.stderr


def test_eval_adopt_convert_auto_scaffolds_missing_package(
    tmp_path, monkeypatch
) -> None:
    """Convert mode scaffolds `benchmarks/<name>/` when it does not yet exist,
    before driving the conversion. We point the auto-scaffold at a tmp
    benchmarks dir and stub the live launch (no codex needed)."""
    import benchflow.agent_router as agent_router

    shared._DEPRECATION_WARNED.clear()
    # Make the live launch a no-op success so the test isolates the scaffold step.
    monkeypatch.setattr(
        agent_router, "run_agent_adoption", lambda *a, **k: 0, raising=True
    )
    res = runner.invoke(
        app,
        [
            "eval",
            "adopt",
            "github.com/foo/auto-bench",
            "--benchmarks-dir",
            str(tmp_path),
        ],
    )
    assert res.exit_code == 0, res.output
    # The package was auto-scaffolded under the tmp benchmarks dir.
    assert (tmp_path / "auto-bench" / "benchflow.py").exists()
    assert "Scaffolded" in click.unstyle(res.output)


def test_eval_adopt_convert_does_not_rescaffold_existing(tmp_path, monkeypatch) -> None:
    """If the package already exists, convert mode does NOT re-scaffold (no
    BenchmarkExistsError) — it proceeds straight to the conversion."""
    import benchflow.agent_router as agent_router

    shared._DEPRECATION_WARNED.clear()
    create_benchmark("auto-bench", tmp_path)
    sentinel = tmp_path / "auto-bench" / "README.md"
    original = sentinel.read_text()
    monkeypatch.setattr(
        agent_router, "run_agent_adoption", lambda *a, **k: 0, raising=True
    )
    res = runner.invoke(
        app,
        [
            "eval",
            "adopt",
            "github.com/foo/auto-bench",
            "--benchmarks-dir",
            str(tmp_path),
        ],
    )
    assert res.exit_code == 0, res.output
    # Existing scaffold untouched, and no "already exists" failure.
    assert sentinel.read_text() == original
    assert "already exists" not in click.unstyle(res.output)
    assert "Scaffolded" not in click.unstyle(res.output)


def test_eval_adopt_verify_runs_the_gate(tmp_path) -> None:
    """Verify mode: `bench eval adopt <name> --verify` scores the parity record."""
    shared._DEPRECATION_WARNED.clear()
    create_benchmark("my-bench", tmp_path)
    parity = tmp_path / "my-bench" / "parity_experiment.json"
    parity.write_text(
        json.dumps(
            {
                "conversion_parity": {
                    "tasks": [
                        {
                            "task_id": "t1",
                            "criteria_results": [
                                {
                                    "criterion_id": "C-1",
                                    "original_verdict": "pass",
                                    "adapted_verdict": "pass",
                                }
                            ],
                        }
                    ]
                }
            }
        )
    )
    res = runner.invoke(
        app,
        ["eval", "adopt", "my-bench", "--verify", "--benchmarks-dir", str(tmp_path)],
    )
    assert res.exit_code == 0, res.output
    assert "parity-confirmed" in click.unstyle(res.output)


def test_eval_adopt_verify_and_scaffold_only_together_errors(tmp_path) -> None:
    """The two mode flags are mutually exclusive → friendly error, exit 2."""
    res = runner.invoke(
        app,
        ["eval", "adopt", "my-bench", "--verify", "--scaffold-only"],
    )
    assert res.exit_code == 2
    assert "mutually exclusive" in click.unstyle(res.output)


def test_eval_adopt_missing_target_errors(tmp_path) -> None:
    """No positional target in any mode → friendly error, exit 2."""
    res = runner.invoke(app, ["eval", "adopt", "--scaffold-only"])
    assert res.exit_code == 2
    assert "missing target" in click.unstyle(res.output)


def test_eval_adopt_convert_honors_benchmarks_dir_in_prompt(tmp_path) -> None:
    """`--benchmarks-dir` is honored end-to-end in convert: the codex prompt's
    target path reflects the custom root, matching where the package is
    scaffolded — not the default `benchmarks/`. Regression for the dropped-root
    bug where the scaffold went to the custom dir but codex was told to edit the
    repo's benchmarks/."""
    res = runner.invoke(
        app,
        [
            "eval",
            "adopt",
            "github.com/foo/bar",
            "--name",
            "my-bench",
            "--benchmarks-dir",
            str(tmp_path),
            "--dry-run",
        ],
    )
    assert res.exit_code == 0, res.output
    out = click.unstyle(res.output)
    assert f"{tmp_path}/my-bench" in out


def test_legacy_top_level_adopt_still_works_and_warns(tmp_path) -> None:
    shared._DEPRECATION_WARNED.clear()
    res = runner.invoke(
        app, ["adopt", "init", "legacy-adopt", "--benchmarks-dir", str(tmp_path)]
    )
    assert res.exit_code == 0  # the alias still scaffolds
    assert "Scaffolded" in res.output  # real work happened on stdout
    # the notice is on stderr (res.output mixes both streams; res.stderr is pure)
    # and points at the new canonical scaffold shape. Collapse rich's
    # line-wrapping (the hint can wrap mid-string on narrow consoles).
    assert "deprecation" in res.stderr
    stderr = " ".join(_ANSI_RE.sub("", res.stderr).split())
    assert "bench eval adopt <name> --scaffold-only" in stderr


def test_legacy_agent_create_still_works_and_warns(tmp_path) -> None:
    shared._DEPRECATION_WARNED.clear()
    res = runner.invoke(
        app, ["agent", "create", "legacy-bench", "--benchmarks-dir", str(tmp_path)]
    )
    assert res.exit_code == 0  # the alias still scaffolds
    assert "Scaffolded" in res.output  # real work happened on stdout
    # the notice is on stderr and points at the new canonical scaffold shape
    # (collapse rich's line-wrapping before the substring check).
    assert "deprecation" in res.stderr
    stderr = " ".join(_ANSI_RE.sub("", res.stderr).split())
    assert "bench eval adopt <name> --scaffold-only" in stderr


def test_deprecation_fires_once_per_process(tmp_path) -> None:
    shared._DEPRECATION_WARNED.clear()
    first = runner.invoke(
        app, ["agent", "create", "b1", "--benchmarks-dir", str(tmp_path)]
    )
    second = runner.invoke(
        app, ["agent", "create", "b2", "--benchmarks-dir", str(tmp_path)]
    )
    assert "deprecation" in first.stderr
    assert "deprecation" not in second.stderr  # already warned this process


def test_environment_list_hosted_is_deprecated_json_stays_clean(monkeypatch) -> None:
    # Hosted browsing moved to `bench hub env list`; `environment list
    # --provider`/`--hub` now warn (deprecated) but still work, and the warning
    # goes to stderr ONLY so `--json >out` consumers stay clean.
    import benchflow.hosted_env as hosted

    monkeypatch.setattr(hosted, "prime_env_list", lambda **kw: '{"environments": []}')
    for flag in ("--provider", "--hub"):
        shared._DEPRECATION_WARNED.clear()
        res = runner.invoke(
            app, ["environment", "list", flag, "primeintellect", "--json"]
        )
        assert res.exit_code == 0
        assert "deprecation" in res.stderr and "bench hub env list" in res.stderr
        assert "environments" not in res.stderr  # JSON did not leak to stderr
        assert '{"environments": []}' in res.output  # JSON present on stdout
