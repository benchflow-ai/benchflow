"""Benchmark adoption is canonically ``bench eval adopt init|convert|verify``.

Adoption lives under ``eval`` because ``eval`` is the universal benchmark entry
point (``eval create`` runs a benchmark; ``eval adopt`` makes a foreign one
runnable). Two prior spellings stay as hidden deprecated aliases for one release
(removed in 0.7), each emitting a one-line stderr notice that points at
``bench eval adopt``:

* the top-level ``bench adopt`` (the 0.6-dev intermediate name), and
* the original ``bench agent create|run|verify``.

``--hub`` likewise remains the deprecated spelling of ``environment list
--provider`` (hosted browsing moved to ``bench hub env list``). These tests pin
the canonical surface and every back-compat path so the renames stay additive.
"""

from __future__ import annotations

import re

from typer.testing import CliRunner

import benchflow.cli._shared as shared
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


def test_eval_adopt_group_exposes_init_convert_verify() -> None:
    res = runner.invoke(app, ["eval", "adopt", "--help"])
    assert res.exit_code == 0
    for verb in ("init", "convert", "verify"):
        assert verb in res.output, f"`bench eval adopt` missing {verb!r}: {res.output}"
    # each subcommand resolves
    for verb in ("init", "convert", "verify"):
        sub = runner.invoke(app, ["eval", "adopt", verb, "--help"])
        assert sub.exit_code == 0, (
            f"`bench eval adopt {verb} --help` failed: {sub.output}"
        )


def test_eval_help_shows_adopt_subgroup() -> None:
    res = runner.invoke(app, ["eval", "--help"])
    assert res.exit_code == 0
    assert "adopt" in res.output, f"`bench eval` should list adopt: {res.output}"


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


def test_eval_adopt_init_is_canonical_and_silent(tmp_path) -> None:
    shared._DEPRECATION_WARNED.clear()
    res = runner.invoke(
        app,
        ["eval", "adopt", "init", "canonical-bench", "--benchmarks-dir", str(tmp_path)],
    )
    assert res.exit_code == 0
    assert "deprecation" not in res.stderr


def test_legacy_top_level_adopt_still_works_and_warns(tmp_path) -> None:
    shared._DEPRECATION_WARNED.clear()
    res = runner.invoke(
        app, ["adopt", "init", "legacy-adopt", "--benchmarks-dir", str(tmp_path)]
    )
    assert res.exit_code == 0  # the alias still scaffolds
    assert "Scaffolded" in res.output  # real work happened on stdout
    # the notice is on stderr (res.output mixes both streams; res.stderr is pure)
    assert "deprecation" in res.stderr and "bench eval adopt init" in res.stderr


def test_legacy_agent_create_still_works_and_warns(tmp_path) -> None:
    shared._DEPRECATION_WARNED.clear()
    res = runner.invoke(
        app, ["agent", "create", "legacy-bench", "--benchmarks-dir", str(tmp_path)]
    )
    assert res.exit_code == 0  # the alias still scaffolds
    assert "Scaffolded" in res.output  # real work happened on stdout
    # the notice is on stderr and points at the canonical command
    assert "deprecation" in res.stderr and "bench eval adopt init" in res.stderr


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
