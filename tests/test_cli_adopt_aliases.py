"""The benchmark-adoption verbs moved from ``bench agent`` to ``bench adopt``.

`bench adopt init|convert|verify` is canonical; the legacy
`bench agent create|run|verify` stay as hidden deprecated aliases (one-release
window) emitting a stderr deprecation notice, and `--hub` is the deprecated
spelling of `environment list --provider`. These tests pin both the new surface
and the back-compat behavior so the rename is additive (no hard break).
"""

from __future__ import annotations

from typer.testing import CliRunner

import benchflow.cli._shared as shared
from benchflow.cli.main import app

runner = CliRunner()


def test_adopt_group_exposes_init_convert_verify() -> None:
    res = runner.invoke(app, ["adopt", "--help"])
    assert res.exit_code == 0
    for verb in ("init", "convert", "verify"):
        assert verb in res.output, f"`bench adopt` missing {verb!r}: {res.output}"
    # each subcommand resolves
    for verb in ("init", "convert", "verify"):
        sub = runner.invoke(app, ["adopt", verb, "--help"])
        assert sub.exit_code == 0, f"`bench adopt {verb} --help` failed: {sub.output}"


def test_agent_help_shows_management_hides_adoption() -> None:
    res = runner.invoke(app, ["agent", "--help"])
    assert res.exit_code == 0
    assert "list" in res.output and "show" in res.output
    # adoption verbs are hidden on the agent group now
    for verb in ("create", "run", "verify"):
        assert verb not in res.output, f"deprecated {verb!r} still shown in agent help"


def test_legacy_agent_create_still_works_and_warns(tmp_path) -> None:
    shared._DEPRECATION_WARNED.clear()
    res = runner.invoke(
        app, ["agent", "create", "legacy-bench", "--benchmarks-dir", str(tmp_path)]
    )
    assert res.exit_code == 0  # the alias still scaffolds
    assert "Scaffolded" in res.output  # real work happened on stdout
    # the notice is on stderr (res.output mixes both streams; res.stderr is pure)
    assert "deprecation" in res.stderr and "bench adopt init" in res.stderr


def test_adopt_init_is_canonical_and_silent(tmp_path) -> None:
    shared._DEPRECATION_WARNED.clear()
    res = runner.invoke(
        app, ["adopt", "init", "canonical-bench", "--benchmarks-dir", str(tmp_path)]
    )
    assert res.exit_code == 0
    assert "deprecation" not in res.stderr


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
