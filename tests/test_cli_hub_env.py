"""`bench hub env` is the canonical home for hosted-provider environment reads.

The overloaded `bench environment` group (gap #5) was split: local sandbox
lifecycle stays on `bench environment` (create/list/cleanup), and the read-only
hosted-provider browsing moved to `bench hub env list|show|inspect`. The old
`environment show`/`inspect` and `environment list --provider`/`--hub` remain as
hidden deprecated aliases (stderr notice) through 0.6.
"""

from __future__ import annotations

from typer.testing import CliRunner

import benchflow.cli._shared as shared
import benchflow.hosted_env as hosted
from benchflow.cli.main import app

runner = CliRunner()


def test_hub_env_subgroup_exposes_list_show_inspect() -> None:
    res = runner.invoke(app, ["hub", "env", "--help"])
    assert res.exit_code == 0
    for verb in ("list", "show", "inspect"):
        assert verb in res.output, f"`bench hub env` missing {verb!r}: {res.output}"
    # and the subgroup is reachable from `bench hub`
    assert "env" in runner.invoke(app, ["hub", "--help"]).output


def test_hub_env_list_is_canonical_and_silent(monkeypatch) -> None:
    monkeypatch.setattr(hosted, "prime_env_list", lambda **kw: '{"environments": []}')
    shared._DEPRECATION_WARNED.clear()
    res = runner.invoke(app, ["hub", "env", "list", "--json"])  # defaults provider
    assert res.exit_code == 0
    assert res.output.strip() == '{"environments": []}'
    assert "deprecation" not in res.stderr  # canonical surface never warns


def test_hub_env_show_delegates(monkeypatch) -> None:
    monkeypatch.setattr(hosted, "prime_env_info", lambda ref: "META: general-agent")
    shared._DEPRECATION_WARNED.clear()
    res = runner.invoke(app, ["hub", "env", "show", "primeintellect/general-agent"])
    assert res.exit_code == 0
    assert "META: general-agent" in res.output
    assert "deprecation" not in res.stderr


def test_environment_show_is_deprecated_alias_of_hub_env_show(monkeypatch) -> None:
    monkeypatch.setattr(hosted, "prime_env_info", lambda ref: "META: general-agent")
    shared._DEPRECATION_WARNED.clear()
    res = runner.invoke(app, ["environment", "show", "primeintellect/general-agent"])
    assert res.exit_code == 0
    assert "META: general-agent" in res.output  # behavior identical
    assert "deprecation" in res.stderr and "bench hub env show" in res.stderr


def test_environment_inspect_is_deprecated_alias(monkeypatch) -> None:
    monkeypatch.setattr(hosted, "prime_env_inspect", lambda ref, path: f"FILE:{path}")
    shared._DEPRECATION_WARNED.clear()
    res = runner.invoke(app, ["environment", "inspect", "primeintellect/general-agent"])
    assert res.exit_code == 0
    assert "FILE:README.md" in res.output
    assert "deprecation" in res.stderr and "bench hub env inspect" in res.stderr


def test_environment_group_is_hidden_but_still_resolves() -> None:
    # The whole `environment` group is now a hidden deprecated alias (local
    # lifecycle → `bench sandbox`, hosted reads → `bench hub env`). It must not
    # appear in top-level `bench --help`, but must still resolve for back-compat.
    import re

    top = runner.invoke(app, ["--help"], terminal_width=200).output
    rows = {m.group(1) for m in re.finditer(r"^\s*│\s+([A-Za-z][\w-]*)\s", top, re.M)}
    assert "sandbox" in rows  # canonical local group is visible
    assert "environment" not in rows  # deprecated alias group is hidden
    assert runner.invoke(app, ["environment", "create", "--help"]).exit_code == 0
