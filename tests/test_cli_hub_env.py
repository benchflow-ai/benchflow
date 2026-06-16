"""`bench hub list|show|inspect` is the canonical home for hosted-provider reads.

The overloaded `bench environment` group (gap #5) was split: local sandbox
lifecycle stays on `bench environment` (create/list/cleanup), and the read-only
hosted-provider browsing moved to `bench hub list|show|inspect`. The old
`environment show`/`inspect` and `environment list --provider`/`--hub` remain as
hidden deprecated aliases (stderr notice) through 0.6. The `bench hub env *`
nesting is itself a hidden back-compat alias of the flattened verbs.
"""

from __future__ import annotations

import json

from typer.testing import CliRunner

import benchflow.cli._shared as shared
import benchflow.hosted_env as hosted
import benchflow.hub.harbor_registry as harbor
from benchflow.cli.main import app

runner = CliRunner()

_FAKE_HARBOR = [
    {
        "name": "aider-polyglot",
        "version": "1.0",
        "description": "a coding benchmark",
        "tasks": [{"name": "t1"}, {"name": "t2"}],
    },
    {
        "name": "swe-bench",
        "version": "2.0",
        "description": "issue fixing",
        "tasks": [{"name": "a"}],
    },
]


def test_hub_flattens_verbs_and_hides_env_alias() -> None:
    """`list`/`show`/`inspect`/`check` are canonical under `bench hub`; the old
    `env` nesting is hidden but still resolves for back-compat.

    Asserts against the Click command registry (authoritative), not a regex over
    rendered `--help` (which is ANSI/width-fragile and "env" also appears inside
    the word "environments")."""
    import typer

    hub = typer.main.get_command(app).commands["hub"]
    visible = {
        name for name, sub in hub.commands.items() if not getattr(sub, "hidden", False)
    }
    assert {"list", "show", "inspect", "check"} <= visible
    assert "env" not in visible  # the redundant nesting is hidden…
    assert "env" in hub.commands  # …but still registered for back-compat
    assert runner.invoke(app, ["hub", "env", "list", "--help"]).exit_code == 0


def test_hub_env_alias_still_lists(monkeypatch) -> None:
    """The flattened verbs still resolve under the hidden `bench hub env`."""
    monkeypatch.setattr(harbor, "load_harbor_registry", lambda src: _FAKE_HARBOR)
    res = runner.invoke(app, ["hub", "env", "list", "--provider", "harbor", "--json"])
    assert res.exit_code == 0
    assert [d["name"] for d in json.loads(res.output)] == [
        "aider-polyglot",
        "swe-bench",
    ]


def test_hub_env_list_is_canonical_and_silent(monkeypatch) -> None:
    monkeypatch.setattr(hosted, "prime_env_list", lambda **kw: '{"environments": []}')
    shared._DEPRECATION_WARNED.clear()
    res = runner.invoke(app, ["hub", "list", "--json"])  # defaults provider
    assert res.exit_code == 0
    assert res.output.strip() == '{"environments": []}'
    assert "deprecation" not in res.stderr  # canonical surface never warns


def test_hub_env_show_delegates(monkeypatch) -> None:
    monkeypatch.setattr(hosted, "prime_env_info", lambda ref: "META: general-agent")
    shared._DEPRECATION_WARNED.clear()
    res = runner.invoke(app, ["hub", "show", "primeintellect/general-agent"])
    assert res.exit_code == 0
    assert "META: general-agent" in res.output
    assert "deprecation" not in res.stderr


def test_environment_show_is_deprecated_alias_of_hub_env_show(monkeypatch) -> None:
    monkeypatch.setattr(hosted, "prime_env_info", lambda ref: "META: general-agent")
    shared._DEPRECATION_WARNED.clear()
    res = runner.invoke(app, ["environment", "show", "primeintellect/general-agent"])
    assert res.exit_code == 0
    assert "META: general-agent" in res.output  # behavior identical
    assert "deprecation" in res.stderr and "bench hub show" in res.stderr


def test_environment_inspect_is_deprecated_alias(monkeypatch) -> None:
    monkeypatch.setattr(hosted, "prime_env_inspect", lambda ref, path: f"FILE:{path}")
    shared._DEPRECATION_WARNED.clear()
    res = runner.invoke(app, ["environment", "inspect", "primeintellect/general-agent"])
    assert res.exit_code == 0
    assert "FILE:README.md" in res.output
    assert "deprecation" in res.stderr and "bench hub inspect" in res.stderr


def test_environment_group_is_hidden_but_still_resolves() -> None:
    # The whole `environment` group is now a hidden deprecated alias (local
    # lifecycle → `bench sandbox`, hosted reads → `bench hub env`). It must not
    # appear in top-level `bench --help`, but must still resolve for back-compat.
    #
    # Assert against the Click command registry (authoritative), NOT a regex over
    # rendered `--help` rows: that text is environment-fragile (Rich emits ANSI
    # color codes on CI that break a `│`-anchored match → empty set → false fail).
    import typer

    cmd = typer.main.get_command(app)
    visible = {
        name for name, sub in cmd.commands.items() if not getattr(sub, "hidden", False)
    }
    assert "sandbox" in visible  # canonical local group is visible
    assert "environment" not in visible  # deprecated alias group is hidden
    assert "environment" in cmd.commands  # …but still registered for back-compat
    assert runner.invoke(app, ["environment", "create", "--help"]).exit_code == 0


# ── multi-hub: `bench hub list` browses harbor too, not just primeintellect ─


def test_hub_env_list_harbor_renders_registry(monkeypatch) -> None:
    """`hub env list --provider harbor` lists the Harbor benchmark registry."""
    monkeypatch.setattr(harbor, "load_harbor_registry", lambda src: _FAKE_HARBOR)
    res = runner.invoke(app, ["hub", "list", "--provider", "harbor"])
    assert res.exit_code == 0, res.output
    assert res.exception is None
    assert "Harbor" in res.output  # title/footer name the hub


def test_hub_env_list_harbor_json_is_the_registry(monkeypatch) -> None:
    monkeypatch.setattr(harbor, "load_harbor_registry", lambda src: _FAKE_HARBOR)
    res = runner.invoke(app, ["hub", "list", "--provider", "harbor", "--json"])
    assert res.exit_code == 0
    assert [d["name"] for d in json.loads(res.output)] == [
        "aider-polyglot",
        "swe-bench",
    ]


def test_hub_env_list_harbor_search_filters(monkeypatch) -> None:
    monkeypatch.setattr(harbor, "load_harbor_registry", lambda src: _FAKE_HARBOR)
    res = runner.invoke(
        app,
        ["hub", "list", "--provider", "harbor", "--search", "coding", "--json"],
    )
    assert res.exit_code == 0
    assert [d["name"] for d in json.loads(res.output)] == ["aider-polyglot"]


def test_hub_env_list_harbor_limit(monkeypatch) -> None:
    monkeypatch.setattr(harbor, "load_harbor_registry", lambda src: _FAKE_HARBOR)
    res = runner.invoke(
        app, ["hub", "list", "--provider", "harbor", "--limit", "1", "--json"]
    )
    assert res.exit_code == 0
    assert len(json.loads(res.output)) == 1


def test_hub_env_list_unknown_provider_errors() -> None:
    res = runner.invoke(app, ["hub", "list", "--provider", "bogus"])
    assert res.exit_code == 1
    assert "Unknown --provider" in res.output


def test_hub_env_list_harbor_rejects_owner(monkeypatch) -> None:
    monkeypatch.setattr(harbor, "load_harbor_registry", lambda src: _FAKE_HARBOR)
    res = runner.invoke(app, ["hub", "list", "--provider", "harbor", "--owner", "x"])
    assert res.exit_code == 1
    assert "owner" in res.output.lower()
