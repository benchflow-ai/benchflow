"""Miss-driven remote-manifest auto-load (#876 Phase 2a).

Contract: an unknown ``--agent`` name triggers at most ONE fetch of the pinned
agents source; only DECLARATIVE manifests register (gap-fill — local names and
aliases always win); a broken manifest or unreachable source degrades to the
normal unknown-agent error, never a crash. Tests use a local directory source
(``BENCHFLOW_AGENTS_SOURCE=<dir>``) so nothing touches the network.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from benchflow.agents import registry, remote_manifests
from benchflow.agents.registry import resolve_agent

_MANIFEST = """\
contract_version = "1.0"
name = "{name}"
description = "test agent"
protocol = "acp"
install_cmd = "true"
launch_cmd = "true"
{extra}
"""


def _write_manifest(root, dirname, name, extra=""):
    d = root / dirname
    d.mkdir(parents=True)
    (d / "manifest.toml").write_text(_MANIFEST.format(name=name, extra=extra))


@pytest.fixture()
def source_dir(tmp_path, monkeypatch):
    monkeypatch.delenv(remote_manifests.AGENTS_DIR_ENV, raising=False)
    monkeypatch.setenv(remote_manifests.AGENTS_SOURCE_ENV, str(tmp_path))
    remote_manifests._reset_for_tests()
    maps = (
        registry.AGENTS,
        registry.AGENT_ALIASES,
        registry.AGENT_INSTALLERS,
        registry.AGENT_LAUNCH,
    )
    snapshots = tuple(mapping.copy() for mapping in maps)
    registered: list[str] = []
    yield tmp_path, registered
    remote_manifests._reset_for_tests()
    for mapping, snapshot in zip(maps, snapshots, strict=True):
        mapping.clear()
        mapping.update(snapshot)


def test_unknown_agent_triggers_autoload_and_resolves(source_dir):
    root, registered = source_dir
    _write_manifest(root, "probe-remote", "probe-remote")
    registered.append("probe-remote")
    assert resolve_agent("probe-remote").name == "probe-remote"


def test_gap_fill_never_overwrites_local(source_dir):
    root, registered = source_dir
    # remote manifest reuses an existing core name with different commands.
    _write_manifest(root, "mimo", "mimo")
    _write_manifest(root, "probe-remote2", "probe-remote2")
    registered.append("probe-remote2")
    before = registry.AGENTS["mimo"]
    resolve_agent("probe-remote2")  # triggers the load
    assert registry.AGENTS["mimo"] is before  # untouched


def test_colliding_alias_is_stripped_not_fatal(source_dir):
    root, registered = source_dir
    # alias "claude" already maps to claude-agent-acp locally.
    _write_manifest(
        root, "probe-remote3", "probe-remote3", extra='aliases = ["claude"]\n'
    )
    registered.append("probe-remote3")
    assert resolve_agent("probe-remote3").name == "probe-remote3"
    assert registry.AGENT_ALIASES["claude"] == "claude-agent-acp"


@pytest.mark.parametrize(
    ("first_alias", "second_alias"),
    [("shared-batch-alias", "shared-batch-alias"), ("probe-b", "")],
)
def test_batch_alias_collisions_are_typed_not_raw(
    source_dir, first_alias, second_alias
):
    """Guards PR #1090 against raw alias collisions during catalog commit."""
    root, registered = source_dir
    _write_manifest(root, "a", "probe-a", f'aliases = ["{first_alias}"]\n')
    extra = f'aliases = ["{second_alias}"]\n' if second_alias else ""
    _write_manifest(root, "b", "probe-b", extra)
    registered.extend(("probe-a", "probe-b"))

    assert resolve_agent("probe-a").name == "probe-a"
    assert resolve_agent("probe-b").name == "probe-b"
    assert any(
        issue.kind.value == "collision"
        for issue in remote_manifests.ensure_manifest_catalog().issues
    )


def test_broken_manifest_skipped_others_load(source_dir, caplog):
    root, registered = source_dir
    (root / "broken").mkdir()
    (root / "broken" / "manifest.toml").write_text("not toml [[[")
    _write_manifest(root, "probe-remote4", "probe-remote4")
    registered.append("probe-remote4")
    assert resolve_agent("probe-remote4").name == "probe-remote4"


def test_off_disables_and_error_mentions_source(source_dir, monkeypatch):
    monkeypatch.setenv(remote_manifests.AGENTS_SOURCE_ENV, "off")
    with pytest.raises(KeyError) as exc:
        resolve_agent("agent-that-definitely-does-not-exist")
    assert "disabled" in str(exc.value)


def test_one_shot_per_process(source_dir, monkeypatch):
    _root, _registered = source_dir
    calls: list[int] = []
    real = remote_manifests._source_root

    def counting(spec):
        calls.append(1)
        return real(spec)

    monkeypatch.setattr(remote_manifests, "_source_root", counting)
    with pytest.raises(KeyError):
        resolve_agent("nope-1")
    with pytest.raises(KeyError):
        resolve_agent("nope-2")
    assert len(calls) == 1


def test_acp_namespace_retries_after_catalog_autoload(source_dir):
    """Guards PR #1090 generic acp:<id> catalog resolution."""
    root, registered = source_dir
    _write_manifest(root, "probe-namespace", "probe-namespace")
    registered.append("probe-namespace")
    assert resolve_agent("acp:probe-namespace").name == "probe-namespace"


def test_directory_override_wins_over_source(source_dir, monkeypatch, tmp_path):
    """Guards PR #1090 single catalog-source precedence."""
    source, registered = source_dir
    local = tmp_path / "local"
    remote = tmp_path / "remote"
    _write_manifest(local, "probe-local", "probe-local")
    _write_manifest(remote, "probe-source", "probe-source")
    monkeypatch.setenv(remote_manifests.AGENTS_DIR_ENV, str(local))
    monkeypatch.setenv(remote_manifests.AGENTS_SOURCE_ENV, str(remote))
    registered.append("probe-local")

    assert resolve_agent("probe-local").name == "probe-local"
    assert remote_manifests.ensure_manifest_catalog().source == str(local)
    assert "probe-source" not in registry.AGENTS
    assert source != local


def test_resolve_then_listing_reuses_applied_catalog(source_dir, monkeypatch):
    """Guards PR #1090 against listing colliding with its applied catalog."""
    root, registered = source_dir
    _write_manifest(root, "probe-list", "probe-list", 'aliases = ["probe-short"]\n')
    registered.append("probe-list")
    calls = 0
    real = remote_manifests._source_root

    def counted(request):
        nonlocal calls
        calls += 1
        return real(request)

    monkeypatch.setattr(remote_manifests, "_source_root", counted)
    assert resolve_agent("probe-list").name == "probe-list"
    listing = remote_manifests.manifest_catalog_for_listing()
    assert [manifest.config.name for manifest in listing.manifests] == ["probe-list"]
    assert not any(issue.kind.value == "collision" for issue in listing.issues)
    assert calls == 1


def test_failed_registration_is_not_published(source_dir, monkeypatch):
    """Guards PR #1090 registry commit preceding snapshot publication."""
    root, registered = source_dir
    _write_manifest(root, "probe-retry", "probe-retry")
    registered.append("probe-retry")
    real = remote_manifests._register_catalog
    calls = 0

    def fail_once(catalog, loaded, *, local_override):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("injected before commit")
        return real(catalog, loaded, local_override=local_override)

    monkeypatch.setattr(remote_manifests, "_register_catalog", fail_once)
    with pytest.raises(RuntimeError, match="injected before commit"):
        remote_manifests.ensure_manifest_catalog()
    assert remote_manifests._snapshot is None
    assert resolve_agent("probe-retry").name == "probe-retry"
    assert calls == 2


def test_source_diagnostics_are_sanitized(source_dir, monkeypatch):
    """Guards PR #1090 against catalog-source secret leakage."""
    monkeypatch.setenv(
        remote_manifests.AGENTS_SOURCE_ENV,
        "https://user:password@example.invalid/repo@feature/x?token=secret#fragment",
    )
    monkeypatch.setattr(
        remote_manifests,
        "_source_root",
        lambda _: (_ for _ in ()).throw(OSError("raw secret cause")),
    )
    catalog = remote_manifests.ensure_manifest_catalog()
    assert catalog.source == "https://***@example.invalid/repo"
    assert catalog.ref == "feature/x"
    assert catalog.issues[0].kind.value == "unreachable"
    assert catalog.issues[0].cause == "OSError"
    diagnostic = " ".join((catalog.source, *catalog.warnings))
    assert not any(
        secret in diagnostic
        for secret in ("user", "password", "token", "secret", "fragment")
    )


def test_cli_listing_keeps_valid_sibling_and_reports_broken_once(tmp_path):
    """Guards PR #1090 generic partial-catalog listing."""
    _write_manifest(
        tmp_path, "probe-visible", "probe-visible", 'aliases = ["probe-alias"]\n'
    )
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "manifest.toml").write_text("not toml [[[\n")
    env = os.environ.copy()
    env.pop(remote_manifests.AGENTS_DIR_ENV, None)
    env[remote_manifests.AGENTS_SOURCE_ENV] = str(tmp_path)
    result = subprocess.run(
        [sys.executable, "-m", "benchflow.cli.main", "agent", "list"],
        capture_output=True,
        text=True,
        env=env,
        timeout=20,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0
    assert "probe-visible" in output
    assert "probe-alias" in output
    assert output.count("Agent catalog incomplete") == 1
    assert output.count("broken/manifest.toml") == 1
