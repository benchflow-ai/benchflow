"""Miss-driven remote-manifest auto-load (#876 Phase 2a).

Contract: an unknown ``--agent`` name triggers at most ONE fetch of the pinned
agents source; only DECLARATIVE manifests register (gap-fill — local names and
aliases always win); a broken manifest or unreachable source degrades to the
normal unknown-agent error, never a crash. Tests use a local directory source
(``BENCHFLOW_AGENTS_SOURCE=<dir>``) so nothing touches the network.
"""

from __future__ import annotations

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
