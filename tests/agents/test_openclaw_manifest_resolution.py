"""OpenClaw remains fail-closed after extraction from BenchFlow core."""

from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from benchflow.agents import registry, remote_manifests

_SPECS = (
    "openclaw",
    "acp/openclaw",
    "acpx/openclaw",
    "acp:openclaw",
    " ACP:OPENCLAW ",
)
_MANIFEST = """\
contract_version = "1.0"
name = "openclaw"
aliases = ["openclaw"]
description = "OpenClaw fixture"
protocol = "acp"
install_cmd = "install-openclaw"
launch_cmd = "openclaw-acp-shim"
"""


def _source(tmp_path: Path) -> Path:
    target = tmp_path / "acp" / "openclaw"
    target.mkdir(parents=True)
    (target / "manifest.toml").write_text(_MANIFEST)
    return tmp_path


def _manifest(root: Path, rel: str, body: str) -> None:
    target = root / rel / "manifest.toml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body)


@pytest.fixture(autouse=True)
def _clean_runtime_catalog(monkeypatch):
    """Guards PR B for issue #1090 against process-state order dependence."""
    monkeypatch.delenv("BENCHFLOW_AGENTS_DIR", raising=False)
    remote_manifests._reset_for_tests()
    for mapping in (
        registry.AGENTS,
        registry.AGENT_INSTALLERS,
        registry.AGENT_LAUNCH,
        registry.AGENT_ALIASES,
    ):
        mapping.pop("openclaw", None)
        mapping.pop("fixture-foo", None)
        mapping.pop("acpx:openclaw", None)
    for alias, target in list(registry.AGENT_ALIASES.items()):
        if target in {"openclaw", "fixture-foo", "not-openclaw"}:
            registry.AGENT_ALIASES.pop(alias, None)
    mappings = (
        registry.AGENTS,
        registry.AGENT_INSTALLERS,
        registry.AGENT_LAUNCH,
        registry.AGENT_ALIASES,
    )
    snapshots = tuple(mapping.copy() for mapping in mappings)
    yield
    remote_manifests._reset_for_tests()
    for mapping, snapshot in zip(mappings, snapshots, strict=True):
        mapping.clear()
        mapping.update(snapshot)


def test_openclaw_absent_before_catalog_then_registers(monkeypatch, tmp_path):
    """Guards PR B for issue #1090: agents repository is sole config owner."""
    monkeypatch.setenv(remote_manifests.AGENTS_SOURCE_ENV, str(_source(tmp_path)))
    assert "openclaw" not in registry.AGENTS
    assert "openclaw" not in registry.AGENT_ALIASES
    assert "openclaw" not in registry.AGENT_INSTALLERS
    assert "openclaw" not in registry.AGENT_LAUNCH
    assert registry.MIGRATED_MANIFESTS["openclaw"] == ("acp/openclaw/manifest.toml")

    config = registry.resolve_agent("openclaw")
    assert config.name == "openclaw"
    assert registry.AGENT_ALIASES["openclaw"] == "openclaw"
    assert registry.AGENT_INSTALLERS["openclaw"] == config.install_cmd
    assert registry.AGENT_LAUNCH["openclaw"] == config.launch_cmd


@pytest.mark.parametrize("spec", _SPECS)
def test_disabled_openclaw_is_typed_and_never_raw_fallback(monkeypatch, spec):
    """Guards PR B for issue #1090 against PATH-collision fallback."""
    monkeypatch.setenv(remote_manifests.AGENTS_SOURCE_ENV, "off")
    with pytest.raises(registry.AgentManifestResolutionError) as first:
        registry.resolve_agent(spec)
    with pytest.raises(registry.AgentManifestResolutionError) as second:
        registry.resolve_agent_key(spec)
    assert first.value is not second.value
    assert first.value.category == second.value.category == "disabled"
    assert str(first.value) == str(second.value)


def test_reserved_detection_precedes_alias_lookup(monkeypatch):
    """Guards PR B for issue #1090 against plugin/PATH identity shadowing."""
    monkeypatch.setenv(remote_manifests.AGENTS_SOURCE_ENV, "off")
    registry.AGENT_ALIASES["openclaw"] = "mimo"
    with pytest.raises(registry.AgentManifestResolutionError):
        registry.resolve_agent("openclaw")


def test_direct_local_collision_is_typed(monkeypatch, tmp_path):
    """Guards PR B for issue #1090 against local reserved-id replacement."""
    monkeypatch.setenv(remote_manifests.AGENTS_SOURCE_ENV, str(_source(tmp_path)))
    registry.AGENTS["openclaw"] = registry.AgentConfig(
        name="openclaw", install_cmd="local", launch_cmd="local"
    )
    with pytest.raises(registry.AgentManifestResolutionError) as exc:
        registry.resolve_agent("openclaw")
    assert exc.value.category == "collision"
    catalog = remote_manifests.ensure_manifest_catalog()
    assert [
        issue.kind.value
        for issue in catalog.issues
        if issue.path == "acp/openclaw/manifest.toml"
    ].count("collision") == 1


def test_rejected_local_record_does_not_poison_openclaw_alias(monkeypatch, tmp_path):
    """Guards PR #1090 against rejected records owning catalog aliases."""
    source = _source(tmp_path)
    openclaw = source / "acp" / "openclaw" / "manifest.toml"
    openclaw.write_text(
        _MANIFEST.replace('["openclaw"]', '["openclaw", "shared-alias"]')
    )
    _manifest(
        source,
        "acp/mimo",
        _MANIFEST.replace('name = "openclaw"', 'name = "mimo"').replace(
            '["openclaw"]', '["shared-alias"]'
        ),
    )
    monkeypatch.setenv(remote_manifests.AGENTS_SOURCE_ENV, str(source))

    assert registry.resolve_agent("openclaw").name == "openclaw"
    assert registry.AGENT_ALIASES["shared-alias"] == "openclaw"


def test_reserved_manifest_survives_stripped_alias_warning(monkeypatch, tmp_path):
    """Guards PR #1090: nonfatal alias warnings must not reject OpenClaw."""
    source = _source(tmp_path)
    manifest = source / "acp" / "openclaw" / "manifest.toml"
    manifest.write_text(_MANIFEST.replace('"openclaw"]', '"openclaw", "claude"]'))
    monkeypatch.setenv(remote_manifests.AGENTS_SOURCE_ENV, str(source))

    assert registry.resolve_agent("openclaw").name == "openclaw"
    catalog = remote_manifests.ensure_manifest_catalog()
    assert catalog.issue_for("openclaw") is None
    assert any("alias 'claude' collides" in warning for warning in catalog.warnings)


def test_both_env_paths_share_one_ingestion_plane(tmp_path):
    """Guards PR #1090 against local/source self-collision."""
    source = _source(tmp_path)
    env = os.environ.copy()
    env[remote_manifests.AGENTS_SOURCE_ENV] = str(source)
    env["BENCHFLOW_AGENTS_DIR"] = str(source)
    result = subprocess.run(
        [sys.executable, "-m", "benchflow.cli.main", "agent", "show", "openclaw"],
        capture_output=True,
        text=True,
        env=env,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "openclaw" in result.stdout


@pytest.mark.parametrize("spec", _SPECS)
@pytest.mark.parametrize(
    ("case", "category"),
    [
        ("absent", "missing"),
        ("malformed", "malformed"),
        ("schema", "malformed"),
        ("missing-contract", "malformed"),
        ("case-name", "malformed"),
        ("space-name", "malformed"),
        ("incompatible", "incompatible"),
        ("duplicate-name", "duplicate"),
        ("duplicate-alias", "duplicate"),
    ],
)
def test_invalid_catalog_cases_are_typed_for_both_public_apis(
    monkeypatch, tmp_path, spec, case, category
):
    """Guards PR B for issue #1090 across reserved identity failure classes."""
    marker = tmp_path / "path-openclaw-ran"
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    executable = binary_dir / "openclaw"
    executable.write_text(f"#!/bin/sh\ntouch {marker}\n")
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", f"{binary_dir}:{os.environ.get('PATH', '')}")
    if case == "malformed":
        _manifest(tmp_path, "acp/openclaw", "not toml [[[")
    elif case == "schema":
        _manifest(
            tmp_path,
            "acp/openclaw",
            _MANIFEST.replace('name = "openclaw"', "name = 123"),
        )
    elif case == "missing-contract":
        _manifest(
            tmp_path,
            "acp/openclaw",
            _MANIFEST.replace('contract_version = "1.0"\n', ""),
        )
    elif case in {"case-name", "space-name"}:
        replacement = (
            'name = "OpenClaw"' if case == "case-name" else 'name = " openclaw "'
        )
        _manifest(
            tmp_path,
            "acp/openclaw",
            _MANIFEST.replace('name = "openclaw"', replacement),
        )
    elif case == "incompatible":
        _manifest(
            tmp_path,
            "acp/openclaw",
            _MANIFEST.replace('contract_version = "1.0"', 'contract_version = "2.0"'),
        )
    elif case.startswith("duplicate"):
        _manifest(tmp_path, "acp/openclaw", _MANIFEST)
        body = _MANIFEST.replace(
            'name = "openclaw"',
            'name = "openclaw"' if case == "duplicate-name" else 'name = "other"',
        )
        _manifest(tmp_path, "acp/other", body)
    monkeypatch.setenv(remote_manifests.AGENTS_SOURCE_ENV, str(tmp_path))
    calls = 0
    real_source_root = remote_manifests._source_root

    def counted(request):
        nonlocal calls
        calls += 1
        return real_source_root(request)

    monkeypatch.setattr(remote_manifests, "_source_root", counted)
    with pytest.raises(registry.AgentManifestResolutionError) as direct:
        registry.resolve_agent(spec)
    with pytest.raises(registry.AgentManifestResolutionError) as key:
        registry.resolve_agent_key(spec)
    assert direct.value.category == key.value.category == category
    assert direct.value is not key.value
    assert calls == 1
    assert not marker.exists()
    catalog = remote_manifests.ensure_manifest_catalog()
    reserved_issues = [
        issue
        for issue in catalog.issues
        if issue.path == "acp/openclaw/manifest.toml" and issue.kind.value == category
    ]
    assert len(reserved_issues) == 1


@pytest.mark.parametrize("spec", _SPECS)
def test_unreachable_source_is_typed(monkeypatch, spec):
    """Guards PR B for issue #1090 against fetch errors becoming raw commands."""
    monkeypatch.setenv(remote_manifests.AGENTS_SOURCE_ENV, "owner/repo@deadbeef")
    monkeypatch.setattr(
        remote_manifests,
        "_source_root",
        lambda _: (_ for _ in ()).throw(OSError("secret-bearing failure")),
    )
    with pytest.raises(registry.AgentManifestResolutionError) as direct:
        registry.resolve_agent(spec)
    with pytest.raises(registry.AgentManifestResolutionError) as key:
        registry.resolve_agent_key(spec)
    assert direct.value.category == key.value.category == "unreachable"
    assert direct.value is not key.value
    assert "secret-bearing" not in str(direct.value)
    result = remote_manifests.ensure_manifest_catalog()
    assert result.issue_for("openclaw").cause == "OSError"


def test_source_and_ref_are_sanitized_once(monkeypatch):
    """Guards PR B for issue #1090 against source diagnostic secret leakage."""
    source = "https://user:password@example.invalid/repo@main?token=secret#fragment"
    monkeypatch.setenv(remote_manifests.AGENTS_SOURCE_ENV, source)
    monkeypatch.setattr(
        remote_manifests,
        "_source_root",
        lambda _: (_ for _ in ()).throw(OSError("raw secret cause")),
    )
    with pytest.raises(registry.AgentManifestResolutionError) as exc:
        registry.resolve_agent("openclaw")
    diagnostic = str(exc.value)
    assert "user" not in diagnostic
    assert "password" not in diagnostic
    assert "token" not in diagnostic
    assert "secret" not in diagnostic
    assert "fragment" not in diagnostic
    assert exc.value.source == "https://***@example.invalid/repo"
    assert exc.value.ref == "main"


def test_slash_branch_ref_preserved(monkeypatch):
    """Guards PR B for issue #1090 owner/repo@feature/foo source contract."""
    monkeypatch.setenv(
        remote_manifests.AGENTS_SOURCE_ENV, "benchflow-ai/agents@feature/foo"
    )
    seen = []

    def fail(request):
        seen.append(request)
        raise OSError("offline")

    monkeypatch.setattr(remote_manifests, "_source_root", fail)
    with pytest.raises(registry.AgentManifestResolutionError) as exc:
        registry.resolve_agent("openclaw")
    assert exc.value.source == "benchflow-ai/agents"
    assert exc.value.ref == "feature/foo"
    assert seen[0].raw_source == "benchflow-ai/agents"
    assert seen[0].raw_ref == "feature/foo"


def test_url_userinfo_without_ref_is_not_misparsed():
    """Guards PR B for issue #1090 userinfo/ref delimiter ambiguity."""
    parsed = remote_manifests._parse_source("https://user:pass@example.test/repo")
    assert parsed.source == "https://***@example.test/repo"
    assert parsed.ref == ""
    assert parsed.raw_source == "https://user:pass@example.test/repo"


def test_fresh_process_cli_list_keeps_rows_aliases_and_one_warning(tmp_path):
    """Guards PR B for issue #1090 list discovery under partial failure."""
    _manifest(tmp_path, "acp/openclaw", _MANIFEST)
    _manifest(
        tmp_path,
        "acp/foo",
        _MANIFEST.replace('name = "openclaw"', 'name = "visible-foo"').replace(
            'aliases = ["openclaw"]', 'aliases = ["visible-alias"]'
        ),
    )
    _manifest(tmp_path, "acp/broken", "bad [[[")
    env = os.environ.copy()
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
    assert "visible-foo" in output
    assert "visible-alias" in output
    assert output.count("Agent catalog incomplete") == 1
    assert output.count("acp/broken/manifest.toml") == 1


def test_fresh_process_cli_list_disabled_warns_once():
    """Guards PR B for issue #1090 disabled listing total-result contract."""
    env = os.environ.copy()
    env[remote_manifests.AGENTS_SOURCE_ENV] = "off"
    result = subprocess.run(
        [sys.executable, "-m", "benchflow.cli.main", "agent", "list"],
        capture_output=True,
        text=True,
        env=env,
        timeout=20,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0
    assert output.count("Agent catalog incomplete") == 1
    assert output.count("agents source disabled") == 1


def test_fresh_process_cli_list_reports_missing_reserved_manifest(tmp_path):
    """Guards PR #1090 generated reserved issues reaching list diagnostics."""
    env = os.environ.copy()
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
    assert output.count("Agent catalog incomplete") == 1
    assert output.count("reserved manifest 'openclaw' is absent") == 1


def test_reserved_parse_listing_reports_issue_once(tmp_path, monkeypatch):
    """Guards PR B for issue #1090 against duplicate reserved parse warnings."""
    _manifest(tmp_path, "acp/openclaw", "bad [[[")
    monkeypatch.setenv(remote_manifests.AGENTS_SOURCE_ENV, str(tmp_path))
    result = remote_manifests.manifest_catalog_for_listing()
    assert len(result.warnings) == 1
    assert result.warnings[0].startswith("acp/openclaw/manifest.toml: malformed:")


def test_unknown_command_remains_exact_after_catalog_failure(monkeypatch):
    """Guards PR B for issue #1090: strict tombstone does not break raw commands."""
    monkeypatch.setenv(remote_manifests.AGENTS_SOURCE_ENV, "off")
    assert registry.resolve_agent_key("some-explicit-command --flag") == (
        "some-explicit-command --flag"
    )
    with pytest.raises(registry.AgentManifestResolutionError):
        registry.resolve_agent("openclaw")


def test_evaluation_validation_propagates_before_sandbox(monkeypatch, tmp_path):
    """Guards PR B for issue #1090 against Evaluation raw-launch fallback."""
    from benchflow.evaluation import EvaluationConfig

    monkeypatch.setenv(remote_manifests.AGENTS_SOURCE_ENV, "off")
    executable = tmp_path / "openclaw"
    marker = tmp_path / "executed"
    executable.write_text(f"#!/bin/sh\ntouch {marker}\n")
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ.get('PATH', '')}")
    with pytest.raises(registry.AgentManifestResolutionError):
        EvaluationConfig(agent="openclaw")
    assert not marker.exists()


@pytest.mark.parametrize("spec", _SPECS)
def test_concurrent_failures_share_status_not_exception(monkeypatch, tmp_path, spec):
    """Guards PR B for issue #1090 against concurrent result replacement."""
    monkeypatch.setenv(remote_manifests.AGENTS_SOURCE_ENV, str(tmp_path))
    calls = 0
    real_source_root = remote_manifests._source_root

    def counted(request):
        nonlocal calls
        calls += 1
        return real_source_root(request)

    monkeypatch.setattr(remote_manifests, "_source_root", counted)

    def lookup(_):
        try:
            registry.resolve_agent(spec)
        except registry.AgentManifestResolutionError as exc:
            return exc
        raise AssertionError("resolution unexpectedly succeeded")

    with ThreadPoolExecutor(max_workers=4) as pool:
        errors = list(pool.map(lookup, range(4)))
    assert len({id(exc) for exc in errors}) == 4
    assert {str(exc) for exc in errors} == {str(errors[0])}
    assert {exc.category for exc in errors} == {"missing"}
    assert calls == 1


def test_listing_failure_does_not_poison_runtime(monkeypatch, tmp_path):
    """Guards PR B for issue #1090: listing and runtime have independent state."""
    monkeypatch.setenv(remote_manifests.AGENTS_SOURCE_ENV, "off")
    listing = remote_manifests.manifest_catalog_for_listing()
    assert listing.manifests == ()
    monkeypatch.setenv(remote_manifests.AGENTS_SOURCE_ENV, str(_source(tmp_path)))
    assert registry.resolve_agent("openclaw").name == "openclaw"


def test_directory_override_wins_over_configured_source(monkeypatch, tmp_path):
    """Guards PR #1090 single-source precedence when both env vars are set."""
    local = tmp_path / "local"
    remote = tmp_path / "remote"
    _source(local)
    _manifest(
        remote,
        "acp/other",
        _MANIFEST.replace('name = "openclaw"', 'name = "source-only"').replace(
            'aliases = ["openclaw"]', "aliases = []"
        ),
    )
    monkeypatch.setenv(remote_manifests.AGENTS_DIR_ENV, str(local))
    monkeypatch.setenv(remote_manifests.AGENTS_SOURCE_ENV, str(remote))

    assert registry.resolve_agent("openclaw").name == "openclaw"
    assert remote_manifests.ensure_manifest_catalog().source == str(local)
    assert "source-only" not in registry.AGENTS


def test_blank_directory_falls_back_to_source(monkeypatch, tmp_path):
    """Guards PR #1090 blank local override source selection."""
    source = _source(tmp_path)
    monkeypatch.setenv(remote_manifests.AGENTS_DIR_ENV, "  ")
    monkeypatch.setenv(remote_manifests.AGENTS_SOURCE_ENV, str(source))
    assert registry.resolve_agent("openclaw").name == "openclaw"
    assert remote_manifests.ensure_manifest_catalog().source == str(source)


def test_resolve_then_listing_reuses_applied_catalog(monkeypatch, tmp_path):
    """Guards PR #1090 against listing colliding with its applied catalog."""
    monkeypatch.setenv(remote_manifests.AGENTS_SOURCE_ENV, str(_source(tmp_path)))
    calls = 0
    real = remote_manifests._source_root

    def counted(request):
        nonlocal calls
        calls += 1
        return real(request)

    monkeypatch.setattr(remote_manifests, "_source_root", counted)
    assert registry.resolve_agent("openclaw").name == "openclaw"
    listing = remote_manifests.manifest_catalog_for_listing()
    assert [manifest.config.name for manifest in listing.manifests] == ["openclaw"]
    assert not any(issue.kind.value == "collision" for issue in listing.issues)
    assert calls == 1


def test_resolve_then_cli_list_uses_registered_nonself_alias_once(
    monkeypatch, tmp_path
):
    """Guards PR #1090 applied listing against catalog alias self-overlay."""
    from typer.testing import CliRunner

    from benchflow.cli import agent as agent_cli
    from benchflow.cli.main import app

    monkeypatch.setattr(agent_cli.console, "_width", 200)

    source = _source(tmp_path)
    manifest = source / "acp" / "openclaw" / "manifest.toml"
    manifest.write_text(_MANIFEST.replace('["openclaw"]', '["openclaw", "oc-short"]'))
    monkeypatch.setenv(remote_manifests.AGENTS_SOURCE_ENV, str(source))
    calls = 0
    real = remote_manifests._source_root

    def counted(request):
        nonlocal calls
        calls += 1
        return real(request)

    monkeypatch.setattr(remote_manifests, "_source_root", counted)
    assert registry.resolve_agent("openclaw").name == "openclaw"
    result = CliRunner().invoke(app, ["agent", "list"])
    assert result.exit_code == 0, result.output
    assert result.output.count("oc-short") == 1
    assert "collision" not in result.output
    assert calls == 1


def test_preview_cli_list_dedupes_same_owner_core_aliases(monkeypatch, tmp_path):
    """Guards PR #1090 preview alias overlay against duplicate display."""
    from typer.testing import CliRunner

    from benchflow.cli import agent as agent_cli
    from benchflow.cli.main import app

    monkeypatch.setattr(agent_cli.console, "_width", 200)

    for name, alias in (
        ("claude-agent-acp", "claude"),
        ("codex-acp", "codex"),
    ):
        _manifest(
            tmp_path,
            f"acp/{name}",
            _MANIFEST.replace('name = "openclaw"', f'name = "{name}"').replace(
                'aliases = ["openclaw"]', f'aliases = ["{alias}"]'
            ),
        )
    monkeypatch.setenv(remote_manifests.AGENTS_DIR_ENV, str(tmp_path))
    monkeypatch.setenv(remote_manifests.AGENTS_SOURCE_ENV, "off")

    result = CliRunner().invoke(app, ["agent", "list"])
    assert result.exit_code == 0, result.output
    claude_row = next(
        line for line in result.output.splitlines() if "claude-agent-acp" in line
    )
    codex_row = next(line for line in result.output.splitlines() if "codex-acp" in line)
    assert claude_row.count("claude") == 2  # canonical name + one alias
    assert codex_row.count("codex") == 2  # canonical name + one alias
    assert remote_manifests.manifest_catalog_for_listing().applied is False


def test_failed_registry_application_is_not_published(monkeypatch, tmp_path):
    """Guards PR #1090 against terminal snapshot publication before commit."""
    monkeypatch.setenv(remote_manifests.AGENTS_SOURCE_ENV, str(_source(tmp_path)))
    real = remote_manifests._register_catalog
    maps = (
        registry.AGENTS,
        registry.AGENT_ALIASES,
        registry.AGENT_INSTALLERS,
        registry.AGENT_LAUNCH,
    )
    before = tuple(mapping.copy() for mapping in maps)
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
    assert tuple(mapping.copy() for mapping in maps) == before
    assert registry.resolve_agent("openclaw").name == "openclaw"
    assert calls == 2


def test_catalog_and_supported_local_writer_have_stable_winner(monkeypatch, tmp_path):
    """Guards PR #1090 registry/catalog race with one typed terminal outcome."""
    monkeypatch.setenv(remote_manifests.AGENTS_SOURCE_ENV, str(_source(tmp_path)))

    def load_catalog():
        return remote_manifests.ensure_manifest_catalog()

    def register_local():
        try:
            return registry.register_agent("openclaw", "local", "local")
        except Exception as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        catalog_future = pool.submit(load_catalog)
        local_future = pool.submit(register_local)
        catalog = catalog_future.result()
        local = local_future.result()

    issue = catalog.issue_for("openclaw")
    if issue is None:
        assert isinstance(local, remote_manifests.AgentManifestError)
        assert registry.resolve_agent("openclaw").install_cmd == "install-openclaw"
    else:
        assert issue.kind.value == "collision"
        with pytest.raises(registry.AgentManifestResolutionError) as first:
            registry.resolve_agent("openclaw")
        with pytest.raises(registry.AgentManifestResolutionError) as second:
            registry.resolve_agent("openclaw")
        assert first.value.category == second.value.category == "collision"


def test_successful_runtime_source_cannot_switch(monkeypatch, tmp_path):
    """Guards PR B for issue #1090 terminal successful-source immutability."""
    first_source = tmp_path / "first"
    second_source = tmp_path / "second"
    _source(first_source)
    second_source.mkdir()
    monkeypatch.setenv(remote_manifests.AGENTS_SOURCE_ENV, str(first_source))
    first = registry.resolve_agent("openclaw")
    monkeypatch.setenv(remote_manifests.AGENTS_SOURCE_ENV, str(second_source))
    second = registry.resolve_agent("openclaw")
    assert second is first
    assert remote_manifests.ensure_manifest_catalog().source == str(first_source)


def test_wrong_reserved_identity_excluded_while_unrelated_agent_registers(
    monkeypatch, tmp_path
):
    """Guards PR B for issue #1090 against registering invalid claimants."""
    reserved = tmp_path / "acp" / "openclaw"
    reserved.mkdir(parents=True)
    (reserved / "manifest.toml").write_text(
        _MANIFEST.replace('name = "openclaw"', 'name = "not-openclaw"')
    )
    foo = tmp_path / "acp" / "foo"
    foo.mkdir(parents=True)
    (foo / "manifest.toml").write_text(
        _MANIFEST.replace('name = "openclaw"', 'name = "fixture-foo"').replace(
            'aliases = ["openclaw"]', 'aliases = ["foo-alias"]'
        )
    )
    monkeypatch.setenv(remote_manifests.AGENTS_SOURCE_ENV, str(tmp_path))

    assert registry.resolve_agent("fixture-foo").name == "fixture-foo"
    assert "not-openclaw" not in registry.AGENTS
    with pytest.raises(registry.AgentManifestResolutionError) as exc:
        registry.resolve_agent("openclaw")
    assert exc.value.category == "malformed"
