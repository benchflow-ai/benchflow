import sys
import types
from types import SimpleNamespace
from typing import ClassVar

from typer.testing import CliRunner

from benchflow.cli.main import app


def _install_fake_daytona(monkeypatch, sandboxes):
    class FakeDaytona:
        instances: ClassVar[list] = []

        def __init__(self):
            self.deleted = []
            self.__class__.instances.append(self)

        def list(self, query=None):
            # Mirror daytona SDK >=0.18: list() returns an auto-paginating
            # Iterator[Sandbox], not a page object with `.items`.
            _ = query
            return iter(sandboxes)

        def delete(self, sandbox, timeout=60):
            _ = timeout
            self.deleted.append(sandbox.id)

    fake_daytona = types.ModuleType("daytona")
    fake_daytona.Daytona = FakeDaytona
    monkeypatch.setitem(sys.modules, "daytona", fake_daytona)
    return FakeDaytona


def test_environment_cleanup_dry_run_lists_old_daytona_sandboxes(monkeypatch):
    """Guards PR #605: cleanup must iterate Daytona.list() (Iterator[Sandbox],
    SDK >=0.18) instead of the removed paged ``.items`` page object.

    Also guards ownership scoping: only benchflow-owned sandboxes are listed as
    delete candidates; a foreign sandbox sharing the API key is left alone even
    when it is far past the age cutoff.
    """
    sandboxes = [
        SimpleNamespace(
            id="old-sandbox",
            state="started",
            created_at="2025-01-01T00:00:00Z",
            labels={"benchflow.managed": "1"},
        ),
        SimpleNamespace(
            id="foreign-sandbox",
            state="started",
            created_at="2025-01-01T00:00:00Z",
            labels={"owner": "someone-else"},
        ),
    ]
    fake_daytona = _install_fake_daytona(monkeypatch, sandboxes)

    result = CliRunner().invoke(
        app, ["environment", "cleanup", "--dry-run", "--max-age", "60"]
    )

    assert result.exit_code == 0
    assert "old-sandbox" in result.output
    assert "(delete)" in result.output
    assert "foreign-sandbox" not in result.output
    assert fake_daytona.instances[0].deleted == []


def test_environment_cleanup_deletes_age_eligible_sandboxes(monkeypatch):
    """Guards PR #605: `bench environment cleanup` deletes age-eligible sandboxes
    while iterating Daytona.list()'s Iterator[Sandbox] (SDK >=0.18)."""
    sandboxes = [
        SimpleNamespace(
            id="old-sandbox",
            state="started",
            created_at="2025-01-01T00:00:00Z",
            labels={"benchflow.managed": "1"},
        ),
        SimpleNamespace(
            id="foreign-sandbox",
            state="started",
            created_at="2025-01-01T00:00:00Z",
            labels={"owner": "someone-else"},
        ),
    ]
    fake_daytona = _install_fake_daytona(monkeypatch, sandboxes)

    result = CliRunner().invoke(app, ["environment", "cleanup", "--max-age", "60"])

    assert result.exit_code == 0
    assert "1 sandboxes deleted" in result.output
    assert fake_daytona.instances[0].deleted == ["old-sandbox"]


def test_environment_list_uses_daytona_import_compat(monkeypatch):
    """Guards PR #605 (iterating Daytona.list()'s Iterator[Sandbox], SDK >=0.18)
    and the anyio import-compat shim for `bench environment list`."""
    import anyio

    monkeypatch.delattr(anyio, "AsyncContextManagerMixin", raising=False)
    sandboxes = [
        SimpleNamespace(
            id="active-sandbox",
            state="started",
            created_at="2025-01-01T00:00:00Z",
            target="benchflow",
        )
    ]
    _install_fake_daytona(monkeypatch, sandboxes)

    result = CliRunner().invoke(app, ["environment", "list"])

    assert result.exit_code == 0
    assert hasattr(anyio, "AsyncContextManagerMixin")
    assert "active-sandb" in result.output
    assert "1 sandbox(es)" in result.output


def test_sandbox_list_degrades_cleanly_when_daytona_sdk_absent(monkeypatch):
    """`bench sandbox list` is a read command: a missing optional Daytona SDK is
    an empty result (exit 0), not a hard failure (exit 1) telling the user to
    install a dependency they may not want. Docker sandboxes are ephemeral and
    Guards PR #789 (CLI error-handling hardening).
    not listable, so there is genuinely nothing else to show."""
    # A None entry in sys.modules makes `import daytona` raise ImportError,
    # simulating the SDK being absent regardless of the host's install state.
    monkeypatch.setitem(sys.modules, "daytona", None)

    result = CliRunner().invoke(app, ["sandbox", "list"])

    assert result.exit_code == 0
    assert "No active sandboxes" in result.output
    assert "sandbox-daytona" in result.output


def test_sandbox_cleanup_degrades_cleanly_when_daytona_sdk_absent(monkeypatch):
    """`bench sandbox cleanup` mirrors `list`: no Daytona SDK means nothing to
    Guards PR #789 (CLI error-handling hardening).
    reap, reported as a clean no-op rather than an install-nag exit 1."""
    monkeypatch.setitem(sys.modules, "daytona", None)
    # No Docker host either (bf-snap image cleanup: its own tests).
    from benchflow.cli import sandbox as sandbox_cli

    monkeypatch.setattr(sandbox_cli, "cleanup_docker_snapshots", lambda **_: None)

    result = CliRunner().invoke(app, ["sandbox", "cleanup", "--dry-run"])

    assert result.exit_code == 0
    assert "Nothing to clean up" in result.output
    assert "sandbox-daytona" in result.output


def test_cleanup_summary_does_not_call_other_owners_sandboxes_young(monkeypatch):
    """With --max-age 0 and only another owner's
    sandboxes listed, cleanup printed "0 sandboxes deleted (1 skipped, younger
    than 0m)"; the skipped sandbox was skipped for its owner, not its age."""
    sandboxes = [
        SimpleNamespace(
            id="foreign-sandbox",
            state="started",
            created_at="2025-01-01T00:00:00Z",
            labels={"owner": "someone-else"},
        ),
    ]
    _install_fake_daytona(monkeypatch, sandboxes)

    result = CliRunner().invoke(app, ["environment", "cleanup", "--max-age", "0"])

    assert result.exit_code == 0
    assert "skipped, younger than" not in result.output
    assert "another owner's" in " ".join(result.output.split())


def test_cleanup_snapshot_summary_reads_as_a_sentence(monkeypatch):
    """The snapshot line printed "2 bf-snap-me-* snapshots deleted, 0m"."""
    import benchflow.sandbox.daytona as daytona_mod
    import benchflow.sandbox.daytona_reaper as reaper

    _install_fake_daytona(monkeypatch, [])
    monkeypatch.setattr(reaper, "owner_snapshot_prefix", lambda: "bf-snap-me-")
    monkeypatch.setattr(
        daytona_mod,
        "reap_stale_snapshots",
        lambda client, **_: {"deleted": 2, "failed": 0},
    )

    result = CliRunner().invoke(app, ["environment", "cleanup", "--max-age", "0"])

    output = " ".join(result.output.split())
    assert result.exit_code == 0, result.output
    assert "deleted, 0m" not in output
    assert "2 bf-snap-me-* snapshots deleted (older than 0m; 0 failed)" in output
