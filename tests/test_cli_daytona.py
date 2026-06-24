import sys
import types
from types import SimpleNamespace
from typing import ClassVar

from typer.testing import CliRunner

from benchflow.cli.main import app


def _install_fake_daytona(monkeypatch, sandboxes, snapshots=()):
    class _FakeSnapshotService:
        """Mirror the SDK's ``client.snapshot`` service (paginated list/delete)."""

        def __init__(self):
            self.deleted = []

        def list(self, page=None, limit=None):
            _ = (page, limit)
            items = list(snapshots)
            return SimpleNamespace(items=items, total=len(items), page=1, total_pages=1)

        def delete(self, snapshot):
            self.deleted.append(snapshot.name)

    class FakeDaytona:
        instances: ClassVar[list] = []

        def __init__(self):
            self.deleted = []
            self.snapshot = _FakeSnapshotService()
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


def test_environment_cleanup_reaps_leaked_snapshots(monkeypatch):
    """`environment cleanup` also reaps leaked ``bf-snap-*`` snapshots, scoped by
    name prefix (Daytona snapshots have no labels). Foreign snapshots are never
    touched even though they share the API key."""
    snapshots = [
        SimpleNamespace(name="bf-snap-demo-abc123", state="active"),
        SimpleNamespace(name="someone-elses-snapshot", state="active"),
    ]
    fake_daytona = _install_fake_daytona(monkeypatch, sandboxes=[], snapshots=snapshots)

    result = CliRunner().invoke(app, ["environment", "cleanup", "--max-age", "60"])

    assert result.exit_code == 0
    assert "1 snapshots deleted" in result.output
    # Sandbox and snapshot tiers each build their own sync client; aggregate the
    # snapshot deletions across every client instance the run created.
    deleted = [n for inst in fake_daytona.instances for n in inst.snapshot.deleted]
    assert deleted == ["bf-snap-demo-abc123"]


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

    result = CliRunner().invoke(app, ["sandbox", "cleanup", "--dry-run"])

    assert result.exit_code == 0
    assert "Nothing to clean up" in result.output
    assert "sandbox-daytona" in result.output
