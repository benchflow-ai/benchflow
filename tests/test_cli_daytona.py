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
    SDK >=0.18) instead of the removed paged ``.items`` page object."""
    sandboxes = [
        SimpleNamespace(
            id="old-sandbox",
            state="started",
            created_at="2025-01-01T00:00:00Z",
        )
    ]
    fake_daytona = _install_fake_daytona(monkeypatch, sandboxes)

    result = CliRunner().invoke(
        app, ["environment", "cleanup", "--dry-run", "--max-age", "60"]
    )

    assert result.exit_code == 0
    assert "old-sandbox" in result.output
    assert "(delete)" in result.output
    assert fake_daytona.instances[0].deleted == []


def test_legacy_cleanup_delegates_to_daytona_cleanup(monkeypatch):
    """Guards PR #605: `bench cleanup` deletes age-eligible sandboxes while
    iterating Daytona.list()'s Iterator[Sandbox] (SDK >=0.18)."""
    sandboxes = [
        SimpleNamespace(
            id="old-sandbox",
            state="started",
            created_at="2025-01-01T00:00:00Z",
        )
    ]
    fake_daytona = _install_fake_daytona(monkeypatch, sandboxes)

    result = CliRunner().invoke(app, ["cleanup", "--max-age", "60"])

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
