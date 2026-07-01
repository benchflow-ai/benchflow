"""Contract tests for the ``benchflow.agents`` entry-point autoloader.

The loader (``registry._load_agent_plugin_packages``) runs at import time on
every ``import benchflow``, so its failure modes are silent by nature — these
lock the three behaviors that matter:

* a callable-form entry point (``module:register``) is INVOKED, not just
  loaded (the subtle branch: 7 of the 10 benchflow-ai/agents plugins are
  callable-form, and a plain ``ep.load()`` would silently drop them all);
* a module-form entry point registers via import side effect (exercised
  through a real ``importlib.metadata.EntryPoint``);
* a raising plugin is isolated — warning logged, recorded in
  ``FAILED_AGENT_PLUGINS``, siblings still load, import never breaks — and the
  eventual "Unknown agent" error carries the breadcrumb.

The loader resolves ``entry_points`` from ``importlib.metadata`` at call time,
so an ``importlib.metadata.entry_points`` monkeypatch reaches the real code
path without any subprocess machinery.
"""

from __future__ import annotations

import logging
import sys
from types import SimpleNamespace

import pytest

from benchflow.agents import registry


@pytest.fixture(autouse=True)
def _clean_failed_plugins(monkeypatch):
    """Isolate FAILED_AGENT_PLUGINS per test (module-global breadcrumb dict)."""
    monkeypatch.setattr(registry, "FAILED_AGENT_PLUGINS", {})


def test_callable_entry_point_is_invoked(monkeypatch):
    called = []
    fake_ep = SimpleNamespace(name="p", load=lambda: lambda: called.append(1))
    monkeypatch.setattr("importlib.metadata.entry_points", lambda group: [fake_ep])
    registry._load_agent_plugin_packages()
    assert called == [1]


def test_module_entry_point_registers_by_import(monkeypatch, tmp_path):
    """Module-form value goes through a REAL EntryPoint.load() (an import), and
    the module's import-side-effect registration lands in AGENTS. Also locks
    that a non-callable load result is left alone (a module is not invoked)."""
    from importlib.metadata import EntryPoint

    (tmp_path / "bf_fake_plugin.py").write_text(
        "from benchflow.agents.registry import register_agent\n"
        "register_agent('probe-plugin', 'true', 'true')\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    ep = EntryPoint(name="fp", value="bf_fake_plugin", group="benchflow.agents")
    monkeypatch.setattr("importlib.metadata.entry_points", lambda group: [ep])
    try:
        registry._load_agent_plugin_packages()
        assert "probe-plugin" in registry.AGENTS
        assert not registry.FAILED_AGENT_PLUGINS
    finally:
        registry.AGENTS.pop("probe-plugin", None)
        registry.AGENT_INSTALLERS.pop("probe-plugin", None)
        registry.AGENT_LAUNCH.pop("probe-plugin", None)
        sys.modules.pop("bf_fake_plugin", None)


def test_broken_plugin_warns_records_and_does_not_block_others(monkeypatch, caplog):
    def boom():
        raise RuntimeError("kaput")

    hits: dict[str, bool] = {}
    eps = [
        SimpleNamespace(name="bad", load=boom),
        SimpleNamespace(
            name="good", load=lambda: lambda: hits.setdefault("good", True)
        ),
    ]
    monkeypatch.setattr("importlib.metadata.entry_points", lambda group: eps)
    with caplog.at_level(logging.WARNING, logger="benchflow.agents.registry"):
        registry._load_agent_plugin_packages()

    # broken plugin: warned + recorded; sibling still loaded; no exception.
    assert "bad" in caplog.text
    assert registry.FAILED_AGENT_PLUGINS.get("bad") == "RuntimeError: kaput"
    assert hits.get("good")


def test_unknown_agent_error_carries_failed_plugin_breadcrumb(monkeypatch):
    monkeypatch.setattr(
        registry, "FAILED_AGENT_PLUGINS", {"omnigent": "RuntimeError: kaput"}
    )
    with pytest.raises(KeyError) as exc:
        registry.resolve_agent("agent-that-definitely-does-not-exist")
    assert "failed to load" in str(exc.value)
    assert "omnigent" in str(exc.value)
