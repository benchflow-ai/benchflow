"""DaytonaClientManager works for helper scripts, not only inside a rollout.

Regression test: ``get_client()`` before any ``DaytonaSandbox`` existed
raised ``TypeError: 'NoneType' object is not callable`` because the SDK
loads lazily, and a second ``asyncio.run()`` got the singleton client bound
to the first, closed loop (``DaytonaError: ... Event loop is closed``), which
lost a run's record and skipped its snapshot cleanup.
"""

from __future__ import annotations

import asyncio

from benchflow.sandbox import daytona as daytona_mod


class _FakeAsyncDaytona:
    async def close(self) -> None:
        return None


def _fake_sdk(monkeypatch) -> list[int]:
    loads: list[int] = []
    monkeypatch.setattr(daytona_mod, "AsyncDaytona", None)

    def load() -> None:
        loads.append(1)
        daytona_mod.AsyncDaytona = _FakeAsyncDaytona

    monkeypatch.setattr(daytona_mod, "_load_daytona_sdk", load)
    monkeypatch.setattr(daytona_mod.atexit, "register", lambda *_a, **_k: None)
    return loads


def test_get_client_loads_the_sdk_first(monkeypatch):
    loads = _fake_sdk(monkeypatch)
    manager = daytona_mod.DaytonaClientManager()
    client = asyncio.run(manager.get_client())
    assert isinstance(client, _FakeAsyncDaytona)
    assert loads


def test_a_new_event_loop_gets_a_new_client(monkeypatch):
    _fake_sdk(monkeypatch)
    manager = daytona_mod.DaytonaClientManager()

    async def twice():
        return await manager.get_client(), await manager.get_client()

    first_a, first_b = asyncio.run(twice())
    assert first_a is first_b  # one client per loop
    second = asyncio.run(manager.get_client())
    assert second is not first_a
