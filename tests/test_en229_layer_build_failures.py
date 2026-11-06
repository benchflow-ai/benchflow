"""Failed cached installs must start from the plain image."""

from __future__ import annotations

import pytest

from benchflow.sandbox import daytona_layers as layers


@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize("only_if_reused", [False, True])
async def test_failed_recipe_is_not_rebuilt_by_later_trials(
    tmp_path, monkeypatch, reuse, only_if_reused
):
    """Review F1/F2: failed full and guard-only recipes stay plain."""
    from tests.test_daytona_install_layers import FakeDaytona, _start
    from tests.test_dfx_scale_r3 import _fake_client

    monkeypatch.setattr(layers, "_BUILT", set())
    monkeypatch.setattr(layers, "_BUILDING", set())
    monkeypatch.setenv(layers.LAYER_REUSE_ENV, "1" if reuse else "0")
    monkeypatch.delenv("BENCHFLOW_DAYTONA_SNAPSHOTS", raising=False)
    client = _fake_client(monkeypatch, FakeDaytona, fail_layered_builds=True)
    options = dict(guard_daemon=True, only_if_reused=only_if_reused)
    first = await _start(client, tmp_path / "first", **options)
    second = await _start(client, tmp_path / "second", **options)
    recipes = [p.image.dockerfile() for p in client.requests]
    assert len(recipes) == 3 and "RUN " in recipes[0]
    assert recipes[1] == recipes[2] and "RUN " not in recipes[2]
    assert first._install_layer_keys == second._install_layer_keys == {}
    assert second._install_layers_record["mode"] == "fresh"
    assert "earlier in this process" in second._install_layers_record["fallback"]
    assert second._layer_build_allowance_sec == 0
    assert not layers._BUILDING

    # A different recipe still gets its own attempt.
    await _start(client, tmp_path / "other", image="example/other:en229", **options)
    assert len(client.requests) == 5
    assert "RUN " in client.requests[3].image.dockerfile()

    # A new process has no failure history; the same recipe may succeed there.
    monkeypatch.setattr(layers, "_FAILED", set())
    client.fail_layered_builds = False
    retry = await _start(client, tmp_path / "new-process", **options)
    assert (
        retry._install_layer_keys and "RUN " in client.requests[-1].image.dockerfile()
    )


def test_failure_memory_requires_a_confirmed_build_failure(monkeypatch):
    """Review: releasing a claim after a transport error may retry."""
    monkeypatch.setattr(layers, "_BUILT", set())
    monkeypatch.setattr(layers, "_BUILDING", set())
    monkeypatch.setenv(layers.LAYER_REUSE_ENV, "1")
    keys = {"guard-daemon": "en229-review"}
    assert layers.claim_layer_build(keys) == "build"
    layers.layer_build_finished(keys, built=False)
    assert layers.claim_layer_build(keys) == "build"
    layers.layer_build_finished(keys, built=False, failed=True)
    assert layers.layer_build_failed(keys)
    assert layers.claim_layer_build(keys) is None
    # A concurrent success cannot erase an observed unsafe/failed build.
    layers.layer_build_finished(keys, built=True)
    assert layers.claim_layer_build(keys) is None
