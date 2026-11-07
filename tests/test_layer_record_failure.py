"""Guard FF-install-C8: cd4d492d let cache I/O abort trial startup."""

import pytest

from benchflow.sandbox import _layer_lock, env_layers
from tests.test_env_layers_proxy import RECIPE, ProxyFakeDocker


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["mkdir", "append"])
async def test_trust_record_io_failure_uses_the_layer_fallback(
    tmp_path, monkeypatch, failure
):
    fake = ProxyFakeDocker()
    monkeypatch.setattr(env_layers.DockerLayers, "_docker", fake.run)
    fake.add_image("bf__task")
    # Layer locks (merged from integ/py-0.8) in this test's folder.
    monkeypatch.setenv(_layer_lock.LOCK_DIR_ENV, str(tmp_path / "layer-locks"))
    _layer_lock.clear_process_locks()
    root = tmp_path / "trust"
    layers = env_layers.DockerLayers()
    base = await layers.inspect_base("bf__task")
    key = env_layers.proxy_layer_key(RECIPE, base)
    if failure == "mkdir":
        root.write_text("not a directory")
    else:
        root.mkdir()
        (root / f"proxy-{key}").mkdir()
    monkeypatch.setattr(env_layers, "_trust_root", lambda: root)
    with pytest.raises(env_layers.LayerError, match="trust record"):
        await layers.ensure_proxy_layer(RECIPE, "bf__task", prune=False)
    assert sum(c[0] == "build" for c in fake.calls) == 1
    assert env_layers.trusted_layer_ids("proxy", key) == set()
