"""Guards orphan-pin parent preservation at PY8 base 363d815b."""

import pytest

from benchflow.sandbox.env_layers import ALIAS_PREFIX, DockerLayers, PruneReport
from tests.test_cli_cache import DEAD_OWNER, KEY_A, _bench
from tests.test_cli_cache import docker as docker


@pytest.mark.parametrize("kind", ["agent", "proxy"])
@pytest.mark.parametrize("store", ["classic", "containerd"])
def test_orphan_pin_removal_cannot_prune_parents(docker, kind, store):
    """Guards 363d815b: each selected pin removes only its own reference."""
    docker.store = store
    pin = f"benchflow-layer-base/{kind}-{KEY_A}:{DEAD_OWNER}"
    docker.add_image(pin)
    result = _bench("clean", "-s", "docker", "--only", "images", "--remove")
    assert result.exit_code == 0, result.output
    removals = [c for c in docker.calls if c[:2] == ["image", "rm"]]
    assert removals == [["image", "rm", "--no-prune", pin]]


@pytest.mark.parametrize("kind", ["agent", "proxy"])
async def test_last_stale_alias_cannot_prune_parents(docker, kind):
    """Guards 363d815b's sibling path when a stale alias is the last name."""
    ref = f"{ALIAS_PREFIX}{kind}-{KEY_A}:{DEAD_OWNER}"
    docker.add_image(ref)
    assert await DockerLayers()._remove_stale_alias(ref, PruneReport())
    removals = [c for c in docker.calls if c[:2] == ["image", "rm"]]
    assert removals == [["image", "rm", "--no-prune", ref]]
