"""Guard SU-CONFIRM-2's remaining Compose coercion gap at fc67ac0a.

Compose's external flag is a boolean, including its accepted string forms,
or a legacy mapping. Its truth value must never depend on Python truthiness.
"""

from itertools import product

import pytest

from benchflow.sandbox.cloud_metadata import (
    cloud_metadata_plan,
    task_compose_topology,
)
from benchflow.task.config import TaskConfig


def _spellings(word):
    return ["".join(chars) for chars in product(*[(c, c.upper()) for c in word])]


BOOLEAN_SPELLINGS = [
    (spelling, external)
    for words, external in [("false n no off", False), ("true y yes on", True)]
    for word in words.split()
    for spelling in _spellings(word)
]


def _compose(value, joined="networks: [host]", key="host", extra=""):
    return (
        f"services:\n  main:\n    image: dummy\n    {joined}\n"
        f"networks:\n  {key}:\n    external: {value}\n{extra}"
    )


def _check(tmp_path, compose, outside):
    (tmp_path / "docker-compose.yaml").write_text(compose)
    topology = task_compose_topology(tmp_path)
    assert topology.main_shares_outside_network is outside
    plan = cloud_metadata_plan(
        TaskConfig(),
        sandbox_user="agent",
        launches_agent=True,
        environment_dir=tmp_path,
    )
    assert plan is not None and plan.armed is not outside
    if outside:
        assert plan.state == "not_installed"


@pytest.mark.parametrize("value,external", BOOLEAN_SPELLINGS)
@pytest.mark.parametrize("scalar", ["{}", "'{}'", '"{}"', "!override '{}'"])
@pytest.mark.parametrize(
    "joined",
    [
        "networks: [host]",
        "networks: {host: null}",
        "networks: {host: {aliases: [dummy]}}",
    ],
)
def test_external_boolean_spellings_keep_compose_meaning(
    tmp_path, value, external, scalar, joined
):
    """Guards fc67ac0a: all Compose bool spellings, tags and membership forms."""
    _check(tmp_path, _compose(scalar.format(value), joined), external)


@pytest.mark.parametrize(
    "value,key,extra,outside",
    [
        ("{}", "host", "", True),
        ("{name: host}", "default", "", True),
        ("{name: dummy-bridge}", "host", "", False),
        ("{}", "host", "    name: dummy-bridge\n", False),
        ("'false'", "default", "", False),
        ("'false'", "default", "    name: host\n", True),
        ("'false'", "host", "    name: dummy-bridge\n", False),
        ("'true'", "host", "    name: dummy-bridge\n", False),
        ("'false'", "default", "    name: ${DUMMY_NETWORK}\n", True),
        ("!reset true", "host", "", False),
        ("null", "host", "", False),
    ],
)
@pytest.mark.parametrize("joined", ["", "networks: []", "networks: {}", "explicit"])
def test_external_defaults_legacy_mappings_and_names(
    tmp_path, value, key, extra, outside, joined
):
    """Guards fc67ac0a, including empty legacy mappings and explicit names."""
    if joined == "explicit" or key != "default":
        joined = f"networks: [{key}]"
    _check(tmp_path, _compose(value, joined, key, extra), outside)


@pytest.mark.parametrize(
    "value", ["''", "'0'", "'1'", "0", "1", "[]", "' false '", "'${DUMMY_EXTERNAL}'"]
)
def test_unresolved_external_is_never_assumed_private(tmp_path, value):
    """Guards fc67ac0a: invalid/dynamic flags cannot authorize a firewall."""
    _check(tmp_path, _compose(value, key="default", joined=""), True)


@pytest.mark.parametrize("backend", ["docker", "remote-docker", "daytona"])
@pytest.mark.parametrize(
    "value,external",
    [(f"'{value}'", external) for value, external in BOOLEAN_SPELLINGS]
    + [("{}", True), ("{name: host}", True), ("false", False), ("true", True)],
)
def test_every_compose_backend_uses_the_normalized_external_flag(
    tmp_path, monkeypatch, backend, value, external
):
    """Guards fc67ac0a at each factory, independently of rollout planning."""
    from benchflow.sandbox._compose import COMPOSE_NET_ADMIN_PATH
    from benchflow.sandbox.daytona_dind import _DaytonaDinD
    from tests.test_cloud_metadata_block import OPEN, _sandbox_for

    monkeypatch.setenv("BENCHFLOW_REMOTE_DOCKER_HOST", "ssh://dummy.invalid")
    sandbox = _sandbox_for(
        tmp_path, OPEN, block=True, backend=backend, compose=_compose(value)
    )
    assert sandbox.task_env_config.cloud_metadata_block is not external
    if backend == "daytona":
        assert isinstance(sandbox._strategy, _DaytonaDinD)
        path = f"{_DaytonaDinD._COMPOSE_DIR}/{COMPOSE_NET_ADMIN_PATH.name}"
        assert (path in sandbox._strategy._compose_file_flags()) is not external
    else:
        assert (COMPOSE_NET_ADMIN_PATH in sandbox._docker_compose_paths) is not external
