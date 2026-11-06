"""Guard SU-CONFIRM-3: Compose names lost to YAML 1.1 coercion at 1ee65ef0."""

from itertools import product

import pytest
import yaml

from benchflow.sandbox.cloud_metadata import (
    _compose_loader,
    cloud_metadata_plan,
    task_compose_topology,
)
from benchflow.task.config import TaskConfig


def _spellings(words):
    return [
        "".join(chars)
        for word in words.split()
        for chars in product(*[(c, c.upper()) for c in word])
    ]


LEGACY_WORDS = _spellings("y yes on n no off")
BOOLEAN_WORDS = LEGACY_WORDS + _spellings("true false")
LAYOUTS = [
    "modern-list",
    "modern-map",
    "modern-alias",
    "modern-default",
    "modern-override",
    "legacy",
    "legacy-override",
    "merged",
    "key-list",
    "key-map",
    "key-collision",
]


def _compose(value, layout):
    key = "host"
    joined = "networks: [host]"
    spec = f"external: {{}}\n    name: {value}"
    prefix = ""
    if layout == "modern-map":
        joined = "networks: {host: null}"
        spec = f"external: true\n    name: {value}"
    elif layout == "modern-alias":
        joined = "networks: {host: {aliases: [dummy]}}"
        prefix = f"x-name: &network_name {value}\n"
        spec = "external: {}\n    name: *network_name"
    elif layout == "modern-default":
        key = "default"
        joined = ""
    elif layout == "modern-override":
        spec = f"external: {{}}\n    name: !override {value}"
    elif layout == "legacy":
        spec = f"external: {{name: {value}}}"
    elif layout == "legacy-override":
        spec = f"external: !override {{name: {value}}}"
    elif layout == "merged":
        prefix = f"x-network: &network\n  external: {{}}\n  name: {value}\n"
        spec = "<<: *network"
    elif layout.startswith("key-"):
        key = value
        # Quoted references must address the same key as a plain declaration.
        reference = value if value.startswith(("'", '"')) else f"'{value}'"
        joined = (
            f"networks: {{{reference}: null}}"
            if layout == "key-map"
            else f"networks: [{reference}]"
        )
        spec = "external: {}\n    name: host"
    services = f"services:\n  main:\n    image: dummy\n    {joined}\n"
    if layout == "modern-default":
        services = ""  # Backend base file supplies main and its default network.
    text = prefix + services + f"networks:\n  {key}:\n    {spec}\n"
    if layout == "key-collision":
        # YAML 1.1 collapsed yes/on and no/off into boolean mapping keys.
        sibling = "off" if value.strip("'\"").lower() != "off" else "no"
        text += f"  {sibling}:\n    external: false\n"
    return text


@pytest.mark.parametrize("word", BOOLEAN_WORDS)
@pytest.mark.parametrize("scalar", ["{}", "'{}'", '"{}"', "!!str {}"])
def test_compose_loader_preserves_string_names(word, scalar):
    """Guards 1ee65ef0: only YAML 1.2 boolean literals have implicit bool type."""
    value = yaml.load(f"name: {scalar.format(word)}", Loader=_compose_loader())["name"]
    if scalar == "{}" and word in {"true", "True", "TRUE", "false", "False", "FALSE"}:
        assert type(value) is bool
        assert value is (word.lower() == "true")
    else:
        assert type(value) is str
        assert value == word


@pytest.mark.parametrize("consumer", ["plan", "docker", "remote-docker", "daytona"])
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("scalar", ["{}", "'{}'", '"{}"'])
@pytest.mark.parametrize("word", LEGACY_WORDS)
def test_network_names_keep_protection_in_every_consumer(
    tmp_path, monkeypatch, consumer, layout, scalar, word
):
    """Guards 1ee65ef0: modern/legacy names and joined keys keep Compose meaning."""
    compose = _compose(scalar.format(word), layout)
    outside = layout.startswith("key-")
    if consumer == "plan":
        (tmp_path / "docker-compose.yaml").write_text(compose)
        assert task_compose_topology(tmp_path).main_shares_outside_network is outside
        plan = cloud_metadata_plan(
            TaskConfig(),
            sandbox_user="agent",
            launches_agent=True,
            environment_dir=tmp_path,
        )
        assert plan is not None and plan.armed is not outside
        if outside:
            assert plan.state == "not_installed"
        return

    from benchflow.sandbox._compose import COMPOSE_NET_ADMIN_PATH
    from benchflow.sandbox.daytona_dind import _DaytonaDinD
    from tests.test_cloud_metadata_block import OPEN, _sandbox_for

    monkeypatch.setenv("BENCHFLOW_REMOTE_DOCKER_HOST", "ssh://dummy.invalid")
    sandbox = _sandbox_for(
        tmp_path, OPEN, block=True, backend=consumer, compose=compose
    )
    assert sandbox.task_env_config.cloud_metadata_block is not outside
    if consumer == "daytona":
        assert isinstance(sandbox._strategy, _DaytonaDinD)
        path = f"{_DaytonaDinD._COMPOSE_DIR}/{COMPOSE_NET_ADMIN_PATH.name}"
        assert (path in sandbox._strategy._compose_file_flags()) is not outside
    else:
        assert (COMPOSE_NET_ADMIN_PATH in sandbox._docker_compose_paths) is not outside


def test_compose_boolean_resolver_is_local_and_safe():
    """Guards 1ee65ef0 without changing YAML semantics in other task loaders."""
    loader = _compose_loader()
    assert yaml.safe_load("off") is False
    assert yaml.load("off", Loader=loader) == "off"
    assert yaml.safe_load("off") is False
    assert yaml.load("!!bool false", Loader=loader) is False
    assert yaml.load("value: !reset on", Loader=loader) == {"value": None}
    with pytest.raises(yaml.constructor.ConstructorError):
        yaml.load("!!python/object:dummy {}", Loader=loader)
