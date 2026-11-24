"""(ruling R7, option c): Docker Engine 29 and Compose 2.40 are the minimum.

Below it ``bench doctor`` warns with both versions and the minimum, a trial
under a network policy is refused before anything starts, an open-network
trial runs, and the build-only-service and ``models:`` errors name the
minimum. Versions are faked here; one run on a real CE 28.0.4 VM confirms.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from benchflow.doctor import CommandResult, check_docker
from benchflow.sandbox import _compose_runtime, service_cover
from tests.test_doctor import make_probes

OLD_INFO = CommandResult(
    0,
    json.dumps(
        {
            "ServerVersion": "28.0.4",
            "NCPU": 4,
            "MemTotal": 8 * 1024**3,
            "OperatingSystem": "Ubuntu 24.04",
        }
    ),
    "",
)


@pytest.mark.parametrize(
    ("engine", "compose", "below"),
    [
        ("28.0.4", "2.38.2", True),
        ("29.1.3", "2.38.2", True),
        ("28.5.1", "2.40.3", True),
        ("29.1.3", "2.40.3+ds1-0ubuntu1~24.04.1", False),
        ("29.9.3", "5.7.1", False),
        ("29.0.0", "v2.40.0", False),
    ],
)
def test_the_minimum(engine, compose, below):
    reason = _compose_runtime.docker_below_minimum(engine, compose)
    assert (reason is not None) is below
    if below:
        assert engine in reason and compose in reason
        assert "29" in reason and "2.40" in reason


def test_doctor_warns_below_the_minimum_and_says_nothing_above(tmp_path):
    (tmp_path / "old").mkdir()
    (tmp_path / "new").mkdir()
    old = make_probes(
        tmp_path / "old",
        system="Linux",
        commands={
            ("docker", "info", "--format", "{{json .}}"): OLD_INFO,
            ("docker", "compose", "version", "--short"): CommandResult(
                0, "2.38.2\n", ""
            ),
        },
    )
    rows = [c for c in check_docker(old, required=True) if c.id == "docker-version"]
    assert [c.status for c in rows] == ["warn"]
    assert "28.0.4" in rows[0].summary and "2.38.2" in rows[0].summary
    assert "29" in rows[0].summary and "2.40" in rows[0].summary
    new = make_probes(tmp_path / "new", system="Linux")
    assert not [c for c in check_docker(new, required=True) if c.id == "docker-version"]


class _FakeDocker:
    """Answers the version probe as a Docker CLI would, and records a start."""

    def __init__(self, engine: str, compose: str) -> None:
        self.engine, self.compose = engine, compose
        self.started = False
        self.asked = 0

    def _docker_compose_global_args(self) -> list[str]:
        return ["--host", f"unix:///fake/{id(self)}.sock"]

    async def _run_docker_process(self, argv, **_kwargs):
        self.asked += 1
        out = self.compose if "compose" in argv else self.engine
        return SimpleNamespace(return_code=0, stdout=out + "\n", stderr="")

    async def start(self, force_build=False):
        self.started = True


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["allowlist", "denylist", "no-network"])
async def test_a_policy_trial_is_refused_before_anything_starts(monkeypatch, policy):
    monkeypatch.setattr(service_cover, "restricted_policy", lambda config: policy)
    monkeypatch.setattr(service_cover, "ask_start_holds", lambda *a, **k: None)
    sandbox = _FakeDocker("28.0.4", "2.38.2")
    with pytest.raises(service_cover.ServiceCoverError) as refused:
        await service_cover.start_sandbox(
            sandbox, SimpleNamespace(config=None, paths=None), agent="oracle"
        )
    text = str(refused.value)
    assert text.startswith("sandbox startup refused")
    assert "28.0.4" in text and "2.38.2" in text and policy in text
    assert not sandbox.started


@pytest.mark.asyncio
async def test_an_open_network_trial_runs_below_the_minimum(monkeypatch):
    monkeypatch.setattr(service_cover, "restricted_policy", lambda config: "")
    monkeypatch.setattr(service_cover, "ask_start_holds", lambda *a, **k: None)
    sandbox = _FakeDocker("28.0.4", "2.38.2")
    await service_cover.start_sandbox(
        sandbox, SimpleNamespace(config=None, paths=None), agent="oracle"
    )
    assert sandbox.started


@pytest.mark.asyncio
async def test_a_policy_trial_runs_at_the_minimum_and_the_probe_is_cached(monkeypatch):
    monkeypatch.setattr(service_cover, "restricted_policy", lambda config: "allowlist")
    monkeypatch.setattr(service_cover, "ask_start_holds", lambda *a, **k: None)
    sandbox = _FakeDocker("29.1.3", "2.40.3")
    for _ in range(3):
        await service_cover.start_sandbox(
            sandbox, SimpleNamespace(config=None, paths=None), agent="oracle"
        )
    assert sandbox.started and sandbox.asked == 2


@pytest.mark.asyncio
async def test_the_build_only_service_error_names_the_minimum():
    async def compose(argv, **_kwargs):
        return SimpleNamespace(
            return_code=0, stdout=json.dumps({"target": {}}), stderr=""
        )

    config = {"services": {"main": {"image": "x"}, "side": {"build": {"context": "."}}}}
    with pytest.raises(ValueError) as caught:
        await _compose_runtime.implicit_build_aliases(
            config, prebuilt=False, compose=compose
        )
    assert "omitted implicit build outputs" in str(caught.value)
    assert "2.40" in str(caught.value) and "29" in str(caught.value)


def test_the_models_error_names_the_minimum():
    from benchflow.task.compose_env import _config_refusal

    refusal = _config_refusal(
        "validating docker-compose-no-net-raw.yaml: services.main.models.m must be a mapping"
    )
    assert "2.40" in (refusal.shown or "")
