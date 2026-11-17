"""A main that joins another compose service's network namespace
(``network_mode: service:<x>``: TB 4.1 ctr-optimization, nextjs-performance)
is in the project's own namespace, owned by <x>: one namespace, one rule set,
main's (EN-609, the lead's 04:5xZ ruling). A chain to the host's network is
refused before start."""

from __future__ import annotations

import pytest

from benchflow.sandbox.cloud_metadata import task_compose_topology
from benchflow.sandbox.service_cover import cover_task_services


def _compose(tmp_path, body: str):
    (tmp_path / "docker-compose.yaml").write_text("services:\n" + body)
    return tmp_path


@pytest.mark.parametrize(
    ("body", "owner", "end", "outside"),
    [
        (
            "  main:\n    network_mode: service:api\n  api:\n    image: x\n",
            "api",
            "",
            False,
        ),
        (
            "  main:\n    network_mode: service:api\n  api:\n    network_mode: service:db\n"
            "  db:\n    image: x\n",
            "db",
            "",
            False,
        ),
        (
            "  main:\n    network_mode: service:api\n  api:\n    network_mode: host\n",
            "api",
            "host",
            True,
        ),
        (
            "  main:\n    network_mode: service:api\n  api:\n    network_mode: service:main\n",
            "api",
            "outside",
            True,
        ),
        ("  main:\n    network_mode: service:gone\n", "gone", "outside", True),
        (
            "  main:\n    network_mode: service:api\n  api:\n    network_mode: container:abc\n",
            "api",
            "outside",
            True,
        ),
        ("  main:\n    network_mode: host\n", "", "", True),
        ("  main:\n    image: x\n", "", "", False),
    ],
    ids=["owner", "chain", "host", "loop", "unknown", "container", "main-host", "own"],
)
def test_where_mains_namespace_is(tmp_path, body, owner, end, outside):
    topology = task_compose_topology(_compose(tmp_path, body))
    assert (topology.main_netns_owner, topology.main_chain_end) == (owner, end)
    assert topology.main_shares_outside_network is outside


class Env:
    """The helper's answers for main's cover (its namespace is api's)."""

    def __init__(self) -> None:
        self.installed: list[str] = []
        self._main_cover_record = {
            "service": "main",
            "state": "covered",
            "proven": True,
            "before_release": True,
        }

    async def metadata_service_targets(self, include_main: bool = False):
        return [
            {
                "service": "main",
                "id": "m" * 64,
                "network_mode": "container:" + "a" * 64,
            },
            {
                "service": "api",
                "id": "a" * 64,
                "network_mode": "bf_default",
                "cap_add": [],
                "privileged": False,
                "subnets": ["172.20.0.0/16"],
                "gateways": ["172.20.0.1"],
            },
        ]

    async def metadata_service_install(self, container, command):
        self.installed.append(container)
        raise AssertionError("no rule on top of main's in its namespace")

    def metadata_service_close(self):
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["public", "allowlist", "no-network"])
async def test_the_namespace_owner_gets_mains_rules_not_its_own(mode):
    env = Env()
    records = await cover_task_services(
        env, None, mode=mode, metadata=True, opt_out=False, main_metadata=True
    )
    main, api = records
    assert main["state"] == "covered"
    assert api["service"] == "api" and api["state"] == "main's namespace"
    assert api["proven"] is True
    assert env.installed == []
