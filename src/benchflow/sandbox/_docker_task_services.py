"""The task's compose services as ``network_mode = "allowlist"`` sees them, read from ``docker inspect``.

Pure parsing, so the rules are tested without a daemon: ``DockerSandbox``
feeds it the project's containers and their networks, and
:func:`benchflow.sandbox.egress_denylist.admit_task_services` decides what the
agent may reach and whether the policy can hold at all.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from benchflow.sandbox.egress_denylist import TaskService

_SERVICE_LABEL = "com.docker.compose.service"


class TaskServiceInspectError(RuntimeError):
    """``docker inspect`` did not describe the project well enough to check it."""


def task_services_from_inspect(
    containers: Sequence[Mapping[str, Any]], networks: Mapping[str, Mapping[str, Any]]
) -> list[TaskService]:
    """Every container but ``main``, with its names and addresses on the networks it shares with ``main``.

    ``containers`` are ``docker inspect`` records of the project's running
    containers; ``networks`` maps each network id they use to its ``docker
    network inspect`` record. A container has a route out when it uses the
    host's network, sits on a network that is not ``internal`` (Docker NATs
    every other bridge to the outside), or shares the network namespace of
    ``main``, of a container outside the project, or of a project container
    that has a route out. ``network_mode: none`` attaches a container to the
    ``null``-driver network, which has no route at all.
    """
    main = [c for c in containers if _labels(c).get(_SERVICE_LABEL) == "main"]
    if len(main) != 1:
        raise TaskServiceInspectError(
            f"expected one running main container, found {len(main)}"
        )
    shared = {
        str(net.get("NetworkID") or "") for net in _networks(main[0]).values()
    } - {""}
    services: dict[str, TaskService] = {}
    borrowing: list[tuple[str, Mapping[str, Any], str]] = []
    for container in containers:
        if container is main[0]:
            continue
        service = _labels(container).get(_SERVICE_LABEL) or _name(container)
        mode = str((container.get("HostConfig") or {}).get("NetworkMode") or "")
        if mode.startswith(("container:", "service:")):
            # Its routes are its namespace owner's: judged once all are read.
            borrowing.append((service, container, mode.split(":", 1)[1]))
            continue
        if mode == "host":
            services[str(container.get("Id"))] = TaskService(
                service, outbound=("it uses the host's network",)
            )
            continue
        outbound: list[str] = []
        names: dict[str, None] = {}
        addresses: dict[str, None] = {}
        for name, net in _networks(container).items():
            network_id = str(net.get("NetworkID") or "")
            record = networks.get(network_id)
            if record is None:
                raise TaskServiceInspectError(
                    f"network {name!r} of service {service!r} was not inspected"
                )
            if record.get("Driver") == "null":
                continue
            if not record.get("Internal"):
                outbound.append(f"network {name!r} is not internal")
            if network_id not in shared:
                continue
            names.update(
                dict.fromkeys(
                    n.lower()
                    for n in (
                        service,
                        _name(container),
                        *(net.get("Aliases") or ()),
                        *(net.get("DNSNames") or ()),
                    )
                    if n
                )
            )
            addresses.update(
                dict.fromkeys(
                    a for a in (net.get("IPAddress"), net.get("GlobalIPv6Address")) if a
                )
            )
        services[str(container.get("Id"))] = TaskService(
            service, tuple(names), tuple(addresses), tuple(outbound)
        )
    main_id = str(main[0].get("Id"))
    for service, container, owner in borrowing:
        if _same_container(owner, main_id):
            why = ("it shares main's network namespace",)
        else:
            target = next(
                (s for cid, s in services.items() if _same_container(owner, cid)), None
            )
            why = (
                ("it shares the namespace of a container outside the task",)
                if target is None
                else tuple(f"through {target.service!r}: {w}" for w in target.outbound)
            )
        services[str(container.get("Id"))] = TaskService(service, outbound=why)
    return list(services.values())


def _same_container(reference: str, container_id: str) -> bool:
    """Docker names a namespace owner by its full id or a prefix of it."""
    return bool(reference) and container_id.startswith(reference)


def _labels(container: Mapping[str, Any]) -> Mapping[str, str]:
    return (container.get("Config") or {}).get("Labels") or {}


def _name(container: Mapping[str, Any]) -> str:
    return str(container.get("Name") or "").lstrip("/")


def _networks(container: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    return (container.get("NetworkSettings") or {}).get("Networks") or {}
