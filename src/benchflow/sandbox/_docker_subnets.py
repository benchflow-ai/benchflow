"""Small subnets for Docker trial networks, so many trials fit one daemon.

Every Docker trial is its own Compose project with its own networks. Left to
itself, Compose asks the daemon for each network's subnet, and the daemon
hands out a whole pool chunk per network: a /16 from 172.17-31 and a /20 from
192.168 on a stock daemon, about 30 networks in all. Above about 30 live
trials ``compose up`` then fails with "all predefined address pools have
been fully subnetted" and the trial ends unscored.

BenchFlow instead picks each trial network's subnet itself: a /28 (13
addresses after the network, broadcast and gateway; larger only when the
task attaches more services to that network) carved out of the daemon's own
address pools, written into a compose override as ``ipam.config.subnet``.
That is thousands of trial networks on a stock daemon, and no new private
space: the default range is the daemon's ``default-address-pools`` when it
sets them and Docker's built-in pools otherwise, never a 10.x range that
could shadow a cloud network's routes.

How a subnet is chosen:

- Docker's own auto-allocation (Docker 25 and later) skips any chunk that
  overlaps an on-link route of the daemon's host (a route with no gateway)
  or a nameserver in its resolv.conf; an explicit subnet gets no such
  check. BenchFlow does that check itself, and in the daemon's pools blocks
  each whole chunk such a route or nameserver touches (a /16 of 172.17-31,
  a /20 of 192.168 on a stock daemon), so it never takes an address the
  daemon would leave alone. Routes through a gateway (a VPN's 0.0.0.0/1 and
  128.0.0.0/1, a split tunnel's 172.16.0.0/12) are not counted, as Docker
  does not count them. When the host's routes leave no subnet at all, the
  daemon chooses, as before. BenchFlow chooses subnets only when it can
  read those routes: the daemon runs on this host (Linux, ``docker info``
  names this host, and the client's effective endpoint is a Unix socket).
  For any other daemon (Docker Desktop's or Colima's VM, a remote host) the
  daemon keeps choosing, as before, unless ``BENCHFLOW_DOCKER_SUBNETS``
  names the range to use.
- The daemon arbitrates overlaps: it refuses a network whose subnet overlaps
  one it already has ("Pool overlaps with other one on this address
  space"), whoever made it. BenchFlow skips the subnets of every network the
  daemon listed when this process first asked, and of every network this
  process holds; a create the daemon still refuses (another process got
  there first) re-reads the daemon's networks and tries the next free
  subnet. Each process starts at a random point of the range, so two
  processes seldom meet. No lock is needed.
- A task's network that sets its own ``ipam``, is ``external``, has a fixed
  ``name``, uses a driver other than ``bridge`` or enables IPv6 is left to
  the task and the daemon, as is every network of a compose file BenchFlow
  cannot read. The IPv4 subnets such a network pins are kept clear of every
  subnet BenchFlow picks in this process, for any trial: a job registers
  its tasks' pinned subnets before the first trial starts, and every trial
  adds its own. A pinned network the daemon still refuses (a network of
  another process, or a second trial of the same task, holds part of it)
  ends the trial at once, naming the subnet.

``BENCHFLOW_DOCKER_SUBNETS``: unset or empty, the default above; ``daemon``
(or ``off``), never choose (the daemon does, as before 0.8); otherwise a
comma- or space-separated list of IPv4 ranges (``172.20.0.0/14,
192.168.0.0/16``), used for any daemon. When the range has no free subnet
left the trial fails with what to do: widen the range or lower
``--concurrency``.
"""

from __future__ import annotations

import asyncio
import bisect
import hashlib
import ipaddress
import json
import logging
import os
import random
import re
import socket
import subprocess
import sys
import tempfile
import threading
import uuid
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger("benchflow")

ENV_VAR = "BENCHFLOW_DOCKER_SUBNETS"
_OFF_VALUES = frozenset({"daemon", "off", "none"})

#: The size of a trial network unless the task attaches more services to it.
NETWORK_PREFIX = 28
#: The largest network BenchFlow carves (4,093 containers); a task that needs
#: more is left to the daemon.
_LARGEST_PREFIX = 20
#: Addresses a bridge network cannot give a container: network, broadcast and
#: the gateway.
_RESERVED_PER_NETWORK = 3
#: Spare addresses per network: branch restore makes the replacement ``main``
#: before the old one is gone, and Compose may recreate a container.
_SPARE_PER_NETWORK = 2

#: Docker's built-in local address pools (moby ``ipamutils``), used when the
#: daemon sets no ``default-address-pools``: 172.17-31 carved into /16s and
#: 192.168.0.0/16 into /20s.
DOCKER_DEFAULT_POOLS: tuple[str, ...] = (
    "172.17.0.0/16",
    "172.18.0.0/16",
    "172.19.0.0/16",
    "172.20.0.0/14",
    "172.24.0.0/14",
    "172.28.0.0/14",
    "192.168.0.0/16",
)
_BUILTIN_192 = ipaddress.IPv4Network("192.168.0.0/16")  # carved into /20s

# The daemon refusing an explicit subnet that overlaps one of its networks.
# Docker 29 says "invalid pool request: Pool overlaps with other one on this
# address space"; older bridge drivers "networks have overlapping IPv4" or
# "conflicts with network".
_POOL_OVERLAP = re.compile(
    r"pool overlaps with other one on this address space"
    r"|networks have overlapping ipv4"
    r"|conflicts with network",
    re.IGNORECASE,
)
# The daemon out of address pools for a network it chooses itself (Docker 29,
# and before it).
_POOLS_EXHAUSTED = re.compile(
    r"all predefined address pools have been fully subnetted"
    r"|could not find an available, non-overlapping ipv4 address pool",
    re.IGNORECASE,
)


def is_pool_overlap_error(message: str) -> bool:
    """Whether a ``compose up`` failed because a subnet BenchFlow chose overlaps."""
    return bool(_POOL_OVERLAP.search(message))


# Compose naming the network the daemon refused ("failed to create network
# <project>_<key>: Error response from daemon: ...").
_REFUSED_NETWORK = re.compile(r"failed to create network ([^\s:]+)", re.IGNORECASE)


def refused_network(message: str) -> str | None:
    """The network a failed ``compose up`` could not create, if it says."""
    match = _REFUSED_NETWORK.search(message)
    return match.group(1) if match else None


def pinned_overlap_message(
    network: str, pinned: Sequence[ipaddress.IPv4Network]
) -> str:
    """Why a network whose subnet the task pins was refused, and what to do."""
    return (
        f"Docker refused this task's network {network}: the task's compose "
        f"file pins its subnet ({', '.join(str(n) for n in pinned)}), and a "
        "network the daemon already has overlaps it (another trial of this "
        "or another task, another BenchFlow process, or a network made outside "
        "BenchFlow). Choose non-overlapping subnets in the task's ipam, or "
        "stop/remove the conflicting network only if you own it. If another "
        "trial of this task holds that subnet, run those trials one at a time. "
        "Choosing a different automatic /28 cannot fix a pinned-subnet conflict."
    )


def is_pools_exhausted_error(message: str) -> bool:
    """Whether the daemon ran out of address pools for a network it chose."""
    return bool(_POOLS_EXHAUSTED.search(message))


def pools_exhausted_message(*, chooses: bool, unreadable: bool = False) -> str:
    """What to do when the daemon's own address pools are used up.

    ``chooses``: whether BenchFlow chose this trial's subnets (then only a
    network it left to the daemon ran out). ``unreadable``: BenchFlow chose
    nothing because the task's compose file uses ``include`` or does not
    parse, so a range in the setting would not help.
    """
    if unreadable and not chooses:
        return (
            "Docker's address pools are used up: the daemon gives each trial "
            "network a whole pool chunk, about 30 on a stock daemon, and "
            "BenchFlow cannot give this task's networks small subnets because "
            "its compose file uses include or does not parse. Lower "
            "--concurrency, or keep the task's networks in a compose file "
            "without include."
        )
    if chooses:
        what = (
            "Docker has no address pool left for a network of this trial that "
            "BenchFlow leaves to the daemon (one the task's compose file "
            "configures itself)"
        )
    else:
        what = (
            "Docker's address pools are used up: the daemon gives each trial "
            "network a whole pool chunk, about 30 on a stock daemon"
        )
    return (
        f"{what}. Lower --concurrency, or set {ENV_VAR} to IPv4 ranges that "
        "are free on the Docker host (for example the daemon's "
        "default-address-pools) so BenchFlow gives each trial a /28 from them."
    )


class SubnetRangeExhausted(RuntimeError):
    """The range has no free subnet left for a trial's networks."""


@dataclass(frozen=True)
class NetworkNeed:
    """A network of the trial's project that BenchFlow gives a subnet."""

    name: str  # the compose key: "default", or the task's network
    prefix: int = NETWORK_PREFIX


@dataclass(frozen=True)
class NetworkPlan:
    """What a trial's project needs: the networks BenchFlow gives a subnet,
    and the IPv4 subnets the task's own networks pin (its ``ipam``), which
    those must not overlap."""

    needs: list[NetworkNeed]
    pinned: list[ipaddress.IPv4Network]


def prefix_for(containers: int) -> int | None:
    """The smallest network (largest prefix) with room for *containers*.

    ``None`` when even a /20 is too small; the daemon then chooses.
    """
    need = max(containers, 1) + _SPARE_PER_NETWORK
    for prefix in range(NETWORK_PREFIX, _LARGEST_PREFIX - 1, -1):
        if 2 ** (32 - prefix) - _RESERVED_PER_NETWORK >= need:
            return prefix
    return None


# --- which networks of a task get a subnet --------------------------------


def _compose_loader() -> type:
    """A SafeLoader that also reads Compose's ``!reset`` and ``!override``."""
    import yaml

    class ComposeLoader(yaml.SafeLoader):
        pass

    def override(loader: yaml.SafeLoader, node: yaml.Node) -> Any:
        if isinstance(node, yaml.MappingNode):
            return loader.construct_mapping(node, deep=True)
        if isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node, deep=True)
        assert isinstance(node, yaml.ScalarNode)
        return loader.construct_scalar(node)

    ComposeLoader.add_constructor("!override", override)
    ComposeLoader.add_constructor("!reset", lambda loader, node: None)
    return ComposeLoader


def _replicas(service: Mapping[str, Any]) -> int:
    for value in (service.get("scale"), (service.get("deploy") or {}).get("replicas")):
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return 1


def _attached(service: Mapping[str, Any]) -> list[str]:
    """The networks a service joins (Compose's rules: none with a
    ``network_mode``, ``default`` without a ``networks`` key)."""
    if service.get("network_mode") is not None:
        return []
    networks = service.get("networks")
    if networks is None:
        return ["default"]
    if isinstance(networks, Mapping):
        return [str(name) for name in networks]
    if isinstance(networks, list):
        return [str(name) for name in networks]
    return []


def _pinned(config: Any) -> list[ipaddress.IPv4Network]:
    """The IPv4 subnets a top-level network's own ``ipam.config`` sets."""
    ipam = config.get("ipam") if isinstance(config, Mapping) else None
    configs = ipam.get("config") if isinstance(ipam, Mapping) else None
    out: list[ipaddress.IPv4Network] = []
    for entry in configs if isinstance(configs, list) else ():
        if not isinstance(entry, Mapping) or entry.get("subnet") is None:
            continue
        try:
            net = ipaddress.ip_network(str(entry["subnet"]), strict=False)
        except ValueError:
            continue
        if isinstance(net, ipaddress.IPv4Network):
            out.append(net)
    return out


def _left_to_task(config: Any) -> bool:
    """Whether a top-level network's own settings rule out a chosen subnet."""
    if config is None:
        return False
    if not isinstance(config, Mapping):
        return True
    if "ipam" in config or config.get("external") or "name" in config:
        return True
    if config.get("enable_ipv6"):
        return True
    driver = config.get("driver")
    return driver is not None and str(driver) != "bridge"


class SubnetConfigError(ValueError):
    """Compose could not resolve a task's dynamic network configuration."""


def _admission_sandbox(
    task_compose: Path,
    environment_manifest: Any = None,
    config_override: dict | None = None,
):
    """Build startup's Compose environment without starting a sandbox."""
    from benchflow.sandbox.docker import DockerSandbox
    from benchflow.task.config import SandboxConfig
    from benchflow.task.task import Task

    task_dir = task_compose.parent.parent
    config = SandboxConfig()
    if (task_dir / "task.md").is_file() or (task_dir / "task.toml").is_file():
        from benchflow._utils.config_override import apply_config_override

        task_config = Task(task_dir).config
        if config_override:
            task_config = apply_config_override(task_config, config_override)
        config = task_config.sandbox
    persistent_env = None
    if environment_manifest is not None:
        from benchflow.environment.manifest import (
            resolve_manifest_image,
            resolve_manifest_runtime_env,
        )

        image = resolve_manifest_image(environment_manifest)
        if image:
            config = config.model_copy(update={"docker_image": image})
        persistent_env = resolve_manifest_runtime_env(
            environment_manifest, task_id=task_dir.name
        )
    return DockerSandbox(
        environment_dir=task_compose.parent,
        environment_name=task_dir.name,
        session_id="subnet-admission",
        rollout_paths=None,
        task_env_config=config,
        persistent_env=persistent_env,
    )


def _resolved_compose(
    path: Path,
    *,
    env: Mapping[str, str] | None,
    main_networked: bool,
    backend: str,
    environment_manifest: Any,
    config_override: dict | None,
) -> Any:
    """Let Compose resolve its own interpolation, dotenv, includes and extends.

    This is config only: it neither contacts a daemon nor runs task commands.
    Do not log its output or errors; either can contain environment values.
    """
    from benchflow.sandbox._compose import COMPOSE_NO_NETWORK_PATH

    sandbox = _admission_sandbox(path, environment_manifest, config_override)
    args = [
        "docker",
        "compose",
        "--project-name",
        "benchflow-subnet-admission",
        "--project-directory",
        str(path.parent.resolve()),
    ]
    for compose in (
        sandbox._DOCKER_COMPOSE_BASE_PATH,
        sandbox._DOCKER_COMPOSE_BUILD_PATH,
        path,
    ):
        args += ["-f", str(compose.resolve())]
    if not main_networked:
        args += ["-f", str(COMPOSE_NO_NETWORK_PATH)]
    resolved_env = dict(env) if env is not None else sandbox._docker_compose_env()
    if env is None and backend == "remote-docker":
        from benchflow.sandbox.remote_docker import resolve_remote_docker_host

        resolved_env = resolve_remote_docker_host().apply(resolved_env)
    try:
        import yaml

        # Unnormalized Compose does not attach implicit default networks and
        # drops unused declarations. A config-only sentinel keeps the authored
        # default declaration alive; remove it before counting services. YAML
        # also avoids JSON's synthetic empty ipam objects. Thus only authored
        # network names/settings can opt out, even through include/extends.
        sentinel = "benchflow-subnet-" + uuid.uuid4().hex
        with tempfile.TemporaryDirectory(prefix="benchflow-subnet-") as tmp:
            keep_default = Path(tmp) / "default.json"
            keep_default.write_text(
                json.dumps(
                    {
                        "services": {
                            sentinel: {
                                "image": "scratch",
                                "networks": ["default"],
                            }
                        }
                    }
                )
            )
            result = subprocess.run(
                [
                    *args,
                    "-f",
                    str(keep_default),
                    "config",
                    "--format",
                    "yaml",
                    "--no-env-resolution",
                    "--no-normalize",
                    "--no-consistency",
                ],
                env=resolved_env,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=30,
                check=True,
            )
        loaded = yaml.safe_load(result.stdout)
        loaded["services"].pop(sentinel)
        return loaded
    except (OSError, subprocess.SubprocessError, ValueError, yaml.YAMLError):
        raise SubnetConfigError(
            "Cannot resolve this task's Docker network configuration. Check "
            "docker compose config and the task's interpolation variables "
            "before allocating trial networks."
        ) from None


def plan_networks(
    task_compose: Path | None,
    *,
    main_networked: bool = True,
    compose_env: Mapping[str, str] | None = None,
    backend: str = "docker",
    environment_manifest: Any = None,
    config_override: dict | None = None,
) -> NetworkPlan | None:
    """The networks of a trial's project BenchFlow gives a subnet, with sizes.

    *task_compose* is the task's ``docker-compose.yaml`` (``None`` for a
    Dockerfile task: only ``main`` on ``default``). ``main_networked`` is
    False when BenchFlow's no-network file gives ``main`` ``network_mode:
    none``. A network no service joins is skipped: Compose does not create
    it. Returns ``None`` when the file cannot be read or includes other
    files that Compose cannot resolve: a network there may set its own
    ``ipam``, so the daemon chooses for every network, as before. Dynamic
    topology uses Compose's config resolver and startup's environment; a
    failed interpolation refuses admission instead of losing reservations.

    The plan's ``pinned`` lists the subnets of every network the task
    configures with its own ``ipam`` (a cyber task's 172.20.0.0/16, say):
    the daemon refuses a second network that overlaps one, so the trial's
    other networks must stay clear of them.
    """
    loaded: Any = {}
    if task_compose is not None and task_compose.is_file():
        import yaml

        try:
            # A SafeLoader with two more tags: it builds no Python objects.
            text = task_compose.read_text(encoding="utf-8")
            loaded = yaml.load(text, Loader=_compose_loader())
        except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
            logger.debug("Docker subnets: %s does not parse (%s)", task_compose, exc)
            return None
        if loaded is None:
            loaded = {}
        services = loaded.get("services") if isinstance(loaded, Mapping) else None
        topology = {
            "networks": loaded.get("networks") if isinstance(loaded, Mapping) else None,
            "services": {
                name: {
                    k: v
                    for k, v in service.items()
                    if k in ("networks", "network_mode", "scale", "deploy", "extends")
                }
                for name, service in (
                    services.items() if isinstance(services, Mapping) else ()
                )
                if isinstance(service, Mapping)
            },
        }
        dynamic = "$" in json.dumps(topology, default=str)
        if isinstance(loaded, Mapping) and (
            dynamic
            or "include" in loaded
            or any(
                isinstance(service, Mapping) and "extends" in service
                for service in (
                    services.values() if isinstance(services, Mapping) else ()
                )
            )
        ):
            try:
                loaded = _resolved_compose(
                    task_compose,
                    env=compose_env,
                    main_networked=main_networked,
                    backend=backend,
                    environment_manifest=environment_manifest,
                    config_override=config_override,
                )
            except SubnetConfigError:
                if dynamic:
                    # Skipping an unresolved pin would let earlier tasks
                    # allocate inside it. Refuse before those tasks start.
                    raise
                return None
        if not isinstance(loaded, Mapping) or "include" in loaded:
            return None
    services = loaded.get("services") or {}
    declared = loaded.get("networks") or {}
    if not isinstance(services, Mapping) or not isinstance(declared, Mapping):
        return None
    if "main" not in services:
        # BenchFlow's base file always defines main, on the default network.
        services = {**services, "main": {}}
    counts: dict[str, int] = {}
    for key, service in services.items():
        if not isinstance(service, Mapping):
            service = {}
        if key == "main" and not main_networked:
            continue
        for name in _attached(service):
            counts[name] = counts.get(name, 0) + _replicas(service)
    needs: list[NetworkNeed] = []
    pinned: list[ipaddress.IPv4Network] = []
    for name in ["default", *sorted(str(n) for n in declared if n != "default")]:
        if not counts.get(name):
            continue
        config = declared.get(name)
        if _left_to_task(config):
            if not (isinstance(config, Mapping) and config.get("external")):
                pinned += _pinned(config)
            continue
        prefix = prefix_for(counts[name])
        if prefix is not None:
            needs.append(NetworkNeed(name, prefix))
    return NetworkPlan(needs, pinned)


def override_document(subnets: Mapping[str, ipaddress.IPv4Network]) -> dict[str, Any]:
    """The compose override that gives each network its subnet."""
    return {
        "networks": {
            name: {"ipam": {"config": [{"subnet": str(net)}]}}
            for name, net in subnets.items()
        }
    }


# --- the range and what in it is taken ------------------------------------


def parse_range(text: str) -> list[ipaddress.IPv4Network]:
    """``BENCHFLOW_DOCKER_SUBNETS``'s ranges. Raises ValueError with the fix."""
    ranges: list[ipaddress.IPv4Network] = []
    for item in re.split(r"[,\s]+", text.strip()):
        if not item:
            continue
        try:
            net = ipaddress.ip_network(item, strict=False)
        except ValueError:
            net = None
        if not isinstance(net, ipaddress.IPv4Network):
            raise ValueError(
                f"{ENV_VAR}: {item!r} is not an IPv4 range; give ranges such "
                "as 172.20.0.0/14,192.168.0.0/16, or 'daemon' to let Docker "
                "choose"
            )
        if net.prefixlen > NETWORK_PREFIX:
            raise ValueError(
                f"{ENV_VAR}: {item} is smaller than one /{NETWORK_PREFIX} trial network"
            )
        ranges.append(net)
    if not ranges:
        raise ValueError(f"{ENV_VAR} names no range")
    return ranges


def daemon_pools(pools: Any) -> list[ipaddress.IPv4Network]:
    """The IPv4 bases of ``docker info``'s ``DefaultAddressPools``, or
    Docker's built-in pools when the daemon sets none."""
    return [base for base, _ in daemon_pool_chunks(pools)]


def daemon_pool_chunks(pools: Any) -> list[tuple[ipaddress.IPv4Network, int]]:
    """Each pool of :func:`daemon_pools` with the prefix of the chunks the
    daemon carves it into: its ``Size``, or for Docker's built-in pools a
    /16 in 172.17-31 and a /20 in 192.168.0.0/16.

    A ``Size`` missing, or one that does not fit inside the base, makes the
    whole base one chunk (the cautious reading).
    """
    chunks: list[tuple[ipaddress.IPv4Network, int]] = []
    if isinstance(pools, list):
        for pool in pools:
            if not isinstance(pool, Mapping):
                continue
            try:
                net = ipaddress.ip_network(str(pool.get("Base")), strict=False)
            except ValueError:
                continue
            if not isinstance(net, ipaddress.IPv4Network):
                continue
            size = pool.get("Size")
            if (
                isinstance(size, bool)
                or not isinstance(size, int)
                or not net.prefixlen <= size <= 32
            ):
                size = net.prefixlen
            chunks.append((net, size))
    if chunks:
        return chunks
    return [
        (net, 20 if net.subnet_of(_BUILTIN_192) else 16)
        for net in (ipaddress.IPv4Network(p) for p in DOCKER_DEFAULT_POOLS)
    ]


def _short_host(name: str) -> str:
    return name.strip().split(".", 1)[0].lower()


def daemon_on_this_host(daemon_name: str) -> bool:
    """Whether the daemon shares this host's routes (so they can be read)."""
    return (
        sys.platform.startswith("linux")
        and bool(daemon_name.strip())
        and _short_host(daemon_name) == _short_host(socket.gethostname())
    )


def _hex_ipv4(field: str) -> int:
    # /proc/net/route prints addresses as host-order (little-endian) hex.
    return int.from_bytes(int(field, 16).to_bytes(4, "little"), "big")


ROUTE_FILE = Path("/proc/net/route")
RESOLV_FILES: tuple[Path, ...] = (
    Path("/etc/resolv.conf"),
    Path("/run/systemd/resolve/resolv.conf"),
)
# The bridges of Docker's own networks: the default bridge, swarm's gateway
# bridge, and "br-" plus the first 12 hex digits of a network's id.
_DOCKER_BRIDGE = re.compile(r"docker0|docker_gwbridge|br-[0-9a-f]{12}")


#: /proc/net/route's flag for a route through a gateway (``RTF_GATEWAY``).
_RTF_GATEWAY = 0x2


def host_reserved(
    route_file: Path | None = None,
    resolv_files: Sequence[Path] | None = None,
) -> list[ipaddress.IPv4Network]:
    """What Docker's own pool choice avoids on this host: every on-link route
    (one with no gateway) but the default one, and every IPv4 nameserver.

    Routes through a gateway are not counted: since Docker 25 the daemon
    counts only on-link routes, because a VPN that sends all traffic
    through its tunnel adds 0.0.0.0/1 and 128.0.0.0/1 via its gateway, and
    counting those would leave no pool free.

    Routes on the bridges of Docker's own networks are not counted: those
    networks are the daemon's (read from it, and the daemon refuses an
    overlap with them), and in the daemon's pools they are often
    BenchFlow's own /28s, which must not block their whole pool chunk
    (:class:`SubnetAllocator`'s ``chunks``).
    """
    route_file = ROUTE_FILE if route_file is None else route_file
    resolv_files = RESOLV_FILES if resolv_files is None else resolv_files
    reserved: list[ipaddress.IPv4Network] = []
    try:
        lines = route_file.read_text().splitlines()[1:]
    except OSError:
        lines = []
    for line in lines:
        fields = line.split()
        if len(fields) < 8 or _DOCKER_BRIDGE.fullmatch(fields[0]):
            continue
        try:
            dest, mask = _hex_ipv4(fields[1]), _hex_ipv4(fields[7])
            flags = int(fields[3], 16)
        except ValueError:
            continue
        if mask == 0 or flags & _RTF_GATEWAY:
            continue
        prefix = bin(mask).count("1")
        try:
            reserved.append(ipaddress.IPv4Network((dest & mask, prefix)))
        except ValueError:
            continue
    for path in resolv_files:
        try:
            text = path.read_text()
        except OSError:
            continue
        for line in text.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "nameserver":
                try:
                    addr = ipaddress.ip_address(parts[1].split("%", 1)[0])
                except ValueError:
                    continue
                if isinstance(addr, ipaddress.IPv4Address):
                    reserved.append(ipaddress.IPv4Network(addr))
    return reserved


def network_subnets(inspect_lines: str) -> list[ipaddress.IPv4Network]:
    """The IPv4 subnets in ``docker network inspect --format
    '{{json .IPAM.Config}}'`` output (one JSON list per line)."""
    subnets: list[ipaddress.IPv4Network] = []
    for line in inspect_lines.splitlines():
        line = line.strip()
        if not line.startswith("["):
            continue
        try:
            configs = json.loads(line)
        except ValueError:
            continue
        for config in configs or ():
            if not isinstance(config, Mapping):
                continue
            try:
                net = ipaddress.ip_network(str(config.get("Subnet")), strict=False)
            except ValueError:
                continue
            if isinstance(net, ipaddress.IPv4Network):
                subnets.append(net)
    return subnets


# --- the allocator ----------------------------------------------------------

_Span = tuple[int, int]  # first and last address, inclusive


def _span(net: ipaddress.IPv4Network) -> _Span:
    return int(net.network_address), int(net.broadcast_address)


def _merged(spans: Iterable[_Span]) -> list[_Span]:
    out: list[_Span] = []
    for lo, hi in sorted(spans):
        if out and lo <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def _align_up(value: int, size: int) -> int:
    return -(-value // size) * size


class SubnetAllocator:
    """Hands out free subnets of a range to one process's trials.

    Thread-safe; the daemon, not this object, settles races between
    processes (:mod:`benchflow.sandbox._docker_subnets`).
    """

    def __init__(
        self,
        ranges: Sequence[ipaddress.IPv4Network],
        *,
        reserved: Callable[[], Iterable[ipaddress.IPv4Network]] | None = None,
        chunks: Sequence[tuple[ipaddress.IPv4Network, int]] = (),
        start: int | None = None,
        description: str = "",
        daemon: DockerDaemon | None = None,
    ) -> None:
        """*reserved* gives this host's routes and nameservers, read at each
        allocation. *chunks* are the daemon's pools with the prefix it
        carves each into (:func:`daemon_pool_chunks`): a reservation inside
        one then blocks every chunk it touches, as Docker's own choice
        does, not only its own addresses. Without that, a /28 next to a
        host route (another subnet of an AWS VPC on 172.31.16.0/20, a LAN's
        neighbour of 192.168.1.0/24) would get a bridge route that takes
        the host's traffic for it."""
        self._pools = _merged(_span(n) for n in ranges)
        if not self._pools:
            raise ValueError("no range to choose subnets from")
        self._reserved = reserved
        self._chunks = [(*_span(base), 2 ** (32 - prefix)) for base, prefix in chunks]
        self._taken: list[_Span] = []
        self._live: set[_Span] = set()
        self._lock = threading.Lock()
        self.description = description or ", ".join(str(n) for n in ranges)
        self._daemon = daemon_key(None) if daemon is None else daemon
        size = 2 ** (32 - NETWORK_PREFIX)
        total = sum((hi - lo + 1) // size for lo, hi in self._pools)
        if start is None:
            # A random point of the range, so two processes seldom meet.
            start = random.SystemRandom().randrange(max(total, 1))
        self._cursor = self._address_at(start % max(total, 1), size)

    def _address_at(self, index: int, size: int) -> int:
        for lo, hi in self._pools:
            count = (hi - lo + 1) // size
            if index < count:
                return _align_up(lo, size) + index * size
            index -= count
        return self._pools[0][0]

    @property
    def live(self) -> list[ipaddress.IPv4Network]:
        """The subnets this process holds."""
        with self._lock:
            return [_network(lo, hi) for lo, hi in sorted(self._live)]

    def set_taken(self, subnets: Iterable[ipaddress.IPv4Network]) -> None:
        """The subnets of the daemon's networks, as last read."""
        spans = [_span(n) for n in subnets]
        with self._lock:
            self._taken = spans

    def _whole_chunks(self, span: _Span) -> list[_Span]:
        """*span*, and every pool chunk it touches."""
        lo, hi = span
        out = [span]
        for base_lo, base_hi, size in self._chunks:
            if hi < base_lo or lo > base_hi:
                continue
            first = base_lo + (max(lo, base_lo) - base_lo) // size * size
            last = base_lo + (min(hi, base_hi) - base_lo) // size * size + size - 1
            out.append((first, min(last, base_hi)))
        return out

    def _host_spans(self) -> list[_Span]:
        """This host's routes and nameservers, each grown to its pool chunks."""
        return [
            chunk
            for n in (self._reserved() if self._reserved else ())
            for chunk in self._whole_chunks(_span(n))
        ]

    def has_room(self) -> bool:
        """Whether this host's routes and nameservers leave a /28 of the range."""
        blocked = _merged(self._host_spans())
        with self._lock:
            return self._find(2 ** (32 - NETWORK_PREFIX), blocked) is not None

    def allocate(
        self,
        needs: Sequence[NetworkNeed],
        avoid: Iterable[ipaddress.IPv4Network] = (),
    ) -> dict[str, ipaddress.IPv4Network]:
        """A free subnet for each network; raises :class:`SubnetRangeExhausted`.

        *avoid*: subnets the trial's own networks pin (``NetworkPlan.pinned``).
        They join the process's pinned subnets (:func:`pin_subnets`), which
        this and every later call on this daemon blocks, for every trial: a /28 handed to
        another trial inside a task's 172.20.0.0/16 would make the daemon
        refuse that task's network while the /28 lives.
        """
        pin_subnets(avoid, key=self._daemon)
        reserved = self._host_spans() + pinned_spans(key=self._daemon)
        with self._lock:
            chosen: dict[str, ipaddress.IPv4Network] = {}
            for need in needs:
                blocked = _merged([*self._taken, *self._live, *reserved])
                span = self._find(2 ** (32 - need.prefix), blocked)
                if span is None:
                    for net in chosen.values():
                        self._live.discard(_span(net))
                    raise SubnetRangeExhausted(
                        f"no free /{need.prefix} left in {self.description} for "
                        f"this trial's Docker networks ({len(self._live)} held "
                        "by this BenchFlow process, the rest by other networks "
                        f"or this host's routes). Widen the range with {ENV_VAR} "
                        "(IPv4 ranges free on the Docker host) or lower "
                        "--concurrency."
                    )
                self._live.add(span)
                self._cursor = span[1] + 1
                chosen[need.name] = _network(*span)
            return chosen

    def release(self, subnets: Iterable[ipaddress.IPv4Network]) -> None:
        """Give subnets back once their networks are removed."""
        with self._lock:
            for net in subnets:
                self._live.discard(_span(net))

    def step_aside(self) -> None:
        """Move the cursor a random few subnets ahead.

        After the daemon refused a subnet, another process is handing out
        subnets right where this one is; walking on in step with it would
        meet it again on networks it is creating now, which no re-read of
        the daemon shows yet.
        """
        size = 2 ** (32 - NETWORK_PREFIX)
        with self._lock:
            self._cursor += size * random.SystemRandom().randrange(1, 65)

    def _find(self, size: int, blocked: list[_Span]) -> _Span | None:
        """The first aligned free block of *size* at or after the cursor,
        wrapping round the range once."""
        # The pool holding the cursor, or the next one (the cursor sits just
        # past the last subnet handed out, which may end a pool).
        here = next(
            (i for i, (_, hi) in enumerate(self._pools) if self._cursor <= hi), 0
        )
        order = self._pools[here:] + self._pools[:here]
        segments = [(max(self._cursor, order[0][0]), order[0][1]), *order[1:], order[0]]
        ends = [hi for _, hi in blocked]
        for lo, hi in segments:
            start = _align_up(lo, size)
            while start + size - 1 <= hi:
                i = bisect.bisect_left(ends, start)
                if i == len(blocked) or blocked[i][0] > start + size - 1:
                    return start, start + size - 1
                start = _align_up(blocked[i][1] + 1, size)
        return None


def _network(lo: int, hi: int) -> ipaddress.IPv4Network:
    return ipaddress.IPv4Network((lo, 32 - (hi - lo + 1).bit_length() + 1))


# --- one allocator per daemon, per process ---------------------------------


class _DockerRunner(Protocol):
    def __call__(self, args: list[str]) -> Awaitable[Any]: ...


# --- subnets tasks pin, kept clear in this process ------------------------

_PINNED: dict[str, list[_Span]] = {}
_PINNED_LOCK = threading.Lock()


def pin_subnets(
    subnets: Iterable[ipaddress.IPv4Network], *, key: DockerDaemon | None = None
) -> None:
    """Keep *subnets* (a task's own ``ipam`` subnets) clear of every subnet
    this process chooses from now on, on the same daemon."""
    spans = [_span(n) for n in subnets]
    if not spans:
        return
    identity = daemon_identity(daemon_key(None) if key is None else key)
    with _PINNED_LOCK:
        _PINNED[identity] = _merged([*_PINNED.get(identity, []), *spans])


def pinned_spans(*, key: DockerDaemon | None = None) -> list[_Span]:
    """The subnets :func:`pin_subnets` has kept clear so far."""
    with _PINNED_LOCK:
        if not _PINNED:
            return []
    identity = daemon_identity(daemon_key(None) if key is None else key)
    with _PINNED_LOCK:
        return list(_PINNED.get(identity, []))


def pin_task_subnets(
    compose_files: Iterable[Path],
    *,
    backend: str = "docker",
    environment_manifest: Any = None,
    config_override: dict | None = None,
) -> list[ipaddress.IPv4Network]:
    """Pin the subnets the tasks' compose files set for their own networks.

    A job calls this with every task it will run before the first trial
    starts, so no trial of another task takes a /28 inside them first.
    Returns what it pinned.
    """
    pinned: list[ipaddress.IPv4Network] = []
    for path in compose_files:
        try:
            plan = plan_networks(
                path,
                backend=backend,
                environment_manifest=environment_manifest,
                config_override=config_override,
            )
        except SubnetConfigError:
            raise
        except Exception as exc:  # a task's file is the task's problem
            logger.debug("Docker subnets: %s not read (%s)", path, exc)
            continue
        if plan is not None:
            pinned += plan.pinned
    if pinned:
        pin_subnets(pinned, key=backend_daemon_key(backend))
    return pinned


# Successful allocators own daemon-wide leases. A fallback, or a build that
# might fall back, belongs only to its client: another alias may be local and
# able to check host routes even when this client reaches the daemon remotely.
_ALLOCATORS: dict[str, SubnetAllocator] = {}
_FALLBACKS: set[tuple[str, str]] = set()
_PENDING: dict[tuple[str, str], tuple[asyncio.AbstractEventLoop, asyncio.Task]] = {}
_REGISTRY_LOCK = threading.Lock()

# Environment selectors used by the ordinary CLI (also cleared by test fixtures).
# They are not cache keys: only a queried DaemonIdentity may name shared state.
_CLIENT_SETTINGS = (
    "DOCKER_HOST",
    "DOCKER_CONTEXT",
    "DOCKER_CONFIG",
    "DOCKER_TLS_VERIFY",
    "DOCKER_CERT_PATH",
    "DOCKER_TLS",
    "HOME",
)


@dataclass(frozen=True)
class DaemonIdentity:
    """Queried server identity and the endpoint used to reach it.

    Server-wide locks and reservations share the ID across aliases. Client
    decisions (fallbacks, routes and disk) additionally use the endpoint.
    Neither endpoint nor daemon metadata is diagnostic text.
    """

    server_id: str = field(repr=False)
    endpoint: str = field(repr=False)

    @property
    def server_key(self) -> str:
        return hashlib.sha256(self.server_id.encode()).hexdigest()

    @property
    def connection_key(self) -> tuple[str, str]:
        return self.server_key, hashlib.sha256(self.endpoint.encode()).hexdigest()


@dataclass(eq=False)
class DockerDaemon:
    """One client's immutable transport snapshot, resolved once per sandbox.

    Only the originating provider constructs this object. Probes use its
    actual subprocess environment; they never manufacture another provider
    from a URL scheme. No process-wide cache can answer a new client's ID.
    """

    env: Mapping[str, str] = field(repr=False)
    endpoint_hint: str | None = field(default=None, repr=False)
    redact: Callable[[str], str] | None = field(default=None, repr=False)
    _identity: DaemonIdentity | None = field(default=None, init=False, repr=False)
    _checked: bool = field(default=False, init=False, repr=False)
    _async_lock: asyncio.Lock = field(
        default_factory=asyncio.Lock, init=False, repr=False
    )
    _sync_lock: Any = field(default_factory=threading.RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        from types import MappingProxyType

        self.env = MappingProxyType(dict(self.env))

    def bind(self, server_id: str) -> None:
        with self._sync_lock:
            if self._checked:
                return
            # Empty/error output never gets a guessed identity or shared key.
            if server_id and not any(c.isspace() for c in server_id):
                self._identity = DaemonIdentity(server_id, _read_endpoint(self))
            self._checked = True

    def identity(self) -> DaemonIdentity:
        with self._sync_lock:
            if not self._checked:
                try:
                    self.bind(_read_daemon_id(self))
                finally:
                    self._checked = True
            if self._identity is None:
                raise SubnetConfigError(_IDENTITY_ERROR)
            return self._identity

    async def identify(self) -> DaemonIdentity:
        # Layer readers use the same async lock around their cancellable runner.
        # Subnet consumers only call identity() after this resolution finishes.
        async with self._async_lock:
            return await asyncio.to_thread(self.identity)


_IDENTITY_ERROR = (
    "Cannot identify the Docker daemon for subnet admission. Check docker "
    "info with this backend's client settings before starting trials."
)


def daemon_key(client_env: Mapping[str, str] | None) -> DockerDaemon:
    """Capture an ordinary Docker client's exact environment, without RD policy."""
    return DockerDaemon(dict(os.environ if client_env is None else client_env))


def backend_daemon_key(backend: str) -> DockerDaemon:
    """Capture the transport chosen by the originating provider, also in workers."""
    if backend == "remote-docker":
        from benchflow.sandbox.remote_docker import resolve_remote_docker_host

        host = resolve_remote_docker_host()
        return DockerDaemon(host.client_env(), host.url, host.redact)
    return daemon_key(None)


def _read_daemon_id(key: DockerDaemon) -> str:
    """Ask this client's daemon. Never substitute a transport based on its URL."""
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{.ID}}"],
            env=dict(key.env),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        identity = result.stdout.strip()
        if identity:
            return identity
    except (OSError, subprocess.SubprocessError) as exc:
        # The provider boundary owns all diagnostic transformations, including
        # exceptions. Admission deliberately reports a fixed actionable message.
        if key.redact is not None:
            key.redact(str(exc))
    raise SubnetConfigError(_IDENTITY_ERROR) from None


def daemon_identity(key: DockerDaemon) -> str:
    """Payload-safe server identity; workers query through their own transport."""
    return key.identity().server_key


async def read_taken(docker: _DockerRunner) -> list[ipaddress.IPv4Network]:
    """The subnets of every network the daemon has now."""
    listed = await docker(["network", "ls", "--quiet", "--no-trunc"])
    ids = (getattr(listed, "stdout", "") or "").split()
    if not ids:
        return []
    # A network removed between the two calls makes inspect exit non-zero;
    # the others are still printed.
    inspected = await docker(
        ["network", "inspect", "--format", "{{json .IPAM.Config}}", *ids]
    )
    return network_subnets(getattr(inspected, "stdout", "") or "")


def _read_endpoint(key: DockerDaemon) -> str:
    """Resolve the endpoint once, using the client's actual context precedence."""
    if key.endpoint_hint is not None:
        return key.endpoint_hint
    if (
        not key.env.get("DOCKER_CONTEXT")
        and key.env.get("DOCKER_HOST")
        and not key.env["DOCKER_HOST"].startswith("unix:///")
    ):
        return key.env["DOCKER_HOST"]
    try:
        result = subprocess.run(
            [
                "docker",
                "context",
                "inspect",
                "--format",
                "{{json .Endpoints.docker.Host}}",
            ],
            env=dict(key.env),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        endpoint = json.loads(result.stdout)
        return endpoint if isinstance(endpoint, str) else ""
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        if key.redact is not None:
            key.redact(str(exc))
        return ""  # Unknown never authorizes reading client routes or disk.


def _uses_local_socket(key: DockerDaemon) -> bool:
    return key.identity().endpoint.startswith("unix:///")


async def _build(
    docker: _DockerRunner,
    key: DockerDaemon | None = None,
    *,
    daemon_display: str = "Docker daemon",
) -> SubnetAllocator | None:
    key = daemon_key(None) if key is None else key
    setting = os.environ.get(ENV_VAR, "").strip()
    if setting.lower() in _OFF_VALUES:
        return None
    info = await docker(
        [
            "info",
            "--format",
            '{"name":{{json .Name}},"pools":{{json .DefaultAddressPools}}}',
        ]
    )
    try:
        facts = json.loads((getattr(info, "stdout", "") or "").strip() or "{}")
    except ValueError:
        facts = {}
    if not isinstance(facts, dict) or not facts.get("name"):
        # Not cached: the next trial asks again (a daemon still starting).
        raise RuntimeError(
            "docker info gave no daemon name: "
            + ((getattr(info, "stderr", "") or "").strip()[:200] or "no output")
        )
    name = str(facts["name"])
    local = daemon_on_this_host(name) and await asyncio.to_thread(
        _uses_local_socket, key
    )
    chunks: list[tuple[ipaddress.IPv4Network, int]] = []
    if setting:
        ranges = parse_range(setting)
        description = f"{ENV_VAR}={setting}"
    elif not local:
        logger.info(
            "Docker trial networks: %s is not on this host, so "
            "BenchFlow cannot check its routes and lets it choose each "
            "network's subnet (about 30 at once on its default pools). Set %s "
            "to ranges free on that host to run more trials at once.",
            daemon_display,
            ENV_VAR,
        )
        return None
    else:
        # A host route or nameserver blocks its whole pool chunk, as in
        # the daemon's own choice.
        chunks = daemon_pool_chunks(facts.get("pools"))
        ranges = [base for base, _ in chunks]
        description = (
            "the daemon's address pools (" + ", ".join(str(n) for n in ranges) + ")"
        )
    allocator = SubnetAllocator(
        ranges,
        reserved=host_reserved if local else None,
        chunks=chunks,
        description=description,
        daemon=key,
    )
    if not setting and not allocator.has_room():
        logger.info(
            "Docker trial networks: this host's routes or nameservers overlap "
            "every chunk of the daemon's address pools (%s), so the daemon "
            "chooses each network's subnet, as before 0.8 (about 30 at once "
            "on its default pools). Set %s to ranges free on this host to run "
            "more trials at once.",
            ", ".join(str(n) for n in ranges),
            ENV_VAR,
        )
        return None
    allocator.set_taken(await read_taken(docker))
    return allocator


async def allocator_for(
    key: DockerDaemon,
    docker: _DockerRunner,
    *,
    daemon_display: str = "Docker daemon",
) -> SubnetAllocator | None:
    """Share successful allocators by daemon, but build/fall back per client.

    Return ``None`` when this client leaves subnet selection to the daemon.

    Raises ValueError for a malformed ``BENCHFLOW_DOCKER_SUBNETS``.
    """
    if os.environ.get(ENV_VAR, "").strip().lower() in _OFF_VALUES:
        return None
    try:
        identity = daemon_identity(key)
        connection = key.identity().connection_key
    except SubnetConfigError:
        with _PINNED_LOCK:
            if _PINNED:
                raise
        # With no reservations, an unreadable daemon keeps the old behavior:
        # choose nothing and let Compose report its own connection error.
        # Never cache a guessed identity or allocate under one.
        return None
    with _REGISTRY_LOCK:
        if identity in _ALLOCATORS:
            return _ALLOCATORS[identity]
        if connection in _FALLBACKS:
            return None
        loop = asyncio.get_running_loop()
        pending = _PENDING.get(connection)
        if pending is None or pending[0] is not loop or pending[1].done():
            task = loop.create_task(_build(docker, key, daemon_display=daemon_display))
            _PENDING[connection] = (loop, task)
        else:
            task = pending[1]
    try:
        allocator = await asyncio.shield(task)
    except ValueError:
        raise
    except Exception as exc:
        # docker info failed: the trial's own compose call reports why.
        logger.debug("Docker subnets: could not read the daemon (%s)", exc)
        with _REGISTRY_LOCK:
            if _PENDING.get(connection, (None, None))[1] is task:
                _PENDING.pop(connection, None)
        return None
    with _REGISTRY_LOCK:
        if allocator is None:
            _FALLBACKS.add(connection)
        else:
            # Alias builds may complete together; publish one allocator so
            # every caller shares the same live leases and reservations.
            _ALLOCATORS.setdefault(identity, allocator)
        if _PENDING.get(connection, (None, None))[1] is task:
            _PENDING.pop(connection, None)
        return _ALLOCATORS.get(identity)


def reset_allocators() -> None:
    """Forget every allocator and pinned subnet (tests, and a changed
    ``BENCHFLOW_DOCKER_SUBNETS``)."""
    with _REGISTRY_LOCK:
        _ALLOCATORS.clear()
        _FALLBACKS.clear()
        _PENDING.clear()
    with _PINNED_LOCK:
        _PINNED.clear()
