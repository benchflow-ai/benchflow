"""The direct Docker path for single-service tasks (PERF-E).

Compose starts a trial's ``main`` container with about ten ``docker compose``
processes (create, config, ps, up, ps again for each check, kill, down), each
of which starts the Compose plugin and loads the project. This path asks
Compose once for the project it would run (``compose config --format json``,
the same files, flags and environment), then makes the calls Compose itself
makes to the daemon (NetworkCreate, ContainerCreate, ContainerStart,
ContainerKill, ContainerRemove, NetworkRemove) over the daemon's socket
(:mod:`benchflow.sandbox._engine_api`), with the request bodies Compose sends:
the same container, labels and config hashes included, so ``docker compose
ps`` and every check that reads Compose's labels find it as before.

It is used only when it can reproduce Compose exactly:

- one service, ``main``, and at most the ``default`` network;
- only the service keys in :data:`_SERVICE_KEYS`, in the shapes this module
  renders (anything else, such as a task healthcheck, a named volume or a
  published port, goes to Compose);
- a local daemon reached at the socket the ``docker`` CLI would use
  (:func:`endpoint_socket`), never a context, TCP, TLS or SSH;
- the image already on the daemon (Compose would pull it).

Anything else is a :class:`Fallback` with its reason, recorded in the trial's
``result.json`` (``direct_docker``), and the trial runs on Compose. Every
hardening step after the containers exist (bind custody, the inert barrier,
the metadata helper, the firewall, the protected kit, the exec channel) is
the same code on either path: it reads containers by Compose's labels.

The parity proof is ``tests/test_direct_docker_parity.py`` plus the VM
differential in ``tools/perf/direct_docker_diff.py``: the recorded request
bodies and the containers' inspect, compared under
:func:`normalise_inspect` (the published normalisation).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
from collections import Counter
from pathlib import Path
from typing import Any

from benchflow.sandbox._engine_api import UNBOUNDED

#: The run-wide setting: ``BENCHFLOW_DIRECT_DOCKER=0`` keeps every trial on
#: Compose. On by default.
SETTING_ENV = "BENCHFLOW_DIRECT_DOCKER"

COMPOSE_PROJECT = "com.docker.compose.project"
COMPOSE_SERVICE = "com.docker.compose.service"
COMPOSE_ONEOFF = "com.docker.compose.oneoff"


class Fallback(Exception):
    """The task goes to Compose; the message is the recorded reason."""


#: The Compose builds, exactly as ``docker compose version --short`` prints
#: them (a release candidate, another vendor's build or another packaging of
#: the same release is a different build: PERF-E-r3-A-L2), and the Engine API
#: versions this path's request bodies and hashes are
#: proven against by the differential (``tools/perf/direct_docker_matrix.py``
#: on the TB2 and synthetic sets, plus ``tests/fixtures/direct_docker``). Any
#: other runs on Compose (PERF-E-A-2). To admit one: record its fixture, run
#: the differential on it with no ``different`` case, then add it here.
PROVEN_COMPOSE_VERSIONS = frozenset({"2.40.3+ds1-0ubuntu1~24.04.1"})
PROVEN_API_VERSIONS = frozenset({"1.51"})
#: The daemons' own API versions the differential ran against (Engine 29.x
#: speaks 1.52): what a daemon reports back in inspect (layer 2) is proven
#: only there (PERF-E-B-L5).
PROVEN_DAEMON_API_VERSIONS = frozenset({"1.52"})


def compose_core(printed: str) -> str | None:
    """``2.40.3`` for ``2.40.3+ds1-0ubuntu1~24.04.1`` (or ``v2.40.3``): what
    Compose writes in its ``com.docker.compose.version`` label."""
    match = re.match(r"v?(\d+\.\d+\.\d+)(?:$|[^0-9.])", printed)
    return match.group(1) if match else None


def check_versions(
    compose_version: str, api_version: str, daemon_api_version: str | None
) -> None:
    """Raise :class:`Fallback` unless every version was proven."""
    if compose_version not in PROVEN_COMPOSE_VERSIONS:
        raise Fallback(
            f"Compose {compose_version}: the direct path is proven on "
            f"{', '.join(sorted(PROVEN_COMPOSE_VERSIONS))} only"
        )
    if api_version not in PROVEN_API_VERSIONS:
        raise Fallback(
            f"Engine API {api_version}: the direct path is proven on "
            f"{', '.join(sorted(PROVEN_API_VERSIONS))} only"
        )
    if daemon_api_version not in PROVEN_DAEMON_API_VERSIONS:
        raise Fallback(
            f"a daemon speaking API {daemon_api_version}: the direct path is "
            f"proven on {', '.join(sorted(PROVEN_DAEMON_API_VERSIONS))} only"
        )


def enabled(environ: dict[str, str] | None = None) -> bool:
    value = (os.environ if environ is None else environ).get(SETTING_ENV, "")
    return value.strip().lower() not in ("0", "false", "no", "off")


# ---------------------------------------------------------------------------
# The endpoint


def endpoint_socket(docker_args: list[str] | tuple[str, ...]) -> str:
    """The Unix socket the ``docker`` CLI reaches with *docker_args* (the
    ``docker <flags> compose`` flags of :func:`~benchflow.task.compose_env.
    docker_connection_args`), or :class:`Fallback`.

    Only two spellings are resolved, both exactly as the CLI does: ``-H
    unix://<path>`` (DOCKER_HOST) and no host at all, with no current context
    in the client configuration (the CLI's ``/var/run/docker.sock``). A
    context, TLS, TCP or SSH endpoint goes to Compose.
    """
    config: str | None = None
    host: str | None = None
    args = iter(docker_args)
    for flag in args:
        value = next(args, None)
        if value is None:
            raise Fallback(f"docker flag {flag} has no value")
        if flag == "--config":
            config = value
        elif flag in ("-H", "--host"):
            if host is not None:
                raise Fallback("more than one Docker host")
            host = value
        else:
            raise Fallback(f"Docker endpoint flag {flag}")
    if host is not None:
        if not host.startswith("unix://") or len(host) <= len("unix://"):
            raise Fallback("DOCKER_HOST is not a local Unix socket")
        return host.removeprefix("unix://")
    folder = Path(config) if config else Path.home() / ".docker"
    try:
        data = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        data = {}
    except (OSError, ValueError) as exc:
        raise Fallback(f"cannot read the Docker client configuration: {exc}") from None
    if not isinstance(data, dict):
        raise Fallback("the Docker client configuration is not an object")
    if data.get("currentContext") not in (None, "", "default"):
        raise Fallback("a Docker context is selected")
    return "/var/run/docker.sock"


# ---------------------------------------------------------------------------
# Compose's hashes


def go_json(value: Any) -> str:
    """``encoding/json``'s compact output for a decoded value: no spaces,
    UTF-8 as is, ``<``, ``>``, ``&``, U+2028 and U+2029 escaped."""
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return (
        text.replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


#: compose-go v2.9's ServiceConfig fields in declaration order (their JSON
#: names): Go marshals a struct in this order, so a merged service is put
#: back in it before it is hashed.
SERVICE_FIELD_ORDER = (
    "profiles",
    "annotations",
    "attach",
    "build",
    "develop",
    "blkio_config",
    "cap_add",
    "cap_drop",
    "cgroup_parent",
    "cgroup",
    "cpu_count",
    "cpu_percent",
    "cpu_period",
    "cpu_quota",
    "cpu_rt_period",
    "cpu_rt_runtime",
    "cpus",
    "cpuset",
    "cpu_shares",
    "command",
    "configs",
    "container_name",
    "credential_spec",
    "depends_on",
    "deploy",
    "device_cgroup_rules",
    "devices",
    "dns",
    "dns_opt",
    "dns_search",
    "dockerfile",
    "domainname",
    "entrypoint",
    "provider",
    "environment",
    "env_file",
    "expose",
    "extends",
    "external_links",
    "extra_hosts",
    "group_add",
    "gpus",
    "hostname",
    "healthcheck",
    "image",
    "init",
    "ipc",
    "isolation",
    "labels",
    "label_file",
    "links",
    "logging",
    "log_driver",
    "log_opt",
    "mem_limit",
    "mem_reservation",
    "memswap_limit",
    "mem_swappiness",
    "mac_address",
    "models",
    "net",
    "network_mode",
    "networks",
    "oom_kill_disable",
    "oom_score_adj",
    "pid",
    "pids_limit",
    "platform",
    "ports",
    "privileged",
    "pull_policy",
    "read_only",
    "restart",
    "runtime",
    "scale",
    "secrets",
    "security_opt",
    "shm_size",
    "stdin_open",
    "stop_grace_period",
    "stop_signal",
    "storage_opt",
    "sysctls",
    "tmpfs",
    "tty",
    "ulimits",
    "use_api_socket",
    "user",
    "userns_mode",
    "uts",
    "volume_driver",
    "volumes",
    "volumes_from",
    "working_dir",
    "post_start",
    "pre_stop",
)
_FIELD_RANK = {name: rank for rank, name in enumerate(SERVICE_FIELD_ORDER)}


def ordered_service(service: dict) -> dict:
    unknown = [key for key in service if key not in _FIELD_RANK]
    if unknown:
        raise Fallback(f"service key {unknown[0]} is not in Compose's model")
    return {key: service[key] for key in sorted(service, key=_FIELD_RANK.__getitem__)}


def service_hash(service: dict) -> str:
    """compose's ``ServiceHash``: the service without build, pull_policy,
    scale, deploy.replicas, depends_on and profiles."""
    kept = {
        key: value
        for key, value in service.items()
        if key not in ("build", "pull_policy", "scale", "depends_on", "profiles")
    }
    if isinstance(kept.get("deploy"), dict):
        kept["deploy"] = {k: v for k, v in kept["deploy"].items() if k != "replicas"}
    return hashlib.sha256(go_json(kept).encode()).hexdigest()


def network_hash(network: dict) -> str:
    """compose's ``NetworkHash``: the network as its model holds it."""
    return hashlib.sha256(go_json(network).encode()).hexdigest()


# ---------------------------------------------------------------------------
# What the path reproduces

#: Service keys this path renders exactly; anything else goes to Compose.
_SERVICE_KEYS = frozenset(
    {
        "build",
        "cap_add",
        "cap_drop",
        "cgroup",
        "command",
        "deploy",
        "domainname",
        "entrypoint",
        "environment",
        "group_add",
        "healthcheck",
        "hostname",
        "image",
        "init",
        "labels",
        "network_mode",
        "networks",
        "pull_policy",
        "read_only",
        "restart",
        "security_opt",
        "shm_size",
        "stdin_open",
        "tty",
        "user",
        "volumes",
        "working_dir",
    }
)
_PULL_POLICIES = frozenset({"missing", "if_not_present", "never", "build"})
#: The only restart settings this path renders: the ones BenchFlow's own
#: no-restart file writes (``_compose_files/docker-compose-no-restart.yaml``),
#: as Compose prints them. Compose sends either as the policy
#: ``no`` with no retries; any other policy goes to Compose.
_NO_RESTART = "no"
_NO_RESTART_POLICY = {"condition": "none", "max_attempts": 0}
_NETWORK_KEYS = frozenset({"name", "ipam", "labels"})


def _strings(value: Any, where: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise Fallback(f"{where} is not a list of strings")
    return value


def _bytes_value(value: Any, where: str) -> int:
    if isinstance(value, bool):
        raise Fallback(f"{where} is not a size")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    raise Fallback(f"{where} is not a size in bytes")


def check_project(project: dict, project_name: str) -> None:
    """Raise :class:`Fallback` unless the path renders *project* exactly."""
    extra = set(project) - {"name", "networks", "services"}
    if extra:
        raise Fallback(f"top-level {sorted(extra)[0]}")
    if project.get("name") != project_name:
        raise Fallback("the project name differs from the sandbox's")
    services = project.get("services")
    if not isinstance(services, dict) or set(services) != {"main"}:
        raise Fallback("more than one service")
    service = services["main"]
    if not isinstance(service, dict):
        raise Fallback("main is not a mapping")
    for key in service:
        if key not in _SERVICE_KEYS:
            raise Fallback(f"service key {key}")
    networks = project.get("networks") or {}
    if not isinstance(networks, dict) or set(networks) - {"default"}:
        raise Fallback("a network other than default")
    for network in networks.values():
        if not isinstance(network, dict) or set(network) - _NETWORK_KEYS:
            raise Fallback("a network setting other than name, ipam and labels")
        ipam = network.get("ipam")
        if ipam is not None:
            if not isinstance(ipam, dict) or set(ipam) - {"config"}:
                raise Fallback("a network ipam setting other than config")
            for pool in ipam.get("config") or []:
                if not isinstance(pool, dict) or set(pool) - {"subnet"}:
                    raise Fallback("an ipam pool setting other than subnet")
    if "network_mode" in service:
        if service["network_mode"] != "none":
            raise Fallback("network_mode other than none")
        if networks or service.get("networks"):
            raise Fallback("network_mode none with networks")
    else:
        if service.get("networks") != {"default": None} or set(networks) != {"default"}:
            raise Fallback("service networks other than the default network")
        network = networks["default"]
        if network.get("name") != f"{project_name}_default":
            raise Fallback("the default network is named otherwise")
    if service.get("pull_policy", "missing") not in _PULL_POLICIES:
        raise Fallback(f"pull_policy {service.get('pull_policy')}")
    if not isinstance(service.get("image"), str) or not service["image"]:
        raise Fallback("main has no image")
    for key in ("command", "entrypoint"):
        if service.get(key) is not None:
            _strings(service[key], key)
    for key in ("cap_add", "cap_drop", "security_opt", "group_add"):
        if key in service:
            _strings(service[key], key)
    for opt in service.get("security_opt", []):
        if opt == "systempaths=unconfined" or opt.startswith(("seccomp", "apparmor")):
            raise Fallback(f"security_opt {opt.split('=')[0].split(':')[0]}")
        if "=" not in opt and ":" not in opt and opt != "no-new-privileges":
            raise Fallback("an unparsed security_opt")
    environment = service.get("environment", {})
    if not isinstance(environment, dict) or not all(
        isinstance(v, str) or v is None for v in environment.values()
    ):
        raise Fallback("environment is not a mapping of strings")
    labels = service.get("labels", {})
    if not isinstance(labels, dict) or not all(
        isinstance(v, str) for v in labels.values()
    ):
        raise Fallback("labels are not a mapping of strings")
    if any(key.startswith("com.docker.compose.") for key in labels):
        raise Fallback("the task sets a com.docker.compose label")
    healthcheck = service.get("healthcheck")
    if healthcheck is not None and healthcheck != {"disable": True}:
        raise Fallback("a healthcheck")
    if service.get("cgroup") not in (None, "private", "host"):
        raise Fallback("cgroup other than private or host")
    for key in ("tty", "stdin_open", "init", "read_only"):
        if key in service and not isinstance(service[key], bool):
            raise Fallback(f"{key} is not a boolean")
    for key in ("user", "working_dir", "hostname", "domainname"):
        if key in service and not isinstance(service[key], str):
            raise Fallback(f"{key} is not a string")
    if "shm_size" in service:
        _bytes_value(service["shm_size"], "shm_size")
    if service.get("restart", _NO_RESTART) != _NO_RESTART:
        raise Fallback("a restart policy other than no")
    _check_deploy(service.get("deploy"))
    _check_volumes(service.get("volumes", []))


def _check_deploy(deploy: Any) -> None:
    if deploy is None:
        return
    if not isinstance(deploy, dict) or set(deploy) - {
        "resources",
        "placement",
        "restart_policy",
    }:
        raise Fallback("a deploy setting other than resources")
    if deploy.get("placement") not in (None, {}):
        raise Fallback("deploy.placement")
    if deploy.get("restart_policy", _NO_RESTART_POLICY) != _NO_RESTART_POLICY:
        raise Fallback("a deploy.restart_policy other than none")
    resources = deploy.get("resources") or {}
    if not isinstance(resources, dict) or set(resources) - {"limits", "reservations"}:
        raise Fallback("a deploy.resources setting other than limits")
    limits = resources.get("limits") or {}
    if not isinstance(limits, dict) or set(limits) - {"cpus", "memory", "pids"}:
        raise Fallback("a deploy limit other than cpus, memory and pids")
    reservations = resources.get("reservations") or {}
    if not isinstance(reservations, dict) or set(reservations) - {"memory"}:
        raise Fallback("a deploy reservation other than memory")
    if "cpus" in limits and (
        isinstance(limits["cpus"], bool) or not isinstance(limits["cpus"], int | float)
    ):
        raise Fallback("deploy cpus is not a number")
    for where, value in (
        ("memory", limits.get("memory")),
        ("reserved memory", reservations.get("memory")),
    ):
        if value is not None:
            _bytes_value(value, where)
    if "pids" in limits and (
        isinstance(limits["pids"], bool) or not isinstance(limits["pids"], int)
    ):
        raise Fallback("deploy pids is not a number")


def _check_volumes(volumes: Any) -> None:
    if not isinstance(volumes, list):
        raise Fallback("volumes is not a list")
    targets = []
    for volume in volumes:
        if not isinstance(volume, dict) or volume.get("type") != "bind":
            raise Fallback("a volume other than a bind")
        if set(volume) - {"type", "source", "target", "read_only", "bind"}:
            raise Fallback("a bind option Compose mounts through the Mount API")
        bind = volume.get("bind")
        if bind is not None and bind != {"create_host_path": True}:
            raise Fallback("a bind option Compose mounts through the Mount API")
        source, target = volume.get("source"), volume.get("target")
        if not (isinstance(source, str) and source.startswith("/")):
            raise Fallback("a bind source that is not absolute")
        if not (isinstance(target, str) and target.startswith("/")):
            raise Fallback("a bind target that is not absolute")
        if not isinstance(volume.get("read_only", False), bool):
            raise Fallback("read_only is not a boolean")
        targets.append(target)
    if len(set(targets)) != len(targets):
        # Compose's order of duplicate destinations is its
        # map's; it cannot be matched.
        raise Fallback("two binds share a destination")


def merge_bootstrap(service: dict, overlay: dict) -> dict:
    """Compose's merge of the inert-bootstrap overlay into *service*
    (``_inert_bootstrap.prepare``'s keys only): entrypoint and command
    replace, environment and labels merge key by key (a null removes the
    value), healthcheck replaces, an empty depends_on adds nothing."""
    allowed = {
        "entrypoint",
        "command",
        "environment",
        "healthcheck",
        "labels",
        "depends_on",
    }
    if set(overlay) - allowed:
        raise Fallback("the bootstrap overlay has a key this path does not merge")
    merged = dict(service)
    for key in ("entrypoint", "command"):
        if key in overlay:
            merged[key] = list(overlay[key])
    for key in ("environment", "labels"):
        if key in overlay:
            values = dict(merged.get(key) or {})
            values.update(overlay[key])
            merged[key] = dict(sorted(values.items()))
    if "healthcheck" in overlay:
        merged["healthcheck"] = dict(overlay["healthcheck"])
    if overlay.get("depends_on"):
        raise Fallback("the bootstrap overlay has dependencies")
    return ordered_service(merged)


# ---------------------------------------------------------------------------
# Request bodies, as Compose 2.40.3 (Docker client v28.5) sends them


def _float32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


def nano_cpus(cpus: float) -> int:
    """``int64(limits.NanoCPUs * 1e9)`` with compose-go's float32."""
    return int(_float32(_float32(cpus) * 1e9))


def network_body(project: dict, key: str, compose_version: str) -> dict:
    """The NetworkCreate body Compose sends for the project's network *key*.

    Compose hashes the network as its model prints it (its own labels are
    not part of that JSON), then adds its labels and the hash."""
    network = project["networks"][key]
    labels = {
        **(network.get("labels") or {}),
        "com.docker.compose.network": key,
        COMPOSE_PROJECT: project["name"],
        "com.docker.compose.version": compose_version,
        "com.docker.compose.config-hash": network_hash(network),
    }
    ipam = network.get("ipam") or {}
    return {
        "Name": network["name"],
        "Driver": "",
        "Scope": "",
        "IPAM": {
            "Driver": "",
            "Options": None,
            "Config": [{"Subnet": pool["subnet"]} for pool in ipam["config"]],
        }
        if ipam.get("config")
        else None,
        "Internal": False,
        "Attachable": False,
        "Ingress": False,
        "ConfigOnly": False,
        "ConfigFrom": None,
        "Options": None,
        "Labels": labels,
    }


def capabilities(caps: list[str] | None) -> list[str] | None:
    """The Docker client's ``normalizeCapabilities`` (v28.5), which runs on
    every ContainerCreate: upper case, ``CAP_`` prefixed (not ``ALL``),
    without repeats, sorted; nothing as null."""
    out: list[str] = []
    for cap in caps or []:
        cap = cap.upper()
        if cap != "ALL" and not cap.startswith("CAP_"):
            cap = "CAP_" + cap
        if cap not in out:
            out.append(cap)
    return sorted(out) or None


def _env_list(environment: dict) -> list[str] | None:
    if not environment:
        return None
    return [k if v is None else f"{k}={v}" for k, v in sorted(environment.items())]


def _bind_string(volume: dict) -> str:
    access = "ro" if volume.get("read_only") else "rw"
    return f"{volume['source']}:{volume['target']}:{access}"


def container_body(
    project: dict,
    service: dict,
    *,
    image_id: str,
    compose_version: str,
    config_files: list[str],
    working_dir: str,
    replace: bool,
) -> dict:
    """The ContainerCreate body Compose sends for ``main`` (number 1): the
    prototype's (``replace=False``, from ``compose create``) or the
    recreated container's (``replace=True``, from ``compose up``)."""
    name = project["name"]
    container_name = f"{name}-main-1"
    labels = dict(service.get("labels") or {})
    labels.update(
        {
            COMPOSE_PROJECT: name,
            COMPOSE_SERVICE: "main",
            "com.docker.compose.version": compose_version,
            COMPOSE_ONEOFF: "False",
            "com.docker.compose.project.working_dir": working_dir,
            "com.docker.compose.project.config_files": ",".join(config_files),
            "com.docker.compose.image": image_id,
            "com.docker.compose.config-hash": service_hash(service),
            "com.docker.compose.container-number": "1",
            "com.docker.compose.depends_on": "",
        }
    )
    if replace:
        labels["com.docker.compose.replace"] = "main-1"
    deploy = service.get("deploy") or {}
    limits = (deploy.get("resources") or {}).get("limits") or {}
    reservations = (deploy.get("resources") or {}).get("reservations") or {}
    memory = _bytes_value(limits["memory"], "memory") if "memory" in limits else 0
    reserved = (
        _bytes_value(reservations["memory"], "memory")
        if "memory" in reservations
        else 0
    )
    cpus = nano_cpus(float(limits["cpus"])) if limits.get("cpus") else 0
    network_mode = service.get("network_mode")
    if network_mode is None:
        network_mode = project["networks"]["default"]["name"]
        networking: dict | None = {
            "EndpointsConfig": {
                network_mode: {
                    "IPAMConfig": None,
                    "Links": None,
                    "Aliases": [container_name, "main"],
                    "DriverOpts": None,
                    "GwPriority": 0,
                    "NetworkID": "",
                    "EndpointID": "",
                    "Gateway": "",
                    "IPAddress": "",
                    "MacAddress": "",
                    "IPPrefixLen": 0,
                    "IPv6Gateway": "",
                    "GlobalIPv6Address": "",
                    "GlobalIPv6PrefixLen": 0,
                    "DNSNames": None,
                }
            }
        }
    else:
        networking = None
    cgroup = service.get("cgroup", "")
    # check_project admits only the no-restart file's settings (_NO_RESTART).
    restart = _NO_RESTART if "restart" in service or "restart_policy" in deploy else ""
    host: dict[str, Any] = {
        "Binds": [_bind_string(v) for v in service.get("volumes", [])] or None,
        "ContainerIDFile": "",
        "LogConfig": {"Type": "", "Config": None},
        "NetworkMode": network_mode,
        "PortBindings": {},
        "RestartPolicy": {"Name": restart, "MaximumRetryCount": 0},
        "AutoRemove": False,
        "VolumeDriver": "",
        "VolumesFrom": None,
        "ConsoleSize": [0, 0],
        "CapAdd": capabilities(service.get("cap_add")),
        "CapDrop": capabilities(service.get("cap_drop")),
        "CgroupnsMode": cgroup,
        "Dns": None,
        "DnsOptions": None,
        "DnsSearch": None,
        "ExtraHosts": [],
        "GroupAdd": service.get("group_add"),
        "IpcMode": "",
        "Cgroup": "",
        "Links": None,
        "OomScoreAdj": 0,
        "PidMode": "",
        "Privileged": False,
        "PublishAllPorts": False,
        "ReadonlyRootfs": service.get("read_only", False),
        "SecurityOpt": service.get("security_opt"),
        "UTSMode": "",
        "UsernsMode": "",
        "ShmSize": _bytes_value(service["shm_size"], "shm_size")
        if "shm_size" in service
        else 0,
        "Isolation": "",
        "CpuShares": 0,
        "Memory": memory,
        "NanoCpus": cpus,
        "CgroupParent": "",
        "BlkioWeight": 0,
        "BlkioWeightDevice": None,
        "BlkioDeviceReadBps": None,
        "BlkioDeviceWriteBps": None,
        "BlkioDeviceReadIOps": None,
        "BlkioDeviceWriteIOps": None,
        "CpuPeriod": 0,
        "CpuQuota": 0,
        "CpuRealtimePeriod": 0,
        "CpuRealtimeRuntime": 0,
        "CpusetCpus": "",
        "CpusetMems": "",
        "Devices": None,
        "DeviceCgroupRules": None,
        "DeviceRequests": None,
        "MemoryReservation": reserved,
        "MemorySwap": 0,
        "MemorySwappiness": None,
        "OomKillDisable": False,
        "PidsLimit": limits["pids"] if limits.get("pids") else None,
        "Ulimits": None,
        "CpuCount": 0,
        "CpuPercent": 0,
        "IOMaximumIOps": 0,
        "IOMaximumBandwidth": 0,
        "MaskedPaths": None,
        "ReadonlyPaths": None,
    }
    if "init" in service:
        host["Init"] = service["init"]
    body: dict[str, Any] = {
        "Hostname": service.get("hostname", ""),
        "Domainname": service.get("domainname", ""),
        "User": service.get("user", ""),
        "AttachStdin": False,
        "AttachStdout": True,
        "AttachStderr": True,
        "Tty": service.get("tty", False),
        "OpenStdin": service.get("stdin_open", False),
        "StdinOnce": False,
        "Env": _env_list(service.get("environment") or {}),
        "Cmd": service.get("command"),
        "Image": service["image"],
        "Volumes": None,
        "WorkingDir": service.get("working_dir", ""),
        "Entrypoint": service.get("entrypoint"),
        "OnBuild": None,
        "Labels": labels,
        "HostConfig": host,
    }
    # CreateRequest's NetworkingConfig is omitempty: network_mode none sends none.
    if networking is not None:
        body["NetworkingConfig"] = networking
    if service.get("healthcheck") == {"disable": True}:
        body["Healthcheck"] = {"Test": ["NONE"]}
    return body


# ---------------------------------------------------------------------------
# The published normalisation


def _empty(value: Any) -> bool:
    return value is None or value == [] or value == {}


def _env_key(row: Any) -> Any:
    return row.partition("=")[0] if isinstance(row, str) else row


def _as_multiset(rows: list, key: Any) -> list:
    """Sorted when no two rows share *key*; otherwise as given."""
    keys = [key(row) for row in rows]
    if len(set(map(_hashable, keys))) != len(keys):
        return rows
    return sorted(rows, key=lambda row: json.dumps(row, sort_keys=True))


def _hashable(value: Any) -> Any:
    return (
        json.dumps(value, sort_keys=True) if isinstance(value, dict | list) else value
    )


def _bind_destination(row: Any) -> Any:
    if not isinstance(row, str):
        return row
    parts = row.split(":")
    return parts[1] if len(parts) >= 2 else row


def _mount_destination(row: Any) -> Any:
    if isinstance(row, dict):
        return row.get("Destination", row.get("Target"))
    return row


def _collapse_empty(value: Any) -> Any:
    """Rule (a): null, [] and {} are one value, and only when empty."""
    if isinstance(value, dict):
        out = {k: _collapse_empty(v) for k, v in value.items()}
        return {k: (None if _empty(v) else v) for k, v in out.items()}
    if isinstance(value, list):
        return [_collapse_empty(v) for v in value]
    return value


def order_rules(body: dict) -> dict:
    """The order rules alone, for a request body (layer 1): ``Env``
    is a multiset unless a key repeats (D965 (2), (3)); ``Binds`` and
    ``Mounts`` (in ``HostConfig``, and inspect's top-level ``Mounts``) are
    multisets unless a destination repeats (D965 (1))."""
    data = json.loads(json.dumps(body))
    config = data.get("Config") if "Config" in data else data
    if isinstance(config, dict) and isinstance(config.get("Env"), list):
        config["Env"] = _as_multiset(config["Env"], _env_key)
    host = data.get("HostConfig")
    if isinstance(host, dict):
        if isinstance(host.get("Binds"), list):
            host["Binds"] = _as_multiset(host["Binds"], _bind_destination)
        if isinstance(host.get("Mounts"), list):
            host["Mounts"] = _as_multiset(host["Mounts"], _mount_destination)
    if isinstance(data.get("Mounts"), list):
        data["Mounts"] = _as_multiset(data["Mounts"], _mount_destination)
    return data


def normalise_inspect(container: dict) -> dict:
    """A container's inspect under the three normalisation rules and the
    order rules, nothing else:

    (a) null, ``[]`` and ``{}`` are the same empty value;
    (b) ``Config.Env`` is a multiset, unless a key repeats, then it keeps
        its order (D965 (3));
    (c) ``Config.Entrypoint`` and ``Config.Cmd`` count only through what
        Docker runs: they are joined into one argv, and ``Path`` and
        ``Args`` stay as they are (compared exactly);
    D965 (1) ``HostConfig.Binds``, ``HostConfig.Mounts`` and the top-level
        ``Mounts`` are multisets, unless a destination repeats.

    Masking of names, IDs and per-trial label values is the comparer's,
    not this function's."""
    data = order_rules(container)
    config = data.get("Config")
    if isinstance(config, dict) and "Path" in data and "Args" in data:
        argv = list(config.get("Entrypoint") or []) + list(config.get("Cmd") or [])
        config.pop("Entrypoint", None)
        config.pop("Cmd", None)
        config["Entrypoint+Cmd"] = argv
    return _collapse_empty(data)


def differences(a: Any, b: Any, path: str = "") -> list[str]:
    """Every path where two normalised values differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                out.append(f"{path}/{key}: {'missing' if key not in a else 'extra'}")
            else:
                out += differences(a[key], b[key], f"{path}/{key}")
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [
            d
            for i, (x, y) in enumerate(zip(a, b, strict=True))
            for d in differences(x, y, f"{path}[{i}]")
        ]
    return [] if a == b else [f"{path}: {json.dumps(a)[:120]} != {json.dumps(b)[:120]}"]


def multiset_counts(rows: list) -> Counter:
    return Counter(_hashable(row) for row in rows)


_SHORT_ID = re.compile(r"^[0-9a-f]{12,64}$")


# ---------------------------------------------------------------------------
# The calls


def daemon_message(exc: BaseException) -> str:
    """Compose's wording for a daemon error, which BenchFlow's retry and
    subnet rules read (``Error response from daemon: ...``)."""
    from benchflow.sandbox._engine_api import EngineError

    if isinstance(exc, EngineError):
        return f"Error response from daemon: {exc.message}"
    return str(exc)


class DirectProject:
    """One trial's ``main`` container and network, made through the Engine
    API from the project Compose resolved (:func:`check_project` passed)."""

    def __init__(
        self,
        client: Any,
        project: dict,
        *,
        image_id: str,
        compose_version: str,
        working_dir: str,
    ) -> None:
        self.client = client
        self.project = project
        self.name = project["name"]
        self.image_id = image_id
        self.compose_version = compose_version
        self.working_dir = working_dir
        self.container: str | None = None
        # The versions the plan was checked against, for config.json.
        self.versions: dict[str, str] = {}

    @property
    def container_name(self) -> str:
        return f"{self.name}-main-1"

    def _own(self, row: Any) -> bool:
        """A listed container or network is this project's: a well-formed ID
        and the project's label on the row itself, whatever filter the daemon
        (or anything between) applied (PERF-E-B-L3)."""
        if not isinstance(row, dict):
            return False
        cid = row.get("Id")
        labels = row.get("Labels") or {}
        return (
            isinstance(cid, str)
            and _SHORT_ID.fullmatch(cid) is not None
            and isinstance(labels, dict)
            and labels.get(COMPOSE_PROJECT) == self.name
        )

    async def _call(self, method: str, path: str, **kwargs: Any) -> Any:
        """A start-time call: no deadline, as Compose's ``up`` has none (the
        rollout's own deadline bounds the start). Teardown's calls keep the
        client's 60 s, inside stop()'s 90 s (PERF-E-B-L4, B-R3-L3)."""
        kwargs.setdefault("timeout", UNBOUNDED)
        return await self.client.call(method, path, **kwargs)

    def _filters(self, **labels: str) -> dict:
        return {"label": {f"{k}={v}": True for k, v in labels.items()}}

    async def ensure_network(self) -> None:
        """Compose's ``ensureNetwork`` for a network this project owns: the
        name must be free (the project name is unique to the trial)."""
        from benchflow.sandbox._engine_api import EngineError

        for key in self.project.get("networks") or {}:
            body = network_body(self.project, key, self.compose_version)
            try:
                await self._call("GET", f"/networks/{self.client.quote(body['Name'])}")
            except EngineError as exc:
                if exc.status != 404:
                    raise
            else:
                raise RuntimeError(
                    f"network {body['Name']} already exists; start a fresh trial"
                )
            try:
                await self._call("POST", "/networks/create", body=body)
            except EngineError as exc:
                raise RuntimeError(
                    f"failed to create network {body['Name']}: {daemon_message(exc)}"
                ) from exc

    async def create(
        self, service: dict, config_files: list[str], *, replace: bool
    ) -> str:
        body = container_body(
            self.project,
            service,
            image_id=self.image_id,
            compose_version=self.compose_version,
            config_files=config_files,
            working_dir=self.working_dir,
            replace=replace,
        )
        from benchflow.sandbox._engine_api import EngineError

        try:
            reply = await self._call(
                "POST",
                "/containers/create",
                query={"name": self.container_name},
                timeout=UNBOUNDED,
                body=body,
            )
        except EngineError as exc:
            raise RuntimeError(daemon_message(exc)) from exc
        cid = (reply.json() or {}).get("Id")
        if not isinstance(cid, str) or not _SHORT_ID.fullmatch(cid):
            raise RuntimeError("Docker did not name the created container")
        self.container = cid
        return cid

    async def remove(
        self,
        cid: str,
        *,
        force: bool = False,
        volumes: bool = False,
        timeout: float = UNBOUNDED,
    ) -> None:
        query = {}
        if force:
            query["force"] = "1"
        if volumes:
            query["v"] = "1"
        await self._call(
            "DELETE", f"/containers/{cid}", query=query or None, timeout=timeout
        )

    async def start(self, cid: str) -> None:
        from benchflow.sandbox._engine_api import EngineError

        try:
            await self._call("POST", f"/containers/{cid}/start", timeout=UNBOUNDED)
        except EngineError as exc:
            raise RuntimeError(daemon_message(exc)) from exc

    async def inspect(self, cid: str) -> dict:
        reply = await self._call("GET", f"/containers/{self.client.quote(cid)}/json")
        data = reply.json()
        if not isinstance(data, dict):
            raise RuntimeError(f"docker inspect {cid!r} returned no container object")
        return data

    async def container_ids(self, *, running_only: bool = False) -> list[str]:
        """``compose ps [--all] -q``: the project's service containers."""
        query: dict[str, Any] = {
            "filters": self._filters(
                **{COMPOSE_PROJECT: self.name, COMPOSE_ONEOFF: "False"}
            )
        }
        if not running_only:
            query["all"] = "1"
        rows = (await self._call("GET", "/containers/json", query=query)).json()
        if not isinstance(rows, list):
            raise RuntimeError("Cannot identify Compose containers; retry startup")
        if not all(
            self._own(row) and (row.get("Labels") or {}).get(COMPOSE_ONEOFF) == "False"
            for row in rows
        ):
            raise RuntimeError("Cannot identify Compose containers; retry startup")
        return [row["Id"] for row in rows]

    async def teardown(self, *, volumes: bool) -> None:
        """``compose kill``, ``rm --force [--volumes]`` and ``down
        --remove-orphans -t 0``: every container of the project (by its
        Compose label, one-off and orphans included) is killed and removed,
        then every network carrying the project's label."""
        from benchflow.sandbox._engine_api import EngineError

        rows = (
            await self.client.call(
                "GET",
                "/containers/json",
                query={
                    "all": "1",
                    "filters": self._filters(**{COMPOSE_PROJECT: self.name}),
                },
            )
        ).json()
        if not isinstance(rows, list):
            # Never "nothing to remove": stop() falls back to its force-kill
            # by label (B-R3-L4).
            raise RuntimeError(
                "Docker's container list was not a list; cannot tear down"
            )
        rows = [row for row in rows if self._own(row)]
        ids = [row["Id"] for row in rows]
        # One container (every plain task): its forced removal is the kill
        # (SIGKILL) and the removal in one call, ``docker rm -f``; a second
        # call costs the daemon's wait for the exit twice (PERF-SBX). Several
        # are all killed first, as Compose's kill does.
        for row in rows if len(rows) > 1 else ():
            if row.get("State") == "running":
                try:
                    await self.client.call(
                        "POST",
                        f"/containers/{row['Id']}/kill",
                        query={"signal": "SIGKILL"},
                    )
                except EngineError as exc:
                    if exc.status not in (404, 409):
                        raise
        for cid in ids:
            try:
                await self.remove(cid, force=True, volumes=volumes, timeout=60.0)
            except EngineError as exc:
                if exc.status != 404:
                    raise
        networks = (
            await self.client.call(
                "GET",
                "/networks",
                query={"filters": self._filters(**{COMPOSE_PROJECT: self.name})},
            )
        ).json()
        if not isinstance(networks, list):
            raise RuntimeError("Docker's network list was not a list; cannot tear down")
        for network in networks:
            if not self._own(network):
                continue
            try:
                await self.client.call("DELETE", f"/networks/{network['Id']}")
            except EngineError as exc:
                if exc.status != 404:
                    raise
        self.container = None
