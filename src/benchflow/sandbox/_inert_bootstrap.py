"""S1-CONF-9 (694245fd): no task process runs before the bind receipt matches.

Compose creates stopped prototypes so Docker, rather than a second interpreter
of Compose/image defaults, supplies the effective argv, environment and user.
Containers with host binds, and their namespace/volume peers, start a fixed
system shell waiting on a signal. Only the trusted controller releases them,
after proving the entire project's binds. Unrelated services keep their native
startup and need no shell or injected tools.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shlex
from typing import Any

from benchflow.sandbox import _bind_targets
from benchflow.sandbox._base import (
    IMAGE_UNRESTORED_VARS,
    SYSTEM_SHELL_UNSET_VARS,
    StartupNamesRefused,
    refused_startup_names,
    startup_refusal,
    system_shell_route,
)
from benchflow.sandbox._compose import compose_container_ids
from benchflow.sandbox._exec_output import exec_raw

# No PATH lookup, startup file or inherited entrypoint occurs before release.
# The waiting loop invokes only the absolute trusted /bin/sleep. Docker supplies
# empty loader/startup variables before /bin/sh's
# loader starts; the real environment is restored only after the USR1 barrier.
BOOTSTRAP = (
    "unset " + " ".join(SYSTEM_SHELL_UNSET_VARS) + "; "
    "released=; trap 'released=1' USR1; "
    'while [ -z "$released" ]; do /bin/sleep 0.05 & wait $!; done; '
    'trap - USR1; exec /usr/bin/env -- "$@"'
)


def restore_bootstrap(kit: bool) -> str:
    """S1's release barrier on PT's protected runtime (b165 integration).

    A checkpoint never runs the image's entrypoint or restores loader inputs
    into root's PID 1. Task commands recover their environment separately.
    """
    from benchflow._runtime_tree import KIT_BIN

    sleep = f"{KIT_BIN}/sleep" if kit else "/bin/sleep"
    return (
        "unset " + " ".join(SYSTEM_SHELL_UNSET_VARS) + "; "
        "released=; trap 'released=1' USR1; "
        f'while [ -z "$released" ]; do {sleep} 0.05 & wait $!; done; '
        'trap - USR1; exec "$@"'
    )


LABEL = "io.benchflow.bootstrap.v1"
# A bare Docker env key / null Compose value removes an image variable when
# it is absent from the client's environment. Empty strings are insufficient:
# glibc acts on the mere presence of LD_SHOW_AUXV, for example.
EMPTY_ENV = dict.fromkeys(SYSTEM_SHELL_UNSET_VARS)


def client_env(environment: dict[str, str] | None) -> dict[str, str]:
    return {
        k: v
        for k, v in (os.environ if environment is None else environment).items()
        if k not in EMPTY_ENV
    }


def original_config(config: dict) -> dict:
    """Only daemon-owned metadata may recover the environment cleared at create."""
    receipt = (config.get("Labels") or {}).get(LABEL)
    if receipt is None:
        return config
    try:
        original = json.loads(base64.b64decode(receipt, validate=True))
        if not isinstance(original, dict):
            raise ValueError("not an object")
        return {**config, **original}
    except (ValueError, TypeError) as exc:
        raise RuntimeError(
            "Checkpoint lacks its original command; start a fresh trial"
        ) from exc


def canonical_env(rows: list) -> list:
    """Env rows in one order whatever order Docker was given them in
    sorted by key, stably, so rows that share a key keep their
    order and ``env --`` (which assigns left to right) gives the same
    result. Compose builds a service's Env from a Go map, so its order
    changes between identical runs; this makes main's argv and receipt the
    same on every run and on the direct Docker path."""
    return sorted(rows, key=lambda row: row.partition("=")[0])


def config_receipt(config: dict) -> str:
    original = {
        key: config.get(key)
        for key in ("Entrypoint", "Cmd", "Env", "Healthcheck", "Shell")
    }
    if isinstance(original["Env"], list) and all(
        isinstance(row, str) for row in original["Env"]
    ):
        original["Env"] = canonical_env(original["Env"])
    return base64.b64encode(json.dumps(original).encode()).decode()


# BenchFlow's keep-alive (the compose files' command), and the same as PID 1
# itself.
KEEP_ALIVE = ["sh", "-c", "sleep infinity"]
KEEP_ALIVE_AS_PID_1 = ["sh", "-c", "exec sleep infinity"]


def _root_user(user: Any) -> bool:
    """Whether a Config.User selector names root (no selector included)."""
    name = str(user or "").partition(":")[0]
    return name in ("", "root") or (
        name.isascii() and name.isdecimal() and int(name) == 0
    )


def entrypoint(config: dict) -> list[str]:
    argv = (config.get("Entrypoint") or []) + (config.get("Cmd") or [])
    environment = config.get("Env") or []
    if (
        not argv
        or not all(isinstance(arg, str) and "\0" not in arg for arg in argv)
        or not isinstance(environment, list)
        or not all(isinstance(row, str) and "=" in row for row in environment)
    ):
        raise RuntimeError("Cannot bootstrap task command; use an exec-form command")
    if argv == KEEP_ALIVE and not _root_user(config.get("User")):
        # PID 1 runs as the image's user, which may be the sandbox user, whose
        # processes the kill before grading ends. A shell PID 1 would exit with
        # its sleep, and the container with it; the kill leaves a sleep PID 1
        # alone (lockdown._KILL_PRELUDE). A root PID 1 stays the shell, which
        # reaps the orphans the kernel gives it; a sleep reaps none.
        argv = KEEP_ALIVE_AS_PID_1
    return [
        "/bin/sh",
        "-c",
        BOOTSTRAP,
        "benchflow-bootstrap",
        *canonical_env(environment),
        *argv,
    ]


def run_args(*, shell: str = "/bin/sh") -> list[str]:
    return [
        "--entrypoint",
        shell,
        "--no-healthcheck",
        *(arg for key in EMPTY_ENV for arg in ("--env", key)),
    ]


async def prepare(env: Any, compose: Any) -> dict:
    """Inspect only stopped containers, then produce a private runtime overlay."""
    # A restored project must not reuse a previously released container.
    # Creating stopped prototypes never runs an image command or healthcheck.
    created = await compose(["create", "--force-recreate", "--no-build"])
    if created.return_code:
        raise RuntimeError("Cannot create inert prototypes; retry startup")
    project = json.loads(
        (await exec_raw(compose, ["config", "--format", "json"])).stdout
    )
    ids = await compose_container_ids(compose)
    return await overlay(env, project, ids)


async def overlay(env: Any, project: dict, ids: list[str]) -> dict:
    """The runtime overlay for the stopped prototypes *ids* of *project*
    (Compose's printed model). Shared by Compose's start and the direct
    Docker path (PERF-E), which makes its prototype through the Engine API."""
    if not ids:
        raise RuntimeError("Cannot identify inert prototypes; retry startup")
    configs = {}
    prototypes = {}
    services = {}
    for cid in ids:
        inspected = await env._inspect_container(cid)
        if (inspected.get("State") or {}).get("Running") is not False:
            raise RuntimeError(
                "Bootstrap prototype is not stopped; start a fresh trial"
            )
        config = inspected["Config"]
        name = (config.get("Labels") or {}).get("com.docker.compose.service")
        if name not in project["services"] or name in configs:
            raise RuntimeError("Cannot identify bootstrap service; use one replica")
        # A checkpoint retains our inert entrypoint. Its trusted label carries
        # the original effective process, without recursively wrapping BOOTSTRAP.
        config = original_config(config)
        # Docker gives every exec the container's Config.Env (the image's
        # ENV and the service's environment); bash's start-up names there
        # would load into root's bash before any command (startup_name).
        refused = refused_startup_names(config.get("Env"))
        if refused:
            raise StartupNamesRefused(
                startup_refusal(refused, f"the {name} service's image or environment")
            )
        configs[name] = config
        prototypes[name] = inspected
    if "main" not in configs:
        raise RuntimeError("Bootstrap has no main service; fix the task services")
    guarded = await protected_services(env, project, prototypes)
    for name in sorted(guarded):
        config = configs[name]
        service = project["services"][name]
        if service.get("post_start") or service.get("pre_stop"):
            raise RuntimeError(
                "Compose lifecycle hooks cannot wait for bind verification; "
                "move them into the task entrypoint before retrying"
            )
        services[name] = {
            "entrypoint": entrypoint(config),
            "command": [],
            "environment": EMPTY_ENV,
            "healthcheck": {"disable": True},
            "labels": {LABEL: config_receipt(config)},
            # Readiness dependencies are enforced after every bind is proved.
            "depends_on": {
                dep: {"condition": "service_started"}
                for dep in project["services"][name].get("depends_on", {})
            },
        }
    env._bootstrap_configs = configs
    env._bootstrap_project = project
    env._bootstrap_services = guarded
    return {"services": services}


async def protected_services(env: Any, project: dict, prototypes: dict) -> set[str]:
    """Close the host-bind set over shared namespaces and daemon volume names.

    Edges are undirected: either end of a namespace or volume relationship can
    affect the other. Use inspected volumes so image VOLUMEs and volumes_from
    aliases are included, not just the task's explicit volume declarations.
    """
    guarded = set()
    peers = {name: set() for name in prototypes}
    volumes: dict[str, set[str]] = {}
    for name, inspected in prototypes.items():
        if await _bind_targets.host_bind_paths(env, inspected):
            guarded.add(name)
        spec = project["services"][name]
        related = []
        for key in ("pid", "ipc", "network_mode"):
            value = spec.get(key, "")
            if value.startswith("service:"):
                related.append(value.removeprefix("service:"))
        related.extend(value.split(":")[0] for value in spec.get("volumes_from", []))
        for other in related:
            if other in peers:
                peers[name].add(other)
                peers[other].add(name)
        for mount in inspected["Mounts"]:
            if mount["Type"] == "volume":
                volumes.setdefault(mount["Name"], set()).add(name)
    for names in volumes.values():
        for name in names:
            peers[name].update(names - {name})
    pending = list(guarded)
    while pending:
        for other in peers[pending.pop()] - guarded:
            guarded.add(other)
            pending.append(other)
    return guarded


def up_args(env: Any, *options: str, detach: str = "--detach") -> list[str] | None:
    """Start only guarded services before proof; leave native prototypes stopped."""
    guarded = getattr(env, "_bootstrap_services", None)
    if guarded is None:
        # Also used by the ordinary Compose path without bootstrap preparation.
        return ["up", detach, *options]
    if not guarded:
        return None
    return ["up", detach, *options, "--no-deps", *sorted(guarded)]


async def release(
    env: Any,
    compose: Any,
    *,
    ids: list[str] | None = None,
    start_native: Any = None,
) -> None:
    """Release only after callers verify all services; retain readiness checks.

    The direct Docker path passes the project's container *ids* and
    *start_native* (an Engine API start) instead of asking Compose."""
    if ids is None:
        ids = await compose_container_ids(compose)
    containers = {}
    for cid in ids:
        inspected = await env._inspect_container(cid)
        name = inspected["Config"]["Labels"]["com.docker.compose.service"]
        if name in containers:
            raise RuntimeError("Duplicate bootstrap service; start a fresh trial")
        containers[name] = cid
    if set(containers) != set(env._bootstrap_configs):
        raise RuntimeError("Bootstrap service set changed; start a fresh trial")
    guarded = env._bootstrap_services
    for name in guarded:
        await wait_ready(env, containers[name])
    pending = set(containers)
    while pending:
        ready = [
            name
            for name in sorted(pending)
            if not (
                set(env._bootstrap_project["services"][name].get("depends_on", {}))
                & pending
            )
        ]
        if not ready:
            raise RuntimeError("Cyclic bootstrap dependencies; fix the task services")
        for name in ready:
            spec = env._bootstrap_project["services"][name]
            for dep, rule in spec.get("depends_on", {}).items():
                if dep not in containers and rule.get("required") is False:
                    continue
                if rule.get("condition") == "service_completed_successfully":
                    await wait_completed(env, containers[dep])
                elif rule.get("condition") == "service_healthy":
                    await wait_healthy(
                        env,
                        containers[dep],
                        env._bootstrap_configs[dep],
                        native=dep not in guarded,
                    )
            if name in guarded:
                await env._docker_cli(["kill", "--signal", "USR1", containers[name]])
            elif start_native is not None:
                await start_native(containers[name])
            else:
                # Retain the stopped prototype's ID and original command,
                # environment, hooks and healthcheck. Never inject tools into
                # an unrelated distroless/scratch service.
                result = await compose(
                    ["up", "--detach", "--no-build", "--no-deps", "--no-recreate", name]
                )
                if result.return_code:
                    raise RuntimeError("Cannot start task service; inspect task logs")
            pending.remove(name)
    # Peers without dependency edges must all be running before any final
    # readiness check: one peer's healthcheck may contact another peer.
    completed = {
        dep
        for spec in env._bootstrap_project["services"].values()
        for dep, rule in spec.get("depends_on", {}).items()
        if rule.get("condition") == "service_completed_successfully"
    }
    await asyncio.gather(
        *(
            wait_healthy(
                env,
                cid,
                env._bootstrap_configs[name],
                allow_completed=name in completed,
                native=name not in guarded,
            )
            for name, cid in containers.items()
        )
    )


async def wait_completed(env: Any, cid: str) -> None:
    for _ in range(1200):
        state = (await env._inspect_container(cid))["State"]
        if not state["Running"]:
            if state["ExitCode"] != 0:
                raise RuntimeError("Task init service failed; inspect the task logs")
            return
        await asyncio.sleep(0.1)
    raise RuntimeError("Task init service timed out; inspect the task logs")


async def wait_healthy(
    env: Any,
    cid: str,
    config: dict,
    *,
    allow_completed: bool = False,
    native: bool = False,
) -> None:
    health = config.get("Healthcheck") or {}
    test = health.get("Test") or ["NONE"]
    if test[0] == "NONE" or allow_completed:
        state = (await env._inspect_container(cid))["State"]
        if not state["Running"]:
            if allow_completed and state["ExitCode"] == 0:
                return
            raise RuntimeError(
                "Task service exited during startup; inspect the task logs"
            )
        if test[0] == "NONE":
            return
    if test[0] == "CMD":
        command = test[1:]
    elif test[0] == "CMD-SHELL":
        command = [*(config.get("Shell") or ["/bin/sh", "-c"]), test[1]]
    else:
        raise RuntimeError("Unknown task healthcheck; use CMD or CMD-SHELL")
    from benchflow._runtime_tree import KIT_BIN
    from benchflow.sandbox.docker import route_exec_argv

    service = (config.get("Labels") or {}).get("com.docker.compose.service", "main")
    interval = (health.get("Interval") or 30_000_000_000) / 1e9
    start_interval = (health.get("StartInterval") or 5_000_000_000) / 1e9
    timeout = (health.get("Timeout") or 30_000_000_000) / 1e9
    start_period = health.get("StartPeriod", 0) / 1e9
    deadline = asyncio.get_running_loop().time() + start_period
    failures = 0
    while failures < (health.get("Retries") or 3):
        if native:
            state = (await env._inspect_container(cid))["State"]
            status = (state.get("Health") or {}).get("Status")
            if state["Running"] and status == "healthy":
                return
            if not state["Running"] or status != "starting":
                break
            await asyncio.sleep(0.1)
            continue
        try:
            # Native services above use Docker health metadata only; they may
            # be shell-less and never need a root runtime or a kit probe.
            route = await system_shell_route(getattr(env, "_env", env), service)
            env_tool = f"{KIT_BIN}/env" if route.kit else "/usr/bin/env"
            result = await asyncio.wait_for(
                env._docker_cli(
                    route_exec_argv(
                        route,
                        shlex.join(
                            [env_tool, "--", *(config.get("Env") or []), *command]
                        ),
                        container=cid,
                        user=config.get("User") or "0",
                    ),
                    check=False,
                ),
                timeout,
            )
            if result.return_code == 0:
                return
        except TimeoutError:
            pass
        if asyncio.get_running_loop().time() >= deadline:
            failures += 1
        if failures < (health.get("Retries") or 3):
            await asyncio.sleep(
                start_interval
                if asyncio.get_running_loop().time() < deadline
                else interval
            )
    raise RuntimeError(
        "Task healthcheck failed after bind verification; inspect task logs"
    )


async def wait_ready(env: Any, cid: str) -> None:
    """Require PID1 to have installed its release handler before signaling."""
    from benchflow.sandbox.docker import route_exec_argv

    inspected = await env._inspect_container(cid)
    service = ((inspected.get("Config") or {}).get("Labels") or {}).get(
        "com.docker.compose.service", "main"
    )
    route = await system_shell_route(getattr(env, "_env", env), service)
    for _ in range(100):
        result = await env._docker_cli(
            route_exec_argv(
                route,
                # Docker's optional init forwards signals to its one child.
                "pid=1; read -r name < /proc/1/comm; "
                "case $name in docker-init|tini) "
                "read -r pid rest < /proc/1/task/1/children || :;; esac; "
                "while read -r key mask rest; do "
                '[ "$key" != SigCgt: ] || { [ "$((0x$mask & 512))" != 0 ] && exit 0; exit 1; }; '
                'done < /proc/"$pid"/status; exit 1',
                container=cid,
                user="0",
            ),
            check=False,
        )
        if result.return_code == 0:
            break
        await asyncio.sleep(0.02)
    else:
        raise RuntimeError("Inert bootstrap did not become ready; start a fresh trial")


def image_env(env: Any, service: str = "main") -> dict[str, str]:
    """*service*'s own environment as Docker created it (image ENV and the
    task's Compose environment), the names with a value, from the daemon's
    record of its stopped prototype (prepare inspects every service); empty
    where S1 did not run."""
    configs = getattr(env, "_bootstrap_configs", {})
    if not isinstance(configs, dict):
        return {}
    return {
        key: value
        for row in (configs.get(service, {}).get("Env") or [])
        for key, _, value in [row.partition("=")]
        if value
    }


def restored_env(env: Any, service: str = "main") -> dict[str, str]:
    """Values cleared for bootstrap return only inside a released task command."""
    return {
        key: value
        for key, value in image_env(env, service).items()
        if key in EMPTY_ENV and key not in IMAGE_UNRESTORED_VARS
    }
