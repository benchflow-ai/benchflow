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
import contextlib
import json
import math
import os
import secrets
import shlex
import time
from typing import Any, NoReturn

from benchflow._bfx.locate import SANDBOX_PATH as WAITER
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

# The hold without BenchFlow's waiter (a backend bfx is not delivered to:
# Daytona's DinD, opt-in there): the image's own /bin/sh waits, looping on the
# image's /bin/sleep. Those are the task image's files and run before the
# release, so a backend that covers its services never uses it: there every
# held service waits in bfx (:func:`waiter_entrypoint`), and a start without
# bfx is refused (:func:`overlay`). Docker supplies empty loader and start-up
# variables to /bin/sh; the real environment is restored after the USR1 barrier.
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
#: The interactive shell's own start-up hooks: only an interactive shell runs
#: them, but a held container's environment is every root exec's, so it does
#: not carry a task's value for them either.
INTERACTIVE_SHELL_VARS = ("PROMPT_COMMAND", "PS0", "PS1", "PS4")
# A bare Docker env key / null Compose value removes an image variable when
# it is absent from the client's environment. Empty strings are insufficient:
# glibc acts on the mere presence of LD_SHOW_AUXV, for example.
EMPTY_ENV = dict.fromkeys((*SYSTEM_SHELL_UNSET_VARS, *INTERACTIVE_SHELL_VARS))


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


def _effective_argv(config: dict) -> list[str]:
    """The service's own process as Docker resolved it on the stopped
    prototype (``Entrypoint`` + ``Cmd``), what its release starts."""
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
    return argv


def entrypoint(config: dict) -> list[str]:
    """The image-shell hold's process (:data:`BOOTSTRAP`)."""
    argv = _effective_argv(config)
    return [
        "/bin/sh",
        "-c",
        BOOTSTRAP,
        "benchflow-bootstrap",
        *canonical_env(config.get("Env") or []),
        *argv,
    ]


def waiter_entrypoint(config: dict, token: str) -> list[str]:
    """The held service's process under BenchFlow's waiter (bfx's ``hold``):
    static, read-only, BenchFlow's own; it runs nothing of the image's until
    the controller releases it, then execs the service's own process. The
    held container's environment has the loader and start-up variables
    cleared (:data:`EMPTY_ENV`), since every ``docker exec`` BenchFlow makes
    into it as root inherits that environment. The values the service had go
    back to its own process alone: bfx sets each ``--setenv`` right before
    its exec (HOLD-CONTRACT.md), so nothing of the image's runs for them, and
    an image with no ``env`` (distroless) gets them as well. They sit in the
    waiter's argv, as visible in the container as its Config.Env was; they
    are never written to a record."""
    pairs = [arg for row in given_back(config) for arg in ("--setenv", row)]
    return [WAITER, "hold", "--token", token, *pairs, "--", *_effective_argv(config)]


def given_back(config: dict) -> list[str]:
    """The ``NAME=VALUE`` rows of *config*'s environment that the held
    container has cleared (:data:`EMPTY_ENV`): the service's own process gets
    them back at its release, and no root exec ever sees them."""
    return [
        row
        for row in canonical_env(config.get("Env") or [])
        if row.partition("=")[0] in EMPTY_ENV
    ]


def no_waiter_refusal(guarded: set[str]) -> str:
    """The refusal of a start whose services wait for their rules where
    BenchFlow's waiter (bfx) is not delivered. scoring classes it
    ``integrity_unavailable``: refused before the turn, never retried."""
    from benchflow._utils.scoring import HOLD_REFUSAL_HEAD, HOLD_WITHOUT_BFX

    return (
        HOLD_REFUSAL_HEAD
        + ", ".join(repr(n) for n in sorted(guarded))
        + " until its rules are in, and only BenchFlow's own waiter (bfx) "
        f"may run there before then; {HOLD_WITHOUT_BFX}, "
        "and the hold never runs the image's own shell instead. next: deliver "
        "bfx (unset BENCHFLOW_BFX_DELIVERY; a release wheel ships it, a "
        "source checkout builds it with `make bfx`); under an open network, a "
        "task whose allow_cloud_metadata the run grants with "
        "--allow-cloud-metadata waits for no rule, and needs no bfx"
    )


def waiter_mount(env: Any) -> dict | None:
    """The bfx mount a held service other than main gets (main has it from
    the bfx overlay), or None where bfx is not delivered."""
    from benchflow._bfx import delivery as bfx_delivery

    inner = getattr(env, "_env", None)
    mode = getattr(env, "bfx_delivery", None) or getattr(inner, "bfx_delivery", None)
    source = getattr(env, "_bfx_source", None)
    if mode not in (bfx_delivery.BIND, bfx_delivery.VOLUME) or not source:
        return None
    return bfx_delivery.mount(mode, source=source)


def _covers_services(env: Any) -> bool:
    """A backend that puts rules into its compose services (Docker, remote
    Docker): its holds wait for those rules, so they never run the image."""
    return callable(getattr(type(env), "metadata_service_targets", None))


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
    hold_mode = getattr(env, "_start_hold_mode", "")
    main_rules = getattr(env, "_main_metadata_rule", False) is True or bool(
        isinstance(hold_mode, str) and hold_mode
    )
    if main_rules:
        # main's cloud metadata rule for every user goes in before its own
        # processes start, and so does the task's network policy
        # on every process: main waits here, whatever its binds.
        guarded.add("main")
    every = getattr(env, "_hold_every_service", False) is True
    if every:
        # Under a network policy, and under an open one for the services'
        # cloud metadata rule, every service waits too, until its rules are
        # in: none of its processes, healthcheck included, runs before them.
        guarded |= set(configs)
    # The wait: BenchFlow's waiter (bfx) where it is delivered, so nothing of
    # the image's runs before the release. On a backend that covers its
    # services, a wait for rules (main's metadata rule, the task's network
    # policy, every service's metadata rule) never runs the image's shell:
    # without bfx it refuses the start by name. A wait for the bind proof
    # alone (an open network whose cloud metadata opt-out the run granted)
    # has no rule for the image to get ahead of, and keeps the image's shell
    # without bfx, as on 0.10.0; so does a backend that puts no rule into its
    # services (Daytona's DinD). A service whose image lacks the shell is
    # refused there, with the reason.
    mount = waiter_mount(env)
    rules = every or main_rules
    if mount is None and guarded and rules and _covers_services(env):
        from benchflow.sandbox.service_cover import ServiceCoverError

        raise ServiceCoverError(no_waiter_refusal(guarded))
    if mount is None:
        await _require_bootstrap_tools(env, guarded - {"main"}, prototypes, every=every)
    tokens: dict[str, str] = {}
    for name in sorted(guarded):
        config = configs[name]
        service = project["services"][name]
        if service.get("post_start") or service.get("pre_stop"):
            raise RuntimeError(
                "Compose lifecycle hooks cannot wait for bind verification; "
                "move them into the task entrypoint before retrying"
            )
        held: dict[str, Any] = {
            "command": [],
            "healthcheck": {"disable": True},
            "labels": {LABEL: config_receipt(config)},
            # Readiness dependencies are enforced after every bind is proved.
            "depends_on": {
                dep: {"condition": "service_started"}
                for dep in project["services"][name].get("depends_on", {})
            },
        }
        # Every docker exec into the held container inherits its environment:
        # the loader and start-up variables are cleared there, and given back
        # to the service's own process alone at its release.
        held["environment"] = EMPTY_ENV
        # Docker's stop sends SIGTERM, whatever the task or its image set: a
        # stop never sends the waiter its release signal.
        held["stop_signal"] = "SIGTERM"
        if name != "main" and _covers_services(env):
            # Docker never starts a held container again: at start-up
            # BenchFlow does, where the task's own policy would, after the
            # checks every start of a held container passes (release); after
            # it, an exit is final. main has its own no-restart file.
            held["restart"] = "no"
            held["deploy"] = {
                "restart_policy": {"condition": "none", "max_attempts": 0}
            }
        if mount is not None:
            tokens[name] = secrets.token_hex(16)
            held["entrypoint"] = waiter_entrypoint(config, tokens[name])
            if name != "main":
                held["volumes"] = [mount]
        else:
            held["entrypoint"] = entrypoint(config)
        services[name] = held
    env._bootstrap_configs = configs
    env._bootstrap_project = project
    env._bootstrap_services = guarded
    env._hold_tokens = tokens
    # The task's own restart policies, as Docker made them for the
    # prototypes (restart: and deploy.restart_policy alike).
    env._task_restart_policies = {
        name: dict(
            (prototypes[name].get("HostConfig") or {}).get("RestartPolicy") or {}
        )
        for name in guarded
    }
    return {"services": services}


#: What :data:`BOOTSTRAP` runs in a service before its release.
BOOTSTRAP_TOOLS = ("/bin/sh", "/bin/sleep", "/usr/bin/env")


async def _require_bootstrap_tools(
    env: Any,
    names: set[str],
    prototypes: dict,
    *,
    every: bool = True,
) -> None:
    """Refuse a service that cannot wait in the bootstrap: its image lacks a
    tool the wait runs (a distroless or scratch image). Read from the stopped
    prototype (``docker cp -L``, which runs nothing in it). The reason names
    what holds it: the task's network policy, else the cloud metadata rule
    under an open network (*every*), else its host binds."""
    import tempfile

    from benchflow.sandbox.service_cover import ServiceCoverError

    policy = str(getattr(env, "_network_policy", "") or "")
    shell = "give the image a shell (/bin/sh, /bin/sleep and /usr/bin/env)"
    if every and policy:
        until = f"the rules of the task's network policy ({policy}) are in"
        ways = f"To run it, {shell}."
    elif every:
        until = "its metadata rule is in"
        ways = (
            "To run it, grant the task's own opt-out (allow_cloud_metadata = "
            "true in the task, and --allow-cloud-metadata on the run) on a "
            f"machine with no cloud credentials, or {shell}."
        )
    else:
        until = "its host binds are proven"
        ways = f"To run it, {shell}, or drop the bind."
    for name in sorted(names):
        cid = str(prototypes[name].get("Id") or "")
        for tool in BOOTSTRAP_TOOLS:
            with tempfile.TemporaryDirectory() as scratch:
                copied = await env._docker_cli(
                    ["cp", "-L", f"{cid}:{tool}", scratch], check=False
                )
            if getattr(copied, "return_code", 1) != 0:
                lacks = "no shell" if tool == "/bin/sh" else f"no {tool}"
                raise ServiceCoverError(
                    f"sandbox startup refused: service {name!r}: this sidecar's "
                    f"image has {lacks}, so BenchFlow can't hold it until "
                    f"{until} (the wait runs {tool}). {ways}"
                )


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
    recover: Any = None,
) -> None:
    """Release only after callers verify all services; retain readiness checks.

    The direct Docker path passes the project's container *ids* and
    *start_native* (an Engine API start) instead of asking Compose.
    *recover* (``await recover(name, cid)``) covers again a held service
    BenchFlow started again at start-up (:func:`_settle_restarts`); without
    it a held service that exits is not started again."""
    if ids is None:
        ids = await compose_container_ids(compose)
    containers = {}
    # Each service's restart count before its release, and whether its PID 1
    # is another container's (pid: service:<x>).
    counts: dict[str, int] = {}
    joined: set[str] = set()
    for cid in ids:
        inspected = await env._inspect_container(cid)
        name = inspected["Config"]["Labels"]["com.docker.compose.service"]
        if name in containers:
            raise RuntimeError("Duplicate bootstrap service; start a fresh trial")
        containers[name] = cid
        counts[name] = int(inspected.get("RestartCount") or 0)
        pid_mode = str((inspected.get("HostConfig") or {}).get("PidMode") or "")
        if pid_mode.startswith("container:"):
            joined.add(name)
    if set(containers) != set(env._bootstrap_configs):
        raise RuntimeError("Bootstrap service set changed; start a fresh trial")
    guarded = env._bootstrap_services
    tokens = getattr(env, "_hold_tokens", None) or {}
    for name in guarded:
        await wait_ready(env, containers[name], token=tokens.get(name))
    # When each held service's own process started (its release): its
    # healthcheck counts from there, as Docker's counts from a start.
    released: dict[str, float] = {}
    starts = _Starts(env, containers, released, recover)
    for name in guarded:
        await starts.mark(name)

    def on_held(name: str) -> Any:
        if name not in guarded:
            return None
        return starts.held_again(name, joined=name in joined)

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
                        released_at=released.get(dep),
                        cap=_health_cap(env),
                        on_held=on_held(dep),
                    )
            if name in guarded:
                await _signal_release(
                    env, containers[name], name, joined=name in joined
                )
                released[name] = asyncio.get_running_loop().time()
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
    if recover is not None:
        await _settle_restarts(env, containers, guarded - {"main"} - joined, starts)
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
                released_at=released.get(name),
                cap=_health_cap(env),
                on_held=on_held(name),
            )
            for name, cid in containers.items()
            if name not in starts.refused
        )
    )
    # From here an exit is final: Docker never starts a held container again,
    # and BenchFlow starts them only at start-up (held_exited).
    env._held_restart_base = {
        name: (containers[name], counts[name]) for name in sorted(guarded - {"main"})
    }
    env._held_start_marks = dict(starts.marks)
    env._held_restart_since = int(time.time())
    if starts.refused:
        keep_exited(
            env,
            [r for r in await held_exited(env) if r["service"] in starts.refused],
            EXITED_AT_START_UP,
        )
    if recover is not None:
        await _watch_exits(
            env, [cid for _, (cid, _n) in env._held_restart_base.items()]
        )


async def _still_waiting(env: Any, cid: str, token: str) -> bool:
    """After a ``--release`` that did not answer 0: whether the waiter may
    still be there, unreleased. A program that exits at once after its
    release takes the container, and the release's own exec with it; that
    release happened (the readiness checks next see the container stopped).
    Only an exited container, or a running one where bfx finds no waiter
    (1), counts as released; anything it can't tell fails closed."""
    try:
        state = (await env._inspect_container(cid)).get("State") or {}
    except Exception:  # the daemon did not say: not known to be released
        return True
    if not state.get("Running"):
        return False
    return await _waiter(env, cid, "ready", token) != 1


#: sh, as root in a held service: USR1 to its bootstrap (PID 1, or the one
#: child of Docker's init), from inside the container.
_RELEASE_SH = (
    "pid=1; read -r name < /proc/1/comm; "
    "case $name in docker-init|tini) "
    "read -r pid rest < /proc/1/task/1/children || :;; esac; "
    'kill -USR1 "$pid"'
)


async def _signal_release(
    env: Any, cid: str, name: str, *, joined: bool = False
) -> None:
    """Release the held service *name* (container *cid*). Under BenchFlow's
    waiter, bfx sends the release from inside the container, to the waiter
    its token names (in a shared pid namespace too). Never ``docker kill``:
    Docker's kill marks a container as stopped by hand and cancels its
    restart policy for its next exit, so a service that Docker would start
    again stayed down. The image-shell hold: main by Docker's kill, as before,
    and so is a service in another container's pid namespace (*joined*),
    whose PID 1 is not its own; any other service by its own shell's kill."""
    token = (getattr(env, "_hold_tokens", None) or {}).get(name)
    if token is not None:
        if await _waiter(env, cid, "release", token) != 0 and await _still_waiting(
            env, cid, token
        ):
            raise RuntimeError(
                f"Cannot release service {name!r} from BenchFlow's waiter; "
                "start a fresh trial"
            )
        return
    if name == "main" or joined:
        await env._docker_cli(["kill", "--signal", "USR1", cid])
        return
    from benchflow.sandbox.docker import route_exec_argv

    route = await system_shell_route(getattr(env, "_env", env), name)
    result = await env._docker_cli(
        route_exec_argv(route, _RELEASE_SH, container=cid, user="0"), check=False
    )
    if getattr(result, "return_code", 1) != 0:
        raise RuntimeError(
            f"Cannot release service {name!r} from the inert bootstrap; start a "
            "fresh trial"
        )


#: The start-up budget's default (``environment.build_timeout_sec``).
DEFAULT_HEALTH_CAP_SEC = 600.0


def _health_cap(env: Any) -> float | None:
    """How long a held service's own healthcheck may take after its release:
    the environment's start-up budget (``environment.build_timeout_sec``)."""
    config = getattr(env, "task_env_config", None)
    value = getattr(config, "build_timeout_sec", None)
    if isinstance(value, (int, float)) and value > 0:
        return float(value)
    # A budget of 0 (or none) still ends the wait: the default budget.
    return DEFAULT_HEALTH_CAP_SEC if config is not None else None


#: Restart policies under which Docker starts a service again by itself.
_RESTARTS = frozenset({"always", "unless-stopped", "on-failure"})
#: How long a held service Docker may restart is watched after each release:
#: a restart in that time is still the start's (covered again, released).
RESTART_SETTLE_SEC = 5.0
#: Restarts at start-up a held service may take before the start fails.
RESTART_LIMIT = 5
#: Why a held service Docker restarted after start-up waits in the bootstrap.
HELD_AFTER_RESTART = "held after a restart; its metadata rule can't be re-proven"
#: Why a held service whose container exited, and that Docker is not
#: starting again, has no rules to prove: it runs nothing.
HELD_EXITED = (
    "exited and did not start again; its network namespace is gone with it, so "
    "it runs nothing and its rules can't be proven"
)
#: When a held service exited (held_exited): before the agent's turn (the
#: task's own one-shot shape), in the agent's turn, or in test.sh.
EXITED_AT_START_UP = "start-up"
EXITED_IN_TURN = "the agent's turn"
EXITED_IN_TEST = "test.sh"


#: How long a check that found its service's waiter not yet released (the
#: release's own exec still on its way) waits before it runs again.
RELEASE_PENDING_SEC = 0.2


def run_if_released(env: Any, service: str) -> tuple[str, ...]:
    """What goes ahead of a command BenchFlow runs in the held *service*: bfx
    runs the command only when no unreleased waiter holds the service's
    token, so it never runs while the service waits in the hold, whatever
    started its container again. () where the service is not held by bfx."""
    token = (getattr(env, "_hold_tokens", None) or {}).get(service)
    if not isinstance(token, str) or not token:
        return ()
    return (WAITER, "hold", "--run-if-released", "--token", token, "--")


#: ``bfx hold --run-if-released``'s exit statuses when it ran nothing: an
#: unreleased waiter holds the token, or /proc could not be listed.
HELD_EXEC = 75
UNCHECKED_EXEC = 125
REFUSED_EXEC = (HELD_EXEC, UNCHECKED_EXEC)


def held_refusal(result: Any) -> bool:
    """Whether a command sent through :func:`run_if_released` ran nothing:
    bfx's exit status and its one line."""
    if getattr(result, "return_code", None) not in (HELD_EXEC, UNCHECKED_EXEC):
        return False
    said = (
        f"{getattr(result, 'stderr', '') or ''}\n{getattr(result, 'stdout', '') or ''}"
    )
    return "bfx hold: " in said and "nothing ran" in said


async def waits_in_hold(env: Any, service: str) -> bool:
    """Whether BenchFlow's waiter holds *service* now (bfx answers, nothing
    of the image's runs): for a command there whose exit status is the one
    bfx gives when it ran nothing, which the command itself could give."""
    token = (getattr(env, "_hold_tokens", None) or {}).get(service)
    if not token:
        return False
    cid = await env._service_container_id(service)
    if not cid:
        return False
    try:
        return await _waiter(env, cid, "ready", token) == 0
    except RuntimeError:  # the daemon did not answer: the status stands
        return False


def note_held_refusal(env: Any, service: str, exit_code: int) -> None:
    """A command BenchFlow sent into the held *service* after the start ran
    nothing (bfx's *exit_code*, its waiter there again): the service waits in
    the hold again, and is named in :func:`held_restarts` from here on."""
    refused = getattr(env, "_held_refusals", None)
    if not isinstance(refused, dict):
        refused = env._held_refusals = {}
    refused.setdefault(
        service,
        {
            "refused_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "exit_code": exit_code,
        },
    )


def _namespace_mark(inspected: dict) -> tuple[str, str]:
    """When a held container last started, and its network namespace
    (Docker's SandboxKey, new at every start)."""
    state = inspected.get("State") or {}
    sandbox = (inspected.get("NetworkSettings") or {}).get("SandboxKey")
    return str(state.get("StartedAt") or ""), str(sandbox or "")


def _same_start(mark: tuple[str, str], inspected: dict) -> bool:
    """Whether *inspected* is the start *mark* names: the same StartedAt and,
    while it runs, the same namespace (a stopped container has none)."""
    started, sandbox = mark
    state = inspected.get("State") or {}
    if str(state.get("StartedAt") or "") != started:
        return False
    now = str((inspected.get("NetworkSettings") or {}).get("SandboxKey") or "")
    return not state.get("Running") or now == sandbox


def _restarts_after(policy: Any, exit_code: Any, starts: int) -> bool:
    """Whether the task's own restart *policy* (Docker's RestartPolicy, as
    its prototype was made) starts a service again after it exited with
    *exit_code*, *starts* BenchFlow restarts in."""
    policy = policy if isinstance(policy, dict) else {}
    name = str(policy.get("Name") or "")
    if name in ("always", "unless-stopped"):
        return True
    if name != "on-failure" or exit_code in (0, None):
        return False
    most = int(policy.get("MaximumRetryCount") or 0)
    return most == 0 or starts < most


class _Starts:
    """The held services' starts while the trial's own start goes on. Docker
    never starts a held container again (the overlay's restart policy is
    ``no``): where the task's policy would, BenchFlow does (:meth:`again`),
    after the checks every start of a held container passes, and covers and
    releases it as at its first start. Each container's start as it was
    released is kept (:meth:`mark`)."""

    def __init__(
        self,
        env: Any,
        containers: dict[str, str],
        released: dict[str, float],
        recover: Any,
    ) -> None:
        self.env = env
        self.containers = containers
        self.released = released
        self.recover = recover
        self.starts: dict[str, int] = {}
        self.marks: dict[str, tuple[str, str]] = {}
        #: Services whose start BenchFlow refused at start-up: held_exited.
        self.refused: set[str] = set()

    async def mark(self, name: str) -> None:
        inspected = await self.env._inspect_container(self.containers[name])
        self.marks[name] = _namespace_mark(inspected)

    def restarts(self, name: str, state: dict) -> bool:
        """Whether the task's policy would start *name* again after its exit
        (*state*, Docker's State of the stopped container)."""
        if not callable(getattr(type(self.env), "_start_held_again", None)):
            return False
        policies = getattr(self.env, "_task_restart_policies", None) or {}
        return _restarts_after(
            policies.get(name), state.get("ExitCode"), self.starts.get(name, 0)
        )

    async def again(self, name: str) -> bool:
        """*name* exited at start-up and its task's policy would start it
        again: BenchFlow starts the same container (``_start_held_again``:
        the checks, then the start, then the bind proof), then covers and
        releases it, as many times as :data:`RESTART_LIMIT` allows. False
        when the checks refuse it: it stays down, ``held_exited`` from the
        start-up, with the reason."""
        from benchflow.sandbox._held_mounts import HeldStartRefused

        cid = self.containers[name]
        self.starts[name] = self.starts.get(name, 0) + 1
        if self.starts[name] > RESTART_LIMIT:
            raise RuntimeError(
                f"Task service {name!r} restarted {self.starts[name]} times during "
                "startup; inspect the task logs"
            )
        try:
            await self.env._start_held_again(name, cid)
        except HeldStartRefused as exc:
            self.refused.add(name)
            refusals = getattr(self.env, "_held_start_refusals", None)
            if not isinstance(refusals, dict):
                refusals = self.env._held_start_refusals = {}
            refusals[name] = str(exc)
            # A start BenchFlow made and stopped again (its binds refused) is
            # BenchFlow's own: the service exited, nothing started it.
            await self.mark(name)
            return False
        # Its PID 1 is the waiter again, with the same token: nothing of the
        # image's runs before its namespace is covered.
        await wait_ready(
            self.env,
            cid,
            token=(getattr(self.env, "_hold_tokens", None) or {}).get(name),
        )
        if self.recover is not None:
            await self.recover(name, cid)
        await self.mark(name)
        await _signal_release(self.env, cid, name)
        self.released[name] = asyncio.get_running_loop().time()
        return True

    def held_again(self, name: str, *, joined: bool = False) -> Any:
        """For :func:`wait_healthy`, when a check of *name* did not pass: what
        the service is doing. ``"again"``: it had exited, and BenchFlow
        started it again (its check counts from that release);
        ``"refused"``: it exited and its start was refused (it stays down);
        ``"pending"``: bfx found its waiter unreleased with nothing else
        changed, the release's own exec still on its way; ``"running"``: its
        program runs, and the failure counts; ``"exited"``: it exited for
        good (main, and a service in another's pid namespace, are never
        started again). Raises when something other than BenchFlow started
        it."""

        async def held(refused: bool) -> str:
            inspected = await self.env._inspect_container(self.containers[name])
            state = inspected.get("State") or {}
            if state.get("Running"):
                if name in self.marks and not _same_start(self.marks[name], inspected):
                    raise RuntimeError(
                        f"Task service {name!r} started again outside BenchFlow "
                        "during startup; inspect the task logs"
                    )
                if refused:
                    await asyncio.sleep(RELEASE_PENDING_SEC)
                    return "pending"
                return "running"
            if name == "main" or joined or not self.restarts(name, state):
                return "exited"
            return "again" if await self.again(name) else "refused"

        return held


async def _settle_restarts(
    env: Any,
    containers: dict[str, str],
    held: set[str],
    starts: _Starts,
) -> None:
    """A held service whose program exits at start-up, where the task's own
    restart policy would start it again (one that exits once on its first
    start): BenchFlow starts it again (:meth:`_Starts.again`). Each service
    with such a policy is watched until it has run
    :data:`RESTART_SETTLE_SEC` since its last release, or is down for good
    (its policy is done with it, or its start was refused)."""
    policies = getattr(env, "_task_restart_policies", None) or {}
    watched = {
        name
        for name in held
        if str((policies.get(name) or {}).get("Name") or "") in _RESTARTS
    }
    if not watched:
        return
    loop = asyncio.get_running_loop()
    since = {name: starts.released.get(name, loop.time()) for name in watched}
    while watched:
        for name in sorted(watched):
            state = (await env._inspect_container(containers[name])).get("State") or {}
            if state.get("Running"):
                if loop.time() - since[name] >= RESTART_SETTLE_SEC:
                    watched.discard(name)
            elif starts.restarts(name, state) and await starts.again(name):
                since[name] = starts.released[name]
            else:
                # Down for good: the readiness check says so, or its start
                # was refused (held_exited).
                watched.discard(name)
        if watched:
            await asyncio.sleep(0.2)


async def held_restarts(env: Any) -> list[dict[str, Any]]:
    """The held services Docker restarted after the start: each waits in the
    bootstrap again, in a new network namespace whose rules can't be put
    back and proven, so none of its code runs (:data:`HELD_AFTER_RESTART`).
    One record per service: its name, how many restarts, when the last one
    started (Docker's StartedAt) and why. A service where bfx ran nothing of
    BenchFlow's (:func:`note_held_refusal`) is one of them, and says so."""
    base = getattr(env, "_held_restart_base", None)
    refusals = getattr(env, "_held_refusals", None) or {}
    found: list[dict[str, Any]] = []
    for name, (cid, count) in sorted((base or {}).items()):
        try:
            inspected = await env._inspect_container(cid)
        except Exception:  # a container gone is not a restart
            continue
        now = int(inspected.get("RestartCount") or 0)
        # Docker waiting out its restart delay: the next start is the
        # waiter's again, as for one already restarted.
        pending = (inspected.get("State") or {}).get("Restarting") is True
        if (
            now > count
            or pending
            or _started_since(env, name, inspected)
            or name in refusals
        ):
            record = {
                "service": name,
                "restarts": now - count,
                "restarted_at": (inspected.get("State") or {}).get("StartedAt"),
                "why": HELD_AFTER_RESTART,
            }
            if pending:
                record["restart_pending"] = True
            if name in refusals:
                record["exec_refused"] = dict(refusals[name])
            seen = (getattr(env, "_held_exits", None) or {}).get(cid)
            exit_code = seen if seen is not None else await _last_exit(env, cid)
            if exit_code is not None:
                record["exit_code"] = exit_code
            found.append(record)
    return found


def _started_since(env: Any, name: str, inspected: dict) -> bool:
    """Whether the held *name* started again after the start (its StartedAt
    or its namespace is not the one its rules went into)."""
    mark = (getattr(env, "_held_start_marks", None) or {}).get(name)
    return mark is not None and not _same_start(mark, inspected)


async def held_exited(env: Any) -> list[dict[str, Any]]:
    """The held services whose container exited after the start and that
    Docker is not starting again (no restart policy, or one that gave up): no
    network namespace, nothing running (:data:`HELD_EXITED`). A restarted or
    restarting one is :func:`held_restarts`'. One record per service: its
    name, its exit status and when it exited (Docker's FinishedAt)."""
    base = getattr(env, "_held_restart_base", None)
    found: list[dict[str, Any]] = []
    for name, (cid, count) in sorted((base or {}).items()):
        try:
            inspected = await env._inspect_container(cid)
        except Exception:  # a container gone: nothing to say of it
            continue
        state = inspected.get("State") or {}
        if (
            state.get("Running") is not False
            or state.get("Restarting") is not False
            or int(inspected.get("RestartCount") or 0) > count
            or _started_since(env, name, inspected)
        ):
            continue
        record = {"service": name, "finished_at": state.get("FinishedAt")}
        if isinstance(state.get("ExitCode"), int):
            record["exit_code"] = state["ExitCode"]
        refused = (getattr(env, "_held_start_refusals", None) or {}).get(name)
        if refused:
            # BenchFlow did not start it again: a check refused it.
            record["start_refused"] = refused
        found.append(record)
    return found


def keep_exited(
    env: Any, exited: list[dict[str, Any]], phase: str
) -> list[dict[str, Any]]:
    """Keep each :func:`held_exited` record on the sandbox (result.json's
    ``held_exited``), with its phase: when it was first seen, start-up for
    one that had exited before the agent's turn, else *phase*. Returns what
    is kept."""
    kept = {r["service"]: r for r in getattr(env, "_held_exited", None) or []}
    before = getattr(env, "_held_exited_before_turn", None) or set()
    for record in exited:
        name = record["service"]
        if name not in kept:
            record["phase"] = EXITED_AT_START_UP if name in before else phase
            record["why"] = HELD_EXITED
            kept[name] = record
    env._held_exited = [kept[name] for name in sorted(kept)]
    return env._held_exited


async def service_down(env: Any, service: str, phase: str) -> str | None:
    """Why the held *service* can't run a command after the start: it exited
    and did not start again (kept, :func:`keep_exited`), or it waits in the
    hold again (:func:`held_restarts`); None when neither."""
    if not isinstance(getattr(env, "_held_restart_base", None), dict):
        return None
    for record in await held_exited(env):
        if record["service"] == service:
            keep_exited(env, [record], phase)
            return f"compose service {service!r} {HELD_EXITED}"
    for record in await held_restarts(env):
        if record["service"] == service:
            return f"compose service {service!r} was {HELD_AFTER_RESTART}"
    return None


async def note_exits_before_turn(env: Any) -> None:
    """Which held services had exited before the agent's turn began (a
    one-shot service done at start-up, the task's own shape): kept once, on
    the turn's first start, for :func:`held_exited`'s phase."""
    if not isinstance(getattr(env, "_held_restart_base", None), dict):
        return
    if isinstance(getattr(env, "_held_exited_before_turn", None), set):
        return
    env._held_exited_before_turn = {r["service"] for r in await held_exited(env)}


async def _watch_exits(env: Any, ids: list[str]) -> None:
    """Keep the exit status of each held container's death from its release
    on (``env._held_exits``, container id to its last exit code): a running
    restarted container shows only its new run, and Docker keeps a short
    backlog of events, so asking afterwards can come back empty. One
    ``docker events`` stream, stopped by :func:`stop_exit_watch` (the
    sandbox's stop) or when the sandbox is collected."""
    if not ids:
        return
    import weakref

    argv = ["docker", "events", "--filter", "event=die"]
    for cid in ids:
        argv += ["--filter", f"container={cid}"]
    argv += ["--format", "{{.Actor.ID}} {{.Actor.Attributes.exitCode}}"]
    client_env = getattr(env, "_docker_client_env", None)
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=client_env() if callable(client_env) else None,
        )
    except OSError:  # the record goes without exit codes
        return
    exits: dict[str, int] = {}
    env._held_exits = exits

    async def read() -> None:
        assert proc.stdout is not None
        async for raw in proc.stdout:
            parts = raw.decode(errors="replace").split()
            if len(parts) == 2 and parts[1].isdigit():
                exits[parts[0]] = int(parts[1])

    env._held_exit_watch = (proc, asyncio.ensure_future(read()))
    weakref.finalize(env, _kill_quietly, proc)


def _kill_quietly(proc: Any) -> None:
    if getattr(proc, "returncode", 0) is None:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()


async def stop_exit_watch(env: Any) -> None:
    """End :func:`_watch_exits`' stream (the sandbox's stop)."""
    watch = getattr(env, "_held_exit_watch", None)
    if not watch:
        return
    env._held_exit_watch = None
    proc, reader = watch
    _kill_quietly(proc)
    reader.cancel()
    with contextlib.suppress(TimeoutError, ProcessLookupError):
        await asyncio.wait_for(proc.wait(), 5)


async def _last_exit(env: Any, cid: str) -> int | None:
    """The exit status the container *cid* last died with since the start's
    release, from Docker's own events (a restarted container's state shows
    only its new run). Asked a few times, as a busy daemon can answer an
    events window empty. None when Docker does not say."""
    since = getattr(env, "_held_restart_since", None)
    if not isinstance(since, int):
        return None
    for attempt in range(_EXIT_TRIES):
        try:
            result = await exec_raw(
                env._docker_cli,
                [
                    "events",
                    "--since",
                    str(since - 2),
                    "--until",
                    str(int(time.time()) + 1),
                    "--filter",
                    f"container={cid}",
                    "--filter",
                    "event=die",
                    "--format",
                    "{{.Actor.Attributes.exitCode}}",
                ],
                check=False,
            )
        except Exception:  # the record goes without it
            return None
        codes = [
            line.strip()
            for line in str(getattr(result, "stdout", "") or "").splitlines()
            if line.strip().isdigit()
        ]
        if codes:
            return int(codes[-1])
        if attempt + 1 < _EXIT_TRIES:
            await asyncio.sleep(1.0)
    return None


#: How many times :func:`_last_exit` asks Docker's events.
_EXIT_TRIES = 3


async def wait_completed(env: Any, cid: str) -> None:
    for _ in range(1200):
        state = (await env._inspect_container(cid))["State"]
        if not state["Running"]:
            if state["ExitCode"] != 0:
                raise RuntimeError("Task init service failed; inspect the task logs")
            return
        await asyncio.sleep(0.1)
    raise RuntimeError("Task init service timed out; inspect the task logs")


#: How often a readiness command's watch looks again, in seconds.
READINESS_TICK_SEC = 0.1
#: How long the host waits past a readiness command's timeout for its exec
#: to answer: the command's process group ends in the container at the
#: timeout, so this is only a backstop.
READINESS_GRACE_SEC = 10.0


def bounded_readiness(command: str, timeout_sec: float) -> str:
    """sh: *command* in the background of the readiness exec's shell, and
    the exec's whole process group ended (SIGKILL) once *timeout_sec* has
    passed: the command, every process it started that stayed in its group,
    and the shell and its watch, so nothing of a readiness check BenchFlow
    stopped waiting for runs on (such a process would run outside
    the task's network policy, and R1 would refuse the trial for it). A
    ``docker exec`` is a session and process group of its own (runc makes
    its process a session leader), so ``kill -KILL 0`` reaches exactly the
    exec's processes. The exit status is the command's when it ends in
    time. The watch counts ticks of :data:`READINESS_TICK_SEC` and needs
    only the shell, ``sleep`` (the bootstrap's own) and ``/proc``."""
    ticks = max(1, math.ceil(timeout_sec / READINESS_TICK_SEC))
    return (
        f"{command} & bf_p=$!; bf_n=0; "
        'while kill -0 "$bf_p" 2>/dev/null; do '
        '{ read -r bf_s < "/proc/$bf_p/stat"; } 2>/dev/null || break; '
        'case $bf_s in *") Z "*) break;; esac; '
        f'[ "$bf_n" -lt {ticks} ] || kill -KILL 0; '
        f"sleep {READINESS_TICK_SEC}; bf_n=$((bf_n + 1)); done; "
        'wait "$bf_p"'
    )


#: The line BenchFlow's waiter writes when the service's own program does
#: not start at its release (HOLD-CONTRACT.md): the program and the reason.
_EXEC_FAILED = "bfx hold: cannot start "


async def _failed_exec(env: Any, cid: str, program: str | None, exit_code: Any) -> str:
    """Why the service *cid* stopped at its start, from the container's last
    output: bfx's own line (``bfx hold: cannot start PROGRAM: REASON``) only
    when it names *program*, the ARGV[0] BenchFlow gave the waiter. Any other
    line of that shape is the container's own output (a program can print
    it), shown as that, with the exit status; "" when there is none."""
    try:
        result = await exec_raw(
            env._docker_cli, ["logs", "--tail", "20", cid], check=False
        )
    except Exception:
        return ""
    lines = [
        line.strip()
        for stream in (getattr(result, "stderr", ""), getattr(result, "stdout", ""))
        for line in reversed((stream or "").splitlines())
    ]
    if program:
        own = f"{_EXEC_FAILED}{program}: "
        for line in lines:
            if line.startswith(own):
                return f"bfx could not start {program!r}: {line[len(own) :][:200]}"
    for line in lines:
        if line.startswith(_EXEC_FAILED):
            code = f"exit {exit_code}; " if isinstance(exit_code, int) else ""
            return (
                f"the service's program did not start or stopped at once ({code}"
                f"the container's last output, not BenchFlow's: {line[:200]!r})"
            )
    return ""


async def wait_healthy(
    env: Any,
    cid: str,
    config: dict,
    *,
    allow_completed: bool = False,
    native: bool = False,
    released_at: float | None = None,
    cap: float | None = None,
    on_held: Any = None,
) -> None:
    """Wait until the service *cid* (its original *config*) is healthy, by
    its own healthcheck. A held service's check runs here, from the host, on
    Docker's timing from *released_at* (the loop time of its release); a
    native one's is Docker's own. Under bfx the check runs only once the
    service is released (:func:`run_if_released`). For a held service,
    ``await on_held(refused)`` (:meth:`_Starts.held_again`) says what a check
    that did not pass met: a service BenchFlow started again (its check
    counts from that release), one whose start was refused (no wait), a
    release still on its way, or one that exited for good; all within the
    same budget."""
    health = config.get("Healthcheck") or {}
    test = health.get("Test") or ["NONE"]
    if test[0] == "NONE" or allow_completed:
        while True:
            state = (await env._inspect_container(cid))["State"]
            if state["Running"]:
                break
            if allow_completed and state["ExitCode"] == 0:
                return
            answer = await on_held(False) if on_held is not None else "exited"
            if answer == "refused":
                return
            if answer != "again":
                await _exited_during_startup(env, cid, config, state)
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
    retries = health.get("Retries") or 3
    if native:
        # Native services use Docker health metadata only; they may be
        # shell-less and never need a root runtime or a kit probe.
        while True:
            state = (await env._inspect_container(cid))["State"]
            status = (state.get("Health") or {}).get("Status")
            if state["Running"] and status == "healthy":
                return
            if not state["Running"] or status != "starting":
                break
            await asyncio.sleep(0.1)
        raise RuntimeError(
            f"Task healthcheck failed after bind verification (service "
            f"{service!r}); inspect task logs"
        )
    interval = (health.get("Interval") or 30_000_000_000) / 1e9
    start_interval = (health.get("StartInterval") or 5_000_000_000) / 1e9
    timeout = (health.get("Timeout") or 30_000_000_000) / 1e9
    start_period = (health.get("StartPeriod") or 0) / 1e9
    loop = asyncio.get_running_loop()
    # Docker's own counting (its daemon's health monitor), from when the
    # service's process started: its release here, the container's start
    # under Docker. Docker's first probe comes one pause after the start (the
    # interval, or start_interval when there is a start period), and a probe
    # inside the start period counts no failure. This one probes from the
    # start, so a ready service goes on at once, but a failure before
    # Docker's first probe would have run counts nothing: a check that
    # passes under Docker passes here.
    started = loop.time() if released_at is None else released_at
    first = started + (start_interval if start_period > 0 else interval)
    # A check that never passes (an hour's interval, a test that is always
    # false) would hold the start for as long as Docker would wait: it stops
    # at the environment's own start-up budget instead (no agent code has run).
    give_up = None if cap is None else loop.time() + cap

    def pause() -> float:
        now = loop.time()
        wait = start_interval if now - started < start_period else interval
        return min(wait, first - now) if now < first else wait

    hold = run_if_released(env, service)
    failures = 0
    while True:
        probe_started = loop.time()
        try:
            route = await system_shell_route(getattr(env, "_env", env), service)
            env_tool = f"{KIT_BIN}/env" if route.kit else "/usr/bin/env"
            # The command ends with its process group at its timeout, in
            # the container (bounded_readiness): when BenchFlow stops
            # waiting, nothing of it runs on.
            result = await asyncio.wait_for(
                exec_raw(
                    env._docker_cli,
                    route_exec_argv(
                        route,
                        bounded_readiness(
                            shlex.join(
                                [env_tool, "--", *(config.get("Env") or []), *command]
                            ),
                            timeout,
                        ),
                        container=cid,
                        user=config.get("User") or "0",
                        hold=hold,
                    ),
                    check=False,
                ),
                timeout + READINESS_GRACE_SEC,
            )
            if result.return_code == 0:
                return
            refused = bool(hold) and held_refusal(result)
            if on_held is not None:
                answer = await on_held(refused)
                if answer == "refused":
                    return
                if answer == "exited":
                    state = (await env._inspect_container(cid))["State"]
                    await _exited_during_startup(env, cid, config, state)
                if answer in ("again", "pending"):
                    # Nothing of its own ran, so no failure counts; a
                    # service started again counts from that release, as
                    # Docker's check counts from a restart.
                    if answer == "again":
                        started = loop.time()
                        first = started + (
                            start_interval if start_period > 0 else interval
                        )
                        failures = 0
                    if give_up is not None and loop.time() > give_up:
                        _start_budget_spent(service, cap)
                    continue
            elif refused:
                raise RuntimeError(
                    f"Task service {service!r} waits in the hold again; "
                    "inspect the task logs"
                )
        except TimeoutError:
            pass
        if probe_started >= first and probe_started - started >= start_period:
            failures += 1
            if failures >= retries:
                break
        if give_up is not None and loop.time() + pause() > give_up:
            _start_budget_spent(service, cap)
        await asyncio.sleep(pause())
    raise RuntimeError(
        f"Task healthcheck failed after bind verification (service {service!r}); "
        "inspect task logs"
    )


async def _exited_during_startup(
    env: Any, cid: str, config: dict, state: dict
) -> NoReturn:
    service = (config.get("Labels") or {}).get("com.docker.compose.service", "main")
    held = service in (getattr(env, "_hold_tokens", None) or {})
    try:
        program = _effective_argv(config)[0] if held else None
    except RuntimeError:
        program = None
    failed = await _failed_exec(env, cid, program, state.get("ExitCode"))
    raise RuntimeError(
        f"Task service {service!r} exited during startup"
        + (f" ({failed})" if failed else "")
        + "; inspect the task logs"
    )


def _start_budget_spent(service: str, cap: float | None) -> NoReturn:
    from benchflow.sandbox.protocol import EnvironmentStartTimeoutError

    raise EnvironmentStartTimeoutError(
        f"Task healthcheck of service {service!r} did not pass within "
        f"{cap:g}s of its release (environment.build_timeout_sec, the "
        "environment's start-up budget); inspect task logs"
    )


#: How many times, and how far apart, the controller asks bfx whether a
#: waiter is ready (its handlers installed), about 10 s in all.
WAITER_READY_TRIES = 200
WAITER_READY_SEC = 0.05
#: How long one of those questions may take.
WAITER_CALL_SEC = 30.0


async def _waiter(env: Any, cid: str, verb: str, token: str) -> int:
    """``bfx hold --ready|--release TOKEN`` in the container, as root, on
    the bfx route (TRUSTED-RUNTIME v2 I1): bfx by absolute path from the
    read-only /.bfx, nothing of the image's. Its exit
    status."""
    try:
        result = await asyncio.wait_for(
            exec_raw(
                env._docker_cli,
                ["exec", "-u", "0", cid, WAITER, "hold", f"--{verb}", token],
                check=False,
            ),
            WAITER_CALL_SEC,
        )
    except TimeoutError:
        raise RuntimeError(
            f"BenchFlow's waiter did not answer --{verb} within "
            f"{WAITER_CALL_SEC:g}s; the Docker daemon is not answering"
        ) from None
    return int(getattr(result, "return_code", 1))


async def wait_ready(env: Any, cid: str, *, token: str | None = None) -> None:
    """Require PID1 to have installed its release handler before signaling.
    With *token* the holder is BenchFlow's waiter, and bfx itself answers
    (nothing of the image's runs); without, the image-shell hold's PID 1 is
    read through the system shell route."""
    if token is not None:
        for _ in range(WAITER_READY_TRIES):
            status = await _waiter(env, cid, "ready", token)
            if status == 0:
                return
            if status != 1:
                break
            await asyncio.sleep(WAITER_READY_SEC)
        raise RuntimeError(
            "BenchFlow's waiter did not become ready; start a fresh trial"
        )
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
