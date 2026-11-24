"""Every compose service's network namespace, covered from BenchFlow's helper.

The cloud metadata block and the agent's network policy were rules in
``main``'s namespace only. A task's other compose services (a database, a
web app the agent drives) have namespaces of their own, Docker NATs their
traffic out through the host, and their processes run as root: agent code that
reached one (through the service itself, or the service's own requests on
the agent's behalf) reached the VM's metadata server and any host.

Now, before the agent starts, every running service of the project but
``main`` gets rules in its own namespace, added from a BenchFlow helper
container that joins it (``docker run --network container:<service>``, the
helper of :mod:`benchflow.sandbox._metadata_helper`), so nothing is installed
in the service and its processes cannot remove them unless the task gives it
NET_ADMIN:

- **Metadata.** TCP to every cloud metadata endpoint but port 53, from every
  user, root included (:func:`benchflow.sandbox.lockdown._cloud_metadata_rule`
  with ``cover_root``), unless the task opts out of the block.
- **Egress**, by the agent's network policy (``[agent]`` override included:
  the agent drives these services). ``no-network``, and ``denylist`` while
  :data:`DENYLIST_SIDECARS_FAIL_CLOSED` holds, unless the
  task names the service in ``metadata.x-service-egress``: loopback, replies
  to connections made to the service, and the project's own networks (so the
  services still reach each other), but never a network's gateway (the
  host); the rest refused. ``allowlist``: the same, plus the allowlist's IP
  entries; its host names need the DNS filter and proxy, which run in
  ``main`` only (per-service host names are 's, post-rc). ``public``:
  no egress rule.
- **Refused trials.** A service the task gives NET_ADMIN or NET_RAW
  (``cap_add``, or ``privileged``) could remove its rules: the trial is
  refused unless the task opts out of the metadata block. So is a service on
  the host's network (its rules would be the host's own firewall).
- **main** gets the metadata rule the same way, for every user, root
  included, from right after the sandbox starts; only the task's own opt-out
  (``allow_cloud_metadata`` / ``x-allow-cloud-metadata``) lifts it. Its
  egress stays the agent's policy (the uid firewall and proxy).
- **What came before.** Right before its rules go in, the helper reads the
  namespace's socket tables (``/proc/net/tcp``, ``tcp6``, ``udp``, ``udp6``,
  and ``nf_conntrack`` where the kernel has it): every socket to a
  destination outside loopback is recorded with its state, beside the
  container's start time and the coverage time, so a connection task
  start-up code made before coverage shows in the record.
- **Proof**, per service, from the helper as uid 0 in its namespace: a
  connect to 169.254.169.254:80 refused and counted by the metadata rule, and
  where egress rules hold, one to TEST-NET-1 refused and counted by them.
  ``service-network.json`` in the trial folder lists every service with its
  rules, counters and proof, so a later scan can tell from the record alone
  which trials were covered.
"""

from __future__ import annotations

import contextlib
import ipaddress
import json
import logging
import re
import shlex
from typing import Any

logger = logging.getLogger(__name__)

#: The lead's ruling (2026-11-14): a denylist task's services get what a
#: no-network task's get unless the task names them in
#: ``metadata.x-service-egress``. False keeps them to the metadata rule alone.
DENYLIST_SIDECARS_FAIL_CLOSED = True
#: The task's list of services that need external egress.
SERVICE_EGRESS_KEY = "x-service-egress"
#: The trial folder's record of every service's coverage.
SERVICE_NETWORK_RECORD = "service-network.json"
_PROOF_ADDRESS = "192.0.2.1"
_NET_CAPS = {"NET_ADMIN", "NET_RAW", "CAP_NET_ADMIN", "CAP_NET_RAW", "ALL"}


class ServiceCoverError(RuntimeError):
    """A service of the trial could not be covered: the trial does not start."""


def sandbox_network_mode(sandbox_config: Any) -> str:
    """A sandbox config's network: ``allow_internet = false`` is ``no-network``."""
    if getattr(sandbox_config, "allow_internet", True) is False:
        return "no-network"
    mode = getattr(sandbox_config, "network_mode", None)
    return str(getattr(mode, "value", mode) or "public")


def _compose_services(environment_dir: Any) -> dict[str, Any]:
    """The task's own compose file's services, read from the file."""
    from pathlib import Path

    if not isinstance(environment_dir, Path):
        return {}
    path = environment_dir / "docker-compose.yaml"
    if not path.is_file():
        return {}
    import yaml

    from benchflow.sandbox.cloud_metadata import _compose_loader

    try:
        loaded = yaml.load(path.read_text(encoding="utf-8"), Loader=_compose_loader())
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return {}
    services = (loaded or {}).get("services") if isinstance(loaded, dict) else None
    return services if isinstance(services, dict) else {}


def _compose_networks(environment_dir: Any) -> dict[str, Any]:
    """The task's own compose file's top-level networks, read from the file."""
    from pathlib import Path

    if not isinstance(environment_dir, Path):
        return {}
    path = environment_dir / "docker-compose.yaml"
    if not path.is_file():
        return {}
    import yaml

    from benchflow.sandbox.cloud_metadata import _compose_loader

    try:
        loaded = yaml.load(path.read_text(encoding="utf-8"), Loader=_compose_loader())
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return {}
    networks = (loaded or {}).get("networks") if isinstance(loaded, dict) else None
    return networks if isinstance(networks, dict) else {}


def _plain_caps(caps: Any) -> set[str]:
    """Capability names without ``CAP_``, upper case."""
    if isinstance(caps, str):
        caps = [caps]
    return {
        str(c).strip().upper().removeprefix("CAP_")
        for c in caps or ()
        if isinstance(c, str)
    }


def _powers(caps: Any, privileged: Any, network_mode: Any) -> list[str]:
    held = sorted(_plain_caps(caps) & {"NET_ADMIN", "NET_RAW", "ALL"})
    if privileged in (True, "true", "True"):
        held.append("privileged")
    if str(network_mode or "").strip() == "host":
        held.append("the host's network")
    return held


def network_powers(services: Any) -> dict[str, list[str]]:
    """Each Compose service (a ``services`` mapping) given a way around the
    trial's rules: NET_ADMIN or NET_RAW in ``cap_add`` (or ALL),
    ``privileged``, or the host's network."""
    powers: dict[str, list[str]] = {}
    for name, service in (services or {}).items() if isinstance(services, dict) else ():
        if not isinstance(service, dict):
            continue
        held = _powers(
            service.get("cap_add"),
            service.get("privileged"),
            service.get("network_mode"),
        )
        if held:
            powers[str(name)] = held
    return powers


def compose_network_powers(environment_dir: Any) -> dict[str, list[str]]:
    """:func:`network_powers` of the task's raw ``docker-compose.yaml``: for a
    sandbox whose Compose BenchFlow does not resolve here (a provider's)."""
    return network_powers(_compose_services(environment_dir))


def ask_start_holds(
    sandbox: Any,
    task_config: Any,
    environment_dir: Any,
    *,
    every_process: bool,
    mode: str | None = None,
) -> None:
    """Ask a Docker sandbox's start for the holds that keep every process of
    the trial under its network policy from its first instruction:
    every compose service waits in the inert bootstrap and is covered before
    release, and, where *every_process* (nothing of BenchFlow's in main needs
    the network: the oracle, a ``script:`` agent, ``nop``, a separate
    verifier sandbox), main's every-process rules go in before release too.
    *mode* is the policy (default: the agent's). Under an open network the
    services' cloud metadata rule is the reason: every service waits until its
    rule is in and proven, so no start-up code of a task's service reaches the
    endpoints (0.10.0 let them run first); the task's granted opt-out asks for
    nothing. Set on the sandbox before its start; a sandbox without the
    inert bootstrap ignores it."""
    from benchflow.sandbox.egress_denylist import (
        agent_network_sandbox_config,
        egress_denylist_for,
    )

    if mode is None:
        mode = sandbox_network_mode(agent_network_sandbox_config(task_config))
    opt_out = getattr(task_config, "cloud_metadata_opt_out", "")
    task_opts_out = isinstance(opt_out, str) and bool(opt_out)
    if mode not in ("allowlist", "denylist", "no-network"):
        if task_opts_out or not callable(
            getattr(type(sandbox), "metadata_service_targets", None)
        ):
            # Nothing to wait for: the opt-out is granted, or the backend puts
            # no rule into its services (Daytona's DinD; the 0.10.1 notes).
            return
        with contextlib.suppress(AttributeError):
            sandbox._hold_every_service = True
            sandbox._start_cover = {
                "task_config": task_config,
                "mode": mode,
                "metadata": True,
                "opt_out": False,
                "environment_dir": environment_dir,
            }
        return
    with contextlib.suppress(AttributeError):
        sandbox._hold_every_service = True
        sandbox._start_cover = {
            "task_config": task_config,
            "mode": mode,
            "metadata": not task_opts_out,
            "opt_out": task_opts_out,
            "environment_dir": environment_dir,
        }
        if every_process:
            sandbox._start_hold_mode = mode
            if mode == "allowlist":
                policy = egress_denylist_for(agent_network_sandbox_config(task_config))
                sandbox._start_hold_allow = tuple(
                    getattr(policy, "allow_networks", ()) or ()
                )


def holds_main_from_release(agent: str | None, *, verifier_only: bool = False) -> bool:
    """Whether nothing of BenchFlow's in main needs the network for *agent*
    (the oracle, ``nop``, a ``script:`` agent), so main is held from its
    release. A model agent's install and model proxy need it, and so may an agent not named yet. A verifier-only rollout (a
    regrade, a verifier recovery: an oracle stand-in that never runs
    solve.sh) runs no agent at all, so its main is held from release as a
    normal oracle trial's is (L4), whatever agent it names."""
    if verifier_only:
        return True
    if agent is None:
        return False
    from benchflow.agents.registry import is_operator_script

    return agent in ("oracle", "nop") or is_operator_script(agent)


async def start_sandbox(
    sandbox: Any,
    task: Any,
    *,
    agent: str | None,
    verifier_only: bool = False,
    separate_verifier: bool = False,
    force_build: bool = False,
    timeout_sec: float | None = None,
    agent_uid: dict[str, Any] | None = None,
) -> None:
    """Start a trial's sandbox under the start holds: the one place BenchFlow
    starts one. Every start path comes here (a rollout's through
    ``_start_env_and_upload``, the equivalence battery's and acceptance-live's
    through it too, ``Environment.start`` and so ``Runtime.execute``, a
    separate verifier sandbox's), so :func:`ask_start_holds` is asked before
    every start; ``tests/test_en698_start_paths.py`` fails on a sandbox
    start anywhere else in src.

    *agent* is what runs in main (:func:`holds_main_from_release`); None
    for a model agent's or an agent not named yet (an ``Environment`` its
    caller starts), whose services are held and whose main keeps its open
    start. *separate_verifier*: the verifier sandbox, every process held
    under the verifier's network. *timeout_sec* bounds the whole start.
    *agent_uid*: a model agent's uid rules before main's release (
    M1, the rollout's ``_agent_uid_start_hold``), on a Docker sandbox."""
    import asyncio

    config = getattr(task, "config", None)
    environment_dir = getattr(getattr(task, "paths", None), "environment_dir", None)
    policy = restricted_policy(config)
    if policy:
        # Below Docker Engine 29 / Compose 2.40 the policy does not hold
        # main's ENTRYPOINT from its first instruction.
        from benchflow.sandbox._compose_runtime import old_docker_policy_refusal

        refusal = await old_docker_policy_refusal(sandbox, policy)
        if refusal is not None:
            raise ServiceCoverError(refusal)
    if separate_verifier:
        from benchflow.sandbox.verifier_network import separate_verifier_network

        ask_start_holds(
            sandbox,
            config,
            environment_dir,
            every_process=True,
            mode=separate_verifier_network(config),
        )
    else:
        ask_start_holds(
            sandbox,
            config,
            environment_dir,
            every_process=holds_main_from_release(agent, verifier_only=verifier_only),
        )
    if agent_uid is not None and not separate_verifier:
        with contextlib.suppress(AttributeError):
            sandbox._start_uid_hold = dict(agent_uid)
    start = sandbox.start(force_build=force_build)
    if timeout_sec is None:
        await start
    else:
        await asyncio.wait_for(start, timeout_sec)


def caller_start_refusal(
    sandbox: Any, task: Any, *, agent: str | None, verifier_only: bool = False
) -> str | None:
    """Why a run is refused in a sandbox its caller started (``Environment.
    start`` before :meth:`~benchflow.runtime.Environment.prepare_kept_services`
    named the agent), or None: the run holds main from release
    (:func:`holds_main_from_release`) under the task's network policy, main
    has a route out, the start did not hold it, and the image's ENTRYPOINT
    started there, outside the policy. Docker sandboxes only (the
    others start nothing of the task's in main)."""
    from benchflow.sandbox.egress_denylist import agent_network_sandbox_config

    if not holds_main_from_release(agent, verifier_only=verifier_only):
        return None
    config = getattr(task, "config", None)
    if getattr(getattr(config, "sandbox", None), "allow_internet", True) is False:
        return None  # main's network is none from its creation
    mode = sandbox_network_mode(agent_network_sandbox_config(config))
    if mode not in ("allowlist", "denylist", "no-network"):
        return None
    held = getattr(sandbox, "_start_hold_mode", "")
    if isinstance(held, str) and held:
        return None
    if not callable(getattr(type(sandbox), "metadata_rules_install", None)):
        return None
    from benchflow.sandbox.lockdown import _image_entrypoint

    entry = " ".join(_image_entrypoint(sandbox))[:120]
    if not entry:
        return None
    return (
        "sandbox startup refused: this sandbox was started before the run named "
        f"its agent, so main's image ENTRYPOINT ({entry}) ran outside the task's "
        f"network policy ({mode}) from its release, and a {agent} run holds main "
        "from its first instruction. Call Environment.prepare_kept_services(agent) "
        "before Environment.start(), or let Runtime.execute start the environment"
    )


def restricted_policy(task_config: Any) -> str:
    """The network policy that holds every service ("" when the agent's and
    the verifier's networks are both public): allowlist, denylist,
    no-network, or the verifier's (offline)."""
    from benchflow.sandbox.egress_denylist import agent_network_sandbox_config
    from benchflow.sandbox.verifier_network import verifier_network_mode

    agent = sandbox_network_mode(agent_network_sandbox_config(task_config))
    verifier = verifier_network_mode(task_config)
    if agent == "public" and verifier == "public":
        return ""
    return agent if agent != "public" else f"{verifier}, the verifier's"


#: What the resolved model is called in a refusal.
RESOLVED_MODEL = "the task's Compose project as the run resolves it"


def powers_refusal(
    policy: str, powers: dict[str, list[str]], *, source: str = RESOLVED_MODEL
) -> str | None:
    """Why the trial is refused before anything starts, or None: a network
    policy holds and a service has a way around its rules. Whatever the cloud
    metadata opt-out says: it lifts only the metadata rule, never the network
    policy (the lead's 03:28Z ruling)."""
    if not policy or not powers:
        return None
    named = "; ".join(f"{s} {', '.join(p)}" for s, p in sorted(powers.items()))
    return (
        f"sandbox startup refused: the task's network policy ({policy}) holds "
        f"every service, and {source} gives {named}, so root there could remove "
        "or go around those rules. The cloud metadata opt-out lifts only the "
        "metadata rule, never the network policy, and "
        "BENCHFLOW_ALLOW_TASK_HOST_ACCESS and BENCHFLOW_ALLOW_HOST_ACCESS_ON_CLOUD "
        "allow host access only under an open network: drop the capability"
    )


def policy_power_refusal(task_config: Any, environment_dir: Any) -> str | None:
    """:func:`powers_refusal` from the task's raw compose file: only for a
    sandbox whose Compose BenchFlow does not resolve here. A Docker sandbox
    checks the model its run resolves (``DockerSandbox.freeze_launch``)."""
    return powers_refusal(
        restricted_policy(task_config),
        compose_network_powers(environment_dir),
        source="the task's compose file",
    )


def running_power_drift(
    targets: list[dict[str, Any]], project: dict[str, Any]
) -> dict[str, list[str]]:
    """What each running service holds (its NetworkMode, CapAdd and
    Privileged, as the daemon reports them) that the resolved model that was
    checked and frozen did not give it: a check of the launch, from outside."""
    services = project.get("services") if isinstance(project, dict) else None
    services = services if isinstance(services, dict) else {}
    drift: dict[str, list[str]] = {}
    for target in targets:
        name = str(target.get("service") or "")
        seen = _powers(
            target.get("cap_add"), target.get("privileged"), target.get("network_mode")
        )
        checked = set(network_powers({name: services.get(name) or {}}).get(name, []))
        extra = [p for p in seen if p not in checked]
        if extra:
            drift[name] = extra
    return drift


def main_compose_capabilities(environment_dir: Any) -> list[str]:
    """What the task's own compose file gives ``main`` that would let root
    there remove its rules: NET_ADMIN or NET_RAW in ``cap_add`` (or ALL), or
    ``privileged``. Read from the file, not the container, which BenchFlow's
    own overlay may give NET_ADMIN for the agent's firewall."""
    main = _compose_services(environment_dir).get("main")
    if not isinstance(main, dict):
        return []
    caps = main.get("cap_add") or []
    held = sorted({str(c).upper() for c in caps if isinstance(c, str)} & _NET_CAPS)
    if main.get("privileged") in (True, "true", "True"):
        held.append("privileged")
    return held


def service_egress_names(task_config: Any) -> tuple[str, ...]:
    """The services the task says need external egress."""
    raw = (getattr(task_config, "metadata", None) or {}).get(SERVICE_EGRESS_KEY)
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list) or not all(isinstance(n, str) for n in raw):
        return ()
    return tuple(raw)


def egress_kind(mode: str, service: str, opened: tuple[str, ...]) -> str:
    """Which egress rules a service gets: ``closed`` (loopback and the
    project), ``allowlist`` (closed plus the allowlist's IPs), ``open``."""
    if service in opened:
        return "open"
    if mode == "no-network":
        return "closed"
    if mode == "denylist":
        return "closed" if DENYLIST_SIDECARS_FAIL_CLOSED else "open"
    if mode == "allowlist":
        return "allowlist"
    return "open"


def _metadata_rules() -> tuple[list[str], list[str]]:
    from benchflow.sandbox.cloud_metadata import CLOUD_METADATA_V4, CLOUD_METADATA_V6
    from benchflow.sandbox.lockdown import _cloud_metadata_rule

    return (
        [_cloud_metadata_rule(a, cover_root=True) for a in CLOUD_METADATA_V4],
        [_cloud_metadata_rule(a, cover_root=True) for a in CLOUD_METADATA_V6],
    )


def service_rules(
    *,
    metadata: bool,
    egress: str,
    subnets: tuple[str, ...],
    gateways: tuple[str, ...],
    allow_networks: tuple[str, ...] = (),
    refuse_docker_dns: bool = False,
) -> tuple[list[str], list[str]]:
    """The IPv4 and IPv6 rules for one service, top of OUTPUT first.

    *refuse_docker_dns* (a closed service where test.sh runs: the verifier
    phase's ``verifier.service``): Docker's resolver refused too
    (verifier_network.DOCKER_DNS_REFUSAL)."""
    v4: list[str] = []
    v6: list[str] = []
    if metadata:
        m4, m6 = _metadata_rules()
        v4 += m4
        v6 += m6
    if egress in ("closed", "allowlist"):
        nets = [ipaddress.ip_network(n, strict=False) for n in subnets]
        allowed = [ipaddress.ip_network(n, strict=False) for n in allow_networks]
        for gw in gateways:
            addr = ipaddress.ip_address(gw)
            (v4 if addr.version == 4 else v6).append(f"-d {addr} -j REJECT")
        if refuse_docker_dns and egress == "closed":
            from benchflow.sandbox.verifier_network import DOCKER_DNS_REFUSAL

            v4.append(DOCKER_DNS_REFUSAL)
        for table in (v4, v6):
            table.append("-o lo -j ACCEPT")
            table.append("-m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT")
        for net in [*nets, *(allowed if egress == "allowlist" else [])]:
            (v4 if net.version == 4 else v6).append(f"-d {net.compressed} -j ACCEPT")
        v4.append("-j REJECT")
        v6.append("-j REJECT")
    return v4, v6


def sockets_cmd() -> str:
    """The helper's read of the namespace's sockets before its rules: one
    ``BF service-socket <proto> <remote> <state>`` line per socket (hex, as
    /proc/net prints them), and ``BF service-conntrack ...`` per tracked
    flow where the kernel lists them."""
    return (
        "for p in tcp tcp6 udp udp6; do [ -r /proc/net/$p ] && "
        "awk -v p=$p 'NR > 1 { print \"BF service-socket\", p, $3, $4 }' /proc/net/$p; done; "
        "[ -r /proc/net/nf_conntrack ] && head -n 200 /proc/net/nf_conntrack | "
        "sed 's/^/BF service-conntrack /'; "
    )


_TCP_STATES = {
    "01": "established",
    "02": "syn-sent",
    "03": "syn-recv",
    "04": "fin-wait1",
    "05": "fin-wait2",
    "06": "time-wait",
    "07": "close",
    "08": "close-wait",
    "09": "last-ack",
    "0A": "listen",
    "0B": "closing",
}


def _hex_endpoint(text: str) -> tuple[str, int] | None:
    host, _, port = text.partition(":")
    try:
        raw = bytes.fromhex(host)
        if len(raw) == 4:
            addr = ipaddress.IPv4Address(raw[::-1])
        elif len(raw) == 16:
            addr = ipaddress.IPv6Address(
                b"".join(raw[i : i + 4][::-1] for i in range(0, 16, 4))
            )
        else:
            return None
        return str(addr), int(port, 16)
    except ValueError:
        return None


def seen_before_coverage(
    out: str, subnets: tuple[str, ...] = ()
) -> list[dict[str, Any]]:
    """The sockets and tracked flows in *out* (:func:`sockets_cmd`) to a
    destination other than loopback or the unspecified address; each says
    whether it left the project's own networks (*subnets*)."""
    nets = [ipaddress.ip_network(n, strict=False) for n in subnets]
    seen: list[dict[str, Any]] = []

    def add(proto: str, addr: str, port: int, state: str) -> None:
        ip = ipaddress.ip_address(addr)
        if ip.is_unspecified or ip.is_loopback:
            return
        mapped = getattr(ip, "ipv4_mapped", None)
        inside = any(
            (mapped or ip) in n for n in nets if n.version == (mapped or ip).version
        )
        entry: dict[str, Any] = {
            "proto": proto,
            "remote": f"{addr}:{port}",
            "state": state,
        }
        if not inside:
            entry["outside_project"] = True
        if entry not in seen and len(seen) < 50:
            seen.append(entry)

    for line in (out or "").splitlines():
        parts = line.split()
        if line.startswith("BF service-socket ") and len(parts) >= 5:
            found = _hex_endpoint(parts[3])
            if found:
                state = (
                    _TCP_STATES.get(parts[4].upper(), parts[4])
                    if "tcp" in parts[2]
                    else "open"
                )
                add(parts[2], found[0], found[1], state)
        elif line.startswith("BF service-conntrack "):
            dst = re.search(r"\bdst=(\S+)", line)
            dport = re.search(r"\bdport=(\d+)", line)
            proto = parts[4] if len(parts) > 4 else "?"
            if dst:
                with contextlib.suppress(ValueError):
                    add(
                        f"conntrack-{proto}",
                        dst[1],
                        int(dport[1]) if dport else 0,
                        "tracked",
                    )
    return seen


def install_cmd(v4: list[str], v6: list[str]) -> str:
    """The helper's script: read the namespace's sockets (:func:`sockets_cmd`),
    add each rule once, in order (it brings its own iptables), then print the
    counters the proof reads."""

    def adds(tool: str, rules: list[str]) -> str:
        out = ""
        for position, rule in enumerate(rules, start=1):
            q = shlex.quote(rule)
            out += (
                f"r={q}; {tool} -w -C OUTPUT $r 2>/dev/null || "
                f"{tool} -w -I OUTPUT {position} $r || status=1; "
            )
        return out

    return (
        sockets_cmd()
        + "status=0; "
        + adds("iptables", v4)
        + adds("ip6tables", v6)
        + 'echo "BF service-rules status=$status"; '
        + counters_cmd()
    )


def counters_cmd() -> str:
    """The packets the metadata rules and the final REJECT refused. Its line
    carries the read's ``status=``, which a fresh run from the helper reads
    as its report (``_metadata_helper._status``)."""
    return (
        "o=$(iptables -w -S OUTPUT -v 2>/dev/null); s=$?; "
        'printf \'%s\\n\' "$o" | awk -v s="$s" \''
        '{ n = 0; for (i = 1; i < NF; i++) if ($i == "-c") n = $(i + 1) } '
        "/-d 169\\.254\\.169\\.254\\/32 / && /tcp-reset/ { m += n } "
        "/^-A OUTPUT -c [0-9]+ [0-9]+ -j REJECT/ { e += n } "
        'END { printf "BF service-counters metadata=%d egress=%d status=%s\\n", m, e, s }\''
    )


def connect_cmd(address: str) -> str:
    """A TCP connect and nothing more, from the helper (BusyBox wget on an
    ftp:// URL, as the block's own helper connect)."""
    return (
        "conn=failed; "
        f"err=$(timeout 5 env -i /usr/bin/wget -q -Y off -T 3 -O /dev/null ftp://{address}:80/ 2>&1); rc=$?; "
        "case $err in *[Rr]efused*) conn=refused;; *'timed out'*) conn=timeout;; "
        "*[Uu]nreachable*) conn=unreachable;; "
        "*'error getting response'*|*'unexpected server response'*) conn=reached;; "
        "*) case $rc in 0) conn=reached;; 124|143) conn=timeout;; esac;; esac; "
        'printf "BF service-connect %s\\n" "$conn"'
    )


def _counters(out: str) -> tuple[int, int] | None:
    m = re.search(r"BF service-counters metadata=(\d+) egress=(\d+)", out or "")
    return (int(m[1]), int(m[2])) if m else None


def _connect(out: str) -> str:
    m = re.search(r"BF service-connect (\w+)", out or "")
    return m[1] if m else ""


async def cover_task_services(
    env: Any,
    task_config: Any,
    *,
    mode: str,
    metadata: bool,
    opt_out: bool,
    environment_dir: Any = None,
    covered: dict[str, dict[str, Any]] | None = None,
    main_metadata: bool | None = None,
) -> list[dict[str, Any]] | None:
    """Cover every service of the trial but ``main`` (module doc) and return
    their records; None on a sandbox without the helper. Raises
    :class:`ServiceCoverError` when a service cannot be covered. *covered*
    (container ID to record) keeps what an earlier call of the trial covered:
    a container in it is not covered again, and each new one is added.
    *main_metadata* (where the sandbox lists main too): main's own metadata
    rule, for every user, unless the task opts out; its record is first."""
    targets_of = getattr(type(env), "metadata_service_targets", None)
    if not callable(targets_of):
        return None
    held = main_compose_capabilities(environment_dir)
    if held and not opt_out:
        raise ServiceCoverError(
            "sandbox startup refused: the task's compose file gives main "
            f"{', '.join(held)}, so root there could remove the rules that keep "
            "the cloud metadata endpoints and the task's network policy from it; "
            "drop the capability, or set allow_cloud_metadata = true in the "
            "task's [environment] table and run with --allow-cloud-metadata"
        )
    if main_metadata is None:
        targets = await env.metadata_service_targets()
    else:
        targets = await env.metadata_service_targets(include_main=True)
    resolved = getattr(env, "_resolved_compose", None)
    if isinstance(resolved, dict):
        # The containers run what was checked.
        drift = running_power_drift(targets, resolved)
        if drift:
            named = "; ".join(f"{s} {', '.join(p)}" for s, p in sorted(drift.items()))
            raise ServiceCoverError(
                f"sandbox startup refused: the running {named}, which "
                "the Compose project BenchFlow checked and froze for the run "
                "did not give it"
            )
    opened = service_egress_names(task_config)
    allow: tuple[str, ...] = ()
    if mode == "allowlist":
        from benchflow.sandbox.egress_denylist import (
            agent_network_sandbox_config,
            egress_denylist_for,
        )

        policy = egress_denylist_for(agent_network_sandbox_config(task_config))
        allow = tuple(getattr(policy, "allow_networks", ()) or ())
    records: list[dict[str, Any]] = []
    # A main that joins another service's namespace (network_mode:
    # service:<x>, which the daemon reports as container:<x's id>): one
    # namespace, one rule set, main's; <x>'s own
    # rules are not added on top.
    owner = next(
        (
            str(t.get("network_mode") or "").split(":", 1)[1]
            for t in targets
            if t.get("service") == "main"
            and str(t.get("network_mode") or "").startswith("container:")
        ),
        "",
    )
    main_record: dict[str, Any] = {}
    try:
        for target in targets:
            key = str(target.get("id") or "")
            if covered is not None and key in covered:
                records.append(covered[key])
                continue
            if owner and key == owner and target.get("service") != "main":
                record = {
                    "service": str(target.get("service")),
                    "network_mode": target.get("network_mode"),
                    "state": "main's namespace",
                    "why": "main joins this service's network namespace; main's "
                    "rules hold it, for every process there",
                    "proven": bool(main_record.get("proven")),
                }
                if covered is not None and key:
                    covered[key] = record
                records.append(record)
                continue
            if target.get("service") == "main":
                early = getattr(env, "_main_cover_record", None)
                if isinstance(early, dict) and early.get("service") == "main":
                    # Covered at the start (while main still waited in the
                    # inert bootstrap: before_release).
                    record = early
                else:
                    record = await _cover_main(env, target, bool(main_metadata))
                main_record = record
            else:
                record = await _cover_one(
                    env, target, mode, metadata, opt_out, opened, allow
                )
            if covered is not None and key:
                covered[key] = record
            records.append(record)
    finally:
        close = getattr(env, "metadata_service_close", None)
        if callable(close):
            close()
    return records


async def _cover_main(
    env: Any, target: dict[str, Any], metadata: bool
) -> dict[str, Any]:
    """main's metadata rule, for every user, root included, from BenchFlow's
    helper: the agent phase's root
    processes (a task daemon acting on the agent's input) never reach the
    metadata server. Its egress stays the agent's policy. Proven by uid 0's
    refused, counted connect."""
    network = str(target.get("network_mode") or "")
    record: dict[str, Any] = {"service": "main", "network_mode": network}
    if not metadata:
        record.update(state="no rule", why="the task opts out", proven=True)
        return record
    if network == "none":
        record.update(state="no route", proven=True)
        return record
    v4, v6 = service_rules(metadata=True, egress="open", subnets=(), gateways=())
    container = str(target["id"])
    outcome = await env.metadata_service_install(container, install_cmd(v4, v6))
    out = str(getattr(outcome[0], "stdout", "") or "") if outcome else ""
    if outcome is None or outcome[1] or "BF service-rules status=0" not in out:
        why = getattr(env, "metadata_service_unavailable", None)
        detail = why(container) if callable(why) else ""
        raise ServiceCoverError(
            "sandbox startup refused: BenchFlow's helper could not add main's "
            "cloud metadata rule for every user"
            + (f" ({detail})" if isinstance(detail, str) and detail else "")
            + ", so the task's processes would have the cloud metadata endpoints "
            "open to them; only the task's own opt-out (allow_cloud_metadata = "
            "true in its [environment] table), granted by the run's "
            "--allow-cloud-metadata, starts it without the rule"
        )
    _note_seen(record, target, out)
    before = _counters(out) or (0, 0)
    ran, _ = await env.metadata_service_run(
        container, connect_cmd("169.254.169.254"), timeout_sec=30, user="0:0"
    )
    counted, _ = await env.metadata_service_run(
        container, counters_cmd(), timeout_sec=30
    )
    after = _counters(str(getattr(counted, "stdout", "") or "")) or before
    proof = {
        "metadata_connect": _connect(str(getattr(ran, "stdout", "") or "")),
        "metadata_counted": after[0] - before[0],
    }
    ok = _held(proof["metadata_connect"], proof["metadata_counted"])
    if ok:
        from benchflow.sandbox import rule_ledger

        rule_ledger.record(
            env,
            rule_ledger.InstalledRules(
                where="main", phase="start", container=container, v4=v4, v6=v6
            ),
        )
    record.update(
        state="covered" if ok else "unproven",
        rules=len(v4) + len(v6),
        metadata_rule=True,
        every_user=True,
        proof=proof,
        proven=ok,
    )
    if not ok:
        raise ServiceCoverError(
            "sandbox startup refused: main's cloud metadata rule for every user "
            f"is not proven ({json.dumps(proof, sort_keys=True)})"
        )
    return record


def _held(connect: str, counted: int) -> bool:
    """A proof connect that held: refused and counted by the rule, or
    answered "unreachable" by the kernel (a service on an internal network,
    with no route out: nothing leaves, so no rule sees it; RCI on 0650e09)."""
    return (connect == "refused" and counted >= 1) or connect == "unreachable"


def _note_seen(record: dict[str, Any], target: dict[str, Any], out: str) -> None:
    """What the namespace's sockets showed before its rules went in."""
    from datetime import datetime, timezone

    record["started_at"] = target.get("started_at") or None
    record["covered_at"] = datetime.now(timezone.utc).isoformat(timespec="milliseconds")  # noqa: UP017
    record["seen_before_coverage"] = seen_before_coverage(
        out, tuple(target.get("subnets") or ())
    )


async def _cover_one(
    env, target, mode, metadata, opt_out, opened, allow, phase: str = "start"
):
    service = str(target["service"])
    record: dict[str, Any] = {
        "service": service,
        "network_mode": target.get("network_mode"),
    }
    caps = {str(c).upper() for c in target.get("cap_add") or ()}
    held = sorted(caps & _NET_CAPS) + (
        ["privileged"] if target.get("privileged") else []
    )
    if held:
        record["capabilities"] = held
        if not opt_out:
            raise ServiceCoverError(
                f"sandbox startup refused: the task's compose service {service!r} "
                f"holds {', '.join(held)}, so it could remove the rules that keep "
                "the cloud metadata endpoints and the task's network policy from "
                "it; drop the capability, or set allow_cloud_metadata = true in "
                "the task's [environment] table and run with --allow-cloud-metadata"
            )
    network = str(target.get("network_mode") or "")
    if network == "host":
        # No rule of the trial's goes into the host's namespace (RCI
        # 03:59Z); a restricted policy refused this service before start
        # (policy_power_refusal), so only an open network gets here.
        from benchflow.sandbox._metadata_helper import HOST_NETWORK

        record.update(
            state="host network",
            why=HOST_NETWORK.format(what=f"service {service!r}"),
            proven=False,
        )
        return record
    if network == "none":
        record.update(state="no route", proven=True)
        return record
    if network.startswith(("container:", "service:")):
        record.update(
            state="shares a namespace", owner=network.split(":", 1)[1], proven=True
        )
        return record
    egress = egress_kind(mode, service, opened)
    record["egress"] = egress
    if egress == "open" and service in opened:
        record["egress_opened_by"] = f"metadata.{SERVICE_EGRESS_KEY}"
    v4, v6 = service_rules(
        metadata=metadata,
        egress=egress,
        subnets=tuple(target.get("subnets") or ()),
        gateways=tuple(target.get("gateways") or ()),
        allow_networks=allow,
        # test.sh runs here in the verifier phase: no names from Docker's
        # resolver under a closed network (RCI on b8b278706).
        refuse_docker_dns=phase == "verifier",
    )
    if not v4 and not v6:
        record.update(state="no rule", proven=True)
        return record
    container = str(target["id"])
    outcome = await env.metadata_service_install(container, install_cmd(v4, v6))
    if outcome is None:
        raise ServiceCoverError(
            f"sandbox startup refused: BenchFlow's helper could not join the task's "
            f"compose service {service!r} to add its rules"
        )
    result, failed = outcome
    out = str(getattr(result, "stdout", "") or "")
    _note_seen(record, target, out)
    if failed or "BF service-rules status=0" not in out:
        raise ServiceCoverError(
            f"sandbox startup refused: the rules for the task's compose service "
            f"{service!r} could not be added ({failed or 'status not 0'})"
        )
    before = _counters(out) or (0, 0)
    proof: dict[str, Any] = {}
    if metadata:
        ran, _ = await env.metadata_service_run(
            container, connect_cmd("169.254.169.254"), timeout_sec=30, user="0:0"
        )
        proof["metadata_connect"] = _connect(str(getattr(ran, "stdout", "") or ""))
    if egress in ("closed", "allowlist"):
        ran, _ = await env.metadata_service_run(
            container, connect_cmd(_PROOF_ADDRESS), timeout_sec=30, user="0:0"
        )
        proof["egress_connect"] = _connect(str(getattr(ran, "stdout", "") or ""))
    counted, _ = await env.metadata_service_run(
        container, counters_cmd(), timeout_sec=30
    )
    after = _counters(str(getattr(counted, "stdout", "") or "")) or before
    proof["metadata_counted"] = after[0] - before[0]
    proof["egress_counted"] = after[1] - before[1]
    ok = (
        not metadata or _held(proof["metadata_connect"], proof["metadata_counted"])
    ) and (
        egress not in ("closed", "allowlist")
        or _held(proof["egress_connect"], proof["egress_counted"])
    )
    record.update(
        state="covered" if ok else "unproven",
        rules=len(v4) + len(v6),
        metadata_rule=metadata,
        proof=proof,
        proven=ok,
    )
    if ok:
        from benchflow.sandbox import rule_ledger

        rule_ledger.record(
            env,
            rule_ledger.InstalledRules(
                where=f"service {service}",
                phase=phase,
                container=container,
                v4=v4,
                v6=v6,
            ),
        )
    if not ok:
        raise ServiceCoverError(
            f"sandbox startup refused: the rules in the task's compose service "
            f"{service!r} are not proven ({json.dumps(proof, sort_keys=True)})"
        )
    return record
