"""The task's network policy for the oracle's processes, during its turn only.

A task with ``network_mode = "allowlist"`` or ``"denylist"`` confines its agent
(the uid firewall in :mod:`benchflow.sandbox.lockdown` and the loopback proxy
in :mod:`benchflow.sandbox.egress_denylist`), but the oracle ran as root
outside both: under ``network = ["pypi.org"]`` it reached example.com and
github.com, so an oracle that needs a host the task forbids passed and proved
nothing about the task under its declared policy (REPORT B10).

The same policy now covers the oracle's processes while ``solve.sh`` runs
right before it starts, root starts the egress proxy and adds
firewall rules that match only the oracle's processes; right after it ends,
before the verifier, root removes them and stops the proxy. Nothing else is
covered at any point: BenchFlow's own installs (the agent, the in-sandbox
model proxy, iptables itself) run before or outside the oracle's turn, and the
verifier runs after it, so they keep reaching their hosts. Never a
whole-sandbox or a permanent root rule.

How the rules find the oracle's processes:

- A root oracle (the usual case) runs with one more supplementary group,
  :data:`ORACLE_EGRESS_GID`, joined just before ``solve.sh`` starts; the rules
  match that group (``--gid-owner ... --suppl-groups``). Its children inherit
  it; the proxy, ``docker exec`` and every other root process lack it. The
  files it writes keep their usual group (the effective group is unchanged).
- An oracle that runs as a non-root user (the task's ``[agent].user``, as
  ``su`` resets supplementary groups, or the image's ``USER``, which cannot
  join a group) is matched by that user's uid instead.

What it cannot do: an oracle process that leaves the group (``su``, ``sudo``,
``newgrp``) or flushes the rules (root holds NET_ADMIN under these policies)
escapes them. The policy proves solvability for a cooperative reference
solution; it is not a sandbox for a hostile one.
"""

from __future__ import annotations

import contextlib
import ipaddress
import logging
import shlex
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchflow.sandbox._exec_output import exec_raw
from benchflow.sandbox.egress_denylist import (
    DNS_PORT,
    EgressDenylist,
    denylist_agent_env,
    start_egress_denylist,
    stop_egress_denylist,
)
from benchflow.sandbox.lockdown import (
    EGRESS_ALLOW_NETWORKS_ENV,
    EGRESS_ALLOWLIST_ENV,
    EGRESS_DENYLIST_ENV,
    _exec_failure_detail,
    _exec_return_code,
)

logger = logging.getLogger(__name__)

#: The supplementary group a root oracle joins while the policy covers it.
#: Arming refuses an image whose /etc/group already uses it.
ORACLE_EGRESS_GID = 61628

#: Exit status of the arm script when the policy cannot be enforced.
_ARM_FAILED = 86

# The proxy's markers in the agent env; the oracle's processes need only the
# proxy and CA variables.
_AGENT_ONLY_ENV = (EGRESS_DENYLIST_ENV, EGRESS_ALLOWLIST_ENV, EGRESS_ALLOW_NETWORKS_ENV)

# Joins ORACLE_EGRESS_GID, then becomes the command. The interpreter is the
# one the egress proxy itself needs (python3 in the task image).
_JOIN_GROUP_PY = (
    "import os,sys;g=int(sys.argv[1]);"
    "os.setgroups(sorted(set(os.getgroups())|{g}));"
    "os.execvp(sys.argv[2],sys.argv[2:])"
)


@dataclass(frozen=True)
class _Rule:
    """One firewall rule: ``binary -t table -I OUTPUT 1 spec`` (or ``-A``)."""

    binary: str
    table: str
    spec: str
    append: bool = False

    def add(self) -> str:
        where = "-A OUTPUT" if self.append else "-I OUTPUT 1"
        return (
            f"{self.binary} -t {self.table} -C OUTPUT {self.spec} 2>/dev/null || "
            f"{self.binary} -t {self.table} {where} {self.spec}"
        )

    def remove(self) -> str:
        return (
            f"while {self.binary} -t {self.table} -C OUTPUT {self.spec} "
            f"2>/dev/null; do {self.binary} -t {self.table} -D OUTPUT {self.spec} "
            "|| break; done"
        )


def owner_match(*, gid: int | None = None, uid: int | None = None) -> str:
    """The iptables owner match for the oracle's processes (one of the two)."""
    if gid is not None and uid is None:
        return f"-m owner --gid-owner {int(gid)} --suppl-groups"
    if uid is not None and gid is None:
        return f"-m owner --uid-owner {int(uid)}"
    raise ValueError("match the oracle by its group or by its uid, not both")


def oracle_firewall_rules(
    match: str, allow_networks: tuple[str, ...] | None
) -> tuple[list[_Rule], list[_Rule]]:
    """The IPv4 and IPv6 rules, in the order they are added.

    The same rules the agent's uid firewall uses (``_agent_egress_firewall_cmd``
    and ``_allowlist_firewall_cmd`` in lockdown), with the oracle's owner match:
    loopback only (the proxy, the DNS filter, local services), the rest
    refused; under an allowlist, DNS goes to the proxy's filter, Docker's
    embedded resolver is refused, and listed IP ranges are reachable directly.
    """
    v4 = [
        _Rule("iptables", "filter", f"-o lo {match} -j ACCEPT"),
        _Rule("iptables", "filter", f"{match} -j REJECT", append=True),
    ]
    v6 = [
        _Rule("ip6tables", "filter", f"-o lo {match} -j ACCEPT"),
        _Rule("ip6tables", "filter", f"{match} -j REJECT", append=True),
    ]
    if allow_networks is None:
        return v4, v6
    parsed = [ipaddress.ip_network(n, strict=False) for n in allow_networks]
    for proto in ("udp", "tcp"):
        v4 += [
            _Rule(
                "iptables",
                "nat",
                f"-p {proto} --dport 53 {match} -j REDIRECT --to-ports {DNS_PORT}",
            ),
            _Rule(
                "iptables", "filter", f"-o lo -p {proto} --dport 53 {match} -j REJECT"
            ),
            # After the REDIRECT the filter chain does not see the query as
            # "-o lo" (observed on Daytona), so admit the rewritten destination.
            _Rule(
                "iptables",
                "filter",
                f"-d 127.0.0.1 -p {proto} --dport {DNS_PORT} {match} -j ACCEPT",
            ),
        ]
        v6.append(
            _Rule(
                "ip6tables", "filter", f"-o lo -p {proto} --dport 53 {match} -j REJECT"
            )
        )
    v4.append(_Rule("iptables", "filter", f"-d 127.0.0.11 {match} -j REJECT"))
    for net in parsed:
        rule = f"-d {net.compressed} {match} -j ACCEPT"
        if net.version == 4:
            v4.append(_Rule("iptables", "filter", rule))
        else:
            v6.append(_Rule("ip6tables", "filter", rule))
    return v4, v6


def _install_iptables_cmd() -> str:
    """Install iptables when the image has none (before the oracle's turn).

    apt's package lists are left as they were: an image built with them
    removed keeps them removed, so the reference solution sees the image its
    task declares, not one BenchFlow refreshed.
    """
    return (
        "if ! command -v iptables >/dev/null 2>&1; then "
        "if command -v apt-get >/dev/null 2>&1; then "
        "export DEBIAN_FRONTEND=noninteractive; "
        "bf_lists=$(ls /var/lib/apt/lists 2>/dev/null | grep -c _Packages || true); "
        "apt-get update -qq && apt-get install -y -qq iptables >/dev/null; "
        '[ "$bf_lists" -gt 0 ] || rm -rf /var/lib/apt/lists/*; '
        "elif command -v dnf >/dev/null 2>&1; then "
        "dnf -y install iptables >/dev/null; "
        "elif command -v apk >/dev/null 2>&1; then "
        "apk add --no-cache iptables >/dev/null; "
        f"else echo 'No supported iptables package manager' >&2; exit {_ARM_FAILED}; "
        "fi; fi; "
    )


def arm_cmd(
    match: str, allow_networks: tuple[str, ...] | None, *, gid: int | None
) -> str:
    """Root's script: install iptables if needed, then add the oracle's rules.

    Fails closed (exit 86) when the policy cannot be enforced: no iptables, no
    nat table under an allowlist, IPv6 without ip6tables, or the oracle group's
    number already taken in the image.
    """
    v4, v6 = oracle_firewall_rules(match, allow_networks)
    parts = ["set -e; ", _install_iptables_cmd()]
    if gid is not None:
        parts.append(
            f"if grep -q '^[^:]*:[^:]*:{int(gid)}:' /etc/group 2>/dev/null; then "
            f"echo 'group id {int(gid)} is taken in the task image; the oracle "
            f"cannot be matched by its group' >&2; exit {_ARM_FAILED}; fi; "
        )
    if allow_networks is not None:
        parts.append(
            "iptables -t nat -L OUTPUT -n >/dev/null 2>&1 || "
            "{ echo 'iptables nat table unavailable: network_mode=allowlist cannot "
            f"filter DNS' >&2; exit {_ARM_FAILED}; }}; "
        )
    parts.extend(f"{rule.add()}; " for rule in v4)
    parts.append(
        "if [ -e /proc/net/if_inet6 ]; then "
        "command -v ip6tables >/dev/null 2>&1 || "
        f"{{ echo 'IPv6 enabled but ip6tables unavailable' >&2; exit {_ARM_FAILED}; }}; "
        + "".join(f"{rule.add()}; " for rule in v6)
        + "fi"
    )
    return "".join(parts)


def disarm_cmd(match: str, allow_networks: tuple[str, ...] | None) -> str:
    """Root's script: remove every rule :func:`arm_cmd` added (each copy)."""
    v4, v6 = oracle_firewall_rules(match, allow_networks)
    v4_cmds = "; ".join(rule.remove() for rule in reversed(v4))
    v6_cmds = "; ".join(rule.remove() for rule in reversed(v6))
    return (
        f"command -v iptables >/dev/null 2>&1 || exit 0; {v4_cmds}; "
        "if [ -e /proc/net/if_inet6 ] && command -v ip6tables >/dev/null 2>&1; "
        f"then {v6_cmds}; fi"
    )


def join_group_prefix(gid: int) -> str:
    """Shell words that run the rest of the command in the oracle's group."""
    return (
        '"$(command -v python3 || command -v python)" -I -c '
        f"{shlex.quote(_JOIN_GROUP_PY)} {int(gid)} "
    )


def join_group_probe_cmd(gid: int, run_as: str | None) -> str:
    """The oracle's own join, ending in ``id -G`` instead of ``solve.sh``.

    Built the way ``_run_oracle`` builds the oracle's command (inside ``su``
    for a task user, since su resets supplementary groups), and run the way
    it runs it (the sandbox's default user), so a join that would fail for
    solve.sh fails here first.
    """
    probe = f"{join_group_prefix(gid)}id -G"
    if run_as:
        return f"su -s /bin/bash {shlex.quote(run_as)} -c {shlex.quote(probe)}"
    return probe


@dataclass(frozen=True)
class OracleEgress:
    """An armed policy: how the oracle's command joins it, and how to undo it."""

    policy: EgressDenylist
    match: str
    #: The group a root oracle joins; None when its uid is matched instead.
    gid: int | None
    #: The proxy and CA variables for the oracle's processes.
    env: dict[str, str] = field(default_factory=dict)

    @property
    def allow_networks(self) -> tuple[str, ...] | None:
        return self.policy.allow_networks if self.policy.allow_mode else None


async def _oracle_uid(env: Any, user: str | None) -> int:
    """The uid solve.sh runs as: ``user``'s, or the sandbox's default user's."""
    if user:
        result = await exec_raw(
            env.exec, f"id -u {shlex.quote(user)}", user="root", timeout_sec=30
        )
    else:
        # The oracle's own exec names no user: the image's USER (root unless
        # the image sets another), so ask as that same default.
        result = await exec_raw(env.exec, "id -u", timeout_sec=30)
    text = (getattr(result, "stdout", "") or "").strip()
    if _exec_return_code(result) != 0 or not text.isdigit():
        who = repr(user) if user else "(the image's default user)"
        raise RuntimeError(
            f"network_mode applies to the oracle: cannot find the uid of its "
            f"user {who}.{_exec_failure_detail(result)}"
        )
    return int(text)


async def arm_oracle_egress(
    env: Any, policy: EgressDenylist, *, run_as: str | None
) -> OracleEgress:
    """Start the egress proxy and the oracle's rules, right before ``solve.sh``.

    ``run_as`` is the user the oracle runs as (None: the sandbox's default
    user, which is the image's USER).
    Raises ``RuntimeError`` (nothing left armed) when the policy cannot be
    enforced in this sandbox: the oracle never runs unconfined under a policy.
    """
    uid = await _oracle_uid(env, run_as)
    gid = ORACLE_EGRESS_GID if uid == 0 else None
    match = owner_match(gid=gid) if gid is not None else owner_match(uid=uid)
    armed = OracleEgress(
        policy=policy,
        match=match,
        gid=gid,
        env={
            key: value
            for key, value in denylist_agent_env({}, policy).items()
            if key not in _AGENT_ONLY_ENV
        },
    )
    try:
        try:
            # The proxy runs as root either way; the name only satisfies its check.
            await start_egress_denylist(env, run_as or "root", policy)
        except RuntimeError as exc:
            raise RuntimeError(
                f"network_mode={policy.mode!r} applies to the oracle, but its "
                f"egress proxy could not start: {exc}"
            ) from exc
        result = await exec_raw(
            env.exec,
            arm_cmd(match, armed.allow_networks, gid=gid),
            user="root",
            timeout_sec=120,
        )
        if _exec_return_code(result) != 0:
            raise RuntimeError(
                f"network_mode={policy.mode!r} applies to the oracle, but its "
                f"firewall could not be armed.{_exec_failure_detail(result)}"
            )
        if gid is not None:
            await _check_group_join(env, policy, gid, run_as)
    except BaseException:
        with tempfile.TemporaryDirectory() as scratch:
            await _disarm(env, armed, Path(scratch))
        raise
    logger.info(
        "Oracle egress policy armed (%s, %s)", policy.mode, match.split("--", 1)[1]
    )
    return armed


async def _check_group_join(
    env: Any, policy: EgressDenylist, gid: int, run_as: str | None
) -> None:
    """Refuse a sandbox where the root oracle cannot join ``gid``.

    ``setgroups`` needs CAP_SETGID: a task compose that drops it (``cap_drop:
    [ALL]``, with BenchFlow's NET_ADMIN added back) arms the rules but makes
    the join exit before solve.sh starts, and the verifier would score the
    untouched workspace as an oracle failure (R3-R1-A-2). The trial errors
    instead.
    """
    result = await exec_raw(env.exec, join_group_probe_cmd(gid, run_as), timeout_sec=30)
    groups = (getattr(result, "stdout", "") or "").split()
    if _exec_return_code(result) != 0 or str(int(gid)) not in groups:
        raise RuntimeError(
            f"network_mode={policy.mode!r} applies to the oracle, but its "
            f"processes cannot join group {int(gid)}, which the policy's rules "
            "match (joining needs CAP_SETGID; does the task drop it?)."
            f"{_exec_failure_detail(result)}"
        )


async def _disarm(env: Any, armed: OracleEgress, rollout_dir: Path) -> int:
    """Remove the rules, then stop the proxy; the rules' exit status (-1: no answer)."""
    code = -1
    with contextlib.suppress(Exception):
        result = await exec_raw(
            env.exec,
            disarm_cmd(armed.match, armed.allow_networks),
            user="root",
            timeout_sec=60,
        )
        code = _exec_return_code(result)
    # Never raises; also takes the proxy's block log.
    await stop_egress_denylist(env, rollout_dir)
    return code


async def disarm_oracle_egress(
    env: Any, armed: OracleEgress, rollout_dir: Path | None
) -> None:
    """Remove the oracle's rules and stop the proxy, right after ``solve.sh``.

    The proxy's block log goes to ``<rollout_dir>/trajectory/egress_denylist.jsonl``
    (what the oracle tried to reach and was refused). Raises ``RuntimeError``
    when the rules could not be removed, so the verifier never runs under them.
    """
    if rollout_dir is None:
        with tempfile.TemporaryDirectory() as scratch:
            code = await _disarm(env, armed, Path(scratch))
    else:
        code = await _disarm(env, armed, rollout_dir)
    if code != 0:
        raise RuntimeError(
            "the oracle's network policy could not be removed before the "
            f"verifier (exit {code})"
        )
