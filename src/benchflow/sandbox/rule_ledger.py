"""The rules BenchFlow put in a trial's network namespaces, for their re-proof.

Every place that adds rules to a namespace of the trial from BenchFlow's
helper (main at sandbox start, the agent's turn, the oracle's offline turn,
the verifier's network, each compose service, ``verifier.service``) records
here exactly what it added: the rule specs as given to ``iptables -I`` or
``-A`` in the OUTPUT chain (``nat``: the nat table's OUTPUT), the namespace
(``main`` or a service's container), the phase, and when they were last
proven. A later check (:func:`check_cmd`, from the helper joined to that
namespace) asks ``iptables -C`` for each one: a missing rule means the
policy no longer held, which the verdict reports instead of a count of 0.

The ledger lives on the sandbox object (``env._benchflow_rule_ledger``),
never on a copy of the task's config.
"""

from __future__ import annotations

import contextlib
import shlex
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

#: The helper's report line for a check (``_metadata_helper._REPORT_LINES``).
REPORT = "BF rule-check"


@dataclass
class InstalledRules:
    """One namespace's rules from one phase."""

    #: ``main`` or ``service <name>``.
    where: str
    #: ``start`` (sandbox start), ``agent`` (the agent's or the oracle's
    #: turn), ``verifier`` (the verifier's network) or ``verifier-metadata``
    #: (root's cloud metadata cover at the end of the agent's turn).
    phase: str
    #: The container whose namespace the helper joins; None for main through
    #: its own helper (``metadata_rules_install``).
    container: str | None = None
    v4: list[str] = field(default_factory=list)
    v6: list[str] = field(default_factory=list)
    #: IPv4 nat table, OUTPUT chain (allowlist DNS redirect).
    nat: list[str] = field(default_factory=list)
    #: The tools the rules went in with, in the helper (the cloud metadata
    #: block picks iptables, iptables-nft or iptables-legacy).
    tool4: str = "iptables"
    tool6: str = "ip6tables"
    #: When the rules were last proven present and held (ISO 8601, UTC).
    proven_at: str = ""
    #: True once a later phase removed them on purpose (the oracle's
    #: offline rules before the verifier's): no longer expected.
    retired: bool = False

    def as_json(self) -> dict[str, Any]:
        return {
            "where": self.where,
            "phase": self.phase,
            "rules": len(self.v4) + len(self.v6) + len(self.nat),
            "proven_at": self.proven_at,
            **({"retired": True} if self.retired else {}),
        }


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")  # noqa: UP017


def record(env: Any, rules: InstalledRules) -> None:
    """Add *rules* to *env*'s ledger, replacing an earlier entry for the
    same namespace and phase."""
    if not rules.proven_at:
        rules.proven_at = now()
    ledger = entries(env)
    ledger[:] = [
        r for r in ledger if (r.where, r.phase) != (rules.where, rules.phase)
    ] + [rules]
    with contextlib.suppress(AttributeError):
        env._benchflow_rule_ledger = ledger


def retire(env: Any, where: str, phase: str) -> None:
    """Mark a namespace's rules from *phase* as removed on purpose."""
    for r in entries(env):
        if (r.where, r.phase) == (where, phase):
            r.retired = True


def entries(env: Any) -> list[InstalledRules]:
    """*env*'s ledger (the list itself; empty when nothing was recorded)."""
    ledger = getattr(env, "_benchflow_rule_ledger", None)
    return ledger if isinstance(ledger, list) else []


def expected(env: Any, phase: str | None = None) -> list[InstalledRules]:
    """The entries still expected (not retired), of *phase* when given."""
    return [
        r for r in entries(env) if not r.retired and (phase is None or r.phase == phase)
    ]


def capture_cmd(script: str) -> str:
    """*script* (root's rule install, any shell, ``set -e`` allowed) from the
    helper, with what it added: ``BF rule-added <iptables|ip6tables>
    <filter|nat> <-S line>`` for each rule in OUTPUT after it that was not
    there before, then ``BF rules-installed status=<its exit status>``."""
    snap = (
        "bf_snap() { "
        "iptables -w -S OUTPUT 2>/dev/null | sed 's/^/iptables filter /'; "
        "ip6tables -w -S OUTPUT 2>/dev/null | sed 's/^/ip6tables filter /'; "
        "iptables -w -t nat -S OUTPUT 2>/dev/null | sed 's/^/iptables nat /'; "
        "}; "
    )
    return (
        snap
        + "bf_snap > /tmp/bf-rules-before; "
        + "( "
        + script
        + "\n); bf_rc=$?; "
        + "bf_snap > /tmp/bf-rules-after; "
        + "grep -vxF -f /tmp/bf-rules-before /tmp/bf-rules-after "
        + "| sed 's/^/BF rule-added /'; "
        + 'echo "BF rules-installed status=$bf_rc"'
    )


def added_rules(out: str) -> tuple[list[str], list[str], list[str]]:
    """The (v4, v6, nat) specs :func:`capture_cmd` reports added, each as
    ``iptables -C OUTPUT`` takes it."""
    v4: list[str] = []
    v6: list[str] = []
    nat: list[str] = []
    for line in (out or "").splitlines():
        parts = line.split(" ", 4)
        if len(parts) < 5 or parts[:2] != ["BF", "rule-added"]:
            continue
        tool, table, rule = parts[2], parts[3], parts[4]
        if not rule.startswith("-A OUTPUT "):
            continue
        spec = rule[len("-A OUTPUT ") :]
        if table == "nat":
            nat.append(spec)
        elif tool == "ip6tables":
            v6.append(spec)
        else:
            v4.append(spec)
    return v4, v6, nat


def check_cmd(rules: InstalledRules) -> str:
    """The helper's script: ``iptables -C`` for each rule (ip6tables for v6,
    ``-t nat`` for nat), one ``BF rule-check <present|missing> <family>
    <index>`` line each, then the report line with ``missing=<n>``."""

    def checks(tool: str, family: str, rules_: list[str], table: str = "") -> str:
        t = f"-t {table} " if table else ""
        return "".join(
            f"r={shlex.quote(rule)}; if {tool} -w {t}-C OUTPUT $r 2>/dev/null; "
            f'then echo "BF rule-check present {family} {i}"; '
            f'else echo "BF rule-check missing {family} {i}"; m=$((m + 1)); fi; '
            for i, rule in enumerate(rules_)
        )

    return (
        "m=0; "
        + checks(shlex.quote(rules.tool4), "v4", rules.v4)
        + checks(shlex.quote(rules.tool6), "v6", rules.v6)
        + checks(shlex.quote(rules.tool4), "nat", rules.nat, "nat")
        + f'echo "{REPORT} status=0 missing=$m"'
    )


def missing_in(out: str, rules: InstalledRules) -> list[str]:
    """The rule specs *out* (:func:`check_cmd`'s output) reports missing."""
    by_family = {"v4": rules.v4, "v6": rules.v6, "nat": rules.nat}
    gone: list[str] = []
    for line in (out or "").splitlines():
        parts = line.split()
        if len(parts) == 5 and parts[:3] == ["BF", "rule-check", "missing"]:
            family, index = parts[3], parts[4]
            specs = by_family.get(family, [])
            if index.isdigit() and int(index) < len(specs):
                gone.append(f"{family}: {specs[int(index)]}")
    return gone
