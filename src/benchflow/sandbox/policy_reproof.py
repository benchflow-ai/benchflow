"""The trial's network rules, proven again before the trial is scored.

BenchFlow's helper puts the rules of a trial's network namespaces in, and the
rule ledger (:mod:`benchflow.sandbox.rule_ledger`) records each one with its
phase and when it was proven. A rule proven once is not proven for the rest of
the trial. A process that holds NET_ADMIN in the namespace (a root agent where
main kept it, RCI on 86fd158d6) can delete or flush the rules, or zero their
counters, and the verdict then read "held, refused 0". So the rules are proven
again:

- **At the end of the agent's turn**: in hardening, right after the kills and
  before any verifier step runs. Every rule of the ``start`` and ``agent*``
  phases that is still expected.
- **At the end of test.sh** (and of the criteria runner), before its
  network's refusals are counted. Every rule still expected, plus a TCP
  connect, as root from inside the bounded launch, to the cloud metadata
  server (where a rule refuses it) and to TEST-NET (where the verifier's
  network refuses everything else). Each must be refused, or the kernel must
  answer unreachable. It is a connect only: no request is sent.

The helper joined to each namespace runs ``iptables -C`` for every recorded
rule (:func:`benchflow.sandbox.rule_ledger.check_cmd`) and reads each rule's
packet counter. The trial ends "network policy breached" when there is
evidence of one:

- a rule is missing;
- a counter is lower than when it was last read, or lower than its proof
  counted (the counters were zeroed, or the rule was put in again);
- a connect got through.

A check the helper could not make (a loaded host: its helper slow to start)
is made once more; when it still is not, and nothing else shows a breach,
the trial ends "network policy not proven": unscored with its reason, as
BenchFlow's infrastructure, never a breach charged to the agent (the
lead's 05:42Z rule).

That ending is unscored, never retried and logged as an error. Its record
(``verifier/network-policy.json``) says when each affected entry was last
proven present.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import secrets
import shlex
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from benchflow.sandbox import rule_ledger

if TYPE_CHECKING:
    from benchflow.task.verifier_errors import NetworkPolicyBreachedError

#: The opening words of the ending (benchflow._utils.scoring).
NETWORK_POLICY_BREACHED = "network policy breached: "
#: The ending where the check could not be made and nothing showed a breach.
NETWORK_POLICY_UNPROVEN = "network policy not proven: "
#: Seconds before a check the helper could not make is made again.
_RECHECK_SEC = 3.0
#: The trial folder's record of a breach (``verifier/``).
NETWORK_POLICY_RECORD = "network-policy.json"
#: When the rules are proven again.
END_OF_AGENT = "the end of the agent's turn"
END_OF_TEST = "the end of test.sh"
#: The connects from inside the bounded launch: the cloud metadata server,
#: and TEST-NET-1 (RFC 5737, never routed anywhere).
_METADATA = "169.254.169.254"
_TEST_NET = "192.0.2.1"
#: A connect that BenchFlow's rules held.
_HELD_CONNECTS = ("refused", "unreachable")

_COUNTS_ATTR = "_benchflow_rule_counts"
_PROVEN_ATTR = "_benchflow_rule_proven"


def end_of_agent_phases(entry: rule_ledger.InstalledRules) -> bool:
    """The entries the end of the agent's turn proves: sandbox start and the
    agent's or the oracle's turn (``agent``, ``agent-uid``, ...), whatever
    their names."""
    return entry.phase == "start" or entry.phase.startswith("agent")


@dataclass
class Reproof:
    """One re-proof of a trial's rules."""

    at: str
    checked_at: str
    entries: list[dict[str, Any]] = field(default_factory=list)
    connects: dict[str, str] = field(default_factory=dict)
    #: The counters read with the rules (name: (now, at the arm)).
    totals: dict[str, tuple[int, int]] = field(default_factory=dict)
    breaches: list[str] = field(default_factory=list)
    #: Entries whose check the helper could not make (twice): no evidence
    #: either way. Never a breach.
    unproven: list[str] = field(default_factory=list)

    @property
    def held(self) -> bool:
        """Whether nothing showed a breach."""
        return not self.breaches

    @property
    def proven(self) -> bool:
        """Whether every entry was checked and held."""
        return not self.breaches and not self.unproven

    @property
    def last_proven_present(self) -> str:
        """When the rules this re-proof found broken were last proven present:
        the earliest such time among the entries it flags."""
        times = [
            e["last_proven_present"]
            for e in self.entries
            if e.get("state") != "held" and e.get("last_proven_present")
        ]
        return min(times) if times else ""

    def record(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "state": "breached"
            if self.breaches
            else "not proven"
            if self.unproven
            else "held",
            "at": self.at,
            "checked_at": self.checked_at,
            "rules": sum(e.get("rules", 0) for e in self.entries),
        }
        if not self.held:
            out["breaches"] = list(self.breaches)
        if self.unproven:
            out["unproven"] = list(self.unproven)
        if not self.proven:
            out["last_proven_present"] = self.last_proven_present
        out["entries"] = self.entries
        if self.connects:
            out["connects"] = dict(self.connects)
        if self.totals:
            out["counters"] = {
                k: {"now": now, "at_arm": then}
                for k, (now, then) in self.totals.items()
            }
        return out

    def words(self) -> str:
        """The ending's message."""
        when = self.last_proven_present
        if self.held:
            return (
                f"at {self.at}, BenchFlow could not prove the trial's network rules "
                f"again ({'; '.join(self.unproven)}), and nothing showed they "
                "stopped holding (no rule missing, no counter back, no connect "
                "through)"
                + (f"; they were last proven present at {when}" if when else "")
                + ". The run is not scored, as an infrastructure failure, not the "
                "agent's. See verifier/network-policy.json"
            )
        return (
            f"at {self.at}, BenchFlow's network rules for the trial no longer held "
            f"({'; '.join(self.breaches)})"
            + (f"; they were last proven present at {when}" if when else "")
            + ". The run is not scored: its tests ran, or would have run, without "
            "the task's network policy. See verifier/network-policy.json"
        )


def _key(entry: rule_ledger.InstalledRules) -> str:
    return f"{entry.where}|{entry.phase}|{entry.proven_at}"


def _counts_cmd(entry: rule_ledger.InstalledRules) -> str:
    """The helper's read of each recorded rule's copies and packets: ``BF
    rule-count <v4|v6|nat> <index> <copies> <packets>`` (``? ?`` where the
    rule is not there or its line cannot be matched). ``iptables -v -C``
    prints the rule as ``-L`` lists it; the ``-L -v -x`` lines whose rule
    reads the same carry its counters."""

    def counts(tool: str, family: str, specs: list[str], table: str) -> str:
        if not specs:
            return ""
        t = f"-t {table} "
        return f"L=$({tool} -w {t}-L OUTPUT -v -x -n 2>/dev/null); " + "".join(
            f"r={shlex.quote(spec)}; "
            f"f=$({tool} -w {t}-v -C OUTPUT $r 2>/dev/null | head -n 1 | awk "
            '\'$3 == "opt" && $5 == "in" && $7 == "out" && $10 == "->" '
            '{ o = ""; for (i = 1; i <= NF; i++) if (i != 3 && i != 5 && i != 7 && i != 10) '
            'o = o " " $i; print o }\'); '
            'if [ -n "$f" ]; then n=$(printf \'%s\\n\' "$L" | awk -v f="$f" '
            '\'NR > 2 { o = ""; for (i = 3; i <= NF; i++) o = o " " $i; '
            'if (o == f) { c++; s += $1 } } END { printf "%d %d", c, s }\'); '
            f'else n="? ?"; fi; echo "BF rule-count {family} {i} $n"; '
            for i, spec in enumerate(specs)
        )

    return (
        counts(shlex.quote(entry.tool4), "v4", entry.v4, "filter")
        + counts(shlex.quote(entry.tool6), "v6", entry.v6, "filter")
        + counts(shlex.quote(entry.tool4), "nat", entry.nat, "nat")
    )


def reproof_cmd(entries: list[rule_ledger.InstalledRules], tail: str = "") -> str:
    """One namespace's script: for each entry, a ``BF rule-entry <k>`` line,
    its ``iptables -C`` checks and report (:func:`rule_ledger.check_cmd`), then
    its counters (:func:`_counts_cmd`). *tail* (a counter read) comes last, so
    the first report line the helper reads is the first entry's."""
    return (
        "".join(
            f'echo "BF rule-entry {k}"; '
            + rule_ledger.check_cmd(entry)
            + "; "
            + _counts_cmd(entry)
            for k, entry in enumerate(entries)
        )
        + tail
    )


def _segments(out: str, n: int) -> list[str | None]:
    """Each entry's lines of :func:`reproof_cmd`'s output (None: none came)."""
    parts: list[list[str] | None] = [None] * n
    current: list[str] | None = None
    for line in (out or "").splitlines():
        found = re.fullmatch(r"BF rule-entry (\d+)", line.strip())
        if found:
            k = int(found[1])
            current = parts[k] = [] if k < n else None
            continue
        if current is not None:
            current.append(line)
    return ["\n".join(p) if p is not None else None for p in parts]


def _rejects(entry: rule_ledger.InstalledRules) -> set[tuple[str, int]]:
    """The entry's rules that refuse (their counters are what its proof counted)."""
    return {
        (family, i)
        for family, specs in (("v4", entry.v4), ("v6", entry.v6), ("nat", entry.nat))
        for i, spec in enumerate(specs)
        if "-j REJECT" in spec
    }


def _proof_floor(env: Any, entry: rule_ledger.InstalledRules) -> int:
    """What the entry's proof counted at its refusing rules: the ledger's own
    ``counted`` where its installer records it, else, for main's start rule,
    its proof's count (``service-network.json``)."""
    counted = getattr(entry, "counted", None)
    if isinstance(counted, int) and not isinstance(counted, bool):
        return counted
    if (entry.where, entry.phase) == ("main", "start"):
        record = getattr(env, "_main_cover_record", None)
        proof = record.get("proof") if isinstance(record, dict) else None
        value = proof.get("metadata_counted") if isinstance(proof, dict) else None
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return 0


def _judge(
    env: Any,
    entry: rule_ledger.InstalledRules,
    segment: str | None,
    why: str,
    checked_at: str,
) -> dict[str, Any]:
    """One entry's verdict from its lines, its counter floors updated."""
    rules = len(entry.v4) + len(entry.v6) + len(entry.nat)
    proven = getattr(env, _PROVEN_ATTR, None)
    if not isinstance(proven, dict):
        proven = {}
        with contextlib.suppress(AttributeError):
            setattr(env, _PROVEN_ATTR, proven)
    last = proven.get(_key(entry)) or entry.proven_at
    row: dict[str, Any] = {
        "where": entry.where,
        "phase": entry.phase,
        "rules": rules,
        "proven_at": entry.proven_at,
        "last_proven_present": last,
    }
    report = (
        re.search(r"BF rule-check status=0 missing=(\d+)", segment) if segment else None
    )
    if segment is None or report is None:
        row.update(state="unchecked", why=why or "the helper's check did not report")
        return row
    missing = rule_ledger.missing_in(segment, entry)
    if int(report[1]) and not missing:
        missing = [f"{int(report[1])} rules (unnamed)"]
    floors = getattr(env, _COUNTS_ATTR, None)
    if not isinstance(floors, dict):
        floors = {}
        with contextlib.suppress(AttributeError):
            setattr(env, _COUNTS_ATTR, floors)
    back: list[str] = []
    read: dict[tuple[str, int], int] = {}
    specs = {"v4": entry.v4, "v6": entry.v6, "nat": entry.nat}
    for family, index, _copies, packets in re.findall(
        r"(?m)^BF rule-count (v4|v6|nat) (\d+) (\S+) (\S+)$", segment
    ):
        if not packets.isdigit():
            continue
        i = int(index)
        if i >= len(specs[family]):
            continue
        read[(family, i)] = int(packets)
        key = f"{_key(entry)}|{family}|{i}"
        floor = floors.get(key)
        if isinstance(floor, int) and int(packets) < floor:
            back.append(f"{family}: {specs[family][i]} ({floor} -> {packets})")
        floors[key] = max(int(packets), floor if isinstance(floor, int) else 0)
    rejects = _rejects(entry)
    proof = _proof_floor(env, entry)
    refused_now = sum(read.get(r, 0) for r in rejects)
    if proof and rejects and all(r in read for r in rejects) and refused_now < proof:
        back.append(
            f"its refusing rules show {refused_now} packets, but its proof counted {proof}"
        )
    if missing:
        row.update(state="missing", missing=missing)
    elif back:
        row.update(state="counters back", counters_back=back)
    else:
        row["state"] = "held"
        proven[_key(entry)] = checked_at
        row["last_proven_present"] = checked_at
    if missing and back:
        row["counters_back"] = back
    return row


async def _in_namespace(
    env: Any, container: str | None, command: str
) -> tuple[str, str]:
    """*command* from BenchFlow's helper joined to main (*container* None) or
    to *container*: its output, or "" and why it did not run."""
    if container is None:
        from benchflow.sandbox.verifier_network import _has_helper, _via_helper

        if not _has_helper(env):
            return "", "no helper serves main"
        try:
            out = await _via_helper(env, command)
        except Exception as exc:
            return "", f"the helper failed ({type(exc).__name__})"
        if not out:
            why = getattr(env, "metadata_rules_unavailable", "")
            return "", f"the helper did not run the check{f' ({why})' if why else ''}"
        return out, ""
    install = getattr(env, "metadata_service_install", None)
    if not callable(install):
        return "", "no helper serves that namespace"
    try:
        outcome = await install(container, f": {secrets.token_hex(4)}; {command}")
    except Exception as exc:
        return "", f"the helper failed ({type(exc).__name__})"
    if outcome is None or outcome[1]:
        why = getattr(env, "metadata_service_unavailable", None)
        detail = why(container) if callable(why) else ""
        return "", f"the helper did not run the check{f' ({detail})' if detail else ''}"
    out = getattr(outcome[0], "stdout", "") or ""
    return (out if isinstance(out, str) else ""), ""


def _namespaces(
    entries: list[rule_ledger.InstalledRules],
) -> dict[str | None, list[rule_ledger.InstalledRules]]:
    """The entries by the namespace the helper joins: main's (its own helper,
    whatever container its start entry names) or a service's container."""
    groups: dict[str | None, list[rule_ledger.InstalledRules]] = {}
    for entry in entries:
        container = None if entry.where == "main" else entry.container
        groups.setdefault(container, []).append(entry)
    return groups


def _armed_tail(armed: Any, container: str | None) -> str:
    """The counters the verifier's arm read, read again with the rules: main's
    final REJECT and its cloud metadata rules, or a verifier.service's."""
    from benchflow.sandbox.service_cover import counters_cmd
    from benchflow.sandbox.verifier_network import count_cmd

    if armed is None:
        return ""
    if container is None and armed.container is None:
        tail = counters_cmd() + "; "
        if armed.match and armed.via_helper:
            tail += count_cmd() + "; "
        return tail
    if container is not None and container == armed.container:
        return counters_cmd() + "; "
    return ""


def _totals(out: str, armed: Any, container: str | None) -> dict[str, tuple[int, int]]:
    """(now, at the arm) for each counter the arm read (:func:`_armed_tail`)."""
    totals: dict[str, tuple[int, int]] = {}
    if armed is None:
        return totals
    services = re.search(r"BF service-counters metadata=(\d+) egress=(\d+)", out or "")
    refused = re.search(r"BF verifier-network refused (\d+) status=0", out or "")
    if container is None and armed.container is None:
        before = armed.metadata_refused_before
        if services and isinstance(before, int):
            totals["main cloud metadata rules"] = (int(services[1]), before)
        if refused and armed.match and armed.via_helper:
            totals["the verifier network's final REJECT"] = (
                int(refused[1]),
                armed.baseline,
            )
    elif container is not None and container == armed.container and services:
        totals[f"{armed.where}'s final REJECT"] = (int(services[2]), armed.baseline)
    return totals


def _connects_cmd(addresses: list[str]) -> str:
    """The connects, as the bounded launch runs them: BenchFlow's kit Python
    where the kit has one, else bash's /dev/tcp (the kit's bash and timeout
    first); ``BF reproof-connect <address> <refused|unreachable|reached|
    timeout|failed>`` each. A connect only: nothing is sent."""
    from benchflow._runtime_tree import KIT_BIN, KIT_PYTHON

    py = (
        "import socket,sys,errno\n"
        "for a in sys.argv[1:]:\n"
        "    s=socket.socket(); s.settimeout(3)\n"
        "    try:\n"
        "        s.connect((a,80)); r='reached'\n"
        "    except ConnectionRefusedError: r='refused'\n"
        "    except socket.timeout: r='timeout'\n"
        "    except OSError as e:\n"
        "        r='unreachable' if e.errno in (errno.ENETUNREACH,errno.EHOSTUNREACH) else 'failed'\n"
        "    finally: s.close()\n"
        "    print('BF reproof-connect',a,r)\n"
    )
    kit_py = shlex.quote(KIT_PYTHON)
    kit_bash, kit_timeout = (
        shlex.quote(f"{KIT_BIN}/bash"),
        shlex.quote(f"{KIT_BIN}/timeout"),
    )
    each = " ".join(shlex.quote(a) for a in addresses)
    return (
        f"if [ -x {kit_py} ]; then {kit_py} -I -S -B -c {shlex.quote(py)} {each}; else "
        f"B=bash; [ -x {kit_bash} ] && B={kit_bash}; T=timeout; [ -x {kit_timeout} ] && T={kit_timeout}; "
        f"for a in {each}; do "
        'e=$("$T" 3 "$B" -c "exec 3<>/dev/tcp/$a/80" 2>&1); r=$?; '
        "case $r in 0) s=reached;; 124) s=timeout;; *) case $e in "
        "*[Rr]efused*) s=refused;; *[Uu]nreachable*|*'No route'*) s=unreachable;; "
        "*) s=failed;; esac;; esac; "
        'echo "BF reproof-connect $a $s"; done; fi'
    )


def _refuses_metadata(entries: list[rule_ledger.InstalledRules]) -> bool:
    return any(
        "-j REJECT" in spec and re.search(rf"-d {re.escape(_METADATA)}(/32)?\b", spec)
        for e in entries
        for spec in e.v4
    )


def _refuses_the_rest(entries: list[rule_ledger.InstalledRules]) -> bool:
    """Whether a rule refuses every other destination (a final REJECT with no
    match of its own)."""
    return any(
        re.sub(r"--reject-with \S+", "", spec).split() == ["-j", "REJECT"]
        for e in entries
        for spec in e.v4
    )


async def _connects(
    env: Any, service: str, entries: list[rule_ledger.InstalledRules]
) -> dict[str, str]:
    """The connects from inside the bounded launch, as root in *service*, to
    what the namespace's rules refuse; where the launched shell could not make
    one (no kit Python, no bash), the helper makes it as uid 0 in main's
    namespace."""
    from benchflow.sandbox._exec_output import exec_raw
    from benchflow.sandbox.verifier_network import verifier_launch

    addresses = []
    if _refuses_metadata(entries):
        addresses.append(_METADATA)
    if _refuses_the_rest(entries):
        addresses.append(_TEST_NET)
    if not addresses:
        return {}
    found: dict[str, str] = {}
    try:
        result = await exec_raw(
            env.exec,
            verifier_launch(_connects_cmd(addresses)),
            user="root",
            service=service,
            timeout_sec=60,
        )
        out = str(getattr(result, "stdout", "") or "")
    except Exception:
        out = ""
    for address, state in re.findall(r"(?m)^BF reproof-connect (\S+) (\w+)$", out):
        found[address] = state
    run = getattr(env, "metadata_rules_run", None)
    for address in addresses:
        if found.get(address) not in (None, "failed"):
            continue
        state = "not made"
        if service == "main" and callable(run):
            from benchflow.sandbox.service_cover import _connect, connect_cmd

            with contextlib.suppress(Exception):
                ran, failed = await run(
                    connect_cmd(address), timeout_sec=30, user="0:0"
                )
                made = "" if failed else _connect(str(getattr(ran, "stdout", "") or ""))
                if made:
                    state = f"{made} (from the helper, uid 0)"
        found[address] = state
    return found


async def reprove(
    env: Any,
    at: str,
    *,
    phases: Any = None,
    armed: Any = None,
) -> Reproof | None:
    """Prove the trial's rules again from the helper (module doc): the
    expected ledger entries (those *phases* accepts, when given) and their
    counters, and with *armed* (the verifier's network) the counters its arm
    read. None when the ledger holds nothing to prove."""
    entries = [e for e in rule_ledger.expected(env) if phases is None or phases(e)]
    if not entries:
        return None
    checked_at = rule_ledger.now()
    proof = Reproof(at=at, checked_at=checked_at)
    groups = _namespaces(entries)
    keys = list(groups)

    async def check(container: str | None) -> tuple[str, str]:
        group = groups[container]
        return await _in_namespace(
            env, container, reproof_cmd(group, _armed_tail(armed, container))
        )

    try:
        outcomes = list(await asyncio.gather(*(check(c) for c in keys)))
        # A check the helper could not make (a loaded host: its helper slow to
        # start) is made once more before it counts as not proven.
        again = [i for i, (out, _why) in enumerate(outcomes) if not out]
        if again:
            await asyncio.sleep(_RECHECK_SEC)
            for i, outcome in zip(
                again,
                await asyncio.gather(*(check(keys[i]) for i in again)),
                strict=True,
            ):
                if outcome[0]:
                    outcomes[i] = outcome
    finally:
        close = getattr(env, "metadata_service_close", None)
        if callable(close) and any(c is not None for c in keys):
            with contextlib.suppress(Exception):
                close()
    for container, (out, why) in zip(keys, outcomes, strict=True):
        group = groups[container]
        for entry, segment in zip(group, _segments(out, len(group)), strict=True):
            row = _judge(env, entry, segment, why, checked_at)
            proof.entries.append(row)
            name = f"{row['where']} ({row['phase']})"
            if row["state"] == "missing":
                proof.breaches.append(
                    f"{name}: {len(row['missing'])} of {row['rules']} rules missing"
                )
            elif row["state"] == "counters back":
                proof.breaches.append(f"{name}: counters went back")
            elif row["state"] == "unchecked":
                # No evidence either way: never a breach.
                proof.unproven.append(f"{name}: not proven ({row['why']})")
        for name, (now, then) in _totals(out, armed, container).items():
            proof.totals[name] = (now, then)
            if now < then:
                proof.breaches.append(
                    f"{name}: its counter went back from {then} to {now}"
                )
    return proof


def breach_error(proof: Reproof) -> NetworkPolicyBreachedError:
    """The ending's error for *proof* (benchflow.task.verifier_errors): a
    breach where something showed one, else "not proven"."""
    from benchflow.task.verifier_errors import (
        NetworkPolicyBreachedError,
        NetworkPolicyUnprovenError,
    )

    kind = NetworkPolicyBreachedError if proof.breaches else NetworkPolicyUnprovenError
    return kind(proof.words(), record=proof.record())


def ending_head(error: BaseException) -> str:
    """The opening words of *error*'s ending: "network policy not proven: "
    for a check that could not be made, else "network policy breached: "."""
    from benchflow.task.verifier_errors import NetworkPolicyUnprovenError

    if isinstance(error, NetworkPolicyUnprovenError):
        return NETWORK_POLICY_UNPROVEN
    return NETWORK_POLICY_BREACHED


async def require_held_at_end_of_agent(env: Any) -> Reproof | None:
    """The end of the agent's turn (module doc): raise the ending's error when
    a rule of the ``start`` or ``agent*`` phases no longer holds."""
    proof = await reprove(env, END_OF_AGENT, phases=end_of_agent_phases)
    if proof is not None and not proof.proven:
        raise breach_error(proof)
    return proof


async def connect_from_launch(env: Any, proof: Reproof, service: str) -> None:
    """Add to *proof* the connects from inside the bounded launch, as root in
    *service*, to what that namespace's expected rules refuse (module doc).
    Made after the verifier's refusals are counted: a refused connect to
    TEST-NET counts at the final REJECT, and it is not test.sh's."""
    where = "main" if service == "main" else f"service {service}"
    here = [e for e in rule_ledger.expected(env) if e.where == where]
    proof.connects = await _connects(env, service, here)
    for address, state in proof.connects.items():
        if state.split(" ", 1)[0] not in _HELD_CONNECTS and state != "not made":
            proof.breaches.append(
                f"root's connect to {address}:80 from inside the bounded launch "
                f"was {state}"
            )
