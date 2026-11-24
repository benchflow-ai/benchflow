"""Read one window-check run from host-side records only.

The records, none of them written by BenchFlow or by anything in the sandbox:

- dockerd's event stream (``docker events --format '{{json .}}'``, read live): when each exec in the sandbox
  started and ended, in nanoseconds, with its command;
- the host listener on the bridge gateway (``tests/window/listener.py``): every connection it accepted;
- optionally, a root poller's counts of ``--uid-owner`` rules in the sandbox's network namespace
  (``tests/window/netwatch.py``).

A = the start of the exec that launches the agent (its launch command, not the install check). A run passes when:

1. an exec that installs the agent user's rules (an ``iptables`` ``OUTPUT`` rule with ``--uid-owner``) ended
   before A, and none of them first ended after A (a later one is a re-assert of rules already in);
2. when the poller ran, the kernel showed the rules before A;
3. the listener accepted nothing while the run lasted. The agent connects to it as soon as it starts, before any
   ACP request, so an accept is the window open.

"No accept" means something only if a plain container's connect to the same listener is accepted
(:func:`control_seen`): the check runs that control before and after its runs.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterable

LAUNCH = "/opt/bfprobe/acp.sh"


@dataclasses.dataclass(frozen=True)
class Event:
    t: int
    action: str
    container: str
    exec_id: str
    command: str


@dataclasses.dataclass
class Verdict:
    passed: bool
    reasons: list[str]
    launch_ns: int | None = None
    gap_s: float | None = None
    kernel_s: float | None = None
    accepts: int = 0
    rules_execs: int = 0


def parse_events(lines: Iterable[str]) -> list[Event]:
    out = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("Type") != "container":
            continue
        action = str(e.get("Action") or "")
        attrs = (e.get("Actor") or {}).get("Attributes") or {}
        verb, _, command = action.partition(": ")
        out.append(
            Event(
                int(e.get("timeNano") or 0),
                verb,
                str(attrs.get("name") or ""),
                str(attrs.get("execID") or ""),
                command,
            )
        )
    return sorted(out, key=lambda e: e.t)


def parse_listener(lines: Iterable[str]) -> list[tuple[int, str]]:
    out = []
    for line in lines:
        head, _, rest = line.strip().partition(" ")
        if head.isdigit():
            out.append((int(head), rest))
    return out


def parse_kernel(lines: Iterable[str]) -> list[tuple[int, str, int]]:
    out = []
    for line in lines:
        try:
            r = json.loads(line)
        except ValueError:
            continue
        out.append((int(r["t"]), str(r["container"]), int(r["uid_rules"])))
    return out


def _is_uid_rules(command: str) -> bool:
    return "--uid-owner" in command and "OUTPUT" in command and "tables" in command


def analyze(
    events: list[Event],
    listener: list[tuple[int, str]],
    kernel: list[tuple[int, str, int]] | None,
    launch_marker: str = LAUNCH,
) -> Verdict:
    starts = [e for e in events if e.action == "exec_start"]
    launch = next(
        (
            e
            for e in starts
            if launch_marker in e.command
            and "test -x" not in e.command
            and "pkill" not in e.command
        ),
        None,
    )
    if launch is None:
        return Verdict(False, ["no agent launch exec in dockerd's records"])
    a = launch.t
    # Only this trial's containers: its compose project (main and its helpers,
    # such as the metadata helper). Another trial on the same daemon installs
    # rules of its own, which say nothing about this one.
    project = (
        launch.container.rsplit("-main-", 1)[0]
        if "-main-" in launch.container
        else launch.container
    )
    events = [e for e in events if e.container.startswith(project)]
    starts = [e for e in starts if e.container.startswith(project)]
    die = {e.exec_id: e.t for e in events if e.action == "exec_die"}
    rules = [(e.t, die.get(e.exec_id)) for e in starts if _is_uid_rules(e.command)]
    reasons: list[str] = []
    before = [end for start, end in rules if end is not None and end < a]
    if not rules:
        reasons.append("no uid rules exec in dockerd's records")
    elif not before:
        reasons.append("the agent user's rules went in only after the agent launched")
    t0, t1 = events[0].t, events[-1].t
    accepts = sum(
        1 for t, text in listener if t0 <= t <= t1 and text.startswith("accept ")
    )
    if accepts:
        reasons.append(
            f"the host listener accepted {accepts} connection(s) during the run"
        )
    kernel_s = None
    if kernel is not None:
        seen = [t for t, name, n in kernel if n > 0 and name == launch.container]
        first = min(seen) if seen else None
        if first is None or first >= a:
            reasons.append(
                "the kernel showed no uid rules in the sandbox before the launch"
            )
        else:
            kernel_s = (a - first) / 1e9
    gap = (a - max(before)) / 1e9 if before else None
    return Verdict(not reasons, reasons, a, gap, kernel_s, accepts, len(rules))


def control_seen(listener: list[tuple[int, str]], t0: int, t1: int) -> bool:
    """Whether the listener accepted a connection between ``t0`` and ``t1`` (the positive control's)."""
    return any(t0 <= t <= t1 and text.startswith("accept ") for t, text in listener)
