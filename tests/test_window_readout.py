"""The window check's readout (tests/window/readout.py), with planted faults.

The check reads only host-side records: dockerd's event stream (when each
exec started and ended, in nanoseconds), a listener on the bridge gateway,
and the rule counts a root poller reads from the sandbox's network namespace.
A run passes when the agent user's rules were in before the agent launched
and the agent's start-up connect never reached the host. These rows feed the
readout synthetic records, so a readout that passes a planted fault fails
here without Docker.
"""

from __future__ import annotations

import json

import pytest

from tests.window import readout as R

MAIN = "md-probe__x-main-1"
S = 1_000_000_000


def _ev(t, action, exec_id, cmd="", name=MAIN):
    full = f"exec_start: {cmd}" if action == "exec_start" else action
    return json.dumps(
        {
            "Type": "container",
            "Action": full,
            "timeNano": t,
            "Actor": {"ID": "c1", "Attributes": {"execID": exec_id, "name": name}},
        }
    )


def _records(rules_start, rules_die, launch, accepts=(), kernel=None):
    events = [
        _ev(
            rules_start,
            "exec_start",
            "r1",
            "iptables -w -A OUTPUT -m owner --uid-owner 0 -j REJECT",
        ),
        _ev(rules_die, "exec_die", "r1"),
        _ev(launch - 5 * S, "exec_start", "i1", "test -x /opt/bfprobe/acp.sh"),
        _ev(launch, "exec_start", "a1", "setpriv --reuid 0 /opt/bfprobe/acp.sh"),
        _ev(launch + 30 * S, "exec_die", "a1"),
    ]
    listener = [f"{launch - 60 * S} listening 47123"] + [
        f"{t} accept 172.18.0.2:4000" for t in accepts
    ]
    kern = (
        [json.dumps({"t": kernel, "container": MAIN, "uid_rules": 4})]
        if kernel is not None
        else []
    )
    return (
        R.parse_events(events),
        R.parse_listener(listener),
        R.parse_kernel(kern) if kernel is not None else None,
    )


L = 100 * S


def test_rules_in_before_the_launch_and_no_accept_pass():
    ev, li, k = _records(L - 3 * S, L - S // 10, L, kernel=L - 2 * S)
    v = R.analyze(ev, li, k)
    assert v.passed, v.reasons
    assert v.gap_s == pytest.approx(0.1)
    assert v.kernel_s == pytest.approx(2.0)


def test_planted_rules_after_the_launch_fail():
    ev, li, k = _records(L + S // 10, L + S // 5, L, kernel=L + S // 5)
    v = R.analyze(ev, li, k)
    assert not v.passed
    assert any("after the agent" in r for r in v.reasons), v.reasons


def test_planted_accept_of_the_start_up_connect_fails():
    ev, li, k = _records(L - 3 * S, L - S, L, accepts=[L + S // 20], kernel=L - 2 * S)
    v = R.analyze(ev, li, k)
    assert not v.passed and v.accepts == 1


def test_no_rules_exec_at_all_fails():
    events = [
        _ev(L, "exec_start", "a1", "/opt/bfprobe/acp.sh"),
        _ev(L + S, "exec_die", "a1"),
    ]
    v = R.analyze(R.parse_events(events), R.parse_listener([]), None)
    assert not v.passed and any("no uid rules" in r for r in v.reasons)


def test_kernel_rules_seen_only_after_the_launch_fail():
    ev, li, k = _records(L - 3 * S, L - 2 * S, L, kernel=L + S)
    v = R.analyze(ev, li, k)
    assert not v.passed and any("kernel" in r for r in v.reasons)


def test_no_launch_is_no_verdict():
    events = [
        _ev(L, "exec_start", "r1", "iptables -A OUTPUT --uid-owner 0"),
        _ev(L + 1, "exec_die", "r1"),
    ]
    v = R.analyze(R.parse_events(events), R.parse_listener([]), None)
    assert not v.passed and any("no agent launch" in r for r in v.reasons)


def test_the_positive_control_must_be_accepted():
    """Without a control the listener answering, 'no accept' proves nothing."""
    li = R.parse_listener([f"{L} accept 172.17.0.3:5000"])
    assert R.control_seen(li, L - S, L + S)
    assert not R.control_seen(li, L + S, L + 2 * S)


def test_another_trials_rules_never_count():
    """Two trials on one daemon: the other trial's rules exec before this one's
    launch must not pass this one."""
    events = [
        _ev(
            L - 3 * S,
            "exec_start",
            "o1",
            "iptables -A OUTPUT -m owner --uid-owner 0 -j REJECT",
            name="other__y-main-1",
        ),
        _ev(L - 2 * S, "exec_die", "o1", name="other__y-main-1"),
        _ev(L, "exec_start", "a1", "setpriv --reuid 0 /opt/bfprobe/acp.sh"),
        _ev(
            L + S // 10,
            "exec_start",
            "r1",
            "iptables -A OUTPUT -m owner --uid-owner 0 -j REJECT",
        ),
        _ev(L + S // 5, "exec_die", "r1"),
        _ev(L + 30 * S, "exec_die", "a1"),
    ]
    v = R.analyze(R.parse_events(events), R.parse_listener([]), None)
    assert not v.passed and v.rules_execs == 1, v
