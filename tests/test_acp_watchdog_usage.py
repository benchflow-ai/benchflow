"""Guards commit 7f171a1d: issue #1203 usage-only watchdog activity."""

from benchflow.acp.session import ACPSession
from benchflow.acp.watchdog import IdleWatchdog


def _watchdog(session):
    return IdleWatchdog.start(
        session, idle_timeout_sec=900, wall_timeout_sec=3600, now=0
    )


def _usage(session, used=100):
    session.handle_update(
        {
            "sessionUpdate": "usage_update",
            "used": used,
            "size": 200000,
            "cost": {"amount": 0.1, "currency": "USD"},
        }
    )


def test_changing_usage_keeps_a_direct_session_alive():
    session = ACPSession("direct")
    watchdog = _watchdog(session)
    for now in range(300, 3601, 300):
        _usage(session, used=now)
        watchdog.observe(session, now=now)
        assert not watchdog.idle_expired(now)
    assert watchdog.wall_clock_expired(3601)
    watchdog.observe(session, now=4500)
    assert watchdog.idle_expired(4500)


def test_repeated_usage_does_not_extend_idle_deadline():
    session = ACPSession("direct")
    watchdog = _watchdog(session)
    _usage(session)
    watchdog.observe(session, now=300)
    for now in (600, 900, 1199):
        _usage(session)
        watchdog.observe(session, now=now)
        assert not watchdog.idle_expired(now)
    _usage(session)
    watchdog.observe(session, now=1200)
    assert watchdog.idle_expired(1200)


def test_usage_snapshot_does_not_alias_the_incoming_packet():
    session = ACPSession("direct")
    watchdog = _watchdog(session)
    packet = {
        "sessionUpdate": "usage_update",
        "used": 100,
        "cost": {"amount": 0.1, "currency": "USD"},
    }
    session.handle_update(packet)
    watchdog.observe(session, now=300)
    packet["cost"]["amount"] = 0.2
    session.handle_update(packet)
    watchdog.observe(session, now=900)
    assert not watchdog.idle_expired(1200)


def test_usage_does_not_reset_a_stuck_tool_grace():
    session = ACPSession("direct")
    session.handle_update({"sessionUpdate": "tool_call", "toolCallId": "stuck"})
    watchdog = _watchdog(session)
    for now in range(300, 2701, 300):
        _usage(session, used=now)
        watchdog.observe(session, now=now)
    assert watchdog.idle_expired(2700)
