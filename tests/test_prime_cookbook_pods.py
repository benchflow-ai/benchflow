"""The Prime cookbook's pod watchdog decides correctly, offline.

``docs/examples/rl/prime/pods/prime_pods.py`` terminates pods past their
lifetime and every pod at the spend cap. These tests exercise its pure
decision functions and its ledger with fixed clocks; no request reaches Prime.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

PODS = Path(__file__).resolve().parents[1] / "docs" / "examples" / "rl" / "prime" / "pods"
sys.path.insert(0, str(PODS))
import prime_pods  # noqa: E402

NOW = datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc)


def ledger_pod(pod_id: str, *, hours_ago: float, price: float, max_hours: float = 8.0, ended_hours_ago: float | None = None):
    return prime_pods.LedgerPod(
        id=pod_id,
        name=pod_id,
        created_at=NOW - timedelta(hours=hours_ago),
        price_hr=price,
        max_hours=max_hours,
        ended_at=None if ended_hours_ago is None else NOW - timedelta(hours=ended_hours_ago),
    )


def running(pod_id: str, *, hours_ago: float, price: float, status: str = "ACTIVE") -> dict:
    # The API's timestamps are naive UTC.
    created = (NOW - timedelta(hours=hours_ago)).replace(tzinfo=None).isoformat()
    return {"id": pod_id, "name": pod_id, "status": status, "createdAt": created, "priceHr": price}


def test_spend_counts_ended_running_and_unknown_pods() -> None:
    ledger = {
        "old": ledger_pod("old", hours_ago=10, price=2.0, ended_hours_ago=6),  # 4h x $2
        "live": ledger_pod("live", hours_ago=2, price=3.0),  # 2h x $3, running
    }
    active = [running("live", hours_ago=2, price=3.0), running("stranger", hours_ago=1, price=1.5)]
    spend = prime_pods.estimate_spend(ledger, active, NOW)
    assert spend.ledger == pytest.approx(8.0 + 6.0)
    assert spend.foreign == pytest.approx(1.5)
    assert spend.total == pytest.approx(15.5)


def test_a_ledger_pod_with_no_recorded_end_counts_until_now() -> None:
    # Missing from /pods/ but no end recorded yet: keep counting (overestimate).
    spend = prime_pods.estimate_spend({"x": ledger_pod("x", hours_ago=3, price=1.0)}, [], NOW)
    assert spend.total == pytest.approx(3.0)


def test_running_price_and_earlier_start_win_over_the_ledger() -> None:
    ledger = {"p": ledger_pod("p", hours_ago=1, price=1.0)}
    spend = prime_pods.estimate_spend(ledger, [running("p", hours_ago=2, price=4.0)], NOW)
    assert spend.total == pytest.approx(8.0)


def test_terminates_pods_past_their_own_lifetime_only() -> None:
    ledger = {
        "short": ledger_pod("short", hours_ago=0.5, price=0.05, max_hours=0.25),
        "young": ledger_pod("young", hours_ago=0.5, price=0.05, max_hours=6),
    }
    active = [running("short", hours_ago=0.5, price=0.05), running("young", hours_ago=0.5, price=0.05)]
    doomed, _ = prime_pods.pods_to_terminate(ledger, active, NOW, max_hours=8, spend_cap=1400)
    assert [pod_id for pod_id, _ in doomed] == ["short"]


def test_unknown_pods_get_the_global_lifetime_and_it_never_exceeds_8_hours() -> None:
    active = [running("stranger", hours_ago=8.5, price=1.0), running("fresh", hours_ago=1, price=1.0)]
    doomed, _ = prime_pods.pods_to_terminate({}, active, NOW, max_hours=24, spend_cap=1e9)
    assert [pod_id for pod_id, _ in doomed] == ["stranger"]


def test_ledger_lifetime_is_capped_at_8_hours() -> None:
    ledger = {"greedy": ledger_pod("greedy", hours_ago=9, price=1.0, max_hours=100)}
    doomed, _ = prime_pods.pods_to_terminate(
        ledger, [running("greedy", hours_ago=9, price=1.0)], NOW, max_hours=100, spend_cap=1e9
    )
    assert [pod_id for pod_id, _ in doomed] == ["greedy"]


def test_spend_cap_terminates_every_pod() -> None:
    ledger = {"burned": ledger_pod("burned", hours_ago=20, price=70.0, ended_hours_ago=0.5)}  # $1365
    active = [running("a", hours_ago=0.2, price=30.0), running("b", hours_ago=0.2, price=30.0)]  # + $12
    doomed, spend = prime_pods.pods_to_terminate(ledger, active, NOW, max_hours=8, spend_cap=1400)
    assert spend.total < 1400 and doomed == []
    later = NOW + timedelta(hours=0.5)  # + $30 more
    doomed, spend = prime_pods.pods_to_terminate(ledger, active, later, max_hours=8, spend_cap=1400)
    assert spend.total >= 1400
    assert sorted(pod_id for pod_id, _ in doomed) == ["a", "b"]


def test_stop_all_file_terminates_everything_and_terminated_pods_are_skipped() -> None:
    active = [running("a", hours_ago=0.1, price=1.0), running("gone", hours_ago=0.1, price=1.0, status="TERMINATED")]
    doomed, _ = prime_pods.pods_to_terminate({}, active, NOW, max_hours=8, spend_cap=1400, stop_all=True)
    assert [pod_id for pod_id, _ in doomed] == ["a"]


def test_ledger_merges_create_and_end_records(tmp_path: Path) -> None:
    prime_pods.ledger_append(
        tmp_path,
        {"event": "create", "id": "p1", "name": "n", "created_at": "2026-09-30T10:00:00Z", "price_hr": 2.0, "max_hours": 99},
    )
    prime_pods.ledger_append(tmp_path, {"event": "ended", "id": "p1", "ended_at": "2026-09-30T11:30:00Z", "billed": 3.1})
    prime_pods.ledger_append(tmp_path, {"event": "ended", "id": "p1", "ended_at": "2026-09-30T11:45:00Z"})
    (tmp_path / "ledger.jsonl").open("a").write("not json\n")
    pods = prime_pods.ledger_read(tmp_path)
    pod = pods["p1"]
    assert pod.max_hours == 8.0
    assert pod.ended_at == datetime(2026, 9, 30, 11, 30, tzinfo=timezone.utc)
    assert pod.billed == 3.1
    assert prime_pods.estimate_spend(pods, [], NOW).total == pytest.approx(3.0)
    assert oct((tmp_path / "ledger.jsonl").stat().st_mode & 0o777) == "0o600"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("root@1.2.3.4 -p 2222", "root@1.2.3.4:2222"),
        (["ubuntu@5.6.7.8 -p 22"], "ubuntu@5.6.7.8:22"),
        ("ssh root@9.9.9.9", "root@9.9.9.9:22"),
        ([None], None),
        (None, None),
    ],
)
def test_ssh_target_parsing(raw, expected) -> None:
    assert prime_pods.ssh_target({"sshConnection": raw}) == expected


def test_create_refuses_without_a_running_watchdog(tmp_path: Path) -> None:
    class NoPrime:
        def offers(self, gpu_type=None):  # pragma: no cover - must not be reached
            raise AssertionError("create must refuse before asking for offers")

    args = prime_pods.main.__globals__["argparse"].Namespace(
        dir=tmp_path, max_hours=1.0, no_watchdog_check=False, gpu_type="CPU_NODE"
    )
    with pytest.raises(SystemExit, match="watchdog is not running"):
        prime_pods.cmd_create(NoPrime(), args)


def owned(pod_id: str, owner: str, *, hours_ago: float, price: float, max_hours: float = 8.0, ended_hours_ago: float | None = None):
    pod = ledger_pod(pod_id, hours_ago=hours_ago, price=price, max_hours=max_hours, ended_hours_ago=ended_hours_ago)
    pod.owner = owner
    return pod


def test_spend_is_split_by_owner_and_unknown_pods_are_unattributed() -> None:
    ledger = {
        "p": owned("p", "rl-prime", hours_ago=2, price=3.0, ended_hours_ago=1),  # $3
        "m": owned("m", "miles", hours_ago=1, price=5.0),  # $5, running
    }
    active = [running("m", hours_ago=1, price=5.0), running("x", hours_ago=2, price=1.0)]
    spend = prime_pods.estimate_spend(ledger, active, NOW)
    assert spend.by_owner == pytest.approx({"rl-prime": 3.0, "miles": 5.0, prime_pods.UNATTRIBUTED: 2.0})
    assert spend.total == pytest.approx(10.0)


def test_an_owner_at_its_cap_loses_only_its_own_pods() -> None:
    ledger = {
        "m1": owned("m1", "miles", hours_ago=4, price=100.0, ended_hours_ago=0.5),  # $350
        "m2": owned("m2", "miles", hours_ago=0.5, price=100.0),  # +$50 running
        "p1": owned("p1", "rl-prime", hours_ago=0.5, price=2.0),
    }
    active = [running("m2", hours_ago=0.5, price=100.0), running("p1", hours_ago=0.5, price=2.0), running("x", hours_ago=0.5, price=1.0)]
    doomed, spend = prime_pods.pods_to_terminate(
        ledger, active, NOW, max_hours=8, spend_cap=1400, owner_caps={"miles": 400, "rl-prime": 900}
    )
    assert spend.by_owner["miles"] == pytest.approx(400.0)
    assert [pod_id for pod_id, _ in doomed] == ["m2"]
    assert "owner miles" in doomed[0][1]


def test_the_global_cap_still_stops_everyone_below_their_owner_caps() -> None:
    ledger = {"a": owned("a", "rl-prime", hours_ago=10, price=140.0, ended_hours_ago=0)}  # $1400
    active = [running("b", hours_ago=0.1, price=1.0)]
    doomed, _ = prime_pods.pods_to_terminate(
        ledger, active, NOW, max_hours=8, spend_cap=1400, owner_caps={"rl-prime": 5000}
    )
    assert [pod_id for pod_id, _ in doomed] == ["b"]


def test_policy_file_can_tighten_but_never_loosen(tmp_path: Path) -> None:
    (tmp_path / "policy.json").write_text(
        '{"max_hours": 12, "spend_cap": 1300, "owner_caps": {"rl-prime": 900, "miles": 500}}'
    )
    policy = prime_pods.load_policy(tmp_path, max_hours=8, spend_cap=1400)
    assert policy.max_hours == 8
    assert policy.spend_cap == 1300
    assert policy.owner_caps == {"rl-prime": 900.0, "miles": 500.0}
    (tmp_path / "policy.json").write_text('{"spend_cap": 5000, "max_hours": 2}')
    policy = prime_pods.load_policy(tmp_path, max_hours=8, spend_cap=1400)
    assert (policy.spend_cap, policy.max_hours) == (1400, 2)
    (tmp_path / "policy.json").write_text("not json")
    assert prime_pods.load_policy(tmp_path, max_hours=8, spend_cap=1400).spend_cap == 1400


def test_owner_column_and_legacy_records(tmp_path: Path) -> None:
    prime_pods.ledger_append(
        tmp_path,
        {"event": "create", "id": "old", "created_at": "2026-09-30T06:28:27Z", "price_hr": 0.05, "max_hours": 0.1},
    )
    prime_pods.ledger_append(
        tmp_path,
        {"event": "create", "id": "new", "owner": "miles", "created_at": "2026-09-30T07:00:00Z", "price_hr": 2.0, "max_hours": 4},
    )
    pods = prime_pods.ledger_read(tmp_path)
    assert pods["old"].owner == prime_pods.LEGACY_OWNER
    assert pods["new"].owner == "miles"
