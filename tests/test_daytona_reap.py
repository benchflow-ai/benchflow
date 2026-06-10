"""Daytona stale-sandbox auto-reaping (dogfood finding: leakage at scale)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from benchflow.sandbox.daytona import reap_stale_sandboxes


def _sb(sb_id: str, state: str, age_minutes: float):
    created = datetime.now(UTC) - timedelta(minutes=age_minutes)
    return SimpleNamespace(
        id=sb_id, state=state, created_at=created.isoformat().replace("+00:00", "Z")
    )


class FakeClient:
    def __init__(self, sandboxes, fail_ids=()):
        self._sandboxes = sandboxes
        self._fail_ids = set(fail_ids)
        self.deleted = []

    def list(self):
        return iter(self._sandboxes)

    def delete(self, sb):
        if sb.id in self._fail_ids:
            raise RuntimeError("api error")
        self.deleted.append(sb.id)


def test_reaps_old_started_keeps_young():
    client = FakeClient([_sb("old", "STARTED", 1500), _sb("young", "STARTED", 30)])
    counts = reap_stale_sandboxes(client)
    assert client.deleted == ["old"]
    assert counts == {"found": 2, "deleted": 1, "skipped": 1, "failed": 0}


def test_failed_states_use_short_ttl():
    client = FakeClient(
        [_sb("bf-old", "BUILD_FAILED", 180), _sb("bf-young", "BUILD_FAILED", 30)]
    )
    counts = reap_stale_sandboxes(client)
    assert client.deleted == ["bf-old"]
    assert counts["skipped"] == 1


def test_error_state_counts_as_failed_tier():
    client = FakeClient([_sb("err", "ERROR", 180)])
    reap_stale_sandboxes(client)
    assert client.deleted == ["err"]


def test_dry_run_deletes_nothing_but_counts():
    client = FakeClient([_sb("old", "STARTED", 1500)])
    counts = reap_stale_sandboxes(client, dry_run=True)
    assert client.deleted == []
    assert counts["deleted"] == 1


def test_delete_failure_is_counted_not_raised():
    client = FakeClient([_sb("bad", "STARTED", 1500)], fail_ids={"bad"})
    counts = reap_stale_sandboxes(client)
    assert counts["failed"] == 1
    assert counts["deleted"] == 0


def test_missing_created_at_is_skipped():
    client = FakeClient([SimpleNamespace(id="x", state="STARTED", created_at=None)])
    counts = reap_stale_sandboxes(client)
    assert counts == {"found": 1, "deleted": 0, "skipped": 1, "failed": 0}


def test_on_decision_sees_every_sandbox():
    seen = []
    client = FakeClient([_sb("old", "STARTED", 1500), _sb("young", "STARTED", 5)])
    reap_stale_sandboxes(client, on_decision=lambda sb, age, d: seen.append((sb.id, d)))
    assert seen == [("old", True), ("young", False)]


class TestEvaluationAutoReapGate:
    def _eval(self, tmp_path, environment):
        from benchflow.evaluation import Evaluation, EvaluationConfig

        tasks = tmp_path / "tasks"
        tasks.mkdir(exist_ok=True)
        cfg = EvaluationConfig(environment=environment)
        return Evaluation(tasks_dir=tasks, jobs_dir=tmp_path / "jobs", config=cfg)

    def test_docker_runs_never_reap(self, tmp_path, monkeypatch):
        started = []
        monkeypatch.setattr(
            "threading.Thread",
            lambda *a, **k: started.append(k) or SimpleNamespace(start=lambda: None),
        )
        self._eval(tmp_path, "docker")._maybe_start_daytona_reap()
        assert started == []

    def test_daytona_reaps_by_default(self, tmp_path, monkeypatch):
        started = []

        class FakeThread:
            def __init__(self, *a, **k):
                started.append(k.get("name"))

            def start(self):
                pass

        monkeypatch.setattr("benchflow.evaluation.threading.Thread", FakeThread)
        monkeypatch.delenv("BENCHFLOW_DAYTONA_AUTO_REAP", raising=False)
        self._eval(tmp_path, "daytona")._maybe_start_daytona_reap()
        assert started == ["daytona-auto-reap"]

    def test_env_gate_disables(self, tmp_path, monkeypatch):
        started = []
        monkeypatch.setattr(
            "benchflow.evaluation.threading.Thread",
            lambda *a, **k: started.append(k) or SimpleNamespace(start=lambda: None),
        )
        monkeypatch.setenv("BENCHFLOW_DAYTONA_AUTO_REAP", "0")
        self._eval(tmp_path, "daytona")._maybe_start_daytona_reap()
        assert started == []
