"""Daytona stale-sandbox auto-reaping (dogfood finding: leakage at scale).

The reaper deletes sandboxes by age, so it is *ownership-scoped*: it only ever
touches sandboxes benchflow created (those carrying the ``benchflow.managed``
label). On a ``DAYTONA_API_KEY`` shared across an org or with other tools, that
scope is the only thing preventing irreversible deletion of unrelated
sandboxes — so the foreign-sandbox cases below are load-bearing safety tests,
not edge cases.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from benchflow.sandbox.daytona import (
    _is_benchflow_label_orphan,
    _is_benchflow_owned,
    reap_stale_sandboxes,
)

# The ownership label benchflow stamps on every sandbox it creates. Mirrors the
# (private) constants in benchflow.sandbox.daytona so a rename there that breaks
# scoping shows up as a test failure here rather than silently passing.
_OWNED = {"benchflow.managed": "1"}


def _sb(sb_id: str, state: str, age_minutes: float, *, labels: dict | None = _OWNED):
    """A sandbox view as the SDK's ``list()`` yields it.

    Benchflow-owned by default; pass ``labels=`` (``{}``, ``None``, or a foreign
    set) to model a sandbox the reaper must never touch.
    """
    created = datetime.now(UTC) - timedelta(minutes=age_minutes)
    return SimpleNamespace(
        id=sb_id,
        state=state,
        created_at=created.isoformat().replace("+00:00", "Z"),
        labels=dict(labels) if labels is not None else None,
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


# --- TTL tiers (all sandboxes here are benchflow-owned) ----------------------


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
    # Owned, so it clears the scope gate and exercises the created_at skip path.
    client = FakeClient(
        [SimpleNamespace(id="x", state="STARTED", created_at=None, labels=dict(_OWNED))]
    )
    counts = reap_stale_sandboxes(client)
    assert counts == {"found": 1, "deleted": 0, "skipped": 1, "failed": 0}


def test_on_decision_sees_every_owned_sandbox():
    seen = []
    client = FakeClient([_sb("old", "STARTED", 1500), _sb("young", "STARTED", 5)])
    reap_stale_sandboxes(client, on_decision=lambda sb, age, d: seen.append((sb.id, d)))
    assert seen == [("old", True), ("young", False)]


# --- REAP-01: a bad created_at must not abort the whole sweep -----------------


def _naive_ts(age_minutes: float) -> str:
    """An ISO timestamp with NO Z/offset (tz-naive), the value that used to raise."""
    return (
        (datetime.now(UTC) - timedelta(minutes=age_minutes))
        .replace(tzinfo=None)
        .isoformat()
    )


def test_naive_created_at_does_not_abort_sweep():
    # A naive (offset-less) created_at placed BEFORE a clearly-stale owned
    # sandbox: the stale one must still be reaped (the survivor assertion is the
    # mutation-killer — a fix that swallows the error but drops the rest fails).
    naive = SimpleNamespace(
        id="naive", state="STARTED", created_at=_naive_ts(1500), labels=dict(_OWNED)
    )
    client = FakeClient([naive, _sb("stale", "STARTED", 1500)])
    counts = reap_stale_sandboxes(client)
    assert client.deleted == ["naive", "stale"]  # naive coerced to UTC, both reaped
    assert counts["failed"] == 0


def test_non_string_created_at_is_skipped_not_raised():
    bad = SimpleNamespace(
        id="bad", state="STARTED", created_at=datetime.now(UTC), labels=dict(_OWNED)
    )
    client = FakeClient([bad, _sb("stale", "STARTED", 1500)])
    counts = reap_stale_sandboxes(client)
    # The non-string record is skipped; the following stale sandbox still reaps.
    assert client.deleted == ["stale"]
    assert counts["skipped"] >= 1


# --- REAP-03: activity guard protects genuinely live runs --------------------


def _sb_with_activity(sb_id: str, state: str, age_minutes: float, idle_minutes: float):
    created = datetime.now(UTC) - timedelta(minutes=age_minutes)
    activity = datetime.now(UTC) - timedelta(minutes=idle_minutes)
    return SimpleNamespace(
        id=sb_id,
        state=state,
        created_at=created.isoformat().replace("+00:00", "Z"),
        last_activity_at=activity.isoformat().replace("+00:00", "Z"),
        labels=dict(_OWNED),
    )


def test_old_but_recently_active_sandbox_is_not_reaped():
    # Old by creation (over TTL) but active 1 min ago -> protected, skipped.
    client = FakeClient([_sb_with_activity("live", "STARTED", 1500, idle_minutes=1)])
    counts = reap_stale_sandboxes(client)
    assert client.deleted == []
    assert counts["skipped"] == 1


def test_old_and_idle_sandbox_is_reaped():
    # Old by creation and idle well past the guard window -> reaped.
    client = FakeClient(
        [_sb_with_activity("orphan", "STARTED", 1500, idle_minutes=600)]
    )
    reap_stale_sandboxes(client)
    assert client.deleted == ["orphan"]


def test_old_with_missing_activity_is_reaped():
    # Absent last_activity_at must NOT be protective, or nothing old ever reaps.
    client = FakeClient([_sb("old", "STARTED", 1500)])
    reap_stale_sandboxes(client)
    assert client.deleted == ["old"]


def test_failed_tier_reaps_despite_fresh_activity():
    # A FAILED sandbox reaps on its short TTL regardless of fresh activity.
    client = FakeClient(
        [_sb_with_activity("crashed", "BUILD_FAILED", 180, idle_minutes=1)]
    )
    reap_stale_sandboxes(client)
    assert client.deleted == ["crashed"]


# --- Ownership scoping: foreign sandboxes must never be reaped ----------------


def test_foreign_sandbox_never_reaped_even_when_ancient():
    # Empty label set, 9999 minutes old: still untouched.
    client = FakeClient([_sb("foreign", "STARTED", 9999, labels={})])
    counts = reap_stale_sandboxes(client)
    assert client.deleted == []
    assert counts == {"found": 1, "deleted": 0, "skipped": 1, "failed": 0}


def test_unlabeled_sandbox_never_reaped():
    # No labels attribute at all (older/foreign SDK shape) — treated as foreign.
    created = (datetime.now(UTC) - timedelta(minutes=9999)).isoformat()
    client = FakeClient(
        [SimpleNamespace(id="nolabels", state="STARTED", created_at=created)]
    )
    counts = reap_stale_sandboxes(client)
    assert client.deleted == []
    assert counts["skipped"] == 1


def test_foreign_label_set_never_reaped():
    # Belongs to a different tool/user sharing the API key.
    client = FakeClient([_sb("theirs", "STARTED", 9999, labels={"owner": "alice"})])
    reap_stale_sandboxes(client)
    assert client.deleted == []


def test_wrong_label_value_never_reaped():
    # Same key, wrong value: must not match. Kills a truthiness/`in` mutation of
    # the scope check (value must equal "1", not merely be present).
    client = FakeClient([_sb("v0", "STARTED", 9999, labels={"benchflow.managed": "0"})])
    reap_stale_sandboxes(client)
    assert client.deleted == []


def test_owned_stale_reaped_while_foreign_stale_kept():
    client = FakeClient(
        [
            _sb("bf-stale", "STARTED", 9999),
            _sb("foreign-stale", "STARTED", 9999, labels={"owner": "alice"}),
        ]
    )
    counts = reap_stale_sandboxes(client)
    assert client.deleted == ["bf-stale"]
    assert counts == {"found": 2, "deleted": 1, "skipped": 1, "failed": 0}


def test_owned_with_extra_labels_is_still_reaped():
    # Real sandboxes carry the SDK-injected language label alongside ours; the
    # scope check keys off our label, not exact-dict equality.
    client = FakeClient(
        [
            _sb(
                "bf",
                "STARTED",
                9999,
                labels={"benchflow.managed": "1", "code-toolbox-language": "python"},
            )
        ]
    )
    reap_stale_sandboxes(client)
    assert client.deleted == ["bf"]


def test_on_decision_not_called_for_foreign():
    seen = []
    client = FakeClient(
        [_sb("bf", "STARTED", 9999), _sb("foreign", "STARTED", 9999, labels={})]
    )
    reap_stale_sandboxes(client, on_decision=lambda sb, age, d: seen.append(sb.id))
    assert seen == ["bf"]


class TestIsBenchflowOwned:
    """Direct unit coverage of the scope predicate (mutation surface)."""

    def test_exact_label_is_owned(self):
        assert _is_benchflow_owned(SimpleNamespace(labels={"benchflow.managed": "1"}))

    def test_extra_labels_still_owned(self):
        assert _is_benchflow_owned(
            SimpleNamespace(labels={"benchflow.managed": "1", "x": "y"})
        )

    def test_wrong_value_not_owned(self):
        assert not _is_benchflow_owned(
            SimpleNamespace(labels={"benchflow.managed": "0"})
        )

    def test_missing_key_not_owned(self):
        assert not _is_benchflow_owned(SimpleNamespace(labels={"owner": "alice"}))

    def test_empty_labels_not_owned(self):
        assert not _is_benchflow_owned(SimpleNamespace(labels={}))

    def test_none_labels_not_owned(self):
        assert not _is_benchflow_owned(SimpleNamespace(labels=None))

    def test_no_labels_attr_not_owned(self):
        assert not _is_benchflow_owned(SimpleNamespace(id="x"))

    def test_non_mapping_labels_not_owned(self):
        assert not _is_benchflow_owned(SimpleNamespace(labels=["benchflow.managed"]))


class TestIsBenchflowLabelOrphan:
    """The read-only orphan-leak predicate (label-integrity, mutation surface).

    True only for a sandbox that carries the ``benchflow.`` namespace yet fails
    the strict ownership check — a sandbox we likely created whose ownership
    label drifted, which the age-based reaper would otherwise leak forever.
    """

    def test_namespace_key_without_managed_is_orphan(self):
        assert _is_benchflow_label_orphan(
            SimpleNamespace(labels={"benchflow.run": "abc"})
        )

    def test_drifted_managed_value_is_orphan(self):
        # Exact key, wrong value: namespaced but not owned → orphan.
        assert _is_benchflow_label_orphan(
            SimpleNamespace(labels={"benchflow.managed": "0"})
        )

    def test_owned_is_not_orphan(self):
        # A correctly-labeled (owned) sandbox is never an orphan.
        assert not _is_benchflow_label_orphan(
            SimpleNamespace(labels={"benchflow.managed": "1"})
        )

    def test_owned_with_extra_namespace_label_is_not_orphan(self):
        assert not _is_benchflow_label_orphan(
            SimpleNamespace(labels={"benchflow.managed": "1", "benchflow.run": "x"})
        )

    def test_purely_foreign_labels_not_orphan(self):
        # No benchflow namespace at all → not ours, not flagged.
        assert not _is_benchflow_label_orphan(
            SimpleNamespace(labels={"owner": "alice"})
        )

    def test_empty_labels_not_orphan(self):
        assert not _is_benchflow_label_orphan(SimpleNamespace(labels={}))

    def test_none_labels_not_orphan(self):
        assert not _is_benchflow_label_orphan(SimpleNamespace(labels=None))

    def test_no_labels_attr_not_orphan(self):
        assert not _is_benchflow_label_orphan(SimpleNamespace(id="x"))

    def test_non_mapping_labels_not_orphan(self):
        assert not _is_benchflow_label_orphan(SimpleNamespace(labels=["benchflow.run"]))


class TestOrphanLeakGuard:
    """Reaper flags benchflow-namespaced-but-unlabeled sandboxes, never deletes.

    The guard is read-only: a missing/altered ownership label means we cannot
    prove ownership strongly enough to delete on a shared API key, so the
    sandbox is warned about and skipped — never reaped.
    """

    def _orphan_warnings(self, caplog) -> list[str]:
        return [
            r.getMessage()
            for r in caplog.records
            if r.levelno == logging.WARNING and "orphan leak" in r.getMessage()
        ]

    def test_label_orphan_is_warned_and_never_deleted(self, caplog):
        client = FakeClient(
            [_sb("drifted-bf", "STARTED", 9999, labels={"benchflow.run": "r1"})]
        )
        with caplog.at_level(logging.WARNING, logger="benchflow"):
            counts = reap_stale_sandboxes(client)
        # Read-only: skipped, not deleted, counts contract unchanged.
        assert client.deleted == []
        assert counts == {"found": 1, "deleted": 0, "skipped": 1, "failed": 0}
        warnings = self._orphan_warnings(caplog)
        assert len(warnings) == 1
        # Mutation guards: the warning names the sandbox and the missing label.
        assert "drifted-bf" in warnings[0]
        assert "benchflow.managed" in warnings[0]

    def test_drifted_ownership_value_is_warned(self, caplog):
        # Same key, wrong value — still a benchflow sandbox the reaper can't reap.
        client = FakeClient(
            [_sb("v0", "STARTED", 9999, labels={"benchflow.managed": "0"})]
        )
        with caplog.at_level(logging.WARNING, logger="benchflow"):
            reap_stale_sandboxes(client)
        assert client.deleted == []
        assert len(self._orphan_warnings(caplog)) == 1

    def test_owned_sandbox_is_not_warned(self, caplog):
        # A correctly-labeled stale sandbox is reaped with no orphan warning.
        client = FakeClient([_sb("bf-stale", "STARTED", 9999)])
        with caplog.at_level(logging.WARNING, logger="benchflow"):
            reap_stale_sandboxes(client)
        assert client.deleted == ["bf-stale"]
        assert self._orphan_warnings(caplog) == []

    def test_purely_foreign_sandbox_is_not_warned(self, caplog):
        # Foreign labels (no benchflow namespace) must not trip the guard.
        client = FakeClient([_sb("theirs", "STARTED", 9999, labels={"owner": "alice"})])
        with caplog.at_level(logging.WARNING, logger="benchflow"):
            reap_stale_sandboxes(client)
        assert client.deleted == []
        assert self._orphan_warnings(caplog) == []


class TestCliCleanupWrapper:
    """The CLI cleanup wrapper still works and inherits the ownership scope."""

    def test_cli_wrapper_scopes_to_owned(self, monkeypatch):
        from benchflow.cli import main as cli_main

        client = FakeClient(
            [
                _sb("bf-stale", "STARTED", 9999),
                _sb("foreign-stale", "STARTED", 9999, labels={"owner": "bob"}),
            ]
        )
        monkeypatch.setattr(cli_main, "_daytona_client_or_exit", lambda: client)
        cli_main._cleanup_daytona_sandboxes(dry_run=False, max_age_minutes=1440)
        assert client.deleted == ["bf-stale"]

    def test_cli_wrapper_dry_run_deletes_nothing(self, monkeypatch):
        from benchflow.cli import main as cli_main

        client = FakeClient([_sb("bf-stale", "STARTED", 9999)])
        monkeypatch.setattr(cli_main, "_daytona_client_or_exit", lambda: client)
        cli_main._cleanup_daytona_sandboxes(dry_run=True, max_age_minutes=1440)
        assert client.deleted == []


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

    @pytest.mark.parametrize(
        "value", ["0", "false", "False", "FALSE", "no", "off", " off "]
    )
    def test_env_gate_disables(self, tmp_path, monkeypatch, value):
        started = []
        monkeypatch.setattr(
            "benchflow.evaluation.threading.Thread",
            lambda *a, **k: started.append(k) or SimpleNamespace(start=lambda: None),
        )
        monkeypatch.setenv("BENCHFLOW_DAYTONA_AUTO_REAP", value)
        self._eval(tmp_path, "daytona")._maybe_start_daytona_reap()
        assert started == [], f"{value!r} should disable the reaper"

    @pytest.mark.parametrize("value", ["1", "true", "on", "yes", "anything-else"])
    def test_env_gate_enabled_for_non_disable_tokens(
        self, tmp_path, monkeypatch, value
    ):
        started = []

        class FakeThread:
            def __init__(self, *a, **k):
                started.append(k.get("name"))

            def start(self):
                pass

        monkeypatch.setattr("benchflow.evaluation.threading.Thread", FakeThread)
        monkeypatch.setenv("BENCHFLOW_DAYTONA_AUTO_REAP", value)
        self._eval(tmp_path, "daytona")._maybe_start_daytona_reap()
        assert started == ["daytona-auto-reap"], f"{value!r} should keep reaper on"
