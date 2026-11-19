"""A refused verifier tool path is an ending of its own, never a verifier crash.

Hardening leaves a folder of the image's PATH out of the verifier's PATH when
a user other than root could write it (the official rust image's 0777
``/usr/local/cargo/bin``) or it is relative; a task whose test.sh needs a
program there declares it in ``[verifier] required_tools``. With no contract,
or a declared program only in such a folder, test.sh does not run. That used
to raise a bare RuntimeError: the trial read "verifier crashed: RuntimeError:
Verifier hardening failed: ...", filed under ``verifier_failure``, a task
problem, and ``bench check --level equivalence`` called every CRUST-Bench
oracle failing. It is now ``verifier tools refused: ...``
(``verifier_tools_refused``): unscored, out of ``score_excl_errors``, never
retried, a setup problem, and the equivalence report says ``unscored`` with
the reason. The real-Docker job is in
tests/test_verifier_tools_refused_docker.py.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow._utils.scoring import (
    _VERIFIER_TOOLS_REFUSED_HEAD,
    VERIFIER_FAILED,
    VERIFIER_TOOLS_REFUSED,
    classify_verifier_error,
    count_unusable_rewards,
    is_unusable_reward,
)
from benchflow._utils.trusted_path import TrustedPathReport
from benchflow.sandbox.lockdown import _check_required_path_tools
from benchflow.task.config import TaskConfig
from benchflow.task.equivalence import GradeOutcome, run_equivalence_battery
from benchflow.task.verifier_errors import (
    VERIFIER_TOOLS_REFUSED_ERROR,
    VerifierToolsRefusedError,
)
from tests._receipt_fakes import receipt_mock

DROPPED = (("/usr/local/cargo/bin", "root's, but writable by group or others"),)
REFUSED = (
    VERIFIER_TOOLS_REFUSED_ERROR
    + "the verifier's PATH leaves out folders of the image's PATH, and the task "
    "declares no [verifier] required_tools contract (complete PATH "
    "dependencies; [] for none), so test.sh did not run. Make each folder "
    "root's alone in the image, or declare required_tools; use absolute paths "
    "for uploaded/generated helpers. Entries left out: /usr/local/cargo/bin: "
    "root's, but writable by group or others"
)
# What rc3 recorded for every CRUST-Bench task (prep-b's records).
RC3_CRASH = (
    "verifier crashed: RuntimeError: Verifier hardening failed: omitted PATH "
    "entries (including relative entries) require an explicit [verifier] "
    "required_tools contract (complete PATH dependencies; [] for none). Use "
    "absolute paths for uploaded/generated helpers. Entries left out: "
    "/usr/local/cargo/bin: root's, but writable by group or others"
)


def test_the_recorded_head_is_the_classifier_s():
    assert VERIFIER_TOOLS_REFUSED_ERROR == _VERIFIER_TOOLS_REFUSED_HEAD


@pytest.mark.parametrize(
    ("verifier_error", "category"),
    [
        (REFUSED, VERIFIER_TOOLS_REFUSED),
        ("[solver-preserved] " + REFUSED, VERIFIER_TOOLS_REFUSED),
        # rc3's crash stays what it was recorded as.
        (RC3_CRASH, VERIFIER_FAILED),
        # Only BenchFlow's own words at the start decide: test.sh's output
        # quoted inside another error never makes it a refusal.
        ("verifier crashed: test.sh printed: " + REFUSED, VERIFIER_FAILED),
    ],
)
def test_the_classifier_reads_the_ending_at_the_start(verifier_error, category):
    assert classify_verifier_error(verifier_error) == category


def _row(task, *, reward=None, verifier_error=None):
    return {
        "task_name": task,
        "rewards": None if reward is None else {"reward": reward},
        "error": None,
        "verifier_error": verifier_error,
        "verifier_error_category": classify_verifier_error(verifier_error),
    }


ROWS = [
    _row("pass-a", reward=1.0),
    _row("pass-b", reward=1.0),
    _row("fail", reward=0.0),
    _row("crust-like", verifier_error=REFUSED),
]


def test_the_trial_is_unscored_not_a_failure_in_every_score():
    """benchflow.score.v1: out of score_excl_errors, never a refusal there."""
    from benchflow.scoring import outcome, score

    outcomes = [outcome(row) for row in ROWS]
    assert outcomes == ["passed", "passed", "failed", "verifier_errored"]
    rule = score(
        [(row["task_name"], kind) for row, kind in zip(ROWS, outcomes, strict=True)]
    )
    assert (rule.passed, rule.n, rule.n_excl_errors) == (2, 4, 3)
    assert rule.score_excl_errors == pytest.approx(2 / 3)
    assert not is_unusable_reward(REFUSED)
    assert count_unusable_rewards(ROWS) == 0


def test_the_job_metrics_leave_it_out_of_the_scored_only_figure():
    from benchflow.metrics import BenchmarkMetrics, TaskMetrics

    metrics = BenchmarkMetrics(
        benchmark="crust-like",
        agent="oracle",
        model=None,
        tasks=[
            TaskMetrics(
                task_name=row["task_name"],
                reward=(row["rewards"] or {}).get("reward"),
                verifier_error=row["verifier_error"],
            )
            for row in ROWS
        ],
    )
    assert metrics.score_excl_errors == pytest.approx(2 / 3)


def test_it_is_never_run_again():
    """The same image meets the same hardening: no rerun, no recovery."""
    from benchflow.evaluation import RetryConfig
    from benchflow.rollout._verifier_recovery import (
        needs_verifier_recovery,
        verifier_error_reruns_trial,
    )

    assert not verifier_error_reruns_trial(REFUSED)
    assert not verifier_error_reruns_trial(REFUSED, on_resume=True)
    assert not needs_verifier_recovery(REFUSED)
    retry = RetryConfig()
    assert not retry.should_retry_verifier_error(REFUSED)
    assert not retry.reruns_verifier_error_on_resume(REFUSED)


def test_bench_explain_calls_it_a_setup_problem():
    from benchflow.cli.explain import _BODIES
    from benchflow.failures import FAULT_WORDS, cause_of, unscored_keys

    cause = cause_of(ROWS[3])
    assert cause is not None and cause.key == "verifier_tools_refused"
    assert cause.fault == "setup"
    assert FAULT_WORDS[cause.fault] != "task problem"
    assert "verifier_tools_refused" in unscored_keys()
    assert "verifier_tools_refused" in _BODIES
    assert "required_tools" in _BODIES["verifier_tools_refused"]


# --- hardening raises the typed refusal ----------------------------------------


async def test_a_left_out_folder_without_a_contract_is_refused_not_crashed():
    env = SimpleNamespace(exec=receipt_mock(SimpleNamespace(return_code=0, stdout="")))
    report = TrustedPathReport(path="/usr/bin:/bin", dropped=DROPPED)
    verifier = TaskConfig.model_validate({"verifier": {}}).verifier
    with pytest.raises(VerifierToolsRefusedError) as raised:
        await _check_required_path_tools(env, report, verifier)
    message = str(raised.value)
    assert "no [verifier] required_tools contract" in message
    assert "/usr/local/cargo/bin: root's, but writable by group or others" in message
    assert VERIFIER_TOOLS_REFUSED_ERROR + message == REFUSED
    # Nothing ran: the refusal is decided from the report alone.
    env.exec.assert_not_called()


async def test_a_declared_tool_only_in_a_left_out_folder_is_refused():
    env = SimpleNamespace(
        exec=receipt_mock(SimpleNamespace(return_code=0, stdout="cargo\x00"))
    )
    report = TrustedPathReport(path="/usr/bin:/bin", dropped=DROPPED)
    verifier = TaskConfig.model_validate(
        {"verifier": {"required_tools": ["cargo", "cmp"]}}
    ).verifier
    with pytest.raises(VerifierToolsRefusedError) as raised:
        await _check_required_path_tools(env, report, verifier)
    message = str(raised.value)
    assert (
        "required_tools absent from the verifier's trusted PATH: ['cargo']" in message
    )
    assert "/usr/local/cargo/bin" in message
    recorded = VERIFIER_TOOLS_REFUSED_ERROR + message
    assert classify_verifier_error(recorded) == VERIFIER_TOOLS_REFUSED


async def test_a_contract_check_that_could_not_run_stays_a_hardening_failure():
    """Not the image's decision: BenchFlow's own exec failed."""
    env = SimpleNamespace(exec=receipt_mock(SimpleNamespace(return_code=1, stdout="")))
    report = TrustedPathReport(path="/usr/bin:/bin", dropped=DROPPED)
    verifier = TaskConfig.model_validate(
        {"verifier": {"required_tools": ["cargo"]}}
    ).verifier
    with pytest.raises(RuntimeError) as raised:
        await _check_required_path_tools(env, report, verifier)
    assert not isinstance(raised.value, VerifierToolsRefusedError)


def test_the_rollout_records_the_ending_before_the_generic_crash():
    """``_verify_hardened`` catches the refusal before ``except Exception``."""
    import inspect

    from benchflow.rollout import _setup

    source = inspect.getsource(_setup._verify_hardened)
    refused = source.index("except VerifierToolsRefusedError as e:")
    generic = source.index("except Exception as e:")
    assert refused < generic
    assert 'verifier_error = f"{VERIFIER_TOOLS_REFUSED_ERROR}{e}"' in source


# --- the equivalence check -----------------------------------------------------


class _Grader:
    def __init__(self, control: GradeOutcome) -> None:
        self.control = control
        self.calls: list[Mapping[str, bytes | None]] = []

    async def grade(self, changes):
        self.calls.append(changes)
        return self.control


def _task(tmp_path: Path) -> Path:
    task = tmp_path / "crust-like"
    (task / "tests").mkdir(parents=True)
    (task / "instruction.md").write_text("Port the C library to safe Rust.\n")
    (task / "tests" / "test.sh").write_text("#!/bin/bash\ncargo test\n")
    return task


@pytest.mark.parametrize(
    ("error", "status", "reason"),
    [
        (REFUSED, "unscored", "the oracle output was not scored: " + REFUSED),
        (
            "verifier crashed: test.sh failed",
            "oracle-fails",
            "oracle output produced no reward: verifier crashed: test.sh failed",
        ),
    ],
)
def test_equivalence_reports_a_refused_tool_path_unscored(
    tmp_path, error, status, reason
):
    grader = _Grader(GradeOutcome(None, error))
    report = asyncio.run(
        run_equivalence_battery(
            _task(tmp_path), grader, {"/workspace/src/lib.rs": b"pub fn f() {}\n"}
        )
    )
    assert (report.status, report.reason) == (status, reason)
    assert report.issues() == [f"equivalence: {reason}"]
    assert report.to_dict()["status"] == status
    # The battery stops at the control: nothing else is graded.
    assert len(grader.calls) == 1


# --- an absent PATH entry only root could make ----------------------------------


async def test_an_absent_record_keeps_the_entry_in_its_place():
    from benchflow.sandbox.lockdown import _judge_trusted_entries

    out = "A /root/.cargo/bin\nK /opt/k\nA /etc/not-asked\n"
    env = SimpleNamespace(exec=receipt_mock(SimpleNamespace(return_code=0, stdout=out)))
    report = await _judge_trusted_entries(
        env, "/root/.local/bin:/root/.cargo/bin:/opt/k:/usr/bin", (), None
    )
    assert report.kept == ("/root/.cargo/bin", "/opt/k")
    assert report.absent_root_only == ("/root/.cargo/bin",)
    assert report.path.split(":")[:2] == ["/root/.cargo/bin", "/opt/k"]
    assert report.to_record()["absent_root_only"] == ["/root/.cargo/bin"]
    # /root/.local/bin had no record: still "not present", as before.
    assert "/root/.local/bin" not in report.kept


def _judge_on_this_host(entry: str) -> str:
    import subprocess

    from benchflow.sandbox.lockdown import _trusted_path_extras_cmd

    return subprocess.run(
        ["/bin/sh", "-c", _trusted_path_extras_cmd(entry, (), None)],
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout


def _root_only_folder() -> str:
    """A folder in which only root could make an entry on this host, the whole
    chain root's and written by no group or other user. Not assumed: GitHub's
    ubuntu-24.04 runner image leaves /usr/share writable by others."""
    import os
    import stat

    for folder in ("/usr/share", "/usr/lib", "/usr/libexec", "/usr/include", "/opt"):
        chain = [Path(folder), *Path(folder).parents]
        try:
            if all(
                (st := os.stat(p)).st_uid == 0
                and not st.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
                for p in chain
            ):
                return folder
        except OSError:
            continue
    pytest.skip("no folder on this host that only root could write")


def test_the_judge_keeps_an_absent_entry_under_root_only_folders():
    """An absent entry under folders only root could write is kept: only root
    could make it."""
    import uuid

    from benchflow._utils.trusted_path import decode_path_field

    entry = f"{_root_only_folder()}/benchflow-absent-{uuid.uuid4().hex}/bin"
    out = _judge_on_this_host(entry)
    records = [line.partition(" ") for line in out.splitlines()]
    assert [(tag, decode_path_field(value)) for tag, _, value in records] == [
        ("A", entry)
    ]


def test_the_judge_refuses_an_absent_entry_another_user_could_make(tmp_path):
    from benchflow.sandbox.lockdown import _parse_trusted_path_output

    entry = f"{tmp_path}/tools/bin"
    kept, _, dropped = _parse_trusted_path_output(_judge_on_this_host(entry), [entry])
    assert kept == []
    ((folder, why),) = dropped
    assert folder == entry
    assert why.startswith("not present when the verifier PATH was checked, and ")
    # The first part another user could write: the shared temp folder or
    # this test's own folders.
    assert "writable by group or others" in why or "is not root's (uid " in why


def test_the_judge_refuses_a_dotdot_part_as_written():
    """RUSAGE on the rule holds literally. `/usr/share/a/../b/bin`
    would walk only root's folders, but a `..` part is refused, not
    followed, so no spelling of a path decides the walk."""
    import uuid

    from benchflow.sandbox.lockdown import _parse_trusted_path_output

    entry = f"/usr/share/benchflow-absent-{uuid.uuid4().hex}/../share/x/bin"
    kept, _, dropped = _parse_trusted_path_output(_judge_on_this_host(entry), [entry])
    assert kept == []
    assert dropped == [
        (entry, "not present when the verifier PATH was checked, and it has a .. part")
    ]
