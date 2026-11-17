"""a verifier that cannot write its own output file is BenchFlow's failure.

On Docker BenchFlow makes the trial's verifier/test-stdout.txt on the host, as
the host's user, mode 0644, before the test command; the folder is the
sandbox's /logs/verifier. A verifier that runs as another user, not root
(``[verifier] user``, else the image's own USER), cannot open it: the shell's
redirect fails with exit 2, test.sh never runs, and the trial used to end as
"verifier wrote no usable reward", a failure in every score (the task's
problem). The verifier's user now opens the file first; when it cannot, the
trial ends as a verifier setup failure (``verifier_infra``): unscored, out of
``score_excl_errors``, never "task problem". One classifier,
``scoring.reward_less_verdict``, decides refused (a failure in every score)
against unscored for summary.json, ``benchflow.score.v1``, the job's metrics,
the CLI line and ``bench explain``. The real-Docker job is in
tests/test_verifier_output_guard_docker.py.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow._utils.scoring import (
    NO_USABLE_REWARD,
    VERIFIER_FAILED,
    VERIFIER_INFRA,
    VERIFIER_OUTPUT_UNWRITABLE,
    classify_verifier_error,
    count_unusable_rewards,
    is_output_unwritable,
    is_unusable_reward,
    reward_less_verdict,
)

# What the rollout records for each (rollout/_setup.py's _verify_rollout).
UNWRITABLE = (
    "verifier crashed: Verifier setup failed: "
    + VERIFIER_OUTPUT_UNWRITABLE
    + "/logs/verifier/test-stdout.txt: the verifier runs as uid 65534, and the "
    "file is uid 1001's, mode 0644, made by BenchFlow on the host before the "
    'test command. Run the verifier as root ([verifier] user = "root") until '
    "BenchFlow leaves that file to the test command"
)
NO_REWARD = (
    NO_USABLE_REWARD + "verifier exited with rc=0; no reward file found at "
    "verifier/reward.txt or verifier/reward.json"
)


def _row(task: str, *, reward=None, verifier_error=None, category=None) -> dict:
    return {
        "task_name": task,
        "rewards": None if reward is None else {"reward": reward},
        "error": None,
        "verifier_error": verifier_error,
        "verifier_error_category": category,
    }


ROWS = [
    _row("pass-a", reward=1.0),
    _row("pass-b", reward=1.0),
    _row("pass-c", reward=1.0),
    _row("task-problem", verifier_error=NO_REWARD, category=VERIFIER_FAILED),
    _row("verifier-infra", verifier_error=UNWRITABLE, category=VERIFIER_INFRA),
]


@pytest.mark.parametrize(
    ("verifier_error", "verdict", "category"),
    [
        (NO_REWARD, "refused", VERIFIER_FAILED),
        ("[solver-preserved] " + NO_REWARD, "refused", VERIFIER_FAILED),
        (UNWRITABLE, "unscored", VERIFIER_INFRA),
        ("[solver-preserved] " + UNWRITABLE, "unscored", VERIFIER_INFRA),
        # The class name verifier_failure_reason may put before the message.
        (
            UNWRITABLE.replace("crashed: ", "crashed: VerifierSetupError: "),
            "unscored",
            VERIFIER_INFRA,
        ),
        (None, None, None),
    ],
)
def test_one_classifier_decides_refused_against_unscored(
    verifier_error, verdict, category
):
    assert reward_less_verdict(verifier_error) == verdict
    assert classify_verifier_error(verifier_error) == category
    assert is_unusable_reward(verifier_error) is (verdict == "refused")
    assert is_output_unwritable(verifier_error) is (category == VERIFIER_INFRA)


def test_the_unwritable_marker_counts_only_at_the_start():
    """Text the agent chose sits after BenchFlow's own words, never before."""
    planted = NO_USABLE_REWARD + "output said: " + UNWRITABLE
    assert reward_less_verdict(planted) == "refused"
    assert not is_output_unwritable(planted)
    assert not is_output_unwritable("verifier exited: " + UNWRITABLE)


def test_score_v1_keeps_the_infra_trial_out_of_the_error_excluding_score():
    """benchflow.score.v1: score is passed over every trial (an unscored one
    as a fail, as every errored trial is); score_excl_errors keeps passed,
    failed and refused, so the infra trial is out of its denominator and the
    task's problem stays in as a failure."""
    from benchflow.scoring import outcome, score

    outcomes = [outcome(row) for row in ROWS]
    assert outcomes == ["passed", "passed", "passed", "refused", "verifier_errored"]
    rule = score(
        [(row["task_name"], kind) for row, kind in zip(ROWS, outcomes, strict=True)]
    )
    assert (rule.passed, rule.n, rule.n_excl_errors) == (3, 5, 4)
    assert rule.score == pytest.approx(3 / 5)
    assert rule.score_excl_errors == pytest.approx(3 / 4)
    # Without the task's problem, the infra trial alone leaves 3 of 3.
    alone = [r for r in ROWS if r["task_name"] != "task-problem"]
    rule = score([(r["task_name"], outcome(r)) for r in alone])
    assert (rule.passed, rule.n_excl_errors, rule.score_excl_errors) == (3, 3, 1.0)


def test_summary_counts_take_the_same_answer():
    from benchflow._utils.scoring import count_guard_refusals, mean_scored_reward

    assert count_unusable_rewards(ROWS) == 1
    assert count_guard_refusals(ROWS) == 0
    # The task's problem counts as 0 in the scored mean; the infra trial is
    # out of it.
    assert mean_scored_reward(ROWS) == pytest.approx(3 / 4)


def test_the_job_metrics_take_the_same_answer():
    from benchflow.metrics import BenchmarkMetrics, TaskMetrics

    tasks = [
        TaskMetrics(
            task_name=row["task_name"],
            reward=(row["rewards"] or {}).get("reward"),
            verifier_error=row["verifier_error"],
        )
        for row in ROWS
    ]
    metrics = BenchmarkMetrics(
        benchmark="en675", agent="oracle", model=None, tasks=tasks
    )
    assert metrics.score_excl_errors == pytest.approx(3 / 4)


@pytest.mark.parametrize(
    ("row", "key", "fault"),
    [
        (ROWS[3], "verifier_no_usable_reward", "task"),
        (ROWS[4], "verifier_output_unwritable", "infrastructure"),
    ],
)
def test_bench_explain_names_the_cause_and_whose_problem(tmp_path, row, key, fault):
    from benchflow.cli.explain import _BODIES as EXPLANATIONS
    from benchflow.failures import FAULT_WORDS, cause_of, unscored_keys

    cause = cause_of(row)
    assert cause is not None and cause.key == key
    assert cause.fault == fault
    assert key in unscored_keys() and key in EXPLANATIONS
    if key == "verifier_output_unwritable":
        assert FAULT_WORDS[cause.fault] != "task problem"
        assert "task problem" not in EXPLANATIONS[key]


# --- the check before the test command -----------------------------------------


class _Sandbox:
    """Records each exec. The probe answers *answer* (or raises); the uid read
    answers *uid*. *hardening_uid* is what hardening's read of the verifier's
    uid left on the sandbox ("" when it could not tell; None: never hardened,
    as a stand-in sandbox)."""

    def __init__(self, answer="BF stdout-unwritable", uid="65534", hardening_uid=""):
        self.answer = answer
        self.uid = uid
        if hardening_uid is not None:
            self._benchflow_verifier_uid = hardening_uid
        self.calls: list[tuple[str, object]] = []

    async def exec(self, command, user=None, **kwargs):
        self.calls.append((command, user))
        if "true >>" in command:
            if isinstance(self.answer, Exception):
                raise self.answer
            return SimpleNamespace(stdout=self.answer + "\n", stderr="", return_code=0)
        return SimpleNamespace(stdout=self.uid + "\n", stderr="", return_code=0)


def _verifier(tmp_path: Path, sandbox: _Sandbox, user):
    from benchflow._utils.reachable_write import ReachablePath
    from benchflow.task.verifier_core import Verifier

    verifier = Verifier.__new__(Verifier)
    verifier._sandbox = sandbox
    verifier._task = SimpleNamespace(
        config=SimpleNamespace(verifier=SimpleNamespace(user=user))
    )
    folder = tmp_path / "verifier"
    folder.mkdir()
    (folder / "test-stdout.txt").write_text("")
    os.chmod(folder / "test-stdout.txt", 0o644)
    verifier._rollout_paths = SimpleNamespace(
        test_stdout_path=ReachablePath.rollout(tmp_path, "verifier") / "test-stdout.txt"
    )
    return verifier


async def _check(tmp_path, sandbox, user):
    await _verifier(tmp_path, sandbox, user)._require_writable_test_stdout(
        "/logs/verifier/test-stdout.txt", "main"
    )


def _probes(sandbox):
    return [call for call in sandbox.calls if "true >>" in call[0]]


@pytest.mark.parametrize("user", ["root", "0", 0])
async def test_a_root_verifier_is_never_checked(tmp_path, user):
    sandbox = _Sandbox(hardening_uid="")
    await _check(tmp_path, sandbox, user)
    assert sandbox.calls == []


@pytest.mark.parametrize("user", [None, "tester"])
async def test_a_root_answer_from_hardening_adds_no_exec(tmp_path, user):
    """A root image: hardening's read heard uid 0."""
    sandbox = _Sandbox(hardening_uid="0")
    await _check(tmp_path, sandbox, user)
    assert sandbox.calls == []


@pytest.mark.parametrize("user", [None, "tester"])
async def test_a_sandbox_never_hardened_is_left_alone(tmp_path, user):
    sandbox = _Sandbox(hardening_uid=None)
    await _check(tmp_path, sandbox, user)
    assert sandbox.calls == []


@pytest.mark.parametrize(
    "answer", ["BF stdout-writable", "", "something else", RuntimeError("lost")]
)
async def test_only_the_probe_s_own_refusal_stops_the_test(tmp_path, answer):
    sandbox = _Sandbox(answer=answer, hardening_uid="")
    await _check(tmp_path, sandbox, None)
    assert len(_probes(sandbox)) == 1


@pytest.mark.parametrize(
    ("user", "hardening_uid"), [(None, ""), (None, "65534"), ("nobody", "")]
)
async def test_an_unwritable_file_is_a_setup_failure_naming_file_owner_and_uid(
    tmp_path, user, hardening_uid
):
    from benchflow.task.verifier_errors import VerifierSetupError

    sandbox = _Sandbox(hardening_uid=hardening_uid)
    with pytest.raises(VerifierSetupError) as raised:
        await _check(tmp_path, sandbox, user)
    message = str(raised.value)
    ((probe, probe_user),) = _probes(sandbox)
    assert probe_user == user and "/logs/verifier/test-stdout.txt" in probe
    assert "/logs/verifier/test-stdout.txt" in message
    assert "uid 65534" in message
    assert f"uid {os.getuid()}'s, mode 0644" in message
    # As the rollout records it, the shared classifier calls it infra.
    recorded = f"verifier crashed: {message}"
    assert reward_less_verdict(recorded) == "unscored"
    assert classify_verifier_error(recorded) == VERIFIER_INFRA


# --- never retried: the same file meets the same verifier user again ----------

OTHER_INFRA = (
    "verifier crashed: Verifier setup failed: the verifier user's uid could not "
    "be read (rc=1)"
)


@pytest.mark.parametrize(
    ("verifier_error", "reruns"),
    [(UNWRITABLE, False), (OTHER_INFRA, True)],
)
def test_the_check_s_failure_is_never_run_again(verifier_error, reruns):
    from benchflow.evaluation import RetryConfig
    from benchflow.rollout._verifier_recovery import (
        needs_verifier_recovery,
        verifier_error_reruns_trial,
    )

    assert classify_verifier_error(verifier_error) == VERIFIER_INFRA
    assert verifier_error_reruns_trial(verifier_error) is reruns
    assert verifier_error_reruns_trial(verifier_error, on_resume=True) is reruns
    assert needs_verifier_recovery(verifier_error) is reruns
    retry = RetryConfig()
    assert retry.should_retry_verifier_error(verifier_error) is reruns
    assert retry.reruns_verifier_error_on_resume(verifier_error) is reruns


@pytest.mark.parametrize("shell", ["sh", "bash", "dash", "busybox"])
def test_the_probe_answers_in_a_real_shell(tmp_path, shell):
    """The probe's own text, run by the shells images ship, against a file it
    may and may not open (a dash ``:`` with a refused redirect exits 2)."""
    import shlex
    import shutil
    import subprocess

    from benchflow.task.verifier_core import stdout_probe_command

    argv = [shell, "sh"] if shell == "busybox" else [shell]
    if shutil.which(argv[0]) is None:
        pytest.skip(f"no {shell} here")
    writable = tmp_path / "writable.txt"
    writable.write_text("kept\n")
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    target = blocked / "test-stdout.txt"
    target.write_text("")
    target.chmod(0o444)
    if os.access(target, os.W_OK):
        pytest.skip("this user can write any file (root)")
    for path, answer in (
        (writable, "BF stdout-writable"),
        (target, "BF stdout-unwritable"),
    ):
        out = subprocess.run(
            [*argv, "-c", stdout_probe_command(shlex.quote(str(path)))],
            capture_output=True,
            text=True,
        )
        assert out.returncode == 0, out
        assert out.stdout.strip() == answer, out
        assert out.stderr == "", out
    assert writable.read_text() == "kept\n"


# --- one classifier, every reader (the lead's rule) -----------------------------

GUARD_REFUSAL = (
    "verifier crashed: PluginGuardLoadError: the pytest plugin guard "
    "_runtime_guard_0123abcd refused a pytest plugin it could not trust: "
    "/app/plugin.py"
)
OTHER_INFRA_KIND = (
    "verifier crashed: VerifierSetupError: Verifier setup failed: the verifier "
    "user's uid could not be read (rc=1)"
)
# R647's seven kinds: the verdict, the category results.jsonl carries, the
# cause and whose problem the CLI and report.md name.
KINDS = {
    "no_usable_reward": (
        NO_REWARD,
        "refused",
        VERIFIER_FAILED,
        "verifier_no_usable_reward",
        "task",
    ),
    "no_usable_reward_preserved": (
        "[solver-preserved] " + NO_REWARD,
        "refused",
        VERIFIER_FAILED,
        "verifier_no_usable_reward",
        "task",
    ),
    "guard_refusal": (
        GUARD_REFUSAL,
        "refused",
        VERIFIER_FAILED,
        "verifier_plugin_refused",
        "agent",
    ),
    "output_unwritable": (
        UNWRITABLE,
        "unscored",
        VERIFIER_INFRA,
        "verifier_output_unwritable",
        "infrastructure",
    ),
    "output_unwritable_preserved": (
        "[solver-preserved] " + UNWRITABLE,
        "unscored",
        VERIFIER_INFRA,
        "verifier_output_unwritable",
        "infrastructure",
    ),
    "other_infra": (
        OTHER_INFRA_KIND,
        "unscored",
        VERIFIER_INFRA,
        "verifier_infra",
        "infrastructure",
    ),
    "timeout": (
        "verifier timed out after 120.0s",
        "unscored",
        "verifier_timeout",
        "verifier_timeout",
        "task",
    ),
}


def _kind_row(verifier_error: str) -> dict:
    return {
        "task_name": "kind",
        "rewards": None,
        "error": None,
        "verifier_error": verifier_error,
    }


PASSED = {
    "task_name": "pass",
    "rewards": {"reward": 1.0},
    "error": None,
    "verifier_error": None,
}


@pytest.mark.parametrize("kind", KINDS)
def test_summary_json_counts_take_the_verdict(kind):
    """summary.json: guard_refusals, unusable_rewards and score_excl_errors."""
    from benchflow._utils.scoring import count_guard_refusals, score_summary_fields

    verifier_error, verdict, *_ = KINDS[kind]
    row = _kind_row(verifier_error)
    assert reward_less_verdict(verifier_error) == verdict
    refused = count_guard_refusals([row]) + count_unusable_rewards([row])
    assert refused == (1 if verdict == "refused" else 0)
    fields = score_summary_fields([PASSED, row])
    assert fields["guard_refusals"] + fields["unusable_rewards"] == refused
    # One passed trial beside it: a refusal stays in the error-excluding score
    # as a failure, an unscored trial leaves it.
    assert fields["score_excl_errors"] == (
        "50.0%" if verdict == "refused" else "100.0%"
    )


@pytest.mark.parametrize("kind", KINDS)
def test_results_jsonl_category_takes_the_verdict(kind):
    """results.jsonl: the row's info.diagnostics.verifier_error_category."""
    from benchflow.trajectories.results import _structured_diagnostics_info

    verifier_error, verdict, category, *_ = KINDS[kind]
    assert classify_verifier_error(verifier_error) == category
    info = _structured_diagnostics_info(
        error_category=None,
        verifier_error_category=classify_verifier_error(verifier_error),
        diagnostics=None,
    )
    assert info["diagnostics"]["verifier_error_category"] == category
    if verdict == "refused":
        assert category == VERIFIER_FAILED
    else:
        assert category != VERIFIER_FAILED


@pytest.mark.parametrize("kind", KINDS)
def test_the_cli_and_report_cause_take_the_verdict(kind):
    """The CLI's and report.md's cause line and whose problem it is, and
    benchflow.score.v1's outcome."""
    from benchflow.failures import cause_of
    from benchflow.scoring import outcome

    verifier_error, verdict, _, key, fault = KINDS[kind]
    row = _kind_row(verifier_error)
    cause = cause_of(row)
    assert (cause.key, cause.fault) == (key, fault)
    assert outcome(row) == ("refused" if verdict == "refused" else "verifier_errored")
    if key in ("verifier_output_unwritable", "verifier_infra"):
        assert fault == "infrastructure"


# --- every category classify_verifier_error returns (merged with #234) ----------

_PRESERVED = "[solver-preserved] "


def _categories():
    from benchflow._utils import scoring as sc

    return [
        # #234's ending-based categories, plain and recovery-tagged.
        (
            "network policy breached: the rule count fell",
            sc.NETWORK_POLICY_BREACHED,
            True,
        ),
        (
            "verifier network refused: allowlist on a shared verifier",
            sc.VERIFIER_NETWORK_REFUSED,
            True,
        ),
        (
            "verifier launch refused: test.sh would run unheld",
            sc.VERIFIER_LAUNCH_REFUSED,
            True,
        ),
        (
            "verifier network not held: the rules did not go in",
            sc.VERIFIER_NETWORK_UNHELD,
            True,
        ),
        # Refused (reward_less_verdict): no usable reward, a guard refusal.
        (NO_REWARD, VERIFIER_FAILED, True),
        (GUARD_REFUSAL, VERIFIER_FAILED, True),
        # Its unwritable output.
        (UNWRITABLE, VERIFIER_INFRA, True),
        # A guard message that is not a refusal.
        (
            "verifier crashed: PluginGuardLoadError: pytest plugin foo was installed "
            "after the agent stopped, by the verifier",
            VERIFIER_FAILED,
            False,
        ),
        (
            "verifier crashed: PluginGuardMarkersUnreadableError: markers lost",
            VERIFIER_INFRA,
            True,
        ),
        ("verifier crashed: RuntimeUnavailable: pytest-x 1.0", VERIFIER_INFRA, True),
        (
            "separate verifier sandbox never started: create failed",
            VERIFIER_INFRA,
            False,
        ),
        (
            "verifier recovery failed: reward out of range: 7 outside [0, 1]",
            sc.VERIFIER_REWARD_OUT_OF_RANGE,
            False,
        ),
        (
            "verifier recovery failed: verifier timed out after 60s",
            sc.VERIFIER_TIMEOUT,
            True,
        ),
        ("verifier crashed: verifier_wedge: no output", VERIFIER_INFRA, True),
        (
            "verifier crashed: OSError: " + sc.DISK_FULL_MARKER,
            sc.DISK_FULL,
            False,
        ),
        (
            "verifier crashed: dependency install failed for pytest",
            sc.VERIFIER_DEP_INSTALL,
            False,
        ),
        (OTHER_INFRA_KIND, VERIFIER_INFRA, False),
        ("verifier crashed: ValueError: boom", VERIFIER_FAILED, False),
        ("verifier timed out after 120.0s", sc.VERIFIER_TIMEOUT, False),
        ("the verifier left something else", sc.VERIFIER_OTHER, False),
    ]


def test_every_category_classify_verifier_error_returns():
    """One row per category the merged classifier returns, with the
    recovery-tagged form where it keeps its category."""
    from benchflow._utils import scoring as sc

    seen = set()
    for message, category, preserved_too in _categories():
        assert classify_verifier_error(message) == category, message
        seen.add(category)
        if preserved_too:
            tagged = _PRESERVED + message
            assert classify_verifier_error(tagged) == category, tagged
    assert seen == {
        sc.NETWORK_POLICY_BREACHED,
        sc.VERIFIER_NETWORK_REFUSED,
        sc.VERIFIER_LAUNCH_REFUSED,
        sc.VERIFIER_NETWORK_UNHELD,
        VERIFIER_FAILED,
        VERIFIER_INFRA,
        sc.VERIFIER_REWARD_OUT_OF_RANGE,
        sc.VERIFIER_TIMEOUT,
        sc.DISK_FULL,
        sc.VERIFIER_DEP_INSTALL,
        sc.VERIFIER_OTHER,
    }


@pytest.mark.parametrize(
    "message",
    [
        "network policy breached: x",
        "verifier network refused: x",
        "verifier launch refused: x",
        "verifier network not held: x",
    ],
)
def test_the_network_endings_are_unscored_never_refused(message):
    """#234's categories stay unscored under the shared verdict, tagged or not."""
    assert reward_less_verdict(message) == "unscored"
    assert reward_less_verdict(_PRESERVED + message) == "unscored"
