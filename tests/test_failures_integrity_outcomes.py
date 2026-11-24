"""the two integrity outcomes, as the run summary and the docs name them.

``integrity_unavailable`` is one cause, BenchFlow's, whether the trial was
refused before its turn (bfx can't run in the sandbox) or its grading had no
record of the links from before the turn. rc4's tag run filed the first as
"other (infrastructure)"; the second was "verifier_other (task problem)".
A scored ``provenance_incomplete`` 0 stays a score with no cause, and has its
own explain entry. tools/dt reads both in a job's summary.json.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from rich.console import Console

import benchflow.cli._shared as shared
from benchflow import failures
from benchflow._utils import scoring
from benchflow.evaluation import EvaluationConfig, EvaluationResult
from benchflow.models import RolloutResult
from benchflow.rollout import _grading_outcome, _reward_provenance
from benchflow.sandbox import _verifier_links

# The refusal before the turn is the subject: no stand-in bfx (conftest).
pytestmark = pytest.mark.bfx_refusal

KEY = "integrity_unavailable"
HEADLINE = "integrity check unavailable (BenchFlow)"
UID_UNREAD = "the verifier user's uid could not be read"


class ModalSandbox:
    """A sandbox with no bfx route: walker_unavailable's first answer."""


def _refused_before_the_turn() -> str:
    """The pre-turn refusal as the rollout records it (``str(e)``)."""
    why = _verifier_links.walker_unavailable(ModalSandbox())
    assert why is not None and why.startswith(_verifier_links.BFX_UNAVAILABLE)
    error = _grading_outcome.IntegrityUnavailableError(
        scoring.INTEGRITY_UNAVAILABLE_HEADER + why
    )
    return str(error)


def _provenance(**overrides):
    ran = {"ran": True, "why": None, "links": []}
    args = {
        "agent": "claude-agent-acp",
        "sandbox_user": "agent",
        "record": None,
        "outputs_closed": True,
        "outputs_open_why": None,
        "links_before": ran,
        "links_after": ran,
    }
    args.update(overrides)
    return _reward_provenance.verdict(**args)


def _ending(provenance, *, sandbox_user: str | None = "agent"):
    return _grading_outcome.final_outcome(
        _grading_outcome.Attempt(
            rewards={"reward": 1.0},
            verifier_error=None,
            diagnostic=None,
            test_sh_reward=True,
            provenance=provenance,
            agent="claude-agent-acp",
            sandbox_user=sandbox_user,
        )
    )


def _no_record_after_grading() -> str:
    """The post-grading gap as final_outcome writes it."""
    gap = {"ran": False, "why": _verifier_links.NO_RECORD, "links": []}
    rewards, error, _ = _ending(_provenance(links_before=gap, links_after=gap))
    assert rewards is None and error, (rewards, error)
    return error


@pytest.mark.parametrize("prefixed", [False, True])
@pytest.mark.parametrize("stored", [False, True])
def test_a_refusal_before_the_turn_is_benchflow_s_integrity_cause(prefixed, stored):
    """Not "other (infrastructure)": its own key, BenchFlow's."""
    error = _refused_before_the_turn()
    if prefixed:
        error = f"IntegrityUnavailableError: {error}"
    record = {"task_name": "t", "error": error, "rewards": None}
    if stored:
        record["error_category"] = scoring.classify_error(error)
        assert record["error_category"] == KEY
    cause = failures.cause_of(record)
    assert cause is not None
    assert (cause.key, cause.code) == (KEY, KEY), cause
    assert cause.headline() == HEADLINE
    assert cause.fault == "benchflow"


@pytest.mark.parametrize("stored", [False, True])
def test_the_gap_after_grading_is_the_same_cause(stored):
    """Not "verifier_other (task problem)": BenchFlow's gap, never the task's."""
    error = _no_record_after_grading()
    record = {"task_name": "t", "verifier_error": error, "rewards": None}
    if stored:
        record["verifier_error_category"] = scoring.classify_verifier_error(error)
        assert record["verifier_error_category"] == KEY
    cause = failures.cause_of(record)
    assert cause is not None
    assert (cause.key, cause.code) == (KEY, KEY), cause
    assert cause.headline() == HEADLINE
    assert cause.fault == "benchflow"


def test_the_key_is_listed_once():
    assert failures.cause_keys().count(KEY) == 1
    described = failures.describe(KEY)
    assert described is not None and described.headline() == HEADLINE


def _trial(job: Path, name: str, **fields) -> RolloutResult:
    folder = job / f"{name}__abcd1234"
    folder.mkdir(parents=True)
    payload = {"task_name": name, "rollout_name": folder.name, **fields}
    (folder / "result.json").write_text(json.dumps(payload))
    return RolloutResult.from_dict(payload, rollout_dir=folder)


def test_the_run_summary_names_the_cause_and_its_page(tmp_path, monkeypatch):
    """The rc4 tag run's line, both forms: one cause, one explain id."""
    job = tmp_path / "jobs" / "run"
    job.mkdir(parents=True)
    results = {
        "before": _trial(
            job,
            "before",
            error=_refused_before_the_turn(),
            error_category=KEY,
        ),
        "after": _trial(
            job,
            "after",
            verifier_error=_no_record_after_grading(),
            verifier_error_category=KEY,
        ),
    }
    result = EvaluationResult(
        job_name="run",
        config=EvaluationConfig(),
        total=2,
        passed=0,
        errored=1,
        verifier_errored=1,
        elapsed_sec=1.0,
        job_dir=job,
        results=results,
    )
    rec = Console(file=io.StringIO(), width=240)
    monkeypatch.setattr(shared, "console", rec)
    shared._report_eval_result(result, job)
    out = rec.file.getvalue()
    assert f"1 errored: {HEADLINE}: before" in out, out
    assert f"1 unscored: {HEADLINE}: after" in out, out
    assert f"`bench explain {KEY}` says more" in out, out
    assert "other error" not in out and "the verifier failed" not in out, out


def test_a_scored_provenance_zero_has_no_cause():
    """Guard: the rule's 0 is a score; cause_of never changes a bucket."""
    rewards, error, _ = _ending(
        _provenance(outputs_closed=False, outputs_open_why=UID_UNREAD)
    )
    assert (rewards, error) == ({"reward": 0.0}, None)
    row = {
        "task_name": "t",
        "rewards": rewards,
        "warnings": [scoring.PROVENANCE_INCOMPLETE_ZERO + "x"],
        "verifier_error_category": scoring.PROVENANCE_INCOMPLETE,
    }
    assert scoring.classify_score_outcome(row) == "failed"
    assert failures.cause_of(row) is None


def test_provenance_incomplete_is_described_as_a_scored_zero():
    key = scoring.PROVENANCE_INCOMPLETE
    assert key in failures.scored_zero_keys()
    assert key not in failures.cause_keys()
    cause = failures.describe(key)
    assert cause is not None and cause.key == key
    assert cause.label.startswith("scored 0")
    assert cause.next_step.strip()


@pytest.mark.parametrize("sandbox_user", ["agent", None])
def test_an_unread_verifier_uid_is_an_incomplete_close(sandbox_user):
    """Pins errors.md's fix for "the verifier user's uid could not be read"
    (verifier_core's outputs_open_why): an agent user's attempt scores 0
    (provenance_incomplete); a root agent's reward stands, unverifiable."""
    provenance = _provenance(
        sandbox_user=sandbox_user, outputs_closed=False, outputs_open_why=UID_UNREAD
    )
    assert provenance["verdict"] == _reward_provenance.UNVERIFIABLE
    rewards, error, _ = _ending(provenance, sandbox_user=sandbox_user)
    assert error is None
    if sandbox_user is None:
        assert rewards == {"reward": 1.0}
        assert "outcome" not in provenance
    else:
        assert rewards == {"reward": 0.0}
        assert provenance["outcome"] == scoring.PROVENANCE_INCOMPLETE
        assert UID_UNREAD in provenance["outcome_reason"]


def test_dt_reads_both_outcomes_in_a_job_summary(tmp_path):
    """tools/dt files an unplaced category as an error: both are placed."""
    from tools.dt import normalise

    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "error_categories": {KEY: 1},
                "verifier_error_categories": {
                    KEY: 1,
                    scoring.PROVENANCE_INCOMPLETE: 1,
                },
            }
        )
    )
    summary = normalise.summary_counts(tmp_path)
    assert sum(summary["families"].values()) == 3, summary
    table = normalise.load_categories()["python"]
    assert table["error_category"][KEY] == table["verifier_error_category"][KEY]


# -- the hold's refusal where BenchFlow's waiter (bfx) is not delivered -----------------------------------------------


@pytest.mark.parametrize("guarded", [{"main"}, {"main", "db"}])
def test_the_holds_refusal_without_bfx_is_the_same_cause(guarded):
    """A start whose services wait for their rules, where bfx is not delivered, is refused by the hold before the
    turn: the same outcome as an integrity refusal, never retried. Mutant: every cover refusal classed so."""
    from benchflow.evaluation import RetryConfig
    from benchflow.sandbox._inert_bootstrap import no_waiter_refusal
    from benchflow.sandbox.service_cover import ServiceCoverError

    error = f"{ServiceCoverError.__name__}: {no_waiter_refusal(guarded)}"
    assert scoring.classify_error(error) == KEY
    assert scoring.retry_category(error, scoring.classify_error(error)) == KEY
    assert not RetryConfig(max_retries=3).should_retry(error)


@pytest.mark.parametrize(
    "refusal",
    [
        # Another cause after the hold's own head.
        "sandbox startup refused: BenchFlow holds 'main' until its rules are in, and main stopped before its rule",
        # Covers that failed: the namespace unreadable, main not in its bootstrap.
        "sandbox startup refused: main's network namespace could not be told from the host's, so its cloud "
        "metadata rule could not go in",
        "sandbox startup refused: main was not running in the inert bootstrap, so its cloud metadata rule could "
        "not go in before its own processes",
    ],
)
def test_another_cover_refusal_keeps_its_class(refusal):
    """Only the no-bfx refusal moves: any other refusal of a cover keeps its class (a sandbox setup failure)."""
    error = f"ServiceCoverError: {refusal}"
    assert scoring.classify_error(error) == scoring.SANDBOX_SETUP
