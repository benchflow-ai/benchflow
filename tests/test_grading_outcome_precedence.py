"""How an attempt's ending is decided, at the unit tier (F2b-DEVCYCLE.md
rows E, D, Y, B and Q7g; rulings Q1, Q2, Q6, Q7 of 02:38Z).

``final_outcome`` decides once (§2.4): an earlier ending is kept as it is,
with its error, diagnostic, category and retry decision; otherwise a reward
test.sh wrote is scored only when every provenance check completed and its
action took effect. Where one did not, a non-root agent's attempt scores 0
with the reason (``provenance_incomplete``, counted, never retried), a root
agent's is scored and ``unverifiable``, and the oracle's is not marked.
Every row asserts its precondition: the provenance record exists and says
which check did not complete.
"""

from __future__ import annotations

import ast
import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import benchflow
from benchflow._utils.scoring import classify_verifier_error
from benchflow._utils.trial_warnings import recording_trial_warnings
from benchflow.rollout import _reward_provenance, _setup
from benchflow.rollout._setup import _verify_rollout
from benchflow.rollout._verifier_recovery import verifier_error_reruns_trial
from benchflow.sandbox import _verifier_links
from benchflow.task import RolloutPaths
from benchflow.task import verifier_errors as errors

# The design's own fakes answer as bfx would; no stand-in bfx (tests/_fake_bfx.py).
pytestmark = pytest.mark.bfx_refusal

# -- the lead's rulings (02:38Z) these rows rest on ----------------------------
#: Q6: a non-root agent's check defeated during the run: reward 0, counted.
Q6_INCOMPLETE = "zero"
#: Q1: a non-root agent's link into the output folder after grading: the same 0.
Q1_AFTER_GRADING_NONROOT = "zero"
#: Q2: provenance_incomplete is its own category and never retried.
Q2_RETRIED = False
#: Q7: the verifies that score run final_outcome with their trial's agent;
#: these are out of scope, named, and never publish a score.
Q7_OUT_OF_SCOPE = {
    "task/equivalence.py": "internal tool: compares graders, publishes no score",
    "task/acceptance_live.py": "internal tool: task acceptance, publishes no score",
    "rollout/_separate_verifier.py": "strict's separate verifier, its own guarantees",
}

AGENT = "claude-agent-acp"
INCOMPLETE = "Verifier scored 0: provenance incomplete: "
NOT_COMPLETE = "the link check did not complete"
NOT_REMOVED = "a link into the verifier's output folder could not be removed"
APPEARED = "a link into the verifier's output folder appeared during grading"
PLANTED = {
    "path": "/app/grade.log",
    "target": "/logs/verifier/reward.txt",
    "resolved": "/logs/verifier/reward.txt",
}
SRC = Path(benchflow.__file__).parent


# -- the attempt's world: fakes for the sandbox, the verifier and the walks ----


def _task():
    return SimpleNamespace(
        name="precedence-task",
        task_dir=None,
        config=SimpleNamespace(
            verifier=SimpleNamespace(timeout_sec=0.2, reward_range=None, env={})
        ),
    )


def _walks(monkeypatch, *, ran=True, why=None, before=(), after=()):
    """The link walks' records: *ran* for both, *before* the links the walk
    before test.sh found (each with its ``removed``), *after* those found
    after grading. Returns the steps called, in order."""
    calls: list[str] = []

    def record(links, step):
        async def fake(env, *, timeout_sec=None):
            calls.append(step)
            return {
                "ran": ran,
                "why": None if ran else (why or "the walk was made to fail"),
                "engine": "bfx",
                "links": [dict(row) for row in links],
            }

        return fake

    monkeypatch.setattr(_verifier_links, "remove_turn_links", record(before, "before"))
    monkeypatch.setattr(_verifier_links, "links_after_grading", record(after, "after"))
    return calls


def _env(probe: str = "0\n", code: int = 0):
    return SimpleNamespace(
        exec=AsyncMock(
            return_value=SimpleNamespace(stdout=probe, stderr="", return_code=code)
        ),
        # The processes alive at grading were listed, and none was left.
        _benchflow_verifier_inputs={
            "flags": [],
            "changes": [],
            "processes_at_grading": [],
        },
    )


def _planes(verify, *, harden=None, receipt=None):
    # A real verifier reports its close of the output folder.
    verifier = SimpleNamespace(
        verify=verify, execution_receipt=receipt, outputs_closed=True
    )
    return SimpleNamespace(
        harden_before_verify=harden or AsyncMock(),
        quiesce_agent=AsyncMock(),
        verifier=lambda **_: verifier,
    )


def _returns(value):
    async def verify():
        return SimpleNamespace(rewards={"reward": value})

    return verify


def _raises(exc):
    async def verify():
        raise exc

    return verify


async def _wedge():
    await asyncio.sleep(3600)


async def _attempt(
    env, planes, tmp_path, name, *, agent=AGENT, user="agent", eligible=False
):
    """One verify, as the rollout runs it; the trial's warnings with it."""
    with recording_trial_warnings() as warnings:
        rewards, error, diagnostic = await _verify_rollout(
            env,
            _task(),
            RolloutPaths(rollout_dir=tmp_path / name),
            {},
            planes,
            sandbox_user=user,
            integrity="audit",
            recovery_eligible=eligible,
            agent=agent,
        )
    return SimpleNamespace(
        rewards=rewards,
        error=error,
        diagnostic=type(diagnostic).__name__,
        category=classify_verifier_error(error),
        reruns=verifier_error_reruns_trial(error),
        reruns_on_resume=verifier_error_reruns_trial(error, on_resume=True),
        warnings=list(warnings),
        provenance=env._benchflow_verifier_inputs.get("provenance"),
    )


def _assert_incomplete_zero(got, reason: str) -> None:
    """Q6 / Q1: reward 0, counted, with the reason; its own category; never
    retried."""
    from benchflow._utils import scoring

    assert Q6_INCOMPLETE == "zero" and Q1_AFTER_GRADING_NONROOT == "zero"
    assert (got.rewards, got.error) == ({"reward": 0.0}, None), got
    said = [w for w in got.warnings if w.startswith(INCOMPLETE)]
    assert said and reason in said[0], got.warnings
    assert got.provenance["outcome"] == "provenance_incomplete", got.provenance
    assert scoring.PROVENANCE_INCOMPLETE == "provenance_incomplete"
    row = {"rewards": got.rewards, "verifier_error": None, "warnings": got.warnings}
    assert scoring.provenance_incomplete(row) is True
    assert got.reruns is Q2_RETRIED and got.reruns_on_resume is Q2_RETRIED


# -- E: an earlier ending is kept as it is -------------------------------------


def _ending(kind, monkeypatch):
    """(env, planes, recovery_eligible) ending the attempt as *kind*: FXSTRICT's
    endings (F2b-FXSTRICT-ENDINGS.md §1), numbered as there."""
    env, harden, verify, eligible, receipt = _env(), None, _returns(1.0), False, None
    breach = errors.NetworkPolicyBreachedError(
        "the ipv6 rule is gone", record={"ipv6": "lost"}
    )
    if kind == "1 timeout, scored 0":
        verify, env = _wedge, _env(probe="4242\n")
    elif kind == "1c timeout with a breach":
        verify, env = _wedge, _env(probe="4242\n")

        async def timed_out_breach(_verifier):
            return breach

        monkeypatch.setattr(_setup, "_timed_out_breach", timed_out_breach)
    elif kind == "2 breach at the end of the turn":
        harden = AsyncMock(side_effect=breach)
    elif kind == "2 breach at the end of test.sh":
        verify = _raises(breach)
    elif kind == "3 network refused":
        verify = _raises(errors.VerifierNetworkRefusedError("pypi.org refused"))
    elif kind == "4 launch refused":
        verify = _raises(errors.VerifierLaunchRefusedError("no launch record"))
    elif kind == "5 tools refused":
        harden = AsyncMock(
            side_effect=errors.VerifierToolsRefusedError(
                "pytest only in /home/agent/bin"
            )
        )
    elif kind == "6 network not held":
        verify = _raises(errors.VerifierNetworkUnheldError("rules not proven"))
    elif kind == "7 guard markers unreadable":
        verify = _raises(errors.PluginGuardMarkersUnreadableError("markers cut short"))
    elif kind == "9 setup failure":
        harden = AsyncMock(
            side_effect=errors.VerifierSetupError("chmod of test.sh failed")
        )
    elif kind == "10 infrastructure crash":
        verify = _raises(errors.VerifierInfrastructureError("upload of tests/ failed"))
    elif kind == "11 no usable reward":
        verify = _raises(errors.RewardFileNotFoundError("verifier/reward.txt"))
    elif kind == "12 crash":
        verify = _raises(ConnectionResetError("connection reset"))
    elif kind == "13 start never confirmed":
        verify, env, receipt = _wedge, _env(code=1), ("/logs/verifier/s", "main")
    elif kind == "14 wedge left to recovery":
        verify, env, receipt = _wedge, _env(code=1), ("/logs/verifier/s", "main")
        eligible = True
    elif kind == "20 disk full":
        verify = _raises(OSError(28, "No space left on device"))
    else:
        raise AssertionError(kind)
    return env, _planes(verify, harden=harden, receipt=receipt), eligible


EARLIER_ENDINGS = (
    "1 timeout, scored 0",
    "1c timeout with a breach",
    "2 breach at the end of the turn",
    "2 breach at the end of test.sh",
    "3 network refused",
    "4 launch refused",
    "5 tools refused",
    "6 network not held",
    "7 guard markers unreadable",
    "9 setup failure",
    "10 infrastructure crash",
    "11 no usable reward",
    "12 crash",
    "13 start never confirmed",
    "14 wedge left to recovery",
    "20 disk full",
)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", EARLIER_ENDINGS)
@pytest.mark.parametrize(
    "defeat", ["walk incomplete", "link not removed", "verdict raised"]
)
async def test_an_earlier_ending_is_kept_as_it_is(kind, defeat, tmp_path, monkeypatch):
    """Row E (F18, F19). The same attempt with every check passing, and with
    one defeated: the ending, its error, diagnostic, category and retry
    decision are the same, and the provenance is recorded for both."""
    seen = []
    for defeated in (False, True):
        if defeated and defeat == "walk incomplete":
            _walks(monkeypatch, ran=False)
        elif defeated and defeat == "link not removed":
            _walks(
                monkeypatch, before=[{**PLANTED, "removed": False, "error": "EXDEV"}]
            )
        else:
            _walks(monkeypatch)
        if defeated and defeat == "verdict raised":
            monkeypatch.setattr(
                "benchflow.rollout._reward_provenance.verdict",
                lambda **_: (_ for _ in ()).throw(RuntimeError("a bug")),
            )
        env, planes, eligible = _ending(kind, monkeypatch)
        got = await _attempt(env, planes, tmp_path, f"{defeated}", eligible=eligible)
        # Precondition: the provenance code ran and left its record.
        assert got.provenance is not None, (kind, defeat)
        assert got.provenance.get("checks"), got.provenance
        seen.append(
            (
                got.rewards,
                got.error,
                got.diagnostic,
                got.category,
                got.reruns,
                got.reruns_on_resume,
            )
        )
    assert seen[0] == seen[1], (kind, defeat, seen)
    rewards, error = seen[1][0], seen[1][1]
    assert error is None or ("provenance" not in error and "link check" not in error)
    assert rewards is None or rewards == {"reward": 0.0}, (kind, rewards)


@pytest.mark.asyncio
@pytest.mark.parametrize("user", ["agent", None])
async def test_an_out_of_range_reward_test_sh_wrote_waits_on_the_checks(
    user, tmp_path, monkeypatch
):
    """Row E (§2.4 step 2): a reward test.sh wrote outside the task's range,
    kept as written with its diagnostic in audit, is still test.sh's reward:
    a defeated check gives the non-root 0, never a score."""
    _walks(monkeypatch)
    reference = await _attempt(
        _env(), _planes(_returns(5.0)), tmp_path, "ref", user=user
    )
    if reference.error is not None:
        pytest.skip(
            f"audit refuses an out-of-range reward on this line: {reference.error}"
        )
    assert reference.rewards == {"reward": 5.0}  # precondition: scored as written
    _walks(monkeypatch, ran=False)
    got = await _attempt(_env(), _planes(_returns(5.0)), tmp_path, "def", user=user)
    if user is None:
        assert got.rewards == {"reward": 5.0}
        assert got.provenance["verdict"] == "unverifiable"
    else:
        _assert_incomplete_zero(got, NOT_COMPLETE)


@pytest.mark.asyncio
async def test_retry_and_recovery_run_the_checks_again(tmp_path, monkeypatch):
    """Row E (§2.4): a crash keeps its ending; the next attempt on the same
    sandbox runs every check again and is scored only from its own."""
    env = _env()
    _walks(monkeypatch)
    first = await _attempt(
        env, _planes(_raises(ConnectionResetError("connection reset"))), tmp_path, "1"
    )
    assert first.category is not None and first.reruns is True  # precondition
    calls = _walks(monkeypatch, ran=False)
    got = await _attempt(env, _planes(_returns(1.0)), tmp_path, "2")
    assert calls == ["before", "after"]
    _assert_incomplete_zero(got, NOT_COMPLETE)


def test_nothing_sets_the_ending_after_final_outcome():
    """Row E (§2.4): final_outcome is the last word in _verify_hardened: no
    assignment to the error, the rewards or the diagnostic after its call,
    and the function returns what it decided."""
    tree = ast.parse((SRC / "rollout/_setup.py").read_text())
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "_verify_hardened"
    )
    calls = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and getattr(n.func, "id", getattr(n.func, "attr", None)) == "final_outcome"
    ]
    assert len(calls) == 1, "_verify_hardened must call final_outcome exactly once"
    after = calls[0].lineno
    late = [
        (t.id, n.lineno)
        for n in ast.walk(fn)
        if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign))
        and n.lineno > after
        for t in (n.targets if isinstance(n, ast.Assign) else [n.target])
        if isinstance(t, ast.Name)
        and t.id in {"rewards", "verifier_error", "verifier_diagnostic"}
    ]
    assert not late, f"set after final_outcome (line {after}): {late}"


# -- D: the class rule, by agent user and by the way the check failed ---------

INCOMPLETE_PATHS = {
    "bfx failed": "bfx scan failed: too-many: more than 4000000 entries",
    "timeout": "bfx scan failed: Command timed out after 180 seconds",
    "kernel resolution rc 127": "the kernel's resolution failed (rc 127)",
}
USERS = {
    "non-root": ("agent", AGENT),
    "root": (None, AGENT),
    "oracle": ("agent", "oracle"),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("path", list(INCOMPLETE_PATHS))
@pytest.mark.parametrize("who", list(USERS))
async def test_the_class_rule_for_a_walk_that_did_not_complete(
    who, path, tmp_path, monkeypatch
):
    """Row D: test.sh wrote 1.0 and no walk completed. Non-root: 0 with the
    reason (Q6); root: scored, unverifiable, the check named; oracle: scored,
    not marked."""
    user, agent = USERS[who]
    _walks(monkeypatch, ran=False, why=INCOMPLETE_PATHS[path])
    got = await _attempt(
        _env(), _planes(_returns(1.0)), tmp_path, "d", agent=agent, user=user
    )
    found = got.provenance
    assert found["links_checked"] is False  # precondition
    assert INCOMPLETE_PATHS[path] in " ".join(found["reasons"])
    if who == "non-root":
        _assert_incomplete_zero(got, NOT_COMPLETE)
        return
    assert (got.rewards, got.error) == ({"reward": 1.0}, None), got
    if who == "root":
        assert found["verdict"] == "unverifiable"
    assert not any(w.startswith(INCOMPLETE) for w in got.warnings)


@pytest.mark.asyncio
@pytest.mark.parametrize("who", list(USERS))
async def test_a_flagged_link_that_was_not_removed(who, tmp_path, monkeypatch):
    """Row D (F15, F25): the walk completed and flagged a link, and its
    removal failed (bfx EXDEV). Non-root: 0, "could not be removed"; root:
    unverifiable; oracle: not marked."""
    user, agent = USERS[who]
    _walks(
        monkeypatch,
        before=[
            {**PLANTED, "removed": False, "error": "EXDEV (another mount, not crossed)"}
        ],
        after=[PLANTED],
    )
    got = await _attempt(
        _env(), _planes(_returns(1.0)), tmp_path, "nr", agent=agent, user=user
    )
    assert got.provenance["links_checked"] is True  # precondition: walks completed
    assert got.provenance["links_removed_before_test"][0]["removed"] is False
    if who == "non-root":
        _assert_incomplete_zero(got, NOT_REMOVED)
    else:
        assert got.rewards == {"reward": 1.0}
        assert got.provenance["verdict"] == "unverifiable"


@pytest.mark.asyncio
@pytest.mark.parametrize("who", list(USERS))
async def test_a_link_that_appeared_during_grading(who, tmp_path, monkeypatch):
    """Row D (Q1, F22): no link before test.sh; one into the output folder
    after grading. Non-root: 0, "appeared during grading"; root: unverifiable;
    oracle: not marked."""
    user, agent = USERS[who]
    _walks(monkeypatch, after=[PLANTED])
    got = await _attempt(
        _env(), _planes(_returns(1.0)), tmp_path, "q1", agent=agent, user=user
    )
    assert got.provenance["links_after_grading"]  # precondition
    if who == "non-root":
        _assert_incomplete_zero(got, APPEARED)
    else:
        assert got.rewards == {"reward": 1.0}
        assert got.provenance["verdict"] == "unverifiable"


@pytest.mark.asyncio
@pytest.mark.parametrize("who", list(USERS))
async def test_a_fault_in_the_provenance_code_fails_closed(who, tmp_path, monkeypatch):
    """Row E/D (F19): verdict() raises. The check did not run: non-root 0,
    root unverifiable with the fault named, oracle not marked. Never skipped."""
    user, agent = USERS[who]
    _walks(monkeypatch)

    def broken(**_):
        raise RuntimeError("a bug in the verdict")

    monkeypatch.setattr("benchflow.rollout._reward_provenance.verdict", broken)
    got = await _attempt(
        _env(), _planes(_returns(1.0)), tmp_path, "f19", agent=agent, user=user
    )
    assert got.provenance is not None, "a fault must still leave a provenance record"
    assert "a bug in the verdict" in " ".join(got.provenance["reasons"])
    if who == "non-root":
        _assert_incomplete_zero(got, "a bug in the verdict")
    else:
        assert got.rewards == {"reward": 1.0}
        assert got.provenance["verdict"] == "unverifiable"


@pytest.mark.asyncio
async def test_all_checks_passing_scores_test_sh_verified(tmp_path, monkeypatch):
    """Row D, control: every check ran and passed; a flagged link was removed:
    test.sh's own value, verified, for every agent user."""
    for who, (user, agent) in USERS.items():
        _walks(monkeypatch, before=[{**PLANTED, "removed": True}])
        got = await _attempt(
            _env(), _planes(_returns(0.25)), tmp_path, who, agent=agent, user=user
        )
        assert (got.rewards, got.error) == ({"reward": 0.25}, None), (who, got)
        assert got.provenance["verdict"] == "verified", (who, got.provenance)


# -- Y: every verify that scores runs final_outcome with its trial's agent ----


def _verify_rollout_calls():
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "id", getattr(node.func, "attr", None))
                == "_verify_rollout"
            ):
                yield rel, node


def test_every_scoring_verify_passes_the_trials_agent():
    """Row Y (Q7, F33): every call of _verify_rollout outside the named
    out-of-scope tools passes ``agent=``, so the checks and final_outcome
    run: the rollout, recovery, the SDK's verify and regrade."""
    found = list(_verify_rollout_calls())
    assert len(found) >= 5, found  # precondition: the scan sees the call sites
    missing = sorted(
        f"{rel}:{node.lineno}"
        for rel, node in found
        if rel not in Q7_OUT_OF_SCOPE
        and not any(k.arg == "agent" for k in node.keywords)
    )
    assert not missing, f"verifies that score without the agent: {missing}"
    for rel in (
        "rollout/__init__.py",
        "rollout/_verifier_recovery.py",
        "sdk.py",
        "eval_regrade.py",
    ):
        assert any(r == rel for r, _ in found), f"no verify found in {rel}"


@pytest.mark.asyncio
async def test_the_sdk_verify_runs_final_outcome(tmp_path, monkeypatch):
    """Row Y: the SDK's verify, for a non-root agent with a defeated check,
    ends as the rollout's does: 0 with the reason."""
    from benchflow.sdk import SDK

    _walks(monkeypatch, ran=False)
    env = _env()
    monkeypatch.setattr(
        "benchflow.sdk.default_rollout_planes", lambda: _planes(_returns(1.0))
    )
    with recording_trial_warnings() as warnings:
        rewards, error, _ = await SDK._verify(
            SimpleNamespace(),
            env,
            _task(),
            RolloutPaths(rollout_dir=tmp_path / "sdk"),
            {},
            sandbox_user="agent",
            integrity="audit",
            agent=AGENT,
        )
    got = SimpleNamespace(
        rewards=rewards,
        error=error,
        warnings=list(warnings),
        provenance=env._benchflow_verifier_inputs.get("provenance"),
        reruns=verifier_error_reruns_trial(error),
        reruns_on_resume=verifier_error_reruns_trial(error, on_resume=True),
    )
    _assert_incomplete_zero(got, NOT_COMPLETE)


# -- Q7g: the out-of-scope verifies never publish a score ----------------------

PUBLISHERS = {
    "RolloutResult",
    "_build_result",
    "write_result",
    "result.json",
    "push_to_hub",
    "hub_push",
    "_hub_client",
    "publish_score",
}


@pytest.mark.parametrize("rel", ["task/equivalence.py", "task/acceptance_live.py"])
def test_an_out_of_scope_verify_never_publishes_a_score(rel):
    """Row Q7g: equivalence and acceptance_live run no final_outcome, so
    neither writes a result, pushes to the hub or feeds a leaderboard."""
    text = (SRC / rel).read_text()
    assert "_verify_rollout" in text  # precondition: it does verify
    tree = ast.parse(text)
    names = {
        getattr(n, "id", None) or getattr(n, "attr", None)
        for n in ast.walk(tree)
        if isinstance(n, (ast.Name, ast.Attribute))
    }
    strings = {
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }
    hits = sorted(p for p in PUBLISHERS if p in names or any(p in s for s in strings))
    assert not hits, f"{rel} reaches a score publisher: {hits}"


# -- B: every root read of a sandbox path goes through read_back --------------

#: Where the verifier's code runs as root against the sandbox.
VERIFIER_CODE = (
    "task/verifier_core.py",
    "rollout/_setup.py",
    "taskmd/verify.py",
    "task/verifier_outputs.py",
)
#: A command that reads a file's bytes or metadata back.
READS = ("cat ", "head ", "tail ", "base64 ", " od ", "readlink ", "stat ")
#: Read calls that are not commands.
READ_CALLS = {"download_file", "download_dir", "read_file", "copy_to"}


def _root_reads(rel):
    tree = ast.parse((SRC / rel).read_text())
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", getattr(node.func, "attr", None))
            text = " ".join(
                s.value
                for s in ast.walk(node)
                if isinstance(s, ast.Constant) and isinstance(s.value, str)
            )
            as_root = any(
                k.arg == "user" and getattr(k.value, "value", None) == "root"
                for k in node.keywords
            )
            if name in READ_CALLS or (as_root and any(r in text for r in READS)):
                yield fn.name, node.lineno, name


#: Root reads that are not read-backs of BenchFlow's own state, each named
#: with why: their bytes are the agent's by design, or come before grading.
READ_BACK_EXEMPT = {
    ("rollout/_setup.py", "_run_oracle"): "the oracle's own run, before grading",
    ("task/verifier_core.py", "_resolve_ors_episode_input"): (
        "the agent's episode, a judge's input by design"
    ),
    ("taskmd/verify.py", "_tar_kept_dir"): (
        "a judge's copy of the agent's own files, by design"
    ),
}


def _guarded(rel, fn_name, line):
    """Whether the read at *line* in *fn_name* follows a ``read_back`` (or
    the verifier's ``_read_back``) in the same function."""
    tree = ast.parse((SRC / rel).read_text())
    for fn in ast.walk(tree):
        if (
            not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
            or fn.name != fn_name
        ):
            continue
        if not (fn.lineno <= line <= (fn.end_lineno or fn.lineno)):
            continue
        for node in ast.walk(fn):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "id", getattr(node.func, "attr", None))
                in ("read_back", "_read_back")
                and node.lineno <= line
            ):
                return True
    return False


def test_every_root_read_of_a_sandbox_path_goes_through_read_back():
    """Row B (static registry, F2, F5-F12): during and after grading, the
    host or root reads a sandbox path only through ``read_back``, which
    refuses a path with a folder or file the agent could write. A read that
    bypasses it fails here, by file, function and line."""
    from benchflow.task import _read_back  # noqa: F401  (the primitive exists)

    reads = [(rel, *site) for rel in VERIFIER_CODE for site in _root_reads(rel)]
    assert reads, "precondition: the scan finds the verifier's root reads"
    bypass = sorted(
        f"{rel}:{line} {fn}() {call}"
        for rel, fn, line, call in reads
        if (rel, fn) not in READ_BACK_EXEMPT and not _guarded(rel, fn, line)
    )
    assert not bypass, "root reads that bypass read_back:\n" + "\n".join(bypass)


# -- S (FXSTRICT): an unset close -------------------------------------------------
def _verdict(**changes):
    found = {
        "agent": "bfprobe",
        "sandbox_user": "agent",
        "record": None,
        "outputs_closed": True,
        "outputs_open_why": None,
        "links_before": {"ran": True, "why": None, "links": []},
        "links_after": {"ran": True, "why": None, "links": []},
    }
    found.update(changes)
    return _reward_provenance.verdict(**found)


def test_s_an_unset_close_is_a_check_that_did_not_run():
    """S (F34). With kept services the close returns before it sets
    ``outputs_closed`` (the ``before_test`` path). An unset close is a check
    that did not run, never one left out: the verdict is `unverifiable`.
    Mutant: a ``None`` close omitted from ``checks[]``."""
    found = _verdict(outputs_closed=None)
    closed = [c for c in found["checks"] if c.get("check") == "outputs-closed"]
    assert closed and closed[0]["ran"] is False, found["checks"]
    assert found["verdict"] == _reward_provenance.UNVERIFIABLE, found


def test_s_a_close_never_due_is_no_check():
    """S (F2b-R3REV finding 1, 11-17). A verify with no test command
    (llm-judge, agent-judge, an ORS episode) never closes the folder: its
    reward is not test.sh's, so the close is no check of it, and clean walks
    give `verified`. Mutant: a close never due counted as one that did not
    run (every honest non-root judge trial scored 0)."""
    found = _verdict(outputs_closed=None, outputs_close_applies=False)
    assert [c for c in found["checks"] if c.get("check") == "outputs-closed"] == []
    assert found["verdict"] == _reward_provenance.VERIFIED, found


@pytest.mark.asyncio
async def test_s_a_close_is_due_once_asked_even_where_it_returns_unset():
    """S (F34 with finding 1). Kept services: the close returns before it
    sets ``outputs_closed``, but it was due, so the verdict still counts it as
    a check that did not run (F34). A verify that asked for none is not due.
    Mutant: due set only when the close sets a value."""
    from benchflow.task.verifier_core import Verifier

    verifier = Verifier.__new__(Verifier)
    assert verifier.outputs_close_due is False
    verifier.before_test = object()
    await verifier._close_output_dir("main", mounted=True)
    assert verifier.outputs_close_due is True
    assert verifier.outputs_closed is None


def test_s_a_closed_folder_with_clean_walks_is_verified():
    """S, control: the folder closed and both walks clean give `verified`."""
    found = _verdict()
    assert found["verdict"] == _reward_provenance.VERIFIED, found


# -- T (F39): every agent turn has its own record of the links before it -------


@pytest.mark.asyncio
async def test_t_a_walk_with_no_record_removes_nothing():
    """T (F39). A sandbox with no record from before the turn: no link can be
    told from the image's own (/dev/stdin, /etc/mtab lead into /proc), so the
    walks remove nothing and report that they did not run, with BenchFlow's
    reason. Mutant: no record = every link into the folder is the turn's."""
    env = SimpleNamespace(exec=AsyncMock(), bfx_call=AsyncMock())
    for check in (
        _verifier_links.remove_turn_links,
        _verifier_links.links_after_grading,
    ):
        got = await check(env, timeout_sec=5)
        assert got["ran"] is False and got["why"] == _verifier_links.NO_RECORD, got
        assert got["links"] == []
    env.exec.assert_not_called()
    env.bfx_call.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("who", list(USERS))
async def test_t_no_record_is_benchflow_s_gap_never_the_agent_s_zero(
    who, tmp_path, monkeypatch
):
    """T (F39, F2b 03:17Z). A verify whose sandbox had no record: a non-root
    agent's attempt is unscored with BenchFlow's reason
    (``integrity_unavailable``, never retried), never the counted 0; a root
    agent's stands, unverifiable; the oracle's is not marked."""
    from benchflow._utils import scoring

    user, agent = USERS[who]
    _walks(monkeypatch, ran=False, why=_verifier_links.NO_RECORD)
    got = await _attempt(
        _env(), _planes(_returns(1.0)), tmp_path, "t", agent=agent, user=user
    )
    assert not any(w.startswith(INCOMPLETE) for w in got.warnings), got.warnings
    if who == "non-root":
        assert got.rewards is None, got
        assert (
            got.error
            == scoring.INTEGRITY_UNAVAILABLE_VERIFIER + _verifier_links.NO_RECORD
        )
        assert got.category == scoring.INTEGRITY_UNAVAILABLE
        assert got.reruns is False and got.reruns_on_resume is False
    else:
        assert (got.rewards, got.error) == ({"reward": 1.0}, None), got


def test_t_the_record_survives_the_snapshot_with_odd_names(tmp_path):
    """T (F39, F40). The record before the turn is saved beside the trial and
    loaded by a child or a resume, names as they are (a non-UTF-8 byte, a
    newline); an entry the record cannot hold makes no record at all."""
    names = {"/app/x\udcffk": "/logs/verifier/reward.txt", "/app/a\nb": "/etc/x"}
    env = SimpleNamespace()
    _verifier_links.inherit_record(
        env, _verifier_links.Walked("bfx", links=dict(names))
    )
    _verifier_links.save_record(env, tmp_path)
    back = _verifier_links.load_record(tmp_path)
    assert back is not None and back.links == names
    broken = tmp_path / "broken"
    (broken / "integrity").mkdir(parents=True)
    (broken / "integrity" / _verifier_links.RECORD_FILE).write_text(
        '{"engine": "bfx", "links": {"/app/x": 7}}'
    )
    assert _verifier_links.load_record(broken) is None


@pytest.mark.asyncio
async def test_t_the_first_record_of_a_sandbox_is_kept():
    """T (F39). A record already there (a child's inherited one, or a fresh
    verifier sandbox's from before the frozen workspace) is never replaced by
    a walk that already sees the agent's links."""
    first = _verifier_links.Walked("bfx", links={"/etc/mtab": "/proc/mounts"})
    env = SimpleNamespace(exec=AsyncMock(), bfx_call=AsyncMock())
    _verifier_links.inherit_record(env, first)
    await _verifier_links.record_before_turn(env, timeout_sec=5)
    assert getattr(env, _verifier_links.INVENTORY) is first
    env.exec.assert_not_called()


def test_t_a_link_the_parent_made_is_the_child_s_turn_link():
    """T (F39, row T's forge at the unit tier). With the parent's record from
    before its turn, a link the parent made before the branch is a turn link
    of the child: flagged; the image's own system links are not."""
    parent = _verifier_links.Walked("bfx", links={"/dev/stdin": "/proc/self/fd/0"})
    now = _verifier_links.Walked(
        "bfx",
        links={
            "/dev/stdin": "/proc/self/fd/0",
            "/app/grade.log": "/logs/verifier/reward.txt",
        },
        kernel={"/app/grade.log": "/logs/verifier/reward.txt"},
    )
    rows = _verifier_links.turn_links(parent, now)
    assert [r["path"] for r in rows] == ["/app/grade.log"]


def _calls_in(rel: str, fn_name: str) -> list[tuple[int, str]]:
    tree = ast.parse((SRC / rel).read_text())
    found = []
    for fn in ast.walk(tree):
        if (
            isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
            and fn.name == fn_name
        ):
            for node in ast.walk(fn):
                if isinstance(node, ast.Call):
                    name = getattr(node.func, "id", getattr(node.func, "attr", None))
                    found.append((node.lineno, name))
    return sorted(found)


@pytest.mark.parametrize(
    ("rel", "fn", "records", "before"),
    [
        # A fresh rollout: the record, then it is saved beside the trial.
        (
            "rollout/__init__.py",
            "seed_then_runtime",
            "record_before_turn",
            "prepare_verifier_runtime",
        ),
        # Recovery: the fresh sandbox's record before the solver's workspace.
        (
            "rollout/_verifier_recovery.py",
            "_recover_verifier",
            "record_before_turn",
            "install_uploaded_workspace",
        ),
        # Regrade: the same, before the frozen workspace.
        (
            "eval_regrade.py",
            "run_verifier_on_frozen_trial",
            "record_before_turn",
            "_install_bundle",
        ),
    ],
)
def test_t_every_path_into_a_verify_with_a_turn_records_first(rel, fn, records, before):
    """T (F39, the static pin of F2b 03:17Z): every path into a verify with
    an agent's turn makes its record before the agent's (or the frozen
    workspace's) links arrive."""
    calls = _calls_in(rel, fn)
    made = [line for line, name in calls if name == records]
    later = [line for line, name in calls if name == before]
    assert made, f"{rel} {fn}() makes no record"
    assert later and min(made) < min(later), (rel, fn, made, later)


@pytest.mark.parametrize(
    ("rel", "fn"),
    [
        ("rollout_branch.py", None),
        ("branch_run.py", None),
        ("rollout/__init__.py", "verifier_baseline"),
    ],
)
def test_t_a_branch_child_and_a_resume_inherit_the_source_record(rel, fn):
    """T (F39, the static pin): a branch child (rollout_branch) and a resume
    from a checkpoint (branch_run) carry their source's record, and the
    child's baseline step applies it (inherited and re-captured alike)."""
    text = (SRC / rel).read_text()
    if fn is None:
        assert "_inherited_link_record" in text, rel
        return
    calls = [name for _, name in _calls_in(rel, fn)]
    assert calls.count("inherit_record") >= 2, (rel, fn, calls)


# -- H/L (F2b 04:00Z): a dangling link is judged by where it leads ------------


def _turn(links, kernel=None):
    return _verifier_links.turn_links(
        _verifier_links.Walked("bfx", links={}),
        _verifier_links.Walked("bfx", links=links, kernel=kernel),
    )


@pytest.mark.parametrize("kernel", ["none", "readlink-m"])
def test_dangling_links_are_judged_by_where_they_lead(kernel):
    """A dangling link to a workspace file not built yet, or to a host path
    the sandbox lacks, is left as the agent left it (no row); a dangling link
    to the reward (missing before test.sh) and a dangling chain whose
    existing part is a link into the output folder are rows "into the
    verifier's output folder". With the kernel's answer (readlink -m) or
    without one, the same."""
    links = {
        "/app/out": "/app/build/result.txt",
        "/logs/agent/host": "/home/someone/notes.txt",
        "/app/grade.log": "/logs/verifier/reward.txt",
        "/tmp/hop": "/logs/verifier",
        "/app/deep": "/tmp/hop/sub/reward.txt",
    }
    answers = {
        "none": {p: None for p in links},
        "readlink-m": {
            "/app/out": "/app/build/result.txt",
            "/logs/agent/host": "/home/someone/notes.txt",
            "/app/grade.log": "/logs/verifier/reward.txt",
            "/tmp/hop": "/logs/verifier",
            "/app/deep": "/logs/verifier/sub/reward.txt",
        },
    }[kernel]
    rows = {r["path"]: r for r in _turn(links, answers)}
    assert set(rows) == {"/app/grade.log", "/tmp/hop", "/app/deep"}, rows
    assert {r["class"] for r in rows.values()} == {_verifier_links.INTO_OUTPUTS}


def test_a_link_through_proc_is_its_own_class():
    rows = _turn(
        {"/app/x": "/proc/self/root/logs/verifier/reward.txt"}, {"/app/x": None}
    )
    assert [r["class"] for r in rows] == [_verifier_links.THROUGH_PROC], rows


@pytest.mark.asyncio
async def test_a_loop_makes_the_walk_incomplete_and_a_dangling_link_elsewhere_does_not(
    monkeypatch,
):
    """Only a link no resolution can follow (a loop) makes the check
    incomplete; a dangling link elsewhere is no failure and is not removed."""
    env = SimpleNamespace(
        bfx_call=AsyncMock(return_value={}), exec=AsyncMock(), bfx_delivery="bind"
    )
    _verifier_links.inherit_record(env, _verifier_links.Walked("bfx", links={}))

    async def walk(_env, *, roots, timeout_sec):
        return _verifier_links.Walked("bfx", links=dict(now_links))

    async def kernel(_env, paths, timeout_sec):
        return {p: None for p in paths}, None

    monkeypatch.setattr(_verifier_links, "walk", walk)
    monkeypatch.setattr(_verifier_links, "kernel_resolve", kernel)
    removed = []

    async def remove(_env, path, *_a, **_k):
        removed.append(path)
        return True, None

    monkeypatch.setattr(_verifier_links, "_remove", remove)
    now_links = {"/app/out": "/app/build/result.txt"}
    got = await _verifier_links.remove_turn_links(env, timeout_sec=5)
    assert got["ran"] is True and got["links"] == [] and removed == [], got
    now_links = {"/app/a": "/app/b", "/app/b": "/app/a"}
    got = await _verifier_links.remove_turn_links(env, timeout_sec=5)
    assert got["ran"] is False and _verifier_links.UNRESOLVED in got["why"], got
    assert removed == []


# -- Z (F36, Q4): bfx on Daytona native, through a fake exec stream -----------


def _daytona_fake(tmp_path, monkeypatch, *, machine="x86_64", sha_rc=0):
    """A DaytonaSandbox's bfx methods on a fake: exec answers by command,
    upload_file records what it put where."""
    import types

    from benchflow._bfx import locate
    from benchflow.sandbox.daytona import DaytonaSandbox

    binary = tmp_path / "bfx-pinned"
    binary.write_bytes(b"\x7fELF pinned")
    monkeypatch.setattr(locate, "verified_binary", lambda arch: binary)
    calls: list[str] = []
    uploads: list[tuple[str, str]] = []

    async def exec(command, user=None, timeout_sec=None, **_):
        calls.append(command)
        if command == "uname -m":
            return SimpleNamespace(stdout=machine + "\n", stderr="", return_code=0)
        if "sha256sum" in command and sha_rc:
            return SimpleNamespace(
                stdout="", stderr="bfx pin mismatch", return_code=sha_rc
            )
        if "bfx" in command and "--json" in command:
            return SimpleNamespace(
                stdout='{"ok": true, "result": {"entries": []}}\n', return_code=0
            )
        return SimpleNamespace(stdout="", stderr="", return_code=0)

    async def upload_file(source, target, *, mode=None, admission=None):
        uploads.append((str(source), target))

    fake = SimpleNamespace(
        exec=exec,
        upload_file=upload_file,
        logger=logging.getLogger("z"),
        bfx_delivery=None,
    )
    for name in ("plan_bfx_upload", "_bfx_binary", "bfx_call"):
        setattr(fake, name, types.MethodType(getattr(DaytonaSandbox, name), fake))
    return fake, calls, uploads


@pytest.mark.asyncio
async def test_z_daytona_native_plans_an_upload_where_a_pin_exists(
    tmp_path, monkeypatch
):
    """Z. Daytona native, x86_64 with a pinned bfx: delivery "per-call", and
    BenchFlow's walker can run there (no refusal before the turn); an arch
    without a pinned bfx: no delivery, refused before the turn."""
    fake, _, uploads = _daytona_fake(tmp_path, monkeypatch)
    await fake.plan_bfx_upload()
    assert fake.bfx_delivery == "per-call" and uploads == []
    assert _verifier_links.walker_unavailable(fake) is None
    other, _, _ = _daytona_fake(tmp_path, monkeypatch, machine="riscv64")
    await other.plan_bfx_upload()
    assert other.bfx_delivery is None
    assert _verifier_links.walker_unavailable(other) is not None


@pytest.mark.asyncio
async def test_z_bfx_is_uploaded_afresh_after_the_turn_and_checked_each_call(
    tmp_path, monkeypatch
):
    """Z. The pinned bfx goes into a root-only folder made for it before the
    turn (for the record), and into a fresh one once the turn is over; every
    call checks its SHA-256 against the pin first."""
    from benchflow._bfx import pins
    from benchflow.sandbox._bounded_exec import end_agent_turn

    fake, calls, uploads = _daytona_fake(tmp_path, monkeypatch)
    await fake.plan_bfx_upload()
    await fake.bfx_call("scan", {"paths": ["/"]}, timeout_sec=5)
    await fake.bfx_call("scan", {"paths": ["/"]}, timeout_sec=5)
    assert len(uploads) == 1
    before = uploads[0][1]
    end_agent_turn(fake)
    await fake.bfx_call("scan", {"paths": ["/"]}, timeout_sec=5)
    assert len(uploads) == 2 and uploads[1][1] != before
    made = [c for c in calls if c.startswith("umask 077 && mkdir -m 0700 ")]
    assert len(made) == 2
    runs = [c for c in calls if "--json" in c]
    assert len(runs) == 3
    pin = pins.SHA256[fake._bfx_arch]
    assert all("sha256sum " in r and pin in r for r in runs), runs


@pytest.mark.asyncio
async def test_z_a_bfx_that_does_not_match_its_pin_never_runs(tmp_path, monkeypatch):
    """Z. The uploaded file's hash differs from the pin: the call is refused
    (bfx unavailable), so the walk did not run, never one taken from it."""
    from benchflow._bfx.locate import BfxUnavailable

    fake, _, _ = _daytona_fake(tmp_path, monkeypatch, sha_rc=98)
    await fake.plan_bfx_upload()
    with pytest.raises(BfxUnavailable):
        await fake.bfx_call("scan", {"paths": ["/"]}, timeout_sec=5)
