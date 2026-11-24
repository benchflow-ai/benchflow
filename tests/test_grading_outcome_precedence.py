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
            verifier=SimpleNamespace(timeout_sec=0.2, reward_range=None, env={}),
            # Every Task's config has an [agent] table; the oracle's provenance
            # reads its user (none here: solve.sh runs as root).
            agent=SimpleNamespace(user=None),
        ),
    )


def _walks(monkeypatch, *, ran=True, why=None, before=(), after=()):
    """The link walks' records: *ran* for both, *before* the links the walk
    before test.sh found (each with its ``removed``), *after* those found
    after grading. Returns the steps called, in order."""
    calls: list[str] = []

    def record(links, step):
        async def fake(env, *, timeout_sec=None, **_kwargs):
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

    async def none_alive(env, user, *, timeout_sec=None):
        # Rule S (sandbox/_agent_alive.py) in the walks' empty sandbox.
        return {
            "user": user,
            "uid": 1000,
            "alive": False,
            "by": "getpriority",
            "why": None,
        }

    from benchflow.sandbox import _agent_alive

    monkeypatch.setattr(_agent_alive, "ask", none_alive)
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

_TIMED_OUT_0 = "Verifier scored 0: verifier timed out after 0.2s"
_BREACH = "network policy breached: the ipv6 rule is gone"
# 813-U1: each ending as main has it, literally: (rewards, verifier_error, category, diagnostic
#: type, reruns, reruns on resume, the ending's own warning or None). Taken once from the unmutated tree
#: (14abd830) and read against F2b-FXSTRICT-ENDINGS.md §1. A checks-passing run that a mutant changes the same way
#: as its defeated run (M1, the gate unconditional) no longer hides here.
E_EXPECTED = {
    "1 timeout, scored 0": (
        {"reward": 0.0},
        None,
        None,
        "VerifierTimeoutDiagnostic",
        False,
        False,
        _TIMED_OUT_0,
    ),
    "1c timeout with a breach": (
        None,
        _BREACH,
        "network_policy_breached",
        "NoneType",
        False,
        False,
        None,
    ),
    "2 breach at the end of the turn": (
        None,
        _BREACH,
        "network_policy_breached",
        "NoneType",
        False,
        False,
        None,
    ),
    "2 breach at the end of test.sh": (
        None,
        _BREACH,
        "network_policy_breached",
        "NoneType",
        False,
        False,
        None,
    ),
    "4 launch refused": (
        None,
        "verifier launch refused: no launch record",
        "verifier_launch_refused",
        "NoneType",
        False,
        False,
        None,
    ),
    "5 tools refused": (
        None,
        "verifier tools refused: pytest only in /home/agent/bin",
        "verifier_tools_refused",
        "NoneType",
        False,
        False,
        None,
    ),
    "6 network not held": (
        None,
        "verifier network not held: rules not proven",
        "verifier_network_unheld",
        "NoneType",
        False,
        False,
        None,
    ),
    "7 guard markers unreadable": (
        {"reward": 0.0},
        None,
        None,
        "NoneType",
        False,
        False,
        "Verifier scored 0: pytest plugin guard not confirmed: markers cut short; test.sh's reward is not used. "
        "See verifier/test-stdout.txt",
    ),
    "9 setup failure": (
        None,
        "verifier crashed: VerifierSetupError: chmod of test.sh failed",
        "verifier_failure",
        "NoneType",
        False,
        False,
        None,
    ),
    "10 infrastructure crash": (
        None,
        "verifier crashed: VerifierInfrastructureError: upload of tests/ failed",
        "verifier_failure",
        "NoneType",
        False,
        False,
        None,
    ),
    "11 no usable reward": (
        None,
        "verifier wrote no usable reward: verifier/reward.txt",
        "verifier_failure",
        "NoneType",
        False,
        False,
        None,
    ),
    "12 crash": (
        None,
        "verifier crashed: ConnectionResetError: connection reset",
        "verifier_infra",
        "NoneType",
        True,
        True,
        None,
    ),
    "13 start never confirmed": (
        {"reward": 0.0},
        None,
        None,
        "VerifierTimeoutDiagnostic",
        False,
        False,
        _TIMED_OUT_0,
    ),
    "14 wedge left to recovery": (
        {"reward": 0.0},
        None,
        None,
        "NoneType",
        False,
        False,
        "Verifier scored 0: verifier start receipt is missing: the command may not have started, or test.sh's user "
        "may have removed the receipt",
    ),
    "20 disk full": (
        None,
        "verifier crashed: OSError: [Errno 28] No space left on device",
        "verifier_failure",
        "NoneType",
        False,
        False,
        None,
    ),
}


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
        # 813-U1: each run, the checks-passing one too, is the ending itself, literally: its own warning, never
        # the Q6 one, and no provenance outcome.
        rewards, error, category, diagnostic, reruns, on_resume, own = E_EXPECTED[kind]
        assert (got.rewards, got.error, got.category, got.diagnostic) == (
            rewards,
            error,
            category,
            diagnostic,
        ), (kind, defeat, defeated, got)
        assert (got.reruns, got.reruns_on_resume) == (reruns, on_resume), (kind, got)
        assert not any(w.startswith(INCOMPLETE) for w in got.warnings), got.warnings
        assert own is None or own in got.warnings, (own, got.warnings)
        assert got.provenance.get("outcome") is None, got.provenance
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("uids", "refused"),
    [
        (("2000", "1000"), False),
        (("1000", "1000"), True),
        ((None, "1000"), True),
        (("2000", None), True),
    ],
    ids=["own-user", "same-uid", "verifier-unreadable", "agent-unreadable"],
)
async def test_rb2_a_uid_that_cannot_be_read_refuses_before_the_turn(uids, refused):
    """REVIEW-FINAL F7. RB2 compares the verifier user's uid with the agent's, read before the turn. Where either
    can't be read, nothing shows they differ: the trial is refused before the turn with the reason (fail closed),
    never let through.

    Mutant: an unreadable uid let through (today's ``uid is not None`` guard)."""
    from benchflow.rollout import _grading_outcome

    refusal = getattr(_grading_outcome, "rb2_refusal", None)
    assert refusal is not None, "RB2's uid check has no fail-closed form"
    said = await refusal("grader", "agent", lambda user: _uid(uids, user))
    if refused:
        assert said is not None and ("uid" in said), said
    else:
        assert said is None, said


async def _uid(uids, user):
    return uids[0] if user == "grader" else uids[1]


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


# == 0.10.0-final, B1 (map-H1; map-H2 §1.2; DESIGN.md §2.1-2.8; rulings R-810, R-811, R-812, R-816, R1: yes) =========

# why the link checks did not run when the record from before the turn is incomplete.
INCOMPLETE_RECORD = "the record of the links before the turn is incomplete"
IU_HEADER = "integrity checks unavailable before the turn: "


def _listing_bfx(entries, calls):
    """A bfx whose walk lists *entries* (and one mount, /); *calls* records each command."""

    async def bfx_call(cmd, args=None, *, timeout_sec=None):
        args = dict(args or {})
        calls.append((cmd, args))
        if args.get("walks"):
            return {"entries": [dict(e) for e in entries], "errors": [], "mounts": []}
        if args.get("mounts"):
            return {
                "entries": [],
                "mounts": [{"id": 1, "mount_point": "/"}],
                "errors": [],
            }
        return {"entries": [], "errors": [], "mounts": []}

    return bfx_call


@pytest.mark.asyncio
async def test_t_an_incomplete_record_removes_nothing():
    """809-U1 (/ probe P4). The walk before the turn failed (a timeout), so its record holds no
    link: every image link (/dev/stdin, /dev/fd, /etc/mtab) would read as the turn's. Neither check acts on it:
    nothing is walked or removed, and each reports that it did not run, with BenchFlow's reason.

    Mutant: the record's ``complete`` not read (the image's /dev/stdin removed before test.sh)."""
    calls = []
    env = SimpleNamespace(
        exec=AsyncMock(),
        bfx_call=_listing_bfx(
            [
                {
                    "path": "/dev/stdin",
                    "type": "link",
                    "link": "/proc/self/fd/0",
                    "dev": 5,
                    "ino": 9,
                }
            ],
            calls,
        ),
        bfx_delivery="bind",
    )
    _verifier_links.inherit_record(
        env,
        _verifier_links.Walked(
            "bfx", complete=False, why="bfx scan failed: simulated: the walk timed out"
        ),
    )
    for check in (
        _verifier_links.remove_turn_links,
        _verifier_links.links_after_grading,
    ):
        got = await check(env, timeout_sec=5)
        assert got["ran"] is False, got
        assert str(got["why"]).startswith(INCOMPLETE_RECORD), got
        assert got["links"] == [], got
    assert calls == [], calls
    env.exec.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("who", list(USERS))
async def test_t_an_incomplete_record_is_benchflow_s_gap_never_the_agent_s_zero(
    who, tmp_path, monkeypatch
):
    """809-U2 (Q4/Q6). A verify whose record from before the turn is incomplete (the refusal before the
    turn was bypassed: an inherited record, a recovery, a regrade): a non-root agent's attempt is unscored with
    BenchFlow's reason (``integrity_unavailable``, never retried), never the counted 0; a root agent's is scored,
    unverifiable; the oracle's is not marked.

    Mutant: the reason missing from ``benchflow_gap`` (the agent's 0)."""
    from benchflow._utils import scoring

    user, agent = USERS[who]
    why = f"{INCOMPLETE_RECORD}: bfx scan failed: simulated"
    _walks(monkeypatch, ran=False, why=why)
    got = await _attempt(
        _env(), _planes(_returns(1.0)), tmp_path, "t809", agent=agent, user=user
    )
    assert not any(w.startswith(INCOMPLETE) for w in got.warnings), got.warnings
    if who == "non-root":
        assert got.rewards is None, got
        assert got.error == scoring.INTEGRITY_UNAVAILABLE_VERIFIER + why, got
        assert got.category == scoring.INTEGRITY_UNAVAILABLE
        assert got.reruns is False and got.reruns_on_resume is False
    else:
        assert (got.rewards, got.error) == ({"reward": 1.0}, None), got


def _branch_calls(fn_name: str) -> list[str]:
    """The names called directly in the ``if present["baseline"]`` branch of *fn_name* in rollout/__init__.py."""
    tree = ast.parse((SRC / "rollout/__init__.py").read_text())
    for fn in ast.walk(tree):
        if (
            isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
            and fn.name == fn_name
        ):
            for node in ast.walk(fn):
                if isinstance(node, ast.If) and "present" in ast.dump(node.test):
                    return [
                        getattr(c.func, "id", getattr(c.func, "attr", None))
                        for stmt in node.body
                        for c in ast.walk(stmt)
                        if isinstance(c, ast.Call)
                    ]
    return []


@pytest.mark.parametrize("record", ["fresh", "inherited", "bfx-unavailable"])
def test_t_an_incomplete_record_refuses_before_the_turn(record):
    """809-U3 (the generic guard). Known before the turn, BenchFlow's gap is a refusal
    before the turn (Q4): unscored, ``integrity_unavailable``, for every model agent, non-root or root; the oracle
    is exempt and runs. A fresh sandbox's record, an inherited one (a branch child, a resume), and a walk whose
    bfx could not run all refuse.

    Mutant: the refusal removed (the turn runs and ends unscored after it)."""
    from benchflow.rollout import _grading_outcome

    refusal = getattr(_grading_outcome, "pre_turn_refusal", None)
    assert refusal is not None, "no refusal before the turn for an incomplete record"
    why = (
        f"{_verifier_links.BFX_UNAVAILABLE} (not delivered)"
        if record == "bfx-unavailable"
        else "bfx scan failed: simulated"
    )
    walked = _verifier_links.Walked("bfx", complete=False, why=why)
    said = refusal(walked, primary_agent=AGENT)
    assert said is not None and said.startswith(IU_HEADER), said
    assert why in said, said
    if record != "bfx-unavailable":
        assert INCOMPLETE_RECORD in said, said
    assert refusal(walked, primary_agent="oracle") is None
    assert refusal(_verifier_links.Walked("bfx"), primary_agent=AGENT) is None
    if record == "inherited":
        assert "pre_turn_refusal" in _branch_calls("verifier_baseline"), (
            "the inherited record's path never asks"
        )
    else:
        calls = [
            name for _, name in _calls_in("rollout/__init__.py", "seed_then_runtime")
        ]
        made = (
            calls.index("record_before_turn") if "record_before_turn" in calls else -1
        )
        assert "pre_turn_refusal" in calls[made + 1 :], calls


@pytest.mark.asyncio
async def test_t_the_walk_before_the_turn_is_tried_once_more_on_a_failure_never_on_a_cap():
    """809-U4. No agent process exists yet, so a walk that failed (an exception: a timeout on a loaded
    host) is tried once more before the trial is refused; a walk over a cap (``too-many``, ``too-large``) would
    only fail the same way, and is not.

    Mutants: no retry; a retry on a cap."""
    from benchflow._bfx.client import BfxError

    for how in ("exception", "capped"):
        walks = []

        async def bfx_call(cmd, args=None, *, timeout_sec=None, how=how, walks=walks):
            args = dict(args or {})
            if args.get("walks"):
                walks.append(1)
                if how == "capped":
                    raise BfxError("too-many", "more than 4000000 entries")
                if len(walks) == 1:
                    raise RuntimeError("simulated: the walk before the turn timed out")
                return {"entries": [], "errors": [], "mounts": []}
            if args.get("mounts"):
                return {
                    "entries": [],
                    "mounts": [{"id": 1, "mount_point": "/"}],
                    "errors": [],
                }
            return {"entries": [], "errors": [], "mounts": []}

        env = SimpleNamespace(exec=AsyncMock(), bfx_call=bfx_call, bfx_delivery="bind")
        await _verifier_links.record_before_turn(env, timeout_sec=5)
        walked = getattr(env, _verifier_links.INVENTORY)
        if how == "exception":
            assert walked.complete is True and len(walks) == 2, (how, walked, walks)
        else:
            assert walked.complete is False and len(walks) == 1, (how, walked, walks)


@pytest.mark.asyncio
async def test_q4_a_first_bfx_call_that_fails_refuses_before_the_turn():
    """Z11 (folded into). Delivery was planned, but the first real bfx call fails (an upload
    that never landed, a noexec mount): the record before the turn says bfx cannot run, and the trial is refused
    before the turn with that reason, never a warning and a turn that ends unscored after it.

    Mutant: the guard removed."""
    from benchflow._bfx.locate import BfxUnavailable
    from benchflow.rollout import _grading_outcome

    async def unavailable(cmd, args=None, *, timeout_sec=None):
        raise BfxUnavailable("the per-call upload did not land")

    env = SimpleNamespace(
        exec=AsyncMock(), bfx_call=unavailable, bfx_delivery="per-call"
    )
    await _verifier_links.record_before_turn(env, timeout_sec=5)
    record = getattr(env, _verifier_links.INVENTORY)
    assert record.complete is False, record
    assert str(record.why).startswith(_verifier_links.BFX_UNAVAILABLE), record
    refusal = getattr(_grading_outcome, "pre_turn_refusal", None)
    assert refusal is not None, (
        "no refusal before the turn: the turn runs and ends unscored after it"
    )
    assert refusal(record, primary_agent=AGENT) == IU_HEADER + record.why


def _root_verdict(guarantee):
    try:
        return _verdict(
            sandbox_user=None,
            # The root agent's processes were listed, and none was alive.
            record={"processes_at_grading": []},
            walker={"engine": "bfx", "guarantee": guarantee, "why": None},
        )
    except TypeError as exc:
        pytest.fail(
            f"verdict() takes no walker record ({exc}): a shell-launched walker is never named"
        )


@pytest.mark.parametrize("guarantee", ["non-root-agent", "any-agent"])
def test_a_root_agent_on_a_shell_launched_walker_is_unverifiable(guarantee):
    """Z6 (map-H2 §1.2b, ruling R1: yes). Where bfx starts through the image's shell (Daytona native and DinD
    per-call, AgentCore), a root agent can replace that shell and own the walk: its reward is kept, marked
    ``unverifiable`` by a root row of its own, as live root processes are. Where the walker holds against any agent
    (an argv launch), the row passes.

    Mutant: the row removed (a root agent on Daytona reads `verified`)."""
    found = _root_verdict(guarantee)
    row = next(
        (c for c in found["checks"] if c.get("check") == "walker-holds-against-root"),
        None,
    )
    assert row is not None and row["ran"] is True, found["checks"]
    if guarantee == "any-agent":
        assert row["passed"] is True and found["verdict"] == "verified", found
    else:
        assert row["passed"] is False and found["verdict"] == "unverifiable", found
        assert any("image's shell" in r for r in found["reasons"]), found["reasons"]
    non_root = _verdict(
        walker={"engine": "bfx", "guarantee": "non-root-agent", "why": None}
    )
    assert not [
        c for c in non_root["checks"] if c["check"] == "walker-holds-against-root"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("delivery", "kit"), [("per-call", True), ("bind", False), (None, True)]
)
async def test_z_daytona_freeze_keeps_its_channels(tmp_path, delivery, kit):
    """Z10 (map-H2 §1.2f; R2-2). Which channels the freeze leaves running follows the channel actually in use:
    ``bfx serve`` only where bfx sits at /.bfx (``bfx_delivered``); per-call (Daytona native, DinD's default) and
    bfx off use the kit's helpers, so the step text carries the kit's and Daytona's patterns.

    Mutant: the predicate back to ``bfx_delivery is None`` (per-call: Daytona's helper stopped, the thaw hangs)."""
    from unittest.mock import MagicMock

    from benchflow.task.verifier_core import Verifier

    task = MagicMock()
    task.config.verifier.user = "root"
    sandbox = SimpleNamespace(bfx_delivery=delivery)
    verifier = Verifier(
        task=task, rollout_paths=RolloutPaths(tmp_path), sandbox=sandbox
    )
    verifier.freeze_at_exit = True
    step = await verifier._freeze_step("main")
    daytona = "/opt/runtime/rt/channel/python/bin/python3.12-I-S-B/opt/runtime/run/"
    assert (daytona in step) is kit, (delivery, step[:400])
    assert ("python3-I-S-B-c" in step) is kit, delivery


def test_s_a_close_never_due_leaves_a_row_that_does_not_count():
    """R2-3. A verify that runs no test command was never due a close: the record says so with a row
    (``applies: false``), which a reader can tell from a record made before the check existed, and which counts
    toward nothing.

    Mutant: the row counted (every honest non-root judge trial scores 0, as J's rows show)."""
    from benchflow.rollout._grading_outcome import incomplete_reason

    found = _verdict(outputs_closed=None, outputs_close_applies=False)
    row = next((c for c in found["checks"] if c.get("check") == "outputs-closed"), None)
    assert row is not None, found["checks"]
    assert row.get("applies") is False and row.get("ran") is False, row
    assert found["verdict"] == _reward_provenance.VERIFIED, found
    assert incomplete_reason(found) is None


_HELD = (
    "verifier crashed: VerifierOutputsHeldError: agent processes kept writing in /logs/verifier while it was "
    "being closed"
)


@pytest.mark.parametrize(
    "detail",
    [
        "rc=1: rm: cannot remove '/logs/verifier/connection reset': Directory not empty",
        "rc=1: chown: /logs/verifier/x/broken pipe: command timed out after 9s",
        "the clear before test.sh: temporarily unavailable",
    ],
)
@pytest.mark.parametrize("preserved", [False, True])
def test_a_held_ending_with_infra_words_in_its_detail_is_never_retried(
    detail, preserved
):
    """812-U1 (the retry hole). A held ending's detail carries sandbox stderr that names the
    agent's own files. Words in it that read as infrastructure (``connection reset``, ``broken pipe``, ``command
    timed out after``) must not make it VERIFIER_INFRA: then the trial is retried, agent and all, or recovered.
    The held head decides, before any phrase in its detail: the verifier's failure, never retried or recovered.

    Mutant: the anchored head removed (classification by phrase anywhere)."""
    from benchflow._utils.scoring import VERIFIER_FAILED, classify_verifier_error
    from benchflow.rollout._verifier_recovery import needs_verifier_recovery

    error = f"{_HELD} ({detail})"
    if preserved:
        error = "[solver-preserved] " + error
    assert classify_verifier_error(error) == VERIFIER_FAILED, error
    assert verifier_error_reruns_trial(error) is False
    assert verifier_error_reruns_trial(error, on_resume=True) is False
    assert needs_verifier_recovery(error) is False


# -- B, widened: every root read of a sandbox path, by any user that may be root, guarded on the same path ---------

_SYNTHETIC = {
    "variable-user": (
        "rollout/_synthetic.py",
        "async def f(env, who):\n"
        "    await exec_raw(env.exec, 'cat /logs/verifier/reward.txt', user=who)\n",
    ),
    "system-shell": (
        "rollout/_synthetic.py",
        "async def f(env):\n"
        "    await exec_raw(system_shell_exec, env, 'cat /logs/verifier/x.log')\n",
    ),
    "guard-on-another-path": (
        "task/_synthetic.py",
        "async def f(self):\n"
        "    await self._read_back('/logs/verifier/a', 'main')\n"
        "    await exec_raw(self._sandbox.exec, 'cat /logs/verifier/b', user='root')\n",
    ),
}


@pytest.mark.parametrize("case", list(_SYNTHETIC))
def test_the_registry_catches_synthetic_bypasses(case):
    """(the narrow read_back registry). The registry saw four files, only a literal ``user="root"``, and
    took any earlier ``read_back`` in the function as the guard. Each synthetic bypass is a root read it must
    catch: a variable user (root at run time), the system shell's exec (root), and a guard on another path.

    Mutants: the literal-root rule; any earlier guard accepted."""
    _rel, source = _SYNTHETIC[case]
    reads_in = globals().get("_root_reads_in")
    guarded_in = globals().get("_guarded_in")
    assert reads_in is not None and guarded_in is not None, (
        "the registry reads only named files"
    )
    tree = ast.parse(source)
    reads = list(reads_in(tree))
    assert reads, f"{case}: the read was not seen"
    assert not any(guarded_in(tree, fn, line) for fn, line, _call in reads), case


def test_the_registry_scans_every_module_that_runs_after_the_turn():
    """. Every module under task/, rollout/ and taskmd/ is scanned, not four named files.

    Mutant: the four-file list put back."""
    modules = globals().get("REGISTRY_MODULES")
    assert modules is not None, "the registry reads only VERIFIER_CODE"
    for rel in (
        "rollout/__init__.py",
        "rollout/_verifier_recovery.py",
        "taskmd/grading.py",
    ):
        assert rel in modules, rel


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

# Every module that may run during or after grading (the registry
#: read four named files): all of task/, rollout/ and taskmd/.
REGISTRY_MODULES = tuple(
    sorted(
        str(path.relative_to(SRC))
        for folder in ("task", "rollout", "taskmd")
        for path in (SRC / folder).rglob("*.py")
    )
)
#: A command that reads a file's bytes or metadata back.
READS = ("cat ", "head ", "tail ", "base64 ", " od ", "readlink ", "stat ")
#: Read calls that are not commands.
READ_CALLS = {"download_file", "download_dir", "read_file", "copy_to"}
#: Calls that run their command as root whatever user they are given.
ROOT_RUNNERS = {"system_shell_exec"}


def _may_be_root(node: ast.Call) -> bool:
    """A call that may run as root: a ``user`` that isn't a literal non-root
    name (a variable is root at run time when the task says so), or the
    system shell's exec, which is root's."""
    names = {getattr(a, "id", getattr(a, "attr", None)) for a in ast.walk(node)}
    if names & ROOT_RUNNERS:
        return True
    for k in node.keywords:
        if k.arg == "user":
            value = k.value
            return not (
                isinstance(value, ast.Constant) and value.value not in ("root", 0, "0")
            )
    return False


def _root_reads_in(tree: ast.AST):
    """Every root read of a sandbox path in *tree*: (function, line, call)."""
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
            if name in READ_CALLS or (
                _may_be_root(node) and any(r in text for r in READS)
            ):
                yield fn.name, node.lineno, name


def _root_reads(rel):
    yield from _root_reads_in(ast.parse((SRC / rel).read_text()))


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
    # Seen once the registry read every module, each named:
    ("rollout/__init__.py", "_soft_verify"): (
        "soft verify's log for the simulated user, never a score (the system shell, C8)"
    ),
    ("rollout/__init__.py", "_capture_setup_command_dir"): (
        "a task setup command's output folder, before the agent's turn"
    ),
    ("rollout/_collect_hooks.py", "collect_sidecar_artifacts"): (
        "the agent's own artifacts, kept as evidence; no score reads them"
    ),
    ("rollout/_recovery_submission.py", "capture_submission"): (
        "the agent's submission, by design: it is graded, never trusted"
    ),
    ("rollout/_separate_verifier.py", "_publish_verifier_outputs"): (
        "strict's separate verifier, whose sandbox runs no agent process (Q7)"
    ),
    ("rollout/_session_mirror.py", "sync"): (
        "the agent's own session log, mirrored for the trajectory"
    ),
    ("rollout/_user_loop.py", "_export_generated_skills"): (
        "the skills the agent wrote, exported as its output"
    ),
    ("rollout/_verifier_recovery.py", "_recover_verifier"): (
        "a host copy of the recovery's own trial folder, not a sandbox read"
    ),
    ("task/equivalence.py", "check_equivalence_async"): (
        "an internal tool that publishes no score (Q7)"
    ),
    ("task/verifier_core.py", "_recover_main_verifier_outputs"): (
        "each canonical name in the output folder, after read_back of that folder"
    ),
}


def _same_path(guard: ast.expr, read: ast.Call) -> bool:
    """Whether the read names the path the guard checked: the guard's path
    expression is part of the read, or its literal text is in the read's."""
    if isinstance(guard, ast.Constant) and isinstance(guard.value, str):
        return any(
            isinstance(s, ast.Constant)
            and isinstance(s.value, str)
            and guard.value in s.value
            for s in ast.walk(read)
        )
    want = ast.dump(guard)
    return any(ast.dump(n) == want for n in ast.walk(read) if n is not guard)


def _guarded_in(tree: ast.AST, fn_name: str, line: int) -> bool:
    """Whether the read at *line* in *fn_name* follows a ``read_back`` (or
    the verifier's ``_read_back``) of the same path in the same function."""
    for fn in ast.walk(tree):
        if (
            not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
            or fn.name != fn_name
            or not (fn.lineno <= line <= (fn.end_lineno or fn.lineno))
        ):
            continue
        reads = [
            n for n in ast.walk(fn) if isinstance(n, ast.Call) and n.lineno == line
        ]
        for node in ast.walk(fn):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "id", getattr(node.func, "attr", None))
                in ("read_back", "_read_back")
                and node.lineno <= line
                and node.args
                and any(_same_path(node.args[0], read) for read in reads)
            ):
                return True
    return False


def _guarded(rel, fn_name, line):
    return _guarded_in(ast.parse((SRC / rel).read_text()), fn_name, line)


def test_every_root_read_of_a_sandbox_path_goes_through_read_back():
    """Row B (static registry, F2, F5-F12): during and after grading, the
    host or root reads a sandbox path only through ``read_back``, which
    refuses a path with a folder or file the agent could write. A read that
    bypasses it fails here, by file, function and line."""
    from benchflow.task import _read_back  # noqa: F401  (the primitive exists)

    reads = [(rel, *site) for rel in REGISTRY_MODULES for site in _root_reads(rel)]
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
        # Rule S's answer (sandbox/_agent_alive.py): none of the agent's alive.
        "alive_at_test": {"user": "agent", "uid": 1000, "alive": False, "why": None},
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
    rows = [c for c in found["checks"] if c.get("check") == "outputs-closed"]
    assert all(r.get("applies") is False for r in rows), rows
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


@pytest.mark.asyncio
@pytest.mark.parametrize("who", list(USERS))
async def test_s_is_asked_after_the_walk_before_test_sh_and_its_answer_is_used(
    tmp_path, monkeypatch, who
):
    """S-2. Rule S is asked once in the verify itself, for the trial's non-root agent, after the walk before
    test.sh and before the verifier runs; its answer is the one the provenance uses (alive: the reward kept,
    `unverifiable`, with the record). The oracle and a root agent are not asked.

    Mutant: S never asked, taken as none alive (``_setup._ask_alive_at_test``)."""
    from benchflow.sandbox import _agent_alive

    user, agent = USERS[who]
    calls = _walks(monkeypatch)
    asked: list[str] = []

    async def alive(env, user, *, timeout_sec=None):
        calls.append("alive")
        asked.append(user)
        return {
            "user": user,
            "uid": 1000,
            "alive": True,
            "by": "getpriority",
            "why": None,
        }

    monkeypatch.setattr(_agent_alive, "ask", alive)

    async def verify():
        calls.append("verify")
        return SimpleNamespace(rewards={"reward": 1.0})

    got = await _attempt(
        _env(), _planes(verify), tmp_path, f"s-asked-{who}", agent=agent, user=user
    )
    assert (got.rewards, got.error) == ({"reward": 1.0}, None), (who, got)
    if who == "non-root":
        assert calls == ["before", "alive", "verify", "after"], calls
        assert asked == [user], asked
        found = got.provenance
        assert found["verdict"] == "unverifiable", found
        assert found["alive_at_test"]["alive"] is True, found
    else:
        assert "alive" not in calls and asked == [], (who, calls)


@pytest.mark.asyncio
async def test_a_refused_packet_in_test_sh_is_scored_as_test_sh_says_and_flagged(
    tmp_path, monkeypatch
):
    """The verifier's network refused test.sh something: no ending of its
    own (its packets can be test.sh's or the agent's code it ran), so the
    reward is test.sh's, and the trial's warnings carry the read flag."""
    _walks(monkeypatch)
    planes = _planes(_returns(0.0))
    planes.verifier().network_refusal = {
        "phase": "test.sh",
        "where": "in-container",
        "mode": "no-network",
        "refused": 3,
        "refused_words": "3 packets",
        "exit_code": 1,
        "passed": False,
    }
    got = await _attempt(_env(), planes, tmp_path, "refused")
    assert (got.rewards, got.error, got.category) == ({"reward": 0.0}, None, None)
    assert got.reruns is False
    assert any(
        "refused test.sh 3 packets" in w and "scored as test.sh says" in w
        for w in got.warnings
    ), got.warnings
