"""judge-loop@1: the session loop, judge-tools@1's results, submit_review's checks, and citations."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest

from benchflow.taskmd import judging
from benchflow.taskmd._vendor import judgeprompt as jp

ASSIGNMENT = {
    "behaviors": [],
    "criteria": [
        {"id": "method", "text": "Uses MLE.", "outcome": "good", "evidence": ["file:/work/report.md", "trajectory:/judge/trajectory.jsonl"], "levels": {"0": "no", "2": "some", "4": "yes"}},
        {"id": "caveats", "text": "Names a caveat.", "outcome": "good", "evidence": ["file:/work/report.md"]},
    ],
}


def _submission(**overrides) -> dict:
    verdicts = [
        {"id": "method", "verdict": "level", "level": "4", "citations": [{"source": "file", "path": "/work/report.md", "lines": [1, 1], "quote": "maximum likelihood"}], "rationale": "MLE."},
        {"id": "caveats", "verdict": "pass", "citations": [{"source": "judge", "step": 1, "quote": "caveat"}], "rationale": "It names one."},
    ]
    return {"verdicts": verdicts, "tags": []} | overrides


def test_a_complete_submission_is_accepted() -> None:
    assert judging.validate_submission(_submission(), ASSIGNMENT) == []


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        (lambda s: s["verdicts"].pop(), "no verdict for criterion caveats"),
        (lambda s: s["verdicts"][0].update(level="3"), "takes verdict \"level\""),
        (lambda s: s["verdicts"][1].update(verdict="level", level="4"), "takes pass or fail"),
        (lambda s: s["verdicts"].append(dict(s["verdicts"][1])), "more than one verdict"),
        (lambda s: s["verdicts"][1].update(id="other"), "not a criterion"),
        (lambda s: s["verdicts"][1].update(extra=1), "keys the schema does not define"),
        (lambda s: s["verdicts"][1]["citations"].append({"source": "web"}), "a citation has a source"),
        (lambda s: s.update(notes="x"), "exactly verdicts and tags"),
    ],
)
def test_submissions_that_break_the_rules_are_rejected(change, problem: str) -> None:
    submission = _submission()
    change(submission)
    problems = judging.validate_submission(submission, ASSIGNMENT)
    assert any(problem in p for p in problems), problems


def test_a_score_range_takes_a_value_within_it() -> None:
    assignment = {"behaviors": [], "criteria": [{"id": "c", "text": "t", "outcome": "good", "evidence": [], "score": {"min": 0, "max": 10}}]}
    ok = {"verdicts": [{"id": "c", "verdict": "value", "value": 7.5, "citations": [], "rationale": "r"}], "tags": []}
    assert judging.validate_submission(ok, assignment) == []
    ok["verdicts"][0]["value"] = 11
    assert judging.validate_submission(ok, assignment)


@pytest.fixture
def files(tmp_path) -> judging.JudgeFiles:
    root = tmp_path / "fs"
    (root / "work" / "sub").mkdir(parents=True)
    (root / "work" / "report.md").write_text("We used maximum likelihood.\nCaveat: x_min.\n")
    (root / "work" / "fit.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
    (root / "work" / "blob.bin").write_bytes(b"\xff\xfe\x00")
    (root / "judge").mkdir()
    (root / "judge" / "instruction.md").write_text("Fit it.\n")
    (root / "judge" / "trajectory.jsonl").write_text('{"id":1}\n')
    (root / "judge" / "tests.json").write_text('{"summary":{},"tests":[]}\n')
    return judging.JudgeFiles(root=root, kept=["/work"])


def test_read_serves_the_kept_copy_and_the_judge_files(files) -> None:
    text, blocks, record = judging.read_result(files, ASSIGNMENT, 1, "f00d", {"path": "/work/report.md"})
    assert text == "[judge step 1] read /work/report.md: lines 1-2 of 2\n<<<DATA f00d\nWe used maximum likelihood.\nCaveat: x_min.\nEND DATA f00d>>>\n"
    assert blocks == [] and record["path"] == "/work/report.md"
    text, _, _ = judging.read_result(files, ASSIGNMENT, 2, "f00d", {"path": "/work/report.md", "lines": [2, 9]})
    assert text.startswith("[judge step 2] read /work/report.md: lines 2-2 of 2\n") and "Caveat" in text and "maximum" not in text
    text, _, _ = judging.read_result(files, ASSIGNMENT, 3, "f00d", {"path": "/work"})
    assert text == "[judge step 3] read /work: 4 entries\n<<<DATA f00d\nfile blob.bin 3\nfile fit.png 12\nfile report.md 43\ndir sub/\nEND DATA f00d>>>\n"
    text, blocks, _ = judging.read_result(files, ASSIGNMENT, 4, "f00d", {"path": "/work/fit.png"})
    assert text.startswith("[judge step 4] read /work/fit.png: image/png, 12 bytes, sha256:") and blocks[0]["type"] == "image"
    text, _, _ = judging.read_result(files, ASSIGNMENT, 5, "f00d", {"path": "/work/blob.bin"})
    assert "binary file, 3 bytes" in text
    text, _, _ = judging.read_result(files, ASSIGNMENT, 6, "f00d", {"path": "/etc/passwd"})
    assert text == "[judge step 6] read: /etc/passwd is not served; use run"
    text, _, _ = judging.read_result(files, ASSIGNMENT, 7, "f00d", {"path": "/judge/trajectory.jsonl"})
    assert "lines 1-1 of 1" in text
    # tests.json is served only when an evidence item names the tests
    text, _, _ = judging.read_result(files, ASSIGNMENT, 8, "f00d", {"path": "/judge/tests.json"})
    assert "is not served" in text


async def test_run_caps_output_and_reports_timeouts() -> None:
    async def runner(command: str, timeout: int):
        if "sleep" in command:
            return 124, True, b"partial"
        return 0, False, b"x" * 20_000

    text, record = await judging.run_result(runner, 3, "beef", {"command": "yes x"})
    assert text.startswith("[judge step 3] run: exit 0\n<<<DATA beef\n")
    assert "[... 3616 bytes omitted ...]" in text and record["command"] == "yes x"
    text, _ = await judging.run_result(runner, 4, "beef", {"command": "sleep 999", "timeout": 5})
    assert text.startswith("[judge step 4] run: timed out after 5 s")
    text, _ = await judging.run_result(runner, 5, "beef", {"command": "ls", "timeout": 601})
    assert "from 1 to 600" in text


class _Script:
    """A fake Messages API that answers with a scripted list of content lists."""

    def __init__(self, responses: list[list[dict]], usage: int = 100) -> None:
        self.responses = list(responses)
        self.bodies: list[dict] = []
        self.usage = usage

    async def __call__(self, body: dict, timeout: float) -> dict:
        self.bodies.append(json.loads(json.dumps(body)))
        return {"content": self.responses.pop(0), "usage": {"input_tokens": self.usage, "output_tokens": 10}}


def _spec(tools=("read", "run", "submit_review"), tool_calls=None, tokens=None) -> judging.SessionSpec:
    return judging.SessionSpec(
        role="agent" if "run" in tools else "llm",
        unit="rubric",
        model="claude-haiku-4-5-20251001",
        assignment=ASSIGNMENT,
        prompt_masked=jp.prompt_text(None, ASSIGNMENT),
        brief=None,
        budget_text="1200 seconds",
        seconds=1200,
        tokens=tokens,
        tool_calls=tool_calls,
        tools=tools,
    )


def _use(name: str, args: dict, ident: str = "t1") -> dict:
    return {"type": "tool_use", "id": ident, "name": name, "input": args}


async def test_a_session_reminds_rejects_then_accepts(files) -> None:
    async def runner(command, timeout):
        return 0, False, b"one caveat\n"

    bad = _submission()
    bad["verdicts"].pop()
    script = _Script([
        [{"type": "text", "text": "Let me think."}],
        [_use("read", {"path": "/work/report.md"}, "a"), _use("run", {"command": "grep -i caveat /work/report.md"}, "b")],
        [_use("submit_review", bad, "c")],
        [_use("submit_review", _submission(), "d")],
    ])
    creds = judging.Credentials("api-key", "placeholder")
    result = await judging.run_session(_spec(), call=script, credentials=creds, files=files, evidence_text=None, runner=runner)
    assert result.end == "accepted" and result.accepted == _submission()
    first = script.bodies[0]
    assert first["system"] == jp.SYSTEM_PROMPT_1 and first["max_tokens"] == 32000 and first["tool_choice"] == {"type": "auto"}
    assert [t["name"] for t in first["tools"]] == ["read", "run", "submit_review"]
    assert first["messages"][0]["content"].startswith("## Judging\n") and f"<<<DATA {result.fence}" in first["messages"][0]["content"]
    assert script.bodies[1]["messages"][-1] == {"role": "user", "content": jp.REMINDER_1}
    rejected = script.bodies[3]["messages"][-1]["content"][0]["content"]
    assert rejected.startswith("Review not accepted:\n- no verdict for criterion caveats\n") and rejected.endswith("Fix these and submit again.")
    assert result.judge_steps[2]["command"] == "grep -i caveat /work/report.md"
    assert result.usage["tool_calls"] == 2 and result.usage["calls"] == 4


async def test_oauth_sends_claude_codes_identity_first(files) -> None:
    script = _Script([[_use("submit_review", _submission())]])
    creds = judging.Credentials("claude-code-oauth", "placeholder")
    assert creds.headers()["anthropic-beta"] == "oauth-2025-04-20" and creds.headers()["authorization"] == "Bearer placeholder"
    result = await judging.run_session(_spec(tools=("submit_review",)), call=script, credentials=creds, files=files, evidence_text=lambda f: "", runner=None)
    assert result.end == "accepted"
    assert script.bodies[0]["system"] == [
        {"type": "text", "text": judging.CLAUDE_CODE_IDENTITY},
        {"type": "text", "text": jp.SYSTEM_PROMPT_1},
    ]


async def test_three_quiet_responses_end_the_session(files) -> None:
    script = _Script([[{"type": "text", "text": "hm"}]] * 3)
    result = await judging.run_session(_spec(), call=script, credentials=judging.Credentials("api-key", "k"), files=files, evidence_text=None, runner=None)
    assert result.end == "no-tool-call" and result.accepted is None


async def test_three_rejections_end_the_session(files) -> None:
    script = _Script([[_use("submit_review", {"verdicts": [], "tags": []}, str(n))] for n in range(3)])
    result = await judging.run_session(_spec(), call=script, credentials=judging.Credentials("api-key", "k"), files=files, evidence_text=None, runner=None)
    assert result.end == "rejected"


async def test_budgets_bind(files) -> None:
    script = _Script([[_use("read", {"path": "/work/report.md"}, "a")], [_use("read", {"path": "/work/report.md"}, "b")], [_use("submit_review", _submission(), "c")]])
    result = await judging.run_session(_spec(tool_calls=1), call=script, credentials=judging.Credentials("api-key", "k"), files=files, evidence_text=None, runner=None)
    spent = script.bodies[2]["messages"][-1]["content"][0]["content"]
    assert spent.startswith("[judge step 2] read: the tool-call budget is spent; submit your review")
    assert "[runtime: 0 tool calls and" in spent
    assert result.end == "accepted"
    script = _Script([[_use("read", {"path": "/work/report.md"})]], usage=5_000)
    result = await judging.run_session(_spec(tokens=1_000), call=script, credentials=judging.Credentials("api-key", "k"), files=files, evidence_text=None, runner=None)
    assert result.end == "token-budget"


def test_citations_are_checked_against_what_the_judge_was_shown(files) -> None:
    record = {"steps": [{"id": 1, "source": "model", "text": "I fit by MLE", "tool_calls": [{"id": "c", "name": "bash", "arguments": json.dumps({"command": "python3 fit.py"}), "result": {"text": "exit 0\nalpha=2.5\n", "origin": "sandbox"}}]}]}
    ctx = judging.CitationContext(files=files, record=record, tests={"t1": {"id": "t1", "outcome": "passed", "duration_ms": 0}}, instruction=b"Fit it.\n", workdir="/work")
    steps = {1: {"tool": "run", "command": "grep -c . /work/report.md", "text": "2\n"}, 2: {"tool": "read", "path": "/work/report.md", "text": "We used maximum likelihood.\n"}}
    check = lambda c: judging.check_citation(c, ctx, steps)  # noqa: E731
    assert check({"source": "file", "path": "report.md", "lines": [1, 1], "quote": "maximum likelihood"}) == {"source": "file", "path": "/work/report.md", "lines": [1, 1], "quote": "maximum likelihood", "label": "solver", "verified": True, "match": "exact"}
    assert check({"source": "file", "path": "/work/report.md", "quote": "maximum  likelihood"})["match"] == "normalized"
    assert check({"source": "file", "path": "/work/report.md", "quote": "least squares"})["verified"] is False
    assert check({"source": "file", "path": "/work/fit.png"})["verified"] is None
    assert check({"source": "trajectory", "step": 1, "quote": "python3 fit.py"})["verified"] is True
    assert check({"source": "trajectory", "step": 9, "quote": "x"})["verified"] is False
    assert check({"source": "judge", "step": 1, "quote": "grep -c"}).items() >= {"label": "judge", "verified": True}.items()
    assert check({"source": "judge", "step": 1, "quote": "2"})["label"] == "solver-executed"
    assert check({"source": "judge", "step": 2, "quote": "maximum"})["label"] == "solver"
    assert check({"source": "tests", "path": "t1", "quote": '"outcome":"passed"'}).items() >= {"label": "environment", "verified": True}.items()
    assert check({"source": "instruction", "quote": "Fit it."})["label"] == "environment"


def test_combining_samples() -> None:
    assert judging.combine([Fraction(0), Fraction(1)], "median") == 0
    assert judging.combine([Fraction(4), Fraction(2), Fraction(4)], "median") == 4
    assert judging.combine([Fraction(2), Fraction(4)], "majority") == 2
    assert judging.combine([Fraction(2), Fraction(4), Fraction(4)], "majority") == 4
    assert judging.combine([Fraction(3), Fraction(1)], "min") == 1


def test_independent_citation_rule() -> None:
    criterion = {"id": "c", "cite": "independent"}
    solver_only = {"verdict": "pass", "citations": [{"verified": True, "label": "solver"}]}
    verdict, flags = judging.apply_citation_rules(criterion, solver_only)
    assert verdict["verdict"] == "fail" and flags == ["self-cited"]
    with_judge = {"verdict": "pass", "citations": [{"verified": True, "label": "judge"}]}
    assert judging.apply_citation_rules(criterion, with_judge)[0]["verdict"] == "pass"
    unverified = {"verdict": "fail", "citations": [{"verified": False, "label": "solver"}]}
    assert judging.apply_citation_rules({"id": "c"}, unverified)[1] == ["no-verified-citation"]


def test_credentials_and_model_overrides(monkeypatch) -> None:
    assert judging.credentials_from_env({}) is None
    assert judging.credentials_from_env({"ANTHROPIC_API_KEY": "a", "CLAUDE_CODE_OAUTH_TOKEN": "o"}).kind == "api-key"
    assert judging.credentials_from_env({"CLAUDE_CODE_OAUTH_TOKEN": "o"}).kind == "claude-code-oauth"
    assert "o" not in repr(judging.credentials_from_env({"CLAUDE_CODE_OAUTH_TOKEN": "o"}).token[:0])
    settings = {"model": "claude-opus-5-5"}
    assert judging.model_for("agent", settings, {}) == "claude-opus-5-5"
    assert judging.model_for("agent", settings, {judging.MODEL_ENV: "claude-haiku-4-5-20251001"}) == "claude-haiku-4-5-20251001"
    assert judging.model_for("agent", settings, {judging.MODEL_ENV: "llm=claude-x"}) == "claude-opus-5-5"
    assert judging.model_for("llm", {}, {judging.MODEL_ENV: "llm=claude-x,agent=claude-y"}) == "claude-x"


def test_the_session_limit(monkeypatch) -> None:
    limit = judging.SessionLimit()
    monkeypatch.setenv(judging.SESSION_LIMIT_ENV, "2")
    limit.acquire()
    limit.acquire()
    with pytest.raises(judging.JudgeError, match="spent"):
        limit.acquire()


def test_files_outside_the_kept_copy_are_not_served(tmp_path: Path) -> None:
    root = tmp_path / "fs"
    (root / "etc").mkdir(parents=True)
    (root / "etc" / "x").write_text("secret")
    files = judging.JudgeFiles(root=root, kept=["/work"])
    text, _, _ = judging.read_result(files, ASSIGNMENT, 1, "f", {"path": "/work/../etc/x"})
    assert "is not served" in text
