"""Regressions for native provider provenance at frozen 7bc81e6ba.

Use the real HTTP client, session, rubric aggregation and saved RolloutResult.
Only the HTTP transport is fake. Criterion deadlines run the command wrapper.
"""

import asyncio
import json
import shutil
from collections import Counter
from unittest.mock import AsyncMock

import httpx
import pytest

from benchflow.task import Task
from benchflow.task.formats import materialize_task_dir
from benchflow.task.paths import RolloutPaths
from benchflow.task.verifier import Verifier
from benchflow.task.verifier_errors import (
    verifier_failure_reason,
    verifier_infrastructure_failure,
)
from benchflow.taskmd import judging
from benchflow.taskmd import verify as taskmd
from tests.test_r5_release_contract import observe
from tests.test_verifier_own_failures import _assess


def native(tmp_path, monkeypatch, prior, *, ids=("prints-42",), samples=1, minimum=1):
    from tests.test_taskmd_agent_judge import FIXTURE
    from tests.test_taskmd_verify import _Sandbox

    monkeypatch.setenv("BENCHFLOW_TASK_FORMAT_CACHE", str(tmp_path / "format-cache"))
    monkeypatch.delenv(judging.MODEL_ENV, raising=False)
    monkeypatch.delenv(judging.SESSION_LIMIT_ENV, raising=False)
    package = tmp_path / "package"
    shutil.copytree(FIXTURE, package)
    if prior:
        (package / "verifier/test.sh").write_text("#!/bin/sh\nexit 0\n")
    document = package / "task.md"
    document.write_text(
        document.read_text().replace(
            'timeout = "3m"',
            'timeout = "1s"\nper = "criterion"\n'
            f"samples = {samples}\nmin_samples = {minimum}",
        )
    )
    rubric_path = package / "verifier/rubric.json"
    rubric = json.loads(rubric_path.read_text())
    criterion = rubric["criteria"][0]
    rubric["criteria"] = [{**criterion, "id": ident} for ident in ids]
    rubric_path.write_text(json.dumps(rubric))
    root = tmp_path / "sandbox"
    (root / "app").mkdir(parents=True)
    (root / "app/main.py").write_text("print(42)\n")
    sandbox = _Sandbox(root)
    sandbox.is_mounted = True
    sandbox.upload_dir = AsyncMock()
    paths = RolloutPaths(tmp_path / "trial")
    paths.mkdir()
    verifier = Verifier(Task(materialize_task_dir(package)), paths, sandbox)
    credentials = judging.Credentials("api-key", "fixture-placeholder")
    monkeypatch.setattr(judging, "resolve_credentials", lambda: (credentials, None))
    monkeypatch.setattr(judging, "SESSIONS", judging.SessionLimit())
    return sandbox, verifier


def submission(ident="prints-42"):
    return {
        "verdicts": [
            {
                "id": ident,
                "verdict": "pass",
                "rationale": "fixture valid verdict",
                "citations": [
                    {"source": "file", "path": "/app/main.py", "quote": "print(42)"}
                ],
            }
        ],
        "tags": [],
    }


def response(ident="prints-42"):
    return {
        "content": [
            {
                "type": "tool_use",
                "id": "submit",
                "name": "submit_review",
                "input": submission(ident),
            }
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def transport(monkeypatch, handler):
    original = httpx.AsyncClient
    mock = httpx.MockTransport(handler)
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kwargs: original(transport=mock, **kwargs)
    )


BAD_RESPONSES = [
    "invalid-json",
    "list",
    "usage-list",
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "negative-usage",
    "bool-usage",
    "content-object",
    "block-scalar",
    "block-type",
    "tool-id",
    "tool-name",
    "tool-input",
    "text",
    "nonfinite",
    "unicode",
    "http-refused",
    "read-path-nul",
    "duplicate-tool-id",
    "huge-usage",
]


def bad_response(case):
    if case == "invalid-json":
        return httpx.Response(200, content=b"PRIVATE_PROVIDER_BYTES invalid JSON")
    if case == "http-refused":
        return httpx.Response(400, content=b"PRIVATE_PROVIDER_BYTES")
    value = response()
    if case == "list":
        value = ["PRIVATE_PROVIDER_BYTES"]
    elif case == "usage-list":
        value["usage"] = ["PRIVATE_PROVIDER_BYTES"]
    elif case.endswith("tokens"):
        value["usage"][case] = "PRIVATE_PROVIDER_BYTES"
    elif case == "negative-usage":
        value["usage"]["input_tokens"] = -1
    elif case == "bool-usage":
        value["usage"]["input_tokens"] = True
    elif case == "huge-usage":
        value["usage"]["input_tokens"] = 10**1000
    elif case == "content-object":
        value["content"] = {"PRIVATE_PROVIDER_BYTES": True}
    elif case == "block-scalar":
        value["content"] = ["PRIVATE_PROVIDER_BYTES"]
    elif case == "block-type":
        value["content"][0]["type"] = ["PRIVATE_PROVIDER_BYTES"]
    elif case.startswith("tool-"):
        value["content"][0][case.removeprefix("tool-")] = ["PRIVATE_PROVIDER_BYTES"]
    elif case == "text":
        value["content"].insert(0, {"type": "text", "text": ["PRIVATE_PROVIDER_BYTES"]})
    elif case == "nonfinite":
        value["content"][0]["input"]["verdicts"][0]["value"] = float("nan")
    elif case == "unicode":
        value["content"][0]["input"]["verdicts"][0]["rationale"] = "\ud800"
    elif case == "read-path-nul":
        value["content"][0].update(name="read", input={"path": "/app/\0bad"})
    elif case == "duplicate-tool-id":
        value["content"].append(value["content"][0])
    else:
        raise AssertionError(case)
    # Raw HTTP bytes also allow testing illegal JSON numbers and escaped lone
    # surrogates; httpx's outgoing JSON encoder correctly refuses those itself.
    return httpx.Response(200, content=json.dumps(value).encode("ascii"))


@pytest.mark.parametrize("prior", [False, True])
@pytest.mark.parametrize("case", BAD_RESPONSES)
async def test_native_http_response_is_a_sanitized_own_failure(
    tmp_path, monkeypatch, prior, case
):
    sandbox, verifier = native(tmp_path, monkeypatch, prior)
    called = []

    async def handle(request):
        assert request.url.path == "/v1/messages"
        called.append(bool(getattr(sandbox, "_benchflow_test_command_issued", False)))
        assert called[-1] is prior
        return bad_response(case)

    transport(monkeypatch, handle)
    rewards, error, warnings = await _assess(sandbox, verifier)
    assert called == [prior, prior], "both real session attempts must await HTTP"
    observe(tmp_path, rewards, error, "unscored", "provider")
    assert "PRIVATE_PROVIDER_BYTES" not in error
    assert not warnings
    receipts = [
        json.loads(p.read_text())
        for p in (tmp_path / "trial").rglob("agent-criterion-*.json")
    ]
    assert len(receipts) == 2
    assert all(r["failure_kind"] == "controller" for r in receipts)


@pytest.mark.parametrize("prior", [False, True])
@pytest.mark.parametrize("mode", ["valid", "retry", "repair-tag", "repair-citation"])
async def test_native_http_success_and_repairs_remain_scored(
    tmp_path, monkeypatch, prior, mode
):
    sandbox, verifier = native(tmp_path, monkeypatch, prior)
    calls = []

    async def handle(request):
        calls.append(json.loads(request.content))
        assert bool(getattr(sandbox, "_benchflow_test_command_issued", False)) is prior
        if len(calls) == 1 and mode == "retry":
            return bad_response("invalid-json")
        value = response()
        if len(calls) == 1 and mode == "repair-tag":
            value["content"][0]["input"]["tags"] = [
                {"id": [], "detected": False, "severity": 0, "spans": []}
            ]
        if len(calls) == 1 and mode == "repair-citation":
            value["content"][0]["input"]["verdicts"][0]["citations"][0]["path"] = (
                "/app/\0bad"
            )
        return httpx.Response(200, json=value)

    transport(monkeypatch, handle)
    rewards, error, warnings = await _assess(sandbox, verifier)
    assert len(calls) == (1 if mode == "valid" else 2)
    observe(tmp_path, rewards, error, 1)
    assert not warnings
    if mode.startswith("repair-"):
        assert len(calls[1]["messages"]) > len(calls[0]["messages"])


@pytest.mark.parametrize("prior", [False, True])
@pytest.mark.parametrize("provider_first", [False, True])
@pytest.mark.parametrize("mode", ["mixed", "deadlines", "retry", "minimum", "complete"])
async def test_all_outstanding_criteria_choose_the_failure_category(
    tmp_path, monkeypatch, prior, provider_first, mode
):
    ids = ["controller-criterion", "deadline-criterion"]
    if not provider_first:
        ids.reverse()
    sandbox, verifier = native(
        tmp_path,
        monkeypatch,
        prior,
        ids=ids,
        samples=2 if mode in ("minimum", "complete") else 1,
        minimum=1,
    )
    calls = Counter()
    commands = []
    original = sandbox.exec

    async def execute(command, **kwargs):
        if taskmd.COMMAND_FILE in command:
            commands.append(command)
            await asyncio.sleep(1.1)
        return await original(command, **kwargs)

    sandbox.exec = AsyncMock(side_effect=execute)

    async def handle(request):
        body = json.loads(request.content)
        prompt = body["messages"][0]["content"]
        ident = next(i for i in ids if i in prompt)
        calls[ident] += 1
        if mode == "complete" or (
            ident == "controller-criterion" and mode != "deadlines"
        ):
            failures = 1 if mode == "retry" else 2
            if calls[ident] <= failures:
                return bad_response("invalid-json")
            return httpx.Response(200, json=response(ident))
        return httpx.Response(
            200,
            json={
                "content": [
                    {
                        "type": "tool_use",
                        "id": "run",
                        "name": "run",
                        "input": {"command": "true", "timeout": 1},
                    }
                ],
                "usage": {},
            },
        )

    transport(monkeypatch, handle)
    rewards, error, warnings = await _assess(sandbox, verifier)
    if mode == "mixed":
        assert calls == {"controller-criterion": 2, "deadline-criterion": 2}
        assert len(commands) == 2
        observe(tmp_path, rewards, error, "unscored", "invalid response JSON")
        assert all(ident in error for ident in ids)
        assert not warnings
    elif mode == "complete":
        assert calls == {ident: 3 for ident in ids}
        assert not commands
        observe(tmp_path, rewards, error, 1)
        assert not warnings
    else:
        expected_calls = {
            "deadlines": {ident: 2 for ident in ids},
            "retry": {"controller-criterion": 2, "deadline-criterion": 2},
            "minimum": {"controller-criterion": 3, "deadline-criterion": 4},
        }[mode]
        assert calls == expected_calls
        assert len(commands) == (2 if mode == "retry" else 4)
        observe(tmp_path, rewards, error, 0)
        assert warnings


def test_provider_sanitization_stops_before_its_original_cause():
    cause = ValueError("PRIVATE_PROVIDER_BYTES")
    error = judging.JudgeError("invalid provider JSON")
    error.__cause__ = cause
    assert verifier_infrastructure_failure(error)
    assert "PRIVATE_PROVIDER_BYTES" not in verifier_failure_reason(error)
    assert not verifier_infrastructure_failure(cause)


def test_numeric_submission_overflow_is_a_repairable_verdict_error():
    args = submission()
    args["verdicts"][0].update(verdict="value", value=10**1000)
    assignment = {"criteria": [{"id": "prints-42", "score": {"min": 0, "max": 1}}]}
    assert judging.validate_submission(args, assignment)
