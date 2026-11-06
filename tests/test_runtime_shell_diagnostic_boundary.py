"""Guard diagnostic confidentiality and attribution regressed at 6d8b546f72cd."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow._diagnostics import render_text
from benchflow.rollout import write_error_traceback
from benchflow.sandbox import lockdown
from benchflow.sandbox._base import ExecResult
from benchflow.sandbox._exec_output import diagnostic_result, raw_output_requested
from benchflow.sandbox.protocol import ExecResult as PublicResult
from tests._receipt_fakes import receipt_answer
from tests.test_compose_raw_output import emit_process, transport

SYNTAX = "/bin/sh: eval: line 930: syntax error near unexpected token `}'"
SHELL = "/bin/sh -> /usr/bin/bash\nbash 5.1.8(1)-release\n"
BLAME = "syntax error in BenchFlow's setup script"


def protocol_env(answers):
    calls = []
    raw_results = []

    async def execute(command, **kwargs):
        assert raw_output_requested()
        calls.append(kwargs)
        answer, checked = answers[len(calls) - 1]
        if isinstance(answer, Exception):
            raise answer
        result = receipt_answer(command, answer) if checked else answer
        # The public value type copies only the three transport fields. The
        # receipt reader must establish provenance after this boundary.
        result = PublicResult(
            result.return_code, result.stdout or "", result.stderr or ""
        )
        raw_results.append(result)
        return result

    return SimpleNamespace(supports_runtime_kit=True, exec=execute), calls, raw_results


@pytest.mark.parametrize("backend", ["docker", "remote-docker"])
@pytest.mark.parametrize("route", ["container", "compose"])
@pytest.mark.parametrize("checked", [False, True], ids=["outer", "payload"])
@pytest.mark.parametrize("status", [1, 2, 127])
async def test_public_exec_distinguishes_parser_from_checked_child(
    tmp_path, monkeypatch, backend, route, checked, status
):
    """Guards 6d8b546f72cd through production exec result copying, not _read_exec mocks."""
    env = transport(tmp_path, backend, "fixture-login")
    env._main_container = "fixture-container" if route == "container" else None
    env.kit_cache_lookup = AsyncMock(return_value=None)
    calls = []

    def output(args):
        calls.append(args)
        if len(calls) == 1:
            # Even a copied receipt-error string is not receipt provenance.
            result = ExecResult(
                return_code=status,
                stdout="partial" if checked else "",
                stderr=SYNTAX + "\nexec output receipt not printed (exec status 2)",
            )
            if checked:
                result = receipt_answer(args[-1], result)
        else:
            result = receipt_answer(args[-1], ExecResult(return_code=0, stdout=SHELL))
        return result.stdout or "", result.stderr or "", result.return_code

    emit_process(monkeypatch, output)
    with pytest.raises(RuntimeError) as caught:
        await lockdown.prepare_runtime_tree(env, phase="build")
    parser = not checked and status in (1, 2)
    assert (BLAME in str(caught.value)) == parser
    assert len(calls) == (2 if parser else 1)
    assert ("image needs" in str(caught.value)) != parser
    assert "line 930" in str(caught.value)
    assert "will not run root steps" in str(caught.value)
    if checked:
        assert "stdout: partial" in str(caught.value)


@pytest.mark.parametrize("status", [1, 2, 127])
@pytest.mark.parametrize("checked", [False, True])
async def test_non_shell_failure_never_requests_a_probe(status, checked):
    """Guards 6d8b546f72cd: ordinary ownership/refusal handling stays intact."""
    env, calls, _ = protocol_env(
        [
            (
                ExecResult(
                    return_code=status, stderr="image has no root-owned /bin/sh"
                ),
                checked,
            )
        ]
    )
    with pytest.raises(RuntimeError, match="image needs a root-owned /bin/sh"):
        await lockdown.prepare_runtime_tree(env, phase="build")
    assert len(calls) == 1


@pytest.mark.parametrize("failure", ["exception", "failed", "empty"])
@pytest.mark.parametrize("budget", [4, 300])
async def test_probe_failure_preserves_original_error_and_timeout(failure, budget):
    """Guards 6d8b546f72cd: a best-effort diagnostic cannot replace the refusal."""
    probe = (
        OSError("synthetic unavailable diagnostic")
        if failure == "exception"
        else ExecResult(return_code=1 if failure == "failed" else 0, stdout="")
    )
    env, calls, _ = protocol_env(
        [(ExecResult(return_code=2, stderr=SYNTAX), False), (probe, True)]
    )
    with pytest.raises(RuntimeError) as caught:
        await lockdown.prepare_runtime_tree(env, phase="build", timeout_sec=budget)
    assert BLAME in str(caught.value)
    assert "line 930" in str(caught.value)
    assert "version unavailable" in str(caught.value)
    assert "will not run root steps" in str(caught.value)
    assert calls == [
        {"user": "root", "timeout_sec": budget},
        {"user": "root", "timeout_sec": min(budget, 10)},
    ]


@pytest.mark.parametrize("kind", ["literal", "multiline", "long", "padded"])
async def test_complete_probe_is_redacted_before_display_and_artifacts(
    monkeypatch, tmp_path, kind
):
    """Guards 6d8b546f72cd M1, including intact-probe and final-output controls."""
    from benchflow.cli import _errors

    secret = {
        "literal": "synthetic-r2-" + "a" * 24,
        "multiline": "synthetic-r2-first\nsynthetic-r2-second",
        "long": "synthetic-r2-" + "x" * 360,
        "padded": "  synthetic-r2-padded  ",
    }[kind]
    monkeypatch.setattr(_errors, "_secrets", lambda: [secret])
    probe_text = "/bin/sh -> /bin/bash\nbash " + secret + "\n"
    probe = ExecResult(return_code=0, stdout=probe_text)
    intact = diagnostic_result(probe).stdout.render()
    assert "synthetic-r2-" not in intact
    assert "***" in intact
    env, calls, raw = protocol_env(
        [(ExecResult(return_code=2, stderr=SYNTAX), False), (probe, True)]
    )
    with pytest.raises(RuntimeError) as caught:
        await lockdown.prepare_runtime_tree(env, phase="build")
    assert len(calls) == 2
    assert BLAME in str(caught.value)
    assert "***" in str(caught.value)
    assert "synthetic-r2-" not in str(caught.value)
    assert "synthetic-r2-" not in render_text(str(caught.value))
    artifact = write_error_traceback(tmp_path, caught.value)
    assert artifact is not None
    assert "synthetic-r2-" not in artifact.read_text()
    # Diagnostics must not modify receipt fields, statuses, or call counts.
    assert probe.stdout == probe_text
    assert raw[0].return_code == 2 and raw[1].return_code == 0


@pytest.mark.parametrize("kind", ["multiline", "long"])
async def test_later_outer_rendering_still_has_no_probe_fragments(monkeypatch, kind):
    """Guards 6d8b546f72cd M1 independently at the later outer-rendering sink."""
    from benchflow.cli import _errors

    secret = (
        "synthetic-r2-first\nsynthetic-r2-second"
        if kind == "multiline"
        else "synthetic-r2-" + "x" * 360
    )
    monkeypatch.setattr(_errors, "_secrets", lambda: [secret])
    env, calls, _ = protocol_env(
        [
            (ExecResult(return_code=2, stderr=SYNTAX), False),
            (ExecResult(return_code=0, stdout="/bin/sh\nbash " + secret + "\n"), True),
        ]
    )
    with pytest.raises(RuntimeError) as caught:
        await lockdown.prepare_runtime_tree(env, phase="build")
    assert len(calls) == 2
    assert BLAME in str(caught.value)
    assert "synthetic-r2-" not in render_text(str(caught.value))


async def test_sidecar_failure_does_not_probe_or_authorize_root_steps():
    """Guards 6d8b546f72cd: main-service diagnostics do not change sidecar refusal."""
    env, calls, _ = protocol_env([(ExecResult(return_code=1, stderr=SYNTAX), False)])
    saved = []
    env.set_runtime_kit = lambda *args: saved.append(args)
    assert not await lockdown.prepare_runtime_tree(env, phase="build", service="aux")
    assert calls == [{"user": "root", "timeout_sec": 300, "service": "aux"}]
    assert saved == [(None, "aux")]
