"""Guard FF-TEST-ENV-C7's child-side leak at e2d9343a before any line selection."""

from __future__ import annotations

import ast
import json
import shlex
import sys
import time
from pathlib import Path
from urllib.parse import quote, quote_plus

import pytest

from benchflow import doctor, doctor_smoke
from tests.test_cli_credential_diagnostics import (
    isolated_credentials as isolated_credentials,
)
from tests.test_doctor_smoke import _job


@pytest.mark.parametrize("sandbox", ["docker", "daytona"])
@pytest.mark.parametrize(
    "source",
    ["stdout", "trace", "message", "name", "verifier_error", "metric", "guard"],
)
@pytest.mark.parametrize(
    "separator",
    [
        "\n",
        "\r",
        "\r\n",
        "\v",
        "\f",
        "\x1c",
        "\x1d",
        "\x1e",
        "\x85",
        "\u2028",
        "\u2029",
    ],
)
def test_actual_child_failure_reporting(
    source, separator, sandbox, monkeypatch, tmp_path
):
    """e2d9343a C7: the real child report must protect evidence before projection."""
    secret = "synthetic-first" + separator + "synthetic-second"
    monkeypatch.setenv("DUMMY_TOKEN", secret)
    script = """
import json, os
from pathlib import Path
from types import SimpleNamespace
from benchflow.cli._shared import _report_eval_result
from benchflow.task.paths import RolloutPaths
source, job_dir = SOURCE, Path(JOB_DIR)
secret = os.environ['DUMMY_TOKEN']
paths = RolloutPaths(rollout_dir=job_dir / 'trial')
paths.verifier_dir.mkdir(parents=True)
error = None
if source in ('stdout', 'guard'):
    prefix = 'FAILED ' if source == 'stdout' else 'Verifier plugin trust rejected: '
    paths.test_stdout_path.write_text('x\\n' + prefix + secret + '\\ny\\n', newline='')
elif source == 'metric':
    paths.reward_json_path.write_text(json.dumps({'metrics': {secret: 0}}))
elif source == 'verifier_error':
    error = secret
else:
    test = {'name': 'test_example', 'status': 'failed'}
    test[source] = ('E AssertionError: ' if source == 'trace' else '') + secret
    (paths.verifier_dir / 'ctrf.json').write_text(json.dumps({'results': {'tests': [test]}}))
failure = SimpleNamespace(task_name='hello', verifier_error=error, rewards={'reward': 0}, rollout_name='trial')
result = SimpleNamespace(total=1, passed=0, score=0, task_failures=[failure])
_report_eval_result(result, job_dir)
""".replace("SOURCE", repr(source)).replace("JOB_DIR", repr(str(tmp_path / "evidence")))
    # Explicitly pin the child too: an editable install must not select another
    # archive when the fails-before overlay runs beside the final tree.
    monkeypatch.setenv("PYTHONPATH", str(Path(doctor.__file__).parents[1]))
    monkeypatch.setattr(
        doctor_smoke, "_bench_argv", lambda args: [sys.executable, "-c", script]
    )
    job = _job(tmp_path)
    job = doctor_smoke.SmokeJob(
        job.target, job.task_dir, job.jobs_dir, job.log_path, sandbox, job.timeout_sec
    )
    outcome = doctor_smoke.subprocess_runner(job)
    shown = job.log_path.read_text() + outcome.reason
    assert outcome.exit_code == 0, shown
    assert "synthetic-first" not in shown
    assert "synthetic-second" not in shown
    assert "***" in shown
    assert "Score: 0/1" in shown


@pytest.mark.parametrize("channel", ["stdout", "stderr"])
@pytest.mark.parametrize(
    "secret",
    [
        "synthetic-first\nsynthetic-second",
        "synthetic-first\r\nsynthetic-second",
        "synthetic-☃-credential",
        "synthetic-\udcff-credential",
    ],
)
def test_captured_child_output_is_protected(channel, secret, monkeypatch):
    """e2d9343a C7: captures cannot hand raw secrets to shortening consumers."""
    monkeypatch.setenv("DUMMY_TOKEN", secret)
    script = f"import os; value = {secret!r}.encode('utf-8', 'surrogateescape'); [os.write({1 if channel == 'stdout' else 2}, bytes([b])) for b in value]"
    result = doctor._run_command([sys.executable, "-c", script], 10)
    assert result.returncode == 0
    assert getattr(result, channel) == "***"
    assert getattr(result, "stderr" if channel == "stdout" else "stdout") == ""


VALUES = ["synthetic-first\nsynthetic-second", "  synthetic-☃'\"\\value\r\nnext  "]
SPELLINGS = [
    (value, spelling)
    for value in VALUES
    for spelling in sorted(
        {
            value,
            shlex.quote(value),
            json.dumps(value)[1:-1],
            json.dumps(value, ensure_ascii=False)[1:-1],
            repr(value)[1:-1],
            quote(value),
            quote(value, safe=""),
            quote_plus(value, safe=""),
        }
    )
]


@pytest.mark.parametrize("secret,spelling", SPELLINGS)
def test_every_spelling_at_every_chunk_offset(secret, spelling):
    """e2d9343a C7: raw, shell, repr, JSON and URL spellings cross every cut."""
    from benchflow._diagnostics import redact, redact_stream

    for offset in range(len(spelling) + 1):
        chunks = ["before " + spelling[:offset], spelling[offset:] + " after"]
        assert (
            "".join(redact_stream(chunks, [secret]))
            == redact("".join(chunks), [secret])
            == "before *** after"
        )
    assert "".join(redact_stream(iter(spelling), [secret])) == "***"


@pytest.mark.parametrize(
    "separator",
    [
        "\n",
        "\r\n",
        "\r",
        "\v",
        "\f",
        "\x1c",
        "\x1d",
        "\x1e",
        "\x85",
        "\u2028",
        "\u2029",
    ],
)
def test_every_offset_across_lines(separator):
    """e2d9343a C7: line-oriented sources never flush a partial credential."""
    from benchflow._diagnostics import redact_stream

    value = "synthetic-credential"
    for offset in range(1, len(value)):
        secret = value[:offset] + separator + value[offset:]
        text = "before " + secret + " after"
        assert (
            "".join(redact_stream(text.splitlines(keepends=True), [secret]))
            == "before *** after"
        )


@pytest.mark.parametrize(
    "text,secrets",
    [
        ("...https://bob:synthetic-password@proxy.example/a", []),
        ("-+https://alice:synthetic@extra@proxy.example/a", []),
        (
            "before https://bob:synthetic@proxy.example/a after",
            ["bob:synthetic@proxy.example"],
        ),
        ("aaaaaab aaaaaa aaaaaab", ["aaaaaa", "aaaaaab", "*** aaaaaa"]),
        ("short x abcde and untouched", ["x", "abcde"]),
        ("clean\r\n☃\udcff end", []),
    ],
)
def test_stream_parity_and_close(text, secrets):
    """e2d9343a C7: URL overlaps, invalid bytes and EOF use original ranges."""
    from benchflow._diagnostics import redact, redact_stream

    for offset in range(len(text) + 1):
        assert "".join(
            redact_stream([text[:offset], text[offset:]], secrets)
        ) == redact(text, secrets)
    assert "".join(redact_stream(iter(text), secrets)) == redact(text, secrets)


def test_stream_releases_safe_output_before_close():
    """e2d9343a C7: the boundary publishes complete lines, never raw prefixes."""
    from benchflow._diagnostics import redact_stream

    def chunks():
        yield "safe diagnostic " * 100 + "\nsynthetic-"
        pytest.fail("the first safe prefix should be available before the next read")

    output = next(redact_stream(chunks(), ["synthetic-credential"]))
    assert output.startswith("safe diagnostic ") and "synthetic-" not in output


@pytest.mark.parametrize("kind", ["words", "token", "url", "multiline"])
def test_one_megabyte_stream_timing(kind):
    """e2d9343a C7: linear work even for an unfinished megabyte URL/token."""
    from benchflow._diagnostics import redact, redact_stream

    secret = "synthetic-first\nsynthetic-second"
    text = {
        "words": "safe log " * 131072,
        "token": "x" * 1048576,
        "url": "https://" + "x" * 1048576 + "@host/path",
        "multiline": secret + "x" * 1048576,
    }[kind]
    text += "\n" + secret + "\nend"
    expected = redact(text, [secret])
    started = time.monotonic()
    actual = "".join(
        redact_stream((text[i : i + 4096] for i in range(0, len(text), 4096)), [secret])
    )
    elapsed = time.monotonic() - started
    print(f"stream {kind}: {len(text)} characters in {elapsed:.6f}s")
    assert actual == expected
    assert elapsed < 2.0


def test_tail_is_selected_after_redaction(monkeypatch, tmp_path):
    """e2d9343a C7: the retained tail may not start inside a raw credential."""
    from benchflow.cli._failure_evidence import (
        _ARTIFACT_READ_BYTES,
        _stdout_tail_failure_line,
    )

    secret = "synthetic-first\nFAILED synthetic-second" + "z" * _ARTIFACT_READ_BYTES
    monkeypatch.setenv("DUMMY_TOKEN", secret)
    path = tmp_path / "stdout.txt"
    path.write_text("FAILED safe-control\n" + secret + "\nend")
    evidence = _stdout_tail_failure_line(path)
    assert evidence is not None and evidence.body == "FAILED safe-control"


def test_child_output_sinks_cannot_bypass_stream_redaction():
    """e2d9343a C7: grep-style guard covers capture, file and evidence sinks."""
    root = Path(doctor.__file__).parent
    smoke = ast.parse((root / "doctor_smoke.py").read_text())
    runner = next(
        node
        for node in smoke.body
        if isinstance(node, ast.FunctionDef) and node.name == "subprocess_runner"
    )
    calls = [node for node in ast.walk(runner) if isinstance(node, ast.Call)]
    sinks = [
        call
        for call in calls
        if isinstance(call.func, ast.Attribute)
        and call.func.attr
        in {"write", "writelines", "write_text", "write_bytes", "print"}
    ]
    assert {ast.unparse(call.func) for call in sinks} == {
        "log.write",
        "output.writelines",
    }
    for call in sinks:
        if ast.unparse(call.func) == "log.write":
            assert ast.unparse(call.args[0]) == "'$ bench ' + shlex.join(args) + '\\n'"
        else:
            assert (
                isinstance(call.args[0], ast.Call)
                and ast.unparse(call.args[0].func) == "redact_stream"
            )
    popen = next(call for call in calls if ast.unparse(call.func) == "subprocess.Popen")
    assert {kw.arg: ast.unparse(kw.value) for kw in popen.keywords}["stdout"] == "log"
    source = (root / "doctor.py").read_text()
    assert (
        "CommandResult(proc.returncode, captured(proc.stdout), captured(proc.stderr))"
        in source
    )
    assert "redact_stream((value.decode(" in source
    evidence = (root / "cli/_failure_evidence.py").read_text()
    assert "redact_stream((text[i : i + 8192]" in evidence
    assert "raw.read(budget)" in evidence
    assert "_captured_text(trace).splitlines()" in evidence
    assert "_captured_text(message).split()" in evidence
    assert "_display_test_name(_captured_text(" in evidence
    shared = (root / "cli/_shared.py").read_text()
    assert "_captured_text(failure.verifier_error).split()" in shared
    assert "escape(_captured_text(headline))" in shared
    assert "escape(_captured_text(shown))" in shared
    causes = (root / "failures.py").read_text()
    assert 'render_diagnostic((str(data.get("error")' in causes


@pytest.mark.parametrize("offset", range(1, 20))
def test_bounded_tail_never_publishes_a_seek_fragment(offset, monkeypatch, tmp_path):
    """e2d9343a C7: complete and seek-bisected values share the same tail boundary."""
    from benchflow.cli._failure_evidence import (
        _ARTIFACT_READ_BYTES,
        _stdout_tail_failure_line,
    )

    secret = "synthetic-first\nFAILED synthetic-second"
    monkeypatch.setenv("DUMMY_TOKEN", secret)
    # Position the known credential across the old tail seek. A leading giant
    # URL token also checks that a truncated authority is discarded as a unit.
    path = tmp_path / "stdout.txt"
    padding = "z" * (_ARTIFACT_READ_BYTES - len(secret) + offset)
    path.write_text(
        "https://"
        + "x" * (2 * _ARTIFACT_READ_BYTES)
        + "@host/path\nFAILED "
        + secret
        + padding
        + "\nFAILED safe-control\n"
    )
    evidence = _stdout_tail_failure_line(path)
    assert evidence is not None and evidence.body == "FAILED safe-control"


def test_stdout_evidence_reads_a_bounded_window(monkeypatch):
    """e2d9343a C7: redaction must not turn tail mining into an unbounded read."""
    import io

    from benchflow.cli._failure_evidence import _stdout_tail_failure_line

    secret = "synthetic-first\nsynthetic-second"
    monkeypatch.setenv("DUMMY_TOKEN", secret)
    data = ("x" * (2 * 1024 * 1024) + "\nFAILED " + secret + "\n").encode()
    reads = []

    class Capture(io.BytesIO):
        def read(self, size=-1):
            assert 0 < size < 128 * 1024
            reads.append(size)
            return super().read(size)

    class Artifact:
        def open(self, *args, **kwargs):
            return Capture(data)

    evidence = _stdout_tail_failure_line(Artifact())
    assert evidence is not None and evidence.body == "FAILED ***"
    assert len(reads) == 1


@pytest.mark.parametrize("sandbox", ["docker", "daytona"])
@pytest.mark.parametrize("secret,spelling", SPELLINGS)
def test_smoke_log_read_boundaries(secret, spelling, sandbox, monkeypatch, tmp_path):
    """e2d9343a C7: every spelling/cut reaches the actual public smoke log."""
    from benchflow._diagnostics import redact

    monkeypatch.setenv("DUMMY_TOKEN", secret)
    job = _job(tmp_path)
    job = doctor_smoke.SmokeJob(
        job.target, job.task_dir, job.jobs_dir, job.log_path, sandbox, job.timeout_sec
    )
    header = "$ bench " + shlex.join(doctor_smoke.eval_command(job)) + "\n"
    text = ""
    for offset in range(1, len(spelling)):
        padding = (8192 - (len(header) + len(text)) % 8192 - offset) % 8192
        text += "x" * padding + spelling + " safe\n"
    # The fixture file only transports synthetic output; the child writes both
    # streams, as the real eval subprocess does, without an argv-size limit.
    payload = tmp_path / "child-output.txt"
    payload.write_text(text)
    script = f"import os; data = open({str(payload)!r}, 'rb').read(); midpoint = len(data)//2; os.write(1, data[:midpoint]); os.write(2, data[midpoint:])"
    monkeypatch.setattr(
        doctor_smoke, "_bench_argv", lambda args: [sys.executable, "-c", script]
    )
    outcome = doctor_smoke.subprocess_runner(job)
    assert outcome.exit_code == 0
    assert job.log_path.read_text() == redact(header + text, [secret])


@pytest.mark.parametrize("source", ["paired-metric", "node-id", "task-name"])
def test_projected_evidence_fields_keep_whole_credentials(
    source, monkeypatch, tmp_path
):
    """e2d9343a C7: suffix/path removal and task-name clipping are sibling cuts."""
    import io
    from types import SimpleNamespace

    from rich.console import Console

    from benchflow.cli import _shared
    from benchflow.cli._failure_evidence import _ctrf_failure_line, metric_breakdown

    if source == "paired-metric":
        secret = "synthetic-first_found"
        monkeypatch.setenv("DUMMY_TOKEN", secret)
        shown = metric_breakdown({secret: 0, "synthetic-first_total": 2})
        assert shown == "*** 0/2"
    elif source == "node-id":
        secret = "synthetic-first::synthetic-second"
        monkeypatch.setenv("DUMMY_TOKEN", secret)
        path = tmp_path / "ctrf.json"
        path.write_text(
            json.dumps({"results": {"tests": [{"name": secret, "status": "failed"}]}})
        )
        evidence = _ctrf_failure_line(path)
        assert evidence is not None and evidence.body == "*** failed"
    else:
        secret = "synthetic-first synthetic-second"
        monkeypatch.setenv("DUMMY_TOKEN", secret)
        output = io.StringIO()
        monkeypatch.setattr(
            _shared, "console", Console(file=output, width=160, color_system=None)
        )
        failure = SimpleNamespace(
            task_name="x" * 75 + " " + secret,
            verifier_error=None,
            rewards={"reward": 0},
            rollout_name="trial",
        )
        result = SimpleNamespace(total=1, passed=0, score=0, task_failures=[failure])
        _shared._report_eval_result(result)
        assert "synthetic-first" not in output.getvalue()
        assert "***" in output.getvalue()


@pytest.mark.parametrize("sandbox", ["docker", "daytona"])
@pytest.mark.parametrize("source", ["other-error", "outcome-name", "outcome-label"])
def test_actual_child_outcome_reporting(source, sandbox, monkeypatch, tmp_path):
    """e2d9343a C7: cause clipping and Rich wrapping are child-side projections."""
    secret = (
        "synthetic-first\nsynthetic-second"
        if source == "other-error"
        else "synthetic-first" + "X" * 100 + "synthetic-second"
    )
    monkeypatch.setenv("DUMMY_TOKEN", secret)
    script = """
import os
from types import SimpleNamespace
from rich.console import Console
from benchflow.cli import _shared
secret = os.environ['DUMMY_TOKEN']
source = SOURCE
_shared.console = Console(width=50, color_system=None)
name = secret if source == 'outcome-name' else 'hello'
record = {'error': secret, 'error_category': 'other'}
if source != 'other-error':
    record['error_category'] = 'provider_auth'
if source == 'outcome-label':
    record['error_category'] = 'usage_limit'
    record['usage_limit_info'] = {'login': secret}
_shared._report_outcomes(SimpleNamespace(results={name: record}), None)
""".replace("SOURCE", repr(source))
    monkeypatch.setenv("PYTHONPATH", str(Path(doctor.__file__).parents[1]))
    monkeypatch.setattr(
        doctor_smoke, "_bench_argv", lambda args: [sys.executable, "-c", script]
    )
    job = _job(tmp_path)
    job = doctor_smoke.SmokeJob(
        job.target, job.task_dir, job.jobs_dir, job.log_path, sandbox, job.timeout_sec
    )
    outcome = doctor_smoke.subprocess_runner(job)
    shown = job.log_path.read_text() + outcome.reason
    assert outcome.exit_code == 0, shown
    assert "synthetic-first" not in shown
    assert "synthetic-second" not in shown
    assert "***" in shown
