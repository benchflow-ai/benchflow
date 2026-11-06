"""Guard FF-TEST-ENV-C7/C8 at 727eab15 at the actual child output boundary."""

from __future__ import annotations

import shlex
import sys

import pytest

from benchflow import doctor_smoke
from benchflow._diagnostics import redact, redact_stream
from tests.test_cli_credential_diagnostics import (
    isolated_credentials as isolated_credentials,
)
from tests.test_doctor_smoke import _job


@pytest.mark.parametrize("scheme", ["https", "HTTP", "git+https", "custom-2.1"])
@pytest.mark.parametrize("userinfo", ["bob:pass", "bob:pass@word", "bob:p%40ss"])
@pytest.mark.parametrize("ending", [" ", "\n", "\r\n", ""])
def test_url_every_split(scheme, userinfo, ending):
    """727eab15 C8: retain schemes and userinfo together at every read split."""
    text = f"{scheme}://{userinfo}@h{ending}"
    expected = f"{scheme}://***@h{ending}"
    for size in range(6, len(text) + 8):
        secrets = ["z" * size]
        for offset in range(len(text) + 1):
            assert (
                "".join(redact_stream([text[:offset], text[offset:]], secrets))
                == expected
            )
    assert "".join(redact_stream(iter(text), ["abcdef"])) == expected


@pytest.mark.parametrize("sandbox", ["docker", "daytona"])
def test_url_every_public_log_split(sandbox, monkeypatch, tmp_path):
    """727eab15 C8: all scheme/userinfo cuts cross the real 8 KiB log read."""
    monkeypatch.setenv("DUMMY_TOKEN", "abcdef")
    original = _job(tmp_path)
    job = doctor_smoke.SmokeJob(
        original.target,
        original.task_dir,
        original.jobs_dir,
        original.log_path,
        sandbox,
        original.timeout_sec,
    )
    header = "$ bench " + shlex.join(doctor_smoke.eval_command(job)) + "\n"
    text = ""
    url = "https://bob:pass@h "
    for offset in range(len(url) + 1):
        text += (
            "x" * ((8192 - (len(header) + len(text)) % 8192 - offset - 1) % 8192)
            + " "
            + url
            + "\n"
        )
    payload = tmp_path / "synthetic-output.txt"
    payload.write_text(text)
    script = f"import os; data = open({str(payload)!r}, 'rb').read(); [os.write(1, data[i:i+7]) for i in range(0, len(data), 7)]"
    monkeypatch.setattr(
        doctor_smoke, "_bench_argv", lambda args: [sys.executable, "-c", script]
    )
    outcome = doctor_smoke.subprocess_runner(job)
    assert outcome.exit_code == 0
    assert job.log_path.read_text() == redact(header + text, ["abcdef"])


@pytest.mark.parametrize("kind", ["literal", "url", "multiline"])
@pytest.mark.parametrize("step", [1, 7, 64, 512])
def test_oversized_units_never_publish_a_fragment(kind, step, monkeypatch):
    """727eab15 C8: the cap suppresses a whole unit and its continuation."""
    import benchflow._diagnostics as diagnostics

    monkeypatch.setattr(diagnostics, "_OUTPUT_UNIT_LIMIT", 64)
    secret = "synthetic-first" + "X" * 200 + "synthetic-last"
    if kind == "multiline":
        secret = "synthetic-first\n" + "X" * 200 + "synthetic-last"
    body = "https://bob:" + secret + "@host" if kind == "url" else secret
    text = "safe\n" + body + "\nafter\n"
    shown = "".join(
        redact_stream((text[i : i + step] for i in range(0, len(text), step)), [secret])
    )
    assert "synthetic-" not in shown and "bob:" not in shown and "XXX" not in shown
    assert "oversized diagnostic omitted" in shown
    if kind != "multiline":
        assert shown == "safe\n" + diagnostics._OMITTED_UNIT + "after\n"
    else:
        assert shown == diagnostics._OMITTED_UNIT


def test_no_partial_line_is_published_before_flush():
    """727eab15 C8: complete lines are the only non-EOF publication units."""
    consumed = []

    def chunks():
        yield "https://bob:"
        consumed.append("read continuation")
        yield "pass@h\n"
        pytest.fail("the complete line must be available before another read")

    assert next(redact_stream(chunks(), ["abcdef"])) == "https://***@h\n"
    assert consumed == ["read continuation"]
