"""Guard FF-TEST-ENV-C5/C6 at 87114550 through final diagnostic outputs."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchflow import doctor, doctor_smoke
from benchflow.cli._errors import run_cli
from benchflow.cli._shared import print_error, print_hint, stopped_run_result
from benchflow.errors import MissingCredentialError
from tests.test_cli_credential_diagnostics import (
    isolated_credentials as isolated_credentials,
)
from tests.test_cli_opaque_diagnostics import _app
from tests.test_doctor_smoke import _job, _write_result


@pytest.mark.parametrize("source", ["error", "verifier_error", "log", "fallback-log"])
@pytest.mark.parametrize(
    "separator",
    [
        "",
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
@pytest.mark.parametrize("padded", [False, True])
def test_complete_credentials_survive_smoke_line_selection(
    source, separator, padded, tmp_path
):
    """87114550 C5: preserve the complete source through split/strip/cut."""
    secret = f"synthetic-first{separator}synthetic-second"
    if padded:
        secret = "  " + secret + "  "
    text = ("ERROR " if source != "fallback-log" else "diagnostic ") + secret
    job = _job(tmp_path)
    if source.endswith("log"):
        job.log_path.parent.mkdir(parents=True)
        job.log_path.write_text("ordinary preface\n" + text + "\n")
    else:
        _write_result(job, rewards=None, **{source: text})
    outcome = doctor_smoke.read_outcome(
        job, seconds=1, exit_code=1, environ={"DUMMY_TOKEN": secret}
    )
    assert "synthetic-first" not in outcome.reason
    assert "synthetic-second" not in outcome.reason
    assert "***" in outcome.reason
    assert "ordinary" not in outcome.reason


@pytest.mark.parametrize(
    "prefix",
    ["", "...", "-", "+", ".-+", "0.", "(", "[", "'", '"', "\\n", "Ω", "ERROR ..."],
)
@pytest.mark.parametrize("scheme", ["https", "HTTP", "git+https", "custom-2.1"])
@pytest.mark.parametrize("redactor", [doctor.redact, doctor.redact_exact])
def test_url_userinfo_is_masked_after_any_prefix(prefix, scheme, redactor):
    """87114550 C6: consume a maximal run without rejecting its URL suffix."""
    text = f"{prefix}{scheme}://bob:synthetic-password@proxy.example/v1"
    assert redactor(text, []) == f"{prefix}{scheme}://***@proxy.example/v1"


@pytest.mark.parametrize(
    "userinfo",
    [
        "bob:synthetic-password",
        "bob:synthetic@password",
        "bob:synthetic@@password",
        "bob:synthetic%40password",
    ],
)
@pytest.mark.parametrize(
    "redactor",
    [
        doctor.redact,
        doctor.redact_exact,
        lambda text, secrets: doctor._redact_url(text),
    ],
)
def test_complete_url_authority_uses_the_same_boundary(userinfo, redactor):
    """87114550 C6: userinfo ends at the authority's last @, on every path."""
    text = f"...https://{userinfo}@proxy.example/v1"
    assert redactor(text, []) == "...https://***@proxy.example/v1"


def test_protocol_relative_url_compatibility():
    """87114550: the URL helper retains its protocol-relative input contract."""
    assert (
        doctor._redact_url("//bob:synthetic@password@proxy.example/v1")
        == "//***@proxy.example/v1"
    )


@pytest.mark.parametrize("backend", ["click", "typer"])
@pytest.mark.parametrize("as_json", [False, True])
@pytest.mark.parametrize("how", ["expected", "printed", "stopped", "usage", "bug"])
@pytest.mark.parametrize(
    "secret",
    [
        "synthetic-first\nsynthetic-second",
        "  synthetic-padded  ",
        "...https://bob:synthetic-password@proxy.example/v1",
    ],
)
def test_cli_errors_hints_and_logs_share_the_choke_point(
    backend, as_json, how, secret, monkeypatch, capsys, tmp_path
):
    """87114550: keep original fields through both CLI implementations."""
    monkeypatch.setenv("DUMMY_TOKEN", secret)

    def fail(value, bad_parameter):
        message, hint = f"rejected: {secret}", f"replace {secret}"
        if how == "printed":
            print_error(message, next_step=hint)
            raise SystemExit(1)
        if how == "stopped":
            stopped_run_result(MissingCredentialError(message, hint=hint))
        if how == "usage":
            raise bad_parameter(message)
        if how == "bug":
            raise RuntimeError(message)
        raise MissingCredentialError(message, hint=hint)

    with pytest.raises(SystemExit):
        run_cli(_app(backend, fail), ["--json"] if as_json else [])
    out, err = capsys.readouterr()
    shown = (
        out + err + "".join(path.read_text() for path in (tmp_path / "logs").glob("*"))
    )
    assert "synthetic-first" not in shown and "synthetic-second" not in shown
    assert "synthetic-padded" not in shown and "synthetic-password" not in shown
    assert "***" in shown and "rejected:" in shown
    if as_json:
        assert json.loads(out)["exit_code"] in (1, 2)


def test_unregistered_url_userinfo_reaches_every_cli_sink(monkeypatch, capsys):
    """87114550 C6: base-URL credentials need no credential-named variable."""
    url = "...https://bob:synthetic-password@proxy.example/v1"
    print_hint("endpoint " + url)
    print_error("endpoint " + url, next_step="replace " + url)
    shown = "".join(capsys.readouterr())
    assert "synthetic-password" not in shown
    assert shown.count("...https://***@proxy.example/v1") == 3


@pytest.mark.parametrize("source", ["env", "login-file"])
@pytest.mark.parametrize("separator", ["\n", "\r\n"])
def test_complete_subprocess_log_is_masked_before_publication(
    source, separator, monkeypatch, tmp_path
):
    """87114550 C5/C6: child write boundaries and login sources cannot bypass."""
    secret = "  synthetic-first" + separator + "synthetic-second  "
    if source == "env":
        monkeypatch.setenv("DUMMY_TOKEN", secret)
    else:
        login = tmp_path / "home" / ".codex" / "auth.json"
        login.parent.mkdir(parents=True)
        login.write_text(json.dumps({"access_token": secret}))
    script = (
        "import sys; "
        f"text = {'ERROR ' + secret + ' ...https://bob:synthetic-password@proxy.example/v1'!r}; "
        "[sys.stdout.write(char) for char in text]; sys.exit(1)"
    )
    monkeypatch.setattr(
        doctor_smoke, "_bench_argv", lambda args: [sys.executable, "-c", script]
    )
    job = _job(tmp_path)
    outcome = doctor_smoke.subprocess_runner(job)
    shown = job.log_path.read_text() + outcome.reason
    assert "synthetic-first" not in shown and "synthetic-second" not in shown
    assert "synthetic-password" not in shown
    assert "ERROR *** ...https://***@proxy.example/v1" in shown


def test_summary_file_masks_all_fields(monkeypatch, tmp_path):
    """87114550: custom runners cannot bypass the summary-file sink."""
    from datetime import UTC, datetime

    secret = "synthetic-first\nsynthetic-second"
    monkeypatch.setenv("DUMMY_TOKEN", secret)
    job = _job(tmp_path)
    outcome = doctor_smoke.SmokeOutcome(
        job.target,
        "error",
        None,
        1,
        job.log_path,
        reason="ERROR "
        + secret
        + " ...https://bob:synthetic-password@proxy.example/v1",
    )
    path = doctor_smoke.write_summary(
        tmp_path, [outcome], sandbox="docker", started_at=datetime.now(UTC)
    )
    reason = json.loads(path.read_text())["runs"][0]["reason"]
    assert reason == "ERROR *** ...https://***@proxy.example/v1"


@pytest.mark.parametrize("source", ["error", "verifier_error", "log"])
def test_final_join_cannot_create_an_unmasked_credential(source, tmp_path):
    """87114550 C5: match the completed summary as well as its intact sources."""
    job = _job(tmp_path)
    if source == "log":
        job.log_path.parent.mkdir(parents=True)
        job.log_path.write_text("ERROR synthetic-second")
        secret = "bench exited 1 without writing result.json: ERROR synthetic-second"
    else:
        _write_result(
            job,
            rewards=None,
            **{source: "synthetic-second", source + "_category": "synthetic-first"},
        )
        secret = "synthetic-first: synthetic-second"
    outcome = doctor_smoke.read_outcome(
        job, seconds=1, exit_code=1, environ={"DUMMY_TOKEN": secret}
    )
    assert outcome.reason == "***"


def test_json_protocol_separators_are_never_redaction_input(monkeypatch, tmp_path):
    """87114550 C5: mask final string values without consuming JSON syntax."""
    from datetime import UTC, datetime

    secret = '"status": "error"'
    monkeypatch.setenv("DUMMY_TOKEN", secret)
    job = _job(tmp_path)
    outcome = doctor_smoke.SmokeOutcome(
        job.target, "error", None, 1, job.log_path, reason=secret
    )
    path = doctor_smoke.write_summary(
        tmp_path, [outcome], sandbox="docker", started_at=datetime.now(UTC)
    )
    payload = json.loads(path.read_text())
    assert payload["runs"][0]["status"] == "error"
    assert payload["runs"][0]["reason"] == "***"


def test_final_cli_message_hint_join_is_masked(monkeypatch, capsys):
    """87114550 C5: the final CLI boundary owns message/next joining."""
    monkeypatch.setenv("DUMMY_TOKEN", "synthetic-first\n  next: synthetic-second")
    print_error("synthetic-first", next_step="synthetic-second")
    assert capsys.readouterr().err == "***\n"


@pytest.mark.parametrize(
    "secret",
    ["synthetic-first\nsynthetic-second", "synthetic-first\udcffsynthetic-second"],
)
def test_non_utf8_child_output_keeps_log_publication_total(
    secret, monkeypatch, tmp_path
):
    """87114550 C5: arbitrary child bytes cannot break the output boundary."""
    monkeypatch.setenv("DUMMY_TOKEN", secret)
    payload = b"\xffERROR " + secret.encode("utf-8", errors="surrogateescape") + b"\xfe"
    script = f"import sys; sys.stdout.buffer.write({payload!r}); sys.exit(1)"
    monkeypatch.setattr(
        doctor_smoke, "_bench_argv", lambda args: [sys.executable, "-c", script]
    )
    job = _job(tmp_path)
    outcome = doctor_smoke.subprocess_runner(job)
    shown = job.log_path.read_text() + outcome.reason
    assert "synthetic-first" not in shown and "synthetic-second" not in shown
    assert "ERROR ***" in shown


@pytest.mark.parametrize(
    "secret",
    ["synthetic[first]", "synthetic-first\nsynthetic-second", "  synthetic-padded  "],
)
def test_doctor_and_smoke_terminal_fields_are_protected_before_markup(
    secret, monkeypatch, capsys, tmp_path
):
    """87114550: markup and column assembly must stay inside the choke point."""
    from benchflow.cli.doctor import _render_outcomes, render_report

    monkeypatch.setenv("DUMMY_TOKEN", secret)
    check = doctor.Check(
        "custom", "network", "endpoint " + secret, "fail", "ERROR " + secret
    )
    render_report(doctor.DoctorReport([check], "docker"))
    job = _job(tmp_path)
    outcome = doctor_smoke.SmokeOutcome(
        doctor_smoke.SmokeTarget("codex-acp", "model " + secret, "test"),
        "error",
        None,
        1,
        job.log_path,
        reason="ERROR " + secret,
    )
    _render_outcomes([outcome])
    shown = "".join(capsys.readouterr())
    assert "synthetic" not in shown
    assert "***" in shown


def test_overlap_between_literal_and_url_matches():
    """87114550: URL matching cannot consume half of a longer known value."""
    secret = "synthetic-password@proxy.example/v1"
    assert doctor.redact("...https://bob:" + secret, [secret]) == "...https://***"
    assert (
        doctor.redact("synthetic://bob:password@host", ["synthetic"])
        == "***://***@host"
    )


def test_traceback_frame_fields_reach_the_same_boundary():
    """87114550: a raw frame path must survive traceback/group formatting."""
    secret = "synthetic-frame\nsynthetic-path"
    try:
        exec(
            compile(
                "raise RuntimeError('failure')", "/tmp/" + secret + "/input", "exec"
            )
        )
    except RuntimeError as exc:
        shown = doctor.redacted_traceback(exc, [secret])
    assert "synthetic-frame" not in shown and "synthetic-path" not in shown
    assert "/tmp/***/input" in shown


def test_marker_collision_and_malformed_input_remain_bounded():
    """87114550 C4/C5/C6: marker selection and scheme matching stay bounded."""
    script = """
from benchflow import doctor
from time import perf_counter
text = "".join(map(chr, range(0xe000, 0xf900))) + "a-" * 524288
start = perf_counter()
assert doctor.redact(text, ["synthetic-secret"]) == text
assert perf_counter() - start < 2
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=5)


def test_user_visible_diagnostic_sinks_are_registered():
    """87114550: a new direct sink or an early fragment redactor fails this sweep."""
    root = Path(doctor.__file__).parent
    # This is a grep-style inventory of actual output operations, not an
    # inventory of redaction spellings. Adding a sink requires explicit review.
    expected = {
        "cli/_errors.py": {
            "_WatchedStdout": 2,
            "write_crash_log": 1,
            "_emit_json": 1,
            "report": 1,
            "_run_cli": 3,
        },
        "cli/_shared.py": {
            "print_hint": 1,
            "print_error": 1,
            "warn_deprecated": 1,
            "_report_checkpoint_retries": 1,
            "_report_eval_result": 9,
            "_report_outcomes": 5,
        },
        # PT's key check (a48cb677): the provider, the variable's name and
        # the HTTP status, never the key.
        "cli/doctor.py": {"_check_provider_key": 1, "_print": 1, "register_doctor": 1},
        "doctor_smoke.py": {"subprocess_runner": 2, "write_summary": 1},
    }
    actual = {}
    for relative in expected:
        tree = ast.parse((root / relative).read_text())
        found = {}
        for node in tree.body:
            owner = (
                node.name
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                )
                else "<module>"
            )
            count = sum(
                isinstance(call, ast.Call)
                and (
                    (isinstance(call.func, ast.Name) and call.func.id == "print")
                    or (
                        isinstance(call.func, ast.Attribute)
                        and call.func.attr
                        in {
                            "print",
                            "print_json",
                            "write",
                            "writelines",
                            "write_text",
                            "write_bytes",
                            "echo",
                            "log",
                            "debug",
                            "info",
                            "warning",
                            "warn",
                            "error",
                            "exception",
                            "critical",
                            "fatal",
                        }
                    )
                )
                for call in ast.walk(node)
            )
            if count:
                found[owner] = found.get(owner, 0) + count
        actual[relative] = found
    assert actual == expected
    smoke_source = (root / "doctor_smoke.py").read_text()
    assert "redact(line" not in smoke_source
    assert "redact(reason" not in smoke_source
    assert "render_diagnostic" in smoke_source
