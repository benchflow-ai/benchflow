"""Guards FF-CLI-C3/C4 at 2a867932: credentials survive all error formatting."""

from __future__ import annotations

import json
import logging
import os

import pytest
import typer

from benchflow import doctor
from benchflow.cli._errors import run_cli
from benchflow.errors import MissingCredentialError


@pytest.fixture(autouse=True)
def isolated_credentials(tmp_path, monkeypatch):
    """Only synthetic credentials, with no login files or inherited keys."""
    from benchflow.cli._errors import RECENT_LOG

    for key in list(os.environ):
        if doctor._SECRET_NAME_RE.search(key):
            monkeypatch.delenv(key)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("BENCHFLOW_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.chdir(tmp_path)
    RECENT_LOG.lines.clear()


def invoke(action, as_json):
    app = typer.Typer()

    @app.command()
    def go(output_json: bool = typer.Option(False, "--json")):
        action()

    with pytest.raises(SystemExit) as caught:
        run_cli(app, ["--json"] if as_json else [])
    assert caught.value.code == 1


@pytest.mark.parametrize("secret", ["abc\ndef\n", "\nZ\n", "Q", "xy", " " * 8 + "Z"])
@pytest.mark.parametrize("how", ["raised", "printed", "stopped", "file", "bug"])
@pytest.mark.parametrize("as_json", [False, True])
def test_credential_cutoff_is_applied_before_error_formatting(
    secret, how, as_json, monkeypatch, capsys, tmp_path
):
    """FF-CLI-C3 siblings retain b29's explicit six-character cutoff."""
    from benchflow.cli._shared import print_error, stopped_run_result

    monkeypatch.setenv("D2_PASSWORD", secret)

    def fail():
        if how == "printed":
            print_error(f"rejected: {secret}", next_step=f"replace {secret}")
            raise typer.Exit(1)
        if how == "stopped":
            stopped_run_result(MissingCredentialError(f"rejected: {secret}"))
        if how == "file":
            raise FileNotFoundError(2, "missing", secret)
        if how == "bug":
            raise RuntimeError(f"rejected: {secret}")
        raise MissingCredentialError(f"rejected: {secret}", hint=f"replace {secret}")

    invoke(fail, as_json)
    out, err = capsys.readouterr()
    shown = out + err + "".join(p.read_text() for p in (tmp_path / "logs").glob("*"))
    # Batch 7 at f79b2260: match the whole credential, never infer fragments
    # (a padded Z must not erase the unrelated UTC Z in the log filename).
    if len(secret) >= 6:
        assert secret not in shown
        assert f"rejected: {secret.strip()}" not in shown
        assert "***" in shown
    else:
        # Normalization may strip whitespace; the placeholder remains readable.
        assert secret.strip() in shown
    if as_json:
        assert json.loads(out)["exit_code"] == 1


@pytest.mark.parametrize("chain", ["cause", "context"])
@pytest.mark.parametrize("as_json", [False, True])
def test_nested_groups_chains_and_notes_are_masked_without_mutating_the_error(
    chain, as_json, monkeypatch, capsys, tmp_path
):
    """Guards FF-CLI-C4 in both sinks, nested groups, chains and notes."""
    secret = "synthetic-C4-first\nsynthetic-C4-second\n"
    monkeypatch.setenv("D2_PASSWORD", secret)
    child = RuntimeError(f"rejected: {secret}")
    child.add_note(f"child note: {secret}")
    inner = ExceptionGroup(f"inner: {secret}", [child])
    outer = ExceptionGroup("batch failed", [inner])
    outer.add_note(f"outer note: {secret}")
    previous = ValueError(f"previous: {secret}")
    previous.add_note(f"previous note: {secret}")
    if chain == "cause":
        outer.__cause__ = previous
    else:
        outer.__context__ = previous

    def fail():
        raise outer

    invoke(fail, as_json)
    out, err = capsys.readouterr()
    (log,) = (tmp_path / "logs").glob("*")
    for shown in (out, err, log.read_text()):
        for component in secret.splitlines():
            assert component not in shown
    assert "ExceptionGroup" in err and "RuntimeError" in err
    # FF-CLI-C6: note payloads are now omitted without inspecting their values.
    assert "*** (exception notes omitted)" in err
    assert "previous note:" not in err and "child note:" not in err
    assert str(child) == f"rejected: {secret}"
    assert child.__notes__ == [f"child note: {secret}"]
    assert str(inner).startswith(f"inner: {secret}")


def test_recent_log_exception_groups_are_redacted(monkeypatch, capsys, tmp_path):
    """Guards FF-CLI-C4 when logging formatted a group before the crash."""
    secret = "synthetic-log-first\nsynthetic-log-second\n"
    monkeypatch.setenv("D2_PASSWORD", secret)

    def fail():
        try:
            raise ExceptionGroup("logged", [ValueError(secret)])
        except ExceptionGroup:
            logging.getLogger("benchflow.redaction").exception("earlier failure")
        raise RuntimeError("later failure")

    invoke(fail, True)
    capsys.readouterr()
    (log,) = (tmp_path / "logs").glob("*")
    shown = log.read_text()
    # FF-CLI-C5: even nonmatching recent-log payloads are omitted.
    assert "earlier failure" not in shown and "***" in shown
    for component in secret.splitlines():
        assert component not in shown


@pytest.mark.parametrize("as_json", [False, True])
def test_base_exception_groups_use_the_same_redacted_report(
    as_json, monkeypatch, capsys, tmp_path
):
    """Guards FF-CLI-C4 for groups containing a non-Exception leaf."""
    secret = "synthetic-base-first\nsynthetic-base-second\n"
    monkeypatch.setenv("D2_PASSWORD", secret)
    exc = BaseExceptionGroup("base group", [BaseException(secret)])

    def fail():
        raise exc

    invoke(fail, as_json)
    out, err = capsys.readouterr()
    (log,) = (tmp_path / "logs").glob("*")
    for shown in (out, err, log.read_text()):
        for component in secret.splitlines():
            assert component not in shown


@pytest.mark.parametrize(
    "secret", ["Z", "xy", "abc\ndef\n", "\nZ\n", "ab\r\ncd\r\n", "ab\u2028cd\u2029"]
)
def test_shared_redactor_uses_whole_values_without_inferred_components(secret):
    """b29 supersedes FF-CLI-C3/C4's fragment matching; redact before splitting."""
    secrets = doctor.secret_values({"D2_PASSWORD": secret})
    assert secret in secrets
    assert doctor.redact(secret, secrets) == ("***" if len(secret) >= 6 else secret)
    rendered = "\n".join(f"  | {line}" for line in secret.strip().splitlines())
    assert doctor.redact(rendered, secrets) == rendered


def test_doctor_redacts_a_multiline_value_before_taking_the_first_line():
    """Guards FF-CLI-C3/C4's doctor sibling before split and truncation."""
    secret = "synthetic-doctor-first\nsynthetic-doctor-second\n"
    assert (
        doctor._first_line(f"rejected: {secret}", secrets=[secret]) == "rejected: ***"
    )


@pytest.mark.parametrize("render", [repr, json.dumps])
def test_escaped_multiline_credentials_are_masked_in_formatted_errors(render):
    """Guards FF-CLI-C3/C4's OSError, source-line and SDK formatting siblings."""
    secret = "synthetic-'quoted\nsynthetic-\"quoted\\value\n"
    # OSError formats filenames with repr before the CLI sees str(exc), and
    # SDK errors can already contain serialized JSON. Mask the whole value.
    shown = doctor.redact(f"rejected: {render(secret)}", [secret])
    assert shown in ("rejected: '***'", 'rejected: "***"')


@pytest.mark.parametrize("as_json", [False, True])
def test_printed_and_usage_errors_collect_explicit_argv(as_json, capsys):
    """Guards FF-CLI-C3 in print_error and Click's formatted usage errors."""
    from benchflow.cli._shared import print_error

    app = typer.Typer()

    @app.command()
    def go(
        hf_token: str = typer.Option("", "--hf-token"),
        output_json: bool = typer.Option(False, "--json"),
    ):
        print_error(f"rejected: {hf_token}", next_step=f"replace {hf_token}")
        raise typer.BadParameter(f"invalid: {hf_token}")

    with pytest.raises(SystemExit) as caught:
        run_cli(app, ["--hf-token", "Z", *(["--json"] if as_json else [])])
    assert caught.value.code == 2
    out, err = capsys.readouterr()
    assert "rejected: Z" in err
    assert "invalid: Z" in out + err


def test_syntax_error_source_and_notes_use_the_shared_redactor(
    monkeypatch, capsys, tmp_path
):
    """Guards FF-CLI-C4's separately formatted SyntaxError fields."""
    secret = "synthetic-syntax-first\nsynthetic-syntax-second\n"
    monkeypatch.setenv("D2_PASSWORD", secret)
    exc = SyntaxError(secret, ("input.py", 1, 1, secret))
    exc.add_note(secret)

    def fail():
        raise exc

    invoke(fail, True)
    out, err = capsys.readouterr()
    (log,) = (tmp_path / "logs").glob("*")
    shown = out + err + log.read_text()
    assert "SyntaxError" in shown and "***" in shown
    for component in secret.splitlines():
        assert component not in shown
    assert exc.text == secret and exc.msg == secret and exc.__notes__ == [secret]


@pytest.mark.parametrize("as_json", [False, True])
@pytest.mark.parametrize(
    "shape", ["str", "bytes", "int", "list", "chars", "tuple", "dict"]
)
@pytest.mark.parametrize("link", ["root", "group", "cause", "context"])
def test_exception_notes_are_opaque_for_every_shape(
    as_json, shape, link, capsys, tmp_path, monkeypatch
):
    """FF-CLI-C6: scalar, sequence and malformed notes never reach a formatter."""
    secret = "synthetic-note-token"
    monkeypatch.setenv("D2_PASSWORD", secret)
    notes = {
        "str": secret,
        "bytes": secret.encode(),
        "int": 123456789,
        "list": [secret],
        "chars": list(secret),
        "tuple": (secret,),
        "dict": {secret: secret},
    }[shape]
    child = RuntimeError("rejected")
    child.__notes__ = notes
    outer = child
    if link == "group":
        outer = ExceptionGroup("batch failed", [ExceptionGroup("nested", [child])])
    elif link in ("cause", "context"):
        outer = RuntimeError("wrapper")
        setattr(outer, f"__{link}__", child)

    def fail():
        raise outer

    invoke(fail, as_json)
    out, err = capsys.readouterr()
    (log,) = (tmp_path / "logs").glob("*")
    for shown in (err, log.read_text()):
        assert secret not in shown
        assert "*** (exception notes omitted)" in shown
        assert "RuntimeError" in shown
    assert child.__notes__ is notes
    if as_json:
        assert json.loads(out)["exit_code"] == 1


def test_notes_are_omitted_without_known_secrets():
    """FF-CLI-C6: no matching credential is required to omit note payloads."""
    exc = RuntimeError("ordinary failure")
    exc.__notes__ = "private-note-payload"
    shown = doctor.redacted_traceback(exc, [])
    assert "*** (exception notes omitted)" in shown
    assert "ordinary failure" in shown
    assert exc.__notes__ == "private-note-payload"
