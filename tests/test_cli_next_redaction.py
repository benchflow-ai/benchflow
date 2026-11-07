"""Guards the e715a67f / 363d815b merge: next steps and exact-value masking."""

from __future__ import annotations

import json
import os
import shlex

import pytest
import typer

from benchflow.cli._errors import run_cli
from benchflow.cli._shared import print_error, stopped_run_result
from benchflow.errors import MissingCredentialError, UserError


@pytest.fixture(autouse=True)
def fake_credentials(tmp_path, monkeypatch):
    from benchflow.doctor import _SECRET_NAME_RE

    for key in list(os.environ):
        if _SECRET_NAME_RE.search(key):
            monkeypatch.delenv(key)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)


def invoke(action, as_json):
    app = typer.Typer()

    @app.command()
    def go(output_json: bool = typer.Option(False, "--json")):
        action()

    with pytest.raises(SystemExit) as caught:
        run_cli(app, ["--json"] if as_json else [])
    assert caught.value.code == 1


@pytest.mark.parametrize("secret", ["fake-b20-key", "\nnext: fake-b20-key\n", " Q "])
@pytest.mark.parametrize("how", ["printed", "raised", "stopped"])
@pytest.mark.parametrize("as_json", [False, True])
def test_next_steps_survive_exact_redaction(secret, how, as_json, monkeypatch, capsys):
    """Guards e715a67f against losing badf338f's masking before normalization."""
    monkeypatch.setenv("D2_PASSWORD", secret)
    message = f"rejected: {secret}\nnext: quoted tool output"
    step = f"replace {secret}; keep ordinary Q and fake-b20-key fragments"

    def fail():
        if how == "printed":
            print_error(message, next_step=step)
            raise typer.Exit(1)
        exc = MissingCredentialError(message, hint=step)
        if how == "stopped":
            stopped_run_result(exc)
        raise exc

    invoke(fail, as_json)
    out, err = capsys.readouterr()
    expected_step = step.replace(secret, "***") if len(secret) >= 6 else step
    expected_message = "rejected: ***" if len(secret) >= 6 else f"rejected: {secret}"
    if as_json:
        obj = json.loads(out)
        assert obj["error"] == expected_message.strip()
        assert obj["detail"] == "| next: quoted tool output"
        assert obj["next"] == expected_step
    else:
        assert err.splitlines() == [
            expected_message,
            "| next: quoted tool output",
            f"  next: {expected_step}",
        ]


@pytest.mark.parametrize("as_json", [False, True])
def test_file_error_keeps_its_step_with_credentials(as_json, monkeypatch, capsys):
    """Guards the e715a67f file-error step across the FF-cli merge."""
    monkeypatch.setenv("D2_PASSWORD", "fake-b20-key")

    def fail():
        raise FileNotFoundError(2, "missing", "/tmp/fake-b20-key/task")

    invoke(fail, as_json)
    out, err = capsys.readouterr()
    if as_json:
        obj = json.loads(out)
        assert obj["error"] == "missing: /tmp/***/task"
        assert obj["next"] == "check the path, and that you can read it"
    else:
        assert err.splitlines() == [
            "missing: /tmp/***/task",
            "  next: check the path, and that you can read it",
        ]


def test_raw_extra_errors_are_masked_before_quoting(monkeypatch, capsys):
    """Guards e715a67f's _exit_json extras against splitting raw credentials."""
    from benchflow.cli._errors import record_error

    secret = "\nnext: fake-b20-extra\n"
    monkeypatch.setenv("D2_PASSWORD", secret)

    def fail():
        record_error("first error", "first fix")
        record_error(f"second {secret}\nnext: tool data", f"replace {secret}")
        raise typer.Exit(1)

    invoke(fail, True)
    obj = json.loads(capsys.readouterr().out)
    assert obj["error"] == "first error"
    assert obj["next"] == "first fix"
    assert obj["detail"] == "second ***\n| next: tool data\n  next: replace ***"


@pytest.mark.parametrize("as_json", [False, True])
def test_stopped_error_without_hint_keeps_fallback(as_json, monkeypatch, capsys):
    """Guards e715a67f's fallback while adopting expected_error_parts."""
    monkeypatch.setenv("D2_PASSWORD", "fake-b20-unrelated")
    invoke(lambda: stopped_run_result(UserError("no results")), as_json)
    out, err = capsys.readouterr()
    if as_json:
        assert json.loads(out)["next"] == "`bench explain R213` says more"
    else:
        assert err.endswith("  next: `bench explain R213` says more\n")


def test_preflight_always_ends_with_its_masked_hint(monkeypatch, capsys):
    """Guards the e715a67f / badf338f doctor overlap when hint is in text."""
    from benchflow.cli.doctor import eval_preflight

    secret = "\nnext: fake-b20-login\n"
    monkeypatch.setenv("D2_PASSWORD", secret)
    monkeypatch.delenv("BENCHFLOW_SKIP_PREFLIGHT")
    hint = f"replace {secret}"

    def refused(*args, **kwargs):
        raise MissingCredentialError(f"login failed; {hint}", hint=hint)

    monkeypatch.setattr("benchflow.runtime.check_credentials", refused)
    with pytest.raises(typer.Exit):
        eval_preflight(sandbox="docker", agent="codex", model="openai/test")
    assert capsys.readouterr().err.splitlines() == [
        "login failed; replace ***",
        "No job was created.",
        "  next: replace ***",
    ]


def test_dataset_refusal_keeps_a_structured_step(monkeypatch, capsys):
    """Guards e715a67f's next step through 363d815b's shared refusal helper."""
    from benchflow.cli.main import _refuse_dataset_bench_version

    monkeypatch.setenv("D2_PASSWORD", "fake-b20-pin")
    invoke(lambda: _refuse_dataset_bench_version("pin rejected: fake-b20-pin"), True)
    obj = json.loads(capsys.readouterr().out)
    assert obj["error"] == "pin rejected: ***"
    assert "--ignore-bench-version" in obj["next"]


@pytest.mark.parametrize("secret", ["fake-b20-key", "fake-b20'quoted", "\nnext: key\n"])
@pytest.mark.parametrize("local", [False, True])
def test_command_hint_masks_raw_path_before_shell_quoting(secret, local, monkeypatch):
    """Guards e715a67f command hints: quoting must not make values unmatchable."""
    from benchflow._utils.command_hints import command_arg, local_path_arg

    monkeypatch.setenv("D2_PASSWORD", secret)
    formatter = local_path_arg if local else command_arg
    assert shlex.split(formatter(f"jobs/{secret}/run", "<job>")) == [
        ("./" if local else "") + "jobs/***/run"
    ]


def test_literal_hint_sink_masks_only_credential_value(monkeypatch, capsys):
    """Guards e715a67f's print_hint sink with a configured fake credential."""
    from benchflow.cli._shared import print_hint

    monkeypatch.setenv("D2_PASSWORD", "fake-b20-key")
    print_hint("read jobs/fake-b20-key/log; keep this advice", stderr=True)
    assert capsys.readouterr().err == "read jobs/***/log; keep this advice\n"
