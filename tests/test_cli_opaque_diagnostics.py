"""Guards FF-CLI-C5/C6 at 4c7cc8cc: diagnostics mask credentials while retaining messages."""

from __future__ import annotations

import json
import os

import click
import pytest
import typer

from benchflow import doctor
from benchflow.cli._errors import RECENT_LOG, run_cli
from benchflow.cli._shared import print_error
from benchflow.errors import MissingCredentialError


@pytest.fixture(autouse=True)
def isolated_credentials(tmp_path, monkeypatch):
    for key in list(os.environ):
        if doctor._SECRET_NAME_RE.search(key):
            monkeypatch.delenv(key)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("BENCHFLOW_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.chdir(tmp_path)
    RECENT_LOG.lines.clear()


def _app(backend, action):
    if backend == "click":

        @click.command()
        @click.option("--hf-token", default="")
        @click.option("--json", "output_json", is_flag=True)
        def command(hf_token, output_json):
            action(hf_token, click.BadParameter)

        return command
    app = typer.Typer()

    @app.command()
    def go(
        hf_token: str = typer.Option("", "--hf-token"),
        output_json: bool = typer.Option(False, "--json"),
    ):
        action(hf_token, typer.BadParameter)

    return app


@pytest.mark.parametrize("backend", ["click", "typer"])
@pytest.mark.parametrize("as_json", [False, True])
@pytest.mark.parametrize("attached", [False, True])
@pytest.mark.parametrize("secret", ["-ab", "--synthetic-token", "--", "-", "Z"])
def test_credential_option_values_respect_the_placeholder_cutoff(
    backend, as_json, attached, secret, capsys
):
    """FF-CLI-C5 with b29's cutoff: keep each option's surrounding message."""
    seen = []

    def fail(value, bad_parameter):
        seen.append(value)
        print_error(f"rejected: {value}", next_step=f"replace {value}")
        raise bad_parameter(f"invalid: {value}")

    args = [f"--hf-token={secret}"] if attached else ["--hf-token", secret]
    if as_json:
        args.append("--json")
    with pytest.raises(SystemExit) as caught:
        run_cli(_app(backend, fail), args)
    assert seen == [secret]
    assert caught.value.code == 2
    out, err = capsys.readouterr()
    masked = "***" if len(secret) >= 6 else secret
    assert f"rejected: {masked}" in err and f"invalid: {masked}" in out + err
    if as_json:
        assert json.loads(out)["exit_code"] == 2


@pytest.mark.parametrize("as_json", [False, True])
@pytest.mark.parametrize("source", ["argv", "env", "dotenv", "login"])
@pytest.mark.parametrize("how", ["printed", "expected", "file", "bug", "usage"])
def test_credential_context_preserves_unrelated_payloads(
    as_json, source, how, capsys, tmp_path, monkeypatch
):
    """Batch 7 at f79b2260: only exact values are credentials to redact."""
    secret = "synthetic-opaque-credential"
    # Transformed strings are not exact credential values. The user-approved
    # batch-7 contract keeps them, including when a credential is configured.
    payload = secret[::-1] + " " + str(list(secret.encode()))
    if source == "env":
        monkeypatch.setenv("D2_PASSWORD", secret)
    elif source == "dotenv":
        (tmp_path / ".env").write_text(f"D2_PASSWORD={secret}\n")
    elif source == "login":
        folder = tmp_path / "home" / ".codex"
        folder.mkdir(parents=True)
        (folder / "auth.json").write_text(json.dumps({"access_token": secret}))

    def fail(value, bad_parameter):
        if how == "printed":
            print_error(payload, next_step=payload)
            raise typer.Exit(1)
        if how == "expected":
            raise MissingCredentialError(payload, hint=payload)
        if how == "file":
            raise FileNotFoundError(2, payload, payload)
        if how == "usage":
            raise bad_parameter(payload)
        RECENT_LOG.lines.append(payload)
        raise RuntimeError(payload)

    args = ["--hf-token", secret] if source == "argv" else []
    if as_json:
        args.append("--json")
    with pytest.raises(SystemExit) as caught:
        run_cli(_app("typer", fail), args)
    assert caught.value.code == (2 if how == "usage" else 1)
    out, err = capsys.readouterr()
    shown = out + err + "".join(p.read_text() for p in (tmp_path / "logs").glob("*"))
    assert secret[::-1] in shown
    assert secret not in shown
    if as_json:
        assert json.loads(out)["exit_code"] == caught.value.code


@pytest.mark.parametrize("as_json", [False, True])
def test_credential_tracebacks_keep_messages_and_mask_values(
    as_json, capsys, monkeypatch, tmp_path
):
    """Batch 7 at f79b2260: groups retain useful messages and custom kinds."""
    monkeypatch.setenv("D2_PASSWORD", "synthetic-opaque-credential")

    class PrivateError(Exception):
        pass

    exc = PrivateError("invalid: synthetic-opaque-credential")

    def fail(value, bad_parameter):
        raise ExceptionGroup("private group message", [exc])

    with pytest.raises(SystemExit) as caught:
        run_cli(_app("typer", fail), ["--json"] if as_json else [])
    assert caught.value.code == 1
    _out, err = capsys.readouterr()
    (log,) = (tmp_path / "logs").glob("*")
    for shown in (err, log.read_text()):
        assert "private group message" in shown
        assert "PrivateError: invalid: ***" in shown
        assert "synthetic-opaque-credential" not in shown
    assert str(exc) == "invalid: synthetic-opaque-credential"
