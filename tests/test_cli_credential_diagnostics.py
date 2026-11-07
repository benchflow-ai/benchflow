"""Guards batch 7's FF-cli regression at f79b2260: preserve error text."""

from __future__ import annotations

import json
import os

import click
import pytest
import typer

from benchflow import doctor
from benchflow.cli._errors import RECENT_LOG, _redact, run_cli
from benchflow.cli._shared import print_error, stopped_run_result
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


@pytest.mark.parametrize("backend", ["click", "typer"])
@pytest.mark.parametrize("as_json", [False, True])
@pytest.mark.parametrize(
    "source",
    [
        "env",
        "dotenv",
        "codex",
        "claude",
        "login-path",
        "inline-json",
        "option",
        "attached",
        "assignment",
        "attached-assignment",
    ],
)
@pytest.mark.parametrize("secret_in_text", [False, True])
@pytest.mark.parametrize(
    "how",
    [
        "expected",
        "printed",
        "stopped",
        "missing",
        "not-directory",
        "is-directory",
        "permission",
        "usage",
    ],
)
def test_expected_errors_keep_their_text_with_credentials(
    backend, as_json, source, secret_in_text, how, tmp_path, monkeypatch, capsys
):
    """f79b2260: every expected-error route keeps text and masks only values."""
    secret = "synthetic-batch7-credential"
    args = []
    if source == "env":
        monkeypatch.setenv("ANTHROPIC_API_KEY", secret)
    elif source == "dotenv":
        (tmp_path / ".env").write_text(f"ANTHROPIC_API_KEY={secret}\n")
    elif source in ("codex", "claude", "login-path", "inline-json"):
        data = json.dumps({"access_token": secret})
        if source == "inline-json":
            monkeypatch.setenv("CODEX_AUTH_JSON", data)
        else:
            folder = tmp_path / "home" / (".claude" if source == "claude" else ".codex")
            folder.mkdir(parents=True)
            path = folder / (".credentials.json" if source == "claude" else "auth.json")
            if source == "login-path":
                path = tmp_path / "login.json"
                monkeypatch.setenv("CODEX_AUTH_JSON", str(path))
            path.write_text(data)
    elif source == "option":
        args = ["--hf-token", secret]
    elif source == "attached":
        args = [f"--hf-token={secret}"]
    elif source == "assignment":
        args = ["--agent-env", f"ANTHROPIC_API_KEY={secret}"]
    else:
        args = [f"--agent-env=ANTHROPIC_API_KEY={secret}"]
    value = secret if secret_in_text else "ordinary-value"
    message = f"configuration rejected: {value}"
    hint = f"replace configuration: {value}"
    shown_value = "***" if secret_in_text else value

    file_errors = {
        "missing": FileNotFoundError,
        "not-directory": NotADirectoryError,
        "is-directory": IsADirectoryError,
        "permission": PermissionError,
    }

    def fail(bad_parameter, exit_type):
        if how == "printed":
            print_error(message, next_step=hint)
            raise exit_type(1)
        if how == "stopped":
            stopped_run_result(MissingCredentialError(message, hint=hint))
        if how in file_errors:
            raise file_errors[how](13, "cannot open", f"/tmp/{value}/input")
        if how == "usage":
            raise bad_parameter(message)
        raise MissingCredentialError(message, hint=hint)

    if backend == "click":

        @click.command()
        @click.option("--hf-token", default="")
        @click.option("--agent-env", default="")
        @click.option("--json", "output_json", is_flag=True)
        def app(hf_token, agent_env, output_json):
            fail(click.BadParameter, click.exceptions.Exit)

    else:
        app = typer.Typer()

        @app.command()
        def go(
            hf_token: str = typer.Option("", "--hf-token"),
            agent_env: str = typer.Option("", "--agent-env"),
            output_json: bool = typer.Option(False, "--json"),
        ):
            fail(typer.BadParameter, typer.Exit)

    if as_json:
        args.append("--json")
    with pytest.raises(SystemExit) as caught:
        run_cli(app, args)
    assert caught.value.code == (2 if how == "usage" else 1)
    out, err = capsys.readouterr()
    result = json.loads(out) if as_json else None
    shown = json.dumps(result) if result else err
    assert secret not in shown
    if how in file_errors:
        assert f"cannot open: /tmp/{shown_value}/input" in shown
    else:
        assert f"configuration rejected: {shown_value}" in shown
    if how in ("expected", "printed", "stopped"):
        assert f"replace configuration: {shown_value}" in shown
    if result:
        assert result["exit_code"] == caught.value.code
        assert set(result) == {"error", "code", "owner", "next", "detail", "exit_code"}


@pytest.mark.parametrize(
    "secret", ["Z", "xy", "-ab", "--", "a.b[0]*", "  spaced  ", "abc\ndef\n", ""]
)
def test_redactor_replaces_only_exact_values(secret):
    """f79b2260: credentials cannot erase unrelated text or their fragments."""
    ordinary = "network_mode='allowlist' requires a sandbox_user"
    assert _redact(ordinary, [secret]) == ordinary
    assert _redact(f"before<{secret}>after", [secret]) == (
        "before<***>after" if len(secret) >= 6 else f"before<{secret}>after"
    )
    if secret and secret.strip() != secret:
        assert _redact(secret.strip(), [secret]) == secret.strip()
    if "\n" in secret:
        assert _redact("abc and def", [secret]) == "abc and def"
    assert (
        _redact("long-prefix-value / long-prefix", ["long-prefix", "long-prefix-value"])
        == "*** / ***"
    )
