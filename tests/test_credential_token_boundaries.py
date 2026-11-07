"""Guards FF-test-env at 7a2387ac with b29's explicit placeholder cutoff."""

from __future__ import annotations

import json

import click
import pytest
import typer

from benchflow import doctor
from benchflow.cli._errors import run_cli
from benchflow.cli._shared import print_error, stopped_run_result
from benchflow.errors import MissingCredentialError
from tests.test_cli_credential_diagnostics import (
    isolated_credentials as isolated_credentials,
)


@pytest.mark.parametrize("redactor", [doctor.redact, doctor.redact_exact])
@pytest.mark.parametrize("secret", ["x", "xy", "é", "--", "a.b", "synthetic-key"])
@pytest.mark.parametrize("joiner", ["", "_", "-", ".", "+", "é", "9"])
def test_only_placeholders_inside_words_and_paths_are_preserved(
    redactor, secret, joiner
):
    """b29: short placeholders stay visible; longer values match anywhere."""
    text = (
        f"prefix{joiner}{secret} {secret}{joiner}suffix /fixture/{secret}{joiner}file"
    )
    expected = text if len(secret) < 6 else text.replace(secret, "***")
    assert redactor(text, [secret]) == expected


@pytest.mark.parametrize("redactor", [doctor.redact, doctor.redact_exact])
@pytest.mark.parametrize(
    "secret", ["x", "xy", "-", "--", "*", "a.b[0]*", "é", "synthetic-key"]
)
@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("", ""),
        ("key=", ", retry"),
        ("'", "'"),
        ('"', '"'),
        ("<", ">"),
        ("(", ")"),
        ("[", "]"),
        ("/fixture/", "/input"),
        ("\t", "\n"),
    ],
)
def test_credential_cutoff_applies_in_every_context(redactor, secret, before, after):
    """b29 updates FF-CLI-C3/C5: placeholders are ignored even alone."""
    expected = secret if len(secret) < 6 else "***"
    # The complete shell-quoted spelling is longer than its raw contents.
    if secret == "a.b[0]*" and before == after == "'":
        assert redactor(before + secret + after, [secret]) == "***"
    else:
        assert redactor(before + secret + after, [secret]) == before + expected + after


@pytest.mark.parametrize("secret", [" x ", "abc\ndef\n", "\nZ\n"])
def test_whitespace_in_the_exact_value_is_not_stripped_or_split(secret):
    """b29 retains FF-CLI-C3/C4's whole-value matching before formatting."""
    expected = secret if len(secret) < 6 else "***"
    assert (
        doctor.redact_exact("before:" + secret + "after", [secret])
        == "before:" + expected + "after"
    )
    assert doctor.redact_exact(secret.strip(), [secret]) == secret.strip()


@pytest.mark.parametrize("backend", ["click", "typer"])
@pytest.mark.parametrize("as_json", [False, True])
@pytest.mark.parametrize("secret", ["x", "xy", "-ab"])
@pytest.mark.parametrize("source", ["env", "option", "attached", "assignment"])
@pytest.mark.parametrize(
    "how", ["expected", "printed", "stopped", "usage", "file", "bug"]
)
def test_placeholder_keeps_paths_and_hints_through_every_cli_error_sink(
    backend, as_json, secret, source, how, monkeypatch, capsys, tmp_path
):
    """b29: the same short placeholder stays visible alone and inside paths."""
    path = f"/fixture/prefix{secret}suffix/file-{secret}.json"
    message = f"cannot open {path}; credential={secret}"
    hint = f"check {path}"
    args = []
    if source == "env":
        monkeypatch.setenv("ANTHROPIC_API_KEY", secret)
    elif source == "option":
        args = ["--hf-token", secret]
    elif source == "attached":
        args = [f"--hf-token={secret}"]
    else:
        args = ["--agent-env", f"ANTHROPIC_API_KEY={secret}"]

    def fail():
        if how == "printed":
            print_error(message, next_step=hint)
            raise click.exceptions.Exit(1)
        if how == "stopped":
            stopped_run_result(MissingCredentialError(message, hint=hint))
        if how == "usage":
            raise click.BadParameter(message)
        if how == "file":
            raise FileNotFoundError(2, message, path)
        if how == "bug":
            raise RuntimeError(message)
        raise MissingCredentialError(message, hint=hint)

    if backend == "click":

        @click.command()
        @click.option("--hf-token", default="")
        @click.option("--agent-env", default="")
        @click.option("--json", "output_json", is_flag=True)
        def app(hf_token, agent_env, output_json):
            fail()

    else:
        app = typer.Typer()

        @app.command()
        def go(
            hf_token: str = typer.Option("", "--hf-token"),
            agent_env: str = typer.Option("", "--agent-env"),
            output_json: bool = typer.Option(False, "--json"),
        ):
            fail()

    with pytest.raises(SystemExit) as caught:
        run_cli(app, args + (["--json"] if as_json else []))
    assert caught.value.code == (2 if how == "usage" else 1)
    out, err = capsys.readouterr()
    shown = out + err
    if as_json:
        assert json.loads(out)["exit_code"] == caught.value.code
    if how == "bug":
        shown += "".join(p.read_text() for p in (tmp_path / "logs").glob("*"))
    assert path in shown
    assert f"credential={secret}" in shown
    assert "credential=***" not in shown
    if how in ("expected", "printed", "stopped"):
        assert hint in shown
