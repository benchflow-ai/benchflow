"""Guards FF-sw-cli-other-C1 against the incomplete fix at 47fd073b."""

import ast
import json

import pytest

from benchflow.cli._errors import run_cli
from benchflow.cli.main import app
from benchflow.doctor import redact, redact_exact, redacted_traceback


@pytest.mark.parametrize(
    "secret",
    [
        "demo's-secret-0123456789",
        'demo"s-secret-0123456789',
        "demo\\'s-秘密-0123456789",
        "demo\n's-秘密-0123456789",
    ],
)
@pytest.mark.parametrize("context", ["", "x'", 'x"', "x'\""])
@pytest.mark.parametrize("render", [repr, ascii])
def test_embedded_python_credentials(secret, context, render):
    """Guards 47fd073b across quote choices, backslashes and Unicode."""
    operand = context + secret
    rendered = render(operand)
    for redact_value in (redact_exact, redact):
        out = redact_value(rendered, [secret])
        assert "***" in out and "0123456789" not in out, out
        try:
            parsed = ast.literal_eval(out)
        except SyntaxError:
            # Integ's shared redactor (668bbccb) also masks the shell-quoted
            # spelling, which can take a quote next to the value with it: the
            # credential is still gone, only the literal no longer parses.
            continue
        assert parsed == context + "***"
    assert redacted_traceback(ValueError(rendered), [secret]).endswith(
        "ValueError: " + redact_exact(rendered, [secret])
    )


@pytest.mark.parametrize("json_mode", [False, True])
def test_explain_embedded_credential(tmp_path, monkeypatch, capsys, json_mode):
    """Guards 47fd073b at the actual CLI exception rendering boundary."""
    secret = "demo's-secret-0123456789"
    monkeypatch.setenv("D2_PASSWORD", secret)
    monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(tmp_path / "dummy.env"))
    with pytest.raises(SystemExit):
        run_cli(app, ["explain", 'x"' + secret, *(["--json"] if json_mode else [])])
    out, err = capsys.readouterr()
    output = str(json.loads(out)) if json_mode else out + err
    assert "***" in output
    assert "0123456789" not in output


def test_exact_redaction_keeps_unrelated_fragments():
    """Guards 47fd073b's exact-match contract while adding quote contexts."""
    assert redact_exact("demo and secret", ["demo's-secret"]) == "demo and secret"
