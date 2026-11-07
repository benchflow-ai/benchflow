"""Guard FF-TEST-ENV-C1..C4 at 4cc9c3f1 with the b29 substring contract."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from urllib.parse import quote, quote_plus

import pytest

from benchflow import doctor, doctor_smoke
from benchflow.cli._errors import run_cli
from benchflow.errors import MissingCredentialError
from tests.test_cli_credential_diagnostics import (
    isolated_credentials as isolated_credentials,
)
from tests.test_cli_opaque_diagnostics import _app
from tests.test_doctor_smoke import _job, _write_result


@pytest.mark.parametrize("redactor", [doctor.redact, doctor.redact_exact])
@pytest.mark.parametrize(
    ("text", "secret", "expected"),
    [
        (r'ERROR "\nsynthetic-secret"', "synthetic-secret", r'ERROR "\n***"'),
        (
            "ERROR user's payload: \"can't\\nsynthetic-secret\"",
            "synthetic-secret",
            "ERROR user's payload: \"can't\\n***\"",
        ),
        (r'ERROR "alpha\nbeta"', r"alpha\nbeta", 'ERROR "***"'),
    ],
    ids=["C1", "C2", "C3"],
)
def test_review_inputs(redactor, text, secret, expected):
    """4cc9c3f1: prose and raw backslashes must not change matching."""
    assert redactor(text, [secret]) == expected


@pytest.mark.parametrize("redactor", [doctor.redact, doctor.redact_exact])
@pytest.mark.parametrize("secret", ["", "x", "xy", "--", "abcde", "é" * 5, " " * 5])
def test_placeholders_are_ignored_before_encoding(redactor, secret):
    """b29: even a long encoding of a short value remains a placeholder."""
    text = f"prefix{secret}suffix /fixture/x.json {json.dumps(secret)} {quote(secret)}"
    assert redactor(text, [secret]) == text


@pytest.mark.parametrize("redactor", [doctor.redact, doctor.redact_exact])
@pytest.mark.parametrize(
    "secret",
    [
        "abcdef",
        "--dummy",
        "a.b[0]*",
        "alpha'beta",
        'alpha"beta',
        "alpha\nbeta\t",
        r"alpha\nbeta",
        "alpha beta/+?&%",
        "é😀/東京value",
        "  padded  ",
    ],
)
@pytest.mark.parametrize(
    "spelling", ["raw", "shell", "json", "unicode", "repr", "url", "query"]
)
def test_literal_spellings_match_without_boundaries(redactor, secret, spelling):
    """b29 replaces generated spellings anywhere, without parsing diagnostics."""
    encoded = {
        "raw": secret,
        "shell": shlex.quote(secret),
        "json": json.dumps(secret)[1:-1],
        "unicode": json.dumps(secret, ensure_ascii=False)[1:-1],
        "repr": repr(secret)[1:-1],
        "url": quote(secret, safe=""),
        "query": quote_plus(secret, safe=""),
    }[spelling]
    text = f"user's prefix{encoded}suffix /fixture/{encoded}.json"
    assert redactor(text, [secret]) == "user's prefix***suffix /fixture/***.json"


@pytest.mark.parametrize("redactor", [doctor.redact, doctor.redact_exact])
def test_longest_literal_wins_without_rescanning_replacements(redactor):
    """b29: prefixes, regex punctuation, and replacement-like values are literal."""
    secrets = ["abcdef", "abcdefgh", "***abcdef", "[a-z]+", "[a-z]+long"]
    assert (
        redactor("abcdefgh ***abcdef [a-z]+long abcdef", secrets) == "*** *** *** ***"
    )


@pytest.mark.parametrize("redactor", [doctor.redact, doctor.redact_exact])
def test_no_inferred_fragments(redactor):
    """b29: only whole values and their fixed encodings enter the match set."""
    assert redactor("padded", ["  padded  "]) == "padded"
    assert redactor("alpha and beta", ["alpha\nbeta"]) == "alpha and beta"


@pytest.mark.parametrize("backend", ["click", "typer"])
@pytest.mark.parametrize("as_json", [False, True])
@pytest.mark.parametrize("source", ["env", "argv"])
@pytest.mark.parametrize("render", [shlex.quote, json.dumps, quote, quote_plus])
def test_encoded_values_through_cli_backends(
    backend, as_json, source, render, monkeypatch, capsys
):
    """b29: FF-CLI-C3..C6 sinks share the same fixed spelling set."""
    secret = "dummy'quoted/with space\n"
    encoded = render(secret)
    args = []
    if source == "env":
        monkeypatch.setenv("DUMMY_PASSWORD", secret)
    else:
        args = ["--hf-token", secret]
    if as_json:
        args.append("--json")

    def fail(value, bad_parameter):
        raise MissingCredentialError(f"rejected: {encoded}", hint=f"replace {encoded}")

    with pytest.raises(SystemExit) as caught:
        run_cli(_app(backend, fail), args)
    assert caught.value.code == 1
    out, err = capsys.readouterr()
    shown = json.loads(out)["error"] if as_json else err
    assert "***" in shown and encoded not in shown and "dummy" not in shown
    assert "rejected:" in shown


@pytest.mark.parametrize("source", ["error", "verifier_error", "log"])
@pytest.mark.parametrize(
    ("text", "secret"),
    [
        (r'ERROR "\nsynthetic-secret"', "synthetic-secret"),
        ("ERROR user's payload: \"can't\\nsynthetic-secret\"", "synthetic-secret"),
        (r'ERROR "alpha\nbeta"', r"alpha\nbeta"),
    ],
    ids=["C1", "C2", "C3"],
)
def test_review_inputs_through_smoke_summary(source, text, secret, tmp_path):
    """4cc9c3f1: agent error and fallback-log paths reach the same redactor."""
    job = _job(tmp_path)
    if source == "log":
        job.log_path.parent.mkdir(parents=True)
        job.log_path.write_text(text + "\n")
    else:
        _write_result(job, rewards=None, **{source: text})
    outcome = doctor_smoke.read_outcome(
        job, seconds=1.0, exit_code=1, environ={"DUMMY_TOKEN": secret}
    )
    assert outcome.status == "error"
    assert secret not in outcome.reason and "***" in outcome.reason


@pytest.mark.parametrize("redactor", ["redact", "redact_exact"])
@pytest.mark.parametrize("fragment", [r"\'", r"\"", "a-"])
def test_one_megabyte_malformed_quotes_finishes_promptly(redactor, fragment):
    """FF-TEST-ENV-C4 at 4cc9c3f1: timeout also bounds a broken implementation."""
    script = rf"""
from time import perf_counter
from benchflow import doctor
text = "ERROR " + {fragment!r} * 524288
start = perf_counter()
assert doctor.{redactor}(text, ["synthetic-secret"]) == text
elapsed = perf_counter() - start
print(f"{{len(text)}} bytes: {{elapsed:.6f}} seconds")
assert elapsed < 2.0, elapsed
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )
    print(result.stdout.strip())
