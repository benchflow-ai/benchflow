"""Guard FF-TEST-ENV-C1 at e1b9449b using b29's literal substring contract."""

from __future__ import annotations

import json

import pytest

from benchflow import doctor, doctor_smoke
from tests.test_doctor_smoke import _job, _write_result

WHITESPACE = [
    chr(code)
    for code in (
        *range(9, 14),
        *range(28, 33),
        0x85,
        0xA0,
        0x1680,
        *range(0x2000, 0x200B),
        0x2028,
        0x2029,
        0x202F,
        0x205F,
        0x3000,
    )
]
SECRETS = ["x", "xy", "-", "--", "é", "a.b", "synthetic-secret"]


def _json_unicode(value):
    return json.dumps(value, ensure_ascii=False)


def _nested(value):
    return json.dumps(repr(value))


@pytest.mark.parametrize("render", [repr, json.dumps, _json_unicode, _nested])
@pytest.mark.parametrize("space", WHITESPACE)
@pytest.mark.parametrize("secret", SECRETS)
def test_formatted_whitespace_delimits_credentials(render, space, secret):
    """e1b9449b leaks after repr/JSON escapes, including nested SDK formatting."""
    text = "ERROR " + render(space + secret + space)
    masked = secret if len(secret) < 6 else "***"
    assert doctor.redact(text, [secret]) == "ERROR " + render(space + masked + space)


@pytest.mark.parametrize("quote", ["'", '"'])
@pytest.mark.parametrize(
    "space",
    [
        r"\t",
        r"\n",
        r"\v",
        r"\f",
        r"\r",
        r"\a",
        r"\b",
        r"\0",
        r"\11",
        r"\12",
        r"\x20",
        r"\040",
        r"\u0020",
        r"\U00000020",
        r"\N{SPACE}",
    ],
)
@pytest.mark.parametrize("secret", SECRETS)
def test_equivalent_escape_spellings_keep_the_original_encoding(quote, space, secret):
    """e1b9449b: hex, octal, Unicode and named escapes have the same boundary."""
    text = f"ERROR {quote}{space}{secret}{space}{quote}"
    masked = secret if len(secret) < 6 else "***"
    assert (
        doctor.redact(text, [secret]) == f"ERROR {quote}{space}{masked}{space}{quote}"
    )


@pytest.mark.parametrize("render", [repr, json.dumps, _json_unicode, _nested])
@pytest.mark.parametrize("secret", SECRETS)
@pytest.mark.parametrize("joiner", ["", "_", "-", ".", "+", "é", "9", "\\n"])
def test_formatted_words_and_paths_keep_only_placeholders(render, secret, joiner):
    """b29: every longer raw value is masked even next to escaped characters."""
    text = "ERROR " + render(
        f"prefix{joiner}{secret} {secret}suffix /fixture/{secret}.json"
    )
    expected = text if len(secret) < 6 else text.replace(secret, "***")
    assert doctor.redact(text, [secret]) == expected


@pytest.mark.parametrize(
    "encoded", [r"\x78", r"\170", r"\u0078", r"\U00000078", r"\N{LATIN SMALL LETTER X}"]
)
@pytest.mark.parametrize("secret", SECRETS)
@pytest.mark.parametrize("side", ["before", "after"])
def test_encoded_word_characters_do_not_affect_matching(encoded, secret, side):
    """b29: quote and boundary context cannot suppress a literal match."""
    word = encoded + secret if side == "before" else secret + encoded
    text = f'ERROR "{word}"'
    expected = text if len(secret) < 6 else text.replace(secret, "***")
    assert doctor.redact(text, [secret]) == expected


@pytest.mark.parametrize("secret", ["é", "😀", "a\nb", " x ", "a'b\"c"])
@pytest.mark.parametrize("render", [repr, json.dumps, _json_unicode, _nested])
def test_short_values_remain_placeholders_when_encoded(render, secret):
    """b29: classify before encoding, including nested repr/JSON spellings."""
    text = "ERROR " + render(secret)
    assert doctor.redact(text, [secret]) == text


@pytest.mark.parametrize(
    "broken", [r"\xZ", r"\u123", r"\UFFFFFFFF", r"\N{NO SUCH NAME}", r"\q"]
)
def test_invalid_escapes_leave_diagnostics_readable(broken):
    """e1b9449b: decoding an SDK diagnostic must be total and preserve its text."""
    text = f'ERROR "{broken} synthetic-secret"'
    assert doctor.redact(text, ["synthetic-secret"]) == f'ERROR "{broken} ***"'


@pytest.mark.parametrize("source", ["error", "verifier_error", "log"])
@pytest.mark.parametrize("space", [r"\n", r"\t", r"\u2003", r"\x85", r"\N{SPACE}"])
def test_smoke_summary_redacts_formatted_diagnostics(source, space, tmp_path):
    """e1b9449b: result and fallback-log summaries leaked an escaped ACP error."""
    job = _job(tmp_path)
    text = f'ERROR "{space}synthetic-secret"'
    if source == "log":
        job.log_path.parent.mkdir(parents=True)
        job.log_path.write_text(text + "\n")
    else:
        _write_result(job, rewards=None, **{source: text})
    outcome = doctor_smoke.read_outcome(
        job, seconds=1.0, exit_code=1, environ={"SYNTHETIC_TOKEN": "synthetic-secret"}
    )
    assert outcome.status == "error"
    assert "synthetic-secret" not in outcome.reason
    assert f'ERROR "{space}***"' in outcome.reason


def test_raw_cli_and_doctor_share_literal_matching():
    """b29 and FF-CLI-C3..C6: every shared caller ignores surrounding escapes."""
    text = r'ERROR "\nsynthetic-secret"'
    assert doctor.redact_exact(text, ["synthetic-secret"]) == r'ERROR "\n***"'


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (r'ERROR "\/x\/file"', r'ERROR "\/x\/file"'),
        (r'ERROR "\\nx"', r'ERROR "\\nx"'),
        (r"ERROR C:\next\x.json", r"ERROR C:\next\x.json"),
        (
            r'ERROR {"first": "\nx", "second": "\tx"}',
            r'ERROR {"first": "\nx", "second": "\tx"}',
        ),
        ('ERROR "pre\\\nfixx"', 'ERROR "pre\\\nfixx"'),
        ('ERROR " \\\nx "', 'ERROR " \\\nx "'),
    ],
)
def test_formatted_paths_and_literal_backslashes(text, expected):
    """e1b9449b: keep escaped separators distinct from literal path characters."""
    assert doctor.redact(text, ["x"]) == expected
