"""A credential the agent was given leaves the session-log copies in every
form, escaped or encoded (review of PR #122, I1).

The copy of the agent's session log (made while its turns run, and again at
teardown) replaces each credential value the agent was given, exactly. A log
holds such a value JSON-escaped (a value with a quote, a backslash or a
non-ASCII character, written by ``json.dumps``; twice for a JSON line inside
a string field) or %-encoded (a URL; twice: ``%25XX``), and those forms
survived the exact replacement. The lead's rule: every mask is tested with
escaped and encoded variants. The contexts are the redaction review's own
(``tests/test_redaction_escape_boundary.py``: every prefix, the literal
``\\n`` and the URL escapes included, before every suffix), in both copies.
Every value here is made up.
"""

from __future__ import annotations

import json
import re
from urllib.parse import quote, quote_plus

import pytest

from benchflow.rollout._session_log import _redact, scrub_secrets
from benchflow.rollout._session_mirror import _clean
from tests.test_redaction_escape_boundary import PREFIXES, SUFFIXES

# Shapes no pattern knows, so only the run's own values find them.
VALUES = {
    "quote and backslash": 'zq9"FAKE\\plain-credential-77',
    "slash, percent, space, plus": "zq9/FAKE%plain cred+77-x",
    "non-ASCII": "zq9-FAKE-plaîn-crédential-77",
    "plain": "zq9FAKEplaincredential77",
}


def _json(value: str, times: int, *, ascii_only: bool = True) -> str:
    for _ in range(times):
        value = json.dumps(value, ensure_ascii=ascii_only)[1:-1]
    return value


def _forms(value: str) -> dict[str, str]:
    once = quote(value, safe="")
    return {
        "as given": value,
        "JSON-escaped": _json(value, 1),
        "JSON-escaped, UTF-8 kept": _json(value, 1, ascii_only=False),
        "JSON-escaped twice": _json(value, 2),
        "%-encoded": once,
        "%-encoded, lower-case hex": once.replace("%2F", "%2f").replace("%22", "%22"),
        "%-encoded twice": quote(once, safe=""),
        "JSON-escaped, then %-encoded": quote(_json(value, 1), safe=""),
    }


def _lines(form: str) -> list[str]:
    return [
        json.dumps({"type": "user", "message": {"content": f"note{p}{form}{s}"}})
        for p in PREFIXES.values()
        for s in SUFFIXES.values()
    ]


def _survivors(text: str, value: str, form: str) -> list[str]:
    """What is left of the secret in a redacted copy: the form placed, the
    value, or a JSON-escaped form of either, in the text or in the lines
    decoded."""
    left = []
    for line in text.splitlines():
        decoded = json.loads(line)["message"]["content"]
        for piece in {form, value, _json(form, 1), _json(value, 1)}:
            if piece in line or piece in decoded:
                left.append(line)
                break
    return left


@pytest.mark.parametrize("name", list(VALUES))
def test_no_form_of_a_given_credential_survives_either_copy(tmp_path, name):
    value = VALUES[name]
    secrets = [value]
    for form_name, form in _forms(value).items():
        lines = _lines(form)
        mirrored = "\n".join(_clean(line, secrets) for line in lines)
        assert _survivors(mirrored, value, form) == [], (form_name, "mirror")
        copy = tmp_path / f"{len(form)}-{abs(hash(form_name))}.jsonl"
        copy.write_text("\n".join(lines) + "\n")
        _redact(copy, secrets)
        text = copy.read_text()
        assert _survivors(text, value, form) == [], (form_name, "teardown")
        assert text.count("[redacted]") >= len(lines), form_name


def test_text_around_a_credential_is_kept():
    value = VALUES["quote and backslash"]
    line = json.dumps({"message": {"content": f"run with {value} then stop"}})
    out = json.loads(_clean(line, [value]))
    assert out["message"]["content"] == "run with [redacted] then stop"
    assert scrub_secrets("nothing here", [value]) == "nothing here"
    assert scrub_secrets("nothing here", []) == "nothing here"


def _copies(tmp_path, carried: list[str], secrets: list[str]) -> dict[str, str]:
    """*carried* through both copies: the mirror's line by line, and the
    teardown's as a .jsonl and as a .txt file."""
    out = {"mirror": "\n".join(_clean(line, secrets) for line in carried)}
    for suffix in (".jsonl", ".txt"):
        copy = tmp_path / f"copy-{len(list(tmp_path.iterdir()))}{suffix}"
        copy.write_text("\n".join(carried) + "\n")
        _redact(copy, secrets)
        out[f"teardown {suffix}"] = copy.read_text()
    return out


def test_a_given_value_is_replaced_whole_before_the_patterns(tmp_path):
    """Review of PR #122 (F2b-R122B, L-b): the token patterns ran before the
    exact values, so a given value with a token-shaped tail kept its prefix
    (``zq9FAKEpw-***REDACTED***``) in both copies."""
    value = "zq9FAKEpw-ghp_" + "A1b2C3d4E5f6G7h8I9j0" + "K1l2M3n4O5p6Q7r8"
    carried = [
        json.dumps({"type": "user", "message": {"content": f"note {value} done"}}),
        f"plain note {value} done",
    ]
    for name, text in _copies(tmp_path, carried, [value]).items():
        assert "zq9FAKEpw" not in text and "ghp_" not in text, (name, text)
        assert "note [redacted] done" in text, (name, text)


def _percent(value: str, keep: str, space: str | None = None) -> str:
    """Percent-encoding as the specs say: ASCII letters, digits and *keep*
    stay, a space becomes *space* when given, every other UTF-8 byte is %XX
    (ECMA-262 19.2.6 for encodeURIComponent and encodeURI; the WHATWG URL
    standard's application/x-www-form-urlencoded serializer for a form body
    and URLSearchParams)."""
    out = []
    for byte in value.encode():
        char = chr(byte)
        if byte < 0x80 and (char.isalnum() or char in keep):
            out.append(char)
        elif char == " " and space is not None:
            out.append(space)
        else:
            out.append(f"%{byte:02X}")
    return "".join(out)


URL_ENCODERS = {
    "encodeURIComponent": lambda v: _percent(v, "-_.!~*'()"),
    "encodeURI": lambda v: _percent(v, "-_.!~*'();/?:@&=+$,#"),
    "URLSearchParams": lambda v: _percent(v, "*-._", space="+"),
    "quote_plus, / kept": lambda v: quote_plus(v, safe="/"),
}
# Values with what those encoders treat apart: ! * ' ( ) ~ & = + / and a space.
URL_VALUES = {
    **VALUES,
    "JavaScript's kept marks": "zq9!FAKE/pw (x)~77",
    "form marks": "zq9&FAKE=pw+x/77 y*",
}


def _encoded_forms(value: str) -> dict[str, str]:
    forms = {}
    for name, encode in URL_ENCODERS.items():
        forms[name] = encode(value)
        forms[f"{name} twice"] = encode(encode(value))
    # Another JSON writer's \u escapes, in capitals (json.dumps writes them in
    # small letters): found case-blind as %-escapes are.
    forms["JSON-escaped, capital \\u hex"] = re.sub(
        r"\\u([0-9a-f]{4})", lambda m: "\\u" + m.group(1).upper(), _json(value, 1)
    )
    return forms


@pytest.mark.parametrize("name", list(URL_VALUES))
def test_every_url_encoder_s_form_leaves_both_copies(tmp_path, name):
    """Review of PR #122 (F2b-R122B, L-c): 19 of 366 cells leaked, the
    forms of JavaScript's URL encoders, a form body's, ``quote_plus`` with
    ``/`` kept, and a capital ``\\u`` escape outside JSON."""
    value = URL_VALUES[name]
    for form_name, form in _encoded_forms(value).items():
        carried = [
            json.dumps({"type": "user", "message": {"content": f"see {form} end"}}),
            f"plain see {form} end",
        ]
        for copy, text in _copies(tmp_path, carried, [value]).items():
            decoded = [
                json.loads(line)["message"]["content"]
                for line in text.splitlines()
                if line.startswith("{")
            ]
            for piece in {form, value, _json(value, 1)}:
                assert piece not in text, (form_name, copy)
                assert all(piece not in d for d in decoded), (form_name, copy)
            assert text.count("[redacted]") == 2, (form_name, copy, text)
