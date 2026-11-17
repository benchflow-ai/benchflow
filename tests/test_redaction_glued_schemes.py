"""A scheme word glued into a value, and a URL scheme glued to a key.

Part C: the reviewer's two lows from the re-check of PR #173
(2026-11-13, RPUB). Every value here is made up.

- L1: a secret name's value whose first word ends in an Authorization
  scheme word, then a space and a token (``GH_TOKEN=abcBearer <token>``,
  ``x-api-key%3A%20a1bBearer <token>``). The carriers masked the first word
  and left the token. The scheme carriers' value now takes the glued word,
  the scheme, the space and the token as one (_GLUED_SCHEME, the same scheme
  words and prose as _SCHEME): one rule, in every carrier that reads a
  scheme.
- L2: a key family's match took a URL scheme glued right after the key
  (``sk-<key>https://u:<pw>@host``), so the URL rule, which reads the
  scheme, left the password. Each key body now stops before a scheme of the
  URL rules' own list (_URL_SCHEMES, _key_run), and the URL rule masks the
  password.
"""

from __future__ import annotations

import pytest

from benchflow.publish.redact import redact_value
from benchflow.trajectories import types
from benchflow.trajectories.types import redact_trajectory_text
from tests._timing import assert_linear, linear_seconds
from tests.test_redaction_ansi_spellings import _repeat
from tests.test_redaction_escape_boundary import LAYERS, _unchanged
from tests.test_redaction_escape_boundary import _leaks as _boundary_leaks
from tests.test_redaction_shape_matrix import _leaks as _path_leaks

MASK = "***REDACTED***"
TOKEN = "FAKEglued0123456789abcd"
PASSWORD = "FAKEurlpw0123"


def _leaked(text: str, secret: str) -> set[str]:
    return {*_boundary_leaks(text, secret), *_path_leaks(text, secret)}


PREFIXES = {
    "start": "",
    "after a word": "see ",
    "after an escaped newline": "out\\n",
    "after a colour code": "out\x1b[32m",
}

# --- L1 ----------------------------------------------------------------------

# A carrier of each kind that reads a scheme, and a separator.
L1_NAMES = [
    ("GH_TOKEN", "="),
    ("GH_TOKEN", "%3D"),
    ("x-api-key", ": "),
    ("x-api-key", "%3A%20"),
    ("OPENAI_API_KEY", "="),
    ("master_key", "="),
    ("AWS_BEARER_TOKEN_BEDROCK", "="),
    ("DB_PASS", "="),
    ("KEY_2", "="),
    ('"GH_TOKEN"', ": "),
]
SCHEMES = ["Bearer", "Token", "Basic", "ApiKey", "Bot", "Key"]


@pytest.mark.parametrize(("name", "sep"), L1_NAMES)
def test_a_scheme_glued_to_the_first_word_masks_the_token_too(name, sep):
    """RPUB's L1: the glued word and the token after it are gone in every
    layer, after every prefix, for every scheme word."""
    for scheme in SCHEMES:
        value = f"refresh{scheme} {TOKEN}"
        if name.startswith('"'):
            value = f'"{value}"'
        for where, prefix in PREFIXES.items():
            text = f"{prefix}{name}{sep}{value}"
            assert _leaked(text, TOKEN) == set(), (where, text)
            assert _leaked(text, "refresh") == set(), (where, text)


def test_the_reviewers_forms():
    """RPUB's L1, as the reviewer wrote them."""
    for text in (
        f"GH_TOKEN%3Drefresh_tokenBearer {TOKEN}",
        f"x-api-key%3A%20a1bBearer {TOKEN}",
        f"GH_TOKEN=abcBearer {TOKEN}",
        f"x-api-key: a1bBearer {TOKEN}",
    ):
        out = redact_trajectory_text(text)
        assert out.endswith(MASK) and TOKEN not in out, out
        assert out.split(MASK)[0] in text
        assert TOKEN not in redact_value({"m": text})[0]["m"]


# What stays: a scheme at the start (its word kept, the token masked), prose
# after a glued scheme word (the first word alone is the value), and a value
# of plain words with no scheme.
KEPT_AS_BEFORE = [
    (f"GH_TOKEN=Bearer {TOKEN}", f"GH_TOKEN=Bearer {MASK}"),
    (
        "password: abcBasic authentication is required",
        f"password: {MASK} authentication is required",
    ),
    ("GH_TOKEN=abcdef next words", f"GH_TOKEN={MASK} next words"),
    ("GH_TOKEN=refreshBearer", f"GH_TOKEN={MASK}"),
]


@pytest.mark.parametrize(("text", "expected"), KEPT_AS_BEFORE)
def test_a_value_with_no_glued_scheme_and_token_reads_as_before(text, expected):
    assert redact_trajectory_text(text) == expected
    assert redact_trajectory_text(expected) == expected


# --- L2 ----------------------------------------------------------------------

KEYS = {
    "sk-": "sk-FAKEkey0123456789abcd",
    "sk-ant-": "sk-ant-FAKEkey0123456789abcd",
    "sk-proj-": ("sk" + "-proj-FA" + "KEKey012" + "3456789a" + "bcd"),
    "AIzaSy": "AIzaSyFAKEkey0123456789abcdef",
    "ghp_": "ghp_FAKEkey0123456789abcdef",
    "github_pat_": "github_pat_FAKEkey0123456789abcdef",
    "hf_": "hf_FAKEkey0123456789abcdef",
    "glpat-": ("gl" + "pat-FAKE" + "key01234" + "56789abc" + "def"),
    "xoxb-": ("xo" + "xb-FAKEk" + "ey012345" + "6789"),
    "dtn_": "dtn_FAKEkey0123456789abcd",
    "sk_live_": ("sk" + "_live_FA" + "KEkey012" + "3456789a" + "bcd"),
}
URLS = [
    f"https://deploy:{PASSWORD}@db.example.com/app",
    f"postgres://app:{PASSWORD}@db.example.com:5432/x",
    f"redis://u:{PASSWORD}@cache.example.com",
    f"wss://u:{PASSWORD}@stream.example.com/ws",
]


@pytest.mark.parametrize("family", list(KEYS))
def test_a_url_glued_to_a_key_keeps_its_scheme_for_the_url_rule(family):
    """RPUB's L2: the key and the URL's password are both gone, in every
    layer, after a space, a colour code and its URL-encoded spelling; the
    host is kept."""
    key = KEYS[family]
    prefixes = {**PREFIXES, "after an encoded colour code": "out%1B%5B32m"}
    for url in URLS:
        for where, prefix in prefixes.items():
            text = f"{prefix}{key}{url}"
            assert _leaked(text, PASSWORD) == set(), (where, text)
            assert _leaked(text, key[len(family) :]) == set(), (where, text)
            assert "example.com" in redact_trajectory_text(text), text


def test_a_jwt_glued_to_a_url():
    jwt = (
        "ey"
        + "JhbGciOi"
        + "JIUzI1Ni"
        + "J9.eyJzd"
        + "WIiOiJmY"
        + "WtlIn0.F"
        + "AKEsigFA"
        + "KEsig012"
        + "3456789"
    )
    text = f"{jwt}https://u:{PASSWORD}@h.example.com"
    assert _leaked(text, PASSWORD) == set()
    assert _leaked(text, "FAKEsigFAKEsig0123456789") == set()


def test_a_key_that_merely_holds_scheme_letters_is_masked_whole():
    """L2's stop is a scheme and `://`: letters alone do not end a key."""
    for text in ("sk-FAKEkey0123456789abcdhttpsx", "sk-FAKEhttpkey0123456789abcd"):
        assert redact_trajectory_text(text) == MASK


def test_the_stop_is_the_url_rules_scheme_list():
    """RPUB's L2: one list of schemes, read by the URL rule and by every
    key body's stop."""
    url_rule = next(
        p for p, r, _ in types._REDACTION_PATTERNS if r == r"\1***REDACTED***@"
    )
    assert f"({types._URL_SCHEMES}://)" in url_rule.pattern
    stop = f"(?!{types._URL_SCHEMES}://)"
    bodies = [p.pattern for p, _, _ in types._KEY_FAMILY_PATTERNS]
    assert sum(stop in body for body in bodies) == len(bodies) - 2  # not PEM, AKIA


def test_ordinary_text_beside_the_shapes_is_kept():
    for text in (
        "see https://user@host/path",
        "the token type is Bearer of news",
        "git+ssh://git@github.com/org/repo",
    ):
        assert _unchanged(text) == dict.fromkeys(LAYERS, True), text


@pytest.mark.parametrize(
    "unit",
    [
        f"GH_TOKEN=abcBearer {TOKEN} ",
        "GH_TOKEN=" + "a" * 300 + "Bearer ",
        "GH_TOKEN=aBearer ",
        "x-api-key: a1bBearer ",
        "sk-" + "a" * 40 + "https://u:p@h ",
        "sk-" + "a" * 40 + "https",
        "AIzaSy" + "h" * 30,
        "https://",
    ],
)
def test_part_c_repeated_stays_linear(unit):
    """Part C: each repeated to 50,000 and 200,000 characters: four
    times the text takes about four times as long."""
    seconds, outs = linear_seconds(redact_trajectory_text, lambda n: _repeat(unit, n))
    assert all(TOKEN not in out for out in outs)
    assert_linear(seconds)
