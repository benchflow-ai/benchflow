"""Secrets right after an escape, and the ordinary text beside them.

Review of PR #148, 2026-11-13. Every value here is made up.

The redaction rules with a left boundary refused a secret right after a
letter or a digit. json.dumps writes a newline as the two characters
``\\n``, and ``redact_trajectory_obj`` redacts such a copy
(``_redact_field``), so in every trajectory file and export a key after a
newline, a tab or ``\u2019`` followed the letter ``n``, ``t`` or the digit
``9`` and was kept. Text already escaped (a JSON line in a tool output)
and URL encoding (``%22``, ``%3D``) hid it in every layer, publish
included. Every rule with a left boundary now starts at the shared
``_KEY_START`` of ``benchflow.trajectories.types``.

The same review found three more classes, covered here as well:
whitespace between a name's ``:`` and its value escaped the same way
(``x-api-key:\\t<v>``), names the export redactor did not know as a dict
key (``master_key``, ``DB_PASS``, ``MYSQL_PWD``), and the task judge's own
copy of the key families (``taskmd/judging.py``), which leaked after ``_``,
an escape or a URL escape.

The layers: ``redact_trajectory_text`` (text), ``redact_trajectory_obj``
(obj), the two writers of ``acp_trajectory.jsonl`` (``acp``: the end-of-run
``redact_acp_trajectory_jsonl``; ``live``: ``RedactedJSONLSnapshot``, which
``TrajectoryWriter`` uses) and the export redactor ``redact_value``
(publish); for the key families also what the task judge is shown
(``judging.Scrub.evidence``).
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.publish.redact import VALUE_PATTERNS, redact_value
from benchflow.trajectories import types
from benchflow.trajectories._snapshot import RedactedJSONLSnapshot
from benchflow.trajectories.types import (
    redact_acp_trajectory_jsonl,
    redact_trajectory_obj,
    redact_trajectory_text,
)

MASK = "***REDACTED***"
LAYERS = ("text", "obj", "acp", "live", "publish")


def _layers(text: str) -> dict[str, str]:
    event = {"type": "agent_message", "text": text}
    return {
        "text": redact_trajectory_text(text),
        "obj": json.dumps(redact_trajectory_obj({"message": text})),
        "acp": redact_acp_trajectory_jsonl([event]),
        "live": RedactedJSONLSnapshot().serialize([event]),
        "publish": json.dumps(redact_value({"message": text})[0]),
    }


def _leaks(text: str, secret: str) -> list[str]:
    return [layer for layer, out in _layers(text).items() if secret in out]


AQ_KEY = "AQ.FAKEfake0123456789_FAKE"
SK_LIVE = "sk" + "_live_FA" + "KE012345" + "6789fake"
RK_LIVE = "rk" + "_live_FA" + "KE012345" + "6789fake"
SK_KEY = "sk-FAKE0123456789fakeFAKE"
JWT = (
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
BEARER = "FAKEbearer0123fake"
BARE = "fakebare0123fake"
SID = "FAKEsid0123fake"
PASSWORD = "FAKEpw-0123"
BASIC = "ZmFrZXVzZXI6RkFLRXBhc3MwMTIz"  # base64 of a made-up user:password
X_API_KEY = "FAKExapikey0123fake"

# A made-up secret of every shape the changed rules cover, as the text that
# carries it and the part of it that must not survive.
SECRETS = {
    "AQ. key": (AQ_KEY, AQ_KEY),
    "sk_live_ key": (SK_LIVE, SK_LIVE),
    "rk_live_ key": (RK_LIVE, RK_LIVE),
    "sk- key": (SK_KEY, SK_KEY),
    # The shapes the generic sk- rule is for: OpenAI's legacy 48 characters,
    # and the 32 hex digits other providers issue.
    "sk- key, 48 characters": (
        "sk-FAKE" + "aB3dE" * 8 + "T3BlbkFJ",
        "FAKE" + "aB3dE" * 8,
    ),
    "sk- key, 32 hex digits": ("sk-" + "0f1e2d3c" * 4, "0f1e2d3c" * 4),
    "JWT": (JWT, JWT.rsplit(".", 1)[1]),
    "DB_PASS= digits": ("DB_PASS=48213", "48213"),
    "DB_PASS= word": (f"DB_PASS={PASSWORD}", PASSWORD),
    "SMTP_PASS:": ("SMTP_PASS: 97531", "97531"),
    '"X_PWD": "..."': ('"X_PWD": "FAKEpw0123"', "FAKEpw0123"),
    "user:password@ URL": (
        ("ht" + "tps://de" + "ploy:FAK" + "Epw0123@" + "db.examp" + "le.com/app"),
        "FAKEpw0123",
    ),
    "token userinfo, lower case": (
        "https://fakefakefakefaketoken@github.com/org/repo",
        "fakefakefakefaketoken",
    ),
    "token userinfo, mixed case": (
        "https://FakeFakeFakeFakeToken@github.com/org/repo",
        "FakeFakeFakeFakeToken",
    ),
    "token userinfo, digits": (
        "https://fake0123fake4567fake@github.com/org/repo",
        "fake0123fake4567fake",
    ),
    "Authorization: Bearer": (f"Authorization: Bearer {BEARER}", BEARER),
    "Authorization: bare": (f"Authorization: {BARE}", BARE),
    "Cookie": (f"Cookie: sid={SID}", SID),
    "Set-Cookie": (f"Set-Cookie: sid={SID}; Path=/", SID),
    # A CGI or WSGI env dump: the header names after HTTP_ (lead's ruling on
    # the review of PR #148, 2026-11-13).
    "HTTP_AUTHORIZATION=Bearer": (f"HTTP_AUTHORIZATION=Bearer {BEARER}", BEARER),
    "HTTP_AUTHORIZATION=Basic": (f"HTTP_AUTHORIZATION=Basic {BASIC}", BASIC),
    "REDIRECT_HTTP_AUTHORIZATION=Bearer": (
        f"REDIRECT_HTTP_AUTHORIZATION=Bearer {BEARER}",
        BEARER,
    ),
    "HTTP_COOKIE=": (f"HTTP_COOKIE=sid={SID}; theme=dark", SID),
    "HTTP_X_API_KEY=": (f"HTTP_X_API_KEY={X_API_KEY}", X_API_KEY),
    # A name ending in `per` before TOKEN is no price per token.
    "GOOGLE_ADS_DEVELOPER_TOKEN=": (
        f"GOOGLE_ADS_DEVELOPER_TOKEN={PASSWORD}",
        PASSWORD,
    ),
}
# What comes right before the secret. The real characters become escapes in
# the JSON copy; the literal ones (backslash and a letter) are text already
# escaped, as a JSON line or a Python bytes repr in a tool output is.
PREFIXES = {
    "start of text": "",
    "space": " ",
    "_ (after a word)": "_",
    "newline": "\n",
    "tab": "\t",
    "carriage return": "\r",
    "CRLF": "\r\n",
    "U+2019": "\u2019",
    "U+00A0": "\u00a0",
    "literal \\n": "\\n",
    "literal \\t": "\\t",
    "literal \\r\\n": "\\r\\n",
    "literal \\u2019": "\\u2019",
    "literal \\xNN": "\\xe2\\x80\\x99",
    "=": "=",
    ":": ":",
    '"': '"',
    "'": "'",
    "(": "(",
    "[": "[",
    "{": "{",
    ",": ",",
    ";": ";",
    "%22": "%22",
    "%3D": "%3D",
    "%20": "%20",
    "%0A": "%0A",
    "%3A": "%3A",
    "%0D%0A": "%0D%0A",
    # An ANSI colour code, as a terminal log has it and as text escapes it.
    "ESC[32m": "\x1b[32m",
    "ESC[1;32m": "\x1b[1;32m",
    "literal \\u001b[32m": "\\u001b[32m",
    "literal \\x1b[0;31m": "\\x1b[0;31m",
    "literal \\033[38;5;208m": "\\033[38;5;208m",
    "ESC[truecolor fg+bg]": "\x1b[38;2;255;255;255;48;2;0;0;0m",
    "ESC[?25l": "\x1b[?25l",
    "ESC[?2004h": "\x1b[?2004h",
    "ESC[38:5:208m": "\x1b[38:5:208m",
    "%1B[32m": "%1B[32m",
    "literal \\u001b[truecolor fg+bg]": "\\u001b[38;2;255;255;255;48;2;0;0;0m",
    "literal \\u001b[?25l": "\\u001b[?25l",
    "literal \\x1b[38:5:208m": "\\x1b[38:5:208m",
}
SUFFIXES = {
    "end of text": "",
    "space": " done",
    "newline": "\ndone",
    ".": ".",
    ";": ";",
    ")": ")",
    "]": "]",
    '"': '"',
    "!": "!",
    "?": "?",
    ":": ":",
    ",": ",",
}
# Cells of the matrix where the secret survives a layer on purpose, as
# {(secret, prefix, suffix, layer): why}. A cell that leaks and is not here
# fails the matrix, and so does a cell here that no longer leaks. There is
# none: `note_Cookie:` and `note_Authorization:` (228 cells before) are masked
# since the lead's ruling, the value keeping prose safe instead of the name.
EXCEPTIONS: dict[tuple[str, str, str, str], str] = {}


def _cell(prefix: str, carrier: str, suffix: str) -> str:
    return ("note" + prefix if prefix else "") + carrier + suffix


@pytest.mark.parametrize("name", list(SECRETS))
def test_no_secret_survives_any_prefix_suffix_or_layer(name):
    """Review of PR #148, 2026-11-13: each secret shape, after every prefix
    and before every suffix of the matrix, is masked in all five layers;
    a cell left unmasked must be a named exception in EXCEPTIONS."""
    carrier, secret = SECRETS[name]
    leaked = {
        (name, p_name, s_name, layer)
        for p_name, prefix in PREFIXES.items()
        for s_name, suffix in SUFFIXES.items()
        for layer in _leaks(_cell(prefix, carrier, suffix), secret)
    }
    expected = {cell for cell in EXCEPTIONS if cell[0] == name}
    assert leaked == expected, sorted(leaked - expected)
    unstable = [
        (p_name, s_name)
        for p_name, prefix in PREFIXES.items()
        for s_name, suffix in SUFFIXES.items()
        if not _idempotent(_cell(prefix, carrier, suffix))
    ]
    assert unstable == []


def _idempotent(text: str) -> bool:
    """Redacting twice equals redacting once (text, object and export)."""
    once = redact_trajectory_text(text)
    obj = redact_trajectory_obj({"message": text})
    exported, _ = redact_value({"message": text})
    return (
        redact_trajectory_text(once) == once
        and redact_trajectory_obj(obj) == obj
        and redact_value(exported) == (exported, 0)
    )


def test_the_matrix_covers_every_shape_prefix_and_suffix():
    assert len(SECRETS) * len(PREFIXES) * len(SUFFIXES) == 25 * 42 * 12


# The cases the review and the follow-up leak sweep named, one by one.
REVIEW_ESCAPED = [
    ("AQ. after a newline", f"error:\n{AQ_KEY}", AQ_KEY),
    ("AQ. after a tab", f"key\t{AQ_KEY}", AQ_KEY),
    ("AQ. after U+2019", f"it\u2019{AQ_KEY}", AQ_KEY),
    ("AQ. after %22", f"q=%22{AQ_KEY}", AQ_KEY),
    ("AQ. after %3D", f"key%3D{AQ_KEY}", AQ_KEY),
    ("sk_live_ after a newline", f"error:\n{SK_LIVE}", SK_LIVE),
    ("sk_live_ after a tab", f"key\t{SK_LIVE}", SK_LIVE),
    ("sk_live_ after U+2019", f"it\u2019{SK_LIVE}", SK_LIVE),
    ("sk_live_ after %22", f"q=%22{SK_LIVE}", SK_LIVE),
    ("rk_live_ after %3D", f"key%3D{RK_LIVE}", RK_LIVE),
    ("sk- after a newline", f"error:\n{SK_KEY}", SK_KEY),
    ("sk- after %3D", f"key%3D{SK_KEY}", SK_KEY),
    (
        "Bearer after CRLF",
        f"HTTP/1.1 401\r\nAuthorization: Bearer {BEARER}",
        BEARER,
    ),
    (
        "Bearer after a literal CRLF",
        f"HTTP/1.1 401\\r\\nAuthorization: Bearer {BEARER}",
        BEARER,
    ),
    ("Bearer after %22", f"h=%22Authorization: Bearer {BEARER}", BEARER),
    ("Bearer after %3D", f"h%3DAuthorization: Bearer {BEARER}", BEARER),
    ("bare authorization after a literal \\n", f"x\\nAuthorization: {BARE}", BARE),
    ("bare authorization after %22", f"h=%22Authorization: {BARE}", BARE),
    ("bare authorization after %3D", f"h%3DAuthorization: {BARE}", BARE),
    ("cookie after a newline", f"x\nCookie: sid={SID}", SID),
    ("cookie after a literal \\n", f"x\\nCookie: sid={SID}", SID),
    ("cookie after %22", f"h=%22Cookie: sid={SID}", SID),
    ("cookie after %3D", f"h%3DCookie: sid={SID}", SID),
    ("JWT after a newline", f"token:\n{JWT}", JWT),
    ("JWT after a literal \\n", f"token:\\n{JWT}", JWT),
    ("JWT after %22", f"t=%22{JWT}", JWT),
    ("JWT after %3D", f"t%3D{JWT}", JWT),
    ("DB_PASS after a newline", f"env:\nDB_PASS={PASSWORD}", PASSWORD),
    ("DB_PASS after a literal \\n", f"env:\\nDB_PASS={PASSWORD}", PASSWORD),
]


@pytest.mark.parametrize(
    "text,secret",
    [case[1:] for case in REVIEW_ESCAPED],
    ids=[c[0] for c in REVIEW_ESCAPED],
)
def test_review_a_secret_after_an_escape_is_masked(text, secret):
    """Review of PR #148, 2026-11-13: the reviewer's cases. A key after a
    real newline, tab or ``\u2019`` was kept in the JSON-copy layers; after
    already-escaped text or ``%22`` / ``%3D`` in every layer."""
    assert _leaks(text, secret) == []


def test_review_the_trajectory_writer_masks_a_header_after_crlf(tmp_path):
    """Review of PR #148, 2026-11-13: the worst case of the sweep, an HTTP
    header dump in a tool output, through the live ``acp_trajectory.jsonl``
    writer itself."""
    from benchflow.trajectories._capture import TrajectoryWriter

    path = tmp_path / "trajectory" / "acp_trajectory.jsonl"
    output = f"HTTP/1.1 401 Unauthorized\r\nAuthorization: Bearer {BEARER}\r\n"
    TrajectoryWriter(path).write_final(
        [{"type": "tool_call", "tool_call_id": "t1", "content": output}]
    )

    written = path.read_text()
    assert BEARER not in written
    assert "Authorization: Bearer " + MASK in json.loads(written)["content"]


REVIEW_SHORT_PASSWORDS = [
    ("DB_PASS=12345!", "DB_PASS=" + MASK),
    ("DB_PASS=1234!?", "DB_PASS=" + MASK),
    ('{"DB_PASS": "9876!"}', '{"DB_PASS": "' + MASK + '"}'),
    ("SMTP_PASS=2468:", "SMTP_PASS=" + MASK),
    ("DB_PASS=12/34", "DB_PASS=" + MASK),
    # Under a password name any value is masked, a count or a word too.
    ("DB_PASS=12", "DB_PASS=" + MASK),
    ("MYSQL_PWD=7", "MYSQL_PWD=" + MASK),
    ("SMTP_PASS: true", "SMTP_PASS: " + MASK),
    ("ADMIN_PASS=3", "ADMIN_PASS=" + MASK),
    # Under a test-count name only `) ] . ;` may follow a count.
    ("TOTAL_PASS=12!", "TOTAL_PASS=" + MASK),
    ("TESTS_PASS=12:", "TESTS_PASS=" + MASK),
]


@pytest.mark.parametrize("text,expected", REVIEW_SHORT_PASSWORDS)
def test_review_a_short_password_is_masked(text, expected):
    """Review of PR #148, 2026-11-13: the count rule's trailing
    ``[)\\].;:!?]{0,3}`` and ``/\\d`` let short passwords through
    (``DB_PASS=12345!``, ``SMTP_PASS=2468:``, ``DB_PASS=12/34``). A count is
    kept only under a name that counts tests, and only `) ] . ;` follow it."""
    assert redact_trajectory_text(text) == expected
    secret = text.split("=" if "=" in text else ": ", 1)[1].strip('"}')
    assert _leaks(text, secret) == []


REVIEW_USERINFO_TOKENS = [
    "https://fakefakefakefaketoken@github.com/org/repo.git",
    "https://abcdefghijklmnop@gw.example.com/v1",
    "ssh://fakefake_fakefakefaketoken@git.example.com/repo.git",
    "wss://deploy-fakefakefakefaketoken@stream.example.com/ws",
]


@pytest.mark.parametrize("url", REVIEW_USERINFO_TOKENS)
def test_review_a_lower_case_token_as_userinfo_is_masked(url):
    """Review of PR #148, 2026-11-13: an all-lower-case token of sixteen
    characters or more as the userinfo (https://<token>@host) was kept, and
    ``_should_record_env_entry`` let it into config.json. Sixteen
    characters in a row without `-`, `.` or `_` are masked."""
    scheme, rest = url.split("://", 1)
    token, host = rest.split("@", 1)

    assert redact_trajectory_text(url) == f"{scheme}://{MASK}@{host}"
    assert _leaks(url, token) == []


def test_review_a_lower_case_token_url_stays_out_of_config_json(tmp_path):
    """Review of PR #148, 2026-11-13 (item 3): an env value holding a
    lower-case token as a URL's userinfo is not recorded in config.json's
    ``agent_env``; a URL with a plain user name still is."""
    from benchflow._utils.config_redaction import _should_record_env_entry
    from benchflow.rollout._results import _write_config
    from benchflow.skill_policy import SKILL_MODE_NO_SKILL, resolve_task_skill_policy

    token_url = "https://fakefakefakefaketoken@github.com/org/repo.git"
    user_url = "ssh://deploy-automation@git.example.com/repo.git"
    assert _should_record_env_entry("GIT_REMOTE", token_url) is False
    assert _should_record_env_entry("GIT_MIRROR", user_url) is True

    _write_config(
        tmp_path,
        task_path=Path("/tasks/foo"),
        agent="claude-agent-acp",
        model="claude-haiku-4-5-20251001",
        environment="docker",
        skill_policy=resolve_task_skill_policy(
            task_path=Path("/tasks/foo"),
            skill_mode=SKILL_MODE_NO_SKILL,
            runtime_skills_dir=None,
            declared_sandbox_skills_dir=None,
        ),
        sandbox_user=None,
        context_root=None,
        timeout=300,
        started_at=datetime(2026, 11, 13, 12, 0),
        agent_env={"GIT_REMOTE": token_url, "GIT_MIRROR": user_url},
    )

    text = (tmp_path / "config.json").read_text()
    assert "fakefakefakefaketoken" not in text
    assert json.loads(text)["agent_env"] == {"GIT_MIRROR": user_url}


# Ordinary text that must come through every layer unchanged.
KEPT = [
    "FAQ.troubleshooting-network-errors",
    "see docs/FAQ.troubleshooting-network-errors for help",
    "Read the FAQ.ProxySetupAndCertificates_v2 page",
    "IRAQ.population_statistics_by_region",
    ("mask" + "_live_FA" + "KEFAKEFA" + "KEFAKE12" + "34"),
    ("work" + "_live_FA" + "KEFAKEFA" + "KEFAKE12" + "34"),
    "TOTAL_PASS=12",
    "ALL_TESTS_PASS: true",
    "TOTAL_PASS=12.",
    "(TOTAL_PASS=12)",
    "TESTS_PASS=12/15",
    "[N_PASS=3]; ALL_TESTS_PASS: true.",
    "ssh://deploy-automation@host",
    "git+ssh://git@github.com/org/repo",
    "https://user@host/path",
    "ssh://build-service-account@ci.example.com/repo.git",
    "https://abcdefghijklmno@host/x",
    # Words that end in `sk` before `-` and twelve letters are masked as in
    # 0.9.2 (below); `task-notification` alone stays.
    "<task-notification>",
    "</task-notification>",
    "see task-notifications here",
    # Names that end in Cookie or Authorization, in prose and flags: the value
    # is no cookie pair and no token.
    "authorization_required=yes",
    "AUTHORIZATION_HEADER_NAME=X-Auth",
    "has_cookie: 1",
    "cookie: none",
    "set_cookie=false",
    "accept_cookie_banner=true",
    "cookie_policy: strict",
    # A price per token is no token, a word between included.
    "input_cost_per_token=3e-06",
    "usd_per_token: 0.000003",
    "output_cost_per_reasoning_token=1.5e-05",
    "input_cost_per_audio_token: 1e-06",
]
# Prose and switches under a code Authorization name, in KEPT until
# the lead's ruling (b), 2026-11-15 01:3xZ: every value under such a name is masked (two
# are the review of PR #148's rows; test_a_code_names_prose_is_masked).
# has_authorization: true / 1 / =true: a flag's switch or number, kept
# (the lead, 2026-11-16 10:53Z: every name that ends in `authorization` is a carrier; a has_/requires_ name's switch, number or null is kept).
CODE_NAME_PROSE = [
    "my_authorization: required",
]


def _unchanged(text: str) -> dict[str, bool]:
    event = {"type": "agent_message", "text": text}
    return {
        "text": redact_trajectory_text(text) == text,
        "obj": redact_trajectory_obj({"message": text}) == {"message": text},
        "acp": redact_acp_trajectory_jsonl([event]) == json.dumps(event),
        "live": RedactedJSONLSnapshot().serialize([event]) == json.dumps(event),
        "publish": redact_value({"message": text}) == ({"message": text}, 0),
    }


@pytest.mark.parametrize("text", KEPT)
def test_review_ordinary_text_is_kept_in_every_layer(text):
    """Review of PR #148, 2026-11-13: the narrowed rules still keep the
    text they were narrowed for (``FAQ.``, test counts, plain URL user
    names, ``mask_live_``), and the generic ``sk-`` rule now keeps
    ``/app/task-implementation``, in all five layers."""
    assert _unchanged(text) == dict.fromkeys(_unchanged(text), True)


# A plain word after an Authorization name, in KEPT until that change, the lead's ruling of 2026-11-16 08:46Z: it is
# the value, masked in every layer, as 0.9.3 masks it.
AUTHORIZATION_PROSE = [
    "Authorization: required",
    "authorization: none.",
    "Proxy-Authorization: Missing",
    "authorization: none",
]


@pytest.mark.parametrize("text", AUTHORIZATION_PROSE)
def test_a_prose_word_after_an_authorization_name_is_masked_in_every_layer(text):
    assert _unchanged(text) == dict.fromkeys(_unchanged(text), False)


KEPT_SUFFIXES = ["", " done", "\ndone", ".", ";", ")", "]", ",", '"']


# Under `_` after a word the header's own name is a code Authorization name
# (`note_authorization: none`), and every value under such a name is masked
# (the lead's ruling (b), 2026-11-15 01:3xZ), whatever follows it.
AFTER_A_WORD_CODE_NAME = {
    "authorization: none",
    "authorization: none.",
    "Authorization: required",
}


@pytest.mark.parametrize("text", KEPT)
def test_review_ordinary_text_is_kept_after_every_prefix(text):
    """Review of PR #148, 2026-11-13: an escape before ordinary text does
    not make a key of it (``\\nFAQ.``, ``%22task-implementation``,
    ``\u2019TOTAL_PASS=12``): the shared boundary only lets a secret start
    after an escape, and the letter before ``AQ.`` or ``sk-`` still ends a
    word."""
    changed = [
        (p_name, suffix, layer)
        for p_name, prefix in PREFIXES.items()
        for suffix in KEPT_SUFFIXES
        for layer, same in _unchanged(_cell(prefix, text, suffix)).items()
        if not same
    ]
    assert changed == []


# Prefixes after which the shared start boundary reads no name: the name is
# glued to the prefix's last letter or digit (`ESC[1;32mhas_authorization`), so
# prose there stays, as on the rc; a key there is masked by the other
# carriers (test_a_key_after_a_prefix_that_reads_no_name_is_masked). The
# percent prefixes (`%22`, `%3D`, `%20`, `%0A`, `%3A`, `%0D%0A`) left this set
# under (A) (lead 2026-11-15 13:18Z) with ruling (b): the decoded
# views read them as the characters they write, so the name after them is read
# and its prose masked (#192's named residual, closed).
NO_NAME_AFTER = {
    "ESC[1;32m",
    "ESC[truecolor fg+bg]",
    "ESC[?25l",
    "ESC[?2004h",
    "ESC[38:5:208m",
    "literal \\x1b[0;31m",
    "literal \\033[38;5;208m",
    "literal \\u001b[truecolor fg+bg]",
    "literal \\u001b[?25l",
    "literal \\x1b[38:5:208m",
}


@pytest.mark.parametrize("text", CODE_NAME_PROSE)
def test_a_code_names_prose_is_masked_after_every_prefix(text):
    """The lead's ruling (b), 2026-11-15 01:3xZ: every value under a code
    Authorization name is masked, prose and switches included, after every
    prefix and with every suffix, in every layer."""
    kept = [
        (p_name, suffix, layer)
        for p_name, prefix in PREFIXES.items()
        if p_name not in NO_NAME_AFTER
        for suffix in KEPT_SUFFIXES
        for layer, same in _unchanged(_cell(prefix, text, suffix)).items()
        if same
    ]
    assert kept == []
    # After a URL escape no name is read, as before. After a colour code the
    # shared boundary reads one, and every name that ends in `authorization`
    # is a carrier (the lead, 2026-11-16 10:53Z): masked there now.
    unread = [
        p_name
        for p_name in NO_NAME_AFTER
        if p_name.startswith("%")
        and not all(_unchanged(_cell(PREFIXES[p_name], text, "")).values())
    ]
    assert unread == []
    after_a_code = [
        p_name
        for p_name in NO_NAME_AFTER
        if not p_name.startswith("%")
        and any(_unchanged(_cell(PREFIXES[p_name], text, "")).values())
    ]
    assert after_a_code == []


def test_a_key_after_a_prefix_that_reads_no_name_is_masked():
    """After the prefixes in NO_NAME_AFTER a key under a code Authorization
    name is masked in every layer, in six value shapes and two separators."""
    key = "FAKEq7Xw2Lr9Tb4Nc6Vd1Hs8Kf3Mj5Pg0Yt2Rb7Qn4Wx9"
    shown = [
        (p_name, text, layers)
        for p_name in sorted(NO_NAME_AFTER)
        for value, secret in (
            (key, key),
            (f'"{key}"', key),
            (f'f("{key}")', key),
            (f"x[{key}]", key),
            (f"Basic {key}", key),
            ("hunter2FAKE99", "hunter2FAKE99"),
        )
        for text in (f"has_authorization: {value}", f"has_authorization={value}")
        if (layers := _leaks(_cell(PREFIXES[p_name], text, ""), secret))
    ]
    assert shown == []


@pytest.mark.parametrize("text", sorted(AFTER_A_WORD_CODE_NAME))
def test_the_headers_name_after_a_word_is_a_code_name(text):
    """The cells set aside above are masked in every layer (the lead's ruling (b), 2026-11-15 01:3xZ)."""
    prefix = PREFIXES["_ (after a word)"]
    for suffix in KEPT_SUFFIXES:
        assert not any(_unchanged(_cell(prefix, text, suffix)).values()), suffix


# The shared boundary itself.


def test_every_left_boundary_in_both_files_is_the_shared_one():
    """Review of PR #148, 2026-11-13: each rule wrote its own
    ``(?<![A-Za-z0-9])`` or ``\\b``, and every one forgot the escapes. A
    rule written that way again fails here."""
    sources = [pattern.pattern for pattern, _, _ in types._REDACTION_PATTERNS]
    sources += [rule.pattern.pattern for rule in VALUE_PATTERNS]
    bounded = 0
    for source in sources:
        rest = source.replace(types._KEY_START, "")
        bounded += rest != source
        assert "(?<![A-Za-z0-9" not in rest, source
        assert not re.search(r"(?:^|\|)\\b", rest), source
    # AQ./sk_live_, sk-, JWT, three name carriers, cookie, two authorization
    # rules, and the export redactor's AQ.
    assert bounded >= 10


def test_each_escape_lookbehind_is_fixed_width():
    """``re`` refuses a lookbehind whose width varies; each one compiles
    alone, so none of them does."""
    assert len(types._ESCAPES) == 4
    for escape in types._ESCAPES:
        assert re.compile(f"(?<={escape})")
    assert re.compile(types._KEY_START)


@pytest.mark.parametrize(
    "char", ["\n", "\r", "\t", "\b", "\f", "\u2019", "\u00a0", "\x0b", "\U0001f600"]
)
def test_the_boundary_reads_each_escape_json_dumps_writes(char):
    """The two-character ``\\n`` (and ``\\uXXXX``) exactly as json.dumps
    writes them, as ``_redact_field``'s copy holds them, lets a key start;
    the letter alone does not."""
    escaped = json.dumps(char + "AQ")[1:-1]
    assert escaped != char + "AQ"
    assert re.search(types._KEY_START + "AQ", escaped)
    letter = escaped[-3]
    assert not re.search(types._KEY_START + "AQ", letter + "AQ")


def test_a_key_after_a_literal_backslash_n_does_not_hide():
    """A backslash and an ``n`` typed in plain text are an escape to the
    boundary too, in the text layer and in the JSON copy (where the
    backslash itself is escaped, ``\\\\n``): the key after them is masked,
    so already-escaped text cannot hide one."""
    text = r"C:\new\nAQ.FAKEfake0123456789_FAKE"
    assert "\\\\n" + "AQ." in json.dumps(text)

    assert redact_trajectory_text(text) == "C:\\new\\n" + MASK
    assert redact_trajectory_obj({"message": text}) == {"message": "C:\\new\\n" + MASK}
    assert _leaks(text, AQ_KEY) == []


# Whitespace between a name's `:` or `=` and its value: json.dumps writes a
# tab or a newline there as `\\t` / `\\n`, which `\\s*` did not match.
GAP_VALUE = "FAKEgap0aZ9bY0aZ9bY"
GAP_CARRIERS = {
    "x-api-key:": ("x-api-key:", "{gap}" + GAP_VALUE),
    '"x-api-key":': ('"x-api-key":', '{gap}"' + GAP_VALUE + '"'),
    "apiKey:": ("apiKey:", "{gap}" + GAP_VALUE),
    "api_key=": ("api_key=", "{gap}" + GAP_VALUE),
    "private_key:": ("private_key:", "{gap}" + GAP_VALUE),
    "master_key=": ("master_key=", "{gap}" + GAP_VALUE),
    "AWS_BEARER_TOKEN_BEDROCK=": ("AWS_BEARER_TOKEN_BEDROCK=", "{gap}" + GAP_VALUE),
    "GITHUB_TOKEN=": ("GITHUB_TOKEN=", "{gap}" + GAP_VALUE),
    "DB_PASS=": ("DB_PASS=", "{gap}" + GAP_VALUE),
    "Cookie:": ("Cookie:", "{gap}sid=" + GAP_VALUE),
    "Authorization: bare": ("Authorization:", "{gap}" + GAP_VALUE),
    "Authorization:<gap>Bearer": ("Authorization:", "{gap}Bearer " + GAP_VALUE),
    "Authorization: Bearer<gap>": ("Authorization:", " Bearer{gap}" + GAP_VALUE),
}
GAPS = {
    "none": "",
    "space": " ",
    "tab": "\t",
    "newline and indent": "\n  ",
    "CRLF and a tab": "\r\n\t",
    "literal \\t": "\\t",
    "literal \\n and indent": "\\n  ",
    "literal \\r\\n and \\t": "\\r\\n\\t",
    "%20": "%20",
    "%09": "%09",
    "%0A%20%20": "%0A%20%20",
    "%0D%0A%09": "%0D%0A%09",
}


GAP_PREFIXES = {
    name: PREFIXES[name]
    for name in (
        "start of text",
        "newline",
        "U+2019",
        "literal \\n",
        '"',
        "%22",
        "%3D",
    )
}


@pytest.mark.parametrize("name", list(GAP_CARRIERS))
def test_review_whitespace_after_the_separator_hides_no_value(name):
    """Review of PR #148, 2026-11-13 (sweep, class 1): a tab or a newline
    between a header's name and its value (``x-api-key:\\t<v>``,
    ``apiKey:\\n  <v>``, ``Authorization:\\tBearer <v>``) reached the
    carrier as ``\\t`` / ``\\n`` in the JSON copy, and the value was kept.
    Every gap, escaped or URL-encoded, is whitespace (_GAP), between an
    Authorization scheme and its token too."""
    head, tail = GAP_CARRIERS[name]
    leaked = [
        (p_name, g_name, layer)
        for p_name, prefix in GAP_PREFIXES.items()
        for g_name, gap in GAPS.items()
        for layer in _leaks(
            _cell(prefix, head + tail.format(gap=gap), " end"), GAP_VALUE
        )
    ]
    assert leaked == []


# A secret under its name as a dict key.
DICT_VALUE = "FAKEdict0aZ9bY0aZ9bY"
DICT_KEY_NAMES = [
    # The three the export redactor did not know.
    "master_key",
    "DB_PASS",
    "MYSQL_PWD",
    # More of the trajectory redactor's names, as JSON keys come.
    "SMTP_PASS",
    "privateKey",
    "x-api-key",
    "X-Goog-Api-Key",
    "apiKey",
    "AWS_BEARER_TOKEN_BEDROCK",
    "GITHUB_TOKEN",
    "refreshToken",
    "client_secret",
    "DB_PASSWD",
    "GCP_CREDENTIAL",
    "encryptionKey",
    "secretkey",
    "Authorization",
    "Proxy-Authorization",
    "proxy_authorization",
]
COOKIE_KEY_NAMES = ["Cookie", "Set-Cookie", "set_cookie"]


def _dict_leaks(obj: dict, secret: str) -> list[str]:
    text = json.dumps(obj)
    out = {
        "text": redact_trajectory_text(text),
        "obj": json.dumps(redact_trajectory_obj(obj)),
        "acp": redact_acp_trajectory_jsonl([{"type": "x", **obj}]),
        "live": RedactedJSONLSnapshot().serialize([{"type": "x", **obj}]),
        "publish": json.dumps(redact_value(obj)[0]),
    }
    return [layer for layer in LAYERS if secret in out[layer]]


@pytest.mark.parametrize("name", DICT_KEY_NAMES + COOKIE_KEY_NAMES)
def test_review_a_secret_under_its_name_as_a_dict_key_is_masked(name):
    """Review of PR #148, 2026-11-13 (sweep, class 2): ``redact_value``
    decides a dict key by ``_is_sensitive_key``, which did not know
    ``master_key``, ``DB_PASS`` or ``MYSQL_PWD``. It now asks the trajectory
    redactor's own names (``is_secret_field``)."""
    value = ("sid=" if name in COOKIE_KEY_NAMES else "") + DICT_VALUE
    for obj in ({name: value}, {"headers": {name: value}}, {"env": [{name: value}]}):
        assert _dict_leaks(obj, DICT_VALUE) == [], obj


@pytest.mark.parametrize("name", ["DB_PASS", "MYSQL_PWD", "master_key", "api_key"])
def test_review_a_number_under_a_password_name_is_masked_at_export(name):
    """A numeric password under its name is masked at export too."""
    redacted, count = redact_value({"env": {name: 482139}})
    assert "482139" not in json.dumps(redacted) and count == 1


# The names the export redactor masked as a key before the review of PR #148
# (publish/redact.py DENYLISTED_KEYS and its suffixes); the shared names keep
# every one of them.
EXPORT_NAMES_BEFORE = [
    "authorization",
    "proxy_authorization",
    "x_api_key",
    "api_key",
    "apikey",
    "cookie",
    "credentials",
    "private_key",
    "set_cookie",
    "x_goog_api_key",
    "aws_bearer_token_bedrock",
    "aws_secret_access_key",
    "access_token",
    "access_key",
    "account_key",
    "credential",
    "refresh_token",
    "client_secret",
    "encryption_key",
    "password",
    "passwd",
    "secret",
    "secret_key",
    "token",
]
EXPORT_SUFFIXES_BEFORE = [
    "api_key",
    "token",
    "secret",
    "password",
    "passwd",
    "access_key",
    "secret_key",
    "account_key",
    "private_key",
    "encryption_key",
    "credential",
    "credentials",
]


def test_the_shared_names_keep_every_name_the_export_redactor_knew():
    """Review of PR #148, 2026-11-13: the export redactor's own name list is
    gone; the shared one masks each name it masked, separated, compact,
    camel-case and as an option."""
    from benchflow.publish.redact import _is_sensitive_key

    names = list(EXPORT_NAMES_BEFORE)
    for suffix in EXPORT_SUFFIXES_BEFORE:
        compact = suffix.replace("_", "")
        camel = suffix.split("_")[0] + "".join(w.title() for w in suffix.split("_")[1:])
        names += [f"my_{suffix}", f"my{compact}", f"my{camel[0].upper()}{camel[1:]}"]
        names += [f"--my-{suffix.replace('_', '-')}", suffix.upper()]
    assert [name for name in names if not _is_sensitive_key(name)] == []


KEPT_OBJECTS = [
    {"TOTAL_PASS": 12, "ALL_TESTS_PASS": True, "N_PASS": "3", "TESTS_PASS": "12/15"},
    {"first_pass": True, "BYPASS": "1", "PWD": "/app"},
    {"primary_key": "id", "sort_key": "name", "foreign_key": "user_id"},
    {"max_tokens": 1000, "tokenizer": "cl100k", "requires_authorization": True},
    {"session_cookie_name": "sid", "authorization_url": "https://h.example/oauth"},
    {
        "has_authorization": True,
        "has_cookie": 1,
        "cookie_policy": "strict",
        "accept_cookie_banner": True,
        "authorization_required": "yes",
    },
    # Token counts, budgets and prices: cost accounting reads them back.
    {
        "n_input_tokens": 1200,
        "n_output_tokens": 300,
        "n_cache_read_tokens": 900,
        "total_tokens": 1500,
        "max_tokens": 4096,
        "token_budget": 8000,
        "tokens": 12,
        "cache_read_tokens": 900,
        "usage": {"tokens": 1500, "input_tokens": 1200, "output_tokens": 300},
        "usage.tokens": 1500,
        "budget_tokens": 1024,
        "totalTokenCount": 1500,
        "input_cost_per_token": 3e-06,
        "usd_per_token": {"input": 3e-06, "output": 1.5e-05},
        "output_cost_per_reasoning_token": 1.5e-05,
        "input_cost_per_audio_token": 1e-06,
    },
]


def test_a_code_names_prose_is_masked_at_export_and_in_trajectories():
    """``my_authorization: required`` was kept (the review of PR #148);
    under the lead's ruling (b), 2026-11-15 01:3xZ every value under a code Authorization name
    is masked."""
    obj = {"my_authorization": "required"}
    assert redact_value(obj) == (
        {"my_authorization": "<XXX-benchflow-key-values-XXX>"},
        1,
    )
    assert redact_trajectory_obj(obj) == {"my_authorization": "***REDACTED***"}


@pytest.mark.parametrize("obj", KEPT_OBJECTS)
def test_review_ordinary_fields_are_kept_at_export_and_in_trajectories(obj):
    """Review of PR #148, 2026-11-13: the shared names keep a test count
    under a name that counts tests, and names that only resemble a secret's."""
    assert redact_value(obj) == (obj, 0)
    assert redact_trajectory_obj(obj) == obj


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("TOTAL_PASS", "hunter2FAKE"),
        ("TOTAL_PASS", "12345!"),
        ("TESTS_PASS", "482139"),
        ("TOTAL_PASS", "12/34x"),
    ],
)
def test_review_a_password_under_a_test_count_name_is_still_masked(name, value):
    """A name that counts tests keeps only a count or a switch word.

    Changed by the lead's ruling of 2026-11-14 (review of PR #173): after a
    pure count word (n, num, total, count) any integer is a count, so
    ``TOTAL_PASS=482139`` is kept; this test asserted it was masked. The
    five-digit limit stays for the outcome names, where six digits can be a
    PIN, so the case moved to ``TESTS_PASS``. The other three values are as
    they were."""
    redacted, count = redact_value({name: value})
    assert redacted == {name: "<XXX-benchflow-key-values-XXX>"} and count == 1
    if name == "TESTS_PASS":
        assert redact_value({"TOTAL_PASS": value}) == ({"TOTAL_PASS": value}, 0)


@pytest.mark.parametrize("name", ["MYSQL_PASSWD", "GCP_CREDENTIAL", "master_key"])
def test_review_config_json_drops_every_shared_secret_name(name):
    """Review of PR #148, 2026-11-13: config.json's env filter has a wider
    list of its own, and missed ``*_PASSWD`` and ``*_CREDENTIAL`` names; it
    now drops every name the trajectory redactor masks a value under."""
    from benchflow._utils.config_redaction import _should_record_env_entry

    assert _should_record_env_entry(name, "FAKEhunter2") is False


# The task judge's evidence: only the key families, by the trajectory
# redactor's own rules (the judge must see `password=hunter2secret` as the
# submission printed it).
JUDGE_FAMILIES = {
    "AQ.": AQ_KEY,
    "sk_live_": SK_LIVE,
    "sk-": SK_KEY,
    "sk-ant-": ("sk" + "-ant-api" + "03-FAKE0" + "12345678" + "9fakeFAK" + "E"),
    "sk-proj-": ("sk" + "-proj-FA" + "KE012345" + "6789fake" + "FAKE"),
    "ya29.": ("ya" + "29.a0FAK" + "EFAKEFAK" + "EFAKEFAK" + "E1234"),
    "glpat-": ("gl" + "pat-FAKE" + "FAKEFAKE" + "FAKEFAKE" + "12"),
    "ghp_": "ghp_FAKE0123456789fakeFAKE0123",
    "AIzaSy": "AIzaSyFAKE0123456789fakeFAKE01",
    "xoxb-": ("xo" + "xb-1234-" + "FAKE0123" + "456789"),
    "hf_": "hf_FAKE0123456789fakeFAKE01",
    "JWT": JWT,
}


@pytest.mark.parametrize("name", list(JUDGE_FAMILIES))
def test_review_the_judge_never_sees_a_key_family(name):
    """Review of PR #148, 2026-11-13 (sweep, class 3): ``taskmd/judging.py``
    kept its own copy of the key families, which missed ``sk-``, ``ya29.``,
    ``AQ.``, ``glpat-``, ``sk_live_``, JWTs and PEM blocks and leaked every
    family after ``_``, an escape, ``%22``, ``%3D`` or ``%20``. The judge's
    evidence now uses the trajectory redactor's key families."""
    from benchflow.taskmd import judging

    scrub = judging.Scrub.for_run(None, {})
    key = JUDGE_FAMILIES[name]
    secret = key.rsplit(".", 1)[1] if name == "JWT" else key
    leaked = [
        (p_name, s_name)
        for p_name, prefix in PREFIXES.items()
        # The judge's evidence starts a key only where an identifier starts
        # (the lead's ruling): `note_<key>` reads as one identifier.
        if p_name != "_ (after a word)"
        for s_name, suffix in SUFFIXES.items()
        if secret in scrub.evidence(_cell(prefix, key, suffix))
    ]
    assert leaked == []


def test_review_the_judge_sees_a_pem_block_masked():
    from benchflow.taskmd import judging

    body = "MIIEvFAKE" + "QwErTy12" * 6
    pem = "-----BEGIN " + f"PRIVATE KEY-----\n{body}\n-----END PRIVATE KEY-----"
    assert body not in judging.Scrub.for_run(None, {}).evidence(f"cat key.pem\n{pem}")


# The lead's rulings on the review of PR #148, 2026-11-13.

ENV_DUMPS = {
    "HTTP_AUTHORIZATION": (f"Bearer {BEARER}", BEARER),
    "HTTP_AUTHORIZATION (Basic)": (f"Basic {BASIC}", BASIC),
    "REDIRECT_HTTP_AUTHORIZATION": (f"Bearer {BEARER}", BEARER),
    "HTTP_COOKIE": (f"sid={SID}; theme=dark", SID),
    "http_cookie": (f"sid={SID}", SID),
    "HTTP_X_API_KEY": (X_API_KEY, X_API_KEY),
}


@pytest.mark.parametrize("label", list(ENV_DUMPS))
def test_review_a_cgi_env_dump_hides_no_header(label):
    """Lead's ruling, review of PR #148: a CGI or WSGI env dump names the
    headers after HTTP_ (`HTTP_AUTHORIZATION=Bearer <t>`, `HTTP_COOKIE=sid=<v>`).
    Each is masked as a `KEY=value` line among other env lines, as a JSON dict
    entry, and inside the repr of a dict, in all five layers."""
    name = label.split(" ")[0]
    value, secret = ENV_DUMPS[label]
    env = {"PATH": "/usr/bin", "HOME": "/root", name: value, "LANG": "C.UTF-8"}
    lines = "\n".join(f"{key}={val}" for key, val in env.items())
    assert _leaks(lines, secret) == [], "lines"
    assert _leaks(repr(env), secret) == [], "repr"
    assert _dict_leaks(env, secret) == [], "dict"
    assert _dict_leaks({"environ": env}, secret) == [], "nested dict"


SECRET_NUMBERS = {
    "DB_PASS": 48213907,
    "MYSQL_PWD": 7391,
    "api_key": 482139,
    "GITHUB_TOKEN": 12345,
    "password": 1234,
    "x-api-key": 99999999,
    "masterKey": 271828,
    "aws_credentials": 12345,
    "HTTP_AUTHORIZATION": 482139,
    "refresh_token": -48213,
}


@pytest.mark.parametrize("name", list(SECRET_NUMBERS))
def test_review_a_number_under_a_secret_name_is_masked_in_trajectory_files(name):
    """Lead's ruling, review of PR #148: a number under a secret's name (a
    numeric PIN) was kept in trajectory files, where the export redactor masks
    it. ``redact_trajectory_obj`` masks it the same way (``is_secret_field``),
    in both acp_trajectory.jsonl writers too."""
    number = SECRET_NUMBERS[name]
    obj = {"env": {name: number}}
    assert redact_trajectory_obj(obj) == {"env": {name: MASK}}
    assert redact_value(obj)[0] == {"env": {name: "<XXX-benchflow-key-values-XXX>"}}
    assert _dict_leaks(obj, str(number)) == []


# The lead's examples of settings, flags and counts whose names end in a
# secret's word: each value is kept as it is, type included.
KEPT_SCALARS = [
    {"add_bos_token": True},
    {"eos_token": 2},
    {"eos_token": 50256},
    {"bos_token": 151643},
    {"pad_token": 0},
    {"eos_token_ids": [50256, 50257]},
    {"use_auth_token": False},
    {"time_to_first_token": 0.42},
    {"max_token": 4096},
    {"has_password": True},
    {"is_secret": False},
    {"withCredentials": True},
    {"num_credentials": 3},
    {"num_credentials": 1200},
    {"num_credentials": 12345},
    {"use_token": True},
    # A three-digit PIN under a credential's name: kept, by the four-digit rule.
    {"password": 123},
    # A float is never masked, a PIN-like one included.
    {"DB_PASS": 48213907.0},
    # Null holds nothing, under any name.
    {"refresh_token": None},
    {"api_key": None},
    {"password": None},
    # Booleans under a credential's own name.
    {"api_key": True, "set_cookie": False, "password": False},
]


@pytest.mark.parametrize("obj", KEPT_SCALARS, ids=lambda obj: ",".join(obj))
def test_review_a_flag_count_or_null_keeps_its_value_and_type(obj):
    """Lead's ruling, review of PR #148: the number rule fired on any name
    ending in a secret's word and turned flags, tokenizer token ids, counts
    and floats into strings. A boolean, a float or null is never masked; an
    integer only when it has four digits or more and the name is a
    credential's (``name_kind``). Kept in both redactors, nested, and in
    both acp_trajectory.jsonl writers."""
    for value in (obj, {"config": obj}, {"rows": [obj]}):
        assert redact_trajectory_obj(value) == value
        assert redact_value(value) == (value, 0)
        line = json.dumps({"type": "x", **value})
        assert redact_acp_trajectory_jsonl([{"type": "x", **value}]) == line
        assert RedactedJSONLSnapshot().serialize([{"type": "x", **value}]) == line


def test_review_numbers_in_a_list_decide_one_by_one():
    """A list of integers under a field's name: each PIN-like integer under
    a credential's name is masked, the others kept, in both redactors."""
    obj = {"api_key": [12345, 7, 67890, True, None, 1.5], "eos_token_ids": [50256]}
    mask = "<XXX-benchflow-key-values-XXX>"
    assert redact_trajectory_obj(obj) == {
        "api_key": [MASK, 7, MASK, True, None, 1.5],
        "eos_token_ids": [50256],
    }
    assert redact_value(obj) == (
        {"api_key": [mask, 7, mask, True, None, 1.5], "eos_token_ids": [50256]},
        2,
    )


@pytest.mark.parametrize(
    "name",
    [
        "n_input_tokens",
        "n_output_tokens",
        "total_tokens",
        "max_tokens",
        "token_budget",
        "tokens",
        "cache_read_tokens",
        "usage.tokens",
        "input_cost_per_token",
        "usd_per_token",
        "output_cost_per_reasoning_token",
        "input_cost_per_audio_token",
    ],
)
def test_review_token_counts_and_prices_are_no_secret_names(name):
    """Lead's ruling, review of PR #148: masking a token count or a price in
    a trajectory or an export would break cost accounting. None of these is a
    secret's name, so a number under it stays, in both redactors."""
    from benchflow.trajectories.types import is_secret_field

    assert is_secret_field(name, 1500) is False
    for obj in ({name: 1500}, {"usage": {name: 1500}}, {name: 3e-06}):
        assert redact_trajectory_obj(obj) == obj
        assert redact_value(obj) == (obj, 0)


# A hosted run's records and its raw evidence.
HOSTED_ARG_KEY = "FAKEhostedArgKey0123"
HOSTED_ARG_SHAPE = "sk-FAKEhostedShape0123456"
HOSTED_URL_PW = "FAKEhostedUrlPw0123"
HOSTED_LOG_KEY = "sk" + "-ant-api" + "03-FAKEh" + "ostedLog" + "Key01234" + "56789"
# What vf-eval itself writes raw: a key in a tool output of a completion, and
# a leftover venv's file (a run from before vf-results/ moved).
HOSTED_RAW_KEY = "sk" + "-ant-api" + "03-FAKEr" + "awComple" + "tion0123" + "456789"
HOSTED_VENV_KEY = "FAKEvenvLeftover0123456789"
# The sampling args (review of PR #148, L3): a key under its name, two
# headers, and a key's shape under a harmless name; the rest is kept.
SAMPLING_KEY = "FAKEsamplingKey0123"
SAMPLING_HEADER_KEY = "FAKEsamplingHeader0123"
SAMPLING_BEARER = "FAKEsamplingBearer0123456789"
SAMPLING_SHAPE = "sk-FAKEsamplingShape0123456"
SAMPLING_KEPT = {"reasoning_effort": "high", "max_tokens": 512}
HOSTED_SECRETS = (
    HOSTED_ARG_KEY,
    HOSTED_ARG_SHAPE,
    HOSTED_URL_PW,
    HOSTED_LOG_KEY,
    HOSTED_RAW_KEY,
    HOSTED_VENV_KEY,
    SAMPLING_KEY,
    SAMPLING_HEADER_KEY,
    SAMPLING_BEARER,
    SAMPLING_SHAPE,
)
RAW_FOLDERS = {"hosted_env", "vf-results", ".venv"}


def _fake_vf_eval(monkeypatch) -> None:
    """vf-eval as verifiers writes its output: results.jsonl and metadata.json
    (legacy), benchflow/traces.jsonl and the resolved config (v1), each raw."""
    from types import SimpleNamespace

    def fake_run(cmd, **kwargs):
        if not str(cmd[0]).endswith("vf-eval"):
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        out = Path(cmd[cmd.index("--output-dir") + 1])
        (out / "benchflow" / "configs" / "resolved").mkdir(parents=True)
        completion = f"the tool printed {HOSTED_RAW_KEY}"
        (out / "results.jsonl").write_text(
            json.dumps(
                {
                    "prompt": [{"role": "user", "content": "schedule a meeting"}],
                    "completion": [{"role": "assistant", "content": completion}],
                    "reward": 1.0,
                }
            )
            + "\n"
        )
        (out / "metadata.json").write_text(
            json.dumps({"env_args": {"api_key": HOSTED_ARG_KEY}})
        )
        (out / "benchflow" / "traces.jsonl").write_text(
            json.dumps(
                {
                    "task": {"data": {"prompt": "schedule a meeting"}},
                    "ok": True,
                    "traces": [
                        {"rewards": {"r": {"score": 1.0}}, "completion": completion}
                    ],
                }
            )
            + "\n"
        )
        (out / "benchflow" / "configs" / "resolved" / "run.json").write_text(
            json.dumps(
                {
                    "client": {
                        "base_url": f"https://deploy:{HOSTED_URL_PW}@proxy.example/v1"
                    },
                    "env": {"taskset": {"api_key": HOSTED_ARG_KEY}},
                }
            )
        )
        return SimpleNamespace(
            returncode=1,
            stdout=f"vf-eval started with {HOSTED_LOG_KEY}\nreward: avg - 1.0\n",
            stderr=f"warning: key {HOSTED_LOG_KEY} rejected\n",
        )

    monkeypatch.setattr("benchflow.hosted_env.shutil.which", lambda b: f"/bin/{b}")
    monkeypatch.setattr("benchflow.hosted_env.subprocess.run", fake_run)


def _hosted_run(tmp_path, monkeypatch, cli: str, layout: str = "new"):
    """A hosted run with vf-eval's raw output; ``layout="old"`` puts it where
    runs written before the move have it (``vf-results/`` and ``.venv/``
    beside ``hosted_env/``)."""
    from benchflow.hosted_env import HostedEnvRef, HostedEnvRunConfig, run_hosted_env

    _fake_vf_eval(monkeypatch)
    monkeypatch.setattr(
        "benchflow.hosted_env.vf_eval_cli", lambda _venv: (cli, "0.1.0")
    )
    result = run_hosted_env(
        HostedEnvRunConfig(
            source_env=HostedEnvRef.parse(
                "primeintellect/general-agent", version="0.1.1"
            ),
            model="gemini-3.1-flash-lite-preview",
            env_args={
                "task": "calendar_scheduling_t0",
                "api_key": HOSTED_ARG_KEY,
                "note": HOSTED_ARG_SHAPE,
            },
            sampling_args={
                "api_key": SAMPLING_KEY,
                "extra_headers": {
                    "x-api-key": SAMPLING_HEADER_KEY,
                    "Authorization": f"Bearer {SAMPLING_BEARER}",
                },
                "note": SAMPLING_SHAPE,
                **SAMPLING_KEPT,
            },
            api_base_url=f"https://deploy:{HOSTED_URL_PW}@proxy.example/v1",
            agent="gemini",
            jobs_dir=tmp_path / "jobs",
            num_examples=2,
        )
    )
    raw = result.run_dir / "hosted_env" / "vf-results"
    assert (raw / "benchflow" / "configs" / "resolved" / "run.json").is_file()
    if layout == "old":
        raw.rename(result.run_dir / "vf-results")
        (result.run_dir / ".venv").mkdir(exist_ok=True)
        (result.run_dir / ".venv" / "pyvenv.cfg").write_text(
            f"key = {HOSTED_VENV_KEY}\n"
        )
    return result


@pytest.mark.parametrize("cli", ["legacy", "v1"])
def test_review_a_hosted_run_records_no_env_arg_or_command_secret(
    tmp_path, monkeypatch, cli
):
    """Lead's ruling, review of PR #148: hosted_run.json kept ``env_args`` and
    ``command`` raw, and result.json's and config.json's ``env_args`` too. An
    env arg under a secret's name, one of a secret's shape under a harmless
    name and a password in the command's base URL appear in none of them; the
    raw logs keep vf-eval's own output.

    Review of PR #148, 2026-11-13, L3: config.json wrote ``sampling_args``
    raw. A key under its name, ``x-api-key`` and ``Authorization: Bearer`` in
    ``extra_headers`` and a key's shape under a harmless name are in none of
    the three records either; ``reasoning_effort`` and ``max_tokens`` are
    kept as given."""
    result = _hosted_run(tmp_path, monkeypatch, cli)
    hosted = json.loads((result.run_dir / "hosted_env" / "hosted_run.json").read_text())

    assert hosted["env_args"]["task"] == "calendar_scheduling_t0"
    assert any("proxy.example" in word for word in hosted["command"])
    for name in ("result.json", "config.json", "hosted_env/hosted_run.json"):
        text = (result.run_dir / name).read_text()
        assert [s for s in HOSTED_SECRETS if s in text] == [], name
    config = json.loads((result.run_dir / "config.json").read_text())
    sampling = config["hosted_env"]["sampling_args"]
    assert {key: sampling[key] for key in SAMPLING_KEPT} == SAMPLING_KEPT
    assert HOSTED_LOG_KEY in (result.run_dir / "hosted_env" / "stderr.log").read_text()


def _files_under(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _no_raw_evidence(files: dict[str, bytes]) -> None:
    assert files, "nothing was staged"
    assert [name for name in files if RAW_FOLDERS & set(Path(name).parts)] == []
    leaked = [
        n for n, data in files.items() for s in HOSTED_SECRETS if s.encode() in data
    ]
    assert leaked == []


LAYOUTS = ["new", "old"]


@pytest.mark.parametrize("layout", LAYOUTS)
def test_review_traj_upload_stages_nothing_of_hosted_env(tmp_path, monkeypatch, layout):
    """Lead's ruling, review of PR #148: ``bench traj upload`` stages only the
    trajectory's .jsonl files (and masked run metadata); the direct Azure and
    broker uploads send exactly the staged files. No raw vf-eval output is
    among them, where it is now or where older runs have it."""
    from benchflow.publish.traj_capture import stage_trajectory_artifacts

    run_dir = _hosted_run(tmp_path, monkeypatch, "legacy", layout).run_dir
    with stage_trajectory_artifacts(run_dir, source_id="demo") as artifacts:
        staged = {f.relname: f.local_path.read_bytes() for f in artifacts.files}
    _no_raw_evidence(staged)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_review_traj_upload_workspace_zip_leaves_raw_vf_eval_output_out(
    tmp_path, monkeypatch, layout
):
    """Review of PR #148, 2026-11-13, L1: ``bench traj upload --workspace-dir``
    zips the folder it is given as it stands, so a workspace holding a hosted
    run took ``hosted_env/`` (vf-eval's stdout and stderr, and ``vf-results/``
    where it is now) and an older run's ``vf-results/`` beside it. Neither is
    in the zip; an ordinary workspace file is. Since #369 the archive leaves
    out every trial or job folder inside the workspace, so nothing of the
    run's folder is in it at all."""
    import zipfile

    from benchflow.publish.traj_capture import (
        attach_workspace_archive,
        stage_trajectory_artifacts,
    )

    run_dir = _hosted_run(tmp_path, monkeypatch, "legacy", layout).run_dir
    (tmp_path / "notes.md").write_text("an ordinary workspace file\n")
    with stage_trajectory_artifacts(run_dir, source_id="demo") as artifacts:
        attach = attach_workspace_archive(artifacts, tmp_path)
        assert attach.attached is not None, attach.skipped_reason
        with zipfile.ZipFile(attach.attached.local_path) as archive:
            files = {name: archive.read(name) for name in archive.namelist()}
    assert files["notes.md"] == b"an ordinary workspace file\n"
    assert not any(run_dir.name in name for name in files)
    _no_raw_evidence(files)


# Review of PR #148, 2026-11-13: a raw folder named itself, or a folder inside
# one, where a command takes a folder to upload, publish, pack or show to a
# reviewer. Each path refuses it with the same reason; a folder that is only
# called vf-results, with no hosted_env/ beside it, is no raw evidence (N2).
NAMED_RAW = [
    ("new", "hosted_env"),
    ("new", "hosted_env/vf-results/benchflow"),
    ("old", "vf-results"),
    ("old", ".venv/lib"),
]
NAMED_RAW_IDS = [f"{layout}-{relative}" for layout, relative in NAMED_RAW]


def _named_raw(tmp_path, monkeypatch, layout: str, relative: str) -> Path:
    run_dir = _hosted_run(tmp_path, monkeypatch, "legacy", layout).run_dir
    (run_dir / relative).mkdir(parents=True, exist_ok=True)
    return run_dir / relative


def _plain_vf_results(tmp_path) -> Path:
    """A trial folder that is only called vf-results."""
    folder = tmp_path / "plain" / "vf-results"
    (folder / "trajectory").mkdir(parents=True)
    (folder / "trajectory" / "acp_trajectory.jsonl").write_text(
        '{"type":"message","text":"demo"}\n'
    )
    (folder / "result.json").write_text('{"task_name": "demo"}')
    return folder


def _refused(text: str, folder: Path) -> bool:
    flat = "".join(text.split())
    return all(
        "".join(part.split()) in flat
        for part in (str(folder), "raw vf-eval output", "never uploaded")
    )


def _via_traj_path(folder, tmp_path, monkeypatch, sent):
    from benchflow.publish.traj_capture import stage_trajectory_artifacts

    with stage_trajectory_artifacts(folder, source_id="demo") as artifacts:
        sent += [f.relname for f in artifacts.files]


def _via_traj_workspace(folder, tmp_path, monkeypatch, sent):
    from benchflow.publish.traj_capture import (
        attach_workspace_archive,
        stage_trajectory_artifacts,
    )

    trial = tmp_path / "trial"
    (trial / "trajectory").mkdir(parents=True)
    (trial / "trajectory" / "acp_trajectory.jsonl").write_text('{"type":"x"}\n')
    with stage_trajectory_artifacts(trial, source_id="demo") as artifacts:
        attached = attach_workspace_archive(artifacts, folder).attached
        sent += [attached.relname] if attached is not None else []


class _RecordingHfApi:
    def __init__(self, uploaded: dict[str, bytes]) -> None:
        self.uploaded = uploaded

    def create_repo(self, *args, **kwargs):
        self.uploaded["<repo created>"] = b""

    def upload_folder(self, *, folder_path, **kwargs):
        self.uploaded.update(_files_under(Path(folder_path)))


def _via_hf(folder, tmp_path, monkeypatch, sent):
    from benchflow.publish import huggingface

    uploaded: dict[str, bytes] = {}
    monkeypatch.setattr(
        huggingface, "_require_hf_api", lambda: _RecordingHfApi(uploaded)
    )
    try:
        huggingface.publish_folder_to_hf(folder, repo_id="org/runs", path_in_repo="x")
    finally:
        sent += list(uploaded)


def _via_bucket(folder, tmp_path, monkeypatch, sent):
    import sys
    import types as pytypes

    from benchflow.publish import huggingface

    hub = pytypes.ModuleType("huggingface_hub")
    errors = pytypes.ModuleType("huggingface_hub.errors")
    errors.HfHubHTTPError = type("HfHubHTTPError", (Exception,), {})
    hub.errors = errors
    hub.create_bucket = lambda *args, **kwargs: sent.append("<bucket created>")
    hub.sync_bucket = lambda source, remote: sent.extend(_files_under(Path(source)))
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)
    huggingface.publish_folder_to_bucket(folder, bucket_id="org/runs")


def _via_canonical(folder, tmp_path, monkeypatch, sent):
    from benchflow.eval_artifacts import materialize_canonical_job

    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"selected": [{"rollout_dir": str(folder)}]}))
    out = tmp_path / "canonical"
    try:
        materialize_canonical_job(selection, out)
    finally:
        sent += list(_files_under(out)) if out.is_dir() else []


def _via_bundle(folder, tmp_path, monkeypatch, sent):
    from benchflow.evidence_bundle import build_bundle

    out = tmp_path / "bundle.zip"
    try:
        build_bundle(folder, out)
    finally:
        sent += [out.name] if out.exists() else []


def _via_review_discovery(folder, tmp_path, monkeypatch, sent):
    from benchflow.review.runner import discover_rollouts

    sent += [str(path) for path in discover_rollouts(folder)]


def _via_review_copy(folder, tmp_path, monkeypatch, sent):
    from benchflow.review.wrapper import copy_evidence

    destination = tmp_path / "evidence" / "trial"
    try:
        copy_evidence(folder, destination)
    finally:
        sent += list(_files_under(destination)) if destination.is_dir() else []


NAMED_FOLDER_PATHS = {
    "traj-upload-path": _via_traj_path,
    "traj-upload-workspace": _via_traj_workspace,
    "publish-hf": _via_hf,
    "publish-bucket": _via_bucket,
    "canonical-jobs": _via_canonical,
    "bundle": _via_bundle,
    "review-discovery": _via_review_discovery,
    "review-copy": _via_review_copy,
}


@pytest.mark.parametrize("path", list(NAMED_FOLDER_PATHS))
@pytest.mark.parametrize(("layout", "relative"), NAMED_RAW, ids=NAMED_RAW_IDS)
def test_review_a_named_raw_folder_is_refused_by_every_path(
    tmp_path, monkeypatch, path, layout, relative
):
    """Review of PR #148, 2026-11-13: every path asked ``is_raw_evidence`` of
    the entries inside the folder it was given, never of the folder itself,
    so naming ``hosted_env/`` (or a folder inside it, or an older run's
    ``vf-results/`` or ``.venv/``) sent vf-eval's raw output whole. Each path
    now refuses such a folder before anything is staged, copied or sent."""
    from benchflow.evidence_bundle import BundleError

    folder = _named_raw(tmp_path, monkeypatch, layout, relative)
    sent: list[str] = []
    with pytest.raises((ValueError, BundleError)) as refused:
        NAMED_FOLDER_PATHS[path](folder, tmp_path, monkeypatch, sent)
    assert _refused(str(refused.value), folder), str(refused.value)
    assert sent == []


@pytest.mark.parametrize("path", list(NAMED_FOLDER_PATHS))
def test_review_a_folder_only_called_vf_results_is_accepted(
    tmp_path, monkeypatch, path
):
    """Review of PR #148, 2026-11-13: the refusal reads the folder's place,
    not its name; a vf-results/ with no hosted_env/ beside it goes through."""
    sent: list[str] = []
    NAMED_FOLDER_PATHS[path](_plain_vf_results(tmp_path), tmp_path, monkeypatch, sent)
    assert sent


@pytest.mark.parametrize(("layout", "relative"), NAMED_RAW, ids=NAMED_RAW_IDS)
def test_review_regrade_refuses_a_named_raw_folder(
    tmp_path, monkeypatch, layout, relative
):
    """Review of PR #148, 2026-11-13: a regrade copies a trial into its review
    view and the verifier's sandbox; it refuses a raw folder before it looks
    for trials. A folder only called vf-results gets past that check (and
    fails later here only for having no frozen workspace)."""
    from benchflow.eval_regrade import regrade

    def runner(*args, **kwargs):
        raise AssertionError("no verifier runs")

    folder = _named_raw(tmp_path, monkeypatch, layout, relative)
    with pytest.raises(ValueError) as refused:
        regrade(folder, runner=runner)
    assert _refused(str(refused.value), folder)
    plain = _plain_vf_results(tmp_path)
    try:
        regrade(plain, runner=runner)
    except Exception as exc:
        assert not _refused(str(exc), plain), exc


@pytest.mark.parametrize(("layout", "relative"), NAMED_RAW, ids=NAMED_RAW_IDS)
def test_review_job_export_cannot_load_a_raw_folder(
    tmp_path, monkeypatch, layout, relative
):
    """Review of PR #148, 2026-11-13: the job export reads a trial's
    result.json and named files; no raw folder holds a result.json, so none
    loads as a trial."""
    from benchflow.jobs import load_trial

    with pytest.raises(FileNotFoundError):
        load_trial(_named_raw(tmp_path, monkeypatch, layout, relative))


def _traj_upload_cli(monkeypatch, *args: str):
    from typer.testing import CliRunner

    from benchflow.cli import traj
    from benchflow.cli.main import app

    published: list[object] = []
    staged: list[object] = []
    stage = traj.stage_trajectory_artifacts

    def recording_stage(path, **kwargs):
        staged.append(path)
        return stage(path, **kwargs)

    monkeypatch.setattr(traj, "stage_trajectory_artifacts", recording_stage)

    def publish(staged_capture, **_):
        published.append(staged_capture)
        return SimpleNamespace(
            url="https://upload.example/x",
            uploaded=("x",),
            skipped=(),
            already_ingested=True,
        )

    monkeypatch.setattr(traj, "_publish", publish)
    result = CliRunner().invoke(
        app,
        [
            "traj",
            "upload",
            *args,
            "--source-id",
            "demo",
            "--github-id",
            "benchflow-user",
            "--email",
            "user@example.com",
        ],
    )
    return result, staged, published


DESTINATIONS = {
    "broker": (),
    "direct": ("--direct", "--container-url", "https://acct.blob.core.windows.net/c"),
}


@pytest.mark.parametrize("destination", list(DESTINATIONS))
@pytest.mark.parametrize("given", ["path", "workspace"])
@pytest.mark.parametrize(("layout", "relative"), NAMED_RAW, ids=NAMED_RAW_IDS)
def test_review_traj_upload_cli_refuses_a_named_raw_folder(
    tmp_path, monkeypatch, layout, relative, given, destination
):
    """Review of PR #148, 2026-11-13: ``bench traj upload`` with a raw folder
    as the trajectory or as ``--workspace-dir``, to the broker or
    ``--direct``, exits 1 as for a missing workspace folder, says why and what
    to pass, and stages and sends nothing (a raw workspace is refused before
    the trajectory is read)."""
    folder = _named_raw(tmp_path, monkeypatch, layout, relative)
    run_dir = folder.parents[len(Path(relative).parts) - 1]
    args = (
        (str(folder),)
        if given == "path"
        else (str(run_dir), "--workspace-dir", str(folder))
    )
    result, staged, published = _traj_upload_cli(
        monkeypatch, *args, *DESTINATIONS[destination]
    )
    assert result.exit_code == 1, result.output
    assert _refused(result.output, folder), result.output
    assert staged == ([folder] if given == "path" else [])
    assert published == []


def test_review_traj_upload_cli_attaches_a_workspace_only_called_vf_results(
    tmp_path, monkeypatch
):
    """Review of PR #148, 2026-11-13: the keep for the CLI refusal."""
    plain = _plain_vf_results(tmp_path)
    result, _staged, published = _traj_upload_cli(
        monkeypatch, str(plain), "--workspace-dir", str(plain), "--dry-run"
    )
    assert result.exit_code == 0, result.output
    assert "Workspace attached" in result.output
    assert published == []


# Review of PR #148, 2026-11-13 (lead): --publish-hf / --publish-bucket refuse
# a job folder known to be raw evidence before the run is spent. Each row: the
# layout, --jobs-dir under the run folder, --job-name (None: the run names it).
PUBLISH_JOB_FOLDERS = [
    ("new", "", "hosted_env"),
    ("new", "hosted_env", None),
    ("new", "hosted_env/vf-results", "job"),
    ("old", "", "vf-results"),
    ("old", "vf-results", None),
    ("old", "", ".venv"),
]
PUBLISH_FLAGS = {"hf": "--publish-hf", "bucket": "--publish-bucket"}


def _eval_publish(tmp_path, monkeypatch, route, jobs, job_name, flag):
    """``bench eval run`` with a publish flag, its run and every publish
    faked: returns the result, the runs started and what was published."""
    import sys
    import types as pytypes

    from typer.testing import CliRunner

    from benchflow.cli.main import app
    from benchflow.evaluation import Evaluation
    from benchflow.publish import huggingface
    from tests.test_raw_abs_compat import _result, _tasks

    ran: list[Path] = []
    published: list[str] = []

    async def run(self):
        ran.append(self.job_dir)
        return _result(self)

    monkeypatch.setattr(Evaluation, "run", run)
    monkeypatch.setattr(
        huggingface, "_require_hf_api", lambda: _RecordingHfApi(uploaded)
    )
    uploaded: dict[str, bytes] = {}
    hub = pytypes.ModuleType("huggingface_hub")
    errors = pytypes.ModuleType("huggingface_hub.errors")
    errors.HfHubHTTPError = type("HfHubHTTPError", (Exception,), {})
    hub.errors = errors
    hub.create_bucket = lambda *args, **kwargs: published.append("<bucket>")
    hub.sync_bucket = lambda source, remote: published.extend(
        _files_under(Path(source))
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)
    tasks = _tasks(tmp_path)
    names = ["--job-name", job_name] if job_name else []
    if route == "config":
        config = tmp_path / "run.yaml"
        config.write_text(f"tasks_dir: {tasks}\nagent: oracle\njobs_dir: {jobs}\n")
        args = ["--config", str(config)]
    else:
        args = ["--tasks-dir", str(tasks), "--agent", "oracle", "--jobs-dir", str(jobs)]
    result = CliRunner().invoke(
        app, ["eval", "run", *args, *names, PUBLISH_FLAGS[flag], "org/runs"]
    )
    return result, ran, published + list(uploaded)


@pytest.mark.parametrize("flag", list(PUBLISH_FLAGS))
@pytest.mark.parametrize("route", ["tasks-dir", "config"])
@pytest.mark.parametrize(
    ("layout", "jobs", "job_name"),
    PUBLISH_JOB_FOLDERS,
    ids=[f"{a}-{b or '.'}-{c}" for a, b, c in PUBLISH_JOB_FOLDERS],
)
def test_review_eval_publish_refuses_a_raw_job_folder_before_the_run(
    tmp_path, monkeypatch, layout, jobs, job_name, route, flag
):
    """Review of PR #148, 2026-11-13 (lead's ruling): a job folder that is a
    hosted run's raw evidence, or sits inside it, was refused only when the
    finished run was published, so the run was spent for nothing. It is now
    refused before the run starts, with the same reason and exit code."""
    run_dir = _hosted_run(tmp_path, monkeypatch, "legacy", layout).run_dir
    jobs_dir = run_dir / jobs
    folder = jobs_dir / job_name if job_name else jobs_dir
    result, ran, published = _eval_publish(
        tmp_path, monkeypatch, route, jobs_dir, job_name, flag
    )
    assert result.exit_code == 1, result.output
    assert _refused(result.output, folder), result.output
    assert ran == []
    assert published == []


@pytest.mark.parametrize("flag", list(PUBLISH_FLAGS))
def test_review_eval_publish_still_runs_and_publishes_an_ordinary_job(
    tmp_path, monkeypatch, flag
):
    """Review of PR #148, 2026-11-13: the keep for the check before the run."""
    result, ran, published = _eval_publish(
        tmp_path, monkeypatch, "tasks-dir", tmp_path / "jobs", "ordinary", flag
    )
    assert result.exit_code == 0, result.output
    assert ran and published


def _train_publish(tmp_path, monkeypatch, *args: str):
    import io

    from typer.testing import CliRunner

    import benchflow.training.backends.prime_rl as prime_rl
    from benchflow.cli.main import app
    from benchflow.publish import huggingface

    started: list[list[str]] = []
    uploaded: dict[str, bytes] = {}

    class FakeProcess:
        def __init__(self, argv, **kwargs):
            started.append(argv)
            self.stdout = io.StringIO("trainer ok\n")
            self.stderr = io.StringIO("")

        def wait(self):
            return 0

    config = tmp_path / "sft.toml"
    config.write_text("max_steps = 1\n")
    checkout = tmp_path / "prime-rl-checkout"
    checkout.mkdir(exist_ok=True)
    monkeypatch.setattr(prime_rl.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(prime_rl.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(
        huggingface, "_require_hf_api", lambda: _RecordingHfApi(uploaded)
    )
    result = CliRunner().invoke(
        app,
        [
            "train",
            "run",
            "sft",
            "--backend",
            "prime-rl",
            "--config",
            str(config),
            "--data",
            "benchflow/env0-prime-sft",
            "--prime-rl-dir",
            str(checkout),
            "--dry-run",
            "--uv-no-sync",
            *args,
        ],
    )
    return result, started, list(uploaded)


@pytest.mark.parametrize("flag", ["--publish-model", "--publish-artifacts"])
@pytest.mark.parametrize(("layout", "relative"), NAMED_RAW, ids=NAMED_RAW_IDS)
def test_review_train_publish_refuses_a_raw_folder_before_training(
    tmp_path, monkeypatch, layout, relative, flag
):
    """Review of PR #148, 2026-11-13 (lead's ruling): ``bench train``'s
    --publish-model uploads the output folder and --publish-artifacts the work
    folder, both arguments. A raw one is refused before the trainer starts,
    as the publish after it would refuse it."""
    folder = _named_raw(tmp_path, monkeypatch, layout, relative)
    where = ["--output-dir", str(folder), "--work-dir", str(tmp_path / "work")]
    if flag == "--publish-artifacts":
        where = ["--output-dir", str(tmp_path / "out"), "--work-dir", str(folder)]
    result, started, uploaded = _train_publish(
        tmp_path, monkeypatch, *where, flag, "org/sft"
    )
    assert result.exit_code == 1, result.output
    assert _refused(result.output, folder), result.output
    assert started == []
    assert uploaded == []


def test_review_train_publish_still_trains_and_publishes_an_ordinary_folder(
    tmp_path, monkeypatch
):
    """Review of PR #148, 2026-11-13: the keep for the check before training."""
    result, started, uploaded = _train_publish(
        tmp_path,
        monkeypatch,
        "--work-dir",
        str(tmp_path / "work"),
        "--publish-artifacts",
        "org/sft",
    )
    assert result.exit_code == 0, result.output
    assert started and uploaded


@pytest.mark.parametrize(("layout", "relative"), NAMED_RAW, ids=NAMED_RAW_IDS)
def test_review_the_workspace_prompt_asks_again_for_a_raw_folder(
    tmp_path, monkeypatch, capsys, layout, relative
):
    """Review of PR #148, 2026-11-13: the workspace folder typed at the prompt
    is checked as --workspace-dir is; a raw one is refused and asked again,
    as a path that is no folder is."""
    from benchflow.cli import traj

    folder = _named_raw(tmp_path, monkeypatch, layout, relative)
    plain = _plain_vf_results(tmp_path)
    answers = iter([str(folder), str(plain)])
    monkeypatch.setattr(traj.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(traj.typer, "prompt", lambda *a, **kw: next(answers))
    options = traj._UploadOptions(
        path=plain,
        github_id=None,
        email=None,
        source_id="demo",
        repo=False,
        direct=False,
        container_url=None,
        dry_run=True,
        preview_steps=0,
    )
    assert traj._resolve_workspace_dir(options, plain) == (plain, True)
    assert _refused(capsys.readouterr().err, folder)


@pytest.mark.parametrize(("layout", "relative"), NAMED_RAW, ids=NAMED_RAW_IDS)
def test_review_hub_push_refuses_a_named_raw_folder(
    tmp_path, monkeypatch, layout, relative
):
    """Review of PR #148, 2026-11-13: ``bench hub push`` left a named
    ``hosted_env/`` out of its archive but packed a folder inside it whole.
    It now refuses either before it packs or reaches the Hub, exit 2 as for a
    folder that is no finished job."""
    import tarfile

    from typer.testing import CliRunner

    import benchflow.cli._hub_client as hub_client
    from benchflow.cli.main import app

    folder = _named_raw(tmp_path, monkeypatch, layout, relative)
    (folder / "summary.json").write_text("{}")
    packed: list[str] = []
    sent: list[str] = []
    pack = hub_client._pack_job

    def recording_pack(job_dir, dest):
        count = pack(job_dir, dest)
        with tarfile.open(dest) as tar:
            packed.extend(m.name for m in tar.getmembers() if m.isfile())
        return count

    class _Hub:
        url = "https://hub.example"

        def has_registry(self):
            return None

        def upload_run(self, archive, **kwargs):
            sent.append(archive.name)
            return {"imported": []}

    monkeypatch.setattr(hub_client, "_client", lambda hub: _Hub())
    monkeypatch.setattr(hub_client, "_pack_job", recording_pack)
    result = CliRunner().invoke(
        app, ["hub", "push", str(folder), "--hub", "https://hub.example"]
    )
    assert result.exit_code == 2, (result.output, packed, sent)
    assert _refused(result.output, folder), result.output
    assert packed == sent == []


@pytest.mark.parametrize("layout", LAYOUTS)
def test_review_the_evidence_bundle_holds_nothing_of_hosted_env(
    tmp_path, monkeypatch, layout
):
    """``bench bundle`` takes a trial's named files and folders only."""
    import zipfile

    from benchflow.evidence_bundle import build_bundle

    run_dir = _hosted_run(tmp_path, monkeypatch, "legacy", layout).run_dir
    out = tmp_path / "bundle.zip"
    build_bundle(run_dir, out)
    with zipfile.ZipFile(out) as bundle:
        files = {name: bundle.read(name) for name in bundle.namelist()}
    _no_raw_evidence(files)


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("cli", ["legacy", "v1"])
def test_review_hub_push_leaves_raw_vf_eval_output_out(
    tmp_path, monkeypatch, layout, cli
):
    """Review of PR #148, 2026-11-13 (blocker): ``bench hub push`` packs a
    whole job folder, and a hosted run's ``vf-results/`` (raw completions, the
    resolved config with the env args and the base URL's password) sat
    outside ``hosted_env/`` and went into the archive. Both are left out,
    where vf-results/ is now and where older runs have it, and their .venv/."""
    import tarfile

    from benchflow.cli._hub_client import _pack_job

    run_dir = _hosted_run(tmp_path, monkeypatch, cli, layout).run_dir
    job_dir = tmp_path / "jobs"
    (job_dir / "summary.json").write_text("{}")
    archive = tmp_path / "job.tar.gz"
    _pack_job(job_dir, archive)
    with tarfile.open(archive) as tar:
        files = {
            m.name: tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()
        }
    assert any(run_dir.name in name for name in files)
    _no_raw_evidence(files)


class _FakeHfApi:
    def __init__(self, uploaded: dict[str, bytes]) -> None:
        self.uploaded = uploaded

    def create_repo(self, *args, **kwargs):
        return None

    def upload_folder(self, *, folder_path, **kwargs):
        self.uploaded.update(_files_under(Path(folder_path)))


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("cli", ["legacy", "v1"])
def test_review_hf_publish_leaves_raw_vf_eval_output_out(
    tmp_path, monkeypatch, layout, cli
):
    """Review of PR #148, 2026-11-13 (blocker): ``--publish-hf`` uploads a
    snapshot of a job folder (``staged_folder``), vf-results/ included. A
    hosted run's raw output is left out of the snapshot, either layout."""
    from benchflow.publish import huggingface

    run_dir = _hosted_run(tmp_path, monkeypatch, cli, layout).run_dir
    uploaded: dict[str, bytes] = {}
    monkeypatch.setattr(huggingface, "_require_hf_api", lambda: _FakeHfApi(uploaded))
    huggingface.publish_folder_to_hf(
        tmp_path / "jobs", repo_id="org/runs", path_in_repo="jobs"
    )
    assert any(run_dir.name in name for name in uploaded)
    _no_raw_evidence(uploaded)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_review_hf_bucket_publish_leaves_raw_vf_eval_output_out(
    tmp_path, monkeypatch, layout
):
    """Review of PR #148, 2026-11-13 (blocker): ``--publish-bucket`` syncs
    the same snapshot to a bucket; nothing raw is in it either."""
    import sys
    import types as pytypes

    from benchflow.publish import huggingface

    run_dir = _hosted_run(tmp_path, monkeypatch, "v1", layout).run_dir
    uploaded: dict[str, bytes] = {}
    hub = pytypes.ModuleType("huggingface_hub")
    errors = pytypes.ModuleType("huggingface_hub.errors")
    errors.HfHubHTTPError = type("HfHubHTTPError", (Exception,), {})
    hub.errors = errors
    hub.create_bucket = lambda *args, **kwargs: None
    hub.sync_bucket = lambda source, remote: uploaded.update(_files_under(Path(source)))
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)
    huggingface.publish_folder_to_bucket(tmp_path / "jobs", bucket_id="org/runs")
    assert any(run_dir.name in name for name in uploaded)
    _no_raw_evidence(uploaded)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_review_job_export_reads_nothing_of_hosted_env(tmp_path, monkeypatch, layout):
    """``bf.load_trial`` / ``trial_export`` read a trial's named files only."""
    from benchflow.job_export import trial_export
    from benchflow.jobs import load_trial

    run_dir = _hosted_run(tmp_path, monkeypatch, "legacy", layout).run_dir
    document = trial_export(load_trial(run_dir)).model_dump_json()
    assert [s for s in HOSTED_SECRETS if s in document] == []


@pytest.mark.parametrize(
    "text,secret",
    [
        ("DEVELOPER_TOKEN=FAKEdevtok0123", "FAKEdevtok0123"),
        ("super_secret_token=FAKEsuper0123", "FAKEsuper0123"),
        ("helper_token: FAKEhelper0123", "FAKEhelper0123"),
        ('"wrapper_token": "FAKEwrap0123"', "FAKEwrap0123"),
        ("upper_access_token=FAKEupper0123", "FAKEupper0123"),
    ],
)
def test_review_a_word_ending_in_per_is_no_price(text, secret):
    """Review of PR #148, 2026-11-13: the price exception's lookbehind
    `(?<!per[_-])TOKEN` took any name ending in `per` before TOKEN for a
    price, so DEVELOPER_TOKEN, helper_token and super_secret_token values
    were kept. `per` must be a word of its own."""
    name = text.split("=")[0].split(":")[0].strip('"')
    assert _leaks(text, secret) == []
    assert _dict_leaks({name: secret}, secret) == []


def test_review_hosted_env_args_keep_flags_and_counts(tmp_path, monkeypatch):
    """Lead's ruling, review of PR #148: hosted env args masked a flag under
    a secret's word (`use_token: true`). They follow the number rule: a flag,
    a count and a short integer are kept, a PIN-like integer under a
    credential's name is masked, in hosted_run.json, result.json and the
    command alike."""
    from benchflow.hosted_env import HostedEnvRef, HostedEnvRunConfig, run_hosted_env
    from tests.test_hosted_env_rollout_contract import _patch_run_custom

    args = {
        "use_token": True,
        "eos_token": 50256,
        "password": 123,
        "api_key": 739104,
        "GITHUB_TOKEN": None,
        "refresh_token": None,
        "DB_PASS": 48213907.0,
        "is_secret": False,
    }
    for cli in ("legacy", "v1"):
        _patch_run_custom(monkeypatch, stdout="reward: avg - 1.0\n")
        monkeypatch.setattr(
            "benchflow.hosted_env.vf_eval_cli", lambda _venv, cli=cli: (cli, "0.1.0")
        )
        result = run_hosted_env(
            HostedEnvRunConfig(
                source_env=HostedEnvRef.parse(
                    "primeintellect/general-agent", version="0.1.1"
                ),
                model="gemini-3.1-flash-lite-preview",
                env_args=args,
                agent="gemini",
                jobs_dir=tmp_path / cli,
                num_examples=1,
            )
        )
        hosted = json.loads(
            (result.run_dir / "hosted_env" / "hosted_run.json").read_text()
        )
        saved = json.loads((result.run_dir / "result.json").read_text())
        expected = {**args, "api_key": MASK}
        config = json.loads((result.run_dir / "config.json").read_text())
        assert hosted["env_args"] == expected == saved["source"]["env_args"], cli
        assert config["hosted_env"]["env_args"] == expected, cli
        command = " ".join(hosted["command"])
        assert "739104" not in command and "50256" in command, cli


# Diagnostics in result.json (review of another PR, 2026-11-13).
DIAG_KEY = "AQ.FAKEdiagnosticKEY0123456789"
DIAG_VALUE = "FAKEdiagPlainValue0123"


@pytest.mark.parametrize(
    "before",
    ["upstream said ", "upstream said\\n", "upstream%3D"],
    ids=["plain", "esc-nl", "pct3D"],
)
def test_review_a_lost_connection_writes_no_key_into_result_json(tmp_path, before):
    """A connection lost with no ending: ``capture_transport`` keeps the
    exception's text as ``transport_error_info.raw_message``, which went into
    result.json unmasked. A key in it, plain, after an escaped newline or
    after ``%3D``, and a value under ``api_key=``, appear nowhere in
    result.json; the category still comes from the diagnostic."""
    from benchflow.diagnostics import RolloutDiagnostics
    from benchflow.rollout._results import _build_rollout_result

    message = f"Process closed stdout (rc=255): {before}{DIAG_KEY} api_key={DIAG_VALUE}"
    diagnostics = RolloutDiagnostics()
    diagnostics.capture_transport(ConnectionError(message))
    result = _build_rollout_result(
        tmp_path,
        task_name="task",
        rollout_name="task__1",
        agent="gemini",
        agent_name="gemini-cli",
        model=None,
        n_tool_calls=0,
        prompts=["x"],
        error=message,
        verifier_error=None,
        trajectory=[],
        partial_trajectory=True,
        rewards=None,
        started_at=datetime.now(),
        timing={},
        diagnostics=diagnostics,
    )

    text = (tmp_path / "result.json").read_text()
    assert DIAG_KEY not in text and DIAG_VALUE not in text
    saved = json.loads(text)
    assert MASK in saved["transport_error_info"]["raw_message"]
    assert saved["error_category"] == result.error_category == "pipe_closed"


def test_every_diagnostic_masks_its_free_text_in_result_json():
    """Each diagnostic of the registry, with every text field holding a key
    (a transport's stderr tail, a probe's output and traceback, a startup or
    verifier error, integration evidence, a usage limit's detail), writes
    none of it into result.json, nor into the issue line built from it."""
    import dataclasses
    import typing

    from benchflow.diagnostics import DIAGNOSTIC_REGISTRY, RolloutDiagnostics

    text = f"tool said: {DIAG_KEY} and api_key={DIAG_VALUE}"
    checked = []
    for cls in DIAGNOSTIC_REGISTRY:
        hints = typing.get_type_hints(cls)
        texts = [
            f.name
            for f in dataclasses.fields(cls)
            if hints[f.name] in (str, str | None) and f.name != "reason"
        ]
        diagnostics = RolloutDiagnostics()
        diagnostics.set(cls(**dict.fromkeys(texts, text)))
        saved = diagnostics.to_result_fields()[cls.field]
        dumped = json.dumps(saved)
        assert DIAG_KEY not in dumped and DIAG_VALUE not in dumped, cls.field
        line = cls.format_issue_from_dict("task", saved)
        assert DIAG_KEY not in line and DIAG_VALUE not in line, cls.field
        checked.append((cls.field, texts))
    assert len(checked) == len(DIAGNOSTIC_REGISTRY) >= 10


# The gap and line breaks (lead's ruling, review of PR #148, 2026-11-13).
YAML_MASKED = [
    ("apiKey:\n  FAKEyaml0123", "apiKey:\n  " + MASK),
    ("api_key:\r\n\tFAKEyaml0123", "api_key:\r\n\t" + MASK),
    ("password:\n    FAKEyaml0123", "password:\n    " + MASK),
    ("apiKey:\\n  FAKEyaml0123", "apiKey:\\n  " + MASK),
    ("api_key:\\r\\n\\tFAKEyaml0123", "api_key:\\r\\n\\t" + MASK),
    ("apiKey:%0A%20%20FAKEyaml0123", "apiKey:%0A%20%20" + MASK),
    ("password:%0D%0A%09FAKEyaml0123", "password:%0D%0A%09" + MASK),
    # The next line's own secret, under its own name, by its own rule.
    ("password:\n  api_key: FAKEinner0123", "password:\n  api_key: " + MASK),
    # A blank line before the value is crossed too (the lead's scope,
    # 2026-11-15 14:26Z: R192C's class 2, masked on rel 040c1d0c1).
    ("apiKey:\n\n  FAKEyaml0123", "apiKey:\n\n  " + MASK),
]
YAML_KEPT = [
    "GITHUB_TOKEN=\nHOME=/root",
    "API_KEY=\nNEXT_VAR=value",
    "password:\nusername: bob",
    "secret:\n  name: x",
    "token:\n  HOME=/root",
    "GITHUB_TOKEN=\\nHOME=/root",
    "API_KEY=%0ANEXT_VAR=value",
    # A blank value at the end of the text.
    "password:",
    "password: ",
]


@pytest.mark.parametrize("text,expected", YAML_MASKED)
def test_review_a_yaml_block_value_on_the_next_line_is_masked(text, expected):
    """Lead's ruling, review of PR #148: the gap after a name's `:` or `=`
    crosses one line break (real, escaped or URL-encoded) only before an
    indented line that holds no name of its own, so a YAML block's value is
    masked; possessive, so a second pass changes nothing."""
    assert redact_trajectory_text(text) == expected
    assert _leaks(text, "FAKEyaml0123" if "yaml" in text else "FAKEinner0123") == []
    assert _idempotent(text)


@pytest.mark.parametrize("text", YAML_KEPT)
def test_review_the_gap_never_takes_the_next_variable(text):
    """Lead's ruling, review of PR #148: `GITHUB_TOKEN=\\nHOME=/root` masked
    HOME: the gap crossed the line break into the next variable. A blank
    value followed by an unindented line, or by an indented `name: x` of its
    own, or by two line breaks, leaves the text as it is, in every layer."""
    assert _unchanged(text) == dict.fromkeys(_unchanged(text), True)
    assert _idempotent(text)


@pytest.mark.parametrize(
    "scheme",
    ["Bearer", "Basic", "Token", "ApiKey", "Negotiate", "NTLM", "Bot", "Key", "SSWS"],
)
@pytest.mark.parametrize("gap", [" ", "%20", "%09", "\\t", "\t"])
def test_review_an_encoded_gap_keeps_the_scheme_word(scheme, gap):
    """Review of PR #148, 2026-11-13: `Authorization:%20Bearer <t>` masked the
    word Bearer too: the gap gave `%20` back to the value. The scheme word
    stays and only the token is masked, as in `Authorization: Bearer <t>`,
    for every scheme and every gap; Bot, Key and SSWS are schemes now."""
    text = f"Authorization:{gap}{scheme} FAKEtok0123456789abc"
    assert redact_trajectory_text(text) == f"Authorization:{gap}{scheme} {MASK}"
    assert _leaks(text, "FAKEtok0123456789abc") == []


@pytest.mark.parametrize(
    "text",
    [
        'Authorization: Digest username="bob", realm="x", nonce="abc", '
        'response="6629fae49393a05397450978507c4ef1"',
        "Authorization: Digest username=bob, response=6629fae49393a05397450978507c4ef1",
    ],
)
def test_review_a_digest_header_masks_its_parameters(text):
    """Review of PR #148, 2026-11-13: Digest's credential is its parameter
    list; the whole of it after the scheme is masked, in every layer."""
    assert redact_trajectory_text(text) == "Authorization: Digest " + MASK
    assert _leaks(text, "6629fae49393a05397450978507c4ef1") == []


PER_WORDS = [
    "DEVELOPER_TOKEN",
    "GOOGLE_ADS_DEVELOPER_TOKEN",
    "developerToken",
    "HELPER_TOKEN",
    "WRAPPER_TOKEN",
    "SCRAPER_TOKEN",
    "SUPER_TOKEN",
    "ZOOKEEPER_TOKEN",
]


@pytest.mark.parametrize("name", PER_WORDS)
def test_review_a_name_ending_in_per_keeps_its_token_masked(name):
    """Review of PR #148, 2026-11-13: the price exception took any name
    ending in `per` before TOKEN for a price (`develo|per_token`), so
    DEVELOPER_TOKEN and the others leaked. `per` must be a word of its own;
    a secret under each name is masked as text and as a dict key."""
    value = "FAKEperword0123"
    for text in (f"{name}={value}", f'"{name}": "{value}"', f"{name}: {value}"):
        assert _leaks(text, value) == [], text
    assert _dict_leaks({name: value}, value) == []


def test_review_an_ansi_coloured_key_is_masked_and_coloured_prose_kept():
    """Review of PR #148, 2026-11-13: a key right after an ANSI colour code
    (`ESC[32m`, `ESC[1;32m`, `\\u001b[32m` in a JSON copy) followed the
    code's final letter, and the boundary kept it. The code counts as a
    boundary. (the lead, 2026-11-14 19:42Z): a generic sk- key glued
    to a word, and a word ending in `sk` before `-` and twelve letters, are
    masked as in 0.9.2; coloured `<task-notification>` stays."""
    sk48 = "sk-" + "aB3dE" * 9 + "T3Bl"
    hex32 = "sk-" + "0f1e2d3c" * 4
    for code in ("\x1b[32m", "\x1b[1;32m", "\\u001b[32m", "\\x1b[0m", "\\033[2K"):
        for key in (sk48, hex32, AQ_KEY):
            assert _leaks(f"{code}{key} done", key) == [], (code, key)
        assert _leaks(f"{code}task-instructions", "sk-instructions") == []
        assert _unchanged(f"{code}<task-notification>") == dict.fromkeys(LAYERS, True)
    assert _leaks("keysk-" + "0f1e2d3c" * 4, "0f1e2d3c" * 4) == []


# (the lead, 2026-11-14 19:42Z, 0.9.3's #208 alike): 0.9.2's generic
# sk- rule wherever a key stands, so these words, which 0.10's left boundary
# kept, are masked again; the one exception is `task-notification(s)`.
SK_WORDS_MASKED_AS_IN_092 = [
    "/app/task-implementation",
    "task-implementation",
    "risk-assessmentreport2024",
    "desk-reservationsystem",
    "whisk-togetherthebatter",
    "task-instructions",
    "task-specification",
    "disk-0123456789ab",
]


@pytest.mark.parametrize("text", SK_WORDS_MASKED_AS_IN_092)
def test_a_word_ending_in_sk_before_twelve_letters_is_masked_as_in_092(text):
    tail = text[text.index("sk-") :]
    assert _leaks(text, tail) == []


def test_review_the_judge_keeps_identifiers_and_masks_shaped_keys():
    """Review of PR #148, 2026-11-13: the judge's evidence lost its
    identifier boundary and masked `build_dtn_network_topology_config`. A
    key starts where an identifier starts; a JWT and a PEM block are masked
    wherever they are (the evidence cannot tell a declared output file's
    text from the rest)."""
    from benchflow.taskmd import judging

    scrub = judging.Scrub.for_run(None, {})
    for word in (
        "build_dtn_network_topology_config",
        "my-ghp_abcdefghijklmnopqrstuvwxyz0123",
        "task-specification desk-reservations",
    ):
        assert scrub.evidence(word) == word
    assert JWT.rsplit(".", 1)[1] not in scrub.evidence(f"token: {JWT}")


@pytest.mark.parametrize(
    "name",
    ["db_pass_word", "DB_PASS_WORD", "x" * 80 + "_password", "a" * 70 + "ApiKey"],
)
def test_review_the_export_names_keep_compact_and_long_forms(name):
    """Review of PR #148, 2026-11-13: the shared names dropped two forms the
    export redactor's list masked: a compact password (`*pass_word`) and a
    name with more than 64 characters before its marker."""
    from benchflow.publish.redact import _is_sensitive_key

    assert _is_sensitive_key(name)
    assert redact_value({name: "FAKEcompact0123"})[0] == {
        name: "<XXX-benchflow-key-values-XXX>"
    }


# The shared name classifier is the override record's own file.
CREDENTIAL_NAMES_SHA256 = (
    "614cbdbe66678b35092819d5f25364c69996feff57869ba5c1c1a7308c7c8696"
)


def test_the_shared_name_classifier_is_the_override_records_copy():
    """``src/benchflow/_utils/credential_names.py`` is shared with the
    override record (the config-override split): there is one copy,
    which the redactors' number rule reads too. Change this hash with it."""
    import hashlib

    from benchflow._utils import credential_names

    data = Path(credential_names.__file__).read_bytes()
    assert hashlib.sha256(data).hexdigest() == CREDENTIAL_NAMES_SHA256


# Usage blocks: cost and token accounting reads every number in them.
_ANTHROPIC_USAGE = {
    "input_tokens": 123456,
    "cache_creation_input_tokens": 2345678,
    "cache_read_input_tokens": 98765432,
    "output_tokens": 456789,
    "cache_creation": {
        "ephemeral_5m_input_tokens": 1234567,
        "ephemeral_1h_input_tokens": 1111111,
    },
    "server_tool_use": {"web_search_requests": 3},
    "service_tier": "standard",
}
_OPENAI_CHAT_USAGE = {
    "prompt_tokens": 234567,
    "completion_tokens": 34567,
    "total_tokens": 269134,
    "prompt_tokens_details": {"cached_tokens": 200704, "audio_tokens": 0},
    "completion_tokens_details": {
        "reasoning_tokens": 123456,
        "audio_tokens": 0,
        "accepted_prediction_tokens": 1024,
        "rejected_prediction_tokens": 2048,
    },
}
_OPENAI_RESPONSES_USAGE = {
    "input_tokens": 345678,
    "input_tokens_details": {"cached_tokens": 300032},
    "output_tokens": 45678,
    "output_tokens_details": {"reasoning_tokens": 40960},
    "total_tokens": 391356,
}
_GEMINI_USAGE = {
    "promptTokenCount": 456789,
    "candidatesTokenCount": 56789,
    "totalTokenCount": 523456,
    "cachedContentTokenCount": 409600,
    "thoughtsTokenCount": 9878,
    "toolUsePromptTokenCount": 1234,
    "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 456789}],
    "cacheTokensDetails": [{"modality": "TEXT", "tokenCount": 409600}],
    "candidatesTokensDetails": [{"modality": "TEXT", "tokenCount": 56789}],
}
# OpenRouter: built from OpenRouter's documented schema (the usage object of
# a chat completion, and a /api/v1/generation record); the repo has no
# captured OpenRouter response to copy. Not a captured response.
_OPENROUTER_USAGE = {
    "prompt_tokens": 567890,
    "completion_tokens": 67890,
    "total_tokens": 635780,
    "cost": 0.0123456,
    "is_byok": False,
    "cost_details": {
        "upstream_inference_cost": 0.0119,
        "upstream_inference_prompt_cost": 0.009,
        "upstream_inference_completions_cost": 0.0029,
    },
    "prompt_tokens_details": {"cached_tokens": 512000, "audio_tokens": 0},
    "completion_tokens_details": {"reasoning_tokens": 654321, "image_tokens": 0},
}
_OPENROUTER_GENERATION = {
    "data": {
        "id": "gen-1731500000-FAKEgeneration01",
        "upstream_id": "chatcmpl-FAKEupstream0123",
        "total_cost": 0.0123456,
        "cache_discount": 0.0004,
        "upstream_inference_cost": 0.0119,
        "created_at": "2026-11-13T12:00:00.000Z",
        "model": "anthropic/claude-sonnet-4.5",
        "app_id": 123456,
        "streamed": True,
        "cancelled": False,
        "provider_name": "Anthropic",
        "latency": 1234,
        "moderation_latency": 12,
        "generation_time": 4567,
        "finish_reason": "stop",
        "native_finish_reason": "end_turn",
        "tokens_prompt": 567890,
        "tokens_completion": 67890,
        "native_tokens_prompt": 570000,
        "native_tokens_completion": 68000,
        "native_tokens_completion_images": 0,
        "native_tokens_reasoning": 654321,
        "native_tokens_cached": 512000,
        "num_media_prompt": 0,
        "num_media_completion": 0,
        "num_search_results": 0,
        "origin": "https://openrouter.ai/",
        "usage": 0.0123456,
        "is_byok": False,
    }
}
USAGE_BLOCKS = {
    "anthropic messages": {
        "id": "msg_01FAKEFAKEFAKEFAKE",
        "model": "claude-sonnet-4-5",
        "usage": _ANTHROPIC_USAGE,
    },
    "openai chat": {"model": "gpt-5", "usage": _OPENAI_CHAT_USAGE},
    "openai responses": {"model": "gpt-5", "usage": _OPENAI_RESPONSES_USAGE},
    "codex token_count": {
        "timestamp": "2026-11-13T12:00:00.000Z",
        "type": "event_msg",
        "payload": {
            "type": "token_count",
            "info": {
                "total_token_usage": {
                    "input_tokens": 3456789,
                    "cached_input_tokens": 3000000,
                    "output_tokens": 123456,
                    "reasoning_output_tokens": 98765,
                    "total_tokens": 3580245,
                },
                "last_token_usage": {
                    "input_tokens": 123456,
                    "cached_input_tokens": 120000,
                    "output_tokens": 2345,
                    "reasoning_output_tokens": 1234,
                    "total_tokens": 125801,
                },
                "model_context_window": 272000,
            },
        },
    },
    "gemini usageMetadata": {
        "usageMetadata": _GEMINI_USAGE,
        "modelVersion": "gemini-2.5-pro",
    },
    "openrouter usage": {
        "model": "anthropic/claude-sonnet-4.5",
        "usage": _OPENROUTER_USAGE,
    },
    "openrouter generation": _OPENROUTER_GENERATION,
    "proxy log line": {
        "metadata": {
            "call_purpose": "agent",
            "request_model": "claude-sonnet-4-5",
            "provider_model": "anthropic/claude-sonnet-4-5",
            "response_cost": 0.0456,
        },
        "request": {
            "timestamp": "2026-11-13T12:00:00Z",
            "body": {
                "model": "claude-sonnet-4-5",
                "max_tokens": 64000,
                "temperature": 1.0,
            },
        },
        "response": {
            "status_code": 200,
            "body": {"model": "claude-sonnet-4-5", "usage": _ANTHROPIC_USAGE},
        },
        "duration_ms": 12345.6,
    },
    "cache and thought fields": {
        "cached_tokens": 123456,
        "n_cache_tokens": 234567,
        "cache_read_tokens": 345678,
        "thought_tokens": 456789,
        "n_cache_read_tokens": 5678901,
        "n_cache_creation_tokens": 678901,
        "n_input_tokens": 7890123,
        "n_output_tokens": 890123,
        "total_tokens": 9012345,
        "token_budget": 1000000,
        "max_tokens": 64000,
        "cost_usd": 12.345678,
        "usd_per_token": {"input": 3e-06, "output": 1.5e-05},
        "input_cost_per_token": 3e-06,
    },
}


@pytest.mark.parametrize("name", list(USAGE_BLOCKS))
def test_cost_and_token_accounting_usage_blocks_pass_redaction_byte_identical(name):
    """Guards cost and token accounting: every usage block BenchFlow
    reads (Anthropic, OpenAI chat and responses, Codex token_count, Gemini
    usageMetadata, OpenRouter usage and /generation, the proxy log line, and
    the cache and thought fields) comes out of the trajectory redactor, both
    acp_trajectory.jsonl writers and the export redactor byte for byte. A
    mask here changes a bill: fix the rule, never this fixture."""
    block = USAGE_BLOCKS[name]
    canonical = json.dumps(block, sort_keys=True)
    assert json.dumps(redact_trajectory_obj(block), sort_keys=True) == canonical
    assert json.dumps(redact_trajectory_obj({"rows": [block]}), sort_keys=True) == (
        json.dumps({"rows": [block]}, sort_keys=True)
    )
    assert redact_acp_trajectory_jsonl([block]) == json.dumps(block)
    assert RedactedJSONLSnapshot().serialize([block]) == json.dumps(block)
    assert redact_value(block) == (block, 0)


def test_cost_and_token_accounting_parsers_read_the_same_numbers(tmp_path):
    """Guards cost and token accounting: the repo's own usage
    parsers (``_exchange_token_usage``, the proxy log's ``_proxy_scan``)
    read the same numbers from a redacted record as from the original."""
    from types import SimpleNamespace

    from benchflow.trajectories.model_calls import _proxy_scan
    from benchflow.trajectories.types import _exchange_token_usage

    for body in (
        USAGE_BLOCKS["anthropic messages"],
        USAGE_BLOCKS["openai chat"],
        USAGE_BLOCKS["openai responses"],
        USAGE_BLOCKS["gemini usageMetadata"],
        USAGE_BLOCKS["openrouter usage"],
    ):

        def usage(record):
            return _exchange_token_usage(
                SimpleNamespace(response=SimpleNamespace(body=record))
            )

        assert usage(redact_trajectory_obj(body)) == usage(body)
    line = USAGE_BLOCKS["proxy log line"]
    raw, masked = tmp_path / "raw.jsonl", tmp_path / "masked.jsonl"
    raw.write_text(json.dumps(line) + "\n")
    masked.write_text(redact_acp_trajectory_jsonl([line]) + "\n")
    calls, _ = _proxy_scan(raw)
    assert calls and calls[0]["cache_read_tokens"] == 98765432
    assert _proxy_scan(masked) == _proxy_scan(raw)


def test_the_classifier_and_the_integer_guards_on_openrouter_names():
    """Guards cost and token accounting: the names of OpenRouter's
    usage and /generation record (plural `tokens` first, `*_details`, a
    float cost, a boolean is_byok) are no credential's to the integer rule,
    whatever the shared classifier says of each."""
    from benchflow.trajectories.types import is_secret_number

    names = [
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cached_tokens",
        "reasoning_tokens",
        "tokens_prompt",
        "tokens_completion",
        "native_tokens_prompt",
        "native_tokens_completion",
        "native_tokens_reasoning",
        "native_tokens_cached",
        "native_tokens_completion_images",
        "app_id",
        "latency",
        "generation_time",
    ]
    assert [n for n in names if is_secret_number(n, 123456789)] == []
    assert is_secret_number("cost", 0.0123) is False
    assert is_secret_number("is_byok", False) is False


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("cli", ["legacy", "v1"])
def test_review_the_reviewer_sandbox_gets_nothing_of_hosted_env(
    tmp_path, monkeypatch, layout, cli
):
    """Review of PR #148, 2026-11-13 (blockers): ``bench review --rubric``
    and the regrade's review view copy a trial folder into the reviewer's
    sandbox (``review.wrapper.copy_evidence``): hosted_env/'s raw logs went
    with it, and so did vf-results/ (raw completions, the resolved config).
    Neither is copied, where vf-results/ is now or where older runs have it,
    and neither is a leftover .venv/; nor into the regrade's view."""
    from benchflow.eval_regrade import _review_view
    from benchflow.review.wrapper import copy_evidence

    run_dir = _hosted_run(tmp_path, monkeypatch, cli, layout).run_dir
    copy_evidence(run_dir, tmp_path / "sandbox-evidence")
    copied = _files_under(tmp_path / "sandbox-evidence")
    # copy_evidence drops the write bits; give them back for the cleanup.
    for path in (tmp_path / "sandbox-evidence").rglob("*"):
        path.chmod(path.stat().st_mode | 0o200)
    (tmp_path / "sandbox-evidence").chmod(0o700)
    _no_raw_evidence(copied)

    attempt = tmp_path / "attempt"
    attempt.mkdir()
    view = tmp_path / "view"
    _review_view(run_dir, attempt, view, None, "sha256:" + "0" * 64)
    files = {
        name: data
        for name, data in _files_under(view).items()
        if not name.startswith("evidence")
    }
    _no_raw_evidence(files)


def test_review_raw_evidence_is_one_list_for_every_copy_path():
    """The folders that stay on this machine are named once
    (``benchflow._utils.raw_evidence``), and each path that copies a run or a
    job folder elsewhere asks it: a new raw folder is one edit."""
    import inspect

    from benchflow import eval_regrade
    from benchflow.cli import _hub_client
    from benchflow.publish import _snapshot
    from benchflow.review import wrapper

    for module in (_snapshot, _hub_client, wrapper, eval_regrade):
        source = inspect.getsource(module)
        assert "is_raw_evidence" in source, module.__name__
        assert '"hosted_env"' not in source, module.__name__


SCHEMES = [
    "Bearer",
    "Basic",
    "Token",
    "ApiKey",
    "Negotiate",
    "NTLM",
    "Bot",
    "Key",
    "SSWS",
]
NEXT_LINE = {
    "real": "\n  ",
    "CRLF and tab": "\r\n\t",
    "escaped": "\\n  ",
    "encoded": "%0A%20%20",
}


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("form", list(NEXT_LINE))
def test_review_a_token_on_the_next_indented_line_is_masked(scheme, form):
    """Review of PR #148, 2026-11-13: between an Authorization scheme and its
    token only spaces and tabs were allowed, so `Authorization: Bearer\\n
    <token>` leaked in every layer (89bbfed93 masked it). The gap there is
    the separator's: one line break before an indented line with no name of
    its own. Exact output, every layer, and a second pass changes nothing."""
    token = "FAKEnextline0123456789"
    text = f"Authorization: {scheme}{NEXT_LINE[form]}{token}"
    assert (
        redact_trajectory_text(text)
        == f"Authorization: {scheme}{NEXT_LINE[form]}{MASK}"
    )
    assert _leaks(text, token) == []
    assert _idempotent(text)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_review_a_scheme_before_a_header_of_its_own_keeps_it(scheme):
    """The scheme's gap never takes the next header: `Authorization: Bearer`
    followed by `Host: h`, unindented or indented, is left as it is. A
    token after a blank line is the scheme's (2026-11-15 14:26Z)."""
    for text in (
        f"Authorization: {scheme}\nHost: h",
        f"Authorization: {scheme}\n  Host: h",
    ):
        assert _unchanged(text) == dict.fromkeys(LAYERS, True), text
    text = f"Authorization: {scheme}\n\n  FAKEnextline0123456789"
    assert redact_trajectory_text(text) == f"Authorization: {scheme}\n\n  " + MASK
