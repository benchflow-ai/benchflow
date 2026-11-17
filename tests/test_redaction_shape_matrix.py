"""Every mask rule, in every context a secret reaches it from, on every path
that writes it.

A rule with a left boundary (``\\b``, ``(?<![A-Za-z0-9])``) misses a secret
whose text is JSON-escaped: an escaped newline is the two characters ``\\n``,
so the secret follows the letter ``n``; likewise ``\\t`` and ``\\uXXXX``, and
the hex digit of a URL-encoded ``%22`` or ``%3D``. ``_redact_field`` masks
every dict value through a ``json.dumps`` copy, so a real newline, tab or
curly quote before a secret in a tool output reaches the rule escaped:
``acp_trajectory.jsonl``, session logs and the exports go that way.

Each case puts one made-up secret in one context and checks the six write
paths: the text redactor, result.json's ``_masked``, a dict value
(``redact_trajectory_obj``), an ACP JSONL line, a serialized line redacted as
text (the trainer exports), and ``publish.redact.redact_value``.

KNOWN_LEAKS holds a known gap, with the write paths that keep the secret,
as a strict xfail: when a fix closes it, its case passes, the strict xfail
fails, and its entry must go; when a case leaks on other paths than its
entry says, it fails outright. It is empty: every case here passes.

Every value here is made up; the key-shaped ones are built at run time.
"""

from __future__ import annotations

import json
import re

import pytest

from benchflow._utils.evaluation_results import _masked
from benchflow.publish.redact import redact_value
from benchflow.trajectories.types import (
    redact_acp_trajectory_jsonl,
    redact_trajectory_obj,
    redact_trajectory_text,
)

TAIL = "FAKE" + "a1B2c3" * 4
DASHES = "-" * 5
VALUE = "FAKEpl" + "0aZ9bY" * 3  # a carrier's value: no prefix of its own

# id: (secret unit, the part that must not survive). Families are masked
# whole; carriers and URLs keep their name and drop the value.
FAMILIES = {
    "pem": (
        DASHES
        + "BEGIN PRIVATE KEY"
        + DASHES
        + "\nMIIEvFAKE"
        + "QwErTy12" * 4
        + "\n"
        + DASHES
        + "END PRIVATE KEY"
        + DASHES,
        "QwErTy12" * 2,
    ),
    "sk-ant": ("sk-ant-api03-" + TAIL, TAIL),
    "ya29": ("ya29." + TAIL, TAIL),
    "AQ.": ("AQ." + TAIL, TAIL),
    "glpat": ("glpat-" + TAIL, TAIL),
    "sk_live": ("sk_live_" + TAIL, TAIL),
    "sk-benchflow": ("sk-benchflow-" + TAIL, TAIL),
    "sk-proj": ("sk-proj-" + TAIL, TAIL),
    "sk-": ("sk-" + TAIL, TAIL),
    "AIzaSy": ("AIzaSy" + TAIL, TAIL),
    "AKIA": ("AKIA" + "IOSFODNN7EXAMPLE", "IOSFODNN7EXAMPLE"),
    "dtn": ("dtn_" + TAIL, TAIL),
    "ghp": ("ghp_" + TAIL, TAIL),
    "github_pat": ("github_pat_11FAKE_" + TAIL, TAIL),
    "xoxb": ("xoxb-1234-FAKE-" + TAIL, TAIL),
    "gsk": ("gsk_" + TAIL, TAIL),
    "xai": ("xai-" + TAIL, TAIL),
    "r8": ("r8_" + TAIL, TAIL),
    "hf": ("hf_" + TAIL, TAIL),
    "fw": ("fw_" + TAIL, TAIL),
    "jwt": (
        "eyJhbGciOiJIUzI1NiJ9FAKE.eyJzdWIiOiJGQUtFIn0.FAKEsig" + "0123456789abcdef" * 2,
        "0123456789abcdef" * 2,
    ),
}
# id: (name, separator, header-style)
CARRIERS = {
    "x-api-key": ("x-api-key", ": ", True),
    "api_key": ("api_key", "=", False),
    "master_key": ("master_key", "=", False),
    "BEARER_TOKEN": ("AWS_BEARER_TOKEN_BEDROCK", "=", False),
    "GITHUB_TOKEN": ("GITHUB_TOKEN", "=", False),
    "DB_PASS": ("DB_PASS", "=", False),
    "Cookie": ("Cookie", ": sid=", True),
    "Authorization-Bearer": ("Authorization", ": Bearer ", True),
    "Authorization-bare": ("Authorization", ": ", True),
    # A CGI or WSGI env dump names the headers after HTTP_.
    "HTTP_AUTHORIZATION": ("HTTP_AUTHORIZATION", "=Bearer ", False),
    "REDIRECT_HTTP_AUTHORIZATION": ("REDIRECT_HTTP_AUTHORIZATION", "=Basic ", False),
    "HTTP_COOKIE": ("HTTP_COOKIE", "=sid=", False),
    "HTTP_X_API_KEY": ("HTTP_X_API_KEY", "=", False),
    # PER_<word>_TOKEN names a credential unless <word> is a kind of token
    # (review of PR #173, 2026-11-13: the price rule had kept these). The
    # prices that must stay are in test_redaction_price_names.py.
    "MCP_PER_USER_TOKEN": ("MCP_PER_USER_TOKEN", "=", False),
    "PER_USER_TOKEN": ("PER_USER_TOKEN", "=", False),
    "PER_SERVER_TOKEN": ("PER_SERVER_TOKEN", "=", False),
    "GITLAB_PER_PROJECT_TOKEN": ("GITLAB_PER_PROJECT_TOKEN", "=", False),
    "PER_ORG_TOKEN": ("PER_ORG_TOKEN", "=", False),
    "per_request_token": ("per_request_token", ": ", False),
    # A numbered password's name and PASS_VALUE (review of PR #173,
    # 2026-11-13, Finding A: the *_PASS rule had kept them).
    "DB_PASS_2": ("DB_PASS_2", "=", False),
    "PASS_VALUE": ("PASS_VALUE", "=", False),
    "PASSVALUE": ("PASSVALUE", "=", False),
    "PASS_2": ("PASS_2", "=", False),
    # A passphrase's name (review of PR #173, 2026-11-13: no rule read it).
    "PASSPHRASE": ("PASSPHRASE", "=", False),
    "PASS_PHRASE": ("PASS_PHRASE", "=", False),
    "DB_PASSPHRASE": ("DB_PASSPHRASE", "=", False),
    # A secret name, then an Authorization scheme before the token (review
    # of PR #173, 2026-11-13, RPUB's MEDIUM: the carrier took the scheme for
    # the value and left the token, in every layer).
    "x-api-key-Bearer": ("x-api-key", ": Bearer ", True),
    "api_key-Bearer": ("api_key", ": Bearer ", False),
    "OPENAI_API_KEY-Bearer": ("OPENAI_API_KEY", "=Bearer ", False),
    "GITHUB_TOKEN-token": ("GITHUB_TOKEN", "=token ", False),
    "DB_PASS-Basic": ("DB_PASS", "=Basic ", False),
    # A secret's name numbered after its ending (review of PR #173,
    # 2026-11-13: no rule read the number).
    "DB_PASSWORD_2": ("DB_PASSWORD_2", "=", False),
    "API_TOKEN_2": ("API_TOKEN_2", "=", False),
    "GITHUB_TOKEN_1": ("GITHUB_TOKEN_1", "=", False),
    "SECRET_2": ("SECRET_2", "=", False),
    "AWS_SECRET_ACCESS_KEY_2": ("AWS_SECRET_ACCESS_KEY_2", "=", False),
    "DB_PASSPHRASE_2": ("DB_PASSPHRASE_2", "=", False),
}
URLS = {
    "url-password": ("https://user:" + VALUE + "@host.example/x", VALUE),
    "url-token": ("https://" + VALUE + "@github.com/org/repo.git", VALUE),
    "url-access_token": ("https://h.example/p?access_token=" + VALUE + "&x=1", VALUE),
    "url-sig": ("https://acct.blob.example/c?sv=2024&sig=" + VALUE, VALUE),
}

# What comes right before the secret.
PREFIXES = {
    "bol": "",
    "space": "see ",
    "eq": "v=",
    "colon": "v:",
    "dquote": '"',
    "squote": "'",
    "paren": "(",
    "underscore": "X_",
    # A real character that json.dumps escapes on the dict-value path.
    "nl": "out\n",
    "tab": "out\t",
    "rsquo": "it\u2019",
    "crlf": "HTTP/1.1 200 OK\r\n",
    # Text that is already JSON-escaped (a tool printing a JSON file).
    "esc-nl": "out\\n",
    "esc-tab": "out\\t",
    "esc-u2019": "it\\u2019",
    # URL-encoded.
    "pct22": "q=%22",
    "pct3D": "q=v%3D",
    # An ANSI colour code, as a terminal log has it and as JSON escapes it.
    "ansi": "out\x1b[32m",
    "esc-ansi": "out\\u001b[1;32m",
    "ansi-truecolor": "out\x1b[38;2;255;255;255;48;2;0;0;0m",
    "ansi-private": "out\x1b[?25l",
    "ansi-colon": "out\x1b[38:5:208m",
    "pct-ansi": "q=%1B[32m",
}
# What comes right after it.
SUFFIXES = {"suf-esc-nl": "\\nx", "suf-pct22": "%22", "suf-dquote": '"'}
# A letter right before the secret: rules without a left boundary mean to
# catch it; AQ., sk_live_ and the JWT rule keep a word that merely ends so.
GLUED_MASKED = set(FAMILIES) - {"AQ.", "sk_live", "sk-", "jwt", "pem"}

LAYERS = ("text", "_masked", "obj", "acp", "export", "publish")


def _leaks(raw: str, secret: str) -> list[str]:
    """The write paths that keep *secret* when *raw* is a string value."""
    event = {"type": "tool_call_update", "content": [{"type": "text", "text": raw}]}
    out = {
        "text": redact_trajectory_text(raw),
        "_masked": _masked(raw),
        "obj": redact_trajectory_obj({"output": raw})["output"],
        "acp": redact_acp_trajectory_jsonl([event]),
        "export": redact_trajectory_text(json.dumps({"output": raw})),
        "publish": json.dumps(redact_value({"output": raw})[0]),
    }
    return [layer for layer in LAYERS if secret in out[layer]]


def _dict_key_leaks(name: str, value: str, secret: str) -> list[str]:
    """The name as a key of a headers mapping."""
    obj = {"headers": {name: value}}
    text = json.dumps(obj)
    out = {
        "text": redact_trajectory_text(text),
        "_masked": _masked(text),
        "obj": json.dumps(redact_trajectory_obj(obj)),
        "acp": redact_acp_trajectory_jsonl([{"type": "x", **obj}]),
        "export": redact_trajectory_text(text),
        "publish": json.dumps(redact_value(obj)[0]),
    }
    return [layer for layer in LAYERS if secret in out[layer]]


def _cases():
    yield from ((fid, *case) for fid, case in FAMILIES.items())
    for cid, (name, sep, _) in CARRIERS.items():
        yield cid, name + sep + VALUE, VALUE
    yield from ((uid, *case) for uid, case in URLS.items())


DUMPED = ("nl", "tab", "rsquo", "crlf")
ESCAPED = ("esc-nl", "esc-tab", "esc-u2019")
ENCODED = ("pct22", "pct3D")
SEPARATED = ("sep-tab", "sep-nl")
WHY = {
    **dict.fromkeys(
        DUMPED,
        "the dict-value path escapes the character before it (json.dumps in "
        "_redact_field), and the left boundary sees the escape's last letter or digit",
    ),
    **dict.fromkeys(ESCAPED, "the left boundary sees the escape's letter or digit"),
    **dict.fromkeys(ENCODED, "the left boundary sees the hex digit"),
    "underscore": "the left boundary counts _ as part of a word",
    **dict.fromkeys(
        SEPARATED,
        r"\s* between the name and the value does not match an escaped \t or \n",
    ),
}
RULE = {
    "sk_live": r"types.py \b[sr]k_live_",
    "jwt": "types.py (?<![A-Za-z0-9_-])eyJ",
    "DB_PASS": "types.py (?<![A-Za-z0-9])[A-Z0-9_]{1,64}_(?:PASS|PWD)",
    "Cookie": "types.py (?<![A-Za-z0-9_-])(?:set-)?cookie",
    "Authorization-Bearer": "types.py (?<![A-Za-z0-9_-])(?:proxy-)?authorization",
    "Authorization-bare": "types.py (?<![A-Za-z0-9_-])(?:proxy-)?authorization",
    "x-api-key": "types.py (?:x-api-key|x-goog-api-key|api-key)",
}
DICT_KEY_WHY = (
    "publish/redact.py _is_sensitive_key does not count this name as a "
    "credential's, and the value has no key prefix of its own, so redact_value "
    "keeps it"
)

EVERY_PATH = LAYERS
# The paths that json.dumps a dict value before the rules see it.
DUMPS = ("obj", "acp", "export")
# publish's own Bearer rule masks the token whatever comes before the header.
ALL_BUT_PUBLISH = tuple(layer for layer in LAYERS if layer != "publish")

# Gaps found on 2026-11-13: (rule id, context) -> the write paths that keep
# the secret.
KNOWN_LEAKS: dict[tuple[str, str], tuple[str, ...]] = {}


def _known(rules, contexts, paths):
    KNOWN_LEAKS.update(
        {(rule, context): paths for rule in rules for context in contexts}
    )


def _param(rule: str, context: str, *values):
    marks = []
    if (rule, context) in KNOWN_LEAKS:
        why = DICT_KEY_WHY if context == "dict-key" else f"{RULE[rule]}: {WHY[context]}"
        paths = ", ".join(KNOWN_LEAKS[rule, context])
        reason = f"leaks on {paths}. {why}"
        marks = [pytest.mark.xfail(strict=True, raises=AssertionError, reason=reason)]
    return pytest.param(rule, context, *values, id=f"{rule}|{context}", marks=marks)


def _check(rule: str, context: str, kept_by: list[str]) -> None:
    """Pass when no path keeps the secret. A known leak fails as its xfail
    expects only on exactly the paths it was found on: on any other set, the
    case fails outright, so the entry is brought up to date."""
    known = KNOWN_LEAKS.get((rule, context))
    if known is not None and kept_by and set(kept_by) != set(known):
        pytest.fail(
            f"{rule}|{context} was known to leak on {', '.join(known)}; "
            f"it leaks on {', '.join(kept_by)} now",
            pytrace=False,
        )
    assert not kept_by, f"kept by {', '.join(kept_by)}"


def _shape_params():
    for rule, unit, secret in _cases():
        # `X_Cookie: sid=...` and `X_Authorization: Bearer ...` are masked
        # too: the value, not the name, keeps prose safe.
        for context, prefix in PREFIXES.items():
            yield _param(rule, context, prefix + unit + " end", secret)
        for context, suffix in SUFFIXES.items():
            yield _param(rule, context, "see " + unit + suffix, secret)
        if rule in GLUED_MASKED:
            yield _param(rule, "glued", "key" + unit, secret)
    for rule, (name, sep, header) in CARRIERS.items():
        if not header:
            continue
        after = sep.split(":", 1)[1].lstrip()
        for context, ws in (("sep-tab", "\t"), ("sep-nl", "\n  ")):
            yield _param(rule, context, f"see {name}:{ws}{after}{VALUE}", VALUE)


@pytest.mark.parametrize("rule,context,raw,secret", list(_shape_params()))
def test_every_write_path_masks_the_secret(rule, context, raw, secret):
    _check(rule, context, _leaks(raw, secret))


def _dict_key_params():
    for rule, (name, sep, _) in CARRIERS.items():
        value = re.split("[:=]", sep, maxsplit=1)[1].lstrip() + VALUE
        yield _param(rule, "dict-key", name, value)


@pytest.mark.parametrize("rule,context,name,value", list(_dict_key_params()))
def test_a_secret_under_its_name_as_a_dict_key_is_masked(rule, context, name, value):
    _check(rule, context, _dict_key_leaks(name, value, VALUE))


NUMBER = 482139


@pytest.mark.parametrize(
    "name",
    ["api_key", "GITHUB_TOKEN", "password", "x-api-key", "DB_PASS", "master_key"],
)
def test_a_number_under_a_secret_name_is_masked(name):
    """A PIN-like integer (four digits or more) under a credential's name,
    as a dict value, is masked on every path."""
    assert not _dict_key_leaks(name, NUMBER, str(NUMBER))
