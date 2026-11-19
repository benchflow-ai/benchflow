"""Header names that their value decides, and the one word splitter.

Part 2. Every value here is made up.

The classifier (``credential_names.entry_kind``) decides a name
that ends in a Cookie or an Authorization header's name after a prefix
(HTTP_COOKIE, REDIRECT_HTTP_AUTHORIZATION, has_cookie, httpCookie) by its
value: a setting while the whole value is prose (its ``_PROSE``: one plain
word of up to fifteen letters, a count or a switch word), a credential
otherwise. A header's url, name or policy is a setting (authorization_url,
cookie_policy, session_cookie_name), and the header's own name (Cookie,
Set-Cookie, Authorization, Proxy-Authorization) a credential whatever it
holds.

The redactor's whole-name rules ask it for every name that ends in a
header's word, and split a name with its ``name_words`` (a run of digits is
a word of its own: http2cookie). In text:

- the Cookie carrier masks a prefixed name's value unless it is prose (it
  masked cookie pairs only): ``HTTP_COOKIE=<opaque>`` is masked;
- a prefixed name's quoted value is judged whole in both carriers, and a
  quoted camel-case prefixed name is read too (``"sessionCookie": "..."``):
  an entry, and so every dict value (``_redact_field``), gives
  ``entry_kind``'s answer;
- an unquoted value is judged by its first word on its line
  (``has_cookie: please sign in`` stays), as running text gives a value no
  end, and an unquoted camel-case prefix is not read (in the corpus every
  such name before ``:`` or ``=`` is code);
- the header's own name is as it was: cookie pairs under Cookie, anything
  but prose under Authorization.
"""

from __future__ import annotations

import json

import pytest

from benchflow._utils import credential_names
from benchflow._utils.config_redaction import _should_record_env_entry
from benchflow._utils.credential_names import entry_kind, name_words
from benchflow.hosted_env import _masked_env_args
from benchflow.publish.redact import redact_value
from benchflow.trajectories import types
from benchflow.trajectories.types import (
    is_secret_field,
    is_secret_number,
    redact_trajectory_obj,
    redact_trajectory_text,
)
from tests import test_credential_names as classifier_tables
from tests._timing import assert_linear, catches_quadratic, linear_seconds
from tests.test_redaction_ansi_spellings import _repeat
from tests.test_redaction_escape_boundary import (
    BARE,
    BEARER,
    LAYERS,
    SID,
    _dict_leaks,
    _unchanged,
)
from tests.test_redaction_escape_boundary import _leaks as _boundary_leaks
from tests.test_redaction_shape_matrix import _dict_key_leaks
from tests.test_redaction_shape_matrix import _leaks as _path_leaks

MASK = "***REDACTED***"
EXPORT_MASK = "<XXX-benchflow-key-values-XXX>"
PAIRS = f"sid={SID}; theme=dark"
WORDS = "please sign in first"
# Prose: a plain word, a count, a switch word, a ratio, a scheme alone.
PROSE = ["required", "none", "1", "true", "12/15", "Bearer"]
# Anything else, and the part of it that must not survive.
NOT_PROSE = [
    (BARE, BARE),
    (PAIRS, SID),
    ("482139", "482139"),
    (WORDS, "sign in first"),
    (f"Bearer {BEARER}", BEARER),
]


def _code_name(name: str) -> bool:
    """A code Authorization name (A.1): lower case, joined by `_` or `.`."""
    return "authorization" in name and name.islower() and "-" not in name


def _code_authorization_name(name: str) -> bool:
    """A code Authorization name as the classifier reads it (_code_name:
    camel case included), under which every value is a credential's
    (the lead's ruling (b), 2026-11-15 01:3xZ)."""
    return credential_names._code_name(name) and name.lower().endswith("authorization")


def test_the_redactor_splits_a_name_as_the_classifier_does():
    """The lead's ruling (one splitter): _snake_name is name_words
    joined by `_`, on the classifier's own table of splits."""
    for name, words in classifier_tables.SPLITS:
        assert types._snake_name(name) == "_".join(words) == "_".join(name_words(name))
    assert types._snake_name("http2cookie") == "http_2_cookie"
    assert types._snake_name("ByPass") == "bypass"


# Prefixed names in each spelling: snake, kebab, camel, an acronym, a
# digit, a set- or proxy- header after a prefix.
SNAKE_PREFIXED = [
    "HTTP_COOKIE",
    "http_cookie",
    "has_cookie",
    "x-set-cookie",
    "REDIRECT_HTTP_COOKIE",
    "HTTP_AUTHORIZATION",
    "http_authorization",
    "REDIRECT_HTTP_AUTHORIZATION",
    "has_authorization",
    "x-proxy-authorization",
]
CAMEL_PREFIXED = [
    "httpCookie",
    "HTTPCookie",
    "http2cookie",
    "sessionCookie",
    "xSetCookie",
    "httpAuthorization",
    "basicAuthorization",
]
PREFIXED = SNAKE_PREFIXED + CAMEL_PREFIXED


# The lead, 2026-11-16 10:53Z (the rc twin of rel's F4): every name that ends
# in `authorization` is a carrier. Under a has_ or requires_ Authorization
# name only an object's boolean or null is no credential; every value as text
# is masked, as under every code Authorization name (F2b, 2026-11-16 21:47Z).
def _flag_entry(name, value) -> bool:
    words = name_words(name)
    return (
        len(words) >= 2
        and words[0] in ("has", "requires")
        and words[-1] == "authorization"
        and (value is None or isinstance(value, bool))
    )


def _ends_in_authorization(name: str) -> bool:
    return name.lower().replace("-", "_").endswith("authorization")


@pytest.mark.parametrize("name", PREFIXED)
def test_a_prefixed_header_name_keeps_prose_as_a_whole_name(name):
    """prose under a prefixed header name is the classifier's
    setting, as a key at export, in a hosted run's records, in trajectory
    files and as a number. Under a code Authorization name every value is a
    credential's (the lead's ruling (b), 2026-11-15 01:3xZ), masked in each of them."""
    if _code_authorization_name(name) or _ends_in_authorization(name):
        # Every name that ends in `authorization` is a carrier, prose
        # included; a flag's switch or number is kept (the lead, 2026-11-16 10:53Z: every name that ends in `authorization` is a carrier; a has_/requires_ name's switch, number or null is kept).
        for value in PROSE:
            if _flag_entry(name, value) or (
                value == "Bearer" and not _code_authorization_name(name)
            ):
                # A scheme word alone stays, as under the header's own names.
                assert entry_kind(name, value) == "setting", value
                continue
            assert entry_kind(name, value) == "credential", value
            assert is_secret_field(name, value)
            assert redact_value({name: value}) == ({name: EXPORT_MASK}, 1)
            assert _masked_env_args({name: value}) == {name: MASK}
            assert redact_trajectory_obj({name: value}) == {name: MASK}
        flag = _flag_entry(name, 1234)
        assert is_secret_number(name, 1234) is not flag
        assert redact_trajectory_obj({name: 1234}) == {name: 1234 if flag else MASK}
        return
    for value in PROSE:
        assert entry_kind(name, value) == "setting", value
        assert not is_secret_field(name, value)
        assert redact_value({name: value}) == ({name: value}, 0)
        if "authorization" in name.lower() and value != "Bearer":
            # A bare value under an Authorization name is masked where the
            # entry is read as text, prose included (the lead's ruling of 2026-11-16 08:46Z); the classifier,
            # which decides an export's keys, still calls it a setting. A
            # scheme word alone stays, as 0.9.3 keeps it.
            assert _masked_env_args({name: value}) == {name: MASK}
            assert redact_trajectory_obj({name: value}) == {name: MASK}
            continue
        # A test-result count under a code Authorization name is masked where
        # the entry is read as text, in a hosted run's records and trajectory
        # files (the lead, 21:21Z: "Test-result counts like `12/15` under a
        # credential name are masked"); the classifier, which decides an
        # export's keys, still calls it a setting.
        if value == "12/15" and _code_name(name):
            assert _masked_env_args({name: value}) == {name: "***REDACTED***"}
            assert redact_trajectory_obj({name: value}) == {name: "***REDACTED***"}
            continue
        assert _masked_env_args({name: value}) == {name: value}
        assert redact_trajectory_obj({name: value}) == {name: value}
    assert not is_secret_number(name, 1234)
    assert redact_trajectory_obj({name: 1234}) == {name: 1234}


@pytest.mark.parametrize("name", PREFIXED)
def test_a_prefixed_header_name_masks_anything_else_as_a_whole_name(name):
    """The lead's ruling (2026-11-14): an opaque value, cookie pairs,
    a long number, several words and a scheme with its token are a secret's
    under a prefixed name, a cookie name included (stricter than before:
    no `name=` pair needed). As a key, a header's key and a number, in every
    layer the helpers reach."""
    for value, secret in NOT_PROSE:
        if _flag_entry(name, value):  # a flag's number is kept
            assert entry_kind(name, value) == "setting", value
            continue
        assert entry_kind(name, value) == "credential", value
        assert is_secret_field(name, value)
        assert redact_value({name: value})[0] == {name: EXPORT_MASK}
        assert _masked_env_args({name: value}) == {name: MASK}
        assert _dict_leaks({name: value}, secret) == [], value
        assert _dict_key_leaks(name, value, secret) == [], value
    flag = _flag_entry(name, 482139)
    assert is_secret_number(name, 482139) is not flag
    assert redact_trajectory_obj({name: 482139}) == {name: 482139 if flag else MASK}
    if name in SNAKE_PREFIXED and not flag:
        # As JSON text too; a quoted camel-case key with an unquoted value
        # is not read in text (`{"sessionCookie": session_cookie}` is code).
        assert _dict_leaks({name: 482139}, "482139") == []


@pytest.mark.parametrize("name", SNAKE_PREFIXED)
def test_config_json_records_a_prefixed_header_entry_by_its_value(name):
    """config.json's env filter, which asks the classifier, keeps
    prose and withholds the rest; under a code Authorization name it
    withholds prose too (the lead's ruling (b), 2026-11-15 01:3xZ)."""
    authorization = _code_authorization_name(name) or _ends_in_authorization(name)
    for value in PROSE[:4]:
        recorded = _flag_entry(name, value) or not authorization
        assert _should_record_env_entry(name, value) is recorded, value
    for value, _secret in NOT_PROSE[:3]:
        assert _should_record_env_entry(name, value) is _flag_entry(name, value), value


def _text_forms(name: str, value: str) -> dict[str, str]:
    return {
        "env": f"{name}={value}",
        "export": f"export {name}={value}",
        "yaml": f"{name}: {value}",
        "after an escaped newline": f"out\\n{name}={value}",
        "after a colour code": f"out\x1b[32m{name}={value}",
        "entry": f'"{name}": "{value}"',
        "escaped entry": f'\\"{name}\\": \\"{value}\\"',
        "single-quoted entry": f"'{name}': '{value}'",
        "nested entry": json.dumps({"env": {name: value}}),
    }


QUOTED = {"entry", "escaped entry", "single-quoted entry", "nested entry"}
# What a prefixed cookie name's unquoted value after `:` or ` = ` is left
# as: an identifier (alone, or as a scheme's token) reads as code there, as
# after a lower-case pass name (`self.session_cookie = session_cookie`,
# `end_cookie = gen_cookie()`: 22 such code lines in the corpus).
CODE_LIKE = {BARE, f"Bearer {BEARER}"}
# Under a prefixed lower-case authorization name joined by `_` or `.`
# (part A.1) an identifier after `:` read as code until the lead's ruling after
# the delta review of PR #192: the classifier keeps code shape or prose only
# (_CODE_VALUE), so an identifier-shaped opaque value is masked in every form;
# `copy_source_authorization: Optional[str] = None` stays code.
AUTHORIZATION_CODE_LIKE: set[str] = set()


@pytest.mark.parametrize("name", SNAKE_PREFIXED)
def test_a_prefixed_header_name_in_text_keeps_prose_and_masks_the_rest(name):
    """in every text form and every layer, prose is kept and any
    other value masked, as the whole-name rule says. Two exceptions, stated
    in the report: several words unquoted are kept (text gives the value no
    end, and its first word is prose), but not after an upper-case name
    that ends in AUTHORIZATION; and under a cookie name, or a lower-case
    authorization name joined by `_` or `.`, an identifier after `:` or
    ` = ` reads as code (CODE_LIKE, AUTHORIZATION_CODE_LIKE); in an env line
    and quoted it is masked."""
    cookie = "cookie" in name.lower()
    # The lead's A.2 ruling (2026-11-14), "a deliberate over-mask
    # of prose after an Authorization name; one expected class": after an
    # upper-case env name that ends in AUTHORIZATION several words are
    # masked from the second word on, so the several-words exception is
    # not its (HTTP_AUTHORIZATION=please sign in first keeps `please`).
    spoken = name.isupper() and name.endswith("AUTHORIZATION")
    identifier = not cookie and name.islower() and "-" not in name
    for value in PROSE:
        for form, text in _text_forms(name, value).items():
            if _ends_in_authorization(name) and not (
                value == "Bearer" and not _code_authorization_name(name)
            ):
                # Every value under a name that ends in `authorization` is
                # masked but a flag's switch or number (the lead, 2026-11-16 10:53Z: every name that ends in `authorization` is a carrier; a has_/requires_ name's switch, number or null is kept).
                if _flag_entry(name, value):
                    assert value in redact_trajectory_text(text), (form, text)
                    continue
                assert not any(_unchanged(text).values()), (form, text)
                continue
            if "authorization" in name.lower() and value != "Bearer":
                # Prose under any Authorization name is masked (the lead's ruling of 2026-11-16 08:46Z).
                assert not any(_unchanged(text).values()), (form, text)
                continue
            assert _unchanged(text) == dict.fromkeys(LAYERS, True), (form, text)
    for value, secret in NOT_PROSE:
        for form, text in _text_forms(name, value).items():
            if _flag_entry(name, value):  # a flag's number is kept
                assert secret in redact_trajectory_text(text), (form, text)
                continue
            if (
                (value == WORDS and form not in QUOTED and not spoken)
                or (cookie and form == "yaml" and value in CODE_LIKE)
                or (identifier and form == "yaml" and value in AUTHORIZATION_CODE_LIKE)
            ):
                assert secret in redact_trajectory_text(text), (form, text)
                continue
            leaked = {*_boundary_leaks(text, secret), *_path_leaks(text, secret)}
            assert leaked == set(), (form, text)


@pytest.mark.parametrize("name", CAMEL_PREFIXED)
def test_a_camel_case_prefixed_header_name_is_read_in_a_quoted_entry(name):
    """a quoted key is no code, so an entry with a camel-case
    prefix gives the classifier's answer in every layer; unquoted, the name
    is not read (in the corpus every such name before `:` or `=` is code:
    SimpleCookie:, NetworkCookie=, BADCOOKIE= 23). A camel-case code
    Authorization name is read unquoted too, and every value under it is
    masked, prose included (the lead's ruling (b), 2026-11-15 01:3xZ)."""
    code = _code_authorization_name(name)
    for value in PROSE:
        for form in QUOTED:
            text = _text_forms(name, value)[form]
            assert _unchanged(text) == dict.fromkeys(LAYERS, not code), (form, text)
        if code:
            assert redact_trajectory_text(f"{name}: {value}") == f"{name}: {MASK}"
    for value, secret in NOT_PROSE:
        for form in QUOTED:
            text = _text_forms(name, value)[form]
            leaked = {*_boundary_leaks(text, secret), *_path_leaks(text, secret)}
            assert leaked == set(), (form, text)
        if code and value == WORDS:
            # Several words unquoted: the first is the value, as after a
            # snake-case code name (text gives the value no end).
            out = redact_trajectory_text(f"{name}: {value}")
            assert out == f"{name}: {MASK} sign in first", out
        elif code:
            assert secret not in redact_trajectory_text(f"{name}: {value}"), value
        else:
            assert redact_trajectory_text(f"{name}: {value}") == f"{name}: {value}"


# A header's url, name or policy, and the header's own name.
QUALIFIED = [
    "authorization_url",
    "AUTHORIZATION_URL",
    "authorizationUrl",
    "cookie_policy",
    "cookie_name",
    "session_cookie_name",
    "sessionCookieName",
    "http_cookie_name",
    "proxy_authorization_url",
]


@pytest.mark.parametrize("name", QUALIFIED)
def test_a_headers_url_name_or_policy_is_a_setting(name):
    """Its ruling, in the redactor: whatever it holds, in text and as
    a whole name, in every layer."""
    for value in (BARE, "strict", PAIRS):
        assert entry_kind(name, value) == "setting", value
        assert not is_secret_field(name, value)
        assert redact_value({name: value}) == ({name: value}, 0)
        for text in (f"{name}={value}", f'"{name}": "{value}"', f"{name}: {value}"):
            assert _unchanged(text) == dict.fromkeys(LAYERS, True), text


@pytest.mark.parametrize(
    "name",
    ["Cookie", "Set-Cookie", "set_cookie", "Authorization", "Proxy-Authorization"],
)
def test_the_headers_own_name_is_as_it_was(name):
    """the header's own name is a credential whatever it holds as a
    whole name; in text the Cookie header masks cookie pairs only and the
    Authorization header any value, prose included (the lead's ruling of 2026-11-16 08:46Z, it kept prose)."""
    for value in ("required", BARE, PAIRS):
        assert entry_kind(name, value) == "credential", value
        assert is_secret_field(name, value)
        assert redact_value({name: value})[0] == {name: EXPORT_MASK}
    cookie = "cookie" in name.lower()
    assert _unchanged(f"{name}: required") == dict.fromkeys(LAYERS, cookie)
    assert (redact_trajectory_text(f"{name}: {BARE}") == f"{name}: {BARE}") is cookie
    assert SID not in redact_trajectory_text(f"{name}: {PAIRS}")


def test_the_header_patterns_are_built_from_the_classifiers_names():
    """the names, the words, the starts and prose are the
    classifier's (_COOKIE_NAMES, _AUTHORIZATION_NAMES, _PROSE, _COOKIE_PAIR),
    no copy of a list."""
    from benchflow._utils import credential_names as c

    assert types._COOKIE_NAME == "(?:set[_-]cookie|cookie)"
    assert types._AUTHORIZATION_NAME == "(?:proxy[_-]authorization|authorization)"
    assert types._COOKIE_WORDS == ("cookie",)
    assert types._AUTHORIZATION_WORDS == ("authorization",)
    assert types._PROSE_VALUE is c._PROSE
    assert {
        names[-1] for names in (*c._COOKIE_NAMES, *c._AUTHORIZATION_NAMES)
    } == types._HEADER_WORDS
    cookie = next(p for p, _, _ in types._REDACTION_PATTERNS if p.words == ("cookie",))
    assert c._COOKIE_PAIR in cookie.pattern
    assert not hasattr(types, "_HEADER_FIELD")


def test_a_prefixed_header_value_is_masked_once():
    """a second pass finds nothing more to mask (the export
    redactor runs to a fixed point), a scheme's token included."""
    for text in (
        f"HTTP_COOKIE={BARE}",
        f'"has_cookie": "{WORDS}"',
        f'"httpAuthorization": "Bearer {BEARER}"',
        f'"has_authorization": "{WORDS}"',
        f'"sessionCookie": "{PAIRS}"',
    ):
        once = redact_trajectory_text(text)
        assert once != text and redact_trajectory_text(once) == once, text


@pytest.mark.parametrize(
    "unit",
    [
        f"HTTP_COOKIE={BARE}\n",
        "has_cookie: 1 ",
        f'"has_cookie": "{WORDS}", ',
        f'"sessionCookie": "{BARE}", ',
        f'"httpAuthorization": "Bearer {BEARER}", ',
        "x_cookie=",
        "set_cookie=",
        "a_set_cookie: ",
        '"' + "a" * 70 + 'Cookie": "',
        '"' + "a" * 70 + "Cookie",
        "has_authorization: please ",
        f'"has_authorization": "{BARE}", ',
        "Cookie: sid=a; ",
        '"x": "',
    ],
)
def test_header_names_repeated_stay_linear(unit):
    """each repeated to 50,000 and 200,000 characters: four times
    the text takes about four times as long, and a masked value is gone."""
    seconds, outs = linear_seconds(
        redact_trajectory_text, lambda size: _repeat(unit, size)
    )
    assert all(BARE not in out and BEARER not in out for out in outs)
    assert_linear(seconds)


# The lead's ruling (2026-11-14): the timing tests above call the
# shared check (tests/_timing.py). The positive control: the check, on one
# of each test's inputs at its sizes, fails once a quadratic term is added.
def test_the_header_name_timing_check_catches_a_quadratic():
    assert catches_quadratic(
        redact_trajectory_text, lambda size: _repeat("Cookie: sid=a; ", size)
    )
