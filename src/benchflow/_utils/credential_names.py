"""Whether one key or flag name names a credential or a setting.

Shared by the config override record (``config_override.py`` and the env
rule in ``config_redaction.py``) and the trajectory redactors, as one copy
of this file, so the two cannot drift. Keep it free of any import but the
standard library.

A name is split into words (at ``_ . - /`` and spaces, then at camelCase
humps). It names a credential when one of its words is a credential's,
alone (password, key, token, session, sid) or run together (authtoken,
GHTOKEN, JSESSIONID, myaccesskey). Four words can name a setting instead, by
the words beside them:

- ``tokens`` beside any companion word (max_tokens, input_tokens,
  cache_read_tokens, cached_tokens, p95_tokens): a count of them, which is
  never withheld, whatever number follows (input_tokens_2,
  output_tokens_v2); so is ``tokens`` just before a number and a time unit
  as the name's last words, a count by time window (tokens_1h, tokens_5m:
  _TOKEN_WINDOW_UNITS); bare ``tokens`` is not, nor with a number alone
  (TOKENS=a,b and TOKENS_2 can list secrets), and neither is a provider's
  word just before it (cached_refresh_tokens, access_tokens_1h) unless the
  name holds a pure count word (total_access_tokens);
- ``token`` or ``session`` beside a limit, count or time word
  (token_budget, max_token, session_timeout);
- ``token`` or ``tokens`` just after a tokenizer's word (eos_token,
  add_bos_token) or just before ``id`` / ``ids`` (pad_token_id), unless the
  name also has a provider's word (access_token_id, api_token_id);
- ``token`` or ``tokens`` just before a qualifier
  (BENCHFLOW_CAPTURE_TOKEN_LOGPROBS, cache_read_input_token_cost);
- ``credentials`` beside a count word (num_credentials, credentials_count).

Any credential word is a setting's just before a last word that names its
file, path, socket, directory, switch, check, logs, variable or staged
state (BENCHFLOW_HUB_KEY_FILE, SSH_AUTH_SOCK, AUTH_ENABLED, api_key_var),
or before such a word with a number between (KEY_2_FILE, TOKEN_2FA_ENABLED,
TOKEN_V2_PATH); a password's word is one just before a last ``hint``, a
number between or not (PASSWORD_HINT, PASSWORD_2_HINT); a header's word
(cookie, authorization) just before a last ``url``, ``name`` or ``policy``
(authorization_url, cookie_policy; not TOKEN_URL), ``session`` just before
``cookie`` being part of the cookie's name (session_cookie_name);
``resume`` just before a last ``session`` makes a switch (resume_session;
not resume_session_id, a session's id), ``env``
just before a last ``keys`` names env vars (agent_env_keys), and a few names
are settings by name
(_NOT_SECRET_NAMES, GIT_CONFIG_KEY_<n>, anything with ``tiktoken``, the
tokenizer library). GIT_CONFIG_VALUE_<n> is always a credential, and so is
a key's checksum (key_checksum): metadata about a secret. The export
redactor's own name for a replaced mapping key, spelled exactly
(``***REDACTED_KEY_1***``: _PLACEHOLDER_KEY), is a setting.

``pass`` is a password's word but for three exits: just before a metric's
or an outcome's word it is a metric or an outcome (pass_rate, pass_at_k,
pass@k, first_pass_yield, pass_fail, pass_through); as the last word after
a test's, an outcome's or a count's word, an outcome (ALL_TESTS_PASS,
deterministic_pass, fail_to_pass, n_pass); and just before a last-word
qualifier, a setting (PASS_FILE). Anywhere else it names a password
(DB_PASS, DB_PASS_2, PASS_VALUE, by_pass). bypass is one word, no
credential's, in ByPass too (_ONE_WORDS).

A price is a setting (_price_unit), but only after the credential-word and
companion rules: another credential word still decides (per_token_secret),
and a limit, count or time word beside a token makes it a setting whatever
``per`` says (MCP_PER_USER_TOKEN_DEFAULT_TTL). A token just after ``per``
(MODEL_COST_PER_TOKEN) or after ``per`` and a token's kind
(cost_per_reasoning_token) is a price's unit; so is any credential word
after ``per`` in a name with a price word (code_interpreter_cost_per_session,
usd_per_million_tokens), and such a name is a setting whatever its unit
(cost_per_request). The price rule yields when a provider's or a
credential's word sits between ``per`` and its unit: usd_per_access_token,
cost_per_auth_token and cost_per_api_access_token are credentials. user,
org, server, project and request are no such words: with a price word they
are units (cost_per_user_token), without one the name is a credential
(PER_USER_TOKEN), as any other ``per`` keeps a credential.

``pass@<digits>`` reads as ``pass_at_<digits>`` (pass@1, pass@10), a metric
like pass@k; ``pass@word`` stays a password.

One word that glues such words together counts the same (maxtokens,
maxoutputtokens, maxtoken, eostoken, eostokenid; not inputtoken or
accesstoken). Every other credential word (password, pass, pwd, secret, key,
apikey, auth, credential, bearer, cookie, sid, ...) always names a
credential, whatever is beside it, and so does a name with two credential
words (session_token_ttl). A name with no credential word is None.

A name that ends in a Cookie or an Authorization header's name after a
prefix (HTTP_COOKIE, REDIRECT_HTTP_AUTHORIZATION) is a credential's name,
and its entry is one by its value (entry_kind): anything but prose. So is
a primary or a secondary key's name (primary_key, COSMOS_PRIMARY_KEY): a key,
or anything but an identifier, a number or a list of them.
"""

from __future__ import annotations

import re
from typing import Literal

#: Words that name a credential, some written as one word (sessionid,
#: JSESSIONID); so does any word ending in one of _CREDENTIAL_ENDINGS
#: (authtoken, GHTOKEN, clientsecret, myapikey), which no ordinary word
#: does. A session cookie's name counts: connect.sid splits into connect
#: and sid.
_CREDENTIAL_WORDS = frozenset(
    {
        "secret",
        "token",
        "key",
        "password",
        "passwd",
        "pwd",
        "pass",
        "auth",
        "cookie",
        "credential",
        "bearer",
        "authorization",
        "passphrase",
        "passvalue",
        "session",
        "sessionid",
        "jsessionid",
        "sid",
    }
)
#: The keys run together, and the only list of them: each is an ending,
#: never a whole word only (xapikey, awsaccesskey, MYMASTERKEY,
#: myaccesskeyid). ``key`` is no ending (hotkey, monkey, turkey). Publickey
#: is one, as public_key is a credential's name too. No ordinary word ends
#: in one: tests/test_credential_names.py measures it on a corpus.
_GLUED_KEYS = (
    "apikey",
    "authkey",
    "privatekey",
    "secretkey",
    "accesskeyid",
    "accesskey",
    "masterkey",
    "signingkey",
    "sessionkey",
    "encryptionkey",
    "hmackey",
    "licensekey",
    "storagekey",
    "sharedkey",
    "clientkey",
    "rootkey",
    "publickey",
)
_CREDENTIAL_ENDINGS = ("token", "secret", "password", "passwd", *_GLUED_KEYS)
#: Beside ``tokens``, any of these makes a setting.
_COMPANION_WORDS = frozenset(
    {
        "max",
        "min",
        "limit",
        "budget",
        "num",
        "count",
        "total",
        "input",
        "output",
        "thinking",
        "reasoning",
        "context",
        "prompt",
        "completion",
        "timeout",
        "ttl",
        "seconds",
        "secs",
        "interval",
        "size",
        "length",
        "retries",
        "n",
        # the usage words of a token count (ruled in review, 2026-11-13)
        "cached",
        "native",
        "audio",
        "image",
        "text",
        "thought",
        "thoughts",
        "prediction",
        "accepted",
        "rejected",
        "candidates",
        "details",
        # and its statistics
        "final",
        "mean",
        "median",
        "avg",
        "sum",
        "subagent",
        "stream",
        "pre",
        "original",
    }
)
#: Two words in a row that are one companion, as the split leaves them:
#: cache_read_tokens, live_usage_tokens, and p95_tokens (p, then 95).
#: A percentile, as the split leaves it (p95: p, then 95).
_PERCENTILES = frozenset({("p", "50"), ("p", "90"), ("p", "95"), ("p", "99")})
_COMPANION_PAIRS = frozenset(
    {
        ("cache", "read"),
        ("cache", "write"),
        ("cache", "creation"),
        ("live", "usage"),
        *_PERCENTILES,
    }
)
#: Anywhere in a name, a word that makes ``tokens`` a count even after a
#: provider's word (total_access_tokens, n_api_tokens). Not cached: the
#: cached_refresh_tokens are tokens.
_PURE_COUNT_WORDS = frozenset(
    {"total", "n", "num", "count", "max", "min", "avg", "mean", "median", "sum"}
)
#: Beside ``token`` or ``session``, only these do (a limit, count or time).
_LIMIT_WORDS = frozenset(
    {
        "max",
        "min",
        "limit",
        "budget",
        "num",
        "count",
        "total",
        "timeout",
        "ttl",
        "seconds",
        "secs",
        "interval",
        "size",
        "length",
        "retries",
    }
)
#: Beside ``credentials``, only these do (a count).
_COUNT_WORDS = frozenset({"num", "count", "total", "max", "min", "limit"})
#: A tokenizer's special tokens, just before ``token``/``tokens``.
_TOKENIZER_WORDS = frozenset({"bos", "eos", "pad", "unk", "sep", "cls", "mask"})
_ID_WORDS = frozenset({"id", "ids"})
#: After plural ``tokens`` and a number, as the name's last words, the time
#: units of a count by time window (tokens_1h, tokens_5m: LiteLLM's Bedrock
#: cache-write counts by TTL, the only such names in the corpus; the lead's
#: ruling, 2026-11-14). Not a number alone (TOKENS_2) or other letters
#: (tokens_v2).
#: The redactor (trajectories/types.py) builds its patterns from this list:
#: change it here, never copy it.
_TOKEN_WINDOW_UNITS = ("m", "h")
#: The kinds of token a price is per (cost_per_reasoning_token).
#: The redactor (trajectories/types.py) builds its patterns from this list:
#: change it here, never copy it.
_TOKEN_KINDS = (
    "input",
    "output",
    "prompt",
    "completion",
    "reasoning",
    "cached",
    "audio",
    "image",
    "video",
    "text",
)
#: A word that makes a name a price's: whatever follows per is then the
#: price's unit (usd_per_request_token, code_interpreter_cost_per_session).
#: There is no list of units: any one word after per is one, and so is a
#: number with its unit (1k, two words after the split).
#: The redactor (trajectories/types.py) builds its patterns from this list:
#: change it here, never copy it.
_PRICE_WORDS = ("cost", "price", "usd")
#: Just after ``pass``, a metric's or an outcome's word (pass_rate,
#: pass_at_k, pass@k, pass_hat_k, pass_threshold): yield (first_pass_yield,
#: a first-pass yield), fail (pass_fail) and through (pass_through, a mode
#: that passes something on) among them.
#: The redactor (trajectories/types.py) builds its patterns from this list:
#: change it here, never copy it.
_PASS_METRIC_WORDS = (
    "rate",
    "at",
    "k",
    "count",
    "ratio",
    "fraction",
    "pct",
    "score",
    "hat",
    "threshold",
    "total",
    "deltas",
    "yield",
    "fail",
    "through",
)
#: Just before a last ``pass``, a test's or an outcome's word (ruled in
#: review, 2026-11-13): ALL_TESTS_PASS, deterministic_pass, is_pass; and
#: all, only and two (all_pass, only_pass, two_pass: the lead's ruling,
#: 2026-11-14), each holding a test result of up to five digits.
#: The redactor (trajectories/types.py) builds its patterns from this list:
#: change it here, never copy it.
_PASS_OUTCOME_WORDS = (
    "blockers",
    "deterministic",
    "first",
    "second",
    "final",
    "is",
    "expected",
    "did",
    "test",
    "tests",
    "check",
    "checks",
    "gate",
    "gates",
    "blocker",
    "smoke",
    "all",
    "only",
    "two",
)
#: Just before a last ``pass``, a pure count's word, as the redactor's
#: _TEST_COUNT_WORDS has them besides TEST and TESTS, and count and max
#: (n_pass, num_pass, total_pass, count_pass, max_pass: max by the lead's
#: ruling, 2026-11-14): such a name holds a count of any length
#: (_COUNT_RESULT).
#: The redactor (trajectories/types.py) builds its patterns from this list:
#: change it here, never copy it.
_PASS_COUNT_WORDS = ("n", "num", "total", "count", "max")
#: The two words before a last ``pass`` that are an outcome (fail_to_pass),
#: each as the words the split leaves, in order.
#: The redactor (trajectories/types.py) builds its patterns from this list:
#: change it here, never copy it.
_PASS_OUTCOME_PAIRS = (("fail", "to"), ("pass", "to"), ("all", "blockers"))
#: Just after ``token`` or ``tokens``, these make a setting. Each is in a
#: name BenchFlow reads that holds no secret: BENCHFLOW_CAPTURE_TOKEN_LOGPROBS
#: (a switch), LiteLLM's cache_read_input_token_cost (a price), Codex's
#: last_token_usage, the run record's token_capture and the loop sweep's
#: token_ratio_vs_baseline. Not url, env, name, value, key, secret or user: a
#: word is listed here or it is not a qualifier.
#: The redactor (trajectories/types.py) builds its patterns from this list:
#: change it here, never copy it.
_TOKEN_QUALIFIERS = ("capture", "cost", "logprobs", "usage", "ratio")
#: The last word of a name, just after its credential word, that makes the
#: name a setting's: it names the credential's file, path, socket,
#: directory, switch, check or logs, not the credential. Only as the last
#: word, and only just after the credential word or after a number between
#: them (_qualified: KEY_2_FILE, TOKEN_2FA_ENABLED; TOKEN_FILE_PATH is no
#: such name). Not checksum: a key's checksum is metadata about a secret.
#: var names the env var that holds it (api_key_var); staged says it was
#: put in place (credentials_staged).
#: The redactor (trajectories/types.py) builds its patterns from this list:
#: change it here, never copy it.
_LAST_QUALIFIERS = (
    "file",
    "path",
    "sock",
    "dir",
    "enabled",
    "check",
    "logs",
    "var",
    "staged",
)
#: The last word that makes a name a setting's just after a password's word
#: (_PASSWORD_WORDS, or a word that ends in password or passwd), as
#: _LAST_QUALIFIERS do after any credential word: a password's hint
#: (PASSWORD_HINT, passphrase_hint, PASSWORD_2_HINT), shown to its user by
#: design. Not after another credential word: OpenID Connect's
#: id_token_hint holds the ID token itself, and litellm's key_hint a key's
#: first letters.
#: The redactor (trajectories/types.py) builds its patterns from these
#: lists: change them here, never copy them.
_PASSWORD_QUALIFIERS = ("hint",)
_PASSWORD_WORDS = ("password", "passwd", "pwd", "pass", "passphrase")
#: The last word that makes a name a setting's right after a header's word
#: (cookie, authorization: the last words of _COOKIE_NAMES and
#: _AUTHORIZATION_NAMES), and only there: a
#: Cookie or an Authorization header's endpoint, name or rules
#: (authorization_url, cookie_name, cookie_policy). Not after any other
#: credential word: TOKEN_URL, TOKEN_NAME and secret_url stay credentials.
#: A URL's own credentials are withheld by the value's shape.
#: The redactor (trajectories/types.py) builds its patterns from this list:
#: change it here, never copy it.
_HEADER_QUALIFIERS = ("url", "name", "policy")
#: A name that holds a credentials file's path, or the file's content
#: inline: GOOGLE_APPLICATION_CREDENTIALS, a service account's JSON (the
#: lead's ruling, 2026-11-14: this name only). A setting by name, and an
#: entry's kind by its value (entry_kind): a path (_PATH_VALUE), null or an
#: empty value is a setting's; inline JSON (_INLINE_JSON), a mapping, a
#: list and anything else that is no path is a credential's (the review of
#: PR #192, L1: an opaque run, ``Bearer <value>``, a cookie pair or a key
#: under this name was kept).
#: The redactor (trajectories/types.py) builds its patterns from this list:
#: change it here, never copy it.
_PATH_OR_CONTENT_NAMES = (("google", "application", "credentials"),)
#: Inline JSON, read from the value's start: text that opens with ``{``
#: after any whitespace, or that holds the key ``"private_key"``, its
#: quotes escaped or not (``\"private_key\"`` in a JSON copy).
#: The redactor (trajectories/types.py) builds its patterns from this one:
#: change it here, never copy it.
_INLINE_JSON = r'\s*\{|.*?\\{0,8}"private_key\\{0,8}"'
_INLINE_JSON_VALUE = re.compile(_INLINE_JSON, re.DOTALL)
#: A file's path, as a whole value: no ``=`` and no line break anywhere (a
#: cookie pair and a padded key have ``=``), and either
#: anchored, spaces allowed: it starts with ``/``, ``\``, ``~/`` or
#: ``~user/``, ``./`` or ``../``, a drive (``C:\``, ``C:/``) or a variable
#: and a separator (``$HOME/``, ``${HOME}/``, ``%USERPROFILE%\``);
#: or relative, no spaces: segments joined by ``/`` or ``\``, the last one
#: ending in a file's extension, a letter and up to seven more letters or
#: digits (sa.json, creds/sa.json, .config/gcloud/adc.json).
#: "Holds a slash" is not the test: a base64 key and a cookie's ``path=/``
#: hold one. So a relative path with no extension (secrets/sa) is no path
#: here, and a key that starts with ``/`` and has no ``=`` is one (it reads
#: as /var/run/secrets/google/key does). _path_value also caps the length
#: at _PATH_MAX characters.
#: The redactor (trajectories/types.py) builds its patterns from this one:
#: change it here, never copy it.
_PATH_VALUE = (
    r"(?:[/\\]|~[^\\/\s=]*[\\/]|\.{1,2}[\\/]|[A-Za-z]:[\\/]"
    r"|\$\{?[A-Za-z_][A-Za-z0-9_]*\}?[\\/]|%[A-Za-z_][A-Za-z0-9_]*%[\\/])"
    r"[^=\r\n\x00]*"
    r"|(?:[^\s=\\/]+[\\/])*[^\s=\\/]*\.[A-Za-z][A-Za-z0-9]{0,7}"
)
_PATH_VALUE_SHAPE = re.compile(_PATH_VALUE)
_PATH_MAX = 4096
#: Names with a credential's word that hold no secret, as their words: the
#: D-Bus address, the path Google's libraries read a credentials file from
#: (by its value too: _PATH_OR_CONTENT_NAMES),
#: two BenchFlow switches, and a tokenizer's switches and counts; a list of
#: tests (tests_to_pass, beside FAIL_TO_PASS), a callback (on_pass) and two
#: passes of a computation (fidelity_pass, norm_pass) by the lead's ruling
#: of 2026-11-14.
#: GIT_CONFIG_KEY_<n> (a git config key's name; GIT_CONFIG_VALUE_<n> holds
#: the value) is one too, for any n.
_NOT_SECRET_NAMES = frozenset(
    {
        ("dbus", "session", "bus", "address"),
        ("google", "application", "credentials"),
        ("benchflow", "hub", "key", "no", "checksum"),
        ("benchflow", "task", "toml", "allow", "unknown", "keys"),
        ("add", "special", "tokens"),
        ("return", "assistant", "tokens", "mask"),
        ("generate", "tokens"),
        ("custom", "trainer", "pretokenized", "trainable", "tokens"),
        ("fail", "to", "pass"),
        ("pass", "to", "pass"),
        ("pass", "to", "fail"),
        ("first", "pass", "iteration"),
        ("tokens", "to", "pass"),
        ("tests", "to", "pass"),
        ("on", "pass"),
        ("fidelity", "pass"),
        ("norm", "pass"),
        ("attempt", "key"),
        ("idempotency", "key"),
        ("atif", "session", "id"),
        # The record of which operator credentials a trial staged (names,
        # never values), and the run flag that accepts task processes beside
        # them.
        ("operator", "credentials"),
        ("allow", "task", "processes", "with", "credentials"),
        # The names the operator scoped to the run (--scoped-credential).
        ("scoped", "credentials"),
        ("scoped", "credential"),
    }
)
#: A provider's word. Anywhere in the name it keeps the tokenizer and id
#: rules from making a setting (an access token's id identifies a
#: credential), but for those rules ``id`` is not one: it is their own word
#: (token_id). Just before ``tokens`` it makes the name a list of tokens, not
#: a count (cached_refresh_tokens, id_tokens).
_PROVIDER_WORDS = frozenset(
    {
        "access",
        "api",
        "auth",
        "refresh",
        "secret",
        "private",
        "client",
        "bearer",
        "oauth",
        "session",
        "id",
    }
)

# The words of a name: split on _ . - / and spaces, then at camelCase humps
# (githubToken, APIKey), so `author` and `hotkeys` are words of their own.
_PART = re.compile(r"[_.\-/\s]+")
_WORD = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")
#: Words that a camel hump does not split, as one part of a name holds them
#: (ByPass, isByPass: bypass, no credential's word; by_pass is two words).
#: The redactor (trajectories/types.py) splits with name_words: change this
#: here, never copy it.
_ONE_WORDS = frozenset({"bypass"})
# A run-together token word: <stem>token, <stem>tokens, <stem>tokenid(s).
_GLUED_TOKEN = re.compile(r"(?P<stem>[a-z]*?)token(?P<plural>s?)(?P<id>ids?)?")
# pass@k and pass@<digits>, a metric: read as pass_at_k and pass_at_<n>.
_PASS_AT = re.compile(r"(?<![A-Za-z])(pass)@(\d+|k)(?![A-Za-z0-9])", re.IGNORECASE)
#: Where a value ends: nothing of a value next, as the redactor's carriers
#: read one (trajectories/types.py _SECVAL): no value character, no
#: backslash's escape other than a line break's, a tab's or a quote's
#: (`\u0022`, `\&#34;`: a quote ends a word as one written does), and no escaped
#: quote or single quote that the value goes on after (`True\x41...`,
#: `none\ ...`, `abc\"def` are no switch and no word).
#: The redactor (trajectories/types.py) builds its patterns from this one:
#: change it here, never copy it.
_VALUE_END = (
    r"(?![^\"'\s,}\\*&]"
    r"|\\++(?!u00(?:22|27)|[xX](?:22|27)|u\{0*(?:22|27)\}|0?4[27](?![0-7])"
    r"|&(?:quot|apos|#0*3[49]|#[xX]0*2[27]);)[^nrt\"'\r\n\\]"
    r"|(?:\\\\)*+(?:\\[\"']|')[^\"'\s,}\]):\\*&])"
)
#: A test result as a string: a count of up to five digits (a longer number
#: may be a numeric PIN), perhaps a decimal or one of two (12/15), or a
#: switch word, then up to three of ) ] . ; and nothing of a value after.
#: The redactor (trajectories/types.py) builds its patterns from this one:
#: change it here, never copy it.
_TEST_RESULT = (
    r"(?:-?\d{1,5}(?:\.\d{1,6})?(?:/\d{1,5})?"
    r"|(?i:true|false|yes|no|on|off|none|null))"
    r"[)\].;]{0,3}" + _VALUE_END
)
_TEST_RESULT_VALUE = re.compile(_TEST_RESULT)
#: The same shape with a count of any length (n_pass=123456): what a pass
#: name after a pure count word (_PASS_COUNT_WORDS) holds as no secret. The
#: decimal and the ratio's second number keep _TEST_RESULT's limits.
#: The redactor (trajectories/types.py) builds its patterns from this one:
#: change it here, never copy it.
_COUNT_RESULT = _TEST_RESULT.replace(r"-?\d{1,5}", r"-?\d+", 1)
_COUNT_RESULT_VALUE = re.compile(_COUNT_RESULT)
#: The Cookie and Authorization headers' names, as their words. A name
#: that is one whole (Cookie, Set-Cookie, Authorization,
#: Proxy-Authorization) names a credential; a name that ends in one after a
#: prefix (HTTP_COOKIE, REDIRECT_HTTP_AUTHORIZATION, has_authorization)
#: holds one unless its value is prose (entry_kind: _PROSE), unless it has
#: bearer and a token word in a row (X_BEARER_TOKEN_COOKIE).
#: The redactor (trajectories/types.py) builds its patterns from these
#: lists: change them here, never copy them.
_COOKIE_NAMES = (("cookie",), ("set", "cookie"))
_AUTHORIZATION_NAMES = (("authorization",), ("proxy", "authorization"))
#: A cookie pair's start (``sid=``, ``sid%3D``). No longer part of a
#: name's kind (a prefixed cookie name holds a secret unless its value is
#: prose, as an Authorization name does: the lead's ruling, 2026-11-14);
#: the redactor's Cookie carrier uses it to find where a Cookie header's
#: pairs start in running text, and how far they run across ``; ``.
#: The redactor (trajectories/types.py) builds its patterns from this one:
#: change it here, never copy it.
_COOKIE_PAIR = r"[^\s;=\"'\\*]{1,256}(?:=|%(?:25)?3[Dd])"
_COOKIE_PAIR_VALUE = re.compile(_COOKIE_PAIR)
#: Prose: a plain word (letters only, fifteen or fewer: ``required``,
#: ``none``, ``Bearer`` alone) or a test result (_TEST_RESULT: ``1``,
#: ``true``), perhaps closing a sentence or a bracket. The only value a
#: prefixed Cookie or Authorization name holds as no secret, as the whole
#: value: one of several words (``please sign in first``, ``Splunk
#: <token>``), cookie pairs and an opaque value are a secret's.
#: The redactor (trajectories/types.py) builds its patterns from this one:
#: change it here, never copy it.
_PROSE = r"(?:[A-Za-z]{1,15}[)\].;]{0,3}" + _VALUE_END + "|" + _TEST_RESULT + ")"
_PROSE_VALUE = re.compile(_PROSE)
#: A primary or a secondary key's name, as its last words: a table's key in
#: a schema (SQLAlchemy's primary_key=True, BigQuery's primaryKey, DynamoDB's
#: DataModelPrimaryKey) or an access key (Azure's AccessKeys.primary_key and
#: secondary_key, base64 of 44 or 88 characters). Such a name, with a prefix
#: or none (IsPrimaryKey, COSMOS_PRIMARY_KEY) and no other credential word,
#: holds a secret by its value (entry_kind: _IDENTIFIER, _SHORT_IDENTIFIER);
#: one with a word after (PRIMARY_KEY_2) is a credential's.
#: The redactor (trajectories/types.py) builds its patterns from this list:
#: change it here, never copy it.
_KEY_NAMES = (("primary", "key"), ("secondary", "key"))
#: A key's value: 24 or more key characters and no space (an Azure key, an
#: API key), the run the redactor masks after a primary or a secondary
#: key's name unless it is an identifier (_IDENTIFIER). The kind is
#: entry_kind's (_schema_value): anything but a schema's value is a secret.
#: The redactor (trajectories/types.py) builds its patterns from this one:
#: change it here, never copy it.
_KEY_VALUE = r"[A-Za-z0-9+/=_-]{24,}"
#: Code, as a whole value (the lead's ruling, 2026-11-14, after the delta
#: review of PR #192): what source code holds after a name that ends in an
#: Authorization header's name and is itself a name in code
#: (has_authorization, self.sourceAuthorization: _code_name). Only
#: - a dotted name of two to six parts (self.flag, config.auth.value);
#: - a call, a subscript or a generic, to its closing bracket (f(x), x[0],
#:   Optional[str], os.environ["X"], kwargs.pop("x", None)), with ``= `` and
#:   a literal after it or not (Optional[str] = None);
#: - a literal: True, False, None, true, false, null, a number of at most
#:   five digits, or a decimal;
#: - a builtin type's name (_CODE_TYPES), with a literal default or not.
#: A name or a part is at most 24 characters and a quoted name inside the
#: brackets at most 32, so a long opaque run is no part of code. Everything
#: else is no code: a bare word (flag, a1b2c3d4; prose keeps a word of
#: letters, see entry_kind), a quoted string, ``b'...'``, an f-string, a
#: lambda, ``[1, 2]``, ``{}``, six digits or more. Two shapes read as code and are
#: pinned as residuals: a dotted value of short parts (abc.def123) and a
#: call around a short quoted name (f("abc123")).
#: is_code_value also caps the length at _CODE_MAX characters.
#: The redactor (trajectories/types.py) builds its patterns from this one:
#: change it here, never copy it.
_CODE_TYPES = ("bool", "str", "int", "float", "dict", "list")
_CODE_NAME = r"[A-Za-z_][A-Za-z0-9_]{0,23}(?![A-Za-z0-9_])"
_CODE_NUMBER = r"-?(?:[0-9]{1,5}(?![0-9])(?:\.[0-9]{1,9}(?![0-9]))?)(?![A-Za-z_])"
_CODE_LITERAL = (
    rf"(?:(?:True|False|None|true|false|null)(?![A-Za-z0-9_])|{_CODE_NUMBER})"
)
_CODE_QUOTED = r"(?:\"[A-Za-z_][A-Za-z0-9_.-]{0,31}\"|'[A-Za-z_][A-Za-z0-9_.-]{0,31}')"
_CODE_INNER = (
    rf"(?:{_CODE_NAME}|{_CODE_NUMBER}|{_CODE_QUOTED}|[.,:|*=\[\]() \t]){{0,64}}"
)
_CODE_VALUE = (
    rf"{_CODE_NAME}(?:\.{_CODE_NAME}){{1,5}}"
    rf"|{_CODE_NAME}(?:\.{_CODE_NAME}){{0,5}}[(\[]{_CODE_INNER}[)\]](?: = {_CODE_LITERAL})?"
    rf"|{_CODE_LITERAL}"
    rf"|(?:{'|'.join(_CODE_TYPES)})(?![A-Za-z0-9_])(?: = {_CODE_LITERAL})?"
)
_CODE_VALUE_SHAPE = re.compile(_CODE_VALUE)
_CODE_MAX = 256
#: A name in code: letters, digits, ``_`` and ``.`` only, with a lower-case
#: letter (has_authorization, sourceAuthorization, self.x_authorization).
#: HTTP_AUTHORIZATION (an env var) and x-proxy-authorization (a header) are
#: none, and keep the prose rule.
_CODE_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")
#: A bare identifier: one name, no dot (see _own_words).
_BARE_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
#: An identifier of any length: words of
#: letters joined by single ``_`` (external_subscription_id), or one run of
#: letters, camel case included (distributionHostedZoneId). No digit, and
#: no ``+``, ``/``, ``=``, ``-`` or ``.``: base64 and API keys hold them
#: (``-`` and ``_`` in base64url, ``-`` in sk-... keys). What a primary or
#: a secondary key's name holds as a schema's, whatever its length.
#: The redactor (trajectories/types.py) builds its patterns from this one:
#: change it here, never copy it.
_IDENTIFIER = r"[A-Za-z]+(?:_[A-Za-z]+)*"
_IDENTIFIER_VALUE = re.compile(_IDENTIFIER)
#: A short identifier, under 24 characters, digits, ``.`` and ``-``
#: allowed (``id``, ``user_id2``, ``table.id``): a schema's value too, as
#: are a number, a switch, null, or a list of these and of identifiers.
#: The redactor (trajectories/types.py) builds its patterns from this one:
#: change it here, never copy it.
_SHORT_IDENTIFIER = r"[A-Za-z_][A-Za-z0-9_.-]{0,22}"
_SHORT_IDENTIFIER_VALUE = re.compile(_SHORT_IDENTIFIER)
_NUMBER_VALUE = re.compile(r"-?\d+(?:\.\d+)?")
_SCHEMA_ITEM = rf'(?:"(?:{_SHORT_IDENTIFIER}|{_IDENTIFIER})"|-?\d+(?:\.\d+)?)'
_SCHEMA_LIST_VALUE = re.compile(
    rf"\[\s*(?:{_SCHEMA_ITEM}(?:\s*,\s*{_SCHEMA_ITEM})*)?\s*\]"
)
#: A switch word, the only value a passphrase's flag holds as no secret
#: (has_passphrase=false, after one of _FLAG_WORDS): true, false, yes, no,
#: on, off, 0, 1 or none, in
#: any case. Unlike _TEST_RESULT it takes no other count, no decimal or
#: ratio and no trailing ) ] . ; (passphrase=12345 is a numeric passphrase),
#: and not the word null. The redactor (trajectories/types.py) builds its
#: patterns from this one: change it here, never copy it.
_SWITCH = r"(?i:true|false|yes|no|on|off|0|1|none)"
_SWITCH_VALUE = re.compile(_SWITCH)
#: Just before a last ``passphrase``, a word that makes the name a flag's
#: (has_passphrase, requires_passphrase, no_passphrase), which holds a
#: switch as no secret; a bare passphrase name (SSH_KEY_PASSPHRASE,
#: PASSPHRASE) holds a passphrase whatever it is.
#: The redactor (trajectories/types.py) builds its patterns from this list:
#: change it here, never copy it.
_FLAG_WORDS = (
    "has",
    "with",
    "use",
    "uses",
    "needs",
    "need",
    "require",
    "requires",
    "is",
    "no",
    "skip",
    "enable",
    "enabled",
)
# A JSON list of strings, numbers, switch words or null, as a string.
_JSON_ITEM = r'(?:"(?:[^"\\]|\\.)*"|-?\d+(?:\.\d+)?|true|false|null)'
_JSON_LIST_VALUE = re.compile(
    rf"\s*\[\s*(?:{_JSON_ITEM}(?:\s*,\s*{_JSON_ITEM})*)?\s*\]\s*"
)


#: The export redactor's own name for a mapping key it replaced: a key that
#: was itself a credential is renamed to this, numbered (publish/redact.py
#: builds it with placeholder_key). Exactly this spelling, asterisks
#: included, is a setting: a record that is
#: redacted again keeps what its placeholder keys hold. Its words read as a
#: numbered key's (redacted, key, 1), so REDACTED_KEY_1 and KEY_1 without
#: the asterisks stay credentials.
_PLACEHOLDER_KEY = "***REDACTED_KEY_{}***"
_PLACEHOLDER_KEY_NAME = re.compile(
    re.escape(_PLACEHOLDER_KEY).replace(re.escape("{}"), "[0-9]+")
)


def placeholder_key(index: int) -> str:
    """The name the export redactor gives the *index*-th mapping key it
    replaced (_PLACEHOLDER_KEY): the one place that spelling is built."""
    return _PLACEHOLDER_KEY.format(index)


def name_words(name: str) -> list[str]:
    """The words of *name*, lower-cased, in order: the one word splitter of
    the classifier and the redactor (the lead's ruling, 2026-11-14; the
    redactor reads ``"_".join(name_words(name))`` where it read its own
    snake case). Split at ``_ . - /`` and spaces, then at camelCase humps
    (HTTPCookie, AWSSecretKey: http, cookie; aws, secret, key), and a run of
    digits is a word of its own wherever it sits (http2cookie: http, 2,
    cookie; TOKEN_V2: token, v, 2). Any other character between words is
    dropped (pass@k: pass, k). A letter outside ASCII whose case fold is
    ASCII letters is read as them, its case kept (a capital sharp s as SS,
    the fi ligature as fi); any other is dropped. A camel hump inside one
    of _ONE_WORDS is no break (ByPass: bypass)."""
    if not name.isascii():
        name = "".join(_ascii_letters(letter) for letter in name)
    words: list[str] = []
    for part in _PART.split(name):
        pieces = [word.lower() for word in _WORD.findall(part)]
        while pieces:
            if len(pieces) > 1 and pieces[0] + pieces[1] in _ONE_WORDS:
                pieces[:2] = [pieces[0] + pieces[1]]
            words.append(pieces.pop(0))
    return words


def _ascii_letters(letter: str) -> str:
    """*letter* as the ASCII letters its case fold gives (a sharp s: ss or
    SS, the fi ligature: fi, the Kelvin sign: K), or as it is when the fold
    gives none."""
    folded = letter.casefold()
    if letter.isascii() or not (folded.isascii() and folded.isalpha()):
        return letter
    return folded.upper() if letter.isupper() else folded


# The redactor imports the splitter by this name too.
_words = name_words


def _git_config(words: list[str], part: str) -> bool:
    """Whether *words* are GIT_CONFIG_<part>_<n>'s, for any n."""
    return (
        len(words) == 4 and words[:3] == ["git", "config", part] and words[3].isdigit()
    )


def _price_unit(words: list[str], index: int, priced: bool) -> bool:
    """Whether the credential word at *index* is a price's unit: a token
    just after ``per`` or after ``per`` and a token's kind, or, in a name
    with a price word (*priced*), any credential word just after ``per`` or
    after ``per`` and one word. A number with its unit (1k, which the split
    leaves as two words) counts as one word that is no kind. A provider's or
    a credential's word between ``per`` and the token makes no price
    (usd_per_access_token, cost_per_auth_token)."""
    token = words[index] in ("token", "tokens")
    if index and words[index - 1] == "per":
        follows = words[index + 1] if index + 1 < len(words) else ""
        if not token and follows in ("token", "tokens"):
            return False  # cost_per_auth_token: auth between per and token
        return token or priced
    if index > 1 and words[index - 2] == "per":
        between = words[index - 1]
        if between in _PROVIDER_WORDS or _credential_word(between):
            return False  # usd_per_access_token, price_per_refresh_token
        return (token and between in _TOKEN_KINDS) or priced
    number = index > 2 and words[index - 3] == "per" and words[index - 2].isdigit()
    return number and priced


#: A price's value: a sign, digits, a decimal part and an exponent, and
#: nothing else.
_PRICE_NUMBER = re.compile(
    r"[+-]?+(?:[0-9]++(?:\.[0-9]*+)?+|\.[0-9]++)(?:[eE][+-]?+[0-9]++)?+"
)


def _priced_name(name: str) -> bool:
    """Whether a price makes *name* a setting: one of its credential words is
    a price's unit (_price_unit), as name_kind reads it."""
    words = _words(_PASS_AT.sub(r"\1_at_\2", name))
    priced = "per" in words and any(_one_of(word, _PRICE_WORDS) for word in words)
    return any(
        _credential_word(word) and _price_unit(words, index, priced)
        for index, word in enumerate(words)
    )


def _price_value(value: object) -> bool:
    """Whether *value* is a number, which a price's name may hold: an int, a
    float or a string that is one, or a table of them (a mapping or a list
    whose every leaf is one: a usage block's ``usd_per_token: {input: ...}``)."""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        return _PRICE_NUMBER.fullmatch(value) is not None
    if isinstance(value, dict):
        return all(_price_value(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_price_value(item) for item in value)
    return False


def _qualified(words: list[str], index: int) -> bool:
    """Whether the credential word at *index* is a setting's by its name's
    last word (_LAST_QUALIFIERS; _PASSWORD_QUALIFIERS after a password's
    word): just after it, or after a number between them, alone or with one
    to three letters beside it as the split leaves them (KEY_2_FILE;
    TOKEN_2FA_ENABLED: 2, fa; TOKEN_V2_PATH: v, 2; PASSWORD_2_HINT). Any
    other word between makes no qualifier (TOKEN_FILE_PATH). A header's
    word is one by _HEADER_QUALIFIERS too, right after it only
    (authorization_url, cookie_policy)."""
    last = len(words) - 1
    if index >= last:
        return False
    word, qualifier = words[index], words[last]
    if qualifier in _HEADER_QUALIFIERS:
        headers = {names[-1] for names in (*_COOKIE_NAMES, *_AUTHORIZATION_NAMES)}
        return word in headers and index == last - 1
    password = word in _PASSWORD_WORDS or word.endswith(("password", "passwd"))
    if qualifier not in _LAST_QUALIFIERS and not (
        password and qualifier in _PASSWORD_QUALIFIERS
    ):
        return False
    gap = words[index + 1 : last]
    if len(gap) == 2 and not gap[0].isdigit():
        gap = gap[::-1]  # v, 2: the letters before the number
    if len(gap) == 2:
        return gap[0].isdigit() and gap[1].isalpha() and len(gap[1]) <= 3
    return not gap or gap[0].isdigit()


def _pass_outcome(words: list[str], index: int) -> bool:
    """Whether the word ``pass`` at *index* is a metric's or an outcome's,
    not a password's: just before a metric's word, or the last word after a
    test's, an outcome's or a count's word."""
    if index + 1 < len(words):
        return words[index + 1] in _PASS_METRIC_WORDS
    if not index:
        return False
    pair = tuple(words[index - 2 : index])
    before = words[index - 1]
    counted = before in _PASS_OUTCOME_WORDS or before in _PASS_COUNT_WORDS
    return counted or pair in _PASS_OUTCOME_PAIRS


def _one_of(word: str, words: tuple[str, ...] | frozenset[str]) -> bool:
    """Whether *word*, or *word* without a plural s, is one of *words*."""
    return word in words or (word.endswith("s") and word[:-1] in words)


def _credential_word(word: str) -> bool:
    # A word that ends in the Authorization header's word, glued onto a
    # prefix (proxyauthorization, PROXYAUTHORIZATION), is one too: every name
    # that ends in `authorization` is a credential's (the lead, 2026-11-16
    # 10:53Z: the rc twin of rel's F4).
    def one(text: str) -> bool:
        return (
            text in _CREDENTIAL_WORDS
            or text.endswith(_CREDENTIAL_ENDINGS)
            or text.endswith("authorization")
        )

    return one(word) or (word.endswith("s") and one(word[:-1]))


def _companions(text: str) -> list[str] | None:
    """*text* as companion words of three letters or more run together
    (``maxoutput`` is max, output), or None when it is not only those."""
    if not text:
        return []
    for word in sorted(_COMPANION_WORDS, key=lambda w: len(w), reverse=True):
        if len(word) > 2 and text.startswith(word):
            rest = _companions(text[len(word) :])
            if rest is not None:
                return [word, *rest]
    return None


def _glued_setting(word: str, provider: bool) -> bool | None:
    """For one run-together token word (``maxtokens``, ``eostokenid``,
    ``accesstoken``): whether it names a setting; None when *word* is not
    such a word."""
    match = _GLUED_TOKEN.fullmatch(word)
    if match is None or not (match["stem"] or match["id"]):
        return None
    stem, plural = match["stem"], bool(match["plural"])
    if (stem in _TOKENIZER_WORDS or not stem) and not provider:
        return True  # eostoken, eostokenid, tokenid
    if match["id"]:
        return False
    parts = _companions(stem)
    if not parts:
        return False  # accesstoken, authtoken, inputtoken: a credential
    return plural or any(part in _LIMIT_WORDS for part in parts)


def is_test_result(value: object, *, count: bool = False) -> bool:
    """Whether *value* is a test result, the only value a test's, an
    outcome's or a count's ``pass`` name holds as no password: null, a
    switch (a bool), a number whose text is a count (up to five digits, so
    123456 is no count; any length with *count*, the shape after a pure
    count word: _COUNT_RESULT), a string of that shape (``12``, ``12/15``,
    ``true``), a list or tuple of strings, numbers, switches and nulls, or a
    string that is such a JSON list. Anything else (``hunter2``, a mapping)
    is not."""
    shape = _COUNT_RESULT_VALUE if count else _TEST_RESULT_VALUE
    if value is None or isinstance(value, bool):
        return True
    if isinstance(value, list | tuple):
        return all(
            item is None or isinstance(item, str | int | float) for item in value
        )
    if isinstance(value, int | float):
        return shape.fullmatch(str(value)) is not None
    if isinstance(value, str):
        return bool(shape.fullmatch(value) or _JSON_LIST_VALUE.fullmatch(value))
    return False


def is_switch(value: object) -> bool:
    """Whether *value* is a switch: None, a bool, the int 0 or 1, or a
    string that is all _SWITCH (``False``, ``no``, ``1``)."""
    if value is None or isinstance(value, bool):
        return True
    if isinstance(value, int):
        return value in (0, 1)
    return isinstance(value, str) and _SWITCH_VALUE.fullmatch(value) is not None


def _passphrase_flag(name: str) -> bool:
    """Whether *name* is a passphrase's flag: a flag word (_FLAG_WORDS)
    just before its last ``passphrase`` (or ``pass`` ``phrase``)."""
    words = _words(name)
    if words[-1:] == ["passphrase"]:
        return len(words) > 1 and words[-2] in _FLAG_WORDS
    if words[-2:] == ["pass", "phrase"]:
        return len(words) > 2 and words[-3] in _FLAG_WORDS
    return False


def _header(name: str) -> str | None:
    """``cookie`` or ``authorization`` when *name* ends in that header's
    name after a prefix (HTTP_COOKIE, REDIRECT_HTTP_AUTHORIZATION,
    has_authorization): a name whose kind depends on its value. None for
    any other name, the header's own name whole among them (Set-Cookie),
    and for one with bearer and a token word in a row (BEARER_TOKEN_COOKIE),
    a credential's name whatever it holds."""
    words = _words(name)
    header = words[-1] if words else ""
    names = {"cookie": _COOKIE_NAMES, "authorization": _AUTHORIZATION_NAMES}
    if header not in names or tuple(words) in names[header]:
        return None
    for index in range(len(words) - 1):
        if words[index].endswith("bearer") and words[index + 1].startswith("token"):
            return None
    return header


def _key_name(name: str) -> bool:
    """Whether *name* ends in a primary or a secondary key's name
    (_KEY_NAMES), with a prefix or none and no other credential word:
    primary_key, primaryKey, IsPrimaryKey, COSMOS_PRIMARY_KEY; not
    PRIMARY_KEY_2 or token_primary_key."""
    words = _words(name)
    if tuple(words[-2:]) not in _KEY_NAMES:
        return False
    return not any(_credential_word(word) for word in words[:-1])


def _schema_value(value: object) -> bool:
    """Whether *value*, under a primary or a secondary key's name, is a
    schema's: null, a switch, a number (as text too), an identifier of any
    length (_IDENTIFIER: external_subscription_id) or a short one
    (_SHORT_IDENTIFIER: ``id``), or a list of these and of numbers
    (``["id", "ts"]``, as text too). Anything else is a secret's: a key
    (_KEY_VALUE), a long value with a digit, a mapping."""
    names = (_SHORT_IDENTIFIER_VALUE, _IDENTIFIER_VALUE)
    if value is None or isinstance(value, bool | int | float):
        return True
    if isinstance(value, list | tuple):
        return all(
            isinstance(item, int | float)
            or (isinstance(item, str) and any(n.fullmatch(item) for n in names))
            for item in value
        )
    return isinstance(value, str) and any(
        shape.fullmatch(value) for shape in (*names, _NUMBER_VALUE, _SCHEMA_LIST_VALUE)
    )


def _path_value(value: object) -> bool:
    """Whether *value* is a file's path (_PATH_VALUE), whole, of at most
    _PATH_MAX characters."""
    return (
        isinstance(value, str)
        and len(value) <= _PATH_MAX
        and _PATH_VALUE_SHAPE.fullmatch(value) is not None
    )


def is_code_value(value: object, name: str | None = None) -> bool:
    """Whether *value*, read as text (None as ``null``), is code as a whole
    (_CODE_VALUE), of at most _CODE_MAX characters. With the *name* it is
    held under, a bare identifier made of that name's own words is code
    too (_own_words): the name itself or a trailing run of its words
    (copy_source_authorization = source_authorization)."""
    text = "null" if value is None else value if isinstance(value, str) else None
    if text is None:
        if not isinstance(value, bool | int | float):
            return False
        text = str(value)
    if len(text) > _CODE_MAX:
        return False
    if _CODE_VALUE_SHAPE.fullmatch(text) is not None:
        return True
    return name is not None and _own_words(text, name)


def _own_words(text: str, name: str) -> bool:
    """Whether *text* is a bare identifier whose words are all of *name*'s
    words or a trailing run of them, in order: source_authorization and sourceAuthorization under
    copy_source_authorization. A value made of the name's own words holds
    no secret. authorization_source (another order), copy_authorization (a
    gap) and a longer name are none."""
    if _BARE_IDENTIFIER.fullmatch(text) is None:
        return False
    words, own = _words(text), _words(name)
    return 0 < len(words) <= len(own) and own[-len(words) :] == words


def _code_name(name: str) -> bool:
    """Whether *name* is a name in code (_CODE_IDENTIFIER) that is not all
    upper case: the names whose Authorization entries are code or a
    credential (has_authorization, sourceAuthorization)."""
    return _CODE_IDENTIFIER.fullmatch(name) is not None and name != name.upper()


def _outcome_pass(name: str) -> str | None:
    """``count`` when *name* ends in a ``pass`` just after a pure count's
    word (n_pass, TOTAL_PASS), ``outcome`` when after a test's or an
    outcome's (TESTS_PASS, FIRST_PASS): names whose kind depends on their
    value; None otherwise, a name by name among them (FAIL_TO_PASS)."""
    words = _words(_PASS_AT.sub(r"\1_at_\2", name))
    last = len(words) - 1
    if last < 1 or words[last] != "pass" or tuple(words) in _NOT_SECRET_NAMES:
        return None
    if not _pass_outcome(words, last):
        return None
    return "count" if words[last - 1] in _PASS_COUNT_WORDS else "outcome"


def is_code_authorization_name(name: str) -> bool:
    """Whether *name* is an Authorization header's name after a prefix that is
    itself a name in code (has_authorization, sourceAuthorization,
    note_Authorization: _code_name; not HTTP_AUTHORIZATION, not
    x-proxy-authorization, not the header's own name): every value under it
    is a credential's (the lead's ruling (b), 2026-11-15 01:3xZ). entry_kind
    and the text redactor's prose rules ask this."""
    return (
        _code_name(name)
        and _kept_name_kind(name) == "credential"
        and _header(name) == "authorization"
    )


#: An Authorization scheme's word alone, the text redactor's _AUTH_SCHEMES
#: and Digest, in any case.
_SCHEME_WORD = re.compile(
    r"(?:Bearer|Token|Basic|ApiKey|Negotiate|NTLM|Bot|Key|SSWS|Digest)", re.IGNORECASE
)


def flag_authorization_entry(name: str, value: object) -> bool:
    """Whether *name*, a flag's name that ends in `authorization`
    (has_authorization, requires_authorization, hasAuthorization), holds a
    boolean or null as such (an object's `true`, `false`, `null`), no
    credential. Every value as text, a switch word, a number or prose, is a
    credential's, as under every code Authorization name (the lead's ruling
    (b), 2026-11-15 01:3xZ; F2b, 2026-11-16 21:47Z: the gates show nothing a
    base masks)."""
    words = _words(name)
    return (
        len(words) >= 2
        and words[0] in ("has", "requires")
        and words[-1] == "authorization"
        and (value is None or isinstance(value, bool))
    )


def entry_kind(name: str, value: object) -> Literal["credential", "setting"] | None:
    """The kind of one entry, *name* holding *value*: name_kind's, except
     for a name that is a setting only as a test's, an outcome's or a count's
     ``pass`` (TESTS_PASS, ALL_BLOCKERS_PASS, FIRST_PASS, GATES_PASS, n_pass,
     TOTAL_PASS): a setting while *value* is a test result (is_test_result)
     and a credential otherwise (TESTS_PASS=hunter2). Every other name's kind
     does not depend on its value: a metric's name (pass_rate, pass_at_k,
     pass@1), a name by name (FAIL_TO_PASS, PASS_TO_PASS) and all the rest.
     config.json's env, the override record and the redactor's number and
     text rules ask this; a caller with no value asks name_kind.

     And a passphrase's flag (a flag word just before a last ``passphrase``:
     has_passphrase, requires_passphrase), a credential by name, is a
     setting's entry while it holds a switch (is_switch); any other value is
     a credential's, and so is anything under a bare passphrase name
     (SSH_KEY_PASSPHRASE=false, PASSPHRASE=no).

     And a name that ends in a Cookie or an Authorization header's name
     after a prefix (HTTP_COOKIE, REDIRECT_HTTP_AUTHORIZATION,
     has_authorization), a credential by name, is a credential's entry only
     by its value: anything but prose
     (_PROSE) as a whole, read as text, None as ``null``. ``required``,
     ``strict``, ``1`` and ``true`` are prose; cookie pairs (``theme=dark;
     sid=...``), an opaque value, ``please sign in first``, ``Bearer
     <token>``, 123456, a list and a mapping are not. The header's own name
     whole (Cookie, Set-Cookie, Authorization, Proxy-Authorization) and a
     name with bearer and a token word in a row are credentials whatever they
     hold.

     And where such an Authorization name is itself a name in code
     (has_authorization, sourceAuthorization: _code_name; not
     HTTP_AUTHORIZATION, not x-proxy-authorization), every entry is a
     credential's: code, the name's own words, prose, a switch and a count
     included (the lead's ruling (b), 2026-11-15 01:3xZ: leaks outrank
     over-masks; finer display of code returns in a later release).
     config.json, export, is_secret_field and the text redactor all ask
     this.

     And a primary or a secondary key's name (primary_key, secondaryKey,
     COSMOS_PRIMARY_KEY: _KEY_NAMES) holds a schema's key or an access key
    : a setting's entry while it holds null,
     a switch, a number, an identifier or a list of them (``id``,
     external_subscription_id, ``["id", "ts"]``), and a credential's for
     anything else, an Azure key (_KEY_VALUE) included.

     And a price's name (a token after ``per``, _price_unit:
     input_cost_per_token, usd_per_token, cost_per_1k_token,
     time_per_output_token), a setting by name, holds a number only:
     an int, a float or a string that is one (_PRICE_NUMBER: a sign, digits, a
     decimal part, an exponent, nothing else), or a table whose every leaf is
     one, is a setting's; any other value
     (a word, a key, a hex or base64 run, ``15usd``, None, a switch, a list) is
     a credential's, as 0.9.3 masks it.

     And GOOGLE_APPLICATION_CREDENTIALS (_PATH_OR_CONTENT_NAMES), a setting
     by name, holds a credentials file's path as a setting and anything
     else as a credential: None, an empty string and a path (_PATH_VALUE:
     /run/x.json, ~/adc.json, sa.json) are a setting's; the file's content
     (inline JSON, _INLINE_JSON: ``{`` first, or ``"private_key"`` in it; a
     mapping or a list) and every value that is no path (an opaque run,
     ``Bearer <value>``, a cookie pair, a key, a number, a switch) are a
     credential's (the review of PR #192, L1)."""
    kind = _kept_name_kind(name)
    if tuple(_words(name)) in _PATH_OR_CONTENT_NAMES:
        if value is None or (isinstance(value, str) and not value):
            return "setting"
        inline = isinstance(value, str) and _INLINE_JSON_VALUE.match(value)
        return "setting" if not inline and _path_value(value) else "credential"
    if flag_authorization_entry(name, value):
        return "setting"
    if is_code_authorization_name(name):
        return "credential"
    header = _header(name) if kind == "credential" else None
    if header == "authorization":
        # Every name that ends in `authorization` holds a credential, prose
        # included; an object's boolean or null
        # under a flag's name is no credential (flag_authorization_entry,
        # above), nor is a scheme word alone (`Bearer`), as under the
        # header's own names.
        alone = isinstance(value, str) and _SCHEME_WORD.fullmatch(value.strip())
        return "setting" if alone else "credential"
    if header is not None:
        text = "null" if value is None else str(value)
        return "setting" if _PROSE_VALUE.fullmatch(text) else "credential"
    if kind == "credential" and _key_name(name):
        return "setting" if _schema_value(value) else "credential"
    if kind == "setting" and _priced_name(name) and not _price_value(value):
        return "credential"  # cost_per_token=sk-...: a price holds a number
    held = _outcome_pass(name) if kind == "setting" else None
    if held and not is_test_result(value, count=held == "count"):
        return "credential"  # TESTS_PASS=hunter2; n_pass holds any count
    if kind == "credential" and _passphrase_flag(name) and is_switch(value):
        return "setting"
    return kind


#: name_kind's answer depends on the name alone, and entry_kind asks it
#: once per entry: the redactor asks entry_kind once per match, and a log
#: repeats a few names with a different value each time (the review of
#: PR #192, L3). So entry_kind keeps name_kind's answers, for at most
#: _NAME_CACHE_SIZE names of at most _NAME_CACHE_CHARS characters each: a
#: longer name is answered and not kept, and a full cache is emptied. It
#: holds names and their kinds, never a value. name_kind itself keeps
#: nothing (a test that swaps a rule list asks it). A plain dict, not
#: functools.lru_cache: this file imports only ``re`` and ``typing``.
_NAME_CACHE_SIZE = 4096
_NAME_CACHE_CHARS = 256
_NAME_KINDS: dict[str, Literal["credential", "setting"] | None] = {}


def _kept_name_kind(name: str) -> Literal["credential", "setting"] | None:
    """name_kind's answer for *name*, kept for a short name (_NAME_KINDS)."""
    if len(name) > _NAME_CACHE_CHARS:
        return name_kind(name)
    try:
        return _NAME_KINDS[name]
    except KeyError:
        kind = name_kind(name)
    if len(_NAME_KINDS) >= _NAME_CACHE_SIZE:
        _NAME_KINDS.clear()
    _NAME_KINDS[name] = kind
    return kind


def name_kind(name: str) -> Literal["credential", "setting"] | None:
    """``credential`` when *name* (one key or flag name) names a credential,
    ``setting`` when its credential word is a setting's by the words beside
    it (see the module docstring), None when it has no credential word. A
    caller that has the name's value asks :func:`entry_kind` instead."""
    if _PLACEHOLDER_KEY_NAME.fullmatch(name):
        return "setting"  # the export redactor's own name for a replaced key
    words = _words(_PASS_AT.sub(r"\1_at_\2", name))
    if _git_config(words, "value"):
        # actions/checkout passes its token in one: http.extraheader holds
        # Authorization: Bearer <token>.
        return "credential"
    if _git_config(words, "key") or tuple(words) in _NOT_SECRET_NAMES:
        return "setting"
    provider = any(word in _PROVIDER_WORDS - _ID_WORDS for word in words)
    counted = any(_one_of(word, _PURE_COUNT_WORDS) for word in words) or any(
        (words[index], words[index + 1]) in _PERCENTILES
        for index in range(len(words) - 1)
    )
    priced = "per" in words and any(_one_of(word, _PRICE_WORDS) for word in words)
    companions = [word for word in words if _one_of(word, _COMPANION_WORDS)]
    companions += [
        words[index]
        for index in range(len(words) - 1)
        if (words[index], words[index + 1]) in _COMPANION_PAIRS
    ]
    found = False
    # A credential word its own neighbours make a setting's (a price, a
    # qualifier, tiktoken) counts as none: a token word beside it is no
    # second one (input_cost_per_token_above_1k_tokens), and any other
    # credential word still decides (SESSION_TOKEN_FILE).
    settled = False
    for index, word in enumerate(words):
        before = words[index - 1] if index else ""
        after = words[index + 1] if index + 1 < len(words) else ""
        if word == "pass" and _pass_outcome(words, index):
            settled = True  # pass_rate, ALL_TESTS_PASS: a metric, an outcome
            continue
        if word == "session" and after == "cookie":
            continue  # session_cookie_name: the cookie's name, which decides
        if counted and after == "tokens" and word in _PROVIDER_WORDS:
            continue  # max_session_tokens: the kind of token it counts
        if word == "keys" and before == "env" and not after:
            settled = True  # agent_env_keys: the names of env vars
            continue
        if word == "tiktoken" or (_credential_word(word) and _qualified(words, index)):
            settled = True
            continue
        # The price rule, which only adds a setting: a credential word
        # elsewhere still decides, and so does a companion of the token.
        if _credential_word(word) and _price_unit(words, index, priced):
            settled = True
            continue
        if word in ("token", "tokens"):
            if _one_of(after, _TOKEN_QUALIFIERS):
                settled = True
                continue
            tokenizer = (
                before in _TOKENIZER_WORDS or after in _ID_WORDS
            ) and not provider
            if word == "tokens":
                listed = before in _PROVIDER_WORDS and not counted
                window = (
                    len(words) == index + 3
                    and words[index + 1].isdigit()
                    and words[index + 2] in _TOKEN_WINDOW_UNITS
                )  # tokens_1h: a count by time window
                setting = ((bool(companions) or window) and not listed) or tokenizer
            else:
                setting = tokenizer or any(_one_of(c, _LIMIT_WORDS) for c in companions)
        elif word in ("session", "sessions"):
            # resume_session: a switch; resume_session_id is a session's id
            setting = (before == "resume" and not after) or any(
                _one_of(c, _LIMIT_WORDS) for c in companions
            )
        elif word == "credentials":
            setting = any(_one_of(c, _COUNT_WORDS) for c in companions)
        else:
            glued = _glued_setting(word, provider)
            if glued is None:
                if not _credential_word(word):
                    continue
                setting = False
            else:
                setting = glued
        if not setting or found:
            return "credential"  # input_token; session_token_ttl
        found = True
    return "setting" if found or settled or priced else None
