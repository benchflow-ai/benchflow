"""Dependency-neutral redaction policy for persisted execution configuration.

Both solver and reviewer artifacts keep nonsecret execution settings so that
scoring-only resume preserves the inference protocol without persisting auth.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from benchflow._utils.env_values import env_value_text
from benchflow.trajectories.types import redact_trajectory_text

# Substrings that flag an env var name as secret-bearing for ``config.json``
# redaction. Matching is case-insensitive (callers ``.upper()`` the key first)
# and uses substring containment so derived names like ``MY_AUTH_HEADER``,
# ``SESSION_COOKIE``, or ``GH_TOKEN`` are caught. This stays a denylist (rather
# than an allowlist of safe keys) because agent env varies per agent — the
# union of safe keys is not knowable here — but the list now covers the common
# auth-bearing names that issue #410 called out (COOKIE, AUTHORIZATION, AUTH,
# BEARER, SESSION) on top of the original KEY/TOKEN/SECRET/PASSWORD/CREDENTIALS.
_SECRET_ENV_SUBSTRINGS: tuple[str, ...] = (
    "KEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSPHRASE",
    "CREDENTIALS",
    "COOKIE",
    "AUTHORIZATION",
    "AUTH",
    "BEARER",
    "SESSION",
)
_SECRET_URL_PATH_MARKERS: tuple[str, ...] = ("/__benchflow/",)


def _is_secret_env_key(name: str) -> bool:
    """Return True if *name* looks like it carries a secret value.

    Case-insensitive substring match against :data:`_SECRET_ENV_SUBSTRINGS`.
    Used by :func:`_write_config` to drop secret-bearing entries before
    persisting ``agent_env`` to the rollout's ``config.json``.
    """
    upper = name.upper()
    return any(s in upper for s in _SECRET_ENV_SUBSTRINGS)


def _is_secret_env_value(name: str, value: object) -> bool:
    """Return True if a normally public env value embeds a runtime secret.

    A value that is not a string (a YAML number or bool a caller did not
    normalize) is checked as the string the agent would get.
    """
    text = env_value_text(value)
    inline = _inline_config_structure(name, value)
    if inline is not None:
        # The structured walk checks every leaf. Applying text heuristics to
        # the serialized object would mistake env_http_headers references for
        # actual authorization headers.
        # JSONC comments do not affect execution, but the original text is
        # replayed when recorded. Never persist a credential hidden in trivia.
        secret_comment = name.upper() == "OPENCODE_CONFIG_CONTENT" and any(
            _is_secret_env_value("", comment) for comment in _jsonc_parts(text)[1]
        )
        return inline[0] != inline[1] or secret_comment
    # Inline harness JSON/TOML can contain credentials under a harmless env
    # name. Drop the whole value; redacted placeholders cannot safely replay.
    return any(marker in text for marker in _SECRET_URL_PATH_MARKERS) or (
        redact_trajectory_text(text) != text
    )


def _should_record_env_entry(name: str, value: object) -> bool:
    return not _is_secret_env_key(name) and not _is_secret_env_value(name, value)


# Known execution settings whose names also match the credential denylist.
# Numeric shape alone proves nothing: a passphrase, token or session id can
# be a short number.
_PLAIN_EXECUTION_ENV_NAMES = frozenset(
    {
        "MAX_THINKING_TOKENS",
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS",
        "MAX_TOKENS",
        "SESSION_TIMEOUT",
        "AUTH_ENABLED",
    }
)
_PLAIN_VALUE_RE = re.compile(
    r"-?\d{1,9}(\.\d{1,6})?|true|false|on|off|yes|no|none|null", re.IGNORECASE
)


def _is_plain_env_value(name: str, value: object) -> bool:
    """Whether this known execution setting holds a number or switch word.

    ``MAX_THINKING_TOKENS=32000`` changes what a run measures and is no key,
    so a resume records and compares it. Other credential names never qualify.
    A YAML number (``32000``) counts as its string (``"32000"``).
    """
    if name.upper() not in _PLAIN_EXECUTION_ENV_NAMES:
        return False
    return _PLAIN_VALUE_RE.fullmatch(env_value_text(value)) is not None


def _should_compare_env_entry(name: str, value: object) -> bool:
    """Whether a resume records and compares this env entry: the ones a
    trial's config.json records, and recognized numeric execution settings."""
    return _should_record_env_entry(name, value) or _is_plain_env_value(name, value)


_INLINE_JSON_CONFIG_NAMES = frozenset({"CODEX_CONFIG", "OPENCODE_CONFIG_CONTENT"})
# JSON schema fields are not environment-variable names: token limits,
# keybindings and env_key references all affect execution. Match credential
# fields exactly (ignoring spelling separators), and use the env policy only
# inside mappings whose keys really are environment or HTTP header names.
_INLINE_CREDENTIAL_FIELDS = frozenset(
    {
        "key",
        "apikey",
        "accesskeyid",
        "secretaccesskey",
        "privatekey",
        "secretkey",
        "token",
        "accesstoken",
        "refreshtoken",
        "idtoken",
        "oauthtoken",
        "authtoken",
        "sessiontoken",
        "bearertoken",
        "experimentalbearertoken",
        "secret",
        "clientsecret",
        "password",
        "passphrase",
        "credentials",
        "credential",
        "auth",
        "authorization",
        "proxyauthorization",
        "cookie",
        "setcookie",
        "bearer",
        "session",
    }
)
_INLINE_FREE_FORM_MAPPINGS = frozenset(
    {"env", "environment", "headers", "http_headers"}
)


def _inline_public_mapping_kind(name: str, path: tuple[str, ...]) -> str | None:
    """The public schema map owning this path, if any.

    OpenCode permission keys name tools and match patterns, even when a
    pattern is ``password`` or a custom tool is ``headers``. The same holds
    for tool switches and keybindings. Anchor these contexts to their schema
    paths, including per-agent and legacy mode settings; a provider option
    called ``permission`` does not exempt its credential fields (P2-CONFIRM-1).
    """
    if name.upper() != "OPENCODE_CONFIG_CONTENT" or not path:
        return None
    if path[0] in {"permission", "tools", "keybinds"}:
        return path[0]
    if (
        len(path) >= 3
        and path[0] in {"agent", "mode"}
        and path[2] in {"permission", "tools"}
    ):
        return path[2]
    return None


def _jsonc_parts(text: str) -> tuple[str, list[str]]:
    """JSON text and comments from OpenCode's JSONC grammar (P2-CONFIRM-4).

    Treat strings as indivisible JSON tokens; only comments and trailing
    commas outside them may change. Comments become whitespace, so removing
    one cannot join two invalid tokens into a valid value. The JSON decoder
    still validates the complete document and retains object insertion order.
    OpenCode's jsonc-parser enables comments and allowTrailingComma, not JSON5.
    """
    parts: list[str] = []
    comments: list[str] = []
    significant: list[int] = []
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(text):
        char = text[pos]
        if char == '"':
            _, end = decoder.raw_decode(text, pos)
            token = text[pos:end]
        elif text.startswith("//", pos):
            end = pos + 2
            while end < len(text) and text[end] not in "\r\n":
                end += 1
            comments.append(text[pos:end])
            token = " "
        elif text.startswith("/*", pos):
            close = text.find("*/", pos + 2)
            if close == -1:
                raise ValueError("Unterminated JSONC comment")
            end = close + 2
            comments.append(text[pos:end])
            token = " "
        else:
            token = char
            end = pos + 1
        if token not in {" ", "\t", "\r", "\n"}:
            if (
                token in {"}", "]"}
                and len(significant) >= 2
                and parts[significant[-1]] == ","
                and parts[significant[-2]] not in {"{", "[", ",", ":"}
            ):
                parts[significant.pop()] = " "
            significant.append(len(parts))
        parts.append(token)
        pos = end
    return "".join(parts), comments


def _inline_config_structure(
    name: str, value: object
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Original and public JSON/JSONC objects for a supported inline config."""
    if name.upper() not in _INLINE_JSON_CONFIG_NAMES:
        return None
    try:
        text = env_value_text(value)
        if name.upper() == "OPENCODE_CONFIG_CONTENT":
            text, _ = _jsonc_parts(text)
        parsed = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None

    def public(node: Any, *, path: tuple[str, ...] = ()) -> Any:
        if isinstance(node, dict):
            context = path[-1].lower() if path else ""
            public_names = _inline_public_mapping_kind(name, path) is not None
            free_form = not public_names and context in _INLINE_FREE_FORM_MAPPINGS
            return {
                key: public(child, path=(*path, key))
                for key, child in node.items()
                if not (
                    (
                        free_form
                        and _is_secret_env_key(key)
                        and not (
                            isinstance(child, (str, int, float, bool))
                            and _is_plain_env_value(key, child)
                        )
                    )
                    or (
                        not free_form
                        and not public_names
                        # These values name env vars, not header contents.
                        and context != "env_http_headers"
                        and not isinstance(child, dict)
                        and key.lower().replace("_", "").replace("-", "")
                        in _INLINE_CREDENTIAL_FIELDS
                    )
                )
                and not (isinstance(child, str) and _is_secret_env_value("", child))
            }
        if isinstance(node, list):
            return [
                public(child, path=path)
                for child in node
                if not (isinstance(child, str) and _is_secret_env_value("", child))
            ]
        return node

    return parsed, public(parsed)


def _inline_config_fingerprint(name: str, value: object) -> str | None:
    """Comparison metadata for supported inline configs, never replay settings.

    Remove credential leaves before hashing, so rotating a credential or
    reformatting JSON preserves identity while tool and grading settings do
    not disappear with an embedded credential.
    """
    structure = _inline_config_structure(name, value)
    if structure is None:
        return None

    def ordered_rules(node: Any, path: tuple[str, ...] = ()) -> Any:
        if isinstance(node, dict):
            items = [
                (key, ordered_rules(child, (*path, key))) for key, child in node.items()
            ]
            # OpenCode uses the last matching tool/pattern rule, including
            # legacy tools: write/edit/patch all assign permission.edit in
            # insertion order. Encode every schema policy map as ordered
            # pairs, independent of its tool names (P2-CONFIRM-3). Other JSON
            # object order stays irrelevant; this is never replay config.
            if _inline_public_mapping_kind(name, path) in {"permission", "tools"}:
                return items
            return dict(items)
        if isinstance(node, list):
            return [ordered_rules(child, path) for child in node]
        return node

    canonical = json.dumps(
        ordered_rules(structure[1]), sort_keys=True, separators=(",", ":")
    )
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


def _inline_env_fingerprints(env: dict[str, Any]) -> dict[str, str]:
    return {
        name: fingerprint
        for name, value in env.items()
        if (fingerprint := _inline_config_fingerprint(name, value)) is not None
    }
