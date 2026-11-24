"""Placeholder values and the forms an agent can print them in, shared by
the staged-secret masking rows (unit and deterministic tier). No BenchFlow
import, so the trial rows run, and fail for what they check, on a BenchFlow
without the registry.

The placeholders are not shaped like any known key family, so the pattern
redaction alone leaves them.
"""

from __future__ import annotations

import base64
import hashlib
import json
from urllib.parse import quote

# A staged token with a quote and a backslash, so its JSON-escaped form
# differs from it.
TOKEN = 'plc"q\\7Hq2Vx9Lm4Rt8Zw3Nf6Bd'
# A credentials file's secret field.
REFRESH = "rft0Kq8Ws2Ye5Ua9Ic3Ob7Pg1Xn"
# A value whose base64 holds + and / at every byte offset, so the URL-safe
# alphabet writes it differently.
URLSAFE = "~~~???~~~???~~~???~~~???~~~???"
# A long value, so its base64 wraps at 64 and 76 columns.
LONG = "lng" + "Qw3Er5Ty7Ui9Op2As4Df6Gh8Jk1Lz3Xc5Vb7Nm9" * 3
# A multi-line secret (a key block's body): each line is masked on its own.
BLOCK = "\n".join(
    [
        "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC7p",
        "q3Zt8Lk2Vb7Nc4Xm1Rs9Wd6Fh0Jg5Ty8Ui3Op2As7Df4Gh1Jk6Lz",
        "Hq9Wn2Ex5Rc8Tv1Yb4Um7In0Ok3Pl6Qa9Zs2Xd5Cf8Vg1Bh4Nj7",
    ]
)
# The proxy key the agent is given.
PROXY_VALUE = "rtk9Fz3Qw7Ep1Ry5Tu8Io2Pa6Sd4Fg"

STAGED = {
    "BF_STAGED_TOKEN": TOKEN,
    "auth.json.refresh_token": REFRESH,
    "URLSAFE_TOKEN": URLSAFE,
    "LONG_TOKEN": LONG,
    "KEY_BLOCK": BLOCK,
    "proxy master key": PROXY_VALUE,
}


def _b64(data: bytes, alphabet: str) -> str:
    encode = base64.urlsafe_b64encode if alphabet == "url" else base64.b64encode
    return encode(data).decode()


def _value_only(value: str, offset: int, alphabet: str) -> str:
    """The characters of *value*'s base64, at byte *offset* in a buffer,
    that depend on the value alone."""
    encoded = _b64(b"\0" * offset + value.encode(), alphabet)
    return encoded[4 if offset else 0 : ((offset + len(value.encode())) // 3) * 4]


def _wraps(value: str) -> list[str]:
    """*value*'s base64 wrapped at 64 (a key block) and 76 (MIME) columns,
    with \\n and \\r\\n."""
    encoded = _b64(value.encode(), "std")
    return [
        sep.join(encoded[i : i + width] for i in range(0, len(encoded), width))
        for width in (64, 76)
        for sep in ("\n", "\r\n")
    ]


def printed_forms(value: str) -> list[str]:
    """Each form an agent can print *value* in: as it is, JSON-escaped,
    %-encoded, base64 in both alphabets inside a larger buffer at each byte
    offset, and its base64 wrapped."""
    forms = [value, json.dumps(value)[1:-1], quote(value, safe="")]
    for offset in (0, 1, 2):
        for alphabet in ("std", "url"):
            data = b"ab"[:offset] + value.encode() + b"tail-bytes"
            forms.append(_b64(data, alphabet))
    return forms + _wraps(value)


def scan_forms(value: str) -> set[str]:
    """What an exact-value scan looks for: the value, its JSON escape and
    %-form, each line of a multi-line value, the value-only characters of
    its base64 at each offset in both alphabets, and every full line of each
    wrap."""
    found = {value, json.dumps(value)[1:-1], quote(value, safe="")}
    found |= {line for line in value.splitlines() if len(line) >= 16}
    for offset in (0, 1, 2):
        for alphabet in ("std", "url"):
            found.add(_value_only(value, offset, alphabet))
    for wrap in _wraps(value):
        lines = wrap.replace("\r\n", "\n").split("\n")
        found |= {line for line in lines[:-1] if len(line) >= 16}
    return {form for form in found if len(form) >= 16}


def _scan(text: str, values: dict[str, str] = STAGED) -> list[tuple[str, str, int]]:
    """Each form of each value found in *text*, as its name, the form's
    SHA-256 (12 hex digits) and its length: never the form itself."""
    return [
        (name, hashlib.sha256(form.encode()).hexdigest()[:12], len(form))
        for name, value in values.items()
        for form in sorted(scan_forms(value))
        if form in text
    ]
