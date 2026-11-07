"""RFC 8785 JSON Canonicalization Scheme (JCS), and the strict parser it needs.

Every digest the assessment sidecar records (event hashes, attempt ids,
output manifests, collector exports) is SHA-256 over :func:`canonicalize`.
``json.dumps(sort_keys=True)`` is not that form: it sorts keys by code point
rather than UTF-16 code unit, prints ``1.0`` and ``1e-07`` where JCS prints
``1`` and ``1e-7``, and escapes or keeps characters differently, so two
writers could hash one value two ways.

:func:`canonicalize` follows RFC 8785 section 3.2:

* objects sort their members by the UTF-16 code units of their names;
* strings escape ``"``, ``\\`` and the controls below U+0020 the way
  ECMAScript's ``JSON.stringify`` does, and keep every other character;
* numbers are IEEE 754 doubles printed as ECMAScript's
  ``Number.prototype.toString`` prints them (shortest round trip);
* the result is UTF-8 with no whitespace.

Input must be I-JSON (RFC 7493): no NaN or infinity, no lone surrogate in a
string or a name, and integers inside the range a double holds exactly. A
value outside that is refused, never coerced.

:func:`loads_strict` is the matching reader: UTF-8 only, no byte order mark,
no duplicate member names, no NaN or infinity and no lone surrogates. It reads
an integer beyond the double-exact range as the double it names, so canonical
output always reads back unchanged.

``tests/assessment/test_jcs.py`` checks both against RFC 8785's published
examples and number table; the builder acceptance run also compares them with
two independent implementations.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

__all__ = ["CanonicalizationError", "canonicalize", "loads_strict"]

# I-JSON's interoperable integer range (RFC 7493 section 2.2).
MAX_SAFE_INTEGER = 2**53 - 1
_SURROGATE = re.compile("[\ud800-\udfff]")
_ESCAPES = {
    '"': '\\"',
    "\\": "\\\\",
    "\b": "\\b",
    "\t": "\\t",
    "\n": "\\n",
    "\f": "\\f",
    "\r": "\\r",
}
_NEEDS_ESCAPE = re.compile('["\\\\\x00-\x1f]')


class CanonicalizationError(ValueError):
    """A value is not I-JSON, or bytes are not strict JSON."""


def canonicalize(value: Any) -> bytes:
    """The RFC 8785 canonical UTF-8 bytes of ``value``.

    ``value`` is built from ``dict`` (string keys), ``list``/``tuple``,
    ``str``, ``int``, ``float``, ``bool`` and ``None``. Anything else, a
    non-finite float, a lone surrogate or an integer outside
    ``±(2**53 - 1)`` raises :class:`CanonicalizationError`.
    """
    parts: list[str] = []
    _emit(value, parts, depth=0)
    return "".join(parts).encode("utf-8")


def _emit(value: Any, parts: list[str], depth: int) -> None:
    if depth > 128:
        raise CanonicalizationError("value nests deeper than 128 levels")
    if value is None:
        parts.append("null")
    elif value is True:
        parts.append("true")
    elif value is False:
        parts.append("false")
    elif isinstance(value, str):
        parts.append(_string(value))
    elif isinstance(value, int):
        if abs(value) > MAX_SAFE_INTEGER:
            raise CanonicalizationError(
                f"integer {value} is outside the I-JSON range ±(2**53 - 1)"
            )
        parts.append(_number(float(value)))
    elif isinstance(value, float):
        parts.append(_number(value))
    elif isinstance(value, dict):
        names: list[str] = []
        for key in value:
            if not isinstance(key, str):
                raise CanonicalizationError(
                    f"object member name {key!r} is not a string"
                )
            if _SURROGATE.search(key):
                raise CanonicalizationError(
                    "member name holds a lone surrogate (not I-JSON)"
                )
            names.append(key)
        names.sort(key=_utf16_order)
        parts.append("{")
        for index, name in enumerate(names):
            if index:
                parts.append(",")
            parts.append(_string(name))
            parts.append(":")
            _emit(value[name], parts, depth + 1)
        parts.append("}")
    elif isinstance(value, list | tuple):
        parts.append("[")
        for index, item in enumerate(value):
            if index:
                parts.append(",")
            _emit(item, parts, depth + 1)
        parts.append("]")
    else:
        raise CanonicalizationError(f"{type(value).__name__} is not a JSON value")


def _utf16_order(name: str) -> bytes:
    # Big-endian UTF-16 bytes compare exactly as their code units do.
    return name.encode("utf-16-be")


def _string(text: str) -> str:
    if _SURROGATE.search(text):
        raise CanonicalizationError("string holds a lone surrogate (not I-JSON)")
    return '"' + _NEEDS_ESCAPE.sub(_escape, text) + '"'


def _escape(match: re.Match[str]) -> str:
    char = match.group()
    return _ESCAPES.get(char) or f"\\u{ord(char):04x}"


def _number(value: float) -> str:
    """ECMAScript ``Number.prototype.toString(10)`` of a finite double."""
    if not math.isfinite(value):
        raise CanonicalizationError(f"{value!r} is not a finite number (not I-JSON)")
    if value == 0:
        return "0"  # also -0
    if value < 0:
        return "-" + _number(-value)
    # repr() is the shortest digit string that round-trips, the closest one
    # to the value when several have that length: the s, k and n of
    # ECMA-262's Number::toString.
    mantissa, _, exponent = repr(value).partition("e")
    whole, _, fraction = mantissa.partition(".")
    digits = whole + fraction
    n = len(whole) + (int(exponent) if exponent else 0)
    significant = digits.lstrip("0")
    n -= len(digits) - len(significant)
    digits = significant.rstrip("0")
    k = len(digits)
    if k <= n <= 21:
        return digits + "0" * (n - k)
    if 0 < n <= 21:
        return digits[:n] + "." + digits[n:]
    if -6 < n <= 0:
        return "0." + "0" * -n + digits
    sign = "+" if n - 1 >= 0 else "-"
    head = digits if k == 1 else digits[0] + "." + digits[1:]
    return f"{head}e{sign}{abs(n - 1)}"


def loads_strict(data: bytes | str) -> Any:
    """Parse strict I-JSON, refusing what :func:`canonicalize` could not hash.

    Refused: bytes that are not UTF-8, a byte order mark, duplicate member
    names, ``NaN``/``Infinity``, a number too large for a double and a lone
    surrogate in a string or a name. An integer literal outside
    ``±(2**53 - 1)`` reads as the double it names (a ``float``), as every JSON
    number does under RFC 8785, so :func:`canonicalize` of the result
    reproduces canonical input byte for byte.
    """
    if isinstance(data, bytes):
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CanonicalizationError(f"not UTF-8: {exc.reason}") from None
    else:
        text = data
    if text.startswith("\ufeff"):
        raise CanonicalizationError("starts with a byte order mark")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_members,
            parse_constant=_no_constant,
            parse_int=_int,
            parse_float=_float,
        )
    except RecursionError:
        raise CanonicalizationError("JSON nests too deeply") from None
    except json.JSONDecodeError as exc:
        raise CanonicalizationError(f"not JSON: {exc}") from None
    _check_strings(value, depth=0)
    return value


def _unique_members(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise CanonicalizationError(f"duplicate member name {name!r}")
        result[name] = value
    return result


def _no_constant(name: str) -> Any:
    raise CanonicalizationError(f"{name} is not a JSON number (not I-JSON)")


def _int(text: str) -> int | float:
    value = int(text)
    if abs(value) <= MAX_SAFE_INTEGER:
        return value
    # JSON numbers are doubles (RFC 8785 section 3.2.2.3): a larger integer
    # literal reads as the double it names, as JavaScript reads it, so every
    # canonical number (1e+21, 295147905179352830000) reads back unchanged.
    # A record field that must be an integer then refuses the float.
    return _float(text)


def _float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise CanonicalizationError(f"number {text} does not fit a double")
    return value


def _check_strings(value: Any, depth: int) -> None:
    if depth > 128:
        raise CanonicalizationError("JSON nests deeper than 128 levels")
    if isinstance(value, str):
        if _SURROGATE.search(value):
            raise CanonicalizationError("string holds a lone surrogate (not I-JSON)")
    elif isinstance(value, dict):
        for name, item in value.items():
            if _SURROGATE.search(name):
                raise CanonicalizationError(
                    "member name holds a lone surrogate (not I-JSON)"
                )
            _check_strings(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            _check_strings(item, depth + 1)
