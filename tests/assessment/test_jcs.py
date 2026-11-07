"""RFC 8785 canonicalization and the strict I-JSON reader the sidecar hashes with.

The expected bytes are RFC 8785's own examples (section 3.2.2, 3.2.3 and
Appendix B's IEEE 754 table). The builder acceptance run additionally
compares :func:`canonicalize` with two independent implementations.
"""

from __future__ import annotations

import json
import math
import struct

import pytest

from benchflow.assessment.errors import InvalidRecord
from benchflow.assessment.jcs import CanonicalizationError, canonicalize, loads_strict
from benchflow.assessment.records import integer


def _double(bits: str) -> float:
    return struct.unpack(">d", bytes.fromhex(bits))[0]


# RFC 8785 Appendix B: IEEE 754 bit patterns and their canonical text.
APPENDIX_B = [
    ("0000000000000000", "0"),
    ("8000000000000000", "0"),
    ("0000000000000001", "5e-324"),
    ("8000000000000001", "-5e-324"),
    ("7fefffffffffffff", "1.7976931348623157e+308"),
    ("ffefffffffffffff", "-1.7976931348623157e+308"),
    ("4340000000000000", "9007199254740992"),
    ("c340000000000000", "-9007199254740992"),
    ("4430000000000000", "295147905179352830000"),
    ("44b52d02c7e14af5", "9.999999999999997e+22"),
    ("44b52d02c7e14af6", "1e+23"),
    ("44b52d02c7e14af7", "1.0000000000000001e+23"),
    ("444b1ae4d6e2ef4e", "999999999999999700000"),
    ("444b1ae4d6e2ef4f", "999999999999999900000"),
    ("444b1ae4d6e2ef50", "1e+21"),
    ("3eb0c6f7a0b5ed8c", "9.999999999999997e-7"),
    ("3eb0c6f7a0b5ed8d", "0.000001"),
    ("41b3de4355555553", "333333333.3333332"),
    ("41b3de4355555554", "333333333.33333325"),
    ("41b3de4355555555", "333333333.3333333"),
    ("41b3de4355555556", "333333333.3333334"),
    ("41b3de4355555557", "333333333.33333343"),
    ("becbf647612f3696", "-0.0000033333333333333333"),
    ("43143ff3c1cb0959", "1424953923781206.2"),
]


@pytest.mark.parametrize(("bits", "expected"), APPENDIX_B)
def test_numbers_follow_rfc8785_appendix_b(bits: str, expected: str) -> None:
    assert canonicalize(_double(bits)) == expected.encode()


@pytest.mark.parametrize(
    "bits", ["7fffffffffffffff", "7ff0000000000000", "fff0000000000000"]
)
def test_nan_and_infinity_are_refused(bits: str) -> None:
    with pytest.raises(CanonicalizationError):
        canonicalize(_double(bits))


def test_rfc8785_section_3_2_2_example() -> None:
    value = loads_strict(
        '{"numbers": [333333333.33333329, 1E30, 4.50, 2e-3, 0.000000000000000000000000001],'
        ' "string": "\\u20ac$\\u000F\\u000aA\'\\u0042\\u0022\\u005c\\\\\\"\\/",'
        ' "literals": [null, true, false]}'
    )
    assert (
        canonicalize(value)
        == (
            '{"literals":[null,true,false],'
            '"numbers":[333333333.3333333,1e+30,4.5,0.002,1e-27],'
            '"string":"€$\\u000f\\nA\'B\\"\\\\\\\\\\"/"}'
        ).encode()
    )


def test_rfc8785_section_3_2_3_sorts_by_utf16_code_units() -> None:
    value = {
        "\u20ac": "Euro Sign",
        "\r": "Carriage Return",
        "\ufb33": "Hebrew Letter Dalet With Dagesh",
        "1": "One",
        "\U0001f600": "Emoji: Grinning Face",
        "\u0080": "Control",
        "\u00f6": "Latin Small Letter O With Diaeresis",
    }
    names = list(json.loads(canonicalize(value)))
    assert names == ["\r", "1", "\u0080", "\u00f6", "\u20ac", "\U0001f600", "\ufb33"]
    # Code-point order would put U+FB33 before U+1F600: UTF-16 puts the
    # surrogate pair (0xD83D...) first.
    assert sorted(value) != names


def test_strings_escape_exactly_as_json_stringify() -> None:
    assert canonicalize('\b\t\n\f\r\x00\x1f\x7f"\\/\u2028é') == (
        b'"\\b\\t\\n\\f\\r\\u0000\\u001f\x7f\\"\\\\/\xe2\x80\xa8\xc3\xa9"'
    )


def test_integers_and_floats_print_as_doubles() -> None:
    assert canonicalize([1.0, 0.5, 0.0, -0.0, 100, 1e21, 1e-7, 1e20, 0.1]) == (
        b"[1,0.5,0,0,100,1e+21,1e-7,100000000000000000000,0.1]"
    )


def test_out_of_range_values_are_refused() -> None:
    for value in (
        2**53,
        -(2**53),
        float("nan"),
        math.inf,
        "\ud800",
        {"\udc00": 1},
        b"x",
        {1: 2},
    ):
        with pytest.raises(CanonicalizationError):
            canonicalize(value)
    assert canonicalize(2**53 - 1) == b"9007199254740991"


@pytest.mark.parametrize(
    "raw",
    [
        b'{"a":1,"a":2}',
        b'{"a":{"b":1,"b":1}}',
        b"[NaN]",
        b"[Infinity]",
        b"[-Infinity]",
        b"[1e400]",
        b'["\\ud800"]',
        b'{"\\udfff":1}',
        b"\xef\xbb\xbf{}",
        b'{"a":"\xff"}',
        b"{",
        b"[" * 5000 + b"]" * 5000,
    ],
)
def test_strict_reader_refuses(raw: bytes) -> None:
    with pytest.raises(CanonicalizationError):
        loads_strict(raw)


def test_large_integer_literals_read_as_the_doubles_they_name() -> None:
    """Every JSON number is a double (RFC 8785 section 3.2.2.3).

    The RFC's reference test data has such literals, and JCS prints large
    doubles without an exponent (1e20 as 100000000000000000000), so the reader
    must take them back. A record field that must be an integer refuses them.
    """
    for text in (
        b"[9007199254740992]",
        b"[295147905179352830000]",
        b"[100000000000000000000]",
    ):
        value = loads_strict(text)
        assert isinstance(value[0], float)
        assert canonicalize(value) == text
    with pytest.raises(InvalidRecord, match="must be an integer"):
        integer(loads_strict(b"9007199254740993"), "event_seq")


def test_strict_reader_round_trips_canonical_bytes() -> None:
    data = b'{"a":[1,0.5,"\\u0001",null],"b":{"c":true}}'
    assert canonicalize(loads_strict(data)) == data
