"""OTLP ``AnyValue`` / ``KeyValue`` decoding — no IR, no loss model, no OTel.

Split out of :mod:`benchflow.trajectories.ir_from_otel` unchanged. It is the one
part of that edge with no dependency in either direction: it turns the protobuf
JSON attribute wrappers into plain Python values and reports what a plain map
could not hold, and it knows nothing about the canonical IR, about loss records
or about semantic conventions. Keeping it separate makes the edge's remaining
code about mapping rather than about parsing, and leaves the decoder reusable by
a future outbound edge.

**Faithful is not the same as invertible.** :func:`decode_attributes` reports
``faithful=False`` when a *value* would be lost or collapsed — a duplicate key,
an empty ``AnyValue``, a type this reader cannot name. It says nothing about the
wire *spelling*: the canonical protobuf JSON mapping writes an ``int64`` as a
string, much of the ecosystem writes it as a number, and both decode to the same
Python ``int``. The wrapper type and that spelling are gone from the map either
way. Callers that care declare it; see `docs/trace-interop.md` §8.11.

Part of the unwired IR family (`docs/trace-interop.md` §8.7): nothing outside it
may import this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any, cast

ANY_VALUE_MEMBERS = frozenset(
    {
        "stringValue",
        "boolValue",
        "intValue",
        "doubleValue",
        "arrayValue",
        "kvlistValue",
        "bytesValue",
    }
)
NON_FINITE_DOUBLES = {"NaN": float("nan"), "Infinity": float("inf")}


@dataclass
class Attributes:
    """A decoded attribute map, plus whether decoding was faithful."""

    values: dict[str, Any] = dataclass_field(default_factory=dict)
    faithful: bool = True
    notes: list[str] = dataclass_field(default_factory=list)

    def text(self, key: str) -> str | None:
        """The value at *key* when it is a string, else ``None``."""
        value = self.values.get(key)
        return value if isinstance(value, str) else None


def decode_any_value(raw: Any) -> tuple[bool, Any]:
    """Decode one OTLP ``AnyValue`` wrapper.

    Returns ``(decoded, value)``. ``decoded`` is ``False`` whenever the result
    would not be a faithful representation, which the caller turns into a
    declared loss *and* into keeping the original list. The false cases are:

    - not an object, or more than one oneof member set (a oneof holds one);
    - an **empty** ``AnyValue`` (``{}``) — protobuf's "no member set", which a
      plain map cannot distinguish from a key whose value is absent;
    - a member name this reader does not know (a future ``AnyValue`` type);
    - a malformed payload for a known member — ``intValue`` that is not an
      integer, ``arrayValue``/``kvlistValue`` that is not the expected shape, or
      a nested value that is itself not decodable.

    ``bytesValue`` decodes to the base64 text verbatim rather than to bytes: the
    map has to stay JSON-serializable, and re-encoding is the one thing this
    module does not do. It is reported as unfaithful because in the map it is
    then indistinguishable from a string attribute.
    """
    if not isinstance(raw, dict):
        return False, None
    members = [key for key in raw if key in ANY_VALUE_MEMBERS]
    if len(members) != 1 or len(raw) != len(members):
        # Zero members is the empty AnyValue; more than one is not a oneof; an
        # extra unknown key is a member this reader cannot name.
        return False, None

    member = members[0]
    value = raw[member]

    if member == "stringValue":
        return (True, value) if isinstance(value, str) else (False, None)
    if member == "boolValue":
        return (True, value) if isinstance(value, bool) else (False, None)
    if member == "bytesValue":
        return (False, value) if isinstance(value, str) else (False, None)
    if member == "intValue":
        # int64 is a JSON string in the canonical mapping and a JSON number in
        # much of the wild; both are accepted by the pinned parser.
        if isinstance(value, bool):
            return False, None
        if isinstance(value, int):
            return True, value
        if isinstance(value, str):
            try:
                return True, int(value)
            except ValueError:
                return False, None
        return False, None
    if member == "doubleValue":
        if isinstance(value, bool):
            return False, None
        if isinstance(value, (int, float)):
            return True, float(value)
        if isinstance(value, str):
            # The canonical mapping spells the non-finite doubles as strings.
            if value in NON_FINITE_DOUBLES:
                return True, NON_FINITE_DOUBLES[value]
            if value.startswith("-") and value[1:] in NON_FINITE_DOUBLES:
                return True, -NON_FINITE_DOUBLES[value[1:]]
            return False, None
        return False, None
    if member == "arrayValue":
        if not isinstance(value, dict):
            return False, None
        items = value.get("values", [])
        if not isinstance(items, list):
            return False, None
        decoded: list[Any] = []
        for item in items:
            ok, inner = decode_any_value(item)
            if not ok:
                return False, None
            decoded.append(inner)
        return True, decoded
    # kvlistValue
    if not isinstance(value, dict):
        return False, None
    pairs = value.get("values", [])
    if not isinstance(pairs, list):
        return False, None
    mapping: dict[str, Any] = {}
    for pair in pairs:
        if not isinstance(pair, dict):
            return False, None
        key = pair.get("key")
        if not isinstance(key, str) or key in mapping:
            return False, None
        ok, inner = decode_any_value(pair.get("value"))
        if not ok:
            return False, None
        mapping[key] = inner
    return True, mapping


def decode_attributes(raw: Any) -> Attributes:
    """Decode an OTLP ``KeyValue`` list into a map, tracking what that cost.

    OTLP models attributes as a *list*, so duplicate keys are representable and
    a map cannot hold them. That, and every case :func:`decode_any_value`
    refuses, sets :attr:`Attributes.faithful` to ``False`` — which is the
    caller's signal to keep the original list beside the map and declare the
    collapse.
    """
    result = Attributes()
    if raw is None:
        return result
    if not isinstance(raw, list):
        result.faithful = False
        result.notes.append(
            f"attributes is {type(raw).__name__}, not a list of KeyValue objects"
        )
        return result

    for position, entry in enumerate(raw):
        if not isinstance(entry, dict):
            result.faithful = False
            result.notes.append(f"attributes[{position}] is not a JSON object")
            continue
        # ``cast`` only: ``isinstance`` above is the real check, but the
        # narrowed element type is not assignable to an invariant dict.
        pair = cast(dict[str, Any], entry)
        key = pair.get("key")
        if not isinstance(key, str):
            result.faithful = False
            result.notes.append(f"attributes[{position}] has no string key")
            continue
        if key in result.values:
            result.faithful = False
            result.notes.append(
                f"attributes[{position}] repeats key {key!r}; a map keeps one"
            )
        if "value" not in pair:
            result.faithful = False
            result.notes.append(f"attributes[{position}] ({key!r}) carries no value")
            result.values[key] = None
            continue
        ok, value = decode_any_value(pair["value"])
        if not ok:
            result.faithful = False
            result.notes.append(
                f"attributes[{position}] ({key!r}) is an AnyValue this reader "
                "cannot represent faithfully in a map"
            )
        result.values[key] = value
    return result
