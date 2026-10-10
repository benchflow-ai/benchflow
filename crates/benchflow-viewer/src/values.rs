//! How untrusted artifact fields become numbers, text, and flags (the small
//! helpers of 0.8's `payload.py`), and how sidecar files are read.
//!
//! Every artifact is untrusted input: a field of the wrong shape becomes
//! "absent" rather than an error, so a damaged trial still renders.

use std::path::Path;

use crate::py;
use crate::pyjson::{self, Int, Map, Value};

/// A finite number from a number or numeric text (`_finite_float`); booleans
/// are not numbers.
pub fn finite_float(value: &Value) -> Option<f64> {
    let number = match value {
        Value::Int(int) => int.to_f64()?,
        Value::Float(number) => *number,
        Value::Str(text) => py::parse_float(text)?,
        _ => return None,
    };
    number.is_finite().then_some(number)
}

/// A whole, non-negative count (`_nonnegative_int`): `3`, `3.0` and `"3"`
/// all read as 3.
pub fn nonnegative_int(value: &Value) -> Option<Int> {
    let number = finite_float(value)?;
    (number >= 0.0 && number.fract() == 0.0).then(|| Int::from_integral_f64(number))
}

/// A Python `int` that is not a `bool` (`isinstance(v, int)` checks).
pub fn exact_int(value: &Value) -> Option<Int> {
    match value {
        Value::Int(int) => Some(int.clone()),
        _ => None,
    }
}

/// Scalar identity text (`_optional_text`): containers, `null` and
/// non-finite numbers are absent; other scalars print as Python's `str()`.
pub fn optional_text(value: &Value) -> Option<String> {
    match value {
        Value::Null | Value::Array(_) | Value::Object(_) => None,
        Value::Float(number) if !number.is_finite() => None,
        other => Some(py::str_of(other)),
    }
}

/// Any value as display text (`_display_text`): strings as they are,
/// containers as JSON, `null` and non-finite numbers as empty text.
pub fn display_text(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::Str(text) => text.clone(),
        Value::Array(_) | Value::Object(_) => pyjson::dumps(value),
        Value::Float(number) if !number.is_finite() => String::new(),
        other => py::str_of(other),
    }
}

/// Error text when the value is truthy (`_error_text`). `NaN` is truthy
/// and displays as empty text, so it gives `Some("")`, as in 0.8.
pub fn error_text(value: &Value) -> Option<String> {
    py::truthy(value).then(|| display_text(value))
}

pub fn optional_bool(value: &Value) -> Option<bool> {
    match value {
        Value::Bool(flag) => Some(*flag),
        _ => None,
    }
}

/// A copy of an object with non-finite numbers replaced by `null`
/// (`_json_object`); anything else is an empty object.
pub fn json_object(value: &Value) -> Map {
    match value {
        Value::Object(map) => map
            .iter()
            .map(|(key, item)| (key.clone(), scrub(item)))
            .collect(),
        _ => Map::new(),
    }
}

fn scrub(value: &Value) -> Value {
    match value {
        Value::Float(number) if !number.is_finite() => Value::Null,
        Value::Array(items) => Value::Array(items.iter().map(scrub).collect()),
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(key, item)| (key.clone(), scrub(item)))
                .collect(),
        ),
        other => other.clone(),
    }
}

/// Non-empty scalar texts from a list (`_text_list`).
pub fn text_list(value: &Value) -> Vec<String> {
    value
        .as_array()
        .map(|items| {
            items
                .iter()
                .filter_map(optional_text)
                .filter(|text| !text.is_empty())
                .collect()
        })
        .unwrap_or_default()
}

/// A text file as Python's `read_text(errors="replace")` returns it:
/// invalid UTF-8 becomes U+FFFD and `\r\n` or `\r` line ends become `\n`.
pub fn read_text(path: &Path) -> Option<String> {
    let bytes = std::fs::read(path).ok()?;
    Some(decode_text(&bytes))
}

/// Bytes as Python's text mode decodes them (see [`read_text`]).
pub fn decode_text(bytes: &[u8]) -> String {
    let text = String::from_utf8_lossy(bytes);
    if text.contains('\r') {
        text.replace("\r\n", "\n").replace('\r', "\n")
    } else {
        text.into_owned()
    }
}

/// A JSON sidecar, or `None` when it is missing, unreadable, or malformed
/// (`_load_json`).
pub fn load_json(path: &Path) -> Option<Value> {
    pyjson::parse(&read_text(path)?)
}

/// A JSON sidecar that must be an object; anything else reads as `{}`.
pub fn load_object(path: &Path) -> Map {
    match load_json(path) {
        Some(Value::Object(map)) => map,
        _ => Map::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pyjson::parse;

    fn value(text: &str) -> Value {
        parse(text).unwrap()
    }

    #[test]
    fn numbers_come_from_numbers_and_numeric_text() {
        assert_eq!(finite_float(&value("1")), Some(1.0));
        assert_eq!(finite_float(&value("\"0.25\"")), Some(0.25));
        assert_eq!(finite_float(&value("true")), None);
        assert_eq!(finite_float(&value("NaN")), None);
        assert_eq!(finite_float(&value("\"inf\"")), None);
        assert_eq!(finite_float(&value("[1]")), None);
    }

    #[test]
    fn counts_must_be_whole_and_non_negative() {
        assert_eq!(nonnegative_int(&value("\"42\"")), Some(Int::Small(42)));
        assert_eq!(nonnegative_int(&value("3.0")), Some(Int::Small(3)));
        assert_eq!(nonnegative_int(&value("-1")), None);
        assert_eq!(nonnegative_int(&value("2.5")), None);
        assert_eq!(nonnegative_int(&value("-0.0")), Some(Int::Small(0)));
    }

    #[test]
    fn text_follows_python_str_and_json() {
        assert_eq!(optional_text(&value("1034")), Some("1034".into()));
        assert_eq!(optional_text(&value("[\"x\"]")), None);
        assert_eq!(
            display_text(&value("{\"nested\": \"name\"}")),
            "{\"nested\": \"name\"}"
        );
        assert_eq!(display_text(&value("true")), "True");
        assert_eq!(error_text(&value("0")), None);
        assert_eq!(error_text(&value("NaN")), Some(String::new()));
    }

    #[test]
    fn text_mode_translates_line_ends() {
        assert_eq!(decode_text(b"a\r\nb\rc\n\xff"), "a\nb\nc\n\u{fffd}");
    }
}
