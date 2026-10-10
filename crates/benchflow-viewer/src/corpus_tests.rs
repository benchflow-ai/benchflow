//! Value-level goldens recorded from BenchFlow 0.8's Python (see
//! `tests/fixtures/generate.py`): float formatting, `float()` of text, JSON
//! read and written back, and event timestamps.

use std::cell::Cell;
use std::path::PathBuf;

use crate::py::parse_float;
use crate::pyjson::{Value, dumps, float_repr, parse};
use crate::timestamp::parse_ts_with;
use crate::timestamp::tests::los_angeles_2026;

fn golden(name: &str) -> Vec<serde_json::Value> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/golden")
        .join(name);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    serde_json::from_str(&text).unwrap()
}

fn pair(case: &serde_json::Value) -> (&str, &serde_json::Value) {
    (case[0].as_str().unwrap(), &case[1])
}

#[test]
fn floats_print_exactly_as_python_repr() {
    let cases = golden("floats.json");
    assert!(cases.len() > 1000);
    for case in &cases {
        let (bits, expected) = pair(case);
        let value = f64::from_bits(u64::from_str_radix(bits, 16).unwrap());
        assert_eq!(float_repr(value), expected.as_str().unwrap(), "bits {bits}");
    }
}

#[test]
fn float_text_reads_as_python_float() {
    for case in &golden("pyfloat.json") {
        let (text, expected) = pair(case);
        let actual = parse_float(text).map(|value| {
            if value.is_nan() {
                "nan".to_string()
            } else if value.is_infinite() {
                if value > 0.0 { "inf" } else { "-inf" }.to_string()
            } else {
                float_repr(value)
            }
        });
        assert_eq!(actual.as_deref(), expected.as_str(), "float({text:?})");
    }
}

#[test]
fn json_reads_and_writes_as_python_json() {
    for case in &golden("json.json") {
        let (text, expected) = pair(case);
        // 0.8's `_parse_json` returns `None` for a `null` document as for an
        // unreadable one; every caller needs a container, so both are absent.
        let actual = parse(text)
            .filter(|value| !value.is_null())
            .map(|value| dumps(&value));
        assert_eq!(actual.as_deref(), expected.as_str(), "json.loads({text:?})");
    }
}

#[test]
fn aware_timestamps_match_python() {
    let cases = golden("timestamps.json");
    assert!(cases.len() > 200);
    for case in &cases {
        let (text, expected) = pair(case);
        let asked_for_local_time = Cell::new(false);
        let offset = |_unix: i64| {
            asked_for_local_time.set(true);
            None
        };
        let actual = parse_ts_with(&Value::Str(text.to_string()), &offset);
        assert!(
            !asked_for_local_time.get(),
            "{text:?} read as a local time; Python reads it as aware or invalid"
        );
        assert_eq!(actual, expected.as_f64(), "{text:?}");
    }
}

#[test]
fn local_timestamps_match_python_in_los_angeles() {
    let mut checked = 0;
    for case in &golden("timestamps-la.json") {
        let (text, expected) = pair(case);
        // The stand-in time zone models 2026 only.
        if !text.starts_with("2026") {
            continue;
        }
        let actual = parse_ts_with(&Value::Str(text.to_string()), &los_angeles_2026);
        assert_eq!(actual, expected.as_f64(), "{text:?}");
        checked += 1;
    }
    assert!(checked > 20, "only {checked} cases checked");
}
