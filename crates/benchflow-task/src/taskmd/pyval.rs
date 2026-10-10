//! Config values as the reference checker sees them.
//!
//! The reference is Python, so its checks lean on Python's rules: `True` is
//! an `int` unless a check says otherwise, an empty string is false, and `$`
//! in a pattern also matches before a final newline. These helpers apply the
//! same rules to `serde_json::Value`, so a check here accepts exactly what
//! the reference accepts.

use std::sync::LazyLock;

use regex::Regex;
use serde_json::{Map, Value};

static EMPTY: LazyLock<Map<String, Value>> = LazyLock::new(Map::new);

static DURATION: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(?:\d+(?:\.\d+)?h)?(?:\d+(?:\.\d+)?m)?(?:\d+(?:\.\d+)?s)?\n?\z").unwrap()
});
static DURATION_PART: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(\d+(?:\.\d+)?)([hms])").unwrap());
static SIZE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\d+(?:\.\d+)? ?(?:MB|GB|TB)\n?\z").unwrap());
static SIZE_PARTS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^(\d+(?:\.\d+)?) ?(MB|GB|TB)\n?\z").unwrap());
static COUNT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^(\d+)(?:\.(\d+))?([KM])\n?\z").unwrap());

/// The table at `path` under `v`, or an empty table when anything on the way
/// is missing or not a table (the reference's `_table`).
pub(crate) fn table<'a>(v: &'a Value, path: &[&str]) -> &'a Map<String, Value> {
    let mut cur = v;
    for key in path {
        match cur.get(key) {
            Some(next) => cur = next,
            None => return &EMPTY,
        }
    }
    obj(Some(cur))
}

/// A value as a table, or an empty table.
pub(crate) fn obj(v: Option<&Value>) -> &Map<String, Value> {
    match v {
        Some(Value::Object(m)) => m,
        _ => &EMPTY,
    }
}

/// Python's truth value: missing, null, false, 0, "", [] and {} are false.
pub(crate) fn truthy(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64().is_some_and(|f| f != 0.0),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(m)) => !m.is_empty(),
    }
}

/// A number's value (not a bool's): `isinstance(v, (int, float)) and not
/// isinstance(v, bool)`.
pub(crate) fn num(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        _ => None,
    }
}

/// An integer's value (not a bool's, not a float's): `isinstance(v, int) and
/// not isinstance(v, bool)`. As i128, so that u64 and i64 both fit.
pub(crate) fn int(v: &Value) -> Option<i128> {
    match v {
        Value::Number(n) => n
            .as_i64()
            .map(i128::from)
            .or_else(|| n.as_u64().map(i128::from)),
        _ => None,
    }
}

/// The reference's `_count`: a whole number, at least 1, and not a bool.
pub(crate) fn is_count(v: Option<&Value>) -> bool {
    v.and_then(int).is_some_and(|n| n >= 1)
}

/// A string that is not empty or blank (`isinstance(v, str) and v.strip()`).
pub(crate) fn nonblank(v: Option<&Value>) -> Option<&str> {
    match v {
        Some(Value::String(s)) if !s.trim().is_empty() => Some(s),
        _ => None,
    }
}

/// Python's `==` between JSON-like values: `1 == 1.0`, and `True == 1`.
pub(crate) fn py_eq(a: &Value, b: &Value) -> bool {
    fn as_num(v: &Value) -> Option<f64> {
        match v {
            Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
            Value::Number(n) => n.as_f64(),
            _ => None,
        }
    }
    match (a, b) {
        (Value::Null, Value::Null) => true,
        (Value::String(x), Value::String(y)) => x == y,
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| py_eq(p, q))
        }
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len() && x.iter().all(|(k, v)| y.get(k).is_some_and(|w| py_eq(v, w)))
        }
        _ => match (as_num(a), as_num(b)) {
            (Some(x), Some(y)) => x == y,
            _ => false,
        },
    }
}

/// A task.md duration in seconds (`"1h30m"` is 5400), or `None` when the
/// value is not a duration string (the reference's `duration_s`).
pub(crate) fn duration_s(v: Option<&Value>) -> Option<f64> {
    let Some(Value::String(s)) = v else {
        return None;
    };
    duration_str(s)
}

/// [`duration_s`] for a string.
pub(crate) fn duration_str(s: &str) -> Option<f64> {
    if !s.starts_with(|c: char| c.is_numeric()) || !DURATION.is_match(s) {
        return None;
    }
    let mut total = 0.0;
    for caps in DURATION_PART.captures_iter(s) {
        let n: f64 = caps[1].parse().ok()?;
        total += n * match &caps[2] {
            "h" => 3600.0,
            "m" => 60.0,
            _ => 1.0,
        };
    }
    Some(total)
}

/// Whether `v` is a task.md size string such as `"4 GB"`.
pub(crate) fn is_size(v: &Value) -> bool {
    matches!(v, Value::String(s) if SIZE.is_match(s))
}

/// A size string in megabytes, rounded up (1 GB is 1024 MB, as in Docker).
pub(crate) fn size_mb(v: &Value) -> Option<u64> {
    let Value::String(s) = v else {
        return None;
    };
    let caps = SIZE_PARTS.captures(s)?;
    let n: f64 = caps[1].parse().ok()?;
    let mult = match &caps[2] {
        "MB" => 1.0,
        "GB" => 1024.0,
        _ => 1024.0 * 1024.0,
    };
    Some((n * mult).ceil() as u64)
}

/// The reference's `count_value`: a whole number of at least 1, or a string
/// such as `"400K"` or `"2M"` that comes to one. `None` otherwise.
pub(crate) fn count_value(v: Option<&Value>) -> Option<u128> {
    match v? {
        Value::Number(_) => {
            let n = int(v?)?;
            (n >= 1).then_some(n as u128)
        }
        Value::String(s) => {
            let caps = COUNT.captures(s)?;
            let whole: u128 = caps[1].parse().ok()?;
            let frac = caps.get(2).map_or("", |m| m.as_str());
            let mult: u128 = if &caps[3] == "K" { 1_000 } else { 1_000_000 };
            let scale = 10u128.checked_pow(frac.len() as u32)?;
            let frac_n: u128 = if frac.is_empty() {
                0
            } else {
                frac.parse().ok()?
            };
            let scaled = whole
                .checked_mul(scale)?
                .checked_add(frac_n)?
                .checked_mul(mult)?;
            (scaled % scale == 0 && scaled / scale >= 1).then_some(scaled / scale)
        }
        _ => None,
    }
}

/// The reference's `_network_ok`: `"none"`, `"open"`, a list of strings, or
/// a table with exactly one key, `block`, holding a list.
pub(crate) fn network_ok(v: &Value) -> bool {
    match v {
        Value::String(s) => s == "none" || s == "open",
        Value::Array(items) => items.iter().all(Value::is_string),
        Value::Object(m) => m.len() == 1 && m.get("block").is_some_and(Value::is_array),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn durations() {
        assert_eq!(duration_s(Some(&json!("1h30m"))), Some(5400.0));
        assert_eq!(duration_s(Some(&json!("0.5s"))), Some(0.5));
        assert_eq!(duration_s(Some(&json!("10m\n"))), Some(600.0));
        assert_eq!(duration_s(Some(&json!("5"))), None);
        assert_eq!(duration_s(Some(&json!("m"))), None);
        assert_eq!(duration_s(Some(&json!(600))), None);
        assert_eq!(duration_s(None), None);
    }

    #[test]
    fn counts() {
        assert_eq!(count_value(Some(&json!("2M"))), Some(2_000_000));
        assert_eq!(count_value(Some(&json!("1.5K"))), Some(1_500));
        assert_eq!(count_value(Some(&json!("1.0005K"))), None);
        assert_eq!(count_value(Some(&json!(0))), None);
        assert_eq!(count_value(Some(&json!(true))), None);
        assert_eq!(count_value(Some(&json!(2.0))), None);
        assert_eq!(count_value(Some(&json!("0.0001K"))), None);
    }

    #[test]
    fn python_truth_and_equality() {
        assert!(!truthy(Some(&json!(""))));
        assert!(!truthy(Some(&json!(0))));
        assert!(truthy(Some(&json!([0]))));
        assert!(py_eq(&json!(1), &json!(1.0)));
        assert!(py_eq(&json!(false), &json!(0)));
        assert!(!py_eq(&json!("1"), &json!(1)));
        assert_eq!(int(&json!(3)), Some(3));
        assert_eq!(int(&json!(3.0)), None);
        assert_eq!(int(&json!(true)), None);
        assert_eq!(num(&json!(false)), None);
    }

    #[test]
    fn sizes_and_networks() {
        assert!(is_size(&json!("4 GB")) && is_size(&json!("512MB")) && !is_size(&json!(4096)));
        assert_eq!(size_mb(&json!("0.5 GB")), Some(512));
        assert!(network_ok(&json!([])) && network_ok(&json!({"block": [1]})));
        assert!(!network_ok(&json!("offline")) && !network_ok(&json!({"block": "x"})));
    }
}
