//! Reading config values the way Harbor and BenchFlow 0.8 read them.
//!
//! Both load `task.toml` (and 0.8's frontmatter) through pydantic models in
//! "lax" mode, which converts some values: `cpus = "2"` is 2, `cpus = 2.0`
//! is 2, `timeout_sec = 600` is 600.0, and `allow_internet = "yes"` is true.
//! Each function here accepts exactly what pydantic accepts for one field
//! type, so a task that loads there loads here, and returns a short reason
//! when the value is not of that type.

use std::collections::BTreeMap;

use serde_json::Value;

/// A short description of a value for error messages: `"four"`, `1.5`,
/// `a table`.
pub(crate) fn describe(v: &Value) -> String {
    match v {
        Value::Null => "nothing".into(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => format!("{s:?}"),
        Value::Array(_) => "a list".into(),
        Value::Object(_) => "a table".into(),
    }
}

/// A string field: only strings (pydantic does not turn numbers into strings).
pub(crate) fn string(v: &Value) -> Result<String, String> {
    match v {
        Value::String(s) => Ok(s.clone()),
        other => Err(format!("expected a string, found {}", describe(other))),
    }
}

/// An integer field: whole numbers, floats with no fractional part, booleans,
/// and strings that hold a whole number.
pub(crate) fn int(v: &Value) -> Result<i64, String> {
    let fail = || Err(format!("expected a whole number, found {}", describe(v)));
    match v {
        Value::Bool(b) => Ok(i64::from(*b)),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                return Ok(i);
            }
            // pydantic-core's `float_as_int`: whole, finite, and strictly
            // inside the 64-bit range.
            match n.as_f64() {
                Some(f) if f % 1.0 == 0.0 && (i64::MIN as f64) < f && f < (i64::MAX as f64) => {
                    Ok(f as i64)
                }
                _ => fail(),
            }
        }
        // Integers past 64 bits, which pydantic keeps, are refused here.
        Value::String(s) => str_as_int(s).map_or_else(fail, Ok),
        _ => fail(),
    }
}

// The string rules below are ported from pydantic-core 2.41.5 (the version
// pydantic 2.12.5 pins), `src/input/shared.rs`, and the integer syntax of
// jiter 0.11.1's `NumberInt::try_from`. Whitespace is Rust's `str::trim`
// there too, so the two agree on every kind of space.

/// pydantic-core's `str_as_int`: the text as a JSON integer, else the text
/// after `clean_int_str` as one.
fn str_as_int(s: &str) -> Option<i64> {
    json_int(s).or_else(|| json_int(&clean_int_str(s)?))
}

/// The whole text as a JSON integer (`-?(0|[1-9][0-9]*)`) that fits in 64
/// bits.
fn json_int(s: &str) -> Option<i64> {
    let digits = s.strip_prefix('-').unwrap_or(s);
    let well_formed = digits == "0"
        || (digits.starts_with(|c: char| ('1'..='9').contains(&c))
            && digits.bytes().all(|b| b.is_ascii_digit()));
    if well_formed { s.parse().ok() } else { None }
}

/// pydantic-core's `clean_int_str`: trim, drop one `+`, leading zeros, a
/// zero fraction (`.0`, `.000`) and underscores; `None` when nothing
/// changed or the text cannot be a number. It keeps pydantic-core's
/// quirks: a `-` after leading zeros starts the number (`"0-495"` is -495).
fn clean_int_str(text: &str) -> Option<String> {
    let len_before = text.len();
    let mut s = text.trim();
    if let Some(rest) = s.strip_prefix('+') {
        if rest.starts_with('-') {
            return None;
        }
        s = rest;
    }
    let mut negative = false;
    if let Some(rest) = s.strip_prefix('-') {
        if rest.starts_with(['-', '+']) {
            return None;
        }
        negative = true;
        s = rest;
    }
    s = strip_leading_zeros(s)?;
    if let Some(i) = s.find('.') {
        let fraction = &s[i + 1..];
        if !fraction.is_empty() && fraction.bytes().all(|b| b == b'0') {
            s = &s[..i];
        }
    }
    let sign = if negative { "-" } else { "" };
    if let Some(stripped) = strip_underscores(s) {
        return Some(format!("{sign}{stripped}"));
    }
    (len_before != s.len()).then(|| format!("{sign}{s}"))
}

/// pydantic-core's `strip_leading_zeros`: zeros (and underscores among
/// them) before the first other digit, `-` or `.` go; `None` when the text
/// does not start with a digit or `-`.
fn strip_leading_zeros(s: &str) -> Option<&str> {
    let mut chars = s.char_indices();
    match chars.next() {
        Some((_, '0')) => {}
        Some((_, c)) if c.is_ascii_digit() || c == '-' => return Some(s),
        _ => return None,
    }
    for (i, c) in chars {
        match c {
            '0' | '_' => {}
            '1'..='9' | '-' => return Some(&s[i..]),
            // Keep the character before the dot (a zero or an underscore).
            '.' => return Some(&s[i - 1..]),
            _ => return None,
        }
    }
    Some(&s[s.len() - 1..])
}

/// pydantic-core's `strip_underscores`: the text without its underscores,
/// `None` when it has none, or one starts or ends it, or two are together.
fn strip_underscores(s: &str) -> Option<String> {
    if s.starts_with('_') || s.ends_with('_') || !s.contains('_') || s.contains("__") {
        return None;
    }
    Some(s.replace('_', ""))
}

/// A float field: numbers, booleans, and strings that hold a number.
pub(crate) fn float(v: &Value) -> Result<f64, String> {
    match v {
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        Value::Number(n) => n
            .as_f64()
            .ok_or_else(|| format!("expected a number, found {n}")),
        // pydantic-core's `str_as_float`: the trimmed text as Rust reads a
        // float, else the untrimmed text without underscores (so `" 1_0"`
        // is refused).
        Value::String(s) => s
            .trim()
            .parse::<f64>()
            .ok()
            .or_else(|| strip_underscores(s)?.parse().ok())
            .ok_or_else(|| format!("expected a number, found {}", describe(v))),
        other => Err(format!("expected a number, found {}", describe(other))),
    }
}

/// A boolean field: booleans, 0 and 1, and the strings pydantic reads as
/// booleans (`"yes"`, `"off"`, `"t"`, ...).
pub(crate) fn boolean(v: &Value) -> Result<bool, String> {
    const TRUE: [&str; 6] = ["1", "on", "t", "true", "y", "yes"];
    const FALSE: [&str; 6] = ["0", "off", "f", "false", "n", "no"];
    match v {
        Value::Bool(b) => Ok(*b),
        Value::Number(n) if n.as_f64() == Some(1.0) => Ok(true),
        Value::Number(n) if n.as_f64() == Some(0.0) => Ok(false),
        Value::String(s) => {
            // pydantic does not strip spaces here (" true " is refused).
            let t = s.to_ascii_lowercase();
            if TRUE.contains(&t.as_str()) {
                Ok(true)
            } else if FALSE.contains(&t.as_str()) {
                Ok(false)
            } else {
                Err(format!("expected true or false, found {}", describe(v)))
            }
        }
        other => Err(format!("expected true or false, found {}", describe(other))),
    }
}

/// A list of strings.
pub(crate) fn strings(v: &Value) -> Result<Vec<String>, String> {
    match v {
        Value::Array(items) => items
            .iter()
            .map(|x| {
                string(x)
                    .map_err(|_| format!("expected a list of strings, found {} in it", describe(x)))
            })
            .collect(),
        other => Err(format!(
            "expected a list of strings, found {}",
            describe(other)
        )),
    }
}

/// A table of strings, such as `env`. Keys keep their order of appearance
/// only up to sorting; callers that need order read the JSON directly.
pub(crate) fn string_map(v: &Value) -> Result<BTreeMap<String, String>, String> {
    match v {
        Value::Object(map) => map
            .iter()
            .map(|(k, x)| match x {
                Value::String(s) => Ok((k.clone(), s.clone())),
                other => Err(format!(
                    "expected strings as values, found {} for {k}",
                    describe(other)
                )),
            })
            .collect(),
        other => Err(format!(
            "expected a table of strings, found {}",
            describe(other)
        )),
    }
}

/// A user: a name or a numeric id (`str | int`), kept as text.
pub(crate) fn user(v: &Value) -> Result<String, String> {
    match v {
        Value::String(s) => Ok(s.clone()),
        Value::Number(_) | Value::Bool(_) => int(v).map(|i| i.to_string()),
        other => Err(format!(
            "expected a user name or id, found {}",
            describe(other)
        )),
    }
}

/// One of a fixed set of words.
pub(crate) fn choice(v: &Value, allowed: &[&str]) -> Result<String, String> {
    match v {
        Value::String(s) if allowed.contains(&s.as_str()) => Ok(s.clone()),
        other => Err(format!(
            "expected one of {}, found {}",
            allowed
                .iter()
                .map(|a| format!("\"{a}\""))
                .collect::<Vec<_>>()
                .join(", "),
            describe(other)
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Guards the D5 fixes (2176d568, 66b7f937) and the D5b port of
    /// pydantic-core's string rules: int, float and boolean against 0.8's
    /// task.toml model in pydantic 2.12's lax mode on 10,732 values (66
    /// hand-picked by D5, more from its review, then seeded random strings
    /// and JSON numbers; `tests/fixtures/lax/make_fixture.py` writes them
    /// with 0.8's Python). The fixture holds what pydantic returned (`null`
    /// when it refused; `"inf"`/`"nan"` for those). Integers past i64 (which
    /// pydantic keeps) are left out. Before the port, 758 values differed
    /// (`" 0_3"` read as the float 3.0, `"0__1"` refused as an integer, ...).
    #[test]
    fn int_float_and_boolean_match_pydantic() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/lax/pydantic-lax.jsonl");
        let text = std::fs::read_to_string(path).unwrap();
        let mut failures = Vec::new();
        for line in text.lines() {
            let case: Value = serde_json::from_str(line).unwrap();
            let v = &case["value"];
            match &case["int"] {
                Value::Number(n) if n.as_i64().is_none() => {}
                Value::Number(n) => {
                    if int(v).ok() != n.as_i64() {
                        failures.push(format!("int {v}: {:?}, pydantic {n}", int(v)));
                    }
                }
                _ => {
                    if let Ok(i) = int(v) {
                        failures.push(format!("int {v}: {i}, pydantic refuses"));
                    }
                }
            }
            let got = float(v).ok();
            let ok = match &case["float"] {
                Value::Number(n) => got == n.as_f64(),
                Value::String(s) if s == "nan" => got.is_some_and(f64::is_nan),
                Value::String(s) => {
                    got.is_some_and(|f| f.is_infinite() && (f > 0.0) == !s.starts_with('-'))
                }
                _ => got.is_none(),
            };
            if !ok {
                failures.push(format!("float {v}: {got:?}, pydantic {}", case["float"]));
            }
            if boolean(v).ok() != case["bool"].as_bool() {
                failures.push(format!(
                    "bool {v}: {:?}, pydantic {}",
                    boolean(v),
                    case["bool"]
                ));
            }
        }
        // Underscores off digits, which pydantic 2.12 refuses as int.
        for s in ["1_.0", "1._0", "1.0_0", "+_1", "-_1"] {
            if let Ok(i) = int(&Value::from(s)) {
                failures.push(format!("int {s:?}: {i}, pydantic refuses"));
            }
        }
        assert_eq!(int(&Value::from("1_000.0")), Ok(1000));
        assert!(
            failures.is_empty(),
            "{} differences:\n{}",
            failures.len(),
            failures.join("\n")
        );
    }
    use serde_json::json;

    #[test]
    fn ints_follow_pydantic_lax_mode() {
        assert_eq!(int(&json!(2)), Ok(2));
        assert_eq!(int(&json!(2.0)), Ok(2));
        assert_eq!(int(&json!(" 3 ")), Ok(3));
        assert_eq!(int(&json!(true)), Ok(1));
        assert!(int(&json!(2.5)).is_err());
        assert!(int(&json!("four")).is_err());
        assert!(int(&json!([1])).is_err());
    }

    #[test]
    fn floats_and_bools() {
        assert_eq!(float(&json!(600)), Ok(600.0));
        assert_eq!(float(&json!("1.5")), Ok(1.5));
        assert!(float(&json!("soon")).is_err());
        assert_eq!(boolean(&json!("yes")), Ok(true));
        assert_eq!(boolean(&json!("Off")), Ok(false));
        assert_eq!(boolean(&json!(0)), Ok(false));
        assert!(boolean(&json!("maybe")).is_err());
    }

    #[test]
    fn strings_are_never_coerced() {
        assert_eq!(string(&json!("a")), Ok("a".into()));
        assert!(string(&json!(1)).is_err());
        assert!(strings(&json!(["a", 1])).is_err());
        assert!(string_map(&json!({"A": 1})).is_err());
        assert_eq!(user(&json!(1000)), Ok("1000".into()));
        assert_eq!(choice(&json!("public"), &["public"]), Ok("public".into()));
        assert!(choice(&json!("open"), &["public"]).is_err());
    }
}
