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
            match n.as_f64() {
                Some(f) if f.fract() == 0.0 && f.is_finite() && f.abs() < 9.2e18 => Ok(f as i64),
                _ => fail(),
            }
        }
        Value::String(s) => {
            // A zero fraction is a whole number ("1.0", "2.00"; not "1."),
            // and `_` may only sit between digits of the whole part.
            let t = s.trim();
            let whole = match t.split_once('.') {
                Some((whole, frac)) if !frac.is_empty() && frac.bytes().all(|b| b == b'0') => whole,
                Some(_) => return fail(),
                None => t,
            };
            let b = whole.as_bytes();
            let between_digits = |i: usize| {
                i > 0 && b[i - 1].is_ascii_digit() && b.get(i + 1).is_some_and(u8::is_ascii_digit)
            };
            if (0..b.len()).any(|i| b[i] == b'_' && !between_digits(i)) {
                return fail();
            }
            let t = whole.replace('_', "");
            let digits = t.strip_prefix(['+', '-']).unwrap_or(&t);
            if !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()) {
                t.parse::<i64>().or_else(|_| fail())
            } else {
                fail()
            }
        }
        _ => fail(),
    }
}

/// A number's text with its underscores removed, as pydantic reads
/// `"1_000"`: `None` when one starts or ends the text or two are together.
fn without_underscores(t: &str) -> Option<String> {
    if !t.contains('_') {
        return Some(t.to_string());
    }
    if t.starts_with('_') || t.ends_with('_') || t.contains("__") {
        return None;
    }
    Some(t.replace('_', ""))
}

/// A float field: numbers, booleans, and strings that hold a number.
pub(crate) fn float(v: &Value) -> Result<f64, String> {
    match v {
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        Value::Number(n) => n
            .as_f64()
            .ok_or_else(|| format!("expected a number, found {n}")),
        Value::String(s) => without_underscores(s.trim())
            .and_then(|t| t.parse::<f64>().ok())
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

    /// Guards the D5 fix: int, float and boolean against pydantic 2.12's lax
    /// mode (0.8's loader) on 66 spellings (`"1_000"`, `"1.0"`, `" 12 "`,
    /// `" true "`, `"0x10"`, `"1e400"`, Unicode digits, JSON numbers and
    /// bools). The fixture holds what pydantic returned (`null` when it
    /// refused; `"inf"`/`"nan"` for those). Integers past i64 (which pydantic
    /// keeps) are left out. pydantic-core also has quirks on random text that
    /// are not ported (it reads the integer `"0-495"` as -495).
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
