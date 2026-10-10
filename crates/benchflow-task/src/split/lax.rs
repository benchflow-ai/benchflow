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
            let t = s.trim();
            let digits = t.strip_prefix(['+', '-']).unwrap_or(t);
            if !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()) {
                t.parse::<i64>().or_else(|_| fail())
            } else {
                fail()
            }
        }
        _ => fail(),
    }
}

/// A float field: numbers, booleans, and strings that hold a number.
pub(crate) fn float(v: &Value) -> Result<f64, String> {
    match v {
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        Value::Number(n) => n
            .as_f64()
            .ok_or_else(|| format!("expected a number, found {n}")),
        Value::String(s) => s
            .trim()
            .parse::<f64>()
            .map_err(|_| format!("expected a number, found {}", describe(v))),
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
            let t = s.trim().to_ascii_lowercase();
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
