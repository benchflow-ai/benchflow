//! Python's text forms of values, so messages read exactly as the reference
//! checker's do.
//!
//! The reference builds messages with f-strings, `repr()` (`{x!r}`), and
//! `json.dumps`. These functions give the same text for JSON-like values:
//! `repr("it's")` is `"it's"`, `repr(0.00001)` is `1e-05`, and
//! `json.dumps(["a"])` is `["a"]`.

use serde_json::Value;

/// Python's `repr()` of a str: single quotes unless the text holds a single
/// quote and no double quote, with non-printable characters escaped.
pub(crate) fn repr_str(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if printable(c) => out.push(c),
            c => {
                let n = c as u32;
                if n < 0x100 {
                    out.push_str(&format!("\\x{n:02x}"));
                } else if n < 0x10000 {
                    out.push_str(&format!("\\u{n:04x}"));
                } else {
                    out.push_str(&format!("\\U{n:08x}"));
                }
            }
        }
    }
    out.push(quote);
    out
}

/// Python's `str.isprintable()` for one character, for the characters a
/// task file can plausibly hold: control characters, separators other than
/// the space, format characters, and private-use code points are not.
fn printable(c: char) -> bool {
    if c == ' ' {
        return true;
    }
    if c.is_control() {
        return false;
    }
    let n = c as u32;
    !matches!(n,
        0x00A0 | 0x00AD | 0x061C | 0x06DD | 0x070F | 0x08E2 | 0x1680 | 0x180E
        | 0x0600..=0x0605
        | 0x0890..=0x0891
        | 0x2000..=0x200F
        | 0x2028..=0x202F
        | 0x205F..=0x2064
        | 0x2066..=0x206F
        | 0x3000
        | 0xD800..=0xF8FF
        | 0xFDD0..=0xFDEF
        | 0xFEFF
        | 0xFFF9..=0xFFFB
        | 0xFFFE..=0xFFFF
        | 0x110BD | 0x110CD
        | 0x13430..=0x1343F
        | 0x1BCA0..=0x1BCA3
        | 0x1D173..=0x1D17A
        | 0xE0001
        | 0xE0020..=0xE007F
        | 0xF0000..=0x10FFFF
    ) && (n & 0xFFFE) != 0xFFFE
}

/// Python's `repr()` of a float: the shortest text that reads back as the
/// same number, in exponent form below 1e-4 and from 1e16 up.
pub(crate) fn float_repr(f: f64) -> String {
    if f.is_nan() {
        return "nan".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "inf".into() } else { "-inf".into() };
    }
    if f == 0.0 {
        return if f.is_sign_negative() {
            "-0.0".into()
        } else {
            "0.0".into()
        };
    }
    // Rust's `{:e}` gives the shortest round-trip digits, as Python's repr does.
    let e = format!("{:e}", f.abs());
    let (mantissa, exp) = e.split_once('e').unwrap_or((&e, "0"));
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let exp: i32 = exp.parse().unwrap_or(0);
    let decpt = exp + 1;
    let sign = if f < 0.0 { "-" } else { "" };
    if decpt <= -4 || decpt > 16 {
        let (first, rest) = digits.split_at(1);
        let frac = if rest.is_empty() {
            String::new()
        } else {
            format!(".{rest}")
        };
        let x = decpt - 1;
        let xs = if x < 0 { '-' } else { '+' };
        return format!("{sign}{first}{frac}e{xs}{:02}", x.abs());
    }
    let n = digits.len() as i32;
    let body = if decpt <= 0 {
        format!("0.{}{digits}", "0".repeat((-decpt) as usize))
    } else if decpt >= n {
        format!("{digits}{}.0", "0".repeat((decpt - n) as usize))
    } else {
        let (a, b) = digits.split_at(decpt as usize);
        format!("{a}.{b}")
    };
    format!("{sign}{body}")
}

/// Python's text of a number: integers as digits, floats as [`float_repr`].
pub(crate) fn number(n: &serde_json::Number) -> String {
    if let Some(i) = n.as_i64() {
        i.to_string()
    } else if let Some(u) = n.as_u64() {
        u.to_string()
    } else {
        float_repr(n.as_f64().unwrap_or(f64::NAN))
    }
}

/// Python's `repr()` of a JSON-like value.
pub(crate) fn repr(v: &Value) -> String {
    match v {
        Value::Null => "None".into(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::Number(n) => number(n),
        Value::String(s) => repr_str(s),
        Value::Array(items) => format!(
            "[{}]",
            items.iter().map(repr).collect::<Vec<_>>().join(", ")
        ),
        Value::Object(map) => format!(
            "{{{}}}",
            map.iter()
                .map(|(k, v)| format!("{}: {}", repr_str(k), repr(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// Python's `str()` of a JSON-like value: a string as it is, anything else as
/// its `repr()`.
pub(crate) fn str_of(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => repr(other),
    }
}

/// Python's `json.dumps(v)` with its defaults: ASCII only, `", "` between
/// items and `": "` after keys.
pub(crate) fn json_dumps(v: &Value) -> String {
    match v {
        Value::Null => "null".into(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => number(n),
        Value::String(s) => json_str(s),
        Value::Array(items) => format!(
            "[{}]",
            items.iter().map(json_dumps).collect::<Vec<_>>().join(", ")
        ),
        Value::Object(map) => format!(
            "{{{}}}",
            map.iter()
                .map(|(k, v)| format!("{}: {}", json_str(k), json_dumps(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// A JSON string as Python's `json.dumps` writes it, with `ensure_ascii`.
pub(crate) fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 || (c as u32) > 0x7f => {
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// The reference's `fmt_duration`: 90 is `1m30s`, 3900 is `1h05m`, 45 is
/// `45s`. Seconds are rounded half to even, as Python's `round` does.
pub(crate) fn fmt_duration(seconds: f64) -> String {
    let s = seconds.round_ties_even().max(0.0) as u64;
    let (h, m, sec) = (s / 3600, s % 3600 / 60, s % 60);
    let tail = if sec > 0 {
        format!("{sec:02}s")
    } else {
        String::new()
    };
    if h > 0 {
        format!("{h}h{m:02}m{tail}")
    } else if m > 0 {
        format!("{m}m{tail}")
    } else {
        format!("{sec}s")
    }
}

/// `"1 criterion"`, `"2 criterions"`: the reference's `_plural`, which adds
/// an s (callers fix "criterions" themselves, as the reference does).
pub(crate) fn plural(n: usize, word: &str) -> String {
    format!("{n} {word}{}", if n == 1 { "" } else { "s" })
}

/// `plural` for "criterion", spelled "criteria" when there are several.
pub(crate) fn criteria(n: usize) -> String {
    plural(n, "criterion").replace("criterions", "criteria")
}

/// The first `n` characters of `s` (Python's `s[:n]`, which counts code
/// points, not bytes).
pub(crate) fn prefix(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // Expected values are what CPython 3.12 prints.
    #[test]
    fn reprs_of_strings() {
        assert_eq!(repr_str("abc"), "'abc'");
        assert_eq!(repr_str("it's"), "\"it's\"");
        assert_eq!(repr_str("it's \"x\""), "'it\\'s \"x\"'");
        assert_eq!(repr_str("a\nb\\"), "'a\\nb\\\\'");
        assert_eq!(repr_str("\u{7f}\u{a0}\u{200b}é"), "'\\x7f\\xa0\\u200bé'");
        assert_eq!(repr_str("\u{1F600}"), "'\u{1F600}'");
    }

    #[test]
    fn reprs_of_floats() {
        assert_eq!(float_repr(1.0), "1.0");
        assert_eq!(float_repr(0.1), "0.1");
        assert_eq!(float_repr(-0.2), "-0.2");
        assert_eq!(float_repr(0.0001), "0.0001");
        assert_eq!(float_repr(0.00001), "1e-05");
        assert_eq!(float_repr(1e15), "1000000000000000.0");
        assert_eq!(float_repr(1e16), "1e+16");
        assert_eq!(float_repr(1.5e16), "1.5e+16");
        assert_eq!(float_repr(123456.789), "123456.789");
        assert_eq!(float_repr(2.5e-7), "2.5e-07");
        assert_eq!(float_repr(1e100), "1e+100");
        assert_eq!(float_repr(-0.0), "-0.0");
    }

    #[test]
    fn reprs_of_values() {
        assert_eq!(repr(&json!({"penalty": -0.2})), "{'penalty': -0.2}");
        assert_eq!(repr(&json!(["a", 1, true, null])), "['a', 1, True, None]");
        assert_eq!(str_of(&json!("x")), "x");
        assert_eq!(str_of(&json!(3)), "3");
    }

    #[test]
    fn json_dumps_like_python() {
        assert_eq!(
            json_dumps(&json!(["pypi.org", "x"])),
            "[\"pypi.org\", \"x\"]"
        );
        assert_eq!(json_dumps(&json!({"block": ["a"]})), "{\"block\": [\"a\"]}");
        assert_eq!(
            json_dumps(&json!("é\u{1F600}")),
            "\"\\u00e9\\ud83d\\ude00\""
        );
        assert_eq!(json_dumps(&json!(0.5)), "0.5");
    }

    #[test]
    fn durations_like_the_reference() {
        assert_eq!(fmt_duration(90.0), "1m30s");
        assert_eq!(fmt_duration(3900.0), "1h05m");
        assert_eq!(fmt_duration(3600.0), "1h00m");
        assert_eq!(fmt_duration(45.0), "45s");
        assert_eq!(fmt_duration(65.0), "1m05s");
        assert_eq!(fmt_duration(2.5), "2s");
        assert_eq!(fmt_duration(3.5), "4s");
        assert_eq!(fmt_duration(0.0), "0s");
    }

    #[test]
    fn plurals_and_prefixes() {
        assert_eq!(criteria(1), "1 criterion");
        assert_eq!(criteria(3), "3 criteria");
        assert_eq!(plural(2, "behavior"), "2 behaviors");
        assert_eq!(prefix("héllo", 2), "hé");
        assert_eq!(prefix("ab", 5), "ab");
    }
}
