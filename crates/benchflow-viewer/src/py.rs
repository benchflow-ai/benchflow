//! The bits of Python behavior the 0.8 viewer's output depends on: truth
//! values, `str()`, `float()` of text, whitespace, HTML escaping, and how
//! `pathlib` names and prints paths.

use std::ffi::OsStr;
use std::path::{Component, Path, PathBuf};

use crate::pyjson::{Value, float_repr};

/// Python's `str.isspace()` for one character: Unicode whitespace plus the
/// ASCII information separators U+001C..U+001F.
pub fn is_space(ch: char) -> bool {
    ch.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&ch)
}

/// `text.strip()`.
pub fn strip(text: &str) -> &str {
    text.trim_matches(is_space)
}

/// `text.rstrip()`.
pub fn rstrip(text: &str) -> &str {
    text.trim_end_matches(is_space)
}

/// `text.split(None, 1)[0]`: the first whitespace-separated word.
pub fn first_word(text: &str) -> Option<&str> {
    text.split(is_space).find(|word| !word.is_empty())
}

/// Python's truth value of a JSON value.
pub fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::Int(int) => !int.is(0),
        // NaN is true in Python.
        Value::Float(number) => *number != 0.0,
        Value::Str(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(map) => !map.is_empty(),
    }
}

/// Python's `str(value)`.
pub fn str_of(value: &Value) -> String {
    match value {
        Value::Str(text) => text.clone(),
        other => repr_of(other),
    }
}

/// Python's `repr(value)` for a JSON value (what `str()` shows for the
/// members of a list or dict).
fn repr_of(value: &Value) -> String {
    match value {
        Value::Null => "None".to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::Int(int) => int.to_string(),
        Value::Float(number) => float_repr(*number),
        Value::Str(text) => repr_string(text),
        Value::Array(items) => {
            let parts: Vec<String> = items.iter().map(repr_of).collect();
            format!("[{}]", parts.join(", "))
        }
        Value::Object(map) => {
            let parts: Vec<String> = map
                .iter()
                .map(|(key, item)| format!("{}: {}", repr_string(key), repr_of(item)))
                .collect();
            format!("{{{}}}", parts.join(", "))
        }
    }
}

/// Python's `repr()` of a string: single quotes unless the text has a `'`
/// and no `"`, with backslash escapes for unprintable characters.
fn repr_string(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(text.len() + 2);
    out.push(quote);
    for ch in text.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if c < ' ' || c == '\u{7f}' => out.push_str(&format!("\\x{:02x}", c as u32)),
            c if c.is_ascii() || is_printable(c) => out.push(c),
            c if (c as u32) <= 0xff => out.push_str(&format!("\\x{:02x}", c as u32)),
            c if (c as u32) <= 0xffff => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push_str(&format!("\\U{:08x}", c as u32)),
        }
    }
    out.push(quote);
    out
}

/// An approximation of Python's `str.isprintable()` for non-ASCII
/// characters: not a control, separator, or common format character.
fn is_printable(ch: char) -> bool {
    let code = ch as u32;
    !(ch.is_control()
        || ch.is_whitespace()
        || matches!(code, 0xad | 0x600..=0x605 | 0x61c | 0x6dd | 0x70f | 0x180e | 0x200b..=0x200f
            | 0x202a..=0x202e | 0x2060..=0x206f | 0xfeff | 0xfff9..=0xfffb | 0xe000..=0xf8ff
            | 0xf0000..=0x10ffff))
}

/// Python's `float(text)`: surrounding whitespace allowed, `_` between
/// digits allowed, `inf`/`nan` in any case. The result may be infinite or
/// NaN. Only ASCII digits are read (Python also reads other scripts'
/// decimal digits; trajectories do not use them).
pub fn parse_float(text: &str) -> Option<f64> {
    // Non-ASCII whitespace counts as a space; any other non-ASCII
    // character (or DEL) makes the text unreadable.
    let mut ascii = String::with_capacity(text.len());
    for ch in text.chars() {
        if (ch as u32) < 0x7f {
            ascii.push(ch);
        } else if is_space(ch) {
            ascii.push(' ');
        } else {
            return None;
        }
    }
    let ascii = without_digit_underscores(&ascii)?;
    // Only C `isspace` characters are stripped at this point.
    let trimmed = ascii.trim_matches(|c| matches!(c, ' ' | '\t' | '\n' | '\u{b}' | '\u{c}' | '\r'));
    if trimmed.is_empty() {
        return None;
    }
    trimmed.parse::<f64>().ok()
}

/// Python's `int(text)` for the small integers BenchFlow reads from the
/// environment.
pub fn parse_int(text: &str) -> Option<i64> {
    let trimmed = strip(text);
    let digits = without_digit_underscores(trimmed)?;
    let (negative, digits) = match digits.as_bytes().first()? {
        b'-' => (true, &digits[1..]),
        b'+' => (false, &digits[1..]),
        _ => (false, &digits[..]),
    };
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    let magnitude: i128 = digits.parse().unwrap_or(i128::MAX);
    let value = if negative { -magnitude } else { magnitude };
    Some(value.clamp(i64::MIN as i128, i64::MAX as i128) as i64)
}

/// Remove `_` digit separators, or `None` when one is not between two digits
/// (CPython's `_Py_string_to_number_with_underscores`).
fn without_digit_underscores(text: &str) -> Option<String> {
    if !text.contains('_') {
        return Some(text.to_string());
    }
    let mut out = String::with_capacity(text.len());
    let mut prev = '\0';
    for ch in text.chars() {
        if ch == '_' {
            if !prev.is_ascii_digit() {
                return None;
            }
        } else {
            if prev == '_' && !ch.is_ascii_digit() {
                return None;
            }
            out.push(ch);
        }
        prev = ch;
    }
    (prev != '_').then_some(out)
}

/// `text.isalnum()`: non-empty and every character a letter or digit.
pub fn isalnum(text: &str) -> bool {
    !text.is_empty() && text.chars().all(char::is_alphanumeric)
}

/// `html.escape(text)`: `&`, `<`, `>`, `"` and `'`.
pub fn html_escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#x27;"),
            other => out.push(other),
        }
    }
    out
}

/// A file name as Python sees it: bytes that are not UTF-8 become `?` each
/// (Python's surrogate escapes, which 0.8 writes as `?`).
pub fn os_text(name: &OsStr) -> String {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        let mut out = String::new();
        for chunk in name.as_bytes().utf8_chunks() {
            out.push_str(chunk.valid());
            out.extend(std::iter::repeat_n('?', chunk.invalid().len()));
        }
        out
    }
    #[cfg(not(unix))]
    {
        name.to_string_lossy().into_owned()
    }
}

/// `str(path)` for a path.
pub fn path_text(path: &Path) -> String {
    os_text(path.as_os_str())
}

/// `Path(text)` normalized the way `pathlib` normalizes it: `.` components
/// and repeated or trailing separators dropped, `..` kept.
pub fn normalize(path: &Path) -> PathBuf {
    let normalized: PathBuf = path
        .components()
        .filter(|part| !matches!(part, Component::CurDir))
        .collect();
    if normalized.as_os_str().is_empty() {
        PathBuf::from(".")
    } else {
        normalized
    }
}

/// `path.name`: the last component, `..` included, or `""` for `/` and `.`.
pub fn name_of(path: &Path) -> String {
    match path.components().next_back() {
        Some(Component::Normal(name)) => os_text(name),
        Some(Component::ParentDir) => "..".to_string(),
        _ => String::new(),
    }
}

/// `path.parent`: the path itself for `/` and `.`, `.` for a single name.
pub fn parent_of(path: &Path) -> PathBuf {
    match path.parent() {
        None => path.to_path_buf(),
        Some(parent) if parent.as_os_str().is_empty() => PathBuf::from("."),
        Some(parent) => parent.to_path_buf(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pyjson::parse;

    #[test]
    fn truthiness_matches_python() {
        for (text, expected) in [
            ("null", false),
            ("0", false),
            ("0.0", false),
            ("\"\"", false),
            ("[]", false),
            ("{}", false),
            ("false", false),
            ("NaN", true),
            ("-1", true),
            ("\"0\"", true),
            ("[0]", true),
        ] {
            assert_eq!(truthy(&parse(text).unwrap()), expected, "{text}");
        }
    }

    #[test]
    fn str_matches_python() {
        for (text, expected) in [
            ("null", "None"),
            ("true", "True"),
            ("1.0", "1.0"),
            ("1e16", "1e+16"),
            ("\"x\"", "x"),
            (
                r#"[1, "a'b", null, {"k": false}]"#,
                r#"[1, "a'b", None, {'k': False}]"#,
            ),
            (
                "[\"tab\\t\", \"\\u00e9\", \"\\u200b\"]",
                "['tab\\t', '\u{e9}', '\\u200b']",
            ),
        ] {
            assert_eq!(str_of(&parse(text).unwrap()), expected, "{text}");
        }
    }

    #[test]
    fn float_reads_like_python() {
        assert_eq!(parse_float(" 2.5 "), Some(2.5));
        assert_eq!(parse_float("1_000.5"), Some(1000.5));
        assert_eq!(parse_float("1__0"), None);
        assert_eq!(parse_float("1_"), None);
        assert_eq!(parse_float("\u{a0}1\u{a0}"), Some(1.0));
        assert_eq!(parse_float("\u{1c}1"), None);
        assert!(parse_float("-Infinity").unwrap().is_infinite());
        assert!(parse_float("nan").unwrap().is_nan());
        assert_eq!(parse_float("0x10"), None);
        assert_eq!(parse_float(""), None);
    }

    #[test]
    fn int_reads_like_python() {
        assert_eq!(parse_int(" 7 "), Some(7));
        assert_eq!(parse_int("1_000"), Some(1000));
        assert_eq!(parse_int("-3"), Some(-3));
        assert_eq!(parse_int("7.0"), None);
        assert_eq!(parse_int("x"), None);
    }

    #[test]
    fn paths_name_and_parent_like_pathlib() {
        assert_eq!(normalize(Path::new("./jobs/./a/")), PathBuf::from("jobs/a"));
        assert_eq!(normalize(Path::new(".")), PathBuf::from("."));
        assert_eq!(name_of(Path::new("jobs/a")), "a");
        assert_eq!(name_of(Path::new(".")), "");
        assert_eq!(name_of(Path::new("/")), "");
        assert_eq!(name_of(Path::new("a/..")), "..");
        assert_eq!(parent_of(Path::new("jobs")), PathBuf::from("."));
        assert_eq!(parent_of(Path::new(".")), PathBuf::from("."));
        assert_eq!(parent_of(Path::new("/")), PathBuf::from("/"));
    }

    #[test]
    fn html_escape_quotes_too() {
        assert_eq!(
            html_escape("<a href=\"x\">'&'</a>"),
            "&lt;a href=&quot;x&quot;&gt;&#x27;&amp;&#x27;&lt;/a&gt;"
        );
    }

    #[cfg(unix)]
    #[test]
    fn undecodable_names_read_as_question_marks() {
        use std::os::unix::ffi::OsStrExt;
        assert_eq!(os_text(OsStr::from_bytes(b"run-\xff\xfe-x")), "run-??-x");
    }
}
