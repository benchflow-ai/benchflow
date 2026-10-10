//! JSON and `str()` the way Python writes them, so exported text matches
//! what Harbor's exporters, which are Python, put in the same places.

use benchflow_agent::trajectory::python_float_repr;
use serde_json::{Number, Value};

/// How `json.dumps` was called.
#[derive(Debug, Clone, Copy, Default)]
pub struct Dumps {
    /// `ensure_ascii` (Python's default): escape everything outside ASCII.
    pub ascii: bool,
    /// `indent=2` when set.
    pub indent: Option<usize>,
}

impl Dumps {
    /// `json.dumps(value)`.
    pub const ASCII: Dumps = Dumps {
        ascii: true,
        indent: None,
    };
    /// `json.dumps(value, ensure_ascii=False)`.
    pub const UNICODE: Dumps = Dumps {
        ascii: false,
        indent: None,
    };
}

/// `json.dumps(value, ...)`.
pub fn dumps(value: &Value, how: Dumps) -> String {
    let mut out = String::new();
    write(value, how, 0, &mut out);
    out
}

fn number(n: &Number) -> String {
    if n.is_f64() {
        python_float_repr(n.as_f64().unwrap_or(0.0))
    } else {
        n.to_string()
    }
}

fn write(value: &Value, how: Dumps, depth: usize, out: &mut String) {
    let newline = |out: &mut String, depth: usize| {
        if let Some(n) = how.indent {
            out.push('\n');
            out.push_str(&" ".repeat(n * depth));
        }
    };
    let item_sep = if how.indent.is_some() { "," } else { ", " };
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => out.push_str(&number(n)),
        Value::String(s) => string(s, how.ascii, out),
        Value::Array(items) => {
            if items.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(item_sep);
                }
                newline(out, depth + 1);
                write(item, how, depth + 1, out);
            }
            newline(out, depth);
            out.push(']');
        }
        Value::Object(map) => {
            if map.is_empty() {
                out.push_str("{}");
                return;
            }
            out.push('{');
            for (i, (k, v)) in map.iter().enumerate() {
                if i > 0 {
                    out.push_str(item_sep);
                }
                newline(out, depth + 1);
                string(k, how.ascii, out);
                out.push_str(": ");
                write(v, how, depth + 1, out);
            }
            newline(out, depth);
            out.push('}');
        }
    }
}

/// A JSON string as Python escapes it: `"`, `\`, and control characters
/// always; with `ascii`, everything outside printable ASCII as `\uXXXX`.
fn string(s: &str, ascii: bool, out: &mut String) {
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
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c if ascii && (c as u32) > 0x7e => {
                let mut units = [0u16; 2];
                for u in c.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{u:04x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Python's `str(value)` for a parsed JSON value: strings as they are,
/// containers as their `repr`.
pub fn py_str(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => repr(other),
    }
}

fn repr(value: &Value) -> String {
    match value {
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

/// Python's `repr` of a string: single quotes unless the text has a single
/// quote and no double quote.
fn repr_str(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::from(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\x{:02x}", c as u32))
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn matches_pythons_json_dumps() {
        let v = json!({"a": [1, 2.0, 1e20, "é\n"], "b": {}, "c": [], "d": null, "e": true});
        assert_eq!(
            dumps(&v, Dumps::ASCII),
            format!(
                r#"{{"a": [1, 2.0, 1e+20, "{bs}u00e9\n"], "b": {{}}, "c": [], "d": null, "e": true}}"#,
                bs = '\\'
            )
        );
        assert!(dumps(&v, Dumps::UNICODE).contains("\"é\\n\""));
        let pretty = dumps(
            &json!([{"name": "x", "p": [1]}]),
            Dumps {
                ascii: false,
                indent: Some(2),
            },
        );
        assert_eq!(
            pretty,
            "[\n  {\n    \"name\": \"x\",\n    \"p\": [\n      1\n    ]\n  }\n]"
        );
    }

    #[test]
    fn matches_pythons_str() {
        assert_eq!(py_str(&json!("plain")), "plain");
        assert_eq!(
            py_str(&json!({"a": [1, null, true], "b": "it's"})),
            r#"{'a': [1, None, True], 'b': "it's"}"#
        );
    }
}
