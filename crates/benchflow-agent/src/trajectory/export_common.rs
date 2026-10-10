//! What the trainer exports share (0.8's `trajectories/_export_common.py`),
//! plus the bits of Python behavior their output depends on.
//!
//! The Verifiers and ADP exports walk events in their JSON form
//! ([`super::Event::to_json`]), the dicts 0.8's exporters walked, so an
//! event this build keeps as [`super::Event::Other`] exports as it did in
//! 0.8. Text taken from an event goes through Python's `str()`: a nested
//! content block becomes its dict `repr`, as 0.8 wrote it.

use std::io::{self, Write};

use serde::Serialize;
use serde_json::ser::{Formatter, PrettyFormatter, Serializer};
use serde_json::{Map, Value, json};

use super::jsonl::{PyFormatter, python_float_repr};

/// Python's truth value of a JSON value.
pub(crate) fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64() != Some(0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// Python's `str(event.get(key) or "")`.
pub(crate) fn str_or_empty(map: &Map<String, Value>, key: &str) -> String {
    match map.get(key) {
        Some(value) if truthy(value) => py_str(value),
        _ => String::new(),
    }
}

/// Python's `str(event.get(key) or default)`.
pub(crate) fn str_or(map: &Map<String, Value>, key: &str, default: &str) -> String {
    match map.get(key) {
        Some(value) if truthy(value) => py_str(value),
        _ => default.to_string(),
    }
}

/// Python's `str(value)` for a JSON value.
pub(crate) fn py_str(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => py_repr(other),
    }
}

/// Python's `repr(value)` for a JSON value.
pub(crate) fn py_repr(value: &Value) -> String {
    match value {
        Value::Null => "None".into(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::Number(n) => match (n.as_i64(), n.as_u64()) {
            (Some(i), _) => i.to_string(),
            (_, Some(u)) => u.to_string(),
            _ => py_float_repr(n.as_f64().unwrap_or(0.0)),
        },
        Value::String(s) => repr_str(s),
        Value::Array(items) => {
            let parts: Vec<String> = items.iter().map(py_repr).collect();
            format!("[{}]", parts.join(", "))
        }
        Value::Object(map) => {
            let parts: Vec<String> = map
                .iter()
                .map(|(k, v)| format!("{}: {}", repr_str(k), py_repr(v)))
                .collect();
            format!("{{{}}}", parts.join(", "))
        }
    }
}

/// Python's `repr(float)`, `nan` and `inf` included.
pub(crate) fn py_float_repr(value: f64) -> String {
    if value.is_nan() {
        "nan".into()
    } else if value.is_infinite() {
        if value > 0.0 { "inf" } else { "-inf" }.into()
    } else {
        python_float_repr(value)
    }
}

/// Python's `repr()` of a str: single quotes unless the text holds a single
/// quote and no double quote, unprintable characters escaped.
fn repr_str(s: &str) -> String {
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

/// Python's `str.isprintable()` for one character: controls, separators
/// other than the space, format characters, surrogates, private use and
/// noncharacters are not. (Unassigned code points, which Python also counts
/// as unprintable, are not tracked.)
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

/// An event a subagent produced: only an explicit, non-empty
/// `parent_tool_call_id` string counts (0.8's `is_attributed_child_event`).
pub(crate) fn is_attributed_child(event: &Map<String, Value>) -> bool {
    matches!(event.get("parent_tool_call_id"), Some(Value::String(p)) if !p.is_empty())
}

/// What a root-only export left out, or `None` when no event was a
/// subagent's (0.8's `acp_projection_coverage`).
pub fn acp_projection_coverage(events: &[Value]) -> Option<Value> {
    let omitted = events
        .iter()
        .filter_map(Value::as_object)
        .filter(|e| is_attributed_child(e))
        .count();
    (omitted > 0).then(|| {
        json!({
            "scope": "root_agent_projection",
            "excluded_attributed_child_events": omitted,
            "source_artifact": "trajectory/acp_trajectory.jsonl",
            "child_transcript_completeness": "unknown",
            "outcome_and_usage_scope": "rollout_not_partitioned_by_agent",
        })
    })
}

/// 0.8's `content_blocks_to_text`: a string as is; of a list of blocks, the
/// text of flat (`{"text"}`) and nested (`{"content": {"text"}}`) blocks,
/// one per line.
pub(crate) fn content_blocks_to_text(content: Option<&Value>) -> String {
    let items = match content {
        Some(Value::String(s)) => return s.clone(),
        Some(Value::Array(items)) => items,
        _ => return String::new(),
    };
    let mut parts = Vec::new();
    for item in items.iter().filter_map(Value::as_object) {
        let inner = match item.get("content") {
            Some(Value::Object(inner)) => inner.get("text"),
            other => other,
        };
        let text = match item.get("text") {
            Some(t) if truthy(t) => Some(t),
            _ => inner,
        };
        if let Some(text) = text.filter(|t| truthy(t)) {
            parts.push(py_str(text));
        }
    }
    parts.join("\n")
}

/// Consecutive thoughts waiting for the next action (0.8's `ThoughtBuffer`).
#[derive(Default)]
pub(crate) struct ThoughtBuffer(Vec<String>);

impl ThoughtBuffer {
    pub(crate) fn push(&mut self, text: String) {
        self.0.push(text);
    }

    /// The thoughts joined by blank lines, then cleared.
    pub(crate) fn take(&mut self) -> Option<String> {
        if self.0.is_empty() {
            return None;
        }
        let joined = self.0.join("\n\n");
        self.0.clear();
        Some(joined)
    }
}

/// `value` as Python's `json.dumps(value, indent=2)` writes it.
pub fn to_python_json_pretty(value: &Value) -> String {
    let mut out = Vec::new();
    let mut serializer = Serializer::with_formatter(&mut out, PyPrettyFormatter::default());
    value
        .serialize(&mut serializer)
        .expect("serializing a JSON value into memory cannot fail");
    String::from_utf8(out).expect("the formatter writes ASCII")
}

/// Python's `indent=2` layout (`serde_json`'s pretty layout is the same)
/// with Python's string escaping and float forms.
#[derive(Default)]
struct PyPrettyFormatter<'a>(PrettyFormatter<'a>);

impl Formatter for PyPrettyFormatter<'_> {
    fn begin_array<W: ?Sized + Write>(&mut self, w: &mut W) -> io::Result<()> {
        self.0.begin_array(w)
    }
    fn end_array<W: ?Sized + Write>(&mut self, w: &mut W) -> io::Result<()> {
        self.0.end_array(w)
    }
    fn begin_array_value<W: ?Sized + Write>(&mut self, w: &mut W, first: bool) -> io::Result<()> {
        self.0.begin_array_value(w, first)
    }
    fn end_array_value<W: ?Sized + Write>(&mut self, w: &mut W) -> io::Result<()> {
        self.0.end_array_value(w)
    }
    fn begin_object<W: ?Sized + Write>(&mut self, w: &mut W) -> io::Result<()> {
        self.0.begin_object(w)
    }
    fn end_object<W: ?Sized + Write>(&mut self, w: &mut W) -> io::Result<()> {
        self.0.end_object(w)
    }
    fn begin_object_key<W: ?Sized + Write>(&mut self, w: &mut W, first: bool) -> io::Result<()> {
        self.0.begin_object_key(w, first)
    }
    fn begin_object_value<W: ?Sized + Write>(&mut self, w: &mut W) -> io::Result<()> {
        self.0.begin_object_value(w)
    }
    fn end_object_value<W: ?Sized + Write>(&mut self, w: &mut W) -> io::Result<()> {
        self.0.end_object_value(w)
    }
    fn write_string_fragment<W: ?Sized + Write>(&mut self, w: &mut W, s: &str) -> io::Result<()> {
        PyFormatter.write_string_fragment(w, s)
    }
    fn write_f64<W: ?Sized + Write>(&mut self, w: &mut W, value: f64) -> io::Result<()> {
        PyFormatter.write_f64(w, value)
    }
    fn write_f32<W: ?Sized + Write>(&mut self, w: &mut W, value: f32) -> io::Result<()> {
        PyFormatter.write_f32(w, value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reprs_match_python() {
        // Expected values from CPython 3.12's str() and repr().
        let value = json!({"type": "text", "text": "it's \"q\"\n\u{e9}\u{200b}", "n": [1, 2.5, null, true]});
        assert_eq!(
            py_str(&value),
            "{'type': 'text', 'text': 'it\\'s \"q\"\\n\u{e9}\\u200b', 'n': [1, 2.5, None, True]}"
        );
        assert_eq!(py_str(&json!("it's")), "it's");
        assert_eq!(py_repr(&json!("it's")), "\"it's\"");
        assert_eq!(py_float_repr(f64::NAN), "nan");
        assert_eq!(py_float_repr(1e-5), "1e-05");
    }

    #[test]
    fn pretty_json_matches_python_indent_2() {
        // Python: json.dumps({'a': [], 'b': {}, 'c': [1, 1.0, '\u00e9'], 'd': {'e': None}}, indent=2)
        let text = to_python_json_pretty(
            &json!({"a": [], "b": {}, "c": [1, 1.0, "\u{e9}"], "d": {"e": null}}),
        );
        assert_eq!(
            text,
            "{\n  \"a\": [],\n  \"b\": {},\n  \"c\": [\n    1,\n    1.0,\n    \"\\u00e9\"\n  ],\n  \"d\": {\n    \"e\": null\n  }\n}"
        );
    }

    #[test]
    fn content_blocks_read_flat_and_nested_text() {
        let content = json!([
            {"type": "content", "content": {"type": "text", "text": "nested"}},
            {"text": "flat"},
            {"text": "", "content": "fallback"},
            {"type": "diff"},
            "not a block",
        ]);
        assert_eq!(
            content_blocks_to_text(Some(&content)),
            "nested\nflat\nfallback"
        );
        assert_eq!(content_blocks_to_text(Some(&json!("raw"))), "raw");
        assert_eq!(content_blocks_to_text(None), "");
    }
}
