//! Config text with positions: parse TOML (or YAML) into a JSON value and
//! remember where every key and value sits, so a diagnostic can point at the
//! exact line and column.
//!
//! Paths are dotted key paths with array indices: `agent.timeout`,
//! `sandbox.outputs[1].path`, `credits[0].name`. A key containing a dot is
//! quoted the way TOML writes it: `env."A.B"`.

use std::collections::HashMap;
use std::ops::Range;

use toml_edit::{Document, Item, Value};

/// Where a key and its value are, as byte ranges into the parsed text.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Spans {
    pub key: Option<Range<usize>>,
    pub value: Option<Range<usize>>,
}

/// A parsed config with positions.
#[derive(Debug, Clone)]
pub struct Source {
    /// The config as JSON (TOML datetimes become strings).
    pub value: serde_json::Value,
    spans: HashMap<String, Spans>,
    /// Byte offset of each line start, for line/column lookups.
    line_starts: Vec<usize>,
    /// Offset of this text in the enclosing file (a config block in task.md
    /// starts some way into the file).
    pub base: usize,
}

/// A parse error with its position.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    pub message: String,
    pub span: Option<Range<usize>>,
}

impl Source {
    /// Parse TOML. `base` is the byte offset of `text` in its file.
    pub fn toml(text: &str, base: usize) -> Result<Self, ParseError> {
        let doc = Document::parse(text).map_err(|e| ParseError {
            message: e.message().trim().to_string(),
            span: e.span().map(|r| r.start + base..r.end + base),
        })?;
        let mut spans = HashMap::new();
        let value = walk_table_like(doc.as_table(), "", &mut spans);
        Ok(Self {
            value,
            spans,
            line_starts: line_starts(text),
            base,
        })
    }

    /// Parse YAML (no positions beyond the document; YAML configs are the
    /// rare form). `base` is the byte offset of `text` in its file.
    pub fn yaml(text: &str, base: usize) -> Result<Self, ParseError> {
        let value: serde_json::Value = serde_yaml_ng::from_str(text).map_err(|e| ParseError {
            message: e.to_string(),
            span: e.location().map(|l| l.index() + base..l.index() + base + 1),
        })?;
        let value = if value.is_null() {
            serde_json::Value::Object(Default::default())
        } else {
            value
        };
        Ok(Self {
            value,
            spans: HashMap::new(),
            line_starts: line_starts(text),
            base,
        })
    }

    /// The spans recorded for a path, offset into the enclosing file.
    pub fn spans(&self, path: &str) -> Spans {
        let s = self.spans.get(path).cloned().unwrap_or_default();
        Spans {
            key: s.key.map(|r| r.start + self.base..r.end + self.base),
            value: s.value.map(|r| r.start + self.base..r.end + self.base),
        }
    }

    /// The span of a key, falling back to its value, then to the nearest
    /// parent that has one.
    pub fn best_span(&self, path: &str) -> Option<Range<usize>> {
        let mut p = path.to_string();
        loop {
            let s = self.spans(&p);
            if let Some(r) = s.key.or(s.value) {
                return Some(r);
            }
            p = parent(&p)?;
        }
    }

    /// The span of a value, falling back to [`Self::best_span`].
    pub fn value_span(&self, path: &str) -> Option<Range<usize>> {
        self.spans(path).value.or_else(|| self.best_span(path))
    }

    /// 1-based line of a byte offset in the enclosing file, counting from
    /// the start of this text (add the text's first line yourself).
    pub fn line_in_text(&self, file_offset: usize) -> usize {
        let off = file_offset.saturating_sub(self.base);
        match self.line_starts.binary_search(&off) {
            Ok(i) => i + 1,
            Err(i) => i,
        }
    }
}

/// Byte offsets where lines start.
pub fn line_starts(text: &str) -> Vec<usize> {
    let mut v = vec![0];
    for (i, b) in text.bytes().enumerate() {
        if b == b'\n' {
            v.push(i + 1);
        }
    }
    v
}

/// 1-based (line, column) of a byte offset in `text`. Columns count chars.
pub fn line_col(text: &str, offset: usize) -> (usize, usize) {
    let offset = offset.min(text.len());
    let before = &text[..offset];
    let line = before.matches('\n').count() + 1;
    let col = before.rsplit('\n').next().map_or(0, |s| s.chars().count()) + 1;
    (line, col)
}

/// Join a path and a key, quoting the key when it needs it.
pub fn join(prefix: &str, key: &str) -> String {
    let needs_quotes = key.is_empty()
        || !key
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
    let k = if needs_quotes {
        format!("\"{}\"", key.replace('\\', "\\\\").replace('"', "\\\""))
    } else {
        key.to_string()
    };
    if prefix.is_empty() {
        k
    } else {
        format!("{prefix}.{k}")
    }
}

fn parent(path: &str) -> Option<String> {
    if let Some(stripped) = path.strip_suffix(']') {
        return stripped.rfind('[').map(|i| path[..i].to_string());
    }
    // Split at the last dot outside quotes.
    let mut in_q = false;
    let mut last = None;
    for (i, c) in path.char_indices() {
        match c {
            '"' => in_q = !in_q,
            '.' if !in_q => last = Some(i),
            _ => {}
        }
    }
    last.map(|i| path[..i].to_string())
}

trait TableLike {
    fn entries(&self) -> Vec<(&toml_edit::Key, &Item)>;
}

impl TableLike for toml_edit::Table {
    fn entries(&self) -> Vec<(&toml_edit::Key, &Item)> {
        self.iter()
            .filter_map(|(k, _)| self.get_key_value(k))
            .collect()
    }
}

fn walk_table_like(
    table: &toml_edit::Table,
    prefix: &str,
    spans: &mut HashMap<String, Spans>,
) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    for (key, item) in table.entries() {
        let path = join(prefix, key.get());
        let value = walk_item(item, &path, spans);
        let entry = spans.entry(path.clone()).or_default();
        if entry.key.is_none() {
            entry.key = key.span();
        }
        if entry.value.is_none() {
            entry.value = item.span();
        }
        map.insert(key.get().to_string(), value);
    }
    serde_json::Value::Object(map)
}

fn walk_item(item: &Item, path: &str, spans: &mut HashMap<String, Spans>) -> serde_json::Value {
    match item {
        Item::None => serde_json::Value::Null,
        Item::Value(v) => walk_value(v, path, spans),
        Item::Table(t) => {
            let v = walk_table_like(t, path, spans);
            if let Some(span) = t.span() {
                spans
                    .entry(path.to_string())
                    .or_default()
                    .key
                    .get_or_insert(span);
            }
            v
        }
        Item::ArrayOfTables(arr) => {
            let mut out = Vec::new();
            for (i, t) in arr.iter().enumerate() {
                let p = format!("{path}[{i}]");
                let v = walk_table_like(t, &p, spans);
                if let Some(span) = t.span() {
                    spans.entry(p).or_default().key.get_or_insert(span);
                }
                out.push(v);
            }
            serde_json::Value::Array(out)
        }
    }
}

fn walk_value(v: &Value, path: &str, spans: &mut HashMap<String, Spans>) -> serde_json::Value {
    if let Some(span) = v.span() {
        spans
            .entry(path.to_string())
            .or_default()
            .value
            .get_or_insert(span);
    }
    match v {
        Value::String(s) => serde_json::Value::String(s.value().clone()),
        Value::Integer(i) => serde_json::Value::from(*i.value()),
        Value::Float(f) => serde_json::Number::from_f64(*f.value())
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        Value::Boolean(b) => serde_json::Value::Bool(*b.value()),
        Value::Datetime(d) => serde_json::Value::String(d.value().to_string()),
        Value::Array(a) => serde_json::Value::Array(
            a.iter()
                .enumerate()
                .map(|(i, v)| walk_value(v, &format!("{path}[{i}]"), spans))
                .collect(),
        ),
        Value::InlineTable(t) => {
            let mut map = serde_json::Map::new();
            for (k, item) in t.iter() {
                let p = join(path, k);
                let val = walk_value(item, &p, spans);
                if let Some((key, _)) = t.get_key_value(k) {
                    spans
                        .entry(p.clone())
                        .or_default()
                        .key
                        .get_or_insert_with(|| key.span().unwrap_or(0..0));
                }
                map.insert(k.to_string(), val);
            }
            serde_json::Value::Object(map)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEXT: &str = "name = \"a/b\"\n\n[agent]\ntimout = \"10m\"\n\n[sandbox]\noutputs = [{ path = \"/x\" }]\nenv = { \"A.B\" = \"1\" }\n\n[[credits]]\nname = \"Ada\"\n";

    #[test]
    fn values_and_spans() {
        let src = Source::toml(TEXT, 0).unwrap();
        assert_eq!(src.value["agent"]["timout"], "10m");
        let key = src.spans("agent.timout").key.unwrap();
        assert_eq!(&TEXT[key.clone()], "timout");
        assert_eq!(line_col(TEXT, key.start), (4, 1));
        let v = src.spans("agent.timout").value.unwrap();
        assert_eq!(&TEXT[v], "\"10m\"");
        let p = src.spans("sandbox.outputs[0].path").value.unwrap();
        assert_eq!(&TEXT[p], "\"/x\"");
        assert!(src.spans("sandbox.env.\"A.B\"").value.is_some());
        let n = src.spans("credits[0].name").key.unwrap();
        assert_eq!(&TEXT[n], "name");
    }

    #[test]
    fn base_offsets_shift_spans() {
        let src = Source::toml(TEXT, 100).unwrap();
        let key = src.spans("agent.timout").key.unwrap();
        assert_eq!(&TEXT[key.start - 100..key.end - 100], "timout");
    }

    #[test]
    fn best_span_walks_up() {
        let src = Source::toml(TEXT, 0).unwrap();
        assert!(src.best_span("agent.nothing").is_some());
    }

    #[test]
    fn parse_errors_carry_a_span() {
        let err = Source::toml("a = \n", 10).unwrap_err();
        assert!(err.span.is_some());
        assert!(err.span.unwrap().start >= 10);
    }

    #[test]
    fn yaml_parses_to_json() {
        let src = Source::yaml("agent:\n  timeout: 10m\n", 0).unwrap();
        assert_eq!(src.value["agent"]["timeout"], "10m");
    }
}
