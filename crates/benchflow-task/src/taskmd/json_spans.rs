//! Where each key and value sits in a JSON file, by path.
//!
//! `serde_json` reads the values; this module only finds byte ranges, so a
//! diagnostic about `criteria[2].judge` in `verifier/rubric.json` can point at
//! that key. Paths use the same syntax as [`crate::source`]: `criteria[2].id`,
//! `scoring.method`, `"$schema"`.

use std::collections::HashMap;
use std::ops::Range;

use crate::source::{Spans, join};

/// Byte ranges of the keys and values of a JSON text, by path.
#[derive(Debug, Clone, Default)]
pub(crate) struct JsonSpans {
    spans: HashMap<String, Spans>,
}

impl JsonSpans {
    /// Index `text`. Text that is not valid JSON gets the spans found before
    /// the first error.
    pub(crate) fn index(text: &str) -> Self {
        let mut p = Parser {
            b: text.as_bytes(),
            text,
            i: 0,
            spans: HashMap::new(),
        };
        p.ws();
        let _ = p.value("");
        Self { spans: p.spans }
    }

    /// The span of the key at `path`.
    pub(crate) fn key(&self, path: &str) -> Option<Range<usize>> {
        self.spans.get(path).and_then(|s| s.key.clone())
    }

    /// The span of the value at `path`.
    pub(crate) fn value(&self, path: &str) -> Option<Range<usize>> {
        self.spans.get(path).and_then(|s| s.value.clone())
    }
}

struct Parser<'a> {
    b: &'a [u8],
    text: &'a str,
    i: usize,
    spans: HashMap<String, Spans>,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn value(&mut self, path: &str) -> Option<()> {
        let start = self.i;
        match *self.b.get(self.i)? {
            b'{' => self.object(path)?,
            b'[' => self.array(path)?,
            b'"' => {
                self.string()?;
            }
            _ => {
                while self.i < self.b.len()
                    && !matches!(
                        self.b[self.i],
                        b',' | b'}' | b']' | b' ' | b'\t' | b'\n' | b'\r'
                    )
                {
                    self.i += 1;
                }
                if self.i == start {
                    return None;
                }
            }
        }
        self.spans.entry(path.to_string()).or_default().value = Some(start..self.i);
        Some(())
    }

    fn object(&mut self, path: &str) -> Option<()> {
        self.i += 1;
        self.ws();
        if self.b.get(self.i) == Some(&b'}') {
            self.i += 1;
            return Some(());
        }
        loop {
            self.ws();
            let kstart = self.i;
            let key = self.string()?;
            let kspan = kstart..self.i;
            self.ws();
            if self.b.get(self.i) != Some(&b':') {
                return None;
            }
            self.i += 1;
            self.ws();
            let child = join(path, &key);
            // A repeated key: the last one wins, as in serde_json and Python.
            self.spans.remove(&child);
            self.spans.entry(child.clone()).or_default().key = Some(kspan);
            self.value(&child)?;
            self.ws();
            match self.b.get(self.i)? {
                b',' => self.i += 1,
                b'}' => {
                    self.i += 1;
                    return Some(());
                }
                _ => return None,
            }
        }
    }

    fn array(&mut self, path: &str) -> Option<()> {
        self.i += 1;
        self.ws();
        if self.b.get(self.i) == Some(&b']') {
            self.i += 1;
            return Some(());
        }
        let mut n = 0;
        loop {
            self.ws();
            self.value(&format!("{path}[{n}]"))?;
            n += 1;
            self.ws();
            match self.b.get(self.i)? {
                b',' => self.i += 1,
                b']' => {
                    self.i += 1;
                    return Some(());
                }
                _ => return None,
            }
        }
    }

    /// A string literal starting at `self.i`, decoded.
    fn string(&mut self) -> Option<String> {
        if self.b.get(self.i) != Some(&b'"') {
            return None;
        }
        let start = self.i;
        self.i += 1;
        while self.i < self.b.len() {
            match self.b[self.i] {
                b'\\' => self.i += 2,
                b'"' => {
                    self.i += 1;
                    return serde_json::from_str(&self.text[start..self.i]).ok();
                }
                _ => self.i += 1,
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_keys_and_values() {
        let text = r#"{"$schema": "x", "criteria": [{"id": "a", "points": 2}, {"id": "b\"c", "gate": true}]}"#;
        let s = JsonSpans::index(text);
        assert_eq!(&text[s.key("\"$schema\"").unwrap()], "\"$schema\"");
        assert_eq!(&text[s.value("criteria[0].points").unwrap()], "2");
        assert_eq!(&text[s.value("criteria[1].id").unwrap()], "\"b\\\"c\"");
        assert_eq!(&text[s.key("criteria[1].gate").unwrap()], "\"gate\"");
        assert!(text[s.value("criteria[1]").unwrap()].starts_with('{'));
    }

    #[test]
    fn stops_at_errors() {
        let s = JsonSpans::index(r#"{"a": 1, "b": }"#);
        assert!(s.value("a").is_some());
        assert!(s.value("b").is_none());
    }
}
