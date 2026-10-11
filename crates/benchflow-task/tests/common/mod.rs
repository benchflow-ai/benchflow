//! Shared by the differential tests: BenchFlow's wording where the reference
//! tools (task-md's `taskmd.py` and `convert.py`, BenchFlow 0.8) name another
//! evaluation framework. BenchFlow's text names none, so before comparing
//! word for word the tests put each such phrase of the reference's output in
//! BenchFlow's words, from `fixtures/neutral-wording.tsv`, and check with
//! [`named`] that BenchFlow's own text, unmapped, names no framework.
//! Everything else is still compared exactly. The import record keeps its
//! name, `[import.harbor]`, on both sides: it is an identifier other tools
//! read.

#![allow(dead_code)]

use std::path::Path;
use std::sync::LazyLock;

use serde_json::Value;

#[path = "../../../benchflow-cli/src/other_frameworks.rs"]
mod other_frameworks;

pub use other_frameworks::named;

/// Every string in `texts` that names another framework, with what it names.
pub fn naming<'a>(texts: impl IntoIterator<Item = &'a str>) -> Vec<String> {
    texts
        .into_iter()
        .filter(|t| !named(t).is_empty())
        .map(|t| format!("{:?}: {t}", named(t)))
        .collect()
}

/// (the reference's phrase, BenchFlow's), in the order they apply.
static ROWS: LazyLock<Vec<(String, String)>> = LazyLock::new(|| {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/neutral-wording.tsv");
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
        .map(|l| {
            let (old, new) = l
                .split_once('\t')
                .unwrap_or_else(|| panic!("neutral-wording.tsv: no tab in {l:?}"));
            (old.to_string(), new.to_string())
        })
        .collect()
});

/// The reference's `text` with its phrases in BenchFlow's words. A
/// replacement that starts a sentence (after a period or at a line's start)
/// is capitalized. Only ever applied to the reference's side.
pub fn neutral(text: &str) -> String {
    let mut text = text.to_string();
    for (old, new) in ROWS.iter() {
        let mut out = String::with_capacity(text.len());
        let mut rest = text.as_str();
        while let Some(i) = rest.find(old.as_str()) {
            out.push_str(&rest[..i]);
            let before = out.trim_end_matches(' ');
            if before.ends_with('.') || before.ends_with('\n') {
                let mut c = new.chars();
                if let Some(first) = c.next() {
                    out.extend(first.to_uppercase());
                    out.push_str(c.as_str());
                }
            } else {
                out.push_str(new);
            }
            rest = &rest[i + old.len()..];
        }
        out.push_str(rest);
        text = out;
    }
    text
}

/// Every string in `v` (keys included) put through [`neutral`].
pub fn neutral_json(v: &Value) -> Value {
    match v {
        Value::String(s) => Value::String(neutral(s)),
        Value::Array(a) => Value::Array(a.iter().map(neutral_json).collect()),
        Value::Object(m) => Value::Object(
            m.iter()
                .map(|(k, v)| (neutral(k), neutral_json(v)))
                .collect(),
        ),
        other => other.clone(),
    }
}
