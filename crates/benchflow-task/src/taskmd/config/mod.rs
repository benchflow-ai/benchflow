//! The config block: parsing it, and every check that reads only the config.
//!
//! [`Checker::run`] ports the reference's `check_config` with everything it
//! calls (`_check_nested`, `check_outputs`, `_check_ready`,
//! `check_snapshots`, `_check_verifier_keys`, the clock, family, and
//! training rules). It needs no file system, so it can run again on an
//! edited config to test whether a fix holds.

mod clock;
mod family;
mod judges;
mod keys;
mod values;

use std::collections::HashMap;
use std::ops::Range;
use std::path::Path;

use serde_json::Value;
use toml_edit::{Document, Item};

use crate::diag::Diagnostic;
use crate::source::{Source, join};

use super::code;
use super::report::{At, ConfigLoc, Locate, Sink};
use super::scan::{Block, BlockKind};
use super::text::Text;

pub(crate) use keys::{dym_or, suggestion};

/// The language a config block is written in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Lang {
    Toml,
    Yaml,
}

/// A parsed config block: what the checks need beyond the config's
/// [`Source`], which the document holds.
#[derive(Debug, Clone)]
pub(crate) struct ConfigDoc {
    pub lang: Lang,
    /// The block's body, normalized.
    pub body: String,
    /// Where the body starts in the normalized text.
    pub body_start: usize,
    /// The line of the block's opening fence.
    pub block_line: usize,
    /// TOML date-time values, by path. The config holds them as strings;
    /// the clock rules need to know which were date-times.
    pub datetimes: HashMap<String, toml_edit::Datetime>,
}

impl ConfigDoc {
    /// Parse a config block, reporting a syntax error (and returning an
    /// empty config) when it does not parse.
    pub(crate) fn parse(block: &Block, sink: &mut Sink<'_>, text: &Text) -> (Source, Self) {
        let lang = if block.arg.as_deref() == Some("toml") {
            Lang::Toml
        } else {
            Lang::Yaml
        };
        let parsed = match lang {
            Lang::Toml => Source::toml(&block.body, block.body_start),
            Lang::Yaml => Source::yaml(&block.body, block.body_start),
        };
        let source = match parsed {
            Ok(src) if src.value.is_object() => src,
            Ok(_) => {
                sink.error(
                    code::CONFIG_SYNTAX,
                    At::Line(block.line),
                    "config block must be a mapping",
                );
                empty_source(block.body_start)
            }
            Err(e) => {
                let what = if lang == Lang::Toml { "TOML" } else { "YAML" };
                let at = match &e.span {
                    Some(r) => {
                        let (s, end) = text.span(r.clone());
                        At::Span(s, end)
                    }
                    None => At::Line(block.line),
                };
                sink.error(
                    code::CONFIG_SYNTAX,
                    at,
                    format!("config block is not valid {what}: {}", e.message),
                )
                .label = Some("the config stops parsing here".into());
                empty_source(block.body_start)
            }
        };
        let datetimes = if lang == Lang::Toml {
            toml_datetimes(&block.body)
        } else {
            HashMap::new()
        };
        let doc = Self {
            lang,
            body: block.body.clone(),
            body_start: block.body_start,
            block_line: block.line,
            datetimes,
        };
        (source, doc)
    }

    /// The metadata of a document with no config block.
    pub(crate) fn none() -> Self {
        Self {
            lang: Lang::Toml,
            body: String::new(),
            body_start: 0,
            block_line: 1,
            datetimes: HashMap::new(),
        }
    }
}

/// An empty config whose spans start at `base`.
pub(crate) fn empty_source(base: usize) -> Source {
    Source::toml("", base).expect("an empty document parses")
}

/// The paths of TOML date-time values in `body`.
fn toml_datetimes(body: &str) -> HashMap<String, toml_edit::Datetime> {
    fn walk_value(
        v: &toml_edit::Value,
        path: &str,
        out: &mut HashMap<String, toml_edit::Datetime>,
    ) {
        match v {
            toml_edit::Value::Datetime(d) => {
                out.insert(path.to_string(), *d.value());
            }
            toml_edit::Value::Array(a) => {
                for (i, x) in a.iter().enumerate() {
                    walk_value(x, &format!("{path}[{i}]"), out);
                }
            }
            toml_edit::Value::InlineTable(t) => {
                for (k, x) in t.iter() {
                    walk_value(x, &join(path, k), out);
                }
            }
            _ => {}
        }
    }
    fn walk_item(item: &Item, path: &str, out: &mut HashMap<String, toml_edit::Datetime>) {
        match item {
            Item::Value(v) => walk_value(v, path, out),
            Item::Table(t) => {
                for (k, x) in t.iter() {
                    walk_item(x, &join(path, k), out);
                }
            }
            Item::ArrayOfTables(a) => {
                for (i, t) in a.iter().enumerate() {
                    for (k, x) in t.iter() {
                        walk_item(x, &join(&format!("{path}[{i}]"), k), out);
                    }
                }
            }
            Item::None => {}
        }
    }
    let mut out = HashMap::new();
    if let Ok(doc) = Document::parse(body) {
        for (k, x) in doc.as_table().iter() {
            walk_item(x, &join("", k), &mut out);
        }
    }
    out
}

/// Runs the config checks over one config.
pub(crate) struct Checker<'a> {
    pub config: &'a Value,
    pub blocks: &'a [Block],
    /// The parsed block and where it sits, when checking task.md itself.
    /// `None` while testing a fix: then no fixes are proposed and the slow
    /// family overlap scan is skipped.
    pub doc: Option<(&'a ConfigDoc, &'a ConfigLoc<'a>)>,
    /// The package folder, for the few config checks that look at a file
    /// (a judge's brief); `None` while testing a fix.
    pub dir: Option<&'a Path>,
    /// Read the files the config names (a judge's brief), not only check
    /// that they exist.
    pub deep: bool,
    pub sink: Sink<'a>,
}

impl<'a> Checker<'a> {
    pub(crate) fn new(
        config: &'a Value,
        blocks: &'a [Block],
        doc: Option<(&'a ConfigDoc, &'a ConfigLoc<'a>)>,
    ) -> Self {
        let loc = doc.map(|(_, l)| l as &dyn Locate);
        Self {
            config,
            blocks,
            doc,
            dir: None,
            deep: false,
            sink: Sink::new(loc),
        }
    }

    fn err(
        &mut self,
        code: &'static str,
        at: At<'_>,
        message: impl Into<String>,
    ) -> &mut Diagnostic {
        self.sink.error(code, at, message)
    }

    fn warn(
        &mut self,
        code: &'static str,
        at: At<'_>,
        message: impl Into<String>,
    ) -> &mut Diagnostic {
        self.sink.warning(code, at, message)
    }

    /// Whether replacing `range` (offsets in the normalized text) with
    /// `replacement` leaves a config whose checks report no error at or
    /// under `new_path` and no longer report `message`.
    fn fix_holds(
        &self,
        range: Range<usize>,
        replacement: &str,
        new_path: &str,
        message: &str,
    ) -> bool {
        let Some((doc, _)) = self.doc else {
            return false;
        };
        if doc.lang != Lang::Toml || range.start < doc.body_start {
            return false;
        }
        let (s, mut e) = (range.start - doc.body_start, range.end - doc.body_start);
        // A whole-line edit of the body's last line also takes the line end
        // before the closing fence, which the body does not hold.
        if e == doc.body.len() + 1 {
            e -= 1;
        }
        if s > e
            || e > doc.body.len()
            || !doc.body.is_char_boundary(s)
            || !doc.body.is_char_boundary(e)
        {
            return false;
        }
        let edited = format!("{}{replacement}{}", &doc.body[..s], &doc.body[e..]);
        let Ok(src) = Source::toml(&edited, 0) else {
            return false;
        };
        let mut again = Checker::new(&src.value, self.blocks, None);
        again.run();
        let under = |p: &str| {
            p == new_path
                || p.starts_with(&format!("{new_path}."))
                || p.starts_with(&format!("{new_path}["))
        };
        !again.sink.found.iter().any(|f| {
            f.diag.is_error() && (f.path.as_deref().is_some_and(under) || f.diag.message == message)
        })
    }

    /// The span from the start of a key to the end of its value, in the
    /// normalized text. (An edit over a table's header breaks the TOML, so
    /// [`Checker::fix_holds`] refuses it.)
    fn key_to_value(&self, path: &str) -> Option<Range<usize>> {
        let (_, loc) = self.doc?;
        let key = loc.own_key_range(path)?;
        let value = loc.source.spans(path).value?;
        (value.end > key.start).then_some(key.start..value.end)
    }

    /// The span of a key in the normalized text (a header's table name, not
    /// the whole header).
    fn key_range(&self, path: &str) -> Option<Range<usize>> {
        let (_, loc) = self.doc?;
        loc.own_key_range(path)
    }

    /// The span of a value in the normalized text.
    fn value_range(&self, path: &str) -> Option<Range<usize>> {
        let (_, loc) = self.doc?;
        loc.source.spans(path).value
    }

    /// `range` widened to whole lines when nothing else is on them, so that
    /// deleting it leaves no blank line behind.
    fn whole_lines(&self, range: Range<usize>) -> Range<usize> {
        let Some((_, loc)) = self.doc else {
            return range;
        };
        let text = loc.text.as_str();
        let start = text[..range.start].rfind('\n').map_or(0, |i| i + 1);
        let end = text[range.end..]
            .find('\n')
            .map_or(text.len(), |i| range.end + i);
        if text[start..range.start].trim().is_empty() && text[range.end..end].trim().is_empty() {
            start..(end + 1).min(text.len())
        } else {
            range
        }
    }

    /// A machine-applicable fix over a range of the normalized text.
    fn make_fix(
        &self,
        range: Range<usize>,
        replacement: String,
        description: String,
    ) -> Option<crate::diag::Fix> {
        let (_, loc) = self.doc?;
        Some(crate::diag::Fix {
            file: loc.file.to_path_buf(),
            span: loc.text.span(range),
            replacement,
            description,
        })
    }

    /// The TOML date-time at `path`, when the config is TOML and the value
    /// was written as one.
    fn datetime(&self, path: &str) -> Option<toml_edit::Datetime> {
        self.doc.and_then(|(d, _)| d.datetimes.get(path).copied())
    }

    /// Every config check, in the reference's order.
    pub(crate) fn run(&mut self) {
        let config = self.config;
        let table = super::pyval::table;
        keys::top_level(self);
        keys::closed_tables(self);
        keys::about_opt_out(self);
        values::enums(self);
        values::verifier_keys(self, table(config, &["verifier"]), "[verifier]", "verifier");
        for (name, st) in table(config, &["stages"]) {
            if let Some(Value::Object(v)) = st.get("verifier") {
                let path = join(&join("stages", name), "verifier");
                values::verifier_keys(self, v, &format!("[stages.{name}.verifier]"), &path);
            }
        }
        if let Some(Value::Object(vs)) = table(config, &["verifier"]).get("sandbox") {
            values::phase(self, vs, "[verifier.sandbox]", "verifier.sandbox");
        }
        let sandbox = table(config, &["sandbox"]);
        values::ready(
            self,
            sandbox.get("ready"),
            "[sandbox] ready",
            "sandbox.ready",
        );
        values::services(self);
        values::budget(self);
        values::credits_and_authors(self);
        keys::nested(self);
        values::snapshots(self, table(config, &["verifier"]), "[verifier]", "verifier");
        for (name, st) in table(config, &["stages"]) {
            if let Some(Value::Object(v)) = st.get("verifier") {
                let path = join(&join("stages", name), "verifier");
                values::snapshots(self, v, &format!("[stages.{name}.verifier]"), &path);
            }
        }
        values::world_kind(self);
        values::stages(self);
        values::outputs(self, sandbox.get("outputs"), "[sandbox]", "sandbox");
        clock::check(self);
        family::check(self);
        values::training(self);
        values::injection(self);
        values::roles(self);
        judges::check(self);
    }

    /// The names of the stage blocks.
    fn stage_blocks(&self) -> Vec<&'a str> {
        self.blocks
            .iter()
            .filter(|b| b.kind == BlockKind::Stage)
            .filter_map(|b| b.arg.as_deref())
            .collect()
    }

    /// The line of the first block of `kind` named `name`.
    fn block_line(&self, kind: BlockKind, name: &str) -> Option<usize> {
        self.blocks
            .iter()
            .find(|b| b.kind == kind && b.arg.as_deref() == Some(name))
            .map(|b| b.line)
    }
}

/// Run the config checks on a document's config. `dir` is the package
/// folder; with `deep`, the files the config names are read too.
pub(crate) fn check_config(
    doc: &ConfigDoc,
    loc: &ConfigLoc<'_>,
    blocks: &[Block],
    dir: &Path,
    deep: bool,
) -> Vec<Diagnostic> {
    let mut checker = Checker::new(&loc.source.value, blocks, Some((doc, loc)));
    checker.dir = Some(dir);
    checker.deep = deep;
    checker.run();
    checker.sink.into_diagnostics()
}

/// A TOML literal for a value, for fixes: strings, numbers, booleans, and
/// arrays of them.
pub(crate) fn toml_literal(v: &Value) -> Option<String> {
    Some(match v {
        Value::String(s) => toml_edit::Value::from(s.as_str())
            .to_string()
            .trim()
            .to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::Array(items) => {
            let parts: Option<Vec<String>> = items.iter().map(toml_literal).collect();
            format!("[{}]", parts?.join(", "))
        }
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn toml_literals() {
        assert_eq!(toml_literal(&json!("10m")).as_deref(), Some("\"10m\""));
        assert_eq!(
            toml_literal(&json!(["x", 1])).as_deref(),
            Some("[\"x\", 1]")
        );
        assert_eq!(toml_literal(&json!({})), None);
        let quoted = toml_literal(&json!("a\"b")).unwrap();
        let back: toml_edit::Value = quoted.parse().unwrap();
        assert_eq!(back.as_str(), Some("a\"b"));
    }

    #[test]
    fn datetimes_are_found() {
        let d =
            toml_datetimes("[sandbox]\nclock = { start = 2026-03-14T09:00:00 }\nx = \"2026\"\n");
        assert!(d.contains_key("sandbox.clock.start"));
        assert_eq!(d.len(), 1);
    }
}
