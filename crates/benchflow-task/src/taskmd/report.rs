//! Collecting diagnostics, and pointing each one at its place in a file.
//!
//! Checks name what they are about by path (`agent.timeout`,
//! `criteria[2].judge`) through [`At`]; a [`Locate`] turns that into a file,
//! a line, and a byte span. A check run with no locator (to test whether a
//! fix would leave errors) still records each diagnostic's path.

use std::ops::Range;
use std::path::{Path, PathBuf};

use crate::diag::{Diagnostic, Location, Severity};
use crate::source::Source;

use super::json_spans::JsonSpans;
use super::text::Text;

/// What a diagnostic is about.
#[derive(Debug, Clone, Copy)]
pub(crate) enum At<'p> {
    /// A key, by path: the span of the key, else of its value, else of the
    /// nearest table that holds it.
    Key(&'p str),
    /// A value, by path: the span of the value, else as [`At::Key`].
    Value(&'p str),
    /// A 1-based line of the file.
    Line(usize),
    /// A byte range of the file as written.
    Span(usize, usize),
    /// The file as a whole.
    File,
}

impl At<'_> {
    fn path(&self) -> Option<&str> {
        match self {
            At::Key(p) | At::Value(p) => Some(p),
            _ => None,
        }
    }
}

/// Turns an [`At`] into a place in one file.
pub(crate) trait Locate {
    fn locate(&self, at: At<'_>) -> Location;
}

/// A diagnostic and the path it is about.
#[derive(Debug, Clone)]
pub(crate) struct Found {
    pub path: Option<String>,
    pub diag: Diagnostic,
}

/// Collects diagnostics.
pub(crate) struct Sink<'a> {
    loc: Option<&'a dyn Locate>,
    pub found: Vec<Found>,
}

impl<'a> Sink<'a> {
    pub(crate) fn new(loc: Option<&'a dyn Locate>) -> Self {
        Self {
            loc,
            found: Vec::new(),
        }
    }

    /// Record a diagnostic; returns it so the caller can add a label, a
    /// help line, or a fix.
    pub(crate) fn push(
        &mut self,
        code: &'static str,
        severity: Severity,
        at: At<'_>,
        message: impl Into<String>,
    ) -> &mut Diagnostic {
        let mut diag = Diagnostic::new(code, severity, message);
        diag.location = self.loc.map(|l| l.locate(at));
        self.found.push(Found {
            path: at.path().map(str::to_string),
            diag,
        });
        &mut self.found.last_mut().expect("just pushed").diag
    }

    pub(crate) fn error(
        &mut self,
        code: &'static str,
        at: At<'_>,
        message: impl Into<String>,
    ) -> &mut Diagnostic {
        self.push(code, Severity::Error, at, message)
    }

    pub(crate) fn warning(
        &mut self,
        code: &'static str,
        at: At<'_>,
        message: impl Into<String>,
    ) -> &mut Diagnostic {
        self.push(code, Severity::Warning, at, message)
    }

    /// The diagnostics, without their paths.
    pub(crate) fn into_diagnostics(self) -> Vec<Diagnostic> {
        self.found.into_iter().map(|f| f.diag).collect()
    }
}

/// Places in task.md's config block, through the config's [`Source`].
pub(crate) struct ConfigLoc<'a> {
    pub file: &'a Path,
    pub text: &'a Text,
    pub source: &'a Source,
    /// Where to point when the config has no spans (a YAML block): the
    /// block's first line.
    pub block_line: usize,
}

impl ConfigLoc<'_> {
    /// The span, in the file as written, of a range in the normalized text.
    fn place(&self, range: Option<Range<usize>>) -> Location {
        match range {
            Some(r) => Location {
                file: self.file.to_path_buf(),
                line: Some(self.text.line(r.start)),
                span: Some(self.text.span(r)),
            },
            None => Location {
                file: self.file.to_path_buf(),
                line: Some(self.block_line),
                span: None,
            },
        }
    }

    /// The span of a key, or of the nearest key that holds it, in the
    /// normalized text. A table written with a header (`[a.b]`, `[[c]]`) is
    /// narrowed to the header's last key, the table's own name.
    pub(crate) fn key_range(&self, path: &str) -> Option<Range<usize>> {
        self.source
            .best_span(path)
            .map(|r| narrow_header(self.text.as_str(), r))
    }

    /// The span of exactly this key (not a parent's), narrowed as
    /// [`ConfigLoc::key_range`] narrows it.
    pub(crate) fn own_key_range(&self, path: &str) -> Option<Range<usize>> {
        self.source
            .spans(path)
            .key
            .map(|r| narrow_header(self.text.as_str(), r))
    }
}

/// A table header's span narrowed to its last key: `[world.desktop]` to
/// `desktop`, `[[credits]]` to `credits`, `[a."b.c"]` to `"b.c"`. Any other
/// span is returned as it is.
pub(crate) fn narrow_header(text: &str, range: Range<usize>) -> Range<usize> {
    let Some(header) = text.get(range.clone()) else {
        return range;
    };
    if !header.starts_with('[') {
        return range;
    }
    let open = if header.starts_with("[[") { 2 } else { 1 };
    let mut last_start = open;
    let mut quote: Option<char> = None;
    let mut escaped = false;
    for (i, c) in header.char_indices().skip(open) {
        match quote {
            Some(q) => {
                if escaped {
                    escaped = false;
                } else if c == '\\' && q == '"' {
                    escaped = true;
                } else if c == q {
                    quote = None;
                }
            }
            None => match c {
                '"' | '\'' => quote = Some(c),
                '.' => last_start = i + 1,
                ']' => {
                    let key = &header[last_start..i];
                    let lead = key.len() - key.trim_start().len();
                    let start = range.start + last_start + lead;
                    return start..start + key.trim().len();
                }
                _ => {}
            },
        }
    }
    range
}

impl Locate for ConfigLoc<'_> {
    fn locate(&self, at: At<'_>) -> Location {
        match at {
            At::Key(p) => self.place(self.key_range(p)),
            At::Value(p) => self.place(self.source.value_span(p)),
            At::Line(n) => Location {
                file: self.file.to_path_buf(),
                line: Some(n),
                span: Some(self.text.line_span(n)),
            },
            At::Span(s, e) => Location {
                file: self.file.to_path_buf(),
                line: Some(self.text.line(s)),
                span: Some((s, e)),
            },
            At::File => Location {
                file: self.file.to_path_buf(),
                line: None,
                span: None,
            },
        }
    }
}

/// Places in a JSON file (`verifier/rubric.json`), read as written.
pub(crate) struct JsonLoc {
    pub file: PathBuf,
    pub text: String,
    pub spans: JsonSpans,
}

impl JsonLoc {
    pub(crate) fn new(file: impl Into<PathBuf>, text: &str) -> Self {
        Self {
            file: file.into(),
            spans: JsonSpans::index(text),
            text: text.to_string(),
        }
    }

    fn place(&self, range: Option<Range<usize>>) -> Location {
        let span = range.map(|r| (r.start, r.end));
        Location {
            file: self.file.clone(),
            line: Some(span.map_or(1, |(s, _)| line_of(&self.text, s))),
            span,
        }
    }

    /// The span of the value at `path`, else of the nearest parent with one.
    fn value_or_parent(&self, path: &str) -> Option<Range<usize>> {
        let mut p = path.to_string();
        loop {
            if let Some(r) = self.spans.value(&p) {
                return Some(r);
            }
            p = parent(&p)?;
        }
    }
}

impl Locate for JsonLoc {
    fn locate(&self, at: At<'_>) -> Location {
        match at {
            At::Key(p) => self.place(self.spans.key(p).or_else(|| self.value_or_parent(p))),
            At::Value(p) => self.place(self.value_or_parent(p)),
            At::Line(n) => Location {
                file: self.file.clone(),
                line: Some(n),
                span: None,
            },
            At::Span(s, e) => Location {
                file: self.file.clone(),
                line: Some(line_of(&self.text, s)),
                span: Some((s, e)),
            },
            At::File => Location {
                file: self.file.clone(),
                line: None,
                span: None,
            },
        }
    }
}

/// Places in a plain file (a script, a brief), by line or byte range.
pub(crate) struct FileLoc {
    pub file: PathBuf,
    pub text: String,
}

impl Locate for FileLoc {
    fn locate(&self, at: At<'_>) -> Location {
        let (line, span) = match at {
            At::Line(n) => (Some(n), line_span(&self.text, n)),
            At::Span(s, e) => (Some(line_of(&self.text, s)), Some((s, e))),
            _ => (None, None),
        };
        Location {
            file: self.file.clone(),
            line,
            span,
        }
    }
}

/// The 1-based line of a byte offset.
pub(crate) fn line_of(text: &str, offset: usize) -> usize {
    text.as_bytes()[..offset.min(text.len())]
        .iter()
        .filter(|b| **b == b'\n')
        .count()
        + 1
}

/// The span of a 1-based line, without its line end.
pub(crate) fn line_span(text: &str, line: usize) -> Option<(usize, usize)> {
    let mut start = 0;
    for _ in 1..line {
        start += text[start..].find('\n')? + 1;
    }
    let end = text[start..].find('\n').map_or(text.len(), |i| start + i);
    let end = if text[start..end].ends_with('\r') {
        end - 1
    } else {
        end
    };
    Some((start, end))
}

/// The parent of a path: `a.b[2]` is `a.b`, `a.b` is `a`.
pub(crate) fn parent(path: &str) -> Option<String> {
    if let Some(stripped) = path.strip_suffix(']') {
        return stripped.rfind('[').map(|i| path[..i].to_string());
    }
    let mut in_quotes = false;
    let mut last = None;
    for (i, c) in path.char_indices() {
        match c {
            '"' => in_quotes = !in_quotes,
            '.' if !in_quotes => last = Some(i),
            _ => {}
        }
    }
    last.map(|i| path[..i].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_locations_fall_back_to_parents() {
        let text = "{\n  \"criteria\": [\n    {\"id\": \"a\"}\n  ]\n}\n";
        let loc = JsonLoc::new("verifier/rubric.json", text);
        let l = loc.locate(At::Key("criteria[0].text"));
        assert_eq!(l.line, Some(3));
        assert_eq!(
            &text[l.span.unwrap().0..l.span.unwrap().1],
            "{\"id\": \"a\"}"
        );
        let l = loc.locate(At::Value("criteria[0].id"));
        assert_eq!(&text[l.span.unwrap().0..l.span.unwrap().1], "\"a\"");
    }

    #[test]
    fn headers_narrow_to_their_last_key() {
        let text = "[world.desktop]\n[[credits]]\n[ a . \"b.c]\" ]\nx = 1\n";
        let narrow = |s: usize, e: usize| {
            let r = narrow_header(text, s..e);
            &text[r]
        };
        assert_eq!(narrow(0, 15), "desktop");
        assert_eq!(narrow(16, 27), "credits");
        assert_eq!(narrow(28, 42), "\"b.c]\"");
        assert_eq!(narrow(43, 48), "x = 1");
    }

    #[test]
    fn lines_and_parents() {
        assert_eq!(line_of("a\nb\nc", 4), 3);
        assert_eq!(line_span("a\r\nbc\n", 2), Some((3, 5)));
        assert_eq!(parent("a.b[2]").as_deref(), Some("a.b"));
        assert_eq!(parent("a.\"x.y\"").as_deref(), Some("a"));
        assert_eq!(parent("a"), None);
    }
}
