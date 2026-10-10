//! Diagnostics: what a check found, where, and what to do about it.
//!
//! A [`Diagnostic`] carries a stable code (`T104`), a one-line message, an
//! optional location with a byte span into the file, notes, a help line, and
//! an optional machine-applicable [`Fix`]. The CLI renders them rustc-style;
//! `--json` emits them as they are.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Error,
    Warning,
    Note,
}

/// A place in a file. `span` is a byte range into the file's text when known;
/// `line` (1-based) is always set when the location is in a file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Location {
    pub file: PathBuf,
    pub line: Option<usize>,
    pub span: Option<(usize, usize)>,
}

/// A machine-applicable edit: replace `span` in `file` with `replacement`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fix {
    pub file: PathBuf,
    pub span: (usize, usize),
    pub replacement: String,
    /// One line for `--fix` output: "rename `timout` to `timeout`".
    pub description: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Diagnostic {
    /// Stable code, such as `T104`. `bench explain T104` documents it.
    pub code: &'static str,
    pub severity: Severity,
    /// One line, lower case, no trailing period: what is wrong.
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub location: Option<Location>,
    /// Text shown under the caret at the span ("did you mean `timeout`?").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// Extra facts ("[agent] holds: timeout, user, ...").
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
    /// What to do.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub help: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fix: Option<Fix>,
}

impl Diagnostic {
    pub fn new(code: &'static str, severity: Severity, message: impl Into<String>) -> Self {
        Self {
            code,
            severity,
            message: message.into(),
            location: None,
            label: None,
            notes: Vec::new(),
            help: None,
            fix: None,
        }
    }
    pub fn error(code: &'static str, message: impl Into<String>) -> Self {
        Self::new(code, Severity::Error, message)
    }
    pub fn warning(code: &'static str, message: impl Into<String>) -> Self {
        Self::new(code, Severity::Warning, message)
    }
    pub fn note(code: &'static str, message: impl Into<String>) -> Self {
        Self::new(code, Severity::Note, message)
    }
    pub fn at(
        mut self,
        file: impl Into<PathBuf>,
        line: Option<usize>,
        span: Option<(usize, usize)>,
    ) -> Self {
        self.location = Some(Location {
            file: file.into(),
            line,
            span,
        });
        self
    }
    pub fn in_file(self, file: impl Into<PathBuf>) -> Self {
        self.at(file, None, None)
    }
    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
    pub fn note_line(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }
    pub fn help(mut self, help: impl Into<String>) -> Self {
        self.help = Some(help.into());
        self
    }
    pub fn fix(mut self, fix: Fix) -> Self {
        self.fix = Some(fix);
        self
    }
    pub fn is_error(&self) -> bool {
        self.severity == Severity::Error
    }
}

/// True when any diagnostic is an error.
pub fn has_errors(diags: &[Diagnostic]) -> bool {
    diags.iter().any(Diagnostic::is_error)
}

/// Every diagnostic code, with a title and a long explanation for
/// `bench explain`. Codes are grouped by letter: `T` task document and
/// config, `R` rubric and verifier files, `P` package layout, `X` import and
/// export.
pub fn explain(code: &str) -> Option<&'static Explanation> {
    crate::codes::all()
        .into_iter()
        .find(|e| e.code.eq_ignore_ascii_case(code))
}

#[derive(Debug)]
pub struct Explanation {
    pub code: &'static str,
    pub title: &'static str,
    pub body: &'static str,
}
