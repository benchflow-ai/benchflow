//! The result of checking one package.

use std::path::PathBuf;

use serde::Serialize;

use crate::diag::{Diagnostic, has_errors};
use crate::package::{Format, TaskPackage};

/// How hard to check.
#[derive(Debug, Clone, Default)]
pub struct CheckOptions {
    /// Also run checks that read files beyond the config and rubric (the
    /// `stated` quotes in sidecar files, the oracle and controls on disk).
    /// On by default in `bench check`; off when loading for a run.
    pub deep: bool,
}

/// What checking a package found.
#[derive(Debug, Clone, Serialize)]
pub struct CheckReport {
    pub root: PathBuf,
    /// The format the folder is in; `None` when it is not a task.
    pub format: Option<Format>,
    /// The normalized package, when the config parsed far enough to build
    /// one. Present even when there are errors in fields it does not need.
    #[serde(skip)]
    pub package: Option<TaskPackage>,
    pub diagnostics: Vec<Diagnostic>,
    /// Facts for people, such as what grading needs and its worst-case time
    /// ("grading needs a model: no", "worst-case grading time: 2m").
    pub facts: Vec<String>,
}

impl CheckReport {
    pub fn new(root: impl Into<PathBuf>, format: Option<Format>) -> Self {
        Self {
            root: root.into(),
            format,
            package: None,
            diagnostics: Vec::new(),
            facts: Vec::new(),
        }
    }
    pub fn ok(&self) -> bool {
        !has_errors(&self.diagnostics)
    }
}
