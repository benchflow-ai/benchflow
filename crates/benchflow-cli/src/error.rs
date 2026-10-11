//! Errors that end a command: what happened, whose problem it is, and what
//! to do next. A bug in BenchFlow writes a crash log instead of a stack trace.

use std::fmt;

use benchflow_run::Owner;

use crate::ui::{self, RED, bold, dim, paint};

/// Exit codes, documented in docs/rust/cli.md.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Exit {
    /// Done; for `run`, every trial scored and every gate held.
    Ok = 0,
    /// Done, but something failed: a check found errors, a trial went
    /// unscored, a gate failed; or something to fetch first (a registry, a
    /// git repository) could not be read, and nothing ran.
    Failed = 1,
    /// A bad flag or argument; nothing ran.
    Usage = 2,
    /// The machine is not ready: no sandbox backend, no login.
    NotReady = 3,
}

#[derive(Debug)]
pub struct CliError {
    pub message: String,
    /// A failure code `bench explain` documents (`R211`), when it has one.
    pub code: Option<String>,
    pub owner: Option<Owner>,
    pub next: Option<String>,
    pub detail: Option<String>,
    pub exit: Exit,
}

impl CliError {
    pub fn new(exit: Exit, message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            code: None,
            owner: None,
            next: None,
            detail: None,
            exit,
        }
    }
    pub fn usage(message: impl Into<String>) -> Self {
        Self::new(Exit::Usage, message).owner(Owner::You)
    }
    pub fn not_ready(message: impl Into<String>) -> Self {
        Self::new(Exit::NotReady, message).owner(Owner::You)
    }
    pub fn failed(message: impl Into<String>) -> Self {
        Self::new(Exit::Failed, message)
    }
    pub fn owner(mut self, owner: Owner) -> Self {
        self.owner = Some(owner);
        self
    }
    pub fn code(mut self, code: impl Into<String>) -> Self {
        self.code = Some(code.into());
        self
    }
    pub fn next(mut self, next: impl Into<String>) -> Self {
        self.next = Some(next.into());
        self
    }
    pub fn detail(mut self, detail: impl Into<String>) -> Self {
        let d = detail.into();
        if !d.trim().is_empty() {
            self.detail = Some(d);
        }
        self
    }

    /// An internal error: write a crash log and point at it.
    pub fn bug(context: &str, err: impl fmt::Display) -> Self {
        let text = format!("bench {}\n{context}\n{err}\n", env!("CARGO_PKG_VERSION"),);
        let path = benchflow_run::config::default_cache_dir()
            .join("logs")
            .join(format!(
                "crash-{}.log",
                chrono::Local::now().format("%Y%m%d-%H%M%S")
            ));
        let saved = std::fs::create_dir_all(path.parent().unwrap_or(&path))
            .and_then(|_| std::fs::write(&path, &text))
            .is_ok();
        let mut e = Self::new(Exit::Failed, format!("{context}: {err}")).owner(Owner::Benchflow);
        e.next = Some(if saved {
            format!(
                "this is a BenchFlow bug; please report it at https://github.com/benchflow-ai/benchflow/issues with {}",
                ui::path(&path)
            )
        } else {
            "this is a BenchFlow bug; please report it at https://github.com/benchflow-ai/benchflow/issues".into()
        });
        e
    }

    pub fn print(&self) {
        if ui::ui().json {
            let v = serde_json::json!({
                "error": self.message,
                "code": self.code,
                "owner": self.owner,
                "next": self.next,
                "detail": self.detail,
                "exit_code": self.exit as u8,
            });
            println!("{v}");
            return;
        }
        let head = match &self.code {
            Some(code) => format!("error[{code}]:"),
            None => "error:".to_string(),
        };
        let width = ui::wrap_width();
        let head = format!("{} ", paint(RED, bold(head)));
        let indent = ui::visible_width(&head);
        anstream::eprintln!("{}", ui::wrap(&head, &self.message, indent, width));
        if let Some(d) = &self.detail {
            for line in d.lines().take(30) {
                anstream::eprintln!("  {}", dim(line));
            }
        }
        if let Some(o) = self.owner
            && (o != Owner::You || self.next.is_none())
        {
            anstream::eprintln!("  {} {}", dim("whose problem:"), o.label());
        }
        if let Some(n) = &self.next {
            anstream::eprintln!("{}", ui::wrap(&format!("  {} ", dim("next:")), n, 8, width));
        }
    }
}

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

pub type Result<T> = std::result::Result<T, CliError>;
