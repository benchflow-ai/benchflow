//! Terminal output: one palette, a few marks, and helpers every command uses.
//!
//! Conventions:
//! - Results go to stdout; progress and diagnostics go to stderr.
//! - Finished things start with a mark: `✓` (green), `✗` (red), `!` (yellow).
//! - Secondary text is dim. Nothing blinks; nothing is bold for decoration.
//! - `NO_COLOR` turns color off; `--color always` forces it.

pub mod diag;
pub mod live;
pub mod table;

use std::io::IsTerminal;
use std::sync::OnceLock;
use std::time::Duration;

use anstyle::{AnsiColor, Style};

/// How loud to be.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Verbosity {
    Quiet,
    Normal,
    Verbose,
    Debug,
}

/// Output settings for the whole process, fixed at startup.
#[derive(Debug, Clone, Copy)]
pub struct Ui {
    pub json: bool,
    pub color: bool,
    pub verbosity: Verbosity,
    /// stderr is a terminal (live displays allowed).
    pub interactive: bool,
}

static UI: OnceLock<Ui> = OnceLock::new();

pub fn init(ui: Ui) {
    let _ = UI.set(ui);
    if !ui.color {
        anstream::ColorChoice::Never.write_global();
    } else {
        anstream::ColorChoice::Always.write_global();
    }
}

pub fn ui() -> Ui {
    *UI.get().unwrap_or(&Ui {
        json: false,
        color: false,
        verbosity: Verbosity::Normal,
        interactive: false,
    })
}

/// Decide color from `--color`, `NO_COLOR`, `CLICOLOR_FORCE`, and the tty.
pub fn color_enabled(choice: &str) -> bool {
    match choice {
        "always" => true,
        "never" => false,
        _ => {
            if std::env::var_os("NO_COLOR").is_some_and(|v| !v.is_empty()) {
                return false;
            }
            if std::env::var_os("CLICOLOR_FORCE").is_some_and(|v| v != "0") {
                return true;
            }
            std::io::stderr().is_terminal() && std::env::var("TERM").map_or(true, |t| t != "dumb")
        }
    }
}

pub const GREEN: Style = AnsiColor::Green.on_default();
pub const RED: Style = AnsiColor::Red.on_default();
pub const YELLOW: Style = AnsiColor::Yellow.on_default();
pub const CYAN: Style = AnsiColor::Cyan.on_default();
pub const DIM: Style = Style::new().dimmed();
pub const BOLD: Style = Style::new().bold();

/// Wrap `text` in a style (a no-op without color: anstream strips it).
pub fn paint(style: Style, text: impl std::fmt::Display) -> String {
    format!("{style}{text}{style:#}")
}

pub fn ok_mark() -> String {
    paint(GREEN, "✓")
}
pub fn fail_mark() -> String {
    paint(RED, "✗")
}
pub fn warn_mark() -> String {
    paint(YELLOW, "!")
}
pub fn dim(text: impl std::fmt::Display) -> String {
    paint(DIM, text)
}
pub fn bold(text: impl std::fmt::Display) -> String {
    paint(BOLD, text)
}

/// Print a line to stderr (progress, notes), unless quiet or JSON.
#[macro_export]
macro_rules! note {
    ($($arg:tt)*) => {{
        let u = $crate::ui::ui();
        if !u.json && u.verbosity > $crate::ui::Verbosity::Quiet {
            anstream::eprintln!($($arg)*);
        }
    }};
}

/// Print a line to stdout (results), unless JSON.
#[macro_export]
macro_rules! say {
    ($($arg:tt)*) => {{
        if !$crate::ui::ui().json {
            anstream::println!($($arg)*);
        }
    }};
}

/// `31ms`, `4.2s`, `42s`, `3m10s`, `1h02m`.
pub fn duration(d: Duration) -> String {
    let s = d.as_secs();
    if s == 0 && d.as_millis() < 1000 {
        return format!("{}ms", d.as_millis());
    }
    if s < 60 {
        // 9.96 s reads 10s, not 10.0s.
        if d.as_secs_f64() < 9.95 && !d.as_millis().is_multiple_of(1000) {
            return format!("{:.1}s", d.as_secs_f64());
        }
        format!("{}s", s.max(d.as_secs_f64().round() as u64))
    } else if s < 3600 {
        format!("{}m{:02}s", s / 60, s % 60)
    } else {
        format!("{}h{:02}m", s / 3600, (s % 3600) / 60)
    }
}

pub fn secs(s: f64) -> String {
    duration(benchflow_task::values::duration_from_secs(s))
}

/// The terminal's width, or 100.
pub fn width() -> usize {
    terminal_size::terminal_size()
        .map(|(w, _)| w.0 as usize)
        .unwrap_or(100)
        .clamp(40, 200)
}

/// Display width of text without ANSI escapes.
pub fn visible_width(s: &str) -> usize {
    use unicode_width::UnicodeWidthStr;
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            for c2 in chars.by_ref() {
                if c2.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out.width()
}

/// The width to wrap an error at: the terminal's, or no wrapping at all
/// when the output is not a terminal (a log, a pipe, a test), so nothing
/// that reads it finds a sentence broken in two.
pub fn wrap_width() -> usize {
    terminal_size::terminal_size()
        .map(|(w, _)| (w.0 as usize).clamp(40, 200))
        .unwrap_or(usize::MAX)
}

/// Cut plain text to `max` display columns, with an ellipsis.
pub fn truncate(s: &str, max: usize) -> String {
    use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};
    if s.width() <= max {
        return s.to_string();
    }
    let mut w = 0;
    let mut out = String::new();
    for c in s.chars() {
        let cw = c.width().unwrap_or(0);
        if w + cw > max.saturating_sub(1) {
            out.push('…');
            return out;
        }
        w += cw;
        out.push(c);
    }
    out
}

/// A path for people: relative to the current folder when shorter.
pub fn path(p: &std::path::Path) -> String {
    if let Ok(cwd) = std::env::current_dir() {
        // Also through symlinks: macOS's /tmp is /private/tmp.
        let real = |q: &std::path::Path| std::fs::canonicalize(q).ok();
        let rel = p
            .strip_prefix(&cwd)
            .ok()
            .map(std::path::Path::to_path_buf)
            .or_else(|| {
                let (rp, rc) = (real(p)?, real(&cwd)?);
                rp.strip_prefix(&rc).ok().map(std::path::Path::to_path_buf)
            });
        if let Some(rel) = rel {
            let s = rel.display().to_string();
            return if s.is_empty() { ".".into() } else { s };
        }
    }
    if let Some(home) = dirs::home_dir()
        && let Ok(rel) = p.strip_prefix(&home)
    {
        return format!("~/{}", rel.display());
    }
    p.display().to_string()
}

/// Join `parts` with ` · `, starting a new line indented by `indent` spaces
/// wherever the next part would pass `width` columns.
pub fn join_fitting(parts: &[String], indent: usize, width: usize) -> String {
    let mut out = String::new();
    let mut line = 0;
    for (i, part) in parts.iter().enumerate() {
        let w = visible_width(part);
        if i == 0 {
            out.push_str(part);
            line = w;
        } else if line + 3 + w > width {
            out.push('\n');
            out.push_str(&" ".repeat(indent));
            out.push_str(part);
            line = indent + w;
        } else {
            out.push_str(" · ");
            out.push_str(part);
            line += 3 + w;
        }
    }
    out
}

/// `first`, `sep`, and `second` on one line when they fit in `width`;
/// otherwise `second` starts a new line indented by `indent` (keeping a
/// word separator such as ` on `, dropping ` · `).
pub fn fit_or_break(first: &str, sep: &str, second: &str, indent: usize, width: usize) -> String {
    let one = format!("{first}{sep}{second}");
    if visible_width(&one) <= width {
        return one;
    }
    let word = sep.trim();
    let lead = if word.is_empty() || word == "·" {
        String::new()
    } else {
        format!("{word} ")
    };
    format!("{first}\n{}{lead}{second}", " ".repeat(indent))
}

/// Word-wrap plain `text` to `width` columns: the first line after `first`
/// (already printed, its width counted), later lines indented by `indent`.
pub fn wrap(first: &str, text: &str, indent: usize, width: usize) -> String {
    let mut out = first.to_string();
    let mut line = visible_width(first);
    for (i, para) in text.split('\n').enumerate() {
        if i > 0 {
            out.push('\n');
            out.push_str(&" ".repeat(indent));
            line = indent;
        }
        // Nothing yet on this line but its prefix.
        let mut fresh = true;
        for word in para.split(' ') {
            let w = visible_width(word);
            if !fresh && line + 1 + w > width {
                out.push('\n');
                out.push_str(&" ".repeat(indent));
                line = indent;
                fresh = true;
            }
            if !fresh {
                out.push(' ');
                line += 1;
            }
            out.push_str(word);
            line += w;
            fresh = false;
        }
    }
    out
}

/// How long ago `t` was: `just now`, `12 min ago`, `3 h ago`, `5 days ago`.
pub fn ago(t: std::time::SystemTime) -> String {
    let secs = t.elapsed().map(|d| d.as_secs()).unwrap_or(0);
    match secs {
        0..=59 => "just now".into(),
        60..=3599 => format!("{} min ago", secs / 60),
        3600..=86_399 => format!("{} h ago", secs / 3600),
        _ => format!("{} days ago", secs / 86_400),
    }
}

/// A size in decimal units: `512 B`, `3.4 KB`, `12 MB`, `1.5 GB`.
pub fn bytes(n: u64) -> String {
    let n = n as f64;
    let (v, unit) = if n >= 1e9 {
        (n / 1e9, "GB")
    } else if n >= 1e6 {
        (n / 1e6, "MB")
    } else if n >= 1e3 {
        (n / 1e3, "KB")
    } else {
        return format!("{n:.0} B");
    };
    if v >= 10.0 {
        format!("{v:.0} {unit}")
    } else {
        format!("{v:.1} {unit}")
    }
}

/// Dollars: cents when there are some, else enough digits to not read $0.00.
pub fn usd(x: f64) -> String {
    if x >= 0.01 || x == 0.0 {
        format!("${x:.2}")
    } else if x >= 0.0001 {
        format!("${x:.4}")
    } else {
        "<$0.0001".into()
    }
}

/// The model cost of some trials, from each one's recorded cost (what the
/// provider billed, when the forwarder saw it): the sum, with how many
/// recorded one when some did not, or "not recorded" when none did, so a
/// missing cost never reads as $0.
pub fn cost(costs: &[Option<f64>]) -> String {
    let recorded: Vec<f64> = costs.iter().flatten().copied().collect();
    if recorded.is_empty() {
        return "cost not recorded".into();
    }
    let sum = usd(recorded.iter().sum());
    if recorded.len() == costs.len() {
        sum
    } else {
        format!(
            "{sum} ({} of {} recorded a cost)",
            recorded.len(),
            plural(costs.len(), "trial")
        )
    }
}

/// `1 task`, `3 tasks`.
pub fn plural(n: usize, word: &str) -> String {
    if n == 1 {
        format!("{n} {word}")
    } else if word.ends_with('s')
        || word.ends_with('x')
        || word.ends_with("sh")
        || word.ends_with("ch")
    {
        format!("{n} {word}es")
    } else if let Some(stem) = word.strip_suffix('y') {
        format!("{n} {stem}ies")
    } else {
        format!("{n} {word}s")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_durations() {
        assert_eq!(duration(Duration::from_secs(42)), "42s");
        assert_eq!(duration(Duration::from_secs(190)), "3m10s");
        assert_eq!(duration(Duration::from_secs(3720)), "1h02m");
        assert_eq!(duration(Duration::from_millis(2500)), "2.5s");
        assert_eq!(duration(Duration::from_millis(31)), "31ms");
        assert_eq!(duration(Duration::from_millis(9960)), "10s");
        assert_eq!(duration(Duration::from_millis(9940)), "9.9s");
    }

    #[test]
    fn long_lines_break_at_parts_and_words() {
        let parts = [
            "Running builtin (vendor/a-long-model-name)".to_string(),
            "2 trials".into(),
            "daytona · 2 at a time".into(),
        ];
        assert_eq!(
            join_fitting(&parts, 2, 60),
            "Running builtin (vendor/a-long-model-name) · 2 trials\n  daytona · 2 at a time"
        );
        assert_eq!(
            join_fitting(&parts[1..], 2, 60),
            "2 trials · daytona · 2 at a time"
        );
        assert_eq!(
            wrap(
                "  next: ",
                "run it with docker, which starts each service",
                8,
                30
            ),
            "  next: run it with docker,\n        which starts each\n        service"
        );
        assert_eq!(
            fit_or_break("Testing 1 task", " on ", "daytona", 2, 40),
            "Testing 1 task on daytona"
        );
        assert_eq!(
            fit_or_break("Testing 1 task", " on ", "daytona", 2, 20),
            "Testing 1 task\n  on daytona"
        );
        assert_eq!(
            fit_or_break("Running x", " · ", "daytona", 2, 10),
            "Running x\n  daytona"
        );
        assert_eq!(usd(0.000634), "$0.0006");
        assert_eq!(usd(1.5), "$1.50");
    }

    #[test]
    fn a_cost_not_recorded_is_not_zero() {
        assert_eq!(cost(&[Some(0.1), Some(0.04)]), "$0.14");
        assert_eq!(cost(&[None, None]), "cost not recorded");
        assert_eq!(cost(&[]), "cost not recorded");
        assert_eq!(
            cost(&[Some(0.1), None, Some(0.04)]),
            "$0.14 (2 of 3 trials recorded a cost)"
        );
        assert_eq!(cost(&[Some(0.0)]), "$0.00");
    }

    #[test]
    fn plurals() {
        assert_eq!(plural(1, "task"), "1 task");
        assert_eq!(plural(2, "task"), "2 tasks");
        // `bench run`'s cleanup line said "removed 2 sandboxs".
        assert_eq!(plural(2, "sandbox"), "2 sandboxes");
        assert_eq!(plural(2, "probe"), "2 probes");
        assert_eq!(plural(3, "class"), "3 classes");
        assert_eq!(plural(2, "policy"), "2 policies");
    }

    #[test]
    fn widths_ignore_escapes() {
        assert_eq!(visible_width("\u{1b}[32m✓\u{1b}[0m ok"), 4);
        assert_eq!(truncate("abcdef", 4), "abc…");
        assert_eq!(truncate("abc", 4), "abc");
        assert_eq!(truncate("abcd", 4), "abcd");
    }
}
