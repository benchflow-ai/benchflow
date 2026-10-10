//! Rendering diagnostics the way rustc does: the file, the line, a caret
//! under the exact key or value, then notes, help, and the suggested edit.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use annotate_snippets::renderer::DecorStyle;
use annotate_snippets::{AnnotationKind, Element, Group, Level, Patch, Renderer, Snippet};
use benchflow_task::diag::{Diagnostic, Severity};

/// Renders diagnostics, caching file contents.
pub struct DiagRenderer {
    files: HashMap<PathBuf, Option<String>>,
    renderer: Renderer,
}

impl DiagRenderer {
    pub fn new(color: bool) -> Self {
        let renderer = if color {
            Renderer::styled()
        } else {
            Renderer::plain()
        };
        Self {
            files: HashMap::new(),
            renderer: renderer.decor_style(DecorStyle::Unicode),
        }
    }

    fn source(&mut self, path: &Path) -> Option<String> {
        self.files
            .entry(path.to_path_buf())
            .or_insert_with(|| std::fs::read_to_string(path).ok())
            .clone()
    }

    /// Render one diagnostic. `base` resolves relative file paths.
    pub fn render(&mut self, d: &Diagnostic, base: &Path) -> String {
        let level = match d.severity {
            Severity::Error => Level::ERROR,
            Severity::Warning => Level::WARNING,
            Severity::Note => Level::NOTE,
        };
        // A long message keeps its first clause as the headline and moves the
        // rest to a note, so the headline fits on one line.
        let (title_text, mut rest) = split_title(&d.message);
        // The caret's label already asks "did you mean"; the note need not.
        if let (Some(label), Some(r)) = (&d.label, &rest)
            && label.starts_with("did you mean")
            && let Some((dym, after)) = r.split_once("? ")
            && dym.starts_with("did you mean")
        {
            rest = Some(after.to_string());
        }
        let title = level.primary_title(title_text.as_str()).id(d.code);
        let mut elements: Vec<Element> = Vec::new();
        // Owned strings must outlive the report; keep them here.
        let mut owned: Vec<String> = Vec::new();
        let mut source_for_patch: Option<(String, String, (usize, usize))> = None;
        if let Some(loc) = &d.location {
            let full = if loc.file.is_absolute() {
                loc.file.clone()
            } else {
                base.join(&loc.file)
            };
            let shown = crate::ui::path(&full);
            match (self.source(&full), loc.span) {
                (Some(text), Some((start, end))) if end <= text.len() && start <= end => {
                    let label = d.label.clone().unwrap_or_default();
                    owned.push(text.clone());
                    owned.push(shown.clone());
                    owned.push(label.clone());
                    let ann = if label.is_empty() {
                        AnnotationKind::Primary
                            .span(start..end.max(start + 1).min(text.len().max(start + 1)))
                    } else {
                        AnnotationKind::Primary
                            .span(start..end.max(start + 1).min(text.len().max(start + 1)))
                            .label(label)
                    };
                    elements.push(
                        Snippet::source(text.clone())
                            .path(shown.clone())
                            .fold(true)
                            .annotation(ann)
                            .into(),
                    );
                    if let Some(fix) = &d.fix {
                        source_for_patch = Some((text, shown, fix.span));
                    }
                }
                _ => {
                    let at = match loc.line {
                        Some(l) => format!("{shown}:{l}"),
                        None => shown,
                    };
                    elements.push(Level::NOTE.no_name().message(format!("in {at}")).into());
                }
            }
        }
        let width = crate::ui::wrap_width().saturating_sub(NOTE_PREFIX);
        for n in rest.iter().chain(&d.notes) {
            let n = crate::ui::wrap("", n, 0, width.max(40));
            elements.push(Level::NOTE.message(n).into());
        }
        let group = if elements.is_empty() {
            Group::with_title(title)
        } else {
            Group::with_title(title).elements(elements)
        };
        let mut report: Vec<Group> = vec![group];
        match (&d.fix, source_for_patch) {
            (Some(fix), Some((text, shown, span))) => {
                let title = d.help.clone().unwrap_or_else(|| fix.description.clone());
                report.push(
                    Level::HELP.secondary_title(title).element(
                        Snippet::source(text)
                            .path(shown)
                            .patch(Patch::new(span.0..span.1, fix.replacement.clone())),
                    ),
                );
            }
            _ => {
                if let Some(h) = &d.help {
                    report.push(Group::with_title(Level::HELP.secondary_title(h.clone())));
                }
            }
        }
        let _ = owned;
        self.renderer.render(&report).to_string()
    }
}

/// Columns before a note's text: the gutter, `╰ note: `, and a margin.
const NOTE_PREFIX: usize = 12;

/// A headline longer than this is split at its first `; `.
const TITLE_MAX: usize = 72;

/// Split a message too long for one headline at its first `; `: the first
/// clause is the headline, the rest a note.
fn split_title(message: &str) -> (String, Option<String>) {
    if crate::ui::visible_width(message) <= TITLE_MAX {
        return (message.to_string(), None);
    }
    match message.split_once("; ") {
        Some((head, rest)) if !head.is_empty() && !rest.trim().is_empty() => {
            (head.to_string(), Some(rest.trim().to_string()))
        }
        _ => (message.to_string(), None),
    }
}

/// Apply every machine-applicable fix, file by file, last span first so
/// offsets stay valid. Returns the fixes applied, as descriptions.
pub fn apply_fixes(diags: &[Diagnostic], base: &Path) -> std::io::Result<Vec<String>> {
    let mut by_file: HashMap<PathBuf, Vec<&benchflow_task::diag::Fix>> = HashMap::new();
    for d in diags {
        if let Some(f) = &d.fix {
            let p = if f.file.is_absolute() {
                f.file.clone()
            } else {
                base.join(&f.file)
            };
            by_file.entry(p).or_default().push(f);
        }
    }
    let mut applied = Vec::new();
    for (file, mut fixes) in by_file {
        let mut text = std::fs::read_to_string(&file)?;
        fixes.sort_by_key(|f| std::cmp::Reverse(f.span.0));
        let mut last_start = usize::MAX;
        for f in fixes {
            // Skip overlapping edits; the next `bench check --fix` gets them.
            if f.span.1 > last_start || f.span.1 > text.len() || f.span.0 > f.span.1 {
                continue;
            }
            text.replace_range(f.span.0..f.span.1, &f.replacement);
            last_start = f.span.0;
            applied.push(format!("{}: {}", crate::ui::path(&file), f.description));
        }
        std::fs::write(&file, text)?;
    }
    applied.sort();
    Ok(applied)
}

#[cfg(test)]
mod tests {
    use super::*;
    use benchflow_task::diag::Fix;

    #[test]
    fn renders_a_caret_under_the_key() {
        let d = tempfile::tempdir().unwrap();
        let file = d.path().join("task.md");
        std::fs::write(
            &file,
            "Do it.\n\n```toml task\n[agent]\ntimout = \"10m\"\n```\n",
        )
        .unwrap();
        let text = std::fs::read_to_string(&file).unwrap();
        let start = text.find("timout").unwrap();
        let diag = Diagnostic::error("T201", "unknown key `timout` in [agent]")
            .at(&file, Some(5), Some((start, start + 6)))
            .label("did you mean `timeout`?")
            .note_line("[agent] holds: timeout, on_timeout, budget")
            .fix(Fix {
                file: file.clone(),
                span: (start, start + 6),
                replacement: "timeout".into(),
                description: "rename `timout` to `timeout`".into(),
            });
        let out = DiagRenderer::new(false).render(&diag, d.path());
        assert!(
            out.contains("error[T201]: unknown key `timout` in [agent]"),
            "{out}"
        );
        assert!(out.contains("did you mean `timeout`?"), "{out}");
        assert!(out.contains("timeout = \"10m\""), "{out}");
        let applied = apply_fixes(&[diag], d.path()).unwrap();
        assert_eq!(applied.len(), 1);
        assert!(
            std::fs::read_to_string(&file)
                .unwrap()
                .contains("\ntimeout = \"10m\"")
        );
    }

    #[test]
    fn a_long_headline_keeps_its_first_clause() {
        let long = format!(
            "unknown key [agent] timout; did you mean timeout? [agent] holds {}",
            "x, ".repeat(30)
        );
        let (head, rest) = split_title(&long);
        assert_eq!(head, "unknown key [agent] timout");
        assert!(rest.unwrap().starts_with("did you mean timeout?"));
        let short = "unknown key [agent] timout; did you mean timeout?";
        assert_eq!(split_title(short), (short.to_string(), None));
    }
}
