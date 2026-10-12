//! `bench check`: validate tasks without building or running anything.

use std::time::Instant;

use benchflow_task::diag::{Diagnostic, Severity, has_errors};
use benchflow_task::{CheckOptions, CheckReport};

use super::review;
use crate::cli::CheckArgs;
use crate::context;
use crate::error::{Exit, Result};
use crate::ui::{
    self, Verbosity, diag::DiagRenderer, dim, fail_mark, ok_mark, plural, ui, warn_mark,
};
use crate::{note, say};

pub fn run(args: CheckArgs) -> Result<Exit> {
    if args.watch {
        return watch(args);
    }
    once(&args)
}

/// Check, then check again on every change, until Ctrl-C.
fn watch(args: CheckArgs) -> Result<Exit> {
    let project = context::project()?;
    let roots = context::task_roots(&project, &args.paths);
    let mut last = fingerprint(&roots);
    loop {
        if ui().interactive {
            anstream::eprint!("\x1b[2J\x1b[H");
        }
        let _ = once(&args);
        note!(
            "{}",
            dim(format!(
                "watching {} · Ctrl-C to stop",
                roots
                    .iter()
                    .map(|r| ui::path(r))
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        );
        loop {
            std::thread::sleep(std::time::Duration::from_millis(250));
            let now = fingerprint(&roots);
            if now != last {
                // Let a burst of saves settle.
                std::thread::sleep(std::time::Duration::from_millis(150));
                last = fingerprint(&roots);
                break;
            }
        }
    }
}

/// Every file's path, size, and modification time under the roots.
fn fingerprint(
    roots: &[std::path::PathBuf],
) -> Vec<(std::path::PathBuf, u64, std::time::SystemTime)> {
    let mut v = Vec::new();
    for r in roots {
        for e in walkdir::WalkDir::new(r)
            .into_iter()
            .filter_entry(|e| {
                let n = e.file_name().to_string_lossy();
                e.depth() == 0
                    || !matches!(
                        n.as_ref(),
                        ".git" | "jobs" | "target" | "node_modules" | ".cache" | "__pycache__"
                    )
            })
            .filter_map(std::result::Result::ok)
        {
            if let Ok(m) = e.metadata()
                && m.is_file()
            {
                v.push((
                    e.into_path(),
                    m.len(),
                    m.modified().unwrap_or(std::time::UNIX_EPOCH),
                ));
            }
        }
    }
    v.sort();
    v
}

fn once(args: &CheckArgs) -> Result<Exit> {
    let started = Instant::now();
    let project = context::project()?;
    let roots = context::task_roots(&project, &args.paths);
    let paths = context::discover(&roots)?;
    if paths.is_empty() {
        return Err(context::no_tasks(&roots));
    }
    let opts = CheckOptions { deep: true };
    let mut reports = check_paths(&paths, &opts, args.schema);

    if args.fix {
        // A diagnostic names its file relative to its task's folder.
        let mut applied = Vec::new();
        for r in &reports {
            applied.extend(
                ui::diag::apply_fixes(&r.diagnostics, &r.root)
                    .map_err(|e| crate::error::CliError::bug("applying fixes", e))?,
            );
        }
        for a in &applied {
            note!("  {} fixed {}", ok_mark(), a);
        }
        if !applied.is_empty() {
            reports = check_paths(&paths, &opts, args.schema);
        } else {
            note!("  {}", dim("nothing to fix automatically"));
        }
    }

    if ui().json {
        let mut v: Vec<serde_json::Value> = reports.iter().map(report_json).collect();
        let mut exit = exit_for(&reports, args.strict);
        if args.review {
            match review::review_tasks(&reports, args.model.as_deref(), args.save) {
                Ok(reviews) => {
                    for (j, r) in v.iter_mut().zip(&reviews) {
                        j["review"] = review::review_json(r);
                    }
                    if reviews.iter().any(|r| r.is_err()) {
                        exit = Exit::Failed;
                    }
                }
                Err(e) => {
                    for j in &mut v {
                        j["review"] = serde_json::json!({"error": e.message, "next": e.next});
                    }
                    exit = e.exit;
                }
            }
        }
        println!("{}", serde_json::to_string_pretty(&v).unwrap_or_default());
        return Ok(exit);
    }

    let mut renderer = DiagRenderer::new(ui().color);
    let single = reports.len() == 1;
    let mut errors = 0usize;
    let mut warnings = 0usize;
    let mut notes = 0usize;
    let mut failed_tasks = 0usize;
    // Fixable diagnostics by severity: errors, warnings, notes.
    let mut fixable = [0usize; 3];
    for r in &reports {
        let e = r
            .diagnostics
            .iter()
            .filter(|d| d.severity == Severity::Error)
            .count();
        let w = r
            .diagnostics
            .iter()
            .filter(|d| d.severity == Severity::Warning)
            .count();
        errors += e;
        warnings += w;
        notes += r.diagnostics.len() - e - w;
        for d in r.diagnostics.iter().filter(|d| d.fix.is_some()) {
            fixable[d.severity as usize] += 1;
        }
        if e > 0 || (args.strict && w > 0) {
            failed_tasks += 1;
        }
        for d in &r.diagnostics {
            if d.severity == Severity::Note && ui().verbosity < Verbosity::Verbose {
                continue;
            }
            // Files are named relative to the task's folder.
            anstream::eprintln!("{}", renderer.render(d, &r.root));
        }
        let name = r
            .package
            .as_ref()
            .map(|p| p.name.clone())
            .unwrap_or_else(|| {
                r.root
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default()
            });
        let format = r.format.map(|f| f.label()).unwrap_or("not a task");
        if e > 0 {
            say!(
                "  {} {}  {}",
                fail_mark(),
                name,
                dim(format!("{} · {}", plural(e, "error"), format))
            );
        } else if w > 0 {
            say!(
                "  {} {}  {}",
                warn_mark(),
                name,
                dim(format!("{} · {}", plural(w, "warning"), format))
            );
        } else if single || ui().verbosity >= Verbosity::Verbose {
            say!("  {} {}  {}", ok_mark(), name, dim(format));
        }
        if single || ui().verbosity >= Verbosity::Verbose {
            for f in &r.facts {
                say!("      {}", dim(f));
            }
        }
    }
    let ok = reports.len() - failed_tasks;
    let mut line = format!(
        "{} {}",
        if failed_tasks == 0 {
            ok_mark()
        } else {
            fail_mark()
        },
        plural(reports.len(), "task")
    );
    line.push_str(&format!(": {ok} ok"));
    if failed_tasks > 0 {
        line.push_str(&format!(", {failed_tasks} with errors"));
    }
    if warnings > 0 {
        line.push_str(&format!(" · {}", plural(warnings, "warning")));
    }
    if notes > 0 {
        line.push_str(&format!(" · {}", plural(notes, "note")));
        if ui().verbosity < Verbosity::Verbose {
            line.push_str(" (-v shows them)");
        }
    }
    let _ = errors;
    say!("{line}  {}", dim(ui::duration(started.elapsed())));
    if fixable.iter().any(|n| *n > 0) && !args.fix {
        let parts: Vec<String> = fixable
            .iter()
            .zip(["error", "warning", "note"])
            .filter(|(n, _)| **n > 0)
            .map(|(n, word)| plural(*n, word))
            .collect();
        say!(
            "  {}",
            dim(format!(
                "{} can be fixed automatically: bench check --fix",
                parts.join(" and ")
            ))
        );
    }
    let mut exit = exit_for(&reports, args.strict);
    // The model's opinion comes after the checker's results and never
    // changes their exit code; a review that could not run does.
    if args.review {
        let reviews = review::review_tasks(&reports, args.model.as_deref(), args.save)?;
        if !review::print_task_reviews(&reports, &reviews, args.save) && exit == Exit::Ok {
            exit = Exit::Failed;
        }
    }
    Ok(exit)
}

fn exit_for(reports: &[CheckReport], strict: bool) -> Exit {
    let bad = reports.iter().any(|r| {
        has_errors(&r.diagnostics)
            || (strict
                && r.diagnostics
                    .iter()
                    .any(|d| d.severity == Severity::Warning))
    });
    if bad { Exit::Failed } else { Exit::Ok }
}

fn report_json(r: &CheckReport) -> serde_json::Value {
    serde_json::json!({
        "path": r.root,
        "format": r.format,
        "ok": r.ok(),
        "task": r.package.as_ref().map(|p| serde_json::json!({
            "id": p.id, "name": p.name, "digest": p.digest, "title": p.title, "version": p.version,
            "unsupported": p.unsupported,
        })),
        "diagnostics": r.diagnostics,
        "facts": r.facts,
    })
}

/// Check every task, keeping only configuration diagnostics with `--schema`.
fn check_paths(
    paths: &[std::path::PathBuf],
    opts: &CheckOptions,
    schema: bool,
) -> Vec<CheckReport> {
    let mut reports = context::check_all(paths, opts);
    if schema {
        reports
            .iter_mut()
            .for_each(benchflow_task::keep_schema_only);
    } else {
        reports.iter_mut().for_each(warn_credentials);
        reports.iter_mut().for_each(check_judge);
    }
    reports
}

/// Check the settings of the rubric's model judge: its model and the
/// references it reads.
fn check_judge(r: &mut CheckReport) {
    let Some(task) = r.package.as_ref() else {
        return;
    };
    match benchflow_run::judge::plan(task, None) {
        Ok(Some(plan)) => {
            let with_reference = plan
                .items
                .iter()
                .filter(|i| i.reference.is_some() || plan.reference.is_some())
                .count();
            r.facts.push(format!(
                "model-judged criteria: {} (judge {}; {} with a reference)",
                plan.items.len(),
                plan.model,
                with_reference
            ));
        }
        Ok(None) => {}
        Err(e) => r.diagnostics.push(
            Diagnostic::error(
                "P004",
                format!("the rubric's judge settings cannot be used: {e}"),
            )
            .in_file(std::path::Path::new("verifier/rubric.json"))
            .help("`bench explain P004` says how to name the judge model and its references"),
        ),
    }
}

/// Warn about credential files in the task's build folder, by path and
/// kind, never content.
fn warn_credentials(r: &mut CheckReport) {
    let Some(task) = r.package.as_ref() else {
        return;
    };
    for f in benchflow_run::credentials::scan_task_build(task) {
        r.diagnostics.push(
            Diagnostic::warning(
                "P003",
                format!(
                    "`{}` is a credential file ({}) in the build folder, which the image build receives",
                    f.path, f.kind
                ),
            )
            .help(f.fix()),
        );
    }
}
