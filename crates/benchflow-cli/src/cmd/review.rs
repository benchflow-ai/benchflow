//! `bench check --review` and `bench results --analyze`: connecting to the
//! model, running the reviews, and showing what the model said as its
//! opinion, after what BenchFlow itself checked or recorded.

use std::future::Future;
use std::path::{Path, PathBuf};

use benchflow_agent::credentials::{HostEnv, MIN_SECRET_LEN};
use benchflow_agent::native::ClientOptions;
use benchflow_run::Owner;
use benchflow_run::review::analyze::{self, JobAnalysis};
use benchflow_run::review::task::{self, CRITERIA, TaskReview};
use benchflow_run::review::{DEFAULT_MODEL, ReviewError, ReviewErrorKind, Reviewer, Spend};
use benchflow_task::CheckReport;
use benchflow_task::diag::Severity;

use crate::error::{CliError, Result};
use crate::ui::{self, bold, diag::DiagRenderer, dim, fail_mark, ok_mark, plural, warn_mark};
use crate::{note, say};

/// Run `fut` to completion from a synchronous command, on the command's
/// runtime.
fn block_on<F: Future>(fut: F) -> F::Output {
    tokio::task::block_in_place(|| tokio::runtime::Handle::current().block_on(fut))
}

/// Connect to `model` (or the default) with this machine's key.
fn connect(model: Option<&str>) -> Result<Reviewer> {
    let model = model.unwrap_or(DEFAULT_MODEL);
    let host = HostEnv::from_process();
    block_on(Reviewer::connect(
        model,
        &host,
        machine_secrets(),
        ClientOptions::default(),
    ))
    .map_err(cli_error)
}

fn cli_error(e: ReviewError) -> CliError {
    match e.kind {
        ReviewErrorKind::NoKey => CliError::not_ready(e.message).next(e.next),
        ReviewErrorKind::Model => CliError::usage(e.message).next(e.next),
        ReviewErrorKind::Provider => CliError::failed(e.message).owner(Owner::Model).next(e.next),
    }
}

/// Credential values on this machine that must never reach the model: every
/// variable whose name says it holds one, and the values in BenchFlow's
/// credential files.
fn machine_secrets() -> Vec<String> {
    let secret_name = |name: &str| {
        let n = name.to_ascii_uppercase();
        [
            "KEY",
            "TOKEN",
            "SECRET",
            "PASSWORD",
            "PASSWD",
            "CREDENTIAL",
            "AUTH",
        ]
        .iter()
        .any(|w| n.contains(w))
    };
    let mut out: Vec<String> = std::env::vars()
        .filter(|(k, v)| secret_name(k) && v.trim().len() >= MIN_SECRET_LEN)
        .map(|(_, v)| v.trim().to_string())
        .collect();
    let dir = dirs::home_dir().map(|h| h.join(".config/benchflow"));
    for entry in dir
        .and_then(|d| std::fs::read_dir(d).ok())
        .into_iter()
        .flatten()
        .filter_map(std::result::Result::ok)
    {
        if !entry.file_name().to_string_lossy().ends_with(".env") {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(entry.path()) else {
            continue;
        };
        for line in text.lines() {
            let line = line.trim().trim_start_matches("export ").trim();
            if let Some((k, v)) = line.split_once('=') {
                let v = v.trim().trim_matches('"').trim_matches('\'').trim();
                if secret_name(k) && v.len() >= MIN_SECRET_LEN {
                    out.push(v.to_string());
                }
            }
        }
    }
    out
}

// ------------------------------------------------------------ check --

/// Each report's review, in report order.
pub fn review_tasks(
    reports: &[CheckReport],
    model: Option<&str>,
    save: bool,
) -> Result<Vec<std::result::Result<TaskReview, ReviewError>>> {
    let reviewer = connect(model)?;
    let tasks: Vec<&CheckReport> = reports.iter().filter(|r| r.format.is_some()).collect();
    note!(
        "{}",
        dim(format!(
            "asking {} to review {} (key: {}) …",
            reviewer.model,
            plural(tasks.len(), "task"),
            reviewer.key_label()
        ))
    );
    let reviews = block_on(task::review_all(&reviewer, reports));
    if save {
        for r in reviews.iter().flatten() {
            let path = r.task.join(task::FILE_NAME);
            if let Err(e) = write_json(&path, r) {
                note!("  {} could not save {}: {e}", warn_mark(), ui::path(&path));
            }
        }
    }
    Ok(reviews)
}

fn write_json(path: &Path, value: &impl serde::Serialize) -> std::io::Result<()> {
    let text = serde_json::to_string_pretty(value).map_err(std::io::Error::other)?;
    std::fs::write(path, text + "\n")
}

/// The review in `--json` output.
pub fn review_json(r: &std::result::Result<TaskReview, ReviewError>) -> serde_json::Value {
    match r {
        Ok(review) => serde_json::to_value(review).unwrap_or_default(),
        Err(e) => serde_json::json!({"error": e.message, "next": e.next}),
    }
}

/// Print the reviews after the checker's output. Returns whether every
/// review finished.
pub fn print_task_reviews(
    reports: &[CheckReport],
    reviews: &[std::result::Result<TaskReview, ReviewError>],
    saved: bool,
) -> bool {
    let mut renderer = DiagRenderer::new(ui::ui().color);
    let base = std::env::current_dir().unwrap_or_default();
    let mut all_ok = true;
    let mut spend = Spend::default();
    let model = reviews
        .iter()
        .flatten()
        .map(|r| r.model.clone())
        .next()
        .unwrap_or_else(|| DEFAULT_MODEL.into());
    say!(
        "\n{} {}",
        bold(format!("Model review by {model}")),
        dim("· the model's opinion, not checked facts; the checker's results above stand")
    );
    for (report, review) in reports.iter().zip(reviews) {
        let name = report
            .package
            .as_ref()
            .map(|p| p.name.clone())
            .unwrap_or_else(|| ui::path(&report.root));
        let review = match review {
            Ok(r) => r,
            Err(e) => {
                all_ok = false;
                say!("  {} {name}: the review failed: {}", fail_mark(), e.message);
                say!("      {} {}", dim("next:"), e.next);
                continue;
            }
        };
        spend.add(&review.spend);
        for f in &review.findings {
            anstream::eprintln!(
                "{}",
                renderer.render(&f.diagnostic(&review.task, &review.model), &base)
            );
        }
        let failed: Vec<&str> = review
            .verdicts
            .iter()
            .filter(|v| v.verdict == "fail")
            .map(|v| v.criterion.as_str())
            .collect();
        let passed = review
            .verdicts
            .iter()
            .filter(|v| v.verdict == "pass")
            .count();
        let unclear = review.verdicts.len() - failed.len() - passed;
        let count = |s: Severity| review.findings.iter().filter(|f| f.severity == s).count();
        let mut parts: Vec<String> = Vec::new();
        for (s, word) in [
            (Severity::Error, "error"),
            (Severity::Warning, "warning"),
            (Severity::Note, "note"),
        ] {
            if count(s) > 0 {
                parts.push(plural(count(s), word));
            }
        }
        let mark = if count(Severity::Error) > 0 {
            fail_mark()
        } else if review.findings.is_empty() {
            ok_mark()
        } else {
            warn_mark()
        };
        say!(
            "  {mark} {name}  {}",
            dim(if parts.is_empty() {
                "the model found nothing to report".to_string()
            } else {
                parts.join(", ")
            })
        );
        let mut line = format!("{passed} of {} criteria pass", CRITERIA.len());
        if !failed.is_empty() {
            line.push_str(&format!(", fail: {}", failed.join(", ")));
        }
        if unclear > 0 {
            line.push_str(&format!(", unclear: {unclear}"));
        }
        say!("      {}", dim(format!("in the model's view: {line}")));
        for v in review.verdicts.iter().filter(|v| v.verdict == "fail") {
            if !v.reason.is_empty() && ui::ui().verbosity >= ui::Verbosity::Verbose {
                say!("      {} {}: {}", dim("·"), v.criterion, v.reason);
            }
        }
        if !review.files_not_sent.is_empty() {
            say!(
                "      {}",
                dim(format!(
                    "not sent: {}",
                    review
                        .files_not_sent
                        .iter()
                        .map(|w| w.file.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ))
            );
        }
        if saved {
            say!(
                "      {}",
                dim(format!(
                    "saved {}",
                    ui::path(&review.task.join(task::FILE_NAME))
                ))
            );
        }
    }
    say!(
        "  {}",
        dim(format!(
            "{} · findings never change the exit code · `bench explain M101` describes each criterion",
            spend.describe()
        ))
    );
    all_ok
}

// ---------------------------------------------------------- results --

/// Analyze the failed trials of `jobs`, at most `max_trials` in all.
pub fn analyze_jobs(
    jobs: &[PathBuf],
    model: Option<&str>,
    max_trials: usize,
    save: bool,
) -> Result<Vec<JobAnalysis>> {
    let reviewer = connect(model)?;
    let mut left = max_trials;
    let mut out = Vec::new();
    note!(
        "{}",
        dim(format!(
            "asking {} why trials failed (key: {})",
            reviewer.model,
            reviewer.key_label()
        ))
    );
    for job in jobs {
        let failed = analyze::failed_trials(job).len();
        note!(
            "{}",
            dim(format!(
                "  {}: {} of {} …",
                ui::path(job),
                failed.min(left),
                plural(failed, "failed trial")
            ))
        );
        let a = block_on(analyze::analyze_job(&reviewer, job, left)).map_err(cli_error)?;
        left = left.saturating_sub(a.trials.len());
        if save && !a.trials.is_empty() {
            let path = job.join(analyze::FILE_NAME);
            if let Err(e) = write_json(&path, &a) {
                note!(
                    "  {} could not save {}: {e}; pass --no-save for a job folder you cannot write",
                    warn_mark(),
                    ui::path(&path)
                );
            }
        }
        out.push(a);
    }
    Ok(out)
}

/// `text` on one line.
fn squash(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// For several jobs: every analyzed trial by cause, across the jobs.
pub fn print_overall(analyses: &[JobAnalysis]) {
    let trials: Vec<_> = analyses
        .iter()
        .flat_map(|a| a.trials.iter())
        .cloned()
        .collect();
    let groups = analyze::group(&trials);
    if analyses.len() < 2 || groups.is_empty() {
        return;
    }
    let mut spend = Spend::default();
    for a in analyses {
        spend.add(&a.spend);
    }
    say!(
        "\n{} {}",
        bold(format!(
            "Across {}, by cause",
            plural(analyses.len(), "job")
        )),
        dim("· the model's opinion")
    );
    let width = groups.iter().map(|g| g.label.len()).max().unwrap_or(0);
    for g in &groups {
        say!(
            "    {:<width$}  {:>3}  {}",
            g.label,
            g.trials.len(),
            dim(ui::truncate(&g.trials.join(", "), 90))
        );
    }
    let failed: usize = analyses.iter().map(|a| a.failed).sum();
    say!(
        "  {}",
        dim(format!(
            "analyzed {} of {} · {}",
            trials.len(),
            plural(failed, "failed trial"),
            spend.describe()
        ))
    );
}

pub fn print_analysis(a: &JobAnalysis, saved: bool) {
    say!(
        "\n{} {}",
        bold(format!("Why trials failed in {}", ui::path(&a.job))),
        dim(format!(
            "· {}'s opinion; what BenchFlow recorded is shown first",
            a.model
        ))
    );
    if a.failed == 0 {
        say!("  {} every trial passed; nothing to analyze", ok_mark());
        return;
    }
    for t in &a.trials {
        let status = match t.recorded.reward {
            Some(r) => format!("{:?}, reward {r}", t.recorded.status).to_lowercase(),
            None => format!("{:?}", t.recorded.status).to_lowercase(),
        };
        say!("  {} {}  {}", fail_mark(), t.trial, dim(status));
        if let (Some(code), Some(msg)) = (&t.recorded.code, &t.recorded.message) {
            say!(
                "      {} [{code}] {} {}",
                dim("recorded:"),
                ui::truncate(&squash(msg), 100),
                dim(format!(
                    "· whose problem: {}",
                    t.recorded.owner.as_deref().unwrap_or("not recorded")
                ))
            );
        }
        if let Some(e) = &t.error {
            say!("      {} {}", dim("model:"), e);
            continue;
        }
        let confidence = if t.confidence == "unstated" {
            String::new()
        } else {
            dim(format!("({} confidence)", t.confidence))
        };
        say!(
            "      {} {} {confidence}",
            dim("model:"),
            bold(t.part.label())
        );
        say!("        {}", t.explanation);
        for q in &t.evidence {
            let check = if q.found {
                dim("in the evidence sent")
            } else {
                format!(
                    "{} {}",
                    warn_mark(),
                    dim("not in the evidence sent: paraphrased or made up")
                )
            };
            say!(
                "        {} {} {}",
                dim(format!(
                    "{}:",
                    if q.file.is_empty() { "quote" } else { &q.file }
                )),
                ui::truncate(&format!("\"{}\"", squash(&q.quote)), 100),
                check
            );
        }
    }
    if !a.by_cause.is_empty() {
        say!("  {}", bold("By cause, in the model's opinion:"));
        let width = a.by_cause.iter().map(|g| g.label.len()).max().unwrap_or(0);
        for g in &a.by_cause {
            say!(
                "    {:<width$}  {:>3}  {}",
                g.label,
                g.trials.len(),
                dim(ui::truncate(&g.trials.join(", "), 90))
            );
        }
    }
    let mut line = format!(
        "analyzed {} of {}",
        a.trials.len(),
        plural(a.failed, "failed trial")
    );
    if !a.not_analyzed.is_empty() {
        line.push_str(&format!(
            " ({} not analyzed: raise --max-trials)",
            a.not_analyzed.len()
        ));
    }
    line.push_str(&format!(" · {}", a.spend.describe()));
    if saved && !a.trials.is_empty() {
        line.push_str(&format!(
            " · saved {}",
            ui::path(&a.job.join(analyze::FILE_NAME))
        ));
    }
    say!("  {}", dim(line));
}
