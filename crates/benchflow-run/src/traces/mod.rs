//! Trace export: a job's trajectories as a Hugging Face dataset folder or
//! as OpenTelemetry spans (`bench export <job> --format hf|otel`).
//!
//! ```text
//! <job>-hf/                         --format hf
//!   README.md                       the dataset card (loads with `datasets`)
//!   data/train.jsonl                one row per episode (see hf)
//! <job>-otel.jsonl                  --format otel: one OTLP JSON trace per trial
//! ```
//!
//! Each trial's trajectory is its ATIF document: `trainer/atif.json` when
//! the trial has one (both engines write it), else `agent/trajectory.json`,
//! else one built from `trajectory/acp_trajectory.jsonl` as the trainer
//! files build it. Jobs written by BenchFlow 0.8 and by this engine read
//! the same way. The conversion of each document follows Harbor's
//! `traces export`, so a trajectory gives the same rows and spans there.

pub mod hf;
pub mod otel;
pub mod pyjson;

use std::path::{Path, PathBuf};

use benchflow_agent::trajectory::{Redactor, python_float_repr, to_atif};
use serde::Serialize;
use serde_json::{Value, json};

use crate::outcome::Status;
use crate::record::TrialRecord;
use crate::trainer::{self, RolloutInfo};

/// Which episodes of a trial become rows.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Episodes {
    /// One row per agent step.
    #[default]
    All,
    /// Only the last, the whole conversation.
    Last,
}

/// Which trials to export.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Filter {
    #[default]
    All,
    /// Reward above 0.
    Success,
    /// Everything else, errored trials included.
    Failure,
}

/// How an export runs.
#[derive(Debug, Clone, Default)]
pub struct ExportOptions {
    pub episodes: Episodes,
    pub filter: Filter,
    /// Add a `conversations_sharegpt` column.
    pub sharegpt: bool,
    /// Add an `instruction` column: the trial's first message.
    pub instruction: bool,
    /// Add a `verifier_output` column: the verifier's stdout and stderr.
    pub verifier_output: bool,
    /// Replace an existing output.
    pub overwrite: bool,
}

/// Where a trial's ATIF document came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum AtifSource {
    /// `trainer/atif.json`.
    Trainer,
    /// `agent/trajectory.json`.
    Agent,
    /// Built from `trajectory/acp_trajectory.jsonl`.
    Rebuilt,
}

/// What happened to one trial.
#[derive(Debug, Clone, Serialize)]
pub struct TrialExport {
    pub trial: String,
    /// Rows (hf) or spans (otel) written.
    pub written: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub atif: Option<AtifSource>,
    /// Why nothing was written.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skipped: Option<String>,
}

/// What an export wrote.
#[derive(Debug, Clone, Serialize)]
pub struct ExportReport {
    pub job: PathBuf,
    pub out: PathBuf,
    pub format: &'static str,
    /// Rows (hf) or traces (otel) written.
    pub written: usize,
    pub trials: Vec<TrialExport>,
}

/// Why an export did not happen.
#[derive(Debug, thiserror::Error)]
pub enum ExportError {
    #[error("{0} is not a job folder: it has no trial folders with result.json")]
    NotAJob(PathBuf),
    #[error("{0} already exists; choose another output or allow overwriting")]
    OutputExists(PathBuf),
    #[error("no trial in {job} has a trajectory to export ({skipped} skipped)")]
    Nothing { job: PathBuf, skipped: usize },
    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

fn io(path: &Path) -> impl FnOnce(std::io::Error) -> ExportError + '_ {
    move |source| ExportError::Io {
        path: path.to_path_buf(),
        source,
    }
}

/// A trial folder and what its result.json says.
struct Trial {
    dir: PathBuf,
    rec: TrialRecord,
    /// result.json's `started_at` as written (the HF `date` column). Only
    /// this is kept: holding each trial's whole result.json made an export
    /// of a 10,000-trial job peak at over 400 MB.
    started_at: String,
}

/// The largest file an export reads from a trial.
const MAX_READ_BYTES: u64 = crate::hardening::MAX_OUTPUT_FILE_BYTES;

/// `rel` in the trial folder `dir`, when it and every folder on the way are
/// real (no links) and it is a regular file of at most [`MAX_READ_BYTES`]. A
/// job can come from someone else, and what an export reads goes into a
/// dataset meant to be shared: a link must not pull in a file of this
/// machine.
fn plain_file(dir: &Path, rel: &str) -> Option<PathBuf> {
    let mut at = dir.to_path_buf();
    let parts: Vec<&str> = rel.split('/').collect();
    for (i, part) in parts.iter().enumerate() {
        at.push(part);
        let m = std::fs::symlink_metadata(&at).ok()?;
        let last = i + 1 == parts.len();
        let fits = if last {
            m.file_type().is_file() && m.len() <= MAX_READ_BYTES
        } else {
            m.file_type().is_dir()
        };
        if !fits {
            return None;
        }
    }
    Some(at)
}

/// Whether `rel` exists in `dir` but is not a [`plain_file`].
fn refused(dir: &Path, rel: &str) -> bool {
    dir.join(rel).exists() && plain_file(dir, rel).is_none()
}

/// A job's trials, read one at a time as the export reaches them: an export
/// reads one trial's record and trajectory at a time (the rows it writes
/// are still all held until the end).
fn trials(job: &Path) -> impl Iterator<Item = Trial> {
    let dirs: Vec<PathBuf> = if job.join("result.json").is_file() {
        vec![job.to_path_buf()]
    } else {
        crate::read::trial_dirs(job).collect()
    };
    dirs.into_iter()
        .filter(|dir| std::fs::symlink_metadata(dir).is_ok_and(|m| m.file_type().is_dir()))
        .filter_map(|dir| {
            let text = std::fs::read_to_string(plain_file(&dir, "result.json")?).ok()?;
            let raw: Value = serde_json::from_str(&text).ok()?;
            let rec = crate::read::record_from_json(&raw, &dir)?;
            let started_at = raw
                .get("started_at")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            Some(Trial {
                dir,
                rec,
                started_at,
            })
        })
}

/// A trial's ATIF document, and where it came from.
pub fn trial_atif(dir: &Path, rec: &TrialRecord) -> Result<(Value, AtifSource), String> {
    for (rel, from) in [
        (trainer::ATIF, AtifSource::Trainer),
        ("agent/trajectory.json", AtifSource::Agent),
    ] {
        if refused(dir, rel) {
            return Err(format!(
                "{rel} is a link, not a regular file, or over 100 MB; it was not read"
            ));
        }
        let Some(path) = plain_file(dir, rel) else {
            continue;
        };
        {
            let text = std::fs::read_to_string(&path).map_err(|e| format!("{rel}: {e}"))?;
            let doc: Value =
                serde_json::from_str(&text).map_err(|e| format!("{rel} is not JSON: {e}"))?;
            return Ok((doc, from));
        }
    }
    let events_file = "trajectory/acp_trajectory.jsonl";
    if refused(dir, events_file) {
        return Err(format!(
            "{events_file} is a link, not a regular file, or over 100 MB; it was not read"
        ));
    }
    let events =
        trainer::read_events(dir).map_err(|e| format!("trajectory/acp_trajectory.jsonl: {e}"))?;
    let mut doc = to_atif(&events, &trainer::atif_meta(&RolloutInfo::from_record(rec)))
        .ok_or_else(|| no_steps_reason(rec))?;
    Redactor::default().redact_json(&mut doc);
    Ok((doc, AtifSource::Rebuilt))
}

/// Why a trial has no trajectory: it stopped before its agent ran, or its
/// agent recorded nothing.
fn no_steps_reason(rec: &TrialRecord) -> String {
    use crate::outcome::Phase;
    match &rec.problem {
        Some(p) if matches!(p.phase, Phase::Image | Phase::Start | Phase::Setup) => format!(
            "no trajectory: the trial stopped while {}, before the agent ran ({})",
            p.phase.label(),
            p.code
        ),
        _ => "no trajectory: the agent recorded no steps".to_string(),
    }
}

fn passes(filter: Filter, rec: &TrialRecord) -> bool {
    let success = rec.reward.is_some_and(|r| r > 0.0);
    match filter {
        Filter::All => true,
        Filter::Success => success,
        Filter::Failure => !success,
    }
}

fn job_name(job: &Path) -> String {
    job.canonicalize()
        .ok()
        .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
        .unwrap_or_else(|| "job".into())
}

fn run_meta(t: &Trial, run_id: &str) -> hf::RunMeta {
    let model = t.rec.model.clone().filter(|m| !m.is_empty());
    let provider = model
        .as_deref()
        .and_then(|m| m.split_once('/'))
        .map_or("unknown-provider".to_string(), |(p, _)| p.to_string());
    let result = match t.rec.reward {
        Some(r) => Some(python_float_repr(r)),
        None => t
            .rec
            .problem
            .as_ref()
            .map(|p| p.code.clone())
            .or_else(|| (t.rec.status == Status::Errored).then(|| "errored".to_string())),
    };
    hf::RunMeta {
        agent: if t.rec.agent.is_empty() {
            "unknown-agent".into()
        } else {
            t.rec.agent.clone()
        },
        model: model.unwrap_or_else(|| "unknown-model".into()),
        model_provider: provider,
        date: t.started_at.clone(),
        run_id: run_id.to_string(),
        task: t.rec.task_name.clone(),
        trial_name: t.rec.rollout_name.clone(),
        result,
    }
}

/// The first non-blank message, as Harbor's `instruction` column.
fn instruction(doc: &Value) -> Value {
    doc.get("steps")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|s| s.get("message").and_then(Value::as_str))
        .map(str::trim)
        .find(|m| !m.is_empty())
        .map_or(Value::Null, |m| json!(m))
}

fn verifier_output(dir: &Path) -> Value {
    let parts: Vec<String> = ["verifier/test-stdout.txt", "verifier/test-stderr.txt"]
        .iter()
        .filter_map(|f| std::fs::read(plain_file(dir, f)?).ok())
        .map(|b| String::from_utf8_lossy(&b).into_owned())
        .collect();
    if parts.is_empty() {
        Value::Null
    } else {
        json!(parts.concat())
    }
}

fn prepare(out: &Path, overwrite: bool, dir: bool) -> Result<(), ExportError> {
    if out.exists() {
        if !overwrite {
            return Err(ExportError::OutputExists(out.to_path_buf()));
        }
        if out.is_dir() {
            std::fs::remove_dir_all(out).map_err(io(out))?;
        } else {
            std::fs::remove_file(out).map_err(io(out))?;
        }
    }
    let make = if dir { Some(out) } else { out.parent() };
    if let Some(d) = make.filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(d).map_err(io(d))?;
    }
    Ok(())
}

fn check_job(job: &Path) -> Result<std::iter::Peekable<impl Iterator<Item = Trial>>, ExportError> {
    let mut all = trials(job).peekable();
    if all.peek().is_none() {
        return Err(ExportError::NotAJob(job.to_path_buf()));
    }
    Ok(all)
}

/// What the dataset card says about the trials it read.
#[derive(Default)]
struct Seen {
    trials: usize,
    agents: std::collections::BTreeSet<String>,
    models: std::collections::BTreeSet<String>,
}

impl Seen {
    fn add(&mut self, rec: &TrialRecord) {
        self.trials += 1;
        self.agents.insert(rec.agent.clone());
        if let Some(m) = rec.model.as_ref().filter(|m| !m.is_empty()) {
            self.models.insert(m.clone());
        }
    }
}

/// Write `job`'s trajectories as a Hugging Face dataset folder at `out`.
pub fn export_hf(
    job: &Path,
    out: &Path,
    opts: &ExportOptions,
) -> Result<ExportReport, ExportError> {
    let all = check_job(job)?;
    let run_id = job_name(job);
    let mut lines = Vec::new();
    let mut report = Vec::new();
    let mut seen = Seen::default();
    for t in all {
        seen.add(&t.rec);
        let mut entry = TrialExport {
            trial: t.rec.rollout_name.clone(),
            written: 0,
            atif: None,
            skipped: None,
        };
        if !passes(opts.filter, &t.rec) {
            entry.skipped = Some("left out by --filter".into());
            report.push(entry);
            continue;
        }
        match trial_atif(&t.dir, &t.rec) {
            Err(why) => entry.skipped = Some(why),
            Ok((doc, from)) => {
                entry.atif = Some(from);
                match hf::rows(&doc, &run_meta(&t, &run_id)) {
                    Err(e) => entry.skipped = Some(e.to_string()),
                    Ok(mut rows) => {
                        if opts.episodes == Episodes::Last && rows.len() > 1 {
                            rows = rows.split_off(rows.len() - 1);
                        }
                        if rows.is_empty() {
                            entry.skipped = Some("the trajectory has no agent step".into());
                        }
                        for mut row in rows {
                            if opts.instruction {
                                row.insert("instruction".into(), instruction(&doc));
                            }
                            if opts.verifier_output {
                                row.insert("verifier_output".into(), verifier_output(&t.dir));
                            }
                            if opts.sharegpt {
                                let conv =
                                    row["conversations"].as_array().cloned().unwrap_or_default();
                                row.insert(
                                    "conversations_sharegpt".into(),
                                    Value::Array(hf::sharegpt(&conv)),
                                );
                            }
                            lines.push(serde_json::to_string(&Value::Object(row)).expect("JSON"));
                            entry.written += 1;
                        }
                    }
                }
            }
        }
        report.push(entry);
    }
    if lines.is_empty() {
        return Err(ExportError::Nothing {
            job: job.to_path_buf(),
            skipped: report.len(),
        });
    }
    prepare(out, opts.overwrite, true)?;
    let data = out.join("data/train.jsonl");
    std::fs::create_dir_all(out.join("data")).map_err(io(out))?;
    std::fs::write(&data, lines.join("\n") + "\n").map_err(io(&data))?;
    let card = out.join("README.md");
    let rep = ExportReport {
        job: job.to_path_buf(),
        out: out.to_path_buf(),
        format: "hf",
        written: lines.len(),
        trials: report,
    };
    std::fs::write(&card, dataset_card(&run_id, &seen, &rep, opts)).map_err(io(&card))?;
    Ok(rep)
}

/// Write `job`'s trajectories as OTLP JSON lines at `out`, one trace per
/// trial.
pub fn export_otel(
    job: &Path,
    out: &Path,
    opts: &ExportOptions,
) -> Result<ExportReport, ExportError> {
    let all = check_job(job)?;
    let mut lines = Vec::new();
    let mut report = Vec::new();
    for t in all {
        let name = t
            .dir
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| t.rec.rollout_name.clone());
        let mut entry = TrialExport {
            trial: t.rec.rollout_name.clone(),
            written: 0,
            atif: None,
            skipped: None,
        };
        if !passes(opts.filter, &t.rec) {
            entry.skipped = Some("left out by --filter".into());
        } else {
            match trial_atif(&t.dir, &t.rec) {
                Err(why) => entry.skipped = Some(why),
                Ok((doc, from)) => {
                    entry.atif = Some(from);
                    let line = otel::trace_line(&doc, &name, "benchflow");
                    entry.written = line.matches("\"spanId\": ").count();
                    lines.push(line);
                }
            }
        }
        report.push(entry);
    }
    if lines.is_empty() {
        return Err(ExportError::Nothing {
            job: job.to_path_buf(),
            skipped: report.len(),
        });
    }
    prepare(out, opts.overwrite, false)?;
    std::fs::write(out, lines.join("\n") + "\n").map_err(io(out))?;
    Ok(ExportReport {
        job: job.to_path_buf(),
        out: out.to_path_buf(),
        format: "otel",
        written: lines.len(),
        trials: report,
    })
}

fn dataset_card(run_id: &str, seen: &Seen, rep: &ExportReport, opts: &ExportOptions) -> String {
    let (agents, models) = (&seen.agents, &seen.models);
    let exported = rep.trials.iter().filter(|t| t.written > 0).count();
    let count = |s: AtifSource| {
        rep.trials
            .iter()
            .filter(|t| t.written > 0 && t.atif == Some(s))
            .count()
    };
    let mut flags = String::from("--format hf");
    if opts.sharegpt {
        flags.push_str(" --sharegpt");
    }
    if opts.episodes == Episodes::Last {
        flags.push_str(" --episodes last");
    }
    match opts.filter {
        Filter::All => {}
        Filter::Success => flags.push_str(" --filter success"),
        Filter::Failure => flags.push_str(" --filter failure"),
    }
    if opts.instruction {
        flags.push_str(" --instruction");
    }
    if opts.verifier_output {
        flags.push_str(" --verifier-output");
    }
    let mut columns = vec![
        (
            "conversations",
            "The chat up to and including this episode's agent step: a list of `{role, content}` with roles `user` and `assistant`",
        ),
        ("agent", "The agent, as the trajectory names it"),
        ("model", "The model"),
        (
            "model_provider",
            "The model's provider: the part of the model name before the first `/`",
        ),
        ("date", "When the trial started"),
        ("task", "The task's name"),
        (
            "episode",
            "`episode-<n>`: the n-th agent step of the trial, from 0",
        ),
        ("run_id", "The job's name"),
        ("trial_name", "The trial's folder name"),
        (
            "result",
            "The trial's reward, or the code of what stopped it (`bench explain <code>`)",
        ),
    ];
    if opts.instruction {
        columns.push(("instruction", "The trial's first message"));
    }
    if opts.verifier_output {
        columns.push(("verifier_output", "The verifier's stdout and stderr"));
    }
    if opts.sharegpt {
        columns.push((
            "conversations_sharegpt",
            "`conversations` in ShareGPT form: `{from, value}` with `human` and `gpt`",
        ));
    }
    let mut s = String::from(
        "---\nconfigs:\n- config_name: default\n  data_files:\n  - split: train\n    path: data/train.jsonl\ntags:\n- benchflow\n- agent-traces\n- atif\n---\n\n",
    );
    s.push_str(&format!("# Agent traces from {run_id}\n\n"));
    s.push_str(&format!(
        "{} from {exported} of {} in the BenchFlow job `{run_id}`: one row per agent step, holding the conversation up to that step (`--episodes last` keeps one row per trial). Agents: {}. Models: {}.\n\n",
        count_of(rep.written, "row"),
        count_of(seen.trials, "trial"),
        agents.iter().map(|a| format!("`{a}`")).collect::<Vec<_>>().join(", "),
        if models.is_empty() { "none recorded".to_string() } else { models.iter().map(|m| format!("`{m}`")).collect::<Vec<_>>().join(", ") },
    ));
    s.push_str(&format!(
        "Written by `bench export <job> {flags}` (bench {}). Trajectories came from each trial's `trainer/atif.json` ({}), `agent/trajectory.json` ({}), or were rebuilt from `trajectory/acp_trajectory.jsonl` ({}).\n\n",
        env!("CARGO_PKG_VERSION"),
        count(AtifSource::Trainer),
        count(AtifSource::Agent),
        count(AtifSource::Rebuilt),
    ));
    s.push_str("Load it with `datasets.load_dataset(\"<this folder>\")`.\n\n## Columns\n\n| Column | Meaning |\n|---|---|\n");
    for (c, m) in &columns {
        s.push_str(&format!("| `{c}` | {m} |\n"));
    }
    let skipped: Vec<&TrialExport> = rep.trials.iter().filter(|t| t.skipped.is_some()).collect();
    if !skipped.is_empty() {
        s.push_str("\n## Trials left out\n\n");
        for t in skipped {
            s.push_str(&format!(
                "- `{}`: {}\n",
                t.trial,
                t.skipped.as_deref().unwrap_or("")
            ));
        }
    }
    s.push_str("\n## Before sharing\n\nThe rows hold the tasks' instructions and files as the agents saw them. Many benchmarks ask that their tasks stay out of training data, and some carry a canary string for that; check each task's terms before publishing this dataset.\n");
    s
}

/// `1 row`, `2 rows`.
fn count_of(n: usize, word: &str) -> String {
    if n == 1 {
        format!("1 {word}")
    } else {
        format!("{n} {word}s")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn providers_come_from_the_model_name() {
        let t = Trial {
            dir: PathBuf::new(),
            rec: crate::read::record_from_json(
                &json!({"task_name": "t", "rollout_name": "t__1", "agent": "claude", "model": "vercel/deepseek/x", "rewards": {"reward": 1}}),
                Path::new("."),
            )
            .unwrap(),
            started_at: "2026-10-10 05:37:38.402901".into(),
        };
        let m = run_meta(&t, "job");
        assert_eq!(m.model_provider, "vercel");
        assert_eq!(m.result.as_deref(), Some("1.0"));
        assert_eq!(m.date, "2026-10-10 05:37:38.402901");
    }

    #[test]
    fn a_trial_stopped_before_its_agent_says_so() {
        let rec = |problem: Value| {
            crate::read::record_from_json(
                &json!({"task_name": "t", "rollout_name": "t__1", "status": "errored", "problem": problem}),
                Path::new("."),
            )
            .unwrap()
        };
        let setup = rec(json!({"code": "R401", "owner": "you", "phase": "setup",
            "message": "m", "next": "n", "transient": false}));
        assert_eq!(
            no_steps_reason(&setup),
            "no trajectory: the trial stopped while setting up the sandbox, before the agent ran (R401)"
        );
        let agent = rec(json!({"code": "R202", "owner": "agent", "phase": "agent",
            "message": "m", "next": "n", "transient": false}));
        assert_eq!(
            no_steps_reason(&agent),
            "no trajectory: the agent recorded no steps"
        );
    }
}
