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

use benchflow_agent::trajectory::{
    DERIVED_TEXT_CAP, Redactor, cap_long_strings, python_float_repr, to_atif,
};
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
    let dirs: Vec<PathBuf> = if crate::read::is_trial(job) {
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

/// The HF columns that do not come from the (already redacted) ATIF
/// document, redacted when a row is written.
const NOT_FROM_THE_DOCUMENT: &[&str] = &[
    "verifier_output",
    "agent",
    "model",
    "model_provider",
    "date",
    "task",
    "episode",
    "run_id",
    "trial_name",
];

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
            let mut doc: Value =
                serde_json::from_str(&text).map_err(|e| format!("{rel} is not JSON: {e}"))?;
            // Another harness's document never passed BenchFlow's redactor,
            // and an export is meant to be shared.
            Redactor::default().redact_json(&mut doc);
            // Documents from 0.8 and earlier builds start with the prompt
            // twice, and may hold a huge string again; the export holds each
            // prompt once and caps what the raw log keeps whole.
            drop_doubled_prompt(&mut doc);
            cap_long_strings(
                &mut doc,
                DERIVED_TEXT_CAP,
                "trajectory/acp_trajectory.jsonl",
            );
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
    let mut doc = to_atif(
        &events,
        &trainer::atif_meta(&RolloutInfo::from_record(rec).without_recorded_prompts(&events)),
    )
    .ok_or_else(|| no_steps_reason(rec))?;
    Redactor::default().redact_json(&mut doc);
    cap_long_strings(
        &mut doc,
        DERIVED_TEXT_CAP,
        "trajectory/acp_trajectory.jsonl",
    );
    Ok((doc, AtifSource::Rebuilt))
}

/// Drop the first of two leading `user` steps with the same message (the
/// prompt written twice by 0.8 and earlier builds), renumbering the steps.
fn drop_doubled_prompt(doc: &mut Value) {
    let Some(steps) = doc.get_mut("steps").and_then(Value::as_array_mut) else {
        return;
    };
    let doubled = steps.len() >= 2
        && steps[0]["source"] == "user"
        && steps[1]["source"] == "user"
        && steps[0]["message"] == steps[1]["message"];
    if !doubled {
        return;
    }
    steps.remove(0);
    for (i, step) in steps.iter_mut().enumerate() {
        if let Some(step) = step.as_object_mut() {
            step.insert("step_id".into(), json!(i + 1));
        }
    }
    let total = steps.len();
    if let Some(m) = doc.get_mut("final_metrics").and_then(Value::as_object_mut)
        && m.contains_key("total_steps")
    {
        m.insert("total_steps".into(), json!(total));
    }
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

/// The `instruction` column: the first non-blank user message, else the
/// first non-blank message. A native loop's trajectory starts with its system
/// prompt, which is the same for every task and not the instruction.
fn instruction(doc: &Value) -> Value {
    let steps: Vec<&Value> = doc
        .get("steps")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .collect();
    let first = |user_only: bool| {
        steps
            .iter()
            .filter(|s| !user_only || s.get("source").and_then(Value::as_str) == Some("user"))
            .filter_map(|s| s.get("message").and_then(Value::as_str))
            .map(str::trim)
            .find(|m| !m.is_empty())
    };
    first(true)
        .or_else(|| first(false))
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
                            // The columns built from the trajectory come from the
                            // document `trial_atif` redacted; the rest (the
                            // verifier's output, the trial's metadata) are
                            // redacted here, each value once. Redacting a whole
                            // conversation again would see the message and every
                            // tool call as one string, and a key's BEGIN marker in
                            // one call and END marker in the next would swallow
                            // what lies between.
                            let redactor = Redactor::default();
                            for key in NOT_FROM_THE_DOCUMENT {
                                if let Some(value) = row.get_mut(*key) {
                                    redactor.redact_json(value);
                                }
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
    let text = dataset_card(&run_id, &seen, &rep, opts);
    std::fs::write(&card, Redactor::default().redact(&text).as_ref()).map_err(io(&card))?;
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
                    // The document was redacted value by value in `trial_atif`;
                    // the rest of the line is names, ids and numbers. Redacting
                    // the serialized line could break an escape or join two
                    // values (a key's BEGIN and END markers in two spans).
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

    /// E1: another harness's job folder has a result.json of its own next
    /// to its trial folders; the export read the folder as one trial with
    /// no trajectory and refused the job.
    #[test]
    fn a_job_with_its_own_result_json_exports_its_trials() {
        let job = tempfile::tempdir().unwrap();
        std::fs::write(job.path().join("result.json"), r#"{"n_total_trials": 1}"#).unwrap();
        let trial = job.path().join("fix-git__SFjiUpt");
        std::fs::create_dir_all(trial.join("agent")).unwrap();
        std::fs::write(
            trial.join("result.json"),
            json!({"task_name": "fix-git", "verifier_result": {"rewards": {"reward": 1.0}},
                   "agent_info": {"name": "opencode", "model_info": {"name": "m", "provider": "p"}},
                   "started_at": "2026-10-12T02:27:36.383539Z"})
            .to_string(),
        )
        .unwrap();
        std::fs::write(
            trial.join("agent/trajectory.json"),
            json!({"schema_version": "ATIF-v1.7", "agent": {"name": "opencode", "version": "1.18.11"},
                   "steps": [{"step_id": 1, "source": "user", "message": "Fix it."},
                             {"step_id": 2, "source": "agent", "message": "Done."}]})
            .to_string(),
        )
        .unwrap();
        let found: Vec<Trial> = trials(job.path()).collect();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].dir, trial);
        assert_eq!(run_meta(&found[0], "job").result.as_deref(), Some("1.0"));
        let (doc, _) = trial_atif(&trial, &found[0].rec).unwrap();
        assert_eq!(doc["steps"][1]["message"], "Done.");
    }

    /// Review E1-A 1: a native loop's ATIF starts with its system prompt;
    /// the instruction column is the task's prompt, not that.
    #[test]
    fn the_instruction_is_the_first_user_message() {
        let doc = json!({"steps": [
            {"step_id": 1, "source": "system", "message": "You are an autonomous agent."},
            {"step_id": 2, "source": "user", "message": "  Fix the repo.  "},
            {"step_id": 3, "source": "agent", "message": "Done."}]});
        assert_eq!(instruction(&doc), json!("Fix the repo."));
        // With no user step, the first message stands in, as before.
        let doc = json!({"steps": [{"step_id": 1, "source": "agent", "message": "hi"}]});
        assert_eq!(instruction(&doc), json!("hi"));
    }

    /// Review E1-B 2: another harness's `agent/trajectory.json` is exported
    /// redacted, as BenchFlow's own documents are.
    #[test]
    fn another_harness_trajectory_is_redacted_before_export() {
        let trial = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(trial.path().join("agent")).unwrap();
        std::fs::write(
            trial.path().join("result.json"),
            json!({"task_name": "t", "verifier_result": {"rewards": {"reward": 1.0}}}).to_string(),
        )
        .unwrap();
        let key = "sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789";
        std::fs::write(
            trial.path().join("agent/trajectory.json"),
            json!({"schema_version": "ATIF-v1.7", "agent": {"name": "a", "version": "1"},
                   "steps": [{"step_id": 1, "source": "agent", "message": format!("the key is {key}")}]})
            .to_string(),
        )
        .unwrap();
        let rec = crate::read::read_trial(trial.path()).unwrap();
        let (doc, _) = trial_atif(trial.path(), &rec).unwrap();
        let text = doc.to_string();
        assert!(!text.contains(key), "{text}");
    }

    /// Review E1-C-B 1: `--verifier-output` copied the verifier's stdout
    /// into the dataset unredacted, while every other column came from the
    /// redacted trajectory.
    #[test]
    fn the_verifier_output_column_is_redacted() {
        let job = tempfile::tempdir().unwrap();
        let trial = job.path().join("t__1");
        std::fs::create_dir_all(trial.join("agent")).unwrap();
        std::fs::create_dir_all(trial.join("verifier")).unwrap();
        std::fs::write(
            trial.join("result.json"),
            json!({"task_name": "t", "rollout_name": "t__1", "rewards": {"reward": 0.0}})
                .to_string(),
        )
        .unwrap();
        std::fs::write(
            trial.join("agent/trajectory.json"),
            json!({"schema_version": "ATIF-v1.8", "agent": {"name": "a", "version": "1"},
                   "steps": [{"step_id": 1, "source": "user", "message": "Do it."},
                             {"step_id": 2, "source": "agent", "message": "Done."}]})
            .to_string(),
        )
        .unwrap();
        let key = "sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789";
        std::fs::write(
            trial.join("verifier/test-stdout.txt"),
            format!("FAILED: env ANTHROPIC_API_KEY={key}\n"),
        )
        .unwrap();
        let out = job.path().join("out");
        let opts = ExportOptions {
            verifier_output: true,
            ..ExportOptions::default()
        };
        export_hf(job.path(), &out, &opts).unwrap();
        let rows = std::fs::read_to_string(out.join("data/train.jsonl")).unwrap();
        assert!(rows.contains("FAILED: env ANTHROPIC_API_KEY="), "{rows}");
        assert!(!rows.contains(key), "{rows}");
        let otel = job.path().join("otel.jsonl");
        export_otel(job.path(), &otel, &ExportOptions::default()).unwrap();
        assert!(!std::fs::read_to_string(&otel).unwrap().contains(key));
    }

    /// Review E1-D-A 1: redacting the serialized OpenTelemetry line matched
    /// `?api_key=` up to the backslash of an escaped quote and left the line
    /// unreadable. The trace's values are redacted before it is serialized.
    #[test]
    fn a_redacted_trace_line_is_still_json() {
        let job = tempfile::tempdir().unwrap();
        let trial = job.path().join("t__1");
        std::fs::create_dir_all(trial.join("agent")).unwrap();
        std::fs::write(
            trial.join("result.json"),
            json!({"task_name": "t", "rollout_name": "t__1", "rewards": {"reward": 1.0}})
                .to_string(),
        )
        .unwrap();
        let code = r#"url = "https://api.example.com/v1/data?api_key=" + key"#;
        let key = "sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789";
        std::fs::write(
            trial.join("agent/trajectory.json"),
            json!({"schema_version": "ATIF-v1.8", "agent": {"name": "a", "version": "1"},
                   "steps": [{"step_id": 1, "source": "user", "message": "Write the client."},
                             {"step_id": 2, "source": "agent", "message": code,
                              "tool_calls": [{"tool_call_id": "c1", "function_name": "write",
                                              "arguments": {"content": code}}],
                              "observation": {"results": [{"source_call_id": "c1",
                                  "content": format!("{code}\nexport KEY={key}")}]}}]})
            .to_string(),
        )
        .unwrap();
        let out = job.path().join("otel.jsonl");
        export_otel(job.path(), &out, &ExportOptions::default()).unwrap();
        let text = std::fs::read_to_string(&out).unwrap();
        assert!(!text.contains(key), "{text}");
        for line in text.lines() {
            let trace: Value = serde_json::from_str(line).expect("each trace line is JSON");
            assert!(trace.to_string().contains("?api_key="), "{trace}");
        }
    }

    /// Review E1-D-B 2: redacting a whole row again saw an assistant turn's
    /// message and tool calls as one string, and a key's BEGIN marker in one
    /// call and END marker in the next merged the two calls. Rows and traces
    /// keep both calls.
    #[test]
    fn key_markers_in_two_tool_calls_stay_two_tool_calls() {
        let job = tempfile::tempdir().unwrap();
        let trial = job.path().join("t__1");
        std::fs::create_dir_all(trial.join("agent")).unwrap();
        std::fs::write(
            trial.join("result.json"),
            json!({"task_name": "t", "rollout_name": "t__1", "rewards": {"reward": 1.0}})
                .to_string(),
        )
        .unwrap();
        let begin = "head -1 /app/ssl/server.key | grep -qx -- '-----BEGIN PRIVATE KEY-----'";
        let end = "tail -1 /app/ssl/server.key | grep -qx -- '-----END PRIVATE KEY-----'";
        std::fs::write(
            trial.join("agent/trajectory.json"),
            json!({"schema_version": "ATIF-v1.8", "agent": {"name": "a", "version": "1"},
                   "steps": [{"step_id": 1, "source": "user", "message": "Check the key."},
                             {"step_id": 2, "source": "agent", "message": "Checking both ends.",
                              "tool_calls": [
                                  {"tool_call_id": "c1", "function_name": "bash", "arguments": {"command": begin}},
                                  {"tool_call_id": "c2", "function_name": "bash", "arguments": {"command": end}}],
                              "observation": {"results": [
                                  {"source_call_id": "c1", "content": "ok"},
                                  {"source_call_id": "c2", "content": "ok"}]}}]})
            .to_string(),
        )
        .unwrap();
        let out = job.path().join("hf");
        let opts = ExportOptions {
            sharegpt: true,
            ..ExportOptions::default()
        };
        export_hf(job.path(), &out, &opts).unwrap();
        let rows = std::fs::read_to_string(out.join("data/train.jsonl")).unwrap();
        let row: Value = serde_json::from_str(rows.lines().next().unwrap()).unwrap();
        let assistant = row["conversations"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["role"] == "assistant")
            .unwrap()["content"]
            .as_str()
            .unwrap()
            .to_string();
        assert_eq!(assistant.matches("<tool_call>").count(), 2, "{assistant}");
        assert!(assistant.contains("BEGIN PRIVATE KEY") && assistant.contains("END PRIVATE KEY"));
        let otel = job.path().join("otel.jsonl");
        export_otel(job.path(), &otel, &ExportOptions::default()).unwrap();
        let line = std::fs::read_to_string(&otel).unwrap();
        let trace: Value = serde_json::from_str(line.trim()).unwrap();
        let text = trace.to_string();
        assert!(
            text.contains("BEGIN PRIVATE KEY") && text.contains("END PRIVATE KEY"),
            "{text}"
        );
    }

    /// Review E1-E-B 1: a private key read in two tool calls left half in
    /// each result; neither value held both markers, so neither was redacted,
    /// and the row's tool message joined the halves into the whole key.
    #[test]
    fn a_key_split_over_two_tool_results_is_not_exported() {
        let job = tempfile::tempdir().unwrap();
        let trial = job.path().join("t__1");
        std::fs::create_dir_all(trial.join("agent")).unwrap();
        std::fs::write(
            trial.join("result.json"),
            json!({"task_name": "t", "rollout_name": "t__1", "rewards": {"reward": 1.0}})
                .to_string(),
        )
        .unwrap();
        let line = |c: char| std::iter::repeat_n(c, 64).collect::<String>();
        let (b1, b2, b3, b4) = (line('A'), line('B'), line('C'), line('D'));
        let first = format!("-----BEGIN PRIVATE KEY-----\n{b1}\n{b2}");
        let second = format!("{b3}\n{b4}\n-----END PRIVATE KEY-----");
        std::fs::write(
            trial.join("agent/trajectory.json"),
            json!({"schema_version": "ATIF-v1.8", "agent": {"name": "a", "version": "1"},
                   "steps": [{"step_id": 1, "source": "user", "message": "Read the key."},
                             {"step_id": 2, "source": "agent", "message": "",
                              "tool_calls": [
                                  {"tool_call_id": "c1", "function_name": "bash", "arguments": {"command": "head -n 3 server.key"}},
                                  {"tool_call_id": "c2", "function_name": "bash", "arguments": {"command": "tail -n +4 server.key"}}],
                              "observation": {"results": [
                                  {"source_call_id": "c1", "content": first},
                                  {"source_call_id": "c2", "content": second}]}},
                             {"step_id": 3, "source": "agent", "message": "Done."}]})
            .to_string(),
        )
        .unwrap();
        let out = job.path().join("hf");
        export_hf(job.path(), &out, &ExportOptions::default()).unwrap();
        let rows = std::fs::read_to_string(out.join("data/train.jsonl")).unwrap();
        let otel = job.path().join("otel.jsonl");
        export_otel(job.path(), &otel, &ExportOptions::default()).unwrap();
        let trace = std::fs::read_to_string(&otel).unwrap();
        for text in [&rows, &trace] {
            for body in [&b1, &b2, &b3, &b4] {
                assert!(!text.contains(body.as_str()), "{text}");
            }
            assert!(text.contains("BEGIN PRIVATE KEY") && text.contains("END PRIVATE KEY"));
        }
        for line in rows.lines().chain(trace.lines()) {
            serde_json::from_str::<Value>(line).expect("JSON");
        }
        // Still two tool calls.
        let row: Value = serde_json::from_str(rows.lines().last().unwrap()).unwrap();
        let calls: usize = row["conversations"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|m| m["role"] == "assistant")
            .map(|m| {
                m["content"]
                    .as_str()
                    .unwrap()
                    .matches("<tool_call>")
                    .count()
            })
            .sum();
        assert_eq!(calls, 2, "{row}");
    }

    /// Review E1-F-B: an OpenSSH RSA key (made up here) read in two tool
    /// calls. Its last body line is short (`AQIDBA==`), and the END half,
    /// which holds the key's last prime, was exported whole.
    #[test]
    fn an_openssh_rsa_key_split_over_two_tool_results_is_not_exported() {
        let job = tempfile::tempdir().unwrap();
        let trial = job.path().join("t__1");
        std::fs::create_dir_all(trial.join("agent")).unwrap();
        std::fs::write(
            trial.join("result.json"),
            json!({"task_name": "t", "rollout_name": "t__1", "rewards": {"reward": 1.0}})
                .to_string(),
        )
        .unwrap();
        let line = |c: char| std::iter::repeat_n(c, 70).collect::<String>();
        let head_body = format!(
            "b3BlbnNzaC1rZXktdjEAAAAABG5vbmU{}\n{}",
            "x".repeat(39),
            line('h')
        );
        let tail_body = [line('p'), line('q'), line('Q'), "AQIDBA==".to_string()];
        let first = format!("-----BEGIN OPENSSH PRIVATE KEY-----\n{head_body}");
        let second = format!(
            "{}\n-----END OPENSSH PRIVATE KEY-----\n",
            tail_body.join("\n")
        );
        std::fs::write(
            trial.join("agent/trajectory.json"),
            json!({"schema_version": "ATIF-v1.8", "agent": {"name": "a", "version": "1"},
                   "steps": [{"step_id": 1, "source": "user", "message": "Read the key."},
                             {"step_id": 2, "source": "agent", "message": "",
                              "tool_calls": [
                                  {"tool_call_id": "c1", "function_name": "bash", "arguments": {"command": "head -n 3 id_rsa"}},
                                  {"tool_call_id": "c2", "function_name": "bash", "arguments": {"command": "tail -n +4 id_rsa"}}],
                              "observation": {"results": [
                                  {"source_call_id": "c1", "content": first},
                                  {"source_call_id": "c2", "content": second}]}},
                             {"step_id": 3, "source": "agent", "message": "Done."}]})
            .to_string(),
        )
        .unwrap();
        let out = job.path().join("hf");
        export_hf(job.path(), &out, &ExportOptions::default()).unwrap();
        let rows = std::fs::read_to_string(out.join("data/train.jsonl")).unwrap();
        let otel = job.path().join("otel.jsonl");
        export_otel(job.path(), &otel, &ExportOptions::default()).unwrap();
        let trace = std::fs::read_to_string(&otel).unwrap();
        for text in [&rows, &trace] {
            for body in tail_body.iter().chain([&line('h')]) {
                assert!(!text.contains(body.as_str()), "{body} in {text}");
            }
            assert!(text.contains("END OPENSSH PRIVATE KEY"));
        }
        for l in rows.lines().chain(trace.lines()) {
            serde_json::from_str::<Value>(l).expect("JSON");
        }
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

    #[test]
    fn a_document_with_the_prompt_twice_exports_it_once() {
        let mut doc = json!({"steps": [
            {"step_id": 1, "source": "user", "message": "fix it"},
            {"step_id": 2, "source": "user", "message": "fix it", "timestamp": "2026-10-10T00:00:00+00:00"},
            {"step_id": 3, "source": "agent", "message": "done"},
        ], "final_metrics": {"total_steps": 3}});
        drop_doubled_prompt(&mut doc);
        let steps = doc["steps"].as_array().unwrap();
        assert_eq!(steps.len(), 2);
        assert_eq!(steps[0]["timestamp"], "2026-10-10T00:00:00+00:00");
        assert_eq!(steps[1]["step_id"], 2);
        assert_eq!(doc["final_metrics"]["total_steps"], 2);
        // Two different user turns stay.
        let mut two = json!({"steps": [
            {"step_id": 1, "source": "user", "message": "a"},
            {"step_id": 2, "source": "user", "message": "b"},
        ]});
        drop_doubled_prompt(&mut two);
        assert_eq!(two["steps"].as_array().unwrap().len(), 2);
    }
}
