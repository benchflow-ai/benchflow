//! `bench results --analyze`: a model reads each failed trial and says
//! which part failed and why.
//!
//! A trial is analyzed when it errored, went unscored, or was scored and did
//! not pass. The model gets the trial's record (status, reward, the problem
//! BenchFlow recorded), the instruction, the trajectory's tail, the
//! verifier's output, and the agent's logs (capped, redacted), and answers
//! with the part it blames (the agent, the task, the model provider, the
//! sandbox, the setup, BenchFlow) in a sentence or two, with quotes from the
//! files. Each quote is checked against what was sent. Works on job folders
//! written by this engine and by BenchFlow 0.8 (same trial layout).

use std::path::{Path, PathBuf};

use futures::StreamExt;
use serde::Serialize;
use serde_json::Value;

use super::evidence::{head_tail, squash, tail};
use super::{OPINION, ReviewError, Reviewer, Spend, now_iso, text_field};
use crate::outcome::Status;
use crate::read;
use crate::record::TrialRecord;

/// The file the analysis is saved to, in the job folder.
pub const FILE_NAME: &str = "analysis.json";

/// How many trials are analyzed unless the caller raises it.
pub const DEFAULT_MAX_TRIALS: usize = 10;

/// Model calls in flight at once.
const CONCURRENCY: usize = 4;

/// The part of a run a failure belongs to, as the model judges it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Part {
    Agent,
    Task,
    Provider,
    Sandbox,
    Setup,
    Benchflow,
    Unclear,
}

impl Part {
    pub const ALL: [Part; 7] = [
        Part::Agent,
        Part::Task,
        Part::Provider,
        Part::Sandbox,
        Part::Setup,
        Part::Benchflow,
        Part::Unclear,
    ];

    pub fn id(self) -> &'static str {
        match self {
            Part::Agent => "agent",
            Part::Task => "task",
            Part::Provider => "provider",
            Part::Sandbox => "sandbox",
            Part::Setup => "setup",
            Part::Benchflow => "benchflow",
            Part::Unclear => "unclear",
        }
    }

    /// For people: `the model provider`.
    pub fn label(self) -> &'static str {
        match self {
            Part::Agent => "the agent",
            Part::Task => "the task",
            Part::Provider => "the model provider",
            Part::Sandbox => "the sandbox",
            Part::Setup => "the run's setup",
            Part::Benchflow => "BenchFlow",
            Part::Unclear => "unclear",
        }
    }

    fn parse(s: &str) -> Option<Part> {
        let s = s.trim().to_ascii_lowercase();
        let s = s.trim_start_matches("the ");
        Part::ALL.into_iter().find(|p| p.id() == s).or(match s {
            "model" | "model provider" | "model_provider" | "provider_error" => {
                Some(Part::Provider)
            }
            "you" | "user" | "config" | "configuration" => Some(Part::Setup),
            "harness" => Some(Part::Benchflow),
            "environment" | "infrastructure" | "infra" => Some(Part::Sandbox),
            _ => None,
        })
    }
}

/// A quote the model gave as evidence, checked against what it was sent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Quote {
    pub file: String,
    pub quote: String,
    /// The quoted text is in what the model was sent.
    pub found: bool,
}

/// What BenchFlow recorded for the trial: checked facts, not the model's.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Recorded {
    pub status: Status,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reward: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    /// Whose problem BenchFlow said it was.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub owner: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

/// The model's account of one trial.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TrialAnalysis {
    pub trial: String,
    pub task: String,
    pub dir: PathBuf,
    pub recorded: Recorded,
    pub part: Part,
    pub explanation: String,
    pub evidence: Vec<Quote>,
    /// `high`, `medium`, or `low`, as the model rated itself.
    pub confidence: String,
    /// Why the model could not analyze it, when it could not.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub spend: Spend,
}

/// Trials of one cause, for the summary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CauseGroup {
    pub part: Part,
    pub label: &'static str,
    pub trials: Vec<String>,
}

/// A model's analysis of a job's failed trials.
#[derive(Debug, Clone, Serialize)]
pub struct JobAnalysis {
    pub job: PathBuf,
    pub model: String,
    pub analyzed_at: String,
    pub opinion: &'static str,
    /// Trials that failed, errored, or went unscored.
    pub failed: usize,
    pub trials: Vec<TrialAnalysis>,
    /// Failed trials over the cap, not analyzed.
    pub not_analyzed: Vec<String>,
    pub by_cause: Vec<CauseGroup>,
    pub spend: Spend,
}

/// A trial folder and its record.
pub struct TrialDir {
    pub dir: PathBuf,
    pub record: TrialRecord,
}

/// The job's trials that did not pass, in folder order. A trial folder given
/// as the job is a job of one.
pub fn failed_trials(job: &Path) -> Vec<TrialDir> {
    let dirs: Vec<PathBuf> = if job.join("result.json").is_file() {
        vec![job.to_path_buf()]
    } else {
        read::trial_dirs(job).collect()
    };
    dirs.into_iter()
        .filter_map(|dir| read::read_trial(&dir).map(|record| TrialDir { dir, record }))
        .filter(|t| needs_analysis(&t.record))
        .collect()
}

fn needs_analysis(r: &TrialRecord) -> bool {
    match r.status {
        Status::Errored | Status::Unscored => true,
        Status::Scored => r.passed != Some(true),
        Status::Skipped => false,
    }
}

/// One piece of evidence: a label (usually a file in the trial folder) and
/// the text sent.
struct Piece {
    label: String,
    text: String,
}

/// Size caps for each piece, in bytes.
const INSTRUCTION_CAP: (usize, usize) = (5_000, 2_000);
const TRAJECTORY_CAP: usize = 14_000;
const EVENT_CAP: usize = 1_200;
const VERIFIER_CAP: usize = 6_000;
const LOG_CAP: usize = 3_000;
const DETAIL_CAP: usize = 3_000;

/// Everything the model reads about one trial, redacted.
fn evidence(t: &TrialDir, redactor: &benchflow_agent::trajectory::Redactor) -> Vec<Piece> {
    let r = &t.record;
    let dir = &t.dir;
    let mut pieces: Vec<Piece> = Vec::new();
    let read_text = |rel: &str| std::fs::read_to_string(dir.join(rel)).ok();

    // The record.
    let mut rec = format!(
        "task: {}\ntrial: {}\nagent: {}\nmodel: {}\nstatus: {}\nreward: {}\nrewards: {}\ntimed out: {}\ntool calls: {}\ntokens in/out: {}/{}\ntime: setup {:.0}s, agent {:.0}s, verifier {:.0}s, total {:.0}s\n",
        r.task_name,
        r.rollout_name,
        r.agent,
        r.model.as_deref().unwrap_or("not recorded"),
        status_word(r.status),
        r.reward
            .map(|x| x.to_string())
            .unwrap_or_else(|| "none".into()),
        r.rewards
            .as_ref()
            .map(|m| serde_json::to_string(m).unwrap_or_default())
            .unwrap_or_else(|| "none".into()),
        r.timed_out,
        r.tool_calls,
        r.usage.input_tokens,
        r.usage.output_tokens,
        r.timing.environment_setup,
        r.timing.agent_execution,
        r.timing.verifier,
        r.timing.total,
    );
    if let Some(code) = r.verifier_exit {
        rec.push_str(&format!("verifier exit code: {code}\n"));
    }
    if let Some(p) = &r.problem {
        rec.push_str(&format!(
            "problem BenchFlow recorded: [{}] {} (phase: {:?}; whose problem: {})\nits suggested next step: {}\n",
            p.code,
            p.message,
            p.phase,
            p.owner.label(),
            p.next
        ));
        if let Some(d) = &p.detail {
            rec.push_str(&format!("problem detail:\n{}\n", tail(d, DETAIL_CAP)));
        }
    }
    pieces.push(Piece {
        label: "result.json (BenchFlow's record of the trial)".into(),
        text: rec,
    });

    // The instruction.
    let instruction = read_text("prompts.json")
        .and_then(|t| serde_json::from_str::<Vec<String>>(&t).ok())
        .and_then(|p| p.into_iter().next())
        .or_else(|| r.prompts.first().cloned());
    if let Some(i) = instruction {
        pieces.push(Piece {
            label: "prompts.json (the instruction)".into(),
            text: head_tail(&i, INSTRUCTION_CAP.0, INSTRUCTION_CAP.1),
        });
    }

    // The trajectory's tail.
    for rel in [
        "trajectory/acp_trajectory.jsonl",
        "agent/acp_trajectory.jsonl",
    ] {
        if let Some(text) = read_text(rel) {
            let (events, total) = trajectory_tail(&text);
            pieces.push(Piece {
                label: format!("{rel} (last events; {total} in all)"),
                text: if events.is_empty() {
                    "(the file holds no events)".into()
                } else {
                    events
                },
            });
            break;
        }
    }

    // The verifier's output.
    for rel in [
        "verifier/test-stdout.txt",
        "verifier/test-stderr.txt",
        "verifier/reward.txt",
        "verifier/reward.json",
        "verifier/reward-details.json",
        "verifier/ctrf.json",
    ] {
        if let Some(text) = read_text(rel).filter(|t| !t.trim().is_empty()) {
            pieces.push(Piece {
                label: rel.to_string(),
                text: tail(&text, VERIFIER_CAP),
            });
        }
    }

    // The agent's logs and any exception.
    let mut logs: Vec<String> = std::fs::read_dir(dir.join("agent"))
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|n| n.ends_with(".txt") && !n.starts_with("install"))
        .collect();
    logs.sort_by_key(|n| (n != "stderr.txt", n.clone()));
    for name in logs.into_iter().take(3) {
        let rel = format!("agent/{name}");
        if let Some(text) = read_text(&rel).filter(|t| !t.trim().is_empty()) {
            pieces.push(Piece {
                label: rel,
                text: tail(&text, LOG_CAP),
            });
        }
    }
    if let Some(text) = read_text("exception.txt") {
        pieces.push(Piece {
            label: "exception.txt".into(),
            text: tail(&text, LOG_CAP),
        });
    }

    for p in &mut pieces {
        p.text = redactor.redact(&p.text).into_owned();
    }
    pieces
}

fn status_word(s: Status) -> &'static str {
    match s {
        Status::Scored => "scored",
        Status::Unscored => "unscored (the verifier gave no reward)",
        Status::Errored => "errored",
        Status::Skipped => "skipped",
    }
}

/// The last events of a trajectory, one line each, newest last, and how many
/// events there are.
fn trajectory_tail(text: &str) -> (String, usize) {
    let events: Vec<Value> = serde_json::Deserializer::from_str(text)
        .into_iter::<Value>()
        .map_while(Result::ok)
        .collect();
    let total = events.len();
    let mut lines: Vec<String> = Vec::new();
    let mut bytes = 0;
    for (i, e) in events.iter().enumerate().rev() {
        let line = event_line(i + 1, e);
        if bytes + line.len() > TRAJECTORY_CAP && !lines.is_empty() {
            lines.push(format!("…(events 1–{} not sent)", i + 1));
            break;
        }
        bytes += line.len();
        lines.push(line);
    }
    lines.reverse();
    (lines.join("\n"), total)
}

fn event_line(n: usize, e: &Value) -> String {
    let kind = text_field(e, "type");
    let body = match kind.as_str() {
        "tool_call" => {
            let title = text_field(e, "title");
            let status = text_field(e, "status");
            let output = e
                .get("raw_output")
                .map(compact)
                .filter(|s| !s.is_empty() && s != "null")
                .or_else(|| e.get("content").map(compact))
                .unwrap_or_default();
            format!("[{status}] {title} → {output}")
        }
        _ => {
            let t = text_field(e, "text");
            if t.is_empty() { compact(e) } else { t }
        }
    };
    let body = squash(&body);
    let body = if body.len() > EVENT_CAP {
        let mut cut = EVENT_CAP;
        while !body.is_char_boundary(cut) {
            cut -= 1;
        }
        format!("{}…", &body[..cut])
    } else {
        body
    };
    format!("#{n} {kind}: {body}")
}

fn compact(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// The system prompt for one trial.
pub fn system_prompt() -> String {
    let parts = "- `agent`: the agent ran and did the task wrong, stopped early, looped, or ran out of time while working.\n\
- `task`: the task is at fault: the instruction is unclear or wrong, the verifier crashed or checks something the instruction does not ask, the image is broken, or the task cannot be solved as given.\n\
- `provider`: the model provider failed or refused: a rejected key, a rate or usage limit, an overloaded or retired model, provider errors.\n\
- `sandbox`: the sandbox could not be built, started, or reached, or ran out of disk or memory.\n\
- `setup`: the person running it chose something that cannot work: a model name the agent does not accept, a missing or placeholder key, a wrong flag.\n\
- `benchflow`: the harness running the trial misbehaved (a crash or a wrong record that is not any of the above).\n\
- `unclear`: the files do not show which.";
    format!(
        "You explain why one trial of a benchmark task did not pass. In a trial an AI agent works on a task in a sandbox, then a verifier checks the result and writes a reward (1 means solved). You get BenchFlow's record of the trial (status, reward, and the problem it recorded, with whose problem it thought it was), the instruction, the last events of the agent's trajectory, the verifier's output, and the agent's logs.

Decide which part failed:
{parts}

BenchFlow's record is evidence, not the answer: disagree with it when the files show otherwise. For example, an agent that \"failed\" because every model call was refused with 401 is a `provider` or `setup` failure, not an `agent` failure; a trial with no agent activity at all did not fail because of the agent's skill.

Answer with one JSON object and nothing else:
{{\"part\": \"agent\" | \"task\" | \"provider\" | \"sandbox\" | \"setup\" | \"benchflow\" | \"unclear\", \"explanation\": \"<one or two sentences: what went wrong and why>\", \"evidence\": [{{\"file\": \"<the label of the section it is from, up to the first space>\", \"quote\": \"<text copied exactly from that section, under 200 characters>\"}}], \"confidence\": \"high\" | \"medium\" | \"low\"}}
Give one to three quotes that show it."
    )
}

fn user_prompt(pieces: &[Piece]) -> String {
    let mut out = String::new();
    for p in pieces {
        out.push_str(&format!("=== {} ===\n{}\n\n", p.label, p.text.trim_end()));
    }
    out
}

/// Analyze one trial.
pub async fn analyze_trial(reviewer: &Reviewer, t: &TrialDir) -> TrialAnalysis {
    let pieces = evidence(t, reviewer.redactor());
    let r = &t.record;
    let recorded = Recorded {
        status: r.status,
        reward: r.reward,
        code: r.problem.as_ref().map(|p| p.code.clone()),
        owner: r.problem.as_ref().map(|p| p.owner.label().to_string()),
        message: r.problem.as_ref().map(|p| p.message.clone()),
    };
    let mut out = TrialAnalysis {
        trial: r.rollout_name.clone(),
        task: r.task_name.clone(),
        dir: t.dir.clone(),
        recorded,
        part: Part::Unclear,
        explanation: String::new(),
        evidence: Vec::new(),
        confidence: String::new(),
        error: None,
        spend: Spend::default(),
    };
    match reviewer
        .ask_json(
            &system_prompt(),
            &user_prompt(&pieces),
            12_000,
            &["part", "explanation"],
        )
        .await
    {
        Ok((answer, spend)) => {
            out.spend = spend;
            read_answer(&answer, &pieces, &mut out);
        }
        Err((e, spend)) => {
            out.spend = spend;
            out.error = Some(format!("{} ({})", e.message, e.next));
        }
    }
    out
}

/// Fill `out` from the model's answer; odd fields become defaults.
fn read_answer(answer: &Value, pieces: &[Piece], out: &mut TrialAnalysis) {
    let part_text = text_field(answer, "part");
    out.part = Part::parse(&part_text).unwrap_or(Part::Unclear);
    out.explanation = text_field(answer, "explanation");
    if out.explanation.is_empty() {
        out.explanation = "the model gave no explanation".into();
    }
    if Part::parse(&part_text).is_none() && !part_text.is_empty() {
        out.explanation.push_str(&format!(
            " (the model named the part `{part_text}`, which is not one of the choices)"
        ));
    }
    out.confidence = match text_field(answer, "confidence")
        .to_ascii_lowercase()
        .as_str()
    {
        c @ ("high" | "medium" | "low") => c.to_string(),
        _ => "unstated".into(),
    };
    for q in answer
        .get("evidence")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .take(5)
    {
        let quote = text_field(q, "quote");
        if quote.is_empty() {
            continue;
        }
        let file = text_field(q, "file");
        let wanted = squash(&quote);
        let found = !wanted.is_empty() && pieces.iter().any(|p| squash(&p.text).contains(&wanted));
        out.evidence.push(Quote { file, quote, found });
    }
}

/// Analyze up to `max_trials` of the job's failed trials, a few at a time.
pub async fn analyze_job(
    reviewer: &Reviewer,
    job: &Path,
    max_trials: usize,
) -> Result<JobAnalysis, ReviewError> {
    let failed = failed_trials(job);
    let total = failed.len();
    let (now, later) = failed.split_at(total.min(max_trials));
    let trials: Vec<TrialAnalysis> = futures::stream::iter(now)
        .map(|t| analyze_trial(reviewer, t))
        .buffered(CONCURRENCY)
        .collect()
        .await;
    // Every call failed the same way: say so once rather than per trial.
    if !trials.is_empty()
        && trials.iter().all(|t| t.error.is_some())
        && trials.iter().all(|t| t.spend.calls == 0)
    {
        return Err(ReviewError {
            kind: super::ReviewErrorKind::Provider,
            message: trials[0].error.clone().unwrap_or_default(),
            next: "nothing was analyzed; fix the model access, then run again".into(),
        });
    }
    let mut spend = Spend::default();
    for t in &trials {
        spend.add(&t.spend);
    }
    Ok(JobAnalysis {
        job: job.to_path_buf(),
        model: reviewer.model.clone(),
        analyzed_at: now_iso(),
        opinion: OPINION,
        failed: total,
        by_cause: group(&trials),
        trials,
        not_analyzed: later
            .iter()
            .map(|t| t.record.rollout_name.clone())
            .collect(),
        spend,
    })
}

/// Trials grouped by the part the model blamed, biggest group first; trials
/// the model could not analyze are left out.
pub fn group(trials: &[TrialAnalysis]) -> Vec<CauseGroup> {
    let mut groups: Vec<CauseGroup> = Vec::new();
    for t in trials.iter().filter(|t| t.error.is_none()) {
        match groups.iter_mut().find(|g| g.part == t.part) {
            Some(g) => g.trials.push(t.trial.clone()),
            None => groups.push(CauseGroup {
                part: t.part,
                label: t.part.label(),
                trials: vec![t.trial.clone()],
            }),
        }
    }
    groups.sort_by(|a, b| {
        b.trials
            .len()
            .cmp(&a.trials.len())
            .then(a.part.cmp(&b.part))
    });
    groups
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn parts_accept_near_names_and_quotes_are_checked() {
        assert_eq!(Part::parse("The model provider"), Some(Part::Provider));
        assert_eq!(Part::parse("AGENT"), Some(Part::Agent));
        assert_eq!(Part::parse("weather"), None);
        let pieces = vec![Piece {
            label: "agent/stderr.txt".into(),
            text: "line one\nException: [410]   Gone\n".into(),
        }];
        let mut out = TrialAnalysis {
            trial: "t".into(),
            task: "x".into(),
            dir: PathBuf::new(),
            recorded: Recorded {
                status: Status::Errored,
                reward: None,
                code: None,
                owner: None,
                message: None,
            },
            part: Part::Unclear,
            explanation: String::new(),
            evidence: Vec::new(),
            confidence: String::new(),
            error: None,
            spend: Spend::default(),
        };
        read_answer(
            &json!({"part": "weather", "explanation": "", "confidence": "very",
                    "evidence": [{"file": "agent/stderr.txt", "quote": "Exception: [410] Gone"},
                                 {"file": "x", "quote": "made up"}, {"quote": ""}]}),
            &pieces,
            &mut out,
        );
        assert_eq!(out.part, Part::Unclear);
        assert!(out.explanation.contains("`weather`"));
        assert_eq!(out.confidence, "unstated");
        assert_eq!(out.evidence.len(), 2);
        assert!(out.evidence[0].found);
        assert!(!out.evidence[1].found);
    }

    #[test]
    fn the_trajectory_tail_keeps_the_newest_events() {
        let mut text = String::new();
        for i in 0..500 {
            text.push_str(&format!(
                "{{\"type\": \"agent_message\", \"text\": \"step {i} {}\"}}\n",
                "z".repeat(100)
            ));
        }
        text.push_str("{\"type\": \"tool_call\", \"title\": \"ls /app\", \"status\": \"failed\", \"raw_output\": {\"exit_code\": 2}}\n");
        let (tail, total) = trajectory_tail(&text);
        assert_eq!(total, 501);
        assert!(tail.len() <= TRAJECTORY_CAP + 200);
        assert!(tail.ends_with("#501 tool_call: [failed] ls /app → {\"exit_code\":2}"));
        assert!(tail.contains("not sent"));
        assert!(!tail.contains("step 0 "));
    }
}
