//! The compare, outcomes, and Pareto views: what each trial's `result.json`
//! and `timing.json` say about how it ended, summed up per job.
//!
//! These views are new in the Rust viewer (0.8 had only the run list and
//! the run page). They read the same files the run list reads and accept
//! both layouts: the Rust engine's `result.json` has `status`, `passed`,
//! and a `problem` with `code`, `owner`, `message`, and `next`; a 0.8
//! `result.json` has `error` and `verifier_error` with their categories,
//! and the outcome is worked out from those and the reward.
//!
//! A job is the folder a trial sits in: a trial's id without its last part.
//! Trials directly under the served folder belong to the job `.`.

use std::collections::{BTreeMap, BTreeSet};

use crate::catalog::Run;
use crate::classify::{classify_error, classify_verifier_error};
use crate::lineage::load_lineage;
use crate::metadata::normalize_metadata;
use crate::py;
use crate::pyjson::{Int, Map, Value, get, object};
use crate::stats::{mean_with_t_interval, quantile, wilson};
use crate::values::{
    error_text, finite_float, load_json, load_object, optional_bool, optional_text,
};

/// The phases whose times the outcomes view reports, in `timing.json`'s names.
pub const PHASES: [&str; 3] = ["environment_setup", "agent_execution", "verifier"];

/// The job of trials that sit directly in the served folder.
const ROOT_JOB: &str = ".";

/// Messages longer than this are cut (they are shown in a table cell).
const MESSAGE_CHARS: usize = 400;

/// With this many columns or fewer, every pair is compared; with more, the
/// first column is compared with each of the others.
const ALL_PAIRS_UP_TO: usize = 5;

/// How a trial ended, in the four words the views use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Outcome {
    /// Scored, and passed.
    Passed,
    /// Scored, and did not pass.
    Failed,
    /// The agent ran but no usable verdict was recorded.
    Unscored,
    /// The trial could not run to the end (or was skipped).
    Errored,
    /// The user stopped the run while the trial ran (R001): listed apart,
    /// and left out of pass rates, as `bench results` does.
    Stopped,
}

impl Outcome {
    pub const ALL: [Outcome; 5] = [
        Outcome::Passed,
        Outcome::Failed,
        Outcome::Unscored,
        Outcome::Errored,
        Outcome::Stopped,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Outcome::Passed => "passed",
            Outcome::Failed => "failed",
            Outcome::Unscored => "unscored",
            Outcome::Errored => "errored",
            Outcome::Stopped => "stopped",
        }
    }

    pub fn parse(text: &str) -> Option<Outcome> {
        Outcome::ALL
            .into_iter()
            .find(|outcome| outcome.as_str() == text)
    }
}

/// Why a trial did not end well: the engine's `problem`, or a 0.8 error.
#[derive(Debug, Clone, PartialEq)]
pub struct Problem {
    /// The engine's code (`R304`), or a 0.8 category (`agent_timeout`).
    pub code: String,
    /// The engine's owner (`task`, `agent`, ...); 0.8 does not record one.
    pub owner: Option<String>,
    pub message: String,
    pub next: Option<String>,
    /// Where it came from: `problem`, `error` (0.8), `verifier_error` (0.8),
    /// or `result.json` when that file is missing or unreadable.
    pub source: &'static str,
}

/// One trial as the views see it.
#[derive(Debug, Clone)]
pub struct Trial {
    /// The run-list id; `/?run=<id>` opens it.
    pub id: String,
    pub job: String,
    pub task: String,
    /// The agent's name, or empty text when none is recorded.
    pub agent: String,
    pub model: Option<String>,
    pub outcome: Outcome,
    pub reward: Option<f64>,
    pub problem: Option<Problem>,
    /// Seconds per [`PHASES`] entry.
    pub phases: [Option<f64>; 3],
    /// The trial's whole time in seconds (`timing.json`'s `total`).
    pub seconds: Option<f64>,
    /// What the agent's model calls cost, when the trial recorded it
    /// (`agent_result.cost_usd`).
    pub cost_usd: Option<f64>,
    /// Input plus output tokens, when the trial recorded usage
    /// (`agent_result.total_tokens`, with a `usage_source` other than
    /// `unavailable`).
    pub tokens: Option<f64>,
}

impl Trial {
    /// The agent and model, as a column label: `claude-agent-acp · opus`.
    fn arm(&self) -> String {
        [Some(self.agent.as_str()), self.model.as_deref()]
            .into_iter()
            .flatten()
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join(" \u{b7} ")
    }

    fn link(&self) -> Value {
        object([
            ("id", self.id.clone().into()),
            ("task", self.task.clone().into()),
            ("outcome", self.outcome.as_str().into()),
            ("reward", self.reward.into()),
        ])
    }
}

/// The job a run-list id belongs to.
pub fn job_of(id: &str) -> &str {
    id.rsplit_once('/').map_or(ROOT_JOB, |(job, _)| job)
}

/// The file a running `bench run` holds locked in its job folder.
const RUNNING_FILE: &str = ".bench-run.lock";

/// A `bench branch` parent discarded at its checkpoint (`skipped`, R000):
/// unscored by design, so the trial views leave it out, as `bench results`
/// does; its children are the fork's measure.
pub fn discarded_parent(run: &Run) -> bool {
    crate::catalog::is_branch_parent(&run.dir)
        && get(&load_object(&run.dir.join("result.json")), "status").as_str() == Some("skipped")
}

/// Whether a `bench run` is still running this trial: it has no
/// `result.json` yet, and a live process holds its job folder's
/// `.bench-run.lock`. The views leave such trials out, as `bench results`
/// does, instead of counting them as errored; a trial without a result in a
/// job no run holds stopped without one, and stays errored.
pub fn in_progress(run: &Run) -> bool {
    if run.dir.join("result.json").exists() {
        return false;
    }
    let Some(job) = run.dir.parent() else {
        return false;
    };
    let Ok(file) = std::fs::File::open(job.join(RUNNING_FILE)) else {
        return false;
    };
    match file.try_lock_shared() {
        Ok(()) => {
            let _ = file.unlock();
            false
        }
        Err(std::fs::TryLockError::WouldBlock) => true,
        Err(std::fs::TryLockError::Error(_)) => false,
    }
}

/// A planned trial that never ran: its agent and model as `lock.json`
/// names them (`None` when the lock does not say).
#[derive(Debug, Clone)]
pub struct Planned {
    pub agent: Option<String>,
    pub model: Option<String>,
}

/// Per job (by id), the trials its `lock.json` planned that have no folder
/// with a result: never started (a drained or stopped run, or a login that
/// ran out). A job a live run holds is left out: its trials may still
/// start. Only jobs with a listed trial are counted.
pub fn never_ran_by_job(runs: &[Run]) -> BTreeMap<String, Vec<Planned>> {
    let mut jobs: BTreeMap<String, std::path::PathBuf> = BTreeMap::new();
    for run in runs {
        if let Some(dir) = run.dir.parent() {
            jobs.entry(job_of(&run.id).to_string())
                .or_insert_with(|| dir.to_path_buf());
        }
    }
    let mut out = BTreeMap::new();
    for (job, dir) in jobs {
        let running = std::fs::File::open(dir.join(RUNNING_FILE))
            .map(|f| matches!(f.try_lock_shared(), Err(std::fs::TryLockError::WouldBlock)))
            .unwrap_or(false);
        if running {
            continue;
        }
        let Some(lock) = load_json(&dir.join("lock.json")) else {
            continue;
        };
        let planned: Vec<Planned> = lock
            .get("trials")
            .as_array()
            .map(|trials| {
                trials
                    .iter()
                    .filter(|t| {
                        t.get("trial").as_str().is_some_and(|name| {
                            !dir.join(name).join("result.json").exists() && !dir.join(name).is_dir()
                        })
                    })
                    .map(|t| {
                        let text = |key: &str| {
                            t.get("attempt")
                                .get(key)
                                .as_str()
                                .filter(|v| !v.is_empty())
                                .map(str::to_string)
                        };
                        Planned {
                            agent: text("agent"),
                            model: text("model"),
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();
        if !planned.is_empty() {
            out.insert(job, planned);
        }
    }
    out
}

/// Whether a planned trial belongs to a `/compare` column of its job: the
/// same agent and model (the column's `arm`); else, when the lock names no
/// model or another spelling of it, the same agent; a lock that names no
/// agent, or an agent with no column in its job (none of its trials
/// started), counts toward every column of its job, so the job's columns
/// are not shown whole.
fn planned_in_column(planned: &Planned, arm: &str, arms: &[String]) -> bool {
    let Some(agent) = &planned.agent else {
        return true;
    };
    let exact = [Some(agent.as_str()), planned.model.as_deref()]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join(" \u{b7} ");
    let agent_of = |arm: &str| arm.split(" \u{b7} ").next().unwrap_or_default().to_string();
    if arms.contains(&exact) {
        arm == exact
    } else if arms.iter().any(|a| agent_of(a) == *agent) {
        agent_of(arm) == *agent
    } else {
        true
    }
}

/// Add the never-ran counts to a view: each `/compare` column and
/// `/pareto` point gets `never_ran` (the planned trials of its own agent
/// and model) and `partial` (stopped or never-ran trials, so its rate is
/// not the whole job's); each `/outcomes` job row gets the job's
/// `never_ran` and `partial`; and the view gets `never_ran` by job.
pub fn mark_never_ran(view: &mut Value, never_ran: &BTreeMap<String, Vec<Planned>>) {
    let Value::Object(map) = view else {
        return;
    };
    for key in ["columns", "points"] {
        let Some(Value::Array(columns)) = map.get_mut(key) else {
            continue;
        };
        let arms_of = |job: &str, columns: &[Value]| -> Vec<String> {
            columns
                .iter()
                .filter(|c| c.get("job").as_str() == Some(job))
                .filter_map(|c| c.get("arm").as_str().map(str::to_string))
                .collect()
        };
        let arms: Vec<Vec<String>> = columns
            .iter()
            .map(|c| arms_of(c.get("job").as_str().unwrap_or_default(), columns))
            .collect();
        for (column, arms) in columns.iter_mut().zip(arms) {
            let Value::Object(c) = column else { continue };
            let job = get(c, "job").as_str().unwrap_or_default().to_string();
            let arm = get(c, "arm").as_str().unwrap_or_default().to_string();
            let n = never_ran.get(&job).map_or(0, |planned| {
                planned
                    .iter()
                    .filter(|p| planned_in_column(p, &arm, &arms))
                    .count()
            });
            // `counts` (a column's) carries `stopped` only when some trial
            // was; a point counts them in `stopped`.
            let stopped = !matches!(get(c, "counts").get("stopped"), Value::Null)
                || !matches!(get(c, "stopped"), Value::Null | Value::Int(Int::Small(0)));
            c.insert("never_ran".into(), n.into());
            c.insert("partial".into(), (n > 0 || stopped).into());
        }
    }
    if let Some(Value::Array(summary)) = map.get_mut("summary") {
        for row in summary.iter_mut() {
            let Value::Object(r) = row else { continue };
            let job = get(r, "job").as_str().unwrap_or_default().to_string();
            let n = never_ran.get(&job).map_or(0, Vec::len);
            let stopped = !matches!(get(r, "counts").get("stopped"), Value::Null);
            r.insert("never_ran".into(), n.into());
            r.insert("partial".into(), (n > 0 || stopped).into());
        }
    }
    map.insert(
        "never_ran".into(),
        Value::Object(
            never_ran
                .iter()
                .map(|(job, planned)| (job.clone(), planned.len().into()))
                .collect(),
        ),
    );
}

/// Read one trial's `result.json` and timing.
pub fn read_trial(run: &Run) -> Trial {
    let result = load_object(&run.dir.join("result.json"));
    let preferred =
        load_json(&run.dir.join("timing.json")).filter(|timing| timing.as_object().is_some());
    let timing = preferred
        .as_ref()
        .or_else(|| Some(get(&result, "timing")).filter(|timing| timing.as_object().is_some()));
    let metadata = normalize_metadata(&result, timing);
    let phases = PHASES.map(|phase| {
        metadata
            .timing
            .as_ref()
            .and_then(|timing| match timing.values.get(phase) {
                Some(Value::Float(seconds)) if *seconds >= 0.0 => Some(*seconds),
                _ => None,
            })
    });
    let usage = &metadata.usage;
    let recorded = optional_text(get(&usage.values, "usage_source"))
        .is_some_and(|source| !source.is_empty() && source != "unavailable");
    let tokens = match &usage.total_tokens {
        Some(crate::pyjson::Int::Small(n)) if recorded && *n >= 0 => Some(*n as f64),
        _ => None,
    };
    Trial {
        id: run.id.clone(),
        job: job_of(&run.id).to_string(),
        task: metadata
            .task_name
            .clone()
            .filter(|task| !task.is_empty())
            .unwrap_or_else(|| py::name_of(&run.dir)),
        agent: metadata.agent_name.clone().unwrap_or_default(),
        model: metadata.model.clone().filter(|model| !model.is_empty()),
        outcome: outcome_of(&result, metadata.reward),
        reward: metadata.reward,
        problem: problem_of(&result),
        phases,
        seconds: metadata
            .timing
            .as_ref()
            .and_then(|timing| timing.total)
            .filter(|seconds| *seconds >= 0.0),
        cost_usd: usage.cost_usd.filter(|cost| *cost >= 0.0),
        tokens,
    }
}

/// The engine's `status` when there is one; else (0.8) a reward means
/// scored, an `error` means errored, and anything else is unscored.
fn outcome_of(result: &Map, reward: Option<f64>) -> Outcome {
    let passed = optional_bool(get(result, "passed"));
    let scored = || {
        if passed.unwrap_or_else(|| reward.is_some_and(|reward| reward >= 1.0)) {
            Outcome::Passed
        } else {
            Outcome::Failed
        }
    };
    match get(result, "status").as_str() {
        Some("scored") => scored(),
        Some("unscored") => Outcome::Unscored,
        Some("skipped") if get(result, "problem").get("code").as_str() == Some("R001") => {
            Outcome::Stopped
        }
        Some("errored" | "skipped") => Outcome::Errored,
        _ if result.is_empty() => Outcome::Errored,
        _ if reward.is_some() => scored(),
        _ if error_text(get(result, "error")).is_some() => Outcome::Errored,
        _ => Outcome::Unscored,
    }
}

/// The engine's `problem`, else a 0.8 `error` or `verifier_error`.
fn problem_of(result: &Map) -> Option<Problem> {
    if result.is_empty() {
        return Some(Problem {
            code: "no result".into(),
            owner: None,
            message: "result.json is missing or unreadable".into(),
            next: None,
            source: "result.json",
        });
    }
    if let Some(problem) = get(result, "problem").as_object() {
        let text = |key: &str| optional_text(get(problem, key)).filter(|text| !text.is_empty());
        return Some(Problem {
            code: text("code").unwrap_or_else(|| "(no code)".into()),
            owner: text("owner"),
            message: clip(&text("message").unwrap_or_default()),
            next: text("next"),
            source: "problem",
        });
    }
    let category = |key: &str| optional_text(get(result, key)).filter(|text| !text.is_empty());
    if let Some(error) = error_text(get(result, "error")).filter(|text| !text.is_empty()) {
        let code = category("error_category")
            .or_else(|| classify_error(&error).map(str::to_string))
            .unwrap_or_else(|| "error".into());
        return Some(Problem {
            code,
            owner: None,
            message: clip(&error),
            next: None,
            source: "error",
        });
    }
    if let Some(error) = error_text(get(result, "verifier_error")).filter(|text| !text.is_empty()) {
        let code = category("verifier_error_category")
            .or_else(|| classify_verifier_error(&error).map(str::to_string))
            .unwrap_or_else(|| "verifier_error".into());
        return Some(Problem {
            code,
            owner: None,
            message: clip(&error),
            next: None,
            source: "verifier_error",
        });
    }
    None
}

/// The first line of `text`, at most [`MESSAGE_CHARS`] characters.
fn clip(text: &str) -> String {
    let line = text.trim().lines().next().unwrap_or("");
    if line.chars().count() > MESSAGE_CHARS {
        let mut cut: String = line.chars().take(MESSAGE_CHARS).collect();
        cut.push('\u{2026}');
        cut
    } else {
        line.to_string()
    }
}

/// What the URL asks for: jobs (in order) and filters.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ViewQuery {
    /// Job ids; none means every job.
    pub jobs: Vec<String>,
    /// Outcomes to keep; none means all.
    pub status: BTreeSet<Outcome>,
    /// Keep tasks whose name contains this (case-insensitive).
    pub task: String,
    /// Keep trials of this agent.
    pub agent: String,
}

impl ViewQuery {
    fn keeps(&self, trial: &Trial) -> bool {
        (self.task.is_empty()
            || trial
                .task
                .to_lowercase()
                .contains(&self.task.to_lowercase()))
            && (self.agent.is_empty() || trial.agent == self.agent)
    }

    fn echo(&self, selected: &[String]) -> Value {
        object([
            ("runs", selected.to_vec().into()),
            (
                "status",
                self.status
                    .iter()
                    .map(|outcome| outcome.as_str().to_string())
                    .collect::<Vec<_>>()
                    .into(),
            ),
            ("task", self.task.clone().into()),
            ("agent", self.agent.clone().into()),
        ])
    }
}

/// Everything the two views share: the jobs on offer, which of them the
/// query selects, and the agents among those.
struct Selection<'a> {
    jobs: Vec<(String, usize)>,
    selected: Vec<String>,
    unknown: Vec<String>,
    agents: Vec<String>,
    /// Trials of the selected jobs that pass the task and agent filters.
    trials: Vec<&'a Trial>,
}

fn select<'a>(trials: &'a [Trial], query: &ViewQuery) -> Selection<'a> {
    let mut jobs: Vec<(String, usize)> = Vec::new();
    for trial in trials {
        match jobs.iter_mut().find(|(job, _)| *job == trial.job) {
            Some((_, count)) => *count += 1,
            None => jobs.push((trial.job.clone(), 1)),
        }
    }
    let mut selected = Vec::new();
    let mut unknown = Vec::new();
    for asked in &query.jobs {
        match match_job(&jobs, asked) {
            Some(job) => {
                if !selected.contains(&job) {
                    selected.push(job);
                }
            }
            None if !unknown.contains(asked) => unknown.push(asked.clone()),
            None => {}
        }
    }
    if query.jobs.is_empty() {
        selected = jobs.iter().map(|(job, _)| job.clone()).collect();
    }
    let in_selection = |trial: &&Trial| selected.contains(&trial.job);
    let agents: BTreeSet<String> = trials
        .iter()
        .filter(in_selection)
        .map(|trial| trial.agent.clone())
        .filter(|agent| !agent.is_empty())
        .collect();
    let kept = trials
        .iter()
        .filter(in_selection)
        .filter(|trial| query.keeps(trial))
        .collect();
    Selection {
        jobs,
        selected,
        unknown,
        agents: agents.into_iter().collect(),
        trials: kept,
    }
}

/// Move each job the query named that holds only discarded branch parents
/// (`set_aside`, the jobs of the trials the view left out) from `unknown` to
/// `set_aside`, as `{"id", "name"}` (the root job by the served folder's
/// name, as the other lists name it): the server serves it, so the page says
/// why it has no trial here instead of calling it not found. The rest of
/// `unknown` stays.
pub fn mark_set_aside(view: &mut Value, set_aside: &[String]) {
    let Value::Object(map) = view else {
        return;
    };
    let root_name = get(map, "root").as_str().unwrap_or(ROOT_JOB).to_string();
    let jobs: Vec<(String, usize)> = set_aside.iter().map(|job| (job.clone(), 0)).collect();
    let mut found: Vec<String> = Vec::new();
    if let Some(Value::Array(unknown)) = map.get_mut("unknown") {
        unknown.retain(|asked| {
            let Some(job) = asked.as_str().and_then(|asked| match_job(&jobs, asked)) else {
                return true;
            };
            if !found.contains(&job) {
                found.push(job);
            }
            false
        });
    }
    let found = found
        .into_iter()
        .map(|job| {
            let name = job_name(&job, &root_name);
            object([("id", job.into()), ("name", name.into())])
        })
        .collect();
    map.insert("set_aside".into(), Value::Array(found));
}

/// The job a `runs=` entry names. Ids are relative to the folder served
/// (`a` when serving `jobs`), but a path as typed in the shell works too
/// (`jobs/a`, `./jobs/a/`): the one job whose id ends the path, for a
/// relative path without `..`.
fn match_job(jobs: &[(String, usize)], asked: &str) -> Option<String> {
    let asked = asked.trim_end_matches('/');
    if let Some((job, _)) = jobs.iter().find(|(known, _)| known == asked) {
        return Some(job.clone());
    }
    // Only a plain relative path: `..` or an absolute path never names a
    // job here, even one that ends the same way.
    if asked.starts_with('/') || asked.split('/').any(|part| part == "..") {
        return None;
    }
    let mut found = jobs
        .iter()
        .filter(|(known, _)| known != ROOT_JOB && asked.ends_with(&format!("/{known}")));
    match (found.next(), found.next()) {
        (Some((job, _)), None) => Some(job.clone()),
        _ => None,
    }
}

fn job_name(job: &str, root_name: &str) -> String {
    if job == ROOT_JOB {
        root_name.to_string()
    } else {
        job.to_string()
    }
}

fn header(
    view: &str,
    root_name: &str,
    capped: bool,
    selection: &Selection<'_>,
    query: &ViewQuery,
) -> Map {
    let mut map = Map::new();
    map.insert("mode".into(), view.into());
    map.insert("root".into(), root_name.into());
    map.insert("capped".into(), capped.into());
    map.insert(
        "jobs".into(),
        Value::Array(
            selection
                .jobs
                .iter()
                .map(|(job, count)| {
                    object([
                        ("id", job.clone().into()),
                        ("name", job_name(job, root_name).into()),
                        ("trials", (*count).into()),
                    ])
                })
                .collect(),
        ),
    );
    map.insert("query".into(), query.echo(&selection.selected));
    map.insert("unknown".into(), selection.unknown.clone().into());
    map.insert("agents".into(), selection.agents.clone().into());
    map
}

fn interval(pair: Option<(f64, f64)>) -> Value {
    pair.map_or(Value::Null, |(low, high)| {
        Value::Array(vec![low.into(), high.into()])
    })
}

fn mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

/// Counts of each outcome.
fn counts<'a>(trials: impl IntoIterator<Item = &'a Trial>) -> BTreeMap<Outcome, usize> {
    let mut counts: BTreeMap<Outcome, usize> = Outcome::ALL.iter().map(|o| (*o, 0)).collect();
    for trial in trials {
        *counts.entry(trial.outcome).or_default() += 1;
    }
    counts
}

/// The counts as JSON; `stopped` only when some trial was.
fn counts_value(counts: &BTreeMap<Outcome, usize>) -> Value {
    Value::Object(
        counts
            .iter()
            .filter(|(outcome, count)| **outcome != Outcome::Stopped || **count > 0)
            .map(|(outcome, count)| (outcome.as_str().to_string(), (*count).into()))
            .collect(),
    )
}

/// `/compare`: tasks by job (one column per job, or per agent and model
/// when a job ran several), each job's pass rate with its 95% Wilson
/// interval, and paired differences over shared tasks. A pair's `excluded`
/// counts tasks both sides ran that one side did not score; `one_side`
/// counts tasks only one side ran.
///
/// The task and agent filters choose trials; the status filter chooses
/// rows, keeping a task when any of its trials has one of the outcomes.
/// Every number is computed over the rows shown.
pub fn compare(trials: &[Trial], root_name: &str, capped: bool, query: &ViewQuery) -> Value {
    let selection = select(trials, query);

    // Columns: per job, one per agent and model found in it.
    let mut columns: Vec<(String, String)> = Vec::new();
    for job in &selection.selected {
        let arms: BTreeSet<String> = selection
            .trials
            .iter()
            .filter(|trial| trial.job == *job)
            .map(|trial| trial.arm())
            .collect();
        if arms.is_empty() {
            columns.push((job.clone(), String::new()));
        }
        columns.extend(arms.into_iter().map(|arm| (job.clone(), arm)));
    }
    let column_of = |trial: &Trial| {
        columns
            .iter()
            .position(|(job, arm)| *job == trial.job && *arm == trial.arm())
    };

    // Rows: one per task, its trials per column.
    let mut tasks: BTreeMap<&str, Vec<Vec<&Trial>>> = BTreeMap::new();
    for trial in &selection.trials {
        if let Some(column) = column_of(trial) {
            tasks
                .entry(trial.task.as_str())
                .or_insert_with(|| vec![Vec::new(); columns.len()])[column]
                .push(trial);
        }
    }
    let mut rows: Vec<(&str, Vec<Vec<&Trial>>)> = tasks
        .into_iter()
        .filter(|(_, cells)| {
            query.status.is_empty()
                || cells
                    .iter()
                    .flatten()
                    .any(|trial| query.status.contains(&trial.outcome))
        })
        .collect();
    for (_, cells) in &mut rows {
        for cell in cells.iter_mut() {
            cell.sort_by(|left, right| left.id.cmp(&right.id));
        }
    }
    // A cell's value is the mean reward of its scored trials.
    let cell_mean = |cell: &[&Trial]| {
        mean(
            &cell
                .iter()
                .filter(|trial| matches!(trial.outcome, Outcome::Passed | Outcome::Failed))
                .filter_map(|trial| trial.reward)
                .collect::<Vec<_>>(),
        )
    };
    let disagreement = |cells: &[Vec<&Trial>]| {
        let means: Vec<f64> = cells.iter().filter_map(|cell| cell_mean(cell)).collect();
        let spread = match (
            means.iter().copied().reduce(f64::max),
            means.iter().copied().reduce(f64::min),
        ) {
            (Some(high), Some(low)) => high - low,
            _ => 0.0,
        };
        let outcomes: BTreeSet<Outcome> =
            cells.iter().flatten().map(|trial| trial.outcome).collect();
        (spread, outcomes.len() > 1)
    };
    rows.sort_by(|left, right| {
        let (left_spread, left_mixed) = disagreement(&left.1);
        let (right_spread, right_mixed) = disagreement(&right.1);
        right_spread
            .total_cmp(&left_spread)
            .then(right_mixed.cmp(&left_mixed))
            .then(left.0.cmp(right.0))
    });

    let column_values: Vec<Value> = columns
        .iter()
        .enumerate()
        .map(|(index, (job, arm))| {
            let members: Vec<&Trial> = rows
                .iter()
                .flat_map(|(_, cells)| cells[index].iter().copied())
                .collect();
            let counts = counts(members.iter().copied());
            let passed = counts[&Outcome::Passed];
            // Stopped trials are counted apart, not in the pass rate.
            let total = members.len() - counts[&Outcome::Stopped];
            let rewards: Vec<f64> = members.iter().filter_map(|trial| trial.reward).collect();
            object([
                ("job", job.clone().into()),
                ("job_name", job_name(job, root_name).into()),
                ("arm", arm.clone().into()),
                ("trials", total.into()),
                (
                    "tasks",
                    rows.iter()
                        .filter(|(_, cells)| !cells[index].is_empty())
                        .count()
                        .into(),
                ),
                ("counts", counts_value(&counts)),
                (
                    "pass_rate",
                    (total > 0).then(|| passed as f64 / total as f64).into(),
                ),
                ("pass_interval", interval(wilson(passed, total))),
                ("mean_reward", mean(&rewards).into()),
            ])
        })
        .collect();

    let pairs: Vec<(usize, usize)> = if columns.len() <= ALL_PAIRS_UP_TO {
        (0..columns.len())
            .flat_map(|a| (a + 1..columns.len()).map(move |b| (a, b)))
            .collect()
    } else {
        (1..columns.len()).map(|b| (0, b)).collect()
    };
    let pair_values: Vec<Value> = pairs
        .into_iter()
        .map(|(a, b)| {
            let mut differences = Vec::new();
            let (mut better, mut worse, mut tied, mut excluded) = (0usize, 0usize, 0usize, 0usize);
            // Tasks only one side ran are not shared; they are counted apart.
            let mut one_side = 0usize;
            // Rewards are usually 0 to 1; wider ones widen the possible range.
            let (mut lowest, mut highest) = (0.0_f64, 1.0_f64);
            for (_, cells) in &rows {
                match (cells[a].is_empty(), cells[b].is_empty()) {
                    (true, true) => continue,
                    (true, false) | (false, true) => {
                        one_side += 1;
                        continue;
                    }
                    (false, false) => {}
                }
                match (cell_mean(&cells[a]), cell_mean(&cells[b])) {
                    (Some(left), Some(right)) => {
                        let difference = left - right;
                        if difference.abs() < 1e-12 {
                            tied += 1;
                        } else if difference > 0.0 {
                            better += 1;
                        } else {
                            worse += 1;
                        }
                        differences.push(difference);
                        for reward in [left, right] {
                            lowest = lowest.min(reward);
                            highest = highest.max(reward);
                        }
                    }
                    _ => excluded += 1,
                }
            }
            let paired = mean_with_t_interval(&differences, highest - lowest);
            object([
                ("a", a.into()),
                ("b", b.into()),
                ("shared", differences.len().into()),
                (
                    "mean_difference",
                    paired.as_ref().map(|paired| paired.mean).into(),
                ),
                (
                    "interval",
                    interval(paired.as_ref().and_then(|paired| paired.interval)),
                ),
                (
                    "no_interval",
                    paired.and_then(|paired| paired.no_interval).into(),
                ),
                ("better", better.into()),
                ("worse", worse.into()),
                ("tied", tied.into()),
                ("excluded", excluded.into()),
                ("one_side", one_side.into()),
            ])
        })
        .collect();

    let row_values: Vec<Value> = rows
        .iter()
        .map(|(task, cells)| {
            let (spread, mixed) = disagreement(cells);
            object([
                ("task", (*task).into()),
                ("spread", spread.into()),
                ("mixed", mixed.into()),
                (
                    "cells",
                    Value::Array(
                        cells
                            .iter()
                            .map(|cell| {
                                Value::Array(cell.iter().map(|trial| trial.link()).collect())
                            })
                            .collect(),
                    ),
                ),
            ])
        })
        .collect();

    let mut map = header("compare", root_name, capped, &selection, query);
    map.insert("columns".into(), Value::Array(column_values));
    map.insert("rows".into(), Value::Array(row_values));
    map.insert("pairs".into(), Value::Array(pair_values));
    Value::Object(map)
}

/// `/branches`: every fork in the selected jobs (`bench branch`): where it
/// forked, its status and value, the checkpoint, what became of the parent,
/// and each child with its label, status, reward, and advantage (reward
/// minus the fork's value), with the ref that opens the child's trajectory
/// from its parent's run page. Read through [`load_lineage`], so a
/// malformed `tree.json` gives a fork with an error, never a broken page.
/// The task and agent filters choose parents; the status filter is unused.
pub fn branches(
    trials: &[Trial],
    runs: &[Run],
    root_name: &str,
    capped: bool,
    query: &ViewQuery,
) -> Value {
    let selection = select(trials, query);
    let mut forks: Vec<Value> = Vec::new();
    let (mut children, mut scored) = (0usize, 0usize);
    for trial in &selection.trials {
        let Some(run) = runs.iter().find(|run| run.id == trial.id) else {
            continue;
        };
        let Some(lineage) = load_lineage(&run.dir) else {
            continue;
        };
        let tree = load_object(&run.dir.join("tree.json"));
        let raw: Vec<Value> = get(&tree, "forks")
            .as_array()
            .map(<[Value]>::to_vec)
            .unwrap_or_default();
        if let Some(error) = &lineage.error {
            forks.push(object([
                ("run", trial.id.clone().into()),
                ("job", job_name(&trial.job, root_name).into()),
                ("task", trial.task.clone().into()),
                ("error", error.clone().into()),
            ]));
            continue;
        }
        for fork in &lineage.forks {
            let extra = raw
                .iter()
                .find(|f| f.get("id").as_str() == Some(fork.id.as_str()))
                .cloned()
                .unwrap_or(Value::Null);
            let snapshot = extra.get("snapshot");
            let rows: Vec<Value> = fork
                .children
                .iter()
                .map(|child| {
                    children += 1;
                    if child.status.as_deref() == Some("scored") {
                        scored += 1;
                    }
                    let advantage = match (child.reward, fork.value) {
                        (Some(r), Some(v)) => Value::from(((r - v) * 1e6).round() / 1e6),
                        _ => Value::Null,
                    };
                    object([
                        (
                            "node_id",
                            child
                                .node_id
                                .clone()
                                .map(Value::from)
                                .unwrap_or(Value::Null),
                        ),
                        (
                            "label",
                            child.label().map(Value::from).unwrap_or(Value::Null),
                        ),
                        // 0.8 records a request for every child; a child's
                        // own prompt reads `own prompt (N characters, …)`.
                        (
                            "own_prompt",
                            child.intervention[1]
                                .1
                                .as_deref()
                                .is_some_and(|r| r.starts_with("own prompt"))
                                .into(),
                        ),
                        (
                            "status",
                            child.status.clone().map(Value::from).unwrap_or(Value::Null),
                        ),
                        (
                            "reward",
                            child.reward.map(Value::from).unwrap_or(Value::Null),
                        ),
                        ("advantage", advantage),
                        (
                            "error",
                            child.error.clone().map(Value::from).unwrap_or(Value::Null),
                        ),
                        (
                            "ref",
                            child
                                .child_ref
                                .clone()
                                .map(Value::from)
                                .unwrap_or(Value::Null),
                        ),
                    ])
                })
                .collect();
            let text = |v: &Value| optional_text(v).map(Value::from).unwrap_or(Value::Null);
            let number = |v: &Value| finite_float(v).map(Value::from).unwrap_or(Value::Null);
            forks.push(object([
                ("run", trial.id.clone().into()),
                ("job", job_name(&trial.job, root_name).into()),
                ("task", trial.task.clone().into()),
                ("arm", trial.arm().into()),
                ("fork_id", fork.id.clone().into()),
                ("fork_point", text(extra.get("fork_point"))),
                (
                    "status",
                    fork.status.clone().map(Value::from).unwrap_or(Value::Null),
                ),
                ("value", fork.value.map(Value::from).unwrap_or(Value::Null)),
                ("value_stderr", number(extra.get("value_stderr"))),
                (
                    "parent_restore",
                    fork.parent_restore
                        .clone()
                        .map(Value::from)
                        .unwrap_or(Value::Null),
                ),
                ("parent_reward", number(extra.get("parent_reward"))),
                ("agent_session", text(snapshot.get("agent_session"))),
                ("provider", text(snapshot.get("sandbox").get("provider"))),
                ("retention", text(snapshot.get("retention"))),
                (
                    "checkpoint_seconds",
                    number(extra.get("timing_sec").get("checkpoint")),
                ),
                (
                    "error",
                    fork.error.clone().map(Value::from).unwrap_or(Value::Null),
                ),
                ("children", Value::Array(rows)),
            ]));
        }
    }
    let mut map = header("branches", root_name, capped, &selection, query);
    map.insert(
        "totals".into(),
        object([
            ("forks", forks.len().into()),
            ("children", children.into()),
            ("scored", scored.into()),
        ]),
    );
    map.insert("forks".into(), Value::Array(forks));
    Value::Object(map)
}

/// `/outcomes`: per job, how many trials passed, failed, went unscored, or
/// errored; the problems behind them by code and owner; and the median and
/// 90th percentile of each phase's time. Every filter chooses trials.
pub fn outcomes(trials: &[Trial], root_name: &str, capped: bool, query: &ViewQuery) -> Value {
    let selection = select(trials, query);
    let kept: Vec<&Trial> = selection
        .trials
        .iter()
        .copied()
        .filter(|trial| query.status.is_empty() || query.status.contains(&trial.outcome))
        .collect();
    let of_job = |job: &str| -> Vec<&Trial> {
        kept.iter()
            .copied()
            .filter(|trial| trial.job == job)
            .collect()
    };

    let summary: Vec<Value> = selection
        .selected
        .iter()
        .map(|job| {
            let members = of_job(job);
            let counts = counts(members.iter().copied());
            object([
                ("job", job.clone().into()),
                ("name", job_name(job, root_name).into()),
                ("trials", members.len().into()),
                ("counts", counts_value(&counts)),
            ])
        })
        .collect();

    // Problems, grouped by code, owner, and where they were recorded.
    let mut groups: BTreeMap<(String, Option<String>, &'static str), Vec<&Trial>> = BTreeMap::new();
    for trial in &kept {
        if let Some(problem) = &trial.problem {
            groups
                .entry((problem.code.clone(), problem.owner.clone(), problem.source))
                .or_default()
                .push(trial);
        }
    }
    let per_job = |members: &[&Trial]| {
        Value::Object(
            selection
                .selected
                .iter()
                .map(|job| {
                    let count = members.iter().filter(|trial| trial.job == *job).count();
                    (job.clone(), count.into())
                })
                .collect(),
        )
    };
    let mut problem_groups: Vec<_> = groups.into_iter().collect();
    problem_groups
        .sort_by(|left, right| right.1.len().cmp(&left.1.len()).then(left.0.cmp(&right.0)));
    let problems: Vec<Value> = problem_groups
        .iter()
        .map(|((code, owner, source), members)| {
            // The most common message and next step stand for the group.
            let most_common = |pick: fn(&Problem) -> Option<&str>| {
                let mut tally: BTreeMap<&str, usize> = BTreeMap::new();
                for trial in members {
                    if let Some(text) = trial.problem.as_ref().and_then(pick) {
                        *tally.entry(text).or_default() += 1;
                    }
                }
                tally
                    .into_iter()
                    .max_by(|left, right| left.1.cmp(&right.1).then(right.0.cmp(left.0)))
                    .map(|(text, _)| text.to_string())
            };
            object([
                ("code", code.clone().into()),
                ("owner", owner.clone().into()),
                ("source", (*source).into()),
                (
                    "message",
                    most_common(|problem| Some(problem.message.as_str())).into(),
                ),
                (
                    "next",
                    most_common(|problem| problem.next.as_deref()).into(),
                ),
                ("total", members.len().into()),
                ("by_job", per_job(members)),
                (
                    "trials",
                    Value::Array(
                        members
                            .iter()
                            .map(|trial| {
                                let mut link = trial.link();
                                if let Value::Object(map) = &mut link {
                                    map.insert("job".into(), trial.job.clone().into());
                                }
                                link
                            })
                            .collect(),
                    ),
                ),
            ])
        })
        .collect();

    // Owners, summed over codes.
    let mut owners: BTreeMap<Option<String>, Vec<&Trial>> = BTreeMap::new();
    for ((_, owner, _), members) in &problem_groups {
        owners
            .entry(owner.clone())
            .or_default()
            .extend(members.iter().copied());
    }
    let mut owner_rows: Vec<_> = owners.into_iter().collect();
    owner_rows.sort_by(|left, right| right.1.len().cmp(&left.1.len()).then(left.0.cmp(&right.0)));
    let owner_values: Vec<Value> = owner_rows
        .iter()
        .map(|(owner, members)| {
            object([
                ("owner", owner.clone().into()),
                ("total", members.len().into()),
                ("by_job", per_job(members)),
            ])
        })
        .collect();

    let timing: Vec<Value> = selection
        .selected
        .iter()
        .map(|job| {
            let phases: Map = PHASES
                .iter()
                .enumerate()
                .map(|(index, phase)| {
                    let seconds: Vec<f64> = of_job(job)
                        .iter()
                        .filter_map(|trial| trial.phases[index])
                        .collect();
                    (
                        (*phase).to_string(),
                        object([
                            ("n", seconds.len().into()),
                            ("median", quantile(&seconds, 0.5).into()),
                            ("p90", quantile(&seconds, 0.9).into()),
                        ]),
                    )
                })
                .collect();
            object([
                ("job", job.clone().into()),
                ("name", job_name(job, root_name).into()),
                ("phases", Value::Object(phases)),
            ])
        })
        .collect();

    let mut map = header("outcomes", root_name, capped, &selection, query);
    map.insert("summary".into(), Value::Array(summary));
    map.insert("problems".into(), Value::Array(problems));
    map.insert("owners".into(), Value::Array(owner_values));
    map.insert("timing".into(), Value::Array(timing));
    Value::Object(map)
}

/// The axes the Pareto view plots pass rate against, in its key names: each
/// point's value is the mean or median over the trials that recorded it.
pub const PARETO_AXES: [&str; 3] = ["cost_per_trial", "median_seconds", "tokens_per_trial"];

/// Which points no other point beats: for each point with a value `x`
/// (lower is better), whether no other point has a pass rate at least as
/// high and an `x` at least as low, with one of the two strictly better.
/// Points without `x` are `None`.
pub fn non_dominated(points: &[(f64, Option<f64>)]) -> Vec<Option<bool>> {
    points
        .iter()
        .map(|(rate, x)| {
            let x = (*x)?;
            Some(!points.iter().any(|(other_rate, other_x)| {
                other_x.is_some_and(|other_x| {
                    *other_rate >= *rate && other_x <= x && (*other_rate > *rate || other_x < x)
                })
            }))
        })
        .collect()
}

/// `/pareto`: one point per job (per agent and model when a job ran
/// several, as `/compare` has columns): its pass rate with the 95% Wilson
/// interval, against what a trial cost (the mean recorded `cost_usd`), how
/// long it took (the median whole-trial time), and how many tokens it used
/// (the mean recorded total), each with the number of trials that recorded
/// it. A point is on an axis's frontier when no other point has a pass rate
/// at least as high for no more of that axis, with one of the two strictly
/// better. `same_tasks` says whether every point ran the same tasks (pass
/// rates over different tasks do not compare). The task and agent filters
/// choose trials; the status filter does not apply (a pass rate needs every
/// outcome).
pub fn pareto(trials: &[Trial], root_name: &str, capped: bool, query: &ViewQuery) -> Value {
    let selection = select(trials, query);
    let mut groups: Vec<((String, String), Vec<&Trial>)> = Vec::new();
    for job in &selection.selected {
        let mut arms: BTreeMap<String, Vec<&Trial>> = BTreeMap::new();
        for trial in selection.trials.iter().filter(|trial| trial.job == *job) {
            arms.entry(trial.arm()).or_default().push(trial);
        }
        groups.extend(
            arms.into_iter()
                .map(|(arm, members)| ((job.clone(), arm), members)),
        );
    }
    struct Point {
        rate: f64,
        axes: [Option<f64>; 3],
        counts: [usize; 3],
    }
    let points: Vec<Point> = groups
        .iter()
        .map(|(_, members)| {
            let passed = members
                .iter()
                .filter(|trial| trial.outcome == Outcome::Passed)
                .count();
            let counted = members
                .iter()
                .filter(|trial| trial.outcome != Outcome::Stopped)
                .count()
                .max(1);
            let pick = |value: fn(&Trial) -> Option<f64>| -> Vec<f64> {
                members.iter().filter_map(|trial| value(trial)).collect()
            };
            let costs = pick(|trial| trial.cost_usd);
            let seconds = pick(|trial| trial.seconds);
            let tokens = pick(|trial| trial.tokens);
            Point {
                rate: passed as f64 / counted as f64,
                axes: [mean(&costs), quantile(&seconds, 0.5), mean(&tokens)],
                counts: [costs.len(), seconds.len(), tokens.len()],
            }
        })
        .collect();
    let frontiers: Vec<Vec<Option<bool>>> = (0..PARETO_AXES.len())
        .map(|axis| {
            non_dominated(
                &points
                    .iter()
                    .map(|point| (point.rate, point.axes[axis]))
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    let values: Vec<Value> = groups
        .iter()
        .zip(&points)
        .enumerate()
        .map(|(index, (((job, arm), members), point))| {
            let passed = members
                .iter()
                .filter(|trial| trial.outcome == Outcome::Passed)
                .count();
            let tasks: BTreeSet<&str> = members.iter().map(|trial| trial.task.as_str()).collect();
            let mut map = Map::new();
            map.insert("job".into(), job.clone().into());
            map.insert("job_name".into(), job_name(job, root_name).into());
            map.insert("arm".into(), arm.clone().into());
            map.insert("tasks".into(), tasks.len().into());
            map.insert("trials".into(), members.len().into());
            map.insert("passed".into(), passed.into());
            map.insert(
                "stopped".into(),
                members
                    .iter()
                    .filter(|trial| trial.outcome == Outcome::Stopped)
                    .count()
                    .into(),
            );
            map.insert("pass_rate".into(), point.rate.into());
            map.insert(
                "pass_interval".into(),
                interval(wilson(
                    passed,
                    members
                        .iter()
                        .filter(|trial| trial.outcome != Outcome::Stopped)
                        .count(),
                )),
            );
            for (axis, name) in PARETO_AXES.iter().enumerate() {
                map.insert((*name).into(), point.axes[axis].into());
                map.insert(format!("{name}_n"), point.counts[axis].into());
            }
            let mut frontier = Map::new();
            for (axis, name) in PARETO_AXES.iter().enumerate() {
                frontier.insert((*name).into(), frontiers[axis][index].into());
            }
            map.insert("frontier".into(), Value::Object(frontier));
            Value::Object(map)
        })
        .collect();
    // Pass rates over different tasks do not compare.
    let task_sets: BTreeSet<BTreeSet<&str>> = groups
        .iter()
        .map(|(_, members)| members.iter().map(|trial| trial.task.as_str()).collect())
        .collect();
    let mut map = header("pareto", root_name, capped, &selection, query);
    map.insert("same_tasks".into(), (task_sets.len() <= 1).into());
    map.insert("points".into(), Value::Array(values));
    Value::Object(map)
}

#[cfg(test)]
mod tests {
    /// Round-2 review of rust/e9-decisions (L5): a job drained before one
    /// of its agents started any trial has no column for that agent; its
    /// planned trials count toward every column of the job, which is then
    /// partial, instead of matching none and leaving the job shown whole.
    #[test]
    fn an_agent_with_no_trial_marks_every_column_of_its_job() {
        let arms = vec![
            "claude \u{b7} opus".to_string(),
            "codex \u{b7} gpt".to_string(),
        ];
        let planned = |agent: &str, model: &str| super::Planned {
            agent: Some(agent.into()),
            model: Some(model.into()),
        };
        // Its own arm only, when it has one.
        assert!(super::planned_in_column(
            &planned("claude", "opus"),
            &arms[0],
            &arms
        ));
        assert!(!super::planned_in_column(
            &planned("claude", "opus"),
            &arms[1],
            &arms
        ));
        // An agent with no column: every column.
        for arm in &arms {
            assert!(super::planned_in_column(
                &planned("gemini", "flash"),
                arm,
                &arms
            ));
        }
    }

    use super::*;
    use crate::pyjson::{dumps, parse};

    fn result(text: &str) -> Map {
        parse(text).unwrap().as_object().unwrap().clone()
    }

    fn outcome(text: &str) -> Outcome {
        let result = result(text);
        let metadata = normalize_metadata(&result, None);
        outcome_of(&result, metadata.reward)
    }

    #[test]
    fn a_point_is_dominated_only_by_one_at_least_as_good_on_both_and_better_on_one() {
        assert_eq!(
            non_dominated(&[
                (1.0, Some(0.4)),
                (0.5, Some(0.1)),
                (0.5, Some(0.2)),
                (0.25, None),
                (1.0, Some(0.4)),
                (0.0, Some(0.0)),
            ]),
            [
                Some(true),
                Some(true),
                Some(false),
                None,
                Some(true),
                Some(true),
            ],
            "ties stay on the frontier; nothing beats the cheapest"
        );
    }

    /// `/branches` reads each trial's tree.json: a fork's point, value,
    /// checkpoint, and parent, and each child with its advantage and the
    /// ref that opens it; a trial without a tree has no fork, and an
    /// unreadable tree is an error row, not a broken page.
    #[test]
    fn the_branches_view_lists_each_fork_and_its_children() {
        let tmp = tempfile::tempdir().unwrap();
        let job = tmp.path().join("job");
        let trial = |name: &str, tree: Option<&str>| {
            let dir = job.join(name);
            std::fs::create_dir_all(dir.join("trajectory")).unwrap();
            std::fs::write(dir.join("trajectory/acp_trajectory.jsonl"), "").unwrap();
            std::fs::write(
                dir.join("result.json"),
                r#"{"task_name": "hello", "agent": "oracle", "status": "unscored"}"#,
            )
            .unwrap();
            if let Some(tree) = tree {
                std::fs::write(dir.join("tree.json"), tree).unwrap();
            }
            Run {
                id: format!("job/{name}"),
                dir,
            }
        };
        let tree = r#"{"schema_version": 1, "kind": "benchflow-branch-tree",
          "nodes": [{"id": "root", "parent": null, "step_id": null}],
          "forks": [{"id": "f1", "parent_node": "root", "fork_point": "stage:first",
            "status": "completed", "value": 0.5, "value_stderr": 0.5,
            "parent_restore": "not_needed", "parent_reward": 1.0,
            "snapshot": {"sandbox": {"provider": "docker"}, "retention": "deleted", "agent_session": "fresh"},
            "timing_sec": {"checkpoint": 0.9},
            "children": [
              {"index": 0, "node_id": "n1", "status": "scored", "reward": 1.0,
               "intervention": {"label": "keep", "requested": null},
               "artifacts": {"status": "available", "path": "branches/f1/children/n1"}},
              {"index": 1, "node_id": "n2", "status": "scored", "reward": 0.0,
               "intervention": {"label": "bye", "requested": "own prompt (8 characters, sha256:0123456789ab)"},
               "artifacts": {"status": "unavailable", "path": null}}]}]}"#;
        let runs = vec![
            trial("a", Some(tree)),
            trial("b", None),
            trial("c", Some("not json")),
        ];
        // A parent that forked at start has a tree and no trajectory: it is
        // still a trial the run list shows.
        std::fs::remove_file(job.join("a/trajectory/acp_trajectory.jsonl")).unwrap();
        assert!(crate::catalog::is_rollout_dir(&job.join("a")));
        assert_eq!(
            crate::catalog::discover(tmp.path(), 10)
                .iter()
                .map(|r| r.id.as_str())
                .collect::<Vec<_>>(),
            ["job/a", "job/b", "job/c"]
        );
        // A stray tree.json that is not a branch tree makes no trial.
        let stray = job.join("d");
        std::fs::create_dir_all(&stray).unwrap();
        std::fs::write(stray.join("tree.json"), "{}").unwrap();
        std::fs::write(stray.join("result.json"), "{}").unwrap();
        assert!(!crate::catalog::is_rollout_dir(&stray));
        // A discarded parent (skipped) is left out of the trial views.
        std::fs::write(
            job.join("a/result.json"),
            r#"{"task_name": "hello", "agent": "oracle", "status": "skipped"}"#,
        )
        .unwrap();
        assert!(discarded_parent(&runs[0]));
        assert!(!discarded_parent(&runs[1]));
        let trials: Vec<Trial> = runs.iter().map(read_trial).collect();
        let out = branches(&trials, &runs, "root", false, &ViewQuery::default());
        let forks = out.get("forks").as_array().unwrap();
        assert_eq!(forks.len(), 2, "{}", dumps(&out));
        let f = &forks[0];
        assert_eq!(f.get("fork_point").as_str(), Some("stage:first"));
        assert_eq!(f.get("parent_restore").as_str(), Some("not_needed"));
        assert_eq!(finite_float(f.get("parent_reward")), Some(1.0));
        assert_eq!(f.get("provider").as_str(), Some("docker"));
        let kids = f.get("children").as_array().unwrap();
        assert_eq!(finite_float(kids[0].get("advantage")), Some(0.5));
        assert_eq!(finite_float(kids[1].get("advantage")), Some(-0.5));
        assert_eq!(kids[1].get("label").as_str(), Some("bye"));
        assert_eq!(kids[1].get("own_prompt"), &Value::Bool(true));
        assert!(kids[1].get("ref").is_null());
        assert!(forks[1].get("error").as_str().is_some(), "{}", dumps(&out));
        assert_eq!(out.get("totals").get("children"), &Value::from(2usize));
    }

    /// A trial a live `bench run` has not finished is left out; once no
    /// run holds the job, a trial without a result is a real error again.
    #[test]
    fn a_trial_still_running_is_not_counted_as_errored() {
        let tmp = tempfile::tempdir().unwrap();
        let job = tmp.path().join("job");
        let run = Run {
            id: "job/t__1".into(),
            dir: job.join("t__1"),
        };
        std::fs::create_dir_all(run.dir.join("trajectory")).unwrap();
        assert!(!in_progress(&run), "no run lock: not running");
        let lock = std::fs::File::create(job.join(RUNNING_FILE)).unwrap();
        lock.lock().unwrap();
        assert!(in_progress(&run), "a run holds the job");
        std::fs::write(run.dir.join("result.json"), r#"{"status": "scored"}"#).unwrap();
        assert!(!in_progress(&run), "finished trials always count");
        std::fs::remove_file(run.dir.join("result.json")).unwrap();
        drop(lock);
        assert!(
            !in_progress(&run),
            "the run is gone: the trial stopped without a result"
        );
        assert_eq!(read_trial(&run).outcome, Outcome::Errored);
    }

    /// Behavior review 6: a trial the user stopped (R001) is its own
    /// outcome, left out of the compare pass rate, as `bench results` does.
    #[test]
    fn stopped_trials_are_their_own_outcome_and_left_out_of_pass_rates() {
        assert_eq!(
            outcome(r#"{"status": "skipped", "problem": {"code": "R001"}}"#),
            Outcome::Stopped
        );
        assert_eq!(
            outcome(r#"{"status": "skipped", "problem": {"code": "R000"}}"#),
            Outcome::Errored
        );
        let mut trials: Vec<Trial> = (0..3)
            .map(|i| {
                trial(
                    &format!("j/t{i}__1"),
                    &format!("t{i}"),
                    "x",
                    Outcome::Passed,
                    Some(1.0),
                )
            })
            .collect();
        for i in 3..5 {
            trials.push(trial(
                &format!("j/t{i}__1"),
                &format!("t{i}"),
                "x",
                Outcome::Stopped,
                None,
            ));
        }
        let view = compare(&trials, "jobs", false, &ViewQuery::default());
        let column = &view.get("columns").as_array().unwrap()[0];
        assert_eq!(dumps(column.get("pass_rate")), "1.0");
        assert_eq!(dumps(column.get("counts").get("stopped")), "2");
    }

    #[test]
    fn outcomes_follow_the_engine_status_then_0_8_fields() {
        assert_eq!(
            outcome(r#"{"status": "scored", "passed": true, "rewards": {"reward": 0.5}}"#),
            Outcome::Passed
        );
        assert_eq!(
            outcome(r#"{"status": "scored", "passed": false, "rewards": {"reward": 1.0}}"#),
            Outcome::Failed
        );
        assert_eq!(
            outcome(r#"{"status": "unscored", "rewards": null}"#),
            Outcome::Unscored
        );
        assert_eq!(outcome(r#"{"status": "errored"}"#), Outcome::Errored);
        assert_eq!(outcome(r#"{"status": "skipped"}"#), Outcome::Errored);
        // 0.8: a reward decides; then an error; else unscored.
        assert_eq!(outcome(r#"{"rewards": {"reward": 1.0}}"#), Outcome::Passed);
        assert_eq!(
            outcome(r#"{"rewards": {"reward": 0.25}, "error": "late"}"#),
            Outcome::Failed
        );
        assert_eq!(
            outcome(r#"{"rewards": null, "error": "ACP error"}"#),
            Outcome::Errored
        );
        assert_eq!(
            outcome(r#"{"rewards": null, "verifier_error": "crashed"}"#),
            Outcome::Unscored
        );
        assert_eq!(outcome(r#"{"task_name": "x"}"#), Outcome::Unscored);
        assert_eq!(outcome("{}"), Outcome::Errored);
    }

    #[test]
    fn problems_come_from_the_engine_or_0_8_errors() {
        let engine = problem_of(&result(
            r#"{"problem": {"code": "R304", "owner": "task", "phase": "verify", "message": "no reward file\nmore", "next": "fix the verifier"}}"#,
        ))
        .unwrap();
        assert_eq!(engine.code, "R304");
        assert_eq!(engine.owner.as_deref(), Some("task"));
        assert_eq!(engine.message, "no reward file");
        assert_eq!(engine.next.as_deref(), Some("fix the verifier"));
        let old = problem_of(&result(
            r#"{"error": "boom", "error_category": "acp_error"}"#,
        ))
        .unwrap();
        assert_eq!(
            (old.code.as_str(), old.owner, old.source),
            ("acp_error", None, "error")
        );
        let verifier = problem_of(&result(
            r#"{"verifier_error": "verifier crashed: Failed to get session command: "}"#,
        ))
        .unwrap();
        assert_eq!(
            (verifier.code.as_str(), verifier.source),
            ("verifier_infra", "verifier_error")
        );
        assert_eq!(
            problem_of(&result(r#"{"problem": null, "error": null}"#)),
            None
        );
    }

    fn trial(id: &str, task: &str, agent: &str, outcome: Outcome, reward: Option<f64>) -> Trial {
        Trial {
            id: id.into(),
            job: job_of(id).into(),
            task: task.into(),
            agent: agent.into(),
            model: None,
            outcome,
            reward,
            problem: None,
            phases: [Some(1.0), Some(10.0), None],
            seconds: None,
            cost_usd: None,
            tokens: None,
        }
    }

    fn corpus() -> Vec<Trial> {
        use Outcome::*;
        vec![
            trial("a/t1__1", "t1", "x", Passed, Some(1.0)),
            trial("a/t2__1", "t2", "x", Passed, Some(1.0)),
            trial("a/t3__1", "t3", "x", Failed, Some(0.0)),
            trial("a/t4__1", "t4", "x", Errored, None),
            trial("b/t1__1", "t1", "x", Passed, Some(1.0)),
            trial("b/t2__1", "t2", "x", Failed, Some(0.0)),
            trial("b/t3__1", "t3", "x", Failed, Some(0.5)),
            trial("b/t4__1", "t4", "x", Passed, Some(1.0)),
            trial("b/t5__1", "t5", "x", Passed, Some(1.0)),
        ]
    }

    #[test]
    fn compare_pairs_shared_scored_tasks_and_sorts_by_disagreement() {
        let trials = corpus();
        let query = ViewQuery {
            jobs: vec!["a".into(), "b".into(), "missing".into()],
            ..ViewQuery::default()
        };
        let view = compare(&trials, "root", false, &query);
        let tasks: Vec<&str> = view
            .get("rows")
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row.get("task").as_str().unwrap())
            .collect();
        // t2 differs by 1.0, t3 by 0.5, t4 has mixed outcomes, t1 and t5 agree.
        assert_eq!(tasks, ["t2", "t3", "t4", "t1", "t5"]);
        assert_eq!(dumps(view.get("unknown")), r#"["missing"]"#);
        let pair = &view.get("pairs").as_array().unwrap()[0];
        // Shared scored tasks t1, t2, t3: differences 0, 1, -0.5.
        assert_eq!(dumps(pair.get("shared")), "3");
        assert_eq!(dumps(pair.get("mean_difference")), "0.16666666666666666");
        assert_eq!(
            [
                pair.get("better"),
                pair.get("worse"),
                pair.get("tied"),
                pair.get("excluded"),
                pair.get("one_side")
            ]
            .map(dumps),
            ["1", "1", "1", "1", "1"]
        );
        let column = &view.get("columns").as_array().unwrap()[0];
        assert_eq!(
            dumps(column.get("counts")),
            r#"{"passed": 2, "failed": 1, "unscored": 0, "errored": 1}"#
        );
        assert_eq!(dumps(column.get("pass_rate")), "0.5");
    }

    #[test]
    fn the_status_filter_keeps_rows_with_a_matching_trial() {
        let trials = corpus();
        let query = ViewQuery {
            status: [Outcome::Errored].into(),
            ..ViewQuery::default()
        };
        let view = compare(&trials, "root", false, &query);
        let rows = view.get("rows").as_array().unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("task").as_str(), Some("t4"));
        // Both columns count only the rows shown.
        let columns = view.get("columns").as_array().unwrap();
        assert_eq!(dumps(columns[1].get("pass_rate")), "1.0");
    }

    #[test]
    fn a_job_with_two_agents_gets_two_columns() {
        use Outcome::*;
        let trials = vec![
            trial("job/t1__1", "t1", "oracle", Passed, Some(1.0)),
            trial("job/t1__2", "t1", "nop", Failed, Some(0.0)),
        ];
        let view = compare(&trials, "root", false, &ViewQuery::default());
        let arms: Vec<String> = view
            .get("columns")
            .as_array()
            .unwrap()
            .iter()
            .map(|column| dumps(column.get("arm")))
            .collect();
        assert_eq!(arms, [r#""nop""#, r#""oracle""#]);
        let only_oracle = ViewQuery {
            agent: "oracle".into(),
            ..ViewQuery::default()
        };
        let view = compare(&trials, "root", false, &only_oracle);
        assert_eq!(view.get("columns").as_array().unwrap().len(), 1);
    }

    #[test]
    fn outcomes_count_group_and_time_per_job() {
        let mut trials = corpus();
        trials[3].problem = Some(Problem {
            code: "R201".into(),
            owner: Some("sandbox".into()),
            message: "sandbox did not start".into(),
            next: Some("check the backend".into()),
            source: "problem",
        });
        trials[3].phases = [Some(3.0), None, None];
        let view = outcomes(
            &trials,
            "root",
            false,
            &ViewQuery {
                jobs: vec!["a".into()],
                ..ViewQuery::default()
            },
        );
        let summary = &view.get("summary").as_array().unwrap()[0];
        assert_eq!(
            dumps(summary.get("counts")),
            r#"{"passed": 2, "failed": 1, "unscored": 0, "errored": 1}"#
        );
        let problem = &view.get("problems").as_array().unwrap()[0];
        assert_eq!(dumps(problem.get("code")), r#""R201""#);
        assert_eq!(dumps(problem.get("by_job")), r#"{"a": 1}"#);
        let setup = view.get("timing").as_array().unwrap()[0]
            .get("phases")
            .get("environment_setup")
            .clone();
        assert_eq!(
            dumps(&setup),
            r#"{"n": 4, "median": 1.0, "p90": 2.4000000000000004}"#
        );
        let passed_only = ViewQuery {
            status: [Outcome::Passed].into(),
            task: "T1".into(),
            ..ViewQuery::default()
        };
        let view = outcomes(&trials, "root", false, &passed_only);
        let counts: Vec<String> = view
            .get("summary")
            .as_array()
            .unwrap()
            .iter()
            .map(|job| dumps(job.get("trials")))
            .collect();
        assert_eq!(counts, ["1", "1"]);
    }
}
