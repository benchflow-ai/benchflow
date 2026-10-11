//! Reuse plans against earlier jobs (`bench run --diff <job>`).
//!
//! Before a job runs, each of its planned trials is matched against the
//! finished trials of one or more earlier jobs, read from their lock.json
//! and trial folders, and sorted into:
//!
//! - **reuse**: an earlier trial ran the same task (same digest) with the
//!   same agent, model, effort, and settings that change results, and was
//!   scored. Its folder is copied into the new job as it is.
//! - **regrade**: the agent's side matches (same inputs digest, agent, and
//!   settings) but grading changed (the task's tests or verifier, or the
//!   verifier's time limits), and the earlier trial saved its outputs. The
//!   outputs are graded again with the new task's verifier; the agent does
//!   not run.
//! - **rerun**: anything else, with the reason.
//!
//! Each earlier trial is used at most once.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use benchflow_task::TaskPackage;
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::agent::Attempt;
use crate::config::Rerun;
use crate::lock::{Lock, LockOptions, LockTrial};
use crate::outcome::Status;
use crate::record::TrialRecord;
use crate::regrade::{self, Outputs};
use crate::trial::TrialSpec;

/// What happens to one planned trial.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Action {
    Reuse,
    Regrade,
    Rerun,
    /// No earlier job had a trial of this task: it runs for the first time.
    Run,
}

impl Action {
    pub fn as_str(self) -> &'static str {
        match self {
            Action::Reuse => "reuse",
            Action::Regrade => "regrade",
            Action::Rerun => "rerun",
            Action::Run => "run",
        }
    }
}

/// One planned trial's verdict.
#[derive(Debug, Clone, Serialize)]
pub struct TrialDiff {
    pub action: Action,
    /// The planned trial's task (folder name) and id.
    pub task: String,
    pub task_id: String,
    pub agent: String,
    /// Which of the task's trials this is (0 for `-k 1`).
    pub index: usize,
    /// The earlier trial used, for reuse and regrade.
    pub source: Option<SourceRef>,
    pub reason: String,
}

/// An earlier trial, by job and trial folder.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SourceRef {
    pub job: PathBuf,
    pub trial: String,
}

/// An earlier job that could not be read, and why.
#[derive(Debug, Clone, Serialize)]
pub struct Skipped {
    pub job: PathBuf,
    pub reason: String,
}

/// The whole preview.
#[derive(Debug, Clone, Serialize)]
pub struct DiffPlan {
    pub sources: Vec<PathBuf>,
    pub skipped: Vec<Skipped>,
    pub trials: Vec<TrialDiff>,
}

impl DiffPlan {
    pub fn count(&self, action: Action) -> usize {
        self.trials.iter().filter(|t| t.action == action).count()
    }

    /// What to carry into the new job: reused and regraded trials.
    pub fn carries(&self, sources: &[SourceTrial]) -> Vec<Carry> {
        self.trials
            .iter()
            .filter(|t| matches!(t.action, Action::Reuse | Action::Regrade))
            .filter_map(|t| {
                let s = t.source.as_ref()?;
                let src = sources
                    .iter()
                    .find(|x| x.job == s.job && x.lock.trial == s.trial)?;
                Some(Carry {
                    task_id: t.task_id.clone(),
                    agent: t.agent.clone(),
                    index: t.index,
                    action: t.action,
                    dir: src.dir.clone(),
                    record: src.record.clone()?,
                    outputs: src.outputs.clone(),
                })
            })
            .collect()
    }
}

/// An earlier trial to bring into a new job: copied (reuse) or graded again
/// from its saved outputs (regrade).
#[derive(Debug, Clone)]
pub struct Carry {
    pub task_id: String,
    /// The attempt label it stands in for.
    pub agent: String,
    pub index: usize,
    pub action: Action,
    /// The earlier trial's folder.
    pub dir: PathBuf,
    pub record: TrialRecord,
    pub outputs: Option<Outputs>,
}

/// A finished trial of an earlier job.
#[derive(Debug, Clone)]
pub struct SourceTrial {
    pub job: PathBuf,
    pub dir: PathBuf,
    pub lock: LockTrial,
    pub options: LockOptions,
    /// Its result.json; `None` when it never finished.
    pub record: Option<TrialRecord>,
    /// `--effort`, from lock.json or (older jobs) config.json.
    pub effort: Option<String>,
    /// `--ak` values, from config.json (none for older jobs).
    pub agent_options: serde_json::Map<String, Value>,
    pub outputs: Option<Outputs>,
}

/// Read the earlier jobs' trials. A job that cannot be read is skipped
/// with the reason; the caller decides whether that is fatal.
pub fn load_sources(jobs: &[PathBuf]) -> (Vec<SourceTrial>, Vec<Skipped>) {
    let mut trials = Vec::new();
    let mut skipped = Vec::new();
    for job in jobs {
        match load_job(job) {
            Ok(mut t) => trials.append(&mut t),
            Err(reason) => skipped.push(Skipped {
                job: job.clone(),
                reason,
            }),
        }
    }
    (trials, skipped)
}

fn load_job(job: &Path) -> Result<Vec<SourceTrial>, String> {
    if !job.is_dir() {
        return Err("no such job folder".into());
    }
    let lock = match crate::read::read_lock(job) {
        Ok(l) => l,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(
                "it has no lock.json (BenchFlow 0.8 and earlier wrote none), so what its trials ran with is unknown"
                    .into(),
            );
        }
        Err(e) => return Err(format!("its lock.json is unreadable: {e}")),
    };
    if let Some(of) = &lock.regrade_of {
        // A regrade's lock names a placeholder attempt, not the agent's
        // install, so its trials cannot be matched on the agent's side.
        return Err(format!(
            "it regrades {}; pass that job instead",
            of.display()
        ));
    }
    // A job can come from someone else: each trial must be a real folder
    // directly in it, named by one plain component.
    let in_job = |name: &str| {
        let plain = matches!(
            Path::new(name).components().collect::<Vec<_>>()[..],
            [std::path::Component::Normal(_)]
        );
        plain && std::fs::symlink_metadata(job.join(name)).is_ok_and(|m| m.file_type().is_dir())
    };
    Ok(lock
        .trials
        .iter()
        .filter(|t| in_job(&t.trial))
        .map(|t| {
            let dir = job.join(&t.trial);
            let config: Option<Value> = std::fs::read_to_string(dir.join("config.json"))
                .ok()
                .and_then(|s| serde_json::from_str(&s).ok());
            SourceTrial {
                job: job.to_path_buf(),
                lock: t.clone(),
                options: lock.options.clone(),
                record: crate::read::read_trial(&dir),
                effort: t.attempt.effort.clone().or_else(|| {
                    config
                        .as_ref()
                        .and_then(|c| c.get("reasoning_effort"))
                        .and_then(Value::as_str)
                        .map(str::to_string)
                }),
                agent_options: config
                    .as_ref()
                    .and_then(|c| c.get("agent_options"))
                    .and_then(Value::as_object)
                    .cloned()
                    .unwrap_or_default(),
                outputs: regrade::saved_outputs(&dir),
                dir,
            }
        })
        .collect())
}

/// How close an earlier trial came, for picking the reason a planned trial
/// reruns: higher is closer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Closeness {
    Agent,
    Settings,
    Inputs,
    Status,
    Regrade,
    Reuse,
}

/// Compare `planned` (with its `target` lock, built for the same trials in
/// the same order) against earlier trials.
pub fn plan(
    planned: &[TrialSpec],
    target: &Lock,
    sources: Vec<PathBuf>,
    found: &[SourceTrial],
    skipped: Vec<Skipped>,
    rerun: &Rerun,
) -> DiffPlan {
    let mut used = vec![false; found.len()];
    let mut index_of: Vec<((String, String), usize)> = Vec::new();
    let mut trials = Vec::new();
    for (spec, t) in planned.iter().zip(&target.trials) {
        let key = (t.task.id.clone(), t.attempt.agent.clone());
        let index = match index_of.iter_mut().find(|(k, _)| *k == key) {
            Some((_, n)) => {
                *n += 1;
                *n
            }
            None => {
                index_of.push((key, 0));
                0
            }
        };
        let mut best: Option<(Closeness, usize, String)> = None;
        // The closest earlier trial another planned trial already took.
        let mut taken: Option<Closeness> = None;
        let mut any = false;
        for (i, s) in found.iter().enumerate() {
            if s.lock.task.id != t.task.id {
                continue;
            }
            any = true;
            let (c, reason) = judge(
                &spec.task,
                t,
                &spec.attempt.options(),
                &target.options,
                s,
                rerun,
            );
            if used[i] {
                taken = taken.max(Some(c));
            } else if best.as_ref().is_none_or(|(b, _, _)| c > *b) {
                best = Some((c, i, reason));
            }
        }
        let best_free = best.as_ref().map(|(c, _, _)| *c);
        let (action, source, reason) = match best {
            Some((Closeness::Reuse, i, reason)) => (Action::Reuse, Some(i), reason),
            Some((Closeness::Regrade, i, reason)) => (Action::Regrade, Some(i), reason),
            _ if taken >= Some(Closeness::Regrade) && taken > best_free => (
                Action::Rerun,
                None,
                "the earlier trials that match are already used by other planned trials".into(),
            ),
            Some((_, _, reason)) => (Action::Rerun, None, reason),
            None if any => (
                Action::Rerun,
                None,
                "the earlier trials that match are already used by other planned trials".into(),
            ),
            None => (
                Action::Run,
                None,
                "no trial of this task in the earlier jobs".into(),
            ),
        };
        if let Some(i) = source {
            used[i] = true;
        }
        trials.push(TrialDiff {
            action,
            task: t.task.name.clone(),
            task_id: t.task.id.clone(),
            agent: t.attempt.agent.clone(),
            index,
            source: source.map(|i| SourceRef {
                job: found[i].job.clone(),
                trial: found[i].lock.trial.clone(),
            }),
            reason,
        });
    }
    DiffPlan {
        sources,
        skipped,
        trials,
    }
}

/// Whether two agent fingerprints name the same install, counting an earlier
/// build's name for one as the same.
fn same_install(then: &Option<String>, now: &Option<String>) -> bool {
    let current = crate::external_agent::current_fingerprint;
    then.as_deref().map(current) == now.as_deref().map(current)
}

/// One planned trial against one earlier trial of the same task.
fn judge(
    task: &TaskPackage,
    t: &LockTrial,
    agent_options: &serde_json::Map<String, Value>,
    opts: &LockOptions,
    s: &SourceTrial,
    rerun: &Rerun,
) -> (Closeness, String) {
    let was = |what: &str, a: &dyn std::fmt::Display, b: &dyn std::fmt::Display| {
        format!("{what} differs: {a} then, {b} now")
    };
    let show = |o: &Option<String>| o.clone().unwrap_or_else(|| "none".into());
    if s.lock.attempt.agent != t.attempt.agent {
        return (
            Closeness::Agent,
            was("the agent", &s.lock.attempt.agent, &t.attempt.agent),
        );
    }
    if s.lock.attempt.model != t.attempt.model {
        return (
            Closeness::Agent,
            was(
                "the model",
                &show(&s.lock.attempt.model),
                &show(&t.attempt.model),
            ),
        );
    }
    if s.effort != t.attempt.effort {
        return (
            Closeness::Agent,
            was("the effort", &show(&s.effort), &show(&t.attempt.effort)),
        );
    }
    if !same_install(
        &s.lock.attempt.agent_fingerprint,
        &t.attempt.agent_fingerprint,
    ) {
        return (
            Closeness::Agent,
            "the agent's install changed (its version, harness, or install steps)".into(),
        );
    }
    if let Some(reason) = agent_settings(&s.options, &s.lock.sandbox, opts, &t.sandbox) {
        return (Closeness::Settings, reason);
    }
    if s.agent_options != *agent_options {
        return (
            Closeness::Settings,
            "the agent options differ (--ak)".into(),
        );
    }
    let Some(rec) = &s.record else {
        return (Closeness::Status, "the earlier trial never finished".into());
    };
    let same_task = s.lock.task.digest == t.task.digest;
    let same_grading = same_task && verifier_settings(&s.options, opts).is_none();
    if same_grading {
        return match rec.status {
            Status::Scored if rerun.failed && rec.passed != Some(true) => (
                Closeness::Status,
                "the earlier trial failed, and --rerun failed runs failed trials again".into(),
            ),
            Status::Scored => (
                Closeness::Reuse,
                "same task, agent, model, and settings; it was scored".into(),
            ),
            other => (Closeness::Status, ended(other, rec)),
        };
    }
    // Grading changed. The agent's side must not have.
    let what = if same_task {
        verifier_settings(&s.options, opts).unwrap_or_default()
    } else {
        "the task's tests or verifier changed".into()
    };
    match (&s.lock.task.inputs_digest, &t.task.inputs_digest) {
        (Some(a), Some(b)) if a == b => {}
        (None, _) if !same_task => {
            return (
                Closeness::Inputs,
                "the task changed, and the earlier lock.json predates inputs digests, so whether the agent's side changed is unknown".into(),
            );
        }
        (Some(_), Some(_)) if s.lock.image.digest != t.image.digest => {
            return (Closeness::Inputs, "the task's image changed".into());
        }
        (Some(_), Some(_)) => {
            return (
                Closeness::Inputs,
                "the task changed what the agent works from (instruction, settings, skills, mounts, services, or hooks)".into(),
            );
        }
        _ => {}
    }
    if !matches!(rec.status, Status::Scored | Status::Unscored) {
        return (Closeness::Status, ended(rec.status, rec));
    }
    let Some(outputs) = &s.outputs else {
        return (
            Closeness::Status,
            format!("{what}, and the earlier trial saved no outputs to grade again"),
        );
    };
    if let Some(p) = crate::features::regrade_refusal(task) {
        return (Closeness::Status, format!("{what}; {}", p.message));
    }
    if let Some(reason) = regrade::stray_outputs(task, &s.dir, outputs) {
        return (Closeness::Status, format!("{what}, but {reason}"));
    }
    (
        Closeness::Regrade,
        format!("{what}; the agent's side is the same and its outputs were saved"),
    )
}

fn ended(status: Status, rec: &TrialRecord) -> String {
    let word = match status {
        Status::Scored => "was scored",
        Status::Unscored => "was not scored",
        Status::Errored => "errored",
        Status::Skipped if rec.stopped() => "was stopped before it finished",
        Status::Skipped => "never ran",
    };
    match &rec.problem {
        Some(p) => format!("the earlier trial {word} ({})", p.code),
        None => format!("the earlier trial {word}"),
    }
}

/// The first run setting that changes what the agent does, when one
/// differs. A target backend of "" (not chosen yet) matches any.
fn agent_settings(
    a: &LockOptions,
    a_sandbox: &str,
    b: &LockOptions,
    b_sandbox: &str,
) -> Option<String> {
    let f = |x: Option<f64>| x.map(|v| format!("{v}s")).unwrap_or_else(|| "none".into());
    let u = |x: Option<u64>| x.map(|v| v.to_string()).unwrap_or_else(|| "none".into());
    let checks: [(&str, String, String); 9] = [
        (
            "the sandbox backend",
            a_sandbox.to_string(),
            if b_sandbox.is_empty() {
                a_sandbox.to_string()
            } else {
                b_sandbox.to_string()
            },
        ),
        (
            "the sandbox user",
            a.sandbox_user.clone(),
            b.sandbox_user.clone(),
        ),
        (
            "the agent timeout multiplier",
            a.agent_timeout_multiplier.to_string(),
            b.agent_timeout_multiplier.to_string(),
        ),
        (
            "the default agent timeout",
            f(a.default_agent_timeout_sec),
            f(b.default_agent_timeout_sec),
        ),
        (
            "the idle timeout",
            f(Some(a.idle_timeout_sec)),
            f(Some(b.idle_timeout_sec)),
        ),
        (
            "the process limit",
            a.pids_limit.to_string(),
            b.pids_limit.to_string(),
        ),
        (
            "--runtime",
            a.runtime.clone().unwrap_or_else(|| "default".into()),
            b.runtime.clone().unwrap_or_else(|| "default".into()),
        ),
        ("--max-tool-calls", u(a.max_tool_calls), u(b.max_tool_calls)),
        ("--max-tokens", u(a.max_tokens), u(b.max_tokens)),
    ];
    for (what, x, y) in checks {
        if x != y {
            return Some(format!("{what} differs: {x} then, {y} now"));
        }
    }
    if a.extra_instruction != b.extra_instruction {
        return Some("--extra-instruction differs".into());
    }
    if a.skills != b.skills || a.task_skills != b.task_skills {
        return Some("the skills differ (--skill, --no-task-skills)".into());
    }
    if a.load_trajectory != b.load_trajectory {
        return Some("--load-trajectory differs".into());
    }
    if a.resume_sessions != b.resume_sessions {
        return Some("--resume-sessions differs".into());
    }
    if a.agent_env_keys != b.agent_env_keys {
        return Some(format!(
            "the --env variables differ: {} then, {} now",
            names(&a.agent_env_keys),
            names(&b.agent_env_keys)
        ));
    }
    None
}

/// Variable names for a reason: `A, B`, or `none`.
fn names(keys: &[String]) -> String {
    if keys.is_empty() {
        "none".into()
    } else {
        keys.join(", ")
    }
}

/// The first run setting that changes only grading, when one differs.
fn verifier_settings(a: &LockOptions, b: &LockOptions) -> Option<String> {
    if a.verifier_timeout_multiplier != b.verifier_timeout_multiplier {
        return Some(format!(
            "the verifier timeout multiplier differs: {} then, {} now",
            a.verifier_timeout_multiplier, b.verifier_timeout_multiplier
        ));
    }
    if a.default_verifier_timeout_sec != b.default_verifier_timeout_sec {
        return Some(format!(
            "the default verifier timeout differs: {}s then, {}s now",
            a.default_verifier_timeout_sec, b.default_verifier_timeout_sec
        ));
    }
    None
}

/// The inputs digest of `task` for `attempt` (see
/// [`crate::lock::LockTask::inputs_digest`]): everything the agent works
/// from and everything that runs before its outputs are saved, and nothing
/// that only grades. Package paths are hashed by content, so moving a task
/// folder does not change it.
pub fn inputs_digest(task: &TaskPackage, attempt: &Attempt) -> String {
    let mut h = Sha256::new();
    let mut field = |name: &str, bytes: &[u8]| {
        h.update(name.as_bytes());
        h.update(b"\0");
        h.update(Sha256::digest(bytes));
    };
    field("v", b"benchflow-inputs-v1");
    field("instruction", task.instruction.as_bytes());
    field("stages", &json_bytes(&task.stages));
    field("agent", &json_bytes(&task.agent));
    let mut sandbox = serde_json::to_value(&task.sandbox).unwrap_or(Value::Null);
    if let Some(m) = sandbox.as_object_mut() {
        m.insert(
            "image".into(),
            Value::String(crate::images::spec_for(task, Vec::new()).digest),
        );
        m.insert(
            "skills".into(),
            match &task.sandbox.skills {
                Some(p) => Value::String(tree_digest(&in_pkg(task, p))),
                None => Value::Null,
            },
        );
    }
    field("sandbox", &json_bytes(&sandbox));
    let x = &task.extras;
    field("steps", &json_bytes(&x.steps));
    let services: Vec<Value> = x
        .services
        .iter()
        .map(|s| {
            let mut v = serde_json::to_value(s).unwrap_or(Value::Null);
            v["image"] =
                Value::String(crate::images::spec_for_image(task, &s.image, &s.name).digest);
            v
        })
        .collect();
    field("services", &json_bytes(&services));
    field("hooks", &json_bytes(&x.hooks));
    let mounts: Vec<Value> = x
        .mounts
        .iter()
        .map(|m| serde_json::json!({"target": m.target, "content": tree_digest(&in_pkg(task, &m.source))}))
        .collect();
    field("mounts", &json_bytes(&mounts));
    field("gpu_types", &json_bytes(&x.gpu_types));
    match attempt {
        Attempt::Oracle => {
            if let Some(o) = &task.oracle {
                field("oracle", tree_digest(&in_pkg(task, &o.dir)).as_bytes());
                field(
                    "oracle_run",
                    &json_bytes(
                        &serde_json::json!({"mount": o.mount, "entry": o.entry, "env": o.env}),
                    ),
                );
            }
        }
        Attempt::Control(c) => {
            field("control", tree_digest(&in_pkg(task, &c.script)).as_bytes());
        }
        _ => {}
    }
    format!("sha256:{}", hex::encode(h.finalize()))
}

/// A package path, as `trial` resolves it: policies hold paths already
/// joined to the task folder, or relative to it.
fn in_pkg(task: &TaskPackage, p: &Path) -> PathBuf {
    crate::trial::in_package(&task.root, p)
}

fn json_bytes<T: Serialize + ?Sized>(v: &T) -> Vec<u8> {
    serde_json::to_vec(v).unwrap_or_default()
}

/// A file's or folder's content: relative paths and file hashes, sorted;
/// links are not followed. A missing path hashes as missing.
fn tree_digest(path: &Path) -> String {
    let mut h = Sha256::new();
    if path.is_file() {
        h.update(b"file\0");
        if let Ok(b) = std::fs::read(path) {
            h.update(Sha256::digest(&b));
        }
    } else if path.is_dir() {
        let mut files: Vec<PathBuf> = walkdir::WalkDir::new(path)
            .follow_links(false)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|e| e.file_type().is_file())
            .map(|e| e.into_path())
            .collect();
        files.sort();
        for f in files {
            let rel = f.strip_prefix(path).unwrap_or(&f);
            h.update(rel.to_string_lossy().as_bytes());
            h.update(b"\0");
            if let Ok(b) = std::fs::read(&f) {
                h.update(Sha256::digest(&b));
            }
        }
    } else {
        h.update(b"missing");
    }
    hex::encode(h.finalize())
}

/// The planned trials of a job, as `job::run` plans them, for a preview
/// before the job's folder exists.
pub fn planned_trials(
    units: &[(Arc<TaskPackage>, Vec<Attempt>)],
    trials: usize,
    job: &str,
) -> Vec<TrialSpec> {
    let mut out = Vec::new();
    for (task, attempts) in units {
        for attempt in attempts {
            for i in 0..trials.max(1) {
                let name = crate::job::trial_name(job, task, attempt, i);
                out.push(TrialSpec {
                    task: task.clone(),
                    attempt: attempt.clone(),
                    dir: PathBuf::from(&name),
                    name,
                });
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn copy(from: &Path, to: &Path) {
        for e in walkdir::WalkDir::new(from) {
            let e = e.unwrap();
            let dest = to.join(e.path().strip_prefix(from).unwrap());
            if e.file_type().is_dir() {
                std::fs::create_dir_all(&dest).unwrap();
            } else {
                std::fs::copy(e.path(), &dest).unwrap();
            }
        }
    }

    fn load(dir: &Path) -> TaskPackage {
        benchflow_task::load(dir).expect("the fixture loads")
    }

    #[test]
    fn reuse_compares_resume_sessions_and_env_names() {
        let a = crate::lock::lock_options(&crate::config::RunOptions::default(), 1, 1);
        assert_eq!(agent_settings(&a, "docker", &a, "docker"), None);
        let mut b = a.clone();
        b.resume_sessions = true;
        assert_eq!(
            agent_settings(&a, "docker", &b, "docker").as_deref(),
            Some("--resume-sessions differs")
        );
        let mut b = a.clone();
        b.agent_env_keys = vec!["DB_URL".into()];
        assert_eq!(
            agent_settings(&a, "docker", &b, "docker").as_deref(),
            Some("the --env variables differ: none then, DB_URL now")
        );
    }

    /// Guards 044f1d73 (`bench run --diff`): a shared job's lock.json named
    /// trial folders outside it, and links were followed to its trials and
    /// saved outputs.
    #[cfg(unix)]
    #[test]
    fn a_shared_jobs_trials_must_be_its_own_folders() {
        use std::os::unix::fs::symlink;
        let fixture =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/export/jobs/rust-claude");
        let d = tempfile::tempdir().unwrap();
        let elsewhere = d.path().join("elsewhere");
        copy(&fixture.join("hello__5af89adc"), &elsewhere);
        std::fs::write(elsewhere.join("outputs.tar"), "x").unwrap();
        let job = d.path().join("job");
        copy(&fixture, &job);
        symlink(&elsewhere, job.join("linked")).unwrap();
        let real = job.join("hello__5af89adc");
        symlink(elsewhere.join("outputs.tar"), real.join("outputs.tar")).unwrap();
        let lock = job.join("lock.json");
        let mut v: Value = serde_json::from_str(&std::fs::read_to_string(&lock).unwrap()).unwrap();
        let first = v["trials"][0].clone();
        let named = |n: &str| {
            let mut t = first.clone();
            t["trial"] = Value::String(n.into());
            t
        };
        v["trials"] = Value::Array(vec![
            named("hello__5af89adc"),
            named("../elsewhere"),
            named(&elsewhere.display().to_string()),
            named("linked"),
        ]);
        std::fs::write(&lock, v.to_string()).unwrap();
        let (found, skipped) = load_sources(std::slice::from_ref(&job));
        assert!(skipped.is_empty(), "{skipped:?}");
        let names: Vec<&str> = found.iter().map(|t| t.lock.trial.as_str()).collect();
        assert_eq!(names, ["hello__5af89adc"]);
        assert!(
            found[0].outputs.is_none(),
            "a linked outputs.tar is not used"
        );
    }

    /// The external agent's fingerprint was renamed; an earlier job's
    /// trials of it are the same install, not "the agent's install changed".
    #[test]
    fn the_external_agents_older_fingerprint_is_the_same_install() {
        let job =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/export/jobs/rust-claude");
        let (mut found, _) = load_sources(std::slice::from_ref(&job));
        let mut s = found.remove(0);
        let task = load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tasks/hello"));
        let fp = |f: &str| Some(f.to_string());
        s.lock.attempt.agent = "external".into();
        let mut t = s.lock.clone();
        let reason = |s: &SourceTrial, t: &LockTrial| {
            judge(&task, t, &s.agent_options, &s.options, s, &Rerun::default()).1
        };
        s.lock.attempt.agent_fingerprint = fp("external:karotte-backend-protocol");
        t.attempt.agent_fingerprint = fp(crate::external_agent::FINGERPRINT);
        assert!(
            !reason(&s, &t).contains("install changed"),
            "{}",
            reason(&s, &t)
        );
        t.attempt.agent_fingerprint = fp("external:something-else");
        assert!(reason(&s, &t).contains("install changed"));
    }

    #[test]
    fn inputs_digest_ignores_grading_and_location_but_not_what_the_agent_sees() {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tasks/hello");
        let d = tempfile::tempdir().unwrap();
        let a = d.path().join("a/hello");
        let b = d.path().join("b/hello");
        copy(&fixture, &a);
        copy(&fixture, &b);
        let base = load(&a);
        let digest = |t: &TaskPackage, at: &Attempt| inputs_digest(t, at);
        // Moving the folder changes nothing.
        assert_eq!(
            digest(&base, &Attempt::Nop),
            digest(&load(&b), &Attempt::Nop)
        );
        // The tests change: the task digest moves, the inputs do not.
        std::fs::write(b.join("verifier/test.sh"), "#!/bin/bash\nexit 1\n").unwrap();
        let tests_changed = load(&b);
        assert_ne!(base.digest, tests_changed.digest);
        assert_eq!(
            digest(&base, &Attempt::Nop),
            digest(&tests_changed, &Attempt::Nop)
        );
        // The oracle's files count for the oracle only.
        std::fs::write(b.join("oracle/solve.sh"), "#!/bin/bash\necho other\n").unwrap();
        let oracle_changed = load(&b);
        assert_eq!(
            digest(&base, &Attempt::Nop),
            digest(&oracle_changed, &Attempt::Nop)
        );
        assert_ne!(
            digest(&base, &Attempt::Oracle),
            digest(&oracle_changed, &Attempt::Oracle)
        );
        // The image's files count.
        std::fs::write(b.join("sandbox/Dockerfile"), "FROM ubuntu:22.04\n").unwrap();
        assert_ne!(
            digest(&base, &Attempt::Nop),
            digest(&load(&b), &Attempt::Nop)
        );
        // So does the instruction.
        let mut reworded = base.clone();
        reworded.instruction.push_str("\nAlso this.");
        assert_ne!(
            digest(&base, &Attempt::Nop),
            digest(&reworded, &Attempt::Nop)
        );
    }
}
