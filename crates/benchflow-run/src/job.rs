//! A job: many trials run together, in one folder, resumable.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use benchflow_task::TaskPackage;
use chrono::Local;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

use crate::agent::Attempt;
use crate::config::{Rerun, RunOptions};
use crate::events::Event;
use crate::metrics::{Rate, TaskStats, pass_at_k_table};
use crate::outcome::Status;
use crate::record::{TrialRecord, write_json};
use crate::trial::{self, Runtime, TrialSpec};

/// What to run.
#[derive(Clone)]
pub struct JobSpec {
    /// Each task with its attempts: one model agent for `bench run`, or
    /// `bench test`'s mix of oracle, nop, controls, and probes.
    pub units: Vec<(Arc<TaskPackage>, Vec<Attempt>)>,
    /// Trials per (task, attempt).
    pub trials: usize,
    pub jobs_dir: PathBuf,
    /// Use this job name (resumed when it exists).
    pub job_name: Option<String>,
    /// Never resume; start a new job.
    pub fresh: bool,
    pub concurrency: usize,
    /// Where the tasks came from, for evaluation.json.
    pub source: Value,
    /// Earlier trials to bring in instead of running (`bench run --diff`).
    pub carry: Vec<crate::diff::Carry>,
}

/// A finished job.
pub struct JobOutcome {
    pub dir: PathBuf,
    pub name: String,
    /// Every trial: the ones run now and the ones reused.
    pub records: Vec<TrialRecord>,
    /// How many came from an earlier run of this job, or were copied from
    /// an earlier job (`carried_reused`).
    pub reused: usize,
    /// Of `reused`, how many were copied from earlier jobs (`--diff`).
    pub carried_reused: usize,
    /// How many were graded again from an earlier job's saved outputs.
    pub carried_regraded: usize,
    /// Trials not started because a login ran out or the run was stopped.
    pub not_started: usize,
    pub summary: Value,
    pub resumed: bool,
    /// The trials whose agent did not run now (kept from an earlier run of
    /// this job, copied, or regraded), by name: their usage was spent
    /// earlier.
    pub earlier: std::collections::BTreeSet<String>,
}

/// What a run of a job would do with its folder, worked out as [`run`]
/// does but writing nothing (`bench run --dry-run`).
pub struct Preview {
    pub dir: PathBuf,
    /// Whether it would resume an earlier job.
    pub resumed: bool,
    /// Trials that job finished and the run would keep.
    pub kept: usize,
    /// Trials the job plans.
    pub total: usize,
}

/// Preview a run of `spec` with `opts`: the folder it would use, and how
/// many of the trials there it would keep.
pub fn preview(spec: &JobSpec, opts: &RunOptions) -> Preview {
    let (dir, resumed) = job_dir(spec, &fingerprint(spec));
    let name = dir
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    let (mut kept, mut total) = (0, 0);
    for (task, attempts) in &spec.units {
        for attempt in attempts {
            for i in 0..spec.trials.max(1) {
                total += 1;
                if !resumed {
                    continue;
                }
                let tname = trial_name(&name, task, attempt, i);
                let t = TrialSpec {
                    task: task.clone(),
                    attempt: attempt.clone(),
                    dir: dir.join(&tname),
                    name: tname,
                };
                if load_finished(&t, &opts.rerun, opts).is_some() {
                    kept += 1;
                }
            }
        }
    }
    Preview {
        dir,
        resumed,
        kept,
        total,
    }
}

/// The job folder to use: a named one, the latest matching one (resume), or
/// a new timestamped one. Returns the folder and whether it is resumed.
pub fn job_dir(spec: &JobSpec, fingerprint: &str) -> (PathBuf, bool) {
    if let Some(name) = &spec.job_name {
        let dir = spec.jobs_dir.join(name);
        let resumed = dir.join("evaluation.json").is_file();
        return (dir, resumed);
    }
    if !spec.fresh
        && let Some(latest) = latest_run(&spec.jobs_dir, fingerprint)
    {
        return (latest, true);
    }
    let name = Local::now().format("%Y-%m-%d__%H-%M-%S").to_string();
    let mut dir = spec.jobs_dir.join(&name);
    let mut n = 1;
    while dir.exists() {
        dir = spec.jobs_dir.join(format!("{name}-{n}"));
        n += 1;
    }
    (dir, false)
}

/// The alphabetically last job folder (timestamps sort in time order).
pub fn latest_job(jobs_dir: &Path) -> Option<PathBuf> {
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(jobs_dir)
        .ok()?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.join("evaluation.json").is_file() || p.join("summary.json").is_file())
        .collect();
    dirs.sort();
    dirs.pop()
}

/// The latest job a `bench run` with this fingerprint could resume: the
/// newest job folder with the same settings, not only the newest folder, so
/// a command run again after another job still resumes its own. Regrades
/// (`<job>-regrade-<n>`, which sort after their job) are passed over.
fn latest_run(jobs_dir: &Path, fingerprint: &str) -> Option<PathBuf> {
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(jobs_dir)
        .ok()?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.join("evaluation.json").is_file())
        .collect();
    dirs.sort();
    dirs.into_iter().rev().find_map(|d| {
        let v: Value =
            serde_json::from_str(&std::fs::read_to_string(d.join("evaluation.json")).ok()?).ok()?;
        let same = v
            .pointer("/engine/fingerprint")
            .and_then(Value::as_str)
            .is_some_and(|f| f == fingerprint);
        (v.get("regrade_of").is_none() && same).then_some(d)
    })
}

/// The file a running `bench run` holds locked in its job folder.
pub const RUNNING_FILE: &str = ".bench-run.lock";

/// Whether a live process holds `dir`'s run lock (a `bench run` or `bench
/// regrade` still running the job). Probes with a shared lock it drops at
/// once, which [`hold`] tolerates.
pub fn is_running(dir: &Path) -> bool {
    let Ok(file) = std::fs::File::open(dir.join(RUNNING_FILE)) else {
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

/// Lock the job folder for this process: trial folders are named the same
/// in every run of a job, so two runs at once would write the same trials.
/// The lock goes with the process, so a crashed run leaves nothing to clear.
pub fn hold(dir: &Path) -> std::io::Result<std::fs::File> {
    let path = dir.join(RUNNING_FILE);
    let file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .read(true)
        .open(&path)?;
    // `bench view` probes the lock with a shared lock it drops at once, so a
    // refusal is retried briefly before it is believed.
    let mut tries = 0;
    let locked = loop {
        match file.try_lock() {
            Err(std::fs::TryLockError::WouldBlock) if tries < 10 => {
                tries += 1;
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
            other => break other,
        }
    };
    match locked {
        Ok(()) => {
            let _ = file.set_len(0);
            let _ = std::io::Write::write_all(
                &mut &file,
                format!("{}\n", std::process::id()).as_bytes(),
            );
            Ok(file)
        }
        Err(std::fs::TryLockError::WouldBlock) => {
            let pid = std::fs::read_to_string(&path).unwrap_or_default();
            let pid = pid.trim();
            let by = if pid.is_empty() {
                String::new()
            } else {
                format!(" (process {pid})")
            };
            Err(std::io::Error::new(
                std::io::ErrorKind::WouldBlock,
                format!(
                    "another bench run{by} is running the job in {}; wait for it to finish, or name a new job with --job",
                    dir.display()
                ),
            ))
        }
        Err(std::fs::TryLockError::Error(e)) => Err(e),
    }
}

/// A stable name for trial `index` of `attempt` on `task` in `job`.
pub fn trial_name(job: &str, task: &TaskPackage, attempt: &Attempt, index: usize) -> String {
    let mut h = Sha256::new();
    h.update(job.as_bytes());
    h.update(b"\0");
    h.update(task.id.as_bytes());
    h.update(b"\0");
    h.update(attempt.label().as_bytes());
    h.update(b"\0");
    h.update(index.to_le_bytes());
    let hex = hex::encode(h.finalize());
    let base: String = task.name.chars().take(48).collect();
    format!("{base}__{}", &hex[..8])
}

/// What decides whether an earlier job can be resumed: the attempts, the
/// models, the tasks (by digest), and the trial count.
pub fn fingerprint(spec: &JobSpec) -> String {
    let mut h = Sha256::new();
    let mut units: Vec<String> = spec
        .units
        .iter()
        .map(|(t, attempts)| {
            let labels: Vec<String> = attempts
                .iter()
                .map(|a| format!("{}:{}", a.label(), a.model().unwrap_or_default()))
                .collect();
            format!("{}|{}", t.digest, labels.join(","))
        })
        .collect();
    units.sort_unstable();
    for u in units {
        h.update(u.as_bytes());
        h.update(b"\0");
    }
    h.update(spec.trials.to_le_bytes());
    hex::encode(h.finalize())[..16].to_string()
}

/// Run a job: plan its trials, skip the ones an earlier run finished, run
/// the rest with bounded concurrency, and write the job's summary files.
pub async fn run(spec: JobSpec, rt: Arc<Runtime>) -> std::io::Result<JobOutcome> {
    let started = Instant::now();
    let fp = fingerprint(&spec);
    let (dir, resumed) = job_dir(&spec, &fp);
    std::fs::create_dir_all(&dir)?;
    let _running = hold(&dir)?;
    let name = dir
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    write_json(
        &dir.join("evaluation.json"),
        &evaluation_json(&spec, &rt, &name, &fp),
    )?;

    // Plan.
    let mut planned: Vec<TrialSpec> = Vec::new();
    let mut carried: Vec<Option<&crate::diff::Carry>> = Vec::new();
    for (task, attempts) in &spec.units {
        for attempt in attempts {
            for i in 0..spec.trials.max(1) {
                let tname = trial_name(&name, task, attempt, i);
                carried.push(
                    spec.carry.iter().find(|c| {
                        c.task_id == task.id && c.agent == attempt.label() && c.index == i
                    }),
                );
                planned.push(TrialSpec {
                    task: task.clone(),
                    attempt: attempt.clone(),
                    dir: dir.join(&tname),
                    name: tname,
                });
            }
        }
    }
    let mut lock = crate::lock::build(
        crate::lock::LockHeader {
            job_name: &name,
            fingerprint: &fp,
            source: &spec.source,
            trials_per_task: spec.trials.max(1),
            concurrency: spec.concurrency,
            regrade_of: None,
        },
        &planned,
        &rt.options,
    );
    for (lt, c) in lock.trials.iter_mut().zip(&carried) {
        lt.carried_from = c.map(|c| crate::lock::LockCarry {
            action: c.action.as_str().into(),
            trial_dir: c.dir.clone(),
        });
    }
    write_lock(&dir, &lock)?;
    // Reuse finished trials, except the kinds asked to run again; then
    // bring in earlier jobs' trials, copied or to be regraded.
    let scratch = tempfile::tempdir()?;
    let (mut carried_reused, mut carried_regraded) = (0, 0);
    let mut reused_records = Vec::new();
    let mut earlier = std::collections::BTreeSet::new();
    let mut todo: Vec<(TrialSpec, Option<trial::Regrade>)> = Vec::new();
    for (t, carry) in planned.into_iter().zip(carried) {
        if let Some(rec) = load_finished(&t, &rt.options.rerun, &rt.options) {
            reused_records.push(rec);
            continue;
        }
        set_aside(&dir, &t.dir);
        match carry {
            Some(c) if c.action == crate::diff::Action::Reuse => {
                if let Some(rec) = copy_trial(c, &t)
                    .ok()
                    .and_then(|_| load_finished(&t, &Rerun::default(), &rt.options))
                {
                    reused_records.push(rec);
                    carried_reused += 1;
                    continue;
                }
                // The copy did not read back: run it.
                let _ = std::fs::remove_dir_all(&t.dir);
                todo.push((t, None));
            }
            Some(c) if c.action == crate::diff::Action::Regrade => {
                match regrade_input(c, &t, scratch.path()) {
                    Ok(r) => {
                        earlier.insert(t.name.clone());
                        // The agent does not run: no agent layers.
                        let spec = TrialSpec {
                            attempt: Attempt::Nop,
                            ..t
                        };
                        todo.push((spec, Some(r)));
                        carried_regraded += 1;
                    }
                    Err(_) => todo.push((t, None)),
                }
            }
            _ => todo.push((t, None)),
        }
    }
    let reused = reused_records.len();
    earlier.extend(reused_records.iter().map(|r| r.rollout_name.clone()));
    rt.events.send(Event::JobStarted {
        dir: dir.clone(),
        total: todo.len() + reused,
        reused,
        resumed,
    });

    // Run.
    let sem = Arc::new(Semaphore::new(spec.concurrency.max(1)));
    let halt = Arc::new(AtomicBool::new(false));
    let mut set = JoinSet::new();
    let mut not_started = 0usize;
    for (t, regrade) in todo {
        let sem = sem.clone();
        let rt = rt.clone();
        let halt = halt.clone();
        // Each image is released after the last trial that needs it.
        let image_key = crate::images::key_for(
            &t.task,
            &t.attempt,
            &rt.provider.capabilities(),
            &rt.options.sandbox_user,
        );
        let job_dir = dir.clone();
        rt.images.expect(&image_key);
        set.spawn(async move {
            let rec = run_counted(&t, regrade, &rt, &halt, &sem, &job_dir).await;
            rt.images.finished(&image_key, rt.provider.as_ref()).await;
            rec
        });
    }
    let mut records = reused_records;
    while let Some(res) = set.join_next().await {
        match res {
            Ok(Some(rec)) => records.push(rec),
            Ok(None) => not_started += 1,
            Err(e) => {
                // A panic in a trial is a BenchFlow bug; keep the run going.
                eprintln!("benchflow: a trial task failed: {e}");
                not_started += 1;
            }
        }
    }
    records.sort_by(|a, b| a.rollout_name.cmp(&b.rollout_name));

    let summary = summary_json(
        &name,
        &JobLabels {
            agent: attempt_labels(&spec),
            model: first_model(&spec),
            concurrency: spec.concurrency,
        },
        &rt,
        &records,
        started.elapsed().as_secs_f64(),
        not_started,
    );
    write_results(&dir, &summary, &records)?;
    Ok(JobOutcome {
        dir,
        name,
        records,
        reused,
        carried_reused,
        carried_regraded,
        not_started,
        summary,
        resumed,
        earlier,
    })
}

/// A trial an earlier run finished (scored, or unscored for a task reason),
/// read back from its result.json, unless `rerun` asks for it again.
fn load_finished(t: &TrialSpec, rerun: &Rerun, opts: &RunOptions) -> Option<TrialRecord> {
    let text = std::fs::read_to_string(t.dir.join("result.json")).ok()?;
    let v: Value = serde_json::from_str(&text).ok()?;
    let status: Status = serde_json::from_value(v.get("status")?.clone()).ok()?;
    let rec = crate::read::record_from_json(&v, &t.dir)?;
    // A job named with `--job` resumes whatever ran under that name, so the
    // trial must be of this task (same digest), agent, and model, run with
    // the same agent options (`--ak`) and effort, skills, and loaded session.
    let model = t.attempt.model();
    if rec.task_digest != t.task.digest
        || rec.agent != t.attempt.label()
        || (model.is_some() && rec.model != model)
        || !same_settings(&t.dir, &t.attempt, opts)
    {
        return None;
    }
    let reuse = match status {
        Status::Scored => !(rerun.failed && rec.passed != Some(true)),
        Status::Unscored => !rerun.unscored,
        // The same unexplained loss twice in a row: kept, so code inside a
        // sandbox cannot buy a rerun on every resume.
        Status::Errored => !rerun.errored && rec.problem.as_ref().is_some_and(|p| p.rerun_capped),
        _ => false,
    };
    reuse.then_some(rec)
}

/// Whether the trial in `dir` ran with `attempt`'s agent options and effort,
/// and the run's skills and loaded session, as its config.json recorded
/// them (a trial from before one was recorded counts as having none).
fn same_settings(dir: &std::path::Path, attempt: &Attempt, opts: &RunOptions) -> bool {
    let config: Value = std::fs::read_to_string(dir.join("config.json"))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or(Value::Null);
    let options = config
        .get("agent_options")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let effort = config
        .get("reasoning_effort")
        .and_then(Value::as_str)
        .map(str::to_string);
    // A trial with no skills record had none to install, so whether the
    // task's were left out made no difference to it.
    let skills = config.get("skills").filter(|v| !v.is_null());
    let run: Vec<PathBuf> = skills
        .and_then(|s| serde_json::from_value(s.get("run")?.clone()).ok())
        .unwrap_or_default();
    let task_skills = skills
        .and_then(|s| s.get("task_skills")?.as_bool())
        .unwrap_or(opts.task_skills);
    let loaded = config
        .get("load_trajectory")
        .and_then(|l| l.get("path")?.as_str())
        .map(PathBuf::from);
    options == attempt.options()
        && effort == attempt.effort()
        && run == opts.skills
        && task_skills == opts.task_skills
        && loaded == opts.load_trajectory
}

/// Copy an earlier job's trial folder in as `t`, renamed, with result.json
/// naming where it came from.
fn copy_trial(c: &crate::diff::Carry, t: &TrialSpec) -> std::io::Result<()> {
    for e in walkdir::WalkDir::new(&c.dir).follow_links(false) {
        let e = e.map_err(std::io::Error::other)?;
        let rel = e
            .path()
            .strip_prefix(&c.dir)
            .map_err(std::io::Error::other)?;
        let dest = t.dir.join(rel);
        if e.file_type().is_dir() {
            std::fs::create_dir_all(&dest)?;
        } else if e.file_type().is_file() {
            std::fs::copy(e.path(), &dest)?;
        }
    }
    let path = t.dir.join("result.json");
    let mut v: Value =
        serde_json::from_str(&std::fs::read_to_string(&path)?).map_err(std::io::Error::other)?;
    if let Some(m) = v.as_object_mut() {
        m.insert("rollout_name".into(), json!(t.name));
        m.insert(
            "carried_from".into(),
            json!({"action": "reuse", "trial_dir": c.dir}),
        );
    }
    write_json(&path, &v)
}

/// What regrading an earlier trial into `t` needs: its outputs as one
/// archive, and what its agent did.
fn regrade_input(
    c: &crate::diff::Carry,
    t: &TrialSpec,
    scratch: &Path,
) -> std::io::Result<trial::Regrade> {
    let archive = match &c.outputs {
        Some(crate::regrade::Outputs::Archive(p)) => p.clone(),
        Some(crate::regrade::Outputs::Folder(f)) => {
            let p = scratch.join(format!("{}.tar", t.name));
            crate::regrade::pack_folder(f, &p)?;
            p
        }
        None => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "no saved outputs",
            ));
        }
    };
    Ok(trial::Regrade {
        archive,
        agent: c.record.agent.clone(),
        model: c.record.model.clone(),
        output: crate::regrade::recorded_agent(&c.record),
        agent_files: c.dir.clone(),
        allowed: crate::regrade::restore_paths(&t.task, &c.dir)
            .map_err(|why| std::io::Error::new(std::io::ErrorKind::InvalidData, why))?,
    })
}

/// Move an earlier attempt at a trial out of the way before it runs again,
/// into `<job>/.rerun/<trial>-<n>/`, so no stale file is mixed into the new
/// one and the old result is kept. Any folder is moved, with or without a
/// result.json: one left by a crash or an unsaved record (R107) holds the
/// lost attempt's verifier output, outputs, and logs.
fn set_aside(job: &Path, trial_dir: &Path) {
    move_aside(job, trial_dir, ".rerun");
}

/// Move `trial_dir` into `<job>/<store>/<trial>-<n>/`; where it went.
pub(crate) fn move_aside(job: &Path, trial_dir: &Path, store: &str) -> Option<PathBuf> {
    if !trial_dir.is_dir() {
        return None;
    }
    let name = trial_dir.file_name()?.to_string_lossy().to_string();
    let store = job.join(store);
    std::fs::create_dir_all(&store).ok()?;
    let mut n = 1;
    while store.join(format!("{name}-{n}")).exists() {
        n += 1;
    }
    let to = store.join(format!("{name}-{n}"));
    std::fs::rename(trial_dir, &to).ok()?;
    Some(to)
}

/// Write `lock.json` into a job folder.
pub fn write_lock(dir: &Path, lock: &crate::lock::Lock) -> std::io::Result<()> {
    write_json(
        &dir.join(crate::lock::FILE_NAME),
        &serde_json::to_value(lock).map_err(std::io::Error::other)?,
    )
}

/// The distinct attempt labels of a job, in order.
fn attempt_labels(spec: &JobSpec) -> String {
    let mut labels: Vec<String> = Vec::new();
    for (_, attempts) in &spec.units {
        for a in attempts {
            let l = a.label();
            if !labels.contains(&l) {
                labels.push(l);
            }
        }
    }
    labels.join(",")
}

fn first_model(spec: &JobSpec) -> Option<String> {
    spec.units
        .iter()
        .flat_map(|(_, a)| a.iter())
        .find_map(Attempt::model)
}

fn evaluation_json(spec: &JobSpec, rt: &Runtime, name: &str, fp: &str) -> Value {
    json!({
        "schema_version": 1,
        "job_name": name,
        "tasks_dir": spec.source.get("path").cloned().unwrap_or(Value::Null),
        "source": spec.source,
        "config": {
            "agent": attempt_labels(spec),
            "model": first_model(spec),
            "environment": rt.provider.name(),
            "concurrency": spec.concurrency,
            "trials": spec.trials,
            "sandbox_user": rt.options.sandbox_user,
            "agent_idle_timeout": rt.options.idle_timeout.as_secs(),
            "timeout_multiplier": rt.options.timeout_multiplier,
            // Whether reward files were read as BenchFlow 0.8's lenient
            // mode reads them (BENCHFLOW_REWARD_LENIENT): the same file can
            // score in one mode and not the other.
            "reward_lenient": benchflow_task::reward::lenient_from_env(),
            "tasks": spec.units.iter().map(|(t, _)| json!({"name": t.name, "id": t.id, "digest": t.digest, "path": t.root})).collect::<Vec<_>>(),
        },
        "engine": { "name": "benchflow-rs", "version": env!("CARGO_PKG_VERSION"), "fingerprint": fp },
    })
}

/// Per-task statistics for the model-agent trials (controls excluded).
pub fn task_stats(records: &[TrialRecord]) -> BTreeMap<String, TaskStats> {
    let mut m: BTreeMap<String, TaskStats> = BTreeMap::new();
    for r in records.iter().filter(|r| !r.stopped()) {
        let s = m.entry(r.task_name.clone()).or_default();
        s.trials += 1;
        if r.status == Status::Scored {
            s.scored += 1;
            s.reward_sum += r.reward.unwrap_or(0.0);
            if r.passed == Some(true) {
                s.passed += 1;
            }
        }
    }
    m
}

/// Write summary.json, the trainer files (`verifiers.jsonl`, `adp.jsonl`),
/// and results.jsonl.
pub(crate) fn write_results(
    dir: &Path,
    summary: &Value,
    records: &[TrialRecord],
) -> std::io::Result<()> {
    write_json(&dir.join("summary.json"), summary)?;
    for e in crate::trainer::write_job(dir) {
        tracing::warn!("{e}");
    }
    let rows: String = records
        .iter()
        .map(|r| {
            format!(
                "{}\n",
                serde_json::to_string(&r.results_row()).unwrap_or_default()
            )
        })
        .collect();
    crate::record::write_atomic(&dir.join("results.jsonl"), rows.as_bytes())
}

/// What a job's summary says ran it.
pub(crate) struct JobLabels {
    /// The distinct attempt labels, comma-separated.
    pub agent: String,
    pub model: Option<String>,
    pub concurrency: usize,
}

pub(crate) fn summary_json(
    name: &str,
    labels: &JobLabels,
    rt: &Runtime,
    records: &[TrialRecord],
    elapsed: f64,
    not_started: usize,
) -> Value {
    // Trials the user stopped are counted apart.
    let tally = crate::record::Tally::of(records);
    let (total, passed, scored, stopped) =
        (tally.counted, tally.passed, tally.scored, tally.stopped);
    let failed = scored - passed;
    let errored = records
        .iter()
        .filter(|r| r.status == Status::Errored)
        .count();
    let unscored = records
        .iter()
        .filter(|r| r.status == Status::Unscored)
        .count();
    let rewards: Vec<f64> = records.iter().filter_map(|r| r.reward).collect();
    let mean_reward =
        (!rewards.is_empty()).then(|| rewards.iter().sum::<f64>() / rewards.len() as f64);
    let rate = Rate::new(passed, total);
    let rate_excl = Rate::new(passed, scored);
    let mut by_cause: BTreeMap<String, Value> = BTreeMap::new();
    for r in records {
        if let Some(p) = &r.problem {
            let e = by_cause.entry(p.code.clone()).or_insert_with(|| {
                json!({"code": p.code, "owner": p.owner, "message": p.message, "next": p.next, "trials": []})
            });
            if let Some(arr) = e.get_mut("trials").and_then(Value::as_array_mut) {
                arr.push(json!(r.rollout_name));
            }
        }
    }
    let usage_in: u64 = records.iter().map(|r| r.usage.input_tokens).sum();
    let usage_out: u64 = records.iter().map(|r| r.usage.output_tokens).sum();
    let cache_read: u64 = records.iter().map(|r| r.usage.cache_read_tokens).sum();
    let cache_write: u64 = records.iter().map(|r| r.usage.cache_write_tokens).sum();
    let costs: Vec<f64> = records.iter().filter_map(|r| r.usage.cost_usd).collect();
    let stats = task_stats(records);
    // Scored trials whose logs show the environment failing.
    let environment_warnings: Vec<Value> = records
        .iter()
        .filter(|r| !r.environment_warnings.is_empty())
        .map(|r| json!({"trial": r.rollout_name, "task": r.task_name, "warnings": r.environment_warnings}))
        .collect();
    json!({
        "job_name": name,
        "agent": labels.agent,
        "model": labels.model,
        "environment": rt.provider.name(),
        "concurrency": labels.concurrency,
        "total": total,
        "passed": passed,
        "failed": failed,
        "errored": errored,
        "verifier_errored": unscored,
        "unscored": unscored,
        "pass": passed,
        "fail": failed,
        "error": errored,
        "not_started": not_started,
        "stopped": stopped,
        // Trials stopped or never started: the score is not the whole job's.
        "partial": stopped + not_started > 0,
        // Null when no trial counted (all stopped), not a 0.0% score.
        "score": (total > 0).then(|| format!("{:.1}%", rate.value * 100.0)),
        "score_ratio": rate.value,
        "score_interval": [rate.low, rate.high],
        "score_excl_errors": format!("{:.1}%", rate_excl.value * 100.0),
        "score_excl_errors_ratio": rate_excl.value,
        "mean_reward": mean_reward,
        "elapsed_sec": elapsed,
        "total_input_tokens": usage_in,
        "total_output_tokens": usage_out,
        "total_cache_read_tokens": cache_read,
        "total_cache_creation_tokens": cache_write,
        "total_tokens": usage_in + usage_out + cache_read + cache_write,
        "total_cost_usd": (!costs.is_empty()).then(|| costs.iter().sum::<f64>()),
        "pass_at_k": pass_at_k_table(&stats),
        "failures_by_cause": by_cause.into_values().collect::<Vec<_>>(),
        "environment_warnings": environment_warnings,
        // Scored although locked_system_dirs failed; the scores stand.
        "graded_unlocked_system_dirs": tally.graded_unlocked,
        // Rewards outside [0, 1], counted as written in mean_reward.
        "rewards_out_of_range": records.iter().filter(|r| !r.stopped()).filter_map(|r| {
            r.reward_out_of_range().map(|x| json!({"trial": r.rollout_name, "reward": x}))
        }).collect::<Vec<_>>(),
        "engine": { "name": "benchflow-rs", "version": env!("CARGO_PKG_VERSION") },
    })
}

/// Run one planned trial, with its retries, once a concurrency slot frees.
/// `None` when the run stopped first.
async fn run_counted(
    t: &TrialSpec,
    regrade: Option<trial::Regrade>,
    rt: &Arc<Runtime>,
    halt: &Arc<AtomicBool>,
    sem: &Arc<Semaphore>,
    job_dir: &Path,
) -> Option<TrialRecord> {
    let _permit = sem.clone().acquire_owned().await.ok()?;
    if halt.load(Ordering::SeqCst) || !rt.starting_new() {
        return None;
    }
    let mut attempt_no = 0;
    loop {
        let rec = match &regrade {
            Some(r) => trial::regrade(t, rt, r.clone()).await,
            None => trial::run(t, rt).await,
        };
        let transient =
            rec.status == Status::Errored && rec.problem.as_ref().is_some_and(|p| p.transient);
        if rec.problem.as_ref().is_some_and(|p| p.code == "R213") {
            halt.store(true, Ordering::SeqCst);
        }
        if transient
            && attempt_no < rt.options.retries
            && !halt.load(Ordering::SeqCst)
            && rt.starting_new()
        {
            attempt_no += 1;
            rt.events.send(Event::Retrying {
                trial: t.name.clone(),
                attempt: attempt_no,
                reason: rec
                    .problem
                    .as_ref()
                    .map(|p| p.message.clone())
                    .unwrap_or_default(),
            });
            // A drain, a stop, or a spent login during the wait ends it:
            // the errored attempt stands (convergence reviews E9-E-B 2,
            // E9-E-A).
            if !wait_to_retry(retry_delay(&rec, attempt_no, &rt.options), || {
                !halt.load(Ordering::SeqCst) && rt.starting_new()
            })
            .await
            {
                return Some(rec);
            }
            // Start the retry from a clean folder, keeping this one.
            set_aside(job_dir, &t.dir);
            continue;
        }
        return Some(rec);
    }
}

/// Wait `delay` before a retry, checking `still` (no drain, stop or spent
/// login) a few times a second. False, at once, when it no longer holds:
/// the retry must not start.
async fn wait_to_retry(delay: Duration, still: impl Fn() -> bool) -> bool {
    let end = std::time::Instant::now() + delay;
    loop {
        if !still() {
            return false;
        }
        let now = std::time::Instant::now();
        if now >= end {
            return true;
        }
        tokio::time::sleep((end - now).min(Duration::from_millis(200))).await;
    }
}

/// How long to wait before an in-run retry: a few seconds, or, after a
/// sandbox loss the backend did not confirm, long enough that a short
/// outage of the sandbox's own channel does not catch the retry too (the
/// second such loss is capped).
fn retry_delay(rec: &TrialRecord, attempt_no: u32, options: &RunOptions) -> Duration {
    let base = Duration::from_secs(3 * attempt_no as u64);
    if rec
        .problem
        .as_ref()
        .is_some_and(|p| p.loss_confirmed == Some(false))
    {
        base.max(Duration::from_secs(options.unexplained_loss_retry_secs))
    } else {
        base
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Convergence reviews E9-E-B 2 and E9-E-A: the two-minute wait before
    /// a retry ignored a drain or a stop, so a drained run still started
    /// the retry. The wait ends as soon as the run no longer starts trials.
    #[tokio::test]
    async fn a_drain_ends_the_wait_before_a_retry() {
        let stopped = std::sync::atomic::AtomicBool::new(false);
        let waiting = wait_to_retry(Duration::from_secs(120), || !stopped.load(Ordering::SeqCst));
        let drain = async {
            tokio::time::sleep(Duration::from_millis(300)).await;
            stopped.store(true, Ordering::SeqCst);
        };
        let (go, ()) = tokio::time::timeout(Duration::from_secs(5), async {
            tokio::join!(waiting, drain)
        })
        .await
        .expect("the wait ends within moments of the drain");
        assert!(!go, "no retry after a drain");
        assert!(wait_to_retry(Duration::ZERO, || true).await);
    }

    /// Convergence review E9-D-A 3: the retry after an unconfirmed loss
    /// came 3 s later, so one short proxy outage caught both attempts and
    /// capped the trial. It now waits minutes; other retries keep 3 s.
    #[test]
    fn a_retry_after_an_unexplained_loss_waits_minutes() {
        let o = RunOptions::default();
        let mut rec = crate::read::record_from_json(
            &serde_json::json!({"task_name": "t", "rollout_name": "t__1", "status": "errored"}),
            std::path::Path::new("/nonexistent"),
        )
        .unwrap();
        let mut p = crate::outcome::Problem::new(
            "R103",
            crate::outcome::Owner::Sandbox,
            crate::outcome::Phase::Verify,
            "lost",
            "retry",
        );
        rec.problem = Some(p.clone());
        assert_eq!(retry_delay(&rec, 1, &o), Duration::from_secs(3));
        p.loss_confirmed = Some(true);
        rec.problem = Some(p.clone());
        assert_eq!(retry_delay(&rec, 1, &o), Duration::from_secs(3));
        p.loss_confirmed = Some(false);
        rec.problem = Some(p);
        assert_eq!(retry_delay(&rec, 1, &o), Duration::from_secs(120));
    }

    /// Review of rust/d4-chaos (7cd50c84 tested only its helper): the job's
    /// results.jsonl goes through the temporary file, so a write that fails
    /// leaves the old file whole.
    #[test]
    fn a_failed_results_write_leaves_the_old_file_whole() {
        let dir = tempfile::tempdir().unwrap();
        write_results(dir.path(), &serde_json::json!({}), &[]).unwrap();
        std::fs::write(dir.path().join("results.jsonl"), "{\"kept\":1}\n").unwrap();
        std::fs::create_dir(dir.path().join("results.jsonl.tmp")).unwrap();
        assert!(write_results(dir.path(), &serde_json::json!({}), &[]).is_err());
        assert_eq!(
            std::fs::read_to_string(dir.path().join("results.jsonl")).unwrap(),
            "{\"kept\":1}\n"
        );
    }

    /// A named job resumed with other agent options or effort runs its
    /// trials again; a trial recorded before either was recorded counts as
    /// having none.
    #[test]
    fn resuming_needs_the_same_agent_options_and_effort() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config.json");
        let o = RunOptions::default();
        std::fs::write(&config, r#"{"agent": "oracle"}"#).unwrap();
        assert!(same_settings(dir.path(), &Attempt::Oracle, &o));
        std::fs::write(
            &config,
            r#"{"agent_options": {}, "reasoning_effort": null}"#,
        )
        .unwrap();
        assert!(same_settings(dir.path(), &Attempt::Oracle, &o));
        std::fs::write(&config, r#"{"agent_options": {"max_turns": 20}}"#).unwrap();
        assert!(!same_settings(dir.path(), &Attempt::Oracle, &o));
        std::fs::write(&config, r#"{"reasoning_effort": "low"}"#).unwrap();
        assert!(!same_settings(dir.path(), &Attempt::Oracle, &o));
    }

    /// The run's skills and loaded session are settings too: a trial that
    /// ran with other ones runs again.
    #[test]
    fn resuming_needs_the_same_skills_and_loaded_session() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config.json");
        let same = |o: &RunOptions| same_settings(dir.path(), &Attempt::Oracle, o);
        let plain = RunOptions::default();
        let with_skill = RunOptions {
            skills: vec![PathBuf::from("/s/csv")],
            ..RunOptions::default()
        };
        let no_task = RunOptions {
            task_skills: false,
            ..RunOptions::default()
        };
        let loaded = RunOptions {
            load_trajectory: Some(PathBuf::from("/x/s.jsonl")),
            ..RunOptions::default()
        };
        std::fs::write(&config, r#"{"skills": null, "load_trajectory": null}"#).unwrap();
        assert!(same(&plain));
        assert!(
            same(&no_task),
            "a task without skills ran the same either way"
        );
        assert!(!same(&with_skill));
        assert!(!same(&loaded));
        std::fs::write(
            &config,
            r#"{"skills": {"run": ["/s/csv"], "task_skills": true}, "load_trajectory": {"path": "/x/s.jsonl"}}"#,
        )
        .unwrap();
        assert!(!same(&plain));
        let both = RunOptions {
            skills: vec![PathBuf::from("/s/csv")],
            load_trajectory: Some(PathBuf::from("/x/s.jsonl")),
            ..RunOptions::default()
        };
        assert!(same(&both));
        std::fs::write(&config, r#"{"skills": {"run": [], "task_skills": true}}"#).unwrap();
        assert!(!same(&no_task));
    }

    #[test]
    fn a_regrade_does_not_hide_its_job_from_resume() {
        let d = tempfile::tempdir().unwrap();
        let job = d.path().join("2026-10-10__09-00-00");
        let regrade = d.path().join("2026-10-10__09-00-00-regrade-1");
        for (dir, eval) in [
            (&job, json!({"engine": {"fingerprint": "fp"}})),
            (
                &regrade,
                json!({"regrade_of": job, "engine": {"fingerprint": "regrade:x"}}),
            ),
        ] {
            std::fs::create_dir_all(dir).unwrap();
            write_json(&dir.join("evaluation.json"), &eval).unwrap();
        }
        let spec = JobSpec {
            carry: Vec::new(),
            units: Vec::new(),
            trials: 1,
            jobs_dir: d.path().to_path_buf(),
            job_name: None,
            fresh: false,
            concurrency: 1,
            source: Value::Null,
        };
        assert_eq!(job_dir(&spec, "fp"), (job, true));
    }

    #[test]
    fn a_later_job_with_other_settings_does_not_hide_a_job_from_resume() {
        let d = tempfile::tempdir().unwrap();
        let first = d.path().join("2026-10-10__09-00-00");
        let other = d.path().join("2026-10-10__10-00-00");
        for (dir, fp) in [(&first, "fp"), (&other, "other")] {
            std::fs::create_dir_all(dir).unwrap();
            write_json(
                &dir.join("evaluation.json"),
                &json!({"engine": {"fingerprint": fp}}),
            )
            .unwrap();
        }
        let mut spec = JobSpec {
            carry: Vec::new(),
            units: Vec::new(),
            trials: 1,
            jobs_dir: d.path().to_path_buf(),
            job_name: None,
            fresh: false,
            concurrency: 1,
            source: Value::Null,
        };
        assert_eq!(job_dir(&spec, "fp"), (first, true));
        assert_eq!(job_dir(&spec, "other"), (other, true));
        let (new, resumed) = job_dir(&spec, "third");
        assert!(!resumed && new.parent() == Some(d.path()));
        spec.fresh = true;
        assert!(!job_dir(&spec, "fp").1);
    }
}
