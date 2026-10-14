//! Regrading: score a finished job's trials again with each task's current
//! verifier, from the outputs each trial saved, without running the agent.
//!
//! Each trial gets a fresh sandbox from its task's image, its outputs
//! restored at their own paths, and the verifier run there, the way a task
//! with a separate verifier is graded. The results go to a new job folder,
//! `<job>-regrade-<n>`, next to the original, whose evaluation.json and
//! lock.json name the job it regrades (`regrade_of`).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use benchflow_task::TaskPackage;
use serde::Serialize;
use serde_json::{Value, json};
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

use crate::agent::{AgentOutput, Attempt};
use crate::events::Event;
use crate::job::{self, JobLabels};
use crate::lock::{self, LockHeader};
use crate::outcome::Status;
use crate::read;
use crate::record::{TrialRecord, write_json};
use crate::trial::{self, Regrade, Runtime, TrialSpec};

/// A trial that can be regraded.
pub struct Regradable {
    /// The original trial folder.
    pub dir: PathBuf,
    pub record: TrialRecord,
    pub task: Arc<TaskPackage>,
    /// Its saved outputs: `outputs.tar`, or `outputs/` packed into one.
    pub outputs: Outputs,
}

/// Where a trial's saved outputs are.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outputs {
    Archive(PathBuf),
    Folder(PathBuf),
}

/// A trial that cannot be regraded, and why.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct NotRegradable {
    pub trial: String,
    pub task: String,
    pub reason: String,
    pub next: String,
}

/// The trials of a job, sorted into the ones that can be regraded and the
/// ones that cannot.
pub struct Plan {
    pub job: PathBuf,
    pub regradable: Vec<Regradable>,
    pub not_regradable: Vec<NotRegradable>,
    /// The original job's evaluation.json, when it has one.
    pub evaluation: Option<Value>,
}

/// Read a finished job and plan its regrade: each trial needs its task
/// folder (from config.json or lock.json) to still load, and saved outputs.
/// A task loads as [`benchflow_task::load::load_to_regrade`] loads it: a
/// service command Docker Compose cannot split (P014) does not stop a
/// regrade, which starts no service.
pub fn plan(job: &Path) -> std::io::Result<Plan> {
    plan_with(job, &|p| {
        benchflow_task::load::load_to_regrade(p)
            .map(Arc::new)
            .map_err(|d| {
                d.iter()
                    .find(|x| x.is_error())
                    .map(|x| x.message.clone())
                    .unwrap_or_else(|| "it does not load".into())
            })
    })
}

/// [`plan`], loading tasks with `load`.
pub fn plan_with(
    job: &Path,
    load: &dyn Fn(&Path) -> Result<Arc<TaskPackage>, String>,
) -> std::io::Result<Plan> {
    let view = read::read_job(job);
    let lock = read::read_lock(job).ok();
    let mut regradable = Vec::new();
    let mut not_regradable = Vec::new();
    let mut loaded: Vec<(PathBuf, Result<Arc<TaskPackage>, String>)> = Vec::new();
    for dir in read::trial_dirs(job) {
        let Some(record) = read::read_trial(&dir) else {
            continue;
        };
        let refuse = |reason: String, next: &str| NotRegradable {
            trial: record.rollout_name.clone(),
            task: record.task_name.clone(),
            reason,
            next: next.to_string(),
        };
        let Some(outputs) = saved_outputs(&dir) else {
            // A trial stopped before its sandbox started has nothing to grade,
            // whatever --save-outputs said.
            if record.sandbox_id.is_none() && record.status != Status::Scored {
                let reason = match &record.problem {
                    Some(p) => format!("it never started a sandbox ({})", p.code),
                    None => "it never started a sandbox".into(),
                };
                not_regradable.push(refuse(
                    reason,
                    "fix what stopped it (bench results names it), then run it again",
                ));
                continue;
            }
            not_regradable.push(refuse(
                "it saved no outputs".into(),
                "rerun it with --save-outputs (tasks that declare outputs or use a separate verifier save them anyway)",
            ));
            continue;
        };
        if let Some(changed) = outputs_changed(&outputs, &record.custody) {
            not_regradable.push(refuse(
                changed,
                "restore the trial's outputs/ folder from a copy, or run the trial again",
            ));
            continue;
        }
        let Some(task_path) = task_path(&dir, job, lock.as_ref(), &record) else {
            not_regradable.push(refuse(
                "its task folder is not recorded, or no longer exists".into(),
                "regrade from the folder the job ran in, or restore the task at the path in the trial's config.json",
            ));
            continue;
        };
        let task = match loaded.iter().find(|(p, _)| *p == task_path) {
            Some((_, t)) => t.clone(),
            None => {
                let t = load(&task_path);
                loaded.push((task_path.clone(), t.clone()));
                t
            }
        };
        match task {
            // Staged tasks were graded stage by stage, in the sandbox.
            Ok(task) if crate::features::regrade_refusal(&task).is_some() => {
                let p = crate::features::regrade_refusal(&task).expect("checked");
                not_regradable.push(refuse(p.message, &p.next));
                continue;
            }
            Ok(task) => {
                if let Some(reason) = stray_outputs(&task, &dir, &outputs) {
                    not_regradable.push(refuse(
                        reason,
                        "run the trial again: a regrade restores only regular files and folders inside the task's outputs, and these saved outputs hold more (an older bench saved its outputs archive unchecked, or the trial folder was changed after the run)",
                    ));
                    continue;
                }
                regradable.push(Regradable {
                    dir,
                    record,
                    task,
                    outputs,
                })
            }
            Err(e) => not_regradable.push(refuse(
                format!("its task at {} does not load: {e}", task_path.display()),
                &format!(
                    "run `bench check {}` and fix what it reports",
                    task_path.display()
                ),
            )),
        }
    }
    Ok(Plan {
        job: job.to_path_buf(),
        regradable,
        not_regradable,
        evaluation: view.evaluation,
    })
}

/// Why a trial's saved outputs cannot be restored for `task`'s verifier:
/// they are unpacked as root at `/`, so only what
/// [`crate::custody::filter_outputs`] keeps for the task's output paths may
/// be in them, the rule a live trial saved them by. `None` when they fit.
pub(crate) fn stray_outputs(task: &TaskPackage, dir: &Path, outputs: &Outputs) -> Option<String> {
    let allowed = match restore_paths(task, dir) {
        Ok(a) => a,
        Err(why) => return Some(why),
    };
    match outputs {
        Outputs::Archive(a) => match crate::custody::stray_entries(a, &allowed) {
            Err(e) => Some(format!("its saved outputs cannot be read: {e}")),
            // Folders above the outputs are dropped and say nothing: an
            // archive packed from `outputs/` holds them.
            Ok(f) => {
                let n = f.refused_count - f.folders_above;
                let first = f
                    .refused
                    .iter()
                    .find(|r| r.reason != crate::custody::ABOVE_OUTPUTS)
                    .map(|r| format!(" (the first, {}, is a {}: {})", r.path, r.kind, r.reason))
                    .unwrap_or_default();
                (n > 0).then(|| {
                    format!("its saved outputs hold {n} entries a verifier is not given{first}")
                })
            }
        },
        Outputs::Folder(f) => {
            // A folder from before the byte budget can name far more than
            // a restore takes (review F2-RA-1): one file copied per hard
            // link.
            let bytes = folder_bytes(f);
            if bytes > crate::custody::OUTPUT_BUDGET {
                return Some(format!(
                    "its saved outputs hold {bytes} bytes, past the {} a restore takes",
                    crate::custody::OUTPUT_BUDGET
                ));
            }
            let stray = crate::custody::stray_files(f, &allowed);
            stray.first().map(|first| {
                format!(
                    "its saved outputs hold {} path(s) outside the task's outputs ({}), or links",
                    stray.len(),
                    first
                )
            })
        }
    }
}

/// The paths a trial's saved outputs may restore for `task`'s verifier:
/// the task's output paths, in the folder the trial worked in. `Err` with
/// why when the folder its config records is one root's restore could use
/// to reach system or grading files.
pub(crate) fn restore_paths(task: &TaskPackage, dir: &Path) -> Result<Vec<String>, String> {
    // The folder the trial worked in: the task's when it declares one, else
    // as the trial's config recorded it (the image's WORKDIR); older trials
    // recorded none, and used the task's. A trial folder can come from
    // someone else, so a recorded folder that root's restore could use to
    // reach system or grading files is refused.
    let recorded = std::fs::read_to_string(dir.join("config.json"))
        .ok()
        .and_then(|s| serde_json::from_str::<Value>(&s).ok())
        .and_then(|v| v.get("workdir").and_then(Value::as_str).map(String::from))
        .filter(|w| !w.is_empty());
    let workdir = match (&task.sandbox.workdir, recorded) {
        (Some(declared), _) => declared.clone(),
        (None, Some(w)) => {
            // Only a task without declared outputs restores its workdir.
            if task.sandbox.outputs.is_empty()
                && let Some(why) = unsafe_restore_dir(task, &w)
            {
                return Err(format!(
                    "its config records the working folder {w:?}, which {why}"
                ));
            }
            w
        }
        (None, None) => task.workdir().to_string(),
    };
    Ok(trial::output_paths(task, &workdir)
        .into_iter()
        .map(|(p, _)| p)
        .collect())
}

/// Why outputs restored as root into `workdir` could reach files that are
/// not the agent's: a relative or unnormalized path, `/`, a system folder,
/// or a folder that holds or sits in BenchFlow's state, the logs, or the
/// task's verifier or oracle files. `None` when it is a plain working
/// folder.
fn unsafe_restore_dir(task: &TaskPackage, workdir: &str) -> Option<&'static str> {
    use std::path::Component;
    let p = Path::new(workdir);
    let parts: Vec<Component> = p.components().collect();
    if !p.is_absolute()
        || parts.len() < 2
        || parts[1..]
            .iter()
            .any(|c| !matches!(c, Component::Normal(_)))
    {
        return Some("is not a plain absolute folder");
    }
    const SYSTEM: &[&str] = &[
        "/bin", "/sbin", "/usr", "/lib", "/lib32", "/lib64", "/libx32", "/etc", "/boot", "/dev",
        "/proc", "/sys", "/run",
    ];
    let w = workdir.trim_end_matches('/');
    let within = |a: &str, b: &str| a == b || a.starts_with(&format!("{b}/"));
    if SYSTEM.iter().any(|s| within(w, s)) {
        return Some("is a system folder");
    }
    let mut guarded = vec![
        crate::hardening::STATE_DIR.to_string(),
        "/logs".to_string(),
        "/tests".to_string(),
        task.verifier.mount.trim_end_matches('/').to_string(),
    ];
    if let Some(o) = &task.oracle {
        guarded.push(o.mount.trim_end_matches('/').to_string());
    }
    if guarded.iter().any(|g| within(w, g) || within(g, w)) {
        return Some("holds or sits in BenchFlow's or the verifier's files");
    }
    None
}

/// Why a trial's saved `outputs/` folder is not what the run saved, by the
/// files and bytes custody.json recorded; `None` when they match, or when
/// the run recorded none (older jobs), as long as every file can be read.
/// Grading a folder with a file cut or deleted since measured the damage,
/// not the agent (E5: a regrade scored such a trial 0, with nothing saying
/// why).
fn outputs_changed(outputs: &Outputs, custody: &crate::record::Custody) -> Option<String> {
    let Outputs::Folder(folder) = outputs else {
        return None;
    };
    let (mut files, mut bytes) = (0u64, 0u64);
    for entry in walkdir::WalkDir::new(folder).min_depth(1) {
        // One unreadable file stopped the whole regrade as "the job folder
        // could not be written" (E5); it is this trial's alone.
        let unreadable = |what: &Path| {
            format!(
                "its saved outputs cannot all be read ({})",
                what.strip_prefix(folder).unwrap_or(what).display()
            )
        };
        let entry = match entry {
            Ok(entry) => entry,
            Err(e) => return Some(unreadable(e.path().unwrap_or(folder))),
        };
        if entry.file_type().is_file() {
            if std::fs::File::open(entry.path()).is_err() {
                return Some(unreadable(entry.path()));
            }
            files += 1;
            bytes += entry.metadata().map(|m| m.len()).unwrap_or(0);
        }
    }
    if custody.saved_files == 0 && custody.saved_bytes == 0 {
        return None;
    }
    (files != custody.saved_files || bytes != custody.saved_bytes).then(|| {
        format!(
            "its saved outputs changed since the run: {files} files and {bytes} bytes in outputs/, where the run saved {} files and {} bytes",
            custody.saved_files, custody.saved_bytes
        )
    })
}

/// A trial folder's saved outputs, when it has any.
pub fn saved_outputs(dir: &Path) -> Option<Outputs> {
    // Never through a link: a trial folder can come from someone else.
    let kind = |p: &Path| std::fs::symlink_metadata(p).ok().map(|m| m.file_type());
    let tar = dir.join("outputs.tar");
    if kind(&tar).is_some_and(|t| t.is_file()) {
        return Some(Outputs::Archive(tar));
    }
    let folder = dir.join("outputs");
    if !kind(&folder).is_some_and(|t| t.is_dir()) {
        return None;
    }
    let any = std::fs::read_dir(&folder).ok()?.next().is_some();
    any.then_some(Outputs::Folder(folder))
}

/// The trial's task folder: config.json's `task_path`, else lock.json's.
/// A relative path is tried from here, then from the folder the jobs
/// folder sits in.
fn task_path(
    dir: &Path,
    job: &Path,
    lock: Option<&lock::Lock>,
    record: &TrialRecord,
) -> Option<PathBuf> {
    let from_config = std::fs::read_to_string(dir.join("config.json"))
        .ok()
        .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        .and_then(|v| {
            v.get("task_path")
                .and_then(Value::as_str)
                .map(PathBuf::from)
        });
    let from_lock = lock.and_then(|l| {
        l.trials
            .iter()
            .find(|t| t.trial == record.rollout_name)
            .map(|t| t.task.path.clone())
    });
    let project = job.parent().and_then(Path::parent);
    [from_config, from_lock]
        .into_iter()
        .flatten()
        .flat_map(|p| {
            if p.is_absolute() {
                vec![p]
            } else {
                let mut v = vec![p.clone()];
                if let Some(root) = project {
                    v.push(root.join(&p));
                }
                v
            }
        })
        .find(|p| p.is_dir())
}

/// Pack an `outputs/` folder into a tar archive with the same layout the
/// runner builds in the sandbox: paths relative to `/`, regular files and
/// folders only.
/// More than [`crate::custody::OUTPUT_BUDGET`] bytes of files is refused
/// with [`std::io::ErrorKind::FileTooLarge`] before anything is written.
pub fn pack_folder(folder: &Path, archive: &Path) -> std::io::Result<()> {
    let bytes = folder_bytes(folder);
    if bytes > crate::custody::OUTPUT_BUDGET {
        return Err(std::io::Error::new(
            std::io::ErrorKind::FileTooLarge,
            format!(
                "{} holds {bytes} bytes of files, past the {} a restore takes",
                folder.display(),
                crate::custody::OUTPUT_BUDGET
            ),
        ));
    }
    let file = std::fs::File::create(archive)?;
    let mut b = tar::Builder::new(file);
    b.follow_symlinks(false);
    for entry in walkdir::WalkDir::new(folder)
        .min_depth(1)
        .follow_links(false)
    {
        let entry = entry.map_err(std::io::Error::other)?;
        let rel = entry
            .path()
            .strip_prefix(folder)
            .map_err(std::io::Error::other)?;
        if entry.file_type().is_dir() {
            b.append_dir(rel, entry.path())?;
        } else if entry.file_type().is_file() {
            b.append_path_with_name(entry.path(), rel)?;
        }
    }
    b.finish()
}

/// The bytes of the regular files under `folder`, links not followed.
fn folder_bytes(folder: &Path) -> u64 {
    walkdir::WalkDir::new(folder)
        .min_depth(1)
        .follow_links(false)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
        .map(|e| e.metadata().map(|m| m.len()).unwrap_or(0))
        .sum()
}

/// One trial's old and new score.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Change {
    pub trial: String,
    pub task: String,
    pub old_status: Status,
    pub old_reward: Option<f64>,
    pub new_status: Status,
    pub new_reward: Option<f64>,
    pub changed: bool,
}

/// A finished regrade.
pub struct RegradeOutcome {
    pub dir: PathBuf,
    pub name: String,
    pub records: Vec<TrialRecord>,
    pub changes: Vec<Change>,
    pub not_regradable: Vec<NotRegradable>,
    pub summary: Value,
    /// Planned regrades that never started (the regrade was drained or
    /// stopped first).
    pub not_started: usize,
}

impl RegradeOutcome {
    pub fn changed(&self) -> usize {
        self.changes.iter().filter(|c| c.changed).count()
    }
}

/// The first free `<job>-regrade-<n>` next to the job.
pub fn regrade_dir(job: &Path) -> PathBuf {
    let name = job
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "job".into());
    let parent = job.parent().unwrap_or(Path::new("."));
    let mut n = 1;
    loop {
        let dir = parent.join(format!("{name}-regrade-{n}"));
        if !dir.exists() {
            return dir;
        }
        n += 1;
    }
}

/// Regrade the planned trials, `concurrency` at a time, into a new job
/// folder.
pub async fn run(
    plan: Plan,
    rt: Arc<Runtime>,
    concurrency: usize,
) -> std::io::Result<RegradeOutcome> {
    let started = Instant::now();
    let dir = regrade_dir(&plan.job);
    std::fs::create_dir_all(&dir)?;
    // Held like `bench run` holds its job, so `bench view` leaves the trials
    // still being regraded out instead of counting them as errored.
    let _running = job::hold(&dir)?;
    let name = dir
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    let scratch = tempfile::tempdir()?;
    let source = plan
        .evaluation
        .as_ref()
        .and_then(|e| e.get("source").cloned())
        .unwrap_or(Value::Null);
    let specs: Vec<TrialSpec> = plan
        .regradable
        .iter()
        .map(|r| TrialSpec {
            task: r.task.clone(),
            attempt: Attempt::Nop,
            name: r.record.rollout_name.clone(),
            dir: dir.join(&r.record.rollout_name),
        })
        .collect();
    let fingerprint = format!("regrade:{}", plan.job.display());
    let mut lock = lock::build(
        LockHeader {
            job_name: &name,
            fingerprint: &fingerprint,
            source: &source,
            trials_per_task: 1,
            concurrency,
            regrade_of: Some(plan.job.clone()),
        },
        &specs,
        &rt.options,
    );
    // The lock names who made each trial's outputs, not the placeholder
    // attempt the regrade runs under.
    for (t, r) in lock.trials.iter_mut().zip(&plan.regradable) {
        t.attempt.agent = r.record.agent.clone();
        t.attempt.model = r.record.model.clone();
        t.attempt.kind = "regrade".into();
    }
    job::write_lock(&dir, &lock)?;
    write_json(
        &dir.join("evaluation.json"),
        &json!({
            "schema_version": 1,
            "job_name": name,
            "regrade_of": plan.job,
            "source": source,
            "config": {
                "environment": rt.provider.name(),
                "concurrency": concurrency,
                "tasks": plan.regradable.iter().map(|r| json!({"name": r.task.name, "id": r.task.id, "digest": r.task.digest, "path": r.task.root})).collect::<Vec<_>>(),
            },
            "engine": { "name": "benchflow-rs", "version": env!("CARGO_PKG_VERSION"), "fingerprint": fingerprint },
        }),
    )?;
    rt.events.send(Event::JobStarted {
        dir: dir.clone(),
        total: specs.len(),
        reused: 0,
        resumed: false,
    });

    let sem = Arc::new(Semaphore::new(concurrency.max(1)));
    let mut set = JoinSet::new();
    for (i, (spec, r)) in specs.into_iter().zip(&plan.regradable).enumerate() {
        let archive = match &r.outputs {
            Outputs::Archive(p) => p.clone(),
            Outputs::Folder(f) => {
                let p = scratch.path().join(format!("{i}.tar"));
                pack_folder(f, &p)?;
                p
            }
        };
        let regrade = Regrade {
            archive,
            agent: r.record.agent.clone(),
            model: r.record.model.clone(),
            output: recorded_agent(&r.record),
            agent_files: r.dir.clone(),
            // Planned only when they fit, so never empty here.
            allowed: restore_paths(&r.task, &r.dir).unwrap_or_default(),
        };
        let sem = sem.clone();
        let rt = rt.clone();
        set.spawn(async move {
            let _permit = sem.acquire_owned().await.ok()?;
            if !rt.starting_new() {
                return None;
            }
            Some(trial::regrade(&spec, &rt, regrade).await)
        });
    }
    let mut records = Vec::new();
    let mut not_started = 0;
    while let Some(res) = set.join_next().await {
        match res {
            Ok(Some(rec)) => records.push(rec),
            Ok(None) => not_started += 1,
            Err(e) => {
                eprintln!("benchflow: a regrade task failed: {e}");
                not_started += 1;
            }
        }
    }
    records.sort_by(|a, b| a.rollout_name.cmp(&b.rollout_name));
    let changes: Vec<Change> = records
        .iter()
        .filter_map(|new| {
            let old = &plan
                .regradable
                .iter()
                .find(|r| r.record.rollout_name == new.rollout_name)?
                .record;
            Some(Change {
                trial: new.rollout_name.clone(),
                task: new.task_name.clone(),
                old_status: old.status,
                old_reward: old.reward,
                new_status: new.status,
                new_reward: new.reward,
                changed: old.status != new.status || !same_reward(old.reward, new.reward),
            })
        })
        .collect();
    let mut agents: Vec<String> = Vec::new();
    for r in &records {
        if !agents.contains(&r.agent) {
            agents.push(r.agent.clone());
        }
    }
    let mut summary = job::summary_json(
        &name,
        &JobLabels {
            agent: agents.join(","),
            model: records.iter().find_map(|r| r.model.clone()),
            concurrency,
        },
        &rt,
        &records,
        started.elapsed().as_secs_f64(),
        not_started,
    );
    let changed = changes.iter().filter(|c| c.changed).count();
    if let Some(m) = summary.as_object_mut() {
        m.insert("regrade_of".into(), json!(plan.job));
        m.insert(
            "regrade".into(),
            json!({
                "of": plan.job,
                "regraded": changes.len(),
                "changed": changed,
                "not_regradable": plan.not_regradable,
                "trials": changes,
            }),
        );
    }
    job::write_results(&dir, &summary, &records)?;
    Ok(RegradeOutcome {
        dir,
        name,
        records,
        changes,
        not_regradable: plan.not_regradable,
        summary,
        not_started,
    })
}

fn same_reward(a: Option<f64>, b: Option<f64>) -> bool {
    match (a, b) {
        (Some(x), Some(y)) => (x - y).abs() < 1e-9,
        (None, None) => true,
        _ => false,
    }
}

/// What the original trial's agent did, kept in the regraded record.
pub(crate) fn recorded_agent(r: &TrialRecord) -> AgentOutput {
    AgentOutput {
        tool_calls: r.tool_calls,
        prompts: r.prompts.clone(),
        usage: r.usage.clone(),
        trajectory_summary: r.trajectory_summary.clone(),
        timed_out: r.timed_out,
        problem: None,
        env_keys: r.env_keys.clone(),
        login: r.login.clone(),
        skip_grading: false,
        model: r.model.clone(),
        ..AgentOutput::default()
    }
}

/// Copy the original trial's trajectory and agent logs, so the regraded
/// trial reads like the original in the viewer.
pub(crate) fn copy_agent_files(from: &Path, to: &Path) {
    for sub in ["trajectory", "agent"] {
        let src = from.join(sub);
        if !src.is_dir() {
            continue;
        }
        for e in walkdir::WalkDir::new(&src)
            .follow_links(false)
            .into_iter()
            .filter_map(Result::ok)
        {
            let Ok(rel) = e.path().strip_prefix(from) else {
                continue;
            };
            let dest = to.join(rel);
            if e.file_type().is_dir() {
                let _ = std::fs::create_dir_all(&dest);
            } else if e.file_type().is_file() {
                let _ = std::fs::copy(e.path(), &dest);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packs_an_outputs_folder_with_paths_from_root() {
        let d = tempfile::tempdir().unwrap();
        let out = d.path().join("outputs");
        std::fs::create_dir_all(out.join("app/sub")).unwrap();
        std::fs::write(out.join("app/sub/answer.txt"), "42").unwrap();
        let archive = d.path().join("o.tar");
        pack_folder(&out, &archive).unwrap();
        let back = d.path().join("back");
        let u = crate::custody::unpack(&archive, &back).unwrap();
        assert_eq!(u.files, 1);
        assert_eq!(
            std::fs::read_to_string(back.join("app/sub/answer.txt")).unwrap(),
            "42"
        );
    }

    /// Guards 044f1d73 (`bench run --diff`) and `bench regrade`: a trial's
    /// config.json chose the folder its outputs are restored into as root.
    #[test]
    fn a_recorded_workdir_cannot_widen_what_is_restored() {
        let d = tempfile::tempdir().unwrap();
        let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tasks/hello");
        let mut task = benchflow_task::load(&src).unwrap();
        task.sandbox.workdir = None;
        let trial = d.path().join("t");
        let out = d.path().join("outputs");
        std::fs::create_dir_all(out.join("usr/bin")).unwrap();
        std::fs::write(out.join("usr/bin/x"), "x").unwrap();
        std::fs::create_dir_all(&trial).unwrap();
        for w in [
            "/",
            "/usr",
            "/usr/local/lib",
            "/var",
            "/tests",
            "app",
            "/app/../usr",
            "/logs/x",
        ] {
            std::fs::write(
                trial.join("config.json"),
                serde_json::json!({ "workdir": w }).to_string(),
            )
            .unwrap();
            let why = stray_outputs(&task, &trial, &Outputs::Folder(out.clone()));
            assert!(why.is_some(), "{w} was trusted");
        }
        // A plain working folder holds them, and the task's declared one wins.
        std::fs::write(trial.join("config.json"), r#"{"workdir": "/usr"}"#).unwrap();
        task.sandbox.workdir = Some("/usr".into());
        assert_eq!(
            stray_outputs(&task, &trial, &Outputs::Folder(out.clone())),
            None
        );
        let ok = d.path().join("ok");
        std::fs::create_dir_all(ok.join("root/work")).unwrap();
        std::fs::write(ok.join("root/work/a"), "a").unwrap();
        task.sandbox.workdir = None;
        std::fs::write(trial.join("config.json"), r#"{"workdir": "/root/work"}"#).unwrap();
        assert_eq!(stray_outputs(&task, &trial, &Outputs::Folder(ok)), None);
    }

    #[test]
    fn regrade_folders_count_up() {
        let d = tempfile::tempdir().unwrap();
        let job = d.path().join("2026-10-10__09-00-00");
        std::fs::create_dir_all(&job).unwrap();
        assert_eq!(
            regrade_dir(&job),
            d.path().join("2026-10-10__09-00-00-regrade-1")
        );
        std::fs::create_dir_all(d.path().join("2026-10-10__09-00-00-regrade-1")).unwrap();
        assert_eq!(
            regrade_dir(&job),
            d.path().join("2026-10-10__09-00-00-regrade-2")
        );
    }

    /// Guards the regrade of tasks P014 refuses (the D5b runtime review of
    /// cbf42071, finding 7): a sidecar command Docker Compose cannot split
    /// keeps the task from running, but a job that ran it under an earlier
    /// bench is still regraded, as grading starts no service.
    #[test]
    fn a_task_with_a_command_compose_cannot_split_is_regraded() {
        let d = tempfile::tempdir().unwrap();
        let task = d.path().join("tasks/sidecar");
        for (rel, text) in [
            (
                "task.toml",
                "version = \"1.0\"\n\n[environment]\nworkdir = \"/app\"\n",
            ),
            ("instruction.md", "Ask the api for its token.\n"),
            ("environment/Dockerfile", "FROM ubuntu:24.04\n"),
            (
                "environment/docker-compose.yaml",
                "services:\n  api:\n    image: python:3.12-slim\n    command: python -c print(1)\n",
            ),
            ("tests/test.sh", "echo 1 > /logs/verifier/reward.txt\n"),
        ] {
            let p = task.join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, text).unwrap();
        }
        let refused = benchflow_task::load(&task).unwrap_err();
        assert!(refused.iter().any(|x| x.code == "P014"), "{refused:#?}");

        let job = d.path().join("jobs/run");
        let trial = job.join("sidecar__00000001");
        std::fs::create_dir_all(trial.join("outputs/app")).unwrap();
        std::fs::write(trial.join("outputs/app/token.txt"), "t0k3n").unwrap();
        std::fs::write(
            trial.join("result.json"),
            r#"{"task_name": "sidecar", "rollout_name": "sidecar__00000001", "agent": "oracle", "status": "scored", "sandbox_id": "sb-1", "rewards": {"reward": 1.0}}"#,
        )
        .unwrap();
        std::fs::write(
            trial.join("config.json"),
            json!({ "task_path": task, "workdir": "/app" }).to_string(),
        )
        .unwrap();
        let plan = plan(&job).unwrap();
        let why: Vec<&str> = plan
            .not_regradable
            .iter()
            .map(|n| n.reason.as_str())
            .collect();
        assert!(why.is_empty(), "{why:#?}");
        assert_eq!(plan.regradable.len(), 1);
        let task = &plan.regradable[0].task;
        assert!(task.unsupported.is_empty(), "{:#?}", task.unsupported);
        assert!(
            trial::refusal(task, &Attempt::Nop).is_none(),
            "the regrade would stop on R401"
        );
    }

    /// A trial refused before its sandbox started says so, not that it
    /// saved no outputs: --save-outputs would not have helped it.
    #[test]
    fn a_trial_that_never_started_says_so() {
        let d = tempfile::tempdir().unwrap();
        let trial = d.path().join("job/sidecar__00000001");
        std::fs::create_dir_all(&trial).unwrap();
        std::fs::write(
            trial.join("result.json"),
            r#"{"task_name": "sidecar", "rollout_name": "sidecar__00000001", "agent": "builtin", "status": "errored", "sandbox_id": null,
                "problem": {"code": "R401", "owner": "you", "phase": "setup", "message": "no services here", "next": "use docker", "transient": false}}"#,
        )
        .unwrap();
        let plan = plan_with(&d.path().join("job"), &|_| Err("unused".into())).unwrap();
        assert_eq!(plan.not_regradable.len(), 1);
        assert_eq!(
            plan.not_regradable[0].reason,
            "it never started a sandbox (R401)"
        );
        assert!(plan.not_regradable[0].next.contains("bench results"));
    }

    /// Review F2-RA-1: an `outputs/` folder from before the byte budget can
    /// name far more than a restore takes (a file copied per hard link).
    /// Packing it for a regrade or `--diff` stops before writing anything.
    #[test]
    fn pack_folder_refuses_a_folder_past_the_budget() {
        let d = tempfile::tempdir().unwrap();
        let out = d.path().join("outputs/app");
        std::fs::create_dir_all(&out).unwrap();
        for name in ["a", "b"] {
            // Sparse: 600 MiB each, nothing written.
            let f = std::fs::File::create(out.join(name)).unwrap();
            f.set_len(600 << 20).unwrap();
        }
        let archive = d.path().join("outputs.tar");
        let e = pack_folder(&d.path().join("outputs"), &archive).unwrap_err();
        assert_eq!(e.kind(), std::io::ErrorKind::FileTooLarge, "{e}");
        assert!(std::fs::metadata(&archive).map_or(0, |m| m.len()) == 0);
    }
}
