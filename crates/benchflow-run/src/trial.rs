//! One trial: an attempt at one task in one sandbox, start to finish.
//!
//! ```text
//! image → sandbox → setup → attempt → handoff → verify → save → teardown
//! ```
//!
//! Every phase is timed; every failure becomes a [`Problem`] with an owner;
//! the defenses that held are recorded. A trial never panics the run: the
//! worst it does is come back errored.

use std::collections::BTreeMap;

use futures::FutureExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use benchflow_sandbox::{
    Exec, ExecOutput, Image, Presence, Provider, Sandbox, SandboxError, SandboxSpec, shell,
};
use benchflow_task::TaskPackage;
use benchflow_task::package::{Format, Isolation, OnTimeout};
use chrono::Local;
use serde_json::{Value, json};

use crate::agent::{AgentContext, AgentOutput, Attempt, Usage};
use crate::config::{Keep, RunOptions};
use crate::events::{Event, Events, TrialSummary};
use crate::hardening::{self, Report};
use crate::images::{self, Images};
use crate::netpolicy;
use crate::outcome::{DefenseState, Defenses, Owner, Phase, Problem, Status, defense};
use crate::record::{Custody, Timing, TrialRecord};
use crate::scoring;

#[path = "agent_files.rs"]
mod agent_files;
#[path = "branch.rs"]
pub mod branch;
#[path = "collect.rs"]
mod collect;
#[path = "exhaust.rs"]
mod exhaust;
#[path = "services.rs"]
mod services;
#[path = "stages.rs"]
mod stages;

/// What one trial is.
#[derive(Debug, Clone)]
pub struct TrialSpec {
    pub task: Arc<TaskPackage>,
    pub attempt: Attempt,
    /// The trial's name, `<task>__<8 hex>`; also its folder name.
    pub name: String,
    /// The trial folder.
    pub dir: PathBuf,
}

/// What every trial of a run shares.
pub struct Runtime {
    pub provider: Arc<dyn Provider>,
    pub images: Images,
    pub options: RunOptions,
    pub events: Events,
    /// Becomes true when the run should stop: no new trials start, and
    /// running ones are cancelled and torn down.
    pub stop: tokio::sync::watch::Receiver<bool>,
}

impl Runtime {
    pub fn stopping(&self) -> bool {
        *self.stop.borrow()
    }
    /// Whether new trials (and retries) may still start: not once the run
    /// is stopping or draining.
    pub fn starting_new(&self) -> bool {
        !self.stopping() && !self.options.drain.on()
    }
}

/// A runtime stop switch: the sender stops the run.
pub fn stop_switch() -> (
    tokio::sync::watch::Sender<bool>,
    tokio::sync::watch::Receiver<bool>,
) {
    tokio::sync::watch::channel(false)
}

/// Sandboxes still being made, each in a task of its own. A trial stopped
/// meanwhile no longer waits for its sandbox, so the task removes it once
/// the backend hands it over.
static STARTS: std::sync::Mutex<Vec<tokio::task::JoinHandle<()>>> =
    std::sync::Mutex::new(Vec::new());

fn track_start(start: tokio::task::JoinHandle<()>) {
    let mut starts = STARTS.lock().unwrap_or_else(|e| e.into_inner());
    starts.retain(|s| !s.is_finished());
    starts.push(start);
}

/// Sandboxes this process could not remove (the backend did not answer),
/// as `<backend> <id>`, for the end-of-run warning.
static LEFT_BEHIND: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// Note that `sb` could not be removed.
pub(crate) fn note_left_behind(sb: &dyn Sandbox) {
    LEFT_BEHIND
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push(format!("{} {}", sb.backend(), sb.id()));
}

/// The sandboxes this process could not remove, so far.
pub fn left_behind() -> Vec<String> {
    LEFT_BEHIND
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
}

/// Wait, up to `limit`, for sandboxes still being made when the run was
/// stopped, so each is removed rather than left for the next run's sweep.
/// True when none is left.
pub async fn settle_stopped_starts(limit: Duration) -> bool {
    let starts = std::mem::take(&mut *STARTS.lock().unwrap_or_else(|e| e.into_inner()));
    let all = futures::future::join_all(starts);
    tokio::time::timeout(limit, all).await.is_ok()
}

/// The longest a setup command may take.
const SETUP_TIMEOUT: Duration = Duration::from_secs(180);
/// The longest giving the workdir to a new owner may take (a large `/root`
/// on an overlay filesystem is slow the first time).
const OWN_TIMEOUT: Duration = Duration::from_secs(600);
/// The longest an exploit probe may run.
const PROBE_TIMEOUT: Duration = Duration::from_secs(120);

/// Fields of a task this runner refuses for this attempt, as a problem.
pub fn refusal(task: &TaskPackage, attempt: &Attempt) -> Option<Problem> {
    let model = matches!(attempt, Attempt::Model(m) if m.needs_network());
    let mut fields: Vec<String> = task
        .unsupported
        .iter()
        .filter(|u| !u.agents_only || model)
        .map(|u| format!("{} ({})", u.field, u.reason))
        .collect();
    fields.extend(crate::features::refused(task));
    fields.extend(netpolicy::refusals(task, model));
    if let Some(r) = &task.verifier.rubric {
        let roles: std::collections::BTreeSet<&str> = r
            .criteria
            .iter()
            .filter(|c| c.judge.is_model() && c.judge != benchflow_task::rubric::Judge::Llm)
            .map(|c| c.judge.as_str())
            .collect();
        if !roles.is_empty() {
            fields.push(format!(
                "rubric criteria judged by {} (only test- and llm-judged criteria run in Rust yet)",
                roles.into_iter().collect::<Vec<_>>().join(", ")
            ));
        }
    }
    if matches!(attempt, Attempt::Oracle) && task.oracle.is_none() {
        return Some(Problem::new(
            "R306",
            Owner::Task,
            Phase::Setup,
            format!(
                "{} has no oracle (oracle/solve.sh or solution/solve.sh)",
                task.name
            ),
            "add a reference solution, or run a model agent instead of the oracle",
        ));
    }
    if fields.is_empty() {
        return None;
    }
    Some(Problem::new(
        "R401",
        Owner::You,
        Phase::Setup,
        format!(
            "{} asks for what this runner cannot do yet: {}",
            task.name,
            fields.join("; ")
        ),
        "run it with BenchFlow 0.8 (`bench eval run`), or remove the field",
    ))
}

/// Run one trial to completion. Never panics; failures come back in the
/// record.
pub async fn run(spec: &TrialSpec, rt: &Runtime) -> TrialRecord {
    run_trial(spec, rt, None, None).await
}

/// A finished trial to grade again: its saved outputs, and what its agent
/// did, which the new record keeps.
#[derive(Clone)]
pub struct Regrade {
    /// A tar archive of the outputs, paths relative to `/`.
    pub archive: PathBuf,
    pub agent: String,
    pub model: Option<String>,
    pub output: AgentOutput,
    /// The earlier trial's folder: its trajectory and agent files are copied
    /// into the new trial when it starts, not before (a drained regrade
    /// leaves no half-made folders behind).
    pub agent_files: PathBuf,
    /// The paths the archive may restore, as [`crate::regrade::restore_paths`]
    /// gives them for the earlier trial.
    pub allowed: Vec<String>,
}

/// Grade a finished trial again with its task's current verifier: restore
/// its saved outputs in a fresh sandbox and verify there, as a separate
/// verifier does. The agent does not run.
pub async fn regrade(spec: &TrialSpec, rt: &Runtime, regrade: Regrade) -> TrialRecord {
    crate::regrade::copy_agent_files(&regrade.agent_files, &spec.dir);
    run_trial(spec, rt, Some(regrade), None).await
}

async fn run_trial(
    spec: &TrialSpec,
    rt: &Runtime,
    regrade: Option<Regrade>,
    fork: Option<branch::Role>,
) -> TrialRecord {
    let started = Local::now();
    let clock = Instant::now();
    let (label, model) = match &regrade {
        Some(r) => (r.agent.clone(), r.model.clone()),
        None => (spec.attempt.label(), spec.attempt.model()),
    };
    rt.events.send(Event::TrialStarted {
        trial: spec.name.clone(),
        task: spec.task.name.clone(),
        agent: label.clone(),
    });
    let _ = std::fs::create_dir_all(&spec.dir);
    let mut t = Trial {
        spec,
        rt,
        timing: Timing::default(),
        defenses: Defenses::new(),
        custody: Custody::default(),
        sandbox: None,
        retired: Vec::new(),
        sandbox_id: None,
        home: String::new(),
        user: String::new(),
        workdir: String::new(),
        image_path: None,
        image_loader: Vec::new(),
        verifier_exit: None,
        stage_worked: false,
        stage_own_timeout: false,
        agent_turned: false,
        regrade: regrade.as_ref().map(|r| r.archive.clone()),
        restore_paths: regrade.as_ref().map(|r| r.allowed.clone()),
        agent: regrade.map(|r| r.output).unwrap_or_default(),
        // A child enforces the parent's resolved networks again.
        net: branch::parent_net(fork.as_ref()),
        phase: Phase::Image,
        x: Default::default(),
        skills: None,
        loaded: None,
        agent_session: None,
        resources_raised: None,
        killed_before_save: 0,
        layer_watch: None,
        watch_ended: false,
        disk_seen: Default::default(),
        memory_before: None,
        turn_kill_failed: None,
        base_image: None,
        fork,
    };
    // A bug that panics ends this trial, not the run: it is recorded (R901)
    // and its sandbox is still removed below.
    let driven = {
        let mut stop = rt.stop.clone();
        let drive = std::panic::AssertUnwindSafe(t.drive()).catch_unwind();
        tokio::pin!(drive);
        tokio::select! {
            // A trial that ends in the same poll as the stop keeps its result.
            biased;
            r = &mut drive => r.map_err(|panic| panic_text(panic.as_ref())),
            // A dropped switch is not a stop: `wait_for` then fails, and
            // the branch is disabled.
            Ok(_) = stop.wait_for(|v| *v) => Ok(Err(Problem::new(
                "R001",
                Owner::You,
                Phase::Agent,
                "stopped before it finished",
                "rerun the same command to resume the job",
            ))),
        }
    };
    // A stop names the phase it interrupted. A failure already known when
    // the stop came is recorded instead: an agent that had failed (the stop
    // landed while its failure was being checked), or one that had timed out
    // on a task whose timeout fails it. Left as stopped, such trials would
    // leave the pass rate and raise it.
    let driven =
        driven.map(|r| known_before_stop(r, &t.agent, spec.task.agent.on_timeout, t.phase));
    let result = driven.unwrap_or_else(|panic| {
        Err(Problem::new(
            "R901",
            Owner::Benchflow,
            t.phase,
            format!(
                "a bug in BenchFlow stopped the trial while {}: {panic}",
                t.phase.label()
            ),
            "please report it at https://github.com/benchflow-ai/benchflow/issues with the trial folder",
        ))
    });
    // The disk limit's watch runs to the end of the trial (grading
    // included), and a stop or a panic may have cut a turn short: end it
    // here, before the sandbox goes, so nothing goes on measuring or
    // stopping the sandbox (a kept one included). A stop it made replaces
    // the result: the agent's R316, or the sandbox's R110.
    let result = if t.layer_watch.is_some() {
        let pending = result.as_ref().err().cloned();
        match t.end_watch(pending.as_ref()).await {
            Some(p) => Err(p),
            None => result,
        }
    } else {
        result
    };
    let result = match result {
        Err(p) => {
            let p = t.sandbox_lost_instead(p).await;
            let p = t.exhausted_instead(p).await;
            Err(t.lost_again(p).await)
        }
        ok => ok,
    };
    let mut passed_for_keep = false;
    let mut rec = TrialRecord {
        task_name: spec.task.name.clone(),
        task_id: spec.task.id.clone(),
        rollout_name: spec.name.clone(),
        agent: label.clone(),
        model: model.clone().or_else(|| t.agent.model.clone()),
        status: Status::Errored,
        rewards: None,
        reward: None,
        passed: None,
        timed_out: t.agent.timed_out,
        own_timed_out: if t.spec.task.extras.steps.is_some() {
            t.stage_own_timeout
        } else {
            t.agent.timed_out
        },
        problem: None,
        usage: t.agent.usage.clone(),
        tool_calls: t.agent.tool_calls,
        prompts: t.agent.prompts.clone(),
        trajectory_summary: t.agent.trajectory_summary.clone(),
        timing: Timing::default(),
        started_at: started,
        finished_at: started,
        task_digest: spec.task.digest.clone(),
        sandbox_id: t.sandbox_id.clone(),
        sandbox_backend: rt.provider.name().to_string(),
        defenses: Defenses::new(),
        custody: Custody::default(),
        login: t.agent.login.clone(),
        agent_version: t.agent.agent_version.clone(),
        env_keys: t.agent.env_keys.clone(),
        verifier_exit: t.verifier_exit,
        rubric: None,
        step_results: None,
        agent_session: None,
        resumed_session: None,
        resources_raised: None,
        environment_warnings: Vec::new(),
        reward_range: spec.task.verifier.reward_range,
        dataset: rt
            .options
            .dataset
            .as_ref()
            .map(|d| d.stamp(&spec.task.root)),
    };
    match result {
        Ok(score) => {
            rec.status = Status::Scored;
            let mut reward = score.reward;
            let mut passed = score.passed;
            if t.agent.timed_out && spec.task.agent.on_timeout == OnTimeout::Fail {
                reward = 0.0;
                passed = false;
            }
            let mut values = score.values.clone();
            values.insert("reward".into(), reward);
            rec.rewards = Some(values);
            rec.reward = Some(reward);
            rec.passed = Some(passed);
            rec.rubric = score.rubric;
            passed_for_keep = passed;
        }
        // The agent exhausted its own sandbox (decision 21): a 0, kept on
        // resume and counted in every score.
        Err(p) if p.code == exhaust::EXHAUSTED && p.owner == Owner::Agent => {
            rec.status = Status::Scored;
            rec.rewards = Some(BTreeMap::from([("reward".to_string(), 0.0)]));
            rec.reward = Some(0.0);
            rec.passed = Some(false);
            rec.problem = Some(p);
        }
        Err(p) if p.code == "R000" || p.code == "R001" => {
            if p.code == "R001" && rec.tool_calls == 0 {
                stopped_counts(&spec.dir, &mut rec);
            }
            rec.status = Status::Skipped;
            rec.problem = Some(p);
        }
        Err(p) => {
            rec.status = match (p.phase, p.owner) {
                // The task's verifier or image failed after the agent ran.
                // A sandbox that failed before grading (R305, R309) is
                // infrastructure: errored, so a resumed job runs it again.
                (Phase::Verify | Phase::Handoff, Owner::Task) => Status::Unscored,
                // The task failed preparing a later stage, after an earlier
                // stage's agent ran to the end: that work is not run again
                // for the task's fault. A fault before any agent worked
                // stays errored, as in a single-stage task.
                (Phase::Setup, Owner::Task) if t.stage_worked => Status::Unscored,
                // The rubric's model judge could not decide: nothing is
                // known about the agent's work.
                (Phase::Verify, Owner::Model) => Status::Unscored,
                _ => Status::Errored,
            };
            rec.problem = Some(p);
        }
    }
    // Agents that report the outcome elsewhere (the external agent's backend).
    if t.regrade.is_none()
        && let Attempt::Model(m) = &spec.attempt
    {
        m.graded(&spec.name, &rec).await;
    }
    // Teardown, unless asked to keep.
    let keep = match rt.options.keep {
        Keep::Always => true,
        Keep::OnFailure => !passed_for_keep,
        Keep::Never => false,
    };
    let sandboxes: Vec<Box<dyn Sandbox>> = t.retired.drain(..).chain(t.sandbox.take()).collect();
    for sb in sandboxes {
        if keep {
            rt.events.log(
                &spec.name,
                format!("kept sandbox {} ({})", sb.id(), sb.backend()),
            );
        } else if let Err(e) = sb.destroy().await {
            rt.events.log(
                &spec.name,
                format!("could not destroy sandbox {}: {e}", sb.id()),
            );
            note_left_behind(sb.as_ref());
        }
    }
    t.teardown_services(keep).await;
    rec.step_results = t.x.step_results.take();
    rec.agent_session = t.agent_session.take();
    rec.resumed_session = t
        .agent
        .resumed
        .then(|| t.agent.session_id.clone())
        .flatten();
    rec.resources_raised = t.resources_raised.take();
    t.timing.total = clock.elapsed().as_secs_f64();
    rec.timing = t.timing.clone();
    rec.defenses = t.defenses.clone();
    rec.custody = t.custody.clone();
    rec.verifier_exit = t.verifier_exit;
    // A 0 from a broken environment reads like the agent's failure: say so.
    if rec.status == Status::Scored && rec.passed == Some(false) {
        rec.environment_warnings = crate::envcheck::warnings(&spec.dir);
        for w in &rec.environment_warnings {
            rt.events
                .log(&spec.name, format!("the environment may have failed: {w}"));
        }
    }
    rec.finished_at = Local::now();
    let mut config = trial_config(spec, rt, &t, &started);
    config["agent"] = json!(label);
    config["model"] = json!(model);
    if save_record(&spec.dir, &mut rec, &config, &|line| {
        rt.events.log(&spec.name, line)
    }) {
        for e in crate::trainer::write_trial(&spec.dir, &rec) {
            rt.events.log(&spec.name, e.to_string());
        }
    }
    rt.events.send(Event::TrialFinished(TrialSummary {
        trial: spec.name.clone(),
        task: spec.task.name.clone(),
        agent: rec.agent.clone(),
        status: rec.status,
        reward: rec.reward,
        passed: rec.passed,
        timed_out: rec.timed_out,
        reason: rec.reason(),
        seconds: rec.timing.total,
        dir: spec.dir.clone(),
    }));
    rec
}

/// A trial stopped while its agent worked (R001) never got the agent's
/// counts, which are filled when the agent's run returns: take the tool calls
/// and the trajectory summary from the raw log the run left, marked partial.
/// (Its token usage stays unknown.)
fn stopped_counts(dir: &Path, rec: &mut TrialRecord) {
    let Ok(events) = crate::trainer::read_events(dir) else {
        return;
    };
    if events.is_empty() {
        return;
    }
    rec.tool_calls = benchflow_agent::trajectory::count_tool_calls(&events) as u64;
    let summary = benchflow_agent::trajectory::summarize(&events, true, None);
    if let Ok(v) = serde_json::to_value(summary) {
        rec.trajectory_summary = v;
    }
}

/// A stopped trial's result (R001), or the failure it already had.
fn known_before_stop(
    result: Result<scoring::Score, Problem>,
    agent: &AgentOutput,
    on_timeout: OnTimeout,
    phase: Phase,
) -> Result<scoring::Score, Problem> {
    match result {
        Err(mut p) if p.code == "R001" => {
            if let Some(known) = agent.problem.clone() {
                Err(known)
            } else if agent.timed_out && on_timeout == OnTimeout::Fail {
                Ok(scoring::Score {
                    reward: 0.0,
                    values: BTreeMap::new(),
                    passed: false,
                    rubric: None,
                })
            } else {
                p.phase = phase;
                Err(p)
            }
        }
        other => other,
    }
}

/// Write the trial's record into `dir`; true when it was saved. When it
/// could not be (a full disk), the trial is marked unsaved and its folder
/// moved into `<job>/.unsaved/`, kept whole (verifier output, outputs,
/// logs) rather than deleted, so a rerun starts from a clean folder.
fn save_record(dir: &Path, rec: &mut TrialRecord, config: &Value, log: &dyn Fn(String)) -> bool {
    let Err(e) = rec.write(dir, config) else {
        return true;
    };
    log(format!("could not write the trial record: {e}"));
    let moved = dir
        .parent()
        .and_then(|job| crate::job::move_aside(job, dir, ".unsaved"));
    mark_unsaved(rec, &e, moved.as_deref());
    false
}

/// A trial whose record could not be written (a full disk) did not count:
/// `bench results` never sees it and a rerun of the job runs it again. So it
/// is errored here too, with the file that failed, and the run's report
/// lists it and exits non-zero. Its reward goes into the message only, so
/// no mean reward or trainer file counts a trial nothing else counts.
fn mark_unsaved(rec: &mut TrialRecord, e: &crate::record::RecordWriteError, moved: Option<&Path>) {
    let outcome = match (rec.status, rec.reward) {
        (Status::Scored, Some(r)) => format!("the trial scored {}", crate::record::fmt_reward(r)),
        _ => "the trial ended".to_string(),
    };
    let mut problem = Problem::new(
        "R107",
        Owner::You,
        Phase::Save,
        format!(
            "{outcome}, but its record could not be written ({e}){}; bench results will not \
             count it, and rerunning the job runs it again",
            moved
                .map(|m| format!("; its folder was kept as {}", m.display()))
                .unwrap_or_default()
        ),
        "free space on the job folder's disk (or make it writable), then rerun the same command",
    );
    if let Some(earlier) = rec.problem.take() {
        problem = problem.detail(format!("the trial's own problem: {earlier}"));
    }
    rec.status = Status::Errored;
    rec.passed = None;
    rec.reward = None;
    rec.rewards = None;
    rec.rubric = None;
    rec.problem = Some(problem);
}

fn trial_config(
    spec: &TrialSpec,
    rt: &Runtime,
    t: &Trial<'_>,
    started: &chrono::DateTime<Local>,
) -> Value {
    json!({
        "task_path": spec.task.root,
        "task_id": spec.task.id,
        "task_format": spec.task.format,
        "agent": spec.attempt.label(),
        "model": spec.attempt.model(),
        "reasoning_effort": spec.attempt.effort(),
        "agent_options": spec.attempt.options(),
        "resume_sessions": rt.options.resume_sessions,
        "skills": t.skills,
        "load_trajectory": t.loaded,
        "environment": rt.provider.name(),
        "sandbox_user": t.user,
        // The folder the trial worked in: declared, or the image's own.
        "workdir": t.workdir,
        "timeout_sec": agent_timeout(&spec.task, &rt.options).map(|d| d.as_secs_f64()),
        "agent_idle_timeout_sec": rt.options.idle_timeout.as_secs(),
        "agent_env_keys": t.agent.env_keys,
        "pids_limit": rt.options.pids_limit,
        "started_at": started.format("%Y-%m-%d %H:%M:%S%.6f").to_string(),
        "task_config": spec.task.config,
        "engine": { "name": "benchflow-rs", "version": env!("CARGO_PKG_VERSION") },
    })
}

/// The agent's time limit: the task's (or the run's default), scaled.
pub fn agent_timeout(task: &TaskPackage, opts: &RunOptions) -> Option<Duration> {
    task.agent
        .timeout
        .or(opts.default_agent_timeout)
        .map(|d| benchflow_task::values::scale_duration(d, opts.timeout_multiplier.max(0.01)))
}

struct Trial<'a> {
    spec: &'a TrialSpec,
    rt: &'a Runtime,
    timing: Timing,
    defenses: Defenses,
    custody: Custody,
    sandbox: Option<Box<dyn Sandbox>>,
    /// Sandboxes done with before the trial ends (the agent's, when a
    /// separate verifier grades): kept or destroyed with the trial's.
    retired: Vec<Box<dyn Sandbox>>,
    sandbox_id: Option<String>,
    home: String,
    user: String,
    /// The folder the agent, the oracle, and the handoff work in: the task's
    /// declared workdir, else the image's own (see [`Trial::resolve_workdir`]).
    workdir: String,
    /// The image's own `PATH`, read when the sandbox starts, before any
    /// agent.
    image_path: Option<String>,
    /// The loader variables ([`hardening::LOADER_VARS`]) the image's
    /// environment sets, read with `image_path`.
    image_loader: Vec<String>,
    agent: AgentOutput,
    /// A stage of a multi-stage task ran its agent to the end (its attempt
    /// succeeded), so a task fault found preparing a later stage does not
    /// rerun that work. A prompt is recorded even for an attempt that failed,
    /// so the prompts alone cannot say this.
    stage_worked: bool,
    /// An agent's turn ran out of time in a stage this trial ran itself (a
    /// stage-fork child also carries its parent's time-out).
    stage_own_timeout: bool,
    /// An attempt (an agent's turn) has run in this sandbox: a `tar` its
    /// setup used that is gone now is the sandbox's content, not the
    /// image's (review F2-A-r12-1, the sibling).
    agent_turned: bool,
    verifier_exit: Option<i32>,
    /// Grade these saved outputs instead of running an attempt.
    regrade: Option<PathBuf>,
    /// The paths the saved outputs may restore in a fresh sandbox: the
    /// output paths they were packed from (a live trial's, set when they
    /// are collected), or a regrade's.
    restore_paths: Option<Vec<String>>,
    /// The trial's networks, resolved when its first sandbox starts.
    net: Option<netpolicy::Plan>,
    /// The phase the trial is in.
    phase: Phase,
    /// Services, stages, and what grading a stage changes.
    x: services::Extra,
    /// What became of the trial's skills, for config.json.
    skills: Option<Value>,
    /// The session loaded with `--load-trajectory`, for config.json.
    loaded: Option<Value>,
    /// The agent's own session files, saved at the end of its turn.
    agent_session: Option<crate::record::AgentSession>,
    /// Agent processes killed before its session was saved, counted into
    /// the kill before grading.
    killed_before_save: u64,
    /// What the agent's resource floor raised, for result.json.
    resources_raised: Option<crate::record::ResourcesRaised>,
    /// The trial's part in a fork (`bench branch`).
    fork: Option<branch::Role>,
    /// The disk limit's watch, from the agent's first turn to the end of
    /// the trial.
    layer_watch: Option<benchflow_sandbox::LayerWatch>,
    /// The watch was ended: no other starts.
    watch_ended: bool,
    /// The kill at the end of the agent's turn failed: why.
    turn_kill_failed: Option<String>,
    /// The task's built image the agent's sandbox started from (a branch
    /// child's: its parent's), for the probe of the daemon's disk.
    base_image: Option<String>,
    /// What the disk limit's watch could not see, over the agent's turns.
    disk_seen: exhaust::DiskSeen,
    /// The sandbox's own memory counters when the agent's turn began.
    memory_before: Option<exhaust::Memory>,
}

impl<'a> Trial<'a> {
    fn task(&self) -> &TaskPackage {
        &self.spec.task
    }

    /// The agent's sandbox size: the task's, raised to the agent's floor
    /// (its manifest's `[resources]`), and what the floor raised.
    fn agent_size(
        &self,
    ) -> (
        Option<f64>,
        Option<u64>,
        Option<crate::record::ResourcesRaised>,
    ) {
        let task = &self.task().sandbox;
        match &self.spec.attempt {
            Attempt::Model(agent) => apply_floor(
                agent.name(),
                task.cpus,
                task.memory_mb,
                agent.resource_floor(),
            ),
            _ => (task.cpus, task.memory_mb, None),
        }
    }

    fn name(&self) -> &str {
        &self.spec.name
    }

    async fn drive(&mut self) -> Result<scoring::Score, Problem> {
        if let Some(p) = refusal(self.task(), &self.spec.attempt) {
            return Err(p);
        }
        if let Some(archive) = self.regrade.clone() {
            if let Some(p) = crate::features::regrade_refusal(self.task()) {
                return Err(p);
            }
            self.enter(Phase::Verify);
            let clock = Instant::now();
            let score = self.verify_archive(&archive).await;
            self.timing.verifier = clock.elapsed().as_secs_f64();
            return score;
        }
        if let Some(p) = crate::features::backend_refusal(self.task(), self.rt.provider.as_ref()) {
            return Err(p);
        }
        // Image.
        let setup_clock = Instant::now();
        self.enter(Phase::Image);
        let image = self.prepare_image().await?;
        // Sandbox.
        self.enter(Phase::Start);
        self.start_services().await?;
        let clock = Instant::now();
        let sandbox = self.start_sandbox(&image, netpolicy::Host::Agent).await?;
        self.timing.sandbox_start = clock.elapsed().as_secs_f64();
        self.sandbox_id = Some(sandbox.id().to_string());
        self.sandbox = Some(sandbox);
        self.x.attach = None;
        self.resolve_workdir().await?;
        // Setup.
        self.enter(Phase::Setup);
        let clock = Instant::now();
        self.setup().await?;
        // After setup, which gives the agent the workdir.
        let mounts = self.task().extras.mounts.clone();
        self.place_mounts(&mounts, Phase::Setup, false, false, None)
            .await?;
        // A child's skills are in its checkpoint already; placing them
        // again would write as root where the parent's agent had a turn.
        if !self.is_child() {
            self.place_skills().await?;
        }
        self.check_gpu_types().await?;
        self.enforce_network(netpolicy::Stage::Agent, netpolicy::Host::Agent)
            .await?;
        self.timing.setup = clock.elapsed().as_secs_f64();
        self.timing.environment_setup = setup_clock.elapsed().as_secs_f64();
        if self.is_parent() && self.fork_at_start() {
            self.at_fork_point(None).await?;
        }
        if self.task().extras.steps.is_some() {
            return self.run_steps().await;
        }
        // Attempt.
        self.enter(Phase::Agent);
        let agent_clock = Instant::now();
        self.turn_begin().await?;
        let attempted = self.attempt().await;
        self.agent_turned = true;
        let mut attempted = self.turn_end(attempted, exhaust::TurnEnd::Stop).await;
        // A sandbox the disk limit stopped is decided already.
        let exhausted = matches!(&attempted, Err(p) if p.code == exhaust::DISK_STOP || p.code == exhaust::EXHAUSTED);
        // An agent that exited, went silent, or broke off because its
        // sandbox died did not fail on its own.
        if let Err(p) = &attempted
            && p.owner == Owner::Agent
            && !exhausted
        {
            // A sandbox lost under the agent (gone, stopped from outside,
            // or stopped for a reason the backend cannot name): the
            // sandbox's R103. Whether it is rerun, capped, or the agent's
            // own stop is decided with every other loss (`lost_again`).
            if let Some(e) = self.not_running(false).await {
                let what = format!("the agent stopped ({})", p.message);
                attempted =
                    Err(lost(Phase::Agent, &what, e).detail(p.detail.clone().unwrap_or_default()));
            }
        }
        // An agent that did nothing or went silent while its model call
        // never got an answer, or while the provider refused its calls, or
        // that failed on an answer the provider garbled, stopped on the
        // provider (E5: each of these was the agent's R206 or R205).
        if let Err(p) = &attempted
            && p.owner == Owner::Agent
            && !exhausted
            && let Some(provider) =
                provider_instead(&p.code, &self.spec.dir.join("agent"), agent_clock.elapsed())
        {
            attempted = Err(provider.detail(p.message.clone()));
        }
        // An agent whose every model request the provider rejected never
        // worked on the task: not a scored 0 (E7: MiMo answered Codex's
        // `web_search` tool with HTTP 400, and the error text became the
        // agent's answer).
        if attempted.is_ok()
            && let Some(p) = provider_rejected_all(&self.spec.dir.join("agent"))
        {
            attempted = Err(p);
        }
        // An agent that ran out of time while the provider refused its
        // requests was not measured: the provider spent its budget.
        if attempted.is_ok()
            && self.agent.timed_out
            && let Some(p) = throttled(&self.spec.dir.join("agent"), Some(agent_clock.elapsed()))
        {
            attempted = Err(p);
        }
        self.timing.agent_execution = agent_clock.elapsed().as_secs_f64();
        self.save_session(true).await;
        attempted?;
        // A parent forking after its prompts: the checkpoint, then grading
        // only when it goes on.
        if self.is_parent() && self.fork_at_prompt() {
            self.at_fork_point(None).await?;
        }
        if self.agent.skip_grading {
            return Err(Problem::new(
                "R000",
                Owner::You,
                Phase::Agent,
                "not graded, as asked",
                "run it again without --no-grade to grade it",
            ));
        }
        // Handoff and verify.
        let verify_clock = Instant::now();
        self.enter(Phase::Handoff);
        self.handoff().await?;
        self.timing.handoff = verify_clock.elapsed().as_secs_f64();
        self.run_hooks().await?;
        // Outputs are what the agent left, so save them before the verifier
        // can change the workdir. A separate verifier saves them itself.
        if self.task().verifier.isolation == Isolation::Shared
            && (self.rt.options.save_outputs || !self.task().sandbox.outputs.is_empty())
        {
            self.enter(Phase::Save);
            self.save_outputs().await;
        }
        self.enter(Phase::Verify);
        let clock = Instant::now();
        let score = match self.task().verifier.isolation {
            Isolation::Shared => self.verify_shared().await,
            Isolation::Separate => self.verify_separate().await,
        };
        self.timing.verify = clock.elapsed().as_secs_f64();
        self.timing.verifier = verify_clock.elapsed().as_secs_f64();
        score
    }

    /// An agent has had a turn in this sandbox: this trial's attempt ran,
    /// or a fork child restored one made after the parent's turn.
    fn after_turn(&self) -> bool {
        self.agent_turned || self.child_after_agent()
    }

    fn sb(&self) -> &dyn Sandbox {
        self.sandbox.as_deref().expect("sandbox started")
    }

    /// Start a phase: remember it and tell the run display.
    fn enter(&mut self, phase: Phase) {
        self.phase = phase;
        self.rt.events.phase(self.name(), phase);
    }

    /// Run one of BenchFlow's own programs as root ([`Exec::system`]), with
    /// `PATH` the system folders only and no loader or shell start-up
    /// variables: after the agent's turn, a folder on the image's `PATH`
    /// may be the agent's (a venv in the workdir), and `sh` or any tool
    /// found there would run as root; so would a library or start-up file
    /// the image's `LD_PRELOAD`, `LD_LIBRARY_PATH` or `BASH_ENV` names
    /// there. The image's `PATH` goes along in
    /// [`hardening::IMAGE_PATH_VAR`] for the programs that judge it.
    async fn root(&self, script: String, timeout: Duration) -> Result<ExecOutput, SandboxError> {
        let mut exec = Exec::new(hardening::with_plumbing_umask(&script))
            .system()
            .timeout(timeout);
        if let Some(p) = &self.image_path {
            exec = exec.env(hardening::IMAGE_PATH_VAR, p.clone());
        }
        self.sb().exec(exec).await
    }

    async fn prepare_image(&mut self) -> Result<Image, Problem> {
        if let Some(image) = self.checkpoint_image() {
            return Ok(image);
        }
        let started = Instant::now();
        let agent = match &self.spec.attempt {
            Attempt::Model(m) => Some(m.name().to_string()),
            _ => None,
        };
        let layers = images::layers_for(
            self.task(),
            &self.spec.attempt,
            &self.rt.provider.capabilities(),
            &self.rt.options.sandbox_user,
        );
        let spec = images::spec_for(self.task(), layers);
        let log_path = self.spec.dir.join("image.log");
        let events = self.rt.events.clone();
        let trial = self.spec.name.clone();
        let file = std::sync::Mutex::new(std::fs::File::create(&log_path).ok());
        // A builder's log can carry credentials (Daytona's prints signed
        // URLs with a SAS signature and a JWT): redact each line as the
        // trajectory is.
        let redactor = benchflow_agent::trajectory::Redactor::default();
        let log = move |line: &str| {
            let line = redactor.redact(line);
            if let Ok(mut f) = file.lock()
                && let Some(f) = f.as_mut()
            {
                use std::io::Write;
                let _ = writeln!(f, "{line}");
            }
            events.log(&trial, line.into_owned());
        };
        let image = self
            .rt
            .images
            .get(self.rt.provider.as_ref(), &spec, &log, agent.as_deref())
            .await;
        self.timing.image = started.elapsed().as_secs_f64();
        image
    }

    async fn start_sandbox(
        &mut self,
        image: &Image,
        host: netpolicy::Host,
    ) -> Result<Box<dyn Sandbox>, Problem> {
        if self.net.is_none() {
            self.net =
                Some(netpolicy::Plan::resolve(self.task(), &self.rt.options.sandbox_user).await?);
        }
        let caps = self.rt.provider.capabilities();
        let policy = self
            .net
            .as_ref()
            .expect("resolved above")
            .start_policy(host, &caps);
        let task = self.task();
        let mut labels = BTreeMap::new();
        labels.insert("ai.benchflow.trial".to_string(), self.spec.name.clone());
        labels.insert(
            "ai.benchflow.task".to_string(),
            images::sanitize_name(&task.name),
        );
        let (cpus, memory_mb, raised) = self.agent_size();
        if host == netpolicy::Host::Agent
            && let Some(raised) = raised
        {
            self.rt.events.log(self.name(), raised_line(&raised));
            self.resources_raised = Some(raised);
        }
        // A branch child keeps its parent's disk-limit decision and probes
        // the daemon's disk with its parent's built image.
        let (disk_limit_from_parent, base_image) = match &self.fork {
            Some(branch::Role::Child { state, .. }) if host == netpolicy::Host::Agent => {
                (state.disk_limit.clone(), state.base_image.clone())
            }
            _ => (None, None),
        };
        let task = self.task();
        let spec = SandboxSpec {
            image: image.clone(),
            name: images::sanitize_name(&self.spec.name),
            cpus,
            memory_mb,
            disk_mb: task.sandbox.disk_mb,
            gpus: task.sandbox.gpus,
            pids_limit: Some(self.rt.options.pids_limit),
            network: policy,
            env: task.sandbox.env.clone(),
            // Only a declared workdir: otherwise the sandbox keeps the
            // image's WORKDIR, which resolve_workdir reads back.
            workdir: task.sandbox.workdir.clone().filter(|w| !w.is_empty()),
            labels,
            runtime: self.rt.options.runtime.clone(),
            disk_limit_from_parent,
            base_image,
        };
        if let Some(d) = crate::confine::runtime(spec.runtime.as_deref()) {
            self.defenses.insert(d.name.clone(), d);
        }
        defense(
            &mut self.defenses,
            "process_limit",
            if caps.pids_limit {
                DefenseState::Enforced
            } else {
                DefenseState::NotSupported
            },
            Some(if caps.pids_limit {
                format!("at most {} processes", self.rt.options.pids_limit)
            } else {
                format!(
                    "the {} backend cannot cap the number of processes (BenchFlow asks for {})",
                    self.rt.provider.name(),
                    self.rt.options.pids_limit
                )
            }),
        );
        // The agent's sandbox: the agent's chain may guard the daemon later.
        if caps.root_service && host == netpolicy::Host::Agent {
            defense(
                &mut self.defenses,
                "root_service",
                DefenseState::NotSupported,
                Some(format!(
                    "the {} backend runs a service inside the sandbox that runs commands as root for any user who reaches it, so the agent could act as root",
                    self.rt.provider.name()
                )),
            );
        }
        let mut attempt = 0;
        loop {
            match self.create_sandbox(&spec).await {
                Ok(sb) => {
                    if host == netpolicy::Host::Agent {
                        self.record_disk_limit(sb.as_ref(), &caps);
                        self.base_image = Some(
                            spec.base_image
                                .clone()
                                .unwrap_or_else(|| image.reference.clone()),
                        );
                    }
                    return Ok(sb);
                }
                Err(e) if e.is_transient() && attempt < 2 => {
                    attempt += 1;
                    self.rt
                        .events
                        .log(self.name(), format!("sandbox start failed ({e}); retrying"));
                    tokio::time::sleep(Duration::from_secs(2 * attempt as u64)).await;
                }
                Err(SandboxError::Unavailable { message, fix }) => {
                    return Err(Problem::new(
                        "R103",
                        Owner::Sandbox,
                        Phase::Start,
                        message,
                        fix,
                    ));
                }
                Err(e @ (SandboxError::Unsupported(_) | SandboxError::Io { .. })) => {
                    return Err(sandbox_problem(Phase::Start, e));
                }
                Err(e) => {
                    let why = match &e {
                        SandboxError::Start(why) => why.clone(),
                        other => other.to_string(),
                    };
                    let p = Problem::new(
                        "R103",
                        Owner::Sandbox,
                        Phase::Start,
                        format!("the sandbox did not start: {why}"),
                        "retry, or lower --concurrency; `bench doctor` checks the backend",
                    );
                    return Err(if e.is_transient() { p.transient() } else { p });
                }
            }
        }
    }

    /// Enforce a stage's network and self-test it before that stage runs;
    /// the result is recorded as a defense.
    async fn enforce_network(
        &mut self,
        stage: netpolicy::Stage,
        host: netpolicy::Host,
    ) -> Result<(), Problem> {
        let Some(plan) = self.net.clone() else {
            return Ok(());
        };
        let user = match stage {
            netpolicy::Stage::Agent => self.user.clone(),
            netpolicy::Stage::Verifier => self
                .task()
                .verifier
                .user
                .clone()
                .unwrap_or_else(|| "root".into()),
        };
        let caps = self.rt.provider.capabilities();
        // After the agent's turn (the verifier stage) the network self-test
        // runs as the verifier's user with the image's PATH; a program the
        // agent planted on it (a venv in the workdir) would run. Give it the
        // verifier's trusted PATH and drop the loader variables, as the
        // verifier and the hooks get. Before the agent's turn there is
        // nothing planted, so the image's PATH is kept.
        let unset: Vec<String> = match stage {
            netpolicy::Stage::Verifier => hardening::unset_vars(&self.task().verifier.env)
                .iter()
                .map(|s| s.to_string())
                .collect(),
            netpolicy::Stage::Agent => Vec::new(),
        };
        let trusted = match stage {
            netpolicy::Stage::Verifier => self.verifier_trusted_path().await?,
            netpolicy::Stage::Agent => None,
        };
        let self_test = netpolicy::SelfTest {
            path: trusted.as_deref(),
            unset: &unset,
        };
        // The reference solution runs as root (see `run_script`); before it,
        // the agent's chain must cover root too, or the oracle would pass
        // under a policy it never met.
        let root_attempt =
            matches!(self.spec.attempt, Attempt::Oracle) && self.task().agent.user.is_none();
        let out = netpolicy::enforce(
            self.sb(),
            &caps,
            &plan,
            stage,
            host,
            &user,
            self_test,
            root_attempt,
        )
        .await;
        if let Some((state, note)) = out.defense {
            defense(&mut self.defenses, stage.defense(), state, Some(note));
        }
        if let Some((state, note)) = out.root_service {
            defense(&mut self.defenses, "root_service", state, Some(note));
        }
        if let Some((state, note)) = out.metadata {
            defense(
                &mut self.defenses,
                netpolicy::METADATA_DEFENSE,
                state,
                Some(note),
            );
        }
        match out.problem {
            Some(p) => Err(p),
            None => Ok(()),
        }
    }

    /// The folder the trial works in. A task that declares no workdir works
    /// in its image's WORKDIR, as BenchFlow 0.8 and Harbor run it: 0.8 asks
    /// the sandbox for `pwd` and falls back to `/root` when the image leaves
    /// it at `/`. Tasks whose files sit in `/root` (`WORKDIR /root`) and whose
    /// reference solution uses relative paths depend on this.
    async fn resolve_workdir(&mut self) -> Result<(), Problem> {
        // The image's PATH and loader variables, read before anyone else
        // has used the sandbox.
        let probe = format!(
            r#"printf 'BF image-path %s\n' "${{PATH:-}}"; for v in {}; do eval "[ -n \"\${{$v+x}}\" ]" && echo "BF image-loader $v"; done; true"#,
            hardening::LOADER_VARS.join(" ")
        );
        let out = self
            .sb()
            .exec(Exec::new(probe).root().timeout(SETUP_TIMEOUT))
            .await
            .map_err(|e| sandbox_problem(Phase::Setup, e))?;
        let report = Report::parse(&out.stdout);
        self.image_path = report
            .first("image-path")
            .map(str::to_string)
            .filter(|p| !p.is_empty());
        self.image_loader = report
            .all("image-loader")
            .iter()
            .map(|v| v.to_string())
            .collect();
        if let Some(w) = self
            .task()
            .sandbox
            .workdir
            .clone()
            .filter(|w| !w.is_empty())
        {
            self.workdir = w;
            return Ok(());
        }
        let out = self
            .root("pwd".into(), SETUP_TIMEOUT)
            .await
            .map_err(|e| sandbox_problem(Phase::Setup, e))?;
        self.workdir = image_workdir(&out.stdout);
        Ok(())
    }

    async fn setup(&mut self) -> Result<(), Problem> {
        if self.is_child() {
            return self.child_setup().await;
        }
        let task = self.spec.task.clone();
        let workdir = self.workdir.clone();
        let user = task
            .agent
            .user
            .clone()
            .unwrap_or_else(|| self.rt.options.sandbox_user.clone());
        let fail = |msg: String, out: Option<&ExecOutput>| {
            Problem::new(
                "R104",
                Owner::Task,
                Phase::Setup,
                msg,
                "the image needs a shell and useradd (or adduser); or set [agent] user = \"root\"",
            )
            .detail(out.map(|o| o.tail(2000)).unwrap_or_default())
        };
        if hardening::holds_state(&workdir) {
            return Err(Problem::new(
                "R104",
                Owner::Task,
                Phase::Setup,
                format!(
                    "the workdir {workdir} would give the agent user the folder that holds BenchFlow's root-only state ({})",
                    hardening::STATE_DIR
                ),
                "set [sandbox] workdir (or the image's WORKDIR) to the task's own folder, such as /app",
            ));
        }
        if user == "root" || user == "0" {
            defense(
                &mut self.defenses,
                "agent_user",
                DefenseState::Off,
                Some(if task.agent.user.is_some() {
                    "the task runs its agent as root".into()
                } else {
                    "the agent runs as root: the sandbox-user = \"root\" setting".into()
                }),
            );
            let out = self
                .root(hardening::prepare_root(&[&workdir]), SETUP_TIMEOUT)
                .await
                .map_err(|e| sandbox_problem(Phase::Setup, e))?;
            if !out.success() {
                return Err(fail("could not prepare the workdir".into(), Some(&out)));
            }
            self.user = "root".into();
            self.home = "/root".into();
        } else {
            let out = self
                .root(hardening::setup_user(&user, &workdir), SETUP_TIMEOUT)
                .await
                .map_err(|e| sandbox_problem(Phase::Setup, e))?;
            if !out.success() {
                return Err(fail(
                    format!("could not create the agent user `{user}`"),
                    Some(&out),
                ));
            }
            let report = Report::parse(&out.stdout);
            let fields: Vec<&str> = report
                .first("user")
                .unwrap_or("")
                .split_whitespace()
                .collect();
            self.home = fields
                .get(2)
                .map(|s| s.to_string())
                .unwrap_or(format!("/home/{user}"));
            self.user = user.clone();
            let who = format!("{user}, uid {}", fields.get(1).unwrap_or(&"?"));
            let (state, mut note) = self.root_paths(&user, who).await?;
            if let Some(n) = report.first("workdir-unowned") {
                note.push_str(&format!(
                    "; {n} paths in {workdir} kept their owner (root could not chown them), so the agent may be unable to change them"
                ));
            }
            defense(&mut self.defenses, "agent_user", state, Some(note));
            if is_root_home(&workdir) {
                defense(
                    &mut self.defenses,
                    "root_home",
                    DefenseState::Off,
                    Some(format!(
                        "the agent user works in {workdir}, root's home, as the image's WORKDIR says (as in 0.8): start-up and config files there that root's programs read, the verifier's included, are the agent's to write"
                    )),
                );
            }
        }
        // Keep hidden folders out of the agent's reach.
        let mut hidden: Vec<String> = task.hidden_mounts();
        for d in ["/oracle", "/solution", "/verifier", "/tests"] {
            if !hidden.iter().any(|h| h == d) {
                hidden.push(d.to_string());
            }
        }
        let out = self
            .root(hardening::lock_hidden(&hidden), SETUP_TIMEOUT)
            .await
            .map_err(|e| sandbox_problem(Phase::Setup, e))?;
        let report = Report::parse(&out.stdout);
        self.custody.locked = report.all("locked").iter().map(|s| s.to_string()).collect();
        defense(
            &mut self.defenses,
            "hidden_verifier",
            if self.custody.locked.is_empty() {
                DefenseState::Enforced
            } else {
                DefenseState::Failed
            },
            if self.custody.locked.is_empty() {
                Some("the verifier and oracle are uploaded only when needed".into())
            } else {
                Some(format!(
                    "the image ships {}; locked to root before the agent ran",
                    self.custody.locked.join(", ")
                ))
            },
        );
        // Keep the agent from replacing the programs root commands run.
        if self.user != "root" {
            let out = self
                .root(hardening::lock_system_path(&workdir), SETUP_TIMEOUT)
                .await
                .map_err(|e| sandbox_problem(Phase::Setup, e))?;
            let unlocked = Report::parse(&out.stdout).all("unlocked-path").join(", ");
            defense(
                &mut self.defenses,
                "locked_system_path",
                if unlocked.is_empty() {
                    DefenseState::Enforced
                } else {
                    DefenseState::Failed
                },
                Some(if unlocked.is_empty() {
                    "the folders on PATH are root's alone".into()
                } else {
                    format!(
                        "the image lets others write {unlocked}; locked to root before the agent ran"
                    )
                }),
            );
        }
        // Snapshot the files that steer test runners, and record the
        // workdir's virtual environments, whose interpreter links the
        // handoff re-creates: setup runs before the agent's first turn (a
        // child's runs `child_setup` and keeps its parent's record).
        let out = self
            .root(hardening::baseline(Some(&workdir)), SETUP_TIMEOUT)
            .await
            .map_err(|e| sandbox_problem(Phase::Setup, e))?;
        let baseline_ok = out.success() && Report::parse(&out.stdout).first("baseline").is_some();
        defense(
            &mut self.defenses,
            "planted_file_cleanup",
            if baseline_ok {
                DefenseState::Enforced
            } else {
                DefenseState::Failed
            },
            None,
        );
        // Lock system folders, snapshot ld.so.preload, order the OOM killer,
        // read the memory limit.
        let sb = self.sandbox.as_deref().expect("sandbox started");
        let confined = crate::confine::before_agent(
            sb,
            &self.user,
            &workdir,
            &self.home,
            self.agent_size().1,
            self.rt.provider.name(),
        )
        .await
        .map_err(|e| sandbox_problem(Phase::Setup, e))?;
        for d in confined {
            self.defenses.insert(d.name.clone(), d);
        }
        if self.rt.options.scan_credentials && matches!(self.spec.attempt, Attempt::Nop) {
            self.custody.credentials = crate::confine::scan_image(sb, &self.user)
                .await
                .map_err(|e| sandbox_problem(Phase::Setup, e))?;
        }
        // Readiness.
        if let Some(ready) = &task.sandbox.ready {
            self.wait_ready(ready, false).await?;
        }
        Ok(())
    }

    /// Whether the agent user can become root, by read-only checks before
    /// the agent starts. If it can, the separation from root does not hold,
    /// and `agent_user` says so; the trial still runs, since a task may
    /// need `sudo`.
    async fn root_paths(&self, user: &str, who: String) -> Result<(DefenseState, String), Problem> {
        let as_user = self
            .sb()
            .exec(
                Exec::new(hardening::ROOT_PATHS_AS_USER)
                    .user(user)
                    .env("PATH", benchflow_sandbox::shell::SYSTEM_PATH)
                    .timeout(SETUP_TIMEOUT),
            )
            .await
            .map_err(|e| sandbox_problem(Phase::Setup, e))?;
        let as_root = self
            .root(hardening::root_paths_as_root(user), SETUP_TIMEOUT)
            .await
            .map_err(|e| sandbox_problem(Phase::Setup, e))?;
        let (a, r) = (
            Report::parse(&as_user.stdout),
            Report::parse(&as_root.stdout),
        );
        Ok(root_path_verdict(
            who,
            &[a.all("root-path"), r.all("root-path")].concat(),
            &a.all("root-path-unchecked"),
        ))
    }

    /// The verifier's trusted `PATH` (the image's `PATH` less the folders
    /// the agent can write), computed in the sandbox as [`wait_ready`] and
    /// the hooks do. For the verifier-stage network self-test, which runs
    /// after the agent's turn.
    async fn verifier_trusted_path(&self) -> Result<Option<String>, Problem> {
        let out = self
            .root(hardening::trusted_path(&self.workdir), SETUP_TIMEOUT)
            .await
            .map_err(|e| sandbox_problem(Phase::Verify, e))?;
        // Fail closed, as `verifier_env`, the hooks and `wait_ready` do: with
        // no answer the self-test would run on the image's `PATH`.
        match Report::parse(&out.stdout).first("path") {
            Some(p) => Ok(Some(p.to_string())),
            None => Err(unreported_in(
                Phase::Verify,
                "the trusted PATH check for the network self-test",
                &out,
            )),
        }
    }

    /// Run the readiness check until it passes. `after_agent`: an agent
    /// has had a turn in this sandbox (a later stage's check), so its files
    /// may shadow programs on the image's `PATH`; the check then runs in
    /// the verifier's environment, as hooks do.
    pub(crate) async fn wait_ready(
        &self,
        ready: &benchflow_task::package::Ready,
        after_agent: bool,
    ) -> Result<(), Problem> {
        let mut exec = Exec::new(ready.run.clone()).root().timeout(ready.timeout);
        if after_agent {
            let out = self
                .root(hardening::trusted_path(&self.workdir), SETUP_TIMEOUT)
                .await
                .map_err(|e| sandbox_problem(Phase::Setup, e))?;
            let Some(path) = Report::parse(&out.stdout).first("path").map(str::to_string) else {
                return Err(unreported(
                    "the trusted PATH check for the readiness check",
                    &out,
                ));
            };
            let unset = hardening::unset_vars(&self.task().verifier.env);
            exec.command = format!("{}{}", hardening::verifier_env_prefix(&unset), ready.run);
            exec = exec.env("PATH", path).unset(unset.iter().copied());
        }
        let start = Instant::now();
        let mut failures = 0u32;
        loop {
            let in_grace = start.elapsed() < ready.start_period;
            let out = self.sb().exec(exec.clone()).await;
            match out {
                Ok(o) if o.success() => return Ok(()),
                Ok(o) => {
                    if !in_grace {
                        failures += 1;
                    }
                    if failures > ready.retries {
                        return Err(Problem::new(
                            "R105",
                            Owner::Task,
                            Phase::Setup,
                            format!(
                                "the readiness check `{}` failed {} times",
                                ready.run, failures
                            ),
                            "try the command by hand in `bench shell <task>` (Docker or Podman); if the service is only slow to start, give `ready` a start_period or more retries",
                        )
                        .detail(o.tail(2000)));
                    }
                }
                Err(e) => return Err(sandbox_problem(Phase::Setup, e)),
            }
            tokio::time::sleep(if in_grace {
                ready.start_interval
            } else {
                ready.interval
            })
            .await;
        }
    }

    async fn attempt(&mut self) -> Result<(), Problem> {
        let task = self.spec.task.clone();
        let timeout = agent_timeout(&task, &self.rt.options);
        let traj_dir = self.spec.dir.join("trajectory");
        let agent_dir = self.spec.dir.join("agent");
        let _ = std::fs::create_dir_all(&traj_dir);
        let _ = std::fs::create_dir_all(&agent_dir);
        let instruction = match &self.rt.options.extra_instruction {
            Some(extra) => format!("{}\n\n{}", task.instruction.trim_end(), extra),
            None => task.instruction.clone(),
        };
        let instruction = self.child_task_prompt(&instruction).unwrap_or(instruction);
        match self.spec.attempt.clone() {
            Attempt::Nop => {
                write_trajectory(&traj_dir, &[json!({"type": "nop"})]);
                self.agent.trajectory_summary = scripted_summary("nop");
                self.agent.prompts = vec![instruction];
                Ok(())
            }
            Attempt::Oracle => {
                let oracle = task.oracle.clone().expect("checked by refusal");
                self.sb()
                    .upload(&in_package(&task.root, &oracle.dir), &oracle.mount)
                    .await
                    .map_err(|e| crate::trial::put_problem(Phase::Agent, e, self.after_turn()))?;
                let _ = self
                    .root(
                        format!("chmod -R a+rX {m}", m = shell::quote(&oracle.mount)),
                        SETUP_TIMEOUT,
                    )
                    .await;
                let script = format!("{}/{}", oracle.mount.trim_end_matches('/'), oracle.entry);
                let label = match task.format {
                    Format::Split => "solution/solve.sh",
                    _ => "oracle/solve.sh",
                };
                let mut env = oracle.env.clone();
                env.insert("DEBIAN_FRONTEND".into(), "noninteractive".into());
                // Verifiers written for 0.8 grade an answer the oracle
                // printed, reading it where 0.8 writes it (MMMLU's
                // `cat /logs/agent/oracle.txt`). The oracle writes it there
                // itself, stdout and stderr in the order they were printed,
                // as 0.8's `> /logs/agent/oracle.txt 2>&1` does; the trial
                // folder's copy is the same stream (review E4-B, 2).
                self.run_script(
                    "oracle",
                    label,
                    &oracle_command(&script),
                    env,
                    timeout,
                    &instruction,
                )
                .await?;
                // The oracle's files are not the verifier's business.
                let _ = self
                    .root(
                        format!("rm -rf {m}", m = shell::quote(&oracle.mount)),
                        SETUP_TIMEOUT,
                    )
                    .await;
                Ok(())
            }
            Attempt::Control(control) => {
                let bytes =
                    std::fs::read(in_package(&task.root, &control.script)).map_err(|e| {
                        Problem::new(
                            "R306",
                            Owner::Task,
                            Phase::Agent,
                            format!("control {} is unreadable: {e}", control.id),
                            format!("fix {} in the task folder", control.script.display()),
                        )
                    })?;
                self.sb()
                    .write_file("/tmp/.benchflow-control.sh", &bytes, 0o755)
                    .await
                    .map_err(|e| crate::trial::put_problem(Phase::Agent, e, self.after_turn()))?;
                let label = format!(
                    "controls/{}",
                    control
                        .script
                        .file_name()
                        .map(|f| f.to_string_lossy().to_string())
                        .unwrap_or_default()
                );
                self.run_script(
                    "control",
                    &label,
                    &shell::script_file("/tmp/.benchflow-control.sh"),
                    BTreeMap::new(),
                    timeout,
                    &instruction,
                )
                .await
            }
            Attempt::Script(script) => {
                self.sb()
                    .write_file("/tmp/.benchflow-script.sh", &script.bytes, 0o755)
                    .await
                    .map_err(|e| crate::trial::put_problem(Phase::Agent, e, self.after_turn()))?;
                self.run_script(
                    "script",
                    &format!("script:{}", script.name),
                    &shell::script_file("/tmp/.benchflow-script.sh"),
                    BTreeMap::new(),
                    timeout,
                    &instruction,
                )
                .await
            }
            Attempt::Probe(probe) => {
                self.sb()
                    .write_file("/tmp/.benchflow-probe.sh", probe.script.as_bytes(), 0o755)
                    .await
                    .map_err(|e| crate::trial::put_problem(Phase::Agent, e, self.after_turn()))?;
                let t = Some(timeout.map_or(PROBE_TIMEOUT, |t| t.min(PROBE_TIMEOUT)));
                self.run_script(
                    "probe",
                    &format!("probe:{}", probe.id),
                    &shell::script_file("/tmp/.benchflow-probe.sh"),
                    BTreeMap::new(),
                    t,
                    &instruction,
                )
                .await
            }
            Attempt::Model(agent) => {
                // A branch child at `prompt:N` resumes the parent's session,
                // and fails rather than start a fresh one.
                let resume = match self.child_session() {
                    Some(id) => Some(crate::agent::Resume {
                        session_id: Some(id),
                        loaded: true,
                    }),
                    None => self.place_session(agent.as_ref()).await?,
                };
                let agent_sb = crate::confine::AgentSandbox::new(self.sb());
                let mut env = task.sandbox.env.clone();
                env.extend(self.rt.options.agent_env.clone());
                self.skills_env(&mut env);
                let stages = task.stages.iter().map(|s| s.prompt.clone()).collect();
                let ctx = AgentContext {
                    sandbox: &agent_sb,
                    user: &self.user,
                    home: &self.home,
                    workdir: &self.workdir,
                    instruction: &instruction,
                    stages,
                    timeout,
                    idle_timeout: self.rt.options.idle_timeout,
                    max_tool_calls: self.rt.options.max_tool_calls,
                    max_tokens: self.rt.options.max_tokens,
                    env,
                    trial_dir: self.spec.dir.clone(),
                    trial: &self.spec.name,
                    events: &self.rt.events,
                    mcp: &task.sandbox.mcp,
                    resume,
                    needs_load: self.parent_resumes(),
                    earlier_turns: self.child_after_agent(),
                };
                let out = agent.run(ctx).await;
                if let Some(loaded) = self.loaded.as_mut() {
                    loaded["resumed"] = json!(out.resumed);
                }
                let problem = out.problem.clone();
                self.agent = out;
                match problem {
                    Some(p) => Err(p),
                    None => Ok(()),
                }
            }
        }
    }

    /// Run a scripted attempt as the agent user, in the workdir.
    async fn run_script(
        &mut self,
        kind: &str,
        label: &str,
        command: &str,
        env: BTreeMap<String, String>,
        timeout: Option<Duration>,
        instruction: &str,
    ) -> Result<(), Problem> {
        let task = self.spec.task.clone();
        // The reference solution runs as root unless the task names a user,
        // as BenchFlow 0.8 and Harbor run it: solutions install packages and
        // write system files. Controls and probes stand in for an agent, so
        // they run as the agent user.
        let as_root = kind == "oracle" && task.agent.user.is_none();
        let (user, home) = if as_root {
            let _ = self
                .root(
                    hardening::trust_workdir_for_git(&self.workdir),
                    SETUP_TIMEOUT,
                )
                .await;
            ("root".to_string(), "/root".to_string())
        } else {
            (self.user.clone(), self.home.clone())
        };
        let mut exec = Exec::new(command)
            .user(user)
            .cwd(self.workdir.clone())
            .envs(env)
            .timeout_opt(timeout);
        exec.env.insert("HOME".into(), home);
        let out = crate::confine::AgentSandbox::new(self.sb())
            .exec(exec)
            .await
            .map_err(|e| sandbox_problem(Phase::Agent, e))?;
        let combined = out.combined();
        let _ = std::fs::write(
            self.spec.dir.join("agent").join(format!("{kind}.txt")),
            &combined,
        );
        let tail: String = {
            let n = combined.chars().count();
            combined.chars().skip(n.saturating_sub(4000)).collect()
        };
        write_trajectory(
            &self.spec.dir.join("trajectory"),
            &[
                json!({"type": kind, "command": label, "return_code": out.exit_code, "stdout": tail, "timed_out": out.timed_out}),
            ],
        );
        self.agent.trajectory_summary = scripted_summary(kind);
        self.agent.prompts = vec![instruction.to_string()];
        self.agent.timed_out = out.timed_out;
        Ok(())
    }

    async fn handoff(&mut self) -> Result<(), Problem> {
        if self.user != "root" {
            let out = self
                .root(hardening::kill_user(&self.user), SETUP_TIMEOUT)
                .await
                .map_err(|e| sandbox_problem(Phase::Handoff, e))?;
            let report = Report::parse(&out.stdout);
            if !out.success() {
                defense(
                    &mut self.defenses,
                    "kill_before_grading",
                    DefenseState::Failed,
                    report.first("survivors").map(str::to_string),
                );
                return Err(Problem::new(
                    "R305",
                    Owner::Sandbox,
                    Phase::Handoff,
                    "agent processes survived the kill before grading, so the trial was not graded",
                    "retry; if it repeats for this task, report it at https://github.com/benchflow-ai/benchflow/issues with the trial folder",
                )
                .detail(out.combined()));
            }
            let Some(killed) = report.first("killed") else {
                defense(
                    &mut self.defenses,
                    "kill_before_grading",
                    DefenseState::Failed,
                    Some("the kill program did not report back".into()),
                );
                return Err(unreported("the kill before grading", &out));
            };
            // Each graded stage's handoff kills too: add to the earlier ones'.
            self.custody.killed_processes +=
                killed.parse::<u64>().unwrap_or(0) + std::mem::take(&mut self.killed_before_save);
            let mut note = format!("{} processes", self.custody.killed_processes);
            if let Some(why) = &self.turn_kill_failed {
                note += &format!("; {why}");
            }
            defense(
                &mut self.defenses,
                "kill_before_grading",
                DefenseState::Enforced,
                Some(note),
            );
        } else {
            defense(
                &mut self.defenses,
                "kill_before_grading",
                DefenseState::Off,
                Some("the agent ran as root".into()),
            );
        }
        let verifier_user = self.task().verifier.user.clone();
        // Whoever worked in the sandbox could not act as root: a change in
        // BenchFlow's state folder is then the agent's doing, and stops the
        // grading. Otherwise (a root agent, one with a way to root, the
        // reference solution run as root) it is noted and grading goes on,
        // as before the check existed.
        let rootless_agent = self.rootless_agent();
        let root_solution =
            matches!(self.spec.attempt, Attempt::Oracle) && self.task().agent.user.is_none();
        let strict = rootless_agent && !root_solution;
        let out = self
            .root(
                hardening::handoff(&self.workdir, verifier_user.as_deref(), strict),
                SETUP_TIMEOUT,
            )
            .await
            .map_err(|e| sandbox_problem(Phase::Handoff, e))?;
        let r = Report::parse(&out.stdout);
        // The last line alone says how it ended: a path it printed earlier
        // cannot pass for it.
        let last = out.stdout.lines().rev().find(|l| !l.trim().is_empty());
        let tampered = r.all("state-owner");
        let unchecked = r.first("state-unchecked");
        if last == Some("BF handoff stopped") {
            if !tampered.is_empty() {
                defense(
                    &mut self.defenses,
                    "state_custody",
                    DefenseState::Failed,
                    Some(format!("not root's alone: {}", tampered.join(", "))),
                );
                return Err(Problem::new(
                    "R314",
                    Owner::Agent,
                    Phase::Handoff,
                    format!(
                        "BenchFlow's root-only state folder was changed before grading ({}), so the baseline grading compares against cannot be trusted and the trial was not graded",
                        tampered.join(", ")
                    ),
                    "read the agent's trajectory for what it did to that folder; if nothing it did explains it, report it at https://github.com/benchflow-ai/benchflow/issues with the trial folder",
                )
                .detail(out.combined()));
            }
            if let Some(why) = unchecked {
                defense(
                    &mut self.defenses,
                    "state_custody",
                    DefenseState::Failed,
                    Some("the state folder could not be checked".into()),
                );
                return Err(if why == "no-find" {
                    Problem::new(
                        "R307",
                        Owner::Task,
                        Phase::Handoff,
                        "the image has no `find`, which the handoff before grading needs to check BenchFlow's state folder and clean up after the agent, so the trial was not graded",
                        "add `find` to the image (findutils, or busybox's); `bench explain R307` has more",
                    )
                } else {
                    Problem::new(
                        "R305",
                        Owner::Sandbox,
                        Phase::Handoff,
                        "the check of BenchFlow's state folder before grading failed (`find` exited with an error), so the trial was not graded",
                        "retry; if it repeats for this task, report it at https://github.com/benchflow-ai/benchflow/issues with the trial folder",
                    )
                }
                .detail(out.combined()));
            }
        }
        // A planted or changed test-runner file that could not be removed
        // or put back (a full disk: E6-E-A H2, E6M-B F1): not graded with
        // it in place. Whether the agent filled the disk is decided with
        // the sandbox's own evidence (`exhaust`); without it, the
        // sandbox's.
        let unrestored = r.all("restore-failed");
        if last == Some("BF handoff stopped") && !unrestored.is_empty() {
            defense(
                &mut self.defenses,
                "special_file_custody",
                DefenseState::Failed,
                Some(format!(
                    "could not remove or put back: {}",
                    unrestored.join(", ")
                )),
            );
            return Err(Problem::new(
                "R305",
                Owner::Sandbox,
                Phase::Handoff,
                format!(
                    "a test-runner file the agent planted or changed ({}) could not be removed or put back before grading, so the trial was not graded; a full disk is the usual cause",
                    unrestored.join(", ")
                ),
                "look at what filled the sandbox's disk; if it was not the agent, give the task more disk (`[sandbox] disk`)",
            )
            .detail(out.combined()));
        }
        if out.timed_out {
            for name in [
                "clean_verifier_logs",
                "special_file_custody",
                "state_custody",
            ] {
                defense(
                    &mut self.defenses,
                    name,
                    DefenseState::Failed,
                    Some("the handoff ran out of time".into()),
                );
            }
            return Err(Problem::new(
                "R305",
                Owner::Sandbox,
                Phase::Handoff,
                format!(
                    "the handoff before grading did not finish in {} s, so the trial was not graded; very many files in the workdir or the temp folders (often ones the agent left) are the usual cause",
                    SETUP_TIMEOUT.as_secs()
                ),
                "look at what the agent left in the workdir and /tmp; if the task's own files are that many, report it at https://github.com/benchflow-ai/benchflow/issues with the trial folder",
            )
            .detail(out.tail(1500)));
        }
        if last != Some("BF handoff done") || !out.success() {
            for name in [
                "clean_verifier_logs",
                "special_file_custody",
                "state_custody",
            ] {
                defense(
                    &mut self.defenses,
                    name,
                    DefenseState::Failed,
                    Some("the handoff program did not report back".into()),
                );
            }
            return Err(unreported("the handoff", &out));
        }
        // A task with stages hands off before each graded stage: add each
        // handoff's findings to the earlier ones', so custody.json keeps
        // what the first stage's handoff removed.
        let add = |list: &mut Vec<String>, v: Vec<&str>| {
            for f in v {
                if !list.iter().any(|x| x == f) {
                    list.push(f.to_string());
                }
            }
        };
        add(&mut self.custody.special_files, r.all("special"));
        add(&mut self.custody.escaping_symlinks, r.all("symlink"));
        add(&mut self.custody.repaired_links, r.all("repaired"));
        add(&mut self.custody.planted, r.all("planted"));
        add(&mut self.custody.restored, r.all("restored"));
        add(&mut self.custody.tmp_modules, r.all("tmp-module"));
        add(&mut self.custody.large_files, r.all("large"));
        defense(
            &mut self.defenses,
            "clean_verifier_logs",
            DefenseState::Enforced,
            None,
        );
        // Without `find` (a lenient handoff goes on without it), the
        // handoff could not look for special files or planted ones.
        if unchecked == Some("no-find") {
            defense(
                &mut self.defenses,
                "special_file_custody",
                DefenseState::Failed,
                Some("the image has no `find`, so the handoff could not look for special or planted files".into()),
            );
        } else {
            defense(
                &mut self.defenses,
                "special_file_custody",
                DefenseState::Enforced,
                None,
            );
        }
        if strict {
            defense(
                &mut self.defenses,
                "state_custody",
                DefenseState::Enforced,
                Some("BenchFlow's state folder was root's alone at the handoff".into()),
            );
        } else if !(root_solution && r.all("state-noted").is_empty() && unchecked.is_none()) {
            // The reference solution is the task's own and trusted: the
            // defense is recorded for it only when its check found something.
            let who = if root_solution {
                "the reference solution ran as root"
            } else {
                "the agent ran as root or had a way to become root (see agent_user)"
            };
            let noted = r.all("state-noted");
            let unchecked = match unchecked {
                None => String::new(),
                Some("") => "; the folder could not be checked".to_string(),
                Some(why) => format!("; the folder could not be checked ({why})"),
            };
            let seen = if noted.is_empty() {
                String::new()
            } else {
                format!("; not root's alone: {}", noted.join(", "))
            };
            defense(
                &mut self.defenses,
                "state_custody",
                DefenseState::Off,
                Some(format!(
                    "{who}, so it could have changed BenchFlow's state folder without a trace{seen}{unchecked}"
                )),
            );
        }
        if self.user != "root" {
            let owner = match verifier_user.as_deref() {
                Some(u) if u != "root" && u != "0" => u.to_string(),
                _ => "root".to_string(),
            };
            let out = self
                .root(
                    hardening::own_workdir(&self.workdir, &owner, &self.user),
                    OWN_TIMEOUT,
                )
                .await
                .map_err(|e| sandbox_problem(Phase::Handoff, e))?;
            // As 0.8, a workdir that could not be given to its owner is not
            // graded: the verifier would run against the agent's ownership.
            let (owned, kept) = own_result(
                &out,
                &owner,
                "giving the workdir to the verifier",
                Phase::Handoff,
            )?;
            self.custody.workdir_owner = Some(owned);
            self.custody.workdir_unchanged.extend(kept);
        }
        let preload = crate::confine::check_preload(self.sb())
            .await
            .map_err(|e| sandbox_problem(Phase::Handoff, e))?;
        let d = preload?;
        self.defenses.insert(d.name.clone(), d);
        Ok(())
    }

    async fn verifier_env(&mut self) -> Result<BTreeMap<String, String>, Problem> {
        let task = self.spec.task.clone();
        let mut env = task.verifier.env.clone();
        let out = self
            .root(hardening::trusted_path(&self.workdir), SETUP_TIMEOUT)
            .await
            .map_err(|e| sandbox_problem(Phase::Verify, e))?;
        let Some(path) = Report::parse(&out.stdout).first("path").map(str::to_string) else {
            defense(
                &mut self.defenses,
                "trusted_path",
                DefenseState::Failed,
                Some("the PATH check did not report back".into()),
            );
            return Err(unreported("the trusted PATH check", &out));
        };
        env.insert("PATH".into(), path.clone());
        defense(
            &mut self.defenses,
            "trusted_path",
            DefenseState::Enforced,
            Some(path),
        );
        env.entry("PYTEST_ADDOPTS".into())
            .or_insert_with(|| "-p no:cacheprovider".into());
        env.insert("PYTHONDONTWRITEBYTECODE".into(), "1".into());
        crate::confine::add_safepath(&mut env, &task.verifier.env, task.format);
        Ok(env)
    }

    /// Make `path` safe for root to write into ([`hardening::claim`]), and
    /// note what was removed in the custody report.
    pub(crate) async fn claim(
        &mut self,
        path: &str,
        foreign: bool,
        phase: Phase,
    ) -> Result<(), Problem> {
        let out = self
            .root(hardening::claim(path, foreign), SETUP_TIMEOUT)
            .await
            .map_err(|e| sandbox_problem(phase, e))?;
        let r = Report::parse(&out.stdout);
        self.custody
            .escaping_symlinks
            .extend(r.all("symlink").into_iter().map(str::to_string));
        self.custody
            .planted
            .extend(r.all("planted").into_iter().map(str::to_string));
        Ok(())
    }

    /// Run the verifier in the agent's sandbox, after the handoff.
    async fn verify_shared(&mut self) -> Result<scoring::Score, Problem> {
        self.enforce_network(netpolicy::Stage::Verifier, netpolicy::Host::Agent)
            .await?;
        let env = self.verifier_env().await?;
        self.run_verifier(env).await
    }

    async fn run_verifier(
        &mut self,
        env: BTreeMap<String, String>,
    ) -> Result<scoring::Score, Problem> {
        let task = self.spec.task.clone();
        let v = &task.verifier;
        // The verifier's files must not mix with any the agent left at
        // their mount, or reach it through a link the agent made. (A
        // separate verifier's fresh sandbox has had no agent in it.)
        let agent_was_here = v.isolation == Isolation::Shared && self.regrade.is_none();
        self.claim(&v.mount, agent_was_here, Phase::Verify).await?;
        for dir in self.verifier_dirs() {
            self.sb()
                .upload(&dir, &v.mount)
                .await
                .map_err(|e| crate::trial::put_problem(Phase::Verify, e, agent_was_here))?;
        }
        let mount = shell::quote(&v.mount);
        let _ = self
            .root(
                format!("chmod -R a+rX {mount}; find {mount} -name '*.sh' -exec chmod +x {{}} +"),
                SETUP_TIMEOUT,
            )
            .await;
        // Verifiers carried over from older benchmarks name `/tests` (SkillsBench's
        // `pytest --confcutdir=/tests`, `cd /tests`); as 0.8 does, `/tests` points
        // at a verifier mounted elsewhere, unless the image ships one (locked to
        // root before the agent ran).
        if v.mount.trim_end_matches('/') != LEGACY_VERIFIER_MOUNT {
            let _ = self.root(legacy_tests_link(&v.mount), SETUP_TIMEOUT).await;
        }
        let entry = format!("{}/{}", v.mount.trim_end_matches('/'), v.entry);
        // The verifier starts in the workdir, as 0.8 runs test.sh in the
        // sandbox's working folder (verifiers open the agent's outputs by
        // relative path); a `verifier.md` script strategy starts in its own
        // folder, as 0.8 runs it with `cd <verifier folder> && <command>`.
        let cwd = if v.starts_in_mount {
            v.mount.clone()
        } else {
            self.workdir.clone()
        };
        let timeout = self
            .rt
            .options
            .verifier_timeout(self.x.verifier_timeout.or(v.timeout));
        let unset = hardening::unset_vars(&v.env);
        let safepath_off = crate::confine::safepath_off(task.format, &v.env);
        let mut exec = Exec::new(crate::confine::verifier_command(&unset, &entry))
            .cwd(cwd)
            .envs(env)
            .timeout(timeout);
        exec.user = Some(v.user.clone().unwrap_or_else(|| "root".into()));
        let clean = crate::confine::check_verifier_env(
            self.sb(),
            &exec,
            &unset,
            safepath_off,
            Some(&self.image_loader),
        )
        .await
        .map_err(|e| sandbox_problem(Phase::Verify, e))?;
        let d = clean?;
        self.defenses.insert(d.name.clone(), d);
        // The command unsets the loader variables too, but only once its own
        // shell runs, and that shell (root, as a rule) would already have
        // loaded or read what the image's `LD_PRELOAD` or `BASH_ENV` names.
        // The backend drops them before anything starts; the check above
        // ran without this, so it still reports what was dropped.
        let out = self
            .sb()
            .exec(exec.unset(unset.iter().copied()))
            .await
            .map_err(|e| sandbox_problem(Phase::Verify, e))?;
        self.verifier_exit = Some(out.exit_code);
        let vdir = self.spec.dir.join("verifier");
        let _ = std::fs::create_dir_all(&vdir);
        // Files an earlier, killed attempt left here are never graded.
        clear_folder(&vdir);
        // Under a deadline as well as the cap (review F2-A-r8 M2): in a
        // shared verifier, a root agent's leftover processes still run, and
        // one can make the copy never end. A copy past either is one that
        // broke off.
        let copy = || async {
            crate::copy_out::copy_out(
                self.sb(),
                "/logs/verifier",
                &vdir,
                VERIFIER_COPY_CAP,
                crate::copy_out::COPY_OUT_TIMEOUT,
                crate::copy_out::Shape::Any,
            )
            .await
            .map_err(crate::copy_out::CopyOutError::into_sandbox)
        };
        let mut downloaded = copy().await;
        // A copy that broke off (a dropped connection, a file that changed
        // as it was read) is made once more, from scratch, after a pause.
        if let Err(e) = &downloaded
            && !remote_missing(e)
        {
            self.rt.events.log(
                self.name(),
                format!("could not download /logs/verifier: {e}; trying once more"),
            );
            clear_folder(&vdir);
            tokio::time::sleep(VERIFIER_COPY_PAUSE).await;
            downloaded = copy().await;
        }
        // One that broke off again leaves some of the verifier's files and
        // not others: grading them would score the copy, not the agent.
        // (A folder the verifier never left gets no reward, below.)
        let broken = match downloaded {
            Err(e) if !remote_missing(&e) => {
                self.rt.events.log(
                    self.name(),
                    format!("could not download /logs/verifier: {e}"),
                );
                clear_folder(&vdir);
                Some(e)
            }
            _ => None,
        };
        // A failed copy removes the folder, a missing one included.
        let _ = std::fs::create_dir_all(&vdir);
        let stdout_file = vdir.join("test-stdout.txt");
        if !stdout_file.exists() {
            let _ = std::fs::write(&stdout_file, out.combined());
        }
        if out.timed_out {
            return Err(Problem::new(
                "R303",
                Owner::Task,
                Phase::Verify,
                format!("the verifier ran past its {}s timeout", timeout.as_secs()),
                "install test dependencies in the image rather than in test.sh, or raise [verifier] timeout",
            )
            .detail(out.tail(1500)));
        }
        if let Some(e) = broken {
            return Err(self.copy_problem(e).await.detail(out.tail(1500)));
        }
        let judged = self.judge_rubric(&vdir).await?;
        match scoring::score(&task, &vdir, judged.as_ref().map(|(j, m)| (j, m.as_str()))) {
            // A score that did not pass on a full disk (review F2-C, F-6):
            // the agent's when it filled its own disk, the sandbox's when
            // the disk is shown full by what the agent cannot fake.
            Ok(score) if !score.passed => {
                match self
                    .full_disk_instead(&out.combined(), agent_was_here)
                    .await
                {
                    Some(p) => Err(p.detail(format!(
                        "the verifier scored {} before this: {}",
                        score.reward,
                        out.tail(1500)
                    ))),
                    None => Ok(score),
                }
            }
            Ok(score) => Ok(score),
            Err(mut p) => {
                // No reward, because the sandbox died under the verifier, is
                // not the task's.
                if p.code == "R301"
                    && let Some(gone) = self.sandbox_gone().await
                {
                    return Err(gone.detail(out.tail(1500)));
                }
                if out.exit_code != 0 && p.code == "R301" {
                    p = Problem::new(
                        "R304",
                        Owner::Task,
                        Phase::Verify,
                        format!(
                            "the verifier exited with code {} and wrote no reward",
                            out.exit_code
                        ),
                        "make the verifier write a reward on every path, even when tests fail; verifier/test-stdout.txt in the trial folder has its output",
                    );
                }
                // A full disk is not a verifier bug: say so.
                if matches!(p.code.as_str(), "R301" | "R304")
                    && out.combined().contains("No space left on device")
                {
                    p.message.push_str(" (the sandbox's disk was full)");
                    p.next = "give the task more disk (`[sandbox] disk`), or see what filled it: the agent's files are in the trial's outputs when saved".into();
                }
                Err(p.detail(out.tail(1500)))
            }
        }
    }

    /// When no reward came back because the sandbox went away, the problem
    /// that says so.
    async fn sandbox_gone(&self) -> Option<Problem> {
        let e = self.unreachable().await?;
        Some(lost(
            Phase::Verify,
            "no reward came back from the verifier",
            e,
        ))
    }

    /// A copy of the verifier's folder that failed twice, as a problem.
    /// The sandbox lost (the backend no longer has it), or the connection
    /// to one that runs failing, is the sandbox's (R103, rerun). This
    /// machine's own disk, quota or permissions, or its job folder gone, is
    /// yours (R107, rerun): the copy's caps (content, entries, header size)
    /// keep the verifier from filling this machine itself, except where a
    /// Docker sandbox with no disk limit shares this machine's disk. What
    /// the folder causes (past a cap, files `tar` refuses, names this
    /// machine cannot store, no `tar`, an empty or broken archive), or a
    /// connection that
    /// fails because code inside stopped the sandbox, is the task's
    /// (R301), and never rerun: the verifier may have run the agent's code.
    async fn copy_problem(&self, e: SandboxError) -> Problem {
        if let Some(gone) = self.unreachable().await {
            let what =
                format!("the verifier's results could not be copied out of the sandbox ({e})");
            return lost(Phase::Verify, &what, gone);
        }
        if host_side_io(&e) {
            return sandbox_problem(Phase::Verify, e);
        }
        let channel = matches!(
            e,
            SandboxError::Transport(_)
                | SandboxError::Throttled(_)
                | SandboxError::Unavailable { .. }
                | SandboxError::Start(_)
                | SandboxError::Image { .. }
        );
        // A sandbox stopped from outside, or by something the backend
        // cannot name (`Halted`), is lost; one code inside stopped is not.
        let runs = match self.sb().presence().await {
            Ok(Presence::Running | Presence::Halted) | Err(SandboxError::Unsupported(_)) => true,
            Ok(_) => false,
            Err(_) => true,
        };
        if channel && runs {
            return lost(
                Phase::Verify,
                "the verifier's results could not be copied out of the sandbox",
                e,
            );
        }
        let why = if channel {
            "the sandbox stopped under the verifier (code inside it can stop it)"
        } else {
            "something in /logs/verifier stopped the copy"
        };
        Problem::new(
            "R301",
            Owner::Task,
            Phase::Verify,
            format!(
                "the verifier's results could not be copied out of the sandbox, so the trial was not graded: {why}: {e}"
            ),
            format!(
                "make the verifier write only small files (under {} GiB in all, and at most 100,000 files, links and folders) that it can read to /logs/verifier, and keep `tar` in the image; verifier/test-stdout.txt in the trial folder has the verifier's output",
                VERIFIER_COPY_CAP >> 30
            ),
        )
    }

    /// The error for a sandbox its backend no longer has, or cannot be
    /// asked about. The backend decides (`docker inspect`, Daytona's API),
    /// not a command in the sandbox: an agent that replaced the shell could
    /// answer one with "No such container" and have its trial rerun. A
    /// sandbox the backend says code inside stopped is not lost: root code
    /// inside can stop it (`kill -TERM 1`), and a failure it causes stays
    /// its own. A sandbox the backend stopped or lost itself is (the
    /// backend answers with an error). A backend that cannot tell (the test
    /// fakes) is asked to run `true`.
    async fn unreachable(&self) -> Option<SandboxError> {
        self.not_running(false).await
    }

    /// [`Self::unreachable`]; with `stopped_is_lost`, a sandbox stopped from
    /// inside is lost too.
    async fn not_running(&self, stopped_is_lost: bool) -> Option<SandboxError> {
        let sb = self.sandbox.as_ref()?;
        match sb.presence().await {
            Ok(Presence::Running) => None,
            Ok(Presence::Stopped) if !stopped_is_lost => None,
            Ok(Presence::Stopped) => Some(SandboxError::Transport(format!(
                "the {} backend reports sandbox {} stopped",
                sb.backend(),
                sb.id()
            ))),
            Ok(Presence::Gone) => Some(SandboxError::Transport(format!(
                "the {} backend no longer has sandbox {}: something outside the run removed it",
                sb.backend(),
                sb.id()
            ))),
            Ok(Presence::Halted) => Some(SandboxError::Transport(format!(
                "the {} backend reports sandbox {} stopped, and cannot say what stopped it",
                sb.backend(),
                sb.id()
            ))),
            Err(SandboxError::Unsupported(_)) => self
                .root("true".into(), Duration::from_secs(30))
                .await
                .err(),
            Err(e) => Some(e),
        }
    }

    /// One rule for every loss of a trial's sandbox, whatever the phase
    /// (setup, the agent, the handoff, the verifier, saving) or the owner
    /// of the failure (the sandbox's R103, or a sandbox error behind an
    /// agent's or a script's failure), with one counter per trial (convergence
    /// reviews E9-D-A 1, E9-D-B 1):
    /// - a loss the backend confirms in its own answer (the sandbox gone,
    ///   stopped from outside, the control plane's refusal or outage) is
    ///   rerun every time: nothing inside can cause it;
    /// - a sandbox code inside stopped while the agent worked is the
    ///   agent's own failure, not run again (a rerun would reward it);
    /// - any other loss (the backend cannot name the cause, or the sandbox
    ///   still runs and the connection to it broke) is rerun once per
    ///   trial; after one such loss in any earlier attempt, it is capped:
    ///   not run again, and counted as not passed. Code inside the sandbox
    ///   can break the connection every time, and an outage only the
    ///   sandbox's own channel shows looks the same.
    async fn lost_again(&self, p: Problem) -> Problem {
        if matches!(p.phase, Phase::Image | Phase::Start)
            || matches!(p.code.as_str(), "R000" | "R001")
            || p.loss_confirmed == Some(true)
        {
            return p;
        }
        let Some(sb) = self.sandbox.as_ref() else {
            return p;
        };
        // BenchFlow stopped it (the disk limit): not a loss, and not code
        // inside stopping it; but the agent may have filled the layer, and
        // nothing shows it did, so it counts as an unexplained failure.
        if p.code == exhaust::EXHAUSTED {
            return p;
        }
        // A sandbox the disk limit stopped, or hardening before grading
        // that failed (a full disk or process table nothing shows the agent
        // caused, a stuck process): the sandbox's, run again once; a repeat
        // is kept as it is, never scored (decision 21, as clarified).
        if matches!(p.code.as_str(), exhaust::DISK_STOP | "R305") {
            return self.capped_if_before(p);
        }
        let presence = sb.presence().await;
        if p.phase == Phase::Agent && matches!(presence, Ok(Presence::Stopped)) {
            let mut own = Problem::new(
                "R204",
                Owner::Agent,
                Phase::Agent,
                format!(
                    "the agent's sandbox stopped under it, and the backend says code inside stopped it, so it is not run again: {}",
                    p.message
                ),
                "look at what the agent does to its sandbox (a root agent can stop it); `--rerun errored` runs the trial again",
            )
            .detail(p.detail.clone().unwrap_or_default());
            own.loss_confirmed = Some(false);
            own.rerun_capped = true;
            return own;
        }
        let loss = p.code == "R103" || (p.phase == Phase::Agent && p.owner == Owner::Sandbox);
        if !loss {
            return p;
        }
        let confirmed = match presence {
            Ok(Presence::Running | Presence::Stopped | Presence::Halted) => false,
            Ok(Presence::Gone) => true,
            Err(SandboxError::Unsupported(_)) => self
                .root("true".into(), Duration::from_secs(30))
                .await
                .is_err(),
            Err(_) => true,
        };
        let mut p = p;
        p.loss_confirmed = Some(confirmed);
        if confirmed {
            return p;
        }
        self.capped_if_before(p)
    }

    /// `p`, an unexplained failure of the sandbox, capped when an earlier
    /// attempt of the trial had one too ([`lost_before`]): not run again.
    fn capped_if_before(&self, mut p: Problem) -> Problem {
        p.loss_confirmed = Some(false);
        let label = self.spec.attempt.label();
        let model = self.spec.attempt.model();
        let same = Same {
            digest: &self.spec.task.digest,
            agent: &label,
            model: model.as_deref(),
        };
        if !lost_before(&self.spec.dir, &same) {
            return p;
        }
        p.message = if p.code == "R103" || p.phase == Phase::Agent && p.code != exhaust::DISK_STOP {
            format!(
                "{} (an earlier attempt of this trial lost its sandbox the same way, and the backend still has this one, so it is not run again: code inside the sandbox can break it)",
                p.message
            )
        } else {
            format!(
                "{} (an earlier attempt of this trial also failed in a way the backend could not explain, so it is not run again: code inside the sandbox can cause it; it stays the sandbox's error, never a score)",
                p.message
            )
        };
        p.next = "check the backend first (`bench doctor`); if it is healthy, look at what the agent's code and the verifier do to the sandbox; `--rerun errored` runs the trial again".into();
        p.transient = false;
        p.rerun_capped = true;
        p
    }

    /// A hardening program that did not report back (R307), a network
    /// self-test that failed (R109), or a setup step that failed (R104),
    /// because the sandbox went away (deleted,
    /// or the backend lost it) says nothing about the image or BenchFlow:
    /// the sandbox failed (R103), and the trial is retried. Any other
    /// problem, or one whose sandbox still runs, is returned as it is.
    pub(crate) async fn sandbox_lost_instead(&self, p: Problem) -> Problem {
        let what = match p.code.as_str() {
            "R307" => "the sandbox was lost while one of BenchFlow's checks ran",
            // E5: the Docker daemon killed during setup made `useradd` fail,
            // and the trial was the task's ("the image needs useradd").
            "R104" => "the sandbox was lost while BenchFlow prepared it",
            "R109" if p.owner != Owner::Sandbox => {
                "the sandbox was lost during the network self-test"
            }
            _ => return p,
        };
        match self.unreachable().await {
            Some(e) => lost(p.phase, what, e).detail(p.detail.clone().unwrap_or_default()),
            None => p,
        }
    }

    /// Judge the rubric's `llm` criteria on the host: fetch their evidence
    /// from the sandbox, then one model call per criterion. `None` when the
    /// rubric has none. A criterion left undecided leaves the trial
    /// unscored (R312); judge settings the task got wrong are R313.
    async fn judge_rubric(
        &mut self,
        vdir: &Path,
    ) -> Result<Option<(crate::judge::Judged, String)>, Problem> {
        let task = self.spec.task.clone();
        let plan =
            crate::judge::plan(&task, self.rt.options.judge_model.as_deref()).map_err(|e| {
                Problem::new(
                    "R313",
                    Owner::Task,
                    Phase::Verify,
                    format!("the rubric's judge settings cannot be used: {e}"),
                    "fix verifier/rubric.json (`bench check` checks the judge settings)",
                )
            })?;
        let Some(plan) = plan else {
            return Ok(None);
        };
        self.rt.events.log(
            self.name(),
            format!(
                "judging {} rubric criteria with {}",
                plan.items.len(),
                plan.model
            ),
        );
        let evidence = crate::judge::fetch(self.sb(), &self.workdir, &plan)
            .await
            .map_err(|e| lost(Phase::Verify, "the rubric's evidence could not be read", e))?;
        let reviewer = crate::judge::connect(&plan, &self.rt.options.judge_env.0)
            .await
            .map_err(|e| {
                Problem::new(
                    "R312",
                    Owner::Model,
                    Phase::Verify,
                    format!(
                        "the rubric judge {} could not be reached, so no model-judged criterion was decided: {}",
                        plan.model, e.message
                    ),
                    e.next,
                )
            })?;
        let judged = crate::judge::judge(&plan, &evidence, &reviewer, vdir).await;
        if !judged.failures.is_empty() {
            let (id, why) = &judged.failures[0];
            return Err(Problem::new(
                "R312",
                Owner::Model,
                Phase::Verify,
                format!(
                    "the rubric judge {} decided {} of {} criteria, so the trial is not scored; {id}: {why}",
                    plan.model,
                    judged.verdicts.len(),
                    plan.items.len()
                ),
                "verifier/judgments.jsonl in the trial folder has every call; run again, or pass --judge-model with another model",
            ));
        }
        Ok(Some((judged, plan.model)))
    }

    /// Run the verifier in a fresh sandbox that holds only the agent's
    /// outputs.
    async fn verify_separate(&mut self) -> Result<scoring::Score, Problem> {
        // Take custody of the outputs, then retire the agent's sandbox,
        // its disk watch ended first.
        let archive = self.collect_outputs().await?;
        if let Some(p) = self.end_watch(None).await {
            return Err(p);
        }
        if let Some(sb) = self.sandbox.take() {
            if self.rt.options.keep == Keep::Never {
                if let Err(e) = sb.destroy().await {
                    self.rt.events.log(
                        self.name(),
                        format!("could not destroy sandbox {}: {e}", sb.id()),
                    );
                    note_left_behind(sb.as_ref());
                }
            } else {
                self.retired.push(sb);
            }
        }
        self.verify_archive(&archive).await
    }

    /// Start a fresh sandbox from the task's image, restore the outputs in
    /// `archive` at their own paths, and run the verifier there.
    async fn verify_archive(&mut self, archive: &Path) -> Result<scoring::Score, Problem> {
        let task = self.spec.task.clone();
        let mut base = collect::verifier_image_spec(&task)
            .unwrap_or_else(|| images::spec_for(&task, Vec::new()));
        if let Some(layer) = netpolicy::firewall_layer(
            &task,
            netpolicy::Host::SeparateVerifier,
            &self.rt.provider.capabilities(),
            &self.rt.options.sandbox_user,
        ) {
            base.layers.insert(0, layer);
        }
        let log = |_: &str| {};
        let image = self
            .rt
            .images
            .get(self.rt.provider.as_ref(), &base, &log, None)
            .await?;
        let sandbox = self
            .start_sandbox(&image, netpolicy::Host::SeparateVerifier)
            .await?;
        self.sandbox_id = Some(sandbox.id().to_string());
        self.sandbox = Some(sandbox);
        self.note_verifier_image();
        self.resolve_workdir().await?;
        let _ = self.root(hardening::prepare_root(&[]), SETUP_TIMEOUT).await;
        // Restore the outputs at their own paths in the fresh sandbox: only
        // what the filter keeps, the same rule a live trial saved them by
        // (a regrade's archive, packed from `outputs/`, loses the folders
        // above the outputs here).
        let mut bytes = Vec::new();
        let paths = self.restore_paths.clone().unwrap_or_else(|| {
            output_paths(&self.spec.task, &self.workdir)
                .into_iter()
                .map(|(p, _)| p)
                .collect()
        });
        // In memory, under the copy-out's cap (review F2-RA-1).
        let capped = crate::custody::CappedVec::new(&mut bytes, crate::copy_out::ARCHIVE_CAP);
        let budget = self.unpack_budget();
        let filtered = crate::custody::filter_within(archive, capped, &paths, budget)
            .map_err(|e| filter_problem(e, Phase::Verify))?;
        self.note_refused(filtered, false);
        self.sb()
            .write_file(hardening::OUTPUTS_TAR, &bytes, 0o600)
            .await
            .map_err(|e| sandbox_problem(Phase::Verify, e))?;
        drop(bytes);
        // Then nothing the archive could have put in the verifier's log
        // folder is there when the verifier starts (a planted
        // `reward.json`). The verifier's mount and `/tests` stay: an image
        // built for the verifier may ship files there.
        let out = self
            .root(
                format!(
                    "cd / && tar -xf {t} && rm -f {t} && rm -rf /logs/verifier && mkdir -p /logs/verifier",
                    t = shell::quote(hardening::OUTPUTS_TAR)
                ),
                SETUP_TIMEOUT,
            )
            .await
            .map_err(|e| sandbox_problem(Phase::Verify, e))?;
        if !out.success() {
            return Err(Problem::new(
                "R103",
                Owner::Sandbox,
                Phase::Verify,
                "the outputs could not be restored in the verifier's sandbox",
                "retry; if it repeats, report it with the trial folder",
            )
            .detail(out.tail(1500)));
        }
        // The verifier starts in the workdir, which the verifier's image may
        // not have when the outputs live elsewhere; 0.8 makes it first too.
        let _ = self
            .root(
                format!("mkdir -p {}", shell::quote(&self.workdir)),
                SETUP_TIMEOUT,
            )
            .await;
        let mut note = "graded in a fresh sandbox holding the verifier's files and the saved outputs: regular files and folders inside the task's outputs, without setuid, setgid or sticky bits".to_string();
        match self.custody.outputs_refused_count {
            0 => {}
            n => note.push_str(&format!(
                "; {n} other entries of the outputs archive were dropped (custody.json lists them)"
            )),
        }
        defense(
            &mut self.defenses,
            "separate_verifier",
            DefenseState::Enforced,
            Some(note),
        );
        self.enforce_network(
            netpolicy::Stage::Verifier,
            netpolicy::Host::SeparateVerifier,
        )
        .await?;
        let env = self.verifier_env().await?;
        self.run_verifier(env).await
    }

    /// Pack the outputs in the sandbox under the custody caps, download the
    /// archive, and unpack it into the trial's `outputs/`. Returns the
    /// archive's path on the host.
    /// [`Self::collect_outputs_held`], with the disk watch held while the
    /// archive is in the sandbox, and the archive removed after.
    async fn collect_outputs(&mut self) -> Result<PathBuf, Problem> {
        self.hold_watch();
        let r = self.collect_outputs_held().await;
        self.release_watch(hardening::OUTPUTS_TAR).await;
        r
    }

    async fn collect_outputs_held(&mut self) -> Result<PathBuf, Problem> {
        let task = self.spec.task.clone();
        let paths: Vec<String> = output_paths(&task, &self.workdir)
            .into_iter()
            .map(|(p, _)| p)
            .collect();
        let out = self
            .root(hardening::pack_outputs(&paths), SETUP_TIMEOUT)
            .await
            .map_err(|e| sandbox_problem(Phase::Save, e))?;
        let r = Report::parse(&out.stdout);
        let own = |v: Vec<&str>| v.into_iter().map(str::to_string).collect::<Vec<_>>();
        self.custody.skipped_special = own(r.all("skipped-special"));
        self.custody.skipped_large = own(r.all("skipped-large"));
        if !out.success() {
            return Err(Problem::new(
                "R103",
                Owner::Sandbox,
                Phase::Save,
                "the outputs could not be packed",
                "retry; if it repeats, report it with the trial folder",
            )
            .detail(out.tail(1500)));
        }
        let raw = self.spec.dir.join("outputs.raw.tar");
        // Under a byte cap and a deadline (review F2-A-r8 H1): a root
        // agent's leftover process can swap the archive for a huge sparse
        // file, or make the copy never end.
        crate::copy_out::copy_out(
            self.sb(),
            hardening::OUTPUTS_TAR,
            &raw,
            crate::copy_out::ARCHIVE_CAP,
            crate::copy_out::COPY_OUT_TIMEOUT,
            crate::copy_out::Shape::File,
        )
        .await
        .map_err(|e| outputs_copy_problem(e, "the outputs"))?;
        // The archive came from the image's own `tar`, which a root agent
        // controls (review F2-A-r12 RESTORE): only what the filter keeps is
        // saved, given to a separate verifier, or regraded later, and what
        // it refuses is recorded, never an error.
        let archive = self.spec.dir.join("outputs.tar");
        let filtered = filter_into_file(&raw, &archive, &paths, self.unpack_budget());
        crate::copy_out::remove(&raw);
        let filtered = filtered.map_err(|e| {
            crate::copy_out::remove(&archive);
            filter_problem(e, Phase::Save)
        })?;
        self.note_refused(filtered, true);
        self.restore_paths = Some(paths);
        let outputs = self.spec.dir.join("outputs");
        let all = |rel: &Path| Some(rel.to_path_buf());
        match crate::custody::unpack_within(&archive, &outputs, &all, self.unpack_budget()) {
            Ok(u) => {
                self.custody.saved_files = u.files as u64;
                self.custody.saved_bytes = u.bytes;
                collect::write_artifacts(
                    &self.spec.dir.join("outputs"),
                    &task.sandbox.outputs,
                    &self.spec.dir.join("artifacts"),
                );
            }
            Err(e) => {
                // A partial copy is not kept (review F2-RA-1).
                let _ = std::fs::remove_dir_all(&outputs);
                self.rt
                    .events
                    .log(self.name(), format!("outputs were not unpacked: {e}"));
            }
        }
        Ok(archive)
    }

    /// Record what [`crate::custody::filter_outputs`] refused, after what
    /// an earlier pass refused. With `above` false, folders at or above an
    /// output are left out: an archive packed from a regraded trial's
    /// `outputs/` holds them, and dropping them says nothing about the
    /// agent.
    fn note_refused(&mut self, f: crate::custody::Filtered, above: bool) {
        let (count, refused) = if above {
            (f.refused_count, f.refused)
        } else {
            let rest = f.refused.into_iter();
            (
                f.refused_count - f.folders_above,
                rest.filter(|r| r.reason != crate::custody::ABOVE_OUTPUTS)
                    .collect(),
            )
        };
        if count == 0 {
            return;
        }
        let c = &mut self.custody;
        c.outputs_refused_count += count;
        let room = crate::custody::LISTED_REFUSALS.saturating_sub(c.outputs_refused.len());
        c.outputs_refused.extend(refused.into_iter().take(room));
        self.rt.events.log(
            self.name(),
            format!(
                "{count} entries of the outputs archive were not saved or restored: links, special files, or paths outside the task's outputs (custody.json lists them)"
            ),
        );
    }

    /// The byte budget for what the outputs, a stage's outputs and the
    /// session unpack to on this machine.
    pub(crate) fn unpack_budget(&self) -> u64 {
        let full = crate::custody::OUTPUT_BUDGET;
        self.rt.options.unpack_budget.map_or(full, |b| b.min(full))
    }

    async fn save_outputs(&mut self) {
        if self.sandbox.is_none() {
            return;
        }
        match self.collect_outputs().await {
            Ok(archive) => crate::copy_out::remove(&archive),
            Err(p) => self.rt.events.log(
                self.name(),
                format!("outputs were not saved: {}", p.message),
            ),
        }
    }
}

/// [`crate::custody::filter_outputs`] from the archive at `raw` into a new
/// file at `archive`.
fn filter_into_file(
    raw: &Path,
    archive: &Path,
    allowed: &[String],
    budget: u64,
) -> Result<crate::custody::Filtered, crate::custody::FilterError> {
    let file = std::fs::File::create(archive).map_err(crate::custody::FilterError::Write)?;
    crate::custody::filter_within(raw, std::io::BufWriter::new(file), allowed, budget)
}

/// A filter of the outputs archive that stopped: an archive it could not
/// read, or one past its limits, is what the sandbox's `tar` made
/// (`R103`), and so is a checked copy past the copy-out's cap in memory; a
/// copy it could not write is this machine's (`R107`).
fn filter_problem(e: crate::custody::FilterError, phase: Phase) -> Problem {
    match e {
        crate::custody::FilterError::Write(e) if e.kind() == std::io::ErrorKind::FileTooLarge => {
            Problem::new(
                "R103",
                Owner::Sandbox,
                phase,
                format!("the checked outputs are past what BenchFlow restores: {e}"),
                "retry; if it repeats, report it with the trial folder",
            )
        }
        crate::custody::FilterError::Write(e) => sandbox_problem(
            phase,
            SandboxError::io("writing the checked copy of the outputs", e),
        ),
        crate::custody::FilterError::Archive(e) => Problem::new(
            "R103",
            Owner::Sandbox,
            phase,
            if e.kind() == std::io::ErrorKind::FileTooLarge {
                format!(
                    "the outputs archive the sandbox's `tar` made is past what BenchFlow reads: {e}"
                )
            } else {
                format!("the outputs archive the sandbox's `tar` made could not be read: {e}")
            },
            "retry; if it repeats, report it with the trial folder",
        ),
    }
}

/// The `agent_user` record: enforced unless the image gives the agent user
/// a way to root (`ways`), in which case it is failed, with the ways, and
/// the trial still runs. Checks the image had no tool for are noted.
fn root_path_verdict(who: String, ways: &[&str], unchecked: &[&str]) -> (DefenseState, String) {
    let mut note = who;
    let state = if ways.is_empty() {
        DefenseState::Enforced
    } else {
        note.push_str(&format!(
            ", but it can become root: {}; the trial ran anyway, since a task may need sudo, so every defense that relies on the agent not being root (the firewall, the daemon guard, the locks, the kill before grading) holds only against an agent that does not use it; a separate verifier still gets only regular files and folders inside the task's outputs (anything else in the outputs archive is dropped and listed in custody.json), but test-runner files the agent puts inside the outputs (a conftest.py, a .pth file) reach it",
            ways.join("; ")
        ));
        DefenseState::Failed
    };
    if !unchecked.is_empty() {
        note.push_str(&format!("; not checked: {}", unchecked.join(", ")));
    }
    (state, note)
}

/// Whether `workdir` is root's home folder, which the agent user then owns.
/// (`/`, above it, is refused: it holds BenchFlow's state.)
fn is_root_home(workdir: &str) -> bool {
    workdir.trim_end_matches('/') == "/root"
}

/// The workdir a `pwd` in a fresh sandbox reports: the image's WORKDIR, or
/// `/root` when that is `/` or unreadable, as BenchFlow 0.8 resolves it.
fn image_workdir(pwd: &str) -> String {
    match pwd.lines().map(str::trim).rfind(|l| !l.is_empty()) {
        Some(w) if w.starts_with('/') && w != "/" => w.to_string(),
        _ => "/root".to_string(),
    }
}

/// The outputs a task saves: the declared ones, or the workdir.
pub(crate) fn output_paths(task: &TaskPackage, workdir: &str) -> Vec<(String, String)> {
    if task.sandbox.outputs.is_empty() {
        let w = workdir.to_string();
        let name = Path::new(&w)
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "workdir".into());
        vec![(w, name)]
    } else {
        task.sandbox
            .outputs
            .iter()
            .map(|o| (o.path.clone(), o.save_as.clone()))
            .collect()
    }
}

/// Where 0.8 writes the oracle's output in the sandbox, for verifiers that
/// grade what it printed.
const ORACLE_OUTPUT: &str = "/logs/agent/oracle.txt";

/// The oracle's command: `script` with stdout and stderr in one file at
/// [`ORACLE_OUTPUT`], as 0.8 runs it, then that file printed (for the trial
/// folder's copy) and the script's exit status kept.
fn oracle_command(script: &str) -> String {
    format!(
        "mkdir -p /logs/agent 2>/dev/null; {{ {}; }} > {ORACLE_OUTPUT} 2>&1; bf_rc=$?; cat {ORACLE_OUTPUT}; exit $bf_rc",
        // bash where the image has it, else sh (images without bash).
        shell::script_file(script)
    )
}
/// Where verifiers from older benchmarks expect their files.
const LEGACY_VERIFIER_MOUNT: &str = "/tests";

/// Point `/tests` at the verifier's mount when nothing is there.
fn legacy_tests_link(mount: &str) -> String {
    format!(
        "[ -e {l} ] || [ -L {l} ] || ln -s {m} {l}",
        l = LEGACY_VERIFIER_MOUNT,
        m = shell::quote(mount.trim_end_matches('/'))
    )
}

/// A path from a task reader, as a path bench can open. The task.md reader
/// gives folders under the task's root (`<root>/oracle`) but controls
/// relative to it (`controls/near-miss.sh`); the split and 0.8 readers give
/// everything relative to it (`solution`, `tests`, `environment/Dockerfile`).
/// A relative path must never be read from the folder bench runs in.
pub fn in_package(root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() || path.starts_with(root) {
        path.to_path_buf()
    } else {
        root.join(path)
    }
}

/// What a panic said.
fn panic_text(panic: &(dyn std::any::Any + Send)) -> String {
    panic
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_else(|| "a panic without a message".to_string())
}

/// A hardening program that exited 0 but printed none of its `BF` lines:
/// something on the image's `PATH` answered in place of the shell or a tool
/// it uses, so no defense it should have applied can be trusted.
/// What `hardening::own_workdir` reported: the new owner and the paths
/// in the workdir that kept theirs, or why the workdir could not be given
/// to it. Only the workdir folder itself must change: a path below it that
/// could not (one the agent nested past what the image's `find` can walk,
/// a file `chown` refused) is noted, as is a walk that ran past its time after the folder
/// changed (an agent that made millions of files). Neither stops the trial:
/// the agent's processes are gone, and a retry would only let an agent that
/// arranged it dodge its score. A run that ended without its report and
/// without an error is a program answering in the shell's place.
pub(crate) fn own_result(
    out: &ExecOutput,
    owner: &str,
    what: &str,
    phase: Phase,
) -> Result<(String, Vec<String>), Problem> {
    let report = Report::parse(&out.stdout);
    let mut kept: Vec<String> = report
        .all("unchanged")
        .into_iter()
        .map(str::to_string)
        .collect();
    let owned = report.first("owned").map(str::to_string);
    let rooted = out.stdout.lines().any(|l| l.trim() == "BF root-owned");
    let owner_of = |out: &ExecOutput| {
        out.stderr
            .lines()
            .find_map(|l| l.trim().strip_prefix("BF error "))
            .map(str::to_string)
    };
    match owned {
        Some(o) if out.exit_code == 0 && !out.timed_out => Ok((o, kept)),
        None if out.exit_code == 0 && !out.timed_out => Err(unreported_in(phase, what, out)),
        // Past its time, or ended by the sandbox (a kill), once the folder
        // itself changed: noted, as for paths that kept their owner.
        _ if rooted && (out.timed_out || !matches!(out.exit_code, 0 | 2 | 3)) => {
            kept.push(if out.timed_out {
                format!(
                    "{what} ran past its {} s timeout after the workdir itself changed; paths it had not reached kept their owner",
                    OWN_TIMEOUT.as_secs()
                )
            } else {
                format!(
                    "{what} ended with exit {} after the workdir itself changed; paths it had not reached kept their owner",
                    out.exit_code
                )
            });
            Ok((owner.to_string(), kept))
        }
        _ if out.timed_out || !matches!(out.exit_code, 2 | 3) => Err(Problem::new(
            "R103",
            Owner::Sandbox,
            phase,
            if out.timed_out {
                format!(
                    "{what} ran past its {} s timeout before the workdir itself changed",
                    OWN_TIMEOUT.as_secs()
                )
            } else {
                format!("{what} ended with exit {} before it reported", out.exit_code)
            },
            "retry; if it keeps failing, run `bench doctor`",
        )
        .detail(out.tail(1500))
        .transient()),
        _ => Err(Problem::new(
            "R315",
            Owner::Task,
            phase,
            match owner_of(out) {
                Some(e) if e.starts_with("no-user ") => format!(
                    "{what} failed: the image has no user `{}`",
                    e.trim_start_matches("no-user ")
                ),
                Some(e) if e.starts_with("workdir ") => {
                    format!("{what} failed: the workdir is the root folder `/`")
                }
                Some(e) if e == "no-find" || e == "find" => format!(
                    "{what} failed: the image has no `find` that takes -xdev, -mindepth, -maxdepth, and -links"
                ),
                _ => format!(
                    "{what} failed: the workdir folder itself could not change owner (exit {})",
                    out.exit_code
                ),
            },
            "add the user to the image, or fix [verifier] user; keep the workdir off a read-only mount, in its own folder",
        )
        .detail(out.tail(1500))),
    }
}

pub(crate) fn unreported(what: &str, out: &ExecOutput) -> Problem {
    unreported_in(Phase::Handoff, what, out)
}

/// [`unreported`] for a check in another phase.
pub(crate) fn unreported_in(phase: Phase, what: &str, out: &ExecOutput) -> Problem {
    Problem::new(
        "R307",
        Owner::Task,
        phase,
        format!(
            "{what} ran but did not report back, so the trial was not graded: a program on the image's PATH (sh, bash, or a tool BenchFlow's checks use) answered in its place"
        ),
        "make the folders on the image's PATH root-owned and not writable by others (no `chmod 777 /usr/local/bin`); `bench explain R307` has more",
    )
    .detail(out.tail(1500))
}

/// A sandbox operation's failure as a trial problem. A failure on this
/// machine (a file BenchFlow reads or writes, the container CLI it starts)
/// is the user's to fix, not the sandbox's; a backend that cannot do what
/// the task asks is a choice of backend.
pub(crate) fn sandbox_problem(phase: Phase, e: SandboxError) -> Problem {
    let transient = e.is_transient();
    let p = match e {
        // The backend's own answer (Daytona's API refusing the key or the
        // account; the container CLI, asked itself, unable to reach its
        // daemon): nothing code inside the sandbox can cause. An outage seen
        // as 5xx or 429 is a `Transport` error instead, which looks the same
        // as code inside breaking the connection, so it counts toward the
        // repeat cap (`--rerun errored` runs such a trial again).
        SandboxError::Unavailable { message, fix } => {
            let mut p = Problem::new("R103", Owner::Sandbox, phase, message, fix);
            p.loss_confirmed = Some(true);
            p
        }
        // The backend's rate limit, in its own answer: confirmed too.
        SandboxError::Throttled(message) => {
            let mut p = Problem::new(
                "R103",
                Owner::Sandbox,
                phase,
                format!("the sandbox failed while {}: {message}", phase.label()),
                "retry; with lower `--concurrency` if it keeps happening",
            );
            p.loss_confirmed = Some(true);
            p
        }
        SandboxError::Io { context, source } => Problem::new(
            "R107",
            Owner::You,
            phase,
            format!("{context} failed on this machine: {source}"),
            "fix what the message names (a missing or unreadable file, a full disk, a container CLI not on PATH), then run again",
        ),
        SandboxError::Unsupported(message) => Problem::new(
            "R401",
            Owner::You,
            phase,
            message,
            "run it on a backend that supports it (`--sandbox docker`), or remove the setting",
        ),
        other => Problem::new(
            "R103",
            Owner::Sandbox,
            phase,
            format!("the sandbox failed while {}: {other}", phase.label()),
            "retry; if it keeps failing, run `bench doctor`",
        ),
    };
    if transient { p.transient() } else { p }
}

/// The error for a file or folder that could not be put into the sandbox.
/// After an agent's turn in it (`after_turn`), a `tar` that says it is
/// missing is the sandbox's content (R103), not an image without one
/// (R401): the same sandbox ran `tar` at setup, and a root agent can
/// remove or replace it (review F2-A-r12-1, the sibling). Before the first
/// turn, an image without `tar` is the user's to fix, as before.
pub(crate) fn put_problem(phase: Phase, e: SandboxError, after_turn: bool) -> Problem {
    match e {
        SandboxError::Unsupported(m) if after_turn => sandbox_problem(
            phase,
            SandboxError::Content(format!(
                "{m} (it worked before the agent's turn, so something in the sandbox removed or replaced it)"
            )),
        ),
        e => sandbox_problem(phase, e),
    }
}

/// The error for an outputs archive that could not be copied out. Any
/// failure that is not this machine's own ([`host_side_io`]: its disk,
/// quota or permissions, or its folder gone) is R103 (the sandbox's, at
/// saving), as an archive that could not be packed is: past its cap, not
/// arrived as a file, gone just after it was packed, empty or broken, or a
/// `tar` that says it is missing (the pack just ran `tar` in that sandbox,
/// so its absence is the sandbox's content). Code inside can cause each of these, so none may be
/// the user's (R107, R401), which resumes rerun without a cap (reviews
/// F2-A-r10 M1, F2-A-r11 M1/M2, F2-B-r11 M1). One that never finished is
/// the sandbox's lost connection (run again once); the backend's own
/// answers (an outage, a rate limit) are as [`sandbox_problem`] says.
pub(crate) fn outputs_copy_problem(e: crate::copy_out::CopyOutError, what: &str) -> Problem {
    use crate::copy_out::CopyOutError;
    let what = format!("{what} could not be copied out of the sandbox");
    let transient = match &e {
        CopyOutError::TimedOut(_) => return lost(Phase::Save, &what, e.into_sandbox()),
        CopyOutError::Failed(se)
            if host_side_io(se)
                || matches!(
                    se,
                    SandboxError::Unavailable { .. } | SandboxError::Throttled(_)
                ) =>
        {
            return sandbox_problem(Phase::Save, e.into_sandbox());
        }
        CopyOutError::Failed(se) => se.is_transient(),
        CopyOutError::NotAFile => false,
    };
    let p = Problem::new(
        "R103",
        Owner::Sandbox,
        Phase::Save,
        format!("{what}: {e}"),
        "retry; if it repeats, report it with the trial folder (a failure to start the copy on this machine, such as too many open files, shows here too)",
    );
    if transient { p.transient() } else { p }
}

/// A copy's failure on this machine's side: its disk, its quota, its
/// permissions, or a folder of its own gone while the copy was unpacked.
/// Not the remote path missing, nor an archive that streamed nothing
/// under it: both are what came out of the sandbox (review F2-A-r11 L2).
/// The archive's own modes cannot cause a `PermissionDenied`: the copy
/// makes an archive readable and the unpacker replaces a file it meets
/// again (review F2-A-r11 M1, M2).
pub(crate) fn host_side_io(e: &SandboxError) -> bool {
    let SandboxError::Io { source, .. } = e else {
        return false;
    };
    match source.kind() {
        std::io::ErrorKind::StorageFull
        | std::io::ErrorKind::QuotaExceeded
        | std::io::ErrorKind::ReadOnlyFilesystem
        | std::io::ErrorKind::PermissionDenied => true,
        std::io::ErrorKind::NotFound => {
            !remote_missing(e) && !benchflow_sandbox::archive::is_empty_archive(source)
        }
        _ => false,
    }
}

/// When at least a quarter of an agent's model requests (and three or
/// more) were refused with 429 or a 5xx, and the provider was still
/// holding the agent back, the provider's problem: the agent that timed
/// out spent its time waiting, so its score would measure the provider.
/// Holding back means the refusals and the waits after them (a client's
/// backoff) took a quarter of `elapsed`, the agent's time (else of the time
/// the requests span), as [`held_back`] counts them. A few refusals early,
/// then a long stretch of the agent's own work, do not excuse the timeout,
/// nor do refused side requests the agent went on without. The requests
/// are a native agent's `model_calls.jsonl` (`started_at`, `duration_ms`,
/// `attempt`), or the provider forwarder's log for an agent behind it
/// (`time` at the end, `ms`, `body_hash`; its model calls are the `POST`s
/// it passed on, not its own refusals); a log without times is judged by
/// its last request alone. Only durations and gaps within one log are
/// used, so a sandbox clock that differs from this machine's does not
/// matter.
pub fn throttled(agent_dir: &Path, elapsed: Option<Duration>) -> Option<Problem> {
    let (mut total, mut refused) = (0u64, 0u64);
    let mut statuses = BTreeMap::new();
    let mut holding = false;
    for name in ["model_calls.jsonl", "forwarder.log"] {
        let text = std::fs::read_to_string(agent_dir.join(name)).unwrap_or_default();
        // In the log's order.
        let mut calls: Vec<Call> = Vec::new();
        for line in text.lines() {
            let Ok(v) = serde_json::from_str::<Value>(line) else {
                continue;
            };
            let Some(s) = v.get("status").and_then(Value::as_u64) else {
                continue;
            };
            let method = v.get("method").and_then(Value::as_str);
            if method.is_some_and(|m| m != "POST") || v.get("note").is_some() {
                continue;
            }
            total += 1;
            let refusal = s == 429 || (500..600).contains(&s);
            if refusal {
                refused += 1;
                *statuses.entry(s).or_insert(0u64) += 1;
            }
            let sent = v
                .get("body_hash")
                .and_then(Value::as_str)
                .map(|h| Sent::Body(h.to_string()))
                .or_else(|| {
                    let attempt = v.get("attempt").and_then(Value::as_u64)?;
                    let seq = v.get("seq").and_then(Value::as_u64);
                    Some(Sent::Attempt { seq, attempt })
                });
            calls.push(Call {
                span: call_span(&v),
                refused: refusal,
                sent,
            });
        }
        holding |= held_back(&calls, elapsed);
    }
    if refused < 3 || refused * 4 < total || !holding {
        return None;
    }
    let seen: Vec<String> = statuses
        .iter()
        .map(|(s, n)| format!("{n} × HTTP {s}"))
        .collect();
    Some(
        Problem::new(
            "R214",
            Owner::Model,
            Phase::Agent,
            format!(
                "the agent ran out of time while the model provider refused {refused} of its {total} requests ({})",
                seen.join(", ")
            ),
            "lower --concurrency or use a provider plan with a higher rate limit, then rerun the same command to resume the job; agent/model_calls.jsonl or agent/forwarder.log in the trial folder has each answer",
        )
        .transient(),
    )
}

/// The model provider rejected every request the agent sent (a 4xx other
/// than 429, which is throttling) and answered none: the agent never got a
/// reply to work from, so the trial is the provider's (R214), never a
/// scored 0. Read from the same logs as [`throttled`]; the forwarder's own
/// refusals (lines with a `note`, such as a model the run may not use) are
/// not the provider's answers and are left out. A rejected request is not
/// retried (decision 17).
pub fn provider_rejected_all(agent_dir: &Path) -> Option<Problem> {
    let mut statuses: BTreeMap<u64, u64> = BTreeMap::new();
    let (mut answered, mut rejected) = (0u64, 0u64);
    for name in ["model_calls.jsonl", "forwarder.log"] {
        let text = std::fs::read_to_string(agent_dir.join(name)).unwrap_or_default();
        for line in text.lines() {
            let Ok(v) = serde_json::from_str::<Value>(line) else {
                continue;
            };
            let Some(s) = v.get("status").and_then(Value::as_u64) else {
                continue;
            };
            let method = v.get("method").and_then(Value::as_str);
            if method.is_some_and(|m| m != "POST") || v.get("note").is_some() {
                continue;
            }
            if (200..300).contains(&s) {
                answered += 1;
            } else if (400..500).contains(&s) && s != 429 {
                rejected += 1;
                *statuses.entry(s).or_insert(0) += 1;
            }
        }
    }
    if answered > 0 || rejected == 0 {
        return None;
    }
    let seen: Vec<String> = statuses
        .iter()
        .map(|(s, n)| format!("{n} × HTTP {s}"))
        .collect();
    let what = format!(
        "the model provider rejected every request the agent sent ({}) and answered none, so the agent never worked on the task",
        seen.join(", ")
    );
    let only = |codes: &[u64]| statuses.keys().all(|s| codes.contains(s));
    // As the harnesses file them: a key the provider refused (401, 403) is
    // R212, a spent plan (402) the usage limit R213, which stops the job
    // (review E7-C-A 1), and a model it does not serve (404) R215; anything
    // else it rejected is R214 (review E7-B).
    Some(if only(&[402]) {
        Problem::new(
            "R213",
            Owner::Model,
            Phase::Agent,
            what,
            "the provider says the plan's credit or quota is spent (HTTP 402); add credit or wait for the quota to reset, then rerun the same command to resume the job",
        )
    } else if only(&[401, 403]) {
        Problem::new(
            "R212",
            Owner::Model,
            Phase::Agent,
            what,
            "the model provider rejected the login; check the key, then rerun the same command",
        )
    } else if only(&[404]) {
        Problem::new(
            "R215",
            Owner::You,
            Phase::Agent,
            what,
            "check the model's name with the provider (`bench agents <name>` lists the providers an agent takes), then rerun",
        )
    } else {
        Problem::new(
            "R214",
            Owner::Model,
            Phase::Agent,
            what,
            "check that the model takes what this agent sends (an agent option may turn a feature off, such as a tool the provider lacks), or pick another model; agent/forwarder.log or agent/model_calls.jsonl in the trial folder has each answer",
        )
    })
}

/// A request's start and end in milliseconds: a native call's `started_at`
/// and `duration_ms`, or a forwarder line's `time` (its end) and `ms`.
fn call_span(v: &Value) -> Option<(i64, i64)> {
    let ms = |k: &str| v.get(k).and_then(Value::as_i64).filter(|d| *d >= 0);
    let at = |k: &str| {
        v.get(k)
            .and_then(Value::as_str)
            .and_then(|t| chrono::DateTime::parse_from_rfc3339(t).ok())
            .map(|t| t.timestamp_millis())
    };
    if let (Some(start), Some(d)) = (at("started_at"), ms("duration_ms")) {
        return Some((start, start.saturating_add(d)));
    }
    if let (Some(end), Some(d)) = (at("time"), ms("ms")) {
        return Some((end.saturating_sub(d), end));
    }
    None
}

/// The model provider's problem in place of an agent's `code` when the
/// agent's logs show the provider caused it: an agent that did nothing or
/// went silent (R206, R203) while a call [`never_answered`] or the provider
/// held it back ([`throttled`]); an agent that failed its turn or exited
/// (R205, R204) right after an answer it could not read
/// ([`unreadable_answer`]).
pub fn provider_instead(code: &str, agent_dir: &Path, elapsed: Duration) -> Option<Problem> {
    match code {
        "R203" | "R206" => {
            never_answered(agent_dir).or_else(|| throttled(agent_dir, Some(elapsed)))
        }
        "R204" | "R205" => unreadable_answer(agent_dir),
        _ => None,
    }
}

/// The forwarder's last answered model call, when the provider's answer
/// could not be read (`unreadable` in `agent/forwarder.log`: JSON that does
/// not parse, or a stream with garbled events).
pub fn unreadable_answer(agent_dir: &Path) -> Option<Problem> {
    let text = std::fs::read_to_string(agent_dir.join("forwarder.log")).ok()?;
    let last = text
        .lines()
        .rev()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .find(|v| {
            v.get("method").and_then(Value::as_str) == Some("POST")
                && v.get("note").is_none()
                && v.get("status").is_some()
        })?;
    if last.get("unreadable").and_then(Value::as_bool) != Some(true) {
        return None;
    }
    let path = last.get("path").and_then(Value::as_str).unwrap_or_default();
    Some(
        Problem::new(
            "R214",
            Owner::Model,
            Phase::Agent,
            format!(
                "the model provider's last answer (to {path}) could not be read: its body or stream was garbled, and the agent stopped on it"
            ),
            "check the provider's status, then rerun the same command to resume the job; agent/forwarder.log in the trial folder marks the answer `unreadable`",
        )
        .transient(),
    )
}

/// The provider never answered the call the agent was waiting on, from
/// `agent/forwarder.log`: the agent's latest call with tools (its own turn,
/// not a title or a summary; the latest call of any kind in a log that does
/// not mark tools) never got its answer started, neither an `answering`
/// note nor a line with a status for it. A `dropped` note does not answer
/// it: BenchFlow's own kill of a stalled agent closes the call too (E5
/// review C-A 1), and a call dropped earlier is not the latest. Earlier
/// calls, answered or not, say nothing about the stall (E5 review A1: a
/// call dropped long before blamed a later hang on the provider). A
/// `retrying` note starts the call again. Lines are matched to calls by
/// body hash only (by path for a call without a body), never one call's
/// line to another's (E5 review C-A 2). Only the log's own order is used,
/// so the sandbox's clock does not matter. `None` when the provider is not
/// shown at fault.
pub fn never_answered(agent_dir: &Path) -> Option<Problem> {
    struct Call {
        key: String,
        path: String,
        time: String,
        tools: bool,
        /// An answer (a status) or a `retrying` note ended it.
        open: bool,
        answering: bool,
        /// The agent hung up on it (or BenchFlow's kill did): it takes no
        /// later line, so a resend of the same body gets its own answer
        /// (E5 review D-A), but unanswered it still counts as unanswered.
        dropped: bool,
    }
    let text = std::fs::read_to_string(agent_dir.join("forwarder.log")).ok()?;
    let mut calls: Vec<Call> = Vec::new();
    for line in text.lines() {
        let Ok(v) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        if v.get("method").and_then(Value::as_str) != Some("POST") {
            continue;
        }
        let path = v
            .get("path")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let key = v
            .get("body_hash")
            .and_then(Value::as_str)
            .map_or_else(|| path.clone(), str::to_string);
        let note = v.get("note").and_then(Value::as_str).unwrap_or_default();
        // The earliest open call this line is about: the same body.
        let find = |calls: &[Call]| {
            calls
                .iter()
                .position(|c| c.open && !c.dropped && c.key == key)
        };
        if note == "waiting" {
            calls.push(Call {
                key: key.clone(),
                path,
                time: v
                    .get("time")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                tools: v.get("tools").and_then(Value::as_bool) == Some(true),
                open: true,
                answering: false,
                dropped: false,
            });
        } else if note == "answering" {
            if let Some(i) = find(&calls) {
                calls[i].answering = true;
            }
        } else if note == "dropped" {
            if let Some(i) = find(&calls) {
                calls[i].dropped = true;
            }
        } else if (note.starts_with("retrying") || v.get("status").is_some())
            && let Some(i) = find(&calls)
        {
            calls[i].open = false;
        }
    }
    let marked = calls.iter().any(|c| c.tools);
    let last = calls.iter().rev().find(|c| c.tools || !marked)?;
    if !last.open || last.answering {
        return None;
    }
    Some(
        Problem::new(
            "R214",
            Owner::Model,
            Phase::Agent,
            format!(
                "the model provider never answered: the agent's last model call (to {}, sent at {} by the sandbox's clock) had no answer, not even its start, when the agent stopped",
                last.path, last.time
            ),
            "check the provider's status, then rerun the same command to resume the job; agent/forwarder.log in the trial folder has each call",
        )
        .transient(),
    )
}

/// One model request in an agent's log, as [`throttled`] reads it.
struct Call {
    /// Its start and end in milliseconds, when the line says.
    span: Option<(i64, i64)>,
    /// Refused with 429 or a 5xx.
    refused: bool,
    /// What tells this request sent again from a new one, when the line
    /// records it.
    sent: Option<Sent>,
}

/// What says a later request is an earlier one sent again: the same body
/// (the forwarder's `body_hash`), or a later `attempt` of the native
/// client's same call (its `seq`).
enum Sent {
    Body(String),
    Attempt { seq: Option<u64>, attempt: u64 },
}

impl Sent {
    /// Whether `later` is this request sent again. A native try the log
    /// skips (one with no status: a reset or a timeout) leaves a gap in the
    /// call's attempts; a log without `seq` can only take the next attempt.
    fn again(&self, later: &Sent) -> bool {
        match (self, later) {
            (Sent::Body(a), Sent::Body(b)) => a == b,
            (
                Sent::Attempt {
                    seq: Some(x),
                    attempt: a,
                },
                Sent::Attempt {
                    seq: Some(y),
                    attempt: b,
                },
            ) => x == y && b > a,
            (
                Sent::Attempt {
                    seq: None,
                    attempt: a,
                },
                Sent::Attempt {
                    seq: None,
                    attempt: b,
                },
            ) => a.checked_add(1) == Some(*b),
            _ => false,
        }
    }
}

/// The longest a refusal counts as waiting after its own time when no later
/// request shows it sent again: a client's backoff between tries is seconds
/// to a minute, while what follows a request the agent did not send again
/// may be its own work (a tool run) as much as a wait.
const UNSEEN_WAIT_MS: i64 = 60_000;

/// Whether one log's requests show the provider holding the agent back, as
/// [`throttled`] says. The waiting is each refusal's stretch from its start
/// until the agent sent the same request again (its own time and the
/// backoff after it). A refusal the log shows was not sent again, though
/// the agent sent others after it (a title or next-speaker check it went on
/// without), counts only its own time. Where the log cannot tell (a line
/// with no `body_hash` or `attempt`), or nothing was sent after it (the
/// agent may have been backing off when its time ran out), it counts until
/// the next request, or the end of the agent's time, and at most
/// [`UNSEEN_WAIT_MS`] after its end. All such stretches are counted once
/// however many requests ran side by side, less any time a request that was
/// answered was in progress: the agent was not held back while an answer
/// streamed. A log without times is judged by its last request alone.
fn held_back(calls: &[Call], elapsed: Option<Duration>) -> bool {
    let mut timed: Vec<(&Call, (i64, i64))> =
        calls.iter().filter_map(|c| Some((c, c.span?))).collect();
    if timed.len() < calls.len() || timed.is_empty() {
        return calls.last().is_some_and(|c| c.refused);
    }
    timed.sort_by_key(|(_, (start, _))| *start);
    let first = timed[0].1.0;
    let last = timed
        .iter()
        .map(|(_, (_, end))| *end)
        .max()
        .unwrap_or(first);
    let span = elapsed
        .map_or(last.saturating_sub(first), |e| {
            i64::try_from(e.as_millis()).unwrap_or(i64::MAX)
        })
        .max(1);
    let agent_end = elapsed.map_or(last, |_| first.saturating_add(span));
    let mut waits = Vec::new();
    let mut answered = Vec::new();
    for (i, (call, (start, end))) in timed.iter().enumerate() {
        if !call.refused {
            answered.push((*start, *end));
            continue;
        }
        let later = &timed[i + 1..];
        let again = call.sent.as_ref().and_then(|sent| {
            later
                .iter()
                .find(|(c, _)| c.sent.as_ref().is_some_and(|l| sent.again(l)))
        });
        let next = later.iter().map(|(_, (s, _))| *s).find(|s| s >= end);
        let until = match (again, next) {
            (Some((_, (s, _))), _) => *s,
            (None, Some(_)) if call.sent.is_some() => *end,
            (None, next) => next
                .unwrap_or(agent_end)
                .min(end.saturating_add(UNSEEN_WAIT_MS)),
        };
        waits.push((*start, until.max(*end)));
    }
    let waits = merged(waits);
    let answered = merged(answered);
    let waited = covered(&waits).saturating_sub(overlap(&waits, &answered));
    waited.saturating_mul(4) >= span
}

/// Intervals (start, end) merged where they touch or overlap, in order.
fn merged(mut spans: Vec<(i64, i64)>) -> Vec<(i64, i64)> {
    spans.sort_unstable();
    let mut out: Vec<(i64, i64)> = Vec::new();
    for (a, b) in spans {
        match out.last_mut() {
            Some(last) if a <= last.1 => last.1 = last.1.max(b),
            _ => out.push((a, b)),
        }
    }
    out
}

/// The time merged intervals cover.
fn covered(spans: &[(i64, i64)]) -> i64 {
    spans
        .iter()
        .fold(0i64, |sum, (a, b)| sum.saturating_add(b.saturating_sub(*a)))
}

/// The time two lists of merged intervals cover together.
fn overlap(x: &[(i64, i64)], y: &[(i64, i64)]) -> i64 {
    let (mut i, mut j, mut sum) = (0, 0, 0i64);
    while i < x.len() && j < y.len() {
        let (a, b) = (x[i].0.max(y[j].0), x[i].1.min(y[j].1));
        if a < b {
            sum = sum.saturating_add(b.saturating_sub(a));
        }
        if x[i].1 < y[j].1 {
            i += 1;
        } else {
            j += 1;
        }
    }
    sum
}

/// The download failed because the remote path does not exist (both
/// backends say so with `NotFound`), not because the copy broke off.
fn remote_missing(e: &SandboxError) -> bool {
    matches!(e, SandboxError::Io { context, source }
        if source.kind() == std::io::ErrorKind::NotFound && context.starts_with("downloading"))
}

/// The most the verifier's folder may hold when it is copied out: more is
/// the folder's fault (R301), and the cap keeps it from filling this
/// machine's disk.
const VERIFIER_COPY_CAP: u64 = 1 << 30;
/// The pause before the verifier's folder is copied a second time.
const VERIFIER_COPY_PAUSE: Duration = Duration::from_secs(3);

/// The attempt whose earlier losses [`lost_before`] counts: its task
/// (by digest), its agent, and its model when it names one.
struct Same<'s> {
    digest: &'s str,
    agent: &'s str,
    model: Option<&'s str>,
}

/// Whether an earlier attempt of the trial (set aside under
/// `<job>/.rerun/<trial>-<n>/` by a retry or a resume) lost its sandbox in
/// a way the backend did not confirm, in any phase: the one counter of
/// `Trial::lost_again`. Only attempts of the same task, agent and model
/// count, and only those after the newest attempt a plain resume would
/// have kept: a capped one, run again only by `--rerun errored`, and a
/// scored or unscored one, run again only by `--rerun failed` or
/// `--rerun unscored`. An explicit rerun starts the count anew (convergence
/// reviews E9-E-A 2, E9-F-A). A record without the field (an older bench)
/// does not count.
fn lost_before(trial_dir: &Path, same: &Same<'_>) -> bool {
    let (Some(job), Some(name)) = (trial_dir.parent(), trial_dir.file_name()) else {
        return false;
    };
    let store = job.join(".rerun");
    let name = name.to_string_lossy();
    let mut lost = false;
    for dir in (1..)
        .map(|n| store.join(format!("{name}-{n}")))
        .take_while(|dir| dir.is_dir())
    {
        let Some(v) = std::fs::read_to_string(dir.join("result.json"))
            .ok()
            .and_then(|text| serde_json::from_str::<Value>(&text).ok())
        else {
            continue;
        };
        let problem = &v["problem"];
        let kept = problem["rerun_capped"].as_bool() == Some(true)
            || matches!(v["status"].as_str(), Some("scored" | "unscored"));
        if kept {
            lost = false;
            continue;
        }
        let of_this = v["task_digest"].as_str() == Some(same.digest)
            && v["agent"].as_str() == Some(same.agent)
            && same.model.is_none_or(|m| v["model"].as_str() == Some(m));
        if of_this && problem["loss_confirmed"].as_bool() == Some(false) {
            lost = true;
        }
    }
    lost
}

/// Empty `dir` of what a copy left in it.
fn clear_folder(dir: &Path) {
    let _ = std::fs::remove_dir_all(dir);
    let _ = std::fs::create_dir_all(dir);
}

/// A failure that happened because the sandbox stopped answering: the
/// sandbox's, and worth a retry.
fn lost(phase: Phase, what: &str, e: SandboxError) -> Problem {
    let mut p = sandbox_problem(phase, e);
    if p.owner == Owner::Sandbox {
        p.message = format!("{what}: {}", p.message);
        p = p.transient();
    }
    p
}

fn write_trajectory(dir: &Path, events: &[Value]) {
    let _ = std::fs::create_dir_all(dir);
    let text: String = events
        .iter()
        .map(|e| format!("{}\n", serde_json::to_string(e).unwrap_or_default()))
        .collect();
    let _ = std::fs::write(dir.join("acp_trajectory.jsonl"), text);
}

fn scripted_summary(kind: &str) -> Value {
    json!({
        "steps": 1, "tool_call_steps": 0, "user_message_steps": 0, "agent_message_steps": 0,
        "agent_thought_steps": 0, "other_steps": 1,
        "event_type_counts": { kind: 1 }, "tool_call_status_counts": {},
        "partial_trajectory": false, "trajectory_source": "scripted",
    })
}

/// Raise a task's CPUs and memory to an agent's floor. A size the task left
/// to the backend's default counts as below any floor: the default is the
/// backend's to change.
pub fn apply_floor(
    agent: &str,
    cpus: Option<f64>,
    memory_mb: Option<u64>,
    floor: benchflow_agent::manifest::ResourceFloor,
) -> (
    Option<f64>,
    Option<u64>,
    Option<crate::record::ResourcesRaised>,
) {
    use crate::record::{Raised, ResourcesRaised};
    let mut raised = ResourcesRaised {
        agent: agent.to_string(),
        memory_mb: None,
        cpus: None,
    };
    let memory = match floor.memory_mb {
        Some(min) if memory_mb.is_none_or(|m| m < min) => {
            raised.memory_mb = Some(Raised {
                task: memory_mb.map(|m| m as f64),
                sandbox: min as f64,
            });
            Some(min)
        }
        _ => memory_mb,
    };
    let cpus = match floor.cpus.map(f64::from) {
        Some(min) if cpus.is_none_or(|c| c < min) => {
            raised.cpus = Some(Raised {
                task: cpus,
                sandbox: min,
            });
            Some(min)
        }
        _ => cpus,
    };
    let any = raised.memory_mb.is_some() || raised.cpus.is_some();
    (cpus, memory, any.then_some(raised))
}

/// The run log's line for a raised size.
fn raised_line(r: &crate::record::ResourcesRaised) -> String {
    let what = |name: &str, unit: &str, x: &crate::record::Raised| match x.task {
        Some(t) => format!("{name} {t}{unit} to {}{unit}", x.sandbox),
        None => format!("{name} (the backend's default) to {}{unit}", x.sandbox),
    };
    let parts: Vec<String> = [
        r.memory_mb.as_ref().map(|m| what("memory", " MB", m)),
        r.cpus.as_ref().map(|c| what("CPUs", "", c)),
    ]
    .into_iter()
    .flatten()
    .collect();
    format!(
        "{}'s manifest needs more than the task asks for: raised {} (result.json: resources_raised)",
        r.agent,
        parts.join(" and ")
    )
}

/// Usage for scripted attempts: none.
pub fn no_usage() -> Usage {
    Usage {
        source: "unavailable".into(),
        ..Usage::default()
    }
}

#[cfg(test)]
mod tests {
    /// Diff review of rust/e9-decisions (M2): the repeat rule counts the
    /// previous attempt only when its loss was one the backend did not
    /// confirm. A sandbox deleted or stopped from outside, then a
    /// connection blip in the retry, is no sign of code inside; nor is a
    /// record an older bench wrote without the field.
    /// Round-2 review E9-A M3: the backend's own refusal (its API refusing
    /// the key or account, a daemon the CLI itself cannot reach) is a loss
    /// the backend confirmed, which code inside cannot cause. An outage seen
    /// as 5xx or 429 is a transport error, decided later from the backend
    /// (convergence review E9-C-B 3: the old comment and test claimed 503).
    #[test]
    fn an_unavailable_backend_is_a_confirmed_loss() {
        let p = sandbox_problem(
            Phase::Verify,
            SandboxError::Unavailable {
                message: "Daytona refused checking the sandbox (HTTP 403: forbidden)".into(),
                fix: "check the key".into(),
            },
        );
        assert_eq!((p.code.as_str(), p.loss_confirmed), ("R103", Some(true)));
        let p = sandbox_problem(Phase::Verify, SandboxError::Transport("reset".into()));
        assert_eq!(p.loss_confirmed, None, "decided later, from the backend");
        // Daytona's rate limit, in its own answer (convergence review E9-C-A
        // N2): confirmed, and retried like any transport failure.
        let p = sandbox_problem(
            Phase::Verify,
            SandboxError::Throttled("Daytona refused it (HTTP 429)".into()),
        );
        assert_eq!(
            (p.code.as_str(), p.loss_confirmed, p.transient),
            ("R103", Some(true), true)
        );
    }

    #[test]
    fn only_an_unconfirmed_previous_loss_counts_as_a_repeat() {
        let d = tempfile::tempdir().unwrap();
        let trial = d.path().join("job/hello__abc");
        std::fs::create_dir_all(&trial).unwrap();
        let same = Same {
            digest: "sha256:t1",
            agent: "claude",
            model: Some("m1"),
        };
        let write = |n: u32, problem: serde_json::Value, digest: &str, model: &str| {
            let dir = d.path().join(format!("job/.rerun/hello__abc-{n}"));
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(
                dir.join("result.json"),
                serde_json::json!({"task_digest": digest, "agent": "claude", "model": model,
                    "problem": problem})
                .to_string(),
            )
            .unwrap();
        };
        let loss = |confirmed: Option<bool>, phase: &str| {
            let mut problem = serde_json::json!({"code": "R103", "owner": "sandbox", "phase": phase,
                "message": "lost", "next": "retry", "transient": true});
            if let Some(c) = confirmed {
                problem["loss_confirmed"] = serde_json::json!(c);
            }
            problem
        };
        // One counter per trial (convergence reviews E9-D-A 1, E9-D-B 1): an
        // unconfirmed loss in any phase of any earlier attempt counts.
        write(1, loss(Some(false), "verify"), "sha256:t1", "m1");
        assert!(lost_before(&trial, &same));
        write(1, loss(Some(false), "setup"), "sha256:t1", "m1");
        assert!(lost_before(&trial, &same));
        write(1, loss(Some(true), "verify"), "sha256:t1", "m1");
        assert!(!lost_before(&trial, &same));
        write(1, loss(None, "verify"), "sha256:t1", "m1");
        assert!(!lost_before(&trial, &same));
        // Convergence review E9-E-A 2: only attempts of the same task and
        // model count...
        write(1, loss(Some(false), "agent"), "sha256:t0", "m1");
        assert!(!lost_before(&trial, &same), "another task digest");
        write(1, loss(Some(false), "agent"), "sha256:t1", "m0");
        assert!(!lost_before(&trial, &same), "another model");
        // ...and a later attempt that ended otherwise does not hide one...
        write(1, loss(Some(false), "agent"), "sha256:t1", "m1");
        write(2, serde_json::json!(null), "sha256:t1", "m1");
        assert!(lost_before(&trial, &same));
        // ...but a capped attempt, run again only by `--rerun errored`,
        // starts the count anew.
        let mut capped = loss(Some(false), "verify");
        capped["rerun_capped"] = serde_json::json!(true);
        write(2, capped, "sha256:t1", "m1");
        assert!(!lost_before(&trial, &same));
        write(3, loss(Some(false), "verify"), "sha256:t1", "m1");
        assert!(lost_before(&trial, &same));
        // Convergence review E9-F-A: an attempt that got past the loss and
        // finished, scored or unscored, is run again only by `--rerun
        // failed` or `--rerun unscored`, which start the count anew too.
        for status in ["scored", "unscored"] {
            let dir = d.path().join("job/.rerun/hello__abc-4");
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(
                dir.join("result.json"),
                serde_json::json!({"task_digest": "sha256:t1", "agent": "claude", "model": "m1",
                    "status": status})
                .to_string(),
            )
            .unwrap();
            assert!(!lost_before(&trial, &same), "{status}");
        }
    }

    use super::*;

    /// The oracle's command keeps stdout and stderr in the order printed in
    /// its file (a warning on stderr before the answer on stdout leaves the
    /// answer last, as `tail -n1` graders expect), prints that file, and
    /// exits with the script's status (review E4-B, 2). Run in `sh` with the
    /// file's folder moved to a temp folder.
    #[test]
    fn the_oracle_command_keeps_the_printed_order_and_the_status() {
        let d = tempfile::tempdir().unwrap();
        let script = d.path().join("solve.sh");
        std::fs::write(&script, "echo warning >&2\necho 'Answer: D'\nexit 3\n").unwrap();
        let logs = d.path().join("logs/agent");
        let cmd = oracle_command(&script.display().to_string())
            .replace("/logs/agent", &logs.display().to_string());
        let out = std::process::Command::new("sh")
            .arg("-c")
            .arg(&cmd)
            .output()
            .unwrap();
        assert_eq!(out.status.code(), Some(3));
        let file = std::fs::read_to_string(logs.join("oracle.txt")).unwrap();
        assert_eq!(file, "warning\nAnswer: D\n");
        assert_eq!(String::from_utf8_lossy(&out.stdout), file);
    }

    /// Behavior review 7: a stop that lands after the agent failed, or after
    /// it timed out on a task whose timeout fails it, records that failure,
    /// not R001 (which the pass rate leaves out).
    #[test]
    fn a_failure_known_before_the_stop_is_recorded_instead() {
        let stopped = || {
            Err(Problem::new(
                "R001",
                Owner::You,
                Phase::Agent,
                "stopped before it finished",
                "rerun",
            ))
        };
        let quiet = AgentOutput::default();
        let r = known_before_stop(stopped(), &quiet, OnTimeout::Fail, Phase::Verify);
        let p = r.unwrap_err();
        assert_eq!((p.code.as_str(), p.phase), ("R001", Phase::Verify));
        let failed = AgentOutput {
            problem: Some(Problem::new(
                "R204",
                Owner::Agent,
                Phase::Agent,
                "exited",
                "",
            )),
            ..AgentOutput::default()
        };
        let r = known_before_stop(stopped(), &failed, OnTimeout::Fail, Phase::Agent);
        assert_eq!(r.unwrap_err().code, "R204");
        let timed_out = AgentOutput {
            timed_out: true,
            ..AgentOutput::default()
        };
        let r = known_before_stop(stopped(), &timed_out, OnTimeout::Fail, Phase::Verify);
        let score = r.unwrap();
        assert_eq!((score.reward, score.passed), (0.0, false));
        // When the task grades a timed-out agent's work, nothing is known yet.
        let r = known_before_stop(stopped(), &timed_out, OnTimeout::Grade, Phase::Verify);
        assert_eq!(r.unwrap_err().code, "R001");
    }

    /// A record the disk would not take (`No space left on device` on
    /// 2026-10-10) was a verbose-only log line while the run counted the
    /// trial as passed and exited 0. Then (d2b) a later file's failure
    /// deleted a complete result.json. Now result.json is written last, the
    /// trial is errored without its reward, and its folder is kept aside,
    /// whole, so a rerun starts clean. Here `custody.json` cannot be written
    /// (a folder sits in its place).
    #[test]
    fn an_unsaved_record_is_errored_and_its_folder_kept_aside() {
        let job = tempfile::tempdir().unwrap();
        let dir = job.path().join("t__1");
        std::fs::create_dir_all(dir.join("custody.json")).unwrap();
        std::fs::write(dir.join("custody.json/x"), "").unwrap();
        std::fs::create_dir_all(dir.join("verifier")).unwrap();
        std::fs::write(dir.join("verifier/reward.txt"), "1").unwrap();
        let mut rec = crate::read::record_from_json(
            &json!({"status": "scored", "rewards": {"reward": 1.0}, "passed": true}),
            &dir,
        )
        .unwrap();
        let logged = std::sync::Mutex::new(Vec::new());
        let saved = save_record(&dir, &mut rec, &json!({}), &|l| {
            logged.lock().unwrap().push(l)
        });
        assert!(!saved);
        assert!(
            !dir.exists(),
            "the folder is moved aside, so a rerun starts clean"
        );
        let kept = job.path().join(".unsaved/t__1-1");
        assert!(kept.join("config.json").is_file(), "nothing is deleted");
        assert!(kept.join("verifier/reward.txt").is_file());
        assert!(
            !kept.join("result.json").exists(),
            "result.json is written last"
        );
        assert!(crate::read::read_trial(&kept).is_none());
        assert_eq!(rec.status, Status::Errored);
        assert_eq!(
            (rec.passed, rec.reward, rec.rewards.is_none()),
            (None, None, true)
        );
        let p = rec.problem.as_ref().unwrap();
        assert_eq!(p.code, "R107");
        assert!(!p.transient, "a retry would hit the same disk");
        let why = rec.reason().unwrap();
        assert!(
            why.starts_with(
                "the trial scored 1, but its record could not be written (custody.json: "
            ),
            "{why}"
        );
        assert!(why.contains(".unsaved/t__1-1"), "{why}");
        assert!(why.ends_with("rerunning the job runs it again"), "{why}");
        assert_eq!(logged.lock().unwrap().len(), 1);
    }

    /// The workdir's change of owner: notes for paths that kept theirs and
    /// for a walk past its time once the folder itself changed; R103 (the
    /// sandbox's, retried) for a timeout before; R315 (the task's) for a
    /// user the image lacks or a folder that cannot change; R307 for a run
    /// that said nothing. Guards the D7b fixes of FIX-REVIEW-A 1 and 7 and
    /// FIX-REVIEW-B 3 on b7d315aa, which failed the trial on any of these
    /// and blamed the sandbox.
    #[test]
    fn the_workdirs_change_of_owner_fails_only_on_the_folder_itself() {
        let out = |code: i32, stdout: &str, stderr: &str, timed_out: bool| ExecOutput {
            exit_code: code,
            stdout: stdout.into(),
            stderr: stderr.into(),
            timed_out,
            ..Default::default()
        };
        let what = "giving the workdir to the verifier";
        let (o, kept) = own_result(
            &out(
                0,
                "BF root-owned\nBF unchanged chown: ./deep: File name too long\nBF owned root\n",
                "",
                false,
            ),
            "root",
            what,
            Phase::Handoff,
        )
        .unwrap();
        assert_eq!(o, "root");
        assert_eq!(kept, vec!["chown: ./deep: File name too long".to_string()]);
        let (o, kept) = own_result(
            &out(-1, "BF root-owned\n", "", true),
            "root",
            what,
            Phase::Handoff,
        )
        .unwrap();
        assert_eq!(o, "root");
        assert!(kept[0].contains("ran past its"), "{kept:?}");
        let p = own_result(&out(-1, "", "", true), "root", what, Phase::Handoff).unwrap_err();
        assert_eq!(
            (p.code.as_str(), p.owner, p.transient),
            ("R103", Owner::Sandbox, true)
        );
        let p = own_result(
            &out(2, "", "BF error no-user tester\n", false),
            "tester",
            what,
            Phase::Handoff,
        )
        .unwrap_err();
        assert_eq!((p.code.as_str(), p.owner), ("R315", Owner::Task));
        assert!(p.message.contains("no user `tester`"), "{}", p.message);
        let p = own_result(
            &out(
                3,
                "",
                "chown: /app: Read-only file system\nBF error chown /app\n",
                false,
            ),
            "root",
            what,
            Phase::Handoff,
        )
        .unwrap_err();
        assert_eq!((p.code.as_str(), p.owner), ("R315", Owner::Task));
        let p = own_result(&out(0, "", "", false), "root", what, Phase::Handoff).unwrap_err();
        assert_eq!(p.code, "R307");
        // Killed before it reported: the sandbox's (REVIEW-CORRECT L14).
        let p = own_result(&out(137, "", "", false), "root", what, Phase::Handoff).unwrap_err();
        assert_eq!(
            (p.code.as_str(), p.owner, p.transient),
            ("R103", Owner::Sandbox, true)
        );
        // Killed after the folder itself changed: noted (REVIEW-OUTCOMES L4).
        let (o, kept) = own_result(
            &out(137, "BF root-owned\n", "", false),
            "root",
            what,
            Phase::Handoff,
        )
        .unwrap();
        assert_eq!(o, "root");
        assert!(kept[0].contains("exit 137"), "{kept:?}");
        // An unreported return before a later stage is R307 in its own
        // phase (REVIEW-OUTCOMES L8).
        let p = own_result(&out(0, "", "", false), "agent", what, Phase::Setup).unwrap_err();
        assert_eq!((p.code.as_str(), p.phase), ("R307", Phase::Setup));
        // An image without a find it can use: the task's (L4).
        let p = own_result(
            &out(3, "", "BF error no-find\n", false),
            "root",
            what,
            Phase::Handoff,
        )
        .unwrap_err();
        assert_eq!((p.code.as_str(), p.owner), ("R315", Owner::Task));
        assert!(p.message.contains("no `find`"), "{}", p.message);
    }

    #[test]
    fn an_agent_user_that_can_become_root_is_recorded_as_failed_but_runs() {
        let (state, note) = root_path_verdict("agent, uid 1001".into(), &[], &[]);
        assert_eq!(state, DefenseState::Enforced);
        assert_eq!(note, "agent, uid 1001");
        let (state, note) = root_path_verdict(
            "agent, uid 1001".into(),
            &["sudo runs commands for it without a password ((ALL) NOPASSWD: ALL)"],
            &["file capabilities (no getcap)"],
        );
        assert_eq!(state, DefenseState::Failed);
        assert!(
            note.starts_with("agent, uid 1001, but it can become root: sudo runs commands"),
            "{note}"
        );
        assert!(
            note.ends_with("; not checked: file capabilities (no getcap)"),
            "{note}"
        );
    }

    #[test]
    fn an_agents_floor_raises_a_smaller_task_and_says_so() {
        use benchflow_agent::manifest::ResourceFloor;
        let floor = ResourceFloor {
            memory_mb: Some(4096),
            cpus: None,
        };
        // hello asks for nothing: the backend's default is raised.
        let (cpus, memory, raised) = apply_floor("openclaw", None, None, floor);
        assert_eq!((cpus, memory), (None, Some(4096)));
        let raised = raised.unwrap();
        assert_eq!(raised.memory_mb.as_ref().unwrap().task, None);
        assert_eq!(raised.memory_mb.as_ref().unwrap().sandbox, 4096.0);
        assert!(raised.cpus.is_none());
        assert_eq!(
            raised_line(&raised),
            "openclaw's manifest needs more than the task asks for: raised memory (the backend's default) to 4096 MB (result.json: resources_raised)"
        );
        // A task asking for 2 GB gets 4; one asking for 8 keeps 8.
        let (_, memory, raised) = apply_floor("openclaw", Some(1.0), Some(2048), floor);
        assert_eq!(memory, Some(4096));
        assert_eq!(raised.unwrap().memory_mb.unwrap().task, Some(2048.0));
        let (cpus, memory, raised) = apply_floor("openclaw", Some(1.0), Some(8192), floor);
        assert_eq!((cpus, memory, raised), (Some(1.0), Some(8192), None));
        // CPUs too.
        let floor = ResourceFloor {
            memory_mb: None,
            cpus: Some(2),
        };
        let (cpus, _, raised) = apply_floor("a", Some(1.0), None, floor);
        assert_eq!(cpus, Some(2.0));
        assert_eq!(raised.unwrap().cpus.unwrap().task, Some(1.0));
        let (_, _, raised) = apply_floor("a", None, None, ResourceFloor::default());
        assert!(raised.is_none());
    }

    #[test]
    fn tests_points_at_a_verifier_mounted_elsewhere_unless_something_is_there() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let verifier = root.join("verifier");
        std::fs::create_dir(&verifier).unwrap();
        let script = legacy_tests_link(verifier.to_str().unwrap())
            .replace(LEGACY_VERIFIER_MOUNT, root.join("tests").to_str().unwrap());
        let run = || {
            std::process::Command::new("sh")
                .arg("-c")
                .arg(&script)
                .status()
                .unwrap()
        };
        assert!(run().success());
        assert_eq!(std::fs::read_link(root.join("tests")).unwrap(), verifier);
        // A second run, or an image's own /tests, is left alone.
        assert!(run().success());
        std::fs::remove_file(root.join("tests")).unwrap();
        std::fs::create_dir(root.join("tests")).unwrap();
        assert!(run().success());
        assert!(root.join("tests").is_dir() && !root.join("tests").is_symlink());
    }

    #[test]
    fn package_paths_resolve_against_the_task_root() {
        let root = Path::new("tasks/hello-split");
        assert_eq!(
            in_package(root, Path::new("environment/Dockerfile")),
            Path::new("tasks/hello-split/environment/Dockerfile")
        );
        assert_eq!(
            in_package(root, Path::new("tasks/hello-split/sandbox")),
            Path::new("tasks/hello-split/sandbox")
        );
        assert_eq!(
            in_package(root, Path::new("/abs/sandbox")),
            Path::new("/abs/sandbox")
        );
    }

    #[test]
    fn the_image_workdir_falls_back_to_root_as_in_0_8() {
        assert_eq!(image_workdir("/root\n"), "/root");
        assert_eq!(image_workdir("/app"), "/app");
        assert_eq!(image_workdir("/\n"), "/root");
        assert_eq!(image_workdir(""), "/root");
        assert_eq!(image_workdir("motd\n/srv/work\n"), "/srv/work");
    }

    /// E7 round 2: Codex on MiMo sent a tool MiMo does not take; the one
    /// request was answered HTTP 400 (the forwarder then refused Codex's
    /// fallback model, lines with a `note`), and the trial was scored 0 with
    /// no problem. Every request rejected and none answered is the
    /// provider's R214; one answered request, or throttling, is not.
    #[test]
    fn a_provider_that_rejected_every_request_owns_the_trial() {
        let dir = tempfile::tempdir().unwrap();
        let codex = [
            r#"{"time":"2026-10-12T04:45:41.877Z","method":"POST","path":"/openai/responses","status":400,"bytes":174,"ms":1062}"#,
            r#"{"time":"2026-10-12T04:45:41.944Z","method":"POST","path":"/openai/responses","status":403,"bytes":126,"ms":1,"note":"this run may use only mimo-v2.6-flash; the request asked for gpt-5.6-luna"}"#,
        ]
        .join("\n");
        std::fs::write(dir.path().join("forwarder.log"), &codex).unwrap();
        let p = provider_rejected_all(dir.path()).expect("every request rejected");
        assert_eq!(
            (p.code.as_str(), p.owner, p.transient),
            ("R214", Owner::Model, false)
        );
        assert!(p.message.contains("(1 × HTTP 400)"), "{}", p.message);
        // One answered request: the agent had something to work from.
        std::fs::write(
            dir.path().join("forwarder.log"),
            format!("{codex}\n{}", r#"{"method":"POST","status":200}"#),
        )
        .unwrap();
        assert!(provider_rejected_all(dir.path()).is_none());
        // A refused key is R212, a spent plan R213 (which stops the job),
        // an unknown model R215, as the harnesses file them.
        for (status, code, owner) in [
            (401, "R212", Owner::Model),
            (403, "R212", Owner::Model),
            (402, "R213", Owner::Model),
            (404, "R215", Owner::You),
        ] {
            std::fs::write(
                dir.path().join("forwarder.log"),
                format!(r#"{{"method":"POST","status":{status}}}"#),
            )
            .unwrap();
            let p = provider_rejected_all(dir.path()).expect("rejected");
            assert_eq!((p.code.as_str(), p.owner), (code, owner), "{status}");
        }
        // Throttling is `throttled`'s, and no requests is nothing to say.
        std::fs::write(
            dir.path().join("forwarder.log"),
            r#"{"method":"POST","status":429}"#,
        )
        .unwrap();
        assert!(provider_rejected_all(dir.path()).is_none());
        std::fs::write(dir.path().join("forwarder.log"), "").unwrap();
        assert!(provider_rejected_all(dir.path()).is_none());
    }

    /// The agent user owns its workdir, and the owner of a folder can move
    /// a root-owned folder inside it aside. BenchFlow's root-only state must
    /// never be inside a workdir the runner hands out: the image's
    /// WORKDIR, its `/root` fallback, or a declared one.
    #[test]
    fn a_call_the_provider_never_answered_is_read_from_the_forwarders_log() {
        let dir = tempfile::tempdir().unwrap();
        let line = |v: serde_json::Value| v.to_string();
        let p = "/v1/chat/completions";
        let wait = |t: &str, h: &str, tools: bool| {
            if tools {
                line(
                    json!({"time": t, "method": "POST", "path": p, "note": "waiting", "attempt": 1, "body_hash": h, "tools": true}),
                )
            } else {
                line(
                    json!({"time": t, "method": "POST", "path": p, "note": "waiting", "attempt": 1, "body_hash": h}),
                )
            }
        };
        let note = |n: &str, h: &str| {
            line(json!({"method": "POST", "path": p, "note": n, "body_hash": h}))
        };
        let done = |h: &str| {
            line(json!({"method": "POST", "path": p, "status": 200, "ms": 900, "body_hash": h}))
        };
        let judge = |log: Vec<String>| {
            std::fs::write(dir.path().join("forwarder.log"), log.join("\n")).unwrap();
            never_answered(dir.path())
        };
        // The agent's last call with tools never started its answer, though
        // a title call beside it was answered: the provider's.
        let p1 = judge(vec![
            wait("t0", "a", true),
            done("a"),
            wait("t2", "b", true),
            wait("t3", "title", false),
            done("title"),
        ])
        .expect("call b had no answer");
        assert_eq!((p1.code.as_str(), p1.owner), ("R214", Owner::Model));
        assert!(p1.message.contains("sent at t2"), "{}", p1.message);
        // E5 review (A1, B1): a call the agent dropped early, then answered
        // calls, then a stall of the agent's own: not the provider's.
        assert!(
            judge(vec![
                wait("t0", "a", true),
                note("dropped", "a"),
                wait("t1", "b", true),
                done("b")
            ])
            .is_none()
        );
        // E5 review (C-A 1): BenchFlow's kill of the stalled agent drops the
        // call that was never answered; it is still the provider's.
        assert!(
            judge(vec![
                wait("t0", "a", true),
                done("a"),
                wait("t1", "b", true),
                note("dropped", "b")
            ])
            .is_some()
        );
        // E5 review (D-A): the agent drops a call and sends the same body
        // again, which is answered: the answer is the resend's, and the
        // agent's later stall is its own.
        assert!(
            judge(vec![
                wait("t0", "a", true),
                note("dropped", "a"),
                wait("t1", "a", true),
                note("answering", "a"),
                done("a")
            ])
            .is_none()
        );
        // E5 review (C-A 2): a refusal line without a body hash (a title
        // call's, on the same path) ends no other call.
        let unhashed = line(
            json!({"method": "POST", "path": p, "status": 502, "note": "the provider could not be reached (ECONNRESET)"}),
        );
        assert!(
            judge(vec![
                wait("t0", "a", true),
                wait("t1", "title", false),
                unhashed
            ])
            .is_some()
        );
        // An earlier call left open (a log without `dropped`) with a later
        // call answered: the stall is not that call's.
        assert!(
            judge(vec![
                wait("t0", "a", true),
                wait("t1", "b", true),
                done("b")
            ])
            .is_none()
        );
        // The provider had started answering (a slow stream): not silent.
        assert!(judge(vec![wait("t0", "c", true), note("answering", "c")]).is_none());
        // Retried and answered.
        assert!(
            judge(vec![
                wait("t0", "d", true),
                note("retrying after ECONNRESET", "d"),
                wait("t1", "d", true),
                done("d")
            ])
            .is_none()
        );
        // The same agent errors with an unreadable last answer, and a
        // storm of refusals until the time ran out (E5's 429 storm).
        let garbled = [
            line(json!({"time": "t0", "method": "POST", "path": "/v1/chat/completions", "status": 200, "unreadable": true})),
        ]
        .join("\n");
        std::fs::write(dir.path().join("forwarder.log"), &garbled).unwrap();
        let p = provider_instead("R205", dir.path(), Duration::from_secs(120)).expect("garbled");
        assert_eq!((p.code.as_str(), p.owner), ("R214", Owner::Model));
        assert!(provider_instead("R201", dir.path(), Duration::from_secs(120)).is_none());
        let storm: Vec<String> = (0..6)
            .map(|i| line(json!({"time": format!("2026-10-12T00:00:{:02}Z", i * 20), "method": "POST", "path": "/v1/chat/completions", "status": 429, "ms": 50, "body_hash": "a"})))
            .collect();
        std::fs::write(dir.path().join("forwarder.log"), storm.join("\n")).unwrap();
        let p = provider_instead("R206", dir.path(), Duration::from_secs(120)).expect("throttled");
        assert_eq!((p.code.as_str(), p.owner), ("R214", Owner::Model));
        // A log from before these lines (or none): nothing to say.
        std::fs::write(
            dir.path().join("forwarder.log"),
            line(json!({"method": "POST", "path": "/v1/x", "status": 200})),
        )
        .unwrap();
        assert!(never_answered(dir.path()).is_none());
    }

    #[test]
    fn throttling_is_read_from_native_calls_and_the_forwarders_log() {
        let dir = tempfile::tempdir().unwrap();
        let fwd = [
            r#"{"pin": "deepseek"}"#,
            r#"{"method": "GET", "status": 429}"#,
            r#"{"method": "POST", "status": 200}"#,
            r#"{"method": "POST", "status": 429}"#,
            r#"{"method": "POST", "status": 403, "note": "a model this run may not use"}"#,
            r#"{"method": "POST", "status": 529}"#,
        ]
        .join("\n");
        std::fs::write(dir.path().join("forwarder.log"), &fwd).unwrap();
        // Two refusals: not enough to say.
        assert!(throttled(dir.path(), None).is_none());
        std::fs::write(
            dir.path().join("forwarder.log"),
            format!("{fwd}\n{}", r#"{"method": "POST", "status": 503}"#),
        )
        .unwrap();
        let p = throttled(dir.path(), None).expect("3 of 4 model calls refused");
        assert_eq!((p.code.as_str(), p.owner), ("R214", Owner::Model));
        assert!(p.message.contains("3 of its 4"), "{}", p.message);
        // Refusals the agent got past, then an hour of its own work: the
        // timeout is the agent's.
        let recovered: Vec<&str> = [429, 429, 503, 200, 200, 200, 200, 200, 200, 200, 200, 200]
            .iter()
            .map(|s| match s {
                429 => r#"{"status": 429}"#,
                503 => r#"{"status": 503}"#,
                _ => r#"{"status": 200}"#,
            })
            .collect();
        let other = tempfile::tempdir().unwrap();
        std::fs::write(other.path().join("model_calls.jsonl"), recovered.join("\n")).unwrap();
        assert!(throttled(other.path(), None).is_none());
        // With times: a provider that kept refusing (16 of 40, each followed
        // by a minute's backoff) then answered last, before a tool run hit
        // the timeout, still held the agent back (REVIEW-OUTCOMES M3)...
        let line = |start_s: i64, status: u16| {
            let at = chrono::DateTime::from_timestamp(1_760_000_000 + start_s, 0).unwrap();
            format!(
                r#"{{"started_at": "{}", "duration_ms": 2000, "status": {status}}}"#,
                at.to_rfc3339()
            )
        };
        let mut t = 0;
        let mut starved = Vec::new();
        for i in 0..40 {
            let refused = i % 5 < 2 && i < 39;
            starved.push(line(t, if refused { 429 } else { 200 }));
            t += if refused { 62 } else { 10 };
        }
        let timed = tempfile::tempdir().unwrap();
        std::fs::write(timed.path().join("model_calls.jsonl"), starved.join("\n")).unwrap();
        assert!(throttled(timed.path(), Some(Duration::from_secs(1800))).is_some());
        // The same in the forwarder's log: `time` at each request's end.
        let fwd: Vec<String> = starved
            .iter()
            .map(|l| {
                let v: Value = serde_json::from_str(l).unwrap();
                let start = chrono::DateTime::parse_from_rfc3339(v["started_at"].as_str().unwrap())
                    .unwrap();
                let end = start + chrono::Duration::milliseconds(2000);
                format!(
                    r#"{{"time": "{}", "method": "POST", "status": {}, "ms": 2000}}"#,
                    end.to_rfc3339(),
                    v["status"]
                )
            })
            .collect();
        let behind = tempfile::tempdir().unwrap();
        std::fs::write(behind.path().join("forwarder.log"), fwd.join("\n")).unwrap();
        assert!(throttled(behind.path(), Some(Duration::from_secs(1800))).is_some());
        // ...while three refusals in the first minute of an hour the agent
        // then spent on its own work do not.
        let mut early: Vec<String> = (0..3).map(|i| line(i * 20, 429)).collect();
        early.extend((0..9).map(|i| line(120 + i * 400, 200)));
        std::fs::write(timed.path().join("model_calls.jsonl"), early.join("\n")).unwrap();
        assert!(throttled(timed.path(), Some(Duration::from_secs(3600))).is_none());
        // Requests side by side are counted once, and an answer streaming
        // is no waiting (REVIEW-OUTCOMES (d7c) M2): 20 answers of 60 s, with
        // seven side requests refused in 0.3 s during them...
        let span_line = |start_ms: i64, ms: i64, status: u16| {
            let at = chrono::DateTime::from_timestamp_millis(1_760_000_000_000 + start_ms).unwrap();
            format!(
                r#"{{"started_at": "{}", "duration_ms": {ms}, "status": {status}}}"#,
                at.to_rfc3339()
            )
        };
        let mut side = Vec::new();
        for i in 0..20 {
            side.push(span_line(i * 61_000, 60_000, 200));
            if i % 3 == 0 && i < 20 {
                side.push(span_line(i * 61_000 + 10_000, 300, 429));
            }
        }
        side.push(span_line(20 * 61_000, 1_000, 200));
        std::fs::write(timed.path().join("model_calls.jsonl"), side.join("\n")).unwrap();
        // In 20 minutes: the old count (each refusal to the next answer's
        // start, 357 s) passed a 300 s bar (REVIEW-B (d7d) F4).
        assert!(throttled(timed.path(), Some(Duration::from_secs(1200))).is_none());
        // ...and three bursts of four parallel requests refused after 120 s
        // (about 370 s of waiting in a 30-minute run).
        let mut bursts = Vec::new();
        for b in 0..3 {
            for _ in 0..4 {
                bursts.push(span_line(b * 123_000, 120_000, 504));
            }
        }
        for k in 0..12 {
            bursts.push(span_line(369_000 + k * 100_000, 2_000, 200));
        }
        std::fs::write(timed.path().join("model_calls.jsonl"), bursts.join("\n")).unwrap();
        assert!(throttled(timed.path(), Some(Duration::from_secs(1800))).is_none());
        // Durations past any clock do not overflow (REVIEW-SECURITY (d7c) 1).
        std::fs::write(
            timed.path().join("model_calls.jsonl"),
            [
                span_line(0, i64::MAX, 429),
                span_line(1, i64::MAX, 429),
                span_line(2, i64::MAX, 429),
                span_line(3, 1, 200),
            ]
            .join("\n"),
        )
        .unwrap();
        let _ = throttled(timed.path(), Some(Duration::from_secs(60)));
        // A run whose provider mostly answered is the agent's to lose.
        let ok: Vec<String> = (0..20).map(|_| r#"{"status": 200}"#.to_string()).collect();
        std::fs::write(dir.path().join("model_calls.jsonl"), ok.join("\n")).unwrap();
        assert!(throttled(dir.path(), None).is_none());
    }

    /// A refusal waits only until the agent sends the same request again: a
    /// refused side request it went on without does not make its tool time
    /// waiting, and a refusal last in a log with times no longer means held
    /// back whatever the times. Guards the D7e fix of REVIEW-A (d7d)
    /// finding 1 (failures A and B).
    #[test]
    fn a_refusal_waits_until_the_same_request_is_sent_again() {
        // A forwarder line: it ended at `end_ms` after `ms`, with its body's hash.
        let fwd = |end_ms: i64, ms: i64, status: u16, hash: &str| {
            let at = chrono::DateTime::from_timestamp_millis(1_760_000_000_000 + end_ms).unwrap();
            format!(
                r#"{{"time": "{}", "method": "POST", "path": "/anthropic/v1/messages", "status": {status}, "ms": {ms}, "body_hash": "{hash}"}}"#,
                at.to_rfc3339()
            )
        };
        let thirty = Some(Duration::from_secs(1800));
        // 30 turns in 30 minutes: an 8 s answer, then 0.2 s later a 0.5 s
        // side request (refused on 20 turns and never sent again), then a
        // 50 s tool run. No request the agent needed was refused; the old
        // count made 1,010 s of its tool time waiting.
        let mut side = Vec::new();
        let mut t = 0;
        for turn in 0..30 {
            side.push(fwd(t + 8_000, 8_000, 200, &format!("main{turn:02}")));
            let status = if turn % 3 < 2 { 429 } else { 200 };
            side.push(fwd(t + 8_700, 500, status, &format!("side{turn:02}")));
            t += 58_700;
        }
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("forwarder.log"), side.join("\n")).unwrap();
        assert!(throttled(dir.path(), thirty).is_none());
        // The provider refusing the main request instead, sent again after
        // each 30 s backoff, did hold the agent back.
        let mut main = Vec::new();
        let mut t = 0;
        for turn in 0..20 {
            let hash = format!("main{turn:02}");
            for _ in 0..2 {
                main.push(fwd(t + 1_000, 1_000, 529, &hash));
                t += 31_000;
            }
            main.push(fwd(t + 8_000, 8_000, 200, &hash));
            t += 28_000;
        }
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("forwarder.log"), main.join("\n")).unwrap();
        assert!(throttled(dir.path(), thirty).is_some());
        // The same from a native agent, whose tries of one call are its
        // `attempt`s: 12 calls answered on the third try, a minute apart.
        let tried = |start_s: i64, status: u16, attempt: u32| {
            let at = chrono::DateTime::from_timestamp(1_760_000_000 + start_s, 0).unwrap();
            format!(
                r#"{{"started_at": "{}", "duration_ms": 2000, "status": {status}, "attempt": {attempt}}}"#,
                at.to_rfc3339()
            )
        };
        let mut native = Vec::new();
        for call in 0..12 {
            let t = call * 136;
            native.push(tried(t, 429, 1));
            native.push(tried(t + 62, 429, 2));
            native.push(tried(t + 124, 200, 3));
        }
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model_calls.jsonl"), native.join("\n")).unwrap();
        assert!(throttled(dir.path(), thirty).is_some());
        // A native call's refusal waits for that call's next try, never
        // another call's (SMALL-DELTAS-B, d7e): three calls refused, their
        // second tries lost (no status) and their third answered 20 s
        // later, 25 minutes of the agent's own turns, then one late 429 and
        // its retry. About 70 s of waiting, not held back; matching the
        // attempt alone ran each early refusal to the late call's retry.
        let call = |seq: u64, attempt: u32, start_s: i64, status: &str| {
            let at = chrono::DateTime::from_timestamp(1_760_000_000 + start_s, 0).unwrap();
            format!(
                r#"{{"seq": {seq}, "attempt": {attempt}, "started_at": "{}", "duration_ms": 2000, "status": {status}}}"#,
                at.to_rfc3339()
            )
        };
        let mut calls = Vec::new();
        for seq in 1..=3 {
            let t = (seq as i64 - 1) * 30;
            calls.push(call(seq, 1, t, "429"));
            calls.push(call(seq, 2, t + 8, "null"));
            calls.push(call(seq, 3, t + 20, "200"));
        }
        for seq in 4..=6 {
            calls.push(call(seq, 1, (seq as i64 - 3) * 300, "200"));
        }
        calls.push(call(7, 1, 1_700, "429"));
        calls.push(call(7, 2, 1_710, "200"));
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model_calls.jsonl"), calls.join("\n")).unwrap();
        assert!(throttled(dir.path(), thirty).is_none());
        // The same calls retried after long backoffs did hold it back.
        let mut slow = Vec::new();
        for seq in 1..=12 {
            let t = (seq as i64 - 1) * 136;
            slow.push(call(seq, 1, t, "429"));
            slow.push(call(seq, 2, t + 62, "null"));
            slow.push(call(seq, 3, t + 124, "200"));
        }
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model_calls.jsonl"), slow.join("\n")).unwrap();
        assert!(throttled(dir.path(), thirty).is_some());
        // Three refusals in the first minute, an hour of the agent's own
        // work, then one refused side request last in a log that has no
        // way to tell it was not sent again: 164 s of waiting, not held
        // back, though the old rule said so from the last line alone.
        let line = |start_s: i64, status: u16| {
            let at = chrono::DateTime::from_timestamp(1_760_000_000 + start_s, 0).unwrap();
            format!(
                r#"{{"started_at": "{}", "duration_ms": 2000, "status": {status}}}"#,
                at.to_rfc3339()
            )
        };
        let mut late: Vec<String> = (0..3).map(|i| line(i * 20, 429)).collect();
        late.extend((0..9).map(|i| line(120 + i * 400, 200)));
        late.push(line(3_330, 429));
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model_calls.jsonl"), late.join("\n")).unwrap();
        assert!(throttled(dir.path(), Some(Duration::from_secs(3600))).is_none());
        // Without times, the last line still decides.
        let untimed = [
            r#"{"status": 200}"#,
            r#"{"status": 429}"#,
            r#"{"status": 429}"#,
            r#"{"status": 429}"#,
        ];
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model_calls.jsonl"), untimed.join("\n")).unwrap();
        assert!(throttled(dir.path(), None).is_some());
    }

    #[test]
    fn root_only_state_is_never_inside_the_workdir() {
        for pwd in [
            "/root\n",
            "/\n",
            "",
            "/app\n",
            "/root/work\n",
            "/home/agent\n",
        ] {
            let w = image_workdir(pwd);
            assert!(
                !hardening::holds_state(&w),
                "{w} would hold {}",
                hardening::STATE_DIR
            );
            assert!(
                !std::path::Path::new(hardening::STATE_DIR).starts_with(&w),
                "{w} holds {}",
                hardening::STATE_DIR
            );
        }
        for w in ["/", "/var", "/var/lib/", hardening::STATE_DIR] {
            assert!(hardening::holds_state(w), "{w}");
        }
        assert!(is_root_home("/root") && is_root_home("/root/"));
        assert!(!is_root_home("/root/app") && !is_root_home("/app"));
    }
}
