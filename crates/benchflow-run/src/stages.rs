//! Stages graded one after another in one sandbox (the split layout's
//! `[[steps]]`, task.md stages graded on their own).
//!
//! For each stage, in order:
//!
//! ```text
//! reset → stage files → attempt → handoff → outputs → hooks → verify → gate
//! ```
//!
//! The reset removes the previous stage's verifier and reference files, so
//! the agent never sees them. The attempt gets the stage's own prompt and
//! time: the oracle runs the stage's solution (else the task's), `nop` does
//! nothing, a control or probe runs its script with `BENCHFLOW_STAGE` set,
//! and a model agent gets the stage's prompt in a fresh session. The
//! handoff kills every agent process before the stage is graded, as for a
//! single prompt. A gate the stage misses stops the trial there.
//!
//! With `--resume-sessions`, each stage
//! after the first resumes the agent's own session instead, when the agent
//! can ([`crate::agent::ModelAgent::session_dirs`]): its home outlives the
//! handoff, so the session's files are still there; BenchFlow checks that
//! they are, and the harness starts the agent on them (ACP `session/load`
//! with the previous stage's session id, or the headless command's resume
//! flag). An agent that cannot resume, or whose session folders are empty,
//! starts fresh. Each stage's row says which happened (`session`: `id`,
//! `resumed`, and `why` when a stage that could have resumed did not).
//!
//! Each stage leaves `steps/<name>/` in the trial folder (`agent/`,
//! `trajectory/`, `verifier/`, `artifacts/`, `hooks/`), and its result in
//! result.json's `step_results`, in the split layout's shape (`step_name`,
//! `verifier_result.rewards`) plus the gate's outcome. The trial's reward
//! follows the task's rule ([`Combine`], [`Unreached`]); `verifier/` at the
//! top holds it, with the last graded stage's output.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use benchflow_sandbox::shell;
use benchflow_task::extras::{Combine, StepSpec, Steps, Unreached};
use serde_json::{Value, json};

use super::{
    OWN_TIMEOUT, PROBE_TIMEOUT, SETUP_TIMEOUT, Trial, agent_timeout, sandbox_problem,
    scripted_summary,
};
use crate::agent::{AgentContext, Attempt, Resume};
use crate::hardening::{self, Report};
use crate::outcome::{Owner, Phase, Problem};
use crate::scoring::Score;

/// How a stage ended, when it ran to the end.
struct Graded {
    values: BTreeMap<String, f64>,
    gate_failed: bool,
}

impl Trial<'_> {
    /// Run every stage, then combine their rewards.
    pub(super) async fn run_steps(&mut self) -> Result<Score, Problem> {
        let task = self.spec.task.clone();
        let steps = task.extras.steps.clone().unwrap_or_default();
        let mut rows: Vec<Value> = Vec::new();
        let mut graded: Vec<BTreeMap<String, f64>> = Vec::new();
        let mut events: Vec<Value> = Vec::new();
        let mut prompts: Vec<String> = Vec::new();
        let mut failure: Option<Problem> = None;
        let mut last_graded: Option<String> = None;
        // A child of a `stage:<name>` fork goes on after the parent's
        // stages, whose rows and grades it carries (marked `from_parent`).
        let mut first = 0;
        // A stage worked in this sandbox already: a stage-fork child's
        // parent ran the stages it carries.
        let mut worked = false;
        if let Some(c) = self.child_stage_carry() {
            first = c.index + 1;
            worked = true;
            rows = c.rows;
            for r in &mut rows {
                r["from_parent"] = json!(true);
            }
            graded = c.graded;
            prompts = c.prompts;
            last_graded = c.last_graded;
            self.agent.timed_out = c.timed_out;
        }
        let mut fork_at = self.fork_stage();
        for (i, st) in steps.stages.iter().enumerate().skip(first) {
            let mut row = json!({ "step_name": st.name, "reached": true, "graded": st.graded });
            let ended = self
                .run_stage(i, st, &mut row, &mut events, &mut prompts)
                .await;
            rows.push(row);
            match ended {
                Ok(None) => worked = true,
                Ok(Some(g)) => {
                    worked = true;
                    graded.push(g.values);
                    last_graded = Some(st.name.clone());
                    if g.gate_failed {
                        break;
                    }
                }
                Err(p) => {
                    failure = Some(p);
                    break;
                }
            }
            // The parent's fork point: right after this graded stage, whose
            // handoff stopped the agent and checked the sandbox.
            if fork_at.as_deref() == Some(st.name.as_str()) {
                let carry = super::branch::StageCarry {
                    index: i,
                    rows: rows.clone(),
                    graded: graded.clone(),
                    prompts: prompts.clone(),
                    last_graded: last_graded.clone(),
                    timed_out: self.agent.timed_out,
                };
                // Once: a later stage of the same name (two `[[steps]]`
                // may share one) forks nothing more.
                fork_at = None;
                if let Err(p) = self.at_fork_point(Some(carry)).await {
                    failure = Some(p);
                    break;
                }
            }
        }
        let reached = rows.len();
        for st in &steps.stages[reached..] {
            rows.push(json!({
                "step_name": st.name, "reached": false, "graded": st.graded,
                "verifier_result": null,
            }));
        }
        // Every stage's stop, in order, with its stage: `stops` lists them
        // and `stop` is the first, so no stage's stop is lost to a later
        // stage's summary (a fork child's carried rows count too).
        let stops: Vec<Value> = rows
            .iter()
            .filter_map(|r| {
                let mut stop = r.get("stop")?.clone();
                if let Some(m) = stop.as_object_mut() {
                    m.insert("stage".into(), r["step_name"].clone());
                }
                Some(stop)
            })
            .collect();
        if let Some(first) = stops.first() {
            if !self.agent.trajectory_summary.is_object() {
                self.agent.trajectory_summary = json!({});
            }
            if let Some(m) = self.agent.trajectory_summary.as_object_mut() {
                m.insert("stop".into(), first.clone());
                m.insert("stops".into(), Value::Array(stops.clone()));
            }
        }
        write_lines(&self.spec.dir.join("trajectory"), &events);
        self.agent.trajectory_summary = staged_summary(&events, &self.agent.trajectory_summary);
        self.agent.prompts = prompts;
        self.stage_worked = worked;
        self.x.step_results = Some(Value::Array(rows));
        if let Some(p) = failure {
            return Err(p);
        }
        let unreached = steps.stages[reached..].iter().filter(|s| s.graded).count();
        let score = combine(&steps, &graded, unreached);
        self.publish_combined(&score, last_graded.as_deref());
        Ok(score)
    }

    /// One stage. `Ok(None)` for an ungraded stage; `row` gathers its record.
    async fn run_stage(
        &mut self,
        index: usize,
        st: &StepSpec,
        row: &mut Value,
        events: &mut Vec<Value>,
        prompts: &mut Vec<String>,
    ) -> Result<Option<Graded>, Problem> {
        let task = self.spec.task.clone();
        // The name is a folder in the trial: one plain component, never a
        // path out of it (the task readers refuse other names).
        let plain = matches!(
            Path::new(&st.name).components().collect::<Vec<_>>()[..],
            [std::path::Component::Normal(_)]
        );
        if !plain {
            return Err(Problem::new(
                "R104",
                Owner::Task,
                Phase::Setup,
                format!("stage name {:?} is not a plain folder name", st.name),
                "rename the stage: no slashes, no `.` or `..`",
            ));
        }
        let dir = self.spec.dir.join("steps").join(&st.name);
        let _ = std::fs::create_dir_all(&dir);
        self.rt
            .events
            .log(self.name(), format!("stage {}: {}", index + 1, st.name));
        if index > 0 {
            // The previous stage's grading and reference files go before
            // the agent works again.
            let mut gone = vec![task.verifier.mount.clone()];
            if let Some(o) = &task.oracle {
                gone.push(o.mount.clone());
            }
            let gone: Vec<String> = gone.iter().map(|p| shell::quote(p)).collect();
            self.root(
                format!(
                    "rm -rf {} /logs/verifier; mkdir -p /logs/verifier",
                    gone.join(" ")
                ),
                SETUP_TIMEOUT,
            )
            .await
            .map_err(|e| sandbox_problem(Phase::Setup, e))?;
            // Grading set the verifier's network; the agent gets its own back.
            self.enforce_network(
                crate::netpolicy::Stage::Agent,
                crate::netpolicy::Host::Agent,
            )
            .await?;
        }
        // An ungraded stage's agent keeps its processes into this one.
        let agent_alive = index > 0
            && self.user != "root"
            && task
                .extras
                .steps
                .as_ref()
                .is_some_and(|s| !s.stages[index - 1].graded);
        self.prepare_stage(st, index > 0, agent_alive).await?;

        let (resume, fresh_why) = self.stage_resume(index).await?;
        if let Some(why) = &fresh_why {
            self.rt
                .events
                .log(self.name(), format!("stage {}: {why}", st.name));
        }

        self.rt.events.phase(self.name(), Phase::Agent);
        let clock = Instant::now();
        let attempted = self.stage_attempt(st, resume, index > 0).await;
        // The session is saved once the agent's work stops: after a graded
        // stage (whose kill comes first anyway), the last stage, or a stage
        // that failed. An ungraded stage's agent keeps its processes for the
        // next stage, so its session waits for the next save.
        let last = task
            .extras
            .steps
            .as_ref()
            .is_none_or(|s| index + 1 >= s.stages.len());
        if st.graded || last || attempted.is_err() {
            self.save_session(true).await;
        }
        if matches!(self.spec.attempt, Attempt::Model(_)) {
            row["session"] = json!({
                "id": self.agent.session_id,
                "resumed": self.agent.resumed,
                "why": fresh_why.or(self.agent.resume_note.clone()),
            });
        }
        let agent_secs = clock.elapsed().as_secs_f64();
        self.timing.agent_execution += agent_secs;
        prompts.push(self.stage_prompt(st));
        archive_attempt(&self.spec.dir, &dir, &st.name, events);
        row["timed_out"] = json!(self.agent.timed_out);
        row["agent_execution_sec"] = json!(agent_secs);
        // A loop the model kept failing (DECISIONS row 27) is this stage's:
        // the next stage's summary replaces the agent's, so the row keeps it.
        if let Some(stop) = self.agent.trajectory_summary.get("stop") {
            row["stop"] = stop.clone();
        }
        if let Err(p) = attempted {
            row["error"] = json!(p.to_string());
            return Err(p);
        }
        if !st.graded {
            return Ok(None);
        }

        self.rt.events.phase(self.name(), Phase::Handoff);
        let clock = Instant::now();
        let graded = self.grade_stage(st, &dir).await;
        let verifier_secs = clock.elapsed().as_secs_f64();
        self.timing.verifier += verifier_secs;
        row["verifier_sec"] = json!(verifier_secs);
        row["verifier_exit_code"] = json!(self.verifier_exit);
        let score = match graded {
            Ok(s) => s,
            Err(p) => {
                // The row records what the trial reports.
                let p = self.sandbox_lost_instead(p).await;
                row["verifier_result"] = Value::Null;
                row["error"] = json!(p.to_string());
                return Err(p);
            }
        };
        let mut values = score.values.clone();
        values.insert("reward".into(), score.reward);
        row["verifier_result"] = json!({ "rewards": values });
        row["reward"] = json!(score.reward);
        row["passed"] = json!(score.passed);
        let mut gate_failed = false;
        if let Some(gate) = &st.gate {
            let failure = gate.failure(Some(&values));
            gate_failed = failure.is_some();
            row["gate"] = json!({ "min_reward": gate, "passed": !gate_failed, "why": failure });
            if let Some(why) = failure {
                self.rt.events.log(
                    self.name(),
                    format!(
                        "stage {} did not pass its gate ({why}); later stages are not run",
                        st.name
                    ),
                );
            }
        }
        Ok(Some(Graded {
            values,
            gate_failed,
        }))
    }

    /// The stage's files, mounts, setup script, and readiness check.
    /// `later`: an earlier stage's agent has worked in this sandbox.
    /// `agent_alive`: that stage was not graded, so no handoff stopped its
    /// processes, and they still run.
    ///
    /// Root never writes into the workdir here: the agent owns it, may
    /// have left links in it, and with `agent_alive` can swap any path
    /// while root works. The stage's files are staged by root and copied in
    /// by the agent user ([`crate::place::tree_for_agent`]).
    async fn prepare_stage(
        &mut self,
        st: &StepSpec,
        later: bool,
        agent_alive: bool,
    ) -> Result<(), Problem> {
        let task = self.spec.task.clone();
        if later && !agent_alive && self.user != "root" {
            // Grading the last stage gave the workdir to root, and its
            // handoff stopped every agent process; the agent works in it
            // again. After a stage not graded it is still the agent's.
            let user = self.user.clone();
            let out = self
                .root(
                    hardening::own_workdir(&self.workdir, &user, &user),
                    OWN_TIMEOUT,
                )
                .await
                .map_err(|e| sandbox_problem(Phase::Setup, e))?;
            let (_, kept) = super::own_result(
                &out,
                &user,
                "giving the workdir back to the agent",
                Phase::Setup,
            )?;
            self.custody.workdir_unchanged.extend(kept);
        }
        self.place_mounts(&st.mounts, Phase::Setup, later, agent_alive, Some(&st.name))
            .await?;
        if let Some(rel) = &st.workdir {
            let local = super::in_package(&task.root, rel);
            let workdir = self.workdir.clone();
            if self.user == "root" {
                self.sb()
                    .upload(&local, &workdir)
                    .await
                    .map_err(|e| sandbox_problem(Phase::Setup, e))?;
            } else {
                let user = crate::place::AgentUser {
                    name: &self.user,
                    home: &self.home,
                    earlier_turns: later,
                };
                let task_paths = self.task_paths(Some(&st.name));
                crate::place::tree_for_agent(self.sb(), &user, &local, &workdir, &task_paths)
                    .await?;
            }
            if local.join("setup.sh").is_file() {
                let script = format!("{}/setup.sh", workdir.trim_end_matches('/'));
                let mut exec = benchflow_sandbox::Exec::new(shell::script_file(&script))
                    .user(self.user.clone())
                    .cwd(workdir.clone())
                    .timeout(SETUP_TIMEOUT);
                exec.env.insert("HOME".into(), self.home.clone());
                let out = self
                    .sb()
                    .exec(exec)
                    .await
                    .map_err(|e| sandbox_problem(Phase::Setup, e))?;
                if !out.success() {
                    return Err(Problem::new(
                        "R104",
                        Owner::Task,
                        Phase::Setup,
                        format!(
                            "stage {}'s setup.sh exited with code {}",
                            st.name, out.exit_code
                        ),
                        "fix the stage's workdir/setup.sh",
                    )
                    .detail(out.tail(1500)));
                }
            }
        }
        // The test-runner files the agent is handed now include the stage's
        // own (a conftest.py in its workdir/), which the next handoff must
        // keep, and restore if the agent edits them. The baseline describes
        // the task as shipped: with no agent process left (before the first
        // turn, or after a graded stage's handoff put back every file the
        // agent planted or edited), it is taken again; while an earlier
        // stage's agent still runs, only the stage's own files are added, as
        // read on this machine.
        if agent_alive {
            self.add_to_baseline(st).await?;
        } else {
            let out = self
                .root(hardening::baseline(), SETUP_TIMEOUT)
                .await
                .map_err(|e| sandbox_problem(Phase::Setup, e))?;
            if !(out.success() && Report::parse(&out.stdout).first("baseline").is_some()) {
                return Err(super::unreported("the stage's test-runner baseline", &out));
            }
        }
        if let Some(ready) = &st.ready {
            self.wait_ready(ready, later).await?;
        }
        Ok(())
    }

    /// Add the test-runner files stage `st` ships (in its workdir/ folder
    /// and its mounts) to the baseline, from the files on this machine: the
    /// copies in the sandbox may already be the running agent's
    /// ([`hardening::baseline_add`]).
    async fn add_to_baseline(&mut self, st: &StepSpec) -> Result<(), Problem> {
        let task = self.spec.task.clone();
        let mut shipped = Vec::new();
        if let Some(rel) = &st.workdir {
            shipped.extend(steering_in(
                &super::in_package(&task.root, rel),
                &self.workdir,
            ));
        }
        for m in &st.mounts {
            shipped.extend(steering_in(
                &super::in_package(&task.root, &m.source),
                &m.target,
            ));
        }
        if shipped.is_empty() {
            return Ok(());
        }
        let mut entries = Vec::new();
        for (path, bytes, mode) in shipped {
            let copy = format!("{}/baseline/added{path}", hardening::STATE_DIR);
            self.sb()
                .write_file(&copy, &bytes, mode & 0o755)
                .await
                .map_err(|e| sandbox_problem(Phase::Setup, e))?;
            entries.push((
                path,
                hex::encode(<sha2::Sha256 as sha2::Digest>::digest(&bytes)),
            ));
        }
        let out = self
            .root(hardening::baseline_add(&entries), SETUP_TIMEOUT)
            .await
            .map_err(|e| sandbox_problem(Phase::Setup, e))?;
        let report = Report::parse(&out.stdout);
        if report.first("baseline-missing").is_some() {
            return Ok(());
        }
        if !(out.success() && report.first("baseline-added").is_some()) {
            return Err(super::unreported("the stage's test-runner baseline", &out));
        }
        Ok(())
    }

    /// Whether stage `index` resumes the agent's session: the session to
    /// resume, or why it starts fresh when it could have resumed.
    async fn stage_resume(
        &mut self,
        index: usize,
    ) -> Result<(Option<Resume>, Option<String>), Problem> {
        let Attempt::Model(agent) = self.spec.attempt.clone() else {
            return Ok((None, None));
        };
        // A child's first stage resumes the parent's session when asked
        // (`--resume-session`, which fails rather than start fresh), and
        // otherwise starts a fresh one.
        if let Some(c) = self.child_stage_carry()
            && index == c.index + 1
        {
            return Ok((
                self.child_session().map(|id| Resume {
                    session_id: Some(id),
                    loaded: true,
                }),
                None,
            ));
        }
        // The first stage resumes only a session loaded from a file.
        if index == 0 {
            return Ok((self.place_session(agent.as_ref()).await?, None));
        }
        if !self.rt.options.resume_sessions {
            return Ok((None, None));
        }
        let Some(dirs) = agent.session_dirs() else {
            return Ok((
                None,
                Some(format!(
                    "{} cannot resume its own session on this harness, so this stage started a fresh one",
                    agent.name()
                )),
            ));
        };
        // The session's files must have outlived the handoff.
        let home = self.home.trim_end_matches('/');
        let paths: Vec<String> = dirs
            .iter()
            .map(|d| shell::quote(&format!("{home}/{d}")))
            .collect();
        let out = self
            .root(
                format!(
                    "for d in {}; do [ -n \"$(ls -A \"$d\" 2>/dev/null)\" ] && echo \"BF kept $d\"; done; true",
                    paths.join(" ")
                ),
                SETUP_TIMEOUT,
            )
            .await
            .map_err(|e| sandbox_problem(Phase::Setup, e))?;
        if !out.stdout.contains("BF kept ") {
            return Ok((
                None,
                Some(format!(
                    "the agent's session folders (~/{}) were empty when this stage started, so it started a fresh session",
                    dirs.join(", ~/")
                )),
            ));
        }
        Ok((
            Some(Resume {
                session_id: self.agent.session_id.clone(),
                loaded: false,
            }),
            None,
        ))
    }

    /// The prompt stage `st` sends: its own, or for a child of a
    /// `stage:<name>` fork, in its first stage, the child's own prompt
    /// (`@instruction` in it is the stage's prompt).
    fn stage_prompt(&self, st: &StepSpec) -> String {
        let first_of_child = self.child_stage_carry().is_some_and(|c| {
            self.spec
                .task
                .extras
                .steps
                .as_ref()
                .and_then(|s| s.stages.get(c.index + 1))
                .is_some_and(|s| s.name == st.name)
        });
        first_of_child
            .then(|| self.child_prompt(&st.prompt))
            .flatten()
            .unwrap_or_else(|| st.prompt.clone())
    }

    /// Run the attempt for one stage, resuming `resume` when set.
    async fn stage_attempt(
        &mut self,
        st: &StepSpec,
        resume: Option<Resume>,
        earlier_turns: bool,
    ) -> Result<(), Problem> {
        let task = self.spec.task.clone();
        let opts = &self.rt.options;
        let timeout = st
            .timeout
            .map(|d| benchflow_task::values::scale_duration(d, opts.timeout_multiplier.max(0.01)))
            .or_else(|| agent_timeout(&task, opts));
        let base = self.stage_prompt(st);
        let prompt = match &opts.extra_instruction {
            Some(extra) => format!("{}\n\n{}", base.trim_end(), extra),
            None => base,
        };
        let _ = std::fs::create_dir_all(self.spec.dir.join("trajectory"));
        let _ = std::fs::create_dir_all(self.spec.dir.join("agent"));
        let mut stage_env = BTreeMap::new();
        stage_env.insert("BENCHFLOW_STAGE".to_string(), st.name.clone());
        let earlier_timeout = self.agent.timed_out;
        let result = match self.spec.attempt.clone() {
            Attempt::Nop => {
                write_lines(&self.spec.dir.join("trajectory"), &[json!({"type": "nop"})]);
                self.agent.trajectory_summary = scripted_summary("nop");
                Ok(())
            }
            Attempt::Oracle => {
                let oracle = task.oracle.clone().expect("checked by refusal");
                let dir = st.oracle_dir.clone().unwrap_or(oracle.dir.clone());
                self.sb()
                    .upload(&super::in_package(&task.root, &dir), &oracle.mount)
                    .await
                    .map_err(|e| sandbox_problem(Phase::Agent, e))?;
                let m = shell::quote(&oracle.mount);
                let _ = self.root(format!("chmod -R a+rX {m}"), SETUP_TIMEOUT).await;
                let script = format!("{}/{}", oracle.mount.trim_end_matches('/'), oracle.entry);
                let label = format!(
                    "{}/{}",
                    dir.strip_prefix(&task.root).unwrap_or(&dir).display(),
                    oracle.entry
                );
                let mut env = oracle.env.clone();
                env.insert("DEBIAN_FRONTEND".into(), "noninteractive".into());
                env.extend(stage_env);
                let r = self
                    .run_script(
                        "oracle",
                        &label,
                        &shell::script_file(&script),
                        env,
                        timeout,
                        &prompt,
                    )
                    .await;
                let _ = self.root(format!("rm -rf {m}"), SETUP_TIMEOUT).await;
                r
            }
            Attempt::Control(control) => {
                let bytes =
                    std::fs::read(super::in_package(&task.root, &control.script)).map_err(|e| {
                        Problem::new(
                            "R306",
                            Owner::Task,
                            Phase::Agent,
                            format!("control {} is unreadable: {e}", control.id),
                            "check the file under controls/",
                        )
                    })?;
                self.sb()
                    .write_file("/tmp/.benchflow-control.sh", &bytes, 0o755)
                    .await
                    .map_err(|e| sandbox_problem(Phase::Agent, e))?;
                let label = format!("controls/{}", control.id);
                self.run_script(
                    "control",
                    &label,
                    &shell::script_file("/tmp/.benchflow-control.sh"),
                    stage_env,
                    timeout,
                    &prompt,
                )
                .await
            }
            Attempt::Script(script) => {
                self.sb()
                    .write_file("/tmp/.benchflow-script.sh", &script.bytes, 0o755)
                    .await
                    .map_err(|e| sandbox_problem(Phase::Agent, e))?;
                self.run_script(
                    "script",
                    &format!("script:{}", script.name),
                    &shell::script_file("/tmp/.benchflow-script.sh"),
                    stage_env,
                    timeout,
                    &prompt,
                )
                .await
            }
            Attempt::Probe(probe) => {
                self.sb()
                    .write_file("/tmp/.benchflow-probe.sh", probe.script.as_bytes(), 0o755)
                    .await
                    .map_err(|e| sandbox_problem(Phase::Agent, e))?;
                let t = Some(timeout.map_or(PROBE_TIMEOUT, |t| t.min(PROBE_TIMEOUT)));
                self.run_script(
                    "probe",
                    &format!("probe:{}", probe.id),
                    &shell::script_file("/tmp/.benchflow-probe.sh"),
                    stage_env,
                    t,
                    &prompt,
                )
                .await
            }
            Attempt::Model(agent) => {
                let agent_sb = crate::confine::AgentSandbox::new(self.sb());
                let mut env = task.sandbox.env.clone();
                env.extend(self.rt.options.agent_env.clone());
                env.extend(stage_env);
                self.skills_env(&mut env);
                let ctx = AgentContext {
                    sandbox: &agent_sb,
                    user: &self.user,
                    home: &self.home,
                    workdir: &self.workdir,
                    instruction: &prompt,
                    stages: Vec::new(),
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
                    earlier_turns,
                };
                let out = agent.run(ctx).await;
                let problem = out.problem.clone();
                let a = &mut self.agent;
                a.tool_calls += out.tool_calls;
                a.usage.add(&out.usage);
                let partial_before = a.trajectory_summary["partial_trajectory"] == true;
                a.trajectory_summary = out.trajectory_summary.clone();
                if partial_before && a.trajectory_summary.is_object() {
                    a.trajectory_summary["partial_trajectory"] = json!(true);
                }
                a.timed_out = out.timed_out;
                a.env_keys = out.env_keys.clone();
                a.login = out.login.clone().or(a.login.take());
                a.session_id = out.session_id.clone();
                a.resumed = out.resumed;
                a.resume_note = out.resume_note.clone();
                a.model = out.model.clone().or(a.model.take());
                a.agent_version = out.agent_version.clone().or(a.agent_version.take());
                match problem {
                    Some(p) => Err(p),
                    None => Ok(()),
                }
            }
        };
        self.stage_own_timeout |= self.agent.timed_out;
        self.agent.timed_out |= earlier_timeout;
        result
    }

    /// Hand off, save outputs, run hooks, and grade one stage with its own
    /// verifier folders (or the task's).
    async fn grade_stage(&mut self, st: &StepSpec, dir: &Path) -> Result<Score, Problem> {
        let task = self.spec.task.clone();
        self.handoff().await?;
        if !st.outputs.is_empty() {
            self.collect_into(&st.outputs, dir).await?;
        }
        if self.rt.options.save_outputs || !task.sandbox.outputs.is_empty() {
            self.rt.events.phase(self.name(), Phase::Save);
            self.save_outputs().await;
        }
        self.run_hooks().await?;
        if self.spec.dir.join("hooks").is_dir() {
            copy_dir(&self.spec.dir.join("hooks"), &dir.join("hooks"));
        }
        self.rt.events.phase(self.name(), Phase::Verify);
        self.x.verifier_dirs = (!st.verifier_dirs.is_empty()).then(|| st.verifier_dirs.clone());
        self.x.verifier_timeout = st.verifier_timeout;
        let score = self.verify_shared().await;
        self.x.verifier_dirs = None;
        self.x.verifier_timeout = None;
        // The stage's verifier output is the stage's.
        let top = self.spec.dir.join("verifier");
        let mine = dir.join("verifier");
        let _ = std::fs::remove_dir_all(&mine);
        if std::fs::rename(&top, &mine).is_err() {
            copy_dir(&top, &mine);
        }
        score
    }

    /// The verifier folders to upload, in order: the stage's while a stage
    /// grades, else the task's.
    pub(super) fn verifier_dirs(&self) -> Vec<PathBuf> {
        let task = &self.spec.task;
        match &self.x.verifier_dirs {
            Some(dirs) => dirs
                .iter()
                .map(|d| super::in_package(&task.root, d))
                .collect(),
            None => vec![super::in_package(&task.root, &task.verifier.dir)],
        }
    }

    /// The trial's `verifier/`: the combined reward, with the last graded
    /// stage's output beside it.
    fn publish_combined(&self, score: &Score, last: Option<&str>) {
        let top = self.spec.dir.join("verifier");
        let _ = std::fs::create_dir_all(&top);
        if let Some(name) = last {
            copy_dir(
                &self.spec.dir.join("steps").join(name).join("verifier"),
                &top,
            );
        }
        let mut values = score.values.clone();
        values.insert("reward".into(), score.reward);
        let _ = std::fs::write(top.join("reward.txt"), format!("{}\n", score.reward));
        let _ = std::fs::write(
            top.join("reward.json"),
            serde_json::to_string_pretty(&values).unwrap_or_default(),
        );
    }
}

/// The trial's reward from its graded stages' rewards: each key's mean, or
/// the last stage's; a stage never reached counts as 0 or is left out.
/// The reward of a trial whose agent failed in a stage, by the task's
/// rule over its `step_results` rows: each graded stage reached (carried
/// from a fork's parent or its own) as graded, the failed stage 0 when it
/// is graded, and each graded stage not reached per `unreached`. `None`
/// without rows.
pub(crate) fn reward_after_agent_failure(steps: &Steps, rows: &Value) -> Option<f64> {
    let rows = rows.as_array().filter(|r| !r.is_empty())?;
    let mut reached: Vec<BTreeMap<String, f64>> = Vec::new();
    let mut unreached = 0;
    for row in rows {
        if row["graded"] != json!(true) {
            continue;
        }
        if row["reached"] != json!(true) {
            unreached += 1;
            continue;
        }
        let values: BTreeMap<String, f64> = row["verifier_result"]["rewards"]
            .as_object()
            .map(|m| {
                m.iter()
                    .filter_map(|(k, v)| v.as_f64().map(|v| (k.clone(), v)))
                    .collect()
            })
            .unwrap_or_else(|| BTreeMap::from([("reward".to_string(), 0.0)]));
        reached.push(values);
    }
    Some(combine(steps, &reached, unreached).reward)
}

fn combine(steps: &Steps, reached: &[BTreeMap<String, f64>], unreached: usize) -> Score {
    let zero = steps.unreached == Unreached::Zero;
    let mut values: BTreeMap<String, f64> = BTreeMap::new();
    match steps.combine {
        Combine::Final => {
            if unreached > 0 && zero {
                for r in reached {
                    for k in r.keys() {
                        values.insert(k.clone(), 0.0);
                    }
                }
            } else if let Some(last) = reached.last() {
                values = last.clone();
            }
        }
        Combine::Mean => {
            let count = reached.len() + if zero { unreached } else { 0 };
            for r in reached {
                for k in r.keys() {
                    values.entry(k.clone()).or_insert(0.0);
                }
            }
            if count > 0 {
                for (k, v) in values.iter_mut() {
                    *v = reached
                        .iter()
                        .map(|r| r.get(k).copied().unwrap_or(0.0))
                        .sum::<f64>()
                        / count as f64;
                }
            }
        }
    }
    let reward = values.get("reward").copied().unwrap_or(0.0);
    values.insert("reward".into(), reward);
    Score {
        reward,
        passed: reward == 1.0,
        values,
        rubric: None,
    }
}

/// Move the stage's agent folder and trajectory into `steps/<name>/`, and
/// add its trajectory events, tagged with the stage, to `events`.
fn archive_attempt(trial: &Path, dir: &Path, stage: &str, events: &mut Vec<Value>) {
    let traj = trial.join("trajectory").join("acp_trajectory.jsonl");
    if let Ok(text) = std::fs::read_to_string(&traj) {
        let _ = std::fs::create_dir_all(dir.join("trajectory"));
        let _ = std::fs::write(dir.join("trajectory").join("acp_trajectory.jsonl"), &text);
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let mut v: Value =
                serde_json::from_str(line).unwrap_or_else(|_| json!({ "raw": line }));
            if let Value::Object(m) = &mut v {
                m.insert("stage".into(), json!(stage));
            }
            events.push(v);
        }
        let _ = std::fs::remove_file(&traj);
    }
    let agent = trial.join("agent");
    if agent.is_dir() {
        // The agent's saved session stays with the trial: the last stage's
        // is the one `bench handoff` resumes.
        let sessions = trial.join(crate::record::SESSIONS_DIR);
        let kept = trial.join(".sessions");
        let moved = sessions.is_dir() && std::fs::rename(&sessions, &kept).is_ok();
        copy_dir(&agent, &dir.join("agent"));
        let _ = std::fs::remove_dir_all(&agent);
        let _ = std::fs::create_dir_all(&agent);
        if moved {
            let _ = std::fs::rename(&kept, &sessions);
        }
    }
}

/// The test-runner files ([`hardening::is_steering_name`]) under `local`,
/// a file or folder on this machine that lands at `at` in the sandbox: the
/// sandbox path, contents, and mode of each. Links are not followed, and
/// `node_modules` and `.git` are skipped, as the baseline's scan skips them.
fn steering_in(local: &Path, at: &str) -> Vec<(String, Vec<u8>, u32)> {
    use std::os::unix::fs::PermissionsExt;
    let mut out = Vec::new();
    let mut todo = vec![(local.to_path_buf(), at.trim_end_matches('/').to_string())];
    while let Some((path, there)) = todo.pop() {
        let Ok(meta) = std::fs::symlink_metadata(&path) else {
            continue;
        };
        if meta.is_dir() {
            let Ok(entries) = std::fs::read_dir(&path) else {
                continue;
            };
            for e in entries.flatten() {
                let name = e.file_name().to_string_lossy().into_owned();
                if name == "node_modules" || name == ".git" || name.contains(['\n', '/']) {
                    continue;
                }
                todo.push((e.path(), format!("{there}/{name}")));
            }
        } else if meta.is_file() {
            // The name it has in the sandbox: a file mounted under another
            // name is that name there.
            let name = there.rsplit('/').next().unwrap_or("").to_string();
            if hardening::is_steering_name(&name)
                && let Ok(bytes) = std::fs::read(&path)
                // As the handoff counts one: only when it names pytest11.
                && (name != "entry_points.txt"
                    || bytes.windows(8).any(|w| w == b"pytest11"))
            {
                out.push((there, bytes, meta.permissions().mode()));
            }
        }
    }
    out.sort();
    out
}

/// The trial's trajectory summary over every stage's events. Each stage's
/// agent summarizes only its own stage; the trial's record counts them all,
/// keeping the last summary's source and whether any stage ended early.
fn staged_summary(events: &[Value], last: &Value) -> Value {
    if events.is_empty() || !last.is_object() {
        return last.clone();
    }
    let parsed: Vec<benchflow_agent::trajectory::Event> = events
        .iter()
        .filter_map(|e| e.as_object().cloned())
        .map(|m| crate::trainer::without_stage(benchflow_agent::trajectory::Event::from_json(m)))
        .collect();
    let summary = benchflow_agent::trajectory::summarize(
        &parsed,
        last["partial_trajectory"] == true,
        last["trajectory_source"].as_str(),
    );
    let mut out = serde_json::to_value(summary).unwrap_or_else(|_| last.clone());
    // Counts come from every stage's events; what the summary does not
    // count (each stage's `stop` and `stops`, a cost note) is kept.
    if let (Some(out), Some(last)) = (out.as_object_mut(), last.as_object()) {
        for (k, v) in last {
            out.entry(k.clone()).or_insert_with(|| v.clone());
        }
    }
    out
}

fn write_lines(dir: &Path, events: &[Value]) {
    let _ = std::fs::create_dir_all(dir);
    let text: String = events
        .iter()
        .map(|e| format!("{}\n", serde_json::to_string(e).unwrap_or_default()))
        .collect();
    let _ = std::fs::write(dir.join("acp_trajectory.jsonl"), text);
}

fn copy_dir(from: &Path, to: &Path) {
    for entry in walkdir::WalkDir::new(from)
        .follow_links(false)
        .into_iter()
        .filter_map(Result::ok)
    {
        let Ok(rel) = entry.path().strip_prefix(from) else {
            continue;
        };
        let dest = to.join(rel);
        if entry.file_type().is_dir() {
            let _ = std::fs::create_dir_all(&dest);
        } else if entry.file_type().is_file() {
            if let Some(p) = dest.parent() {
                let _ = std::fs::create_dir_all(p);
            }
            let _ = std::fs::copy(entry.path(), &dest);
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_staged_summary_counts_every_stage() {
        let events = vec![
            serde_json::json!({"type": "user_message", "text": "a", "stage": "first"}),
            serde_json::json!({"type": "tool_call", "tool_call_id": "c1", "kind": "execute", "title": "t", "status": "completed", "content": [], "stage": "first"}),
            serde_json::json!({"type": "user_message", "text": "b", "stage": "second"}),
            serde_json::json!({"type": "tool_call", "tool_call_id": "c2", "kind": "execute", "title": "t", "status": "completed", "content": [], "stage": "second"}),
            serde_json::json!({"type": "agent_message", "text": "done", "stage": "second"}),
        ];
        // The last stage's own summary: one prompt, one call, one message.
        let last = serde_json::json!({
            "steps": 3, "tool_call_steps": 1, "user_message_steps": 1,
            "agent_message_steps": 1, "agent_thought_steps": 0, "other_steps": 0,
            "event_type_counts": {"user_message": 1, "tool_call": 1, "agent_message": 1},
            "tool_call_status_counts": {"completed": 1},
            "partial_trajectory": true, "trajectory_source": "acp",
        });
        let s = super::staged_summary(&events, &last);
        assert_eq!(s["steps"], 5);
        assert_eq!(s["tool_call_steps"], 2);
        assert_eq!(s["user_message_steps"], 2);
        assert_eq!(s["other_steps"], 0);
        assert_eq!(
            s["event_type_counts"],
            serde_json::json!({"user_message": 2, "tool_call": 2, "agent_message": 1})
        );
        assert_eq!(
            s["tool_call_status_counts"],
            serde_json::json!({"completed": 2})
        );
        assert_eq!(s["partial_trajectory"], true);
        assert_eq!(s["trajectory_source"], "acp");
        // No events (a nop run): the summary stands.
        assert_eq!(super::staged_summary(&[], &last), last);
    }

    /// The test-runner files a stage ships are found by the name they have
    /// in the sandbox: a file mounted under another name counts as that
    /// name, and nested files, links, and `node_modules` are as the scan
    /// sees them. Guards the D7b fix of REVIEW-CORRECT L6.
    #[cfg(unix)]
    #[test]
    fn the_stages_own_test_runner_files_are_found_by_their_sandbox_names() {
        let tmp = tempfile::tempdir().unwrap();
        let d = tmp.path();
        std::fs::write(d.join("conftest_v2.py"), "MOUNTED").unwrap();
        std::fs::create_dir_all(d.join("work/sub")).unwrap();
        std::fs::create_dir_all(d.join("work/node_modules/x")).unwrap();
        std::fs::write(d.join("work/sub/conftest.py"), "S").unwrap();
        std::fs::write(d.join("work/notes.txt"), "N").unwrap();
        std::fs::write(d.join("work/node_modules/x/conftest.py"), "skip").unwrap();
        std::os::unix::fs::symlink(d.join("conftest_v2.py"), d.join("work/pytest.ini")).unwrap();
        // An entry_points.txt counts only when it names pytest11.
        std::fs::create_dir_all(d.join("work/a.dist-info")).unwrap();
        std::fs::create_dir_all(d.join("work/b.dist-info")).unwrap();
        std::fs::write(
            d.join("work/a.dist-info/entry_points.txt"),
            "[pytest11]\nx = y\n",
        )
        .unwrap();
        std::fs::write(
            d.join("work/b.dist-info/entry_points.txt"),
            "[console_scripts]\nx = y\n",
        )
        .unwrap();
        let mounted = steering_in(&d.join("conftest_v2.py"), "/app/conftest.py");
        assert_eq!(mounted.len(), 1);
        assert_eq!(mounted[0].0, "/app/conftest.py");
        let shipped: Vec<String> = steering_in(&d.join("work"), "/app/")
            .into_iter()
            .map(|f| f.0)
            .collect();
        assert_eq!(
            shipped,
            vec![
                "/app/a.dist-info/entry_points.txt".to_string(),
                "/app/sub/conftest.py".to_string()
            ]
        );
    }
    use super::*;

    /// A child whose agent failed in a stage scores by the task's rule: the
    /// stages before it as graded (the parent's, carried, included), the
    /// failed stage 0, and the stages it never reached per `unreached`.
    #[test]
    fn an_agent_failure_in_a_stage_scores_by_the_task_rule() {
        let steps = |combine, unreached| Steps {
            stages: Vec::new(),
            combine,
            unreached,
        };
        let rows = json!([
            {"step_name": "a", "reached": true, "graded": true, "from_parent": true,
             "verifier_result": {"rewards": {"reward": 1.0}}},
            {"step_name": "b", "reached": true, "graded": true, "error": "R204 crashed"},
            {"step_name": "c", "reached": false, "graded": true, "verifier_result": null},
        ]);
        let mean_zero = steps(Combine::Mean, Unreached::Zero);
        assert_eq!(
            reward_after_agent_failure(&mean_zero, &rows),
            Some(1.0 / 3.0)
        );
        let mean_excl = steps(Combine::Mean, Unreached::Exclude);
        assert_eq!(reward_after_agent_failure(&mean_excl, &rows), Some(0.5));
        let last = steps(Combine::Final, Unreached::Exclude);
        assert_eq!(reward_after_agent_failure(&last, &rows), Some(0.0));
        // Failed in an ungraded stage: only the graded ones count.
        let rows = json!([
            {"step_name": "a", "reached": true, "graded": true,
             "verifier_result": {"rewards": {"reward": 1.0}}},
            {"step_name": "b", "reached": true, "graded": false, "error": "R205"},
        ]);
        assert_eq!(reward_after_agent_failure(&mean_zero, &rows), Some(1.0));
        assert_eq!(reward_after_agent_failure(&mean_zero, &json!(null)), None);
    }
    use benchflow_task::extras::StepSpec;

    fn steps(combine: Combine, unreached: Unreached) -> Steps {
        Steps {
            stages: vec![StepSpec::default(); 3],
            combine,
            unreached,
        }
    }

    fn r(pairs: &[(&str, f64)]) -> BTreeMap<String, f64> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    #[test]
    fn mean_counts_unreached_stages_as_zero_or_leaves_them_out() {
        let reached = [r(&[("reward", 1.0)]), r(&[("reward", 0.5), ("style", 1.0)])];
        let s = combine(&steps(Combine::Mean, Unreached::Zero), &reached, 1);
        assert_eq!(s.reward, 0.5);
        assert_eq!(s.values["style"], 1.0 / 3.0);
        let s = combine(&steps(Combine::Mean, Unreached::Exclude), &reached, 1);
        assert_eq!(s.reward, 0.75);
        assert_eq!(s.values["style"], 0.5, "a missing key counts as 0");
    }

    #[test]
    fn final_is_the_last_stage_or_zero_when_it_was_never_reached() {
        let reached = [r(&[("reward", 1.0)]), r(&[("reward", 0.25)])];
        assert_eq!(
            combine(&steps(Combine::Final, Unreached::Exclude), &reached, 1).reward,
            0.25
        );
        assert_eq!(
            combine(&steps(Combine::Final, Unreached::Zero), &reached, 1).reward,
            0.0
        );
        let all = combine(&steps(Combine::Final, Unreached::Zero), &reached, 0);
        assert_eq!(all.reward, 0.25);
        assert!(!all.passed);
    }
}
