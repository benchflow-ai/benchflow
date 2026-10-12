//! Rollout branching: run a trial to a fork point, store the sandbox's
//! filesystem as a checkpoint, and run several children from it, each in a
//! sandbox of its own, each graded by the task's verifier.
//!
//! ```text
//! parent: image → sandbox → setup → checkpoint ─┬─ child n1: sandbox from the checkpoint → attempt → handoff → verify
//!                                                ├─ child n2: …
//!                                                └─ child nK: …
//! ```
//!
//! Forks run at `start` (after setup, before any agent has run), at
//! `prompt:N` (after the parent's N-th prompt), or at `stage:<name>` (after a
//! graded stage; children run the stages after it). The parent sends its
//! checkpoint to the children as soon as it takes it; then it is discarded
//! (0.8's `--parent discard`) or goes on in its own sandbox and is graded
//! (`--parent continue`). The records are BenchFlow 0.8's: `tree.json`,
//! `checkpoints.json`, `branches/<fork>/labels.json`,
//! `branches/<fork>/children/n<k>/` (each a full trial folder plus
//! `observation.json`), and the `branches` block of the trial's
//! `result.json`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use benchflow_sandbox::checkpoint::{self, Checkpoint};
use benchflow_sandbox::{Exec, Image, shell};
use benchflow_task::TaskPackage;
use benchflow_task::package::Isolation;
use chrono::Utc;
use serde_json::{Value, json};

use super::{SETUP_TIMEOUT, Trial, TrialSpec, sandbox_problem};
use crate::agent::Attempt;
use crate::config::RunOptions;
use crate::hardening::{self, Report};
use crate::netpolicy;
use crate::outcome::{DefenseState, Defenses, Owner, Phase, Problem, Status};
use crate::record::{TrialRecord, write_json};

/// Where a rollout forks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ForkPoint {
    /// After setup, before any agent has run.
    Start,
    /// After the parent's N-th prompt (round 2).
    Prompt(u32),
    /// After a named stage (round 2).
    Stage(String),
}

impl ForkPoint {
    pub fn parse(s: &str) -> Result<Self, String> {
        let s = s.trim();
        if s == "start" {
            return Ok(Self::Start);
        }
        if let Some(n) = s.strip_prefix("prompt:") {
            return match n.parse::<u32>() {
                Ok(0) => Ok(Self::Start),
                Ok(n) => Ok(Self::Prompt(n)),
                Err(_) => Err(format!("`prompt:` takes a prompt number, not `{n}`")),
            };
        }
        if let Some(name) = s.strip_prefix("stage:")
            && !name.is_empty()
        {
            return Ok(Self::Stage(name.to_string()));
        }
        Err(format!(
            "--at takes start, prompt:N, or stage:<name>, not `{s}`"
        ))
    }

    pub fn as_str(&self) -> String {
        match self {
            Self::Start => "start".into(),
            Self::Prompt(n) => format!("prompt:{n}"),
            Self::Stage(s) => format!("stage:{s}"),
        }
    }
}

/// One child to run from the checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChildPlan {
    pub label: String,
    /// The child's own prompt; `None`: the task's instruction. `@instruction`
    /// in it stands for the task's instruction.
    pub prompt: Option<String>,
}

impl ChildPlan {
    /// Parse `label=a[,prompt=…|prompt-file=…]` (0.8's `--child`). A prompt
    /// runs to the end of the value, commas included.
    pub fn parse(s: &str) -> Result<Self, String> {
        let mut label = None;
        let mut prompt = None;
        let mut rest = s.trim();
        while !rest.is_empty() {
            let (key, value_and_rest) = rest
                .split_once('=')
                .ok_or_else(|| format!("--child takes key=value pairs, not `{rest}`"))?;
            match key.trim() {
                "prompt" => {
                    prompt = Some(value_and_rest.to_string());
                    break;
                }
                "prompt-file" | "prompt_file" => {
                    let path = value_and_rest.trim();
                    prompt = Some(
                        std::fs::read_to_string(path)
                            .map_err(|e| format!("--child prompt-file {path}: {e}"))?,
                    );
                    break;
                }
                "label" => {
                    let (v, r) = value_and_rest
                        .split_once(',')
                        .unwrap_or((value_and_rest, ""));
                    label = Some(v.trim().to_string());
                    rest = r.trim();
                }
                "parent" => {
                    return Err(
                        "nested forks (`parent=` in --child) are not in the Rust CLI yet".into(),
                    );
                }
                other => {
                    return Err(format!(
                        "--child takes label=, prompt=, and prompt-file=, not `{other}=`"
                    ));
                }
            }
        }
        let label = label
            .filter(|l| !l.is_empty())
            .ok_or_else(|| "--child needs label=<name>".to_string())?;
        if !label
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
        {
            return Err(format!(
                "a child's label takes letters, digits, `-`, `_`, and `.`, not `{label}`"
            ));
        }
        Ok(Self { label, prompt })
    }
}

/// A fork: where, which children, what to keep.
#[derive(Debug, Clone)]
pub struct ForkPlan {
    pub at: ForkPoint,
    pub children: Vec<ChildPlan>,
    /// Keep the checkpoint after the fork (`--retain-checkpoints`).
    pub retain: bool,
    /// Children resume the parent's agent session (`--resume-session`,
    /// prompt forks only); otherwise each starts a fresh session in the
    /// parent's files.
    pub resume: bool,
    /// Children at once.
    pub concurrency: usize,
    /// The parent goes on after its fork and is graded too (`--parent
    /// continue`); otherwise it is discarded at the checkpoint.
    pub continue_parent: bool,
}

impl ForkPlan {
    /// `--children K`: K children asked the task's own instruction,
    /// labelled `c1`…`cK`.
    pub fn same_prompt(k: usize) -> Vec<ChildPlan> {
        (1..=k)
            .map(|i| ChildPlan {
                label: format!("c{i}"),
                prompt: None,
            })
            .collect()
    }
}

/// Why this fork cannot run, before any sandbox starts. `caps` are the
/// backend's.
pub fn refusals(
    task: &TaskPackage,
    attempt: &Attempt,
    plan: &ForkPlan,
    options: &RunOptions,
    caps: &benchflow_sandbox::Capabilities,
    backend: &str,
) -> Vec<String> {
    let mut out = Vec::new();
    match &plan.at {
        ForkPoint::Start => {
            if plan.resume {
                out.push("--resume-session at `start`: there is no session yet to resume".into());
            }
        }
        ForkPoint::Prompt(n) => {
            match attempt {
                // A checkpoint after the agent ran must leave its login
                // behind; only agents that say what it wrote can.
                Attempt::Model(m) if !m.knows_login_material() => out.push(format!(
                    "{} does not say which files and values its login leaves in the sandbox, so a checkpoint after its prompts could keep them (prompt:N forks take ACP agents)",
                    m.name()
                )),
                Attempt::Model(_) if !plan.resume => {}
                Attempt::Model(m) if m.resumes_over_acp() => {}
                Attempt::Model(m) => out.push(format!(
                    "{} cannot resume a session over ACP (`session/load`), which children at prompt:{n} need",
                    m.name()
                )),
                other => out.push(format!(
                    "{} has no session for children to resume (prompt:N forks need a model agent over ACP)",
                    other.label()
                )),
            }
            // The parent's prompts: the instruction, then each stage's (and
            // tasks with stages are refused below).
            let prompts = 1 + task.stages.len() as u32;
            if *n > prompts {
                out.push(format!(
                    "prompt:{n} is past the parent's last prompt ({prompts})"
                ));
            }
            // A child with no prompt of its own runs the parent's remaining
            // prompts, or the task's instruction again after the last one
            // (0.8's rule; tasks with stages are refused, so it is the
            // instruction).
            if task.sandbox.ready.is_some() {
                out.push(
                    "the task has a readiness check: services it waits for die at the checkpoint, and nothing restarts them in a child"
                        .into(),
                );
            }
        }
        ForkPoint::Stage(name) => stage_refusals(task, attempt, plan, name, &mut out),
    }
    if plan.children.len() < 2 {
        // As 0.8: a group of one has no advantage and no spread.
        out.push("a fork needs at least two children (a group of one has no advantage)".into());
    }
    let mut seen = std::collections::BTreeSet::new();
    for c in &plan.children {
        if !seen.insert(c.label.as_str()) {
            out.push(format!("two children are labelled `{}`", c.label));
        }
    }
    if !caps.checkpoints {
        out.push(format!("the {backend} backend cannot take checkpoints"));
    }
    if options.runtime.is_some() {
        out.push(format!(
            "--runtime {}: no live test yet shows a parent's files reaching a child under it",
            options.runtime.as_deref().unwrap_or_default()
        ));
    }
    if !task.extras.services.is_empty() {
        out.push("the task has sidecar services, which a checkpoint does not hold".into());
    }
    // Graded stages fork at `stage:<name>`, after a graded stage; a
    // `prompt:N` count across them is not supported.
    if matches!(plan.at, ForkPoint::Prompt(_)) {
        if task.extras.steps.is_some() {
            out.push("the task has graded stages: fork after one with `--at stage:<name>`".into());
        } else if !task.stages.is_empty() {
            out.push(
                "the task sends more than one prompt, and prompt:N forks take tasks with one"
                    .into(),
            );
        }
    }
    if task.verifier.isolation == Isolation::Separate {
        out.push("the task's verifier runs in a sandbox of its own".into());
    }
    let user = task
        .agent
        .user
        .clone()
        .unwrap_or_else(|| options.sandbox_user.clone());
    if user == "root" || user == "0" {
        out.push(
            "the agent runs as root, so nothing keeps it out of BenchFlow's state before the checkpoint"
                .into(),
        );
    }
    if matches!(attempt, Attempt::Probe(_) | Attempt::Control(_)) {
        out.push(format!("{} attempts do not branch", attempt.label()));
    }
    if matches!(attempt, Attempt::Oracle | Attempt::Nop | Attempt::Script(_))
        && plan.children.iter().any(|c| c.prompt.is_some())
    {
        out.push(format!(
            "{} takes no prompt, so a child's own prompt would not reach it",
            attempt.label()
        ));
    }
    if options.keep != crate::config::Keep::Never {
        out.push("--keep holds sandboxes, which a fork's cleanup would leave behind".into());
    }
    out
}

/// Why a fork after stage `name` cannot run: the stage must exist, be
/// graded (its handoff stopped every agent process and checked the
/// sandbox), and not be the last; nothing a stage up to it started may be
/// needed later (services die at the checkpoint, and no stage's setup runs
/// again in a child); a model agent must say what its login leaves.
fn stage_refusals(
    task: &TaskPackage,
    attempt: &Attempt,
    plan: &ForkPlan,
    name: &str,
    out: &mut Vec<String>,
) {
    let Some(steps) = &task.extras.steps else {
        out.push(format!(
            "stage:{name}: the task has no graded stages (fork at `start` or `prompt:N`)"
        ));
        return;
    };
    let names: Vec<&str> = steps.stages.iter().map(|s| s.name.as_str()).collect();
    let Some(k) = names.iter().position(|n| *n == name) else {
        out.push(format!(
            "the task has no stage `{name}` (its stages: {})",
            names.join(", ")
        ));
        return;
    };
    let st = &steps.stages[k];
    if !st.graded {
        out.push(format!(
            "stage `{name}` is not graded: forks follow a graded stage, whose handoff stopped the agent and checked the sandbox"
        ));
    }
    if k + 1 == steps.stages.len() {
        out.push(format!(
            "stage `{name}` is the last, so its children would have nothing to run"
        ));
    }
    if task.sandbox.ready.is_some() {
        out.push(
            "the task has a readiness check: services it waits for die at the checkpoint, and nothing restarts them in a child"
                .into(),
        );
    }
    for st in &steps.stages[..=k] {
        let setup = st
            .workdir
            .as_ref()
            .is_some_and(|w| super::in_package(&task.root, w).join("setup.sh").is_file());
        if st.ready.is_some() || setup {
            out.push(format!(
                "stage `{}` has {}: what it started dies at the checkpoint, and a child never runs it again",
                st.name,
                if setup { "a setup.sh" } else { "a readiness check" }
            ));
        }
    }
    match attempt {
        Attempt::Model(m) if !m.knows_login_material() => out.push(format!(
            "{} does not say which files and values its login leaves in the sandbox, so a checkpoint after its stages could keep them",
            m.name()
        )),
        Attempt::Model(m) if plan.resume && !m.resumes_over_acp() => out.push(format!(
            "{} cannot resume a session over ACP (`session/load`), which children after stage `{name}` need",
            m.name()
        )),
        Attempt::Model(_) => {}
        other if plan.resume => out.push(format!(
            "{} has no session for children to resume",
            other.label()
        )),
        _ => {}
    }
}

/// What the parent leaves its children: the checkpoint and what setup
/// learned, which the children's sandboxes hold on disk but the trial
/// keeps in memory.
#[derive(Debug, Clone)]
pub(super) struct ParentState {
    pub checkpoint: Checkpoint,
    pub user: String,
    pub uid: String,
    pub home: String,
    pub workdir: String,
    pub image_path: Option<String>,
    pub net: Option<netpolicy::Plan>,
    pub defenses: Defenses,
    pub locked: Vec<String>,
    /// Processes of the agent user killed before the checkpoint.
    pub killed: Vec<String>,
    /// The parent's agent session (prompt forks).
    pub session_id: Option<String>,
    /// Children resume `session_id` (`--resume-session`).
    pub resume: bool,
    /// Prompts the parent sent before the checkpoint.
    pub after_prompt: u32,
    /// Events in the parent's trajectory at the checkpoint.
    pub trajectory_events: usize,
    /// Login files that were there and were removed before the checkpoint.
    pub login_files_removed: usize,
    /// Login files root left in place: a folder on their way was a link, or
    /// `rm -f` could not remove them (the scan found none of the login's
    /// values in them).
    pub login_files_left: Vec<String>,
    /// The parent took the test-runner baseline, so the checkpoint holds it.
    pub baseline: bool,
    /// At `stage:<name>`: the stages the parent ran, which its children go
    /// on from.
    pub stage: Option<StageCarry>,
}

/// What the parent of a `stage:<name>` fork hands its children: its stages
/// up to the fork, graded, and their records.
#[derive(Debug, Clone)]
pub(super) struct StageCarry {
    /// The fork stage's index; children start at the next one.
    pub index: usize,
    /// `step_results` rows of the parent's stages, the fork's included.
    pub rows: Vec<Value>,
    /// Their graded values, in order, for the task's rule.
    pub graded: Vec<BTreeMap<String, f64>>,
    pub prompts: Vec<String>,
    pub last_graded: Option<String>,
    /// An agent's turn ran out of time in one of these stages (a task's
    /// `on_timeout = "fail"` then zeroes the child, as it would the run).
    pub timed_out: bool,
}

/// A trial's part in a fork.
#[derive(Clone)]
pub(super) enum Role {
    /// Run to the fork point, take the checkpoint, and send it to the
    /// children (who start at once); then stop, or go on (`continues`).
    Parent {
        out: Arc<Mutex<Option<tokio::sync::oneshot::Sender<ParentState>>>>,
        retain: bool,
        at: ForkPoint,
        resume: bool,
        continues: bool,
        /// A refusal at the fork point that left a continuing parent
        /// gradeable: the parent goes on, and the fork records this.
        refused: Arc<Mutex<Option<Problem>>>,
    },
    /// Start from the parent's checkpoint.
    Child {
        state: Arc<ParentState>,
        prompt: Option<String>,
    },
}

/// A child's networks: the parent's, as resolved before its agent ran.
pub(super) fn parent_net(fork: Option<&Role>) -> Option<netpolicy::Plan> {
    match fork {
        Some(Role::Child { state, .. }) => state.net.clone(),
        _ => None,
    }
}

impl Trial<'_> {
    pub(super) fn is_child(&self) -> bool {
        matches!(self.fork, Some(Role::Child { .. }))
    }

    pub(super) fn is_parent(&self) -> bool {
        matches!(self.fork, Some(Role::Parent { .. }))
    }

    /// A parent that forks before its agent runs.
    pub(super) fn fork_at_start(&self) -> bool {
        matches!(
            self.fork,
            Some(Role::Parent {
                at: ForkPoint::Start,
                ..
            })
        )
    }

    /// A child whose parent's agent already worked in the sandbox it comes
    /// from: what that agent did to its home is the agent's own.
    pub(super) fn child_after_agent(&self) -> bool {
        matches!(&self.fork, Some(Role::Child { state, .. }) if state.after_prompt > 0)
    }

    /// A parent that forks after its prompts (`prompt:N`).
    pub(super) fn fork_at_prompt(&self) -> bool {
        matches!(
            self.fork,
            Some(Role::Parent {
                at: ForkPoint::Prompt(_),
                ..
            })
        )
    }

    /// The stage after which this parent forks (`stage:<name>`).
    pub(super) fn fork_stage(&self) -> Option<String> {
        match &self.fork {
            Some(Role::Parent {
                at: ForkPoint::Stage(name),
                ..
            }) => Some(name.clone()),
            _ => None,
        }
    }

    /// A parent that goes on after its fork (`--parent continue`).
    pub(super) fn parent_continues(&self) -> bool {
        matches!(
            self.fork,
            Some(Role::Parent {
                continues: true,
                ..
            })
        )
    }

    /// The stages a child of a `stage:<name>` fork goes on from.
    pub(super) fn child_stage_carry(&self) -> Option<StageCarry> {
        match &self.fork {
            Some(Role::Child { state, .. }) => state.stage.clone(),
            _ => None,
        }
    }

    /// A parent whose children will resume its session.
    pub(super) fn parent_resumes(&self) -> bool {
        matches!(self.fork, Some(Role::Parent { resume: true, .. }))
    }

    /// The session a child at `prompt:N` resumes.
    pub(super) fn child_session(&self) -> Option<String> {
        match &self.fork {
            Some(Role::Child { state, .. }) if state.resume => state.session_id.clone(),
            _ => None,
        }
    }

    /// A child's image: the checkpoint (the backend starts from it).
    pub(super) fn checkpoint_image(&self) -> Option<Image> {
        match &self.fork {
            Some(Role::Child { state, .. }) => Some(Image {
                reference: state.checkpoint.reference.clone(),
                fresh: false,
                duration: std::time::Duration::ZERO,
            }),
            _ => None,
        }
    }

    pub(super) fn child_checkpoint(&self) -> Option<Checkpoint> {
        match &self.fork {
            Some(Role::Child { state, .. }) => Some(state.checkpoint.clone()),
            _ => None,
        }
    }

    /// A child's own prompt, with `@instruction` replaced by `instruction`
    /// (the task's, or at `stage:<name>` the next stage's prompt).
    pub(super) fn child_prompt(&self, instruction: &str) -> Option<String> {
        match &self.fork {
            Some(Role::Child {
                prompt: Some(p), ..
            }) => Some(p.replace("@instruction", instruction)),
            _ => None,
        }
    }

    /// A child's own prompt in place of the task's instruction, with the
    /// task's `prompt_prefix` (`[agent]`) first, as 0.8 prefixes every
    /// child's prompt so each arm sees the same harness policy;
    /// `@instruction` is the instruction without the prefix.
    pub(super) fn child_task_prompt(&self, instruction: &str) -> Option<String> {
        let Some(Role::Child {
            prompt: Some(own), ..
        }) = &self.fork
        else {
            return None;
        };
        Some(compose_child_prompt(
            own,
            instruction,
            prompt_prefix(self.task()).as_deref(),
        ))
    }

    /// The parent at its fork point: take the checkpoint, then stop the
    /// parent ([`discarded`]) or let it go on. A continuing parent whose
    /// checkpoint is refused for a reason that leaves it gradeable goes on
    /// too (graded as a plain trial; the fork is recorded as failed with the
    /// refusal). It stops for a changed state folder (R314) or processes
    /// that survived the kill (R305), which void its own grade as well.
    pub(super) async fn at_fork_point(&mut self, stage: Option<StageCarry>) -> Result<(), Problem> {
        match self.take_checkpoint(stage).await {
            Ok(()) if self.parent_continues() => Ok(()),
            Ok(()) => Err(discarded(self.phase)),
            Err(p) if self.parent_continues() && !matches!(p.code.as_str(), "R314" | "R305") => {
                self.rt.events.log(
                    self.name(),
                    format!("{}; the parent goes on and is graded", p.message),
                );
                if let Some(Role::Parent { refused, .. }) = &self.fork {
                    *refused.lock().unwrap_or_else(|e| e.into_inner()) = Some(p);
                }
                Ok(())
            }
            Err(p) => Err(p),
        }
    }

    /// The parent at its fork point: stop the agent user's processes,
    /// remove and scan for secrets, store the checkpoint, and send it to the
    /// children. The caller then stops the parent ([`discarded`]) or lets it
    /// go on. `stage`: what a `stage:<name>` fork's children go on from.
    pub(super) async fn take_checkpoint(
        &mut self,
        stage: Option<StageCarry>,
    ) -> Result<(), Problem> {
        let Some(Role::Parent {
            out,
            retain,
            at,
            resume,
            ..
        }) = self.fork.clone()
        else {
            unreachable!("only a parent takes the checkpoint");
        };
        let refuse_by = |owner: Owner, msg: String| {
            Problem::new(
                "R401",
                owner,
                Phase::Setup,
                format!("the fork was refused: {msg}"),
                "run it with `bench run` instead, or change what the refusal names",
            )
        };
        let refuse = |msg: String| refuse_by(Owner::You, msg);
        // Refused when the agent user's separation from root does not hold:
        // the kill below is what keeps the checkpoint clean.
        match self.defenses.get("agent_user").map(|d| d.state) {
            Some(DefenseState::Enforced) => {}
            other => {
                // The task's image decides whether it holds.
                return Err(refuse_by(
                    Owner::Task,
                    format!(
                        "the agent user's separation from root is not enforced ({})",
                        other.map_or("unknown".into(), |s| format!("{s:?}").to_lowercase())
                    ),
                ));
            }
        }
        // After prompts, the children resume the agent's session: it must
        // have one.
        let stage_fork = stage.is_some();
        // Prompts sent so far: at `stage:<name>`, one per stage run.
        let after_prompt = match (&at, &stage) {
            (ForkPoint::Prompt(n), _) => *n,
            (ForkPoint::Stage(_), Some(c)) => c.prompts.len() as u32,
            _ => 0,
        };
        // A turn cut off by the time limit is not "after prompt N". After
        // a graded stage it is: the stage's state is what was graded.
        if matches!(at, ForkPoint::Prompt(_)) && self.agent.timed_out {
            return Err(refuse_by(
                Owner::Agent,
                format!(
                    "the agent ran out of time during its prompts, so there is no state after prompt {after_prompt} to fork from"
                ),
            ));
        }
        let session_id = self.agent.session_id.clone();
        if resume && session_id.is_none() {
            return Err(refuse(
                "the agent reported no session id, so its children could not resume its session"
                    .into(),
            ));
        }
        // What the agent's login left: its files go, its values must not
        // be found anywhere.
        let (login_files, login_secrets) = match (&self.spec.attempt, after_prompt) {
            (Attempt::Model(m), n) if n > 0 => m.login_material(&self.home),
            _ => (Vec::new(), Vec::new()),
        };
        // 1. Stop the agent user's processes, and say which: the shared
        // kill, which reads /proc with shell builtins (no procps needed) and
        // fails when any process survives.
        let user = self.user.clone();
        let out_k = self
            .root(
                format!("{}\n{}", list_processes(&user), hardening::kill_user(&user)),
                SETUP_TIMEOUT,
            )
            .await
            .map_err(|e| sandbox_problem(Phase::Setup, e))?;
        let report = Report::parse(&out_k.stdout);
        if let Some(survivors) = report.first("survivors") {
            // As before grading: nothing about the sandbox can be trusted.
            return Err(Problem::new(
                "R305",
                Owner::Sandbox,
                Phase::Setup,
                format!(
                    "the fork was refused: processes of {user} survived the kill before the checkpoint ({survivors}); the world is not safe to copy"
                ),
                "retry; if it repeats for this task, report it at https://github.com/benchflow-ai/benchflow/issues with the trial folder",
            ));
        }
        if !out_k.success() || report.first("killed").is_none() {
            return Err(super::unreported("the kill before the checkpoint", &out_k));
        }
        let uid = report.first("uid").unwrap_or("").to_string();
        let killed: Vec<String> = report.all("proc").iter().map(|s| s.to_string()).collect();
        // Then, with no process of the agent user left to swap a folder
        // after the check: BenchFlow's tars go, and root removes the login
        // files, never through a link. A login file with a link among the
        // folders on its way (the agent's, planted during its turn) stays
        // where it is, named, and the scan below refuses the fork if a value
        // of the login is still anywhere. A child's launch writes the files
        // again as the agent user (`place.rs`), so a link there gains the
        // agent nothing; round 1 refused every such fork, from when a
        // child's launch wrote the files as root. Last, the state folder
        // the children keep (setup never runs again in them) must still be
        // root's alone.
        let mut removes = login_removal(&self.home, &login_files);
        if stage_fork {
            // The fork stage's verifier and reference files, and what the
            // verifier wrote, go before the scan and the commit (a child
            // removes them before its agent anyway; a kept checkpoint would
            // keep them).
            let task = self.task();
            let mut gone = vec![shell::quote(&task.verifier.mount)];
            if let Some(o) = &task.oracle {
                gone.push(shell::quote(&o.mount));
            }
            removes.push_str(&format!(
                "rm -rf {} /logs/verifier; mkdir -p /logs/verifier\n",
                gone.join(" ")
            ));
        }
        let out_r = self
            .root(
                format!(
                    "rm -f {sessions} {outputs}\n{removes}{check}",
                    sessions = hardening::SESSIONS_TAR,
                    outputs = hardening::OUTPUTS_TAR,
                    check = hardening::state_check(),
                ),
                SETUP_TIMEOUT,
            )
            .await
            .map_err(|e| sandbox_problem(Phase::Setup, e))?;
        let report = Report::parse(&out_r.stdout);
        if !reported_last(&out_r.stdout, "BF state-checked") {
            return Err(super::unreported(
                "the removal of the login before the checkpoint",
                &out_r,
            ));
        }
        let left: Vec<String> = report.all("left").iter().map(|s| s.to_string()).collect();
        let removed = report.all("removed").len();
        if let Some(p) = state_problem(&report, after_prompt > 0, "before the checkpoint") {
            return Err(p);
        }
        // A login or credential file left in place (behind a link, or not
        // removable) may hold a token the agent refreshed during its turn,
        // whose value the search below cannot know: no checkpoint. The link
        // (or what blocks the removal) is the agent's.
        if !left.is_empty() {
            return Err(refuse_by(
                Owner::Agent,
                format!(
                    "login or credential files could not be removed before the checkpoint ({}): a folder on their way is a link, or something blocks their removal, so a checkpoint could keep a credential the search for secret values does not know",
                    left.join(", ")
                ),
            ));
        }
        // 2. Scan the filesystem for the run's secret values.
        // The task's variables whose names look secret (`*_TOKEN`, `*_KEY`,
        // …), as the trial's logs redact them (a value such as
        // `noninteractive` is in image files by design); every `-e` value,
        // since a person passed it for this run; the login's.
        let mut secrets: Vec<String> = crate::agent::env_secrets(&self.task().sandbox.env);
        secrets.extend(self.rt.options.agent_env.values().cloned());
        secrets.extend(login_secrets);
        if stage_fork {
            // After a graded stage, the verifier, its hooks, and the oracle
            // have run too, with their own values.
            let task = self.task();
            secrets.extend(crate::agent::env_secrets(&task.verifier.env));
            if let Some(o) = &task.oracle {
                secrets.extend(crate::agent::env_secrets(&o.env));
            }
        }
        let secrets = scan_patterns(&secrets);
        if !secrets.is_empty() {
            let hits = self.scan_for(&secrets).await?;
            if !hits.is_empty() {
                return Err(refuse(format!(
                    "a value of the run's environment is in the sandbox's files ({}), so the checkpoint would keep it",
                    hits.join(", ")
                )));
            }
        }
        // 3. The checkpoint, without the sandbox's own variables.
        let scrub: Vec<String> = self.task().sandbox.env.keys().cloned().collect();
        let name = checkpoint::name(retain);
        self.rt
            .events
            .log(self.name(), format!("taking checkpoint {name}"));
        let started = Instant::now();
        let mut cp = self
            .sb()
            .checkpoint(&name, &scrub)
            .await
            .map_err(|e| match e {
                benchflow_sandbox::SandboxError::Unsupported(m) => refuse(m),
                e => sandbox_problem(Phase::Setup, e),
            })?;
        if cp.seconds == 0.0 {
            cp.seconds = started.elapsed().as_secs_f64();
        }
        let sent = out.lock().unwrap_or_else(|e| e.into_inner()).take();
        let state = ParentState {
            checkpoint: cp,
            user,
            uid,
            home: self.home.clone(),
            workdir: self.workdir.clone(),
            image_path: self.image_path.clone(),
            net: self.net.clone(),
            defenses: self.defenses.clone(),
            locked: self.custody.locked.clone(),
            killed,
            session_id,
            resume,
            after_prompt,
            trajectory_events: trajectory_events(&self.spec.dir)
                + stage_trajectory_events(&self.spec.dir),
            login_files_removed: removed,
            login_files_left: left,
            baseline: self
                .defenses
                .get("planted_file_cleanup")
                .is_some_and(|d| d.state == DefenseState::Enforced),
            stage,
        };
        if let Some(tx) = sent {
            // The fork's runner waits on the other end; if it is gone, so is
            // the run.
            let _ = tx.send(state);
        }
        // The checkpoint's processes were killed: a parent that goes on
        // starts its agent again (`prompt:N` grades it; `stage:` and
        // `start` run what is left).
        Ok(())
    }

    /// Paths holding any of `secrets`, searched as root over the sandbox's
    /// own filesystem. The values travel in a root-only file, removed after.
    async fn scan_for(&self, secrets: &[String]) -> Result<Vec<String>, Problem> {
        let list = format!("{}/scan-values", hardening::STATE_DIR);
        let mut body = secrets.join("\n");
        body.push('\n');
        self.sb()
            .write_file(&list, body.as_bytes(), 0o600)
            .await
            .map_err(|e| sandbox_problem(Phase::Setup, e))?;
        let l = shell::quote(&list);
        // Files the agent user owns are searched whatever their size: an
        // agent could pad a file past the cap to hide a value in it.
        let u = shell::quote(&self.user);
        let out = self
            .root(
                format!(
                    "if command -v find >/dev/null && command -v xargs >/dev/null \
                        && command -v head >/dev/null && command -v sed >/dev/null \
                        && grep -qF -f {l} {l} 2>/dev/null; then\n\
                     find / -xdev -type f \\( -size -64M -o -user {u} \\) ! -path {l} ! -path '/proc/*' ! -path '/sys/*' -print0 2>/dev/null \
                     | xargs -0 -r grep -lF -f {l} 2>/dev/null | head -20 | sed 's/^/BF hit /'\n\
                     printf 'BF scanned 1\\n'\n\
                     fi\n\
                     rm -f {l}; test ! -e {l} && printf 'BF done 1\\n'"
                ),
                std::time::Duration::from_secs(600),
            )
            .await
            .map_err(|e| sandbox_problem(Phase::Setup, e))?;
        let report = Report::parse(&out.stdout);
        if report.first("done").is_none() {
            return Err(super::unreported(
                "the secret scan before the checkpoint",
                &out,
            ));
        }
        // grep found its own list: the search works. Without that, or
        // without find and xargs, nothing was searched.
        if report.first("scanned").is_none() {
            return Err(Problem::new(
                "R401",
                Owner::Task,
                Phase::Setup,
                "the fork was refused: the sandbox could not be searched for the run's secret values (the image lacks find, xargs, grep, head, or sed), so the checkpoint could keep them",
                "add findutils and grep to the task's image, or run the task with `bench run`",
            ));
        }
        Ok(report.all("hit").iter().map(|s| s.to_string()).collect())
    }

    /// A child's setup, in place of the parent's: the sandbox already holds
    /// the agent user, the locked folders, and the pre-agent baselines; what
    /// lives in processes or in memory is set up again.
    pub(super) async fn child_setup(&mut self) -> Result<(), Problem> {
        let Some(Role::Child { state, .. }) = self.fork.clone() else {
            unreachable!("only a child");
        };
        self.user = state.user.clone();
        self.home = state.home.clone();
        self.workdir = state.workdir.clone();
        self.image_path = state.image_path.clone();
        for (k, d) in &state.defenses {
            self.defenses.entry(k.clone()).or_insert_with(|| d.clone());
        }
        self.custody.locked = state.locked.clone();
        // The agent user, as the parent left it.
        let out = self
            .sb()
            .exec(
                Exec::new(format!("id -u {}", shell::quote(&state.user)))
                    .root()
                    .env("PATH", shell::SYSTEM_PATH)
                    .timeout(SETUP_TIMEOUT),
            )
            .await
            .map_err(|e| sandbox_problem(Phase::Setup, e))?;
        if out.stdout.trim() != state.uid {
            return Err(Problem::new(
                "R104",
                Owner::Benchflow,
                Phase::Setup,
                format!(
                    "the child's agent user {} has uid {:?}, not the parent's {}",
                    state.user,
                    out.stdout.trim(),
                    state.uid
                ),
                "please report it with the trial folder",
            ));
        }
        // The state folder came with the checkpoint (setup, which empties
        // it, never runs in a child): it must still be root's alone, with
        // the parent's pre-agent baseline in it, or the handoff would grade
        // against something else.
        let out = self
            .root(hardening::state_check(), SETUP_TIMEOUT)
            .await
            .map_err(|e| sandbox_problem(Phase::Setup, e))?;
        if !reported_last(&out.stdout, "BF state-checked") {
            return Err(super::unreported(
                "the check of the checkpoint's state folder",
                &out,
            ));
        }
        let report = Report::parse(&out.stdout);
        let missing_baseline = state.baseline && report.first("state-baseline").is_none();
        if !report.all("state-owner").is_empty()
            || report.first("state-unchecked").is_some()
            || missing_baseline
        {
            let what = if missing_baseline && report.all("state-owner").is_empty() {
                "without the parent's test-runner baseline".to_string()
            } else {
                format!(
                    "with it not root's alone ({})",
                    report.all("state-owner").join(", ")
                )
            };
            return Err(Problem::new(
                "R103",
                Owner::Sandbox,
                Phase::Setup,
                format!(
                    "the child's sandbox came from the checkpoint {what}: BenchFlow's state folder did not survive the checkpoint as the parent left it"
                ),
                "retry; if it repeats on this backend, report it at https://github.com/benchflow-ai/benchflow/issues with the trial folder",
            )
            .detail(out.tail(2000)));
        }
        // What lives in processes: the OOM order and the memory limit.
        let sb = self.sandbox.as_deref().expect("sandbox started");
        let oom = crate::confine::oom_order(sb, &self.user)
            .await
            .map_err(|e| sandbox_problem(Phase::Setup, e))?;
        self.defenses.insert(oom.name.clone(), oom);
        let mem = crate::confine::memory_limit(sb, self.agent_size().1, self.rt.provider.name())
            .await
            .map_err(|e| sandbox_problem(Phase::Setup, e))?;
        self.defenses.insert(mem.name.clone(), mem);
        if let Some(ready) = &self.task().sandbox.ready.clone() {
            self.wait_ready(ready, true).await?;
        }
        Ok(())
    }
}

/// The task's `[agent] prompt_prefix`, as the loader folded it into the
/// instruction (trimmed).
fn prompt_prefix(task: &TaskPackage) -> Option<String> {
    task.config["agent"]["prompt_prefix"]
        .as_str()
        .map(|p| p.trim().to_string())
        .filter(|p| !p.is_empty())
}

/// `own` with `@instruction` replaced by `instruction` less its `prefix`,
/// and the prefix first.
fn compose_child_prompt(own: &str, instruction: &str, prefix: Option<&str>) -> String {
    let Some(prefix) = prefix else {
        return own.replace("@instruction", instruction);
    };
    let bare = instruction
        .strip_prefix(&format!("{prefix}\n\n"))
        .unwrap_or(instruction);
    format!("{prefix}\n\n{}", own.replace("@instruction", bare))
}

/// A discarded parent's end after its fork (R000): not graded, by design.
/// `phase` is where the parent stopped: setup for a fork at the start, the
/// agent's turn for one after a prompt or a stage.
pub(super) fn discarded(phase: Phase) -> Problem {
    Problem::new(
        "R000",
        Owner::You,
        phase,
        "not graded: the parent was discarded after its fork (its children are graded)",
        "see the children in branches/ (`bench branches <job>`)",
    )
}

/// Python's `math.fsum`: a sum without rounding error along the way
/// (Shewchuk's partials), so a fork's value matches 0.8's to the bit.
fn fsum(xs: impl IntoIterator<Item = f64>) -> f64 {
    let mut partials: Vec<f64> = Vec::new();
    for mut x in xs {
        let mut i = 0;
        for j in 0..partials.len() {
            let mut y = partials[j];
            if x.abs() < y.abs() {
                std::mem::swap(&mut x, &mut y);
            }
            let hi = x + y;
            let lo = y - (hi - x);
            if lo != 0.0 {
                partials[i] = lo;
                i += 1;
            }
            x = hi;
        }
        partials.truncate(i);
        partials.push(x);
    }
    // CPython's last step: add the partials from the top, stopping at the
    // first inexact sum, then correct a half-way rounding.
    let mut n = partials.len();
    if n == 0 {
        return 0.0;
    }
    n -= 1;
    let mut hi = partials[n];
    let mut lo = 0.0;
    while n > 0 {
        let x = hi;
        n -= 1;
        let y = partials[n];
        hi = x + y;
        let yr = hi - x;
        lo = y - yr;
        if lo != 0.0 {
            break;
        }
    }
    if n > 0 && ((lo < 0.0 && partials[n - 1] < 0.0) || (lo > 0.0 && partials[n - 1] > 0.0)) {
        let y = lo * 2.0;
        let x = hi + y;
        let yr = x - hi;
        if y == yr {
            hi = x;
        }
    }
    hi
}

/// 0.8's `value_stderr`: sample standard deviation over sqrt(n), rounded
/// to 6 places; `None` below two rewards.
fn stderr(rewards: &[f64]) -> Option<f64> {
    let n = rewards.len();
    if n < 2 {
        return None;
    }
    let mean = fsum(rewards.iter().copied()) / n as f64;
    let var = fsum(rewards.iter().map(|r| (r - mean).powi(2))) / (n - 1) as f64;
    Some(((var / n as f64).sqrt() * 1e6).round() / 1e6)
}

/// The scan's fixed-string patterns: one per line of each value (`grep -f`
/// reads lines, and a value's newline would add an empty pattern that
/// matches every file), lines shorter than the redaction minimum dropped.
fn scan_patterns(values: &[String]) -> Vec<String> {
    let mut out: Vec<String> = values
        .iter()
        .flat_map(|v| v.lines())
        .map(|l| l.trim_end_matches('\r'))
        .filter(|l| l.trim().len() >= benchflow_agent::credentials::MIN_SECRET_LEN)
        .map(str::to_string)
        .collect();
    out.sort();
    out.dedup();
    out
}

/// The folders between `home` and `file`, top-down, where a link would
/// lead root's removal of the file elsewhere: those below `home` for a file
/// in it; every folder below `/` for a file outside it.
fn folders_on_the_way(home: &str, file: &str) -> Vec<String> {
    let home = home.trim_end_matches('/');
    let (mut at, rest) = match file.strip_prefix(&format!("{home}/")) {
        Some(rest) => (home.to_string(), rest),
        None => (String::new(), file.trim_start_matches('/')),
    };
    let parts: Vec<&str> = rest.split('/').filter(|p| !p.is_empty()).collect();
    let mut out = Vec::new();
    for part in &parts[..parts.len().saturating_sub(1)] {
        at = format!("{at}/{part}");
        out.push(at.clone());
    }
    out
}

/// Root's removal of `files`: `BF removed <path>` for each that was there
/// and is gone; `BF left <path>` for each with a link among the folders on
/// its way (not removed through it), or that `rm -f` could not remove (a
/// folder in its place).
fn login_removal(home: &str, files: &[String]) -> String {
    // The home and every folder above it too: a home the agent could
    // replace with a link (one inside its workdir) would lead the removal
    // elsewhere, as one below it would.
    let home = home.trim_end_matches('/');
    let mut above: Vec<String> = Vec::new();
    let mut at = std::path::Path::new(home);
    while at.as_os_str().len() > 1 {
        above.push(at.to_string_lossy().to_string());
        at = at.parent().unwrap_or(std::path::Path::new("/"));
    }
    files
        .iter()
        .map(|f| {
            let mut folders: Vec<String> = Vec::new();
            if f.starts_with(&format!("{home}/")) {
                folders.extend(above.iter().rev().cloned());
            }
            folders.extend(folders_on_the_way(home, f));
            let tests: Vec<String> = folders
                .iter()
                .map(|d| format!("[ -L {} ] || ", shell::quote(d)))
                .collect();
            let f = shell::quote(f);
            format!(
                "if {links}false; then {{ [ -e {f} ] || [ -L {f} ]; }} && printf 'BF left %s\\n' {f}\n\
                 elif [ -e {f} ] || [ -L {f} ]; then rm -f {f} 2>/dev/null\n\
                 \x20 if [ -e {f} ] || [ -L {f} ]; then printf 'BF left %s\\n' {f}; else printf 'BF removed %s\\n' {f}; fi\n\
                 fi\n",
                links = tests.concat()
            )
        })
        .collect()
}

/// The live processes of `user`, by name (`BF proc <name>`, unusual
/// characters shown as `?`), and its uid (`BF uid <n>`), read from /proc
/// with shell builtins.
fn list_processes(user: &str) -> String {
    format!(
        r#"u={u}; uid=$(id -u "$u" 2>/dev/null) && printf 'BF uid %s\n' "$uid"
for e in /proc/[0-9]*; do
  st=; eu=
  if {{ while read -r k a b rest || [ -n "$k" ]; do
          case "$k" in State:) st=$a;; Uid:) eu=$b;; esac; k=
        done < "$e/status"; }} 2>/dev/null && [ -n "$st" ] && [ "$st" != Z ] && [ "$eu" = "$uid" ]; then
    c=; read -r c < "$e/comm" 2>/dev/null
    printf 'BF proc %s\n' "$(printf %s "$c" | tr -c '[:alnum:]._+-' '?')"
  fi
done"#,
        u = shell::quote(user)
    )
}

/// Whether `marker` is the last line `stdout` printed.
fn reported_last(stdout: &str, marker: &str) -> bool {
    stdout.lines().rev().find(|l| !l.trim().is_empty()) == Some(marker)
}

/// What a [`hardening::state_check`] report says is wrong with the state
/// folder, `when` it ran. After the agent's turn (`agent_ran`) a change is
/// the agent's (R314, as the handoff says it); before any agent ran, the
/// image's or a mount's.
fn state_problem(report: &Report, agent_ran: bool, when: &str) -> Option<Problem> {
    let tampered = report.all("state-owner");
    if !tampered.is_empty() {
        let list = tampered.join(", ");
        return Some(if agent_ran {
            Problem::new(
                "R314",
                Owner::Agent,
                // Found at the fork point, after the agent's turn.
                Phase::Agent,
                format!(
                    "BenchFlow's root-only state folder was changed {when} ({list}), so a checkpoint would carry a baseline the agent could have chosen; the fork was not taken"
                ),
                "read the agent's trajectory for what it did to that folder; if nothing it did explains it, report it at https://github.com/benchflow-ai/benchflow/issues with the trial folder",
            )
        } else {
            Problem::new(
                "R104",
                Owner::Task,
                Phase::Setup,
                format!(
                    "BenchFlow's state folder is not root's alone {when} ({list}), before any agent ran: an image or a mount put it there"
                ),
                "keep /var/lib/benchflow out of the task's image and mounts",
            )
        });
    }
    report.first("state-unchecked").map(|why| {
        Problem::new(
            "R401",
            Owner::Task,
            Phase::Setup,
            format!(
                "the fork was refused: BenchFlow's state folder could not be checked {when}{}",
                if why == "no-find" {
                    " (the image has no `find`)"
                } else {
                    " (`find` failed)"
                }
            ),
            "add findutils to the task's image, or run the task with `bench run`",
        )
    })
}

/// A trial's ACP trajectory events, as written (already redacted); lines
/// that are not JSON objects are left out.
fn trajectory_of(trial: &Path) -> Vec<Value> {
    std::fs::read_to_string(trial.join("trajectory").join("acp_trajectory.jsonl"))
        .map(|t| {
            t.lines()
                .filter_map(|l| serde_json::from_str::<Value>(l).ok())
                .filter(Value::is_object)
                .collect()
        })
        .unwrap_or_default()
}

/// The node a fork after prompts hangs off: `p<prompts>`, with the step id
/// of the parent's last event before the fork (`step-<i>-<type>`, the index
/// into the parent's `trajectory/acp_trajectory.jsonl`). `None` at start, or
/// when the parent recorded no event.
fn fork_node(trial: &Path, s: &ParentState) -> Option<(String, String)> {
    if s.after_prompt == 0 || s.trajectory_events == 0 {
        return None;
    }
    let index = s.trajectory_events - 1;
    let kind = std::fs::read_to_string(trial.join("trajectory").join("acp_trajectory.jsonl"))
        .ok()
        .and_then(|t| {
            t.lines()
                .filter(|l| !l.trim().is_empty())
                .nth(index)
                .and_then(|l| serde_json::from_str::<Value>(l).ok())
        })
        .and_then(|e| e["type"].as_str().map(str::to_string))
        .filter(|k| !k.is_empty())
        .unwrap_or_else(|| "event".into());
    Some((
        format!("p{}", s.after_prompt),
        format!("step-{index}-{kind}"),
    ))
}

/// Lines in the trajectories a staged trial archived per stage
/// (`steps/<name>/trajectory/`), which the top-level file no longer holds.
fn stage_trajectory_events(trial: &Path) -> usize {
    std::fs::read_dir(trial.join("steps"))
        .map(|rd| {
            rd.filter_map(Result::ok)
                .map(|e| trajectory_events(&e.path()))
                .sum()
        })
        .unwrap_or(0)
}

/// Lines in a trial's ACP trajectory so far.
fn trajectory_events(trial: &Path) -> usize {
    std::fs::read_to_string(trial.join("trajectory").join("acp_trajectory.jsonl"))
        .map(|t| t.lines().filter(|l| !l.trim().is_empty()).count())
        .unwrap_or(0)
}

/// One child's outcome.
#[derive(Debug, Clone)]
pub struct ChildOutcome {
    pub index: usize,
    pub node_id: String,
    pub label: String,
    pub prompt: Option<String>,
    pub dir: PathBuf,
    pub record: Option<TrialRecord>,
    pub attempts: u32,
    pub retried_after: Option<Problem>,
    /// The parent's session this child resumed (`--resume-session`).
    pub resumes: Option<String>,
    /// In a task with stages, the reward the task's rule gives a child whose
    /// agent failed (its stages before as graded, the failed one 0).
    pub stage_rule_reward: Option<f64>,
}

impl ChildOutcome {
    /// 0.8's child status and reward: an agent's own failure is a scored 0;
    /// only infrastructure leaves a child unscored.
    fn status(&self) -> (&'static str, Option<f64>, Option<&'static str>) {
        let Some(r) = &self.record else {
            return ("not_started", None, None);
        };
        // A child that was to resume the parent's session and did not (its
        // `session/load` failed) never ran from the fork. After a stage
        // fork, its first own stage is the one that loaded the session
        // (later stages may start fresh ones); otherwise the trial says
        // whether its agent resumed one.
        let first_stage = r.step_results.as_ref().and_then(|rows| {
            rows.as_array()?
                .iter()
                .find(|row| row.get("from_parent").is_none())
                .map(|row| {
                    (row["session"]["resumed"] == json!(true))
                        .then(|| row["session"]["id"].as_str().map(str::to_string))
                        .flatten()
                })
        });
        let loaded = match first_stage {
            Some(id) => id,
            None => r.resumed_session.clone(),
        };
        let resume_failed = self
            .resumes
            .as_ref()
            .is_some_and(|want| loaded.as_deref() != Some(want.as_str()));
        classify(
            r.status,
            r.problem.as_ref(),
            // The child's own time-out: a stage child carries its parent's
            // on its record, and 0.8 fails a child only for its own.
            r.own_timed_out,
            r.reward,
            resume_failed,
            self.stage_rule_reward,
        )
    }
}

/// 0.8's status, reward, and reward source for a child that ran:
/// - a child that was to resume the parent's session and did not (its
///   `session/load` failed) never ran from the fork: `failed`;
/// - graded: `scored` by the verifier, unless its agent ran out of time,
///   which 0.8 counts `failed` (its own result.json keeps the grade);
/// - its agent crashed or failed its turn (R204, R205 in the agent phase):
///   its own doing, `scored` 0, or by the task's stage rule
///   (`stage_rule`);
/// - its agent never installed, never started, went silent, or never
///   reached its model (R201-R203, R206), or infrastructure failed:
///   `failed`; stopped: `cancelled`; no verdict: `unscored`.
fn classify(
    status: Status,
    problem: Option<&Problem>,
    timed_out: bool,
    reward: Option<f64>,
    resume_failed: bool,
    stage_rule: Option<f64>,
) -> (&'static str, Option<f64>, Option<&'static str>) {
    if status == Status::Errored && resume_failed {
        return ("failed", None, None);
    }
    match status {
        Status::Scored if timed_out => ("failed", None, None),
        Status::Scored => ("scored", reward, Some("verifier")),
        Status::Errored
            if problem.is_some_and(|p| {
                p.owner == Owner::Agent
                    && p.phase == Phase::Agent
                    && matches!(p.code.as_str(), "R204" | "R205")
            }) =>
        {
            (
                "scored",
                Some(stage_rule.unwrap_or(0.0)),
                Some("agent_failure"),
            )
        }
        Status::Skipped => ("cancelled", None, None),
        Status::Unscored => ("unscored", None, None),
        Status::Errored => ("failed", None, None),
    }
}

/// What a fork did.
#[derive(Debug, Clone)]
pub struct ForkOutcome {
    pub id: String,
    pub parent: TrialRecord,
    /// Why the checkpoint was refused when a continuing parent went on
    /// without it (the fork's error; the parent's own record is its grade).
    pub refusal: Option<Problem>,
    pub checkpoint: Option<Checkpoint>,
    pub retention: Option<String>,
    pub children: Vec<ChildOutcome>,
    pub status: String,
    pub value: Option<f64>,
    pub value_stderr: Option<f64>,
    pub seconds: f64,
    /// The parent's agent session the children resumed (prompt forks).
    pub session_id: Option<String>,
    /// The parent went on after its fork and was graded (`--parent
    /// continue`).
    pub parent_continued: bool,
    /// 0.8's word for the parent's fate: `skipped` (discarded at its
    /// checkpoint), `not_needed` (it went on in its own sandbox), or
    /// `not_attempted` (it never took its checkpoint, so it ended as a
    /// plain trial).
    pub parent_restore: &'static str,
    /// The parent has not ended yet (only in the records written while
    /// the fork runs): `parent` is a placeholder, with no grade.
    pub parent_running: bool,
}

/// Run the parent, then the children from its checkpoint, then delete the
/// checkpoint (unless kept), and write the records into the parent's
/// trial folder.
pub async fn run(spec: &TrialSpec, rt: &super::Runtime, plan: &ForkPlan) -> ForkOutcome {
    let clock = Instant::now();
    let fork_id = uuid::Uuid::new_v4().simple().to_string();
    // The parent sends its checkpoint when it takes it; the children start
    // then, while a continuing parent goes on. A parent that ends without
    // one drops the sender, and no child runs.
    let (tx, rx) = tokio::sync::oneshot::channel::<ParentState>();
    let refused = Arc::new(Mutex::new(None));
    // What has ended so far, for the records written while the fork runs
    // (0.8 rewrites tree.json at every child's end): a run killed mid-fork
    // leaves readable records of the parent and the children that ended.
    let progress = Progress::default();
    let parent = async {
        let rec = super::run_trial(
            spec,
            rt,
            None,
            Some(Role::Parent {
                out: Arc::new(Mutex::new(Some(tx))),
                retain: plan.retain,
                at: plan.at.clone(),
                resume: plan.resume,
                continues: plan.continue_parent,
                refused: refused.clone(),
            }),
        )
        .await;
        *lock(&progress.parent) = Some(rec.clone());
        progress.write(spec, plan, &fork_id, clock);
        rec
    };
    let planned: Vec<ChildOutcome> = plan
        .children
        .iter()
        .enumerate()
        .map(|(i, c)| ChildOutcome {
            index: i,
            node_id: format!("n{}", i + 1),
            label: c.label.clone(),
            prompt: c.prompt.clone(),
            dir: spec
                .dir
                .join("branches")
                .join(&fork_id)
                .join("children")
                .join(format!("n{}", i + 1)),
            record: None,
            attempts: 0,
            retried_after: None,
            resumes: None,
            stage_rule_reward: None,
        })
        .collect();
    let forked = async {
        let Ok(state) = rx.await else {
            return (None, planned, None);
        };
        let resumes = Some(&state)
            .filter(|s| s.resume)
            .and_then(|s| s.session_id.clone());
        *lock(&progress.state) = Some(state.clone());
        *lock(&progress.children) = planned.clone();
        // The fork exists from here: a run killed before any child ends
        // still leaves its records (every child `not_started`).
        progress.write(spec, plan, &fork_id, clock);
        let shared = Arc::new(state.clone());
        let sem = Arc::new(tokio::sync::Semaphore::new(plan.concurrency.max(1)));
        let progress = &progress;
        let fork_id = &fork_id;
        let runs = planned.into_iter().map(|mut c| {
            let sem = sem.clone();
            let state = shared.clone();
            c.resumes = resumes.clone();
            async move {
                let _permit = sem.acquire().await;
                let done = run_child(spec, rt, state, c).await;
                lock(&progress.children)[done.index] = done.clone();
                progress.write(spec, plan, fork_id, clock);
                done
            }
        });
        let children = futures::future::join_all(runs).await;
        let retention = Some(if plan.retain {
            "kept".to_string()
        } else {
            delete(rt, &state.checkpoint).await
        });
        (Some(state), children, retention)
    };
    let (parent, (state, children, retention)) = futures::future::join(parent, forked).await;
    let scored: Vec<f64> = children
        .iter()
        .filter_map(|c| match c.status() {
            ("scored", r, _) => r,
            _ => None,
        })
        .collect();
    // A stop before the checkpoint cancelled the fork; it did not fail.
    let stopped = parent.problem.as_ref().is_some_and(|p| p.code == "R001") || rt.stopping();
    let status = if state.is_none() && stopped {
        "cancelled"
    } else if state.is_none() {
        "failed"
    } else if children.iter().any(|c| c.status().0 == "cancelled") {
        "cancelled"
    } else if scored.len() == children.len() {
        "completed"
    } else {
        "partial"
    };
    // As 0.8: fsum(r / n), and the standard error of the mean.
    let n = scored.len() as f64;
    let value =
        (status == "completed" && !scored.is_empty()).then(|| fsum(scored.iter().map(|r| r / n)));
    let value_stderr = (status == "completed").then(|| stderr(&scored)).flatten();
    let refusal = refused.lock().unwrap_or_else(|e| e.into_inner()).take();
    let outcome = ForkOutcome {
        id: fork_id,
        parent,
        refusal,
        checkpoint: state.as_ref().map(|s| s.checkpoint.clone()),
        retention,
        children,
        status: status.to_string(),
        value,
        value_stderr,
        seconds: clock.elapsed().as_secs_f64(),
        session_id: state
            .as_ref()
            .filter(|s| s.resume)
            .and_then(|s| s.session_id.clone()),
        parent_continued: plan.continue_parent && state.is_some(),
        parent_restore: match (&state, plan.continue_parent) {
            (None, _) => "not_attempted",
            (Some(_), true) => "not_needed",
            (Some(_), false) => "skipped",
        },
        parent_running: false,
    };
    if let Err(e) = write_records(spec, plan, &outcome, state.as_ref()) {
        rt.events.log(
            &spec.name,
            format!("could not write the fork's records: {e}"),
        );
    }
    outcome
}

async fn run_child(
    parent: &TrialSpec,
    rt: &super::Runtime,
    state: Arc<ParentState>,
    mut c: ChildOutcome,
) -> ChildOutcome {
    let spec = TrialSpec {
        task: parent.task.clone(),
        attempt: parent.attempt.clone(),
        name: format!("{}.{}", parent.name, c.node_id),
        dir: c.dir.clone(),
    };
    loop {
        c.attempts += 1;
        // An earlier attempt's folder is kept beside the child's, as 0.8
        // keeps it (`n1.attempt-1`).
        if c.dir.exists() {
            let kept = c
                .dir
                .with_file_name(format!("{}.attempt-{}", c.node_id, c.attempts - 1));
            let _ = std::fs::rename(&c.dir, kept);
        }
        let rec = super::run_trial(
            &spec,
            rt,
            None,
            Some(Role::Child {
                state: state.clone(),
                prompt: c.prompt.clone(),
            }),
        )
        .await;
        // A child that failed before its agent did anything is retried
        // once, and only for what may pass on a second try (the sandbox
        // backend or the provider), never after a usage limit or a stop: a
        // uid mismatch (BenchFlow's) or a smaller sandbox (refused, R401)
        // would fail the same way again.
        let early = rec.status == Status::Errored
            && rec.problem.as_ref().is_some_and(|p| {
                matches!(p.phase, Phase::Image | Phase::Start | Phase::Setup)
                    && matches!(p.owner, Owner::Sandbox | Owner::Model)
                    && p.code != "R214"
            });
        if early && c.attempts == 1 && !rt.stopping() {
            c.retried_after = rec.problem.clone();
            continue;
        }
        c.stage_rule_reward = spec.task.extras.steps.as_ref().and_then(|steps| {
            rec.step_results
                .as_ref()
                .and_then(|rows| super::stages::reward_after_agent_failure(steps, rows))
        });
        c.record = Some(rec);
        return c;
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// A fork's parts that have ended, shared by the parent's and the
/// children's futures.
#[derive(Default)]
struct Progress {
    parent: Mutex<Option<TrialRecord>>,
    state: Mutex<Option<ParentState>>,
    children: Mutex<Vec<ChildOutcome>>,
}

impl Progress {
    /// Write the fork's records as they stand (status `running`, no value),
    /// once the checkpoint is there. A parent that has not ended yet (a
    /// continuing parent beside its children) is written as running, with
    /// no grade. The records written when the fork ends replace them.
    fn write(&self, spec: &TrialSpec, plan: &ForkPlan, fork_id: &str, clock: Instant) {
        let Some(state) = lock(&self.state).clone() else {
            return;
        };
        let parent = lock(&self.parent).clone();
        let parent_running = parent.is_none();
        let o = ForkOutcome {
            id: fork_id.to_string(),
            parent: parent.unwrap_or_else(|| running_parent(spec)),
            refusal: None,
            checkpoint: Some(state.checkpoint.clone()),
            retention: None,
            children: lock(&self.children).clone(),
            status: "running".into(),
            value: None,
            value_stderr: None,
            seconds: clock.elapsed().as_secs_f64(),
            session_id: Some(&state)
                .filter(|s| s.resume)
                .and_then(|s| s.session_id.clone()),
            parent_continued: plan.continue_parent,
            parent_restore: if plan.continue_parent {
                "not_needed"
            } else {
                "skipped"
            },
            parent_running,
        };
        let _ = write_records(spec, plan, &o, Some(&state));
    }
}

/// The placeholder for a parent that has not ended yet: no grade, no
/// problem, no time. Only the records written while the fork runs hold it.
fn running_parent(spec: &TrialSpec) -> TrialRecord {
    let now = chrono::Local::now();
    TrialRecord {
        task_name: spec.task.name.clone(),
        task_id: spec.task.id.clone(),
        rollout_name: spec.name.clone(),
        agent: spec.attempt.label(),
        model: None,
        status: Status::Unscored,
        rewards: None,
        reward: None,
        passed: None,
        timed_out: false,
        own_timed_out: false,
        problem: None,
        usage: Default::default(),
        tool_calls: 0,
        prompts: Vec::new(),
        trajectory_summary: Value::Null,
        timing: Default::default(),
        started_at: now,
        finished_at: now,
        task_digest: spec.task.digest.clone(),
        sandbox_id: None,
        sandbox_backend: String::new(),
        defenses: Defenses::new(),
        custody: Default::default(),
        login: None,
        agent_version: None,
        env_keys: Vec::new(),
        verifier_exit: None,
        rubric: None,
        dataset: None,
        step_results: None,
        agent_session: None,
        resumed_session: None,
        resources_raised: None,
        environment_warnings: Vec::new(),
        reward_range: None,
    }
}

/// Delete a checkpoint, saying what happened in 0.8's retention words.
async fn delete(rt: &super::Runtime, cp: &Checkpoint) -> String {
    match rt.provider.delete_checkpoint(cp).await {
        Ok(true) => "deleted".into(),
        Ok(false) => "deferred".into(),
        Err(e) => {
            tracing::warn!("could not delete checkpoint {}: {e}", cp.reference);
            "delete_failed".into()
        }
    }
}

/// A continuing parent's own reward (`None` when it was discarded, or not
/// scored).
fn parent_reward(o: &ForkOutcome) -> Option<f64> {
    (o.parent_restore != "skipped" && o.parent.status == Status::Scored)
        .then_some(o.parent.reward)
        .flatten()
}

/// Why a fork that has no checkpoint failed: the refusal a continuing
/// parent went on past, else the parent's own problem.
fn fork_error(o: &ForkOutcome) -> Option<&Problem> {
    o.refusal.as_ref().or(o.parent.problem.as_ref())
}

/// tree.json's `intervention.requested`, as 0.8 words it: never the prompt
/// text (it is in the child's trajectory), but its length and digest; the
/// oracle's script; or the prompts a child without its own runs.
fn requested(spec: &TrialSpec, plan: &ForkPlan, after_prompt: u32, own: Option<&str>) -> String {
    if matches!(spec.attempt, Attempt::Oracle) {
        return "oracle solve.sh".into();
    }
    if let Some(p) = own {
        use sha2::Digest;
        let digest = hex::encode(sha2::Sha256::digest(p.as_bytes()));
        return format!(
            "own prompt ({} characters, sha256:{})",
            p.chars().count(),
            &digest[..12]
        );
    }
    let task = &spec.task;
    let total = task
        .extras
        .steps
        .as_ref()
        .map_or(1 + task.stages.len(), |s| s.stages.len()) as u32;
    let remaining = total.saturating_sub(after_prompt);
    let _ = plan;
    if remaining > 0 {
        format!("parent's remaining prompts ({remaining})")
    } else {
        format!("the task's prompts ({total})")
    }
}

/// The fork's error when the parent ended well but never reached its fork
/// point (a stage's gate stopped it first).
fn not_reached(plan: &ForkPlan) -> Value {
    let _ = plan;
    json!({"type": "not_reached", "code": "not_reached"})
}

fn usage_json(r: &TrialRecord) -> Value {
    let u = &r.usage;
    json!({
        "n_input_tokens": u.input_tokens,
        "n_output_tokens": u.output_tokens,
        "n_cache_read_tokens": u.cache_read_tokens,
        "n_cache_creation_tokens": u.cache_write_tokens,
        "total_tokens": u.input_tokens + u.output_tokens + u.cache_read_tokens + u.cache_write_tokens,
    })
}

fn timing_json(r: &TrialRecord) -> Value {
    json!({
        "agent_setup": round3(r.timing.environment_setup),
        "agent_execution": round3(r.timing.agent_execution),
        "verifier": round3(r.timing.verifier),
    })
}

fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

/// An error as tree.json and checkpoints.json record it: 0.8 keeps them an
/// allowlisted display record, so no message (which can name paths or
/// echo output); the message is in the trial's own result.json, or for a
/// refused checkpoint in `branch-refusal.json` beside tree.json.
fn error_json(p: &Problem) -> Value {
    json!({"type": p.code, "code": p.code})
}

/// 0.8's fork `cost`: the children's tokens and USD (summed only when
/// every child reported one), and sandbox seconds. The parent is
/// discarded at its checkpoint, so its sandbox seconds are its own trial's.
fn fork_cost(o: &ForkOutcome) -> Value {
    let records: Vec<&TrialRecord> = o
        .children
        .iter()
        .filter_map(|c| c.record.as_ref())
        .collect();
    let usds: Vec<Option<f64>> = records.iter().map(|r| r.usage.cost_usd).collect();
    let known =
        !usds.is_empty() && usds.len() == o.children.len() && usds.iter().all(Option::is_some);
    let children_seconds: f64 = records.iter().map(|r| r.timing.total).sum();
    // A parent still running has no time yet.
    let parent = (!o.parent_running).then_some(o.parent.timing.total);
    json!({
        "tokens": records.iter().map(|r| usage_json(r)["total_tokens"].as_u64().unwrap_or(0)).sum::<u64>(),
        "usd": known.then(|| (usds.iter().flatten().sum::<f64>() * 1e6).round() / 1e6),
        "usd_known": known,
        "wall_seconds": round3(o.seconds),
        "parent_sandbox_seconds": parent.map(round3),
        "children_sandbox_seconds": round3(children_seconds),
        "sandbox_seconds": parent.map(|p| round3(p + children_seconds)),
    })
}

/// The fork's documents, in BenchFlow 0.8's shapes.
fn write_records(
    spec: &TrialSpec,
    plan: &ForkPlan,
    o: &ForkOutcome,
    state: Option<&ParentState>,
) -> std::io::Result<()> {
    let dir = &spec.dir;
    std::fs::create_dir_all(dir)?;
    let rel = |p: &Path| {
        p.strip_prefix(dir)
            .map(|r| r.to_string_lossy().replace('\\', "/"))
            .unwrap_or_default()
    };
    let cp = o.checkpoint.as_ref();
    let resumed = state.is_some_and(|s| s.resume && s.session_id.is_some());
    let after_prompt = state.map_or(
        match plan.at {
            ForkPoint::Prompt(n) => n,
            _ => 0,
        },
        |s| s.after_prompt,
    );
    // After prompts, the fork hangs off a node of its own whose step is the
    // parent's last event before it (0.8's `step-<i>-<type>`): 0.8's
    // exporter cuts the shared prefix there. At start, off the root.
    let fork_node = state.and_then(|s| fork_node(dir, s));
    let parent_node = fork_node
        .as_ref()
        .map_or("root".to_string(), |(id, _)| id.clone());
    let mut nodes = vec![json!({"id": "root", "parent": null, "step_id": null})];
    if let Some((id, step)) = &fork_node {
        nodes.push(json!({"id": id, "parent": "root", "step_id": step}));
    }
    let mut tree_children = Vec::new();
    let mut result_nodes = Vec::new();
    let mut labels = Vec::new();
    let (mut usage, mut timing) = (
        BTreeMap::<String, u64>::new(),
        BTreeMap::<String, f64>::new(),
    );
    for c in &o.children {
        let (status, reward, source) = c.status();
        nodes.push(json!({"id": c.node_id, "parent": parent_node, "step_id": null}));
        let available = c.record.is_some() && c.dir.join("result.json").is_file();
        let path = available.then(|| rel(&c.dir));
        let error = c
            .record
            .as_ref()
            .and_then(|r| r.problem.as_ref())
            .filter(|_| status != "scored" || source == Some("agent_failure"))
            .map(error_json)
            .or_else(|| {
                // Graded, but its agent ran out of time: 0.8's failure.
                c.record
                    .as_ref()
                    .filter(|r| status == "failed" && r.own_timed_out)
                    .map(|_| json!({"type": "AgentPromptTimeoutError", "code": "timeout"}))
            });
        let mut child = json!({
            "index": c.index,
            "node_id": c.node_id,
            "status": status,
            "reward": reward,
            "reward_source": source,
            "error": error,
            "cleanup_error": null,
            "intervention": {
                "label": c.label,
                "requested": requested(spec, plan, after_prompt, c.prompt.as_deref()),
                "execution": match (resumed, c.prompt.is_some()) {
                    // 0.8's word for a child the runner drove differently.
                    (_, true) => "runner",
                    (true, false) => "resumed_session",
                    (false, false) => "unspecified",
                },
                "evidence": null,
            },
            "artifacts": {
                "status": if available { "available" } else { "unavailable" },
                "path": path,
            },
            "attempts": c.attempts,
            "retried_after": c.retried_after.as_ref().map(error_json),
        });
        if let Some(r) = &c.record {
            child["usage"] = usage_json(r);
            child["timing_sec"] = timing_json(r);
            child["cost"] = json!({
                "tokens": usage_json(r)["total_tokens"],
                "usd": r.usage.cost_usd,
                "usd_known": r.usage.cost_usd.is_some(),
                "sandbox_seconds": round3(r.timing.total),
            });
            child["snapshot_start"] = json!({"seconds": round3(r.timing.sandbox_start)});
            for (k, v) in usage_json(r).as_object().into_iter().flatten() {
                *usage.entry(k.clone()).or_default() += v.as_u64().unwrap_or(0);
            }
            for (k, v) in timing_json(r).as_object().into_iter().flatten() {
                *timing.entry(k.clone()).or_default() += v.as_f64().unwrap_or(0.0);
            }
            if available {
                write_json(
                    &c.dir.join("observation.json"),
                    &json!({
                        "kind": "branch-child",
                        "node_id": c.node_id,
                        "lineage": {
                            "parent_rollout": spec.name,
                            "fork_id": o.id,
                            "parent_node": parent_node,
                            "index": c.index,
                            "label": c.label,
                        },
                        "reward": reward,
                        // 0.8's observation keeps the error as text.
                        "error": c.record.as_ref().and_then(|r| r.problem.as_ref()).filter(|_| child["error"].is_object()).map(|p| p.message.clone()),
                        "verifier_error": null,
                        "export_error": null,
                        "timing": timing_json(r),
                        "native_usage": usage_json(r),
                        "snapshot_start": child["snapshot_start"],
                        // 0.8's child archive holds the child's own events,
                        // which the viewer shows as its trajectory.
                        "trajectory": trajectory_of(&c.dir),
                    }),
                )?;
            }
        }
        result_nodes.push(json!({
            "index": c.index,
            "node_id": c.node_id,
            "label": c.label,
            "status": status,
            "reward": reward,
            "reward_source": source,
            "path": child["artifacts"]["path"],
        }));
        labels.push(json!({
            "label": c.label,
            "node_id": c.node_id,
            "folder": available.then(|| c.node_id.clone()),
            "status": status,
            "reward": reward,
        }));
        tree_children.push(child);
    }
    let scored = o
        .children
        .iter()
        .filter(|c| c.status().0 == "scored")
        .count();
    let rewards: Vec<f64> = o.children.iter().filter_map(|c| c.status().1).collect();
    let value_scored =
        (!rewards.is_empty()).then(|| rewards.iter().sum::<f64>() / rewards.len() as f64);
    let fork = json!({
        "id": o.id,
        "rollout": spec.name,
        "parent_node": parent_node,
        "fork_point": plan.at.as_str(),
        "requested_children": plan.children.len(),
        "snapshot": {
            "requested_layers": ["sandbox"],
            "captured_layers": if cp.is_some() { json!(["sandbox"]) } else { json!([]) },
            "environment": null,
            "sandbox": cp.map(|c| json!({
                "provider": c.provider,
                "ref": c.reference,
                "digest": null,
                "size_bytes": c.stored_bytes,
            })),
            "excluded": if resumed {
                json!(["process_memory", "mounted_contents", "external_services", "physical_state"])
            } else {
                json!(["agent_session", "process_memory", "mounted_contents", "external_services", "physical_state"])
            },
            "agent_session": if resumed { "resumed" } else { "fresh" },
            "login_files_removed": state.map(|s| s.login_files_removed),
            "login_files_left": state.map(|s| s.login_files_left.clone()).unwrap_or_default(),
            "lifetime": "provider_local",
            "restore_available": o.retention.as_deref() == Some("kept"),
            "retention": o.retention,
            "lossy": state.is_some_and(|s| !s.killed.is_empty()),
            "killed": state.map(|s| s.killed.clone()).unwrap_or_default(),
        },
        "status": o.status,
        // 0.8's words: a discarded parent was not restored; a continuing one
        // never needed it (its children ran in sandboxes of their own).
        "parent_restore": o.parent_restore,
        "children_mode": {"isolated": true, "concurrency": plan.concurrency},
        "timing_sec": {
            "checkpoint": cp.map(|c| round3(c.seconds)),
            "child_restore": o.children.iter().filter_map(|c| c.record.as_ref().map(|r| round3(r.timing.sandbox_start))).collect::<Vec<_>>(),
            "children": round3(o.seconds),
            "parent_restore": null,
        },
        "value": o.value,
        "value_stderr": o.value_stderr,
        "cost": fork_cost(o),
        "value_scored": value_scored,
        "scored": scored,
        "parent_reward": parent_reward(o),
        "error": if o.status == "failed" { Some(fork_error(o).map(error_json).unwrap_or_else(|| not_reached(plan))) } else { None },
        "parent_restore_error": null,
        "artifact_error": null,
        "children": tree_children,
    });
    write_json(
        &dir.join("tree.json"),
        &json!({
            "schema_version": 1,
            "kind": "benchflow-branch-tree",
            "nodes": nodes,
            "forks": [fork],
        }),
    )?;
    if let Some(p) = &o.refusal {
        write_json(&dir.join("branch-refusal.json"), &json!(p))?;
    }
    let fork_dir = dir.join("branches").join(&o.id);
    if fork_dir.is_dir() {
        write_json(
            &fork_dir.join("labels.json"),
            &json!({"kind": "benchflow-branch-labels", "fork_id": o.id, "children": labels}),
        )?;
    }
    // checkpoints.json: the fork's checkpoint, as 0.8's rows.
    let row = json!({
        "id": format!("prompt:{after_prompt}"),
        "after_prompt": after_prompt,
        "node_id": parent_node,
        "trajectory_events": state.map_or(0, |s| s.trajectory_events),
        "agent_session_id": state.and_then(|s| s.session_id.clone()),
        "provider": cp.map(|c| c.provider.clone()),
        "ref": cp.map(|c| c.reference.clone()),
        "created_at": Utc::now().to_rfc3339(),
        "seconds": cp.map(|c| round3(c.seconds)),
        "status": match (cp, o.retention.as_deref()) {
            (None, _) => "failed",
            (Some(_), Some("deferred")) => "delete_deferred",
            (Some(_), Some(r)) => r,
            (Some(_), None) => "kept",
        },
        "error": if cp.is_none() { fork_error(o).map(error_json) } else { None },
        "parent_id": null,
        "depth": 0,
        "stored_bytes": cp.and_then(|c| c.stored_bytes),
        "size_bytes": cp.and_then(|c| c.stored_bytes),
        "fork_point": plan.at.as_str(),
        "fork_id": o.id,
        "lossy": state.is_some_and(|s| !s.killed.is_empty()),
    });
    write_json(
        &dir.join("checkpoints.json"),
        &json!({
            "schema_version": 1,
            "kind": "benchflow-checkpoints",
            "rollout": spec.name,
            "policy": {"every": false, "after": if after_prompt > 0 { json!([after_prompt]) } else { json!([]) }, "keep": 1, "fork": plan.at.as_str()},
            "checkpoints": [row],
        }),
    )?;
    // The trial's result.json gets 0.8's `branches` block (once the parent
    // has ended: the file is the parent's own until then).
    let result_path = dir.join("result.json");
    if !o.parent_running
        && let Ok(text) = std::fs::read_to_string(&result_path)
        && let Ok(mut result) = serde_json::from_str::<Value>(&text)
    {
        result["branches"] = json!({
            "tree": "tree.json",
            "forks": [{
                "id": o.id,
                "status": o.status,
                "value": o.value,
                "children": o.children.len(),
                "scored": scored,
                "parent_node": parent_node,
                "parent_restore": o.parent_restore,
                "parent_reward": parent_reward(o),
                "nodes": result_nodes,
                "cost": fork_cost(o),
            }],
            "children": o.children.len(),
            "child_usage": usage,
            "child_timing_sec": timing.iter().map(|(k, v)| (k.clone(), round3(*v))).collect::<BTreeMap<_, _>>(),
            "cost": fork_cost(o),
            "parent": if o.parent_restore == "skipped" { "discarded" } else { "kept" },
        });
        write_json(&result_path, &result)?;
    }
    Ok(())
}

/// A finished `bench branch` job.
#[derive(Debug, Clone)]
pub struct BranchJob {
    pub dir: PathBuf,
    pub forks: Vec<ForkOutcome>,
}

/// Run one fork per task into a new job folder under `jobs_dir`, with
/// lock.json (children carry `branched_from`), evaluation.json, and 0.8's
/// branch-job summary.json.
pub async fn run_job(
    tasks: Vec<Arc<TaskPackage>>,
    attempt: Attempt,
    plan: ForkPlan,
    jobs_dir: PathBuf,
    job_name: Option<String>,
    rt: Arc<super::Runtime>,
) -> std::io::Result<BranchJob> {
    let dir = match &job_name {
        Some(n) => jobs_dir.join(n),
        None => {
            let name = chrono::Local::now()
                .format("%Y-%m-%d__%H-%M-%S")
                .to_string();
            let mut dir = jobs_dir.join(&name);
            let mut n = 1;
            while dir.exists() {
                dir = jobs_dir.join(format!("{name}-{n}"));
                n += 1;
            }
            dir
        }
    };
    if dir.join("evaluation.json").is_file() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            format!(
                "{} already holds a job; branch runs never resume",
                dir.display()
            ),
        ));
    }
    std::fs::create_dir_all(&dir)?;
    let _running = crate::job::hold(&dir)?;
    let name = dir
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    let parents: Vec<TrialSpec> = tasks
        .iter()
        .map(|t| {
            let tname = crate::job::trial_name(&name, t, &attempt, 0);
            TrialSpec {
                task: t.clone(),
                attempt: attempt.clone(),
                dir: dir.join(&tname),
                name: tname,
            }
        })
        .collect();
    let fingerprint = format!("branch:{}", uuid::Uuid::new_v4().simple());
    let source = json!({"targets": tasks.iter().map(|t| t.root.clone()).collect::<Vec<_>>()});
    let header = || crate::lock::LockHeader {
        job_name: &name,
        fingerprint: &fingerprint,
        source: &source,
        trials_per_task: 1,
        concurrency: plan.concurrency,
        regrade_of: None,
    };
    crate::job::write_lock(&dir, &crate::lock::build(header(), &parents, &rt.options))?;
    write_json(
        &dir.join("evaluation.json"),
        &json!({
            "schema_version": 1,
            "job_name": name,
            "kind": "branch",
            "source": source,
            "config": {
                "environment": rt.provider.name(),
                "agent": attempt.label(),
                "model": attempt.model(),
                "fork_point": plan.at.as_str(),
                "children": plan.children.iter().map(|c| json!({"label": c.label, "own_prompt": c.prompt.is_some()})).collect::<Vec<_>>(),
                "tasks": tasks.iter().map(|t| json!({"name": t.name, "id": t.id, "digest": t.digest, "path": t.root})).collect::<Vec<_>>(),
            },
            "engine": { "name": "benchflow-rs", "version": env!("CARGO_PKG_VERSION"), "fingerprint": fingerprint },
        }),
    )?;
    // The live view's total: each parent and each of its children is one
    // trial line (a branch job never resumes, so none is reused).
    rt.events.send(crate::events::Event::JobStarted {
        dir: dir.clone(),
        total: parents.len() * (1 + plan.children.len()),
        reused: 0,
        resumed: false,
    });
    let mut forks = Vec::new();
    for parent in &parents {
        if rt.stopping() {
            break;
        }
        forks.push(run(parent, &rt, &plan).await);
        // The job's own records after each fork, so a run killed later
        // leaves them for the forks that ended.
        write_json(
            &dir.join("summary.json"),
            &job_summary(&name, &attempt, &plan, &forks, &rt),
        )?;
        std::fs::write(dir.join("results.jsonl"), results_rows(&forks))?;
    }
    // The lock, now with every child, each naming where it came from.
    let mut planned = parents.clone();
    let mut from = vec![None; parents.len()];
    for (f, p) in forks.iter().zip(&parents) {
        for c in &f.children {
            planned.push(TrialSpec {
                task: p.task.clone(),
                attempt: p.attempt.clone(),
                name: format!("{}.{}", p.name, c.node_id),
                dir: c.dir.clone(),
            });
            from.push(Some(crate::lock::LockBranch {
                trial: p.name.clone(),
                checkpoint: f.checkpoint.as_ref().map(|c| c.reference.clone()),
                session_id: f.session_id.clone(),
            }));
        }
    }
    let mut lock = crate::lock::build(header(), &planned, &rt.options);
    for (t, (spec, b)) in lock.trials.iter_mut().zip(planned.iter().zip(from)) {
        if b.is_some() {
            t.trial = spec
                .dir
                .strip_prefix(&dir)
                .map(|r| r.to_string_lossy().to_string())
                .unwrap_or_else(|_| spec.name.clone());
        }
        t.branched_from = b;
    }
    crate::job::write_lock(&dir, &lock)?;
    write_json(
        &dir.join("summary.json"),
        &job_summary(&name, &attempt, &plan, &forks, &rt),
    )?;
    std::fs::write(dir.join("results.jsonl"), results_rows(&forks))?;
    Ok(BranchJob { dir, forks })
}

/// The job's results.jsonl: one row per child (the unit of a branch job),
/// and one for each continuing parent, as `bench run` writes rows, with
/// where each sits in its fork under `info.branch`. A discarded parent has
/// no row: it is unscored by design.
fn results_rows(forks: &[ForkOutcome]) -> String {
    let mut out = String::new();
    let mut push = |mut row: Value, branch: Value| {
        row["info"]["branch"] = branch;
        out.push_str(&serde_json::to_string(&row).unwrap_or_default());
        out.push('\n');
    };
    for f in forks {
        if f.parent_restore != "skipped" {
            push(
                f.parent.results_row(),
                json!({"fork_id": f.id, "node_id": "root", "role": "parent"}),
            );
        }
        for c in &f.children {
            let Some(r) = &c.record else { continue };
            let (status, reward, source) = c.status();
            let mut row = r.results_row();
            // A child scored for its agent's failure has the reward the
            // fork counted, as tree.json says, not the trial's null.
            if source == Some("agent_failure") {
                row["reward"] = json!(reward);
                row["passed"] = json!(false);
                row["metrics"]["reward"] = json!(reward);
            }
            // A child failed for its own time-out was not counted, as
            // tree.json says, whatever its verifier gave it.
            if status == "failed" && r.own_timed_out {
                row["reward"] = Value::Null;
                row["passed"] = Value::Null;
                row["metrics"]["reward"] = Value::Null;
            }
            push(
                row,
                json!({
                    "fork_id": f.id,
                    "node_id": c.node_id,
                    "role": "child",
                    "label": c.label,
                    "parent_rollout": f.parent.rollout_name,
                    "fork_status": f.status,
                    "child_status": status,
                    "child_reward": reward,
                    "reward_source": source,
                    "advantage": match (reward, f.value) {
                        (Some(r), Some(v)) => json!(((r - v) * 1e6).round() / 1e6),
                        _ => Value::Null,
                    },
                }),
            );
        }
    }
    out
}

/// The job's cost: its forks' summed (USD only when every fork's is known).
fn job_cost(forks: &[ForkOutcome]) -> Value {
    let costs: Vec<Value> = forks.iter().map(fork_cost).collect();
    let known = !costs.is_empty() && costs.iter().all(|c| c["usd_known"] == true);
    json!({
        "tokens": costs.iter().map(|c| c["tokens"].as_u64().unwrap_or(0)).sum::<u64>(),
        "usd": known.then(|| (fsum(costs.iter().map(|c| c["usd"].as_f64().unwrap_or(0.0))) * 1e6).round() / 1e6),
        "usd_known": known,
        "sandbox_seconds": round3(costs.iter().map(|c| c["sandbox_seconds"].as_f64().unwrap_or(0.0)).sum()),
    })
}

/// 0.8's `benchflow-branch-job` summary: counts are over children.
fn job_summary(
    name: &str,
    attempt: &Attempt,
    plan: &ForkPlan,
    forks: &[ForkOutcome],
    rt: &super::Runtime,
) -> Value {
    let children: Vec<&ChildOutcome> = forks.iter().flat_map(|f| &f.children).collect();
    let rewards: Vec<f64> = children.iter().filter_map(|c| c.status().1).collect();
    json!({
        "kind": "benchflow-branch-job",
        "job_name": name,
        "agent": attempt.label(),
        "model": attempt.model(),
        "sandbox": rt.provider.name(),
        "checkpoint_after_prompt": match plan.at {
            ForkPoint::Prompt(n) => n,
            _ => 0,
        },
        "fork_point": plan.at.as_str(),
        "snapshot_layers": ["sandbox"],
        "parent": if plan.continue_parent { "continue" } else { "discard" },
        "children_mode": {"isolated": true, "concurrency": plan.concurrency},
        "children_requested": plan.children.iter().map(|c| json!({"label": c.label, "own_prompt": c.prompt.is_some(), "parent": null})).collect::<Vec<_>>(),
        "total": children.len(),
        "passed": rewards.iter().filter(|r| **r >= 1.0).count(),
        "scored": rewards.len(),
        "cost": job_cost(forks),
        "score": if children.is_empty() { 0.0 } else { ((rewards.iter().sum::<f64>() / children.len() as f64) * 10000.0).round() / 10000.0 },
        "trials": forks.iter().map(|f| json!({
            "trial": f.parent.rollout_name,
            "rollout_dir": f.parent.rollout_name,
            "task": f.parent.task_name,
            "fork_id": f.id,
            "fork_status": f.status,
            "value": f.value,
            "parent_reward": parent_reward(f),
            "parent_restore": f.parent_restore,
            "checkpoint_retention": f.retention,
            "retention": f.retention,
            "kept_snapshot": (f.retention.as_deref() == Some("kept")).then(|| f.checkpoint.as_ref().map(|c| c.reference.clone())).flatten(),
            "cost": fork_cost(f),
            "children": f.children.iter().map(|c| json!({"label": c.label, "node_id": c.node_id, "status": c.status().0, "reward": c.status().1})).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fork_points_and_children_parse_as_in_0_8() {
        assert_eq!(ForkPoint::parse("start"), Ok(ForkPoint::Start));
        assert_eq!(ForkPoint::parse("prompt:0"), Ok(ForkPoint::Start));
        assert_eq!(ForkPoint::parse("prompt:2"), Ok(ForkPoint::Prompt(2)));
        assert_eq!(
            ForkPoint::parse("stage:build"),
            Ok(ForkPoint::Stage("build".into()))
        );
        assert!(ForkPoint::parse("middle").is_err());
        let c = ChildPlan::parse("label=hint,prompt=Do a, then b.").unwrap();
        assert_eq!(c.label, "hint");
        assert_eq!(c.prompt.as_deref(), Some("Do a, then b."));
        assert_eq!(
            ChildPlan::parse("label=base").unwrap(),
            ChildPlan {
                label: "base".into(),
                prompt: None
            }
        );
        assert!(ChildPlan::parse("prompt=x").is_err());
        assert!(ChildPlan::parse("label=a b").is_err());
        assert!(ChildPlan::parse("label=a,parent=b").is_err());
        assert_eq!(ForkPlan::same_prompt(3)[2].label, "c3");
    }

    fn fixture(name: &str) -> TaskPackage {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/tasks")
            .join(name);
        benchflow_task::load(&dir).expect("the fixture loads")
    }

    fn plan(at: ForkPoint, children: Vec<ChildPlan>) -> ForkPlan {
        ForkPlan {
            at,
            children,
            retain: false,
            resume: false,
            concurrency: 1,
            continue_parent: false,
        }
    }

    #[test]
    fn refusals_name_each_reason_before_any_sandbox() {
        let caps = benchflow_sandbox::Capabilities {
            checkpoints: true,
            ..Default::default()
        };
        let options = RunOptions::default();
        let hello = fixture("hello");
        let three = plan(ForkPoint::Start, ForkPlan::same_prompt(3));
        // A plain task, oracle, at start, on a backend with checkpoints: runs.
        assert_eq!(
            refusals(&hello, &Attempt::Oracle, &three, &options, &caps, "docker"),
            Vec::<String>::new()
        );
        let has = |r: Vec<String>, needle: &str| {
            assert!(
                r.iter().any(|m| m.contains(needle)),
                "{needle:?} not in {r:?}"
            )
        };
        // No checkpoints on the backend.
        let none = benchflow_sandbox::Capabilities::default();
        has(
            refusals(&hello, &Attempt::Oracle, &three, &options, &none, "modal"),
            "cannot take checkpoints",
        );
        has(
            refusals(
                &hello,
                &Attempt::Oracle,
                &plan(ForkPoint::Stage("b".into()), ForkPlan::same_prompt(2)),
                &options,
                &caps,
                "docker",
            ),
            "no graded stages",
        );
        // prompt:N: the oracle has no session; past the last prompt; a
        // resuming at start.
        let p1 = plan(ForkPoint::Prompt(1), ForkPlan::same_prompt(2));
        has(
            refusals(&hello, &Attempt::Oracle, &p1, &options, &caps, "docker"),
            "has no session",
        );
        has(
            refusals(
                &hello,
                &Attempt::Oracle,
                &plan(ForkPoint::Prompt(2), vec![]),
                &options,
                &caps,
                "docker",
            ),
            "past the parent's last prompt",
        );
        let mut resume_at_start = plan(ForkPoint::Start, ForkPlan::same_prompt(1));
        resume_at_start.resume = true;
        has(
            refusals(
                &hello,
                &Attempt::Oracle,
                &resume_at_start,
                &options,
                &caps,
                "docker",
            ),
            "no session yet to resume",
        );
        // No children; two with one label.
        has(
            refusals(
                &hello,
                &Attempt::Oracle,
                &plan(ForkPoint::Start, vec![]),
                &options,
                &caps,
                "docker",
            ),
            "at least two children",
        );
        // One child, as 0.8 refuses it: no advantage, no spread.
        has(
            refusals(
                &hello,
                &Attempt::Oracle,
                &plan(ForkPoint::Start, ForkPlan::same_prompt(1)),
                &options,
                &caps,
                "docker",
            ),
            "at least two children",
        );
        let twins = vec![
            ChildPlan::parse("label=a").unwrap(),
            ChildPlan::parse("label=a").unwrap(),
        ];
        has(
            refusals(
                &hello,
                &Attempt::Oracle,
                &plan(ForkPoint::Start, twins),
                &options,
                &caps,
                "docker",
            ),
            "two children are labelled `a`",
        );
        // A gVisor-style runtime; --keep.
        let runsc = RunOptions {
            runtime: Some("runsc".into()),
            ..RunOptions::default()
        };
        has(
            refusals(&hello, &Attempt::Oracle, &three, &runsc, &caps, "docker"),
            "--runtime runsc",
        );
        let keep = RunOptions {
            keep: crate::config::Keep::Always,
            ..RunOptions::default()
        };
        has(
            refusals(&hello, &Attempt::Oracle, &three, &keep, &caps, "docker"),
            "--keep",
        );
        // The agent as root.
        let root = RunOptions {
            sandbox_user: "root".into(),
            ..RunOptions::default()
        };
        has(
            refusals(&hello, &Attempt::Oracle, &three, &root, &caps, "docker"),
            "runs as root",
        );
        // Sidecars, stages, a separate verifier.
        has(
            refusals(
                &fixture("sidecar"),
                &Attempt::Oracle,
                &three,
                &options,
                &caps,
                "docker",
            ),
            "sidecar",
        );
        // Stages: a fork at `start` runs them all in each child; after a
        // graded stage that is not the last, the rest; `prompt:N` is not
        // counted across stages.
        let staged = fixture("two-stage");
        assert_eq!(
            refusals(&staged, &Attempt::Oracle, &three, &options, &caps, "docker"),
            Vec::<String>::new()
        );
        let at = |name: &str| plan(ForkPoint::Stage(name.into()), ForkPlan::same_prompt(2));
        assert_eq!(
            refusals(
                &staged,
                &Attempt::Oracle,
                &at("first"),
                &options,
                &caps,
                "docker"
            ),
            Vec::<String>::new()
        );
        has(
            refusals(
                &staged,
                &Attempt::Oracle,
                &at("second"),
                &options,
                &caps,
                "docker",
            ),
            "is the last",
        );
        has(
            refusals(
                &staged,
                &Attempt::Oracle,
                &at("third"),
                &options,
                &caps,
                "docker",
            ),
            "no stage `third` (its stages: first, second)",
        );
        let mut resuming = at("first");
        resuming.resume = true;
        has(
            refusals(&staged, &Attempt::Nop, &resuming, &options, &caps, "docker"),
            "no session for children to resume",
        );
        has(
            refusals(
                &staged,
                &Attempt::Oracle,
                &plan(ForkPoint::Prompt(1), ForkPlan::same_prompt(2)),
                &options,
                &caps,
                "docker",
            ),
            "--at stage:<name>",
        );
        has(
            refusals(
                &fixture("separate-verifier"),
                &Attempt::Oracle,
                &three,
                &options,
                &caps,
                "docker",
            ),
            "sandbox of its own",
        );
        // The oracle and nop take no prompt.
        let own = vec![ChildPlan::parse("label=h,prompt=Try harder.").unwrap()];
        for a in [Attempt::Oracle, Attempt::Nop] {
            has(
                refusals(
                    &hello,
                    &a,
                    &plan(ForkPoint::Start, own.clone()),
                    &options,
                    &caps,
                    "docker",
                ),
                "takes no prompt",
            );
        }
    }

    #[test]
    fn value_and_stderr_match_python() {
        // math.fsum([0.1] * 10) == 1.0, where a plain sum gives 0.9999999999999999.
        assert_eq!(fsum([0.1; 10]), 1.0);
        assert_eq!(
            fsum([1e100, 1.0, -1e100, 1e-100, 1e50, -1.0, -1e50]),
            1e-100
        );
        assert_eq!(fsum([1.0 / 3.0; 3]), 1.0);
        assert_eq!(stderr(&[1.0, 0.0]), Some(0.5));
        assert_eq!(stderr(&[1.0]), None);
        assert_eq!(stderr(&[1.0, 1.0, 1.0]), Some(0.0));
    }

    #[test]
    fn scan_patterns_split_lines_and_drop_short_ones() {
        let pem = "-----BEGIN KEY-----\nabcdefghijkl\n\n-----END KEY-----\n".to_string();
        let p = scan_patterns(&[
            pem,
            "short".into(),
            "sk-1234567890".into(),
            "sk-1234567890".into(),
        ]);
        assert_eq!(
            p,
            [
                "-----BEGIN KEY-----",
                "-----END KEY-----",
                "abcdefghijkl",
                "sk-1234567890"
            ]
        );
        assert!(p.iter().all(|l| !l.is_empty()));
    }

    /// Root's removal checks the folders on the way to each login file,
    /// not the file itself (`rm -f` removes a link, never its target).
    #[test]
    fn the_folders_on_the_way_to_a_login_file_are_checked_for_links() {
        assert_eq!(
            folders_on_the_way("/home/agent/", "/home/agent/.config/opencode/auth.json"),
            ["/home/agent/.config", "/home/agent/.config/opencode"]
        );
        assert!(folders_on_the_way("/home/agent", "/home/agent/.claude.json").is_empty());
        assert_eq!(
            folders_on_the_way("/home/agent", "/etc/x/y"),
            ["/etc", "/etc/x"]
        );
    }

    /// Run on this machine: a login file behind a linked folder is named
    /// and left alone, along with the link's target; the others go, and
    /// each removal is reported.
    #[cfg(unix)]
    #[test]
    fn root_never_removes_a_login_file_through_a_link() {
        let tmp = tempfile::tempdir().unwrap();
        let home = tmp.path().canonicalize().unwrap().join("home");
        let elsewhere = tmp.path().canonicalize().unwrap().join("elsewhere");
        std::fs::create_dir_all(home.join(".codex")).unwrap();
        std::fs::create_dir_all(&elsewhere).unwrap();
        std::fs::write(home.join(".codex/auth.json"), "x").unwrap();
        std::fs::write(elsewhere.join("settings.json"), "kept").unwrap();
        std::os::unix::fs::symlink(&elsewhere, home.join(".claude")).unwrap();
        let h = home.to_str().unwrap();
        let files = [
            format!("{h}/.codex/auth.json"),
            format!("{h}/.claude/settings.json"),
        ];
        let script = login_removal(h, &files);
        let out = std::process::Command::new("sh")
            .arg("-c")
            .arg(&script)
            .output()
            .unwrap();
        let out = String::from_utf8(out.stdout).unwrap();
        assert!(!home.join(".codex/auth.json").exists(), "{out}");
        assert_eq!(
            std::fs::read_to_string(elsewhere.join("settings.json")).unwrap(),
            "kept"
        );
        assert_eq!(
            out,
            format!("BF removed {h}/.codex/auth.json\nBF left {h}/.claude/settings.json\n")
        );
        // A file already gone is neither; a folder in a file's place is left.
        std::fs::create_dir_all(home.join(".codex/auth.json")).unwrap();
        let files = [format!("{h}/.codex/auth.json"), format!("{h}/.gone.json")];
        let out = std::process::Command::new("sh")
            .arg("-c")
            .arg(login_removal(h, &files))
            .output()
            .unwrap();
        let out = String::from_utf8(out.stdout).unwrap();
        assert_eq!(out, format!("BF left {h}/.codex/auth.json\n"));
        // A home that is itself a link: nothing is removed through it, and
        // only files that exist there are named.
        let real = tmp.path().canonicalize().unwrap().join("real-home");
        std::fs::create_dir_all(real.join(".codex")).unwrap();
        std::fs::write(real.join(".codex/auth.json"), "x").unwrap();
        let linked = tmp.path().canonicalize().unwrap().join("linked-home");
        std::os::unix::fs::symlink(&real, &linked).unwrap();
        let l = linked.to_str().unwrap();
        let files = [format!("{l}/.codex/auth.json"), format!("{l}/.gone.json")];
        let out = std::process::Command::new("sh")
            .arg("-c")
            .arg(login_removal(l, &files))
            .output()
            .unwrap();
        let out = String::from_utf8(out.stdout).unwrap();
        assert!(real.join(".codex/auth.json").exists(), "{out}");
        assert_eq!(out, format!("BF left {l}/.codex/auth.json\n"));
    }

    /// How a child that ran counts: an agent's crash or failed turn is a
    /// scored 0 (or the stage rule's reward); a time-out is failed, as in
    /// 0.8, even when graded; a resume that did not load is failed; the
    /// agent never starting, infrastructure, and stops are not scored.
    #[test]
    fn children_are_classified_as_0_8_counts_them() {
        let p = |code: &str, owner: Owner, phase: Phase| Problem::new(code, owner, phase, "m", "n");
        let crash = p("R204", Owner::Agent, Phase::Agent);
        assert_eq!(
            classify(Status::Errored, Some(&crash), false, None, false, None),
            ("scored", Some(0.0), Some("agent_failure"))
        );
        assert_eq!(
            classify(Status::Errored, Some(&crash), false, None, false, Some(0.5)),
            ("scored", Some(0.5), Some("agent_failure"))
        );
        let turn = p("R205", Owner::Agent, Phase::Agent);
        assert_eq!(
            classify(Status::Errored, Some(&turn), false, None, false, None).0,
            "scored"
        );
        // R205 from placing files in setup (a home the parent's agent broke)
        // is not the child's turn: failed.
        let setup = p("R205", Owner::Agent, Phase::Setup);
        assert_eq!(
            classify(Status::Errored, Some(&setup), false, None, false, None).0,
            "failed"
        );
        for code in ["R201", "R202", "R203", "R206"] {
            let never = p(code, Owner::Agent, Phase::Agent);
            assert_eq!(
                classify(Status::Errored, Some(&never), false, None, false, None).0,
                "failed",
                "{code}"
            );
        }
        assert_eq!(
            classify(Status::Scored, None, false, Some(1.0), false, None),
            ("scored", Some(1.0), Some("verifier"))
        );
        assert_eq!(
            classify(Status::Scored, None, true, Some(1.0), false, None),
            ("failed", None, None)
        );
        assert_eq!(
            classify(Status::Errored, Some(&crash), false, None, true, None),
            ("failed", None, None)
        );
        assert_eq!(
            classify(Status::Skipped, None, false, None, false, None).0,
            "cancelled"
        );
        assert_eq!(
            classify(Status::Unscored, None, false, None, false, None).0,
            "unscored"
        );
    }

    /// A child's own prompt gets the task's prompt_prefix first, once, as
    /// 0.8's children do; `@instruction` is the instruction less the prefix.
    #[test]
    fn a_childs_own_prompt_keeps_the_prompt_prefix() {
        let instruction = "Be brief.\n\nWrite hello.txt.";
        assert_eq!(
            compose_child_prompt("Check your work.", instruction, Some("Be brief.")),
            "Be brief.\n\nCheck your work."
        );
        assert_eq!(
            compose_child_prompt(
                "@instruction Start with the tests.",
                instruction,
                Some("Be brief.")
            ),
            "Be brief.\n\nWrite hello.txt. Start with the tests."
        );
        assert_eq!(
            compose_child_prompt("@instruction Then stop.", "Write hello.txt.", None),
            "Write hello.txt. Then stop."
        );
        let mut task = fixture("hello");
        task.config = json!({"agent": {"prompt_prefix": "  Be brief. "}});
        assert_eq!(prompt_prefix(&task).as_deref(), Some("Be brief."));
    }

    /// The process listing names live processes of a user, and its output
    /// is valid shell on this machine.
    #[cfg(target_os = "linux")]
    #[test]
    fn the_process_listing_names_the_users_processes() {
        let me = std::process::Command::new("id")
            .arg("-un")
            .output()
            .unwrap();
        let me = String::from_utf8(me.stdout).unwrap().trim().to_string();
        let out = std::process::Command::new("sh")
            .arg("-c")
            .arg(list_processes(&me))
            .output()
            .unwrap();
        let out = String::from_utf8(out.stdout).unwrap();
        assert!(out.starts_with("BF uid "), "{out}");
        assert!(out.contains("BF proc "), "{out}");
    }

    /// A state folder report: changes after the agent ran are the agent's
    /// (R314); before, the task's; an unchecked folder refuses the fork.
    #[test]
    fn state_reports_name_who_changed_the_folder() {
        let r =
            Report::parse("BF state-owner /var/lib/benchflow/baseline/list\nBF state-checked\n");
        let p = state_problem(&r, true, "before the checkpoint").unwrap();
        assert_eq!((p.code.as_str(), p.owner), ("R314", Owner::Agent));
        assert_eq!(p.phase, Phase::Agent);
        let p = state_problem(&r, false, "before the checkpoint").unwrap();
        assert_eq!((p.code.as_str(), p.owner), ("R104", Owner::Task));
        let r = Report::parse("BF state-unchecked no-find\nBF state-checked\n");
        assert_eq!(state_problem(&r, true, "x").unwrap().code, "R401");
        let r = Report::parse("BF state-baseline\nBF state-checked\n");
        assert!(state_problem(&r, true, "x").is_none());
        assert!(reported_last("a\nBF state-checked\n\n", "BF state-checked"));
        assert!(!reported_last(
            "BF state-checked\nBF left x\n",
            "BF state-checked"
        ));
    }
}
