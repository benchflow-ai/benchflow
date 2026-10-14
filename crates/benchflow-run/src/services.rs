//! Services beside the agent's sandbox, package files mounted read-only,
//! and the GPU type check: what a trial sets up around its sandbox.
//!
//! Services start before the agent's sandbox, each from its own image, on a
//! private network (see [`benchflow_sandbox::services`]); the sandbox joins
//! it and reaches each service by name. They are torn down with the trial,
//! and their logs are kept in `services/<name>.log`.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use benchflow_sandbox::services::ServiceSpec;
use benchflow_sandbox::{Exec, Sandbox, SandboxError, SandboxSpec, shell};
use benchflow_task::extras::Mount;
use benchflow_task::package::Ready;

use super::{SETUP_TIMEOUT, Trial, sandbox_problem};
use crate::hardening::{self, Report};
use crate::images;
use crate::outcome::{DefenseState, Owner, Phase, Problem, defense};

/// What a trial holds beyond its sandbox.
#[derive(Default)]
pub(super) struct Extra {
    /// The services' network, for the agent's sandbox to join when it starts.
    pub attach: Option<String>,
    /// The services' network, to remove at teardown.
    pub network: Option<String>,
    /// Each service's name and container.
    pub services: Vec<(String, Box<dyn Sandbox>)>,
    /// A stage's verifier folders, replacing the task's while it grades.
    pub verifier_dirs: Option<Vec<std::path::PathBuf>>,
    /// A stage's verifier time, replacing the task's while it grades.
    pub verifier_timeout: Option<Duration>,
    /// Each stage's result, for result.json.
    pub step_results: Option<serde_json::Value>,
}

fn service_problem(name: &str, message: String, logs: String) -> Problem {
    Problem::new(
        "R106",
        Owner::Task,
        Phase::Start,
        format!("service {name} {message}"),
        "check the service's image, command, and readiness check in the task; the trial's services/ folder has its log",
    )
    .detail(logs)
}

impl Trial<'_> {
    /// Start the task's services on a private network, each ready before
    /// the next starts, and leave the network for the agent's sandbox.
    pub(super) async fn start_services(&mut self) -> Result<(), Problem> {
        let task = self.spec.task.clone();
        if task.extras.services.is_empty() {
            return Ok(());
        }
        let provider = self.rt.provider.clone();
        let Some(host) = provider.services() else {
            return Err(crate::features::no_services(&task, provider.name()));
        };
        let mut labels = BTreeMap::new();
        labels.insert("ai.benchflow.trial".to_string(), self.spec.name.clone());
        let network = host
            .create_network(&images::sanitize_name(&self.spec.name), &labels)
            .await
            .map_err(|e| sandbox_problem(Phase::Start, e))?;
        self.x.network = Some(network.clone());
        let log = |_: &str| {};
        for svc in &task.extras.services {
            let spec = images::spec_for_image(
                &task,
                &svc.image,
                &format!("{}-svc-{}", task.name, svc.name),
            );
            let image = self
                .rt
                .images
                .get(provider.as_ref(), &spec, &log, None)
                .await?;
            let started = host
                .start_service(
                    &network,
                    &ServiceSpec {
                        name: svc.name.clone(),
                        image,
                        command: svc.command.clone(),
                        env: svc.env.clone(),
                        pids_limit: Some(self.rt.options.pids_limit),
                        labels: labels.clone(),
                    },
                )
                .await;
            let sandbox = match started {
                Ok(s) => s,
                Err(SandboxError::Start(message)) => {
                    return Err(service_problem(
                        &svc.name,
                        format!("did not start: {message}"),
                        String::new(),
                    ));
                }
                Err(e) => return Err(sandbox_problem(Phase::Start, e)),
            };
            self.rt.events.log(
                self.name(),
                format!("service {} started ({})", svc.name, sandbox.id()),
            );
            self.x.services.push((svc.name.clone(), sandbox));
            if let Some(ready) = &svc.ready {
                let last = self.x.services.len() - 1;
                if let Err(out) = wait_ready(self.x.services[last].1.as_ref(), ready).await {
                    let logs = host.logs(self.x.services[last].1.id()).await;
                    return Err(service_problem(
                        &svc.name,
                        format!("never passed its readiness check `{}`", ready.run),
                        format!("{out}\n{logs}"),
                    ));
                }
            }
        }
        defense(
            &mut self.defenses,
            "services_network",
            DefenseState::Enforced,
            Some(format!(
                "{} service(s) on a private network with no route out and no address for the host",
                task.extras.services.len()
            )),
        );
        self.x.attach = Some(network);
        Ok(())
    }

    /// Start a sandbox, joined to the services' network when there is one.
    ///
    /// The backend's call runs in a task of its own: when the run is
    /// stopped while the sandbox is being made, the trial stops waiting,
    /// and the sandbox is removed as soon as the backend hands it over
    /// (see [`super::settle_stopped_starts`]).
    pub(super) async fn create_sandbox(
        &self,
        spec: &SandboxSpec,
    ) -> Result<Box<dyn Sandbox>, SandboxError> {
        let provider = self.rt.provider.clone();
        let attach = self.x.attach.clone();
        let spec = spec.clone();
        let checkpoint = self.child_checkpoint();
        let (tx, rx) = tokio::sync::oneshot::channel();
        let start = tokio::spawn(async move {
            let made = match (&attach, provider.services(), &checkpoint) {
                (_, _, Some(cp)) => provider.create_from_checkpoint(&spec, cp).await,
                (Some(network), Some(host), None) => host.create_on_network(&spec, network).await,
                _ => provider.create(&spec).await,
            };
            // Nobody waits for it any more: the trial was stopped.
            if let Err(Ok(sandbox)) = tx.send(made)
                && sandbox.destroy().await.is_err()
            {
                super::note_left_behind(sandbox.as_ref());
            }
        });
        super::track_start(start);
        rx.await.unwrap_or_else(|_| {
            Err(SandboxError::Start(
                "the backend's call to make the sandbox ended without an answer".into(),
            ))
        })
    }

    /// Keep each service's log, then remove the services and their network
    /// (unless asked to keep them).
    pub(super) async fn teardown_services(&mut self, keep: bool) {
        let services = std::mem::take(&mut self.x.services);
        let host = self.rt.provider.services();
        if !services.is_empty() {
            let dir = self.spec.dir.join("services");
            let _ = std::fs::create_dir_all(&dir);
            for (name, sandbox) in &services {
                if let Some(host) = host {
                    let _ = std::fs::write(
                        dir.join(format!("{name}.log")),
                        host.logs(sandbox.id()).await,
                    );
                }
            }
        }
        for (name, sandbox) in services {
            if keep {
                self.rt.events.log(
                    self.name(),
                    format!("kept service {name} ({})", sandbox.id()),
                );
            } else if let Err(e) = sandbox.destroy().await {
                self.rt
                    .events
                    .log(self.name(), format!("could not remove service {name}: {e}"));
                super::note_left_behind(sandbox.as_ref());
            }
        }
        if let (Some(network), Some(host)) = (self.x.network.take(), host)
            && !keep
            && let Err(e) = host.remove_network(&network).await
        {
            self.rt.events.log(self.name(), e.to_string());
        }
    }

    /// The paths the task itself places: its own mounts' targets, and with
    /// `through`, every stage's up to and including that one.
    pub(super) fn task_paths(&self, through: Option<&str>) -> Vec<String> {
        let task = self.task();
        let mut paths: Vec<String> = task
            .extras
            .mounts
            .iter()
            .map(|m| m.target.clone())
            .collect();
        // Up to the stage being prepared: a later stage's mounts are not
        // there yet, and say nothing about whose a failed copy is.
        if let (Some(steps), Some(through)) = (&task.extras.steps, through) {
            for st in &steps.stages {
                paths.extend(st.mounts.iter().map(|m| m.target.clone()));
                if st.name == through {
                    break;
                }
            }
        }
        paths
    }

    /// Place package files in the sandbox, owned by root and read-only.
    /// `after_agent`: an agent has had a turn in this sandbox (a later
    /// stage's mounts), so nothing found at a target is kept.
    /// `agent_alive`: that agent's processes still run (its stage was not
    /// graded). Then a target under a folder the agent user can write
    /// ([`hardening::agent_writable_parents`]) is not root's to write: the agent could swap any path on the way for a
    /// link while root removes, unpacks, and changes modes there. The agent
    /// user copies those files in instead ([`crate::place::tree_for_agent`]):
    /// they are its own, and the defense says so (it could have moved
    /// root's aside anyway).
    pub(super) async fn place_mounts(
        &mut self,
        mounts: &[Mount],
        phase: Phase,
        after_agent: bool,
        agent_alive: bool,
        stage: Option<&str>,
    ) -> Result<(), Problem> {
        // A child's sandbox holds the parent's task-level mounts already; a
        // later stage's mounts (after an agent's turn) a stage child places
        // itself, since the parent never reached that stage before the fork.
        if mounts.is_empty() || (self.is_child() && !after_agent) {
            return Ok(());
        }
        let task = self.spec.task.clone();
        let mut open = Vec::new();
        let mut copied = Vec::new();
        for m in mounts {
            if agent_alive {
                let out = self
                    .root(
                        hardening::agent_writable_parents(&m.target, &self.user),
                        SETUP_TIMEOUT,
                    )
                    .await
                    .map_err(|e| sandbox_problem(phase, e))?;
                let report = Report::parse(&out.stdout);
                let above = report.all("open");
                // A folder already at the target that the agent user cannot
                // write (an earlier mount placed it root's and read-only, or
                // the image did): the agent's copy cannot merge into it, and
                // root writing there could be led anywhere by the running
                // agent. Neither can place the files safely.
                if let Some(d) = above.first()
                    && report.first("target-closed").is_some()
                {
                    return Err(Problem::new(
                        "R104",
                        Owner::Task,
                        phase,
                        format!(
                            "this stage mounts {} where a folder the agent user cannot write is already (an earlier mount, or the image, placed it), while an earlier ungraded stage's agent still runs and can write {d}: neither BenchFlow nor the agent user can put the files there safely",
                            m.target
                        ),
                        "mount the stage's files at a path no earlier mount uses, or grade the earlier stage (its handoff stops the agent first)",
                    ));
                }
                if let Some(d) = above.first() {
                    let user = crate::place::AgentUser {
                        name: &self.user,
                        home: &self.home,
                        earlier_turns: true,
                    };
                    let others = without_one(self.task_paths(stage), &m.target);
                    crate::place::tree_for_agent(
                        self.sb(),
                        &user,
                        &super::in_package(&task.root, &m.source),
                        &m.target,
                        &others,
                    )
                    .await?;
                    copied.push(format!("{} (in {d})", m.target));
                    continue;
                }
            }
            // After an agent's turn (a later stage's mounts), nothing the
            // agent left at the target or on the way to it is kept.
            self.claim(&m.target, after_agent, phase).await?;
            self.sb()
                .upload(&super::in_package(&task.root, &m.source), &m.target)
                .await
                .map_err(|e| crate::trial::put_problem(phase, e, after_agent))?;
            let t = shell::quote(&m.target);
            let out = self
                .root(
                    format!("chown -R 0:0 {t} && chmod -R a-w,a+rX {t}"),
                    SETUP_TIMEOUT,
                )
                .await
                .map_err(|e| sandbox_problem(phase, e))?;
            if !out.success() {
                return Err(Problem::new(
                    "R104",
                    Owner::Task,
                    phase,
                    format!(
                        "{} could not be made read-only at {}",
                        m.source.display(),
                        m.target
                    ),
                    "the image needs chown and chmod",
                )
                .detail(out.tail(1000)));
            }
            let out = self
                .root(hardening::open_parents(&m.target), SETUP_TIMEOUT)
                .await
                .map_err(|e| sandbox_problem(phase, e))?;
            for d in Report::parse(&out.stdout).all("open") {
                open.push(format!("{} (in {d})", m.target));
            }
        }
        let user = if self.user.is_empty() {
            task.agent
                .user
                .clone()
                .unwrap_or_else(|| self.rt.options.sandbox_user.clone())
        } else {
            self.user.clone()
        };
        let root_agent = user == "root" || user == "0";
        let (state, note) = if root_agent {
            (
                DefenseState::Off,
                "the agent runs as root and could change the mounted files".to_string(),
            )
        } else if !copied.is_empty() {
            (
                DefenseState::Failed,
                format!(
                    "copied in by the agent user and its own, since an earlier stage's agent still ran and a folder above is not root's alone: {}",
                    copied.join(", ")
                ),
            )
        } else if !open.is_empty() {
            (
                DefenseState::Failed,
                format!(
                    "read-only, but a folder above is not root's alone, so the agent user can move the files aside and put others in their place: {}",
                    open.join(", ")
                ),
            )
        } else {
            (
                DefenseState::Enforced,
                format!("{} path(s), owned by root and read-only", mounts.len()),
            )
        };
        defense(&mut self.defenses, "read_only_mounts", state, Some(note));
        Ok(())
    }

    /// Check the sandbox's GPUs against the task's accepted types.
    pub(super) async fn check_gpu_types(&mut self) -> Result<(), Problem> {
        let task = self.spec.task.clone();
        let types = &task.extras.gpu_types;
        if types.is_empty() || task.sandbox.gpus == 0 {
            return Ok(());
        }
        let out = self
            .root(
                "nvidia-smi --query-gpu=name --format=csv,noheader".into(),
                SETUP_TIMEOUT,
            )
            .await
            .map_err(|e| sandbox_problem(Phase::Setup, e))?;
        let names: Vec<String> = out
            .stdout
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .map(str::to_string)
            .collect();
        if !out.success() || names.is_empty() {
            return Err(Problem::new(
                "R401",
                Owner::You,
                Phase::Setup,
                format!(
                    "{} accepts only these GPU types: {}; the sandbox has no nvidia-smi to say which GPUs it got",
                    task.name,
                    types.join(", ")
                ),
                "run it on a host whose GPUs nvidia-smi can list, or remove gpu_types",
            ));
        }
        let wrong: Vec<&String> = names.iter().filter(|n| !gpu_matches(n, types)).collect();
        if !wrong.is_empty() {
            return Err(Problem::new(
                "R401",
                Owner::You,
                Phase::Setup,
                format!(
                    "{} accepts only these GPU types: {}; this host gave it {}",
                    task.name,
                    types.join(", "),
                    names.join(", ")
                ),
                "run it on a host with one of those GPUs",
            ));
        }
        defense(
            &mut self.defenses,
            "gpu_types",
            DefenseState::Enforced,
            Some(names.join(", ")),
        );
        Ok(())
    }
}

/// Run a readiness check in `sandbox` until it passes; the last output when
/// it never does.
async fn wait_ready(sandbox: &dyn Sandbox, ready: &Ready) -> Result<(), String> {
    let start = Instant::now();
    let mut failures = 0u32;
    loop {
        let in_grace = start.elapsed() < ready.start_period;
        match sandbox
            .exec(Exec::new(ready.run.clone()).timeout(ready.timeout))
            .await
        {
            Ok(o) if o.success() => return Ok(()),
            Ok(o) => {
                if !in_grace {
                    failures += 1;
                }
                if failures > ready.retries {
                    return Err(o.tail(1500));
                }
            }
            Err(e) => return Err(e.to_string()),
        }
        tokio::time::sleep(if in_grace {
            ready.start_interval
        } else {
            ready.interval
        })
        .await;
    }
}

/// Whether a GPU's name (`NVIDIA H100 80GB HBM3`) is one of `types`
/// (`H100`, `a100-80gb`): every word of the type is a word of the name,
/// ignoring case, with spaces and dashes between words. Whole words, so
/// `A10` is not an A100 and `L4` is not an L40S.
fn gpu_matches(name: &str, types: &[String]) -> bool {
    let words = |s: &str| {
        s.to_ascii_lowercase()
            .split(|c: char| !c.is_ascii_alphanumeric())
            .filter(|w| !w.is_empty())
            .map(str::to_string)
            .collect::<Vec<_>>()
    };
    let n = words(name);
    types.iter().any(|t| {
        let t = words(t);
        !t.is_empty() && t.iter().all(|w| n.contains(w))
    })
}

#[cfg(test)]
mod tests {
    use super::gpu_matches;

    #[test]
    fn gpu_types_match_by_model() {
        let types = vec!["H100".to_string(), "a100-80gb".to_string()];
        assert!(gpu_matches("NVIDIA H100 80GB HBM3", &types));
        assert!(gpu_matches("NVIDIA A100-SXM4-80GB", &["A100".into()]));
        assert!(gpu_matches("NVIDIA A100 80GB PCIe", &types));
        assert!(!gpu_matches("Tesla T4", &types));
        assert!(!gpu_matches("NVIDIA L4", &[String::new()]));
        // Whole model names only.
        assert!(!gpu_matches("NVIDIA A100-SXM4-80GB", &["A10".into()]));
        assert!(!gpu_matches("NVIDIA L40S", &["L4".into()]));
        assert!(gpu_matches("NVIDIA L4", &["L4".into()]));
        assert!(gpu_matches("NVIDIA A100-SXM4-80GB", &types));
        assert!(gpu_matches("NVIDIA GeForce RTX 4090", &["rtx 4090".into()]));
    }
}

/// The task's paths other than one mount's own entry: a target another
/// mount also uses stays among them.
fn without_one(mut paths: Vec<String>, target: &str) -> Vec<String> {
    if let Some(i) = paths.iter().position(|p| p == target) {
        paths.remove(i);
    }
    paths
}

#[cfg(test)]
mod path_tests {
    /// Guards D7c's owner fix for a target two mounts use
    /// (REVIEW-OUTCOMES (d7c) L6).
    #[test]
    fn a_target_two_mounts_use_stays_the_tasks() {
        let paths = vec![
            "/app/data".to_string(),
            "/app/data".to_string(),
            "/srv".to_string(),
        ];
        assert_eq!(
            super::without_one(paths.clone(), "/app/data"),
            vec!["/app/data".to_string(), "/srv".to_string()]
        );
        assert_eq!(super::without_one(paths, "/srv").len(), 2);
    }
}
