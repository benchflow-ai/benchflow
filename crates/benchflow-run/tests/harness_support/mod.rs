//! A scripted in-memory sandbox and provider for whole-trial tests of the
//! external agent and computer-1 (`external_trial.rs`,
//! `computer_trial.rs`).
//!
//! The sandbox answers the engine's own setup and hardening programs, keeps
//! a file table for the builtin tools' write command, and hands every other
//! command to the test's handler. It grades the task by a file's content.

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use base64::Engine;
use benchflow_run::agent::{Attempt, ModelAgent};
use benchflow_run::config::RunOptions;
use benchflow_run::events::Events;
use benchflow_run::images::Images;
use benchflow_run::job::{self, JobSpec};
use benchflow_run::record::TrialRecord;
use benchflow_run::trial::{Runtime, stop_switch};
use benchflow_sandbox::{
    Capabilities, Exec, ExecOutput, Image, ImageSpec, Process, Provider, ProviderStatus, Sandbox,
    SandboxError, SandboxSpec,
};
use serde_json::json;

/// What the sandbox holds and what it was asked.
#[derive(Default)]
pub struct World {
    pub files: BTreeMap<String, Vec<u8>>,
    /// Every command, with the user it ran as.
    pub commands: Vec<(String, Option<String>)>,
    /// Image specs the provider was asked to prepare.
    pub images: Vec<String>,
}

/// Answers a command the sandbox does not know, or `None`.
pub type Handler = Arc<dyn Fn(&str, &mut World) -> Option<ExecOutput> + Send + Sync>;

pub fn ok(stdout: impl Into<String>) -> ExecOutput {
    ExecOutput {
        exit_code: 0,
        stdout: stdout.into(),
        stderr: String::new(),
        timed_out: false,
        duration: Duration::from_millis(1),
    }
}

pub fn failed(code: i32, stderr: impl Into<String>) -> ExecOutput {
    ExecOutput {
        exit_code: code,
        stderr: stderr.into(),
        ..ok("")
    }
}

fn words(command: &str) -> Vec<String> {
    shlex::split(command).unwrap_or_default()
}

pub struct FakeSandbox {
    world: Arc<Mutex<World>>,
    handler: Handler,
    /// The file and the content that solve the task.
    solved_by: (String, Vec<u8>),
}

impl FakeSandbox {
    fn run(&self, c: &str) -> ExecOutput {
        let mut w = self.world.lock().unwrap();
        if c.contains("useradd") {
            return ok("BF user agent 1000 /home/agent\n");
        }
        if c.contains("kill -9 -1") {
            return ok("BF killed 1\n");
        }
        // The hardening checks before grading report back, as a sound
        // image's shell would.
        if c.contains("ld.so.preload") {
            return ok("BF preload absent\n");
        }
        if c.contains("BF venv-done") {
            return ok(
                "BF venv-safepath 1\nBF venv-path /usr/local/bin:/usr/bin:/bin\nBF venv-done\n",
            );
        }
        if c.contains("BF path") {
            return ok("BF path /usr/local/bin:/usr/bin:/bin\n");
        }
        if c.contains("BF handoff done") {
            return ok("BF handoff done\n");
        }
        // The builtin tools' `write`: `printf %s <b64> | base64 -d > <path>`.
        if let Some(rest) = c.split("printf %s ").nth(1)
            && c.contains("| base64 -d > ")
        {
            let parts = words(rest);
            let data = base64::engine::general_purpose::STANDARD
                .decode(&parts[0])
                .unwrap();
            let path = parts.last().unwrap().clone();
            w.files.insert(path, data);
            return ok("");
        }
        (self.handler)(c, &mut w).unwrap_or_else(|| ok(""))
    }
}

#[async_trait]
impl Sandbox for FakeSandbox {
    fn id(&self) -> &str {
        "fake-harness"
    }
    fn backend(&self) -> &str {
        "fake"
    }
    async fn exec(&self, exec: Exec) -> Result<ExecOutput, SandboxError> {
        self.world
            .lock()
            .unwrap()
            .commands
            .push((exec.command.clone(), exec.user.clone()));
        Ok(self.run(&exec.command))
    }
    async fn spawn(&self, _exec: Exec) -> Result<Process, SandboxError> {
        Err(SandboxError::Unsupported(
            "these agents spawn nothing".into(),
        ))
    }
    async fn upload(&self, _local: &Path, _remote: &str) -> Result<(), SandboxError> {
        Ok(())
    }
    async fn write_file(&self, remote: &str, c: &[u8], _mode: u32) -> Result<(), SandboxError> {
        self.world
            .lock()
            .unwrap()
            .files
            .insert(remote.to_string(), c.to_vec());
        Ok(())
    }
    async fn download(&self, remote: &str, local: &Path) -> Result<(), SandboxError> {
        if remote == "/logs/verifier" {
            std::fs::create_dir_all(local).unwrap();
            let w = self.world.lock().unwrap();
            let solved = w.files.get(&self.solved_by.0) == Some(&self.solved_by.1);
            std::fs::write(local.join("reward.txt"), if solved { "1\n" } else { "0\n" }).unwrap();
            let status = if solved { "passed" } else { "failed" };
            let ctrf = json!({"results": {"tool": {"name": "test.sh"},
                "tests": [{"name": "test_hello", "status": status}]}});
            std::fs::write(local.join("ctrf.json"), ctrf.to_string()).unwrap();
        }
        Ok(())
    }
    async fn destroy(&self) -> Result<(), SandboxError> {
        Ok(())
    }
}

pub struct FakeProvider {
    pub worlds: Mutex<Vec<Arc<Mutex<World>>>>,
    handler: Handler,
    solved_by: (String, Vec<u8>),
}

#[async_trait]
impl Provider for FakeProvider {
    fn name(&self) -> &str {
        "fake"
    }
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            network_none: true,
            pids_limit: true,
            build_from_dockerfile: true,
            ..Capabilities::default()
        }
    }
    async fn status(&self) -> ProviderStatus {
        ProviderStatus {
            ok: true,
            summary: "fake".into(),
            fix: None,
        }
    }
    async fn prepare(
        &self,
        spec: &ImageSpec,
        _log: &(dyn for<'a> Fn(&'a str) + Send + Sync),
    ) -> Result<Image, SandboxError> {
        Ok(Image {
            reference: format!("fake/{}", spec.name),
            fresh: true,
            duration: Duration::from_millis(1),
        })
    }
    async fn create(&self, _spec: &SandboxSpec) -> Result<Box<dyn Sandbox>, SandboxError> {
        let world = Arc::new(Mutex::new(World::default()));
        self.worlds.lock().unwrap().push(world.clone());
        Ok(Box::new(FakeSandbox {
            world,
            handler: self.handler.clone(),
            solved_by: self.solved_by.clone(),
        }))
    }
}

pub fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/tasks")
        .join(name)
}

pub struct Run {
    pub rec: TrialRecord,
    pub dir: PathBuf,
    pub world: Arc<Mutex<World>>,
    pub _tmp: tempfile::TempDir,
}

/// One trial of `agent` on the fixture `task`, solved when `solved_by.0`
/// holds `solved_by.1`.
pub async fn run_trial(
    task: &str,
    agent: Arc<dyn ModelAgent>,
    handler: Handler,
    solved_by: (&str, &[u8]),
    options: RunOptions,
) -> Run {
    let tmp = tempfile::tempdir().unwrap();
    let task = Arc::new(benchflow_task::load(&fixture(task)).expect("the fixture loads"));
    let provider = Arc::new(FakeProvider {
        worlds: Mutex::new(Vec::new()),
        handler,
        solved_by: (solved_by.0.to_string(), solved_by.1.to_vec()),
    });
    let (tx, rx) = stop_switch();
    std::mem::forget(tx);
    let rt = Arc::new(Runtime {
        provider: provider.clone(),
        images: Images::new(),
        options,
        events: Events::none(),
        stop: rx,
    });
    let spec = JobSpec {
        carry: Vec::new(),
        units: vec![(task.clone(), vec![Attempt::Model(agent)])],
        trials: 1,
        jobs_dir: tmp.path().join("jobs"),
        job_name: Some("harness-job".into()),
        fresh: true,
        concurrency: 1,
        source: json!({"path": fixture(&task.name)}),
    };
    let out = job::run(spec, rt).await.unwrap();
    let rec = out.records[0].clone();
    let dir = out.dir.join(&rec.rollout_name);
    let world = provider.worlds.lock().unwrap()[0].clone();
    Run {
        rec,
        dir,
        world,
        _tmp: tmp,
    }
}

pub fn options(idle: Duration) -> RunOptions {
    RunOptions {
        idle_timeout: idle,
        ..RunOptions::default()
    }
}
