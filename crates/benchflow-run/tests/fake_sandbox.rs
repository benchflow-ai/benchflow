//! Whole trials through the real engine, on a scripted in-memory sandbox.
//!
//! The fake answers the engine's hardening programs the way a Linux sandbox
//! would, tracks whether the task was solved, and serves `/logs/verifier`.
//! It checks the engine's control flow, records, resume, and summaries
//! without any container. Set `BENCHFLOW_KEEP_FAKE_JOB=<dir>` to keep the job
//! folder (for checking it with BenchFlow 0.8's `bf.load_job`).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use benchflow_run::agent::Attempt;
use benchflow_run::config::RunOptions;
use benchflow_run::events::Events;
use benchflow_run::images::Images;
use benchflow_run::job::{self, JobSpec};
use benchflow_run::outcome::{DefenseState, Status};
use benchflow_run::trial::{Runtime, stop_switch};
use benchflow_sandbox::{
    Capabilities, Exec, ExecOutput, Image, ImageSpec, Process, Provider, ProviderStatus, Sandbox,
    SandboxError, SandboxSpec,
};
use benchflow_task::TaskPackage;
use benchflow_task::package::*;
use benchflow_task::values::Network;

/// Where the engine packs a trial's outputs in its sandbox.
const OUTPUTS_TAR: &str = "/var/lib/benchflow/outputs.tar";

/// Checkpoints the fake took: name to whether the task was solved then.
static CHECKPOINTS: Mutex<BTreeMap<String, bool>> = Mutex::new(BTreeMap::new());
/// Sandboxes the fake started from a checkpoint.
static FROM_CHECKPOINT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

#[derive(Default)]
struct World {
    solved: bool,
    commands: Vec<String>,
    /// Each command with the folder it was asked to run in.
    cwds: Vec<(String, Option<String>)>,
    /// The workdir the engine asked the provider for.
    spec_workdir: Option<String>,
    /// (user, command, variables) of every command.
    execs: Vec<(Option<String>, String, BTreeMap<String, String>)>,
    /// The variables each command runs without (`Exec::unset`), in the
    /// order of `execs`.
    unsets: Vec<Vec<String>>,
    destroyed: bool,
    /// A `daemon-gone` task's backend went away at `useradd`.
    daemon_gone: bool,
    /// Every file the engine wrote, by path, with its contents.
    written: Vec<(String, Vec<u8>)>,
}

struct FakeSandbox {
    id: String,
    world: Arc<Mutex<World>>,
    solve_marker: String,
    /// A folder on PATH the image lets others write.
    writable_path: Option<String>,
    /// `FAKE_FLAKY_MARKER` in the sandbox's env: until this file exists,
    /// the verifier's command fails as if the container vanished (and makes
    /// the file), and saved outputs hold `app/first-attempt.txt`.
    flaky: Option<PathBuf>,
    /// `FAKE_LEAK_TOKEN` in the sandbox's env: the secret scan before a
    /// checkpoint finds a value in a file.
    leak: bool,
}

#[async_trait]
impl Sandbox for FakeSandbox {
    fn id(&self) -> &str {
        &self.id
    }
    fn backend(&self) -> &str {
        "fake"
    }
    async fn exec(&self, exec: Exec) -> Result<ExecOutput, SandboxError> {
        let c = exec.command.clone();
        if self.id.contains("daemon-gone") {
            let mut w = self.world.lock().unwrap();
            if w.daemon_gone {
                return Err(SandboxError::Transport(
                    "the docker CLI cannot reach its daemon".into(),
                ));
            }
            if c.contains("useradd") {
                w.daemon_gone = true;
                return Ok(ExecOutput {
                    exit_code: 1,
                    stdout: String::new(),
                    stderr: "Cannot connect to the Docker daemon at tcp://127.0.0.1:23751. Is the docker daemon running?\n".into(),
                    timed_out: false,
                    duration: Duration::ZERO,
                });
            }
        }
        if let Some(marker) = &self.flaky
            && c.contains("/verifier/test.sh")
            && !marker.exists()
        {
            std::fs::write(marker, "").unwrap();
            return Err(SandboxError::Transport("the container vanished".into()));
        }
        let mut w = self.world.lock().unwrap();
        w.commands.push(c.clone());
        w.cwds.push((c.clone(), exec.cwd.clone()));
        w.execs
            .push((exec.user.clone(), c.clone(), exec.env.clone()));
        w.unsets.push(exec.unset.clone());
        let stdout = if self.id.contains("shadowed") {
            // The agent replaced `sh`: every program "succeeds" and says nothing.
            String::new()
        } else if c.contains("BF scanned") {
            if self.leak {
                "BF hit /etc/leaked\nBF scanned 1\nBF done 1\n".to_string()
            } else {
                "BF scanned 1\nBF done 1\n".to_string()
            }
        } else if c.contains("BF state-checked") {
            // The state folder check (a fork's parent and children): root's
            // alone, with its baseline.
            "BF state-baseline\nBF state-checked\n".to_string()
        } else if c.contains("bf_try()") {
            // The network self-test, answered as a closed network would.
            closed_network_canaries(&c)
        } else if c.lines().last() == Some("pwd") {
            // An image built with `WORKDIR /root`, or one that sets none.
            if self.id.contains("imgroot") {
                "/root\n"
            } else {
                "/\n"
            }
            .to_string()
        } else if c.contains("BF image-path") {
            // An image whose PATH starts in a venv inside the workdir.
            "BF image-path /app/.venv/bin:/usr/local/bin:/usr/bin:/bin\nBF image-loader BASH_ENV\n"
                .to_string()
        } else if c.contains("BF open") {
            // Folders above a path that are not root's alone: the workdir.
            if c.contains("/app/") {
                "BF open /app\n".to_string()
            } else {
                String::new()
            }
        } else if c.contains("useradd") {
            "BF user agent 1000 /home/agent\n".to_string()
        } else if c.contains("baseline") && c.contains("files.tar") && !c.contains("after") {
            "BF baseline 2\n".to_string()
        } else if c.contains("kill -9 -1") {
            "BF killed 3\n".to_string()
        } else if c.contains("BF unlocked-path") {
            match &self.writable_path {
                Some(d) => format!("BF unlocked-path {d}\n"),
                None => String::new(),
            }
        } else if c.contains("BF syswrite-left") {
            "BF syswrite-count 0\nBF syswrite-left 0\n".to_string()
        } else if c.contains("ld.so.preload") {
            "BF preload absent\n".to_string()
        } else if c.contains("cat /proc/self/oom_score_adj") {
            if c.contains("echo 1000 > /proc/self/oom_score_adj") {
                "1000\n".to_string()
            } else {
                "0\n".to_string()
            }
        } else if c.contains("BF mem ") {
            "BF mem v2 1073741824 max\nBF memtotal 1024000\n".to_string()
        } else if c.contains("BF venv-done") {
            "BF venv-safepath 1\nBF venv-path /usr/local/bin:/usr/bin:/bin\nBF venv-done\n"
                .to_string()
        } else if c.contains("BF path")
            && self.id.contains("nopath")
            && w.commands.iter().any(|x| x.contains("kill -9 -1"))
        {
            // After the agent's turn the trusted PATH check says nothing.
            String::new()
        } else if c.contains("BF path") {
            "BF path /usr/local/bin:/usr/bin:/bin\n".to_string()
        } else if c.contains("BF handoff done") {
            "BF planted /app/conftest.py\nBF handoff done\n".to_string()
        } else if c.contains("BF owned") {
            let owner = c
                .split("; o=")
                .nth(1)
                .and_then(|s| s.lines().next())
                .and_then(|l| l.split("; a=").next());
            format!("BF owned {}\n", owner.unwrap_or("").trim_matches('\''))
        } else if c.contains("BF stage-ready") {
            "BF stage-ready\n".to_string()
        } else if c.contains("BF tree placed") {
            "BF tree placed\n".to_string()
        } else if c.contains("BF baseline-added") {
            "BF baseline-added 0\n".to_string()
        } else if c.contains("BF staged") {
            "BF staged\n".to_string()
        } else if c.contains("BF placement done") {
            "BF placement done\n".to_string()
        } else if c.contains("BF locked") && self.id.contains("ships") {
            // An image that ships its own /solution.
            "BF locked /solution\n".to_string()
        } else if c.contains("BF missing") {
            // The rubric judge's probe: the solved answer is a 6-byte file.
            if w.solved && c.contains("/app/answer.txt") {
                "BF file 6 /app/answer.txt\n".to_string()
            } else {
                "BF missing\n".to_string()
            }
        } else if c.contains(&self.solve_marker) {
            w.solved = true;
            String::new()
        } else {
            String::new()
        };
        Ok(ExecOutput {
            exit_code: 0,
            stdout,
            stderr: String::new(),
            timed_out: false,
            duration: Duration::from_millis(1),
        })
    }
    async fn spawn(&self, _exec: Exec) -> Result<Process, SandboxError> {
        Err(SandboxError::Unsupported("no processes in the fake".into()))
    }
    async fn upload(&self, local: &Path, _remote: &str) -> Result<(), SandboxError> {
        // Like a real backend, an upload reads the local files.
        std::fs::symlink_metadata(local)
            .map(|_| ())
            .map_err(|e| SandboxError::io(format!("reading {} to upload it", local.display()), e))
    }
    async fn write_file(
        &self,
        remote: &str,
        contents: &[u8],
        _mode: u32,
    ) -> Result<(), SandboxError> {
        self.world
            .lock()
            .unwrap()
            .written
            .push((remote.to_string(), contents.to_vec()));
        // Restored outputs carry the solved work into a verifier's sandbox.
        if remote == OUTPUTS_TAR && contents.windows(6).any(|w| w == b"SOLVED") {
            self.world.lock().unwrap().solved = true;
        }
        Ok(())
    }
    async fn download(&self, remote: &str, local: &Path) -> Result<(), SandboxError> {
        if remote == OUTPUTS_TAR {
            let body: &[u8] = if self.world.lock().unwrap().solved {
                b"SOLVED"
            } else {
                b"nothing"
            };
            let mut b = tar::Builder::new(std::fs::File::create(local).unwrap());
            let mut h = tar::Header::new_gnu();
            h.set_size(body.len() as u64);
            h.set_mode(0o644);
            b.append_data(&mut h, "app/answer.txt", body).unwrap();
            if self.flaky.as_ref().is_some_and(|m| !m.exists()) {
                let mut h = tar::Header::new_gnu();
                h.set_size(1);
                h.set_mode(0o644);
                b.append_data(&mut h, "app/first-attempt.txt", &b"1"[..])
                    .unwrap();
            }
            b.finish().unwrap();
        }
        if remote == "/app/answer.txt" {
            std::fs::write(local, "SOLVED").unwrap();
        }
        if remote == "/logs/verifier" {
            std::fs::create_dir_all(local).unwrap();
            let solved = self.world.lock().unwrap().solved;
            std::fs::write(local.join("reward.txt"), if solved { "1\n" } else { "0\n" }).unwrap();
            let stdout = if self.id.contains("apt-gone") && !solved {
                // chess-best-move's oracle today: the package lists were gone.
                "E: Unable to locate package stockfish\n0 passed\n"
            } else {
                "1 passed\n"
            };
            std::fs::write(local.join("test-stdout.txt"), stdout).unwrap();
        }
        Ok(())
    }
    async fn destroy(&self) -> Result<(), SandboxError> {
        self.world.lock().unwrap().destroyed = true;
        Ok(())
    }
    async fn checkpoint(
        &self,
        name: &str,
        _scrub_env: &[String],
    ) -> Result<benchflow_sandbox::checkpoint::Checkpoint, SandboxError> {
        let solved = self.world.lock().unwrap().solved;
        CHECKPOINTS.lock().unwrap().insert(name.to_string(), solved);
        Ok(benchflow_sandbox::checkpoint::Checkpoint {
            provider: "fake".into(),
            reference: name.to_string(),
            seconds: 0.5,
            stored_bytes: Some(1000),
            default_user: None,
        })
    }
}

/// What a network self-test script prints in a sandbox with no network:
/// every connection probe refused, every name unresolved.
fn closed_network_canaries(script: &str) -> String {
    let mut out = String::new();
    for line in script.lines() {
        let Some(i) = line.find("echo \"BF canary ") else {
            continue;
        };
        let rest = &line[i + "echo \"BF canary ".len()..];
        let kind = rest.split(' ').next().unwrap_or("");
        let Some(j) = rest.find(") \"") else {
            continue;
        };
        let target = rest[j + 3..].split(';').next().unwrap_or("").trim();
        let target = target.trim_matches('\'');
        let result = if rest.contains("$(bf_resolve") {
            "unresolved"
        } else {
            "refused"
        };
        out.push_str(&format!("BF canary {kind} {result} {target}\n"));
    }
    out
}

/// A build-log line with a signed URL: a SAS signature and a JWT.
const SIGNED_URL_LINE: &str = "GET https://store.example.net/ctx/build.tar?sv=2018-11-09&sig=fHQ0v5V%2FO0Tk%2FNl%2BCjLnneax9lf%2FUOd6MVE2lNtukYo%3D&jwt=eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJidWlsZC1jdHgifQ.c2lnbmF0dXJlLXZhbHVlLWhlcmU 200";

struct FakeProvider {
    worlds: Mutex<Vec<Arc<Mutex<World>>>>,
    writable_path: Option<String>,
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
            checkpoints: true,
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
        log: &(dyn for<'a> Fn(&'a str) + Send + Sync),
    ) -> Result<Image, SandboxError> {
        log("built");
        // A builder that prints a signed URL, as Daytona's build log does.
        log(SIGNED_URL_LINE);
        Ok(Image {
            reference: format!("fake/{}:{}", spec.name, &spec.digest[..12]),
            fresh: true,
            duration: Duration::from_millis(1),
        })
    }
    async fn create(&self, spec: &SandboxSpec) -> Result<Box<dyn Sandbox>, SandboxError> {
        let world = Arc::new(Mutex::new(World {
            spec_workdir: spec.workdir.clone(),
            ..World::default()
        }));
        self.worlds.lock().unwrap().push(world.clone());
        Ok(Box::new(FakeSandbox {
            id: format!("fake-{}", spec.name),
            world,
            solve_marker: "/oracle/solve.sh".into(),
            writable_path: self.writable_path.clone(),
            flaky: spec.env.get("FAKE_FLAKY_MARKER").map(PathBuf::from),
            leak: spec.env.contains_key("FAKE_LEAK_TOKEN"),
        }))
    }
    async fn create_from_checkpoint(
        &self,
        spec: &SandboxSpec,
        checkpoint: &benchflow_sandbox::checkpoint::Checkpoint,
    ) -> Result<Box<dyn Sandbox>, SandboxError> {
        let solved = *CHECKPOINTS
            .lock()
            .unwrap()
            .get(&checkpoint.reference)
            .expect("a child starts from a checkpoint the fake took");
        let sb = self.create(spec).await?;
        FROM_CHECKPOINT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.worlds
            .lock()
            .unwrap()
            .last()
            .unwrap()
            .lock()
            .unwrap()
            .solved = solved;
        Ok(sb)
    }
    async fn delete_checkpoint(
        &self,
        checkpoint: &benchflow_sandbox::checkpoint::Checkpoint,
    ) -> Result<bool, SandboxError> {
        CHECKPOINTS.lock().unwrap().remove(&checkpoint.reference);
        Ok(true)
    }
}

fn task(root: &Path, name: &str) -> TaskPackage {
    std::fs::create_dir_all(root.join("oracle")).unwrap();
    std::fs::create_dir_all(root.join("verifier")).unwrap();
    std::fs::write(root.join("oracle/solve.sh"), "#!/bin/bash\n").unwrap();
    std::fs::write(root.join("verifier/test.sh"), "#!/bin/bash\n").unwrap();
    TaskPackage {
        root: root.to_path_buf(),
        format: Format::TaskMd,
        id: format!("fixtures/{name}"),
        name: name.to_string(),
        title: None,
        version: None,
        instruction: "Do the thing.".into(),
        stages: Vec::new(),
        agent: AgentPolicy {
            timeout: Some(Duration::from_secs(60)),
            ..AgentPolicy::default()
        },
        sandbox: SandboxPolicy {
            image: ImageRef::Pull {
                reference: "ubuntu:24.04".into(),
            },
            cpus: None,
            memory_mb: None,
            disk_mb: None,
            gpus: 0,
            build_timeout: None,
            network: Network::Open,
            workdir: Some("/app".into()),
            env: BTreeMap::new(),
            ready: None,
            outputs: Vec::new(),
            skills: None,
            skills_dir: None,
            mcp: Vec::new(),
            allow_cloud_metadata: false,
        },
        verifier: VerifierPolicy {
            dir: root.join("verifier"),
            mount: "/verifier".into(),
            entry: "test.sh".into(),
            starts_in_mount: false,
            timeout: Some(Duration::from_secs(60)),
            user: None,
            env: BTreeMap::new(),
            network: None,
            isolation: Isolation::Shared,
            rubric: None,
            reward_range: None,
        },
        oracle: Some(OraclePolicy {
            dir: root.join("oracle"),
            mount: "/oracle".into(),
            entry: "solve.sh".into(),
            env: BTreeMap::new(),
        }),
        controls: Vec::new(),
        unsupported: Vec::new(),
        extras: Default::default(),
        config: serde_json::json!({}),
        digest: format!("sha256:{name}"),
    }
}

fn runtime(provider: Arc<FakeProvider>) -> Arc<Runtime> {
    let (_tx, rx) = stop_switch();
    std::mem::forget(_tx);
    Arc::new(Runtime {
        provider,
        images: Images::new(),
        options: RunOptions::default(),
        events: Events::none(),
        stop: rx,
    })
}

fn jobs_dir(tmp: &Path) -> PathBuf {
    match std::env::var_os("BENCHFLOW_KEEP_FAKE_JOB") {
        Some(d) => PathBuf::from(d),
        None => tmp.join("jobs"),
    }
}

#[tokio::test]
async fn oracle_and_nop_run_score_and_resume() {
    let tmp = tempfile::tempdir().unwrap();
    let tasks: Vec<Arc<TaskPackage>> = ["alpha", "beta"]
        .iter()
        .map(|n| Arc::new(task(&tmp.path().join(n), n)))
        .collect();
    let provider = Arc::new(FakeProvider {
        worlds: Mutex::new(Vec::new()),
        writable_path: None,
    });
    let spec = JobSpec {
        carry: Vec::new(),
        units: tasks
            .iter()
            .map(|t| (t.clone(), vec![Attempt::Oracle, Attempt::Nop]))
            .collect(),
        trials: 1,
        jobs_dir: jobs_dir(tmp.path()),
        job_name: Some("fake-job".into()),
        fresh: false,
        concurrency: 3,
        source: serde_json::json!({"path": tmp.path()}),
    };
    let out = job::run(spec.clone(), runtime(provider.clone()))
        .await
        .unwrap();
    assert_eq!(out.records.len(), 4);
    for r in &out.records {
        assert_eq!(r.status, Status::Scored, "{:?}", r.problem);
        let expect = r.agent == "oracle";
        assert_eq!(r.passed, Some(expect), "{} {}", r.task_name, r.agent);
        assert_eq!(r.reward, Some(if expect { 1.0 } else { 0.0 }));
        assert_eq!(r.custody.killed_processes, 3);
        assert_eq!(r.custody.workdir_owner.as_deref(), Some("root"));
        assert_eq!(r.custody.planted, vec!["/app/conftest.py".to_string()]);
        for d in [
            "agent_user",
            "kill_before_grading",
            "trusted_path",
            "planted_file_cleanup",
            "process_limit",
        ] {
            assert!(r.defenses.contains_key(d), "missing defense {d}");
        }
        // Each phase is timed, and the parts sit inside 0.8's totals.
        let t = &r.timing;
        assert!(
            t.sandbox_start > 0.0 && t.setup > 0.0 && t.handoff > 0.0 && t.verify > 0.0,
            "{t:?}"
        );
        assert!(
            t.image + t.sandbox_start + t.setup <= t.environment_setup + 1e-6,
            "{t:?}"
        );
        assert!(t.handoff + t.verify <= t.verifier + 1e-6, "{t:?}");
        let dir = out.dir.join(&r.rollout_name);
        // BenchFlow 0.8 judges a trial with no ACP trajectory source an
        // agent that exited at once; a scripted run must say it is one.
        let result: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("result.json")).unwrap())
                .unwrap();
        assert_eq!(result["trajectory_source"], "scripted");
        for f in [
            "result.json",
            "config.json",
            "timing.json",
            "prompts.json",
            "custody.json",
            "trajectory/acp_trajectory.jsonl",
            "verifier/reward.txt",
        ] {
            assert!(dir.join(f).is_file(), "missing {f}");
        }
    }
    // The reference solution runs as root, so its handoff only notes a
    // change to BenchFlow's state folder, and the defense is not recorded
    // for it; an attempt as the agent user, with no way to root, is held
    // to the strict check.
    for r in &out.records {
        let custody = r.defenses.get("state_custody").map(|d| d.state);
        if r.agent == "oracle" {
            assert_eq!(custody, None, "{:?}", r.defenses);
        } else {
            let rootless =
                r.defenses["agent_user"].state == benchflow_run::outcome::DefenseState::Enforced;
            let expect = if rootless {
                benchflow_run::outcome::DefenseState::Enforced
            } else {
                benchflow_run::outcome::DefenseState::Off
            };
            assert_eq!(custody, Some(expect), "{:?}", r.defenses);
        }
    }
    for w in provider.worlds.lock().unwrap().iter() {
        let w = w.lock().unwrap();
        let oracle = w.commands.iter().any(|c| c.contains("solve.sh"));
        let handoff = w
            .commands
            .iter()
            .find(|c| c.contains("BF handoff done"))
            .unwrap();
        assert_eq!(handoff.contains("BF state-noted"), oracle, "{handoff}");
    }
    // The workdir is root's from the handoff until grading, as in 0.8.
    for w in provider.worlds.lock().unwrap().iter() {
        let w = w.lock().unwrap();
        let at = |needle: &str| w.commands.iter().position(|c| c.contains(needle));
        let (handoff, owned, graded) = (
            at("BF handoff done"),
            at("BF owned"),
            at("/verifier/test.sh"),
        );
        assert!(
            handoff < owned && owned < graded,
            "{handoff:?} {owned:?} {graded:?}"
        );
        assert!(
            w.commands[owned.unwrap()].contains("o='root'")
                || w.commands[owned.unwrap()].contains("o=root")
        );
    }
    // Every sandbox was destroyed.
    assert!(
        provider
            .worlds
            .lock()
            .unwrap()
            .iter()
            .all(|w| w.lock().unwrap().destroyed)
    );
    // The summary counts match.
    assert_eq!(out.summary["passed"], 2);
    assert_eq!(out.summary["failed"], 2);
    assert_eq!(out.summary["total"], 4);
    // A second run resumes: everything is reused, nothing runs.
    let again = job::run(spec, runtime(provider.clone())).await.unwrap();
    assert_eq!(again.reused, 4);
    assert!(again.resumed);
    assert_eq!(
        provider.worlds.lock().unwrap().len(),
        4,
        "no new sandboxes on resume"
    );
}

/// A backend whose build fails after printing a signed URL, as a Daytona
/// build that cannot fetch its context does.
struct FailingBuild;

#[async_trait]
impl Provider for FailingBuild {
    fn name(&self) -> &str {
        "fake"
    }
    fn capabilities(&self) -> Capabilities {
        Capabilities {
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
        log: &(dyn for<'a> Fn(&'a str) + Send + Sync),
    ) -> Result<Image, SandboxError> {
        log(SIGNED_URL_LINE);
        Err(SandboxError::Image {
            name: spec.name.clone(),
            message: format!("the build failed: {SIGNED_URL_LINE}"),
            log_tail: format!("step 1/3\n{SIGNED_URL_LINE}\nerror: context fetch failed"),
            task_fault: true,
        })
    }
    async fn create(&self, _spec: &SandboxSpec) -> Result<Box<dyn Sandbox>, SandboxError> {
        unreachable!("the build failed")
    }
}

#[tokio::test]
async fn a_failed_builds_problem_carries_no_signed_url() {
    // The builder's last lines go into the problem's message and detail,
    // and so into result.json: they are redacted too.
    let tmp = tempfile::tempdir().unwrap();
    let t = Arc::new(task(&tmp.path().join("alpha"), "alpha"));
    let spec = JobSpec {
        carry: Vec::new(),
        units: vec![(t, vec![Attempt::Oracle])],
        trials: 1,
        jobs_dir: jobs_dir(tmp.path()),
        job_name: Some("failed-build".into()),
        fresh: false,
        concurrency: 1,
        source: serde_json::json!({"path": tmp.path()}),
    };
    let (_tx, rx) = stop_switch();
    std::mem::forget(_tx);
    let rt = Arc::new(Runtime {
        provider: Arc::new(FailingBuild),
        images: Images::new(),
        options: RunOptions::default(),
        events: Events::none(),
        stop: rx,
    });
    let out = job::run(spec, rt).await.unwrap();
    let r = &out.records[0];
    let p = r.problem.as_ref().unwrap();
    assert_eq!(p.code, "R101", "{p:?}");
    let result = std::fs::read_to_string(
        jobs_dir(tmp.path())
            .join("failed-build")
            .join(&r.rollout_name)
            .join("result.json"),
    )
    .unwrap();
    assert!(
        result.contains("store.example.net/ctx/build.tar"),
        "{result}"
    );
    assert!(!result.contains("fHQ0v5V"), "{result}");
    assert!(!result.contains("eyJzdWIiOiJidWlsZC1jdHgifQ"), "{result}");
}

#[tokio::test]
async fn image_log_lines_are_redacted() {
    // A builder's log can carry signed URLs (a SAS signature, a JWT); the
    // trial's image.log keeps the line but not the credentials.
    let tmp = tempfile::tempdir().unwrap();
    let t = Arc::new(task(&tmp.path().join("alpha"), "alpha"));
    let provider = Arc::new(FakeProvider {
        worlds: Mutex::new(Vec::new()),
        writable_path: None,
    });
    let spec = JobSpec {
        carry: Vec::new(),
        units: vec![(t, vec![Attempt::Oracle])],
        trials: 1,
        jobs_dir: jobs_dir(tmp.path()),
        job_name: Some("signed-urls".into()),
        fresh: false,
        concurrency: 1,
        source: serde_json::json!({"path": tmp.path()}),
    };
    let out = job::run(spec, runtime(provider)).await.unwrap();
    let r = &out.records[0];
    let log = std::fs::read_to_string(
        jobs_dir(tmp.path())
            .join("signed-urls")
            .join(&r.rollout_name)
            .join("image.log"),
    )
    .unwrap();
    assert!(log.contains("built"), "{log}");
    assert!(log.contains("store.example.net/ctx/build.tar"), "{log}");
    assert!(!log.contains("fHQ0v5V"), "{log}");
    assert!(!log.contains("eyJzdWIiOiJidWlsZC1jdHgifQ"), "{log}");
}

/// Verifiers written for 0.8 grade what the oracle printed, reading it at
/// `/logs/agent/oracle.txt` in the sandbox (MMMLU's test.sh). The oracle
/// writes it there itself, stdout and stderr in the order printed, as 0.8's
/// `> /logs/agent/oracle.txt 2>&1` does, and no copy assembled on this
/// machine (stdout, then stderr) is written over it (review E4-B, 2). A nop
/// runs no such command.
#[tokio::test]
async fn the_oracles_output_is_where_verifiers_read_it() {
    let tmp = tempfile::tempdir().unwrap();
    let t = Arc::new(task(&tmp.path().join("printed"), "printed"));
    let provider = Arc::new(FakeProvider {
        worlds: Mutex::new(Vec::new()),
        writable_path: None,
    });
    let spec = JobSpec {
        carry: Vec::new(),
        units: vec![(t, vec![Attempt::Oracle, Attempt::Nop])],
        trials: 1,
        jobs_dir: tmp.path().join("jobs"),
        job_name: None,
        fresh: true,
        concurrency: 1,
        source: serde_json::json!({}),
    };
    job::run(spec, runtime(provider.clone())).await.unwrap();
    let worlds = provider.worlds.lock().unwrap();
    let mut redirected = 0;
    for w in worlds.iter() {
        let w = w.lock().unwrap();
        redirected += w
            .commands
            .iter()
            .filter(|c| {
                c.contains("solve.sh")
                    && c.contains("> /logs/agent/oracle.txt 2>&1")
                    && c.contains("cat /logs/agent/oracle.txt")
            })
            .count();
        assert!(
            !w.written.iter().any(|(p, _)| p == "/logs/agent/oracle.txt"),
            "nothing is written over the oracle's own file"
        );
    }
    assert_eq!(redirected, 1, "only the oracle's sandbox runs it");
}

#[tokio::test]
async fn the_oracle_runs_with_sh_on_an_image_without_bash() {
    let tmp = tempfile::tempdir().unwrap();
    let t = Arc::new(task(&tmp.path().join("nobash"), "nobash"));
    let provider = Arc::new(FakeProvider {
        worlds: Mutex::new(Vec::new()),
        writable_path: None,
    });
    let spec = JobSpec {
        carry: Vec::new(),
        units: vec![(t, vec![Attempt::Oracle])],
        trials: 1,
        jobs_dir: tmp.path().join("jobs"),
        job_name: None,
        fresh: true,
        concurrency: 1,
        source: serde_json::json!({}),
    };
    job::run(spec, runtime(provider.clone())).await.unwrap();
    let worlds = provider.worlds.lock().unwrap();
    let w = worlds[0].lock().unwrap();
    let solve = w
        .commands
        .iter()
        .find(|c| c.contains("solve.sh") && !c.contains("chmod") && !c.contains("rm -rf"))
        .expect("the oracle ran");
    // Alpine and busybox images have no bash; `bash solve.sh` exited 127
    // there and the trial was scored 0 as if the solution had failed.
    assert!(
        solve.contains("else sh ") && solve.contains("command -v bash"),
        "the oracle needs bash: {solve}"
    );
}

#[tokio::test]
async fn a_task_without_an_oracle_is_refused_by_name() {
    let tmp = tempfile::tempdir().unwrap();
    let mut t = task(&tmp.path().join("gamma"), "gamma");
    t.oracle = None;
    t.unsupported.push(Unsupported {
        field: "[sandbox] clock".into(),
        reason: "BenchFlow sets no clock".into(),
        agents_only: false,
    });
    let provider = Arc::new(FakeProvider {
        worlds: Mutex::new(Vec::new()),
        writable_path: None,
    });
    let spec = JobSpec {
        carry: Vec::new(),
        units: vec![(Arc::new(t), vec![Attempt::Oracle, Attempt::Nop])],
        trials: 1,
        jobs_dir: tmp.path().join("jobs"),
        job_name: None,
        fresh: true,
        concurrency: 2,
        source: serde_json::json!({}),
    };
    let out = job::run(spec, runtime(provider.clone())).await.unwrap();
    let oracle = out.records.iter().find(|r| r.agent == "oracle").unwrap();
    assert_eq!(oracle.problem.as_ref().unwrap().code, "R306");
    let nop = out.records.iter().find(|r| r.agent == "nop").unwrap();
    let p = nop.problem.as_ref().unwrap();
    assert_eq!(p.code, "R401");
    assert!(p.message.contains("[sandbox] clock"), "{}", p.message);
    assert!(
        provider.worlds.lock().unwrap().is_empty(),
        "refused before any sandbox"
    );
}

fn runtime_with(provider: Arc<FakeProvider>, options: RunOptions) -> Arc<Runtime> {
    let (_tx, rx) = stop_switch();
    std::mem::forget(_tx);
    Arc::new(Runtime {
        provider,
        images: Images::new(),
        options,
        events: Events::none(),
        stop: rx,
    })
}

#[tokio::test]
async fn every_job_writes_a_lock_and_stamps_the_dataset() {
    let tmp = tempfile::tempdir().unwrap();
    let t = Arc::new(task(&tmp.path().join("alpha"), "alpha"));
    let provider = Arc::new(FakeProvider {
        worlds: Mutex::new(Vec::new()),
        writable_path: None,
    });
    let mut options = RunOptions {
        bench_version: Some("0.8.1-dev.0 (abc123def 2026-10-10)".into()),
        verifier_timeout_multiplier: 2.0,
        ..RunOptions::default()
    };
    options
        .agent_env
        .insert("SECRET_TOKEN".into(), "do-not-write".into());
    let mut digests = BTreeMap::new();
    digests.insert(t.root.clone(), "sha256:pinned".to_string());
    options.job_files = vec![benchflow_run::jobfile::JobFileRecord {
        source: "https://github.com/o/r/blob/main/job.yaml".into(),
        fetched_from: Some("https://raw.githubusercontent.com/o/r/main/job.yaml".into()),
        digest: "sha256:abc".into(),
        bytes: 12,
    }];
    options.dataset = Some(benchflow_run::config::DatasetStamp {
        name: "demo".into(),
        version: "1.0".into(),
        registry: "registry.json".into(),
        digests,
    });
    let spec = JobSpec {
        carry: Vec::new(),
        units: vec![(t.clone(), vec![Attempt::Oracle, Attempt::Nop])],
        trials: 2,
        jobs_dir: tmp.path().join("jobs"),
        job_name: Some("locked".into()),
        fresh: false,
        concurrency: 2,
        source: serde_json::json!({"targets": ["alpha"]}),
    };
    let out = job::run(spec, runtime_with(provider, options))
        .await
        .unwrap();
    let lock = benchflow_run::read::read_lock(&out.dir).unwrap();
    assert_eq!(lock.schema_version, 1);
    assert_eq!(lock.job_name, "locked");
    assert_eq!(lock.bench.commit.as_deref(), Some("abc123def"));
    assert_eq!(lock.trials.len(), 4);
    assert_eq!(lock.options.trials_per_task, 2);
    assert_eq!(lock.options.verifier_timeout_multiplier, 2.0);
    assert_eq!(
        lock.options.agent_env_keys,
        vec!["SECRET_TOKEN".to_string()]
    );
    assert_eq!(lock.dataset.as_ref().unwrap().name, "demo");
    assert_eq!(lock.job_files.len(), 1);
    assert_eq!(
        lock.job_files[0].fetched_from.as_deref(),
        Some("https://raw.githubusercontent.com/o/r/main/job.yaml")
    );
    assert_eq!(lock.job_files[0].digest, "sha256:abc");
    let first = &lock.trials[0];
    assert_eq!(first.task.id, "fixtures/alpha");
    assert_eq!(first.task.format, "task-md");
    assert_eq!(first.task.registry_digest.as_deref(), Some("sha256:pinned"));
    assert_eq!(first.image.source, "pull");
    assert_eq!(first.image.reference.as_deref(), Some("ubuntu:24.04"));
    assert_eq!(first.sandbox, "docker");
    let text = std::fs::read_to_string(out.dir.join("lock.json")).unwrap();
    assert!(
        !text.contains("do-not-write"),
        "lock.json must not hold env values"
    );
    // Each trial's result.json names the dataset.
    let r = &out.records[0];
    let v: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(out.dir.join(&r.rollout_name).join("result.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(v["dataset_name"], "demo");
    assert_eq!(v["dataset_version"], "1.0");
    assert_eq!(v["registry_digest"], "sha256:pinned");
    let back = benchflow_run::read::read_trial(&out.dir.join(&r.rollout_name)).unwrap();
    assert_eq!(
        back.dataset.unwrap().registry_digest.as_deref(),
        Some("sha256:pinned")
    );
}

#[tokio::test]
async fn rerun_failed_runs_failed_trials_again_and_keeps_the_old_ones() {
    let tmp = tempfile::tempdir().unwrap();
    let t = Arc::new(task(&tmp.path().join("alpha"), "alpha"));
    let provider = Arc::new(FakeProvider {
        worlds: Mutex::new(Vec::new()),
        writable_path: None,
    });
    let spec = JobSpec {
        carry: Vec::new(),
        units: vec![(t.clone(), vec![Attempt::Oracle, Attempt::Nop])],
        trials: 1,
        jobs_dir: tmp.path().join("jobs"),
        job_name: Some("rerun".into()),
        fresh: false,
        concurrency: 2,
        source: serde_json::json!({}),
    };
    job::run(spec.clone(), runtime(provider.clone()))
        .await
        .unwrap();
    assert_eq!(provider.worlds.lock().unwrap().len(), 2);
    let options = RunOptions {
        rerun: benchflow_run::config::Rerun::parse("failed").unwrap(),
        ..RunOptions::default()
    };
    let again = job::run(spec, runtime_with(provider.clone(), options))
        .await
        .unwrap();
    assert_eq!(again.reused, 1, "the passing oracle trial is reused");
    assert_eq!(
        provider.worlds.lock().unwrap().len(),
        3,
        "only the failed nop trial ran again"
    );
    let kept: Vec<_> = std::fs::read_dir(again.dir.join(".rerun"))
        .unwrap()
        .collect();
    assert_eq!(kept.len(), 1, "the earlier attempt is kept under .rerun/");
    assert_eq!(benchflow_run::read::read_job(&again.dir).records.len(), 2);
}

#[tokio::test]
async fn regrade_rescores_saved_outputs_in_a_new_job_folder() {
    let tmp = tempfile::tempdir().unwrap();
    let t = Arc::new(task(&tmp.path().join("alpha"), "alpha"));
    let provider = Arc::new(FakeProvider {
        worlds: Mutex::new(Vec::new()),
        writable_path: None,
    });
    let spec = JobSpec {
        carry: Vec::new(),
        units: vec![(t.clone(), vec![Attempt::Oracle, Attempt::Nop])],
        trials: 1,
        jobs_dir: tmp.path().join("jobs"),
        job_name: Some("graded".into()),
        fresh: false,
        concurrency: 2,
        source: serde_json::json!({"targets": ["alpha"]}),
    };
    let out = job::run(spec, runtime(provider.clone())).await.unwrap();
    let oracle = out.records.iter().find(|r| r.agent == "oracle").unwrap();
    let nop = out.records.iter().find(|r| r.agent == "nop").unwrap();
    // Only the oracle trial saved its outputs.
    let saved = out.dir.join(&oracle.rollout_name).join("outputs/app");
    std::fs::create_dir_all(&saved).unwrap();
    std::fs::write(saved.join("answer.txt"), "42").unwrap();

    let load = |_: &Path| Ok(t.clone());
    let plan = benchflow_run::regrade::plan_with(&out.dir, &load).unwrap();
    assert_eq!(plan.regradable.len(), 1);
    assert_eq!(plan.not_regradable.len(), 1);
    assert_eq!(plan.not_regradable[0].trial, nop.rollout_name);
    assert!(plan.not_regradable[0].next.contains("--save-outputs"));

    let before = provider.worlds.lock().unwrap().len();
    let re = benchflow_run::regrade::run(plan, runtime(provider.clone()), 2)
        .await
        .unwrap();
    assert_eq!(re.dir, tmp.path().join("jobs/graded-regrade-1"));
    assert_eq!(
        provider.worlds.lock().unwrap().len(),
        before + 1,
        "one fresh sandbox"
    );
    // The fake's fresh sandbox never saw the oracle run, so the reward drops.
    assert_eq!(re.changes.len(), 1);
    let c = &re.changes[0];
    assert_eq!(
        (c.old_reward, c.new_reward, c.changed),
        (Some(1.0), Some(0.0), true)
    );
    assert_eq!(re.changed(), 1);
    let rec = &re.records[0];
    assert_eq!(rec.agent, "oracle", "the record keeps who made the outputs");
    assert!(rec.defenses.contains_key("separate_verifier"));
    let eval: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(re.dir.join("evaluation.json")).unwrap())
            .unwrap();
    assert_eq!(eval["regrade_of"], serde_json::json!(out.dir));
    let lock = benchflow_run::read::read_lock(&re.dir).unwrap();
    assert_eq!(lock.regrade_of.as_deref(), Some(out.dir.as_path()));
    assert_eq!(lock.trials[0].attempt.agent, "oracle");
    assert_eq!(re.summary["regrade"]["changed"], 1);
    assert!(
        re.dir
            .join(&rec.rollout_name)
            .join("trajectory/acp_trajectory.jsonl")
            .is_file()
    );
    // A second regrade gets the next folder.
    let again = benchflow_run::regrade::plan_with(&out.dir, &load).unwrap();
    let re2 = benchflow_run::regrade::run(again, runtime(provider), 1)
        .await
        .unwrap();
    assert_eq!(re2.name, "graded-regrade-2");
}

/// E5 (the VM's Docker daemon killed mid-run): `useradd` failed because
/// the daemon was gone, and the trial was R104, the task's ("the image
/// needs a shell and useradd"). The backend cannot be asked any more, so
/// it is the sandbox's, and rerun.
#[tokio::test]
async fn a_daemon_lost_during_setup_is_the_sandboxs() {
    let tmp = tempfile::tempdir().unwrap();
    let t = Arc::new(task(&tmp.path().join("daemon-gone"), "daemon-gone"));
    let provider = Arc::new(FakeProvider {
        worlds: Mutex::new(Vec::new()),
        writable_path: None,
    });
    let spec = JobSpec {
        carry: Vec::new(),
        units: vec![(t, vec![Attempt::Oracle])],
        trials: 1,
        jobs_dir: tmp.path().join("jobs"),
        job_name: Some("lost".into()),
        fresh: true,
        concurrency: 1,
        source: serde_json::json!({"targets": ["daemon-gone"]}),
    };
    let out = job::run(spec, runtime(provider)).await.unwrap();
    let p = out.records[0].problem.clone().expect("a problem");
    assert_eq!(
        (p.code.as_str(), p.owner),
        ("R103", benchflow_run::outcome::Owner::Sandbox),
        "{p:?}"
    );
}

/// E5 (corrupted job folders, live on Docker): a saved output cut or
/// deleted after the run was regraded as it was, and the trial scored 0
/// with nothing saying why. custody.json's counts now catch it.
#[tokio::test]
async fn regrade_refuses_outputs_changed_since_the_run() {
    let tmp = tempfile::tempdir().unwrap();
    let t = Arc::new(task(&tmp.path().join("alpha"), "alpha"));
    let provider = Arc::new(FakeProvider {
        worlds: Mutex::new(Vec::new()),
        writable_path: None,
    });
    let spec = JobSpec {
        carry: Vec::new(),
        units: vec![(t.clone(), vec![Attempt::Oracle])],
        trials: 1,
        jobs_dir: tmp.path().join("jobs"),
        job_name: Some("graded".into()),
        fresh: false,
        concurrency: 1,
        source: serde_json::json!({"targets": ["alpha"]}),
    };
    let out = job::run(spec, runtime(provider.clone())).await.unwrap();
    let trial = out.dir.join(&out.records[0].rollout_name);
    std::fs::create_dir_all(trial.join("outputs/app")).unwrap();
    std::fs::write(trial.join("outputs/app/answer.txt"), "42").unwrap();
    // What the run saved, as collect_outputs records it.
    let custody_path = trial.join("custody.json");
    let mut custody: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&custody_path).unwrap()).unwrap();
    custody["saved_files"] = 1.into();
    custody["saved_bytes"] = 2.into();
    std::fs::write(&custody_path, custody.to_string()).unwrap();
    let load = |_: &Path| Ok(t.clone());
    let plan = benchflow_run::regrade::plan_with(&out.dir, &load).unwrap();
    assert_eq!(plan.regradable.len(), 1, "as saved: regradable");

    std::fs::write(trial.join("outputs/app/answer.txt"), "4").unwrap();
    let plan = benchflow_run::regrade::plan_with(&out.dir, &load).unwrap();
    assert!(plan.regradable.is_empty());
    assert!(
        plan.not_regradable[0]
            .reason
            .contains("changed since the run"),
        "{:?}",
        plan.not_regradable[0]
    );
    std::fs::remove_file(trial.join("outputs/app/answer.txt")).unwrap();
    std::fs::write(trial.join("outputs/app/other.txt"), "x").unwrap();
    let plan = benchflow_run::regrade::plan_with(&out.dir, &load).unwrap();
    assert!(plan.regradable.is_empty());

    // A file this machine may not read is this trial's refusal, not the
    // whole regrade's error (unless the tests run as root, who reads it).
    std::fs::remove_file(trial.join("outputs/app/other.txt")).unwrap();
    let answer = trial.join("outputs/app/answer.txt");
    std::fs::write(&answer, "42").unwrap();
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&answer, std::fs::Permissions::from_mode(0o000)).unwrap();
    if std::fs::File::open(&answer).is_err() {
        let plan = benchflow_run::regrade::plan_with(&out.dir, &load).unwrap();
        assert!(plan.regradable.is_empty());
        assert!(
            plan.not_regradable[0].reason.contains("cannot all be read"),
            "{:?}",
            plan.not_regradable[0]
        );
    }
    std::fs::set_permissions(&answer, std::fs::Permissions::from_mode(0o644)).unwrap();
}

/// A control's script is a path in the package (`controls/near-miss.sh`), so
/// it must be read from the task's folder, not the folder `bench` runs in.
/// Guards the fix for `bench test tasks/` reporting every control as
/// unreadable unless it was started inside the task folder.
#[tokio::test]
async fn a_control_is_read_from_the_task_folder() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("delta");
    let mut t = task(&root, "delta");
    std::fs::create_dir_all(root.join("controls")).unwrap();
    std::fs::write(root.join("controls/near-miss.sh"), "#!/bin/bash\n").unwrap();
    let control = Control {
        id: "near-miss".into(),
        script: PathBuf::from("controls/near-miss.sh"),
        max_reward: None,
    };
    t.controls.push(control.clone());
    let provider = Arc::new(FakeProvider {
        worlds: Mutex::new(Vec::new()),
        writable_path: None,
    });
    let spec = JobSpec {
        carry: Vec::new(),
        units: vec![(Arc::new(t), vec![Attempt::Control(control)])],
        trials: 1,
        jobs_dir: tmp.path().join("jobs"),
        job_name: None,
        fresh: true,
        concurrency: 1,
        source: serde_json::json!({}),
    };
    let out = job::run(spec, runtime(provider)).await.unwrap();
    let r = &out.records[0];
    assert_eq!(r.status, Status::Scored, "{:?}", r.problem);
    assert_eq!(r.passed, Some(false));
}

/// An image that lets others write a folder on PATH (here /usr/local/bin)
/// gets it locked before the agent runs, and the trial says so: the
/// path-shadow probe on Daytona replaced /usr/local/bin/sh, after which the
/// sandbox could run no command at all, and a root command could have run
/// the agent's program.
#[tokio::test]
async fn a_writable_folder_on_path_is_locked_and_reported() {
    let tmp = tempfile::tempdir().unwrap();
    let t = Arc::new(task(&tmp.path().join("eps"), "eps"));
    let run = |writable: Option<&str>| {
        let provider = Arc::new(FakeProvider {
            worlds: Mutex::new(Vec::new()),
            writable_path: writable.map(str::to_string),
        });
        let spec = JobSpec {
            carry: Vec::new(),
            units: vec![(t.clone(), vec![Attempt::Nop])],
            trials: 1,
            jobs_dir: tmp.path().join("jobs"),
            job_name: None,
            fresh: true,
            concurrency: 1,
            source: serde_json::json!({}),
        };
        async move { job::run(spec, runtime(provider)).await.unwrap() }
    };
    let out = run(Some("/usr/local/bin")).await;
    let d = &out.records[0].defenses["locked_system_path"];
    assert_eq!(d.state, DefenseState::Failed);
    assert!(
        d.note.as_deref().unwrap().contains("/usr/local/bin"),
        "{d:?}"
    );
    let out = run(None).await;
    let d = &out.records[0].defenses["locked_system_path"];
    assert_eq!(d.state, DefenseState::Enforced);
}

/// After the agent's turn, a folder on the image's PATH may be the agent's
/// (here a venv in the workdir). Every root program the runner sends then
/// finds its tools in the system folders only; the programs that judge the
/// image's PATH get it in a variable of their own.
#[tokio::test]
async fn root_programs_never_look_programs_up_on_the_images_path() {
    let tmp = tempfile::tempdir().unwrap();
    let t = Arc::new(task(&tmp.path().join("venv"), "venv"));
    let provider = Arc::new(FakeProvider {
        worlds: Mutex::new(Vec::new()),
        writable_path: None,
    });
    let spec = JobSpec {
        units: vec![(t, vec![Attempt::Nop])],
        trials: 1,
        jobs_dir: tmp.path().join("jobs"),
        job_name: None,
        fresh: true,
        concurrency: 1,
        source: serde_json::json!({}),
        carry: Vec::new(),
    };
    let out = job::run(spec, runtime(provider.clone())).await.unwrap();
    assert_eq!(out.records.len(), 1);
    let worlds = provider.worlds.lock().unwrap();
    let w = worlds[0].lock().unwrap();
    let kill = w
        .execs
        .iter()
        .position(|(_, c, _)| c.contains("kill -9 -1"))
        .expect("the kill before grading ran");
    let system = benchflow_sandbox::shell::SYSTEM_PATH;
    let mut judged = false;
    for (user, c, env) in &w.execs[kill..] {
        if user.as_deref() != Some("root") {
            continue;
        }
        let path = env.get("PATH").map(String::as_str);
        if c.contains("/verifier/test.sh") || c.contains("BF venv-done") {
            // The verifier (and the check of its environment) gets the
            // trusted PATH the runner computed.
            assert_eq!(path, Some("/usr/local/bin:/usr/bin:/bin"), "{c}");
            continue;
        }
        assert_eq!(path, Some(system), "{c}");
        if c.contains("BF path") {
            judged = true;
            assert_eq!(
                env.get("BENCHFLOW_IMAGE_PATH").map(String::as_str),
                Some("/app/.venv/bin:/usr/local/bin:/usr/bin:/bin")
            );
        }
    }
    assert!(judged, "the trusted PATH check ran after the agent");
}

/// The image's environment may name a library or a start-up file in the
/// workdir (`LD_PRELOAD`, `BASH_ENV`, ...). Every one of BenchFlow's own
/// root programs runs without the loader variables, before and after the
/// agent's turn.
#[tokio::test]
async fn root_programs_run_without_the_loader_variables() {
    let tmp = tempfile::tempdir().unwrap();
    let t = Arc::new(task(&tmp.path().join("venv"), "venv"));
    let provider = Arc::new(FakeProvider {
        worlds: Mutex::new(Vec::new()),
        writable_path: None,
    });
    let spec = JobSpec {
        units: vec![(t, vec![Attempt::Nop])],
        trials: 1,
        jobs_dir: tmp.path().join("jobs"),
        job_name: None,
        fresh: true,
        concurrency: 1,
        source: serde_json::json!({}),
        carry: Vec::new(),
    };
    let out = job::run(spec, runtime(provider.clone())).await.unwrap();
    assert_eq!(out.records.len(), 1);
    let worlds = provider.worlds.lock().unwrap();
    let w = worlds[0].lock().unwrap();
    let system = benchflow_sandbox::shell::SYSTEM_PATH;
    let mut own = 0;
    for ((user, c, env), unset) in w.execs.iter().zip(&w.unsets) {
        if user.as_deref() != Some("root") || env.get("PATH").map(String::as_str) != Some(system) {
            continue;
        }
        own += 1;
        for v in benchflow_sandbox::shell::LOADER_VARS {
            assert!(unset.iter().any(|u| u == v), "{v} is kept for: {c}");
        }
    }
    assert!(own >= 5, "only {own} of BenchFlow's own root commands ran");
    // The verifier's shell, and the check of its environment, start
    // without them too.
    let unset_of = |needle: &str| {
        w.execs
            .iter()
            .zip(&w.unsets)
            .find(|((_, c, _), _)| c.contains(needle))
            .map(|(_, u)| u.clone())
            .unwrap_or_else(|| panic!("no command with {needle}"))
    };
    for needle in ["/verifier/test.sh", "BF venv-done"] {
        let unset = unset_of(needle);
        for v in benchflow_sandbox::shell::LOADER_VARS {
            assert!(unset.iter().any(|u| u == v), "{v} is kept for {needle}");
        }
    }
    // The check still says what the image set, read before the agent ran.
    let note = out.records[0].defenses["clean_verifier_env"]
        .note
        .clone()
        .unwrap_or_default();
    assert!(
        note.contains("unset from the sandbox's own environment: BASH_ENV"),
        "{note}"
    );
}

/// With `[verifier] network = "none"`, the verifier stage's network
/// self-test runs after the agent's turn as the verifier's user (root). It
/// must run with the verifier's trusted `PATH` (the fake's `BF path`
/// answer), not the image's, and without the loader variables.
#[tokio::test]
async fn the_verifier_network_self_test_runs_on_the_trusted_path() {
    let tmp = tempfile::tempdir().unwrap();
    let mut t = task(&tmp.path().join("netnone"), "netnone");
    t.sandbox.network = Network::None;
    t.verifier.network = Some(Network::None);
    let provider = Arc::new(FakeProvider {
        worlds: Mutex::new(Vec::new()),
        writable_path: None,
    });
    let spec = JobSpec {
        units: vec![(Arc::new(t), vec![Attempt::Nop])],
        trials: 1,
        jobs_dir: tmp.path().join("jobs"),
        job_name: None,
        fresh: true,
        concurrency: 1,
        source: serde_json::json!({}),
        carry: Vec::new(),
    };
    let out = job::run(spec, runtime(provider.clone())).await.unwrap();
    assert_eq!(out.records.len(), 1);
    let worlds = provider.worlds.lock().unwrap();
    let w = worlds[0].lock().unwrap();
    // The self-test script defines `bf_try()`; the verifier stage's runs
    // as root, the agent stage's as the agent user.
    let tests: Vec<_> = w
        .execs
        .iter()
        .zip(&w.unsets)
        .filter(|((_, c, _), _)| c.contains("bf_try()"))
        .collect();
    let verifier = tests
        .iter()
        .find(|((user, _, _), _)| user.as_deref() == Some("root"))
        .unwrap_or_else(|| panic!("no verifier-stage self-test: {:?}", out.records[0].problem));
    let ((_, _, env), unset) = verifier;
    assert_eq!(
        env.get("PATH").map(String::as_str),
        Some("/usr/local/bin:/usr/bin:/bin"),
        "the verifier self-test's PATH"
    );
    for v in benchflow_sandbox::shell::LOADER_VARS {
        assert!(unset.iter().any(|u| u == v), "{v} kept for the self-test");
    }
    // It starts in `/`, not the agent's workdir.
    let cwds: Vec<_> = w
        .cwds
        .iter()
        .filter(|(c, _)| c.contains("bf_try()"))
        .map(|(_, cwd)| cwd.clone())
        .collect();
    assert!(
        cwds.iter().all(|c| c.as_deref() == Some("/")),
        "self-test folders: {cwds:?}"
    );
}

/// With no answer from the trusted PATH check after the agent's turn, the
/// verifier-stage self-test does not run on the image's `PATH`: the trial
/// stops, as the verifier's own check does.
#[tokio::test]
async fn the_verifier_network_self_test_needs_the_trusted_path() {
    let tmp = tempfile::tempdir().unwrap();
    let mut t = task(&tmp.path().join("nopath"), "nopath");
    t.sandbox.network = Network::None;
    t.verifier.network = Some(Network::None);
    let provider = Arc::new(FakeProvider {
        worlds: Mutex::new(Vec::new()),
        writable_path: None,
    });
    let spec = JobSpec {
        units: vec![(Arc::new(t), vec![Attempt::Nop])],
        trials: 1,
        jobs_dir: tmp.path().join("jobs"),
        job_name: None,
        fresh: true,
        concurrency: 1,
        source: serde_json::json!({}),
        carry: Vec::new(),
    };
    let out = job::run(spec, runtime(provider.clone())).await.unwrap();
    let r = &out.records[0];
    let p = r.problem.as_ref().expect("the trial stops");
    assert_eq!(p.code, "R307", "{p:?}");
    assert!(p.message.contains("network self-test"), "{}", p.message);
    let worlds = provider.worlds.lock().unwrap();
    let w = worlds[0].lock().unwrap();
    assert!(
        !w.execs
            .iter()
            .any(|(user, c, _)| c.contains("bf_try()") && user.as_deref() == Some("root")),
        "the verifier-stage self-test ran without a trusted PATH"
    );
}

/// Read-only package files are recorded as held only when no folder above
/// them is the agent's; the verifier's mount is claimed (links and files
/// the agent left there removed) before the verifier is uploaded into it.
#[tokio::test]
async fn read_only_mounts_and_the_verifier_mount_are_checked_after_the_agent() {
    let tmp = tempfile::tempdir().unwrap();
    let mut units = Vec::new();
    for (name, target) in [("inwork", "/app/data"), ("outside", "/opt/data")] {
        let root = tmp.path().join(name);
        let mut t = task(&root, name);
        std::fs::create_dir_all(root.join("data")).unwrap();
        std::fs::write(root.join("data/table.csv"), "a,b\n").unwrap();
        t.extras.mounts = vec![benchflow_task::extras::Mount {
            source: "data".into(),
            target: target.into(),
        }];
        units.push((Arc::new(t), vec![Attempt::Nop]));
    }
    let provider = Arc::new(FakeProvider {
        worlds: Mutex::new(Vec::new()),
        writable_path: None,
    });
    let spec = JobSpec {
        units,
        trials: 1,
        jobs_dir: tmp.path().join("jobs"),
        job_name: None,
        fresh: true,
        concurrency: 1,
        source: serde_json::json!({}),
        carry: Vec::new(),
    };
    let out = job::run(spec, runtime(provider.clone())).await.unwrap();
    for r in &out.records {
        let d = &r.defenses["read_only_mounts"];
        if r.task_name == "inwork" {
            assert_eq!(d.state, DefenseState::Failed, "{d:?}");
            assert!(d.note.as_deref().unwrap().contains("/app/data"), "{d:?}");
        } else {
            assert_eq!(d.state, DefenseState::Enforced, "{d:?}");
        }
    }
    for w in provider.worlds.lock().unwrap().iter() {
        let w = w.lock().unwrap();
        let kill = w
            .commands
            .iter()
            .position(|c| c.contains("kill -9 -1"))
            .unwrap();
        assert!(
            w.commands[kill..].iter().any(|c| c.contains("BF claimed")
                && c.contains("t=/verifier")
                && c.contains("BF planted")),
            "the verifier's mount was not claimed after the agent"
        );
    }
}

#[tokio::test]
async fn oracle_and_nop_trials_and_their_job_get_trainer_files() {
    let tmp = tempfile::tempdir().unwrap();
    let t = Arc::new(task(&tmp.path().join("alpha"), "alpha"));
    let provider = Arc::new(FakeProvider {
        worlds: Mutex::new(Vec::new()),
        writable_path: None,
    });
    let spec = JobSpec {
        carry: Vec::new(),
        units: vec![(t, vec![Attempt::Oracle, Attempt::Nop])],
        trials: 1,
        jobs_dir: tmp.path().join("jobs"),
        job_name: Some("trainer".into()),
        fresh: false,
        concurrency: 2,
        source: serde_json::json!({"path": tmp.path()}),
    };
    let out = job::run(spec, runtime(provider)).await.unwrap();
    let read = |path: &Path| std::fs::read_to_string(path).unwrap();
    let mut lines = Vec::new();
    for r in &out.records {
        let dir = out.dir.join(&r.rollout_name);
        let verifiers = read(&dir.join("trainer/verifiers.jsonl"));
        let record: serde_json::Value = serde_json::from_str(&verifiers).unwrap();
        assert_eq!(record["reward"], serde_json::json!(r.reward.unwrap()));
        assert_eq!(record["info"]["task_id"], "alpha");
        let atif: serde_json::Value =
            serde_json::from_str(&read(&dir.join("trainer/atif.json"))).unwrap();
        assert_eq!(atif["session_id"], serde_json::json!(r.rollout_name));
        let steps = atif["steps"].as_array().unwrap();
        let adp = read(&dir.join("trainer/adp.jsonl"));
        if r.agent == "oracle" {
            // The Rust oracle's event carries `timed_out`, which 0.8's did
            // not; it is still the oracle step, in all three files.
            assert_eq!(
                steps.last().unwrap()["message"],
                "[oracle: oracle/solve.sh]"
            );
            assert!(
                verifiers.contains("[oracle: oracle/solve.sh]"),
                "{verifiers}"
            );
            assert!(
                adp.contains(r#""content": "[oracle: oracle/solve.sh]", "reward": 1.0"#),
                "{adp}"
            );
        } else {
            // The nop's event exports nothing; the prompt is the only step,
            // and the reward goes in details.
            assert_eq!(steps.len(), 1);
            assert!(adp.contains(r#""terminal_reward": 0.0"#), "{adp}");
        }
        lines.push((r.rollout_name.clone(), verifiers, adp));
    }
    // The job's files: each trial's line, in trial-name order.
    lines.sort();
    let verifiers: String = lines.iter().map(|(_, v, _)| v.as_str()).collect();
    let adp: String = lines.iter().map(|(_, _, a)| a.as_str()).collect();
    assert_eq!(read(&out.dir.join("verifiers.jsonl")), verifiers);
    assert_eq!(read(&out.dir.join("adp.jsonl")), adp);
}

/// A build's context and Dockerfile are paths in the package, and the split
/// reader gives them relative (`environment/Dockerfile`). They resolve
/// against the task's folder, so two such tasks get their own images.
/// Guards the e2e-docker fix: `bench test` on hello-split and hello-v06 ran
/// `docker build ... environment` from the current folder, and both tasks
/// shared one cached failure.
#[test]
fn relative_build_paths_resolve_against_the_task_folder() {
    let tmp = tempfile::tempdir().unwrap();
    let specs: Vec<_> = ["one", "two"]
        .iter()
        .map(|n| {
            let root = tmp.path().join(n);
            let mut t = task(&root, n);
            std::fs::create_dir_all(root.join("environment")).unwrap();
            std::fs::write(
                root.join("environment/Dockerfile"),
                format!("FROM ubuntu:24.04\nRUN echo {n}\n"),
            )
            .unwrap();
            t.sandbox.image = ImageRef::Build {
                context: PathBuf::from("environment"),
                dockerfile: PathBuf::from("environment/Dockerfile"),
            };
            (root, benchflow_run::images::spec_for(&t, Vec::new()))
        })
        .collect();
    for (root, spec) in &specs {
        match &spec.source {
            benchflow_sandbox::ImageSource::Build {
                context,
                dockerfile,
            } => {
                assert_eq!(context, &root.join("environment"));
                assert_eq!(dockerfile, &root.join("environment/Dockerfile"));
            }
            other => panic!("not a build: {other:?}"),
        }
    }
    assert_ne!(specs[0].1.digest, specs[1].1.digest);
}

/// The split and 0.8 readers give the oracle and verifier folders relative
/// to the task (`solution`, `tests`); the trial uploads them from the task's
/// folder. Guards the e2e-docker fix: every hello-split and hello-v06 trial
/// failed with "reading tests to upload it: No such file or directory".
#[tokio::test]
async fn relative_oracle_and_verifier_folders_upload_from_the_task() {
    let tmp = tempfile::tempdir().unwrap();
    let mut t = task(&tmp.path().join("epsilon"), "epsilon");
    t.verifier.dir = PathBuf::from("verifier");
    t.oracle.as_mut().unwrap().dir = PathBuf::from("oracle");
    let provider = Arc::new(FakeProvider {
        worlds: Mutex::new(Vec::new()),
        writable_path: None,
    });
    let spec = JobSpec {
        carry: Vec::new(),
        units: vec![(Arc::new(t), vec![Attempt::Oracle, Attempt::Nop])],
        trials: 1,
        jobs_dir: tmp.path().join("jobs"),
        job_name: None,
        fresh: true,
        concurrency: 2,
        source: serde_json::json!({}),
    };
    let out = job::run(spec, runtime(provider)).await.unwrap();
    for r in &out.records {
        assert_eq!(r.status, Status::Scored, "{} {:?}", r.agent, r.problem);
        assert_eq!(r.passed, Some(r.agent == "oracle"));
    }
}

/// A 0 whose verifier output shows the environment failing keeps its score
/// and carries a warning in result.json and summary.json; `bench test` says
/// a failing oracle means the task is broken, not the agent.
#[tokio::test]
async fn a_broken_environment_is_flagged_but_keeps_its_score() {
    let tmp = tempfile::tempdir().unwrap();
    let t = Arc::new(task(&tmp.path().join("apt-gone"), "apt-gone"));
    let provider = Arc::new(FakeProvider {
        worlds: Mutex::new(Vec::new()),
        writable_path: None,
    });
    let spec = JobSpec {
        carry: Vec::new(),
        units: vec![(t.clone(), vec![Attempt::Oracle, Attempt::Nop])],
        trials: 1,
        jobs_dir: tmp.path().join("jobs"),
        job_name: None,
        fresh: true,
        concurrency: 2,
        source: serde_json::json!({}),
    };
    let out = job::run(spec, runtime(provider)).await.unwrap();
    let nop = out.records.iter().find(|r| r.agent == "nop").unwrap();
    assert_eq!((nop.status, nop.reward), (Status::Scored, Some(0.0)));
    assert_eq!(
        nop.environment_warnings.len(),
        1,
        "{:?}",
        nop.environment_warnings
    );
    assert!(nop.environment_warnings[0].contains("Unable to locate package stockfish"));
    let oracle = out.records.iter().find(|r| r.agent == "oracle").unwrap();
    assert!(
        oracle.environment_warnings.is_empty(),
        "a pass is not flagged"
    );
    let warned = out.summary["environment_warnings"].as_array().unwrap();
    assert_eq!(warned.len(), 1);
    assert_eq!(warned[0]["trial"], serde_json::json!(nop.rollout_name));
    // result.json keeps it, and reading the job back finds it.
    let back = benchflow_run::read::trial_dirs(&out.dir)
        .filter_map(|d| benchflow_run::read::read_trial(&d))
        .find(|r| r.agent == "nop")
        .unwrap();
    assert_eq!(back.environment_warnings, nop.environment_warnings);

    // An oracle that fails the same way: the task is broken, not the agent.
    let mut failed = nop.clone();
    failed.agent = "oracle".into();
    let records = vec![&failed, nop];
    let v = benchflow_run::testplan::judge(&t, &records, &[]);
    let check = v.checks.iter().find(|c| c.attempt == "oracle").unwrap();
    assert!(!check.ok);
    assert!(
        check.says.contains("the task is broken, not the agent"),
        "{}",
        check.says
    );
    assert!(check.says.contains("stockfish"), "{}", check.says);
}

/// `bench test` names what a weak defense noted, not only its name, so a
/// task whose image ships /solution says so. Guards the e2e-docker fix: the
/// ships-solution fixture reported only "defenses not enforced in some
/// trials: hidden_verifier".
#[tokio::test]
async fn a_weak_defense_says_what_the_trial_noted() {
    let tmp = tempfile::tempdir().unwrap();
    let t = Arc::new(task(&tmp.path().join("ships"), "ships"));
    let provider = Arc::new(FakeProvider {
        worlds: Mutex::new(Vec::new()),
        writable_path: None,
    });
    let spec = JobSpec {
        carry: Vec::new(),
        units: vec![(t.clone(), vec![Attempt::Oracle, Attempt::Nop])],
        trials: 1,
        jobs_dir: tmp.path().join("jobs"),
        job_name: None,
        fresh: true,
        concurrency: 2,
        source: serde_json::json!({}),
    };
    let out = job::run(spec, runtime(provider)).await.unwrap();
    let records: Vec<_> = out.records.iter().collect();
    let v = benchflow_run::testplan::judge(&t, &records, &[]);
    assert!(v.ok, "{:?}", v.checks);
    // The fake backend installs no firewall, so the cloud metadata
    // endpoints are not refused there either (F1), and the record says so.
    assert_eq!(
        v.weak_defenses,
        vec!["cloud_metadata".to_string(), "hidden_verifier".to_string()]
    );
    assert_eq!(v.weak_details.len(), 2);
    let note = |name: &str| {
        v.weak_details
            .iter()
            .find(|d| d.name == name)
            .and_then(|d| d.note.clone())
    };
    assert_eq!(
        note("hidden_verifier").as_deref(),
        Some("the image ships /solution; locked to root before the agent ran")
    );
    assert!(
        note("cloud_metadata")
            .unwrap()
            .starts_with("the fake backend installs no firewall, so the agent can reach")
    );
}

/// When the agent replaces the shell, BenchFlow's root programs "succeed"
/// and print nothing. The trial must not be graded, and no defense may be
/// recorded as enforced. Guards the e2e-docker fix: on Docker, path-shadow
/// on the writable-bin fixture replaced /usr/local/bin/sh, the kill,
/// handoff, and PATH check never ran, and the record still said
/// kill_before_grading and clean_verifier_logs were enforced.
#[tokio::test]
async fn a_shell_that_does_not_report_back_is_not_graded() {
    let tmp = tempfile::tempdir().unwrap();
    let t = Arc::new(task(&tmp.path().join("shadowed"), "shadowed"));
    let provider = Arc::new(FakeProvider {
        worlds: Mutex::new(Vec::new()),
        writable_path: None,
    });
    let spec = JobSpec {
        carry: Vec::new(),
        units: vec![(t, vec![Attempt::Nop])],
        trials: 1,
        jobs_dir: tmp.path().join("jobs"),
        job_name: None,
        fresh: true,
        concurrency: 1,
        source: serde_json::json!({}),
    };
    let out = job::run(spec, runtime(provider)).await.unwrap();
    let r = &out.records[0];
    assert_eq!(r.status, Status::Unscored, "{:?}", r.problem);
    let p = r.problem.as_ref().unwrap();
    assert_eq!(p.code, "R307", "{p:?}");
    assert!(p.message.contains("did not report back"), "{}", p.message);
    let kill = &r.defenses["kill_before_grading"];
    assert_eq!(kill.state, benchflow_run::outcome::DefenseState::Failed);
    assert!(r.reward.is_none());
}

#[tokio::test]
async fn a_named_job_does_not_reuse_trials_of_a_changed_task() {
    let tmp = tempfile::tempdir().unwrap();
    let provider = Arc::new(FakeProvider {
        worlds: Mutex::new(Vec::new()),
        writable_path: None,
    });
    let spec = |t: TaskPackage| JobSpec {
        carry: Vec::new(),
        units: vec![(Arc::new(t), vec![Attempt::Oracle])],
        trials: 1,
        jobs_dir: tmp.path().join("jobs"),
        job_name: Some("named".into()),
        fresh: false,
        concurrency: 1,
        source: serde_json::json!({}),
    };
    let first = task(&tmp.path().join("delta"), "delta");
    let out = job::run(spec(first.clone()), runtime(provider.clone()))
        .await
        .unwrap();
    assert_eq!(out.records[0].status, Status::Scored);
    // The same job name after the task changed: its old trial is not this
    // task's, so it runs again.
    let mut changed = first;
    changed.digest = "sha256:delta-edited".into();
    let again = job::run(spec(changed), runtime(provider.clone()))
        .await
        .unwrap();
    assert_eq!(again.reused, 0, "a trial of the old task was reused");
    assert_eq!(again.records[0].task_digest, "sha256:delta-edited");
    assert_eq!(provider.worlds.lock().unwrap().len(), 2);
}

/// A regrade unpacks saved outputs as root at `/` in a fresh sandbox, so a
/// trial folder whose outputs reach outside the task's output paths (a
/// replaced `python3`, say) is not regraded.
#[tokio::test]
async fn regrade_refuses_outputs_outside_the_tasks_output_paths() {
    let tmp = tempfile::tempdir().unwrap();
    let t = Arc::new(task(&tmp.path().join("alpha"), "alpha"));
    let provider = Arc::new(FakeProvider {
        worlds: Mutex::new(Vec::new()),
        writable_path: None,
    });
    let spec = JobSpec {
        carry: Vec::new(),
        units: vec![(t.clone(), vec![Attempt::Oracle])],
        trials: 1,
        jobs_dir: tmp.path().join("jobs"),
        job_name: Some("graded".into()),
        fresh: false,
        concurrency: 1,
        source: serde_json::json!({}),
    };
    let out = job::run(spec, runtime(provider.clone())).await.unwrap();
    let trial = out.dir.join(&out.records[0].rollout_name);
    for (p, body) in [
        ("outputs/app/answer.txt", "42"),
        (
            "outputs/usr/bin/python3",
            "#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n",
        ),
    ] {
        let f = trial.join(p);
        std::fs::create_dir_all(f.parent().unwrap()).unwrap();
        std::fs::write(f, body).unwrap();
    }
    let load = |_: &Path| Ok(t.clone());
    let plan = benchflow_run::regrade::plan_with(&out.dir, &load).unwrap();
    assert_eq!(plan.regradable.len(), 0);
    assert!(
        plan.not_regradable[0].reason.contains("outside"),
        "{:?}",
        plan.not_regradable
    );
    // An archive with the same stray entry is refused too.
    std::fs::remove_dir_all(trial.join("outputs/usr")).unwrap();
    benchflow_run::regrade::pack_folder(&trial.join("outputs"), &trial.join("outputs.tar"))
        .unwrap();
    assert_eq!(
        benchflow_run::regrade::plan_with(&out.dir, &load)
            .unwrap()
            .regradable
            .len(),
        1
    );
    let mut b = tar::Builder::new(std::fs::File::create(trial.join("outputs.tar")).unwrap());
    let mut h = tar::Header::new_gnu();
    h.set_size(1);
    h.set_mode(0o755);
    b.append_data(&mut h, "usr/bin/python3", &b"x"[..]).unwrap();
    b.finish().unwrap();
    drop(b);
    assert_eq!(
        benchflow_run::regrade::plan_with(&out.dir, &load)
            .unwrap()
            .regradable
            .len(),
        0
    );
}

/// With a separate verifier the agent's sandbox is retired before grading;
/// `--keep on-failure` must still destroy it when the trial passes.
#[tokio::test]
async fn a_separate_verifier_run_destroys_the_agents_sandbox_too() {
    let tmp = tempfile::tempdir().unwrap();
    let mut t = task(&tmp.path().join("sep"), "sep");
    t.verifier.isolation = Isolation::Separate;
    let provider = Arc::new(FakeProvider {
        worlds: Mutex::new(Vec::new()),
        writable_path: None,
    });
    let options = RunOptions {
        keep: benchflow_run::config::Keep::OnFailure,
        ..RunOptions::default()
    };
    let spec = JobSpec {
        carry: Vec::new(),
        units: vec![(Arc::new(t), vec![Attempt::Oracle])],
        trials: 1,
        jobs_dir: tmp.path().join("jobs"),
        job_name: None,
        fresh: true,
        concurrency: 1,
        source: serde_json::json!({}),
    };
    let out = job::run(spec, runtime_with(provider.clone(), options))
        .await
        .unwrap();
    assert_eq!(
        out.records[0].passed,
        Some(true),
        "{:?}",
        out.records[0].problem
    );
    let worlds = provider.worlds.lock().unwrap();
    assert_eq!(worlds.len(), 2, "an agent sandbox and a verifier sandbox");
    assert!(
        worlds.iter().all(|w| w.lock().unwrap().destroyed),
        "a sandbox of a passing trial was left running"
    );
}

/// A trial retried after a transient failure starts from a clean folder: the
/// failed attempt's saved outputs are not mixed into the retry's (a regrade
/// would grade them), and the failed attempt is kept under `.rerun/`.
#[tokio::test]
async fn a_retried_trial_does_not_keep_the_failed_attempts_files() {
    let tmp = tempfile::tempdir().unwrap();
    let mut t = task(&tmp.path().join("flaky"), "flaky");
    t.sandbox.env.insert(
        "FAKE_FLAKY_MARKER".into(),
        tmp.path().join("failed-once").display().to_string(),
    );
    let provider = Arc::new(FakeProvider {
        worlds: Mutex::new(Vec::new()),
        writable_path: None,
    });
    let options = RunOptions {
        save_outputs: true,
        // No two-minute wait before the retry of an unexplained loss.
        unexplained_loss_retry_secs: 0,
        ..RunOptions::default()
    };
    let spec = JobSpec {
        carry: Vec::new(),
        units: vec![(Arc::new(t), vec![Attempt::Oracle])],
        trials: 1,
        jobs_dir: tmp.path().join("jobs"),
        job_name: Some("retried".into()),
        fresh: false,
        concurrency: 1,
        source: serde_json::json!({}),
    };
    let out = job::run(spec, runtime_with(provider.clone(), options))
        .await
        .unwrap();
    let rec = &out.records[0];
    assert_eq!(rec.status, Status::Scored, "{:?}", rec.problem);
    assert_eq!(provider.worlds.lock().unwrap().len(), 2, "one retry");
    let trial = out.dir.join(&rec.rollout_name);
    assert!(trial.join("outputs/app/answer.txt").is_file());
    assert!(
        !trial.join("outputs/app/first-attempt.txt").exists(),
        "the failed attempt's outputs were mixed into the retry's"
    );
    assert!(
        out.dir
            .join(".rerun")
            .join(format!("{}-1/result.json", rec.rollout_name))
            .is_file()
    );
}

/// Every trial records the confinement around its agent: system folders
/// locked, ld.so.preload checked, the agent first for the out-of-memory
/// killer, the memory limit, and a clean verifier environment. A user's
/// script (`--agent script:<file>`) runs as the agent, under the same
/// out-of-memory prefix, and the verifier starts with loader variables
/// unset.
#[tokio::test]
async fn confinement_is_recorded_and_a_user_script_runs_as_the_agent() {
    let tmp = tempfile::tempdir().unwrap();
    let t = Arc::new(task(&tmp.path().join("zeta"), "zeta"));
    let provider = Arc::new(FakeProvider {
        worlds: Mutex::new(Vec::new()),
        writable_path: None,
    });
    let script = benchflow_run::agent::UserScript {
        name: "wrong.sh".into(),
        path: PathBuf::from("wrong.sh"),
        bytes: Arc::new(b"#!/bin/bash\necho 'Hello' > /app/hello.txt\n".to_vec()),
    };
    let spec = JobSpec {
        carry: Vec::new(),
        units: vec![(t.clone(), vec![Attempt::Script(script)])],
        trials: 1,
        jobs_dir: tmp.path().join("jobs"),
        job_name: None,
        fresh: true,
        concurrency: 1,
        source: serde_json::json!({}),
    };
    let out = job::run(spec, runtime(provider.clone())).await.unwrap();
    let r = &out.records[0];
    assert_eq!(r.agent, "script:wrong.sh");
    assert_eq!(r.status, Status::Scored, "{:?}", r.problem);
    assert_eq!(r.passed, Some(false));
    for (name, state) in [
        ("locked_system_dirs", DefenseState::Enforced),
        ("ld_so_preload", DefenseState::Enforced),
        ("oom_order", DefenseState::Enforced),
        ("memory_limit", DefenseState::Enforced),
        ("clean_verifier_env", DefenseState::Enforced),
    ] {
        let d = r
            .defenses
            .get(name)
            .unwrap_or_else(|| panic!("missing defense {name}"));
        assert_eq!(d.state, state, "{name}: {:?}", d.note);
    }
    assert!(
        !r.defenses.contains_key("runtime"),
        "no runtime was asked for"
    );
    let world = provider.worlds.lock().unwrap()[0].clone();
    let commands = world.lock().unwrap().commands.clone();
    let attempt = commands
        .iter()
        .find(|c| c.contains("/tmp/.benchflow-script.sh"))
        .expect("the script ran");
    assert!(
        attempt.starts_with("{ echo 1000 > /proc/self/oom_score_adj; } 2>/dev/null\n"),
        "{attempt}"
    );
    let verifier = commands
        .iter()
        .find(|c| c.ends_with("exec /verifier/test.sh"))
        .expect("the verifier ran");
    assert!(
        verifier.starts_with("unset LD_PRELOAD LD_LIBRARY_PATH"),
        "{verifier}"
    );
    let lock: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out.dir.join("lock.json")).unwrap()).unwrap();
    assert_eq!(lock["trials"][0]["attempt"]["kind"], "script");
}

/// The verifier runs in the workdir in every task format, as BenchFlow 0.8
/// runs `test.sh` (in the sandbox's working folder). SkillsBench's
/// `video-silence-remover` (a 0.8 task.md) opens `compressed_video.mp4` by
/// relative path from `/root`; run from `/verifier` it failed the oracle.
#[tokio::test]
async fn the_verifier_runs_in_the_workdir_in_every_format() {
    let tmp = tempfile::tempdir().unwrap();
    let units = [Format::TaskMd, Format::TaskMdV06, Format::Split]
        .into_iter()
        .enumerate()
        .map(|(i, format)| {
            let mut t = task(&tmp.path().join(format!("t{i}")), &format!("t{i}"));
            t.format = format;
            t.sandbox.workdir = None;
            (Arc::new(t), vec![Attempt::Oracle])
        })
        .collect();
    let provider = Arc::new(FakeProvider {
        worlds: Mutex::new(Vec::new()),
        writable_path: None,
    });
    let spec = JobSpec {
        carry: Vec::new(),
        units,
        trials: 1,
        jobs_dir: tmp.path().join("jobs"),
        job_name: Some("verifier-cwd".into()),
        fresh: true,
        concurrency: 1,
        source: serde_json::json!({"path": tmp.path()}),
    };
    job::run(spec, runtime(provider.clone())).await.unwrap();
    let worlds = provider.worlds.lock().unwrap();
    assert_eq!(worlds.len(), 3);
    for w in worlds.iter() {
        let w = w.lock().unwrap();
        let cwd_of = |needle: &str| {
            w.cwds
                .iter()
                .find(|(c, _)| c.contains(needle))
                .and_then(|(_, cwd)| cwd.clone())
        };
        let verifier = cwd_of("/test.sh").expect("the verifier ran");
        assert_eq!(verifier, "/root", "{:?}", w.cwds);
        assert_eq!(Some(verifier), cwd_of("/oracle/solve.sh"));
    }
}

/// A `verifier.md` script strategy starts in the verifier's folder, as
/// BenchFlow 0.8 runs it with `cd <verifier folder> && <command>`: started in
/// the agent's workdir, a verifier's `python3 -` would import a `json.py` the
/// agent left there, and `pytest test_outputs.py` would find the agent's file.
#[tokio::test]
async fn a_verifier_md_script_starts_in_the_verifier_folder() {
    let tmp = tempfile::tempdir().unwrap();
    let units = [Format::TaskMdV06, Format::Split]
        .into_iter()
        .enumerate()
        .map(|(i, format)| {
            let mut t = task(&tmp.path().join(format!("t{i}")), &format!("t{i}"));
            t.format = format;
            t.sandbox.workdir = None;
            t.verifier.starts_in_mount = true;
            (Arc::new(t), vec![Attempt::Oracle])
        })
        .collect();
    let provider = Arc::new(FakeProvider {
        worlds: Mutex::new(Vec::new()),
        writable_path: None,
    });
    let spec = JobSpec {
        carry: Vec::new(),
        units,
        trials: 1,
        jobs_dir: tmp.path().join("jobs"),
        job_name: Some("verifier-md-cwd".into()),
        fresh: true,
        concurrency: 1,
        source: serde_json::json!({"path": tmp.path()}),
    };
    job::run(spec, runtime(provider.clone())).await.unwrap();
    let worlds = provider.worlds.lock().unwrap();
    assert_eq!(worlds.len(), 2);
    for w in worlds.iter() {
        let w = w.lock().unwrap();
        let cwd_of = |needle: &str| {
            w.cwds
                .iter()
                .find(|(c, _)| c.contains(needle))
                .and_then(|(_, cwd)| cwd.clone())
        };
        let verifier = cwd_of("/test.sh").expect("the verifier ran");
        assert_eq!(verifier, "/verifier", "{:?}", w.cwds);
        assert_eq!(cwd_of("/oracle/solve.sh").as_deref(), Some("/root"));
    }
}

/// A task that declares no workdir works in its image's WORKDIR, as BenchFlow
/// 0.8 runs it: SkillsBench's `econ-detrending-correlation` builds
/// `WORKDIR /root` and its solve.sh reads its spreadsheets by relative path, so
/// running it in `/app` failed the oracle.
#[tokio::test]
async fn a_task_without_a_workdir_works_in_the_image_workdir() {
    let tmp = tempfile::tempdir().unwrap();
    let mut declared = task(&tmp.path().join("declared"), "declared");
    declared.sandbox.workdir = Some("/app".into());
    let mut from_image = task(&tmp.path().join("imgroot"), "imgroot");
    from_image.sandbox.workdir = None;
    let mut unset = task(&tmp.path().join("unset"), "unset");
    unset.sandbox.workdir = None;
    let provider = Arc::new(FakeProvider {
        worlds: Mutex::new(Vec::new()),
        writable_path: None,
    });
    let spec = JobSpec {
        carry: Vec::new(),
        units: [declared, from_image, unset]
            .into_iter()
            .map(|t| (Arc::new(t), vec![Attempt::Oracle]))
            .collect(),
        trials: 1,
        jobs_dir: tmp.path().join("jobs"),
        job_name: Some("workdir".into()),
        fresh: true,
        concurrency: 1,
        source: serde_json::json!({"path": tmp.path()}),
    };
    let out = job::run(spec, runtime(provider.clone())).await.unwrap();
    for r in &out.records {
        assert_eq!(r.passed, Some(true), "{} {:?}", r.task_name, r.problem);
        // An agent user working in root's home is recorded as such.
        let root_home = r.defenses.get("root_home").map(|d| d.state);
        if r.task_name == "declared" {
            assert_eq!(root_home, None, "{:?}", r.defenses);
        } else {
            assert_eq!(root_home, Some(DefenseState::Off), "{:?}", r.defenses);
        }
    }
    let worlds = provider.worlds.lock().unwrap();
    let mut seen = Vec::new();
    for w in worlds.iter() {
        let w = w.lock().unwrap();
        let oracle_cwd = w
            .cwds
            .iter()
            .find(|(c, _)| c.contains("/oracle/solve.sh"))
            .and_then(|(_, cwd)| cwd.clone());
        let probed = w.commands.iter().any(|c| c.lines().last() == Some("pwd"));
        let setup = w
            .commands
            .iter()
            .find(|c| c.contains("useradd"))
            .cloned()
            .unwrap();
        seen.push((w.spec_workdir.clone(), probed, oracle_cwd.clone()));
        // BenchFlow's root-only state is never inside the agent's workdir.
        assert!(!setup.contains("/root/.benchflow"), "{setup}");
        // The agent user gets the same folder the oracle runs in.
        let wd = oracle_cwd.unwrap();
        assert!(
            setup.contains(&format!("w={wd}")) || setup.contains(&format!("w='{wd}'")),
            "{setup}"
        );
    }
    seen.sort();
    assert_eq!(
        seen,
        vec![
            // No declared workdir: the sandbox keeps the image's, and the
            // engine asks for it (`/` falls back to /root, as in 0.8).
            (None, true, Some("/root".to_string())),
            (None, true, Some("/root".to_string())),
            // A declared one is used as is, without asking.
            (Some("/app".to_string()), false, Some("/app".to_string())),
        ]
    );
}

/// Plan `units` against earlier jobs, as `bench run --diff` does.
fn diff_plan(
    units: &[(Arc<TaskPackage>, Vec<Attempt>)],
    options: &RunOptions,
    jobs: &[PathBuf],
) -> (
    benchflow_run::diff::DiffPlan,
    Vec<benchflow_run::diff::SourceTrial>,
) {
    use benchflow_run::{diff, lock};
    let planned = diff::planned_trials(units, 1, "next");
    let target = lock::build(
        lock::LockHeader {
            job_name: "next",
            fingerprint: "",
            source: &serde_json::Value::Null,
            trials_per_task: 1,
            concurrency: 1,
            regrade_of: None,
        },
        &planned,
        options,
    );
    let (found, skipped) = diff::load_sources(jobs);
    let plan = diff::plan(
        &planned,
        &target,
        jobs.to_vec(),
        &found,
        skipped,
        &options.rerun,
    );
    (plan, found)
}

#[tokio::test]
async fn diff_reuses_same_trials_regrades_changed_tests_and_reruns_the_rest() {
    use benchflow_run::diff::Action;
    let tmp = tempfile::tempdir().unwrap();
    let t = Arc::new(task(&tmp.path().join("alpha"), "alpha"));
    let provider = Arc::new(FakeProvider {
        worlds: Mutex::new(Vec::new()),
        writable_path: None,
    });
    let units = vec![(t.clone(), vec![Attempt::Oracle, Attempt::Nop])];
    let first = job::run(
        JobSpec {
            carry: Vec::new(),
            units: units.clone(),
            trials: 1,
            jobs_dir: tmp.path().join("jobs"),
            job_name: Some("first".into()),
            fresh: false,
            concurrency: 2,
            source: serde_json::json!({}),
        },
        runtime(provider.clone()),
    )
    .await
    .unwrap();
    let oracle = first.records.iter().find(|r| r.agent == "oracle").unwrap();
    // Only the oracle trial saved outputs.
    let saved = first.dir.join(&oracle.rollout_name).join("outputs/app");
    std::fs::create_dir_all(&saved).unwrap();
    std::fs::write(saved.join("answer.txt"), "42").unwrap();
    let jobs = vec![first.dir.clone()];
    let options = RunOptions::default();

    // The same task and settings: both reused, and nothing runs.
    let (plan, found) = diff_plan(&units, &options, &jobs);
    assert_eq!(plan.count(Action::Reuse), 2, "{plan:#?}");
    let before = provider.worlds.lock().unwrap().len();
    let second = job::run(
        JobSpec {
            carry: plan.carries(&found),
            units: units.clone(),
            trials: 1,
            jobs_dir: tmp.path().join("jobs"),
            job_name: Some("second".into()),
            fresh: false,
            concurrency: 2,
            source: serde_json::json!({}),
        },
        runtime(provider.clone()),
    )
    .await
    .unwrap();
    assert_eq!((second.reused, second.carried_reused), (2, 2));
    assert_eq!(provider.worlds.lock().unwrap().len(), before, "no sandbox");
    for r in &second.records {
        assert!(r.rollout_name.starts_with("alpha__"));
        assert!(
            second
                .dir
                .join(&r.rollout_name)
                .join("result.json")
                .is_file()
        );
    }
    // Guards 044f1d73: an earlier trial that ran with other agent options
    // (`--ak`) is not reused as if its settings were the same.
    let nop = first.records.iter().find(|r| r.agent == "nop").unwrap();
    let config_path = first.dir.join(&nop.rollout_name).join("config.json");
    let original = std::fs::read_to_string(&config_path).unwrap();
    let mut config: serde_json::Value = serde_json::from_str(&original).unwrap();
    config["agent_options"] = serde_json::json!({ "max_turns": 5 });
    std::fs::write(&config_path, config.to_string()).unwrap();
    let (plan, _) = diff_plan(&units, &options, &jobs);
    let nop_row = plan.trials.iter().find(|x| x.agent == "nop").unwrap();
    assert_eq!(nop_row.action, Action::Rerun, "{plan:#?}");
    assert!(nop_row.reason.contains("agent options differ"), "{plan:#?}");
    std::fs::write(&config_path, original).unwrap();
    let lock = benchflow_run::read::read_lock(&second.dir).unwrap();
    assert!(lock.trials.iter().all(|t| {
        t.carried_from
            .as_ref()
            .is_some_and(|c| c.action == "reuse" && c.trial_dir.starts_with(&first.dir))
    }));

    // The verifier changed (a new digest, same inputs): the oracle trial is
    // regraded from its outputs; the nop trial saved none, so it reruns.
    let mut changed = (*t).clone();
    changed.digest = "sha256:alpha-new-tests".into();
    let changed = Arc::new(changed);
    let units2 = vec![(changed.clone(), vec![Attempt::Oracle, Attempt::Nop])];
    let (plan, found) = diff_plan(&units2, &options, &jobs);
    let by = |agent: &str| plan.trials.iter().find(|x| x.agent == agent).unwrap();
    assert_eq!(by("oracle").action, Action::Regrade, "{plan:#?}");
    assert_eq!(by("nop").action, Action::Rerun);
    assert!(by("nop").reason.contains("saved no outputs"), "{plan:#?}");
    let before = provider.worlds.lock().unwrap().len();
    let third = job::run(
        JobSpec {
            carry: plan.carries(&found),
            units: units2,
            trials: 1,
            jobs_dir: tmp.path().join("jobs"),
            job_name: Some("third".into()),
            fresh: false,
            concurrency: 2,
            source: serde_json::json!({}),
        },
        runtime(provider.clone()),
    )
    .await
    .unwrap();
    assert_eq!((third.reused, third.carried_regraded), (0, 1));
    assert_eq!(
        provider.worlds.lock().unwrap().len(),
        before + 2,
        "a fresh sandbox for the regrade and one for the rerun"
    );
    let regraded = third.records.iter().find(|r| r.agent == "oracle").unwrap();
    assert!(regraded.defenses.contains_key("separate_verifier"));
    assert_eq!(regraded.task_digest, "sha256:alpha-new-tests");
    let lock = benchflow_run::read::read_lock(&third.dir).unwrap();
    let carried: Vec<_> = lock
        .trials
        .iter()
        .filter_map(|t| t.carried_from.as_ref())
        .collect();
    assert_eq!(carried.len(), 1);
    assert_eq!(carried[0].action, "regrade");
    // Resuming the third job reuses its own regraded trial.
    let again = job::run(
        JobSpec {
            carry: Vec::new(),
            units: vec![(changed.clone(), vec![Attempt::Oracle, Attempt::Nop])],
            trials: 1,
            jobs_dir: tmp.path().join("jobs"),
            job_name: Some("third".into()),
            fresh: false,
            concurrency: 2,
            source: serde_json::json!({}),
        },
        runtime(provider.clone()),
    )
    .await
    .unwrap();
    assert_eq!(again.reused, 2);

    // The instruction changed: what the agent works from differs.
    let mut reworded = (*t).clone();
    reworded.instruction = "Do the other thing.".into();
    reworded.digest = "sha256:alpha-reworded".into();
    let (plan, _) = diff_plan(
        &[(Arc::new(reworded), vec![Attempt::Oracle])],
        &options,
        &jobs,
    );
    assert_eq!(plan.trials[0].action, Action::Rerun);
    assert!(
        plan.trials[0].reason.contains("what the agent works from"),
        "{plan:#?}"
    );

    // A setting that changes the agent's run: rerun, naming it.
    let slower = RunOptions {
        idle_timeout: Duration::from_secs(1),
        ..RunOptions::default()
    };
    let (plan, _) = diff_plan(&units, &slower, &jobs);
    assert!(plan.trials.iter().all(|x| x.action == Action::Rerun));
    assert!(plan.trials[0].reason.contains("idle timeout"), "{plan:#?}");

    // A setting that changes only grading: regrade where outputs exist.
    let stricter = RunOptions {
        verifier_timeout_multiplier: 2.0,
        ..RunOptions::default()
    };
    let (plan, _) = diff_plan(&units, &stricter, &jobs);
    assert_eq!(by_agent(&plan, "oracle").action, Action::Regrade);
    assert!(
        by_agent(&plan, "oracle")
            .reason
            .contains("verifier timeout multiplier")
    );

    // --rerun failed: a failed earlier trial is not reused.
    let failed = RunOptions {
        rerun: benchflow_run::config::Rerun::parse("failed").unwrap(),
        ..RunOptions::default()
    };
    let (plan, _) = diff_plan(&units, &failed, &jobs);
    assert_eq!(by_agent(&plan, "oracle").action, Action::Reuse);
    assert_eq!(by_agent(&plan, "nop").action, Action::Rerun);

    // Each earlier trial is used once: two planned trials, one earlier.
    let twice = vec![(t.clone(), vec![Attempt::Oracle, Attempt::Oracle])];
    let (plan, _) = diff_plan(&twice, &options, &jobs);
    assert_eq!(plan.count(Action::Reuse), 1);
    assert!(plan.trials[1].reason.contains("already used"), "{plan:#?}");

    // A job without lock.json (BenchFlow 0.8) is skipped with the reason.
    let old = tmp.path().join("old-job");
    std::fs::create_dir_all(&old).unwrap();
    let (plan, _) = diff_plan(&units, &options, &[old]);
    assert_eq!(plan.skipped.len(), 1);
    assert!(plan.skipped[0].reason.contains("no lock.json"));
}

fn by_agent<'a>(
    plan: &'a benchflow_run::diff::DiffPlan,
    agent: &str,
) -> &'a benchflow_run::diff::TrialDiff {
    plan.trials.iter().find(|x| x.agent == agent).unwrap()
}

/// A backend that makes the sandbox at once but takes a while to hand it
/// over, as a remote backend does.
struct SlowStart {
    inner: FakeProvider,
    delay: Duration,
}

#[async_trait]
impl Provider for SlowStart {
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn capabilities(&self) -> Capabilities {
        self.inner.capabilities()
    }
    async fn status(&self) -> ProviderStatus {
        self.inner.status().await
    }
    async fn prepare(
        &self,
        spec: &ImageSpec,
        log: &(dyn for<'a> Fn(&'a str) + Send + Sync),
    ) -> Result<Image, SandboxError> {
        self.inner.prepare(spec, log).await
    }
    async fn create(&self, spec: &SandboxSpec) -> Result<Box<dyn Sandbox>, SandboxError> {
        let sandbox = self.inner.create(spec).await?;
        tokio::time::sleep(self.delay).await;
        Ok(sandbox)
    }
}

/// Drain mode (the first Ctrl-C): the trial running finishes and is
/// scored, and the queued one never starts.
#[tokio::test]
async fn a_drained_run_finishes_running_trials_and_starts_no_more() {
    let tmp = tempfile::tempdir().unwrap();
    let units: Vec<_> = ["one", "two"]
        .iter()
        .map(|n| {
            (
                Arc::new(task(&tmp.path().join(n), n)),
                vec![Attempt::Oracle],
            )
        })
        .collect();
    let provider = Arc::new(SlowStart {
        inner: FakeProvider {
            worlds: Mutex::new(Vec::new()),
            writable_path: None,
        },
        delay: Duration::from_millis(400),
    });
    let (_stop_tx, stop_rx) = stop_switch();
    let options = RunOptions::default();
    let drain = options.drain.clone();
    let rt = Arc::new(Runtime {
        provider: provider.clone(),
        images: Images::new(),
        options,
        events: Events::none(),
        stop: stop_rx,
    });
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(100)).await;
        drain.start();
    });
    let spec = JobSpec {
        carry: Vec::new(),
        units,
        trials: 1,
        jobs_dir: tmp.path().join("jobs"),
        job_name: None,
        fresh: true,
        concurrency: 1,
        source: serde_json::json!({}),
    };
    let out = job::run(spec, rt).await.unwrap();
    assert_eq!(out.records.len(), 1, "{:?}", out.records);
    assert_eq!(out.records[0].status, Status::Scored);
    assert_eq!(out.records[0].reward, Some(1.0));
    assert_eq!(out.not_started, 1);
    assert_eq!(provider.inner.worlds.lock().unwrap().len(), 1);
}

/// Delta review B1: a `bench test` drained after its first trial must not
/// call the task proved. The oracle ran and passed; the nop never started,
/// so the verdict fails with a "did not run" check for it.
#[tokio::test]
async fn a_drained_bench_test_does_not_prove_its_task() {
    let tmp = tempfile::tempdir().unwrap();
    let t = Arc::new(task(&tmp.path().join("probe-me"), "probe-me"));
    let planned = benchflow_run::testplan::attempts(&t, &benchflow_run::testplan::ProbeSet::None);
    assert_eq!(planned.len(), 2, "the oracle and the nop");
    let provider = Arc::new(SlowStart {
        inner: FakeProvider {
            worlds: Mutex::new(Vec::new()),
            writable_path: None,
        },
        delay: Duration::from_millis(400),
    });
    let (_stop_tx, stop_rx) = stop_switch();
    let options = RunOptions::default();
    let drain = options.drain.clone();
    let rt = Arc::new(Runtime {
        provider,
        images: Images::new(),
        options,
        events: Events::none(),
        stop: stop_rx,
    });
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(100)).await;
        drain.start();
    });
    let spec = JobSpec {
        carry: Vec::new(),
        units: vec![(t.clone(), planned.clone())],
        trials: 1,
        jobs_dir: tmp.path().join("jobs"),
        job_name: None,
        fresh: true,
        concurrency: 1,
        source: serde_json::json!({}),
    };
    let out = job::run(spec, rt).await.unwrap();
    assert_eq!((out.records.len(), out.not_started), (1, 1));
    let records: Vec<&benchflow_run::record::TrialRecord> = out.records.iter().collect();
    let v = benchflow_run::testplan::judge(&t, &records, &planned);
    assert!(!v.ok, "{v:?}");
    let missing: Vec<&str> = v
        .checks
        .iter()
        .filter(|c| c.says.starts_with("did not run"))
        .map(|c| c.attempt.as_str())
        .collect();
    assert_eq!(missing, ["nop"], "{v:?}");
}

/// Delta review B2 and B3: a regrade drained after its first trial reports
/// the one it never started, and leaves no folder for it (the agent files
/// are copied only when a trial starts).
#[tokio::test]
async fn a_drained_regrade_reports_what_it_never_regraded() {
    let tmp = tempfile::tempdir().unwrap();
    let tasks: Vec<Arc<TaskPackage>> = ["ra", "rb"]
        .iter()
        .map(|n| Arc::new(task(&tmp.path().join(n), n)))
        .collect();
    let spec = JobSpec {
        carry: Vec::new(),
        units: tasks
            .iter()
            .map(|t| (t.clone(), vec![Attempt::Oracle]))
            .collect(),
        trials: 1,
        jobs_dir: tmp.path().join("jobs"),
        job_name: Some("graded".into()),
        fresh: false,
        concurrency: 1,
        source: serde_json::json!({}),
    };
    let fake = || FakeProvider {
        worlds: Mutex::new(Vec::new()),
        writable_path: None,
    };
    let out = job::run(spec, runtime(Arc::new(fake()))).await.unwrap();
    for r in &out.records {
        std::fs::create_dir_all(out.dir.join(&r.rollout_name).join("outputs/app")).unwrap();
    }
    let by_name: BTreeMap<String, Arc<TaskPackage>> =
        tasks.iter().map(|t| (t.name.clone(), t.clone())).collect();
    let load = |p: &Path| {
        let name = p.file_name().unwrap().to_string_lossy().to_string();
        Ok(by_name[&name].clone())
    };
    let plan = benchflow_run::regrade::plan_with(&out.dir, &load).unwrap();
    assert_eq!(plan.regradable.len(), 2);
    let options = RunOptions::default();
    let drain = options.drain.clone();
    let (_tx, rx) = stop_switch();
    let rt = Arc::new(Runtime {
        provider: Arc::new(SlowStart {
            inner: fake(),
            delay: Duration::from_millis(400),
        }),
        images: Images::new(),
        options,
        events: Events::none(),
        stop: rx,
    });
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(100)).await;
        drain.start();
    });
    let re = benchflow_run::regrade::run(plan, rt, 1).await.unwrap();
    assert_eq!((re.records.len(), re.not_started), (1, 1));
    let folders: Vec<String> = std::fs::read_dir(&re.dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir() && !e.file_name().to_string_lossy().starts_with('.'))
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();
    assert_eq!(
        folders,
        [re.records[0].rollout_name.clone()],
        "no folder for the trial never started"
    );
}

/// A trial folder without result.json (a crash, or an unsaved record) is
/// moved aside before the trial runs again, so the rerun cannot read the
/// lost attempt's files (such as an old verifier reward).
#[tokio::test]
async fn resume_moves_a_folder_without_a_result_aside() {
    let tmp = tempfile::tempdir().unwrap();
    let t = Arc::new(task(&tmp.path().join("gamma"), "gamma"));
    let spec = JobSpec {
        carry: Vec::new(),
        units: vec![(t, vec![Attempt::Nop])],
        trials: 1,
        jobs_dir: tmp.path().join("jobs"),
        job_name: Some("leftover".into()),
        fresh: false,
        concurrency: 1,
        source: serde_json::json!({}),
    };
    let fake = || {
        runtime(Arc::new(FakeProvider {
            worlds: Mutex::new(Vec::new()),
            writable_path: None,
        }))
    };
    let out = job::run(spec.clone(), fake()).await.unwrap();
    let name = out.records[0].rollout_name.clone();
    let dir = out.dir.join(&name);
    // The record was lost; a stale reward file is left behind.
    std::fs::remove_file(dir.join("result.json")).unwrap();
    std::fs::write(dir.join("stale-marker"), "lost attempt").unwrap();
    let again = job::run(spec, fake()).await.unwrap();
    assert_eq!(again.records.len(), 1);
    assert!(dir.join("result.json").is_file());
    assert!(!dir.join("stale-marker").exists(), "the rerun starts clean");
    assert!(
        again
            .dir
            .join(".rerun")
            .join(format!("{name}-1/stale-marker"))
            .is_file()
    );
}

/// A trial whose unexplained sandbox loss came back in the attempt after
/// it is capped (`rerun_capped`): a resumed job keeps it as it is, so code
/// inside a sandbox cannot buy a rerun on every resume, and `--rerun
/// errored` runs it again (round-2 reviews of rust/e9-decisions).
#[tokio::test]
async fn resume_keeps_a_capped_trial_unless_errored_trials_are_asked_for() {
    let tmp = tempfile::tempdir().unwrap();
    let t = Arc::new(task(&tmp.path().join("delta"), "delta"));
    let spec = JobSpec {
        carry: Vec::new(),
        units: vec![(t, vec![Attempt::Nop])],
        trials: 1,
        jobs_dir: tmp.path().join("jobs"),
        job_name: Some("capped".into()),
        fresh: false,
        concurrency: 1,
        source: serde_json::json!({}),
    };
    let fake = |options: RunOptions| {
        runtime_with(
            Arc::new(FakeProvider {
                worlds: Mutex::new(Vec::new()),
                writable_path: None,
            }),
            options,
        )
    };
    let out = job::run(spec.clone(), fake(RunOptions::default()))
        .await
        .unwrap();
    let dir = out.dir.join(&out.records[0].rollout_name);
    // Make the record a capped loss, as a repeat writes it.
    let mut v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("result.json")).unwrap()).unwrap();
    v["status"] = serde_json::json!("errored");
    v["rewards"] = serde_json::Value::Null;
    v["passed"] = serde_json::Value::Null;
    v["problem"] = serde_json::json!({"code": "R103", "owner": "sandbox", "phase": "verify",
        "message": "capped-marker", "next": "x", "transient": false,
        "loss_confirmed": false, "rerun_capped": true});
    std::fs::write(dir.join("result.json"), v.to_string()).unwrap();
    // The job records how reward files were read (round-2 list, L6).
    let evaluation: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out.dir.join("evaluation.json")).unwrap())
            .unwrap();
    assert_eq!(
        evaluation["config"]["reward_lenient"],
        serde_json::json!(benchflow_task::reward::lenient_from_env()),
        "{evaluation}"
    );
    let kept = job::run(spec.clone(), fake(RunOptions::default()))
        .await
        .unwrap();
    assert_eq!(kept.records[0].status, Status::Errored);
    assert_eq!(
        kept.records[0].problem.as_ref().map(|p| p.message.as_str()),
        Some("capped-marker"),
        "a resume keeps the capped trial"
    );
    let mut options = RunOptions::default();
    options.rerun.errored = true;
    let rerun = job::run(spec, fake(options)).await.unwrap();
    assert_eq!(
        rerun.records[0].status,
        Status::Scored,
        "--rerun errored runs it"
    );
}

#[tokio::test]
async fn a_sandbox_being_made_when_the_run_stops_is_removed() {
    let tmp = tempfile::tempdir().unwrap();
    let t = Arc::new(task(&tmp.path().join("slow"), "slow"));
    let provider = Arc::new(SlowStart {
        inner: FakeProvider {
            worlds: Mutex::new(Vec::new()),
            writable_path: None,
        },
        delay: Duration::from_millis(400),
    });
    let (stop_tx, stop_rx) = stop_switch();
    let rt = Arc::new(Runtime {
        provider: provider.clone(),
        images: Images::new(),
        options: RunOptions::default(),
        events: Events::none(),
        stop: stop_rx,
    });
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(100)).await;
        let _ = stop_tx.send(true);
    });
    let spec = JobSpec {
        carry: Vec::new(),
        units: vec![(t, vec![Attempt::Nop])],
        trials: 1,
        jobs_dir: tmp.path().join("jobs"),
        job_name: None,
        fresh: true,
        concurrency: 1,
        source: serde_json::json!({}),
    };
    let out = job::run(spec, rt).await.unwrap();
    assert_eq!(out.records[0].problem.as_ref().unwrap().code, "R001");
    assert!(benchflow_run::trial::settle_stopped_starts(Duration::from_secs(5)).await);
    let worlds = provider.inner.worlds.lock().unwrap();
    assert_eq!(worlds.len(), 1);
    assert!(worlds[0].lock().unwrap().destroyed);
}

/// A backend that notes, as each sandbox is made, whether a live process
/// holds `job`'s run lock (what `bench view` checks to leave unfinished
/// trials out).
struct LockProbe {
    inner: FakeProvider,
    job: PathBuf,
    held: Mutex<Vec<bool>>,
}

#[async_trait]
impl Provider for LockProbe {
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn capabilities(&self) -> Capabilities {
        self.inner.capabilities()
    }
    async fn status(&self) -> ProviderStatus {
        self.inner.status().await
    }
    async fn prepare(
        &self,
        spec: &ImageSpec,
        log: &(dyn for<'a> Fn(&'a str) + Send + Sync),
    ) -> Result<Image, SandboxError> {
        self.inner.prepare(spec, log).await
    }
    async fn create(&self, spec: &SandboxSpec) -> Result<Box<dyn Sandbox>, SandboxError> {
        let held = std::fs::File::open(self.job.join(job::RUNNING_FILE))
            .map(|f| matches!(f.try_lock_shared(), Err(std::fs::TryLockError::WouldBlock)))
            .unwrap_or(false);
        self.held.lock().unwrap().push(held);
        self.inner.create(spec).await
    }
}

/// A live `bench regrade` holds its new job's run lock, as `bench run`
/// does, so the viewer does not count its unfinished trials as errored.
#[tokio::test]
async fn a_regrade_holds_its_job_while_it_runs() {
    let tmp = tempfile::tempdir().unwrap();
    let t = Arc::new(task(&tmp.path().join("alpha"), "alpha"));
    let spec = JobSpec {
        carry: Vec::new(),
        units: vec![(t.clone(), vec![Attempt::Oracle])],
        trials: 1,
        jobs_dir: tmp.path().join("jobs"),
        job_name: Some("graded".into()),
        fresh: false,
        concurrency: 1,
        source: serde_json::json!({"targets": ["alpha"]}),
    };
    let out = job::run(
        spec,
        runtime(Arc::new(FakeProvider {
            worlds: Mutex::new(Vec::new()),
            writable_path: None,
        })),
    )
    .await
    .unwrap();
    let rec = &out.records[0];
    std::fs::create_dir_all(out.dir.join(&rec.rollout_name).join("outputs/app")).unwrap();
    let load = |_: &Path| Ok(t.clone());
    let plan = benchflow_run::regrade::plan_with(&out.dir, &load).unwrap();
    let provider = Arc::new(LockProbe {
        inner: FakeProvider {
            worlds: Mutex::new(Vec::new()),
            writable_path: None,
        },
        job: tmp.path().join("jobs/graded-regrade-1"),
        held: Mutex::new(Vec::new()),
    });
    let (_tx, rx) = stop_switch();
    let rt = Arc::new(Runtime {
        provider: provider.clone(),
        images: Images::new(),
        options: RunOptions::default(),
        events: Events::none(),
        stop: rx,
    });
    let re = benchflow_run::regrade::run(plan, rt, 1).await.unwrap();
    assert_eq!(re.dir, provider.job);
    assert_eq!(*provider.held.lock().unwrap(), vec![true]);
}

#[tokio::test]
async fn a_second_run_of_a_running_job_is_refused_and_resumes_after() {
    let tmp = tempfile::tempdir().unwrap();
    let t = Arc::new(task(&tmp.path().join("delta"), "delta"));
    let spec = || JobSpec {
        carry: Vec::new(),
        units: vec![(t.clone(), vec![Attempt::Oracle])],
        trials: 1,
        jobs_dir: tmp.path().join("jobs"),
        job_name: Some("shared".into()),
        fresh: false,
        concurrency: 1,
        source: serde_json::json!({}),
    };
    let provider = || {
        Arc::new(FakeProvider {
            worlds: Mutex::new(Vec::new()),
            writable_path: None,
        })
    };
    // Another process holds the job (a run that is still going).
    let dir = tmp.path().join("jobs/shared");
    std::fs::create_dir_all(&dir).unwrap();
    let other = job::hold(&dir).unwrap();
    let refused = provider();
    let Err(e) = job::run(spec(), runtime(refused.clone())).await else {
        panic!("a second run of a running job must be refused");
    };
    assert_eq!(e.kind(), std::io::ErrorKind::WouldBlock);
    assert!(e.to_string().contains("--job"), "{e}");
    assert!(
        refused.worlds.lock().unwrap().is_empty(),
        "no sandbox started"
    );
    // It ends (or crashes: the lock goes with the process); the job runs.
    drop(other);
    let out = job::run(spec(), runtime(provider())).await.unwrap();
    assert_eq!(out.records[0].status, Status::Scored);
    // And a later run resumes it without running the trial again.
    let p = provider();
    let again = job::run(spec(), runtime(p.clone())).await.unwrap();
    assert_eq!(again.records[0].reward, Some(1.0));
    assert!(p.worlds.lock().unwrap().is_empty(), "the trial was reused");
}

/// A judge that says `answered` holds when the request shows the solved
/// answer and nothing else holds, answering like the Vercel AI Gateway.
async fn fake_judge() -> (Arc<Mutex<Vec<serde_json::Value>>>, BTreeMap<String, String>) {
    use axum::routing::{get, post};
    let seen: Arc<Mutex<Vec<serde_json::Value>>> = Arc::default();
    let log = seen.clone();
    let app = axum::Router::new()
        .route(
            "/v1/chat/completions",
            post(move |axum::Json(body): axum::Json<serde_json::Value>| {
                let log = log.clone();
                async move {
                    let seen = body.to_string();
                    // Only the answer criterion's statement holds of a solved run.
                    let solved = seen.contains("SOLVED") && seen.contains("CRITERION `answered`");
                    log.lock().unwrap().push(body);
                    let text = serde_json::json!({"holds": solved, "reason": if solved { "The answer is there." } else { "No answer file." }, "quotes": []});
                    axum::Json(serde_json::json!({"choices": [{"finish_reason": "stop",
                        "message": {"role": "assistant", "content": text.to_string()}}],
                        "usage": {"prompt_tokens": 500, "completion_tokens": 20, "cost": 0.0002}}))
                }
            }),
        )
        .route(
            "/v1/models/{*path}",
            get(|| async { axum::Json(serde_json::json!({"data": {"endpoints": [{"provider_name": "novita"}]}})) }),
        );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let vars = BTreeMap::from([
        (
            "AI_GATEWAY_API_KEY".to_string(),
            "vck_fake_judge_key_0123".to_string(),
        ),
        (
            "BENCHFLOW_NATIVE_BASE_URL".to_string(),
            format!("http://{addr}/v1"),
        ),
    ]);
    (seen, vars)
}

#[tokio::test]
async fn a_model_judged_rubric_scores_oracle_and_nop_on_the_host() {
    let tmp = tempfile::tempdir().unwrap();
    let mut t = task(&tmp.path().join("judged"), "judged");
    t.verifier.rubric = Some(
        benchflow_task::rubric::Rubric::from_value(&serde_json::json!({
            "$schema": "https://task.md/schema/rubric-1.json",
            "x-judge": {"model": "vercel/judge/model"},
            "criteria": [
                {"id": "answered", "text": "The answer file holds the answer.", "judge": "llm",
                 "points": 2, "evidence": ["/app/answer.txt"]},
                {"id": "no-junk", "text": "The agent left junk files.", "judge": "llm", "outcome": "bad",
                 "points": 1, "evidence": ["/app/answer.txt"]}
            ]
        }))
        .unwrap(),
    );
    let provider = Arc::new(FakeProvider {
        worlds: Mutex::new(Vec::new()),
        writable_path: None,
    });
    let (seen, vars) = fake_judge().await;
    let options = RunOptions {
        judge_env: benchflow_run::config::JudgeEnv(vars),
        ..RunOptions::default()
    };
    let spec = JobSpec {
        carry: Vec::new(),
        units: vec![(Arc::new(t), vec![Attempt::Oracle, Attempt::Nop])],
        trials: 1,
        jobs_dir: jobs_dir(tmp.path()),
        job_name: Some("judged".into()),
        fresh: true,
        concurrency: 2,
        source: serde_json::json!({"path": tmp.path()}),
    };
    let out = job::run(spec, runtime_with(provider, options))
        .await
        .unwrap();
    assert_eq!(out.records.len(), 2);
    for r in &out.records {
        assert_eq!(r.status, Status::Scored, "{:?}", r.problem);
        let oracle = r.agent == "oracle";
        // Oracle: 2 + 1 of 3; nop: 0 + 1 (no junk) of 3.
        let want = if oracle { 1.0 } else { 1.0 / 3.0 };
        assert!(
            (r.reward.unwrap() - want).abs() < 1e-9,
            "{} {:?}",
            r.agent,
            r.reward
        );
        let dir = out.dir.join(&r.rollout_name);
        let lines = std::fs::read_to_string(dir.join("verifier/judgments.jsonl")).unwrap();
        assert_eq!(lines.lines().count(), 2);
        assert!(!lines.contains("vck_fake_judge_key"));
        let review: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join("verifier/review.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(
            review["verdicts"][0]["judge"]["model"],
            "vercel/judge/model"
        );
        assert_eq!(
            review["verdicts"][0]["verdict"],
            if oracle { "pass" } else { "fail" }
        );
        assert!(review["verdicts"][0]["rationale"].as_str().unwrap().len() > 3);
        assert_eq!(review["x-benchflow"]["judge"]["calls"], 2);
        let result = std::fs::read_to_string(dir.join("result.json")).unwrap();
        assert!(!result.contains("vck_fake_judge_key"));
        let config = std::fs::read_to_string(dir.join("config.json")).unwrap();
        assert!(!config.contains("vck_fake_judge_key"));
    }
    assert_eq!(seen.lock().unwrap().len(), 4);
}

/// A task of two graded stages, each solved by the oracle.
fn staged(root: &Path, name: &str) -> TaskPackage {
    use benchflow_task::extras::{Combine, StepSpec, Steps, Unreached};
    let mut t = task(root, name);
    let stage = |n: &str| StepSpec {
        name: n.into(),
        prompt: format!("Do stage {n}."),
        graded: true,
        ..StepSpec::default()
    };
    t.extras.steps = Some(Steps {
        stages: vec![stage("first"), stage("second")],
        combine: Combine::Mean,
        unreached: Unreached::default(),
    });
    t
}

fn read_json(p: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

/// A fork after a graded stage: each child runs the stages after it,
/// carrying the parent's rows and grades; a continuing parent is graded
/// too, and the records say it was kept. Then a discarded parent at start
/// with nop: its children score 0, and it is not graded.
#[tokio::test]
async fn stage_forks_and_a_continuing_parent_run_through_the_engine() {
    use benchflow_run::trial::branch::{self, ForkPlan, ForkPoint};
    let tmp = tempfile::tempdir().unwrap();
    let t = Arc::new(staged(&tmp.path().join("two"), "two"));
    let provider = Arc::new(FakeProvider {
        worlds: Mutex::new(Vec::new()),
        writable_path: None,
    });
    let plan = ForkPlan {
        at: ForkPoint::Stage("first".into()),
        children: ForkPlan::same_prompt(2),
        retain: false,
        resume: false,
        concurrency: 2,
        continue_parent: true,
    };
    let out = branch::run_job(
        vec![t.clone()],
        Attempt::Oracle,
        plan,
        jobs_dir(tmp.path()),
        Some("stage-fork".into()),
        runtime(provider.clone()),
    )
    .await
    .unwrap();
    let f = &out.forks[0];
    assert_eq!(f.status, "completed");
    assert_eq!(f.value, Some(1.0));
    assert!(f.parent_continued);
    // The parent went on to its second stage and was graded.
    assert_eq!(f.parent.status, Status::Scored, "{:?}", f.parent.problem);
    assert_eq!(f.parent.reward, Some(1.0));
    let trial = out.dir.join(&f.parent.rollout_name);
    let parent_steps = read_json(&trial.join("result.json"))["step_results"].clone();
    assert_eq!(parent_steps.as_array().unwrap().len(), 2);
    assert!(trial.join("steps/second").is_dir());
    // Each child ran only the second stage; the first is the parent's.
    for c in &f.children {
        let r = read_json(&c.dir.join("result.json"));
        let rows = r["step_results"].as_array().unwrap();
        assert_eq!(rows.len(), 2, "{r}");
        assert_eq!(rows[0]["step_name"], "first");
        assert_eq!(rows[0]["from_parent"], true);
        assert_eq!(rows[1]["step_name"], "second");
        assert!(rows[1].get("from_parent").is_none());
        assert!(!c.dir.join("steps/first").exists());
        assert!(c.dir.join("steps/second").is_dir());
        assert_eq!(r["rewards"]["reward"], 1.0, "{r}");
    }
    let tree = read_json(&trial.join("tree.json"));
    let fork = &tree["forks"][0];
    assert_eq!(fork["fork_point"], "stage:first");
    // After prompts the fork hangs off a node that points at the parent's
    // last event before it (0.8's exporter cuts the shared prefix there),
    // and the children hang off it too.
    assert_eq!(fork["parent_node"], "p1", "{tree}");
    let nodes = tree["nodes"].as_array().unwrap();
    let p1 = nodes
        .iter()
        .find(|n| n["id"] == "p1")
        .expect("the fork node");
    assert_eq!(p1["parent"], "root");
    assert_eq!(p1["step_id"], "step-0-oracle");
    assert!(
        nodes
            .iter()
            .filter(|n| n["id"].as_str().is_some_and(|i| i.starts_with('n')))
            .all(|n| n["parent"] == "p1"),
        "{tree}"
    );
    let parent_events = std::fs::read_to_string(trial.join("trajectory/acp_trajectory.jsonl"))
        .unwrap()
        .lines()
        .count();
    assert!(
        parent_events >= 1,
        "the parent's trajectory holds the prefix"
    );
    let obs = read_json(&f.children[0].dir.join("observation.json"));
    assert_eq!(obs["lineage"]["parent_node"], "p1");
    assert_eq!(fork["parent_restore"], "not_needed");
    assert_eq!(fork["parent_reward"], 1.0);
    let cps = read_json(&trial.join("checkpoints.json"));
    assert_eq!(cps["checkpoints"][0]["after_prompt"], 1);
    assert_eq!(cps["checkpoints"][0]["status"], "deleted");
    let result = read_json(&trial.join("result.json"));
    assert_eq!(result["branches"]["parent"], "kept");
    let summary = read_json(&out.dir.join("summary.json"));
    assert_eq!(summary["parent"], "continue");
    assert_eq!(summary["trials"][0]["parent_reward"], 1.0);
    assert!(CHECKPOINTS.lock().unwrap().is_empty());
    // results.jsonl: the continuing parent, then each child.
    let rows: Vec<serde_json::Value> = std::fs::read_to_string(out.dir.join("results.jsonl"))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0]["info"]["branch"]["role"], "parent");
    assert_eq!(rows[1]["info"]["branch"]["role"], "child");
    assert_eq!(rows[1]["info"]["branch"]["advantage"], 0.0);
    assert_eq!(rows[2]["reward"], 1.0);
    // 0.8's child archive carries the child's trajectory for the viewer.
    let obs = read_json(&f.children[0].dir.join("observation.json"));
    assert!(!obs["trajectory"].as_array().unwrap().is_empty(), "{obs}");
    // Both children started from the parent's checkpoint.
    assert_eq!(FROM_CHECKPOINT.load(std::sync::atomic::Ordering::SeqCst), 2);

    // nop at start, parent discarded: children 0, parent not graded.
    let plan = ForkPlan {
        at: ForkPoint::Start,
        children: ForkPlan::same_prompt(2),
        retain: false,
        resume: false,
        concurrency: 1,
        continue_parent: false,
    };
    let out = branch::run_job(
        vec![t],
        Attempt::Nop,
        plan,
        jobs_dir(tmp.path()),
        Some("start-nop".into()),
        runtime(provider),
    )
    .await
    .unwrap();
    let f = &out.forks[0];
    assert_eq!(f.status, "completed");
    assert_eq!(f.value, Some(0.0));
    assert!(!f.parent_continued);
    assert_eq!(f.parent.problem.as_ref().unwrap().code, "R000");
    // A fork at the start stops the parent before its agent runs.
    assert_eq!(
        f.parent.problem.as_ref().unwrap().phase,
        benchflow_run::outcome::Phase::Setup
    );
    let trial = out.dir.join(&f.parent.rollout_name);
    let tree = read_json(&trial.join("tree.json"));
    assert_eq!(tree["forks"][0]["parent_restore"], "skipped");
    // A fork at start hangs off the root, which has no step.
    assert_eq!(tree["forks"][0]["parent_node"], "root");
    assert!(tree["forks"][0]["parent_reward"].is_null());
    // No row for the discarded parent.
    let rows = std::fs::read_to_string(out.dir.join("results.jsonl")).unwrap();
    assert_eq!(rows.lines().count(), 2, "{rows}");
    // Each child ran both stages.
    for c in &f.children {
        let r = read_json(&c.dir.join("result.json"));
        assert_eq!(r["step_results"].as_array().unwrap().len(), 2, "{r}");
        assert!(c.dir.join("steps/first").is_dir());
    }
}

/// A refusal at the fork point (here the secret scan finds a value): a
/// discarded parent ends there, unscored, with no child; a continuing
/// parent goes on and is graded as a plain trial, and the fork records the
/// refusal (0.8's spelling fills in --parent continue).
#[tokio::test]
async fn a_refused_checkpoint_still_grades_a_continuing_parent() {
    use benchflow_run::trial::branch::{self, ForkPlan, ForkPoint};
    let tmp = tempfile::tempdir().unwrap();
    let mut t = task(&tmp.path().join("leaky"), "leaky");
    t.sandbox.env.insert(
        "FAKE_LEAK_TOKEN".into(),
        "leak-0123456789abcdef-0123456789".into(),
    );
    let t = Arc::new(t);
    let provider = Arc::new(FakeProvider {
        worlds: Mutex::new(Vec::new()),
        writable_path: None,
    });
    for continue_parent in [true, false] {
        let plan = ForkPlan {
            at: ForkPoint::Start,
            children: ForkPlan::same_prompt(2),
            retain: false,
            resume: false,
            concurrency: 2,
            continue_parent,
        };
        let out = branch::run_job(
            vec![t.clone()],
            Attempt::Oracle,
            plan,
            jobs_dir(tmp.path()),
            Some(format!("leak-{continue_parent}")),
            runtime(provider.clone()),
        )
        .await
        .unwrap();
        let f = &out.forks[0];
        assert_eq!(f.status, "failed");
        assert!(f.checkpoint.is_none());
        assert!(f.children.iter().all(|c| c.record.is_none()));
        let tree = read_json(&out.dir.join(&f.parent.rollout_name).join("tree.json"));
        let fork = &tree["forks"][0];
        assert_eq!(fork["error"]["code"], "R401", "{fork}");
        // tree.json keeps no message (0.8's display record); a continuing
        // parent's refusal is written beside it.
        assert!(fork["error"].get("message").is_none(), "{fork}");
        if continue_parent {
            let refusal = read_json(
                &out.dir
                    .join(&f.parent.rollout_name)
                    .join("branch-refusal.json"),
            );
            assert!(
                refusal["message"].as_str().unwrap().contains("/etc/leaked"),
                "{refusal}"
            );
            assert_eq!(f.parent.status, Status::Scored, "{:?}", f.parent.problem);
            assert_eq!(f.parent.reward, Some(1.0));
            assert_eq!(fork["parent_restore"], "not_attempted");
            assert_eq!(fork["parent_reward"], 1.0);
        } else {
            assert_eq!(f.parent.problem.as_ref().unwrap().code, "R401");
            assert!(fork["parent_reward"].is_null());
        }
    }
}
