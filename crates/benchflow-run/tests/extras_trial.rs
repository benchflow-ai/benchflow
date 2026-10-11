//! Stages, services, hooks, and a verifier image through the real engine,
//! on a scripted in-memory sandbox, with the fixtures `bench test` runs live
//! (`tests/fixtures/tasks/{two-stage,sidecar,separate-verifier,snapshot-hook}`).

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use benchflow_run::agent::Attempt;
use benchflow_run::config::RunOptions;
use benchflow_run::events::Events;
use benchflow_run::images::Images;
use benchflow_run::job::{self, JobSpec};
use benchflow_run::outcome::Status;
use benchflow_run::record::TrialRecord;
use benchflow_run::trial::{Runtime, stop_switch};
use benchflow_sandbox::services::{ServiceHost, ServiceSpec};
use benchflow_sandbox::{
    Capabilities, Exec, ExecOutput, Image, ImageSpec, Process, Provider, ProviderStatus, Sandbox,
    SandboxError, SandboxSpec,
};
use benchflow_task::TaskPackage;
use benchflow_task::extras::{Combine, Gate, Unreached};
use benchflow_task::package::{ImageRef, Isolation};
use serde_json::Value;

fn fixture(name: &str) -> Arc<TaskPackage> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/tasks")
        .join(name);
    Arc::new(benchflow_task::load(&dir).unwrap_or_else(|d| panic!("{name}: {d:#?}")))
}

#[test]
fn the_fixtures_read_as_intended() {
    let t = fixture("two-stage");
    assert!(t.unsupported.is_empty(), "{:#?}", t.unsupported);
    let steps = t.extras.steps.as_ref().unwrap();
    assert_eq!(
        (steps.combine, steps.unreached),
        (Combine::Mean, Unreached::Zero)
    );
    let first = &steps.stages[0];
    assert_eq!(first.name, "first");
    assert_eq!(first.gate, Some(Gate::Reward(1.0)));
    assert_eq!(
        first.verifier_dirs,
        [PathBuf::from("stages/first/verifier")]
    );
    assert_eq!(first.outputs.len(), 1);
    let second = &steps.stages[1];
    assert!(second.prompt.starts_with("Now write the word `beta`"));
    assert!(second.verifier_dirs.is_empty(), "the task's verifier");
    assert!(second.oracle_dir.is_some());

    let t = fixture("sidecar");
    assert!(t.unsupported.is_empty(), "{:#?}", t.unsupported);
    let api = &t.extras.services[0];
    assert_eq!(api.name, "api");
    assert!(
        matches!(&api.image, ImageRef::Build { context, .. } if context == Path::new("environment/api"))
    );
    assert_eq!(
        api.ready.as_ref().unwrap().run,
        "wget -qO- http://127.0.0.1:8000/token"
    );

    let t = fixture("separate-verifier");
    assert!(t.unsupported.is_empty(), "{:#?}", t.unsupported);
    assert_eq!(t.verifier.isolation, Isolation::Separate);
    assert_eq!(
        t.extras.verifier_image.as_ref().unwrap().source,
        "verifier/Dockerfile"
    );

    let t = fixture("snapshot-hook");
    assert!(t.unsupported.is_empty(), "{:#?}", t.unsupported);
    let hook = &t.extras.hooks[0];
    assert_eq!(
        (hook.service.as_deref(), hook.timeout),
        (None, Duration::from_secs(30))
    );
}

/// What the scripted sandboxes saw.
#[derive(Default)]
struct World {
    /// Stages whose reference solution ran.
    solved: BTreeSet<String>,
    /// The verifier folder uploaded last.
    verifier: Option<PathBuf>,
    commands: Vec<String>,
    /// (user, command) of every command.
    users: Vec<(String, String)>,
    /// The variables each command was given.
    envs: Vec<BTreeMap<String, String>>,
    networks: Vec<String>,
    services: Vec<String>,
    attached: Vec<String>,
    removed: Vec<String>,
    /// (local, remote) of every upload.
    uploads: Vec<(PathBuf, String)>,
    /// (remote, contents) of every file written.
    writes: Vec<(String, Vec<u8>)>,
}

struct Fake {
    id: String,
    world: Arc<Mutex<World>>,
}

#[async_trait]
impl Sandbox for Fake {
    fn id(&self) -> &str {
        &self.id
    }
    fn backend(&self) -> &str {
        "fake"
    }
    async fn exec(&self, exec: Exec) -> Result<ExecOutput, SandboxError> {
        // Root's plumbing commands start with the plumbing umask; the fake
        // answers and records them by what comes after it.
        let prefix = format!("{}\n", benchflow_run::hardening::PLUMBING_UMASK);
        let c = exec
            .command
            .strip_prefix(prefix.as_str())
            .unwrap_or(&exec.command)
            .to_string();
        let mut w = self.world.lock().unwrap();
        w.commands.push(c.clone());
        w.users
            .push((exec.user.clone().unwrap_or_default(), c.clone()));
        w.envs.push(exec.env.clone());
        let stdout = if c.contains("useradd") {
            "BF user agent 1000 /home/agent\n".to_string()
        } else if c.contains("baseline") && c.contains("files.tar") && !c.contains("after") {
            "BF baseline 2\n".to_string()
        } else if c.contains("kill -9 -1") {
            "BF killed 1\n".to_string()
        } else if c.contains("ld.so.preload") {
            "BF preload absent\n".to_string()
        } else if c.contains("BF venv-done") {
            "BF venv-safepath 1\nBF venv-path /usr/bin:/bin\nBF venv-done\n".to_string()
        } else if c.contains("BF path") {
            "BF path /usr/bin:/bin\n".to_string()
        } else if c.contains("BF handoff done") {
            "BF handoff done\n".to_string()
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
        } else if c.contains("BF open") && c.starts_with("d=/app/locked;") {
            // A target an earlier mount placed root's and read-only.
            "BF open /app\nBF target-closed\n".to_string()
        } else if c.contains("BF open")
            && (c.starts_with("d=/app/data;") || c.starts_with("d='/app/data';"))
        {
            // Before placing a mount, while an agent may still run: the
            // agent user can write /app.
            "BF open /app\n".to_string()
        } else if c.contains("BF baseline-added") {
            "BF baseline-added 0\n".to_string()
        } else if c.contains("BF staged") {
            "BF staged\n".to_string()
        } else if c.contains("BF placement done") {
            "BF placement done\n".to_string()
        } else {
            if c.contains("/solve.sh")
                && let Some(stage) = exec.env.get("BENCHFLOW_STAGE")
            {
                w.solved.insert(stage.clone());
            }
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
    async fn upload(&self, local: &Path, remote: &str) -> Result<(), SandboxError> {
        let mut w = self.world.lock().unwrap();
        if remote == "/verifier" || remote == "/tests" {
            w.verifier = Some(local.to_path_buf());
        }
        w.uploads.push((local.to_path_buf(), remote.to_string()));
        Ok(())
    }
    async fn write_file(&self, r: &str, c: &[u8], _m: u32) -> Result<(), SandboxError> {
        self.world
            .lock()
            .unwrap()
            .writes
            .push((r.to_string(), c.to_vec()));
        Ok(())
    }
    async fn download(&self, remote: &str, local: &Path) -> Result<(), SandboxError> {
        let w = self.world.lock().unwrap();
        if remote == "/logs/verifier" {
            std::fs::create_dir_all(local).unwrap();
            let first = w.solved.contains("first");
            let reward = match &w.verifier {
                Some(v) if v.ends_with("stages/first/verifier") => first,
                _ => first && w.solved.contains("second"),
            };
            std::fs::write(local.join("reward.txt"), if reward { "1\n" } else { "0\n" }).unwrap();
        } else if remote.ends_with("outputs.tar") {
            let mut b = tar::Builder::new(std::fs::File::create(local).unwrap());
            if w.solved.contains("first") {
                let mut h = tar::Header::new_gnu();
                h.set_size(5);
                h.set_mode(0o644);
                h.set_cksum();
                b.append_data(&mut h, "app/one.txt", &b"alpha"[..]).unwrap();
            }
            b.finish().unwrap();
        }
        Ok(())
    }
    async fn destroy(&self) -> Result<(), SandboxError> {
        let mut w = self.world.lock().unwrap();
        let id = self.id.clone();
        w.removed.push(id);
        Ok(())
    }
}

struct FakeProvider {
    world: Arc<Mutex<World>>,
    with_services: bool,
    /// Sandboxes on an image whose name ends with this fail to start.
    failing_image: Option<&'static str>,
}

#[async_trait]
impl Provider for FakeProvider {
    fn name(&self) -> &str {
        if self.with_services {
            "fake"
        } else {
            "fakecloud"
        }
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
    async fn create(&self, spec: &SandboxSpec) -> Result<Box<dyn Sandbox>, SandboxError> {
        if self
            .failing_image
            .is_some_and(|end| spec.image.reference.ends_with(end))
        {
            return Err(SandboxError::Unsupported(
                "no sandbox for this image".into(),
            ));
        }
        Ok(Box::new(Fake {
            id: format!("sb-{}", spec.name),
            world: self.world.clone(),
        }))
    }
    fn services(&self) -> Option<&dyn ServiceHost> {
        self.with_services.then_some(self as &dyn ServiceHost)
    }
}

#[async_trait]
impl ServiceHost for FakeProvider {
    async fn create_network(
        &self,
        name: &str,
        _labels: &BTreeMap<String, String>,
    ) -> Result<String, SandboxError> {
        let net = format!("net-{name}");
        self.world.lock().unwrap().networks.push(net.clone());
        Ok(net)
    }
    async fn start_service(
        &self,
        _network: &str,
        spec: &ServiceSpec,
    ) -> Result<Box<dyn Sandbox>, SandboxError> {
        self.world.lock().unwrap().services.push(spec.name.clone());
        Ok(Box::new(Fake {
            id: format!("svc-{}", spec.name),
            world: self.world.clone(),
        }))
    }
    async fn create_on_network(
        &self,
        spec: &SandboxSpec,
        network: &str,
    ) -> Result<Box<dyn Sandbox>, SandboxError> {
        self.world
            .lock()
            .unwrap()
            .attached
            .push(network.to_string());
        self.create(spec).await
    }
    async fn logs(&self, _id: &str) -> String {
        "serving".into()
    }
    async fn remove_network(&self, network: &str) -> Result<(), SandboxError> {
        self.world.lock().unwrap().removed.push(network.to_string());
        Ok(())
    }
}

type Ran = (
    Vec<TrialRecord>,
    PathBuf,
    Arc<Mutex<World>>,
    tempfile::TempDir,
);

async fn run(task: Arc<TaskPackage>, attempts: Vec<Attempt>, with_services: bool) -> Ran {
    run_failing(task, attempts, with_services, None).await
}

/// [`run`], with sandboxes on an image whose name ends with
/// `failing_image` failing to start.
async fn run_failing(
    task: Arc<TaskPackage>,
    attempts: Vec<Attempt>,
    with_services: bool,
    failing_image: Option<&'static str>,
) -> Ran {
    let tmp = tempfile::tempdir().unwrap();
    let world = Arc::new(Mutex::new(World::default()));
    let (_tx, rx) = stop_switch();
    std::mem::forget(_tx);
    let rt = Arc::new(Runtime {
        provider: Arc::new(FakeProvider {
            world: world.clone(),
            with_services,
            failing_image,
        }),
        images: Images::new(),
        options: RunOptions::default(),
        events: Events::none(),
        stop: rx,
    });
    let spec = JobSpec {
        carry: Vec::new(),
        units: vec![(task, attempts)],
        trials: 1,
        jobs_dir: tmp.path().join("jobs"),
        job_name: Some("extras".into()),
        fresh: true,
        concurrency: 1,
        source: serde_json::json!({}),
    };
    let out = job::run(spec, rt).await.unwrap();
    (out.records, out.dir, world, tmp)
}

fn result_json(dir: &Path, r: &TrialRecord) -> Value {
    serde_json::from_str(
        &std::fs::read_to_string(dir.join(&r.rollout_name).join("result.json")).unwrap(),
    )
    .unwrap()
}

/// The two-stage fixture, copied into `dir`, whose second stage ships a
/// workdir/ folder with a `conftest.py` and a mount with another, and whose
/// opening stage is graded or not.
fn shipping_stages(dir: &Path, opening_graded: bool) -> Arc<TaskPackage> {
    shipping_stages_at(dir, opening_graded, "/app/data")
}

/// [`shipping_stages`], with the second stage's mount at `target`.
fn shipping_stages_at(dir: &Path, opening_graded: bool, target: &str) -> Arc<TaskPackage> {
    fn copy(from: &Path, to: &Path) {
        std::fs::create_dir_all(to).unwrap();
        for e in std::fs::read_dir(from).unwrap().flatten() {
            let t = to.join(e.file_name());
            if e.file_type().unwrap().is_dir() {
                copy(&e.path(), &t);
            } else {
                std::fs::copy(e.path(), &t).unwrap();
            }
        }
    }
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tasks/two-stage");
    copy(&src, dir);
    for (f, body) in [
        ("stages/second/workdir/conftest.py", "SHIPPED = 2\n"),
        ("stages/second/workdir/notes.txt", "notes\n"),
        ("stages/second/data/conftest.py", "MOUNTED = 2\n"),
    ] {
        std::fs::create_dir_all(dir.join(f).parent().unwrap()).unwrap();
        std::fs::write(dir.join(f), body).unwrap();
    }
    let mut t = benchflow_task::load(dir).unwrap_or_else(|e| panic!("{e:#?}"));
    let steps = t.extras.steps.as_mut().unwrap();
    steps.stages[0].graded = opening_graded;
    steps.stages[0].gate = None;
    steps.stages[1].workdir = Some("stages/second/workdir".into());
    steps.stages[1].mounts = vec![benchflow_task::extras::Mount {
        source: "stages/second/data".into(),
        target: target.into(),
    }];
    Arc::new(t)
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::Digest;
    hex::encode(sha2::Sha256::digest(bytes))
}

/// After an opening stage that is not graded, its agent's processes still
/// run while BenchFlow prepares the next stage, so root writes nothing into
/// the workdir there and changes no owner: the stage's files and a mount
/// under a folder the agent can write are staged by root and copied in by
/// the agent user, the workdir is not handed back (it never left), and the
/// test-runner baseline is not taken again (a conftest.py the opening
/// stage planted would join it) but extended with the stage's own files,
/// read on this machine. Guards the D7b fixes of the stage boundary
/// (FIX-REVIEW-B 1 and the D7b brief's part 2).
#[tokio::test]
async fn after_an_ungraded_stage_root_writes_nothing_where_the_agent_runs() {
    let pkg = tempfile::tempdir().unwrap();
    let task = shipping_stages(pkg.path(), false);
    let (records, _dir, world, _tmp) = run(task, vec![Attempt::Oracle], true).await;
    assert!(records[0].problem.is_none(), "{:?}", records[0].problem);
    let w = world.lock().unwrap();
    let place = "/var/lib/benchflow-place/";
    // The stage's files and the mount went to root's own stage, never
    // straight to /app.
    let staged: Vec<&(PathBuf, String)> = w
        .uploads
        .iter()
        .filter(|(l, _)| l.ends_with("stages/second/workdir") || l.ends_with("stages/second/data"))
        .collect();
    assert_eq!(staged.len(), 2, "{:?}", w.uploads);
    assert!(
        staged.iter().all(|(_, r)| r.starts_with(place)),
        "{staged:?}"
    );
    // The agent user copied both in.
    let copies: Vec<&(String, String)> = w
        .users
        .iter()
        .filter(|(_, c)| c.contains("BF tree placed"))
        .collect();
    assert_eq!(copies.len(), 2, "{copies:?}");
    assert!(copies.iter().all(|(u, _)| u == "agent"), "{copies:?}");
    // The agent user removes the trees handed to it; root removes only
    // empty folders (REVIEW-CORRECT M4).
    let removed = w
        .users
        .iter()
        .filter(|(u, c)| u == "agent" && c.starts_with("rm -rf /var/lib/benchflow-place/"))
        .count();
    assert_eq!(removed, 2, "{:?}", w.users);
    assert!(
        !w.commands
            .iter()
            .any(|c| c.starts_with("rm -rf /var/lib/benchflow-place/")
                && w.users.iter().any(|(u, uc)| uc == c && u != "agent"))
    );
    assert!(copies.iter().any(|(_, c)| c.contains("d=/app/data\n")));
    // Root changed no owner in the workdir and claimed no mount there.
    for c in &w.commands {
        assert!(!c.contains("chown -R agent"), "{c}");
        assert!(!c.contains("chown -R 0:0 /app/data"), "{c}");
        assert!(!(c.contains("BF owned") && c.contains("o=agent;")), "{c}");
    }
    // One full baseline in setup, one before the first turn; none later.
    let full = w
        .commands
        .iter()
        .filter(|c| c.contains("BF baseline $("))
        .count();
    assert_eq!(full, 2);
    let added = w
        .commands
        .iter()
        .find(|c| c.contains("BF baseline-added"))
        .expect("the stage's files were added to the baseline");
    // The workdir's (the fake image's is /root) and the mount's.
    for (at, body) in [
        ("/conftest.py", "SHIPPED = 2\n"),
        ("/app/data/conftest.py", "MOUNTED = 2\n"),
    ] {
        let sum = sha256_hex(body.as_bytes());
        let path = added
            .lines()
            .find_map(|l| l.strip_prefix("add ")?.strip_suffix(&format!(" {sum}")))
            .unwrap_or_else(|| panic!("{body:?} not added: {added}"));
        assert!(path.ends_with(at), "{path}");
        let copy = format!("/var/lib/benchflow/baseline/added{path}");
        assert!(
            w.writes
                .iter()
                .any(|(r, c)| *r == copy && c == body.as_bytes()),
            "{copy}"
        );
    }
    assert!(!added.contains("notes.txt"), "{added}");
}

/// A later stage that mounts where a folder the agent cannot write already
/// is (an earlier mount placed it root's and read-only), while an earlier
/// ungraded stage's agent runs: neither root nor the agent user can place
/// it safely, and the agent's copy would fail every time. The trial is
/// refused as the task's (R104), before any copy, not blamed on the agent.
/// Guards the D7c fix of REVIEW-OUTCOMES M2.
#[tokio::test]
async fn a_remount_over_a_folder_the_agent_cannot_write_is_the_tasks() {
    let pkg = tempfile::tempdir().unwrap();
    let task = shipping_stages_at(pkg.path(), false, "/app/locked");
    let (records, _dir, world, _tmp) = run(task, vec![Attempt::Oracle], true).await;
    let p = records[0].problem.as_ref().expect("refused");
    assert_eq!(
        (p.code.as_str(), p.owner),
        ("R104", benchflow_run::outcome::Owner::Task),
        "{}",
        p.message
    );
    assert!(p.message.contains("/app/locked"), "{}", p.message);
    // After the opening stage's agent ran to the end: the task's fault is
    // not run again on resume (D7d brief, REVIEW-OUTCOMES (d7c) L5).
    assert_eq!(records[0].status, Status::Unscored);
    let w = world.lock().unwrap();
    assert!(
        !w.commands
            .iter()
            .any(|c| c.contains("BF tree placed") && c.contains("/app/locked")),
        "no copy was tried"
    );
}

/// After a graded opening stage, whose handoff stopped every agent process,
/// the workdir goes back to the agent user and the baseline is taken again;
/// the stage's files still come in through the agent user's copy, and the
/// mount the root way (no agent process can swap its path).
#[tokio::test]
async fn after_a_graded_stage_the_workdir_returns_and_the_baseline_is_retaken() {
    let pkg = tempfile::tempdir().unwrap();
    let task = shipping_stages(pkg.path(), true);
    let (records, _dir, world, _tmp) = run(task, vec![Attempt::Oracle], true).await;
    assert!(records[0].problem.is_none(), "{:?}", records[0].problem);
    let w = world.lock().unwrap();
    let first_grade = w
        .commands
        .iter()
        .position(|c| c.starts_with("unset LD_PRELOAD") && c.contains("exec /"))
        .expect("a stage was graded");
    let after: Vec<&String> = w.commands[first_grade..].iter().collect();
    assert!(
        after
            .iter()
            .any(|c| c.contains("BF owned") && c.contains("o=agent;"))
    );
    assert!(after.iter().any(|c| c.contains("BF baseline $(")));
    assert!(!w.commands.iter().any(|c| c.contains("BF baseline-added")));
    let copies = w
        .users
        .iter()
        .filter(|(u, c)| u == "agent" && c.contains("BF tree placed"))
        .count();
    assert_eq!(copies, 1);
    assert!(after.iter().any(|c| c.contains("chown -R 0:0 /app/data")));
}

#[tokio::test]
async fn stages_run_in_order_and_a_missed_gate_stops_the_trial() {
    // One run per attempt: the fake's state is per run.
    let (records, dir, world, _tmp) = run(fixture("two-stage"), vec![Attempt::Oracle], true).await;
    // The second stage's test-runner files are snapshotted again once the
    // first is graded, so the next handoff keeps the stage's own.
    {
        let w = world.lock().unwrap();
        let baselines: Vec<usize> = w
            .commands
            .iter()
            .enumerate()
            .filter(|(_, c)| c.contains("BF baseline"))
            .map(|(i, _)| i)
            .collect();
        let first_grade = w
            .commands
            .iter()
            .position(|c| c.starts_with("unset LD_PRELOAD") && c.contains("exec /"))
            .expect("a stage was graded");
        assert!(
            baselines.iter().any(|&b| b > first_grade),
            "no baseline after the first stage's grading"
        );
        // Grading gave the workdir to root; the second stage gives it back
        // to the agent before the agent works again.
        let owned: Vec<(usize, &String)> = w
            .commands
            .iter()
            .enumerate()
            .filter(|(_, c)| c.contains("BF owned"))
            .collect();
        let to = |c: &str, who: &str| {
            c.contains(&format!("o='{who}';")) || c.contains(&format!("o={who};"))
        };
        assert!(
            owned.iter().any(|(i, c)| *i < first_grade && to(c, "root")),
            "the workdir was not root's when the first stage was graded"
        );
        assert!(
            owned
                .iter()
                .any(|(i, c)| *i > first_grade && to(c, "agent")),
            "the second stage did not give the workdir back to the agent"
        );
    }
    let oracle = &records[0];
    assert_eq!(oracle.status, Status::Scored, "{:?}", oracle.problem);
    assert_eq!((oracle.reward, oracle.passed), (Some(1.0), Some(true)));
    assert_eq!(oracle.prompts.len(), 2, "one prompt per stage");
    let rows = result_json(&dir, oracle)["step_results"].clone();
    let rows = rows.as_array().unwrap();
    assert_eq!(rows[0]["step_name"], "first");
    assert_eq!(rows[0]["verifier_result"]["rewards"]["reward"], 1.0);
    assert_eq!(rows[0]["gate"]["passed"], true);
    assert_eq!(rows[1]["step_name"], "second");
    assert_eq!(rows[1]["reached"], true);
    let trial = dir.join(&oracle.rollout_name);
    for f in [
        "steps/first/verifier/reward.txt",
        "steps/second/verifier/reward.txt",
        "steps/first/trajectory/acp_trajectory.jsonl",
        "steps/first/agent/oracle.txt",
        "steps/second/agent/oracle.txt",
        "steps/first/artifacts/manifest.json",
        "steps/first/artifacts/one.txt",
        "verifier/reward.txt",
        "trajectory/acp_trajectory.jsonl",
    ] {
        assert!(trial.join(f).is_file(), "missing {f}");
    }
    let traj = std::fs::read_to_string(trial.join("trajectory/acp_trajectory.jsonl")).unwrap();
    assert!(traj.contains("\"stage\":\"first\"") && traj.contains("\"stage\":\"second\""));

    let (records, dir, _world, _tmp) = run(fixture("two-stage"), vec![Attempt::Nop], true).await;
    let nop = &records[0];
    assert_eq!(nop.status, Status::Scored, "{:?}", nop.problem);
    assert_eq!((nop.reward, nop.passed), (Some(0.0), Some(false)));
    let rows = result_json(&dir, nop)["step_results"].clone();
    assert_eq!(rows[0]["gate"]["passed"], false);
    assert_eq!(rows[1]["reached"], false, "the gate stopped the trial");
    assert!(!dir.join(&nop.rollout_name).join("steps/second").exists());
}

#[tokio::test]
async fn services_are_refused_by_name_where_the_backend_cannot_run_them() {
    let (records, _dir, world, _tmp) = run(fixture("sidecar"), vec![Attempt::Oracle], false).await;
    let p = records[0].problem.as_ref().unwrap();
    assert_eq!(p.code, "R401");
    assert!(
        p.message.contains("(api)") && p.message.contains("fakecloud"),
        "{}",
        p.message
    );
    assert!(p.next.contains("--sandbox docker"));
    assert!(world.lock().unwrap().commands.is_empty(), "nothing started");
}

#[tokio::test]
async fn services_start_first_the_sandbox_joins_them_and_all_are_torn_down() {
    let (records, dir, world, _tmp) = run(fixture("sidecar"), vec![Attempt::Oracle], true).await;
    let r = &records[0];
    assert_eq!(r.status, Status::Scored, "{:?}", r.problem);
    let w = world.lock().unwrap();
    assert_eq!(w.services, ["api"]);
    assert_eq!(
        w.attached, w.networks,
        "the sandbox joined the services' network"
    );
    for gone in ["svc-api", w.networks[0].as_str()] {
        assert!(
            w.removed.iter().any(|x| x == gone),
            "{gone} was not removed"
        );
    }
    assert!(
        w.commands
            .iter()
            .any(|c| c == "wget -qO- http://127.0.0.1:8000/token"),
        "the readiness check ran in the service"
    );
    assert!(r.defenses.contains_key("services_network"));
    assert!(dir.join(&r.rollout_name).join("services/api.log").is_file());
}

#[tokio::test]
async fn hooks_run_as_root_after_the_agent_and_before_grading() {
    let (records, dir, world, _tmp) =
        run(fixture("snapshot-hook"), vec![Attempt::Oracle], false).await;
    let r = &records[0];
    assert_eq!(r.status, Status::Scored, "{:?}", r.problem);
    let w = world.lock().unwrap();
    let pos = |needle: &str| {
        w.users
            .iter()
            .position(|(_, c)| c.contains(needle))
            .unwrap_or_else(|| panic!("{needle} never ran"))
    };
    let hook = pos("/root/snapshot/status.txt");
    assert_eq!(w.users[hook].0, "root");
    assert!(pos("solve.sh") < hook, "after the agent's turn");
    assert!(
        pos("kill -9 -1") < hook,
        "after every agent process is gone"
    );
    assert!(hook < pos("exec /tests/test.sh"), "before grading");
    // It works in the agent's files, so it gets the verifier's environment.
    assert!(
        w.users[hook].1.starts_with("unset LD_PRELOAD "),
        "{}",
        w.users[hook].1
    );
    assert_eq!(
        w.envs[hook].get("PATH").map(String::as_str),
        Some("/usr/bin:/bin")
    );
    let out = std::fs::read_to_string(dir.join(&r.rollout_name).join("hooks/1.txt")).unwrap();
    assert!(out.contains("as root; exit code 0"), "{out}");
}

#[tokio::test]
async fn a_separate_verifier_grades_in_its_own_image() {
    let (records, _dir, _world, _tmp) =
        run(fixture("separate-verifier"), vec![Attempt::Oracle], false).await;
    let r = &records[0];
    assert_eq!(r.status, Status::Scored, "{:?}", r.problem);
    let d = &r.defenses["verifier_image"];
    assert!(d.note.as_deref().unwrap().contains("verifier/Dockerfile"));
    assert!(r.defenses.contains_key("separate_verifier"));
}

/// `verifier_image` is recorded once the verifier's sandbox runs on it,
/// not when the image is asked for.
#[tokio::test]
async fn a_verifier_image_whose_sandbox_fails_is_not_recorded_as_used() {
    let (records, _dir, _world, _tmp) = run_failing(
        fixture("separate-verifier"),
        vec![Attempt::Oracle],
        false,
        Some("separate-verifier-verifier"),
    )
    .await;
    let r = &records[0];
    assert_ne!(r.status, Status::Scored);
    assert!(
        !r.defenses.contains_key("verifier_image"),
        "{:?}",
        r.defenses
    );
}
