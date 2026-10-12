//! `bench agents check`: does an agent install and answer ACP?
//!
//! A check builds the agent's install layer on a small image (no task),
//! starts the agent as an unprivileged user the way a trial does, sends
//! ACP's `initialize`, and reads back who the agent says it is: its name,
//! version, protocol version, and the login methods it offers. It sends no
//! prompt and needs no login, so it says nothing about whether the agent
//! can solve a task or reach a model; only that this build installs on this
//! backend and speaks ACP.
//!
//! Each result is appended to `agent-checks.jsonl` in BenchFlow's cache
//! folder, keyed by the agent's version fingerprint, so `bench agents` can
//! show which builds were checked, where, and when.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use benchflow_agent::AgentManifest;
use benchflow_agent::acp::{AcpClient, ClientOptions};
use benchflow_agent::manifest::Harness;
use benchflow_sandbox::{Exec, NetworkPolicy, Provider, Sandbox, SandboxError, SandboxSpec};
use serde::{Deserialize, Serialize};

use crate::acp_agent::acp_problem;
use crate::hardening::{self, Report};
use crate::images;
use crate::outcome::{Owner, Phase, Problem};
use crate::trial::sandbox_problem;

/// The image a check installs agents on when none is named.
pub const DEFAULT_IMAGE: &str = "ubuntu:24.04";
/// How long the agent may take to answer `initialize`.
pub const DEFAULT_HANDSHAKE: Duration = Duration::from_secs(120);
/// The file checks are recorded in, under BenchFlow's cache folder.
pub const RECORD_FILE: &str = "agent-checks.jsonl";
const USER: &str = "agent";
const WORKDIR: &str = "/app";
const STDERR_TAIL: usize = 2000;

/// What to check on.
#[derive(Debug, Clone)]
pub struct CheckSpec {
    pub image: String,
    pub handshake_timeout: Duration,
}

impl Default for CheckSpec {
    fn default() -> Self {
        Self {
            image: DEFAULT_IMAGE.to_string(),
            handshake_timeout: DEFAULT_HANDSHAKE,
        }
    }
}

/// One check's result, as recorded.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CheckRecord {
    pub agent: String,
    pub fingerprint: String,
    pub backend: String,
    pub image: String,
    /// When the check finished, RFC 3339 UTC.
    pub date: String,
    /// `"ok"` when the agent answered `initialize`, else `"failed"`.
    pub outcome: String,
    /// The failure's code (`R201` install, `R202` did not start, ...).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub problem: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owner: Option<String>,
    /// Seconds to build (or find) the image with the agent installed.
    pub install_secs: f64,
    /// Seconds from starting the agent to its `initialize` answer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start_secs: Option<f64>,
    /// The image was built now rather than found in the backend's cache.
    #[serde(default)]
    pub built_now: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub protocol_version: Option<u64>,
    /// The login methods the agent offers, by id.
    #[serde(default)]
    pub auth_methods: Vec<String>,
    /// The tail of the agent's stderr or the install log, on failure.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evidence: Option<String>,
}

impl CheckRecord {
    pub fn ok(&self) -> bool {
        self.outcome == "ok"
    }
}

/// Install `manifest`'s agent on `spec.image`, start it, and run the ACP
/// handshake. `log` receives the image build's output.
pub async fn check(
    provider: &dyn Provider,
    manifest: &AgentManifest,
    spec: &CheckSpec,
    log: &(dyn Fn(&str) + Send + Sync),
) -> CheckRecord {
    let mut record = CheckRecord {
        agent: manifest.name.clone(),
        fingerprint: manifest.version_fingerprint(),
        backend: provider.name().to_string(),
        image: spec.image.clone(),
        date: String::new(),
        outcome: "failed".into(),
        code: None,
        problem: None,
        owner: None,
        install_secs: 0.0,
        start_secs: None,
        built_now: false,
        agent_name: None,
        agent_version: None,
        protocol_version: None,
        auth_methods: Vec::new(),
        evidence: None,
    };
    if let Err(p) = run(provider, manifest, spec, log, &mut record).await {
        record.code = Some(p.code.clone());
        record.owner = Some(p.owner.label().to_string());
        record.problem = Some(format!("{}; next: {}", p.message, p.next));
        record.evidence = p.detail.clone().filter(|d| !d.trim().is_empty());
    } else {
        record.outcome = "ok".into();
    }
    record.date = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();
    record
}

/// A check starts the agent over ACP; an agent without an ACP mode has
/// nothing to start. Its launch command named only BenchFlow's bin folder
/// (`/opt/benchflow/bin/: Is a directory`), and the check blamed the agent
/// (R202) after a full install.
fn without_acp(manifest: &AgentManifest) -> Option<Problem> {
    (!manifest.declares(Harness::Acp)).then(|| {
        Problem::new(
            "R215",
            Owner::You,
            Phase::Setup,
            format!(
                "{} has no ACP mode (it runs headless only), and a check starts the agent over ACP",
                manifest.name
            ),
            format!(
                "check it on a task instead: `bench run <task> -a {}`",
                manifest.name
            ),
        )
    })
}

async fn run(
    provider: &dyn Provider,
    manifest: &AgentManifest,
    spec: &CheckSpec,
    log: &(dyn Fn(&str) + Send + Sync),
    record: &mut CheckRecord,
) -> Result<(), Problem> {
    if let Some(p) = without_acp(manifest) {
        return Err(p);
    }
    let name = format!("check-{}", manifest.name.replace([':', '@'], "-"));
    let mut image_spec = images::pull_spec(&name, &spec.image, vec![manifest.install_layer()]);
    image_spec.build_timeout = Some(manifest.install.timeout);
    let started = Instant::now();
    let image = images::Images::new()
        .get(provider, &image_spec, log, Some(&manifest.name))
        .await;
    record.install_secs = started.elapsed().as_secs_f64();
    let image = image?;
    record.built_now = image.fresh;

    let mut labels = std::collections::BTreeMap::new();
    labels.insert(
        "ai.benchflow.check".to_string(),
        images::sanitize_name(&name),
    );
    let sandbox_spec = SandboxSpec {
        image,
        name: images::sanitize_name(&name),
        cpus: None,
        memory_mb: None,
        disk_mb: None,
        gpus: 0,
        pids_limit: None,
        network: NetworkPolicy::Open,
        env: Default::default(),
        workdir: Some(WORKDIR.to_string()),
        labels,
        runtime: None,
    };
    let sandbox = provider.create(&sandbox_spec).await.map_err(|e| match e {
        SandboxError::Start(why) => Problem::new(
            "R103",
            Owner::Sandbox,
            Phase::Start,
            format!("the sandbox did not start: {why}"),
            "retry; `bench doctor` checks the backend",
        ),
        SandboxError::Transport(_) | SandboxError::Throttled(_) => Problem::new(
            "R103",
            Owner::Sandbox,
            Phase::Start,
            format!("the sandbox did not start: {e}"),
            "retry; `bench doctor` checks the backend",
        ),
        other => sandbox_problem(Phase::Start, other),
    })?;
    let result = handshake(sandbox.as_ref(), manifest, spec, record).await;
    let _ = sandbox.destroy().await;
    result
}

async fn handshake(
    sandbox: &dyn Sandbox,
    manifest: &AgentManifest,
    spec: &CheckSpec,
    record: &mut CheckRecord,
) -> Result<(), Problem> {
    let out = sandbox
        .exec(Exec::new(hardening::setup_user(USER, WORKDIR)).system())
        .await
        .map_err(|e| sandbox_problem(Phase::Setup, e))?;
    if !out.success() {
        return Err(Problem::new(
            "R104",
            Owner::You,
            Phase::Setup,
            format!("could not create the agent user on {}", spec.image),
            "pick an image with useradd or adduser (--image)",
        )
        .detail(out.tail(STDERR_TAIL)));
    }
    let report = Report::parse(&out.stdout);
    let home = report
        .first("user")
        .and_then(|l| l.split_whitespace().nth(2))
        .map(str::to_string)
        .unwrap_or_else(|| format!("/home/{USER}"));
    let mut env: std::collections::BTreeMap<String, String> = manifest
        .launch_env()
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    env.insert("HOME".into(), home);
    let exec = Exec::new(manifest.launch_command())
        .user(USER.to_string())
        .cwd(WORKDIR.to_string())
        .envs(env);
    let started = Instant::now();
    let process = sandbox
        .spawn(exec)
        .await
        .map_err(|e| sandbox_problem(Phase::Agent, e))?;
    let mut options = ClientOptions::for_agent(manifest, None);
    options.handshake_timeout = spec.handshake_timeout;
    let mut client = AcpClient::start(process, options);
    let answer = client.initialize().await;
    let stderr = client.stderr_tail(STDERR_TAIL);
    let _ = client.shutdown().await;
    let info = answer.map_err(|e| acp_problem(&e, &manifest.name, false, stderr))?;
    record.start_secs = Some(started.elapsed().as_secs_f64());
    record.agent_name = info.name.or(info.title);
    record.agent_version = info.version;
    record.protocol_version = Some(info.protocol_version);
    record.auth_methods = info
        .auth_methods
        .iter()
        .filter_map(|m| {
            m.get("id")
                .or_else(|| m.get("name"))
                .and_then(|v| v.as_str())
                .map(str::to_string)
        })
        .collect();
    Ok(())
}

/// The record file under `cache_dir`.
pub fn record_path(cache_dir: &Path) -> PathBuf {
    cache_dir.join(RECORD_FILE)
}

/// Append `record` to the record file.
pub fn save(cache_dir: &Path, record: &CheckRecord) -> std::io::Result<()> {
    use std::io::Write;
    std::fs::create_dir_all(cache_dir)?;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(record_path(cache_dir))?;
    let line = serde_json::to_string(record).map_err(std::io::Error::other)?;
    writeln!(f, "{line}")
}

/// Every recorded check, oldest first. Unreadable lines are skipped.
pub fn load(cache_dir: &Path) -> Vec<CheckRecord> {
    std::fs::read_to_string(record_path(cache_dir))
        .unwrap_or_default()
        .lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect()
}

/// The latest check of exactly this build of `agent` (its name and
/// fingerprint), on any backend.
pub fn latest<'a>(
    records: &'a [CheckRecord],
    agent: &str,
    fingerprint: &str,
) -> Option<&'a CheckRecord> {
    records
        .iter()
        .rev()
        .find(|r| r.agent == agent && r.fingerprint == fingerprint)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(agent: &str, fp: &str, outcome: &str, date: &str) -> CheckRecord {
        CheckRecord {
            agent: agent.into(),
            fingerprint: fp.into(),
            backend: "daytona".into(),
            image: DEFAULT_IMAGE.into(),
            date: date.into(),
            outcome: outcome.into(),
            code: None,
            problem: None,
            owner: None,
            install_secs: 1.0,
            start_secs: None,
            built_now: true,
            agent_name: None,
            agent_version: None,
            protocol_version: None,
            auth_methods: Vec::new(),
            evidence: None,
        }
    }

    #[test]
    fn a_headless_only_agent_is_refused_before_its_install() {
        let all = AgentManifest::builtins();
        let get = |n: &str| all.iter().find(|m| m.name == n).unwrap().clone();
        for name in ["muse-code", "rovodev-cli", "aider", "openhands-sdk"] {
            let p = without_acp(&get(name)).unwrap_or_else(|| panic!("{name}"));
            assert_eq!((p.code.as_str(), p.owner), ("R215", Owner::You), "{name}");
            assert!(p.message.contains("has no ACP mode"), "{}", p.message);
        }
        for name in ["claude", "goose", "hermes"] {
            assert!(without_acp(&get(name)).is_none(), "{name}");
        }
    }

    #[test]
    fn records_round_trip_and_only_the_same_build_counts() {
        let tmp = tempfile::tempdir().unwrap();
        save(
            tmp.path(),
            &record("goose", "a", "failed", "2026-10-09T00:00:00Z"),
        )
        .unwrap();
        save(
            tmp.path(),
            &record("goose", "a", "ok", "2026-10-10T00:00:00Z"),
        )
        .unwrap();
        save(
            tmp.path(),
            &record("goose", "b", "failed", "2026-10-10T01:00:00Z"),
        )
        .unwrap();
        std::fs::write(
            record_path(tmp.path()),
            std::fs::read_to_string(record_path(tmp.path())).unwrap() + "not json\n",
        )
        .unwrap();
        let all = load(tmp.path());
        assert_eq!(all.len(), 3);
        let a = latest(&all, "goose", "a").unwrap();
        assert!(a.ok());
        assert_eq!(a.date, "2026-10-10T00:00:00Z");
        assert!(!latest(&all, "goose", "b").unwrap().ok());
        assert!(latest(&all, "goose", "c").is_none());
        assert!(latest(&all, "kimi", "a").is_none());
    }
}
