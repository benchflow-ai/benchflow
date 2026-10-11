//! Sandboxes: the containers a task's agent and verifier run in.
//!
//! One trait, [`Sandbox`], covers every backend. A [`Provider`] turns an
//! [`ImageSpec`] into a ready image and a [`SandboxSpec`] into a running
//! sandbox. Backends:
//!
//! - [`docker`]: the `docker` CLI (or any CLI that speaks it, such as `podman`),
//!   on this machine or on a remote daemon through `DOCKER_HOST`.
//! - [`daytona`]: Daytona cloud sandboxes over its REST API.
//!
//! Every command runs through `bash -c` inside the sandbox (falling back to
//! `sh -c` when the image has no bash), as the user the request names. The
//! shells, and every program a backend runs as root on the way to the
//! command, are found in the system folders ([`shell::SYSTEM_PATH`]) or
//! named by their path, never through the image's `PATH`; the command gets
//! the `PATH` it was given.

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncRead, AsyncWrite};

pub mod archive;
mod build_log;
mod capture;
pub mod checkpoint;
pub mod daytona;
pub mod docker;
pub mod owner;
pub mod redact;
pub mod services;
pub mod shell;
mod transfer;

/// Removals started in the background for sandboxes whose operation was
/// cancelled midway (a stopped run dropped it), so the run can wait for
/// them before it exits.
static CLEANUPS: std::sync::Mutex<Vec<tokio::task::JoinHandle<()>>> =
    std::sync::Mutex::new(Vec::new());

pub(crate) fn track_cleanup(cleanup: tokio::task::JoinHandle<()>) {
    let mut all = CLEANUPS.lock().unwrap_or_else(|e| e.into_inner());
    all.retain(|c| !c.is_finished());
    all.push(cleanup);
}

/// The backend name [`left_behind`] gives a Daytona builder left to finish
/// its build (or whose state could not be read): deleting it now would
/// cancel a build other runs may share, so it is not one to remove by hand.
pub const DAYTONA_BUILDING: &str = "daytona-building";

/// Sandboxes this process could not remove inside a backend (a builder
/// whose delete failed, one left building, a failed start's cleanup), as
/// `<backend> <id>`, for the end-of-run list.
static LEFT_BEHIND: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

pub(crate) fn note_left_behind(backend: &str, id: &str) {
    LEFT_BEHIND
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push(format!("{backend} {id}"));
}

/// The sandboxes [`note_left_behind`] recorded so far.
pub fn left_behind() -> Vec<String> {
    LEFT_BEHIND
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
}

/// Wait, up to `limit`, for the removals [`track_cleanup`] started. True
/// when all finished.
pub async fn settle_cleanups(limit: std::time::Duration) -> bool {
    let all = std::mem::take(&mut *CLEANUPS.lock().unwrap_or_else(|e| e.into_inner()));
    tokio::time::timeout(limit, futures::future::join_all(all))
        .await
        .is_ok()
}

/// The result of a command that ran to completion (or timed out).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExecOutput {
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    /// The command hit its timeout and was killed; `exit_code` is then 124.
    pub timed_out: bool,
    pub duration: Duration,
}

impl ExecOutput {
    pub fn success(&self) -> bool {
        self.exit_code == 0 && !self.timed_out
    }

    /// stdout then stderr, trimmed, for error messages.
    pub fn combined(&self) -> String {
        let mut s = self.stdout.trim_end().to_string();
        let err = self.stderr.trim_end();
        if !err.is_empty() {
            if !s.is_empty() {
                s.push('\n');
            }
            s.push_str(err);
        }
        s
    }

    /// The last `n` characters of the combined output, for one-line summaries.
    pub fn tail(&self, n: usize) -> String {
        let c = self.combined();
        let count = c.chars().count();
        if count <= n {
            c
        } else {
            let skip = count - n;
            format!("…{}", c.chars().skip(skip).collect::<String>())
        }
    }
}

/// Environment variables as debug output shows them: names only, since
/// values can be keys.
struct EnvNames<'a>(&'a BTreeMap<String, String>);

impl std::fmt::Debug for EnvNames<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_map()
            .entries(self.0.keys().map(|k| (k, "<redacted>")))
            .finish()
    }
}

/// A command to run in a sandbox.
#[derive(Clone, Default)]
pub struct Exec {
    /// Shell source, run with `bash -c` (or `sh -c`).
    pub command: String,
    /// User name or uid. `None` runs as the container's default user.
    pub user: Option<String>,
    /// Working directory. `None` keeps the container's.
    pub cwd: Option<String>,
    /// Extra environment variables for this command only.
    pub env: BTreeMap<String, String>,
    /// Kill the command after this long. `None` waits forever.
    pub timeout: Option<Duration>,
    /// Variables the command runs without, the image's included: the
    /// backend removes them before any program starts, its own shells
    /// included. They win over `env`.
    pub unset: Vec<String>,
}

impl std::fmt::Debug for Exec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Exec")
            .field("command", &self.command)
            .field("user", &self.user)
            .field("cwd", &self.cwd)
            .field("env", &EnvNames(&self.env))
            .field("timeout", &self.timeout)
            .field("unset", &self.unset)
            .finish()
    }
}

impl Exec {
    pub fn new(command: impl Into<String>) -> Self {
        Self {
            command: command.into(),
            ..Self::default()
        }
    }
    pub fn user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }
    pub fn root(self) -> Self {
        self.user("root")
    }
    /// BenchFlow's own command: as root, with `PATH` set to the system
    /// folders ([`shell::SYSTEM_PATH`]) and without the loader and shell
    /// start-up variables ([`shell::LOADER_VARS`]), so nothing the image's
    /// `PATH` or environment names (a folder or file the agent can write)
    /// is run, loaded, or read as root.
    pub fn system(self) -> Self {
        self.root()
            .env("PATH", shell::SYSTEM_PATH)
            .unset(shell::LOADER_VARS.iter().copied())
            .unset(shell::SYSTEM_EXTRA_VARS.iter().copied())
    }
    /// Run without these variables (see [`Exec::unset`](Self#structfield.unset)).
    pub fn unset<S: Into<String>>(mut self, names: impl IntoIterator<Item = S>) -> Self {
        for name in names {
            let name = name.into();
            if !self.unset.contains(&name) {
                self.unset.push(name);
            }
        }
        self
    }
    pub fn cwd(mut self, cwd: impl Into<String>) -> Self {
        self.cwd = Some(cwd.into());
        self
    }
    pub fn env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.insert(key.into(), value.into());
        self
    }
    pub fn envs(mut self, vars: impl IntoIterator<Item = (String, String)>) -> Self {
        self.env.extend(vars);
        self
    }
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }
    pub fn timeout_opt(mut self, timeout: Option<Duration>) -> Self {
        self.timeout = timeout;
        self
    }
}

/// A long-running process with open pipes, such as an ACP agent.
pub struct Process {
    pub stdin: Pin<Box<dyn AsyncWrite + Send>>,
    pub stdout: Pin<Box<dyn AsyncRead + Send>>,
    pub stderr: Pin<Box<dyn AsyncRead + Send>>,
    pub handle: Box<dyn ProcessHandle>,
}

impl fmt::Debug for Process {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Process").finish_non_exhaustive()
    }
}

/// Control over a spawned [`Process`].
#[async_trait]
pub trait ProcessHandle: Send + Sync {
    /// Wait for the process to exit and return its exit code.
    async fn wait(&mut self) -> Result<i32, SandboxError>;
    /// Kill the process (and, where the backend can, its children).
    async fn kill(&mut self) -> Result<(), SandboxError>;
}

/// A running sandbox.
#[async_trait]
pub trait Sandbox: Send + Sync {
    /// The backend's id for this sandbox (a container name, a Daytona id).
    fn id(&self) -> &str;
    /// The backend's name: `docker`, `podman`, `daytona`.
    fn backend(&self) -> &str;

    /// Run a command to completion.
    async fn exec(&self, exec: Exec) -> Result<ExecOutput, SandboxError>;

    /// Start a command and hand back its pipes.
    async fn spawn(&self, exec: Exec) -> Result<Process, SandboxError>;

    /// Copy a local file or folder to `remote` (an absolute path). A folder's
    /// contents land in `remote`, which is created. Files are owned by root.
    async fn upload(&self, local: &Path, remote: &str) -> Result<(), SandboxError>;

    /// Write bytes to a file in the sandbox, creating parent folders.
    async fn write_file(
        &self,
        remote: &str,
        contents: &[u8],
        mode: u32,
    ) -> Result<(), SandboxError>;

    /// Copy a file or folder out of the sandbox to `local`. A folder's
    /// contents land in `local`, which is created.
    async fn download(&self, remote: &str, local: &Path) -> Result<(), SandboxError>;

    /// Stop the sandbox and delete it.
    async fn destroy(&self) -> Result<(), SandboxError>;

    /// What the backend itself says about this sandbox (`docker inspect`,
    /// Daytona's API), never a program inside the sandbox, which the agent
    /// may have replaced. A backend that cannot tell answers
    /// [`SandboxError::Unsupported`].
    async fn presence(&self) -> Result<Presence, SandboxError> {
        Err(SandboxError::Unsupported(format!(
            "the {} backend cannot say whether a sandbox still runs",
            self.backend()
        )))
    }

    /// [`Sandbox::download`], failing (with [`std::io::ErrorKind::FileTooLarge`])
    /// before more than `max_bytes` of file content arrive. Backends without
    /// a cap download it all.
    async fn download_capped(
        &self,
        remote: &str,
        local: &Path,
        max_bytes: u64,
    ) -> Result<(), SandboxError> {
        let _ = max_bytes;
        self.download(remote, local).await
    }

    /// Change a running sandbox's network, where the backend can
    /// ([`Capabilities::network_switch`]).
    async fn set_network(&self, policy: &NetworkPolicy) -> Result<(), SandboxError> {
        let _ = policy;
        Err(SandboxError::Unsupported(format!(
            "the {} backend cannot change a running sandbox's network",
            self.backend()
        )))
    }

    /// Store this sandbox's filesystem as a checkpoint named `name` (see
    /// [`checkpoint::name`]) that [`Provider::create_from_checkpoint`] can
    /// start new sandboxes from, where the backend can
    /// ([`Capabilities::checkpoints`]). `scrub_env` names the variables the
    /// sandbox was started with: the checkpoint must not keep their values
    /// (a child sets them again). Processes and memory are not kept.
    async fn checkpoint(
        &self,
        name: &str,
        scrub_env: &[String],
    ) -> Result<checkpoint::Checkpoint, SandboxError> {
        let _ = (name, scrub_env);
        Err(SandboxError::Unsupported(format!(
            "the {} backend cannot take checkpoints",
            self.backend()
        )))
    }
}

/// What a backend says about a sandbox it made.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Presence {
    /// It runs.
    Running,
    /// The backend still has it, but it does not run: it stopped, or code
    /// inside stopped it.
    Stopped,
    /// The backend no longer has it: something outside the run removed it
    /// (code inside cannot).
    Gone,
}

/// How an image is obtained.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ImageSource {
    /// A prebuilt image reference, pulled if missing.
    Pull { reference: String },
    /// Build `dockerfile` with `context` as the build context.
    Build {
        context: PathBuf,
        dockerfile: PathBuf,
    },
}

/// An image to prepare, plus optional layers on top of it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageSpec {
    pub source: ImageSource,
    /// Shell steps baked on top of the source image as root, in order (agent
    /// installs). Each becomes a `RUN` line.
    pub layers: Vec<String>,
    /// A short name for tags and logs (`hello-world`).
    pub name: String,
    /// Content hash of everything that decides the image. Same hash, same image.
    pub digest: String,
    pub build_timeout: Option<Duration>,
    pub platform: Option<String>,
}

/// A prepared image a sandbox can start from.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Image {
    /// What the backend starts from: a local tag, a registry reference, or a
    /// Daytona snapshot name.
    pub reference: String,
    /// Built (or pulled) now rather than found in the cache.
    pub fresh: bool,
    pub duration: Duration,
}

/// A sandbox's network, as the backend enforces it from outside the sandbox.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "kebab-case")]
pub enum NetworkPolicy {
    /// Full internet access.
    #[default]
    Open,
    /// No network at all.
    None,
    /// The network is on and filtered from inside: root in the sandbox may
    /// install a packet filter (Docker grants it `NET_ADMIN`). When `allow`
    /// is set, a backend that can ([`Capabilities::network_allow_cidrs`])
    /// also limits the whole sandbox to those CIDR ranges itself.
    Filtered { allow: Option<Vec<String>> },
}

/// What a sandbox should look like.
#[derive(Clone, Serialize, Deserialize)]
pub struct SandboxSpec {
    pub image: Image,
    /// A name for logs and backend labels, such as the trial name.
    pub name: String,
    pub cpus: Option<f64>,
    pub memory_mb: Option<u64>,
    pub disk_mb: Option<u64>,
    pub gpus: u32,
    /// Most processes the sandbox may hold at once (fork-bomb guard).
    pub pids_limit: Option<u32>,
    pub network: NetworkPolicy,
    pub env: BTreeMap<String, String>,
    pub workdir: Option<String>,
    pub labels: BTreeMap<String, String>,
    /// The container runtime to start under (`runsc` for gVisor); `None`:
    /// the backend's default. Backends with no choice of runtime refuse it.
    #[serde(default)]
    pub runtime: Option<String>,
}

impl std::fmt::Debug for SandboxSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SandboxSpec")
            .field("image", &self.image)
            .field("name", &self.name)
            .field("cpus", &self.cpus)
            .field("memory_mb", &self.memory_mb)
            .field("disk_mb", &self.disk_mb)
            .field("gpus", &self.gpus)
            .field("pids_limit", &self.pids_limit)
            .field("network", &self.network)
            .field("env", &EnvNames(&self.env))
            .field("workdir", &self.workdir)
            .field("labels", &self.labels)
            .field("runtime", &self.runtime)
            .finish()
    }
}

/// What a backend can do, for capability checks before a run starts.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Capabilities {
    pub network_none: bool,
    pub gpus: bool,
    pub pids_limit: bool,
    pub build_from_dockerfile: bool,
    /// Largest disk the backend gives one sandbox, if it caps it.
    pub max_disk_mb: Option<u64>,
    /// Root in a sandbox started with [`NetworkPolicy::Filtered`] can install
    /// a packet filter, given iptables in the image.
    pub firewall: bool,
    /// The backend itself limits a sandbox to a list of CIDR ranges, of at
    /// most this many entries.
    pub network_allow_cidrs: Option<usize>,
    /// [`Sandbox::set_network`] changes a running sandbox's network.
    pub network_switch: bool,
    /// A service inside each sandbox runs commands as root for any process
    /// that reaches it (Daytona's daemon), so the agent user is kept from
    /// root only if something blocks that service.
    pub root_service: bool,
    /// [`Sandbox::checkpoint`] stores a sandbox's filesystem and
    /// [`Provider::create_from_checkpoint`] starts sandboxes from it.
    #[serde(default)]
    pub checkpoints: bool,
    /// The backend's own `none` leaves IPv6 open (Daytona's block, seen
    /// 2026-10-10 reaching [2606:4700:4700::1111]:443), so a phase with no
    /// network also needs the uid firewall, which filters IPv6 too.
    pub network_none_leaks_ipv6: bool,
}

/// The state of a backend, for `bench doctor`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderStatus {
    pub ok: bool,
    /// One line: what was found ("Docker 28.1.1 on unix:///var/run/docker.sock").
    pub summary: String,
    /// What to do when not ok.
    pub fix: Option<String>,
}

/// A sandbox backend.
#[async_trait]
pub trait Provider: Send + Sync {
    fn name(&self) -> &str;
    fn capabilities(&self) -> Capabilities;
    /// Check that the backend is reachable and usable, without side effects.
    async fn status(&self) -> ProviderStatus;
    /// Build or pull an image (and its layers), reusing a cached one when the
    /// digest matches. `log` receives build output lines.
    ///
    /// The callback is spelled `for<'a> Fn(&'a str)` on purpose: written
    /// `Fn(&str)`, `async_trait` names the inner lifetime, and an
    /// implementation could then not pass a line it formatted itself.
    async fn prepare(
        &self,
        spec: &ImageSpec,
        log: &(dyn for<'a> Fn(&'a str) + Send + Sync),
    ) -> Result<Image, SandboxError>;
    /// Start a sandbox.
    async fn create(&self, spec: &SandboxSpec) -> Result<Box<dyn Sandbox>, SandboxError>;
    /// Remove sandboxes this machine's runs left behind (crashed runs).
    async fn sweep(&self) -> Result<usize, SandboxError> {
        Ok(0)
    }
    /// The container runtimes a sandbox can be started under
    /// ([`SandboxSpec::runtime`]); `None` when the backend offers no choice.
    async fn runtimes(&self) -> Result<Option<Vec<String>>, SandboxError> {
        Ok(None)
    }

    /// The run needs `image` no more (its last trial finished): a backend
    /// that keeps images may now reclaim it when space runs low.
    async fn release(&self, _image: &Image) {}

    /// Services beside a sandbox, when the backend can run them.
    fn services(&self) -> Option<&dyn services::ServiceHost> {
        None
    }

    /// Start a sandbox from a checkpoint: `spec` as for [`Provider::create`]
    /// (its `image` is ignored), the filesystem the checkpoint's.
    async fn create_from_checkpoint(
        &self,
        spec: &SandboxSpec,
        checkpoint: &checkpoint::Checkpoint,
    ) -> Result<Box<dyn Sandbox>, SandboxError> {
        let _ = (spec, checkpoint);
        Err(SandboxError::Unsupported(format!(
            "the {} backend cannot start sandboxes from checkpoints",
            self.name()
        )))
    }

    /// Delete a checkpoint. `Ok(false)`: the backend accepted the delete but
    /// has not finished it (0.8's `delete_deferred`).
    async fn delete_checkpoint(
        &self,
        checkpoint: &checkpoint::Checkpoint,
    ) -> Result<bool, SandboxError> {
        let _ = checkpoint;
        Ok(true)
    }

    /// The checkpoints of this owner scope (all hosts), for `bench cache`
    /// and the sweep.
    async fn list_checkpoints(&self) -> Result<Vec<checkpoint::Listed>, SandboxError> {
        Ok(Vec::new())
    }
}

/// Whether the build context's `.dockerignore` leaves `rel` (relative,
/// `/`-separated) out of what a build of `dockerfile` receives.
pub fn dockerignore(context: &Path, dockerfile: &Path) -> impl Fn(&str) -> bool + use<> {
    let ignore = daytona::dockerignore::DockerIgnore::load(context, dockerfile);
    move |rel: &str| ignore.excludes(rel)
}

/// Why a sandbox operation failed, and whose problem it is.
#[derive(Debug, thiserror::Error)]
pub enum SandboxError {
    /// The backend is not reachable or not configured (no daemon, no key).
    #[error("{message}")]
    Unavailable { message: String, fix: String },
    /// The image could not be built or pulled.
    #[error("image {name}: {message}")]
    Image {
        name: String,
        message: String,
        /// The tail of the build log.
        log_tail: String,
        /// The Dockerfile itself failed (a task problem), rather than the
        /// backend (an infrastructure problem).
        task_fault: bool,
    },
    /// The sandbox could not be started.
    #[error("sandbox did not start: {0}")]
    Start(String),
    /// A command or transfer failed at the transport level.
    #[error("{0}")]
    Transport(String),
    /// A copy failed on what it copies: `tar` in the sandbox ran and
    /// refused some of it (a file it cannot read, a full disk), not the
    /// connection to the sandbox.
    #[error("{0}")]
    Content(String),
    /// The backend refused something the task asks for.
    #[error("{0}")]
    Unsupported(String),
    #[error("{context}: {source}")]
    Io {
        context: String,
        #[source]
        source: std::io::Error,
    },
}

impl SandboxError {
    pub fn io(context: impl Into<String>, source: std::io::Error) -> Self {
        Self::Io {
            context: context.into(),
            source,
        }
    }
    /// True when a download failed because the path is not in the sandbox
    /// (the backends' "downloading <path>" not-found), not on this machine.
    pub fn is_missing_in_sandbox(&self) -> bool {
        matches!(self, Self::Io { context, source }
            if source.kind() == std::io::ErrorKind::NotFound && context.starts_with("downloading "))
    }

    /// True when retrying the same operation may succeed.
    pub fn is_transient(&self) -> bool {
        match self {
            Self::Transport(_) | Self::Start(_) | Self::Content(_) => true,
            Self::Image { task_fault, .. } => !task_fault,
            _ => false,
        }
    }
}

/// Names of the backends this build knows, for help text and errors.
pub const BACKENDS: &[&str] = &["docker", "podman", "daytona"];

/// Make a provider by name. `docker` and `podman` use their CLIs; `daytona`
/// reads `DAYTONA_API_KEY` (and `DAYTONA_API_URL`, `DAYTONA_TARGET`).
pub fn provider(name: &str) -> Result<Box<dyn Provider>, SandboxError> {
    match name {
        "docker" => Ok(Box::new(docker::DockerProvider::new("docker"))),
        "podman" => Ok(Box::new(docker::DockerProvider::new("podman"))),
        "daytona" => Ok(Box::new(daytona::DaytonaProvider::from_env()?)),
        other => Err(SandboxError::Unsupported(format!(
            "unknown sandbox `{other}`; this build has {}",
            BACKENDS.join(", ")
        ))),
    }
}

/// The label every sandbox and image BenchFlow creates carries, so a sweep
/// removes only what BenchFlow made.
pub const OWNER_LABEL: &str = "ai.benchflow.owner";

/// The label naming the host whose BenchFlow process created a sandbox.
pub const HOST_LABEL: &str = "ai.benchflow.host";

/// The label holding the process id of the BenchFlow process that created a
/// sandbox. With [`HOST_LABEL`] it lets a sweep tell a crashed run's
/// leftovers from a live run's sandboxes.
pub const PID_LABEL: &str = "ai.benchflow.pid";

/// The environment variable every command a backend starts carries, set to a
/// fresh id per command. A timeout or a kill finds the command's processes,
/// children included, by this variable in `/proc/<pid>/environ` and kills
/// them inside the sandbox, not just the client that started them.
pub const EXEC_ID_ENV: &str = "BENCHFLOW_EXEC_ID";

/// Exit code reported for a command killed at its timeout, as GNU `timeout`
/// reports it.
pub const TIMEOUT_EXIT_CODE: i32 = 124;

#[cfg(test)]
mod debug_tests {
    use super::*;

    #[test]
    fn a_system_command_is_roots_with_the_system_path_and_no_loader_variables() {
        let exec = Exec::new("true").env("A", "1").system().unset(["A"]);
        assert_eq!(exec.user.as_deref(), Some("root"));
        assert_eq!(exec.env["PATH"], shell::SYSTEM_PATH);
        for v in shell::LOADER_VARS.iter().chain(shell::SYSTEM_EXTRA_VARS) {
            assert!(exec.unset.iter().any(|u| u == v), "{v}");
        }
        let dropped = shell::LOADER_VARS.len() + shell::SYSTEM_EXTRA_VARS.len();
        assert_eq!(exec.unset.len(), dropped + 1);
        // Naming a variable twice keeps one entry.
        assert_eq!(exec.unset(["A"]).unset.len(), dropped + 1);
    }

    /// Debug output names a command's and a sandbox's variables, never
    /// their values, which can be keys.
    #[test]
    fn debug_output_leaves_variable_values_out() {
        let exec = Exec::new("true").env("OPENAI_API_KEY", "sk-secret-value-123");
        let shown = format!("{exec:?}");
        assert!(shown.contains("OPENAI_API_KEY"), "{shown}");
        assert!(!shown.contains("sk-secret-value-123"), "{shown}");
        let spec = SandboxSpec {
            image: Image {
                reference: "ubuntu:24.04".into(),
                fresh: false,
                duration: Duration::from_secs(0),
            },
            name: "t".into(),
            cpus: None,
            memory_mb: None,
            disk_mb: None,
            gpus: 0,
            pids_limit: None,
            network: NetworkPolicy::Open,
            env: BTreeMap::from([("TOKEN".to_string(), "sk-secret-value-123".to_string())]),
            workdir: None,
            labels: BTreeMap::new(),
            runtime: None,
        };
        let shown = format!("{spec:?}");
        assert!(
            shown.contains("TOKEN") && !shown.contains("sk-secret-value-123"),
            "{shown}"
        );
    }
}
