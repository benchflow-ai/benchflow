//! Running commands in a container.
//!
//! Every command is `docker exec -i [-u user] [-w cwd] --env-file … <name>
//! /bin/sh -c '<wrapped>'`, where the wrapper picks bash when the image has
//! it (see [`command_shell`]). The shells are found in the system folders
//! only: `docker exec` looks a bare `sh` up through the command's `PATH`,
//! which for a root command may be the image's, with a folder the agent can
//! write on it.
//! Each command also carries [`EXEC_ID_ENV`] set to a fresh id. Killing the
//! `docker exec` client does not stop the command inside the container, so
//! a timeout (or [`ProcessHandle::kill`]) runs a second `docker exec` that
//! kills every process whose `/proc/<pid>/environ` carries that id: the
//! command and everything it started, unless a process cleared its own
//! environment. A container's root has no `CAP_SYS_PTRACE` under Docker's
//! default capabilities, so it cannot read another user's `environ`: for a
//! command run as a user, the kill runs as that user first, then as root.
//!
//! The private env file is deleted once the CLI has surely read it: when a
//! command ends, or, for a spawned process, at its first output (the CLI
//! reads the file before it asks the daemon for anything), at its end, or
//! after [`ENV_FILE_LIMIT`], whichever comes first.

use std::collections::BTreeMap;
use std::pin::Pin;
use std::process::Stdio;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use tokio::io::{AsyncRead, ReadBuf};
use tokio::sync::Notify;

use super::cli::{Cli, EnvArgs, exit_code};
use crate::capture::{OUTPUT_CAP, TailBuffer, read_tail};
use crate::shell::command_shell;
use crate::{
    EXEC_ID_ENV, Exec, ExecOutput, Process, ProcessHandle, SandboxError, TIMEOUT_EXIT_CODE,
};

/// The longest a spawned process's env file is kept when the process says
/// nothing and keeps running.
const ENV_FILE_LIMIT: Duration = Duration::from_secs(60);

/// How long to wait for output pipes to close after a timeout kill.
const DRAIN_GRACE: Duration = Duration::from_secs(5);

/// A fresh id for one command.
pub(crate) fn new_exec_id() -> String {
    uuid::Uuid::new_v4().simple().to_string()
}

/// The `docker exec` arguments for `exec`, tagged with `exec_id`.
pub(crate) fn exec_args(container: &str, exec: &Exec, env: &EnvArgs) -> Vec<String> {
    let mut args = vec!["exec".to_string(), "-i".to_string()];
    if let Some(user) = exec.user.as_deref().filter(|u| !u.is_empty()) {
        args.push("-u".to_string());
        args.push(user.to_string());
    }
    if let Some(cwd) = exec.cwd.as_deref().filter(|c| !c.is_empty()) {
        args.push("-w".to_string());
        args.push(cwd.to_string());
    }
    args.extend(env.args.iter().cloned());
    args.push(container.to_string());
    args.push("/bin/sh".to_string());
    args.push("-c".to_string());
    args.push(command_shell(&exec.command, &exec.unset));
    args
}

/// The environment of a `docker exec`: `exec`'s variables, the exec id, and
/// each of `exec.unset` that is safe to pass empty set to nothing.
/// `docker exec` cannot remove a variable of the image's, only set it, so a
/// file-naming loader variable is emptied (glibc and the shells treat empty
/// as none). A flag loader variable is never emptied — glibc acts on its
/// presence, so `-e LD_SHOW_AUXV=` would make every glibc program print its
/// auxiliary vector — and is left to the shell's own `unset`
/// ([`command_shell`]).
fn exec_env(exec: &Exec, exec_id: &str) -> BTreeMap<String, String> {
    let mut env = exec.env.clone();
    for name in &exec.unset {
        if crate::shell::empty_safe_vars().any(|v| v == name) {
            env.insert(name.clone(), String::new());
        }
    }
    env.insert(EXEC_ID_ENV.to_string(), exec_id.to_string());
    env
}

/// `-e` arguments for BenchFlow's own root plumbing (kills and copies):
/// `PATH` set to the system folders, and every file-naming loader and
/// start-up variable emptied, so the shell `docker exec` starts and what it
/// runs neither looks a program up nor loads or reads a file through the
/// image's environment. The flag loader variables are left alone (an empty
/// one would make glibc act); an image rarely sets them, and none names a
/// file. `tar` and the rest carry only this environment.
pub(crate) fn plumbing_env_args() -> Vec<String> {
    let mut args = vec![
        "-e".to_string(),
        format!("PATH={}", crate::shell::SYSTEM_PATH),
    ];
    for name in crate::shell::empty_safe_vars() {
        args.push("-e".to_string());
        args.push(format!("{name}="));
    }
    args
}

/// Whether a `docker exec -u` value names root (`root`, `0`, `0:0`, ...).
fn is_root(user: &str) -> bool {
    matches!(user.split(':').next(), Some("root" | "0"))
}

/// The users a kill of a command run as `user` runs as: that user (it alone
/// can read its processes' environments), then root.
pub(crate) fn kill_users(user: Option<&str>) -> Vec<String> {
    match user.filter(|u| !u.is_empty() && !is_root(u)) {
        Some(u) => vec![u.to_string(), "0".to_string()],
        None => vec!["0".to_string()],
    }
}

/// The `docker exec` arguments that kill the processes tagged `exec_id`, as
/// `as_user`, with the plumbing's environment ([`plumbing_env_args`]).
pub(crate) fn kill_args(container: &str, as_user: &str, exec_id: &str) -> Vec<String> {
    let mut args = vec!["exec".to_string(), "-u".to_string(), as_user.to_string()];
    args.extend(plumbing_env_args());
    args.extend([
        container.to_string(),
        "/bin/sh".to_string(),
        "-c".to_string(),
        crate::shell::kill_tagged(exec_id),
    ]);
    args
}

/// Kill every process in `container` tagged with `exec_id`, for a command
/// that ran as `user` (root when `None`).
pub(crate) async fn kill_tagged(
    cli: &Cli,
    container: &str,
    exec_id: &str,
    user: Option<&str>,
) -> Result<(), SandboxError> {
    let mut result = Ok(());
    for as_user in kill_users(user) {
        let args = kill_args(container, &as_user, exec_id);
        let out = cli.run(&args, Some(Duration::from_secs(30))).await?;
        if !out.ok() {
            result = Err(cli.daemon_problem(&out).unwrap_or_else(|| {
                SandboxError::Transport(format!(
                    "could not kill the processes of a command in {container} ({}); they may \
                     still run until the sandbox is destroyed",
                    out.message()
                ))
            }));
        }
    }
    result
}

/// Kills a command's processes in the container if dropped while armed, so
/// a cancelled exec or a dropped process does not keep running.
pub(crate) struct KillGuard {
    armed: Option<(Cli, String, String, Option<String>)>,
}

impl KillGuard {
    pub(crate) fn new(cli: &Cli, container: &str, exec_id: &str, user: Option<&str>) -> Self {
        Self {
            armed: Some((
                cli.clone(),
                container.to_string(),
                exec_id.to_string(),
                user.map(str::to_string),
            )),
        }
    }

    pub(crate) fn disarm(&mut self) {
        self.armed = None;
    }
}

impl Drop for KillGuard {
    fn drop(&mut self) {
        if let Some((cli, container, exec_id, user)) = self.armed.take()
            && let Ok(handle) = tokio::runtime::Handle::try_current()
        {
            handle.spawn(async move {
                if let Err(err) = kill_tagged(&cli, &container, &exec_id, user.as_deref()).await {
                    tracing::debug!("cleanup kill in {container} failed: {err}");
                }
            });
        }
    }
}

/// Keep `env` until `release` is notified or [`ENV_FILE_LIMIT`] passes.
fn release_on(env: EnvArgs, release: Arc<Notify>) {
    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        handle.spawn(async move {
            let _ = tokio::time::timeout(ENV_FILE_LIMIT, release.notified()).await;
            env.close();
        });
    }
}

/// A process's output stream that signals its first byte.
struct FirstOutput<R> {
    inner: R,
    release: Option<Arc<Notify>>,
}

impl<R: AsyncRead + Unpin> AsyncRead for FirstOutput<R> {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        let before = buf.filled().len();
        let polled = Pin::new(&mut self.inner).poll_read(cx, buf);
        if matches!(polled, Poll::Ready(_))
            && buf.filled().len() > before
            && let Some(release) = self.release.take()
        {
            release.notify_one();
        }
        polled
    }
}

/// The error for output that says the container itself is gone, if it does.
pub(super) fn container_gone(cli: &Cli, container: &str, stderr: &str) -> Option<SandboxError> {
    // Only the CLI's own words count: a command may well print "is not
    // running" about something else.
    let lower = stderr.to_lowercase();
    let docker_says = lower.contains("error response from daemon")
        && ["no such container", "is not running", "is paused"]
            .iter()
            .any(|m| lower.contains(m));
    let podman_says = ["no container with name or id", "container state improper"]
        .iter()
        .any(|m| lower.contains(m));
    let gone = docker_says || podman_says;
    gone.then(|| {
        SandboxError::Transport(format!(
            "the sandbox container {container} is no longer running ({}). It was removed, or it \
             stopped (out of memory?); check `{} ps -a --filter name={container}` and `{} logs \
             {container}`.",
            super::cli::tail_chars(stderr.trim(), 300),
            cli.name,
            cli.name
        ))
    })
}

/// Run `exec` to completion in `container`.
pub(super) async fn exec(
    cli: &Cli,
    container: &str,
    exec: Exec,
) -> Result<ExecOutput, SandboxError> {
    let started = Instant::now();
    let exec_id = new_exec_id();
    let env = EnvArgs::new(&exec_env(&exec, &exec_id))?;
    let args = exec_args(container, &exec, &env);
    let mut cmd = cli.command(&args);
    for (k, v) in &env.process_env {
        cmd.env(k, v);
    }
    let mut child = cmd.spawn().map_err(|e| cli.spawn_failed(e))?;
    let mut guard = KillGuard::new(cli, container, &exec_id, exec.user.as_deref());
    let stdout = tokio::spawn(read_tail(child.stdout.take().expect("piped"), OUTPUT_CAP));
    let stderr = tokio::spawn(read_tail(child.stderr.take().expect("piped"), OUTPUT_CAP));

    let waited = match exec.timeout {
        Some(limit) => tokio::time::timeout(limit, child.wait()).await.ok(),
        None => Some(child.wait().await),
    };
    let (status, timed_out) = match waited {
        Some(status) => (
            Some(status.map_err(|e| SandboxError::io("waiting for `docker exec`", e))?),
            false,
        ),
        None => {
            // Kill inside the container first, then the client.
            if let Err(err) = kill_tagged(cli, container, &exec_id, exec.user.as_deref()).await {
                tracing::warn!("timeout kill in {container}: {err}");
            }
            let _ = child.kill().await;
            (None, true)
        }
    };
    guard.disarm();
    env.close();

    let collect = |task: tokio::task::JoinHandle<TailBuffer>| async move {
        match tokio::time::timeout(DRAIN_GRACE, task).await {
            Ok(Ok(tail)) => tail.into_string(),
            _ => String::new(),
        }
    };
    let stdout = collect(stdout).await;
    let stderr = collect(stderr).await;

    if timed_out {
        return Ok(ExecOutput {
            exit_code: TIMEOUT_EXIT_CODE,
            stdout,
            stderr,
            timed_out: true,
            duration: started.elapsed(),
        });
    }
    let code = exit_code(status.expect("set when not timed out"));
    if code != 0 {
        let output = super::cli::Output {
            code,
            stdout: String::new(),
            stderr: stderr.clone(),
        };
        if let Some(err) = cli.daemon_problem(&output) {
            return Err(err);
        }
        if let Some(err) = container_gone(cli, container, &stderr) {
            return Err(err);
        }
    }
    Ok(ExecOutput {
        exit_code: code,
        stdout,
        stderr,
        timed_out: false,
        duration: started.elapsed(),
    })
}

/// Start `exec` in `container` with open pipes.
pub(super) async fn spawn(cli: &Cli, container: &str, exec: Exec) -> Result<Process, SandboxError> {
    let exec_id = new_exec_id();
    let env = EnvArgs::new(&exec_env(&exec, &exec_id))?;
    let args = exec_args(container, &exec, &env);
    let mut cmd = cli.command(&args);
    cmd.stdin(Stdio::piped());
    for (k, v) in &env.process_env {
        cmd.env(k, v);
    }
    let mut child = cmd.spawn().map_err(|e| cli.spawn_failed(e))?;
    let release = Arc::new(Notify::new());
    release_on(env, release.clone());
    let stdin = child.stdin.take().expect("piped");
    let stdout = child.stdout.take().expect("piped");
    let stderr = child.stderr.take().expect("piped");
    Ok(Process {
        stdin: Box::pin(stdin),
        stdout: Box::pin(FirstOutput {
            inner: stdout,
            release: Some(release.clone()),
        }),
        stderr: Box::pin(FirstOutput {
            inner: stderr,
            release: Some(release.clone()),
        }),
        handle: Box::new(DockerProcess {
            guard: KillGuard::new(cli, container, &exec_id, exec.user.as_deref()),
            cli: cli.clone(),
            container: container.to_string(),
            exec_id,
            user: exec.user.clone(),
            child,
            release,
        }),
    })
}

/// A command started by [`spawn`].
struct DockerProcess {
    cli: Cli,
    container: String,
    exec_id: String,
    /// The user the command runs as, for the kill.
    user: Option<String>,
    child: tokio::process::Child,
    guard: KillGuard,
    /// Deletes the env file when notified.
    release: Arc<Notify>,
}

#[async_trait]
impl ProcessHandle for DockerProcess {
    async fn wait(&mut self) -> Result<i32, SandboxError> {
        let status = self
            .child
            .wait()
            .await
            .map_err(|e| SandboxError::io("waiting for `docker exec`", e))?;
        self.guard.disarm();
        self.release.notify_one();
        Ok(exit_code(status))
    }

    async fn kill(&mut self) -> Result<(), SandboxError> {
        let result = kill_tagged(
            &self.cli,
            &self.container,
            &self.exec_id,
            self.user.as_deref(),
        )
        .await;
        let _ = self.child.kill().await;
        self.guard.disarm();
        self.release.notify_one();
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exec_args_put_user_cwd_env_and_wrapper_in_order() {
        let env = EnvArgs::new(&BTreeMap::from([("A".to_string(), "1".to_string())])).unwrap();
        let exec = Exec::new("echo hi").user("agent").cwd("/app");
        let args = exec_args("bf-x-1", &exec, &env);
        assert_eq!(&args[..6], ["exec", "-i", "-u", "agent", "-w", "/app"]);
        assert_eq!(args[6], "--env-file");
        assert_eq!(&args[8..11], ["bf-x-1", "/bin/sh", "-c"]);
        assert_eq!(args[11], command_shell("echo hi", &[]));
        assert_eq!(args.len(), 12);
    }

    /// `Exec::unset` reaches the container as empty values (which the
    /// loader and the shells ignore) and the shell unsets them, even when
    /// `env` sets one.
    #[test]
    fn unset_variables_are_emptied_then_unset() {
        let exec = Exec::new("true")
            .env("LD_PRELOAD", "/app/x.so")
            .env("A", "1")
            .unset(["LD_PRELOAD", "BASH_ENV", "LD_SHOW_AUXV"]);
        let env = exec_env(&exec, "id1");
        // File-naming and start-up variables are emptied (glibc ignores an
        // empty value); the flag is never passed, so glibc does not act on
        // it. The shell unsets all three.
        assert_eq!(env["LD_PRELOAD"], "");
        assert_eq!(env["BASH_ENV"], "");
        assert!(!env.contains_key("LD_SHOW_AUXV"), "{env:?}");
        assert_eq!(env["A"], "1");
        assert_eq!(env[EXEC_ID_ENV], "id1");
        let args = exec_args("bf-x-1", &exec, &EnvArgs::new(&env).unwrap());
        assert!(
            args.last()
                .unwrap()
                .contains("unset LD_PRELOAD BASH_ENV LD_SHOW_AUXV; "),
            "{args:?}"
        );
    }

    /// BenchFlow's own root plumbing (copies in and out, kills) finds its
    /// programs in the system folders, never through the image's `PATH`,
    /// and starts without the image's loader variables.
    #[test]
    fn root_plumbing_runs_with_the_system_path() {
        let path = format!("PATH={}", crate::shell::SYSTEM_PATH);
        for args in [
            kill_args("bf-x-1", "0", "abc"),
            super::super::files::extract_args("bf-x-1", "/app"),
            super::super::files::download_args("bf-x-1", "/logs", "verifier"),
        ] {
            let at = args.iter().position(|a| a == "bf-x-1").unwrap();
            let before = &args[..at];
            let set = |v: &str| before.windows(2).any(|w| w[0] == "-e" && w[1] == v);
            assert!(set(&path), "{args:?}");
            // The file-naming and start-up variables are emptied; the flags
            // (glibc acts on their presence) must never be passed.
            for name in crate::shell::empty_safe_vars() {
                assert!(set(&format!("{name}=")), "{name}: {args:?}");
            }
            for name in crate::shell::FLAG_LOADER_VARS {
                assert!(
                    !set(&format!("{name}=")),
                    "{name} must not be set: {args:?}"
                );
            }
        }
    }

    #[test]
    fn the_kill_script_names_the_exec_id() {
        let s = crate::shell::kill_tagged("abc123");
        assert!(s.contains(" id=abc123; "));
        assert!(s.contains("BENCHFLOW_EXEC_ID=$id"));
        assert!(s.contains("kill -9"));
    }

    #[test]
    fn a_users_command_is_killed_as_that_user_then_as_root() {
        assert_eq!(kill_users(None), ["0"]);
        assert_eq!(kill_users(Some("root")), ["0"]);
        assert_eq!(kill_users(Some("0:0")), ["0"]);
        assert_eq!(kill_users(Some("agent")), ["agent", "0"]);
        assert_eq!(kill_users(Some("1000:1000")), ["1000:1000", "0"]);
    }

    #[test]
    fn the_kill_script_is_valid_sh() {
        let out = std::process::Command::new("sh")
            .arg("-n")
            .arg("-c")
            .arg(crate::shell::kill_tagged("abc"))
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    #[test]
    fn recognizes_a_vanished_container() {
        let cli = Cli {
            name: "docker".into(),
            binary: "docker".into(),
            flavor: super::super::cli::Flavor::Docker,
        };
        assert!(
            container_gone(
                &cli,
                "c",
                "Error response from daemon: No such container: c"
            )
            .is_some()
        );
        assert!(
            container_gone(
                &cli,
                "c",
                "Error response from daemon: container abc is not running"
            )
            .is_some()
        );
        assert!(
            container_gone(
                &cli,
                "c",
                "ls: cannot access 'x': No such file or directory"
            )
            .is_none()
        );
        assert!(container_gone(&cli, "c", "nginx is not running").is_none());
    }
}
