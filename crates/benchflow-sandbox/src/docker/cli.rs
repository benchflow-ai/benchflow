//! Running the docker (or podman) CLI.
//!
//! Every call inherits this process's environment, so `DOCKER_HOST`,
//! `DOCKER_CONTEXT`, `DOCKER_CONFIG` and podman's connection settings apply
//! with no extra setup. This module spawns the CLI, captures its output with
//! a cap, tells a missing CLI or an unreachable daemon apart from a command
//! that simply failed, and passes environment variables through a private
//! file so their values never appear in the host's process table.

use std::collections::BTreeMap;
use std::io::Write as _;
use std::os::unix::process::ExitStatusExt;
use std::path::PathBuf;
use std::process::{ExitStatus, Stdio};
use std::time::Duration;

use tokio::process::Command;

use crate::SandboxError;
use crate::capture::{OUTPUT_CAP, read_tail};

/// Which CLI dialect to speak.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Flavor {
    Docker,
    Podman,
}

/// A docker-compatible CLI.
#[derive(Debug, Clone)]
pub(crate) struct Cli {
    /// `docker` or `podman`, for messages and [`crate::Sandbox::backend`].
    pub name: String,
    /// What to execute: a name looked up on `PATH`, or a path.
    pub binary: PathBuf,
    pub flavor: Flavor,
}

/// A finished CLI call.
#[derive(Debug)]
pub(crate) struct Output {
    pub code: i32,
    pub stdout: String,
    pub stderr: String,
}

impl Output {
    pub fn ok(&self) -> bool {
        self.code == 0
    }

    /// The CLI's own words for what happened: stderr, else stdout, trimmed
    /// to its last 2000 characters.
    pub fn message(&self) -> String {
        let text = if self.stderr.trim().is_empty() {
            self.stdout.trim()
        } else {
            self.stderr.trim()
        };
        tail_chars(text, 2000)
    }
}

/// The last `n` characters of `text`, with `…` in front when cut.
pub(crate) fn tail_chars(text: &str, n: usize) -> String {
    let count = text.chars().count();
    if count <= n {
        text.to_string()
    } else {
        format!("…{}", text.chars().skip(count - n).collect::<String>())
    }
}

/// The exit code of a finished process; 128 + the signal when it was killed.
pub(crate) fn exit_code(status: ExitStatus) -> i32 {
    status
        .code()
        .unwrap_or_else(|| 128 + status.signal().unwrap_or(9))
}

impl Cli {
    /// A command for the CLI with `args`, killed if dropped, stdin closed,
    /// stdout and stderr piped.
    pub fn command<I, S>(&self, args: I) -> Command
    where
        I: IntoIterator<Item = S>,
        S: AsRef<std::ffi::OsStr>,
    {
        let mut cmd = Command::new(&self.binary);
        // A process group of its own: the terminal sends Ctrl-C to bench's
        // whole group, and a `docker exec` that got it would end the trial
        // a drain (the first Ctrl-C) means to let finish. A stop still ends
        // these children (kill_on_drop, and the container's own cleanup).
        #[cfg(unix)]
        cmd.process_group(0);
        cmd.args(args)
            .kill_on_drop(true)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        cmd
    }

    /// Run `args` to completion, capturing output. `timeout` kills the CLI.
    pub async fn run(
        &self,
        args: &[String],
        timeout: Option<Duration>,
    ) -> Result<Output, SandboxError> {
        self.run_command(self.command(args), args, timeout).await
    }

    /// Run a prepared command to completion, capturing output.
    pub async fn run_command(
        &self,
        mut cmd: Command,
        args: &[String],
        timeout: Option<Duration>,
    ) -> Result<Output, SandboxError> {
        let mut child = cmd.spawn().map_err(|e| self.spawn_failed(e))?;
        let stdout = child
            .stdout
            .take()
            .map(|s| tokio::spawn(read_tail(s, OUTPUT_CAP)));
        let stderr = child
            .stderr
            .take()
            .map(|s| tokio::spawn(read_tail(s, OUTPUT_CAP)));
        let status = match timeout {
            Some(limit) => match tokio::time::timeout(limit, child.wait()).await {
                Ok(status) => status,
                Err(_) => {
                    let _ = child.kill().await;
                    return Err(SandboxError::Transport(format!(
                        "`{}` did not finish within {}s and was stopped; the {} daemon may be \
                         overloaded or hung. Check it with `{} info`, then try again.",
                        self.show(args),
                        limit.as_secs(),
                        self.name,
                        self.name
                    )));
                }
            },
            None => child.wait().await,
        }
        .map_err(|e| SandboxError::io(format!("waiting for `{}`", self.show(args)), e))?;
        let stdout = match stdout {
            Some(task) => task.await.map(|t| t.into_string()).unwrap_or_default(),
            None => String::new(),
        };
        let stderr = match stderr {
            Some(task) => task.await.map(|t| t.into_string()).unwrap_or_default(),
            None => String::new(),
        };
        Ok(Output {
            code: exit_code(status),
            stdout,
            stderr,
        })
    }

    /// The error for a CLI that could not be started at all.
    pub fn spawn_failed(&self, err: std::io::Error) -> SandboxError {
        if err.kind() == std::io::ErrorKind::NotFound {
            let install = match self.flavor {
                Flavor::Docker => "Install Docker (https://docs.docker.com/get-docker/)",
                Flavor::Podman => "Install Podman (https://podman.io/docs/installation)",
            };
            SandboxError::Unavailable {
                message: format!(
                    "the `{}` command was not found (looked for {})",
                    self.name,
                    self.binary.display()
                ),
                fix: format!(
                    "{install} and make sure `{}` is on PATH, or run on Daytona with `--sandbox daytona`.",
                    self.name
                ),
            }
        } else {
            SandboxError::io(format!("starting `{}`", self.binary.display()), err)
        }
    }

    /// When a failed call's output says the daemon is unreachable (not
    /// running, or this user may not use its socket), the error that says
    /// so and how to fix it. `None` for any other failure.
    pub fn daemon_problem(&self, output: &Output) -> Option<SandboxError> {
        let text = format!("{}\n{}", output.stderr, output.stdout);
        let lower = text.to_lowercase();
        let line = |marker: &str| -> String {
            text.lines()
                .find(|l| l.to_lowercase().contains(marker))
                .map(|l| tail_chars(l.trim(), 300))
                .unwrap_or_default()
        };
        if lower.contains("permission denied")
            && (lower.contains("docker.sock")
                || lower.contains("daemon socket")
                || lower.contains("podman.sock"))
        {
            return Some(SandboxError::Unavailable {
                message: format!(
                    "this user may not use the {} daemon's socket ({})",
                    self.name,
                    line("permission denied")
                ),
                fix: match self.flavor {
                    Flavor::Docker => "Add yourself to the docker group with `sudo usermod -aG docker $USER`, \
                        then log out and back in (or run `newgrp docker`), and check with `docker version`."
                        .to_string(),
                    Flavor::Podman => "Check the podman socket's permissions (`systemctl --user status podman.socket`), \
                        or run rootless podman as your own user."
                        .to_string(),
                },
            });
        }
        let markers = [
            "cannot connect to the docker daemon",
            "failed to connect to the docker api",
            "is the docker daemon running",
            "error during connect",
            "cannot connect to podman",
            "unable to connect to podman",
        ];
        if let Some(marker) = markers.iter().find(|m| lower.contains(*m)) {
            let remote = std::env::var("DOCKER_HOST")
                .ok()
                .filter(|h| !h.is_empty())
                .map(|h| format!(" (DOCKER_HOST is {h})"))
                .unwrap_or_default();
            return Some(SandboxError::Unavailable {
                message: format!(
                    "the {} CLI cannot reach its daemon{remote}: {}",
                    self.name,
                    line(marker)
                ),
                fix: match self.flavor {
                    Flavor::Docker => "Start Docker (Docker Desktop, OrbStack, or `sudo systemctl start docker`) and \
                        check with `docker version`. For a remote daemon, check DOCKER_HOST and `docker context ls`."
                        .to_string(),
                    Flavor::Podman => "Start podman's service: `podman machine start` on macOS, \
                        `systemctl --user start podman.socket` on Linux. Then check with `podman version`."
                        .to_string(),
                },
            });
        }
        None
    }

    /// `docker <args>` for messages, with long arguments shortened.
    pub fn show(&self, args: &[String]) -> String {
        let mut s = self.name.clone();
        for arg in args {
            s.push(' ');
            if arg.chars().count() > 80 {
                s.push_str(&arg.chars().take(77).collect::<String>());
                s.push_str("...");
            } else {
                s.push_str(arg);
            }
        }
        s
    }
}

/// CLI arguments that pass a set of environment variables without putting
/// any value on a command line.
///
/// Values go in a temp file readable only by this user (`--env-file`). The
/// file format cannot carry a value with a line break, so those go as
/// `-e NAME` with the value in the CLI's own environment, which the CLI
/// copies in. The file is deleted when this value is dropped.
pub(crate) struct EnvArgs {
    pub args: Vec<String>,
    /// Variables to set on the CLI process itself.
    pub process_env: Vec<(String, String)>,
    _file: Option<tempfile::TempPath>,
}

impl EnvArgs {
    pub fn new(env: &BTreeMap<String, String>) -> Result<Self, SandboxError> {
        let mut lines = String::new();
        let mut args = Vec::new();
        let mut process_env = Vec::new();
        for (key, value) in env {
            let bad_name = key.is_empty()
                || key.starts_with('#')
                || key.contains(['=', '\0'])
                || key.chars().any(char::is_whitespace);
            if bad_name || value.contains('\0') {
                tracing::warn!(
                    "leaving out environment variable {key:?}: a process environment cannot hold \
                     that name or a value with a NUL byte"
                );
                continue;
            }
            if value.contains(['\n', '\r']) {
                args.push("-e".to_string());
                args.push(key.clone());
                process_env.push((key.clone(), value.clone()));
            } else {
                lines.push_str(key);
                lines.push('=');
                lines.push_str(value);
                lines.push('\n');
            }
        }
        let file = if lines.is_empty() {
            None
        } else {
            // tempfile creates the file with mode 0600.
            let mut file = tempfile::Builder::new()
                .prefix("benchflow-env-")
                .tempfile()
                .map_err(|e| SandboxError::io("creating a private env file", e))?;
            file.write_all(lines.as_bytes())
                .and_then(|()| file.flush())
                .map_err(|e| SandboxError::io("writing a private env file", e))?;
            let path = file.into_temp_path();
            args.insert(0, path.display().to_string());
            args.insert(0, "--env-file".to_string());
            Some(path)
        };
        Ok(Self {
            args,
            process_env,
            _file: file,
        })
    }

    /// Delete the env file now (the CLI has read it).
    pub fn close(self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    fn output(code: i32, stderr: &str) -> Output {
        Output {
            code,
            stdout: String::new(),
            stderr: stderr.to_string(),
        }
    }

    fn docker() -> Cli {
        Cli {
            name: "docker".into(),
            binary: "docker".into(),
            flavor: Flavor::Docker,
        }
    }

    /// The terminal's Ctrl-C goes to bench's process group; a sandbox CLI
    /// child must not be in it (checked with a real child process).
    #[tokio::test]
    async fn cli_children_get_a_process_group_of_their_own() {
        let pgid = |pid: u32| {
            let out = std::process::Command::new("ps")
                .args(["-o", "pgid=", "-p", &pid.to_string()])
                .output()
                .unwrap();
            String::from_utf8_lossy(&out.stdout).trim().to_string()
        };
        let sh = Cli {
            name: "sh".into(),
            binary: "sh".into(),
            flavor: Flavor::Docker,
        };
        let out = sh
            .command(["-c", "ps -o pgid= -p $$"])
            .output()
            .await
            .unwrap();
        let child = String::from_utf8_lossy(&out.stdout).trim().to_string();
        let ours = pgid(std::process::id());
        assert!(!child.is_empty() && !ours.is_empty());
        assert_ne!(child, ours, "the child shares bench's process group");
    }

    #[test]
    fn env_values_go_in_a_private_file_and_line_breaks_go_by_name() {
        let mut env = BTreeMap::new();
        env.insert("API_KEY".to_string(), "sk-123 with spaces".to_string());
        env.insert("MULTI".to_string(), "a\nb".to_string());
        env.insert("bad name".to_string(), "x".to_string());
        let e = EnvArgs::new(&env).unwrap();
        assert_eq!(e.args[0], "--env-file");
        let path = PathBuf::from(&e.args[1]);
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "API_KEY=sk-123 with spaces\n"
        );
        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
        assert_eq!(&e.args[2..], ["-e", "MULTI"]);
        assert_eq!(
            e.process_env,
            vec![("MULTI".to_string(), "a\nb".to_string())]
        );
        assert!(!e.args.iter().any(|a| a.contains("sk-123")));
        e.close();
        assert!(!path.exists(), "the env file is deleted once closed");
    }

    #[test]
    fn no_env_no_file() {
        let e = EnvArgs::new(&BTreeMap::new()).unwrap();
        assert!(e.args.is_empty());
    }

    #[test]
    fn recognizes_an_unreachable_daemon() {
        let err = docker()
            .daemon_problem(&output(
                1,
                "Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?",
            ))
            .unwrap();
        let SandboxError::Unavailable { message, fix } = err else {
            panic!("wrong variant")
        };
        assert!(message.contains("cannot reach its daemon"), "{message}");
        assert!(fix.contains("Start Docker"), "{fix}");
    }

    #[test]
    fn recognizes_a_socket_permission_problem() {
        let err = docker()
            .daemon_problem(&output(
                1,
                "permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock: \
                 Get \"http://%2Fvar%2Frun%2Fdocker.sock/v1.47/version\": dial unix /var/run/docker.sock: connect: permission denied",
            ))
            .unwrap();
        let SandboxError::Unavailable { fix, .. } = err else {
            panic!("wrong variant")
        };
        assert!(fix.contains("usermod -aG docker"), "{fix}");
    }

    #[test]
    fn an_ordinary_failure_is_not_a_daemon_problem() {
        assert!(
            docker()
                .daemon_problem(&output(1, "Error: No such container: x"))
                .is_none()
        );
    }

    #[test]
    fn a_missing_cli_says_how_to_install_it() {
        let err = docker().spawn_failed(std::io::Error::from(std::io::ErrorKind::NotFound));
        let SandboxError::Unavailable { message, fix } = err else {
            panic!("wrong variant")
        };
        assert!(
            message.contains("`docker` command was not found"),
            "{message}"
        );
        assert!(fix.contains("docs.docker.com"), "{fix}");
    }

    #[test]
    fn shows_long_arguments_shortened() {
        let shown = docker().show(&["run".to_string(), "x".repeat(200)]);
        assert!(shown.starts_with("docker run xxx"));
        assert!(shown.ends_with("..."));
        assert!(shown.len() < 100);
    }
}
