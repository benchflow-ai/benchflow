//! Running commands in a Daytona sandbox to completion.
//!
//! Each command gets its own session (sessions run their commands one at a
//! time), runs through [`wrap`], and is polled until it has an exit code.
//! Environment values are staged in a root-only file through the files API
//! and loaded by the wrapper, so they never appear on a command line inside
//! the sandbox (where an agent could read them with `ps`). A timeout kills
//! every process carrying the command's [`EXEC_ID_ENV`], then reads what the
//! command printed so far (the wrapper prints it once the command is killed;
//! see [`wrap`]).

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use super::SandboxCore;
use super::toolbox::{STAGING, Stdin, Toolbox, Wrap, wrap};
use crate::capture::{OUTPUT_CAP, TailBuffer};
use crate::shell::{exports, quote};
use crate::{EXEC_ID_ENV, Exec, ExecOutput, SandboxError, TIMEOUT_EXIT_CODE};

/// How long a timed-out command's wrapper gets to print its output after the
/// kill.
const AFTER_KILL: Duration = Duration::from_secs(5);

/// A command ready to start.
pub(crate) struct Prepared {
    pub exec_id: String,
    pub line: String,
}

/// Stage `exec`'s environment and build its session command line.
/// `with_defaults` adds the sandbox's own environment, folder and user.
pub(crate) async fn prepare(
    core: &SandboxCore,
    exec: &Exec,
    stdin: Stdin,
    with_defaults: bool,
) -> Result<Prepared, SandboxError> {
    let exec_id = uuid::Uuid::new_v4().simple().to_string();
    let mut env: BTreeMap<String, String> = if with_defaults {
        core.default_env.clone()
    } else {
        BTreeMap::new()
    };
    env.extend(exec.env.clone());
    env.remove(EXEC_ID_ENV);
    for name in &exec.unset {
        env.remove(name);
    }
    let env_file = if env.is_empty() {
        None
    } else {
        let path = format!("{STAGING}/env-{exec_id}");
        core.toolbox
            .upload_bytes(&path, exports(&env).as_bytes())
            .await?;
        Some(path)
    };
    let (cwd, user) = if with_defaults {
        (
            exec.cwd.clone().or_else(|| core.default_cwd.clone()),
            exec.user.clone().or_else(|| core.default_user.clone()),
        )
    } else {
        (exec.cwd.clone(), exec.user.clone())
    };
    let line = wrap(&Wrap {
        env_file: env_file.as_deref(),
        cwd: cwd.as_deref(),
        user: user.as_deref(),
        unset: &exec.unset,
        ..Wrap::new(&exec.command, &exec_id, stdin)
    });
    Ok(Prepared { exec_id, line })
}

/// The session line that kills every process tagged with `exec_id`. It
/// runs as root with the image's environment, so it names `/bin/sh` by its
/// path and starts it without the loader variables (and the script finds
/// `grep` in the system folders).
pub(crate) fn kill_line(exec_id: &str) -> String {
    format!(
        "unset {}; /bin/sh -c {} sh",
        crate::shell::LOADER_VARS.join(" "),
        quote(&crate::shell::kill_tagged(exec_id))
    )
}

/// Kill every process in the sandbox tagged with `exec_id`.
pub(crate) async fn kill_tagged(toolbox: &Toolbox, exec_id: &str) -> Result<(), SandboxError> {
    let (code, output) = toolbox.run_quick(&kill_line(exec_id)).await?;
    if code == 0 {
        Ok(())
    } else {
        Err(SandboxError::Transport(format!(
            "could not kill a command's processes in Daytona sandbox {} (exit code {code}: {}); \
             they may run until the sandbox is destroyed",
            toolbox.sandbox_id,
            output.trim()
        )))
    }
}

/// Cleans up after a command: always closes its session in the background,
/// and first kills its processes unless the command finished.
pub(crate) struct Cleanup {
    toolbox: Toolbox,
    session: String,
    exec_id: String,
    kill: bool,
}

impl Cleanup {
    pub(crate) fn new(toolbox: &Toolbox, session: &str, exec_id: &str) -> Self {
        Self {
            toolbox: toolbox.clone(),
            session: session.to_string(),
            exec_id: exec_id.to_string(),
            kill: true,
        }
    }

    /// The command finished; only its session needs closing.
    pub(crate) fn finished(&mut self) {
        self.kill = false;
    }
}

impl Drop for Cleanup {
    fn drop(&mut self) {
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            return;
        };
        let toolbox = self.toolbox.clone();
        let session = std::mem::take(&mut self.session);
        let exec_id = std::mem::take(&mut self.exec_id);
        let kill = self.kill;
        handle.spawn(async move {
            if kill && let Err(err) = kill_tagged(&toolbox, &exec_id).await {
                tracing::debug!("{err}");
            }
            toolbox.delete_session(&session).await;
        });
    }
}

/// Run `exec` to completion.
pub(crate) async fn run(
    core: &SandboxCore,
    exec: Exec,
    with_defaults: bool,
) -> Result<ExecOutput, SandboxError> {
    let started = Instant::now();
    let toolbox = &core.toolbox;
    let prepared = prepare(core, &exec, Stdin::Null, with_defaults).await?;
    let session = toolbox.create_session().await?;
    let mut cleanup = Cleanup::new(toolbox, &session, &prepared.exec_id);
    let command = match toolbox.start(&session, &prepared.line).await {
        Ok(command) => command,
        Err(err) => {
            cleanup.finished();
            return Err(err);
        }
    };

    let deadline = exec.timeout.map(|t| started + t);
    let mut nap = Duration::from_millis(100);
    let exit_code = loop {
        if let Some(code) = toolbox.exit_code(&session, &command).await? {
            break Some(code);
        }
        let now = Instant::now();
        let wait = match deadline {
            Some(d) if now >= d => break None,
            Some(d) => nap.min(d - now),
            None => nap,
        };
        tokio::time::sleep(wait).await;
        nap = (nap * 2).min(Duration::from_secs(1));
    };

    let timed_out = exit_code.is_none();
    if timed_out {
        if let Err(err) = kill_tagged(toolbox, &prepared.exec_id).await {
            tracing::warn!("{err}");
        }
        // The wrapper outlives the kill to print what the command wrote.
        let grace = Instant::now() + AFTER_KILL;
        while Instant::now() < grace {
            if matches!(toolbox.exit_code(&session, &command).await, Ok(Some(_))) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
    }
    cleanup.finished();
    let logs = toolbox.logs(&session, &command).await.unwrap_or_default();
    let keep = |text: Option<String>| {
        let mut tail = TailBuffer::new(OUTPUT_CAP);
        tail.push(text.unwrap_or_default().as_bytes());
        tail.into_string()
    };
    Ok(ExecOutput {
        exit_code: exit_code.unwrap_or(TIMEOUT_EXIT_CODE),
        stdout: keep(logs.stdout),
        stderr: keep(logs.stderr),
        timed_out,
        duration: started.elapsed(),
    })
}
