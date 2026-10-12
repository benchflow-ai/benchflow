//! The toolbox of one sandbox: the daemon inside it, reached through
//! Daytona's proxy at `<toolboxProxyUrl>/<sandbox id>/...`.
//!
//! What the daemon does, as observed (daemon 0.222) and relied on here:
//!
//! - Commands run in *sessions*: a long-lived bash per session, as root, in
//!   the image's `WORKDIR` (root's home when there is none). Commands of one
//!   session run one after another, so every command gets its own session.
//!   A bare `exit` would end the session's bash, so every command is wrapped
//!   in `/bin/sh -c`. The session's `PATH` is the image's.
//! - A command's stdin, stdout and stderr are FIFOs, not a terminal: no echo
//!   and no line editing. `POST .../input` writes to stdin; with
//!   `suppressInputEcho` the input is not copied into the output log.
//! - Output is logged line by line; a final line without a newline gets
//!   one. The JSON logs answer separates stdout from stderr; the streaming
//!   answer prefixes each line with `\x01\x01\x01` (stdout) or
//!   `\x02\x02\x02` (stderr).
//! - Deleting a session kills its process group. Commands run under
//!   `setsid`, so what they start in the background survives, as it does
//!   after `docker exec`.

use std::sync::Arc;
use std::time::Duration;

use super::api::{Api, ApiError, Retry};
use super::build::IMAGE_ENV_DIR;
use super::types::{CommandDto, LogsDto, SessionExec, SessionExecAnswer};
use crate::shell::{
    SYSTEM_PATH, as_user_in, command_shell_in, keep_path, quote, stash_loader_vars, unset_vars,
};
use crate::{EXEC_ID_ENV, SandboxError};

/// Root-only folder for env files and transfer archives.
pub(crate) const STAGING: &str = "/run/benchflow";

/// The start of the id of every command session BenchFlow opens.
const SESSION_PREFIX: &str = "bf-";

/// Whether BenchFlow opened the session `id`.
fn is_ours(id: &str) -> bool {
    id.starts_with(SESSION_PREFIX)
}

/// One sandbox's toolbox.
#[derive(Clone)]
pub(crate) struct Toolbox {
    pub api: Arc<Api>,
    /// `<toolboxProxyUrl>/<sandbox id>`.
    base: String,
    pub sandbox_id: String,
}

impl std::fmt::Debug for Toolbox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Toolbox")
            .field("sandbox", &self.sandbox_id)
            .finish()
    }
}

impl Toolbox {
    pub(crate) fn new(api: Arc<Api>, proxy_url: &str, sandbox_id: &str) -> Self {
        Self {
            api,
            base: format!("{}/{sandbox_id}", proxy_url.trim_end_matches('/')),
            sandbox_id: sandbox_id.to_string(),
        }
    }

    pub(crate) fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base)
    }

    /// Make a fresh session.
    pub(crate) async fn create_session(&self) -> Result<String, SandboxError> {
        let id = format!(
            "{SESSION_PREFIX}{}",
            &uuid::Uuid::new_v4().simple().to_string()[..16]
        );
        let body = serde_json::json!({ "sessionId": id });
        self.api
            .send("opening a command session", Retry::Unsent, |http| {
                Ok(http
                    .post(self.url("/process/session"))
                    .timeout(Duration::from_secs(60))
                    .json(&body))
            })
            .await
            .map_err(|e| self.error(e))?;
        Ok(id)
    }

    /// The ids of the command sessions the sandbox's daemon keeps; none
    /// when the daemon has no such listing (404).
    pub(crate) async fn list_sessions(&self) -> Result<Vec<String>, SandboxError> {
        let listed: Result<Vec<serde_json::Value>, _> = self
            .api
            .json("listing command sessions", Retry::Safe, |http| {
                Ok(http
                    .get(self.url("/process/session"))
                    .timeout(Duration::from_secs(30)))
            })
            .await;
        match listed {
            Ok(items) => Ok(items
                .iter()
                .filter_map(|s| s["sessionId"].as_str().map(str::to_string))
                .collect()),
            Err(e) if e.status() == Some(404) => Ok(Vec::new()),
            Err(e) => Err(self.error(e)),
        }
    }

    /// Delete every command session BenchFlow opened (`bf-…`), and wait
    /// until none is left: each keeps its commands' output (the agent's
    /// included) on the sandbox's disk, which a snapshot would copy. `Err`
    /// when some are still there after `within`.
    ///
    /// Sessions BenchFlow did not open are left alone and do not hold the
    /// snapshot back. The agent user cannot open sessions (the trial's guard
    /// keeps it off the daemon's port), so such a session is the daemon's
    /// or was opened by root inside the sandbox (the task's own code), and
    /// deleting it would kill what it runs. The daemon reserves the session
    /// id `entrypoint` (creating it is refused) and, as observed on
    /// 2026-10-11, never lists it; under BenchFlow the image's own
    /// ENTRYPOINT and CMD do not run, since the sandbox's first process is
    /// the daemon.
    pub(crate) async fn clear_sessions(&self, within: Duration) -> Result<(), SandboxError> {
        let deadline = std::time::Instant::now() + within;
        loop {
            let listed = self.list_sessions().await?;
            tracing::debug!(
                "command sessions in Daytona sandbox {}: {listed:?}",
                self.sandbox_id
            );
            let (left, foreign): (Vec<String>, Vec<String>) =
                listed.into_iter().partition(|s| is_ours(s));
            if left.is_empty() {
                if !foreign.is_empty() {
                    tracing::debug!(
                        "left alone in Daytona sandbox {} (not BenchFlow's): {foreign:?}",
                        self.sandbox_id
                    );
                }
                return Ok(());
            }
            if std::time::Instant::now() >= deadline {
                return Err(SandboxError::Transport(format!(
                    "{} command sessions of Daytona sandbox {} were still open after {} s, so no snapshot was taken (their output is on the sandbox's disk)",
                    left.len(),
                    self.sandbox_id,
                    within.as_secs()
                )));
            }
            for s in &left {
                self.delete_session(s).await;
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
    }

    /// Delete a session, killing what is left in its process group.
    pub(crate) async fn delete_session(&self, session: &str) {
        let result = self
            .api
            .send("closing a command session", Retry::Safe, |http| {
                Ok(http
                    .delete(self.url(&format!("/process/session/{session}")))
                    .timeout(Duration::from_secs(60)))
            })
            .await;
        if let Err(err) = result
            && err.status() != Some(404)
        {
            tracing::debug!(
                "closing session {session} in {}: {}",
                self.sandbox_id,
                err.message()
            );
        }
    }

    /// Start `command` in `session` without waiting; its command id.
    pub(crate) async fn start(&self, session: &str, command: &str) -> Result<String, SandboxError> {
        let body = SessionExec {
            command,
            run_async: true,
            suppress_input_echo: true,
        };
        let answer: SessionExecAnswer = self
            .api
            .json("starting a command", Retry::Unsent, |http| {
                Ok(http
                    .post(self.url(&format!("/process/session/{session}/exec")))
                    .timeout(Duration::from_secs(60))
                    .json(&body))
            })
            .await
            .map_err(|e| self.error(e))?;
        answer.cmd_id.filter(|c| !c.is_empty()).ok_or_else(|| {
            SandboxError::Transport(
                "Daytona started a command but returned no id for it; run again".to_string(),
            )
        })
    }

    /// Run `command` in a fresh session and wait for it (for short, quiet
    /// housekeeping commands only). Returns the exit code and output.
    pub(crate) async fn run_quick(&self, command: &str) -> Result<(i32, String), SandboxError> {
        let session = self.create_session().await?;
        let body = SessionExec {
            command,
            run_async: false,
            suppress_input_echo: true,
        };
        let answer: Result<SessionExecAnswer, ApiError> = self
            .api
            .json("running a housekeeping command", Retry::Unsent, |http| {
                Ok(http
                    .post(self.url(&format!("/process/session/{session}/exec")))
                    .timeout(Duration::from_secs(120))
                    .json(&body))
            })
            .await;
        self.delete_session(&session).await;
        let answer = answer.map_err(|e| self.error(e))?;
        Ok((
            answer.exit_code.unwrap_or(-1),
            answer.output.unwrap_or_default(),
        ))
    }

    /// The command's exit code, once it has one.
    pub(crate) async fn exit_code(
        &self,
        session: &str,
        command: &str,
    ) -> Result<Option<i32>, SandboxError> {
        let dto: CommandDto = self
            .api
            .json("checking a command", Retry::Safe, |http| {
                Ok(http
                    .get(self.url(&format!("/process/session/{session}/command/{command}")))
                    .timeout(Duration::from_secs(30)))
            })
            .await
            .map_err(|e| self.error(e))?;
        Ok(dto.exit_code)
    }

    /// The command's output so far, stdout and stderr apart.
    pub(crate) async fn logs(&self, session: &str, command: &str) -> Result<LogsDto, SandboxError> {
        self.api
            .json("reading a command's output", Retry::Safe, |http| {
                Ok(http
                    .get(self.url(&format!(
                        "/process/session/{session}/command/{command}/logs"
                    )))
                    .header(reqwest::header::ACCEPT, "application/json")
                    .timeout(Duration::from_secs(300)))
            })
            .await
            .map_err(|e| self.error(e))
    }

    /// Write `data` to the command's stdin. Not retried after it may have
    /// arrived: stdin must not see the same bytes twice.
    pub(crate) async fn input(
        &self,
        session: &str,
        command: &str,
        data: &str,
    ) -> Result<(), SandboxError> {
        let body = serde_json::json!({ "data": data });
        self.api
            .send("writing to a command's stdin", Retry::Unsent, |http| {
                Ok(http
                    .post(self.url(&format!(
                        "/process/session/{session}/command/{command}/input"
                    )))
                    .timeout(Duration::from_secs(300))
                    .json(&body))
            })
            .await
            .map_err(|e| self.error(e))?;
        Ok(())
    }

    /// Upload `file` (`len` bytes) to `path`, replacing what is there.
    pub(crate) async fn upload(
        &self,
        path: &str,
        file: &std::fs::File,
        len: u64,
    ) -> Result<(), SandboxError> {
        self.api
            .send("uploading a file", Retry::Safe, |http| {
                let mut handle = file.try_clone().map_err(|e| ApiError::Decode {
                    message: format!("cannot reopen the upload: {e}"),
                    what: "uploading a file".to_string(),
                })?;
                std::io::Seek::seek(&mut handle, std::io::SeekFrom::Start(0)).map_err(|e| {
                    ApiError::Decode {
                        message: format!("cannot rewind the upload: {e}"),
                        what: "uploading a file".to_string(),
                    }
                })?;
                let body = reqwest::Body::from(tokio::fs::File::from_std(handle));
                let part = reqwest::multipart::Part::stream_with_length(body, len)
                    .file_name("upload")
                    .mime_str("application/octet-stream")
                    .map_err(|e| ApiError::Decode {
                        message: e.to_string(),
                        what: "uploading a file".to_string(),
                    })?;
                Ok(http
                    .post(self.url("/files/upload"))
                    .query(&[("path", path)])
                    .timeout(Duration::from_secs(3600))
                    .multipart(reqwest::multipart::Form::new().part("file", part)))
            })
            .await
            .map_err(|e| self.error(e))?;
        Ok(())
    }

    /// Upload small bytes to `path`.
    pub(crate) async fn upload_bytes(&self, path: &str, data: &[u8]) -> Result<(), SandboxError> {
        let data = data.to_vec();
        self.api
            .send("uploading a file", Retry::Safe, |http| {
                let part = reqwest::multipart::Part::bytes(data.clone())
                    .file_name("upload")
                    .mime_str("application/octet-stream")
                    .map_err(|e| ApiError::Decode {
                        message: e.to_string(),
                        what: "uploading a file".to_string(),
                    })?;
                Ok(http
                    .post(self.url("/files/upload"))
                    .query(&[("path", path)])
                    .timeout(Duration::from_secs(120))
                    .multipart(reqwest::multipart::Form::new().part("file", part)))
            })
            .await
            .map_err(|e| self.error(e))?;
        Ok(())
    }

    /// Stream the file at `path`.
    pub(crate) async fn download(&self, path: &str) -> Result<reqwest::Response, SandboxError> {
        self.api
            .send("downloading a file", Retry::Safe, |http| {
                Ok(http
                    .get(self.url("/files/download"))
                    .query(&[("path", path)])
                    .timeout(Duration::from_secs(3600)))
            })
            .await
            .map_err(|e| self.error(e))
    }

    /// Delete a file (best effort).
    pub(crate) async fn remove(&self, path: &str) {
        let result = self
            .api
            .send("deleting a file", Retry::Safe, |http| {
                Ok(http
                    .delete(self.url("/files"))
                    .query(&[("path", path)])
                    .timeout(Duration::from_secs(60)))
            })
            .await;
        if let Err(err) = result
            && err.status() != Some(404)
        {
            tracing::debug!("deleting {path} in {}: {}", self.sandbox_id, err.message());
        }
    }

    /// Create a folder (and its parents) with `mode`, e.g. `700`.
    pub(crate) async fn mkdir(&self, path: &str, mode: &str) -> Result<(), SandboxError> {
        self.api
            .send("creating a folder", Retry::Safe, |http| {
                Ok(http
                    .post(self.url("/files/folder"))
                    .query(&[("path", path), ("mode", mode)])
                    .timeout(Duration::from_secs(60)))
            })
            .await
            .map_err(|e| self.error(e))?;
        Ok(())
    }

    /// The error for a failed toolbox call, naming the sandbox when the
    /// toolbox says it is not running. A toolbox answer comes through
    /// Daytona's proxy from the daemon inside the sandbox, which root code
    /// there can replace, so none of it is the backend's own word: a 429,
    /// 401 or 403 from it is a transport failure, never the confirmed loss
    /// the control plane's own answer would be (convergence review E9-D-B 2).
    pub(crate) fn error(&self, err: ApiError) -> SandboxError {
        match self.error_as_answered(err) {
            SandboxError::Throttled(m) => SandboxError::Transport(m),
            SandboxError::Unavailable { message, .. } => SandboxError::Transport(format!(
                "{message} (the sandbox's toolbox answered; it is inside the sandbox, so this is not \
                 taken as Daytona's own word)"
            )),
            other => other,
        }
    }

    fn error_as_answered(&self, err: ApiError) -> SandboxError {
        let message = err.message();
        if message.contains("SANDBOX_NOT_RUNNING")
            || message.contains("Is the Sandbox started")
            || message.to_lowercase().contains("no ip address found")
        {
            return SandboxError::Transport(format!(
                "Daytona sandbox {} is not running ({message}). It was stopped or deleted (Daytona \
                 stops a sandbox after its idle interval), or its container crashed.",
                self.sandbox_id
            ));
        }
        err.into()
    }
}

/// How a wrapped command gets its stdin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Stdin {
    /// `/dev/null`, as `docker exec` without input gives.
    Null,
    /// The session's input FIFO, fed by `POST .../input`.
    Pipe,
}

/// The session command line for one command: export the exec id, load the
/// staged env file (then delete it), change folder, then run the command
/// under `setsid`, as `user` when that is not root.
///
/// Daytona's session shell runs every line as root with the container's
/// `PATH` and loader variables. On an image BenchFlow built these are the
/// system folders and empty ([`IMAGE_ENV_DIR`]); on another image the
/// `PATH` may start with a folder the agent user can write and the loader
/// variables may name files there. So the line names `/bin/sh` by its path,
/// the wrapper finds its own programs (`setsid`, `mkdir`, `cat`, `rm`) in
/// the system folders only and runs them without the loader variables
/// ([`stash_loader_vars`]), and the command gets the image's `PATH` and
/// loader variables back as it starts (the env file's win; see
/// [`restore_image_env`] and [`keep_path`]).
///
/// Daytona reports a command's exit code only once its stdout closes, so a
/// command that leaves a background process holding stdout (`server &`)
/// would never finish. A command run to completion ([`Stdin::Null`])
/// therefore writes its output to root-only files in [`STAGING`]; the wrapper
/// waits for the command alone, prints the files, and exits with its code.
/// Only the command carries the exec id, so a timeout kills it and leaves the
/// wrapper to print what it wrote. A spawned command ([`Stdin::Pipe`])
/// streams its output and is not wrapped this way.
pub(crate) fn wrap(w: &Wrap) -> String {
    wrap_in(STAGING, SYSTEM_PATH, IMAGE_ENV_DIR, w)
}

/// Shell that gives the command the image's own `PATH` and loader variables
/// for any it does not already have, read from [`IMAGE_ENV_DIR`] (one file
/// per name; present only on an image BenchFlow built, and only for a
/// variable the image set). "Does not have" means empty or unset: the
/// container's `PATH` is the system folders and its loader variables empty,
/// so empty means the env file set nothing. Runs with builtins only (`read`,
/// `export`), no `PATH` needed.
fn restore_image_env(dir: &str) -> String {
    let names: Vec<&str> = std::iter::once("PATH")
        .chain(crate::shell::LOADER_VARS.iter().copied())
        .collect();
    // Restore only variables the env file left unset (tested by presence,
    // `${v+x}`): a value the task set empty on purpose stays empty. The
    // value is read with the `read` builtin (no program, since `PATH` may
    // not be set yet); a recorded value with a newline keeps its first line
    // only, which `PATH` and the loader variables never have.
    let tmpl = r#"for bf_v in @NAMES@; do eval "bf_set=\${$bf_v+x}"; if [ -z "$bf_set" ] && [ -r @DIR@/$bf_v ]; then IFS= read -r bf_lv < @DIR@/$bf_v; export "$bf_v=$bf_lv"; fi; done"#;
    tmpl.replace("@NAMES@", &names.join(" "))
        .replace("@DIR@", &quote(dir))
}

/// One command for [`wrap`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct Wrap<'a> {
    pub command: &'a str,
    /// Tags the command's processes ([`EXEC_ID_ENV`]).
    pub exec_id: &'a str,
    /// The staged env file to load, then delete.
    pub env_file: Option<&'a str>,
    pub cwd: Option<&'a str>,
    /// Who runs the command; root when `None`.
    pub user: Option<&'a str>,
    pub stdin: Stdin,
    /// Variables the command runs without ([`crate::Exec::unset`]).
    pub unset: &'a [String],
}

impl<'a> Wrap<'a> {
    pub(crate) fn new(command: &'a str, exec_id: &'a str, stdin: Stdin) -> Self {
        Self {
            command,
            exec_id,
            env_file: None,
            cwd: None,
            user: None,
            stdin,
            unset: &[],
        }
    }
}

/// [`wrap`], with the output files in `staging`, the wrapper's own
/// programs looked up in `sys`, and the image's env read from
/// `image_env_dir`.
fn wrap_in(staging: &str, sys: &str, image_env_dir: &str, w: &Wrap) -> String {
    let Wrap {
        command,
        exec_id,
        env_file,
        cwd,
        user,
        stdin,
        unset,
    } = *w;
    let inner = match user.filter(|u| !is_root(u)) {
        Some(user) => as_user_in(sys, user, command, unset),
        None => command_shell_in(sys, command, unset),
    };
    let stash = stash_loader_vars();
    let mut script = format!("export {EXEC_ID_ENV}={}\n", quote(exec_id));
    // The session's `PATH`, kept for the fallback below; then unset, so the
    // env file and the image record can be told to have set one or not.
    script.push_str(&format!("bf_sp=${{PATH-{}}}; unset PATH\n", quote(sys)));
    // Clear the container's loader variables (the image layer sets the
    // file-naming ones empty) before the env file runs, so afterwards a
    // loader variable is set only if the env file set it. Non-empty values
    // (none, on a BenchFlow-built image) are kept in `BF_KEEP_*`.
    script.push_str(&stash);
    script.push('\n');
    if let Some(file) = env_file {
        let f = quote(file);
        // Loading the file runs only `export`, a builtin.
        script.push_str(&format!(
            "if ! . {f}; then PATH={s}; rm -f {f}; echo 'benchflow: could not load the command environment' >&2; exit 126; fi\n",
            s = quote(sys)
        ));
    }
    // A BenchFlow-built image keeps its real `PATH` and loader variables in
    // a file each (see [`IMAGE_ENV_DIR`]). For any the env file did not set
    // (tested by presence, so a value the task set empty on purpose is
    // kept), take the image's, so the command runs as it would have on the
    // image.
    script.push_str(&restore_image_env(image_env_dir));
    script.push('\n');
    // A command that still has no `PATH` (an image BenchFlow did not build,
    // whose `PATH` the session carries) gets the session's.
    script.push_str("if [ -z \"${PATH-}\" ]; then PATH=$bf_sp; fi; export PATH\n");
    // Stash again: the env file and the image record just set the command's
    // loader variables into the live environment, and the wrapper's own
    // programs must not get them. `command_body` and the user switch put
    // them back for the command.
    script.push_str(&stash);
    script.push('\n');
    script.push_str(&keep_path(sys));
    script.push('\n');
    if let Some(file) = env_file {
        script.push_str(&format!("rm -f {}\n", quote(file)));
    }
    if let Some(dir) = cwd.filter(|d| !d.is_empty()) {
        script.push_str(&format!("cd {} || exit 126\n", quote(dir)));
    }
    script.push_str(
        "bf_setsid=$(command -v setsid 2>/dev/null)\ncase \"$bf_setsid\" in /*) ;; *) bf_setsid=;; esac\n",
    );
    let i = quote(&inner);
    // Runs in a subshell, so the wrapper keeps the system `PATH`. `$bf_setsid`
    // is a `setsid` path or empty, named once so the command script is not
    // copied (unquoted on purpose: a path has no spaces, and empty must
    // vanish). An absolute `/bin/sh` runs the script.
    let run = format!("PATH=$bf_path; exec $bf_setsid /bin/sh -c {i} sh");
    match stdin {
        // The wrapper waits for the command instead of becoming it, so a
        // kill leaves the wrapper to exit with the command's code: Daytona's
        // session shell, whose child the wrapper is, would otherwise print
        // "Killed" and this whole script into the command's stderr. The
        // wrapper's own notice of the kill goes to /dev/null; the command's
        // stderr goes through fd 3. The redirections use `exec`, never a
        // redirected `{ ...; }`: dash keeps a copy of the redirected fd for
        // the group, and the command's children inherit it.
        Stdin::Pipe => script.push_str(&format!(
            "exec 3>&2 2>/dev/null\n({run}) 2>&3 3>&-\nexit $?\n"
        )),
        Stdin::Null => {
            let out = quote(&format!("{staging}/out-{exec_id}"));
            let err = quote(&format!("{staging}/err-{exec_id}"));
            script.push_str(&format!(
                "bf_o={out}; bf_e={err}\n\
                 (umask 077; mkdir -p {staging} && : >\"$bf_o\" && : >\"$bf_e\") || {{ echo 'benchflow: could not create the output files' >&2; exit 126; }}\n\
                 exec 4>&2 2>/dev/null\n\
                 ({run}) >\"$bf_o\" 2>\"$bf_e\" 4>&-\n\
                 bf_rc=$?\n\
                 exec 2>&4 4>&-\n\
                 cat \"$bf_o\"; cat \"$bf_e\" >&2; rm -f \"$bf_o\" \"$bf_e\"\n\
                 exit $bf_rc\n",
                staging = quote(staging),
            ));
        }
    }
    // The session's shell runs this line, as root, with the image's
    // environment: the variables the command runs without go first, then
    // the loader variables are stashed, so no root program the wrapper
    // starts loads or reads what they name.
    let mut line = format!(
        "{}{stash}/bin/sh -c {} sh",
        unset_vars(unset),
        quote(&script)
    );
    if stdin == Stdin::Null {
        line.push_str(" </dev/null");
    }
    line
}

pub(crate) fn is_root(user: &str) -> bool {
    user == "root" || user == "0"
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Convergence review E9-D-B 2: root code inside a sandbox can replace
    /// the toolbox daemon and answer 429, 401 or 403, which read as
    /// Daytona's own word (a confirmed loss, rerun on every resume). A
    /// toolbox answer is only ever a transport failure.
    #[test]
    fn a_toolbox_answer_is_never_the_backends_own_word() {
        let api = Arc::new(Api::new("https://app.daytona.io/api", Some("k".into()), None).unwrap());
        let tb = Toolbox::new(api, "https://proxy.app.daytona.io/toolbox", "sb1");
        for (status, message) in [
            (429, "slow down"),
            (401, "Unauthorized"),
            (403, "forbidden"),
        ] {
            let err = tb.error(ApiError::Status {
                status,
                message: message.into(),
                what: "running a command".into(),
                attempts: 3,
            });
            assert!(
                matches!(&err, SandboxError::Transport(_)),
                "{status}: {err:?}"
            );
        }
    }

    /// A folder that is not there: a sandbox whose image BenchFlow did not
    /// build, with no env recorded.
    const NO_IMAGE_ENV: &str = "/nonexistent/benchflow/image-env";

    /// The line for `command` (exec id `x`), with the output files in
    /// `staging` and the real system folders.
    fn line_in(
        staging: &str,
        command: &str,
        env_file: Option<&str>,
        cwd: Option<&str>,
        user: Option<&str>,
        stdin: Stdin,
    ) -> String {
        wrap_in(
            staging,
            SYSTEM_PATH,
            NO_IMAGE_ENV,
            &Wrap {
                env_file,
                cwd,
                user,
                ..Wrap::new(command, "x", stdin)
            },
        )
    }

    fn run(line: &str) -> (i32, String, String) {
        let out = std::process::Command::new("sh")
            .arg("-c")
            .arg(line)
            .output()
            .unwrap();
        (
            out.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        )
    }

    #[test]
    fn the_wrapper_exports_the_id_loads_env_and_changes_folder() {
        let dir = tempfile::tempdir().unwrap();
        let env_file = dir.path().join("env");
        std::fs::write(&env_file, "export GREETING='hello world'\n").unwrap();
        let line = wrap_in(
            dir.path().to_str().unwrap(),
            SYSTEM_PATH,
            NO_IMAGE_ENV,
            &Wrap {
                env_file: env_file.to_str(),
                cwd: dir.path().to_str(),
                ..Wrap::new(
                    "echo \"$GREETING from $(pwd)\"; echo id=$BENCHFLOW_EXEC_ID; exit 7",
                    "abc123",
                    Stdin::Null,
                )
            },
        );
        let (code, out, _) = run(&line);
        assert_eq!(code, 7);
        let here = dir.path().canonicalize().unwrap();
        assert!(out.contains("hello world from "), "{out}");
        assert!(
            out.contains(&here.file_name().unwrap().to_string_lossy().to_string()),
            "{out}"
        );
        assert!(out.contains("id=abc123"), "{out}");
        assert!(!env_file.exists(), "the env file is deleted once loaded");
    }

    #[test]
    fn a_missing_folder_fails_with_126() {
        let line = wrap(&Wrap {
            cwd: Some("/no/such/folder"),
            ..Wrap::new("echo never", "x", Stdin::Null)
        });
        let (code, out, err) = run(&line);
        assert_eq!(code, 126);
        assert!(out.is_empty());
        assert!(!err.is_empty());
    }

    #[test]
    fn stdin_is_closed_unless_piped() {
        let staging = tempfile::tempdir().unwrap();
        let line = line_in(
            staging.path().to_str().unwrap(),
            "cat; echo done",
            None,
            None,
            None,
            Stdin::Null,
        );
        let (code, out, _) = run(&line);
        assert_eq!((code, out.as_str()), (0, "done\n"));
        assert!(!wrap(&Wrap::new("cat", "x", Stdin::Pipe)).ends_with("</dev/null"));
    }

    /// Daytona reports an exit code only once stdout closes, so a command
    /// that leaves `sleep &` behind used to run until its timeout (the
    /// process-flood probe took 2 minutes on Daytona). The wrapper must
    /// return when the command exits, with its output and code.
    #[test]
    fn a_background_child_does_not_hold_the_command_open() {
        let staging = tempfile::tempdir().unwrap();
        let line = line_in(
            staging.path().to_str().unwrap(),
            "(sleep 20 &); echo out; echo err >&2; exit 3",
            None,
            None,
            None,
            Stdin::Null,
        );
        let started = std::time::Instant::now();
        let (code, out, err) = run(&line);
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "waited for the child"
        );
        assert_eq!((code, out.as_str(), err.as_str()), (3, "out\n", "err\n"));
        let left: Vec<_> = std::fs::read_dir(staging.path()).unwrap().collect();
        assert!(left.is_empty(), "the output files are removed");
        // A spawned command streams its output instead.
        let piped = line_in("/s", "true", None, None, None, Stdin::Pipe);
        assert!(!piped.contains("/s/out-x"), "{piped}");
    }

    /// A killed command leaves no "Killed ..." notice in its output. Daytona's
    /// session shell printed one, with the whole wrapper script (20 KB), into
    /// every Codex trial's agent log when BenchFlow stopped the agent.
    #[test]
    fn a_killed_command_leaves_no_notice_in_its_output() {
        let staging = tempfile::tempdir().unwrap();
        for stdin in [Stdin::Pipe, Stdin::Null] {
            let line = line_in(
                staging.path().to_str().unwrap(),
                "echo before >&2; kill -9 $$",
                None,
                None,
                None,
                stdin,
            );
            // Like Daytona's cmd.sh: a bash that runs the line and reports
            // on its children.
            let out = std::process::Command::new("bash")
                .arg("-c")
                .arg(format!("{line}\necho rc=$? >&2"))
                .output()
                .unwrap();
            let err = String::from_utf8_lossy(&out.stderr);
            assert!(!err.contains("Killed"), "{stdin:?}: {err}");
            assert!(err.contains("before"), "{stdin:?}: {err}");
            assert!(err.contains("rc=137"), "{stdin:?}: {err}");
        }
    }

    /// Daytona's session shell runs the line as root with the image's
    /// `PATH`. With a folder the agent can write first on it, nothing the
    /// wrapper runs (the shells, `setsid`, `mkdir`, `cat`, `rm`, and for a
    /// user `id`, `awk`, `grep`, `setpriv` or `su`) may come from there,
    /// and the command still gets the `PATH` it would have had.
    #[test]
    fn the_wrapper_finds_no_program_through_the_sessions_path() {
        let m = crate::shell::markers::Markers::new();
        let staging = tempfile::tempdir().unwrap();
        let st = staging.path().to_str().unwrap();
        let print = r#"printf %s "$PATH""#;
        // A root command with the session's PATH, run to completion and
        // spawned.
        for stdin in [Stdin::Null, Stdin::Pipe] {
            let line = line_in(st, print, None, Some("/"), None, stdin);
            let (code, out, err) = m.sh(&line);
            assert_eq!(code, 0, "{stdin:?}: {err}");
            assert_eq!(out, m.path(), "{stdin:?}");
        }
        // A root command whose env file sets `PATH`: the file is loaded
        // after the wrapper started, and its `PATH` is the command's alone.
        let env_file = staging.path().join("env-x");
        let given = format!("{}:/usr/bin:/bin", m.bin());
        std::fs::write(&env_file, format!("export PATH={}; ", quote(&given))).unwrap();
        let line = line_in(st, print, env_file.to_str(), None, None, Stdin::Null);
        let (code, out, err) = m.sh(&line);
        assert_eq!((code, out.as_str()), (0, given.as_str()), "{err}");
        assert!(!env_file.exists());
        // A user's command. No one has this uid, so the switch fails on this
        // host (or, run as root, runs `true` as that uid); what matters is
        // which programs root looked up on the way.
        let line = line_in(st, "true", None, None, Some("4242424"), Stdin::Null);
        let _ = m.sh(&line);
        assert!(m.ran().is_empty(), "{:?}", m.ran());
    }

    /// Daytona's session shell carries the image's environment, and the
    /// env file may add to it. The wrapper's own root programs (`setsid`,
    /// `mkdir`, `cat`, `rm`, and for a user `id`, `awk`, `grep`, `setpriv`)
    /// start without the loader variables either sets; the command's shell
    /// gets them back (and reads `BASH_ENV`, as a plain `bash -c` would),
    /// unless the command runs without them (`unset`).
    #[test]
    fn the_wrappers_root_programs_start_without_loader_variables() {
        use crate::shell::recorders::{PRINT_LOADER_VARS, Recorders, loader_env};
        let m = crate::shell::markers::Markers::new();
        let staging = tempfile::tempdir().unwrap();
        let st = staging.path().to_str().unwrap();
        let all: Vec<String> = crate::shell::LOADER_VARS
            .iter()
            .map(|v| v.to_string())
            .collect();
        for user in [None, Some("nobody")] {
            for stdin in [Stdin::Null, Stdin::Pipe] {
                for (unset, with_file) in [
                    (&[][..], true),
                    (&[][..], false),
                    (&all[..], true),
                    (&all[..], false),
                ] {
                    let case = format!(
                        "{user:?} {stdin:?} unset={} env file={with_file}",
                        unset.len()
                    );
                    let r = Recorders::new();
                    let env = loader_env(&r.read_file());
                    let env: Vec<(&str, &str)> =
                        env.iter().map(|(k, v)| (*k, v.as_str())).collect();
                    let env_file = staging.path().join("env-x");
                    std::fs::write(&env_file, "export LD_LIBRARY_PATH=/nonexistent-env-file; ")
                        .unwrap();
                    let line = wrap_in(
                        st,
                        &r.sys(),
                        NO_IMAGE_ENV,
                        &Wrap {
                            env_file: env_file.to_str().filter(|_| with_file),
                            user,
                            unset,
                            ..Wrap::new(PRINT_LOADER_VARS, "x", stdin)
                        },
                    );
                    let (code, out, err) = m.sh_env(&line, &env);
                    assert_eq!(code, 0, "{case}: {err}");
                    let ran = r.ran();
                    let (shell, plumbing): (Vec<_>, Vec<_>) =
                        ran.iter().partition(|l| l.starts_with("bash"));
                    // No loader or start-up variable reaches any program on
                    // the way; the command's shell puts them back itself.
                    assert!(
                        plumbing.iter().all(|l| !l.contains(' ')),
                        "{case}: a program on the way saw a loader variable: {ran:?}"
                    );
                    let mut steps = vec!["setsid"];
                    if stdin == Stdin::Null {
                        steps.extend(["mkdir", "cat", "rm"]);
                    }
                    if user.is_some() {
                        steps.extend(["id", "awk", "grep", "setpriv"]);
                    }
                    for step in steps {
                        assert!(
                            plumbing.iter().any(|l| l.split(' ').next() == Some(step)),
                            "{case}: no {step}: {ran:?}"
                        );
                    }
                    assert_eq!(shell.len(), 1, "{case}: {ran:?}");
                    let read = r.read_file();
                    if unset.is_empty() {
                        let lib = if with_file {
                            "/nonexistent-env-file"
                        } else {
                            "/nonexistent-bf-lib"
                        };
                        // The command sees the loader variables and the
                        // start-up files; its shell, which starts with none,
                        // sources BASH_ENV once after the loader variables
                        // are back.
                        assert_eq!(out, format!("[ ][{lib}][{read}][{read}]"), "{case}");
                        assert_eq!(shell[0], "bash", "{case}");
                        assert_eq!(r.reads(), 1, "{case}");
                    } else {
                        // The command drops them all, start-up files too.
                        assert_eq!(out, "[unset][unset][unset][unset]", "{case}");
                        assert_eq!(shell[0], "bash", "{case}");
                        assert_eq!(r.reads(), 0, "{case}");
                    }
                }
            }
        }
        assert!(m.ran().is_empty(), "{:?}", m.ran());
    }

    /// On Linux an `LD_PRELOAD` naming a missing file makes the loader
    /// complain once in every process it starts, so the complaints count
    /// the processes that loaded with it, `/bin/sh` included. The outer
    /// shell (standing in for Daytona's session shell, which BenchFlow
    /// neutralizes on its own images) loads it; after that only a program
    /// the command `exec`s does, and nothing the wrapper itself runs, none
    /// with `unset`, and none for the kill.
    #[cfg(target_os = "linux")]
    #[test]
    fn on_linux_only_the_commands_shell_loads_the_images_ld_preload() {
        let staging = tempfile::tempdir().unwrap();
        let st = staging.path().to_str().unwrap();
        let preload = "/nonexistent-bf-preload.so";
        let all: Vec<String> = crate::shell::LOADER_VARS
            .iter()
            .map(|v| v.to_string())
            .collect();
        let count = |line: &str| {
            let out = std::process::Command::new("/bin/sh")
                .arg("-c")
                .arg(line)
                .env("LD_PRELOAD", preload)
                .stdin(std::process::Stdio::null())
                .output()
                .unwrap();
            String::from_utf8_lossy(&out.stderr)
                .matches(preload)
                .count()
        };
        for stdin in [Stdin::Null, Stdin::Pipe] {
            // `exec true` so the command's shell replaces itself with a
            // program that loads the image's LD_PRELOAD (the shell itself
            // starts clean and restores it first); with `unset` that program
            // starts without it.
            let plain = wrap_in(
                st,
                SYSTEM_PATH,
                NO_IMAGE_ENV,
                &Wrap::new("exec true", "x", stdin),
            );
            assert_eq!(count(&plain), 2, "{stdin:?}");
            let clean = wrap_in(
                st,
                SYSTEM_PATH,
                NO_IMAGE_ENV,
                &Wrap {
                    unset: &all,
                    ..Wrap::new("exec true", "x", stdin)
                },
            );
            assert_eq!(count(&clean), 1, "{stdin:?}");
        }
        assert_eq!(count(&super::super::exec::kill_line("no-such-exec")), 1);
        // The control: the line as 35524763 wrote it, `sh -c` with no
        // stash, loads it in every process on the way.
        assert!(count("sh -c 'setsid sh -c \"bash -c true\"'") >= 4);
    }

    /// On an image BenchFlow built, the session (Daytona's daemon) has the
    /// system folders as its `PATH`, and the image's own is recorded in a
    /// file. A command that sets no `PATH` gets the recorded one, one whose
    /// env file sets it keeps its own, and the wrapper itself still finds
    /// nothing through either.
    #[test]
    fn a_command_gets_the_images_recorded_path_back() {
        let m = crate::shell::markers::Markers::new();
        let staging = tempfile::tempdir().unwrap();
        let st = staging.path().to_str().unwrap();
        // The image-env folder: PATH and BASH_ENV recorded, one file each.
        let dir = staging.path().join("image-env");
        std::fs::create_dir(&dir).unwrap();
        std::fs::write(dir.join("PATH"), m.path()).unwrap();
        std::fs::write(dir.join("BASH_ENV"), "/app/.venv/env.sh").unwrap();
        let d = dir.to_str().unwrap();
        let print = r#"printf '%s|%s' "$PATH" "${BASH_ENV-}""#;
        let run = |line: &str| {
            let out = std::process::Command::new("/bin/sh")
                .arg("-c")
                .arg(line)
                .env("PATH", SYSTEM_PATH)
                .env("BASH_ENV", "") // the container neutralized it
                .stdin(std::process::Stdio::null())
                .output()
                .unwrap();
            (
                out.status.code().unwrap_or(-1),
                String::from_utf8_lossy(&out.stdout).into_owned(),
            )
        };
        // A command that sets nothing: it gets the image's PATH and BASH_ENV.
        for user in [None, Some("4242424")] {
            for stdin in [Stdin::Null, Stdin::Pipe] {
                let line = wrap_in(
                    st,
                    SYSTEM_PATH,
                    d,
                    &Wrap {
                        user,
                        ..Wrap::new(print, "x", stdin)
                    },
                );
                let (code, out) = run(&line);
                if user.is_none() {
                    assert_eq!(
                        (code, out.as_str()),
                        (0, format!("{}|/app/.venv/env.sh", m.path()).as_str()),
                        "{stdin:?}"
                    );
                }
            }
        }
        // An env file that sets PATH keeps its own; BASH_ENV it left unset
        // still comes from the image.
        let env_file = staging.path().join("env-x");
        std::fs::write(&env_file, "export PATH=/usr/bin:/bin; ").unwrap();
        let line = wrap_in(
            st,
            SYSTEM_PATH,
            d,
            &Wrap {
                env_file: env_file.to_str(),
                ..Wrap::new(print, "x", Stdin::Null)
            },
        );
        assert_eq!(
            run(&line),
            (0, "/usr/bin:/bin|/app/.venv/env.sh".to_string())
        );
        // No folder (an image BenchFlow did not build): the session's PATH,
        // and the session's BASH_ENV (here the neutralized empty).
        let line = wrap_in(
            st,
            SYSTEM_PATH,
            NO_IMAGE_ENV,
            &Wrap::new(print, "x", Stdin::Null),
        );
        assert_eq!(run(&line), (0, format!("{SYSTEM_PATH}|")));
        assert!(m.ran().is_empty(), "{:?}", m.ran());
    }

    /// The whole session line names a command once, however deep it nests
    /// (root, a user, each stdin mode): a long command no longer overflows
    /// the argument (D2 found it pasted a dozen times).
    #[test]
    fn the_session_line_names_the_command_once() {
        let marker = "BF_UNIQUE_COMMAND_MARKER";
        for user in [None, Some("agent")] {
            for stdin in [Stdin::Null, Stdin::Pipe] {
                let line = wrap(&Wrap {
                    user,
                    ..Wrap::new(marker, "x", stdin)
                });
                assert_eq!(line.matches(marker).count(), 1, "{user:?} {stdin:?}");
            }
        }
    }

    /// Run `line` as Daytona's session shell does on an image BenchFlow
    /// built: `PATH` the system folders, the file-naming loader variables
    /// and start-up files empty (the layer's `ENV`), nothing else.
    fn layer_session(line: &str) -> (i32, String, String) {
        let mut cmd = std::process::Command::new("/bin/sh");
        cmd.arg("-c")
            .arg(line)
            .env_clear()
            .env("PATH", SYSTEM_PATH)
            .env("HOME", "/root")
            .stdin(std::process::Stdio::null());
        for v in crate::shell::FILE_LOADER_VARS
            .iter()
            .chain(crate::shell::STARTUP_VARS)
        {
            cmd.env(v, "");
        }
        let out = cmd.output().unwrap();
        (
            out.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        )
    }

    /// A stand-in for the image record, one file per variable.
    fn image_record(dir: &std::path::Path, vars: &[(&str, &str)]) -> String {
        let d = dir.join("image-env");
        std::fs::create_dir_all(&d).unwrap();
        for (k, v) in vars {
            std::fs::write(d.join(k), v).unwrap();
        }
        d.to_str().unwrap().to_string()
    }

    /// An image whose `ENV` sets `LD_LIBRARY_PATH` (a CUDA base) and whose
    /// `BASH_ENV` setup file prepends to it: the command sees the setup
    /// file's value, as with Docker and a plain `bash -c`, as root and as a
    /// user. Before this, the loader variables came back after the setup
    /// file ran and reset it to the image's value.
    #[test]
    fn a_bash_env_change_to_an_image_loader_variable_survives() {
        let r = crate::shell::recorders::Recorders::new();
        let dir = tempfile::tempdir().unwrap();
        let setup = dir.path().join("setup.sh");
        std::fs::write(
            &setup,
            "export LD_LIBRARY_PATH=/opt/conda/envs/x/lib:$LD_LIBRARY_PATH\n",
        )
        .unwrap();
        let record = image_record(
            dir.path(),
            &[
                ("PATH", SYSTEM_PATH),
                ("LD_LIBRARY_PATH", "/usr/local/nvidia/lib64"),
                ("BASH_ENV", setup.to_str().unwrap()),
            ],
        );
        let want = "/opt/conda/envs/x/lib:/usr/local/nvidia/lib64";
        for user in [None, Some("nobody")] {
            let line = wrap_in(
                dir.path().to_str().unwrap(),
                &r.sys(),
                &record,
                &Wrap {
                    user,
                    ..Wrap::new(r#"printf %s "$LD_LIBRARY_PATH""#, "x", Stdin::Null)
                },
            );
            let (code, out, err) = layer_session(&line);
            assert_eq!(code, 0, "{user:?}: {err}");
            assert_eq!(out, want, "{user:?}");
        }
        // The Docker path, where the image's environment reaches the shell
        // directly, gives the same.
        let out = std::process::Command::new("/bin/sh")
            .arg("-c")
            .arg(crate::shell::command_shell(
                r#"printf %s "$LD_LIBRARY_PATH""#,
                &[],
            ))
            .env_clear()
            .env("PATH", SYSTEM_PATH)
            .env("LD_LIBRARY_PATH", "/usr/local/nvidia/lib64")
            .env("BASH_ENV", &setup)
            .output()
            .unwrap();
        assert_eq!(String::from_utf8_lossy(&out.stdout), want);
    }

    /// On the layer's environment, a command sees exactly the loader and
    /// start-up variables the image recorded (here only `BASH_ENV`), tested
    /// by presence: none of the layer's empties come back, as root and as a
    /// user, with and without an env file.
    #[test]
    fn a_command_sees_only_the_recorded_loader_variables() {
        let r = crate::shell::recorders::Recorders::new();
        let dir = tempfile::tempdir().unwrap();
        let noop = dir.path().join("noop.sh");
        std::fs::write(&noop, ":\n").unwrap();
        let record = image_record(
            dir.path(),
            &[("PATH", SYSTEM_PATH), ("BASH_ENV", noop.to_str().unwrap())],
        );
        let names = crate::shell::LOADER_VARS.join(" ");
        let print = format!(
            r#"for v in {names}; do eval "[ -n \"\${{$v+x}}\" ]" && printf '%s ' "$v"; done; true"#
        );
        let env_file = dir.path().join("env-x");
        for user in [None, Some("nobody")] {
            for with_file in [false, true] {
                std::fs::write(&env_file, "export BF_TEST_OTHER=1; ").unwrap();
                let line = wrap_in(
                    dir.path().to_str().unwrap(),
                    &r.sys(),
                    &record,
                    &Wrap {
                        user,
                        env_file: env_file.to_str().filter(|_| with_file),
                        ..Wrap::new(&print, "x", Stdin::Null)
                    },
                );
                let (code, out, err) = layer_session(&line);
                assert_eq!(code, 0, "{user:?} {with_file}: {err}");
                assert_eq!(out, "BASH_ENV ", "{user:?} env file={with_file}");
            }
        }
    }

    /// The kill's session line names `/bin/sh` by its path too.
    #[test]
    fn the_kill_line_finds_no_program_through_the_sessions_path() {
        let m = crate::shell::markers::Markers::new();
        let line = super::super::exec::kill_line("no-such-exec");
        let (code, _, err) = m.sh(&line);
        assert_eq!(code, 0, "{err}");
        assert!(m.ran().is_empty(), "{:?}", m.ran());
    }

    #[test]
    fn root_needs_no_privilege_drop() {
        let as_user = |user| {
            wrap(&Wrap {
                user: Some(user),
                ..Wrap::new("id", "x", Stdin::Null)
            })
        };
        assert!(!as_user("root").contains("setpriv"));
        assert!(as_user("agent").contains("setpriv"));
    }
}
