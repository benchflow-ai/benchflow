//! The toolbox of one sandbox: the daemon inside it, reached through
//! Daytona's proxy at `<toolboxProxyUrl>/<sandbox id>/...`.
//!
//! What the daemon does, as observed (daemon 0.222) and relied on here:
//!
//! - Commands run in *sessions*: a long-lived bash per session, as root, in
//!   the image's `WORKDIR` (root's home when there is none). Commands of one
//!   session run one after another, so every command gets its own session.
//!   A bare `exit` would end the session's bash, so every command is wrapped
//!   in `sh -c`.
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
use super::types::{CommandDto, LogsDto, SessionExec, SessionExecAnswer};
use crate::shell::{as_user, login_free_shell, quote};
use crate::{EXEC_ID_ENV, SandboxError};

/// Root-only folder for env files and transfer archives.
pub(crate) const STAGING: &str = "/run/benchflow";

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
        let id = format!("bf-{}", &uuid::Uuid::new_v4().simple().to_string()[..16]);
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
    /// toolbox says it is not running.
    pub(crate) fn error(&self, err: ApiError) -> SandboxError {
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
/// Daytona reports a command's exit code only once its stdout closes, so a
/// command that leaves a background process holding stdout (`server &`)
/// would never finish. A command run to completion ([`Stdin::Null`])
/// therefore writes its output to root-only files in [`STAGING`]; the wrapper
/// waits for the command alone, prints the files, and exits with its code.
/// Only the command carries the exec id, so a timeout kills it and leaves the
/// wrapper to print what it wrote. A spawned command ([`Stdin::Pipe`])
/// streams its output and is not wrapped this way.
pub(crate) fn wrap(
    command: &str,
    exec_id: &str,
    env_file: Option<&str>,
    cwd: Option<&str>,
    user: Option<&str>,
    stdin: Stdin,
) -> String {
    wrap_in(STAGING, command, exec_id, env_file, cwd, user, stdin)
}

/// [`wrap`], with the output files in `staging`.
fn wrap_in(
    staging: &str,
    command: &str,
    exec_id: &str,
    env_file: Option<&str>,
    cwd: Option<&str>,
    user: Option<&str>,
    stdin: Stdin,
) -> String {
    let inner = match user.filter(|u| !is_root(u)) {
        Some(user) => as_user(user, command),
        None => login_free_shell(command),
    };
    let mut script = format!("export {EXEC_ID_ENV}={}\n", quote(exec_id));
    if let Some(file) = env_file {
        let f = quote(file);
        script.push_str(&format!(
            "if ! . {f}; then rm -f {f}; echo 'benchflow: could not load the command environment' >&2; exit 126; fi\nrm -f {f}\n"
        ));
    }
    if let Some(dir) = cwd.filter(|d| !d.is_empty()) {
        script.push_str(&format!("cd {} || exit 126\n", quote(dir)));
    }
    let i = quote(&inner);
    let run = format!(
        "if command -v setsid >/dev/null 2>&1; then exec setsid sh -c {i}; else exec sh -c {i}; fi"
    );
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
    let mut line = format!("sh -c {}", quote(&script));
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
            "echo \"$GREETING from $(pwd)\"; echo id=$BENCHFLOW_EXEC_ID; exit 7",
            "abc123",
            Some(env_file.to_str().unwrap()),
            Some(dir.path().to_str().unwrap()),
            None,
            Stdin::Null,
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
        let line = wrap(
            "echo never",
            "x",
            None,
            Some("/no/such/folder"),
            None,
            Stdin::Null,
        );
        let (code, out, err) = run(&line);
        assert_eq!(code, 126);
        assert!(out.is_empty());
        assert!(!err.is_empty());
    }

    #[test]
    fn stdin_is_closed_unless_piped() {
        let staging = tempfile::tempdir().unwrap();
        let line = wrap_in(
            staging.path().to_str().unwrap(),
            "cat; echo done",
            "x",
            None,
            None,
            None,
            Stdin::Null,
        );
        let (code, out, _) = run(&line);
        assert_eq!((code, out.as_str()), (0, "done\n"));
        assert!(!wrap("cat", "x", None, None, None, Stdin::Pipe).ends_with("</dev/null"));
    }

    /// Daytona reports an exit code only once stdout closes, so a command
    /// that leaves `sleep &` behind used to run until its timeout (the
    /// process-flood probe took 2 minutes on Daytona). The wrapper must
    /// return when the command exits, with its output and code.
    #[test]
    fn a_background_child_does_not_hold_the_command_open() {
        let staging = tempfile::tempdir().unwrap();
        let line = wrap_in(
            staging.path().to_str().unwrap(),
            "(sleep 20 &); echo out; echo err >&2; exit 3",
            "x",
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
        let piped = wrap_in("/s", "true", "x", None, None, None, Stdin::Pipe);
        assert!(!piped.contains("/s/out-x"), "{piped}");
    }

    /// A killed command leaves no "Killed ..." notice in its output. Daytona's
    /// session shell printed one, with the whole wrapper script (20 KB), into
    /// every Codex trial's agent log when BenchFlow stopped the agent.
    #[test]
    fn a_killed_command_leaves_no_notice_in_its_output() {
        let staging = tempfile::tempdir().unwrap();
        for stdin in [Stdin::Pipe, Stdin::Null] {
            let line = wrap_in(
                staging.path().to_str().unwrap(),
                "echo before >&2; kill -9 $$",
                "x",
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

    #[test]
    fn root_needs_no_privilege_drop() {
        assert!(!wrap("id", "x", None, None, Some("root"), Stdin::Null).contains("setpriv"));
        assert!(wrap("id", "x", None, None, Some("agent"), Stdin::Null).contains("setpriv"));
    }
}
