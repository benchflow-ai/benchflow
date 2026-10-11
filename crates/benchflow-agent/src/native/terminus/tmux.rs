//! The tmux session Terminus-2 types into, driven through the sandbox API.
//!
//! Ported from Harbor's `tmux_session.py`
//! (https://github.com/laude-institute/harbor, Apache License 2.0): the
//! session command, `send-keys` batching under tmux's command-length limit,
//! the paste-buffer route for keys too long for one command, and
//! incremental output (`New Terminal Output:` / `Current Terminal Screen:`).
//! Not ported: the asciinema recording and building tmux from source; tmux
//! comes from the agent's image layer, and a package-manager install runs
//! here only when it is missing.

use std::collections::BTreeMap;
use std::time::Duration;

use base64::Engine;
use benchflow_sandbox::shell::SYSTEM_PATH;
use benchflow_sandbox::{Exec, ExecOutput, Sandbox, SandboxError};

use crate::shell::quote;

/// tmux rejects commands over its message buffer (~16 KB); stay under it.
pub const SEND_KEYS_MAX_COMMAND_LENGTH: usize = 16_000;
const COMMAND_TOO_LONG_MARKER: &str = "command too long";
const PASTE_BASE64_CHUNK_LEN: usize = 65_536;
/// The longest one tmux command may take.
const EXEC_TIMEOUT: Duration = Duration::from_secs(120);
/// The longest the fallback tmux install may take.
const INSTALL_TIMEOUT: Duration = Duration::from_secs(240);

/// The root script that installs tmux when the image lacks it: the agent
/// layer for Terminus-2, and the fallback at session start.
pub const INSTALL_SCRIPT: &str = r#"set -e
if command -v tmux >/dev/null 2>&1; then exit 0; fi
if command -v apt-get >/dev/null 2>&1; then
  # The image's package lists stay as the task left them: a solution may
  # install from them without an update.
  bf_lists=$(mktemp -d)
  cp -a /var/lib/apt/lists/. "$bf_lists"/ 2>/dev/null || true
  DEBIAN_FRONTEND=noninteractive apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tmux
  rm -rf /var/lib/apt/lists/*
  cp -a "$bf_lists"/. /var/lib/apt/lists/ 2>/dev/null || true
  rm -rf "$bf_lists"
elif command -v dnf >/dev/null 2>&1; then dnf install -y tmux
elif command -v yum >/dev/null 2>&1; then yum install -y tmux
elif command -v apk >/dev/null 2>&1; then apk add --no-cache tmux bash
elif command -v pacman >/dev/null 2>&1; then pacman -S --noconfirm tmux
elif command -v zypper >/dev/null 2>&1; then zypper install -y -n tmux
else
  echo "benchflow: no package manager found to install tmux, which Terminus-2 needs" >&2
  exit 1
fi
command -v tmux >/dev/null 2>&1
"#;

/// A tmux session in the sandbox.
pub struct TmuxSession<'a> {
    sandbox: &'a dyn Sandbox,
    name: String,
    user: String,
    workdir: String,
    env: BTreeMap<String, String>,
    pane_width: u32,
    pane_height: u32,
    logging_path: String,
    previous_buffer: Option<String>,
}

impl<'a> TmuxSession<'a> {
    pub fn new(sandbox: &'a dyn Sandbox, name: &str, user: &str, workdir: &str) -> Self {
        Self {
            sandbox,
            name: name.to_string(),
            user: user.to_string(),
            workdir: workdir.to_string(),
            env: BTreeMap::new(),
            pane_width: 160,
            pane_height: 40,
            logging_path: "/logs/agent/terminus_2.pane".to_string(),
            previous_buffer: None,
        }
    }

    /// Variables the shell in the session starts with (`tmux new-session -e`).
    pub fn env(mut self, env: BTreeMap<String, String>) -> Self {
        self.env = env;
        self
    }

    pub fn pane(mut self, width: u32, height: u32) -> Self {
        self.pane_width = width;
        self.pane_height = height;
        self
    }

    /// Where `pipe-pane` copies everything the pane shows.
    pub fn logging_path(&self) -> &str {
        &self.logging_path
    }

    async fn exec(&self, command: String) -> Result<ExecOutput, SandboxError> {
        self.sandbox
            .exec(
                Exec::new(command)
                    .user(self.user.clone())
                    .cwd(self.workdir.clone())
                    .timeout(EXEC_TIMEOUT),
            )
            .await
    }

    /// The command that starts the session (Harbor's `_tmux_start_session`).
    pub fn start_command(&self) -> String {
        let env_options: String = self
            .env
            .iter()
            .map(|(k, v)| format!("-e {} ", quote(&format!("{k}={v}"))))
            .collect();
        format!(
            "export TERM=xterm-256color && export SHELL=/bin/bash && tmux new-session {env_options}-x {w} -y {h} -d -s {name} 'bash --login' \\; pipe-pane -t {name} 'cat > {log}'",
            w = self.pane_width,
            h = self.pane_height,
            name = self.name,
            log = self.logging_path,
        )
    }

    /// Make sure tmux exists (installing it as root if not), then start the
    /// session. `Err(Ok(why))` is a failure the harness owns. The check runs
    /// as the session does, as its user with the image's `PATH`, so it finds
    /// the tmux the session will run; the install runs as root with the
    /// system folders' programs only.
    pub async fn start(&mut self) -> Result<(), StartError> {
        let check = self.exec("tmux -V".to_string()).await?;
        if !check.success() {
            let install = self
                .sandbox
                .exec(
                    Exec::new(INSTALL_SCRIPT)
                        .root()
                        .env("PATH", SYSTEM_PATH)
                        .timeout(INSTALL_TIMEOUT),
                )
                .await?;
            if !install.success() {
                return Err(StartError::NoTmux(install.tail(2000)));
            }
        }
        let started = self.exec(self.start_command()).await?;
        if !started.success() {
            return Err(StartError::Session(started.tail(2000)));
        }
        let history = self
            .exec("tmux set-option -g history-limit 10000000".into())
            .await?;
        if !history.success() {
            tracing::debug!(
                "could not raise tmux history-limit: {}",
                history.stderr.trim()
            );
        }
        Ok(())
    }

    pub async fn is_alive(&self) -> Result<bool, SandboxError> {
        Ok(self
            .exec(format!("tmux has-session -t {}", self.name))
            .await?
            .success())
    }

    fn send_keys_prefix(&self) -> String {
        format!("tmux send-keys -t {} --", quote(&self.name))
    }

    fn max_escaped_key_length(&self) -> usize {
        SEND_KEYS_MAX_COMMAND_LENGTH - self.send_keys_prefix().len() - 1
    }

    fn send_keys_command(&self, keys: &[String]) -> String {
        let quoted: Vec<String> = keys.iter().map(|k| quote(k)).collect();
        format!("{} {}", self.send_keys_prefix(), quoted.join(" "))
    }

    /// Whether `key` is too long for one `send-keys` command (UTF-8 bytes
    /// of the quoted key, as tmux measures).
    pub fn key_requires_paste(&self, key: &str) -> bool {
        quote(key).len() > self.max_escaped_key_length()
    }

    /// Group keys into `send-keys` commands that each fit the limit. Every
    /// key must fit on its own (long ones go through [`Self::paste_key`]).
    pub fn batch_keys(&self, keys: &[String]) -> Result<Vec<Vec<String>>, String> {
        let prefix = self.send_keys_prefix().len();
        let mut batches = Vec::new();
        let mut batch: Vec<String> = Vec::new();
        let mut len = prefix;
        for key in keys {
            let escaped = quote(key).len();
            if escaped > self.max_escaped_key_length() {
                return Err("key exceeds the tmux send-keys command-length limit; it must be sent via a paste buffer instead".into());
            }
            let addition = 1 + escaped;
            if !batch.is_empty() && len + addition > SEND_KEYS_MAX_COMMAND_LENGTH {
                batches.push(std::mem::take(&mut batch));
                len = prefix;
            }
            batch.push(key.clone());
            len += addition;
        }
        if !batch.is_empty() {
            batches.push(batch);
        }
        Ok(batches)
    }

    /// The `send-keys` commands for `keys`.
    pub fn send_keys_commands(&self, keys: &[String]) -> Result<Vec<String>, String> {
        Ok(self
            .batch_keys(keys)?
            .iter()
            .map(|b| self.send_keys_command(b))
            .collect())
    }

    fn send_error(&self, command: &str, out: &ExecOutput) -> SandboxError {
        let short: String = command.chars().take(100).collect();
        SandboxError::Transport(format!(
            "{}: failed to send non-blocking keys: command={short:?}, return_code={}, stderr={:?}, stdout={:?}",
            self.sandbox.id(),
            out.exit_code,
            out.stderr,
            out.stdout
        ))
    }

    fn too_long(out: &ExecOutput) -> bool {
        out.exit_code != 0 && out.stderr.contains(COMMAND_TOO_LONG_MARKER)
    }

    /// Deliver `key` through a paste buffer: staged into a file in base64
    /// pieces, then `load-buffer` and `paste-buffer`.
    pub async fn paste_key(&self, key: &str) -> Result<(), SandboxError> {
        if key.is_empty() {
            return Ok(());
        }
        let token = uuid::Uuid::new_v4().simple().to_string();
        let path = quote(&format!("/tmp/.harbor-tmux-paste-{token}"));
        let buffer = format!("harbor-paste-{token}");
        let payload = base64::engine::general_purpose::STANDARD.encode(key.as_bytes());
        let result = async {
            for (i, chunk) in payload
                .as_bytes()
                .chunks(PASTE_BASE64_CHUNK_LEN)
                .enumerate()
            {
                let chunk = std::str::from_utf8(chunk).unwrap_or_default();
                let redirect = if i == 0 { ">" } else { ">>" };
                let command = format!("printf %s {chunk} | base64 -d {redirect} {path}");
                let out = self.exec(command.clone()).await?;
                if !out.success() {
                    return Err(self.send_error(&command, &out));
                }
            }
            let command = format!(
                "tmux load-buffer -b {buffer} {path} && tmux paste-buffer -d -b {buffer} -t {}",
                quote(&self.name)
            );
            let out = self.exec(command.clone()).await?;
            if !out.success() {
                return Err(self.send_error(&command, &out));
            }
            Ok(())
        }
        .await;
        let _ = self.exec(format!("rm -f {path}")).await;
        result
    }

    async fn send_single_key(&self, key: &str) -> Result<(), SandboxError> {
        let command = self.send_keys_command(&[key.to_string()]);
        let out = self.exec(command.clone()).await?;
        if out.success() {
            return Ok(());
        }
        if Self::too_long(&out) {
            return self.paste_key(key).await;
        }
        Err(self.send_error(&command, &out))
    }

    async fn send_key_batches(&self, keys: &[String]) -> Result<(), SandboxError> {
        let batches = self.batch_keys(keys).map_err(SandboxError::Transport)?;
        for batch in batches {
            let command = self.send_keys_command(&batch);
            let out = self.exec(command.clone()).await?;
            if out.success() {
                continue;
            }
            if Self::too_long(&out) {
                // This tmux has a smaller limit: one key at a time, pasting
                // what it still rejects.
                for key in &batch {
                    self.send_single_key(key).await?;
                }
                continue;
            }
            return Err(self.send_error(&command, &out));
        }
        Ok(())
    }

    /// Send keys in order, pasting the ones too long for `send-keys`.
    pub async fn send_keys(&self, keys: &[String]) -> Result<(), SandboxError> {
        let mut batch: Vec<String> = Vec::new();
        for key in keys {
            if self.key_requires_paste(key) {
                if !batch.is_empty() {
                    self.send_key_batches(&std::mem::take(&mut batch)).await?;
                }
                self.paste_key(key).await?;
            } else {
                batch.push(key.clone());
            }
        }
        if !batch.is_empty() {
            self.send_key_batches(&batch).await?;
        }
        Ok(())
    }

    pub async fn capture_pane(&self, entire: bool) -> Result<String, SandboxError> {
        let extra = if entire { " -S -" } else { "" };
        Ok(self
            .exec(format!("tmux capture-pane -p{extra} -t {}", self.name))
            .await?
            .stdout)
    }

    /// New output since the last call, or the visible screen when that
    /// cannot be told (Harbor's `get_incremental_output`).
    pub async fn incremental_output(&mut self) -> Result<String, SandboxError> {
        let current = self.capture_pane(true).await?;
        let Some(previous) = self.previous_buffer.replace(current.clone()) else {
            let screen = self.capture_pane(false).await?;
            return Ok(format!("Current Terminal Screen:\n{screen}"));
        };
        match find_new_content(&previous, &current) {
            Some(new) if !new.trim().is_empty() => Ok(format!("New Terminal Output:\n{new}")),
            _ => {
                let screen = self.capture_pane(false).await?;
                Ok(format!("Current Terminal Screen:\n{screen}"))
            }
        }
    }
}

/// Harbor's `_find_new_content`, kept as it is: when the previous buffer
/// (stripped) occurs in the current one, the output from the position of
/// the previous buffer's last newline onward. That position is an offset
/// into the previous buffer applied to the current one (in characters, as
/// Python indexes), which in practice re-includes the last line seen.
pub fn find_new_content(previous: &str, current: &str) -> Option<String> {
    let pb = previous.trim();
    let byte_idx = current.find(pb)?;
    let char_idx = if pb.contains('\n') {
        let last_nl = pb.rfind('\n').unwrap_or(0);
        pb[..last_nl].chars().count()
    } else {
        current[..byte_idx].chars().count()
    };
    Some(current.chars().skip(char_idx).collect())
}

/// Why the session did not start.
#[derive(Debug)]
pub enum StartError {
    /// tmux is missing and could not be installed (the install's output).
    NoTmux(String),
    /// `tmux new-session` failed (its output).
    Session(String),
    Sandbox(SandboxError),
}

impl From<SandboxError> for StartError {
    fn from(e: SandboxError) -> Self {
        StartError::Sandbox(e)
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::Mutex;

    use async_trait::async_trait;
    use benchflow_sandbox::Process;

    use super::*;

    /// Records commands; fails those containing `fail_on`, or answers
    /// `send-keys` with "command too long" when `too_long` is set.
    #[derive(Default)]
    struct Recording {
        commands: Mutex<Vec<String>>,
        fail_on: Option<&'static str>,
        too_long: bool,
    }

    #[async_trait]
    impl Sandbox for Recording {
        fn id(&self) -> &str {
            "test-session-id"
        }
        fn backend(&self) -> &str {
            "fake"
        }
        async fn exec(&self, exec: Exec) -> Result<ExecOutput, SandboxError> {
            self.commands.lock().unwrap().push(exec.command.clone());
            let fail = self.fail_on.is_some_and(|f| exec.command.contains(f));
            let too_long = self.too_long && exec.command.starts_with("tmux send-keys");
            Ok(ExecOutput {
                exit_code: if fail || too_long { 1 } else { 0 },
                stderr: if too_long {
                    "command too long".into()
                } else {
                    String::new()
                },
                ..ExecOutput::default()
            })
        }
        async fn spawn(&self, _exec: Exec) -> Result<Process, SandboxError> {
            unimplemented!()
        }
        async fn upload(&self, _l: &Path, _r: &str) -> Result<(), SandboxError> {
            Ok(())
        }
        async fn write_file(&self, _r: &str, _c: &[u8], _m: u32) -> Result<(), SandboxError> {
            Ok(())
        }
        async fn download(&self, _r: &str, _l: &Path) -> Result<(), SandboxError> {
            Ok(())
        }
        async fn destroy(&self) -> Result<(), SandboxError> {
            Ok(())
        }
    }

    fn session(sb: &Recording) -> TmuxSession<'_> {
        TmuxSession::new(sb, "test-session", "agent", "/app")
    }

    fn payload(command: &str) -> Vec<String> {
        let parts = shlex::split(command).unwrap();
        assert_eq!(
            parts[..5],
            ["tmux", "send-keys", "-t", "test-session", "--"]
        );
        parts[5..].to_vec()
    }

    fn decode_pasted(commands: &[String]) -> String {
        let re = regex::Regex::new(r"^printf %s (\S+) \| base64 -d >>? \S+$").unwrap();
        let joined: String = commands
            .iter()
            .filter_map(|c| re.captures(c).map(|m| m[1].to_string()))
            .collect();
        String::from_utf8(
            base64::engine::general_purpose::STANDARD
                .decode(joined)
                .unwrap(),
        )
        .unwrap()
    }

    fn keys(k: &[&str]) -> Vec<String> {
        k.iter().map(|s| s.to_string()).collect()
    }

    // Harbor's tests/unit/agents/terminus_2/test_tmux_session.py

    #[test]
    fn send_keys_keeps_small_payload_single_command() {
        let sb = Recording::default();
        let commands = session(&sb)
            .send_keys_commands(&keys(&["echo hello world", "Enter"]))
            .unwrap();
        assert_eq!(commands.len(), 1);
        assert_eq!(payload(&commands[0]), ["echo hello world", "Enter"]);
    }

    #[test]
    fn send_keys_keys_starting_with_dash_are_literal() {
        let sb = Recording::default();
        let commands = session(&sb)
            .send_keys_commands(&keys(&["-x", "-Lfoo", "Enter"]))
            .unwrap();
        assert_eq!(payload(&commands[0]), ["-x", "-Lfoo", "Enter"]);
        assert!(commands[0].contains(" -t test-session -- "));
    }

    #[test]
    fn send_keys_many_small_keys_split_across_commands() {
        let sb = Recording::default();
        let max = SEND_KEYS_MAX_COMMAND_LENGTH;
        let ks: Vec<String> = (0..max / 500 * 3)
            .map(|i| format!("key{i:04}{}", "x".repeat(490)))
            .collect();
        let commands = session(&sb).send_keys_commands(&ks).unwrap();
        assert!(commands.len() >= 2);
        assert!(commands.iter().all(|c| c.len() <= max));
        let all: Vec<String> = commands.iter().flat_map(|c| payload(c)).collect();
        assert_eq!(all, ks);
    }

    #[test]
    fn send_keys_raises_on_oversized_key() {
        let sb = Recording::default();
        let long = "x".repeat(SEND_KEYS_MAX_COMMAND_LENGTH * 2);
        let err = session(&sb).send_keys_commands(&[long]).unwrap_err();
        assert!(err.contains("paste buffer"));
    }

    #[test]
    fn key_requires_paste_uses_utf8_byte_length() {
        let sb = Recording::default();
        let s = session(&sb);
        let multibyte = "é".repeat(SEND_KEYS_MAX_COMMAND_LENGTH * 9 / 16);
        assert!(multibyte.chars().count() < SEND_KEYS_MAX_COMMAND_LENGTH);
        assert!(s.key_requires_paste(&multibyte));
        assert!(!s.key_requires_paste(&"x".repeat(multibyte.chars().count())));
    }

    #[test]
    fn key_requires_paste_accounts_for_quote_inflation() {
        let sb = Recording::default();
        let quote_heavy = "a'".repeat(SEND_KEYS_MAX_COMMAND_LENGTH / 3);
        assert!(quote_heavy.len() < SEND_KEYS_MAX_COMMAND_LENGTH);
        assert!(session(&sb).key_requires_paste(&quote_heavy));
    }

    #[tokio::test]
    async fn send_non_blocking_keys_pastes_oversized_key() {
        let sb = Recording::default();
        let long = format!(
            "cat > /app/answer.txt << 'EOF'\n{}EOF\n",
            "answer line\n".repeat(SEND_KEYS_MAX_COMMAND_LENGTH / 6)
        );
        session(&sb)
            .send_keys(std::slice::from_ref(&long))
            .await
            .unwrap();
        let commands = sb.commands.lock().unwrap().clone();
        assert!(!commands.iter().any(|c| c.starts_with("tmux send-keys")));
        assert_eq!(decode_pasted(&commands), long);
        assert!(
            commands
                .iter()
                .any(|c| c.contains("tmux load-buffer") && c.contains("tmux paste-buffer -d"))
        );
        assert!(
            commands
                .last()
                .unwrap()
                .starts_with("rm -f /tmp/.harbor-tmux-paste-")
        );
    }

    #[tokio::test]
    async fn send_non_blocking_keys_preserves_key_order_around_paste() {
        let sb = Recording::default();
        let long = "y".repeat(SEND_KEYS_MAX_COMMAND_LENGTH * 2);
        session(&sb)
            .send_keys(
                &keys(&["echo before", "Enter"])
                    .into_iter()
                    .chain([long.clone()])
                    .chain(keys(&["Enter"]))
                    .collect::<Vec<_>>(),
            )
            .await
            .unwrap();
        let commands = sb.commands.lock().unwrap().clone();
        assert_eq!(payload(&commands[0]), ["echo before", "Enter"]);
        let paste = commands
            .iter()
            .position(|c| c.contains("paste-buffer"))
            .unwrap();
        let last_send = commands
            .iter()
            .rposition(|c| c.starts_with("tmux send-keys"))
            .unwrap();
        assert!(paste < last_send);
        assert_eq!(payload(&commands[last_send]), ["Enter"]);
    }

    #[tokio::test]
    async fn paste_key_removes_staging_file_on_failure() {
        let sb = Recording {
            fail_on: Some("load-buffer"),
            ..Recording::default()
        };
        let err = session(&sb).paste_key("hello").await.unwrap_err();
        assert!(err.to_string().contains("failed to send"), "{err}");
        let commands = sb.commands.lock().unwrap().clone();
        assert!(
            commands
                .last()
                .unwrap()
                .starts_with("rm -f /tmp/.harbor-tmux-paste-")
        );
    }

    #[tokio::test]
    async fn send_keys_falls_back_to_pasting_when_tmux_says_too_long() {
        let sb = Recording {
            too_long: true,
            ..Recording::default()
        };
        session(&sb).send_keys(&keys(&["echo hi"])).await.unwrap();
        let commands = sb.commands.lock().unwrap().clone();
        // The batch, the key alone, then the paste route.
        assert!(commands[0].starts_with("tmux send-keys"));
        assert!(commands[1].starts_with("tmux send-keys"));
        assert_eq!(decode_pasted(&commands), "echo hi");
    }

    #[tokio::test]
    async fn a_failed_send_names_the_command_and_its_output() {
        let sb = Recording {
            fail_on: Some("send-keys"),
            ..Recording::default()
        };
        let err = session(&sb)
            .send_keys(&keys(&["ls"]))
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("test-session-id: failed to send non-blocking keys"),
            "{err}"
        );
        assert!(err.contains("return_code=1"), "{err}");
    }

    #[test]
    fn the_session_starts_with_the_pane_size_logging_and_env() {
        let sb = Recording::default();
        let s = session(&sb).env(BTreeMap::from([("A".to_string(), "b c".to_string())]));
        assert_eq!(
            s.start_command(),
            "export TERM=xterm-256color && export SHELL=/bin/bash && tmux new-session -e 'A=b c' -x 160 -y 40 -d -s test-session 'bash --login' \\; pipe-pane -t test-session 'cat > /logs/agent/terminus_2.pane'"
        );
    }

    #[test]
    fn new_content_follows_harbor() {
        assert_eq!(find_new_content("a", "a\nb").as_deref(), Some("a\nb"));
        assert_eq!(
            find_new_content("$ ls\nfile\n$ ", "$ ls\nfile\n$ cat x\nhello\n$ ").as_deref(),
            Some("\n$ cat x\nhello\n$ ")
        );
        assert_eq!(find_new_content("gone", "other"), None);
    }
}
