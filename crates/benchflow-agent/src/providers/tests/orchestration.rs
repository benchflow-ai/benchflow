//! Starting the forwarder in a sandbox: what is written where, as whom, and
//! that the key is on no command line.

use std::path::Path;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use benchflow_sandbox::{Exec, ExecOutput, Process, ProcessHandle, Sandbox, SandboxError};
use tokio::io::AsyncWriteExt;
use tokio::sync::watch;

use super::{FAKE_KEY, provider_route};
use crate::providers::{FORWARDER_DIR, FORWARDER_JS, Forwarder, ForwarderError, NODE, ProviderKey};

/// What the forwarder process does in the fake sandbox.
#[derive(Clone, Copy)]
enum Behavior {
    /// Says it listens on 41234 and logs one request.
    Ready,
    /// Exits with an error (that quotes the key, which must not get out).
    Fails,
}

struct FakeSandbox {
    node: bool,
    behavior: Behavior,
    commands: Mutex<Vec<(Option<String>, String)>>,
    files: Mutex<Vec<(String, Vec<u8>, u32)>>,
    killed: Arc<Mutex<bool>>,
}

impl FakeSandbox {
    fn new(node: bool, behavior: Behavior) -> Self {
        Self {
            node,
            behavior,
            commands: Mutex::default(),
            files: Mutex::default(),
            killed: Arc::default(),
        }
    }
}

struct FakeHandle {
    exit: watch::Receiver<Option<i32>>,
    killed: Arc<Mutex<bool>>,
}

#[async_trait]
impl ProcessHandle for FakeHandle {
    async fn wait(&mut self) -> Result<i32, SandboxError> {
        let code = self
            .exit
            .wait_for(Option::is_some)
            .await
            .map_err(|_| SandboxError::Transport("gone".into()))?;
        Ok(code.unwrap_or(-1))
    }

    async fn kill(&mut self) -> Result<(), SandboxError> {
        *self.killed.lock().unwrap() = true;
        Ok(())
    }
}

#[async_trait]
impl Sandbox for FakeSandbox {
    fn id(&self) -> &str {
        "fake"
    }

    fn backend(&self) -> &str {
        "fake"
    }

    async fn exec(&self, exec: Exec) -> Result<ExecOutput, SandboxError> {
        self.commands
            .lock()
            .unwrap()
            .push((exec.user.clone(), exec.command.clone()));
        let missing_node = !self.node && exec.command.contains(&format!("test -x {NODE}"));
        Ok(ExecOutput {
            exit_code: if missing_node { 3 } else { 0 },
            ..ExecOutput::default()
        })
    }

    async fn spawn(&self, exec: Exec) -> Result<Process, SandboxError> {
        self.commands
            .lock()
            .unwrap()
            .push((exec.user.clone(), exec.command.clone()));
        let (stdin, _stdin_far) = tokio::io::duplex(1024);
        let (mut out_far, stdout) = tokio::io::duplex(4096);
        let (mut err_far, stderr) = tokio::io::duplex(4096);
        let (exit_tx, exit) = watch::channel(None);
        let behavior = self.behavior;
        tokio::spawn(async move {
            match behavior {
                Behavior::Ready => {
                    let _ = out_far
                        .write_all(b"{\"benchflow_forwarder\": \"ready\", \"port\": 41234}\n")
                        .await;
                    let _ = err_far
                        .write_all(b"{\"time\":\"t\",\"method\":\"POST\",\"path\":\"/anthropic/v1/messages\",\"status\":200,\"bytes\":10,\"ms\":5}\n")
                        .await;
                    // Keep the pipes open until killed.
                    std::future::pending::<()>().await;
                }
                Behavior::Fails => {
                    let line = format!("{{\"error\": \"cannot listen near {FAKE_KEY}\"}}\n");
                    let _ = err_far.write_all(line.as_bytes()).await;
                    drop(err_far);
                    drop(out_far);
                    exit_tx.send_replace(Some(2));
                }
            }
        });
        Ok(Process {
            stdin: Box::pin(stdin),
            stdout: Box::pin(stdout),
            stderr: Box::pin(stderr),
            handle: Box::new(FakeHandle {
                exit,
                killed: self.killed.clone(),
            }),
        })
    }

    async fn upload(&self, _local: &Path, _remote: &str) -> Result<(), SandboxError> {
        Err(SandboxError::Unsupported("not in this test".into()))
    }

    async fn write_file(
        &self,
        remote: &str,
        contents: &[u8],
        mode: u32,
    ) -> Result<(), SandboxError> {
        self.files
            .lock()
            .unwrap()
            .push((remote.to_string(), contents.to_vec(), mode));
        Ok(())
    }

    async fn download(&self, _remote: &str, _local: &Path) -> Result<(), SandboxError> {
        Err(SandboxError::Unsupported("not in this test".into()))
    }

    async fn destroy(&self) -> Result<(), SandboxError> {
        Ok(())
    }
}

#[tokio::test]
async fn the_key_goes_into_a_root_only_file_and_onto_no_command_line() {
    let sandbox = FakeSandbox::new(true, Behavior::Ready);
    let route = provider_route("claude", "vercel/deepseek/deepseek-v4-flash-0731");
    let key = ProviderKey::for_tests("vercel", FAKE_KEY);
    let forwarder = Forwarder::start(&sandbox, &route, &key).await.unwrap();
    assert_eq!(forwarder.port(), 41234);
    assert_eq!(forwarder.base_url(), "http://127.0.0.1:41234/anthropic");

    let commands = sandbox.commands.lock().unwrap().clone();
    assert_eq!(commands.len(), 2);
    for (user, command) in &commands {
        assert_eq!(user.as_deref(), Some("root"), "{command}");
        assert!(!command.contains(FAKE_KEY), "{command}");
    }
    assert!(commands[0].1.contains(&format!(
        "mkdir -p {FORWARDER_DIR} && chown root:root {FORWARDER_DIR} && chmod 700 {FORWARDER_DIR}"
    )));
    assert!(commands[1].1.ends_with(&format!(
        "exec {NODE} {FORWARDER_DIR}/forwarder.js {FORWARDER_DIR}/config.json"
    )));

    let files = sandbox.files.lock().unwrap().clone();
    let names: Vec<&str> = files.iter().map(|(path, _, _)| path.as_str()).collect();
    assert_eq!(
        names,
        [
            "/opt/benchflow/forwarder/forwarder.js",
            "/opt/benchflow/forwarder/config.json",
            "/opt/benchflow/forwarder/key",
        ]
    );
    assert!(files.iter().all(|(_, _, mode)| *mode == 0o600));
    assert_eq!(files[0].1, FORWARDER_JS.as_bytes());
    let config = String::from_utf8(files[1].1.clone()).unwrap();
    assert!(!config.contains(FAKE_KEY));
    assert!(config.contains("\"key_file\":\"/opt/benchflow/forwarder/key\""));
    assert_eq!(files[2].1, FAKE_KEY.as_bytes());

    let report = forwarder.stop().await;
    assert!(*sandbox.killed.lock().unwrap());
    assert_eq!(report.requests, 1);
    assert_eq!(report.exited_early, None);
    assert!(report.log.contains("/anthropic/v1/messages"));
}

#[tokio::test]
async fn a_forwarder_that_cannot_start_says_why_without_the_key() {
    let route = provider_route("codex", "vercel/alibaba/qwen3.7-flash");
    let key = ProviderKey::for_tests("vercel", FAKE_KEY);

    let sandbox = FakeSandbox::new(false, Behavior::Ready);
    let error = Forwarder::start(&sandbox, &route, &key).await.unwrap_err();
    assert!(matches!(error, ForwarderError::NoNode), "{error:?}");
    assert!(
        sandbox.files.lock().unwrap().is_empty(),
        "nothing is written without Node.js"
    );

    let sandbox = FakeSandbox::new(true, Behavior::Fails);
    let error = Forwarder::start(&sandbox, &route, &key).await.unwrap_err();
    let ForwarderError::NotReady { detail } = &error else {
        panic!("{error:?}")
    };
    assert!(
        detail.contains("cannot listen near ***REDACTED***"),
        "{detail}"
    );
    assert!(!error.to_string().contains(FAKE_KEY));
}
