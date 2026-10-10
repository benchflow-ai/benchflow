//! Unit tests for the docker backend, against a fake `docker` CLI.
//!
//! The fake is a shell script in a temp folder. Every call records its
//! arguments, a copy of any `--env-file` (and that file's mode), its own
//! environment, and (when the scenario asks) its stdin; then it acts out the
//! scenario the test wrote. No Docker is needed or touched.

use std::collections::BTreeMap;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

use super::cli::{Cli, Flavor};
use super::{DockerProvider, DockerSandbox};
use crate::shell::login_free_shell;
use crate::{
    EXEC_ID_ENV, Exec, Image, ImageSource, ImageSpec, NetworkPolicy, OWNER_LABEL, Provider,
    Sandbox, SandboxError, SandboxSpec, owner,
};

const PREAMBLE: &str = r#"#!/bin/sh
# Fake docker CLI for BenchFlow's unit tests.
dir=$(cd "$(dirname "$0")" && pwd)
id=$$
mkdir -p "$dir/calls"
for a in "$@"; do printf '%s\0' "$a"; done > "$dir/calls/$id.args"
prev=
for a in "$@"; do
  if [ "$prev" = "--env-file" ]; then
    cp "$a" "$dir/calls/$id.envfile"
    ls -l "$a" | cut -c1-10 > "$dir/calls/$id.envmode"
  fi
  if [ "$prev" = "-f" ] && [ -f "$a" ]; then cp "$a" "$dir/calls/$id.dockerfile"; fi
  prev=$a
done
env > "$dir/calls/$id.env"
printf '%s\n' "$id" >> "$dir/order"
"#;

struct Fake {
    dir: tempfile::TempDir,
    bin: PathBuf,
}

#[derive(Debug)]
struct Call {
    args: Vec<String>,
    env_file: Option<String>,
    env_mode: Option<String>,
    dockerfile: Option<String>,
    stdin: Option<Vec<u8>>,
    process_env: String,
}

impl Fake {
    /// A fake CLI acting out `scenario`, shell code that runs after the
    /// call is recorded (`$dir` is the fake's folder, `$id` the call id).
    fn new(scenario: &str) -> Self {
        use std::io::Write as _;
        let dir = tempfile::tempdir().unwrap();
        let bin = dir.path().join("docker");
        // A child writes the fake: on Linux a file this multithreaded test
        // process held open for writing cannot be run ("Text file busy")
        // while a child another test forked still holds the descriptor.
        let mut child = std::process::Command::new("sh")
            .arg("-c")
            .arg(r#"cat > "$1" && chmod 755 "$1""#)
            .arg("sh")
            .arg(&bin)
            .stdin(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        child
            .stdin
            .take()
            .unwrap()
            .write_all(format!("{PREAMBLE}{scenario}\n").as_bytes())
            .unwrap();
        assert!(child.wait().unwrap().success(), "writing the fake docker");
        Self { dir, bin }
    }

    fn provider(&self) -> DockerProvider {
        DockerProvider::new("docker").with_binary(&self.bin)
    }

    fn sandbox(&self) -> DockerSandbox {
        DockerSandbox {
            cli: Cli {
                name: "docker".into(),
                binary: self.bin.clone(),
                flavor: Flavor::Docker,
            },
            name: "bf-test".into(),
        }
    }

    fn path(&self, rel: &str) -> PathBuf {
        self.dir.path().join(rel)
    }

    fn calls(&self) -> Vec<Call> {
        let order = std::fs::read_to_string(self.path("order")).unwrap_or_default();
        order
            .lines()
            .map(|id| {
                let read = |ext: &str| std::fs::read(self.path(&format!("calls/{id}.{ext}"))).ok();
                let text = |ext: &str| read(ext).map(|b| String::from_utf8_lossy(&b).into_owned());
                let args = read("args").unwrap_or_default();
                let mut args: Vec<String> = args
                    .split(|b| *b == 0)
                    .map(|a| String::from_utf8_lossy(a).into_owned())
                    .collect();
                args.pop(); // after the last NUL
                Call {
                    args,
                    env_file: text("envfile"),
                    env_mode: text("envmode").map(|m| m.trim().to_string()),
                    dockerfile: text("dockerfile"),
                    stdin: read("stdin"),
                    process_env: text("env").unwrap_or_default(),
                }
            })
            .collect()
    }

    /// The calls whose first argument is `sub`.
    fn calls_to(&self, sub: &str) -> Vec<Call> {
        self.calls()
            .into_iter()
            .filter(|c| c.args.first().map(String::as_str) == Some(sub))
            .collect()
    }
}

fn strings(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| s.to_string()).collect()
}

fn no_log() -> impl Fn(&str) + Send + Sync {
    |_line: &str| {}
}

fn spec(source: ImageSource) -> ImageSpec {
    ImageSpec {
        source,
        layers: Vec::new(),
        name: "My Task".into(),
        digest: "0123456789abcdef0123".into(),
        build_timeout: Some(Duration::from_secs(60)),
        platform: None,
    }
}

fn sandbox_spec() -> SandboxSpec {
    SandboxSpec {
        image: Image {
            reference: "ubuntu:24.04".into(),
            fresh: false,
            duration: Duration::ZERO,
        },
        name: "trial-1".into(),
        cpus: Some(2.0),
        memory_mb: Some(2048),
        disk_mb: Some(10_000),
        gpus: 1,
        pids_limit: Some(256),
        network: NetworkPolicy::None,
        env: BTreeMap::from([
            ("API_KEY".to_string(), "sk-very-secret".to_string()),
            ("MULTI".to_string(), "line1\nline2".to_string()),
        ]),
        workdir: Some("/app".into()),
        labels: BTreeMap::from([("custom".to_string(), "1".to_string())]),
        runtime: None,
    }
}

// --- status ---------------------------------------------------------------

#[tokio::test]
async fn status_names_the_server_version_and_context() {
    let fake = Fake::new(
        r#"case "$1" in version) echo '{"Client":{"Version":"28.1.1","Context":"default"},"Server":{"Version":"28.1.1"}}';; esac"#,
    );
    let status = fake.provider().status().await;
    assert!(status.ok, "{status:?}");
    assert!(
        status
            .summary
            .starts_with("Docker 28.1.1 (server), context default"),
        "{}",
        status.summary
    );
    assert_eq!(
        fake.calls()[0].args,
        strings(&["version", "--format", "{{json .}}"])
    );
}

#[tokio::test]
async fn status_explains_a_stopped_daemon() {
    let fake = Fake::new(
        r#"case "$1" in version)
  echo '{"Client":{"Version":"28.1.1"},"Server":null}'
  echo 'Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?' >&2
  exit 1;; esac"#,
    );
    let status = fake.provider().status().await;
    assert!(!status.ok);
    assert!(
        status.summary.contains("cannot reach its daemon"),
        "{}",
        status.summary
    );
    assert!(status.fix.unwrap().contains("Start Docker"));
}

#[tokio::test]
async fn status_explains_a_socket_permission_problem() {
    let fake = Fake::new(
        r#"echo 'permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock' >&2; exit 1"#,
    );
    let status = fake.provider().status().await;
    assert!(!status.ok);
    assert!(status.fix.unwrap().contains("usermod -aG docker"));
}

#[tokio::test]
async fn status_explains_a_missing_cli() {
    let provider = DockerProvider::new("docker").with_binary("/nonexistent/bin/docker");
    let status = provider.status().await;
    assert!(!status.ok);
    assert!(
        status.summary.contains("was not found"),
        "{}",
        status.summary
    );
    assert!(status.fix.unwrap().contains("Install Docker"));
}

// --- prepare --------------------------------------------------------------

#[tokio::test]
async fn prepare_reuses_a_cached_build() {
    let fake = Fake::new(r#"case "$1" in image) echo sha256:cached;; build) exit 99;; esac"#);
    let ctx = tempfile::tempdir().unwrap();
    let source = ImageSource::Build {
        context: ctx.path().into(),
        dockerfile: ctx.path().join("Dockerfile"),
    };
    let image = fake
        .provider()
        .prepare(&spec(source), &no_log())
        .await
        .unwrap();
    assert_eq!(image.reference, "benchflow/my-task:0123456789ab");
    assert!(!image.fresh);
    assert!(fake.calls_to("build").is_empty());
    assert_eq!(
        fake.calls()[0].args,
        strings(&[
            "image",
            "inspect",
            "--format",
            "{{.Id}}",
            "benchflow/my-task:0123456789ab"
        ])
    );
}

#[tokio::test]
async fn prepare_builds_with_labels_and_streams_the_log() {
    let fake = Fake::new(
        r#"case "$1" in
  image) echo 'Error: No such image' >&2; exit 1;;
  build) echo '#1 [internal] load build definition'; echo '#2 DONE 0.1s' >&2;;
esac"#,
    );
    let ctx = tempfile::tempdir().unwrap();
    let dockerfile = ctx.path().join("Dockerfile");
    let source = ImageSource::Build {
        context: ctx.path().into(),
        dockerfile: dockerfile.clone(),
    };
    let lines = Arc::new(Mutex::new(Vec::<String>::new()));
    let sink = lines.clone();
    let log = move |line: &str| sink.lock().unwrap().push(line.to_string());
    let image = fake.provider().prepare(&spec(source), &log).await.unwrap();
    assert!(image.fresh);
    let builds = fake.calls_to("build");
    assert_eq!(builds.len(), 1);
    assert_eq!(
        builds[0].args,
        vec![
            "build".to_string(),
            "-f".into(),
            dockerfile.display().to_string(),
            "-t".into(),
            "benchflow/my-task:0123456789ab".into(),
            "--label".into(),
            format!("{OWNER_LABEL}={}", owner::scope()),
            "--label".into(),
            "ai.benchflow.digest=0123456789abcdef0123".into(),
            ctx.path().display().to_string(),
        ]
    );
    assert!(builds[0].process_env.contains("BUILDKIT_PROGRESS=plain"));
    let lines = lines.lock().unwrap();
    assert!(
        lines.contains(&"#1 [internal] load build definition".to_string()),
        "{lines:?}"
    );
    assert!(lines.contains(&"#2 DONE 0.1s".to_string()), "{lines:?}");
}

#[tokio::test]
async fn a_failing_dockerfile_step_is_the_tasks_fault_and_not_retried() {
    let fake = Fake::new(
        r#"case "$1" in
  image) exit 1;;
  build)
    echo '#5 [2/2] RUN exit 3' >&2
    echo 'ERROR: failed to solve: process "/bin/sh -c exit 3" did not complete successfully: exit code: 3' >&2
    exit 1;;
esac"#,
    );
    let ctx = tempfile::tempdir().unwrap();
    let source = ImageSource::Build {
        context: ctx.path().into(),
        dockerfile: ctx.path().join("Dockerfile"),
    };
    let err = fake
        .provider()
        .prepare(&spec(source), &no_log())
        .await
        .unwrap_err();
    let SandboxError::Image {
        task_fault,
        log_tail,
        message,
        ..
    } = &err
    else {
        panic!("wrong error: {err:?}")
    };
    assert!(*task_fault);
    assert!(
        log_tail.contains("did not complete successfully"),
        "{log_tail}"
    );
    assert!(message.contains("task's Dockerfile"), "{message}");
    assert!(!err.is_transient());
    assert_eq!(fake.calls_to("build").len(), 1);
}

#[tokio::test]
async fn a_transient_build_failure_is_retried_twice() {
    let fake = Fake::new(
        r#"case "$1" in
  image) exit 1;;
  build)
    n=$(cat "$dir/count" 2>/dev/null || echo 0); n=$((n + 1)); echo "$n" > "$dir/count"
    if [ "$n" -lt 3 ]; then echo 'read tcp 1.2.3.4:443: read: connection reset by peer' >&2; exit 1; fi
    echo built;;
esac"#,
    );
    let ctx = tempfile::tempdir().unwrap();
    let source = ImageSource::Build {
        context: ctx.path().into(),
        dockerfile: ctx.path().join("Dockerfile"),
    };
    let image = fake
        .provider()
        .prepare(&spec(source), &no_log())
        .await
        .unwrap();
    assert!(image.fresh);
    assert_eq!(fake.calls_to("build").len(), 3);
}

#[tokio::test]
async fn a_build_that_keeps_failing_transiently_is_not_the_tasks_fault() {
    let fake = Fake::new(
        r#"case "$1" in
  image) exit 1;;
  build) echo 'ERROR: failed to solve: failed to fetch: i/o timeout' >&2; exit 1;;
esac"#,
    );
    let ctx = tempfile::tempdir().unwrap();
    let source = ImageSource::Build {
        context: ctx.path().into(),
        dockerfile: ctx.path().join("Dockerfile"),
    };
    let err = fake
        .provider()
        .prepare(&spec(source), &no_log())
        .await
        .unwrap_err();
    assert!(
        matches!(
            err,
            SandboxError::Image {
                task_fault: false,
                ..
            }
        ),
        "{err:?}"
    );
    assert!(err.is_transient());
    assert_eq!(fake.calls_to("build").len(), 3);
}

#[tokio::test]
async fn a_build_past_its_timeout_is_stopped() {
    let fake = Fake::new(r#"case "$1" in image) exit 1;; build) exec sleep 30;; esac"#);
    let ctx = tempfile::tempdir().unwrap();
    let mut spec = spec(ImageSource::Build {
        context: ctx.path().into(),
        dockerfile: ctx.path().join("Dockerfile"),
    });
    spec.build_timeout = Some(Duration::from_millis(300));
    let started = std::time::Instant::now();
    let err = fake.provider().prepare(&spec, &no_log()).await.unwrap_err();
    assert!(started.elapsed() < Duration::from_secs(10));
    let SandboxError::Image { message, .. } = err else {
        panic!("wrong error")
    };
    assert!(message.contains("build timeout"), "{message}");
}

#[tokio::test]
async fn prepare_pulls_a_missing_image() {
    let fake = Fake::new(r#"case "$1" in image) exit 1;; pull) echo 'Status: Downloaded';; esac"#);
    let mut spec = spec(ImageSource::Pull {
        reference: "ubuntu:24.04".into(),
    });
    spec.platform = Some("linux/amd64".into());
    let image = fake.provider().prepare(&spec, &no_log()).await.unwrap();
    assert_eq!(image.reference, "ubuntu:24.04");
    assert!(image.fresh);
    assert_eq!(
        fake.calls_to("pull")[0].args,
        strings(&["pull", "--platform", "linux/amd64", "ubuntu:24.04"])
    );
}

#[tokio::test]
async fn an_image_the_registry_lacks_is_the_tasks_fault() {
    let fake = Fake::new(
        r#"case "$1" in image) exit 1;; pull) echo 'Error response from daemon: manifest unknown' >&2; exit 1;; esac"#,
    );
    let spec = spec(ImageSource::Pull {
        reference: "nope/nope:1".into(),
    });
    let err = fake.provider().prepare(&spec, &no_log()).await.unwrap_err();
    assert!(
        matches!(
            err,
            SandboxError::Image {
                task_fault: true,
                ..
            }
        ),
        "{err:?}"
    );
}

#[tokio::test]
async fn layers_build_a_derived_image_tagged_by_base_and_layers() {
    let fake = Fake::new(
        r#"case "$1" in
  image)
    case "$4" in
      '{{.Id}}') if [ "$5" = "ubuntu:24.04" ]; then echo sha256:base; else exit 1; fi;;
      *) printf 'sha256:base\t{"User":"agent","Env":[]}\n';;
    esac;;
  build) echo ok;;
esac"#,
    );
    let mut spec = spec(ImageSource::Pull {
        reference: "ubuntu:24.04".into(),
    });
    spec.layers = vec!["npm i -g x".into(), "echo done".into()];
    let image = fake.provider().prepare(&spec, &no_log()).await.unwrap();
    let expected = super::image::layers_tag(&spec, "sha256:base");
    assert_eq!(image.reference, expected);
    assert!(image.fresh, "the layers were built now");
    let builds = fake.calls_to("build");
    assert_eq!(builds.len(), 1);
    assert_eq!(&builds[0].args[3..5], ["-t".to_string(), expected]);
    assert_eq!(
        builds[0].dockerfile.as_deref(),
        Some(super::image::layers_dockerfile("ubuntu:24.04", &spec.layers, Some("agent")).as_str())
    );
}

// --- create ---------------------------------------------------------------

#[tokio::test]
async fn create_runs_one_container_with_limits_labels_and_a_private_env_file() {
    let fake =
        Fake::new(r#"case "$1" in run) echo 0123abcd;; inspect) printf 'running\t0\n';; esac"#);
    let sandbox = fake.provider().create(&sandbox_spec()).await.unwrap();
    assert!(sandbox.id().starts_with("bf-trial-1-"), "{}", sandbox.id());
    assert_eq!(sandbox.backend(), "docker");

    let run = &fake.calls_to("run")[0];
    let env_path = run.args[run.args.iter().position(|a| a == "--env-file").unwrap() + 1].clone();
    let mut expected = strings(&["run", "-d", "--init", "--name"]);
    expected.push(sandbox.id().to_string());
    for (k, v) in [
        ("ai.benchflow.host", owner::hostname().to_string()),
        ("ai.benchflow.name", "trial-1".to_string()),
        ("ai.benchflow.owner", owner::scope()),
        ("ai.benchflow.pid", std::process::id().to_string()),
        ("custom", "1".to_string()),
    ] {
        expected.push("--label".into());
        expected.push(format!("{k}={v}"));
    }
    expected.extend(strings(&[
        "--cpus",
        "2",
        "--memory",
        "2048m",
        "--memory-swap",
        "2048m",
        "--pids-limit",
        "256",
        "--network",
        "none",
        "--gpus",
        "all",
        "-w",
        "/app",
        "--env-file",
    ]));
    expected.push(env_path.clone());
    expected.extend(strings(&[
        "-e",
        "MULTI",
        "--entrypoint",
        "sh",
        "ubuntu:24.04",
        "-c",
    ]));
    expected.push(super::run::KEEPALIVE.to_string());
    assert_eq!(run.args, expected);

    // The values never reach a command line; the file was private and is gone.
    assert!(!run.args.iter().any(|a| a.contains("sk-very-secret")));
    assert_eq!(run.env_file.as_deref(), Some("API_KEY=sk-very-secret\n"));
    assert_eq!(run.env_mode.as_deref(), Some("-rw-------"));
    assert!(
        !Path::new(&env_path).exists(),
        "the env file is deleted after `docker run`"
    );
    // The multi-line value travels in the CLI's own environment.
    assert!(
        run.process_env.contains("MULTI=line1\nline2"),
        "{}",
        run.process_env
    );

    // Then it waited until the container ran.
    let inspect = &fake.calls_to("inspect")[0];
    assert_eq!(inspect.args.last().unwrap(), sandbox.id());
}

#[tokio::test]
async fn a_filtered_network_gives_root_net_admin_and_keeps_the_network() {
    let fake =
        Fake::new(r#"case "$1" in run) echo 0123abcd;; inspect) printf 'running\t0\n';; esac"#);
    let mut spec = sandbox_spec();
    spec.network = NetworkPolicy::Filtered { allow: None };
    fake.provider().create(&spec).await.unwrap();
    let run = &fake.calls_to("run")[0];
    let cap = run.args.iter().position(|a| a == "--cap-add").unwrap();
    assert_eq!(run.args[cap + 1], "NET_ADMIN");
    assert!(!run.args.iter().any(|a| a == "--network"), "{:?}", run.args);
}

#[tokio::test]
async fn create_fails_clearly_when_the_container_exits_at_once() {
    let fake = Fake::new(
        r#"case "$1" in
  run) echo 0123abcd;;
  inspect) printf 'exited\t127\n';;
  logs) echo 'sh: not found';;
  rm) echo removed;;
esac"#,
    );
    let mut spec = sandbox_spec();
    spec.env.clear();
    let Err(err) = fake.provider().create(&spec).await else {
        panic!("create should fail")
    };
    let SandboxError::Start(message) = err else {
        panic!("wrong error")
    };
    assert!(message.contains("exit code 127"), "{message}");
    assert!(message.contains("sh: not found"), "{message}");
    // The broken container was removed.
    assert_eq!(fake.calls_to("rm").len(), 1);
}

// --- exec -----------------------------------------------------------------

#[tokio::test]
async fn exec_passes_user_cwd_and_env_and_captures_both_streams() {
    let fake = Fake::new(
        r#"case "$1" in exec) printf 'out-line\n'; printf 'err-line\n' >&2; exit 3;; esac"#,
    );
    let out = fake
        .sandbox()
        .exec(
            Exec::new("echo hi")
                .user("agent")
                .cwd("/work")
                .env("FOO", "bar"),
        )
        .await
        .unwrap();
    assert_eq!(out.exit_code, 3);
    assert_eq!(out.stdout, "out-line\n");
    assert_eq!(out.stderr, "err-line\n");
    assert!(!out.timed_out);

    let call = &fake.calls()[0];
    let env_path = call.args[7].clone();
    let mut expected = strings(&["exec", "-i", "-u", "agent", "-w", "/work", "--env-file"]);
    expected.push(env_path);
    expected.extend(strings(&["bf-test", "sh", "-c"]));
    expected.push(login_free_shell("echo hi"));
    assert_eq!(call.args, expected);
    let env_file = call.env_file.clone().unwrap();
    let mut lines = env_file.lines();
    let exec_id = lines
        .next()
        .unwrap()
        .strip_prefix(&format!("{EXEC_ID_ENV}="))
        .unwrap()
        .to_string();
    assert_eq!(exec_id.len(), 32);
    assert_eq!(lines.next(), Some("FOO=bar"));
    assert_eq!(call.env_mode.as_deref(), Some("-rw-------"));
    assert!(
        !Path::new(&call.args[7]).exists(),
        "the env file is deleted once the command ends"
    );
}

#[tokio::test]
async fn a_timeout_kills_the_command_inside_the_container() {
    let fake = Fake::new(
        r#"case "$1" in exec)
  if [ "$2" = "-u" ] && [ "$3" = "0" ] && [ "$7" = "sh" ]; then exit 0; fi
  echo started; exec sleep 30;;
esac"#,
    );
    let started = std::time::Instant::now();
    let out = fake
        .sandbox()
        .exec(Exec::new("sleep 30").timeout(Duration::from_secs(3)))
        .await
        .unwrap();
    assert!(out.timed_out);
    assert_eq!(out.exit_code, crate::TIMEOUT_EXIT_CODE);
    assert_eq!(out.stdout, "started\n");
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "{:?}",
        started.elapsed()
    );

    let calls = fake.calls();
    assert_eq!(calls.len(), 2, "{calls:?}");
    let exec_id = calls
        .iter()
        .find_map(|c| c.env_file.as_deref())
        .unwrap()
        .lines()
        .find_map(|l| l.strip_prefix(&format!("{EXEC_ID_ENV}=")))
        .unwrap()
        .to_string();
    let kill = calls
        .iter()
        .find(|c| c.args.get(6).map(String::as_str) == Some("sh"))
        .expect("the kill ran");
    let path = format!("PATH={}", crate::shell::SYSTEM_PATH);
    assert_eq!(
        &kill.args[..8],
        strings(&["exec", "-u", "0", "-e", &path, "bf-test", "sh", "-c"]).as_slice()
    );
    assert_eq!(kill.args[8], crate::shell::kill_tagged(&exec_id));
}

/// Root in a container cannot read another user's `/proc/<pid>/environ`
/// (no `CAP_SYS_PTRACE`), so a user's command is killed as that user first.
/// Guards the e2e-docker fix: on Docker 29, a timed-out command run as
/// `agent` survived the root-only kill.
#[tokio::test]
async fn a_timeout_kills_a_users_command_as_that_user_then_as_root() {
    let fake = Fake::new(
        r#"case "$1" in exec)
  if [ "$2" = "-u" ] && [ "$7" = "sh" ]; then exit 0; fi
  exec sleep 30;;
esac"#,
    );
    let out = fake
        .sandbox()
        .exec(
            Exec::new("sleep 30")
                .user("agent")
                .timeout(Duration::from_secs(2)),
        )
        .await
        .unwrap();
    assert!(out.timed_out);
    // The kills are the calls that run `sh` (the fake's `$7`). Under load
    // the timed-out command can be killed before its fake records the call,
    // so only the kills are counted.
    let calls = fake.calls();
    let kills: Vec<_> = calls
        .iter()
        .filter(|c| {
            c.args.get(1).map(String::as_str) == Some("-u")
                && c.args.get(6).map(String::as_str) == Some("sh")
        })
        .collect();
    assert_eq!(kills.len(), 2, "{calls:?}");
    let path = format!("PATH={}", crate::shell::SYSTEM_PATH);
    assert_eq!(
        &kills[0].args[..6],
        strings(&["exec", "-u", "agent", "-e", &path, "bf-test"]).as_slice()
    );
    assert_eq!(
        &kills[1].args[..6],
        strings(&["exec", "-u", "0", "-e", &path, "bf-test"]).as_slice()
    );
}

#[tokio::test]
async fn exec_reports_a_vanished_container_as_an_error() {
    let fake =
        Fake::new(r#"echo 'Error response from daemon: No such container: bf-test' >&2; exit 1"#);
    let err = fake.sandbox().exec(Exec::new("true")).await.unwrap_err();
    assert!(
        matches!(&err, SandboxError::Transport(m) if m.contains("no longer running")),
        "{err:?}"
    );
}

#[tokio::test]
async fn exec_reports_a_stopped_daemon_as_unavailable() {
    let fake = Fake::new(
        r#"echo 'Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?' >&2; exit 1"#,
    );
    let err = fake.sandbox().exec(Exec::new("true")).await.unwrap_err();
    assert!(matches!(err, SandboxError::Unavailable { .. }), "{err:?}");
}

#[tokio::test]
async fn exec_keeps_the_tail_of_huge_output() {
    // 2 MiB of output: more than the 1 MiB unit-test builds keep (16 MiB in real builds).
    let fake = Fake::new(
        r#"case "$1" in exec) head -c 2097152 /dev/zero | tr '\0' 'x'; echo; echo THE-END;; esac"#,
    );
    let out = fake.sandbox().exec(Exec::new("noisy")).await.unwrap();
    assert!(
        out.stdout.starts_with("[benchflow: "),
        "{}",
        &out.stdout[..80]
    );
    assert!(out.stdout.ends_with("THE-END\n"));
    assert!(out.stdout.len() <= crate::capture::OUTPUT_CAP + 200);
}

// --- spawn ----------------------------------------------------------------

#[tokio::test]
async fn spawn_gives_live_pipes_and_the_exit_code() {
    let fake = Fake::new(r#"case "$1" in exec) exec cat;; esac"#);
    let mut process = fake.sandbox().spawn(Exec::new("cat")).await.unwrap();
    let mut lines = BufReader::new(process.stdout).lines();
    for i in 0..50 {
        let line = format!(
            "{{\"jsonrpc\":\"2.0\",\"id\":{i},\"pad\":\"{}\"}}",
            "y".repeat(i * 1000)
        );
        process.stdin.write_all(line.as_bytes()).await.unwrap();
        process.stdin.write_all(b"\n").await.unwrap();
        process.stdin.flush().await.unwrap();
        assert_eq!(lines.next_line().await.unwrap().unwrap(), line);
        if i == 0 {
            // The first output proves the CLI read its env file: it goes.
            let call = &fake.calls()[0];
            let env_file = PathBuf::from(
                &call.args[call.args.iter().position(|a| a == "--env-file").unwrap() + 1],
            );
            let deadline = std::time::Instant::now() + Duration::from_secs(2);
            while env_file.exists() && std::time::Instant::now() < deadline {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
            assert!(!env_file.exists(), "the env file outlived the first output");
        }
    }
    process.stdin.shutdown().await.unwrap();
    drop(process.stdin);
    assert_eq!(process.handle.wait().await.unwrap(), 0);
    let call = &fake.calls()[0];
    assert_eq!(&call.args[..2], ["exec".to_string(), "-i".to_string()]);
}

#[tokio::test]
async fn killing_a_spawned_process_kills_it_inside_the_container() {
    let fake = Fake::new(
        r#"case "$1" in exec)
  if [ "$2" = "-u" ] && [ "$3" = "0" ] && [ "$7" = "sh" ]; then exit 0; fi
  echo ready; exec sleep 30;;
esac"#,
    );
    let mut process = fake.sandbox().spawn(Exec::new("agent")).await.unwrap();
    // Wait until the fake has recorded the call and started.
    let mut stdout = BufReader::new(process.stdout);
    let mut line = String::new();
    stdout.read_line(&mut line).await.unwrap();
    assert_eq!(line, "ready\n");
    process.handle.kill().await.unwrap();
    let code = process.handle.wait().await.unwrap();
    assert_eq!(code, 128 + 9);
    let calls = fake.calls();
    assert_eq!(calls.len(), 2);
    assert_eq!(
        &calls[1].args[..7],
        strings(&[
            "exec",
            "-u",
            "0",
            "-e",
            &format!("PATH={}", crate::shell::SYSTEM_PATH),
            "bf-test",
            "sh"
        ])
        .as_slice()
    );
}

// --- files ----------------------------------------------------------------

/// Entries of a tar: (path, mode, contents) for files, (path/, mode, "") for folders.
fn tar_entries(bytes: &[u8]) -> Vec<(String, u32, String)> {
    let mut archive = tar::Archive::new(bytes);
    let mut out = Vec::new();
    for entry in archive.entries().unwrap() {
        let mut entry = entry.unwrap();
        let mut path = entry
            .path()
            .unwrap()
            .display()
            .to_string()
            .trim_end_matches('/')
            .to_string();
        if entry.header().entry_type().is_dir() {
            path.push('/');
        }
        let mode = entry.header().mode().unwrap();
        let mut data = String::new();
        std::io::Read::read_to_string(&mut entry, &mut data).unwrap();
        out.push((path, mode, data));
    }
    out
}

#[tokio::test]
async fn upload_pipes_a_tar_into_tar_as_root() {
    let fake = Fake::new(r#"case "$1" in exec) cat > "$dir/calls/$id.stdin";; esac"#);
    let src = tempfile::tempdir().unwrap();
    let set_mode =
        |p: &Path, m: u32| std::fs::set_permissions(p, std::fs::Permissions::from_mode(m)).unwrap();
    std::fs::create_dir(src.path().join("bin")).unwrap();
    set_mode(&src.path().join("bin"), 0o750);
    std::fs::write(src.path().join("bin/run.sh"), "echo run").unwrap();
    set_mode(&src.path().join("bin/run.sh"), 0o755);
    std::fs::write(src.path().join("notes.txt"), "n").unwrap();
    set_mode(&src.path().join("notes.txt"), 0o600);
    std::os::unix::fs::symlink("/etc/passwd", src.path().join("leak")).unwrap();

    fake.sandbox()
        .upload(src.path(), "/app/data/")
        .await
        .unwrap();
    let call = &fake.calls()[0];
    assert_eq!(
        call.args,
        strings(&[
            "exec",
            "-i",
            "-u",
            "0",
            "-e",
            &format!("PATH={}", crate::shell::SYSTEM_PATH),
            "bf-test",
            "sh",
            "-c",
            "mkdir -p /app/data && tar -xf - -C /app/data"
        ])
    );
    let entries = tar_entries(call.stdin.as_deref().unwrap());
    assert_eq!(
        entries,
        vec![
            ("bin/".to_string(), 0o750, String::new()),
            ("bin/run.sh".to_string(), 0o755, "echo run".to_string()),
            ("notes.txt".to_string(), 0o600, "n".to_string()),
        ]
    );
    // Owned by root, so files land root-owned when tar runs as root.
    let mut archive = tar::Archive::new(call.stdin.as_deref().unwrap());
    for entry in archive.entries().unwrap() {
        let entry = entry.unwrap();
        assert_eq!(
            (entry.header().uid().unwrap(), entry.header().gid().unwrap()),
            (0, 0)
        );
    }
}

#[tokio::test]
async fn uploading_a_file_lands_it_at_the_named_path() {
    let fake = Fake::new(r#"case "$1" in exec) cat > "$dir/calls/$id.stdin";; esac"#);
    let src = tempfile::tempdir().unwrap();
    let file = src.path().join("local-name.sh");
    std::fs::write(&file, "x").unwrap();
    fake.sandbox()
        .upload(&file, "/opt/tools/remote-name.sh")
        .await
        .unwrap();
    let call = &fake.calls()[0];
    assert_eq!(
        call.args[9],
        "mkdir -p /opt/tools && tar -xf - -C /opt/tools"
    );
    let entries = tar_entries(call.stdin.as_deref().unwrap());
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].0, "remote-name.sh");
}

#[tokio::test]
async fn write_file_sends_one_entry_with_its_mode() {
    let fake = Fake::new(r#"case "$1" in exec) cat > "$dir/calls/$id.stdin";; esac"#);
    fake.sandbox()
        .write_file("/logs/agent/config.json", b"{\"a\":1}", 0o600)
        .await
        .unwrap();
    let call = &fake.calls()[0];
    assert_eq!(
        call.args[9],
        "mkdir -p /logs/agent && tar -xf - -C /logs/agent"
    );
    assert_eq!(
        tar_entries(call.stdin.as_deref().unwrap()),
        vec![("config.json".to_string(), 0o600, "{\"a\":1}".to_string())]
    );
}

#[tokio::test]
async fn upload_reports_an_image_without_tar() {
    let fake = Fake::new(
        r#"case "$1" in exec) cat >/dev/null; echo 'sh: 1: tar: not found' >&2; exit 127;; esac"#,
    );
    let err = fake
        .sandbox()
        .write_file("/x/y", b"1", 0o644)
        .await
        .unwrap_err();
    assert!(
        matches!(&err, SandboxError::Unsupported(m) if m.contains("needs `tar`")),
        "{err:?}"
    );
}

/// A fake whose `exec -u 0 <c> tar -cf - -C <parent> <name>` archives
/// `$dir/root<parent>/<name>` with the host's tar.
fn download_fake() -> Fake {
    Fake::new(
        r#"case "$1" in exec)
  COPYFILE_DISABLE=1 exec tar -cf - -C "$dir/root${11}" "${12}";;
esac"#,
    )
}

#[tokio::test]
async fn download_unpacks_files_and_folders_with_modes_and_skips_the_rest() {
    let fake = download_fake();
    let root = fake.path("root/srv/out");
    std::fs::create_dir_all(root.join("bin")).unwrap();
    std::fs::write(root.join("a.txt"), "alpha").unwrap();
    std::fs::set_permissions(root.join("a.txt"), std::fs::Permissions::from_mode(0o640)).unwrap();
    std::fs::write(root.join("bin/run.sh"), "#!/bin/sh").unwrap();
    std::fs::set_permissions(
        root.join("bin/run.sh"),
        std::fs::Permissions::from_mode(0o755),
    )
    .unwrap();
    std::os::unix::fs::symlink("/etc/passwd", root.join("link")).unwrap();
    let fifo = std::process::Command::new("mkfifo")
        .arg(root.join("fifo"))
        .status()
        .unwrap();
    assert!(fifo.success());

    let dst = tempfile::tempdir().unwrap();
    let local = dst.path().join("got");
    fake.sandbox().download("/srv/out", &local).await.unwrap();
    assert_eq!(
        fake.calls()[0].args,
        strings(&[
            "exec",
            "-u",
            "0",
            "-e",
            &format!("PATH={}", crate::shell::SYSTEM_PATH),
            "bf-test",
            "tar",
            "-cf",
            "-",
            "-C",
            "/srv",
            "out"
        ])
    );
    assert_eq!(
        std::fs::read_to_string(local.join("a.txt")).unwrap(),
        "alpha"
    );
    let mode = |p: &Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode(&local.join("a.txt")), 0o640);
    assert_eq!(mode(&local.join("bin/run.sh")), 0o755);
    assert!(std::fs::symlink_metadata(local.join("link")).is_err());
    assert!(std::fs::symlink_metadata(local.join("fifo")).is_err());
}

#[tokio::test]
async fn downloading_one_file_writes_it_at_the_local_path() {
    let fake = download_fake();
    std::fs::create_dir_all(fake.path("root/logs/verifier")).unwrap();
    std::fs::write(fake.path("root/logs/verifier/reward.txt"), "1").unwrap();
    let dst = tempfile::tempdir().unwrap();
    let local = dst.path().join("reward.txt");
    fake.sandbox()
        .download("/logs/verifier/reward.txt", &local)
        .await
        .unwrap();
    assert_eq!(std::fs::read_to_string(&local).unwrap(), "1");
}

#[tokio::test]
async fn downloading_a_missing_path_is_not_found() {
    let fake = download_fake();
    std::fs::create_dir_all(fake.path("root/srv")).unwrap();
    let dst = tempfile::tempdir().unwrap();
    let err = fake
        .sandbox()
        .download("/srv/missing", &dst.path().join("x"))
        .await
        .unwrap_err();
    let SandboxError::Io { source, .. } = &err else {
        panic!("wrong error: {err:?}")
    };
    assert_eq!(source.kind(), std::io::ErrorKind::NotFound);
}

// --- destroy and sweep ----------------------------------------------------

#[tokio::test]
async fn destroy_removes_the_container_and_tolerates_it_being_gone() {
    let fake = Fake::new(
        r#"case "$1" in rm) echo 'Error: No such container: bf-test' >&2; exit 1;; esac"#,
    );
    fake.sandbox().destroy().await.unwrap();
    assert_eq!(fake.calls()[0].args, strings(&["rm", "-f", "bf-test"]));
}

#[tokio::test]
async fn sweep_removes_only_containers_of_dead_processes_on_this_host() {
    let mut child = tokio::process::Command::new("true").spawn().unwrap();
    let dead = child.id().unwrap();
    child.wait().await.unwrap();
    let labels = |pid: u32, host: &str| {
        serde_json::json!({
            OWNER_LABEL: owner::scope(),
            crate::HOST_LABEL: host,
            crate::PID_LABEL: pid.to_string(),
        })
        .to_string()
    };
    let me = std::process::id();
    let listing = format!(
        "dead1\t{}\nlive2\t{}\nother3\t{}\n",
        labels(dead, owner::hostname()),
        labels(me, owner::hostname()),
        labels(dead, "another-host"),
    );
    let fake = Fake::new(&format!(
        r#"case "$1" in
  ps) printf 'dead1\nlive2\nother3\n';;
  inspect) printf '%s' '{listing}';;
  rm) shift 2; for a in "$@"; do echo "$a"; done;;
esac"#
    ));
    let removed = fake.provider().sweep().await.unwrap();
    assert_eq!(removed, 1);
    let ps = &fake.calls_to("ps")[0];
    assert_eq!(
        ps.args,
        vec![
            "ps".to_string(),
            "-a".into(),
            "-q".into(),
            "--no-trunc".into(),
            "--filter".into(),
            format!("label={OWNER_LABEL}={}", owner::scope()),
        ]
    );
    assert_eq!(fake.calls_to("rm")[0].args, strings(&["rm", "-f", "dead1"]));
}

#[tokio::test]
async fn a_runtime_is_passed_to_docker_run() {
    let fake =
        Fake::new(r#"case "$1" in run) echo 0123abcd;; inspect) printf 'running\t0\n';; esac"#);
    let mut spec = sandbox_spec();
    spec.runtime = Some("runsc".into());
    fake.provider().create(&spec).await.unwrap();
    let run = &fake.calls_to("run")[0];
    let at = run.args.iter().position(|a| a == "--runtime").unwrap();
    assert_eq!(run.args[at + 1], "runsc");
}

#[tokio::test]
async fn runtimes_come_from_docker_info() {
    let fake = Fake::new(r#"case "$1" in info) printf 'io.containerd.runc.v2\nrunc\n';; esac"#);
    let have = fake.provider().runtimes().await.unwrap().unwrap();
    assert_eq!(have, ["io.containerd.runc.v2", "runc"]);
}

// --- cache listing and removal --------------------------------------------

#[tokio::test]
async fn cached_images_lists_only_owned_images_and_removal_is_never_forced() {
    let fake = Fake::new(
        r#"case "$1 $2" in
"image ls") printf 'sha256:aa\nsha256:bb\nsha256:aa\n';;
"image inspect") printf 'sha256:aa\t1500000000\t2026-10-01T00:00:00Z\tbenchflow/hello:abc,benchflow/hello:abc-def\nsha256:bb\t20\t2026-10-02T00:00:00Z\t<none>:<none>\n';;
esac
case "$1" in
rmi) case "$2" in sha256:bb) echo 'conflict: image is being used by running container 1234' >&2; exit 1;; esac;;
esac"#,
    );
    let p = fake.provider();
    let images = p.cached_images().await.unwrap();
    assert_eq!(images.len(), 2);
    assert_eq!(images[0].bytes, 1_500_000_000);
    assert_eq!(images[0].tags.len(), 2);
    let ls = &fake.calls_to("image")[0];
    let owner_filter = format!("label={OWNER_LABEL}={}", owner::scope());
    assert!(
        ls.args.contains(&owner_filter),
        "only images with BenchFlow's owner label: {:?}",
        ls.args
    );
    // By tag, so another tool's tags on the same image survive.
    assert!(p.remove_image(&images[0]).await.is_ok());
    let err = p.remove_image(&images[1]).await.unwrap_err();
    assert!(err.contains("being used"), "{err}");
    for call in fake.calls_to("rmi") {
        assert!(!call.args.iter().any(|a| a == "-f" || a == "--force"));
    }
    assert_eq!(
        fake.calls_to("rmi")[0].args,
        strings(&["rmi", "benchflow/hello:abc", "benchflow/hello:abc-def"])
    );
}

#[tokio::test]
async fn cached_images_says_when_the_daemon_is_down() {
    let fake = Fake::new(
        r#"echo 'Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?' >&2; exit 1"#,
    );
    let err = fake.provider().cached_images().await.unwrap_err();
    assert!(err.to_string().to_lowercase().contains("daemon"), "{err}");
}
