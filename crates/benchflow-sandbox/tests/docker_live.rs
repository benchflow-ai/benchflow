//! Live tests of the docker backend against a real Docker daemon.
//!
//! They run only with `BENCHFLOW_LIVE_DOCKER=1`, on a Linux host whose
//! `docker` works for the current user (`docker version` shows a server).
//! Set `BENCHFLOW_LIVE_DOCKER_CLI=podman` to run them against podman.
//!
//!     BENCHFLOW_LIVE_DOCKER=1 cargo test -p benchflow-sandbox --test docker_live -- --nocapture
//!
//! Each test destroys its containers even when an assertion fails. Images
//! are pulled from Docker Hub (`ubuntu:24.04`, `alpine:3.20`) and built
//! under `benchflow/live-*` tags, which `docker image prune` style cleanup
//! may remove afterwards. Lines starting with `LATENCY` report timings.

use std::collections::BTreeMap;
use std::os::unix::fs::PermissionsExt;
use std::panic::AssertUnwindSafe;
use std::path::Path;
use std::time::{Duration, Instant};

use benchflow_sandbox::docker::DockerProvider;
use benchflow_sandbox::{
    Exec, Image, ImageSource, ImageSpec, NetworkPolicy, Provider, Sandbox, SandboxError,
    SandboxSpec,
};
use futures::FutureExt;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

fn provider() -> Option<DockerProvider> {
    if std::env::var("BENCHFLOW_LIVE_DOCKER").ok().as_deref() != Some("1") {
        eprintln!("skipped: set BENCHFLOW_LIVE_DOCKER=1 to run the Docker live tests");
        return None;
    }
    let cli = std::env::var("BENCHFLOW_LIVE_DOCKER_CLI").unwrap_or_else(|_| "docker".to_string());
    Some(DockerProvider::new(&cli))
}

fn cli() -> String {
    std::env::var("BENCHFLOW_LIVE_DOCKER_CLI").unwrap_or_else(|_| "docker".to_string())
}

fn print_log(line: &str) {
    eprintln!("  build | {line}");
}

fn pull(reference: &str, name: &str, layers: &[&str]) -> ImageSpec {
    ImageSpec {
        source: ImageSource::Pull {
            reference: reference.to_string(),
        },
        layers: layers.iter().map(|l| l.to_string()).collect(),
        name: name.to_string(),
        digest: format!("live-{name}"),
        build_timeout: Some(Duration::from_secs(600)),
        platform: None,
    }
}

fn spec_for(image: Image, name: &str) -> SandboxSpec {
    SandboxSpec {
        image,
        name: name.to_string(),
        cpus: Some(1.0),
        memory_mb: Some(512),
        disk_mb: None,
        gpus: 0,
        pids_limit: Some(256),
        network: NetworkPolicy::Open,
        env: BTreeMap::from([("TASK_LEVEL".to_string(), "from-spec".to_string())]),
        workdir: Some("/work".to_string()),
        labels: BTreeMap::from([("ai.benchflow.livetest".to_string(), "1".to_string())]),
        runtime: None,
    }
}

/// Create a container, run `body` against it, and remove it even if `body`
/// panics.
macro_rules! with_sandbox {
    ($provider:expr, $spec:expr, |$sb:ident| $body:block) => {{
        let spec: SandboxSpec = $spec;
        let started = Instant::now();
        let sandbox = $provider.create(&spec).await.expect("create the container");
        eprintln!(
            "LATENCY create {}: {:.2}s",
            spec.name,
            started.elapsed().as_secs_f64()
        );
        let result = AssertUnwindSafe(async {
            let $sb: &dyn Sandbox = &*sandbox;
            $body
        })
        .catch_unwind()
        .await;
        let destroyed = sandbox.destroy().await;
        if let Err(panic) = result {
            std::panic::resume_unwind(panic);
        }
        destroyed.expect("remove the container");
    }};
}

async fn ok(sb: &dyn Sandbox, exec: Exec) -> String {
    let shown = exec.command.clone();
    let out = sb
        .exec(exec)
        .await
        .unwrap_or_else(|e| panic!("`{shown}`: {e}"));
    assert!(out.success(), "`{shown}` failed: {}", out.tail(2000));
    out.stdout
}

#[tokio::test]
async fn status_names_the_daemon() {
    let Some(provider) = provider() else { return };
    let status = provider.status().await;
    eprintln!("status: {}", status.summary);
    assert!(status.ok, "{status:?}");

    let missing = DockerProvider::new("docker").with_binary("/nonexistent/docker");
    let status = missing.status().await;
    assert!(!status.ok);
    assert!(status.fix.unwrap().contains("Install"));
}

#[tokio::test]
async fn pulls_once_then_caches_and_layers_are_cached_too() {
    let Some(provider) = provider() else { return };
    let spec = pull(
        "ubuntu:24.04",
        "live-layers",
        &["echo layer-one > /opt/one.txt", "touch /opt/two"],
    );
    let first = provider.prepare(&spec, &print_log).await.expect("prepare");
    assert!(
        first.reference.starts_with("benchflow/live-layers:"),
        "{}",
        first.reference
    );
    let second = provider
        .prepare(&spec, &print_log)
        .await
        .expect("prepare again");
    assert_eq!(second.reference, first.reference);
    assert!(
        !second.fresh,
        "the second prepare reuses both the base and the layers"
    );

    with_sandbox!(provider, spec_for(second, "live-layers"), |sb| {
        assert_eq!(
            ok(
                sb,
                Exec::new("cat /opt/one.txt; test -e /opt/two && echo two")
            )
            .await,
            "layer-one\ntwo\n"
        );
    });
}

#[tokio::test]
async fn builds_a_dockerfile_and_runs_as_its_user() {
    let Some(provider) = provider() else { return };
    let context = tempfile::tempdir().unwrap();
    std::fs::write(
        context.path().join("Dockerfile"),
        "FROM ubuntu:24.04\nCOPY hello.txt /opt/hello.txt\nRUN useradd -m agent\nUSER agent\nWORKDIR /home/agent\n",
    )
    .unwrap();
    std::fs::write(context.path().join("hello.txt"), "hello from the context\n").unwrap();
    let spec = ImageSpec {
        source: ImageSource::Build {
            context: context.path().to_path_buf(),
            dockerfile: context.path().join("Dockerfile"),
        },
        layers: vec!["echo layer-ok > /opt/layer.txt".to_string()],
        name: "live-build".to_string(),
        digest: format!("live-build-{}", std::process::id()),
        build_timeout: Some(Duration::from_secs(600)),
        platform: None,
    };
    let started = Instant::now();
    let image = provider.prepare(&spec, &print_log).await.expect("build");
    eprintln!("LATENCY build: {:.1}s", started.elapsed().as_secs_f64());
    assert!(image.fresh);

    let mut sandbox = spec_for(image, "live-build");
    sandbox.workdir = None;
    with_sandbox!(provider, sandbox, |sb| {
        // No user named: the image's USER and WORKDIR, as `docker exec` gives.
        assert_eq!(
            ok(sb, Exec::new("id -un; pwd")).await,
            "agent\n/home/agent\n"
        );
        assert_eq!(
            ok(sb, Exec::new("cat /opt/hello.txt /opt/layer.txt")).await,
            "hello from the context\nlayer-ok\n"
        );
    });
}

#[tokio::test]
async fn a_failing_dockerfile_step_is_the_tasks_fault() {
    let Some(provider) = provider() else { return };
    let context = tempfile::tempdir().unwrap();
    std::fs::write(
        context.path().join("Dockerfile"),
        "FROM alpine:3.20\nRUN echo about to fail && exit 3\n",
    )
    .unwrap();
    let spec = ImageSpec {
        source: ImageSource::Build {
            context: context.path().to_path_buf(),
            dockerfile: context.path().join("Dockerfile"),
        },
        layers: Vec::new(),
        name: "live-broken".to_string(),
        digest: format!("live-broken-{}", std::process::id()),
        build_timeout: Some(Duration::from_secs(300)),
        platform: None,
    };
    let err = provider.prepare(&spec, &print_log).await.unwrap_err();
    eprintln!("broken build: {err}");
    assert!(
        matches!(
            &err,
            SandboxError::Image {
                task_fault: true,
                ..
            }
        ),
        "{err:?}"
    );

    let missing = pull("benchflow/does-not-exist:never", "live-missing", &[]);
    let err = provider.prepare(&missing, &print_log).await.unwrap_err();
    assert!(
        matches!(
            &err,
            SandboxError::Image {
                task_fault: true,
                ..
            }
        ),
        "{err:?}"
    );
}

#[tokio::test]
async fn exec_users_env_cwd_exit_codes_and_timeouts() {
    let Some(provider) = provider() else { return };
    let image = provider
        .prepare(&pull("ubuntu:24.04", "live-exec", &[]), &print_log)
        .await
        .unwrap();
    with_sandbox!(provider, spec_for(image, "live-exec"), |sb| {
        assert_eq!(
            ok(sb, Exec::new("id -u; echo $TASK_LEVEL; pwd").root()).await,
            "0\nfrom-spec\n/work\n"
        );

        let mut times = Vec::new();
        for _ in 0..5 {
            let t = Instant::now();
            ok(sb, Exec::new("true")).await;
            times.push(t.elapsed());
        }
        times.sort();
        eprintln!(
            "LATENCY exec round trip (median of 5): {:.3}s",
            times[2].as_secs_f64()
        );

        ok(
            sb,
            Exec::new(
                "useradd -m agent && mkdir -p /home/agent/proj && chown agent /home/agent/proj",
            )
            .root(),
        )
        .await;
        let out = ok(
            sb,
            Exec::new("id -un; echo \"$GREETING\"; pwd; echo $HOME")
                .user("agent")
                .env("GREETING", "hello 'quoted' $world")
                .env("MULTI", "line1\nline2")
                .cwd("/home/agent/proj"),
        )
        .await;
        assert_eq!(
            out,
            "agent\nhello 'quoted' $world\n/home/agent/proj\n/home/agent\n"
        );
        assert_eq!(
            ok(
                sb,
                Exec::new("printf %s \"$MULTI\"").env("MULTI", "line1\nline2")
            )
            .await,
            "line1\nline2"
        );

        let out = sb
            .exec(Exec::new("echo out; echo err >&2; exit 3"))
            .await
            .unwrap();
        assert_eq!(
            (out.exit_code, out.stdout.as_str(), out.stderr.as_str()),
            (3, "out\n", "err\n")
        );

        // Environment values never reach the host's process table: the
        // docker client gets them from a private file.
        let out = ok(
            sb,
            Exec::new("cat /proc/$$/environ | tr '\\0' '\\n' | grep SECRET_TOKEN")
                .env("SECRET_TOKEN", "s3cr3t"),
        )
        .await;
        assert_eq!(out, "SECRET_TOKEN=s3cr3t\n");

        // A timeout kills the command and what it started, inside the
        // container, as root or as a user.
        for user in ["root", "agent"] {
            let t = Instant::now();
            let out = sb
                .exec(
                    Exec::new("sleep 300 & sleep 301; echo never")
                        .user(user)
                        .timeout(Duration::from_secs(2)),
                )
                .await
                .unwrap();
            eprintln!(
                "LATENCY timeout of 2s as {user} returned after {:.2}s",
                t.elapsed().as_secs_f64()
            );
            assert!(out.timed_out && out.exit_code == 124, "{out:?}");
            let left = sb
                .exec(Exec::new("pgrep -fa 'sleep 30[01]' || true").root())
                .await
                .unwrap();
            assert_eq!(
                left.stdout.trim(),
                "",
                "processes survived the timeout as {user}"
            );
        }

        // A command that floods its output keeps the last 16 MiB.
        let out = sb
            .exec(Exec::new(
                "head -c 20000000 /dev/zero | tr '\\0' x; echo; echo END",
            ))
            .await
            .unwrap();
        assert!(
            out.stdout.starts_with("[benchflow: "),
            "the note about dropped output comes first"
        );
        assert!(out.stdout.ends_with("END\n"));
    });
}

#[tokio::test]
async fn exec_falls_back_to_sh_without_bash() {
    let Some(provider) = provider() else { return };
    let image = provider
        .prepare(&pull("alpine:3.20", "live-alpine", &[]), &print_log)
        .await
        .unwrap();
    with_sandbox!(provider, spec_for(image, "live-alpine"), |sb| {
        assert_eq!(
            ok(sb, Exec::new("command -v bash || echo no-bash; echo $0")).await,
            "no-bash\nsh\n"
        );
    });
}

#[tokio::test]
async fn spawn_round_trips_lines_and_reports_exit_codes() {
    let Some(provider) = provider() else { return };
    let image = provider
        .prepare(&pull("ubuntu:24.04", "live-spawn", &[]), &print_log)
        .await
        .unwrap();
    with_sandbox!(provider, spec_for(image, "live-spawn"), |sb| {
        let lines: Vec<String> = (0..200usize)
            .map(|i| {
                let size = if i % 10 == 0 {
                    200_000
                } else {
                    (i * 7919) % 200_000
                };
                let body: String = "abcdefghijklmnopqrstuvwxyz é 中 \\\" "
                    .chars()
                    .cycle()
                    .take(size)
                    .collect();
                serde_json::json!({"jsonrpc": "2.0", "id": i, "params": {"text": body}}).to_string()
            })
            .chain(std::iter::once(
                serde_json::json!({"id": "big", "result": "x".repeat(3_000_000)}).to_string(),
            ))
            .collect();
        let mut process = sb.spawn(Exec::new("cat")).await.unwrap();
        let mut stdin = process.stdin;
        let to_send = lines.clone();
        let started = Instant::now();
        let writer = tokio::spawn(async move {
            for line in to_send {
                stdin.write_all(line.as_bytes()).await.unwrap();
                stdin.write_all(b"\n").await.unwrap();
            }
            stdin.flush().await.unwrap();
            stdin
        });
        let mut reader = BufReader::new(process.stdout);
        for (i, expected) in lines.iter().enumerate() {
            let mut got = Vec::new();
            reader.read_until(b'\n', &mut got).await.unwrap();
            assert!(
                got.ends_with(b"\n") && got[..got.len() - 1] == *expected.as_bytes(),
                "line {i} differs"
            );
        }
        eprintln!(
            "LATENCY spawn round trip of 201 lines: {:.2}s",
            started.elapsed().as_secs_f64()
        );
        let mut stdin = writer.await.unwrap();
        stdin.shutdown().await.unwrap();
        drop(stdin);
        assert_eq!(
            process.handle.wait().await.unwrap(),
            0,
            "cat exits 0 at end of input"
        );

        let mut process = sb.spawn(Exec::new("sleep 300")).await.unwrap();
        process.handle.kill().await.unwrap();
        assert_eq!(process.handle.wait().await.unwrap(), 137);
        let left = sb
            // `[s]leep`: the pattern must not match the shell running pgrep.
            .exec(Exec::new("pgrep -fa '[s]leep 300' || true").root())
            .await
            .unwrap();
        assert_eq!(left.stdout.trim(), "", "kill left the process running");

        let mut process = sb
            .spawn(Exec::new(
                "read line; echo \"got:$line\"; echo bye >&2; exit 7",
            ))
            .await
            .unwrap();
        process.stdin.write_all(b"hello\n").await.unwrap();
        process.stdin.flush().await.unwrap();
        let mut line = String::new();
        BufReader::new(process.stdout)
            .read_line(&mut line)
            .await
            .unwrap();
        assert_eq!(line, "got:hello\n");
        let mut err = String::new();
        BufReader::new(process.stderr)
            .read_line(&mut err)
            .await
            .unwrap();
        assert_eq!(err, "bye\n");
        assert_eq!(process.handle.wait().await.unwrap(), 7);
    });
}

#[tokio::test]
async fn files_keep_modes_and_downloads_refuse_links_and_special_files() {
    let Some(provider) = provider() else { return };
    let image = provider
        .prepare(&pull("ubuntu:24.04", "live-files", &[]), &print_log)
        .await
        .unwrap();
    with_sandbox!(provider, spec_for(image, "live-files"), |sb| {
        sb.write_file("/etc/bf/config.json", b"{\"a\":1}\n", 0o640)
            .await
            .unwrap();
        assert_eq!(
            ok(
                sb,
                Exec::new("stat -c '%a %U' /etc/bf/config.json; cat /etc/bf/config.json")
            )
            .await,
            "640 root\n{\"a\":1}\n"
        );

        let local = tempfile::tempdir().unwrap();
        let src = local.path().join("src");
        std::fs::create_dir_all(src.join("sub")).unwrap();
        std::fs::write(src.join("run.sh"), "#!/bin/sh\n").unwrap();
        std::fs::set_permissions(src.join("run.sh"), std::fs::Permissions::from_mode(0o755))
            .unwrap();
        std::fs::write(src.join("secret.txt"), "s").unwrap();
        std::fs::set_permissions(
            src.join("secret.txt"),
            std::fs::Permissions::from_mode(0o600),
        )
        .unwrap();
        let blob: Vec<u8> = (0..1_048_576u32)
            .map(|i| (i.wrapping_mul(2654435761) >> 24) as u8)
            .collect();
        std::fs::write(src.join("sub/one-mb.bin"), &blob).unwrap();
        // Set the mode: the host's umask (002 on many Linux hosts) decides it otherwise.
        std::fs::set_permissions(
            src.join("sub/one-mb.bin"),
            std::fs::Permissions::from_mode(0o644),
        )
        .unwrap();
        std::os::unix::fs::symlink("/etc/passwd", src.join("link")).unwrap();
        let t = Instant::now();
        sb.upload(&src, "/srv/data").await.unwrap();
        eprintln!(
            "LATENCY upload of a 1 MB folder: {:.3}s",
            t.elapsed().as_secs_f64()
        );
        assert_eq!(
            ok(sb, Exec::new("cd /srv/data && stat -c '%a %U %n' run.sh secret.txt sub/one-mb.bin && test ! -e link && echo no-link")).await,
            "755 root run.sh\n600 root secret.txt\n644 root sub/one-mb.bin\nno-link\n"
        );

        // Plant what an agent might leave for the output collector.
        ok(
            sb,
            Exec::new("cd /srv/data && ln -s /etc/shadow shadow-link && mkfifo pipe").root(),
        )
        .await;
        let back = local.path().join("back");
        sb.download("/srv/data", &back).await.unwrap();
        assert_eq!(std::fs::read(back.join("sub/one-mb.bin")).unwrap(), blob);
        let mode = |p: &Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode(&back.join("run.sh")), 0o755);
        assert_eq!(mode(&back.join("secret.txt")), 0o600);
        assert!(
            std::fs::symlink_metadata(back.join("shadow-link")).is_err(),
            "symlinks are refused"
        );
        assert!(
            std::fs::symlink_metadata(back.join("pipe")).is_err(),
            "FIFOs are refused"
        );

        let file = local.path().join("config.json");
        sb.download("/etc/bf/config.json", &file).await.unwrap();
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "{\"a\":1}\n");
        let err = sb
            .download("/no/such/path", &local.path().join("x"))
            .await
            .unwrap_err();
        assert!(
            matches!(&err, SandboxError::Io { source, .. } if source.kind() == std::io::ErrorKind::NotFound),
            "{err:?}"
        );
    });
}

#[tokio::test]
async fn limits_and_network_none_hold() {
    let Some(provider) = provider() else { return };
    let image = provider
        .prepare(&pull("alpine:3.20", "live-limits", &[]), &print_log)
        .await
        .unwrap();
    let mut spec = spec_for(image, "live-limits");
    spec.network = NetworkPolicy::None;
    spec.pids_limit = Some(64);
    with_sandbox!(provider, spec, |sb| {
        let out = sb
            .exec(
                Exec::new("wget -q -T 5 -O- http://1.1.1.1 >/dev/null")
                    .timeout(Duration::from_secs(30)),
            )
            .await
            .unwrap();
        assert_ne!(
            out.exit_code, 0,
            "a no-network container reached the internet"
        );
        let out = ok(
            sb,
            Exec::new(
                "cat /sys/fs/cgroup/pids.max 2>/dev/null || cat /sys/fs/cgroup/pids/pids.max",
            ),
        )
        .await;
        assert_eq!(out.trim(), "64");
        let out = ok(sb, Exec::new("cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes")).await;
        assert_eq!(out.trim(), (512u64 * 1024 * 1024).to_string());
    });
}

#[tokio::test]
async fn sweep_removes_containers_of_dead_processes_only() {
    let Some(provider) = provider() else { return };
    let mut child = std::process::Command::new("true").spawn().unwrap();
    let dead = child.id();
    child.wait().unwrap();
    let run = |pid: u32, name: &str| {
        let status = std::process::Command::new(cli())
            .args([
                "run",
                "-d",
                "--name",
                name,
                "--label",
                &format!("ai.benchflow.owner={}", benchflow_sandbox::owner::scope()),
                "--label",
                &format!("ai.benchflow.host={}", benchflow_sandbox::owner::hostname()),
                "--label",
                &format!("ai.benchflow.pid={pid}"),
                "alpine:3.20",
                "sleep",
                "300",
            ])
            .status()
            .unwrap();
        assert!(status.success());
    };
    let suffix = std::process::id();
    let dead_name = format!("bf-live-dead-{suffix}");
    let live_name = format!("bf-live-alive-{suffix}");
    run(dead, &dead_name);
    run(std::process::id(), &live_name);

    let removed = provider.sweep().await.unwrap();
    let exists = |name: &str| {
        std::process::Command::new(cli())
            .args(["inspect", name])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    };
    let dead_left = exists(&dead_name);
    let live_left = exists(&live_name);
    let _ = std::process::Command::new(cli())
        .args(["rm", "-f", &dead_name, &live_name])
        .output();
    assert!(
        removed >= 1 && !dead_left,
        "the dead process's container was not swept"
    );
    assert!(live_left, "the sweep removed a live process's container");
}

/// The image cache reads BenchFlow's images through `docker image inspect`
/// templates; a template one Docker version accepts can fail on another
/// (Docker 29 refused `join .RepoTags`, so nothing was ever listed or
/// trimmed). Lists a freshly built image with its tag, then removes it.
#[tokio::test]
async fn cached_images_lists_a_built_image_with_its_tag() {
    let Some(provider) = provider() else { return };
    let context = tempfile::tempdir().unwrap();
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::fs::write(
        context.path().join("Dockerfile"),
        format!("FROM ubuntu:24.04\nRUN echo {stamp} > /stamp\n"),
    )
    .unwrap();
    let spec = ImageSpec {
        source: ImageSource::Build {
            context: context.path().to_path_buf(),
            dockerfile: context.path().join("Dockerfile"),
        },
        layers: Vec::new(),
        name: "live-cache-list".to_string(),
        digest: format!("live-cache-{stamp}"),
        build_timeout: Some(Duration::from_secs(600)),
        platform: None,
    };
    let image = provider.prepare(&spec, &print_log).await.expect("build");
    let images = provider
        .cached_images()
        .await
        .expect("list BenchFlow's images");
    let found = images
        .iter()
        .find(|i| i.tags.contains(&image.reference))
        .unwrap_or_else(|| panic!("{} is not listed: {images:?}", image.reference))
        .clone();
    assert!(found.bytes > 0, "{found:?}");
    provider.remove_image(&found).await.expect("remove it");
    let after = provider.cached_images().await.expect("list again");
    assert!(!after.iter().any(|i| i.id == found.id), "{after:?}");
}
