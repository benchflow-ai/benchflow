//! Live tests of the Daytona backend against the real Daytona API.
//!
//! They run only with `BENCHFLOW_LIVE_DAYTONA=1`. The key comes from
//! `DAYTONA_API_KEY`, or, when that is unset, from
//! `~/.config/benchflow/daytona.env` (a `DAYTONA_API_KEY=...` line). It is
//! never printed.
//!
//! Cost control: at most three sandboxes exist at once (a shared semaphore,
//! also taken while `prepare` runs its builder sandbox), every sandbox is
//! destroyed even when a check fails, and a guard deletes any sandbox a test
//! left behind. Sandboxes also stop after 15 idle minutes and are then
//! deleted by Daytona. Lines starting with `LATENCY` report timings.
//!
//!     BENCHFLOW_LIVE_DAYTONA=1 cargo test -p benchflow-sandbox --test daytona_live -- --nocapture

use std::collections::BTreeMap;
use std::os::unix::fs::PermissionsExt;
use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use benchflow_sandbox::daytona::{DaytonaConfig, DaytonaProvider};
use benchflow_sandbox::{
    Exec, Image, ImageSource, ImageSpec, NetworkPolicy, Provider, Sandbox, SandboxError,
    SandboxSpec,
};
use futures::FutureExt;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::Semaphore;

/// At most this many sandboxes exist at once across the tests of this file.
static SLOTS: Semaphore = Semaphore::const_new(3);

fn config() -> Option<DaytonaConfig> {
    if std::env::var("BENCHFLOW_LIVE_DAYTONA").ok().as_deref() != Some("1") {
        eprintln!("skipped: set BENCHFLOW_LIVE_DAYTONA=1 to run the Daytona live tests");
        return None;
    }
    let mut config = DaytonaConfig::from_env();
    if config.api_key.is_none() {
        let home = std::env::var("HOME").expect("HOME is set");
        let path = Path::new(&home).join(".config/benchflow/daytona.env");
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!(
                "DAYTONA_API_KEY is unset and {} is unreadable: {e}",
                path.display()
            )
        });
        let key = text
            .lines()
            .find_map(|l| l.trim().strip_prefix("DAYTONA_API_KEY="))
            .map(|v| v.trim().trim_matches(['"', '\'']).to_string())
            .unwrap_or_else(|| panic!("{} has no DAYTONA_API_KEY line", path.display()));
        config.api_key = Some(key);
    }
    Some(config)
}

/// Deletes, when dropped, every sandbox still tracked: those a failing test
/// could not destroy. Runs on its own thread and runtime, so it works while
/// a test unwinds.
struct LeakGuard {
    config: DaytonaConfig,
    ids: Arc<Mutex<Vec<String>>>,
}

impl LeakGuard {
    fn new(config: &DaytonaConfig) -> Self {
        Self {
            config: config.clone(),
            ids: Arc::default(),
        }
    }

    fn track(&self, id: &str) {
        self.ids.lock().unwrap().push(id.to_string());
    }

    fn untrack(&self, id: &str) {
        self.ids.lock().unwrap().retain(|i| i != id);
    }
}

impl Drop for LeakGuard {
    fn drop(&mut self) {
        let ids = std::mem::take(&mut *self.ids.lock().unwrap());
        if ids.is_empty() {
            return;
        }
        let config = self.config.clone();
        let _ = std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            runtime.block_on(async {
                let provider = DaytonaProvider::new(config).unwrap();
                for id in ids {
                    match provider.delete_sandbox(&id).await {
                        Ok(()) => eprintln!("leak guard: deleted sandbox {id}"),
                        Err(e) => eprintln!("leak guard: could not delete sandbox {id}: {e}"),
                    }
                }
            });
        })
        .join();
    }
}

fn spec_for(image: Image, name: &str, network: NetworkPolicy) -> SandboxSpec {
    SandboxSpec {
        image,
        name: name.to_string(),
        cpus: Some(1.0),
        memory_mb: Some(1024),
        disk_mb: Some(3 * 1024),
        gpus: 0,
        pids_limit: None,
        network,
        env: BTreeMap::from([("TASK_LEVEL".to_string(), "from-spec".to_string())]),
        workdir: Some("/work".to_string()),
        labels: BTreeMap::from([("ai.benchflow.livetest".to_string(), "1".to_string())]),
        runtime: None,
        disk_limit_from_parent: None,
        base_image: None,
    }
}

fn pull(reference: &str, name: &str) -> ImageSpec {
    ImageSpec {
        source: ImageSource::Pull {
            reference: reference.to_string(),
        },
        layers: Vec::new(),
        name: name.to_string(),
        digest: "livetest".to_string(),
        build_timeout: Some(Duration::from_secs(600)),
        platform: None,
    }
}

fn print_log(line: &str) {
    eprintln!("  build | {line}");
}

/// Prepare under a slot (`prepare` runs a builder sandbox).
async fn prepare(provider: &DaytonaProvider, spec: &ImageSpec) -> Result<Image, SandboxError> {
    let _slot = SLOTS.acquire().await.unwrap();
    let started = Instant::now();
    let image = provider.prepare(spec, &print_log).await;
    eprintln!(
        "LATENCY prepare {}: {:.1}s",
        spec.name,
        started.elapsed().as_secs_f64()
    );
    image
}

/// Create a sandbox, run `body` against it, and destroy it even if `body`
/// panics.
macro_rules! with_sandbox {
    ($provider:expr, $guard:expr, $spec:expr, |$sb:ident| $body:block) => {{
        let _slot = SLOTS.acquire().await.unwrap();
        let spec: SandboxSpec = $spec;
        let started = Instant::now();
        let sandbox = $provider.create(&spec).await.expect("create the sandbox");
        eprintln!(
            "LATENCY create {} ({}): {:.1}s",
            spec.name,
            spec.image.reference,
            started.elapsed().as_secs_f64()
        );
        $guard.track(sandbox.id());
        let result = AssertUnwindSafe(async {
            let $sb: &dyn Sandbox = &*sandbox;
            $body
        })
        .catch_unwind()
        .await;
        let destroyed = sandbox.destroy().await;
        if destroyed.is_ok() {
            $guard.untrack(sandbox.id());
        }
        if let Err(panic) = result {
            std::panic::resume_unwind(panic);
        }
        destroyed.expect("destroy the sandbox");
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
async fn status_reports_a_signed_in_account() {
    let Some(config) = config() else { return };
    let provider = DaytonaProvider::new(config).unwrap();
    let status = provider.status().await;
    eprintln!("status: {}", status.summary);
    assert!(status.ok, "{status:?}");
    assert!(
        status.summary.starts_with("Daytona: signed in, "),
        "{}",
        status.summary
    );

    let bad = DaytonaProvider::new(DaytonaConfig {
        api_key: Some("dtn_not_a_real_key_0000000000".to_string()),
        ..DaytonaConfig::from_env()
    })
    .unwrap();
    let status = bad.status().await;
    assert!(!status.ok);
    let fix = status.fix.unwrap_or_default();
    assert!(
        fix.contains("https://app.daytona.io/dashboard/keys"),
        "{fix}"
    );
    assert!(
        !status.summary.contains("dtn_not_a_real_key"),
        "{}",
        status.summary
    );
}

#[tokio::test]
async fn ubuntu_exec_users_env_cwd_timeouts_and_files() {
    let Some(config) = config() else { return };
    let guard = LeakGuard::new(&config);
    let provider = DaytonaProvider::new(config).unwrap();
    let image = prepare(&provider, &pull("ubuntu:24.04", "live-ubuntu"))
        .await
        .expect("prepare");
    assert_eq!(image.reference, "ubuntu:24.04");
    let spec = spec_for(image, "live-exec", NetworkPolicy::Open);

    with_sandbox!(provider, guard, spec, |sb| {
        assert_eq!(sb.backend(), "daytona");

        // Root, the task's env, and the workdir by default.
        let out = ok(sb, Exec::new("id -u; echo $TASK_LEVEL; pwd").root()).await;
        assert_eq!(out, "0\nfrom-spec\n/work\n");

        // Exec latency: five trivial commands.
        let mut times = Vec::new();
        for _ in 0..5 {
            let t = Instant::now();
            ok(sb, Exec::new("true")).await;
            times.push(t.elapsed());
        }
        times.sort();
        eprintln!(
            "LATENCY exec round trip (median of 5): {:.2}s",
            times[2].as_secs_f64()
        );

        // A user made with useradd: user, env and cwd as Docker gives them.
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
            Exec::new("id -un; echo \"$GREETING\"; pwd; echo $HOME; touch mine && stat -c %U mine")
                .user("agent")
                .env("GREETING", "hello 'quoted' $world")
                .cwd("/home/agent/proj"),
        )
        .await;
        assert_eq!(
            out,
            "agent\nhello 'quoted' $world\n/home/agent/proj\n/home/agent\nagent\n"
        );

        // Exit codes and stderr come back apart.
        let out = sb
            .exec(Exec::new("echo out; echo err >&2; exit 3"))
            .await
            .unwrap();
        assert_eq!(
            (out.exit_code, out.stdout.as_str(), out.stderr.as_str()),
            (3, "out\n", "err\n")
        );

        // The command's environment never shows on its command line.
        let out = ok(
            sb,
            Exec::new("cat /proc/$$/cmdline | tr '\\0' ' '").env("SECRET_TOKEN", "s3cr3t-value"),
        )
        .await;
        assert!(!out.contains("s3cr3t-value"), "{out}");

        // Open network reaches the internet (bash's /dev/tcp; no curl here).
        ok(
            sb,
            Exec::new("timeout 10 bash -c 'exec 3<>/dev/tcp/1.1.1.1/443' && echo open"),
        )
        .await;

        // A timeout kills the command inside the sandbox, as root or as a user.
        for user in ["root", "agent"] {
            let t = Instant::now();
            let out = sb
                .exec(
                    Exec::new("sleep 30; echo never")
                        .user(user)
                        .timeout(Duration::from_secs(2)),
                )
                .await
                .unwrap();
            eprintln!(
                "LATENCY timeout of 2s as {user} returned after {:.1}s",
                t.elapsed().as_secs_f64()
            );
            assert!(out.timed_out, "{out:?}");
            assert_eq!(out.exit_code, 124);
            assert!(!out.stdout.contains("never"));
            // (The daemon keeps its own `sleep infinity`; look for ours.)
            let left = sb
                .exec(Exec::new("pgrep -fx 'sleep 30' || true; ps -eo pid,user,args | grep -v grep | grep 'sleep 30' || true").root())
                .await
                .unwrap();
            assert_eq!(
                left.stdout.trim(),
                "",
                "a sleep survived the timeout as {user}"
            );
        }

        // A command that leaves a child holding its stdout (a service
        // started with `&`) returns when it exits, with its output and code.
        // Daytona reports an exit code only once stdout closes, and Ubuntu's
        // dash hands a saved copy of stderr to children when a brace group
        // redirects it; either one kept the process-flood probe running for
        // its whole 2-minute timeout.
        for user in ["root", "agent"] {
            let t = Instant::now();
            let out = sb
                .exec(
                    Exec::new("(sleep 300 &); echo out; echo err >&2; exit 3")
                        .user(user)
                        .timeout(Duration::from_secs(60)),
                )
                .await
                .unwrap();
            eprintln!(
                "LATENCY a command leaving a background child as {user} returned after {:.1}s",
                t.elapsed().as_secs_f64()
            );
            assert!(!out.timed_out, "{out:?}");
            assert_eq!(
                (out.exit_code, out.stdout.as_str(), out.stderr.as_str()),
                (3, "out\n", "err\n")
            );
        }
        // A timed-out command still gives back what it printed.
        let out = sb
            .exec(Exec::new("echo partial; sleep 31").timeout(Duration::from_secs(2)))
            .await
            .unwrap();
        assert!(out.timed_out, "{out:?}");
        assert_eq!(out.stdout, "partial\n", "{out:?}");

        // write_file: parents created, mode kept, owned by root.
        sb.write_file("/etc/bf/config.json", b"{\"a\":1}\n", 0o640)
            .await
            .unwrap();
        let out = ok(
            sb,
            Exec::new("stat -c '%a %U' /etc/bf/config.json; cat /etc/bf/config.json").root(),
        )
        .await;
        assert_eq!(out, "640 root\n{\"a\":1}\n");

        // A folder up and back down, with modes; symlinks never travel.
        let local = tempfile::tempdir().unwrap();
        let src = local.path().join("src");
        std::fs::create_dir_all(src.join("sub")).unwrap();
        std::fs::write(src.join("run.sh"), "#!/bin/sh\necho run\n").unwrap();
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
        std::os::unix::fs::symlink("/etc/passwd", src.join("link")).unwrap();
        sb.upload(&src, "/srv/data").await.unwrap();
        let out = ok(
            sb,
            Exec::new("cd /srv/data && stat -c '%a %U %n' run.sh secret.txt sub/one-mb.bin && test ! -e link && echo no-link").root(),
        )
        .await;
        assert_eq!(
            out,
            "755 root run.sh\n600 root secret.txt\n644 root sub/one-mb.bin\nno-link\n"
        );

        let back = local.path().join("back");
        sb.download("/srv/data", &back).await.unwrap();
        assert_eq!(std::fs::read(back.join("sub/one-mb.bin")).unwrap(), blob);
        let mode = |p: PathBuf| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode(back.join("run.sh")), 0o755);
        assert_eq!(mode(back.join("secret.txt")), 0o600);

        // One file down to a file path; a missing path is NotFound.
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

        // Upload latency for one 1 MB file.
        let one = local.path().join("one.bin");
        std::fs::write(&one, &blob).unwrap();
        let t = Instant::now();
        sb.upload(&one, "/tmp/one.bin").await.unwrap();
        eprintln!("LATENCY upload of 1 MB: {:.2}s", t.elapsed().as_secs_f64());
        let out = ok(sb, Exec::new("stat -c %s /tmp/one.bin")).await;
        assert_eq!(out.trim(), "1048576");
    });
}

#[tokio::test]
async fn spawn_round_trips_200_json_lines_and_reports_exit_codes() {
    let Some(config) = config() else { return };
    let guard = LeakGuard::new(&config);
    let provider = DaytonaProvider::new(config).unwrap();
    let image = prepare(&provider, &pull("ubuntu:24.04", "live-ubuntu"))
        .await
        .expect("prepare");
    let spec = spec_for(image, "live-spawn", NetworkPolicy::Open);

    with_sandbox!(provider, guard, spec, |sb| {
        // 200 newline-delimited JSON lines of up to 200 KB, through `cat`.
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
            .collect();
        let total: usize = lines.iter().map(|l| l.len() + 1).sum();
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
            let n =
                tokio::time::timeout(Duration::from_secs(300), reader.read_until(b'\n', &mut got))
                    .await
                    .unwrap_or_else(|_| panic!("line {i} did not come back within 300s"))
                    .unwrap();
            assert!(n > 0, "stdout ended before line {i}");
            assert_eq!(got.last(), Some(&b'\n'));
            let got = &got[..got.len() - 1];
            if got != expected.as_bytes() {
                let at = got
                    .iter()
                    .zip(expected.as_bytes())
                    .position(|(a, b)| a != b)
                    .unwrap_or(got.len().min(expected.len()));
                let around = |b: &[u8]| {
                    String::from_utf8_lossy(&b[at.saturating_sub(40)..(at + 40).min(b.len())])
                        .into_owned()
                };
                panic!(
                    "line {i} differs: got {} bytes, expected {}; first difference at byte {at}: got {:?}, expected {:?}",
                    got.len(),
                    expected.len(),
                    around(got),
                    around(expected.as_bytes())
                );
            }
        }
        let elapsed = started.elapsed().as_secs_f64();
        eprintln!(
            "LATENCY spawn round trip of 200 lines ({:.1} MB each way): {elapsed:.1}s ({:.2} MB/s)",
            total as f64 / 1e6,
            total as f64 / 1e6 / elapsed
        );
        // One multi-megabyte line (an ACP tool result can be this big).
        let mut stdin = writer.await.unwrap();
        let big =
            serde_json::json!({"jsonrpc": "2.0", "id": "big", "result": "x".repeat(3_000_000)})
                .to_string();
        let t = Instant::now();
        stdin.write_all(big.as_bytes()).await.unwrap();
        stdin.write_all(b"\n").await.unwrap();
        stdin.flush().await.unwrap();
        let mut got = Vec::new();
        tokio::time::timeout(Duration::from_secs(300), reader.read_until(b'\n', &mut got))
            .await
            .expect("the 3 MB line came back")
            .unwrap();
        assert_eq!(got.len(), big.len() + 1, "the 3 MB line came back whole");
        assert!(got[..big.len()] == *big.as_bytes());
        eprintln!(
            "LATENCY spawn round trip of one 3 MB line: {:.1}s",
            t.elapsed().as_secs_f64()
        );
        process.handle.kill().await.unwrap();
        let code = process.handle.wait().await.unwrap();
        assert_eq!(code, 137, "a killed process reports SIGKILL");
        let left = sb
            .exec(Exec::new("pgrep -x cat || true").root())
            .await
            .unwrap();
        assert_eq!(left.stdout.trim(), "", "cat survived kill");

        // A process that ends on its own: wait returns its code.
        let mut process = sb
            .spawn(Exec::new(
                "read line; echo \"got:$line\"; echo bye >&2; exit 7",
            ))
            .await
            .unwrap();
        process.stdin.write_all(b"hello\n").await.unwrap();
        process.stdin.flush().await.unwrap();
        let mut stdout = BufReader::new(process.stdout);
        let mut line = String::new();
        stdout.read_line(&mut line).await.unwrap();
        assert_eq!(line, "got:hello\n");
        let mut stderr = BufReader::new(process.stderr);
        let mut err_line = String::new();
        stderr.read_line(&mut err_line).await.unwrap();
        assert_eq!(err_line, "bye\n");
        let t = Instant::now();
        assert_eq!(process.handle.wait().await.unwrap(), 7);
        eprintln!(
            "LATENCY wait for a process that already printed its last line: {:.2}s",
            t.elapsed().as_secs_f64()
        );
    });
}

#[tokio::test]
async fn dockerfile_build_with_a_layer_runs_as_its_user_without_network() {
    let Some(config) = config() else { return };
    let guard = LeakGuard::new(&config);
    let provider = DaytonaProvider::new(config).unwrap();

    let context = tempfile::tempdir().unwrap();
    std::fs::write(
        context.path().join("Dockerfile"),
        "FROM ubuntu:24.04\n\
         RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \\\n\
         \x20   && rm -rf /var/lib/apt/lists/*\n\
         COPY hello.txt /opt/hello.txt\n\
         ADD notes/ /opt/notes/\n\
         COPY . /opt/ctx/\n\
         RUN useradd -m agent\n\
         USER agent\n\
         WORKDIR /home/agent\n",
    )
    .unwrap();
    std::fs::write(context.path().join("hello.txt"), "hello from the context\n").unwrap();
    std::fs::create_dir(context.path().join("notes")).unwrap();
    std::fs::write(context.path().join("notes/a.md"), "note\n").unwrap();
    std::fs::write(context.path().join("ignored.bin"), "should not be sent").unwrap();
    std::fs::write(context.path().join(".dockerignore"), "ignored.bin\n").unwrap();
    let spec = ImageSpec {
        source: ImageSource::Build {
            context: context.path().to_path_buf(),
            dockerfile: context.path().join("Dockerfile"),
        },
        layers: vec!["echo layer-ok > /opt/layer.txt".to_string()],
        name: "live-build".to_string(),
        digest: "livetest-build".to_string(),
        build_timeout: Some(Duration::from_secs(900)),
        platform: None,
    };
    let image = prepare(&provider, &spec)
        .await
        .expect("prepare the Dockerfile build");
    assert!(
        image.reference.starts_with("bf-build:live-build:"),
        "{}",
        image.reference
    );
    // Prepared again in this process: no second build.
    let again = provider.prepare(&spec, &print_log).await.unwrap();
    assert!(!again.fresh);
    assert_eq!(again.reference, image.reference);

    let mut sandbox_spec = spec_for(image, "live-build", NetworkPolicy::None);
    sandbox_spec.workdir = None;
    with_sandbox!(provider, guard, sandbox_spec, |sb| {
        // No user named: the Dockerfile's final USER, in its WORKDIR.
        let out = ok(sb, Exec::new("id -un; pwd")).await;
        assert_eq!(out, "agent\n/home/agent\n");
        let out = ok(
            sb,
            Exec::new("cat /opt/hello.txt /opt/layer.txt /opt/notes/a.md").root(),
        )
        .await;
        assert_eq!(out, "hello from the context\nlayer-ok\nnote\n");
        // `.dockerignore`d files never reach Daytona's build context.
        ok(
            sb,
            Exec::new("test -e /opt/ctx/hello.txt && test ! -e /opt/ctx/ignored.bin"),
        )
        .await;
        // Network none: curl cannot reach anything.
        let out = sb
            .exec(
                Exec::new("curl -sS -m 10 https://1.1.1.1 >/dev/null")
                    .timeout(Duration::from_secs(60)),
            )
            .await
            .unwrap();
        eprintln!(
            "network none: curl exit {} ({})",
            out.exit_code,
            out.stderr.trim()
        );
        assert_ne!(
            out.exit_code, 0,
            "curl reached the internet from a no-network sandbox"
        );
    });
}

#[tokio::test]
async fn a_failing_dockerfile_is_the_tasks_fault() {
    let Some(config) = config() else { return };
    let provider = DaytonaProvider::new(config).unwrap();
    let context = tempfile::tempdir().unwrap();
    std::fs::write(
        context.path().join("Dockerfile"),
        "FROM ubuntu:24.04\nRUN echo about to fail && exit 3\n",
    )
    .unwrap();
    let spec = ImageSpec {
        source: ImageSource::Build {
            context: context.path().to_path_buf(),
            dockerfile: context.path().join("Dockerfile"),
        },
        layers: Vec::new(),
        name: "live-broken".to_string(),
        digest: "livetest-broken".to_string(),
        build_timeout: Some(Duration::from_secs(600)),
        platform: None,
    };
    let err = prepare(&provider, &spec).await.unwrap_err();
    eprintln!("broken build: {err}");
    let SandboxError::Image {
        task_fault,
        log_tail,
        ..
    } = &err
    else {
        panic!("expected an image error, got {err:?}")
    };
    assert!(*task_fault, "a failing RUN is the task's fault: {log_tail}");
    assert!(!err.is_transient());
}

const LEAK_HELPER: &str = "BENCHFLOW_LIVE_DAYTONA_LEAK_HELPER";

#[tokio::test]
async fn sweep_deletes_what_a_crashed_process_left() {
    if std::env::var(LEAK_HELPER).is_ok() {
        return; // this is the helper's own process
    }
    let Some(config) = config() else { return };
    let guard = LeakGuard::new(&config);
    let provider = DaytonaProvider::new(config).unwrap();
    // The child makes a builder sandbox, then a trial one: hold two slots.
    let _slots = SLOTS.acquire_many(2).await.unwrap();

    // A child process makes a sandbox and exits without destroying it, as
    // a crashed run would.
    let child = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "leak_helper_creates_a_sandbox_and_exits",
            "--nocapture",
        ])
        .env(LEAK_HELPER, "1")
        .env("BENCHFLOW_LIVE_DAYTONA", "1")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&child.stdout);
    let leaked = stdout
        .lines()
        .find_map(|l| l.strip_prefix("LEAKED "))
        .unwrap_or_else(|| {
            panic!(
                "the helper printed no sandbox id: {stdout} {}",
                String::from_utf8_lossy(&child.stderr)
            )
        })
        .to_string();
    guard.track(&leaked);

    // This process is alive, so the sweep keeps the sandboxes of these
    // tests; the dead child's is a leftover.
    let removed = provider.sweep().await.unwrap();
    eprintln!("sweep removed {removed} leftover sandbox(es), including {leaked}");
    assert!(
        removed >= 1,
        "the crashed process's sandbox {leaked} was not swept"
    );
}

/// Run only by `sweep_deletes_what_a_crashed_process_left`, in a child.
#[tokio::test]
async fn leak_helper_creates_a_sandbox_and_exits() {
    if std::env::var(LEAK_HELPER).is_err() {
        return;
    }
    let Some(config) = config() else { return };
    let provider = DaytonaProvider::new(config).unwrap();
    let image = provider
        .prepare(&pull("ubuntu:24.04", "live-leak"), &|_: &str| {})
        .await
        .unwrap();
    let sandbox = provider
        .create(&spec_for(image, "live-leak", NetworkPolicy::Open))
        .await
        .unwrap();
    println!("LEAKED {}", sandbox.id());
    std::process::exit(0);
}
