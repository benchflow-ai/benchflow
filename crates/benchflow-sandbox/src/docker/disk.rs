//! A task's disk limit on Docker and Podman.
//!
//! A daemon caps a container's writable layer only with `--storage-opt
//! size=` on a storage driver that supports it (overlay2 over xfs with
//! project quotas, zfs; btrfs takes it but its quota does not show inside).
//! Elsewhere (overlay2 over ext4, Docker Desktop) the run is refused. So:
//! ask for the cap; when the run fails, start without it and remember a
//! daemon that refused, and record the limit as not enforced. A cap the
//! daemon took counts only when the sandbox's own filesystem shows it.
//!
//! The layer is measured as the daemon sees it: `inspect --size` for the
//! bytes, `diff` for the number of entries it added. The stop is `pause`
//! then `kill` of the whole container, from outside it. The daemon's own
//! disk is read by a throwaway probe of the task's image ([`host_df`]).

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use tokio::io::AsyncReadExt as _;

use super::cli::Cli;
use crate::SandboxError;

/// Whether each daemon took `--storage-opt size=`, once tried.
fn known() -> &'static Mutex<BTreeMap<String, bool>> {
    static KNOWN: OnceLock<Mutex<BTreeMap<String, bool>>> = OnceLock::new();
    KNOWN.get_or_init(Default::default)
}

/// The daemon `cli` talks to.
fn daemon(cli: &Cli) -> String {
    format!(
        "{}|{}|{}",
        cli.binary.display(),
        std::env::var("DOCKER_HOST").unwrap_or_default(),
        std::env::var("DOCKER_CONTEXT").unwrap_or_default()
    )
}

/// Whether to ask this daemon for a size cap: unless it refused one before.
pub(super) fn try_cap(cli: &Cli) -> bool {
    known()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&daemon(cli))
        .copied()
        != Some(false)
}

/// Remember whether this daemon takes a size cap.
pub(super) fn learned(cli: &Cli, takes_cap: bool) {
    known()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(daemon(cli), takes_cap);
}

/// The daemon's refusal of `--storage-opt size=` (as opposed to any other
/// reason a run fails), in Docker's and Podman's words. Only decides
/// whether the daemon is remembered as one that cannot cap: any failed
/// capped run is tried once more without the cap.
pub(super) fn refused_cap(message: &str) -> bool {
    let m = message.to_ascii_lowercase();
    m.contains("storage-opt")
        || m.contains("storage opt")
        || m.contains("storageopt")
        || m.contains("pquota")
        || (m.contains("size") && m.contains("not supported"))
}

/// The `--storage-opt` arguments for a cap of `mb` MB.
pub(super) fn cap_args(mb: u64) -> [String; 2] {
    ["--storage-opt".to_string(), format!("size={mb}m")]
}

/// The size in MB of the filesystem `df -Pk /` describes, from its output.
pub(super) fn df_total_mb(out: &str) -> Option<u64> {
    let line = out.lines().filter(|l| !l.trim().is_empty()).nth(1)?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb / 1024)
}

/// Whether a container's filesystem of `total_mb` shows a cap of `cap_mb`
/// on a daemon whose own data disk is `disk_mb`: no bigger than the cap
/// and a little more (the filesystem's own rounding), and clearly smaller
/// than the daemon's disk, which an ignored cap would show instead (review
/// F2A-B M3: a 10 GB host disk passed for a 10 GB cap).
pub(super) fn shows_cap(total_mb: u64, cap_mb: u64, disk_mb: u64) -> bool {
    total_mb <= cap_mb + cap_mb / 20 + 16 && total_mb.saturating_mul(10) < disk_mb.saturating_mul(9)
}

/// The bytes of `container`'s writable layer as the daemon sees it
/// (`inspect --size`), within `timeout`.
pub(super) async fn layer_bytes(
    cli: &Cli,
    container: &str,
    timeout: Duration,
) -> Result<u64, SandboxError> {
    let args = [
        "container".to_string(),
        "inspect".to_string(),
        "--size".to_string(),
        "--format".to_string(),
        "{{.SizeRw}}".to_string(),
        container.to_string(),
    ];
    let out = cli.run(&args, Some(timeout)).await?;
    if !out.ok() {
        return Err(SandboxError::Transport(format!(
            "`{} container inspect --size` failed: {}",
            cli.name,
            out.message()
        )));
    }
    out.stdout.trim().parse().map_err(|_| {
        SandboxError::Transport(format!(
            "`{} container inspect --size` gave no size: {}",
            cli.name,
            out.stdout.trim()
        ))
    })
}

/// The entries `container`'s writable layer added as the daemon sees them
/// (`diff`, which can take seconds on a layer being written, and more on
/// the containerd image store), within `timeout`. Only added (`A`) lines
/// count: changed (`C`) and deleted (`D`) ones are bounded by the image's
/// own tree, and an honest agent that deletes a large tree must not pass
/// the limit for it.
pub(super) async fn layer_entries(
    cli: &Cli,
    container: &str,
    timeout: Duration,
) -> Result<u64, SandboxError> {
    let mut child = cli
        .command(&["diff".to_string(), container.to_string()])
        .spawn()
        .map_err(|e| cli.spawn_failed(e))?;
    let stderr = child
        .stderr
        .take()
        .map(|s| tokio::spawn(crate::capture::read_tail(s, 4096)));
    let mut stdout = child.stdout.take().expect("piped");
    let count = async {
        let mut n = 0u64;
        // Whether the next byte begins a line, and the line's first byte.
        let mut at_start = true;
        let mut first = 0u8;
        let mut col = 0usize;
        let mut buf = vec![0u8; 64 * 1024];
        loop {
            let got = stdout.read(&mut buf).await?;
            if got == 0 {
                break;
            }
            for b in &buf[..got] {
                if *b == b'\n' {
                    at_start = true;
                    col = 0;
                    continue;
                }
                if at_start {
                    first = *b;
                    at_start = false;
                }
                col += 1;
                if col == 2 && first == b'A' && *b == b' ' {
                    n += 1;
                }
            }
        }
        let status = child.wait().await?;
        Ok::<_, std::io::Error>((n, status))
    };
    let (n, status) = match tokio::time::timeout(timeout, count).await {
        Ok(r) => r.map_err(|e| SandboxError::io(format!("reading `{} diff`", cli.name), e))?,
        Err(_) => {
            return Err(SandboxError::Transport(format!(
                "`{} diff` did not finish within {}s",
                cli.name,
                timeout.as_secs()
            )));
        }
    };
    if !status.success() {
        let err = match stderr {
            Some(t) => t.await.map(|t| t.into_string()).unwrap_or_default(),
            None => String::new(),
        };
        return Err(SandboxError::Transport(format!(
            "`{} diff` failed: {}",
            cli.name,
            err.trim()
        )));
    }
    Ok(n)
}

/// Whether the daemon says `container` runs, from `{{.State.Status}}`:
/// running or paused (Podman's `.State.Running` says false for a paused
/// container) is `Ok(true)`; exited, created, stopped, dead, configured or
/// removing is `Ok(false)`, as is a container the daemon says it no longer
/// has; anything else, or no answer, is an error (the watch asks again).
pub(super) async fn running(cli: &Cli, container: &str) -> Result<bool, SandboxError> {
    let args = [
        "container",
        "inspect",
        "--format",
        "{{.State.Status}}",
        container,
    ]
    .map(String::from);
    let out = cli.run(&args, Some(Duration::from_secs(30))).await?;
    let gone = out.message().to_lowercase();
    if !out.ok()
        && (gone.contains("no such container") || gone.contains("no container with name or id"))
    {
        // The daemon answered: it no longer has it.
        return Ok(false);
    }
    match (out.ok(), out.stdout.trim()) {
        (true, "running" | "paused") => Ok(true),
        (true, "exited" | "created" | "stopped" | "dead" | "configured" | "removing") => Ok(false),
        _ => Err(SandboxError::Transport(format!(
            "`{} container inspect` did not say whether the container runs: {}",
            cli.name,
            if out.ok() {
                out.stdout.trim().to_string()
            } else {
                out.message()
            }
        ))),
    }
}

/// `pause` or `unpause` `container`, within `timeout`.
async fn freezer(
    cli: &Cli,
    verb: &str,
    container: &str,
    timeout: Duration,
) -> Result<(), SandboxError> {
    let args = [verb.to_string(), container.to_string()];
    let out = cli.run(&args, Some(timeout)).await?;
    if out.ok() {
        Ok(())
    } else {
        Err(SandboxError::Transport(format!(
            "`{} {verb}` failed: {}",
            cli.name,
            out.message()
        )))
    }
}

/// `kill --signal KILL` of `container`, within 60 s. A container already
/// gone or stopped counts as killed.
async fn kill_once(cli: &Cli, container: &str) -> Result<(), SandboxError> {
    let args = [
        "kill".to_string(),
        "--signal".to_string(),
        "KILL".to_string(),
        container.to_string(),
    ];
    let out = cli.run(&args, Some(Duration::from_secs(60))).await?;
    let lower = out.message().to_lowercase();
    if out.ok()
        || lower.contains("is not running")
        || lower.contains("no such container")
        || lower.contains("no container with name or id")
    {
        Ok(())
    } else {
        Err(SandboxError::Transport(format!(
            "`{} kill` failed: {}",
            cli.name,
            out.message()
        )))
    }
}

/// Stop `container` at once, from outside it: nothing in it needs to run.
/// First `pause` (20 s; the cgroup freezer stops every process where it
/// is), then `kill`: live on Docker 29, `kill` alone of a container writing
/// at disk speed took 20 s to return, and the writer wrote 3 GB more
/// meanwhile; `pause` held it within 4 s. A `kill` that fails (Podman 3.x
/// refuses to kill a paused container: "container state improper") is
/// tried once more after `unpause`; if that fails too, the container is
/// thawed again, so a kept sandbox is not left frozen.
pub(super) async fn kill(cli: &Cli, container: &str) -> Result<(), SandboxError> {
    const FREEZER: Duration = Duration::from_secs(20);
    let _ = freezer(cli, "pause", container, FREEZER).await;
    if kill_once(cli, container).await.is_ok() {
        return Ok(());
    }
    let _ = freezer(cli, "unpause", container, FREEZER).await;
    let again = kill_once(cli, container).await;
    if again.is_err() {
        let _ = freezer(cli, "unpause", container, FREEZER).await;
    }
    again
}

/// The number of processes in `container` (its cgroup's `pids.current`,
/// as `stats` reports it) and its own process limit (`PidsLimit`), both as
/// the daemon sees them. An unlimited container is an error: there is no
/// limit of its own to have hit.
pub(super) async fn processes(cli: &Cli, container: &str) -> Result<(u64, u64), SandboxError> {
    let ask = |args: &[&str]| {
        let args: Vec<String> = args.iter().map(|s| s.to_string()).collect();
        async move {
            let out = cli.run(&args, Some(Duration::from_secs(30))).await?;
            if !out.ok() {
                return Err(SandboxError::Transport(format!(
                    "`{} {}` failed: {}",
                    cli.name,
                    args[0],
                    out.message()
                )));
            }
            out.stdout.trim().parse::<i64>().map_err(|_| {
                SandboxError::Transport(format!(
                    "`{} {}` gave no number: {}",
                    cli.name,
                    args[0],
                    out.stdout.trim()
                ))
            })
        }
    };
    let limit = ask(&[
        "container",
        "inspect",
        "--format",
        "{{.HostConfig.PidsLimit}}",
        container,
    ])
    .await?;
    if limit <= 0 {
        return Err(SandboxError::Unsupported(format!(
            "container {container} has no process limit of its own"
        )));
    }
    let now = ask(&["stats", "--no-stream", "--format", "{{.PIDs}}", container]).await?;
    Ok((now.max(0) as u64, limit as u64))
}

/// A container's writable layer, as the daemon measures and stops it.
struct Layer {
    cli: Cli,
    container: String,
}

#[async_trait::async_trait]
impl crate::disk::Layer for Layer {
    async fn bytes(&self, timeout: Duration) -> Result<u64, SandboxError> {
        layer_bytes(&self.cli, &self.container, timeout).await
    }
    async fn entries(&self, timeout: Duration) -> Result<u64, SandboxError> {
        layer_entries(&self.cli, &self.container, timeout).await
    }
    async fn status(&self) -> Result<bool, SandboxError> {
        running(&self.cli, &self.container).await
    }
    async fn stop(&self) -> Result<(), SandboxError> {
        kill(&self.cli, &self.container).await
    }
    async fn unpause(&self) -> Result<(), SandboxError> {
        freezer(
            &self.cli,
            "unpause",
            &self.container,
            Duration::from_secs(20),
        )
        .await
    }
}

/// A [`crate::LayerWatch`] of `container` against `limit_mb`, at `pace`.
pub(super) async fn watch(
    cli: &Cli,
    container: &str,
    limit_mb: u64,
    pace: crate::disk::Pace,
    watched: crate::disk::Watched,
) -> (crate::LayerWatch, crate::disk::StartState) {
    let layer = Arc::new(Layer {
        cli: cli.clone(),
        container: container.to_string(),
    });
    crate::LayerWatch::start(limit_mb, pace, watched, layer).await
}

/// The label every disk probe carries, besides the owner's.
pub(super) const PROBE_LABEL: &str = "ai.benchflow.probe";

/// The `run` arguments of a probe of the daemon's disk ([`host_df`]),
/// named `name`: a throwaway container of `base_image` (the task's built
/// image, never a checkpoint the agent wrote), without a size cap so it
/// sees the daemon's disk, run as root with nothing it can use: no
/// network, a read-only root, no capabilities, no new privileges, 16
/// processes and 64 MB, the sandbox's runtime, a pinned `PATH` and no
/// loader variables, `/bin/df` by full path, and the owner's labels so a
/// sweep finds one left behind.
pub(super) fn host_df_args(base_image: &str, runtime: Option<&str>, name: &str) -> Vec<String> {
    let mut args: Vec<String> = [
        "run",
        "--rm",
        "--pull",
        "never",
        "--network",
        "none",
        "--name",
        name,
    ]
    .map(String::from)
    .to_vec();
    for (k, v) in crate::owner::labels() {
        args.push("--label".into());
        args.push(format!("{k}={v}"));
    }
    args.push("--label".into());
    args.push(format!("{PROBE_LABEL}=df"));
    args.extend(
        [
            "--user",
            "0:0",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "16",
            "--memory",
            "64m",
        ]
        .map(String::from),
    );
    if let Some(r) = runtime.filter(|r| !r.is_empty()) {
        args.push("--runtime".into());
        args.push(r.to_string());
    }
    for e in [
        format!("PATH={}", crate::shell::SYSTEM_PATH),
        "LD_PRELOAD=".into(),
        "LD_LIBRARY_PATH=".into(),
    ] {
        args.push("-e".into());
        args.push(e);
    }
    args.extend(
        [
            "--entrypoint",
            "/bin/sh",
            base_image,
            "-c",
            "/bin/df -Pk /; echo BF-I; /bin/df -Pi /",
        ]
        .map(String::from),
    );
    args
}

/// How long a probe of the daemon's disk may take.
pub(super) const HOST_DF_TIMEOUT: Duration = Duration::from_secs(60);

/// The filesystem the daemon keeps its containers on, blocks and inodes,
/// as a probe of `base_image` sees it ([`host_df_args`]), within
/// `timeout`; a probe past it is removed (`rm -f`). `None` when it cannot
/// be read. `tag` names the probe after its sandbox.
pub(super) async fn host_df(
    cli: &Cli,
    base_image: &str,
    runtime: Option<&str>,
    tag: &str,
    timeout: Duration,
) -> Option<crate::disk::DfRow> {
    static N: AtomicU64 = AtomicU64::new(0);
    if base_image.is_empty() {
        return None;
    }
    let tag: String = tag
        .chars()
        .rev()
        .filter(|c| c.is_ascii_alphanumeric())
        .take(8)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    let name = format!(
        "bf-df-{tag}-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::SeqCst)
    );
    let args = host_df_args(base_image, runtime, &name);
    let out = match cli.run(&args, Some(timeout)).await {
        Ok(out) => out,
        Err(_) => {
            let rm = ["rm".to_string(), "-f".to_string(), name];
            let _ = cli.run(&rm, Some(Duration::from_secs(10))).await;
            return None;
        }
    };
    if !out.ok() {
        return None;
    }
    crate::disk::parse_df(&out.stdout).into_iter().next()
}

/// The size in MB of the filesystem the daemon keeps its containers on
/// ([`host_df`]), once per daemon. `None` when it cannot be read.
pub(super) async fn data_disk_mb(
    cli: &Cli,
    base_image: &str,
    runtime: Option<&str>,
) -> Option<u64> {
    static SEEN: OnceLock<Mutex<BTreeMap<String, u64>>> = OnceLock::new();
    let seen = SEEN.get_or_init(Default::default);
    let key = daemon(cli);
    if let Some(mb) = seen.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
        return Some(*mb);
    }
    let mb = host_df(cli, base_image, runtime, "cap", HOST_DF_TIMEOUT)
        .await?
        .total_kb
        / 1024;
    seen.lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(key, mb);
    Some(mb)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_daemons_refusal_of_a_size_cap_is_told_apart() {
        for m in [
            "docker: Error response from daemon: --storage-opt is supported only for overlay over xfs with 'pquota' mount option.",
            "Error: storage-opt not supported for vfs",
            "Error response from daemon: Storage Opt size not supported",
            "Error response from daemon: invalid StorageOpt",
        ] {
            assert!(refused_cap(m), "{m}");
        }
        for m in [
            "docker: Error response from daemon: No such image: hello:latest",
            "Error response from daemon: Conflict. The container name is already in use",
        ] {
            assert!(!refused_cap(m), "{m}");
        }
    }

    #[test]
    fn the_size_df_shows_is_read() {
        let out = "Filesystem     1024-blocks  Used Available Capacity Mounted on\noverlay           307200    12    307188       1% /\n";
        assert_eq!(df_total_mb(out), Some(300));
        assert!(shows_cap(300, 300, 6_000));
        // The host's disk, seen through an uncapped overlay: no cap.
        assert!(!shows_cap(6_000, 300, 6_000));
        // Review F2A-B M3: a daemon that ignores the cap on a disk no
        // bigger than it shows the disk, which is no cap.
        assert!(!shows_cap(9_800, 10_240, 9_800));
        assert_eq!(df_total_mb("df: not found\n"), None);
    }

    /// U17 (F-5): the probe of the daemon's disk runs the task's image as
    /// root with nothing it can use, `df` by full path, a pinned `PATH`, no
    /// loader variables, labels and a name.
    #[test]
    fn the_disk_probe_runs_hardened() {
        let args = host_df_args("bf-task:abc", Some("runsc"), "bf-df-x-1");
        let joined = args.join(" ");
        for want in [
            "--rm --pull never --network none --name bf-df-x-1",
            "--label ai.benchflow.probe=df",
            "--user 0:0 --read-only --cap-drop ALL --security-opt no-new-privileges --pids-limit 16 --memory 64m --runtime runsc",
            "-e LD_PRELOAD= -e LD_LIBRARY_PATH=",
            "--entrypoint /bin/sh bf-task:abc -c /bin/df -Pk /; echo BF-I; /bin/df -Pi /",
        ] {
            assert!(joined.contains(want), "{want}: {joined}");
        }
        assert!(args.contains(&format!("PATH={}", crate::shell::SYSTEM_PATH)));
        assert!(joined.contains(&format!("{}=", crate::OWNER_LABEL)));
        assert!(!joined.contains("--storage-opt"));
    }
}
