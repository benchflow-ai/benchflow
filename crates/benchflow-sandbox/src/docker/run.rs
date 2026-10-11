//! Starting a container: `docker run -d` with the task's limits, a process
//! that only waits, and BenchFlow's ownership labels.

use std::time::{Duration, Instant};

use super::cli::{Cli, EnvArgs, Flavor};
use crate::{NetworkPolicy, SandboxError, SandboxSpec, owner};

/// What the container runs: a shell that sleeps until it is stopped. A
/// `sleep` in the background plus `wait` lets the TERM trap run at once,
/// and works with every `sleep` (busybox's has no `infinity`). `sleep` is
/// found in the system folders ([`crate::shell::SYSTEM_PATH`]) only and
/// starts without the loader variables: the loop runs it again every hour,
/// often as root, after the agent may have placed a program of that name on
/// the image's `PATH` or a library where the image's `LD_LIBRARY_PATH`
/// points. (`docker exec` takes its environment from the container's
/// configuration, not from this process.)
pub(crate) fn keepalive() -> String {
    format!(
        "PATH={}; unset {}; trap 'exit 0' TERM INT; while :; do sleep 3600 & wait $!; done",
        crate::shell::SYSTEM_PATH,
        crate::shell::LOADER_VARS.join(" ")
    )
}

/// How long a new container may take to reach `running`.
const START_TIMEOUT: Duration = Duration::from_secs(60);

/// A container name for a sandbox called `name`: `bf-<name>-<8 hex>`.
pub(crate) fn container_name(name: &str) -> String {
    let mut clean = String::new();
    for c in name.chars() {
        if c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-' {
            clean.push(c);
        } else if !clean.ends_with('-') {
            clean.push('-');
        }
    }
    let clean: String = clean
        .trim_matches(['-', '.', '_'])
        .chars()
        .take(40)
        .collect();
    let clean = clean.trim_end_matches(['-', '.', '_']);
    let suffix = &uuid::Uuid::new_v4().simple().to_string()[..8];
    if clean.is_empty() {
        format!("bf-{suffix}")
    } else {
        format!("bf-{clean}-{suffix}")
    }
}

/// `1.5` → `1.5`, `2.0` → `2`.
fn cpus_arg(cpus: f64) -> String {
    format!("{cpus}")
}

/// The full `docker run` argument list for `spec`, without the env flags.
pub(crate) fn run_args(cli: &Cli, spec: &SandboxSpec, name: &str, env: &EnvArgs) -> Vec<String> {
    let mut args: Vec<String> = ["run", "-d", "--init", "--name", name]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let mut labels = owner::labels();
    labels.insert("ai.benchflow.name".to_string(), spec.name.clone());
    for (k, v) in &spec.labels {
        labels.entry(k.clone()).or_insert_with(|| v.clone());
    }
    for (k, v) in labels {
        args.push("--label".to_string());
        args.push(format!("{k}={v}"));
    }
    if let Some(cpus) = spec.cpus.filter(|c| *c > 0.0) {
        args.push("--cpus".to_string());
        args.push(cpus_arg(cpus));
    }
    if let Some(mb) = spec.memory_mb.filter(|m| *m > 0) {
        // Swap equal to memory: no swap on top of the limit.
        args.push("--memory".to_string());
        args.push(format!("{mb}m"));
        args.push("--memory-swap".to_string());
        args.push(format!("{mb}m"));
    }
    if let Some(pids) = spec.pids_limit.filter(|p| *p > 0) {
        args.push("--pids-limit".to_string());
        args.push(pids.to_string());
    }
    if let Some(runtime) = spec.runtime.as_deref().filter(|r| !r.is_empty()) {
        args.push("--runtime".to_string());
        args.push(runtime.to_string());
    }
    match &spec.network {
        NetworkPolicy::None => {
            args.push("--network".to_string());
            args.push("none".to_string());
        }
        // Root inside installs the packet filter; the agent user, without
        // capabilities of its own, cannot change it.
        NetworkPolicy::Filtered { .. } => {
            args.push("--cap-add".to_string());
            args.push("NET_ADMIN".to_string());
        }
        NetworkPolicy::Open => {}
    }
    if spec.gpus > 0 {
        match cli.flavor {
            Flavor::Docker => {
                args.push("--gpus".to_string());
                args.push("all".to_string());
            }
            Flavor::Podman => {
                args.push("--device".to_string());
                args.push("nvidia.com/gpu=all".to_string());
            }
        }
    }
    if let Some(dir) = spec.workdir.as_deref().filter(|d| !d.is_empty()) {
        args.push("-w".to_string());
        args.push(dir.to_string());
    }
    args.extend(env.args.iter().cloned());
    args.push("--entrypoint".to_string());
    args.push("/bin/sh".to_string());
    args.push(spec.image.reference.clone());
    args.push("-c".to_string());
    args.push(keepalive());
    args
}

/// Start the container and wait until it runs. Returns its name.
pub(super) async fn start(cli: &Cli, spec: &SandboxSpec) -> Result<String, SandboxError> {
    if spec.disk_mb.is_some() {
        tracing::debug!(
            "{} does not cap a container's disk; disk_mb is not enforced for {}",
            cli.name,
            spec.name
        );
    }
    let name = container_name(&spec.name);
    let env = EnvArgs::new(&spec.env)?;
    let args = run_args(cli, spec, &name, &env);
    let mut cmd = cli.command(&args);
    for (k, v) in &env.process_env {
        cmd.env(k, v);
    }
    // The image is prepared, so this is quick; allow for a slow daemon.
    let out = cli
        .run_command(cmd, &args, Some(Duration::from_secs(300)))
        .await;
    env.close();
    let out = out?;
    if !out.ok() {
        if let Some(err) = cli.daemon_problem(&out) {
            return Err(err);
        }
        // A container may exist even though run failed (it was created but
        // could not start); remove it so nothing leaks.
        remove(cli, &name).await;
        return Err(SandboxError::Start(start_failure(
            cli,
            spec,
            &out.message(),
        )));
    }
    if let Err(err) = wait_running(cli, &name).await {
        remove(cli, &name).await;
        return Err(err);
    }
    Ok(name)
}

fn start_failure(cli: &Cli, spec: &SandboxSpec, message: &str) -> String {
    let lower = message.to_lowercase();
    let hint = if lower.contains("could not select device driver") || lower.contains("nvidia") {
        format!(
            "the task asks for GPUs, and this {} host cannot provide them. Install the NVIDIA \
             Container Toolkit on the host, or run the task where GPUs are available.",
            cli.name
        )
    } else if lower.contains("pull access denied")
        || lower.contains("no such image")
        || lower.contains("not found")
    {
        format!(
            "the image {} is not available to the {} host. Prepare it first (it is built or pulled \
             by `prepare`), or check its name.",
            spec.image.reference, cli.name
        )
    } else if lower.contains("executable file not found")
        || lower.contains("no such file or directory")
    {
        "the image has no `sh`, which BenchFlow needs to keep the sandbox running and to run \
         commands. Base the task's image on one with a shell."
            .to_string()
    } else if lower.contains("memory") || lower.contains("cpus") {
        format!(
            "the {} host refused the task's resource limits. Lower the task's cpus or memory, or \
             give the host more.",
            cli.name
        )
    } else {
        format!(
            "this is a problem with the {} host; run `{} info` and check its logs.",
            cli.name, cli.name
        )
    };
    format!(
        "`{} run` for {} failed: {message}. Next: {hint}",
        cli.name, spec.name
    )
}

/// Poll the container until it runs; fail with its logs if it exits.
async fn wait_running(cli: &Cli, name: &str) -> Result<(), SandboxError> {
    let started = Instant::now();
    let args = [
        "inspect".to_string(),
        "--format".to_string(),
        "{{.State.Status}}\t{{.State.ExitCode}}".to_string(),
        name.to_string(),
    ];
    loop {
        let out = cli.run(&args, Some(Duration::from_secs(30))).await?;
        if !out.ok() {
            if let Some(err) = cli.daemon_problem(&out) {
                return Err(err);
            }
            return Err(SandboxError::Start(format!(
                "container {name} disappeared right after it was started ({}). Something else on \
                 the {} host removed it; check `{} events`.",
                out.message(),
                cli.name,
                cli.name
            )));
        }
        let line = out.stdout.trim().to_string();
        let (status, code) = line.split_once('\t').unwrap_or((line.as_str(), ""));
        match status {
            "running" => return Ok(()),
            "exited" | "dead" | "stopped" => {
                let logs = cli
                    .run(
                        &[
                            "logs".to_string(),
                            "--tail".to_string(),
                            "50".to_string(),
                            name.to_string(),
                        ],
                        Some(Duration::from_secs(30)),
                    )
                    .await
                    .map(|o| format!("{}{}", o.stdout, o.stderr))
                    .unwrap_or_default();
                return Err(SandboxError::Start(format!(
                    "container {name} stopped right after starting (exit code {code}). Its output: \
                     {}. The image's `sh` may be missing or broken; check the task's image.",
                    super::cli::tail_chars(logs.trim(), 1000)
                )));
            }
            _ if started.elapsed() > START_TIMEOUT => {
                return Err(SandboxError::Start(format!(
                    "container {name} was still `{status}` after {}s. The {} host is slow or stuck; \
                     check `{} ps -a` and try again.",
                    START_TIMEOUT.as_secs(),
                    cli.name,
                    cli.name
                )));
            }
            _ => tokio::time::sleep(Duration::from_millis(200)).await,
        }
    }
}

/// `docker rm -f`, ignoring failures (used to clean up after a failed start).
async fn remove(cli: &Cli, name: &str) {
    let args = ["rm".to_string(), "-f".to_string(), name.to_string()];
    if let Ok(out) = cli.run(&args, Some(Duration::from_secs(60))).await
        && !out.ok()
        && !out.message().to_lowercase().contains("no such container")
    {
        tracing::warn!("could not remove container {name}: {}", out.message());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn container_names_are_readable_and_unique() {
        let a = container_name("hello-world__1");
        let b = container_name("hello-world__1");
        assert!(a.starts_with("bf-hello-world__1-"), "{a}");
        assert_ne!(a, b);
        assert_eq!(a.len(), "bf-hello-world__1-".len() + 8);
        assert!(container_name("a b/c").starts_with("bf-a-b-c-"));
        assert!(container_name("///").starts_with("bf-"));
        assert!(container_name(&"x".repeat(100)).len() <= 3 + 40 + 1 + 8);
    }

    /// The keep-alive loop runs `sleep` as the image's user (often root)
    /// every hour of the trial; it never finds it through the image's
    /// `PATH`.
    #[test]
    fn the_keepalive_finds_no_program_through_the_image_path() {
        use std::os::unix::process::CommandExt as _;
        let keepalive = keepalive();
        assert!(
            keepalive.starts_with(&format!(
                "PATH={}; unset LD_PRELOAD ",
                crate::shell::SYSTEM_PATH
            )),
            "{keepalive}"
        );
        let m = crate::shell::markers::Markers::new();
        let mut child = std::process::Command::new("/bin/sh")
            .arg("-c")
            .arg(&keepalive)
            .env("PATH", m.path())
            .process_group(0)
            .spawn()
            .unwrap();
        // Long enough for the loop's first `sleep` to start; then stop the
        // loop and its `sleep` together.
        std::thread::sleep(Duration::from_millis(500));
        let _ = std::process::Command::new("/bin/sh")
            .arg("-c")
            .arg(format!("kill -TERM -{}", child.id()))
            .status();
        let status = child.wait().unwrap();
        assert!(status.success(), "{status:?}");
        assert!(m.ran().is_empty(), "{:?}", m.ran());
    }

    #[test]
    fn cpus_print_without_trailing_zeros() {
        assert_eq!(cpus_arg(2.0), "2");
        assert_eq!(cpus_arg(1.5), "1.5");
    }
}
