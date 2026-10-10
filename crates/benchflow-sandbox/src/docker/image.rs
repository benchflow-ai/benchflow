//! Preparing images: build a task's Dockerfile, pull a prebuilt image, and
//! bake agent installs on top as a cached image of their own.
//!
//! Tags come from content hashes, so an unchanged task never rebuilds:
//!
//! - a task build is `benchflow/<name>:<digest12>`, where `digest12` is the
//!   first 12 characters of the task's digest;
//! - agent layers are `benchflow/<name>:<digest12>-<layers12>`, where
//!   `layers12` hashes the base image's id and the layer scripts, so a
//!   hundred trials install the agent once, and a rebuilt base reinstalls it.

use std::collections::VecDeque;
use std::path::Path;
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};
use tokio::io::{AsyncBufReadExt, BufReader};

use super::cache;
use super::cli::{Cli, Output, exit_code};
use crate::shell::login_free_shell;
use crate::{Image, ImageSource, ImageSpec, OWNER_LABEL, SandboxError, owner};

/// Waits before each retry of a build or pull that failed for a transient
/// reason (BenchFlow 0.8's `_DOCKER_BUILD_RETRY_DELAYS_SEC`).
#[cfg(not(test))]
const RETRY_DELAYS: [Duration; 2] = [Duration::from_secs(2), Duration::from_secs(5)];
#[cfg(test)]
const RETRY_DELAYS: [Duration; 2] = [Duration::from_millis(10), Duration::from_millis(10)];

/// How many lines of build output an error keeps.
const LOG_TAIL_LINES: usize = 40;

/// Build or pull `spec`'s image and its layers.
pub(super) async fn prepare(
    cli: &Cli,
    spec: &ImageSpec,
    log: &(dyn Fn(&str) + Send + Sync),
) -> Result<Image, SandboxError> {
    let started = Instant::now();
    let (base, base_fresh) = match &spec.source {
        ImageSource::Build {
            context,
            dockerfile,
        } => build_task_image(cli, spec, context, dockerfile, log).await?,
        ImageSource::Pull { reference } => pull(cli, spec, reference, log).await?,
    };
    let image = if spec.layers.is_empty() {
        Image {
            reference: base.clone(),
            fresh: base_fresh,
            duration: started.elapsed(),
        }
    } else {
        let (layered, layers_fresh) = build_layers(cli, spec, &base, log).await?;
        Image {
            reference: layered,
            fresh: base_fresh || layers_fresh,
            duration: started.elapsed(),
        }
    };
    let mut ids = Vec::new();
    for reference in [&base, &image.reference] {
        if let Ok(Some(id)) = image_id(cli, reference).await {
            cache::note_use(&id);
            ids.push(id);
        }
    }
    cache::note_reference(&image.reference, ids);
    if image.fresh {
        cache::trim(cli, log).await;
    }
    Ok(image)
}

/// A Docker repository name for `name`: lowercase letters and digits joined
/// by single dashes.
pub(crate) fn repo_name(name: &str) -> String {
    let mut out = String::new();
    for c in name.to_lowercase().chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c);
        } else if !out.is_empty() && !out.ends_with('-') {
            out.push('-');
        }
    }
    let out = out.trim_end_matches('-');
    let out: String = out.chars().take(64).collect();
    let out = out.trim_end_matches('-').to_string();
    if out.is_empty() {
        "task".to_string()
    } else {
        out
    }
}

/// The first 12 characters of `digest` usable in a tag; a hash of the
/// fallback text when the digest is empty.
fn digest12(digest: &str, fallback: &str) -> String {
    let clean: String = digest
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .map(|c| c.to_ascii_lowercase())
        .take(12)
        .collect();
    if clean.is_empty() {
        hex::encode(Sha256::digest(fallback.as_bytes()))[..12].to_string()
    } else {
        clean
    }
}

/// The image's id when `reference` exists locally.
async fn image_id(cli: &Cli, reference: &str) -> Result<Option<String>, SandboxError> {
    let args = [
        "image".to_string(),
        "inspect".to_string(),
        "--format".to_string(),
        "{{.Id}}".to_string(),
        reference.to_string(),
    ];
    let out = cli.run(&args, Some(Duration::from_secs(60))).await?;
    if out.ok() {
        return Ok(Some(out.stdout.trim().to_string()));
    }
    if let Some(err) = cli.daemon_problem(&out) {
        return Err(err);
    }
    // "No such image", "no such object", podman's "image not known": absent.
    Ok(None)
}

async fn build_task_image(
    cli: &Cli,
    spec: &ImageSpec,
    context: &Path,
    dockerfile: &Path,
    log: &(dyn Fn(&str) + Send + Sync),
) -> Result<(String, bool), SandboxError> {
    let digest = digest12(
        &spec.digest,
        &format!("{}\n{}", context.display(), dockerfile.display()),
    );
    if spec.digest.is_empty() {
        tracing::warn!(
            "image {} has no digest; its tag is derived from its paths, so edits to its files \
             will not trigger a rebuild",
            spec.name
        );
    }
    let tag = format!("benchflow/{}:{digest}", repo_name(&spec.name));
    if image_id(cli, &tag).await?.is_some() {
        log(&format!("using cached image {tag}"));
        return Ok((tag, false));
    }
    let mut args = vec![
        "build".to_string(),
        "-f".to_string(),
        dockerfile.display().to_string(),
        "-t".to_string(),
        tag.clone(),
        "--label".to_string(),
        format!("{OWNER_LABEL}={}", owner::scope()),
        "--label".to_string(),
        format!("ai.benchflow.digest={}", spec.digest),
    ];
    if let Some(platform) = &spec.platform {
        args.push("--platform".to_string());
        args.push(platform.clone());
    }
    args.push(context.display().to_string());
    log(&format!("building {tag} from {}", dockerfile.display()));
    run_with_retries(cli, &args, spec, log, Step::TaskBuild).await?;
    Ok((tag, true))
}

async fn pull(
    cli: &Cli,
    spec: &ImageSpec,
    reference: &str,
    log: &(dyn Fn(&str) + Send + Sync),
) -> Result<(String, bool), SandboxError> {
    if image_id(cli, reference).await?.is_some() {
        log(&format!("using local image {reference}"));
        return Ok((reference.to_string(), false));
    }
    let mut args = vec!["pull".to_string()];
    if let Some(platform) = &spec.platform {
        args.push("--platform".to_string());
        args.push(platform.clone());
    }
    args.push(reference.to_string());
    log(&format!("pulling {reference}"));
    run_with_retries(cli, &args, spec, log, Step::Pull).await?;
    Ok((reference.to_string(), true))
}

/// The derived image's Dockerfile: the base, then each layer as one `RUN`
/// (as root, through bash when the image has it), then the base's user back.
pub(crate) fn layers_dockerfile(
    base: &str,
    layers: &[String],
    restore_user: Option<&str>,
) -> String {
    let mut text = format!("FROM {base}\nUSER root\n");
    for layer in layers {
        let argv = serde_json::json!(["/bin/sh", "-c", login_free_shell(layer)]);
        text.push_str(&format!("RUN {argv}\n"));
    }
    if let Some(user) = restore_user.filter(|u| !u.trim().is_empty()) {
        text.push_str(&format!("USER {user}\n"));
    }
    text
}

/// The tag for `layers` on top of the image with id `base_id`.
pub(crate) fn layers_tag(spec: &ImageSpec, base_id: &str) -> String {
    let mut h = Sha256::new();
    h.update(b"benchflow-layers-v1\0");
    h.update(base_id.as_bytes());
    h.update(b"\0");
    h.update(spec.platform.as_deref().unwrap_or("").as_bytes());
    for layer in &spec.layers {
        h.update(b"\0");
        h.update(layer.as_bytes());
    }
    let layers12 = &hex::encode(h.finalize())[..12];
    let digest = digest12(&spec.digest, base_id);
    format!("benchflow/{}:{digest}-{layers12}", repo_name(&spec.name))
}

async fn build_layers(
    cli: &Cli,
    spec: &ImageSpec,
    base: &str,
    log: &(dyn Fn(&str) + Send + Sync),
) -> Result<(String, bool), SandboxError> {
    let args = [
        "image".to_string(),
        "inspect".to_string(),
        "--format".to_string(),
        // `.Config` as JSON: Docker 29 returns it as a map that leaves out
        // an unset `User`, and `{{.Config.User}}` then fails the template.
        "{{.Id}}\t{{json .Config}}".to_string(),
        base.to_string(),
    ];
    let out = cli.run(&args, Some(Duration::from_secs(60))).await?;
    if !out.ok() {
        return Err(cli.daemon_problem(&out).unwrap_or_else(|| SandboxError::Image {
            name: spec.name.clone(),
            message: format!(
                "the base image {base} was prepared but cannot be inspected ({}); this is a problem \
                 with the {} host; run `{} image inspect {base}` to see why",
                out.message(),
                cli.name,
                cli.name
            ),
            log_tail: String::new(),
            task_fault: false,
        }));
    }
    let (base_id, user) = id_and_user(&out.stdout);
    let tag = layers_tag(spec, &base_id);
    if image_id(cli, &tag).await?.is_some() {
        log(&format!("using cached agent layers {tag}"));
        return Ok((tag, false));
    }
    let dir = tempfile::Builder::new()
        .prefix("benchflow-layers-")
        .tempdir()
        .map_err(|e| SandboxError::io("creating a build folder for agent layers", e))?;
    let dockerfile = dir.path().join("Dockerfile");
    std::fs::write(
        &dockerfile,
        layers_dockerfile(base, &spec.layers, Some(&user)),
    )
    .map_err(|e| SandboxError::io("writing the agent layers Dockerfile", e))?;
    let mut args = vec![
        "build".to_string(),
        "-f".to_string(),
        dockerfile.display().to_string(),
        "-t".to_string(),
        tag.clone(),
        "--label".to_string(),
        format!("{OWNER_LABEL}={}", owner::scope()),
        "--label".to_string(),
        format!("ai.benchflow.base={base_id}"),
    ];
    if let Some(platform) = &spec.platform {
        args.push("--platform".to_string());
        args.push(platform.clone());
    }
    args.push(dir.path().display().to_string());
    log(&format!(
        "installing {} agent layer(s) on {base} as {tag}",
        spec.layers.len()
    ));
    run_with_retries(cli, &args, spec, log, Step::Layers).await?;
    Ok((tag, true))
}

/// The image id and configured user from `{{.Id}}\t{{json .Config}}`. The
/// user is empty when the image sets none (or `.Config` is `null`).
pub(crate) fn id_and_user(stdout: &str) -> (String, String) {
    let line = stdout.trim();
    let (id, config) = line.split_once('\t').unwrap_or((line, ""));
    let user = serde_json::from_str::<serde_json::Value>(config)
        .ok()
        .and_then(|c| c.get("User")?.as_str().map(str::to_string))
        .unwrap_or_default();
    (id.to_string(), user)
}

/// What a CLI call in [`run_with_retries`] is doing, for its errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Step {
    TaskBuild,
    Pull,
    Layers,
}

async fn run_with_retries(
    cli: &Cli,
    args: &[String],
    spec: &ImageSpec,
    log: &(dyn Fn(&str) + Send + Sync),
    step: Step,
) -> Result<(), SandboxError> {
    let attempts = RETRY_DELAYS.len() + 1;
    for attempt in 1..=attempts {
        let run = run_streaming(cli, args, spec.build_timeout, log).await?;
        let Some(run) = run else {
            let limit = spec.build_timeout.unwrap_or_default().as_secs();
            return Err(SandboxError::Image {
                name: spec.name.clone(),
                message: format!(
                    "`{}` did not finish within the {limit}s build timeout and was stopped. If the \
                     build is just slow, raise the task's build timeout; if a download stalled, \
                     check the network and run again.",
                    cli.show(args)
                ),
                log_tail: String::new(),
                task_fault: false,
            });
        };
        if run.code == 0 {
            return Ok(());
        }
        let output = Output {
            code: run.code,
            stdout: String::new(),
            stderr: run.text.clone(),
        };
        if let Some(err) = cli.daemon_problem(&output) {
            return Err(err);
        }
        let reading = crate::build_log::read(&run.text);
        let transient = reading.transient;
        if let Some(pattern) = transient
            && attempt < attempts
        {
            let delay = RETRY_DELAYS[attempt - 1];
            log(&format!(
                "retrying after a transient failure ({pattern}), attempt {} of {attempts}, in {}s",
                attempt + 1,
                delay.as_secs()
            ));
            tokio::time::sleep(delay).await;
            continue;
        }
        return Err(failure(cli, args, spec, step, &run, &reading, attempt));
    }
    unreachable!("the loop returns on its last attempt")
}

fn failure(
    cli: &Cli,
    args: &[String],
    spec: &ImageSpec,
    step: Step,
    run: &Streamed,
    reading: &crate::build_log::Reading,
    attempts: usize,
) -> SandboxError {
    let key = &reading.key_line;
    let task_fault = reading.task_fault && step != Step::Layers;
    let message = match (step, task_fault) {
        (Step::TaskBuild, true) => format!(
            "the task's Dockerfile failed to build ({key}). This is a problem in the task: fix the \
             Dockerfile, then check it with `{}`.",
            cli.show(args)
        ),
        (Step::Pull, true) => format!(
            "the image the task names cannot be pulled ({key}). Fix the image reference in the task, \
             or run `{} login` if the registry needs credentials.",
            cli.name
        ),
        (Step::Layers, _) => format!(
            "installing the agent on top of the task image failed ({key}). The agent's install step \
             or the network is at fault, not the task; the log tail shows the failing command."
        ),
        (_, false) => format!(
            "`{}` failed for a reason outside the task ({key}){}. Check the network and the {} host \
             (disk space: `{} system df`), then run again.",
            cli.show(args),
            if attempts > 1 {
                format!(" after {attempts} attempts")
            } else {
                String::new()
            },
            cli.name,
            cli.name
        ),
    };
    SandboxError::Image {
        name: spec.name.clone(),
        message,
        log_tail: run.tail.join("\n"),
        task_fault,
    }
}

/// A finished streaming call.
struct Streamed {
    code: i32,
    /// The last lines, for error messages.
    tail: Vec<String>,
    /// The last 64 KiB of output, for classifying a failure.
    text: String,
}

/// Run the CLI, sending every output line to `log`. `None` when it ran past
/// `timeout` (it is killed).
async fn run_streaming(
    cli: &Cli,
    args: &[String],
    timeout: Option<Duration>,
    log: &(dyn Fn(&str) + Send + Sync),
) -> Result<Option<Streamed>, SandboxError> {
    let mut cmd = cli.command(args);
    // BuildKit prints plain, line-by-line progress; podman ignores this.
    cmd.env("BUILDKIT_PROGRESS", "plain");
    let mut child = cmd.spawn().map_err(|e| cli.spawn_failed(e))?;
    let stdout = child.stdout.take().expect("stdout is piped");
    let stderr = child.stderr.take().expect("stderr is piped");

    let mut tail: VecDeque<String> = VecDeque::new();
    let mut text = String::new();
    let mut emit = |raw: &[u8]| {
        let line = String::from_utf8_lossy(raw);
        let line = line.trim_end_matches(['\n', '\r']);
        let shown = super::cli::tail_chars(line, 4000);
        log(&shown);
        if tail.len() == LOG_TAIL_LINES {
            tail.pop_front();
        }
        tail.push_back(shown.clone());
        text.push_str(&shown);
        text.push('\n');
        if text.len() > 128 * 1024 {
            let cut = text.len() - 64 * 1024;
            let cut = (cut..text.len())
                .find(|i| text.is_char_boundary(*i))
                .unwrap_or(0);
            text.drain(..cut);
        }
    };

    let reading = async {
        let mut out = BufReader::new(stdout);
        let mut err = BufReader::new(stderr);
        let (mut out_buf, mut err_buf) = (Vec::new(), Vec::new());
        let (mut out_open, mut err_open) = (true, true);
        while out_open || err_open {
            tokio::select! {
                read = out.read_until(b'\n', &mut out_buf), if out_open => match read {
                    Ok(0) | Err(_) => {
                        if !out_buf.is_empty() { emit(&out_buf); out_buf.clear(); }
                        out_open = false;
                    }
                    Ok(_) => { emit(&out_buf); out_buf.clear(); }
                },
                read = err.read_until(b'\n', &mut err_buf), if err_open => match read {
                    Ok(0) | Err(_) => {
                        if !err_buf.is_empty() { emit(&err_buf); err_buf.clear(); }
                        err_open = false;
                    }
                    Ok(_) => { emit(&err_buf); err_buf.clear(); }
                },
            }
        }
        child.wait().await
    };
    let status = match timeout {
        Some(limit) => match tokio::time::timeout(limit, reading).await {
            Ok(status) => status,
            Err(_) => return Ok(None),
        },
        None => reading.await,
    }
    .map_err(|e| SandboxError::io(format!("running `{}`", cli.show(args)), e))?;
    Ok(Some(Streamed {
        code: exit_code(status),
        tail: tail.into_iter().collect(),
        text,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repo_names_are_valid_docker_names() {
        assert_eq!(repo_name("Hello World"), "hello-world");
        assert_eq!(repo_name("__weird..name__"), "weird-name");
        assert_eq!(repo_name("a/b:c@d"), "a-b-c-d");
        assert_eq!(repo_name("!!!"), "task");
        assert!(repo_name(&"x".repeat(300)).len() <= 64);
    }

    #[test]
    fn reads_the_user_from_the_config_json() {
        // Docker 29 leaves an unset User out of the map.
        let ubuntu = "sha256:abc\t{\"Env\":[\"PATH=/bin\"],\"Cmd\":[\"/bin/bash\"]}\n";
        assert_eq!(id_and_user(ubuntu), ("sha256:abc".into(), String::new()));
        let user = "sha256:abc\t{\"User\":\"node\",\"Env\":[]}";
        assert_eq!(id_and_user(user), ("sha256:abc".into(), "node".into()));
        assert_eq!(
            id_and_user("sha256:abc\tnull"),
            ("sha256:abc".into(), String::new())
        );
    }

    #[test]
    fn digest12_cleans_and_falls_back() {
        assert_eq!(digest12("sha256:ABCDEF0123456789", ""), "sha256abcdef");
        assert_eq!(digest12("0123456789abcdef", ""), "0123456789ab");
        assert_eq!(digest12("", "x").len(), 12);
    }

    #[test]
    fn layers_become_json_run_lines_and_restore_the_user() {
        let text = layers_dockerfile(
            "benchflow/t:abc",
            &["apt-get install -y \"curl\"\necho 'done'".to_string()],
            Some("agent"),
        );
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], "FROM benchflow/t:abc");
        assert_eq!(lines[1], "USER root");
        assert!(
            lines[2].starts_with(r#"RUN ["/bin/sh","-c","if command -v bash"#),
            "{}",
            lines[2]
        );
        // The JSON on the RUN line decodes back to the wrapped script.
        let argv: Vec<String> =
            serde_json::from_str(lines[2].strip_prefix("RUN ").unwrap()).unwrap();
        assert_eq!(
            argv[2],
            login_free_shell("apt-get install -y \"curl\"\necho 'done'")
        );
        assert_eq!(lines[3], "USER agent");
        assert_eq!(lines.len(), 4);
    }

    #[test]
    fn layers_tag_changes_with_base_and_layers() {
        let mut spec = ImageSpec {
            source: ImageSource::Pull {
                reference: "ubuntu:24.04".into(),
            },
            layers: vec!["echo a".into()],
            name: "My Task".into(),
            digest: "0123456789abcdef".into(),
            build_timeout: None,
            platform: None,
        };
        let a = layers_tag(&spec, "sha256:1");
        assert!(a.starts_with("benchflow/my-task:0123456789ab-"), "{a}");
        assert_ne!(a, layers_tag(&spec, "sha256:2"));
        spec.layers.push("echo b".into());
        assert_ne!(a, layers_tag(&spec, "sha256:1"));
    }
}
