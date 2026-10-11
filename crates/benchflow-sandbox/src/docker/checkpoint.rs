//! Checkpoints as committed images: `docker commit` of a sandbox's
//! container, labeled as a checkpoint and as ours, with the variables the
//! sandbox was started with blanked. A child is the normal `docker run` of
//! that image, which is local and never pulled.
//!
//! A commit leaves out what is mounted (volumes, bind mounts), so a
//! container with mounts, or whose image declares `VOLUME` paths, is
//! refused rather than checkpointed without them.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};

use super::cli::Cli;
use crate::checkpoint::{self, Checkpoint, KIND_LABEL, KIND_VALUE, Listed, Sweep};
use crate::{SandboxError, owner};

/// Bound on the inspect, list and remove calls.
const CALL_TIMEOUT: Duration = Duration::from_secs(60);
/// Bound on a commit, which copies the container's changed files.
const COMMIT_TIMEOUT: Duration = Duration::from_secs(1800);

/// A checkpoint image on the daemon.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Stored {
    pub id: String,
    /// The checkpoint name (the image's repository, without registry or tag).
    pub reference: String,
    pub size: u64,
    pub created: Option<DateTime<Utc>>,
}

/// `docker commit` container `container` as checkpoint `name`.
pub(super) async fn take(
    cli: &Cli,
    container: &str,
    name: &str,
    scrub_env: &[String],
) -> Result<Checkpoint, SandboxError> {
    let started = Instant::now();
    if checkpoint::parse(name).is_none() {
        return Err(SandboxError::Unsupported(format!(
            "`{name}` is not a checkpoint name (a bug in the caller)"
        )));
    }
    let mounted = mounted_paths(cli, container).await?;
    if !mounted.is_empty() {
        return Err(SandboxError::Unsupported(format!(
            "container {container} has mounted paths ({}), which a checkpoint (`{} commit`) leaves \
             out; checkpoints need a task image without VOLUME lines and a sandbox without mounts",
            mounted.join(", "),
            cli.name
        )));
    }
    let args = commit_args(container, name, scrub_env);
    let out = cli.run(&args, Some(COMMIT_TIMEOUT)).await?;
    if !out.ok() {
        return Err(cli.daemon_problem(&out).unwrap_or_else(|| {
            SandboxError::Transport(format!(
                "`{} commit` of {container} failed: {}",
                cli.name,
                out.message()
            ))
        }));
    }
    let size = image_size(cli, name).await;
    Ok(Checkpoint {
        provider: cli.name.clone(),
        reference: name.to_string(),
        seconds: started.elapsed().as_secs_f64(),
        stored_bytes: size,
        default_user: None,
    })
}

/// The `commit` arguments: the checkpoint label, our owner labels, and an
/// `ENV NAME=` for every variable to blank.
pub(crate) fn commit_args(container: &str, name: &str, scrub_env: &[String]) -> Vec<String> {
    let mut labels = owner::labels();
    labels.insert(KIND_LABEL.to_string(), KIND_VALUE.to_string());
    let mut args = vec!["commit".to_string()];
    for (k, v) in labels {
        args.push("--change".to_string());
        args.push(format!("LABEL {k}={}", quoted(&v)));
    }
    for var in scrub_env {
        // The names a sandbox can be started with (see `EnvArgs`); any
        // other could not have been set, and would break the instruction.
        let bad = var.is_empty()
            || var.contains(['=', '\0', '"', '\\'])
            || var.chars().any(char::is_whitespace);
        if bad {
            tracing::warn!("not blanking {var:?} in a checkpoint: not a variable name");
            continue;
        }
        args.push("--change".to_string());
        args.push(format!("ENV {var}="));
    }
    args.push(container.to_string());
    args.push(name.to_string());
    args
}

/// A Dockerfile string literal.
fn quoted(value: &str) -> String {
    format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""))
}

/// The paths a commit would leave out: the container's mounts and the
/// image's `VOLUME` paths.
pub(crate) async fn mounted_paths(cli: &Cli, container: &str) -> Result<Vec<String>, SandboxError> {
    let args = [
        "inspect".to_string(),
        "--format".to_string(),
        "{{json .Mounts}}\t{{json .Config.Volumes}}".to_string(),
        container.to_string(),
    ];
    let out = cli.run(&args, Some(CALL_TIMEOUT)).await?;
    if !out.ok() {
        return Err(cli.daemon_problem(&out).unwrap_or_else(|| {
            SandboxError::Transport(format!(
                "could not inspect container {container} for a checkpoint: {}",
                out.message()
            ))
        }));
    }
    Ok(parse_mounts(&out.stdout))
}

pub(crate) fn parse_mounts(text: &str) -> Vec<String> {
    #[derive(serde::Deserialize)]
    struct Mount {
        #[serde(rename = "Destination", default)]
        destination: String,
    }
    let line = text.trim();
    let (mounts, volumes) = line.split_once('\t').unwrap_or((line, "null"));
    let mut paths: Vec<String> = serde_json::from_str::<Option<Vec<Mount>>>(mounts)
        .ok()
        .flatten()
        .unwrap_or_default()
        .into_iter()
        .map(|m| m.destination)
        .collect();
    let volumes: Option<BTreeMap<String, serde_json::Value>> =
        serde_json::from_str(volumes.trim()).ok().flatten();
    paths.extend(volumes.unwrap_or_default().into_keys());
    paths.sort();
    paths.dedup();
    paths
}

async fn image_size(cli: &Cli, reference: &str) -> Option<u64> {
    let args = [
        "image".to_string(),
        "inspect".to_string(),
        "--format".to_string(),
        "{{.Size}}".to_string(),
        reference.to_string(),
    ];
    let out = cli.run(&args, Some(CALL_TIMEOUT)).await.ok()?;
    out.ok().then(|| out.stdout.trim().parse().ok()).flatten()
}

/// `docker rmi` a checkpoint image. True once it is gone; false while a
/// container still uses it (`-f` would only untag it, leaving an image no
/// list finds again), for the sweep to take later.
pub(super) async fn delete(cli: &Cli, reference: &str) -> Result<bool, SandboxError> {
    if checkpoint::parse(reference).is_none() {
        return Err(SandboxError::Unsupported(format!(
            "`{reference}` is not a checkpoint; only checkpoint images are deleted this way"
        )));
    }
    let args = ["rmi".to_string(), reference.to_string()];
    let out = cli.run(&args, Some(CALL_TIMEOUT)).await?;
    let lower = out.message().to_lowercase();
    if out.ok() || lower.contains("no such image") || lower.contains("image not known") {
        return Ok(true);
    }
    if lower.contains("conflict") || lower.contains("being used") || lower.contains("in use") {
        return Ok(false);
    }
    Err(cli.daemon_problem(&out).unwrap_or_else(|| {
        SandboxError::Transport(format!(
            "could not remove checkpoint image {reference}: {}",
            out.message()
        ))
    }))
}

/// Every checkpoint image of this owner scope on the daemon.
pub(crate) async fn list(cli: &Cli) -> Result<Vec<Stored>, SandboxError> {
    let filter = format!("label={KIND_LABEL}={KIND_VALUE}");
    let args: Vec<String> = ["image", "ls", "-q", "--no-trunc", "--filter", &filter]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let out = cli.run(&args, Some(CALL_TIMEOUT)).await?;
    if !out.ok() {
        return Err(cli.daemon_problem(&out).unwrap_or_else(|| {
            SandboxError::Transport(format!(
                "listing checkpoint images failed: {}",
                out.message()
            ))
        }));
    }
    let mut ids: Vec<String> = out.stdout.split_whitespace().map(str::to_string).collect();
    ids.sort();
    ids.dedup();
    if ids.is_empty() {
        return Ok(Vec::new());
    }
    let mut args: Vec<String> = vec![
        "image".into(),
        "inspect".into(),
        "--format".into(),
        "{{.Id}}\t{{.Size}}\t{{.Created}}\t{{range .RepoTags}}{{.}},{{end}}".into(),
    ];
    args.extend(ids);
    let out = cli.run(&args, Some(CALL_TIMEOUT)).await?;
    if !out.ok() && out.stdout.trim().is_empty() {
        return Err(SandboxError::Transport(format!(
            "reading checkpoint images failed: {}",
            out.message()
        )));
    }
    Ok(parse_list(&out.stdout))
}

/// One entry per checkpoint name of ours in `image inspect` output.
pub(crate) fn parse_list(text: &str) -> Vec<Stored> {
    let mut found = Vec::new();
    for line in text.lines() {
        let mut parts = line.split('\t');
        let (Some(id), Some(size)) = (parts.next(), parts.next()) else {
            continue;
        };
        let created = parts.next().and_then(parse_time);
        for tag in parts.next().unwrap_or_default().split(',') {
            let tag = tag.trim();
            if !checkpoint::ours(tag) {
                continue;
            }
            // `localhost/bf-ckpt-…:latest` on podman: the name alone.
            let base = tag.rsplit('/').next().unwrap_or(tag);
            let reference = base.split(':').next().unwrap_or(base).to_string();
            found.push(Stored {
                id: id.trim().to_string(),
                reference,
                size: size.trim().parse().unwrap_or(0),
                created,
            });
        }
    }
    found
}

/// An image's creation time: RFC 3339 (docker), or Go's default time
/// format (`2026-10-10 09:08:08.123 +0000 UTC`, podman).
fn parse_time(text: &str) -> Option<DateTime<Utc>> {
    let text = text.trim();
    if let Ok(t) = DateTime::parse_from_rfc3339(text) {
        return Some(t.with_timezone(&Utc));
    }
    let mut words: Vec<&str> = text.split_whitespace().collect();
    if words.len() == 4 {
        words.pop(); // the zone's name
    }
    DateTime::parse_from_str(&words.join(" "), "%Y-%m-%d %H:%M:%S%.f %z")
        .ok()
        .map(|t| t.with_timezone(&Utc))
}

pub(crate) fn listed(cli: &Cli, stored: Stored) -> Listed {
    Listed {
        checkpoint: Checkpoint {
            provider: cli.name.clone(),
            reference: stored.reference,
            seconds: 0.0,
            stored_bytes: Some(stored.size),
            default_user: None,
        },
        created: stored.created,
        state: None,
    }
}

/// Delete the checkpoint images [`checkpoint::verdict`] condemns. Best
/// effort: a failure is logged, not returned.
pub(super) async fn sweep(cli: &Cli) -> usize {
    let stored = match list(cli).await {
        Ok(stored) => stored,
        Err(err) => {
            tracing::warn!("could not list checkpoint images to sweep: {err}");
            return 0;
        }
    };
    let now = Utc::now();
    let mut removed = 0;
    for image in stored {
        let age = image
            .created
            .and_then(|c| now.signed_duration_since(c).to_std().ok());
        if checkpoint::verdict(&image.reference, age).await != Sweep::Delete {
            continue;
        }
        match delete(cli, &image.reference).await {
            Ok(false) => {
                tracing::info!(
                    "checkpoint image {} is still in use; left for later",
                    image.reference
                )
            }
            Ok(true) => {
                removed += 1;
                tracing::info!("removed leftover checkpoint image {}", image.reference);
            }
            Err(err) => tracing::warn!(
                "could not remove leftover checkpoint image {}: {err}",
                image.reference
            ),
        }
    }
    removed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mounts_and_declared_volumes_are_both_found() {
        assert!(parse_mounts("[]\tnull\n").is_empty());
        assert!(parse_mounts("null\t{}\n").is_empty());
        let text = r#"[{"Type":"volume","Destination":"/data"},{"Type":"bind","Destination":"/src"}]	{"/var/lib/db":{},"/data":{}}"#;
        assert_eq!(parse_mounts(text), ["/data", "/src", "/var/lib/db"]);
    }

    #[test]
    fn times_read_in_both_dialects() {
        let docker = parse_time("2026-10-10T09:08:08.123456789Z").unwrap();
        let podman = parse_time("2026-10-10 09:08:08.123456789 +0000 UTC").unwrap();
        assert_eq!(docker, podman);
        assert!(parse_time("yesterday").is_none());
    }
}
