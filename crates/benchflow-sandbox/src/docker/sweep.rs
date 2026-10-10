//! Removing containers that crashed runs left behind.
//!
//! A container is removed only when its labels say it is ours (owner scope),
//! it was made on this host, and the process that made it no longer exists
//! (see [`crate::owner`]). Containers of live runs, of other hosts sharing
//! the daemon, and of other tools are never touched.

use std::collections::BTreeMap;
use std::time::Duration;

use super::cli::Cli;
use crate::owner::{self, Verdict};
use crate::{OWNER_LABEL, SandboxError};

pub(super) async fn sweep(cli: &Cli) -> Result<usize, SandboxError> {
    let list = [
        "ps".to_string(),
        "-a".to_string(),
        "-q".to_string(),
        "--no-trunc".to_string(),
        "--filter".to_string(),
        format!("label={OWNER_LABEL}={}", owner::scope()),
    ];
    let out = cli.run(&list, Some(Duration::from_secs(60))).await?;
    if !out.ok() {
        return Err(cli.daemon_problem(&out).unwrap_or_else(|| {
            SandboxError::Transport(format!(
                "listing BenchFlow containers failed: {}",
                out.message()
            ))
        }));
    }
    let ids: Vec<String> = out.stdout.split_whitespace().map(str::to_string).collect();
    if ids.is_empty() {
        return Ok(0);
    }

    // `inspect` prints labels as JSON the same way on docker and podman.
    let mut inspect = vec![
        "inspect".to_string(),
        "--format".to_string(),
        "{{.Id}}\t{{json .Config.Labels}}".to_string(),
    ];
    inspect.extend(ids.iter().cloned());
    let out = cli.run(&inspect, Some(Duration::from_secs(60))).await?;
    // A container removed between the two calls makes inspect fail for it
    // alone; the others still print.
    let mut leftovers = Vec::new();
    for line in out.stdout.lines() {
        let Some((id, json)) = line.split_once('\t') else {
            continue;
        };
        let labels: BTreeMap<String, String> = serde_json::from_str(json).unwrap_or_default();
        if owner::judge(|k| labels.get(k).map(String::as_str)).await == Verdict::Leftover {
            leftovers.push(id.trim().to_string());
        }
    }
    if leftovers.is_empty() {
        return Ok(0);
    }
    let mut rm = vec!["rm".to_string(), "-f".to_string()];
    rm.extend(leftovers.iter().cloned());
    let out = cli.run(&rm, Some(Duration::from_secs(120))).await?;
    if !out.ok() {
        tracing::warn!(
            "removing {} leftover container(s) partly failed: {}",
            leftovers.len(),
            out.message()
        );
        // Count what `rm` confirmed by printing it.
        return Ok(out.stdout.lines().filter(|l| !l.trim().is_empty()).count());
    }
    tracing::info!(
        "removed {} leftover BenchFlow container(s)",
        leftovers.len()
    );
    Ok(leftovers.len())
}
