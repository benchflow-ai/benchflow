//! Checkpoints: a sandbox's filesystem, stored so new sandboxes can start
//! from it (Docker: a committed image; Daytona: a snapshot).
//!
//! Daytona snapshots carry no labels, so a checkpoint says who made it in
//! its name, on every backend alike:
//!
//! ```text
//! bf-ckpt-<owner, lowercased>-<sha6 of owner>-<sha6 of machine>-<pid>-<uuid12>
//! ```
//!
//! The owner is the owner scope ([`crate::owner::scope`]); the two hashes
//! keep scopes that differ only in case apart and name the machine without
//! spelling it out: its machine id ([`crate::owner::machine_id`]; the host
//! name only where there is none), so two machines with one host name keep
//! their checkpoints apart. A checkpoint kept on purpose (`--retain-checkpoints`)
//! carries pid `0`, which no run has.
//!
//! The sweep ([`verdict`]) deletes a checkpoint of this owner and this
//! machine whose process is gone at once; others of this owner (other
//! machines, kept ones, and those named after a host name by earlier
//! builds) only past an age limit; never another owner's.

use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Every checkpoint's name starts with this.
pub const PREFIX: &str = "bf-ckpt-";

/// The label Docker checkpoint images carry (`ai.benchflow.kind=checkpoint`),
/// so the image-cache trim never takes one for a cached image.
pub const KIND_LABEL: &str = "ai.benchflow.kind";
pub const KIND_VALUE: &str = "checkpoint";

/// A checkpoint of another host (whose process cannot be checked) is swept
/// past this age.
pub const OTHER_HOST_AGE: Duration = Duration::from_secs(24 * 3600);
/// A checkpoint kept on purpose is swept past this age.
pub const KEPT_AGE: Duration = Duration::from_secs(7 * 24 * 3600);

/// A stored checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Checkpoint {
    /// The backend: `docker`, `podman`, `daytona`.
    pub provider: String,
    /// What the backend starts from and deletes: an image reference, a
    /// snapshot name.
    pub reference: String,
    /// How long taking it took, seconds.
    #[serde(default)]
    pub seconds: f64,
    /// Its stored size, where the backend says.
    #[serde(default)]
    pub stored_bytes: Option<u64>,
    /// The user commands run as when they name none, as in the sandbox it
    /// was taken from (Daytona: the image's `USER`), for its children.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_user: Option<String>,
}

/// A checkpoint the backend lists.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Listed {
    pub checkpoint: Checkpoint,
    pub created: Option<DateTime<Utc>>,
    /// The backend's state (`active`, `pending`, …), where it has one.
    pub state: Option<String>,
}

fn sha6(s: &str) -> String {
    hex::encode(Sha256::digest(s.as_bytes()))[..6].to_string()
}

fn owner_part(scope: &str) -> String {
    let lower: String = scope
        .to_ascii_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    let trimmed = lower.trim_matches('-').to_string();
    if trimmed.is_empty() {
        "x".into()
    } else {
        trimmed
    }
}

/// A new checkpoint name for this owner scope, host, and process; `kept`:
/// meant to outlive the run (pid `0`).
pub fn name(kept: bool) -> String {
    name_for(
        &crate::owner::scope(),
        crate::owner::machine_id(),
        if kept { 0 } else { std::process::id() },
    )
}

pub(crate) fn name_for(scope: &str, host: &str, pid: u32) -> String {
    let id = uuid::Uuid::new_v4().simple().to_string();
    format!(
        "{PREFIX}{}-{}-{}-{pid}-{}",
        owner_part(scope),
        sha6(scope),
        sha6(host),
        &id[..12]
    )
}

/// What a checkpoint's name says about who made it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Parsed {
    pub owner_hash: String,
    pub host_hash: String,
    pub pid: u32,
}

/// Read a checkpoint name (an image reference's tag is ignored). `None`:
/// not a checkpoint name.
pub fn parse(reference: &str) -> Option<Parsed> {
    let base = reference.rsplit('/').next()?;
    let base = base.split(':').next()?;
    let rest = base.strip_prefix(PREFIX)?;
    let parts: Vec<&str> = rest.rsplitn(5, '-').collect();
    if parts.len() != 5 {
        return None;
    }
    let (uuid, pid, host, owner) = (parts[0], parts[1], parts[2], parts[3]);
    let hexy = |s: &str, n: usize| s.len() == n && s.bytes().all(|b| b.is_ascii_hexdigit());
    if !hexy(uuid, 12) || !hexy(host, 6) || !hexy(owner, 6) || parts[4].is_empty() {
        return None;
    }
    Some(Parsed {
        owner_hash: owner.to_string(),
        host_hash: host.to_string(),
        pid: pid.parse().ok()?,
    })
}

/// Whether this owner scope made the checkpoint.
pub fn ours(reference: &str) -> bool {
    parse(reference).is_some_and(|p| p.owner_hash == sha6(&crate::owner::scope()))
}

/// Whether a run on this host that is still going made the checkpoint (it
/// may still need it, so cleaning leaves it).
pub async fn in_use(reference: &str) -> bool {
    let Some(p) = parse(reference) else {
        return false;
    };
    p.pid != 0
        && p.host_hash == sha6(crate::owner::machine_id())
        && (p.pid == std::process::id()
            || crate::owner::process_state(p.pid).await != crate::owner::ProcessState::Gone)
}

/// Why cleaning should leave a checkpoint of this owner unless asked by
/// age: kept on purpose (`--retain-checkpoints`), or made on another
/// machine, whose run may still need it. `None`: neither.
pub fn spare_reason(reference: &str) -> Option<&'static str> {
    let p = parse(reference)?;
    if p.pid == 0 {
        Some("kept on purpose (--retain-checkpoints); add --older-than to remove it")
    } else if p.host_hash != sha6(crate::owner::machine_id()) {
        Some("made on another machine, whose run may still need it; add --older-than to remove it")
    } else {
        None
    }
}

/// What the sweep does with a checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sweep {
    /// Not ours (or not a checkpoint): never touched.
    Foreign,
    Keep,
    Delete,
}

/// The sweep's rule for one checkpoint, given its age (`None`: unknown,
/// which keeps anything that would need an age).
pub async fn verdict(reference: &str, age: Option<Duration>) -> Sweep {
    let Some(p) = parse(reference) else {
        return Sweep::Foreign;
    };
    if p.owner_hash != sha6(&crate::owner::scope()) {
        return Sweep::Foreign;
    }
    let old = |limit: Duration| age.is_some_and(|a| a > limit);
    if p.pid == 0 {
        return if old(KEPT_AGE) {
            Sweep::Delete
        } else {
            Sweep::Keep
        };
    }
    if p.host_hash != sha6(crate::owner::machine_id()) {
        return if old(OTHER_HOST_AGE) {
            Sweep::Delete
        } else {
            Sweep::Keep
        };
    }
    if p.pid == std::process::id() {
        return Sweep::Keep;
    }
    match crate::owner::process_state(p.pid).await {
        crate::owner::ProcessState::Gone => Sweep::Delete,
        _ => Sweep::Keep,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_round_trip_and_say_who_made_them() {
        let n = name_for("Team_A.b-c", "host-1", 4242);
        assert!(n.starts_with("bf-ckpt-team-a-b-c-"), "{n}");
        assert!(n.len() < 128 && n == n.to_ascii_lowercase(), "{n}");
        let p = parse(&n).unwrap();
        assert_eq!(p.pid, 4242);
        assert_eq!(p.owner_hash, sha6("Team_A.b-c"));
        assert_eq!(p.host_hash, sha6("host-1"));
        // Case-only differences in the scope give different hashes.
        assert_ne!(sha6("d8"), sha6("D8"));
        // An image reference's registry and tag are ignored.
        assert_eq!(parse(&format!("localhost/{n}:latest")), Some(p));
        assert!(parse("bf-ckpt-x").is_none());
        assert!(parse("ubuntu:24.04").is_none());
        assert!(parse("bf-ckpt-own-zzzzzz-abcdef-1-0123456789ab").is_none());
    }

    #[tokio::test]
    async fn the_sweep_spares_others_live_runs_and_young_kept_ones() {
        let scope = crate::owner::scope();
        let here = crate::owner::machine_id();
        let mine = name_for(&scope, here, std::process::id());
        assert_eq!(verdict(&mine, None).await, Sweep::Keep);
        let mut child = tokio::process::Command::new("true").spawn().unwrap();
        let pid = child.id().unwrap();
        child.wait().await.unwrap();
        let dead = name_for(&scope, here, pid);
        assert_eq!(verdict(&dead, None).await, Sweep::Delete);
        let kept = name_for(&scope, here, 0);
        assert_eq!(
            verdict(&kept, Some(Duration::from_secs(60))).await,
            Sweep::Keep
        );
        assert_eq!(verdict(&kept, Some(KEPT_AGE * 2)).await, Sweep::Delete);
        let other = name_for(&scope, "another-host", 1);
        assert_eq!(verdict(&other, None).await, Sweep::Keep);
        assert_eq!(
            verdict(&other, Some(OTHER_HOST_AGE * 2)).await,
            Sweep::Delete
        );
        let foreign = name_for(&format!("{scope}-other"), here, pid);
        assert_eq!(verdict(&foreign, Some(KEPT_AGE * 9)).await, Sweep::Foreign);
    }
}
