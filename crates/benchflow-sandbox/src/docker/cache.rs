//! Keeping BenchFlow's images between runs without filling the disk.
//!
//! Image tags come from content hashes, so a rerun finds its images already
//! built. BenchFlow 0.8 removed every image after each task; here they stay,
//! and after each fresh build the least recently used images BenchFlow built
//! are removed while
//!
//! - the daemon's disk has less free space than
//!   `BENCHFLOW_IMAGE_CACHE_MIN_FREE_GB` (default 10), measured only when the
//!   daemon runs on this machine, or
//! - BenchFlow's images take more than `BENCHFLOW_IMAGE_CACHE_MAX_GB`
//!   (default: no limit).
//!
//! Never removed: images this process prepared (a running job may still need
//! them), images a container uses (`rmi` without `--force` refuses them), and
//! images used in the last 30 minutes. Docker does not record when an image
//! was last used, so BenchFlow keeps that in `<cache>/docker-image-use.json`.

use std::collections::{BTreeMap, HashSet};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use chrono::{DateTime, Utc};

use super::cli::Cli;
use crate::{OWNER_LABEL, SandboxError, owner};

const MIN_FREE_ENV: &str = "BENCHFLOW_IMAGE_CACHE_MIN_FREE_GB";
const MAX_SIZE_ENV: &str = "BENCHFLOW_IMAGE_CACHE_MAX_GB";
const DEFAULT_MIN_FREE_GB: f64 = 10.0;
/// Images used more recently than this are kept, whatever the disk says.
pub(crate) const RECENT: Duration = Duration::from_secs(30 * 60);
const GB: f64 = 1_000_000_000.0;
/// Bound on the CLI calls a trim makes.
const CALL_TIMEOUT: Duration = Duration::from_secs(60);

/// What a trim removed.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct Trimmed {
    pub removed: usize,
    pub bytes: u64,
}

/// When to remove images.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Policy {
    pub min_free: Option<u64>,
    pub max_size: Option<u64>,
}

impl Policy {
    pub fn from_env() -> Self {
        let gb = |name: &str| {
            std::env::var(name)
                .ok()
                .and_then(|v| v.trim().parse::<f64>().ok())
                .filter(|v| v.is_finite() && *v >= 0.0)
        };
        Self {
            min_free: Some(gb(MIN_FREE_ENV).unwrap_or(DEFAULT_MIN_FREE_GB))
                .filter(|g| *g > 0.0)
                .map(|g| (g * GB) as u64),
            max_size: gb(MAX_SIZE_ENV).map(|g| (g * GB) as u64),
        }
    }
}

/// One BenchFlow image on the daemon.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Cached {
    pub id: String,
    pub tags: Vec<String>,
    pub size: u64,
    pub last_used: DateTime<Utc>,
    /// Whether `last_used` is a recorded use; else it is the build time.
    pub used_recorded: bool,
}

/// The images to remove, least recently used first: enough to bring free
/// space up to the policy's minimum and BenchFlow's total under its maximum.
/// `protected` ids are never chosen.
pub(crate) fn plan(
    images: &[Cached],
    free: Option<u64>,
    policy: &Policy,
    protected: &HashSet<String>,
    now: DateTime<Utc>,
) -> Vec<usize> {
    let total: u64 = images.iter().map(|i| i.size).sum();
    let need_free = match (policy.min_free, free) {
        (Some(min), Some(free)) => min.saturating_sub(free),
        _ => 0,
    };
    let need_total = policy.max_size.map_or(0, |max| total.saturating_sub(max));
    let need = need_free.max(need_total);
    if need == 0 {
        return Vec::new();
    }
    let mut order: Vec<usize> = (0..images.len())
        .filter(|&i| {
            let image = &images[i];
            !protected.contains(&image.id)
                && now
                    .signed_duration_since(image.last_used)
                    .to_std()
                    .unwrap_or_default()
                    >= RECENT
        })
        .collect();
    order.sort_by_key(|&i| images[i].last_used);
    let mut freed = 0u64;
    let mut chosen = Vec::new();
    for i in order {
        if freed >= need {
            break;
        }
        freed += images[i].size;
        chosen.push(i);
    }
    chosen
}

/// Image ids this process prepared or used, and still needs.
fn in_use() -> &'static Mutex<HashSet<String>> {
    static SET: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    SET.get_or_init(|| Mutex::new(HashSet::new()))
}

/// The ids behind each prepared reference (an agent layer and its base).
fn ids_by_reference() -> &'static Mutex<BTreeMap<String, Vec<String>>> {
    static MAP: OnceLock<Mutex<BTreeMap<String, Vec<String>>>> = OnceLock::new();
    MAP.get_or_init(|| Mutex::new(BTreeMap::new()))
}

/// Remember that `reference` stands on `ids`, so releasing it frees them.
pub(super) fn note_reference(reference: &str, ids: Vec<String>) {
    if let Ok(mut map) = ids_by_reference().lock() {
        map.insert(reference.to_string(), ids);
    }
}

/// The run needs `reference` no more: its images may be trimmed, unless
/// another reference this process still holds stands on them.
pub(super) fn release(reference: &str) {
    let (Ok(mut map), Ok(mut set)) = (ids_by_reference().lock(), in_use().lock()) else {
        return;
    };
    let Some(ids) = map.remove(reference) else {
        return;
    };
    let still_held: HashSet<&String> = map.values().flatten().collect();
    for id in &ids {
        if !still_held.contains(id) {
            set.remove(id);
        }
    }
}

fn use_file() -> PathBuf {
    let dir = if let Some(dir) = std::env::var_os("BENCHFLOW_CACHE_DIR") {
        PathBuf::from(dir)
    } else if let Some(xdg) = std::env::var_os("XDG_CACHE_HOME").filter(|v| !v.is_empty()) {
        PathBuf::from(xdg).join("benchflow")
    } else {
        dirs::home_dir()
            .map(|h| h.join(".cache"))
            .unwrap_or_else(std::env::temp_dir)
            .join("benchflow")
    };
    dir.join("docker-image-use.json")
}

fn read_uses() -> BTreeMap<String, DateTime<Utc>> {
    std::fs::read_to_string(use_file())
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_default()
}

/// Record that this process uses the image `id` now.
pub(super) fn note_use(id: &str) {
    if id.is_empty() {
        return;
    }
    if let Ok(mut set) = in_use().lock() {
        set.insert(id.to_string());
    }
    // Unit tests drive a fake daemon; its ids stay out of the user's cache.
    if cfg!(test) {
        return;
    }
    let now = Utc::now();
    let mut uses = read_uses();
    // Forget images not used for 90 days: they are gone or will be first out.
    uses.retain(|_, used| now.signed_duration_since(*used) < chrono::Duration::days(90));
    uses.insert(id.to_string(), now);
    let path = use_file();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let tmp = path.with_extension("json.tmp");
    if let Ok(text) = serde_json::to_string(&uses)
        && std::fs::write(&tmp, text).is_ok()
    {
        let _ = std::fs::rename(&tmp, &path);
    }
}

/// A budget in GB as set: `10`, `0.5`, `0.01`.
fn gb_text(bytes: u64) -> String {
    let gb = bytes as f64 / GB;
    let text = format!("{gb:.2}");
    text.trim_end_matches('0').trim_end_matches('.').to_string()
}

/// Remove least recently used BenchFlow images when the policy says so.
/// Best effort: any failure leaves the images where they are.
pub(super) async fn trim(cli: &Cli, log: &(dyn for<'a> Fn(&'a str) + Send + Sync)) -> Trimmed {
    let policy = Policy::from_env();
    if policy.min_free.is_none() && policy.max_size.is_none() {
        return Trimmed::default();
    }
    let images = match list_checked(cli).await {
        Ok(images) if !images.is_empty() => images,
        Ok(_) => return Trimmed::default(),
        Err(e) => {
            // Said once per fresh build, so a broken listing is never silent.
            log(&format!(
                "image cache: could not list BenchFlow's images, so none were removed: {e}"
            ));
            return Trimmed::default();
        }
    };
    let free = free_bytes(cli).await;
    let protected = in_use().lock().map(|s| s.clone()).unwrap_or_default();
    let chosen = plan(&images, free, &policy, &protected, Utc::now());
    let mut done = Trimmed::default();
    for i in chosen {
        let image = &images[i];
        // Remove by tag so an image another tool also tagged keeps its other
        // names; without --force, an image a container uses stays.
        if remove(cli, image).await.is_ok() {
            done.removed += 1;
            done.bytes += image.size;
        }
    }
    if done.removed > 0 {
        let why = match (policy.min_free, free) {
            (Some(min), Some(f)) if f < min => {
                format!("to keep {} GB free on the {} disk", gb_text(min), cli.name)
            }
            _ => format!(
                "to keep BenchFlow's images under {} GB ({MAX_SIZE_ENV})",
                gb_text(policy.max_size.unwrap_or_default())
            ),
        };
        log(&format!(
            "image cache: removed {} least recently used BenchFlow image(s), {:.1} GB, {why}",
            done.removed,
            done.bytes as f64 / GB
        ));
    }
    done
}

/// BenchFlow's images on the daemon, with sizes and last use, or why
/// listing them failed.
pub(super) async fn list_checked(cli: &Cli) -> Result<Vec<Cached>, SandboxError> {
    let filter = format!("label={OWNER_LABEL}={}", owner::scope());
    let args: Vec<String> = ["image", "ls", "-q", "--no-trunc", "--filter", &filter]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let out = cli.run(&args, Some(CALL_TIMEOUT)).await?;
    if !out.ok() {
        return Err(cli.daemon_problem(&out).unwrap_or_else(|| {
            SandboxError::Transport(format!(
                "listing BenchFlow's images failed: {}",
                out.message()
            ))
        }));
    }
    let mut ids: Vec<String> = out
        .stdout
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect();
    ids.sort();
    ids.dedup();
    if ids.is_empty() {
        return Ok(Vec::new());
    }
    let mut args: Vec<String> = vec![
        "image".into(),
        "inspect".into(),
        "--format".into(),
        // `range`, not `join`: Docker 29 hands templates the tags as
        // []interface {}, which `join` refuses.
        "{{.Id}}\t{{.Size}}\t{{.Created}}\t{{range .RepoTags}}{{.}},{{end}}".into(),
    ];
    args.extend(ids);
    let out = cli.run(&args, Some(CALL_TIMEOUT)).await?;
    if !out.ok() {
        return Err(SandboxError::Transport(format!(
            "reading BenchFlow's images failed: {}",
            out.message()
        )));
    }
    let uses = read_uses();
    Ok(parse_inspect(&out.stdout, &uses))
}

/// Remove one image by its tags (by id when it has none), never forced: an
/// image a container uses stays, and the CLI's refusal is returned.
pub(super) async fn remove(cli: &Cli, image: &Cached) -> Result<(), String> {
    let mut args = vec!["rmi".to_string()];
    args.extend(image.tags.iter().cloned());
    if image.tags.is_empty() {
        args.push(image.id.clone());
    }
    match cli.run(&args, Some(CALL_TIMEOUT)).await {
        Ok(out) if out.ok() => {
            forget_uses(std::slice::from_ref(&image.id));
            Ok(())
        }
        Ok(out) => Err(out.message()),
        Err(e) => Err(e.to_string()),
    }
}

/// Drop removed images from the last-use record.
fn forget_uses(ids: &[String]) {
    if cfg!(test) {
        return;
    }
    let mut uses = read_uses();
    let before = uses.len();
    uses.retain(|id, _| !ids.contains(id));
    if uses.len() == before {
        return;
    }
    let path = use_file();
    let tmp = path.with_extension("json.tmp");
    if let Ok(text) = serde_json::to_string(&uses)
        && std::fs::write(&tmp, text).is_ok()
    {
        let _ = std::fs::rename(&tmp, &path);
    }
}

/// A tag BenchFlow gives the images it builds (`benchflow/<task>:<digest>`;
/// Podman shows it under `localhost/`).
fn own_tag(tag: &str) -> bool {
    tag.strip_prefix("localhost/")
        .unwrap_or(tag)
        .starts_with("benchflow/")
}

pub(crate) fn parse_inspect(text: &str, uses: &BTreeMap<String, DateTime<Utc>>) -> Vec<Cached> {
    text.lines()
        .filter_map(|line| {
            let mut parts = line.split('\t');
            let id = parts.next()?.trim().to_string();
            let size = parts.next()?.trim().parse::<u64>().ok()?;
            let created = parts
                .next()
                .and_then(|c| DateTime::parse_from_rfc3339(c.trim()).ok())
                .map(|c| c.with_timezone(&Utc))
                .unwrap_or_else(Utc::now);
            let all: Vec<&str> = parts
                .next()
                .unwrap_or_default()
                .split(',')
                .map(str::trim)
                .filter(|t| !t.is_empty() && *t != "<none>:<none>")
                .collect();
            // Labels pass to images built FROM ours, so the owner label alone
            // does not make an image BenchFlow's: a tagged image must carry
            // one of BenchFlow's tags, and only those are removed.
            let tags: Vec<String> = all
                .iter()
                .filter(|t| own_tag(t))
                .map(|t| t.to_string())
                .collect();
            if !all.is_empty() && tags.is_empty() {
                return None;
            }
            let used = uses.get(&id).copied();
            let last_used = used.map_or(created, |u| u.max(created));
            (!id.is_empty()).then_some(Cached {
                id,
                tags,
                size,
                last_used,
                used_recorded: used.is_some(),
            })
        })
        .collect()
}

/// Free bytes on the daemon's disk, when the daemon runs on this machine.
async fn free_bytes(cli: &Cli) -> Option<u64> {
    let remote = std::env::var("DOCKER_HOST")
        .ok()
        .is_some_and(|h| !h.is_empty() && !h.starts_with("unix://"))
        || std::env::var("DOCKER_CONTEXT").is_ok_and(|c| !c.is_empty() && c != "default");
    if remote {
        return None;
    }
    let args: Vec<String> = ["info", "--format", "{{.DockerRootDir}}"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let out = cli.run(&args, Some(CALL_TIMEOUT)).await.ok()?;
    let root = out.stdout.trim();
    if !out.ok() || root.is_empty() || !std::path::Path::new(root).exists() {
        return None;
    }
    let df = tokio::process::Command::new("df")
        .args(["-Pk", root])
        .output()
        .await
        .ok()?;
    parse_df(&String::from_utf8_lossy(&df.stdout))
}

/// Available bytes from `df -Pk` output.
pub(crate) fn parse_df(text: &str) -> Option<u64> {
    let line = text.lines().nth(1)?;
    let available_kb: u64 = line.split_whitespace().nth(3)?.parse().ok()?;
    Some(available_kb.saturating_mul(1024))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(id: &str, gb: f64, hours_ago: i64, now: DateTime<Utc>) -> Cached {
        Cached {
            id: id.into(),
            tags: vec![format!("benchflow/{id}:abc")],
            size: (gb * GB) as u64,
            last_used: now - chrono::Duration::hours(hours_ago),
            used_recorded: true,
        }
    }

    #[test]
    fn budgets_read_as_set() {
        assert_eq!(gb_text(10_000_000_000), "10");
        assert_eq!(gb_text(500_000_000), "0.5");
        assert_eq!(gb_text(10_000_000), "0.01");
    }

    #[test]
    fn low_disk_removes_least_recently_used_first_and_spares_the_protected() {
        let now = Utc::now();
        let images = vec![
            image("new", 2.0, 1, now),
            image("old", 3.0, 48, now),
            image("older", 1.0, 72, now),
            image("busy", 5.0, 96, now),
        ];
        let policy = Policy {
            min_free: Some((10.0 * GB) as u64),
            max_size: None,
        };
        let protected: HashSet<String> = ["busy".to_string()].into();
        // 6 GB free, 10 wanted: 4 GB to free, oldest unprotected first.
        let chosen = plan(&images, Some((6.0 * GB) as u64), &policy, &protected, now);
        let ids: Vec<&str> = chosen.iter().map(|&i| images[i].id.as_str()).collect();
        assert_eq!(ids, ["older", "old"]);
        // Enough room: nothing goes.
        assert!(plan(&images, Some((20.0 * GB) as u64), &policy, &protected, now).is_empty());
        // Unknown free space (a remote daemon) and no budget: nothing goes.
        assert!(plan(&images, None, &policy, &protected, now).is_empty());
    }

    #[test]
    fn a_size_budget_trims_without_knowing_the_disk_and_recent_images_stay() {
        let now = Utc::now();
        let images = vec![
            image("a", 4.0, 10, now),
            image("b", 4.0, 20, now),
            image("c", 4.0, 0, now),
        ];
        let policy = Policy {
            min_free: None,
            max_size: Some((5.0 * GB) as u64),
        };
        // 12 GB of images, 5 allowed: 7 to free; "c" was used minutes ago.
        let chosen = plan(&images, None, &policy, &HashSet::new(), now);
        let ids: Vec<&str> = chosen.iter().map(|&i| images[i].id.as_str()).collect();
        assert_eq!(ids, ["b", "a"]);
    }

    #[test]
    fn releasing_a_layer_keeps_a_base_another_reference_holds() {
        for id in ["sha256:base", "sha256:codex", "sha256:claude"] {
            note_use(id);
        }
        note_reference("t:codex", vec!["sha256:base".into(), "sha256:codex".into()]);
        note_reference(
            "t:claude",
            vec!["sha256:base".into(), "sha256:claude".into()],
        );
        release("t:codex");
        let held = in_use().lock().unwrap().clone();
        assert!(!held.contains("sha256:codex"));
        assert!(
            held.contains("sha256:base"),
            "claude's layer still stands on it"
        );
        release("t:claude");
        let held = in_use().lock().unwrap().clone();
        assert!(!held.contains("sha256:base") && !held.contains("sha256:claude"));
    }

    /// Also guards cc43e0c2's listing: the owner label alone chose what
    /// `cache clean --remove` and the trim delete, with every tag.
    #[test]
    fn reads_inspect_and_df_output() {
        let mut uses = BTreeMap::new();
        let used = DateTime::parse_from_rfc3339("2026-10-10T10:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        uses.insert("sha256:aa".to_string(), used);
        let text = "sha256:aa\t1500000000\t2026-10-01T00:00:00.123456789Z\tbenchflow/hello:abc,benchflow/hello:abc-def\nsha256:bb\t20\t2026-10-02T00:00:00Z\t<none>:<none>\n";
        let images = parse_inspect(text, &uses);
        assert_eq!(images.len(), 2);
        assert_eq!(images[0].tags.len(), 2);
        assert_eq!(images[0].last_used, used);
        assert!(images[1].tags.is_empty());
        // An image built FROM one of ours inherits the owner label: it is
        // not listed, and a shared image loses only BenchFlow's tag.
        let text = "sha256:cc\t1\t2026-10-01T00:00:00Z\tmyapp:1,\nsha256:dd\t1\t2026-10-01T00:00:00Z\tbenchflow/x:h,myapp:2,localhost/benchflow/y:h,\n";
        let images = parse_inspect(text, &uses);
        assert_eq!(images.len(), 1, "{images:?}");
        assert_eq!(images[0].tags, ["benchflow/x:h", "localhost/benchflow/y:h"]);
        let df = "Filesystem 1024-blocks Used Available Capacity Mounted on\n/dev/sda1 100000000 60000000 40000000 60% /\n";
        assert_eq!(parse_df(df), Some(40_000_000 * 1024));
        assert_eq!(parse_df(""), None);
    }
}
