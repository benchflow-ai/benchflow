//! `bench cache`: what BenchFlow keeps between runs (images it built,
//! fetched task sources and dataset checkouts, registry indexes, logs, and
//! the checkpoints `bench branch` kept), with
//! sizes and last use, and `bench cache clean` to remove them. Cleaning
//! prints what it would remove unless `--remove` is given.

use std::time::{Duration, SystemTime};

use benchflow_run::cache::{self, Kind};
use benchflow_sandbox::checkpoint;
use benchflow_sandbox::docker::{CachedImage, DockerProvider};
use serde::Serialize;
use serde_json::json;

use crate::cli::{CacheArgs, CacheCommand, CacheFilter};
use crate::context;
use crate::error::{CliError, Exit, Result};
use crate::ui::{self, bold, dim, fail_mark, plural, ui, warn_mark};
use crate::{note, say};

/// One removable thing: an image on a daemon, a checkpoint, or a cache
/// folder entry.
#[derive(Debug, Clone, Serialize)]
struct Item {
    kind: Kind,
    /// `docker` or `podman` for images; empty otherwise.
    backend: String,
    name: String,
    bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_used: Option<chrono::DateTime<chrono::Local>>,
    /// Why it stays even when cleaning, if it does.
    #[serde(skip_serializing_if = "Option::is_none")]
    kept: Option<String>,
    #[serde(skip)]
    what: What,
}

#[derive(Debug, Clone)]
enum What {
    Image(CachedImage),
    Entry(cache::Entry),
    Checkpoint(benchflow_sandbox::checkpoint::Checkpoint),
}

pub async fn run(args: CacheArgs) -> Result<Exit> {
    match args.command {
        None => list(&args.filter, false, false).await,
        Some(CacheCommand::List(f)) => list(&f, false, false).await,
        Some(CacheCommand::Clean(c)) => list(&c.filter, true, c.remove).await,
    }
}

async fn list(f: &CacheFilter, cleaning: bool, remove: bool) -> Result<Exit> {
    let mut kinds = kinds(&f.only)?;
    // Checkpoints are only cleaned when named: a kept one may be the start
    // of a later fork.
    if cleaning && f.only.is_empty() {
        kinds.retain(|k| *k != Kind::Checkpoints);
    }
    let older_than = f.older_than.as_deref().map(parse_age).transpose()?;
    let root = benchflow_run::config::default_cache_dir();
    let scan = cache::scan(&root);
    let mut items: Vec<Item> = scan
        .entries
        .iter()
        .filter(|e| kinds.contains(&e.kind))
        .map(|e| Item {
            kind: e.kind,
            backend: String::new(),
            name: e.name.clone(),
            bytes: e.bytes,
            last_used: e.last_used,
            kept: None,
            what: What::Entry(e.clone()),
        })
        .collect();
    let mut notes: Vec<String> = Vec::new();
    if kinds.contains(&Kind::Images) {
        for backend in image_backends(&f.sandbox, &mut notes)? {
            let provider = DockerProvider::new(&backend);
            match provider.cached_images().await {
                Ok(images) => {
                    for i in images {
                        let recent = i
                            .last_used
                            .signed_duration_since(chrono::Utc::now())
                            .abs()
                            .to_std()
                            .unwrap_or_default()
                            < DockerProvider::RECENT_USE;
                        items.push(Item {
                            kind: Kind::Images,
                            backend: backend.clone(),
                            name: i.tags.first().cloned().unwrap_or_else(|| short_id(&i.id)),
                            bytes: i.bytes,
                            last_used: Some(i.last_used.with_timezone(&chrono::Local)),
                            kept: recent.then(|| {
                                "used in the last 30 minutes; a run may still need it".into()
                            }),
                            what: What::Image(i),
                        });
                    }
                }
                Err(e) => notes.push(format!("{backend}: its images were not listed: {e}")),
            }
        }
    }
    if kinds.contains(&Kind::Checkpoints) {
        for backend in checkpoint_backends(&f.sandbox)? {
            let provider = match benchflow_sandbox::provider(&backend) {
                Ok(p) => p,
                Err(e) => {
                    notes.push(format!("{backend}: its checkpoints were not listed: {e}"));
                    continue;
                }
            };
            match provider.list_checkpoints().await {
                Ok(listed) => {
                    for l in listed {
                        let busy = checkpoint::in_use(&l.checkpoint.reference).await;
                        let spare = checkpoint::spare_reason(&l.checkpoint.reference)
                            .filter(|_| older_than.is_none());
                        items.push(Item {
                            kind: Kind::Checkpoints,
                            backend: backend.clone(),
                            name: l.checkpoint.reference.clone(),
                            bytes: l.checkpoint.stored_bytes.unwrap_or(0),
                            last_used: l.created.map(|t| t.with_timezone(&chrono::Local)),
                            kept: if busy {
                                Some("the `bench branch` run that took it is still going".into())
                            } else {
                                spare.map(str::to_string)
                            },
                            what: What::Checkpoint(l.checkpoint),
                        });
                    }
                }
                Err(e) => notes.push(format!("{backend}: its checkpoints were not listed: {e}")),
            }
        }
    }
    if let Some(age) = older_than {
        items.retain(|i| i.last_used.is_none_or(|t| older(t, age)));
    }
    items.sort_by(|a, b| {
        (a.kind, &a.backend, b.bytes, &a.name).cmp(&(b.kind, &b.backend, a.bytes, &b.name))
    });

    let mut removed: Vec<&Item> = Vec::new();
    let mut failed: Vec<(&Item, String)> = Vec::new();
    if remove {
        for item in items.iter().filter(|i| i.kept.is_none()) {
            let done = match &item.what {
                What::Image(img) => DockerProvider::new(&item.backend).remove_image(img).await,
                What::Entry(e) => cache::remove(e).map_err(|e| e.to_string()),
                What::Checkpoint(cp) => match benchflow_sandbox::provider(&item.backend) {
                    Ok(p) => match p.delete_checkpoint(cp).await {
                        Ok(true) => Ok(()),
                        Ok(false) => Err(
                            "still in use (a sandbox runs from it) or still being deleted; the sweep takes it later"
                                .to_string(),
                        ),
                        Err(e) => Err(e.to_string()),
                    },
                    Err(e) => Err(e.to_string()),
                },
            };
            match done {
                Ok(()) => removed.push(item),
                Err(e) => failed.push((item, e)),
            }
        }
    }

    if ui().json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "cache_dir": root,
                "items": items,
                "foreign": scan.foreign.iter().map(|e| json!({"name": e.name, "path": e.path, "bytes": e.bytes})).collect::<Vec<_>>(),
                "removed": remove.then(|| removed.iter().map(|i| json!({"kind": i.kind, "backend": i.backend, "name": i.name, "bytes": i.bytes})).collect::<Vec<_>>()),
                "failed": failed.iter().map(|(i, e)| json!({"name": i.name, "error": e})).collect::<Vec<_>>(),
                "notes": notes,
            }))
            .unwrap_or_default()
        );
        return Ok(if failed.is_empty() {
            Exit::Ok
        } else {
            Exit::Failed
        });
    }

    let removable: Vec<&Item> = items.iter().filter(|i| i.kept.is_none()).collect();
    let total: u64 = removable.iter().map(|i| i.bytes).sum();
    let heading = if remove {
        format!(
            "Removed {} · {}",
            plural(removed.len(), "item"),
            ui::bytes(removed.iter().map(|i| i.bytes).sum())
        )
    } else if cleaning {
        format!(
            "Would remove {} · {} {}",
            plural(removable.len(), "item"),
            ui::bytes(total),
            dim("(nothing is removed without --remove)")
        )
    } else {
        format!(
            "BenchFlow keeps {} · {}",
            plural(items.len(), "item"),
            ui::bytes(items.iter().map(|i| i.bytes).sum())
        )
    };
    say!("{}", bold(heading));
    say!("  {} {}", dim("cache folder"), ui::path(&root));
    let kind_of = |i: &Item| {
        if i.backend.is_empty() {
            i.kind.as_str().to_string()
        } else {
            format!("{} ({})", i.kind.as_str(), i.backend)
        }
    };
    let kw = items.iter().map(|i| kind_of(i).len()).max().unwrap_or(0);
    let w = items
        .iter()
        .map(|i| i.name.chars().count())
        .max()
        .unwrap_or(0)
        .min(ui::width().saturating_sub(kw + 30).max(24));
    for i in &items {
        let used = i
            .last_used
            .map(|t| format!("used {}", ui::ago(SystemTime::from(t))))
            .unwrap_or_default();
        // Kept only matters when something is being removed; a plain list
        // marks nothing.
        let kept = i.kept.as_ref().filter(|_| cleaning);
        let mark = match kept {
            Some(_) => format!("{} ", warn_mark()),
            None if failed.iter().any(|(f, _)| std::ptr::eq(*f, i)) => format!("{} ", fail_mark()),
            None => "  ".into(),
        };
        say!(
            "{mark}{:<kw$}  {:<w$}  {:>7}  {}",
            kind_of(i),
            ui::truncate(&i.name, w),
            ui::bytes(i.bytes),
            dim(used),
            kw = kw,
            w = w
        );
        if let Some(why) = kept {
            say!("    {}", dim(format!("kept: {why}")));
        }
    }
    for (i, e) in &failed {
        say!("  {} could not remove {}: {e}", fail_mark(), i.name);
    }
    if !scan.foreign.is_empty() {
        let names: Vec<String> = scan
            .foreign
            .iter()
            .map(|e| format!("{} ({})", e.name, ui::bytes(e.bytes)))
            .collect();
        say!(
            "  {}",
            dim(format!(
                "not written by this bench, left alone: {}",
                names.join(", ")
            ))
        );
    }
    for n in &notes {
        if n.starts_with("daytona:") {
            note!("  {} {n}", dim("note:"));
        } else {
            note!("  {} {n}", warn_mark());
        }
    }
    if !remove && !removable.is_empty() {
        let next = if cleaning {
            "bench cache clean --remove (add --only or --older-than to narrow it)"
        } else {
            "bench cache clean to see what would be removed"
        };
        say!("  {}  {next}", dim("Next"));
    }
    Ok(if failed.is_empty() {
        Exit::Ok
    } else {
        Exit::Failed
    })
}

fn kinds(only: &[String]) -> Result<Vec<Kind>> {
    if only.is_empty() {
        return Ok(Kind::ALL.to_vec());
    }
    only.iter()
        .map(|s| {
            Kind::parse(s.trim()).ok_or_else(|| {
                CliError::usage(format!(
                    "--only takes images, sources, registries, logs, or checkpoints, not `{s}`"
                ))
            })
        })
        .collect()
}

/// The image backends to list: the ones named, else the Docker and Podman
/// CLIs installed here. Daytona keeps no images between runs.
fn image_backends(named: &[String], notes: &mut Vec<String>) -> Result<Vec<String>> {
    if named.is_empty() {
        return Ok(["docker", "podman"]
            .into_iter()
            .filter(|b| benchflow_run::source::which(b).is_some())
            .map(str::to_string)
            .collect());
    }
    let mut out = Vec::new();
    for b in named {
        match b.as_str() {
            "docker" | "podman" => out.push(b.clone()),
            "daytona" => notes.push(
                "daytona: images are built in Daytona for each sandbox and not kept between runs (checkpoints are listed below)".into(),
            ),
            other => {
                return Err(CliError::usage(format!(
                    "no sandbox backend `{other}`"
                ))
                .next("use docker, podman, or daytona"));
            }
        }
    }
    Ok(out)
}

/// The backends whose checkpoints to list: the ones named, else the Docker
/// and Podman CLIs installed here, and Daytona when its key is set.
fn checkpoint_backends(named: &[String]) -> Result<Vec<String>> {
    if named.is_empty() {
        let mut out: Vec<String> = ["docker", "podman"]
            .into_iter()
            .filter(|b| benchflow_run::source::which(b).is_some())
            .map(str::to_string)
            .collect();
        if std::env::var_os("DAYTONA_API_KEY").is_some() {
            out.push("daytona".into());
        }
        return Ok(out);
    }
    for b in named {
        if !matches!(b.as_str(), "docker" | "podman" | "daytona") {
            return Err(CliError::usage(format!("no sandbox backend `{b}`"))
                .next("use docker, podman, or daytona"));
        }
    }
    Ok(named.to_vec())
}

/// A duration that may also be given in days or weeks (`30d`, `2w`).
fn parse_age(text: &str) -> Result<Duration> {
    let t = text.trim();
    for (suffix, secs) in [("d", 86_400u64), ("w", 7 * 86_400)] {
        if let Some(n) = t.strip_suffix(suffix).and_then(|n| n.parse::<u64>().ok()) {
            return Ok(Duration::from_secs(n.saturating_mul(secs)));
        }
    }
    context::parse_duration(t).map_err(|_| {
        CliError::usage(format!(
            "`{text}` is not an age; write it like 30d, 2w, 12h, or 90m"
        ))
    })
}

fn older(t: chrono::DateTime<chrono::Local>, age: Duration) -> bool {
    chrono::Local::now()
        .signed_duration_since(t)
        .to_std()
        .is_ok_and(|d| d >= age)
}

fn short_id(id: &str) -> String {
    id.trim_start_matches("sha256:").chars().take(12).collect()
}

#[cfg(test)]
mod age_tests {
    use super::*;

    /// Guards the D5 fix: `--older-than 99999999999999999w` overflowed
    /// `n * secs` (a panic in debug builds, a wrapped small age in release,
    /// so a prune removed more than asked).
    #[test]
    fn a_huge_age_saturates() {
        assert_eq!(parse_age("2w").unwrap(), Duration::from_secs(14 * 86_400));
        assert_eq!(
            parse_age("99999999999999999w").unwrap(),
            Duration::from_secs(u64::MAX)
        );
    }
}
