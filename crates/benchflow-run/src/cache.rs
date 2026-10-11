//! What `bench` keeps in its cache folder (`~/.cache/benchflow`), for
//! `bench cache`: fetched task sources, dataset checkouts, registry indexes,
//! and logs, each with its size and last use. Images live with the sandbox
//! backend and are listed by the CLI.
//!
//! Only folders and files this version of `bench` writes are listed as
//! removable; anything else in the cache folder (BenchFlow 0.8's task
//! format runtimes, another tool's files) is reported and left alone.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use serde::Serialize;

/// The kinds of things kept.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Kind {
    /// Images built for tasks and agent layers, on a Docker or Podman daemon.
    Images,
    /// Task folders fetched from GitHub (`bench run org/repo/tasks`), and
    /// dataset checkouts pinned to a commit (`bench run -d name@version`).
    Sources,
    /// Dataset registry indexes and the ACP agent registry.
    Registries,
    /// Crash logs and `bench agents check` logs.
    Logs,
}

impl Kind {
    pub const ALL: [Kind; 4] = [Kind::Images, Kind::Sources, Kind::Registries, Kind::Logs];

    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Images => "images",
            Kind::Sources => "sources",
            Kind::Registries => "registries",
            Kind::Logs => "logs",
        }
    }

    pub fn parse(s: &str) -> Option<Kind> {
        Kind::ALL
            .into_iter()
            .find(|k| k.as_str() == s || k.as_str().trim_end_matches('s') == s)
    }
}

/// One thing in the cache folder that can be removed as a unit.
#[derive(Debug, Clone, Serialize)]
pub struct Entry {
    pub kind: Kind,
    /// What it is, for people (`github.com/org/repo@main`).
    pub name: String,
    pub path: PathBuf,
    pub bytes: u64,
    /// The newest modification time of anything in it.
    pub last_used: Option<chrono::DateTime<chrono::Local>>,
}

/// A cache folder read back.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Scan {
    pub root: PathBuf,
    pub entries: Vec<Entry>,
    /// Top-level files and folders `bench` does not write; never removed.
    pub foreign: Vec<Entry>,
}

/// Read the cache folder at `root`. A missing folder is an empty scan.
pub fn scan(root: &Path) -> Scan {
    let mut out = Scan {
        root: root.to_path_buf(),
        ..Scan::default()
    };
    let Ok(top) = std::fs::read_dir(root) else {
        return out;
    };
    let mut names: Vec<(String, PathBuf)> = top
        .filter_map(Result::ok)
        .map(|e| (e.file_name().to_string_lossy().to_string(), e.path()))
        .collect();
    names.sort();
    for (name, path) in names {
        match name.as_str() {
            "sources" => {
                // Each checkout is a git folder: sources/github.com/<org>/
                // <repo>/<ref> for task sources, sources/git/<host>/<path>/
                // <commit> for dataset checkouts.
                for p in checkouts(&path) {
                    let rel = p
                        .strip_prefix(&path)
                        .unwrap_or(&p)
                        .to_string_lossy()
                        .replace('\\', "/");
                    let (repo, reference) = rel.rsplit_once('/').unwrap_or((&rel, ""));
                    let label = match repo.strip_prefix("git/") {
                        Some(repo) => format!("dataset {repo}@{}", short(reference)),
                        None => format!("{repo}@{reference}"),
                    };
                    out.entries.push(entry(Kind::Sources, label, &p));
                }
            }
            "registries" => {
                if dir_bytes(&path) > 0 {
                    out.entries.push(entry(
                        Kind::Registries,
                        "dataset registry indexes".into(),
                        &path,
                    ));
                }
            }
            "acp-registry" => {
                out.entries
                    .push(entry(Kind::Registries, "ACP agent registry".into(), &path));
            }
            "logs" => out
                .entries
                .push(entry(Kind::Logs, "crash logs".into(), &path)),
            "agent-check-logs" => {
                out.entries
                    .push(entry(Kind::Logs, "`bench agents check` logs".into(), &path))
            }
            // Records `bench` reads back, not caches: `bench agents` shows
            // the last checks, and the image cache's last uses.
            "agent-checks.jsonl" | "docker-image-use.json" => {}
            _ => out.foreign.push(entry(Kind::Sources, name.clone(), &path)),
        }
    }
    out
}

/// Remove one entry from disk.
pub fn remove(e: &Entry) -> std::io::Result<()> {
    if crate::source::checkout_in_use(&e.path) || crate::source::checkout_locked(&e.path) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::ResourceBusy,
            "a running bench is using this checkout; clean it when the run is done",
        ));
    }
    if e.path.is_dir() {
        std::fs::remove_dir_all(&e.path)?;
    } else if e.path.exists() {
        std::fs::remove_file(&e.path)?;
    }
    // A checkout's lock files go with it.
    for suffix in [".lock", ".use"] {
        let mut name = e.path.as_os_str().to_owned();
        name.push(suffix);
        let _ = std::fs::remove_file(PathBuf::from(name));
    }
    // Leave no empty parents behind under sources/.
    let mut parent = e.path.parent();
    while let Some(p) = parent {
        if p.file_name().is_some_and(|n| n == "sources") {
            break;
        }
        if std::fs::remove_dir(p).is_err() {
            break;
        }
        parent = p.parent();
    }
    Ok(())
}

/// The git checkouts under `dir`: folders holding `.git`, not looked into.
fn checkouts(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut walk = walkdir::WalkDir::new(dir)
        .min_depth(1)
        .max_depth(8)
        .follow_links(false)
        .sort_by_file_name()
        .into_iter();
    while let Some(Ok(e)) = walk.next() {
        if !e.file_type().is_dir() {
            continue;
        }
        if e.path().join(".git").exists() {
            out.push(e.into_path());
            walk.skip_current_dir();
        }
    }
    out
}

fn entry(kind: Kind, name: String, path: &Path) -> Entry {
    let (bytes, newest) = usage(path);
    Entry {
        kind,
        name,
        path: path.to_path_buf(),
        bytes,
        last_used: newest.map(chrono::DateTime::<chrono::Local>::from),
    }
}

fn dir_bytes(path: &Path) -> u64 {
    usage(path).0
}

/// Bytes on disk and the newest modification time under `path`; links are
/// not followed.
fn usage(path: &Path) -> (u64, Option<SystemTime>) {
    let mut bytes = 0;
    let mut newest: Option<SystemTime> = None;
    for e in walkdir::WalkDir::new(path)
        .follow_links(false)
        .into_iter()
        .filter_map(Result::ok)
    {
        let Ok(m) = e.metadata() else {
            continue;
        };
        if m.is_file() {
            bytes += m.len();
        }
        if let Ok(t) = m.modified() {
            newest = Some(newest.map_or(t, |n| n.max(t)));
        }
    }
    (bytes, newest)
}

fn short(commit: &str) -> &str {
    if commit.len() == 40 && commit.chars().all(|c| c.is_ascii_hexdigit()) {
        &commit[..12]
    } else {
        commit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(root: &Path, rel: &str, body: &str) {
        let p = root.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, body).unwrap();
    }

    #[test]
    fn scan_names_what_bench_keeps_and_leaves_the_rest_alone() {
        let d = tempfile::tempdir().unwrap();
        let r = d.path();
        write(
            r,
            "sources/github.com/org/repo/main/tasks/a/task.md",
            "aaaa",
        );
        write(r, "sources/github.com/org/repo/main/.git/HEAD", "");
        let commit = "sources/git/github.com/org/data/0123456789abcdef0123456789abcdef01234567";
        write(r, &format!("{commit}/x"), "xx");
        write(r, &format!("{commit}/.git/HEAD"), "");
        write(r, "registries/abc.json", "{}");
        write(r, "acp-registry/registry.json", "{}");
        write(r, "logs/crash-1.log", "boom");
        write(r, "agent-check-logs/acp_x.log", "log");
        write(r, "agent-checks.jsonl", "{}\n");
        write(r, "task-formats/robouse/runtime-1/x", "0.8's");
        let s = scan(r);
        let names: Vec<(Kind, &str)> = s
            .entries
            .iter()
            .map(|e| (e.kind, e.name.as_str()))
            .collect();
        assert_eq!(
            names,
            [
                (Kind::Registries, "ACP agent registry"),
                (Kind::Logs, "`bench agents check` logs"),
                (Kind::Logs, "crash logs"),
                (Kind::Registries, "dataset registry indexes"),
                (Kind::Sources, "dataset github.com/org/data@0123456789ab"),
                (Kind::Sources, "github.com/org/repo@main"),
            ]
        );
        let repo = s.entries.iter().find(|e| e.name.contains("@main")).unwrap();
        assert_eq!(repo.bytes, 4);
        assert!(repo.last_used.is_some());
        assert_eq!(s.foreign.len(), 1);
        assert_eq!(s.foreign[0].name, "task-formats");

        // Removing a source leaves no empty parents and nothing else.
        remove(repo).unwrap();
        assert!(!r.join("sources/github.com/org").exists());
        assert!(r.join("sources/git").exists());
        assert!(r.join("task-formats/robouse/runtime-1/x").exists());
        assert!(r.join("agent-checks.jsonl").exists());
    }

    #[test]
    fn kinds_parse_singular_and_plural() {
        assert_eq!(Kind::parse("images"), Some(Kind::Images));
        assert_eq!(Kind::parse("log"), Some(Kind::Logs));
        assert_eq!(Kind::parse("nope"), None);
    }

    #[test]
    fn a_missing_cache_folder_is_empty() {
        let s = scan(Path::new("/nonexistent/benchflow-cache"));
        assert!(s.entries.is_empty() && s.foreign.is_empty());
    }
}
