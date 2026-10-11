//! Where tasks come from: a local folder, or a GitHub repository fetched
//! into a cache with a sparse checkout of just the folder asked for.
//!
//! | Target | Means |
//! |---|---|
//! | `tasks/`, `./x`, `/abs/path` | a local folder (a task or a folder of tasks) |
//! | `org/repo` | the repository's default branch, whole |
//! | `org/repo/path/in/repo` | that folder of the repository |
//! | `org/repo/path@ref` | at a branch, tag, or commit |
//! | `https://github.com/org/repo/tree/<ref>/<path>` | the same, as a URL |
//!
//! A local folder always wins: `bench run a/b` runs `./a/b` when it exists.

use std::path::{Path, PathBuf};
use std::process::Stdio;

use serde::Serialize;
use tokio::process::Command;

/// A resolved target.
#[derive(Debug, Clone, Serialize)]
pub struct Source {
    /// What the user wrote.
    pub target: String,
    /// The local folder to read tasks from.
    pub path: PathBuf,
    /// For a remote source: the repository, path, ref, and commit fetched.
    pub remote: Option<Remote>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct Remote {
    pub repo: String,
    pub path: Option<String>,
    pub reference: Option<String>,
    pub commit: Option<String>,
}

/// Parse a GitHub target (no network).
pub fn parse_remote(target: &str) -> Option<Remote> {
    let t = target.trim();
    if let Some(rest) = t
        .strip_prefix("https://github.com/")
        .or_else(|| t.strip_prefix("http://github.com/"))
        .or_else(|| t.strip_prefix("github.com/"))
    {
        let parts: Vec<&str> = rest.trim_end_matches('/').split('/').collect();
        if parts.len() < 2 {
            return None;
        }
        let repo = format!("{}/{}", parts[0], parts[1].trim_end_matches(".git"));
        if parts.len() >= 4 && (parts[2] == "tree" || parts[2] == "blob") {
            let reference = Some(parts[3].to_string());
            let path = (parts.len() > 4).then(|| parts[4..].join("/"));
            return Some(Remote {
                repo,
                path,
                reference,
                commit: None,
            });
        }
        let path = (parts.len() > 2).then(|| parts[2..].join("/"));
        return Some(Remote {
            repo,
            path,
            reference: None,
            commit: None,
        });
    }
    let t = t.strip_prefix("gh:").unwrap_or(t);
    if t.starts_with('.') || t.starts_with('/') || t.starts_with('~') || t.contains('\\') {
        return None;
    }
    let (body, reference) = match t.rsplit_once('@') {
        Some((b, r)) if !r.is_empty() && !r.contains('/') => (b, Some(r.to_string())),
        _ => (t, None),
    };
    let parts: Vec<&str> = body.split('/').filter(|p| !p.is_empty()).collect();
    if parts.len() < 2 {
        return None;
    }
    let valid = |s: &str| {
        !s.is_empty()
            && s.chars()
                .all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c))
    };
    if !valid(parts[0]) || !valid(parts[1]) {
        return None;
    }
    Some(Remote {
        repo: format!("{}/{}", parts[0], parts[1]),
        path: (parts.len() > 2).then(|| parts[2..].join("/")),
        reference,
        commit: None,
    })
}

/// A target that could not be resolved.
#[derive(Debug, thiserror::Error)]
pub enum SourceError {
    #[error("`{0}` is not a folder here, and not a GitHub repository (org/repo[/path][@ref])")]
    NotFound(String),
    #[error("could not fetch {repo}: {message}")]
    Fetch { repo: String, message: String },
    #[error("git is not installed; it is needed to fetch {0}")]
    NoGit(String),
}

/// Resolve a target to a local folder, fetching it when it is remote.
pub async fn resolve(target: &str, cache: &Path) -> Result<Source, SourceError> {
    let local = expand_home(target);
    if local.exists() {
        return Ok(Source {
            target: target.to_string(),
            path: local,
            remote: None,
        });
    }
    let Some(mut remote) = parse_remote(target) else {
        return Err(SourceError::NotFound(target.to_string()));
    };
    let path = fetch(&mut remote, cache).await?;
    Ok(Source {
        target: target.to_string(),
        path,
        remote: Some(remote),
    })
}

fn expand_home(p: &str) -> PathBuf {
    if let Some(rest) = p.strip_prefix("~/")
        && let Some(home) = dirs::home_dir()
    {
        return home.join(rest);
    }
    PathBuf::from(p)
}

async fn git(args: &[&str], cwd: Option<&Path>) -> Result<String, String> {
    let mut cmd = Command::new("git");
    cmd.args(args)
        .env("GIT_TERMINAL_PROMPT", "0")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(c) = cwd {
        cmd.current_dir(c);
    }
    let out = cmd.output().await.map_err(|e| e.to_string())?;
    if out.status.success() {
        Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
    } else {
        Err(String::from_utf8_lossy(&out.stderr).trim().to_string())
    }
}

/// Hold an exclusive lock on `<dir>.lock` while a checkout in `dir` is
/// created or changed: two runs started together on the same repository
/// and ref would otherwise race on git's own lock files and one would fail
/// ("Unable to create .../sparse-checkout.lock: File exists"). The lock is
/// released when the file is dropped, or when the process dies.
async fn lock_checkout(dir: &Path) -> std::io::Result<std::fs::File> {
    let mut name = dir.as_os_str().to_owned();
    name.push(".lock");
    let path = PathBuf::from(name);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    tokio::task::spawn_blocking(move || {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&path)?;
        file.lock()?;
        Ok(file)
    })
    .await
    .map_err(std::io::Error::other)?
}

/// Clone (or update) a blobless, sparse checkout of the repository into the
/// cache and return the folder asked for.
async fn fetch(remote: &mut Remote, cache: &Path) -> Result<PathBuf, SourceError> {
    let url = format!("https://github.com/{}.git", remote.repo);
    fetch_from(remote, cache, &url).await
}

/// Whether a ref could be a commit sha, full or abbreviated (`414014c`).
fn looks_like_sha(reference: &str) -> bool {
    (7..=40).contains(&reference.len()) && reference.bytes().all(|b| b.is_ascii_hexdigit())
}

async fn fetch_from(remote: &mut Remote, cache: &Path, url: &str) -> Result<PathBuf, SourceError> {
    if which("git").is_none() {
        return Err(SourceError::NoGit(remote.repo.clone()));
    }
    let reference = remote.reference.clone().unwrap_or_else(|| "HEAD".into());
    unsafe_git_value(&remote.repo, &reference, remote.path.as_slice())?;
    let dir = cache
        .join("sources")
        .join("github.com")
        .join(&remote.repo)
        .join(sanitize(&reference));
    let err = |message: String| SourceError::Fetch {
        repo: remote.repo.clone(),
        message,
    };
    let _lock = lock_checkout(&dir).await.map_err(|e| err(e.to_string()))?;
    if !dir.join(".git").exists() {
        std::fs::create_dir_all(&dir).map_err(|e| err(e.to_string()))?;
        git(&["init", "-q"], Some(&dir)).await.map_err(err)?;
        git(&["remote", "add", "origin", url], Some(&dir))
            .await
            .map_err(err)?;
    }
    if let Some(p) = &remote.path {
        git(
            &[
                "sparse-checkout",
                "set",
                "--no-cone",
                &format!("/{p}/"),
                &format!("/{p}"),
            ],
            Some(&dir),
        )
        .await
        .map_err(err)?;
    } else {
        let _ = git(&["sparse-checkout", "disable"], Some(&dir)).await;
    }
    let shallow = git(
        &[
            "fetch",
            "-q",
            "--depth",
            "1",
            "--filter=blob:none",
            "origin",
            &reference,
        ],
        Some(&dir),
    )
    .await;
    match shallow {
        Ok(_) => git(&["checkout", "-q", "--force", "FETCH_HEAD"], Some(&dir))
            .await
            .map_err(err)?,
        // A commit by an abbreviated sha cannot be fetched by name: fetch the
        // history (still without file contents) and find it there.
        Err(m) if m.contains("couldn't find remote ref") && looks_like_sha(&reference) => {
            git(&["fetch", "-q", "--filter=blob:none", "origin"], Some(&dir))
                .await
                .map_err(err)?;
            git(
                &[
                    "checkout",
                    "-q",
                    "--force",
                    &format!("{reference}^{{commit}}"),
                ],
                Some(&dir),
            )
            .await
            .map_err(|_| err(format!("no branch, tag, or commit named `{reference}`")))?
        }
        Err(m) => {
            return Err(err(if m.contains("couldn't find remote ref") {
                format!("no branch, tag, or commit named `{reference}`")
            } else {
                m
            }));
        }
    };
    remote.commit = git(&["rev-parse", "HEAD"], Some(&dir)).await.ok();
    let path = match &remote.path {
        Some(p) => dir.join(p),
        None => dir.clone(),
    };
    if !path.exists() {
        return Err(err(format!(
            "the repository has no folder `{}` at {reference}",
            remote.path.clone().unwrap_or_default()
        )));
    }
    Ok(path)
}

/// A sparse checkout of one repository at one commit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GitCheckout {
    pub dir: PathBuf,
    /// The commit checked out (the full sha, even when `HEAD` was asked for).
    pub commit: String,
}

/// Fetch `paths` of the repository at `url` at `commit` into the cache, as a
/// blobless, sparse checkout: one folder per repository and commit, holding
/// every path asked for so far. A pinned commit already checked out with
/// these paths is reused without touching the network; `HEAD` is fetched
/// again every time.
pub async fn fetch_git(
    url: &str,
    commit: &str,
    paths: &[String],
    cache: &Path,
) -> Result<GitCheckout, SourceError> {
    if which("git").is_none() {
        return Err(SourceError::NoGit(url.to_string()));
    }
    let err = |message: String| SourceError::Fetch {
        repo: url.to_string(),
        message,
    };
    let unpinned = commit.eq_ignore_ascii_case("head");
    let reference = if unpinned { "HEAD" } else { commit };
    let dir = cache
        .join("sources")
        .join("git")
        .join(repo_folder(url))
        .join(sanitize(reference));
    let wanted: Vec<String> = paths
        .iter()
        .map(|p| p.trim_matches('/').to_string())
        .filter(|p| !p.is_empty())
        .collect();
    let whole = wanted.len() < paths.len();
    unsafe_git_value(url, commit, &wanted)?;
    if url.starts_with('-') {
        return Err(err(format!("`{url}` is not a git URL")));
    }
    let _lock = lock_checkout(&dir).await.map_err(|e| err(e.to_string()))?;
    if !dir.join(".git").exists() {
        std::fs::create_dir_all(&dir).map_err(|e| err(e.to_string()))?;
        git(&["init", "-q"], Some(&dir)).await.map_err(err)?;
        git(&["remote", "add", "origin", url], Some(&dir))
            .await
            .map_err(err)?;
    } else if !unpinned
        && git(&["rev-parse", "HEAD"], Some(&dir))
            .await
            .ok()
            .as_deref()
            == Some(commit)
        && wanted.iter().all(|p| dir.join(p).exists())
        && (!whole
            || git(&["config", "core.sparseCheckout"], Some(&dir))
                .await
                .ok()
                .as_deref()
                != Some("true"))
    {
        return Ok(GitCheckout {
            dir,
            commit: commit.to_string(),
        });
    }
    if whole {
        let _ = git(&["sparse-checkout", "disable"], Some(&dir)).await;
    } else {
        // Keep the folders earlier datasets asked for at this commit.
        let mut patterns: Vec<String> = git(&["sparse-checkout", "list"], Some(&dir))
            .await
            .map(|out| out.lines().map(str::to_string).collect())
            .unwrap_or_default();
        for p in &wanted {
            for pat in [format!("/{p}/"), format!("/{p}")] {
                if !patterns.contains(&pat) {
                    patterns.push(pat);
                }
            }
        }
        let mut args = vec!["sparse-checkout", "set", "--no-cone"];
        args.extend(patterns.iter().map(String::as_str));
        git(&args, Some(&dir)).await.map_err(err)?;
    }
    let shallow = git(
        &[
            "fetch",
            "-q",
            "--depth",
            "1",
            "--filter=blob:none",
            "origin",
            reference,
        ],
        Some(&dir),
    )
    .await;
    if let Err(first) = shallow {
        // Some servers refuse to fetch a commit by its sha alone; fetch the
        // history (still without file contents) and find it there.
        if unpinned {
            return Err(err(first));
        }
        git(&["fetch", "-q", "--filter=blob:none", "origin"], Some(&dir))
            .await
            .map_err(|m| err(format!("{first}; then: {m}")))?;
        git(&["checkout", "-q", "--force", commit], Some(&dir))
            .await
            .map_err(|m| {
                err(
                    if m.contains("did not match") || m.contains("reference is not a tree") {
                        format!("the repository has no commit {commit}")
                    } else {
                        m
                    },
                )
            })?;
    } else {
        git(&["checkout", "-q", "--force", "FETCH_HEAD"], Some(&dir))
            .await
            .map_err(err)?;
    }
    let head = git(&["rev-parse", "HEAD"], Some(&dir)).await.map_err(err)?;
    let missing: Vec<&String> = wanted.iter().filter(|p| !dir.join(p).exists()).collect();
    if !missing.is_empty() {
        let shown: Vec<&str> = missing.iter().take(5).map(|s| s.as_str()).collect();
        return Err(err(format!(
            "the repository has no folder {} at {}{}",
            shown.join(", "),
            &head[..head.len().min(12)],
            if missing.len() > 5 {
                format!(" (and {} more)", missing.len() - 5)
            } else {
                String::new()
            }
        )));
    }
    Ok(GitCheckout { dir, commit: head })
}

/// Refuse a ref that git would read as an option (`--upload-pack=<command>`
/// runs a command), and a path that climbs out of the checkout: a dataset
/// registry is someone else's file.
fn unsafe_git_value(repo: &str, reference: &str, paths: &[String]) -> Result<(), SourceError> {
    let err = |message: String| SourceError::Fetch {
        repo: repo.to_string(),
        message,
    };
    if reference.starts_with('-')
        || reference
            .chars()
            .any(|c| c.is_whitespace() || c.is_control())
    {
        return Err(err(format!(
            "`{reference}` is not a branch, tag, or commit"
        )));
    }
    if let Some(p) = paths
        .iter()
        .find(|p| p.starts_with('/') || p.split(['/', '\\']).any(|c| c == ".."))
    {
        return Err(err(format!(
            "the path `{p}` leaves the repository (absolute, or with `..`)"
        )));
    }
    Ok(())
}

/// A cache folder for a git URL: `github.com/org/repo`.
fn repo_folder(url: &str) -> PathBuf {
    let rest = url
        .split_once("://")
        .map(|(_, r)| r)
        .unwrap_or(url)
        .trim_start_matches("git@")
        .replacen(':', "/", 1);
    let rest = rest.trim_end_matches('/').trim_end_matches(".git");
    rest.split('/')
        .filter(|s| !s.is_empty() && *s != "..")
        .map(sanitize)
        .collect()
}

fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || "-_.".contains(c) {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Find a program on PATH.
pub fn which(program: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|d| d.join(program))
        .find(|p| p.is_file())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_short_forms() {
        assert_eq!(
            parse_remote("benchflow-ai/skillsbench/tasks@main"),
            Some(Remote {
                repo: "benchflow-ai/skillsbench".into(),
                path: Some("tasks".into()),
                reference: Some("main".into()),
                commit: None
            })
        );
        assert_eq!(parse_remote("org/repo").unwrap().path, None);
        assert_eq!(parse_remote("./tasks"), None);
        assert_eq!(parse_remote("tasks"), None);
    }

    #[test]
    fn git_urls_map_to_cache_folders() {
        assert_eq!(
            repo_folder("https://github.com/laude-institute/terminal-bench-2.git"),
            PathBuf::from("github.com/laude-institute/terminal-bench-2")
        );
        assert_eq!(
            repo_folder("git@github.com:org/repo.git"),
            PathBuf::from("github.com/org/repo")
        );
        assert_eq!(
            repo_folder("https://huggingface.co/datasets/a/../b"),
            PathBuf::from("huggingface.co/datasets/a/b")
        );
    }

    /// One checkout per repository and commit, many paths, from a local
    /// repository (no network).
    #[tokio::test]
    async fn fetches_a_pinned_commit_sparsely() {
        if which("git").is_none() {
            return;
        }
        let tmp = tempfile::tempdir().unwrap();
        let repo = tmp.path().join("repo");
        for p in ["tasks/a", "tasks/b", "tasks/c"] {
            std::fs::create_dir_all(repo.join(p)).unwrap();
            std::fs::write(repo.join(p).join("task.md"), p).unwrap();
        }
        let run = |args: &[&str]| {
            let ok = std::process::Command::new("git")
                .args([
                    "-c",
                    "user.name=t",
                    "-c",
                    "user.email=t@t",
                    "-c",
                    "commit.gpgsign=false",
                ])
                .args(args)
                .current_dir(&repo)
                .output()
                .unwrap();
            assert!(
                ok.status.success(),
                "{}",
                String::from_utf8_lossy(&ok.stderr)
            );
            String::from_utf8_lossy(&ok.stdout).trim().to_string()
        };
        run(&["init", "-q"]);
        run(&["config", "uploadpack.allowFilter", "true"]);
        run(&["config", "uploadpack.allowAnySHA1InWant", "true"]);
        run(&["add", "."]);
        run(&["commit", "-q", "-m", "tasks"]);
        let sha = run(&["rev-parse", "HEAD"]);
        let url = format!("file://{}", repo.display());
        let cache = tmp.path().join("cache");
        let co = fetch_git(&url, &sha, &["tasks/a".into()], &cache)
            .await
            .unwrap();
        assert_eq!(co.commit, sha);
        assert!(co.dir.join("tasks/a/task.md").is_file());
        assert!(
            !co.dir.join("tasks/b").exists(),
            "sparse: only what was asked"
        );
        // A second dataset at the same commit adds its folders to the same checkout.
        let co2 = fetch_git(&url, &sha, &["tasks/b".into()], &cache)
            .await
            .unwrap();
        assert_eq!(co2.dir, co.dir);
        assert!(
            co.dir.join("tasks/a/task.md").is_file() && co.dir.join("tasks/b/task.md").is_file()
        );
        let e = fetch_git(&url, &sha, &["tasks/zzz".into()], &cache)
            .await
            .unwrap_err();
        assert!(e.to_string().contains("no folder tasks/zzz"), "{e}");

        // Two runs started together on a new cache both get their folders,
        // instead of one failing on git's sparse-checkout.lock.
        let cache = tmp.path().join("cache-2");
        let fetches = (0..6).map(|i| {
            let (url, sha, cache) = (url.clone(), sha.clone(), cache.clone());
            let path = ["tasks/a", "tasks/b", "tasks/c"][i % 3].to_string();
            tokio::spawn(async move { fetch_git(&url, &sha, &[path], &cache).await })
        });
        for f in fetches.collect::<Vec<_>>() {
            f.await.unwrap().unwrap();
        }
        assert!(cache.join("sources/git").is_dir());
    }

    /// `org/repo/path@<short sha>`: an abbreviated commit, as people write
    /// it, cannot be fetched by name, so the history is fetched and the
    /// commit found there.
    #[tokio::test]
    async fn a_target_at_an_abbreviated_commit_is_found() {
        if which("git").is_none() {
            return;
        }
        let tmp = tempfile::tempdir().unwrap();
        let repo = tmp.path().join("repo");
        let run = |args: &[&str]| {
            let out = std::process::Command::new("git")
                .args(["-c", "user.name=t", "-c", "user.email=t@t"])
                .args(["-c", "commit.gpgsign=false"])
                .args(args)
                .current_dir(&repo)
                .output()
                .unwrap();
            assert!(
                out.status.success(),
                "{}",
                String::from_utf8_lossy(&out.stderr)
            );
            String::from_utf8_lossy(&out.stdout).trim().to_string()
        };
        std::fs::create_dir_all(repo.join("tasks/a")).unwrap();
        std::fs::write(repo.join("tasks/a/task.md"), "old").unwrap();
        run(&["init", "-q"]);
        run(&["config", "uploadpack.allowFilter", "true"]);
        run(&["add", "."]);
        run(&["commit", "-q", "-m", "old"]);
        let old = run(&["rev-parse", "HEAD"]);
        std::fs::write(repo.join("tasks/a/task.md"), "new").unwrap();
        run(&["commit", "-q", "-am", "new"]);
        let url = format!("file://{}", repo.display());
        let cache = tmp.path().join("cache");
        let mut remote = parse_remote(&format!("o/r/tasks@{}", &old[..7])).unwrap();
        let path = fetch_from(&mut remote, &cache, &url).await.unwrap();
        assert_eq!(
            std::fs::read_to_string(path.join("a/task.md")).unwrap(),
            "old"
        );
        assert_eq!(remote.commit.as_deref(), Some(old.as_str()));
        // A ref that is not there still says so.
        let mut remote = parse_remote("o/r/tasks@abcdef0").unwrap();
        let e = fetch_from(&mut remote, &cache, &url).await.unwrap_err();
        assert!(
            e.to_string()
                .contains("no branch, tag, or commit named `abcdef0`"),
            "{e}"
        );
    }

    /// A registry names the commit, the URL, and the paths; none of them may
    /// reach git as an option (`--upload-pack` runs a command) or name a
    /// folder outside the checkout.
    #[tokio::test]
    async fn registry_values_never_become_git_options_or_escape_the_checkout() {
        if which("git").is_none() {
            return;
        }
        let tmp = tempfile::tempdir().unwrap();
        let marker = tmp.path().join("ran");
        let url = format!("file://{}", tmp.path().join("nowhere").display());
        let cache = tmp.path().join("cache");
        let evil = format!("--upload-pack=touch {}; false", marker.display());
        let e = fetch_git(&url, &evil, &["tasks".into()], &cache).await;
        assert!(e.is_err());
        assert!(!marker.exists(), "a registry's commit ran as a git option");
        let e = fetch_git(&evil, "HEAD", &["tasks".into()], &cache).await;
        assert!(e.is_err());
        assert!(!marker.exists(), "a registry's URL ran as a git option");
        let e = fetch_git(&url, "HEAD", &["../../outside".into()], &cache)
            .await
            .unwrap_err();
        assert!(e.to_string().contains(".."), "{e}");
        let mut remote = parse_remote(&format!("org/repo@{}", "--upload-pack=x")).unwrap();
        assert!(fetch(&mut remote, &cache).await.is_err());
    }

    #[test]
    fn parses_urls() {
        let r = parse_remote("https://github.com/org/repo/tree/v1.2/a/b").unwrap();
        assert_eq!(r.repo, "org/repo");
        assert_eq!(r.reference.as_deref(), Some("v1.2"));
        assert_eq!(r.path.as_deref(), Some("a/b"));
        let r = parse_remote("https://github.com/org/repo.git").unwrap();
        assert_eq!(r.repo, "org/repo");
    }
}
