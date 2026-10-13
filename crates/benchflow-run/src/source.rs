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
    /// Keeps the checkout in use for as long as the run holds the source.
    #[serde(skip)]
    pub hold: Option<Hold>,
}

/// A shared lock on a checkout's `<dir>.use`, held while a run reads it:
/// `bench cache clean` leaves a checkout in use alone. Released when the
/// last clone is dropped, or when the process ends.
#[derive(Debug, Clone)]
pub struct Hold(#[allow(dead_code)] std::sync::Arc<std::fs::File>);

impl PartialEq for Hold {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}
impl Eq for Hold {}

/// Written after a checkout and its checks succeed, holding the commit, and
/// removed right before the first command that changes the working tree: a
/// checkout that failed partway has none. It also lets a repository git never finds clean (CRLF
/// files under `text eol=lf`, names that differ only in case on macOS) be
/// reused.
fn done_file(dir: &Path) -> PathBuf {
    dir.join(".git").join("benchflow-done")
}

/// Whether no index entry under `wanted` (anywhere, for the whole
/// repository) is still marked skip-worktree, which git status ignores: a
/// folder whose parent an earlier dataset did not ask for (`tasks` after
/// `tasks/a`), or the rest of a folder an interrupted `sparse-checkout set`
/// wrote only partly. Local only: no network.
async fn none_skipped(locked: &LockedGit, wanted: &[String], whole: bool) -> bool {
    if !whole && wanted.is_empty() {
        return true;
    }
    let mut args = vec!["--literal-pathspecs", "ls-files", "-t", "--"];
    if !whole {
        args.extend(wanted.iter().map(String::as_str));
    }
    locked
        .git(&args)
        .await
        .is_ok_and(|out| !out.lines().any(|l| l.starts_with("S ")))
}

/// Whether git's index matches `HEAD` and the working tree matches the
/// index, for a checkout an earlier build wrote without the done marker: a
/// checkout that failed partway (`HEAD` at the commit, files missing) shows
/// changes here. Local only: no network.
async fn status_clean(locked: &LockedGit) -> bool {
    locked
        .git(&[
            "status",
            "--porcelain",
            "--untracked-files=no",
            "--no-renames",
        ])
        .await
        .is_ok_and(|out| out.trim().is_empty())
}

/// Whether a failed checkout failed because the server refused the file
/// contents git asked for after the contents-free fetch, as git 2.34 to 2.50
/// word it: "could not fetch <blob> from promisor remote", "remote error:
/// upload-pack: not our ref", or "RPC failed" with the HTTP error.
fn contents_refused(stderr: &str) -> bool {
    [
        "promisor remote",
        "not our ref",
        "remote error",
        "RPC failed",
    ]
    .iter()
    .any(|s| stderr.contains(s))
}

/// The file a checkout's users hold shared.
fn use_file(dir: &Path) -> PathBuf {
    let mut name = dir.as_os_str().to_owned();
    name.push(".use");
    PathBuf::from(name)
}

fn hold_checkout(dir: &Path) -> std::io::Result<Hold> {
    let file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(use_file(dir))?;
    file.lock_shared()?;
    Ok(Hold(std::sync::Arc::new(file)))
}

/// Whether a fetch is changing the checkout in `dir` right now.
pub fn checkout_locked(dir: &Path) -> bool {
    let mut name = dir.as_os_str().to_owned();
    name.push(".lock");
    let Ok(file) = std::fs::File::open(PathBuf::from(name)) else {
        return false;
    };
    match file.try_lock() {
        Ok(()) => {
            let _ = file.unlock();
            false
        }
        Err(std::fs::TryLockError::WouldBlock) => true,
        Err(std::fs::TryLockError::Error(_)) => false,
    }
}

/// Whether a run holds the checkout in `dir` (`bench cache clean` asks).
pub fn checkout_in_use(dir: &Path) -> bool {
    let Ok(file) = std::fs::File::open(use_file(dir)) else {
        return false;
    };
    match file.try_lock() {
        Ok(()) => {
            let _ = file.unlock();
            false
        }
        Err(std::fs::TryLockError::WouldBlock) => true,
        Err(std::fs::TryLockError::Error(_)) => false,
    }
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
            hold: None,
        });
    }
    let Some(mut remote) = parse_remote(target) else {
        return Err(SourceError::NotFound(target.to_string()));
    };
    let (path, hold) = fetch(&mut remote, cache).await?;
    Ok(Source {
        target: target.to_string(),
        path,
        remote: Some(remote),
        hold: Some(hold),
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

/// How long a git transfer may run below 1 KB/s before git gives up.
const GIT_STALL: std::time::Duration = std::time::Duration::from_secs(60);

/// The longest one git command may take, whatever it does.
const GIT_CALL_LIMIT: std::time::Duration = std::time::Duration::from_secs(30 * 60);

/// [`run_git`] with no settings and no lock.
#[cfg(test)]
async fn git_within(
    args: &[&str],
    cwd: Option<&Path>,
    stall: std::time::Duration,
    limit: std::time::Duration,
) -> Result<String, String> {
    run_git(&[], args, cwd, stall, limit, None).await
}

/// Run git with `settings` (`-c` pairs) before `args`; with `lock`, git and
/// everything it starts inherit that file's lock (see [`LockedGit`]). git
/// sets no timeout of its own: a server (or a proxy) that stopped sending
/// mid-answer held `bench run` forever (E5, GitHub through a stalling
/// proxy). So a transfer slower than 1 KB/s for `stall` is given up (unless
/// this shell sets git's own low-speed variables), and the whole command at
/// `limit`.
async fn run_git(
    settings: &[String],
    args: &[&str],
    cwd: Option<&Path>,
    stall: std::time::Duration,
    limit: std::time::Duration,
    lock: Option<&std::fs::File>,
) -> Result<String, String> {
    let mut cmd = Command::new("git");
    cmd.args(settings)
        .args(args)
        .env("GIT_TERMINAL_PROMPT", "0")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    if std::env::var_os("GIT_HTTP_LOW_SPEED_LIMIT").is_none()
        && std::env::var_os("GIT_HTTP_LOW_SPEED_TIME").is_none()
    {
        cmd.env("GIT_HTTP_LOW_SPEED_LIMIT", "1000").env(
            "GIT_HTTP_LOW_SPEED_TIME",
            stall.as_secs().max(1).to_string(),
        );
    }
    if let Some(c) = cwd {
        cmd.current_dir(c);
    }
    if let Some(file) = lock {
        inherit_lock(&mut cmd, file);
    }
    let out = match tokio::time::timeout(limit, cmd.output()).await {
        Ok(out) => out.map_err(|e| e.to_string())?,
        Err(_) => {
            return Err(format!(
                "`git {}` did not finish within {} s; the network or the server stalled",
                args.first().copied().unwrap_or_default(),
                limit.as_secs()
            ));
        }
    };
    if out.status.success() {
        Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
    } else {
        Err(String::from_utf8_lossy(&out.stderr).trim().to_string())
    }
}

/// Let the child keep `file`'s descriptor across exec. Rust opens files
/// close-on-exec; this clears the flag on the child's own copy of the
/// descriptor (the parent's is untouched), so the flock, which belongs to
/// the open file, is held until the last process that has it exits.
#[cfg(unix)]
fn inherit_lock(cmd: &mut Command, file: &std::fs::File) {
    use std::os::fd::AsRawFd;
    let fd = file.as_raw_fd();
    // SAFETY: the closure runs in the child between fork and exec, where
    // only async-signal-safe calls are allowed: it makes one `fcntl` call
    // and builds an error from errno, which allocates nothing.
    #[allow(unsafe_code)]
    unsafe {
        cmd.pre_exec(move || {
            if libc::fcntl(fd, libc::F_SETFD, 0) == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
}

#[cfg(not(unix))]
fn inherit_lock(_cmd: &mut Command, _file: &std::fs::File) {}

/// The settings every git under a checkout's lock gets, so that nothing it
/// starts outlives it holding the lock: the fsmonitor daemon (lives until
/// stopped), a detached `gc --auto` or `maintenance run --auto`. A cached
/// dataset checkout needs none of them.
const LOCKED_SETTINGS: [&str; 3] = [
    "core.fsmonitor=false",
    "gc.auto=0",
    "maintenance.auto=false",
];

/// Whether a `credential.helper` value runs git's `cache` helper, which
/// starts `git credential-cache--daemon` (15 minutes by default) when it
/// stores a credential: `cache [options]`, a path to
/// `git-credential-cache`, or a shell command (`!…`) that runs it.
fn cache_helper(value: &str) -> bool {
    let value = value.trim();
    if let Some(shell) = value.strip_prefix('!') {
        return shell.contains("credential-cache");
    }
    let program = value.split_whitespace().next().unwrap_or_default();
    program == "cache" || program.ends_with("/git-credential-cache")
}

/// The `-c` settings that give git the credential helpers in `listing`
/// (`git config --null --get-regexp` of every `credential.helper` and
/// `credential.<url>.helper`, in the order git reads them) without the
/// `cache` helper; empty when there is none to drop. An empty
/// `credential.helper` on the command line, which git reads last, clears
/// every helper read so far, those for one URL too; each other entry is
/// then given again under its own key, so git still matches the URL ones
/// to the repository itself and an empty one still clears the list before
/// it. Whether a helper was dropped is returned too.
fn helper_settings(listing: &str) -> (Vec<String>, bool) {
    let entries: Vec<(&str, &str)> = listing
        .split('\0')
        .filter(|e| !e.is_empty())
        .map(|e| e.split_once('\n').unwrap_or((e, "")))
        .collect();
    if !entries.iter().any(|(_, value)| cache_helper(value)) {
        return (Vec::new(), false);
    }
    let mut settings = vec!["-c".to_string(), "credential.helper=".to_string()];
    for (key, value) in entries {
        // `-c` splits at the first `=`: a URL key holding one cannot be
        // given again, and its helper is left out with the cache helper.
        if cache_helper(value) || key.contains('=') {
            continue;
        }
        settings.push("-c".into());
        settings.push(format!("{key}={value}"));
    }
    (settings, true)
}

/// Whether a git failure is a refusal of the credentials (or a lack of
/// them), as git and curl word it.
fn auth_failure(stderr: &str) -> bool {
    let s = stderr.to_ascii_lowercase();
    [
        "authentication failed",
        "could not read username",
        "could not read password",
        "terminal prompts disabled",
        "invalid username or password",
        "permission denied (publickey",
    ]
    .iter()
    .any(|p| s.contains(p))
        || http_status(&s).is_some_and(|code| code == 401 || code == 403)
}

/// The HTTP status in a curl or git message (`The requested URL returned
/// error: 503`, `RPC failed; HTTP 429`), if any.
fn http_status(lowercase: &str) -> Option<u16> {
    ["returned error: ", "http "].iter().find_map(|marker| {
        lowercase.match_indices(marker).find_map(|(at, m)| {
            let digits = lowercase.get(at + m.len()..at + m.len() + 3)?;
            digits
                .bytes()
                .all(|b| b.is_ascii_digit())
                .then(|| digits.parse().ok())
                .flatten()
        })
    })
}

/// Whether a failed fetch failed for a reason a later attempt may not meet
/// (decision G-6): no connection, a name that did not resolve, a timeout or
/// a stall, a connection cut off (early EOF, RPC failed, reset), or an
/// HTTP 5xx or 429. Never a missing ref, refused or missing credentials, a
/// local disk or permission error, a missing repository, or the whole
/// command's time limit, which a retry would only repeat.
fn network_failure(stderr: &str) -> bool {
    let s = stderr.to_ascii_lowercase();
    let never = [
        "couldn't find remote ref",
        "no space left",
        "disk quota",
        "read-only file system",
        "permission denied",
        "repository not found",
        "does not appear to be a git repository",
        "certificate",
        "did not finish within",
    ];
    if auth_failure(&s) || never.iter().any(|p| s.contains(p)) {
        return false;
    }
    if let Some(code) = http_status(&s) {
        return code >= 500 || code == 429;
    }
    [
        "could not resolve",
        "couldn't resolve",
        "temporary failure in name resolution",
        "name or service not known",
        "failed to connect",
        "could not connect",
        "couldn't connect",
        "connection refused",
        "connection reset",
        "connection timed out",
        "operation timed out",
        "timed out",
        "operation too slow",
        "early eof",
        "rpc failed",
        "unexpected disconnect",
        "the remote end hung up unexpectedly",
        "network is unreachable",
        "empty reply from server",
        "transfer closed",
        "ssl_error",
        "ssl_read",
        "ssl_connect",
        "gnutls",
    ]
    .iter()
    .any(|p| s.contains(p))
}

/// The waits before a fetch's second and third attempts (decision G-6):
/// one GitHub connect failure cost a whole oracle job (E4 6.4). Shorter
/// under test.
const FETCH_BACKOFF: [std::time::Duration; 2] = if cfg!(test) {
    [
        std::time::Duration::from_millis(20),
        std::time::Duration::from_millis(80),
    ]
} else {
    [
        std::time::Duration::from_secs(2),
        std::time::Duration::from_secs(8),
    ]
};

/// Call `fetch` until it succeeds or fails for a reason that is not the
/// network's, at most once more per wait in `backoff`, saying so each time.
async fn retrying<F, Fut>(backoff: &[std::time::Duration], mut fetch: F) -> Result<String, String>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<String, String>>,
{
    let mut tried = 0;
    loop {
        match fetch().await {
            Err(m) if tried < backoff.len() && network_failure(&m) => {
                let wait = backoff[tried];
                tried += 1;
                eprintln!(
                    "  git fetch failed ({}); trying again in {} s (attempt {} of {})",
                    m.lines().last().unwrap_or_default().trim(),
                    wait.as_secs_f32(),
                    tried + 1,
                    backoff.len() + 1
                );
                tokio::time::sleep(wait).await;
            }
            done => return done,
        }
    }
}

/// git in a checkout while `bench` holds its lock (`<dir>.lock`, decision
/// G-4). Every git started here, and everything it starts, inherits the
/// lock, so a git still running after its `bench` was killed (an OOM kill,
/// `kill -9`) keeps the checkout locked: the next run waits for it instead
/// of finding the lock free (review FINAL-B 2). That makes it safe to
/// delete the lock files a killed git left (review FETCH-E-B), which
/// [`LockedGit::new`] does. The price is that nothing git starts may
/// outlive it, so git runs with [`LOCKED_SETTINGS`] and without the `cache`
/// credential helper (other helpers stay): a private repository whose only
/// helper is `cache` fails to authenticate, and the error says so.
struct LockedGit {
    lock: std::fs::File,
    dir: PathBuf,
    settings: Vec<String>,
    /// The `cache` credential helper was left out.
    no_cache: bool,
}

impl LockedGit {
    /// Take the lock on `dir` (waiting at most [`LOCK_WAIT`]), then delete
    /// the lock files a killed git left in it.
    async fn new(dir: &Path) -> std::io::Result<Self> {
        let lock = lock_checkout(dir).await?;
        remove_stale_git_locks(dir);
        Ok(Self {
            lock,
            dir: dir.to_path_buf(),
            settings: LOCKED_SETTINGS
                .iter()
                .flat_map(|s| ["-c".to_string(), s.to_string()])
                .collect(),
            no_cache: false,
        })
    }

    /// Read the credential helpers git would use in the checkout (its own
    /// repository's settings included, so after `git init`), and leave the
    /// `cache` helper out of every later call.
    async fn read_helpers(&mut self) {
        let listing = run_git(
            &[],
            &[
                "config",
                "--null",
                "--get-regexp",
                r"^credential\.(.*\.)?helper$",
            ],
            Some(&self.dir),
            GIT_STALL,
            GIT_CALL_LIMIT,
            None,
        )
        .await
        .unwrap_or_default();
        let (helpers, dropped) = helper_settings(&listing);
        self.settings.extend(helpers);
        self.no_cache = dropped;
    }

    async fn git(&self, args: &[&str]) -> Result<String, String> {
        run_git(
            &self.settings,
            args,
            Some(&self.dir),
            GIT_STALL,
            GIT_CALL_LIMIT,
            Some(&self.lock),
        )
        .await
        .map_err(|m| {
            if self.no_cache && auth_failure(&m) {
                format!(
                    "{m}\n(bench leaves git's `cache` credential helper out of its cached \
                     checkouts, because its daemon would keep the checkout locked for minutes; \
                     if only that helper held these credentials, give them to another, such as \
                     `store` or your system's keychain)"
                )
            } else {
                m
            }
        })
    }

    /// A `git fetch`, tried again after a network failure (decision G-6).
    async fn fetch(&self, args: &[&str]) -> Result<String, String> {
        retrying(&FETCH_BACKOFF, || self.git(args)).await
    }
}

/// Delete the `*.lock` files under `dir/.git` that a git killed while
/// changing the checkout left (`index.lock`, `info/sparse-checkout.lock`,
/// `shallow.lock`, ref locks): SIGKILL from the time limit, an OOM kill or
/// a reboot skips git's own cleanup, and every later change failed with
/// "File exists" until someone deleted them (reviews FETCH-E-B, FINAL-A 1).
/// Only while holding the checkout's lock, which every git `bench` starts
/// there inherits ([`LockedGit`]), so no git of ours is running in it.
/// Never through a symbolic link: a linked `.git`, folder or lock file is
/// left alone.
fn remove_stale_git_locks(dir: &Path) {
    let git_dir = dir.join(".git");
    if !std::fs::symlink_metadata(&git_dir).is_ok_and(|m| m.is_dir()) {
        return;
    }
    for entry in walkdir::WalkDir::new(&git_dir)
        .follow_links(false)
        .follow_root_links(false)
        .into_iter()
        .filter_map(Result::ok)
    {
        if entry.file_type().is_file() && entry.path().extension().is_some_and(|e| e == "lock") {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

/// How long a run waits for another to finish changing a checkout.
const LOCK_WAIT: std::time::Duration = std::time::Duration::from_secs(600);

/// Hold an exclusive lock on `<dir>.lock` while a checkout in `dir` is
/// created or changed: two runs started together on the same repository
/// and ref would otherwise race on git's own lock files and one would fail
/// ("Unable to create .../sparse-checkout.lock: File exists"). The lock is
/// released when the file is dropped and every git started under it has
/// ended ([`LockedGit`]). A run waits at most [`LOCK_WAIT`] for another, and
/// says that it is waiting.
async fn lock_checkout(dir: &Path) -> std::io::Result<std::fs::File> {
    lock_checkout_within(dir, LOCK_WAIT).await
}

async fn lock_checkout_within(
    dir: &Path,
    limit: std::time::Duration,
) -> std::io::Result<std::fs::File> {
    let mut name = dir.as_os_str().to_owned();
    name.push(".lock");
    let path = PathBuf::from(name);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&path)?;
    let started = std::time::Instant::now();
    let mut said = false;
    loop {
        match file.try_lock() {
            Ok(()) => return Ok(file),
            Err(std::fs::TryLockError::Error(e)) => return Err(e),
            Err(std::fs::TryLockError::WouldBlock) => {}
        }
        if started.elapsed() >= limit {
            return Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                format!(
                    "{} was held for {}s by another bench process fetching into the same \
                     checkout, or by a git it started that is still running (perhaps one left \
                     after its bench process was killed, or a helper that git started); run \
                     again when it is done (`lsof {}` names the process; the lock goes when \
                     it ends)",
                    path.display(),
                    limit.as_secs(),
                    path.display()
                ),
            ));
        }
        if !said {
            said = true;
            eprintln!(
                "  waiting for another bench process, or a git it started, to finish with {} \
                 (at most {}s)",
                dir.display(),
                limit.as_secs()
            );
        }
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    }
}

/// Fetch the folder asked for into the cache and return it, held for the
/// run: the ref is resolved to a commit first, and each commit gets a
/// checkout of its own, so a run keeps the commit it resolved even when the
/// branch moves or another run asks for another folder.
async fn fetch(remote: &mut Remote, cache: &Path) -> Result<(PathBuf, Hold), SourceError> {
    let url = format!("https://github.com/{}.git", remote.repo);
    fetch_from(remote, cache, &url).await
}

/// Whether a ref could be a commit sha, full or abbreviated (`414014c`).
fn looks_like_sha(reference: &str) -> bool {
    (7..=40).contains(&reference.len()) && reference.bytes().all(|b| b.is_ascii_hexdigit())
}

async fn fetch_from(
    remote: &mut Remote,
    cache: &Path,
    url: &str,
) -> Result<(PathBuf, Hold), SourceError> {
    if which("git").is_none() {
        return Err(SourceError::NoGit(remote.repo.clone()));
    }
    let reference = remote.reference.clone().unwrap_or_else(|| "HEAD".into());
    let err = |message: String| SourceError::Fetch {
        repo: remote.repo.clone(),
        message,
    };
    if let Some(part) = remote.repo.split('/').find(|p| only_dots(p)) {
        return Err(err(format!("`{part}` is not a repository name")));
    }
    unsafe_git_value(&remote.repo, &reference, remote.path.as_slice())?;
    let commit = resolve_commit(url, &reference, cache).await.map_err(err)?;
    let paths = vec![remote.path.clone().unwrap_or_default()];
    let co = fetch_git(url, &commit, &paths, cache)
        .await
        .map_err(|e| match e {
            SourceError::Fetch { message, .. } => err(message),
            other => other,
        })?;
    remote.commit = Some(co.commit.clone());
    let path = match &remote.path {
        Some(p) => co.dir.join(p),
        None => co.dir.clone(),
    };
    if !path.exists() {
        return Err(err(format!(
            "the repository has no folder `{}` at {reference}",
            remote.path.clone().unwrap_or_default()
        )));
    }
    Ok((path, co.hold))
}

/// The full sha `reference` names in the repository at `url`, found in a
/// blobless clone kept per repository (`<repo>/.resolve`). A ref that looks
/// like a sha must resolve to a commit starting with it: a branch or tag of
/// that name does not count, nor does an annotated tag object; a prefix of
/// two commits is reported as ambiguous.
async fn resolve_commit(url: &str, reference: &str, cache: &Path) -> Result<String, String> {
    if reference.len() == 40 && looks_like_sha(reference) {
        return Ok(reference.to_ascii_lowercase());
    }
    let dir = cache
        .join("sources")
        .join("git")
        .join(repo_folder(url))
        .join(".resolve");
    let mut locked = LockedGit::new(&dir).await.map_err(|e| e.to_string())?;
    if !dir.join(".git").exists() {
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        locked.git(&["init", "-q"]).await?;
        locked.git(&["remote", "add", "origin", url]).await?;
    }
    locked.read_helpers().await;
    let sha_like = looks_like_sha(reference);
    let named = locked
        .fetch(&[
            "fetch",
            "-q",
            "--depth",
            "1",
            "--filter=blob:none",
            "origin",
            reference,
        ])
        .await;
    match named {
        Ok(_) => {
            let sha = locked.git(&["rev-parse", "FETCH_HEAD^{commit}"]).await?;
            if !sha_like || sha.starts_with(&reference.to_ascii_lowercase()) {
                return Ok(sha);
            }
            // A branch or tag named like the sha: not what a sha means.
        }
        Err(m) if !(m.contains("couldn't find remote ref") && sha_like) => {
            return Err(if m.contains("couldn't find remote ref") {
                format!("no branch, tag, or commit named `{reference}`")
            } else {
                m
            });
        }
        Err(_) => {}
    }
    // An abbreviated sha: fetch the history (without file contents) and
    // find the commits that start with it. An earlier depth-1 fetch left
    // this clone shallow, and a fetch does not cross that boundary on its
    // own: unshallow it first.
    let mut args = vec!["fetch", "-q", "--filter=blob:none"];
    if dir.join(".git/shallow").exists() {
        args.push("--unshallow");
    }
    args.extend([
        "origin",
        "+refs/heads/*:refs/remotes/origin/*",
        "+refs/tags/*:refs/tags/*",
    ]);
    locked.fetch(&args).await?;
    let prefix = reference.to_ascii_lowercase();
    let listed = locked
        .git(&["rev-parse", &format!("--disambiguate={prefix}")])
        .await
        .unwrap_or_default();
    let mut commits = Vec::new();
    let mut other = Vec::new();
    for id in listed
        .lines()
        .map(str::trim)
        .filter(|l| l.starts_with(&prefix))
    {
        match locked.git(&["cat-file", "-t", id]).await.as_deref() {
            Ok("commit") => commits.push(id.to_string()),
            Ok(kind) => other.push(format!("{id} ({kind})")),
            Err(_) => {}
        }
    }
    match commits.as_slice() {
        [one] => Ok(one.clone()),
        [] if !other.is_empty() => Err(format!(
            "`{reference}` names {}, not a commit; give the commit's sha",
            other.join(", ")
        )),
        [] => Err(format!("no branch, tag, or commit named `{reference}`")),
        many => Err(format!(
            "`{reference}` is ambiguous: {} commits start with it ({}); give more of the sha, \
             or all 40 characters",
            many.len(),
            many.iter().map(|c| &c[..12]).collect::<Vec<_>>().join(", ")
        )),
    }
}

/// A name made only of dots (`.`, `..`, `...`): never a repository, ref, or
/// folder name here, since it would name a folder of the cache itself.
fn only_dots(part: &str) -> bool {
    !part.is_empty() && part.chars().all(|c| c == '.')
}

/// `path` as a literal for a `sparse-checkout set --no-cone` pattern, which
/// git reads as a gitignore glob: `\`, `*`, `?` and `[` get a backslash, and
/// so does a trailing space (else dropped). Unescaped, `tasks/a[1]` matched
/// `tasks/a1` and left the folder asked for out (review FETCH-E-A, 1).
/// `sparse-checkout list` gives the escaped form back, so the union with
/// earlier patterns still finds its duplicates.
fn sparse_literal(path: &str) -> String {
    let mut out = String::with_capacity(path.len());
    for c in path.chars() {
        if matches!(c, '\\' | '*' | '?' | '[') {
            out.push('\\');
        }
        out.push(c);
    }
    if out.ends_with(' ') {
        out.pop();
        out.push_str("\\ ");
    }
    out
}

/// A sparse checkout of one repository at one commit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GitCheckout {
    pub dir: PathBuf,
    /// The commit checked out (the full sha, even when `HEAD` was asked for).
    pub commit: String,
    /// Keeps the checkout in use while the caller holds it.
    pub hold: Hold,
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
    // `HEAD` moves: resolve it first, so this run gets a checkout of the
    // commit it resolved, which no later run changes.
    let resolved;
    let commit = if commit.eq_ignore_ascii_case("head") {
        resolved = resolve_commit(url, "HEAD", cache).await.map_err(err)?;
        resolved.as_str()
    } else {
        commit
    };
    let reference = commit;
    let dir = cache
        .join("sources")
        .join("git")
        .join(repo_folder(url))
        .join(sanitize(reference));
    let mut locked = LockedGit::new(&dir).await.map_err(|e| err(e.to_string()))?;
    let hold = hold_checkout(&dir).map_err(|e| err(e.to_string()))?;
    let fresh = !dir.join(".git").exists();
    if fresh {
        std::fs::create_dir_all(&dir).map_err(|e| err(e.to_string()))?;
        locked.git(&["init", "-q"]).await.map_err(err)?;
        locked
            .git(&["remote", "add", "origin", url])
            .await
            .map_err(err)?;
    }
    locked.read_helpers().await;
    if !fresh
        && locked.git(&["rev-parse", "HEAD"]).await.ok().as_deref() == Some(commit)
        && wanted.iter().all(|p| dir.join(p).exists())
        && (!whole
            || locked
                .git(&["config", "core.sparseCheckout"])
                .await
                .ok()
                .as_deref()
                != Some("true"))
        && none_skipped(&locked, &wanted, whole).await
        && (std::fs::read_to_string(done_file(&dir)).ok().as_deref() == Some(commit)
            || status_clean(&locked).await)
    {
        if let Some(link) = escaping_link(&dir) {
            return Err(err(link));
        }
        // A complete checkout an earlier build wrote: mark it, so it is
        // reused even if git later finds it unclean.
        let _ = std::fs::write(done_file(&dir), commit);
        return Ok(GitCheckout {
            dir,
            commit: commit.to_string(),
            hold,
        });
    }
    // The fetch changes no file of the checkout, so the done marker stays
    // until the first command that does: a request that fails before then
    // (offline) leaves a complete checkout reusable, which matters for a
    // repository git never finds clean (review FETCH-E-A 2).
    let shallow = locked
        .fetch(&[
            "fetch",
            "-q",
            "--depth",
            "1",
            "--filter=blob:none",
            "origin",
            reference,
        ])
        .await;
    // What to check out: the commit fetched by name is kept by its sha now,
    // since a later fetch (git's own, for file contents) may rewrite
    // FETCH_HEAD.
    let (target, from_history) = match shallow {
        Ok(_) => (
            locked
                .git(&["rev-parse", "FETCH_HEAD^{commit}"])
                .await
                .map_err(err)?,
            false,
        ),
        // A server that could not be reached, or that refused the
        // credentials, would not answer for the history either.
        Err(first) if network_failure(&first) || auth_failure(&first) => {
            return Err(err(first));
        }
        Err(first) => {
            // Some servers refuse to fetch a commit by its sha alone; fetch
            // the history (still without file contents) and find it there.
            locked
                .fetch(&["fetch", "-q", "--filter=blob:none", "origin"])
                .await
                .map_err(|m| err(format!("{first}; then: {m}")))?;
            (commit.to_string(), true)
        }
    };
    let sparse = locked
        .git(&["config", "core.sparseCheckout"])
        .await
        .ok()
        .as_deref()
        == Some("true");
    let started = locked
        .git(&["rev-parse", "-q", "--verify", "HEAD"])
        .await
        .is_ok();
    // From here on the working tree changes: until this checkout succeeds
    // again, it may not be reused.
    let _ = std::fs::remove_file(done_file(&dir));
    if whole {
        // A sparse checkout left as it is would return only the folders
        // earlier datasets asked for, as if they were the whole repository.
        if sparse {
            locked
                .git(&["sparse-checkout", "disable"])
                .await
                .map_err(err)?;
        }
    } else if sparse || !started {
        // Keep the folders earlier datasets asked for at this commit: only
        // lines BenchFlow writes (each starts with `/`), so a `!` line an
        // earlier build let a newline in a path inject is dropped and the
        // checkout heals (review CTRL-PATHS-B).
        let mut patterns: Vec<String> = locked
            .git(&["sparse-checkout", "list"])
            .await
            .map(|out| {
                out.lines()
                    .filter(|l| l.starts_with('/'))
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default();
        for p in &wanted {
            let p = sparse_literal(p);
            for pat in [format!("/{p}/"), format!("/{p}")] {
                if !patterns.contains(&pat) {
                    patterns.push(pat);
                }
            }
        }
        let mut args = vec!["sparse-checkout", "set", "--no-cone"];
        args.extend(patterns.iter().map(String::as_str));
        locked.git(&args).await.map_err(err)?;
    }
    if from_history {
        locked
            .git(&["checkout", "-q", "--force", &target])
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
    } else if let Err(first) = locked.git(&["checkout", "-q", "--force", &target]).await {
        // Some servers send the commit without file contents but refuse the
        // contents asked for afterwards (Hugging Face's datasets); fetch the
        // commit with them. Not after other failures (a full disk, a name
        // this file system refuses, a timeout), which a bigger download
        // cannot fix.
        if !contents_refused(&first) {
            return Err(err(first));
        }
        locked
            .fetch(&[
                "fetch",
                "-q",
                "--depth",
                "1",
                "--no-filter",
                "origin",
                reference,
            ])
            .await
            .map_err(|m| err(format!("{first}; then: {m}")))?;
        locked
            .git(&["checkout", "-q", "--force", &target])
            .await
            .map_err(|m| err(format!("{first}; then: {m}")))?;
    }
    let head = locked.git(&["rev-parse", "HEAD"]).await.map_err(err)?;
    let missing: Vec<&String> = wanted.iter().filter(|p| !dir.join(p).exists()).collect();
    if !missing.is_empty() {
        // Every change git was asked for succeeded, so the checkout is as
        // complete as a successful one; only the folder is not in the
        // repository (a typo). Keep it reusable (review FETCH-E-A 2).
        let _ = std::fs::write(done_file(&dir), &head);
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
    if let Some(link) = escaping_link(&dir) {
        return Err(err(link));
    }
    std::fs::write(done_file(&dir), &head).map_err(|e| err(e.to_string()))?;
    Ok(GitCheckout {
        dir,
        commit: head,
        hold,
    })
}

/// Why the checkout in `dir` may not be used: a symbolic link anywhere in
/// it that leads outside the checkout. A fetched repository is someone
/// else's, and a task folder that resolves to `~/.config/...` would read
/// this machine's files as the task's. Links that stay inside are fine.
fn escaping_link(dir: &Path) -> Option<String> {
    // The whole checkout, every time: a sparse one holds only what was asked
    // for, and a dataset that asks for the root and a folder turns sparse
    // checkout off, so walking only the folders asked for would miss the
    // rest (delta review A3).
    let wanted: &[String] = &[];
    let root = dir.canonicalize().ok()?;
    let outside = |path: &Path| match path.canonicalize() {
        Ok(real) => !real.starts_with(&root),
        // A dangling link reads as nothing.
        Err(_) => false,
    };
    let roots: Vec<PathBuf> = if wanted.is_empty() {
        vec![dir.to_path_buf()]
    } else {
        wanted.iter().map(|p| dir.join(p)).collect()
    };
    for start in roots {
        if outside(&start) {
            return Some(format!(
                "`{}` in the repository leads outside it through a symbolic link; BenchFlow \
                 does not follow it",
                start.strip_prefix(dir).unwrap_or(&start).display()
            ));
        }
        for entry in walkdir::WalkDir::new(&start)
            .follow_links(false)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|e| e.path_is_symlink())
        {
            if outside(entry.path()) {
                return Some(format!(
                    "`{}` in the repository is a symbolic link to a place outside it; BenchFlow \
                     does not follow it",
                    entry
                        .path()
                        .strip_prefix(dir)
                        .unwrap_or(entry.path())
                        .display()
                ));
            }
        }
    }
    None
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
        || only_dots(reference)
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
        .find(|p| p.starts_with('/') || p.split(['/', '\\']).any(only_dots))
    {
        return Err(err(format!(
            "the path `{p}` leaves the repository or names no folder in it (absolute, or with \
             a part made only of dots)"
        )));
    }
    // A newline would add lines of its own to the checkout's sparse
    // patterns (a `!` line hides files other datasets' folders need), and
    // no pattern can escape one (review E4-GLOBS-B).
    if let Some(p) = paths.iter().find(|p| p.chars().any(char::is_control)) {
        return Err(err(format!(
            "the path {p:?} has a control character, which no task folder name may hold"
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

    /// A git server that sends the start of an answer, then nothing.
    async fn stalling_git_server() -> String {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((mut sock, _)) = listener.accept().await {
                tokio::spawn(async move {
                    let mut buf = [0u8; 4096];
                    let _ = sock.read(&mut buf).await;
                    let _ = sock
                        .write_all(
                            b"HTTP/1.1 200 OK\r\nContent-Type: application/x-git-upload-pack-advertisement\r\nContent-Length: 100000\r\n\r\n001e# service=git-upload-pack\n",
                        )
                        .await;
                    tokio::time::sleep(std::time::Duration::from_secs(600)).await;
                });
            }
        });
        format!("http://{addr}/repo.git")
    }

    /// E5 (GitHub through a stalling proxy): git waited forever.
    #[tokio::test]
    async fn a_git_server_that_stalls_is_given_up_on() {
        if std::env::var_os("GIT_HTTP_LOW_SPEED_LIMIT").is_some() {
            return;
        }
        let url = stalling_git_server().await;
        let secs = std::time::Duration::from_secs;
        let started = std::time::Instant::now();
        let r = tokio::time::timeout(
            secs(40),
            git_within(&["ls-remote", &url], None, secs(2), secs(120)),
        )
        .await
        .expect("git gave up on its own");
        assert!(r.is_err(), "{r:?}");
        assert!(started.elapsed() < secs(30), "{:?}", started.elapsed());
        // And the whole command has a limit.
        let r = git_within(&["ls-remote", &url], None, secs(60), secs(1)).await;
        assert!(r.unwrap_err().contains("did not finish within 1 s"));
    }

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
        // A branch fetched first leaves the resolve clone shallow at the new
        // commit (security review F5): the older prefix must still resolve.
        let current = run(&["rev-parse", "--abbrev-ref", "HEAD"]);
        let mut branch = parse_remote(&format!("o/r/tasks@{current}")).unwrap();
        fetch_from(&mut branch, &cache, &url).await.unwrap();
        let mut remote = parse_remote(&format!("o/r/tasks@{}", &old[..7])).unwrap();
        let (path, _hold) = fetch_from(&mut remote, &cache, &url).await.unwrap();
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
        // A branch named like a sha is not a commit starting with it.
        run(&["branch", "deadbee", &old]);
        let mut remote = parse_remote("o/r/tasks@deadbee").unwrap();
        let e = fetch_from(&mut remote, &cache, &url).await.unwrap_err();
        assert!(e.to_string().contains("named `deadbee`"), "{e}");
        // An annotated tag object's sha names a tag, not the commit it tags.
        run(&["tag", "-a", "v1", "-m", "v1", &old]);
        let tag = run(&["rev-parse", "v1"]);
        assert_ne!(tag, old);
        let mut remote = parse_remote(&format!("o/r/tasks@{}", &tag[..9])).unwrap();
        let e = fetch_from(&mut remote, &cache, &url).await.unwrap_err();
        assert!(e.to_string().contains("(tag), not a commit"), "{e}");
    }

    /// Each run gets a checkout of the commit it resolved: a later run of
    /// the same branch after it moved, or of another folder, leaves the
    /// first run's files as they were. The checkout is held while in use,
    /// and `bench cache clean` leaves it alone.
    #[tokio::test]
    async fn a_run_keeps_the_commit_it_resolved() {
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
        for p in ["tasks/a", "tasks/b"] {
            std::fs::create_dir_all(repo.join(p)).unwrap();
            std::fs::write(repo.join(p).join("task.md"), "old").unwrap();
        }
        run(&["init", "-q", "-b", "main"]);
        run(&["config", "uploadpack.allowFilter", "true"]);
        run(&["add", "."]);
        run(&["commit", "-q", "-m", "old"]);
        let url = format!("file://{}", repo.display());
        let cache = tmp.path().join("cache");
        let mut first = parse_remote("o/r/tasks/a@main").unwrap();
        let (a, hold) = fetch_from(&mut first, &cache, &url).await.unwrap();
        // Another run asks for another folder at the same commit.
        let mut other = parse_remote("o/r/tasks/b@main").unwrap();
        let (b, _) = fetch_from(&mut other, &cache, &url).await.unwrap();
        assert_eq!(a.parent(), b.parent());
        assert!(a.join("task.md").exists(), "folders are added, not swapped");
        // main moves; a later run gets the new commit in a checkout of its own.
        std::fs::write(repo.join("tasks/a/task.md"), "new").unwrap();
        run(&["commit", "-q", "-am", "new"]);
        let mut later = parse_remote("o/r/tasks/a@main").unwrap();
        let (a2, _) = fetch_from(&mut later, &cache, &url).await.unwrap();
        assert_ne!(first.commit, later.commit);
        assert_eq!(std::fs::read_to_string(a2.join("task.md")).unwrap(), "new");
        assert_eq!(std::fs::read_to_string(a.join("task.md")).unwrap(), "old");
        // In use while held; free once dropped.
        let checkout = a.parent().unwrap().parent().unwrap().to_path_buf();
        assert!(checkout_in_use(&checkout));
        let entry = crate::cache::Entry {
            kind: crate::cache::Kind::Sources,
            name: "o/r".into(),
            path: checkout.clone(),
            bytes: 0,
            last_used: None,
        };
        assert!(crate::cache::remove(&entry).is_err());
        drop(hold);
        // Other tests' git children may hold an inherited copy of the file
        // for a moment between fork and exec.
        let mut free = false;
        for _ in 0..50 {
            if !checkout_in_use(&checkout) {
                free = true;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        assert!(free, "the checkout is free once its hold is dropped");
        // Security review F6: not while a fetch changes it; then its lock
        // files go with it.
        let changing = lock_checkout(&checkout).await.unwrap();
        assert!(crate::cache::remove(&entry).is_err());
        drop(changing);
        // The same moment for inherited copies of the fetch lock.
        let mut removed = crate::cache::remove(&entry);
        for _ in 0..50 {
            if !matches!(&removed, Err(e) if e.kind() == std::io::ErrorKind::ResourceBusy) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            removed = crate::cache::remove(&entry);
        }
        removed.unwrap();
        let mut lock = checkout.as_os_str().to_owned();
        lock.push(".lock");
        assert!(!checkout.exists() && !use_file(&checkout).exists());
        assert!(!PathBuf::from(lock).exists());
    }

    /// Reviews FETCH-B and FETCH-C-A: a checkout git finds complete is used
    /// as it is, with no network, whichever build wrote it; one that failed
    /// partway (`HEAD` at the commit, a file missing) is checked out again.
    #[tokio::test]
    async fn a_checkout_is_reused_only_when_complete() {
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
        std::fs::write(repo.join("tasks/a/task.md"), "a").unwrap();
        std::fs::write(repo.join("tasks/a/test.sh"), "t").unwrap();
        run(&["init", "-q"]);
        run(&["config", "uploadpack.allowFilter", "true"]);
        run(&["config", "uploadpack.allowAnySHA1InWant", "true"]);
        run(&["add", "."]);
        run(&["commit", "-q", "-m", "a"]);
        let sha = run(&["rev-parse", "HEAD"]);
        let url = format!("file://{}", repo.display());
        let cache = tmp.path().join("cache");
        let co = fetch_git(&url, &sha, &["tasks/a".into()], &cache)
            .await
            .unwrap();
        // Complete, with no marker an earlier build might have written, and
        // the repository out of reach: used as it is.
        let _ = std::fs::remove_file(co.dir.join(".git").join("benchflow-done"));
        let away = tmp.path().join("away");
        std::fs::rename(&repo, &away).unwrap();
        let offline = fetch_git(&url, &sha, &["tasks/a".into()], &cache)
            .await
            .unwrap();
        assert_eq!(offline.dir, co.dir);
        std::fs::rename(&away, &repo).unwrap();
        // What a checkout that failed partway leaves: HEAD at the commit, a
        // file missing, and no done marker (a fetch removes it first).
        let done = co.dir.join(".git").join("benchflow-done");
        std::fs::remove_file(&done).unwrap();
        std::fs::remove_file(co.dir.join("tasks/a/test.sh")).unwrap();
        let again = fetch_git(&url, &sha, &["tasks/a".into()], &cache)
            .await
            .unwrap();
        assert!(
            again.dir.join("tasks/a/test.sh").is_file(),
            "checked out again"
        );
        // Review FETCH-D-B: what an interrupted `sparse-checkout set` leaves:
        // part of the folder written and the rest still marked
        // skip-worktree, which git status ignores.
        let skip = std::process::Command::new("git")
            .args(["update-index", "--skip-worktree", "tasks/a/test.sh"])
            .current_dir(&co.dir)
            .output()
            .unwrap();
        assert!(skip.status.success());
        std::fs::remove_file(&done).unwrap();
        std::fs::remove_file(co.dir.join("tasks/a/test.sh")).unwrap();
        let skipped = fetch_git(&url, &sha, &["tasks/a".into()], &cache)
            .await
            .unwrap();
        assert!(
            skipped.dir.join("tasks/a/test.sh").is_file(),
            "a skipped file is checked out too"
        );
    }

    /// Review FETCH-C-B: a checkout of the whole repository, which another
    /// run may be using, is never narrowed to the folders a later dataset
    /// asks for, even when that dataset names a folder the commit lacks.
    #[tokio::test]
    async fn a_whole_checkout_is_never_narrowed() {
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
        for p in ["tasks/a", "tasks/b"] {
            std::fs::create_dir_all(repo.join(p)).unwrap();
            std::fs::write(repo.join(p).join("task.md"), p).unwrap();
        }
        run(&["init", "-q"]);
        run(&["config", "uploadpack.allowFilter", "true"]);
        run(&["config", "uploadpack.allowAnySHA1InWant", "true"]);
        run(&["add", "."]);
        run(&["commit", "-q", "-m", "tasks"]);
        let sha = run(&["rev-parse", "HEAD"]);
        let url = format!("file://{}", repo.display());
        let cache = tmp.path().join("cache");
        let whole = fetch_git(&url, &sha, &["".into()], &cache).await.unwrap();
        let both = |dir: &Path| {
            dir.join("tasks/a/task.md").is_file() && dir.join("tasks/b/task.md").is_file()
        };
        assert!(both(&whole.dir));
        let e = fetch_git(&url, &sha, &["tasks/zzz".into()], &cache)
            .await
            .unwrap_err();
        assert!(e.to_string().contains("no folder tasks/zzz"), "{e}");
        assert!(both(&whole.dir), "still whole after a missing folder");
        let a = fetch_git(&url, &sha, &["tasks/a".into()], &cache)
            .await
            .unwrap();
        assert_eq!(a.dir, whole.dir);
        assert!(both(&whole.dir), "still whole after a folder of it");
    }

    /// Review FETCH-E-A, 1: sparse patterns are gitignore globs, so the
    /// folder `tasks/a[1]` was fetched as `tasks/a1` and reported missing.
    /// Folders named with glob characters are fetched exactly, alone and
    /// together with folders asked for earlier.
    #[tokio::test]
    async fn a_folder_named_like_a_glob_is_fetched_exactly() {
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
        for p in ["tasks/a[1]", "tasks/a1", "tasks/c*d", "tasks/cxd"] {
            std::fs::create_dir_all(repo.join(p)).unwrap();
            std::fs::write(repo.join(p).join("task.md"), p).unwrap();
        }
        run(&["init", "-q"]);
        run(&["config", "uploadpack.allowFilter", "true"]);
        run(&["config", "uploadpack.allowAnySHA1InWant", "true"]);
        run(&["add", "."]);
        run(&["commit", "-q", "-m", "tasks"]);
        let sha = run(&["rev-parse", "HEAD"]);
        let url = format!("file://{}", repo.display());
        let cache = tmp.path().join("cache");
        let one = fetch_git(&url, &sha, &["tasks/a[1]".into()], &cache)
            .await
            .unwrap();
        assert_eq!(
            std::fs::read_to_string(one.dir.join("tasks/a[1]/task.md")).unwrap(),
            "tasks/a[1]"
        );
        assert!(
            !one.dir.join("tasks/a1").exists(),
            "exactly the folder asked for"
        );
        // A later request keeps the earlier folder (the union reads the
        // escaped patterns back) and adds its own, again exactly.
        let two = fetch_git(&url, &sha, &["tasks/c*d".into()], &cache)
            .await
            .unwrap();
        assert_eq!(two.dir, one.dir);
        assert!(two.dir.join("tasks/a[1]/task.md").is_file());
        assert!(two.dir.join("tasks/c*d/task.md").is_file());
        assert!(!two.dir.join("tasks/cxd").exists());
        assert!(!two.dir.join("tasks/a1").exists());
    }

    #[test]
    fn sparse_patterns_escape_glob_characters() {
        assert_eq!(sparse_literal("tasks/a[1]"), r"tasks/a\[1]");
        assert_eq!(sparse_literal(r"x\y*z?"), r"x\\y\*z\?");
        assert_eq!(sparse_literal("b "), r"b\ ");
        assert_eq!(sparse_literal("plain/name"), "plain/name");
    }

    /// Review FETCH-D-A: a dataset asking for a parent of a folder an
    /// earlier dataset asked for (`tasks` after `tasks/a`) gets the whole
    /// parent, not the earlier folder alone; and a repository git never
    /// finds clean (a CRLF file under `text eol=lf`) is reused after one
    /// fetch, with no network.
    #[tokio::test]
    async fn a_parent_folder_is_fetched_whole_and_an_unclean_repository_reused() {
        if which("git").is_none() {
            return;
        }
        let tmp = tempfile::tempdir().unwrap();
        let repo = tmp.path().join("repo");
        let run = |args: &[&str]| {
            let out = std::process::Command::new("git")
                .args(["-c", "user.name=t", "-c", "user.email=t@t"])
                .args(["-c", "commit.gpgsign=false", "-c", "core.autocrlf=false"])
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
        for p in ["tasks/a", "tasks/b"] {
            std::fs::create_dir_all(repo.join(p)).unwrap();
            std::fs::write(repo.join(p).join("task.md"), p).unwrap();
        }
        std::fs::write(repo.join("tasks/b/crlf.txt"), "one\r\ntwo\r\n").unwrap();
        run(&["init", "-q"]);
        run(&["config", "uploadpack.allowFilter", "true"]);
        run(&["config", "uploadpack.allowAnySHA1InWant", "true"]);
        run(&["add", "."]);
        run(&["commit", "-q", "-m", "tasks"]);
        // Normalization added later, without renormalizing the CRLF file.
        std::fs::write(repo.join(".gitattributes"), "* text eol=lf\n").unwrap();
        run(&["add", ".gitattributes"]);
        run(&["commit", "-q", "-m", "attributes"]);
        let sha = run(&["rev-parse", "HEAD"]);
        let url = format!("file://{}", repo.display());
        let cache = tmp.path().join("cache");
        let a = fetch_git(&url, &sha, &["tasks/a".into()], &cache)
            .await
            .unwrap();
        assert!(!a.dir.join("tasks/b").exists());
        let tasks = fetch_git(&url, &sha, &["tasks".into()], &cache)
            .await
            .unwrap();
        assert_eq!(tasks.dir, a.dir);
        assert!(
            tasks.dir.join("tasks/b/task.md").is_file(),
            "the whole parent, not the earlier folder alone"
        );
        let away = tmp.path().join("away");
        std::fs::rename(&repo, &away).unwrap();
        let offline = fetch_git(&url, &sha, &["tasks".into()], &cache)
            .await
            .unwrap();
        assert_eq!(offline.dir, a.dir);
    }

    /// Reviews FETCH-A and FETCH-B: only a server that refused the contents
    /// (in the words of git 2.34 to 2.50) gets the commit fetched again with
    /// its contents; a bigger download cannot fix a local failure.
    #[test]
    fn only_refused_contents_are_fetched_again() {
        for refused in [
            "fatal: could not fetch 4fdf671c from promisor remote",
            "fatal: git upload-pack: not our ref 4fdf671c",
            "fatal: remote error: upload-pack: not our ref 4fdf671c",
            "error: RPC failed; HTTP 500 curl 22 The requested URL returned error: 500",
        ] {
            assert!(contents_refused(refused), "{refused}");
        }
        for local in [
            "fatal: Unable to create '/c/.git/index.lock': File exists.",
            "error: unable to create file t/a: No space left on device",
            "error: unable to create file t/m\u{fffd}x: Illegal byte sequence",
            "`git checkout` did not finish within 1800 s; the network or the server stalled",
        ] {
            assert!(!contents_refused(local), "{local}");
        }
    }

    /// Security review F2: a repository whose task folder (or a file in
    /// it) is a symbolic link outside the checkout is refused; a link that
    /// stays inside is fine.
    #[tokio::test]
    async fn links_out_of_a_fetched_repository_are_refused() {
        if which("git").is_none() {
            return;
        }
        let tmp = tempfile::tempdir().unwrap();
        let secret = tmp.path().join("secret");
        std::fs::create_dir_all(&secret).unwrap();
        std::fs::write(secret.join("task.md"), "KEY=live").unwrap();
        let repo = tmp.path().join("repo");
        std::fs::create_dir_all(repo.join("tasks/ok")).unwrap();
        std::fs::create_dir_all(repo.join("tasks/file")).unwrap();
        std::fs::write(repo.join("tasks/ok/task.md"), "ok").unwrap();
        std::os::unix::fs::symlink("task.md", repo.join("tasks/ok/alias.md")).unwrap();
        std::os::unix::fs::symlink(&secret, repo.join("tasks/dir")).unwrap();
        std::os::unix::fs::symlink(secret.join("task.md"), repo.join("tasks/file/task.md"))
            .unwrap();
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
        };
        run(&["init", "-q", "-b", "main"]);
        run(&["config", "uploadpack.allowFilter", "true"]);
        run(&["add", "."]);
        run(&["commit", "-q", "-m", "links"]);
        let url = format!("file://{}", repo.display());
        let cache = tmp.path().join("cache");
        let mut ok = parse_remote("o/r/tasks/ok@main").unwrap();
        assert!(fetch_from(&mut ok, &cache, &url).await.is_ok());
        // A dataset asking for the root and a folder turns sparse checkout
        // off; the links elsewhere in the repository are still found.
        let sha = ok.commit.clone().unwrap();
        let e = fetch_git(&url, &sha, &["".into(), "tasks/ok".into()], &cache)
            .await
            .unwrap_err();
        assert!(e.to_string().contains("symbolic link"), "{e}");
        for target in ["o/r/tasks/dir@main", "o/r/tasks/file@main"] {
            let mut remote = parse_remote(target).unwrap();
            let e = fetch_from(&mut remote, &cache, &url).await.unwrap_err();
            assert!(e.to_string().contains("symbolic link"), "{target}: {e}");
        }
    }

    /// Names made only of dots never reach the cache's paths, and a
    /// checkout another process holds is waited for, not forever.
    #[tokio::test]
    async fn dots_are_refused_and_a_held_checkout_is_waited_for_with_a_deadline() {
        if which("git").is_none() {
            return;
        }
        let tmp = tempfile::tempdir().unwrap();
        let cache = tmp.path().join("cache");
        let url = format!("file://{}", tmp.path().join("nowhere").display());
        for target in ["o/r/tasks@..", "o/r/../x@main", "o/r/./x@main"] {
            let mut remote = parse_remote(target).unwrap();
            let e = fetch_from(&mut remote, &cache, &url).await.unwrap_err();
            assert!(
                e.to_string().contains("dots") || e.to_string().contains("is not a branch"),
                "{target}: {e}"
            );
        }
        let mut remote = Remote {
            repo: "../..".into(),
            path: None,
            reference: None,
            commit: None,
        };
        let e = fetch_from(&mut remote, &cache, &url).await.unwrap_err();
        assert!(e.to_string().contains("not a repository name"), "{e}");
        assert!(!tmp.path().join("cache.lock").exists());

        let dir = tmp.path().join("checkout");
        let held = lock_checkout(&dir).await.unwrap();
        let e = lock_checkout_within(&dir, std::time::Duration::from_millis(300))
            .await
            .unwrap_err();
        assert_eq!(e.kind(), std::io::ErrorKind::TimedOut);
        assert!(
            e.to_string().contains("by another bench process")
                && e.to_string().contains("or by a git it started"),
            "{e}"
        );
        drop(held);
        lock_checkout_within(&dir, std::time::Duration::from_millis(300))
            .await
            .unwrap();
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

    /// Review E4-GLOBS-B: a newline in a folder name would add lines of
    /// its own to the checkout's sparse patterns (a `!` line hides files
    /// another dataset's folder needs), so a path with any control
    /// character is refused before git runs.
    #[tokio::test]
    async fn a_path_with_a_control_character_is_refused() {
        if which("git").is_none() {
            return;
        }
        let tmp = tempfile::tempdir().unwrap();
        let cache = tmp.path().join("cache");
        let sha = "0123456789abcdef0123456789abcdef01234567";
        for path in ["tasks/zz\n!/tasks/x/y/task.md", "tasks/a\rb", "tasks/a\tb"] {
            let e = fetch_git("file:///nowhere/repo", sha, &[path.into()], &cache)
                .await
                .unwrap_err();
            assert!(e.to_string().contains("control character"), "{path:?}: {e}");
        }
        assert!(!cache.join("sources").exists(), "nothing was fetched");
    }

    /// Review CTRL-PATHS-B: a cached checkout whose sparse patterns an
    /// earlier build let a newline in a path poison (a `!` line hiding a
    /// file of another folder) heals on the next fetch of that folder.
    #[tokio::test]
    async fn a_poisoned_sparse_pattern_is_dropped() {
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
        std::fs::create_dir_all(repo.join("tasks/x/y")).unwrap();
        std::fs::write(repo.join("tasks/x/task.md"), "x").unwrap();
        std::fs::write(repo.join("tasks/x/y/test.sh"), "t").unwrap();
        run(&["init", "-q"]);
        run(&["config", "uploadpack.allowFilter", "true"]);
        run(&["config", "uploadpack.allowAnySHA1InWant", "true"]);
        run(&["add", "."]);
        run(&["commit", "-q", "-m", "x"]);
        let sha = run(&["rev-parse", "HEAD"]);
        let url = format!("file://{}", repo.display());
        let cache = tmp.path().join("cache");
        let co = fetch_git(&url, &sha, &["tasks/x".into()], &cache)
            .await
            .unwrap();
        // What an earlier build's injected line leaves: a negation that
        // hides one file, and that file skipped in the working tree.
        let sparse = co.dir.join(".git/info/sparse-checkout");
        let mut lines = std::fs::read_to_string(&sparse).unwrap();
        lines.push_str("!/tasks/x/y/test.sh\n");
        std::fs::write(&sparse, lines).unwrap();
        let reapply = std::process::Command::new("git")
            .args(["sparse-checkout", "reapply"])
            .current_dir(&co.dir)
            .output()
            .unwrap();
        assert!(reapply.status.success());
        assert!(!co.dir.join("tasks/x/y/test.sh").exists());
        let again = fetch_git(&url, &sha, &["tasks/x".into()], &cache)
            .await
            .unwrap();
        assert!(
            again.dir.join("tasks/x/y/test.sh").is_file(),
            "the injected line is gone and the file is back"
        );
    }

    /// Run git in `repo` as the tests' origins do (no signing, a fixed
    /// identity), failing the test on an error.
    fn origin_git(repo: &Path, args: &[&str]) -> String {
        let out = std::process::Command::new("git")
            .args(["-c", "user.name=t", "-c", "user.email=t@t"])
            .args(["-c", "commit.gpgsign=false", "-c", "core.autocrlf=false"])
            .args(args)
            .current_dir(repo)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    }

    /// A repository at `repo` holding `files` on branch `main`, fetchable
    /// blobless and by sha; returns its commit.
    fn make_origin(repo: &Path, files: &[(&str, &str)]) -> String {
        for (path, text) in files {
            let path = repo.join(path);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, text).unwrap();
        }
        origin_git(repo, &["init", "-q", "-b", "main"]);
        origin_git(repo, &["config", "uploadpack.allowFilter", "true"]);
        origin_git(repo, &["config", "uploadpack.allowAnySHA1InWant", "true"]);
        origin_git(repo, &["add", "."]);
        origin_git(repo, &["commit", "-q", "-m", "tasks"]);
        origin_git(repo, &["rev-parse", "HEAD"])
    }

    /// Reviews FETCH-E-B and FINAL-A 1: a git killed mid-change (the time
    /// limit's SIGKILL, an OOM kill, a reboot) leaves `index.lock` and
    /// `info/sparse-checkout.lock` in the checkout, or `shallow.lock` in the
    /// `.resolve` clone, and every later run at that commit, at `HEAD`, a
    /// branch or a short sha failed with "File exists" until someone
    /// deleted them. With no git holding the checkout's lock, they go.
    #[tokio::test]
    async fn stale_git_locks_no_longer_block_a_run() {
        if which("git").is_none() {
            return;
        }
        let tmp = tempfile::tempdir().unwrap();
        let repo = tmp.path().join("repo");
        let sha = make_origin(
            &repo,
            &[
                ("tasks/a/task.md", "a"),
                ("tasks/b/task.md", "b"),
                ("tasks/c/task.md", "c"),
                ("tasks/d/task.md", "d"),
            ],
        );
        let url = format!("file://{}", repo.display());
        let cache = tmp.path().join("cache");
        let co = fetch_git(&url, &sha, &["tasks/a".into()], &cache)
            .await
            .unwrap();
        // The `.resolve` clone, as a run at `HEAD` leaves it (shallow).
        fetch_git(&url, "HEAD", &["tasks/a".into()], &cache)
            .await
            .unwrap();
        let resolve = cache
            .join("sources/git")
            .join(repo_folder(&url))
            .join(".resolve");
        assert!(resolve.join(".git/shallow").is_file());
        let plant = || {
            for lock in [
                co.dir.join(".git/index.lock"),
                co.dir.join(".git/info/sparse-checkout.lock"),
                resolve.join(".git/shallow.lock"),
            ] {
                std::fs::write(lock, "").unwrap();
            }
        };
        // Each case on its own, so a failure names every case that fails.
        let mut failed = Vec::new();
        // At a commit: a folder not yet checked out.
        plant();
        match fetch_git(&url, &sha, &["tasks/b".into()], &cache).await {
            Ok(b) => assert!(b.dir.join("tasks/b/task.md").is_file()),
            Err(e) => failed.push(format!("at a commit: {e}")),
        }
        // At `HEAD`.
        plant();
        match fetch_git(&url, "HEAD", &["tasks/c".into()], &cache).await {
            Ok(c) => assert!(c.dir.join("tasks/c/task.md").is_file()),
            Err(e) => failed.push(format!("at HEAD: {e}")),
        }
        // At a branch.
        plant();
        let mut branch = parse_remote("o/r/tasks/d@main").unwrap();
        match fetch_from(&mut branch, &cache, &url).await {
            Ok((d, _)) => assert!(d.join("task.md").is_file()),
            Err(e) => failed.push(format!("at a branch: {e}")),
        }
        // At a short sha.
        plant();
        let mut short = parse_remote(&format!("o/r/tasks/a@{}", &sha[..7])).unwrap();
        match fetch_from(&mut short, &cache, &url).await {
            Ok((a, _)) => {
                assert!(a.join("task.md").is_file());
                assert_eq!(short.commit.as_deref(), Some(sha.as_str()));
            }
            Err(e) => failed.push(format!("at a short sha: {e}")),
        }
        assert!(failed.is_empty(), "{}", failed.join("\n"));
    }

    /// The lock files go only from inside `.git`, and never through a
    /// link: a linked lock file, a linked folder of lock files, a `.git`
    /// that is a link, and a folder named `*.lock` are left as they are.
    #[test]
    fn stale_locks_are_removed_never_through_a_link() {
        let tmp = tempfile::tempdir().unwrap();
        let outside = tmp.path().join("outside");
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(outside.join("keep.lock"), "").unwrap();
        let dir = tmp.path().join("co");
        std::fs::create_dir_all(dir.join(".git/info")).unwrap();
        std::fs::create_dir_all(dir.join(".git/refs/remotes/origin")).unwrap();
        std::fs::create_dir_all(dir.join(".git/dir.lock")).unwrap();
        for stale in [
            "index.lock",
            "info/sparse-checkout.lock",
            "shallow.lock",
            "refs/remotes/origin/main.lock",
        ] {
            std::fs::write(dir.join(".git").join(stale), "").unwrap();
        }
        std::fs::write(dir.join("worktree.lock"), "").unwrap();
        std::os::unix::fs::symlink(outside.join("keep.lock"), dir.join(".git/link.lock")).unwrap();
        std::os::unix::fs::symlink(&outside, dir.join(".git/linked")).unwrap();
        remove_stale_git_locks(&dir);
        for gone in [
            "index.lock",
            "info/sparse-checkout.lock",
            "shallow.lock",
            "refs/remotes/origin/main.lock",
        ] {
            assert!(!dir.join(".git").join(gone).exists(), "{gone}");
        }
        assert!(dir.join(".git/dir.lock").is_dir());
        assert!(dir.join("worktree.lock").is_file(), "only under .git");
        assert!(dir.join(".git/link.lock").symlink_metadata().is_ok());
        assert!(outside.join("keep.lock").is_file());
        // A `.git` that is itself a link is not walked.
        let linked = tmp.path().join("linked-co");
        std::fs::create_dir_all(&linked).unwrap();
        std::os::unix::fs::symlink(&outside, linked.join(".git")).unwrap();
        remove_stale_git_locks(&linked);
        assert!(outside.join("keep.lock").is_file());
    }

    /// Review FETCH-E-A 2: in a repository git never finds clean (a CRLF
    /// file under `text eol=lf`), a request that failed without changing
    /// the checkout (a typo'd folder; a new folder while offline) removed
    /// the done marker, so the folder that was complete needed the network
    /// again. It no longer does.
    #[tokio::test]
    async fn a_failed_request_keeps_a_complete_checkout_reusable_offline() {
        if which("git").is_none() {
            return;
        }
        let tmp = tempfile::tempdir().unwrap();
        let repo = tmp.path().join("repo");
        make_origin(
            &repo,
            &[
                ("tasks/a/task.md", "a"),
                ("tasks/b/crlf.txt", "one\r\ntwo\r\n"),
                ("other/task.md", "o"),
            ],
        );
        // Normalization added later, without renormalizing the CRLF file.
        std::fs::write(repo.join(".gitattributes"), "* text eol=lf\n").unwrap();
        origin_git(&repo, &["add", ".gitattributes"]);
        origin_git(&repo, &["commit", "-q", "-m", "attributes"]);
        let sha = origin_git(&repo, &["rev-parse", "HEAD"]);
        let url = format!("file://{}", repo.display());
        let cache = tmp.path().join("cache");
        let co = fetch_git(&url, &sha, &["tasks".into()], &cache)
            .await
            .unwrap();
        let status = std::process::Command::new("git")
            .args(["status", "--porcelain", "--untracked-files=no"])
            .current_dir(&co.dir)
            .output()
            .unwrap();
        assert!(!status.stdout.is_empty(), "git never finds it clean");
        let away = tmp.path().join("away");
        let offline_run = || async {
            std::fs::rename(&repo, &away).unwrap();
            let r = fetch_git(&url, &sha, &["tasks".into()], &cache).await;
            std::fs::rename(&away, &repo).unwrap();
            r
        };
        // A typo'd folder, online.
        let e = fetch_git(&url, &sha, &["tasks/zzz".into()], &cache)
            .await
            .unwrap_err();
        assert!(e.to_string().contains("no folder tasks/zzz"), "{e}");
        let reused = offline_run().await.unwrap();
        assert_eq!(reused.dir, co.dir);
        // A new folder, offline.
        std::fs::rename(&repo, &away).unwrap();
        let e = fetch_git(&url, &sha, &["other".into()], &cache).await;
        std::fs::rename(&away, &repo).unwrap();
        assert!(e.is_err());
        let reused = offline_run().await.unwrap();
        assert!(reused.dir.join("tasks/b/crlf.txt").is_file());
        // The new folder, online, is added as before.
        let other = fetch_git(&url, &sha, &["other".into()], &cache)
            .await
            .unwrap();
        assert!(other.dir.join("other/task.md").is_file());
    }

    /// The stand-in for a `bench` process that a test kills while its git
    /// runs: fetches what the environment names. Run only by
    /// [`an_orphaned_git_keeps_the_checkout_locked`].
    #[tokio::test]
    #[ignore = "the stand-in parent an_orphaned_git_keeps_the_checkout_locked starts"]
    async fn orphan_standin() {
        let (Ok(url), Ok(sha), Ok(cache)) = (
            std::env::var("G2_STANDIN_URL"),
            std::env::var("G2_STANDIN_SHA"),
            std::env::var("G2_STANDIN_CACHE"),
        ) else {
            return;
        };
        let _ = fetch_git(&url, &sha, &["tasks/a".into()], Path::new(&cache)).await;
    }

    /// Lets every git a slowed `upload-pack` holds go on when dropped.
    struct Release(PathBuf);
    impl Drop for Release {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    /// Review FINAL-B 2: the checkout lock was close-on-exec, so a git
    /// still running after its `bench` was killed held nothing, and
    /// deleting stale lock files could delete its live `index.lock` (two
    /// writers on one index). Git now inherits the lock: the stand-in
    /// `bench` is killed while its `checkout` waits on a slowed
    /// `upload-pack`, and the next fetch waits for the orphan instead of
    /// deleting its lock, then finds the checkout complete.
    #[tokio::test]
    async fn an_orphaned_git_keeps_the_checkout_locked() {
        if which("git").is_none() {
            return;
        }
        let tmp = tempfile::tempdir().unwrap();
        let repo = tmp.path().join("repo");
        let files: Vec<(String, String)> = (0..20)
            .map(|i| (format!("tasks/a/f{i}"), format!("{i}\n")))
            .collect();
        let files: Vec<(&str, &str)> = files
            .iter()
            .map(|(p, t)| (p.as_str(), t.as_str()))
            .collect();
        let sha = make_origin(&repo, &files);
        let url = format!("file://{}", repo.display());
        let cache = tmp.path().join("cache");
        let dir = cache.join("sources/git").join(repo_folder(&url)).join(&sha);
        // The second `upload-pack` (the contents `checkout` asks for) waits
        // while `hold` exists.
        let hold = tmp.path().join("hold");
        std::fs::write(&hold, "").unwrap();
        let release = Release(hold.clone());
        let wrapper = tmp.path().join("slow-upload-pack");
        std::fs::write(
            &wrapper,
            format!(
                "#!/bin/sh\nd='{}'\nn=$(cat \"$d/count\" 2>/dev/null || echo 0)\nn=$((n+1))\n\
                 echo $n > \"$d/count\"\nif [ \"$n\" -ge 2 ]; then\n  touch \"$d/slowed\"\n  \
                 while [ -f \"$d/hold\" ]; do sleep 0.1; done\nfi\nexec git upload-pack \"$@\"\n",
                tmp.path().display()
            ),
        )
        .unwrap();
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&wrapper, std::fs::Permissions::from_mode(0o755)).unwrap();
        let mut standin = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "source::tests::orphan_standin",
                "--exact",
                "--ignored",
                "--test-threads=1",
            ])
            .env("G2_STANDIN_URL", &url)
            .env("G2_STANDIN_SHA", &sha)
            .env("G2_STANDIN_CACHE", &cache)
            .env("GIT_CONFIG_COUNT", "1")
            .env("GIT_CONFIG_KEY_0", "remote.origin.uploadpack")
            .env("GIT_CONFIG_VALUE_0", &wrapper)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let index_lock = dir.join(".git/index.lock");
        let started = std::time::Instant::now();
        while !(tmp.path().join("slowed").exists() && index_lock.exists()) {
            assert!(
                started.elapsed() < std::time::Duration::from_secs(60),
                "the stand-in's checkout never reached the slowed upload-pack"
            );
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        // Kill the stand-in `bench` (SIGKILL: no cleanup runs); its git
        // goes on as an orphan.
        standin.kill().unwrap();
        standin.wait().unwrap();
        assert!(
            checkout_locked(&dir),
            "the orphaned git holds the checkout's lock"
        );
        let second = tokio::spawn({
            let (url, sha, cache) = (url.clone(), sha.clone(), cache.clone());
            async move { fetch_git(&url, &sha, &["tasks/a".into()], &cache).await }
        });
        tokio::time::sleep(std::time::Duration::from_millis(1500)).await;
        assert!(!second.is_finished(), "the second fetch waits");
        assert!(index_lock.exists(), "the orphan's live index.lock is kept");
        drop(release);
        let co = tokio::time::timeout(std::time::Duration::from_secs(60), second)
            .await
            .expect("the second fetch goes on once the orphan ends")
            .unwrap()
            .unwrap();
        for i in 0..20 {
            assert!(co.dir.join(format!("tasks/a/f{i}")).is_file(), "f{i}");
        }
    }

    /// Decision G-6: which failures a fetch is tried again after, in the
    /// words of git and curl.
    #[test]
    fn only_network_failures_are_tried_again() {
        for network in [
            "fatal: unable to access 'https://github.com/o/r.git/': Could not resolve host: github.com",
            "fatal: unable to access 'https://github.com/o/r.git/': Failed to connect to github.com port 443 after 75003 ms: Couldn't connect to server",
            "fatal: unable to access 'https://github.com/o/r.git/': Operation timed out after 300000 milliseconds",
            "error: RPC failed; curl 56 Recv failure: Connection reset by peer\nfatal: early EOF",
            "error: RPC failed; HTTP 502 curl 22 The requested URL returned error: 502",
            "fatal: unable to access 'https://h/r/': The requested URL returned error: 503",
            "fatal: unable to access 'https://h/r/': The requested URL returned error: 429",
            "fatal: the remote end hung up unexpectedly",
            "fatal: unable to access 'https://h/r/': Operation too slow. Less than 1000 bytes/sec transferred the last 60 seconds",
            "ssh: Could not resolve hostname github.com: Temporary failure in name resolution",
            "fatal: unable to access 'https://h/r/': gnutls_handshake() failed: The TLS connection was non-properly terminated.",
        ] {
            assert!(network_failure(network), "{network}");
        }
        for other in [
            "fatal: couldn't find remote ref refs/heads/nope",
            "fatal: Authentication failed for 'https://github.com/o/r.git/'",
            "fatal: could not read Username for 'https://github.com': terminal prompts disabled",
            "fatal: unable to access 'https://h/r/': The requested URL returned error: 403",
            "fatal: unable to access 'https://h/r/': The requested URL returned error: 404",
            "error: RPC failed; HTTP 401 curl 22 The requested URL returned error: 401",
            "git@github.com: Permission denied (publickey).\nfatal: Could not read from remote repository.",
            "fatal: write error: No space left on device\nfatal: early EOF",
            "remote: Repository not found.\nfatal: repository 'https://github.com/o/r.git/' not found",
            "fatal: '/x/repo' does not appear to be a git repository",
            "fatal: unable to access 'https://h/r/': SSL certificate problem: unable to get local issuer certificate",
            "`git fetch` did not finish within 1800 s; the network or the server stalled",
        ] {
            assert!(!network_failure(other), "{other}");
        }
    }

    /// Decision G-6: at most two more attempts, after the waits given, and
    /// only for a network failure.
    #[tokio::test]
    async fn a_fetch_is_tried_again_only_after_a_network_failure() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let backoff = [
            std::time::Duration::from_millis(30),
            std::time::Duration::from_millis(120),
        ];
        let attempts = |stderr: &'static str, succeed_at: usize| async move {
            let calls = AtomicUsize::new(0);
            let started = std::time::Instant::now();
            let r = retrying(&backoff, || {
                let n = calls.fetch_add(1, Ordering::SeqCst) + 1;
                async move {
                    if n == succeed_at {
                        Ok("ok".to_string())
                    } else {
                        Err(stderr.to_string())
                    }
                }
            })
            .await;
            (calls.load(Ordering::SeqCst), r, started.elapsed())
        };
        let down = "fatal: unable to access 'https://h/r/': Could not resolve host: h";
        let (calls, r, took) = attempts(down, 0).await;
        assert_eq!(calls, 3);
        assert_eq!(r.unwrap_err(), down);
        assert!(took >= std::time::Duration::from_millis(150), "{took:?}");
        let (calls, r, _) = attempts(down, 2).await;
        assert_eq!((calls, r.unwrap().as_str()), (2, "ok"));
        for refused in [
            "fatal: couldn't find remote ref refs/heads/nope",
            "fatal: Authentication failed for 'https://h/r/'",
            "fatal: could not read Username for 'https://h': terminal prompts disabled",
            "error: unable to write file x: No space left on device",
        ] {
            let (calls, r, _) = attempts(refused, 0).await;
            assert_eq!(calls, 1, "{refused}");
            assert_eq!(r.unwrap_err(), refused);
        }
    }

    /// A git server that answers every request with `status`, counting
    /// the requests.
    async fn refusing_git_server(
        status: &'static str,
    ) -> (String, std::sync::Arc<std::sync::atomic::AtomicUsize>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let seen = count.clone();
        tokio::spawn(async move {
            while let Ok((mut sock, _)) = listener.accept().await {
                seen.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                tokio::spawn(async move {
                    let mut buf = [0u8; 4096];
                    let _ = sock.read(&mut buf).await;
                    let _ = sock
                        .write_all(
                            format!(
                                "HTTP/1.1 {status}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                            )
                            .as_bytes(),
                        )
                        .await;
                });
            }
        });
        (format!("http://{addr}/repo.git"), count)
    }

    /// E4 6.4: one connect failure cost a whole job. A server answering
    /// 503 is asked three times in all, by name and not again by history.
    #[tokio::test]
    async fn a_fetch_from_an_unavailable_server_is_tried_three_times() {
        if which("git").is_none() {
            return;
        }
        let (url, count) = refusing_git_server("503 Service Unavailable").await;
        let tmp = tempfile::tempdir().unwrap();
        let sha = "0123456789abcdef0123456789abcdef01234567";
        let e = fetch_git(&url, sha, &["tasks".into()], tmp.path())
            .await
            .unwrap_err();
        assert!(e.to_string().contains("503"), "{e}");
        assert_eq!(count.load(std::sync::atomic::Ordering::SeqCst), 3);
        // A missing page is not.
        let (url, count) = refusing_git_server("404 Not Found").await;
        let e = fetch_git(&url, sha, &["tasks".into()], tmp.path())
            .await
            .unwrap_err();
        assert!(e.to_string().contains("not found"), "{e}");
        // The fetch by name, then the history once.
        assert_eq!(count.load(std::sync::atomic::Ordering::SeqCst), 2);
    }

    /// Decision G-4: the `cache` helper (its daemon would hold the
    /// checkout's lock for 15 minutes) is left out; every other helper,
    /// for every URL, is given again in order, and empty values still
    /// clear the list before them.
    #[test]
    fn only_the_cache_credential_helper_is_left_out() {
        let listing = "credential.helper\ncache --timeout=60\0\
                       credential.helper\nstore\0\
                       credential.https://github.com.helper\n\0\
                       credential.https://github.com.helper\n!/opt/homebrew/bin/gh auth git-credential\0\
                       credential.https://h.example.helper\n/usr/lib/git-core/git-credential-cache\0\
                       credential.https://h.example.helper\n!git credential-cache --socket /s\0";
        let (settings, dropped) = helper_settings(listing);
        assert!(dropped);
        assert_eq!(
            settings,
            [
                "-c",
                "credential.helper=",
                "-c",
                "credential.helper=store",
                "-c",
                "credential.https://github.com.helper=",
                "-c",
                "credential.https://github.com.helper=!/opt/homebrew/bin/gh auth git-credential",
            ]
        );
        // No cache helper: git is given the helpers as they are.
        let (settings, dropped) =
            helper_settings("credential.helper\nosxkeychain\0credential.helper\ncacher\0");
        assert!(!dropped && settings.is_empty());
        assert_eq!(helper_settings(""), (Vec::new(), false));
        for cache in [
            "cache",
            "cache --timeout=1",
            "/x/git-credential-cache",
            "!git credential-cache",
        ] {
            assert!(cache_helper(cache), "{cache}");
        }
        for other in [
            "store",
            "osxkeychain",
            "!gh auth git-credential",
            "manager",
            "cacher",
        ] {
            assert!(!cache_helper(other), "{other}");
        }
    }
}
