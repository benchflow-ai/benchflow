//! Decision G-4: every git `bench` starts in a cached checkout inherits the
//! checkout's lock, so nothing git starts may outlive it. With a user
//! configuration that starts the fsmonitor daemon, the `cache` credential
//! helper's daemon, and automatic gc and maintenance, two fetches in a row
//! both finish without waiting, and after each no process holds the lock.
//! The origin is served over HTTP with basic authentication (`git
//! http-backend` behind a small server here), so the credential helpers
//! really run: the user's other helper (`store`, for this URL only) still
//! answers, and the `cache` helper never starts its daemon.
//!
//! One test in this file: it sets the process's environment (`HOME`,
//! `GIT_CONFIG_GLOBAL`), which every git it starts reads, and never the
//! real `~/.gitconfig`.

use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use benchflow_run::source::{checkout_locked, fetch_git};

const USER: &str = "u";
const PASSWORD: &str = "p";

fn git(dir: &Path, args: &[&str]) -> String {
    let out = Command::new("git")
        .args(["-c", "user.name=t", "-c", "user.email=t@t"])
        .args(["-c", "commit.gpgsign=false"])
        .args(args)
        .current_dir(dir)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

/// Serve the repositories under `root` with `git http-backend`, asking for
/// basic authentication as `USER`/`PASSWORD`; `refuse` answers every
/// request with 401 instead. Returns the port and a count of requests.
fn serve(root: PathBuf, refuse: bool) -> (u16, std::sync::Arc<std::sync::atomic::AtomicUsize>) {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let seen = count.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(stream) = stream else { continue };
            seen.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let root = root.clone();
            std::thread::spawn(move || {
                let _ = answer(stream, &root, refuse);
            });
        }
    });
    (port, count)
}

/// One request on `stream`, answered and closed.
fn answer(mut stream: std::net::TcpStream, root: &Path, refuse: bool) -> std::io::Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or_default().to_string();
    let target = parts.next().unwrap_or_default().to_string();
    let mut headers = Vec::new();
    loop {
        let mut h = String::new();
        reader.read_line(&mut h)?;
        let h = h.trim_end().to_string();
        if h.is_empty() {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            headers.push((k.trim().to_ascii_lowercase(), v.trim().to_string()));
        }
    }
    let header = |name: &str| {
        headers
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.clone())
    };
    // base64("u:p")
    let expected = "Basic dTpw";
    if refuse || header("authorization").as_deref() != Some(expected) {
        stream.write_all(
            b"HTTP/1.1 401 Unauthorized\r\nWWW-Authenticate: Basic realm=\"t\"\r\n\
              Content-Length: 0\r\nConnection: close\r\n\r\n",
        )?;
        return Ok(());
    }
    let mut body = Vec::new();
    if let Some(len) = header("content-length").and_then(|l| l.parse::<usize>().ok()) {
        body.resize(len, 0);
        reader.read_exact(&mut body)?;
    } else if header("transfer-encoding").is_some_and(|t| t.contains("chunked")) {
        loop {
            let mut size = String::new();
            reader.read_line(&mut size)?;
            let size = usize::from_str_radix(size.trim(), 16).unwrap_or(0);
            let mut chunk = vec![0; size + 2];
            reader.read_exact(&mut chunk)?;
            if size == 0 {
                break;
            }
            body.extend_from_slice(&chunk[..size]);
        }
    }
    let (path, query) = target.split_once('?').unwrap_or((&target, ""));
    let mut cgi = Command::new("git");
    cgi.arg("http-backend")
        .env("GIT_PROJECT_ROOT", root)
        .env("GIT_HTTP_EXPORT_ALL", "1")
        .env("REQUEST_METHOD", &method)
        .env("PATH_INFO", path)
        .env("QUERY_STRING", query)
        .env("CONTENT_LENGTH", body.len().to_string())
        .env("REMOTE_USER", USER)
        .env("REMOTE_ADDR", "127.0.0.1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    for (name, var) in [
        ("content-type", "CONTENT_TYPE"),
        ("content-encoding", "HTTP_CONTENT_ENCODING"),
        ("git-protocol", "GIT_PROTOCOL"),
    ] {
        if let Some(v) = header(name) {
            cgi.env(var, v);
        }
    }
    let mut child = cgi.spawn()?;
    child.stdin.take().unwrap().write_all(&body)?;
    let out = child.wait_with_output()?;
    let split = out
        .stdout
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .map(|i| (i, 4))
        .or_else(|| {
            out.stdout
                .windows(2)
                .position(|w| w == b"\n\n")
                .map(|i| (i, 2))
        })
        .unwrap_or((out.stdout.len(), 0));
    let head = String::from_utf8_lossy(&out.stdout[..split.0]).to_string();
    let rest = &out.stdout[(split.0 + split.1).min(out.stdout.len())..];
    let mut status = "200 OK".to_string();
    let mut response = String::new();
    for h in head.lines().map(str::trim).filter(|h| !h.is_empty()) {
        match h.split_once(':') {
            Some((k, v)) if k.eq_ignore_ascii_case("status") => status = v.trim().to_string(),
            _ => {
                response.push_str(h);
                response.push_str("\r\n");
            }
        }
    }
    stream.write_all(
        format!(
            "HTTP/1.1 {status}\r\n{response}Content-Length: {}\r\nConnection: close\r\n\r\n",
            rest.len()
        )
        .as_bytes(),
    )?;
    stream.write_all(rest)?;
    Ok(())
}

/// The processes that have `path` open: `lsof` on macOS, `/proc/*/fd` on
/// Linux. Each as `pid command`.
fn holders(path: &Path) -> Vec<String> {
    let path = path.canonicalize().unwrap();
    let pids: Vec<String> = if cfg!(target_os = "linux") {
        let mut pids = Vec::new();
        for proc_dir in std::fs::read_dir("/proc").unwrap().flatten() {
            let pid = proc_dir.file_name().to_string_lossy().to_string();
            if !pid.bytes().all(|b| b.is_ascii_digit()) {
                continue;
            }
            let Ok(fds) = std::fs::read_dir(proc_dir.path().join("fd")) else {
                continue;
            };
            if fds
                .flatten()
                .any(|fd| std::fs::read_link(fd.path()).is_ok_and(|t| t == path))
            {
                pids.push(pid);
            }
        }
        pids
    } else {
        let out = Command::new("lsof")
            .args(["-t", "--"])
            .arg(&path)
            .output()
            .unwrap();
        String::from_utf8_lossy(&out.stdout)
            .split_whitespace()
            .map(str::to_string)
            .collect()
    };
    pids.into_iter()
        .map(|pid| {
            let out = Command::new("ps")
                .args(["-o", "command=", "-p", &pid])
                .output()
                .unwrap();
            format!("{pid} {}", String::from_utf8_lossy(&out.stdout).trim())
        })
        .collect()
}

/// Stops whatever daemon a failing run of this test started.
struct Cleanup {
    checkout: Option<PathBuf>,
}

impl Drop for Cleanup {
    fn drop(&mut self) {
        let _ = Command::new("git")
            .args(["credential-cache", "exit"])
            .stderr(Stdio::null())
            .status();
        if let Some(dir) = &self.checkout {
            let _ = Command::new("git")
                .args(["fsmonitor--daemon", "stop"])
                .current_dir(dir)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
        }
    }
}

#[tokio::test]
async fn nothing_git_starts_keeps_the_checkout_locked() {
    if Command::new("git").arg("--version").output().is_err() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let tmp_path = tmp.path().canonicalize().unwrap();
    let home = tmp_path.join("home");
    std::fs::create_dir_all(&home).unwrap();
    let config = tmp_path.join("gitconfig");
    std::fs::write(&config, "").unwrap();
    // Nothing else in this test binary runs, so the environment is ours.
    #[allow(unsafe_code)]
    // SAFETY: the only test in this binary sets these before it starts any
    // thread or process.
    unsafe {
        std::env::set_var("HOME", &home);
        std::env::set_var("XDG_CONFIG_HOME", home.join(".config"));
        std::env::set_var("XDG_CACHE_HOME", home.join(".cache"));
        std::env::set_var("GIT_CONFIG_GLOBAL", &config);
        std::env::set_var("GIT_CONFIG_NOSYSTEM", "1");
    }
    let served = tmp_path.join("served");
    let repo = served.join("repo");
    for p in ["tasks/a", "tasks/b"] {
        std::fs::create_dir_all(repo.join(p)).unwrap();
        std::fs::write(repo.join(p).join("task.md"), p).unwrap();
    }
    git(&repo, &["init", "-q", "-b", "main"]);
    git(&repo, &["config", "uploadpack.allowFilter", "true"]);
    git(&repo, &["config", "uploadpack.allowAnySHA1InWant", "true"]);
    git(&repo, &["add", "."]);
    git(&repo, &["commit", "-q", "-m", "tasks"]);
    let sha = git(&repo, &["rev-parse", "HEAD"]);
    let (port, _) = serve(served.clone(), false);
    let url = format!("http://127.0.0.1:{port}/repo");
    // The user's credentials for this server, in a `store` file given for
    // this URL only; the `cache` helper for every URL.
    let store = tmp_path.join("credentials");
    std::fs::write(
        &store,
        format!("http://{USER}:{PASSWORD}@127.0.0.1:{port}\n"),
    )
    .unwrap();
    std::fs::write(
        &config,
        format!(
            "[core]\n\tfsmonitor = true\n[gc]\n\tauto = 1\n[maintenance]\n\tauto = true\n\
             [credential]\n\thelper = cache\n\
             [credential \"http://127.0.0.1:{port}\"]\n\thelper = store --file={}\n",
            store.display()
        ),
    )
    .unwrap();
    let mut cleanup = Cleanup { checkout: None };
    let cache = tmp_path.join("cache");
    for folder in ["tasks/a", "tasks/b"] {
        let started = std::time::Instant::now();
        let co = fetch_git(&url, &sha, &[folder.into()], &cache)
            .await
            .unwrap();
        cleanup.checkout = Some(co.dir.clone());
        assert!(co.dir.join(folder).join("task.md").is_file(), "{folder}");
        assert!(
            started.elapsed() < std::time::Duration::from_secs(30),
            "{folder} took {:?}",
            started.elapsed()
        );
        let dir = co.dir.clone();
        drop(co);
        let mut lock = dir.clone().into_os_string();
        lock.push(".lock");
        let lock = PathBuf::from(lock);
        assert!(lock.is_file(), "{}", lock.display());
        let held = holders(&lock);
        assert!(
            held.is_empty(),
            "after fetching {folder}, the lock is held by: {held:?}"
        );
        assert!(!checkout_locked(&dir));
    }
    // The store helper answered (the fetches above signed in), and the
    // cache helper never ran: its daemon would have made its socket.
    assert!(
        !home.join(".cache/git/credential/socket").exists()
            && !home.join(".git-credential-cache").exists(),
        "the cache helper's daemon was started"
    );

    // A server whose credentials no helper has: refused, said so, and not
    // asked again.
    let (port, count) = serve(served.clone(), true);
    let e = fetch_git(
        &format!("http://127.0.0.1:{port}/repo"),
        &sha,
        &["tasks/a".into()],
        &cache,
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(e.contains("`cache` credential helper"), "{e}");
    assert_eq!(count.load(std::sync::atomic::Ordering::SeqCst), 1, "{e}");
    drop(cleanup);
}
