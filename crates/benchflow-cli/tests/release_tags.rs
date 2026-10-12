//! bench's release tags are `bench-v<version>`, apart from the Python
//! package's `v<version>`: a tag starts one release workflow, never both.
//!
//! The `rust` workflow builds the binaries on its tags; `public-release`
//! publishes the Python package to PyPI on `v*`. When both started on `v*`,
//! a bench tag showed a failed PyPI job. `install.sh` turns a version into
//! the bench tag it downloads.

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read(rel: &str) -> String {
    std::fs::read_to_string(repo().join(rel)).unwrap_or_else(|e| panic!("{rel}: {e}"))
}

/// The tag patterns of a workflow's `on: push: tags:`, as written.
fn tag_patterns(workflow: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut in_tags = false;
    for line in workflow.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("tags:") {
            let rest = rest.trim();
            if rest.starts_with('[') {
                out.extend(
                    rest.trim_matches(|c| c == '[' || c == ']')
                        .split(',')
                        .map(|p| p.trim().trim_matches('"').to_string())
                        .filter(|p| !p.is_empty()),
                );
            } else {
                in_tags = true;
            }
            continue;
        }
        if in_tags {
            match t.strip_prefix("- ") {
                Some(p) => out.push(p.trim().trim_matches('"').to_string()),
                None => in_tags = false,
            }
        }
    }
    out
}

/// GitHub's tag filter for the patterns used here: a literal prefix and a
/// trailing `*`.
fn matches(pattern: &str, tag: &str) -> bool {
    match pattern.strip_suffix('*') {
        Some(prefix) => tag.starts_with(prefix),
        None => tag == pattern,
    }
}

#[test]
fn a_bench_tag_starts_only_the_rust_workflow_and_a_python_tag_only_pypi() {
    let rust = tag_patterns(&read(".github/workflows/rust.yml"));
    let pypi = tag_patterns(&read(".github/workflows/public-release.yml"));
    assert_eq!(rust, ["bench-v*"], "the rust workflow's tags");
    assert_eq!(pypi, ["v*"], "the Python release's tags");
    for (tag, rust_runs) in [
        ("bench-v0.8.1", true),
        ("bench-v0.9.0-rc.1", true),
        ("v0.8.1", false),
        ("v0.7.8", false),
    ] {
        let r = rust.iter().any(|p| matches(p, tag));
        let p = pypi.iter().any(|p| matches(p, tag));
        assert_eq!(r, rust_runs, "rust workflow on {tag}");
        assert_eq!(p, !rust_runs, "PyPI workflow on {tag}");
    }
}

#[test]
fn the_release_job_checks_the_tag_against_bench_v_and_the_cargo_version() {
    let rust = read(".github/workflows/rust.yml");
    assert!(
        rust.contains(r#"[ "$GITHUB_REF_NAME" != "bench-v$version" ]"#),
        "the release job must compare the tag with bench-v<version>"
    );
    assert!(
        !rust.contains(r#""v$version""#),
        "a leftover v<version> check"
    );
}

/// `install.sh` with a version, against an address nothing listens on: the
/// first line names the release it would download.
fn install_line(version: &str) -> Option<String> {
    let home = tempfile::tempdir().unwrap();
    let out = Command::new("sh")
        .arg(repo().join("install.sh"))
        .env("HOME", home.path())
        .env("BENCH_TARGET", "x86_64-unknown-linux-musl")
        .env("BENCH_VERSION", version)
        .env("BENCH_RELEASES_URL", "http://127.0.0.1:9")
        .env("NO_COLOR", "1")
        .output()
        .ok()?;
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    assert!(
        !out.status.success(),
        "nothing to download from: {stdout}{stderr}"
    );
    stdout.lines().next().map(str::to_string)
}

#[test]
fn install_sh_downloads_a_version_from_its_bench_v_release() {
    if Command::new("sh")
        .arg("-c")
        .arg("command -v curl || command -v wget")
        .output()
        .map(|o| !o.status.success())
        .unwrap_or(true)
    {
        eprintln!("skipped: neither curl nor wget");
        return;
    }
    for version in ["0.8.1", "v0.8.1", "bench-v0.8.1"] {
        assert_eq!(
            install_line(version).as_deref(),
            Some("Installing bench from release bench-v0.8.1 for x86_64-unknown-linux-musl"),
            "BENCH_VERSION={version}"
        );
    }
}

/// GitHub runs `schedule:` only from the default branch's workflow files,
/// which have no `rust.yml`: a cron there never fires, so neither the
/// workflow nor the docs may promise a nightly run (e2e review, 2026-10-11).
#[test]
fn the_rust_workflow_promises_no_schedule_it_cannot_keep() {
    let rust = read(".github/workflows/rust.yml");
    let code: Vec<&str> = rust
        .lines()
        .filter(|l| !l.trim_start().starts_with('#'))
        .collect();
    assert!(
        !code
            .iter()
            .any(|l| l.trim_start().starts_with("schedule:") || l.contains("'schedule'")),
        "rust.yml has a schedule"
    );
    for doc in ["docs/rust/testing-e2e.md", ".github/workflows/rust.yml"] {
        let text = read(doc).to_lowercase();
        for (i, line) in text.lines().enumerate() {
            assert!(
                !line.contains("nightly") || line.contains("no nightly"),
                "{doc}:{}: {line}",
                i + 1
            );
        }
    }
}

/// Serve `body` to every request on a local port, for `install.sh`.
fn serve(body: &'static str) -> String {
    use std::io::{Read, Write};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut s) = stream else { continue };
            let mut buf = [0u8; 4096];
            let _ = s.read(&mut buf);
            let _ = write!(
                s,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
        }
    });
    format!("http://{addr}/releases")
}

/// With no version, `install.sh` installs the newest bench release that is
/// not a prerelease, from the releases list: the repository's latest
/// release is often the Python package's `v<version>`, which has no bench
/// (diff review of rust/e9-decisions, M3).
#[test]
fn install_sh_finds_the_latest_bench_release_in_the_list() {
    if Command::new("sh")
        .arg("-c")
        .arg("command -v curl || command -v wget")
        .output()
        .map(|o| !o.status.success())
        .unwrap_or(true)
    {
        eprintln!("skipped: neither curl nor wget");
        return;
    }
    let run = |api: &str| {
        let home = tempfile::tempdir().unwrap();
        let out = Command::new("sh")
            .arg(repo().join("install.sh"))
            .env("HOME", home.path())
            .env("BENCH_TARGET", "x86_64-unknown-linux-musl")
            .env_remove("BENCH_VERSION")
            .env("BENCH_RELEASES_URL", "http://127.0.0.1:9")
            .env("BENCH_RELEASES_API", api)
            .env("NO_COLOR", "1")
            .output()
            .unwrap();
        format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        )
    };
    let list = serve(
        r#"[
  {"tag_name": "v0.9.0", "name": "benchflow 0.9.0", "draft": false, "prerelease": false},
  {"tag_name": "bench-v0.8.3-rc.1", "draft": false, "prerelease": true},
  {"tag_name": "bench-v0.8.2", "body": "fixes, like it, a lot", "draft": false, "prerelease": false},
  {"tag_name": "bench-v0.8.1", "draft": false, "prerelease": false}
]"#,
    );
    let out = run(&list);
    assert!(
        out.starts_with(
            "Installing bench from release bench-v0.8.2 (the latest) for x86_64-unknown-linux-musl"
        ),
        "{out}"
    );
    let python_only = serve(r#"[{"tag_name": "v0.9.0", "prerelease": false}]"#);
    let out = run(&python_only);
    assert!(
        out.contains("found no bench release")
            && out.contains("it lists no bench-v<version> release that is not a prerelease"),
        "{out}"
    );
}

/// `GITHUB_TOKEN` goes only over https to GitHub's API (api.github.com, or
/// the host of an https `GITHUB_API_URL`), never to a mirror, and never
/// over plain http (convergence review E9-C-B 4: it went to any
/// `BENCH_RELEASES_API`).
#[test]
fn install_sh_sends_github_token_only_to_githubs_api_over_https() {
    use std::io::{Read, Write};
    if Command::new("sh")
        .arg("-c")
        .arg("command -v curl")
        .output()
        .map(|o| !o.status.success())
        .unwrap_or(true)
    {
        eprintln!("skipped: no curl");
        return;
    }
    // A mirror over http that would answer only with the token.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let log = seen.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut s) = stream else { continue };
            let mut buf = [0u8; 8192];
            let n = s.read(&mut buf).unwrap_or(0);
            let request = String::from_utf8_lossy(&buf[..n]).to_lowercase();
            log.lock().unwrap().push(request.clone());
            let (status, body) = if request.contains("authorization:") {
                (
                    "200 OK",
                    r#"[{"tag_name": "bench-v0.8.2", "prerelease": false}]"#,
                )
            } else {
                ("403 Forbidden", r#"{"message": "API rate limit exceeded"}"#)
            };
            let _ = write!(
                s,
                "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
        }
    });
    let home = tempfile::tempdir().unwrap();
    let out = Command::new("sh")
        .arg(repo().join("install.sh"))
        .env("HOME", home.path())
        .env("BENCH_TARGET", "x86_64-unknown-linux-musl")
        .env_remove("BENCH_VERSION")
        .env_remove("GITHUB_API_URL")
        .env("GITHUB_TOKEN", "tok-123")
        .env("BENCH_RELEASES_URL", "http://127.0.0.1:9")
        .env("BENCH_RELEASES_API", format!("http://{addr}/releases"))
        .env("NO_COLOR", "1")
        .output()
        .unwrap();
    let out = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(out.contains("(HTTP 403)"), "{out}");
    assert!(
        seen.lock().unwrap().iter().all(|r| !r.contains("tok-123")),
        "the token reached the mirror"
    );
    // Which addresses get it.
    let script = std::fs::read_to_string(repo().join("install.sh")).unwrap();
    let start = script.find("token_host() {").expect("token_host");
    let end = start + script[start..].find("\n}\n").expect("its end") + 3;
    let function = &script[start..end];
    let gets = |url: &str, ghe: Option<&str>| {
        let mut c = Command::new("sh");
        c.arg("-c")
            .arg(format!("{function}\ntoken_host \"$1\""))
            .arg("sh")
            .arg(url)
            .env_remove("GITHUB_API_URL");
        if let Some(g) = ghe {
            c.env("GITHUB_API_URL", g);
        }
        c.status().unwrap().success()
    };
    assert!(gets(
        "https://api.github.com/repos/benchflow-ai/benchflow/releases",
        None
    ));
    assert!(!gets("http://api.github.com/repos/x/releases", None));
    assert!(!gets("https://mirror.example/releases.json", None));
    assert!(!gets("https://api.github.com.evil.example/x", None));
    assert!(gets(
        "https://ghe.example/api/v3/repos/x/releases",
        Some("https://ghe.example/api/v3")
    ));
    assert!(!gets(
        "http://ghe.example/api/v3/repos/x",
        Some("http://ghe.example/api/v3")
    ));
}

/// A stale or foreign `GITHUB_TOKEN` is refused by GitHub even on public
/// data (401): `install.sh` asks again without it instead of stopping
/// (convergence review E9-D-B 3). A GitHub Enterprise token never goes to
/// api.github.com.
#[test]
fn install_sh_asks_again_without_a_refused_token() {
    use std::io::{Read, Write};
    use std::os::unix::fs::PermissionsExt;
    let Some(real_curl) = std::env::var_os("PATH").and_then(|p| {
        std::env::split_paths(&p)
            .map(|d| d.join("curl"))
            .find(|c| c.is_file())
    }) else {
        eprintln!("skipped: no curl");
        return;
    };
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let got_token = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let got = got_token.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut s) = stream else { continue };
            let mut buf = [0u8; 8192];
            let n = s.read(&mut buf).unwrap_or(0);
            let request = String::from_utf8_lossy(&buf[..n]).to_lowercase();
            if request.contains("authorization: bearer stale-token") {
                got.store(true, std::sync::atomic::Ordering::SeqCst);
            }
            let (status, body) = if request.contains("authorization:") {
                ("401 Unauthorized", r#"{"message": "Bad credentials"}"#)
            } else {
                (
                    "200 OK",
                    r#"[{"tag_name": "v0.9.0", "prerelease": false}, {"tag_name": "bench-v0.8.2", "prerelease": false}]"#,
                )
            };
            let _ = write!(
                s,
                "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
        }
    });
    // A curl that speaks http where the script asks for https, so the
    // token request can reach this test's server.
    let bin = tempfile::tempdir().unwrap();
    let fake = bin.path().join("curl");
    std::fs::write(
        &fake,
        format!(
            "#!/bin/sh\nprintf '%s\\n' \"$@\" >> \"$CURL_ARGV_LOG\"\nskip=0\nfirst=1\nfor a do\n  if [ $first = 1 ]; then set --; first=0; fi\n  if [ $skip = 1 ]; then skip=0; continue; fi\n  case $a in --proto|--proto-redir) skip=1; continue;; https://*) a=http://${{a#https://}};; esac\n  set -- \"$@\" \"$a\"\ndone\nexec {} \"$@\"\n",
            real_curl.display()
        ),
    )
    .unwrap();
    std::fs::set_permissions(&fake, std::fs::Permissions::from_mode(0o755)).unwrap();
    let path = format!(
        "{}:{}",
        bin.path().display(),
        std::env::var("PATH").unwrap_or_default()
    );
    let home = tempfile::tempdir().unwrap();
    let out = Command::new("sh")
        .arg(repo().join("install.sh"))
        .env("PATH", &path)
        .env("HOME", home.path())
        .env("BENCH_TARGET", "x86_64-unknown-linux-musl")
        .env_remove("BENCH_VERSION")
        .env("GITHUB_TOKEN", "stale-token")
        .env("CURL_ARGV_LOG", bin.path().join("argv.log"))
        .env("GITHUB_API_URL", format!("https://{addr}"))
        .env("BENCH_RELEASES_URL", "http://127.0.0.1:9")
        .env("BENCH_RELEASES_API", format!("https://{addr}/releases"))
        .env("NO_COLOR", "1")
        .output()
        .unwrap();
    let out = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        out.starts_with("Installing bench from release bench-v0.8.2 (the latest)"),
        "{out}"
    );
    // The token reached the server (it answered 401 to it) but never curl's
    // arguments, which other local users can read (convergence review
    // E9-E-B 3).
    let argv = std::fs::read_to_string(bin.path().join("argv.log")).unwrap();
    assert!(argv.contains("-K"), "{argv}");
    assert!(
        got_token.load(std::sync::atomic::Ordering::SeqCst),
        "the token never reached the server"
    );
    assert!(
        !argv.contains("stale-token"),
        "the token is in curl's argv: {argv}"
    );
    // A GitHub Enterprise token stays off api.github.com.
    let script = std::fs::read_to_string(repo().join("install.sh")).unwrap();
    let start = script.find("token_host() {").expect("token_host");
    let end = start + script[start..].find("\n}\n").expect("its end") + 3;
    let ok = Command::new("sh")
        .arg("-c")
        .arg(format!("{}\ntoken_host \"$1\"", &script[start..end]))
        .arg("sh")
        .arg("https://api.github.com/repos/x/releases")
        .env("GITHUB_API_URL", "https://ghe.example/api/v3")
        .status()
        .unwrap()
        .success();
    assert!(!ok, "a GHE token went to api.github.com");
}
