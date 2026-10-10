//! Job files: `bench run -c job.toml`, a run written down.
//!
//! ```toml
//! targets = ["tasks/"]          # or: dataset = "terminal-bench@2.0"
//! include = ["fix-*"]
//! agent = "claude"
//! model = "claude-sonnet-4-6"
//! trials = 3
//! concurrency = 8
//!
//! [env]
//! MY_FLAG = "1"
//! ```
//!
//! The keys are `bench.toml`'s `[run]` keys plus `targets`, `dataset`,
//! `registry`, `include`, and `exclude`. A file ending in `.yaml` or `.yml` is
//! read as YAML, `.json` as JSON, anything else as TOML. Given several files, later ones
//! override earlier ones key by key: tables merge, lists append, and an
//! explicit empty list clears what came before. Relative `targets`,
//! `jobs-dir`, `skills`, `load-trajectory`, and a local `dataset` or
//! `registry` file are relative to the file that names them.
//!
//! A job file can also be an `https://` URL ([`load_sources`]): a raw file,
//! or a GitHub `github.com/<owner>/<repo>/blob/<ref>/<path>` page, which is
//! read from `raw.githubusercontent.com`. It is fetched within
//! [`FETCH_TIMEOUT`] and [`MAX_BYTES`], only parsed (never run), and
//! recorded with its digest ([`JobFileRecord`]) in `lock.json`. Relative
//! paths in a fetched file are relative to the current folder, and it may
//! not name others, nor set what reads this machine's files or variables
//! into the run ([`check_fetched`]).

use std::path::{Path, PathBuf};
use std::time::Duration;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::project::RunDefaults;

/// A job file, or several merged.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub struct JobFile {
    /// Task folders, folders of tasks, or GitHub repositories, as `bench run`
    /// takes them.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub targets: Vec<String>,
    /// A dataset from a registry: `name@version`, or a dataset.toml.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dataset: Option<String>,
    /// The registry to resolve `dataset` from: a URL or a file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub registry: Option<String>,
    /// Only tasks whose names match these globs.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub include: Vec<String>,
    /// Skip tasks whose names match these globs.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub exclude: Vec<String>,
    /// bench.toml's `[run]` keys.
    #[serde(flatten)]
    pub run: RunDefaults,
}

/// The keys a job file takes besides `[run]`'s.
const JOB_KEYS: [&str; 5] = ["targets", "dataset", "registry", "include", "exclude"];

/// A job file that could not be read.
#[derive(Debug, thiserror::Error)]
#[error("{path}: {message}")]
pub struct JobFileError {
    pub path: PathBuf,
    pub message: String,
    /// The file could not be read or fetched; otherwise it was read and is
    /// not a valid job file.
    pub unreadable: bool,
}

/// How long fetching a job file from a URL may take, start to end.
pub const FETCH_TIMEOUT: Duration = Duration::from_secs(30);

/// The largest job file fetched from a URL.
pub const MAX_BYTES: usize = 1024 * 1024;

/// A job file a run read, as `lock.json` records it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct JobFileRecord {
    /// The path or URL as given to `-c`.
    pub source: String,
    /// The URL fetched, when `source` is a URL (a GitHub blob page is
    /// fetched from raw.githubusercontent.com).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fetched_from: Option<String>,
    /// `sha256:<hex>` of the bytes read.
    pub digest: String,
    pub bytes: usize,
}

/// Job files merged, and what was read.
#[derive(Debug, Clone, Default)]
pub struct Loaded {
    pub file: JobFile,
    pub records: Vec<JobFileRecord>,
}

/// Read and merge job files, in order.
pub fn load(paths: &[PathBuf]) -> Result<JobFile, JobFileError> {
    let mut merged = Value::Object(Map::new());
    for path in paths {
        let text = std::fs::read_to_string(path).map_err(|e| JobFileError {
            path: path.to_path_buf(),
            message: e.to_string(),
            unreadable: true,
        })?;
        let v = parse_one(
            &text,
            Format::of(&path.to_string_lossy()),
            path,
            path.parent(),
        )?;
        merge(&mut merged, v);
    }
    finish(merged, paths.last().cloned())
}

/// Read and merge job files given as paths or `https://` URLs, in order,
/// recording each one's digest.
pub async fn load_sources(sources: &[String]) -> Result<Loaded, JobFileError> {
    load_sources_with(sources, &FetchLimits::default()).await
}

/// What a fetch may take; tests shorten these.
#[derive(Debug, Clone)]
pub struct FetchLimits {
    pub timeout: Duration,
    pub max_bytes: usize,
}

impl Default for FetchLimits {
    fn default() -> Self {
        Self {
            timeout: FETCH_TIMEOUT,
            max_bytes: MAX_BYTES,
        }
    }
}

/// [`load_sources`] with limits.
pub async fn load_sources_with(
    sources: &[String],
    limits: &FetchLimits,
) -> Result<Loaded, JobFileError> {
    let mut merged = Value::Object(Map::new());
    let mut records = Vec::new();
    for source in sources {
        let label = PathBuf::from(source);
        let (text, fetched_from, base) = if is_url(source) {
            let url = github_blob_to_raw(source);
            let text = fetch(&url, limits).await.map_err(|message| JobFileError {
                path: label.clone(),
                message,
                unreadable: true,
            })?;
            (text, Some(url), None)
        } else {
            let text = std::fs::read_to_string(source).map_err(|e| JobFileError {
                path: label.clone(),
                message: e.to_string(),
                unreadable: true,
            })?;
            (
                text,
                None,
                Path::new(source).parent().map(Path::to_path_buf),
            )
        };
        let format = Format::of(fetched_from.as_deref().map_or(source.as_str(), url_path));
        if fetched_from.is_some() {
            refuse_web_page(&text).map_err(|message| JobFileError {
                path: label.clone(),
                message,
                unreadable: false,
            })?;
        }
        let v = parse_one(&text, format, &label, base.as_deref())?;
        if fetched_from.is_some() {
            check_fetched(&v).map_err(|message| JobFileError {
                path: label.clone(),
                message,
                unreadable: false,
            })?;
        }
        merge(&mut merged, v);
        records.push(JobFileRecord {
            source: source.clone(),
            fetched_from,
            digest: format!(
                "sha256:{}",
                hex::encode(<sha2::Sha256 as sha2::Digest>::digest(text.as_bytes()))
            ),
            bytes: text.len(),
        });
    }
    let file = finish(merged, sources.last().map(PathBuf::from))?;
    Ok(Loaded { file, records })
}

/// What a job file fetched from a URL may not set: anything that reads
/// this machine's files or variables into the run, or names a place outside
/// the current folder. Whoever wrote the file is not the one running it.
/// The same settings given as flags (or in a local job file) are taken.
fn check_fetched(v: &Value) -> Result<(), String> {
    let Value::Object(map) = v else {
        return Ok(());
    };
    for (key, flag) in [
        ("env", "--env"),
        ("skills", "--skill"),
        ("load-trajectory", "--load-trajectory"),
    ] {
        if map.contains_key(key) {
            return Err(format!(
                "a job file from a URL cannot set `{key}`: it would bring this machine's files or variables into the run; pass {flag} yourself, or download the file, check it, and run it as a local job file (-c ./file)"
            ));
        }
    }
    if let Some(Value::String(agent)) = map.get("agent")
        && agent.starts_with("script:")
    {
        return Err(format!(
            "a job file from a URL cannot run a script from this machine (agent = {agent:?}); pass --agent yourself"
        ));
    }
    let outside = |s: &str| {
        let p = Path::new(s);
        !is_url(s)
            && !s.contains("://")
            && (p.is_absolute()
                || s.starts_with('~')
                || p.components().any(|c| c == std::path::Component::ParentDir))
    };
    let mut paths: Vec<(&str, &str)> = Vec::new();
    if let Some(Value::Array(targets)) = map.get("targets") {
        paths.extend(
            targets
                .iter()
                .filter_map(Value::as_str)
                .map(|t| ("targets", t)),
        );
    }
    for key in ["jobs-dir", "dataset", "registry"] {
        if let Some(Value::String(s)) = map.get(key) {
            paths.push((key, s));
        }
    }
    if let Some((key, p)) = paths.into_iter().find(|(_, p)| outside(p)) {
        return Err(format!(
            "a job file from a URL names only paths inside the current folder; `{key}` names {p:?}"
        ));
    }
    Ok(())
}

fn finish(merged: Value, last: Option<PathBuf>) -> Result<JobFile, JobFileError> {
    // Each file was valid alone; a merge of valid files is valid.
    serde_json::from_value(merged).map_err(|e| JobFileError {
        path: last.unwrap_or_default(),
        message: e.to_string(),
        unreadable: false,
    })
}

/// A job file's syntax, by its name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Format {
    Toml,
    Yaml,
    Json,
}

impl Format {
    fn of(name: &str) -> Format {
        let lower = name.to_ascii_lowercase();
        if lower.ends_with(".yaml") || lower.ends_with(".yml") {
            Format::Yaml
        } else if lower.ends_with(".json") {
            Format::Json
        } else {
            Format::Toml
        }
    }
}

/// Whether `-c` names a URL rather than a file.
pub fn is_url(source: &str) -> bool {
    source.starts_with("https://") || source.starts_with("http://")
}

/// A GitHub blob page's raw file (`github.com/o/r/blob/<ref>/<path>` to
/// `raw.githubusercontent.com/o/r/<ref>/<path>`), dropping any query, as
/// Harbor does; any other URL as given.
pub fn github_blob_to_raw(url: &str) -> String {
    let Ok(parsed) = reqwest::Url::parse(url) else {
        return url.to_string();
    };
    let host = parsed.host_str().unwrap_or("").to_ascii_lowercase();
    if host != "github.com" && host != "www.github.com" {
        return url.to_string();
    }
    let parts: Vec<&str> = parsed.path().trim_start_matches('/').split('/').collect();
    if parts.len() < 5
        || parts[2] != "blob"
        || parts[..2].iter().chain(&parts[3..4]).any(|p| p.is_empty())
    {
        return url.to_string();
    }
    let mut rest = vec![parts[0], parts[1], parts[3]];
    rest.extend(&parts[4..]);
    format!("https://raw.githubusercontent.com/{}", rest.join("/"))
}

/// A URL without its query or fragment, for telling its file's syntax.
fn url_path(url: &str) -> &str {
    url.split(['?', '#']).next().unwrap_or(url)
}

/// Fetch a job file: HTTPS only (plain HTTP only from this machine), within
/// the time and size limits, as UTF-8 text.
async fn fetch(url: &str, limits: &FetchLimits) -> Result<String, String> {
    let parsed = reqwest::Url::parse(url).map_err(|e| format!("not a URL: {e}"))?;
    let loopback = matches!(parsed.host_str(), Some("127.0.0.1" | "localhost" | "[::1]"));
    if parsed.scheme() != "https" && !(parsed.scheme() == "http" && loopback) {
        return Err("job files are fetched over HTTPS only; use the https:// address".into());
    }
    let client = reqwest::Client::builder()
        .timeout(limits.timeout)
        .redirect(reqwest::redirect::Policy::custom(move |attempt| {
            // Each hop as the first: HTTPS, or plain HTTP only on this
            // machine when the first was.
            let next = attempt.url();
            let local = matches!(next.host_str(), Some("127.0.0.1" | "localhost" | "[::1]"));
            if attempt.previous().len() >= 5 {
                attempt.error("more than 5 redirects")
            } else if next.scheme() == "https" || (loopback && local && next.scheme() == "http") {
                attempt.follow()
            } else {
                attempt.error("it redirects off HTTPS")
            }
        }))
        .build()
        .map_err(|e| format!("could not set up the download: {e}"))?;
    let secs = limits.timeout.as_secs_f64();
    let timed_out = || {
        format!("no complete answer within {secs}s; check the URL and the network, then try again")
    };
    let mut resp = client.get(parsed).send().await.map_err(|e| {
        if e.is_timeout() {
            timed_out()
        } else if e.is_redirect() {
            "it redirects off HTTPS or more than 5 times; link the file's final https:// address"
                .to_string()
        } else {
            format!("could not fetch it ({e}); check the URL and the network")
        }
    })?;
    let status = resp.status();
    if status == reqwest::StatusCode::NOT_FOUND {
        return Err(format!(
            "HTTP 404: there is no file at {url}; check the path and the branch (a private repository's files also answer 404)"
        ));
    }
    if !status.is_success() {
        return Err(format!(
            "HTTP {status}: the server did not return the file; check the URL"
        ));
    }
    let too_big = |n: String| {
        format!(
            "the file is {n} bytes, more than a job file may be ({} bytes); link a job file, not a dataset or archive",
            limits.max_bytes
        )
    };
    if let Some(n) = resp.content_length()
        && n > limits.max_bytes as u64
    {
        return Err(too_big(n.to_string()));
    }
    let html = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.to_ascii_lowercase().starts_with("text/html"));
    let mut body: Vec<u8> = Vec::new();
    loop {
        let chunk = resp.chunk().await.map_err(|e| {
            if e.is_timeout() {
                timed_out()
            } else {
                format!("the download broke off ({e}); try again")
            }
        })?;
        let Some(chunk) = chunk else { break };
        body.extend_from_slice(&chunk);
        if body.len() > limits.max_bytes {
            return Err(too_big(format!("more than {}", limits.max_bytes)));
        }
    }
    if html {
        return Err(web_page_error());
    }
    String::from_utf8(body)
        .map_err(|_| "the body is not UTF-8 text, so it is not a job file".to_string())
}

fn web_page_error() -> String {
    "the server returned a web page, not a job file; link the raw file (on GitHub, the Raw button's raw.githubusercontent.com address, or the github.com/<owner>/<repo>/blob/<ref>/<path> page)".into()
}

/// A body that is HTML, whatever its headers say.
fn refuse_web_page(text: &str) -> Result<(), String> {
    let head = text
        .trim_start()
        .get(..15)
        .unwrap_or(text.trim_start())
        .to_ascii_lowercase();
    if head.starts_with("<!doctype html") || head.starts_with("<html") {
        return Err(web_page_error());
    }
    Ok(())
}

fn parse_one(
    text: &str,
    format: Format,
    path: &Path,
    base: Option<&Path>,
) -> Result<Value, JobFileError> {
    let fail = |message: String| JobFileError {
        path: path.to_path_buf(),
        message,
        unreadable: false,
    };
    let mut v: Value = match format {
        Format::Yaml => {
            serde_yaml_ng::from_str(text).map_err(|e| fail(format!("not valid YAML: {e}")))?
        }
        Format::Json => {
            serde_json::from_str(text).map_err(|e| fail(format!("not valid JSON: {e}")))?
        }
        Format::Toml => toml::from_str(text).map_err(|e| {
            fail(format!(
                "not valid TOML (a job file not named .yaml, .yml, or .json is read as TOML): {}",
                e.to_string().trim()
            ))
        })?,
    };
    if v.is_null() {
        v = Value::Object(Map::new());
    }
    let Value::Object(map) = &mut v else {
        return Err(fail("a job file is a table of keys".into()));
    };
    // A [run] table, as in bench.toml, means the same keys at the top.
    if let Some(run) = map.remove("run") {
        let Value::Object(run) = run else {
            return Err(fail(
                "`run` must be a table of bench.toml's [run] keys".into(),
            ));
        };
        for (k, value) in run {
            if map.contains_key(&k) {
                return Err(fail(format!(
                    "`{k}` is set both at the top and in [run]; keep one"
                )));
            }
            map.insert(k, value);
        }
    }
    // The [run] keys must stand alone as bench.toml's do; that check names
    // any unknown key.
    let mut run = map.clone();
    for k in JOB_KEYS {
        run.remove(k);
    }
    serde_json::from_value::<RunDefaults>(Value::Object(run)).map_err(|e| {
        let m = e.to_string();
        fail(if m.starts_with("unknown field") {
            format!(
                "{m}; job files take bench.toml's [run] keys plus {}",
                JOB_KEYS.join(", ")
            )
        } else {
            m
        })
    })?;
    serde_json::from_value::<JobFile>(v.clone()).map_err(|e| fail(e.to_string()))?;
    if let Some(base) = base {
        relative_to(&mut v, base);
    }
    Ok(v)
}

/// Make `targets`, `jobs-dir`, `skills`, `load-trajectory`, and local
/// `dataset` and `registry` paths relative to `base`; leave repositories, URLs, and dataset names as
/// written.
fn relative_to(v: &mut Value, base: &Path) {
    let Value::Object(map) = v else { return };
    if let Some(Value::Array(targets)) = map.get_mut("targets") {
        for t in targets.iter_mut() {
            if let Value::String(s) = t {
                let p = Path::new(s.as_str());
                if !p.is_absolute() && base.join(p).exists() {
                    *s = base.join(p).display().to_string();
                }
            }
        }
    }
    for key in ["dataset", "registry"] {
        if let Some(Value::String(s)) = map.get_mut(key) {
            let p = Path::new(s.as_str());
            if !p.is_absolute() && !s.contains("://") && base.join(p).exists() {
                *s = base.join(p).display().to_string();
            }
        }
    }
    let rebase = |s: &mut String| {
        if !Path::new(s.as_str()).is_absolute() {
            *s = base.join(s.as_str()).display().to_string();
        }
    };
    for key in ["jobs-dir", "load-trajectory"] {
        if let Some(Value::String(d)) = map.get_mut(key) {
            rebase(d);
        }
    }
    if let Some(Value::Array(skills)) = map.get_mut("skills") {
        for s in skills.iter_mut() {
            if let Value::String(s) = s {
                rebase(s);
            }
        }
    }
}

/// Lay `over` on `base`: tables merge, lists append (an empty list clears),
/// anything else is replaced.
pub fn merge(base: &mut Value, over: Value) {
    match (base, over) {
        (Value::Object(b), Value::Object(o)) => {
            for (k, v) in o {
                match b.get_mut(&k) {
                    Some(existing) => merge(existing, v),
                    None => {
                        b.insert(k, v);
                    }
                }
            }
        }
        (Value::Array(b), Value::Array(o)) => {
            if o.is_empty() {
                b.clear();
            } else {
                b.extend(o);
            }
        }
        (b, o) => *b = o,
    }
}

impl JobFile {
    /// The file as TOML, with unset keys left out.
    pub fn to_toml(&self) -> String {
        let mut v = serde_json::to_value(self).unwrap_or(Value::Null);
        strip_nulls(&mut v);
        if let Value::Object(m) = &mut v
            && m.get("env")
                .is_some_and(|e| e.as_object().is_some_and(Map::is_empty))
        {
            m.remove("env");
        }
        toml::to_string(&v).unwrap_or_default()
    }
}

fn strip_nulls(v: &mut Value) {
    if let Value::Object(m) = v {
        m.retain(|_, x| !x.is_null());
        for x in m.values_mut() {
            strip_nulls(x);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_run_table_reads_like_top_level_keys() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("job.toml");
        std::fs::write(
            &p,
            "targets = [\"tasks\"]\n[run]\nagent = \"codex\"\nharness = \"headless\"\n",
        )
        .unwrap();
        let job = load(std::slice::from_ref(&p)).unwrap();
        assert_eq!(job.run.agent.as_deref(), Some("codex"));
        assert_eq!(job.run.harness.as_deref(), Some("headless"));
        std::fs::write(&p, "agent = \"nop\"\n[run]\nagent = \"codex\"\n").unwrap();
        let e = load(std::slice::from_ref(&p)).unwrap_err().to_string();
        assert!(e.contains("both at the top and in [run]"), "{e}");
    }

    fn write(dir: &Path, name: &str, text: &str) -> PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, text).unwrap();
        p
    }

    #[test]
    fn later_files_override_and_lists_append() {
        let d = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(d.path().join("tasks")).unwrap();
        let a = write(
            d.path(),
            "a.toml",
            "targets = [\"tasks\"]\ninclude = [\"a-*\"]\nagent = \"claude\"\nconcurrency = 2\n\n[env]\nX = \"1\"\n",
        );
        let b = write(
            d.path(),
            "b.yaml",
            "include: [\"b-*\"]\nconcurrency: 8\ntimeout-multiplier: 2\nenv:\n  Y: \"2\"\n",
        );
        let j = load(&[a.clone(), b]).unwrap();
        assert_eq!(
            j.targets,
            vec![d.path().join("tasks").display().to_string()]
        );
        assert_eq!(j.include, vec!["a-*", "b-*"]);
        assert_eq!(j.run.agent.as_deref(), Some("claude"));
        assert_eq!(j.run.concurrency, Some(8));
        assert_eq!(j.run.timeout_multiplier, Some(2.0));
        assert_eq!(j.run.env.len(), 2);
        // An explicit empty list clears.
        let c = write(
            d.path(),
            "c.toml",
            "include = []\ntargets = [\"org/repo@v1\"]\n",
        );
        let j = load(&[a, c]).unwrap();
        assert!(j.include.is_empty());
        assert_eq!(j.targets.len(), 2);
        assert_eq!(j.targets[1], "org/repo@v1", "repositories stay as written");
    }

    #[test]
    fn unknown_keys_are_named() {
        let d = tempfile::tempdir().unwrap();
        let p = write(d.path(), "job.toml", "agnet = \"claude\"\n");
        let e = load(&[p]).unwrap_err();
        assert!(e.message.contains("agnet"), "{}", e.message);
        assert!(e.message.contains("targets, dataset"), "{}", e.message);
    }

    #[test]
    fn prints_back_as_a_job_file() {
        let j = JobFile {
            dataset: Some("terminal-bench@2.0".into()),
            run: RunDefaults {
                agent: Some("claude".into()),
                trials: Some(3),
                ..RunDefaults::default()
            },
            ..JobFile::default()
        };
        let text = j.to_toml();
        assert!(text.contains("dataset = \"terminal-bench@2.0\""), "{text}");
        assert!(!text.contains("env"), "{text}");
        let back: JobFile = toml::from_str(&text).unwrap();
        assert_eq!(back, j);
    }

    /// A local server answering each path with (status, content type,
    /// body, whether to send Content-Length, delay).
    type Route = (&'static str, u16, &'static str, Vec<u8>, bool, u64);

    async fn serve(routes: Vec<Route>) -> String {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let routes = std::sync::Arc::new(routes);
        tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else {
                    return;
                };
                let routes = routes.clone();
                tokio::spawn(async move {
                    let mut buf = vec![0u8; 4096];
                    let n = sock.read(&mut buf).await.unwrap_or(0);
                    let req = String::from_utf8_lossy(&buf[..n]).to_string();
                    let path = req.split_whitespace().nth(1).unwrap_or("/").to_string();
                    let (status, ctype, body, length, delay) = routes
                        .iter()
                        .find(|r| r.0 == path)
                        .map(|r| (r.1, r.2, r.3.clone(), r.4, r.5))
                        .unwrap_or((404, "text/plain", b"404: Not Found".to_vec(), true, 0));
                    tokio::time::sleep(Duration::from_millis(delay)).await;
                    let mut head = format!(
                        "HTTP/1.1 {status} X\r\nContent-Type: {ctype}\r\nConnection: close\r\n"
                    );
                    if length {
                        head.push_str(&format!("Content-Length: {}\r\n", body.len()));
                    }
                    head.push_str("\r\n");
                    let _ = sock.write_all(head.as_bytes()).await;
                    let _ = sock.write_all(&body).await;
                    let _ = sock.shutdown().await;
                });
            }
        });
        format!("http://127.0.0.1:{}", addr.port())
    }

    #[test]
    fn github_blob_pages_become_raw_urls() {
        assert_eq!(
            github_blob_to_raw(
                "https://github.com/kobe0938/tb-timeout/blob/main/configs/cheating-judge-smoke.yaml"
            ),
            "https://raw.githubusercontent.com/kobe0938/tb-timeout/main/configs/cheating-judge-smoke.yaml"
        );
        assert_eq!(
            github_blob_to_raw("https://www.github.com/o/r/blob/v1.2/a/b.toml?plain=1"),
            "https://raw.githubusercontent.com/o/r/v1.2/a/b.toml"
        );
        for same in [
            "https://raw.githubusercontent.com/o/r/main/job.yaml",
            "https://github.com/o/r/tree/main/job.yaml",
            "https://github.com/o/r/blob/main",
            "https://example.com/o/r/blob/main/job.toml",
        ] {
            assert_eq!(github_blob_to_raw(same), same);
        }
        assert_eq!(Format::of("https://x/y/job.YML"), Format::Yaml);
        assert_eq!(
            Format::of(url_path("https://x/job.json?raw=1")),
            Format::Json
        );
        assert_eq!(Format::of("https://x/job"), Format::Toml);
    }

    #[tokio::test]
    async fn a_url_is_fetched_parsed_merged_and_recorded() {
        let body = b"concurrency: 7\ntargets: [tasks]\ninclude: [b-*]\n".to_vec();
        let base = serve(vec![(
            "/configs/job.yaml",
            200,
            "text/plain",
            body.clone(),
            true,
            0,
        )])
        .await;
        let d = tempfile::tempdir().unwrap();
        let local = write(
            d.path(),
            "a.toml",
            "agent = \"claude\"\ninclude = [\"a-*\"]\n",
        );
        let url = format!("{base}/configs/job.yaml");
        let loaded = load_sources(&[local.display().to_string(), url.clone()])
            .await
            .unwrap();
        assert_eq!(loaded.file.run.agent.as_deref(), Some("claude"));
        assert_eq!(loaded.file.run.concurrency, Some(7));
        assert_eq!(loaded.file.include, ["a-*", "b-*"]);
        assert_eq!(
            loaded.file.targets,
            ["tasks"],
            "a fetched file's paths stay as written"
        );
        assert_eq!(loaded.records.len(), 2);
        let r = &loaded.records[1];
        assert_eq!(r.source, url);
        assert_eq!(r.fetched_from.as_deref(), Some(url.as_str()));
        assert_eq!(r.bytes, body.len());
        assert_eq!(
            r.digest,
            format!(
                "sha256:{}",
                hex::encode(<sha2::Sha256 as sha2::Digest>::digest(&body))
            )
        );
        assert_eq!(loaded.records[0].fetched_from, None);
    }

    #[tokio::test]
    async fn fetch_failures_say_what_to_do() {
        let big = vec![b'#'; 2048];
        let base = serve(vec![
            (
                "/page.toml",
                200,
                "text/html; charset=utf-8",
                b"<!DOCTYPE html><html></html>".to_vec(),
                true,
                0,
            ),
            (
                "/sneaky.yaml",
                200,
                "text/plain",
                b"\n<html><body>hi</body></html>".to_vec(),
                true,
                0,
            ),
            (
                "/broken.yaml",
                200,
                "text/plain",
                b"concurrency: [\n".to_vec(),
                true,
                0,
            ),
            (
                "/prose.toml",
                200,
                "text/plain",
                b"just some words\n".to_vec(),
                true,
                0,
            ),
            (
                "/unknown.toml",
                200,
                "text/plain",
                b"agnet = \"x\"\n".to_vec(),
                true,
                0,
            ),
            ("/big.toml", 200, "text/plain", big.clone(), true, 0),
            ("/big-stream.toml", 200, "text/plain", big, false, 0),
            (
                "/slow.toml",
                200,
                "text/plain",
                b"trials = 1\n".to_vec(),
                true,
                2000,
            ),
            ("/oops.toml", 500, "text/plain", b"".to_vec(), true, 0),
        ])
        .await;
        let limits = FetchLimits {
            timeout: Duration::from_millis(500),
            max_bytes: 1024,
        };
        let err = |path: &'static str| {
            let url = format!("{base}{path}");
            let limits = limits.clone();
            async move { load_sources_with(&[url], &limits).await.unwrap_err() }
        };
        let e = err("/missing.toml").await;
        assert!(
            e.unreadable
                && e.message.starts_with("HTTP 404")
                && e.message.contains("check the path"),
            "{e}"
        );
        let e = err("/page.toml").await;
        assert!(
            e.message.contains("web page") && e.message.contains("raw"),
            "{e}"
        );
        let e = err("/sneaky.yaml").await;
        assert!(!e.unreadable && e.message.contains("web page"), "{e}");
        let e = err("/broken.yaml").await;
        assert!(
            !e.unreadable && e.message.starts_with("not valid YAML"),
            "{e}"
        );
        let e = err("/prose.toml").await;
        assert!(e.message.starts_with("not valid TOML"), "{e}");
        let e = err("/unknown.toml").await;
        assert!(e.message.contains("agnet"), "{e}");
        let e = err("/big.toml").await;
        assert!(e.unreadable && e.message.contains("2048 bytes"), "{e}");
        let e = err("/big-stream.toml").await;
        assert!(e.message.contains("more than 1024"), "{e}");
        let e = err("/slow.toml").await;
        assert!(
            e.unreadable && e.message.contains("no complete answer within 0.5s"),
            "{e}"
        );
        let e = err("/oops.toml").await;
        assert!(e.message.starts_with("HTTP 500"), "{e}");
    }

    #[tokio::test]
    async fn a_fetched_file_cannot_reach_into_this_machine() {
        let routes: Vec<(&'static str, &'static [u8])> = vec![
            ("/env.toml", b"env = { SOME_BASE_URL = \"x\" }\n"),
            ("/skills.toml", b"skills = [\"s\"]\n"),
            ("/traj.toml", b"load-trajectory = \"a.jsonl\"\n"),
            ("/script.toml", b"agent = \"script:x.sh\"\n"),
            ("/abs.toml", b"targets = [\"/etc\"]\n"),
            ("/up.toml", b"jobs-dir = \"../elsewhere\"\n"),
            ("/home.toml", b"registry = \"~/r.json\"\n"),
            ("/runtable.toml", b"[run]\nenv = { A = \"b\" }\n"),
        ];
        let base = serve(
            routes
                .iter()
                .map(|(p, b)| (*p, 200, "text/plain", b.to_vec(), true, 0))
                .collect(),
        )
        .await;
        for (path, _) in &routes {
            let e = load_sources(&[format!("{base}{path}")]).await.unwrap_err();
            assert!(
                !e.unreadable && e.message.contains("from a URL"),
                "{path}: {e}"
            );
        }
        // The refusal says how to proceed: the flag, or a local copy.
        let e = load_sources(&[format!("{base}/env.toml")])
            .await
            .unwrap_err();
        assert!(
            e.message.contains("--env") && e.message.contains("local job file"),
            "{e}"
        );
        let fine = serve(vec![(
            "/ok.toml",
            200,
            "text/plain",
            b"targets = [\"tasks\", \"https://github.com/o/r\"]\njobs-dir = \"jobs\"\nagent = \"claude\"\n".to_vec(),
            true,
            0,
        )])
        .await;
        assert!(load_sources(&[format!("{fine}/ok.toml")]).await.is_ok());
        // A local job file may still set them.
        let d = tempfile::tempdir().unwrap();
        let local = write(
            d.path(),
            "j.toml",
            "env = { A = \"b\" }\nskills = [\"s\"]\n",
        );
        assert!(load_sources(&[local.display().to_string()]).await.is_ok());
    }

    #[tokio::test]
    async fn a_redirect_off_https_is_not_followed() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            while let Ok((mut sock, _)) = listener.accept().await {
                let mut buf = vec![0u8; 4096];
                let _ = sock.read(&mut buf).await;
                let head = "HTTP/1.1 302 Found\r\nLocation: http://example.com/job.toml\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
                let _ = sock.write_all(head.as_bytes()).await;
                let _ = sock.shutdown().await;
            }
        });
        let e = load_sources(&[format!("http://127.0.0.1:{port}/job.toml")])
            .await
            .unwrap_err();
        assert!(
            e.unreadable && e.message.contains("redirects off HTTPS"),
            "{e}"
        );
    }

    #[tokio::test]
    async fn plain_http_is_refused_off_this_machine() {
        let e = load_sources(&["http://example.com/job.toml".to_string()])
            .await
            .unwrap_err();
        assert!(e.message.contains("HTTPS only"), "{e}");
    }

    #[test]
    fn json_job_files_read_by_extension() {
        let d = tempfile::tempdir().unwrap();
        let p = write(
            d.path(),
            "job.json",
            "{\"agent\": \"codex\", \"trials\": 2}",
        );
        let j = load(&[p]).unwrap();
        assert_eq!(
            (j.run.agent.as_deref(), j.run.trials),
            (Some("codex"), Some(2))
        );
    }
}
