//! Datasets: named, versioned task lists, resolved from a registry and
//! fetched at pinned git commits.
//!
//! Three sources are read:
//!
//! | Source | Shape |
//! |---|---|
//! | BenchFlow 0.8's dataset registry (the default first) | a JSON array of `{name, version, bench_version, tasks: [{name, git_url, git_commit_id, path, digest}]}` |
//! | Harbor's `registry.json` (the default second) | the same array, without digests: `{name, version, description, tasks: [{name, git_url, git_commit_id, path}], metrics}` |
//! | A local `dataset.toml` | `[dataset] name, version` and `[[tasks]] name, digest`, the tasks in folders next to it |
//!
//! `bench run -d name@version` looks the dataset up in BenchFlow's registry,
//! then Harbor's; `--registry <url-or-file>` names one registry instead.
//! Without `@version`, the version is `head` when the registry has one, else
//! the highest release-like version (PEP 440 ordering), else the last by
//! name.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::Deserialize;
use serde_json::Value;

use crate::source::{self, SourceError};

/// BenchFlow 0.8's dataset registry (skillsbench's), with task digests.
pub const BENCHFLOW_REGISTRY: &str =
    "https://raw.githubusercontent.com/benchflow-ai/skillsbench/main/registry.json";
/// Harbor's legacy registry: many public benchmarks, no task digests.
pub const HARBOR_REGISTRY: &str =
    "https://raw.githubusercontent.com/laude-institute/harbor/main/registry.json";
/// The registries searched, in order, when none is named.
pub const DEFAULT_REGISTRIES: [&str; 2] = [BENCHFLOW_REGISTRY, HARBOR_REGISTRY];

/// A fetched registry is reused from the cache for this long.
const CACHE_FRESH: Duration = Duration::from_secs(3600);

/// The file name of a local dataset.
pub const DATASET_FILE: &str = "dataset.toml";

/// `name@version`, or a bare `name` for the default version.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatasetRef {
    pub name: String,
    pub version: Option<String>,
}

impl std::fmt::Display for DatasetRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.version {
            Some(v) => write!(f, "{}@{v}", self.name),
            None => f.write_str(&self.name),
        }
    }
}

/// Parse `name@version` or `name`.
pub fn parse_ref(spec: &str) -> Result<DatasetRef, RegistryError> {
    let spec = spec.trim();
    let (name, version) = match spec.rsplit_once('@') {
        Some((n, v)) => (n, Some(v)),
        None => (spec, None),
    };
    if name.is_empty() || version.is_some_and(str::is_empty) {
        return Err(RegistryError::BadRef(spec.to_string()));
    }
    Ok(DatasetRef {
        name: name.to_string(),
        version: version.map(str::to_string),
    })
}

/// A dataset resolved from a registry: its identity and its tasks.
#[derive(Debug, Clone, PartialEq)]
pub struct Dataset {
    pub name: String,
    pub version: String,
    pub description: Option<String>,
    /// The registry it came from: a URL or a file.
    pub registry: String,
    /// The bench versions the dataset was validated against (0.8's
    /// `bench_version`), when the registry says.
    pub bench_version: Option<String>,
    pub tasks: Vec<DatasetTask>,
}

/// One task of a dataset.
#[derive(Debug, Clone, PartialEq)]
pub struct DatasetTask {
    pub name: String,
    pub location: Location,
    /// The content digest the registry pins, when it gives one.
    pub digest: Option<String>,
}

/// Where a dataset task lives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Location {
    /// A folder of a git repository at a commit (`HEAD` when unpinned).
    Git {
        url: String,
        commit: String,
        path: String,
    },
    /// A folder on this machine (a local `dataset.toml`).
    Local(PathBuf),
}

/// A dataset that could not be resolved or fetched.
#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    #[error("`{0}` is not a dataset name; write it as name@version, such as terminal-bench@2.0")]
    BadRef(String),
    #[error("could not read the registry {registry}: {message}")]
    Unreadable { registry: String, message: String },
    #[error("the registry {registry} is not valid: {message}")]
    Invalid { registry: String, message: String },
    #[error("no dataset `{wanted}` in {searched}{hint}")]
    NotFound {
        wanted: String,
        searched: String,
        hint: String,
    },
    #[error("{0}")]
    Local(String),
    #[error(transparent)]
    Fetch(#[from] SourceError),
    #[error("{dataset}: {count} task(s) do not match the digests the registry pins: {list}")]
    DigestMismatch {
        dataset: String,
        count: usize,
        list: String,
    },
}

/// A registry entry, as both BenchFlow's and Harbor's registries write it.
#[derive(Debug, Deserialize)]
struct Entry {
    name: String,
    version: Value,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    bench_version: Option<String>,
    #[serde(default)]
    tasks: Vec<EntryTask>,
}

#[derive(Debug, Deserialize)]
struct EntryTask {
    #[serde(default)]
    name: Option<String>,
    git_url: String,
    #[serde(default)]
    git_commit_id: Option<String>,
    path: String,
    #[serde(default)]
    digest: Option<String>,
}

fn version_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// A registry's entries, parsed from its JSON text.
pub fn parse_registry(text: &str, source: &str) -> Result<Vec<RegistryEntry>, RegistryError> {
    let invalid = |message: String| RegistryError::Invalid {
        registry: source.to_string(),
        message,
    };
    let raw: Value = serde_json::from_str(text).map_err(|e| invalid(e.to_string()))?;
    let Value::Array(items) = raw else {
        return Err(invalid("it is not a JSON array of datasets".into()));
    };
    let mut out = Vec::with_capacity(items.len());
    for (i, item) in items.into_iter().enumerate() {
        let e: Entry = serde_json::from_value(item)
            .map_err(|err| invalid(format!("dataset #{}: {err}", i + 1)))?;
        out.push(RegistryEntry {
            name: e.name,
            version: version_text(&e.version),
            description: e.description,
            bench_version: e.bench_version,
            tasks: e
                .tasks
                .into_iter()
                .map(|t| DatasetTask {
                    name: t.name.unwrap_or_else(|| last_segment(&t.path)),
                    location: Location::Git {
                        url: t.git_url,
                        commit: t
                            .git_commit_id
                            .filter(|c| !c.trim().is_empty())
                            .unwrap_or_else(|| "HEAD".into()),
                        path: t.path.trim_matches('/').to_string(),
                    },
                    digest: t.digest,
                })
                .collect(),
        });
    }
    Ok(out)
}

fn last_segment(path: &str) -> String {
    path.trim_end_matches('/')
        .rsplit('/')
        .next()
        .unwrap_or(path)
        .to_string()
}

/// One dataset version in a registry.
#[derive(Debug, Clone, PartialEq)]
pub struct RegistryEntry {
    pub name: String,
    pub version: String,
    pub description: Option<String>,
    pub bench_version: Option<String>,
    pub tasks: Vec<DatasetTask>,
}

/// Pick `wanted` from a registry's entries.
pub fn select(
    entries: &[RegistryEntry],
    wanted: &DatasetRef,
    source: &str,
) -> Result<Dataset, RegistryError> {
    let versions: Vec<&RegistryEntry> = entries.iter().filter(|e| e.name == wanted.name).collect();
    let entry = match &wanted.version {
        Some(v) => versions.iter().find(|e| &e.version == v).copied(),
        None => default_version(&versions),
    };
    let Some(entry) = entry else {
        let hint = if versions.is_empty() {
            let close = similar_names(entries, &wanted.name);
            if close.is_empty() {
                String::new()
            } else {
                format!("; similar names: {}", close.join(", "))
            }
        } else {
            let mut vs: Vec<&str> = versions.iter().map(|e| e.version.as_str()).collect();
            vs.sort_unstable();
            vs.dedup();
            format!("; its versions there: {}", vs.join(", "))
        };
        return Err(RegistryError::NotFound {
            wanted: wanted.to_string(),
            searched: source.to_string(),
            hint,
        });
    };
    Ok(Dataset {
        name: entry.name.clone(),
        version: entry.version.clone(),
        description: entry.description.clone(),
        registry: source.to_string(),
        bench_version: entry.bench_version.clone(),
        tasks: entry.tasks.clone(),
    })
}

/// What a dataset missing from every default registry adds to its error:
/// the similar names they hold (a typo), and the registries not read.
fn not_found_hint(similar: &[String], unreadable: &[String]) -> String {
    let mut hint = String::new();
    if !similar.is_empty() {
        hint.push_str(&format!("; similar names: {}", similar.join(", ")));
    }
    if !unreadable.is_empty() {
        hint.push_str(&format!(" ({})", unreadable.join("; ")));
    }
    hint
}

fn similar_names(entries: &[RegistryEntry], name: &str) -> Vec<String> {
    let lower = name.to_ascii_lowercase();
    let mut all: Vec<String> = entries.iter().map(|e| e.name.clone()).collect();
    all.sort();
    all.dedup();
    let mut names: Vec<String> = all
        .iter()
        .filter(|n| {
            let l = n.to_ascii_lowercase();
            l.contains(&lower) || lower.contains(&l)
        })
        .cloned()
        .collect();
    for close in
        benchflow_task::suggest::close_matches(name, all.iter().map(String::as_str), 3, 0.75)
    {
        if !names.contains(&close) {
            names.push(close);
        }
    }
    names.truncate(5);
    names
}

/// `head`, else the highest release-like version, else the last by name.
fn default_version<'a>(versions: &[&'a RegistryEntry]) -> Option<&'a RegistryEntry> {
    if let Some(head) = versions.iter().find(|e| e.version == "head") {
        return Some(head);
    }
    let best = versions
        .iter()
        .filter_map(|e| Pep440::parse(&e.version).map(|v| (v, *e)))
        .max_by(|a, b| a.0.cmp(&b.0))
        .map(|(_, e)| e);
    best.or_else(|| {
        versions
            .iter()
            .max_by(|a, b| a.version.cmp(&b.version))
            .copied()
    })
}

/// A version in PEP 440's common forms: `1.2`, `v2.0.1`, `1.0rc1`,
/// `1.0.post2`, `1.0.dev3`. Ordered as PEP 440 orders them.
#[derive(Debug, Clone)]
struct Pep440 {
    release: Vec<u64>,
    /// dev < a < b < rc < final < post, with their numbers.
    stage: (u8, u64),
    post: u64,
}

impl Pep440 {
    fn parse(text: &str) -> Option<Self> {
        let t = text.trim().to_ascii_lowercase();
        let t = t.strip_prefix('v').unwrap_or(&t);
        let digits_end = t
            .char_indices()
            .find(|(_, c)| !(c.is_ascii_digit() || *c == '.'))
            .map(|(i, _)| i)
            .unwrap_or(t.len());
        let (rel, mut rest) = t.split_at(digits_end);
        let rel = rel.trim_end_matches('.');
        if rel.is_empty() {
            return None;
        }
        let release: Vec<u64> = rel
            .split('.')
            .map(|p| p.parse().ok())
            .collect::<Option<_>>()?;
        let mut stage = (4u8, 0u64);
        let mut post = 0;
        let number = |s: &str| -> Option<(u64, usize)> {
            let n = s.chars().take_while(char::is_ascii_digit).count();
            Some((if n == 0 { 0 } else { s[..n].parse().ok()? }, n))
        };
        for (tag, rank) in [("rc", 2u8), ("a", 0), ("b", 1), ("c", 2)] {
            let r = rest.strip_prefix('.').unwrap_or(rest);
            let r = r.strip_prefix('-').unwrap_or(r);
            if let Some(after) = r.strip_prefix(tag) {
                let (n, used) = number(after)?;
                stage = (rank + 1, n);
                rest = &after[used..];
                break;
            }
        }
        let r = rest.strip_prefix('.').unwrap_or(rest);
        if let Some(after) = r.strip_prefix("post") {
            let (n, used) = number(after)?;
            post = n + 1;
            rest = &after[used..];
        }
        let r = rest.strip_prefix('.').unwrap_or(rest);
        if let Some(after) = r.strip_prefix("dev") {
            let (n, used) = number(after)?;
            if stage.0 == 4 && post == 0 {
                stage = (0, n);
            }
            rest = &after[used..];
        }
        rest.is_empty().then_some(Self {
            release,
            stage,
            post,
        })
    }
}

impl Ord for Pep440 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let n = self.release.len().max(other.release.len());
        let pad = |v: &[u64]| {
            let mut v = v.to_vec();
            v.resize(n, 0);
            v
        };
        pad(&self.release)
            .cmp(&pad(&other.release))
            .then(self.stage.cmp(&other.stage))
            .then(self.post.cmp(&other.post))
    }
}

impl PartialEq for Pep440 {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}

impl Eq for Pep440 {}

impl PartialOrd for Pep440 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// A local `dataset.toml`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DatasetFile {
    dataset: DatasetHeader,
    #[serde(default)]
    tasks: Vec<DatasetFileTask>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DatasetHeader {
    name: String,
    version: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    bench_version: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DatasetFileTask {
    name: String,
    #[serde(default)]
    digest: Option<String>,
    /// The task's folder, relative to the file; by default the folder named
    /// like the task, next to the file or under `tasks/`.
    #[serde(default)]
    path: Option<PathBuf>,
}

/// The `dataset.toml` a path names: the file itself, or one in the folder.
pub fn local_dataset_file(path: &Path) -> Option<PathBuf> {
    if path.is_file() && path.extension().is_some_and(|e| e == "toml") {
        return Some(path.to_path_buf());
    }
    let inside = path.join(DATASET_FILE);
    inside.is_file().then_some(inside)
}

/// Read a local `dataset.toml`, finding each task's folder under the file's.
pub fn read_dataset_file(path: &Path) -> Result<Dataset, RegistryError> {
    let local = |m: String| RegistryError::Local(format!("{}: {m}", path.display()));
    let text = std::fs::read_to_string(path).map_err(|e| local(e.to_string()))?;
    let file: DatasetFile =
        toml::from_str(&text).map_err(|e| local(e.to_string().trim().to_string()))?;
    let base = path.parent().unwrap_or(Path::new("."));
    let mut tasks = Vec::new();
    let mut missing = Vec::new();
    for t in file.tasks {
        let candidates: Vec<PathBuf> = match &t.path {
            Some(p) => vec![base.join(p)],
            None => vec![base.join(&t.name), base.join("tasks").join(&t.name)],
        };
        match candidates.into_iter().find(|c| c.is_dir()) {
            Some(dir) => tasks.push(DatasetTask {
                name: t.name,
                location: Location::Local(dir),
                digest: t.digest,
            }),
            None => missing.push(t.name),
        }
    }
    if !missing.is_empty() {
        return Err(local(format!(
            "no folder for task(s) {}; put each in a folder named like it next to {DATASET_FILE} (or under tasks/), or give its `path`",
            missing.join(", ")
        )));
    }
    if tasks.is_empty() {
        return Err(local("lists no [[tasks]]".into()));
    }
    Ok(Dataset {
        name: file.dataset.name,
        version: file.dataset.version,
        description: file.dataset.description,
        registry: path.display().to_string(),
        bench_version: file.dataset.bench_version,
        tasks,
    })
}

/// Read a registry from a URL (cached for an hour) or a file.
pub async fn load(source: &str, cache: &Path) -> Result<Vec<RegistryEntry>, RegistryError> {
    let text = if source.starts_with("http://") || source.starts_with("https://") {
        fetch_cached(source, cache).await?
    } else {
        std::fs::read_to_string(source).map_err(|e| RegistryError::Unreadable {
            registry: source.to_string(),
            message: e.to_string(),
        })?
    };
    parse_registry(&text, source)
}

async fn fetch_cached(url: &str, cache: &Path) -> Result<String, RegistryError> {
    use sha2::{Digest, Sha256};
    let key = hex::encode(Sha256::digest(url.as_bytes()));
    let file = cache
        .join("registries")
        .join(format!("{}.json", &key[..16]));
    let fresh = std::fs::metadata(&file)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.elapsed().ok())
        .is_some_and(|age| age < CACHE_FRESH);
    if fresh && let Ok(text) = std::fs::read_to_string(&file) {
        return Ok(text);
    }
    let fetched = async {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .map_err(|e| e.to_string())?;
        let resp = client
            .get(url)
            .send()
            .await
            .map_err(|e| e.to_string())?
            .error_for_status()
            .map_err(|e| e.to_string())?;
        resp.text().await.map_err(|e| e.to_string())
    }
    .await;
    match fetched {
        Ok(text) => {
            if let Some(parent) = file.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let tmp = file.with_extension("json.tmp");
            if std::fs::write(&tmp, &text).is_ok() {
                let _ = std::fs::rename(&tmp, &file);
            }
            Ok(text)
        }
        // Offline: an older copy beats nothing.
        Err(message) => std::fs::read_to_string(&file).map_err(|_| RegistryError::Unreadable {
            registry: url.to_string(),
            message,
        }),
    }
}

/// Resolve a dataset: a local `dataset.toml` (when `spec` or `registry`
/// names one), else `spec` in the named registry, else in BenchFlow's
/// registry and then Harbor's.
pub async fn resolve(
    spec: &str,
    registry: Option<&str>,
    cache: &Path,
) -> Result<Dataset, RegistryError> {
    if let Some(file) = local_dataset_file(Path::new(spec)) {
        return read_dataset_file(&file);
    }
    let wanted = parse_ref(spec)?;
    if let Some(reg) = registry {
        if let Some(file) = local_dataset_file(Path::new(reg))
            .filter(|f| f.extension().is_some_and(|e| e == "toml"))
        {
            let d = read_dataset_file(&file)?;
            let version_ok = wanted.version.as_ref().is_none_or(|v| v == &d.version);
            if d.name != wanted.name || !version_ok {
                return Err(RegistryError::NotFound {
                    wanted: wanted.to_string(),
                    searched: reg.to_string(),
                    hint: format!("; it holds {}@{}", d.name, d.version),
                });
            }
            return Ok(d);
        }
        let entries = load(reg, cache).await?;
        return select(&entries, &wanted, reg);
    }
    let mut unreadable = Vec::new();
    let mut similar: Vec<String> = Vec::new();
    for reg in DEFAULT_REGISTRIES {
        match load(reg, cache).await {
            Ok(entries) => match select(&entries, &wanted, reg) {
                Ok(d) => return Ok(d),
                // A version miss in a registry that has the name is final.
                Err(e @ RegistryError::NotFound { .. })
                    if entries.iter().any(|x| x.name == wanted.name) =>
                {
                    return Err(e);
                }
                Err(_) => {
                    for name in similar_names(&entries, &wanted.name) {
                        if !similar.contains(&name) {
                            similar.push(name);
                        }
                    }
                }
            },
            Err(e) => unreadable.push(e.to_string()),
        }
    }
    let searched = DEFAULT_REGISTRIES.join(" or ");
    Err(RegistryError::NotFound {
        wanted: wanted.to_string(),
        searched,
        hint: not_found_hint(&similar, &unreadable),
    })
}

/// A dataset task on disk.
#[derive(Debug, Clone, PartialEq)]
pub struct FetchedTask {
    pub name: String,
    pub path: PathBuf,
    /// The registry's digest, when it pins one.
    pub digest: Option<String>,
    /// The commit fetched, for git tasks.
    pub commit: Option<String>,
    pub check: DigestCheck,
}

/// Whether a task's content matches the digest its registry pins.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DigestCheck {
    Matched,
    Mismatched {
        expected: String,
        actual: String,
    },
    /// The registry gives no digest for the task.
    NotPinned,
    /// The digest could not be checked; the reason says why.
    Skipped(String),
}

/// Fetch every task of a dataset: one sparse, blobless checkout per
/// repository and commit, holding all of that commit's task folders. Then
/// check each task against the digest the registry pins. `progress` hears
/// each repository as it is fetched.
pub async fn fetch(
    dataset: &Dataset,
    cache: &Path,
    progress: &(dyn Fn(&str) + Send + Sync),
) -> Result<Vec<FetchedTask>, RegistryError> {
    let mut groups: BTreeMap<(String, String), Vec<String>> = BTreeMap::new();
    for t in &dataset.tasks {
        if let Location::Git { url, commit, path } = &t.location {
            groups
                .entry((url.clone(), commit.clone()))
                .or_default()
                .push(path.clone());
        }
    }
    let mut checkouts: BTreeMap<(String, String), source::GitCheckout> = BTreeMap::new();
    for ((url, commit), paths) in groups {
        progress(&format!(
            "{url} at {} ({} folder{})",
            short(&commit),
            paths.len(),
            if paths.len() == 1 { "" } else { "s" }
        ));
        let co = source::fetch_git(&url, &commit, &paths, cache).await?;
        checkouts.insert((url, commit), co);
    }
    let mut out = Vec::new();
    for t in &dataset.tasks {
        let (path, commit) = match &t.location {
            Location::Local(p) => (p.clone(), None),
            Location::Git { url, commit, path } => {
                let co = &checkouts[&(url.clone(), commit.clone())];
                (co.dir.join(path), Some(co.commit.clone()))
            }
        };
        let check = verify_digest(&path, t.digest.as_deref());
        out.push(FetchedTask {
            name: t.name.clone(),
            path,
            digest: t.digest.clone(),
            commit,
            check,
        });
    }
    let bad: Vec<String> = out
        .iter()
        .filter_map(|f| match &f.check {
            DigestCheck::Mismatched { expected, actual } => {
                Some(format!("{} (computed {actual}, pinned {expected})", f.name))
            }
            _ => None,
        })
        .collect();
    if !bad.is_empty() {
        return Err(RegistryError::DigestMismatch {
            dataset: format!("{}@{}", dataset.name, dataset.version),
            count: bad.len(),
            list: bad.join("; "),
        });
    }
    Ok(out)
}

fn short(commit: &str) -> &str {
    if commit.len() > 12 && commit.chars().all(|c| c.is_ascii_hexdigit()) {
        &commit[..12]
    } else {
        commit
    }
}

/// Check a task folder against a pinned digest.
pub fn verify_digest(dir: &Path, expected: Option<&str>) -> DigestCheck {
    let Some(expected) = expected else {
        return DigestCheck::NotPinned;
    };
    match task_digests(dir) {
        Some(actual) if actual.iter().any(|d| d == expected) => DigestCheck::Matched,
        Some(actual) => DigestCheck::Mismatched {
            expected: expected.to_string(),
            actual: actual.into_iter().next().unwrap_or_default(),
        },
        None => DigestCheck::Skipped(DIGEST_UNAVAILABLE.into()),
    }
}

/// Why a digest went unchecked.
pub const DIGEST_UNAVAILABLE: &str = "the task folder's digest could not be computed (a file could not be read), so the registry's digest is recorded in lock.json and result.json but not checked";

/// The content digests a registry may pin, computed for the folder:
/// BenchFlow 0.8's task digest (its registry's `digest`), then Harbor's
/// package hash. `None` when the folder cannot be read.
fn task_digests(dir: &Path) -> Option<Vec<String>> {
    let benchflow = benchflow_task::digest::benchflow_digest(dir).ok()?;
    let mut out = vec![benchflow];
    if let Ok(harbor) = benchflow_task::digest::harbor_digest(dir) {
        out.push(harbor);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const REGISTRY: &str = r#"[
      {"name": "demo", "version": "1.0", "description": "first", "tasks": [
        {"name": "a", "git_url": "https://github.com/org/repo.git", "git_commit_id": "abc", "path": "tasks/a"}
      ]},
      {"name": "demo", "version": "1.10", "description": "later", "bench_version": ">=0.8", "tasks": [
        {"name": "a", "git_url": "https://github.com/org/repo.git", "git_commit_id": "def", "path": "tasks/a", "digest": "sha256:aa"},
        {"git_url": "https://github.com/org/repo.git", "git_commit_id": "def", "path": "tasks/b/"}
      ], "metrics": [{"type": "mean"}]},
      {"name": "demo", "version": "1.2rc1", "tasks": []},
      {"name": "other", "version": "head", "tasks": []},
      {"name": "other", "version": "9.0", "tasks": []},
      {"name": "words", "version": "easy", "tasks": []},
      {"name": "words", "version": "hard", "tasks": []}
    ]"#;

    #[test]
    fn parses_refs() {
        assert_eq!(
            parse_ref("terminal-bench@2.0").unwrap(),
            DatasetRef {
                name: "terminal-bench".into(),
                version: Some("2.0".into())
            }
        );
        assert_eq!(parse_ref("demo").unwrap().version, None);
        assert!(parse_ref("demo@").is_err());
        assert!(parse_ref("@1.0").is_err());
    }

    #[test]
    fn picks_versions() {
        let entries = parse_registry(REGISTRY, "reg.json").unwrap();
        let pick = |s: &str| select(&entries, &parse_ref(s).unwrap(), "reg.json");
        let d = pick("demo@1.0").unwrap();
        assert_eq!(d.description.as_deref(), Some("first"));
        // 1.10 > 1.2rc1 > 1.0 in release order, not string order.
        let d = pick("demo").unwrap();
        assert_eq!(d.version, "1.10");
        assert_eq!(d.bench_version.as_deref(), Some(">=0.8"));
        assert_eq!(d.tasks.len(), 2);
        assert_eq!(d.tasks[0].digest.as_deref(), Some("sha256:aa"));
        assert_eq!(
            d.tasks[1].name, "b",
            "a task without a name takes its folder's"
        );
        assert_eq!(
            d.tasks[1].location,
            Location::Git {
                url: "https://github.com/org/repo.git".into(),
                commit: "def".into(),
                path: "tasks/b".into()
            }
        );
        assert_eq!(pick("other").unwrap().version, "head");
        assert_eq!(pick("words").unwrap().version, "hard", "lexically last");
        let e = pick("demo@3.0").unwrap_err().to_string();
        assert!(e.contains("1.0, 1.10, 1.2rc1"), "{e}");
        let e = pick("dem").unwrap_err().to_string();
        assert!(e.contains("similar names: demo"), "{e}");
    }

    #[test]
    fn orders_pep440() {
        let p = |s: &str| Pep440::parse(s).unwrap();
        assert!(p("1.0.dev1") < p("1.0a1"));
        assert!(p("1.0a1") < p("1.0b2"));
        assert!(p("1.0rc1") < p("1.0"));
        assert!(p("1.0") < p("1.0.post1"));
        assert!(p("1.0") == p("v1.0.0"));
        assert!(p("2.0") > p("1.99"));
        assert!(Pep440::parse("lite200").is_none());
        assert!(Pep440::parse("head").is_none());
    }

    #[test]
    fn rejects_registries_that_are_not_arrays() {
        let e = parse_registry("{}", "x.json").unwrap_err().to_string();
        assert!(e.contains("not a JSON array"), "{e}");
        let e = parse_registry(r#"[{"version": "1"}]"#, "x.json")
            .unwrap_err()
            .to_string();
        assert!(e.contains("dataset #1") && e.contains("name"), "{e}");
    }

    #[test]
    fn reads_a_local_dataset_file() {
        let d = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(d.path().join("hello")).unwrap();
        std::fs::create_dir_all(d.path().join("tasks/world")).unwrap();
        std::fs::write(
            d.path().join(DATASET_FILE),
            "[dataset]\nname = \"mine\"\nversion = \"0.1\"\n\n[[tasks]]\nname = \"hello\"\ndigest = \"sha256:11\"\n\n[[tasks]]\nname = \"world\"\n",
        )
        .unwrap();
        let ds = read_dataset_file(&local_dataset_file(d.path()).unwrap()).unwrap();
        assert_eq!((ds.name.as_str(), ds.version.as_str()), ("mine", "0.1"));
        assert_eq!(
            ds.tasks[0].location,
            Location::Local(d.path().join("hello"))
        );
        assert_eq!(
            ds.tasks[1].location,
            Location::Local(d.path().join("tasks/world"))
        );
        std::fs::write(
            d.path().join(DATASET_FILE),
            "[dataset]\nname = \"mine\"\nversion = \"0.1\"\n\n[[tasks]]\nname = \"gone\"\n",
        )
        .unwrap();
        let e = read_dataset_file(&d.path().join(DATASET_FILE))
            .unwrap_err()
            .to_string();
        assert!(e.contains("no folder for task(s) gone"), "{e}");
    }

    #[test]
    fn pinned_digests_are_checked_against_the_folder() {
        let d = tempfile::tempdir().unwrap();
        std::fs::write(d.path().join("task.toml"), "[task]\nname = \"org/t\"\n").unwrap();
        std::fs::write(d.path().join("instruction.md"), "Do it.\n").unwrap();
        assert_eq!(verify_digest(d.path(), None), DigestCheck::NotPinned);
        let benchflow = benchflow_task::digest::benchflow_digest(d.path()).unwrap();
        let harbor = benchflow_task::digest::harbor_digest(d.path()).unwrap();
        assert_eq!(
            verify_digest(d.path(), Some(&benchflow)),
            DigestCheck::Matched
        );
        assert_eq!(verify_digest(d.path(), Some(&harbor)), DigestCheck::Matched);
        assert!(matches!(
            verify_digest(d.path(), Some("sha256:00")),
            DigestCheck::Mismatched { .. }
        ));
        assert!(matches!(
            verify_digest(&d.path().join("gone"), Some("sha256:00")),
            DigestCheck::Skipped(_)
        ));
    }

    #[tokio::test]
    async fn resolves_from_a_registry_file() {
        let d = tempfile::tempdir().unwrap();
        let reg = d.path().join("registry.json");
        std::fs::write(&reg, REGISTRY).unwrap();
        let ds = resolve("demo@1.0", Some(reg.to_str().unwrap()), d.path())
            .await
            .unwrap();
        assert_eq!(ds.registry, reg.display().to_string());
        assert_eq!(ds.tasks.len(), 1);
    }

    /// Harbor's real registry, fetched by hand (about 13 MB; not committed):
    /// `BENCH_HARBOR_REGISTRY=/path/to/registry.json cargo test -p benchflow-run harbor`.
    #[test]
    fn resolves_terminal_bench_from_harbors_registry() {
        let Some(path) = std::env::var_os("BENCH_HARBOR_REGISTRY") else {
            return;
        };
        let text = std::fs::read_to_string(path).unwrap();
        let entries = parse_registry(&text, HARBOR_REGISTRY).unwrap();
        let d = select(
            &entries,
            &parse_ref("terminal-bench@2.0").unwrap(),
            HARBOR_REGISTRY,
        )
        .unwrap();
        assert_eq!(d.tasks.len(), 89);
        let Location::Git { url, commit, .. } = &d.tasks[0].location else {
            panic!("git task");
        };
        assert_eq!(
            url,
            "https://github.com/laude-institute/terminal-bench-2.git"
        );
        assert_eq!(commit.len(), 40);
        assert!(d.tasks.iter().all(|t| t.digest.is_none()));
    }

    #[test]
    fn a_dataset_missing_everywhere_names_similar_ones() {
        assert_eq!(not_found_hint(&[], &[]), "");
        assert_eq!(
            not_found_hint(&["terminal-bench".into()], &[]),
            "; similar names: terminal-bench"
        );
        assert_eq!(not_found_hint(&[], &["offline".into()]), " (offline)");
    }
}
