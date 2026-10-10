//! The public ACP registry: `-a acp:<id>[@<version>]`.
//!
//! The [Agent Client Protocol registry](https://agentclientprotocol.com/get-started/registry)
//! lists agents that speak ACP over stdio, each with a version and a
//! distribution: per-platform `binary` archives, an `npx` package, or a
//! `uvx` package. BenchFlow turns an entry into an [`AgentManifest`] on the
//! fly ([`Entry::to_manifest`]): it installs the distribution for the
//! sandbox's platform, launches the entry's command, and selects the model
//! the `auto` way. The registry does not say which login an agent needs, so
//! an entry runs on no login, unless a built-in manifest names the same
//! agent (`acp_registry = "<id>"`), in which case that manifest's logins
//! and launch settings apply.
//!
//! The index is fetched from [`REGISTRY_URL`] and cached for a day under
//! the user's cache folder ([`load`]). An older version than the one the
//! index lists is looked up in the registry's git history and cached for
//! good. With no network and no cache, BenchFlow uses the copy of the index
//! built into it ([`SNAPSHOT_DATE`]); `BENCHFLOW_ACP_REGISTRY=<file>` reads
//! a local copy instead (tests, offline machines).

use std::fmt;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::manifest::distribution::PLATFORMS;
use crate::manifest::{AgentManifest, ManifestSource, Registry};

/// The registry's index, as the registry publishes it.
pub const REGISTRY_URL: &str =
    "https://cdn.agentclientprotocol.com/registry/v1/latest/registry.json";
/// The registry's git repository on GitHub, for older versions.
const REPOSITORY: &str = "agentclientprotocol/registry";
/// What `--agent` names start with to mean a registry entry.
pub const PREFIX: &str = "acp:";
/// How long a fetched index is used before it is fetched again.
pub const MAX_AGE: Duration = Duration::from_secs(24 * 60 * 60);
/// A local registry index to use instead of the network.
pub const FILE_ENV: &str = "BENCHFLOW_ACP_REGISTRY";
/// The index built into this binary, for machines with no network.
pub const SNAPSHOT: &str = include_str!("../testdata/acp-registry-2026-10-10.json");
/// When [`SNAPSHOT`] was fetched.
pub const SNAPSHOT_DATE: &str = "2026-10-10";
const FETCH_TIMEOUT: Duration = Duration::from_secs(20);

/// The registry's index.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Index {
    #[serde(default)]
    pub version: String,
    pub agents: Vec<Entry>,
}

/// One agent in the registry.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Entry {
    pub id: String,
    #[serde(default)]
    pub name: String,
    pub version: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub repository: Option<String>,
    #[serde(default)]
    pub website: Option<String>,
    #[serde(default)]
    pub authors: Vec<String>,
    #[serde(default)]
    pub license: Option<String>,
    #[serde(default)]
    pub distribution: Distribution,
}

/// How an entry is distributed. An entry may offer several.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Distribution {
    #[serde(default)]
    pub binary: Option<IndexMap<String, BinaryTarget>>,
    #[serde(default)]
    pub npx: Option<Package>,
    #[serde(default)]
    pub uvx: Option<Package>,
}

/// One platform's archive.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BinaryTarget {
    pub archive: String,
    pub cmd: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: IndexMap<String, String>,
    #[serde(default, alias = "checksum")]
    pub sha256: Option<String>,
}

/// An `npx` or `uvx` package.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Package {
    pub package: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: IndexMap<String, String>,
}

/// Why an entry cannot become a manifest. The fix is the entry's.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refusal {
    pub id: String,
    pub version: String,
    pub reason: String,
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "acp:{}@{} cannot run in a sandbox: {}",
            self.id, self.version, self.reason
        )
    }
}

impl std::error::Error for Refusal {}

/// True when `name` names a registry entry (`acp:goose`).
pub fn is_spec(name: &str) -> bool {
    name.starts_with(PREFIX)
}

/// `acp:<id>[@<version>]` (the `acp:` is optional) as its id and version.
pub fn parse_spec(spec: &str) -> Result<(String, Option<String>), RegistryError> {
    let rest = spec.trim().strip_prefix(PREFIX).unwrap_or(spec.trim());
    let (id, version) = match rest.split_once('@') {
        Some((id, version)) => (id, Some(version)),
        None => (rest, None),
    };
    let bad = |why: &str| RegistryError::BadSpec {
        spec: spec.to_string(),
        why: why.to_string(),
    };
    if id.is_empty() {
        return Err(bad("it names no agent id"));
    }
    if !id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
    {
        return Err(bad("an id is letters, digits, `-`, `_` and `.`"));
    }
    match version {
        Some("") => Err(bad("the version after `@` is empty")),
        Some(v) if v.chars().any(|c| c.is_whitespace() || c == '/') => {
            Err(bad("the version has spaces or slashes"))
        }
        _ => Ok((id.to_string(), version.map(str::to_string))),
    }
}

impl Index {
    /// Parse the registry's `registry.json`.
    pub fn parse(text: &str) -> Result<Self, String> {
        serde_json::from_str(text).map_err(|e| format!("not an ACP registry index: {e}"))
    }

    /// The index built into this binary.
    pub fn snapshot() -> Self {
        Self::parse(SNAPSHOT).expect("BenchFlow bug: the built-in ACP registry snapshot parses")
    }

    pub fn find(&self, id: &str) -> Option<&Entry> {
        self.agents.iter().find(|e| e.id == id)
    }

    /// Up to `n` ids most like `id`, best first.
    pub fn closest(&self, id: &str, n: usize) -> Vec<String> {
        let id = id.to_ascii_lowercase();
        let mut scored: Vec<(f64, &str)> = self
            .agents
            .iter()
            .map(|e| {
                let mut score = strsim::jaro_winkler(&id, &e.id.to_ascii_lowercase());
                if e.id.contains(&id) || id.contains(&e.id) {
                    score += 0.2;
                }
                (score, e.id.as_str())
            })
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        scored
            .into_iter()
            .filter(|(score, _)| *score >= 0.72)
            .take(n)
            .map(|(_, id)| id.to_string())
            .collect()
    }
}

/// The distribution BenchFlow installs for an entry.
enum Chosen<'a> {
    Binary(Vec<(&'static str, &'a BinaryTarget)>),
    Npx(&'a Package),
    Uvx(&'a Package),
}

impl Entry {
    /// `acp:<id>@<version>`.
    pub fn spec(&self) -> String {
        format!("{PREFIX}{}@{}", self.id, self.version)
    }

    /// The entry's Linux builds, in [`PLATFORMS`] order.
    pub fn linux_targets(&self) -> Vec<(&'static str, &BinaryTarget)> {
        let Some(binary) = &self.distribution.binary else {
            return Vec::new();
        };
        PLATFORMS
            .iter()
            .filter_map(|(platform, _)| binary.get(*platform).map(|t| (*platform, t)))
            .collect()
    }

    /// What BenchFlow would install, in words, for `bench agents --registry`.
    pub fn install_summary(&self) -> String {
        match self.choose() {
            Ok(Chosen::Binary(targets)) => {
                let arches: Vec<&str> = targets
                    .iter()
                    .map(|(p, _)| p.trim_start_matches("linux-"))
                    .collect();
                format!("binary ({})", arches.join(", "))
            }
            Ok(Chosen::Npx(p)) => format!("npx {}", p.package),
            Ok(Chosen::Uvx(p)) => format!("uvx {}", p.package),
            Err(reason) => format!("none: {reason}"),
        }
    }

    /// Prefer a binary with a build for every Linux platform (no runtime to
    /// install), then npx, then uvx, then a binary for some platforms.
    fn choose(&self) -> Result<Chosen<'_>, String> {
        let linux = self.linux_targets();
        if linux.len() == PLATFORMS.len() {
            return Ok(Chosen::Binary(linux));
        }
        if let Some(npx) = &self.distribution.npx {
            return Ok(Chosen::Npx(npx));
        }
        if let Some(uvx) = &self.distribution.uvx {
            return Ok(Chosen::Uvx(uvx));
        }
        if !linux.is_empty() {
            return Ok(Chosen::Binary(linux));
        }
        match &self.distribution.binary {
            Some(binary) if !binary.is_empty() => {
                let platforms: Vec<&str> = binary.keys().map(String::as_str).collect();
                Err(format!(
                    "it publishes builds only for {}, and sandboxes run Linux",
                    platforms.join(", ")
                ))
            }
            _ => Err("it lists no binary, npx, or uvx distribution".to_string()),
        }
    }

    /// The manifest that runs this entry. `agents` supplies the logins and
    /// launch settings of a built-in manifest that names the same agent
    /// (`acp_registry = "<id>"`).
    pub fn to_manifest(&self, agents: &Registry) -> Result<AgentManifest, Refusal> {
        let refuse = |reason: String| Refusal {
            id: self.id.clone(),
            version: self.version.clone(),
            reason,
        };
        let chosen = self.choose().map_err(refuse)?;
        let mut install = toml::Table::new();
        let (args, env) = match &chosen {
            Chosen::Binary(targets) => {
                let mut binary = toml::Table::new();
                for (platform, t) in targets {
                    let mut target = toml::Table::new();
                    target.insert("archive".into(), t.archive.clone().into());
                    target.insert("cmd".into(), t.cmd.clone().into());
                    if let Some(sha) = &t.sha256 {
                        target.insert("sha256".into(), sha.clone().into());
                    }
                    binary.insert((*platform).to_string(), target.into());
                }
                install.insert("binary".into(), binary.into());
                let (_, first) = targets[0];
                if let Some((platform, _)) = targets
                    .iter()
                    .find(|(_, t)| t.args != first.args || t.env != first.env)
                {
                    return Err(refuse(format!(
                        "its {platform} build starts with other arguments or variables than its {} build",
                        targets[0].0
                    )));
                }
                (first.args.clone(), first.env.clone())
            }
            Chosen::Npx(p) => {
                install.insert("npx".into(), p.package.clone().into());
                (p.args.clone(), p.env.clone())
            }
            Chosen::Uvx(p) => {
                install.insert("uvx".into(), p.package.clone().into());
                (p.args.clone(), p.env.clone())
            }
        };
        let mut launch = toml::Table::new();
        launch.insert(
            "args".into(),
            toml::Value::Array(args.into_iter().map(Into::into).collect()),
        );
        launch.insert(
            "env".into(),
            toml::Value::Table(env.into_iter().map(|(k, v)| (k, v.into())).collect()),
        );
        let mut doc = toml::Table::new();
        doc.insert("name".into(), self.id.clone().into());
        let title = if self.name.is_empty() {
            &self.id
        } else {
            &self.name
        };
        let mut description = format!("{title} {} from the public ACP registry", self.version);
        if !self.description.is_empty() {
            description.push_str(": ");
            description.push_str(&self.description);
        }
        doc.insert("description".into(), description.into());
        doc.insert("install".into(), install.into());
        doc.insert("launch".into(), launch.into());
        let text = toml::to_string(&doc).map_err(|e| refuse(e.to_string()))?;
        let spec = self.spec();
        let mut m = AgentManifest::parse(&text, ManifestSource::AcpRegistry(spec.clone()))
            .map_err(|e| refuse(manifest_reason(&e.to_string(), &spec)))?;

        m.name = spec;
        m.acp_registry = Some(self.id.clone());
        let builtin = agents
            .agents()
            .iter()
            .find(|a| a.acp_registry.as_deref() == Some(self.id.as_str()));
        match builtin {
            Some(b) => {
                m.credentials = b.credentials.clone();
                m.account = b.account.clone();
                m.skill_paths = b.skill_paths.clone();
                m.skills_via = b.skills_via.clone();
                m.home_dirs = b.home_dirs.clone();
                for (k, v) in &b.launch.env {
                    m.launch.env.entry(k.clone()).or_insert_with(|| v.clone());
                }
                m.notes.push(format!(
                    "BenchFlow's `{}` agent runs the same agent, so this entry uses its logins, skill paths, and launch variables with the registry's build",
                    b.name
                ));
            }
            None => m.notes.push(
                "The ACP registry does not say which login this agent needs, so BenchFlow gives it none; pass a key the agent reads with `--env KEY=VALUE`".to_string(),
            ),
        }
        Ok(m)
    }
}

/// A manifest error about a generated manifest, as the entry's problem.
fn manifest_reason(error: &str, spec: &str) -> String {
    let prefix = format!("{}: ", ManifestSource::AcpRegistry(spec.to_string()));
    error.strip_prefix(&prefix).unwrap_or(error).to_string()
}

/// Why `acp:<id>[@<version>]` gives no agent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegistryError {
    /// The name is not `acp:<id>[@<version>]`.
    BadSpec { spec: String, why: String },
    /// The registry has no such id.
    UnknownId { id: String, closest: Vec<String> },
    /// The registry has the id, but never listed that version.
    UnknownVersion {
        id: String,
        version: String,
        /// The version the index lists now.
        listed: String,
        /// Other versions seen in the registry's history.
        seen: Vec<String>,
        /// Why the history could not be read, when it could not.
        history: Option<String>,
    },
    /// The entry exists but cannot run in a sandbox.
    Refused(Refusal),
}

impl RegistryError {
    /// What the user should do.
    pub fn next_step(&self) -> String {
        match self {
            Self::BadSpec { .. } => {
                "write it as acp:<id> or acp:<id>@<version>; `bench agents --registry` lists the ids"
                    .to_string()
            }
            Self::UnknownId { .. } => "`bench agents --registry` lists every entry".to_string(),
            Self::UnknownVersion { id, listed, .. } => {
                format!("use acp:{id}@{listed}, or acp:{id} for whatever the registry lists")
            }
            Self::Refused(_) => {
                "the fix is the registry entry's (report it to the agent's maintainers); `bench agents --registry` shows what each entry would install"
                    .to_string()
            }
        }
    }
}

impl fmt::Display for RegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadSpec { spec, why } => {
                write!(f, "`{spec}` is not an ACP registry agent: {why}")
            }
            Self::UnknownId { id, closest } => {
                write!(f, "the public ACP registry has no agent `{id}`")?;
                if !closest.is_empty() {
                    let ids: Vec<String> = closest.iter().map(|c| format!("`{c}`")).collect();
                    write!(f, "; closest: {}", ids.join(", "))?;
                }
                Ok(())
            }
            Self::UnknownVersion {
                id,
                version,
                listed,
                seen,
                history,
            } => {
                write!(
                    f,
                    "the public ACP registry has no {id} {version}; it lists {id} {listed}"
                )?;
                let older: Vec<&String> = seen.iter().filter(|v| *v != listed).collect();
                if !older.is_empty() {
                    let older: Vec<&str> = older.iter().take(8).map(|v| v.as_str()).collect();
                    write!(f, " (and listed {} before)", older.join(", "))?;
                }
                if let Some(why) = history {
                    write!(f, "; its older versions could not be read: {why}")?;
                }
                Ok(())
            }
            Self::Refused(r) => write!(f, "{r}"),
        }
    }
}

impl std::error::Error for RegistryError {}

/// Resolve `spec` against `index` alone. A version the index does not
/// list is [`RegistryError::UnknownVersion`]; [`resolve`] then reads the
/// registry's history.
pub fn resolve_in(
    index: &Index,
    spec: &str,
    agents: &Registry,
) -> Result<AgentManifest, RegistryError> {
    let (id, version) = parse_spec(spec)?;
    let entry = index.find(&id).ok_or_else(|| RegistryError::UnknownId {
        closest: index.closest(&id, 3),
        id: id.clone(),
    })?;
    if let Some(version) = version
        && version != entry.version
    {
        return Err(RegistryError::UnknownVersion {
            id,
            version,
            listed: entry.version.clone(),
            seen: Vec::new(),
            history: None,
        });
    }
    entry.to_manifest(agents).map_err(RegistryError::Refused)
}

// ---------------------------------------------------------------------------
// Fetching and caching
// ---------------------------------------------------------------------------

/// Where the index comes from.
#[derive(Debug, Clone, Default)]
pub struct Source {
    /// BenchFlow's cache folder (`~/.cache/benchflow`); `None` caches nothing.
    pub cache_dir: Option<PathBuf>,
    /// A local index to use instead of the network ([`FILE_ENV`]).
    pub file: Option<PathBuf>,
}

impl Source {
    /// The cache folder, plus [`FILE_ENV`] when it is set.
    pub fn new(cache_dir: PathBuf) -> Self {
        Self {
            cache_dir: Some(cache_dir),
            file: std::env::var_os(FILE_ENV)
                .filter(|v| !v.is_empty())
                .map(PathBuf::from),
        }
    }

    fn index_path(&self) -> Option<PathBuf> {
        self.cache_dir
            .as_ref()
            .map(|d| d.join("acp-registry").join("registry.json"))
    }

    fn version_path(&self, id: &str, version: &str) -> Option<PathBuf> {
        self.cache_dir.as_ref().map(|d| {
            d.join("acp-registry")
                .join("versions")
                .join(id)
                .join(format!("{version}.json"))
        })
    }
}

/// Where a loaded index came from, for messages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Origin {
    File(PathBuf),
    Fetched,
    Cache {
        age: Duration,
    },
    /// The cache was older than a day and the fetch failed.
    StaleCache {
        age: Duration,
        error: String,
    },
    /// No cache and the fetch failed: the copy built into BenchFlow.
    Snapshot {
        error: String,
    },
}

impl Origin {
    /// One line for people, or `None` when the index is fresh.
    pub fn warning(&self) -> Option<String> {
        match self {
            Self::StaleCache { age, error } => Some(format!(
                "could not fetch the ACP registry ({error}); using the copy cached {} ago",
                human_age(*age)
            )),
            Self::Snapshot { error } => Some(format!(
                "could not fetch the ACP registry ({error}); using the copy built into bench ({SNAPSHOT_DATE})"
            )),
            _ => None,
        }
    }

    pub fn describe(&self) -> String {
        match self {
            Self::File(p) => format!("{} ({FILE_ENV})", p.display()),
            Self::Fetched => format!("{REGISTRY_URL}, fetched now"),
            Self::Cache { age } => format!("{REGISTRY_URL}, cached {} ago", human_age(*age)),
            Self::StaleCache { age, .. } => {
                format!("{REGISTRY_URL}, cached {} ago (stale)", human_age(*age))
            }
            Self::Snapshot { .. } => format!("the copy built into bench ({SNAPSHOT_DATE})"),
        }
    }
}

fn human_age(age: Duration) -> String {
    let secs = age.as_secs();
    match secs {
        0..=119 => format!("{secs}s"),
        120..=7199 => format!("{}m", secs / 60),
        7200..=172_799 => format!("{}h", secs / 3600),
        _ => format!("{}d", secs / 86_400),
    }
}

/// An index ready to use, and where it came from.
#[derive(Debug, Clone)]
pub struct Loaded {
    pub index: Index,
    pub origin: Origin,
}

/// The index: the local file when set, else the cache when it is under a
/// day old, else a fresh fetch (cached), else the stale cache, else the
/// copy built into BenchFlow. Fails only when the local file is unusable.
pub async fn load(source: &Source) -> Result<Loaded, String> {
    if let Some(file) = &source.file {
        let text = std::fs::read_to_string(file)
            .map_err(|e| format!("cannot read {} ({FILE_ENV}): {e}", file.display()))?;
        let index =
            Index::parse(&text).map_err(|e| format!("{} ({FILE_ENV}): {e}", file.display()))?;
        return Ok(Loaded {
            index,
            origin: Origin::File(file.clone()),
        });
    }
    let cached = source.index_path().and_then(|p| read_cached(&p));
    if let Some((index, age)) = &cached
        && *age < MAX_AGE
    {
        return Ok(Loaded {
            index: index.clone(),
            origin: Origin::Cache { age: *age },
        });
    }
    match fetch_text(REGISTRY_URL)
        .await
        .and_then(|t| Index::parse(&t).map(|i| (i, t)))
    {
        Ok((index, text)) => {
            if let Some(path) = source.index_path() {
                let _ = write_atomic(&path, &text);
            }
            Ok(Loaded {
                index,
                origin: Origin::Fetched,
            })
        }
        Err(error) => Ok(match cached {
            Some((index, age)) => Loaded {
                index,
                origin: Origin::StaleCache { age, error },
            },
            None => Loaded {
                index: Index::snapshot(),
                origin: Origin::Snapshot { error },
            },
        }),
    }
}

/// Resolve `acp:<id>[@<version>]`: the loaded index, then for another
/// version the registry's history (cached for good).
pub async fn resolve(
    loaded: &Loaded,
    spec: &str,
    source: &Source,
    agents: &Registry,
) -> Result<AgentManifest, RegistryError> {
    match resolve_in(&loaded.index, spec, agents) {
        Err(RegistryError::UnknownVersion {
            id,
            version,
            listed,
            ..
        }) => {
            if let Some(path) = source.version_path(&id, &version)
                && let Some(entry) = std::fs::read_to_string(&path)
                    .ok()
                    .and_then(|t| serde_json::from_str::<Entry>(&t).ok())
            {
                return entry.to_manifest(agents).map_err(RegistryError::Refused);
            }
            if matches!(loaded.origin, Origin::File(_)) {
                return Err(RegistryError::UnknownVersion {
                    id,
                    version,
                    listed,
                    seen: Vec::new(),
                    history: Some(format!(
                        "{FILE_ENV} is set, so BenchFlow stays off the network"
                    )),
                });
            }
            match history(&id, &version, source).await {
                Ok((Some(entry), _)) => entry.to_manifest(agents).map_err(RegistryError::Refused),
                Ok((None, seen)) => Err(RegistryError::UnknownVersion {
                    id,
                    version,
                    listed,
                    seen,
                    history: None,
                }),
                Err(why) => Err(RegistryError::UnknownVersion {
                    id,
                    version,
                    listed,
                    seen: Vec::new(),
                    history: Some(why),
                }),
            }
        }
        other => other,
    }
}

/// Walk `<id>/agent.json`'s commits, newest first, until one lists
/// `version`. Every version seen is cached. Returns the entry, if found,
/// and the versions seen.
async fn history(
    id: &str,
    version: &str,
    source: &Source,
) -> Result<(Option<Entry>, Vec<String>), String> {
    let url = format!(
        "https://api.github.com/repos/{REPOSITORY}/commits?path={id}/agent.json&per_page=100"
    );
    let commits: Vec<serde_json::Value> = serde_json::from_str(&fetch_text(&url).await?)
        .map_err(|e| format!("GitHub's answer for {id}'s history is not a commit list: {e}"))?;
    let mut seen: Vec<String> = Vec::new();
    for commit in commits {
        let Some(sha) = commit.get("sha").and_then(|s| s.as_str()) else {
            continue;
        };
        let raw = format!("https://raw.githubusercontent.com/{REPOSITORY}/{sha}/{id}/agent.json");
        let Ok(text) = fetch_text(&raw).await else {
            continue;
        };
        let Ok(entry) = serde_json::from_str::<Entry>(&text) else {
            continue;
        };
        if !seen.contains(&entry.version) {
            if let Some(path) = source.version_path(id, &entry.version) {
                let _ = write_atomic(&path, &text);
            }
            seen.push(entry.version.clone());
        }
        if entry.version == version {
            return Ok((Some(entry), seen));
        }
    }
    Ok((None, seen))
}

fn read_cached(path: &Path) -> Option<(Index, Duration)> {
    let modified = std::fs::metadata(path).ok()?.modified().ok()?;
    let age = SystemTime::now()
        .duration_since(modified)
        .unwrap_or_default();
    let index = Index::parse(&std::fs::read_to_string(path).ok()?).ok()?;
    Some((index, age))
}

fn write_atomic(path: &Path, text: &str) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension(format!("tmp-{}", std::process::id()));
    std::fs::write(&tmp, text)?;
    std::fs::rename(&tmp, path)
}

async fn fetch_text(url: &str) -> Result<String, String> {
    let client = reqwest::Client::builder()
        .timeout(FETCH_TIMEOUT)
        .user_agent(concat!("benchflow/", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(|e| e.to_string())?;
    let response = client
        .get(url)
        .header("Accept", "application/json")
        .send()
        .await
        .map_err(|e| format!("{url}: {e}"))?;
    let status = response.status();
    if !status.is_success() {
        return Err(format!("{url} answered HTTP {}", status.as_u16()));
    }
    response.text().await.map_err(|e| format!("{url}: {e}"))
}

#[cfg(test)]
mod tests;
