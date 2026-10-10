//! The task config of the split layout (Harbor's `task.toml`, schema 1.0 to
//! 1.4) and of BenchFlow 0.8, which took its model from Harbor.
//!
//! [`read`] validates a parsed config the way the two loaders do and returns
//! the settings the runtime needs. Where Harbor and 0.8 disagree, it accepts
//! what either accepts and says what the other would do.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::LazyLock;

use regex::Regex;
use serde_json::{Map, Value};

use super::hosts;
use super::keys::is_env_shorthand;
use super::lax;
use super::schema::Dialect;
use crate::diag::{Diagnostic, Severity};
use crate::source::join;

static ORG_NAME: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*/[a-zA-Z0-9][a-zA-Z0-9._-]*$").unwrap()
});
static TOPOLOGY: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^[1-9]\d*(x[1-9]\d*)+$").unwrap());
static SERVICE_NAME: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$").unwrap());

/// Finds where a config path is in its file.
pub(crate) trait Locate {
    /// The 1-based line and byte span of the key at `path` (or of its value
    /// when `value` is set), falling back to the nearest parent.
    fn locate(&self, path: &str, value: bool) -> (Option<usize>, Option<(usize, usize)>);
}

/// The codes a reader emits; each format has its own.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Codes {
    /// A value of the wrong type or outside its rules (an error).
    pub invalid: &'static str,
    /// A setting that has no effect or that one of the two loaders would
    /// read differently (a warning).
    pub ignored: &'static str,
    /// A deprecated spelling that is still read (a note).
    pub deprecated: &'static str,
    /// A BenchFlow 0.8 spelling that Harbor ignores (a warning in task.toml).
    pub benchflow_only: &'static str,
}

/// A network mode, as `network_mode` spells it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NetworkMode {
    NoNetwork,
    Public,
    Allowlist,
    Denylist,
}

impl NetworkMode {
    fn parse(text: &str) -> Option<Self> {
        Some(match text {
            "no-network" => Self::NoNetwork,
            "public" => Self::Public,
            "allowlist" => Self::Allowlist,
            "denylist" => Self::Denylist,
            _ => return None,
        })
    }
}

/// A role's network override (`[agent]`, `[verifier]`).
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct PhaseNetwork {
    pub mode: Option<NetworkMode>,
    pub allowed_hosts: Vec<String>,
}

/// The sandbox's network for its whole life.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SandboxNetwork {
    pub mode: NetworkMode,
    pub allowed_hosts: Vec<String>,
    pub blocked_hosts: Vec<String>,
    pub blocked_urls: Vec<String>,
}

impl Default for SandboxNetwork {
    fn default() -> Self {
        Self {
            mode: NetworkMode::Public,
            allowed_hosts: Vec::new(),
            blocked_hosts: Vec::new(),
            blocked_urls: Vec::new(),
        }
    }
}

/// `[task]`: the package's identity.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct Package {
    pub name: Option<String>,
    pub version: Option<String>,
    pub description: Option<String>,
}

/// `[agent]`.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct Agent {
    pub timeout_sec: Option<f64>,
    pub user: Option<String>,
    pub network: PhaseNetwork,
    pub prompt_prefix: Option<String>,
}

/// `[environment] healthcheck`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Healthcheck {
    pub command: String,
    pub interval_sec: f64,
    pub timeout_sec: f64,
    pub start_period_sec: f64,
    pub start_interval_sec: f64,
    pub retries: i64,
}

/// `[environment]` (Harbor) or `[sandbox]` (BenchFlow 0.8).
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct Sandbox {
    pub network: SandboxNetwork,
    pub build_timeout_sec: Option<f64>,
    pub docker_image: Option<String>,
    pub os: Option<String>,
    pub cpus: Option<i64>,
    pub memory_mb: Option<i64>,
    pub storage_mb: Option<i64>,
    pub gpus: Option<i64>,
    pub gpu_types: Option<Vec<String>>,
    pub tpu: bool,
    pub mcp_servers: Vec<Value>,
    pub env: BTreeMap<String, String>,
    pub skills_dir: Option<String>,
    pub healthcheck: Option<Healthcheck>,
    pub workdir: Option<String>,
    pub setup_commands: usize,
    pub bugswarm_image_tag: Option<String>,
}

/// `[verifier]`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Verifier {
    pub timeout_sec: f64,
    pub env: BTreeMap<String, String>,
    pub user: Option<String>,
    pub network: PhaseNetwork,
    /// `environment_mode` (Harbor) or `sandbox_mode` (0.8), as written.
    pub mode: Option<String>,
    /// The key of a separate verifier environment table, when one is set
    /// (`environment` or `sandbox`).
    pub separate_environment: Option<&'static str>,
    pub collect: usize,
    pub kind: String,
    pub submission_files: Vec<String>,
    pub workspace_recovery: bool,
    pub reward_range: Option<(f64, f64)>,
    pub service: String,
    pub cleanup_conftests: bool,
    pub pytest_plugins: Vec<String>,
    pub expected_skills: Option<Vec<String>>,
}

impl Default for Verifier {
    fn default() -> Self {
        Self {
            timeout_sec: 600.0,
            env: BTreeMap::new(),
            user: None,
            network: PhaseNetwork::default(),
            mode: None,
            separate_environment: None,
            collect: 0,
            kind: "test-script".into(),
            submission_files: Vec::new(),
            workspace_recovery: false,
            reward_range: None,
            service: "main".into(),
            cleanup_conftests: true,
            pytest_plugins: Vec::new(),
            expected_skills: None,
        }
    }
}

/// `[solution]` (also `[oracle]` in 0.8).
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct Oracle {
    pub env: BTreeMap<String, String>,
    pub timeout_sec: Option<f64>,
}

/// One `artifacts` entry.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Artifact {
    pub source: String,
    pub destination: Option<String>,
    pub exclude: Vec<String>,
    pub service: Option<String>,
}

/// A task config, read and checked.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct Config {
    pub schema_version: Option<String>,
    pub package: Package,
    pub agent: Agent,
    pub sandbox: Sandbox,
    /// The key the sandbox table was read from (`environment` or `sandbox`).
    pub sandbox_key: Option<&'static str>,
    pub verifier: Verifier,
    pub oracle: Oracle,
    pub source: Option<String>,
    pub multi_step_reward_strategy: Option<String>,
    /// The number of `[[steps]]` (Harbor's multi-step tasks).
    pub steps: usize,
    pub artifacts: Vec<Artifact>,
}

/// Reads one config, collecting diagnostics.
pub(crate) struct Reader<'a> {
    pub dialect: Dialect,
    pub file: &'a Path,
    pub loc: &'a dyn Locate,
    pub codes: Codes,
    pub diags: Vec<Diagnostic>,
}

type Conv<T> = fn(&Value) -> Result<T, String>;

/// Builder steps for a diagnostic already pushed.
trait Amend {
    fn and_help(&mut self, help: impl Into<String>) -> &mut Self;
    fn and_note(&mut self, note: impl Into<String>) -> &mut Self;
}

impl Amend for Diagnostic {
    fn and_help(&mut self, help: impl Into<String>) -> &mut Self {
        self.help = Some(help.into());
        self
    }

    fn and_note(&mut self, note: impl Into<String>) -> &mut Self {
        self.notes.push(note.into());
        self
    }
}

impl Reader<'_> {
    fn push(
        &mut self,
        code: &'static str,
        severity: Severity,
        path: &str,
        message: String,
        value: bool,
    ) -> &mut Diagnostic {
        let (line, span) = self.loc.locate(path, value);
        self.diags
            .push(Diagnostic::new(code, severity, message).at(self.file, line, span));
        self.diags.last_mut().expect("just pushed")
    }

    /// An error at a value.
    fn invalid(&mut self, path: &str, message: String) -> &mut Diagnostic {
        self.push(self.codes.invalid, Severity::Error, path, message, true)
    }

    /// A warning at a key.
    fn ignored(&mut self, path: &str, message: String) -> &mut Diagnostic {
        self.push(self.codes.ignored, Severity::Warning, path, message, false)
    }

    /// Read `key` of `map` with `conv`; a wrong value is an error and `None`.
    fn get<T>(
        &mut self,
        map: &Map<String, Value>,
        prefix: &str,
        key: &str,
        conv: Conv<T>,
    ) -> Option<T> {
        let v = map.get(key)?;
        match conv(v) {
            Ok(x) => Some(x),
            Err(e) => {
                let where_ = where_(prefix);
                self.invalid(&join(prefix, key), format!("invalid `{key}`{where_}: {e}"));
                None
            }
        }
    }

    /// A positive, finite number of seconds.
    fn seconds(&mut self, map: &Map<String, Value>, prefix: &str, key: &str) -> Option<f64> {
        let v = self.get(map, prefix, key, lax::float)?;
        if v.is_finite() && v > 0.0 {
            Some(v)
        } else {
            let where_ = where_(prefix);
            self.invalid(
                &join(prefix, key),
                format!("invalid `{key}`{where_}: a time limit must be a positive number of seconds, not {v}"),
            );
            None
        }
    }

    /// A table under `key`; anything else is an error and `None`.
    fn table<'v>(
        &mut self,
        map: &'v Map<String, Value>,
        prefix: &str,
        key: &str,
    ) -> Option<&'v Map<String, Value>> {
        match map.get(key)? {
            Value::Object(m) => Some(m),
            other => {
                let msg = format!(
                    "`{key}`{} must be a table, not {}",
                    where_(prefix),
                    lax::describe(other)
                );
                self.invalid(&join(prefix, key), msg);
                None
            }
        }
    }
}

/// " in [verifier]" for messages, or nothing at the top level.
fn where_(prefix: &str) -> String {
    if prefix.is_empty() {
        String::new()
    } else {
        // Drop list indices: steps[0].agent -> steps.agent.
        let mut table = String::new();
        let mut depth = 0;
        for c in prefix.chars() {
            match c {
                '[' => depth += 1,
                ']' => depth -= 1,
                _ if depth == 0 => table.push(c),
                _ => {}
            }
        }
        format!(" in [{}]", table.replace('"', ""))
    }
}

/// Read a whole config.
pub(crate) fn read(r: &mut Reader<'_>, config: &Value) -> Config {
    let mut out = Config::default();
    let Value::Object(top) = config else {
        r.invalid("", "the config must be a table".into());
        return out;
    };
    let schema_key = if top.contains_key("schema_version") {
        "schema_version"
    } else {
        "version"
    };
    out.schema_version = r.get(top, "", schema_key, lax::string);
    if let Some(v) = out.schema_version.clone() {
        check_schema_version(r, schema_key, &v);
    }
    if let Some(task) = r.table(top, "", "task") {
        out.package = read_package(r, task);
    }
    if let Some(agent) = r.table(top, "", "agent") {
        out.agent = read_agent(r, agent, "agent");
    }
    let (sandbox_key, _) = pick_spelling(r, top, "", "environment", "sandbox");
    out.sandbox_key = sandbox_key;
    if let Some(key) = sandbox_key
        && let Some(t) = r.table(top, "", key)
    {
        out.sandbox = read_sandbox(r, t, key);
    }
    if r.dialect == Dialect::Frontmatter {
        apply_benchflow_defaults(&mut out.sandbox);
    }
    if let Some(v) = r.table(top, "", "verifier") {
        out.verifier = read_verifier(r, v, "verifier");
    }
    let (oracle_key, _) = pick_spelling(r, top, "", "solution", "oracle");
    if let Some(key) = oracle_key
        && let Some(t) = r.table(top, "", key)
    {
        out.oracle = read_oracle(r, t, key);
    }
    out.source = r.get(top, "", "source", lax::string);
    out.multi_step_reward_strategy = r.get(top, "", "multi_step_reward_strategy", |v| {
        lax::choice(v, &["mean", "final"])
    });
    match top.get("steps") {
        Some(Value::Array(steps)) => {
            out.steps = steps.len();
            for (i, step) in steps.iter().enumerate() {
                read_step(r, step, &format!("steps[{i}]"));
            }
        }
        Some(other) => {
            let msg = format!(
                "`steps` must be a list of tables, not {}",
                lax::describe(other)
            );
            r.invalid("steps", msg);
        }
        None => {}
    }
    if out.steps == 0 && out.multi_step_reward_strategy.is_some() {
        r.ignored(
            "multi_step_reward_strategy",
            "`multi_step_reward_strategy` has no effect without [[steps]]".into(),
        );
    }
    if let Some(items) = top.get("artifacts") {
        out.artifacts = read_artifacts(r, items, "artifacts");
    }
    if let Some(reward) = top.get("reward")
        && !reward.is_object()
    {
        let msg = format!("`reward` must be a table, not {}", lax::describe(reward));
        r.invalid("reward", msg);
    }
    out
}

/// Decide which of two spellings of one table to read. In `task.toml` the
/// first (Harbor's) wins and the second (0.8's) is read only alone; in a 0.8
/// frontmatter only the second exists, and the first is an error.
fn pick_spelling(
    r: &mut Reader<'_>,
    map: &Map<String, Value>,
    prefix: &str,
    harbor: &'static str,
    benchflow: &'static str,
) -> (Option<&'static str>, Option<&'static str>) {
    let has_h = map.contains_key(harbor);
    let has_b = map.contains_key(benchflow);
    match r.dialect {
        Dialect::TaskToml => {
            if has_h && has_b {
                let msg = format!(
                    "`{benchflow}`{} is ignored because `{harbor}` is set: they are the same table, and Harbor reads only `{harbor}`",
                    where_(prefix)
                );
                r.ignored(&join(prefix, benchflow), msg).and_help(format!(
                    "merge the two and keep `{harbor}`; BenchFlow 0.8 refuses a config with both"
                ));
                (Some(harbor), Some(benchflow))
            } else if has_b {
                let path = join(prefix, benchflow);
                let msg = format!(
                    "`{benchflow}`{} is BenchFlow 0.8's spelling of `{harbor}`; Harbor ignores it",
                    where_(prefix)
                );
                let code = r.codes.benchflow_only;
                r.push(code, Severity::Warning, &path, msg, false)
                    .and_help(format!("rename it to `{harbor}` so every harness reads it"));
                (Some(benchflow), None)
            } else if has_h {
                (Some(harbor), None)
            } else {
                (None, None)
            }
        }
        Dialect::Frontmatter => {
            if harbor == "solution" {
                // 0.8 accepts `solution` as an alias of `oracle`, but not both.
                if has_h && has_b {
                    r.invalid(
                        &join(prefix, harbor),
                        "the config has both `oracle` and its old spelling `solution`; keep only `oracle`".into(),
                    );
                    return (Some(benchflow), None);
                }
                return (
                    if has_b {
                        Some(benchflow)
                    } else if has_h {
                        Some(harbor)
                    } else {
                        None
                    },
                    None,
                );
            }
            (has_b.then_some(benchflow), None)
        }
    }
}

fn check_schema_version(r: &mut Reader<'_>, key: &str, v: &str) {
    let major = v.split('.').next().unwrap_or("");
    if major == "1" || major.parse::<u32>() == Ok(1) {
        return;
    }
    let message = format!(
        "schema_version {v:?} is not a 1.x version; BenchFlow 0.8 reads only major version 1"
    );
    match r.dialect {
        Dialect::TaskToml => {
            r.ignored(key, message);
        }
        Dialect::Frontmatter => {
            r.invalid(key, message);
        }
    }
}

fn read_package(r: &mut Reader<'_>, t: &Map<String, Value>) -> Package {
    let name = match t.get("name") {
        None => {
            r.invalid(
                "task",
                "[task] needs a `name`, such as \"org/task-name\"".into(),
            );
            None
        }
        Some(_) => r.get(t, "task", "name", lax::string),
    };
    if let Some(n) = &name
        && (!ORG_NAME.is_match(n) || n.contains(".."))
    {
        r.invalid(
                "task.name",
                format!("invalid `name` in [task]: {n:?} is not \"org/name\" (letters, digits, '.', '_', '-', starting with a letter or digit, no '..')"),
            );
    }
    let version = r.get(t, "task", "version", lax::string);
    let description = r.get(t, "task", "description", lax::string);
    match t.get("authors") {
        Some(Value::Array(authors)) => {
            for (i, a) in authors.iter().enumerate() {
                let path = format!("task.authors[{i}]");
                match a {
                    Value::Object(m) => {
                        if !m.contains_key("name") {
                            r.invalid(&path, "an author needs a `name`".into());
                        }
                        r.get(m, &path, "name", lax::string);
                        r.get(m, &path, "email", lax::string);
                    }
                    other => {
                        let msg = format!(
                            "an author must be a table with `name` and `email`, not {}",
                            lax::describe(other)
                        );
                        r.invalid(&path, msg);
                    }
                }
            }
        }
        Some(other) => {
            let msg = format!(
                "`authors` in [task] must be a list of tables, not {}",
                lax::describe(other)
            );
            r.invalid("task.authors", msg);
        }
        None => {}
    }
    r.get(t, "task", "keywords", lax::strings);
    Package {
        name,
        version: version.filter(|v| !v.is_empty()),
        description,
    }
}

fn read_phase_network(
    r: &mut Reader<'_>,
    t: &Map<String, Value>,
    prefix: &str,
    role: &str,
) -> PhaseNetwork {
    let mode = r
        .get(t, prefix, "network_mode", |v| {
            lax::choice(v, &["no-network", "public", "allowlist", "denylist"])
        })
        .and_then(|m| NetworkMode::parse(&m));
    if mode == Some(NetworkMode::Denylist) {
        r.invalid(
            &join(prefix, "network_mode"),
            format!("{role}.network_mode = \"denylist\" is not supported; declare the denylist on the sandbox's network_mode"),
        );
    }
    let hosts = read_allowed_hosts(r, t, prefix);
    check_allowlist(r, prefix, mode, hosts.as_deref());
    PhaseNetwork {
        mode: mode.filter(|m| *m != NetworkMode::Denylist),
        allowed_hosts: hosts.unwrap_or_default(),
    }
}

fn read_allowed_hosts(
    r: &mut Reader<'_>,
    t: &Map<String, Value>,
    prefix: &str,
) -> Option<Vec<String>> {
    let hosts = r.get(t, prefix, "allowed_hosts", lax::strings)?;
    match hosts::allowed_hosts(&hosts) {
        Ok(h) => Some(h),
        Err(e) => {
            r.invalid(
                &join(prefix, "allowed_hosts"),
                format!("invalid `allowed_hosts`{}: {e}", where_(prefix)),
            );
            None
        }
    }
}

/// `allowed_hosts` only with `allowlist` (Harbor and 0.8), and a non-empty
/// list with it (0.8; Harbor allows an empty one, which reaches nothing).
fn check_allowlist(
    r: &mut Reader<'_>,
    prefix: &str,
    mode: Option<NetworkMode>,
    hosts: Option<&[String]>,
) {
    let has_hosts = hosts.is_some_and(|h| !h.is_empty());
    if mode != Some(NetworkMode::Allowlist)
        && hosts.is_some()
        && (has_hosts || r.dialect == Dialect::TaskToml)
    {
        r.invalid(
            &join(prefix, "allowed_hosts"),
            format!(
                "`allowed_hosts`{} is only valid with network_mode = \"allowlist\"",
                where_(prefix)
            ),
        );
    }
    if mode == Some(NetworkMode::Allowlist) && !has_hosts {
        let path = join(prefix, "network_mode");
        let msg = format!(
            "network_mode = \"allowlist\"{} needs a non-empty `allowed_hosts`",
            where_(prefix)
        );
        match r.dialect {
            Dialect::Frontmatter => {
                r.invalid(&path, msg);
            }
            Dialect::TaskToml => {
                r.ignored(&path, msg)
                    .and_note("Harbor runs it with no host reachable; BenchFlow 0.8 refuses it");
            }
        }
    }
}

fn read_agent(r: &mut Reader<'_>, t: &Map<String, Value>, prefix: &str) -> Agent {
    let timeout_sec = r.seconds(t, prefix, "timeout_sec");
    let user = r.get(t, prefix, "user", lax::user);
    let network = read_phase_network(r, t, prefix, "agent");
    let prompt_prefix = r
        .get(t, prefix, "prompt_prefix", lax::string)
        .and_then(|p| {
            let trimmed = p.trim().to_string();
            if trimmed.is_empty() {
                r.invalid(
                    &join(prefix, "prompt_prefix"),
                    "`prompt_prefix` must contain text other than whitespace".into(),
                );
                None
            } else {
                Some(trimmed)
            }
        });
    Agent {
        timeout_sec,
        user,
        network,
        prompt_prefix,
    }
}

/// Harbor's parser for its deprecated `memory = "4G"` spelling.
pub(crate) fn harbor_size_mb(text: &str) -> Option<i64> {
    let s = text.trim().to_uppercase();
    let (num, mult): (&str, f64) = if let Some(n) = s.strip_suffix('G') {
        (n, 1024.0)
    } else if let Some(n) = s.strip_suffix('M') {
        (n, 1.0)
    } else {
        let n = s.strip_suffix('K')?;
        (n, 1.0 / 1024.0)
    };
    let f: f64 = num.trim().parse().ok()?;
    let mb = f * mult;
    mb.is_finite().then_some(mb.trunc() as i64)
}

fn read_sandbox(r: &mut Reader<'_>, t: &Map<String, Value>, prefix: &str) -> Sandbox {
    let mut s = Sandbox {
        build_timeout_sec: r.seconds(t, prefix, "build_timeout_sec"),
        ..Sandbox::default()
    };
    let image_key = if t.contains_key("docker_image") || !t.contains_key("image") {
        "docker_image"
    } else {
        "image"
    };
    s.docker_image = r.get(t, prefix, image_key, lax::string);
    if image_key == "image" && r.dialect == Dialect::TaskToml {
        let path = join(prefix, "image");
        let msg = format!(
            "`image`{} is BenchFlow 0.8's spelling of `docker_image`; Harbor ignores it",
            where_(prefix)
        );
        let code = r.codes.benchflow_only;
        r.push(code, Severity::Warning, &path, msg, false)
            .and_help("rename it to `docker_image`");
    }
    s.os = r.get(t, prefix, "os", |v| {
        let text = lax::string(v)?.to_lowercase();
        lax::choice(&Value::String(text), &["linux", "windows"])
    });
    s.cpus = r.get(t, prefix, "cpus", lax::int);
    s.memory_mb = r.get(t, prefix, "memory_mb", lax::int);
    s.storage_mb = r.get(t, prefix, "storage_mb", lax::int);
    for (old, new) in [("memory", "memory_mb"), ("storage", "storage_mb")] {
        read_size_spelling(r, t, prefix, old, new, &mut s);
    }
    s.gpus = r.get(t, prefix, "gpus", lax::int);
    s.gpu_types = r.get(t, prefix, "gpu_types", lax::strings);
    if let Some(tpu) = r.table(t, prefix, "tpu") {
        s.tpu = true;
        let tp = join(prefix, "tpu");
        match tpu.get("type") {
            None => {
                r.invalid(&tp, "[tpu] needs a `type`, such as \"v6e\"".into());
            }
            Some(_) => {
                if r.get(tpu, &tp, "type", lax::string)
                    .is_some_and(|x| x.is_empty())
                {
                    r.invalid(
                        &join(&tp, "type"),
                        "`type` in [tpu] must not be empty".into(),
                    );
                }
            }
        }
        match r.get(tpu, &tp, "topology", lax::string) {
            Some(topo) if !TOPOLOGY.is_match(topo.trim()) => {
                r.invalid(
                    &join(&tp, "topology"),
                    format!("invalid `topology` in [tpu]: {topo:?} is not dimensions joined by 'x', such as \"2x4\""),
                );
            }
            None if !tpu.contains_key("topology") => {
                r.invalid(&tp, "[tpu] needs a `topology`, such as \"2x4\"".into());
            }
            _ => {}
        }
    }
    if let Some(servers) = t.get("mcp_servers") {
        s.mcp_servers = read_mcp_servers(r, servers, &join(prefix, "mcp_servers"));
    }
    s.env = r.get(t, prefix, "env", lax::string_map).unwrap_or_default();
    s.skills_dir = r.get(t, prefix, "skills_dir", lax::string);
    if let Some(hc) = r.table(t, prefix, "healthcheck") {
        s.healthcheck = read_healthcheck(r, hc, &join(prefix, "healthcheck"));
    }
    s.workdir = r.get(t, prefix, "workdir", lax::string);
    s.network = read_sandbox_network(r, t, prefix);
    match t.get("setup_commands") {
        Some(Value::Array(cmds)) => {
            s.setup_commands = cmds.len();
            for (i, c) in cmds.iter().enumerate() {
                let path = format!("{}[{i}]", join(prefix, "setup_commands"));
                match c {
                    Value::Object(m) => {
                        let cmd = r.get(m, &path, "command", lax::string);
                        if cmd.is_none_or(|c| c.trim().is_empty())
                            && !m.get("command").is_some_and(|v| !v.is_string())
                        {
                            r.invalid(&path, "a setup command needs a non-empty `command`".into());
                        }
                    }
                    other => {
                        let msg = format!(
                            "a setup command must be a table, not {}",
                            lax::describe(other)
                        );
                        r.invalid(&path, msg);
                    }
                }
            }
        }
        Some(other) => {
            let msg = format!(
                "`setup_commands` must be a list of tables, not {}",
                lax::describe(other)
            );
            r.invalid(&join(prefix, "setup_commands"), msg);
        }
        None => {}
    }
    s.bugswarm_image_tag = r.get(t, prefix, "bugswarm_image_tag", lax::string);
    s
}

/// Harbor's deprecated `memory = "4G"` and `storage = "10G"`.
fn read_size_spelling(
    r: &mut Reader<'_>,
    t: &Map<String, Value>,
    prefix: &str,
    old: &str,
    new: &str,
    s: &mut Sandbox,
) {
    let Some(v) = t.get(old) else { return };
    let path = join(prefix, old);
    let Value::String(text) = v else {
        match r.dialect {
            Dialect::TaskToml => {
                let msg = format!(
                    "`{old}`{} is ignored: Harbor reads it only as a size string such as \"4G\"",
                    where_(prefix)
                );
                r.ignored(&path, msg)
                    .and_help(format!("write `{new}` as a number of megabytes"));
            }
            Dialect::Frontmatter => {
                let msg = format!(
                    "invalid `{old}`{}: expected a size string such as \"4G\"",
                    where_(prefix)
                );
                r.invalid(&path, msg);
            }
        }
        return;
    };
    let Some(mb) = harbor_size_mb(text) else {
        r.invalid(
            &path,
            format!(
                "invalid `{old}`{}: {text:?} is not a size such as \"4G\" or \"512M\"",
                where_(prefix)
            ),
        );
        return;
    };
    let current = if new == "memory_mb" {
        s.memory_mb
    } else {
        s.storage_mb
    };
    if let Some(cur) = current
        && cur != mb
    {
        let msg = format!(
            "`{old} = {text:?}` ({mb} MB) disagrees with `{new} = {cur}`{}; `{old}` is used",
            where_(prefix)
        );
        r.ignored(&path, msg)
            .and_note("Harbor refuses this config; BenchFlow 0.8 uses the old spelling")
            .and_help(format!("keep only `{new}`"));
    }
    let code = r.codes.deprecated;
    let msg = format!(
        "`{old}`{} is a deprecated spelling; it is read as {new} = {mb}",
        where_(prefix)
    );
    let fix_text = format!("{new} = {mb}");
    let span = combined_span(r.loc, &path);
    let file = r.file.to_path_buf();
    let d = r.push(code, Severity::Note, &path, msg, false);
    if current.is_none()
        && let Some((start, end)) = span
    {
        d.fix = Some(crate::diag::Fix {
            file,
            span: (start, end),
            replacement: fix_text.clone(),
            description: format!("replace `{old} = {text:?}` with `{fix_text}`"),
        });
    }
    if new == "memory_mb" {
        s.memory_mb = Some(mb);
    } else {
        s.storage_mb = Some(mb);
    }
}

/// The span from a key to the end of its value, for fixes that rewrite a
/// whole `key = value` line. Only when both are known and the key comes
/// first.
fn combined_span(loc: &dyn Locate, path: &str) -> Option<(usize, usize)> {
    let (_, key) = loc.locate(path, false);
    let (_, value) = loc.locate(path, true);
    match (key, value) {
        (Some((ks, _)), Some((_, ve))) if ks < ve => Some((ks, ve)),
        _ => None,
    }
}

fn read_sandbox_network(
    r: &mut Reader<'_>,
    t: &Map<String, Value>,
    prefix: &str,
) -> SandboxNetwork {
    let explicit = r
        .get(t, prefix, "network_mode", |v| {
            lax::choice(v, &["no-network", "public", "allowlist", "denylist"])
        })
        .and_then(|m| NetworkMode::parse(&m));
    let allowed = read_allowed_hosts(r, t, prefix);
    let blocked_hosts = r
        .get(t, prefix, "blocked_hosts", lax::strings)
        .and_then(|h| {
            hosts::blocked_hosts(&h)
                .map_err(|e| {
                    r.invalid(
                        &join(prefix, "blocked_hosts"),
                        format!("invalid `blocked_hosts`{}: {e}", where_(prefix)),
                    );
                })
                .ok()
        });
    let blocked_urls = r
        .get(t, prefix, "blocked_urls", lax::strings)
        .and_then(|u| {
            hosts::blocked_urls(&u)
                .map_err(|e| {
                    r.invalid(
                        &join(prefix, "blocked_urls"),
                        format!("invalid `blocked_urls`{}: {e}", where_(prefix)),
                    );
                })
                .ok()
        });
    let allow_internet = r.get(t, prefix, "allow_internet", lax::boolean);
    let mut mode = explicit.unwrap_or(NetworkMode::Public);
    if let Some(allow) = allow_internet {
        let path = join(prefix, "allow_internet");
        if explicit.is_some() || allowed.is_some() {
            let contradicts = !allow && explicit.is_some_and(|m| m != NetworkMode::NoNetwork);
            let msg = format!(
                "`allow_internet`{} is ignored because `network_mode` or `allowed_hosts` is set",
                where_(prefix)
            );
            let d = if contradicts && r.dialect == Dialect::Frontmatter {
                r.invalid(
                    &path,
                    format!(
                        "allow_internet = false contradicts the explicit network_mode{}",
                        where_(prefix)
                    ),
                )
            } else {
                r.ignored(&path, msg)
            };
            if contradicts {
                d.notes.push("BenchFlow 0.8 refuses allow_internet = false with any network_mode but \"no-network\"".into());
            }
            d.help = Some("remove `allow_internet`".into());
        } else {
            mode = if allow {
                NetworkMode::Public
            } else {
                NetworkMode::NoNetwork
            };
            let mode_name = if allow { "public" } else { "no-network" };
            let replacement = if r.dialect == Dialect::Frontmatter {
                format!("network_mode: {mode_name}")
            } else {
                format!("network_mode = \"{mode_name}\"")
            };
            let span = combined_span(r.loc, &path);
            let file = r.file.to_path_buf();
            let code = r.codes.deprecated;
            let d = r.push(
                code,
                Severity::Note,
                &path,
                format!(
                    "`allow_internet`{} is a deprecated spelling; it is read as {replacement}",
                    where_(prefix)
                ),
                false,
            );
            d.help = Some(format!("write `{replacement}` instead"));
            if let Some(span) = span {
                d.fix = Some(crate::diag::Fix {
                    file,
                    span,
                    replacement: replacement.clone(),
                    description: format!("replace `allow_internet = {allow}` with `{replacement}`"),
                });
            }
        }
    }
    if explicit.is_none() && allowed.is_some() {
        r.invalid(
            &join(prefix, "allowed_hosts"),
            format!(
                "`allowed_hosts`{} is only valid with network_mode = \"allowlist\"",
                where_(prefix)
            ),
        );
    } else if explicit.is_some() {
        check_allowlist(r, prefix, explicit, allowed.as_deref());
    }
    let has_blocked = blocked_hosts.as_ref().is_some_and(|h| !h.is_empty())
        || blocked_urls.as_ref().is_some_and(|u| !u.is_empty());
    if mode == NetworkMode::Denylist && !has_blocked {
        r.invalid(
            &join(prefix, "network_mode"),
            "network_mode = \"denylist\" needs `blocked_urls` or `blocked_hosts`".into(),
        );
    }
    if mode != NetworkMode::Denylist && has_blocked {
        let at = if blocked_urls.is_some() {
            "blocked_urls"
        } else {
            "blocked_hosts"
        };
        r.invalid(
            &join(prefix, at),
            "`blocked_urls` and `blocked_hosts` are only valid with network_mode = \"denylist\""
                .into(),
        );
    }
    SandboxNetwork {
        mode,
        allowed_hosts: allowed.unwrap_or_default(),
        blocked_hosts: blocked_hosts.unwrap_or_default(),
        blocked_urls: blocked_urls.unwrap_or_default(),
    }
}

fn read_healthcheck(
    r: &mut Reader<'_>,
    t: &Map<String, Value>,
    prefix: &str,
) -> Option<Healthcheck> {
    let command = match t.get("command") {
        None => {
            r.invalid(prefix, "a healthcheck needs a `command`".into());
            None
        }
        Some(_) => r.get(t, prefix, "command", lax::string),
    };
    let mut secs = |key: &str, default: f64| -> f64 {
        match r.get(t, prefix, key, lax::float) {
            Some(v) if v.is_finite() && v >= 0.0 => v,
            Some(v) => {
                r.invalid(
                    &join(prefix, key),
                    format!(
                        "invalid `{key}`{}: {v} is not a number of seconds",
                        where_(prefix)
                    ),
                );
                default
            }
            None => default,
        }
    };
    let interval_sec = secs("interval_sec", 5.0);
    let timeout_sec = secs("timeout_sec", 30.0);
    let start_period_sec = secs("start_period_sec", 0.0);
    let start_interval_sec = secs("start_interval_sec", 5.0);
    let retries = r.get(t, prefix, "retries", lax::int).unwrap_or(3);
    Some(Healthcheck {
        command: command?,
        interval_sec,
        timeout_sec,
        start_period_sec,
        start_interval_sec,
        retries,
    })
}

fn read_mcp_servers(r: &mut Reader<'_>, v: &Value, prefix: &str) -> Vec<Value> {
    let Value::Array(items) = v else {
        let msg = format!(
            "`mcp_servers` must be a list of tables, not {}",
            lax::describe(v)
        );
        r.invalid(prefix, msg);
        return Vec::new();
    };
    let mut out = Vec::new();
    for (i, item) in items.iter().enumerate() {
        let path = format!("{prefix}[{i}]");
        let Value::Object(m) = item else {
            let msg = format!("an MCP server must be a table, not {}", lax::describe(item));
            r.invalid(&path, msg);
            continue;
        };
        if !m.contains_key("name") {
            r.invalid(&path, "an MCP server needs a `name`".into());
        }
        r.get(m, &path, "name", lax::string);
        let transport = r
            .get(m, &path, "transport", |v| {
                let t = lax::string(v)?;
                let t = if t == "http" {
                    "streamable-http".to_string()
                } else {
                    t
                };
                lax::choice(&Value::String(t), &["stdio", "sse", "streamable-http"])
            })
            .unwrap_or_else(|| "sse".into());
        let url = r.get(m, &path, "url", lax::string);
        let command = r.get(m, &path, "command", lax::string);
        r.get(m, &path, "args", lax::strings);
        r.get(m, &path, "cwd", lax::string);
        r.get(m, &path, "env", lax::string_map);
        r.get(m, &path, "headers", lax::string_map);
        for key in ["tools", "include_tags", "exclude_tags"] {
            if let Some(list) = r.get(m, &path, key, lax::strings)
                && list.iter().all(|x| x.trim().is_empty())
            {
                r.invalid(
                    &join(&path, key),
                    format!("`{key}` must hold at least one name"),
                );
            }
        }
        if (transport == "sse" || transport == "streamable-http")
            && url.as_deref().is_none_or(str::is_empty)
        {
            r.invalid(
                &path,
                format!("an MCP server with transport \"{transport}\" needs a `url`"),
            );
        }
        if transport == "stdio" && command.as_deref().is_none_or(str::is_empty) {
            r.invalid(
                &path,
                "an MCP server with transport \"stdio\" needs a `command`".into(),
            );
        }
        out.push(item.clone());
    }
    out
}

fn read_verifier(r: &mut Reader<'_>, t: &Map<String, Value>, prefix: &str) -> Verifier {
    let mut v = Verifier::default();
    if let Some(s) = r.seconds(t, prefix, "timeout_sec") {
        v.timeout_sec = s;
    }
    v.env = r.get(t, prefix, "env", lax::string_map).unwrap_or_default();
    v.user = r.get(t, prefix, "user", lax::user);
    v.network = read_phase_network(r, t, prefix, "verifier");
    let (mode_key, _) = pick_spelling(r, t, prefix, "environment_mode", "sandbox_mode");
    if let Some(key) = mode_key {
        v.mode = r.get(t, prefix, key, |x| lax::choice(x, &["shared", "separate"]));
    }
    let (env_key, _) = pick_spelling(r, t, prefix, "environment", "sandbox");
    if let Some(key) = env_key
        && let Some(env) = r.table(t, prefix, key)
    {
        v.separate_environment = Some(key);
        let _ = read_sandbox(r, env, &join(prefix, key));
        if v.mode.as_deref() == Some("shared") {
            r.invalid(
                    &join(prefix, key),
                    format!("a verifier with environment_mode = \"shared\" cannot have its own [{prefix}.{key}]; set it to \"separate\" or remove the table"),
                );
        }
    }
    match t.get("collect") {
        Some(Value::Array(items)) => {
            v.collect = items.len();
            for (i, item) in items.iter().enumerate() {
                let path = format!("{}[{i}]", join(prefix, "collect"));
                match item {
                    Value::Object(m) => {
                        if !m.contains_key("command") {
                            r.invalid(&path, "a collect hook needs a `command`".into());
                        }
                        r.get(m, &path, "command", lax::string);
                        if let Some(svc) = r.get(m, &path, "service", lax::string)
                            && !SERVICE_NAME.is_match(svc.trim())
                        {
                            r.invalid(
                                &join(&path, "service"),
                                format!("invalid `service`: {svc:?} is not a Compose service name"),
                            );
                        }
                        r.get(m, &path, "timeout_sec", lax::float);
                        r.get(m, &path, "user", lax::user);
                    }
                    other => {
                        let msg = format!(
                            "a collect hook must be a table, not {}",
                            lax::describe(other)
                        );
                        r.invalid(&path, msg);
                    }
                }
            }
        }
        Some(other) => {
            let msg = format!(
                "`collect` must be a list of tables, not {}",
                lax::describe(other)
            );
            r.invalid(&join(prefix, "collect"), msg);
        }
        None => {}
    }
    if let Some(kind) = r.get(t, prefix, "type", |x| {
        lax::choice(x, &["test-script", "llm-judge"])
    }) {
        v.kind = kind;
    }
    v.submission_files = r
        .get(t, prefix, "submission_files", lax::strings)
        .unwrap_or_default();
    v.workspace_recovery = r
        .get(t, prefix, "workspace_recovery", lax::boolean)
        .unwrap_or(false);
    if v.workspace_recovery && !v.submission_files.is_empty() {
        r.invalid(
            &join(prefix, "workspace_recovery"),
            "choose either `workspace_recovery` or `submission_files`, not both".into(),
        );
    }
    v.reward_range = read_reward_range(r, t, prefix);
    if let Some(service) = r.get(t, prefix, "service", lax::string) {
        v.service = service;
    }
    if let Some(judge) = r.table(t, prefix, "judge") {
        let jp = join(prefix, "judge");
        for key in ["model", "rubric_path", "input_dir", "context"] {
            r.get(judge, &jp, key, lax::string);
        }
        r.get(judge, &jp, "input_type", |x| {
            lax::choice(x, &["deliverables"])
        });
    }
    if let Some(memory) = r.table(t, prefix, "memory") {
        let mp = join(prefix, "memory");
        if memory.get("expected_skills").is_some_and(|x| !x.is_null()) {
            v.expected_skills = r.get(memory, &mp, "expected_skills", lax::strings);
        }
    }
    if let Some(hardening) = r.table(t, prefix, "hardening") {
        let hp = join(prefix, "hardening");
        if let Some(c) = r.get(hardening, &hp, "cleanup_conftests", lax::boolean) {
            v.cleanup_conftests = c;
        }
    }
    v.pytest_plugins = r
        .get(t, prefix, "pytest_plugins", lax::strings)
        .unwrap_or_default();
    v
}

/// `reward_range = [lo, hi]` may only widen [0, 1] (BenchFlow 0.8).
fn read_reward_range(
    r: &mut Reader<'_>,
    t: &Map<String, Value>,
    prefix: &str,
) -> Option<(f64, f64)> {
    let v = t.get("reward_range")?;
    let path = join(prefix, "reward_range");
    let pair = match v {
        Value::Array(items) if items.len() == 2 => {
            match (lax::float(&items[0]), lax::float(&items[1])) {
                (Ok(lo), Ok(hi)) => Some((lo, hi)),
                _ => None,
            }
        }
        _ => None,
    };
    let Some((lo, hi)) = pair else {
        r.invalid(&path, "`reward_range` must be two numbers, [lo, hi]".into());
        return None;
    };
    if !(lo.is_finite() && hi.is_finite() && lo <= 0.0 && hi >= 1.0 && lo < hi) {
        r.invalid(
            &path,
            format!(
                "`reward_range` [{lo}, {hi}] may only widen [0, 1]: lo <= 0, hi >= 1, both finite"
            ),
        );
        return None;
    }
    Some((lo, hi))
}

fn read_oracle(r: &mut Reader<'_>, t: &Map<String, Value>, prefix: &str) -> Oracle {
    let mut env = r.get(t, prefix, "env", lax::string_map).unwrap_or_default();
    let timeout_sec = r.get(t, prefix, "timeout_sec", lax::float);
    for (key, value) in t {
        if key == "env" || key == "timeout_sec" || !is_env_shorthand(key, value) {
            continue;
        }
        let text = value.as_str().unwrap_or_default().to_string();
        let path = join(prefix, key);
        if env.get(key).is_some_and(|existing| existing != &text) {
            r.invalid(
                &path,
                format!("conflicting values for the oracle's environment variable {key}"),
            );
            continue;
        }
        if r.dialect == Dialect::TaskToml {
            let code = r.codes.benchflow_only;
            let msg = format!(
                "`{key}` in [{prefix}] is BenchFlow 0.8's shorthand for [{prefix}.env] {key}; Harbor ignores it"
            );
            r.push(code, Severity::Warning, &path, msg, false)
                .and_help(format!("move it under [{prefix}.env]"));
        }
        env.insert(key.clone(), text);
    }
    Oracle { env, timeout_sec }
}

fn read_step(r: &mut Reader<'_>, step: &Value, prefix: &str) {
    let Value::Object(m) = step else {
        let msg = format!("a step must be a table, not {}", lax::describe(step));
        r.invalid(prefix, msg);
        return;
    };
    match m.get("name") {
        None => {
            r.invalid(prefix, "a step needs a `name`".into());
        }
        Some(_) => {
            if let Some(name) = r.get(m, prefix, "name", lax::string)
                && !valid_step_name(&name)
            {
                r.invalid(
                    &join(prefix, "name"),
                    format!(
                        "invalid step name {name:?}: a step name must be a portable folder name"
                    ),
                );
            }
        }
    }
    if let Some(agent) = r.table(m, prefix, "agent") {
        read_agent(r, agent, &join(prefix, "agent"));
    }
    if let Some(verifier) = r.table(m, prefix, "verifier") {
        read_verifier(r, verifier, &join(prefix, "verifier"));
    }
    if let Some(hc) = r.table(m, prefix, "healthcheck") {
        read_healthcheck(r, hc, &join(prefix, "healthcheck"));
    }
    if let Some(items) = m.get("artifacts") {
        read_artifacts(r, items, &join(prefix, "artifacts"));
    }
}

/// Harbor's rule for step names: a portable folder name. Stage names in
/// task.md follow it too: each names a folder in the task and in the trial.
pub(crate) fn valid_step_name(name: &str) -> bool {
    const RESERVED: [&str; 6] = ["CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"];
    if name.trim().is_empty() || name == "." || name == ".." || name.ends_with(['.', ' ']) {
        return false;
    }
    if name
        .chars()
        .any(|c| "<>:\"/\\|?*".contains(c) || (c as u32) < 0x20 || c as u32 == 0x7f)
    {
        return false;
    }
    let stem = name
        .split('.')
        .next()
        .unwrap_or("")
        .trim_end_matches(' ')
        .to_uppercase();
    let device = RESERVED.contains(&stem.as_str())
        || ((stem.starts_with("COM") || stem.starts_with("LPT"))
            && stem.chars().count() == 4
            && stem
                .chars()
                .last()
                .is_some_and(|c| "123456789¹²³".contains(c)));
    !device && name.len() <= 255
}

fn read_artifacts(r: &mut Reader<'_>, v: &Value, prefix: &str) -> Vec<Artifact> {
    let Value::Array(items) = v else {
        let msg = format!("`artifacts` must be a list, not {}", lax::describe(v));
        r.invalid(prefix, msg);
        return Vec::new();
    };
    let mut out = Vec::new();
    for (i, item) in items.iter().enumerate() {
        let path = format!("{prefix}[{i}]");
        let artifact = match item {
            Value::String(s) => Artifact {
                source: s.clone(),
                destination: None,
                exclude: Vec::new(),
                service: None,
            },
            Value::Object(m) => {
                if !m.contains_key("source") {
                    r.invalid(&path, "an artifact needs a `source`".into());
                    continue;
                }
                let Some(source) = r.get(m, &path, "source", lax::string) else {
                    continue;
                };
                Artifact {
                    source,
                    destination: r
                        .get(m, &path, "destination", lax::string)
                        .filter(|d| !d.is_empty()),
                    exclude: r.get(m, &path, "exclude", lax::strings).unwrap_or_default(),
                    service: r.get(m, &path, "service", lax::string),
                }
            }
            other => {
                let msg = format!(
                    "an artifact must be a path or a table, not {}",
                    lax::describe(other)
                );
                r.invalid(&path, msg);
                continue;
            }
        };
        if artifact.source.split('/').any(|p| p == "..") || artifact.source.trim().is_empty() {
            r.invalid(
                &path,
                format!(
                    "artifact source {:?} must be a path without '..'",
                    artifact.source
                ),
            );
            continue;
        }
        if let Some(dest) = &artifact.destination {
            let parts: Vec<&str> = dest
                .split('/')
                .filter(|p| !p.is_empty() && *p != ".")
                .collect();
            if dest.contains('\\')
                || dest.starts_with('/')
                || parts.is_empty()
                || parts.contains(&"..")
                || dest.trim_end_matches('/') == "manifest.json"
            {
                r.invalid(
                    &join(&path, "destination"),
                    format!("artifact destination {dest:?} must be a relative path inside the trial's artifacts folder, without '..', and not manifest.json"),
                );
                continue;
            }
        }
        if let Some(service) = &artifact.service {
            if !SERVICE_NAME.is_match(service.trim()) {
                r.invalid(
                    &join(&path, "service"),
                    format!("invalid `service`: {service:?} is not a Compose service name"),
                );
                continue;
            }
            if service.trim() != "main" && !artifact.source.starts_with('/') {
                r.invalid(
                    &join(&path, "source"),
                    format!("artifact source {:?} collected from service {service:?} must be an absolute path", artifact.source),
                );
                continue;
            }
        }
        out.push(artifact);
    }
    out
}

/// BenchFlow 0.8's sandbox defaults, which apply to its `task.md`.
fn apply_benchflow_defaults(s: &mut Sandbox) {
    s.build_timeout_sec.get_or_insert(600.0);
    s.cpus.get_or_insert(1);
    s.memory_mb.get_or_insert(2048);
    s.storage_mb.get_or_insert(10240);
    s.gpus.get_or_insert(0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::{Source, line_col};

    struct Toml<'a> {
        src: &'a Source,
        text: &'a str,
    }

    impl Locate for Toml<'_> {
        fn locate(&self, path: &str, value: bool) -> (Option<usize>, Option<(usize, usize)>) {
            let span = if value {
                self.src.value_span(path)
            } else {
                self.src.best_span(path)
            };
            match span {
                Some(r) => (Some(line_col(self.text, r.start).0), Some((r.start, r.end))),
                None => (None, None),
            }
        }
    }

    const CODES: Codes = Codes {
        invalid: "E",
        ignored: "W",
        deprecated: "N",
        benchflow_only: "B",
    };

    fn read_toml(text: &str) -> (Config, Vec<Diagnostic>) {
        let src = Source::toml(text, 0).unwrap();
        let loc = Toml { src: &src, text };
        let mut r = Reader {
            dialect: Dialect::TaskToml,
            file: Path::new("task.toml"),
            loc: &loc,
            codes: CODES,
            diags: Vec::new(),
        };
        let cfg = read(&mut r, &src.value);
        (cfg, r.diags)
    }

    #[test]
    fn reads_a_skillsbench_config() {
        let (cfg, diags) = read_toml(
            "version = \"1.0\"\n[verifier]\ntimeout_sec = 1200.0\n[agent]\ntimeout_sec = 1200.0\n[environment]\nbuild_timeout_sec = 600.0\ncpus = 4\ngpus = 0\nmemory = \"4G\"\nstorage = \"5G\"\nallow_internet = true\n",
        );
        assert_eq!(cfg.schema_version.as_deref(), Some("1.0"));
        assert_eq!(cfg.sandbox.memory_mb, Some(4096));
        assert_eq!(cfg.sandbox.storage_mb, Some(5120));
        assert_eq!(cfg.sandbox.network.mode, NetworkMode::Public);
        assert_eq!(cfg.agent.timeout_sec, Some(1200.0));
        assert!(
            diags.iter().all(|d| d.severity == Severity::Note),
            "{diags:#?}"
        );
        let memory = diags
            .iter()
            .find(|d| d.message.contains("`memory`"))
            .unwrap();
        let fix = memory.fix.as_ref().unwrap();
        assert_eq!(fix.replacement, "memory_mb = 4096");
    }

    #[test]
    fn allow_internet_false_cuts_the_network() {
        let (cfg, _) = read_toml("[environment]\nallow_internet = false\n");
        assert_eq!(cfg.sandbox.network.mode, NetworkMode::NoNetwork);
        let (cfg, diags) =
            read_toml("[environment]\nallow_internet = false\nnetwork_mode = \"public\"\n");
        assert_eq!(cfg.sandbox.network.mode, NetworkMode::Public);
        assert!(
            diags
                .iter()
                .any(|d| d.code == "W" && d.message.contains("allow_internet"))
        );
    }

    #[test]
    fn bad_values_are_errors_at_the_value() {
        let text = "[environment]\ncpus = \"four\"\nnetwork_mode = \"offline\"\n[task]\nname = \"no-slash\"\n";
        let (_, diags) = read_toml(text);
        let errors: Vec<_> = diags.iter().filter(|d| d.is_error()).collect();
        assert_eq!(errors.len(), 3, "{errors:#?}");
        let cpus = errors.iter().find(|d| d.message.contains("cpus")).unwrap();
        let span = cpus.location.as_ref().unwrap().span.unwrap();
        assert_eq!(&text[span.0..span.1], "\"four\"");
        assert_eq!(cpus.location.as_ref().unwrap().line, Some(2));
    }

    #[test]
    fn allowlists_need_hosts_and_hosts_need_an_allowlist() {
        let (cfg, diags) =
            read_toml("[agent]\nnetwork_mode = \"allowlist\"\nallowed_hosts = [\"PyPI.org\"]\n");
        assert!(diags.is_empty(), "{diags:#?}");
        assert_eq!(
            cfg.agent.network.allowed_hosts,
            vec!["pypi.org".to_string()]
        );
        let (_, diags) = read_toml("[environment]\nallowed_hosts = [\"pypi.org\"]\n");
        assert!(diags.iter().any(|d| d.is_error()));
    }

    #[test]
    fn harbor_and_benchflow_spellings() {
        let (cfg, diags) = read_toml("[sandbox]\ncpus = 2\n[oracle]\nenv = { A = \"1\" }\n");
        assert_eq!(cfg.sandbox.cpus, Some(2));
        assert_eq!(cfg.oracle.env.get("A").map(String::as_str), Some("1"));
        assert_eq!(diags.iter().filter(|d| d.code == "B").count(), 2);
        let (cfg, diags) = read_toml("[environment]\ncpus = 1\n[sandbox]\ncpus = 2\n");
        assert_eq!(cfg.sandbox.cpus, Some(1));
        assert!(diags.iter().any(|d| d.code == "W"));
    }

    #[test]
    fn solution_env_shorthand_is_read() {
        let (cfg, diags) = read_toml("[solution]\nAPI_KEY = \"x\"\n[solution.env]\nB = \"y\"\n");
        assert_eq!(cfg.oracle.env.len(), 2);
        assert!(diags.iter().any(|d| d.code == "B"));
    }

    #[test]
    fn steps_and_artifacts() {
        let (cfg, diags) = read_toml(
            "artifacts = [\"/logs/x\", { source = \"/var/log\", destination = \"logs\" }]\n[[steps]]\nname = \"one\"\n[[steps]]\nname = \"two\"\n[steps.agent]\ntimeout_sec = 10\n",
        );
        assert!(diags.is_empty(), "{diags:#?}");
        assert_eq!(cfg.steps, 2);
        assert_eq!(cfg.artifacts.len(), 2);
        let (_, diags) = read_toml("artifacts = [{ source = \"../up\" }]\n");
        assert!(diags.iter().any(|d| d.is_error()));
    }

    #[test]
    fn harbor_sizes() {
        assert_eq!(harbor_size_mb("4G"), Some(4096));
        assert_eq!(harbor_size_mb(" 512m "), Some(512));
        assert_eq!(harbor_size_mb("1.5G"), Some(1536));
        assert_eq!(harbor_size_mb("2048K"), Some(2));
        assert_eq!(harbor_size_mb("4GB"), None);
    }

    #[test]
    fn step_names() {
        assert!(valid_step_name("build"));
        assert!(!valid_step_name("a/b"));
        assert!(!valid_step_name("CON"));
        assert!(!valid_step_name("com1.txt"));
        assert!(!valid_step_name("trailing."));
    }
}
