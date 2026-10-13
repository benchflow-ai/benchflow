//! A checked draft 2 package as the run engine's [`TaskPackage`].
//!
//! Fields map as BenchFlow 0.8 maps them (docs/task-authoring-taskmd-v2.md,
//! "What BenchFlow supports, field by field"): honored fields fill the
//! package, and each field 0.8 refuses becomes an [`Unsupported`] entry
//! naming it, so a run is refused by name before anything starts. A field
//! 0.8 honors but this package has no place for (Compose services, GPU
//! types, a verifier's own image, families) is refused the same way, since
//! the engine cannot honor what it never receives.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::time::Duration;

use regex::Regex;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::package::{
    AgentPolicy, Control, Format, ImageRef, Isolation, OnTimeout, OraclePolicy, Output, Ready,
    SandboxPolicy, Stage, TaskPackage, Unsupported, VerifierPolicy,
};
use crate::rubric::Rubric;
use crate::values::Network;

use super::judges::resolve;
use super::judgment::Judgment;
use super::pyfmt::{repr, str_of};
use super::pyval::{duration_s, int, num, py_eq, size_mb, table, truthy};
use super::{BlockKind, TaskDocument};

/// `${VAR}` or `${VAR:-default}` in an env value.
static HOST_VAR: LazyLock<Regex> = LazyLock::new(|| Regex::new(super::vocab::HOST_VAR).unwrap());

/// Collects the fields this runtime cannot honor.
#[derive(Default)]
pub(super) struct Refusals(pub(super) Vec<Unsupported>);

impl Refusals {
    fn refuse(&mut self, field: impl Into<String>, reason: impl Into<String>) {
        self.0.push(Unsupported {
            field: field.into(),
            reason: reason.into(),
            agents_only: false,
        });
    }

    fn refuse_agents(&mut self, field: impl Into<String>, reason: impl Into<String>) {
        self.0.push(Unsupported {
            field: field.into(),
            reason: reason.into(),
            agents_only: true,
        });
    }
}

fn seconds(v: Option<&Value>) -> Option<Duration> {
    duration_s(v).map(crate::values::duration_from_secs)
}

/// An env table as strings, refusing `${VAR}` values, which a runtime may
/// resolve only for variables the operator allows.
fn literal_env(
    refusals: &mut Refusals,
    where_: &str,
    env: Option<&Value>,
) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    let Some(env) = env else {
        return out;
    };
    let Value::Object(env) = env else {
        refusals.refuse(where_, "must be a table of strings");
        return out;
    };
    for (key, value) in env {
        let Value::String(text) = value else {
            refusals.refuse(format!("{where_}.{key}"), "must be a string");
            continue;
        };
        if HOST_VAR.is_match(text) {
            refusals.refuse(
                format!("{where_}.{key}"),
                "${VAR} resolves from the host only for variables the operator allows, and never for a provider key; this runtime does \
not resolve host variables, so templated values are refused",
            );
            continue;
        }
        out.insert(key.clone(), text.clone());
    }
    out
}

/// A network value as a [`Network`]; `None` (with a refusal) when it is
/// not one.
fn network(refusals: &mut Refusals, where_: &str, v: &Value) -> Option<Network> {
    match Network::from_value(v) {
        Ok(n) => Some(n),
        Err(e) => {
            refusals.refuse(where_, e);
            None
        }
    }
}

fn sandbox(cfg: &Value, root: &Path, refusals: &mut Refusals) -> SandboxPolicy {
    let sb = table(cfg, &["sandbox"]);
    let image = match sb
        .get("image")
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty())
    {
        Some(reference) => ImageRef::Pull {
            reference: reference.to_string(),
        },
        None => ImageRef::Build {
            context: root.join("sandbox"),
            dockerfile: root.join("sandbox/Dockerfile"),
        },
    };
    let mut policy = SandboxPolicy {
        image,
        cpus: None,
        memory_mb: None,
        disk_mb: None,
        gpus: 0,
        build_timeout: seconds(sb.get("build_timeout")),
        network: Network::Open,
        workdir: sb.get("workdir").map(str_of),
        env: literal_env(refusals, "[sandbox] env", sb.get("env")),
        ready: sb.get("ready").and_then(ready),
        outputs: Vec::new(),
        skills: None,
        skills_dir: None,
        mcp: Vec::new(),
        allow_cloud_metadata: false,
    };
    for (key, value) in sb {
        let where_ = format!("[sandbox] {key}");
        match key.as_str() {
            "os" if value.as_str() != Some("linux") => {
                refusals.refuse(where_, format!("{} sandboxes are not provided: this runtime runs Linux containers", repr(value)));
            }
            "cpus" => match num(value).filter(|c| *c > 0.0) {
                Some(c) => policy.cpus = Some(c),
                None => refusals.refuse(where_, "must be a positive number of CPUs"),
            },
            "memory" => policy.memory_mb = size_mb(value),
            "disk" => policy.disk_mb = size_mb(value),
            "gpus" => match int(value).and_then(|g| u32::try_from(g).ok()) {
                Some(g) => policy.gpus = g,
                None => refusals.refuse(where_, "must be a whole number of GPUs"),
            },
            "tpu" if [Value::Null, Value::from(0), Value::Object(Map::new()), Value::Array(vec![])].iter().any(|x| py_eq(value, x)) => {}
            "tpu" => refusals.refuse(where_, "TPUs are not mapped yet"),
            "network" => {
                if let Some(n) = network(refusals, &where_, value) {
                    policy.network = n;
                }
            }
            // bench's own, in the format's extension space: other runners
            // never refuse the cloud metadata endpoints, so they may skip it.
            "x-allow-cloud-metadata" => match value {
                Value::Bool(b) => policy.allow_cloud_metadata = *b,
                _ => refusals.refuse(where_, "must be true or false"),
            },
            // A folder in the package (`sandbox/skills`), or an absolute path
            // the image fills (a Harbor task's skills_dir).
            "skills" => match value.as_str() {
                Some(s) if s.starts_with('/') => policy.skills_dir = Some(s.to_string()),
                Some(s) => policy.skills = Some(root.join(s)),
                None => refusals.refuse(where_, "must be a folder: `sandbox/skills`, or an absolute path in the sandbox"),
            },
            "mcp" => match value {
                Value::Array(servers) => policy.mcp = servers.clone(),
                Value::Null => {}
                _ => refusals.refuse(where_, "must be a list of MCP server tables"),
            },
            "outputs" => policy.outputs = outputs(value, refusals),
            "boundary" if value.as_str() == Some("container") => {}
            "boundary" => refusals.refuse(where_, format!("{} isolation is not guaranteed by this runtime's sandboxes", repr(value))),
            "clock" => refusals.refuse(where_, "this runtime sets no clock: the agent would see the real time"),
            "timezone" => refusals.refuse(
                where_,
                "this runtime cannot check that the image holds the zone's data, and without it the zone falls back to UTC without an \
error",
            ),
            _ => {}
        }
    }
    policy
}

pub(super) fn ready(v: &Value) -> Option<Ready> {
    let r = v.as_object()?;
    let dur = |key: &str, default: f64| {
        duration_s(r.get(key)).map_or_else(
            || Duration::from_secs_f64(default),
            crate::values::duration_from_secs,
        )
    };
    Some(Ready {
        run: r.get("run").map(str_of)?,
        interval: dur("interval", 5.0),
        timeout: dur("timeout", 30.0),
        start_period: dur("start_period", 0.0),
        start_interval: dur("start_interval", 5.0),
        retries: r
            .get("retries")
            .and_then(int)
            .and_then(|n| u32::try_from(n).ok())
            .unwrap_or(3),
    })
}

pub(super) fn outputs(v: &Value, refusals: &mut Refusals) -> Vec<Output> {
    let Value::Array(items) = v else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for (n, item) in items.iter().enumerate() {
        let (path, entry) = match item {
            Value::String(p) => (p.clone(), None),
            Value::Object(m) => (m.get("path").map(str_of).unwrap_or_default(), Some(m)),
            _ => continue,
        };
        let default_name = path
            .trim_end_matches('/')
            .rsplit('/')
            .next()
            .unwrap_or("")
            .to_string();
        let mut output = Output {
            save_as: default_name,
            path,
            max_bytes: None,
            exclude: Vec::new(),
        };
        if let Some(m) = entry {
            if let Some(s) = m.get("save_as").and_then(Value::as_str) {
                output.save_as = s.to_string();
            }
            if let Some(ex) = m.get("exclude").and_then(Value::as_array) {
                output.exclude = ex
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect();
            }
            if let Some(mb) = m.get("max_bytes") {
                output.max_bytes = crate::values::parse_bytes(mb)
                    .or_else(|| size_mb(mb).map(|x| x.saturating_mul(1024 * 1024)));
                refusals.refuse(
                    format!("[sandbox] outputs[{n}] max_bytes"),
                    "per-output caps are not implemented yet",
                );
            }
            if m.contains_key("service") {
                refusals.refuse(
                    format!("[sandbox] outputs[{n}] service"),
                    "outputs saved from a service's container are not implemented",
                );
            }
        }
        out.push(output);
    }
    out
}

fn agent(cfg: &Value, sandbox_network: &Network, refusals: &mut Refusals) -> AgentPolicy {
    let a = table(cfg, &["agent"]);
    let mut policy = AgentPolicy {
        timeout: seconds(a.get("timeout")),
        ..AgentPolicy::default()
    };
    for (key, value) in a {
        let where_ = format!("[agent] {key}");
        match key.as_str() {
            "on_timeout" => match value.as_str() {
                Some("grade") => policy.on_timeout = OnTimeout::Grade,
                other => {
                    policy.on_timeout = if other == Some("fail") {
                        OnTimeout::Fail
                    } else {
                        OnTimeout::GradeFlagged
                    };
                    refusals.refuse(
                        where_,
                        format!("{} is not implemented: this runtime grades what the agent left at the time limit", repr(value)),
                    );
                }
            },
            "budget" => refusals.refuse_agents(
                where_,
                "this runtime enforces no tool-call or token budget on an agent",
            ),
            "user" => policy.user = value.as_str().map(str::to_string),
            "network" => {
                let Some(n) = network(refusals, &where_, value) else {
                    continue;
                };
                let online = *sandbox_network != Network::None;
                let narrower = n == *sandbox_network
                    || (n == Network::None && online)
                    || (matches!(n, Network::Allow(_)) && online)
                    || n.within(sandbox_network);
                if narrower {
                    policy.network = Some(n);
                } else {
                    refusals.refuse(
                        where_,
                        format!(
                            "{} differs from [sandbox] network ({sandbox_network}); the agent works inside the sandbox's container, so its \
network can be narrower than the sandbox's, never wider",
                            repr(value)
                        ),
                    );
                }
            }
            "system_prompt_append" => refusals.refuse_agents(
                where_,
                "this runtime's harnesses take no system prompt addition",
            ),
            "timeout_basis" if value.as_str() == Some("wall") => {}
            "timeout_basis" => {
                refusals.refuse(where_, "this runtime counts only wall time, and cannot freeze the sandbox while a policy generates");
            }
            _ => {}
        }
    }
    policy
}

fn verifier(
    cfg: &Value,
    root: &Path,
    sandbox_network: &Network,
    rubric: Option<Rubric>,
    refusals: &mut Refusals,
) -> VerifierPolicy {
    let v = table(cfg, &["verifier"]);
    let separate = v.get("isolation").and_then(Value::as_str) == Some("separate");
    let mut policy = VerifierPolicy {
        dir: root.join("verifier"),
        mount: "/verifier".into(),
        entry: "test.sh".into(),
        starts_in_mount: false,
        timeout: seconds(v.get("timeout")),
        user: v.get("user").and_then(Value::as_str).map(str::to_string),
        env: literal_env(refusals, "[verifier] env", v.get("env")),
        network: None,
        isolation: if separate {
            Isolation::Separate
        } else {
            Isolation::Shared
        },
        rubric,
        reward_range: None,
    };
    for (key, value) in v {
        let where_ = format!("[verifier] {key}");
        match key.as_str() {
            "mount" => match value.as_str() {
                Some(m @ ("/verifier" | "/tests")) => policy.mount = m.to_string(),
                _ => refusals.refuse(
                    where_,
                    format!("{} is not supported: the verifier appears at /verifier or /tests", repr(value)),
                ),
            },
            "sandbox" if !separate => refusals.refuse(
                where_,
                "[verifier.sandbox] sets a separate verifier's container; with isolation = \"shared\" the verifier runs in the agent's \
container, so these settings would apply to nothing",
            ),
            "sandbox" => {
                for k in table(cfg, &["verifier", "sandbox"]).keys() {
                    if k != "image" && k != "network" && !k.starts_with("x-") {
                        refusals.refuse(
                            format!("[verifier.sandbox] {k}"),
                            "a separate verifier's sandbox takes the task's resources; only its image and network are read",
                        );
                    }
                }
            }
            "models" if value == &Value::Bool(true) => refusals.refuse(
                where_,
                "a verifier script that calls a model needs the model proxy's seat token, which this runtime does not provide yet",
            ),
            "human" => refusals.refuse(where_, "human assessment is not supported"),
            _ => {}
        }
    }
    // The verifier's network.
    let declared = v.get("network");
    if separate {
        let own = table(cfg, &["verifier", "sandbox"])
            .get("network")
            .or(declared);
        if let Some(n) = own
            && let Some(n) = network(refusals, "[verifier] network", n)
        {
            policy.network = Some(n);
        }
        if root.join("verifier/docker-compose.yaml").is_file()
            && !root.join("verifier/Dockerfile").is_file()
        {
            refusals.refuse(
                "verifier/docker-compose.yaml",
                "a verifier image built from Compose is not supported",
            );
        }
    } else if let Some(d) = declared
        && let Some(n) = network(refusals, "[verifier] network", d)
    {
        match &n {
            Network::None => policy.network = Some(n),
            Network::Open if *sandbox_network == Network::None => refusals.refuse(
                "[verifier] network",
                "\"open\", and the agent's container is offline ([sandbox] network = \"none\"); a shared verifier cannot get a network \
the container lacks: use isolation = \"separate\"",
            ),
            Network::Open => policy.network = Some(n),
            _ if n.within(sandbox_network) => policy.network = Some(n),
            _ => refusals.refuse(
                "[verifier] network",
                format!(
                    "{} is wider than [sandbox] network ({sandbox_network}); a shared verifier grades in the agent's container, so its \
network can be narrower than the sandbox's, never wider: use isolation = \"separate\"",
                    repr(d)
                ),
            ),
        }
    }
    policy
}

fn oracle(cfg: &Value, root: &Path, refusals: &mut Refusals) -> Option<OraclePolicy> {
    let o = table(cfg, &["oracle"]);
    let env = literal_env(refusals, "[oracle] env", o.get("env"));
    let mut mount = "/oracle".to_string();
    if let Some(m) = o.get("mount") {
        match m.as_str() {
            Some(x @ ("/oracle" | "/solution")) => mount = x.to_string(),
            _ => refusals.refuse(
                "[oracle] mount",
                format!(
                    "{} is not supported: the oracle appears at /oracle or /solution",
                    repr(m)
                ),
            ),
        }
    }
    root.join("oracle/solve.sh")
        .is_file()
        .then(|| OraclePolicy {
            dir: root.join("oracle"),
            mount,
            entry: "solve.sh".into(),
            env,
        })
}

/// The top-level tables beyond the four phases.
fn tables(cfg: &Value, refusals: &mut Refusals) {
    let Value::Object(top) = cfg else {
        return;
    };
    for (key, value) in top {
        let where_ = format!("[{key}]");
        match key.as_str() {
            "world" => refusals.refuse(where_, "worlds (desktop, browser, simulator, robot, lab) are not provided"),
            "tiers" => refusals.refuse(where_, "sim-to-real tiers are not provided"),
            "conventions" => refusals.refuse(where_, "conventions are read by a world's profile, and this runtime provides no world"),
            "roles" => refusals.refuse(where_, ROLES_REFUSED),
            "user" => refusals.refuse(where_, USER_REFUSED),
            "interaction" => refusals.refuse(where_, "multi-agent interaction is not mapped"),
            "matrix" => refusals.refuse(where_, "condition matrices are not run: only the base task would run"),
            "preference" => refusals.refuse(where_, "the human-preference protocol is not run"),
            "family" => refusals.refuse(
                where_,
                "task families are not run by this runtime yet: it does not run the generator, so no instance exists",
            ),
            "stages" => {
                for (name, stage) in table(cfg, &["stages"]) {
                    for k in super::pyval::obj(Some(stage)).keys() {
                        if !["unlock", "gate", "outputs", "agent", "verifier", "ready", "mounts"].contains(&k.as_str())
                            && !k.starts_with("x-")
                        {
                            refusals.refuse(
                                format!("[stages.{name}] {k}"),
                                "stages unlocked by a submitted file are not implemented; a stage unlocks when the agent ends its \
previous turn",
                            );
                        }
                    }
                }
            }
            "training" if table(cfg, &["training"]).contains_key("max_concurrent") => {
                refusals.refuse("[training] max_concurrent", "this runtime does not limit how many trials of one task run at once yet");
            }
            "trajectory" => {
                let require = table(cfg, &["trajectory"]).get("require");
                let items: Vec<&Value> = match require {
                    Some(Value::Array(a)) => a.iter().collect(),
                    Some(other) => vec![other],
                    None => Vec::new(),
                };
                for item in items {
                    if item.as_str() != Some("tool-calls") {
                        refusals.refuse(
                            format!("[trajectory] require {}", str_of(item)),
                            "this runtime cannot guarantee a trajectory holds this",
                        );
                    }
                }
            }
            "integrity" => {
                let integrity = table(cfg, &["integrity"]);
                if integrity.contains_key("forbidden_sources") {
                    refusals.refuse("[integrity] forbidden_sources", "these sources are not blocked yet");
                }
                if integrity.contains_key("resources") {
                    refusals.refuse("[integrity] resources", "per-path resource classes are not enforced");
                }
                if let Some(profile) = integrity.get("profile") {
                    let separate = table(cfg, &["verifier"]).get("isolation").and_then(Value::as_str) == Some("separate");
                    match profile.as_str() {
                        Some("shared-hardened") => {}
                        Some("separated-verifier") if separate => {}
                        Some("separated-verifier") => {
                            refusals.refuse("[integrity] profile", "needs [verifier] isolation = \"separate\"");
                        }
                        _ => refusals.refuse("[integrity] profile", format!("{} is not provided", repr(profile))),
                    }
                }
            }
            "x-benchflow" if truthy(Some(value)) => refusals.refuse(
                where_,
                "BenchFlow owns this extension and would have to honor it, and it does not read 0.8 settings from a draft 2 package",
            ),
            _ => {}
        }
    }
}

const ROLES_REFUSED: &str = "multi-agent roles are not mapped: task.md leaves open what each role is told and in which order roles act";
const USER_REFUSED: &str = "simulated users are not mapped: a persona that answers the agent's questions needs a conversation this \
runtime does not hold";

/// Stage, role, and user blocks; stages that unlock in a chain from the
/// instruction become later prompts of the same session.
fn stages(doc: &TaskDocument, refusals: &mut Refusals) -> Vec<Stage> {
    for block in &doc.blocks {
        match block.kind {
            BlockKind::Role => refusals.refuse(
                format!("```role {}", block.arg.as_deref().unwrap_or("")),
                ROLES_REFUSED,
            ),
            BlockKind::User => refusals.refuse("```user", USER_REFUSED),
            _ => {}
        }
    }
    let mut out = Vec::new();
    let mut previous: Option<String> = None;
    for (index, (name, stage)) in table(doc.config_value(), &["stages"]).iter().enumerate() {
        let unlock = stage.get("unlock");
        let where_ = format!("[stages.{name}] unlock");
        if unlock.and_then(Value::as_str) == Some("at_start") {
            if index != 0 {
                refusals.refuse(where_, "at_start must be the first stage");
            }
            previous = Some(name.clone());
            continue;
        }
        let text = unlock.and_then(Value::as_str);
        let chained = text == Some("on_submit")
            || (previous.is_some()
                && text == previous.as_ref().map(|p| format!("after:{p}")).as_deref());
        if !chained {
            refusals.refuse(
                where_,
                format!(
                    "{} is not mapped: a stage is revealed only after the agent ends its previous turn, so stages must unlock in \
declaration order (on_submit, or after:<the stage before>)",
                    unlock.map_or_else(|| "None".to_string(), repr)
                ),
            );
            continue;
        }
        out.push(Stage {
            name: name.clone(),
            prompt: doc.stage_prompt(name).unwrap_or_default(),
        });
        previous = Some(name.clone());
    }
    out
}

/// The package's folders and files this runtime does not use.
fn package_files(cfg: &Value, root: &Path, refusals: &mut Refusals) {
    if root.join("verifier/verifier.md").is_file() {
        refusals.refuse(
            "verifier/verifier.md",
            "a BenchFlow verifier.md would replace task.md's verifier contract",
        );
    }
    if root.join(super::extras::COMPOSE_PATH).is_file()
        && !table(cfg, &["sandbox"]).contains_key("compose")
    {
        refusals.refuse(
            super::extras::COMPOSE_PATH,
            "the services in this file would start beside the agent's container, but [sandbox] compose does not declare it: add \
[sandbox] compose = \"sandbox/docker-compose.yaml\"",
        );
    }
}

/// Evidence items a judge can be served: files, the trajectory, the
/// trajectory's reasoning, and the tests.
const HONORED_EVIDENCE: &[&str] = &["trajectory", "trajectory:reasoning", "tests"];

/// The rubric and behaviors: what this runtime grades, and what it refuses.
fn judging(
    cfg: &Value,
    rubric: Option<&Judgment>,
    behaviors: Option<&Judgment>,
    root: &Path,
    verifier_network: Option<&Network>,
    refusals: &mut Refusals,
) {
    if let Some(b) = behaviors
        && (truthy(b.value.get("watch")) || truthy(b.value.get("paired")))
    {
        refusals.refuse(
            "verifier/behaviors.json",
            "watched and paired behaviors are not detected, and their consequences and tags are not applied",
        );
    }
    let judges = table(cfg, &["verifier", "judges"]);
    let Some(rubric) = rubric else {
        if !judges.is_empty() {
            refusals.refuse(
                "[verifier] judges",
                "the task has no task.md rubric for a judge to grade",
            );
        }
        return;
    };
    let r = &rubric.value;
    if let Some(Value::Array(urls)) = r.get("extends") {
        for url in urls {
            refusals.refuse(
                "verifier/rubric.json extends",
                format!(
                    "{} is not at hand: shared rubrics are not fetched by this runtime",
                    str_of(url)
                ),
            );
        }
        if !urls.is_empty() {
            return;
        }
    }
    let crits: Vec<&Map<String, Value>> = match r.get("criteria") {
        Some(Value::Array(c)) => c.iter().filter_map(Value::as_object).collect(),
        _ => Vec::new(),
    };
    let streams: Vec<&str> = match table(cfg, &["world", "record"]).get("streams") {
        Some(Value::Array(s)) => s.iter().filter_map(Value::as_str).collect(),
        _ => Vec::new(),
    };
    let tests = crits
        .iter()
        .filter(|c| c.get("judge").and_then(Value::as_str) == Some("test"))
        .count();
    if tests > 0 && !root.join("verifier/test.sh").is_file() {
        refusals.refuse(
            "verifier/test.sh",
            format!("{tests} criteria are decided by tests, and the package has no verifier/test.sh to report them"),
        );
    }
    let mut roles: Vec<&str> = Vec::new();
    for c in &crits {
        let judge = c.get("judge").and_then(Value::as_str).unwrap_or("");
        let where_ = format!(
            "verifier/rubric.json {}",
            c.get("id").map(str_of).unwrap_or_default()
        );
        match judge {
            "test" => continue,
            "rule" | "world" | "replay" | "human" => {
                refusals.refuse(
                    where_,
                    format!("criteria judged by {judge} are not implemented"),
                );
                continue;
            }
            "vlm" => {
                refusals.refuse(where_, "the vlm role is not implemented");
                continue;
            }
            "panel" => {
                refusals.refuse(
                    where_,
                    "the panel role is not implemented (task.md leaves the panel role undefined)",
                );
                continue;
            }
            _ => {}
        }
        if !roles.contains(&judge) {
            roles.push(judge);
        }
        if let Some(Value::Array(items)) = c.get("evidence") {
            for item in items {
                let text = str_of(item);
                let kind = text.split('#').next().unwrap_or("");
                if HONORED_EVIDENCE.contains(&kind) {
                    continue;
                }
                let runtime = ["screenshots", "video", "request-record"];
                if runtime.contains(&kind) || streams.contains(&kind) || kind.starts_with("diff:") {
                    refusals.refuse(
                        format!("{where_} evidence {text}"),
                        "not provided yet: judges are served files, trajectory, trajectory:reasoning, and tests",
                    );
                }
            }
        }
    }
    let separate = table(cfg, &["verifier"])
        .get("isolation")
        .and_then(Value::as_str)
        == Some("separate");
    for role in roles {
        let settings = resolve(cfg, role, None, Some(root));
        let where_ = format!("[verifier.judges.{role}]");
        if settings.get("effort").is_some() {
            refusals.refuse(
                format!("{where_} effort"),
                "reasoning effort is not passed to the judge model yet",
            );
        }
        if let Some(h) = settings.get("harness")
            && h.as_str() != Some("judge-loop@1")
        {
            refusals.refuse(
                format!("{where_} harness"),
                format!(
                    "{}: only judge-loop@1 runs; hosted harnesses are not admitted",
                    repr(h)
                ),
            );
        }
        if truthy(settings.get("resources")) {
            refusals.refuse(
                format!("{where_} resources"),
                "a judge's runner uses the verifier sandbox's resources",
            );
        }
        if truthy(settings.get("services")) {
            refusals.refuse(
                format!("{where_} services"),
                "a judge's runner cannot reach the task's services",
            );
        }
        if let Some(Value::Array(views)) = settings.get("views")
            && views
                .iter()
                .any(|v| v.get("format").and_then(Value::as_str) != Some("trajectory-1"))
        {
            refusals.refuse(
                format!("{where_} views"),
                "only trajectory-1 views are defined",
            );
        }
        if role == "agent" {
            if !separate {
                refusals.refuse(
                    where_.clone(),
                    "an agent judge's commands run in a fresh copy of the submission's environment, which needs [verifier] isolation = \
\"separate\"",
                );
            }
            if verifier_network != Some(&Network::None) {
                refusals.refuse(
                    where_,
                    "an agent judge's runners have no network: set [verifier] network = \"none\"",
                );
            }
        }
    }
}

/// Controls: scripts in `controls/` and every path `[integrity.controls]`
/// names. A no-op or cheat control must score 0; a known-bad or near-miss
/// control must score below the pass threshold.
fn controls(cfg: &Value, root: &Path) -> Vec<Control> {
    let mut declared: Vec<(String, Option<f64>)> = Vec::new();
    for (key, value, _) in super::files::control_paths(cfg) {
        let Value::String(path) = value else {
            continue;
        };
        let max = matches!(key.as_str(), "no_op" | "cheats").then_some(0.0);
        declared.push((path, max));
    }
    let mut scripts: Vec<String> = declared.iter().map(|(p, _)| p.clone()).collect();
    if let Ok(entries) = std::fs::read_dir(root.join("controls")) {
        let mut found: Vec<String> = entries
            .filter_map(Result::ok)
            .filter(|e| e.path().is_file())
            .filter_map(|e| e.file_name().to_str().map(|n| format!("controls/{n}")))
            .filter(|p| !p.ends_with(".patch") && !p.ends_with(".diff"))
            .filter(|p| {
                p.ends_with(".sh")
                    || std::fs::read(root.join(p)).is_ok_and(|b| b.starts_with(b"#!"))
            })
            .collect();
        found.sort();
        for p in found {
            if !scripts.contains(&p) {
                scripts.push(p);
            }
        }
    }
    let mut out: Vec<Control> = Vec::new();
    for script in scripts {
        if !root.join(&script).is_file() {
            continue;
        }
        let stem = Path::new(&script)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("control")
            .to_string();
        let id = if out.iter().any(|c| c.id == stem) {
            script.replace('/', "-")
        } else {
            stem
        };
        let max_reward = declared
            .iter()
            .find(|(p, _)| *p == script)
            .and_then(|(_, m)| *m);
        out.push(Control {
            id,
            script: PathBuf::from(&script),
            max_reward,
        });
    }
    out
}

/// A provisional content digest: `sha256:` over each file's path relative to
/// the package, a zero byte, and the SHA-256 of its bytes, in path order.
/// Symbolic links are skipped. `crate::digest` will replace this with the
/// canonical tree hash.
pub(crate) fn provisional_tree_digest(root: &Path) -> String {
    let mut files: Vec<(String, PathBuf)> = walkdir::WalkDir::new(root)
        .follow_links(false)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
        .filter_map(|e| {
            let rel = e
                .path()
                .strip_prefix(root)
                .ok()?
                .to_string_lossy()
                .replace('\\', "/");
            Some((rel, e.into_path()))
        })
        .collect();
    files.sort_by(|a, b| a.0.cmp(&b.0));
    let mut hasher = Sha256::new();
    for (rel, path) in files {
        let Ok(bytes) = std::fs::read(&path) else {
            continue;
        };
        hasher.update(rel.as_bytes());
        hasher.update([0u8]);
        hasher.update(Sha256::digest(&bytes));
    }
    format!("sha256:{}", hex::encode(hasher.finalize()))
}

/// Build the package from a document that checked clean.
pub(crate) fn package(
    doc: &TaskDocument,
    rubric: Option<&Judgment>,
    behaviors: Option<&Judgment>,
    root: &Path,
) -> TaskPackage {
    let cfg = doc.config_value();
    let mut refusals = Refusals::default();
    let mut sandbox = sandbox(cfg, root, &mut refusals);
    let agent = agent(cfg, &sandbox.network, &mut refusals);
    let model = rubric.and_then(|r| Rubric::from_value(&r.value).ok());
    let verifier = verifier(cfg, root, &sandbox.network, model, &mut refusals);
    let oracle = oracle(cfg, root, &mut refusals);
    tables(cfg, &mut refusals);
    let stages = stages(doc, &mut refusals);
    package_files(cfg, root, &mut refusals);
    let mut extras = super::extras::read(
        doc,
        cfg,
        root,
        verifier.isolation == Isolation::Separate,
        &mut refusals.0,
    );
    // The task's own environment wins over its Compose file's.
    for (k, v) in std::mem::take(&mut extras.main_env) {
        sandbox.env.entry(k).or_insert(v);
    }
    let verifier_network = verifier.network.as_ref().or(Some(&sandbox.network));
    judging(
        cfg,
        rubric,
        behaviors,
        root,
        verifier_network,
        &mut refusals,
    );
    // The folder's name as given (a link is named after the link, as 0.8
    // names it); a path with no last name (`.`, `x/..`) is resolved first.
    let name = root
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .or_else(|| {
            root.canonicalize()
                .ok()
                .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
        })
        .unwrap_or_else(|| "task".into());
    let id = cfg
        .get("name")
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty())
        .map_or_else(|| name.clone(), str::to_string);
    TaskPackage {
        root: root.to_path_buf(),
        format: Format::TaskMd,
        id,
        name,
        title: cfg.get("title").and_then(Value::as_str).map(str::to_string),
        version: cfg
            .get("version")
            .and_then(Value::as_str)
            .map(str::to_string),
        instruction: doc.agent_view(),
        stages,
        agent,
        sandbox,
        verifier,
        oracle,
        controls: controls(cfg, root),
        unsupported: refusals.0,
        extras,
        config: cfg.clone(),
        digest: provisional_tree_digest(root),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn a_huge_output_cap_saturates() {
        // `size_mb` takes a trailing newline, as Python's `$` does, so this
        // one skips the saturating `parse_bytes`.
        let mut refusals = Refusals(Vec::new());
        let out = outputs(
            &json!([{"path": "/app/x", "max_bytes": "99999999999999 TB\n"}]),
            &mut refusals,
        );
        assert_eq!(out[0].max_bytes, Some(u64::MAX));
    }
}
