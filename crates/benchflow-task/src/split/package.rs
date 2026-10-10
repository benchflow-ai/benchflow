//! From a checked config and the package's folders to the [`TaskPackage`]
//! the run engine reads. Shared by the split layout and 0.8's task.md.

use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_json::Value;

use super::config::{Config, NetworkMode, PhaseNetwork, SandboxNetwork};
use super::schema::Dialect;
use crate::package::{
    AgentPolicy, Format, ImageRef, Isolation, OnTimeout, OraclePolicy, Output, Ready,
    SandboxPolicy, TaskPackage, Unsupported, VerifierPolicy,
};
use crate::values::Network;

/// Where a package keeps its verifier and oracle, and where they appear in
/// the sandbox.
#[derive(Debug, Clone)]
pub(crate) struct Layout {
    /// `verifier` or `tests`.
    pub verifier_dir: &'static str,
    /// `/verifier` or `/tests`.
    pub verifier_mount: &'static str,
    /// The script the verifier runs, relative to its folder.
    pub verifier_entry: String,
    /// `oracle` or `solution`, when the package has a solve script there.
    pub oracle_dir: Option<&'static str>,
    /// `/oracle` or `/solution`.
    pub oracle_mount: &'static str,
    /// The oracle's script, relative to its folder.
    pub oracle_entry: String,
}

impl Layout {
    /// The folders BenchFlow 0.8 picks: `verifier/` over `tests/` and
    /// `oracle/` over `solution/` when both exist (the native name wins and
    /// never falls back), each mounted under its own name.
    /// A Windows task runs `test.bat` and `solve.bat` instead, as Harbor's
    /// script discovery picks them for `os = "windows"`.
    pub(crate) fn detect(dir: &Path, windows: bool) -> Self {
        let (verifier_dir, verifier_mount) = if dir.join("verifier").is_dir() {
            ("verifier", "/verifier")
        } else {
            ("tests", "/tests")
        };
        let (oracle_dir, oracle_mount) = if dir.join("oracle").is_dir() {
            ("oracle", "/oracle")
        } else {
            ("solution", "/solution")
        };
        let ext = if windows { "bat" } else { "sh" };
        let has_oracle = dir.join(oracle_dir).join(format!("solve.{ext}")).is_file();
        Self {
            verifier_dir,
            verifier_mount,
            verifier_entry: format!("test.{ext}"),
            oracle_dir: has_oracle.then_some(oracle_dir),
            oracle_mount,
            oracle_entry: format!("solve.{ext}"),
        }
    }
}

/// Names config fields the way the file spells them, for `Unsupported`
/// entries: `[environment] skills_dir` in task.toml, `sandbox.skills_dir`
/// in a 0.8 frontmatter.
pub(crate) struct Names {
    pub dialect: Dialect,
    /// The sandbox table's key (`environment` or `sandbox`).
    pub sandbox: &'static str,
}

impl Names {
    pub(crate) fn field(&self, table: &str, key: &str) -> String {
        let table = if table == "sandbox" {
            self.sandbox
        } else {
            table
        };
        match self.dialect {
            Dialect::TaskToml if table.is_empty() => key.to_string(),
            Dialect::TaskToml => format!("[{table}] {key}"),
            Dialect::Frontmatter if table.is_empty() => key.to_string(),
            Dialect::Frontmatter => format!("{table}.{key}"),
        }
    }
}

/// Everything [`build`] needs besides the config.
pub(crate) struct Parts<'a> {
    pub root: &'a Path,
    pub format: Format,
    pub instruction: String,
    pub layout: Layout,
    /// The config as written, for the trial's record.
    pub raw: Value,
    pub names: Names,
    /// Fields already found unsupported (from files rather than keys).
    pub unsupported: Vec<Unsupported>,
}

fn secs(s: f64) -> Duration {
    Duration::from_secs_f64(s.max(0.0))
}

fn phase_network(p: &PhaseNetwork) -> Option<Network> {
    p.mode.map(|m| match m {
        NetworkMode::NoNetwork => Network::None,
        NetworkMode::Public => Network::Open,
        NetworkMode::Allowlist if p.allowed_hosts.is_empty() => Network::None,
        NetworkMode::Allowlist => Network::Allow(p.allowed_hosts.clone()),
        // Refused by the reader; read as open if it ever gets here.
        NetworkMode::Denylist => Network::Open,
    })
}

fn sandbox_network(n: &SandboxNetwork) -> Network {
    match n.mode {
        NetworkMode::NoNetwork => Network::None,
        NetworkMode::Public => Network::Open,
        NetworkMode::Allowlist if n.allowed_hosts.is_empty() => Network::None,
        NetworkMode::Allowlist => Network::Allow(n.allowed_hosts.clone()),
        NetworkMode::Denylist if n.blocked_hosts.is_empty() => Network::Open,
        NetworkMode::Denylist => Network::Block(n.blocked_hosts.clone()),
    }
}

fn unsupported(
    field: impl Into<String>,
    reason: impl Into<String>,
    agents_only: bool,
) -> Unsupported {
    Unsupported {
        field: field.into(),
        reason: reason.into(),
        agents_only,
    }
}

/// Build the normalized package. The digest is left empty; [`crate::load`]
/// computes it, since it reads every file.
pub(crate) fn build(cfg: &Config, parts: Parts<'_>) -> TaskPackage {
    let Parts {
        root,
        format,
        instruction,
        layout,
        raw,
        names,
        unsupported: mut refused,
    } = parts;
    let name = root
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| root.to_string_lossy().into_owned());
    let id = cfg.package.name.clone().unwrap_or_else(|| name.clone());
    let instruction = match &cfg.agent.prompt_prefix {
        Some(prefix) => format!("{prefix}\n\n{instruction}"),
        None => instruction,
    };

    let s = &cfg.sandbox;
    let image = match &s.docker_image {
        Some(reference) => ImageRef::Pull {
            reference: reference.clone(),
        },
        None => ImageRef::Build {
            context: PathBuf::from("environment"),
            dockerfile: PathBuf::from("environment/Dockerfile"),
        },
    };
    if s.os.as_deref() == Some("windows") {
        refused.push(unsupported(
            names.field("sandbox", "os"),
            "Windows containers are not supported yet",
            false,
        ));
    }
    if s.tpu {
        refused.push(unsupported(
            names.field("sandbox", "tpu"),
            "TPU scheduling is not supported yet",
            false,
        ));
    }
    // Harbor's skills_dir is a folder the image fills, copied to the agent at
    // run time. 0.8's means where to mount environment/skills, which this
    // runner does not do.
    let skills_dir = match (&s.skills_dir, names.dialect) {
        (Some(d), Dialect::TaskToml) if d.starts_with('/') => Some(d.clone()),
        (Some(_), Dialect::TaskToml) => {
            refused.push(unsupported(
                names.field("sandbox", "skills_dir"),
                "skills_dir must be an absolute path in the sandbox, such as /skills",
                true,
            ));
            None
        }
        (Some(_), Dialect::Frontmatter) => {
            refused.push(unsupported(
                names.field("sandbox", "skills_dir"),
                "skills_dir names where 0.8 mounted the task's skills; this runner copies environment/skills/ to the agent's own skill folders instead",
                true,
            ));
            None
        }
        (None, _) => None,
    };
    if s.setup_commands > 0 {
        refused.push(unsupported(
            names.field("sandbox", "setup_commands"),
            "setup commands before the agent starts are not supported yet",
            false,
        ));
    }
    if !s.network.blocked_urls.is_empty() {
        refused.push(unsupported(
            names.field("sandbox", "blocked_urls"),
            "blocking URL prefixes is not supported yet; only whole hosts can be blocked",
            false,
        ));
    }
    let ready = s.healthcheck.as_ref().map(|h| Ready {
        run: h.command.clone(),
        interval: secs(h.interval_sec),
        timeout: secs(h.timeout_sec),
        start_period: secs(h.start_period_sec),
        start_interval: secs(h.start_interval_sec),
        retries: u32::try_from(h.retries.max(0)).unwrap_or(u32::MAX),
    });
    let mut outputs = Vec::new();
    for (i, a) in cfg.artifacts.iter().enumerate() {
        let field = format!("{}[{i}]", names.field("", "artifacts"));
        if a.service.as_deref().is_some_and(|svc| svc.trim() != "main") {
            refused.push(unsupported(
                field,
                "collecting outputs from a Compose service is not supported yet",
                false,
            ));
            continue;
        }
        let path = if a.source.starts_with('/') {
            a.source.clone()
        } else if let Some(workdir) = &s.workdir {
            format!(
                "{}/{}",
                workdir.trim_end_matches('/'),
                a.source.trim_start_matches("./")
            )
        } else {
            refused.push(unsupported(
                field,
                "a relative artifact source depends on the image's working folder; make it absolute or set workdir",
                false,
            ));
            continue;
        };
        let save_as = a.destination.clone().unwrap_or_else(|| {
            path.trim_end_matches('/')
                .rsplit('/')
                .next()
                .filter(|n| !n.is_empty())
                .unwrap_or("artifact")
                .to_string()
        });
        outputs.push(Output {
            path,
            save_as,
            max_bytes: None,
            exclude: a.exclude.clone(),
        });
    }
    let skills = root.join("environment").join("skills");
    let mut sandbox = SandboxPolicy {
        image,
        cpus: s.cpus.map(|c| c as f64),
        memory_mb: s.memory_mb.and_then(|m| u64::try_from(m).ok()),
        disk_mb: s.storage_mb.and_then(|m| u64::try_from(m).ok()),
        gpus: s.gpus.and_then(|g| u32::try_from(g).ok()).unwrap_or(0),
        build_timeout: Some(secs(s.build_timeout_sec.unwrap_or(600.0))),
        network: sandbox_network(&s.network),
        workdir: s.workdir.clone(),
        env: s.env.clone(),
        ready,
        outputs,
        // A task that names an in-sandbox folder gets that one: its image
        // copies environment/skills/ there.
        skills: (skills_dir.is_none() && skills.is_dir())
            .then(|| PathBuf::from("environment/skills")),
        skills_dir,
        mcp: s.mcp_servers.clone(),
    };

    let v = &cfg.verifier;
    let separate = v.mode.as_deref() == Some("separate")
        || (v.mode.is_none() && v.separate_environment.is_some());
    if v.kind == "llm-judge" {
        refused.push(unsupported(
            names.field("verifier", "type"),
            "the LLM-judge verifier of BenchFlow 0.8 is not supported yet",
            false,
        ));
    }
    if v.service != "main" {
        refused.push(unsupported(
            names.field("verifier", "service"),
            "running the verifier in a Compose service other than main is not supported yet",
            false,
        ));
    }
    if v.reward_range.is_some() {
        refused.push(unsupported(
            names.field("verifier", "reward_range"),
            "rewards outside [0, 1] are not supported yet",
            false,
        ));
    }
    if !v.cleanup_conftests {
        refused.push(unsupported(
            names.field("verifier.hardening", "cleanup_conftests"),
            "turning off conftest.py cleanup is not supported yet; the runner removes planted conftest.py files before grading",
            false,
        ));
    }
    if v.expected_skills.is_some() {
        refused.push(unsupported(
            names.field("verifier.memory", "expected_skills"),
            "Memory-space scoring is not supported yet",
            false,
        ));
    }
    let verifier = VerifierPolicy {
        dir: PathBuf::from(layout.verifier_dir),
        mount: layout.verifier_mount.to_string(),
        entry: layout.verifier_entry.clone(),
        timeout: Some(secs(v.timeout_sec)),
        user: v.user.clone(),
        env: v.env.clone(),
        network: phase_network(&v.network),
        isolation: if separate {
            Isolation::Separate
        } else {
            Isolation::Shared
        },
        rubric: None,
    };
    let mut extras = super::extras::read(
        &super::extras::Context {
            root,
            raw: &raw,
            names: &names,
            layout: &layout,
            separate,
            docker_image: s.docker_image.as_deref(),
            workdir: s.workdir.as_deref(),
            gpu_types: s.gpu_types.as_deref().unwrap_or_default(),
        },
        &mut refused,
    );
    // The task's own environment wins over its Compose file's.
    for (k, v) in std::mem::take(&mut extras.main_env) {
        sandbox.env.entry(k).or_insert(v);
    }
    // A multi-step task's reference is per step; it has one when every step
    // has its own or the task's.
    let step_oracles = extras.steps.as_ref().is_some_and(|st| {
        !st.stages.is_empty() && st.stages.iter().all(|s| s.oracle_dir.is_some())
    });
    let oracle = layout
        .oracle_dir
        .or(step_oracles.then_some(if layout.oracle_mount == "/oracle" {
            "oracle"
        } else {
            "solution"
        }))
        .map(|dir| OraclePolicy {
            dir: PathBuf::from(dir),
            mount: layout.oracle_mount.to_string(),
            entry: layout.oracle_entry.clone(),
            env: cfg.oracle.env.clone(),
        });
    let instruction = match extras.steps.as_ref().and_then(|st| st.stages.first()) {
        Some(first) if instruction.trim().is_empty() => first.prompt.clone(),
        _ => instruction,
    };
    let agent = AgentPolicy {
        timeout: cfg.agent.timeout_sec.map(secs),
        on_timeout: OnTimeout::Grade,
        user: cfg.agent.user.clone(),
        network: phase_network(&cfg.agent.network),
    };
    TaskPackage {
        root: root.to_path_buf(),
        format,
        id,
        name,
        title: None,
        version: cfg.package.version.clone(),
        instruction,
        stages: Vec::new(),
        agent,
        sandbox,
        verifier,
        oracle,
        controls: Vec::new(),
        unsupported: refused,
        extras,
        config: raw,
        digest: String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_follow_the_file() {
        let toml = Names {
            dialect: Dialect::TaskToml,
            sandbox: "environment",
        };
        assert_eq!(toml.field("sandbox", "tpu"), "[environment] tpu");
        assert_eq!(toml.field("", "steps"), "steps");
        let md = Names {
            dialect: Dialect::Frontmatter,
            sandbox: "sandbox",
        };
        assert_eq!(md.field("sandbox", "tpu"), "sandbox.tpu");
        assert_eq!(
            md.field("verifier.hardening", "cleanup_conftests"),
            "verifier.hardening.cleanup_conftests"
        );
    }

    #[test]
    fn networks_map_onto_the_runtime_kinds() {
        let mut n = SandboxNetwork::default();
        assert_eq!(sandbox_network(&n), Network::Open);
        n.mode = NetworkMode::NoNetwork;
        assert_eq!(sandbox_network(&n), Network::None);
        n.mode = NetworkMode::Allowlist;
        n.allowed_hosts = vec!["pypi.org".into()];
        assert_eq!(sandbox_network(&n), Network::Allow(vec!["pypi.org".into()]));
        n.mode = NetworkMode::Denylist;
        n.blocked_hosts = vec!["x.com".into()];
        assert_eq!(sandbox_network(&n), Network::Block(vec!["x.com".into()]));
        assert_eq!(phase_network(&PhaseNetwork::default()), None);
    }
}
