//! Every key of Harbor's `task.toml` (schema 1.0 to 1.4) and of BenchFlow
//! 0.8's task config, with who reads it.
//!
//! BenchFlow 0.8 took its config model from Harbor and renamed a few tables:
//! `[environment]` is `[sandbox]`, `[solution]` is also `[oracle]`, and
//! `[verifier] environment_mode` is `sandbox_mode`. In `task.toml` 0.8 reads
//! both spellings; in a 0.8 `task.md` frontmatter only its own.

/// What a key holds.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Shape {
    /// A value that is not a table: a string, number, boolean, date, or list.
    Value,
    /// A value that may be a table whose keys are not checked
    /// (`min_reward`, which is a number or a table of numbers).
    Any,
    /// A table whose keys the task chooses (`[metadata]`, `env`).
    Open,
    /// A table with these keys.
    Table(&'static [Field]),
    /// A list of tables with these keys (`[[steps]]`).
    Tables(&'static [Field]),
    /// A list of strings or tables with these keys (`artifacts`).
    StringsOrTables(&'static [Field]),
}

/// One key and who reads it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Field {
    pub name: &'static str,
    pub shape: Shape,
    /// Harbor reads it in `task.toml`.
    pub harbor: bool,
    /// BenchFlow 0.8 reads it in `task.toml`.
    pub bf_toml: bool,
    /// BenchFlow 0.8 reads it in `task.md` frontmatter.
    pub bf_md: bool,
}

/// Which file a config came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Dialect {
    /// `task.toml`: Harbor's keys and BenchFlow 0.8's are both read.
    TaskToml,
    /// A BenchFlow 0.8 `task.md` frontmatter: 0.8's keys only.
    Frontmatter,
}

impl Field {
    /// Whether a config in this dialect reads the key.
    pub(crate) fn read_in(&self, dialect: Dialect) -> bool {
        match dialect {
            Dialect::TaskToml => self.harbor || self.bf_toml,
            Dialect::Frontmatter => self.bf_md,
        }
    }
}

/// Read by Harbor and BenchFlow 0.8 everywhere.
const fn all(name: &'static str, shape: Shape) -> Field {
    Field {
        name,
        shape,
        harbor: true,
        bf_toml: true,
        bf_md: true,
    }
}

/// Harbor's spelling, which BenchFlow 0.8 renames when it reads `task.toml`.
const fn harbor_spelling(name: &'static str, shape: Shape) -> Field {
    Field {
        name,
        shape,
        harbor: true,
        bf_toml: true,
        bf_md: false,
    }
}

/// Read by Harbor only.
const fn harbor(name: &'static str, shape: Shape) -> Field {
    Field {
        name,
        shape,
        harbor: true,
        bf_toml: false,
        bf_md: false,
    }
}

/// Read by BenchFlow 0.8 only.
const fn benchflow(name: &'static str, shape: Shape) -> Field {
    Field {
        name,
        shape,
        harbor: false,
        bf_toml: true,
        bf_md: true,
    }
}

use Shape::{Any, Open, StringsOrTables, Table, Tables, Value};

pub(crate) static AUTHOR: &[Field] = &[all("name", Value), all("email", Value)];

pub(crate) static PACKAGE: &[Field] = &[
    all("name", Value),
    all("version", Value),
    all("description", Value),
    all("authors", Tables(AUTHOR)),
    all("keywords", Value),
];

pub(crate) static AGENT: &[Field] = &[
    all("timeout_sec", Value),
    all("user", Value),
    all("network_mode", Value),
    all("allowed_hosts", Value),
    benchflow("prompt_prefix", Value),
];

pub(crate) static TPU: &[Field] = &[all("type", Value), all("topology", Value)];

pub(crate) static MCP_SERVER: &[Field] = &[
    all("name", Value),
    all("transport", Value),
    all("url", Value),
    all("command", Value),
    all("args", Value),
    benchflow("cwd", Value),
    benchflow("env", Open),
    benchflow("headers", Open),
    benchflow("tools", Value),
    benchflow("include_tags", Value),
    benchflow("exclude_tags", Value),
];

pub(crate) static HEALTHCHECK: &[Field] = &[
    all("command", Value),
    all("interval_sec", Value),
    all("timeout_sec", Value),
    all("start_period_sec", Value),
    all("start_interval_sec", Value),
    all("retries", Value),
];

pub(crate) static SETUP_COMMAND: &[Field] = &[
    benchflow("command", Value),
    benchflow("timeout_sec", Value),
    benchflow("cwd", Value),
    benchflow("env", Open),
    benchflow("user", Value),
    benchflow("service", Value),
    benchflow("host_lock", Value),
    benchflow("capture_dir", Value),
    benchflow("capture_dir_b64_env", Value),
    benchflow("capture_dir_b64_env_file_var", Value),
];

pub(crate) static ENVIRONMENT: &[Field] = &[
    all("network_mode", Value),
    all("allowed_hosts", Value),
    all("build_timeout_sec", Value),
    all("docker_image", Value),
    all("os", Value),
    all("cpus", Value),
    all("memory_mb", Value),
    all("storage_mb", Value),
    all("gpus", Value),
    all("gpu_types", Value),
    all("tpu", Table(TPU)),
    all("mcp_servers", Tables(MCP_SERVER)),
    all("env", Open),
    all("skills_dir", Value),
    all("healthcheck", Table(HEALTHCHECK)),
    all("workdir", Value),
    all("allow_internet", Value),
    all("memory", Value),
    all("storage", Value),
    benchflow("blocked_urls", Value),
    benchflow("blocked_hosts", Value),
    benchflow("image", Value),
    benchflow("setup_commands", Tables(SETUP_COMMAND)),
    benchflow("bugswarm_image_tag", Value),
];

pub(crate) static COLLECT: &[Field] = &[
    harbor("command", Value),
    harbor("service", Value),
    harbor("timeout_sec", Value),
    harbor("user", Value),
];

pub(crate) static JUDGE: &[Field] = &[
    benchflow("model", Value),
    benchflow("rubric_path", Value),
    benchflow("input_dir", Value),
    benchflow("input_type", Value),
    benchflow("context", Value),
];

pub(crate) static MEMORY: &[Field] = &[benchflow("expected_skills", Value)];

pub(crate) static HARDENING: &[Field] = &[benchflow("cleanup_conftests", Value)];

pub(crate) static VERIFIER: &[Field] = &[
    all("timeout_sec", Value),
    all("env", Open),
    all("user", Value),
    all("network_mode", Value),
    all("allowed_hosts", Value),
    harbor_spelling("environment_mode", Value),
    harbor_spelling("environment", Table(ENVIRONMENT)),
    harbor("collect", Tables(COLLECT)),
    benchflow("type", Value),
    benchflow("submission_files", Value),
    benchflow("workspace_recovery", Value),
    benchflow("reward_range", Value),
    benchflow("service", Value),
    benchflow("sandbox_mode", Value),
    benchflow("sandbox", Table(ENVIRONMENT)),
    benchflow("judge", Table(JUDGE)),
    benchflow("memory", Table(MEMORY)),
    benchflow("hardening", Table(HARDENING)),
    benchflow("pytest_plugins", Value),
];

pub(crate) static SOLUTION: &[Field] = &[all("env", Open), benchflow("timeout_sec", Value)];

pub(crate) static ARTIFACT: &[Field] = &[
    all("source", Value),
    all("destination", Value),
    all("exclude", Value),
    harbor("service", Value),
];

pub(crate) static STEP: &[Field] = &[
    all("name", Value),
    all("agent", Table(AGENT)),
    all("verifier", Table(VERIFIER)),
    all("min_reward", Any),
    all("healthcheck", Table(HEALTHCHECK)),
    all("artifacts", StringsOrTables(ARTIFACT)),
];

/// The top level of the config.
pub(crate) static TOP: &[Field] = &[
    all("schema_version", Value),
    all("version", Value),
    all("task", Table(PACKAGE)),
    all("metadata", Open),
    all("verifier", Table(VERIFIER)),
    all("agent", Table(AGENT)),
    harbor_spelling("environment", Table(ENVIRONMENT)),
    benchflow("sandbox", Table(ENVIRONMENT)),
    all("solution", Table(SOLUTION)),
    benchflow("oracle", Table(SOLUTION)),
    all("source", Value),
    all("multi_step_reward_strategy", Value),
    all("steps", Tables(STEP)),
    all("artifacts", StringsOrTables(ARTIFACT)),
    benchflow("reward", Open),
];

/// The field named `name` in `fields`, if the dialect reads it.
pub(crate) fn field(
    fields: &'static [Field],
    name: &str,
    dialect: Dialect,
) -> Option<&'static Field> {
    fields.iter().find(|f| f.name == name && f.read_in(dialect))
}

/// Every table that holds a key named `name` in this dialect, as dotted
/// table paths (`""` for the top level, `steps.agent` for a step's agent).
pub(crate) fn homes(name: &str, dialect: Dialect) -> Vec<String> {
    let mut out = Vec::new();
    collect_homes(TOP, "", name, dialect, &mut out);
    out
}

fn collect_homes(
    fields: &'static [Field],
    at: &str,
    name: &str,
    dialect: Dialect,
    out: &mut Vec<String>,
) {
    for f in fields.iter().filter(|f| f.read_in(dialect)) {
        if f.name == name && !out.iter().any(|h| h == at) {
            out.push(at.to_string());
        }
        let below = if at.is_empty() {
            f.name.to_string()
        } else {
            format!("{at}.{}", f.name)
        };
        match f.shape {
            Shape::Table(sub) | Shape::Tables(sub) | Shape::StringsOrTables(sub) => {
                collect_homes(sub, &below, name, dialect, out)
            }
            Shape::Value | Shape::Any | Shape::Open => {}
        }
    }
}

/// The keys a table holds in this dialect, for "did you mean" and for
/// listing them.
pub(crate) fn names(fields: &'static [Field], dialect: Dialect) -> Vec<&'static str> {
    fields
        .iter()
        .filter(|f| f.read_in(dialect))
        .map(|f| f.name)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dialects_differ_where_benchflow_renamed() {
        assert!(field(TOP, "environment", Dialect::TaskToml).is_some());
        assert!(field(TOP, "environment", Dialect::Frontmatter).is_none());
        assert!(field(TOP, "sandbox", Dialect::Frontmatter).is_some());
        assert!(field(VERIFIER, "collect", Dialect::TaskToml).is_some());
        assert!(field(VERIFIER, "collect", Dialect::Frontmatter).is_none());
        assert!(field(ARTIFACT, "service", Dialect::Frontmatter).is_none());
    }

    #[test]
    fn homes_lists_every_table_with_the_key() {
        assert_eq!(
            homes("multi_step_reward_strategy", Dialect::TaskToml),
            vec![""]
        );
        let h = homes("timeout_sec", Dialect::TaskToml);
        assert!(h.contains(&"agent".to_string()));
        assert!(h.contains(&"verifier".to_string()));
        assert!(h.contains(&"environment.healthcheck".to_string()));
        assert!(homes("nothing_like_this", Dialect::TaskToml).is_empty());
    }
}
