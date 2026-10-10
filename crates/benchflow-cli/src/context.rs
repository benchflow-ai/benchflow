//! What commands share: the project, the tasks it points at, the sandbox
//! backend to use, and the agent to run.

use std::path::PathBuf;
use std::sync::Arc;

use benchflow_run::project::Project;
use benchflow_sandbox::Provider;
use benchflow_task::{CheckOptions, CheckReport, TaskPackage};

use crate::error::{CliError, Exit, Result};

/// The project for the current folder (bench.toml, if any).
pub fn project() -> Result<Project> {
    let cwd =
        std::env::current_dir().map_err(|e| CliError::bug("reading the current folder", e))?;
    Project::discover(&cwd).map_err(|e| {
        CliError::usage(format!("bench.toml is invalid: {}", e.message)).next(format!(
            "fix {}; docs/rust/cli.md lists the keys bench.toml takes",
            crate::ui::path(&e.path)
        ))
    })
}

/// Read `KEY=VALUE` lines from the project's `.env` file (next to bench.toml,
/// or in the current folder) into the environment, never overriding a
/// variable that is already set.
///
/// Must run before any other thread starts: it sets environment variables.
pub fn load_dotenv_before_threads() {
    let Ok(cwd) = std::env::current_dir() else {
        return;
    };
    let root = Project::discover(&cwd).map(|p| p.root).unwrap_or(cwd);
    let Ok(text) = std::fs::read_to_string(root.join(".env")) else {
        return;
    };
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line = line.strip_prefix("export ").unwrap_or(line);
        let Some((k, v)) = line.split_once('=') else {
            continue;
        };
        let k = k.trim();
        if k.is_empty() || !k.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            continue;
        }
        let v = v.trim().trim_matches('"').trim_matches('\'');
        if std::env::var_os(k).is_none() {
            // SAFETY: called from main before the async runtime or any other
            // thread exists, so nothing reads the environment concurrently.
            #[allow(unsafe_code)]
            unsafe {
                std::env::set_var(k, v)
            }
        }
    }
}

/// The folders a command should look for tasks in: the arguments, else
/// bench.toml's `[tasks] paths`, else the current folder.
pub fn task_roots(project: &Project, args: &[PathBuf]) -> Vec<PathBuf> {
    if !args.is_empty() {
        return args.to_vec();
    }
    let from_project = project.task_paths();
    if !from_project.is_empty() {
        return from_project;
    }
    vec![PathBuf::from(".")]
}

/// Every package under the roots, deduplicated, in order.
pub fn discover(roots: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut out: Vec<PathBuf> = Vec::new();
    for r in roots {
        if !r.exists() {
            return Err(CliError::usage(format!("{} does not exist", r.display())));
        }
        for p in benchflow_task::discover(r) {
            if !out.contains(&p) {
                out.push(p);
            }
        }
    }
    Ok(out)
}

/// Check packages in parallel, keeping their order.
pub fn check_all(paths: &[PathBuf], opts: &CheckOptions) -> Vec<CheckReport> {
    let n = paths.len();
    if n == 0 {
        return Vec::new();
    }
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(n)
        .max(1);
    let chunk = n.div_ceil(threads);
    let mut out: Vec<Option<CheckReport>> = (0..n).map(|_| None).collect();
    std::thread::scope(|s| {
        let handles: Vec<_> = paths
            .chunks(chunk)
            .enumerate()
            .map(|(i, c)| {
                s.spawn(move || {
                    c.iter()
                        .enumerate()
                        .map(|(j, p)| (i * chunk + j, benchflow_task::check(p, opts)))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        for h in handles {
            if let Ok(v) = h.join() {
                for (i, r) in v {
                    out[i] = Some(r);
                }
            }
        }
    });
    out.into_iter().flatten().collect()
}

/// Load tasks for running: discover, check, and keep the ones that load.
/// Prints the problems of the ones that do not, and fails when none load.
pub fn load_tasks(roots: &[PathBuf]) -> Result<(Vec<Arc<TaskPackage>>, Vec<CheckReport>)> {
    let paths = discover(roots)?;
    if paths.is_empty() {
        let shown: Vec<String> = roots.iter().map(|r| crate::ui::path(r)).collect();
        return Err(
            CliError::usage(format!("no tasks found in {}", shown.join(", "))).next(
                "a task is a folder with task.md (or task.toml); `bench new <name>` creates one",
            ),
        );
    }
    let reports = check_all(&paths, &CheckOptions::default());
    let mut tasks = Vec::new();
    let mut bad = Vec::new();
    for r in reports {
        match (&r.package, r.ok()) {
            (Some(p), true) => tasks.push(Arc::new(p.clone())),
            _ => bad.push(r),
        }
    }
    Ok((tasks, bad))
}

/// Filter tasks by name globs and a count limit.
pub fn select(
    tasks: Vec<Arc<TaskPackage>>,
    include: &[String],
    exclude: &[String],
    limit: Option<usize>,
) -> Result<Vec<Arc<TaskPackage>>> {
    let compile = |pats: &[String]| -> Result<Option<globset::GlobSet>> {
        if pats.is_empty() {
            return Ok(None);
        }
        let mut b = globset::GlobSetBuilder::new();
        for p in pats {
            b.add(
                globset::Glob::new(p)
                    .map_err(|e| CliError::usage(format!("bad pattern `{p}`: {e}")))?,
            );
        }
        Ok(Some(b.build().map_err(|e| CliError::usage(e.to_string()))?))
    };
    let inc = compile(include)?;
    let exc = compile(exclude)?;
    let mut out: Vec<Arc<TaskPackage>> = tasks
        .into_iter()
        .filter(|t| {
            inc.as_ref()
                .is_none_or(|g| g.is_match(&t.name) || g.is_match(&t.id))
        })
        .filter(|t| {
            exc.as_ref()
                .is_none_or(|g| !(g.is_match(&t.name) || g.is_match(&t.id)))
        })
        .collect();
    out.sort_by(|a, b| a.name.cmp(&b.name));
    if let Some(n) = limit {
        out.truncate(n);
    }
    Ok(out)
}

/// The sandbox backend: the flag, else bench.toml, else the first backend
/// that is ready here (Docker, then Podman, then Daytona when a key is set).
pub async fn provider(requested: Option<&str>) -> Result<Arc<dyn Provider>> {
    if let Some(name) = requested {
        let p = benchflow_sandbox::provider(name).map_err(|e| {
            CliError::usage(e.to_string()).next(format!(
                "use one of: {}",
                benchflow_sandbox::BACKENDS.join(", ")
            ))
        })?;
        let status = p.status().await;
        if !status.ok {
            let mut err = CliError::not_ready(format!(
                "the {name} sandbox is not ready: {}",
                status.summary
            ));
            if let Some(fix) = status.fix {
                err = err.next(fix);
            }
            return Err(err);
        }
        return Ok(Arc::from(p));
    }
    let mut tried = Vec::new();
    for name in ["docker", "podman", "daytona"] {
        if name == "daytona" && std::env::var_os("DAYTONA_API_KEY").is_none() {
            tried.push("daytona (DAYTONA_API_KEY is not set)".to_string());
            continue;
        }
        if name != "daytona" && benchflow_run::source::which(name).is_none() {
            tried.push(format!("{name} (not installed)"));
            continue;
        }
        if let Ok(p) = benchflow_sandbox::provider(name) {
            let s = p.status().await;
            if s.ok {
                return Ok(Arc::from(p));
            }
            tried.push(format!("{name} ({})", s.summary));
        }
    }
    Err(CliError::new(Exit::NotReady, "no sandbox backend is ready on this machine")
        .owner(benchflow_run::Owner::You)
        .detail(format!("tried: {}", tried.join("; ")))
        .next("start Docker, or set DAYTONA_API_KEY to use Daytona's cloud sandboxes (https://app.daytona.io/dashboard/keys); `bench doctor` checks both"))
}

/// Parse `KEY=VALUE` pairs.
pub fn parse_env(pairs: &[String]) -> Result<std::collections::BTreeMap<String, String>> {
    let mut m = std::collections::BTreeMap::new();
    for p in pairs {
        let Some((k, v)) = p.split_once('=') else {
            return Err(CliError::usage(format!("`{p}` is not KEY=VALUE")));
        };
        m.insert(k.to_string(), v.to_string());
    }
    Ok(m)
}

/// Parse `--ak name=value` pairs, in order.
pub fn parse_agent_options(pairs: &[String]) -> Result<Vec<(String, String)>> {
    pairs
        .iter()
        .map(|p| {
            benchflow_agent::manifest::options::parse_assignment(p)
                .map_err(|e| CliError::usage(e.message).next(e.next))
        })
        .collect()
}

/// A job file's or bench.toml's `agent-options` as `--ak` pairs: text as
/// written, other values as JSON (`30`, `true`).
pub fn agent_option_pairs(
    options: &std::collections::BTreeMap<String, serde_json::Value>,
) -> Vec<(String, String)> {
    options
        .iter()
        .map(|(k, v)| {
            let text = match v {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            (k.replace('-', "_"), text)
        })
        .collect()
}

/// Parse a duration like `10m`, `90s`, `1h30m`, or a number of seconds.
pub fn parse_duration(text: &str) -> Result<std::time::Duration> {
    if let Ok(n) = text.parse::<f64>() {
        return Ok(std::time::Duration::from_secs_f64(n));
    }
    benchflow_task::values::parse_duration(text).ok_or_else(|| {
        CliError::usage(format!(
            "`{text}` is not a duration; write it like 90s, 10m, or 1h30m"
        ))
    })
}
