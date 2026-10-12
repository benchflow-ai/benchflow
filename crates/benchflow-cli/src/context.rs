//! What commands share: the project, the tasks it points at, the sandbox
//! backend to use, and the agent to run.

use std::path::PathBuf;
use std::sync::Arc;

use benchflow_run::project::Project;
use benchflow_sandbox::Provider;
use benchflow_task::fnmatch::name_matches;
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
/// A cloned repository brings its `.env`, so only credentials and a few
/// settings are taken from it ([`dotenv_allows`]); anything else (where keys
/// are sent, proxies, certificates, git's settings, the loader) is ignored
/// and named in a note, and must come from this shell's environment.
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
    // A .env that git tracks came with the clone: it is the repository's,
    // not the user's, so nothing in it is taken.
    if tracked_by_git(&root, ".env") {
        eprintln!(
            "  ! ignoring {}: git tracks it, so it came with the repository; set keys in this \
             shell or in ~/.config/benchflow",
            root.join(".env").display()
        );
        return;
    }
    let mut ignored = Vec::new();
    let mut taken = Vec::new();
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
        if !dotenv_allows(k) {
            ignored.push(k.to_string());
            continue;
        }
        if std::env::var_os(k).is_none() {
            taken.push(k.to_string());
            // SAFETY: called from main before the async runtime or any other
            // thread exists, so nothing reads the environment concurrently.
            #[allow(unsafe_code)]
            unsafe {
                std::env::set_var(k, v)
            }
        }
    }
    if !taken.is_empty() {
        taken.sort();
        eprintln!(
            "  note: took {} from {}",
            taken.join(", "),
            root.join(".env").display()
        );
    }
    if !ignored.is_empty() {
        ignored.sort();
        ignored.dedup();
        eprintln!(
            "  ! ignoring {} in {}: a project's .env may set only keys and tokens (names ending \
             in _API_KEY, _KEY, or _TOKEN), DAYTONA_TARGET, and BENCHFLOW_OWNER; set others in \
             this shell",
            ignored.join(", "),
            root.join(".env").display()
        );
    }
}

/// Whether git tracks `file` in the repository at `root` (a clone brings
/// only tracked files). False when git is missing or `root` is no repository.
fn tracked_by_git(root: &std::path::Path, file: &str) -> bool {
    std::process::Command::new("git")
        .args(["ls-files", "--error-unmatch", "--", file])
        .current_dir(root)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

/// Whether a project's `.env` may set `name`: credentials (`*_API_KEY`,
/// `*_KEY`, `*_TOKEN`) and the settings that move nothing elsewhere.
/// Base URLs, proxies, certificates, `GIT_*`, `DOCKER_*`, `PATH`, and the
/// loader's variables are left to this shell.
pub fn dotenv_allows(name: &str) -> bool {
    const SETTINGS: &[&str] = &["DAYTONA_TARGET", "BENCHFLOW_OWNER"];
    // Keys that decide whose machines get the run (and the keys and logins
    // it carries): only from this shell or ~/.config/benchflow.
    const NOT_FROM_A_PROJECT: &[&str] = &["DAYTONA_API_KEY"];
    let upper = name.to_ascii_uppercase();
    if NOT_FROM_A_PROJECT.contains(&upper.as_str()) {
        return false;
    }
    if SETTINGS.contains(&upper.as_str()) {
        return true;
    }
    let credential = ["_API_KEY", "_KEY", "_TOKEN"]
        .iter()
        .any(|suffix| upper.ends_with(suffix));
    // `GIT_SSH_KEY`-style names are git's settings, not credentials bench reads.
    credential && !upper.starts_with("GIT_") && !upper.starts_with("SSL_")
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
    check_each(paths, |p| check_one(p, opts))
}

/// `check` on every path in parallel, keeping their order. Every path gets a
/// report: when a thread stops (a crash that `check` did not catch), the
/// task it was on and the ones it had not reached get P005.
fn check_each(
    paths: &[PathBuf],
    check: impl Fn(&std::path::Path) -> CheckReport + Sync,
) -> Vec<CheckReport> {
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
    let out: std::sync::Mutex<Vec<Option<CheckReport>>> =
        std::sync::Mutex::new((0..n).map(|_| None).collect());
    let mut stopped: Vec<(usize, String)> = Vec::new();
    std::thread::scope(|s| {
        let handles: Vec<_> = paths
            .chunks(chunk)
            .enumerate()
            .map(|(i, c)| {
                let (out, check) = (&out, &check);
                s.spawn(move || {
                    for (j, p) in c.iter().enumerate() {
                        let report = check(p);
                        out.lock().unwrap_or_else(|e| e.into_inner())[i * chunk + j] = Some(report);
                    }
                })
            })
            .collect();
        for (i, h) in handles.into_iter().enumerate() {
            if let Err(e) = h.join() {
                stopped.push((i, panic_text(e.as_ref())));
            }
        }
    });
    let mut out = out.into_inner().unwrap_or_else(|e| e.into_inner());
    for (i, what) in stopped {
        let end = ((i + 1) * chunk).min(n);
        let mut first = true;
        for k in i * chunk..end {
            if out[k].is_none() {
                let message = if first {
                    format!("checking this task crashed BenchFlow: {what}")
                } else {
                    "not checked: BenchFlow crashed on an earlier task".to_string()
                };
                first = false;
                out[k] = Some(crashed(&paths[k], message));
            }
        }
    }
    out.into_iter().flatten().collect()
}

/// Check one package. A crash while checking is a P005 error on that
/// package; before, the thread's tasks vanished and `bench check` said
/// "0 tasks: 0 ok".
fn check_one(path: &std::path::Path, opts: &CheckOptions) -> CheckReport {
    guarded(path, || benchflow_task::check(path, opts))
}

/// `check()`, with a crash turned into a P005 error on `path`.
fn guarded(path: &std::path::Path, check: impl FnOnce() -> CheckReport) -> CheckReport {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(check)).unwrap_or_else(|e| {
        crashed(
            path,
            format!(
                "checking this task crashed BenchFlow: {}",
                panic_text(e.as_ref())
            ),
        )
    })
}

fn panic_text(e: &(dyn std::any::Any + Send)) -> String {
    e.downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| e.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "no message".into())
}

/// A P005 report on `path`. Its format is detected under a guard too, so a
/// crash there cannot lose the task either.
fn crashed(path: &std::path::Path, message: String) -> CheckReport {
    let format = std::panic::catch_unwind(|| benchflow_task::detect(path)).unwrap_or(None);
    let mut report = CheckReport::new(path, format);
    report.diagnostics.push(
        benchflow_task::Diagnostic::error("P005", message)
            .help("this is a BenchFlow bug; report it with the task folder"),
    );
    report
}

/// The error for task roots that hold no task. Terminal-Bench v1 tasks
/// (`task.yaml`) are found but not run as they are, so when the roots hold
/// some, the error says so and gives the `bench import` that converts them.
pub fn no_tasks(roots: &[PathBuf]) -> CliError {
    let shown: Vec<String> = roots.iter().map(|r| crate::ui::path(r)).collect();
    let err = CliError::usage(format!("no tasks found in {}", shown.join(", ")));
    let tbench: Vec<&PathBuf> = roots
        .iter()
        .filter(|r| !benchflow_task::convert::discover_tbench(r).is_empty())
        .collect();
    match tbench.first() {
        Some(root) => {
            let n: usize = tbench
                .iter()
                .map(|r| benchflow_task::convert::discover_tbench(r).len())
                .sum();
            let what = if n == 1 {
                "it holds a Terminal-Bench v1 task (task.yaml)".to_string()
            } else {
                format!("it holds {n} Terminal-Bench v1 tasks (task.yaml)")
            };
            err.next(format!(
                "{what}, which runs once converted to task.md: `bench import {}`",
                crate::ui::path(root),
            ))
        }
        None => err
            .next("a task is a folder with task.md (or task.toml); `bench new <name>` creates one"),
    }
}

/// Load tasks for running: discover, check, and keep the ones that load.
/// Prints the problems of the ones that do not, and fails when none load.
pub fn load_tasks(roots: &[PathBuf]) -> Result<(Vec<Arc<TaskPackage>>, Vec<CheckReport>)> {
    load_tasks_noting(roots).map(|(tasks, bad, _)| (tasks, bad))
}

/// A warning the check of a loadable task found, with the task's name: a
/// setting the reader ignored or dropped, which a run should show.
pub type TaskWarning = (String, benchflow_task::Diagnostic);

/// The tasks that load, the reports of those that do not, and the loaded
/// tasks' warnings.
pub type Loaded = (Vec<Arc<TaskPackage>>, Vec<CheckReport>, Vec<TaskWarning>);

/// As [`load_tasks`], and also the warnings of the tasks that load, so a
/// run can say what it will not apply (`allow_internet` in the wrong
/// table, a Compose value taken from the host) as `bench check` does.
pub fn load_tasks_noting(roots: &[PathBuf]) -> Result<Loaded> {
    let paths = discover(roots)?;
    if paths.is_empty() {
        return Err(no_tasks(roots));
    }
    let reports = check_all(&paths, &CheckOptions::default());
    let mut tasks = Vec::new();
    let mut bad = Vec::new();
    let mut warnings = Vec::new();
    for r in reports {
        match (&r.package, r.ok()) {
            (Some(p), true) => {
                warnings.extend(
                    r.diagnostics
                        .iter()
                        .filter(|d| d.severity == benchflow_task::Severity::Warning)
                        .map(|d| (p.name.clone(), d.clone())),
                );
                tasks.push(Arc::new(p.clone()));
            }
            _ => bad.push(r),
        }
    }
    Ok((tasks, bad, warnings))
}

/// Filter tasks by name globs and a count limit.
pub fn select(
    tasks: Vec<Arc<TaskPackage>>,
    include: &[String],
    exclude: &[String],
    limit: Option<usize>,
) -> Result<Vec<Arc<TaskPackage>>> {
    // 0.8's matching (exact names, else fnmatch), so no pattern is invalid.
    let hit = |pats: &[String], t: &TaskPackage| {
        pats.iter()
            .any(|p| name_matches(&t.name, p) || name_matches(&t.id, p))
    };
    let mut out: Vec<Arc<TaskPackage>> = tasks
        .into_iter()
        .filter(|t| include.is_empty() || hit(include, t))
        .filter(|t| !hit(exclude, t))
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
        if !n.is_finite() || n < 0.0 {
            return Err(CliError::usage(format!(
                "`{text}` is not a duration; write a number of seconds of 0 or more, or a time like 90s, 10m, or 1h30m"
            )));
        }
        return Ok(benchflow_task::values::duration_from_secs(n));
    }
    benchflow_task::values::parse_duration(text).ok_or_else(|| {
        CliError::usage(format!(
            "`{text}` is not a duration; write it like 90s, 10m, or 1h30m"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Guards the D5 fix: a task whose check panicked (timeout_sec = 1e20,
    /// before that was fixed) disappeared, and `bench check` said
    /// "0 tasks: 0 ok" and exited 0.
    #[test]
    fn a_crash_while_checking_is_an_error_on_the_task() {
        let r = guarded(std::path::Path::new("/nonexistent/t"), || panic!("boom"));
        assert_eq!(r.root, std::path::Path::new("/nonexistent/t"));
        assert!(
            r.diagnostics
                .iter()
                .any(|d| d.code == "P005" && d.is_error() && d.message.contains("boom"))
        );
    }

    /// A crash the per-task guard does not catch (in the guard itself)
    /// stopped the thread, and its share of the tasks vanished.
    #[test]
    fn a_stopped_thread_loses_no_task() {
        let paths: Vec<PathBuf> = (0..40)
            .map(|i| PathBuf::from(format!("/nonexistent/t{i}")))
            .collect();
        let reports = check_each(&paths, |p| {
            if p.ends_with("t5") {
                panic!("boom in {}", p.display());
            }
            CheckReport::new(p, None)
        });
        let roots: Vec<&std::path::Path> = reports.iter().map(|r| r.root.as_path()).collect();
        assert_eq!(
            roots,
            paths.iter().map(PathBuf::as_path).collect::<Vec<_>>()
        );
        let r = reports.iter().find(|r| r.root.ends_with("t5")).unwrap();
        assert!(r.diagnostics.iter().any(|d| d.code == "P005"
            && d.is_error()
            && d.message.contains("boom in /nonexistent/t5")));
        let ok = reports.iter().filter(|r| r.diagnostics.is_empty()).count();
        let skipped = reports
            .iter()
            .filter(|r| {
                r.diagnostics
                    .iter()
                    .any(|d| d.message.starts_with("not checked"))
            })
            .count();
        assert_eq!(ok + skipped + 1, 40);
    }

    /// Guards the D5 fix: `--idle-timeout -1`, `nan`, `inf` or `1e30` (also
    /// from bench.toml or a fetched job file) panicked in
    /// `Duration::from_secs_f64`.
    #[test]
    fn odd_numbers_of_seconds_are_refused_or_capped() {
        assert_eq!(
            parse_duration("90").unwrap(),
            std::time::Duration::from_secs(90)
        );
        assert_eq!(
            parse_duration("1h30m").unwrap(),
            std::time::Duration::from_secs(5400)
        );
        for bad in ["-1", "nan", "NaN", "inf", "-inf"] {
            assert!(parse_duration(bad).is_err(), "{bad} should be refused");
        }
        assert_eq!(
            parse_duration("1e30").unwrap(),
            benchflow_task::values::LONGEST
        );
    }
}
