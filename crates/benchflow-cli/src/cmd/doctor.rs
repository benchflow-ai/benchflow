//! `bench doctor`: is this machine ready? One line per check, each with a
//! fix when it is not.

use std::path::Path;

use serde_json::json;

use benchflow_run::Owner;

use crate::cli::DoctorArgs;
use crate::context;
use crate::error::{CliError, Exit, Result};
use crate::ui::{dim, fail_mark, ok_mark, ui, warn_mark};
use crate::{note, say};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Level {
    Ok,
    /// Worth knowing, nothing to fix (agents with no login here).
    Info,
    Warn,
    Fail,
}

/// With a sandbox ready, a backend not set up here is worth knowing, not a
/// warning: a new machine with Docker should not warn that Podman is missing.
fn soften(lines: &mut [Line], missing: &[usize]) {
    for &i in missing {
        if let Some(l) = lines.get_mut(i) {
            l.level = Level::Info;
        }
    }
}

struct Line {
    name: String,
    level: Level,
    summary: String,
    fix: Option<String>,
}

pub async fn run(args: DoctorArgs) -> Result<Exit> {
    if args.confinement {
        return confinement(args).await;
    }
    let mut lines: Vec<Line> = Vec::new();

    // bench.toml
    match context::project() {
        Ok(p) => lines.push(Line {
            name: "bench.toml".into(),
            level: Level::Ok,
            summary: match &p.file {
                Some(f) => crate::ui::path(f),
                None => "none (defaults apply)".into(),
            },
            fix: None,
        }),
        Err(e) => lines.push(Line {
            name: "bench.toml".into(),
            level: Level::Fail,
            summary: e.message.clone(),
            fix: e.next.clone(),
        }),
    }

    // Sandboxes.
    let backends: Vec<String> = if args.sandbox.is_empty() {
        benchflow_sandbox::BACKENDS
            .iter()
            .map(|s| s.to_string())
            .collect()
    } else {
        args.sandbox.clone()
    };
    let mut any_sandbox = false;
    // Backends with nothing set up here: no binary, or no Daytona key.
    let mut missing = Vec::new();
    for b in &backends {
        if b == "daytona" && std::env::var_os("DAYTONA_API_KEY").is_none() {
            missing.push(lines.len());
            lines.push(Line {
                name: b.clone(),
                level: Level::Warn,
                summary: "DAYTONA_API_KEY is not set".into(),
                fix: Some("for cloud sandboxes, create a key at https://app.daytona.io/dashboard/keys and export DAYTONA_API_KEY".into()),
            });
            continue;
        }
        if b != "daytona" && benchflow_run::source::which(b).is_none() {
            missing.push(lines.len());
            lines.push(Line {
                name: b.clone(),
                level: Level::Warn,
                summary: "not installed".into(),
                fix: None,
            });
            continue;
        }
        match benchflow_sandbox::provider(b) {
            Ok(p) => {
                let s = p.status().await;
                if s.ok {
                    any_sandbox = true;
                    let swept = p.sweep().await.unwrap_or(0);
                    let mut summary = s.summary;
                    if swept > 0 {
                        summary.push_str(&format!(" · removed {swept} left-behind sandboxes"));
                    }
                    lines.push(Line {
                        name: b.clone(),
                        level: Level::Ok,
                        summary,
                        fix: None,
                    });
                } else {
                    lines.push(Line {
                        name: b.clone(),
                        level: Level::Fail,
                        summary: s.summary,
                        fix: s.fix,
                    });
                }
            }
            Err(e) => lines.push(Line {
                name: b.clone(),
                level: Level::Fail,
                summary: e.to_string(),
                fix: None,
            }),
        }
    }
    if any_sandbox && args.sandbox.is_empty() {
        soften(&mut lines, &missing);
    }
    if !any_sandbox {
        lines.push(Line {
            name: "sandbox".into(),
            level: Level::Fail,
            summary: "no sandbox backend is ready, so nothing can run".into(),
            fix: Some("start Docker, or set DAYTONA_API_KEY".into()),
        });
    }

    // Agent logins: each agent that has one, then one line for the rest.
    // Most of the 40-odd agents have no login on a given machine, and a
    // warning per agent would bury the checks that matter.
    let logins = crate::cmd::agents::login_lines();
    let without: Vec<String> = logins
        .iter()
        .filter(|(_, ready, _, _)| !ready)
        .map(|(name, _, _, _)| name.trim_start_matches("agent ").to_string())
        .collect();
    let ready = logins.len() - without.len();
    for (name, level, summary, fix) in logins.into_iter().filter(|(_, ready, _, _)| *ready) {
        lines.push(Line {
            name,
            level: if level { Level::Ok } else { Level::Warn },
            summary,
            fix,
        });
    }
    if !without.is_empty() {
        lines.push(Line {
            name: "agents".into(),
            level: if ready == 0 { Level::Warn } else { Level::Info },
            summary: if ready == 0 {
                format!("no agent has a login here ({} agents checked)", without.len())
            } else {
                format!(
                    "{} more have no login here: {}",
                    without.len(),
                    crate::ui::truncate(&without.join(", "), 120)
                )
            },
            fix: (ready == 0).then(|| {
                "`codex login` or `claude setup-token` sets up the most common ones; `bench agents <name>` says what any agent needs"
                    .to_string()
            }),
        });
    }

    // Provider keys, checked with each provider's free request.
    let (providers, without) = crate::cmd::agents::provider_lines().await;
    for (name, verdict, summary, fix) in providers {
        lines.push(Line {
            name,
            level: match verdict {
                Some(true) => Level::Ok,
                Some(false) => Level::Fail,
                None => Level::Warn,
            },
            summary,
            fix,
        });
    }
    if !without.is_empty() {
        lines.push(Line {
            name: "providers".into(),
            level: Level::Info,
            summary: format!("no key here for {}", without.join(", ")),
            fix: None,
        });
    }

    // Tools and disk.
    match tool_version("git", &["--version"]) {
        Some(v) => lines.push(Line {
            name: "git".into(),
            level: Level::Ok,
            summary: v,
            fix: None,
        }),
        None => lines.push(Line {
            name: "git".into(),
            level: Level::Warn,
            summary: "not installed".into(),
            fix: Some("install git to run tasks from GitHub repositories".into()),
        }),
    }
    let cache = benchflow_run::config::default_cache_dir();
    let _ = std::fs::create_dir_all(&cache);
    match free_gb(&cache) {
        Some(gb) if gb < 5.0 => lines.push(Line {
            name: "disk".into(),
            level: Level::Warn,
            summary: format!("{gb:.1} GB free in {}", crate::ui::path(&cache)),
            fix: Some("image builds and job folders need room; free some space".into()),
        }),
        Some(gb) => lines.push(Line {
            name: "disk".into(),
            level: Level::Ok,
            summary: format!("{gb:.0} GB free in {}", crate::ui::path(&cache)),
            fix: None,
        }),
        None => {}
    }

    if ui().json {
        let v: Vec<_> = lines
            .iter()
            .map(|l| json!({"check": l.name, "status": match l.level { Level::Ok => "ok", Level::Info => "info", Level::Warn => "warn", Level::Fail => "fail" }, "summary": l.summary, "fix": l.fix}))
            .collect();
        println!("{}", serde_json::to_string_pretty(&v).unwrap_or_default());
    } else {
        let w = lines.iter().map(|l| l.name.len()).max().unwrap_or(8);
        let width = crate::ui::width();
        for l in &lines {
            let mark = match l.level {
                Level::Ok => ok_mark(),
                Level::Info => dim("·").to_string(),
                Level::Warn => warn_mark(),
                Level::Fail => fail_mark(),
            };
            // Long lines wrap under their own column.
            let first = format!("  {mark} {:<w$}  ", l.name);
            say!("{}", crate::ui::wrap(&first, &l.summary, w + 6, width));
            if let Some(f) = &l.fix {
                let first = format!("    {:<w$}  {} ", "", dim("fix:"));
                say!("{}", crate::ui::wrap(&first, f, w + 11, width));
            }
        }
    }
    Ok(if any_sandbox {
        Exit::Ok
    } else {
        Exit::NotReady
    })
}

/// `bench doctor --confinement`: one sandbox, confined as a trial is, each
/// setting read back.
async fn confinement(args: DoctorArgs) -> Result<Exit> {
    use benchflow_run::confinement::{self, State};
    if args.sandbox.len() > 1 {
        return Err(
            CliError::usage("--confinement checks one backend at a time")
                .next("pass one -s: docker, podman, or daytona"),
        );
    }
    let project = context::project()?;
    let requested = args
        .sandbox
        .first()
        .cloned()
        .or_else(|| project.run_defaults(None).ok().and_then(|d| d.sandbox));
    let provider = context::provider(requested.as_deref()).await?;
    let runtime = args.runtime.clone().or_else(|| {
        project
            .run_defaults(None)
            .ok()
            .and_then(|d| d.runtime)
            .filter(|r| !r.is_empty())
    });
    let opts = confinement::Options {
        image: args.image.clone(),
        runtime,
        ..confinement::Options::default()
    };
    note!(
        "{} one {} sandbox from {}{} and reading its settings back",
        crate::ui::bold("Confining"),
        provider.name(),
        opts.image,
        opts.runtime
            .as_ref()
            .map(|r| format!(" under {r}"))
            .unwrap_or_default()
    );
    let report = confinement::check(provider.as_ref(), &opts)
        .await
        .map_err(|f| {
            CliError::not_ready(f.message)
                .owner(Owner::You)
                .next(f.next)
        })?;
    if ui().json {
        println!(
            "{}",
            serde_json::to_string_pretty(&report).unwrap_or_default()
        );
    } else {
        let w = report
            .settings
            .iter()
            .map(|s| s.name.len())
            .max()
            .unwrap_or(8);
        for s in &report.settings {
            let mark = match s.state {
                State::Enforced => ok_mark(),
                State::NotSupported => warn_mark(),
                State::NotSet => fail_mark(),
            };
            say!(
                "  {mark} {:<w$}  {:<13}  {}",
                s.name,
                s.state.label(),
                s.summary
            );
            say!("    {:<w$}  {}", "", dim(&s.evidence));
        }
        say!(
            "  {}",
            dim(format!(
                "{} on {} · sandbox {} (destroyed)",
                report.image, report.backend, report.sandbox
            ))
        );
    }
    Ok(Exit::Ok)
}

fn tool_version(cmd: &str, args: &[&str]) -> Option<String> {
    let out = std::process::Command::new(cmd).args(args).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// Free space on the filesystem holding `path`, from `df -Pk`.
fn free_gb(path: &Path) -> Option<f64> {
    let out = std::process::Command::new("df")
        .arg("-Pk")
        .arg(path)
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let line = text.lines().nth(1)?;
    let avail_kb: f64 = line.split_whitespace().nth(3)?.parse().ok()?;
    Some(avail_kb / 1024.0 / 1024.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(name: &str, level: Level) -> Line {
        Line {
            name: name.into(),
            level,
            summary: String::new(),
            fix: None,
        }
    }

    #[test]
    fn a_backend_not_set_up_is_only_noted_beside_a_ready_one() {
        let mut lines = vec![
            line("docker", Level::Ok),
            line("podman", Level::Warn),
            line("daytona", Level::Warn),
            line("git", Level::Warn),
        ];
        soften(&mut lines, &[1, 2]);
        let levels: Vec<Level> = lines.iter().map(|l| l.level).collect();
        assert!(
            levels == [Level::Ok, Level::Info, Level::Info, Level::Warn],
            "only the missing backends are softened"
        );
    }
}
