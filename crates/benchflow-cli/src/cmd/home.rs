//! `bench` with no command: what is in this folder, and what to do next.
//! Fast by design: it discovers tasks and reads the last job's summary, but
//! checks nothing and starts nothing.

use std::path::PathBuf;

use crate::context;
use crate::error::{Exit, Result};
use crate::say;
use crate::ui::{self, bold, dim, plural, ui};

pub fn run() -> Result<Exit> {
    let project = context::project()?;
    let roots = context::task_roots(&project, &[]);
    let tasks: Vec<PathBuf> = roots
        .iter()
        .filter(|r| r.exists())
        .flat_map(|r| benchflow_task::discover(r))
        .collect();
    let jobs_dir = project
        .config
        .run
        .jobs_dir
        .clone()
        .map(|d| project.root.join(d))
        .unwrap_or_else(|| PathBuf::from("jobs"));
    let last = if jobs_dir.exists() {
        benchflow_run::job::latest_job(&jobs_dir)
    } else {
        None
    };
    if ui().json {
        println!(
            "{}",
            serde_json::json!({
                "version": env!("CARGO_PKG_VERSION"),
                "project": project.file,
                "tasks": tasks,
                "last_job": last,
            })
        );
        return Ok(Exit::Ok);
    }
    say!(
        "{} {} {}",
        bold("BenchFlow"),
        env!("CARGO_PKG_VERSION"),
        dim("· run AI agents on tasks in sandboxes, and score them")
    );
    say!("");
    // Here.
    let mut here = Vec::new();
    if !tasks.is_empty() {
        let mut formats: Vec<&str> = tasks
            .iter()
            .filter_map(|t| benchflow_task::detect(t))
            .map(|f| match f {
                benchflow_task::Format::TaskMd => "task.md",
                benchflow_task::Format::TaskMdV06 => "0.8 task.md",
                benchflow_task::Format::Split => "task.toml",
            })
            .collect();
        formats.sort_unstable();
        formats.dedup();
        let where_: Vec<String> = roots.iter().map(|r| ui::path(r)).collect();
        here.push(format!(
            "{} ({}) in {}",
            plural(tasks.len(), "task"),
            formats.join(", "),
            where_.join(", ")
        ));
    } else {
        here.push("no tasks here yet".to_string());
    }
    if let Some(f) = &project.file {
        let r = &project.config.run;
        let mut bits = Vec::new();
        if let Some(a) = &r.agent {
            bits.push(format!("agent {a}"));
        }
        if let Some(s) = &r.sandbox {
            bits.push(format!("sandbox {s}"));
        }
        here.push(format!(
            "{}{}",
            ui::path(f),
            if bits.is_empty() {
                String::new()
            } else {
                format!(": {}", bits.join(", "))
            }
        ));
    }
    say!("  {}  {}", dim("Here    "), here.join(" · "));
    if let Some(job) = &last {
        let view = benchflow_run::read::read_job(job);
        let total = view.records.len();
        let passed = view
            .records
            .iter()
            .filter(|r| r.passed == Some(true))
            .count();
        say!(
            "  {}  {} · {} · {}/{} passed",
            dim("Last job"),
            ui::path(job),
            view.agent(),
            passed,
            total
        );
    }
    say!("");
    let rows: Vec<(String, &str)> = if tasks.is_empty() {
        vec![
            (
                "bench new my-task".into(),
                "create a task that already passes `bench test`",
            ),
            (
                "bench import <folder>".into(),
                "convert task.toml (split layout) tasks to task.md",
            ),
            (
                "bench run org/repo/tasks".into(),
                "run tasks straight from a GitHub repository",
            ),
            (
                "bench doctor".into(),
                "check this machine: sandboxes, agent logins, disk",
            ),
        ]
    } else {
        vec![
            ("bench check".into(), "validate every task here (instant)"),
            (
                "bench test".into(),
                "prove each task: the oracle passes, doing nothing fails, exploits are refused",
            ),
            ("bench run -a claude".into(), "run an agent on every task"),
            (
                if last.is_some() {
                    "bench view".into()
                } else {
                    "bench new my-task".into()
                },
                if last.is_some() {
                    "open the last job in your browser"
                } else {
                    "create another task"
                },
            ),
        ]
    };
    let w = rows.iter().map(|r| r.0.len()).max().unwrap_or(10);
    for (cmd, what) in rows {
        say!("  {}  {}", bold(format!("{cmd:<w$}")), dim(what));
    }
    say!("");
    say!(
        "  {}",
        dim("bench doctor checks this machine · bench --help lists every command")
    );
    Ok(Exit::Ok)
}
