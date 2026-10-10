//! `bench new`: a task that already passes `bench test`.

use std::path::{Path, PathBuf};

use serde_json::json;

use crate::cli::NewArgs;
use crate::error::{CliError, Exit, Result};
use crate::say;
use crate::ui::{self, bold, dim, ok_mark, table::Table, ui};

pub fn run(args: NewArgs) -> Result<Exit> {
    let templates = benchflow_task::scaffold::templates();
    if args.list {
        if ui().json {
            let v: Vec<_> = templates
                .iter()
                .map(|t| json!({"name": t.name, "description": t.description}))
                .collect();
            println!("{}", serde_json::Value::Array(v));
            return Ok(Exit::Ok);
        }
        let mut t = Table::new(&["template", "what you get"]);
        for tpl in templates {
            t.row(vec![tpl.name.to_string(), tpl.description.to_string()]);
        }
        say!("{}", t.render(2));
        return Ok(Exit::Ok);
    }
    let name = args.name.clone().unwrap_or_default();
    if !templates.is_empty() && !templates.iter().any(|t| t.name == args.template) {
        let names: Vec<&str> = templates.iter().map(|t| t.name).collect();
        return Err(CliError::usage(format!("no template `{}`", args.template))
            .next(format!("templates: {}", names.join(", "))));
    }
    // `tasks/my-task` where tasks/ is a folder (or `./x`, `../x`, `/x`) is
    // a path; otherwise `org/my-task` is a name with its organization.
    let as_path = name.starts_with("./")
        || name.starts_with("../")
        || name.starts_with('/')
        || (name.contains('/')
            && Path::new(&name)
                .parent()
                .is_some_and(|p| args.dir.clone().unwrap_or_default().join(p).is_dir()));
    let short = name.rsplit('/').next().unwrap_or(&name).to_string();
    let (base, task_name) = if as_path {
        let parent = Path::new(&name).parent().unwrap_or(Path::new(""));
        let base = match &args.dir {
            Some(d) => d.join(parent),
            None => parent.to_path_buf(),
        };
        (base, short.clone())
    } else {
        let base = match &args.dir {
            Some(d) => d.clone(),
            None => {
                let tasks = PathBuf::from("tasks");
                if tasks.is_dir() {
                    tasks
                } else {
                    PathBuf::from(".")
                }
            }
        };
        (base, name.clone())
    };
    let dir = base.join(&short);
    if dir.exists() {
        return Err(
            CliError::usage(format!("{} already exists", ui::path(&dir)))
                .next("pick another name, or pass --dir"),
        );
    }
    let files = benchflow_task::scaffold::render(&args.template, &task_name, &dir)
        .map_err(|e| CliError::usage(e.to_string()))?;
    if ui().json {
        println!("{}", json!({"task": dir, "files": files}));
        return Ok(Exit::Ok);
    }
    say!(
        "  {} created {} {}",
        ok_mark(),
        bold(ui::path(&dir)),
        dim(format!("({} template)", args.template))
    );
    for f in &files {
        let rel = f.strip_prefix(&dir).unwrap_or(f);
        say!("      {}", dim(rel.display()));
    }
    let p = ui::path(&dir);
    say!("");
    say!("  Next:");
    let mut next = vec![
        (format!("bench check {p}"), "validate it (instant)"),
        (
            format!("bench test {p}"),
            "prove it: the oracle passes, doing nothing fails, exploits fail",
        ),
    ];
    // Suggest an agent this machine can run; else say how to get one.
    match crate::cmd::agents::first_ready_agent() {
        Some(agent) => next.push((
            format!("bench run {p} -a {agent}"),
            "see how an agent does (it has a login here)",
        )),
        None => next.push((
            "bench agents".into(),
            "pick an agent to run on it; none has a login on this machine yet",
        )),
    }
    let w = next
        .iter()
        .map(|(c, _)| c.chars().count())
        .max()
        .unwrap_or(0);
    for (command, what) in &next {
        say!("    {}  {}", bold(format!("{command:<w$}")), dim(*what));
    }
    let instruction = if dir.join("task.md").is_file() {
        format!("the top of {p}/task.md")
    } else {
        format!("{p}/instruction.md")
    };
    say!(
        "  {}",
        dim(format!(
            "Then make it yours: the instruction is {instruction}."
        ))
    );
    Ok(Exit::Ok)
}
