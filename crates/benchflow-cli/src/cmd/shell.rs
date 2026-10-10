//! `bench shell`: a shell in a task's sandbox, as the agent would see it;
//! then grade what you left with the same handoff and verifier a run uses.
//!
//! The shell is a "model agent" whose model is you: the trial pipeline runs
//! unchanged, and the attempt phase hands the terminal over.

use std::process::Stdio;
use std::sync::Arc;

use async_trait::async_trait;
use benchflow_run::agent::{AgentContext, AgentOutput, Attempt, ModelAgent};
use benchflow_run::job::JobSpec;
use benchflow_run::outcome::{Owner, Phase, Problem, Status};
use benchflow_run::record::fmt_reward;
use serde_json::json;

use crate::cli::{KeepArg, SandboxArgs, ShellArgs};
use crate::cmd::engine;
use crate::context;
use crate::error::{CliError, Exit, Result};
use crate::ui::{bold, dim, fail_mark, ok_mark, warn_mark};
use crate::{note, say};

struct Shell {
    backend: String,
    as_root: bool,
    grade: bool,
}

#[async_trait]
impl ModelAgent for Shell {
    fn name(&self) -> &str {
        "shell"
    }
    fn model(&self) -> Option<&str> {
        None
    }
    fn install_layers(&self) -> Vec<String> {
        Vec::new()
    }
    fn fingerprint(&self) -> String {
        String::new()
    }
    fn preflight(&self) -> std::result::Result<(), Problem> {
        Ok(())
    }
    fn needs_network(&self) -> bool {
        false
    }
    async fn run(&self, ctx: AgentContext<'_>) -> AgentOutput {
        let user = if self.as_root { "root" } else { ctx.user };
        let cli = match self.backend.as_str() {
            "docker" | "podman" => self.backend.clone(),
            other => {
                return AgentOutput {
                    problem: Some(Problem::new(
                        "R401",
                        Owner::You,
                        Phase::Agent,
                        format!(
                            "`bench shell` needs a local container backend; {other} is not one yet"
                        ),
                        "run it with --sandbox docker",
                    )),
                    ..AgentOutput::default()
                };
            }
        };
        say!("");
        say!(
            "  {} {} in {} as {}",
            bold("Shell"),
            ctx.sandbox.id(),
            ctx.workdir,
            user
        );
        say!("  {}", dim("The instruction the agent gets:"));
        for line in ctx.instruction.lines().take(12) {
            say!("    {}", dim(line));
        }
        if ctx.instruction.lines().count() > 12 {
            say!("    {}", dim("…"));
        }
        say!(
            "  {}",
            dim(if self.grade {
                "Exit the shell to grade the sandbox as you left it."
            } else {
                "Exit the shell to destroy the sandbox."
            })
        );
        let shell_cmd = shell_command(if self.as_root { "/root" } else { ctx.home }, ctx.workdir);
        let status = tokio::process::Command::new(&cli)
            .args([
                "exec",
                "-it",
                "-u",
                user,
                "-w",
                ctx.workdir,
                ctx.sandbox.id(),
                "sh",
                "-c",
                &shell_cmd,
            ])
            .stdin(Stdio::inherit())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .status()
            .await;
        let mut out = AgentOutput {
            skip_grading: !self.grade,
            prompts: vec![ctx.instruction.to_string()],
            ..AgentOutput::default()
        };
        if let Err(e) = status {
            out.problem = Some(Problem::new(
                "R107",
                Owner::You,
                Phase::Agent,
                format!("starting `{cli} exec` failed on this machine: {e}"),
                format!(
                    "check that {cli} is installed and on your PATH; `bench doctor` checks the backend"
                ),
            ));
        }
        out
    }
}

pub async fn run(args: ShellArgs) -> Result<Exit> {
    let project = context::project()?;
    let sargs = SandboxArgs {
        sandbox: args.sandbox.clone(),
        concurrency: Some(1),
        keep: KeepArg::Never,
        jobs_dir: None,
        profile: None,
        runtime: None,
    };
    let st = engine::settings(&project, &sargs)?;
    let task_path = args.task.clone().unwrap_or_else(|| ".".into());
    let (tasks, bad) = context::load_tasks(std::slice::from_ref(&task_path))?;
    if !bad.is_empty() && tasks.is_empty() {
        return Err(
            CliError::failed(format!("{} has errors", task_path.display()))
                .next(format!("bench check {}", task_path.display())),
        );
    }
    if tasks.len() != 1 {
        return Err(CliError::usage(format!(
            "{} holds {} tasks; name one",
            task_path.display(),
            tasks.len()
        )));
    }
    let provider = context::provider(st.sandbox.as_deref()).await?;
    // The shell is `docker exec -it` (or podman's): refuse before building
    // anything on a backend without one.
    if !matches!(provider.name(), "docker" | "podman") {
        return Err(CliError::usage(format!(
            "`bench shell` needs a local container backend (docker or podman); {} is not one yet",
            provider.name()
        ))
        .next("run it with --sandbox docker"));
    }
    let options = engine::options(&st, provider.as_ref(), KeepArg::Never)?;
    let shell = Shell {
        backend: provider.name().to_string(),
        as_root: args.user == "root",
        grade: !args.no_grade,
    };
    note!(
        "{} {} on {}",
        bold("Preparing"),
        tasks[0].name,
        provider.name()
    );
    let spec = JobSpec {
        carry: Vec::new(),
        units: vec![(tasks[0].clone(), vec![Attempt::Model(Arc::new(shell))])],
        trials: 1,
        jobs_dir: st.jobs_dir.join("shell"),
        job_name: None,
        fresh: true,
        concurrency: 1,
        source: json!({"targets": [task_path], "kind": "bench shell"}),
    };
    // The shell owns the terminal: no spinner or heartbeat over it.
    let view = crate::ui::live::View {
        still: true,
        ..Default::default()
    };
    let outcome = engine::run_job(spec, provider, options, view).await?;
    let Some(r) = outcome.records.first() else {
        return Err(CliError::bug("bench shell", "no trial record"));
    };
    match (r.status, r.passed) {
        (Status::Skipped, _) => {
            say!("  {} not graded", dim("·"));
            Ok(Exit::Ok)
        }
        (Status::Scored, Some(true)) => {
            say!(
                "  {} passed (reward {})  {}",
                ok_mark(),
                r.reward.map(fmt_reward).unwrap_or_default(),
                dim(crate::ui::path(&outcome.dir))
            );
            Ok(Exit::Ok)
        }
        (Status::Scored, _) => {
            say!(
                "  {} failed (reward {})  {}",
                fail_mark(),
                r.reward.map(fmt_reward).unwrap_or_default(),
                dim(crate::ui::path(&outcome.dir))
            );
            Ok(Exit::Failed)
        }
        _ => {
            if let Some(p) = &r.problem {
                say!("  {} {} [{}]", warn_mark(), p.message, p.code);
                say!("      {} {}", dim("next:"), p.next);
            }
            Ok(Exit::Failed)
        }
    }
}

/// The shell `bench shell` opens in the sandbox. It starts with the line
/// every agent command starts with, so what you run there is, like an
/// agent's processes, the first the kernel stops when memory runs out.
fn shell_command(home: &str, workdir: &str) -> String {
    format!(
        "{oom}export HOME={home}; cd {wd} 2>/dev/null; if command -v bash >/dev/null; then exec bash -l; else exec sh -l; fi",
        oom = benchflow_run::hardening::oom_prefix(),
        home = benchflow_sandbox::shell::quote(home),
        wd = benchflow_sandbox::shell::quote(workdir),
    )
}

#[cfg(test)]
mod tests {
    use super::shell_command;

    #[test]
    fn the_shell_starts_with_the_agents_oom_line() {
        let cmd = shell_command("/home/agent", "/app");
        assert!(
            cmd.starts_with(&benchflow_run::hardening::oom_prefix()),
            "{cmd}"
        );
        assert!(cmd.ends_with("else exec sh -l; fi"), "{cmd}");
    }
}
