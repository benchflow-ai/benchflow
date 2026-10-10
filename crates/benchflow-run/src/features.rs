//! Which of a task's runtime extras ([`benchflow_task::extras::Extras`]) a
//! backend can honor, and the refusals, by name, for the rest. A task is
//! refused before its sandbox starts, never run without what it asked for.

use benchflow_sandbox::Provider;
use benchflow_task::TaskPackage;

use crate::outcome::{Owner, Phase, Problem};

/// The extras of `task` no backend of this engine can honor yet, as refusal
/// lines.
pub fn refused(_task: &TaskPackage) -> Vec<String> {
    Vec::new()
}

/// The refusal for a task with services on a backend that cannot run them.
pub fn no_services(task: &TaskPackage, backend: &str) -> Problem {
    let names: Vec<&str> = task
        .extras
        .services
        .iter()
        .map(|s| s.name.as_str())
        .collect();
    Problem::new(
        "R401",
        Owner::You,
        Phase::Setup,
        format!(
            "{} needs services beside the agent's container ({}), and the {backend} backend cannot run them",
            task.name,
            names.join(", ")
        ),
        "run it with `--sandbox docker` (or podman), which starts each service on a private network",
    )
}

/// What `task` asks that the backend behind `provider` cannot provide, as a
/// refusal naming the field; `None` when it can provide everything.
pub fn backend_refusal(task: &TaskPackage, provider: &dyn Provider) -> Option<Problem> {
    if !task.extras.services.is_empty() && provider.services().is_none() {
        return Some(no_services(task, provider.name()));
    }
    let caps = provider.capabilities();
    if task.sandbox.gpus > 0 && !caps.gpus {
        let types = if task.extras.gpu_types.is_empty() {
            String::new()
        } else {
            format!(" of type {}", task.extras.gpu_types.join(" or "))
        };
        return Some(Problem::new(
            "R401",
            Owner::You,
            Phase::Setup,
            format!(
                "{} needs {} GPU(s){types}, and the {} backend has none",
                task.name,
                task.sandbox.gpus,
                provider.name()
            ),
            "run it with `--sandbox docker` on a host with GPUs",
        ));
    }
    None
}

/// Regrading a task graded in stages is refused: each stage was graded in
/// the sandbox as it stood when the stage ended, which the saved outputs do
/// not hold.
pub fn regrade_refusal(task: &TaskPackage) -> Option<Problem> {
    task.extras.steps.as_ref().map(|_| {
        Problem::new(
            "R401",
            Owner::You,
            Phase::Verify,
            format!(
                "{} is graded in stages, and each stage was graded in the sandbox as it stood when the stage ended; the saved outputs cannot be graded again stage by stage",
                task.name
            ),
            "run the task again instead of regrading it",
        )
    })
}
