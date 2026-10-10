//! `bench view`: open a job in the browser.

use std::path::PathBuf;

use benchflow_viewer::{ServeOptions, Target};

use crate::cli::ViewArgs;
use crate::context;
use crate::error::{CliError, Exit, Result};
use crate::note;
use crate::ui::{self, dim, plural};

pub async fn run(args: ViewArgs) -> Result<Exit> {
    let project = context::project()?;
    let roots = match args.jobs.len() {
        0 => {
            let jobs = project
                .config
                .run
                .jobs_dir
                .clone()
                .map(|d| project.root.join(d))
                .unwrap_or_else(|| PathBuf::from("jobs"));
            if !jobs.exists() {
                return Err(CliError::usage("no ./jobs folder here")
                    .next("pass a job folder: bench view jobs/<job>"));
            }
            vec![jobs]
        }
        _ => args.jobs,
    };
    if let Some(missing) = roots.iter().find(|root| !root.exists()) {
        return Err(CliError::usage(format!(
            "{} does not exist",
            missing.display()
        )));
    }
    let options = ServeOptions {
        port: args.port,
        open_browser: !args.no_open,
        ..ServeOptions::default()
    };
    let handle = benchflow_viewer::serve_jobs(&roots, options).await.map_err(|e| {
        CliError::failed(format!("the viewer did not start: {e:#}"))
            .next("pass a job folder, a folder of jobs, or a trial folder; try another --port if one is busy")
    })?;
    let what = describe(&handle.target);
    note!(
        "  {} {}  {}{}",
        dim("viewing"),
        what,
        handle.url,
        handle.target.start_path()
    );
    note!("  {}", dim("Ctrl-C to stop"));
    handle
        .wait()
        .await
        .map_err(|e| CliError::bug("serving the viewer", e))?;
    Ok(Exit::Ok)
}

/// What the viewer serves, counted in trials: the viewer calls each trial a
/// "run", and "viewing 3 runs" after one job of 3 trials read as 3 jobs.
fn describe(target: &Target) -> String {
    match target {
        Target::Trial { dir } => format!("trial {}", ui::path(dir)),
        Target::Runs { root, runs, capped } => format!(
            "{} in {}{}",
            plural(*runs, "trial"),
            ui::path(root),
            if *capped {
                " (capped; raise BENCHFLOW_VIEWER_MAX_RUNS)"
            } else {
                ""
            }
        ),
        Target::Jobs { jobs, runs, capped } => format!(
            "{} in {} jobs{}",
            plural(*runs, "trial"),
            jobs.len(),
            if *capped {
                " (capped; raise BENCHFLOW_VIEWER_MAX_RUNS)"
            } else {
                ""
            }
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_banner_counts_trials() {
        let one_job = Target::Runs {
            root: PathBuf::from("jobs"),
            runs: 3,
            capped: false,
        };
        assert_eq!(describe(&one_job), "3 trials in jobs");
    }
}
