//! `bench`: run AI agents on tasks in sandboxes, and score them.

mod cli;
mod cmd;
mod context;
mod error;
#[cfg(test)]
mod fuzz_tests;
mod legacy;
#[cfg(test)]
mod reference;
mod ui;

use std::process::ExitCode;

use clap::{CommandFactory, FromArgMatches};

use crate::error::{CliError, Exit};
use crate::ui::{Ui, Verbosity};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let (args, rewrite_note) = legacy::translate(args);
    // build.rs adds the git commit and its date to the version, when known.
    let parsed = cli::Cli::command()
        .version(env!("BENCH_LONG_VERSION"))
        .try_get_matches_from(&args)
        .and_then(|m| cli::Cli::from_arg_matches(&m));
    let cli = match parsed {
        Ok(c) => c,
        Err(e) => {
            let _ = e.print();
            return ExitCode::from(match e.kind() {
                clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion => 0,
                _ => Exit::Usage as u8,
            });
        }
    };
    let g = &cli.global;
    let verbosity = if g.quiet {
        Verbosity::Quiet
    } else {
        match g.verbose {
            0 => Verbosity::Normal,
            1 => Verbosity::Verbose,
            _ => Verbosity::Debug,
        }
    };
    let color = ui::color_enabled(&g.color);
    ui::init(Ui {
        json: g.json,
        color,
        verbosity,
        interactive: std::io::IsTerminal::is_terminal(&std::io::stderr()),
    });
    if verbosity == Verbosity::Debug {
        let _ = tracing_subscriber::fmt()
            .with_env_filter("benchflow=debug,bench=debug")
            .with_writer(std::io::stderr)
            .try_init();
    }
    if let Some(dir) = &g.dir
        && let Err(e) = std::env::set_current_dir(dir)
    {
        let err = CliError::usage(format!("cannot change to {}: {e}", dir.display()));
        err.print();
        return ExitCode::from(err.exit as u8);
    }
    context::load_dotenv_before_threads();
    if let Some(note) = rewrite_note {
        note!("{} {}", ui::dim("note:"), note);
    }
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(r) => r,
        Err(e) => {
            eprintln!("bench: could not start the async runtime: {e}");
            return ExitCode::from(1);
        }
    };
    let result = runtime.block_on(cmd::dispatch(cli));
    // Do not wait on stray blocking tasks (an open browser, a stuck pipe).
    runtime.shutdown_timeout(std::time::Duration::from_millis(500));
    match result {
        Ok(exit) => ExitCode::from(exit as u8),
        Err(err) => {
            err.print();
            ExitCode::from(err.exit as u8)
        }
    }
}
