//! One module per command.

pub(crate) mod agents;
mod branch;
mod cache;
mod check;
mod convert;
mod doctor;
mod engine;
mod explain;
pub(crate) mod handoff;
mod home;
mod new;
mod regrade;
mod results;
mod review;
mod run;
mod schema;
mod shell;
mod test;
mod view;

use clap::CommandFactory;

use crate::cli::{Cli, Command};
use crate::error::{Exit, Result};

pub async fn dispatch(cli: Cli) -> Result<Exit> {
    match cli.command {
        None => home::run(),
        Some(Command::New(a)) => new::run(a),
        Some(Command::Check(a)) => check::run(a),
        Some(Command::Test(a)) => test::run(a).await,
        Some(Command::Run(a)) => run::run(a).await,
        Some(Command::Results(a)) => results::run(a),
        Some(Command::View(a)) => view::run(a).await,
        Some(Command::Shell(a)) => shell::run(a).await,
        Some(Command::Handoff(a)) => handoff::run(a),
        Some(Command::Regrade(a)) => regrade::run(a).await,
        Some(Command::Branch(a)) => branch::run(a).await,
        Some(Command::Branches(a)) => branch::show(a),
        Some(Command::Schema(a)) => schema::run(a),
        Some(Command::Doctor(a)) => doctor::run(a).await,
        Some(Command::Cache(a)) => cache::run(a).await,
        Some(Command::Agents(a)) => agents::run(a).await,
        Some(Command::Import(a)) => convert::import(a),
        Some(Command::Export(a)) => convert::export(a),
        Some(Command::Explain(a)) => explain::run(a),
        Some(Command::Completions { shell }) => {
            let mut cmd = Cli::command();
            clap_complete::generate(shell, &mut cmd, "bench", &mut std::io::stdout());
            Ok(Exit::Ok)
        }
    }
}
