//! `bench schema`: the JSON Schemas of BenchFlow's files.

use benchflow_run::jobfile::JobFile;
use benchflow_run::lock::Lock;
use benchflow_run::record::ResultFile;

use crate::cli::{SchemaArgs, SchemaKind};
use crate::error::{Exit, Result};

pub fn run(args: SchemaArgs) -> Result<Exit> {
    let schema = match args.kind {
        SchemaKind::Task => benchflow_task::taskmd::schema::config_schema(),
        SchemaKind::Job => schemars::schema_for!(JobFile).to_value(),
        SchemaKind::Result => schemars::schema_for!(ResultFile).to_value(),
        SchemaKind::Lock => schemars::schema_for!(Lock).to_value(),
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&schema).unwrap_or_default()
    );
    Ok(Exit::Ok)
}
