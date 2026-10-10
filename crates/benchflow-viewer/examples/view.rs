//! Serve a trial folder, a job folder, or a folder of jobs with the viewer,
//! or write one trial as a self-contained page:
//!
//! ```text
//! cargo run -p benchflow-viewer --example view -- <path> [--port N] [--host IP] [--open]
//! cargo run -p benchflow-viewer --example view -- <jobA> <jobB> [...]   # opens /compare
//! cargo run -p benchflow-viewer --example view -- <trial> --static <out.html>
//! ```
//!
//! The startup lines match 0.8's `bench eval view`.

use std::path::PathBuf;

use anyhow::Context;
use benchflow_viewer::{ServeOptions, Target, render_static, serve_jobs};

const USAGE: &str = "usage: view <path>... [--port N] [--host IP] [--open] [--static OUT.html]";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let mut paths = Vec::new();
    let mut options = ServeOptions::default();
    let mut static_out = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--port" => options.port = Some(args.next().context(USAGE)?.parse().context("--port")?),
            "--host" => options.host = args.next().context(USAGE)?.parse().context("--host")?,
            "--open" => options.open_browser = true,
            "--static" => static_out = Some(PathBuf::from(args.next().context(USAGE)?)),
            _ => paths.push(PathBuf::from(arg)),
        }
    }
    let path = paths.first().cloned().context(USAGE)?;

    if let Some(out) = static_out {
        std::fs::write(&out, render_static(&path)?)
            .with_context(|| format!("writing {}", out.display()))?;
        println!("Wrote {}", out.display());
        return Ok(());
    }

    let handle = serve_jobs(&paths, options).await?;
    match &handle.target {
        Target::Trial { dir } => {
            println!("Trajectory viewer: {}", handle.url);
            println!("Trial: {}", dir.display());
        }
        Target::Runs { root, runs, capped } => {
            let mut runs = format!("{runs} run{}", if *runs == 1 { "" } else { "s" });
            if *capped {
                runs = format!("first {runs} (capped — raise BENCHFLOW_VIEWER_MAX_RUNS)");
            }
            println!("Trajectory browser: {}", handle.url);
            println!("Scanning: {} ({runs})", root.display());
        }
        Target::Jobs { jobs, runs, .. } => {
            println!("Compare: {}{}", handle.url, handle.target.start_path());
            for (name, dir) in jobs {
                println!("Job {name}: {}", dir.display());
            }
            println!("{runs} runs");
        }
    }
    println!("Press Ctrl+C to stop\n");
    handle.wait().await?;
    println!("\nStopped.");
    Ok(())
}
