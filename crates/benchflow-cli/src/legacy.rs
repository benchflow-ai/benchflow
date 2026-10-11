//! BenchFlow 0.8's spellings, translated to the new commands so scripts keep
//! working. Each translation prints one note saying the new spelling.

/// The 0.8 commands this CLI does not have yet, with what to do instead.
const NOT_YET: &[(&str, &str)] = &[
    (
        "eval resume",
        "rerun the same `bench run` command; it resumes the latest matching job",
    ),
    ("eval score", "rescoring is not in the Rust CLI yet"),
    (
        "eval smoke",
        "run `bench doctor`, then `bench run <task> -a <agent>`",
    ),
    (
        "eval adopt",
        "benchmark adoption is not in the Rust CLI yet",
    ),
    (
        "eval continue",
        "continued runs are not in the Rust CLI yet",
    ),
    (
        "eval continue-batch",
        "continued runs are not in the Rust CLI yet",
    ),
    ("tasks normalize", "not in the Rust CLI yet"),
    ("tasks snapshot-hf", "not in the Rust CLI yet"),
    (
        "tasks digest",
        "`bench check --json` prints each task's digest",
    ),
    ("tasks overlap", "not in the Rust CLI yet"),
    ("tasks generate", "not in the Rust CLI yet"),
    ("tasks list-sources", "not in the Rust CLI yet"),
];

/// Commands of 0.8 with no Rust equivalent at all yet.
const PYTHON_ONLY: &[&str] = &[
    "train",
    "skills",
    "sandbox",
    "environment",
    "traj",
    "hub",
    "review",
    "embodied",
    "monitor",
];

/// Flags of `bench eval run` and their new spelling (`None`: same name).
const RUN_FLAGS: &[(&str, Option<&str>)] = &[
    ("--agent", None),
    ("--model", None),
    ("--sandbox", None),
    ("--concurrency", None),
    ("--jobs-dir", None),
    ("--fresh", None),
    ("--include", None),
    ("--exclude", None),
    ("--timeout-multiplier", None),
    ("--extra-instruction", None),
    ("--fail-under", None),
    ("--dry-run", None),
    ("--quiet", None),
    ("--job-name", Some("--job")),
    ("--n-tasks", Some("--limit")),
    ("--agent-env", Some("--env")),
    ("--reasoning-effort", Some("--effort")),
    ("--dataset", None),
    ("--registry", None),
    // `bench`'s own options, which may follow any command.
    ("--json", None),
    ("--verbose", None),
    ("--color", None),
];

/// Flags in [`RUN_FLAGS`] that take no value.
const RUN_SWITCHES: &[&str] = &["--fresh", "--dry-run", "--quiet", "--json", "--verbose"];

const PYTHON_HINT: &str = "The Python CLI (BenchFlow 0.8.0) has it: `uv tool install benchflow`, or `uv run bench ...` in a checkout of release/0.8";

/// Rewrite 0.8 argv into the new commands. `Err` is a message for a 0.8
/// command that has no Rust equivalent yet.
pub fn translate(args: Vec<String>) -> (Vec<String>, Option<String>) {
    match try_translate(&args) {
        Ok(Some((new, note))) => (new, Some(note)),
        Ok(None) => (args, None),
        Err(msg) => {
            anstream::eprintln!("error: {msg}");
            std::process::exit(2);
        }
    }
}

fn try_translate(args: &[String]) -> Result<Option<(Vec<String>, String)>, String> {
    let Some(first) = args.get(1).map(String::as_str) else {
        return Ok(None);
    };
    if PYTHON_ONLY.contains(&first) {
        return Err(format!(
            "`bench {first}` is not in the Rust CLI yet. {PYTHON_HINT}"
        ));
    }
    if !matches!(first, "eval" | "tasks" | "agent") {
        return Ok(None);
    }
    let sub = args.get(2).map(String::as_str).unwrap_or("");
    let old = format!("{first} {sub}");
    if let Some((_, what)) = NOT_YET.iter().find(|(k, _)| *k == old) {
        return Err(format!(
            "`bench {old}` is not in the Rust CLI yet: {what}. {PYTHON_HINT}"
        ));
    }
    let rest: Vec<String> = args.iter().skip(3).cloned().collect();
    let bench = args[0].clone();
    let (new_cmd, mut tail): (&str, Vec<String>) = match (first, sub) {
        ("eval", "run" | "create") => ("run", translate_run_flags(&rest)?),
        ("eval", "list" | "metrics") => ("results", rest),
        ("eval", "inspect") => ("results", [rest, vec!["--trials".into()]].concat()),
        ("eval", "compare") => ("results", rest),
        ("eval", "view") => ("view", rest),
        ("eval", "regrade") => ("regrade", rest),
        ("eval", "branch") => ("branch", translate_branch_flags(&rest)?),
        ("eval", "branches") => ("branches", rest),
        ("tasks", "init") => ("new", rest),
        ("tasks", "check") => ("check", translate_level(rest)),
        ("tasks", "migrate") => ("import", strip_flags(rest, &["--remove-legacy"])),
        ("tasks", "export") => ("export", rest),
        ("agent", "list") => ("agents", rest),
        ("agent", "show") => ("agents", rest),
        _ => {
            return Err(format!(
                "`bench {old}` is not a command. Run `bench --help` for the commands."
            ));
        }
    };
    let mut new = vec![bench, new_cmd.to_string()];
    new.append(&mut tail);
    let note = format!("`bench {old}` is now `bench {new_cmd}` (the old spelling keeps working)");
    Ok(Some((new, note)))
}

/// `bench eval branch`'s flags in `bench branch`'s spelling.
///
/// 0.8's defaults differ from `bench branch`'s, so where a flag is absent its
/// 0.8 default is filled in, or named as an error when the Rust CLI does not
/// have it: the agent (`claude-agent-acp`, named), the fork point (after
/// prompt 1, prompt 0 for the oracle; filled in), the parent (`continue`;
/// filled in), and children in place unless `--isolate-children` or
/// `--concurrency` above 1 (named).
fn translate_branch_flags(rest: &[String]) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    let mut it = rest.iter();
    let (mut agent, mut at, mut parent, mut isolated) = (None::<String>, false, false, false);
    let (mut has_j, mut has_sandbox) = (false, false);
    while let Some(a) = it.next() {
        let (flag, inline) = match a.split_once('=') {
            Some((f, v)) if f.starts_with("--") => (f.to_string(), Some(v.to_string())),
            _ => (a.clone(), None),
        };
        let mut value = || {
            inline
                .clone()
                .or_else(|| it.next().cloned())
                .ok_or(format!("{flag} needs a value"))
        };
        match flag.as_str() {
            "--help" | "--json" | "--verbose" | "--quiet" => out.push(flag),
            "--tasks-dir" => out.push(value()?),
            "--agent" | "-a" => {
                let a = value()?;
                agent = Some(a.clone());
                out.extend(["--agent".into(), a]);
            }
            "--model" | "-m" => out.extend(["--model".into(), value()?]),
            "--reasoning-effort" => out.extend(["--effort".into(), value()?]),
            "--sandbox" | "-s" => {
                has_sandbox = true;
                out.extend(["--sandbox".into(), value()?]);
            }
            "--agent-env" => out.extend(["--env".into(), value()?]),
            "--child" => out.extend(["--child".into(), value()?]),
            "--parent" => {
                parent = true;
                out.extend(["--parent".into(), value()?]);
            }
            "--concurrency" | "-j" => {
                has_j = true;
                let j = value()?;
                isolated |= j.parse::<u32>().is_ok_and(|n| n > 1);
                out.extend(["-j".into(), j]);
            }
            "--jobs-dir" => out.extend(["--jobs-dir".into(), value()?]),
            "--job-name" => out.extend(["--job".into(), value()?]),
            "--retain-snapshots" => out.push("--retain-checkpoints".into()),
            "--resume-session" => out.push("--resume-session".into()),
            "--checkpoint-after-prompt" => {
                at = true;
                let n = value()?;
                out.extend(["--at".into(), format!("prompt:{n}")]);
            }
            // Children always have sandboxes of their own here.
            "--isolate-children" => isolated = true,
            "--snapshot-layers" => {
                let v = value()?;
                if v != "sandbox" {
                    return Err(format!(
                        "--snapshot-layers {v}: only the sandbox layer is in the Rust CLI yet. {PYTHON_HINT}"
                    ));
                }
            }
            f if f.starts_with('-') => {
                return Err(format!(
                    "`{f}` (a `bench eval branch` flag) is not in the Rust CLI yet. {PYTHON_HINT}"
                ));
            }
            positional => out.push(positional.to_string()),
        }
    }
    // Help prints whatever else the line says.
    if out.iter().any(|a| a == "--help") {
        return Ok(out);
    }
    let Some(agent) = agent else {
        return Err(format!(
            "`bench eval branch` runs claude-agent-acp when --agent is absent; name the agent with --agent. {PYTHON_HINT}"
        ));
    };
    if !parent {
        out.extend(["--parent".into(), "continue".into()]);
    }
    // 0.8's defaults: one child at a time, on Docker.
    if !has_j {
        out.extend(["-j".into(), "1".into()]);
    }
    if !has_sandbox {
        out.extend(["--sandbox".into(), "docker".into()]);
    }
    if !isolated {
        return Err(format!(
            "`bench eval branch` runs children in place in the parent's sandbox unless --isolate-children (or --concurrency above 1); the Rust CLI runs each child in a sandbox of its own, so add --isolate-children. {PYTHON_HINT}"
        ));
    }
    if !at && agent != "oracle" {
        out.extend(["--at".into(), "prompt:1".into()]);
    }
    Ok(out)
}

fn translate_run_flags(rest: &[String]) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    let mut repo: Option<String> = None;
    let mut path: Option<String> = None;
    let mut reference: Option<String> = None;
    let mut it = rest.iter().peekable();
    while let Some(a) = it.next() {
        let (flag, inline) = match a.split_once('=') {
            Some((f, v)) if f.starts_with("--") => (f.to_string(), Some(v.to_string())),
            _ => (a.clone(), None),
        };
        let value =
            |it: &mut std::iter::Peekable<std::slice::Iter<'_, String>>| -> Result<String, String> {
                inline
                    .clone()
                    .or_else(|| it.next().cloned())
                    .ok_or(format!("{flag} needs a value"))
            };
        match flag.as_str() {
            "--help" => out.push(flag),
            "--tasks-dir" => out.push(value(&mut it)?),
            "--source-repo" => repo = Some(value(&mut it)?),
            "--source-path" => path = Some(value(&mut it)?),
            "--source-ref" => reference = Some(value(&mut it)?),
            "--agent-idle-timeout" => {
                let v = value(&mut it)?;
                out.push("--idle-timeout".into());
                out.push(format!("{v}s"));
            }
            "--config" | "--run-config" => {
                return Err("0.8's YAML run configs (`--config`) use different keys from the new job files; write a job file for `bench run -c` instead (`bench schema job` lists its keys)".into());
            }
            f if f.starts_with("--") => match RUN_FLAGS.iter().find(|(k, _)| *k == f) {
                Some((k, new)) => {
                    let name = new.unwrap_or(k).to_string();
                    let takes_value = !RUN_SWITCHES.contains(k);
                    out.push(name);
                    if takes_value {
                        out.push(value(&mut it)?);
                    }
                }
                None => {
                    return Err(format!(
                        "`{f}` (a `bench eval run` flag) is not in the Rust CLI yet. {PYTHON_HINT}"
                    ));
                }
            },
            positional => out.push(positional.to_string()),
        }
    }
    if let Some(r) = repo {
        let mut target = r;
        if let Some(p) = path {
            target.push('/');
            target.push_str(p.trim_matches('/'));
        }
        if let Some(f) = reference {
            target.push('@');
            target.push_str(&f);
        }
        out.insert(0, target);
    }
    Ok(out)
}

fn strip_flags(args: Vec<String>, flags: &[&str]) -> Vec<String> {
    args.into_iter()
        .filter(|a| !flags.contains(&a.as_str()))
        .collect()
}

/// 0.8's `bench tasks check --level schema` checks only the task file:
/// `--schema` here. Its other levels are what `bench check` always does.
fn translate_level(args: Vec<String>) -> Vec<String> {
    let mut schema = false;
    let mut iter = args.iter().peekable();
    while let Some(a) = iter.next() {
        if a == "--level" {
            schema |= iter.peek().is_some_and(|v| *v == "schema");
        } else if let Some(v) = a.strip_prefix("--level=") {
            schema |= v == "schema";
        }
    }
    let mut out = strip_flag_with_value(args, "--level");
    if schema {
        out.push("--schema".into());
    }
    out
}

fn strip_flag_with_value(args: Vec<String>, flag: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut skip = false;
    for a in args {
        if skip {
            skip = false;
            continue;
        }
        if a == flag {
            skip = true;
            continue;
        }
        if a.starts_with(&format!("{flag}=")) {
            continue;
        }
        out.push(a);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tr(s: &str) -> Vec<String> {
        let args: Vec<String> = s.split_whitespace().map(str::to_string).collect();
        match try_translate(&args).unwrap() {
            Some((new, _)) => new,
            None => args,
        }
    }

    #[test]
    fn eval_run_becomes_run() {
        assert_eq!(
            tr("bench eval run --tasks-dir tasks/x --agent claude --job-name j --n-tasks 3"),
            vec![
                "bench", "run", "tasks/x", "--agent", "claude", "--job", "j", "--limit", "3"
            ]
        );
        assert_eq!(
            tr(
                "bench eval run --source-repo org/repo --source-path tasks --source-ref main --agent oracle"
            ),
            vec!["bench", "run", "org/repo/tasks@main", "--agent", "oracle"]
        );
    }

    #[test]
    fn other_spellings() {
        assert_eq!(tr("bench tasks init x"), vec!["bench", "new", "x"]);
        assert_eq!(
            tr("bench tasks check t --level schema"),
            vec!["bench", "check", "t", "--schema"]
        );
        assert_eq!(
            tr("bench tasks check t --level=structure"),
            vec!["bench", "check", "t"]
        );
        assert_eq!(
            tr("bench agent show claude"),
            vec!["bench", "agents", "claude"]
        );
        assert_eq!(
            tr("bench eval inspect j"),
            vec!["bench", "results", "j", "--trials"]
        );
        assert_eq!(tr("bench run x"), vec!["bench", "run", "x"]);
        assert_eq!(tr("bench eval run --help"), vec!["bench", "run", "--help"]);
        assert_eq!(tr("bench eval run -h"), vec!["bench", "run", "-h"]);
        assert_eq!(
            tr("bench eval run t --color never --json --verbose --dry-run"),
            vec![
                "bench",
                "run",
                "t",
                "--color",
                "never",
                "--json",
                "--verbose",
                "--dry-run"
            ]
        );
    }

    #[test]
    fn eval_branch_maps_onto_bench_branch() {
        let args: Vec<String> = "bench eval branch --tasks-dir t -a oracle --checkpoint-after-prompt 0 --child label=a --child label=b --retain-snapshots --concurrency 2 --job-name x --parent discard"
            .split_whitespace()
            .map(str::to_string)
            .collect();
        let (new, _) = try_translate(&args).unwrap().unwrap();
        assert_eq!(
            new[1..].join(" "),
            "branch t --agent oracle --at prompt:0 --child label=a --child label=b --retain-checkpoints -j 2 --job x --parent discard --sandbox docker"
        );
        // 0.8's defaults: after prompt 1 for model agents, a continuing
        // parent, one child at a time, and Docker, filled in; the agent and
        // in-place children are named when absent.
        let tr = |line: &str| {
            let args: Vec<String> = line.split_whitespace().map(str::to_string).collect();
            try_translate(&args).map(|r| r.map(|(n, _)| n[1..].join(" ")))
        };
        assert_eq!(
            tr("bench eval branch --tasks-dir t -a opencode --parent discard --isolate-children --resume-session").unwrap(),
            Some("branch t --agent opencode --parent discard --resume-session -j 1 --sandbox docker --at prompt:1".into())
        );
        assert!(tr("bench eval branch --tasks-dir t --parent discard --isolate-children").is_err());
        assert!(tr("bench eval branch --help").unwrap().is_some());
        assert_eq!(
            tr("bench eval branch --tasks-dir t -a oracle --isolate-children").unwrap(),
            Some("branch t --agent oracle --parent continue -j 1 --sandbox docker".into())
        );
        assert!(tr("bench eval branch --tasks-dir t -a oracle --parent discard").is_err());
        let args: Vec<String> = "bench eval branches jobs/x --json"
            .split_whitespace()
            .map(str::to_string)
            .collect();
        let (new, _) = try_translate(&args).unwrap().unwrap();
        assert_eq!(new[1..].join(" "), "branches jobs/x --json");
    }

    #[test]
    fn unsupported_are_errors() {
        let args: Vec<String> = "bench eval branch --tasks-dir t --from-checkpoint x"
            .split_whitespace()
            .map(str::to_string)
            .collect();
        assert!(try_translate(&args).is_err());
        let args: Vec<String> = "bench eval run --checkpoints every-prompt"
            .split_whitespace()
            .map(str::to_string)
            .collect();
        assert!(try_translate(&args).is_err());
        let args: Vec<String> = "bench train convert"
            .split_whitespace()
            .map(str::to_string)
            .collect();
        assert!(try_translate(&args).is_err());
    }
}
