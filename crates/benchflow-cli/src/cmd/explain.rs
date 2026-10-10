//! `bench explain`: the long form of every diagnostic and failure code.

use benchflow_run::review::{
    self,
    task::{CRITERIA, Criterion},
};

use crate::cli::ExplainArgs;
use crate::error::{CliError, Exit, Result};
use crate::say;
use crate::ui::{bold, dim, table::Table, ui};

pub fn run(args: ExplainArgs) -> Result<Exit> {
    let Some(code) = args.code else {
        return list();
    };
    let code = code.trim().to_ascii_uppercase();
    if let Some(e) = benchflow_task::diag::explain(&code) {
        if ui().json {
            println!(
                "{}",
                serde_json::json!({"code": e.code, "title": e.title, "body": e.body})
            );
        } else {
            say!("{} {}\n", bold(e.code), e.title);
            say!("{}", e.body.trim_end());
        }
        return Ok(Exit::Ok);
    }
    if let Some(e) = benchflow_run::outcome::explain(&code) {
        if ui().json {
            println!(
                "{}",
                serde_json::json!({"code": e.code, "title": e.title, "owner": e.owner, "body": e.body})
            );
        } else {
            say!("{} {}", bold(e.code), e.title);
            say!("{}\n", dim(format!("whose problem: {}", e.owner.label())));
            say!("{}", e.body.trim_end());
        }
        return Ok(Exit::Ok);
    }
    if let Some(c) = review_criterion(&code) {
        let body = review_body(c);
        if ui().json {
            println!(
                "{}",
                serde_json::json!({"code": c.code, "title": c.title, "criterion": c.id, "body": body})
            );
        } else {
            say!("{} {}", bold(c.code), c.title);
            say!("{}\n", dim(format!("model review criterion `{}`", c.id)));
            say!("{body}");
        }
        return Ok(Exit::Ok);
    }
    Err(CliError::usage(format!("no code {code}")).next("`bench explain` lists every code"))
}

/// The criteria `bench check --review` judges, and M100 for anything else.
fn review_criterion(code: &str) -> Option<&'static Criterion> {
    CRITERIA
        .iter()
        .chain(std::iter::once(&review::task::OTHER))
        .find(|c| c.code.eq_ignore_ascii_case(code))
}

fn review_body(c: &Criterion) -> String {
    let question = if c.question.is_empty() {
        "A problem the model found that none of the named criteria covers.".to_string()
    } else {
        format!("What the model is asked: {}", c.question)
    };
    format!(
        "{question}\n\nOnly `bench check --review` reports this code. It is a model's opinion: BenchFlow checks that the file and line it points to exist, not that the claim is right, and it never changes `bench check`'s exit code."
    )
}

fn list() -> Result<Exit> {
    let task_codes = benchflow_task::codes::all();
    let run_codes = benchflow_run::outcome::EXPLANATIONS;
    if ui().json {
        let v: Vec<serde_json::Value> =
            task_codes
                .iter()
                .map(|e| serde_json::json!({"code": e.code, "title": e.title}))
                .chain(run_codes.iter().map(
                    |e| serde_json::json!({"code": e.code, "title": e.title, "owner": e.owner}),
                ))
                .chain(
                    CRITERIA
                        .iter()
                        .chain(std::iter::once(&review::task::OTHER))
                        .map(|c| serde_json::json!({"code": c.code, "title": c.title})),
                )
                .collect();
        println!("{}", serde_json::Value::Array(v));
        return Ok(Exit::Ok);
    }
    if !task_codes.is_empty() {
        say!("{}", bold("Task checks (bench check)"));
        let mut t = Table::new(&["code", "what"]);
        for e in &task_codes {
            t.row(vec![e.code.to_string(), e.title.to_string()]);
        }
        say!("{}", t.render(2));
    }
    say!("{}", bold("Run failures (bench run, bench test)"));
    let mut t = Table::new(&["code", "whose", "what"]);
    for e in run_codes {
        t.row(vec![
            e.code.to_string(),
            e.owner.label().to_string(),
            e.title.to_string(),
        ]);
    }
    say!("{}", t.render(2));
    say!(
        "{}",
        bold("Model review (bench check --review; a model's opinion)")
    );
    let mut t = Table::new(&["code", "what"]);
    for c in CRITERIA.iter().chain(std::iter::once(&review::task::OTHER)) {
        t.row(vec![c.code.to_string(), c.title.to_string()]);
    }
    say!("{}", t.render(2));
    say!("{}", dim("`bench explain <code>` gives the long form."));
    Ok(Exit::Ok)
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};
    use std::path::{Path, PathBuf};

    use super::*;

    /// Files that explain codes or map them for BenchFlow 0.8, rather than
    /// emit them.
    const TABLES: &[&str] = &[
        "benchflow-run/src/outcome.rs",
        "benchflow-run/src/record.rs",
        "benchflow-run/src/review/task.rs",
        "benchflow-task/src/taskmd/code.rs",
        "benchflow-task/src/taskmd/explain.rs",
    ];

    /// Every source file of the workspace's crates, with test code left out:
    /// `tests/` folders, `tests.rs` files, and `#[cfg(test)] mod tests`.
    fn sources() -> Vec<(String, String)> {
        let crates = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let mut files = Vec::new();
        for entry in walkdir::WalkDir::new(&crates)
            .into_iter()
            .filter_map(std::result::Result::ok)
        {
            let path = entry.path();
            let rel = path
                .strip_prefix(&crates)
                .unwrap_or(path)
                .to_string_lossy()
                .replace('\\', "/");
            let in_src = rel.split('/').nth(1) == Some("src");
            if !in_src
                || path.extension().is_none_or(|e| e != "rs")
                || rel.ends_with("tests.rs")
                || rel.contains("/tests/")
            {
                continue;
            }
            let text = std::fs::read_to_string(path).unwrap();
            let code = match text.find("#[cfg(test)]\nmod tests") {
                Some(i) => text[..i].to_string(),
                None => text,
            };
            files.push((rel, code));
        }
        files
    }

    /// The text with every `EXPLANATIONS` table cut out.
    fn without_tables(text: &str) -> String {
        let mut out = text.to_string();
        while let Some(start) = out.find("EXPLANATIONS: &[Explanation] = &[") {
            let end = out[start..]
                .find("\n];")
                .map_or(out.len(), |e| start + e + 3);
            out.replace_range(start..end, "");
        }
        out
    }

    /// Code literals (`"R211"`, `"X100"`, `"P001"`) a file emits: every one
    /// outside a comparison (`p.code == "R213"` reads a code, it does not
    /// emit one).
    fn emitted_literals(text: &str) -> Vec<String> {
        let bytes = text.as_bytes();
        let mut found = Vec::new();
        for (i, w) in bytes.windows(6).enumerate() {
            let is_code = w[0] == b'"'
                && matches!(w[1], b'R' | b'X' | b'P')
                && w[2..5].iter().all(u8::is_ascii_digit)
                && w[5] == b'"';
            if !is_code {
                continue;
            }
            let before = text[..i].trim_end();
            if before.ends_with("==") || before.ends_with("!=") {
                continue;
            }
            found.push(text[i + 1..i + 5].to_string());
        }
        found
    }

    /// The task.md checker's codes, by their constant's name in
    /// `taskmd/code.rs`.
    fn task_md_constants(files: &[(String, String)]) -> BTreeMap<String, String> {
        let crates = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let text =
            std::fs::read_to_string(crates.join("benchflow-task/src/taskmd/code.rs")).unwrap();
        let _ = files;
        text.lines()
            .filter_map(|line| {
                let rest = line.trim().strip_prefix("pub const ")?;
                let (name, value) = rest.split_once(": &str = \"")?;
                Some((name.to_string(), value.trim_end_matches("\";").to_string()))
            })
            .collect()
    }

    fn explained() -> BTreeSet<String> {
        benchflow_task::codes::all()
            .iter()
            .map(|e| e.code.to_string())
            .chain(
                benchflow_run::outcome::EXPLANATIONS
                    .iter()
                    .map(|e| e.code.to_string()),
            )
            .chain(
                CRITERIA
                    .iter()
                    .chain(std::iter::once(&review::task::OTHER))
                    .map(|c| c.code.to_string()),
            )
            .collect()
    }

    /// A code exists without an explanation, or an explanation without a
    /// code anything emits: either way `bench explain` and what users see
    /// have drifted apart.
    #[test]
    fn every_emitted_code_is_explained_and_every_explanation_is_emitted() {
        let files = sources();
        let mut emitted: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for (rel, text) in &files {
            if TABLES.iter().any(|t| rel.ends_with(t)) {
                continue;
            }
            for code in emitted_literals(&without_tables(text)) {
                emitted.entry(code).or_default().push(rel.clone());
            }
        }
        // task.md codes are emitted through their constants.
        for (name, code) in task_md_constants(&files) {
            let used = files.iter().any(|(rel, text)| {
                !TABLES.iter().any(|t| rel.ends_with(t)) && text.contains(&format!("code::{name}"))
            });
            if used {
                emitted
                    .entry(code)
                    .or_default()
                    .push(format!("code::{name}"));
            }
        }
        // A review finding carries its criterion's code: the criteria are
        // what the model is asked about.
        for c in CRITERIA.iter().chain(std::iter::once(&review::task::OTHER)) {
            emitted
                .entry(c.code.to_string())
                .or_default()
                .push(format!("criterion {}", c.id));
        }
        let explained = explained();
        let unexplained: Vec<String> = emitted
            .iter()
            .filter(|(code, _)| !explained.contains(*code))
            .map(|(code, at)| format!("{code} (emitted in {})", at.join(", ")))
            .collect();
        let unused: Vec<&String> = explained
            .iter()
            .filter(|code| !emitted.contains_key(*code))
            .collect();
        assert!(
            unexplained.is_empty() && unused.is_empty(),
            "codes without an explanation: {unexplained:?}\nexplanations no code emits: {unused:?}"
        );
        // Every code is explained once, in one family.
        let mut seen = BTreeSet::new();
        for code in benchflow_task::codes::all()
            .iter()
            .map(|e| e.code)
            .chain(benchflow_run::outcome::EXPLANATIONS.iter().map(|e| e.code))
            .chain(CRITERIA.iter().map(|c| c.code))
        {
            assert!(seen.insert(code), "{code} is explained twice");
        }
    }

    #[test]
    fn the_scan_finds_emitted_codes_and_skips_comparisons() {
        let text =
            r#"Problem::new("R211", o, p, m, n); if p.code == "R213" {} let c = ("R214", x);"#;
        assert_eq!(emitted_literals(text), ["R211", "R214"]);
        let table = "a\npub static EXPLANATIONS: &[Explanation] = &[\n  \"X100\"\n];\nb(\"X101\")";
        assert_eq!(emitted_literals(&without_tables(table)), ["X101"]);
        let _ = PathBuf::new();
    }
}
