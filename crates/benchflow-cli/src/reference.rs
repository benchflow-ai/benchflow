//! The generated part of docs/rust/cli.md: every command's usage, flags,
//! and examples, rendered from the same clap definitions the binary parses.
//! A test keeps the file in step; `UPDATE_DOCS=1 cargo test -p
//! benchflow-cli reference` rewrites it.

use clap::CommandFactory;

use crate::cli::Cli;

pub const BEGIN: &str = "<!-- BEGIN GENERATED: bench commands -->";
pub const END: &str = "<!-- END GENERATED -->";

/// Markdown for every visible command.
pub fn markdown() -> String {
    let root = Cli::command();
    let mut out = String::new();
    for sub in root.get_subcommands().filter(|c| !c.is_hide_set()) {
        let name = sub.get_name();
        out.push_str(&format!("### bench {name}\n\n"));
        if let Some(about) = sub.get_about() {
            out.push_str(&format!("{about}.\n\n"));
        }
        let mut sub = sub.clone().bin_name(format!("bench {name}"));
        let usage = sub.render_usage().to_string();
        let usage = usage.trim_start_matches("Usage: ").trim();
        out.push_str(&format!("```text\n{usage}\n```\n\n"));
        let args: Vec<&clap::Arg> = sub
            .get_arguments()
            .filter(|a| !a.is_hide_set() && !a.is_global_set() && a.get_id() != "help")
            .collect();
        if !args.is_empty() {
            out.push_str("| Argument | Meaning |\n|---|---|\n");
            for a in args {
                let mut flag = match (a.get_short(), a.get_long()) {
                    (Some(s), Some(l)) => format!("`-{s}`, `--{l}`"),
                    (None, Some(l)) => format!("`--{l}`"),
                    (Some(s), None) => format!("`-{s}`"),
                    (None, None) => format!("`<{}>`", a.get_id().as_str().to_uppercase()),
                };
                if a.get_long().is_some()
                    && a.get_action().takes_values()
                    && let Some(v) = a.get_value_names().and_then(|v| v.first())
                {
                    flag.push_str(&format!(" `<{v}>`"));
                }
                let help = a.get_help().map(|h| h.to_string()).unwrap_or_default();
                let mut meaning = help.replace('|', "\\|");
                let defaults: Vec<String> = a
                    .get_default_values()
                    .iter()
                    .map(|d| d.to_string_lossy().to_string())
                    .collect();
                if !defaults.is_empty() && !a.get_action().takes_values().eq(&false) {
                    meaning.push_str(&format!(" (default: `{}`)", defaults.join(", ")));
                }
                let possible: Vec<String> = a
                    .get_possible_values()
                    .iter()
                    .filter(|p| !p.is_hide_set())
                    .map(|p| format!("`{}`", p.get_name()))
                    .collect();
                if !possible.is_empty() && a.get_long() != Some("color") {
                    meaning.push_str(&format!("; one of {}", possible.join(", ")));
                }
                out.push_str(&format!("| {flag} | {meaning} |\n"));
            }
            out.push('\n');
        }
        if let Some(after) = sub.get_after_help() {
            let text = after.to_string();
            out.push_str(&format!("```text\n{}\n```\n\n", text.trim_end()));
        }
        let _ = sub.render_help();
    }
    out.trim_end().to_string() + "\n"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn after_help_lists_keep_their_indent() {
        // A `\` line continuation in a string literal eats the next line's
        // leading spaces, so a list right after it loses its indent.
        let root = Cli::command();
        for sub in root.get_subcommands() {
            let Some(after) = sub.get_after_help() else {
                continue;
            };
            let text = after.to_string();
            let first = text.lines().next().unwrap_or_default();
            assert!(
                first.starts_with(' ') || !first.contains("    "),
                "bench {}'s help starts with an unindented list item: {first:?}",
                sub.get_name()
            );
        }
    }

    #[test]
    fn reference_is_current() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/rust/cli.md");
        let text = std::fs::read_to_string(&path).expect("docs/rust/cli.md");
        let (Some(b), Some(e)) = (text.find(BEGIN), text.find(END)) else {
            panic!("docs/rust/cli.md lacks the generated-block markers");
        };
        let current = &text[b + BEGIN.len()..e];
        let fresh = format!("\n\n{}\n", markdown());
        if current != fresh {
            if std::env::var_os("UPDATE_DOCS").is_some() {
                let updated = format!("{}{}{}{}", &text[..b], BEGIN, fresh, &text[e..]);
                std::fs::write(&path, updated).expect("write cli.md");
            } else {
                panic!(
                    "docs/rust/cli.md is out of date; run `UPDATE_DOCS=1 cargo test -p benchflow-cli reference`"
                );
            }
        }
    }
}
