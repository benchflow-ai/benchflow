//! BenchFlow's own text never names another evaluation framework: help,
//! `bench explain`, agent descriptions, and the generated reference in
//! docs/rust, templates. Values a user may still type with such a name (an
//! agent's old alias, a BenchFlow 0.8 profile) are read but never listed.
//! Web addresses and the import record's identifiers (`[import.harbor]`,
//! which task.md's convert.py and earlier bench builds wrote and other tools
//! read) are data, not text, and are not counted.
//!
//! `tests/no_other_frameworks.rs` checks the same through the built binary.

use clap::CommandFactory;

use crate::cli::Cli;

#[path = "other_frameworks.rs"]
mod other_frameworks;

pub use other_frameworks::{named, named_except};

/// Every place in `texts` that names a framework, as `where: name`.
fn offenders(texts: &[(String, String)]) -> Vec<String> {
    texts
        .iter()
        .flat_map(|(place, text)| {
            named(text)
                .into_iter()
                .map(move |n| format!("{place}: {n}"))
        })
        .collect()
}

/// Every command's long help, hidden commands included.
fn help(cmd: &mut clap::Command, path: &str, out: &mut Vec<(String, String)>) {
    let mut cmd = cmd.clone().bin_name(path);
    out.push((format!("{path} --help"), cmd.render_long_help().to_string()));
    for sub in cmd.get_subcommands_mut() {
        let name = format!("{path} {}", sub.get_name());
        help(sub, &name, out);
    }
}

#[test]
fn the_names_are_found_as_whole_words_only() {
    assert_eq!(named("as Harbor's task migrate does"), ["harbor"]);
    assert_eq!(named("KAROTTE-external"), ["karotte"]);
    assert_eq!(named("uses Inspect AI logs"), ["inspect ai"]);
    assert!(named("Keep sandboxes after their trials, for inspection").is_empty());
    assert!(named("harbors and harborage").is_empty());
}

/// Spellings, separators, line breaks and CamelCase do not hide a name.
#[test]
fn the_names_are_found_however_they_are_written() {
    for (text, name) in [
        ("uses lm-evaluation-harness", "lm evaluation harness"),
        ("like OpenAI Evals", "openai evals"),
        ("via inspect-evals", "inspect evals"),
        ("as Harbour does", "harbour"),
        ("a HarborTask", "harbor"),
        ("the KarotteBackend", "karotte"),
        ("from Inspect\nAI", "inspect ai"),
        ("from Inspect  AI", "inspect ai"),
        ("from Inspect\tAI", "inspect ai"),
        ("from Inspect\u{a0}AI", "inspect ai"),
        ("from inspect_ai", "inspect ai"),
    ] {
        assert_eq!(named(text), [name], "{text:?}");
    }
    assert!(named("OpenAI's API").is_empty());
    assert!(named("docker inspect a container").is_empty());
    // An address ends at its JSON quote; the next field is text.
    assert_eq!(
        named(r#"{"url":"https://example.com/x","next":"Harbor runs"}"#),
        ["harbor"]
    );
    // A typed value counts only whole: the suggestion it is part of is bench's.
    assert_eq!(
        named_except("did you mean \"harbor-compatible\"?", &["harbor-compat"]),
        ["harbor"]
    );
}

/// Addresses, the import record's identifiers, and what the user typed are
/// not BenchFlow's words; the same names elsewhere in the text still are.
#[test]
fn addresses_identifiers_and_typed_values_are_not_text() {
    let url = "https://raw.githubusercontent.com/laude-institute/harbor/main/registry.json";
    assert!(named(&format!("could not read {url}: timed out")).is_empty());
    assert!(named(&format!("(see {url})")).is_empty());
    assert!(named("cached github.com/laude-institute/harbor-datasets@0123abc").is_empty());
    assert!(named("recorded under [import.harbor] trajectory, in import/harbor/").is_empty());
    assert_eq!(named(&format!("Harbor's registry {url}")), ["harbor"]);
    assert_eq!(
        named("recorded under [import.harbor] as Harbor does"),
        ["harbor"]
    );
    let typed = "unknown profile \"harbor-compat\"; known profiles: a, b";
    assert_eq!(named(typed), ["harbor"]);
    assert!(named_except(typed, &["harbor-compat"]).is_empty());
    assert_eq!(
        named_except("a Harbor task", &["harbor-compat"]),
        ["harbor"]
    );
    // A short value typed (a flag) does not break up a longer one.
    assert!(named_except("no agent `harbor-agent`", &["-a", "harbor-agent"]).is_empty());
    // A file name is not a host.
    assert_eq!(named("task.toml/harbor"), ["harbor"]);
}

#[test]
fn help_and_the_generated_reference_name_no_other_framework() {
    let mut texts = Vec::new();
    help(&mut Cli::command(), "bench", &mut texts);
    texts.push(("reference::markdown()".into(), crate::reference::markdown()));
    let docs = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/rust");
    for entry in std::fs::read_dir(&docs).expect("docs/rust") {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|e| e == "md") {
            let text = std::fs::read_to_string(&path).unwrap();
            texts.push((path.display().to_string(), text));
        }
    }
    assert!(texts.len() > 25, "only {} help texts", texts.len());
    let found = offenders(&texts);
    assert!(found.is_empty(), "{}", found.join("\n"));
}

#[test]
fn explanations_name_no_other_framework() {
    let mut texts: Vec<(String, String)> = Vec::new();
    for e in benchflow_task::codes::all() {
        texts.push((e.code.into(), format!("{}\n{}", e.title, e.body)));
    }
    for e in benchflow_run::outcome::EXPLANATIONS {
        texts.push((
            e.code.into(),
            format!("{}\n{}\n{}", e.title, e.owner.label(), e.body),
        ));
    }
    for c in benchflow_run::review::task::CRITERIA
        .iter()
        .chain(std::iter::once(&benchflow_run::review::task::OTHER))
    {
        texts.push((c.code.into(), format!("{}\n{}", c.title, c.question)));
    }
    assert!(texts.len() > 100, "only {} codes", texts.len());
    let found = offenders(&texts);
    assert!(found.is_empty(), "{}", found.join("\n"));
}

/// Every file `bench new --template` writes, and each template's line in
/// `bench new --list`.
#[test]
fn templates_name_no_other_framework() {
    let mut texts: Vec<(String, String)> = Vec::new();
    for t in benchflow_task::scaffold::templates() {
        texts.push((format!("template {}", t.name), t.description.into()));
        for (path, text) in t.files() {
            texts.push((format!("template {}: {path}", t.name), text.into()));
        }
    }
    assert!(texts.len() > 10, "only {} template files", texts.len());
    let found = offenders(&texts);
    assert!(found.is_empty(), "{}", found.join("\n"));
}

/// A manifest as `bench agents <name>` can show it, one entry per string
/// it holds: everything but the runner scripts a headless agent writes into
/// its home in the sandbox, which bench never prints (their comments credit
/// the code they follow, as its license asks; `include_str_targets_...`
/// reads the rest of them).
fn shown(place: &str, m: &benchflow_agent::manifest::AgentManifest) -> Vec<(String, String)> {
    let mut m = m.clone();
    if let Some(h) = m.headless.as_mut() {
        h.files.iter_mut().for_each(|f| f.contents.clear());
    }
    let mut out = Vec::new();
    strings(&serde_json::to_value(&m).unwrap(), &mut |s| {
        out.push((place.to_string(), s.to_string()))
    });
    out
}

/// Every string in `v`, keys included.
fn strings(v: &serde_json::Value, f: &mut dyn FnMut(&str)) {
    match v {
        serde_json::Value::String(s) => f(s),
        serde_json::Value::Array(a) => a.iter().for_each(|x| strings(x, f)),
        serde_json::Value::Object(o) => o.iter().for_each(|(k, x)| {
            f(k);
            strings(x, f)
        }),
        _ => {}
    }
}

/// Everything a built-in agent's manifest holds (descriptions, options,
/// install steps, login hints), and the native agents' descriptions and
/// options.
#[test]
fn agent_descriptions_name_no_other_framework() {
    let mut texts: Vec<(String, String)> = Vec::new();
    for m in benchflow_agent::manifest::Registry::builtin().agents() {
        texts.extend(shown(&format!("agent {}", m.name), m));
    }
    for (name, aliases, what) in benchflow_agent::native::AGENTS {
        let options = benchflow_agent::native::options::declared(name);
        texts.push((
            format!("agent {name}"),
            format!("{aliases:?}\n{what}\n{options:?}"),
        ));
    }
    let dspy = benchflow_run::dspy_agent::manifest();
    texts.extend(shown(&format!("agent {}", dspy.name), dspy));
    texts.push((
        "agent external".into(),
        crate::cmd::agents::EXTERNAL_WHAT.into(),
    ));
    assert!(texts.len() > 300, "only {} agent strings", texts.len());
    let found = offenders(&texts);
    assert!(found.is_empty(), "{}", found.join("\n"));
}

/// A string literal in Rust source.
#[derive(Debug, PartialEq)]
struct Lit {
    line: usize,
    /// Its value, escapes and line continuations resolved.
    text: String,
    /// It is the path of an `include_str!`.
    include: bool,
}

/// The string literals of Rust source `src`, outside comments and outside
/// what `#[cfg(test)]` is on: an item (a `mod tests { .. }`, or a single
/// `use`, `const`, `fn` or `impl`), or a field, variant, struct-literal
/// field, match arm or parameter, which ends at a `,` or at the enclosing
/// closer. The attribute counts only in code, not in a comment or a string.
/// An error when a skip runs to the end of the file: a shape this lexer
/// does not know, which must fail loudly rather than hide the rest.
fn literals(src: &str) -> Result<Vec<Lit>, String> {
    let b = src.as_bytes();
    let mut out = Vec::new();
    let (mut i, mut line) = (0, 1);
    // Inside what a `#[cfg(test)]` is on: its bracket depth, and the line
    // the attribute is on.
    let mut skipping: Option<i32> = None;
    let mut skip_line = 0;
    let at = |i: usize, s: &str| b[i..].starts_with(s.as_bytes());
    let ident = |j: usize| b[j].is_ascii_alphanumeric() || b[j] == b'_';
    while i < b.len() {
        let c = b[i];
        let raw_start = c == b'r'
            && (at(i, "r\"") || at(i, "r#"))
            && (i == 0 || !ident(i - 1) || (b[i - 1] == b'b' && (i < 2 || !ident(i - 2))));
        if c == b'\n' {
            line += 1;
            i += 1;
        } else if at(i, "//") {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
        } else if at(i, "/*") {
            let end = src[i + 2..].find("*/").map_or(b.len(), |e| i + 2 + e + 2);
            line += src[i..end].matches('\n').count();
            i = end;
        } else if raw_start {
            let hashes = b[i + 1..].iter().take_while(|&&h| h == b'#').count();
            if b.get(i + 1 + hashes) != Some(&b'"') {
                i += 1;
                continue;
            }
            let start = i + 2 + hashes;
            let close = format!("\"{}", "#".repeat(hashes));
            let end = src[start..].find(&close).map_or(b.len(), |e| start + e);
            if skipping.is_none() {
                out.push(Lit {
                    line,
                    text: src[start..end].to_string(),
                    include: false,
                });
            }
            line += src[i..end].matches('\n').count();
            i = end + close.len();
        } else if c == b'"' {
            let start = i + 1;
            let mut j = start;
            while j < b.len() && b[j] != b'"' {
                j += if b[j] == b'\\' { 2 } else { 1 };
            }
            let end = j.min(b.len());
            if skipping.is_none() {
                out.push(Lit {
                    line,
                    text: unescape(&src[start..end]),
                    include: src[..i].trim_end().ends_with("include_str!("),
                });
            }
            line += src[i..end].matches('\n').count();
            i = end + 1;
        } else if c == b'\'' {
            // A char literal ('x', '\n', '\u{..}'), or a lifetime.
            if at(i + 1, "\\") {
                i += src[i + 2..].find('\'').map_or(b.len(), |e| e + 3);
            } else if src[i + 1..].chars().nth(1) == Some('\'') {
                i += 1 + src[i + 1..].chars().next().map_or(1, char::len_utf8) + 1;
            } else {
                i += 1;
            }
        } else if skipping.is_none() && at(i, "#[cfg(test)]") {
            skipping = Some(0);
            skip_line = line;
            i += "#[cfg(test)]".len();
        } else if let Some(depth) = skipping.as_mut() {
            match c {
                b'{' | b'(' | b'[' => *depth += 1,
                // The enclosing struct, enum, literal, match or parameter
                // list closes: the attribute was on its last member.
                b')' | b']' | b'}' if *depth == 0 => skipping = None,
                b')' | b']' => *depth -= 1,
                b'}' => {
                    *depth -= 1;
                    // `mod x { .. }`, `fn`, `impl` end here; `const X: T =
                    // T { .. };` goes on to its `;`.
                    let next = src[i + 1..].trim_start().chars().next();
                    if *depth == 0 && next != Some(';') {
                        skipping = None;
                    }
                }
                // An item ends at `;`; a field, variant, struct-literal
                // field, arm or parameter at `,`. (A `,` in an item's
                // generics ends the skip early, which only reads more.)
                b';' | b',' if *depth == 0 => skipping = None,
                _ => {}
            }
            i += 1;
        } else {
            i += 1;
        }
    }
    match skipping {
        Some(_) => Err(format!(
            "the #[cfg(test)] at line {skip_line} runs to the end of the file"
        )),
        None => Ok(out),
    }
}

/// A non-raw Rust string literal's value: its escapes resolved, and a `\`
/// at a line's end dropped with the next line's leading whitespace.
fn unescape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut it = s.chars().peekable();
    while let Some(c) = it.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match it.next() {
            Some('n') => out.push('\n'),
            Some('t') => out.push('\t'),
            Some('r') => out.push('\r'),
            Some('0') => out.push('\0'),
            Some('x') => {
                let hex: String = it.by_ref().take(2).collect();
                out.extend(u8::from_str_radix(&hex, 16).ok().map(char::from));
            }
            Some('u') => {
                let hex: String = it
                    .by_ref()
                    .skip_while(|&c| c == '{')
                    .take_while(|&c| c != '}')
                    .collect();
                out.extend(u32::from_str_radix(&hex, 16).ok().and_then(char::from_u32));
            }
            Some('\n') => {
                while it.peek().is_some_and(|c| c.is_whitespace()) {
                    it.next();
                }
            }
            Some(other) => out.push(other),
            None => {}
        }
    }
    out
}

#[test]
fn literals_are_found_outside_comments_and_test_items() {
    let src = "let a = \"one\"; // \"not\"\n/* \"no\" */ let b = r#\"two\n\"x\"\"#;\nlet c = '\"'; fn f<'a>(x: &'a str) {}\n#[cfg(test)]\nuse x::y;\nlet d = \"t\\\"hree\\n\\\n    Four\";\n#[cfg(test)]\nmod tests { fn t() { let e = \"five\"; } }\nconst F: &str = include_str!(\"six.txt\");\n// #[cfg(test)]\nlet g = \"seven #[cfg(test)]\";\nlet h = \"eight\";";
    let got: Vec<(usize, String, bool)> = literals(src)
        .unwrap()
        .into_iter()
        .map(|l| (l.line, l.text, l.include))
        .collect();
    let want = [
        (1, "one", false),
        (2, "two\n\"x\"", false),
        (7, "t\"hree\nFour", false),
        (11, "six.txt", true),
        (13, "seven #[cfg(test)]", false),
        (14, "eight", false),
    ];
    let want: Vec<(usize, String, bool)> = want
        .iter()
        .map(|(l, t, i)| (*l, t.to_string(), *i))
        .collect();
    assert_eq!(got, want);
}

/// `#[cfg(test)]` on a field, a variant, a struct-literal field or a match
/// arm skips only that member, not the production code after it; a skip
/// that never ends is an error.
#[test]
fn a_test_only_member_skips_only_itself() {
    let shapes = [
        "struct S {\n    #[cfg(test)]\n    probe: u8,\n    real: u8,\n}\nimpl S { fn f() { say(\"after a field\"); } }",
        "enum E {\n    #[cfg(test)]\n    Probe(u8),\n    Real,\n}\nfn g() { say(\"after a variant\"); }",
        "fn h() -> S {\n    S {\n        #[cfg(test)]\n        probe: 1,\n        real: 2,\n    }\n}\nfn k() { say(\"after a literal field\"); }",
        "fn m(x: u8) -> &'static str {\n    match x {\n        #[cfg(test)]\n        0 => \"test arm\",\n        _ => \"after an arm\",\n    }\n}",
        "struct T {\n    real: u8,\n    #[cfg(test)]\n    probe: u8\n}\nfn n() { say(\"after a last field\"); }",
    ];
    for src in shapes {
        let got: Vec<String> = literals(src).unwrap().into_iter().map(|l| l.text).collect();
        assert!(got.iter().any(|t| t.starts_with("after")), "{src}\n{got:?}");
        assert!(!got.iter().any(|t| t == "test arm"), "{src}\n{got:?}");
    }
    assert!(literals("#[cfg(test)]\nmod tests {\n    fn t() {}\n").is_err());
}

/// The crates' source files the guards read, by path under `crates/`.
fn source_files() -> Vec<(String, std::path::PathBuf)> {
    let crates = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let mut out: Vec<(String, std::path::PathBuf)> = walkdir::WalkDir::new(&crates)
        .into_iter()
        .filter_map(Result::ok)
        .map(|e| e.path().to_path_buf())
        .filter(|p| p.extension().is_some_and(|x| x == "rs"))
        .filter_map(|p| {
            let rel = p
                .strip_prefix(&crates)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            (rel.contains("/src/")
                && !rel.starts_with("target")
                && !KEPT_FILES.contains(&rel.as_str()))
            .then_some((rel, p))
        })
        .collect();
    out.sort();
    out
}

/// What bench prints names no other framework. Identifiers and data from
/// outside bench stay as they are: dataset git URLs and descriptions from
/// the public registry, the registry's address in `--json` output and lock
/// files, `[import.harbor]` and `import/harbor/` in import notes and T222,
/// `"to": "harbor"` in `export-report.json`, and `harbor-compatible` when
/// the user typed it. bench's own words around them must not name another
/// framework.
///
/// This test reads every string literal in the crates' source outside
/// comments and `#[cfg(test)]` items (messages, notes recorded into trial
/// results, and everything else the binary can print come from these), with
/// escapes resolved, and fails on one that names another framework, except
/// the literals in [`KEPT`].
#[test]
fn source_literals_name_no_other_framework() {
    let mut found = Vec::new();
    let mut seen: std::collections::BTreeMap<(String, String), usize> = Default::default();
    let files = source_files();
    for (rel, path) in &files {
        let text = std::fs::read_to_string(path).unwrap();
        for lit in literals(&text).unwrap_or_else(|e| panic!("{rel}: {e}")) {
            if named(&lit.text).is_empty() {
                continue;
            }
            if KEPT.iter().any(|(f, l, _, _)| f == rel && *l == lit.text) {
                *seen.entry((rel.clone(), lit.text)).or_default() += 1;
                continue;
            }
            found.push(format!("{rel}:{}: {:?}", lit.line, lit.text));
        }
    }
    assert!(files.len() > 200, "only {} source files", files.len());
    // Each kept literal appears exactly as often as listed, so a new use of
    // it is a new decision.
    for (f, l, n, _) in KEPT {
        let got = seen
            .get(&(f.to_string(), l.to_string()))
            .copied()
            .unwrap_or(0);
        if got != *n {
            found.push(format!("{f}: {l:?} is kept {n} time(s) but appears {got}"));
        }
    }
    assert!(found.is_empty(), "{}", found.join("\n"));
}

/// Every file an `include_str!` outside tests puts into the binary: the
/// viewer's pages, the agents' manifests and runner scripts, the prompt
/// templates, the provider files, the ACP registry snapshot. Comments are
/// not text (agent manifests' provenance notes, runner scripts' license
/// credits), nor is a Python file's module docstring, where the runners put
/// the same credits; every other docstring is read. A runner also sets
/// `HARBOR_MODEL` in the sandbox for apps written to read it.
#[test]
fn included_files_name_no_other_framework() {
    let mut found = Vec::new();
    let mut read = 0;
    for (rel, path) in source_files() {
        let text = std::fs::read_to_string(&path).unwrap();
        for lit in literals(&text)
            .unwrap_or_else(|e| panic!("{rel}: {e}"))
            .into_iter()
            .filter(|l| l.include)
        {
            let target = path.parent().unwrap().join(&lit.text);
            let body = std::fs::read_to_string(&target)
                .unwrap_or_else(|e| panic!("{rel}:{}: {}: {e}", lit.line, target.display()));
            read += 1;
            let ext = target.extension().and_then(|e| e.to_str()).unwrap_or("");
            let body = without_comments(&body, ext).replace("HARBOR_MODEL", " ");
            let mut on_lines = std::collections::BTreeSet::new();
            for (n, l) in body.lines().enumerate() {
                if !named(l).is_empty() {
                    on_lines.extend(named(l));
                    found.push(format!("{}:{}: {l}", target.display(), n + 1));
                }
            }
            // A two-word name split across lines is found in the whole.
            for name in named(&body) {
                if !on_lines.contains(name) {
                    found.push(format!("{}: {name}, across lines", target.display()));
                }
            }
        }
    }
    assert!(read > 60, "only {read} included files");
    assert!(found.is_empty(), "{}", found.join("\n"));
}

/// `text` with its comments (and, in Python, the module docstring, where
/// the runners credit the code they follow) blanked, lines kept. Other
/// docstrings stay: DSPy, for one, sends a tool's docstring to the model.
fn without_comments(text: &str, ext: &str) -> String {
    let blank = |s: &str| {
        s.chars()
            .map(|c| if c == '\n' { '\n' } else { ' ' })
            .collect::<String>()
    };
    let mut text = text.to_string();
    if ext == "py"
        && let Some((i, end)) = module_docstring(&text)
    {
        let b = blank(&text[i..end]);
        text.replace_range(i..end, &b);
    }
    if matches!(ext, "js" | "mjs" | "css") {
        while let Some(i) = text.find("/*") {
            let end = text[i + 2..]
                .find("*/")
                .map_or(text.len(), |e| i + 2 + e + 2);
            let b = blank(&text[i..end]);
            text.replace_range(i..end, &b);
        }
    }
    let marker = match ext {
        "py" | "toml" | "sh" => Some("#"),
        "js" | "mjs" => Some("//"),
        _ => None,
    };
    text.lines()
        .map(|l| match marker {
            Some(m) if l.trim_start().starts_with(m) => "",
            _ => l,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Where a Python file's module docstring is (its first statement, after
/// blank lines, comments and a `#!` line), as a byte range.
fn module_docstring(text: &str) -> Option<(usize, usize)> {
    let mut at = 0;
    for line in text.split_inclusive('\n') {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            at += line.len();
            continue;
        }
        let start = at + (line.len() - line.trim_start().len());
        let quote = ["\"\"\"", "'''"]
            .into_iter()
            .find(|q| text[start..].starts_with(q))?;
        let end = text[start + 3..].find(quote).map(|e| start + 3 + e + 3)?;
        return Some((start, end));
    }
    None
}

#[test]
fn only_the_module_docstring_is_blanked() {
    let py = "#!/usr/bin/env python3\n# a comment\n\"\"\"Credits Harbor.\n\nMore.\"\"\"\n\nclass T:\n    \"\"\"Read a file the way Harbor does.\"\"\"\n";
    let out = without_comments(py, "py");
    assert!(!out.contains("Credits"), "{out}");
    assert!(out.contains("Read a file the way Harbor does."), "{out}");
    assert_eq!(out.lines().count(), py.lines().count());
    // A file that starts with code has no module docstring.
    assert_eq!(module_docstring("x = 1\n\"\"\"not one\"\"\"\n"), None);
}

/// Files that are tests, or the guards' own lists, by path under `crates/`.
const KEPT_FILES: &[&str] = &[
    "benchflow-cli/src/names.rs",
    "benchflow-cli/src/other_frameworks.rs",
    "benchflow-cli/src/fuzz_tests.rs",
];

/// Literals kept on purpose: (file, the literal's whole value, how many
/// times it appears there, why).
const KEPT: &[(&str, &str, usize, &str)] = &[
    (
        "benchflow-agent/src/external/backend.rs",
        "KAROTTE_BACKEND_TOKEN_PATH",
        1,
        "TOKEN_PATH_ENV_OLD: the token variable's older name, still read",
    ),
    (
        "benchflow-run/src/external_agent.rs",
        "karotte-external",
        1,
        "OLD_NAMES: the external agent's older name, still read",
    ),
    (
        "benchflow-run/src/external_agent.rs",
        "external:karotte-backend-protocol",
        1,
        "OLD_FINGERPRINT: what earlier builds recorded in bench.lock",
    ),
    (
        "benchflow-task/src/convert/export.rs",
        "harbor",
        1,
        "export-report.json's `to`, as convert.py writes it",
    ),
    (
        "benchflow-task/src/taskmd/vocab.rs",
        "harbor",
        1,
        "SPLIT_RECORD: the [import.harbor] identifier convert.py writes and other tools read",
    ),
    (
        "benchflow-task/src/convert/rewardkit/mod.rs",
        r"harbor-rewardkit(?:\[[\w,\s-]*\])?(?P<spec>\s*(?:===|==|~=|!=|<=|>=|<|>)\s*[\w.*+!-]+(?:\s*,\s*(?:===|==|~=|!=|<=|>=|<|>)\s*[\w.*+!-]+)*)?",
        1,
        "PIN: RewardKit's package name in the pip spec a task's test.sh installs",
    ),
    (
        "benchflow-task/src/convert/rewardkit/mod.rs",
        r"harbor-rewardkit\s*@\s*git\+https://github\.com/harbor-framework/harbor(?:\.git)?@(?P<rev>[\w.-]+)",
        1,
        "GIT: RewardKit's git install spec",
    ),
    (
        "benchflow-task/src/convert/rewardkit/mod.rs",
        "harbor@{}",
        1,
        "RewardKit's install spec, written back on export",
    ),
    (
        "benchflow-task/src/convert/rewardkit/mod.rs",
        "harbor@",
        1,
        "RewardKit's install spec, recognized on export",
    ),
    (
        "benchflow-task/src/convert/rewardkit/mod.rs",
        "harbor-rewardkit{spec}",
        1,
        "RewardKit's pip spec, written back on export",
    ),
    (
        "benchflow-task/src/v06/normalize.rs",
        "harbor-compatible",
        3,
        "UNLISTED_PROFILE and PROFILES: BenchFlow 0.8's profile and its category value, still read, never listed",
    ),
    (
        "benchflow-task/src/v06/normalize.rs",
        "harbor",
        1,
        "PROFILES: the `compatibility.harbor` key 0.8's profile expands to",
    ),
];
