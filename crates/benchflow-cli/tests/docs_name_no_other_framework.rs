//! The documentation names no other evaluation, environment or training
//! framework, and no other hosted environment hub.
//!
//! What is read: every `.md` file git tracks, or has untracked and not
//! ignored, except test fixtures (`fixtures/` folders) and benchmark task
//! data (`benchmarks/*/tasks/`), plus every such file under `docs/`,
//! `.agents/skills/` and the templates `bench new` copies
//! (`crates/benchflow-task/templates/`). Outside a git work tree the folder
//! is walked instead. Strings bench prints are out of scope here.
//!
//! What is caught: the names in [`NAMES`], in any case, and the short forms
//! in [`CASED`], case-sensitively. Markdown prose is read a paragraph at a
//! time (a blockquote without its markers), so a name split by a line
//! break, a no-break space, `&nbsp;` or two spaces is still caught;
//! invisible characters are dropped first. A name counts only where no
//! ASCII letter or digit comes just before it, so text in another script
//! glued to a name is caught too. `Inspect` and `Verifiers` count in
//! headings, links, table cells, list items, parentheses, before
//! punctuation and before the verbs in [`VERBS`], but not as an imperative
//! ("Inspect a run") or a plural ("Verifiers run after the agent").
//!
//! What is allowed: code (a code span or fenced block in markdown, or any
//! line of a file that is not markdown) may show a name only inside one
//! token that is an identifier users type or tools read: one matching
//! [`IDENTIFIERS`] in full, or a bare value right after its own flag in
//! [`VALUES`] (in markdown, on a line that shows that flag as a code span);
//! inside fences only, also the lines in [`FENCE_LINES`]. A name found
//! across two tokens is always a finding.
//!
//! Integrations BenchFlow uses and users name as values stay allowed as
//! code: the `verifiers` package and its `vf-eval` CLI, the `prime-rl`
//! backend, the `prime-sft` and `trl-sft` formats, and the `openreward:` and
//! `primeintellect:` hub prefixes. Lowercase `verifiers`, `vf-eval` and `trl`
//! are not names here, so they pass in prose too; the rest pass only as the
//! identifiers listed.
//!
//! Known limits, by decision (detecting names in free text is open-ended):
//! - a code span that is a name on its own (`` `Inspect` ``, `` `prime env
//!   push` ``, `prime` or `uv pip install verifiers` in a fence): the short
//!   names are checked in prose only;
//! - prose in files that are not markdown (templates such as
//!   `task.md.template`, notebooks' markdown cells, HTML, comments and
//!   docstrings): every line is read as code, so the short names and
//!   names wrapped over two lines pass there;
//! - other case, plural and markup forms: "PRIME", "the inspect library",
//!   "FineEnvs", "**Inspect** AI", "Harbour", "OpenAI's Evals";
//! - "Inspect and ..." and "Inspect tasks" at the start of a heading or
//!   list item, which also read as the imperative the docs use;
//! - names not in [`NAMES`] of frameworks bench does not integrate with
//!   (OpenEnv, verl, SkyRL, Tinker, Atropos, EleutherAI's harness,
//!   simple-evals, the Environments Hub by its product name);
//! - fence edge cases: a fence indented four or more spaces opens a fence
//!   here, and a fence left open in a list item stays open after the list,
//!   so prose after either is read as code;
//! - lookalike letters from other scripts.

use std::path::{Path, PathBuf};
use std::process::Command;

use regex::Regex;

/// Names matched in any case: other frameworks, libraries, trainers and
/// hosted hubs. Patterns are regexes over text whose whitespace runs are
/// single spaces or line breaks.
const NAMES: &[&str] = &[
    r"harbor",
    r"karotte",
    r"prime[\s_-]*intellect",
    r"prime[\s_-]*(rl|sft)",
    r"inspect[\s_-]*(ai|evals)\b",
    r"open[\s_-]*reward",
    r"reward[\s_-]*kit",
    r"openai[\s/_-]*evals",
    r"lm[\s_-]*eval",
    r"light[\s_-]*eval",
    r"opencompass",
    r"oddish",
    r"fine[_-]?env\b",
    r"verifiers[\s-](library|package|shaped)\b",
    r"agent[\s_-]*beats",
    // The two short names' own sites.
    r"inspect\.aisi\.org\.uk",
    r"inspect\.ai-safety-institute\.org\.uk",
    r"ukgovernmentbeis",
    r"willccbb/verifiers",
];

/// Short names matched case-sensitively as whole words. `Prime` and `TRL`
/// always count (so `trl-sft`, the format, passes). `Inspect` and
/// `Verifiers` are also English words (an imperative, a plural), so they
/// count only where an imperative or a plural would not stand: see
/// [`short_name_counts`].
const CASED: &[(&str, bool)] = &[
    ("Prime", true),
    ("TRL", true),
    ("Inspect", false),
    ("Verifiers", false),
];

/// Words after which `Inspect` or `Verifiers` is the framework, as in
/// "Inspect can only copy".
const VERBS: &[&str] = &[
    "can",
    "cannot",
    "can't",
    "could",
    "is",
    "isn't",
    "was",
    "has",
    "does",
    "doesn't",
    "will",
    "won't",
    "would",
    "ships",
    "uses",
    "provides",
    "supports",
    "runs",
    "also",
    "only",
    "lets",
    "needs",
    "requires",
    "stores",
    "writes",
    "treats",
    "expects",
    "forks",
    "copies",
    "environments",
    "vs.",
    "vs",
    "versus",
    "compatibility",
    "integration",
    "interop",
    "support",
];

/// Characters after which (spaces between allowed) `Inspect` or `Verifiers`
/// is a name: an opening bracket, quote or dash.
const OPENERS: &[char] = &['(', '[', '"', '\u{201c}', '\u{2014}', '\u{2013}', '<', '>'];

/// Characters glued just before `Inspect` or `Verifiers` that make it a
/// name: "non-Inspect", "bench-to-Inspect", "**Inspect**".
const GLUED: &[char] = &['-', '/', '*', '_'];

/// Characters that, as the next one after spaces, make `Inspect` or
/// `Verifiers` a name: the end of link text, a table cell, a parenthesis,
/// punctuation, or emphasis.
const CLOSERS: &[char] = &[
    ']', '|', ')', '(', ',', '.', '*', '_', '"', '\u{201d}', '-', ':', ';', '!', '?', '<',
];

/// Words that contain a name but mean something else, allowed anywhere: a
/// place, in the bibliographies a skill example ships.
const NOT_A_NAME: &[&str] = &[r"Cold\s+Spring\s+Harbor"];

/// Identifiers allowed in code, each matched against a whole token (or a
/// whole code span), case and all.
const IDENTIFIERS: &[&str] = &[
    // The profile name 0.8's task.md standard reads.
    r"harbor-compatible",
    // Where a dataset lives.
    r"(https://github\.com/|\.\./\.\./)?harbor-framework/terminal-bench-2(/[$A-Za-z0-9_.-]*)?",
    // The registry 0.8's `bench hub check` reads by default.
    r"https://raw\.githubusercontent\.com/harbor-framework/harbor/main/registry\.json",
    // BenchFlow 0.8's Python API, cache folder and release-test tooling.
    r"(benchflow\.adapters\.)?HarborAdapter",
    r"build_harbor_roundtrip_conformance_report(\(\))?",
    r"(tests/integration/)?check_skillsbench_harbor_parity(\.py)?",
    r"skillsbench-harbor-parity",
    r"--execute-skillsbench-harbor-parity",
    r"--skillsbench-harbor-(task|benchflow-root|baseline-root)",
    r"--harbor-baseline-root",
    r"\.cache/hub/harbor",
    // Hosted environments' ids and their package index.
    r"primeintellect/[a-z0-9_-]+",
    r"primeintellect:primeintellect/[a-z0-9_-]+@[0-9a-z.]+",
    r"harbor:[a-z0-9_-]+/[a-z0-9_-]+@[0-9a-f]{7,40}",
    r"openreward:[A-Za-z0-9_-]+/[A-Za-z0-9_-]+@[0-9a-f-]+",
    r"(openreward|primeintellect):",
    r"https://hub\.primeintellect\.ai/primeintellect/simple/",
    // 0.8's trainer backend and format values, flags, paths and functions.
    r"prime-sft",
    r"--prime-rl-dir",
    r"(<work-dir>/|train-runs/[a-z0-9-]+/|\.local/)?prime-rl(-output|-dataset)?/?",
    r"benchflow/env0-prime-sft",
    r"(benchflow\.trajectories\.)?export_prime_sft(\.export_prime_sft_jsonl(\([a-z_]*\))?)?",
    // The verifier strategy 0.8's task.md standard names, and its files.
    r"reward-kit",
    r"reward-kit-manifest\.json",
    r"(verifier\.)?reward_kit(\.metric)?:?",
    r"rewardkit:?",
];

/// Values allowed as a bare token right after their flag (or, in markdown,
/// on a line that shows the flag as its own code span): the values of
/// 0.8's `--target`, `--provider` and `--owner`.
const VALUES: &[(&str, &[&str])] = &[
    ("--target", &["harbor"]),
    ("--provider", &["harbor", "primeintellect"]),
    ("--owner", &["primeintellect"]),
];

/// The values [`VALUES`] allows after `flag`.
fn values_of(flag: &str) -> &'static [&'static str] {
    VALUES
        .iter()
        .find(|(f, _)| *f == flag)
        .map_or(&[], |(_, v)| *v)
}

/// Lines of a fenced block allowed as they are (trimmed): the compatibility
/// key and source value of 0.8's task.md standard.
const FENCE_LINES: &[&str] = &["harbor:", "source: harbor"];

/// Characters dropped before matching: soft hyphen, zero-width spaces and
/// joiners, word joiner, byte-order mark.
const INVISIBLE: &[char] = &[
    '\u{ad}', '\u{200b}', '\u{200c}', '\u{200d}', '\u{2060}', '\u{feff}',
];

struct Matchers {
    names: Vec<Regex>,
    not_a_name: Vec<Regex>,
    identifiers: Vec<Regex>,
}

fn matchers() -> &'static Matchers {
    static M: std::sync::OnceLock<Matchers> = std::sync::OnceLock::new();
    M.get_or_init(|| Matchers {
        names: NAMES
            .iter()
            .map(|p| Regex::new(&format!("(?i){p}")).unwrap())
            .collect(),
        not_a_name: NOT_A_NAME.iter().map(|p| Regex::new(p).unwrap()).collect(),
        identifiers: IDENTIFIERS
            .iter()
            .map(|p| Regex::new(&format!("^(?:{p})$")).unwrap())
            .collect(),
    })
}

/// Markup that stands for a space: no-break space entities and line breaks
/// in table cells.
const SPACE_MARKUP: &[&str] = &[
    "&nbsp;", "&#160;", "&#xa0;", "&#xA0;", "<br>", "<br/>", "<br />",
];

/// Drop invisible characters, and turn space markup and every whitespace
/// character other than a line break into a space.
fn normalize(text: &str) -> String {
    let mut text = text.to_string();
    for m in SPACE_MARKUP {
        text = text.replace(m, " ");
    }
    text.chars()
        .filter(|c| !INVISIBLE.contains(c))
        .map(|c| {
            if c.is_whitespace() && c != '\n' {
                ' '
            } else {
                c
            }
        })
        .collect()
}

fn ascii_word(c: Option<char>) -> bool {
    c.is_some_and(|c| c.is_ascii_alphanumeric())
}

/// The names `text` mentions, each with the byte offset where it starts.
/// `text` is already normalized.
fn names_at(text: &str, cased: bool) -> Vec<(usize, String)> {
    let m = matchers();
    let mut text = text.to_string();
    for re in &m.not_a_name {
        text = re
            .replace_all(&text, |c: &regex::Captures| {
                c[0].chars()
                    .map(|ch| if ch == '\n' { '\n' } else { ' ' })
                    .collect::<String>()
            })
            .into_owned();
    }
    let mut out = Vec::new();
    let mut spans = Vec::new();
    for re in &m.names {
        for hit in re.find_iter(&text) {
            if !ascii_word(text[..hit.start()].chars().next_back()) {
                out.push((hit.start(), hit.as_str().replace('\n', " ")));
                spans.push(hit.range());
            }
        }
    }
    if cased {
        for (word, always) in CASED {
            for (i, _) in text.match_indices(word) {
                let before = text[..i].chars().next_back();
                let after = text[i + word.len()..].chars().next();
                if ascii_word(before) || ascii_word(after) || spans.iter().any(|r| r.contains(&i)) {
                    continue;
                }
                if *always || short_name_counts(&text[..i], &text[i + word.len()..]) {
                    out.push((i, word.to_string()));
                }
            }
        }
    }
    out.sort();
    out
}

/// Whether `Inspect` or `Verifiers`, between `before` and `after`, is the
/// framework rather than an imperative ("Inspect the logs") or a plural
/// ("Verifiers run after the agent"): possessive; mid-sentence (after a
/// letter, digit or comma); after `vs.`, `e.g.`, `i.e.` or an opener; glued
/// to a dash, slash or emphasis; or followed by one of [`VERBS`], by one of
/// [`CLOSERS`], or by nothing (the end of a heading, list item or cell).
fn short_name_counts(before: &str, after: &str) -> bool {
    let glued = before.chars().next_back();
    let prev_text = before.trim_end();
    let prev = prev_text.chars().next_back();
    let prev_word = prev_text.rsplit(char::is_whitespace).next().unwrap_or("");
    let rest = after.trim_start();
    let next = rest.chars().next();
    let next_word = rest.split_whitespace().next().unwrap_or("");
    matches!(after.chars().next(), Some('\'' | '\u{2019}'))
        || prev.is_some_and(|c| c.is_ascii_alphanumeric() || c == ',' || OPENERS.contains(&c))
        || ["vs.", "e.g.", "i.e."].contains(&prev_word.to_lowercase().as_str())
        || glued.is_some_and(|c| GLUED.contains(&c))
        || next.is_none_or(|c| CLOSERS.contains(&c))
        || VERBS.contains(&next_word.trim_end_matches([',', '.', ';', ':']))
}

/// Whether `token` (one word of code) is an allowed identifier, or one of
/// `values` (the bare values its flag allows).
fn allowed_token(token: &str, values: &[&str]) -> bool {
    values.contains(&token) || matchers().identifiers.iter().any(|re| re.is_match(token))
}

/// Characters that end a token of code.
fn separator(c: char) -> bool {
    c.is_whitespace() || "\"'`,;|()[]{}".contains(c)
}

/// The tokens of `code`, each with its byte range.
fn tokens(code: &str) -> Vec<(std::ops::Range<usize>, &str)> {
    let mut out = Vec::new();
    let mut start = None;
    for (i, c) in code.char_indices().chain([(code.len(), ' ')]) {
        match (start, separator(c)) {
            (None, false) => start = Some(i),
            (Some(s), true) => {
                out.push((s..i, &code[s..i]));
                start = None;
            }
            _ => {}
        }
    }
    out
}

/// The names a piece of code shows that are not allowed identifiers.
/// Names are found on the whole line, so one written as two words is
/// caught; a hit passes only when it lies inside one token and that token
/// is an allowed identifier, or a value its flag allows. `line_values` are
/// values allowed anywhere in this code (their flag is on the same line).
fn names_in_code(code: &str, line_values: &[&str]) -> Vec<String> {
    let code = normalize(code);
    let toks = tokens(&code);
    let mut out = Vec::new();
    for (at, name) in names_at(&code, false) {
        let hit = at..at + name.len();
        let Some(k) = toks
            .iter()
            .position(|(r, _)| r.start <= hit.start && hit.end <= r.end)
        else {
            out.push(name);
            continue;
        };
        let token = toks[k].1;
        let prev = k.checked_sub(1).map_or("", |j| toks[j].1);
        let (value, values) = match token.split_once('=') {
            Some((flag, v)) if !values_of(flag).is_empty() => (v, values_of(flag)),
            _ => (token, values_of(prev)),
        };
        if !(allowed_token(value, values) || allowed_token(value, line_values)) {
            out.push(name);
        }
    }
    out
}

/// The values allowed on a markdown line: those of each flag the line shows
/// as a code span of its own, or as the start of one (`` `--provider` ``,
/// `` `--provider x` ``, `` `--provider=x` ``).
fn line_values(line: &str) -> Vec<&'static str> {
    VALUES
        .iter()
        .filter(|(flag, _)| {
            line.match_indices(&format!("`{flag}"))
                .any(|(i, m)| matches!(line[i + m.len()..].chars().next(), Some('`' | ' ' | '=')))
        })
        .flat_map(|(_, v)| v.iter().copied())
        .collect()
}

/// A code span's names: the whole span must be one allowed identifier,
/// unless it is a command line or a flag, which is read token by token.
fn names_in_span(span: &str, line: &str) -> Vec<String> {
    let span = normalize(span).replace('\n', " ");
    let span = span.trim();
    let values = line_values(line);
    let command = span.starts_with("bench ") || span.starts_with("--");
    if command {
        return names_in_code(span, &values);
    }
    let found = names_at(span, false);
    if found.is_empty() || allowed_token(span, &values) {
        return Vec::new();
    }
    found.into_iter().map(|(_, n)| n).collect()
}

/// A CommonMark fence opener or closer: the character, the run length and
/// the rest of the line.
fn fence(line: &str) -> Option<(char, usize, &str)> {
    let t = line.trim_start();
    let c = t.chars().next().filter(|c| *c == '`' || *c == '~')?;
    let n = t.chars().take_while(|x| *x == c).count();
    let rest = &t[n..];
    (n >= 3 && !(c == '`' && rest.contains('`'))).then_some((c, n, rest))
}

/// Lines that start a block of their own in markdown, so a code span or
/// a wrapped name never runs across them.
fn starts_block(line: &str) -> bool {
    let t = line.trim_start();
    let digits = t.chars().take_while(char::is_ascii_digit).count();
    t.starts_with('#')
        || t.starts_with('|')
        || ["- ", "* ", "+ "].iter().any(|m| t.starts_with(m))
        || (digits > 0 && (t[digits..].starts_with(". ") || t[digits..].starts_with(") ")))
}

/// The findings in one markdown paragraph that starts on line `first`
/// (1-based): code spans by backtick runs, the rest as prose.
fn paragraph(lines: &[&str], first: usize, out: &mut Vec<String>) {
    let text = lines.join("\n");
    let line_at = |i: usize| first + text[..i].matches('\n').count();
    let bytes = text.as_bytes();
    let mut prose = text.clone().into_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'\\' {
            i += 2;
            continue;
        }
        if bytes[i] != b'`' {
            i += 1;
            continue;
        }
        let n = bytes[i..].iter().take_while(|b| **b == b'`').count();
        let mut j = i + n;
        let close = loop {
            let Some(k) = bytes[j..].iter().position(|b| *b == b'`') else {
                break None;
            };
            let k = j + k;
            let m = bytes[k..].iter().take_while(|b| **b == b'`').count();
            if m == n {
                break Some(k);
            }
            j = k + m;
        };
        let Some(k) = close else {
            i += n; // a literal run of backticks
            continue;
        };
        let n_line = line_at(i);
        let line = lines[n_line - first];
        for name in names_in_span(&text[i + n..k], line) {
            out.push(format!("{n_line}: {name} (code span)"));
        }
        // Blank the span in the prose view, keeping its line breaks and a
        // word in its place.
        for (o, b) in prose[i..k + n].iter_mut().enumerate() {
            if *b != b'\n' {
                *b = if o == 0 { b'x' } else { b' ' };
            }
        }
        i = k + n;
    }
    let prose = normalize(&String::from_utf8_lossy(&prose));
    for (at, name) in names_at(&prose, true) {
        let n_line = first + prose[..at].matches('\n').count();
        out.push(format!("{n_line}: {name} (prose)"));
    }
}

fn flush(para: &mut Vec<&str>, start: usize, out: &mut Vec<String>) {
    if !para.is_empty() {
        paragraph(para, start, out);
        para.clear();
    }
}

/// Every name `text` mentions, as `line: name (where)`. A markdown file is
/// split into fenced blocks, code spans and prose; any other file is code
/// throughout.
fn findings(text: &str, markdown: bool) -> Vec<String> {
    let mut out = Vec::new();
    let lines: Vec<&str> = text.lines().collect();
    if !markdown {
        for (n, line) in lines.iter().enumerate() {
            for name in names_in_code(line, &[]) {
                out.push(format!("{}: {name} (code)", n + 1));
            }
        }
        return out;
    }
    let mut open: Option<(char, usize)> = None;
    let mut para: Vec<&str> = Vec::new();
    let mut para_start = 0;
    let mut in_quote = false;
    for (n, line) in lines.iter().enumerate() {
        let n1 = n + 1;
        // A blockquote's lines are read without their markers, and a quote
        // starts and ends a paragraph of its own.
        let (quoted, line) = unquote(line);
        if quoted != in_quote {
            flush(&mut para, para_start, &mut out);
            in_quote = quoted;
        }
        if let Some((c, len)) = open {
            let closes = fence(line)
                .is_some_and(|(fc, fl, rest)| fc == c && fl >= len && rest.trim().is_empty());
            if closes {
                open = None;
            } else if !FENCE_LINES.contains(&line.trim()) {
                for name in names_in_code(line, &[]) {
                    out.push(format!("{n1}: {name} (code block)"));
                }
            }
            continue;
        }
        if let Some((c, len, info)) = fence(line) {
            flush(&mut para, para_start, &mut out);
            open = Some((c, len));
            for (_, name) in names_at(&normalize(info), true) {
                out.push(format!("{n1}: {name} (fence)"));
            }
            continue;
        }
        if line.trim().is_empty() || starts_block(line) {
            flush(&mut para, para_start, &mut out);
        }
        if line.trim().is_empty() {
            continue;
        }
        if para.is_empty() {
            para_start = n1;
        }
        para.push(line);
        let t = line.trim_start();
        if t.starts_with('#') || t.starts_with('|') {
            flush(&mut para, para_start, &mut out);
        }
    }
    flush(&mut para, para_start, &mut out);
    out
}

/// A line without its blockquote markers (`>` and one optional space, at
/// any depth), and whether it had any.
fn unquote(line: &str) -> (bool, &str) {
    let mut rest = line;
    let mut quoted = false;
    while let Some(r) = rest.trim_start().strip_prefix('>') {
        quoted = true;
        rest = r.strip_prefix(' ').unwrap_or(r);
    }
    (quoted, if quoted { rest } else { line })
}

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

/// Whether the guard reads `path` (relative to the repository, `/`-separated).
fn in_scope(path: &str) -> bool {
    let fixture = path.split('/').any(|part| part == "fixtures");
    let task_data = path.starts_with("benchmarks/") && path.split('/').nth(2) == Some("tasks");
    let tree = [
        "docs/",
        ".agents/skills/",
        "crates/benchflow-task/templates/",
    ]
    .iter()
    .any(|d| path.starts_with(d));
    (path.ends_with(".md") && !fixture && !task_data) || tree
}

/// `git` run in `dir`, with no `GIT_*` variable inherited from the caller.
/// `git bisect run`, `git rebase -x` and hooks export `GIT_DIR` (and hooks
/// `GIT_INDEX_FILE`); inherited, they would point these commands at the
/// caller's repository instead of the one `dir` is in.
fn git(dir: &Path) -> Command {
    let mut cmd = Command::new("git");
    cmd.current_dir(dir);
    for (key, _) in std::env::vars_os() {
        if key.to_string_lossy().starts_with("GIT_") {
            cmd.env_remove(key);
        }
    }
    cmd
}

/// The files this guard reads, in scope: what git tracks under `root` plus
/// untracked files it does not ignore. Outside a git work tree (a source
/// archive), every file under `root` except `.git/` and `target/`.
fn documents(root: &Path) -> Vec<PathBuf> {
    let inside = git(root)
        .args(["rev-parse", "--is-inside-work-tree"])
        .output()
        .is_ok_and(|o| o.status.success() && o.stdout.starts_with(b"true"));
    let paths: Vec<String> = if inside {
        let out = git(root)
            .args([
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
            ])
            .output()
            .expect("the docs guard runs git ls-files");
        assert!(out.status.success(), "git ls-files failed in {root:?}");
        out.stdout
            .split(|b| *b == 0)
            .filter(|p| !p.is_empty())
            .map(|p| String::from_utf8_lossy(p).into_owned())
            .collect()
    } else {
        walkdir::WalkDir::new(root)
            .into_iter()
            .filter_entry(|e| {
                e.depth() != 1 || !matches!(e.file_name().to_str(), Some(".git" | "target"))
            })
            .filter_map(Result::ok)
            .filter(|e| !e.file_type().is_dir())
            .map(|e| {
                let rel = e.path().strip_prefix(root).unwrap();
                rel.to_string_lossy().replace('\\', "/")
            })
            .collect()
    };
    let mut files: Vec<PathBuf> = paths
        .into_iter()
        .filter(|p| in_scope(p))
        .map(PathBuf::from)
        .collect();
    files.sort();
    files.dedup();
    files
}

/// The findings in one file, read lossily (the names are ASCII). A symlink
/// is refused: its target would be read twice or not at all.
fn check_file(root: &Path, file: &Path) -> Result<Vec<String>, String> {
    let path = root.join(file);
    let meta = std::fs::symlink_metadata(&path).map_err(|e| format!("{}: {e}", file.display()))?;
    if meta.file_type().is_symlink() {
        return Err(format!(
            "{} is a symlink; the docs guard refuses links",
            file.display()
        ));
    }
    let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", file.display()))?;
    let text = String::from_utf8_lossy(&bytes);
    let markdown = file.extension().is_some_and(|e| e == "md");
    Ok(findings(&text, markdown)
        .into_iter()
        .map(|f| format!("{}:{f}", file.display()))
        .collect())
}

#[test]
fn the_docs_name_no_other_framework() {
    let root = repo();
    let files = documents(&root);
    let mut found = Vec::new();
    let mut read = 0;
    for file in &files {
        match check_file(&root, file) {
            Ok(f) => {
                read += 1;
                found.extend(f);
            }
            Err(e) => found.push(e),
        }
    }
    // Findings (and files that could not be read) first, so a failure
    // names the file.
    assert!(found.is_empty(), "{}", found.join("\n"));
    assert!(
        read > 300 && read == files.len(),
        "read {read} of {} files",
        files.len()
    );
    for must in [
        "README.md",
        "CHANGELOG.md",
        "tests/e2e/README.md",
        "docs/rust/README.md",
        "crates/benchflow-task/templates/split/task.toml",
        ".agents/skills/task-creator/assets/task.md.template",
    ] {
        assert!(files.contains(&PathBuf::from(must)), "{must} is not read");
    }
}

#[test]
fn the_guard_catches_prose_and_unlisted_code() {
    let md = |t: &str| findings(t, true);
    for caught in [
        // Caught before.
        "Runs Harbor tasks.",
        "the harbor registry",
        "Converted from Karotte's format",
        "an Inspect AI task",
        "a PrimeIntellect hub",
        "`Harbor`",
        "`harbor run -p tasks`",
        "names [import.harbor] in prose",
        "see https://github.com/harbor-framework/harbor",
        "```\n# regenerate Harbor inventory\n```",
        "```\nharbor run\n```",
        // Missed before: short names and the other frameworks.
        "so nothing is uploaded to Prime",
        "into Prime SFT-ready JSONL",
        "The OpenReward selections",
        "the Verifiers record",
        "a Verifiers/ORS record",
        "that prime-rl ingests",
        "the native Prime-RL SFT trainer",
        "Inspect's `fork()` deep-copies",
        "Inspect can only copy",
        "the library Inspect ships",
        "full Reward Kit parity",
        "use the verifiers library",
        "the OpenAI Evals suite",
        "an LM Evaluation Harness run",
        // Missed before: names split by a line break or odd spaces.
        "a Prime\nIntellect environment",
        "a Prime\u{a0}Intellect environment",
        "an Inspect  AI task",
        "an Inspect\nEvals task",
        "- runs Open\n  Reward environments",
        // Missed before: invisible characters and text in other scripts.
        "a Har\u{ad}bor task",
        "Prime\u{200b}Intellect",
        "迁移自Harbor的任务",
        "从PrimeIntellect导入",
        // Missed before: code spans that read as prose.
        "Tasks that `harbor` runs also run here",
        "We beat `primeintellect` on every hub",
        "`primeintellect/verifiers is what we copied`",
        "`harbor:is the reference runner`",
        "`HarborAdapterFactory`",
        "`skillsbench-harbor-anything`",
        "the key `harbor:` inline",
        "inline `source: harbor`",
        // Missed before: fences that do not open or close where a line
        // starting with backticks or tildes suggests.
        "````\n```\n````\nThe HarborAdapter is prose here",
        "```\n~~~\n```\nThe HarborAdapter is prose here",
        "```bash``` is inline\nThe HarborAdapter is prose here",
        "Use a ` (one backtick) with HarborAdapter.",
    ] {
        assert!(!md(caught).is_empty(), "missed: {caught:?}");
    }
    for allowed in [
        "the split layout",
        "the `harbor-compatible` profile",
        "`--target` | `harbor` |",
        "`--provider`: `primeintellect` (hosted) and `harbor` (the registry)",
        "`bench hub list --provider harbor`",
        "`bench hub list --provider=primeintellect --owner primeintellect`",
        "`harbor:terminal-bench/adaptive-rejection-sampler@69671fba`",
        "`primeintellect/general-agent`",
        "`HarborAdapter` reads a split-layout directory",
        "```yaml\ncompat:\n  harbor:\n  source: harbor\n```",
        "````\n```\nHarborAdapter\n```\n````",
        "Cold Spring Harbor Laboratory Press",
        "Cold Spring\nHarbor Laboratory Press",
        "an llm-eval loop",
        "`--backend prime-rl` with `--prime-rl-dir .local/prime-rl`",
        "the `prime-sft` and `trl-sft` formats",
        "the `verifiers` package and its `vf-eval`",
        "Inspect the trace before you trust it.",
        "1. Inspect the logs",
        "Verifiers run after the agent.",
        "the verifiers run in their own sandbox",
        "a prime number, a primary key",
        "an `ors-episode` verifier and the `ORSAdapter`",
        "a `reward-kit` verifier",
    ] {
        assert!(
            md(allowed).is_empty(),
            "flagged: {allowed:?}: {:?}",
            md(allowed)
        );
    }
    assert!(!findings("publisher = {Harbor Press}", false).is_empty());
    assert!(!findings("# exports Prime-RL rows", false).is_empty());
    assert!(findings("repo: harbor-framework/terminal-bench-2", false).is_empty());
    assert!(findings("address = {Cold Spring Harbor, NY}", false).is_empty());
    assert!(findings("bench hub list --provider harbor", false).is_empty());
    // Line numbers count from the paragraph's first line.
    assert_eq!(md("one\n\ntwo\nthree Harbor"), ["4: Harbor (prose)"]);
    assert_eq!(md("a Prime\nIntellect hub"), ["1: Prime Intellect (prose)"]);
    assert_eq!(md("Cold Spring\nHarbor and\nHarbor"), ["3: Harbor (prose)"]);
}

#[test]
fn the_guard_catches_names_in_code_quotes_headings_and_links() {
    let md = |t: &str| findings(t, true);
    let code = |t: &str| findings(t, false);
    for caught in [
        // Two-word names in code (a regression before).
        "```\n# pull from the Prime Intellect hub\n```",
        "```\n# port of an Inspect AI task\n```",
        "```\n# OpenAI Evals format\n```",
        "`bench hub list # Prime Intellect`",
        // AgentBeats and TRL.
        "AgentBeats-style assessor agents",
        "the TRL exporter",
        "TRL's `GRPOTrainer`",
        // Inspect and Verifiers outside mid-sentence prose.
        "See [Inspect](https://example.org/) for details.",
        "See https://inspect.aisi.org.uk/ for details.",
        "Built on https://github.com/willccbb/verifiers today.",
        "## Inspect",
        "## Verifiers",
        "## BenchFlow vs. Inspect",
        "## Inspect-style tasks",
        "| Inspect | yes |",
        "- Verifiers",
        "e.g. Inspect.",
        "(Inspect)",
        "**Inspect** forks",
        "a non-Inspect task",
        "Inspect doesn't fork.",
        "Inspect lets you",
        "Verifiers environments run unchanged.",
        // Blockquotes wrapped over lines, and space markup.
        "> runs Open\n> Reward environments",
        "> the OpenAI\n> Evals suite",
        "an Open&nbsp;Reward environment",
        "| Open<br>Reward | yes |",
        // A bare value away from its own flag.
        "Set `--provider-url`; then `harbor` runs it.",
        "`--owner` | `harbor` |",
    ] {
        assert!(!md(caught).is_empty(), "missed: {caught:?}");
    }
    for caught in [
        "description = \"Converted from an Inspect AI task\"",
        "\"\"\"Export rows for the Prime Intellect hub.\"\"\"",
        "bench hub list --owner harbor",
    ] {
        assert!(!code(caught).is_empty(), "missed: {caught:?}");
    }
    for allowed in [
        "## Inspect before uploading",
        "| Inspect a run |",
        "### Step 4: Inspect results",
        "Inspect them; then fix the task.",
        "Inspect the available tools first.",
        "> Inspect the logs\n> before you rerun.",
        "the `trl-sft` format",
        "`bench hub list --provider harbor --owner primeintellect`",
        "`--target` | `harbor` |",
    ] {
        assert!(
            md(allowed).is_empty(),
            "flagged: {allowed:?}: {:?}",
            md(allowed)
        );
    }
    assert!(code("bench hub list --provider=harbor").is_empty());
}

#[cfg(unix)]
#[test]
fn the_guard_reads_tracked_files_lossily_and_refuses_links() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let run = |args: &[&str]| {
        let ok = git(root).args(args).output().unwrap().status.success();
        assert!(ok, "git {args:?}");
    };
    run(&["init", "-q"]);
    std::fs::create_dir_all(root.join("docs")).unwrap();
    std::fs::create_dir_all(root.join("crates/x/tests/fixtures")).unwrap();
    std::fs::write(
        root.join("docs/latin1.md"),
        b"Runs Harbor tasks \xe9t\xe9.\n",
    )
    .unwrap();
    std::fs::write(root.join("docs/notes.md"), "Runs Harbor tasks.\n").unwrap();
    std::fs::write(root.join("docs/ignored.md"), "Runs Harbor tasks.\n").unwrap();
    std::fs::write(root.join(".gitignore"), "ignored.md\n").unwrap();
    std::fs::write(root.join("crates/x/tests/fixtures/README.md"), "Harbor\n").unwrap();
    std::fs::write(root.join("guide.md"), "fine\n").unwrap();
    std::os::unix::fs::symlink("notes.md", root.join("docs/link.md")).unwrap();
    run(&["add", "docs/latin1.md", "docs/link.md", "guide.md"]);
    run(&["add", "-f", "crates/x/tests/fixtures/README.md"]);
    // docs/notes.md is untracked but not ignored, so it is read too;
    // docs/ignored.md is ignored, and fixtures are out of scope.
    let files = documents(root);
    assert_eq!(
        files,
        [
            "docs/latin1.md",
            "docs/link.md",
            "docs/notes.md",
            "guide.md"
        ]
        .map(PathBuf::from)
    );
    let latin1 = check_file(root, Path::new("docs/latin1.md")).unwrap();
    assert_eq!(latin1, ["docs/latin1.md:1: Harbor (prose)"]);
    let link = check_file(root, Path::new("docs/link.md")).unwrap_err();
    assert!(link.contains("docs/link.md"), "{link}");
    let gone = check_file(root, Path::new("docs/gone.md")).unwrap_err();
    assert!(gone.contains("docs/gone.md"), "{gone}");
}

/// Outside a git work tree (a source archive), the guard walks the folder.
#[test]
fn the_guard_walks_a_folder_outside_git() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    for f in ["docs/a.md", "guide.md", "target/x.md", "src/lib.rs"] {
        std::fs::create_dir_all(root.join(f).parent().unwrap()).unwrap();
        std::fs::write(root.join(f), "fine\n").unwrap();
    }
    assert_eq!(
        documents(root),
        ["docs/a.md", "guide.md"].map(PathBuf::from)
    );
}

/// With `GIT_DIR`, `GIT_WORK_TREE` and `GIT_INDEX_FILE` pointing at another
/// repository (as `git bisect run`, `git rebase -x` and hooks set them),
/// the git test above leaves that repository untouched.
#[cfg(unix)]
#[test]
fn inherited_git_variables_never_reach_git() {
    let dir = tempfile::tempdir().unwrap();
    let decoy = dir.path();
    let run = |args: &[&str]| git(decoy).args(args).output().unwrap();
    assert!(run(&["init", "-q"]).status.success());
    std::fs::write(decoy.join("kept.md"), "kept\n").unwrap();
    assert!(run(&["add", "kept.md"]).status.success());
    let state = || {
        let bare = run(&["config", "--get", "core.bare"]).stdout;
        let index = run(&["ls-files", "--stage"]).stdout;
        (bare, index)
    };
    let before = state();
    let out = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "the_guard_reads_tracked_files_lossily_and_refuses_links",
            "--test-threads",
            "1",
        ])
        .env("GIT_DIR", decoy.join(".git"))
        .env("GIT_WORK_TREE", decoy)
        .env("GIT_INDEX_FILE", decoy.join(".git/index"))
        .output()
        .unwrap();
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success() && text.contains("1 passed"),
        "{text}{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(state(), before, "the decoy repository changed");
    assert_eq!(String::from_utf8_lossy(&before.0).trim(), "false");
}
