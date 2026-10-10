//! Just enough of a shell reader to find RewardKit calls in a `test.sh`,
//! ported from task.md's `tools/convert.py` (`_sh_lines`, `_sh_code`,
//! `_sh_words`, `_sh_depths`, `_sh_expand`, `_sh_state`). Words are split
//! as Python's `shlex.shlex(posix=True, punctuation_chars=True)` with
//! `whitespace_split` splits them, which is what the reference converter
//! uses.

use std::sync::LazyLock;

use indexmap::IndexMap;
use regex::Regex;

static HEREDOC: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"<<(-?)[ \t]*(['"]?)([A-Za-z_][A-Za-z0-9_]*)"#).unwrap());
static ASSIGN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(?:export\s+|readonly\s+|declare\s+(?:-\w+\s+)*|local\s+)?([A-Za-z_]\w*)=(.*)$")
        .unwrap()
});
static VAR: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\$\{([A-Za-z_]\w*)(?:(:?)([-=+?])((?:[^{}]|\{[^{}]*\})*))?\}|\$([A-Za-z_]\w*)")
        .unwrap()
});
static FUNCTION: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^(?:function\s+)?[\w-]+\s*\(\)\s*\{?$").unwrap());

const PUNCTUATION: &str = "();<>|&";
const OPEN: [&str; 7] = ["if", "case", "for", "while", "until", "select", "{"];
const CLOSE: [&str; 4] = ["fi", "esac", "done", "}"];
const CMD_AFTER: [&str; 18] = [
    ";", ";;", "&&", "||", "|", "&", "(", ")", "{", "!", "then", "do", "else", "elif", "if",
    "while", "until", "time",
];
const TRIVIAL: [&str; 13] = [
    "set", "shopt", "mkdir", "echo", "printf", "true", ":", "trap", "umask", "cd", "export",
    "readonly", "declare",
];

fn is_ws(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\r' | '\n')
}

/// Python's `shlex` words of `text`; `None` on an unclosed quote or a
/// trailing escape.
pub(crate) fn words(text: &str) -> Option<Vec<String>> {
    #[derive(Clone, Copy, PartialEq)]
    enum St {
        Space,
        Word,
        Punct,
        Quote(char),
        Escape,
        Done,
    }
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0usize;
    let mut pushback: Option<char> = None;
    let mut next = |pb: &mut Option<char>| -> Option<char> {
        if let Some(c) = pb.take() {
            return Some(c);
        }
        let c = chars.get(i).copied();
        if c.is_some() {
            i += 1;
        }
        c
    };
    let mut out = Vec::new();
    let mut state = St::Space;
    loop {
        // One read_token call.
        let mut token = String::new();
        let mut quoted = false;
        let mut escaped_state = St::Word;
        loop {
            if state == St::Done {
                break;
            }
            let c = next(&mut pushback);
            match state {
                St::Space => match c {
                    None => {
                        state = St::Done;
                        break;
                    }
                    Some(c) if is_ws(c) => {
                        if !token.is_empty() || quoted {
                            break;
                        }
                    }
                    Some('\\') => {
                        escaped_state = St::Word;
                        state = St::Escape;
                    }
                    Some(c) if PUNCTUATION.contains(c) => {
                        token.push(c);
                        state = St::Punct;
                    }
                    Some(c) if c == '\'' || c == '"' => state = St::Quote(c),
                    Some(c) => {
                        token.push(c);
                        state = St::Word;
                    }
                },
                St::Quote(q) => {
                    quoted = true;
                    match c {
                        None => return None,
                        Some(c) if c == q => state = St::Word,
                        Some('\\') if q == '"' => {
                            escaped_state = St::Quote(q);
                            state = St::Escape;
                        }
                        Some(c) => token.push(c),
                    }
                }
                St::Escape => {
                    let c = c?;
                    if let St::Quote(q) = escaped_state
                        && c != '\\'
                        && c != q
                    {
                        token.push('\\');
                    }
                    token.push(c);
                    state = escaped_state;
                }
                St::Word | St::Punct => match c {
                    None => {
                        state = St::Done;
                        break;
                    }
                    Some(c) if is_ws(c) => {
                        state = St::Space;
                        if !token.is_empty() || quoted {
                            break;
                        }
                    }
                    Some(c) if state == St::Punct => {
                        if PUNCTUATION.contains(c) {
                            token.push(c);
                        } else {
                            pushback = Some(c);
                            state = St::Space;
                            break;
                        }
                    }
                    Some(c) if c == '\'' || c == '"' => state = St::Quote(c),
                    Some('\\') => {
                        escaped_state = St::Word;
                        state = St::Escape;
                    }
                    Some(c) if !PUNCTUATION.contains(c) => token.push(c),
                    Some(c) => {
                        pushback = Some(c);
                        state = St::Space;
                        if !token.is_empty() || quoted {
                            break;
                        }
                    }
                },
                St::Done => unreachable!("checked above"),
            }
        }
        if token.is_empty() && !quoted {
            return Some(out);
        }
        out.push(token);
    }
}

/// A line without its trailing comment.
pub(crate) fn code(line: &str) -> &str {
    let chars: Vec<(usize, char)> = line.char_indices().collect();
    let mut quote: Option<char> = None;
    let mut k = 0;
    while k < chars.len() {
        let (pos, ch) = chars[k];
        if ch == '\\' && quote != Some('\'') {
            k += 2;
            continue;
        }
        match quote {
            Some(q) => {
                if ch == q {
                    quote = None;
                }
            }
            None if ch == '\'' || ch == '"' => quote = Some(ch),
            None if ch == '#' && (k == 0 || matches!(chars[k - 1].1, ' ' | '\t' | ';')) => {
                return line[..pos].trim_end();
            }
            None => {}
        }
        k += 1;
    }
    line
}

/// One logical line of a script: continuations joined, with the body of
/// any heredoc it opens.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Line {
    /// 1-based first and last physical lines.
    pub start: usize,
    pub end: usize,
    pub text: String,
}

/// A script's logical lines.
pub(crate) fn lines(text: &str) -> Vec<Line> {
    let normalized = text.replace("\r\n", "\n");
    let raw: Vec<&str> = normalized.split('\n').collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < raw.len() {
        let start = i;
        let mut parts: Vec<String> = vec![raw[i].to_string()];
        while parts
            .last()
            .is_some_and(|p| p.ends_with('\\') && !p.trim_start().starts_with('#'))
            && i + 1 < raw.len()
        {
            let last = parts.last_mut().expect("not empty");
            last.pop();
            i += 1;
            parts.push(raw[i].to_string());
        }
        let line = parts
            .iter()
            .map(|p| p.trim())
            .collect::<Vec<_>>()
            .join(" ")
            .trim()
            .to_string();
        let stripped = code(&line).to_string();
        for m in HEREDOC.captures_iter(&stripped) {
            let quote = m.get(2).map_or("", |g| g.as_str());
            if !quote.is_empty() && !stripped[m.get(0).expect("match").end()..].starts_with(quote) {
                continue;
            }
            let tabs = !m[1].is_empty();
            let delim = &m[3];
            i += 1;
            while i < raw.len()
                && (if tabs {
                    raw[i].trim_start_matches('\t')
                } else {
                    raw[i]
                }) != delim
            {
                i += 1;
            }
        }
        out.push(Line {
            start: start + 1,
            end: i + 1,
            text: line,
        });
        i += 1;
    }
    out
}

/// How deep in if, case, loop, and brace blocks each logical line starts.
pub(crate) fn depths(lines: &[Line]) -> Vec<usize> {
    let mut depth = 0usize;
    let mut out = Vec::new();
    for ln in lines {
        out.push(depth);
        let c = code(&ln.text);
        let ws: Vec<String> = match words(c) {
            Some(w) if !w.is_empty() => w,
            _ => c.split_whitespace().map(str::to_string).collect(),
        };
        let mut prev: Option<&str> = None;
        for w in &ws {
            let command_position = match prev {
                None => true,
                Some(p) => CMD_AFTER.contains(&p) || p.chars().all(|ch| ";&|()".contains(ch)),
            };
            if command_position {
                if OPEN.contains(&w.as_str()) {
                    depth += 1;
                } else if CLOSE.contains(&w.as_str()) {
                    depth = depth.saturating_sub(1);
                }
            }
            prev = Some(w);
        }
    }
    out
}

/// A variable's value as far as the script can tell.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum EnvVal {
    Str(String),
    /// Taken from the operator's environment when Harbor starts the
    /// verifier: `${VAR}` or `${VAR:-default}`.
    Host {
        var: String,
        default: Option<String>,
    },
    /// Something nothing in the task fixes.
    Unknown,
}

/// Variables by name, in order of assignment.
pub(crate) type Env = IndexMap<String, EnvVal>;

/// `word` with `$VAR` and `${VAR:-default}` expanded; `None` when a value
/// nothing in the task fixes decides it.
pub(crate) fn expand(word: &str, env: &Env) -> Option<String> {
    let mut unknown = false;
    let mut out = String::new();
    let mut last = 0;
    for m in VAR.captures_iter(word) {
        let whole = m.get(0).expect("match");
        out.push_str(&word[last..whole.start()]);
        last = whole.end();
        let name = m.get(1).or(m.get(5)).map_or("", |g| g.as_str());
        let colon = m.get(2).is_some_and(|g| !g.as_str().is_empty());
        let op = m.get(3).map(|g| g.as_str());
        let arg = m.get(4).map_or("", |g| g.as_str());
        let mut val = env.get(name).cloned();
        if let Some(EnvVal::Host { default, .. }) = &val {
            val = Some(match default {
                Some(d) => EnvVal::Str(d.clone()),
                None => EnvVal::Unknown,
            });
        }
        let replacement = if matches!(op, Some("-") | Some("=")) {
            let missing = match &val {
                None | Some(EnvVal::Unknown) => true,
                Some(EnvVal::Str(s)) => colon && s.is_empty(),
                Some(EnvVal::Host { .. }) => unreachable!("resolved above"),
            };
            if missing {
                match expand(arg, env) {
                    Some(d) => d,
                    None => {
                        unknown = true;
                        String::new()
                    }
                }
            } else {
                match val {
                    Some(EnvVal::Str(s)) => s,
                    _ => String::new(),
                }
            }
        } else {
            match val {
                Some(EnvVal::Unknown) => {
                    unknown = true;
                    String::new()
                }
                Some(EnvVal::Str(s)) => s,
                _ => String::new(),
            }
        };
        out.push_str(&replacement);
    }
    out.push_str(&word[last..]);
    (!unknown).then_some(out)
}

/// Python's `posixpath.normpath`.
pub(crate) fn normpath(path: &str) -> String {
    if path.is_empty() {
        return ".".into();
    }
    let slashes = if path.starts_with("//") && !path.starts_with("///") {
        2
    } else if path.starts_with('/') {
        1
    } else {
        0
    };
    let mut comps: Vec<&str> = Vec::new();
    for comp in path.split('/') {
        if comp.is_empty() || comp == "." {
            continue;
        }
        if comp != ".." || (slashes == 0 && comps.is_empty()) || comps.last() == Some(&"..") {
            comps.push(comp);
        } else if !comps.is_empty() {
            comps.pop();
        }
    }
    let joined = format!("{}{}", "/".repeat(slashes), comps.join("/"));
    if joined.is_empty() {
        ".".into()
    } else {
        joined
    }
}

/// A command a script runs before the RewardKit call.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Before {
    pub command: String,
}

/// Run a script's top-level assignments and `cd` commands before line
/// `upto`: the variables then, and the other commands before it.
pub(crate) fn state(
    lines: &[Line],
    depths: &[usize],
    upto: usize,
    env: &Env,
) -> (Env, Vec<Before>) {
    let mut env = env.clone();
    let mut cwd: Option<String> = match env.get("PWD") {
        Some(EnvVal::Str(s)) => Some(s.clone()),
        _ => None,
    };
    let mut before = Vec::new();
    for n in 0..upto {
        let c = code(&lines[n].text);
        if c.is_empty() || c.starts_with("#!") || depths[n] > 0 {
            continue;
        }
        let ws: Vec<String> = match words(c) {
            Some(w) if !w.is_empty() => w,
            _ => c.split_whitespace().map(str::to_string).collect(),
        };
        if let Some(m) = ASSIGN.captures(c)
            && !m[1].contains(' ')
        {
            let value = match words(&m[2]) {
                Some(rhs) if !rhs.is_empty() => match expand(&rhs[0], &env) {
                    Some(v) => EnvVal::Str(v),
                    None => EnvVal::Unknown,
                },
                _ => EnvVal::Str(String::new()),
            };
            env.insert(m[1].to_string(), value);
            continue;
        }
        if ws.first().map(String::as_str) == Some("cd") && ws.len() > 1 {
            cwd = match expand(&ws[1], &env) {
                None => None,
                Some(target) if cwd.is_some() || target.starts_with('/') => {
                    let base = cwd.clone().unwrap_or_else(|| "/".into());
                    let joined = if target.starts_with('/') {
                        target
                    } else if base.ends_with('/') {
                        format!("{base}{target}")
                    } else {
                        format!("{base}/{target}")
                    };
                    Some(normpath(&joined))
                }
                Some(_) => None,
            };
            env.insert(
                "PWD".into(),
                match &cwd {
                    Some(c) if !c.is_empty() => EnvVal::Str(c.clone()),
                    _ => EnvVal::Unknown,
                },
            );
            continue;
        }
        if ws.first().is_some_and(|w| !TRIVIAL.contains(&w.as_str())) && !FUNCTION.is_match(c) {
            before.push(Before {
                command: c.to_string(),
            });
        }
    }
    (env, before)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn w(s: &str) -> Vec<String> {
        words(s).unwrap()
    }

    #[test]
    fn words_split_like_python_shlex() {
        assert_eq!(
            w("a && b; c && d || e; f >'abc'; (def \"ghi\")"),
            vec![
                "a", "&&", "b", ";", "c", "&&", "d", "||", "e", ";", "f", ">", "abc", ";", "(",
                "def", "ghi", ")"
            ]
        );
        assert_eq!(
            w("rewardkit /tests 2>&1"),
            vec!["rewardkit", "/tests", "2", ">&", "1"]
        );
        assert_eq!(w("x '' y"), vec!["x", "", "y"]);
        assert_eq!(w(r#"a"b c"d e\ f"#), vec!["ab cd", "e f"]);
        assert_eq!(w(r#""a\"b\$c""#), vec![r#"a"b\$c"#]);
        assert_eq!(w("$HOME/x ${A:-b}"), vec!["$HOME/x", "${A:-b}"]);
        assert_eq!(words("'unclosed"), None);
        assert_eq!(words("trailing\\"), None);
        assert_eq!(w(""), Vec::<String>::new());
    }

    #[test]
    fn comments_and_lines() {
        assert_eq!(code("echo hi # note"), "echo hi");
        assert_eq!(code("echo '#not' x"), "echo '#not' x");
        assert_eq!(code("echo a#b"), "echo a#b");
        let ls = lines("a \\\n  b\ncat <<'EOF'\nbody\nEOF\nend\n");
        assert_eq!(ls[0].text, "a b");
        assert_eq!((ls[0].start, ls[0].end), (1, 2));
        assert_eq!(ls[1].text, "cat <<'EOF'");
        assert_eq!(ls[2].text, "end");
    }

    #[test]
    fn depths_follow_blocks() {
        let ls = lines("if true; then\n  x\nfi\ny\n");
        assert_eq!(depths(&ls), vec![0, 1, 1, 0, 0]);
    }

    #[test]
    fn expansion_and_state() {
        let mut env = Env::new();
        env.insert("A".into(), EnvVal::Str("1".into()));
        env.insert(
            "H".into(),
            EnvVal::Host {
                var: "H".into(),
                default: None,
            },
        );
        assert_eq!(expand("x$A${B:-/tests}", &env).as_deref(), Some("x1/tests"));
        assert_eq!(expand("$H", &env), None);
        assert_eq!(expand("${H:-d}", &env).as_deref(), Some("d"));
        let ls = lines(
            "set -e\nDIR=\"${T:-/tests}\"\npip install harbor-rewardkit\ncd /app\nrewardkit $DIR\n",
        );
        let ds = depths(&ls);
        let (env, before) = state(&ls, &ds, 4, &Env::new());
        assert_eq!(env.get("DIR"), Some(&EnvVal::Str("/tests".into())));
        assert_eq!(env.get("PWD"), Some(&EnvVal::Str("/app".into())));
        assert_eq!(
            before,
            vec![Before {
                command: "pip install harbor-rewardkit".into()
            }]
        );
    }

    #[test]
    fn normpath_matches_python() {
        assert_eq!(normpath("/a/./b/../c"), "/a/c");
        assert_eq!(normpath("//a"), "//a");
        assert_eq!(normpath("///a"), "/a");
        assert_eq!(normpath("../x"), "../x");
        assert_eq!(normpath(""), ".");
    }
}
