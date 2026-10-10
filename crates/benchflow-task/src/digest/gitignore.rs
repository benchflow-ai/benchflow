//! `.gitignore` patterns as Harbor's packager reads them: Python's pathspec
//! (1.x) with its `gitignore` pattern factory, `GitIgnoreBasicPattern`.
//!
//! Each line becomes a regular expression that is searched in a path
//! relative to the task folder; the last pattern that matches decides, and
//! a `!` pattern re-includes. This follows pathspec's documented-gitignore
//! reading, which differs from Git in one case: a file inside an ignored
//! folder can be re-included with `!`.

use regex::Regex;

/// A compiled `.gitignore`.
#[derive(Debug, Clone)]
pub struct GitIgnore {
    /// (pattern, true when a match ignores the path, false when it keeps it)
    patterns: Vec<(Regex, bool)>,
}

/// Harbor's default ignores, used when a task has no `.gitignore`.
pub const DEFAULT_IGNORES: [&str; 6] =
    ["__pycache__/", "*.pyc", ".DS_Store", "*.swp", "*.swo", "*~"];

impl GitIgnore {
    /// Compile the lines of a `.gitignore`. Lines that pathspec would refuse
    /// (an escape with nothing after it, a lone `!`) are skipped.
    pub fn new<'a>(lines: impl IntoIterator<Item = &'a str>) -> Self {
        let patterns = lines
            .into_iter()
            .filter(|l| !l.is_empty())
            .filter_map(compile)
            .collect();
        Self { patterns }
    }

    /// Read `.gitignore` text: Python's universal newlines, then
    /// `str.splitlines()`.
    pub fn parse(text: &str) -> Self {
        let text = text.replace("\r\n", "\n").replace('\r', "\n");
        Self::new(text.split(is_line_break))
    }

    /// Harbor's default ignores.
    pub fn harbor_defaults() -> Self {
        Self::new(DEFAULT_IGNORES)
    }

    /// Whether `path` (relative, `/`-separated) is ignored.
    pub fn is_ignored(&self, path: &str) -> bool {
        let path = path
            .strip_prefix('/')
            .or_else(|| path.strip_prefix("./"))
            .unwrap_or(path);
        self.patterns
            .iter()
            .rev()
            .find(|(re, _)| re.is_match(path))
            .is_some_and(|(_, ignore)| *ignore)
    }
}

/// The characters Python's `str.splitlines()` breaks lines at (besides
/// `\r`, which universal newlines already turned into `\n`).
fn is_line_break(c: char) -> bool {
    matches!(
        c,
        '\n' | '\x0b' | '\x0c' | '\x1c' | '\x1d' | '\x1e' | '\u{85}' | '\u{2028}' | '\u{2029}'
    )
}

/// Python's `str.rstrip()` whitespace.
fn is_py_space(c: char) -> bool {
    c.is_whitespace() || matches!(c, '\x1c'..='\x1f')
}

fn compile(line: &str) -> Option<(Regex, bool)> {
    let pattern = if line.ends_with("\\ ") {
        line
    } else {
        line.trim_end_matches(is_py_space)
    };
    if pattern.is_empty() || pattern.starts_with('#') {
        return None;
    }
    let (pattern, ignore) = match pattern.strip_prefix('!') {
        Some(rest) => (rest, false),
        None => (pattern, true),
    };
    let mut segs: Vec<String> = pattern.split('/').map(str::to_string).collect();
    let mut is_dir = segs.last().is_some_and(|s| s.is_empty());
    if pattern == "/" {
        is_dir = false;
    }
    let regex = match normalize(is_dir, &mut segs)? {
        Normalized::Override(re) => re.to_string(),
        Normalized::Segments => translate(&segs)?,
    };
    Regex::new(&regex).ok().map(|re| (re, ignore))
}

enum Normalized {
    Override(&'static str),
    Segments,
}

/// pathspec's `__normalize_segments`. `None` when the pattern normalizes to
/// nothing.
fn normalize(is_dir: bool, segs: &mut Vec<String>) -> Option<Normalized> {
    if segs[0].is_empty() {
        segs.remove(0);
    } else if (segs.len() == 1 || (segs.len() == 2 && segs[1].is_empty())) && segs[0] != "**" {
        segs.insert(0, "**".into());
    }
    if segs.is_empty() {
        return None;
    }
    if segs.last().is_some_and(String::is_empty) {
        *segs.last_mut().expect("not empty") = "**".into();
    }
    let mut i = segs.len() - 1;
    while i > 0 {
        if segs[i - 1] == "**" && segs[i] == "**" {
            segs.remove(i);
        }
        i -= 1;
    }
    let s: Vec<&str> = segs.iter().map(String::as_str).collect();
    Some(match s.as_slice() {
        ["**"] if is_dir => Normalized::Override("/"),
        ["**"] | ["**", "*"] => Normalized::Override("."),
        ["**", "*", "**"] => Normalized::Override("/"),
        _ => Normalized::Segments,
    })
}

/// pathspec's `__translate_segments`.
fn translate(segs: &[String]) -> Option<String> {
    let mut out = String::new();
    let mut need_slash = false;
    let end = segs.len() - 1;
    for (i, seg) in segs.iter().enumerate() {
        if seg == "**" {
            if i == 0 {
                out.push_str("^(?:.+/)?");
            } else if i < end {
                out.push_str("(?:/.+)?");
                need_slash = true;
            } else {
                out.push('/');
            }
            continue;
        }
        if i == 0 {
            out.push('^');
        }
        if need_slash {
            out.push('/');
        }
        if seg == "*" {
            out.push_str("[^/]+");
        } else {
            out.push_str(&glob(seg)?);
        }
        if i == end {
            out.push_str(if seg == "*" { "/?$" } else { "(?:/|$)" });
        }
        need_slash = true;
    }
    Some(out)
}

/// pathspec's `_translate_segment_glob` with invalid ranges read as
/// literals. `None` for a trailing escape, which pathspec refuses.
fn glob(pattern: &str) -> Option<String> {
    let chars: Vec<char> = pattern.chars().collect();
    let mut out = String::new();
    let mut escape = false;
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        i += 1;
        if escape {
            escape = false;
            out.push_str(&regex::escape(&c.to_string()));
        } else if c == '\\' {
            escape = true;
        } else if c == '*' {
            out.push_str("[^/]*");
        } else if c == '?' {
            out.push_str("[^/]");
        } else if c == '[' {
            let mut j = i;
            if j < chars.len() && (chars[j] == '!' || chars[j] == '^') {
                j += 1;
            }
            if j < chars.len() && chars[j] == ']' {
                j += 1;
            }
            while j < chars.len() && chars[j] != ']' {
                j += 1;
            }
            if j < chars.len() {
                j += 1;
                let mut expr = String::from("[");
                let mut k = i;
                if chars[k] == '!' || chars[k] == '^' {
                    expr.push('^');
                    k += 1;
                }
                expr.push_str(&class_body(&chars[k..j - 1]));
                expr.push(']');
                out.push_str(&expr);
                i = j;
            } else {
                out.push_str("\\[");
            }
        } else {
            out.push_str(&regex::escape(&c.to_string()));
        }
    }
    (!escape).then_some(out)
}

/// The inside of a bracket expression, written so Rust's regex reads it as
/// Python's does: every character literal except `-` between two others,
/// which makes a range.
fn class_body(chars: &[char]) -> String {
    let mut out = String::new();
    for (n, c) in chars.iter().enumerate() {
        let range = *c == '-' && n > 0 && n + 1 < chars.len();
        if range {
            out.push('-');
        } else if c.is_alphanumeric() {
            out.push(*c);
        } else {
            out.push('\\');
            out.push(*c);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn harbor_defaults() {
        let g = GitIgnore::harbor_defaults();
        assert!(g.is_ignored("tests/__pycache__/x.cpython-312.pyc"));
        assert!(g.is_ignored("environment/a/__pycache__/m.txt"));
        assert!(g.is_ignored("solution/x.pyc"));
        assert!(g.is_ignored(".DS_Store"));
        assert!(g.is_ignored("environment/sub/.DS_Store"));
        assert!(g.is_ignored("tests/notes.txt~"));
        assert!(!g.is_ignored("tests/test.sh"));
        assert!(!g.is_ignored("environment/pycache/x"));
    }

    #[test]
    fn anchors_negation_and_directories() {
        let g = GitIgnore::parse("/build\n*.log\n!keep.log\ndata/\n# comment\n\ndocs/*.md\n");
        assert!(g.is_ignored("build"));
        assert!(g.is_ignored("build/out.bin"));
        assert!(!g.is_ignored("src/build"));
        assert!(g.is_ignored("a/b.log"));
        assert!(!g.is_ignored("a/keep.log"));
        assert!(g.is_ignored("data/x"));
        assert!(g.is_ignored("x/data/y"));
        assert!(!g.is_ignored("data"));
        assert!(g.is_ignored("docs/a.md"));
        assert!(!g.is_ignored("docs/sub/a.md"));
        assert!(!g.is_ignored("x/docs/a.md"));
    }

    #[test]
    fn double_stars_and_specials() {
        let g = GitIgnore::parse("a/**/b\n**/c\nd/**\n");
        assert!(g.is_ignored("a/b"));
        assert!(g.is_ignored("a/x/y/b"));
        assert!(g.is_ignored("q/c"));
        assert!(g.is_ignored("c"));
        assert!(g.is_ignored("d/e/f"));
        assert!(!g.is_ignored("d"));
        assert!(GitIgnore::parse("*").is_ignored("anything/at/all"));
        assert!(GitIgnore::parse("/").is_ignored("x"));
        assert!(!GitIgnore::parse("*/").is_ignored("top.txt"));
        assert!(GitIgnore::parse("*/").is_ignored("d/top.txt"));
    }

    #[test]
    fn brackets_and_escapes() {
        let g = GitIgnore::parse("[abc].txt\nfile[0-9]\n[!x]y\n\\#hash\nweird[\n");
        assert!(g.is_ignored("a.txt"));
        assert!(!g.is_ignored("d.txt"));
        assert!(g.is_ignored("file7"));
        assert!(g.is_ignored("zy"));
        assert!(!g.is_ignored("xy"));
        assert!(g.is_ignored("#hash"));
        assert!(g.is_ignored("weird["));
        assert!(GitIgnore::parse("trail\\").patterns.is_empty());
        assert!(GitIgnore::parse("!").patterns.is_empty());
    }
}
