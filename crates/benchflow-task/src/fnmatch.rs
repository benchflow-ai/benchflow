//! Python's `fnmatch.fnmatchcase`, which 0.8 uses for task name filters
//! (`--include`, `--exclude`).
//!
//! The dialect differs from a shell glob or `globset`: `*` also matches `/`,
//! there are no braces and no backslash escapes, `[!a]` negates and `[^a]`
//! does not, a `[` that never closes is a literal `[`, and a range written
//! backwards (`[z-a]`) is dropped. No pattern is invalid.

/// One step of a pattern.
#[derive(Debug, Clone, PartialEq)]
enum Token {
    /// `*`: any run of characters.
    Star,
    /// `?`: any one character.
    Any,
    /// A character matched as is.
    Literal(char),
    /// `[...]`: one character in (or, negated, not in) the ranges. Single
    /// characters are ranges of one.
    Set {
        negated: bool,
        ranges: Vec<(char, char)>,
    },
    /// A set that can match nothing (`[z-a]`).
    Never,
}

/// Whether `name` matches `pattern`, case-sensitively, as
/// `fnmatch.fnmatchcase(name, pattern)` decides.
pub fn fnmatchcase(name: &str, pattern: &str) -> bool {
    matches(&name.chars().collect::<Vec<_>>(), &parse(pattern))
}

/// Whether a task name passes one `--include`/`--exclude` pattern, as 0.8's
/// `task_name_selected` decides: the exact name, or a match when the
/// pattern holds `*`, `?` or `[`.
pub fn name_matches(name: &str, pattern: &str) -> bool {
    name == pattern || (pattern.contains(['*', '?', '[']) && fnmatchcase(name, pattern))
}

/// What a task filter pattern means here (0.8's `fnmatchcase`) that a shell
/// glob would read differently, one note per difference, for a warning:
/// `[^`, braces and backslashes.
pub fn dialect_notes(pattern: &str) -> Vec<&'static str> {
    let mut notes = Vec::new();
    if pattern.contains("[^") {
        notes.push(
            "`[^...]` matches `^` or the characters listed, not the others (`[!...]` means not), as in BenchFlow 0.8",
        );
    }
    if pattern.contains(['{', '}']) {
        notes.push(
            "braces match themselves, so `{a,b}` is not a choice (give one pattern per flag), as in BenchFlow 0.8",
        );
    }
    if pattern.contains('\\') {
        notes.push(
            "a backslash matches itself and escapes nothing (`[*]`, `[?]` and `[[]` match those characters), as in BenchFlow 0.8",
        );
    }
    notes
}

fn parse(pattern: &str) -> Vec<Token> {
    let pat: Vec<char> = pattern.chars().collect();
    let n = pat.len();
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        let c = pat[i];
        i += 1;
        match c {
            '*' => {
                if out.last() != Some(&Token::Star) {
                    out.push(Token::Star);
                }
            }
            '?' => out.push(Token::Any),
            '[' => {
                let mut j = i;
                if j < n && pat[j] == '!' {
                    j += 1;
                }
                if j < n && pat[j] == ']' {
                    j += 1;
                }
                while j < n && pat[j] != ']' {
                    j += 1;
                }
                if j >= n {
                    out.push(Token::Literal('['));
                    continue;
                }
                out.push(set(&pat, i, j));
                i = j + 1;
            }
            c => out.push(Token::Literal(c)),
        }
    }
    out
}

/// The set `pat[i..j]` (between the brackets), as `fnmatch.translate`
/// builds it: chunks split at the hyphens that make ranges, backwards
/// ranges dropped, a leading `!` negating.
fn set(pat: &[char], mut i: usize, j: usize) -> Token {
    let stuff = &pat[i..j];
    let chunks: Vec<Vec<char>> = if !stuff.contains(&'-') {
        vec![stuff.to_vec()]
    } else {
        let mut chunks: Vec<Vec<char>> = Vec::new();
        let mut k = if pat[i] == '!' { i + 2 } else { i + 1 };
        while let Some(at) = (k..j).find(|&x| pat[x] == '-') {
            chunks.push(pat[i..at].to_vec());
            i = at + 1;
            k = at + 3;
        }
        let last = pat[i..j].to_vec();
        if last.is_empty() {
            if let Some(prev) = chunks.last_mut() {
                prev.push('-');
            }
        } else {
            chunks.push(last);
        }
        for k in (1..chunks.len()).rev() {
            let (Some(&end), Some(&start)) = (chunks[k - 1].last(), chunks[k].first()) else {
                continue;
            };
            if end > start {
                let tail: Vec<char> = chunks[k][1..].to_vec();
                chunks[k - 1].pop();
                chunks[k - 1].extend(tail);
                chunks.remove(k);
            }
        }
        chunks
    };
    if chunks.iter().all(Vec::is_empty) && chunks.len() <= 1 {
        return Token::Never;
    }
    if chunks.len() == 1 && chunks[0] == ['!'] {
        return Token::Any;
    }
    let mut chunks = chunks;
    let negated = chunks[0].first() == Some(&'!');
    if negated {
        chunks[0].remove(0);
    }
    let mut ranges = Vec::new();
    for (k, chunk) in chunks.iter().enumerate() {
        ranges.extend(chunk.iter().map(|&c| (c, c)));
        if k > 0 {
            match (chunks[k - 1].last(), chunk.first()) {
                (Some(&a), Some(&b)) => ranges.push((a, b)),
                // A hyphen with nothing before it is a literal hyphen.
                _ => ranges.push(('-', '-')),
            }
        }
    }
    Token::Set { negated, ranges }
}

fn one(token: &Token, c: char) -> bool {
    match token {
        Token::Any => true,
        Token::Literal(l) => *l == c,
        Token::Set { negated, ranges } => ranges.iter().any(|&(a, b)| a <= c && c <= b) != *negated,
        Token::Never | Token::Star => false,
    }
}

/// Every token but `*` takes one character, so backtracking to the last
/// star is enough (linear in practice, quadratic at worst).
fn matches(name: &[char], pattern: &[Token]) -> bool {
    let (mut n, mut p) = (0, 0);
    let mut star: Option<(usize, usize)> = None;
    while n < name.len() {
        if p < pattern.len() && pattern[p] == Token::Star {
            star = Some((p, n));
            p += 1;
        } else if p < pattern.len() && one(&pattern[p], name[n]) {
            n += 1;
            p += 1;
        } else if let Some((sp, sn)) = star {
            p = sp + 1;
            n = sn + 1;
            star = Some((sp, sn + 1));
        } else {
            return false;
        }
    }
    pattern[p..].iter().all(|t| *t == Token::Star)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_dialect() {
        assert!(fnmatchcase("libero-10-a", "libero-10-*"));
        assert!(fnmatchcase("a/b", "a*b"));
        assert!(!fnmatchcase("fix-a", "fix-{a,b}"));
        assert!(fnmatchcase("^x", "[^a]x"));
        assert!(!fnmatchcase("bx", "[^a]x"));
        assert!(fnmatchcase("bx", "[!a]x"));
        assert!(fnmatchcase("task[1", "task[1"));
        assert!(fnmatchcase("a\\b", "a\\b"));
        assert!(!fnmatchcase("y", "[z-a]"));
        assert!(fnmatchcase("y", "[!z-a]"));
        assert!(!fnmatchcase("A", "a"));
    }

    /// Every case in the fixture (20,000: hand-picked ones, then seeded
    /// random names and patterns over `*?[]!^-\`, `/`, a few letters and
    /// now and then a newline or a non-ASCII letter;
    /// `tests/fixtures/fnmatch/make_fixture.py` writes it with 0.8's Python
    /// 3.12) matches as `fnmatch.fnmatchcase` decided.
    #[test]
    fn matches_python_on_generated_cases() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/fnmatch/python-fnmatchcase.jsonl");
        let text = std::fs::read_to_string(path).unwrap();
        let mut n = 0;
        for line in text.lines() {
            let case: serde_json::Value = serde_json::from_str(line).unwrap();
            let (name, pat) = (
                case["name"].as_str().unwrap(),
                case["pattern"].as_str().unwrap(),
            );
            assert_eq!(
                fnmatchcase(name, pat),
                case["match"].as_bool().unwrap(),
                "{name:?} {pat:?}"
            );
            n += 1;
        }
        assert!(n > 5000);
    }

    #[test]
    fn patterns_a_shell_reads_differently_get_a_note() {
        assert_eq!(dialect_notes("x-[^0-9]*").len(), 1);
        assert!(dialect_notes("x-[^0-9]*")[0].contains("[!...]"));
        assert!(dialect_notes("fix-{a,b}")[0].starts_with("braces"));
        assert!(dialect_notes(r"a\*b")[0].starts_with("a backslash"));
        assert_eq!(dialect_notes(r"{x}[^a]\").len(), 3);
        for plain in ["fix-*", "x-[!0-9]*", "task[12]", "a^b", "[ab^]"] {
            assert!(dialect_notes(plain).is_empty(), "{plain}");
        }
    }
}
