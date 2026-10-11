//! The names BenchFlow's text never uses: other evaluation frameworks. Shared
//! by the unit test in `names.rs` and `tests/no_other_frameworks.rs`.

/// Other evaluation frameworks, matched case-insensitively as whole words.
/// Before matching, `-`, `_` and every run of whitespace (newlines, tabs,
/// no-break spaces) become one space, so `inspect-ai`, `Inspect\nAI` and
/// `inspect_ai` are all "inspect ai"; a change from a lower-case to an
/// upper-case letter also starts a word, so `HarborTask` names "harbor".
pub const OTHER_FRAMEWORKS: &[&str] = &[
    "harbor",
    "harbour",
    "karotte",
    "inspect ai",
    "inspect evals",
    "lm eval",
    "lm evaluation harness",
    "lighteval",
    "openai evals",
    "openai/evals",
    "opencompass",
];

/// Identifiers and data from outside bench, which it keeps as they are: the
/// import record and its folder, as task.md's `convert.py` writes them and
/// other tools read them. bench's own words around them must still name no
/// framework.
pub const IDENTIFIERS: &[&str] = &["import.harbor", "import/harbor"];

/// The framework names `text` mentions. Web addresses are data (a
/// dataset's URL is where it lives), and so are [`IDENTIFIERS`].
pub fn named(text: &str) -> Vec<&'static str> {
    named_except(text, &[])
}

/// [`named`], where `typed` are values the user gave, which a message may
/// repeat. Each is skipped only where it stands as a whole value (not
/// inside a longer word), longest first.
pub fn named_except(text: &str, typed: &[&str]) -> Vec<&'static str> {
    let mut text = without_addresses(text);
    let mut typed: Vec<&str> = typed.iter().copied().filter(|s| !s.is_empty()).collect();
    typed.sort_by_key(|s| std::cmp::Reverse(s.len()));
    for t in typed {
        text = remove(&text, t, true);
    }
    for id in IDENTIFIERS {
        text = remove(&text, id, false);
    }
    let forms = [words(&text.to_lowercase()), words(&split_camel(&text))];
    OTHER_FRAMEWORKS
        .iter()
        .copied()
        .filter(|name| {
            let name = words(name);
            forms.iter().any(|f| has_word(f, &name))
        })
        .collect()
}

/// `text` with `-`, `_` and whitespace runs as single spaces.
pub fn words(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut space = false;
    for c in text.chars() {
        if c.is_whitespace() || c == '-' || c == '_' {
            space = true;
        } else {
            if space && !out.is_empty() {
                out.push(' ');
            }
            space = false;
            out.push(c);
        }
    }
    out
}

/// `text` as plain words: CamelCase split, lower case, and every character
/// that is not a letter or digit a single space. The leak finder puts a line
/// and the package's text in this form before comparing them.
#[allow(dead_code)] // only tests/no_other_frameworks.rs calls it
pub fn plain_words(text: &str) -> String {
    let spaced: String = split_camel(text)
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect();
    spaced.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// A space before each upper-case letter that follows a lower-case one.
fn split_camel(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 8);
    let mut prev_lower = false;
    for c in text.chars() {
        if c.is_uppercase() && prev_lower {
            out.push(' ');
        }
        prev_lower = c.is_lowercase();
        out.push(c);
    }
    out.to_lowercase()
}

/// Whether `name` occurs in `text` with no letter or digit on either side.
fn has_word(text: &str, name: &str) -> bool {
    let word = |c: Option<char>| c.is_some_and(char::is_alphanumeric);
    text.match_indices(name).any(|(i, m)| {
        !word(text[..i].chars().next_back()) && !word(text[i + m.len()..].chars().next())
    })
}

/// `text` without each occurrence of `value`, ignoring ASCII case. With
/// `whole`, only where it stands alone: no letter, digit, `-` or `_` right
/// before or after it.
fn remove(text: &str, value: &str, whole: bool) -> String {
    let lower = text.to_ascii_lowercase();
    let value = value.to_ascii_lowercase();
    let part = |c: Option<char>| c.is_some_and(|c| c.is_alphanumeric() || c == '-' || c == '_');
    let mut out = String::with_capacity(text.len());
    let mut at = 0;
    while let Some(found) = lower[at..].find(&value) {
        let (i, end) = (at + found, at + found + value.len());
        let alone =
            !whole || (!part(text[..i].chars().next_back()) && !part(text[end..].chars().next()));
        out.push_str(&text[at..i]);
        out.push_str(if alone { " " } else { &text[i..end] });
        at = end;
    }
    out.push_str(&text[at..]);
    out
}

/// Top-level domains a bare host name is recognized by, so that a file name
/// such as `task.toml/x` is not taken for one.
const WEB_TLDS: &[&str] = &["com", "org", "net", "io", "ai", "dev", "co", "app"];

/// `text` with every web address blanked: a URL, or a host name and path
/// such as `github.com/org/repo`. An address starts the text, or follows
/// whitespace or one of `"'(<[{=,:`, and ends at whitespace or one of
/// `"'),>]}`, so in compact JSON it stops at its own closing quote.
fn without_addresses(text: &str) -> String {
    const OPEN: &str = "\"'(<[{=,:`";
    const CLOSE: &str = "\"'),>]}`";
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    let mut prev: Option<char> = None;
    while i < text.len() {
        let starts = prev.is_none_or(|p| p.is_whitespace() || OPEN.contains(p));
        if starts && let Some(len) = address_len(&text[i..], CLOSE) {
            out.push(' ');
            prev = text[..i + len].chars().next_back();
            i += len;
            continue;
        }
        let c = text[i..].chars().next().unwrap();
        out.push(c);
        prev = Some(c);
        i += c.len_utf8();
    }
    out
}

/// The length of the address `s` starts with, if it starts with one.
fn address_len(s: &str, close: &str) -> Option<usize> {
    let end = s
        .find(|c: char| c.is_whitespace() || close.contains(c))
        .unwrap_or(s.len());
    let w = &s[..end];
    if w.starts_with("http://") || w.starts_with("https://") {
        return Some(end);
    }
    let (host, _) = w.split_once('/')?;
    let host_ok = host.contains('.')
        && !host.starts_with('.')
        && !host.ends_with('.')
        && host
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '-')
        && host
            .rsplit('.')
            .next()
            .is_some_and(|tld| WEB_TLDS.contains(&tld));
    host_ok.then_some(end)
}
