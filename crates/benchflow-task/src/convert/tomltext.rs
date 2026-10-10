//! Writing TOML, and finding the comments in TOML text, the way task.md's
//! `tools/tomltext.py` does, so converted files match its output byte for
//! byte.
//!
//! [`dumps`] writes a table with comments attached to key paths;
//! [`scan_comments`] finds them: comments before a key or table header
//! belong to it, one at the end of its line is its inline comment, and
//! comments at the end of the text belong to [`end`].

use std::collections::HashSet;

use indexmap::IndexMap;

use super::value::{KPath, Seg, Table, Val, py_float_repr};

/// How wide a written line may get before a list wraps.
const WIDTH: usize = 100;

/// The comments on one key or table.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct Comment {
    /// Whole-line comments above it, in order.
    pub lead: Vec<String>,
    /// The comment at the end of its line.
    pub inline: Option<String>,
}

/// Comments by the path of the key or table they belong to.
pub(crate) type Comments = IndexMap<KPath, Comment>;

/// The path comments at the end of the text belong to.
pub(crate) fn end() -> KPath {
    vec![Seg::Key("__end__".into())]
}

fn is_bare(k: &str) -> bool {
    // Python's `$` also matches before one trailing newline.
    let body = k.strip_suffix('\n').unwrap_or(k);
    !body.is_empty()
        && body
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

/// A key as TOML writes it: bare when it can be, else quoted.
pub(crate) fn key(k: &str) -> String {
    if is_bare(k) { k.to_string() } else { basic(k) }
}

fn escape(s: &str, multiline: bool) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        let o = ch as u32;
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' if multiline => out.push('\n'),
            '\n' => out.push_str("\\n"),
            '\t' if multiline => out.push('\t'),
            '\t' => out.push_str("\\t"),
            _ if o < 0x20 || o == 0x7f => out.push_str(&format!("\\u{o:04x}")),
            _ => out.push(ch),
        }
    }
    out
}

fn basic(s: &str) -> String {
    format!("\"{}\"", escape(s, false))
}

fn string(s: &str) -> String {
    if s.trim_end_matches('\n').contains('\n') && s.chars().count() > 60 {
        format!("\"\"\"\n{}\"\"\"", escape(s, true))
    } else {
        basic(s)
    }
}

/// A value as TOML writes it. `None` for a value TOML cannot hold (null).
pub(crate) fn value(v: &Val, indent: &str) -> Option<String> {
    Some(match v {
        Val::Bool(b) => {
            if *b {
                "true".into()
            } else {
                "false".into()
            }
        }
        Val::Int(i) => i.to_string(),
        Val::Float(f) => {
            if f.is_nan() {
                "nan".into()
            } else if f.is_infinite() {
                if *f > 0.0 {
                    "inf".into()
                } else {
                    "-inf".into()
                }
            } else {
                let r = py_float_repr(*f);
                if r.contains(['.', 'e', 'n']) {
                    r
                } else {
                    format!("{r}.0")
                }
            }
        }
        Val::Str(s) => string(s),
        Val::Date(d) => d.clone(),
        Val::List(items) => {
            let inner = format!("{indent}  ");
            let written: Vec<String> = items
                .iter()
                .map(|x| value(x, &inner))
                .collect::<Option<_>>()?;
            let flat = format!("[{}]", written.join(", "));
            if flat.chars().count() <= WIDTH && !flat.contains('\n') {
                return Some(flat);
            }
            let scalars = items
                .iter()
                .all(|x| !matches!(x, Val::Table(_) | Val::List(_)));
            if scalars && written.iter().all(|x| !x.contains('\n')) {
                let mut lines: Vec<String> = Vec::new();
                let mut cur = String::new();
                for x in &written {
                    let width =
                        indent.chars().count() + 2 + cur.chars().count() + x.chars().count() + 2;
                    if !cur.is_empty() && width > WIDTH {
                        lines.push(std::mem::take(&mut cur));
                    }
                    cur.push_str(x);
                    cur.push_str(", ");
                }
                lines.push(cur);
                let body: String = lines
                    .iter()
                    .map(|l| format!("{indent}  {}\n", l.trim_end()))
                    .collect();
                return Some(format!("[\n{body}{indent}]"));
            }
            let body: String = written
                .iter()
                .map(|x| format!("{indent}  {x},\n"))
                .collect();
            format!("[\n{body}{indent}]")
        }
        Val::Table(t) => {
            if t.is_empty() {
                return Some("{}".into());
            }
            let parts: Vec<String> = t
                .iter()
                .map(|(k, x)| value(x, "").map(|w| format!("{} = {w}", key(k))))
                .collect::<Option<_>>()?;
            format!("{{ {} }}", parts.join(", "))
        }
        Val::Null => return None,
    })
}

fn is_aot(v: &Val) -> bool {
    matches!(v, Val::List(items) if !items.is_empty() && items.iter().all(|x| matches!(x, Val::Table(_))))
}

/// Small flat tables read better inline: `env = { A = "1" }`.
fn inline_ok(t: &Table) -> bool {
    if t.is_empty() {
        return true;
    }
    if t.values().any(|x| matches!(x, Val::Table(_)) || is_aot(x)) {
        return false;
    }
    match value(&Val::Table(t.clone()), "") {
        Some(text) => text.chars().count() <= 80 && !text.contains('\n'),
        None => false,
    }
}

fn header_path(path: &[Seg]) -> String {
    path.iter()
        .filter_map(|s| match s {
            Seg::Key(k) => Some(key(k)),
            Seg::Index(_) => None,
        })
        .collect::<Vec<_>>()
        .join(".")
}

/// Write a table as TOML text with its comments. `None` when a value
/// cannot be written (a YAML null that was not dropped).
pub(crate) fn dumps(data: &Table, comments: &Comments) -> Option<String> {
    let mut under: HashSet<KPath> = HashSet::new();
    for path in comments.keys() {
        for n in 0..=path.len() {
            under.insert(path[..n].to_vec());
        }
    }
    let mut lines: Vec<String> = Vec::new();
    lead(&mut lines, comments.get(&Vec::new()));
    table(&mut lines, &[], data, comments, &under, false)?;
    lead(&mut lines, comments.get(&end()));
    let text = lines.join("\n");
    Some(format!("{}\n", text.trim_matches('\n')))
}

fn lead(lines: &mut Vec<String>, c: Option<&Comment>) {
    if let Some(c) = c {
        lines.extend(c.lead.iter().cloned());
    }
}

fn inline(c: Option<&Comment>) -> String {
    match c.and_then(|c| c.inline.as_ref()) {
        Some(text) => format!("  {text}"),
        None => String::new(),
    }
}

fn table(
    lines: &mut Vec<String>,
    path: &[Seg],
    data: &Table,
    comments: &Comments,
    under: &HashSet<KPath>,
    aot: bool,
) -> Option<()> {
    let only_tables = data.values().all(|x| matches!(x, Val::Table(_)));
    let as_table = |k: &str, v: &Val| -> bool {
        if is_aot(v) {
            return true;
        }
        let Val::Table(t) = v else { return false };
        let mut child = path.to_vec();
        child.push(Seg::Key(k.to_string()));
        path.is_empty()
            || only_tables
            || !inline_ok(t)
            || under
                .iter()
                .any(|p| p.len() > child.len() && p[..child.len()] == child[..])
    };
    let simple: Vec<(&String, &Val)> = data.iter().filter(|(k, v)| !as_table(k, v)).collect();
    let nested: Vec<(&String, &Val)> = data.iter().filter(|(k, v)| as_table(k, v)).collect();
    if !path.is_empty() {
        let c = comments.get(path);
        if !simple.is_empty() || aot || nested.is_empty() || c.is_some() {
            if !lines.is_empty() {
                lines.push(String::new());
            }
            lead(lines, c);
            let (open, close) = if aot { ("[[", "]]") } else { ("[", "]") };
            lines.push(format!("{open}{}{close}{}", header_path(path), inline(c)));
        }
    }
    for (k, v) in simple {
        let mut p = path.to_vec();
        p.push(Seg::Key(k.clone()));
        let c = comments.get(&p);
        lead(lines, c);
        lines.push(format!("{} = {}{}", key(k), value(v, "")?, inline(c)));
    }
    for (k, v) in nested {
        let mut p = path.to_vec();
        p.push(Seg::Key(k.clone()));
        match v {
            Val::Table(t) => table(lines, &p, t, comments, under, false)?,
            Val::List(items) => {
                if let Some(c) = comments.get(&p) {
                    lines.push(String::new());
                    lead(lines, Some(c));
                    if let Some(text) = &c.inline {
                        lines.push(text.clone());
                    }
                }
                for (i, item) in items.iter().enumerate() {
                    let mut ip = p.clone();
                    ip.push(Seg::Index(i));
                    if let Val::Table(t) = item {
                        table(lines, &ip, t, comments, under, true)?;
                    }
                }
            }
            _ => {}
        }
    }
    Some(())
}

/// The key path a TOML key spells (`a."b.c"` is `["a", "b.c"]`).
fn key_path(k: &str) -> Result<KPath, String> {
    let mut table = super::value::parse_toml(&format!("{k} = 0"))?;
    let mut out = Vec::new();
    loop {
        let (name, v) = table.into_iter().next().ok_or("an empty key")?;
        out.push(Seg::Key(name));
        match v {
            Val::Table(inner) => table = inner,
            _ => return Ok(out),
        }
    }
}

/// Split `key = value` at the first `=` outside quotes.
fn split_statement(s: &str) -> Result<(String, String), String> {
    let mut quote: Option<char> = None;
    for (i, c) in s.char_indices() {
        match quote {
            Some(q) if c == q => quote = None,
            Some(_) => {}
            None if c == '"' || c == '\'' => quote = Some(c),
            None if c == '=' => return Ok((s[..i].trim().to_string(), s[i + 1..].to_string())),
            None => {}
        }
    }
    Err(format!("not a key/value line: {s:?}"))
}

/// A table header's name and what follows its closing bracket(s).
fn header(s: &str, n: usize) -> Result<(String, String), String> {
    let chars: Vec<char> = s.chars().collect();
    let mut quote: Option<char> = None;
    for i in n..chars.len() {
        let c = chars[i];
        match quote {
            Some(q) if c == q => quote = None,
            Some(_) => {}
            None if c == '"' || c == '\'' => quote = Some(c),
            None if c == ']' => {
                let name: String = chars[n..i].iter().collect();
                let rest: String = chars
                    .get(i + n..)
                    .map(|r| r.iter().collect())
                    .unwrap_or_default();
                return Ok((name.trim().to_string(), rest));
            }
            None => {}
        }
    }
    Err(format!("bad table header: {s:?}"))
}

/// Scan a value that may span lines: (inline comment, comments inside the
/// value, the next line index).
fn value_span(lines: &[&str], i: usize, rest: &str) -> (Option<String>, Vec<String>, usize) {
    let mut inner: Vec<String> = Vec::new();
    let mut depth: i64 = 0;
    let mut state: Option<String> = None;
    let mut j = i;
    let mut line: Vec<char> = rest.chars().collect();
    loop {
        let mut k = 0;
        while k < line.len() {
            let c = line[k];
            let starts = |at: usize, pat: &str| -> bool {
                let p: Vec<char> = pat.chars().collect();
                line.len() >= at + p.len() && line[at..at + p.len()] == p[..]
            };
            match state.as_deref() {
                None => {
                    if starts(k, "\"\"\"") || starts(k, "'''") {
                        state = Some(line[k..k + 3].iter().collect());
                        k += 3;
                        continue;
                    }
                    if c == '"' || c == '\'' {
                        state = Some(c.to_string());
                    } else if c == '[' || c == '{' {
                        depth += 1;
                    } else if c == ']' || c == '}' {
                        depth -= 1;
                    } else if c == '#' {
                        let text: String = line[k..].iter().collect::<String>().trim().to_string();
                        if depth <= 0 {
                            return (Some(text), inner, j + 1);
                        }
                        inner.push(text);
                        break;
                    }
                    k += 1;
                }
                Some(s) if (s == "\"" || s == "\"\"\"") && c == '\\' => k += 2,
                Some(s) if s.len() == 3 && starts(k, s) => {
                    let q = line[k];
                    k += 3;
                    while k < line.len() && line[k] == q {
                        k += 1;
                    }
                    state = None;
                }
                Some(s) => {
                    if s.chars().count() == 1 && s.starts_with(c) {
                        state = None;
                    }
                    k += 1;
                }
            }
        }
        let open_triple = matches!(state.as_deref(), Some("\"\"\"") | Some("'''"));
        if !open_triple && depth <= 0 {
            return (None, inner, j + 1);
        }
        j += 1;
        if j >= lines.len() {
            return (None, inner, j);
        }
        line = lines[j].chars().collect();
    }
}

fn attach(out: &mut Comments, path: KPath, lead: Vec<String>, inline: Option<String>) {
    if lead.is_empty() && inline.is_none() {
        return;
    }
    let c = out.entry(path).or_default();
    c.lead.extend(lead);
    if let Some(text) = inline {
        if c.inline.is_some() {
            c.lead.push(text);
        } else {
            c.inline = Some(text);
        }
    }
}

/// Add comment `c` to `path`.
pub(crate) fn merge(comments: &mut Comments, path: KPath, c: &Comment) {
    attach(comments, path, c.lead.clone(), c.inline.clone());
}

/// Every comment in TOML text, by the path of the key or table it belongs
/// to.
pub(crate) fn scan_comments(text: &str) -> Result<Comments, String> {
    let normalized = text.replace("\r\n", "\n");
    let lines: Vec<&str> = normalized.split('\n').collect();
    let mut out = Comments::new();
    let mut pending: Vec<String> = Vec::new();
    let mut current: KPath = Vec::new();
    let mut arrays: IndexMap<KPath, usize> = IndexMap::new();
    let resolve = |p: &[Seg], arrays: &IndexMap<KPath, usize>| -> KPath {
        let mut r: KPath = Vec::new();
        for part in p {
            r.push(part.clone());
            if let Some(n) = arrays.get(&r) {
                r.push(Seg::Index(n - 1));
            }
        }
        r
    };
    let mut i = 0;
    while i < lines.len() {
        let s = lines[i].trim();
        if s.is_empty() {
            i += 1;
        } else if s.starts_with('#') {
            pending.push(s.to_string());
            i += 1;
        } else if s.starts_with('[') {
            let n = if s.starts_with("[[") { 2 } else { 1 };
            let (name, rest) = header(s, n)?;
            let p = key_path(&name)?;
            if n == 2 {
                let mut base = resolve(&p[..p.len() - 1], &arrays);
                base.push(p[p.len() - 1].clone());
                let count = arrays.entry(base.clone()).or_insert(0);
                *count += 1;
                let mut t = base;
                t.push(Seg::Index(*count - 1));
                current = t;
            } else {
                current = resolve(&p, &arrays);
            }
            let rest = rest.trim();
            let inline = rest.starts_with('#').then(|| rest.to_string());
            attach(
                &mut out,
                current.clone(),
                std::mem::take(&mut pending),
                inline,
            );
            i += 1;
        } else {
            let (k, rest) = split_statement(s)?;
            let (inline, inner, next) = value_span(&lines, i, &rest);
            let mut path = current.clone();
            path.extend(key_path(&k)?);
            let mut lead = std::mem::take(&mut pending);
            lead.extend(inner);
            attach(&mut out, path, lead, inline);
            i = next;
        }
    }
    if !pending.is_empty() {
        attach(&mut out, end(), pending, None);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convert::value::{kp, parse_toml};

    fn round(text: &str) -> String {
        let data = parse_toml(text).unwrap();
        let comments = scan_comments(text).unwrap();
        dumps(&data, &comments).unwrap()
    }

    #[test]
    fn writes_tables_and_inline_tables() {
        let text = "name = \"a/b\"\n\n[agent]\ntimeout = \"10m\"\n\n[sandbox]\nenv = { A = \"1\" }\nimage = \"x\"\n";
        assert_eq!(
            round(text),
            "name = \"a/b\"\n\n[agent]\ntimeout = \"10m\"\n\n[sandbox]\nenv = { A = \"1\" }\nimage = \"x\"\n"
        );
    }

    #[test]
    fn keeps_comments_where_they_were() {
        let text = "# top\nversion = \"1.0\"  # the schema\n\n# about the agent\n[agent]\ntimeout_sec = 600 # ten minutes\n# trailing\n";
        let c = scan_comments(text).unwrap();
        assert_eq!(c[&kp(&["version"])].inline.as_deref(), Some("# the schema"));
        assert_eq!(c[&kp(&["version"])].lead, vec!["# top".to_string()]);
        assert_eq!(
            c[&kp(&["agent"])].lead,
            vec!["# about the agent".to_string()]
        );
        assert_eq!(
            c[&kp(&["agent", "timeout_sec"])].inline.as_deref(),
            Some("# ten minutes")
        );
        assert_eq!(c[&end()].lead, vec!["# trailing".to_string()]);
        assert_eq!(
            round(text),
            "# top\nversion = \"1.0\"  # the schema\n\n# about the agent\n[agent]\ntimeout_sec = 600  # ten minutes\n# trailing\n"
        );
    }

    #[test]
    fn arrays_of_tables_and_comments_inside_values() {
        let text = "[[steps]]\nname = \"a\"\n[[steps]]\nname = \"b\" # second\nhosts = [\n  \"x\", # first host\n  \"y\",\n]\n";
        let c = scan_comments(text).unwrap();
        let second = vec![
            Seg::Key("steps".into()),
            Seg::Index(1),
            Seg::Key("name".into()),
        ];
        assert_eq!(c[&second].inline.as_deref(), Some("# second"));
        let hosts = vec![
            Seg::Key("steps".into()),
            Seg::Index(1),
            Seg::Key("hosts".into()),
        ];
        assert_eq!(c[&hosts].lead, vec!["# first host".to_string()]);
        assert_eq!(
            round(text),
            "[[steps]]\nname = \"a\"\n\n[[steps]]\nname = \"b\"  # second\n# first host\nhosts = [\"x\", \"y\"]\n"
        );
    }

    #[test]
    fn long_lists_wrap_and_long_strings_go_multiline() {
        let mut t = Table::new();
        let hosts: Vec<Val> = (0..12)
            .map(|i| Val::Str(format!("host-number-{i}.example.com")))
            .collect();
        t.insert("hosts".into(), Val::List(hosts));
        let text = dumps(&t, &Comments::new()).unwrap();
        assert!(text.starts_with("hosts = [\n  \"host-number-0.example.com\", "));
        assert!(text.lines().all(|l| l.chars().count() <= 100));
        let long =
            "line one of a long text\nline two of a long text\nline three ends it\n".to_string();
        assert_eq!(string(&long), format!("\"\"\"\n{long}\"\"\""));
        assert_eq!(string("short\nx"), "\"short\\nx\"");
    }

    #[test]
    fn keys_and_escapes() {
        assert_eq!(key("a-b_1"), "a-b_1");
        assert_eq!(key("a.b"), "\"a.b\"");
        assert_eq!(basic("tab\there\u{7f}"), "\"tab\\there\\u007f\"");
        assert_eq!(key_path("a.\"b.c\"").unwrap(), kp(&["a", "b.c"]));
    }
}
