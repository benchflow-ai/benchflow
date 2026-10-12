//! Redaction: secrets never reach a trajectory, a log line, or an export.
//!
//! A [`Redactor`] replaces, in every string it is given:
//!
//! 1. the exact secret values of the run's login ([`crate::credentials::Login::secrets`]);
//! 2. what 0.8's canonical patterns recognize (`trajectories/types.py`):
//!    whole token families (`sk-ant-…`, `AIzaSy…`, `ghp_…`, JWTs, PEM private
//!    keys, …), credentials in URLs and query strings, and the values of
//!    `name: value` carriers whose name says secret (`x-api-key`,
//!    `*_TOKEN`, `authorization`, …), in raw, JSON, Python-repr and
//!    escaped-JSON forms.
//!
//! Redaction works on values, never on serialized text, so JSON escapes stay
//! valid. In a JSON object a string field is redacted with its key in view
//! (0.8's `_redact_field`), so `{"x-api-key": "<prefixless>"}` is caught too.
//!
//! The `regex` crate has no look-around; rules that need it check the
//! neighbouring characters after a match and otherwise search again from
//! the next character, which finds what Python's `re.sub` finds.

use std::borrow::Cow;
use std::sync::LazyLock;

use regex::{Captures, Regex};
use serde_json::{Map, Value};

use super::jsonl::to_python_json;

/// What a redacted value is replaced with (0.8's marker).
pub const REDACTED: &str = "***REDACTED***";

/// Replaces secrets in text and JSON values. Its `Debug` output counts the
/// secrets and never shows them.
#[derive(Clone)]
pub struct Redactor {
    /// Exact values, longest first so a secret containing another goes whole.
    secrets: Vec<String>,
    patterns: bool,
}

impl std::fmt::Debug for Redactor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Redactor")
            .field("secrets", &format_args!("<{} values>", self.secrets.len()))
            .field("patterns", &self.patterns)
            .finish()
    }
}

impl Default for Redactor {
    /// 0.8's patterns, no exact secrets.
    fn default() -> Self {
        Self {
            secrets: Vec::new(),
            patterns: true,
        }
    }
}

impl Redactor {
    /// Redact these exact values (a login's secrets) and 0.8's patterns.
    pub fn new<I, S>(secrets: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut secrets: Vec<String> = secrets
            .into_iter()
            .map(Into::into)
            .filter(|s| !s.is_empty())
            .collect();
        secrets.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
        secrets.dedup();
        Self {
            secrets,
            patterns: true,
        }
    }

    /// Only the exact secrets, without 0.8's patterns.
    pub fn exact_only(mut self) -> Self {
        self.patterns = false;
        self
    }

    /// Add more exact secrets.
    pub fn with_secrets<I, S>(self, more: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let patterns = self.patterns;
        let mut redactor = Self::new(
            self.secrets
                .into_iter()
                .chain(more.into_iter().map(Into::into)),
        );
        redactor.patterns = patterns;
        redactor
    }

    /// `text` with every secret replaced.
    pub fn redact<'a>(&self, text: &'a str) -> Cow<'a, str> {
        let exact = self.redact_exact(text);
        if !self.patterns {
            return exact;
        }
        match apply_rules(&exact) {
            Cow::Borrowed(_) => exact,
            Cow::Owned(changed) => Cow::Owned(changed),
        }
    }

    fn redact_exact<'a>(&self, text: &'a str) -> Cow<'a, str> {
        let mut out = Cow::Borrowed(text);
        for secret in &self.secrets {
            if out.contains(secret.as_str()) {
                out = Cow::Owned(out.replace(secret.as_str(), REDACTED));
            }
        }
        out
    }

    /// Redact every string in a JSON value, in place. Object keys are kept.
    pub fn redact_json(&self, value: &mut Value) {
        match value {
            Value::Object(map) => self.redact_map(map),
            Value::Array(items) => items.iter_mut().for_each(|item| self.redact_json(item)),
            Value::String(s) => {
                if let Cow::Owned(redacted) = self.redact(s) {
                    *s = redacted;
                }
            }
            _ => {}
        }
    }

    fn redact_map(&self, map: &mut Map<String, Value>) {
        for (key, value) in map.iter_mut() {
            match value {
                Value::String(s) => {
                    if let Some(redacted) = self.redact_field(key, s) {
                        *s = redacted;
                    }
                }
                other => self.redact_json(other),
            }
        }
    }

    /// A string field's value redacted with its key in view, or `None` when
    /// nothing changes.
    fn redact_field(&self, key: &str, value: &str) -> Option<String> {
        let exact = self.redact_exact(value);
        if !self.patterns {
            return matches!(exact, Cow::Owned(_)).then(|| exact.into_owned());
        }
        let mut probe_map = Map::new();
        probe_map.insert(key.to_string(), Value::String(exact.to_string()));
        let probe = to_python_json(&Value::Object(probe_map));
        if let Cow::Owned(redacted) = apply_rules(&probe) {
            if let Ok(Value::Object(mut restored)) = serde_json::from_str::<Value>(&redacted)
                && let Some(Value::String(v)) = restored.remove(key)
            {
                return Some(v);
            }
            return Some(apply_rules(&exact).into_owned());
        }
        matches!(exact, Cow::Owned(_)).then(|| exact.into_owned())
    }
}

/// How a rule rewrites a match.
#[derive(Clone, Copy)]
enum Rewrite {
    /// Replace the whole match.
    Whole,
    /// Keep group 1, replace the rest.
    KeepPrefix,
    /// Keep group 1, then the marker and `@` (URL user info).
    UserInfo,
}

/// A regex plus the look-around 0.8's pattern had.
struct Rule {
    regex: Regex,
    rewrite: Rewrite,
    /// Python's `(?<![...])`: the character before the match must not be one of these.
    not_after: Option<fn(char) -> bool>,
    /// Any other look-around, checked on the match.
    check: Option<fn(&str, &Captures<'_>) -> bool>,
}

const ESCQ: &str = r#"(?:\\{0,8}["']|)"#;
const SECVAL: &str = r#"[^"'\s,}\\*&]+"#;
const NAME: &str = "[A-Za-z0-9_]{0,64}";

fn alnum(c: char) -> bool {
    c.is_ascii_alphanumeric()
}

fn word_or_dash(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '-'
}

fn rule(pattern: &str, rewrite: Rewrite) -> Rule {
    Rule {
        regex: Regex::new(pattern)
            .unwrap_or_else(|e| panic!("BenchFlow bug: bad redaction pattern {pattern}: {e}")),
        rewrite,
        not_after: None,
        check: None,
    }
}

static RULES: LazyLock<Vec<Rule>> = LazyLock::new(|| {
    use Rewrite::{KeepPrefix, UserInfo, Whole};
    vec![
        // Token families, redacted whole so a leak audit grep finds no key shape.
        rule(r"sk-ant-[a-zA-Z0-9_-]{12,}", Whole),
        rule(r"sk-(?:benchflow|svcacct|or-v1)-[A-Za-z0-9_-]{12,}", Whole),
        // OpenAI sk-proj-/sk-admin- keys always carry an uppercase character
        // in their first 201; lowercase kebab slugs (`sk-proj-refactor`) survive.
        Rule {
            check: Some(|_, caps| {
                let body = caps.get(1).map_or("", |m| m.as_str());
                body.chars().take(201).any(|c| c.is_ascii_uppercase())
            }),
            ..rule(r"sk-(?:proj|admin)-([A-Za-z0-9_-]{12,})", Whole)
        },
        rule(r"sk-[a-zA-Z0-9]{12,}", Whole),
        rule(r"AIzaSy[A-Za-z0-9_-]{20,}", Whole),
        // AWS access keys: exactly 16 characters, not part of a longer word.
        Rule {
            check: Some(|text, caps| {
                let end = caps.get(0).map_or(0, |m| m.end());
                !text[end..].starts_with(|c: char| c.is_ascii_uppercase() || c.is_ascii_digit())
            }),
            ..rule(r"(?:AKIA|ASIA)[A-Z0-9]{16}", Whole)
        },
        rule(r"dtn_[A-Za-z0-9_]{16,}", Whole),
        rule(r"(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}", Whole),
        rule(r"github_pat_[A-Za-z0-9_]{20,}", Whole),
        rule(r"xox[baprs]-[A-Za-z0-9-]{10,}", Whole),
        rule(r"gsk_[A-Za-z0-9]{20,}", Whole),
        rule(r"xai-[A-Za-z0-9]{20,}", Whole),
        rule(r"r8_[A-Za-z0-9]{20,}", Whole),
        rule(r"hf_[A-Za-z0-9]{20,}", Whole),
        rule(r"fw_[A-Za-z0-9]{20,}", Whole),
        // JSON Web Tokens, not starting mid-identifier.
        Rule {
            not_after: Some(word_or_dash),
            ..rule(
                r"eyJ[A-Za-z0-9_-]{10,512}\.[A-Za-z0-9_-]{6,512}\.[A-Za-z0-9_-]{20,512}",
                Whole,
            )
        },
        // Credentials in the user info of URLs with credential-bearing schemes.
        rule(
            r"((?:https?|ftp|ftps|postgres|postgresql|mysql|mongodb(?:\+srv)?|redis|rediss|amqp|amqps)://)[^\s/?#:@]{0,256}:[^\s/?#]{0,256}@",
            UserInfo,
        ),
        // Secrets in query parameters with curated names.
        rule(
            r#"(?i)([?&](?:api_?key|access_key|access_token|refresh_token|session_token|session_id|sessionid|client_secret|aws_secret_access_key|account_key|secret|password_hash|password|passwd|signature|x-amz-signature|x-amz-credential|x-amz-security-token|x-goog-signature|x-goog-credential|sas)=)[^&\s"'*]+"#,
            KeepPrefix,
        ),
        rule(r#"(?i)([?&]sig=)[^&\s"'*]{16,}"#, KeepPrefix),
        // `name: value` carriers: the name is kept, the value dropped.
        rule(
            &format!(
                r"(?i)((?:{ESCQ}(?:x-api-key|x-goog-api-key|api-key){ESCQ})\s*[:=]\s*{ESCQ}){SECVAL}"
            ),
            KeepPrefix,
        ),
        rule(
            &format!(r"(?i)({ESCQ}api[_-]?key{ESCQ}\s*[:=]\s*{ESCQ}){SECVAL}"),
            KeepPrefix,
        ),
        rule(
            &format!(r"(?i)({ESCQ}(?:master|private)[_-]key{ESCQ}\s*[:=]\s*{ESCQ}){SECVAL}"),
            KeepPrefix,
        ),
        Rule {
            not_after: Some(alnum),
            ..rule(
                &format!(r"(?i)({ESCQ}{NAME}BEARER_TOKEN{NAME}{ESCQ}\s*[:=]\s*{ESCQ}){SECVAL}"),
                KeepPrefix,
            )
        },
        Rule {
            not_after: Some(alnum),
            ..rule(
                &format!(
                    r"(?i)({ESCQ}{NAME}(?:TOKEN|SECRET|PASSWORD|PASSWD|ACCESS_KEY|SECRET_KEY|ACCOUNT_KEY|PRIVATE_KEY|ENCRYPTION_KEY|CREDENTIALS?){ESCQ}\s*[:=]\s*{ESCQ}){SECVAL}"
                ),
                KeepPrefix,
            )
        },
        // An authorization header with a scheme keeps the scheme.
        Rule {
            not_after: Some(word_or_dash),
            ..rule(
                &format!(
                    r"(?i)({ESCQ}authorization{ESCQ}\s*[:=]\s*{ESCQ}(?:Bearer|Token|Basic|ApiKey)\s+){SECVAL}"
                ),
                KeepPrefix,
            )
        },
        // One with a bare value; scheme-prefixed values were handled above,
        // and a Python string prefix (`r"`, `b'`) is not a value.
        Rule {
            not_after: Some(word_or_dash),
            check: Some(|text, caps| {
                let end = caps.get(1).map_or(0, |m| m.end());
                let rest = &text[end..];
                let lower = rest.get(..6).unwrap_or(rest).to_ascii_lowercase();
                let scheme = ["bearer", "token", "basic", "apikey"].iter().any(|scheme| {
                    lower.starts_with(scheme)
                        && !rest[scheme.len()..]
                            .starts_with(|c: char| c.is_alphanumeric() || c == '_')
                });
                let prefix_len = rest.chars().take_while(|c| "rRuUbBfF".contains(*c)).count();
                let string_prefix =
                    (1..=3).contains(&prefix_len) && rest[prefix_len..].starts_with(['"', '\'']);
                !scheme && !string_prefix
            }),
            ..rule(
                &format!(r"(?i)({ESCQ}authorization{ESCQ}\s*[:=]\s*{ESCQ}){SECVAL}"),
                KeepPrefix,
            )
        },
    ]
});

static PEM_BEGIN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"-----BEGIN [A-Z0-9 ]{0,40}PRIVATE KEY-----").expect("valid regex")
});
static PEM_END: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"-----END [A-Z0-9 ]{0,40}PRIVATE KEY-----").expect("valid regex"));

// Half PEM blocks. A body line is a run of base64 (`B`). Lines are split by
// whitespace or by a newline, return or tab escape as JSON writes it, at any
// escaping depth up to `ESCQ`'s (a key held on one line with `\n` escapes, as
// in a service-account file, inside a JSON field's probe) (`S`), and a line
// may start with a line-number gutter (`cat -n`, `nl`, `grep -n` and its
// `N-` context lines, `    21→`) (`G`). Recursive or multi-file grep puts
// the file name first (`./k/id_rsa-14-`, `key.pem-`); a prefix must hold a
// `/` or `.`, so a prose word before an END marker (`is:`) is not one, and
// the marker's own dashes never make one (redaction stays idempotent). Its
// letters are ASCII: a Unicode `\w` this long is over the regex size limit (`P`).
const HALF_B: &str = r"[A-Za-z0-9+/=]";
const HALF_S: &str = r"(?:\s|\\{1,8}[nrt])";
const HALF_P: &str = r"[A-Za-z0-9_~-]{0,128}[./][A-Za-z0-9_./~-]{0,128}?[:-]";
static HALF_G: LazyLock<String> = LazyLock::new(|| {
    format!(r"(?:(?:{HALF_P})?\d{{1,7}}(?::|-|\s|\\{{1,8}}t|→|\\u2192)|{HALF_P})")
});

/// A BEGIN marker and the key body after it (its END is in another value).
static PEM_HALF_BEGIN: LazyLock<Regex> = LazyLock::new(|| {
    let (b, s, g) = (HALF_B, HALF_S, HALF_G.as_str());
    Regex::new(&format!(
        r"(-----BEGIN [A-Z0-9 ]{{0,40}}PRIVATE KEY-----){s}*(?:{g}{s}*)?{b}{{16,}}(?:{s}+(?:{g}{s}*)?{b}{{16,}})*"
    ))
    .expect("valid regex")
});
/// The key body before an END marker (its BEGIN is in another value). The
/// last body line is a remainder and may be short (`AQIDBA==` ends an
/// OpenSSH RSA-3072 key), so the run next to the marker may have any length.
static PEM_HALF_END: LazyLock<Regex> = LazyLock::new(|| {
    let (b, s, g) = (HALF_B, HALF_S, HALF_G.as_str());
    Regex::new(&format!(
        r"(?:{b}{{16,}}{s}+(?:{g}{s}*)?)*{b}+{s}*(?:{g}{s}*)?(-----END [A-Z0-9 ]{{0,40}}PRIVATE KEY-----)"
    ))
    .expect("valid regex")
});

/// 0.8's patterns, in 0.8's order, after whole and half PEM blocks.
fn apply_rules(text: &str) -> Cow<'_, str> {
    let mut out = redact_pem(text);
    if let Cow::Owned(changed) = redact_pem_halves(&out) {
        out = Cow::Owned(changed);
    }
    for rule in RULES.iter() {
        if let Cow::Owned(changed) = apply_rule(rule, &out) {
            out = Cow::Owned(changed);
        }
    }
    out
}

/// Half PEM private-key blocks: a key split over two values (two tool
/// results, two screens, `cat a.key b.key` cut on both sides) leaves a BEGIN
/// half or an END half (or both, of two keys) in a value. Each value is
/// redacted on its own, so each half's body goes and its marker stays.
/// Whole blocks are gone by now ([`redact_pem`]), so any marker left is
/// unpaired, or further than 0.8's 8192 characters from its partner, where
/// the body next to it goes too.
pub(super) fn redact_pem_halves(text: &str) -> Cow<'_, str> {
    let mut out = Cow::Borrowed(text);
    if let Cow::Owned(changed) = PEM_HALF_BEGIN.replace_all(&out, format!("$1 {REDACTED}")) {
        out = Cow::Owned(changed);
    }
    if let Cow::Owned(changed) = redact_half_ends(&out) {
        out = Cow::Owned(changed);
    }
    out
}

/// The end of a text that stops inside a JSON escape: right after its
/// backslash, or inside a `\uXXXX` (Python-style JSON escapes non-ASCII).
static INSIDE_ESCAPE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\\(?:u[0-9A-Fa-f]{0,3})?$").expect("valid regex"));

/// [`PEM_HALF_END`]'s matches, each replaced by the redaction and its
/// marker. A match that starts inside an escape (the `n` of `\n`, the hex
/// digits of `\uff1a` in serialized JSON) is skipped and the search goes on
/// from the next character, so serialized text stays valid.
fn redact_half_ends(text: &str) -> Cow<'_, str> {
    let mut out = String::new();
    let mut copied = 0;
    let mut at = 0;
    while at <= text.len() {
        let Some(caps) = PEM_HALF_END.captures_at(text, at) else {
            break;
        };
        let (whole, marker) = (caps.get(0).expect("match"), caps.get(1).expect("marker"));
        let mut from = whole.start().saturating_sub(8);
        while !text.is_char_boundary(from) {
            from -= 1;
        }
        if INSIDE_ESCAPE.is_match(&text[from..whole.start()]) {
            at = next_char(text, whole.start());
            continue;
        }
        out.push_str(&text[copied..whole.start()]);
        out.push_str(REDACTED);
        out.push(' ');
        out.push_str(marker.as_str());
        copied = whole.end();
        at = whole.end();
    }
    if copied == 0 {
        Cow::Borrowed(text)
    } else {
        out.push_str(&text[copied..]);
        Cow::Owned(out)
    }
}

/// PEM private-key blocks, armour included, at most 8192 characters long
/// between the markers.
fn redact_pem(text: &str) -> Cow<'_, str> {
    let mut out = String::new();
    let mut copied = 0;
    let mut search = 0;
    while let Some(begin) = PEM_BEGIN.find_at(text, search) {
        let end = PEM_END
            .find_at(text, begin.end())
            .filter(|end| text[begin.end()..end.start()].chars().count() <= 8192);
        match end {
            Some(end) => {
                out.push_str(&text[copied..begin.start()]);
                out.push_str(REDACTED);
                copied = end.end();
                search = end.end();
            }
            None => search = next_char(text, begin.start()),
        }
    }
    if copied == 0 {
        Cow::Borrowed(text)
    } else {
        out.push_str(&text[copied..]);
        Cow::Owned(out)
    }
}

fn next_char(text: &str, at: usize) -> usize {
    at + text[at..].chars().next().map_or(1, char::len_utf8)
}

fn apply_rule<'a>(rule: &Rule, text: &'a str) -> Cow<'a, str> {
    let mut out = String::new();
    let mut copied = 0;
    let mut search = 0;
    while search <= text.len() {
        let Some(caps) = rule.regex.captures_at(text, search) else {
            break;
        };
        let whole = caps.get(0).expect("group 0 always matches");
        let before_ok = rule
            .not_after
            .is_none_or(|bad| !text[..whole.start()].chars().next_back().is_some_and(bad));
        if !before_ok || !rule.check.is_none_or(|check| check(text, &caps)) {
            search = next_char(text, whole.start());
            continue;
        }
        out.push_str(&text[copied..whole.start()]);
        match rule.rewrite {
            Rewrite::Whole => out.push_str(REDACTED),
            Rewrite::KeepPrefix => {
                out.push_str(caps.get(1).map_or("", |m| m.as_str()));
                out.push_str(REDACTED);
            }
            Rewrite::UserInfo => {
                out.push_str(caps.get(1).map_or("", |m| m.as_str()));
                out.push_str(REDACTED);
                out.push('@');
            }
        }
        copied = whole.end();
        search = if whole.is_empty() {
            next_char(text, whole.end())
        } else {
            whole.end()
        };
    }
    if copied == 0 && out.is_empty() {
        Cow::Borrowed(text)
    } else {
        out.push_str(&text[copied..]);
        Cow::Owned(out)
    }
}

#[cfg(test)]
mod tests;
