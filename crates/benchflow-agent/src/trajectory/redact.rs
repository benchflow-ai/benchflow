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
//!    escaped-JSON forms; and, beyond 0.8, OpenPGP private key blocks and
//!    the half of a private key that one value shows when it is split.
//!
//! [`Redactor::redact_record`] goes further on one record (a trainer
//! file's line): a key read in parts loses the body lines of the parts
//! between its markers.
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

    /// Key body lines with no marker of their own, across the strings of
    /// one record (a job-level trainer line). Run it after
    /// [`Self::redact_json`], or use [`Self::redact_record`].
    ///
    /// A key read in parts (`head`, `sed -n`, `tail`) leaves its middle
    /// part in a string with no BEGIN or END marker, which
    /// [`Self::redact_json`] cannot tell from other base64. This walks the
    /// text of the record's messages (verifiers' `prompt` and `completion`,
    /// ADP's `content` items) in file order; the rest of the record (task
    /// id, environment, model, ids, rewards) is never changed, though its
    /// length counts toward a span's reach. A marker alone on its line, a
    /// BEGIN followed by a body (or its redaction) or by nothing worded,
    /// and an END after a redacted body open a span (see [`KEY_SPAN`] for
    /// how far it reaches; an END marker does not close it, since the parts
    /// may come in any order); inside it, each
    /// string loses only its runs of key body lines (an optional
    /// line-number or file-name gutter, then 16 or more base64 characters;
    /// lines split at newlines, their `\n`/`\r` escapes, and quotes), each
    /// run replaced by one marker. Everything else stays: the
    /// `[tool_call: …]` title, a fence, a description, a Python `repr`
    /// wrapper, prose, object keys, numbers and the shape of the value. A
    /// string with nothing to remove is not touched, and a second walk
    /// changes nothing.
    pub fn redact_key_chunks(&self, value: &mut Value) {
        if self.patterns {
            KeyWalk::default().value(value);
        }
    }

    /// One trainer record (a line of `verifiers.jsonl` or `adp.jsonl`):
    /// walked for key parts ([`Self::redact_key_chunks`]), redacted value by
    /// value ([`Self::redact_json`]), and walked again. The first walk sees
    /// the markers of a key that one value shows whole, which the value pass
    /// replaces marker and all, in a record built from events nobody has
    /// redacted yet (a run's trajectory file is redacted event by event, so
    /// there it changes nothing); the second sees the half rule's work and
    /// measures the span on what the file holds, so the result comes out of
    /// a second call unchanged.
    pub fn redact_record(&self, value: &mut Value) {
        self.redact_key_chunks(value);
        self.redact_json(value);
        self.redact_key_chunks(value);
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

/// A private key's label: PEM's (`RSA PRIVATE KEY`, `OPENSSH PRIVATE KEY`,
/// `PRIVATE KEY`, …) and OpenPGP's armour (`PGP PRIVATE KEY BLOCK`).
const KEY_LABEL: &str = r"[A-Z0-9 ]{0,40}PRIVATE KEY(?: BLOCK)?";

static PEM_BEGIN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(&format!(r"-----BEGIN {KEY_LABEL}-----")).expect("valid regex"));
static PEM_END: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(&format!(r"-----END {KEY_LABEL}-----")).expect("valid regex"));

// Half PEM blocks. A body line is a run of base64 (`B`). Lines are split by
// whitespace or by a newline, return or tab escape as JSON writes it, at any
// escaping depth up to `ESCQ`'s (a key held on one line with `\n` escapes, as
// in a service-account file, inside a JSON field's probe) (`S`), and a line
// may start with a line-number gutter (`cat -n`, `nl`, `grep -n` and its
// `N-` context lines, `    21→`) (`G`). Recursive or multi-file grep puts
// the file name first (`./k/id_rsa-14-`, `key.pem-`, `/home/a@b/.ssh/id_rsa-`,
// `./密钥/id_rsa-`); a prefix must hold a `/` or `.`, so a prose word before
// an END marker (`is:`) is not one, and the marker's own dashes never make
// one (redaction stays idempotent). A name character is an ASCII letter,
// digit or one of `_~+@,%=-`, any non-ASCII character, or the `\uXXXX`
// escape a JSON field's probe writes for one (`P`). Not a space: with one,
// prose before an END marker (`Note e.g. the key is:`) would be a prefix, so
// a recursive grep under a folder whose name has a space keeps its body
// lines (a known limit, written in the redaction docs).
const HALF_B: &str = r"[A-Za-z0-9+/=]";
const HALF_S: &str = r"(?:\s|\\{1,8}[nrt])";
const HALF_NAME: &str = r"(?:[A-Za-z0-9_~+@,%=-]|[^\x00-\x7F]|\\{1,8}u[0-9A-Fa-f]{4})";
const HALF_PATH: &str = r"(?:[A-Za-z0-9_./~+@,%=-]|[^\x00-\x7F]|\\{1,8}u[0-9A-Fa-f]{4})";
static HALF_P: LazyLock<String> =
    LazyLock::new(|| format!(r"{HALF_NAME}{{0,128}}[./]{HALF_PATH}{{0,128}}?[:-]"));
static HALF_G: LazyLock<String> = LazyLock::new(|| {
    let p = HALF_P.as_str();
    format!(r"(?:(?:{p})?\d{{1,7}}(?::|-|\s|\\{{1,8}}t|→|\\u2192)|{p})")
});

/// An armour header line between a BEGIN marker and the body: OpenPGP's
/// `Version: GnuPG v1` or `Comment: …` (GnuPG 1.x, `--comment`, Keybase and
/// Sequoia write them), PEM's `Proc-Type: …` and `DEK-Info: …`. Its value
/// ends at a line split, real or escaped.
const HALF_H: &str = r"[A-Za-z][A-Za-z0-9-]{0,63}: (?:[^\n\r\\]|\\{1,8}[^\\nrt])*";

/// A BEGIN marker, any armour header lines and the blank line after them,
/// and the key body (its END is in another value). In a numbered view
/// (`cat -n`, the Read tool, `grep -n ''`, `grep -rn -A`) each header line
/// and the blank line carry a gutter of their own (review G1-R2-A-2).
static PEM_HALF_BEGIN: LazyLock<Regex> = LazyLock::new(|| {
    let (b, s, g, h) = (HALF_B, HALF_S, HALF_G.as_str(), HALF_H);
    Regex::new(&format!(
        r"(-----BEGIN {KEY_LABEL}-----){s}*(?:{g}{s}*|{h}{s}+)*{b}{{16,}}(?:{s}+(?:{g}{s}*)?{b}{{16,}})*"
    ))
    .expect("valid regex")
});
/// The key body before an END marker (its BEGIN is in another value). The
/// last body line is a remainder and may be short (`AQIDBA==` ends an
/// OpenSSH RSA-3072 key), so the run next to the marker may have any length.
/// A gutter comes only after a line split, so a gutter's first digits
/// (`27:-----END …`) are not a remainder; a short run with no body line
/// before it counts only at a line's start ([`redact_half_ends`]), so a
/// prose word (`ends at -----END …`) is not one either (review G1-R2-A-1).
/// OpenPGP armour puts a CRC-24 checksum line (`=XXXX`) between the body
/// and the marker; the remainder before it is short in GnuPG's default
/// keys, so without stepping over the checksum the rule took the checksum
/// alone and kept every body line (review G1-R1-A-1).
static PEM_HALF_END: LazyLock<Regex> = LazyLock::new(|| {
    let (b, s, g) = (HALF_B, HALF_S, HALF_G.as_str());
    Regex::new(&format!(
        r"(?P<lines>(?:{b}{{16,}}{s}+(?:{g}{s}*)?)*)(?P<last>{b}+)(?:{s}+(?:{g}{s}*)?={b}{{4}})?(?:{s}+(?:{g}{s}*)?)?(?P<marker>-----END {KEY_LABEL}-----)"
    ))
    .expect("valid regex")
});

/// How far a key's span reaches in a job-level trainer line
/// ([`Redactor::redact_key_chunks`]): 8,192 characters of the serialized
/// line (0.8's distance between a key's markers) after the latest of: its
/// BEGIN marker, an END marker with a redacted body just before it (a
/// `tail` part, review G1-R3-B-2) or alone on its line (`tail -1`, review
/// G1-R1-B-1 at 6d38ceed), and the last full-width body line the walk
/// removed ([`KEY_WIDTH`]). An END marker does not end it (review
/// G1-R2-B-1: `head`, then `tail`, then the middle with `sed -n`, kept the
/// middle part when the `tail` output's END closed the span). The cost is
/// that lines or quoted strings of 16 or more base64 characters and
/// nothing else (a commit hash, a path such as
/// `/usr/local/bin/python3`, `"application/json"`) in the 8,192
/// characters after a key's part,
/// or after a marker alone on its line in other output (a source file's
/// `END = "-----END RSA PRIVATE KEY-----"`, a call that closes right after
/// its quoted marker, or a JSON or dict field that holds only the marker,
/// `{"end": "-----END RSA PRIVATE KEY-----", "next": …}`), are replaced
/// too. Each part
/// of a key read in parts must start within it, so a part that starts
/// more than 8,192 characters after the part before
/// it (long output or messages between the reads) keeps its body lines in
/// the job files, and so does a part read before any part with a marker
/// (`sed -n` for the middle, then `head`): known limits, written in the
/// redaction docs. The bound keeps a key from redacting base64-like lines
/// (hashes, ids) in the rest of the record.
pub const KEY_SPAN: usize = 8192;

/// How many base64 characters a removed line needs to carry the span on
/// ([`KEY_SPAN`]): key bodies are wrapped at 64 (PEM, OpenPGP), 70
/// (OpenSSH) or 76 characters, and a SHA-1 hash or most paths are shorter,
/// so the hashes and paths after a key are removed only within 8,192
/// characters of it instead of each re-arming the span for the rest of the
/// record (review G1-R3-B-3). A SHA-256 hash (64) still re-arms it.
const KEY_WIDTH: usize = 48;

/// A key marker, for the walk.
static KEY_MARKER: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(r"-----(BEGIN|END) {KEY_LABEL}-----")).expect("valid regex")
});

/// What splits a string into lines for the walk: a newline or return, the
/// same as an escape at any depth (a Python `repr` block in a verifiers
/// message, JSON inside a string), or a quote (where a `repr` or JSON
/// string starts and ends).
static KEY_SPLIT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"\r\n|[\n\r]|\\{1,8}[nr]|\\{0,8}["']"#).expect("valid regex"));

/// A key body line, whole: an optional output prefix ([`OUTPUT_PREFIX`])
/// and gutter the half rules take, then 16 or more base64 characters, with
/// spaces or tabs (real or escaped) around.
static KEY_LINE: LazyLock<Regex> = LazyLock::new(|| {
    let (b, g) = (HALF_B, HALF_G.as_str());
    let pad = r"(?:[ \t]|\\{1,8}t)*";
    Regex::new(&format!(
        r"^{pad}(?:{OUTPUT_PREFIX}{pad})?(?:{g}{pad})?{b}{{16,}}{pad}$"
    ))
    .expect("valid regex")
});

/// A run of base64 characters, to measure a body line ([`KEY_WIDTH`]).
static KEY_RUN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(&format!("{HALF_B}+")).expect("valid regex"));

/// What may follow a BEGIN marker in its string, short of the body itself
/// (which the half rule has replaced by then), for the marker to open a
/// span: nothing with a letter or digit (the string was cut right after the
/// marker), escapes aside. A marker quoted in prose
/// (`starts with -----BEGIN RSA PRIVATE KEY----- and`) opens none.
static NO_WORDS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^(?:[^A-Za-z0-9\\]|\\+[A-Za-z]?)*$").expect("valid regex"));

/// What an agent puts before the first line of a command's output, which
/// the walk steps over at a line's start as it steps over padding: one or
/// more terminal escapes, each raw, in a `repr` (`\\x1b`) or in JSON
/// (`\\u001b`): a CSI with any final byte (OpenHands's `\x1b[?2004l`, a
/// color code, bracketed paste's `\x1b[200~`), an OSC ended by BEL or ST
/// (a window title, `\x1b]0;title\x07`, its text bounded at 256
/// characters, with no backslash in it), a character-set choice (`\x1b(B`) or a two-byte escape
/// (`\x1b=`, `\x1b>`, `\x1b7`, `\x1b8`); or Qwen Code's `Output: `, or
/// Hermes over ACP's `**output:** ` with or without its list dash. Without
/// it, a marker that `head -1`, `tail -1` or `grep -n END` showed was not
/// alone on its line and opened no span, and the middle part's first body
/// line was not a key line, so the job files kept body lines that the job
/// pass's text pass (which the walk replaced) had removed, with the
/// messages between the markers (reviews G1F-A-1 and G1F-A-2 at fb5d1c52,
/// G1-R7-A-1 at 8290c756). Through [`ALONE_LEAD`], the half rule's line
/// start ([`at_line_start`], for an END marker after a short remainder)
/// steps over the prefix too, in every redacted text (trajectories, logs,
/// exports), not only in the trainer files' walk: `Output: AQIDBA==` before
/// an END marker becomes `Output: ***REDACTED***` (review G1-R7-A-2). The
/// cost: in a span, a line of 16 or more base64 characters after the prefix
/// (`Output: ` and a commit hash, a color code and a hash) is removed with
/// its prefix, and a short word after a prefix right before an END marker
/// is removed with the half. Not stepped over, a known limit in the
/// README: other escapes (a C1 control, a lone BEL or SI, `ESC c`, a
/// device-control or tmux title string, a longer title or one with a
/// backslash), and an escape after a line's text (a color reset at its
/// end); no shipped agent puts these next to its output (review G1-R8-A-1
/// at c67f24dc).
const OUTPUT_PREFIX: &str = concat!(
    r"(?:(?:(?:\x1b|\\{1,8}x1b|\\{1,8}u001b)",
    r"(?:\[[0-?]*[ -/]*[@-~]",
    r"|\][^\x07\x1b\\\n\r]{0,256}(?:\x07|\\{1,8}x07|\\{1,8}u0007|\\{1,8}a|(?:\x1b|\\{1,8}x1b|\\{1,8}u001b)\\{1,16})",
    r"|[()*+][@-~]|[=>78]))+",
    r"|Output:|-?[ \t]*\*\*output:\*\*)"
);

/// Padding inside a line: spaces and tabs, real or escaped.
const LINE_PAD: &str = r"(?:[ \t]|\\{1,8}t)";

/// What the walk has written just before an END marker when the key's body
/// was there: the half rule's `***REDACTED*** -----END …`, or the walk's
/// own marker, then line splits and the END line's gutter. Such an END
/// carries the span on, or opens one (`tail` before the middle part).
static REDACTED_BEFORE: LazyLock<Regex> = LazyLock::new(|| {
    let (s, g) = (HALF_S, HALF_G.as_str());
    Regex::new(&format!(r"\*\*\*REDACTED\*\*\*(?:{s}|{g})*$")).expect("valid regex")
});

/// What may stand before a marker that stands alone on its line, from a
/// line split (a newline or return, real or escaped, or a quote): padding,
/// an output prefix ([`OUTPUT_PREFIX`]) and a gutter.
/// [`ALONE_LEAD_AT_START`] is the same from the string's start.
static ALONE_LEAD: LazyLock<Regex> = LazyLock::new(|| {
    let (g, pad) = (HALF_G.as_str(), LINE_PAD);
    Regex::new(&format!(
        r#"(?:[\n\r]|\\{{1,8}}[nr]|\\{{0,8}}["']){pad}*(?:{OUTPUT_PREFIX}{pad}*)?(?:{g}{pad}*)?$"#
    ))
    .expect("valid regex")
});
static ALONE_LEAD_AT_START: LazyLock<Regex> = LazyLock::new(|| {
    let (g, pad) = (HALF_G.as_str(), LINE_PAD);
    Regex::new(&format!(
        r"^{pad}*(?:{OUTPUT_PREFIX}{pad}*)?(?:{g}{pad}*)?$"
    ))
    .expect("valid regex")
});
/// What may follow a marker that stands alone on its line: padding and
/// quotes, then the line's end. After a quote, closing brackets may come
/// too: a verifiers `repr` block whose text ends at the marker ends
/// `…-----'}` (review G1-R2-A-1 at cd992801: `tail -1 id` with no newline
/// opened no span, so the middle read next kept every body line). Or,
/// after a quote, the wrapper's next field: a comma, a quoted field name
/// and a colon (review G1-R3-A-1 at 47a1a11a: Hermes's headless JSON
/// `{"output": "…-----", "exit_code": 0, …}`, or a text-first `repr` block
/// `{'text': '…-----', 'type': 'text'}`, opened no span). The cost: a
/// marker that a JSON or dict field holds alone (`{"end": "-----END RSA
/// PRIVATE KEY-----", "next": …}`) opens a span too, so strings of 16 or
/// more base64 characters and nothing else in the 8,192 characters after
/// it go, field names that long included, and the
/// marker itself may be gone from the output once a later pass redacts it.
static ALONE_TRAIL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(
        r#"^(?:{LINE_PAD}|\\{{0,8}}["'])*(?:(?:\\{{0,8}}["'][)\]}}]+(?:{LINE_PAD}|[)\]}}])*)?(?:$|[\n\r]|\\{{1,8}}[nr])|{LINE_PAD}*\\{{0,8}}["'][ \t]*,[ \t]*\\{{0,8}}["'][A-Za-z_][A-Za-z0-9_]{{0,63}}\\{{0,8}}["'][ \t]*:)"#
    ))
    .expect("valid regex")
});

/// The record's fields that hold its messages: verifiers' `prompt` and
/// `completion`, ADP's `content`.
const MESSAGE_LISTS: &[&str] = &["prompt", "completion", "content"];

/// A message's fields that hold its text: a verifiers message's `content`,
/// an ADP item's `content`, `reasoning_content` and `description` (a tool
/// call's title).
const MESSAGE_TEXT: &[&str] = &["content", "reasoning_content", "description"];

/// Where a value sits in a trainer record, for the walk.
#[derive(Clone, Copy, PartialEq)]
enum Scope {
    /// The record itself.
    Record,
    /// A list of messages ([`MESSAGE_LISTS`]).
    Messages,
    /// One message.
    Message,
    /// A message's text ([`MESSAGE_TEXT`]), all of it.
    Text,
    /// Anything else: the task id, the environment, the model, ids, rewards
    /// and metrics (review G1-R1-A-2: a task named `sshkeyrotationaudit`
    /// after a key's head part became `***REDACTED***`, and a trainer that
    /// groups rollouts by it lost the rollout). Counted, never changed, and
    /// a marker here opens no span.
    Metadata,
}

impl Scope {
    /// The scope of the value under `key` in an object of this scope.
    fn field(self, key: &str) -> Self {
        match self {
            Self::Record if MESSAGE_LISTS.contains(&key) => Self::Messages,
            Self::Message if MESSAGE_TEXT.contains(&key) => Self::Text,
            Self::Text => Self::Text,
            _ => Self::Metadata,
        }
    }

    /// The scope of an item in an array of this scope.
    fn item(self) -> Self {
        match self {
            Self::Messages => Self::Message,
            Self::Text => Self::Text,
            _ => Self::Metadata,
        }
    }
}

/// [`Redactor::redact_key_chunks`]'s state.
#[derive(Default)]
struct KeyWalk {
    /// Characters of the serialized line (`to_python_json`) before the
    /// current point, counted on what the walk writes, so a second walk
    /// measures the same distances.
    at: usize,
    /// Where the open span ends, if a key is open.
    open_until: Option<usize>,
}

impl KeyWalk {
    /// Walk a record: only its messages' text changes, and every value
    /// counts toward the span's reach.
    fn value(&mut self, value: &mut Value) {
        self.scoped(value, Scope::Record);
    }

    fn scoped(&mut self, value: &mut Value, scope: Scope) {
        match value {
            Value::Object(map) => {
                self.at += 1;
                for (i, (key, item)) in map.iter_mut().enumerate() {
                    // `, ` before all but the first, `"key": ` before the value.
                    self.at += if i == 0 { 0 } else { 2 } + py_json_len(key) + 4;
                    self.scoped(item, scope.field(key));
                }
                self.at += 1;
            }
            Value::Array(items) => {
                self.at += 1;
                for (i, item) in items.iter_mut().enumerate() {
                    self.at += if i == 0 { 0 } else { 2 };
                    self.scoped(item, scope.item());
                }
                self.at += 1;
            }
            Value::String(s) if scope == Scope::Text => {
                self.at += 1;
                if let Some(cut) = self.string(s) {
                    *s = cut;
                }
                self.at += 1;
            }
            Value::String(s) => self.at += py_json_len(s) + 2,
            other => self.at += to_python_json(other).len(),
        }
    }

    /// `s` without the key body lines inside the open span, or `None` when
    /// nothing goes. Moves `at` past the string's serialized text.
    fn string(&mut self, s: &str) -> Option<String> {
        if self.open_until.is_some_and(|until| self.at > until) {
            self.open_until = None;
        }
        if self.open_until.is_none() && !KEY_MARKER.is_match(s) {
            self.at += py_json_len(s);
            return None;
        }
        let mut out = String::with_capacity(s.len());
        let mut changed = false;
        // Separators after a removed line, held back so that a run of body
        // lines becomes one marker.
        let mut held = String::new();
        let mut in_run = false;
        let mut copied = 0;
        // Where the previous marker in `s` ends, and in `out`.
        let mut prev = None;
        let mut written = 0;
        let splits = KEY_SPLIT
            .find_iter(s)
            .map(|m| (m.start(), m.end()))
            .chain(std::iter::once((s.len(), s.len())));
        for (sep_start, sep_end) in splits {
            let line = &s[copied..sep_start];
            let start = copied;
            copied = sep_end;
            if !line.is_empty() {
                let pending = py_json_len(&held);
                if KEY_MARKER.is_match(line) {
                    self.flush(&mut out, &mut held);
                    in_run = false;
                    let mut done = 0;
                    for caps in KEY_MARKER.captures_iter(line) {
                        let marker = caps.get(0).expect("match");
                        self.push(&mut out, &line[done..marker.start()]);
                        // Read on what the walk wrote, back to the previous
                        // marker, so a second walk sees the same.
                        let tail_part =
                            &caps[1] == "END" && REDACTED_BEFORE.is_match(&out[written..]);
                        self.push(&mut out, marker.as_str());
                        written = out.len();
                        done = marker.end();
                        // The text back to the previous marker and on to
                        // the next one (which decides for itself) says
                        // whether a marker opens a span; reading no further
                        // keeps the walk linear. A marker alone on its line
                        // opens one, an END too (review G1-R1-B-1: `tail -1
                        // id` or `grep -n END id` shows only the END line,
                        // and the middle read after it kept its body lines).
                        // An END closes nothing: a key's parts may come in
                        // any order (`head`, `tail`, then `sed -n` for the
                        // middle), so the span runs on to `KEY_SPAN` past
                        // its last removed line.
                        let (before, after) = (start + marker.start(), start + marker.end());
                        let next = KEY_MARKER.find_at(s, after).map_or(s.len(), |m| m.start());
                        let alone = alone(
                            &s[prev.unwrap_or(0)..before],
                            prev.is_none(),
                            &s[after..next],
                        );
                        if tail_part || alone || &caps[1] == "BEGIN" && opens(&s[after..next]) {
                            self.open_until = Some(self.at + KEY_SPAN);
                        }
                        prev = Some(after);
                    }
                    self.push(&mut out, &line[done..]);
                } else if self
                    .open_until
                    .is_some_and(|until| self.at + pending <= until)
                    && KEY_LINE.is_match(line)
                {
                    if in_run {
                        held.clear();
                    } else {
                        self.flush(&mut out, &mut held);
                        self.push(&mut out, REDACTED);
                    }
                    in_run = true;
                    changed = true;
                    if KEY_RUN.find_iter(line).any(|run| run.len() >= KEY_WIDTH) {
                        self.open_until = Some(self.at + KEY_SPAN);
                    }
                } else {
                    self.flush(&mut out, &mut held);
                    in_run = false;
                    self.push(&mut out, line);
                }
            }
            let sep = &s[sep_start..sep_end];
            if in_run {
                held.push_str(sep);
            } else {
                self.push(&mut out, sep);
            }
        }
        self.flush(&mut out, &mut held);
        changed.then_some(out)
    }

    fn push(&mut self, out: &mut String, text: &str) {
        out.push_str(text);
        self.at += py_json_len(text);
    }

    fn flush(&mut self, out: &mut String, held: &mut String) {
        if !held.is_empty() {
            let text = std::mem::take(held);
            self.push(out, &text);
        }
    }
}

/// Whether a marker stands alone on its line, given the text `before` it
/// (back to the previous marker, or to the string's start when `at_start`)
/// and `rest` after it: only padding, quotes and a gutter (`12:`,
/// `./k/id_rsa:1:`) with it. A BEGIN that does opens a span whatever the
/// later lines of its output say (`head -1 id; wc -l id`).
fn alone(before: &str, at_start: bool, rest: &str) -> bool {
    ALONE_TRAIL.is_match(rest)
        && (ALONE_LEAD.is_match(before) || at_start && ALONE_LEAD_AT_START.is_match(before))
}

/// Whether a BEGIN marker that shares its line with other text, followed by
/// `rest` (its string up to the next marker), opens a span: the half rule
/// replaced a body after it, a key body line follows it (one the half rule
/// could not reach, past a header or a fence), or nothing worded does.
fn opens(rest: &str) -> bool {
    rest.contains(REDACTED)
        || NO_WORDS.is_match(rest)
        || KEY_SPLIT.split(rest).any(|line| KEY_LINE.is_match(line))
}

/// How many characters `s` takes inside a string of `to_python_json`'s
/// output: Python's `ensure_ascii` escapes.
fn py_json_len(s: &str) -> usize {
    s.chars()
        .map(|c| match c {
            '"' | '\\' | '\n' | '\r' | '\t' | '\u{8}' | '\u{c}' => 2,
            c if (c as u32) < 0x20 => 6,
            c if (c as u32) < 0x7f => 1,
            c => 6 * c.len_utf16(),
        })
        .sum()
}

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

/// A line split, real or escaped.
static LINE_SPLIT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[\n\r]|\\{1,8}[nr]").expect("valid regex"));

/// [`PEM_HALF_END`]'s matches, each replaced by the redaction and its
/// marker. A match that starts inside an escape (the `n` of `\n`, the hex
/// digits of `\uff1a` in serialized JSON) is skipped and the search goes on
/// from the next character, so serialized text stays valid.
fn redact_half_ends(text: &str) -> Cow<'_, str> {
    // The rule starts with a body, not the marker, so its search finds no
    // literal to skip ahead with; most values hold no marker at all.
    if !text.contains("-----END ") {
        return Cow::Borrowed(text);
    }
    let mut out = String::new();
    let mut copied = 0;
    let mut at = 0;
    while at <= text.len() {
        let Some(caps) = PEM_HALF_END.captures_at(text, at) else {
            break;
        };
        let (whole, marker) = (caps.get(0).expect("match"), &caps["marker"]);
        let mut from = whole.start().saturating_sub(8);
        while !text.is_char_boundary(from) {
            from -= 1;
        }
        let last = caps.name("last").expect("group");
        let short = caps["lines"].is_empty() && last.len() < 16;
        // A short run of digits on the marker's own line is its gutter
        // (`    27\t-----END …` in `cat -n id | grep KEY`), not a remainder:
        // taking it made a second pass redact the listing (review G1-R3-A-1).
        let gutter = short
            && last.as_str().bytes().all(|b| b.is_ascii_digit())
            && !LINE_SPLIT.is_match(&text[last.end()..caps.name("marker").expect("group").start()]);
        if INSIDE_ESCAPE.is_match(&text[from..whole.start()])
            || short && !at_line_start(text, whole.start())
            || gutter
        {
            at = next_char(text, whole.start());
            continue;
        }
        out.push_str(&text[copied..whole.start()]);
        out.push_str(REDACTED);
        out.push(' ');
        out.push_str(marker);
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

/// Whether `at` in `text` starts a line: after a line split (a newline or
/// return, real or escaped, or a quote) or the text's start, then padding
/// and a gutter. Looks back 512 bytes at most; a line that starts further
/// back is not a key's.
fn at_line_start(text: &str, at: usize) -> bool {
    let mut from = at.saturating_sub(512);
    while !text.is_char_boundary(from) {
        from -= 1;
    }
    let before = &text[from..at];
    ALONE_LEAD.is_match(before) || from == 0 && ALONE_LEAD_AT_START.is_match(before)
}

/// What stands between the markers of a pair with no body: line splits and
/// gutters only.
static NO_BODY: LazyLock<Regex> = LazyLock::new(|| {
    let (s, g) = (HALF_S, HALF_G.as_str());
    Regex::new(&format!(r"^(?:{s}|{g})*$")).expect("valid regex")
});

/// PEM private-key blocks, armour included, at most 8192 characters long
/// between the markers. A pair with nothing but line splits and gutters
/// between its markers (a `grep -n` listing of both marker lines) holds no
/// secret and stays, so a trainer file's walk still sees its BEGIN and
/// removes the parts read after it (review G1-R2-A-1: each event is
/// redacted before the trainer files are built from the trajectory).
///
/// An END pairs with the nearest BEGIN before it (0.8 paired it with the
/// first): in `grep -rn 'PRIVATE KEY' .`, a source file that quotes a BEGIN
/// marker comes before the key file's two marker lines, and pairing that
/// quote with the key's END replaced the key's BEGIN too, so the walk never
/// saw it (review G1-R3-B-1). A BEGIN left with no END of its own is a half,
/// and the half rule takes the body after it. Searching for the END only up
/// to the next BEGIN keeps this linear.
fn redact_pem(text: &str) -> Cow<'_, str> {
    let mut out = String::new();
    let mut copied = 0;
    let mut search = 0;
    while let Some(begin) = PEM_BEGIN.find_at(text, search) {
        let next = PEM_BEGIN.find_at(text, begin.end());
        let upto = next.map_or(text.len(), |next| next.start());
        let end = PEM_END.find_at(&text[..upto], begin.end()).filter(|end| {
            let between = &text[begin.end()..end.start()];
            between.chars().count() <= 8192 && !NO_BODY.is_match(between)
        });
        match (end, next) {
            (Some(end), _) => {
                out.push_str(&text[copied..begin.start()]);
                out.push_str(REDACTED);
                copied = end.end();
                search = end.end();
            }
            (None, Some(next)) => search = next.start(),
            (None, None) => break,
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
