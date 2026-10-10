//! Keeping secrets out of error messages and logs.
//!
//! A [`Redactor`] holds the secret values a backend knows (an API key, the
//! values of secret-looking environment variables) and replaces every
//! occurrence of them in a piece of text with [`MASK`]. It also masks any
//! `Bearer <token>` it finds, so an echoed `Authorization` header never leaks
//! even when the token is one the redactor was not told about.
//!
//! Every error message a backend builds from text it did not write itself
//! (a CLI's stderr, an HTTP response body, a request URL) goes through
//! [`Redactor::redact`] first.

use std::collections::BTreeMap;
use std::fmt;

/// What replaces a secret in text.
pub const MASK: &str = "[redacted]";

/// Values shorter than this are never treated as secrets: masking a short
/// value such as `1` or `true` would garble every message containing it.
const MIN_SECRET_LEN: usize = 6;

/// Masks known secret values, and bearer tokens, in text.
#[derive(Clone, Default)]
pub struct Redactor {
    /// Longest first, so a secret that contains another is masked whole.
    secrets: Vec<String>,
}

impl fmt::Debug for Redactor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Never print the secrets themselves.
        f.debug_struct("Redactor")
            .field("secrets", &self.secrets.len())
            .finish()
    }
}

impl Redactor {
    pub fn new() -> Self {
        Self::default()
    }

    /// Mask `secret` from now on. Values shorter than six characters are
    /// ignored (see the module docs).
    pub fn add(&mut self, secret: impl Into<String>) {
        let secret = secret.into();
        if secret.chars().count() < MIN_SECRET_LEN || self.secrets.contains(&secret) {
            return;
        }
        self.secrets.push(secret);
        self.secrets.sort_by_key(|s| std::cmp::Reverse(s.len()));
    }

    /// Builder form of [`Redactor::add`].
    pub fn with(mut self, secret: impl Into<String>) -> Self {
        self.add(secret);
        self
    }

    /// Mask the values of the variables in `env` whose names look secret
    /// (see [`looks_secret`]).
    pub fn add_env(&mut self, env: &BTreeMap<String, String>) {
        for (key, value) in env {
            if looks_secret(key) {
                self.add(value.clone());
            }
        }
    }

    /// True when the redactor knows no secret values (it still masks bearer
    /// tokens).
    pub fn is_empty(&self) -> bool {
        self.secrets.is_empty()
    }

    /// `text` with every known secret and every bearer token masked.
    pub fn redact(&self, text: &str) -> String {
        let mut out = text.to_string();
        for secret in &self.secrets {
            if out.contains(secret.as_str()) {
                out = out.replace(secret.as_str(), MASK);
            }
        }
        mask_bearer_tokens(&out)
    }
}

/// Whether an environment variable name suggests its value is a secret:
/// `ANTHROPIC_API_KEY`, `GITHUB_TOKEN`, `DB_PASSWORD`, `AWS_SECRET_ACCESS_KEY`.
pub fn looks_secret(key: &str) -> bool {
    let key = key.to_ascii_uppercase();
    [
        "KEY",
        "TOKEN",
        "SECRET",
        "PASSWORD",
        "PASSWD",
        "CREDENTIAL",
        "AUTH",
        "COOKIE",
        "PRIVATE",
    ]
    .iter()
    .any(|marker| key.contains(marker))
}

/// `text` with the token after every `Bearer ` (any case) masked.
pub fn mask_bearer_tokens(text: &str) -> String {
    const WORD: &str = "bearer ";
    let lower = text.to_ascii_lowercase();
    if !lower.contains(WORD) {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len());
    let mut rest = 0;
    let mut search = 0;
    while let Some(found) = lower[search..].find(WORD) {
        let token_start = search + found + WORD.len();
        let token_len = text[token_start..]
            .bytes()
            .take_while(|b| b.is_ascii_alphanumeric() || b"-._~+/=".contains(b))
            .count();
        if token_len > 0 {
            out.push_str(&text[rest..token_start]);
            out.push_str(MASK);
            rest = token_start + token_len;
        }
        search = token_start + token_len;
    }
    out.push_str(&text[rest..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masks_known_secrets_everywhere() {
        let r = Redactor::new().with("dtn_0123456789abcdef");
        assert_eq!(
            r.redact("key dtn_0123456789abcdef rejected; dtn_0123456789abcdef again"),
            "key [redacted] rejected; [redacted] again"
        );
    }

    #[test]
    fn ignores_short_values_and_masks_longest_first() {
        let mut r = Redactor::new();
        r.add("1");
        r.add("abcdef");
        r.add("abcdefghij");
        assert_eq!(r.redact("1 abcdefghij abcdef"), "1 [redacted] [redacted]");
    }

    #[test]
    fn masks_bearer_tokens_it_was_not_told_about() {
        assert_eq!(
            mask_bearer_tokens("Authorization: Bearer abc.DEF-123/x= then more"),
            "Authorization: Bearer [redacted] then more"
        );
        assert_eq!(
            mask_bearer_tokens("bearer t1 and BEARER t2"),
            "bearer [redacted] and BEARER [redacted]"
        );
        assert_eq!(mask_bearer_tokens("no token here"), "no token here");
        assert_eq!(mask_bearer_tokens("Bearer "), "Bearer ");
    }

    #[test]
    fn env_values_with_secret_names_are_masked() {
        let mut env = BTreeMap::new();
        env.insert("ANTHROPIC_API_KEY".to_string(), "sk-ant-secret".to_string());
        env.insert("HOME".to_string(), "/home/agent".to_string());
        let mut r = Redactor::new();
        r.add_env(&env);
        assert_eq!(
            r.redact("sk-ant-secret in /home/agent"),
            "[redacted] in /home/agent"
        );
    }

    #[test]
    fn debug_never_prints_secrets() {
        let r = Redactor::new().with("supersecret");
        assert_eq!(format!("{r:?}"), "Redactor { secrets: 1 }");
    }
}
