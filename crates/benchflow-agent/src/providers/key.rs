//! Finding a provider's key: its variables, then its `NAME=value` files.

use std::fmt;
use std::path::PathBuf;

use super::Provider;
use crate::credentials::{CredentialError, HostEnv, Login, MIN_SECRET_LEN, PlainKey};

/// A provider's key, found on this machine. It goes to the forwarder's
/// root-only key file and nowhere else; `Debug` never shows it.
#[derive(Clone, PartialEq, Eq)]
pub struct ProviderKey {
    pub provider: String,
    /// Where it was found, for people: `AI_GATEWAY_API_KEY (environment)`,
    /// `AI_GATEWAY_API_KEY in ~/.config/benchflow/vercel-ai-gateway.env`.
    pub source: String,
    value: String,
}

impl fmt::Debug for ProviderKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProviderKey")
            .field("provider", &self.provider)
            .field("source", &self.source)
            .field("value", &format_args!("<{} chars>", self.value.len()))
            .finish()
    }
}

/// What the forwarder sends in place of the key to a plain-http test
/// endpoint ([`ProviderKey::withheld`]).
pub const WITHHELD_KEY: &str = "benchflow-test-endpoint-placeholder";

impl ProviderKey {
    /// The key itself, for the forwarder's key file. Never put it in an
    /// environment variable, a command line, or a log.
    pub fn value(&self) -> &str {
        &self.value
    }

    /// A login that labels the key and redacts it everywhere, without giving
    /// the agent anything: no variables, no files. `via` says which endpoint
    /// it serves.
    pub fn login(&self, via: &str) -> Login {
        let mut login = Login::none();
        login.label = format!("{}, held by BenchFlow's forwarder for {via}", self.source);
        login.add_secret(&self.value);
        login
    }

    /// The plain API key the agent's own login holds, for the forwarder.
    pub(crate) fn from_login(provider: &str, key: &PlainKey) -> Self {
        Self {
            provider: provider.to_string(),
            source: key.source.clone(),
            value: key.value.trim().to_string(),
        }
    }

    /// A stand-in for this key, for an upstream reached over plain http
    /// (the test hook's): the real key never travels unencrypted, and a
    /// listener there, which inside a sandbox the agent may own, never sees
    /// it.
    pub(crate) fn withheld(&self) -> Self {
        Self {
            provider: self.provider.clone(),
            source: format!(
                "{} (withheld: a placeholder goes to the plain-http test endpoint)",
                self.source
            ),
            value: WITHHELD_KEY.to_string(),
        }
    }

    #[cfg(test)]
    pub(crate) fn for_tests(provider: &str, value: &str) -> Self {
        Self {
            provider: provider.to_string(),
            source: "TEST_KEY (environment)".to_string(),
            value: value.to_string(),
        }
    }
}

/// Where a provider's key would come from here, without the value: for
/// `bench doctor` and `bench agents`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyStatus {
    pub provider: String,
    /// The source [`find_provider_key`] would use.
    pub found: Option<String>,
    /// What is looked at, in order.
    pub checked: Vec<String>,
    /// What to do when nothing was found.
    pub fix: Option<String>,
}

/// The variables and files a provider's key is read from, in order, and the
/// one that would be used.
pub fn key_status(provider: &Provider, host: &HostEnv) -> KeyStatus {
    let (checked, next_step) = checked_and_next_step(provider);
    match find_provider_key(provider, host) {
        Ok(key) => KeyStatus {
            provider: provider.name.clone(),
            found: Some(key.source),
            checked,
            fix: None,
        },
        Err(_) => KeyStatus {
            provider: provider.name.clone(),
            found: None,
            checked,
            fix: Some(next_step),
        },
    }
}

/// `provider`'s key: the first of its variables that is set (a blank value
/// counts as unset), else the first of its files that sets one of them.
pub fn find_provider_key(
    provider: &Provider,
    host: &HostEnv,
) -> Result<ProviderKey, CredentialError> {
    for var in &provider.key.env {
        if let Some((value, origin)) = host.get(var) {
            return Ok(ProviderKey {
                provider: provider.name.clone(),
                source: format!("{var} ({origin})"),
                value: value.trim().to_string(),
            });
        }
    }
    for file in &provider.key.files {
        let Some(path) = expand(file, host) else {
            continue;
        };
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let vars = parse_env_file(&text);
        for var in &provider.key.env {
            if let Some((_, value)) = vars
                .iter()
                .find(|(name, value)| name == var && !value.is_empty())
            {
                return Ok(ProviderKey {
                    provider: provider.name.clone(),
                    source: format!("{var} in {file}"),
                    value: value.clone(),
                });
            }
        }
    }
    let (checked, next_step) = checked_and_next_step(provider);
    Err(CredentialError::NoProviderKey {
        provider: provider.name.clone(),
        checked,
        next_step,
    })
}

fn checked_and_next_step(provider: &Provider) -> (Vec<String>, String) {
    let mut checked: Vec<String> = provider.key.env.clone();
    checked.extend(provider.key.files.iter().cloned());
    let var = provider.key.env.first().map_or("<NAME>", String::as_str);
    let mut next_step = format!("`export {var}=<key>`");
    if let Some(file) = provider.key.files.first() {
        next_step.push_str(&format!(", or put `{var}=<key>` in {file}"));
    }
    (checked, next_step)
}

/// `~/x` under the host's home; absolute paths as they are.
fn expand(file: &str, host: &HostEnv) -> Option<PathBuf> {
    match file.strip_prefix("~/") {
        Some(rest) => host.home().map(|home| home.join(rest)),
        None => Some(PathBuf::from(file)),
    }
}

/// `NAME=value` lines (with an optional `export `), `#` comments, and
/// values in single or double quotes. Values shorter than a secret are
/// dropped: they cannot be a key.
fn parse_env_file(text: &str) -> Vec<(String, String)> {
    let mut vars = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line = line.strip_prefix("export ").unwrap_or(line).trim_start();
        let Some((name, value)) = line.split_once('=') else {
            continue;
        };
        let name = name.trim();
        let value = value.trim();
        let value = if let Some(inner) = value
            .strip_prefix('"')
            .and_then(|v| v.strip_suffix('"'))
            .or_else(|| value.strip_prefix('\'').and_then(|v| v.strip_suffix('\'')))
        {
            inner.to_string()
        } else {
            // An unquoted value ends at a ` #` comment.
            value
                .split_once(" #")
                .map_or(value, |(v, _)| v)
                .trim()
                .to_string()
        };
        if value.len() >= MIN_SECRET_LEN && !value.chars().any(char::is_whitespace) {
            vars.push((name.to_string(), value));
        }
    }
    vars
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_files_take_exports_quotes_and_comments() {
        let vars = parse_env_file(
            "# a comment\n\nexport A_KEY=\"quoted-value-1234\"\nB_KEY='single-quoted-99'\nC_KEY=plain-value-5678 # note\nSHORT=abc\nnot a line\n",
        );
        assert_eq!(
            vars,
            [
                ("A_KEY".to_string(), "quoted-value-1234".to_string()),
                ("B_KEY".to_string(), "single-quoted-99".to_string()),
                ("C_KEY".to_string(), "plain-value-5678".to_string()),
            ]
        );
    }
}
