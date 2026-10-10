//! The manifest schema's keys, and positions for error messages.
//!
//! Serde reports an unknown field without a suggestion, so before the
//! manifest is deserialized every key is checked against [`ROOT`] here, and
//! an unknown one is reported at its line and column with the closest
//! allowed key. The same parsed document answers "where is this key?" for
//! the validation errors that come after.

use std::ops::Range;

use toml::Spanned;
use toml::de::{DeTable, DeValue};

use super::{ManifestError, ManifestSource, UnknownKeyError};

/// What a key holds.
pub(crate) enum Kind {
    /// A value whose type serde checks.
    Value,
    /// A table with its own fixed keys.
    Table(&'static [(&'static str, Kind)]),
    /// A table with free-form keys (environment variables).
    Map,
    /// An array of tables, each with these keys.
    Tables(&'static [(&'static str, Kind)]),
    /// A table with free-form keys, each a table with these keys
    /// (`[options.<name>]`).
    Entries(&'static [(&'static str, Kind)]),
}

use Kind::{Entries, Tables};
pub(crate) use Kind::{Map, Table, Value};

/// A schema: the keys a table takes, and what each holds.
pub(crate) type Schema = &'static [(&'static str, Kind)];

const ROOT: Schema = &[
    ("name", Value),
    ("aliases", Value),
    ("description", Value),
    ("skill_paths", Value),
    ("skills_via", Value),
    ("home_dirs", Value),
    ("harness", Value),
    ("acp_registry", Value),
    ("install", Table(INSTALL)),
    ("launch", Table(LAUNCH)),
    ("headless", Table(HEADLESS)),
    ("credentials", Table(CREDENTIALS)),
    ("model", Table(MODEL)),
    ("acp", Table(ACP)),
    ("providers", Table(PROVIDERS)),
    ("options", Entries(OPTION)),
    ("resume", Table(RESUME)),
    ("resources", Table(RESOURCES)),
];
const RESOURCES: &[(&str, Kind)] = &[("min_memory", Value), ("min_cpus", Value)];
const OPTION: &[(&str, Kind)] = &[
    ("type", Value),
    ("description", Value),
    ("default", Value),
    ("choices", Value),
    ("min", Value),
    ("max", Value),
    ("env", Value),
    ("flag", Value),
    ("file", Value),
    ("key", Value),
    ("install", Value),
    ("harnesses", Value),
];
const RESUME: &[(&str, Kind)] = &[
    ("acp", Value),
    ("headless", Value),
    ("session_dirs", Value),
    ("load", Value),
];
const INSTALL: &[(&str, Kind)] = &[
    ("npm", Value),
    ("bin", Value),
    ("companions", Value),
    ("launcher", Value),
    ("launcher_env", Map),
    ("verify", Value),
    ("script", Value),
    ("uv-tool", Value),
    ("uv-venv", Value),
    ("python", Value),
    ("installer", Value),
    ("links", Value),
    ("timeout", Value),
    // ACP registry distributions (`distribution.rs`).
    ("binary", Map),
    ("npx", Value),
    ("uvx", Value),
];
const LAUNCH: &[(&str, Kind)] = &[("command", Value), ("args", Value), ("env", Map)];
const HEADLESS: &[(&str, Kind)] = &[
    ("command", Value),
    ("output", Value),
    ("env", Map),
    ("files", Tables(HOME_FILE)),
];
const HOME_FILE: &[(&str, Kind)] = &[
    ("path", Value),
    ("contents", Value),
    ("runner", Value),
    ("mode", Value),
];
const CREDENTIALS: &[(&str, Kind)] = &[("family", Value), ("env", Value), ("account", Value)];
const MODEL: &[(&str, Kind)] = &[
    ("select", Value),
    ("config_id", Value),
    ("effort_config_id", Value),
    ("format", Value),
    ("default", Value),
    ("strip_prefixes", Value),
];
const ACP: &[(&str, Kind)] = &[("typed_failures", Value), ("subagent_transcript", Value)];
const PROVIDERS: &[(&str, Kind)] = &[
    ("anthropic", Table(AGENT_PROVIDER)),
    ("openai", Table(AGENT_PROVIDER)),
    ("gateway", Table(AGENT_PROVIDER)),
];
const AGENT_PROVIDER: &[(&str, Kind)] = &[
    ("api", Value),
    ("base_url", Value),
    ("key", Value),
    ("clear", Value),
    ("env", Map),
    ("model", Value),
    ("effort", Value),
];

/// A parsed manifest with the position of every key and value.
pub(crate) struct Document<'a> {
    root: Spanned<DeTable<'a>>,
    text: &'a str,
    source: ManifestSource,
    line_starts: Vec<usize>,
}

impl<'a> Document<'a> {
    pub(crate) fn parse(text: &'a str, source: &ManifestSource) -> Result<Self, ManifestError> {
        let line_starts = std::iter::once(0)
            .chain(text.match_indices('\n').map(|(i, _)| i + 1))
            .collect();
        match DeTable::parse(text) {
            Ok(root) => Ok(Self {
                root,
                text,
                source: source.clone(),
                line_starts,
            }),
            Err(e) => {
                let doc = Self {
                    root: Spanned::new(0..0, DeTable::default()),
                    text,
                    source: source.clone(),
                    line_starts,
                };
                let (line, column) = e.span().map(|s| doc.position(s.start)).unwrap_or((0, 0));
                Err(ManifestError::Syntax {
                    origin: source.clone(),
                    line,
                    column,
                    message: e.message().trim().to_string(),
                })
            }
        }
    }

    /// Fail on the first key (in document order) that the agent manifest
    /// schema lacks.
    pub(super) fn check_keys(&self) -> Result<(), ManifestError> {
        self.check_against(ROOT)
    }

    /// Fail on the first key (in document order) that `schema` lacks.
    pub(crate) fn check_against(&self, schema: Schema) -> Result<(), ManifestError> {
        self.check_table(self.root.get_ref(), schema, "")
    }

    fn check_table(
        &self,
        table: &DeTable<'_>,
        schema: &'static [(&'static str, Kind)],
        prefix: &str,
    ) -> Result<(), ManifestError> {
        for (key, value) in table.iter() {
            let name: &str = key.get_ref();
            let path = if prefix.is_empty() {
                name.to_string()
            } else {
                format!("{prefix}.{name}")
            };
            match schema.iter().find(|(allowed, _)| *allowed == name) {
                Some((_, Table(inner))) => {
                    if let DeValue::Table(inner_table) = value.get_ref() {
                        self.check_table(inner_table, inner, &path)?;
                    }
                }
                Some((_, Tables(inner))) => {
                    if let DeValue::Array(items) = value.get_ref() {
                        for (i, item) in items.iter().enumerate() {
                            if let DeValue::Table(item_table) = item.get_ref() {
                                self.check_table(item_table, inner, &format!("{path}[{i}]"))?;
                            }
                        }
                    }
                }
                Some((_, Entries(inner))) => {
                    if let DeValue::Table(entries) = value.get_ref() {
                        for (entry, item) in entries.iter() {
                            if let DeValue::Table(item_table) = item.get_ref() {
                                let at = format!("{path}.{}", entry.get_ref());
                                self.check_table(item_table, inner, &at)?;
                            }
                        }
                    }
                }
                Some((_, Value | Map)) => {}
                None => {
                    let allowed: Vec<String> =
                        schema.iter().map(|(k, _)| (*k).to_string()).collect();
                    let (line, column) = self.position(key.span().start);
                    return Err(ManifestError::UnknownKey(Box::new(UnknownKeyError {
                        origin: self.source.clone(),
                        line,
                        column,
                        suggestion: crate::suggest::did_you_mean(
                            name,
                            allowed.iter().map(String::as_str),
                        ),
                        key: path,
                        table: match prefix.split_once('[') {
                            _ if prefix.is_empty() => "the top level".to_string(),
                            Some((array, _)) => format!("each [[{array}]]"),
                            None => format!("[{prefix}]"),
                        },
                        allowed,
                    })));
                }
            }
        }
        Ok(())
    }

    /// 1-based line and column (in characters) of a byte offset.
    pub(crate) fn position(&self, offset: usize) -> (usize, usize) {
        let offset = offset.min(self.text.len());
        let line = match self.line_starts.binary_search(&offset) {
            Ok(i) => i,
            Err(i) => i - 1,
        };
        let start = self.line_starts[line];
        let column = self
            .text
            .get(start..offset)
            .map_or(1, |s| s.chars().count() + 1);
        (line + 1, column)
    }

    /// Line and column of a key path such as `install.bin`, `aliases[1]` or
    /// `launch.env.FOO`. A path that is not in the file points at its nearest
    /// parent that is; `(0, 0)` when none is.
    pub(crate) fn key_position(&self, path: &str) -> (usize, usize) {
        self.span_of(path)
            .map(|span| self.position(span.start))
            .unwrap_or((0, 0))
    }

    fn span_of(&self, path: &str) -> Option<Range<usize>> {
        let mut table = self.root.get_ref();
        let mut best = None;
        let mut segments = path.split('.').peekable();
        while let Some(segment) = segments.next() {
            let (key, index) = match segment.split_once('[') {
                Some((key, rest)) => (key, rest.trim_end_matches(']').parse::<usize>().ok()),
                None => (segment, None),
            };
            let Some((found_key, value)) = table.iter().find(|(k, _)| k.get_ref() == key) else {
                return best;
            };
            best = Some(found_key.span());
            if let Some(index) = index {
                if let DeValue::Array(items) = value.get_ref()
                    && let Some(item) = items.get(index)
                {
                    best = Some(item.span());
                }
                return best;
            }
            match value.get_ref() {
                DeValue::Table(inner) if segments.peek().is_some() => table = inner,
                _ => return best,
            }
        }
        best
    }
}
