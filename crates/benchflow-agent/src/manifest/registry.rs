//! The agents a run can use: the built-ins plus the project's manifests.

use std::fmt;
use std::path::{Path, PathBuf};

use super::{AgentManifest, ManifestError, ManifestSource};
use crate::builtin;

/// Every agent this run knows: the built-in manifests, then the project's.
///
/// A project manifest named like an agent already registered replaces it
/// (to pin another version, say); its aliases then move with it. Any other
/// clash of names or aliases between manifests is an error.
#[derive(Debug, Clone)]
pub struct Registry {
    agents: Vec<AgentManifest>,
}

/// `--agent <name>` matched no agent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownAgent {
    pub name: String,
    /// The closest agent name or alias, when one is likely meant.
    pub suggestion: Option<String>,
    /// Every agent name, for the error message.
    pub available: Vec<String>,
}

impl fmt::Display for UnknownAgent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unknown agent `{}`", self.name)?;
        if let Some(suggestion) = &self.suggestion {
            write!(f, "; did you mean `{suggestion}`?")?;
        }
        write!(
            f,
            " (agents: {}, and the scripted {} and {}; `bench agents` describes them)",
            self.available.join(", "),
            builtin::ORACLE,
            builtin::NOP,
        )
    }
}

impl std::error::Error for UnknownAgent {}

impl Default for Registry {
    fn default() -> Self {
        Self::builtin()
    }
}

impl Registry {
    /// The built-in agents.
    pub fn builtin() -> Self {
        Self {
            agents: AgentManifest::builtins(),
        }
    }

    /// Exactly these agents, with no built-ins (tests, tools).
    pub fn from_manifests(agents: Vec<AgentManifest>) -> Self {
        Self { agents }
    }

    /// The built-in agents plus project manifests. Each path is a manifest
    /// file, or a folder whose `*.toml` files are all manifests (read in
    /// name order). Later manifests replace earlier ones of the same name.
    pub fn with_manifests<I, P>(paths: I) -> Result<Self, ManifestError>
    where
        I: IntoIterator<Item = P>,
        P: AsRef<Path>,
    {
        let mut registry = Self::builtin();
        for path in paths {
            for file in manifest_files(path.as_ref())? {
                registry.add(AgentManifest::load(&file)?)?;
            }
        }
        Ok(registry)
    }

    /// Add a manifest, replacing any agent of the same name.
    pub fn add(&mut self, manifest: AgentManifest) -> Result<(), ManifestError> {
        let replaced = self.agents.iter().position(|a| a.name == manifest.name);
        for name in std::iter::once(&manifest.name).chain(&manifest.aliases) {
            let clash = self
                .agents
                .iter()
                .enumerate()
                .find(|(i, other)| Some(*i) != replaced && other.answers_to(name));
            if let Some((_, other)) = clash {
                return Err(ManifestError::Duplicate {
                    name: name.clone(),
                    first: other.source.clone(),
                    second: manifest.source.clone(),
                });
            }
        }
        match replaced {
            Some(i) => self.agents[i] = manifest,
            None => self.agents.push(manifest),
        }
        Ok(())
    }

    /// The agent `name` refers to, by name or alias.
    pub fn find(&self, name: &str) -> Result<&AgentManifest, UnknownAgent> {
        if let Some(agent) = self.agents.iter().find(|a| a.name == name) {
            return Ok(agent);
        }
        if let Some(agent) = self.agents.iter().find(|a| a.answers_to(name)) {
            return Ok(agent);
        }
        let candidates = self.agents.iter().flat_map(|a| {
            std::iter::once(a.name.as_str()).chain(a.aliases.iter().map(String::as_str))
        });
        Err(UnknownAgent {
            name: name.to_string(),
            suggestion: crate::suggest::did_you_mean(name, candidates),
            available: self.agents.iter().map(|a| a.name.clone()).collect(),
        })
    }

    /// Every agent, built-ins first, in registration order.
    pub fn agents(&self) -> &[AgentManifest] {
        &self.agents
    }
}

/// The manifest files a path names: itself, or a folder's `*.toml` files.
fn manifest_files(path: &Path) -> Result<Vec<PathBuf>, ManifestError> {
    let read_error = |error| ManifestError::Read {
        path: path.to_path_buf(),
        error,
    };
    if !path.is_dir() {
        return Ok(vec![path.to_path_buf()]);
    }
    let mut files = Vec::new();
    for entry in std::fs::read_dir(path).map_err(read_error)? {
        let file = entry.map_err(read_error)?.path();
        if file.extension().is_some_and(|ext| ext == "toml") && file.is_file() {
            files.push(file);
        }
    }
    files.sort();
    Ok(files)
}

impl ManifestSource {
    /// The file this manifest was read from, when it was not built in.
    pub fn path(&self) -> Option<&Path> {
        match self {
            Self::Builtin(_) | Self::BuiltinProvider(_) | Self::AcpRegistry(_) => None,
            Self::File(path) => Some(path),
        }
    }
}
