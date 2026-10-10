//! `.dockerignore`: which files of a build context Daytona receives.
//!
//! Docker's rules: one pattern per line, `#` comments, patterns relative to
//! the context root (a leading `/` is dropped), `*` and `?` within one path
//! component, `**` across components, `!` re-includes. A path is left out
//! when the last pattern matching it, or one of its parent folders, is not a
//! `!` exception.

use std::path::Path;

use globset::{GlobBuilder, GlobMatcher};

use crate::archive::PackFilter;

/// The parsed patterns of a `.dockerignore`.
#[derive(Debug, Default)]
pub(crate) struct DockerIgnore {
    rules: Vec<(GlobMatcher, bool)>,
    has_exceptions: bool,
}

impl DockerIgnore {
    /// Read `<context>/.dockerignore`, or `<Dockerfile>.dockerignore` next to
    /// the Dockerfile when that exists (BuildKit prefers it). No file: keep
    /// everything.
    pub(crate) fn load(context: &Path, dockerfile: &Path) -> Self {
        let specific = dockerfile.with_file_name(format!(
            "{}.dockerignore",
            dockerfile
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default()
        ));
        let text = std::fs::read_to_string(&specific)
            .or_else(|_| std::fs::read_to_string(context.join(".dockerignore")))
            .unwrap_or_default();
        Self::parse(&text)
    }

    pub(crate) fn parse(text: &str) -> Self {
        let mut ignore = Self::default();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (exception, pattern) = match line.strip_prefix('!') {
                Some(rest) => (true, rest.trim()),
                None => (false, line),
            };
            let pattern = clean(pattern);
            if pattern.is_empty() {
                continue;
            }
            match GlobBuilder::new(&pattern).literal_separator(true).build() {
                Ok(glob) => {
                    ignore.rules.push((glob.compile_matcher(), exception));
                    ignore.has_exceptions |= exception;
                }
                Err(err) => tracing::warn!(".dockerignore: skipping pattern {line:?}: {err}"),
            }
        }
        ignore
    }

    /// Whether `rel` (relative, `/`-separated) is left out of the context.
    pub(crate) fn excludes(&self, rel: &str) -> bool {
        let mut excluded = false;
        for (matcher, exception) in &self.rules {
            if matches_or_parent(matcher, rel) {
                excluded = !exception;
            }
        }
        excluded
    }
}

impl PackFilter for DockerIgnore {
    fn keep(&self, rel: &str, _is_dir: bool) -> bool {
        !self.excludes(rel)
    }

    fn descend(&self, rel_dir: &str) -> bool {
        // With an exception anywhere, something below an excluded folder may
        // come back, so walk it.
        self.has_exceptions || !self.excludes(rel_dir)
    }
}

fn matches_or_parent(matcher: &GlobMatcher, rel: &str) -> bool {
    if matcher.is_match(rel) {
        return true;
    }
    let mut end = rel.len();
    while let Some(slash) = rel[..end].rfind('/') {
        if matcher.is_match(&rel[..slash]) {
            return true;
        }
        end = slash;
    }
    false
}

/// Clean a pattern like Go's `filepath.Clean`, relative to the root.
fn clean(pattern: &str) -> String {
    let mut parts: Vec<&str> = Vec::new();
    for part in pattern.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            p => parts.push(p),
        }
    }
    parts.join("/")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn follows_dockers_rules() {
        let ignore = DockerIgnore::parse(
            "# comment\n\nnode_modules\n*.log\n/build/\n**/secret.txt\ndata/*\n!data/keep.csv\n",
        );
        assert!(ignore.excludes("node_modules"));
        assert!(
            ignore.excludes("node_modules/x/y.js"),
            "a folder excludes its contents"
        );
        assert!(ignore.excludes("app.log"));
        assert!(!ignore.excludes("logs/app.log"), "* does not cross folders");
        assert!(ignore.excludes("build/out.o"));
        assert!(ignore.excludes("a/b/secret.txt"));
        assert!(ignore.excludes("secret.txt"));
        assert!(ignore.excludes("data/big.bin"));
        assert!(!ignore.excludes("data/keep.csv"), "! brings a path back");
        assert!(!ignore.excludes("src/main.rs"));
        assert!(ignore.descend("data"), "exceptions force a walk");
    }

    #[test]
    fn without_exceptions_excluded_folders_are_not_walked() {
        let ignore = DockerIgnore::parse("node_modules\n");
        assert!(!ignore.descend("node_modules"));
        assert!(ignore.descend("src"));
    }

    #[test]
    fn an_empty_file_keeps_everything() {
        let ignore = DockerIgnore::parse("");
        assert!(!ignore.excludes("anything/at/all"));
    }

    #[test]
    fn loads_the_dockerfile_specific_file_first() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(".dockerignore"), "a\n").unwrap();
        std::fs::write(dir.path().join("Dockerfile.dockerignore"), "b\n").unwrap();
        let ignore = DockerIgnore::load(dir.path(), &dir.path().join("Dockerfile"));
        assert!(ignore.excludes("b") && !ignore.excludes("a"));
        std::fs::remove_file(dir.path().join("Dockerfile.dockerignore")).unwrap();
        let ignore = DockerIgnore::load(dir.path(), &dir.path().join("Dockerfile"));
        assert!(ignore.excludes("a"));
    }
}
