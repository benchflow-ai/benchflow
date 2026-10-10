//! "Did you mean" suggestions for agent names and manifest keys.

/// The candidate closest to `word`, when one is close enough to be a likely
/// typo (Jaro-Winkler similarity of at least 0.8, compared case-insensitively).
/// Ties go to the candidate listed first.
pub fn did_you_mean<'a, I>(word: &str, candidates: I) -> Option<String>
where
    I: IntoIterator<Item = &'a str>,
{
    let word = word.to_ascii_lowercase();
    let mut best: Option<(f64, &str)> = None;
    for candidate in candidates {
        let score = strsim::jaro_winkler(&word, &candidate.to_ascii_lowercase());
        if score >= 0.8 && best.is_none_or(|(top, _)| score > top) {
            best = Some((score, candidate));
        }
    }
    best.map(|(_, candidate)| candidate.to_string())
}

#[cfg(test)]
mod tests {
    use super::did_you_mean;

    #[test]
    fn suggests_close_names_only() {
        let names = ["claude", "codex", "gemini", "opencode"];
        assert_eq!(did_you_mean("cluade", names), Some("claude".into()));
        assert_eq!(did_you_mean("codx", names), Some("codex".into()));
        assert_eq!(did_you_mean("Gemni", names), Some("gemini".into()));
        assert_eq!(did_you_mean("open-code", names), Some("opencode".into()));
        assert_eq!(did_you_mean("zzz", names), None);
    }

    #[test]
    fn suggests_manifest_keys() {
        let keys = [
            "npm",
            "bin",
            "companions",
            "launcher_env",
            "verify",
            "timeout",
        ];
        assert_eq!(did_you_mean("timout", keys), Some("timeout".into()));
        assert_eq!(did_you_mean("companion", keys), Some("companions".into()));
        assert_eq!(
            did_you_mean("launcher-env", keys),
            Some("launcher_env".into())
        );
    }
}
