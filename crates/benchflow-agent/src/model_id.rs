//! Model ids: which provider a model belongs to, and how an agent spells it.

use crate::manifest::{ModelFormat, ModelSpec};

/// The models.dev provider of a model id: the part before the first `/`
/// (`openrouter/x` is `openrouter`), or for a bare id, the provider its name
/// implies (`gpt-5` is `openai`). `None` when neither says.
pub(crate) fn provider(model: &str) -> Option<String> {
    if let Some((provider, rest)) = model.split_once('/') {
        return (!provider.is_empty() && !rest.is_empty()).then(|| provider.to_ascii_lowercase());
    }
    infer_provider(model).map(str::to_string)
}

/// 0.8's `_MODELSDEV_PROVIDER_HEURISTICS`: the first substring of a bare
/// model id that names a provider family.
pub(crate) fn infer_provider(bare: &str) -> Option<&'static str> {
    const HEURISTICS: &[(&str, &str)] = &[
        ("gemini", "google"),
        ("gemma", "google"),
        ("gpt", "openai"),
        ("o1", "openai"),
        ("o3", "openai"),
        ("o4", "openai"),
        ("claude", "anthropic"),
        ("haiku", "anthropic"),
        ("sonnet", "anthropic"),
        ("opus", "anthropic"),
        ("mistral", "mistral"),
        ("codestral", "mistral"),
    ];
    let lower = bare.to_ascii_lowercase();
    HEURISTICS
        .iter()
        .find(|(needle, _)| lower.contains(needle))
        .map(|(_, provider)| *provider)
}

/// `model` as the agent expects it over ACP: the manifest's prefixes
/// stripped, then written in its format. Codex's `model[effort]` variant is
/// chosen later, from what the session offers.
pub fn format_model(spec: &ModelSpec, model: &str) -> String {
    let model = spec
        .strip_prefixes
        .iter()
        .find_map(|prefix| model.strip_prefix(prefix.as_str()))
        .unwrap_or(model);
    match spec.format {
        ModelFormat::Bare | ModelFormat::ModelEffort => model.to_string(),
        ModelFormat::ProviderModel if model.contains('/') => model.to_string(),
        ModelFormat::ProviderModel => match infer_provider(model) {
            Some(provider) => format!("{provider}/{model}"),
            None => {
                // 0.8's fallback, with its warning.
                tracing::warn!(
                    "cannot infer the models.dev provider of {model:?}; using anthropic/"
                );
                format!("anthropic/{model}")
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Registry;

    fn spec(agent: &str) -> ModelSpec {
        Registry::builtin().find(agent).unwrap().model.clone()
    }

    #[test]
    fn providers_come_from_the_prefix_or_the_name() {
        assert_eq!(
            provider("openrouter/qwen/qwen3-coder").as_deref(),
            Some("openrouter")
        );
        assert_eq!(provider("Google/gemini-3-pro").as_deref(), Some("google"));
        assert_eq!(provider("gpt-5").as_deref(), Some("openai"));
        assert_eq!(provider("claude-sonnet-4-6").as_deref(), Some("anthropic"));
        assert_eq!(provider("gemini-2.5-flash").as_deref(), Some("google"));
        assert_eq!(provider("codestral-latest").as_deref(), Some("mistral"));
        assert_eq!(provider("qwen3-coder"), None);
        assert_eq!(provider("/x"), None);
    }

    #[test]
    fn agents_get_their_spelling() {
        // OpenCode wants models.dev ids (0.8's _format_acp_model).
        assert_eq!(
            format_model(&spec("opencode"), "gemini-3.1-pro-preview"),
            "google/gemini-3.1-pro-preview"
        );
        assert_eq!(format_model(&spec("opencode"), "gpt-5"), "openai/gpt-5");
        assert_eq!(
            format_model(&spec("opencode"), "openrouter/x/y"),
            "openrouter/x/y"
        );
        assert_eq!(
            format_model(&spec("opencode"), "mystery-model"),
            "anthropic/mystery-model"
        );
        // Gemini CLI takes bare Google ids.
        assert_eq!(
            format_model(&spec("gemini"), "google/gemini-2.5-pro"),
            "gemini-2.5-pro"
        );
        assert_eq!(
            format_model(&spec("gemini"), "gemini-2.5-pro"),
            "gemini-2.5-pro"
        );
        // Codex drops 0.8's `openai/` provider prefix; Claude takes ids as given.
        assert_eq!(
            format_model(&spec("codex"), "openai/gpt-6.1-sol"),
            "gpt-6.1-sol"
        );
        assert_eq!(
            format_model(&spec("claude"), "claude-opus-5-5"),
            "claude-opus-5-5"
        );
    }
}
