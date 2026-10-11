use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use serde_json::Value;

use super::*;
use crate::Registry;
use crate::credentials::{CredentialError, HostEnv};

mod forwarder_js;
mod orchestration;

fn agent(name: &str) -> crate::AgentManifest {
    Registry::builtin().find(name).unwrap().clone()
}

fn providers() -> ProviderRegistry {
    ProviderRegistry::builtin()
}

fn parse(text: &str) -> Result<Provider, ManifestError> {
    Provider::parse(
        text,
        ManifestSource::File(PathBuf::from("providers/test.toml")),
    )
}

fn provider_route(agent_name: &str, model: &str) -> ProviderRoute {
    match route(&agent(agent_name), Some(model), &providers()).unwrap() {
        Route::Provider(route) => *route,
        Route::Agent => panic!("{agent_name} on {model} took the agent's own route"),
    }
}

const FAKE_KEY: &str = "vck_fake-gateway-key-0123456789abcdef";

fn host(vars: &[(&str, &str)], home: Option<PathBuf>) -> HostEnv {
    let vars: BTreeMap<String, String> = vars
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    HostEnv::new(vars, home)
}

#[test]
fn builtins_are_the_seven_providers_with_their_endpoints() {
    let names: Vec<String> = Provider::builtins().into_iter().map(|p| p.name).collect();
    assert_eq!(
        names,
        [
            "vercel",
            "openai",
            "anthropic",
            "openrouter",
            "zai",
            "deepseek",
            "mimo"
        ]
    );
    let registry = providers();
    let vercel = registry.find("vercel").unwrap();
    let openai = vercel.openai.as_ref().unwrap();
    assert_eq!(openai.base_url, "https://ai-gateway.vercel.sh/v1");
    assert_eq!(openai.auth, Auth::Bearer);
    assert_eq!(openai.apis, [OpenAiApi::Chat, OpenAiApi::Responses]);
    let anthropic = vercel.anthropic.as_ref().unwrap();
    assert_eq!(
        anthropic.base_url,
        "https://ai-gateway.vercel.sh/claude-code"
    );
    assert_eq!(anthropic.auth, Auth::Bearer);
    // fx's own protocol, from the gateway's root.
    let gateway = vercel.gateway.as_ref().unwrap();
    assert_eq!(gateway.base_url, "https://ai-gateway.vercel.sh");
    assert_eq!(gateway.auth, Auth::Bearer);
    assert_eq!(
        vercel.key.env,
        [
            "AI_GATEWAY_API_KEY",
            "VERCEL_AI_GATEWAY_API_KEY",
            "VERCEL_OIDC_TOKEN"
        ]
    );
    assert_eq!(
        vercel.key.files,
        ["~/.config/benchflow/vercel-ai-gateway.env"]
    );
    assert_eq!(vercel.check.as_ref().unwrap().path, "/credits");
    assert_eq!(
        vercel.serves(),
        "OpenAI-compatible (Chat Completions, Responses API), Anthropic-compatible and the gateway's own API"
    );
    // OpenAI's key under Codex's own name too.
    assert_eq!(
        registry.find("openai").unwrap().key.env,
        ["OPENAI_API_KEY", "CODEX_API_KEY"]
    );
    // OpenRouter names the denied upstreams in each request.
    assert_eq!(
        registry.find("openrouter").unwrap().upstream_routing(),
        Some(Routing::openrouter())
    );
    // Anthropic's own API takes x-api-key; Z.ai has no free check.
    let anthropic = registry.find("anthropic").unwrap();
    assert_eq!(anthropic.anthropic.as_ref().unwrap().auth, Auth::XApiKey);
    assert!(anthropic.openai.is_none());
    assert!(registry.find("zai").unwrap().check.is_none());
    assert_eq!(
        registry
            .find("deepseek")
            .unwrap()
            .openai
            .as_ref()
            .unwrap()
            .apis,
        [OpenAiApi::Chat]
    );
    // MiMo's token plan serves both OpenAI APIs and Anthropic's, on one host.
    let mimo = registry.find("mimo").unwrap();
    assert_eq!(
        mimo.openai.as_ref().unwrap().base_url,
        "https://token-plan-sgp.xiaomimimo.com/v1"
    );
    assert_eq!(
        mimo.openai.as_ref().unwrap().apis,
        [OpenAiApi::Chat, OpenAiApi::Responses]
    );
    assert_eq!(mimo.anthropic.as_ref().unwrap().auth, Auth::XApiKey);
    assert_eq!(mimo.key.env, ["MIMO_API_KEY"]);
    assert_eq!(
        ManifestSource::BuiltinProvider("vercel.toml").to_string(),
        "built-in providers/vercel.toml"
    );
}

#[test]
fn provider_files_are_checked_like_manifests() {
    let base =
        "name = \"p\"\n[openai]\nbase_url = \"https://x.example/v1\"\n[key]\nenv = [\"P_KEY\"]\n";
    assert!(parse(base).is_ok());
    let cases: &[(&str, &str)] = &[
        (
            "name = \"p\"\n[openai]\nbase_ur = \"https://x.example/v1\"\n[key]\nenv = [\"P_KEY\"]\n",
            "providers/test.toml:3:1: unknown key `openai.base_ur`; did you mean `base_url`?",
        ),
        (
            "name = \"p\"\n[openai]\nbase_url = \"http://x.example/v1\"\n[key]\nenv = [\"P_KEY\"]\n",
            "must be an `https://` URL",
        ),
        (
            "name = \"p\"\n[openai]\nbase_url = \"https://x.example/v1?a=b\"\n[key]\nenv = [\"P_KEY\"]\n",
            "without a query",
        ),
        (
            "name = \"p\"\n[openai]\nbase_url = \"https://x.example/v1\"\nauth = \"basic\"\n[key]\nenv = [\"P_KEY\"]\n",
            "`basic` is not `bearer` or `x-api-key`",
        ),
        (
            "name = \"p\"\n[openai]\nbase_url = \"https://x.example/v1\"\napis = [\"completions\"]\n[key]\nenv = [\"P_KEY\"]\n",
            "`completions` is not `chat` or `responses`",
        ),
        (
            "name = \"p\"\n[key]\nenv = [\"P_KEY\"]\n",
            "needs an `[openai]`, `[anthropic]` or `[gateway]` endpoint",
        ),
        (
            "name = \"p\"\n[openai]\nbase_url = \"https://x.example/v1\"\n",
            "say which variables hold the key",
        ),
        (
            "name = \"P\"\n[openai]\nbase_url = \"https://x.example/v1\"\n[key]\nenv = [\"P_KEY\"]\n",
            "not a valid provider name",
        ),
        (
            "name = \"p\"\n[openai]\nbase_url = \"https://x.example/v1\"\n[key]\nenv = [\"P_KEY\"]\n[check]\nendpoint = \"anthropic\"\npath = \"/models\"\n",
            "has no `[anthropic]` endpoint to check",
        ),
        (
            "name = \"p\"\n[openai]\nbase_url = \"https://x.example/v1\"\n[key]\nenv = [\"P_KEY\"]\n[check]\nendpoint = \"openai\"\npath = \"/models\"\nheaders = { Authorization = \"x\" }\n",
            "the key's header is added by BenchFlow",
        ),
    ];
    for (text, expected) in cases {
        let error = parse(text).unwrap_err().to_string();
        assert!(error.contains(expected), "{error}\n  expected: {expected}");
    }
    // A trailing slash is dropped.
    let p = parse(&base.replace("/v1\"", "/v1/\"")).unwrap();
    assert_eq!(p.openai.unwrap().base_url, "https://x.example/v1");
}

#[test]
fn a_project_provider_replaces_a_builtin_of_the_same_name() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("vercel.toml"),
        "name = \"vercel\"\n[openai]\nbase_url = \"https://gateway.example/v1\"\n[key]\nenv = [\"MY_GATEWAY_KEY\"]\n",
    )
    .unwrap();
    std::fs::write(
        dir.path().join("local.toml"),
        "name = \"local\"\n[openai]\nbase_url = \"https://llm.example/v1\"\n[key]\nenv = [\"LOCAL_KEY\"]\n",
    )
    .unwrap();
    let registry = ProviderRegistry::with_files([dir.path()]).unwrap();
    assert_eq!(registry.providers().len(), 8);
    let vercel = registry.find("vercel").unwrap();
    assert_eq!(vercel.key.env, ["MY_GATEWAY_KEY"]);
    assert!(vercel.anthropic.is_none());
    assert!(registry.find("local").is_some());
}

#[test]
fn agents_declare_the_endpoints_they_can_use() {
    let claude = agent("claude");
    assert_eq!(claude.providers.len(), 1);
    let spec = &claude.providers[0];
    assert_eq!(spec.protocol, Protocol::Anthropic);
    assert_eq!(spec.base_url.as_deref(), Some("ANTHROPIC_BASE_URL"));
    assert_eq!(spec.key.as_deref(), Some("ANTHROPIC_AUTH_TOKEN"));
    assert_eq!(spec.clear, ["ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"]);
    assert_eq!(spec.describe(), "Anthropic-compatible");

    let codex = agent("codex");
    let spec = &codex.providers[0];
    assert_eq!(spec.protocol, Protocol::OpenAi);
    assert_eq!(spec.api, Some(OpenAiApi::Responses));
    assert_eq!(spec.effort.as_deref(), Some("medium"));
    assert_eq!(spec.describe(), "OpenAI-compatible (Responses API)");

    let opencode = agent("opencode");
    let protocols: Vec<Protocol> = opencode.providers.iter().map(|s| s.protocol).collect();
    assert_eq!(protocols, [Protocol::OpenAi, Protocol::Anthropic]);
    assert!(agent("gemini").providers.is_empty());
}

#[test]
fn provider_tables_in_manifests_are_checked() {
    let manifest = |providers: &str| {
        crate::AgentManifest::parse(
            &format!("name = \"x\"\n[install]\nnpm = \"x@1.0.0\"\nbin = \"x\"\n{providers}"),
            ManifestSource::File(PathBuf::from("agents/x.toml")),
        )
    };
    assert!(manifest("[providers.openai]\nbase_url = \"OPENAI_BASE_URL\"\n").is_ok());
    let cases: &[(&str, &str)] = &[
        (
            "[providers.gemini]\nbase_url = \"X\"\n",
            "unknown key `providers.gemini`",
        ),
        (
            "[providers.openai]\nbase_urls = \"X\"\n",
            "did you mean `base_url`?",
        ),
        (
            "[providers.anthropic]\napi = \"chat\"\nbase_url = \"X\"\n",
            "only `[providers.openai]` takes `api`",
        ),
        (
            "[providers.openai]\napi = \"completions\"\nbase_url = \"X\"\n",
            "`completions` is not `chat` or `responses`",
        ),
        (
            "[providers.openai]\nkey = \"K\"\n",
            "say where the endpoint goes",
        ),
        (
            "[providers.openai]\nbase_url = \"X\"\nenv = { M = \"{modle}\" }\n",
            "`{modle}` is not a placeholder",
        ),
        (
            "[providers.openai]\nbase_url = \"X\"\nenv = { M = \"{effort}\" }\n",
            "a template uses `{effort}`",
        ),
        (
            "[providers.openai]\nbase_url = \"X\"\nclear = [\"X\"]\n",
            "cannot be both set and cleared",
        ),
        (
            "[providers.openai]\nbase_url = \"X\"\neffort = \"High!\"\n",
            "is not an effort level",
        ),
    ];
    for (providers, expected) in cases {
        let error = manifest(providers).unwrap_err().to_string();
        assert!(error.contains(expected), "{error}\n  expected: {expected}");
    }
    // A base URL in a template is enough, and JSON braces are not placeholders.
    let ok = manifest(
        "[providers.openai]\nenv = { CONFIG = '{\"url\": \"{base_url}\", \"m\": {\"{model}\": {}}}' }\n",
    )
    .unwrap();
    assert_eq!(ok.providers[0].model, "{model}");
}

#[test]
fn a_registered_provider_prefix_routes_through_its_endpoint() {
    // Claude Code speaks the Anthropic API.
    let claude = provider_route("claude", "vercel/deepseek/deepseek-v4-flash-0731");
    assert_eq!(claude.provider.name, "vercel");
    assert_eq!(
        claude.model.as_deref(),
        Some("deepseek/deepseek-v4-flash-0731")
    );
    assert!(!claude.own_key);
    assert_eq!(claude.endpoint.protocol, Protocol::Anthropic);
    assert_eq!(
        claude.endpoint.base_url,
        "https://ai-gateway.vercel.sh/claude-code"
    );
    assert_eq!(claude.prefix(), "/anthropic");
    // Codex needs the Responses API, which Vercel serves.
    let codex = provider_route("codex", "vercel/alibaba/qwen3.7-flash");
    assert_eq!(codex.endpoint.protocol, Protocol::OpenAi);
    assert_eq!(codex.model.as_deref(), Some("alibaba/qwen3.7-flash"));
    // OpenCode prefers Chat Completions.
    let opencode = provider_route("opencode", "vercel/deepseek/deepseek-v4-flash-0731");
    assert_eq!(opencode.endpoint.protocol, Protocol::OpenAi);
    // DeepSeek's own Anthropic endpoint for Claude Code.
    let deepseek = provider_route("claude", "deepseek/deepseek-chat");
    assert_eq!(
        deepseek.endpoint.base_url,
        "https://api.deepseek.com/anthropic"
    );
    // OpenCode on Anthropic's API goes through the forwarder too.
    let anthropic = provider_route("opencode", "anthropic/claude-sonnet-4-6");
    assert_eq!(anthropic.endpoint.protocol, Protocol::Anthropic);
}

#[test]
fn the_agents_own_prefixes_and_bare_ids_keep_its_own_logins() {
    for (name, model) in [
        ("codex", "openai/gpt-5"),
        ("claude", "anthropic/claude-haiku-4-5"),
        ("gemini", "google/gemini-2.5-pro"),
        ("claude", "claude-haiku-4-5-20251001"),
        ("codex", "gpt-6.1-sol"),
        ("opencode", "google/gemini-3-pro"),
        ("opencode", "mystery/model"),
    ] {
        assert_eq!(
            route(&agent(name), Some(model), &providers()).unwrap(),
            Route::Agent,
            "{name} {model}"
        );
    }
    assert_eq!(
        route(&agent("claude"), None, &providers()).unwrap(),
        Route::Agent
    );
}

#[test]
fn an_agent_that_cannot_use_the_provider_is_refused_with_its_options() {
    let error = route(
        &agent("gemini"),
        Some("vercel/deepseek/deepseek-v4-flash-0731"),
        &providers(),
    )
    .unwrap_err();
    assert_eq!(
        error.to_string(),
        "agent `gemini` cannot use provider `vercel`: its manifest has no [providers] table, so it runs only on its own logins"
    );
    assert!(
        error
            .next_step()
            .contains("pick an agent that can take a base URL")
    );

    // Codex needs the Responses API; Z.ai serves Chat Completions and the
    // Anthropic API.
    let error = route(&agent("codex"), Some("zai/glm-5"), &providers()).unwrap_err();
    assert_eq!(
        error.to_string(),
        "agent `codex` cannot use provider `zai`: the agent speaks OpenAI-compatible (Responses API), and zai serves OpenAI-compatible (Chat Completions) and Anthropic-compatible"
    );
    assert_eq!(
        error.next_step(),
        "pick another agent, or a provider `codex` can use: vercel, openai, openrouter, mimo"
    );

    let error = route(&agent("claude"), Some("vercel/"), &providers()).unwrap_err();
    assert!(matches!(error, RouteError::BadModel { .. }), "{error:?}");
    let error = route(&agent("claude"), Some("vercel/x\"y"), &providers()).unwrap_err();
    assert!(
        error.to_string().contains("is not allowed in a model id"),
        "{error}"
    );
}

#[test]
fn claude_is_pointed_at_the_forwarder_with_a_placeholder_key() {
    let route = provider_route("claude", "vercel/deepseek/deepseek-v4-flash-0731");
    let env = route
        .agent_env("http://127.0.0.1:41234/anthropic", None)
        .unwrap();
    let model = "deepseek/deepseek-v4-flash-0731";
    let expected: BTreeMap<String, String> = [
        ("ANTHROPIC_BASE_URL", "http://127.0.0.1:41234/anthropic"),
        ("ANTHROPIC_AUTH_TOKEN", KEY_PLACEHOLDER),
        ("ANTHROPIC_API_KEY", ""),
        ("CLAUDE_CODE_OAUTH_TOKEN", ""),
        ("ANTHROPIC_MODEL", model),
        ("ANTHROPIC_DEFAULT_OPUS_MODEL", model),
        ("ANTHROPIC_DEFAULT_SONNET_MODEL", model),
        ("ANTHROPIC_DEFAULT_HAIKU_MODEL", model),
        ("ANTHROPIC_SMALL_FAST_MODEL", model),
        ("CLAUDE_CODE_SUBAGENT_MODEL", model),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect();
    assert_eq!(env, expected);
    // Claude Code lists the aliases, not the id: the variables carry it.
    assert_eq!(route.acp_model(None).unwrap(), None);
}

#[test]
fn codex_gets_the_model_and_effort_in_its_config_file() {
    let route = provider_route("codex", "vercel/alibaba/qwen3.7-flash");
    let env = route
        .agent_env("http://127.0.0.1:41234/openai", None)
        .unwrap();
    // codex-acp ignores OPENAI_BASE_URL: the endpoint is in config.toml.
    assert_eq!(env["OPENAI_BASE_URL"], "");
    assert_eq!(env["OPENAI_API_KEY"], KEY_PLACEHOLDER);
    assert_eq!(env["CODEX_API_KEY"], "");
    assert_eq!(
        env["BENCHFLOW_CODEX_HOME_CONFIG"],
        "# benchflow-codex-home-config\n\
         model = \"alibaba/qwen3.7-flash\"\n\
         model_reasoning_effort = \"medium\"\n\
         model_provider = \"benchflow\"\n\
         \n\
         [model_providers.benchflow]\n\
         name = \"BenchFlow forwarder\"\n\
         base_url = \"http://127.0.0.1:41234/openai\"\n\
         env_key = \"OPENAI_API_KEY\"\n\
         wire_api = \"responses\"\n\
         supports_websockets = false\n"
    );
    // It is valid TOML.
    let config: toml::Table = env["BENCHFLOW_CODEX_HOME_CONFIG"].parse().unwrap();
    assert_eq!(
        config["model_providers"]["benchflow"]["wire_api"].as_str(),
        Some("responses")
    );
    let high = route
        .agent_env("http://127.0.0.1:41234/openai", Some("high"))
        .unwrap();
    assert!(high["BENCHFLOW_CODEX_HOME_CONFIG"].contains("model_reasoning_effort = \"high\""));
    assert!(matches!(
        route.agent_env("http://x", Some("high\"\nmodel = \"x")),
        Err(RouteError::BadEffort { .. })
    ));
    assert_eq!(
        route.acp_model(None).unwrap().as_deref(),
        Some("alibaba/qwen3.7-flash")
    );
}

#[test]
fn opencode_gets_a_provider_of_its_own_in_valid_json() {
    let route = provider_route("opencode", "vercel/deepseek/deepseek-v4-flash-0731");
    let env = route
        .agent_env("http://127.0.0.1:41234/openai", None)
        .unwrap();
    let config: Value = serde_json::from_str(&env["OPENCODE_CONFIG_CONTENT"]).unwrap();
    let provider = &config["provider"]["benchflow"];
    assert_eq!(provider["npm"], "@ai-sdk/openai-compatible");
    assert_eq!(
        provider["options"]["baseURL"],
        "http://127.0.0.1:41234/openai"
    );
    assert_eq!(provider["options"]["apiKey"], KEY_PLACEHOLDER);
    assert!(provider["models"]["deepseek/deepseek-v4-flash-0731"].is_object());
    assert_eq!(config["model"], "benchflow/deepseek/deepseek-v4-flash-0731");
    assert_eq!(
        route.acp_model(None).unwrap().as_deref(),
        Some("benchflow/deepseek/deepseek-v4-flash-0731")
    );
    let route = provider_route("opencode", "anthropic/claude-sonnet-4-6");
    let env = route
        .agent_env("http://127.0.0.1:41234/anthropic", None)
        .unwrap();
    let config: Value = serde_json::from_str(&env["OPENCODE_CONFIG_CONTENT"]).unwrap();
    assert_eq!(
        config["provider"]["benchflow-anthropic"]["options"]["baseURL"],
        "http://127.0.0.1:41234/anthropic/v1"
    );
}

#[test]
fn the_forwarder_config_holds_no_secret() {
    let route = provider_route("claude", "vercel/deepseek/deepseek-v4-flash-0731");
    let config = route.forwarder_config("/opt/benchflow/forwarder/key");
    assert_eq!(
        config,
        serde_json::json!({
            "listen": "127.0.0.1",
            "port": 0,
            "key_file": "/opt/benchflow/forwarder/key",
            "provider": "vercel",
            "routes": [{
                "prefix": "/anthropic",
                "upstream": "https://ai-gateway.vercel.sh/claude-code",
                "auth": "bearer",
                "models": ["deepseek/deepseek-v4-flash-0731"],
            }],
        })
    );
}

#[test]
fn the_provider_key_is_the_login_and_the_agent_gets_none_of_it() {
    let manifest = agent("claude");
    let host = host(
        &[
            ("AI_GATEWAY_API_KEY", FAKE_KEY),
            // The agent's own login is not used on a provider.
            ("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-fake-token-000000"),
        ],
        None,
    );
    let access = resolve(
        &manifest,
        Some("vercel/deepseek/deepseek-v4-flash-0731"),
        &host,
        &providers(),
    )
    .unwrap();
    let Route::Provider(route) = &access.route else {
        panic!("{:?}", access.route)
    };
    assert_eq!(route.provider.name, "vercel");
    assert!(access.login.env.is_empty());
    assert!(access.login.files.is_empty());
    assert_eq!(access.login.secrets, [FAKE_KEY]);
    assert_eq!(
        access.login.label,
        "AI_GATEWAY_API_KEY (environment), held by BenchFlow's forwarder for vercel's Anthropic-compatible endpoint"
    );
    let key = access.key.unwrap();
    assert_eq!(key.value(), FAKE_KEY);
    assert!(!format!("{key:?}").contains(FAKE_KEY));
    let env = route
        .agent_env("http://127.0.0.1:1/anthropic", None)
        .unwrap();
    assert!(env.values().all(|v| !v.contains(FAKE_KEY)));

    // Without a model the agent runs on its own login.
    let own = resolve(&manifest, None, &host, &providers()).unwrap();
    assert_eq!(own.route, Route::Agent);
    assert!(own.login.label.starts_with("CLAUDE_CODE_OAUTH_TOKEN"));
}

#[test]
fn the_key_comes_from_the_providers_file_after_the_environment() {
    let home = tempfile::tempdir().unwrap();
    let dir = home.path().join(".config/benchflow");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("vercel-ai-gateway.env"),
        format!("# the gateway\nexport AI_GATEWAY_API_KEY='{FAKE_KEY}'\n"),
    )
    .unwrap();
    let vercel = providers().find("vercel").unwrap().clone();
    let key = find_provider_key(&vercel, &host(&[], Some(home.path().to_path_buf()))).unwrap();
    assert_eq!(key.value(), FAKE_KEY);
    assert_eq!(
        key.source,
        "AI_GATEWAY_API_KEY in ~/.config/benchflow/vercel-ai-gateway.env"
    );
    // The environment wins; a blank variable does not count.
    let env_key = find_provider_key(
        &vercel,
        &host(
            &[("AI_GATEWAY_API_KEY", "vck_from-the-environment-000")],
            Some(home.path().to_path_buf()),
        ),
    )
    .unwrap();
    assert_eq!(env_key.value(), "vck_from-the-environment-000");
    let blank = find_provider_key(
        &vercel,
        &host(
            &[("AI_GATEWAY_API_KEY", "  ")],
            Some(home.path().to_path_buf()),
        ),
    )
    .unwrap();
    assert_eq!(blank.value(), FAKE_KEY);
    let status = key_status(&vercel, &host(&[], Some(home.path().to_path_buf())));
    assert_eq!(
        status.found.as_deref(),
        Some("AI_GATEWAY_API_KEY in ~/.config/benchflow/vercel-ai-gateway.env")
    );
}

#[test]
fn a_missing_key_says_where_to_put_it() {
    let empty_home = tempfile::tempdir().unwrap();
    let host = host(&[], Some(empty_home.path().to_path_buf()));
    let error = resolve(
        &agent("codex"),
        Some("vercel/alibaba/qwen3.7-flash"),
        &host,
        &providers(),
    )
    .unwrap_err();
    let AccessError::Login(CredentialError::NoProviderKey {
        provider,
        checked,
        next_step,
    }) = &error
    else {
        panic!("{error:?}")
    };
    assert_eq!(provider, "vercel");
    assert_eq!(
        checked,
        &[
            "AI_GATEWAY_API_KEY".to_string(),
            "VERCEL_AI_GATEWAY_API_KEY".to_string(),
            "VERCEL_OIDC_TOKEN".to_string(),
            "~/.config/benchflow/vercel-ai-gateway.env".to_string()
        ]
    );
    assert_eq!(
        next_step,
        "`export AI_GATEWAY_API_KEY=<key>`, or put `AI_GATEWAY_API_KEY=<key>` in ~/.config/benchflow/vercel-ai-gateway.env"
    );
    assert_eq!(
        error.to_string(),
        "no key for provider `vercel`: looked for AI_GATEWAY_API_KEY, VERCEL_AI_GATEWAY_API_KEY, VERCEL_OIDC_TOKEN or ~/.config/benchflow/vercel-ai-gateway.env; next: `export AI_GATEWAY_API_KEY=<key>`, or put `AI_GATEWAY_API_KEY=<key>` in ~/.config/benchflow/vercel-ai-gateway.env"
    );
    let status = key_status(providers().find("vercel").unwrap(), &host);
    assert!(status.found.is_none());
    assert!(status.fix.is_some());

    // A route error comes before the key is looked for.
    let error = resolve(&agent("gemini"), Some("vercel/x/y"), &host, &providers()).unwrap_err();
    assert!(matches!(error, AccessError::Route(_)), "{error:?}");
}

/// A fake gateway listing: `/v1/models/{model}/endpoints` for three models,
/// 404 for any other. Counts the requests.
async fn fake_listing() -> (String, Arc<std::sync::atomic::AtomicUsize>) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}/v1", listener.local_addr().unwrap());
    let hits = Arc::new(AtomicUsize::new(0));
    let counter = hits.clone();
    tokio::spawn(async move {
        while let Ok((stream, _)) = listener.accept().await {
            counter.fetch_add(1, Ordering::SeqCst);
            let mut reader = BufReader::new(stream);
            let mut line = String::new();
            let _ = reader.read_line(&mut line).await;
            loop {
                let mut header = String::new();
                if reader.read_line(&mut header).await.unwrap_or(0) == 0 || header.trim().is_empty()
                {
                    break;
                }
            }
            let path = line.split_whitespace().nth(1).unwrap_or("").to_string();
            let listing = |names: &[&str]| {
                let endpoints: Vec<Value> = names
                    .iter()
                    .map(|n| serde_json::json!({"provider_name": n, "name": n}))
                    .collect();
                serde_json::json!({"data": {"id": "m", "endpoints": endpoints}}).to_string()
            };
            let (status, body) = match path.as_str() {
                "/v1/models/deepseek/deepseek-v4-flash-0731/endpoints" => {
                    ("200 OK", listing(&["alibaba", "baseten", "novita"]))
                }
                "/v1/models/m/only-baseten/endpoints" => ("200 OK", listing(&["baseten"])),
                "/v1/models/m/wafer-and-novita/endpoints" => {
                    ("200 OK", listing(&["wafer", "novita"]))
                }
                _ => ("404 Not Found", "{}".to_string()),
            };
            let answer = format!(
                "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = reader.get_mut().write_all(answer.as_bytes()).await;
        }
    });
    (base, hits)
}

#[test]
fn a_route_on_a_gateway_is_pinned_once_per_model() {
    use std::sync::atomic::Ordering;
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let (base, hits) = runtime.block_on(fake_listing());
    let mut registry = providers();
    let mut gateway = registry.find("vercel").unwrap().clone();
    gateway.routing = Some(Routing {
        style: Style::Only {
            endpoints: format!("{base}/models/{{model}}/endpoints"),
        },
        deny: Vec::new(),
    });
    registry.add(gateway);
    let route_for = |model: &str| match route(&agent("claude"), Some(model), &registry).unwrap() {
        Route::Provider(route) => *route,
        Route::Agent => panic!("{model}"),
    };

    // Baseten is listed and dropped; the rest is the `only` list.
    let mut route = route_for("vercel/deepseek/deepseek-v4-flash-0731");
    let pin = pin_route(&mut route, &host(&[], None)).unwrap().unwrap();
    let allowed = Upstreams::Only(vec!["alibaba".into(), "novita".into()]);
    assert_eq!(pin.upstreams, allowed);
    assert_eq!(pin.excluded, ["baseten"]);
    assert_eq!(route.upstreams, Some(allowed));
    let config = route.forwarder_config("/k");
    assert_eq!(
        config["routes"][0]["only"],
        serde_json::json!(["alibaba", "novita"])
    );
    assert!(config["routes"][0].get("ignore").is_none());
    // The listing is read once per process.
    let mut again = route_for("vercel/deepseek/deepseek-v4-flash-0731");
    pin_route(&mut again, &host(&[], None)).unwrap();
    assert_eq!(hits.load(Ordering::SeqCst), 1);

    // A model only Baseten serves stops the run, naming both.
    let mut only_baseten = route_for("vercel/m/only-baseten");
    let error = pin_route(&mut only_baseten, &host(&[], None)).unwrap_err();
    assert_eq!(
        error.to_string(),
        "no upstream vercel could send `m/only-baseten` to is allowed: it is served only by baseten, which BenchFlow does not use"
    );
    // The user's denylist applies too.
    let mut wafer = route_for("vercel/m/wafer-and-novita");
    let pin = pin_route(&mut wafer, &host(&[(DENY_ENV, "wafer")], None))
        .unwrap()
        .unwrap();
    assert_eq!(pin.upstreams, Upstreams::Only(vec!["novita".into()]));
    // An unknown model, and a listing that cannot be read, stop it too.
    let mut unknown = route_for("vercel/m/unknown");
    assert!(matches!(
        pin_route(&mut unknown, &host(&[], None)),
        Err(PinError::NotServed { .. })
    ));
    let mut gone = registry.find("vercel").unwrap().clone();
    gone.routing = Some(Routing {
        style: Style::Only {
            endpoints: "http://127.0.0.1:9/v1/models/{model}/endpoints".into(),
        },
        deny: Vec::new(),
    });
    let mut unreachable = route_for("vercel/deepseek/deepseek-v4-flash-0731");
    unreachable.provider = gone;
    assert!(matches!(
        pin_route(&mut unreachable, &host(&[], None)),
        Err(PinError::Unlisted { .. })
    ));
}

#[test]
fn a_provider_on_the_gateway_is_pinned_even_without_routing_of_its_own() {
    let mut vercel = providers().find("vercel").unwrap().clone();
    assert_eq!(
        vercel.upstream_routing().unwrap().style,
        Style::Only {
            endpoints: "https://ai-gateway.vercel.sh/v1/models/{model}/endpoints".into()
        }
    );
    // A project file that leaves `[routing]` out cannot turn it off.
    vercel.routing = None;
    assert_eq!(vercel.upstream_routing(), Some(Routing::vercel()));
    assert_eq!(
        providers().find("deepseek").unwrap().upstream_routing(),
        None
    );
    let error = parse(
        "name = \"p\"\n[openai]\nbase_url = \"https://x.example/v1\"\n[key]\nenv = [\"P_KEY\"]\n[routing]\nendpoints = \"https://x.example/v1/models\"\n",
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("with `{model}` in it"), "{error}");
}

#[test]
fn routing_says_how_and_never_both() {
    let base = "name = \"p\"\n[openai]\nbase_url = \"https://x.example/v1\"\n[key]\nenv = [\"P_KEY\"]\n[routing]\n";
    let ignore = parse(&format!("{base}ignore = true\ndeny = [\"wafer\"]\n")).unwrap();
    assert_eq!(
        ignore.upstream_routing(),
        Some(Routing {
            style: Style::Ignore,
            deny: vec!["wafer".into()]
        })
    );
    for (extra, expected) in [
        (
            "ignore = true\nendpoints = \"https://x.example/{model}\"\n",
            "not both",
        ),
        ("deny = [\"x\"]\n", "say how the router is pinned"),
    ] {
        let error = parse(&format!("{base}{extra}")).unwrap_err().to_string();
        assert!(error.contains(expected), "{error}\n  expected: {expected}");
    }
    // A gateway-only provider is valid.
    let gateway = parse(
        "name = \"g\"\n[gateway]\nbase_url = \"https://g.example\"\n[key]\nenv = [\"G_KEY\"]\n",
    )
    .unwrap();
    assert_eq!(gateway.gateway.unwrap().auth, Auth::Bearer);
}

/// OpenRouter is pinned without a listing: its requests name the denied
/// upstreams in `provider.ignore`.
#[test]
fn a_route_on_openrouter_ignores_the_denied_upstreams() {
    let mut route = provider_route("opencode", "openrouter/deepseek/deepseek-chat");
    assert_eq!(route.provider.name, "openrouter");
    let pin = pin_route(&mut route, &host(&[(DENY_ENV, "together")], None))
        .unwrap()
        .unwrap();
    let denied = Upstreams::Ignore(vec!["baseten".into(), "together".into()]);
    assert_eq!(pin.upstreams, denied);
    assert_eq!(route.upstreams, Some(denied));
    let config = route.forwarder_config("/k");
    assert_eq!(
        config["routes"][0]["ignore"],
        serde_json::json!(["baseten", "together"])
    );
    assert!(config["routes"][0].get("only").is_none());
    assert_eq!(
        config["routes"][0]["models"],
        serde_json::json!(["deepseek/deepseek-chat"])
    );
}

const ANTHROPIC_KEY: &str = "sk-ant-api03-fake-key-0123456789";
const OPENAI_KEY: &str = "sk-proj-fake-key-0123456789abcdef";

fn own_route(access: &Access) -> &ProviderRoute {
    match &access.route {
        Route::Provider(route) if route.own_key => route,
        other => panic!("not on the agent's own key: {other:?}"),
    }
}

/// A plain API key the agent found goes to the forwarder; the agent keeps
/// picking its own model and helpers, and gets a placeholder.
#[test]
fn claudes_api_key_is_held_by_the_forwarder() {
    let host = host(
        &[
            ("ANTHROPIC_API_KEY", ANTHROPIC_KEY),
            ("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-fake-token-000000"),
        ],
        None,
    );
    let access = resolve(
        &agent("claude"),
        Some("claude-haiku-4-5-20251001"),
        &host,
        &providers(),
    )
    .unwrap();
    let route = own_route(&access);
    assert_eq!(route.provider.name, "anthropic");
    assert_eq!(route.endpoint.base_url, "https://api.anthropic.com");
    assert_eq!(route.model.as_deref(), Some("claude-haiku-4-5-20251001"));
    assert_eq!(
        access.login.label,
        "ANTHROPIC_API_KEY (environment), held by BenchFlow's forwarder for anthropic's Anthropic-compatible endpoint"
    );
    assert!(access.login.env.is_empty() && access.login.files.is_empty());
    assert_eq!(access.login.secrets, [ANTHROPIC_KEY]);
    assert_eq!(access.key.as_ref().unwrap().value(), ANTHROPIC_KEY);
    // Only the endpoint and the key change: no model variables, so Claude
    // Code keeps its own model choices (its helpers' too).
    let env = route
        .agent_env("http://127.0.0.1:41234/anthropic", None)
        .unwrap();
    let expected: BTreeMap<String, String> = [
        ("ANTHROPIC_BASE_URL", "http://127.0.0.1:41234/anthropic"),
        ("ANTHROPIC_AUTH_TOKEN", KEY_PLACEHOLDER),
        ("ANTHROPIC_API_KEY", ""),
        ("CLAUDE_CODE_OAUTH_TOKEN", ""),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect();
    assert_eq!(env, expected);
    assert_eq!(route.acp_model(None).unwrap(), None);
    // The forwarder lets any model through, as the key would on its own.
    let config = route.forwarder_config("/k");
    assert_eq!(config["routes"][0]["models"], serde_json::json!([]));
    assert_eq!(config["routes"][0]["auth"], "x-api-key");

    // A subscription login reaches the agent as it is.
    let oauth = host_without_api_key();
    let access = resolve(&agent("claude"), None, &oauth, &providers()).unwrap();
    assert_eq!(access.route, Route::Agent);
    assert!(access.login.env.contains_key("CLAUDE_CODE_OAUTH_TOKEN"));
    assert!(access.key.is_none());
}

fn host_without_api_key() -> HostEnv {
    host(
        &[("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-fake-token-000000")],
        None,
    )
}

#[test]
fn codexs_api_key_is_held_by_the_forwarder_and_its_config_names_no_empty_model() {
    for var in ["OPENAI_API_KEY", "CODEX_API_KEY"] {
        let access = resolve(
            &agent("codex"),
            None,
            &host(&[(var, OPENAI_KEY)], None),
            &providers(),
        )
        .unwrap();
        let route = own_route(&access);
        assert_eq!(route.provider.name, "openai", "{var}");
        assert_eq!(route.model, None);
        let env = route
            .agent_env("http://127.0.0.1:41234/openai", None)
            .unwrap();
        assert_eq!(env["OPENAI_API_KEY"], KEY_PLACEHOLDER);
        assert_eq!(env["CODEX_API_KEY"], "");
        // No model and no effort: Codex keeps its own defaults.
        assert_eq!(
            env["BENCHFLOW_CODEX_HOME_CONFIG"],
            "# benchflow-codex-home-config\n\
             model_provider = \"benchflow\"\n\
             \n\
             [model_providers.benchflow]\n\
             name = \"BenchFlow forwarder\"\n\
             base_url = \"http://127.0.0.1:41234/openai\"\n\
             env_key = \"OPENAI_API_KEY\"\n\
             wire_api = \"responses\"\n\
             supports_websockets = false\n"
        );
        let high = route.agent_env("http://x", Some("high")).unwrap();
        assert!(high["BENCHFLOW_CODEX_HOME_CONFIG"].contains("model_reasoning_effort = \"high\""));
        assert!(!high["BENCHFLOW_CODEX_HOME_CONFIG"].contains("model = "));
        assert!(
            env.values()
                .chain(high.values())
                .all(|v| !v.contains(OPENAI_KEY))
        );
    }

    // An API key stored in ~/.codex/auth.json too; a ChatGPT login is not.
    let home = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(home.path().join(".codex")).unwrap();
    let auth = home.path().join(".codex/auth.json");
    std::fs::write(&auth, format!(r#"{{"OPENAI_API_KEY": "{OPENAI_KEY}"}}"#)).unwrap();
    let stored = host(&[], Some(home.path().to_path_buf()));
    let access = resolve(&agent("codex"), None, &stored, &providers()).unwrap();
    own_route(&access);
    assert_eq!(
        access.login.label,
        "OPENAI_API_KEY in ~/.codex/auth.json, held by BenchFlow's forwarder for openai's OpenAI-compatible endpoint"
    );
    assert!(access.login.files.is_empty());
    std::fs::write(
        &auth,
        r#"{"tokens": {"refresh_token": "rt-fake-refresh-token-0000", "access_token": "a.b.c"}}"#,
    )
    .unwrap();
    let chatgpt = resolve(&agent("codex"), None, &stored, &providers()).unwrap();
    assert_eq!(chatgpt.route, Route::Agent);
    assert_eq!(chatgpt.login.files.len(), 1);
}

/// A run that points the agent at an endpoint of its own keeps the key with
/// the agent: it is meant for that endpoint, not the provider's.
#[test]
fn a_key_for_an_endpoint_of_the_users_choosing_stays_with_the_agent() {
    let host = host(&[("ANTHROPIC_API_KEY", ANTHROPIC_KEY)], None).with_overrides(
        [(
            "ANTHROPIC_BASE_URL".to_string(),
            "https://llm.example.com/anthropic".to_string(),
        )]
        .into(),
    );
    let access = resolve(&agent("claude"), None, &host, &providers()).unwrap();
    assert_eq!(access.route, Route::Agent);
    assert_eq!(access.login.env["ANTHROPIC_API_KEY"], ANTHROPIC_KEY);
}

/// Router keys never reach a router unpinned: an agent that cannot take
/// them through the forwarder is refused.
#[test]
fn a_routers_key_the_agent_would_take_there_itself_is_refused() {
    let junie = resolve(
        &agent("junie"),
        None,
        &host(
            &[("JUNIE_OPENROUTER_API_KEY", "sk-or-v1-fake-key-0123456789")],
            None,
        ),
        &providers(),
    )
    .unwrap_err();
    let AccessError::Route(error) = &junie else {
        panic!("{junie:?}")
    };
    assert_eq!(
        error.to_string(),
        "agent `junie` would send its requests to openrouter itself (JUNIE_OPENROUTER_API_KEY), where BenchFlow cannot keep them off the upstreams it never uses (baseten)"
    );
    assert!(error.next_step().contains("`openrouter/<model>`"));

    // A `--env` base URL on a router's host is refused the same way.
    let pointed = host_without_api_key().with_overrides(
        [(
            "ANTHROPIC_BASE_URL".to_string(),
            "https://openrouter.ai/api".to_string(),
        )]
        .into(),
    );
    let error = resolve(&agent("claude"), None, &pointed, &providers()).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("to openrouter itself (--env ANTHROPIC_BASE_URL)"),
        "{error}"
    );

    // An agent whose ids name providers keeps its own logins, but not on a
    // router: without a [providers] table it is refused there.
    for name in ["aider", "langgraph"] {
        let own = route(
            &agent(name),
            Some("anthropic/claude-sonnet-4-5"),
            &providers(),
        )
        .unwrap();
        assert_eq!(own, Route::Agent, "{name}");
    }
    for model in ["openrouter/deepseek/deepseek-chat", "vercel/x/y"] {
        assert!(
            matches!(
                route(&agent("langgraph"), Some(model), &providers()),
                Err(RouteError::Unsupported { .. })
            ),
            "{model}"
        );
        // aider has a table for OpenAI-compatible endpoints, so a router's
        // model goes through the forwarder.
        let Route::Provider(on_router) = route(&agent("aider"), Some(model), &providers()).unwrap()
        else {
            panic!("{model}")
        };
        assert_eq!(on_router.spec.protocol, Protocol::OpenAi, "{model}");
    }
}

/// fx speaks the gateway's own API: on its gateway key it runs through the
/// forwarder, pinned, whatever the id's creator prefix.
#[test]
fn fx_runs_on_its_gateway_key_through_the_forwarder() {
    let host = host(&[("AI_GATEWAY_API_KEY", FAKE_KEY)], None);
    for (model, id) in [
        (
            "vercel/deepseek/deepseek-v4-flash-0731",
            "deepseek/deepseek-v4-flash-0731",
        ),
        ("anthropic/claude-sonnet-4.6", "anthropic/claude-sonnet-4.6"),
    ] {
        for name in ["fx", "fx-dev"] {
            let access = resolve(&agent(name), Some(model), &host, &providers()).unwrap();
            let route = own_route(&access);
            assert_eq!(route.provider.name, "vercel");
            assert_eq!(route.endpoint.protocol, Protocol::Gateway);
            assert_eq!(route.model.as_deref(), Some(id), "{name} {model}");
            assert_eq!(route.prefix(), "");
            assert_eq!(route.describe(), "vercel's own API");
            let env = route.agent_env("http://127.0.0.1:41234", None).unwrap();
            assert_eq!(env["FX_GATEWAY_BASE_URL"], "http://127.0.0.1:41234");
            assert_eq!(
                env["FX_GATEWAY_CHAT_URL"],
                "http://127.0.0.1:41234/v4/ai/language-model"
            );
            assert_eq!(env["AI_GATEWAY_API_KEY"], KEY_PLACEHOLDER);
            assert_eq!(env["VERCEL_AI_GATEWAY_API_KEY"], "");
            assert_eq!(env["VERCEL_OIDC_TOKEN"], "");
        }
    }
    // Pinned per model, the forwarder takes only the run's model.
    let access = resolve(
        &agent("fx"),
        Some("vercel/deepseek/deepseek-v4-flash-0731"),
        &host,
        &providers(),
    )
    .unwrap();
    let mut route = own_route(&access).clone();
    route.upstreams = Some(Upstreams::Only(vec!["novita".into()]));
    let config = route.forwarder_config("/k");
    assert_eq!(config["routes"][0]["prefix"], "");
    assert_eq!(
        config["routes"][0]["upstream"],
        "https://ai-gateway.vercel.sh"
    );
    assert_eq!(
        config["routes"][0]["models"],
        serde_json::json!(["deepseek/deepseek-v4-flash-0731"])
    );
    assert_eq!(config["routes"][0]["only"], serde_json::json!(["novita"]));
    // With no model there is nothing to pin, and the run stops.
    let access = resolve(&agent("fx"), None, &host, &providers()).unwrap();
    let mut unpinned = own_route(&access).clone();
    assert_eq!(
        pin_route(&mut unpinned, &host).unwrap_err(),
        PinError::NeedsModel {
            provider: "vercel".into()
        }
    );
    // A router that is not the gateway is refused.
    assert!(matches!(
        route_of("fx", "openrouter/x/y"),
        Err(RouteError::Unsupported { .. })
    ));
}

fn route_of(name: &str, model: &str) -> Result<Route, RouteError> {
    route(&agent(name), Some(model), &providers())
}

#[test]
fn bench_agents_says_which_credentials_the_forwarder_holds() {
    let registry = providers();
    let handling = |name: &str, var: &str, kind| key_handling(&agent(name), var, kind, &registry);
    use crate::credentials::SourceKind::*;
    assert_eq!(
        handling("claude", "ANTHROPIC_API_KEY", ApiKey),
        KeyHandling::Forwarded {
            to: "anthropic's Anthropic-compatible endpoint".into()
        }
    );
    assert!(matches!(
        handling("claude", "CLAUDE_CODE_OAUTH_TOKEN", OauthToken),
        KeyHandling::ToAgent { .. }
    ));
    assert!(matches!(
        handling("claude", "ANTHROPIC_AUTH_TOKEN", AuthToken),
        KeyHandling::ToAgent { .. }
    ));
    assert_eq!(
        handling("codex", "CODEX_API_KEY", ApiKey),
        KeyHandling::Forwarded {
            to: "openai's OpenAI-compatible endpoint".into()
        }
    );
    let KeyHandling::ToAgent { why } = handling("codex", "~/.codex/auth.json", AuthJson) else {
        panic!()
    };
    assert!(
        why.contains("an API key stored in it goes to the forwarder"),
        "{why}"
    );
    assert_eq!(
        handling("fx", "AI_GATEWAY_API_KEY", ApiKey),
        KeyHandling::Forwarded {
            to: "vercel's own API".into()
        }
    );
    assert_eq!(
        handling("gemini", "GEMINI_API_KEY", ApiKey),
        KeyHandling::ToAgent {
            why: "no built-in provider reads it".into()
        }
    );
    assert_eq!(
        handling("langgraph", "OPENAI_API_KEY", ApiKey),
        KeyHandling::ToAgent {
            why: "the agent has no [providers] table for an endpoint that takes it".into()
        }
    );
    assert_eq!(
        handling("qwen-code", "OPENAI_API_KEY", ApiKey),
        KeyHandling::Forwarded {
            to: "openai's OpenAI-compatible endpoint".into()
        }
    );
    assert_eq!(
        handling("junie", "JUNIE_OPENROUTER_API_KEY", ApiKey),
        KeyHandling::Refused {
            router: "openrouter".into()
        }
    );
    // Every built-in agent's credentials have an answer, and the kinds list
    // what each family reads.
    let kinds = crate::credentials::credential_kinds(&agent("claude"));
    assert_eq!(kinds[0], ("ANTHROPIC_API_KEY".to_string(), ApiKey));
    assert_eq!(kinds.len(), 5);
    for manifest in Registry::builtin().agents() {
        for (name, kind) in crate::credentials::credential_kinds(manifest) {
            let _ = key_handling(manifest, &name, kind, &registry).to_string();
        }
    }
}

/// The `[providers]` tables added for the gateway sweep: on a gateway model,
/// each agent gets exactly the forwarder's address, the placeholder key and
/// its model, in the variables it reads, and the model it is given on the
/// command line or over ACP (`$BENCHFLOW_MODEL` headless).
#[test]
fn more_agents_are_pointed_at_the_forwarder_on_a_gateway_model() {
    let base = "http://127.0.0.1:41234/openai";
    let id = "deepseek/deepseek-v4-flash-0731";
    let litellm = format!("openai/{id}");
    let host = format!("{base}/");
    let langchain = format!("openai:{id}");
    // (agent, its variables, the model it is given on the command line or
    // over ACP)
    type Case<'a> = (&'a str, &'a [(&'a str, &'a str)], Option<&'a str>);
    let cases: &[Case] = &[
        (
            "qwen-code",
            &[
                ("OPENAI_BASE_URL", base),
                ("OPENAI_API_KEY", KEY_PLACEHOLDER),
                ("OPENAI_MODEL", id),
            ],
            None,
        ),
        (
            "openhands",
            &[
                ("LLM_BASE_URL", base),
                ("LLM_API_KEY", KEY_PLACEHOLDER),
                ("LLM_MODEL", &litellm),
            ],
            None,
        ),
        (
            "openhands-sdk",
            &[("LLM_BASE_URL", base), ("LLM_API_KEY", KEY_PLACEHOLDER)],
            Some(&litellm),
        ),
        (
            "mini-swe-agent",
            &[
                ("OPENAI_BASE_URL", base),
                ("OPENAI_API_BASE", base),
                ("OPENAI_API_KEY", KEY_PLACEHOLDER),
                ("MSWEA_API_KEY", ""),
            ],
            Some(&litellm),
        ),
        (
            "swe-agent",
            &[
                ("OPENAI_BASE_URL", base),
                ("OPENAI_API_BASE", base),
                ("OPENAI_API_KEY", KEY_PLACEHOLDER),
            ],
            Some(&litellm),
        ),
        (
            "aider",
            &[
                ("OPENAI_API_BASE", base),
                ("OPENAI_BASE_URL", base),
                ("OPENAI_API_KEY", KEY_PLACEHOLDER),
            ],
            Some(&litellm),
        ),
        (
            "goose",
            &[
                ("OPENAI_API_KEY", KEY_PLACEHOLDER),
                ("OPENAI_BASE_URL", ""),
                ("GOOSE_PROVIDER", "openai"),
                ("GOOSE_MODEL", id),
                ("OPENAI_HOST", &host),
                ("OPENAI_BASE_PATH", "chat/completions"),
            ],
            None,
        ),
        (
            "hermes",
            &[
                ("OPENAI_BASE_URL", base),
                ("OPENAI_API_KEY", KEY_PLACEHOLDER),
                ("BENCHFLOW_MODEL", &litellm),
            ],
            Some(&litellm),
        ),
        (
            "mcode",
            &[
                ("OPENAI_BASE_URL", base),
                ("OPENAI_API_KEY", KEY_PLACEHOLDER),
                ("BENCHFLOW_MODEL", &litellm),
            ],
            Some(&litellm),
        ),
        (
            "kimi-code",
            &[
                ("OPENAI_BASE_URL", base),
                ("OPENAI_API_KEY", KEY_PLACEHOLDER),
            ],
            Some(&litellm),
        ),
        (
            "trae-agent",
            &[
                ("OPENAI_BASE_URL", base),
                ("OPENAI_API_KEY", KEY_PLACEHOLDER),
            ],
            Some(&litellm),
        ),
        (
            "code-puppy",
            &[
                ("OPENAI_BASE_URL", base),
                ("OPENAI_API_KEY", KEY_PLACEHOLDER),
            ],
            Some(&litellm),
        ),
        (
            "nemo-agent",
            &[
                ("OPENAI_BASE_URL", base),
                ("OPENAI_API_KEY", KEY_PLACEHOLDER),
            ],
            Some(&litellm),
        ),
        (
            "deerflow",
            &[
                ("OPENAI_BASE_URL", base),
                ("OPENAI_API_KEY", KEY_PLACEHOLDER),
            ],
            Some(&litellm),
        ),
        (
            "cline",
            &[
                ("OPENAI_API_KEY", KEY_PLACEHOLDER),
                ("BENCHFLOW_CLINE_BASE_URL", base),
                ("BENCHFLOW_CLINE_MODEL", id),
            ],
            None,
        ),
        (
            "deepagents",
            &[
                ("OPENAI_BASE_URL", base),
                ("OPENAI_API_KEY", KEY_PLACEHOLDER),
                ("BENCHFLOW_DEEPAGENTS_MODEL", &langchain),
            ],
            None,
        ),
    ];
    for (name, vars, model) in cases {
        let route = provider_route(name, &format!("vercel/{id}"));
        assert_eq!(route.endpoint.protocol, Protocol::OpenAi, "{name}");
        assert_eq!(route.endpoint.base_url, "https://ai-gateway.vercel.sh/v1");
        assert_eq!(route.prefix(), "/openai");
        let expected: BTreeMap<String, String> = vars
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        assert_eq!(route.agent_env(base, None).unwrap(), expected, "{name}");
        assert_eq!(route.acp_model(None).unwrap().as_deref(), *model, "{name}");
        assert_eq!(route.allowed_models(), vec![id.to_string()], "{name}");
    }

    // Agents configured through a file BenchFlow writes from a variable.
    let env_of = |name: &str| {
        let route = provider_route(name, &format!("vercel/{id}"));
        assert_eq!(route.endpoint.protocol, Protocol::OpenAi, "{name}");
        (
            route.agent_env(base, None).unwrap(),
            route.acp_model(None).unwrap(),
        )
    };
    // pi: a provider of its own in models.json, the default in settings.json.
    let (env, model) = env_of("pi");
    assert_eq!(model, None);
    assert_eq!(env["OPENAI_API_KEY"], KEY_PLACEHOLDER);
    let models: Value = serde_json::from_str(&env["BENCHFLOW_PI_MODELS_JSON"]).unwrap();
    let provider = &models["providers"]["benchflow"];
    assert_eq!(provider["baseUrl"], base);
    assert_eq!(provider["apiKey"], "$OPENAI_API_KEY");
    assert_eq!(provider["api"], "openai-completions");
    assert_eq!(provider["models"][0]["id"], id);
    let settings: Value = serde_json::from_str(&env["BENCHFLOW_PI_SETTINGS_JSON"]).unwrap();
    assert_eq!(settings["defaultProvider"], "benchflow");
    assert_eq!(settings["defaultModel"], id);
    // mimo: OpenCode's provider shape in MIMOCODE_CONFIG_CONTENT.
    let (env, model) = env_of("mimo");
    assert_eq!(model.as_deref(), Some(format!("benchflow/{id}").as_str()));
    let config: Value = serde_json::from_str(&env["MIMOCODE_CONFIG_CONTENT"]).unwrap();
    assert_eq!(config["provider"]["benchflow"]["options"]["baseURL"], base);
    assert_eq!(
        config["provider"]["benchflow"]["options"]["apiKey"],
        KEY_PLACEHOLDER
    );
    assert_eq!(config["small_model"], format!("benchflow/{id}"));
    // mistral-vibe: a generic OpenAI-style provider in config.toml.
    let (env, model) = env_of("mistral-vibe");
    assert_eq!(model, None);
    assert_eq!(env["OPENAI_API_KEY"], KEY_PLACEHOLDER);
    let config: toml::Value = toml::from_str(&env["BENCHFLOW_VIBE_CONFIG"]).unwrap();
    assert_eq!(config["active_model"].as_str(), Some(id));
    assert_eq!(config["providers"][0]["api_base"].as_str(), Some(base));
    assert_eq!(config["providers"][0]["backend"].as_str(), Some("generic"));
    assert_eq!(config["providers"][0]["api_style"].as_str(), Some("openai"));
    assert_eq!(config["models"][0]["name"].as_str(), Some(id));
    assert_eq!(config["models"][0]["provider"].as_str(), Some("benchflow"));
    // trae-agent's openai client calls the Responses API.
    assert_eq!(
        provider_route("trae-agent", &format!("vercel/{id}"))
            .spec
            .api,
        Some(OpenAiApi::Responses)
    );
    // kimi: an openai_legacy provider with thinking on (off, kimi sends
    // `reasoning_effort: null`, which the gateway refuses).
    let (env, model) = env_of("kimi");
    assert_eq!(model, None);
    assert_eq!(env["OPENAI_BASE_URL"], base);
    let config: toml::Value = toml::from_str(&env["BENCHFLOW_KIMI_CONFIG"]).unwrap();
    assert_eq!(config["default_model"].as_str(), Some("benchflow"));
    assert_eq!(config["default_thinking"].as_bool(), Some(true));
    assert_eq!(
        config["providers"]["benchflow"]["type"].as_str(),
        Some("openai_legacy")
    );
    assert_eq!(
        config["providers"]["benchflow"]["base_url"].as_str(),
        Some(base)
    );
    assert_eq!(
        config["providers"]["benchflow"]["api_key"].as_str(),
        Some(KEY_PLACEHOLDER)
    );
    assert_eq!(config["models"]["benchflow"]["model"].as_str(), Some(id));
    assert_eq!(
        config["models"]["benchflow"]["capabilities"][0].as_str(),
        Some("thinking")
    );
    // grok: the model as a chat_completions entry, every helper pinned to it.
    let (env, model) = env_of("grok");
    assert_eq!(model, None);
    let config: toml::Value = toml::from_str(&env["BENCHFLOW_GROK_CONFIG"]).unwrap();
    for helper in [
        "default",
        "session_summary",
        "image_description",
        "web_search",
    ] {
        assert_eq!(config["models"][helper].as_str(), Some(id), "{helper}");
    }
    let entry = &config["model"][id];
    assert_eq!(entry["base_url"].as_str(), Some(base));
    assert_eq!(entry["env_key"].as_str(), Some("OPENAI_API_KEY"));
    assert_eq!(entry["api_backend"].as_str(), Some("chat_completions"));
    assert_eq!(
        config["harness"]["disable_codebase_upload"].as_bool(),
        Some(true)
    );
    // openclaw: a custom openai-completions provider beside Harbor's config.
    let (env, model) = env_of("openclaw");
    assert_eq!(model.as_deref(), Some(format!("benchflow/{id}").as_str()));
    let config: Value = serde_json::from_str(&env["BENCHFLOW_OPENCLAW_CONFIG"]).unwrap();
    let provider = &config["models"]["providers"]["benchflow"];
    assert_eq!(provider["baseUrl"], base);
    assert_eq!(provider["apiKey"], KEY_PLACEHOLDER);
    assert_eq!(provider["api"], "openai-completions");
    assert_eq!(provider["models"][0]["id"], id);
    assert_eq!(config["tools"]["deny"][0], "message");
    // junie: a custom model profile on the full Chat Completions URL.
    let (env, model) = env_of("junie");
    assert_eq!(model, None);
    assert_eq!(env["BENCHFLOW_JUNIE_MODEL"], "custom:benchflow");
    let profile: Value = serde_json::from_str(&env["BENCHFLOW_JUNIE_PROFILE"]).unwrap();
    assert_eq!(profile["id"], id);
    assert_eq!(profile["baseUrl"], format!("{base}/chat/completions"));
    assert_eq!(profile["apiType"], "OpenAICompletion");
    assert_eq!(profile["apiKey"], KEY_PLACEHOLDER);
}

/// Code Puppy on a provider: an `openai/` model with OPENAI_BASE_URL set
/// becomes a custom_openai endpoint there, keyed by the placeholder; without
/// the URL it stays Code Puppy's native openai type.
#[cfg(unix)]
#[test]
fn code_puppy_registers_an_openai_model_on_the_forwarder_as_a_custom_endpoint() {
    let command = agent("code-puppy").headless.unwrap().command;
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("task.md"), "say hi").unwrap();
    let extra_models = |base_url: Option<&str>| -> Value {
        let home = dir.path().join(format!("home-{}", base_url.is_some()));
        std::fs::create_dir_all(&home).unwrap();
        let mut sh = std::process::Command::new("sh");
        sh.args(["-c", &command])
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env("HOME", &home)
            .env("BENCHFLOW_INSTRUCTION_FILE", dir.path().join("task.md"))
            .env("BENCHFLOW_MODEL", "openai/deepseek/deepseek-v4-flash-0731")
            .env("OPENAI_API_KEY", KEY_PLACEHOLDER);
        if let Some(url) = base_url {
            sh.env("OPENAI_BASE_URL", url);
        }
        // Code Puppy itself is not installed here: the command fails after
        // writing its model file.
        let _ = sh.output().unwrap();
        let text = std::fs::read_to_string(home.join(".code_puppy/extra_models.json")).unwrap();
        serde_json::from_str(&text).unwrap()
    };
    let key = "openai/deepseek/deepseek-v4-flash-0731";
    let on_forwarder = extra_models(Some("http://127.0.0.1:41234/openai"));
    assert_eq!(on_forwarder[key]["type"], "custom_openai");
    assert_eq!(on_forwarder[key]["name"], "deepseek/deepseek-v4-flash-0731");
    assert_eq!(
        on_forwarder[key]["custom_endpoint"]["url"],
        "http://127.0.0.1:41234/openai"
    );
    assert_eq!(
        on_forwarder[key]["custom_endpoint"]["api_key"],
        "$CODE_PUPPY_API_KEY"
    );
    let native = extra_models(None);
    assert_eq!(native[key]["type"], "openai");
}

#[test]
fn a_misspelled_model_is_answered_with_the_closest_listed_one() {
    use super::routing::{closest_model, models_url, parse_model_ids};
    assert_eq!(
        models_url("https://ai-gateway.vercel.sh/v1/models/{model}/endpoints").as_deref(),
        Some("https://ai-gateway.vercel.sh/v1/models")
    );
    let ids = parse_model_ids(&serde_json::json!({"data": [
        {"id": "deepseek/deepseek-v4-flash"},
        {"id": "deepseek/deepseek-v4-flash-0731"},
        {"id": "deepseek/deepseek-v4-pro"},
    ]}));
    assert_eq!(ids.len(), 3);
    let close = closest_model("deepseek/deepseek-v4-flsh", &ids);
    assert_eq!(close.as_deref(), Some("deepseek/deepseek-v4-flash"));
    assert_eq!(closest_model("qwen/qwen3-coder", &ids), None);
    let e = PinError::NotServed {
        provider: "vercel".into(),
        model: "deepseek/deepseek-v4-flsh".into(),
        close,
        list: Some("https://ai-gateway.vercel.sh/v1/models".into()),
    };
    assert_eq!(
        e.to_string(),
        "vercel serves no model `deepseek/deepseek-v4-flsh`"
    );
    assert_eq!(
        e.next_step(),
        "did you mean `vercel/deepseek/deepseek-v4-flash`?"
    );
    let e = PinError::NotServed {
        provider: "vercel".into(),
        model: "x".into(),
        close: None,
        list: Some("https://ai-gateway.vercel.sh/v1/models".into()),
    };
    assert_eq!(
        e.next_step(),
        "check the model id; https://ai-gateway.vercel.sh/v1/models lists every model vercel serves"
    );
}
