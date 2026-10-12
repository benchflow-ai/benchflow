//! The provider hook both manifest harnesses (ACP and headless) share.
//!
//! A run whose model names a provider (`vercel/<model>`), or whose agent
//! found a plain API key a provider reads (`ANTHROPIC_API_KEY` for Claude
//! Code), goes through BenchFlow's forwarder: before the agent starts, the
//! forwarder starts as root in the sandbox holding the key, and the agent
//! gets its address and a placeholder key (`benchflow_agent::providers`). On
//! a router (the Vercel AI Gateway, OpenRouter) every request is kept off
//! denied upstreams, and a model that cannot be pinned stops the run before
//! any sandbox starts. The forwarder's log (one line per request, no key) is
//! kept as `agent/forwarder.log`, headed by the pin.

use benchflow_agent::AgentManifest;
use benchflow_agent::credentials::{CredentialError, HostEnv, Login};
use benchflow_agent::providers::{
    self, Access, AccessError, Forwarder, ForwarderError, ForwarderReport, Pin, PinError,
    ProviderKey, ProviderRegistry, ProviderRoute, Route, RouteError, Upstreams,
};
use benchflow_sandbox::SandboxError;

use crate::agent::{AgentContext, COST_BY_PROVIDER, Usage};
use crate::outcome::{Owner, Phase, Problem};

/// A run through the forwarder: the endpoint, the key it holds, and the pin.
pub(crate) struct ProviderAccess {
    pub route: Box<ProviderRoute>,
    pub key: ProviderKey,
    /// On a router: what every request names to stay off denied upstreams.
    pub pin: Option<Pin>,
}

/// The login `manifest`'s agent runs on with `model` and `effort`, or why it
/// cannot run, and the forwarder's part when the run goes through it. The
/// effort is checked and a router's upstreams are pinned here, before any
/// sandbox starts.
pub(crate) fn resolve(
    manifest: &AgentManifest,
    model: Option<&str>,
    effort: Option<&str>,
    host: &HostEnv,
    providers: &ProviderRegistry,
) -> (Result<Login, Problem>, Option<ProviderAccess>) {
    match providers::resolve(manifest, model, host, providers) {
        Ok(Access {
            route: Route::Provider(mut route),
            login,
            key: Some(key),
        }) => match route.acp_model(effort) {
            // An effort the agent's variables cannot carry is refused here.
            Err(e) => (Err(route_problem(&e)), None),
            Ok(_) => match providers::pin_route(&mut route, host) {
                Err(e) => (Err(pin_problem(&e)), None),
                Ok(pin) => (Ok(login), Some(ProviderAccess { route, key, pin })),
            },
        },
        Ok(access) => (Ok(access.login), None),
        Err(AccessError::Login(e)) => (Err(login_problem(&e)), None),
        Err(AccessError::Route(e)) => (Err(route_problem(&e)), None),
    }
}

/// Keep the key the forwarder holds out of the agent's environment: drop
/// every variable whose value holds it (a key given with `--env` arrives in
/// the task and run environment under its own name, which the agent's
/// provider table may not clear). Call it before the forwarder's own
/// variables are added.
pub(crate) fn withhold_key(
    env: &mut std::collections::BTreeMap<String, String>,
    access: &ProviderAccess,
) {
    let key = access.key.value().trim();
    if key.is_empty() {
        return;
    }
    env.retain(|_, v| !v.contains(key));
}

/// Start the forwarder for `access` in the trial's sandbox, and say where
/// the requests may go.
pub(crate) async fn start(
    ctx: &AgentContext<'_>,
    access: &ProviderAccess,
) -> Result<Forwarder, Problem> {
    let forwarder = Forwarder::start(ctx.sandbox, &access.route, &access.key)
        .await
        .map_err(|e| forwarder_problem(&e))?;
    let model = access
        .route
        .model
        .as_deref()
        .unwrap_or("the agent's own model");
    let mut line = format!(
        "{} requests for {model} go through BenchFlow's forwarder ({})",
        access.route.provider.name, access.key.source
    );
    if let Some(pin) = &access.pin {
        line.push_str(&format!(
            "; upstreams: {} (never {})",
            pin.upstreams.describe(),
            providers::ALWAYS_DENIED.join(", ")
        ));
    }
    // A test build's fault endpoint in place of the provider: each trial
    // says so, not only `bench run`'s banner (E5 review B4).
    if let Some(to) = providers::test_upstream_note(&HostEnv::from_process()) {
        line.push_str(&format!("; sent to {to}, not the provider"));
    }
    ctx.events.log(ctx.trial, line);
    Ok(forwarder)
}

/// Stop the forwarder and keep its log (one line per request, no key) as
/// `agent/forwarder.log`, headed by the pin every request carried.
pub(crate) async fn keep_log(
    forwarder: Forwarder,
    pin: Option<&Pin>,
    ctx: &AgentContext<'_>,
) -> ForwarderReport {
    let report = forwarder.stop().await;
    let dir = ctx.trial_dir.join("agent");
    let _ = std::fs::create_dir_all(&dir);
    let mut log = String::new();
    if let Some(pin) = pin {
        let (kind, list) = match &pin.upstreams {
            Upstreams::Only(list) => ("only", list),
            Upstreams::Ignore(list) => ("ignore", list),
        };
        let mut head = serde_json::json!({"model": pin.model, "excluded": pin.excluded});
        head[kind] = serde_json::json!(list);
        log.push_str(&serde_json::json!({ "upstreams": head }).to_string());
        log.push('\n');
    }
    log.push_str(&report.log);
    let _ = std::fs::write(dir.join("forwarder.log"), log);
    if let Some(code) = report.exited_early {
        let code = code.map_or_else(|| "no exit code".to_string(), |c| format!("exit code {c}"));
        ctx.events.log(
            ctx.trial,
            format!("the provider forwarder exited before the agent finished ({code}); see agent/forwarder.log"),
        );
    }
    report
}

/// The provider rejected the key on every model request (R212): whatever
/// else the agent did (retry, go quiet until its time ran out), it never had
/// a model, so the trial is the key's problem, not a score.
pub(crate) fn rejected_key(report: &ForwarderReport, access: &ProviderAccess) -> Option<Problem> {
    let (status, count) = report.rejected_key()?;
    let provider = &access.route.provider.name;
    let (what, check) = if status == 401 {
        ("rejected the key", "check the key")
    } else {
        (
            "refused the key's requests (it lacks access to the model, or is not allowed)",
            "check the key and its access to the model",
        )
    };
    Some(Problem::new(
        "R212",
        Owner::Model,
        Phase::Agent,
        format!(
            "{provider} {what} on every model request ({count} answered HTTP {status}); the key is {}",
            access.key.source
        ),
        format!(
            "{check}: `bench doctor` tries it against {provider}; agent/forwarder.log in the trial folder has each request"
        ),
    ))
}

/// Record what the provider said the run used and cost, in place of what
/// the agent said. An agent prices its own tokens at the prices it knows
/// (Claude Code at Claude's, whatever model it ran), so on a run through the
/// forwarder its figure is never kept: the cost is the sum the provider's
/// answers gave, or unknown, with a note saying why. The token counts are the
/// provider's when its answers had any, else the agent's.
pub(crate) fn provider_usage(report: &ForwarderReport, agent: &str, usage: &mut Usage) {
    let spend = report.spend();
    if spend.with_usage > 0 {
        usage.input_tokens = spend.input_tokens;
        usage.cache_read_tokens = spend.cache_read_tokens;
        usage.cache_write_tokens = spend.cache_write_tokens;
        usage.output_tokens = spend.output_tokens;
        usage.source = "forwarder".into();
    }
    let claimed = usage.cost_usd.take();
    usage.cost_source = None;
    usage.cost_note = None;
    if let Some(cost) = spend.cost_usd() {
        usage.cost_usd = Some(cost);
        usage.cost_source = Some(COST_BY_PROVIDER.into());
        return;
    }
    if spend.calls == 0 && usage.total() == 0 && claimed.is_none() {
        return;
    }
    let mut note = spend.unknown_cost().unwrap_or_default();
    if let Some(claimed) = claimed {
        note.push_str(&format!(
            "; {agent} reported ${claimed:.4}, priced by the agent itself, which is not kept"
        ));
    }
    usage.cost_note = Some(note);
}

/// No login (R211): what was looked for, and how to log in.
pub(crate) fn login_problem(e: &CredentialError) -> Problem {
    // The credentials layer names a vendor account in its next step.
    let next = e.next_step().to_string();
    Problem::new(
        "R211",
        Owner::You,
        Phase::Setup,
        e.to_string()
            .split("; next: ")
            .next()
            .unwrap_or_default()
            .to_string(),
        next,
    )
}

/// The agent cannot use the provider the model names, or cannot be kept
/// off denied upstreams (R215).
pub(crate) fn route_problem(e: &RouteError) -> Problem {
    // An agent that would reach a router itself is the agent's to fix.
    let owner = match e {
        RouteError::Unpinnable { .. } => Owner::Agent,
        _ => Owner::You,
    };
    Problem::new("R215", owner, Phase::Setup, e.to_string(), e.next_step())
}

/// The model's requests cannot be pinned to allowed upstreams: R215 when no
/// allowed upstream serves it (or the run names no model to pin), R214 when
/// the gateway's list cannot be read.
pub(crate) fn pin_problem(e: &PinError) -> Problem {
    match e {
        PinError::Unlisted { .. } => Problem::new(
            "R214",
            Owner::Model,
            Phase::Setup,
            e.to_string(),
            e.next_step(),
        )
        .transient(),
        PinError::AllDenied { .. } | PinError::NotServed { .. } | PinError::NeedsModel { .. } => {
            Problem::new(
                "R215",
                Owner::You,
                Phase::Setup,
                e.to_string(),
                e.next_step(),
            )
        }
    }
}

/// The forwarder did not start: the sandbox's problem (or this machine's,
/// when a file BenchFlow writes could not be read), an agent image without
/// Node.js, or BenchFlow's own program failing.
pub(crate) fn forwarder_problem(e: &ForwarderError) -> Problem {
    let message = e.to_string();
    match e {
        ForwarderError::NoNode => {
            Problem::new("R201", Owner::Agent, Phase::Setup, message, e.next_step())
        }
        ForwarderError::Sandbox {
            error: SandboxError::Io { .. },
            ..
        } => Problem::new(
            "R107",
            Owner::You,
            Phase::Setup,
            message,
            "fix what the message names (a missing or unreadable file, a full disk), then run again",
        ),
        ForwarderError::Sandbox { .. } => {
            Problem::new("R103", Owner::Sandbox, Phase::Setup, message, e.next_step()).transient()
        }
        ForwarderError::NotReady { .. } => Problem::new(
            "R901",
            Owner::Benchflow,
            Phase::Setup,
            message,
            e.next_step(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    fn agent(name: &str) -> AgentManifest {
        benchflow_agent::Registry::builtin()
            .find(name)
            .unwrap()
            .clone()
    }

    fn host(vars: &[(&str, &str)]) -> HostEnv {
        HostEnv::new(
            vars.iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect::<BTreeMap<_, _>>(),
            None,
        )
    }

    #[test]
    fn a_router_key_the_agent_would_take_there_itself_is_the_agents_problem() {
        let (login, access) = resolve(
            &agent("junie"),
            None,
            None,
            &host(&[("JUNIE_OPENROUTER_API_KEY", "sk-or-v1-fake-key-0123456789")]),
            &ProviderRegistry::builtin(),
        );
        let p = login.unwrap_err();
        assert!(access.is_none());
        assert_eq!((p.code.as_str(), p.owner), ("R215", Owner::Agent));
        assert!(p.message.contains("openrouter itself"), "{}", p.message);
    }

    #[test]
    fn a_gateway_agent_without_a_model_cannot_be_pinned() {
        let (login, access) = resolve(
            &agent("fx"),
            None,
            None,
            &host(&[("AI_GATEWAY_API_KEY", "vck_fake-gateway-key-0123456789")]),
            &ProviderRegistry::builtin(),
        );
        let p = login.unwrap_err();
        assert!(access.is_none());
        assert_eq!((p.code.as_str(), p.owner), ("R215", Owner::You));
        assert!(p.next.contains("--model"), "{}", p.next);
    }

    #[test]
    fn an_openrouter_route_is_pinned_without_the_network() {
        let (login, access) = resolve(
            &agent("opencode"),
            Some("openrouter/deepseek/deepseek-chat"),
            None,
            &host(&[("OPENROUTER_API_KEY", "sk-or-v1-fake-key-0123456789")]),
            &ProviderRegistry::builtin(),
        );
        assert!(login.is_ok());
        let access = access.unwrap();
        assert_eq!(
            access.pin.unwrap().upstreams,
            Upstreams::Ignore(vec!["baseten".into()])
        );
    }

    fn report(lines: &[serde_json::Value]) -> ForwarderReport {
        ForwarderReport {
            log: lines.iter().map(|l| format!("{l}\n")).collect(),
            ..ForwarderReport::default()
        }
    }

    /// Claude Code headless on deepseek-v4-flash through the gateway, as
    /// T7.1 recorded it: Claude Code said $0.1135 at Claude's prices.
    fn claude_on_deepseek() -> Usage {
        Usage {
            input_tokens: 16913,
            cache_read_tokens: 29184,
            output_tokens: 575,
            cost_usd: Some(0.1135),
            source: "headless".into(),
            cost_source: Some(crate::agent::COST_BY_AGENT.into()),
            ..Usage::default()
        }
    }

    #[test]
    fn the_providers_cost_replaces_the_agents_own() {
        let log = report(&[
            serde_json::json!({"method": "POST", "path": "/anthropic/v1/messages", "status": 200,
                "usage": {"input": 16000, "cache_read": 29184, "cache_write": 0, "output": 500}, "cost": 0.0021}),
            serde_json::json!({"method": "POST", "path": "/anthropic/v1/messages", "status": 200,
                "usage": {"input": 913, "cache_read": 0, "cache_write": 0, "output": 75}, "cost": 0.0003}),
            // Not billed: a token count, the forwarder's own refusal, a failure.
            serde_json::json!({"method": "POST", "path": "/anthropic/v1/messages/count_tokens", "status": 200}),
            serde_json::json!({"method": "POST", "path": "/anthropic/v1/messages", "status": 403, "note": "this run may use only m"}),
            serde_json::json!({"method": "POST", "path": "/anthropic/v1/messages", "status": 500}),
        ]);
        let mut usage = claude_on_deepseek();
        provider_usage(&log, "claude", &mut usage);
        assert_eq!(usage.cost_usd, Some(0.0021 + 0.0003));
        assert_eq!(usage.cost_source.as_deref(), Some("provider"));
        assert_eq!(usage.cost_note, None);
        assert_eq!(
            (
                usage.input_tokens,
                usage.cache_read_tokens,
                usage.output_tokens
            ),
            (16913, 29184, 575)
        );
        assert_eq!(usage.source, "forwarder");
    }

    #[test]
    fn without_the_providers_cost_the_cost_is_unknown_and_the_record_says_why() {
        // A provider's own API (Anthropic's, DeepSeek's) reports tokens only.
        let log = report(&[
            serde_json::json!({"method": "POST", "path": "/anthropic/v1/messages",
            "status": 200, "usage": {"input": 100, "cache_read": 0, "cache_write": 0, "output": 7}}),
        ]);
        let mut usage = claude_on_deepseek();
        provider_usage(&log, "claude", &mut usage);
        assert_eq!(usage.cost_usd, None);
        assert_eq!(usage.cost_source, None);
        let note = usage.cost_note.unwrap();
        assert!(note.contains("did not say what they cost"), "{note}");
        assert!(note.contains("claude reported $0.1135"), "{note}");
        assert_eq!((usage.input_tokens, usage.output_tokens), (100, 7));

        // Answers without usage (a format the forwarder cannot read): the
        // agent's tokens stay, its cost does not.
        let log = report(&[serde_json::json!({"method": "POST", "path": "/v1/x", "status": 200})]);
        let mut usage = claude_on_deepseek();
        provider_usage(&log, "claude", &mut usage);
        assert_eq!((usage.cost_usd, usage.input_tokens), (None, 16913));
        assert_eq!(usage.source, "headless");
        assert!(usage.cost_note.is_some());
    }

    #[test]
    fn a_run_that_called_no_model_gets_no_note() {
        let mut usage = Usage::default();
        provider_usage(&report(&[]), "claude", &mut usage);
        assert_eq!(usage, Usage::default());
    }

    #[test]
    fn stage_costs_add_up_only_when_each_is_known() {
        let priced = |cost: Option<f64>, tokens: u64| Usage {
            input_tokens: tokens,
            cost_usd: cost,
            cost_source: cost.map(|_| "provider".to_string()),
            source: "forwarder".into(),
            ..Usage::default()
        };
        let mut total = Usage::default();
        total.add(&priced(Some(0.01), 10));
        total.add(&priced(Some(0.02), 10));
        assert!((total.cost_usd.unwrap() - 0.03).abs() < 1e-12);
        assert_eq!(total.cost_source.as_deref(), Some("provider"));
        // A stage that used nothing does not make the cost unknown.
        total.add(&priced(None, 0));
        assert!(total.cost_usd.is_some());
        total.add(&priced(None, 10));
        assert_eq!((total.cost_usd, total.cost_source.as_deref()), (None, None));
        assert!(total.cost_note.is_some());
        assert_eq!(total.input_tokens, 30);
    }
}
