//! The native model client on the real Vercel AI Gateway, pinned to the
//! allowed upstreams: one small call per API.
//!
//! Runs only with `BENCHFLOW_LIVE_GATEWAY=1`. The key comes from the gateway
//! provider's usual places (`AI_GATEWAY_API_KEY`, or
//! `~/.config/benchflow/vercel-ai-gateway.env`) and is never printed.
//!
//!     BENCHFLOW_LIVE_GATEWAY=1 cargo test -p benchflow-agent --test gateway_pin_live -- --nocapture
//!
//! Each call goes through [`ModelClient`] exactly as a native agent's does.
//! The request it records must name only allowed upstreams (no Baseten), and
//! the gateway's routing metadata in the answer must say the provider set
//! was restricted to them and name the upstream that served the call.

use std::collections::BTreeMap;
use std::time::Duration;

use benchflow_agent::credentials::HostEnv;
use benchflow_agent::native::chat::{ChatRequest, Message};
use benchflow_agent::native::{BuiltinProviders, ClientOptions, EndpointResolver, ModelClient};
use benchflow_agent::providers::{ProviderRegistry, Upstreams, find_provider_key, routing};
use benchflow_agent::trajectory::Redactor;
use serde_json::Value;

/// Every `routing` object in a JSON value (the gateway puts it under
/// `provider_metadata.gateway` at different depths per API).
fn routing_records(value: &Value, out: &mut Vec<Value>) {
    match value {
        Value::Object(map) => {
            if let Some(r) = map.get("routing").filter(|r| r.is_object()) {
                out.push(r.clone());
            }
            for v in map.values() {
                routing_records(v, out);
            }
        }
        Value::Array(items) => {
            for v in items {
                routing_records(v, out);
            }
        }
        _ => {}
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn native_gateway_calls_reach_only_allowed_upstreams() {
    if std::env::var("BENCHFLOW_LIVE_GATEWAY").ok().as_deref() != Some("1") {
        eprintln!("skipped: set BENCHFLOW_LIVE_GATEWAY=1 to call the Vercel AI Gateway");
        return;
    }
    let vercel = ProviderRegistry::builtin().find("vercel").unwrap().clone();
    let key = find_provider_key(&vercel, &HostEnv::from_process()).expect("a gateway key");
    let host = HostEnv::new(
        BTreeMap::from([("AI_GATEWAY_API_KEY".to_string(), key.value().to_string())]),
        None,
    );
    let dir = tempfile::tempdir().unwrap();
    for model in [
        "vercel/deepseek/deepseek-v4-flash-0731",
        "vercel-anthropic/deepseek/deepseek-v4-flash-0731",
    ] {
        let mut endpoint = BuiltinProviders.resolve(model, &host).unwrap();
        let upstreams = routing::for_native_endpoint(&endpoint.provider, &endpoint.base_url)
            .expect("a gateway endpoint is pinned");
        let pin = routing::pin(&endpoint.provider, &upstreams, Some(&endpoint.model), &host)
            .await
            .unwrap();
        assert!(pin.excluded.contains(&"baseten".to_string()), "{pin:?}");
        let Upstreams::Only(allowed) = pin.upstreams.clone() else {
            panic!("the gateway is pinned with an allowlist: {pin:?}")
        };
        endpoint.upstreams = Some(pin.upstreams.clone());
        let log = dir.path().join(format!("{}.jsonl", endpoint.provider));
        let client = ModelClient::new(
            endpoint,
            ClientOptions {
                timeout: Duration::from_secs(60),
                max_attempts: 2,
                ..ClientOptions::default()
            },
        )
        .unwrap()
        .with_log(&log, Redactor::default());
        let request = ChatRequest {
            messages: vec![Message::user("Say hi")],
            max_tokens: Some(16),
            ..ChatRequest::default()
        };
        client.chat(&request, "live-check").await.unwrap();

        let recorded: Value = serde_json::from_str(
            std::fs::read_to_string(&log)
                .unwrap()
                .lines()
                .last()
                .unwrap(),
        )
        .unwrap();
        let only: Vec<String> = serde_json::from_value(
            recorded["request"]["providerOptions"]["gateway"]["only"].clone(),
        )
        .unwrap();
        assert_eq!(only, allowed);
        assert!(!only.iter().any(|u| u == "baseten"));
        let mut records = Vec::new();
        routing_records(&recorded["response"], &mut records);
        let routing = records.first().expect("routing metadata in the answer");
        let planning = routing["planningReasoning"].as_str().unwrap_or("");
        let served_by = routing["finalProvider"].as_str().unwrap_or("");
        // The gateway's own words: the set it was restricted to, and the
        // order it would try. (Its reasoning may first rank every provider,
        // Baseten included, before the restriction applies.)
        let restricted = sentence(planning, "Provider set restricted to: ");
        let order = sentence(planning, "Total execution order: ");
        let attempted: Vec<String> = routing["modelAttempts"]
            .as_array()
            .into_iter()
            .flatten()
            .flat_map(|m| {
                m["providerAttempts"]
                    .as_array()
                    .cloned()
                    .unwrap_or_default()
            })
            .filter_map(|a| a["provider"].as_str().map(str::to_string))
            .collect();
        println!(
            "{model}: sent only = [{}]; gateway restricted to [{restricted}]; execution order {order}; attempts {attempted:?}; served by {served_by}",
            only.join(", ")
        );
        assert_eq!(
            restricted
                .split(", ")
                .map(str::to_string)
                .collect::<Vec<_>>(),
            only,
            "{planning}"
        );
        assert!(
            !order.is_empty() && !order.contains("baseten"),
            "{planning}"
        );
        assert!(!attempted.is_empty() && !attempted.iter().any(|a| a == "baseten"));
        assert!(
            !served_by.is_empty() && served_by != "baseten",
            "{served_by}"
        );
        assert!(!std::fs::read_to_string(&log).unwrap().contains(key.value()));
    }
}

/// The text after `label` up to the end of its sentence.
fn sentence<'a>(text: &'a str, label: &str) -> &'a str {
    text.split_once(label)
        .map(|(_, rest)| {
            rest.split(". ")
                .next()
                .unwrap_or(rest)
                .trim_end_matches('.')
        })
        .unwrap_or("")
}
