//! The provider forwarder in a real Daytona sandbox, with a real provider key.
//!
//! Runs only with `BENCHFLOW_LIVE_DAYTONA=1`. The Daytona key comes from
//! `DAYTONA_API_KEY` (or `~/.config/benchflow/daytona.env`), the provider key
//! from the Vercel AI Gateway's usual places (`AI_GATEWAY_API_KEY`, or
//! `~/.config/benchflow/vercel-ai-gateway.env`). Neither is printed.
//!
//! One sandbox (ubuntu:24.04, with BenchFlow's Node.js installed at runtime)
//! for a few minutes, and one model request of a few tokens.
//!
//!     BENCHFLOW_LIVE_DAYTONA=1 cargo test -p benchflow-agent --test forwarder_live -- --nocapture
//!
//! It proves, from inside the sandbox and as the agent user, while the
//! forwarder runs and a stand-in agent process holds the exact environment
//! BenchFlow gives the agent:
//!
//! - a request through the forwarder reaches the provider and is answered,
//!   pinned to the allowed upstreams: the gateway's routing metadata says the
//!   provider set was restricted to the forwarder's `only` list (no Baseten),
//!   and names a non-Baseten upstream as the one that served it;
//! - the key is in no environment, no readable file, and no process command
//!   line or environment (`/proc/*/cmdline`, `/proc/*/environ` where
//!   readable), found by hashing every window that starts with the key's
//!   public format prefix (the key itself never enters the sandbox for the
//!   check);
//! - the forwarder's folder is closed to the agent user, and its key file is
//!   gone once read.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::{Duration, Instant};

use benchflow_agent::Registry;
use benchflow_agent::credentials::HostEnv;
use benchflow_agent::providers::{
    Forwarder, ProviderRegistry, Route, Upstreams, find_provider_key, node_layer, pin_route, route,
};
use benchflow_sandbox::daytona::{DaytonaConfig, DaytonaProvider};
use benchflow_sandbox::{
    Exec, ImageSource, ImageSpec, NetworkPolicy, Provider, Sandbox, SandboxSpec,
};
use serde_json::Value;
use sha2::{Digest, Sha256};

const MODEL: &str = "vercel/deepseek/deepseek-v4-flash-0731";
const SCANNER: &str = include_str!("leak_scan.js");

fn config() -> Option<DaytonaConfig> {
    if std::env::var("BENCHFLOW_LIVE_DAYTONA").ok().as_deref() != Some("1") {
        eprintln!("skipped: set BENCHFLOW_LIVE_DAYTONA=1 to run the Daytona live tests");
        return None;
    }
    let mut config = DaytonaConfig::from_env();
    if config.api_key.is_none() {
        let home = std::env::var("HOME").expect("HOME is set");
        let path = Path::new(&home).join(".config/benchflow/daytona.env");
        let text = std::fs::read_to_string(&path).expect("daytona.env");
        config.api_key = text
            .lines()
            .find_map(|l| l.trim().strip_prefix("DAYTONA_API_KEY="))
            .map(|v| v.trim().trim_matches(['"', '\'']).to_string());
    }
    Some(config)
}

async fn run(sb: &dyn Sandbox, exec: Exec) -> (i32, String) {
    let shown = exec.command.chars().take(80).collect::<String>();
    let out = sb
        .exec(exec)
        .await
        .unwrap_or_else(|e| panic!("`{shown}`: {e}"));
    (out.exit_code, out.combined())
}

#[tokio::test]
async fn the_forwarder_holds_the_key_and_the_agent_user_cannot_find_it() {
    let Some(config) = config() else { return };
    let host = HostEnv::from_process();
    let providers = ProviderRegistry::builtin();
    let claude = Registry::builtin().find("claude").unwrap().clone();
    let Route::Provider(mut route) = route(&claude, Some(MODEL), &providers).unwrap() else {
        panic!("{MODEL} is a provider route")
    };
    // As the engine does before any sandbox: pin to the allowed upstreams.
    let pin = pin_route(&mut route, &host)
        .expect("the model's upstreams are listed")
        .expect("vercel is a gateway");
    assert!(pin.excluded.contains(&"baseten".to_string()), "{pin:?}");
    let Upstreams::Only(allowed) = pin.upstreams.clone() else {
        panic!("the gateway is pinned with an allowlist: {pin:?}")
    };
    eprintln!(
        "pinned to {} upstreams: {}",
        allowed.len(),
        allowed.join(", ")
    );
    let key = find_provider_key(&route.provider, &host).expect("a Vercel AI Gateway key");
    let target = hex::encode(Sha256::digest(key.value().as_bytes()));
    let prefix = "vck_";
    assert!(
        key.value().starts_with(prefix),
        "the scan looks for the documented `vck_` key format"
    );

    let daytona = DaytonaProvider::new(config).unwrap();
    let image = daytona
        .prepare(
            &ImageSpec {
                source: ImageSource::Pull {
                    reference: "ubuntu:24.04".into(),
                },
                layers: Vec::new(),
                name: "live-forwarder".into(),
                digest: "livetest".into(),
                build_timeout: Some(Duration::from_secs(600)),
                platform: None,
            },
            &|line: &str| eprintln!("  build | {line}"),
        )
        .await
        .expect("prepare");
    let spec = SandboxSpec {
        image,
        name: "live-forwarder".into(),
        cpus: Some(1.0),
        memory_mb: Some(1024),
        disk_mb: Some(3 * 1024),
        gpus: 0,
        pids_limit: None,
        network: NetworkPolicy::Open,
        env: BTreeMap::new(),
        workdir: Some("/app".into()),
        labels: BTreeMap::from([("ai.benchflow.livetest".to_string(), "1".to_string())]),
        runtime: None,
    };
    let sandbox = daytona.create(&spec).await.expect("create the sandbox");
    let started = Instant::now();
    let outcome = check(&*sandbox, &route, &key, &target, prefix, &pin).await;
    let destroyed = sandbox.destroy().await;
    eprintln!("sandbox time: {:.0}s", started.elapsed().as_secs_f64());
    outcome.unwrap_or_else(|e| panic!("{e}"));
    destroyed.expect("destroy the sandbox");
}

async fn check(
    sb: &dyn Sandbox,
    route: &benchflow_agent::providers::ProviderRoute,
    key: &benchflow_agent::providers::ProviderKey,
    target: &str,
    prefix: &str,
    pin: &benchflow_agent::providers::Pin,
) -> Result<(), String> {
    // BenchFlow's Node.js (npm agents bring it in their image layer) and the
    // agent user, as a trial has them.
    let t = Instant::now();
    let (code, out) = run(
        sb,
        Exec::new(node_layer())
            .root()
            .timeout(Duration::from_secs(600)),
    )
    .await;
    if code != 0 {
        return Err(format!("installing Node.js failed: {out}"));
    }
    eprintln!("LATENCY node install: {:.0}s", t.elapsed().as_secs_f64());
    let (code, out) = run(
        sb,
        Exec::new("id agent >/dev/null 2>&1 || useradd -m -s /bin/bash agent; mkdir -p /app && chown agent /app").root(),
    )
    .await;
    if code != 0 {
        return Err(format!("creating the agent user failed: {out}"));
    }

    // The forwarder, as the engine starts it.
    let t = Instant::now();
    let forwarder = Forwarder::start(sb, route, key)
        .await
        .map_err(|e| format!("the forwarder did not start: {e}"))?;
    eprintln!(
        "LATENCY forwarder start: {:.1}s (port {})",
        t.elapsed().as_secs_f64(),
        forwarder.port()
    );

    // A stand-in agent with exactly the environment the engine gives it.
    let mut env = route
        .agent_env(&forwarder.base_url(), None)
        .map_err(|e| e.to_string())?;
    env.insert("HOME".into(), "/home/agent".into());
    let stand_in = sb
        .spawn(
            Exec::new("exec sleep 900")
                .user("agent")
                .cwd("/app")
                .envs(env.clone()),
        )
        .await
        .map_err(|e| format!("the stand-in agent did not start: {e}"))?;

    // A request through the forwarder, as the agent would send it.
    // The agent's own body asks for Baseten; the forwarder must override it.
    let request = r#"fetch(process.env.ANTHROPIC_BASE_URL + "/v1/messages", {method: "POST", headers: {"content-type": "application/json", "anthropic-version": "2023-06-01", "authorization": "Bearer " + process.env.ANTHROPIC_AUTH_TOKEN}, body: JSON.stringify({model: process.env.ANTHROPIC_MODEL, max_tokens: 8, messages: [{role: "user", content: "Say hi"}], providerOptions: {gateway: {only: ["baseten"]}}})}).then(async (r) => { console.log(r.status); const j = await r.json(); const g = (j.provider_metadata || {}).gateway || {}; console.log(JSON.stringify(g.routing || {})); })"#;
    sb.write_file("/tmp/bf-request.js", request.as_bytes(), 0o644)
        .await
        .map_err(|e| e.to_string())?;
    let (_, answer) = run(
        sb,
        Exec::new("/opt/benchflow/node/bin/node /tmp/bf-request.js")
            .user("agent")
            .cwd("/app")
            .envs(env.clone())
            .timeout(Duration::from_secs(120)),
    )
    .await;
    eprintln!(
        "request through the forwarder: {}",
        answer.lines().next().unwrap_or("")
    );
    if !answer.starts_with("200") {
        return Err(format!(
            "the request through the forwarder failed: {answer}"
        ));
    }
    let routing: Value = serde_json::from_str(answer.lines().nth(1).unwrap_or("{}"))
        .map_err(|e| format!("no routing metadata in {answer}: {e}"))?;
    let planning = routing["planningReasoning"].as_str().unwrap_or("");
    let restricted = planning
        .split_once("Provider set restricted to: ")
        .map(|(_, rest)| {
            rest.split(". ")
                .next()
                .unwrap_or(rest)
                .trim_end_matches('.')
        })
        .unwrap_or("");
    let order = planning
        .split_once("Total execution order: ")
        .map(|(_, rest)| rest.split(". ").next().unwrap_or(rest))
        .unwrap_or("");
    let served_by = routing["finalProvider"].as_str().unwrap_or("");
    eprintln!(
        "GATEWAY restricted to [{restricted}]; execution order {order}; served by {served_by}"
    );
    let restricted: Vec<String> = restricted.split(", ").map(str::to_string).collect();
    if Upstreams::Only(restricted.clone()) != pin.upstreams
        || order.is_empty()
        || order.contains("baseten")
        || served_by.is_empty()
        || served_by == "baseten"
    {
        return Err(format!("the gateway did not keep to the pin: {routing}"));
    }

    // The scan, as the agent user, while both processes run.
    sb.write_file("/tmp/bf-leak-scan.js", SCANNER.as_bytes(), 0o644)
        .await
        .map_err(|e| e.to_string())?;
    let t = Instant::now();
    let (code, scan) = run(
        sb,
        Exec::new(format!(
            "/opt/benchflow/node/bin/node /tmp/bf-leak-scan.js {target} {} {prefix}",
            key.value().len()
        ))
        .user("agent")
        .cwd("/app")
        .envs(env.clone())
        .timeout(Duration::from_secs(900)),
    )
    .await;
    eprintln!("LATENCY scan: {:.0}s", t.elapsed().as_secs_f64());
    if code != 0 {
        return Err(format!("the scan failed: {scan}"));
    }
    let report: Value = serde_json::from_str(scan.lines().last().unwrap_or(""))
        .map_err(|e| format!("the scan printed {scan}: {e}"))?;
    eprintln!("SCAN {report}");

    // As root: the key file is gone, and the forwarder's environment and
    // command line do not hold the key either.
    let (_, root_view) = run(
        sb,
        Exec::new(format!(
            "test -e /opt/benchflow/forwarder/key && echo KEYFILE-PRESENT || echo KEYFILE-GONE; ls -ld /opt/benchflow/forwarder | cut -c1-10; for pid in $(pgrep -x node); do printf 'node %s: ' \"$pid\"; tr '\\0' ' ' < /proc/$pid/cmdline; echo; done; /opt/benchflow/node/bin/node -e 'const c=require(\"crypto\"),f=require(\"fs\");let n=0,checked=0;for(const pid of process.argv.slice(1)){{let b;try{{b=Buffer.concat([f.readFileSync(\"/proc/\"+pid+\"/environ\"),f.readFileSync(\"/proc/\"+pid+\"/cmdline\")]);}}catch(e){{continue;}}checked++;for(let i=b.indexOf(\"{prefix}\");i!==-1;i=b.indexOf(\"{prefix}\",i+1)){{if(c.createHash(\"sha256\").update(b.subarray(i,i+{len})).digest(\"hex\")===\"{target}\")n++;}}}}console.log(\"forwarder processes checked: \"+checked+\"; environ+cmdline key hits: \"+n)' $(pgrep -f forwarder.js)",

            len = key.value().len()
        ))
        .root(),
    )
    .await;
    eprintln!("ROOT {}", root_view.replace('\n', " | "));

    let report_log = forwarder.stop().await;
    eprintln!(
        "forwarder answered {} request(s); log: {}",
        report_log.requests,
        report_log.log.trim()
    );
    drop(stand_in);

    let mut failures = Vec::new();
    if report["uid"] == 0 {
        failures.push("the scan ran as root, not as the agent user".to_string());
    }
    if report["hits"].as_array().is_none_or(|h| !h.is_empty()) {
        failures.push(format!("the key was found: {}", report["hits"]));
    }
    if report["forwarder_dir"] != "EACCES" {
        failures.push(format!(
            "the forwarder's folder was {} to the agent user",
            report["forwarder_dir"]
        ));
    }
    if report["files"].as_u64().unwrap_or(0) < 1000 {
        failures.push(format!("the scan read only {} files", report["files"]));
    }
    if report["proc_environ_read"].as_u64().unwrap_or(0) == 0 {
        failures.push("no process environment was readable (the stand-in's should be)".into());
    }
    if !root_view.contains("KEYFILE-GONE") {
        failures.push(format!("the key file is still there: {root_view}"));
    }
    if !root_view.contains("drwx------") {
        failures.push(format!("the forwarder's folder is not 700: {root_view}"));
    }
    if !root_view.contains("environ+cmdline key hits: 0")
        || root_view.contains("forwarder processes checked: 0;")
        || !root_view
            .contains("/opt/benchflow/forwarder/forwarder.js /opt/benchflow/forwarder/config.json")
    {
        failures.push(format!(
            "the forwarder's own environment or command line holds the key: {root_view}"
        ));
    }
    if report_log.log.contains(key.value()) || answer.contains(key.value()) {
        failures.push("the key is in the forwarder's log or the answer".into());
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures.join("\n"))
    }
}
