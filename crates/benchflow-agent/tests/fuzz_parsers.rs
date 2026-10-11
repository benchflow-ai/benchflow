//! Property tests over the agent crate's readers of outside input: every
//! headless output format (an agent's stdout, line by line), ACP
//! `session/update`s merged into a trajectory, acp_trajectory.jsonl and its
//! exports (ATIF, ADP, Verifiers), agent and provider manifests, the ACP
//! registry index, model replies (OpenAI, Anthropic, Terminus-2 JSON and
//! XML, computer use), the external backend's messages, and `--ak`
//! assignments. None may panic or take too long on any input.

#[path = "../../benchflow-task/tests/fuzz_support/mod.rs"]
mod fuzz_support;

use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use benchflow_agent::headless::{Out, OutputFormat};
use benchflow_agent::manifest::{AgentManifest, ManifestSource};
use benchflow_agent::native::terminus::{json_parser, xml_parser};
use benchflow_agent::native::{anthropic, computer, openai};
use benchflow_agent::trajectory::{
    AdpMeta, AtifMeta, Receipt, Redactor, Trajectory, VerifiersMeta, parse_jsonl, summarize,
    to_adp_record, to_atif, to_jsonl, to_verifiers_record,
};
use fuzz_support::*;
use proptest::prelude::*;
use serde_json::Value;

fn here() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

static WORDS: LazyLock<&'static [&'static str]> =
    LazyLock::new(|| words(&here().join("tests/fuzz_seeds/stream-words.txt")));

/// Lines of agent output: real recordings and test lines, one per seed.
fn stream_lines() -> Vec<String> {
    let mut out: Vec<String> = seeds(
        &[
            "benchflow-agent/src/headless/parsers/testdata",
            "benchflow-agent/runners/tests/fixtures",
            "benchflow-agent/tests/fuzz_seeds",
        ],
        &|n| n.ends_with(".jsonl") || n.ends_with(".json"),
        100,
    )
    .iter()
    .flat_map(|t| t.lines().map(str::to_string).collect::<Vec<_>>())
    .filter(|l| !l.is_empty() && l.len() < 64 * 1024)
    .collect();
    out.truncate(4000);
    out
}

static LINES: LazyLock<Vec<String>> = LazyLock::new(stream_lines);

/// A line of output: an edited real line, random JSON in the formats'
/// vocabulary, or free text.
fn line() -> impl Strategy<Value = String> {
    prop_oneof![
        4 => mutated(LINES.clone()),
        4 => json_with(*WORDS).prop_map(|v| v.to_string()),
        1 => "\\PC{0,80}",
    ]
}

fn acp_trajectories() -> Vec<String> {
    seeds(
        &[
            "benchflow-viewer/tests/fixtures",
            "benchflow-run/tests/fixtures",
        ],
        &|n| n == "acp_trajectory.jsonl",
        60,
    )
}

fn agent_tomls() -> Vec<String> {
    seeds(&["benchflow-agent/agents"], &|n| n.ends_with(".toml"), 100)
}

fn provider_tomls() -> Vec<String> {
    seeds(
        &["benchflow-agent/providers"],
        &|n| n.ends_with(".toml"),
        100,
    )
}

proptest! {
    #![proptest_config(config())]

    /// Every headless format reads any run of lines, then finishes.
    #[test]
    fn headless_formats(
        format in proptest::sample::select(OutputFormat::ALL.to_vec()),
        lines in prop::collection::vec(line(), 0..24),
    ) {
        timed(format.name(), || {
            let mut parser = format.parser();
            let mut traj = Trajectory::new();
            let mut out = Vec::new();
            for l in &lines {
                parser.line(l, &mut out);
            }
            parser.finish(&mut out);
            for o in &out {
                if let Out::Update(u) = o {
                    traj.apply_update(u, Receipt { first_observed_ns: 1, last_observed_ns: 2 });
                }
            }
            let events: Vec<_> = traj.events().map(|e| e.into_owned()).collect();
            let _ = summarize(events.iter(), false, Some("acp"));
            let _ = to_atif(&events, &AtifMeta::default());
        });
    }

    /// ACP `session/update`s in any order and shape merge into a trajectory
    /// that exports.
    #[test]
    fn acp_updates(updates in prop::collection::vec(json_with(*WORDS), 0..32)) {
        timed("apply_update", || {
            let mut traj = Trajectory::new();
            for u in &updates {
                traj.apply_update(u, Receipt { first_observed_ns: 1, last_observed_ns: 2 });
            }
            let events: Vec<_> = traj.events().map(|e| e.into_owned()).collect();
            let json: Vec<Value> = events.iter().map(|e| e.to_json()).collect();
            let _ = to_atif(&events, &AtifMeta::default());
            let _ = to_adp_record(&json, &AdpMeta::default());
            let _ = to_verifiers_record(&json, &VerifiersMeta::default());
            let _ = to_jsonl(events.iter(), &Redactor::new(["sk-secret"]));
        });
    }

    /// acp_trajectory.jsonl as written by 0.8 and by `bench`, edited, read
    /// back and exported.
    #[test]
    fn trajectory_jsonl(text in mutated(acp_trajectories())) {
        timed("parse_jsonl", || {
            if let Ok(events) = parse_jsonl(&text) {
                let json: Vec<Value> = events.iter().map(|e| e.to_json()).collect();
                let _ = summarize(events.iter(), true, None);
                let _ = to_atif(&events, &AtifMeta::default());
                let _ = to_adp_record(&json, &AdpMeta::default());
                let _ = to_verifiers_record(&json, &VerifiersMeta::default());
            }
        });
    }

    /// Agent manifests (a project may add its own).
    #[test]
    fn agent_manifest(text in mutated(agent_tomls())) {
        timed("AgentManifest::parse", || {
            let _ = AgentManifest::parse(&text, ManifestSource::File(PathBuf::from("agent.toml")));
        });
    }

    /// Provider manifests.
    #[test]
    fn provider_manifest(text in mutated(provider_tomls())) {
        timed("Provider::parse", || {
            let _ = benchflow_agent::providers::Provider::parse(&text, ManifestSource::File(PathBuf::from("p.toml")));
        });
    }

    /// The ACP registry's index and `acp:<id>@<version>` specs.
    #[test]
    fn acp_registry(
        doc in json_with(&["version", "agents", "id", "name", "version", "description", "distribution", "npx", "uvx", "binary", "package", "args", "env", "cmd", "archive", "url", "sha256", "linux-x86_64", "darwin-aarch64"]),
        spec in prop_oneof!["acp:\\PC{0,20}", "\\PC{0,20}", mutated(vec!["acp:goose@1.54.0".into(), "acp:claude-code".into()])],
    ) {
        timed("acp registry", || {
            let _ = benchflow_agent::acp_registry::Index::parse(&doc.to_string());
            let _ = benchflow_agent::acp_registry::parse_spec(&spec);
        });
    }

    /// Model replies to the native loops.
    #[test]
    fn model_replies(
        body in json_with(&["choices", "message", "content", "role", "tool_calls", "id", "type", "function", "name", "arguments", "usage", "prompt_tokens", "completion_tokens", "total_tokens", "finish_reason", "stop_reason", "text", "tool_use", "input", "input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens", "thinking", "reasoning_content", "error", "prompt_tokens_details", "cached_tokens"]),
        text in prop_oneof![
            "\\PC{0,200}",
            mutated(vec![
                r#"{"analysis": "a", "plan": "b", "commands": [{"keystrokes": "ls\n", "duration": 0.1}], "task_complete": false}"#.into(),
                "<response><analysis>a</analysis><plan>b</plan><commands><keystrokes duration=\"0.1\">ls\n</keystrokes></commands><task_complete>true</task_complete></response>".into(),
                "```json\n{\"action\": \"click\", \"x\": 10, \"y\": 20}\n```".into(),
                "Action: type(\"hello\")".into(),
            ]),
        ],
    ) {
        timed("model replies", || {
            let _ = openai::parse_response(&body);
            let _ = anthropic::parse_response(&body);
            let _ = json_parser::parse_response(&text);
            let _ = xml_parser::parse_response(&text);
            let _ = computer::parse(&text);
        });
    }

    /// The external backend's messages and `--ak key=value` assignments.
    #[test]
    fn backend_and_assignments(
        text in prop_oneof![json_with(&["type", "action", "command", "keystrokes", "message", "done", "reward", "error", "content", "timeout", "id"]).prop_map(|v| v.to_string()), "\\PC{0,60}"],
    ) {
        timed("backend", || {
            let _ = benchflow_agent::external::backend::parse_message(&text);
            let _ = benchflow_agent::manifest::options::parse_assignment(&text);
        });
    }
}
