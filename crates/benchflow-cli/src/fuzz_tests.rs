//! Property tests over the command line's readers of outside input: time
//! values (flags, bench.toml, fetched job files), `--env` and `--ak` pairs,
//! and the session files `bench handoff` takes from a trial folder that may
//! be someone else's. None may panic; an accepted duration is finite and
//! within the longest limit; an id that passes `safe_id` is plain.

#[path = "../../benchflow-task/tests/fuzz_support/mod.rs"]
mod fuzz_support;

use fuzz_support::*;
use proptest::prelude::*;

use crate::cmd::handoff;
use crate::context;

proptest! {
    #![proptest_config(config())]

    #[test]
    fn durations(text in prop_oneof![
        "-?[0-9]{0,30}(\\.[0-9]{0,20})?(e-?[0-9]{1,3})?",
        "([0-9]{1,24}[hms]){0,3}",
        proptest::sample::select(TOKENS).prop_map(str::to_string),
        "\\PC{0,16}",
    ]) {
        timed("parse_duration", move || {
            if let Ok(d) = context::parse_duration(&text) {
                prop_assert!(d <= benchflow_task::values::LONGEST, "{text:?} gave {d:?}");
            }
            Ok(())
        })?;
    }

    #[test]
    fn pairs(pairs in prop::collection::vec(prop_oneof!["\\PC{0,20}", "[a-z-_]{0,8}=\\PC{0,12}", " *[a-z-]{1,8} *=.{0,8}"], 0..6)) {
        timed("pairs", move || {
            let _ = context::parse_env(&pairs);
            let _ = context::parse_agent_options(&pairs);
        });
    }

    /// A trial's saved Codex rollout: the id from its first line, else its
    /// file name.
    #[test]
    fn handoff_ids(
        first in prop_oneof![
            json_with(&["payload", "id", "type", "session_meta", "timestamp"]).prop_map(|v| v.to_string()),
            "\\PC{0,60}",
        ],
        name in prop_oneof![
            "rollout-2026-10-10T12-00-00-[0-9a-f-]{30,40}\\.jsonl",
            "\\PC{0,50}",
        ],
    ) {
        let tmp = tempfile::tempdir().unwrap();
        let name = name.replace(['/', '\0'], "_");
        let file = tmp.path().join(if name.is_empty() || name == "." || name == ".." { "x".into() } else { name });
        let _ = std::fs::write(&file, format!("{first}\n"));
        timed("handoff ids", move || {
            if let Some(id) = handoff::codex_id(&file)
                && handoff::safe_id(&id)
            {
                prop_assert!(id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'));
                prop_assert!(!id.starts_with('-'));
            }
            let _ = handoff::claude_slug(&file);
            Ok(())
        })?;
    }
}
