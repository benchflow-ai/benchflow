//! Trajectories: what an agent did during a run.
//!
//! - [`Trajectory`] records a session as it happens: the ACP client feeds it
//!   every `session/update`, and it merges them into events the way 0.8 did
//!   (see [`record`]).
//! - [`Event`] is one line of `trajectory/acp_trajectory.jsonl`: a user or
//!   agent message, a thought, a tool call with its updates merged in, a
//!   notice, BenchFlow's timeout marker, or a scripted agent's record.
//! - [`to_jsonl`] writes events exactly as 0.8 does (Python `json.dumps`
//!   formatting, every string redacted), [`parse_jsonl`] reads them back, and
//!   [`TrajectoryWriter`] rewrites the file atomically as a run goes.
//! - [`summarize`], [`count_tool_calls`] and [`count_skill_invocations`]
//!   give the counts `result.json` carries.
//! - [`to_atif`] exports ATIF v1.8 for Harbor's tools and trainers;
//!   [`to_verifiers_record`] and [`to_adp_record`] export 0.8's Verifiers /
//!   ORS and ADP trainer records, byte for byte as 0.8 wrote them.
//! - [`Redactor`] removes the login's secrets and 0.8's secret patterns from
//!   everything written.
//!
//! A 0.8 file read with [`parse_jsonl`] and written with [`to_jsonl`] comes
//! out byte for byte the same.

mod adp;
mod atif;
mod event;
mod export_common;
mod jsonl;
pub mod record;
mod redact;
mod skill;
mod summary;
mod verifiers;

pub use adp::{ADP_SCHEMA_VERSION, AdpMeta, adp_line, to_adp_record};
pub use atif::{ATIF_SCHEMA_VERSION, AtifMeta, MAX_SUBAGENT_DEPTH, to_atif};
pub use event::{
    AgentTimeout, Event, NOP_NOTE, Notice, Oracle, Receipt, Text, ToolCall, ToolCallStatus,
    iso_from_ns,
};
pub use export_common::{acp_projection_coverage, to_python_json_pretty};
pub use jsonl::{
    JsonlError, PyFormatter, TrajectoryWriter, parse_jsonl, python_float_repr, to_jsonl,
    to_python_json,
};
pub use record::{Activity, Change, ContextUsage, RawUpdate, Trajectory, Usage};
pub use redact::{REDACTED, Redactor};
pub use summary::{TrajectorySummary, count_skill_invocations, count_tool_calls, summarize};
pub use verifiers::{VerifiersMeta, to_verifiers_record, verifiers_line};

#[cfg(test)]
mod tests;
