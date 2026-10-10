//! `bench view`: a local web page for a job's runs and trajectories.
//!
//! This is BenchFlow 0.8's trajectory viewer in Rust. The page and its
//! scripts are 0.8's, embedded unchanged; this crate reads a job folder
//! (0.8's layout) and serves the same routes and the same JSON, byte for
//! byte, so the page behaves exactly as it did.
//!
//! - [`serve`] starts the server on a trial folder, a job folder, or a
//!   folder of jobs and returns a [`ServerHandle`]; [`ServerHandle::wait`]
//!   serves until Ctrl-C.
//! - [`render_static`] writes one trial as a self-contained HTML file that
//!   opens from disk (0.8's `trajectory.html`).
//! - [`detect`] says what a path is without serving it.
//! - [`serve_jobs`] serves several job folders side by side and opens on
//!   the compare view.
//!
//! Beside 0.8's pages, the server has two views of its own: `/compare`
//! (tasks by job, pass rates with Wilson intervals, paired differences) and
//! `/outcomes` (how trials ended, problems by code and owner, time per
//! phase). The run list links to them; 0.8's pages are otherwise unchanged.
//!
//! Everything in a job folder is untrusted input: a missing, malformed, or
//! hostile file leaves its part of the page empty and never breaks the
//! page, and trajectory text is always shown as text.

mod catalog;
mod classify;
mod lineage;
mod metadata;
mod page;
mod payload;
mod py;
mod pyjson;
mod rubric;
mod scoring;
mod server;
mod stats;
mod steps;
mod subagents;
mod target;
mod timestamp;
mod values;
mod verifier;
mod views;

#[cfg(test)]
mod corpus_tests;

pub use page::render_static;
pub use server::{ServeOptions, ServerHandle, app, serve, serve_jobs};
pub use target::{Target, ViewError, detect, detect_jobs};
