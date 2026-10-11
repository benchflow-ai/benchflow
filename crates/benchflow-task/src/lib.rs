//! Task packages for BenchFlow.
//!
//! A task is a folder: what the agent is asked to do, the sandbox it works
//! in, the verifier that scores it, and a reference solution. This crate
//! reads packages in every format BenchFlow runs, checks them, and scores
//! verifier output.
//!
//! - [`load`] reads a package into a normalized [`TaskPackage`].
//! - [`check`] validates a package and returns diagnostics.
//! - [`discover`] finds packages under a folder.

pub mod check;
pub mod codes;
pub(crate) mod compose;
pub mod convert;
pub mod ctrf;
pub mod diag;
pub mod digest;
pub mod extras;
pub mod fnmatch;
pub mod load;
pub mod package;
pub mod reward;
pub mod rubric;
pub mod scaffold;
pub mod source;
pub mod split;
pub mod suggest;
pub mod taskmd;
pub mod v06;
pub mod values;

pub use check::{CheckOptions, CheckReport};
pub use diag::{Diagnostic, Severity};
pub use load::{check, check_schema, detect, discover, keep_schema_only, load};
pub use package::{Format, TaskPackage};
