//! The diagnostic codes task.md checks emit.
//!
//! `T1xx` is the document's structure, `T2xx` config keys and values, `T3xx`
//! rubrics, behaviors, and judges, `T4xx` the package's other files, and
//! `T9xx` notes. A code is stable: `bench explain <code>` documents it, and
//! tools may match on it. [`super::EXPLANATIONS`] holds each code's
//! explanation.

// T1xx: the document.

/// Text after the first task.md block.
pub const TEXT_AFTER_BLOCKS: &str = "T101";
/// A fenced block is never closed.
pub const UNTERMINATED_FENCE: &str = "T102";
/// A ` ```rubric ` or ` ```behaviors ` fence, which moved to `verifier/`.
pub const RESERVED_FENCE: &str = "T103";
/// An ordinary fence after the first task.md block.
pub const FENCE_AFTER_BLOCKS: &str = "T104";
/// The instruction is empty.
pub const EMPTY_INSTRUCTION: &str = "T105";
/// The same block twice.
pub const DUPLICATE_BLOCK: &str = "T106";
/// The config block does not parse, or is not a table.
pub const CONFIG_SYNTAX: &str = "T107";
/// Harbor's `task.toml` beside task.md.
pub const HARBOR_TASK_TOML: &str = "T108";
/// Legacy YAML frontmatter.
pub const FRONTMATTER: &str = "T109";
/// Config in more than one place.
pub const CONFIG_TWICE: &str = "T110";
/// A canary built on a public GUID.
pub const SHARED_CANARY: &str = "T111";
/// task.md is missing, unreadable, or not UTF-8.
pub const UNREADABLE: &str = "T112";

// T2xx: config keys and values.

/// A key its table does not define.
pub const UNKNOWN_KEY: &str = "T201";
/// Harbor's spelling of a task.md key.
pub const HARBOR_KEY: &str = "T202";
/// A key or value earlier drafts used, now gone or moved.
pub const KEY_GONE: &str = "T203";
/// A table, list, or value of the wrong shape.
pub const WRONG_SHAPE: &str = "T204";
/// Not a duration.
pub const BAD_DURATION: &str = "T205";
/// Not a size.
pub const BAD_SIZE: &str = "T206";
/// Not a network.
pub const BAD_NETWORK: &str = "T207";
/// Not one of the allowed values.
pub const BAD_CHOICE: &str = "T208";
/// Not a whole number or a count.
pub const BAD_NUMBER: &str = "T209";
/// A required key is missing.
pub const MISSING_KEY: &str = "T210";
/// A sandbox path that is relative or climbs out with `..`.
pub const BAD_PATH: &str = "T211";
/// Stage rules: unlock, at_start, and stage blocks.
pub const STAGES: &str = "T212";
/// Service rules: names.
pub const SERVICES: &str = "T213";
/// Clock settings.
pub const CLOCK: &str = "T214";
/// Time zone settings, and a clock's offset in its zone.
pub const TIMEZONE: &str = "T215";
/// A set clock needs no network while it runs.
pub const CLOCK_NETWORK: &str = "T216";
/// The run may cross midnight.
pub const CLOCK_MIDNIGHT: &str = "T217";
/// Family split rules and `[family]` settings.
pub const FAMILY: &str = "T218";
/// `[training]` settings.
pub const TRAINING: &str = "T219";
/// `[integrity.controls] injection`.
pub const INJECTION: &str = "T220";
/// An env value resolved from the host (`${VAR}`).
pub const HOST_VAR: &str = "T221";
/// A Harbor feature an import recorded, which task.md does not run.
pub const IMPORTED: &str = "T222";

// T3xx: rubrics, behaviors, judges.

/// `rubric.json` or `behaviors.json` is not a JSON object.
pub const JUDGMENT_FILE: &str = "T301";
/// A JSON file without task.md's `$schema`.
pub const NO_SCHEMA: &str = "T302";
/// A key a rubric, criterion, or behavior does not define.
pub const RUBRIC_KEY: &str = "T303";
/// A rubric's scoring settings.
pub const SCORING: &str = "T304";
/// A rubric with no criteria.
pub const NO_CRITERIA: &str = "T305";
/// Criterion ids: their form, and duplicates.
pub const CRITERION_ID: &str = "T306";
/// Removal entries of a shared rubric.
pub const REMOVAL: &str = "T307";
/// A criterion's fields.
pub const CRITERION: &str = "T308";
/// `cite = "independent"`.
pub const CITE: &str = "T309";
/// Continuous scores, and `missing = "no-penalty"`.
pub const CONTINUOUS: &str = "T310";
/// Evidence a judge cannot read.
pub const EVIDENCE: &str = "T311";
/// `stated` quotes.
pub const STATED: &str = "T312";
/// A criterion that says neither where it was asked for nor why it need not
/// be.
pub const NOT_STATED: &str = "T313";
/// Watched and paired behaviors.
pub const BEHAVIOR: &str = "T320";
/// A behavior's consequence and severity map.
pub const CONSEQUENCE: &str = "T321";
/// How a behavior is detected, and who tags it.
pub const DETECT: &str = "T322";
/// `[verifier.judges]` settings.
pub const JUDGES: &str = "T330";
/// A judge's brief.
pub const JUDGE_BRIEF: &str = "T331";
/// An agent judge without a token budget.
pub const JUDGE_BUDGET: &str = "T332";
/// A verifier script that calls a model without `[verifier] models = true`.
pub const SCRIPT_MODELS: &str = "T333";

// T4xx: the package's files.

/// Nothing says what the agent runs in.
pub const NO_SANDBOX: &str = "T401";
/// Nothing grades the task.
pub const NO_GRADER: &str = "T402";
/// A control `[integrity.controls]` names.
pub const CONTROL_PATH: &str = "T403";
/// A scripted seat's comment says what it is.
pub const SEAT_COMMENT: &str = "T404";
/// A Compose setting that reaches the host.
pub const COMPOSE_HOST: &str = "T405";
/// A family's generator file.
pub const GENERATOR: &str = "T406";

// T9xx: notes.

/// A family's placeholders were not verified, since checking runs no
/// package code.
pub const GENERATOR_NOT_RUN: &str = "T901";
