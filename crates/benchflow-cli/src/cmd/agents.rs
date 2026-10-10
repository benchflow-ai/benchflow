//! `bench agents`, and resolving `--agent` to a model agent.
//!
//! Agents are data: the built-in manifests, then any `[agents.<name>]`
//! tables in the project's bench.toml (a table named like a built-in agent
//! replaces it, to pin another version), and any entry of the public ACP
//! registry as `acp:<id>[@<version>]`.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use benchflow_agent::acp_registry::{self, RegistryError, Source};
use benchflow_agent::credentials::{HostEnv, credential_kinds, login_status};
use benchflow_agent::manifest::{CredentialFamily, Harness, InstallMethod, ManifestSource};
use benchflow_agent::providers::{
    CheckOutcome, ProviderRegistry, check_key, find_provider_key, key_handling,
};
use benchflow_agent::{AgentManifest, Registry, UnknownAgent};
use benchflow_run::Owner;
use benchflow_run::acp_agent::AcpAgent;
use benchflow_run::agent::Attempt;
use benchflow_run::agent_check::{self, CheckRecord, CheckSpec};
use benchflow_run::config::default_cache_dir;
use benchflow_run::external_agent::ExternalAgent;
use benchflow_run::headless_agent::HeadlessAgent;
use benchflow_run::native_agent::NativeAgent;

use crate::cli::{AgentsArgs, AgentsCheckArgs, AgentsCommand};
use crate::context;
use crate::error::{CliError, Exit, Result};
use crate::ui::{
    bold, dim, fail_mark, ok_mark, table::Table, truncate, visible_width, warn_mark, wrap,
    wrap_width,
};
use crate::{note, say};

/// Every agent this project knows.
pub fn registry() -> Result<Registry> {
    let mut registry = Registry::builtin();
    let project = context::project()?;
    let source = project
        .file
        .clone()
        .unwrap_or_else(|| project.root.join("bench.toml"));
    for (name, table) in &project.config.agents {
        let mut table = table.clone();
        if let Some(t) = table.as_table_mut() {
            t.entry("name")
                .or_insert_with(|| toml::Value::String(name.clone()));
        }
        let text = toml::to_string(&table)
            .map_err(|e| CliError::bug("re-reading a bench.toml agent table", e))?;
        let manifest =
            AgentManifest::parse(&text, ManifestSource::File(source.clone())).map_err(|e| {
                CliError::usage(format!("bench.toml: agent `{name}` is invalid: {e}")).next(
                    "fix the [agents.<name>] table; `bench agents codex` shows a working manifest",
                )
            })?;
        registry.add(manifest).map_err(|e| {
            CliError::usage(format!("bench.toml: agent `{name}`: {e}"))
                .next("rename the agent or its aliases")
        })?;
    }
    Ok(registry)
}

/// The manifest `name` refers to: an agent by name or alias, or an entry of
/// the public ACP registry (`acp:<id>[@<version>]`).
pub async fn manifest(name: &str, registry: &Registry) -> Result<AgentManifest> {
    if acp_registry::is_spec(name) {
        return registry_manifest(name, registry).await;
    }
    registry.find(name).cloned().map_err(|u| unknown_agent(&u))
}

fn unknown_agent(u: &UnknownAgent) -> CliError {
    let e = CliError::usage(format!("no agent `{}`", u.name));
    if let Some(s) = &u.suggestion {
        return e.next(format!(
            "did you mean `{s}`? `bench agents` lists every agent"
        ));
    }
    // The built-in copy of the registry needs no network.
    if acp_registry::Index::snapshot().find(&u.name).is_some() {
        return e.next(format!(
            "the public ACP registry has `{0}`: run it as `acp:{0}`; `bench agents` lists the built-in agents",
            u.name
        ));
    }
    e.next("`bench agents` lists every agent; `bench agents --registry` lists the ACP registry")
}

async fn registry_index() -> Result<(acp_registry::Loaded, Source)> {
    let source = Source::new(default_cache_dir());
    let loaded = acp_registry::load(&source).await.map_err(|e| {
        CliError::usage(e).next(format!(
            "unset {} or point it at a registry.json",
            acp_registry::FILE_ENV
        ))
    })?;
    if let Some(warning) = loaded.origin.warning() {
        note!("  {} {warning}", warn_mark());
    }
    Ok((loaded, source))
}

async fn registry_manifest(spec: &str, registry: &Registry) -> Result<AgentManifest> {
    let (loaded, source) = registry_index().await?;
    acp_registry::resolve(&loaded, spec, &source, registry)
        .await
        .map_err(|e| {
            let error = match &e {
                RegistryError::Refused(_) => CliError::failed(e.to_string()).owner(Owner::Agent),
                _ => CliError::usage(e.to_string()),
            };
            error.next(e.next_step())
        })
}

/// Resolve a model agent by name, alias, or `acp:<id>[@<version>]`.
pub async fn model_attempt(
    name: &str,
    model: Option<&str>,
    effort: Option<&str>,
    harness: Option<&str>,
    backend: Option<&str>,
    env: &BTreeMap<String, String>,
    options: &[(String, String)],
) -> Result<Attempt> {
    let host = HostEnv::from_process().with_overrides(env.clone());
    let refuse_options = |agent: &str| match options.first() {
        Some((key, _)) => Err(CliError::usage(format!(
            "agent `{agent}` takes no options, and --ak {key} was given"
        ))
        .next(format!(
            "drop --ak {key}; pass a variable the agent reads with --env instead"
        ))),
        None => Ok(()),
    };
    if benchflow_run::external_agent::is_external(name) {
        refuse_options("external")?;
        if let Some(h) = harness {
            return Err(CliError::usage(format!(
                "the external agent runs only on its backend, not the `{h}` harness"
            ))
            .next("drop --harness"));
        }
        return Ok(Attempt::Model(Arc::new(ExternalAgent::new(
            backend, model, effort, &host,
        ))));
    }
    if benchflow_run::dspy_agent::is_dspy_rlm(name) {
        refuse_options("dspy-rlm")?;
        if let Some(h) = harness.filter(|h| *h != "headless") {
            return Err(CliError::usage(format!(
                "dspy-rlm runs only on the headless harness, not `{h}`"
            ))
            .next("drop --harness"));
        }
        return Ok(Attempt::Model(Arc::new(
            benchflow_run::dspy_agent::DspyRlmAgent::new(model, effort, &host),
        )));
    }
    if let Some(agent) = NativeAgent::new(name, model, effort, &host) {
        if let Some(h) = harness.filter(|h| *h != "native") {
            return Err(CliError::usage(format!(
                "{} runs only on the native harness, not `{h}`",
                benchflow_agent::native::find(name).unwrap_or(name)
            ))
            .next("drop --harness, or pass --harness native"));
        }
        let agent = agent.with_options(options).map_err(option_error)?;
        return Ok(Attempt::Model(Arc::new(agent)));
    }
    let registry = registry()?;
    let manifest = manifest(name, &registry).await?;
    for note in &manifest.notes {
        note!("  {} {}", dim("note:"), note);
    }
    let requested = harness.map(|h| {
        Harness::parse(h).ok_or_else(|| {
            CliError::usage(format!("unknown harness `{h}`"))
                .next("use --harness acp or --harness headless")
        })
    });
    let harness = manifest
        .pick_harness(requested.transpose()?)
        .map_err(|message| {
            CliError::usage(message).next(format!(
                "drop --harness, or see `bench agents {}` for the harnesses it has",
                manifest.name
            ))
        })?;
    let values = manifest
        .resolve_options(options, harness)
        .map_err(option_error)?;
    let manifest = manifest.with_options(&values);
    Ok(match harness {
        Harness::Acp => Attempt::Model(Arc::new(AcpAgent::new(manifest, model, effort, &host))),
        Harness::Headless => {
            Attempt::Model(Arc::new(HeadlessAgent::new(manifest, model, effort, &host)))
        }
    })
}

/// A run's agent options that do not fit the agent: the user's to fix.
fn option_error(e: benchflow_agent::manifest::OptionError) -> CliError {
    CliError::usage(e.message).next(e.next)
}

/// One line per provider with a key on this machine, for `bench doctor`:
/// (name, verdict, summary, fix), the verdict from the provider's free check
/// request (`None` when it could not say); then the providers without a key.
pub async fn provider_lines() -> (
    Vec<(String, Option<bool>, String, Option<String>)>,
    Vec<String>,
) {
    let host = HostEnv::from_process();
    let mut lines = Vec::new();
    let mut without = Vec::new();
    for provider in ProviderRegistry::builtin().providers() {
        let Ok(key) = find_provider_key(provider, &host) else {
            without.push(provider.name.clone());
            continue;
        };
        let outcome = check_key(provider, &key, Duration::from_secs(10)).await;
        let verdict = match outcome {
            CheckOutcome::Accepted => Some(true),
            CheckOutcome::Rejected { .. } => Some(false),
            CheckOutcome::Inconclusive { .. } | CheckOutcome::NoCheck => None,
        };
        let fix = matches!(outcome, CheckOutcome::Rejected { .. })
            .then(|| format!("replace the key ({})", key.source));
        lines.push((
            format!("provider {}", provider.name),
            verdict,
            format!("{}; {}", key.source, outcome.describe()),
            fix,
        ));
    }
    (lines, without)
}

/// The first agent with a login on this machine, preferring the common ones.
pub fn first_ready_agent() -> Option<String> {
    let registry = registry().ok()?;
    let host = HostEnv::from_process();
    let ready: Vec<&AgentManifest> = registry
        .agents()
        .iter()
        .filter(|m| login_status(m, None, &host).ready())
        .collect();
    ["claude", "codex", "gemini"]
        .iter()
        .find_map(|name| ready.iter().find(|m| m.name == *name))
        .or_else(|| ready.first())
        .map(|m| m.name.clone())
}

/// One line per agent login for `bench doctor`: (name, ready, summary, fix).
pub fn login_lines() -> Vec<(String, bool, String, Option<String>)> {
    let Ok(registry) = registry() else {
        return Vec::new();
    };
    let host = HostEnv::from_process();
    registry
        .agents()
        .iter()
        .map(|m| {
            let status = login_status(m, None, &host);
            (
                format!("agent {}", m.name),
                status.ready(),
                status.summary(),
                status.fix.clone(),
            )
        })
        .collect()
}

pub async fn run(args: AgentsArgs) -> Result<Exit> {
    if let Some(AgentsCommand::Check(check)) = args.command {
        return run_checks(check).await;
    }
    if args.registry {
        return list_registry().await;
    }
    let registry = registry()?;
    match args.name {
        None => list(&registry),
        Some(name) if benchflow_run::dspy_agent::is_dspy_rlm(&name) => {
            show(benchflow_run::dspy_agent::manifest());
            Ok(Exit::Ok)
        }
        Some(name) if benchflow_run::external_agent::is_external(&name) => {
            show_external();
            Ok(Exit::Ok)
        }
        Some(name) if benchflow_agent::native::find(&name).is_some() => {
            show_native(&name);
            Ok(Exit::Ok)
        }
        Some(name) => {
            let manifest = manifest(&name, &registry).await?;
            show(&manifest);
            Ok(Exit::Ok)
        }
    }
}

fn list(registry: &Registry) -> Result<Exit> {
    let host = HostEnv::from_process();
    let checks = agent_check::load(&default_cache_dir());
    let mut rows = vec![
        AgentRow {
            name: "oracle".into(),
            aliases: Vec::new(),
            harness: "script".into(),
            login: dim("none needed"),
            checked: dim("-"),
            what: "runs the task's reference solution".into(),
        },
        AgentRow {
            name: "nop".into(),
            aliases: Vec::new(),
            harness: "script".into(),
            login: dim("none needed"),
            checked: dim("-"),
            what: "does nothing; the verifier scores the untouched workspace".into(),
        },
    ];
    for m in registry.agents() {
        let status = login_status(m, None, &host);
        let login = if status.ready() {
            format!("{} {}", ok_mark(), chosen_login(&status))
        } else {
            format!("{} no login", fail_mark())
        };
        rows.push(AgentRow {
            name: m.name.clone(),
            aliases: m.aliases.clone(),
            harness: harnesses(m),
            login,
            checked: checked_cell(&checks, m),
            what: m.description.clone(),
        });
    }
    for (name, aliases, what) in benchflow_agent::native::AGENTS {
        rows.push(AgentRow {
            name: name.to_string(),
            aliases: aliases.iter().map(|a| a.to_string()).collect(),
            harness: "native".into(),
            login: native_login(&host),
            checked: dim("-"),
            what: what.to_string(),
        });
    }
    let dspy = benchflow_run::dspy_agent::manifest();
    rows.push(AgentRow {
        name: dspy.name.clone(),
        aliases: dspy.aliases.clone(),
        harness: "headless".into(),
        login: native_login(&host),
        checked: dim("-"),
        what: dspy.description.clone(),
    });
    rows.push(AgentRow {
        name: benchflow_run::external_agent::NAME.into(),
        aliases: benchflow_run::external_agent::ALIASES
            .iter()
            .map(|a| a.to_string())
            .collect(),
        harness: "backend".into(),
        login: dim("--backend <url>"),
        checked: dim("-"),
        what: EXTERNAL_WHAT.into(),
    });
    let width = wrap_width();
    say!("{}", agents_table(&rows, width));
    say!(
        "{}",
        dim(wrap(
            "",
            "The first harness is the agent's default; `bench run --harness <name>` picks another. `bench agents <name>` shows how an agent is installed, started, and logged in. `checked` is the last `bench agents check` of exactly this build (it installed and answered ACP's initialize; no prompt was sent). Any entry of the public ACP registry runs as acp:<id> (`bench agents --registry`). Exploit probes run as probe:<id>; `bench test --list-probes` lists them.",
            0,
            width
        ))
    );
    let providers: Vec<String> = ProviderRegistry::builtin()
        .providers()
        .iter()
        .map(|p| p.name.clone())
        .collect();
    say!(
        "{}",
        dim(wrap(
            "",
            &format!(
                "`-m <provider>/<model>` runs an agent on a provider's key, which BenchFlow's forwarder holds so the agent never sees it: {}.",
                providers.join(", ")
            ),
            0,
            width
        ))
    );
    Ok(Exit::Ok)
}

/// One agent in `bench agents`.
struct AgentRow {
    name: String,
    aliases: Vec<String>,
    harness: String,
    login: String,
    checked: String,
    what: String,
}

/// The agents as a table, or, when that is wider than `width`, the table
/// without its `what` column, each agent followed by what it is and its
/// aliases, wrapped under it.
fn agents_table(rows: &[AgentRow], width: usize) -> String {
    let mut t = Table::new(&["agent", "harness", "login", "checked", "what"]);
    for r in rows {
        let name = if r.aliases.is_empty() {
            r.name.clone()
        } else {
            format!("{} {}", r.name, dim(format!("({})", r.aliases.join(", "))))
        };
        t.row(vec![
            name,
            r.harness.clone(),
            r.login.clone(),
            r.checked.clone(),
            r.what.clone(),
        ]);
    }
    let table = t.render(2);
    if table.lines().all(|l| visible_width(l) <= width) {
        return table;
    }
    let mut t = Table::new(&["agent", "harness", "login", "checked"]);
    for r in rows {
        t.row(vec![
            r.name.clone(),
            r.harness.clone(),
            r.login.clone(),
            r.checked.clone(),
        ]);
    }
    let table = t.render(2);
    let mut lines = table.lines();
    let mut out = vec![lines.next().unwrap_or_default().to_string()];
    for (line, r) in lines.zip(rows) {
        out.push(line.to_string());
        let mut what = r.what.clone();
        if !r.aliases.is_empty() {
            what.push_str(&format!(" (also {})", r.aliases.join(", ")));
        }
        out.push(wrap("    ", &dim(what), 4, width));
    }
    out.join("\n") + "\n"
}

/// The harnesses an agent supports, its default first.
fn harnesses(m: &AgentManifest) -> String {
    std::iter::once(m.harness)
        .chain(m.harnesses.iter().copied().filter(|h| *h != m.harness))
        .map(|h| h.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

/// The last check of this exact build: `✓ daytona 2026-10-10`.
fn checked_cell(checks: &[CheckRecord], m: &AgentManifest) -> String {
    match agent_check::latest(checks, &m.name, &m.version_fingerprint()) {
        Some(r) if r.ok() => format!("{} {} {}", ok_mark(), r.backend, day(&r.date)),
        Some(r) => format!(
            "{} {} {} {}",
            fail_mark(),
            r.code.as_deref().unwrap_or("failed"),
            r.backend,
            day(&r.date)
        ),
        None => dim("not checked"),
    }
}

fn day(date: &str) -> &str {
    date.get(..10).unwrap_or(date)
}

/// The chosen login's name, without notes.
fn chosen_login(status: &benchflow_agent::credentials::LoginStatus) -> String {
    status
        .chosen
        .map(|i| status.sources[i].name.clone())
        .unwrap_or_default()
}

fn show(m: &AgentManifest) {
    let host = HostEnv::from_process();
    say!("{}  {}", bold(&m.name), m.description);
    if !m.aliases.is_empty() {
        field("aliases", &m.aliases.join(", "));
    }
    field("source", &m.source.to_string());
    if let Some(id) = &m.acp_registry
        && !matches!(m.source, ManifestSource::AcpRegistry(_))
    {
        field(
            "registry",
            &format!("{id} (the registry's build runs as acp:{id})"),
        );
    }
    field(
        "harness",
        &format!("{} {}", harnesses(m), dim("(the first is the default)")),
    );
    let install = match &m.install.method {
        InstallMethod::Npm(npm) => format!("npm {} (runs {})", npm.package, npm.bin),
        InstallMethod::Script(_) => "a shell script (below)".to_string(),
        InstallMethod::UvTool(uv) => format!("PyPI {} with uv (runs {})", uv.requirement, uv.bin),
        InstallMethod::Installer(home) => format!(
            "the vendor's installer (below), linking {}",
            home.links.join(", ")
        ),
        InstallMethod::Binary(b) => {
            let platforms: Vec<&str> = b.targets.keys().map(String::as_str).collect();
            format!("a release archive for {}", platforms.join(", "))
        }
        InstallMethod::Npx(n) => format!("npm {} (run as npx runs it)", n.package),
        InstallMethod::Uvx(u) => format!("PyPI {} with uv (runs {})", u.package, u.bin),
        InstallMethod::UvVenv(venv) => format!(
            "PyPI {} with uv, in /opt/benchflow/agents/{}/venv",
            venv.requirements.join(" "),
            m.name
        ),
    };
    field("install", &install);
    if let Some(account) = &m.account {
        field("account", &format!("needs {account}"));
    }
    if let Some(model) = m.default_model() {
        field("model", &format!("{model} by default"));
    }
    let checks = agent_check::load(&default_cache_dir());
    let checked = match agent_check::latest(&checks, &m.name, &m.version_fingerprint()) {
        Some(r) if r.ok() => format!(
            "{} installed and answered initialize on {} ({}): {} {}, protocol {}, login methods: {}",
            ok_mark(),
            r.backend,
            day(&r.date),
            r.agent_name.as_deref().unwrap_or("unnamed"),
            r.agent_version.as_deref().unwrap_or("(no version)"),
            r.protocol_version.unwrap_or(0),
            if r.auth_methods.is_empty() {
                "none".to_string()
            } else {
                r.auth_methods.join(", ")
            }
        ),
        Some(r) => format!(
            "{} failed on {} ({}): {}",
            fail_mark(),
            r.backend,
            day(&r.date),
            r.problem.as_deref().unwrap_or("no reason recorded")
        ),
        None => dim(format!(
            "this build was never checked; `bench agents check {} -s <backend>`",
            m.name
        )),
    };
    field("checked", &checked);
    let resumes: Vec<&str> = m
        .harnesses
        .iter()
        .filter(|h| m.resume.on(**h))
        .map(|h| h.as_str())
        .collect();
    let resume = if resumes.is_empty() {
        dim("no: with --resume-sessions, each graded stage starts a fresh session")
    } else {
        format!(
            "on {} with --resume-sessions (its sessions live in ~/{})",
            resumes.join(" and "),
            m.resume.session_dirs.join(", ~/")
        )
    };
    field("resume", &resume);
    let loads = match m.resume.load {
        Some(format) if !resumes.is_empty() => format!(
            "on {} with --load-trajectory: {}",
            resumes.join(" and "),
            format.expects()
        ),
        _ => dim("no: --load-trajectory is refused"),
    };
    field("loads", &loads);
    let skills = if m.skill_paths.is_empty() {
        dim("none: the agent takes no skills")
    } else {
        let via = m
            .skills_via
            .as_ref()
            .map(|v| format!("; {v}"))
            .unwrap_or_default();
        format!("copied into {}{via}", m.skill_paths.join(", "))
    };
    field("skills", &skills);
    for note in &m.notes {
        field("note", note);
    }
    show_options(&m.name, &m.options);
    let status = login_status(m, None, &host);
    say!("");
    say!("{}", bold("Logins, in the order the agent prefers them"));
    if status.sources.is_empty() {
        say!("  {}", dim("none found"));
    }
    for (i, source) in status.sources.iter().enumerate() {
        let mark = if Some(i) == status.chosen {
            ok_mark()
        } else if source.usable {
            dim("·")
        } else {
            fail_mark()
        };
        say!("  {mark} {source}");
    }
    for note in &status.notes {
        say!("  {}", dim(note));
    }
    if let Some(fix) = &status.fix {
        say!(
            "{}",
            wrap(&format!("  {} ", dim("to log in:")), fix, 4, wrap_width())
        );
    }
    let registry = ProviderRegistry::builtin();
    say!("");
    say!(
        "{}",
        bold("Keys and BenchFlow's forwarder (the agent never sees a key the forwarder holds)")
    );
    let kinds = credential_kinds(m);
    let width = kinds.iter().map(|(name, _)| name.len()).max().unwrap_or(0);
    for (name, kind) in &kinds {
        say!(
            "{}",
            wrap(
                &format!("  {name:<width$}  "),
                &key_handling(m, name, *kind, &registry).to_string(),
                width + 4,
                wrap_width()
            )
        );
    }
    match (&m.credentials, m.providers.is_empty()) {
        (CredentialFamily::None, _) => say!("  {}", dim("none: it needs no login from BenchFlow")),
        (CredentialFamily::Model, true) => say!(
            "  {}",
            "the key of the model's provider: given to the agent (no [providers] table); a model on a router (vercel/, openrouter/) is refused"
        ),
        (CredentialFamily::Model, false) => say!(
            "  {}",
            "the key of the model's provider: held by the forwarder when BenchFlow knows the provider, else given to the agent"
        ),
        _ => {}
    }
    if !m.providers.is_empty() {
        say!(
            "  {}",
            dim("a `<provider>/<model>` run: the provider's key, held by the forwarder")
        );
    }
    say!("");
    say!(
        "{}",
        bold("Providers it can use (`-m <provider>/<model>`), in the order it prefers them")
    );
    if m.providers.is_empty() {
        say!(
            "  {}",
            dim("none: its manifest has no [providers] table, so it runs only on its own logins")
        );
    }
    for spec in &m.providers {
        let usable: Vec<&str> = registry
            .providers()
            .iter()
            .filter(|p| p.usable_by(spec).is_some())
            .map(|p| p.name.as_str())
            .collect();
        let mut how = Vec::new();
        if let Some(var) = &spec.base_url {
            how.push(format!("base URL in {var}"));
        }
        if let Some(var) = &spec.key {
            how.push(format!("a placeholder key in {var}"));
        }
        if !spec.env.is_empty() {
            how.push(format!(
                "{} set from templates",
                spec.env.keys().cloned().collect::<Vec<_>>().join(", ")
            ));
        }
        say!(
            "{}",
            wrap(
                &format!("  {}  ", spec.describe()),
                &dim(how.join("; ")),
                4,
                wrap_width()
            )
        );
        say!(
            "    {}",
            if usable.is_empty() {
                "no built-in provider serves it".to_string()
            } else {
                usable.join(", ")
            }
        );
    }
    say!("");
    let width = wrap_width();
    let mut cut_any = false;
    if m.declares(Harness::Acp) {
        say!("{}", bold("Started with (acp)"));
        say!("  {}", m.launch_command());
        for (k, v) in m.launch_env() {
            say!("  {}", dim(format!("{k}={v}")));
        }
    }
    if let Some(headless) = &m.headless {
        if m.declares(Harness::Acp) {
            say!("");
        }
        say!(
            "{} {}",
            bold("Started with (headless)"),
            dim(format!("output read as {}", headless.output))
        );
        let (lines, cut) = script_lines(&headless.command, width);
        cut_any |= cut;
        for line in lines {
            say!("{line}");
        }
    }
    say!("");
    say!(
        "{}",
        bold("Installed with (as root, cached as an image layer)")
    );
    let (lines, cut) = script_lines(&m.install_script(), width);
    cut_any |= cut;
    for line in lines {
        say!("{}", dim(line));
    }
    if cut_any {
        say!(
            "{}",
            dim(wrap(
                "",
                &format!(
                    "Lines ending in … are cut to fit the terminal; `bench agents {} | less -S` shows them whole.",
                    m.name
                ),
                0,
                width
            ))
        );
    }
}

/// A shell script's lines, indented by two spaces and cut to `width`
/// columns with an ellipsis (a shell line wrapped by the terminal is hard
/// to read, and one wrapped by us would not run); true when any was cut.
fn script_lines(script: &str, width: usize) -> (Vec<String>, bool) {
    let mut cut = false;
    let lines = script
        .lines()
        .map(|line| {
            let line = format!("  {line}");
            if visible_width(&line) > width {
                cut = true;
                truncate(&line, width)
            } else {
                line
            }
        })
        .collect();
    (lines, cut)
}

/// The keys native agents can use here, by provider (never a value).
fn native_login(host: &HostEnv) -> String {
    use benchflow_agent::native::{BuiltinProviders, EndpointResolver};
    let ready: Vec<&str> = ["vercel", "openai", "anthropic", "openrouter"]
        .into_iter()
        .filter(|p| BuiltinProviders.resolve(&format!("{p}/x"), host).is_ok())
        .collect();
    if ready.is_empty() {
        format!("{} no provider key", fail_mark())
    } else {
        format!("{} {}/…", ok_mark(), ready.join("/…, "))
    }
}

const EXTERNAL_WHAT: &str =
    "an outside backend (an RL trainer) decides each step over HTTP; the tools run in the sandbox";

fn show_external() {
    say!(
        "{}  {}",
        bold(benchflow_run::external_agent::NAME),
        EXTERNAL_WHAT
    );
    say!(
        "  {:<10} {}",
        dim("aliases"),
        benchflow_run::external_agent::ALIASES.join(", ")
    );
    say!(
        "  {:<10} backend: bench speaks the backend protocol to --backend <url> from this machine",
        dim("harness")
    );
    say!(
        "  {:<10} GET transcript_length, POST create_transcript, append_transcript, update_run_state, GET get_message (404 while none), POST delete_message, under /api/internal/, with ?run_id=<trial name>",
        dim("routes")
    );
    say!(
        "  {:<10} bash, view, replace, write, submit (listed in the task_pre_hook_completed event's metadata)",
        dim("tools")
    );
    say!(
        "  {:<10} {}",
        dim("login"),
        "none; a bearer token from the file BENCHFLOW_BACKEND_TOKEN_PATH names, when it exists"
    );
    say!(
        "  {:<10} --idle-timeout (the longest wait for a message), the task's timeout, --max-tool-calls",
        dim("budgets")
    );
    say!(
        "  {:<10} agent/backend_transcript.json (every event sent), trajectory/acp_trajectory.jsonl",
        dim("records")
    );
    say!(
        "  {:<10} -m is only a label for the backend (task_started.model); the backend serves the model",
        dim("model")
    );
}

/// The options an agent takes (`--ak name=value`), as a table.
/// One `label value` line of `bench agents <name>`, wrapped under its value.
/// The label is padded before it is dimmed: `{:<10}` would count the color
/// codes and leave it unpadded on a terminal.
fn field(label: &str, value: &str) {
    say!("{}", field_line(label, value, wrap_width()));
}

fn field_line(label: &str, value: &str, width: usize) -> String {
    wrap(
        &format!("  {} ", dim(format!("{label:<10}"))),
        value,
        13,
        width,
    )
}

fn show_options(agent: &str, options: &[benchflow_agent::manifest::AgentOption]) {
    say!("");
    if options.is_empty() {
        say!(
            "{} {}",
            bold("Options"),
            dim("none; pass variables the agent reads with --env")
        );
        return;
    }
    say!(
        "{} {}",
        bold("Options"),
        dim(format!(
            "(bench run -a {agent} --ak <name>=<value>; unset ones are not passed)"
        ))
    );
    say!("{}", options_table(options, wrap_width()));
}

/// The options as a table, or, when that is wider than `width`, two lines
/// per option: what it does, then what it takes and how it reaches the agent.
fn options_table(options: &[benchflow_agent::manifest::AgentOption], width: usize) -> String {
    let rows: Vec<[String; 5]> = options
        .iter()
        .map(|o| {
            let mut how = o.route.describe();
            if !o.harnesses.is_empty() {
                let on: Vec<&str> = o.harnesses.iter().map(|h| h.as_str()).collect();
                how.push_str(&format!(" ({} only)", on.join(", ")));
            }
            let default = o.default.as_ref().map(|d| d.to_string());
            [
                o.name.clone(),
                o.expects(),
                how,
                default.unwrap_or_default(),
                o.description.clone(),
            ]
        })
        .collect();
    let mut t = Table::new(&["option", "takes", "reaches the agent as", "default", "what"]);
    for [name, takes, how, default, what] in &rows {
        let default = if default.is_empty() {
            dim("-")
        } else {
            default.clone()
        };
        t.row(vec![
            name.clone(),
            takes.clone(),
            how.clone(),
            default,
            what.clone(),
        ]);
    }
    let table = t.render(2);
    if table.lines().all(|l| visible_width(l) <= width) {
        return table;
    }
    let name_w = rows.iter().map(|r| r[0].len()).max().unwrap_or(0);
    let indent = name_w + 4;
    let mut out = Vec::new();
    for [name, takes, how, default, what] in &rows {
        out.push(wrap(&format!("  {name:<name_w$}  "), what, indent, width));
        let mut detail = format!("takes {takes} · {how}");
        if !default.is_empty() {
            detail.push_str(&format!(" · default {default}"));
        }
        out.push(wrap(&" ".repeat(indent), &dim(detail), indent, width));
    }
    out.join("\n")
}

fn show_native(name: &str) {
    let Some(found) = benchflow_agent::native::find(name) else {
        return;
    };
    let Some((name, aliases, what)) = benchflow_agent::native::AGENTS
        .iter()
        .find(|(n, _, _)| *n == found)
    else {
        return;
    };
    say!("{}  {}", bold(name), what);
    if !aliases.is_empty() {
        field("aliases", &aliases.join(", "));
    }
    say!(
        "  {:<10} native (the loop runs in bench; the model is called from this machine)",
        dim("harness")
    );
    say!(
        "  {:<10} <provider>/<model>, such as vercel/deepseek/deepseek-v4-flash-0731",
        dim("model")
    );
    say!(
        "  {:<10} vercel (AI_GATEWAY_API_KEY), openai (OPENAI_API_KEY), anthropic (ANTHROPIC_API_KEY), openrouter (OPENROUTER_API_KEY)",
        dim("providers")
    );
    say!(
        "  {:<10} {}",
        dim("login"),
        native_login(&HostEnv::from_process())
    );
    say!(
        "  {:<10} --max-tool-calls, --max-tokens, the task's timeout, --idle-timeout",
        dim("budgets")
    );
    say!(
        "  {:<10} agent/model_calls.jsonl (every request and response, key redacted)",
        dim("records")
    );
    say!("");
    if *name == benchflow_agent::native::COMPUTER_1 {
        say!(
            "{}",
            dim(format!(
                "Nothing is installed: the task's image must have the desktop ({}, a window manager, and a browser). Screenshots go to agent/screenshots/ in the trial folder; the final answer to /logs/agent/final_answer.txt.",
                benchflow_agent::native::computer::desktop::REQUIRED
            ))
        );
    } else if *name == benchflow_agent::native::TERMINUS_2 {
        say!(
            "{}",
            bold("Installed with (as root, cached as an image layer)")
        );
        for line in benchflow_agent::native::terminus::tmux::INSTALL_SCRIPT.lines() {
            say!("  {}", dim(line));
        }
    } else {
        say!(
            "{}",
            dim("Nothing is installed in the sandbox; the tools run through the sandbox API.")
        );
    }
    show_options(name, &benchflow_agent::native::options::declared(name));
}

async fn list_registry() -> Result<Exit> {
    let (loaded, _) = registry_index().await?;
    let builtins = registry()?;
    let mut t = Table::new(&["agent", "version", "installs", "built in as", "what"]);
    for entry in &loaded.index.agents {
        let builtin = builtins
            .agents()
            .iter()
            .find(|a| a.acp_registry.as_deref() == Some(entry.id.as_str()))
            .map(|a| a.name.clone())
            .unwrap_or_else(|| dim("-"));
        let what = crate::ui::truncate(&entry.description, 60);
        t.row(vec![
            format!("acp:{}", entry.id),
            entry.version.clone(),
            entry.install_summary(),
            builtin,
            what,
        ]);
    }
    say!("{}", t.render(2));
    say!(
        "{}",
        dim(format!(
            "{} entries from {}. Run one with `-a acp:<id>` (or acp:<id>@<version> to pin it); `bench agents acp:<id>` shows how it is installed and logged in.",
            loaded.index.agents.len(),
            loaded.origin.describe()
        ))
    );
    Ok(Exit::Ok)
}

async fn run_checks(args: AgentsCheckArgs) -> Result<Exit> {
    let handshake_timeout = context::parse_duration(&args.handshake_timeout)?;
    let registry = registry()?;
    let mut manifests = Vec::new();
    for name in &args.names {
        manifests.push(manifest(name, &registry).await?);
    }
    let provider = context::provider(args.sandbox.as_deref()).await?;
    let spec = CheckSpec {
        image: args.image.clone(),
        handshake_timeout,
    };
    let cache = default_cache_dir();
    let logs = cache.join("agent-check-logs");
    let _ = std::fs::create_dir_all(&logs);
    note!(
        "  checking {} on {} ({}): install, start, ACP initialize; no prompt is sent",
        crate::ui::plural(manifests.len(), "agent"),
        provider.name(),
        spec.image
    );
    let limit = Arc::new(tokio::sync::Semaphore::new(args.concurrency.max(1)));
    let mut set = tokio::task::JoinSet::new();
    for (i, m) in manifests.into_iter().enumerate() {
        let provider = provider.clone();
        let spec = spec.clone();
        let limit = limit.clone();
        let log_path = logs.join(format!(
            "{}.log",
            m.name
                .replace(|c: char| !c.is_ascii_alphanumeric() && c != '-', "_")
        ));
        set.spawn(async move {
            let _permit = limit.acquire_owned().await;
            note!("  {} {}", dim("·"), m.name);
            let file = std::sync::Mutex::new(std::fs::File::create(&log_path).ok());
            let log = move |line: &str| {
                if let Ok(mut f) = file.lock()
                    && let Some(f) = f.as_mut()
                {
                    use std::io::Write;
                    let _ = writeln!(f, "{line}");
                }
            };
            let record = agent_check::check(provider.as_ref(), &m, &spec, &log).await;
            (i, record, log_path)
        });
    }
    let mut results = Vec::new();
    while let Some(joined) = set.join_next().await {
        let (i, record, log_path) =
            joined.map_err(|e| CliError::bug("running an agent check", e))?;
        if let Err(e) = agent_check::save(&cache, &record) {
            note!(
                "  {} could not record the check in {}: {e}",
                warn_mark(),
                agent_check::record_path(&cache).display()
            );
        }
        print_check(&record, &log_path);
        results.push((i, record));
    }
    results.sort_by_key(|(i, _)| *i);
    let failed = results.iter().filter(|(_, r)| !r.ok()).count();
    if results.len() > 1 {
        let mut t = Table::new(&[
            "agent",
            "result",
            "agent says",
            "protocol",
            "login methods",
            "install",
            "start",
        ]);
        for (_, r) in &results {
            t.row(vec![
                r.agent.clone(),
                if r.ok() {
                    format!("{} ok", ok_mark())
                } else {
                    format!("{} {}", fail_mark(), r.code.as_deref().unwrap_or("failed"))
                },
                match (&r.agent_name, &r.agent_version) {
                    (Some(n), Some(v)) => format!("{n} {v}"),
                    (Some(n), None) => n.clone(),
                    _ => dim("-"),
                },
                r.protocol_version
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| dim("-")),
                if r.auth_methods.is_empty() {
                    dim("-")
                } else {
                    r.auth_methods.join(", ")
                },
                crate::ui::duration(Duration::from_secs_f64(r.install_secs)),
                r.start_secs
                    .map(|s| crate::ui::duration(Duration::from_secs_f64(s)))
                    .unwrap_or_else(|| dim("-")),
            ]);
        }
        say!("");
        say!("{}", t.render(2));
    }
    say!(
        "{}",
        dim(format!(
            "Recorded in {}. A check proves the agent installs and answers initialize; it sends no prompt, so it says nothing about logins or models.",
            agent_check::record_path(&cache).display()
        ))
    );
    Ok(if failed == 0 { Exit::Ok } else { Exit::Failed })
}

fn print_check(r: &CheckRecord, log_path: &std::path::Path) {
    let install = crate::ui::duration(Duration::from_secs_f64(r.install_secs));
    if r.ok() {
        say!(
            "  {} {}: {} {} answered initialize (protocol {}; login methods: {}); install {}{}, start {}",
            ok_mark(),
            bold(&r.agent),
            r.agent_name.as_deref().unwrap_or("unnamed agent"),
            r.agent_version.as_deref().unwrap_or("(no version)"),
            r.protocol_version.unwrap_or(0),
            if r.auth_methods.is_empty() {
                "none".to_string()
            } else {
                r.auth_methods.join(", ")
            },
            install,
            if r.built_now { "" } else { " (cached)" },
            r.start_secs
                .map(|s| crate::ui::duration(Duration::from_secs_f64(s)))
                .unwrap_or_default(),
        );
    } else {
        say!(
            "  {} {}: [{}] {}",
            fail_mark(),
            bold(&r.agent),
            r.code.as_deref().unwrap_or("?"),
            r.problem.as_deref().unwrap_or("failed")
        );
        if let Some(owner) = &r.owner {
            say!("      {} {owner}", dim("whose:"));
        }
        if let Some(evidence) = &r.evidence {
            for line in evidence
                .lines()
                .rev()
                .take(8)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
            {
                say!("      {}", dim(line));
            }
        }
        say!("      {} {}", dim("build log:"), log_path.display());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Text without its color codes.
    fn plain(s: &str) -> String {
        let mut out = String::new();
        let mut esc = false;
        for c in s.chars() {
            match (esc, c) {
                (_, '\u{1b}') => esc = true,
                (true, c) if c.is_ascii_alphabetic() => esc = false,
                (true, _) => {}
                (false, c) => out.push(c),
            }
        }
        out
    }

    fn sample_rows() -> Vec<AgentRow> {
        let registry = registry().unwrap();
        registry
            .agents()
            .iter()
            .map(|m| AgentRow {
                name: m.name.clone(),
                aliases: m.aliases.clone(),
                harness: harnesses(m),
                login: "✗ no login".into(),
                checked: "not checked".into(),
                what: m.description.clone(),
            })
            .collect()
    }

    #[test]
    fn the_agent_list_fits_a_narrow_terminal() {
        let rows = sample_rows();
        let wide = plain(&agents_table(&rows, usize::MAX));
        assert!(
            wide.lines().any(|l| visible_width(l) > 100),
            "the full table is wider than 100 columns"
        );
        assert!(
            wide.contains("claude (claude-agent-acp, claude-code)"),
            "{wide}"
        );
        let narrow = plain(&agents_table(&rows, 100));
        for line in narrow.lines() {
            assert!(visible_width(line) <= 100, "wider than 100: {line}");
        }
        let lines: Vec<&str> = narrow.lines().collect();
        assert!(lines[0].starts_with("  agent"), "{narrow}");
        assert!(!lines[0].contains("what"), "{narrow}");
        let claude = lines
            .iter()
            .position(|l| l.starts_with("  claude "))
            .unwrap();
        assert!(lines[claude].contains("✗ no login"), "{narrow}");
        assert!(
            lines[claude + 1].starts_with("    Claude Code via ACP"),
            "{narrow}"
        );
        assert!(
            narrow
                .replace("\n    ", " ")
                .contains("(also claude-agent-acp, claude-code)"),
            "{narrow}"
        );
    }

    #[test]
    fn long_script_lines_are_cut_to_the_terminal() {
        let script = format!("short\n{}", "x ".repeat(200));
        let (lines, cut) = script_lines(&script, 100);
        assert!(cut);
        assert_eq!(lines[0], "  short");
        assert_eq!(visible_width(&lines[1]), 100);
        assert!(lines[1].ends_with('…'));
        let (lines, cut) = script_lines(&script, usize::MAX);
        assert!(!cut, "nothing is cut when the output is not a terminal");
        assert_eq!(
            lines[1],
            format!("  {}", "x ".repeat(200)).trim_end_matches('\n')
        );
    }

    #[test]
    fn options_fit_a_narrow_terminal() {
        let registry = registry().unwrap();
        let m = registry.find("claude").unwrap();
        let wide = options_table(&m.options, usize::MAX);
        assert!(wide.contains("reaches the agent as"), "{wide}");
        let narrow = options_table(&m.options, 100);
        for line in narrow.lines() {
            assert!(visible_width(line) <= 100, "wider than 100: {line}");
        }
        assert!(narrow.contains("Most agent turns."), "{narrow}");
        assert!(
            narrow.contains("takes a whole number from 1 · flag --max-turns (headless only)"),
            "{narrow}"
        );
        assert!(narrow.contains("default 2.1.280"), "{narrow}");
    }

    #[test]
    fn fields_line_up_and_wrap_under_their_value() {
        let line = field_line("loads", "one two three four", 23);
        let plain: Vec<String> = line
            .lines()
            .map(|l| {
                let mut out = String::new();
                let mut esc = false;
                for c in l.chars() {
                    match (esc, c) {
                        (_, '\u{1b}') => esc = true,
                        (true, c) if c.is_ascii_alphabetic() => esc = false,
                        (true, _) => {}
                        (false, c) => out.push(c),
                    }
                }
                out
            })
            .collect();
        assert_eq!(plain, ["  loads      one two", "             three four"]);
    }
}
