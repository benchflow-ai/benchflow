//! Which credentials each family accepts, in the order its CLI prefers them.

use std::collections::BTreeMap;

use super::files;
use super::{CredentialError, HostEnv, Login, LoginSource, PlainKey, SourceKind};
use crate::manifest::{AgentManifest, CredentialFamily};

/// One credential found, with the login it gives (or why it gives none).
pub(super) struct Candidate {
    pub source: LoginSource,
    pub login: Result<Login, CredentialError>,
    /// The plain API key it holds, if it is one (not a subscription login).
    pub api_key: Option<PlainKey>,
}

/// Everything a family's scan found, plus what to say when nothing works.
pub(super) struct Scan {
    pub candidates: Vec<Candidate>,
    /// What was looked for, for the "no login" message.
    pub checked: Vec<String>,
    pub next_step: String,
    pub notes: Vec<String>,
}

const CLAUDE_NEXT_STEP: &str = "run `claude setup-token`, then `export CLAUDE_CODE_OAUTH_TOKEN=<token>` \
     (or `export ANTHROPIC_API_KEY=<key>`)";
const CODEX_NEXT_STEP: &str = "run `codex login` (a ChatGPT plan; it writes ~/.codex/auth.json), \
     or `export OPENAI_API_KEY=<key>`";
const GEMINI_NEXT_STEP: &str = "`export GEMINI_API_KEY=<key>` (create one at https://aistudio.google.com/apikey), \
     or run `gemini` once and sign in with Google (it writes ~/.gemini/oauth_creds.json)";

pub(super) fn scan(
    manifest: &AgentManifest,
    model: Option<&str>,
    host: &HostEnv,
) -> Result<Scan, CredentialError> {
    let agent = manifest.name.as_str();
    let mut scan = match &manifest.credentials {
        CredentialFamily::Anthropic => anthropic(agent, host),
        CredentialFamily::OpenAi => openai(agent, host),
        CredentialFamily::Gemini => gemini(agent, host),
        CredentialFamily::Model => from_model(agent, model, host)?,
        CredentialFamily::Env(names) => {
            let names: Vec<&str> = names.iter().map(String::as_str).collect();
            let next_step = match names.first() {
                Some(first) => format!("`export {first}=<key>`"),
                None => "list the variables the agent reads in its manifest's [credentials] env"
                    .to_string(),
            };
            env_family(host, &names, &[], next_step)
        }
        CredentialFamily::None => Scan {
            candidates: Vec::new(),
            checked: Vec::new(),
            next_step: String::new(),
            notes: Vec::new(),
        },
    };
    // A vendor account says how to get its key better than a variable name.
    if let Some(account) = &manifest.account {
        scan.next_step = format!("this agent needs {account}");
    }
    Ok(scan)
}

/// A login from one variable: `sets` lists the variables the sandbox gets
/// (the credential's own name, or the names the agent reads it under).
fn env_candidate(host: &HostEnv, name: &str, kind: SourceKind, sets: &[&str]) -> Option<Candidate> {
    let (value, origin) = host.get(name)?;
    let mut login = Login {
        label: format!("{name} ({origin})"),
        env: BTreeMap::new(),
        files: Vec::new(),
        secrets: Vec::new(),
        notes: Vec::new(),
    };
    for target in sets {
        login.env.insert((*target).to_string(), value.to_string());
    }
    login.add_secret(value);
    Some(Candidate {
        source: LoginSource {
            name: name.to_string(),
            kind,
            origin,
            usable: true,
            note: None,
            expires_at: None,
        },
        login: Ok(login),
        api_key: (kind == SourceKind::ApiKey).then(|| PlainKey {
            var: name.to_string(),
            source: format!("{name} ({origin})"),
            value: value.to_string(),
        }),
    })
}

/// Claude Code's variables: name, kind, and the variables the sandbox gets.
const ANTHROPIC_VARS: [(&str, SourceKind, &[&str]); 4] = [
    (
        "ANTHROPIC_API_KEY",
        SourceKind::ApiKey,
        &["ANTHROPIC_API_KEY"],
    ),
    (
        "CLAUDE_CODE_OAUTH_TOKEN",
        SourceKind::OauthToken,
        &["CLAUDE_CODE_OAUTH_TOKEN"],
    ),
    // The older name; Claude Code reads CLAUDE_CODE_OAUTH_TOKEN.
    (
        "CLAUDE_OAUTH_TOKEN",
        SourceKind::OauthToken,
        &["CLAUDE_CODE_OAUTH_TOKEN"],
    ),
    (
        "ANTHROPIC_AUTH_TOKEN",
        SourceKind::AuthToken,
        &["ANTHROPIC_AUTH_TOKEN"],
    ),
];

/// Codex's variables, as [`ANTHROPIC_VARS`].
const OPENAI_VARS: [(&str, SourceKind, &[&str]); 3] = [
    ("OPENAI_API_KEY", SourceKind::ApiKey, &["OPENAI_API_KEY"]),
    // Codex's own name for the key; the launcher writes auth.json from
    // OPENAI_API_KEY, so the sandbox gets both.
    (
        "CODEX_API_KEY",
        SourceKind::ApiKey,
        &["CODEX_API_KEY", "OPENAI_API_KEY"],
    ),
    (
        "CODEX_ACCESS_TOKEN",
        SourceKind::AccessToken,
        &["CODEX_ACCESS_TOKEN"],
    ),
];

/// Every credential `manifest`'s agent reads, by name and kind, in the order
/// it prefers them; the `model` family's depend on the model and are not
/// listed.
pub(super) fn kinds(manifest: &AgentManifest) -> Vec<(String, SourceKind)> {
    let vars = |list: &[(&str, SourceKind, &[&str])]| -> Vec<(String, SourceKind)> {
        list.iter().map(|(n, k, _)| (n.to_string(), *k)).collect()
    };
    match &manifest.credentials {
        CredentialFamily::Anthropic => {
            let mut all = vars(&ANTHROPIC_VARS);
            all.push(("~/.claude/.credentials.json".into(), SourceKind::LoginFile));
            all
        }
        CredentialFamily::OpenAi => {
            let mut all = vars(&OPENAI_VARS);
            all.push(("CODEX_AUTH_JSON".into(), SourceKind::AuthJson));
            all.push(("~/.codex/auth.json".into(), SourceKind::AuthJson));
            all
        }
        CredentialFamily::Gemini => vec![
            ("GEMINI_API_KEY".into(), SourceKind::ApiKey),
            ("GOOGLE_API_KEY".into(), SourceKind::ApiKey),
            ("~/.gemini/oauth_creds.json".into(), SourceKind::LoginFile),
        ],
        CredentialFamily::Env(names) => names
            .iter()
            .map(|n| (n.clone(), SourceKind::ApiKey))
            .collect(),
        CredentialFamily::Model | CredentialFamily::None => Vec::new(),
    }
}

/// Claude Code's logins.
fn anthropic(agent: &str, host: &HostEnv) -> Scan {
    let mut candidates: Vec<Candidate> = ANTHROPIC_VARS
        .into_iter()
        .filter_map(|(name, kind, sets)| env_candidate(host, name, kind, sets))
        .collect();
    candidates.extend(files::claude(agent, host, CLAUDE_NEXT_STEP));

    let mut notes = Vec::new();
    let has = |name: &str| candidates.iter().any(|c| c.source.name == name);
    if has("ANTHROPIC_API_KEY") && (has("CLAUDE_CODE_OAUTH_TOKEN") || has("CLAUDE_OAUTH_TOKEN")) {
        notes.push(
            "ANTHROPIC_API_KEY overrides the OAuth token; unset it to run on the subscription"
                .to_string(),
        );
    }
    Scan {
        candidates,
        checked: [
            "ANTHROPIC_API_KEY",
            "CLAUDE_CODE_OAUTH_TOKEN",
            "CLAUDE_OAUTH_TOKEN",
            "ANTHROPIC_AUTH_TOKEN",
            "~/.claude/.credentials.json",
        ]
        .map(String::from)
        .to_vec(),
        next_step: CLAUDE_NEXT_STEP.to_string(),
        notes,
    }
}

/// Codex's logins.
fn openai(agent: &str, host: &HostEnv) -> Scan {
    let mut candidates: Vec<Candidate> = OPENAI_VARS
        .into_iter()
        .filter_map(|(name, kind, sets)| env_candidate(host, name, kind, sets))
        .collect();
    if let Some((inline, origin)) = host.get("CODEX_AUTH_JSON") {
        candidates.push(files::codex_auth_json(
            agent,
            "CODEX_AUTH_JSON",
            origin,
            inline.as_bytes().to_vec(),
            CODEX_NEXT_STEP,
        ));
    }
    candidates.extend(files::codex(agent, host, CODEX_NEXT_STEP));

    let mut notes = Vec::new();
    let has_key = candidates
        .iter()
        .any(|c| c.source.kind == SourceKind::ApiKey);
    let has_login = candidates.iter().any(|c| {
        matches!(
            c.source.kind,
            SourceKind::AuthJson | SourceKind::AccessToken
        )
    });
    if has_key && has_login {
        notes.push(
            "OPENAI_API_KEY/CODEX_API_KEY override the ChatGPT login; unset them to run on the subscription"
                .to_string(),
        );
    }
    Scan {
        candidates,
        checked: [
            "OPENAI_API_KEY",
            "CODEX_API_KEY",
            "CODEX_ACCESS_TOKEN",
            "CODEX_AUTH_JSON",
            "~/.codex/auth.json",
        ]
        .map(String::from)
        .to_vec(),
        next_step: CODEX_NEXT_STEP.to_string(),
        notes,
    }
}

/// Gemini CLI's logins. The CLI reads GEMINI_API_KEY; GOOGLE_API_KEY is
/// accepted too, and each is mirrored to the other.
fn gemini(agent: &str, host: &HostEnv) -> Scan {
    const KEYS: &[&str] = &["GEMINI_API_KEY", "GOOGLE_API_KEY"];
    let mut scan = env_family(
        host,
        KEYS,
        &["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        GEMINI_NEXT_STEP.to_string(),
    );
    scan.candidates
        .extend(files::gemini(agent, host, GEMINI_NEXT_STEP));
    scan.checked.push("~/.gemini/oauth_creds.json".to_string());
    scan
}

/// The first of `names` that is set; the sandbox gets it under its own name
/// plus every name in `mirror`.
fn env_family(host: &HostEnv, names: &[&str], mirror: &[&str], next_step: String) -> Scan {
    let candidates = names
        .iter()
        .filter_map(|name| {
            let mut sets: Vec<&str> = vec![name];
            sets.extend(mirror.iter().filter(|m| *m != name));
            env_candidate(host, name, SourceKind::ApiKey, &sets)
        })
        .collect();
    Scan {
        candidates,
        checked: names.iter().map(|n| n.to_string()).collect(),
        next_step,
        notes: Vec::new(),
    }
}

/// OpenCode-style agents: the provider in `provider/model` decides the key.
fn from_model(agent: &str, model: Option<&str>, host: &HostEnv) -> Result<Scan, CredentialError> {
    let provider = model.and_then(crate::model_id::provider).ok_or_else(|| {
        CredentialError::UnknownProvider {
            agent: agent.to_string(),
            model: model.map(str::to_string),
            next_step:
                "name the model as `provider/model`, such as `anthropic/claude-sonnet-4-6` or \
                    `openai/gpt-5`"
                    .to_string(),
        }
    })?;
    let names = provider_key_names(&provider);
    let names: Vec<&str> = names.iter().map(String::as_str).collect();
    let mut next_step = format!("`export {}=<key>`", names[0]);
    if provider == "anthropic" {
        next_step.push_str("; a Claude subscription token does not work with this agent");
    }
    Ok(env_family(host, &names, &names, next_step))
}

/// The variables a models.dev provider's key is read from, preferred first.
fn provider_key_names(provider: &str) -> Vec<String> {
    match provider {
        "anthropic" => vec!["ANTHROPIC_API_KEY".into()],
        "openai" => vec!["OPENAI_API_KEY".into()],
        // OpenCode's google provider reads GOOGLE_GENERATIVE_AI_API_KEY;
        // 0.8 mirrored GEMINI_API_KEY into it.
        "google" => vec![
            "GOOGLE_GENERATIVE_AI_API_KEY".into(),
            "GEMINI_API_KEY".into(),
            "GOOGLE_API_KEY".into(),
        ],
        other => {
            let stem: String = other
                .chars()
                .map(|c| {
                    if c.is_ascii_alphanumeric() {
                        c.to_ascii_uppercase()
                    } else {
                        '_'
                    }
                })
                .collect();
            vec![format!("{stem}_API_KEY")]
        }
    }
}
