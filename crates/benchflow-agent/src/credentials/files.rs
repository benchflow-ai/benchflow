//! Login files: whether an agent can run on them, and which of their values
//! are secrets.
//!
//! A file is copied into the sandbox as it is (mode 0600); BenchFlow only
//! reads it to check that it holds a login and to learn the token values to
//! redact.

use std::collections::BTreeMap;
use std::path::Path;

use base64::Engine;
use chrono::{DateTime, TimeZone, Utc};
use serde_json::Value;

use super::families::Candidate;
use super::{
    CredentialError, HostEnv, Login, LoginSource, Origin, PlainKey, SandboxFile, SourceKind,
};

/// Login files are small; anything bigger is not one.
const MAX_LOGIN_FILE: u64 = 1024 * 1024;

/// What reading one login file found.
struct Found {
    /// The file's name for messages: `~/.claude/.credentials.json`.
    display: String,
    contents: Vec<u8>,
    json: Value,
}

/// Read and parse `~/<rel>`; `None` when the file does not exist.
fn read_home_json(host: &HostEnv, rel: &str) -> Option<Result<Found, (String, String)>> {
    let home = host.home()?;
    let path = home.join(rel);
    if !path.is_file() {
        return None;
    }
    let display = format!("~/{rel}");
    Some(
        read_json(&path)
            .map(|(contents, json)| Found {
                display: display.clone(),
                contents,
                json,
            })
            .map_err(|problem| (display, problem)),
    )
}

fn read_json(path: &Path) -> Result<(Vec<u8>, Value), String> {
    let size = std::fs::metadata(path)
        .map_err(|e| format!("cannot read it: {e}"))?
        .len();
    if size > MAX_LOGIN_FILE {
        return Err(format!("it is {size} bytes, too big for a login file"));
    }
    let contents = std::fs::read(path).map_err(|e| format!("cannot read it: {e}"))?;
    let json = serde_json::from_slice(&contents).map_err(|_| "it is not JSON".to_string())?;
    Ok((contents, json))
}

/// A candidate for a file the agent cannot use.
fn unusable(
    agent: &str,
    name: String,
    kind: SourceKind,
    origin: Origin,
    problem: String,
    next_step: &str,
) -> Candidate {
    Candidate {
        source: LoginSource {
            name: name.clone(),
            kind,
            origin,
            usable: false,
            note: Some(problem.clone()),
            expires_at: None,
        },
        login: Err(CredentialError::BadFile {
            agent: agent.to_string(),
            path: name,
            problem,
            next_step: next_step.to_string(),
        }),
        api_key: None,
    }
}

/// A login that copies files into the sandbox.
fn file_login(label: String, files: Vec<(String, Vec<u8>)>, secrets_from: &[&Value]) -> Login {
    let mut login = Login {
        label,
        env: BTreeMap::new(),
        files: files
            .into_iter()
            .map(|(path, contents)| SandboxFile {
                path,
                contents,
                mode: 0o600,
            })
            .collect(),
        secrets: Vec::new(),
        notes: Vec::new(),
    };
    for json in secrets_from {
        collect_secrets(json, &mut login);
    }
    login
}

/// Every string under a key that names a token, key, secret or password.
fn collect_secrets(json: &Value, login: &mut Login) {
    match json {
        Value::Object(map) => {
            for (key, value) in map {
                let key = key.to_ascii_lowercase();
                let secret_key = ["token", "key", "secret", "password"]
                    .iter()
                    .any(|m| key.contains(m));
                match value {
                    Value::String(s) if secret_key => login.add_secret(s),
                    _ => collect_secrets(value, login),
                }
            }
        }
        Value::Array(items) => items.iter().for_each(|item| collect_secrets(item, login)),
        _ => {}
    }
}

/// A Unix time in milliseconds, as login files write expiry.
fn from_millis(value: Option<&Value>) -> Option<DateTime<Utc>> {
    let ms = value?.as_f64()?;
    ms.is_finite()
        .then(|| Utc.timestamp_millis_opt(ms as i64).single())
        .flatten()
}

fn when(t: DateTime<Utc>) -> String {
    t.format("%Y-%m-%d %H:%M UTC").to_string()
}

/// `~/.claude/.credentials.json`, written by `claude login` on Linux.
pub(super) fn claude(agent: &str, host: &HostEnv, next_step: &str) -> Option<Candidate> {
    let found = match read_home_json(host, ".claude/.credentials.json")? {
        Ok(found) => found,
        Err((name, problem)) => {
            return Some(unusable(
                agent,
                name,
                SourceKind::LoginFile,
                Origin::File,
                problem,
                next_step,
            ));
        }
    };
    let oauth = found.json.get("claudeAiOauth");
    let token = oauth
        .and_then(|o| o.get("accessToken"))
        .and_then(Value::as_str);
    if token.is_none_or(str::is_empty) {
        return Some(unusable(
            agent,
            found.display,
            SourceKind::LoginFile,
            Origin::File,
            "it holds no OAuth token".to_string(),
            next_step,
        ));
    }
    let expires_at = from_millis(oauth.and_then(|o| o.get("expiresAt")));
    let expired = expires_at.is_some_and(|t| t <= host.now());
    let note = expired.then(|| {
        format!(
            "access token expired {}; usable only if its refresh token still works \
             (on macOS Claude Code keeps live logins in the Keychain, so this file is often stale)",
            when(expires_at.unwrap_or_default())
        )
    });
    let mut login = file_login(
        format!("host login ({})", found.display),
        vec![(
            "$HOME/.claude/.credentials.json".to_string(),
            found.contents,
        )],
        &[&found.json],
    );
    login.notes.extend(note.clone());
    Some(Candidate {
        source: LoginSource {
            name: found.display,
            kind: SourceKind::LoginFile,
            origin: Origin::File,
            usable: !expired,
            note,
            expires_at,
        },
        login: Ok(login),
        api_key: None,
    })
}

/// `~/.codex/auth.json`, written by `codex login`.
pub(super) fn codex(agent: &str, host: &HostEnv, next_step: &str) -> Option<Candidate> {
    let found = match read_home_json(host, ".codex/auth.json")? {
        Ok(found) => found,
        Err((name, problem)) => {
            return Some(unusable(
                agent,
                name,
                SourceKind::AuthJson,
                Origin::File,
                problem,
                next_step,
            ));
        }
    };
    let label = format!("host login ({})", found.display);
    Some(codex_candidate(
        agent,
        found.display,
        Origin::File,
        label,
        found.contents,
        found.json,
        next_step,
    ))
}

/// A Codex `auth.json` passed inline in a variable (`CODEX_AUTH_JSON`).
pub(super) fn codex_auth_json(
    agent: &str,
    name: &str,
    origin: Origin,
    contents: Vec<u8>,
    next_step: &str,
) -> Candidate {
    match serde_json::from_slice::<Value>(&contents) {
        Ok(json) => codex_candidate(
            agent,
            name.to_string(),
            origin,
            format!("{name} ({origin})"),
            contents,
            json,
            next_step,
        ),
        Err(_) => unusable(
            agent,
            name.to_string(),
            SourceKind::AuthJson,
            origin,
            "it is not JSON".to_string(),
            next_step,
        ),
    }
}

fn codex_candidate(
    agent: &str,
    name: String,
    origin: Origin,
    label: String,
    contents: Vec<u8>,
    json: Value,
    next_step: &str,
) -> Candidate {
    if !json.is_object() {
        return unusable(
            agent,
            name,
            SourceKind::AuthJson,
            origin,
            "it is not a JSON object".into(),
            next_step,
        );
    }
    let mut expires_at = None;
    let stored_key = json
        .get("OPENAI_API_KEY")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|k| !k.is_empty())
        .map(str::to_string);
    let note = if stored_key.is_some() {
        "API key stored in file".to_string()
    } else {
        let tokens = json.get("tokens");
        if tokens
            .and_then(|t| t.get("refresh_token"))
            .and_then(Value::as_str)
            .is_none_or(str::is_empty)
        {
            return unusable(
                agent,
                name,
                SourceKind::AuthJson,
                origin,
                "it holds no ChatGPT tokens".into(),
                next_step,
            );
        }
        let claims = tokens
            .and_then(|t| t.get("access_token"))
            .and_then(Value::as_str)
            .and_then(jwt_claims)
            .unwrap_or(Value::Null);
        expires_at = claims
            .get("exp")
            .and_then(Value::as_i64)
            .and_then(|exp| Utc.timestamp_opt(exp, 0).single());
        let mode = json
            .get("auth_mode")
            .and_then(Value::as_str)
            .unwrap_or("chatgpt");
        let mut note = format!("ChatGPT login (auth_mode={mode})");
        if let Some(plan) = claims
            .pointer("/https:~1~1api.openai.com~1auth/chatgpt_plan_type")
            .and_then(Value::as_str)
        {
            note.push_str(&format!(", plan {plan}"));
        }
        // Codex refreshes an expired access token itself, so expiry is
        // only informational.
        note
    };
    let login = file_login(
        label,
        vec![("$HOME/.codex/auth.json".to_string(), contents)],
        &[&json],
    );
    let api_key = stored_key.map(|value| PlainKey {
        var: "OPENAI_API_KEY".to_string(),
        source: match origin {
            Origin::File => format!("OPENAI_API_KEY in {name}"),
            origin => format!("OPENAI_API_KEY in {name} ({origin})"),
        },
        value,
    });
    Candidate {
        source: LoginSource {
            name,
            kind: SourceKind::AuthJson,
            origin,
            usable: true,
            note: Some(note),
            expires_at,
        },
        login: Ok(login),
        api_key,
    }
}

/// The claims of a JWT, without checking its signature.
fn jwt_claims(token: &str) -> Option<Value> {
    let payload = token.split('.').nth(1)?;
    let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(payload.trim_end_matches('='))
        .ok()?;
    serde_json::from_slice(&bytes).ok()
}

/// `~/.gemini/oauth_creds.json`, written when Gemini CLI signs in with
/// Google, copied with the CLI's `settings.json` and `google_accounts.json`.
pub(super) fn gemini(agent: &str, host: &HostEnv, next_step: &str) -> Option<Candidate> {
    let found = match read_home_json(host, ".gemini/oauth_creds.json")? {
        Ok(found) => found,
        Err((name, problem)) => {
            return Some(unusable(
                agent,
                name,
                SourceKind::LoginFile,
                Origin::File,
                problem,
                next_step,
            ));
        }
    };
    if found
        .json
        .get("refresh_token")
        .and_then(Value::as_str)
        .is_none_or(str::is_empty)
    {
        return Some(unusable(
            agent,
            found.display,
            SourceKind::LoginFile,
            Origin::File,
            "it holds no refresh token".to_string(),
            next_step,
        ));
    }
    let mut files = vec![("$HOME/.gemini/oauth_creds.json".to_string(), found.contents)];
    let mut jsons = vec![found.json.clone()];
    for extra in ["settings.json", "google_accounts.json"] {
        if let Some(Ok(more)) = read_home_json(host, &format!(".gemini/{extra}")) {
            files.push((format!("$HOME/.gemini/{extra}"), more.contents));
            jsons.push(more.json);
        }
    }
    let refs: Vec<&Value> = jsons.iter().collect();
    let login = file_login(format!("host login ({})", found.display), files, &refs);
    Some(Candidate {
        source: LoginSource {
            name: found.display,
            kind: SourceKind::LoginFile,
            origin: Origin::File,
            // Gemini CLI refreshes the access token itself.
            usable: true,
            note: Some("Google sign-in".to_string()),
            expires_at: from_millis(found.json.get("expiry_date")),
        },
        login: Ok(login),
        api_key: None,
    })
}
