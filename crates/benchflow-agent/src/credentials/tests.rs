//! Every test builds its own environment map and home folder; none reads the
//! process environment or a real login. All tokens are fake.

use std::collections::BTreeMap;
use std::path::Path;

use base64::Engine;
use chrono::TimeZone;

use super::*;
use crate::Registry;
use crate::manifest::{AgentManifest, ManifestSource};

const FAKE_KEY: &str = "sk-ant-api03-FAKE-key-for-tests-0000";
const FAKE_OAUTH: &str = "sk-ant-oat01-FAKE-oauth-for-tests-1111";

fn agent(name: &str) -> AgentManifest {
    Registry::builtin().find(name).unwrap().clone()
}

fn host(vars: &[(&str, &str)], home: Option<&Path>) -> HostEnv {
    let vars: BTreeMap<String, String> = vars
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    HostEnv::new(vars, home.map(Path::to_path_buf))
        .at(Utc.with_ymd_and_hms(2026, 10, 10, 12, 0, 0).unwrap())
}

fn write(home: &Path, rel: &str, contents: &str) {
    let path = home.join(rel);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, contents).unwrap();
}

fn claude_file(expires_ms: i64) -> String {
    format!(
        r#"{{"claudeAiOauth": {{"accessToken": "sk-ant-oat01-FAKE-file-access", "refreshToken": "sk-ant-ort01-FAKE-file-refresh", "expiresAt": {expires_ms}, "scopes": ["user:inference"]}}}}"#
    )
}

fn fake_jwt(claims: &str) -> String {
    let engine = base64::engine::general_purpose::URL_SAFE_NO_PAD;
    format!(
        "{}.{}.{}",
        engine.encode(r#"{"alg":"none"}"#),
        engine.encode(claims),
        "FAKE-signature-not-checked"
    )
}

#[test]
fn claude_prefers_the_api_key_like_claude_code() {
    let env = host(
        &[
            ("ANTHROPIC_API_KEY", FAKE_KEY),
            ("CLAUDE_CODE_OAUTH_TOKEN", FAKE_OAUTH),
        ],
        None,
    );
    let login = find_login(&agent("claude"), None, &env).unwrap();
    assert_eq!(login.label, "ANTHROPIC_API_KEY (environment)");
    // Only the chosen credential reaches the sandbox.
    assert_eq!(
        login.env,
        BTreeMap::from([("ANTHROPIC_API_KEY".to_string(), FAKE_KEY.to_string())])
    );
    assert_eq!(login.secrets, [FAKE_KEY]);
    assert!(login.files.is_empty());

    let status = login_status(&agent("claude"), None, &env);
    assert!(status.ready());
    assert_eq!(
        status.summary(),
        "ANTHROPIC_API_KEY (environment); also found CLAUDE_CODE_OAUTH_TOKEN; \
         ANTHROPIC_API_KEY overrides the OAuth token; unset it to run on the subscription"
    );
}

#[test]
fn claude_falls_through_oauth_alias_auth_token_then_file() {
    let env = host(&[("CLAUDE_OAUTH_TOKEN", FAKE_OAUTH)], None);
    let login = find_login(&agent("claude"), None, &env).unwrap();
    assert_eq!(login.label, "CLAUDE_OAUTH_TOKEN (environment)");
    // Claude Code reads the token under its current name.
    assert_eq!(
        login.env.keys().collect::<Vec<_>>(),
        ["CLAUDE_CODE_OAUTH_TOKEN"]
    );

    let env = host(&[("ANTHROPIC_AUTH_TOKEN", "gateway-token-FAKE")], None);
    assert_eq!(
        find_login(&agent("claude"), None, &env).unwrap().label,
        "ANTHROPIC_AUTH_TOKEN (environment)"
    );

    let home = tempfile::tempdir().unwrap();
    write(
        home.path(),
        ".claude/.credentials.json",
        &claude_file(1_893_456_000_000),
    );
    let env = host(&[("ANTHROPIC_API_KEY", "   ")], Some(home.path()));
    let login = find_login(&agent("claude"), None, &env).unwrap();
    assert_eq!(login.label, "host login (~/.claude/.credentials.json)");
    assert!(login.env.is_empty());
    assert_eq!(login.files.len(), 1);
    assert_eq!(login.files[0].path, "$HOME/.claude/.credentials.json");
    assert_eq!(login.files[0].mode, 0o600);
    assert_eq!(
        login.files[0].contents,
        claude_file(1_893_456_000_000).into_bytes()
    );
    let mut secrets = login.secrets.clone();
    secrets.sort();
    assert_eq!(
        secrets,
        [
            "sk-ant-oat01-FAKE-file-access",
            "sk-ant-ort01-FAKE-file-refresh"
        ]
    );
    assert!(login.notes.is_empty());
}

#[test]
fn an_expired_claude_file_is_used_with_a_warning() {
    let home = tempfile::tempdir().unwrap();
    // Expired 2026-10-01; "now" is 2026-10-10.
    write(
        home.path(),
        ".claude/.credentials.json",
        &claude_file(1_790_812_800_000),
    );
    let env = host(&[], Some(home.path()));
    let login = find_login(&agent("claude"), None, &env).unwrap();
    assert_eq!(login.notes.len(), 1);
    assert!(
        login.notes[0].starts_with("access token expired 2026-10-01 00:00 UTC"),
        "{:?}",
        login.notes
    );

    let status = login_status(&agent("claude"), None, &env);
    assert!(!status.ready());
    assert!(
        status
            .fix
            .as_deref()
            .unwrap()
            .contains("claude setup-token")
    );
    assert_eq!(status.sources[0].kind, SourceKind::LoginFile);
    assert_eq!(
        status.sources[0].expires_at.unwrap().to_rfc3339(),
        "2026-10-01T00:00:00+00:00"
    );
}

#[test]
fn a_claude_file_without_a_token_is_an_error_with_the_fix() {
    let home = tempfile::tempdir().unwrap();
    write(
        home.path(),
        ".claude/.credentials.json",
        r#"{"claudeAiOauth": {}}"#,
    );
    let err = find_login(&agent("claude"), None, &host(&[], Some(home.path()))).unwrap_err();
    assert_eq!(
        err.to_string(),
        "cannot use ~/.claude/.credentials.json as a login for agent `claude`: it holds no OAuth token; \
         next: run `claude setup-token`, then `export CLAUDE_CODE_OAUTH_TOKEN=<token>` \
         (or `export ANTHROPIC_API_KEY=<key>`)"
    );
    write(home.path(), ".claude/.credentials.json", "not json");
    let err = find_login(&agent("claude"), None, &host(&[], Some(home.path()))).unwrap_err();
    assert!(
        matches!(&err, CredentialError::BadFile { problem, .. } if problem == "it is not JSON")
    );
}

#[test]
fn no_claude_login_says_what_was_checked_and_what_to_run() {
    let empty_home = tempfile::tempdir().unwrap();
    let err = find_login(&agent("claude"), None, &host(&[], Some(empty_home.path()))).unwrap_err();
    assert_eq!(
        err.to_string(),
        "no login for agent `claude`: looked for ANTHROPIC_API_KEY, CLAUDE_CODE_OAUTH_TOKEN, \
         CLAUDE_OAUTH_TOKEN, ANTHROPIC_AUTH_TOKEN or ~/.claude/.credentials.json; next: run \
         `claude setup-token`, then `export CLAUDE_CODE_OAUTH_TOKEN=<token>` (or `export ANTHROPIC_API_KEY=<key>`)"
    );
    let status = login_status(&agent("claude"), None, &host(&[], None));
    assert_eq!(status.summary(), "no login found");
    assert!(!status.ready());
}

#[test]
fn explicit_overrides_win_and_say_so() {
    let env = host(&[("ANTHROPIC_API_KEY", FAKE_KEY)], None).with_overrides(BTreeMap::from([(
        "ANTHROPIC_API_KEY".to_string(),
        "sk-ant-api03-FAKE-override".to_string(),
    )]));
    let login = find_login(&agent("claude"), None, &env).unwrap();
    assert_eq!(login.label, "ANTHROPIC_API_KEY (agent_env)");
    assert_eq!(login.env["ANTHROPIC_API_KEY"], "sk-ant-api03-FAKE-override");
}

#[test]
fn a_blank_override_hides_an_inherited_key() {
    // Running on the subscription while the shell exports an API key.
    let env = host(
        &[
            ("ANTHROPIC_API_KEY", FAKE_KEY),
            ("CLAUDE_CODE_OAUTH_TOKEN", FAKE_OAUTH),
        ],
        None,
    )
    .with_overrides(BTreeMap::from([(
        "ANTHROPIC_API_KEY".to_string(),
        String::new(),
    )]));
    let login = find_login(&agent("claude"), None, &env).unwrap();
    assert_eq!(login.label, "CLAUDE_CODE_OAUTH_TOKEN (environment)");
    assert!(!login.env.contains_key("ANTHROPIC_API_KEY"));
}

#[test]
fn a_custom_label_names_pooled_accounts() {
    let env = host(
        &[
            ("CLAUDE_CODE_OAUTH_TOKEN", FAKE_OAUTH),
            (LOGIN_LABEL_ENV, "  pool\n account 7 "),
        ],
        None,
    );
    let login = find_login(&agent("claude"), None, &env).unwrap();
    assert_eq!(login.label, "pool account 7");
    assert!(!login.env.contains_key(LOGIN_LABEL_ENV));
}

#[test]
fn codex_bridges_its_key_and_copies_its_login_file() {
    let env = host(&[("CODEX_API_KEY", "sk-proj-FAKE-codex-key")], None);
    let login = find_login(&agent("codex"), None, &env).unwrap();
    assert_eq!(login.label, "CODEX_API_KEY (environment)");
    // The launcher writes auth.json from OPENAI_API_KEY (0.8's bridge).
    assert_eq!(login.env["OPENAI_API_KEY"], "sk-proj-FAKE-codex-key");
    assert_eq!(login.env["CODEX_API_KEY"], "sk-proj-FAKE-codex-key");

    let env = host(&[("CODEX_ACCESS_TOKEN", "FAKE-access-token-xyz")], None);
    let login = find_login(&agent("codex"), None, &env).unwrap();
    assert_eq!(login.env.keys().collect::<Vec<_>>(), ["CODEX_ACCESS_TOKEN"]);

    let home = tempfile::tempdir().unwrap();
    let jwt = fake_jwt(
        r#"{"exp": 1790812800, "https://api.openai.com/auth": {"chatgpt_plan_type": "plus"}}"#,
    );
    let auth = format!(
        r#"{{"auth_mode": "chatgpt", "OPENAI_API_KEY": null, "tokens": {{"id_token": "{jwt}", "access_token": "{jwt}", "refresh_token": "rt-FAKE-refresh-token", "account_id": "acct"}}}}"#
    );
    write(home.path(), ".codex/auth.json", &auth);
    let env = host(&[], Some(home.path()));
    let login = find_login(&agent("codex"), None, &env).unwrap();
    assert_eq!(login.label, "host login (~/.codex/auth.json)");
    assert_eq!(login.files[0].path, "$HOME/.codex/auth.json");
    assert!(login.secrets.contains(&jwt));
    assert!(login.secrets.contains(&"rt-FAKE-refresh-token".to_string()));
    assert!(!login.secrets.contains(&"acct".to_string()));

    let status = login_status(&agent("codex"), None, &env);
    // Codex refreshes its own access token, so expiry is not a problem.
    assert!(status.ready());
    assert_eq!(
        status.summary(),
        "~/.codex/auth.json: ChatGPT login (auth_mode=chatgpt), plan plus"
    );
}

#[test]
fn codex_takes_an_inline_auth_json_before_the_file() {
    let home = tempfile::tempdir().unwrap();
    write(
        home.path(),
        ".codex/auth.json",
        r#"{"OPENAI_API_KEY": "sk-proj-FAKE-from-file"}"#,
    );
    let inline = r#"{"OPENAI_API_KEY": "sk-proj-FAKE-inline-key"}"#;
    let env = host(&[("CODEX_AUTH_JSON", inline)], Some(home.path()));
    let login = find_login(&agent("codex"), None, &env).unwrap();
    assert_eq!(login.label, "CODEX_AUTH_JSON (environment)");
    assert_eq!(login.files[0].contents, inline.as_bytes());

    // A broken inline file is skipped for the next login, and the status says so.
    let env = host(&[("CODEX_AUTH_JSON", "{nope")], Some(home.path()));
    assert_eq!(
        find_login(&agent("codex"), None, &env).unwrap().label,
        "host login (~/.codex/auth.json)"
    );
    let status = login_status(&agent("codex"), None, &env);
    assert_eq!(
        status.summary(),
        "~/.codex/auth.json: API key stored in file; ignored CODEX_AUTH_JSON (environment): it is not JSON"
    );
}

#[test]
fn codex_without_a_login_points_at_codex_login() {
    let err = find_login(&agent("codex"), Some("gpt-5"), &host(&[], None)).unwrap_err();
    assert!(err.next_step().starts_with("run `codex login`"), "{err}");
    let home = tempfile::tempdir().unwrap();
    write(home.path(), ".codex/auth.json", r#"{"tokens": {}}"#);
    let err = find_login(&agent("codex"), None, &host(&[], Some(home.path()))).unwrap_err();
    assert!(
        err.to_string().contains("it holds no ChatGPT tokens"),
        "{err}"
    );
}

#[test]
fn gemini_mirrors_its_two_key_names() {
    for name in ["GEMINI_API_KEY", "GOOGLE_API_KEY"] {
        let env = host(&[(name, "AIzaSyFAKE-gemini-key-000000000")], None);
        let login = find_login(&agent("gemini"), None, &env).unwrap();
        assert_eq!(login.label, format!("{name} (environment)"));
        assert_eq!(
            login.env["GEMINI_API_KEY"],
            "AIzaSyFAKE-gemini-key-000000000"
        );
        assert_eq!(
            login.env["GOOGLE_API_KEY"],
            "AIzaSyFAKE-gemini-key-000000000"
        );
        assert_eq!(login.secrets.len(), 1);
    }
}

#[test]
fn gemini_copies_its_google_sign_in_files() {
    let home = tempfile::tempdir().unwrap();
    write(
        home.path(),
        ".gemini/oauth_creds.json",
        r#"{"access_token": "ya29.FAKE-access", "refresh_token": "1//FAKE-refresh-token", "expiry_date": 1790812800000}"#,
    );
    write(
        home.path(),
        ".gemini/settings.json",
        r#"{"security": {"auth": {"selectedType": "oauth-personal"}}}"#,
    );
    let env = host(&[], Some(home.path()));
    let login = find_login(&agent("gemini"), None, &env).unwrap();
    assert_eq!(login.label, "host login (~/.gemini/oauth_creds.json)");
    let paths: Vec<&str> = login.files.iter().map(|f| f.path.as_str()).collect();
    assert_eq!(
        paths,
        [
            "$HOME/.gemini/oauth_creds.json",
            "$HOME/.gemini/settings.json"
        ]
    );
    assert_eq!(login.secrets.len(), 2);

    let err = find_login(&agent("gemini"), None, &host(&[], None)).unwrap_err();
    assert!(
        err.to_string().starts_with(
            "no login for agent `gemini`: looked for GEMINI_API_KEY, GOOGLE_API_KEY or ~/.gemini/oauth_creds.json; \
             next: `export GEMINI_API_KEY=<key>`"
        ),
        "{err}"
    );
}

#[test]
fn opencode_reads_the_key_of_the_models_provider() {
    let env = host(
        &[
            ("ANTHROPIC_API_KEY", FAKE_KEY),
            ("OPENAI_API_KEY", "sk-proj-FAKE-openai"),
            ("GEMINI_API_KEY", "AIzaSyFAKE-gemini-key-000000000"),
            ("OPENROUTER_API_KEY", "sk-or-v1-FAKE-openrouter"),
        ],
        None,
    );
    let opencode = agent("opencode");
    let label = |model: &str| find_login(&opencode, Some(model), &env).unwrap();
    assert_eq!(
        label("anthropic/claude-sonnet-4-6").label,
        "ANTHROPIC_API_KEY (environment)"
    );
    assert_eq!(label("gpt-5").label, "OPENAI_API_KEY (environment)");
    assert_eq!(
        label("openrouter/qwen/qwen3-coder").label,
        "OPENROUTER_API_KEY (environment)"
    );
    let google = label("google/gemini-3.1-pro-preview");
    assert_eq!(google.label, "GEMINI_API_KEY (environment)");
    // OpenCode's google provider reads GOOGLE_GENERATIVE_AI_API_KEY.
    assert_eq!(
        google.env.keys().collect::<Vec<_>>(),
        [
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "GOOGLE_GENERATIVE_AI_API_KEY"
        ]
    );
    // Only the provider's key goes in.
    assert_eq!(label("anthropic/claude-sonnet-4-6").env.len(), 1);

    let err = find_login(&opencode, Some("xai/grok-5"), &env).unwrap_err();
    assert_eq!(err.next_step(), "`export XAI_API_KEY=<key>`");
    let err = find_login(&opencode, Some("qwen3-coder"), &env).unwrap_err();
    assert!(
        matches!(err, CredentialError::UnknownProvider { .. }),
        "{err}"
    );
    assert!(
        err.to_string().contains("`qwen3-coder` names none"),
        "{err}"
    );
    let err = find_login(&opencode, None, &env).unwrap_err();
    assert!(err.to_string().contains("no model was given"), "{err}");
    let err = find_login(
        &opencode,
        Some("anthropic/claude-x"),
        &host(&[("CLAUDE_CODE_OAUTH_TOKEN", FAKE_OAUTH)], None),
    )
    .unwrap_err();
    assert!(
        err.next_step()
            .contains("a Claude subscription token does not work"),
        "{err}"
    );
}

#[test]
fn env_and_none_families() {
    let custom = AgentManifest::parse(
        "name = \"mine\"\n[install]\nscript = \"true\"\n[launch]\ncommand = \"mine\"\n[credentials]\nenv = [\"MINE_KEY\", \"MINE_TOKEN\"]\n",
        ManifestSource::Builtin("mine.toml"),
    )
    .unwrap();
    let login = find_login(
        &custom,
        None,
        &host(&[("MINE_TOKEN", "FAKE-mine-token")], None),
    )
    .unwrap();
    assert_eq!(login.label, "MINE_TOKEN (environment)");
    let err = find_login(&custom, None, &host(&[], None)).unwrap_err();
    assert_eq!(err.next_step(), "`export MINE_KEY=<key>`");

    let free = AgentManifest::parse(
        "name = \"free\"\n[install]\nscript = \"true\"\n[launch]\ncommand = \"free\"\n",
        ManifestSource::Builtin("free.toml"),
    )
    .unwrap();
    let login = find_login(&free, None, &host(&[], None)).unwrap();
    assert!(login.env.is_empty() && login.secrets.is_empty());
    assert_eq!(
        login_status(&free, None, &host(&[], None)).notes,
        ["needs no login from BenchFlow"]
    );
}

#[test]
fn sandbox_files_resolve_under_the_agent_home() {
    let home = tempfile::tempdir().unwrap();
    write(
        home.path(),
        ".claude/.credentials.json",
        &claude_file(1_893_456_000_000),
    );
    let login = find_login(&agent("claude"), None, &host(&[], Some(home.path()))).unwrap();
    assert_eq!(
        login.files[0].path_for("/home/agent"),
        "/home/agent/.claude/.credentials.json"
    );
    assert_eq!(
        login.files[0].path_for("/root/"),
        "/root/.claude/.credentials.json"
    );
}

#[test]
fn short_values_are_not_redacted_everywhere() {
    let login = find_login(
        &agent("claude"),
        None,
        &host(&[("ANTHROPIC_API_KEY", "abc")], None),
    )
    .unwrap();
    assert!(login.secrets.is_empty());
}

#[test]
fn debug_output_never_shows_values() {
    let home = tempfile::tempdir().unwrap();
    write(
        home.path(),
        ".claude/.credentials.json",
        &claude_file(1_893_456_000_000),
    );
    let env = host(
        &[("CLAUDE_CODE_OAUTH_TOKEN", FAKE_OAUTH)],
        Some(home.path()),
    );
    let login = find_login(&agent("claude"), None, &env).unwrap();
    let shown = format!(
        "{login:?} {env:?} {:?}",
        login_status(&agent("claude"), None, &env)
    );
    assert!(!shown.contains(FAKE_OAUTH), "{shown}");
    assert!(!shown.contains("FAKE-file"), "{shown}");
    let json = serde_json::to_string(&login_status(&agent("claude"), None, &env)).unwrap();
    assert!(
        !json.contains(FAKE_OAUTH) && !json.contains("FAKE-file"),
        "{json}"
    );
}
