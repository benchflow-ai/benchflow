//! Finding credential files a task leaves where its agent could find them:
//! in the task's build folder (on this machine) or in its image (inside a
//! sandbox, before the agent starts).
//!
//! A finding names the file and what kind of credential it holds, never its
//! content. The same rules drive both scans: [`scan_dir`] walks a folder
//! here, and [`scan_program`] is the shell program that walks a sandbox.

use std::path::Path;

use serde::{Deserialize, Serialize};

/// Where a rule looks.
#[derive(Debug, Clone, Copy)]
enum Place {
    /// A file with this name, anywhere.
    Name(&'static str),
    /// A file whose path ends with this (`.aws/credentials`).
    Suffix(&'static str),
    /// Any file inside a folder whose path ends with this (`.ssh`).
    InDir(&'static str),
}

/// One kind of credential file.
struct Rule {
    /// What it is, for people (`a .netrc file with host logins`).
    kind: &'static str,
    place: Place,
    /// The file counts only when it contains one of these; `None`: any
    /// non-empty file counts.
    needles: Option<&'static [&'static str]>,
}

const SSH_KEY: &[&str] = &["PRIVATE KEY-----"];

static RULES: &[Rule] = &[
    Rule {
        kind: "a .netrc file with host logins",
        place: Place::Name(".netrc"),
        needles: None,
    },
    Rule {
        kind: "a _netrc file with host logins",
        place: Place::Name("_netrc"),
        needles: None,
    },
    Rule {
        kind: "a .git-credentials file with git logins",
        place: Place::Name(".git-credentials"),
        needles: None,
    },
    Rule {
        kind: "AWS credentials",
        place: Place::Suffix(".aws/credentials"),
        needles: None,
    },
    Rule {
        kind: "a Docker config with registry logins",
        place: Place::Suffix(".docker/config.json"),
        needles: Some(&["\"auth\"", "\"identitytoken\"", "\"registrytoken\""]),
    },
    Rule {
        kind: "an .npmrc with an auth token",
        place: Place::Name(".npmrc"),
        needles: Some(&["_authToken", "_auth=", "_auth =", "_password"]),
    },
    Rule {
        kind: "a .pypirc with a password",
        place: Place::Name(".pypirc"),
        needles: Some(&["password"]),
    },
    Rule {
        kind: "a private SSH key",
        place: Place::Name("id_rsa"),
        needles: Some(SSH_KEY),
    },
    Rule {
        kind: "a private SSH key",
        place: Place::Name("id_dsa"),
        needles: Some(SSH_KEY),
    },
    Rule {
        kind: "a private SSH key",
        place: Place::Name("id_ecdsa"),
        needles: Some(SSH_KEY),
    },
    Rule {
        kind: "a private SSH key",
        place: Place::Name("id_ed25519"),
        needles: Some(SSH_KEY),
    },
    Rule {
        kind: "a private SSH key",
        place: Place::InDir(".ssh"),
        needles: Some(SSH_KEY),
    },
    Rule {
        kind: "Google Cloud credentials",
        place: Place::Suffix(".config/gcloud/application_default_credentials.json"),
        needles: None,
    },
    Rule {
        kind: "Google Cloud credentials",
        place: Place::Suffix(".config/gcloud/credentials.db"),
        needles: None,
    },
    Rule {
        kind: "Google Cloud credentials",
        place: Place::Suffix(".config/gcloud/access_tokens.db"),
        needles: None,
    },
    Rule {
        kind: "Google Cloud credentials",
        place: Place::InDir(".config/gcloud/legacy_credentials"),
        needles: None,
    },
    Rule {
        kind: "Azure CLI tokens",
        place: Place::Suffix(".azure/accessTokens.json"),
        needles: None,
    },
    Rule {
        kind: "Azure CLI tokens",
        place: Place::Suffix(".azure/msal_token_cache.json"),
        needles: None,
    },
    Rule {
        kind: "a kubeconfig with credentials",
        place: Place::Suffix(".kube/config"),
        needles: Some(&["token:", "client-key-data", "password:"]),
    },
    Rule {
        kind: "a GitHub CLI token",
        place: Place::Suffix(".config/gh/hosts.yml"),
        needles: Some(&["oauth_token"]),
    },
    Rule {
        kind: "a Hugging Face token",
        place: Place::Suffix(".huggingface/token"),
        needles: None,
    },
    Rule {
        kind: "a Hugging Face token",
        place: Place::Suffix(".cache/huggingface/token"),
        needles: None,
    },
    Rule {
        kind: "Terraform Cloud credentials",
        place: Place::Suffix(".terraform.d/credentials.tfrc.json"),
        needles: None,
    },
];

/// Files over this size are not credential files; they are not read.
const MAX_BYTES: u64 = 1024 * 1024;

/// Where a finding was made.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum Location {
    /// In the task's build folder, which the image build receives.
    BuildFolder,
    /// In the task's image, found in a sandbox before the agent started.
    Image,
}

/// A credential file, by path and kind; never its content.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Finding {
    pub path: String,
    pub kind: String,
    pub location: Location,
    /// Whether the agent user could read it (image findings only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_can_read: Option<bool>,
}

impl Finding {
    /// What to do about it.
    pub fn fix(&self) -> &'static str {
        match self.location {
            Location::BuildFolder => {
                "move it out of the task's build folder, or list it in .dockerignore so the build never receives it"
            }
            Location::Image => {
                "delete it in the same Dockerfile RUN step that uses it, or pass it as a build secret (RUN --mount=type=secret); a file deleted in a later step stays in the image's layers"
            }
        }
    }
}

/// The rule a path (`/`-separated, absolute or relative) falls under, by
/// place alone.
fn rule_for(path: &str) -> Option<&'static Rule> {
    let path = format!("/{}", path.trim_start_matches('/'));
    RULES.iter().find(|r| match r.place {
        Place::Name(n) => path.rsplit('/').next() == Some(n),
        Place::Suffix(s) => path.ends_with(&format!("/{s}")),
        Place::InDir(d) => path.rsplit_once('/').is_some_and(|(dir, _)| {
            dir.ends_with(&format!("/{d}")) || dir.contains(&format!("/{d}/"))
        }),
    })
}

/// Files agent CLIs keep logins in, by name: Claude Code's
/// `.credentials.json`, Codex's and OpenCode's `auth.json`, Gemini CLI's
/// `oauth_creds.json`, goose's `secrets.yaml`, Devin's `credentials.toml`.
const LOGIN_FILE_NAMES: &[&str] = &[
    ".credentials.json",
    "credentials.json",
    "credentials.toml",
    "auth.json",
    "mcp-auth.json",
    "oauth_creds.json",
    "google_accounts.json",
    "secrets.yaml",
    "secrets.json",
    ".env",
];

/// Whether the file at `path` (`/`-separated) is a login file, by its path
/// alone: an agent CLI's login file, or one the credential scan looks for.
/// Such files are never saved from an agent's session folders.
pub fn is_login_file(path: &str) -> bool {
    let name = path.rsplit('/').next().unwrap_or(path);
    LOGIN_FILE_NAMES.contains(&name) || rule_for(path).is_some()
}

/// Whether `bytes` hold what `rule` looks for.
fn holds_secret(rule: &Rule, bytes: &[u8]) -> bool {
    if bytes.iter().all(u8::is_ascii_whitespace) {
        return false;
    }
    match rule.needles {
        None => true,
        Some(needles) => {
            let text = String::from_utf8_lossy(bytes);
            needles.iter().any(|n| text.contains(n))
        }
    }
}

/// Scan a task's build folder (here, not in a sandbox), skipping what
/// `excluded` says the build never receives (`.dockerignore`). Paths come
/// back relative to `dir`.
pub fn scan_dir(dir: &Path, excluded: &dyn Fn(&str) -> bool) -> Vec<Finding> {
    let mut out = Vec::new();
    for entry in walkdir::WalkDir::new(dir)
        .follow_links(false)
        .into_iter()
        .filter_entry(|e| e.depth() == 0 || e.file_name() != ".git")
        .filter_map(Result::ok)
    {
        if !entry.file_type().is_file() {
            continue;
        }
        let Ok(rel) = entry.path().strip_prefix(dir) else {
            continue;
        };
        let rel = rel.to_string_lossy().replace('\\', "/");
        let Some(rule) = rule_for(&rel) else {
            continue;
        };
        if excluded(&rel) {
            continue;
        }
        if entry.metadata().is_ok_and(|m| m.len() > MAX_BYTES) {
            continue;
        }
        let Ok(bytes) = std::fs::read(entry.path()) else {
            continue;
        };
        if holds_secret(rule, &bytes) {
            out.push(Finding {
                path: rel,
                kind: rule.kind.to_string(),
                location: Location::BuildFolder,
                agent_can_read: None,
            });
        }
    }
    out.sort_by(|a, b| a.path.cmp(&b.path));
    out
}

/// Scan the folder a task's image is built from, when it is built (a
/// pulled image has none). Paths come back relative to the task's folder.
pub fn scan_task_build(task: &benchflow_task::TaskPackage) -> Vec<Finding> {
    let benchflow_task::package::ImageRef::Build {
        context,
        dockerfile,
    } = &task.sandbox.image
    else {
        return Vec::new();
    };
    // task.md packages hold these under the task's folder; split-layout
    // packages hold them relative to it.
    let under_root = |p: &Path| {
        if p.is_absolute() || p.starts_with(&task.root) {
            p.to_path_buf()
        } else {
            task.path(p)
        }
    };
    let (context, dockerfile) = (under_root(context), under_root(dockerfile));
    let excluded = benchflow_sandbox::dockerignore(&context, &dockerfile);
    let mut found = scan_dir(&context, &excluded);
    if let Ok(rel) = context.strip_prefix(&task.root) {
        for f in &mut found {
            f.path = rel.join(&f.path).to_string_lossy().replace('\\', "/");
        }
    }
    found
}

/// The shell program, run as root in a sandbox, that finds credential files
/// in its filesystem. Prints `BF credential <path>` for each; reads each
/// candidate only to test it, and prints nothing of its content.
pub fn scan_program() -> String {
    let mut names = Vec::new();
    let mut cases = Vec::new();
    for (i, r) in RULES.iter().enumerate() {
        let (find, glob) = match r.place {
            Place::Name(n) => (format!("-name {}", crate::shell_quote(n)), format!("*/{n}")),
            Place::Suffix(s) => (
                format!("-path {}", crate::shell_quote(&format!("*/{s}"))),
                format!("*/{s}"),
            ),
            Place::InDir(d) => (
                format!("-path {}", crate::shell_quote(&format!("*/{d}/*"))),
                format!("*/{d}/*"),
            ),
        };
        names.push(find);
        let test = match r.needles {
            None => "grep -q '[^[:space:]]' \"$f\"".to_string(),
            Some(needles) => {
                let pats: Vec<String> = needles
                    .iter()
                    .map(|n| format!("-e {}", crate::shell_quote(n)))
                    .collect();
                format!("grep -qF {} \"$f\"", pats.join(" "))
            }
        };
        cases.push(format!(
            "  {glob}) {test} && {{ echo \"BF credential {i} $f\"; continue; }};;",
            glob = glob.replace(' ', "\\ "),
        ));
    }
    let prune = [
        "/proc",
        "/sys",
        "/dev",
        "/run",
        crate::hardening::STATE_DIR,
        "/usr/share/doc",
        "/usr/share/man",
    ]
    .iter()
    .map(|d| format!("-path {}", crate::shell_quote(d)))
    .collect::<Vec<_>>()
    .join(" -o ");
    format!(
        r#"find / -xdev \( {prune} \) -prune -o -type f -size -{kb}k \( {names} \) -print 2>/dev/null | while IFS= read -r f; do
  case "$f" in
{cases}
  esac
done
true"#,
        names = names.join(" -o "),
        cases = cases.join("\n"),
        kb = MAX_BYTES / 1024 + 1,
    )
}

/// Read [`scan_program`]'s output into findings.
pub fn parse_scan(output: &str) -> Vec<Finding> {
    let mut out: Vec<Finding> = crate::hardening::Report::parse(output)
        .all("credential")
        .into_iter()
        .filter_map(|line| {
            let (i, path) = line.split_once(' ')?;
            let rule = RULES.get(i.parse::<usize>().ok()?)?;
            Some(Finding {
                path: path.to_string(),
                kind: rule.kind.to_string(),
                location: Location::Image,
                agent_can_read: None,
            })
        })
        .collect();
    out.sort_by(|a, b| a.path.cmp(&b.path));
    out.dedup_by(|a, b| a.path == b.path);
    out
}

/// A program, run as the agent user, that prints `BF readable <path>` for
/// each of `paths` it can read.
pub fn readable_program(paths: &[String]) -> String {
    let list: Vec<String> = paths.iter().map(|p| crate::shell_quote(p)).collect();
    format!(
        "for f in {}; do [ -r \"$f\" ] && echo \"BF readable $f\"; done; true",
        list.join(" ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn login_files_are_told_from_session_files() {
        for login in [
            ".local/share/opencode/auth.json",
            ".claude/.credentials.json",
            ".gemini/oauth_creds.json",
            "x/.netrc",
            ".aws/credentials",
        ] {
            assert!(is_login_file(login), "{login}");
        }
        for session in [
            ".claude/projects/-app/0b6e2f1c.jsonl",
            ".codex/sessions/2026/10/10/rollout-2026-10-10T09-00-00-0199.jsonl",
            ".local/share/opencode/storage/session/info/ses_1.json",
            ".local/share/goose/sessions/sessions.db",
        ] {
            assert!(!is_login_file(session), "{session}");
        }
    }

    #[test]
    fn rules_match_by_place() {
        assert_eq!(
            rule_for("root/.netrc").unwrap().kind,
            "a .netrc file with host logins"
        );
        assert_eq!(
            rule_for(".netrc").unwrap().kind,
            "a .netrc file with host logins"
        );
        assert_eq!(
            rule_for("/home/a/.aws/credentials").unwrap().kind,
            "AWS credentials"
        );
        assert!(rule_for("/home/a/.aws/config").is_none());
        assert_eq!(
            rule_for("/home/a/.ssh/deploy").unwrap().kind,
            "a private SSH key"
        );
        assert!(rule_for("/home/a/ssh/deploy").is_none());
        assert_eq!(
            rule_for("/root/.config/gcloud/legacy_credentials/me@x/adc.json")
                .unwrap()
                .kind,
            "Google Cloud credentials"
        );
        assert!(rule_for("/etc/ssh/ssh_host_ed25519_key").is_none());
    }

    #[test]
    fn content_decides_where_a_rule_needs_it() {
        let docker = rule_for(".docker/config.json").unwrap();
        assert!(!holds_secret(
            docker,
            br#"{"auths": {}, "credsStore": "desktop"}"#
        ));
        assert!(holds_secret(
            docker,
            br#"{"auths": {"r": {"auth": "eA=="}}}"#
        ));
        let npm = rule_for(".npmrc").unwrap();
        assert!(!holds_secret(
            npm,
            b"registry=https://registry.npmjs.org/\n"
        ));
        assert!(holds_secret(npm, b"//registry.npmjs.org/:_authToken=x\n"));
        let netrc = rule_for(".netrc").unwrap();
        assert!(!holds_secret(netrc, b"\n  \n"));
        assert!(holds_secret(
            netrc,
            b"machine example.com login a password b\n"
        ));
    }

    #[test]
    fn a_build_folder_scan_names_files_and_honors_exclusions() {
        let d = tempfile::tempdir().unwrap();
        std::fs::write(d.path().join(".netrc"), "machine x login a password b\n").unwrap();
        std::fs::create_dir_all(d.path().join("keys/.ssh")).unwrap();
        std::fs::write(
            d.path().join("keys/.ssh/id_ed25519"),
            "-----BEGIN OPENSSH PRIVATE KEY-----\nAAAA\n",
        )
        .unwrap();
        std::fs::write(
            d.path().join("keys/.ssh/id_ed25519.pub"),
            "ssh-ed25519 AAAA\n",
        )
        .unwrap();
        std::fs::write(d.path().join(".npmrc"), "registry=https://x/\n").unwrap();
        let all = scan_dir(d.path(), &|_| false);
        let paths: Vec<&str> = all.iter().map(|f| f.path.as_str()).collect();
        assert_eq!(paths, [".netrc", "keys/.ssh/id_ed25519"]);
        assert!(all.iter().all(|f| f.location == Location::BuildFolder));
        let kept = scan_dir(d.path(), &|rel| rel.starts_with("keys"));
        assert_eq!(kept.len(), 1);
    }

    #[test]
    fn the_sandbox_scan_is_valid_shell_and_parses_back() {
        let out = std::process::Command::new("sh")
            .arg("-n")
            .arg("-c")
            .arg(scan_program())
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let f = parse_scan(
            "BF credential 0 /root/.netrc\nBF credential 3 /home/a/.aws/credentials\nBF credential 99 /x\n",
        );
        assert_eq!(f.len(), 2);
        assert_eq!(f[0].kind, "AWS credentials");
        assert_eq!(f[1].path, "/root/.netrc");
        assert_eq!(f[1].location, Location::Image);
    }
}
