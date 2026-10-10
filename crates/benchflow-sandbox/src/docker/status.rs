//! `bench doctor`'s view of Docker: is the CLI there, and does its daemon
//! answer?

use std::time::Duration;

use serde_json::Value;

use super::cli::{Cli, Flavor};
use crate::{ProviderStatus, SandboxError};

const STATUS_TIMEOUT: Duration = Duration::from_secs(20);

/// Ask `docker version` and describe the answer in one line.
pub(super) async fn status(cli: &Cli) -> ProviderStatus {
    let args = ["version", "--format", "{{json .}}"].map(String::from);
    let output = match cli.run(&args, Some(STATUS_TIMEOUT)).await {
        Ok(output) => output,
        Err(err) => return not_ok(err),
    };
    if let Some(err) = cli.daemon_problem(&output) {
        return not_ok(err);
    }
    let json: Option<Value> = output
        .stdout
        .lines()
        .find(|l| l.trim_start().starts_with('{'))
        .and_then(|l| serde_json::from_str(l).ok());
    let Some(json) = json else {
        return ProviderStatus {
            ok: false,
            summary: format!(
                "`{} version` gave no version information: {}",
                cli.name,
                output.message()
            ),
            fix: Some(format!(
                "Run `{} version` yourself and fix what it reports.",
                cli.name
            )),
        };
    };
    describe(cli, &json, output.ok(), &output.message())
}

/// The runtimes the daemon has registered (`runc`, `runsc`, ...), from
/// `docker info`. Podman reports runtimes differently and accepts any OCI
/// runtime on its `PATH`, so it gives `None` (no list to check against).
pub(super) async fn runtimes(cli: &Cli) -> Result<Option<Vec<String>>, SandboxError> {
    if cli.flavor == Flavor::Podman {
        return Ok(None);
    }
    let args = [
        "info",
        "--format",
        "{{range $name, $_ := .Runtimes}}{{$name}}\n{{end}}",
    ]
    .map(String::from);
    let output = cli.run(&args, Some(STATUS_TIMEOUT)).await?;
    if let Some(err) = cli.daemon_problem(&output) {
        return Err(err);
    }
    if !output.ok() {
        return Err(SandboxError::Transport(format!(
            "`{} info` failed: {}",
            cli.name,
            output.message()
        )));
    }
    Ok(Some(
        output
            .stdout
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .map(str::to_string)
            .collect(),
    ))
}

fn describe(cli: &Cli, json: &Value, exited_ok: bool, message: &str) -> ProviderStatus {
    let server = json
        .get("Server")
        .and_then(|s| s.get("Version"))
        .and_then(Value::as_str);
    let client = json
        .get("Client")
        .and_then(|c| c.get("Version"))
        .and_then(Value::as_str);
    let context = json
        .get("Client")
        .and_then(|c| c.get("Context"))
        .and_then(Value::as_str);
    let host = std::env::var("DOCKER_HOST").ok().filter(|h| !h.is_empty());

    match (cli.flavor, server) {
        (_, Some(version)) if exited_ok => {
            let product = match cli.flavor {
                Flavor::Docker => "Docker",
                Flavor::Podman => "Podman",
            };
            let mut summary = format!("{product} {version} (server)");
            if let Some(context) = context {
                summary.push_str(&format!(", context {context}"));
            }
            if let Some(host) = host {
                summary.push_str(&format!(", DOCKER_HOST {host}"));
            }
            ProviderStatus {
                ok: true,
                summary,
                fix: None,
            }
        }
        // Podman on Linux runs without a service; the client is enough.
        (Flavor::Podman, None) if exited_ok => ProviderStatus {
            ok: true,
            summary: format!("Podman {} (local)", client.unwrap_or("unknown version")),
            fix: None,
        },
        _ => ProviderStatus {
            ok: false,
            summary: format!(
                "the {} CLI {} answered, but its daemon did not: {message}",
                cli.name,
                client.unwrap_or("")
            ),
            fix: Some(match cli.flavor {
                Flavor::Docker => "Start Docker (Docker Desktop, OrbStack, or `sudo systemctl start docker`) and \
                    check with `docker version`. For a remote daemon, check DOCKER_HOST and `docker context ls`."
                    .to_string(),
                Flavor::Podman => "Start podman's service (`podman machine start` on macOS) and check with \
                    `podman version`."
                    .to_string(),
            }),
        },
    }
}

fn not_ok(err: SandboxError) -> ProviderStatus {
    match err {
        SandboxError::Unavailable { message, fix } => ProviderStatus {
            ok: false,
            summary: message,
            fix: Some(fix),
        },
        other => ProviderStatus {
            ok: false,
            summary: other.to_string(),
            fix: Some("Run `docker version` yourself and fix what it reports.".to_string()),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cli(flavor: Flavor) -> Cli {
        Cli {
            name: match flavor {
                Flavor::Docker => "docker",
                Flavor::Podman => "podman",
            }
            .into(),
            binary: "x".into(),
            flavor,
        }
    }

    #[test]
    fn a_healthy_docker_names_version_and_context() {
        let json: Value = serde_json::from_str(
            r#"{"Client":{"Version":"28.1.1","Context":"default"},"Server":{"Version":"28.1.1"}}"#,
        )
        .unwrap();
        let s = describe(&cli(Flavor::Docker), &json, true, "");
        assert!(s.ok);
        assert!(
            s.summary
                .starts_with("Docker 28.1.1 (server), context default"),
            "{}",
            s.summary
        );
    }

    #[test]
    fn docker_without_a_server_is_not_ok() {
        let json: Value =
            serde_json::from_str(r#"{"Client":{"Version":"28.1.1"},"Server":null}"#).unwrap();
        let s = describe(&cli(Flavor::Docker), &json, false, "cannot connect");
        assert!(!s.ok);
        assert!(s.fix.unwrap().contains("Start Docker"));
    }

    #[test]
    fn podman_without_a_service_is_fine() {
        let json: Value = serde_json::from_str(r#"{"Client":{"Version":"5.2.1"}}"#).unwrap();
        let s = describe(&cli(Flavor::Podman), &json, true, "");
        assert!(s.ok);
        assert_eq!(s.summary, "Podman 5.2.1 (local)");
    }
}
