//! The Daytona API's request and response shapes that BenchFlow uses.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::api::Secret;

/// `POST /sandbox`: from a recipe (`build_info`), or from a snapshot
/// (`snapshot`, with no build info and no resources).
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CreateSandbox<'a> {
    /// The snapshot's name (or id) to start from.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snapshot: Option<&'a str>,
    pub labels: BTreeMap<String, String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu: Option<u32>,
    /// GiB.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory: Option<u32>,
    /// GiB.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub disk: Option<u32>,
    pub network_block_all: bool,
    /// Comma-separated CIDR ranges the sandbox may reach; Daytona takes at
    /// most [`super::MAX_ALLOW_CIDRS`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub network_allow_list: Option<String>,
    /// Minutes without activity before Daytona stops the sandbox.
    pub auto_stop_interval: u32,
    /// Minutes after a stop before Daytona deletes it; 0 deletes at once.
    pub auto_delete_interval: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub build_info: Option<BuildInfo<'a>>,
}

/// `POST /sandbox/{id}/network-settings`.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct NetworkSettings {
    pub network_block_all: bool,
    /// Comma-separated CIDR ranges; empty lifts the list.
    pub network_allow_list: String,
}

/// How Daytona builds the sandbox's image.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct BuildInfo<'a> {
    pub dockerfile_content: &'a str,
    pub context_hashes: &'a [String],
}

/// A sandbox, as `GET /sandbox/{id}` and the listing return it.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SandboxDto {
    pub id: String,
    #[serde(default)]
    pub state: Option<String>,
    #[serde(default)]
    pub error_reason: Option<String>,
    #[serde(default)]
    pub labels: BTreeMap<String, String>,
    #[serde(default)]
    pub toolbox_proxy_url: Option<String>,
    #[serde(default)]
    pub created_at: Option<String>,
    #[serde(default)]
    pub last_activity_at: Option<String>,
    /// CPUs, and GiB of memory and disk.
    #[serde(default)]
    pub cpu: Option<f64>,
    #[serde(default)]
    pub memory: Option<f64>,
    #[serde(default)]
    pub disk: Option<f64>,
}

impl SandboxDto {
    pub(crate) fn state(&self) -> &str {
        self.state.as_deref().unwrap_or("unknown")
    }
}

/// `POST /sandbox/{id}/snapshot`.
#[derive(Debug, Serialize)]
pub(crate) struct CreateSnapshot<'a> {
    pub name: &'a str,
}

/// A snapshot, as `GET /snapshots/{id or name}` and the listing return it.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SnapshotDto {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub state: Option<String>,
    /// GB, once Daytona has measured it.
    #[serde(default)]
    pub size: Option<f64>,
    #[serde(default)]
    pub error_reason: Option<String>,
    #[serde(default)]
    pub created_at: Option<String>,
}

impl SnapshotDto {
    pub(crate) fn state(&self) -> &str {
        self.state.as_deref().unwrap_or("unknown")
    }
}

/// One page of `GET /snapshots`.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SnapshotPage {
    #[serde(default)]
    pub items: Vec<SnapshotDto>,
    #[serde(default)]
    pub total_pages: Option<f64>,
}

/// One page of `GET /sandbox`.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SandboxPage {
    #[serde(default)]
    pub items: Vec<SandboxDto>,
    #[serde(default)]
    pub next_cursor: Option<String>,
}

/// `GET /object-storage/push-access`: short-lived credentials for uploading
/// a build context.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct PushAccess {
    pub access_key: String,
    pub secret: Secret,
    pub session_token: Secret,
    pub storage_url: String,
    pub organization_id: String,
    pub bucket: String,
    #[serde(default)]
    pub region: Option<String>,
}

/// An answer that is just a URL.
#[derive(Debug, Deserialize)]
pub(crate) struct UrlDto {
    pub url: String,
}

/// `POST /process/session/{id}/exec`.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SessionExec<'a> {
    pub command: &'a str,
    pub run_async: bool,
    pub suppress_input_echo: bool,
}

/// The answer to [`SessionExec`].
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SessionExecAnswer {
    #[serde(default)]
    pub cmd_id: Option<String>,
    #[serde(default)]
    pub exit_code: Option<i32>,
    #[serde(default)]
    pub output: Option<String>,
}

/// `GET /process/session/{id}/command/{cmd}`.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CommandDto {
    #[serde(default)]
    pub exit_code: Option<i32>,
}

/// `GET /process/session/{id}/command/{cmd}/logs` as JSON.
#[derive(Debug, Default, Deserialize)]
pub(crate) struct LogsDto {
    #[serde(default)]
    pub stdout: Option<String>,
    #[serde(default)]
    pub stderr: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_body_uses_daytonas_field_names() {
        let hashes = vec!["abc".to_string()];
        let body = CreateSandbox {
            snapshot: None,
            labels: BTreeMap::from([("k".to_string(), "v".to_string())]),
            target: None,
            cpu: Some(2),
            memory: Some(4),
            disk: None,
            network_block_all: true,
            network_allow_list: None,
            auto_stop_interval: 15,
            auto_delete_interval: 0,
            build_info: Some(BuildInfo {
                dockerfile_content: "FROM x\n",
                context_hashes: &hashes,
            }),
        };
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "labels": {"k": "v"},
                "cpu": 2,
                "memory": 4,
                "networkBlockAll": true,
                "autoStopInterval": 15,
                "autoDeleteInterval": 0,
                "buildInfo": {"dockerfileContent": "FROM x\n", "contextHashes": ["abc"]},
            })
        );
    }

    #[test]
    fn an_allow_list_is_one_comma_separated_string() {
        let hashes: Vec<String> = Vec::new();
        let body = CreateSandbox {
            snapshot: None,
            labels: BTreeMap::new(),
            target: None,
            cpu: None,
            memory: None,
            disk: None,
            network_block_all: false,
            network_allow_list: Some("140.82.114.4/32,10.0.0.0/8".into()),
            auto_stop_interval: 15,
            auto_delete_interval: 0,
            build_info: Some(BuildInfo {
                dockerfile_content: "FROM x\n",
                context_hashes: &hashes,
            }),
        };
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(json["networkAllowList"], "140.82.114.4/32,10.0.0.0/8");
        assert_eq!(json["networkBlockAll"], false);
    }

    #[test]
    fn push_access_secrets_do_not_print() {
        let access: PushAccess = serde_json::from_str(
            r#"{"accessKey":"AK","secret":"SECRETVALUE","sessionToken":"TOKENVALUE","storageUrl":"https://s3.example","organizationId":"org","bucket":"b","region":"us-east-1"}"#,
        )
        .unwrap();
        let debug = format!("{access:?}");
        assert!(
            !debug.contains("SECRETVALUE") && !debug.contains("TOKENVALUE"),
            "{debug}"
        );
        assert_eq!(access.secret.expose(), "SECRETVALUE");
    }

    #[test]
    fn sandbox_dto_reads_the_fields_we_use() {
        let dto: SandboxDto = serde_json::from_str(
            r#"{"id":"s1","state":"started","labels":{"a":"b"},"toolboxProxyUrl":"https://proxy/toolbox","createdAt":"2026-10-10T09:08:08.456Z","extra":1}"#,
        )
        .unwrap();
        assert_eq!(dto.state(), "started");
        assert_eq!(
            dto.toolbox_proxy_url.as_deref(),
            Some("https://proxy/toolbox")
        );
    }
}
