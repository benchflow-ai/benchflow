//! MCP servers attached to an ACP session at `session/new` (0.8's
//! `McpServerSpec`).

use indexmap::IndexMap;
use serde_json::{Map, Value, json};

/// How the agent reaches an MCP server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum McpTransport {
    /// A command the agent starts.
    Stdio {
        command: String,
        args: Vec<String>,
        cwd: Option<String>,
        env: IndexMap<String, String>,
    },
    /// A server at a URL: `sse` or `http` (streamable HTTP).
    Remote {
        kind: String,
        url: String,
        headers: IndexMap<String, String>,
    },
}

/// One MCP server for a session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpServer {
    pub name: String,
    pub transport: McpTransport,
    /// BenchFlow's tool filters, sent under `_meta.benchflow.mcp_tool_filters`.
    pub tools: Option<Vec<String>>,
    pub include_tags: Option<Vec<String>>,
    pub exclude_tags: Option<Vec<String>>,
}

impl McpServer {
    /// A server as a task declares it (Harbor's `[[sandbox.mcp_servers]]`
    /// table): `name`, `transport` (`stdio`, `sse`, `streamable-http` or
    /// `http`; default `sse`), `command`/`args`/`cwd`/`env` or
    /// `url`/`headers`, and optional `tools`, `include_tags`, `exclude_tags`.
    pub fn from_task_table(table: &Value) -> Result<Self, String> {
        let map = table.as_object().ok_or("an MCP server must be a table")?;
        let text = |key: &str| map.get(key).and_then(Value::as_str).map(str::to_string);
        let list = |key: &str| -> Result<Option<Vec<String>>, String> {
            match map.get(key) {
                None | Some(Value::Null) => Ok(None),
                Some(Value::Array(items)) => items
                    .iter()
                    .map(|v| {
                        v.as_str()
                            .map(str::to_string)
                            .ok_or(format!("`{key}` must list strings"))
                    })
                    .collect::<Result<Vec<_>, _>>()
                    .map(Some),
                Some(_) => Err(format!("`{key}` must be a list of strings")),
            }
        };
        let strings = |key: &str| -> Result<IndexMap<String, String>, String> {
            match map.get(key) {
                None | Some(Value::Null) => Ok(IndexMap::new()),
                Some(Value::Object(m)) => m
                    .iter()
                    .map(|(k, v)| {
                        v.as_str()
                            .map(|v| (k.clone(), v.to_string()))
                            .ok_or(format!("`{key}.{k}` must be a string"))
                    })
                    .collect(),
                Some(_) => Err(format!("`{key}` must be a table of strings")),
            }
        };
        let name = text("name").ok_or("an MCP server needs a `name`")?;
        let transport = match text("transport").as_deref().unwrap_or("sse") {
            "stdio" => McpTransport::Stdio {
                command: text("command")
                    .ok_or(format!("MCP server `{name}`: stdio needs a `command`"))?,
                args: list("args")?.unwrap_or_default(),
                cwd: text("cwd"),
                env: strings("env")?,
            },
            kind @ ("sse" | "streamable-http" | "http") => McpTransport::Remote {
                kind: if kind == "sse" { "sse" } else { "http" }.to_string(),
                url: text("url").ok_or(format!("MCP server `{name}`: {kind} needs a `url`"))?,
                headers: strings("headers")?,
            },
            other => return Err(format!("MCP server `{name}`: unknown transport `{other}`")),
        };
        Ok(Self {
            name,
            transport,
            tools: list("tools")?,
            include_tags: list("include_tags")?,
            exclude_tags: list("exclude_tags")?,
        })
    }

    /// The server as `session/new` takes it: stdio servers carry
    /// `command`/`args`/`env` and no `type`; remote ones `type`, `url` and
    /// `headers`. Maps become ACP's `[{name, value}]` lists.
    pub fn to_acp(&self) -> Value {
        let pairs = |m: &IndexMap<String, String>| -> Value {
            m.iter()
                .map(|(k, v)| json!({"name": k, "value": v}))
                .collect()
        };
        let mut out = Map::new();
        match &self.transport {
            McpTransport::Stdio {
                command,
                args,
                cwd,
                env,
            } => {
                out.insert("name".into(), self.name.clone().into());
                out.insert("command".into(), command.clone().into());
                out.insert("args".into(), args.clone().into());
                out.insert("env".into(), pairs(env));
                if let Some(meta) = self.filters_meta() {
                    out.insert("_meta".into(), meta);
                }
                if let Some(cwd) = cwd {
                    out.insert("cwd".into(), cwd.clone().into());
                }
            }
            McpTransport::Remote { kind, url, headers } => {
                out.insert("type".into(), kind.clone().into());
                out.insert("name".into(), self.name.clone().into());
                out.insert("url".into(), url.clone().into());
                out.insert("headers".into(), pairs(headers));
                if let Some(meta) = self.filters_meta() {
                    out.insert("_meta".into(), meta);
                }
            }
        }
        Value::Object(out)
    }

    fn filters_meta(&self) -> Option<Value> {
        let mut filters = Map::new();
        for (key, value) in [
            ("tools", &self.tools),
            ("include_tags", &self.include_tags),
            ("exclude_tags", &self.exclude_tags),
        ] {
            if let Some(value) = value {
                filters.insert(key.into(), value.clone().into());
            }
        }
        (!filters.is_empty()).then(|| json!({"benchflow": {"mcp_tool_filters": filters}}))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_tables_become_acp_servers() {
        let stdio = McpServer::from_task_table(&json!({
            "name": "files", "transport": "stdio", "command": "npx", "args": ["-y", "server"],
            "env": {"ROOT": "/app"}, "tools": ["read"],
        }))
        .unwrap();
        assert_eq!(
            stdio.to_acp(),
            json!({
                "name": "files", "command": "npx", "args": ["-y", "server"],
                "env": [{"name": "ROOT", "value": "/app"}],
                "_meta": {"benchflow": {"mcp_tool_filters": {"tools": ["read"]}}},
            })
        );
        let http = McpServer::from_task_table(&json!({
            "name": "web", "transport": "streamable-http", "url": "http://localhost:3000/mcp",
            "headers": {"X-Key": "k"},
        }))
        .unwrap();
        assert_eq!(
            http.to_acp(),
            json!({"type": "http", "name": "web", "url": "http://localhost:3000/mcp", "headers": [{"name": "X-Key", "value": "k"}]})
        );
        // sse is Harbor's default transport.
        let sse = McpServer::from_task_table(&json!({"name": "s", "url": "http://h/sse"})).unwrap();
        assert_eq!(sse.to_acp()["type"], "sse");
        assert!(McpServer::from_task_table(&json!({"name": "x", "transport": "stdio"})).is_err());
        assert!(
            McpServer::from_task_table(&json!({"name": "x", "transport": "carrier-pigeon"}))
                .is_err()
        );
    }
}
