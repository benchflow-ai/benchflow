//! Install scripts for npm-distributed agents, generated from manifest data.
//!
//! The script reproduces BenchFlow 0.8's `_js_agent_install` byte for byte
//! for the built-in agents:
//!
//! 1. bootstrap a private Node.js 22.20.0 into `/opt/benchflow/node` when it
//!    is not there (installing curl, tar and xz with apt, dnf or apk when the
//!    image lacks them, and picking the x64 or arm64 build);
//! 2. `npm install -g --prefix /opt/benchflow/js-agents` the agent and its
//!    companions;
//! 3. write a launcher to `/opt/benchflow/bin/<bin>` that exports the
//!    manifest's `launcher_env` and starts the agent on that Node, so the
//!    task's own PATH and Node are never touched;
//! 4. `chmod -R a+rX /opt/benchflow` so the non-root agent user can run it;
//! 5. check that the agent and its launcher are executable, then run the
//!    manifest's `verify` command.
//!
//! Every step is safe to repeat, nothing prompts (`DEBIAN_FRONTEND`
//! noninteractive, `-y`), and nothing reads a secret.

use sha2::{Digest, Sha256};

use super::{HomeInstall, Launcher, NpmInstall, UvToolInstall, UvVenvInstall};
use crate::shell::quote;

/// Where BenchFlow's private Node.js lives.
pub(crate) const NODE_PREFIX: &str = "/opt/benchflow/node";
/// The npm prefix agents are installed into.
pub(crate) const JS_PREFIX: &str = "/opt/benchflow/js-agents";
/// Where launchers are written.
pub(crate) const BIN_PREFIX: &str = "/opt/benchflow/bin";

/// 0.8's `_NODE_INSTALL`, verbatim. The Node version is pinned here.
pub(crate) const NODE_BOOTSTRAP: &str = concat!(
    "export DEBIAN_FRONTEND=noninteractive; ",
    "BF_NODE_DIR=/opt/benchflow/node; ",
    "BF_NODE_VERSION=22.20.0; ",
    r#"if [ ! -x "$BF_NODE_DIR/bin/node" ]; then "#,
    "  if ! command -v curl >/dev/null 2>&1 || ",
    "     ! command -v tar >/dev/null 2>&1 || ",
    "     ! command -v xz >/dev/null 2>&1; then ",
    "    if command -v apt-get >/dev/null 2>&1; then ",
    "      apt-get update -qq && ",
    "      apt-get install -y -qq curl ca-certificates tar xz-utils; ",
    "    elif command -v dnf >/dev/null 2>&1; then ",
    "      dnf -y install curl ca-certificates tar xz; ",
    "    elif command -v apk >/dev/null 2>&1; then ",
    "      apk add --no-cache curl ca-certificates tar xz; ",
    "    else ",
    "      echo 'BenchFlow JS agent bootstrap requires curl, tar, and xz' >&2; ",
    "      exit 127; ",
    "    fi; ",
    "  fi; ",
    r#"  arch="$(uname -m)"; "#,
    r#"  case "$arch" in "#,
    "    x86_64|amd64) node_arch=x64 ;; ",
    "    aarch64|arm64) node_arch=arm64 ;; ",
    r#"    *) echo "Unsupported architecture for Node.js: $arch" >&2; exit 1 ;; "#,
    "  esac; ",
    r#"  tmp="$(mktemp -d)"; "#,
    "  mkdir -p /opt/benchflow; ",
    r#"  curl -fsSLo "$tmp/node.tar.xz" "#,
    r#""https://nodejs.org/dist/v${BF_NODE_VERSION}/node-v${BF_NODE_VERSION}-linux-${node_arch}.tar.xz"; "#,
    r#"  rm -rf "$BF_NODE_DIR"; "#,
    r#"  mkdir -p "$BF_NODE_DIR"; "#,
    r#"  tar -xJf "$tmp/node.tar.xz" -C "$BF_NODE_DIR" --strip-components=1 --no-same-owner; "#,
    r#"  rm -rf "$tmp"; "#,
    "fi; ",
    r#"export PATH="/opt/benchflow/node/bin:$PATH"; "#,
    r#""$BF_NODE_DIR/bin/node" --version; "#,
    r#""$BF_NODE_DIR/bin/npm" --version"#,
);

/// An npm install spec: an unversioned package installs `@latest`
/// (0.8's `_npm_package_spec`).
pub(crate) fn package_spec(package: &str) -> String {
    if package.trim_start_matches('@').contains('@') {
        package.to_string()
    } else {
        format!("{package}@latest")
    }
}

/// The launcher written to `/opt/benchflow/bin/<bin>` (0.8's
/// `js_agent_launcher`, plus the `auto` variant that runs a native
/// executable directly).
pub(crate) fn launcher_script(npm: &NpmInstall) -> String {
    let agent_bin = format!("{JS_PREFIX}/bin/{}", npm.bin);
    let mut lines = vec!["#!/bin/sh".to_string()];
    lines.extend(
        npm.launcher_env
            .iter()
            .map(|(name, value)| format!("export {name}={}", quote(value))),
    );
    match npm.launcher {
        Launcher::Node => {
            lines.push(format!(r#"exec {NODE_PREFIX}/bin/node {agent_bin} "$@""#));
        }
        Launcher::Auto => {
            lines.push(format!(
                r##"if [ "$(head -c2 {agent_bin} 2>/dev/null)" = "#!" ]; then"##
            ));
            lines.push(format!(r#"  exec {NODE_PREFIX}/bin/node {agent_bin} "$@""#));
            lines.push("else".to_string());
            lines.push(format!(
                r#"  PATH="{NODE_PREFIX}/bin:$PATH" exec {agent_bin} "$@""#
            ));
            lines.push("fi".to_string());
        }
    }
    lines.iter().map(|line| format!("{line}\n")).collect()
}

/// The full install script for an npm agent.
pub(crate) fn npm_install_script(npm: &NpmInstall) -> String {
    let agent_bin = format!("{JS_PREFIX}/bin/{}", npm.bin);
    let wrapper = format!("{BIN_PREFIX}/{}", npm.bin);
    let spec = package_spec(&npm.package);
    // An unpinned package is not reinstalled over an existing install.
    let guard = if spec == npm.package {
        String::new()
    } else {
        format!("[ -x {agent_bin} ] || ")
    };
    let npm_cmd = format!("{NODE_PREFIX}/bin/npm install -g --prefix {JS_PREFIX}");
    let launcher = launcher_script(npm);
    let launcher_lines: Vec<String> = launcher.lines().map(quote).collect();

    let mut script = format!(
        "{NODE_BOOTSTRAP} && mkdir -p {JS_PREFIX} {BIN_PREFIX} && \
         export PATH=\"{BIN_PREFIX}:{JS_PREFIX}/bin:{NODE_PREFIX}/bin:$PATH\" && \
         ( {guard}{npm_cmd} {} ) && ",
        quote(&spec)
    );
    for companion in &npm.companions {
        script.push_str(&format!("( {npm_cmd} {} ) && ", quote(companion)));
    }
    script.push_str(&format!(
        "printf '%s\\n' {} > {wrapper} && chmod +x {wrapper} && chmod -R a+rX /opt/benchflow && \
         [ -x {agent_bin} ] && [ -x {wrapper} ]",
        launcher_lines.join(" ")
    ));
    if let Some(verify) = &npm.verify {
        script.push_str(" && ");
        script.push_str(verify);
    }
    script
}

/// Where BenchFlow's pinned uv lives.
pub(crate) const UV_PREFIX: &str = "/opt/benchflow/uv";
/// The uv release `uv-tool` installs use. Its musl build runs on any Linux.
pub(crate) const UV_VERSION: &str = "0.13.0";
/// Where installers that write to `$HOME` get their home.
pub(crate) const AGENT_HOMES: &str = "/opt/benchflow/agents";

/// curl, CA certificates, tar and gzip, added with apt, dnf or apk when the
/// image lacks them.
const FETCH_TOOLS: &str = concat!(
    "export DEBIAN_FRONTEND=noninteractive; ",
    "if ! command -v curl >/dev/null 2>&1 || ! command -v tar >/dev/null 2>&1 || ",
    "! command -v gzip >/dev/null 2>&1; then ",
    "  if command -v apt-get >/dev/null 2>&1; then ",
    "    apt-get update -qq && apt-get install -y -qq curl ca-certificates tar gzip; ",
    "  elif command -v dnf >/dev/null 2>&1; then ",
    "    dnf -y install curl ca-certificates tar gzip; ",
    "  elif command -v apk >/dev/null 2>&1; then ",
    "    apk add --no-cache curl ca-certificates tar gzip; ",
    "  else ",
    "    echo 'BenchFlow: this agent install needs curl, tar and gzip, and the image has no apt-get, dnf or apk to add them' >&2; ",
    "    exit 127; ",
    "  fi; ",
    "fi",
);

/// curl and tar (see [`FETCH_TOOLS`]), then BenchFlow's pinned uv (the
/// static musl build, straight from its GitHub release) into
/// `/opt/benchflow/uv` unless it is there; ends with `; ` so a command can
/// follow.
fn uv_bootstrap() -> String {
    format!(
        "{FETCH_TOOLS}; \
         if [ ! -x {UV_PREFIX}/uv ]; then \
         case \"$(uname -m)\" in x86_64|amd64) uv_arch=x86_64 ;; aarch64|arm64) uv_arch=aarch64 ;; \
         *) echo \"BenchFlow: uv has no build for $(uname -m)\" >&2; exit 1 ;; esac; \
         tmp=\"$(mktemp -d)\"; \
         curl -fsSLo \"$tmp/uv.tar.gz\" \"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/uv-${{uv_arch}}-unknown-linux-musl.tar.gz\" && \
         mkdir -p {UV_PREFIX} && tar -xzf \"$tmp/uv.tar.gz\" -C {UV_PREFIX} --strip-components=1 --no-same-owner && rm -rf \"$tmp\" || exit 1; \
         fi; "
    )
}

/// The variables every uv step runs with: uv-managed Pythons under
/// `/opt/benchflow/python` (never the image's), a throwaway cache, no
/// progress bars.
const UV_ENV: &str = "UV_PYTHON_INSTALL_DIR=/opt/benchflow/python UV_PYTHON_PREFERENCE=only-managed \
     UV_CACHE_DIR=/tmp/benchflow-uv-cache UV_NO_PROGRESS=1";

/// The install script for a `uv-tool` agent: a pinned uv, then `uv tool
/// install` with a uv-managed Python, all under `/opt/benchflow`, the tool's
/// executables linked into `/opt/benchflow/bin`.
pub(crate) fn uv_tool_install_script(uv: &UvToolInstall) -> String {
    let python = uv
        .python
        .as_ref()
        .map(|p| format!(" --python {}", quote(p)))
        .unwrap_or_default();
    let mut script = format!(
        "{}mkdir -p {BIN_PREFIX} && \
         UV_TOOL_DIR=/opt/benchflow/uv-tools UV_TOOL_BIN_DIR={BIN_PREFIX} \
         {UV_ENV} \
         {UV_PREFIX}/uv tool install{python} {} && \
         rm -rf /tmp/benchflow-uv-cache && chmod -R a+rX /opt/benchflow && [ -x {BIN_PREFIX}/{} ]",
        uv_bootstrap(),
        quote(&uv.requirement),
        uv.bin,
    );
    if let Some(verify) = &uv.verify {
        script.push_str(" && ");
        script.push_str(verify);
    }
    script
}

/// Where a `uv-venv` agent's virtual environment lives.
pub(crate) fn venv_dir(agent: &str) -> String {
    format!("{AGENT_HOMES}/{agent}/venv")
}

/// The install script for a `uv-venv` agent: a pinned uv, a fresh virtual
/// environment on a uv-managed Python at `/opt/benchflow/agents/<name>/venv`,
/// and the framework's packages installed into it. Nothing is linked into
/// `/opt/benchflow/bin`: the agent's headless command runs the venv's
/// Python on a runner script.
pub(crate) fn uv_venv_install_script(agent: &str, venv: &UvVenvInstall) -> String {
    let dir = venv_dir(agent);
    let python = venv
        .python
        .as_ref()
        .map(|p| format!(" --python {}", quote(p)))
        .unwrap_or_default();
    let requirements: Vec<String> = venv.requirements.iter().map(|r| quote(r)).collect();
    let mut script = format!(
        "{}mkdir -p {AGENT_HOMES}/{agent} && \
         {UV_ENV} {UV_PREFIX}/uv venv --clear{python} {dir} && \
         {UV_ENV} {UV_PREFIX}/uv pip install --python {dir}/bin/python {} && \
         rm -rf /tmp/benchflow-uv-cache && chmod -R a+rX /opt/benchflow && [ -x {dir}/bin/python ]",
        uv_bootstrap(),
        requirements.join(" "),
    );
    if let Some(verify) = &venv.verify {
        script.push_str(" && ");
        script.push_str(verify);
    }
    script
}

/// The install script for an `installer` agent: the vendor's script, run as
/// root in a subshell with `HOME=/opt/benchflow/agents/<name>`, then each of
/// its executables linked into `/opt/benchflow/bin`.
pub(crate) fn home_install_script(agent: &str, home: &HomeInstall) -> String {
    let dir = format!("{AGENT_HOMES}/{agent}");
    let mut script = format!(
        "{FETCH_TOOLS}; mkdir -p {dir} {BIN_PREFIX} && \
         ( export HOME={dir}; cd {dir} || exit 1\n{}\n) && ",
        home.script.trim_end()
    );
    for link in &home.links {
        let name = link.rsplit('/').next().unwrap_or(link);
        script.push_str(&format!(
            "[ -x {dir}/{link} ] && ln -sf {dir}/{link} {BIN_PREFIX}/{name} && ",
            link = quote(link),
            name = quote(name),
        ));
    }
    script.push_str("chmod -R a+rX /opt/benchflow");
    if let Some(verify) = &home.verify {
        script.push_str(" && ");
        script.push_str(verify);
    }
    script
}

/// The install fingerprint: SHA-256 of a format tag and the script.
pub(crate) fn fingerprint(script: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"benchflow-agent-install/1\n");
    hasher.update(script.as_bytes());
    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unversioned_packages_install_latest() {
        assert_eq!(package_spec("pi-acp"), "pi-acp@latest");
        assert_eq!(package_spec("@scope/name"), "@scope/name@latest");
        assert_eq!(package_spec("@scope/name@1.2.3"), "@scope/name@1.2.3");
        assert_eq!(package_spec("name@^1"), "name@^1");
    }

    #[test]
    fn fingerprint_is_stable_hex() {
        let a = fingerprint("echo hi");
        assert_eq!(a.len(), 64);
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
        assert_eq!(a, fingerprint("echo hi"));
        assert_ne!(a, fingerprint("echo hi "));
    }
}
