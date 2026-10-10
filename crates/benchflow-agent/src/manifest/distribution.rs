//! Install methods for agents published the way the public ACP registry
//! publishes them: a per-platform `binary` archive, an `npx` package, or a
//! `uvx` package.
//!
//! ```toml
//! [install.binary.linux-x86_64]
//! archive = "https://github.com/block/goose/releases/download/v1.54.0/goose-x86_64-unknown-linux-gnu.tar.bz2"
//! cmd = "./goose"
//! sha256 = "…"                      # checked when given
//!
//! [install]
//! npx = "@qwen-code/qwen-code@0.25.0"   # or
//! uvx = "fast-agent-acp==0.10.1"
//! ```
//!
//! | Key | Type | Meaning |
//! |---|---|---|
//! | `binary` | table of platforms | `linux-x86_64` and `linux-aarch64`, each with `archive` (an `https://` URL: `.tar.gz`, `.tgz`, `.tar.bz2`, `.tar.xz`, `.zip`, or a bare executable), `cmd` (the executable's path inside the archive, as the registry gives it), and optional `sha256`. The install picks the sandbox's platform with `uname -m`, checks the download against `sha256` when one is given, extracts it under `/opt/benchflow/agents/<name>/dist/`, and writes `/opt/benchflow/bin/<name>`, which runs the executable by its real path inside the archive. |
//! | `npx` | string | An npm package spec, installed like `npm` (BenchFlow's private Node.js, `/opt/benchflow/js-agents`). The executable is `bin` when given, else the one `npx` would run: the package's only executable, or the one named like the package. |
//! | `uvx` | string | A PyPI package pinned as `name==version`, installed with a pinned uv (`uv tool install`) into `/opt/benchflow/agents/<name>/`, Python included. The executable is `bin` when given, else the package's name, as `uvx` runs it. |
//! | `python` | string | With `uvx`: the Python version uv installs for the agent. Default `3.12`. |
//!
//! All three launch as `/opt/benchflow/bin/<name>` followed by `[launch]
//! args`, run as root in a cached image layer, need no credentials, and are
//! safe to repeat: a pinned build already in place is not downloaded again.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use super::install::{BIN_PREFIX, JS_PREFIX, NODE_BOOTSTRAP, NODE_PREFIX};
use crate::shell::quote;

/// Where agents installed from a distribution live, one folder per agent.
pub const AGENTS_PREFIX: &str = "/opt/benchflow/agents";
/// The uv BenchFlow installs `uvx` agents with, and where.
const UV_VERSION: &str = "0.12.5";
const UV_PREFIX: &str = "/opt/benchflow/uv";
/// The Python a `uvx` agent gets when its manifest names none.
pub const DEFAULT_PYTHON: &str = "3.12";

/// The Linux platforms a sandbox can be, as the ACP registry names them,
/// with the `uname -m` values each covers.
pub const PLATFORMS: [(&str, &str); 2] = [
    ("linux-x86_64", "x86_64|amd64"),
    ("linux-aarch64", "aarch64|arm64"),
];

/// A per-platform archive.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BinaryInstall {
    /// The folder under [`AGENTS_PREFIX`] and the launcher's name.
    pub dir: String,
    /// By platform (`linux-x86_64`), in manifest order.
    pub targets: IndexMap<String, BinaryTarget>,
    pub verify: Option<String>,
}

/// One platform's build.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BinaryTarget {
    pub archive: String,
    /// The executable's path inside the archive, without a leading `./`.
    pub cmd: String,
    /// Lowercase hex, when the distribution gives one.
    pub sha256: Option<String>,
    pub kind: ArchiveKind,
}

/// How a download is unpacked, from its URL.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ArchiveKind {
    TarGz,
    TarBz2,
    TarXz,
    Zip,
    /// The download is the executable itself.
    File,
}

impl ArchiveKind {
    /// The kind of `url`'s file, by its name; `None` for an archive format
    /// BenchFlow cannot unpack.
    pub fn of(url: &str) -> Option<Self> {
        let path = url
            .split(['?', '#'])
            .next()
            .unwrap_or(url)
            .to_ascii_lowercase();
        let name = path.rsplit('/').next().unwrap_or(&path);
        let kind = if name.ends_with(".tar.gz") || name.ends_with(".tgz") {
            Self::TarGz
        } else if name.ends_with(".tar.bz2") || name.ends_with(".tbz2") {
            Self::TarBz2
        } else if name.ends_with(".tar.xz") || name.ends_with(".txz") {
            Self::TarXz
        } else if name.ends_with(".zip") {
            Self::Zip
        } else if [
            ".tar.zst", ".tzst", ".7z", ".rar", ".tar", ".gz", ".bz2", ".xz",
        ]
        .iter()
        .any(|ext| name.ends_with(ext))
        {
            return None;
        } else {
            Self::File
        };
        Some(kind)
    }

    fn label(self) -> &'static str {
        match self {
            Self::TarGz => "tar.gz",
            Self::TarBz2 => "tar.bz2",
            Self::TarXz => "tar.xz",
            Self::Zip => "zip",
            Self::File => "file",
        }
    }

    /// The commands unpacking needs besides `curl`.
    fn tools(self) -> &'static str {
        match self {
            Self::TarGz => "curl tar gzip",
            Self::TarBz2 => "curl tar bzip2",
            Self::TarXz => "curl tar xz",
            Self::Zip => "curl unzip",
            Self::File => "curl",
        }
    }
}

/// An npm package run the way `npx` runs it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct NpxInstall {
    pub dir: String,
    pub package: String,
    /// The package's executable; `None` picks it as `npx` does.
    pub bin: Option<String>,
    pub verify: Option<String>,
}

/// A PyPI package run the way `uvx` runs it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct UvxInstall {
    pub dir: String,
    /// `name==version`, possibly with extras.
    pub package: String,
    pub bin: String,
    pub python: String,
    pub verify: Option<String>,
}

/// `[install.binary.<platform>]` as written.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RawBinaryTarget {
    archive: String,
    cmd: String,
    sha256: Option<String>,
}

/// Why a distribution field is refused: the key and the message.
pub(super) type Refusal = (String, String);

fn refuse(key: impl Into<String>, message: impl Into<String>) -> Refusal {
    (key.into(), message.into())
}

/// A path inside an archive, as the registry writes it (`./dist/agent`),
/// normalized to `dist/agent`. Refuses absolute paths, `..`, and characters
/// that would need quoting in the launcher.
pub fn normalize_cmd(cmd: &str) -> Result<String, String> {
    let trimmed = cmd.trim();
    let mut rest = trimmed;
    while let Some(r) = rest.strip_prefix("./") {
        rest = r;
    }
    if rest.is_empty() {
        return Err(format!("`{cmd}` names no file"));
    }
    if rest.starts_with('/') {
        return Err(format!(
            "`{cmd}` is an absolute path; give the executable's path inside the archive"
        ));
    }
    if rest.contains('\\') {
        return Err(format!("`{cmd}` is a Windows path"));
    }
    if rest.split('/').any(|part| part == ".." || part.is_empty()) {
        return Err(format!("`{cmd}` leaves the archive or has an empty part"));
    }
    if !rest
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-' | '+' | '@' | '/'))
    {
        return Err(format!(
            "`{cmd}` has characters BenchFlow does not put in a launcher (use letters, digits, and . _ - + @ /)"
        ));
    }
    Ok(rest.to_string())
}

/// A SHA-256 digest as hex or `sha256:<hex>`, lowercased.
pub fn normalize_sha256(value: &str) -> Result<String, String> {
    let hex = value.trim();
    let hex = hex.strip_prefix("sha256:").unwrap_or(hex);
    if hex.len() == 64 && hex.chars().all(|c| c.is_ascii_hexdigit()) {
        Ok(hex.to_ascii_lowercase())
    } else {
        Err(format!("`{value}` is not a SHA-256 digest (64 hex digits)"))
    }
}

fn check_archive_url(url: &str) -> Result<ArchiveKind, String> {
    if !url.starts_with("https://") {
        return Err(format!("`{url}` is not an https:// URL"));
    }
    if url
        .chars()
        .any(|c| c.is_whitespace() || c.is_control() || matches!(c, '\'' | '"' | '\\' | '`'))
    {
        return Err(format!("`{url}` has characters a URL cannot have"));
    }
    ArchiveKind::of(url).ok_or_else(|| {
        format!(
            "BenchFlow cannot unpack `{url}`; it unpacks .tar.gz, .tgz, .tar.bz2, .tar.xz, and .zip archives, or downloads a bare executable"
        )
    })
}

pub(super) fn binary(
    dir: &str,
    raw: IndexMap<String, RawBinaryTarget>,
    verify: Option<String>,
) -> Result<BinaryInstall, Refusal> {
    let known: Vec<&str> = PLATFORMS.iter().map(|(p, _)| *p).collect();
    let mut targets = IndexMap::new();
    for (platform, target) in raw {
        let key = format!("install.binary.{platform}");
        if !known.contains(&platform.as_str()) {
            return Err(refuse(
                &key,
                format!(
                    "`{platform}` is not a platform a sandbox runs; use {}",
                    known.join(" or ")
                ),
            ));
        }
        let kind = check_archive_url(&target.archive).map_err(|m| refuse(&key, m))?;
        let cmd = normalize_cmd(&target.cmd).map_err(|m| refuse(format!("{key}.cmd"), m))?;
        let sha256 = target
            .sha256
            .as_deref()
            .map(normalize_sha256)
            .transpose()
            .map_err(|m| refuse(format!("{key}.sha256"), m))?;
        targets.insert(
            platform,
            BinaryTarget {
                archive: target.archive,
                cmd,
                sha256,
                kind,
            },
        );
    }
    if targets.is_empty() {
        return Err(refuse(
            "install.binary",
            format!("list at least one platform: {}", known.join(", ")),
        ));
    }
    Ok(BinaryInstall {
        dir: dir.to_string(),
        targets,
        verify,
    })
}

/// The package name of an npm spec: `@scope/name@1.2.3` is `@scope/name`.
pub fn npm_package_name(spec: &str) -> &str {
    let (scope, rest) = match spec.strip_prefix('@') {
        Some(rest) => match rest.find('/') {
            Some(slash) => (slash + 2, &spec[slash + 2..]),
            None => (0, spec),
        },
        None => (0, spec),
    };
    match rest.find('@') {
        Some(at) => &spec[..scope + at],
        None => spec,
    }
}

pub(super) fn npx(
    dir: &str,
    package: String,
    bin: Option<String>,
    verify: Option<String>,
) -> Result<NpxInstall, Refusal> {
    let name = npm_package_name(&package);
    if package.is_empty()
        || package.starts_with('-')
        || name.is_empty()
        || package.chars().any(|c| c.is_whitespace() || c.is_control())
    {
        return Err(refuse(
            "install.npx",
            format!("`{package}` is not an npm package spec"),
        ));
    }
    Ok(NpxInstall {
        dir: dir.to_string(),
        package,
        bin,
        verify,
    })
}

/// A PEP 508 name with optional extras, then `==version` (or uvx's
/// `@version`).
fn split_pinned_pypi(spec: &str) -> Option<(&str, &str)> {
    let (name, version) = spec.split_once("==").or_else(|| spec.split_once('@'))?;
    let base = name.split('[').next().unwrap_or(name);
    let ok_name = !base.is_empty()
        && base
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'));
    let ok_extras = name.len() == base.len()
        || (name.ends_with(']')
            && name[base.len()..]
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '[' | ']' | ',' | '_' | '-')));
    let ok_version = !version.is_empty()
        && version
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-' | '+' | '!'));
    (ok_name && ok_extras && ok_version).then_some((base, version))
}

pub(super) fn uvx(
    dir: &str,
    package: String,
    bin: Option<String>,
    python: Option<String>,
    verify: Option<String>,
) -> Result<UvxInstall, Refusal> {
    let Some((name, version)) = split_pinned_pypi(&package) else {
        return Err(refuse(
            "install.uvx",
            format!(
                "`{package}` is not a pinned PyPI package; write it as `name==version` so every image gets the same agent"
            ),
        ));
    };
    let python = python.unwrap_or_else(|| DEFAULT_PYTHON.to_string());
    if python.is_empty()
        || !python
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '+'))
    {
        return Err(refuse(
            "install.python",
            format!("`{python}` is not a Python version such as 3.12"),
        ));
    }
    // `uv tool install` takes `name==version`; uvx also takes `name@version`.
    let requirement = package.split(['=', '@']).next().unwrap_or(&package);
    Ok(UvxInstall {
        dir: dir.to_string(),
        bin: bin.unwrap_or_else(|| name.to_string()),
        package: format!("{requirement}=={version}"),
        python,
        verify,
    })
}

// ---------------------------------------------------------------------------
// Install scripts
// ---------------------------------------------------------------------------

/// Install the commands in `$bf_tools` that the image lacks, with apt-get,
/// dnf, or apk.
fn ensure_tools(agent: &str) -> String {
    format!(
        r#"bf_missing=""
for bf_t in $bf_tools; do command -v "$bf_t" >/dev/null 2>&1 || bf_missing="$bf_missing $bf_t"; done
if [ -n "$bf_missing" ]; then
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update -qq && apt-get install -y -qq ca-certificates $(printf '%s\n' $bf_missing | sed 's/^xz$/xz-utils/')
  elif command -v dnf >/dev/null 2>&1; then
    dnf -y install ca-certificates $bf_missing
  elif command -v apk >/dev/null 2>&1; then
    apk add --no-cache ca-certificates $bf_missing
  else
    echo "BenchFlow: installing {agent} needs$bf_missing, and the image has no apt-get, dnf, or apk to install them" >&2
    exit 127
  fi
fi
"#
    )
}

/// Write `/opt/benchflow/bin/<name>` to run `$bf_target` (expanded now).
fn write_launcher(name: &str) -> String {
    let launcher = format!("{BIN_PREFIX}/{name}");
    format!(
        "mkdir -p {BIN_PREFIX}\n\
         printf '#!/bin/sh\\nexec \"%s\" \"$@\"\\n' \"$bf_target\" > {launcher}\n\
         chmod +x {launcher}\n"
    )
}

fn finish(target_check: &str, verify: Option<&str>) -> String {
    let mut s = format!("chmod -R a+rX /opt/benchflow\n{target_check}\n");
    if let Some(verify) = verify {
        s.push_str(verify);
        s.push('\n');
    }
    s
}

/// The install script for a per-platform archive.
pub(super) fn binary_script(b: &BinaryInstall) -> String {
    let dir = format!("{AGENTS_PREFIX}/{}", b.dir);
    let name = &b.dir;
    let mut s = String::from("set -e\nexport DEBIAN_FRONTEND=noninteractive\n");
    s.push_str(&format!("bf_dir={dir}\n"));
    s.push_str("case \"$(uname -m)\" in\n");
    for (platform, arches) in PLATFORMS {
        let Some(t) = b.targets.get(platform) else {
            continue;
        };
        s.push_str(&format!(
            "  {arches}) bf_url={}; bf_sha={}; bf_cmd={}; bf_kind={}; bf_tools={} ;;\n",
            quote(&t.archive),
            quote(t.sha256.as_deref().unwrap_or("")),
            quote(&t.cmd),
            t.kind.label(),
            quote(t.kind.tools()),
        ));
    }
    let listed: Vec<&str> = b.targets.keys().map(String::as_str).collect();
    s.push_str(&format!(
        "  *) echo \"BenchFlow: {name} publishes no build for linux-$(uname -m); its distribution lists {}\" >&2; exit 1 ;;\nesac\n",
        listed.join(", ")
    ));
    s.push_str("if [ ! -x \"$bf_dir/dist/$bf_cmd\" ]; then\n");
    for line in ensure_tools(name).lines() {
        s.push_str("  ");
        s.push_str(line);
        s.push('\n');
    }
    s.push_str(&format!(
        r#"  bf_tmp="$(mktemp -d)"
  curl -fsSL --retry 3 -o "$bf_tmp/download" "$bf_url"
  if [ -n "$bf_sha" ]; then
    bf_got="$( (sha256sum "$bf_tmp/download" 2>/dev/null || shasum -a 256 "$bf_tmp/download") | cut -d ' ' -f 1)"
    if [ "$bf_got" != "$bf_sha" ]; then
      echo "BenchFlow: $bf_url has SHA-256 $bf_got, but {name}'s distribution lists $bf_sha; the download is not the file the distribution names" >&2
      exit 1
    fi
  fi
  rm -rf "$bf_dir/dist"
  mkdir -p "$bf_dir/dist"
  case "$bf_kind" in
    tar.gz) tar -xzf "$bf_tmp/download" -C "$bf_dir/dist" --no-same-owner ;;
    tar.bz2) tar -xjf "$bf_tmp/download" -C "$bf_dir/dist" --no-same-owner ;;
    tar.xz) tar -xJf "$bf_tmp/download" -C "$bf_dir/dist" --no-same-owner ;;
    zip) unzip -q -o "$bf_tmp/download" -d "$bf_dir/dist" ;;
    file) mkdir -p "$(dirname "$bf_dir/dist/$bf_cmd")" && cp "$bf_tmp/download" "$bf_dir/dist/$bf_cmd" ;;
  esac
  rm -rf "$bf_tmp"
  if [ ! -f "$bf_dir/dist/$bf_cmd" ]; then
    echo "BenchFlow: {name}'s archive has no $bf_cmd, the executable its distribution names; the archive holds: $(cd "$bf_dir/dist" && find . -maxdepth 3 | head -n 30 | tr '\n' ' ')" >&2
    exit 1
  fi
  chmod +x "$bf_dir/dist/$bf_cmd"
fi
bf_target="$bf_dir/dist/$bf_cmd"
"#
    ));
    s.push_str(&write_launcher(name));
    s.push_str(&finish(
        &format!("[ -x \"$bf_target\" ] && [ -x {BIN_PREFIX}/{name} ]"),
        b.verify.as_deref(),
    ));
    s
}

/// Picks the executable `npx` would run from an installed package's
/// package.json (its only `bin`, or the one named like the package).
const NPX_BIN_PICKER: &str = r#"const p=require(process.argv[1]+"/package.json");const n=p.name.split("/").pop();const b=p.bin;let r="";if(typeof b==="string"){r=n}else if(b&&typeof b==="object"){const k=Object.keys(b);r=k.length===1?k[0]:(k.includes(n)?n:"")}if(!r){console.error("BenchFlow: "+p.name+" installs "+(b&&typeof b==="object"?"the executables "+Object.keys(b).join(", "):"no executable")+", and none is named like the package; set install.bin to the one that speaks ACP");process.exit(1)}console.log(r)"#;

/// The install script for an `npx` package.
pub(super) fn npx_script(n: &NpxInstall) -> String {
    let name = &n.dir;
    let spec = super::install::package_spec(&n.package);
    let pkg_dir = format!(
        "{JS_PREFIX}/lib/node_modules/{}",
        npm_package_name(&n.package)
    );
    let pick = match &n.bin {
        Some(bin) => format!("bf_bin={}", quote(bin)),
        None => format!(
            "bf_bin=\"$({NODE_PREFIX}/bin/node -e {} {})\"",
            quote(NPX_BIN_PICKER),
            quote(&pkg_dir)
        ),
    };
    // An unpinned package is not reinstalled over an existing install.
    let guard = if spec == n.package {
        String::new()
    } else {
        format!("[ -d {} ] || ", quote(&pkg_dir))
    };
    let mut s = String::from("set -e\n");
    s.push_str(NODE_BOOTSTRAP);
    s.push('\n');
    s.push_str(&format!(
        "mkdir -p {JS_PREFIX} {BIN_PREFIX}\n\
         export PATH=\"{BIN_PREFIX}:{JS_PREFIX}/bin:{NODE_PREFIX}/bin:$PATH\"\n\
         {guard}{NODE_PREFIX}/bin/npm install -g --prefix {JS_PREFIX} {}\n\
         {pick}\n",
        quote(&spec)
    ));
    s.push_str(&format!(
        r#"case "$bf_bin" in
  ''|*[!A-Za-z0-9._-]*) echo "BenchFlow: {name}'s package has no usable executable (got '$bf_bin')" >&2; exit 1 ;;
esac
bf_js="{JS_PREFIX}/bin/$bf_bin"
[ -x "$bf_js" ] || {{ echo "BenchFlow: {} installed no executable $bf_js" >&2; exit 1; }}
printf '%s\n' '#!/bin/sh' "if [ \"\$(head -c2 $bf_js 2>/dev/null)\" = '#!' ]; then" "  exec {NODE_PREFIX}/bin/node $bf_js \"\$@\"" else "  PATH=\"{NODE_PREFIX}/bin:\$PATH\" exec $bf_js \"\$@\"" fi > {BIN_PREFIX}/{name}
chmod +x {BIN_PREFIX}/{name}
"#,
        n.package
    ));
    s.push_str(&finish(
        &format!("[ -x {BIN_PREFIX}/{name} ]"),
        n.verify.as_deref(),
    ));
    s
}

/// The install script for a `uvx` package.
pub(super) fn uvx_script(u: &UvxInstall) -> String {
    let name = &u.dir;
    let dir = format!("{AGENTS_PREFIX}/{name}");
    let mut s = String::from("set -e\nexport DEBIAN_FRONTEND=noninteractive\n");
    s.push_str(&format!(
        "bf_dir={dir}\nbf_uv={UV_PREFIX}/uv\nif [ ! -x \"$bf_uv\" ]; then\n  bf_tools='curl tar gzip'\n"
    ));
    for line in ensure_tools(name).lines() {
        s.push_str("  ");
        s.push_str(line);
        s.push('\n');
    }
    s.push_str(&format!(
        r#"  case "$(uname -m)" in
    x86_64|amd64) bf_uv_arch=x86_64 ;;
    aarch64|arm64) bf_uv_arch=aarch64 ;;
    *) echo "BenchFlow: uv has no build for linux-$(uname -m), so {name} cannot be installed" >&2; exit 1 ;;
  esac
  bf_tmp="$(mktemp -d)"
  curl -fsSL --retry 3 -o "$bf_tmp/uv.tar.gz" "https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/uv-$bf_uv_arch-unknown-linux-musl.tar.gz"
  mkdir -p {UV_PREFIX}
  tar -xzf "$bf_tmp/uv.tar.gz" -C {UV_PREFIX} --strip-components=1 --no-same-owner
  rm -rf "$bf_tmp"
fi
bf_target="$bf_dir/bin/"{bin}
if [ ! -x "$bf_target" ]; then
  UV_TOOL_DIR="$bf_dir/tools" UV_TOOL_BIN_DIR="$bf_dir/bin" UV_PYTHON_INSTALL_DIR="$bf_dir/python" UV_CACHE_DIR="$bf_dir/.cache" UV_PYTHON_PREFERENCE=only-managed UV_NO_PROGRESS=1 "$bf_uv" tool install --python {python} {package}
  rm -rf "$bf_dir/.cache"
fi
if [ ! -x "$bf_target" ]; then
  echo "BenchFlow: {package_plain} installed no executable named {bin_plain}; it installed: $(ls "$bf_dir/bin" 2>/dev/null | tr '\n' ' '); set install.bin to the one that speaks ACP" >&2
  exit 1
fi
"#,
        bin = quote(&u.bin),
        python = quote(&u.python),
        package = quote(&u.package),
        package_plain = u.package,
        bin_plain = u.bin,
    ));
    s.push_str(&write_launcher(name));
    s.push_str(&finish(
        &format!("[ -x {BIN_PREFIX}/{name} ]"),
        u.verify.as_deref(),
    ));
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{AgentManifest, InstallMethod, ManifestSource};

    fn parse(text: &str) -> Result<AgentManifest, crate::ManifestError> {
        AgentManifest::parse(text, ManifestSource::Builtin("test.toml"))
    }

    /// `sh -n`: the script parses as POSIX shell.
    fn assert_parses(script: &str) {
        let out = std::process::Command::new("sh")
            .arg("-n")
            .arg("-c")
            .arg(script)
            .output()
            .expect("sh runs");
        assert!(
            out.status.success(),
            "{}\n---\n{script}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    const SHA: &str = "6b4fcfb4cfbd1d27e2ab1b9e0d8df0bfd54ee1a64b1c8c5fb7c7a5ef2a52ec9e";

    #[test]
    fn a_binary_install_picks_the_platform_and_launches_the_real_path() {
        let m = parse(&format!(
            r#"
name = "cortex-code"
[install.binary.linux-x86_64]
archive = "https://example.com/coco-1.0.73%2B1-linux-amd64.tar.gz"
cmd = "./coco-1.0.73+1-linux-amd64/cortex"
sha256 = "sha256:{SHA}"
[install.binary.linux-aarch64]
archive = "https://example.com/coco-1.0.73%2B1-linux-arm64.zip"
cmd = "./coco-1.0.73+1-linux-arm64/cortex"
[launch]
args = ["acp", "serve"]
"#
        ))
        .unwrap();
        let InstallMethod::Binary(b) = &m.install.method else {
            panic!()
        };
        assert_eq!(
            b.targets["linux-x86_64"].cmd,
            "coco-1.0.73+1-linux-amd64/cortex"
        );
        assert_eq!(b.targets["linux-aarch64"].kind, ArchiveKind::Zip);
        assert_eq!(
            m.launch_command(),
            "/opt/benchflow/bin/cortex-code acp serve"
        );
        let script = m.install_script();
        assert_parses(&script);
        assert!(script.contains("bf_dir=/opt/benchflow/agents/cortex-code\n"));
        assert!(script.contains(&format!("bf_sha={SHA};")), "{script}");
        assert!(script.contains(
            "x86_64|amd64) bf_url=https://example.com/coco-1.0.73%2B1-linux-amd64.tar.gz;"
        ));
        assert!(script.contains(
            "bf_cmd=coco-1.0.73+1-linux-arm64/cortex; bf_kind=zip; bf_tools='curl unzip'"
        ));
        // The launcher runs the executable by its path inside the archive,
        // not by its file name (Harbor's generic ACP agent ran
        // `dist/<basename>`, which misses binaries kept in a subfolder).
        assert!(script.contains("bf_target=\"$bf_dir/dist/$bf_cmd\""));
        assert!(script.contains("> /opt/benchflow/bin/cortex-code"));
        assert!(script.contains("publishes no build for linux-$(uname -m); its distribution lists linux-x86_64, linux-aarch64"));
    }

    #[test]
    fn a_bare_executable_download_is_copied_to_its_cmd() {
        let m = parse(
            r#"
name = "grok-build"
[install.binary.linux-x86_64]
archive = "https://x.ai/cli/grok-0.2.20-linux-x86_64"
cmd = "./grok-0.2.20-linux-x86_64"
"#,
        )
        .unwrap();
        let script = m.install_script();
        assert_parses(&script);
        assert!(
            script.contains("bf_kind=file; bf_tools=curl ;;"),
            "{script}"
        );
        assert!(
            !script.contains("aarch64|arm64) bf_url"),
            "only listed platforms"
        );
    }

    #[test]
    fn an_npx_install_reuses_the_npm_install_and_picks_the_bin_as_npx_does() {
        let m = parse(
            r#"
name = "qwen-code"
[install]
npx = "@qwen-code/qwen-code@0.25.0"
[launch]
args = ["--acp", "--experimental-skills"]
"#,
        )
        .unwrap();
        assert_eq!(
            m.launch_command(),
            "/opt/benchflow/bin/qwen-code --acp --experimental-skills"
        );
        let script = m.install_script();
        assert_parses(&script);
        assert!(script.contains(NODE_BOOTSTRAP));
        assert!(script.contains("/opt/benchflow/node/bin/npm install -g --prefix /opt/benchflow/js-agents @qwen-code/qwen-code@0.25.0\n"));
        assert!(script.contains("/opt/benchflow/js-agents/lib/node_modules/@qwen-code/qwen-code"));
        assert!(script.contains("> /opt/benchflow/bin/qwen-code"));
        // A named bin skips the picker.
        let named = parse(
            r#"
name = "copilot"
[install]
npx = "@github/copilot@1.0.95"
bin = "copilot"
"#,
        )
        .unwrap();
        let script = named.install_script();
        assert_parses(&script);
        assert!(script.contains("bf_bin=copilot\n"));
        assert!(!script.contains("package.json"));
    }

    #[test]
    fn a_uvx_install_pins_uv_python_and_the_package() {
        let m = parse(
            r#"
name = "minion-code"
[install]
uvx = "minion-code@0.1.44"
[launch]
args = ["acp"]
"#,
        )
        .unwrap();
        let InstallMethod::Uvx(u) = &m.install.method else {
            panic!()
        };
        assert_eq!(u.package, "minion-code==0.1.44");
        assert_eq!((u.bin.as_str(), u.python.as_str()), ("minion-code", "3.12"));
        let script = m.install_script();
        assert_parses(&script);
        assert!(script.contains(&format!(
            "releases/download/{UV_VERSION}/uv-$bf_uv_arch-unknown-linux-musl.tar.gz"
        )));
        assert!(
            script.contains("tool install --python 3.12 minion-code==0.1.44\n"),
            "{script}"
        );
        assert!(script.contains("bf_target=\"$bf_dir/bin/\"minion-code\n"));
        assert_eq!(m.launch_command(), "/opt/benchflow/bin/minion-code acp");
    }

    #[test]
    fn every_pin_moves_the_fingerprint() {
        let base = |sha: &str, url: &str| {
            parse(&format!(
                "name = \"a\"\n[install.binary.linux-x86_64]\narchive = \"{url}\"\ncmd = \"a\"\nsha256 = \"{sha}\"\n"
            ))
            .unwrap()
            .version_fingerprint()
        };
        let a = base(SHA, "https://e/a-1.tar.gz");
        assert_ne!(a, base(&"0".repeat(64), "https://e/a-1.tar.gz"));
        assert_ne!(a, base(SHA, "https://e/a-2.tar.gz"));
        let uvx = |spec: &str, python: &str| {
            parse(&format!(
                "name = \"a\"\n[install]\nuvx = \"{spec}\"\npython = \"{python}\"\n"
            ))
            .unwrap()
            .version_fingerprint()
        };
        assert_ne!(uvx("a==1", "3.12"), uvx("a==2", "3.12"));
        assert_ne!(uvx("a==1", "3.12"), uvx("a==1", "3.13"));
    }

    #[test]
    fn distribution_mistakes_are_refused_with_the_fix() {
        for (text, needle) in [
            (
                "name = \"a\"\n[install]\nnpx = \"a@1\"\nuvx = \"a==1\"\n",
                "use one install method, not npx and uvx",
            ),
            (
                "name = \"a\"\n[install]\nuvx = \"a\"\n",
                "not a pinned PyPI package",
            ),
            (
                "name = \"a\"\n[install.binary.windows-x86_64]\narchive = \"https://e/a.zip\"\ncmd = \"a.exe\"\n",
                "not a platform a sandbox runs",
            ),
            (
                "name = \"a\"\n[install.binary.linux-x86_64]\narchive = \"http://e/a.tar.gz\"\ncmd = \"a\"\n",
                "not an https:// URL",
            ),
            (
                "name = \"a\"\n[install.binary.linux-x86_64]\narchive = \"https://e/a.tar.zst\"\ncmd = \"a\"\n",
                "cannot unpack",
            ),
            (
                "name = \"a\"\n[install.binary.linux-x86_64]\narchive = \"https://e/a.tar.gz\"\ncmd = \"../a\"\n",
                "leaves the archive",
            ),
            (
                "name = \"a\"\n[install.binary.linux-x86_64]\narchive = \"https://e/a.tar.gz\"\ncmd = \"a\"\nsha256 = \"xyz\"\n",
                "not a SHA-256 digest",
            ),
            (
                "name = \"a\"\n[install]\nnpx = \"a@1\"\npython = \"3.12\"\n",
                "does not apply to `npx` installs",
            ),
            (
                "name = \"a\"\n[install]\nnpm = \"a@1\"\nbin = \"a\"\npython = \"3.12\"\n",
                "only applies to `uv-tool`, `uv-venv`, or `uvx` installs",
            ),
        ] {
            let err = parse(text).unwrap_err().to_string();
            assert!(err.contains(needle), "{needle}: {err}");
        }
    }

    #[test]
    fn archive_kinds_come_from_the_file_name() {
        for (url, kind) in [
            ("https://x/a.tar.gz", Some(ArchiveKind::TarGz)),
            ("https://x/a.TGZ?sig=1", Some(ArchiveKind::TarGz)),
            ("https://x/a.tar.bz2", Some(ArchiveKind::TarBz2)),
            ("https://x/a.tar.xz", Some(ArchiveKind::TarXz)),
            ("https://x/a.zip", Some(ArchiveKind::Zip)),
            (
                "https://x.ai/cli/grok-0.2.20-linux-x86_64",
                Some(ArchiveKind::File),
            ),
            ("https://x/sigit-linux-amd64", Some(ArchiveKind::File)),
            ("https://x/a.tar.zst", None),
            ("https://x/a.7z", None),
        ] {
            assert_eq!(ArchiveKind::of(url), kind, "{url}");
        }
    }

    #[test]
    fn commands_are_paths_inside_the_archive() {
        assert_eq!(normalize_cmd("./goose").unwrap(), "goose");
        assert_eq!(
            normalize_cmd("./dist-package/cursor-agent").unwrap(),
            "dist-package/cursor-agent"
        );
        assert_eq!(
            normalize_cmd("./coco-1.0.73+180523.e6179a031de9-linux-amd64/cortex").unwrap(),
            "coco-1.0.73+180523.e6179a031de9-linux-amd64/cortex"
        );
        for bad in [
            "",
            "./",
            "/usr/bin/x",
            "../x",
            "a/../../x",
            "./dist-package\\x.cmd",
            "a b",
            "a$b",
        ] {
            assert!(normalize_cmd(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn digests_accept_both_spellings() {
        let hex = "AB".repeat(32);
        assert_eq!(normalize_sha256(&hex).unwrap(), "ab".repeat(32));
        assert_eq!(
            normalize_sha256(&format!("sha256:{hex}")).unwrap(),
            "ab".repeat(32)
        );
        assert!(normalize_sha256("abc").is_err());
    }

    #[test]
    fn npm_names_drop_the_version() {
        assert_eq!(
            npm_package_name("@github/copilot@1.0.95"),
            "@github/copilot"
        );
        assert_eq!(npm_package_name("@scope/name"), "@scope/name");
        assert_eq!(npm_package_name("cline@3.0.70"), "cline");
        assert_eq!(npm_package_name("pi-acp"), "pi-acp");
    }

    #[test]
    fn pypi_specs_must_be_pinned() {
        assert_eq!(
            split_pinned_pypi("fast-agent-acp==0.10.1"),
            Some(("fast-agent-acp", "0.10.1"))
        );
        assert_eq!(
            split_pinned_pypi("mistral-vibe[acp]==2.19.0"),
            Some(("mistral-vibe", "2.19.0"))
        );
        assert_eq!(
            split_pinned_pypi("minion-code@0.1.44"),
            Some(("minion-code", "0.1.44"))
        );
        for bad in ["minion-code", "minion-code>=0.1", "a==", "==1", "a b==1"] {
            assert_eq!(split_pinned_pypi(bad), None, "{bad}");
        }
    }
}
