//! Copying files in and out of a Daytona sandbox.
//!
//! Folders and files travel as one tar archive, staged in a root-only
//! folder ([`STAGING`]) through the toolbox files API, and unpacked (or
//! made) by a root `tar` in the sandbox: modes survive, files land owned by
//! root, and a secret never sits world-readable at its destination. A
//! download is unpacked on the host with the same refusals as Docker's
//! (see [`crate::archive::unpack`]).

use std::path::Path;
use std::time::Duration;

use futures::StreamExt;

use super::SandboxCore;
use super::exec;
use super::toolbox::STAGING;
use crate::shell::quote;
use crate::transfer::{
    Unpacker, pack_bytes_to_tempfile, pack_to_tempfile, remote_path, report_skipped, split_remote,
    unpack_error,
};
use crate::{Exec, ExecOutput, SandboxError};

/// How long a transfer's `tar` may run in the sandbox.
const TAR_TIMEOUT: Duration = Duration::from_secs(1800);

fn staged_name() -> String {
    format!("{STAGING}/xfer-{}.tar", uuid::Uuid::new_v4().simple())
}

/// Run a housekeeping script as root, with none of the sandbox's defaults.
async fn run_root(core: &SandboxCore, script: String) -> Result<ExecOutput, SandboxError> {
    exec::run(
        core,
        Exec::new(script).root().cwd("/").timeout(TAR_TIMEOUT),
        false,
    )
    .await
}

/// Copy a local file or folder to `remote` in the sandbox.
pub(crate) async fn upload(
    core: &SandboxCore,
    local: &Path,
    remote: &str,
) -> Result<(), SandboxError> {
    let remote = remote_path(remote)?;
    let meta = tokio::fs::symlink_metadata(local)
        .await
        .map_err(|e| SandboxError::io(format!("reading {} to upload it", local.display()), e))?;
    let (dst, as_name) = if meta.is_dir() {
        (remote.clone(), None)
    } else {
        let (parent, name) = split_remote(&remote);
        if name == "." {
            return Err(SandboxError::Unsupported(format!(
                "cannot upload the file {} to `/`; name a file path",
                local.display()
            )));
        }
        (parent, Some(name))
    };
    let (archive, packed) = pack_to_tempfile(local, as_name).await?;
    report_skipped(&format!("upload of {}", local.display()), &packed.skipped);
    put_archive(core, &dst, archive).await
}

/// Write `contents` to `remote` with `mode`.
pub(crate) async fn write_file(
    core: &SandboxCore,
    remote: &str,
    contents: &[u8],
    mode: u32,
) -> Result<(), SandboxError> {
    let remote = remote_path(remote)?;
    let (dst, name) = split_remote(&remote);
    if name == "." {
        return Err(SandboxError::Unsupported(
            "cannot write a file at `/`".to_string(),
        ));
    }
    let archive = pack_bytes_to_tempfile(name, contents.to_vec(), mode).await?;
    put_archive(core, &dst, archive).await
}

async fn put_archive(
    core: &SandboxCore,
    dst: &str,
    archive: std::fs::File,
) -> Result<(), SandboxError> {
    let len = archive
        .metadata()
        .map_err(|e| SandboxError::io("reading the upload archive", e))?
        .len();
    let staged = staged_name();
    core.toolbox.upload(&staged, &archive, len).await?;
    let (d, s) = (quote(dst), quote(&staged));
    let out = run_root(
        core,
        format!("mkdir -p {d} && tar -xf {s} -C {d}; rc=$?; rm -f {s}; exit $rc"),
    )
    .await?;
    if out.success() {
        return Ok(());
    }
    Err(tar_failure(
        core,
        &out,
        &format!("copying files into {dst}"),
    ))
}

/// Copy `remote` out of the sandbox to `local`.
pub(crate) async fn download(
    core: &SandboxCore,
    remote: &str,
    local: &Path,
) -> Result<(), SandboxError> {
    let remote = remote_path(remote)?;
    let (parent, name) = split_remote(&remote);
    let staged = staged_name();
    let (p, n, s) = (quote(&parent), quote(&name), quote(&staged));
    let out = run_root(core, format!("tar -cf {s} -C {p} {n}")).await?;
    // GNU tar exits 1 when a file changed while it was read; the archive is
    // still usable.
    let changed = out.exit_code == 1 && out.stderr.contains("file changed as we read it");
    if !out.success() && !changed {
        core.toolbox.remove(&staged).await;
        let lower = out.stderr.to_lowercase();
        if lower.contains("no such file or directory") && !lower.contains("tar: not found") {
            return Err(SandboxError::io(
                format!("downloading {remote}"),
                std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!(
                        "{remote} does not exist in Daytona sandbox {}",
                        core.toolbox.sandbox_id
                    ),
                ),
            ));
        }
        return Err(tar_failure(core, &out, &format!("copying {remote} out")));
    }

    let result = fetch_and_unpack(core, &staged, &name, local).await;
    core.toolbox.remove(&staged).await;
    match result {
        Ok(unpacked) => {
            report_skipped(&format!("download of {remote}"), &unpacked.skipped);
            Ok(())
        }
        Err(Fetch::Api(err)) => Err(err),
        Err(Fetch::Unpack(err)) => Err(unpack_error(&remote, local, err)),
    }
}

enum Fetch {
    Api(SandboxError),
    Unpack(std::io::Error),
}

async fn fetch_and_unpack(
    core: &SandboxCore,
    staged: &str,
    name: &str,
    local: &Path,
) -> Result<crate::archive::Unpacked, Fetch> {
    let response = core.toolbox.download(staged).await.map_err(Fetch::Api)?;
    let unpacker = Unpacker::start(name.to_string(), local.to_path_buf(), None);
    let mut stream = response.bytes_stream();
    let mut transport_error = None;
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(bytes) => {
                if !unpacker.feed(bytes).await {
                    break;
                }
            }
            Err(err) => {
                transport_error = Some(err);
                break;
            }
        }
    }
    let unpacked = unpacker.finish().await;
    if let Some(err) = transport_error {
        return Err(Fetch::Api(SandboxError::Transport(format!(
            "the download from Daytona sandbox {} broke off ({}); run again",
            core.toolbox.sandbox_id,
            core.toolbox.api.redact(&err.to_string())
        ))));
    }
    unpacked.map_err(Fetch::Unpack)
}

/// The error for a `tar` that failed inside the sandbox.
fn tar_failure(core: &SandboxCore, out: &ExecOutput, what: &str) -> SandboxError {
    let message = out.tail(1000);
    let lower = message.to_lowercase();
    if out.exit_code == 127
        || lower.contains("tar: not found")
        || lower.contains("tar: command not found")
    {
        return SandboxError::Unsupported(format!(
            "{what} needs `tar` inside the sandbox, and the image has none ({message}). Add it to the \
             task's image (for example `apt-get install -y tar` or `apk add tar`)."
        ));
    }
    if lower.contains("no space left on device") {
        return SandboxError::Transport(format!(
            "{what} in Daytona sandbox {} failed: its disk is full ({message}). Daytona gives a \
             sandbox at most 10 GB; have the task write less.",
            core.toolbox.sandbox_id
        ));
    }
    SandboxError::Transport(format!(
        "{what} in Daytona sandbox {} failed (exit code {}): {message}",
        core.toolbox.sandbox_id, out.exit_code
    ))
}
