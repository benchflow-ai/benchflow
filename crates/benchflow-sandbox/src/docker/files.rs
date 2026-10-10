//! Copying files in and out of a container through `tar` streams.
//!
//! Upload packs on the host and pipes the archive into `docker exec -i -u 0
//! <name> sh -c 'mkdir -p <dst> && tar -xf - -C <dst>'`, so files land owned
//! by root with their modes. Download runs `docker exec -u 0 <name> tar -cf
//! - -C <parent> <name>` and unpacks on the host, refusing anything but
//! regular files and folders (see [`crate::archive::unpack`]). Both run with
//! `PATH` set to [`crate::shell::SYSTEM_PATH`].

use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use bytes::Bytes;
use tokio::io::AsyncReadExt;

use super::cli::{Cli, Output};
use crate::SandboxError;
use crate::capture::read_tail;
use crate::shell::quote;
use crate::transfer::{
    Unpacker, pack_bytes_to_tempfile, pack_to_tempfile, remote_path, report_skipped, split_remote,
    unpack_error,
};

/// Copy a local file or folder to `remote` in the container.
pub(super) async fn upload(
    cli: &Cli,
    container: &str,
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
    extract(cli, container, &dst, archive).await
}

/// Write `contents` to `remote` with `mode`, creating parent folders.
pub(super) async fn write_file(
    cli: &Cli,
    container: &str,
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
    extract(cli, container, &dst, archive).await
}

/// `PATH=<system folders>` for `docker exec -e`: the copies run as root and
/// find `sh` and `tar` there, never in a folder on the image's `PATH` the
/// agent could write.
fn system_path_env() -> String {
    format!("PATH={}", crate::shell::SYSTEM_PATH)
}

/// The `docker exec` arguments that unpack a tar on stdin into `dst`.
pub(crate) fn extract_args(container: &str, dst: &str) -> Vec<String> {
    [
        "exec",
        "-i",
        "-u",
        "0",
        "-e",
        &system_path_env(),
        container,
        "sh",
        "-c",
        &extract_script(dst),
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

/// The `docker exec` arguments that pack `name` in `parent` to stdout.
pub(crate) fn download_args(container: &str, parent: &str, name: &str) -> Vec<String> {
    [
        "exec",
        "-u",
        "0",
        "-e",
        &system_path_env(),
        container,
        "tar",
        "-cf",
        "-",
        "-C",
        parent,
        name,
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

/// The shell command that unpacks a tar on stdin into `dst`.
pub(crate) fn extract_script(dst: &str) -> String {
    let d = quote(dst);
    format!("mkdir -p {d} && tar -xf - -C {d}")
}

async fn extract(
    cli: &Cli,
    container: &str,
    dst: &str,
    archive: std::fs::File,
) -> Result<(), SandboxError> {
    let args = extract_args(container, dst);
    let mut cmd = cli.command(&args);
    cmd.stdin(Stdio::from(archive));
    let out = cli.run_command(cmd, &args, None).await?;
    if out.ok() {
        return Ok(());
    }
    Err(transfer_failure(
        cli,
        container,
        &out,
        &format!("copying files into {dst}"),
    ))
}

/// Copy `remote` (a file or folder) out of the container to `local`.
pub(super) async fn download(
    cli: &Cli,
    container: &str,
    remote: &str,
    local: &Path,
) -> Result<(), SandboxError> {
    let remote = remote_path(remote)?;
    let (parent, name) = split_remote(&remote);
    let args = download_args(container, &parent, &name);
    let mut child = cli
        .command(&args)
        .spawn()
        .map_err(|e| cli.spawn_failed(e))?;
    let mut stdout = child.stdout.take().expect("piped");
    let stderr = tokio::spawn(read_tail(child.stderr.take().expect("piped"), 64 * 1024));

    let unpacker = Unpacker::start(name.clone(), local.to_path_buf(), None);
    let mut buf = vec![0u8; 256 * 1024];
    let mut read_error = None;
    loop {
        match stdout.read(&mut buf).await {
            Ok(0) => break,
            Ok(n) => {
                if !unpacker.feed(Bytes::copy_from_slice(&buf[..n])).await {
                    // The unpacker refused the archive; stop the transfer.
                    let _ = child.kill().await;
                    break;
                }
            }
            Err(e) => {
                read_error = Some(e);
                let _ = child.kill().await;
                break;
            }
        }
    }
    let unpacked = unpacker.finish().await;
    let status = tokio::time::timeout(Duration::from_secs(30), child.wait()).await;
    let stderr = stderr.await.map(|t| t.into_string()).unwrap_or_default();
    let code = match status {
        Ok(Ok(status)) => super::cli::exit_code(status),
        _ => {
            let _ = child.kill().await;
            -1
        }
    };

    // GNU tar exits 1 when a file changed while it was read; the copy is
    // still complete enough to use.
    let tar_ok = code == 0 || (code == 1 && stderr.contains("file changed as we read it"));
    if !tar_ok && unpacked.is_ok() {
        let out = Output {
            code,
            stdout: String::new(),
            stderr,
        };
        let lower = out.stderr.to_lowercase();
        if lower.contains("no such file or directory")
            && !lower.contains("executable file not found")
        {
            return Err(SandboxError::io(
                format!("downloading {remote}"),
                std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("{remote} does not exist in sandbox {container}"),
                ),
            ));
        }
        return Err(transfer_failure(
            cli,
            container,
            &out,
            &format!("copying {remote} out"),
        ));
    }
    if let Some(err) = read_error {
        return Err(SandboxError::io(
            format!("reading {remote} from `{} exec`", cli.name),
            err,
        ));
    }
    match unpacked {
        Ok(out) => {
            report_skipped(&format!("download of {remote}"), &out.skipped);
            Ok(())
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound && !tar_ok => Err(SandboxError::io(
            format!("downloading {remote}"),
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("{remote} does not exist in sandbox {container}"),
            ),
        )),
        Err(err) => Err(unpack_error(&remote, local, err)),
    }
}

/// The error for a `tar` transfer that failed inside the container.
fn transfer_failure(cli: &Cli, container: &str, out: &Output, what: &str) -> SandboxError {
    if let Some(err) = cli.daemon_problem(out) {
        return err;
    }
    let message = out.message();
    let lower = message.to_lowercase();
    if lower.contains("tar") && (lower.contains("not found") || out.code == 127 || out.code == 126)
    {
        return SandboxError::Unsupported(format!(
            "{what} needs `tar` inside the sandbox, and the image has none ({message}). Add it to \
             the task's image (for example `apt-get install -y tar` or `apk add tar`)."
        ));
    }
    if lower.contains("no space left on device") {
        return SandboxError::Transport(format!(
            "{what} in {container} failed: the sandbox's disk is full ({message}). Give the task more \
             disk, or have it write less."
        ));
    }
    SandboxError::Transport(format!(
        "{what} in {container} failed (exit code {}): {message}. If the container is gone, check \
         `{} ps -a --filter name={container}`.",
        out.code, cli.name
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_script_quotes_the_destination() {
        assert_eq!(
            extract_script("/app/data"),
            "mkdir -p /app/data && tar -xf - -C /app/data"
        );
        assert_eq!(
            extract_script("/app/my dir"),
            "mkdir -p '/app/my dir' && tar -xf - -C '/app/my dir'"
        );
    }
}
