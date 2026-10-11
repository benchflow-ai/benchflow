//! Copying files in and out of a container through `tar` streams.
//!
//! Upload packs on the host and pipes the archive into `docker exec -i -u 0
//! <name> /bin/sh -c 'mkdir -p <dst> && tar -xf - -C <dst>'`, so files land
//! owned by root with their modes. Download runs `docker exec -u 0 <name>
//! tar -cf - -C <parent> <name>` and unpacks on the host, refusing anything
//! but regular files and folders (see [`crate::archive::unpack`]). Both run
//! with `PATH` set to [`crate::shell::SYSTEM_PATH`] and the loader variables
//! set to nothing.

use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use bytes::Bytes;
use tokio::io::AsyncReadExt;

use super::cli::{Cli, Output};
use super::exec::plumbing_env_args;
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

/// `docker exec` arguments, as root with the plumbing's environment
/// ([`plumbing_env_args`]: the copies find `sh` and `tar` in the system
/// folders, never in a folder on the image's `PATH` the agent could write,
/// and load nothing the image's loader variables name), then `rest`.
fn root_args(interactive: bool, container: &str, rest: &[&str]) -> Vec<String> {
    let mut args = vec!["exec".to_string()];
    if interactive {
        args.push("-i".to_string());
    }
    args.extend(["-u".to_string(), "0".to_string()]);
    args.extend(plumbing_env_args());
    args.push(container.to_string());
    args.extend(rest.iter().map(|s| s.to_string()));
    args
}

/// The `docker exec` arguments that unpack a tar on stdin into `dst`.
pub(crate) fn extract_args(container: &str, dst: &str) -> Vec<String> {
    root_args(true, container, &["/bin/sh", "-c", &extract_script(dst)])
}

/// The `docker exec` arguments that pack `name` in `parent` to stdout. The
/// name follows `--`: one the agent chose may start with `-`.
pub(crate) fn download_args(container: &str, parent: &str, name: &str) -> Vec<String> {
    root_args(
        false,
        container,
        &["tar", "-cf", "-", "-C", parent, "--", name],
    )
}

/// The shell command that unpacks a tar on stdin into `dst`. `umask 022` so
/// a parent folder `mkdir -p` makes comes out `0755`, not the `0777` a
/// `docker exec` umask of `0000` would give (the files' own modes come from
/// the archive).
pub(crate) fn extract_script(dst: &str) -> String {
    let d = quote(dst);
    format!("umask 022; mkdir -p {d} && tar -xf - -C {d}")
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
    max_bytes: Option<u64>,
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

    let unpacker = Unpacker::start(name.clone(), local.to_path_buf(), max_bytes);
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
    // A failed exec that sent nothing (a stopped container, a daemon that
    // failed this one exec) leaves the unpacker with an empty archive, as a
    // missing file does: only `tar`'s own words say the file is missing.
    let empty = matches!(&unpacked, Err(e) if e.kind() == std::io::ErrorKind::NotFound);
    if !tar_ok && (unpacked.is_ok() || empty) {
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
        // Nothing arrived and the copy failed: the folder is missing only
        // when tar says so; a daemon hiccup, a missing `tar` or an exec
        // error streams nothing too.
        Err(err) if err.kind() == std::io::ErrorKind::NotFound && !tar_ok => {
            let lower = stderr.to_lowercase();
            if lower.contains("no such file or directory")
                && !lower.contains("executable file not found")
            {
                Err(SandboxError::io(
                    format!("downloading {remote}"),
                    std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        format!("{remote} does not exist in sandbox {container}"),
                    ),
                ))
            } else {
                let out = Output {
                    code,
                    stdout: String::new(),
                    stderr,
                };
                Err(transfer_failure(
                    cli,
                    container,
                    &out,
                    &format!("copying {remote} out"),
                ))
            }
        }
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
    if let Some(err) = super::exec::container_gone(cli, container, &out.stderr) {
        return err;
    }
    if lower.contains("no space left on device") {
        return SandboxError::Content(format!(
            "{what} in {container} failed: the sandbox's disk is full ({message}). Give the task more \
             disk, or have it write less."
        ));
    }
    // tar ran and refused part of what it copies (exit 1 or 2 with its
    // own words): the files' fault, not the connection's.
    if (out.code == 1 || out.code == 2) && lower.contains("tar:") {
        return SandboxError::Content(format!(
            "{what} in {container} failed (exit code {}): {message}",
            out.code
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
            "umask 022; mkdir -p /app/data && tar -xf - -C /app/data"
        );
        assert_eq!(
            extract_script("/app/my dir"),
            "umask 022; mkdir -p '/app/my dir' && tar -xf - -C '/app/my dir'"
        );
    }

    /// A download from a glibc image gives a clean tar: no auxiliary-vector
    /// text in front. This fails on the branch before the loader-variable
    /// split, where `download_args` passed `-e LD_SHOW_AUXV=` and glibc then
    /// printed about 20 `AT_...` lines to stdout from `tar`, which broke the
    /// archive and every graded Docker trial (R107). It needs a Docker daemon
    /// (the glibc is in the container, so any host will do), so it is ignored
    /// by default and CI's ubuntu job runs it; asked for without Docker, it
    /// fails. `BF_DIND_GLIBC_TEST` may name another glibc image. The control
    /// re-adds `-e LD_SHOW_AUXV=` and asserts the output is then dirty, so
    /// the image and the mechanism are real.
    #[test]
    #[ignore = "needs a Docker daemon and a glibc image; cargo test -p benchflow-sandbox --lib -- --ignored a_glibc_download"]
    fn a_glibc_download_is_not_corrupted_by_an_empty_loader_flag() {
        let image = std::env::var("BF_DIND_GLIBC_TEST")
            .ok()
            .filter(|v| !v.is_empty() && v != "1")
            .unwrap_or_else(|| "ubuntu:24.04".into());
        let info = std::process::Command::new("docker")
            .args(["info", "--format", "{{.ServerVersion}}"])
            .output();
        assert!(
            matches!(&info, Ok(o) if o.status.success()),
            "this test was asked for, but no Docker daemon answers: {info:?}"
        );
        let docker = |args: &[&str], stdin: Option<&[u8]>| -> (Vec<u8>, Vec<u8>, bool) {
            use std::io::Write as _;
            let mut cmd = std::process::Command::new("docker");
            cmd.args(args)
                .stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped());
            let mut child = cmd.spawn().unwrap();
            if let Some(data) = stdin {
                child.stdin.take().unwrap().write_all(data).unwrap();
            }
            let out = child.wait_with_output().unwrap();
            (out.stdout, out.stderr, out.status.success())
        };
        let name = format!("bf-d10b-glibc-{}", std::process::id());
        // Remove the container even if an assertion below panics.
        struct Rm(String);
        impl Drop for Rm {
            fn drop(&mut self) {
                let _ = std::process::Command::new("docker")
                    .args(["rm", "-f", &self.0])
                    .output();
            }
        }
        let (_, err, ok) = docker(
            &[
                "run",
                "-d",
                "--rm",
                "--name",
                &name,
                "--entrypoint",
                "/bin/sh",
                &image,
                "-c",
                "sleep 300",
            ],
            None,
        );
        assert!(ok, "docker run: {}", String::from_utf8_lossy(&err));
        let _rm = Rm(name.clone());
        let seed = docker(
            &[
                "exec",
                "-u",
                "0",
                &name,
                "/bin/sh",
                "-c",
                "mkdir -p /logs/verifier && printf 1 > /logs/verifier/reward.txt",
            ],
            None,
        );
        assert!(seed.2, "seed: {}", String::from_utf8_lossy(&seed.1));

        // The real download arguments: no `AT_` text, and the archive opens.
        let args: Vec<String> = download_args(&name, "/logs", "verifier");
        let argv: Vec<&str> = args.iter().map(String::as_str).collect();
        let (stdout, err, ok) = docker(&argv, None);
        assert!(ok, "download: {}", String::from_utf8_lossy(&err));
        assert!(
            !stdout.windows(3).any(|w| w == b"AT_"),
            "the download stream carries auxiliary-vector text: {:?}",
            String::from_utf8_lossy(&stdout[..stdout.len().min(200)])
        );
        let mut archive = tar::Archive::new(&stdout[..]);
        let entries: Vec<_> = archive
            .entries()
            .unwrap()
            .map(|e| e.unwrap().path().unwrap().display().to_string())
            .collect();
        assert!(
            entries.iter().any(|p| p.contains("reward.txt")),
            "no reward.txt in the archive: {entries:?}"
        );

        // Control: with an empty `LD_SHOW_AUXV` added, glibc's tar prints the
        // auxiliary vector, so the image and the mechanism are genuine.
        let mut dirty = vec!["exec", "-u", "0", "-e", "LD_SHOW_AUXV="];
        dirty.extend([name.as_str(), "tar", "-cf", "-", "-C", "/logs", "verifier"]);
        let (control, _, _) = docker(&dirty, None);
        let had_auxv = control.windows(3).any(|w| w == b"AT_");
        assert!(
            had_auxv,
            "the control did not show the auxiliary vector; is {image} glibc?"
        );
    }
}
