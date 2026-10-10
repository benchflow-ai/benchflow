//! Plumbing shared by both backends for moving files: checking remote
//! paths, packing local files into a temp file off the async threads, and
//! unpacking a tar stream that arrives in chunks.

use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use bytes::Bytes;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::SandboxError;
use crate::archive::{self, Packed, Skipped, Unpacked};

/// `remote` as a clean absolute path: `.` and `..` resolved, no repeated or
/// trailing `/`.
pub(crate) fn remote_path(remote: &str) -> Result<String, SandboxError> {
    if !remote.starts_with('/') {
        return Err(SandboxError::Unsupported(format!(
            "sandbox path {remote:?} is not absolute; BenchFlow needs absolute paths inside \
             sandboxes (this is a bug in the caller)"
        )));
    }
    let mut parts: Vec<&str> = Vec::new();
    for part in remote.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            p => parts.push(p),
        }
    }
    Ok(format!("/{}", parts.join("/")))
}

/// Split a clean absolute path into its parent folder and its last name.
/// `/` splits into `/` and `.`.
pub(crate) fn split_remote(remote: &str) -> (String, String) {
    match remote.rsplit_once('/') {
        Some(("", "")) | None => ("/".to_string(), ".".to_string()),
        Some(("", name)) => ("/".to_string(), name.to_string()),
        Some((parent, name)) => (parent.to_string(), name.to_string()),
    }
}

/// Log the entries a transfer left out, one line each (at most 20).
pub(crate) fn report_skipped(what: &str, skipped: &[Skipped]) {
    for s in skipped.iter().take(20) {
        tracing::warn!("{what}: left out {} ({})", s.path, s.reason);
    }
    if skipped.len() > 20 {
        tracing::warn!("{what}: left out {} more paths", skipped.len() - 20);
    }
}

/// Pack a local file or folder into an anonymous temp file, rewound.
/// `as_name` packs a file under another name.
pub(crate) async fn pack_to_tempfile(
    local: &Path,
    as_name: Option<String>,
) -> Result<(std::fs::File, Packed), SandboxError> {
    let local = local.to_path_buf();
    let shown = local.display().to_string();
    let result = tokio::task::spawn_blocking(move || -> io::Result<(std::fs::File, Packed)> {
        let mut file = tempfile::tempfile()?;
        let packed = match &as_name {
            Some(name) => archive::pack_file_as(&local, name, &mut file)?,
            None => archive::pack(&local, &mut file)?,
        };
        file.seek(SeekFrom::Start(0))?;
        Ok((file, packed))
    })
    .await
    .map_err(|e| SandboxError::io("packing files", io::Error::other(e)))?;
    result.map_err(|e| SandboxError::io(format!("packing {shown} for upload"), e))
}

/// An archive holding one file, in an anonymous temp file, rewound.
pub(crate) async fn pack_bytes_to_tempfile(
    name: String,
    contents: Vec<u8>,
    mode: u32,
) -> Result<std::fs::File, SandboxError> {
    let result = tokio::task::spawn_blocking(move || -> io::Result<std::fs::File> {
        let mut file = tempfile::tempfile()?;
        archive::pack_bytes(&name, &contents, mode, &mut file)?;
        file.seek(SeekFrom::Start(0))?;
        Ok(file)
    })
    .await
    .map_err(|e| SandboxError::io("packing a file", io::Error::other(e)))?;
    result.map_err(|e| SandboxError::io("packing a file for upload", e))
}

/// Unpacks a tar stream fed chunk by chunk, on a blocking thread.
pub(crate) struct Unpacker {
    tx: mpsc::Sender<Bytes>,
    task: JoinHandle<io::Result<Unpacked>>,
}

impl Unpacker {
    /// Start unpacking into `dest` (see [`archive::unpack`]).
    pub(crate) fn start(root: String, dest: PathBuf, max_bytes: Option<u64>) -> Self {
        let (tx, rx) = mpsc::channel(16);
        let task = tokio::task::spawn_blocking(move || {
            let mut reader = ChannelReader {
                rx,
                chunk: Bytes::new(),
            };
            let result = archive::unpack(&mut reader, &root, &dest, max_bytes);
            // Read what is left (tar's end padding) so the sender never
            // blocks on a full channel; stop at the first error.
            if result.is_ok() {
                let _ = io::copy(&mut reader, &mut io::sink());
            }
            result
        });
        Self { tx, task }
    }

    /// Hand over the next chunk. False once the unpacker has stopped.
    pub(crate) async fn feed(&self, chunk: Bytes) -> bool {
        self.tx.send(chunk).await.is_ok()
    }

    /// Signal the end of the stream and wait for the result.
    pub(crate) async fn finish(self) -> io::Result<Unpacked> {
        drop(self.tx);
        self.task.await.map_err(io::Error::other)?
    }
}

/// A blocking reader over chunks sent from async code.
struct ChannelReader {
    rx: mpsc::Receiver<Bytes>,
    chunk: Bytes,
}

impl Read for ChannelReader {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        while self.chunk.is_empty() {
            match self.rx.blocking_recv() {
                Some(chunk) => self.chunk = chunk,
                None => return Ok(0),
            }
        }
        let n = out.len().min(self.chunk.len());
        out[..n].copy_from_slice(&self.chunk.split_to(n));
        Ok(n)
    }
}

/// The error for an extraction that failed on the host side.
pub(crate) fn unpack_error(remote: &str, local: &Path, err: io::Error) -> SandboxError {
    SandboxError::io(
        format!(
            "copying {remote} out of the sandbox into {}",
            local.display()
        ),
        err,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remote_paths_are_cleaned() {
        assert_eq!(remote_path("/a//b/./c/").unwrap(), "/a/b/c");
        assert_eq!(remote_path("/a/b/../c").unwrap(), "/a/c");
        assert_eq!(remote_path("/../..").unwrap(), "/");
        assert!(remote_path("relative/x").is_err());
    }

    #[test]
    fn remote_paths_split_into_parent_and_name() {
        assert_eq!(split_remote("/app/out"), ("/app".into(), "out".into()));
        assert_eq!(split_remote("/out"), ("/".into(), "out".into()));
        assert_eq!(split_remote("/"), ("/".into(), ".".into()));
    }

    #[tokio::test]
    async fn unpacks_a_stream_fed_in_small_chunks() {
        let src = tempfile::tempdir().unwrap();
        let root = src.path().join("data");
        std::fs::create_dir_all(root.join("x")).unwrap();
        std::fs::write(root.join("x/f.txt"), "hello").unwrap();
        let mut tar_bytes = Vec::new();
        // Build what `tar -cf - -C <parent> data` would send.
        {
            let mut b = tar::Builder::new(&mut tar_bytes);
            b.append_dir_all("data", &root).unwrap();
            b.finish().unwrap();
        }
        let dst = tempfile::tempdir().unwrap();
        let unpacker = Unpacker::start("data".into(), dst.path().join("got"), None);
        for chunk in tar_bytes.chunks(7) {
            assert!(unpacker.feed(Bytes::copy_from_slice(chunk)).await);
        }
        let out = unpacker.finish().await.unwrap();
        assert_eq!(out.files, 1);
        assert_eq!(
            std::fs::read_to_string(dst.path().join("got/x/f.txt")).unwrap(),
            "hello"
        );
    }

    #[tokio::test]
    async fn packs_into_a_rewound_temp_file() {
        let src = tempfile::tempdir().unwrap();
        std::fs::write(src.path().join("a"), "1").unwrap();
        let (mut file, packed) = pack_to_tempfile(src.path(), None).await.unwrap();
        assert_eq!(packed.files, 1);
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).unwrap();
        assert!(bytes.len() >= 1024);
    }
}
