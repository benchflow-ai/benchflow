//! Plumbing shared by both backends for moving files: checking remote
//! paths, packing local files into a temp file off the async threads, and
//! unpacking a tar stream that arrives in chunks.

use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use bytes::Bytes;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::SandboxError;
use crate::archive::{self, Limits, Packed, Skipped, Unpacked};

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

/// Log how many more entries a transfer left out than it kept.
pub(crate) fn report_more(what: &str, more: u64) {
    if more > 0 {
        tracing::warn!("{what}: left out {more} more entries");
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
///
/// Dropped before [`Unpacker::finish`] (a copy cut at its deadline), it
/// stops the thread and waits for it, so nothing is written at the
/// destination after the caller goes on to remove it (review F2-A-r9 L1).
pub(crate) struct Unpacker {
    tx: Option<mpsc::Sender<Bytes>>,
    task: Option<JoinHandle<io::Result<Unpacked>>>,
    stop: Arc<AtomicBool>,
    done: Arc<(Mutex<bool>, Condvar)>,
}

/// The longest a dropped unpacker waits for its thread.
const STOP_WAIT: Duration = Duration::from_secs(10);

impl Unpacker {
    /// Start unpacking into `dest` (see [`archive::unpack_limited`]).
    pub(crate) fn start(root: String, dest: PathBuf, limits: Limits) -> Self {
        let (tx, rx) = mpsc::channel(16);
        let stop = Arc::new(AtomicBool::new(false));
        let done = Arc::new((Mutex::new(false), Condvar::new()));
        let (thread_stop, thread_done) = (stop.clone(), done.clone());
        let task = tokio::task::spawn_blocking(move || {
            // Says the thread is done however it ends.
            struct Done(Arc<(Mutex<bool>, Condvar)>);
            impl Drop for Done {
                fn drop(&mut self) {
                    let (lock, cvar) = &*self.0;
                    *lock.lock().unwrap_or_else(|e| e.into_inner()) = true;
                    cvar.notify_all();
                }
            }
            let _done = Done(thread_done);
            let mut reader = ChannelReader {
                rx,
                chunk: Bytes::new(),
                stop: thread_stop,
            };
            let result = archive::unpack_limited(&mut reader, &root, &dest, limits);
            // Read what is left (tar's end padding) so the sender never
            // blocks on a full channel; stop at the first error.
            if result.is_ok() {
                let _ = io::copy(&mut reader, &mut io::sink());
            }
            result
        });
        Self {
            tx: Some(tx),
            task: Some(task),
            stop,
            done,
        }
    }

    /// Hand over the next chunk. False once the unpacker has stopped.
    pub(crate) async fn feed(&self, chunk: Bytes) -> bool {
        match &self.tx {
            Some(tx) => tx.send(chunk).await.is_ok(),
            None => false,
        }
    }

    /// Signal the end of the stream and wait for the result.
    pub(crate) async fn finish(mut self) -> io::Result<Unpacked> {
        drop(self.tx.take());
        let task = self.task.take().expect("finished once");
        task.await.map_err(io::Error::other)?
    }
}

impl Drop for Unpacker {
    fn drop(&mut self) {
        if self.task.is_none() {
            return;
        }
        self.stop.store(true, Ordering::SeqCst);
        drop(self.tx.take());
        let (lock, cvar) = &*self.done;
        let guard = lock.lock().unwrap_or_else(|e| e.into_inner());
        let _ = cvar.wait_timeout_while(guard, STOP_WAIT, |done| !*done);
    }
}

/// A blocking reader over chunks sent from async code.
struct ChannelReader {
    rx: mpsc::Receiver<Bytes>,
    chunk: Bytes,
    stop: Arc<AtomicBool>,
}

impl Read for ChannelReader {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if self.stop.load(Ordering::SeqCst) {
            return Err(io::Error::other("the copy was stopped"));
        }
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
        let unpacker = Unpacker::start("data".into(), dst.path().join("got"), Limits::default());
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
    /// A stream of folders or empty files past the entry budget fails with
    /// `FileTooLarge` through the unpacker Daytona's copies use (review
    /// F2-A-r12-3), with a content cap of one byte that it never reaches.
    #[tokio::test]
    async fn a_flood_of_entries_fails_at_the_budget() {
        let limits = Limits {
            bytes: Some(1),
            entries: Some(50),
        };
        for folders in [true, false] {
            let mut tar_bytes = Vec::new();
            {
                let mut b = tar::Builder::new(&mut tar_bytes);
                for i in 0..200 {
                    let mut h = tar::Header::new_gnu();
                    h.set_size(0);
                    h.set_mode(0o755);
                    h.set_entry_type(if folders {
                        tar::EntryType::Directory
                    } else {
                        tar::EntryType::Regular
                    });
                    b.append_data(&mut h, format!("data/e{i}"), &b""[..])
                        .unwrap();
                }
                b.finish().unwrap();
            }
            let dst = tempfile::tempdir().unwrap();
            let unpacker = Unpacker::start("data".into(), dst.path().join("got"), limits);
            for chunk in tar_bytes.chunks(4096) {
                if !unpacker.feed(Bytes::copy_from_slice(chunk)).await {
                    break;
                }
            }
            let e = unpacker.finish().await.unwrap_err();
            assert_eq!(e.kind(), io::ErrorKind::FileTooLarge, "{e}");
        }
    }

    /// Review F2-A-r9 L1: an unpacker dropped mid-stream (a copy cut at its
    /// deadline) has stopped writing when the drop returns, so removing
    /// what arrived removes it for good.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_dropped_unpacker_writes_nothing_after_the_drop() {
        let mut tar_bytes = Vec::new();
        {
            let mut b = tar::Builder::new(&mut tar_bytes);
            for i in 0..2_000 {
                let mut h = tar::Header::new_gnu();
                h.set_size(1);
                h.set_mode(0o644);
                b.append_data(&mut h, format!("data/f{i}"), &b"x"[..])
                    .unwrap();
            }
            b.finish().unwrap();
        }
        let dst = tempfile::tempdir().unwrap();
        let got = dst.path().join("got");
        let unpacker = Unpacker::start("data".into(), got.clone(), Limits::default());
        // Fill the channel, so the thread has a backlog to write.
        for chunk in tar_bytes.chunks(64 * 1024).take(16) {
            assert!(unpacker.feed(Bytes::copy_from_slice(chunk)).await);
        }
        drop(unpacker);
        let _ = std::fs::remove_dir_all(&got);
        tokio::time::sleep(Duration::from_millis(500)).await;
        assert!(!got.exists(), "the unpacker wrote after it was dropped");
    }
}
