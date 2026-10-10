//! Long-running processes in a Daytona sandbox, such as an ACP agent.
//!
//! The transport is a session command (see [`super::toolbox`]):
//!
//! - **stdin**: `POST .../command/{id}/input`, one request at a time so
//!   bytes arrive in order. The daemon ends every input with a newline when
//!   it lacks one, so stdin goes in whole lines: complete lines are batched
//!   (up to 4 MiB a request; one longer line goes alone, never split), and a
//!   line still being written waits for its newline. The input API carries
//!   text, so stdin must be UTF-8, which newline-delimited JSON is. Shutting
//!   stdin down sends what is left but cannot send end-of-file;
//!   [`ProcessHandle::kill`] ends the process.
//! - **stdout and stderr**: the `logs?follow=true` websocket, whose
//!   messages carry each line behind a `\x01\x01\x01` (stdout) or
//!   `\x02\x02\x02` (stderr) marker. The markers are removed; lines arrive
//!   byte for byte. If the socket drops while the process runs, it is
//!   reopened and the part already received is skipped.
//! - **exit code**: the session command's status.
//!
//! BenchFlow 0.8 used Daytona's PTY websocket instead. A PTY merges stderr
//! into stdout, echoes input and applies line discipline unless switched to
//! raw mode in time, and Daytona closed it (1008) on multi-megabyte
//! messages; session commands have none of these problems.

use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message;

use super::SandboxCore;
use super::exec::{Cleanup, kill_tagged, prepare};
use super::toolbox::{Stdin, Toolbox};
use crate::{Exec, Process, ProcessHandle, SandboxError};

/// Lines are batched into one stdin request up to this size.
const INPUT_BATCH: usize = 4 << 20;
/// Bytes of stdin queued before writes wait.
const INPUT_HIGH_WATER: u64 = 8 << 20;
/// stderr chunks buffered before new ones are dropped (stderr is
/// diagnostics; it must never stall stdout).
const STDERR_BUFFER: usize = 1024;

/// Start `exec` with live pipes.
pub(crate) async fn spawn(core: &SandboxCore, exec: Exec) -> Result<Process, SandboxError> {
    let toolbox = core.toolbox.clone();
    let prepared = prepare(core, &exec, Stdin::Pipe, true).await?;
    let session = toolbox.create_session().await?;
    let mut cleanup = Cleanup::new(&toolbox, &session, &prepared.exec_id);
    let command = match toolbox.start(&session, &prepared.line).await {
        Ok(command) => command,
        Err(err) => {
            cleanup.finished();
            return Err(err);
        }
    };
    let logs_url = toolbox.url(&format!(
        "/process/session/{session}/command/{command}/logs?follow=true"
    ));
    let socket = toolbox
        .api
        .websocket(&logs_url, "following a process's output")
        .await
        .map_err(|e| toolbox.error(e))?;

    let (out_tx, out_rx) = mpsc::channel::<Bytes>(64);
    let (err_tx, err_rx) = mpsc::channel::<Bytes>(STDERR_BUFFER);
    let reader = tokio::spawn(read_output(
        toolbox.clone(),
        session.clone(),
        command.clone(),
        logs_url,
        socket,
        out_tx,
        err_tx,
    ));
    let stdin = StdinWriter::start(toolbox.clone(), session.clone(), command.clone());
    Ok(Process {
        stdin: Box::pin(stdin),
        stdout: Box::pin(ChunkReader::new(out_rx)),
        stderr: Box::pin(ChunkReader::new(err_rx)),
        handle: Box::new(DaytonaProcess {
            toolbox,
            session,
            command,
            exec_id: prepared.exec_id,
            reader: Some(reader),
            exit_code: None,
            cleanup: Some(cleanup),
        }),
    })
}

/// Which stream a byte belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stream {
    Out,
    Err,
}

/// Splits the log stream at its `\x01\x01\x01` / `\x02\x02\x02` markers.
#[derive(Debug, Default)]
struct Demux {
    current: Option<Stream>,
    /// The end of the last chunk, when it may be the start of a marker.
    carry: Vec<u8>,
}

impl Demux {
    /// The data in `chunk`, by stream, in order.
    fn push(&mut self, chunk: &[u8]) -> Vec<(Stream, Bytes)> {
        let mut buf = std::mem::take(&mut self.carry);
        buf.extend_from_slice(chunk);
        let mut out = Vec::new();
        let mut start = 0;
        let mut i = 0;
        while i + 3 <= buf.len() {
            let marker = match &buf[i..i + 3] {
                [1, 1, 1] => Some(Stream::Out),
                [2, 2, 2] => Some(Stream::Err),
                _ => None,
            };
            if let Some(stream) = marker {
                self.emit(&mut out, &buf[start..i]);
                self.current = Some(stream);
                i += 3;
                start = i;
            } else {
                i += 1;
            }
        }
        // Hold back a trailing `\x01` or `\x01\x01` (or `\x02`...): it may be
        // the first part of a marker split across messages.
        let unscanned = &buf[start.max(buf.len().saturating_sub(2))..];
        let keep = match unscanned {
            [.., a, b] if a == b && (*a == 1 || *a == 2) => 2,
            [.., b] if *b == 1 || *b == 2 => 1,
            _ => 0,
        };
        let end = buf.len() - keep;
        self.emit(&mut out, &buf[start..end]);
        self.carry = buf[end..].to_vec();
        out
    }

    /// Whatever is held back, at the end of the stream.
    fn finish(&mut self) -> Vec<(Stream, Bytes)> {
        let mut out = Vec::new();
        let carry = std::mem::take(&mut self.carry);
        self.emit(&mut out, &carry);
        out
    }

    fn emit(&self, out: &mut Vec<(Stream, Bytes)>, data: &[u8]) {
        if !data.is_empty() {
            out.push((
                self.current.unwrap_or(Stream::Out),
                Bytes::copy_from_slice(data),
            ));
        }
    }
}

type Socket =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

/// Copy the follow socket into the stdout and stderr channels until the
/// command ends. Reopens a dropped socket while the command still runs.
async fn read_output(
    toolbox: Toolbox,
    session: String,
    command: String,
    url: String,
    socket: Socket,
    out_tx: mpsc::Sender<Bytes>,
    err_tx: mpsc::Sender<Bytes>,
) {
    let mut demux = Demux::default();
    let mut received: u64 = 0;
    let mut socket = Some(socket);
    let mut reopened = 0;
    let mut dropped_stderr: u64 = 0;
    'outer: loop {
        let Some(mut ws) = socket.take() else { break };
        // Bytes of a reopened socket's replay already delivered.
        let mut skip = received;
        let mut idle_checks = 0;
        loop {
            let next = tokio::time::timeout(Duration::from_secs(5), ws.next()).await;
            let message = match next {
                Ok(Some(Ok(message))) => message,
                Ok(Some(Err(err))) => {
                    tracing::debug!("output socket of {command}: {err}");
                    break;
                }
                Ok(None) => break,
                Err(_) => {
                    // Quiet for 5 s: has the command ended without the
                    // socket closing?
                    idle_checks += 1;
                    if let Ok(Some(_)) = toolbox.exit_code(&session, &command).await {
                        if idle_checks >= 2 {
                            break 'outer;
                        }
                    } else {
                        idle_checks = 0;
                    }
                    continue;
                }
            };
            idle_checks = 0;
            let data: Bytes = match message {
                Message::Binary(b) => b,
                Message::Text(t) => Bytes::copy_from_slice(t.as_bytes()),
                Message::Close(_) => break 'outer,
                _ => continue,
            };
            let data = if skip > 0 {
                let n = (skip as usize).min(data.len());
                skip -= n as u64;
                data.slice(n..)
            } else {
                data
            };
            if data.is_empty() {
                continue;
            }
            received += data.len() as u64;
            for (stream, bytes) in demux.push(&data) {
                if !deliver(stream, bytes, &out_tx, &err_tx, &mut dropped_stderr).await {
                    break 'outer;
                }
            }
        }
        // The socket ended without a close frame. Reopen it if the command
        // still runs.
        match toolbox.exit_code(&session, &command).await {
            Ok(None) if reopened < 5 => {
                reopened += 1;
                tokio::time::sleep(Duration::from_millis(500 * reopened)).await;
                match toolbox
                    .api
                    .websocket(&url, "following a process's output")
                    .await
                {
                    Ok(ws) => socket = Some(ws),
                    Err(err) => tracing::warn!(
                        "lost the output of a process in {}: {}",
                        toolbox.sandbox_id,
                        err.message()
                    ),
                }
            }
            _ => break,
        }
    }
    for (stream, bytes) in demux.finish() {
        deliver(stream, bytes, &out_tx, &err_tx, &mut dropped_stderr).await;
    }
    if dropped_stderr > 0 {
        tracing::warn!("dropped {dropped_stderr} bytes of a process's stderr nobody was reading");
    }
}

/// Send to the right channel; false when stdout's reader is gone.
async fn deliver(
    stream: Stream,
    bytes: Bytes,
    out_tx: &mpsc::Sender<Bytes>,
    err_tx: &mpsc::Sender<Bytes>,
    dropped_stderr: &mut u64,
) -> bool {
    match stream {
        Stream::Out => out_tx.send(bytes).await.is_ok(),
        Stream::Err => {
            let len = bytes.len() as u64;
            if let Err(mpsc::error::TrySendError::Full(_)) = err_tx.try_send(bytes) {
                *dropped_stderr += len;
            }
            true
        }
    }
}

/// An [`AsyncRead`] over chunks from a channel.
struct ChunkReader {
    rx: mpsc::Receiver<Bytes>,
    chunk: Bytes,
}

impl ChunkReader {
    fn new(rx: mpsc::Receiver<Bytes>) -> Self {
        Self {
            rx,
            chunk: Bytes::new(),
        }
    }
}

impl AsyncRead for ChunkReader {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        while self.chunk.is_empty() {
            match self.rx.poll_recv(cx) {
                Poll::Ready(Some(chunk)) => self.chunk = chunk,
                Poll::Ready(None) => return Poll::Ready(Ok(())),
                Poll::Pending => return Poll::Pending,
            }
        }
        let n = buf.remaining().min(self.chunk.len());
        let part = self.chunk.split_to(n);
        buf.put_slice(&part);
        Poll::Ready(Ok(()))
    }
}

/// State shared by [`StdinWriter`] and its sender task.
#[derive(Default)]
struct StdinShared {
    /// Bytes of complete lines handed to the sender and not yet delivered.
    queued: AtomicU64,
    waker: Mutex<Option<Waker>>,
    error: Mutex<Option<String>>,
}

impl StdinShared {
    fn wake(&self) {
        if let Some(waker) = self.waker.lock().expect("lock").take() {
            waker.wake();
        }
    }

    fn park(&self, cx: &Context<'_>) {
        *self.waker.lock().expect("lock") = Some(cx.waker().clone());
    }

    fn fail(&self, message: String) {
        *self.error.lock().expect("lock") = Some(message);
        self.wake();
    }

    fn error(&self) -> Option<std::io::Error> {
        self.error
            .lock()
            .expect("lock")
            .clone()
            .map(|m| std::io::Error::new(std::io::ErrorKind::BrokenPipe, m))
    }
}

/// The process's stdin. Writes are cut into whole lines, which a task sends
/// in order; the part after the last newline waits for the rest of its line.
struct StdinWriter {
    tx: Option<mpsc::UnboundedSender<Vec<u8>>>,
    /// The line being written, not yet ended by a newline.
    partial: Vec<u8>,
    shared: Arc<StdinShared>,
}

impl StdinWriter {
    fn start(toolbox: Toolbox, session: String, command: String) -> Self {
        let (tx, mut rx) = mpsc::unbounded_channel::<Vec<u8>>();
        let shared = Arc::new(StdinShared::default());
        let state = shared.clone();
        tokio::spawn(async move {
            while let Some(first) = rx.recv().await {
                // Batch whole lines; one long line still goes alone.
                let mut batch = first;
                while batch.len() < INPUT_BATCH {
                    match rx.try_recv() {
                        Ok(more) => batch.extend(more),
                        Err(_) => break,
                    }
                }
                let text = match String::from_utf8(batch) {
                    Ok(text) => text,
                    Err(_) => {
                        state.fail(
                            "stdin data is not UTF-8 text; Daytona's input API carries text only"
                                .to_string(),
                        );
                        return;
                    }
                };
                if let Err(err) = toolbox.input(&session, &command, &text).await {
                    state.fail(err.to_string());
                    return;
                }
                state.queued.fetch_sub(text.len() as u64, Ordering::SeqCst);
                state.wake();
            }
        });
        Self {
            tx: Some(tx),
            partial: Vec::new(),
            shared,
        }
    }

    /// Hand `bytes` (whole lines) to the sender.
    fn send(&mut self, bytes: Vec<u8>) -> std::io::Result<()> {
        let Some(tx) = &self.tx else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "stdin was shut down",
            ));
        };
        self.shared
            .queued
            .fetch_add(bytes.len() as u64, Ordering::SeqCst);
        tx.send(bytes).map_err(|_| {
            self.shared.error().unwrap_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "the process's stdin is closed",
                )
            })
        })
    }
}

impl AsyncWrite for StdinWriter {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        if let Some(err) = self.shared.error() {
            return Poll::Ready(Err(err));
        }
        if self.tx.is_none() {
            return Poll::Ready(Err(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "stdin was shut down",
            )));
        }
        if self.shared.queued.load(Ordering::SeqCst) >= INPUT_HIGH_WATER {
            self.shared.park(cx);
            if self.shared.queued.load(Ordering::SeqCst) >= INPUT_HIGH_WATER {
                return Poll::Pending;
            }
        }
        self.partial.extend_from_slice(buf);
        if let Some(last) = self.partial.iter().rposition(|b| *b == b'\n') {
            let rest = self.partial.split_off(last + 1);
            let lines = std::mem::replace(&mut self.partial, rest);
            if let Err(err) = self.send(lines) {
                return Poll::Ready(Err(err));
            }
        }
        Poll::Ready(Ok(buf.len()))
    }

    /// Ready once every complete line is delivered. A line without its
    /// newline yet stays buffered: sent alone, Daytona would end it early.
    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        if let Some(err) = self.shared.error() {
            return Poll::Ready(Err(err));
        }
        if self.shared.queued.load(Ordering::SeqCst) == 0 {
            return Poll::Ready(Ok(()));
        }
        self.shared.park(cx);
        if self.shared.queued.load(Ordering::SeqCst) == 0 {
            Poll::Ready(Ok(()))
        } else {
            Poll::Pending
        }
    }

    /// Send what is left (Daytona ends an unfinished last line with a
    /// newline), wait for delivery, and stop accepting writes.
    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        if !self.partial.is_empty() {
            let rest = std::mem::take(&mut self.partial);
            if let Err(err) = self.send(rest) {
                return Poll::Ready(Err(err));
            }
        }
        match self.as_mut().poll_flush(cx) {
            Poll::Ready(Ok(())) => {
                self.tx = None;
                Poll::Ready(Ok(()))
            }
            other => other,
        }
    }
}

/// A process started by [`spawn`].
struct DaytonaProcess {
    toolbox: Toolbox,
    session: String,
    command: String,
    exec_id: String,
    reader: Option<tokio::task::JoinHandle<()>>,
    exit_code: Option<i32>,
    cleanup: Option<Cleanup>,
}

#[async_trait]
impl ProcessHandle for DaytonaProcess {
    async fn wait(&mut self) -> Result<i32, SandboxError> {
        if let Some(code) = self.exit_code {
            return Ok(code);
        }
        // The output ends when the command does; then its code is known.
        if let Some(reader) = self.reader.take() {
            let _ = reader.await;
        }
        let mut nap = Duration::from_millis(200);
        let code = loop {
            if let Some(code) = self.toolbox.exit_code(&self.session, &self.command).await? {
                break code;
            }
            tokio::time::sleep(nap).await;
            nap = (nap * 2).min(Duration::from_secs(2));
        };
        self.exit_code = Some(code);
        if let Some(mut cleanup) = self.cleanup.take() {
            cleanup.finished();
        }
        Ok(code)
    }

    async fn kill(&mut self) -> Result<(), SandboxError> {
        kill_tagged(&self.toolbox, &self.exec_id).await
    }
}

impl Drop for DaytonaProcess {
    fn drop(&mut self) {
        if let Some(reader) = self.reader.take() {
            reader.abort();
        }
        // `cleanup` kills what is left (unless the process finished) and
        // closes the session as it drops.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect(parts: Vec<(Stream, Bytes)>) -> Vec<(Stream, String)> {
        parts
            .into_iter()
            .map(|(s, b)| (s, String::from_utf8_lossy(&b).into_owned()))
            .collect()
    }

    #[test]
    fn demux_splits_streams_and_drops_markers() {
        let mut d = Demux::default();
        let parts =
            collect(d.push(b"\x01\x01\x01out line\n\x02\x02\x02err line\n\x01\x01\x01more\n"));
        assert_eq!(
            parts,
            vec![
                (Stream::Out, "out line\n".to_string()),
                (Stream::Err, "err line\n".to_string()),
                (Stream::Out, "more\n".to_string()),
            ]
        );
    }

    #[test]
    fn demux_handles_markers_split_across_messages() {
        let mut d = Demux::default();
        let mut parts = collect(d.push(b"\x01\x01\x01a\n\x02"));
        parts.extend(collect(d.push(b"\x02")));
        parts.extend(collect(d.push(b"\x02e\n\x01")));
        parts.extend(collect(d.push(b"\x01\x01b")));
        parts.extend(collect(d.finish()));
        let joined: Vec<(Stream, String)> =
            parts.into_iter().filter(|(_, s)| !s.is_empty()).collect();
        assert_eq!(
            joined,
            vec![
                (Stream::Out, "a\n".to_string()),
                (Stream::Err, "e\n".to_string()),
                (Stream::Out, "b".to_string()),
            ]
        );
    }

    #[test]
    fn demux_keeps_a_lone_control_byte_that_is_not_a_marker() {
        let mut d = Demux::default();
        let mut parts = collect(d.push(b"\x01\x01\x01x\x01"));
        parts.extend(collect(d.push(b"y\n")));
        let text: String = parts.iter().map(|(_, s)| s.as_str()).collect();
        assert_eq!(text, "x\x01y\n");
    }

    #[tokio::test]
    async fn chunk_reader_reads_in_order_then_ends() {
        use tokio::io::AsyncReadExt;
        let (tx, rx) = mpsc::channel(4);
        tx.send(Bytes::from_static(b"hello ")).await.unwrap();
        tx.send(Bytes::from_static(b"world")).await.unwrap();
        drop(tx);
        let mut reader = ChunkReader::new(rx);
        let mut text = String::new();
        reader.read_to_string(&mut text).await.unwrap();
        assert_eq!(text, "hello world");
    }
}
