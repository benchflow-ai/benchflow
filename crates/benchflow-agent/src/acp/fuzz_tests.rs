//! Property tests over what an ACP agent writes: the line reader on any
//! byte stream in any chunking, envelope decoding with terminal noise, and
//! the rewrite of over-long lines. None may panic; the reader never holds
//! more than its cap; a rewritten line fits its limit.

#[path = "../../../benchflow-task/tests/fuzz_support/mod.rs"]
mod fuzz_support;

use std::pin::Pin;

use fuzz_support::*;
use proptest::prelude::*;
use serde_json::json;
use tokio::io::AsyncRead;

use super::wire::{self, Line, LineReader};

proptest! {
    #![proptest_config(config())]

    /// Lines come back as written, in any chunking; those over the cap come
    /// back as their length only.
    #[test]
    fn line_reader(
        lines in prop::collection::vec(prop::collection::vec(any::<u8>().prop_map(|b| if b == b'\n' { b'x' } else { b }), 0..300), 0..12),
        last_newline in any::<bool>(),
        max in 1usize..200,
        chunk in 1usize..64,
    ) {
        timed("line_reader", move || {
            let mut stream = Vec::new();
            for (i, l) in lines.iter().enumerate() {
                stream.extend_from_slice(l);
                if i + 1 < lines.len() || last_newline {
                    stream.push(b'\n');
                }
            }
            let reader = tokio_test_reader(stream, chunk);
            let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
            let got = rt.block_on(async {
                let mut r = LineReader::new(reader, max);
                let mut out = Vec::new();
                while let Some(line) = r.next_line().await.unwrap() {
                    out.push(line);
                }
                out
            });
            let want: Vec<&Vec<u8>> = lines.iter().enumerate()
                .filter(|(i, l)| !(*i + 1 == lines.len() && !last_newline && l.is_empty()))
                .map(|(_, l)| l)
                .collect();
            prop_assert_eq!(got.len(), want.len());
            for (g, w) in got.iter().zip(want) {
                let mut w = w.clone();
                if w.last() == Some(&b'\r') {
                    w.pop();
                }
                match g {
                    Line::Text(t) => {
                        prop_assert!(t.len() <= max);
                        prop_assert_eq!(t, &w);
                    }
                    Line::TooLong(n) => prop_assert!(*n > max),
                }
            }
            Ok(())
        })?;
    }

    /// Any text: decoding never panics, terminal noise or not.
    #[test]
    fn decode_first(text in prop_oneof![
        "\\PC{0,80}",
        json_with(&["jsonrpc", "2.0", "id", "method", "params", "result", "error", "session/update", "sessionId", "update"]).prop_map(|v| v.to_string()),
        ("(\\x1b\\[[0-9;]{0,6}[A-Za-z]|\\x1b\\][^\\x07]{0,10}\\x07|[$#] ){0,4}", "\\{\"jsonrpc\":\"2.0\",\"method\":\"x\"\\}").prop_map(|(a, b)| a + &b),
    ]) {
        timed("decode_first", move || { let _ = wire::decode_first(&text); });
    }

    /// A rewritten line always fits the limit and keeps what identifies
    /// the message.
    #[test]
    fn shrink_fits(
        doc in json_with(&["jsonrpc", "id", "method", "params", "update", "content", "data", "text", "mimeType", "type", "image", "audio", "resource", "blob", "rawOutput", "sessionUpdate", "toolCallId"]),
        long in prop::collection::vec(("[A-Za-z0-9+/]{0,10}", 1usize..3000), 0..4),
        medium in prop::collection::vec(("[a-zé ]{1,8}", 1usize..1200), 0..80),
        limit in 2000usize..20_000,
    ) {
        let mut doc = json!({"jsonrpc": "2.0", "method": "session/update", "params": {"update": doc}});
        for (i, (s, n)) in long.iter().enumerate() {
            doc["params"][format!("big{i}")] = json!(s.repeat(*n));
        }
        for (i, (s, n)) in medium.iter().enumerate() {
            doc["params"]["many"][format!("g{}", i % 7)][format!("t{i}")] = json!(s.repeat(*n));
        }
        let line = serde_json::to_vec(&doc).unwrap();
        timed("shrink", move || {
            let Some(shrunk) = wire::shrink(&line, limit) else {
                return Err(TestCaseError::fail("a JSON object always has a rewrite"));
            };
            let size = serde_json::to_vec(&shrunk.value).unwrap().len();
            // The generated keys that are never rewritten hold short values,
            // so every rewrite fits.
            prop_assert!(shrunk.fits && size <= limit, "rewritten line is {size} bytes, limit {limit}, was {}", line.len());
            prop_assert_eq!(&shrunk.value["jsonrpc"], "2.0");
            prop_assert_eq!(&shrunk.value["method"], "session/update");
            Ok(())
        })?;
    }
}

/// A reader that hands out `data` `chunk` bytes at a time.
fn tokio_test_reader(data: Vec<u8>, chunk: usize) -> Pin<Box<dyn AsyncRead + Send>> {
    struct Chunked {
        data: Vec<u8>,
        at: usize,
        chunk: usize,
    }
    impl AsyncRead for Chunked {
        fn poll_read(
            mut self: Pin<&mut Self>,
            _: &mut std::task::Context<'_>,
            buf: &mut tokio::io::ReadBuf<'_>,
        ) -> std::task::Poll<std::io::Result<()>> {
            let n = self
                .chunk
                .min(self.data.len() - self.at)
                .min(buf.remaining());
            let at = self.at;
            buf.put_slice(&self.data[at..at + n]);
            self.at += n;
            std::task::Poll::Ready(Ok(()))
        }
    }
    Box::pin(Chunked { data, at: 0, chunk })
}
