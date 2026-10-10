//! Bounded capture of a command's output.
//!
//! A command can print far more than anyone wants in memory (a test suite in
//! a loop, `yes`). [`TailBuffer`] keeps the last `cap` bytes, which is where
//! the error usually is, counts what it dropped, and says so in the text it
//! returns.

use tokio::io::{AsyncRead, AsyncReadExt};

/// How much of each of stdout and stderr a command's result keeps.
#[cfg(not(test))]
pub(crate) const OUTPUT_CAP: usize = 16 * 1024 * 1024;
/// Smaller in unit tests, so testing the cap stays cheap.
#[cfg(test)]
pub(crate) const OUTPUT_CAP: usize = 1024 * 1024;

/// The last `cap` bytes written to it.
#[derive(Debug)]
pub(crate) struct TailBuffer {
    cap: usize,
    buf: Vec<u8>,
    dropped: u64,
}

impl TailBuffer {
    pub(crate) fn new(cap: usize) -> Self {
        Self {
            cap,
            buf: Vec::new(),
            dropped: 0,
        }
    }

    pub(crate) fn push(&mut self, data: &[u8]) {
        self.buf.extend_from_slice(data);
        // Trim in batches so a stream of small writes stays linear.
        if self.buf.len() > self.cap.saturating_mul(2).max(64 * 1024) {
            self.trim();
        }
    }

    fn trim(&mut self) {
        if self.buf.len() > self.cap {
            let excess = self.buf.len() - self.cap;
            self.buf.drain(..excess);
            self.dropped += excess as u64;
        }
    }

    /// Bytes dropped from the front so far.
    #[cfg(test)]
    pub(crate) fn dropped(&self) -> u64 {
        self.dropped + self.buf.len().saturating_sub(self.cap) as u64
    }

    /// The kept bytes as text (invalid UTF-8 replaced), preceded by a note
    /// when earlier output was dropped.
    pub(crate) fn into_string(mut self) -> String {
        self.trim();
        let mut start = 0;
        if self.dropped > 0 {
            // Do not start in the middle of a UTF-8 sequence.
            while start < self.buf.len() && (self.buf[start] & 0xC0) == 0x80 {
                start += 1;
            }
        }
        let text = String::from_utf8_lossy(&self.buf[start..]);
        if self.dropped > 0 {
            format!(
                "[benchflow: {} earlier bytes of output were dropped; the last {} are kept]\n{text}",
                self.dropped + start as u64,
                self.buf.len() - start
            )
        } else {
            text.into_owned()
        }
    }
}

/// Read `reader` to its end into a [`TailBuffer`] of `cap` bytes. A read
/// error ends the capture early; whatever arrived is kept.
pub(crate) async fn read_tail(mut reader: impl AsyncRead + Unpin, cap: usize) -> TailBuffer {
    let mut tail = TailBuffer::new(cap);
    let mut chunk = vec![0u8; 64 * 1024];
    loop {
        match reader.read(&mut chunk).await {
            Ok(0) | Err(_) => break,
            Ok(n) => tail.push(&chunk[..n]),
        }
    }
    tail
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_everything_under_the_cap() {
        let mut t = TailBuffer::new(10);
        t.push(b"hello");
        assert_eq!(t.dropped(), 0);
        assert_eq!(t.into_string(), "hello");
    }

    #[test]
    fn keeps_the_tail_and_says_what_was_dropped() {
        let mut t = TailBuffer::new(4);
        for _ in 0..100_000 {
            t.push(b"ab");
        }
        t.push(b"WXYZ");
        assert_eq!(
            t.into_string(),
            "[benchflow: 200000 earlier bytes of output were dropped; the last 4 are kept]\nWXYZ"
        );
    }

    #[test]
    fn never_starts_inside_a_utf8_sequence() {
        let mut t = TailBuffer::new(3);
        t.push("aé€".as_bytes()); // 1 + 2 + 3 bytes
        let s = t.into_string();
        assert!(s.ends_with("\n€"), "{s:?}");
    }

    #[tokio::test]
    async fn reads_a_stream_to_its_end() {
        let data = vec![b'x'; 200_000];
        let tail = read_tail(&data[..], 1000).await;
        assert_eq!(tail.dropped(), 199_000);
    }
}
