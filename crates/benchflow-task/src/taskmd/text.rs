//! A file's text as the reference checker reads it, with a map back to the
//! file on disk.
//!
//! The reference reads files with Python's universal newlines: CR LF and a
//! lone CR both become LF. This module does the same, and remembers where
//! each removed CR was, so a byte offset found in the normalized text can be
//! turned back into an offset in the file as written (which is what a
//! diagnostic's span and a fix need). A lone CR becomes an LF in place, so
//! it moves nothing.

/// Normalized text and the offsets needed to map it back.
#[derive(Debug, Clone)]
pub(crate) struct Text {
    normalized: String,
    /// Offsets in the normalized text of each LF that had a CR before it.
    removed: Vec<usize>,
    /// Offsets in the normalized text where lines start.
    line_starts: Vec<usize>,
}

impl Text {
    /// Normalize `original`: each CR LF becomes LF, and so does a lone CR.
    pub(crate) fn new(original: &str) -> Self {
        let mut normalized = String::with_capacity(original.len());
        let mut removed = Vec::new();
        let mut rest = original;
        while let Some(i) = rest.find('\r') {
            normalized.push_str(&rest[..i]);
            if rest[i + 1..].starts_with('\n') {
                removed.push(normalized.len());
            } else {
                normalized.push('\n');
            }
            rest = &rest[i + 1..];
        }
        normalized.push_str(rest);
        let line_starts = crate::source::line_starts(&normalized);
        Self {
            normalized,
            removed,
            line_starts,
        }
    }

    /// The text with CR LF turned into LF.
    pub(crate) fn as_str(&self) -> &str {
        &self.normalized
    }

    /// The offset in the original file of the character at `offset` in the
    /// normalized text (use for the start of a span).
    pub(crate) fn start(&self, offset: usize) -> usize {
        offset + self.removed.partition_point(|&r| r <= offset)
    }

    /// The offset in the original file just after the character before
    /// `offset` (use for the exclusive end of a span).
    pub(crate) fn end(&self, offset: usize) -> usize {
        offset + self.removed.partition_point(|&r| r < offset)
    }

    /// A span in the normalized text as a span in the original file.
    pub(crate) fn span(&self, range: std::ops::Range<usize>) -> (usize, usize) {
        let start = self.start(range.start);
        (start, self.end(range.end).max(start))
    }

    /// The 1-based line of an offset in the normalized text.
    pub(crate) fn line(&self, offset: usize) -> usize {
        self.line_starts.partition_point(|&s| s <= offset)
    }

    /// The offset in the normalized text where 1-based `line` starts.
    pub(crate) fn line_start(&self, line: usize) -> usize {
        self.line_starts
            .get(line.saturating_sub(1))
            .copied()
            .unwrap_or(self.normalized.len())
    }

    /// The span in the original file of 1-based `line`, without its line end.
    pub(crate) fn line_span(&self, line: usize) -> (usize, usize) {
        let start = self.line_start(line);
        let end = self.normalized[start..]
            .find('\n')
            .map_or(self.normalized.len(), |i| start + i);
        self.span(start..end)
    }
}

/// Text with Python's universal newlines (CR LF and lone CR become LF), as
/// `Path.read_text()` returns it.
pub(crate) fn universal(text: &str) -> String {
    Text::new(text).normalized
}

/// A file read as Python's `read_text(encoding="utf-8")` reads it: `None`
/// when it cannot be read or is not UTF-8.
pub(crate) fn read_text(path: &std::path::Path) -> Option<String> {
    let bytes = std::fs::read(path).ok()?;
    String::from_utf8(bytes).ok().map(|t| universal(&t))
}

/// A file read as `read_text(errors="replace")` reads it.
pub(crate) fn read_text_lossy(path: &std::path::Path) -> Option<String> {
    let bytes = std::fs::read(path).ok()?;
    Some(universal(&String::from_utf8_lossy(&bytes)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crlf_offsets_map_back() {
        let original = "ab\r\ncd\r\n\r\ne";
        let t = Text::new(original);
        assert_eq!(t.as_str(), "ab\ncd\n\ne");
        // 'c' is at 3 in the normalized text and 4 in the original.
        assert_eq!(t.start(3), 4);
        assert_eq!(&original[t.start(3)..t.end(5)], "cd");
        // "ab" ends before the CR.
        assert_eq!(t.span(0..2), (0, 2));
        // 'e', after two CR LFs in a row.
        let (s, e) = t.span(7..8);
        assert_eq!(&original[s..e], "e");
    }

    #[test]
    fn lines_count_from_one() {
        let t = Text::new("a\nb\r\nc");
        assert_eq!(t.line(0), 1);
        assert_eq!(t.line(2), 2);
        assert_eq!(t.line(4), 3);
        assert_eq!(t.line_span(2), (2, 3));
        assert_eq!(t.line_span(3), (5, 6));
    }

    #[test]
    fn lone_cr_is_a_line_end() {
        let t = Text::new("a\rb\r\nc");
        assert_eq!(t.as_str(), "a\nb\nc");
        assert_eq!(t.line(2), 2);
        assert_eq!(t.span(4..5), (5, 6));
    }

    /// Python's universal newlines, for files read as text.
    #[test]
    fn universal_newlines() {
        assert_eq!(universal("a\r\nb\rc\n"), "a\nb\nc\n");
    }
}
