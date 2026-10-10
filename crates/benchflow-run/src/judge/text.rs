//! Evidence files as text for a judge: plain text as written, Word, Excel,
//! and PowerPoint files from their XML, PDF text from its content streams.
//! Anything else is noted as not readable, never sent as bytes.

use std::io::Read;

/// The most bytes a document may unpack to (a zip bomb stops here).
const UNPACK_CAP: u64 = 64 * 1024 * 1024;

/// A file's text, or why there is none.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Text {
    /// The text, and what kind of file it came from (`markdown`, `docx`).
    Read { kind: &'static str, text: String },
    /// Why it cannot be read as text.
    Unreadable(String),
}

const PLAIN: &[(&str, &str)] = &[
    ("md", "markdown"),
    ("markdown", "markdown"),
    ("txt", "text"),
    ("text", "text"),
    ("json", "json"),
    ("jsonl", "json lines"),
    ("csv", "csv"),
    ("tsv", "tsv"),
    ("yaml", "yaml"),
    ("yml", "yaml"),
    ("toml", "toml"),
    ("html", "html"),
    ("htm", "html"),
    ("xml", "xml"),
    ("tex", "latex"),
    ("log", "log"),
    ("rst", "text"),
    ("py", "code"),
    ("rs", "code"),
    ("js", "code"),
    ("ts", "code"),
    ("sh", "code"),
    ("sql", "code"),
    ("go", "code"),
    ("java", "code"),
    ("c", "code"),
    ("h", "code"),
    ("cpp", "code"),
];

/// Read `bytes`, a file named `name`, as text.
pub fn to_text(name: &str, bytes: &[u8]) -> Text {
    let ext = name
        .rsplit_once('.')
        .map(|(_, e)| e.to_ascii_lowercase())
        .unwrap_or_default();
    let result = match ext.as_str() {
        "docx" => office(bytes, "docx", docx),
        "pptx" => office(bytes, "pptx", pptx),
        "xlsx" => office(bytes, "xlsx", xlsx),
        "pdf" => pdf(bytes).map(|t| Text::Read {
            kind: "pdf",
            text: t,
        }),
        _ => {
            let kind = PLAIN
                .iter()
                .find(|(e, _)| *e == ext)
                .map(|(_, k)| *k)
                .unwrap_or("text");
            if bytes.contains(&0) || std::str::from_utf8(bytes).is_err() && kind == "text" {
                let what = if ext.is_empty() {
                    "a binary file".to_string()
                } else {
                    format!("a binary .{ext} file")
                };
                return Text::Unreadable(format!(
                    "{what}; this judge reads text, markdown, json, csv, docx, xlsx, pptx, and pdf"
                ));
            }
            Ok(Text::Read {
                kind,
                text: String::from_utf8_lossy(bytes).into_owned(),
            })
        }
    };
    result.unwrap_or_else(Text::Unreadable)
}

fn office(
    bytes: &[u8],
    kind: &'static str,
    read: impl Fn(&Zip<'_>) -> Result<String, String>,
) -> Result<Text, String> {
    let zip = Zip::open(bytes).map_err(|e| format!("not a readable .{kind} file ({e})"))?;
    let text = read(&zip)?;
    if text.trim().is_empty() {
        return Err(format!("a .{kind} file with no text in it"));
    }
    Ok(Text::Read { kind, text })
}

fn docx(z: &Zip<'_>) -> Result<String, String> {
    let xml = z.text("word/document.xml")?;
    Ok(xml_text(
        &xml,
        &[("</w:p>", "\n"), ("<w:tab/>", "\t"), ("<w:br/>", "\n")],
    ))
}

fn pptx(z: &Zip<'_>) -> Result<String, String> {
    let mut slides: Vec<(u32, &str)> = z
        .names()
        .filter_map(|n| {
            let num = n
                .strip_prefix("ppt/slides/slide")?
                .strip_suffix(".xml")?
                .parse()
                .ok()?;
            Some((num, n))
        })
        .collect();
    slides.sort();
    let mut out = String::new();
    for (num, name) in slides {
        out.push_str(&format!("--- slide {num} ---\n"));
        out.push_str(&xml_text(&z.text(name)?, &[("</a:p>", "\n")]));
        out.push('\n');
    }
    Ok(out)
}

fn xlsx(z: &Zip<'_>) -> Result<String, String> {
    let shared: Vec<String> = z
        .text("xl/sharedStrings.xml")
        .map(|x| {
            elements(&x, "si")
                .iter()
                .map(|si| xml_text(si, &[]))
                .collect()
        })
        .unwrap_or_default();
    let workbook = z.text("xl/workbook.xml").unwrap_or_default();
    let names: Vec<String> = workbook
        .split("<sheet ")
        .skip(1)
        .map(|s| attr(s, "name").unwrap_or_default())
        .collect();
    let mut sheets: Vec<(u32, &str)> = z
        .names()
        .filter_map(|n| {
            let num = n
                .strip_prefix("xl/worksheets/sheet")?
                .strip_suffix(".xml")?
                .parse()
                .ok()?;
            Some((num, n))
        })
        .collect();
    sheets.sort();
    let mut out = String::new();
    for (num, name) in sheets {
        let title = names
            .get(num as usize - 1)
            .cloned()
            .unwrap_or_else(|| format!("sheet {num}"));
        out.push_str(&format!("=== sheet: {} ===\n", unescape(&title)));
        let xml = z.text(name)?;
        for row in elements(&xml, "row") {
            let cells: Vec<String> = row
                .split("<c ")
                .skip(1)
                .map(|c| {
                    let r = attr(c, "r").unwrap_or_default();
                    let shared_ref = attr(c, "t").as_deref() == Some("s");
                    let value = if let Some(is) = c.find("<is>") {
                        xml_text(&c[is..], &[])
                    } else {
                        between(c, "<v>", "</v>").map(unescape).unwrap_or_default()
                    };
                    let value = if shared_ref {
                        value
                            .trim()
                            .parse::<usize>()
                            .ok()
                            .and_then(|i| shared.get(i).cloned())
                            .unwrap_or(value)
                    } else {
                        value
                    };
                    format!("{r}={value}")
                })
                .filter(|c| !c.ends_with('='))
                .collect();
            if !cells.is_empty() {
                out.push_str(&cells.join("\t"));
                out.push('\n');
            }
        }
    }
    Ok(out)
}

/// The text of an XML fragment: `breaks` turned into their text, tags
/// dropped, entities decoded.
fn xml_text(xml: &str, breaks: &[(&str, &str)]) -> String {
    let mut s = xml.to_string();
    for (from, to) in breaks {
        s = s.replace(from, &format!("<bf-break>{to}"));
    }
    let mut out = String::with_capacity(s.len() / 2);
    let mut in_tag = false;
    for ch in s.chars() {
        match ch {
            '<' => in_tag = true,
            '>' if in_tag => in_tag = false,
            c if !in_tag => out.push(c),
            _ => {}
        }
    }
    unescape(&out)
}

fn unescape(s: &str) -> String {
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&#10;", "\n")
        .replace("&amp;", "&")
}

/// The bodies of `<tag ...>...</tag>` elements, in order.
fn elements<'a>(xml: &'a str, tag: &str) -> Vec<&'a str> {
    let open = format!("<{tag}");
    let close = format!("</{tag}>");
    let mut out = Vec::new();
    let mut rest = xml;
    while let Some(i) = rest.find(&open) {
        let after = &rest[i + open.len()..];
        // `<rowBreaks>` is not `<row>`.
        if !after.starts_with([' ', '>', '/']) {
            rest = after;
            continue;
        }
        if after.starts_with('/') || after.find('>').is_some_and(|g| after[..g].ends_with('/')) {
            rest = after;
            continue;
        }
        match after.find(&close) {
            Some(j) => {
                let body = after.find('>').map_or(0, |g| g + 1).min(j);
                out.push(&after[body..j]);
                rest = &after[j + close.len()..];
            }
            None => break,
        }
    }
    out
}

fn attr(tag: &str, name: &str) -> Option<String> {
    let end = tag.find('>').unwrap_or(tag.len());
    let head = &tag[..end];
    let key = format!("{name}=\"");
    let mut search = head;
    loop {
        let i = search.find(&key)?;
        // ` r="A1"`, not `tr="..."`.
        if i == 0 || search.as_bytes()[i - 1] == b' ' {
            let v = &search[i + key.len()..];
            return Some(v[..v.find('"')?].to_string());
        }
        search = &search[i + key.len()..];
    }
}

fn between<'a>(s: &'a str, open: &str, close: &str) -> Option<&'a str> {
    let i = s.find(open)? + open.len();
    let j = s[i..].find(close)?;
    Some(&s[i..i + j])
}

/// A zip archive in memory: just enough to read Office files.
struct Zip<'a> {
    bytes: &'a [u8],
    entries: Vec<ZipEntry>,
}

struct ZipEntry {
    name: String,
    method: u16,
    compressed: u64,
    offset: u64,
}

fn u16_at(b: &[u8], i: usize) -> Option<u16> {
    Some(u16::from_le_bytes(b.get(i..i + 2)?.try_into().ok()?))
}

fn u32_at(b: &[u8], i: usize) -> Option<u32> {
    Some(u32::from_le_bytes(b.get(i..i + 4)?.try_into().ok()?))
}

impl<'a> Zip<'a> {
    fn open(bytes: &'a [u8]) -> Result<Self, String> {
        let bad = || "no zip directory".to_string();
        let floor = bytes.len().saturating_sub(22 + 65_535);
        let eocd = (floor..bytes.len().saturating_sub(21))
            .rev()
            .find(|&i| u32_at(bytes, i) == Some(0x0605_4b50))
            .ok_or_else(bad)?;
        let count = u16_at(bytes, eocd + 10).ok_or_else(bad)? as usize;
        let mut at = u32_at(bytes, eocd + 16).ok_or_else(bad)? as usize;
        let mut entries = Vec::with_capacity(count);
        for _ in 0..count {
            if u32_at(bytes, at) != Some(0x0201_4b50) {
                return Err(bad());
            }
            let method = u16_at(bytes, at + 10).ok_or_else(bad)?;
            let compressed = u64::from(u32_at(bytes, at + 20).ok_or_else(bad)?);
            let name_len = u16_at(bytes, at + 28).ok_or_else(bad)? as usize;
            let extra_len = u16_at(bytes, at + 30).ok_or_else(bad)? as usize;
            let comment_len = u16_at(bytes, at + 32).ok_or_else(bad)? as usize;
            let offset = u64::from(u32_at(bytes, at + 42).ok_or_else(bad)?);
            let name = bytes.get(at + 46..at + 46 + name_len).ok_or_else(bad)?;
            entries.push(ZipEntry {
                name: String::from_utf8_lossy(name).into_owned(),
                method,
                compressed,
                offset,
            });
            at += 46 + name_len + extra_len + comment_len;
        }
        Ok(Self { bytes, entries })
    }

    fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|e| e.name.as_str())
    }

    fn text(&self, name: &str) -> Result<String, String> {
        let e = self
            .entries
            .iter()
            .find(|e| e.name == name)
            .ok_or_else(|| format!("{name} is missing"))?;
        let bad = || format!("{name} is damaged");
        let at = e.offset as usize;
        if u32_at(self.bytes, at) != Some(0x0403_4b50) {
            return Err(bad());
        }
        let name_len = u16_at(self.bytes, at + 26).ok_or_else(bad)? as usize;
        let extra_len = u16_at(self.bytes, at + 28).ok_or_else(bad)? as usize;
        let start = at + 30 + name_len + extra_len;
        let data = self
            .bytes
            .get(start..start + e.compressed as usize)
            .ok_or_else(bad)?;
        let mut out = Vec::new();
        match e.method {
            0 => out.extend_from_slice(data),
            8 => {
                flate2::read::DeflateDecoder::new(data)
                    .take(UNPACK_CAP)
                    .read_to_end(&mut out)
                    .map_err(|_| bad())?;
            }
            m => return Err(format!("{name} uses zip method {m}")),
        }
        Ok(String::from_utf8_lossy(&out).into_owned())
    }
}

/// The text a PDF's content streams draw, in stream order. Layout is lost;
/// a scanned PDF has none.
fn pdf(bytes: &[u8]) -> Result<String, String> {
    if !bytes.starts_with(b"%PDF") {
        return Err("not a PDF".into());
    }
    let mut out = String::new();
    let mut rest = bytes;
    while let Some(i) = find(rest, b"stream") {
        let dict_start = rest[..i].windows(2).rposition(|w| w == b"<<").unwrap_or(0);
        let dict = String::from_utf8_lossy(&rest[dict_start..i]).into_owned();
        let mut body = i + b"stream".len();
        if rest.get(body) == Some(&b'\r') {
            body += 1;
        }
        if rest.get(body) == Some(&b'\n') {
            body += 1;
        }
        let Some(len) = find(&rest[body..], b"endstream") else {
            break;
        };
        let raw = &rest[body..body + len];
        rest = &rest[body + len + b"endstream".len()..];
        if dict.contains("/Subtype/Image")
            || dict.contains("/Subtype /Image")
            || dict.contains("/FontFile")
            || dict.contains("/Length1")
        {
            continue;
        }
        let data = if dict.contains("/FlateDecode") {
            let mut v = Vec::new();
            if flate2::read::ZlibDecoder::new(raw)
                .take(UNPACK_CAP)
                .read_to_end(&mut v)
                .is_err()
                && v.is_empty()
            {
                continue;
            }
            v
        } else if dict.contains("/Filter") {
            continue;
        } else {
            raw.to_vec()
        };
        pdf_ops(&data, &mut out);
    }
    let text = out.trim().to_string();
    if text.is_empty() {
        return Err(
            "a PDF with no text this judge can read (scanned, or its fonts use custom encodings)"
                .into(),
        );
    }
    Ok(text)
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

/// Collect the strings of text operators (`Tj`, `TJ`, `'`, `"`), breaking
/// lines on `T*`, `Td`, `TD`, and `ET`.
fn pdf_ops(data: &[u8], out: &mut String) {
    let mut i = 0;
    let mut line = String::new();
    let mut in_text = false;
    while i < data.len() {
        match data[i] {
            b'(' => {
                let (s, next) = pdf_string(data, i);
                if in_text {
                    line.push_str(&s);
                }
                i = next;
                continue;
            }
            b'B' if data.get(i..i + 2) == Some(b"BT") => in_text = true,
            b'E' if data.get(i..i + 2) == Some(b"ET") => {
                in_text = false;
                flush(&mut line, out);
            }
            b'T' if matches!(data.get(i + 1), Some(b'*' | b'd' | b'D')) => flush(&mut line, out),
            _ => {}
        }
        i += 1;
    }
    flush(&mut line, out);
}

fn flush(line: &mut String, out: &mut String) {
    let t = line.trim();
    if !t.is_empty() {
        out.push_str(t);
        out.push('\n');
    }
    line.clear();
}

/// A literal string starting at `data[start] == '('`, and where it ends.
fn pdf_string(data: &[u8], start: usize) -> (String, usize) {
    let mut depth = 0;
    let mut out = Vec::new();
    let mut i = start;
    while i < data.len() {
        let c = data[i];
        match c {
            b'\\' => {
                i += 1;
                match data.get(i) {
                    Some(b'n') => out.push(b'\n'),
                    Some(b'r' | b't') => out.push(b' '),
                    Some(d @ b'0'..=b'7') => {
                        let mut v = u32::from(d - b'0');
                        let mut k = 1;
                        while k < 3 && matches!(data.get(i + 1), Some(b'0'..=b'7')) {
                            i += 1;
                            v = v * 8 + u32::from(data[i] - b'0');
                            k += 1;
                        }
                        out.push((v & 0xff) as u8);
                    }
                    Some(&other) => out.push(other),
                    None => {}
                }
            }
            b'(' => {
                if depth > 0 {
                    out.push(c);
                }
                depth += 1;
            }
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return (latin1(&out), i + 1);
                }
                out.push(c);
            }
            _ => out.push(c),
        }
        i += 1;
    }
    (latin1(&out), i)
}

fn latin1(b: &[u8]) -> String {
    b.iter().map(|&c| c as char).collect()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::io::Write;

    /// A zip archive of `files`, deflated, as Office writes them.
    pub(crate) fn zip(files: &[(&str, &str)]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut central = Vec::new();
        for (name, body) in files {
            let mut enc =
                flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
            enc.write_all(body.as_bytes()).unwrap();
            let data = enc.finish().unwrap();
            let offset = out.len() as u32;
            out.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
            out.extend_from_slice(&[20, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
            out.extend_from_slice(&(data.len() as u32).to_le_bytes());
            out.extend_from_slice(&(body.len() as u32).to_le_bytes());
            out.extend_from_slice(&(name.len() as u16).to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes());
            out.extend_from_slice(name.as_bytes());
            out.extend_from_slice(&data);
            central.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
            central.extend_from_slice(&[20, 0, 20, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
            central.extend_from_slice(&(data.len() as u32).to_le_bytes());
            central.extend_from_slice(&(body.len() as u32).to_le_bytes());
            central.extend_from_slice(&(name.len() as u16).to_le_bytes());
            central.extend_from_slice(&[0; 12]);
            central.extend_from_slice(&offset.to_le_bytes());
            central.extend_from_slice(name.as_bytes());
        }
        let cd_offset = out.len() as u32;
        out.extend_from_slice(&central);
        out.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
        out.extend_from_slice(&[0, 0, 0, 0]);
        out.extend_from_slice(&(files.len() as u16).to_le_bytes());
        out.extend_from_slice(&(files.len() as u16).to_le_bytes());
        out.extend_from_slice(&(central.len() as u32).to_le_bytes());
        out.extend_from_slice(&cd_offset.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out
    }

    fn read(name: &str, bytes: &[u8]) -> String {
        match to_text(name, bytes) {
            Text::Read { text, .. } => text,
            Text::Unreadable(why) => panic!("{name}: {why}"),
        }
    }

    #[test]
    fn a_docx_reads_as_paragraphs() {
        let doc = zip(&[(
            "word/document.xml",
            "<w:document><w:body><w:p><w:r><w:t>Executive Summary</w:t></w:r></w:p>\
             <w:p><w:r><w:t>Fees &amp; costs</w:t></w:r><w:r><w:tab/><w:t>High</w:t></w:r></w:p></w:body></w:document>",
        )]);
        assert_eq!(
            read("memo.docx", &doc),
            "Executive Summary\nFees & costs\tHigh\n"
        );
    }

    #[test]
    fn an_xlsx_reads_as_sheets_of_cells() {
        let book = zip(&[
            (
                "xl/workbook.xml",
                r#"<workbook><sheets><sheet name="Budget" sheetId="1"/></sheets></workbook>"#,
            ),
            (
                "xl/sharedStrings.xml",
                "<sst><si><t>Revenue</t></si><si><t>Cost</t></si></sst>",
            ),
            (
                "xl/worksheets/sheet1.xml",
                r#"<worksheet><sheetData><row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1"><v>4700</v></c></row><row r="2"><c r="A2" t="s"><v>1</v></c><c r="B2" t="inlineStr"><is><t>n/a</t></is></c></row></sheetData></worksheet>"#,
            ),
        ]);
        assert_eq!(
            read("budget.xlsx", &book),
            "=== sheet: Budget ===\nA1=Revenue\tB1=4700\nA2=Cost\tB2=n/a\n"
        );
    }

    #[test]
    fn a_pdf_reads_its_drawn_text() {
        let content =
            b"BT /F1 12 Tf 72 700 Td (Total: 4.7 billion) Tj T* [(Sev) -20 (erity: High)] TJ ET";
        let mut enc = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        enc.write_all(content).unwrap();
        let z = enc.finish().unwrap();
        let mut pdf = b"%PDF-1.4\n1 0 obj << /Length 5 /Filter /FlateDecode >>\nstream\n".to_vec();
        pdf.extend_from_slice(&z);
        pdf.extend_from_slice(b"\nendstream\nendobj\n%%EOF");
        assert_eq!(read("r.pdf", &pdf), "Total: 4.7 billion\nSeverity: High");
    }

    #[test]
    fn text_reads_as_written_and_binary_is_noted() {
        assert_eq!(read("a.md", b"# Hi\n"), "# Hi\n");
        assert!(
            matches!(to_text("a.png", b"\x89PNG\0\0"), Text::Unreadable(w) if w.contains("binary .png"))
        );
        assert!(
            matches!(to_text("a.docx", b"not a zip"), Text::Unreadable(w) if w.contains("not a readable .docx"))
        );
    }
}
