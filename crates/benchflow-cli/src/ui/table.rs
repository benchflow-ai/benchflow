//! Plain aligned tables: no borders, two spaces between columns, a dim
//! header. Cells may hold color; widths ignore it.

use super::{dim, visible_width};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Align {
    Left,
    Right,
}

pub struct Table {
    header: Vec<String>,
    align: Vec<Align>,
    rows: Vec<Vec<String>>,
}

impl Table {
    pub fn new(header: &[&str]) -> Self {
        Self {
            header: header.iter().map(|s| s.to_string()).collect(),
            align: vec![Align::Left; header.len()],
            rows: Vec::new(),
        }
    }

    pub fn align(mut self, col: usize, a: Align) -> Self {
        if col < self.align.len() {
            self.align[col] = a;
        }
        self
    }

    pub fn row(&mut self, cells: Vec<String>) {
        self.rows.push(cells);
    }

    pub fn render(&self, indent: usize) -> String {
        let n = self.header.len();
        let mut widths = vec![0usize; n];
        for (i, h) in self.header.iter().enumerate() {
            widths[i] = visible_width(h);
        }
        for r in &self.rows {
            for (i, c) in r.iter().enumerate().take(n) {
                widths[i] = widths[i].max(visible_width(c));
            }
        }
        let pad = " ".repeat(indent);
        let mut out = String::new();
        let line = |cells: &[String], header: bool| {
            let mut s = pad.clone();
            for (i, c) in cells.iter().enumerate().take(n) {
                let w = visible_width(c);
                let fill = " ".repeat(widths[i].saturating_sub(w));
                let cell = if header { dim(c) } else { c.clone() };
                let last = i + 1 == n;
                match self.align[i] {
                    Align::Left if last => s.push_str(&cell),
                    Align::Left => {
                        s.push_str(&cell);
                        s.push_str(&fill);
                    }
                    Align::Right => {
                        s.push_str(&fill);
                        s.push_str(&cell);
                    }
                }
                if !last {
                    s.push_str("  ");
                }
            }
            s.trim_end().to_string()
        };
        out.push_str(&line(&self.header, true));
        out.push('\n');
        for r in &self.rows {
            out.push_str(&line(r, false));
            out.push('\n');
        }
        out
    }
}
