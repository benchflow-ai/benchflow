//! A small seeded generator for the property tests (xorshift64*): every
//! case is reproducible from its seed, and no attack string is written by
//! hand; names, trees and archives are drawn from it.

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    pub fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    pub fn below(&mut self, n: u64) -> u64 {
        if n == 0 {
            0
        } else {
            self.next() % n
        }
    }

    pub fn chance(&mut self, num: u64, den: u64) -> bool {
        self.below(den) < num
    }

    pub fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[self.below(xs.len() as u64) as usize]
    }

    /// A file name: mostly ordinary, sometimes made only of dots (never
    /// exactly "." or ".."), sometimes with spaces or non-ASCII bytes.
    pub fn name(&mut self) -> Vec<u8> {
        loop {
            let len = 1 + self.below(12) as usize;
            let alphabet: &[u8] = match self.below(6) {
                0 => b".",
                1 => b".a",
                2 => b"ab .-_",
                _ => b"abcdefghijklmnopqrstuvwxyz0123456789_-.",
            };
            let mut n: Vec<u8> = (0..len).map(|_| *self.pick(alphabet)).collect();
            if self.chance(1, 20) {
                n.push(0xc3);
                n.push(0xa9);
            }
            if n != b"." && n != b".." {
                return n;
            }
        }
    }
}
