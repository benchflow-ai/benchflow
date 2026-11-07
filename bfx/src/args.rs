//! Reading a command's JSON arguments. Unknown keys are refused, so a
//! misspelt option never silently means its default.

use crate::err::{Error, Result};
use serde_json::{Map, Value};

pub struct Args<'a> {
    m: &'a Map<String, Value>,
}

impl<'a> Args<'a> {
    pub fn new(v: &'a Value, allowed: &[&str]) -> Result<Self> {
        let m = v.as_object().ok_or_else(|| Error::bad("arguments must be a JSON object"))?;
        for k in m.keys() {
            if !allowed.contains(&k.as_str()) {
                return Err(Error::bad(format!("unknown argument {k:?}")));
            }
        }
        Ok(Args { m })
    }

    pub fn opt(&self, k: &str) -> Option<&'a Value> {
        match self.m.get(k) {
            None | Some(Value::Null) => None,
            Some(v) => Some(v),
        }
    }

    pub fn str(&self, k: &str) -> Result<&'a str> {
        self.opt_str(k)?.ok_or_else(|| Error::bad(format!("{k} is required")))
    }

    pub fn opt_str(&self, k: &str) -> Result<Option<&'a str>> {
        match self.opt(k) {
            None => Ok(None),
            Some(Value::String(s)) => Ok(Some(s.as_str())),
            Some(_) => Err(Error::bad(format!("{k} must be a string"))),
        }
    }

    /// A path given as text under *k* or as raw bytes, base64, under
    /// `<k>_b64` (a name need not be UTF-8); never both.
    pub fn opt_path(&self, k: &str) -> Result<Option<Vec<u8>>> {
        let b64 = format!("{k}_b64");
        match (self.opt_str(k)?, self.opt_str(&b64)?) {
            (None, None) => Ok(None),
            (Some(s), None) => Ok(Some(s.as_bytes().to_vec())),
            (None, Some(b)) => crate::b64::decode(b).map(Some).ok_or_else(|| Error::bad(format!("{b64} is not base64"))),
            (Some(_), Some(_)) => Err(Error::bad(format!("give one of {k} and {b64}"))),
        }
    }

    pub fn path(&self, k: &str) -> Result<Vec<u8>> {
        self.opt_path(k)?.ok_or_else(|| Error::bad(format!("{k} is required")))
    }

    /// A list of paths under *k* (text) and `<k>_b64` (base64 bytes), in
    /// that order.
    pub fn path_list(&self, k: &str) -> Result<Vec<Vec<u8>>> {
        let mut out: Vec<Vec<u8>> = self.str_list(k)?.into_iter().map(|s| s.as_bytes().to_vec()).collect();
        for b in self.str_list(&format!("{k}_b64"))? {
            out.push(crate::b64::decode(b).ok_or_else(|| Error::bad(format!("{k}_b64 must hold base64")))?);
        }
        Ok(out)
    }

    pub fn u64(&self, k: &str) -> Result<u64> {
        self.opt_u64(k)?.ok_or_else(|| Error::bad(format!("{k} is required")))
    }

    pub fn opt_u64(&self, k: &str) -> Result<Option<u64>> {
        match self.opt(k) {
            None => Ok(None),
            Some(v) => v
                .as_u64()
                .map(Some)
                .ok_or_else(|| Error::bad(format!("{k} must be a non-negative integer"))),
        }
    }

    pub fn u32(&self, k: &str) -> Result<u32> {
        self.opt_u32(k)?.ok_or_else(|| Error::bad(format!("{k} is required")))
    }

    pub fn opt_u32(&self, k: &str) -> Result<Option<u32>> {
        match self.opt_u64(k)? {
            None => Ok(None),
            Some(n) if n <= u32::MAX as u64 => Ok(Some(n as u32)),
            Some(_) => Err(Error::bad(format!("{k} is out of range"))),
        }
    }

    pub fn bool(&self, k: &str, default: bool) -> Result<bool> {
        match self.opt(k) {
            None => Ok(default),
            Some(Value::Bool(b)) => Ok(*b),
            Some(_) => Err(Error::bad(format!("{k} must be true or false"))),
        }
    }

    pub fn arr(&self, k: &str) -> Result<&'a Vec<Value>> {
        self.opt_arr(k)?.ok_or_else(|| Error::bad(format!("{k} is required")))
    }

    pub fn opt_arr(&self, k: &str) -> Result<Option<&'a Vec<Value>>> {
        match self.opt(k) {
            None => Ok(None),
            Some(Value::Array(a)) => Ok(Some(a)),
            Some(_) => Err(Error::bad(format!("{k} must be a list"))),
        }
    }

    pub fn str_list(&self, k: &str) -> Result<Vec<&'a str>> {
        match self.opt_arr(k)? {
            None => Ok(Vec::new()),
            Some(a) => a
                .iter()
                .map(|v| v.as_str().ok_or_else(|| Error::bad(format!("{k} must hold strings"))))
                .collect(),
        }
    }

    pub fn u32_list(&self, k: &str) -> Result<Vec<u32>> {
        match self.opt_arr(k)? {
            None => Ok(Vec::new()),
            Some(a) => a
                .iter()
                .map(|v| {
                    v.as_u64()
                        .filter(|n| *n <= u32::MAX as u64)
                        .map(|n| n as u32)
                        .ok_or_else(|| Error::bad(format!("{k} must hold integers")))
                })
                .collect(),
        }
    }
}

/// A mode given as an integer (``493``) or an octal string (``"0755"``).
pub fn mode_value(v: &Value, k: &str) -> Result<u32> {
    let n = match v {
        Value::Number(n) => n.as_u64(),
        Value::String(s) => u64::from_str_radix(s.trim_start_matches("0o"), 8).ok(),
        _ => None,
    };
    match n {
        Some(n) if n <= 0o7777 => Ok(n as u32),
        _ => Err(Error::bad(format!("{k} must be a mode in 0..=0o7777"))),
    }
}
