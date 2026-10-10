//! Event timestamps as the 0.8 viewer reads them (`_parse_ts`).
//!
//! A timestamp is either epoch seconds (a JSON number) or ISO 8601 text read
//! by Python's `datetime.fromisoformat`. The text reader below follows
//! CPython 3.12's C implementation (`Modules/_datetimemodule.c`), quirks
//! included, because the payload carries the resulting numbers and the
//! golden tests compare them exactly. A time without a UTC offset is local
//! time, as `datetime.timestamp()` reads it.

use chrono::{Datelike, NaiveDate, Weekday};

use crate::pyjson::Value;
use crate::values::finite_float;

/// Seconds since the Unix epoch for a trajectory timestamp, or `None` when
/// it is absent or unreadable.
pub fn parse_ts(value: &Value) -> Option<f64> {
    parse_ts_with(value, &local_offset)
}

/// [`parse_ts`] with the local UTC offset supplied by the caller: `offset(t)`
/// is the local offset in seconds at Unix time `t`.
pub(crate) fn parse_ts_with(value: &Value, offset: &dyn Fn(i64) -> Option<i64>) -> Option<f64> {
    match value {
        Value::Str(text) => {
            let parsed = fromisoformat(text)?;
            let seconds = parsed.timestamp(offset)?;
            seconds.is_finite().then_some(seconds)
        }
        other => finite_float(other),
    }
}

/// The local UTC offset at a Unix time, from the system time zone.
fn local_offset(unix: i64) -> Option<i64> {
    use chrono::{Local, Offset, TimeZone};
    let moment = Local.timestamp_opt(unix, 0).single()?;
    Some(i64::from(moment.offset().fix().local_minus_utc()))
}

/// A datetime as `fromisoformat` builds it.
#[derive(Debug, PartialEq)]
struct Parsed {
    date: NaiveDate,
    hour: i64,
    minute: i64,
    second: i64,
    microsecond: i64,
    /// UTC offset in microseconds; `None` for a naive (local) time.
    offset_us: Option<i64>,
}

/// The byte at `index`, or 0 past the end: C reads up to the string's NUL.
fn at(bytes: &[u8], index: usize) -> u8 {
    bytes.get(index).copied().unwrap_or(0)
}

/// Read `count` ASCII digits starting at `*pos`.
fn digits(bytes: &[u8], pos: &mut usize, count: usize) -> Option<i64> {
    let mut value = 0;
    for _ in 0..count {
        let digit = at(bytes, *pos).wrapping_sub(b'0');
        *pos += 1;
        if digit > 9 {
            return None;
        }
        value = value * 10 + i64::from(digit);
    }
    Some(value)
}

/// `datetime.fromisoformat(text)`; `None` where Python raises `ValueError`.
fn fromisoformat(text: &str) -> Option<Parsed> {
    if text.chars().count() < 7 {
        return None;
    }
    let bytes = text.as_bytes();
    let separator = find_separator(bytes)?;
    let date = parse_date(bytes, separator)?;
    let mut time = Time::default();
    if bytes.len() > separator {
        // The separator may be any one character.
        let lead = bytes[separator];
        let width = if lead & 0x80 == 0 {
            1
        } else {
            match lead & 0xf0 {
                0xe0 => 3,
                0xf0 => 4,
                _ => 2,
            }
        };
        time = parse_time(&bytes[(separator + width).min(bytes.len())..])?;
    }
    if !(1..=9999).contains(&date.year())
        || !(0..24).contains(&time.hour)
        || !(0..60).contains(&time.minute)
        || !(0..60).contains(&time.second)
    {
        return None;
    }
    let offset_us = match time.offset {
        None => None,
        // A zero offset in seconds is UTC whatever its microseconds say.
        Some((0, _)) => Some(0),
        Some((seconds, micros)) => {
            let total = seconds * 1_000_000 + micros;
            // `timezone()` takes offsets strictly within a day.
            if total.abs() >= 86_400 * 1_000_000 {
                return None;
            }
            Some(total)
        }
    };
    Some(Parsed {
        date,
        hour: time.hour,
        minute: time.minute,
        second: time.second,
        microsecond: time.microsecond,
        offset_us,
    })
}

/// Where the date part ends (`_find_isoformat_datetime_separator`).
fn find_separator(bytes: &[u8]) -> Option<usize> {
    let len = bytes.len();
    if len == 7 {
        return Some(7);
    }
    if at(bytes, 4) == b'-' {
        if at(bytes, 5) == b'W' {
            if len < 8 {
                return None;
            }
            if len > 8 && at(bytes, 8) == b'-' {
                if len == 9 {
                    return None;
                }
                if len > 10 && at(bytes, 10).is_ascii_digit() {
                    return Some(8);
                }
                return Some(10);
            }
            return Some(8);
        }
        return Some(10);
    }
    if at(bytes, 4) == b'W' {
        let mut index = 7;
        while index < len && at(bytes, index).is_ascii_digit() {
            index += 1;
        }
        if index < 9 {
            return Some(index);
        }
        return Some(if index % 2 == 0 { 7 } else { 8 });
    }
    Some(8)
}

/// `parse_isoformat_date`: `YYYY-MM-DD`, `YYYYMMDD`, or an ISO week date.
fn parse_date(bytes: &[u8], len: usize) -> Option<NaiveDate> {
    let mut pos = 0;
    let year = digits(bytes, &mut pos, 4)?;
    let dashed = at(bytes, pos) == b'-';
    if dashed {
        pos += 1;
    }
    if at(bytes, pos) == b'W' {
        pos += 1;
        let week = digits(bytes, &mut pos, 2)?;
        let day = if pos < len {
            if dashed {
                let separator = at(bytes, pos);
                pos += 1;
                if separator != b'-' {
                    return None;
                }
            }
            digits(bytes, &mut pos, 1)?
        } else {
            1
        };
        return iso_week_date(year, week, day);
    }
    let month = digits(bytes, &mut pos, 2)?;
    if dashed {
        let separator = at(bytes, pos);
        pos += 1;
        if separator != b'-' {
            return None;
        }
    }
    let day = digits(bytes, &mut pos, 2)?;
    NaiveDate::from_ymd_opt(
        i32::try_from(year).ok()?,
        u32::try_from(month).ok()?,
        u32::try_from(day).ok()?,
    )
}

/// `iso_to_ymd`: the calendar date of ISO year, week, and weekday.
fn iso_week_date(year: i64, week: i64, day: i64) -> Option<NaiveDate> {
    if !(1..=9999).contains(&year) || !(1..=7).contains(&day) {
        return None;
    }
    let weekday = Weekday::try_from(u8::try_from(day - 1).ok()?).ok()?;
    NaiveDate::from_isoywd_opt(
        i32::try_from(year).ok()?,
        u32::try_from(week).ok()?,
        weekday,
    )
}

#[derive(Debug, Default)]
struct Time {
    hour: i64,
    minute: i64,
    second: i64,
    microsecond: i64,
    /// UTC offset as (seconds, microseconds), both carrying the sign.
    offset: Option<(i64, i64)>,
}

/// `parse_isoformat_time`: `HH[:MM[:SS[.ffffff]]]` plus an optional `Z` or
/// `±HH[:MM[:SS[.ffffff]]]` offset.
fn parse_time(bytes: &[u8]) -> Option<Time> {
    let len = bytes.len();
    // The offset starts at the first `Z`, `+` or `-` (a do-while in C, so an
    // empty time looks one byte past its end).
    let mut tz = 0;
    loop {
        if matches!(at(bytes, tz), b'Z' | b'+' | b'-') {
            break;
        }
        tz += 1;
        if tz >= len {
            break;
        }
    }
    let (clock, more) = parse_clock(bytes, 0, tz)?;
    let mut time = Time {
        hour: clock[0],
        minute: clock[1],
        second: clock[2],
        microsecond: clock[3],
        offset: None,
    };
    if tz == len {
        return (!more).then_some(time);
    }
    if at(bytes, tz) == b'Z' {
        if at(bytes, tz + 1) != 0 {
            return None;
        }
        time.offset = Some((0, 0));
        return Some(time);
    }
    let sign = if at(bytes, tz) == b'-' { -1 } else { 1 };
    let (offset, more) = parse_clock(bytes, tz + 1, len)?;
    if more {
        return None;
    }
    time.offset = Some((
        sign * (offset[0] * 3600 + offset[1] * 60 + offset[2]),
        sign * offset[3],
    ));
    Some(time)
}

/// `parse_hh_mm_ss_ff` over `bytes[start..end]`: hours, minutes, seconds,
/// microseconds, and whether something other than the end of the text
/// follows (C's return value 1).
fn parse_clock(bytes: &[u8], start: usize, end: usize) -> Option<([i64; 4], bool)> {
    let mut values = [0i64; 4];
    let mut pos = start;
    let mut has_separator = true;
    let mut index = 0;
    while index < 3 {
        values[index] = digits(bytes, &mut pos, 2)?;
        let next = at(bytes, pos);
        pos += 1;
        if index == 0 {
            has_separator = next == b':';
        }
        index += 1;
        if pos >= end {
            return Some((values, next != 0));
        } else if has_separator && next == b':' {
            continue;
        } else if next == b'.' || next == b',' {
            break;
        } else if !has_separator {
            pos -= 1;
        } else {
            return None;
        }
    }
    // Fraction: up to six digits read, any further digits skipped.
    let remaining = end - pos;
    let count = remaining.min(6);
    let mut fraction = digits(bytes, &mut pos, count)?;
    for _ in count..6 {
        fraction *= 10;
    }
    values[3] = fraction;
    while at(bytes, pos).is_ascii_digit() {
        pos += 1;
    }
    Some((values, at(bytes, pos) != 0))
}

impl Parsed {
    /// `datetime.timestamp()`.
    fn timestamp(&self, offset: &dyn Fn(i64) -> Option<i64>) -> Option<f64> {
        let days = i64::from(self.date.num_days_from_ce()) - UNIX_EPOCH_DAYS_FROM_CE;
        let seconds = days * 86_400 + self.hour * 3600 + self.minute * 60 + self.second;
        match self.offset_us {
            Some(offset_us) => {
                // `(dt - epoch).total_seconds()`: exact microseconds divided
                // by 10**6, correctly rounded (parsing the decimal does that).
                let micros = i128::from(seconds) * 1_000_000 + i128::from(self.microsecond)
                    - i128::from(offset_us);
                let sign = if micros < 0 { "-" } else { "" };
                let magnitude = micros.unsigned_abs();
                format!(
                    "{sign}{}.{:06}",
                    magnitude / 1_000_000,
                    magnitude % 1_000_000
                )
                .parse()
                .ok()
            }
            None => {
                let unix = local_to_unix(seconds, offset)?;
                Some(unix as f64 + self.microsecond as f64 / 1e6)
            }
        }
    }
}

/// Days from 0001-01-01 (day 1) to 1970-01-01.
const UNIX_EPOCH_DAYS_FROM_CE: i64 = 719_163;

/// The Unix time of a local wall-clock time, CPython's `local_to_seconds`
/// with `fold=0`: an ambiguous time takes the earlier instant and a time
/// in a gap is read with the offset in effect before the gap.
fn local_to_unix(wall: i64, offset: &dyn Fn(i64) -> Option<i64>) -> Option<i64> {
    const MAX_FOLD_SECONDS: i64 = 24 * 3600;
    let local = |unix: i64| -> Option<i64> { Some(unix + offset(unix)?) };
    let a = local(wall)? - wall;
    let u1 = wall - a;
    let t1 = local(u1)?;
    let b = if t1 == wall {
        let u2 = u1 - MAX_FOLD_SECONDS;
        let b = local(u2)? - u2;
        if a == b {
            return Some(u1);
        }
        b
    } else {
        t1 - u1
    };
    let u2 = wall - b;
    let t2 = local(u2)?;
    if t2 == wall {
        return Some(u2);
    }
    if t1 == wall {
        return Some(u1);
    }
    Some(u1.max(u2))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// America/Los_Angeles in 2026: UTC-7 from 2026-03-08 10:00 UTC until
    /// 2026-11-01 09:00 UTC, UTC-8 otherwise.
    pub(crate) fn los_angeles_2026(unix: i64) -> Option<i64> {
        const DST_START: i64 = 1_772_964_000;
        const DST_END: i64 = 1_793_523_600;
        Some(if (DST_START..DST_END).contains(&unix) {
            -7 * 3600
        } else {
            -8 * 3600
        })
    }

    fn ts(text: &str) -> Option<f64> {
        parse_ts_with(&Value::Str(text.to_string()), &|_| Some(0))
    }

    #[test]
    fn reads_common_iso_forms() {
        assert_eq!(ts("2026-08-15T09:41:03+00:00"), Some(1_786_786_863.0));
        assert_eq!(ts("2026-08-15T09:41:03Z"), Some(1_786_786_863.0));
        assert_eq!(ts("2026-08-15T11:41:03+02:00"), Some(1_786_786_863.0));
        assert_eq!(
            ts("2026-08-15T09:41:07.500000+00:00"),
            Some(1_786_786_867.5)
        );
        assert_eq!(ts("20260815T094103Z"), Some(1_786_786_863.0));
        assert_eq!(ts("2026-08-15 09:41:03,25Z"), Some(1_786_786_863.25));
    }

    #[test]
    fn rejects_what_python_rejects() {
        for text in [
            "not-a-time",
            "2026-13-01T00:00:00Z",
            "2026-08-15T25:00Z",
            "2026-08-15T09:41:03+24:00",
            "2026-08-15T",
            "2026-08-15T09:41:03Z junk",
            "2026-02-29",
        ] {
            assert_eq!(ts(text), None, "{text}");
        }
    }

    #[test]
    fn numbers_are_epoch_seconds() {
        assert_eq!(parse_ts_with(&Value::Float(1.5), &|_| Some(0)), Some(1.5));
        assert_eq!(parse_ts_with(&Value::Bool(true), &|_| Some(0)), None);
        assert_eq!(parse_ts_with(&Value::Float(f64::NAN), &|_| Some(0)), None);
    }

    #[test]
    fn naive_times_are_local_with_python_fold_and_gap_rules() {
        let la = |text: &str| parse_ts_with(&Value::Str(text.to_string()), &los_angeles_2026);
        // Summer: UTC-7.
        assert_eq!(la("2026-07-01T12:00:00"), Some(1_782_932_400.0));
        // 01:30 on 2026-11-01 happens twice; fold=0 takes the first (UTC-7).
        assert_eq!(la("2026-11-01T01:30:00"), Some(1_793_521_800.0));
        // 02:30 on 2026-03-08 does not exist; Python reads it as UTC-8.
        assert_eq!(la("2026-03-08T02:30:00"), Some(1_772_965_800.0));
    }
}
