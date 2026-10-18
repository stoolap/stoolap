// Copyright 2025 Stoolap Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! INTERVAL reading and TIMESTAMP arithmetic: the one place an interval is
//! read and a timestamp is moved by it. A month moves the calendar and
//! clamps the day; a result outside years 0000 to 9999 is an error.

use chrono::{DateTime, Datelike, NaiveDate, TimeDelta, Utc};

use super::integer::integer_from_text;
use super::{Error, Result};

/// A unit an INTERVAL counts in
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntervalUnit {
    Microsecond,
    Millisecond,
    Second,
    Minute,
    Hour,
    Day,
    Week,
    Month,
    Year,
}

impl IntervalUnit {
    /// The unit a name spells, singular, plural or abbreviated, in any case
    pub fn parse(name: &str) -> Option<Self> {
        use IntervalUnit::*;
        const NAMES: &[(&str, IntervalUnit)] = &[
            ("year", Year),
            ("years", Year),
            ("month", Month),
            ("months", Month),
            ("week", Week),
            ("weeks", Week),
            ("day", Day),
            ("days", Day),
            ("hour", Hour),
            ("hours", Hour),
            ("minute", Minute),
            ("minutes", Minute),
            ("min", Minute),
            ("mins", Minute),
            ("second", Second),
            ("seconds", Second),
            ("sec", Second),
            ("secs", Second),
            ("millisecond", Millisecond),
            ("milliseconds", Millisecond),
            ("ms", Millisecond),
            ("microsecond", Microsecond),
            ("microseconds", Microsecond),
            ("us", Microsecond),
        ];
        NAMES
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(name))
            .map(|(_, unit)| *unit)
    }

    /// Months in one unit when it is a calendar unit, else None
    pub fn months(self) -> Option<i64> {
        match self {
            IntervalUnit::Month => Some(1),
            IntervalUnit::Year => Some(12),
            _ => None,
        }
    }

    /// Microseconds in one unit; zero for the calendar units
    pub fn micros(self) -> i64 {
        match self {
            IntervalUnit::Microsecond => 1,
            IntervalUnit::Millisecond => 1_000,
            IntervalUnit::Second => 1_000_000,
            IntervalUnit::Minute => 60_000_000,
            IntervalUnit::Hour => 3_600_000_000,
            IntervalUnit::Day => 86_400_000_000,
            IntervalUnit::Week => 604_800_000_000,
            IntervalUnit::Month | IntervalUnit::Year => 0,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            IntervalUnit::Microsecond => "microsecond",
            IntervalUnit::Millisecond => "millisecond",
            IntervalUnit::Second => "second",
            IntervalUnit::Minute => "minute",
            IntervalUnit::Hour => "hour",
            IntervalUnit::Day => "day",
            IntervalUnit::Week => "week",
            IntervalUnit::Month => "month",
            IntervalUnit::Year => "year",
        }
    }
}

/// An interval: a fixed duration, or a count of calendar months
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interval {
    Duration(TimeDelta),
    Months(i64),
}

impl Interval {
    /// `quantity` of `unit`; an error when it does not fit
    pub fn of(quantity: i64, unit: IntervalUnit) -> Result<Self> {
        let interval = match unit.months() {
            Some(per) => quantity.checked_mul(per).map(Interval::Months),
            None => quantity
                .checked_mul(unit.micros())
                .map(|micros| Interval::Duration(TimeDelta::microseconds(micros))),
        };
        interval.ok_or_else(|| {
            Error::Type(format!(
                "INTERVAL '{quantity} {}' is out of range",
                unit.name()
            ))
        })
    }

    /// `k` of this interval; None when it does not fit
    pub fn times(self, k: i64) -> Option<Self> {
        match self {
            Interval::Months(m) => m.checked_mul(k).map(Interval::Months),
            Interval::Duration(d) => d
                .num_microseconds()
                .and_then(|micros| micros.checked_mul(k))
                .map(|micros| Interval::Duration(TimeDelta::microseconds(micros))),
        }
    }

    /// Whether moving by it goes forward, backward or nowhere
    pub fn signum(self) -> i64 {
        match self {
            Interval::Months(m) => m.signum(),
            Interval::Duration(d) => d.cmp(&TimeDelta::zero()) as i64,
        }
    }
}

/// Reads interval text: `N unit`, a bare `N` counting days, or the
/// `[-]N days HH:MM:SS` and `[-]HH:MM:SS` forms a TIMESTAMP difference
/// prints. None for text that is not an interval; an error for a number
/// outside the range.
pub fn parse_interval(s: &str) -> Result<Option<Interval>> {
    let s = s.trim();
    if let Some(interval) = parse_formatted_duration(s)? {
        return Ok(Some(interval));
    }
    let mut parts = s.split_whitespace();
    let (quantity, unit) = match (parts.next(), parts.next(), parts.next()) {
        (Some(quantity), None, None) => (quantity, IntervalUnit::Day),
        (Some(quantity), Some(unit), None) => match IntervalUnit::parse(unit) {
            Some(unit) => (quantity, unit),
            None => return Ok(None),
        },
        _ => return Ok(None),
    };
    match integer_from_text(quantity)? {
        Some(quantity) => Interval::of(quantity, unit).map(Some),
        None => Ok(None),
    }
}

/// Parse the "[-]N days HH:MM:SS" and "[-]HH:MM:SS" strings a TIMESTAMP
/// difference prints; an error when it does not fit
fn parse_formatted_duration(s: &str) -> Result<Option<Interval>> {
    let (negative, rest) = match s.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, s),
    };
    let (days, hms) = match rest.split_once(' ') {
        Some((days, tail)) => match tail.trim_start().split_once(' ') {
            Some((unit, hms))
                if unit.eq_ignore_ascii_case("days") || unit.eq_ignore_ascii_case("day") =>
            {
                (Some(days), hms.trim_start())
            }
            _ => return Ok(None),
        },
        None => (None, rest),
    };
    let mut fields = hms.split(':');
    let (Some(h), Some(m), Some(sec), None) =
        (fields.next(), fields.next(), fields.next(), fields.next())
    else {
        return Ok(None);
    };
    let days = match days {
        Some(days) => integer_from_text(days)?,
        None => Some(0),
    };
    let (Some(days), Some(hours), Some(minutes), Some(seconds)) = (
        days,
        integer_from_text(h)?,
        integer_from_text(m)?,
        integer_from_text(sec)?,
    ) else {
        return Ok(None);
    };
    if days < 0 || hours < 0 || !(0..60).contains(&minutes) || !(0..60).contains(&seconds) {
        return Ok(None);
    }
    let micros = days
        .checked_mul(IntervalUnit::Day.micros())
        .zip(hours.checked_mul(IntervalUnit::Hour.micros()))
        .and_then(|(d, h)| d.checked_add(h))
        .and_then(|total| total.checked_add(minutes * IntervalUnit::Minute.micros()))
        .and_then(|total| total.checked_add(seconds * IntervalUnit::Second.micros()))
        .ok_or_else(|| Error::Type(format!("INTERVAL '{s}' is out of range")))?;
    let micros = if negative { -micros } else { micros };
    Ok(Some(Interval::Duration(TimeDelta::microseconds(micros))))
}

/// `t` moved by `interval`, backwards when `subtract`; an error when the
/// result lies outside the TIMESTAMP range
pub fn shift(t: DateTime<Utc>, interval: Interval, subtract: bool) -> Result<DateTime<Utc>> {
    let moved = match interval {
        Interval::Duration(d) if subtract => t.checked_sub_signed(d),
        Interval::Duration(d) => t.checked_add_signed(d),
        Interval::Months(m) => {
            if subtract { m.checked_neg() } else { Some(m) }.and_then(|m| add_months(t, m))
        }
    };
    moved
        .filter(|moved| (0..=9999).contains(&moved.year()))
        .ok_or_else(|| Error::Type(format!("{t} moved by an INTERVAL is out of range")))
}

/// `t` moved by `months` calendar months, the day clamped to the month's
/// last and the time of day kept
fn add_months(t: DateTime<Utc>, months: i64) -> Option<DateTime<Utc>> {
    let total = (i64::from(t.year()) * 12 + i64::from(t.month0())).checked_add(months)?;
    let year = i32::try_from(total.div_euclid(12)).ok()?;
    let month = total.rem_euclid(12) as u32 + 1;
    let first = NaiveDate::from_ymd_opt(year, month, 1)?;
    let last = first
        .checked_add_months(chrono::Months::new(1))?
        .pred_opt()?;
    let date = first.with_day(t.day().min(last.day()))?;
    Some(date.and_time(t.time()).and_utc())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ts(s: &str) -> DateTime<Utc> {
        crate::core::parse_timestamp(s).unwrap()
    }

    #[test]
    fn test_interval_text_reads_one_way() {
        let day = Interval::Duration(TimeDelta::days(1));
        assert_eq!(parse_interval("1 day").unwrap(), Some(day));
        assert_eq!(parse_interval(" 1 DAYS ").unwrap(), Some(day));
        assert_eq!(parse_interval("1").unwrap(), Some(day));
        assert_eq!(
            parse_interval("5 ms").unwrap(),
            Some(Interval::Duration(TimeDelta::milliseconds(5)))
        );
        assert_eq!(
            parse_interval("2 years").unwrap(),
            Some(Interval::Months(24))
        );
        assert_eq!(
            parse_interval("-1 day").unwrap(),
            Some(Interval::Duration(-TimeDelta::days(1)))
        );
        assert_eq!(
            parse_interval("30 minutes").unwrap(),
            Some(Interval::Duration(TimeDelta::minutes(30)))
        );
        assert_eq!(
            parse_interval("-1 days 02:00:00").unwrap(),
            Some(Interval::Duration(-TimeDelta::hours(26)))
        );
        assert_eq!(parse_interval("banana").unwrap(), None);
        assert_eq!(parse_interval("1 banana").unwrap(), None);
        assert_eq!(parse_interval("1.5 days").unwrap(), None);
        assert_eq!(parse_interval("1 day 2 hours").unwrap(), None);
        assert!(parse_interval("9223372036854775807 days").is_err());
        assert!(parse_interval("99999999999999999999 days").is_err());
        assert!(parse_interval("9223372036854775807 years").is_err());
        assert!(parse_interval("106751992 days 00:00:00").is_err());
        assert!(parse_interval("9223372036854775807 days 00:00:00").is_err());
        assert!(parse_interval("99999999999999999999 days 00:00:00").is_err());
        assert!(parse_interval("99999999999999999999:00:00").is_err());
        assert_eq!(parse_interval("banana days 00:00:00").unwrap(), None);
        assert_eq!(
            parse_interval("106751991 days 00:00:00")
                .unwrap()
                .and_then(|step| step.times(0)),
            Some(Interval::Duration(TimeDelta::zero()))
        );
    }

    #[test]
    fn test_a_shift_outside_the_range_is_an_error() {
        let end = ts("9999-12-31 12:00:00.5");
        let day = Interval::of(1, IntervalUnit::Day).unwrap();
        assert!(shift(end, day, false).is_err());
        assert_eq!(shift(end, day, true).unwrap(), ts("9999-12-30 12:00:00.5"));
        assert!(shift(end, Interval::Months(1), false).is_err());
        assert!(shift(ts("0000-01-01"), day, true).is_err());
        assert!(shift(end, Interval::Months(i64::MIN), true).is_err());
        assert_eq!(
            shift(ts("2024-01-31 10:00:00.75"), Interval::Months(1), false).unwrap(),
            ts("2024-02-29 10:00:00.75")
        );
        assert_eq!(
            shift(ts("2024-03-31"), Interval::Months(1), true).unwrap(),
            ts("2024-02-29")
        );
    }
}
