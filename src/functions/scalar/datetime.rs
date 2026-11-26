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

//! Date/Time scalar functions

use chrono::{DateTime, Datelike, Duration, TimeZone, Timelike, Utc};

use crate::common::SmartString;
use crate::core::interval::{self, Interval, IntervalUnit};
use crate::core::{parse_timestamp, Error, Result, Value};
use crate::functions::{
    FunctionDataType, FunctionInfo, FunctionSignature, FunctionType, ScalarFunction,
};
use crate::validate_arg_count;

// ============================================================================
// DATE_TRUNC
// ============================================================================

/// DATE_TRUNC function - truncates a timestamp to the specified precision
///
/// # Arguments
/// - unit: String specifying the truncation unit (year, month, day, hour, minute, second)
/// - timestamp: The timestamp to truncate
///
/// # Examples
/// ```sql
/// DATE_TRUNC('year', '2024-03-15 10:30:45') -- Returns '2024-01-01 00:00:00'
/// DATE_TRUNC('month', '2024-03-15 10:30:45') -- Returns '2024-03-01 00:00:00'
/// DATE_TRUNC('day', '2024-03-15 10:30:45') -- Returns '2024-03-15 00:00:00'
/// DATE_TRUNC('hour', '2024-03-15 10:30:45') -- Returns '2024-03-15 10:00:00'
/// ```
#[derive(Default)]
pub struct DateTruncFunction;

impl ScalarFunction for DateTruncFunction {
    fn name(&self) -> &str {
        "DATE_TRUNC"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "DATE_TRUNC",
            FunctionType::Scalar,
            "Truncates a timestamp to the specified precision (year, month, day, hour, minute, second)",
            FunctionSignature::new(
                FunctionDataType::Timestamp,
                vec![FunctionDataType::String, FunctionDataType::Any],
                2,
                2,
            ),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "DATE_TRUNC", 2);

        // First argument: unit (string)
        let unit = match &args[0] {
            Value::Text(s) => s.as_str(),
            _ if args[0].is_null() => return Ok(Value::null_unknown()),
            _ => {
                return Err(Error::invalid_argument(
                    "DATE_TRUNC first argument must be a string",
                ))
            }
        };

        // Second argument: timestamp
        if args[1].is_null() {
            return Ok(Value::null_unknown());
        }

        let ts = match &args[1] {
            Value::Timestamp(t) => *t,
            Value::Text(s) => {
                // Try to parse as timestamp
                parse_timestamp(s).map_err(|_| {
                    Error::invalid_argument(format!("DATE_TRUNC could not parse timestamp: {}", s))
                })?
            }
            _ => {
                return Err(Error::invalid_argument(
                    "DATE_TRUNC second argument must be a timestamp or string",
                ))
            }
        };

        TimeBucket::date_unit(unit)?
            .truncate(ts)
            .map(Value::Timestamp)
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(DateTruncFunction)
    }
}

/// What DATE_TRUNC and TIME_TRUNC truncate a timestamp to: a calendar unit,
/// or an interval in nanoseconds. Every reader of either call truncates by it
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimeBucket {
    Calendar(CalendarUnit),
    Every(i64),
}

/// DATE_TRUNC's units
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CalendarUnit {
    Year,
    Quarter,
    Month,
    Week,
    Day,
    Hour,
    Minute,
    Second,
}

impl TimeBucket {
    /// DATE_TRUNC's `unit`, of any case
    pub fn date_unit(unit: &str) -> Result<Self> {
        const UNITS: [(&str, CalendarUnit); 8] = [
            ("year", CalendarUnit::Year),
            ("month", CalendarUnit::Month),
            ("day", CalendarUnit::Day),
            ("hour", CalendarUnit::Hour),
            ("minute", CalendarUnit::Minute),
            ("second", CalendarUnit::Second),
            ("week", CalendarUnit::Week),
            ("quarter", CalendarUnit::Quarter),
        ];
        UNITS
            .iter()
            .find(|(name, _)| unit.eq_ignore_ascii_case(name))
            .map(|&(_, unit)| Self::Calendar(unit))
            .ok_or_else(|| {
                Error::invalid_argument(format!(
                    "DATE_TRUNC invalid unit: {}. Valid units are: year, quarter, month, week, day, hour, minute, second",
                    unit
                ))
            })
    }

    /// TIME_TRUNC's `duration`
    pub fn interval(duration: &str) -> Result<Self> {
        TimeTruncFunction::cached_parse_duration(duration).map(Self::Every)
    }

    /// The bucket of a call named `function` with `argument` as its first
    /// argument; None for another function or an argument it rejects
    pub fn of_call(function: &str, argument: &str) -> Option<Self> {
        if function.eq_ignore_ascii_case("DATE_TRUNC") {
            Self::date_unit(argument).ok()
        } else if function.eq_ignore_ascii_case("TIME_TRUNC") {
            Self::interval(argument).ok()
        } else {
            None
        }
    }

    /// `ts` truncated to the bucket; an interval reads it in nanoseconds,
    /// which a timestamp outside their range has none of
    pub fn truncate(self, ts: DateTime<Utc>) -> Result<DateTime<Utc>> {
        match self {
            Self::Calendar(unit) => Ok(unit.truncate(ts)),
            Self::Every(_) => {
                crate::core::value::timestamp_nanos(&ts).map(|at| self.truncate_nanos(at))
            }
        }
    }

    /// `truncate` for a timestamp of `at` nanoseconds. A later timestamp
    /// never truncates to an earlier one
    pub fn truncate_nanos(self, at: i64) -> DateTime<Utc> {
        match self {
            Self::Calendar(unit) => unit.truncate(DateTime::from_timestamp_nanos(at)),
            // Rounds down before 1970 too; a start before i64::MIN is subtracted as a time
            Self::Every(nanos) => {
                let past = at.rem_euclid(nanos);
                match at.checked_sub(past) {
                    Some(start) => DateTime::from_timestamp_nanos(start),
                    None => DateTime::from_timestamp_nanos(at) - Duration::nanoseconds(past),
                }
            }
        }
    }

    /// The first and last nanosecond that truncate as `at` does, found by
    /// truncating alone: as truncation never decreases, they bound one run
    pub fn span_nanos(self, at: i64) -> (i64, i64) {
        let start = self.truncate_nanos(at);
        let same = |t: i128| self.truncate_nanos(t as i64) == start;
        let edge = |toward: i128| {
            let limit = match toward > 0 {
                true => i64::MAX as i128,
                false => i64::MIN as i128,
            };
            // Gallop out of the bucket, then halve back to its edge
            let (mut inside, mut step) = (at as i128, 1i128);
            let mut outside = loop {
                let probe = inside + toward * step;
                if (probe - limit) * toward >= 0 {
                    if same(limit) {
                        return limit as i64;
                    }
                    break limit;
                }
                if !same(probe) {
                    break probe;
                }
                inside = probe;
                step *= 2;
            };
            while (outside - inside).abs() > 1 {
                let mid = (inside + outside) / 2;
                match same(mid) {
                    true => inside = mid,
                    false => outside = mid,
                }
            }
            inside as i64
        };
        (edge(-1), edge(1))
    }
}

impl CalendarUnit {
    fn truncate(self, ts: DateTime<Utc>) -> DateTime<Utc> {
        let day = |year, month, day| {
            Utc.with_ymd_and_hms(year, month, day, 0, 0, 0)
                .single()
                .unwrap_or(ts)
        };
        match self {
            Self::Year => day(ts.year(), 1, 1),
            Self::Quarter => day(ts.year(), ((ts.month() - 1) / 3) * 3 + 1, 1),
            Self::Month => day(ts.year(), ts.month(), 1),
            Self::Week => {
                let start = ts - Duration::days(ts.weekday().num_days_from_monday() as i64);
                day(start.year(), start.month(), start.day())
            }
            Self::Day => day(ts.year(), ts.month(), ts.day()),
            Self::Hour => Utc
                .with_ymd_and_hms(ts.year(), ts.month(), ts.day(), ts.hour(), 0, 0)
                .single()
                .unwrap_or(ts),
            Self::Minute => Utc
                .with_ymd_and_hms(ts.year(), ts.month(), ts.day(), ts.hour(), ts.minute(), 0)
                .single()
                .unwrap_or(ts),
            Self::Second => Utc
                .with_ymd_and_hms(
                    ts.year(),
                    ts.month(),
                    ts.day(),
                    ts.hour(),
                    ts.minute(),
                    ts.second(),
                )
                .single()
                .unwrap_or(ts),
        }
    }
}

// ============================================================================
// TIME_TRUNC
// ============================================================================

/// TIME_TRUNC function - truncates a timestamp to a duration interval
///
/// # Arguments
/// - duration: String specifying the duration (e.g., "15m", "1h", "30s")
/// - timestamp: The timestamp to truncate
///
/// # Examples
/// ```sql
/// TIME_TRUNC('15m', '2024-03-15 10:37:45') -- Returns '2024-03-15 10:30:00'
/// TIME_TRUNC('1h', '2024-03-15 10:37:45') -- Returns '2024-03-15 10:00:00'
/// TIME_TRUNC('30s', '2024-03-15 10:37:45') -- Returns '2024-03-15 10:37:30'
/// ```
#[derive(Default)]
pub struct TimeTruncFunction;

impl TimeTruncFunction {
    /// Parse duration with thread-local cache. The duration string (e.g. "15m") is
    /// typically a constant literal repeated for every row. Caching avoids redundant
    /// char-by-char parsing on each call.
    fn cached_parse_duration(duration_str: &str) -> Result<i64> {
        use std::cell::RefCell;
        thread_local! {
            static CACHE: RefCell<(String, Option<i64>)> = const { RefCell::new((String::new(), None)) };
        }
        CACHE.with(|cell| {
            let cached = cell.borrow();
            if cached.0 == duration_str
                && let Some(nanos) = cached.1
            {
                return Ok(nanos);
            }
            drop(cached);
            let nanos = Self::parse_duration(duration_str)?;
            let mut cached = cell.borrow_mut();
            cached.0.clear();
            cached.0.push_str(duration_str);
            cached.1 = Some(nanos);
            Ok(nanos)
        })
    }

    /// Parse a duration string like "15m", "1h", "30s"
    fn parse_duration(duration_str: &str) -> Result<i64> {
        // Handle common duration formats
        let s = duration_str.trim();
        if s.is_empty() {
            return Err(Error::invalid_argument("Empty duration string"));
        }

        // Try parsing as standard duration format (e.g., "1h30m", "15m", "30s")
        let mut total_nanos: i64 = 0;
        let chars: Vec<char> = s.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            // Find the numeric part
            let num_start = i;
            while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                i += 1;
            }

            if num_start == i {
                return Err(Error::invalid_argument(format!(
                    "Invalid duration format: {}",
                    duration_str
                )));
            }

            let num_str: String = chars[num_start..i].iter().collect();
            let num: f64 = num_str.parse().map_err(|_| {
                Error::invalid_argument(format!("Invalid number in duration: {}", num_str))
            })?;

            // Find the unit
            let unit_start = i;
            while i < chars.len() && chars[i].is_alphabetic() {
                i += 1;
            }

            let unit: String = chars[unit_start..i].iter().collect();
            let multiplier: i64 = match unit.as_str() {
                "ns" => 1,
                "us" | "µs" => 1_000,
                "ms" => 1_000_000,
                "s" => 1_000_000_000,
                "m" => 60 * 1_000_000_000,
                "h" => 60 * 60 * 1_000_000_000,
                "" => {
                    return Err(Error::invalid_argument(format!(
                        "Missing unit in duration: {}",
                        duration_str
                    )))
                }
                _ => {
                    return Err(Error::invalid_argument(format!(
                        "Unknown duration unit: {}",
                        unit
                    )))
                }
            };

            total_nanos += (num * multiplier as f64) as i64;
        }

        if total_nanos == 0 {
            return Err(Error::invalid_argument(format!(
                "Invalid duration: {}",
                duration_str
            )));
        }

        Ok(total_nanos)
    }
}

impl ScalarFunction for TimeTruncFunction {
    fn name(&self) -> &str {
        "TIME_TRUNC"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "TIME_TRUNC",
            FunctionType::Scalar,
            "Truncates a timestamp to the specified duration interval (e.g., '15m', '1h', '30s')",
            FunctionSignature::new(
                FunctionDataType::Timestamp,
                vec![FunctionDataType::String, FunctionDataType::Any],
                2,
                2,
            ),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "TIME_TRUNC", 2);

        // First argument: duration string
        let duration_str = match &args[0] {
            Value::Text(s) => s.clone(),
            _ if args[0].is_null() => return Ok(Value::null_unknown()),
            _ => {
                return Err(Error::invalid_argument(
                    "TIME_TRUNC first argument must be a string",
                ))
            }
        };

        // Second argument: timestamp
        if args[1].is_null() {
            return Ok(Value::null_unknown());
        }

        let ts = match &args[1] {
            Value::Timestamp(t) => *t,
            Value::Text(s) => {
                // Try to parse as timestamp
                parse_timestamp(s).map_err(|_| {
                    Error::invalid_argument(format!("TIME_TRUNC could not parse timestamp: {}", s))
                })?
            }
            _ => {
                return Err(Error::invalid_argument(
                    "TIME_TRUNC second argument must be a timestamp or string",
                ))
            }
        };

        TimeBucket::interval(&duration_str)?
            .truncate(ts)
            .map(Value::Timestamp)
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(TimeTruncFunction)
    }
}

// ============================================================================
// VERSION
// ============================================================================

/// VERSION function - returns the Stoolap version string
///
/// This function is commonly used by database tools and ORMs to check compatibility.
#[derive(Default)]
pub struct VersionFunction;

impl ScalarFunction for VersionFunction {
    fn name(&self) -> &str {
        "VERSION"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "VERSION",
            FunctionType::Scalar,
            "Returns the Stoolap database version string",
            FunctionSignature::new(FunctionDataType::String, vec![], 0, 0),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        if !args.is_empty() {
            return Err(Error::invalid_argument(format!(
                "VERSION takes no arguments, got {}",
                args.len()
            )));
        }

        // Use the version info from common module
        Ok(Value::Text(SmartString::from_string(
            crate::common::version::version_info(),
        )))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(VersionFunction)
    }
}

// ============================================================================
// EXTRACT
// ============================================================================

/// EXTRACT function - extracts a date/time field from a timestamp
///
/// # Arguments
/// - field: String specifying the field to extract (year, month, day, hour, minute, second, etc.)
/// - timestamp: The timestamp to extract from
///
/// # Examples
/// ```sql
/// EXTRACT('year', '2024-03-15 10:30:45') -- Returns 2024
/// EXTRACT('month', '2024-03-15 10:30:45') -- Returns 3
/// EXTRACT('day', '2024-03-15 10:30:45') -- Returns 15
/// ```
#[derive(Default)]
pub struct ExtractFunction;

impl ScalarFunction for ExtractFunction {
    fn name(&self) -> &str {
        "EXTRACT"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "EXTRACT",
            FunctionType::Scalar,
            "Extracts a date/time field from a timestamp (year, month, day, hour, minute, second, dow, doy, week, quarter)",
            FunctionSignature::new(
                FunctionDataType::Integer,
                vec![FunctionDataType::String, FunctionDataType::Any],
                2,
                2,
            ),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "EXTRACT", 2);

        // First argument: field (string)
        let field = match &args[0] {
            Value::Text(s) => s.as_str(),
            _ if args[0].is_null() => return Ok(Value::null_unknown()),
            _ => {
                return Err(Error::invalid_argument(
                    "EXTRACT first argument must be a string",
                ))
            }
        };

        // Second argument: timestamp
        if args[1].is_null() {
            return Ok(Value::null_unknown());
        }

        let ts = match &args[1] {
            Value::Timestamp(t) => *t,
            Value::Text(s) => parse_timestamp(s).map_err(|_| {
                Error::invalid_argument(format!("EXTRACT could not parse timestamp: {}", s))
            })?,
            _ => {
                return Err(Error::invalid_argument(
                    "EXTRACT second argument must be a timestamp or string",
                ))
            }
        };

        // Extract based on field (case-insensitive without allocation)
        let result = if field.eq_ignore_ascii_case("year") {
            ts.year() as i64
        } else if field.eq_ignore_ascii_case("month") {
            ts.month() as i64
        } else if field.eq_ignore_ascii_case("day") {
            ts.day() as i64
        } else if field.eq_ignore_ascii_case("hour") {
            ts.hour() as i64
        } else if field.eq_ignore_ascii_case("minute") {
            ts.minute() as i64
        } else if field.eq_ignore_ascii_case("second") {
            ts.second() as i64
        } else if field.eq_ignore_ascii_case("millisecond")
            || field.eq_ignore_ascii_case("milliseconds")
        {
            (ts.nanosecond() / 1_000_000) as i64
        } else if field.eq_ignore_ascii_case("microsecond")
            || field.eq_ignore_ascii_case("microseconds")
        {
            (ts.nanosecond() / 1_000) as i64
        } else if field.eq_ignore_ascii_case("dow") || field.eq_ignore_ascii_case("dayofweek") {
            ts.weekday().num_days_from_sunday() as i64 // 0-6, Sunday=0
        } else if field.eq_ignore_ascii_case("isodow") {
            ts.weekday().num_days_from_monday() as i64 + 1 // 1-7, Monday=1
        } else if field.eq_ignore_ascii_case("doy") || field.eq_ignore_ascii_case("dayofyear") {
            ts.ordinal() as i64
        } else if field.eq_ignore_ascii_case("week") || field.eq_ignore_ascii_case("isoweek") {
            ts.iso_week().week() as i64
        } else if field.eq_ignore_ascii_case("quarter") {
            ((ts.month() - 1) / 3 + 1) as i64
        } else if field.eq_ignore_ascii_case("epoch") {
            ts.timestamp()
        } else {
            return Err(Error::invalid_argument(format!(
                "EXTRACT invalid field: {}. Valid fields are: year, month, day, hour, minute, second, millisecond, microsecond, dow, isodow, doy, week, quarter, epoch",
                field
            )));
        };

        Ok(Value::Integer(result))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(ExtractFunction)
    }
}

// ============================================================================
// YEAR / MONTH / DAY / HOUR / MINUTE / SECOND
// ============================================================================

/// YEAR function - extracts the year from a timestamp
#[derive(Default)]
pub struct YearFunction;

impl ScalarFunction for YearFunction {
    fn name(&self) -> &str {
        "YEAR"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "YEAR",
            FunctionType::Scalar,
            "Extracts the year from a timestamp",
            FunctionSignature::new(FunctionDataType::Integer, vec![FunctionDataType::Any], 1, 1),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "YEAR", 1);

        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }

        let ts = match &args[0] {
            Value::Timestamp(t) => *t,
            Value::Text(s) => parse_timestamp(s).map_err(|_| {
                Error::invalid_argument(format!("YEAR could not parse timestamp: {}", s))
            })?,
            _ => {
                return Err(Error::invalid_argument(
                    "YEAR argument must be a timestamp or string",
                ))
            }
        };

        Ok(Value::Integer(ts.year() as i64))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(YearFunction)
    }
}

/// MONTH function - extracts the month from a timestamp (1-12)
#[derive(Default)]
pub struct MonthFunction;

impl ScalarFunction for MonthFunction {
    fn name(&self) -> &str {
        "MONTH"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "MONTH",
            FunctionType::Scalar,
            "Extracts the month from a timestamp (1-12)",
            FunctionSignature::new(FunctionDataType::Integer, vec![FunctionDataType::Any], 1, 1),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "MONTH", 1);

        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }

        let ts = match &args[0] {
            Value::Timestamp(t) => *t,
            Value::Text(s) => parse_timestamp(s).map_err(|_| {
                Error::invalid_argument(format!("MONTH could not parse timestamp: {}", s))
            })?,
            _ => {
                return Err(Error::invalid_argument(
                    "MONTH argument must be a timestamp or string",
                ))
            }
        };

        Ok(Value::Integer(ts.month() as i64))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(MonthFunction)
    }
}

/// DAY function - extracts the day from a timestamp (1-31)
#[derive(Default)]
pub struct DayFunction;

impl ScalarFunction for DayFunction {
    fn name(&self) -> &str {
        "DAY"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "DAY",
            FunctionType::Scalar,
            "Extracts the day from a timestamp (1-31)",
            FunctionSignature::new(FunctionDataType::Integer, vec![FunctionDataType::Any], 1, 1),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "DAY", 1);

        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }

        let ts = match &args[0] {
            Value::Timestamp(t) => *t,
            Value::Text(s) => parse_timestamp(s).map_err(|_| {
                Error::invalid_argument(format!("DAY could not parse timestamp: {}", s))
            })?,
            _ => {
                return Err(Error::invalid_argument(
                    "DAY argument must be a timestamp or string",
                ))
            }
        };

        Ok(Value::Integer(ts.day() as i64))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(DayFunction)
    }
}

/// HOUR function - extracts the hour from a timestamp (0-23)
#[derive(Default)]
pub struct HourFunction;

impl ScalarFunction for HourFunction {
    fn name(&self) -> &str {
        "HOUR"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "HOUR",
            FunctionType::Scalar,
            "Extracts the hour from a timestamp (0-23)",
            FunctionSignature::new(FunctionDataType::Integer, vec![FunctionDataType::Any], 1, 1),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "HOUR", 1);

        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }

        let ts = match &args[0] {
            Value::Timestamp(t) => *t,
            Value::Text(s) => parse_timestamp(s).map_err(|_| {
                Error::invalid_argument(format!("HOUR could not parse timestamp: {}", s))
            })?,
            _ => {
                return Err(Error::invalid_argument(
                    "HOUR argument must be a timestamp or string",
                ))
            }
        };

        Ok(Value::Integer(ts.hour() as i64))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(HourFunction)
    }
}

/// MINUTE function - extracts the minute from a timestamp (0-59)
#[derive(Default)]
pub struct MinuteFunction;

impl ScalarFunction for MinuteFunction {
    fn name(&self) -> &str {
        "MINUTE"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "MINUTE",
            FunctionType::Scalar,
            "Extracts the minute from a timestamp (0-59)",
            FunctionSignature::new(FunctionDataType::Integer, vec![FunctionDataType::Any], 1, 1),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "MINUTE", 1);

        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }

        let ts = match &args[0] {
            Value::Timestamp(t) => *t,
            Value::Text(s) => parse_timestamp(s).map_err(|_| {
                Error::invalid_argument(format!("MINUTE could not parse timestamp: {}", s))
            })?,
            _ => {
                return Err(Error::invalid_argument(
                    "MINUTE argument must be a timestamp or string",
                ))
            }
        };

        Ok(Value::Integer(ts.minute() as i64))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(MinuteFunction)
    }
}

/// SECOND function - extracts the second from a timestamp (0-59)
#[derive(Default)]
pub struct SecondFunction;

impl ScalarFunction for SecondFunction {
    fn name(&self) -> &str {
        "SECOND"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "SECOND",
            FunctionType::Scalar,
            "Extracts the second from a timestamp (0-59)",
            FunctionSignature::new(FunctionDataType::Integer, vec![FunctionDataType::Any], 1, 1),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "SECOND", 1);

        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }

        let ts = match &args[0] {
            Value::Timestamp(t) => *t,
            Value::Text(s) => parse_timestamp(s).map_err(|_| {
                Error::invalid_argument(format!("SECOND could not parse timestamp: {}", s))
            })?,
            _ => {
                return Err(Error::invalid_argument(
                    "SECOND argument must be a timestamp or string",
                ))
            }
        };

        Ok(Value::Integer(ts.second() as i64))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(SecondFunction)
    }
}

// ============================================================================
// DATE_ADD / DATE_SUB
// ============================================================================

/// DATE_ADD function - adds an interval to a timestamp
#[derive(Default)]
pub struct DateAddFunction;

impl ScalarFunction for DateAddFunction {
    fn name(&self) -> &str {
        "DATE_ADD"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "DATE_ADD",
            FunctionType::Scalar,
            "Adds an interval to a timestamp. Usage: DATE_ADD(timestamp, days) or DATE_ADD(timestamp, interval, unit)",
            FunctionSignature::new(
                FunctionDataType::Timestamp,
                vec![
                    FunctionDataType::Any,
                    FunctionDataType::Integer,
                    FunctionDataType::String,
                ],
                2,
                3,
            ),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "DATE_ADD", 2, 3);
        date_shift("DATE_ADD", args, false)
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(DateAddFunction)
    }
}

/// DATE_SUB function - subtracts an interval from a timestamp
#[derive(Default)]
pub struct DateSubFunction;

impl ScalarFunction for DateSubFunction {
    fn name(&self) -> &str {
        "DATE_SUB"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "DATE_SUB",
            FunctionType::Scalar,
            "Subtracts an interval from a timestamp. Usage: DATE_SUB(timestamp, days) or DATE_SUB(timestamp, interval, unit)",
            FunctionSignature::new(
                FunctionDataType::Timestamp,
                vec![
                    FunctionDataType::Any,
                    FunctionDataType::Integer,
                    FunctionDataType::String,
                ],
                2,
                3,
            ),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "DATE_SUB", 2, 3);
        date_shift("DATE_SUB", args, true)
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(DateSubFunction)
    }
}

// ============================================================================
// DATEDIFF
// ============================================================================

/// DATEDIFF function - returns the difference between two dates in days
#[derive(Default)]
pub struct DateDiffFunction;

impl ScalarFunction for DateDiffFunction {
    fn name(&self) -> &str {
        "DATEDIFF"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "DATEDIFF",
            FunctionType::Scalar,
            "Returns the difference between two dates in days",
            FunctionSignature::new(
                FunctionDataType::Integer,
                vec![FunctionDataType::Any, FunctionDataType::Any],
                2,
                2,
            ),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "DATEDIFF", 2);

        if args[0].is_null() || args[1].is_null() {
            return Ok(Value::null_unknown());
        }

        let ts1 = match &args[0] {
            Value::Timestamp(t) => *t,
            Value::Text(s) => parse_timestamp(s).map_err(|_| {
                Error::invalid_argument(format!("DATEDIFF could not parse first timestamp: {}", s))
            })?,
            _ => {
                return Err(Error::invalid_argument(
                    "DATEDIFF arguments must be timestamps or strings",
                ))
            }
        };

        let ts2 = match &args[1] {
            Value::Timestamp(t) => *t,
            Value::Text(s) => parse_timestamp(s).map_err(|_| {
                Error::invalid_argument(format!("DATEDIFF could not parse second timestamp: {}", s))
            })?,
            _ => {
                return Err(Error::invalid_argument(
                    "DATEDIFF arguments must be timestamps or strings",
                ))
            }
        };

        let diff = ts1.signed_duration_since(ts2);
        Ok(Value::Integer(diff.num_days()))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(DateDiffFunction)
    }
}

// ============================================================================
// DATE_DIFF (alias for DATEDIFF)
// ============================================================================

/// DATE_DIFF function - alias for DATEDIFF for consistency with DATE_ADD/DATE_SUB naming
#[derive(Default)]
pub struct DateDiffAliasFunction;

impl ScalarFunction for DateDiffAliasFunction {
    fn name(&self) -> &str {
        "DATE_DIFF"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "DATE_DIFF",
            FunctionType::Scalar,
            "Returns the difference between two dates in days (alias for DATEDIFF)",
            FunctionSignature::new(
                FunctionDataType::Integer,
                vec![FunctionDataType::Any, FunctionDataType::Any],
                2,
                2,
            ),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        // Delegate to DATEDIFF
        DateDiffFunction.evaluate(args)
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(DateDiffAliasFunction)
    }
}

// ============================================================================
// CURRENT_DATE / CURRENT_TIME / CURRENT_TIMESTAMP
// ============================================================================

/// CURRENT_DATE function - returns the current date
#[derive(Default)]
pub struct CurrentDateFunction;

impl ScalarFunction for CurrentDateFunction {
    fn name(&self) -> &str {
        "CURRENT_DATE"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "CURRENT_DATE",
            FunctionType::Scalar,
            "Returns the current date (at midnight UTC)",
            FunctionSignature::new(FunctionDataType::Timestamp, vec![], 0, 0),
        )
        .per_statement()
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        if !args.is_empty() {
            return Err(Error::invalid_argument(format!(
                "CURRENT_DATE takes no arguments, got {}",
                args.len()
            )));
        }

        let now = Utc::now();
        let date = Utc
            .with_ymd_and_hms(now.year(), now.month(), now.day(), 0, 0, 0)
            .single()
            .unwrap_or(now);
        Ok(Value::Timestamp(date))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(CurrentDateFunction)
    }
}

/// CURRENT_TIMESTAMP function - returns the current timestamp (alias for NOW)
#[derive(Default)]
pub struct CurrentTimestampFunction;

impl ScalarFunction for CurrentTimestampFunction {
    fn name(&self) -> &str {
        "CURRENT_TIMESTAMP"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "CURRENT_TIMESTAMP",
            FunctionType::Scalar,
            "Returns the current timestamp",
            FunctionSignature::new(FunctionDataType::Timestamp, vec![], 0, 0),
        )
        .per_statement()
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        if !args.is_empty() {
            return Err(Error::invalid_argument(format!(
                "CURRENT_TIMESTAMP takes no arguments, got {}",
                args.len()
            )));
        }

        Ok(Value::Timestamp(Utc::now()))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(CurrentTimestampFunction)
    }
}

/// CURRENT_TIME function - returns the current time as "HH:MM:SS"
#[derive(Default)]
pub struct CurrentTimeFunction;

impl ScalarFunction for CurrentTimeFunction {
    fn name(&self) -> &str {
        "CURRENT_TIME"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "CURRENT_TIME",
            FunctionType::Scalar,
            "Returns the current time as HH:MM:SS",
            FunctionSignature::new(FunctionDataType::String, vec![], 0, 0),
        )
        .per_statement()
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        if !args.is_empty() {
            return Err(Error::invalid_argument(format!(
                "CURRENT_TIME takes no arguments, got {}",
                args.len()
            )));
        }

        let time_str = Utc::now().format("%H:%M:%S").to_string();
        Ok(Value::Text(SmartString::from_string(time_str)))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(CurrentTimeFunction)
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// DATE_ADD and DATE_SUB: the timestamp moved by an amount of a unit,
/// days when the unit is left out; NULL when an argument is NULL
fn date_shift(name: &str, args: &[Value], subtract: bool) -> Result<Value> {
    if args.iter().any(Value::is_null) {
        return Ok(Value::null_unknown());
    }
    let ts = match &args[0] {
        Value::Timestamp(t) => *t,
        Value::Text(s) => parse_timestamp(s).map_err(|_| {
            Error::invalid_argument(format!("{name} could not parse timestamp: {s}"))
        })?,
        _ => {
            return Err(Error::invalid_argument(format!(
                "{name} first argument must be a timestamp or string"
            )))
        }
    };
    let amount = match &args[1] {
        Value::Integer(i) => *i,
        Value::Float(f) => crate::core::integer::integer_from_float(*f)?,
        _ => {
            return Err(Error::invalid_argument(format!(
                "{name} second argument must be an integer"
            )))
        }
    };
    let unit = match args.get(2) {
        None => IntervalUnit::Day,
        Some(Value::Text(s)) => IntervalUnit::parse(s).ok_or_else(|| {
            Error::invalid_argument(format!(
                "{name} invalid unit: {s}. Valid units are: year(s), month(s), week(s), day(s), hour(s), minute(s), second(s), millisecond(s), microsecond(s)"
            ))
        })?,
        Some(_) => {
            return Err(Error::invalid_argument(format!(
                "{name} third argument must be a string"
            )))
        }
    };
    let interval = Interval::of(amount, unit)?;
    Ok(Value::Timestamp(interval::shift(ts, interval, subtract)?))
}

// ============================================================================
// TO_CHAR
// ============================================================================

/// TO_CHAR function - formats a timestamp or number as a string
///
/// # Arguments
/// - value: The timestamp or number to format
/// - format: The format pattern string
///
/// # Format Patterns (for dates/timestamps)
/// - YYYY: 4-digit year
/// - YY: 2-digit year
/// - MM: Month as 01-12
/// - MON: Abbreviated month name (JAN, FEB, etc.)
/// - MONTH: Full month name
/// - DD: Day of month as 01-31
/// - DY: Abbreviated day name (SUN, MON, etc.)
/// - DAY: Full day name
/// - HH24: Hour as 00-23
/// - HH12 or HH: Hour as 01-12
/// - MI: Minutes as 00-59
/// - SS: Seconds as 00-59
/// - MS: Milliseconds as 000-999
/// - AM/PM: Meridiem indicator
/// - TZ: Timezone abbreviation
///
/// # Examples
/// ```sql
/// TO_CHAR('2024-03-15 14:30:45', 'YYYY-MM-DD') -- Returns '2024-03-15'
/// TO_CHAR('2024-03-15 14:30:45', 'DD MON YYYY') -- Returns '15 MAR 2024'
/// TO_CHAR('2024-03-15 14:30:45', 'HH24:MI:SS') -- Returns '14:30:45'
/// ```
#[derive(Default)]
pub struct ToCharFunction;

impl ScalarFunction for ToCharFunction {
    fn name(&self) -> &str {
        "TO_CHAR"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "TO_CHAR",
            FunctionType::Scalar,
            "Formats a timestamp or number as a string using a format pattern",
            FunctionSignature::new(
                FunctionDataType::String,
                vec![FunctionDataType::Any, FunctionDataType::String],
                2,
                2,
            ),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "TO_CHAR", 2);

        if args[0].is_null() || args[1].is_null() {
            return Ok(Value::null_unknown());
        }

        let format = match &args[1] {
            Value::Text(s) => s.to_string(),
            _ => {
                return Err(Error::invalid_argument(
                    "TO_CHAR format argument must be a string",
                ))
            }
        };

        // Handle timestamp formatting
        let ts = match &args[0] {
            Value::Timestamp(t) => *t,
            Value::Text(s) => parse_timestamp(s).map_err(|_| {
                Error::invalid_argument(format!("TO_CHAR could not parse timestamp: {}", s))
            })?,
            Value::Integer(i) => {
                // Format integer as string with the given format (for number formatting)
                return Ok(Value::text(format_number(*i as f64, &format)));
            }
            Value::Float(f) => {
                // Format float as string with the given format
                return Ok(Value::text(format_number(*f, &format)));
            }
            _ => {
                return Err(Error::invalid_argument(
                    "TO_CHAR first argument must be a timestamp, string, or number",
                ))
            }
        };

        // Format the timestamp according to the pattern
        let result = format_timestamp(ts, &format);
        Ok(Value::text(&result))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(ToCharFunction)
    }
}

/// Format a timestamp according to a format pattern
fn format_timestamp(ts: DateTime<Utc>, format: &str) -> String {
    let mut result = format.to_string();

    // Month and day names
    let month_names = [
        "JANUARY",
        "FEBRUARY",
        "MARCH",
        "APRIL",
        "MAY",
        "JUNE",
        "JULY",
        "AUGUST",
        "SEPTEMBER",
        "OCTOBER",
        "NOVEMBER",
        "DECEMBER",
    ];
    let month_abbr = [
        "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
    ];
    let day_names = [
        "SUNDAY",
        "MONDAY",
        "TUESDAY",
        "WEDNESDAY",
        "THURSDAY",
        "FRIDAY",
        "SATURDAY",
    ];
    let day_abbr = ["SUN", "MON", "TUE", "WED", "THU", "FRI", "SAT"];

    let year = ts.year();
    let month = ts.month() as usize;
    let day = ts.day();
    let hour = ts.hour();
    let minute = ts.minute();
    let second = ts.second();
    let millisecond = ts.nanosecond() / 1_000_000;
    let weekday = ts.weekday().num_days_from_sunday() as usize;

    // Process format patterns (order matters - longer patterns first)
    // Year patterns
    result = result.replace("YYYY", &format!("{:04}", year));
    result = result.replace("YY", &format!("{:02}", year % 100));

    // Month patterns (do MONTH before MON, MM)
    result = result.replace("MONTH", month_names[month - 1]);
    result = result.replace("Month", &capitalize_first(month_names[month - 1]));
    result = result.replace("month", &month_names[month - 1].to_lowercase());
    result = result.replace("MON", month_abbr[month - 1]);
    result = result.replace("Mon", &capitalize_first(month_abbr[month - 1]));
    result = result.replace("mon", &month_abbr[month - 1].to_lowercase());
    result = result.replace("MM", &format!("{:02}", month));

    // Day patterns (do DAY before DY, DD)
    // "day" before "Day", or it rewrites the tail of "Tuesday"
    result = result.replace("DAY", day_names[weekday]);
    result = result.replace("day", &day_names[weekday].to_lowercase());
    result = result.replace("Day", &capitalize_first(day_names[weekday]));
    result = result.replace("DY", day_abbr[weekday]);
    result = result.replace("Dy", &capitalize_first(day_abbr[weekday]));
    result = result.replace("dy", &day_abbr[weekday].to_lowercase());
    result = result.replace("DD", &format!("{:02}", day));

    // Hour patterns
    result = result.replace("HH24", &format!("{:02}", hour));
    let hour12 = if hour == 0 {
        12
    } else if hour > 12 {
        hour - 12
    } else {
        hour
    };
    result = result.replace("HH12", &format!("{:02}", hour12));
    result = result.replace("HH", &format!("{:02}", hour12));

    // Minute and second patterns
    result = result.replace("MI", &format!("{:02}", minute));
    result = result.replace("SS", &format!("{:02}", second));
    result = result.replace("MS", &format!("{:03}", millisecond));

    // AM/PM - use placeholders to avoid sequential replacement interference
    let meridiem = if hour < 12 { "AM" } else { "PM" };
    let meridiem_dotted = if hour < 12 { "A.M." } else { "P.M." };
    // Replace dotted forms first (longer patterns before shorter)
    result = result.replace("A.M.", "\x01MDDOT\x01");
    result = result.replace("P.M.", "\x01MDDOT\x01");
    result = result.replace("a.m.", "\x01mddot\x01");
    result = result.replace("p.m.", "\x01mddot\x01");
    // Then uppercase/lowercase AM/PM
    result = result.replace("AM", "\x01MD\x01");
    result = result.replace("PM", "\x01MD\x01");
    result = result.replace("am", "\x01md\x01");
    result = result.replace("pm", "\x01md\x01");
    // Now replace placeholders with actual values
    result = result.replace("\x01MDDOT\x01", meridiem_dotted);
    result = result.replace("\x01mddot\x01", &meridiem_dotted.to_lowercase());
    result = result.replace("\x01MD\x01", meridiem);
    result = result.replace("\x01md\x01", &meridiem.to_lowercase());

    // Timezone
    result = result.replace("TZ", "UTC");
    result = result.replace("tz", "utc");

    result
}

/// Format a number according to a format pattern
fn format_number(num: f64, format: &str) -> String {
    // Simple number formatting
    // 9 = digit position
    // 0 = digit position with leading zeros
    // . = decimal point
    // , = thousands separator

    let format_upper = format.to_uppercase();

    // Count decimal places in format
    let decimal_places = if let Some(dot_pos) = format_upper.find('.') {
        format_upper[dot_pos + 1..]
            .chars()
            .filter(|c| *c == '9' || *c == '0')
            .count()
    } else {
        0
    };

    // Check if we need thousands separator
    let has_comma = format_upper.contains(',');

    // Format the number
    let formatted = if decimal_places > 0 {
        format!("{:.prec$}", num, prec = decimal_places)
    } else {
        format!("{:.0}", num)
    };

    // Add thousands separator if needed
    if has_comma {
        let parts: Vec<&str> = formatted.split('.').collect();
        let int_part = parts[0];
        let dec_part = parts.get(1);

        let int_with_commas: String = int_part
            .chars()
            .rev()
            .enumerate()
            .map(|(i, c)| {
                if i > 0 && i % 3 == 0 && c != '-' {
                    format!("{},", c) // comma AFTER char, so it appears BEFORE when reversed
                } else {
                    c.to_string()
                }
            })
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();

        if let Some(dec) = dec_part {
            format!("{}.{}", int_with_commas, dec)
        } else {
            int_with_commas
        }
    } else {
        formatted
    }
}

/// Capitalize only the first letter
fn capitalize_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => first
            .to_uppercase()
            .chain(chars.flat_map(|c| c.to_lowercase()))
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn test_date_trunc_year() {
        let f = DateTruncFunction;
        let ts = Utc.with_ymd_and_hms(2024, 3, 15, 10, 30, 45).unwrap();
        let result = f
            .evaluate(&[Value::text("year"), Value::Timestamp(ts)])
            .unwrap();

        if let Value::Timestamp(t) = result {
            assert_eq!(t.year(), 2024);
            assert_eq!(t.month(), 1);
            assert_eq!(t.day(), 1);
            assert_eq!(t.hour(), 0);
            assert_eq!(t.minute(), 0);
            assert_eq!(t.second(), 0);
        } else {
            panic!("Expected timestamp");
        }
    }

    #[test]
    fn test_date_trunc_month() {
        let f = DateTruncFunction;
        let ts = Utc.with_ymd_and_hms(2024, 3, 15, 10, 30, 45).unwrap();
        let result = f
            .evaluate(&[Value::text("month"), Value::Timestamp(ts)])
            .unwrap();

        if let Value::Timestamp(t) = result {
            assert_eq!(t.year(), 2024);
            assert_eq!(t.month(), 3);
            assert_eq!(t.day(), 1);
            assert_eq!(t.hour(), 0);
        } else {
            panic!("Expected timestamp");
        }
    }

    #[test]
    fn test_date_trunc_day() {
        let f = DateTruncFunction;
        let ts = Utc.with_ymd_and_hms(2024, 3, 15, 10, 30, 45).unwrap();
        let result = f
            .evaluate(&[Value::text("day"), Value::Timestamp(ts)])
            .unwrap();

        if let Value::Timestamp(t) = result {
            assert_eq!(t.year(), 2024);
            assert_eq!(t.month(), 3);
            assert_eq!(t.day(), 15);
            assert_eq!(t.hour(), 0);
            assert_eq!(t.minute(), 0);
        } else {
            panic!("Expected timestamp");
        }
    }

    #[test]
    fn test_date_trunc_hour() {
        let f = DateTruncFunction;
        let ts = Utc.with_ymd_and_hms(2024, 3, 15, 10, 30, 45).unwrap();
        let result = f
            .evaluate(&[Value::text("hour"), Value::Timestamp(ts)])
            .unwrap();

        if let Value::Timestamp(t) = result {
            assert_eq!(t.hour(), 10);
            assert_eq!(t.minute(), 0);
            assert_eq!(t.second(), 0);
        } else {
            panic!("Expected timestamp");
        }
    }

    #[test]
    fn test_date_trunc_minute() {
        let f = DateTruncFunction;
        let ts = Utc.with_ymd_and_hms(2024, 3, 15, 10, 30, 45).unwrap();
        let result = f
            .evaluate(&[Value::text("minute"), Value::Timestamp(ts)])
            .unwrap();

        if let Value::Timestamp(t) = result {
            assert_eq!(t.hour(), 10);
            assert_eq!(t.minute(), 30);
            assert_eq!(t.second(), 0);
        } else {
            panic!("Expected timestamp");
        }
    }

    #[test]
    fn test_date_trunc_quarter() {
        let f = DateTruncFunction;

        // Q1
        let ts = Utc.with_ymd_and_hms(2024, 2, 15, 10, 30, 45).unwrap();
        let result = f
            .evaluate(&[Value::text("quarter"), Value::Timestamp(ts)])
            .unwrap();
        if let Value::Timestamp(t) = result {
            assert_eq!(t.month(), 1);
        }

        // Q2
        let ts = Utc.with_ymd_and_hms(2024, 5, 15, 10, 30, 45).unwrap();
        let result = f
            .evaluate(&[Value::text("quarter"), Value::Timestamp(ts)])
            .unwrap();
        if let Value::Timestamp(t) = result {
            assert_eq!(t.month(), 4);
        }

        // Q3
        let ts = Utc.with_ymd_and_hms(2024, 8, 15, 10, 30, 45).unwrap();
        let result = f
            .evaluate(&[Value::text("quarter"), Value::Timestamp(ts)])
            .unwrap();
        if let Value::Timestamp(t) = result {
            assert_eq!(t.month(), 7);
        }

        // Q4
        let ts = Utc.with_ymd_and_hms(2024, 11, 15, 10, 30, 45).unwrap();
        let result = f
            .evaluate(&[Value::text("quarter"), Value::Timestamp(ts)])
            .unwrap();
        if let Value::Timestamp(t) = result {
            assert_eq!(t.month(), 10);
        }
    }

    #[test]
    fn test_date_trunc_null() {
        let f = DateTruncFunction;
        assert!(f
            .evaluate(&[Value::text("day"), Value::null_unknown()])
            .unwrap()
            .is_null());
        assert!(f
            .evaluate(&[Value::null_unknown(), Value::Timestamp(Utc::now())])
            .unwrap()
            .is_null());
    }

    #[test]
    fn test_date_trunc_invalid_unit() {
        let f = DateTruncFunction;
        let ts = Utc::now();
        assert!(f
            .evaluate(&[Value::text("invalid"), Value::Timestamp(ts)])
            .is_err());
    }

    #[test]
    fn test_time_trunc_15m() {
        let f = TimeTruncFunction;
        let ts = Utc.with_ymd_and_hms(2024, 3, 15, 10, 37, 45).unwrap();
        let result = f
            .evaluate(&[Value::text("15m"), Value::Timestamp(ts)])
            .unwrap();

        if let Value::Timestamp(t) = result {
            assert_eq!(t.hour(), 10);
            assert_eq!(t.minute(), 30);
            assert_eq!(t.second(), 0);
        } else {
            panic!("Expected timestamp");
        }
    }

    #[test]
    fn test_time_trunc_1h() {
        let f = TimeTruncFunction;
        let ts = Utc.with_ymd_and_hms(2024, 3, 15, 10, 37, 45).unwrap();
        let result = f
            .evaluate(&[Value::text("1h"), Value::Timestamp(ts)])
            .unwrap();

        if let Value::Timestamp(t) = result {
            assert_eq!(t.hour(), 10);
            assert_eq!(t.minute(), 0);
            assert_eq!(t.second(), 0);
        } else {
            panic!("Expected timestamp");
        }
    }

    #[test]
    fn test_time_trunc_30s() {
        let f = TimeTruncFunction;
        let ts = Utc.with_ymd_and_hms(2024, 3, 15, 10, 37, 45).unwrap();
        let result = f
            .evaluate(&[Value::text("30s"), Value::Timestamp(ts)])
            .unwrap();

        if let Value::Timestamp(t) = result {
            assert_eq!(t.minute(), 37);
            assert_eq!(t.second(), 30);
        } else {
            panic!("Expected timestamp");
        }
    }

    #[test]
    fn test_time_trunc_null() {
        let f = TimeTruncFunction;
        assert!(f
            .evaluate(&[Value::text("1h"), Value::null_unknown()])
            .unwrap()
            .is_null());
    }

    #[test]
    fn test_version() {
        let f = VersionFunction;
        let result = f.evaluate(&[]).unwrap();
        if let Value::Text(s) = result {
            assert!(s.contains("stoolap"));
        } else {
            panic!("Expected string");
        }
    }

    #[test]
    fn test_version_no_args() {
        let f = VersionFunction;
        assert!(f.evaluate(&[Value::Integer(1)]).is_err());
    }

    #[test]
    fn a_later_timestamp_never_truncates_to_an_earlier_bucket() {
        let buckets = [
            "year", "quarter", "month", "week", "day", "hour", "minute", "second",
        ]
        .map(|unit| TimeBucket::date_unit(unit).unwrap())
        .into_iter()
        .chain(["1s", "15m", "1h30m", "7ms"].map(|d| TimeBucket::interval(d).unwrap()));
        // Nanoseconds across 1970 and across a leap day, in steps that are
        // no unit's multiple
        let step = 7_919_000_003_i64;
        let times: Vec<i64> = (-400..400).map(|k| k * step).collect();
        for bucket in buckets {
            let truncated: Vec<_> = times.iter().map(|&at| bucket.truncate_nanos(at)).collect();
            assert!(
                truncated.windows(2).all(|pair| pair[0] <= pair[1]),
                "{bucket:?} decreases"
            );
            for (&at, &start) in times.iter().zip(&truncated) {
                let ts = DateTime::from_timestamp_nanos(at);
                assert_eq!(bucket.truncate(ts).unwrap(), start, "{bucket:?} at {at}");
            }
        }
    }

    #[test]
    fn a_span_holds_every_nanosecond_of_its_bucket_and_no_other() {
        let buckets = [
            "year", "quarter", "month", "week", "day", "hour", "minute", "second",
        ]
        .map(|unit| TimeBucket::date_unit(unit).unwrap())
        .into_iter()
        .chain(["1s", "15m", "1h30m", "7ms"].map(|d| TimeBucket::interval(d).unwrap()));
        let times = [
            -1_000_000_001,
            -1,
            0,
            1,
            1_709_208_000_000_000_000,
            i64::MAX,
            i64::MIN,
        ];
        for bucket in buckets {
            for at in times {
                let start = bucket.truncate_nanos(at);
                let (lo, hi) = bucket.span_nanos(at);
                assert!(lo <= at && at <= hi, "{bucket:?} at {at}: {lo}..{hi}");
                assert_eq!(
                    bucket.truncate_nanos(lo),
                    start,
                    "{bucket:?} at {at}: first"
                );
                assert_eq!(bucket.truncate_nanos(hi), start, "{bucket:?} at {at}: last");
                if let Some(before) = lo.checked_sub(1) {
                    assert_ne!(bucket.truncate_nanos(before), start, "{bucket:?} at {at}");
                }
                if let Some(after) = hi.checked_add(1) {
                    assert_ne!(bucket.truncate_nanos(after), start, "{bucket:?} at {at}");
                }
            }
        }
    }
}
