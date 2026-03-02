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

//! Table-Valued Functions (TVFs)
//!
//! Functions that return a set of rows, used in FROM clauses.

use super::{FunctionDataType, FunctionInfo, FunctionSignature, FunctionType, ScalarFunction};
use crate::core::{parse_timestamp, Error, Result, Row, RowVec, Value};
use chrono::{DateTime, Utc};

/// Maximum number of rows a TVF can generate to prevent OOM
const MAX_TVF_ROWS: usize = 10_000_000;

/// Parse an interval string like "1 day", "2 hours", "30 minutes" into chrono::Duration.
/// Uses checked arithmetic throughout to avoid panics on extreme values.
fn parse_interval(s: &str) -> std::result::Result<chrono::Duration, Error> {
    let s = s.trim();
    let parts: Vec<&str> = s.split_whitespace().collect();

    let overflow_err = |unit: &str, value: i64| -> Error {
        Error::invalid_argument(format!("Interval overflow: {} {}", value, unit))
    };

    if parts.len() < 2 {
        if let Ok(n) = s.parse::<i64>() {
            return chrono::Duration::try_days(n).ok_or_else(|| overflow_err("days", n));
        }
        return Err(Error::invalid_argument(format!(
            "Invalid interval format: '{}'. Expected format like '1 day', '2 hours'",
            s
        )));
    }

    if parts.len() > 2 {
        return Err(Error::invalid_argument(format!(
            "Compound intervals are not supported: '{}'. Use a single unit (e.g., '1 day' or '26 hours')",
            s
        )));
    }

    let value: i64 = parts[0]
        .parse()
        .map_err(|_| Error::invalid_argument(format!("Invalid interval value: '{}'", parts[0])))?;
    let unit = parts[1];

    if unit.eq_ignore_ascii_case("year") || unit.eq_ignore_ascii_case("years") {
        let days = value
            .checked_mul(365)
            .ok_or_else(|| overflow_err("years", value))?;
        chrono::Duration::try_days(days).ok_or_else(|| overflow_err("years", value))
    } else if unit.eq_ignore_ascii_case("month") || unit.eq_ignore_ascii_case("months") {
        let days = value
            .checked_mul(30)
            .ok_or_else(|| overflow_err("months", value))?;
        chrono::Duration::try_days(days).ok_or_else(|| overflow_err("months", value))
    } else if unit.eq_ignore_ascii_case("week") || unit.eq_ignore_ascii_case("weeks") {
        chrono::Duration::try_weeks(value).ok_or_else(|| overflow_err("weeks", value))
    } else if unit.eq_ignore_ascii_case("day") || unit.eq_ignore_ascii_case("days") {
        chrono::Duration::try_days(value).ok_or_else(|| overflow_err("days", value))
    } else if unit.eq_ignore_ascii_case("hour") || unit.eq_ignore_ascii_case("hours") {
        chrono::Duration::try_hours(value).ok_or_else(|| overflow_err("hours", value))
    } else if unit.eq_ignore_ascii_case("minute")
        || unit.eq_ignore_ascii_case("minutes")
        || unit.eq_ignore_ascii_case("min")
    {
        chrono::Duration::try_minutes(value).ok_or_else(|| overflow_err("minutes", value))
    } else if unit.eq_ignore_ascii_case("second")
        || unit.eq_ignore_ascii_case("seconds")
        || unit.eq_ignore_ascii_case("sec")
    {
        chrono::Duration::try_seconds(value).ok_or_else(|| overflow_err("seconds", value))
    } else if unit.eq_ignore_ascii_case("millisecond")
        || unit.eq_ignore_ascii_case("milliseconds")
        || unit.eq_ignore_ascii_case("ms")
    {
        chrono::Duration::try_milliseconds(value).ok_or_else(|| overflow_err("milliseconds", value))
    } else if unit.eq_ignore_ascii_case("microsecond")
        || unit.eq_ignore_ascii_case("microseconds")
        || unit.eq_ignore_ascii_case("us")
    {
        // microseconds always fits in TimeDelta (max ~292,000 years)
        Ok(chrono::Duration::microseconds(value))
    } else {
        Err(Error::invalid_argument(format!(
            "Unknown interval unit: '{}'. Supported: year, month, week, day, hour, minute, second, millisecond, microsecond",
            unit
        )))
    }
}

/// Trait for table-valued functions (functions that return a set of rows)
pub trait TableValuedFunction: Send + Sync {
    /// Get the function name (uppercase)
    fn name(&self) -> &str;

    /// Get the default column names for this TVF's output
    fn column_names(&self) -> Vec<String>;

    /// Generate rows from the given arguments.
    /// `limit` is an optional hint: if present, the function may stop after generating
    /// that many rows. This enables LIMIT pushdown to avoid generating millions of rows
    /// when only a few are needed.
    fn generate(&self, args: &[Value], limit: Option<usize>) -> Result<RowVec>;
}

/// GENERATE_SERIES(start, stop[, step])
///
/// Generates a series of values from start to stop (inclusive) with optional step.
/// Supports INTEGER, FLOAT, and TIMESTAMP/DATE types.
///
/// Examples:
///   generate_series(1, 5)       => 1, 2, 3, 4, 5
///   generate_series(0, 10, 2)   => 0, 2, 4, 6, 8, 10
///   generate_series(5, 1, -1)   => 5, 4, 3, 2, 1
///   generate_series(5, 1)       => 5, 4, 3, 2, 1 (auto-detect descending)
///   generate_series('2024-01-01', '2024-01-05', '1 day')
///   generate_series('2024-01-01 00:00:00', '2024-01-01 12:00:00', '1 hour')
pub struct GenerateSeriesFunction;

impl GenerateSeriesFunction {
    fn to_i64(val: &Value) -> Result<i64> {
        match val {
            Value::Integer(i) => Ok(*i),
            Value::Float(f) => Ok(*f as i64),
            Value::Null(_) => Err(Error::invalid_argument(
                "GENERATE_SERIES arguments must not be NULL",
            )),
            _ => Err(Error::invalid_argument(format!(
                "GENERATE_SERIES expects numeric arguments, got {}",
                val
            ))),
        }
    }

    fn to_f64(val: &Value) -> Result<f64> {
        match val {
            Value::Integer(i) => Ok(*i as f64),
            Value::Float(f) => Ok(*f),
            Value::Null(_) => Err(Error::invalid_argument(
                "GENERATE_SERIES arguments must not be NULL",
            )),
            _ => Err(Error::invalid_argument(format!(
                "GENERATE_SERIES expects numeric arguments, got {}",
                val
            ))),
        }
    }

    fn generate_integer(args: &[Value], limit: Option<usize>) -> Result<RowVec> {
        let start = Self::to_i64(&args[0])?;
        let stop = Self::to_i64(&args[1])?;
        let step = if args.len() == 3 {
            Self::to_i64(&args[2])?
        } else if start <= stop {
            1
        } else {
            -1
        };

        if step == 0 {
            return Err(Error::invalid_argument(
                "GENERATE_SERIES step must not be zero",
            ));
        }

        // Empty result on direction mismatch (PostgreSQL behavior)
        if (step > 0 && start > stop) || (step < 0 && start < stop) {
            return Ok(RowVec::new());
        }

        let max_rows = limit.unwrap_or(MAX_TVF_ROWS).min(MAX_TVF_ROWS);

        let capacity = if let Some(lim) = limit {
            lim.min(MAX_TVF_ROWS)
        } else {
            let raw_count =
                ((stop as i128 - start as i128) / step as i128).unsigned_abs() as usize + 1;
            raw_count.min(MAX_TVF_ROWS)
        };
        let mut rows = RowVec::with_capacity(capacity);
        let mut current = start;
        let mut row_id = 0i64;

        if step > 0 {
            while current <= stop {
                rows.push((row_id, Row::from_values(vec![Value::Integer(current)])));
                row_id += 1;
                if rows.len() >= max_rows {
                    break;
                }
                current = match current.checked_add(step) {
                    Some(v) => v,
                    None => break,
                };
            }
        } else {
            while current >= stop {
                rows.push((row_id, Row::from_values(vec![Value::Integer(current)])));
                row_id += 1;
                if rows.len() >= max_rows {
                    break;
                }
                current = match current.checked_add(step) {
                    Some(v) => v,
                    None => break,
                };
            }
        }

        Ok(rows)
    }

    /// Try to extract a DateTime<Utc> from a Value (Timestamp or parseable Text)
    fn to_timestamp(val: &Value) -> Result<DateTime<Utc>> {
        match val {
            Value::Timestamp(t) => Ok(*t),
            Value::Text(s) => parse_timestamp(s),
            Value::Null(_) => Err(Error::invalid_argument(
                "GENERATE_SERIES arguments must not be NULL",
            )),
            _ => Err(Error::invalid_argument(format!(
                "GENERATE_SERIES expects timestamp/date string, got {}",
                val
            ))),
        }
    }

    /// Check if a value looks like a timestamp (is Timestamp type, or Text parseable as timestamp)
    fn is_timestamp_like(val: &Value) -> bool {
        matches!(val, Value::Timestamp(_))
            || matches!(val, Value::Text(s) if parse_timestamp(s).is_ok())
    }

    fn generate_timestamp(args: &[Value], limit: Option<usize>) -> Result<RowVec> {
        let start = Self::to_timestamp(&args[0])?;
        let stop = Self::to_timestamp(&args[1])?;

        // Step is required for timestamp series
        let step = if args.len() == 3 {
            match &args[2] {
                Value::Text(s) => parse_interval(s)?,
                Value::Integer(n) => chrono::Duration::try_days(*n).ok_or_else(|| {
                    Error::invalid_argument(format!(
                        "GENERATE_SERIES timestamp step {} days overflows",
                        n
                    ))
                })?,
                _ => {
                    return Err(Error::invalid_argument(
                        "GENERATE_SERIES timestamp step must be an interval string (e.g., '1 day', '2 hours')",
                    ));
                }
            }
        } else if start <= stop {
            chrono::Duration::days(1)
        } else {
            chrono::Duration::days(-1)
        };

        if step.is_zero() {
            return Err(Error::invalid_argument(
                "GENERATE_SERIES step must not be zero",
            ));
        }

        // Empty result on direction mismatch
        if (step > chrono::Duration::zero() && start > stop)
            || (step < chrono::Duration::zero() && start < stop)
        {
            return Ok(RowVec::new());
        }

        let max_rows = limit.unwrap_or(MAX_TVF_ROWS).min(MAX_TVF_ROWS);

        // Pre-allocate capacity based on estimated count
        let step_secs = step.num_seconds().abs().max(1);
        let range_secs = (stop - start).num_seconds().abs();
        let estimated_count = (range_secs / step_secs) as usize + 1;

        let capacity = estimated_count.min(max_rows);
        let mut rows = RowVec::with_capacity(capacity);
        let mut current = start;
        let mut row_id = 0i64;

        if step > chrono::Duration::zero() {
            while current <= stop {
                rows.push((row_id, Row::from_values(vec![Value::Timestamp(current)])));
                row_id += 1;
                if rows.len() >= max_rows {
                    break;
                }
                current = match current.checked_add_signed(step) {
                    Some(next) => next,
                    None => break, // overflow → stop generating
                };
            }
        } else {
            while current >= stop {
                rows.push((row_id, Row::from_values(vec![Value::Timestamp(current)])));
                row_id += 1;
                if rows.len() >= max_rows {
                    break;
                }
                current = match current.checked_add_signed(step) {
                    Some(next) => next,
                    None => break, // overflow → stop generating
                };
            }
        }

        Ok(rows)
    }

    fn generate_float(args: &[Value], limit: Option<usize>) -> Result<RowVec> {
        let start = Self::to_f64(&args[0])?;
        let stop = Self::to_f64(&args[1])?;
        let step = if args.len() == 3 {
            Self::to_f64(&args[2])?
        } else if start <= stop {
            1.0
        } else {
            -1.0
        };

        if step == 0.0 {
            return Err(Error::invalid_argument(
                "GENERATE_SERIES step must not be zero",
            ));
        }

        if !step.is_finite() || !start.is_finite() || !stop.is_finite() {
            return Err(Error::invalid_argument(
                "GENERATE_SERIES arguments must be finite numbers",
            ));
        }

        // Empty result on direction mismatch
        if (step > 0.0 && start > stop) || (step < 0.0 && start < stop) {
            return Ok(RowVec::new());
        }

        let max_rows = limit.unwrap_or(MAX_TVF_ROWS).min(MAX_TVF_ROWS);

        let raw_count = ((stop - start) / step).abs();
        let capacity = if raw_count.is_finite() {
            (raw_count as usize + 1).min(max_rows)
        } else {
            max_rows
        };
        let mut rows = RowVec::with_capacity(capacity);
        let mut row_id = 0i64;
        // Use index-based generation to avoid floating-point drift
        let mut i = 0usize;

        loop {
            let current = start + (i as f64) * step;

            if step > 0.0 {
                // Allow small epsilon for floating-point comparison
                if current > stop + step.abs() * 1e-10 {
                    break;
                }
            } else if current < stop - step.abs() * 1e-10 {
                break;
            }

            rows.push((row_id, Row::from_values(vec![Value::Float(current)])));
            row_id += 1;
            i += 1;

            if i >= max_rows {
                break;
            }
        }

        Ok(rows)
    }
}

impl TableValuedFunction for GenerateSeriesFunction {
    fn name(&self) -> &str {
        "GENERATE_SERIES"
    }

    fn column_names(&self) -> Vec<String> {
        vec!["value".to_string()]
    }

    fn generate(&self, args: &[Value], limit: Option<usize>) -> Result<RowVec> {
        if args.len() < 2 || args.len() > 3 {
            return Err(Error::invalid_argument(
                "GENERATE_SERIES requires 2 or 3 arguments: (start, stop[, step])",
            ));
        }

        // Detect type: check first two args (start, stop) for timestamp-like values
        let has_timestamp = args.iter().take(2).any(Self::is_timestamp_like);

        if has_timestamp {
            Self::generate_timestamp(args, limit)
        } else {
            // If any argument is Float, use float path
            let has_float = args.iter().any(|a| matches!(a, Value::Float(_)));
            if has_float {
                Self::generate_float(args, limit)
            } else {
                Self::generate_integer(args, limit)
            }
        }
    }
}

/// Scalar version of GENERATE_SERIES that returns a JSON array.
/// Used when called as SELECT generate_series(1, 5) (without FROM clause).
/// Returns [1, 2, 3, 4, 5] as a text value, matching DuckDB behavior.
#[derive(Default)]
pub struct GenerateSeriesScalarFunction;

impl ScalarFunction for GenerateSeriesScalarFunction {
    fn name(&self) -> &str {
        "GENERATE_SERIES"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "GENERATE_SERIES",
            FunctionType::Scalar,
            "Generate a series of values as a JSON array",
            FunctionSignature::new(FunctionDataType::String, vec![FunctionDataType::Any], 2, 3),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        let tvf = GenerateSeriesFunction;
        let rows = tvf.generate(args, None)?;

        // Build JSON array string
        let mut result = String::with_capacity(rows.len() * 4);
        result.push('[');
        for (i, (_, row)) in rows.iter().enumerate() {
            if i > 0 {
                result.push_str(", ");
            }
            if let Some(val) = row.get(0) {
                match val {
                    Value::Integer(n) => result.push_str(&n.to_string()),
                    Value::Float(f) => result.push_str(&f.to_string()),
                    Value::Timestamp(t) => {
                        result.push('"');
                        result.push_str(&t.to_rfc3339());
                        result.push('"');
                    }
                    _ => result.push_str(&val.to_string()),
                }
            }
        }
        result.push(']');

        Ok(Value::from(result))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(GenerateSeriesScalarFunction)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_series_basic() {
        let gs = GenerateSeriesFunction;
        let rows = gs
            .generate(&[Value::Integer(1), Value::Integer(5)], None)
            .unwrap();
        assert_eq!(rows.len(), 5);
        assert_eq!(rows[0].1.get(0), Some(&Value::Integer(1)));
        assert_eq!(rows[4].1.get(0), Some(&Value::Integer(5)));
    }

    #[test]
    fn test_generate_series_with_step() {
        let gs = GenerateSeriesFunction;
        let rows = gs
            .generate(
                &[Value::Integer(0), Value::Integer(10), Value::Integer(2)],
                None,
            )
            .unwrap();
        assert_eq!(rows.len(), 6); // 0, 2, 4, 6, 8, 10
        assert_eq!(rows[0].1.get(0), Some(&Value::Integer(0)));
        assert_eq!(rows[5].1.get(0), Some(&Value::Integer(10)));
    }

    #[test]
    fn test_generate_series_descending() {
        let gs = GenerateSeriesFunction;
        let rows = gs
            .generate(
                &[Value::Integer(5), Value::Integer(1), Value::Integer(-1)],
                None,
            )
            .unwrap();
        assert_eq!(rows.len(), 5);
        assert_eq!(rows[0].1.get(0), Some(&Value::Integer(5)));
        assert_eq!(rows[4].1.get(0), Some(&Value::Integer(1)));
    }

    #[test]
    fn test_generate_series_auto_descending() {
        let gs = GenerateSeriesFunction;
        let rows = gs
            .generate(&[Value::Integer(5), Value::Integer(1)], None)
            .unwrap();
        assert_eq!(rows.len(), 5);
        assert_eq!(rows[0].1.get(0), Some(&Value::Integer(5)));
    }

    #[test]
    fn test_generate_series_empty_mismatch() {
        let gs = GenerateSeriesFunction;
        let rows = gs
            .generate(
                &[Value::Integer(1), Value::Integer(5), Value::Integer(-1)],
                None,
            )
            .unwrap();
        assert_eq!(rows.len(), 0);
    }

    #[test]
    fn test_generate_series_single() {
        let gs = GenerateSeriesFunction;
        let rows = gs
            .generate(&[Value::Integer(3), Value::Integer(3)], None)
            .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].1.get(0), Some(&Value::Integer(3)));
    }

    #[test]
    fn test_generate_series_zero_step_error() {
        let gs = GenerateSeriesFunction;
        let result = gs.generate(
            &[Value::Integer(1), Value::Integer(5), Value::Integer(0)],
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_generate_series_wrong_args_error() {
        let gs = GenerateSeriesFunction;
        assert!(gs.generate(&[Value::Integer(1)], None).is_err());
        assert!(gs
            .generate(
                &[
                    Value::Integer(1),
                    Value::Integer(2),
                    Value::Integer(3),
                    Value::Integer(4),
                ],
                None,
            )
            .is_err());
    }

    #[test]
    fn test_generate_series_float() {
        let gs = GenerateSeriesFunction;
        let rows = gs
            .generate(
                &[Value::Float(0.0), Value::Float(1.0), Value::Float(0.5)],
                None,
            )
            .unwrap();
        assert_eq!(rows.len(), 3); // 0.0, 0.5, 1.0
    }

    #[test]
    fn test_generate_series_timestamp_days() {
        let gs = GenerateSeriesFunction;
        let rows = gs
            .generate(
                &[
                    Value::from("2024-01-01"),
                    Value::from("2024-01-05"),
                    Value::from("1 day"),
                ],
                None,
            )
            .unwrap();
        assert_eq!(rows.len(), 5);
        // First and last should be timestamps
        assert!(matches!(rows[0].1.get(0), Some(Value::Timestamp(_))));
        assert!(matches!(rows[4].1.get(0), Some(Value::Timestamp(_))));
    }

    #[test]
    fn test_generate_series_timestamp_hours() {
        let gs = GenerateSeriesFunction;
        let rows = gs
            .generate(
                &[
                    Value::from("2024-01-01 00:00:00"),
                    Value::from("2024-01-01 06:00:00"),
                    Value::from("2 hours"),
                ],
                None,
            )
            .unwrap();
        assert_eq!(rows.len(), 4); // 00:00, 02:00, 04:00, 06:00
    }

    #[test]
    fn test_generate_series_timestamp_auto_step() {
        let gs = GenerateSeriesFunction;
        // Without step, auto-detect 1 day
        let rows = gs
            .generate(
                &[Value::from("2024-01-01"), Value::from("2024-01-03")],
                None,
            )
            .unwrap();
        assert_eq!(rows.len(), 3); // Jan 1, 2, 3
    }

    #[test]
    fn test_generate_series_timestamp_descending() {
        let gs = GenerateSeriesFunction;
        let rows = gs
            .generate(
                &[
                    Value::from("2024-01-05"),
                    Value::from("2024-01-01"),
                    Value::from("-1 day"),
                ],
                None,
            )
            .unwrap();
        assert_eq!(rows.len(), 5);
    }

    #[test]
    fn test_parse_interval() {
        assert_eq!(parse_interval("1 day").unwrap(), chrono::Duration::days(1));
        assert_eq!(
            parse_interval("2 hours").unwrap(),
            chrono::Duration::hours(2)
        );
        assert_eq!(
            parse_interval("30 minutes").unwrap(),
            chrono::Duration::minutes(30)
        );
        assert_eq!(
            parse_interval("1 week").unwrap(),
            chrono::Duration::weeks(1)
        );
        assert_eq!(
            parse_interval("-1 day").unwrap(),
            chrono::Duration::days(-1)
        );
        assert!(parse_interval("invalid").is_err());
        // Compound intervals should error, not silently truncate
        assert!(parse_interval("1 day 2 hours").is_err());
        assert!(parse_interval("1 hour 30 minutes").is_err());
    }
}
