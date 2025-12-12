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

//! Scalar Functions
//!
//! This module provides scalar functions for SQL queries:
//!
//! ## String Functions
//! - [`UpperFunction`] - UPPER(string)
//! - [`LowerFunction`] - LOWER(string)
//! - [`LengthFunction`] - LENGTH(string)
//! - [`ConcatFunction`] - CONCAT(string, ...)
//! - [`SubstringFunction`] - SUBSTRING(string, start, length)
//!
//! ## Math Functions
//! - [`AbsFunction`] - ABS(number)
//! - [`RoundFunction`] - ROUND(number, decimals)
//! - [`FloorFunction`] - FLOOR(number)
//! - [`CeilingFunction`] - CEILING(number)
//!
//! ## Date/Time Functions
//! - [`DateTruncFunction`] - DATE_TRUNC(unit, timestamp)
//! - [`TimeTruncFunction`] - TIME_TRUNC(duration, timestamp)
//!
//! ## Conversion Functions
//! - [`CastFunction`] - CAST(value AS type) - Type conversion
//! - [`CollateFunction`] - COLLATE(string, collation) - Apply collation
//!
//! ## Utility Functions
//! - [`CoalesceFunction`] - COALESCE(value, ...)
//! - [`NowFunction`] - NOW()
//! - [`VersionFunction`] - VERSION()

mod conversion;
mod datetime;
mod math;
mod string;
mod utility;

pub use conversion::{CastFunction, CollateFunction};
pub use datetime::{
    CurrentDateFunction, CurrentTimestampFunction, DateAddFunction, DateDiffAliasFunction,
    DateDiffFunction, DateSubFunction, DateTruncFunction, DayFunction, ExtractFunction,
    HourFunction, MinuteFunction, MonthFunction, SecondFunction, TimeTruncFunction, ToCharFunction,
    VersionFunction, YearFunction,
};
pub use math::{
    AbsFunction, CeilFunction, CeilingFunction, CosFunction, ExpFunction, FloorFunction,
    LnFunction, Log10Function, Log2Function, LogFunction, ModFunction, PiFunction, PowFunction,
    PowerFunction, RandomFunction, RoundFunction, SignFunction, SinFunction, SqrtFunction,
    TanFunction, TruncFunction, TruncateFunction,
};
pub use string::{
    CharFunction, CharLengthFunction, ConcatFunction, ConcatWsFunction, InstrFunction,
    LeftFunction, LengthFunction, LocateFunction, LowerFunction, LpadFunction, LtrimFunction,
    PositionFunction, RepeatFunction, ReplaceFunction, ReverseFunction, RightFunction,
    RpadFunction, RtrimFunction, SplitPartFunction, StrposFunction, SubstrFunction,
    SubstringFunction, TrimFunction, UpperFunction,
};
pub use utility::{
    CoalesceFunction, GreatestFunction, IfNullFunction, IifFunction, JsonArrayFunction,
    JsonArrayLengthFunction, JsonExtractFunction, JsonKeysFunction, JsonObjectFunction,
    JsonTypeFunction, JsonTypeOfFunction, JsonValidFunction, LeastFunction, NowFunction,
    NullIfFunction, SleepFunction, TypeOfFunction,
};

use crate::core::Value;

/// Macro to validate argument count for scalar functions.
///
/// # Usage
/// ```ignore
/// // Exact count
/// validate_arg_count!(args, "UPPER", 1);
///
/// // Range (min, max inclusive)
/// validate_arg_count!(args, "SUBSTRING", 2, 3);
/// ```
#[macro_export]
macro_rules! validate_arg_count {
    // Exact count
    ($args:expr, $name:expr, $exact:expr) => {
        if $args.len() != $exact {
            return Err($crate::core::Error::invalid_argument(format!(
                "{} requires exactly {} argument{}, got {}",
                $name,
                $exact,
                if $exact == 1 { "" } else { "s" },
                $args.len()
            )));
        }
    };
    // Range (min to max inclusive)
    ($args:expr, $name:expr, $min:expr, $max:expr) => {
        if $args.len() < $min || $args.len() > $max {
            return Err($crate::core::Error::invalid_argument(format!(
                "{} requires {} to {} arguments, got {}",
                $name,
                $min,
                $max,
                $args.len()
            )));
        }
    };
}

/// Convert a Value to a string representation
pub fn value_to_string(value: &Value) -> String {
    if value.is_null() {
        return String::new();
    }
    match value {
        Value::Null(_) => String::new(),
        Value::Integer(i) => i.to_string(),
        Value::Float(f) => f.to_string(),
        Value::Text(s) => s.to_string(),
        Value::Boolean(b) => b.to_string(),
        Value::Timestamp(t) => t.to_rfc3339(),
        Value::Json(j) => j.to_string(),
    }
}

/// Try to convert a Value to f64
pub fn value_to_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Integer(i) => Some(*i as f64),
        Value::Float(f) => Some(*f),
        Value::Text(s) => s.parse().ok(),
        _ => None,
    }
}

/// Try to convert a Value to i64
pub fn value_to_i64(value: &Value) -> Option<i64> {
    match value {
        Value::Integer(i) => Some(*i),
        Value::Float(f) => Some(*f as i64),
        Value::Text(s) => s.parse().ok(),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_to_string() {
        assert_eq!(value_to_string(&Value::Integer(42)), "42");
        assert_eq!(value_to_string(&Value::Float(3.5)), "3.5");
        assert_eq!(value_to_string(&Value::text("hello")), "hello");
        assert_eq!(value_to_string(&Value::Boolean(true)), "true");
        assert_eq!(value_to_string(&Value::null_unknown()), "");
    }

    #[test]
    fn test_value_to_f64() {
        assert_eq!(value_to_f64(&Value::Integer(42)), Some(42.0));
        assert_eq!(value_to_f64(&Value::Float(3.5)), Some(3.5));
        assert_eq!(value_to_f64(&Value::text("2.5")), Some(2.5));
        assert_eq!(value_to_f64(&Value::null_unknown()), None);
    }

    #[test]
    fn test_value_to_i64() {
        assert_eq!(value_to_i64(&Value::Integer(42)), Some(42));
        assert_eq!(value_to_i64(&Value::Float(3.7)), Some(3));
        assert_eq!(value_to_i64(&Value::text("100")), Some(100));
        assert_eq!(value_to_i64(&Value::null_unknown()), None);
    }
}
