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

//! Value type for Stoolap - runtime values with type information
//!
//! This module provides a unified Value enum that represents SQL values
//! with full type information and conversion capabilities.

use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use chrono::{DateTime, NaiveDate, NaiveDateTime, NaiveTime, TimeZone, Utc};

use super::error::{Error, Result};
use super::types::DataType;

/// Timestamp formats supported for parsing
/// Order matters - more specific formats first
const TIMESTAMP_FORMATS: &[&str] = &[
    "%Y-%m-%dT%H:%M:%S%.f%:z", // RFC3339 with fractional seconds
    "%Y-%m-%dT%H:%M:%S%:z",    // RFC3339
    "%Y-%m-%dT%H:%M:%SZ",      // RFC3339 UTC
    "%Y-%m-%dT%H:%M:%S",       // ISO without timezone
    "%Y-%m-%d %H:%M:%S%.f",    // SQL-style with fractional seconds
    "%Y-%m-%d %H:%M:%S",       // SQL-style
    "%Y-%m-%d",                // Date only
    "%Y/%m/%d %H:%M:%S",       // Alternative with slashes
    "%Y/%m/%d",                // Alternative date only
    "%m/%d/%Y",                // US format
    "%d/%m/%Y",                // European format
];

const TIME_FORMATS: &[&str] = &[
    "%H:%M:%S%.f", // High precision
    "%H:%M:%S",    // Standard
    "%H:%M",       // Hours and minutes only
];

/// A runtime value with type information
///
/// Each variant carries its data directly, avoiding the need for interface
/// indirection or separate value references.
///
/// Note: Text and Json use Arc<str> for cheap cloning during row operations.
/// This is critical for scan performance where rows are cloned frequently.
#[derive(Debug, Clone)]
pub enum Value {
    /// NULL value with optional type hint
    Null(DataType),

    /// 64-bit signed integer
    Integer(i64),

    /// 64-bit floating point
    Float(f64),

    /// UTF-8 text string (Arc for cheap cloning)
    Text(Arc<str>),

    /// Boolean value
    Boolean(bool),

    /// Timestamp (UTC)
    Timestamp(DateTime<Utc>),

    /// JSON document (Arc for cheap cloning)
    Json(Arc<str>),
}

impl Value {
    // =========================================================================
    // Constructors
    // =========================================================================

    /// Create a NULL value with a type hint
    pub fn null(data_type: DataType) -> Self {
        Value::Null(data_type)
    }

    /// Create a NULL value with unknown type
    pub fn null_unknown() -> Self {
        Value::Null(DataType::Null)
    }

    /// Create an integer value
    pub fn integer(value: i64) -> Self {
        Value::Integer(value)
    }

    /// Create a float value
    pub fn float(value: f64) -> Self {
        Value::Float(value)
    }

    /// Create a text value
    pub fn text(value: impl Into<String>) -> Self {
        Value::Text(Arc::from(value.into().as_str()))
    }

    /// Create a text value from Arc<str> (zero-copy)
    pub fn text_arc(value: Arc<str>) -> Self {
        Value::Text(value)
    }

    /// Create a boolean value
    pub fn boolean(value: bool) -> Self {
        Value::Boolean(value)
    }

    /// Create a timestamp value
    pub fn timestamp(value: DateTime<Utc>) -> Self {
        Value::Timestamp(value)
    }

    /// Create a JSON value
    pub fn json(value: impl Into<String>) -> Self {
        Value::Json(Arc::from(value.into().as_str()))
    }

    /// Create a JSON value from Arc<str> (zero-copy)
    pub fn json_arc(value: Arc<str>) -> Self {
        Value::Json(value)
    }

    // =========================================================================
    // Type accessors
    // =========================================================================

    /// Returns the data type of this value
    pub fn data_type(&self) -> DataType {
        match self {
            Value::Null(dt) => *dt,
            Value::Integer(_) => DataType::Integer,
            Value::Float(_) => DataType::Float,
            Value::Text(_) => DataType::Text,
            Value::Boolean(_) => DataType::Boolean,
            Value::Timestamp(_) => DataType::Timestamp,
            Value::Json(_) => DataType::Json,
        }
    }

    /// Returns true if this value is NULL
    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null(_))
    }

    // =========================================================================
    // Value extractors
    // =========================================================================

    /// Extract as i64, with type coercion
    ///
    /// Returns None if:
    /// - Value is NULL
    /// - Conversion is not possible
    pub fn as_int64(&self) -> Option<i64> {
        match self {
            Value::Null(_) => None,
            Value::Integer(v) => Some(*v),
            Value::Float(v) => Some(*v as i64),
            Value::Text(s) => s
                .parse::<i64>()
                .ok()
                .or_else(|| s.parse::<f64>().ok().map(|f| f as i64)),
            Value::Boolean(b) => Some(if *b { 1 } else { 0 }),
            Value::Timestamp(t) => Some(t.timestamp_nanos_opt().unwrap_or(0)),
            Value::Json(_) => None,
        }
    }

    /// Extract as f64, with type coercion
    pub fn as_float64(&self) -> Option<f64> {
        match self {
            Value::Null(_) => None,
            Value::Integer(v) => Some(*v as f64),
            Value::Float(v) => Some(*v),
            Value::Text(s) => s.parse::<f64>().ok(),
            Value::Boolean(b) => Some(if *b { 1.0 } else { 0.0 }),
            Value::Timestamp(_) => None,
            Value::Json(_) => None,
        }
    }

    /// Extract as boolean, with type coercion
    pub fn as_boolean(&self) -> Option<bool> {
        match self {
            Value::Null(_) => None,
            Value::Integer(v) => Some(*v != 0),
            Value::Float(v) => Some(*v != 0.0),
            Value::Text(s) => {
                // OPTIMIZATION: Use eq_ignore_ascii_case to avoid allocation
                let s_ref: &str = s.as_ref();
                if s_ref.eq_ignore_ascii_case("true")
                    || s_ref.eq_ignore_ascii_case("t")
                    || s_ref.eq_ignore_ascii_case("yes")
                    || s_ref.eq_ignore_ascii_case("y")
                    || s_ref == "1"
                {
                    Some(true)
                } else if s_ref.eq_ignore_ascii_case("false")
                    || s_ref.eq_ignore_ascii_case("f")
                    || s_ref.eq_ignore_ascii_case("no")
                    || s_ref.eq_ignore_ascii_case("n")
                    || s_ref == "0"
                    || s_ref.is_empty()
                {
                    Some(false)
                } else {
                    s_ref.parse::<f64>().ok().map(|f| f != 0.0)
                }
            }
            Value::Boolean(b) => Some(*b),
            Value::Timestamp(_) => None,
            Value::Json(_) => None,
        }
    }

    /// Extract as String, with type coercion
    pub fn as_string(&self) -> Option<String> {
        match self {
            Value::Null(_) => None,
            Value::Integer(v) => Some(v.to_string()),
            Value::Float(v) => Some(format_float(*v)),
            Value::Text(s) => Some(s.to_string()),
            Value::Boolean(b) => Some(if *b { "true" } else { "false" }.to_string()),
            Value::Timestamp(t) => Some(t.to_rfc3339()),
            Value::Json(s) => Some(s.to_string()),
        }
    }

    /// Extract as string reference (avoids clone for Text/Json)
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Text(s) | Value::Json(s) => Some(s),
            _ => None,
        }
    }

    /// Extract as Arc<str> (cheap clone for Text/Json)
    pub fn as_arc_str(&self) -> Option<Arc<str>> {
        match self {
            Value::Text(s) | Value::Json(s) => Some(Arc::clone(s)),
            _ => None,
        }
    }

    /// Extract as DateTime<Utc>
    pub fn as_timestamp(&self) -> Option<DateTime<Utc>> {
        match self {
            Value::Null(_) => None,
            Value::Timestamp(t) => Some(*t),
            Value::Text(s) => parse_timestamp(s).ok(),
            Value::Integer(nanos) => {
                // Interpret as nanoseconds since Unix epoch
                DateTime::from_timestamp(*nanos / 1_000_000_000, (*nanos % 1_000_000_000) as u32)
            }
            _ => None,
        }
    }

    /// Extract as JSON string
    pub fn as_json(&self) -> Option<&str> {
        match self {
            Value::Null(_) => Some("{}"),
            Value::Json(s) => Some(s),
            _ => None,
        }
    }

    // =========================================================================
    // Comparison
    // =========================================================================

    /// Compare two values for ordering
    ///
    /// Returns:
    /// - Ok(Ordering::Less) if self < other
    /// - Ok(Ordering::Equal) if self == other
    /// - Ok(Ordering::Greater) if self > other
    /// - Err if comparison is not possible
    pub fn compare(&self, other: &Value) -> Result<Ordering> {
        // Handle NULL comparisons
        if self.is_null() || other.is_null() {
            if self.is_null() && other.is_null() {
                return Ok(Ordering::Equal);
            }
            return Err(Error::NullComparison);
        }

        // Same type comparison (most efficient path)
        if self.data_type() == other.data_type() {
            return self.compare_same_type(other);
        }

        // Cross-type numeric comparison (integer vs float)
        if self.data_type().is_numeric() && other.data_type().is_numeric() {
            let v1 = self.as_float64().unwrap();
            let v2 = other.as_float64().unwrap();
            return Ok(compare_floats(v1, v2));
        }

        // Fall back to string comparison for mixed types
        let s1 = self.as_string().unwrap_or_default();
        let s2 = other.as_string().unwrap_or_default();
        Ok(s1.cmp(&s2))
    }

    /// Compare values of the same type
    fn compare_same_type(&self, other: &Value) -> Result<Ordering> {
        match (self, other) {
            (Value::Integer(a), Value::Integer(b)) => Ok(a.cmp(b)),
            (Value::Float(a), Value::Float(b)) => Ok(compare_floats(*a, *b)),
            (Value::Text(a), Value::Text(b)) => Ok(a.cmp(b)),
            (Value::Boolean(a), Value::Boolean(b)) => Ok(a.cmp(b)),
            (Value::Timestamp(a), Value::Timestamp(b)) => Ok(a.cmp(b)),
            (Value::Json(a), Value::Json(b)) => {
                // JSON can only test equality, not ordering
                if a == b {
                    Ok(Ordering::Equal)
                } else {
                    Err(Error::IncomparableTypes)
                }
            }
            _ => Err(Error::IncomparableTypes),
        }
    }

    // =========================================================================
    // Construction from typed values
    // =========================================================================

    /// Create a Value from a typed value with explicit data type
    pub fn from_typed(value: Option<&dyn std::any::Any>, data_type: DataType) -> Self {
        match value {
            None => Value::Null(data_type),
            Some(v) => {
                // Try to downcast based on expected type
                match data_type {
                    DataType::Integer => {
                        if let Some(&i) = v.downcast_ref::<i64>() {
                            Value::Integer(i)
                        } else if let Some(&i) = v.downcast_ref::<i32>() {
                            Value::Integer(i as i64)
                        } else if let Some(s) = v.downcast_ref::<String>() {
                            s.parse::<i64>()
                                .map(Value::Integer)
                                .unwrap_or(Value::Null(data_type))
                        } else {
                            Value::Null(data_type)
                        }
                    }
                    DataType::Float => {
                        if let Some(&f) = v.downcast_ref::<f64>() {
                            Value::Float(f)
                        } else if let Some(&i) = v.downcast_ref::<i64>() {
                            Value::Float(i as f64)
                        } else if let Some(s) = v.downcast_ref::<String>() {
                            s.parse::<f64>()
                                .map(Value::Float)
                                .unwrap_or(Value::Null(data_type))
                        } else {
                            Value::Null(data_type)
                        }
                    }
                    DataType::Text => {
                        if let Some(s) = v.downcast_ref::<String>() {
                            Value::Text(Arc::from(s.as_str()))
                        } else if let Some(&s) = v.downcast_ref::<&str>() {
                            Value::Text(Arc::from(s))
                        } else {
                            Value::Null(data_type)
                        }
                    }
                    DataType::Boolean => {
                        if let Some(&b) = v.downcast_ref::<bool>() {
                            Value::Boolean(b)
                        } else if let Some(&i) = v.downcast_ref::<i64>() {
                            Value::Boolean(i != 0)
                        } else {
                            Value::Null(data_type)
                        }
                    }
                    DataType::Timestamp => {
                        if let Some(&t) = v.downcast_ref::<DateTime<Utc>>() {
                            Value::Timestamp(t)
                        } else if let Some(s) = v.downcast_ref::<String>() {
                            parse_timestamp(s)
                                .map(Value::Timestamp)
                                .unwrap_or(Value::Null(data_type))
                        } else {
                            Value::Null(data_type)
                        }
                    }
                    DataType::Json => {
                        if let Some(s) = v.downcast_ref::<String>() {
                            // Validate JSON
                            if serde_json::from_str::<serde_json::Value>(s).is_ok() {
                                Value::Json(Arc::from(s.as_str()))
                            } else {
                                Value::Null(data_type)
                            }
                        } else {
                            Value::Null(data_type)
                        }
                    }
                    DataType::Null => Value::Null(DataType::Null),
                }
            }
        }
    }

    // =========================================================================
    // Type coercion
    // =========================================================================

    /// Coerce this value to the target data type
    ///
    /// Type coercion rules:
    /// - Integer column receiving Float → converts to Integer
    /// - Float column receiving Integer → converts to Float
    /// - Text column receiving any type → converts to Text
    /// - Timestamp column receiving String → parses timestamp
    /// - JSON column receiving valid JSON string → stores as JSON
    /// - Boolean column receiving Integer/String → converts to Boolean
    ///
    /// Returns the coerced value, or NULL if coercion fails.
    pub fn coerce_to_type(&self, target_type: DataType) -> Value {
        // NULL stays NULL (with target type hint)
        if self.is_null() {
            return Value::Null(target_type);
        }

        // Same type - no conversion needed
        if self.data_type() == target_type {
            return self.clone();
        }

        match target_type {
            DataType::Integer => {
                // Convert to INTEGER
                match self {
                    Value::Integer(v) => Value::Integer(*v),
                    Value::Float(v) => Value::Integer(*v as i64),
                    Value::Text(s) => s
                        .parse::<i64>()
                        .map(Value::Integer)
                        .unwrap_or(Value::Null(target_type)),
                    Value::Boolean(b) => Value::Integer(if *b { 1 } else { 0 }),
                    _ => Value::Null(target_type),
                }
            }
            DataType::Float => {
                // Convert to FLOAT
                match self {
                    Value::Float(v) => Value::Float(*v),
                    Value::Integer(v) => Value::Float(*v as f64),
                    Value::Text(s) => s
                        .parse::<f64>()
                        .map(Value::Float)
                        .unwrap_or(Value::Null(target_type)),
                    Value::Boolean(b) => Value::Float(if *b { 1.0 } else { 0.0 }),
                    _ => Value::Null(target_type),
                }
            }
            DataType::Text => {
                // Convert to TEXT - everything can become text
                match self {
                    Value::Text(s) => Value::Text(Arc::clone(s)),
                    Value::Integer(v) => Value::Text(Arc::from(v.to_string().as_str())),
                    Value::Float(v) => Value::Text(Arc::from(format_float(*v).as_str())),
                    Value::Boolean(b) => Value::Text(Arc::from(if *b { "true" } else { "false" })),
                    Value::Timestamp(t) => Value::Text(Arc::from(t.to_rfc3339().as_str())),
                    Value::Json(s) => Value::Text(Arc::clone(s)),
                    Value::Null(_) => Value::Null(target_type),
                }
            }
            DataType::Boolean => {
                // Convert to BOOLEAN
                match self {
                    Value::Boolean(b) => Value::Boolean(*b),
                    Value::Integer(v) => Value::Boolean(*v != 0),
                    Value::Float(v) => Value::Boolean(*v != 0.0),
                    Value::Text(s) => {
                        // OPTIMIZATION: Use eq_ignore_ascii_case to avoid allocation
                        let s_ref: &str = s.as_ref();
                        if s_ref.eq_ignore_ascii_case("true")
                            || s_ref.eq_ignore_ascii_case("t")
                            || s_ref.eq_ignore_ascii_case("yes")
                            || s_ref.eq_ignore_ascii_case("y")
                            || s_ref == "1"
                        {
                            Value::Boolean(true)
                        } else if s_ref.eq_ignore_ascii_case("false")
                            || s_ref.eq_ignore_ascii_case("f")
                            || s_ref.eq_ignore_ascii_case("no")
                            || s_ref.eq_ignore_ascii_case("n")
                            || s_ref == "0"
                        {
                            Value::Boolean(false)
                        } else {
                            Value::Null(target_type)
                        }
                    }
                    _ => Value::Null(target_type),
                }
            }
            DataType::Timestamp => {
                // Convert to TIMESTAMP
                match self {
                    Value::Timestamp(t) => Value::Timestamp(*t),
                    Value::Text(s) => parse_timestamp(s)
                        .map(Value::Timestamp)
                        .unwrap_or(Value::Null(target_type)),
                    Value::Integer(nanos) => {
                        // Interpret as nanoseconds since Unix epoch
                        DateTime::from_timestamp(
                            *nanos / 1_000_000_000,
                            (*nanos % 1_000_000_000) as u32,
                        )
                        .map(Value::Timestamp)
                        .unwrap_or(Value::Null(target_type))
                    }
                    _ => Value::Null(target_type),
                }
            }
            DataType::Json => {
                // Convert to JSON
                match self {
                    Value::Json(s) => Value::Json(Arc::clone(s)),
                    Value::Text(s) => {
                        // Validate JSON
                        if serde_json::from_str::<serde_json::Value>(s).is_ok() {
                            Value::Json(Arc::clone(s))
                        } else {
                            Value::Null(target_type)
                        }
                    }
                    // Convert other types to JSON representation
                    Value::Integer(v) => Value::Json(Arc::from(v.to_string().as_str())),
                    Value::Float(v) => Value::Json(Arc::from(format_float(*v).as_str())),
                    Value::Boolean(b) => Value::Json(Arc::from(if *b { "true" } else { "false" })),
                    _ => Value::Null(target_type),
                }
            }
            DataType::Null => Value::Null(DataType::Null),
        }
    }

    /// Coerce value to target type, consuming self
    /// OPTIMIZATION: Avoids clone when types already match
    #[inline]
    pub fn into_coerce_to_type(self, target_type: DataType) -> Value {
        // NULL stays NULL (with target type hint)
        if self.is_null() {
            return Value::Null(target_type);
        }

        // Same type - no conversion needed, return self directly
        if self.data_type() == target_type {
            return self;
        }

        match target_type {
            DataType::Integer => match &self {
                Value::Integer(v) => Value::Integer(*v),
                Value::Float(v) => Value::Integer(*v as i64),
                Value::Text(s) => s
                    .parse::<i64>()
                    .map(Value::Integer)
                    .unwrap_or(Value::Null(target_type)),
                Value::Boolean(b) => Value::Integer(if *b { 1 } else { 0 }),
                _ => Value::Null(target_type),
            },
            DataType::Float => match &self {
                Value::Float(v) => Value::Float(*v),
                Value::Integer(v) => Value::Float(*v as f64),
                Value::Text(s) => s
                    .parse::<f64>()
                    .map(Value::Float)
                    .unwrap_or(Value::Null(target_type)),
                Value::Boolean(b) => Value::Float(if *b { 1.0 } else { 0.0 }),
                _ => Value::Null(target_type),
            },
            DataType::Text => match self {
                Value::Text(s) => Value::Text(s),
                Value::Integer(v) => Value::Text(Arc::from(v.to_string().as_str())),
                Value::Float(v) => Value::Text(Arc::from(format_float(v).as_str())),
                Value::Boolean(b) => Value::Text(Arc::from(if b { "true" } else { "false" })),
                Value::Timestamp(t) => Value::Text(Arc::from(t.to_rfc3339().as_str())),
                Value::Json(s) => Value::Text(s),
                Value::Null(_) => Value::Null(target_type),
            },
            DataType::Boolean => match &self {
                Value::Boolean(b) => Value::Boolean(*b),
                Value::Integer(v) => Value::Boolean(*v != 0),
                Value::Float(v) => Value::Boolean(*v != 0.0),
                Value::Text(s) => {
                    // OPTIMIZATION: Use eq_ignore_ascii_case to avoid allocation
                    let s_ref: &str = s.as_ref();
                    if s_ref.eq_ignore_ascii_case("true")
                        || s_ref.eq_ignore_ascii_case("t")
                        || s_ref.eq_ignore_ascii_case("yes")
                        || s_ref.eq_ignore_ascii_case("y")
                        || s_ref == "1"
                    {
                        Value::Boolean(true)
                    } else if s_ref.eq_ignore_ascii_case("false")
                        || s_ref.eq_ignore_ascii_case("f")
                        || s_ref.eq_ignore_ascii_case("no")
                        || s_ref.eq_ignore_ascii_case("n")
                        || s_ref == "0"
                    {
                        Value::Boolean(false)
                    } else {
                        Value::Null(target_type)
                    }
                }
                _ => Value::Null(target_type),
            },
            DataType::Timestamp => match self {
                Value::Timestamp(t) => Value::Timestamp(t),
                Value::Text(s) => parse_timestamp(&s)
                    .map(Value::Timestamp)
                    .unwrap_or(Value::Null(target_type)),
                Value::Integer(nanos) => {
                    DateTime::from_timestamp(nanos / 1_000_000_000, (nanos % 1_000_000_000) as u32)
                        .map(Value::Timestamp)
                        .unwrap_or(Value::Null(target_type))
                }
                _ => Value::Null(target_type),
            },
            DataType::Json => match self {
                Value::Json(s) => Value::Json(s),
                Value::Text(s) => {
                    if serde_json::from_str::<serde_json::Value>(&s).is_ok() {
                        Value::Json(s)
                    } else {
                        Value::Null(target_type)
                    }
                }
                Value::Integer(v) => Value::Json(Arc::from(v.to_string().as_str())),
                Value::Float(v) => Value::Json(Arc::from(format_float(v).as_str())),
                Value::Boolean(b) => Value::Json(Arc::from(if b { "true" } else { "false" })),
                _ => Value::Null(target_type),
            },
            DataType::Null => Value::Null(DataType::Null),
        }
    }
}

// =========================================================================
// Trait implementations
// =========================================================================

impl Default for Value {
    fn default() -> Self {
        Value::Null(DataType::Null)
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Null(_) => write!(f, "NULL"),
            Value::Integer(v) => write!(f, "{}", v),
            Value::Float(v) => write!(f, "{}", format_float(*v)),
            Value::Text(s) => write!(f, "{}", s),
            Value::Boolean(b) => write!(f, "{}", if *b { "true" } else { "false" }),
            Value::Timestamp(t) => write!(f, "{}", t.to_rfc3339()),
            Value::Json(s) => write!(f, "{}", s),
        }
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        // Handle NULL: NULL equals NULL
        if self.is_null() && other.is_null() {
            return true;
        }
        if self.is_null() || other.is_null() {
            return false;
        }

        match (self, other) {
            // Same type comparisons
            (Value::Integer(a), Value::Integer(b)) => a == b,
            (Value::Float(a), Value::Float(b)) => {
                // Handle NaN: NaN != NaN in IEEE 754, but we consider them equal
                if a.is_nan() && b.is_nan() {
                    true
                } else {
                    a == b
                }
            }
            // Cross-type numeric comparison: Integer vs Float
            // This is critical for queries like WHERE id = 5.0 or WHERE price = 100
            (Value::Integer(i), Value::Float(f)) | (Value::Float(f), Value::Integer(i)) => {
                *f == (*i as f64)
            }
            (Value::Text(a), Value::Text(b)) => a == b,
            (Value::Boolean(a), Value::Boolean(b)) => a == b,
            (Value::Timestamp(a), Value::Timestamp(b)) => a == b,
            (Value::Json(a), Value::Json(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for Value {}

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Note: We must ensure that values that are equal have the same hash.
        // Since Integer(5) == Float(5.0), they must hash the same.
        // We achieve this by hashing numeric types as their f64 bit representation.
        match self {
            Value::Null(dt) => {
                0u8.hash(state); // discriminant for Null
                dt.hash(state);
            }
            Value::Integer(v) => {
                // Hash as f64 bits so Integer(5) and Float(5.0) hash the same
                1u8.hash(state); // discriminant for numeric types
                (*v as f64).to_bits().hash(state);
            }
            Value::Float(v) => {
                // Hash as f64 bits so Integer(5) and Float(5.0) hash the same
                1u8.hash(state); // discriminant for numeric types
                v.to_bits().hash(state);
            }
            Value::Text(s) => {
                2u8.hash(state);
                s.hash(state);
            }
            Value::Boolean(b) => {
                3u8.hash(state);
                b.hash(state);
            }
            Value::Timestamp(t) => {
                4u8.hash(state);
                t.timestamp_nanos_opt().hash(state);
            }
            Value::Json(s) => {
                5u8.hash(state);
                s.hash(state);
            }
        }
    }
}

// Note: PartialOrd intentionally differs from Ord for SQL semantics
// - PartialOrd: SQL comparison (NULL returns None, cross-type numeric comparison)
// - Ord: BTreeMap ordering (NULLs first, type discriminant ordering)
#[allow(clippy::non_canonical_partial_ord_impl)]
impl PartialOrd for Value {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Use the original compare method for semantic correctness in SQL operations
        // This preserves NULL comparison semantics (returning None for NULL comparisons)
        // and proper cross-type numeric comparison (Integer vs Float)
        self.compare(other).ok()
    }
}

/// Total ordering implementation for Value
///
/// This is required for using Value as a key in BTreeMap/BTreeSet.
/// The ordering is defined as follows:
/// 1. NULLs are always ordered first (smallest)
/// 2. Numeric types (Integer, Float) are compared by numeric value (consistent with PartialEq)
/// 3. Other different data types are ordered by their type discriminant
/// 4. Same data types use their natural ordering
///
/// IMPORTANT: This ordering MUST be consistent with PartialEq. Since Integer(5) == Float(5.0)
/// per PartialEq, we must ensure Integer(5).cmp(&Float(5.0)) == Ordering::Equal.
/// Violating this contract causes BTreeMap corruption.
///
/// Note: This differs from SQL NULL semantics where NULL comparisons
/// return UNKNOWN. This ordering is only for internal index structure.
impl Ord for Value {
    fn cmp(&self, other: &Self) -> Ordering {
        // Handle NULL comparisons - NULLs are ordered first
        match (self.is_null(), other.is_null()) {
            (true, true) => return Ordering::Equal,
            (true, false) => return Ordering::Less,
            (false, true) => return Ordering::Greater,
            (false, false) => {} // Continue to value comparison
        }

        // Cross-type numeric comparison: Integer vs Float
        // This MUST be consistent with PartialEq where Integer(5) == Float(5.0)
        match (self, other) {
            (Value::Integer(i), Value::Float(f)) => {
                let i_as_f64 = *i as f64;
                // Handle NaN: NaN is ordered last
                if f.is_nan() {
                    return Ordering::Less; // Any number < NaN
                }
                return i_as_f64.partial_cmp(f).unwrap_or(Ordering::Equal);
            }
            (Value::Float(f), Value::Integer(i)) => {
                let i_as_f64 = *i as f64;
                // Handle NaN: NaN is ordered last
                if f.is_nan() {
                    return Ordering::Greater; // NaN > any number
                }
                return f.partial_cmp(&i_as_f64).unwrap_or(Ordering::Equal);
            }
            _ => {} // Continue to same-type comparison
        }

        // Helper function to get type discriminant for ordering
        fn type_discriminant(v: &Value) -> u8 {
            match v {
                Value::Null(_) => 0,
                Value::Boolean(_) => 1,
                // Integer and Float share the same discriminant for ordering purposes
                // This ensures they sort together by numeric value
                Value::Integer(_) | Value::Float(_) => 2,
                Value::Text(_) => 3,
                Value::Timestamp(_) => 4,
                Value::Json(_) => 5,
            }
        }

        let self_disc = type_discriminant(self);
        let other_disc = type_discriminant(other);

        // Different types: order by type discriminant
        if self_disc != other_disc {
            return self_disc.cmp(&other_disc);
        }

        // Same type comparison
        match (self, other) {
            (Value::Integer(a), Value::Integer(b)) => a.cmp(b),
            (Value::Float(a), Value::Float(b)) => {
                // Handle NaN: NaN is ordered last
                match (a.is_nan(), b.is_nan()) {
                    (true, true) => Ordering::Equal,
                    (true, false) => Ordering::Greater,
                    (false, true) => Ordering::Less,
                    (false, false) => a.partial_cmp(b).unwrap_or(Ordering::Equal),
                }
            }
            (Value::Text(a), Value::Text(b)) => a.cmp(b),
            (Value::Boolean(a), Value::Boolean(b)) => a.cmp(b),
            (Value::Timestamp(a), Value::Timestamp(b)) => a.cmp(b),
            (Value::Json(a), Value::Json(b)) => a.cmp(b), // Lexicographic for JSON
            _ => Ordering::Equal,                         // Should not reach here
        }
    }
}

// =========================================================================
// From implementations for convenient construction
// =========================================================================

impl From<i64> for Value {
    fn from(v: i64) -> Self {
        Value::Integer(v)
    }
}

impl From<i32> for Value {
    fn from(v: i32) -> Self {
        Value::Integer(v as i64)
    }
}

impl From<i16> for Value {
    fn from(v: i16) -> Self {
        Value::Integer(v as i64)
    }
}

impl From<i8> for Value {
    fn from(v: i8) -> Self {
        Value::Integer(v as i64)
    }
}

impl From<u32> for Value {
    fn from(v: u32) -> Self {
        Value::Integer(v as i64)
    }
}

impl From<u16> for Value {
    fn from(v: u16) -> Self {
        Value::Integer(v as i64)
    }
}

impl From<u8> for Value {
    fn from(v: u8) -> Self {
        Value::Integer(v as i64)
    }
}

impl From<f64> for Value {
    fn from(v: f64) -> Self {
        Value::Float(v)
    }
}

impl From<f32> for Value {
    fn from(v: f32) -> Self {
        Value::Float(v as f64)
    }
}

impl From<String> for Value {
    fn from(v: String) -> Self {
        Value::Text(Arc::from(v.as_str()))
    }
}

impl From<&str> for Value {
    fn from(v: &str) -> Self {
        Value::Text(Arc::from(v))
    }
}

impl From<Arc<str>> for Value {
    fn from(v: Arc<str>) -> Self {
        Value::Text(v)
    }
}

impl From<bool> for Value {
    fn from(v: bool) -> Self {
        Value::Boolean(v)
    }
}

impl From<DateTime<Utc>> for Value {
    fn from(v: DateTime<Utc>) -> Self {
        Value::Timestamp(v)
    }
}

impl<T: Into<Value>> From<Option<T>> for Value {
    fn from(v: Option<T>) -> Self {
        match v {
            Some(val) => val.into(),
            None => Value::Null(DataType::Null),
        }
    }
}

// =========================================================================
// Helper functions
// =========================================================================

/// Parse a timestamp string with multiple format support
pub fn parse_timestamp(s: &str) -> Result<DateTime<Utc>> {
    let s = s.trim();

    // Try each timestamp format
    for format in TIMESTAMP_FORMATS {
        if let Ok(dt) = DateTime::parse_from_str(s, format) {
            return Ok(dt.with_timezone(&Utc));
        }
        // Try parsing as naive datetime and assume UTC
        if let Ok(ndt) = NaiveDateTime::parse_from_str(s, format) {
            return Ok(Utc.from_utc_datetime(&ndt));
        }
    }

    // Try date-only formats
    if let Ok(date) = NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        let datetime = date.and_hms_opt(0, 0, 0).unwrap();
        return Ok(Utc.from_utc_datetime(&datetime));
    }

    // Try time-only formats (use today's date)
    for format in TIME_FORMATS {
        if let Ok(time) = NaiveTime::parse_from_str(s, format) {
            let today = Utc::now().date_naive();
            let datetime = today.and_time(time);
            return Ok(Utc.from_utc_datetime(&datetime));
        }
    }

    Err(Error::parse(format!("invalid timestamp format: {}", s)))
}

/// Format a float value consistently
fn format_float(v: f64) -> String {
    if v.fract() == 0.0 && v.abs() < 1e15 {
        // Integer-like float, format without decimal
        format!("{:.0}", v)
    } else {
        // Use 'g' format: shortest representation
        let s = format!("{:?}", v);
        // Remove trailing zeros after decimal point
        if s.contains('.') && !s.contains('e') && !s.contains('E') {
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            s
        }
    }
}

/// Compare two floats with proper NaN handling
fn compare_floats(a: f64, b: f64) -> Ordering {
    // Handle NaN: treat as greater than all other values for consistency
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => a.partial_cmp(&b).unwrap_or(Ordering::Equal),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Datelike, Timelike};

    // =========================================================================
    // Constructor tests
    // =========================================================================

    #[test]
    fn test_constructors() {
        assert!(Value::null(DataType::Integer).is_null());
        assert_eq!(Value::integer(42).as_int64(), Some(42));
        assert_eq!(Value::float(3.5).as_float64(), Some(3.5));
        assert_eq!(Value::text("hello").as_str(), Some("hello"));
        assert_eq!(Value::boolean(true).as_boolean(), Some(true));
        assert!(Value::json(r#"{"key": "value"}"#).as_json().is_some());
    }

    #[test]
    fn test_from_implementations() {
        let v: Value = 42i64.into();
        assert_eq!(v.as_int64(), Some(42));

        let v: Value = 3.5f64.into();
        assert_eq!(v.as_float64(), Some(3.5));

        let v: Value = "hello".into();
        assert_eq!(v.as_str(), Some("hello"));

        let v: Value = true.into();
        assert_eq!(v.as_boolean(), Some(true));

        let v: Value = Option::<i64>::None.into();
        assert!(v.is_null());

        let v: Value = Some(42i64).into();
        assert_eq!(v.as_int64(), Some(42));
    }

    // =========================================================================
    // Type accessor tests
    // =========================================================================

    #[test]
    fn test_data_type() {
        assert_eq!(
            Value::null(DataType::Integer).data_type(),
            DataType::Integer
        );
        assert_eq!(Value::integer(42).data_type(), DataType::Integer);
        assert_eq!(Value::float(3.5).data_type(), DataType::Float);
        assert_eq!(Value::text("hello").data_type(), DataType::Text);
        assert_eq!(Value::boolean(true).data_type(), DataType::Boolean);
        assert_eq!(
            Value::Timestamp(Utc::now()).data_type(),
            DataType::Timestamp
        );
        assert_eq!(Value::json("{}").data_type(), DataType::Json);
    }

    // =========================================================================
    // AsXxx conversion tests
    // =========================================================================

    #[test]
    fn test_as_int64() {
        // Direct integer
        assert_eq!(Value::integer(42).as_int64(), Some(42));

        // Float to integer (truncates)
        assert_eq!(Value::float(3.7).as_int64(), Some(3));
        assert_eq!(Value::float(-3.7).as_int64(), Some(-3));

        // String to integer
        assert_eq!(Value::text("42").as_int64(), Some(42));
        assert_eq!(Value::text("-42").as_int64(), Some(-42));
        assert_eq!(Value::text("3.7").as_int64(), Some(3)); // Parse as float, convert

        // Boolean to integer
        assert_eq!(Value::boolean(true).as_int64(), Some(1));
        assert_eq!(Value::boolean(false).as_int64(), Some(0));

        // NULL returns None
        assert_eq!(Value::null(DataType::Integer).as_int64(), None);

        // Invalid string
        assert_eq!(Value::text("not a number").as_int64(), None);
    }

    #[test]
    fn test_as_float64() {
        // Direct float
        assert_eq!(Value::float(3.5).as_float64(), Some(3.5));

        // Integer to float
        assert_eq!(Value::integer(42).as_float64(), Some(42.0));

        // String to float
        assert_eq!(Value::text("3.5").as_float64(), Some(3.5));

        // Boolean to float
        assert_eq!(Value::boolean(true).as_float64(), Some(1.0));
        assert_eq!(Value::boolean(false).as_float64(), Some(0.0));

        // NULL returns None
        assert_eq!(Value::null(DataType::Float).as_float64(), None);
    }

    #[test]
    fn test_as_boolean() {
        // Direct boolean
        assert_eq!(Value::boolean(true).as_boolean(), Some(true));
        assert_eq!(Value::boolean(false).as_boolean(), Some(false));

        // Integer to boolean
        assert_eq!(Value::integer(1).as_boolean(), Some(true));
        assert_eq!(Value::integer(0).as_boolean(), Some(false));
        assert_eq!(Value::integer(-1).as_boolean(), Some(true));

        // Float to boolean
        assert_eq!(Value::float(1.0).as_boolean(), Some(true));
        assert_eq!(Value::float(0.0).as_boolean(), Some(false));

        // String to boolean (various string values)
        assert_eq!(Value::text("true").as_boolean(), Some(true));
        assert_eq!(Value::text("TRUE").as_boolean(), Some(true));
        assert_eq!(Value::text("t").as_boolean(), Some(true));
        assert_eq!(Value::text("yes").as_boolean(), Some(true));
        assert_eq!(Value::text("y").as_boolean(), Some(true));
        assert_eq!(Value::text("1").as_boolean(), Some(true));
        assert_eq!(Value::text("false").as_boolean(), Some(false));
        assert_eq!(Value::text("FALSE").as_boolean(), Some(false));
        assert_eq!(Value::text("f").as_boolean(), Some(false));
        assert_eq!(Value::text("no").as_boolean(), Some(false));
        assert_eq!(Value::text("n").as_boolean(), Some(false));
        assert_eq!(Value::text("0").as_boolean(), Some(false));
        assert_eq!(Value::text("").as_boolean(), Some(false));

        // Numeric strings
        assert_eq!(Value::text("42").as_boolean(), Some(true));
        assert_eq!(Value::text("0.0").as_boolean(), Some(false));
    }

    #[test]
    fn test_as_string() {
        // Direct string
        assert_eq!(Value::text("hello").as_string(), Some("hello".to_string()));

        // Integer to string
        assert_eq!(Value::integer(42).as_string(), Some("42".to_string()));

        // Float to string
        assert_eq!(Value::float(3.5).as_string(), Some("3.5".to_string()));

        // Boolean to string
        assert_eq!(Value::boolean(true).as_string(), Some("true".to_string()));
        assert_eq!(Value::boolean(false).as_string(), Some("false".to_string()));

        // NULL returns None
        assert_eq!(Value::null(DataType::Text).as_string(), None);
    }

    // =========================================================================
    // Equality tests
    // =========================================================================

    #[test]
    fn test_equality() {
        // Same type equality
        assert_eq!(Value::integer(42), Value::integer(42));
        assert_ne!(Value::integer(42), Value::integer(43));

        assert_eq!(Value::float(3.5), Value::float(3.5));
        assert_ne!(Value::float(3.5), Value::float(3.15));

        assert_eq!(Value::text("hello"), Value::text("hello"));
        assert_ne!(Value::text("hello"), Value::text("world"));

        assert_eq!(Value::boolean(true), Value::boolean(true));
        assert_ne!(Value::boolean(true), Value::boolean(false));

        // NULL equality
        assert_eq!(Value::null(DataType::Integer), Value::null(DataType::Float));
        assert_ne!(Value::null(DataType::Integer), Value::integer(0));

        // Cross-type numeric comparison: Integer and Float with same value ARE equal
        // This is important for queries like WHERE id = 5.0 or WHERE price = 100
        assert_eq!(Value::integer(1), Value::float(1.0));
        assert_eq!(Value::integer(5), Value::float(5.0));
        assert_ne!(Value::integer(1), Value::float(1.5)); // Different values are not equal

        // Different non-numeric types are not equal
        assert_ne!(Value::text("1"), Value::integer(1));
    }

    #[test]
    fn test_float_nan_equality() {
        // NaN handling: NaN == NaN in our implementation (for consistency)
        let nan = Value::float(f64::NAN);
        assert_eq!(nan, nan.clone());
    }

    // =========================================================================
    // Comparison tests
    // =========================================================================

    #[test]
    fn test_compare_integers() {
        assert_eq!(
            Value::integer(1).compare(&Value::integer(2)).unwrap(),
            Ordering::Less
        );
        assert_eq!(
            Value::integer(2).compare(&Value::integer(2)).unwrap(),
            Ordering::Equal
        );
        assert_eq!(
            Value::integer(3).compare(&Value::integer(2)).unwrap(),
            Ordering::Greater
        );
    }

    #[test]
    fn test_compare_floats() {
        assert_eq!(
            Value::float(1.0).compare(&Value::float(2.0)).unwrap(),
            Ordering::Less
        );
        assert_eq!(
            Value::float(2.0).compare(&Value::float(2.0)).unwrap(),
            Ordering::Equal
        );
        assert_eq!(
            Value::float(3.0).compare(&Value::float(2.0)).unwrap(),
            Ordering::Greater
        );
    }

    #[test]
    fn test_compare_cross_type_numeric() {
        // Integer vs Float comparison
        assert_eq!(
            Value::integer(1).compare(&Value::float(2.0)).unwrap(),
            Ordering::Less
        );
        assert_eq!(
            Value::integer(2).compare(&Value::float(2.0)).unwrap(),
            Ordering::Equal
        );
        assert_eq!(
            Value::float(3.0).compare(&Value::integer(2)).unwrap(),
            Ordering::Greater
        );
    }

    #[test]
    fn test_compare_strings() {
        assert_eq!(
            Value::text("a").compare(&Value::text("b")).unwrap(),
            Ordering::Less
        );
        assert_eq!(
            Value::text("b").compare(&Value::text("b")).unwrap(),
            Ordering::Equal
        );
        assert_eq!(
            Value::text("c").compare(&Value::text("b")).unwrap(),
            Ordering::Greater
        );
    }

    #[test]
    fn test_compare_null() {
        // NULL comparisons
        assert_eq!(
            Value::null(DataType::Integer)
                .compare(&Value::null(DataType::Float))
                .unwrap(),
            Ordering::Equal
        );

        // NULL vs non-NULL should error
        assert!(Value::null(DataType::Integer)
            .compare(&Value::integer(0))
            .is_err());
        assert!(Value::integer(0)
            .compare(&Value::null(DataType::Integer))
            .is_err());
    }

    #[test]
    fn test_compare_json_error() {
        // JSON comparison only allows equality
        let j1 = Value::json(r#"{"a": 1}"#);
        let j2 = Value::json(r#"{"b": 2}"#);
        assert!(j1.compare(&j2).is_err());

        // Same JSON values are equal
        let j3 = Value::json(r#"{"a": 1}"#);
        assert_eq!(j1.compare(&j3).unwrap(), Ordering::Equal);
    }

    // =========================================================================
    // Timestamp parsing tests
    // =========================================================================

    #[test]
    fn test_parse_timestamp() {
        // RFC3339
        let ts = parse_timestamp("2024-01-15T10:30:00Z").unwrap();
        assert_eq!(ts.year(), 2024);
        assert_eq!(ts.month(), 1);
        assert_eq!(ts.day(), 15);
        assert_eq!(ts.hour(), 10);
        assert_eq!(ts.minute(), 30);

        // SQL format
        let ts = parse_timestamp("2024-01-15 10:30:00").unwrap();
        assert_eq!(ts.year(), 2024);

        // Date only
        let ts = parse_timestamp("2024-01-15").unwrap();
        assert_eq!(ts.year(), 2024);
        assert_eq!(ts.hour(), 0);

        // Invalid format
        assert!(parse_timestamp("not a date").is_err());
    }

    // =========================================================================
    // Display tests
    // =========================================================================

    #[test]
    fn test_display() {
        assert_eq!(Value::null(DataType::Integer).to_string(), "NULL");
        assert_eq!(Value::integer(42).to_string(), "42");
        assert_eq!(Value::float(3.5).to_string(), "3.5");
        assert_eq!(Value::text("hello").to_string(), "hello");
        assert_eq!(Value::boolean(true).to_string(), "true");
        assert_eq!(Value::boolean(false).to_string(), "false");
    }

    // =========================================================================
    // Hash tests
    // =========================================================================

    #[test]
    fn test_hash() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(Value::integer(42));
        set.insert(Value::integer(42)); // Duplicate
        set.insert(Value::integer(43));

        assert_eq!(set.len(), 2);
        assert!(set.contains(&Value::integer(42)));
        assert!(set.contains(&Value::integer(43)));
    }
}
