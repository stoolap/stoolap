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

//! Conversion and Collation Functions
//!
//! This module provides type conversion and string collation functions:
//!
//! - [`CastFunction`] - CAST(value AS type) - Convert value to another type
//! - [`CollateFunction`] - COLLATE(string, collation) - Apply collation to string

use std::sync::Arc;

use crate::core::{Error, Result, Value};
use crate::functions::{
    FunctionDataType, FunctionInfo, FunctionSignature, FunctionType, ScalarFunction,
};
use crate::validate_arg_count;

/// CAST function for type conversion
///
/// Converts a value from one type to another.
///
/// # Examples
/// - `CAST(123 AS TEXT)` → '123'
/// - `CAST('456' AS INTEGER)` → 456
/// - `CAST(1 AS BOOLEAN)` → true
#[derive(Default)]
pub struct CastFunction;

impl ScalarFunction for CastFunction {
    fn name(&self) -> &str {
        "CAST"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "CAST",
            FunctionType::Scalar,
            "Converts a value from one data type to another",
            FunctionSignature::new(
                FunctionDataType::Any,
                vec![FunctionDataType::Any, FunctionDataType::String],
                2,
                2,
            ),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "CAST", 2);

        let value = &args[0];
        let target_type = match &args[1] {
            Value::Text(s) => s.to_uppercase(),
            _ => {
                return Err(Error::invalid_argument(
                    "Second argument to CAST must be a string type name",
                ))
            }
        };

        // Handle NULL values
        if value.is_null() {
            return match target_type.as_str() {
                "STRING" | "TEXT" | "VARCHAR" | "CHAR" => Ok(Value::Text(Arc::from(""))),
                "INT" | "INTEGER" => Ok(Value::Integer(0)),
                "FLOAT" | "REAL" | "DOUBLE" => Ok(Value::Float(0.0)),
                "BOOLEAN" | "BOOL" => Ok(Value::Boolean(false)),
                _ => Ok(Value::null_unknown()),
            };
        }

        // Convert based on target type
        match target_type.as_str() {
            "INT" | "INTEGER" => cast_to_integer(value),
            "FLOAT" | "REAL" | "DOUBLE" => cast_to_float(value),
            "STRING" | "TEXT" | "VARCHAR" | "CHAR" => cast_to_string(value),
            "BOOLEAN" | "BOOL" => cast_to_boolean(value),
            "TIMESTAMP" | "DATETIME" | "DATE" | "TIME" => cast_to_timestamp(value),
            "JSON" => cast_to_json(value),
            _ => Err(Error::invalid_argument(format!(
                "Unsupported cast target type: {}",
                target_type
            ))),
        }
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(CastFunction)
    }
}

/// Cast a value to INTEGER
fn cast_to_integer(value: &Value) -> Result<Value> {
    match value {
        Value::Integer(i) => Ok(Value::Integer(*i)),
        Value::Float(f) => Ok(Value::Integer(*f as i64)),
        Value::Boolean(b) => Ok(Value::Integer(if *b { 1 } else { 0 })),
        Value::Text(s) => {
            if s.is_empty() {
                return Ok(Value::Integer(0));
            }
            // Try to parse as integer first
            if let Ok(i) = s.parse::<i64>() {
                return Ok(Value::Integer(i));
            }
            // If that fails, try as float and truncate
            if let Ok(f) = s.parse::<f64>() {
                return Ok(Value::Integer(f as i64));
            }
            // Return 0 for unparseable strings
            Ok(Value::Integer(0))
        }
        Value::Timestamp(t) => Ok(Value::Integer(t.timestamp())),
        Value::Json(_) => Ok(Value::Integer(0)),
        Value::Null(_) => Ok(Value::Integer(0)),
    }
}

/// Cast a value to FLOAT
fn cast_to_float(value: &Value) -> Result<Value> {
    match value {
        Value::Integer(i) => Ok(Value::Float(*i as f64)),
        Value::Float(f) => Ok(Value::Float(*f)),
        Value::Boolean(b) => Ok(Value::Float(if *b { 1.0 } else { 0.0 })),
        Value::Text(s) => {
            if s.is_empty() {
                return Ok(Value::Float(0.0));
            }
            match s.parse::<f64>() {
                Ok(f) => Ok(Value::Float(f)),
                Err(_) => Err(Error::invalid_argument(format!(
                    "Cannot convert '{}' to FLOAT",
                    s
                ))),
            }
        }
        Value::Timestamp(t) => Ok(Value::Float(t.timestamp() as f64)),
        Value::Json(_) => Err(Error::invalid_argument("Cannot convert JSON to FLOAT")),
        Value::Null(_) => Ok(Value::Float(0.0)),
    }
}

/// Cast a value to STRING/TEXT
fn cast_to_string(value: &Value) -> Result<Value> {
    match value {
        Value::Text(s) => Ok(Value::Text(s.clone())),
        Value::Integer(i) => Ok(Value::Text(Arc::from(i.to_string().as_str()))),
        Value::Float(f) => {
            // Format with up to 6 decimal places
            Ok(Value::Text(Arc::from(format!("{:.6}", f).as_str())))
        }
        Value::Boolean(b) => Ok(Value::Text(Arc::from(b.to_string().as_str()))),
        Value::Timestamp(t) => Ok(Value::Text(Arc::from(t.to_rfc3339().as_str()))),
        Value::Json(j) => Ok(Value::Text(j.clone())),
        Value::Null(_) => Ok(Value::Text(Arc::from(""))),
    }
}

/// Cast a value to BOOLEAN
fn cast_to_boolean(value: &Value) -> Result<Value> {
    match value {
        Value::Boolean(b) => Ok(Value::Boolean(*b)),
        Value::Integer(i) => Ok(Value::Boolean(*i != 0)),
        Value::Float(f) => Ok(Value::Boolean(*f != 0.0)),
        Value::Text(s) => {
            let lower = s.to_lowercase();
            let is_true = lower == "true"
                || lower == "yes"
                || lower == "1"
                || (!lower.is_empty() && lower != "0" && lower != "false" && lower != "no");
            Ok(Value::Boolean(is_true))
        }
        Value::Timestamp(_) => Err(Error::invalid_argument(
            "Cannot convert TIMESTAMP to BOOLEAN",
        )),
        Value::Json(_) => Err(Error::invalid_argument("Cannot convert JSON to BOOLEAN")),
        Value::Null(_) => Ok(Value::Boolean(false)),
    }
}

/// Cast a value to TIMESTAMP
fn cast_to_timestamp(value: &Value) -> Result<Value> {
    match value {
        Value::Timestamp(t) => Ok(Value::Timestamp(*t)),
        Value::Text(s) => {
            // Try to parse the timestamp using the core parse_timestamp function
            match crate::core::parse_timestamp(s) {
                Ok(t) => Ok(Value::Timestamp(t)),
                Err(_) => Err(Error::invalid_argument(format!(
                    "Cannot parse '{}' as TIMESTAMP",
                    s
                ))),
            }
        }
        Value::Integer(i) => {
            // Interpret as Unix timestamp
            use chrono::{TimeZone, Utc};
            match Utc.timestamp_opt(*i, 0) {
                chrono::LocalResult::Single(t) => Ok(Value::Timestamp(t)),
                _ => Err(Error::invalid_argument(format!(
                    "Invalid Unix timestamp: {}",
                    i
                ))),
            }
        }
        _ => Err(Error::invalid_argument(format!(
            "Cannot convert {:?} to TIMESTAMP",
            value.data_type()
        ))),
    }
}

/// Cast a value to JSON
fn cast_to_json(value: &Value) -> Result<Value> {
    match value {
        Value::Json(j) => Ok(Value::Json(j.clone())),
        Value::Text(s) => Ok(Value::Json(s.clone())),
        Value::Integer(i) => Ok(Value::Json(Arc::from(i.to_string().as_str()))),
        Value::Float(f) => Ok(Value::Json(Arc::from(f.to_string().as_str()))),
        Value::Boolean(b) => Ok(Value::Json(Arc::from(b.to_string().as_str()))),
        Value::Null(_) => Ok(Value::Json(Arc::from("null"))),
        Value::Timestamp(t) => Ok(Value::Json(Arc::from(
            format!("\"{}\"", t.to_rfc3339()).as_str(),
        ))),
    }
}

/// COLLATE function for string collation
///
/// Applies a collation to a string value for sorting and comparison.
///
/// # Supported Collations
/// - BINARY - Binary comparison (no change)
/// - NOCASE, CASE_INSENSITIVE - Case-insensitive comparison (converts to lowercase)
/// - NOACCENT, ACCENT_INSENSITIVE - Remove accents for comparison
/// - NUMERIC - For numeric-aware string comparison
#[derive(Default)]
pub struct CollateFunction;

impl ScalarFunction for CollateFunction {
    fn name(&self) -> &str {
        "COLLATE"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "COLLATE",
            FunctionType::Scalar,
            "Applies a collation to a string value for sorting and comparison",
            FunctionSignature::new(
                FunctionDataType::String,
                vec![FunctionDataType::Any, FunctionDataType::String],
                2,
                2,
            ),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "COLLATE", 2);

        // Handle NULL input
        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }

        // Convert first argument to string
        let s = match &args[0] {
            Value::Text(s) => s.to_string(),
            Value::Integer(i) => i.to_string(),
            Value::Float(f) => f.to_string(),
            Value::Boolean(b) => b.to_string(),
            Value::Timestamp(t) => t.to_rfc3339(),
            Value::Json(j) => j.to_string(),
            Value::Null(_) => return Ok(Value::null_unknown()),
        };

        // Get collation name
        let collation = match &args[1] {
            Value::Text(c) => c.to_uppercase(),
            _ => {
                return Err(Error::invalid_argument(
                    "COLLATE requires a string as the second argument",
                ))
            }
        };

        // Apply collation
        let result = apply_collation(&s, &collation)?;
        Ok(Value::Text(Arc::from(result.as_str())))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(CollateFunction)
    }
}

/// Apply a collation transformation to a string
fn apply_collation(s: &str, collation: &str) -> Result<String> {
    match collation {
        "BINARY" => Ok(s.to_string()),
        "NOCASE" | "CASE_INSENSITIVE" => Ok(s.to_lowercase()),
        "NOACCENT" | "ACCENT_INSENSITIVE" => Ok(remove_accents(s)),
        "NUMERIC" => Ok(s.to_string()), // No transformation, comparison handles it
        _ => Err(Error::invalid_argument(format!(
            "Unsupported collation: {}",
            collation
        ))),
    }
}

/// Remove accents from characters in a string
fn remove_accents(s: &str) -> String {
    s.chars()
        .filter_map(|c| {
            Some(match c {
                // Latin letters with accents -> base letter
                'À'..='Å' => 'A',
                'à'..='å' => 'a',
                'È'..='Ë' => 'E',
                'è'..='ë' => 'e',
                'Ì'..='Ï' => 'I',
                'ì'..='ï' => 'i',
                'Ò'..='Ö' => 'O',
                'ò'..='ö' => 'o',
                'Ù'..='Ü' => 'U',
                'ù'..='ü' => 'u',
                'Ç' => 'C',
                'ç' => 'c',
                'Ñ' => 'N',
                'ñ' => 'n',
                'Ÿ' => 'Y',
                'ÿ' => 'y',
                // Keep diacritical marks that are combining characters
                _ if c.is_ascii() || !is_combining_mark(c) => c,
                // Remove combining marks
                _ => return None,
            })
        })
        .collect()
}

/// Check if a character is a combining diacritical mark
fn is_combining_mark(c: char) -> bool {
    // Unicode combining diacritical marks range: U+0300 to U+036F
    matches!(c, '\u{0300}'..='\u{036F}')
}

#[cfg(test)]
mod tests {
    use super::*;

    // CAST tests
    #[test]
    fn test_cast_to_integer() {
        let cast = CastFunction;

        // Integer passthrough
        assert_eq!(
            cast.evaluate(&[Value::Integer(42), Value::text("INTEGER")])
                .unwrap(),
            Value::Integer(42)
        );

        // Float to integer
        assert_eq!(
            cast.evaluate(&[Value::Float(3.7), Value::text("INT")])
                .unwrap(),
            Value::Integer(3)
        );

        // String to integer
        assert_eq!(
            cast.evaluate(&[Value::text("123"), Value::text("INTEGER")])
                .unwrap(),
            Value::Integer(123)
        );

        // Boolean to integer
        assert_eq!(
            cast.evaluate(&[Value::Boolean(true), Value::text("INT")])
                .unwrap(),
            Value::Integer(1)
        );
        assert_eq!(
            cast.evaluate(&[Value::Boolean(false), Value::text("INT")])
                .unwrap(),
            Value::Integer(0)
        );

        // Empty string to integer
        assert_eq!(
            cast.evaluate(&[Value::text(""), Value::text("INTEGER")])
                .unwrap(),
            Value::Integer(0)
        );
    }

    #[test]
    fn test_cast_to_float() {
        let cast = CastFunction;

        // Integer to float
        assert_eq!(
            cast.evaluate(&[Value::Integer(42), Value::text("FLOAT")])
                .unwrap(),
            Value::Float(42.0)
        );

        // Float passthrough
        assert_eq!(
            cast.evaluate(&[Value::Float(3.5), Value::text("REAL")])
                .unwrap(),
            Value::Float(3.5)
        );

        // String to float
        assert_eq!(
            cast.evaluate(&[Value::text("2.5"), Value::text("DOUBLE")])
                .unwrap(),
            Value::Float(2.5)
        );
    }

    #[test]
    fn test_cast_to_string() {
        let cast = CastFunction;

        // Integer to string
        assert_eq!(
            cast.evaluate(&[Value::Integer(42), Value::text("TEXT")])
                .unwrap(),
            Value::text("42")
        );

        // Boolean to string
        assert_eq!(
            cast.evaluate(&[Value::Boolean(true), Value::text("STRING")])
                .unwrap(),
            Value::text("true")
        );

        // String passthrough
        assert_eq!(
            cast.evaluate(&[Value::text("hello"), Value::text("VARCHAR")])
                .unwrap(),
            Value::text("hello")
        );
    }

    #[test]
    fn test_cast_to_boolean() {
        let cast = CastFunction;

        // Integer to boolean
        assert_eq!(
            cast.evaluate(&[Value::Integer(1), Value::text("BOOL")])
                .unwrap(),
            Value::Boolean(true)
        );
        assert_eq!(
            cast.evaluate(&[Value::Integer(0), Value::text("BOOLEAN")])
                .unwrap(),
            Value::Boolean(false)
        );

        // String to boolean
        assert_eq!(
            cast.evaluate(&[Value::text("true"), Value::text("BOOL")])
                .unwrap(),
            Value::Boolean(true)
        );
        assert_eq!(
            cast.evaluate(&[Value::text("false"), Value::text("BOOL")])
                .unwrap(),
            Value::Boolean(false)
        );
        assert_eq!(
            cast.evaluate(&[Value::text("yes"), Value::text("BOOL")])
                .unwrap(),
            Value::Boolean(true)
        );
    }

    #[test]
    fn test_cast_null_handling() {
        let cast = CastFunction;

        // NULL to integer
        assert_eq!(
            cast.evaluate(&[Value::null_unknown(), Value::text("INTEGER")])
                .unwrap(),
            Value::Integer(0)
        );

        // NULL to string
        assert_eq!(
            cast.evaluate(&[Value::null_unknown(), Value::text("TEXT")])
                .unwrap(),
            Value::text("")
        );
    }

    // COLLATE tests
    #[test]
    fn test_collate_binary() {
        let collate = CollateFunction;

        assert_eq!(
            collate
                .evaluate(&[Value::text("Hello"), Value::text("BINARY")])
                .unwrap(),
            Value::text("Hello")
        );
    }

    #[test]
    fn test_collate_nocase() {
        let collate = CollateFunction;

        assert_eq!(
            collate
                .evaluate(&[Value::text("HELLO"), Value::text("NOCASE")])
                .unwrap(),
            Value::text("hello")
        );

        assert_eq!(
            collate
                .evaluate(&[Value::text("Hello World"), Value::text("CASE_INSENSITIVE")])
                .unwrap(),
            Value::text("hello world")
        );
    }

    #[test]
    fn test_collate_noaccent() {
        let collate = CollateFunction;

        assert_eq!(
            collate
                .evaluate(&[Value::text("Café"), Value::text("NOACCENT")])
                .unwrap(),
            Value::text("Cafe")
        );

        assert_eq!(
            collate
                .evaluate(&[Value::text("Naïve"), Value::text("ACCENT_INSENSITIVE")])
                .unwrap(),
            Value::text("Naive")
        );
    }

    #[test]
    fn test_collate_null_handling() {
        let collate = CollateFunction;

        let result = collate
            .evaluate(&[Value::null_unknown(), Value::text("NOCASE")])
            .unwrap();
        assert!(result.is_null());
    }

    #[test]
    fn test_collate_unsupported() {
        let collate = CollateFunction;

        let result = collate.evaluate(&[Value::text("test"), Value::text("INVALID")]);
        assert!(result.is_err());
    }

    #[test]
    fn test_remove_accents() {
        assert_eq!(remove_accents("Café"), "Cafe");
        assert_eq!(remove_accents("Naïve"), "Naive");
        assert_eq!(remove_accents("Résumé"), "Resume");
        assert_eq!(remove_accents("Élève"), "Eleve");
        assert_eq!(remove_accents("Über"), "Uber");
        assert_eq!(remove_accents("Español"), "Espanol");
    }
}
