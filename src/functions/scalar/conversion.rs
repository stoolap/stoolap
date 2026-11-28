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

use crate::common::SmartString;
use crate::core::{DataType, Error, Result, Value};
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
        let Value::Text(name) = &args[1] else {
            return Err(Error::invalid_argument(
                "Second argument to CAST must be a string type name",
            ));
        };
        let target_type: DataType = name.parse().map_err(|_| {
            Error::invalid_argument(format!("Unsupported cast target type: {name}"))
        })?;
        args[0].cast_to(target_type)
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(CastFunction)
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

        // The first argument as text
        let s = args[0].as_string().unwrap_or_default();

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
        Ok(Value::Text(SmartString::from_string(result)))
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

    #[test]
    fn the_cast_function_answers_as_a_cast() {
        let values = [
            Value::Integer(42),
            Value::Float(3.7),
            Value::Float(1.5),
            Value::text("123"),
            Value::text("abc"),
            Value::text("3.14"),
            Value::text(""),
            Value::text("yes"),
            Value::text("junk"),
            Value::text("9223372036854775808"),
            Value::Float(1e30),
            Value::Boolean(true),
            Value::null_unknown(),
        ];
        for value in values {
            for name in [
                "INTEGER", "INT", "FLOAT", "REAL", "TEXT", "VARCHAR", "BOOLEAN", "JSON",
            ] {
                let target: DataType = name.parse().unwrap();
                let got = CastFunction.evaluate(&[value.clone(), Value::text(name)]);
                let want = value.cast_to(target);
                assert_eq!(
                    got.as_ref().map_err(|e| e.to_string()),
                    want.as_ref().map_err(|e| e.to_string()),
                    "CAST({value:?} AS {name})"
                );
            }
        }
        assert!(CastFunction
            .evaluate(&[Value::Integer(1), Value::text("NOPE")])
            .is_err());
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
