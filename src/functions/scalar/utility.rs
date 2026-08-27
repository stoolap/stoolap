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

//! Utility scalar functions

use chrono::Utc;

use crate::core::{DataType, Error, Result, Value};
use crate::functions::{
    FunctionDataType, FunctionInfo, FunctionSignature, FunctionType, ScalarFunction,
};
use crate::validate_arg_count;

// ============================================================================
// COALESCE
// ============================================================================

/// COALESCE function - returns the first non-null value in a list
#[derive(Default)]
pub struct CoalesceFunction;

impl ScalarFunction for CoalesceFunction {
    fn name(&self) -> &str {
        "COALESCE"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "COALESCE",
            FunctionType::Scalar,
            "Returns the first non-null value in a list",
            FunctionSignature::variadic(FunctionDataType::Any, FunctionDataType::Any),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        if args.is_empty() {
            return Err(Error::invalid_argument(
                "COALESCE requires at least 1 argument",
            ));
        }

        // Return the first non-null value
        for arg in args {
            if !arg.is_null() {
                return Ok(arg.clone());
            }
        }

        // If all arguments are null, return null
        Ok(Value::null_unknown())
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(CoalesceFunction)
    }
}

// ============================================================================
// NOW
// ============================================================================

/// NOW function - returns the current date and time
#[derive(Default)]
pub struct NowFunction;

impl ScalarFunction for NowFunction {
    fn name(&self) -> &str {
        "NOW"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "NOW",
            FunctionType::Scalar,
            "Returns the current date and time",
            FunctionSignature::new(FunctionDataType::DateTime, vec![], 0, 0),
        )
        .non_deterministic()
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        if !args.is_empty() {
            return Err(Error::invalid_argument(format!(
                "NOW takes no arguments, got {}",
                args.len()
            )));
        }

        Ok(Value::Timestamp(Utc::now()))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(NowFunction)
    }
}

// ============================================================================
// NULLIF
// ============================================================================

/// NULLIF function - returns NULL if the two arguments are equal
#[derive(Default)]
pub struct NullIfFunction;

impl ScalarFunction for NullIfFunction {
    fn name(&self) -> &str {
        "NULLIF"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "NULLIF",
            FunctionType::Scalar,
            "Returns NULL if the two arguments are equal, otherwise returns the first argument",
            FunctionSignature::new(
                FunctionDataType::Any,
                vec![FunctionDataType::Any, FunctionDataType::Any],
                2,
                2,
            ),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "NULLIF", 2);

        // If both are equal, return NULL
        if args[0] == args[1] {
            return Ok(Value::null_unknown());
        }

        // Otherwise return the first argument
        Ok(args[0].clone())
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(NullIfFunction)
    }
}

// ============================================================================
// IFNULL / NVL
// ============================================================================

/// IFNULL function - returns the first argument if it is not NULL, otherwise returns the second
#[derive(Default)]
pub struct IfNullFunction;

impl ScalarFunction for IfNullFunction {
    fn name(&self) -> &str {
        "IFNULL"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "IFNULL",
            FunctionType::Scalar,
            "Returns the first argument if it is not NULL, otherwise returns the second argument",
            FunctionSignature::new(
                FunctionDataType::Any,
                vec![FunctionDataType::Any, FunctionDataType::Any],
                2,
                2,
            ),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "IFNULL", 2);

        // If first argument is not NULL, return it
        if !args[0].is_null() {
            return Ok(args[0].clone());
        }

        // Otherwise return the second argument
        Ok(args[1].clone())
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(IfNullFunction)
    }
}

// ============================================================================
// GREATEST
// ============================================================================

/// GREATEST function - returns the greatest value from a list of values
#[derive(Default)]
pub struct GreatestFunction;

impl ScalarFunction for GreatestFunction {
    fn name(&self) -> &str {
        "GREATEST"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "GREATEST",
            FunctionType::Scalar,
            "Returns the greatest value from a list of values",
            FunctionSignature::variadic(FunctionDataType::Any, FunctionDataType::Any),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        if args.is_empty() {
            return Err(Error::invalid_argument(
                "GREATEST requires at least 1 argument",
            ));
        }

        // If any argument is NULL, return NULL (SQL standard behavior)
        if args.iter().any(|v| v.is_null()) {
            return Ok(Value::null_unknown());
        }

        // Find the greatest value
        let mut greatest = &args[0];
        for arg in args.iter().skip(1) {
            if arg > greatest {
                greatest = arg;
            }
        }

        Ok(greatest.clone())
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(GreatestFunction)
    }
}

// ============================================================================
// LEAST
// ============================================================================

/// LEAST function - returns the smallest value from a list of values
#[derive(Default)]
pub struct LeastFunction;

impl ScalarFunction for LeastFunction {
    fn name(&self) -> &str {
        "LEAST"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "LEAST",
            FunctionType::Scalar,
            "Returns the smallest value from a list of values",
            FunctionSignature::variadic(FunctionDataType::Any, FunctionDataType::Any),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        if args.is_empty() {
            return Err(Error::invalid_argument(
                "LEAST requires at least 1 argument",
            ));
        }

        // If any argument is NULL, return NULL (SQL standard behavior)
        if args.iter().any(|v| v.is_null()) {
            return Ok(Value::null_unknown());
        }

        // Find the smallest value
        let mut least = &args[0];
        for arg in args.iter().skip(1) {
            if arg < least {
                least = arg;
            }
        }

        Ok(least.clone())
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(LeastFunction)
    }
}

// ============================================================================
// IIF
// ============================================================================

/// IIF function - inline if (shorthand for CASE WHEN condition THEN true_value ELSE false_value END)
#[derive(Default)]
pub struct IifFunction;

impl ScalarFunction for IifFunction {
    fn name(&self) -> &str {
        "IIF"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "IIF",
            FunctionType::Scalar,
            "Returns true_value if condition is true, otherwise returns false_value",
            FunctionSignature::new(
                FunctionDataType::Any,
                vec![
                    FunctionDataType::Boolean,
                    FunctionDataType::Any,
                    FunctionDataType::Any,
                ],
                3,
                3,
            ),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "IIF", 3);

        let condition = &args[0];
        let true_value = &args[1];
        let false_value = &args[2];

        // Check if condition is truthy
        let is_true = match condition {
            Value::Boolean(b) => *b,
            Value::Integer(i) => *i != 0,
            Value::Float(f) => *f != 0.0,
            Value::Text(s) => !s.is_empty() && s.to_lowercase() != "false" && s.as_str() != "0",
            Value::Null(_) => false,
            _ => false,
        };

        if is_true {
            Ok(true_value.clone())
        } else {
            Ok(false_value.clone())
        }
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(IifFunction)
    }
}

// ============================================================================
// JSON_EXTRACT
// ============================================================================

/// JSON_EXTRACT function - extracts a value from a JSON string using a path
#[derive(Default)]
pub struct JsonExtractFunction;

impl ScalarFunction for JsonExtractFunction {
    fn name(&self) -> &str {
        "JSON_EXTRACT"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "JSON_EXTRACT",
            FunctionType::Scalar,
            "Extracts a value from JSON using a path expression",
            FunctionSignature::new(
                FunctionDataType::Any,
                vec![FunctionDataType::Any, FunctionDataType::String],
                2,
                2,
            ),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "JSON_EXTRACT", 2);

        // Handle NULL input
        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }

        // Get JSON string
        let json_str = match &args[0] {
            Value::Extension(data) if data.first() == Some(&(DataType::Json as u8)) => {
                std::str::from_utf8(&data[1..]).unwrap_or("").to_string()
            }
            Value::Text(s) => s.to_string(),
            _ => {
                return Err(Error::invalid_argument(
                    "JSON_EXTRACT first argument must be JSON or TEXT",
                ))
            }
        };

        // Get path
        let path = match &args[1] {
            Value::Text(s) => s.to_string(),
            _ => {
                return Err(Error::invalid_argument(
                    "JSON_EXTRACT second argument must be a path string",
                ))
            }
        };

        // Parse JSON
        let json_value: serde_json::Value = match serde_json::from_str(&json_str) {
            Ok(v) => v,
            Err(_) => return Ok(Value::null_unknown()),
        };

        // Extract value using path
        let result = extract_json_path(&json_value, &path);

        match result {
            Some(v) => json_to_value(v),
            None => Ok(Value::null_unknown()),
        }
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(JsonExtractFunction)
    }
}

/// Extract a value from JSON using a path like "$.name" or "$.user.email" or "$.items[0]"
fn extract_json_path<'a>(
    value: &'a serde_json::Value,
    path: &str,
) -> Option<&'a serde_json::Value> {
    // Remove leading "$." or "$" if present
    let path = path
        .strip_prefix("$.")
        .unwrap_or(path.strip_prefix("$").unwrap_or(path));

    if path.is_empty() {
        return Some(value);
    }

    let mut current = value;

    for part in path.split('.') {
        // Check for array index notation like "items[0]"
        if let Some(bracket_pos) = part.find('[') {
            // Find closing bracket - return None if malformed
            let close_bracket_pos = part.find(']')?;
            if close_bracket_pos <= bracket_pos + 1 {
                // Empty index like "items[]" - malformed
                return None;
            }

            let key = &part[..bracket_pos];
            let index_str = &part[bracket_pos + 1..close_bracket_pos];

            // Get the object field first
            if !key.is_empty() {
                current = current.get(key)?;
            }

            // Then get the array element
            let index: usize = index_str.parse().ok()?;
            current = current.get(index)?;
        } else {
            current = current.get(part)?;
        }
    }

    Some(current)
}

/// Convert a serde_json::Value to a stoolap Value
fn json_to_value(json: &serde_json::Value) -> Result<Value> {
    match json {
        serde_json::Value::Null => Ok(Value::null_unknown()),
        serde_json::Value::Bool(b) => Ok(Value::Boolean(*b)),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(Value::Integer(i))
            } else if let Some(f) = n.as_f64() {
                Ok(Value::Float(f))
            } else {
                Ok(Value::text(n.to_string()))
            }
        }
        serde_json::Value::String(s) => Ok(Value::text(s)),
        // For arrays and objects, return as JSON string
        _ => Ok(Value::json(json.to_string())),
    }
}

// ============================================================================
// JSON_ARRAY_LENGTH
// ============================================================================

/// JSON_ARRAY_LENGTH function - returns the length of a JSON array
#[derive(Default)]
pub struct JsonArrayLengthFunction;

impl ScalarFunction for JsonArrayLengthFunction {
    fn name(&self) -> &str {
        "JSON_ARRAY_LENGTH"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "JSON_ARRAY_LENGTH",
            FunctionType::Scalar,
            "Returns the number of elements in a JSON array",
            FunctionSignature::new(
                FunctionDataType::Integer,
                vec![FunctionDataType::Any],
                1,
                2, // Optional path argument
            ),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "JSON_ARRAY_LENGTH", 1, 2);

        // Handle NULL input
        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }

        // Get JSON string
        let json_str = match &args[0] {
            Value::Extension(data) if data.first() == Some(&(DataType::Json as u8)) => {
                std::str::from_utf8(&data[1..]).unwrap_or("").to_string()
            }
            Value::Text(s) => s.to_string(),
            _ => {
                return Err(Error::invalid_argument(
                    "JSON_ARRAY_LENGTH first argument must be JSON or TEXT",
                ))
            }
        };

        // Parse JSON
        let json_value: serde_json::Value = match serde_json::from_str(&json_str) {
            Ok(v) => v,
            Err(_) => return Ok(Value::null_unknown()),
        };

        // If path is provided, extract from that path first
        let target = if args.len() == 2 {
            if args[1].is_null() {
                return Ok(Value::null_unknown());
            }
            let path = match &args[1] {
                Value::Text(s) => s.to_string(),
                _ => {
                    return Err(Error::invalid_argument(
                        "JSON_ARRAY_LENGTH second argument must be a path string",
                    ))
                }
            };
            match extract_json_path(&json_value, &path) {
                Some(v) => v.clone(),
                None => return Ok(Value::null_unknown()),
            }
        } else {
            json_value
        };

        // Return length if it's an array, NULL otherwise
        match target {
            serde_json::Value::Array(arr) => Ok(Value::Integer(arr.len() as i64)),
            _ => Ok(Value::null_unknown()),
        }
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(JsonArrayLengthFunction)
    }
}

// ============================================================================
// JSON_ARRAY
// ============================================================================

/// JSON_ARRAY function - creates a JSON array from the provided values
/// JSON_ARRAY(1, 2, 3) returns '[1, 2, 3]'
/// JSON_ARRAY('a', 'b') returns '["a", "b"]'
#[derive(Default)]
pub struct JsonArrayFunction;

impl ScalarFunction for JsonArrayFunction {
    fn name(&self) -> &str {
        "JSON_ARRAY"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "JSON_ARRAY",
            FunctionType::Scalar,
            "Creates a JSON array from the provided values",
            FunctionSignature::new(
                FunctionDataType::Json,
                vec![], // Variadic - accepts any number of arguments
                0,
                255, // Arbitrary max
            ),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        // Convert all arguments to JSON values
        let json_values: Vec<serde_json::Value> = args.iter().map(value_to_json).collect();

        let json_array = serde_json::Value::Array(json_values);
        Ok(Value::json(json_array.to_string()))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(JsonArrayFunction)
    }
}

// ============================================================================
// JSON_OBJECT
// ============================================================================

/// JSON_OBJECT function - creates a JSON object from key-value pairs
/// JSON_OBJECT('name', 'Alice', 'age', 30) returns '{"name": "Alice", "age": 30}'
#[derive(Default)]
pub struct JsonObjectFunction;

impl ScalarFunction for JsonObjectFunction {
    fn name(&self) -> &str {
        "JSON_OBJECT"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "JSON_OBJECT",
            FunctionType::Scalar,
            "Creates a JSON object from key-value pairs",
            FunctionSignature::new(
                FunctionDataType::Json,
                vec![], // Variadic - accepts pairs of arguments
                0,
                255, // Arbitrary max
            ),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        // Must have even number of arguments (key-value pairs)
        if !args.len().is_multiple_of(2) {
            return Err(Error::invalid_argument(
                "JSON_OBJECT requires an even number of arguments (key-value pairs)",
            ));
        }

        let mut map = serde_json::Map::new();

        for i in (0..args.len()).step_by(2) {
            // Key must be a string
            let key = match &args[i] {
                Value::Text(s) => s.to_string(),
                Value::Null(_) => {
                    return Err(Error::invalid_argument("JSON_OBJECT key cannot be NULL"))
                }
                _ => args[i].to_string(),
            };

            // Convert value to JSON
            let value = value_to_json(&args[i + 1]);
            map.insert(key, value);
        }

        let json_object = serde_json::Value::Object(map);
        Ok(Value::json(json_object.to_string()))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(JsonObjectFunction)
    }
}

/// Helper function to convert a Value to serde_json::Value
fn value_to_json(v: &Value) -> serde_json::Value {
    match v {
        Value::Null(_) => serde_json::Value::Null,
        Value::Boolean(b) => serde_json::Value::Bool(*b),
        Value::Integer(i) => serde_json::Value::Number((*i).into()),
        Value::Float(f) => serde_json::Number::from_f64(*f)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        Value::Text(s) => serde_json::Value::String(s.to_string()),
        Value::Extension(data) if data.first() == Some(&(DataType::Json as u8)) => {
            // Parse the JSON string to get a proper JSON value
            let s = std::str::from_utf8(&data[1..]).unwrap_or("");
            serde_json::from_str(s).unwrap_or(serde_json::Value::String(s.to_string()))
        }
        Value::Timestamp(t) => serde_json::Value::String(t.to_rfc3339()),
        Value::Extension(_) => serde_json::Value::Null,
    }
}

// ============================================================================
// JSON_TYPE
// ============================================================================

/// JSON_TYPE function - returns the type of a JSON value
/// Supports both JSON_TYPE(json) and JSON_TYPE(json, path) forms
#[derive(Default)]
pub struct JsonTypeFunction;

impl ScalarFunction for JsonTypeFunction {
    fn name(&self) -> &str {
        "JSON_TYPE"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "JSON_TYPE",
            FunctionType::Scalar,
            "Returns the type of a JSON value (object, array, string, number, boolean, null). Optional second argument specifies a path.",
            FunctionSignature::new(FunctionDataType::String, vec![FunctionDataType::Any, FunctionDataType::String], 1, 2),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        if args.is_empty() || args.len() > 2 {
            return Err(Error::invalid_argument(
                "JSON_TYPE requires 1 or 2 arguments",
            ));
        }

        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }

        // Get JSON string
        let json_str = match &args[0] {
            Value::Extension(data) if data.first() == Some(&(DataType::Json as u8)) => {
                std::str::from_utf8(&data[1..]).unwrap_or("").to_string()
            }
            Value::Text(s) => s.to_string(),
            _ => {
                return Err(Error::invalid_argument(
                    "JSON_TYPE first argument must be JSON or TEXT",
                ))
            }
        };

        // Parse JSON
        let json_value: serde_json::Value = match serde_json::from_str(&json_str) {
            Ok(v) => v,
            Err(_) => return Ok(Value::null_unknown()),
        };

        // If path is provided, extract the value at that path first
        let target_value = if args.len() == 2 {
            if args[1].is_null() {
                return Ok(Value::null_unknown());
            }
            let path = match &args[1] {
                Value::Text(s) => s.to_string(),
                _ => {
                    return Err(Error::invalid_argument(
                        "JSON_TYPE second argument must be a path string",
                    ))
                }
            };
            match extract_json_path(&json_value, &path) {
                Some(v) => v.clone(),
                None => return Ok(Value::null_unknown()),
            }
        } else {
            json_value
        };

        let type_name = match target_value {
            serde_json::Value::Null => "null",
            serde_json::Value::Bool(_) => "boolean",
            serde_json::Value::Number(_) => "number",
            serde_json::Value::String(_) => "string",
            serde_json::Value::Array(_) => "array",
            serde_json::Value::Object(_) => "object",
        };

        Ok(Value::text(type_name))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(JsonTypeFunction)
    }
}

// ============================================================================
// JSON_TYPEOF (alias for JSON_TYPE - PostgreSQL style)
// ============================================================================

/// JSON_TYPEOF function - alias for JSON_TYPE (PostgreSQL compatibility)
#[derive(Default)]
pub struct JsonTypeOfFunction;

impl ScalarFunction for JsonTypeOfFunction {
    fn name(&self) -> &str {
        "JSON_TYPEOF"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "JSON_TYPEOF",
            FunctionType::Scalar,
            "Returns the type of a JSON value (PostgreSQL-style alias for JSON_TYPE). Optional second argument specifies a path.",
            FunctionSignature::new(FunctionDataType::String, vec![FunctionDataType::Any, FunctionDataType::String], 1, 2),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        // Delegate to JSON_TYPE implementation
        JsonTypeFunction.evaluate(args)
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(JsonTypeOfFunction)
    }
}

// ============================================================================
// JSON_VALID
// ============================================================================

/// JSON_VALID function - checks if a string is valid JSON
#[derive(Default)]
pub struct JsonValidFunction;

impl ScalarFunction for JsonValidFunction {
    fn name(&self) -> &str {
        "JSON_VALID"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "JSON_VALID",
            FunctionType::Scalar,
            "Returns 1 if the argument is valid JSON, 0 otherwise",
            FunctionSignature::new(FunctionDataType::Integer, vec![FunctionDataType::Any], 1, 1),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "JSON_VALID", 1);

        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }

        // Get string to validate — always parse, even for tagged JSON values
        let json_str = match &args[0] {
            Value::Extension(data) if data.first() == Some(&(DataType::Json as u8)) => {
                match std::str::from_utf8(&data[1..]) {
                    Ok(s) => s.to_string(),
                    Err(_) => return Ok(Value::Integer(0)),
                }
            }
            Value::Text(s) => s.to_string(),
            _ => return Ok(Value::Integer(0)), // Non-string types are not valid JSON strings
        };

        // Try to parse as JSON
        let is_valid = serde_json::from_str::<serde_json::Value>(&json_str).is_ok();
        Ok(Value::Integer(if is_valid { 1 } else { 0 }))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(JsonValidFunction)
    }
}

// ============================================================================
// JSON_KEYS
// ============================================================================

/// JSON_KEYS function - returns the keys of a JSON object as a JSON array
#[derive(Default)]
pub struct JsonKeysFunction;

impl ScalarFunction for JsonKeysFunction {
    fn name(&self) -> &str {
        "JSON_KEYS"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "JSON_KEYS",
            FunctionType::Scalar,
            "Returns the keys of a JSON object as a JSON array",
            FunctionSignature::new(FunctionDataType::Json, vec![FunctionDataType::Any], 1, 1),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "JSON_KEYS", 1);

        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }

        // Get JSON string
        let json_str = match &args[0] {
            Value::Extension(data) if data.first() == Some(&(DataType::Json as u8)) => {
                std::str::from_utf8(&data[1..]).unwrap_or("").to_string()
            }
            Value::Text(s) => s.to_string(),
            _ => {
                return Err(Error::invalid_argument(
                    "JSON_KEYS argument must be JSON or TEXT",
                ))
            }
        };

        // Parse JSON
        let json_value: serde_json::Value = match serde_json::from_str(&json_str) {
            Ok(v) => v,
            Err(_) => return Ok(Value::null_unknown()),
        };

        // Extract keys if it's an object
        match json_value {
            serde_json::Value::Object(map) => {
                let keys: Vec<serde_json::Value> = map
                    .keys()
                    .map(|k| serde_json::Value::String(k.clone()))
                    .collect();
                let keys_array = serde_json::Value::Array(keys);
                Ok(Value::json(keys_array.to_string()))
            }
            _ => Ok(Value::null_unknown()), // Not an object, return NULL
        }
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(JsonKeysFunction)
    }
}

// ============================================================================
// SLEEP
// ============================================================================

/// SLEEP function - pauses execution for a specified number of seconds
/// Returns 0 on success. Useful for testing and debugging.
#[derive(Default)]
pub struct SleepFunction;

impl ScalarFunction for SleepFunction {
    fn name(&self) -> &str {
        "SLEEP"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "SLEEP",
            FunctionType::Scalar,
            "Pauses execution for a specified number of seconds",
            FunctionSignature::new(FunctionDataType::Integer, vec![FunctionDataType::Any], 1, 1),
        )
        .non_deterministic()
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "SLEEP", 1);

        #[cfg(target_arch = "wasm32")]
        {
            let _ = &args[0];
            return Err(Error::internal("SLEEP is not supported in WASM mode"));
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            if args[0].is_null() {
                return Ok(Value::null_unknown());
            }

            // Get seconds to sleep (can be fractional)
            let seconds = match &args[0] {
                Value::Integer(i) => *i as f64,
                Value::Float(f) => *f,
                _ => return Err(Error::invalid_argument("SLEEP argument must be a number")),
            };

            if seconds < 0.0 {
                return Err(Error::invalid_argument("SLEEP duration cannot be negative"));
            }

            // Limit to reasonable duration (max 300 seconds = 5 minutes)
            let seconds = seconds.min(300.0);

            // Sleep for the specified duration
            std::thread::sleep(std::time::Duration::from_secs_f64(seconds));

            Ok(Value::Integer(0))
        }
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(SleepFunction)
    }
}

// ============================================================================
// TYPEOF
// ============================================================================

/// TYPEOF function - returns the data type name of a value
#[derive(Default)]
pub struct TypeOfFunction;

impl ScalarFunction for TypeOfFunction {
    fn name(&self) -> &str {
        "TYPEOF"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "TYPEOF",
            FunctionType::Scalar,
            "Returns the data type name of a value",
            FunctionSignature::new(FunctionDataType::String, vec![FunctionDataType::Any], 1, 1),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "TYPEOF", 1);

        let type_name = match &args[0] {
            Value::Null(_) => "NULL",
            Value::Integer(_) => "INTEGER",
            Value::Float(_) => "FLOAT",
            Value::Text(_) => "TEXT",
            Value::Boolean(_) => "BOOLEAN",
            Value::Timestamp(_) => "TIMESTAMP",
            Value::Extension(data) if data.first() == Some(&(DataType::Json as u8)) => "JSON",
            Value::Extension(data) if data.first() == Some(&(DataType::Vector as u8)) => "VECTOR",
            Value::Extension(_) => "EXTENSION",
        };

        Ok(Value::text(type_name))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(TypeOfFunction)
    }
}

// ============================================================================
// JSON Path Helpers
// ============================================================================

#[derive(Debug, Clone, PartialEq)]
enum JsonPathStep {
    Key(String),
    Index(usize),
}

fn parse_json_path_steps(path: &str) -> Option<Vec<JsonPathStep>> {
    let path = path
        .strip_prefix("$.")
        .unwrap_or(path.strip_prefix('$').unwrap_or(path));

    if path.is_empty() {
        return Some(Vec::new());
    }

    let mut steps = Vec::new();
    for part in path.split('.') {
        if part.is_empty() {
            continue;
        }
        let mut remaining = part;
        while let Some(bracket_open) = remaining.find('[') {
            let key = &remaining[..bracket_open];
            if !key.is_empty() {
                steps.push(JsonPathStep::Key(key.to_string()));
            }
            let bracket_close = remaining.find(']')?;
            if bracket_close <= bracket_open + 1 {
                return None;
            }
            let idx_str = &remaining[bracket_open + 1..bracket_close];
            let idx: usize = idx_str.parse().ok()?;
            steps.push(JsonPathStep::Index(idx));
            remaining = &remaining[bracket_close + 1..];
        }
        if !remaining.is_empty() {
            steps.push(JsonPathStep::Key(remaining.to_string()));
        }
    }
    Some(steps)
}

fn json_path_set(
    doc: &mut serde_json::Value,
    steps: &[JsonPathStep],
    val: serde_json::Value,
    insert_only: bool,
    replace_only: bool,
) {
    if steps.is_empty() {
        if !insert_only {
            *doc = val;
        }
        return;
    }

    match &steps[0] {
        JsonPathStep::Key(k) => {
            if steps.len() == 1 {
                if !doc.is_object() {
                    if replace_only {
                        return;
                    }
                    *doc = serde_json::Value::Object(serde_json::Map::new());
                }
                if let serde_json::Value::Object(map) = doc {
                    let exists = map.contains_key(k);
                    if insert_only && exists {
                        return;
                    }
                    if replace_only && !exists {
                        return;
                    }
                    map.insert(k.clone(), val);
                }
            } else {
                if !doc.is_object() {
                    if replace_only {
                        return;
                    }
                    *doc = serde_json::Value::Object(serde_json::Map::new());
                }
                if let serde_json::Value::Object(map) = doc {
                    if !map.contains_key(k) {
                        if replace_only {
                            return;
                        }
                        let next_is_idx = matches!(steps.get(1), Some(JsonPathStep::Index(_)));
                        let intermediate = if next_is_idx {
                            serde_json::Value::Array(Vec::new())
                        } else {
                            serde_json::Value::Object(serde_json::Map::new())
                        };
                        map.insert(k.clone(), intermediate);
                    }
                    if let Some(child) = map.get_mut(k) {
                        json_path_set(child, &steps[1..], val, insert_only, replace_only);
                    }
                }
            }
        }
        JsonPathStep::Index(idx) => {
            let idx = *idx;
            if steps.len() == 1 {
                if !doc.is_array() {
                    if replace_only {
                        return;
                    }
                    *doc = serde_json::Value::Array(Vec::new());
                }
                if let serde_json::Value::Array(arr) = doc {
                    let exists = idx < arr.len();
                    if insert_only && exists {
                        return;
                    }
                    if replace_only && !exists {
                        return;
                    }
                    if idx < arr.len() {
                        arr[idx] = val;
                    } else if !replace_only {
                        arr.push(val);
                    }
                }
            } else {
                if !doc.is_array() {
                    if replace_only {
                        return;
                    }
                    *doc = serde_json::Value::Array(Vec::new());
                }
                if let serde_json::Value::Array(arr) = doc {
                    if idx >= arr.len() {
                        if replace_only {
                            return;
                        }
                        let next_is_idx = matches!(steps.get(1), Some(JsonPathStep::Index(_)));
                        let intermediate = if next_is_idx {
                            serde_json::Value::Array(Vec::new())
                        } else {
                            serde_json::Value::Object(serde_json::Map::new())
                        };
                        arr.push(intermediate);
                    }
                    if let Some(child) = arr.get_mut(idx) {
                        json_path_set(child, &steps[1..], val, insert_only, replace_only);
                    }
                }
            }
        }
    }
}

fn json_path_remove(doc: &mut serde_json::Value, steps: &[JsonPathStep]) {
    if steps.is_empty() {
        return;
    }

    match &steps[0] {
        JsonPathStep::Key(k) => {
            if steps.len() == 1 {
                if let serde_json::Value::Object(map) = doc {
                    map.remove(k);
                }
            } else if let serde_json::Value::Object(map) = doc {
                if let Some(child) = map.get_mut(k) {
                    json_path_remove(child, &steps[1..]);
                }
            }
        }
        JsonPathStep::Index(idx) => {
            let idx = *idx;
            if steps.len() == 1 {
                if let serde_json::Value::Array(arr) = doc {
                    if idx < arr.len() {
                        arr.remove(idx);
                    }
                }
            } else if let serde_json::Value::Array(arr) = doc {
                if let Some(child) = arr.get_mut(idx) {
                    json_path_remove(child, &steps[1..]);
                }
            }
        }
    }
}

fn json_contains_check(target: &serde_json::Value, candidate: &serde_json::Value) -> bool {
    match (target, candidate) {
        (serde_json::Value::Object(target_map), serde_json::Value::Object(candidate_map)) => {
            for (k, cand_v) in candidate_map {
                match target_map.get(k) {
                    Some(target_v) => {
                        if !json_contains_check(target_v, cand_v) {
                            return false;
                        }
                    }
                    None => return false,
                }
            }
            true
        }
        (serde_json::Value::Array(target_arr), serde_json::Value::Array(candidate_arr)) => {
            for cand_elem in candidate_arr {
                if !target_arr.iter().any(|t| json_contains_check(t, cand_elem)) {
                    return false;
                }
            }
            true
        }
        (serde_json::Value::Array(target_arr), scalar_cand) => {
            target_arr.iter().any(|t| json_contains_check(t, scalar_cand))
        }
        (t, c) => t == c,
    }
}

fn json_path_exists_check(doc: &serde_json::Value, steps: &[JsonPathStep]) -> bool {
    let mut current = doc;
    for step in steps {
        match step {
            JsonPathStep::Key(k) => match current {
                serde_json::Value::Object(map) => match map.get(k) {
                    Some(next) => current = next,
                    None => return false,
                },
                _ => return false,
            },
            JsonPathStep::Index(idx) => match current {
                serde_json::Value::Array(arr) => match arr.get(*idx) {
                    Some(next) => current = next,
                    None => return false,
                },
                _ => return false,
            },
        }
    }
    true
}

fn parse_json_arg(val: &Value, fn_name: &str) -> Result<serde_json::Value> {
    let json_str = match val {
        Value::Extension(data) if data.first() == Some(&(DataType::Json as u8)) => {
            std::str::from_utf8(&data[1..]).unwrap_or("").to_string()
        }
        Value::Text(s) => s.to_string(),
        _ => {
            return Err(Error::invalid_argument(format!(
                "{} first argument must be JSON or TEXT",
                fn_name
            )))
        }
    };

    serde_json::from_str(&json_str).map_err(|e| {
        Error::invalid_argument(format!("{} invalid JSON format: {}", fn_name, e))
    })
}

// ============================================================================
// JSON_SET
// ============================================================================

/// JSON_SET function - inserts or updates values in a JSON document
/// JSON_SET(json, path1, val1, [path2, val2, ...])
#[derive(Default)]
pub struct JsonSetFunction;

impl ScalarFunction for JsonSetFunction {
    fn name(&self) -> &str {
        "JSON_SET"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "JSON_SET",
            FunctionType::Scalar,
            "Inserts or updates values in a JSON document at the specified paths",
            FunctionSignature::new(
                FunctionDataType::Json,
                vec![FunctionDataType::Any, FunctionDataType::String, FunctionDataType::Any],
                3,
                255,
            ),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        if args.len() < 3 || !(args.len() - 1).is_multiple_of(2) {
            return Err(Error::invalid_argument(
                "JSON_SET requires an odd number of arguments >= 3 (json, path1, val1, ...)",
            ));
        }

        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }

        let mut doc = parse_json_arg(&args[0], "JSON_SET")?;

        for i in (1..args.len()).step_by(2) {
            if args[i].is_null() {
                return Ok(Value::null_unknown());
            }
            let path_str = match &args[i] {
                Value::Text(s) => s.as_str(),
                _ => {
                    return Err(Error::invalid_argument(
                        "JSON_SET path argument must be a string",
                    ))
                }
            };

            let steps = parse_json_path_steps(path_str).ok_or_else(|| {
                Error::invalid_argument(format!("JSON_SET invalid JSON path: {}", path_str))
            })?;

            let new_val = value_to_json(&args[i + 1]);
            json_path_set(&mut doc, &steps, new_val, false, false);
        }

        Ok(Value::json(doc.to_string()))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(JsonSetFunction)
    }
}

// ============================================================================
// JSON_INSERT
// ============================================================================

/// JSON_INSERT function - inserts values into a JSON document without overwriting existing keys
/// JSON_INSERT(json, path1, val1, [path2, val2, ...])
#[derive(Default)]
pub struct JsonInsertFunction;

impl ScalarFunction for JsonInsertFunction {
    fn name(&self) -> &str {
        "JSON_INSERT"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "JSON_INSERT",
            FunctionType::Scalar,
            "Inserts values into a JSON document at the specified paths only if the path does not exist",
            FunctionSignature::new(
                FunctionDataType::Json,
                vec![FunctionDataType::Any, FunctionDataType::String, FunctionDataType::Any],
                3,
                255,
            ),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        if args.len() < 3 || !(args.len() - 1).is_multiple_of(2) {
            return Err(Error::invalid_argument(
                "JSON_INSERT requires an odd number of arguments >= 3 (json, path1, val1, ...)",
            ));
        }

        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }

        let mut doc = parse_json_arg(&args[0], "JSON_INSERT")?;

        for i in (1..args.len()).step_by(2) {
            if args[i].is_null() {
                return Ok(Value::null_unknown());
            }
            let path_str = match &args[i] {
                Value::Text(s) => s.as_str(),
                _ => return Err(Error::invalid_argument("JSON_INSERT path must be a string")),
            };
            let steps = parse_json_path_steps(path_str)
                .ok_or_else(|| Error::invalid_argument(format!("JSON_INSERT invalid JSON path: {}", path_str)))?;
            let val = value_to_json(&args[i + 1]);
            json_path_set(&mut doc, &steps, val, true, false);
        }

        Ok(Value::json(doc.to_string()))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(JsonInsertFunction)
    }
}

/// JSON_REPLACE(json_doc, path, val[, path, val] ...) — replaces existing values at path
#[derive(Default)]
pub struct JsonReplaceFunction;

impl ScalarFunction for JsonReplaceFunction {
    fn name(&self) -> &str {
        "JSON_REPLACE"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "JSON_REPLACE",
            FunctionType::Scalar,
            "Replaces values in a JSON document at the specified paths only if the path exists",
            FunctionSignature::new(
                FunctionDataType::Json,
                vec![FunctionDataType::Any, FunctionDataType::String, FunctionDataType::Any],
                3,
                255,
            ),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        if args.len() < 3 || !(args.len() - 1).is_multiple_of(2) {
            return Err(Error::invalid_argument(
                "JSON_REPLACE requires an odd number of arguments >= 3 (json, path1, val1, ...)",
            ));
        }

        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }

        let mut doc = parse_json_arg(&args[0], "JSON_REPLACE")?;

        for i in (1..args.len()).step_by(2) {
            if args[i].is_null() {
                return Ok(Value::null_unknown());
            }
            let path_str = match &args[i] {
                Value::Text(s) => s.as_str(),
                _ => {
                    return Err(Error::invalid_argument(
                        "JSON_REPLACE path argument must be a string",
                    ))
                }
            };

            let steps = parse_json_path_steps(path_str).ok_or_else(|| {
                Error::invalid_argument(format!("JSON_REPLACE invalid JSON path: {}", path_str))
            })?;

            let new_val = value_to_json(&args[i + 1]);
            json_path_set(&mut doc, &steps, new_val, false, true);
        }

        Ok(Value::json(doc.to_string()))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(JsonReplaceFunction)
    }
}

// ============================================================================
// JSON_REMOVE
// ============================================================================

/// JSON_REMOVE function - removes elements from a JSON document at specified paths
/// JSON_REMOVE(json, path1, [path2, ...])
#[derive(Default)]
pub struct JsonRemoveFunction;

impl ScalarFunction for JsonRemoveFunction {
    fn name(&self) -> &str {
        "JSON_REMOVE"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "JSON_REMOVE",
            FunctionType::Scalar,
            "Removes data from a JSON document at the specified paths",
            FunctionSignature::new(
                FunctionDataType::Json,
                vec![FunctionDataType::Any, FunctionDataType::String],
                2,
                255,
            ),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        if args.len() < 2 {
            return Err(Error::invalid_argument(
                "JSON_REMOVE requires at least 2 arguments (json, path1, ...)",
            ));
        }

        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }

        let mut doc = parse_json_arg(&args[0], "JSON_REMOVE")?;

        for arg in &args[1..] {
            if arg.is_null() {
                return Ok(Value::null_unknown());
            }
            let path_str = match arg {
                Value::Text(s) => s.as_str(),
                _ => {
                    return Err(Error::invalid_argument(
                        "JSON_REMOVE path argument must be a string",
                    ))
                }
            };

            let steps = parse_json_path_steps(path_str).ok_or_else(|| {
                Error::invalid_argument(format!("JSON_REMOVE invalid JSON path: {}", path_str))
            })?;

            json_path_remove(&mut doc, &steps);
        }

        Ok(Value::json(doc.to_string()))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(JsonRemoveFunction)
    }
}

// ============================================================================
// JSON_CONTAINS
// ============================================================================

/// JSON_CONTAINS function - checks if a JSON candidate is contained within a target JSON document
/// JSON_CONTAINS(target, candidate, [path])
#[derive(Default)]
pub struct JsonContainsFunction;

impl ScalarFunction for JsonContainsFunction {
    fn name(&self) -> &str {
        "JSON_CONTAINS"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "JSON_CONTAINS",
            FunctionType::Scalar,
            "Indicates whether a specific JSON candidate is contained within target JSON document",
            FunctionSignature::new(
                FunctionDataType::Boolean,
                vec![FunctionDataType::Any, FunctionDataType::Any, FunctionDataType::String],
                2,
                3,
            ),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        if args.len() < 2 || args.len() > 3 {
            return Err(Error::invalid_argument(
                "JSON_CONTAINS requires 2 or 3 arguments (target, candidate, [path])",
            ));
        }

        if args[0].is_null() || args[1].is_null() || (args.len() == 3 && args[2].is_null()) {
            return Ok(Value::null_unknown());
        }

        let target_doc = parse_json_arg(&args[0], "JSON_CONTAINS")?;
        let candidate_doc = parse_json_arg(&args[1], "JSON_CONTAINS")
            .unwrap_or_else(|_| value_to_json(&args[1]));

        let search_target = if args.len() == 3 {
            let path_str = match &args[2] {
                Value::Text(s) => s.as_str(),
                _ => {
                    return Err(Error::invalid_argument(
                        "JSON_CONTAINS path argument must be a string",
                    ))
                }
            };
            match extract_json_path(&target_doc, path_str) {
                Some(sub) => sub,
                None => return Ok(Value::Boolean(false)),
            }
        } else {
            &target_doc
        };

        let contains = json_contains_check(search_target, &candidate_doc);
        Ok(Value::Boolean(contains))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(JsonContainsFunction)
    }
}

// ============================================================================
// JSON_CONTAINS_PATH
// ============================================================================

/// JSON_CONTAINS_PATH function - checks if a JSON document contains path(s)
/// JSON_CONTAINS_PATH(json, 'one'|'all', path1, [path2, ...])
#[derive(Default)]
pub struct JsonContainsPathFunction;

impl ScalarFunction for JsonContainsPathFunction {
    fn name(&self) -> &str {
        "JSON_CONTAINS_PATH"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "JSON_CONTAINS_PATH",
            FunctionType::Scalar,
            "Indicates whether a JSON document contains data at the specified path or paths",
            FunctionSignature::new(
                FunctionDataType::Boolean,
                vec![FunctionDataType::Any, FunctionDataType::String, FunctionDataType::String],
                3,
                255,
            ),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        if args.len() < 3 {
            return Err(Error::invalid_argument(
                "JSON_CONTAINS_PATH requires at least 3 arguments (json, 'one'|'all', path1, ...)",
            ));
        }

        for arg in args {
            if arg.is_null() {
                return Ok(Value::null_unknown());
            }
        }

        let doc = parse_json_arg(&args[0], "JSON_CONTAINS_PATH")?;

        let mode = match &args[1] {
            Value::Text(s) => s.to_lowercase(),
            _ => {
                return Err(Error::invalid_argument(
                    "JSON_CONTAINS_PATH mode must be 'one' or 'all'",
                ))
            }
        };

        if mode != "one" && mode != "all" {
            return Err(Error::invalid_argument(
                "JSON_CONTAINS_PATH mode must be 'one' or 'all'",
            ));
        }

        let is_one = mode == "one";

        for arg in &args[2..] {
            let path_str = match arg {
                Value::Text(s) => s.as_str(),
                _ => {
                    return Err(Error::invalid_argument(
                        "JSON_CONTAINS_PATH path argument must be a string",
                    ))
                }
            };

            let steps = match parse_json_path_steps(path_str) {
                Some(s) => s,
                None => {
                    if !is_one {
                        return Ok(Value::Boolean(false));
                    }
                    continue;
                }
            };

            let exists = json_path_exists_check(&doc, &steps);
            if is_one && exists {
                return Ok(Value::Boolean(true));
            }
            if !is_one && !exists {
                return Ok(Value::Boolean(false));
            }
        }

        if is_one {
            Ok(Value::Boolean(false))
        } else {
            Ok(Value::Boolean(true))
        }
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(JsonContainsPathFunction)
    }
}

// ============================================================================
// JSON_QUOTE & JSON_UNQUOTE
// ============================================================================

/// JSON_QUOTE function - quotes a string as a JSON value
#[derive(Default)]
pub struct JsonQuoteFunction;

impl ScalarFunction for JsonQuoteFunction {
    fn name(&self) -> &str {
        "JSON_QUOTE"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "JSON_QUOTE",
            FunctionType::Scalar,
            "Quotes a string as a JSON value",
            FunctionSignature::new(FunctionDataType::Json, vec![FunctionDataType::String], 1, 1),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "JSON_QUOTE", 1);
        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }
        let s = match &args[0] {
            Value::Text(t) => t.as_str(),
            v => {
                let s = v.to_string();
                return Ok(Value::json(serde_json::to_string(&s).unwrap_or(format!("\"{}\"", s))));
            }
        };
        Ok(Value::json(serde_json::to_string(s).unwrap_or(format!("\"{}\"", s))))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(JsonQuoteFunction)
    }
}

/// JSON_UNQUOTE function - unquotes a JSON value
#[derive(Default)]
pub struct JsonUnquoteFunction;

impl ScalarFunction for JsonUnquoteFunction {
    fn name(&self) -> &str {
        "JSON_UNQUOTE"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "JSON_UNQUOTE",
            FunctionType::Scalar,
            "Unquotes JSON value",
            FunctionSignature::new(FunctionDataType::String, vec![FunctionDataType::Any], 1, 1),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "JSON_UNQUOTE", 1);
        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }
        let s = match &args[0] {
            Value::Text(t) => t.to_string(),
            Value::Extension(data) if data.first() == Some(&(DataType::Json as u8)) => {
                std::str::from_utf8(&data[1..]).unwrap_or("").to_string()
            }
            v => v.to_string(),
        };

        if let Ok(serde_json::Value::String(unquoted)) = serde_json::from_str::<serde_json::Value>(&s) {
            return Ok(Value::text(unquoted));
        }

        // If not a JSON quoted string, return as text
        Ok(Value::text(s))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(JsonUnquoteFunction)
    }
}

// ============================================================================
// ARRAY_LENGTH & ARRAY_CONTAINS
// ============================================================================

/// ARRAY_LENGTH function - returns the length of a JSON array or vector
#[derive(Default)]
pub struct ArrayLengthFunction;

impl ScalarFunction for ArrayLengthFunction {
    fn name(&self) -> &str {
        "ARRAY_LENGTH"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "ARRAY_LENGTH",
            FunctionType::Scalar,
            "Returns the number of elements in an array or vector",
            FunctionSignature::new(FunctionDataType::Integer, vec![FunctionDataType::Any], 1, 2),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        // Delegate to JSON_ARRAY_LENGTH
        JsonArrayLengthFunction.evaluate(args)
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(ArrayLengthFunction)
    }
}

/// ARRAY_CONTAINS function - checks if an array contains a specific value
#[derive(Default)]
pub struct ArrayContainsFunction;

impl ScalarFunction for ArrayContainsFunction {
    fn name(&self) -> &str {
        "ARRAY_CONTAINS"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "ARRAY_CONTAINS",
            FunctionType::Scalar,
            "Returns true if the array contains the specified value",
            FunctionSignature::new(FunctionDataType::Boolean, vec![FunctionDataType::Any, FunctionDataType::Any], 2, 2),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "ARRAY_CONTAINS", 2);
        if args[0].is_null() || args[1].is_null() {
            return Ok(Value::null_unknown());
        }

        let doc = parse_json_arg(&args[0], "ARRAY_CONTAINS")?;
        let cand = value_to_json(&args[1]);

        match doc {
            serde_json::Value::Array(arr) => {
                let found = arr.iter().any(|item| json_contains_check(item, &cand));
                Ok(Value::Boolean(found))
            }
            _ => Ok(Value::Boolean(false)),
        }
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(ArrayContainsFunction)
    }
}

// ============================================================================
// GEN_RANDOM_UUID / UUID
// ============================================================================

/// Generate standard RFC 4122 v4 UUID string
fn generate_uuid_v4() -> String {
    let mut bytes = [0u8; 16];
    rand::fill(&mut bytes);
    // Set version 4: bits 12-15 of time_hi_and_version to 0100 (0x40)
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    // Set variant 1: bits 6-7 of clock_seq_hi_and_reserved to 10 (0x80)
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0], bytes[1], bytes[2], bytes[3],
        bytes[4], bytes[5],
        bytes[6], bytes[7],
        bytes[8], bytes[9],
        bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]
    )
}

/// GEN_RANDOM_UUID() — generates a random RFC 4122 version 4 UUID string
#[derive(Default)]
pub struct GenRandomUuidFunction;

impl ScalarFunction for GenRandomUuidFunction {
    fn name(&self) -> &str {
        "GEN_RANDOM_UUID"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "GEN_RANDOM_UUID",
            FunctionType::Scalar,
            "Generates a random UUID (version 4) string",
            FunctionSignature::new(FunctionDataType::String, vec![], 0, 0),
        )
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(GenRandomUuidFunction)
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "GEN_RANDOM_UUID", 0);
        Ok(Value::text(generate_uuid_v4()))
    }
}

/// UUID() — alias for GEN_RANDOM_UUID
#[derive(Default)]
pub struct UuidFunction;

impl ScalarFunction for UuidFunction {
    fn name(&self) -> &str {
        "UUID"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "UUID",
            FunctionType::Scalar,
            "Generates a random UUID (version 4) string",
            FunctionSignature::new(FunctionDataType::String, vec![], 0, 0),
        )
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(UuidFunction)
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "UUID", 0);
        Ok(Value::text(generate_uuid_v4()))
    }
}

// ============================================================================
// INET_ATON / INET_NTOA / IS_IPV4 / IS_IPV6
// ============================================================================

/// INET_ATON(ip_str) — converts IPv4 string to integer
#[derive(Default)]
pub struct InetAtonFunction;

impl ScalarFunction for InetAtonFunction {
    fn name(&self) -> &str {
        "INET_ATON"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "INET_ATON",
            FunctionType::Scalar,
            "Converts an IPv4 network address in dot-decimal notation into an integer",
            FunctionSignature::new(
                FunctionDataType::Integer,
                vec![FunctionDataType::String],
                1,
                1,
            ),
        )
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(InetAtonFunction)
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "INET_ATON", 1);
        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }

        let s = match &args[0] {
            Value::Text(s) => s.as_str(),
            _ => {
                return Err(Error::invalid_argument(
                    "INET_ATON argument must be a string",
                ))
            }
        };

        if let Ok(ip) = s.parse::<std::net::Ipv4Addr>() {
            let num = u32::from(ip) as i64;
            Ok(Value::Integer(num))
        } else {
            Ok(Value::null_unknown())
        }
    }
}

/// INET_NTOA(ip_num) — converts integer to IPv4 string
#[derive(Default)]
pub struct InetNtoaFunction;

impl ScalarFunction for InetNtoaFunction {
    fn name(&self) -> &str {
        "INET_NTOA"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "INET_NTOA",
            FunctionType::Scalar,
            "Converts an integer representation of an IPv4 address into dot-decimal string notation",
            FunctionSignature::new(
                FunctionDataType::String,
                vec![FunctionDataType::Integer],
                1,
                1,
            ),
        )
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(InetNtoaFunction)
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "INET_NTOA", 1);
        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }

        let num = match &args[0] {
            Value::Integer(i) => *i,
            _ => {
                return Err(Error::invalid_argument(
                    "INET_NTOA argument must be an integer",
                ))
            }
        };

        if num < 0 || num > u32::MAX as i64 {
            return Ok(Value::null_unknown());
        }

        let ip = std::net::Ipv4Addr::from(num as u32);
        Ok(Value::text(ip.to_string()))
    }
}

/// IS_IPV4(str) — returns true if string is a valid IPv4 address
#[derive(Default)]
pub struct IsIpv4Function;

impl ScalarFunction for IsIpv4Function {
    fn name(&self) -> &str {
        "IS_IPV4"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "IS_IPV4",
            FunctionType::Scalar,
            "Returns true if string is a valid IPv4 address",
            FunctionSignature::new(
                FunctionDataType::Boolean,
                vec![FunctionDataType::String],
                1,
                1,
            ),
        )
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(IsIpv4Function)
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "IS_IPV4", 1);
        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }

        let s = match &args[0] {
            Value::Text(s) => s.as_str(),
            _ => return Ok(Value::Boolean(false)),
        };

        Ok(Value::Boolean(s.parse::<std::net::Ipv4Addr>().is_ok()))
    }
}

/// IS_IPV6(str) — returns true if string is a valid IPv6 address
#[derive(Default)]
pub struct IsIpv6Function;

impl ScalarFunction for IsIpv6Function {
    fn name(&self) -> &str {
        "IS_IPV6"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "IS_IPV6",
            FunctionType::Scalar,
            "Returns true if string is a valid IPv6 address",
            FunctionSignature::new(
                FunctionDataType::Boolean,
                vec![FunctionDataType::String],
                1,
                1,
            ),
        )
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(IsIpv6Function)
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "IS_IPV6", 1);
        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }

        let s = match &args[0] {
            Value::Text(s) => s.as_str(),
            _ => return Ok(Value::Boolean(false)),
        };

        Ok(Value::Boolean(s.parse::<std::net::Ipv6Addr>().is_ok()))
    }
}

// ============================================================================
// INET6_ATON / INET6_NTOA / IS_VALID_JSON
// ============================================================================

/// INET6_ATON(ipv6_str) — converts IPv6 string to 16-byte hex representation
#[derive(Default)]
pub struct Inet6AtonFunction;

impl ScalarFunction for Inet6AtonFunction {
    fn name(&self) -> &str {
        "INET6_ATON"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "INET6_ATON",
            FunctionType::Scalar,
            "Converts IPv6 or IPv4 address string to its 16-byte hex representation",
            FunctionSignature::new(
                FunctionDataType::String,
                vec![FunctionDataType::String],
                1,
                1,
            ),
        )
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(Inet6AtonFunction)
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "INET6_ATON", 1);
        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }

        let s = match &args[0] {
            Value::Text(s) => s.as_str(),
            _ => return Ok(Value::null_unknown()),
        };

        const HEX_CHARS: &[u8; 16] = b"0123456789ABCDEF";
        let octets = if let Ok(ipv6) = s.parse::<std::net::Ipv6Addr>() {
            Some(ipv6.octets())
        } else if let Ok(ipv4) = s.parse::<std::net::Ipv4Addr>() {
            Some(ipv4.to_ipv6_mapped().octets())
        } else {
            None
        };

        if let Some(bytes) = octets {
            let mut hex = String::with_capacity(32);
            for &b in &bytes {
                hex.push(HEX_CHARS[(b >> 4) as usize] as char);
                hex.push(HEX_CHARS[(b & 0x0F) as usize] as char);
            }
            Ok(Value::text(hex))
        } else {
            Ok(Value::null_unknown())
        }
    }
}

/// INET6_NTOA(hex_or_bytes) — converts 16-byte hex representation to standard IPv6 string
#[derive(Default)]
pub struct Inet6NtoaFunction;

impl ScalarFunction for Inet6NtoaFunction {
    fn name(&self) -> &str {
        "INET6_NTOA"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "INET6_NTOA",
            FunctionType::Scalar,
            "Converts a 16-byte hex representation of an IPv6 address back to standard string format",
            FunctionSignature::new(
                FunctionDataType::String,
                vec![FunctionDataType::String],
                1,
                1,
            ),
        )
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(Inet6NtoaFunction)
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "INET6_NTOA", 1);
        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }

        let hex_str = match &args[0] {
            Value::Text(s) => s.as_str(),
            _ => return Ok(Value::null_unknown()),
        };

        if hex_str.len() != 32 {
            return Ok(Value::null_unknown());
        }

        let mut octets = [0u8; 16];
        for i in 0..16 {
            let byte_str = &hex_str[i * 2..i * 2 + 2];
            match u8::from_str_radix(byte_str, 16) {
                Ok(b) => octets[i] = b,
                Err(_) => return Ok(Value::null_unknown()),
            }
        }

        let ipv6 = std::net::Ipv6Addr::from(octets);
        Ok(Value::text(ipv6.to_string()))
    }
}

/// IS_VALID_JSON(str) — checks if string is a valid JSON document
#[derive(Default)]
pub struct IsValidJsonFunction;

impl ScalarFunction for IsValidJsonFunction {
    fn name(&self) -> &str {
        "IS_VALID_JSON"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "IS_VALID_JSON",
            FunctionType::Scalar,
            "Returns true if string is valid JSON document",
            FunctionSignature::new(
                FunctionDataType::Boolean,
                vec![FunctionDataType::String],
                1,
                1,
            ),
        )
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(IsValidJsonFunction)
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "IS_VALID_JSON", 1);
        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }
        let is_valid = match &args[0] {
            Value::Text(s) => serde_json::from_str::<serde_json::Value>(s.as_str()).is_ok(),
            _ => false,
        };
        Ok(Value::Boolean(is_valid))
    }
}

#[cfg(test)]
mod tests {
    use super::*;


    #[test]
    fn test_coalesce_first_non_null() {
        let f = CoalesceFunction;
        assert_eq!(
            f.evaluate(&[
                Value::null_unknown(),
                Value::null_unknown(),
                Value::Integer(42),
                Value::Integer(100)
            ])
            .unwrap(),
            Value::Integer(42)
        );
    }

    #[test]
    fn test_coalesce_first_arg_not_null() {
        let f = CoalesceFunction;
        assert_eq!(
            f.evaluate(&[Value::text("hello"), Value::null_unknown()])
                .unwrap(),
            Value::text("hello")
        );
    }

    #[test]
    fn test_coalesce_all_null() {
        let f = CoalesceFunction;
        assert!(f
            .evaluate(&[
                Value::null_unknown(),
                Value::null_unknown(),
                Value::null_unknown()
            ])
            .unwrap()
            .is_null());
    }

    #[test]
    fn test_coalesce_single_value() {
        let f = CoalesceFunction;
        assert_eq!(
            f.evaluate(&[Value::Integer(42)]).unwrap(),
            Value::Integer(42)
        );
        assert!(f.evaluate(&[Value::null_unknown()]).unwrap().is_null());
    }

    #[test]
    fn test_coalesce_empty_error() {
        let f = CoalesceFunction;
        assert!(f.evaluate(&[]).is_err());
    }

    #[test]
    fn test_now_returns_timestamp() {
        let f = NowFunction;
        let result = f.evaluate(&[]).unwrap();
        assert!(matches!(result, Value::Timestamp(_)));
    }

    #[test]
    fn test_now_no_args() {
        let f = NowFunction;
        assert!(f.evaluate(&[Value::Integer(1)]).is_err());
    }

    #[test]
    fn test_nullif_equal() {
        let f = NullIfFunction;
        assert!(f
            .evaluate(&[Value::Integer(42), Value::Integer(42)])
            .unwrap()
            .is_null());
        assert!(f
            .evaluate(&[Value::text("hello"), Value::text("hello")])
            .unwrap()
            .is_null());
    }

    #[test]
    fn test_nullif_not_equal() {
        let f = NullIfFunction;
        assert_eq!(
            f.evaluate(&[Value::Integer(42), Value::Integer(100)])
                .unwrap(),
            Value::Integer(42)
        );
        assert_eq!(
            f.evaluate(&[Value::text("hello"), Value::text("world")])
                .unwrap(),
            Value::text("hello")
        );
    }

    #[test]
    fn test_nullif_with_null() {
        let f = NullIfFunction;
        // NULL == NULL in SQL comparisons for NULLIF
        assert!(f
            .evaluate(&[Value::null_unknown(), Value::null_unknown()])
            .unwrap()
            .is_null());
        assert_eq!(
            f.evaluate(&[Value::Integer(42), Value::null_unknown()])
                .unwrap(),
            Value::Integer(42)
        );
        assert!(f
            .evaluate(&[Value::null_unknown(), Value::Integer(42)])
            .unwrap()
            .is_null());
    }

    #[test]
    fn test_ifnull_first_not_null() {
        let f = IfNullFunction;
        assert_eq!(
            f.evaluate(&[Value::Integer(42), Value::Integer(100)])
                .unwrap(),
            Value::Integer(42)
        );
    }

    #[test]
    fn test_ifnull_first_null() {
        let f = IfNullFunction;
        assert_eq!(
            f.evaluate(&[Value::null_unknown(), Value::Integer(100)])
                .unwrap(),
            Value::Integer(100)
        );
    }

    #[test]
    fn test_ifnull_both_null() {
        let f = IfNullFunction;
        assert!(f
            .evaluate(&[Value::null_unknown(), Value::null_unknown()])
            .unwrap()
            .is_null());
    }

    #[test]
    fn test_ifnull_wrong_arg_count() {
        let f = IfNullFunction;
        assert!(f.evaluate(&[Value::Integer(1)]).is_err());
        assert!(f
            .evaluate(&[Value::Integer(1), Value::Integer(2), Value::Integer(3)])
            .is_err());
    }

    // ========================================================================
    // GREATEST tests
    // ========================================================================

    #[test]
    fn test_greatest_integers() {
        let f = GreatestFunction;
        assert_eq!(
            f.evaluate(&[Value::Integer(1), Value::Integer(5), Value::Integer(3)])
                .unwrap(),
            Value::Integer(5)
        );
    }

    #[test]
    fn test_greatest_floats() {
        let f = GreatestFunction;
        assert_eq!(
            f.evaluate(&[Value::Float(1.5), Value::Float(2.5), Value::Float(0.5)])
                .unwrap(),
            Value::Float(2.5)
        );
    }

    #[test]
    fn test_greatest_strings() {
        let f = GreatestFunction;
        assert_eq!(
            f.evaluate(&[
                Value::text("apple"),
                Value::text("banana"),
                Value::text("cherry")
            ])
            .unwrap(),
            Value::text("cherry")
        );
    }

    #[test]
    fn test_greatest_with_null() {
        let f = GreatestFunction;
        assert!(f
            .evaluate(&[Value::Integer(1), Value::null_unknown(), Value::Integer(3)])
            .unwrap()
            .is_null());
    }

    #[test]
    fn test_greatest_single_value() {
        let f = GreatestFunction;
        assert_eq!(
            f.evaluate(&[Value::Integer(42)]).unwrap(),
            Value::Integer(42)
        );
    }

    #[test]
    fn test_greatest_empty_error() {
        let f = GreatestFunction;
        assert!(f.evaluate(&[]).is_err());
    }

    // ========================================================================
    // LEAST tests
    // ========================================================================

    #[test]
    fn test_least_integers() {
        let f = LeastFunction;
        assert_eq!(
            f.evaluate(&[Value::Integer(1), Value::Integer(5), Value::Integer(3)])
                .unwrap(),
            Value::Integer(1)
        );
    }

    #[test]
    fn test_least_floats() {
        let f = LeastFunction;
        assert_eq!(
            f.evaluate(&[Value::Float(1.5), Value::Float(2.5), Value::Float(0.5)])
                .unwrap(),
            Value::Float(0.5)
        );
    }

    #[test]
    fn test_least_strings() {
        let f = LeastFunction;
        assert_eq!(
            f.evaluate(&[
                Value::text("apple"),
                Value::text("banana"),
                Value::text("cherry")
            ])
            .unwrap(),
            Value::text("apple")
        );
    }

    #[test]
    fn test_least_with_null() {
        let f = LeastFunction;
        assert!(f
            .evaluate(&[Value::Integer(1), Value::null_unknown(), Value::Integer(3)])
            .unwrap()
            .is_null());
    }

    #[test]
    fn test_least_single_value() {
        let f = LeastFunction;
        assert_eq!(
            f.evaluate(&[Value::Integer(42)]).unwrap(),
            Value::Integer(42)
        );
    }

    #[test]
    fn test_least_empty_error() {
        let f = LeastFunction;
        assert!(f.evaluate(&[]).is_err());
    }

    // ========================================================================
    // IIF tests
    // ========================================================================

    #[test]
    fn test_iif_true_condition() {
        let f = IifFunction;
        assert_eq!(
            f.evaluate(&[Value::Integer(1), Value::text("yes"), Value::text("no")])
                .unwrap(),
            Value::text("yes")
        );
    }

    #[test]
    fn test_iif_false_condition() {
        let f = IifFunction;
        assert_eq!(
            f.evaluate(&[Value::Integer(0), Value::text("yes"), Value::text("no")])
                .unwrap(),
            Value::text("no")
        );
    }

    #[test]
    fn test_iif_null_condition() {
        let f = IifFunction;
        assert_eq!(
            f.evaluate(&[Value::null_unknown(), Value::text("yes"), Value::text("no")])
                .unwrap(),
            Value::text("no")
        );
    }

    #[test]
    fn test_iif_with_numbers() {
        let f = IifFunction;
        assert_eq!(
            f.evaluate(&[Value::Integer(1), Value::Integer(100), Value::Integer(200)])
                .unwrap(),
            Value::Integer(100)
        );
    }

    #[test]
    fn test_iif_wrong_arg_count() {
        let f = IifFunction;
        assert!(f.evaluate(&[Value::Integer(1), Value::Integer(2)]).is_err());
        assert!(f
            .evaluate(&[
                Value::Integer(1),
                Value::Integer(2),
                Value::Integer(3),
                Value::Integer(4)
            ])
            .is_err());
    }

    // ========================================================================
    // JSON_EXTRACT tests
    // ========================================================================

    #[test]
    fn test_json_extract_simple() {
        let f = JsonExtractFunction;
        let json = Value::json(r#"{"name": "Alice", "age": 30}"#.to_owned());

        // Extract string
        assert_eq!(
            f.evaluate(&[json.clone(), Value::text("$.name")]).unwrap(),
            Value::text("Alice")
        );

        // Extract number
        assert_eq!(
            f.evaluate(&[json.clone(), Value::text("$.age")]).unwrap(),
            Value::Integer(30)
        );
    }

    #[test]
    fn test_json_extract_nested() {
        let f = JsonExtractFunction;
        let json = Value::json(r#"{"user": {"name": "Bob"}}"#.to_owned());

        assert_eq!(
            f.evaluate(&[json, Value::text("$.user.name")]).unwrap(),
            Value::text("Bob")
        );
    }

    #[test]
    fn test_json_extract_array() {
        let f = JsonExtractFunction;
        let json = Value::json(r#"{"items": [1, 2, 3]}"#.to_owned());

        assert_eq!(
            f.evaluate(&[json.clone(), Value::text("$.items[0]")])
                .unwrap(),
            Value::Integer(1)
        );

        assert_eq!(
            f.evaluate(&[json, Value::text("$.items[2]")]).unwrap(),
            Value::Integer(3)
        );
    }

    #[test]
    fn test_json_extract_missing_path() {
        let f = JsonExtractFunction;
        let json = Value::json(r#"{"name": "Alice"}"#.to_owned());

        assert!(f
            .evaluate(&[json, Value::text("$.missing")])
            .unwrap()
            .is_null());
    }

    // ========================================================================
    // TYPEOF tests
    // ========================================================================

    #[test]
    fn test_typeof_integer() {
        let f = TypeOfFunction;
        assert_eq!(
            f.evaluate(&[Value::Integer(123)]).unwrap(),
            Value::text("INTEGER")
        );
    }

    #[test]
    fn test_typeof_float() {
        let f = TypeOfFunction;
        assert_eq!(
            f.evaluate(&[Value::Float(3.5)]).unwrap(),
            Value::text("FLOAT")
        );
    }

    #[test]
    fn test_typeof_text() {
        let f = TypeOfFunction;
        assert_eq!(
            f.evaluate(&[Value::text("hello")]).unwrap(),
            Value::text("TEXT")
        );
    }

    #[test]
    fn test_typeof_boolean() {
        let f = TypeOfFunction;
        assert_eq!(
            f.evaluate(&[Value::Boolean(true)]).unwrap(),
            Value::text("BOOLEAN")
        );
    }

    #[test]
    fn test_typeof_null() {
        let f = TypeOfFunction;
        assert_eq!(
            f.evaluate(&[Value::null_unknown()]).unwrap(),
            Value::text("NULL")
        );
    }

    #[test]
    fn test_typeof_json() {
        let f = TypeOfFunction;
        assert_eq!(
            f.evaluate(&[Value::json("{}".to_owned())]).unwrap(),
            Value::text("JSON")
        );
    }

    // ========================================================================
    // JSON_SET, JSON_INSERT, JSON_REPLACE, JSON_REMOVE tests
    // ========================================================================

    #[test]
    fn test_json_set() {
        let f = JsonSetFunction;
        let json = Value::json(r#"{"a": 1, "b": [10, 20]}"#.to_owned());

        // Update existing property
        let res = f
            .evaluate(&[json.clone(), Value::text("$.a"), Value::Integer(99)])
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_str(res.as_str().unwrap()).unwrap();
        assert_eq!(parsed["a"], 99);
        assert_eq!(parsed["b"][0], 10);

        // Insert new property
        let res = f
            .evaluate(&[json.clone(), Value::text("$.c"), Value::text("new_val")])
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_str(res.as_str().unwrap()).unwrap();
        assert_eq!(parsed["c"], "new_val");

        // Update array element
        let res = f
            .evaluate(&[json.clone(), Value::text("$.b[1]"), Value::Integer(200)])
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_str(res.as_str().unwrap()).unwrap();
        assert_eq!(parsed["b"][1], 200);

        // Multiple updates
        let res = f
            .evaluate(&[
                json,
                Value::text("$.a"),
                Value::Integer(5),
                Value::text("$.d"),
                Value::Boolean(true),
            ])
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_str(res.as_str().unwrap()).unwrap();
        assert_eq!(parsed["a"], 5);
        assert_eq!(parsed["d"], true);
    }

    #[test]
    fn test_json_insert() {
        let f = JsonInsertFunction;
        let json = Value::json(r#"{"a": 1, "b": 2}"#.to_owned());

        // Attempt to insert existing - should NOT overwrite
        let res = f
            .evaluate(&[json.clone(), Value::text("$.a"), Value::Integer(999)])
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_str(res.as_str().unwrap()).unwrap();
        assert_eq!(parsed["a"], 1);

        // Insert new - should add
        let res = f
            .evaluate(&[json, Value::text("$.c"), Value::Integer(3)])
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_str(res.as_str().unwrap()).unwrap();
        assert_eq!(parsed["c"], 3);
    }

    #[test]
    fn test_json_replace() {
        let f = JsonReplaceFunction;
        let json = Value::json(r#"{"a": 1, "b": 2}"#.to_owned());

        // Replace existing - should update
        let res = f
            .evaluate(&[json.clone(), Value::text("$.a"), Value::Integer(100)])
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_str(res.as_str().unwrap()).unwrap();
        assert_eq!(parsed["a"], 100);

        // Replace non-existing - should NOT add
        let res = f
            .evaluate(&[json, Value::text("$.c"), Value::Integer(300)])
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_str(res.as_str().unwrap()).unwrap();
        assert!(parsed.get("c").is_none());
    }

    #[test]
    fn test_json_remove() {
        let f = JsonRemoveFunction;
        let json = Value::json(r#"{"a": 1, "b": [10, 20, 30], "c": "hello"}"#.to_owned());

        // Remove object property
        let res = f
            .evaluate(&[json.clone(), Value::text("$.c")])
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_str(res.as_str().unwrap()).unwrap();
        assert!(parsed.get("c").is_none());
        assert_eq!(parsed["a"], 1);

        // Remove array element
        let res = f
            .evaluate(&[json.clone(), Value::text("$.b[1]")])
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_str(res.as_str().unwrap()).unwrap();
        assert_eq!(parsed["b"].as_array().unwrap().len(), 2);
        assert_eq!(parsed["b"][0], 10);
        assert_eq!(parsed["b"][1], 30);
    }

    #[test]
    fn test_json_contains() {
        let f = JsonContainsFunction;
        let target = Value::json(r#"{"a": 1, "b": [1, 2, 3], "c": {"d": 4}}"#.to_owned());

        // Scalar match in object
        assert_eq!(
            f.evaluate(&[target.clone(), Value::json("1".to_owned()), Value::text("$.a")])
                .unwrap(),
            Value::Boolean(true)
        );

        // Sub-object containment
        assert_eq!(
            f.evaluate(&[target.clone(), Value::json(r#"{"d": 4}"#.to_owned())])
                .unwrap(),
            Value::Boolean(false) // top-level target doesn't have "d", it's in "c"
        );
        assert_eq!(
            f.evaluate(&[
                target.clone(),
                Value::json(r#"{"d": 4}"#.to_owned()),
                Value::text("$.c")
            ])
            .unwrap(),
            Value::Boolean(true)
        );

        // Array containment
        assert_eq!(
            f.evaluate(&[
                target.clone(),
                Value::json("[1, 2]".to_owned()),
                Value::text("$.b")
            ])
            .unwrap(),
            Value::Boolean(true)
        );
        assert_eq!(
            f.evaluate(&[
                target,
                Value::json("[1, 5]".to_owned()),
                Value::text("$.b")
            ])
            .unwrap(),
            Value::Boolean(false)
        );
    }

    #[test]
    fn test_json_contains_path() {
        let f = JsonContainsPathFunction;
        let doc = Value::json(r#"{"a": 1, "b": {"c": 2}}"#.to_owned());

        // 'one' mode
        assert_eq!(
            f.evaluate(&[
                doc.clone(),
                Value::text("one"),
                Value::text("$.a"),
                Value::text("$.missing")
            ])
            .unwrap(),
            Value::Boolean(true)
        );

        // 'all' mode
        assert_eq!(
            f.evaluate(&[
                doc.clone(),
                Value::text("all"),
                Value::text("$.a"),
                Value::text("$.b.c")
            ])
            .unwrap(),
            Value::Boolean(true)
        );
        assert_eq!(
            f.evaluate(&[
                doc,
                Value::text("all"),
                Value::text("$.a"),
                Value::text("$.missing")
            ])
            .unwrap(),
            Value::Boolean(false)
        );
    }

    #[test]
    fn test_json_quote_unquote() {
        let q = JsonQuoteFunction;
        let u = JsonUnquoteFunction;

        let quoted = q.evaluate(&[Value::text("hello world")]).unwrap();
        assert_eq!(quoted, Value::json(r#""hello world""#.to_owned()));

        let unquoted = u.evaluate(&[quoted]).unwrap();
        assert_eq!(unquoted, Value::text("hello world"));
    }

    #[test]
    fn test_array_contains() {
        let f = ArrayContainsFunction;
        let arr = Value::json(r#"[10, 20, 30, "hello"]"#.to_owned());

        assert_eq!(
            f.evaluate(&[arr.clone(), Value::Integer(20)]).unwrap(),
            Value::Boolean(true)
        );
        assert_eq!(
            f.evaluate(&[arr.clone(), Value::text("hello")]).unwrap(),
            Value::Boolean(true)
        );
        assert_eq!(
            f.evaluate(&[arr, Value::Integer(99)]).unwrap(),
            Value::Boolean(false)
        );
    }
}

