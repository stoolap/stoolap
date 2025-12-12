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

use crate::core::{Error, Result, Value};
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
            Value::Text(s) => !s.is_empty() && s.to_lowercase() != "false" && s.as_ref() != "0",
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
            Value::Json(j) => j.to_string(),
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
        _ => Ok(Value::Json(std::sync::Arc::from(json.to_string().as_str()))),
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
            Value::Json(j) => j.to_string(),
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
        Ok(Value::Json(std::sync::Arc::from(
            json_array.to_string().as_str(),
        )))
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
        Ok(Value::Json(std::sync::Arc::from(
            json_object.to_string().as_str(),
        )))
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
        Value::Json(j) => {
            // Parse the JSON string to get a proper JSON value
            serde_json::from_str(j).unwrap_or(serde_json::Value::String(j.to_string()))
        }
        Value::Timestamp(t) => serde_json::Value::String(t.to_rfc3339()),
    }
}

// ============================================================================
// JSON_TYPE
// ============================================================================

/// JSON_TYPE function - returns the type of a JSON value
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
            "Returns the type of a JSON value (object, array, string, number, boolean, null)",
            FunctionSignature::new(FunctionDataType::String, vec![FunctionDataType::Any], 1, 1),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "JSON_TYPE", 1);

        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }

        // Get JSON string
        let json_str = match &args[0] {
            Value::Json(j) => j.to_string(),
            Value::Text(s) => s.to_string(),
            _ => {
                return Err(Error::invalid_argument(
                    "JSON_TYPE argument must be JSON or TEXT",
                ))
            }
        };

        // Parse JSON and determine type
        let json_value: serde_json::Value = match serde_json::from_str(&json_str) {
            Ok(v) => v,
            Err(_) => return Ok(Value::null_unknown()),
        };

        let type_name = match json_value {
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
            "Returns the type of a JSON value (PostgreSQL-style alias for JSON_TYPE)",
            FunctionSignature::new(FunctionDataType::String, vec![FunctionDataType::Any], 1, 1),
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

        // Get string to validate
        let json_str = match &args[0] {
            Value::Json(_) => return Ok(Value::Integer(1)), // Already JSON type, valid
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
            Value::Json(j) => j.to_string(),
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
                Ok(Value::Json(std::sync::Arc::from(
                    keys_array.to_string().as_str(),
                )))
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
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "SLEEP", 1);

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
            Value::Json(_) => "JSON",
        };

        Ok(Value::text(type_name))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(TypeOfFunction)
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
        let json = Value::Json(std::sync::Arc::from(r#"{"name": "Alice", "age": 30}"#));

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
        let json = Value::Json(std::sync::Arc::from(r#"{"user": {"name": "Bob"}}"#));

        assert_eq!(
            f.evaluate(&[json, Value::text("$.user.name")]).unwrap(),
            Value::text("Bob")
        );
    }

    #[test]
    fn test_json_extract_array() {
        let f = JsonExtractFunction;
        let json = Value::Json(std::sync::Arc::from(r#"{"items": [1, 2, 3]}"#));

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
        let json = Value::Json(std::sync::Arc::from(r#"{"name": "Alice"}"#));

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
            f.evaluate(&[Value::Json(std::sync::Arc::from("{}"))])
                .unwrap(),
            Value::text("JSON")
        );
    }
}
