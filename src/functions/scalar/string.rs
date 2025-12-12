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

//! String scalar functions

use std::sync::Arc;

use crate::core::{Error, Result, Value};
use crate::functions::{
    FunctionDataType, FunctionInfo, FunctionSignature, FunctionType, ScalarFunction,
};
use crate::validate_arg_count;

use super::{value_to_i64, value_to_string};

// ============================================================================
// UPPER
// ============================================================================

/// UPPER function - converts a string to uppercase
#[derive(Default)]
pub struct UpperFunction;

impl ScalarFunction for UpperFunction {
    fn name(&self) -> &str {
        "UPPER"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "UPPER",
            FunctionType::Scalar,
            "Converts a string to uppercase",
            FunctionSignature::new(FunctionDataType::String, vec![FunctionDataType::Any], 1, 1),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "UPPER", 1);

        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }

        let s = value_to_string(&args[0]);
        Ok(Value::Text(Arc::from(s.to_uppercase().as_str())))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(UpperFunction)
    }
}

// ============================================================================
// LOWER
// ============================================================================

/// LOWER function - converts a string to lowercase
#[derive(Default)]
pub struct LowerFunction;

impl ScalarFunction for LowerFunction {
    fn name(&self) -> &str {
        "LOWER"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "LOWER",
            FunctionType::Scalar,
            "Converts a string to lowercase",
            FunctionSignature::new(FunctionDataType::String, vec![FunctionDataType::Any], 1, 1),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "LOWER", 1);

        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }

        let s = value_to_string(&args[0]);
        Ok(Value::Text(Arc::from(s.to_lowercase().as_str())))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(LowerFunction)
    }
}

// ============================================================================
// LENGTH
// ============================================================================

/// LENGTH function - returns the length of a string
#[derive(Default)]
pub struct LengthFunction;

impl ScalarFunction for LengthFunction {
    fn name(&self) -> &str {
        "LENGTH"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "LENGTH",
            FunctionType::Scalar,
            "Returns the length of a string",
            FunctionSignature::new(FunctionDataType::Integer, vec![FunctionDataType::Any], 1, 1),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "LENGTH", 1);

        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }

        let s = value_to_string(&args[0]);
        Ok(Value::Integer(s.chars().count() as i64))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(LengthFunction)
    }
}

// ============================================================================
// CONCAT
// ============================================================================

/// CONCAT function - concatenates multiple strings
#[derive(Default)]
pub struct ConcatFunction;

impl ScalarFunction for ConcatFunction {
    fn name(&self) -> &str {
        "CONCAT"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "CONCAT",
            FunctionType::Scalar,
            "Concatenates multiple strings",
            FunctionSignature::variadic(FunctionDataType::String, FunctionDataType::Any),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        if args.is_empty() {
            return Err(Error::invalid_argument(
                "CONCAT requires at least 1 argument",
            ));
        }

        let mut result = String::new();
        for arg in args {
            if !arg.is_null() {
                result.push_str(&value_to_string(arg));
            }
        }

        Ok(Value::Text(Arc::from(result.as_str())))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(ConcatFunction)
    }
}

// ============================================================================
// SUBSTRING
// ============================================================================

/// SUBSTRING function - extracts a substring
#[derive(Default)]
pub struct SubstringFunction;

impl ScalarFunction for SubstringFunction {
    fn name(&self) -> &str {
        "SUBSTRING"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "SUBSTRING",
            FunctionType::Scalar,
            "Extracts a substring from a string",
            FunctionSignature::new(
                FunctionDataType::String,
                vec![
                    FunctionDataType::Any,
                    FunctionDataType::Integer,
                    FunctionDataType::Integer,
                ],
                2, // SUBSTRING(str, start) or SUBSTRING(str, start, length)
                3,
            ),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "SUBSTRING", 2, 3);

        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }

        let s = value_to_string(&args[0]);
        let chars: Vec<char> = s.chars().collect();

        // SQL SUBSTRING uses 1-based indexing
        let start = value_to_i64(&args[1]).ok_or_else(|| {
            Error::invalid_argument("SUBSTRING start position must be an integer")
        })?;

        // Convert to 0-based index, handling negative values
        let start_idx = if start < 1 { 0 } else { (start - 1) as usize };

        if start_idx >= chars.len() {
            return Ok(Value::Text(Arc::from("")));
        }

        let result: String = if args.len() == 3 {
            let length = value_to_i64(&args[2])
                .ok_or_else(|| Error::invalid_argument("SUBSTRING length must be an integer"))?;

            if length < 0 {
                return Ok(Value::Text(Arc::from("")));
            }

            let end_idx = std::cmp::min(start_idx + length as usize, chars.len());
            chars[start_idx..end_idx].iter().collect()
        } else {
            chars[start_idx..].iter().collect()
        };

        Ok(Value::Text(Arc::from(result.as_str())))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(SubstringFunction)
    }
}

// ============================================================================
// SUBSTR (alias for SUBSTRING)
// ============================================================================

/// SUBSTR function - alias for SUBSTRING for compatibility
#[derive(Default)]
pub struct SubstrFunction;

impl ScalarFunction for SubstrFunction {
    fn name(&self) -> &str {
        "SUBSTR"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "SUBSTR",
            FunctionType::Scalar,
            "Extracts a substring from a string (alias for SUBSTRING)",
            FunctionSignature::new(
                FunctionDataType::String,
                vec![
                    FunctionDataType::Any,
                    FunctionDataType::Integer,
                    FunctionDataType::Integer,
                ],
                2,
                3,
            ),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        // Delegate to SUBSTRING implementation
        SubstringFunction.evaluate(args)
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(SubstrFunction)
    }
}

// ============================================================================
// TRIM
// ============================================================================

/// TRIM function - removes leading and trailing whitespace from a string
#[derive(Default)]
pub struct TrimFunction;

impl ScalarFunction for TrimFunction {
    fn name(&self) -> &str {
        "TRIM"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "TRIM",
            FunctionType::Scalar,
            "Removes leading and trailing whitespace from a string",
            FunctionSignature::new(FunctionDataType::String, vec![FunctionDataType::Any], 1, 1),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "TRIM", 1);

        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }

        let s = value_to_string(&args[0]);
        Ok(Value::Text(Arc::from(s.trim())))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(TrimFunction)
    }
}

// ============================================================================
// LTRIM
// ============================================================================

/// LTRIM function - removes leading whitespace from a string
#[derive(Default)]
pub struct LtrimFunction;

impl ScalarFunction for LtrimFunction {
    fn name(&self) -> &str {
        "LTRIM"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "LTRIM",
            FunctionType::Scalar,
            "Removes leading whitespace from a string",
            FunctionSignature::new(FunctionDataType::String, vec![FunctionDataType::Any], 1, 1),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "LTRIM", 1);

        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }

        let s = value_to_string(&args[0]);
        Ok(Value::Text(Arc::from(s.trim_start())))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(LtrimFunction)
    }
}

// ============================================================================
// RTRIM
// ============================================================================

/// RTRIM function - removes trailing whitespace from a string
#[derive(Default)]
pub struct RtrimFunction;

impl ScalarFunction for RtrimFunction {
    fn name(&self) -> &str {
        "RTRIM"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "RTRIM",
            FunctionType::Scalar,
            "Removes trailing whitespace from a string",
            FunctionSignature::new(FunctionDataType::String, vec![FunctionDataType::Any], 1, 1),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "RTRIM", 1);

        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }

        let s = value_to_string(&args[0]);
        Ok(Value::Text(Arc::from(s.trim_end())))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(RtrimFunction)
    }
}

// ============================================================================
// REPLACE
// ============================================================================

/// REPLACE function - replaces all occurrences of a substring with another substring
#[derive(Default)]
pub struct ReplaceFunction;

impl ScalarFunction for ReplaceFunction {
    fn name(&self) -> &str {
        "REPLACE"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "REPLACE",
            FunctionType::Scalar,
            "Replaces all occurrences of a substring with another substring",
            FunctionSignature::new(
                FunctionDataType::String,
                vec![
                    FunctionDataType::Any,
                    FunctionDataType::Any,
                    FunctionDataType::Any,
                ],
                3,
                3,
            ),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "REPLACE", 3);

        // SQL standard: any NULL argument returns NULL
        if args[0].is_null() || args[1].is_null() || args[2].is_null() {
            return Ok(Value::null_unknown());
        }

        let s = value_to_string(&args[0]);
        let from = value_to_string(&args[1]);
        let to = value_to_string(&args[2]);

        Ok(Value::Text(Arc::from(s.replace(&from, &to).as_str())))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(ReplaceFunction)
    }
}

// ============================================================================
// REVERSE
// ============================================================================

/// REVERSE function - reverses a string
#[derive(Default)]
pub struct ReverseFunction;

impl ScalarFunction for ReverseFunction {
    fn name(&self) -> &str {
        "REVERSE"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "REVERSE",
            FunctionType::Scalar,
            "Reverses a string",
            FunctionSignature::new(FunctionDataType::String, vec![FunctionDataType::Any], 1, 1),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "REVERSE", 1);

        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }

        let s = value_to_string(&args[0]);
        let reversed: String = s.chars().rev().collect();
        Ok(Value::Text(Arc::from(reversed.as_str())))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(ReverseFunction)
    }
}

// ============================================================================
// LEFT
// ============================================================================

/// LEFT function - returns the leftmost n characters from a string
#[derive(Default)]
pub struct LeftFunction;

impl ScalarFunction for LeftFunction {
    fn name(&self) -> &str {
        "LEFT"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "LEFT",
            FunctionType::Scalar,
            "Returns the leftmost n characters from a string",
            FunctionSignature::new(
                FunctionDataType::String,
                vec![FunctionDataType::Any, FunctionDataType::Integer],
                2,
                2,
            ),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "LEFT", 2);

        // SQL standard: any NULL argument returns NULL
        if args[0].is_null() || args[1].is_null() {
            return Ok(Value::null_unknown());
        }

        let s = value_to_string(&args[0]);
        let n = value_to_i64(&args[1])
            .ok_or_else(|| Error::invalid_argument("LEFT length must be an integer"))?;

        if n < 0 {
            return Ok(Value::Text(Arc::from("")));
        }

        let result: String = s.chars().take(n as usize).collect();
        Ok(Value::Text(Arc::from(result.as_str())))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(LeftFunction)
    }
}

// ============================================================================
// RIGHT
// ============================================================================

/// RIGHT function - returns the rightmost n characters from a string
#[derive(Default)]
pub struct RightFunction;

impl ScalarFunction for RightFunction {
    fn name(&self) -> &str {
        "RIGHT"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "RIGHT",
            FunctionType::Scalar,
            "Returns the rightmost n characters from a string",
            FunctionSignature::new(
                FunctionDataType::String,
                vec![FunctionDataType::Any, FunctionDataType::Integer],
                2,
                2,
            ),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "RIGHT", 2);

        // SQL standard: any NULL argument returns NULL
        if args[0].is_null() || args[1].is_null() {
            return Ok(Value::null_unknown());
        }

        let s = value_to_string(&args[0]);
        let n = value_to_i64(&args[1])
            .ok_or_else(|| Error::invalid_argument("RIGHT length must be an integer"))?;

        if n < 0 {
            return Ok(Value::Text(Arc::from("")));
        }

        let chars: Vec<char> = s.chars().collect();
        let len = chars.len();
        let start = len.saturating_sub(n as usize);
        let result: String = chars[start..].iter().collect();
        Ok(Value::Text(Arc::from(result.as_str())))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(RightFunction)
    }
}

// ============================================================================
// REPEAT
// ============================================================================

/// REPEAT function - repeats a string n times
#[derive(Default)]
pub struct RepeatFunction;

impl ScalarFunction for RepeatFunction {
    fn name(&self) -> &str {
        "REPEAT"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "REPEAT",
            FunctionType::Scalar,
            "Repeats a string n times",
            FunctionSignature::new(
                FunctionDataType::String,
                vec![FunctionDataType::Any, FunctionDataType::Integer],
                2,
                2,
            ),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "REPEAT", 2);

        // SQL standard: any NULL argument returns NULL
        if args[0].is_null() || args[1].is_null() {
            return Ok(Value::null_unknown());
        }

        let s = value_to_string(&args[0]);
        let n = value_to_i64(&args[1])
            .ok_or_else(|| Error::invalid_argument("REPEAT count must be an integer"))?;

        if n < 0 {
            return Ok(Value::Text(Arc::from("")));
        }

        // Limit to prevent memory exhaustion (max 10MB result)
        const MAX_RESULT_SIZE: usize = 10 * 1024 * 1024;
        let result_size = s.len().saturating_mul(n as usize);
        if result_size > MAX_RESULT_SIZE {
            return Err(Error::invalid_argument(format!(
                "REPEAT would produce {} bytes, exceeding maximum of {} bytes",
                result_size, MAX_RESULT_SIZE
            )));
        }

        let result = s.repeat(n as usize);
        Ok(Value::Text(Arc::from(result.as_str())))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(RepeatFunction)
    }
}

// ============================================================================
// SPLIT_PART
// ============================================================================

/// SPLIT_PART function - splits a string by delimiter and returns the nth part (1-indexed)
#[derive(Default)]
pub struct SplitPartFunction;

impl ScalarFunction for SplitPartFunction {
    fn name(&self) -> &str {
        "SPLIT_PART"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "SPLIT_PART",
            FunctionType::Scalar,
            "Splits a string by delimiter and returns the nth part (1-indexed)",
            FunctionSignature::new(
                FunctionDataType::String,
                vec![
                    FunctionDataType::Any,
                    FunctionDataType::Any,
                    FunctionDataType::Integer,
                ],
                3,
                3,
            ),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "SPLIT_PART", 3);

        // SQL standard: any NULL argument returns NULL
        if args[0].is_null() || args[1].is_null() || args[2].is_null() {
            return Ok(Value::null_unknown());
        }

        let s = value_to_string(&args[0]);
        let delimiter = value_to_string(&args[1]);
        let n = value_to_i64(&args[2])
            .ok_or_else(|| Error::invalid_argument("SPLIT_PART position must be an integer"))?;

        if n <= 0 {
            return Err(Error::invalid_argument(
                "SPLIT_PART position must be positive",
            ));
        }

        let parts: Vec<&str> = s.split(&delimiter).collect();
        let idx = (n - 1) as usize;

        if idx >= parts.len() {
            return Ok(Value::Text(Arc::from("")));
        }

        Ok(Value::Text(Arc::from(parts[idx])))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(SplitPartFunction)
    }
}

// ============================================================================
// POSITION / INSTR
// ============================================================================

/// POSITION function - returns the position of the first occurrence of a substring (1-indexed)
#[derive(Default)]
pub struct PositionFunction;

impl ScalarFunction for PositionFunction {
    fn name(&self) -> &str {
        "POSITION"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "POSITION",
            FunctionType::Scalar,
            "Returns the position of the first occurrence of a substring (1-indexed, 0 if not found)",
            FunctionSignature::new(
                FunctionDataType::Integer,
                vec![FunctionDataType::Any, FunctionDataType::Any],
                2,
                2,
            ),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "POSITION", 2);

        if args[0].is_null() || args[1].is_null() {
            return Ok(Value::null_unknown());
        }

        let substring = value_to_string(&args[0]);
        let s = value_to_string(&args[1]);

        // Find position (1-indexed), return 0 if not found
        match s.find(&substring) {
            Some(pos) => {
                // Convert byte position to char position (1-indexed)
                let char_pos = s[..pos].chars().count() + 1;
                Ok(Value::Integer(char_pos as i64))
            }
            None => Ok(Value::Integer(0)),
        }
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(PositionFunction)
    }
}

/// INSTR function - alias for POSITION with arguments reversed (string, substring)
#[derive(Default)]
pub struct InstrFunction;

impl ScalarFunction for InstrFunction {
    fn name(&self) -> &str {
        "INSTR"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "INSTR",
            FunctionType::Scalar,
            "Returns the position of the first occurrence of a substring (1-indexed, 0 if not found)",
            FunctionSignature::new(
                FunctionDataType::Integer,
                vec![FunctionDataType::Any, FunctionDataType::Any],
                2,
                2,
            ),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "INSTR", 2);

        if args[0].is_null() || args[1].is_null() {
            return Ok(Value::null_unknown());
        }

        let s = value_to_string(&args[0]);
        let substring = value_to_string(&args[1]);

        // Find position (1-indexed), return 0 if not found
        match s.find(&substring) {
            Some(pos) => {
                // Convert byte position to char position (1-indexed)
                let char_pos = s[..pos].chars().count() + 1;
                Ok(Value::Integer(char_pos as i64))
            }
            None => Ok(Value::Integer(0)),
        }
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(InstrFunction)
    }
}

// ============================================================================
// LOCATE
// ============================================================================

/// LOCATE function - returns the position of a substring in a string
/// LOCATE(substring, string [, start_position])
/// MySQL-compatible function with substring first, then string
#[derive(Default)]
pub struct LocateFunction;

impl ScalarFunction for LocateFunction {
    fn name(&self) -> &str {
        "LOCATE"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "LOCATE",
            FunctionType::Scalar,
            "Returns the position of the first occurrence of a substring (1-indexed, 0 if not found)",
            FunctionSignature::new(
                FunctionDataType::Integer,
                vec![
                    FunctionDataType::Any,
                    FunctionDataType::Any,
                    FunctionDataType::Integer,
                ],
                2,
                3,
            ),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "LOCATE", 2, 3);

        if args[0].is_null() || args[1].is_null() {
            return Ok(Value::null_unknown());
        }

        let substring = value_to_string(&args[0]);
        let s = value_to_string(&args[1]);

        // Handle optional start position (1-indexed)
        let start_pos = if args.len() == 3 {
            if args[2].is_null() {
                return Ok(Value::null_unknown());
            }
            let pos = value_to_i64(&args[2]).unwrap_or(1);
            if pos < 1 {
                1usize
            } else {
                pos as usize
            }
        } else {
            1
        };

        // Convert start_pos from 1-indexed character position to byte offset
        if start_pos > 1 {
            // Find the byte offset for the character position
            let char_offset = start_pos - 1;
            let byte_offset: usize = s.chars().take(char_offset).map(|c| c.len_utf8()).sum();

            if byte_offset >= s.len() {
                return Ok(Value::Integer(0));
            }

            // Search in the substring starting from byte_offset
            match s[byte_offset..].find(&substring) {
                Some(pos) => {
                    // Convert byte position to char position, add the offset
                    let char_pos = s[byte_offset..byte_offset + pos].chars().count() + start_pos;
                    Ok(Value::Integer(char_pos as i64))
                }
                None => Ok(Value::Integer(0)),
            }
        } else {
            // Search from the beginning
            match s.find(&substring) {
                Some(pos) => {
                    // Convert byte position to char position (1-indexed)
                    let char_pos = s[..pos].chars().count() + 1;
                    Ok(Value::Integer(char_pos as i64))
                }
                None => Ok(Value::Integer(0)),
            }
        }
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(LocateFunction)
    }
}

// ============================================================================
// LPAD
// ============================================================================

/// LPAD function - left-pads a string to a specified length
#[derive(Default)]
pub struct LpadFunction;

impl ScalarFunction for LpadFunction {
    fn name(&self) -> &str {
        "LPAD"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "LPAD",
            FunctionType::Scalar,
            "Left-pads a string to a specified length",
            FunctionSignature::new(
                FunctionDataType::String,
                vec![
                    FunctionDataType::Any,
                    FunctionDataType::Integer,
                    FunctionDataType::Any,
                ],
                2,
                3,
            ),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "LPAD", 2, 3);

        // SQL standard: NULL in required args returns NULL
        if args[0].is_null() || args[1].is_null() {
            return Ok(Value::null_unknown());
        }
        // Third argument (pad string) being NULL also returns NULL
        if args.len() == 3 && args[2].is_null() {
            return Ok(Value::null_unknown());
        }

        let s = value_to_string(&args[0]);
        let len = value_to_i64(&args[1])
            .ok_or_else(|| Error::invalid_argument("LPAD length must be an integer"))?;

        if len < 0 {
            return Ok(Value::Text(Arc::from("")));
        }

        let target_len = len as usize;
        let pad_str = if args.len() == 3 {
            value_to_string(&args[2])
        } else {
            " ".to_string()
        };

        let current_len = s.chars().count();
        if current_len >= target_len {
            let result: String = s.chars().take(target_len).collect();
            return Ok(Value::Text(Arc::from(result.as_str())));
        }

        let pad_needed = target_len - current_len;
        let mut result = String::with_capacity(target_len);

        // Build padding
        let pad_chars: Vec<char> = pad_str.chars().collect();
        if !pad_chars.is_empty() {
            for pad_idx in 0..pad_needed {
                result.push(pad_chars[pad_idx % pad_chars.len()]);
            }
        }

        result.push_str(&s);
        Ok(Value::Text(Arc::from(result.as_str())))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(LpadFunction)
    }
}

// ============================================================================
// RPAD
// ============================================================================

/// RPAD function - right-pads a string to a specified length
#[derive(Default)]
pub struct RpadFunction;

impl ScalarFunction for RpadFunction {
    fn name(&self) -> &str {
        "RPAD"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "RPAD",
            FunctionType::Scalar,
            "Right-pads a string to a specified length",
            FunctionSignature::new(
                FunctionDataType::String,
                vec![
                    FunctionDataType::Any,
                    FunctionDataType::Integer,
                    FunctionDataType::Any,
                ],
                2,
                3,
            ),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "RPAD", 2, 3);

        // SQL standard: NULL in required args returns NULL
        if args[0].is_null() || args[1].is_null() {
            return Ok(Value::null_unknown());
        }
        // Third argument (pad string) being NULL also returns NULL
        if args.len() == 3 && args[2].is_null() {
            return Ok(Value::null_unknown());
        }

        let s = value_to_string(&args[0]);
        let len = value_to_i64(&args[1])
            .ok_or_else(|| Error::invalid_argument("RPAD length must be an integer"))?;

        if len < 0 {
            return Ok(Value::Text(Arc::from("")));
        }

        let target_len = len as usize;
        let pad_str = if args.len() == 3 {
            value_to_string(&args[2])
        } else {
            " ".to_string()
        };

        let current_len = s.chars().count();
        if current_len >= target_len {
            let result: String = s.chars().take(target_len).collect();
            return Ok(Value::Text(Arc::from(result.as_str())));
        }

        let pad_needed = target_len - current_len;
        // Pre-allocate with capacity to avoid reallocations
        let mut result = String::with_capacity(s.len() + pad_needed);
        result.push_str(&s);

        // Build padding
        let pad_chars: Vec<char> = pad_str.chars().collect();
        if !pad_chars.is_empty() {
            for pad_idx in 0..pad_needed {
                result.push(pad_chars[pad_idx % pad_chars.len()]);
            }
        }

        Ok(Value::Text(Arc::from(result.as_str())))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(RpadFunction)
    }
}

// ============================================================================
// CHAR - Convert ASCII code to character
// ============================================================================

/// CHAR function - returns the character for an ASCII code
#[derive(Default)]
pub struct CharFunction;

impl ScalarFunction for CharFunction {
    fn name(&self) -> &str {
        "CHAR"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "CHAR",
            FunctionType::Scalar,
            "Returns the character for an ASCII code",
            FunctionSignature::new(
                FunctionDataType::String,
                vec![FunctionDataType::Integer],
                1,
                1,
            ),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "CHAR", 1);

        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }

        let code = match &args[0] {
            Value::Integer(n) => *n,
            Value::Float(f) => *f as i64,
            _ => {
                return Err(Error::invalid_argument("CHAR requires an integer argument"));
            }
        };

        // Convert ASCII/Unicode code point to character
        if !(0..=0x10FFFF).contains(&code) {
            return Ok(Value::null_unknown());
        }

        match char::from_u32(code as u32) {
            Some(c) => Ok(Value::Text(Arc::from(c.to_string().as_str()))),
            None => Ok(Value::null_unknown()),
        }
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(CharFunction)
    }
}

// ============================================================================
// CHAR_LENGTH / CHARACTER_LENGTH
// ============================================================================

/// CHAR_LENGTH function - returns the number of characters in a string (alias for LENGTH)
#[derive(Default)]
pub struct CharLengthFunction;

impl ScalarFunction for CharLengthFunction {
    fn name(&self) -> &str {
        "CHAR_LENGTH"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "CHAR_LENGTH",
            FunctionType::Scalar,
            "Returns the number of characters in a string",
            FunctionSignature::new(FunctionDataType::Integer, vec![FunctionDataType::Any], 1, 1),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "CHAR_LENGTH", 1);

        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }

        let s = value_to_string(&args[0]);
        Ok(Value::Integer(s.chars().count() as i64))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(CharLengthFunction)
    }
}

// ============================================================================
// CONCAT_WS - Concatenate With Separator
// ============================================================================

/// CONCAT_WS function - concatenates strings with a separator
#[derive(Default)]
pub struct ConcatWsFunction;

impl ScalarFunction for ConcatWsFunction {
    fn name(&self) -> &str {
        "CONCAT_WS"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "CONCAT_WS",
            FunctionType::Scalar,
            "Concatenates strings with a separator (first argument is separator)",
            FunctionSignature::variadic(FunctionDataType::String, FunctionDataType::Any),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        if args.len() < 2 {
            return Err(Error::invalid_argument(
                "CONCAT_WS requires at least 2 arguments (separator and at least one string)",
            ));
        }

        // First argument is the separator - if NULL, return NULL
        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }

        let separator = value_to_string(&args[0]);

        // Collect non-NULL values
        let parts: Vec<String> = args[1..]
            .iter()
            .filter(|arg| !arg.is_null())
            .map(value_to_string)
            .collect();

        Ok(Value::Text(Arc::from(parts.join(&separator).as_str())))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(ConcatWsFunction)
    }
}

// ============================================================================
// STRPOS - String Position (PostgreSQL style)
// ============================================================================

/// STRPOS function - returns the position of a substring in a string (1-indexed)
/// This is the PostgreSQL-style function: STRPOS(string, substring)
#[derive(Default)]
pub struct StrposFunction;

impl ScalarFunction for StrposFunction {
    fn name(&self) -> &str {
        "STRPOS"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "STRPOS",
            FunctionType::Scalar,
            "Returns the position of a substring in a string (1-indexed, 0 if not found)",
            FunctionSignature::new(
                FunctionDataType::Integer,
                vec![FunctionDataType::Any, FunctionDataType::Any],
                2,
                2,
            ),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "STRPOS", 2);

        if args[0].is_null() || args[1].is_null() {
            return Ok(Value::null_unknown());
        }

        let s = value_to_string(&args[0]);
        let substring = value_to_string(&args[1]);

        // Find position (1-indexed), return 0 if not found
        match s.find(&substring) {
            Some(pos) => {
                // Convert byte position to char position (1-indexed)
                let char_pos = s[..pos].chars().count() + 1;
                Ok(Value::Integer(char_pos as i64))
            }
            None => Ok(Value::Integer(0)),
        }
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(StrposFunction)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_upper() {
        let f = UpperFunction;
        assert_eq!(
            f.evaluate(&[Value::text("hello")]).unwrap(),
            Value::text("HELLO")
        );
        assert_eq!(
            f.evaluate(&[Value::text("HeLLo WoRLd")]).unwrap(),
            Value::text("HELLO WORLD")
        );
        assert!(f.evaluate(&[Value::null_unknown()]).unwrap().is_null());
    }

    #[test]
    fn test_lower() {
        let f = LowerFunction;
        assert_eq!(
            f.evaluate(&[Value::text("HELLO")]).unwrap(),
            Value::text("hello")
        );
        assert_eq!(
            f.evaluate(&[Value::text("HeLLo WoRLd")]).unwrap(),
            Value::text("hello world")
        );
        assert!(f.evaluate(&[Value::null_unknown()]).unwrap().is_null());
    }

    #[test]
    fn test_length() {
        let f = LengthFunction;
        assert_eq!(
            f.evaluate(&[Value::text("hello")]).unwrap(),
            Value::Integer(5)
        );
        assert_eq!(f.evaluate(&[Value::text("")]).unwrap(), Value::Integer(0));
        assert!(f.evaluate(&[Value::null_unknown()]).unwrap().is_null());
        // Unicode characters
        assert_eq!(
            f.evaluate(&[Value::text("héllo")]).unwrap(),
            Value::Integer(5)
        );
    }

    #[test]
    fn test_concat() {
        let f = ConcatFunction;
        assert_eq!(
            f.evaluate(&[Value::text("hello"), Value::text(" "), Value::text("world")])
                .unwrap(),
            Value::text("hello world")
        );
        // NULL values are ignored
        assert_eq!(
            f.evaluate(&[
                Value::text("hello"),
                Value::null_unknown(),
                Value::text("world")
            ])
            .unwrap(),
            Value::text("helloworld")
        );
        // Mixed types
        assert_eq!(
            f.evaluate(&[Value::text("count: "), Value::Integer(42)])
                .unwrap(),
            Value::text("count: 42")
        );
    }

    #[test]
    fn test_substring() {
        let f = SubstringFunction;
        // Basic extraction
        assert_eq!(
            f.evaluate(&[
                Value::text("hello world"),
                Value::Integer(1),
                Value::Integer(5)
            ])
            .unwrap(),
            Value::text("hello")
        );
        // From middle
        assert_eq!(
            f.evaluate(&[
                Value::text("hello world"),
                Value::Integer(7),
                Value::Integer(5)
            ])
            .unwrap(),
            Value::text("world")
        );
        // Without length (to end)
        assert_eq!(
            f.evaluate(&[Value::text("hello world"), Value::Integer(7)])
                .unwrap(),
            Value::text("world")
        );
        // Start beyond string length
        assert_eq!(
            f.evaluate(&[Value::text("hello"), Value::Integer(100), Value::Integer(5)])
                .unwrap(),
            Value::text("")
        );
        // NULL input
        assert!(f
            .evaluate(&[Value::null_unknown(), Value::Integer(1), Value::Integer(5)])
            .unwrap()
            .is_null());
    }
}
