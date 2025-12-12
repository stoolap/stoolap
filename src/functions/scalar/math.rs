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

//! Math scalar functions

use crate::core::{Error, Result, Value};
use crate::functions::{
    FunctionDataType, FunctionInfo, FunctionSignature, FunctionType, ScalarFunction,
};
use crate::validate_arg_count;
use rand::Rng;

use super::{value_to_f64, value_to_i64};

// ============================================================================
// ABS
// ============================================================================

/// ABS function - returns the absolute value of a number
#[derive(Default)]
pub struct AbsFunction;

impl ScalarFunction for AbsFunction {
    fn name(&self) -> &str {
        "ABS"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "ABS",
            FunctionType::Scalar,
            "Returns the absolute value of a number",
            FunctionSignature::new(FunctionDataType::Float, vec![FunctionDataType::Any], 1, 1),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "ABS", 1);

        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }

        // Try to preserve integer type if possible
        if let Some(i) = value_to_i64(&args[0]) {
            if matches!(args[0], Value::Integer(_)) {
                return Ok(Value::Integer(i.abs()));
            }
        }

        let num = value_to_f64(&args[0])
            .ok_or_else(|| Error::invalid_argument("ABS argument must be a number"))?;

        Ok(Value::Float(num.abs()))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(AbsFunction)
    }
}

// ============================================================================
// ROUND
// ============================================================================

/// ROUND function - rounds a number to a specified number of decimal places
#[derive(Default)]
pub struct RoundFunction;

impl ScalarFunction for RoundFunction {
    fn name(&self) -> &str {
        "ROUND"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "ROUND",
            FunctionType::Scalar,
            "Rounds a number to a specified number of decimal places",
            FunctionSignature::new(
                FunctionDataType::Float,
                vec![FunctionDataType::Any, FunctionDataType::Integer],
                1,
                2,
            ),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "ROUND", 1, 2);

        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }

        let num = value_to_f64(&args[0])
            .ok_or_else(|| Error::invalid_argument("ROUND first argument must be a number"))?;

        // Default to 0 decimal places if not specified
        let places = if args.len() == 2 && !args[1].is_null() {
            value_to_i64(&args[1])
                .ok_or_else(|| Error::invalid_argument("ROUND decimal places must be an integer"))?
                as i32
        } else {
            0
        };

        // Round to specified decimal places
        let shift = 10_f64.powi(places);
        let rounded = (num * shift).round() / shift;

        Ok(Value::Float(rounded))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(RoundFunction)
    }
}

// ============================================================================
// FLOOR
// ============================================================================

/// FLOOR function - returns the largest integer value not greater than the argument
#[derive(Default)]
pub struct FloorFunction;

impl ScalarFunction for FloorFunction {
    fn name(&self) -> &str {
        "FLOOR"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "FLOOR",
            FunctionType::Scalar,
            "Returns the largest integer value not greater than the argument",
            FunctionSignature::new(FunctionDataType::Float, vec![FunctionDataType::Any], 1, 1),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "FLOOR", 1);

        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }

        let num = value_to_f64(&args[0])
            .ok_or_else(|| Error::invalid_argument("FLOOR argument must be a number"))?;

        Ok(Value::Float(num.floor()))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(FloorFunction)
    }
}

// ============================================================================
// CEILING
// ============================================================================

/// CEILING function - returns the smallest integer value not less than the argument
#[derive(Default)]
pub struct CeilingFunction;

impl ScalarFunction for CeilingFunction {
    fn name(&self) -> &str {
        "CEILING"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "CEILING",
            FunctionType::Scalar,
            "Returns the smallest integer value not less than the argument",
            FunctionSignature::new(FunctionDataType::Float, vec![FunctionDataType::Any], 1, 1),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "CEILING", 1);

        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }

        let num = value_to_f64(&args[0])
            .ok_or_else(|| Error::invalid_argument("CEILING argument must be a number"))?;

        Ok(Value::Float(num.ceil()))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(CeilingFunction)
    }
}

// ============================================================================
// CEIL (alias for CEILING)
// ============================================================================

/// CEIL function - alias for CEILING, returns the smallest integer value not less than the argument
#[derive(Default)]
pub struct CeilFunction;

impl ScalarFunction for CeilFunction {
    fn name(&self) -> &str {
        "CEIL"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "CEIL",
            FunctionType::Scalar,
            "Returns the smallest integer value not less than the argument (alias for CEILING)",
            FunctionSignature::new(FunctionDataType::Float, vec![FunctionDataType::Any], 1, 1),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "CEIL", 1);

        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }

        let num = value_to_f64(&args[0])
            .ok_or_else(|| Error::invalid_argument("CEIL argument must be a number"))?;

        Ok(Value::Float(num.ceil()))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(CeilFunction)
    }
}

// ============================================================================
// MOD
// ============================================================================

/// MOD function - returns the remainder of a division
#[derive(Default)]
pub struct ModFunction;

impl ScalarFunction for ModFunction {
    fn name(&self) -> &str {
        "MOD"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "MOD",
            FunctionType::Scalar,
            "Returns the remainder of a division (modulo)",
            FunctionSignature::new(
                FunctionDataType::Integer,
                vec![FunctionDataType::Any, FunctionDataType::Any],
                2,
                2,
            ),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "MOD", 2);

        if args[0].is_null() || args[1].is_null() {
            return Ok(Value::null_unknown());
        }

        // Try integer modulo first
        if let (Some(a), Some(b)) = (value_to_i64(&args[0]), value_to_i64(&args[1])) {
            if b == 0 {
                return Err(Error::invalid_argument("MOD: division by zero"));
            }
            return Ok(Value::Integer(a % b));
        }

        // Fall back to float modulo
        let a = value_to_f64(&args[0])
            .ok_or_else(|| Error::invalid_argument("MOD first argument must be a number"))?;
        let b = value_to_f64(&args[1])
            .ok_or_else(|| Error::invalid_argument("MOD second argument must be a number"))?;

        if b == 0.0 {
            return Err(Error::invalid_argument("MOD: division by zero"));
        }

        Ok(Value::Float(a % b))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(ModFunction)
    }
}

// ============================================================================
// POWER / POW
// ============================================================================

/// POWER function - returns base raised to the power of exponent
#[derive(Default)]
pub struct PowerFunction;

impl ScalarFunction for PowerFunction {
    fn name(&self) -> &str {
        "POWER"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "POWER",
            FunctionType::Scalar,
            "Returns base raised to the power of exponent",
            FunctionSignature::new(
                FunctionDataType::Float,
                vec![FunctionDataType::Any, FunctionDataType::Any],
                2,
                2,
            ),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "POWER", 2);

        if args[0].is_null() || args[1].is_null() {
            return Ok(Value::null_unknown());
        }

        let base = value_to_f64(&args[0])
            .ok_or_else(|| Error::invalid_argument("POWER base must be a number"))?;
        let exp = value_to_f64(&args[1])
            .ok_or_else(|| Error::invalid_argument("POWER exponent must be a number"))?;

        Ok(Value::Float(base.powf(exp)))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(PowerFunction)
    }
}

/// POW function - alias for POWER
#[derive(Default)]
pub struct PowFunction;

impl ScalarFunction for PowFunction {
    fn name(&self) -> &str {
        "POW"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "POW",
            FunctionType::Scalar,
            "Returns base raised to the power of exponent (alias for POWER)",
            FunctionSignature::new(
                FunctionDataType::Float,
                vec![FunctionDataType::Any, FunctionDataType::Any],
                2,
                2,
            ),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        PowerFunction.evaluate(args)
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(PowFunction)
    }
}

// ============================================================================
// SQRT
// ============================================================================

/// SQRT function - returns the square root of a number
#[derive(Default)]
pub struct SqrtFunction;

impl ScalarFunction for SqrtFunction {
    fn name(&self) -> &str {
        "SQRT"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "SQRT",
            FunctionType::Scalar,
            "Returns the square root of a number",
            FunctionSignature::new(FunctionDataType::Float, vec![FunctionDataType::Any], 1, 1),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "SQRT", 1);

        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }

        let num = value_to_f64(&args[0])
            .ok_or_else(|| Error::invalid_argument("SQRT argument must be a number"))?;

        if num < 0.0 {
            return Ok(Value::null_unknown()); // SQL standard returns NULL for negative input
        }

        Ok(Value::Float(num.sqrt()))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(SqrtFunction)
    }
}

// ============================================================================
// LOG (base 10)
// ============================================================================

/// LOG function - returns the base-10 logarithm of a number
#[derive(Default)]
pub struct LogFunction;

impl ScalarFunction for LogFunction {
    fn name(&self) -> &str {
        "LOG"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "LOG",
            FunctionType::Scalar,
            "Returns the base-10 logarithm of a number (or log with custom base)",
            FunctionSignature::new(
                FunctionDataType::Float,
                vec![FunctionDataType::Any, FunctionDataType::Any],
                1,
                2,
            ),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "LOG", 1, 2);

        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }

        if args.len() == 1 {
            // LOG(x) - base 10
            let num = value_to_f64(&args[0])
                .ok_or_else(|| Error::invalid_argument("LOG argument must be a number"))?;

            if num <= 0.0 {
                return Ok(Value::null_unknown());
            }

            Ok(Value::Float(num.log10()))
        } else {
            // LOG(base, x)
            if args[1].is_null() {
                return Ok(Value::null_unknown());
            }

            let base = value_to_f64(&args[0])
                .ok_or_else(|| Error::invalid_argument("LOG base must be a number"))?;
            let num = value_to_f64(&args[1])
                .ok_or_else(|| Error::invalid_argument("LOG argument must be a number"))?;

            if base <= 0.0 || base == 1.0 || num <= 0.0 {
                return Ok(Value::null_unknown());
            }

            Ok(Value::Float(num.log(base)))
        }
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(LogFunction)
    }
}

/// LOG10 function - returns the base-10 logarithm of a number
#[derive(Default)]
pub struct Log10Function;

impl ScalarFunction for Log10Function {
    fn name(&self) -> &str {
        "LOG10"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "LOG10",
            FunctionType::Scalar,
            "Returns the base-10 logarithm of a number",
            FunctionSignature::new(FunctionDataType::Float, vec![FunctionDataType::Any], 1, 1),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "LOG10", 1);

        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }

        let num = value_to_f64(&args[0])
            .ok_or_else(|| Error::invalid_argument("LOG10 argument must be a number"))?;

        if num <= 0.0 {
            return Ok(Value::null_unknown());
        }

        Ok(Value::Float(num.log10()))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(Log10Function)
    }
}

/// LOG2 function - returns the base-2 logarithm of a number
#[derive(Default)]
pub struct Log2Function;

impl ScalarFunction for Log2Function {
    fn name(&self) -> &str {
        "LOG2"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "LOG2",
            FunctionType::Scalar,
            "Returns the base-2 logarithm of a number",
            FunctionSignature::new(FunctionDataType::Float, vec![FunctionDataType::Any], 1, 1),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "LOG2", 1);

        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }

        let num = value_to_f64(&args[0])
            .ok_or_else(|| Error::invalid_argument("LOG2 argument must be a number"))?;

        if num <= 0.0 {
            return Ok(Value::null_unknown());
        }

        Ok(Value::Float(num.log2()))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(Log2Function)
    }
}

// ============================================================================
// LN (natural logarithm)
// ============================================================================

/// LN function - returns the natural logarithm of a number
#[derive(Default)]
pub struct LnFunction;

impl ScalarFunction for LnFunction {
    fn name(&self) -> &str {
        "LN"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "LN",
            FunctionType::Scalar,
            "Returns the natural logarithm (base e) of a number",
            FunctionSignature::new(FunctionDataType::Float, vec![FunctionDataType::Any], 1, 1),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "LN", 1);

        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }

        let num = value_to_f64(&args[0])
            .ok_or_else(|| Error::invalid_argument("LN argument must be a number"))?;

        if num <= 0.0 {
            return Ok(Value::null_unknown());
        }

        Ok(Value::Float(num.ln()))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(LnFunction)
    }
}

// ============================================================================
// EXP
// ============================================================================

/// EXP function - returns e raised to the power of the argument
#[derive(Default)]
pub struct ExpFunction;

impl ScalarFunction for ExpFunction {
    fn name(&self) -> &str {
        "EXP"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "EXP",
            FunctionType::Scalar,
            "Returns e raised to the power of the argument",
            FunctionSignature::new(FunctionDataType::Float, vec![FunctionDataType::Any], 1, 1),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "EXP", 1);

        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }

        let num = value_to_f64(&args[0])
            .ok_or_else(|| Error::invalid_argument("EXP argument must be a number"))?;

        Ok(Value::Float(num.exp()))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(ExpFunction)
    }
}

// ============================================================================
// SIGN
// ============================================================================

/// SIGN function - returns the sign of a number (-1, 0, or 1)
#[derive(Default)]
pub struct SignFunction;

impl ScalarFunction for SignFunction {
    fn name(&self) -> &str {
        "SIGN"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "SIGN",
            FunctionType::Scalar,
            "Returns the sign of a number (-1, 0, or 1)",
            FunctionSignature::new(FunctionDataType::Integer, vec![FunctionDataType::Any], 1, 1),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "SIGN", 1);

        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }

        let num = value_to_f64(&args[0])
            .ok_or_else(|| Error::invalid_argument("SIGN argument must be a number"))?;

        let sign = if num > 0.0 {
            1
        } else if num < 0.0 {
            -1
        } else {
            0
        };

        Ok(Value::Integer(sign))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(SignFunction)
    }
}

// ============================================================================
// TRUNCATE / TRUNC
// ============================================================================

/// TRUNCATE function - truncates a number to a specified number of decimal places
#[derive(Default)]
pub struct TruncateFunction;

impl ScalarFunction for TruncateFunction {
    fn name(&self) -> &str {
        "TRUNCATE"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "TRUNCATE",
            FunctionType::Scalar,
            "Truncates a number to a specified number of decimal places",
            FunctionSignature::new(
                FunctionDataType::Float,
                vec![FunctionDataType::Any, FunctionDataType::Integer],
                1,
                2,
            ),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "TRUNCATE", 1, 2);

        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }

        let num = value_to_f64(&args[0])
            .ok_or_else(|| Error::invalid_argument("TRUNCATE argument must be a number"))?;

        let places = if args.len() == 2 && !args[1].is_null() {
            value_to_i64(&args[1]).ok_or_else(|| {
                Error::invalid_argument("TRUNCATE decimal places must be an integer")
            })? as i32
        } else {
            0
        };

        let shift = 10_f64.powi(places);
        let truncated = (num * shift).trunc() / shift;

        Ok(Value::Float(truncated))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(TruncateFunction)
    }
}

/// TRUNC function - alias for TRUNCATE
#[derive(Default)]
pub struct TruncFunction;

impl ScalarFunction for TruncFunction {
    fn name(&self) -> &str {
        "TRUNC"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "TRUNC",
            FunctionType::Scalar,
            "Truncates a number to a specified number of decimal places (alias for TRUNCATE)",
            FunctionSignature::new(
                FunctionDataType::Float,
                vec![FunctionDataType::Any, FunctionDataType::Integer],
                1,
                2,
            ),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        TruncateFunction.evaluate(args)
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(TruncFunction)
    }
}

// ============================================================================
// PI
// ============================================================================

/// PI function - returns the value of pi
#[derive(Default)]
pub struct PiFunction;

impl ScalarFunction for PiFunction {
    fn name(&self) -> &str {
        "PI"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "PI",
            FunctionType::Scalar,
            "Returns the value of pi",
            FunctionSignature::new(FunctionDataType::Float, vec![], 0, 0),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        if !args.is_empty() {
            return Err(Error::invalid_argument(format!(
                "PI requires no arguments, got {}",
                args.len()
            )));
        }

        Ok(Value::Float(std::f64::consts::PI))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(PiFunction)
    }
}

// ============================================================================
// RANDOM / RAND
// ============================================================================

/// RANDOM function - returns a random float between 0 and 1
#[derive(Default)]
pub struct RandomFunction;

impl ScalarFunction for RandomFunction {
    fn name(&self) -> &str {
        "RANDOM"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "RANDOM",
            FunctionType::Scalar,
            "Returns a random float between 0 (inclusive) and 1 (exclusive)",
            FunctionSignature::new(FunctionDataType::Float, vec![], 0, 0),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        if !args.is_empty() {
            return Err(Error::invalid_argument(format!(
                "RANDOM requires no arguments, got {}",
                args.len()
            )));
        }

        // Use thread-local RNG for proper randomness
        let random = rand::rng().random::<f64>();

        Ok(Value::Float(random))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(RandomFunction)
    }
}

// ============================================================================
// SIN
// ============================================================================

/// SIN function - returns the sine of an angle (in radians)
#[derive(Default)]
pub struct SinFunction;

impl ScalarFunction for SinFunction {
    fn name(&self) -> &str {
        "SIN"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "SIN",
            FunctionType::Scalar,
            "Returns the sine of an angle (in radians)",
            FunctionSignature::new(FunctionDataType::Float, vec![FunctionDataType::Any], 1, 1),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "SIN", 1);

        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }

        let num = value_to_f64(&args[0])
            .ok_or_else(|| Error::invalid_argument("SIN argument must be a number"))?;

        Ok(Value::Float(num.sin()))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(SinFunction)
    }
}

// ============================================================================
// COS
// ============================================================================

/// COS function - returns the cosine of an angle (in radians)
#[derive(Default)]
pub struct CosFunction;

impl ScalarFunction for CosFunction {
    fn name(&self) -> &str {
        "COS"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "COS",
            FunctionType::Scalar,
            "Returns the cosine of an angle (in radians)",
            FunctionSignature::new(FunctionDataType::Float, vec![FunctionDataType::Any], 1, 1),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "COS", 1);

        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }

        let num = value_to_f64(&args[0])
            .ok_or_else(|| Error::invalid_argument("COS argument must be a number"))?;

        Ok(Value::Float(num.cos()))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(CosFunction)
    }
}

// ============================================================================
// TAN
// ============================================================================

/// TAN function - returns the tangent of an angle (in radians)
#[derive(Default)]
pub struct TanFunction;

impl ScalarFunction for TanFunction {
    fn name(&self) -> &str {
        "TAN"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "TAN",
            FunctionType::Scalar,
            "Returns the tangent of an angle (in radians)",
            FunctionSignature::new(FunctionDataType::Float, vec![FunctionDataType::Any], 1, 1),
        )
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        validate_arg_count!(args, "TAN", 1);

        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }

        let num = value_to_f64(&args[0])
            .ok_or_else(|| Error::invalid_argument("TAN argument must be a number"))?;

        Ok(Value::Float(num.tan()))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(TanFunction)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_abs_positive() {
        let f = AbsFunction;
        assert_eq!(
            f.evaluate(&[Value::Integer(42)]).unwrap(),
            Value::Integer(42)
        );
        assert_eq!(f.evaluate(&[Value::Float(3.5)]).unwrap(), Value::Float(3.5));
    }

    #[test]
    fn test_abs_negative() {
        let f = AbsFunction;
        assert_eq!(
            f.evaluate(&[Value::Integer(-42)]).unwrap(),
            Value::Integer(42)
        );
        assert_eq!(
            f.evaluate(&[Value::Float(-3.5)]).unwrap(),
            Value::Float(3.5)
        );
    }

    #[test]
    fn test_abs_null() {
        let f = AbsFunction;
        assert!(f.evaluate(&[Value::null_unknown()]).unwrap().is_null());
    }

    #[test]
    fn test_abs_zero() {
        let f = AbsFunction;
        assert_eq!(f.evaluate(&[Value::Integer(0)]).unwrap(), Value::Integer(0));
        assert_eq!(f.evaluate(&[Value::Float(0.0)]).unwrap(), Value::Float(0.0));
    }

    #[test]
    fn test_round_default() {
        let f = RoundFunction;
        assert_eq!(f.evaluate(&[Value::Float(3.7)]).unwrap(), Value::Float(4.0));
        assert_eq!(f.evaluate(&[Value::Float(3.2)]).unwrap(), Value::Float(3.0));
        assert_eq!(f.evaluate(&[Value::Float(3.5)]).unwrap(), Value::Float(4.0));
    }

    #[test]
    fn test_round_with_places() {
        let f = RoundFunction;
        assert_eq!(
            f.evaluate(&[Value::Float(3.54159), Value::Integer(2)])
                .unwrap(),
            Value::Float(3.54)
        );
        assert_eq!(
            f.evaluate(&[Value::Float(3.54159), Value::Integer(4)])
                .unwrap(),
            Value::Float(3.5416)
        );
        assert_eq!(
            f.evaluate(&[Value::Float(1234.5678), Value::Integer(0)])
                .unwrap(),
            Value::Float(1235.0)
        );
    }

    #[test]
    fn test_round_negative_places() {
        let f = RoundFunction;
        // Round to tens
        assert_eq!(
            f.evaluate(&[Value::Float(1234.5), Value::Integer(-1)])
                .unwrap(),
            Value::Float(1230.0)
        );
        // Round to hundreds
        assert_eq!(
            f.evaluate(&[Value::Float(1234.5), Value::Integer(-2)])
                .unwrap(),
            Value::Float(1200.0)
        );
    }

    #[test]
    fn test_round_null() {
        let f = RoundFunction;
        assert!(f.evaluate(&[Value::null_unknown()]).unwrap().is_null());
        assert_eq!(
            f.evaluate(&[Value::Float(3.5), Value::null_unknown()])
                .unwrap(),
            Value::Float(4.0)
        );
    }

    #[test]
    fn test_floor() {
        let f = FloorFunction;
        assert_eq!(f.evaluate(&[Value::Float(3.7)]).unwrap(), Value::Float(3.0));
        assert_eq!(f.evaluate(&[Value::Float(3.2)]).unwrap(), Value::Float(3.0));
        assert_eq!(
            f.evaluate(&[Value::Float(-3.2)]).unwrap(),
            Value::Float(-4.0)
        );
        assert_eq!(f.evaluate(&[Value::Integer(5)]).unwrap(), Value::Float(5.0));
    }

    #[test]
    fn test_floor_null() {
        let f = FloorFunction;
        assert!(f.evaluate(&[Value::null_unknown()]).unwrap().is_null());
    }

    #[test]
    fn test_ceiling() {
        let f = CeilingFunction;
        assert_eq!(f.evaluate(&[Value::Float(3.7)]).unwrap(), Value::Float(4.0));
        assert_eq!(f.evaluate(&[Value::Float(3.2)]).unwrap(), Value::Float(4.0));
        assert_eq!(
            f.evaluate(&[Value::Float(-3.2)]).unwrap(),
            Value::Float(-3.0)
        );
        assert_eq!(f.evaluate(&[Value::Integer(5)]).unwrap(), Value::Float(5.0));
    }

    #[test]
    fn test_ceiling_null() {
        let f = CeilingFunction;
        assert!(f.evaluate(&[Value::null_unknown()]).unwrap().is_null());
    }

    #[test]
    fn test_abs_string_number() {
        let f = AbsFunction;
        assert_eq!(
            f.evaluate(&[Value::text("-42.5")]).unwrap(),
            Value::Float(42.5)
        );
    }

    #[test]
    fn test_abs_invalid_arg_count() {
        let f = AbsFunction;
        assert!(f.evaluate(&[]).is_err());
        assert!(f.evaluate(&[Value::Integer(1), Value::Integer(2)]).is_err());
    }

    #[test]
    fn test_sin() {
        let f = SinFunction;
        // sin(0) = 0
        let result = f.evaluate(&[Value::Float(0.0)]).unwrap();
        if let Value::Float(v) = result {
            assert!((v - 0.0).abs() < 1e-10);
        } else {
            panic!("Expected Float");
        }
        // sin(π/2) = 1
        let result = f
            .evaluate(&[Value::Float(std::f64::consts::FRAC_PI_2)])
            .unwrap();
        if let Value::Float(v) = result {
            assert!((v - 1.0).abs() < 1e-10);
        } else {
            panic!("Expected Float");
        }
        // sin(π) ≈ 0
        let result = f.evaluate(&[Value::Float(std::f64::consts::PI)]).unwrap();
        if let Value::Float(v) = result {
            assert!(v.abs() < 1e-10);
        } else {
            panic!("Expected Float");
        }
    }

    #[test]
    fn test_sin_null() {
        let f = SinFunction;
        assert!(f.evaluate(&[Value::null_unknown()]).unwrap().is_null());
    }

    #[test]
    fn test_cos() {
        let f = CosFunction;
        // cos(0) = 1
        let result = f.evaluate(&[Value::Float(0.0)]).unwrap();
        if let Value::Float(v) = result {
            assert!((v - 1.0).abs() < 1e-10);
        } else {
            panic!("Expected Float");
        }
        // cos(π/2) ≈ 0
        let result = f
            .evaluate(&[Value::Float(std::f64::consts::FRAC_PI_2)])
            .unwrap();
        if let Value::Float(v) = result {
            assert!(v.abs() < 1e-10);
        } else {
            panic!("Expected Float");
        }
        // cos(π) = -1
        let result = f.evaluate(&[Value::Float(std::f64::consts::PI)]).unwrap();
        if let Value::Float(v) = result {
            assert!((v - (-1.0)).abs() < 1e-10);
        } else {
            panic!("Expected Float");
        }
    }

    #[test]
    fn test_cos_null() {
        let f = CosFunction;
        assert!(f.evaluate(&[Value::null_unknown()]).unwrap().is_null());
    }

    #[test]
    fn test_tan() {
        let f = TanFunction;
        // tan(0) = 0
        let result = f.evaluate(&[Value::Float(0.0)]).unwrap();
        if let Value::Float(v) = result {
            assert!((v - 0.0).abs() < 1e-10);
        } else {
            panic!("Expected Float");
        }
        // tan(π/4) = 1
        let result = f
            .evaluate(&[Value::Float(std::f64::consts::FRAC_PI_4)])
            .unwrap();
        if let Value::Float(v) = result {
            assert!((v - 1.0).abs() < 1e-10);
        } else {
            panic!("Expected Float");
        }
        // tan(π) ≈ 0
        let result = f.evaluate(&[Value::Float(std::f64::consts::PI)]).unwrap();
        if let Value::Float(v) = result {
            assert!(v.abs() < 1e-10);
        } else {
            panic!("Expected Float");
        }
    }

    #[test]
    fn test_tan_null() {
        let f = TanFunction;
        assert!(f.evaluate(&[Value::null_unknown()]).unwrap().is_null());
    }

    #[test]
    fn test_trig_with_integers() {
        // Test that integer inputs work correctly
        let sin_f = SinFunction;
        let cos_f = CosFunction;
        let tan_f = TanFunction;

        // sin(0) = 0
        assert!(sin_f.evaluate(&[Value::Integer(0)]).is_ok());
        // cos(0) = 1
        let result = cos_f.evaluate(&[Value::Integer(0)]).unwrap();
        if let Value::Float(v) = result {
            assert!((v - 1.0).abs() < 1e-10);
        }
        // tan(0) = 0
        assert!(tan_f.evaluate(&[Value::Integer(0)]).is_ok());
    }

    #[test]
    fn test_trig_invalid_args() {
        let sin_f = SinFunction;
        let cos_f = CosFunction;
        let tan_f = TanFunction;

        assert!(sin_f.evaluate(&[]).is_err());
        assert!(cos_f.evaluate(&[]).is_err());
        assert!(tan_f.evaluate(&[]).is_err());

        assert!(sin_f
            .evaluate(&[Value::Integer(1), Value::Integer(2)])
            .is_err());
        assert!(cos_f
            .evaluate(&[Value::Integer(1), Value::Integer(2)])
            .is_err());
        assert!(tan_f
            .evaluate(&[Value::Integer(1), Value::Integer(2)])
            .is_err());
    }
}
