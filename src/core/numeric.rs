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

//! Arithmetic on two values of either number type: which of `integer_op`
//! and `float_op` answers a pair, and the error an operand that is not a
//! number gives

use super::float::{float_op, FloatOp};
use super::integer::{integer_neg, integer_op, IntegerOp};
use super::value::NULL_VALUE;
use super::{DataType, Error, Result, Value};

/// `a op b` over two numbers, an INTEGER pair by `int_op`, any pair holding
/// a FLOAT by `float`; NULL for a zero divisor or a NULL operand, an error
/// for any other operand that is not a number
#[inline(always)]
pub fn numeric_op(a: &Value, b: &Value, int_op: IntegerOp, float: FloatOp) -> Result<Value> {
    let floats = match (a, b) {
        (Value::Integer(x), Value::Integer(y)) => {
            return Ok(integer_op(int_op, *x, *y)?.map_or(NULL_VALUE, Value::Integer))
        }
        (Value::Float(x), Value::Float(y)) => (*x, *y),
        (Value::Integer(x), Value::Float(y)) => (*x as f64, *y),
        (Value::Float(x), Value::Integer(y)) => (*x, *y as f64),
        _ if a.is_null() || b.is_null() => return Ok(Value::Null(DataType::Float)),
        _ => return Err(not_a_number(if is_number(a) { b } else { a })),
    };
    Ok(float_op(float, floats.0, floats.1)?.map_or(NULL_VALUE, Value::Float))
}

/// `-a`: an INTEGER past its range is an error, a NULL keeps its type and
/// a value that is not a number is an error
#[inline]
pub fn numeric_neg(a: &Value) -> Result<Value> {
    Ok(match a {
        Value::Integer(i) => Value::Integer(integer_neg(*i)?),
        Value::Float(f) => Value::Float(-f),
        Value::Null(dt) => Value::Null(*dt),
        _ => return Err(not_a_number(a)),
    })
}

fn is_number(value: &Value) -> bool {
    matches!(value, Value::Integer(_) | Value::Float(_))
}

/// The error for `operand` of an arithmetic operator, which is not a number
#[cold]
#[inline(never)]
fn not_a_number(operand: &Value) -> Error {
    Error::Type(format!(
        "arithmetic needs INTEGER or FLOAT, got {}",
        operand.data_type()
    ))
}
