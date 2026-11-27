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
//! and `float_op` answers a pair, and the NULL a pair without numbers gives

use super::float::{float_op, FloatOp};
use super::integer::{integer_neg, integer_op, IntegerOp};
use super::value::NULL_VALUE;
use super::{DataType, Result, Value};

/// `a op b` over two numbers, an INTEGER pair by `int_op`, any pair holding
/// a FLOAT by `float`; NULL for a zero divisor, a NULL operand or a value
/// that is not a number
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
        _ => return Ok(NULL_VALUE),
    };
    Ok(float_op(float, floats.0, floats.1)?.map_or(NULL_VALUE, Value::Float))
}

/// `-a`: an INTEGER past its range is an error, a NULL keeps its type and
/// a value that is not a number gives NULL
#[inline]
pub fn numeric_neg(a: &Value) -> Result<Value> {
    Ok(match a {
        Value::Integer(i) => Value::Integer(integer_neg(*i)?),
        Value::Float(f) => Value::Float(-f),
        Value::Null(dt) => Value::Null(*dt),
        _ => Value::Null(DataType::Null),
    })
}

/// Arithmetic over a row's number columns, as an aggregate call's argument:
/// `+`, `-`, `*`, `/` and `%` of columns and literals, and `-` of one. A
/// column leaf names its place in `NumericProgram::columns`
#[derive(Debug, Clone)]
pub enum NumericExpr {
    Column(usize),
    Literal(Value),
    Neg(Box<NumericExpr>),
    Binary(IntegerOp, FloatOp, Box<NumericExpr>, Box<NumericExpr>),
}

/// A `NumericExpr`, the table columns its leaves read in leaf order, and
/// the call's name and argument as written, which an error names
#[derive(Debug, Clone)]
pub struct NumericProgram {
    pub expr: NumericExpr,
    pub columns: Vec<usize>,
    pub call: (String, String),
}

impl NumericProgram {
    /// The value for a row whose `columns` read as `cell` gives them, by
    /// `numeric_op` and `numeric_neg`, the rule the expression VM computes
    /// by; an error is the call's, as `Error::aggregate_argument` names it
    pub fn evaluate(&self, cell: &impl Fn(usize) -> Value) -> Result<Value> {
        fn walk(expr: &NumericExpr, cell: &impl Fn(usize) -> Value) -> Result<Value> {
            match expr {
                NumericExpr::Column(at) => Ok(cell(*at)),
                NumericExpr::Literal(value) => Ok(value.clone()),
                NumericExpr::Neg(inner) => numeric_neg(&walk(inner, cell)?),
                NumericExpr::Binary(int_op, float, left, right) => {
                    numeric_op(&walk(left, cell)?, &walk(right, cell)?, *int_op, *float)
                }
            }
        }
        walk(&self.expr, cell)
            .map_err(|error| super::Error::aggregate_argument(&self.call.0, &self.call.1, error))
    }
}
