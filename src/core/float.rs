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

//! FLOAT arithmetic: an error for a result two finite operands take past
//! the FLOAT range, NULL for a zero divisor.

use super::{Error, Result, Value};

/// An arithmetic operator on two FLOATs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
}

/// `x op y`; None for a zero divisor, which reads as NULL
#[inline]
pub fn float_op(op: FloatOp, x: f64, y: f64) -> Result<Option<f64>> {
    let result = match op {
        FloatOp::Add => x + y,
        FloatOp::Sub => x - y,
        FloatOp::Mul => x * y,
        FloatOp::Div if y == 0.0 => return Ok(None),
        FloatOp::Div => x / y,
        FloatOp::Mod if y == 0.0 => return Ok(None),
        FloatOp::Mod => x % y,
    };
    if !result.is_finite() && x.is_finite() && y.is_finite() {
        return Err(float_overflow(x, y));
    }
    Ok(Some(result))
}

#[cold]
#[inline(never)]
fn float_overflow(x: f64, y: f64) -> Error {
    Error::Type(format!(
        "Float overflow in arithmetic operation: {} and {}",
        Value::Float(x),
        Value::Float(y)
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_a_finite_result_is_kept_and_an_overflow_refused() {
        assert_eq!(float_op(FloatOp::Mul, 1.5, 2.0).unwrap(), Some(3.0));
        assert_eq!(float_op(FloatOp::Div, 1.0, 0.0).unwrap(), None);
        assert_eq!(float_op(FloatOp::Mod, 1.0, 0.0).unwrap(), None);
        for (op, x, y) in [
            (FloatOp::Add, f64::MAX, f64::MAX),
            (FloatOp::Sub, -f64::MAX, f64::MAX),
            (FloatOp::Mul, 1e308, 10.0),
            (FloatOp::Div, 1e308, 1e-10),
        ] {
            assert!(float_op(op, x, y).is_err(), "{op:?} {x} {y}");
        }
        let infinite = float_op(FloatOp::Mul, f64::INFINITY, 2.0).unwrap();
        assert_eq!(infinite, Some(f64::INFINITY));
    }
}
