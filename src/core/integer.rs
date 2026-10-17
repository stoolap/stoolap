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

//! INTEGER arithmetic: exact, an error for a result outside the INTEGER
//! range, NULL for a zero divisor, and a shift by 0 to 63 places only.

use super::{Error, Result};

/// An operator on two INTEGERs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntegerOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Shl,
    Shr,
}

/// `x op y`; None for a zero divisor, which reads as NULL
pub fn integer_op(op: IntegerOp, x: i64, y: i64) -> Result<Option<i64>> {
    let overflow = || {
        Error::Type(format!(
            "Integer overflow in arithmetic operation: {x} and {y}"
        ))
    };
    let shift = || {
        u32::try_from(y)
            .ok()
            .filter(|places| *places < i64::BITS)
            .ok_or_else(|| Error::Type(format!("Shift amount must be from 0 to 63, got {y}")))
    };
    Ok(Some(match op {
        IntegerOp::Add => x.checked_add(y).ok_or_else(overflow)?,
        IntegerOp::Sub => x.checked_sub(y).ok_or_else(overflow)?,
        IntegerOp::Mul => x.checked_mul(y).ok_or_else(overflow)?,
        IntegerOp::Div if y == 0 => return Ok(None),
        IntegerOp::Div => x.checked_div(y).ok_or_else(overflow)?,
        IntegerOp::Mod if y == 0 => return Ok(None),
        // The remainder always fits: i64::MIN % -1 is 0
        IntegerOp::Mod => x.wrapping_rem(y),
        IntegerOp::Shl => x << shift()?,
        IntegerOp::Shr => x >> shift()?,
    }))
}

/// `-x`
pub fn integer_neg(x: i64) -> Result<i64> {
    x.checked_neg()
        .ok_or_else(|| Error::Type(format!("Integer overflow in arithmetic operation: -{x}")))
}

/// `ABS(x)`
pub fn integer_abs(x: i64) -> Result<i64> {
    x.checked_abs()
        .ok_or_else(|| Error::Type(format!("Integer overflow in ABS: {x}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_a_result_outside_the_range_is_an_error() {
        let ok = |op, x, y, want: Option<i64>| {
            assert_eq!(integer_op(op, x, y).unwrap(), want, "{op:?} {x} {y}");
        };
        let err = |op, x, y| assert!(integer_op(op, x, y).is_err(), "{op:?} {x} {y}");
        ok(IntegerOp::Add, 2, 3, Some(5));
        err(IntegerOp::Add, i64::MAX, 1);
        err(IntegerOp::Sub, i64::MIN, 1);
        err(IntegerOp::Mul, i64::MAX, 2);
        ok(IntegerOp::Div, 7, 2, Some(3));
        ok(IntegerOp::Div, -7, 2, Some(-3));
        ok(IntegerOp::Div, 7, 0, None);
        err(IntegerOp::Div, i64::MIN, -1);
        ok(IntegerOp::Mod, -7, 2, Some(-1));
        ok(IntegerOp::Mod, 7, 0, None);
        ok(IntegerOp::Mod, i64::MIN, -1, Some(0));
        ok(IntegerOp::Shl, 1, 63, Some(i64::MIN));
        ok(IntegerOp::Shr, -8, 1, Some(-4));
        err(IntegerOp::Shl, 1, 64);
        err(IntegerOp::Shl, 1, -1);
        err(IntegerOp::Shr, 1, 64);
        assert!(integer_neg(i64::MIN).is_err());
        assert_eq!(integer_neg(i64::MAX).unwrap(), -i64::MAX);
        assert!(integer_abs(i64::MIN).is_err());
        assert_eq!(integer_abs(-5).unwrap(), 5);
    }
}
