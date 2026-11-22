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

//! The one SUM and AVG accumulator: INTEGER and BOOLEAN values add exactly,
//! FLOAT values add in f64, and the sum answers as `+` does

use super::{Error, Result, Value};

/// A running SUM: every path that sums values, in the executor or in
/// storage, adds into one of these and reads its result from it
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SumAccumulator {
    int: i128,
    float: f64,
    has_float: bool,
    /// A FLOAT added was infinite or NaN, so a sum that is too answers so
    non_finite: bool,
    count: u64,
}

impl SumAccumulator {
    /// Adds an INTEGER, or a BOOLEAN as one or nought
    #[inline]
    pub fn add_int(&mut self, v: i64) {
        self.int += v as i128;
        self.count += 1;
    }

    /// Adds a FLOAT
    #[inline]
    pub fn add_float(&mut self, v: f64) {
        self.float += v;
        self.has_float = true;
        if !v.is_finite() {
            self.non_finite = true;
        }
        self.count += 1;
    }

    /// Adds a FLOAT of a run without noting whether it is finite; the run
    /// ends with [`Self::settle_run`]
    #[inline]
    pub fn add_float_in_run(&mut self, v: f64) {
        self.float += v;
        self.has_float = true;
        self.count += 1;
    }

    /// Ends a run of [`Self::add_float_in_run`]. An infinite or NaN value
    /// keeps the sum non-finite, so a sum still finite needs no look; one
    /// that is not reads the run again through `run_held_non_finite`
    #[inline]
    pub fn settle_run(&mut self, run_held_non_finite: impl FnOnce() -> bool) {
        if !self.non_finite && !self.float.is_finite() && run_held_non_finite() {
            self.non_finite = true;
        }
    }

    /// Adds a value; NULL and non-numeric values add nothing
    #[inline]
    pub fn add(&mut self, value: &Value) {
        match value {
            Value::Integer(i) => self.add_int(*i),
            Value::Float(f) => self.add_float(*f),
            Value::Boolean(b) => self.add_int(*b as i64),
            _ => {}
        }
    }

    /// Adds `count` values summed elsewhere: `int` from INTEGER and BOOLEAN
    /// values, `float` from FLOAT values, `has_float` when any was FLOAT,
    /// `non_finite` when any FLOAT was infinite or NaN
    #[inline]
    pub fn add_parts(
        &mut self,
        int: i128,
        float: f64,
        has_float: bool,
        non_finite: bool,
        count: u64,
    ) {
        self.int += int;
        self.float += float;
        self.has_float |= has_float;
        self.non_finite |= non_finite;
        self.count += count;
    }

    /// Adds what another accumulator holds
    #[inline]
    pub fn merge(&mut self, other: &SumAccumulator) {
        let (int, float) = (other.int, other.float);
        self.add_parts(int, float, other.has_float, other.non_finite, other.count);
    }

    /// The number of values added
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Whether the FLOAT sum is finite, or there is none: a sum of a part of
    /// the values that is not, by an overflow or an infinite or NaN value,
    /// may give onto another sum what all of them added in order do not
    #[inline]
    pub fn sum_finite(&self) -> bool {
        !self.has_float || (self.int as f64 + self.float).is_finite()
    }

    /// The sum: NULL with no value, FLOAT when any value was FLOAT, and
    /// otherwise the exact INTEGER; an overflow error outside either range
    #[inline]
    pub fn sum(&self) -> Result<Value> {
        if self.count == 0 {
            return Ok(Value::null_unknown());
        }
        if self.has_float {
            return self.float_sum("SUM").map(Value::Float);
        }
        i64::try_from(self.int).map(Value::Integer).map_err(|_| {
            Error::Type(format!(
                "Integer overflow in SUM: {} is outside the INTEGER range",
                self.int
            ))
        })
    }

    /// The average as FLOAT: the exact INTEGER sum divided and rounded
    /// once, or the FLOAT sum divided; NULL with no value
    #[inline]
    pub fn avg(&self) -> Result<Value> {
        if self.count == 0 {
            return Ok(Value::null_unknown());
        }
        if self.has_float {
            let sum = self.float_sum("AVG")?;
            return Ok(Value::Float(sum / self.count as f64));
        }
        Ok(Value::Float(ratio(self.int, self.count)))
    }

    /// The FLOAT sum, which finite values take past the FLOAT range only
    /// as an overflow error of `function`, as `+` refuses one
    #[inline]
    fn float_sum(&self, function: &str) -> Result<f64> {
        let sum = self.int as f64 + self.float;
        if !sum.is_finite() && !self.non_finite {
            return Err(sum_overflow(function, self.count));
        }
        Ok(sum)
    }
}

#[cold]
#[inline(never)]
fn sum_overflow(function: &str, count: u64) -> Error {
    Error::Type(format!(
        "Float overflow in {function}: the sum of {count} values is outside the FLOAT range"
    ))
}

/// `num / den` rounded once to the nearest f64: the quotient keeps at least
/// 64 bits, and a set low bit stands for any remainder below them
fn ratio(num: i128, den: u64) -> f64 {
    let magnitude = num.unsigned_abs();
    if magnitude == 0 {
        return 0.0;
    }
    let shift = magnitude.leading_zeros();
    let scaled = magnitude << shift;
    let den = den as u128;
    let quotient = (scaled / den) | !scaled.is_multiple_of(den) as u128;
    let value = quotient as f64 * 2f64.powi(-(shift as i32));
    if num < 0 {
        -value
    } else {
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn of(values: &[Value]) -> SumAccumulator {
        let mut acc = SumAccumulator::default();
        values.iter().for_each(|v| acc.add(v));
        acc
    }

    #[test]
    fn integers_sum_exactly_through_an_intermediate_overflow() {
        let acc = of(&[
            Value::Integer(i64::MAX),
            Value::Integer(10),
            Value::Integer(-20),
        ]);
        assert_eq!(acc.sum().unwrap(), Value::Integer(i64::MAX - 10));
    }

    #[test]
    fn a_sum_outside_the_integer_range_is_an_error() {
        let acc = of(&[Value::Integer(i64::MAX), Value::Integer(1)]);
        assert!(acc.sum().is_err());
        let acc = of(&[Value::Integer(i64::MIN), Value::Integer(-1)]);
        assert!(acc.sum().is_err());
    }

    #[test]
    fn a_float_makes_the_sum_float() {
        let acc = of(&[Value::Integer(2), Value::Float(3.0)]);
        assert_eq!(acc.sum().unwrap(), Value::Float(5.0));
    }

    fn in_runs(runs: &[&[f64]]) -> SumAccumulator {
        let mut acc = SumAccumulator::default();
        for run in runs {
            run.iter().for_each(|v| acc.add_float_in_run(*v));
            acc.settle_run(|| run.iter().any(|v| !v.is_finite()));
        }
        acc
    }

    #[test]
    fn runs_tell_an_overflow_from_an_infinite_value_wherever_each_falls() {
        let inf = f64::INFINITY;
        assert!(in_runs(&[&[1e308, 1e308], &[1.0]]).sum().is_err());
        assert!(in_runs(&[&[1.0], &[1e308, 1e308]]).avg().is_err());
        for runs in [
            &[&[1e308, 1e308, inf][..]][..],
            &[&[inf][..], &[1e308, 1e308][..]][..],
            &[&[1e308, 1e308][..], &[inf][..]][..],
            &[&[1e308][..], &[1e308][..], &[1.0][..], &[inf][..]][..],
        ] {
            assert_eq!(in_runs(runs).sum().unwrap(), Value::Float(inf), "{runs:?}");
        }
        assert_eq!(
            in_runs(&[&[1.0, 2.0], &[3.0]]).sum().unwrap(),
            Value::Float(6.0)
        );
    }

    #[test]
    fn booleans_count_one_or_nought_and_null_adds_nothing() {
        let acc = of(&[
            Value::Boolean(true),
            Value::Boolean(false),
            Value::null_unknown(),
            Value::Boolean(true),
        ]);
        assert_eq!(acc.sum().unwrap(), Value::Integer(2));
        assert_eq!(acc.count(), 3);
        assert!(of(&[Value::null_unknown()]).sum().unwrap().is_null());
        assert!(of(&[]).avg().unwrap().is_null());
    }

    #[test]
    fn avg_divides_the_exact_sum() {
        let acc = of(&[
            Value::Integer(9_007_199_254_740_993),
            Value::Integer(-9_007_199_254_740_992),
            Value::Integer(1),
        ]);
        assert_eq!(acc.avg().unwrap(), Value::Float(2.0 / 3.0));
        let acc = of(&[Value::Integer(i64::MAX), Value::Integer(i64::MAX)]);
        assert_eq!(acc.avg().unwrap(), Value::Float(i64::MAX as f64));
        let acc = of(&[Value::Integer(-3), Value::Integer(-4)]);
        assert_eq!(acc.avg().unwrap(), Value::Float(-3.5));
        let acc = of(&[
            Value::Integer(9_007_199_254_740_992),
            Value::Integer(9_007_199_254_740_995),
        ]);
        assert_eq!(acc.avg().unwrap(), Value::Float(9_007_199_254_740_994.0));
        let acc = of(&[
            Value::Integer(-9_007_199_254_740_992),
            Value::Integer(-9_007_199_254_740_995),
        ]);
        assert_eq!(acc.avg().unwrap(), Value::Float(-9_007_199_254_740_994.0));
        let acc = of(&[Value::Integer(i64::MAX), Value::Integer(i64::MAX - 1)]);
        assert_eq!(acc.avg().unwrap(), Value::Float(i64::MAX as f64));
    }

    #[test]
    fn avg_of_mixed_values_divides_the_float_sum() {
        let acc = of(&[
            Value::Integer(9_007_199_254_740_990),
            Value::Float(-9_007_199_254_740_989.0),
            Value::Integer(0),
        ]);
        assert_eq!(acc.avg().unwrap(), Value::Float(1.0 / 3.0));
    }

    #[test]
    fn merged_parts_answer_as_one_run() {
        let mut left = of(&[Value::Integer(i64::MAX), Value::Integer(1)]);
        left.merge(&of(&[Value::Integer(-2), Value::Float(0.5)]));
        assert_eq!(left.count(), 4);
        assert_eq!(
            left.sum().unwrap(),
            Value::Float(i64::MAX as f64 - 1.0 + 0.5)
        );
        let mut ints = of(&[Value::Integer(i64::MAX), Value::Integer(1)]);
        ints.merge(&of(&[Value::Integer(-2)]));
        assert_eq!(ints.sum().unwrap(), Value::Integer(i64::MAX - 1));
    }
}
