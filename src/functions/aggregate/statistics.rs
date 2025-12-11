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

//! Statistical aggregate functions: STDDEV, VARIANCE, MEDIAN

use crate::core::Value;
use crate::functions::{
    AggregateFunction, FunctionDataType, FunctionInfo, FunctionSignature, FunctionType,
};

use super::DistinctTracker;

/// Helper to extract f64 from Value
fn value_to_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Integer(i) => Some(*i as f64),
        Value::Float(f) => Some(*f),
        Value::Text(s) => s.parse().ok(),
        _ => None,
    }
}

// ============================================================================
// STDDEV_POP / STDDEV - Population Standard Deviation
// ============================================================================

/// STDDEV_POP aggregate function (Population Standard Deviation)
///
/// Computes the population standard deviation of all non-NULL values.
/// Formula: sqrt(sum((x - mean)^2) / N)
#[derive(Default)]
pub struct StddevPopFunction {
    values: Vec<f64>,
    distinct_tracker: Option<DistinctTracker>,
}

impl AggregateFunction for StddevPopFunction {
    fn name(&self) -> &str {
        "STDDEV_POP"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "STDDEV_POP",
            FunctionType::Aggregate,
            "Returns the population standard deviation of non-NULL values",
            FunctionSignature::new(FunctionDataType::Float, vec![FunctionDataType::Any], 1, 1),
        )
    }

    fn accumulate(&mut self, value: &Value, distinct: bool) {
        if value.is_null() {
            return;
        }

        if distinct {
            if self.distinct_tracker.is_none() {
                self.distinct_tracker = Some(DistinctTracker::default());
            }
            if !self.distinct_tracker.as_mut().unwrap().check_and_add(value) {
                return;
            }
        }

        if let Some(f) = value_to_f64(value) {
            self.values.push(f);
        }
    }

    fn result(&self) -> Value {
        if self.values.is_empty() {
            return Value::null_unknown();
        }

        let n = self.values.len() as f64;
        let mean = self.values.iter().sum::<f64>() / n;
        let variance = self.values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        Value::Float(variance.sqrt())
    }

    fn reset(&mut self) {
        self.values.clear();
        self.distinct_tracker = None;
    }

    fn clone_box(&self) -> Box<dyn AggregateFunction> {
        Box::new(StddevPopFunction::default())
    }
}

/// STDDEV aggregate function (alias for STDDEV_POP)
#[derive(Default)]
pub struct StddevFunction {
    inner: StddevPopFunction,
}

impl AggregateFunction for StddevFunction {
    fn name(&self) -> &str {
        "STDDEV"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "STDDEV",
            FunctionType::Aggregate,
            "Returns the population standard deviation (alias for STDDEV_POP)",
            FunctionSignature::new(FunctionDataType::Float, vec![FunctionDataType::Any], 1, 1),
        )
    }

    fn accumulate(&mut self, value: &Value, distinct: bool) {
        self.inner.accumulate(value, distinct);
    }

    fn result(&self) -> Value {
        self.inner.result()
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn clone_box(&self) -> Box<dyn AggregateFunction> {
        Box::new(StddevFunction::default())
    }
}

// ============================================================================
// STDDEV_SAMP - Sample Standard Deviation
// ============================================================================

/// STDDEV_SAMP aggregate function (Sample Standard Deviation)
///
/// Computes the sample standard deviation of all non-NULL values.
/// Formula: sqrt(sum((x - mean)^2) / (N - 1))
/// Uses Bessel's correction (N-1 instead of N).
#[derive(Default)]
pub struct StddevSampFunction {
    values: Vec<f64>,
    distinct_tracker: Option<DistinctTracker>,
}

impl AggregateFunction for StddevSampFunction {
    fn name(&self) -> &str {
        "STDDEV_SAMP"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "STDDEV_SAMP",
            FunctionType::Aggregate,
            "Returns the sample standard deviation of non-NULL values",
            FunctionSignature::new(FunctionDataType::Float, vec![FunctionDataType::Any], 1, 1),
        )
    }

    fn accumulate(&mut self, value: &Value, distinct: bool) {
        if value.is_null() {
            return;
        }

        if distinct {
            if self.distinct_tracker.is_none() {
                self.distinct_tracker = Some(DistinctTracker::default());
            }
            if !self.distinct_tracker.as_mut().unwrap().check_and_add(value) {
                return;
            }
        }

        if let Some(f) = value_to_f64(value) {
            self.values.push(f);
        }
    }

    fn result(&self) -> Value {
        // Sample stddev requires at least 2 values
        if self.values.len() < 2 {
            return Value::null_unknown();
        }

        let n = self.values.len() as f64;
        let mean = self.values.iter().sum::<f64>() / n;
        let variance = self.values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        Value::Float(variance.sqrt())
    }

    fn reset(&mut self) {
        self.values.clear();
        self.distinct_tracker = None;
    }

    fn clone_box(&self) -> Box<dyn AggregateFunction> {
        Box::new(StddevSampFunction::default())
    }
}

// ============================================================================
// VAR_POP / VARIANCE - Population Variance
// ============================================================================

/// VAR_POP aggregate function (Population Variance)
///
/// Computes the population variance of all non-NULL values.
/// Formula: sum((x - mean)^2) / N
#[derive(Default)]
pub struct VarPopFunction {
    values: Vec<f64>,
    distinct_tracker: Option<DistinctTracker>,
}

impl AggregateFunction for VarPopFunction {
    fn name(&self) -> &str {
        "VAR_POP"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "VAR_POP",
            FunctionType::Aggregate,
            "Returns the population variance of non-NULL values",
            FunctionSignature::new(FunctionDataType::Float, vec![FunctionDataType::Any], 1, 1),
        )
    }

    fn accumulate(&mut self, value: &Value, distinct: bool) {
        if value.is_null() {
            return;
        }

        if distinct {
            if self.distinct_tracker.is_none() {
                self.distinct_tracker = Some(DistinctTracker::default());
            }
            if !self.distinct_tracker.as_mut().unwrap().check_and_add(value) {
                return;
            }
        }

        if let Some(f) = value_to_f64(value) {
            self.values.push(f);
        }
    }

    fn result(&self) -> Value {
        if self.values.is_empty() {
            return Value::null_unknown();
        }

        let n = self.values.len() as f64;
        let mean = self.values.iter().sum::<f64>() / n;
        let variance = self.values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        Value::Float(variance)
    }

    fn reset(&mut self) {
        self.values.clear();
        self.distinct_tracker = None;
    }

    fn clone_box(&self) -> Box<dyn AggregateFunction> {
        Box::new(VarPopFunction::default())
    }
}

/// VARIANCE aggregate function (alias for VAR_POP)
#[derive(Default)]
pub struct VarianceFunction {
    inner: VarPopFunction,
}

impl AggregateFunction for VarianceFunction {
    fn name(&self) -> &str {
        "VARIANCE"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "VARIANCE",
            FunctionType::Aggregate,
            "Returns the population variance (alias for VAR_POP)",
            FunctionSignature::new(FunctionDataType::Float, vec![FunctionDataType::Any], 1, 1),
        )
    }

    fn accumulate(&mut self, value: &Value, distinct: bool) {
        self.inner.accumulate(value, distinct);
    }

    fn result(&self) -> Value {
        self.inner.result()
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn clone_box(&self) -> Box<dyn AggregateFunction> {
        Box::new(VarianceFunction::default())
    }
}

// ============================================================================
// VAR_SAMP - Sample Variance
// ============================================================================

/// VAR_SAMP aggregate function (Sample Variance)
///
/// Computes the sample variance of all non-NULL values.
/// Formula: sum((x - mean)^2) / (N - 1)
/// Uses Bessel's correction (N-1 instead of N).
#[derive(Default)]
pub struct VarSampFunction {
    values: Vec<f64>,
    distinct_tracker: Option<DistinctTracker>,
}

impl AggregateFunction for VarSampFunction {
    fn name(&self) -> &str {
        "VAR_SAMP"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "VAR_SAMP",
            FunctionType::Aggregate,
            "Returns the sample variance of non-NULL values",
            FunctionSignature::new(FunctionDataType::Float, vec![FunctionDataType::Any], 1, 1),
        )
    }

    fn accumulate(&mut self, value: &Value, distinct: bool) {
        if value.is_null() {
            return;
        }

        if distinct {
            if self.distinct_tracker.is_none() {
                self.distinct_tracker = Some(DistinctTracker::default());
            }
            if !self.distinct_tracker.as_mut().unwrap().check_and_add(value) {
                return;
            }
        }

        if let Some(f) = value_to_f64(value) {
            self.values.push(f);
        }
    }

    fn result(&self) -> Value {
        // Sample variance requires at least 2 values
        if self.values.len() < 2 {
            return Value::null_unknown();
        }

        let n = self.values.len() as f64;
        let mean = self.values.iter().sum::<f64>() / n;
        let variance = self.values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        Value::Float(variance)
    }

    fn reset(&mut self) {
        self.values.clear();
        self.distinct_tracker = None;
    }

    fn clone_box(&self) -> Box<dyn AggregateFunction> {
        Box::new(VarSampFunction::default())
    }
}

// ============================================================================
// MEDIAN
// ============================================================================

/// MEDIAN aggregate function
///
/// Returns the median (middle value) of all non-NULL values.
/// For even number of values, returns the average of the two middle values.
#[derive(Default)]
pub struct MedianFunction {
    values: Vec<f64>,
    distinct_tracker: Option<DistinctTracker>,
}

impl AggregateFunction for MedianFunction {
    fn name(&self) -> &str {
        "MEDIAN"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "MEDIAN",
            FunctionType::Aggregate,
            "Returns the median (middle value) of non-NULL values",
            FunctionSignature::new(FunctionDataType::Float, vec![FunctionDataType::Any], 1, 1),
        )
    }

    fn accumulate(&mut self, value: &Value, distinct: bool) {
        if value.is_null() {
            return;
        }

        if distinct {
            if self.distinct_tracker.is_none() {
                self.distinct_tracker = Some(DistinctTracker::default());
            }
            if !self.distinct_tracker.as_mut().unwrap().check_and_add(value) {
                return;
            }
        }

        if let Some(f) = value_to_f64(value) {
            self.values.push(f);
        }
    }

    fn result(&self) -> Value {
        if self.values.is_empty() {
            return Value::null_unknown();
        }

        let mut sorted = self.values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted.len();
        let median = if n.is_multiple_of(2) {
            // Even: average of two middle values
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            // Odd: middle value
            sorted[n / 2]
        };

        Value::Float(median)
    }

    fn reset(&mut self) {
        self.values.clear();
        self.distinct_tracker = None;
    }

    fn clone_box(&self) -> Box<dyn AggregateFunction> {
        Box::new(MedianFunction::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // STDDEV_POP tests
    #[test]
    fn test_stddev_pop_basic() {
        let mut stddev = StddevPopFunction::default();
        // Values: 2, 4, 4, 4, 5, 5, 7, 9
        // Mean = 5, Variance = 4, StdDev = 2
        for v in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            stddev.accumulate(&Value::Float(v), false);
        }
        let result = stddev.result();
        if let Value::Float(f) = result {
            assert!((f - 2.0).abs() < 0.0001);
        } else {
            panic!("Expected float result");
        }
    }

    #[test]
    fn test_stddev_pop_single_value() {
        let mut stddev = StddevPopFunction::default();
        stddev.accumulate(&Value::Float(5.0), false);
        // Single value has stddev = 0
        assert_eq!(stddev.result(), Value::Float(0.0));
    }

    #[test]
    fn test_stddev_pop_empty() {
        let stddev = StddevPopFunction::default();
        assert!(stddev.result().is_null());
    }

    #[test]
    fn test_stddev_alias() {
        let mut stddev = StddevFunction::default();
        for v in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            stddev.accumulate(&Value::Float(v), false);
        }
        let result = stddev.result();
        if let Value::Float(f) = result {
            assert!((f - 2.0).abs() < 0.0001);
        } else {
            panic!("Expected float result");
        }
    }

    // STDDEV_SAMP tests
    #[test]
    fn test_stddev_samp_basic() {
        let mut stddev = StddevSampFunction::default();
        // Values: 2, 4, 4, 4, 5, 5, 7, 9 (n=8)
        // Mean = 5, Sample Variance = 32/7 ≈ 4.571, Sample StdDev ≈ 2.138
        for v in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            stddev.accumulate(&Value::Float(v), false);
        }
        let result = stddev.result();
        if let Value::Float(f) = result {
            assert!((f - 2.138).abs() < 0.01);
        } else {
            panic!("Expected float result");
        }
    }

    #[test]
    fn test_stddev_samp_single_value() {
        let mut stddev = StddevSampFunction::default();
        stddev.accumulate(&Value::Float(5.0), false);
        // Sample stddev requires at least 2 values
        assert!(stddev.result().is_null());
    }

    // VAR_POP tests
    #[test]
    fn test_var_pop_basic() {
        let mut var = VarPopFunction::default();
        // Values: 2, 4, 4, 4, 5, 5, 7, 9
        // Mean = 5, Variance = 4
        for v in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            var.accumulate(&Value::Float(v), false);
        }
        let result = var.result();
        if let Value::Float(f) = result {
            assert!((f - 4.0).abs() < 0.0001);
        } else {
            panic!("Expected float result");
        }
    }

    #[test]
    fn test_variance_alias() {
        let mut var = VarianceFunction::default();
        for v in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            var.accumulate(&Value::Float(v), false);
        }
        let result = var.result();
        if let Value::Float(f) = result {
            assert!((f - 4.0).abs() < 0.0001);
        } else {
            panic!("Expected float result");
        }
    }

    // VAR_SAMP tests
    #[test]
    fn test_var_samp_basic() {
        let mut var = VarSampFunction::default();
        // Values: 2, 4, 4, 4, 5, 5, 7, 9 (n=8)
        // Mean = 5, Sample Variance = 32/7 ≈ 4.571
        for v in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            var.accumulate(&Value::Float(v), false);
        }
        let result = var.result();
        if let Value::Float(f) = result {
            assert!((f - 4.571).abs() < 0.01);
        } else {
            panic!("Expected float result");
        }
    }

    #[test]
    fn test_var_samp_single_value() {
        let mut var = VarSampFunction::default();
        var.accumulate(&Value::Float(5.0), false);
        // Sample variance requires at least 2 values
        assert!(var.result().is_null());
    }

    // MEDIAN tests
    #[test]
    fn test_median_odd_count() {
        let mut median = MedianFunction::default();
        // Values: 1, 3, 5, 7, 9 -> median = 5
        for v in [1.0, 3.0, 5.0, 7.0, 9.0] {
            median.accumulate(&Value::Float(v), false);
        }
        assert_eq!(median.result(), Value::Float(5.0));
    }

    #[test]
    fn test_median_even_count() {
        let mut median = MedianFunction::default();
        // Values: 1, 3, 5, 7 -> median = (3+5)/2 = 4
        for v in [1.0, 3.0, 5.0, 7.0] {
            median.accumulate(&Value::Float(v), false);
        }
        assert_eq!(median.result(), Value::Float(4.0));
    }

    #[test]
    fn test_median_unsorted_input() {
        let mut median = MedianFunction::default();
        // Values: 9, 1, 7, 3, 5 -> sorted: 1, 3, 5, 7, 9 -> median = 5
        for v in [9.0, 1.0, 7.0, 3.0, 5.0] {
            median.accumulate(&Value::Float(v), false);
        }
        assert_eq!(median.result(), Value::Float(5.0));
    }

    #[test]
    fn test_median_single_value() {
        let mut median = MedianFunction::default();
        median.accumulate(&Value::Float(42.0), false);
        assert_eq!(median.result(), Value::Float(42.0));
    }

    #[test]
    fn test_median_empty() {
        let median = MedianFunction::default();
        assert!(median.result().is_null());
    }

    #[test]
    fn test_median_with_integers() {
        let mut median = MedianFunction::default();
        for v in [1, 2, 3, 4, 5] {
            median.accumulate(&Value::Integer(v), false);
        }
        assert_eq!(median.result(), Value::Float(3.0));
    }

    #[test]
    fn test_median_ignores_null() {
        let mut median = MedianFunction::default();
        median.accumulate(&Value::Float(1.0), false);
        median.accumulate(&Value::null_unknown(), false);
        median.accumulate(&Value::Float(3.0), false);
        median.accumulate(&Value::null_unknown(), false);
        median.accumulate(&Value::Float(5.0), false);
        assert_eq!(median.result(), Value::Float(3.0));
    }

    #[test]
    fn test_median_distinct() {
        let mut median = MedianFunction::default();
        median.accumulate(&Value::Float(1.0), true);
        median.accumulate(&Value::Float(3.0), true);
        median.accumulate(&Value::Float(3.0), true); // duplicate
        median.accumulate(&Value::Float(5.0), true);
        // Distinct values: 1, 3, 5 -> median = 3
        assert_eq!(median.result(), Value::Float(3.0));
    }

    // Test with NULL handling
    #[test]
    fn test_statistics_ignore_nulls() {
        let mut stddev = StddevPopFunction::default();
        stddev.accumulate(&Value::Float(2.0), false);
        stddev.accumulate(&Value::null_unknown(), false);
        stddev.accumulate(&Value::Float(4.0), false);
        stddev.accumulate(&Value::null_unknown(), false);
        stddev.accumulate(&Value::Float(6.0), false);
        // Mean = 4, Variance = ((2-4)^2 + (4-4)^2 + (6-4)^2) / 3 = 8/3
        let result = stddev.result();
        if let Value::Float(f) = result {
            let expected = (8.0_f64 / 3.0).sqrt();
            assert!((f - expected).abs() < 0.0001);
        } else {
            panic!("Expected float result");
        }
    }

    // Test distinct
    #[test]
    fn test_stddev_distinct() {
        let mut stddev = StddevPopFunction::default();
        stddev.accumulate(&Value::Float(2.0), true);
        stddev.accumulate(&Value::Float(4.0), true);
        stddev.accumulate(&Value::Float(2.0), true); // duplicate
        stddev.accumulate(&Value::Float(6.0), true);
        // Distinct values: 2, 4, 6 -> Mean = 4
        let result = stddev.result();
        if let Value::Float(f) = result {
            let expected = (8.0_f64 / 3.0).sqrt();
            assert!((f - expected).abs() < 0.0001);
        } else {
            panic!("Expected float result");
        }
    }

    // Test reset
    #[test]
    fn test_stddev_reset() {
        let mut stddev = StddevPopFunction::default();
        stddev.accumulate(&Value::Float(2.0), false);
        stddev.accumulate(&Value::Float(4.0), false);
        stddev.reset();
        assert!(stddev.result().is_null());
    }

    #[test]
    fn test_median_reset() {
        let mut median = MedianFunction::default();
        median.accumulate(&Value::Float(1.0), false);
        median.accumulate(&Value::Float(2.0), false);
        median.reset();
        assert!(median.result().is_null());
    }
}
