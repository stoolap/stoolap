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

//! RANK and DENSE_RANK window functions

use crate::core::{Result, Value};
use crate::functions::{
    FunctionDataType, FunctionInfo, FunctionSignature, FunctionType, WindowFunction,
};

/// RANK window function
///
/// Returns the rank of the current row within the partition, with gaps.
/// Rows with equal values receive the same rank, and the next rank
/// is the row number (leaving gaps).
///
/// Example: If two rows tie for rank 1, the next row gets rank 3 (not 2).
#[derive(Default)]
pub struct RankFunction;

impl WindowFunction for RankFunction {
    fn name(&self) -> &str {
        "RANK"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "RANK",
            FunctionType::Window,
            "Returns the rank of the current row within the partition, with gaps for ties",
            FunctionSignature::new(FunctionDataType::Integer, vec![], 0, 0),
        )
    }

    fn process(
        &self,
        partition: &[Value],
        _order_by: &[Value],
        current_row: usize,
    ) -> Result<Value> {
        // RANK assigns the same rank to equal values, with gaps
        // Count how many values before current_row are less than current value
        if partition.is_empty() || current_row >= partition.len() {
            return Ok(Value::Integer((current_row + 1) as i64));
        }

        let current_value = &partition[current_row];
        let mut rank = 1i64;

        // Count rows that come before this one in the ordering
        for (i, value) in partition.iter().enumerate() {
            if i >= current_row {
                break;
            }
            if is_less_than(value, current_value) {
                rank += 1;
            }
        }

        Ok(Value::Integer(rank))
    }

    fn clone_box(&self) -> Box<dyn WindowFunction> {
        Box::new(RankFunction)
    }
}

/// DENSE_RANK window function
///
/// Returns the rank of the current row within the partition, without gaps.
/// Rows with equal values receive the same rank, and the next rank
/// is incremented by 1 (no gaps).
///
/// Example: If two rows tie for rank 1, the next row gets rank 2.
#[derive(Default)]
pub struct DenseRankFunction;

impl WindowFunction for DenseRankFunction {
    fn name(&self) -> &str {
        "DENSE_RANK"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "DENSE_RANK",
            FunctionType::Window,
            "Returns the rank of the current row within the partition, without gaps for ties",
            FunctionSignature::new(FunctionDataType::Integer, vec![], 0, 0),
        )
    }

    fn process(
        &self,
        partition: &[Value],
        _order_by: &[Value],
        current_row: usize,
    ) -> Result<Value> {
        // DENSE_RANK assigns the same rank to equal values, without gaps
        if partition.is_empty() || current_row >= partition.len() {
            return Ok(Value::Integer((current_row + 1) as i64));
        }

        let current_value = &partition[current_row];

        // Count distinct values that are less than current value
        let mut seen_values: Vec<&Value> = Vec::new();
        for (i, value) in partition.iter().enumerate() {
            if i >= current_row {
                break;
            }
            if is_less_than(value, current_value)
                && !seen_values.iter().any(|v| values_equal(v, value))
            {
                seen_values.push(value);
            }
        }

        Ok(Value::Integer(seen_values.len() as i64 + 1))
    }

    fn clone_box(&self) -> Box<dyn WindowFunction> {
        Box::new(DenseRankFunction)
    }
}

/// Compare two values for less than
fn is_less_than(a: &Value, b: &Value) -> bool {
    if a.is_null() || b.is_null() {
        return false;
    }
    match (a, b) {
        (Value::Integer(a), Value::Integer(b)) => a < b,
        (Value::Float(a), Value::Float(b)) => a < b,
        (Value::Integer(a), Value::Float(b)) => (*a as f64) < *b,
        (Value::Float(a), Value::Integer(b)) => *a < (*b as f64),
        (Value::Text(a), Value::Text(b)) => a < b,
        (Value::Boolean(a), Value::Boolean(b)) => !a && *b, // false < true
        (Value::Timestamp(a), Value::Timestamp(b)) => a < b,
        _ => false,
    }
}

/// Check if two values are equal
fn values_equal(a: &Value, b: &Value) -> bool {
    if a.is_null() && b.is_null() {
        return true;
    }
    match (a, b) {
        (Value::Integer(a), Value::Integer(b)) => a == b,
        (Value::Float(a), Value::Float(b)) => a == b,
        (Value::Integer(a), Value::Float(b)) => (*a as f64) == *b,
        (Value::Float(a), Value::Integer(b)) => *a == (*b as f64),
        (Value::Text(a), Value::Text(b)) => a == b,
        (Value::Boolean(a), Value::Boolean(b)) => a == b,
        (Value::Timestamp(a), Value::Timestamp(b)) => a == b,
        _ => false,
    }
}

/// PERCENT_RANK window function
///
/// Returns the relative rank of the current row: (rank - 1) / (total_rows - 1).
/// The result is always between 0 and 1. The first row always has PERCENT_RANK of 0.
///
/// Formula: (rank - 1) / (n - 1), where n is the number of rows in the partition.
#[derive(Default)]
pub struct PercentRankFunction;

impl WindowFunction for PercentRankFunction {
    fn name(&self) -> &str {
        "PERCENT_RANK"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "PERCENT_RANK",
            FunctionType::Window,
            "Returns the relative rank of the current row: (rank - 1) / (total_rows - 1)",
            FunctionSignature::new(FunctionDataType::Float, vec![], 0, 0),
        )
    }

    fn process(
        &self,
        partition: &[Value],
        _order_by: &[Value],
        current_row: usize,
    ) -> Result<Value> {
        let n = partition.len();

        // If only one row or empty, return 0
        if n <= 1 {
            return Ok(Value::Float(0.0));
        }

        if current_row >= n {
            return Ok(Value::Float(0.0));
        }

        let current_value = &partition[current_row];

        // Calculate rank (same as RANK function)
        let mut rank = 1i64;
        for (i, value) in partition.iter().enumerate() {
            if i >= current_row {
                break;
            }
            if is_less_than(value, current_value) {
                rank += 1;
            }
        }

        // PERCENT_RANK = (rank - 1) / (n - 1)
        let percent_rank = (rank - 1) as f64 / (n - 1) as f64;
        Ok(Value::Float(percent_rank))
    }

    fn clone_box(&self) -> Box<dyn WindowFunction> {
        Box::new(PercentRankFunction)
    }
}

/// CUME_DIST window function
///
/// Returns the cumulative distribution of a value within a partition.
/// Formula: (number of rows <= current row) / (total rows)
///
/// The result is always between 0 and 1 (exclusive of 0, inclusive of 1).
#[derive(Default)]
pub struct CumeDistFunction;

impl WindowFunction for CumeDistFunction {
    fn name(&self) -> &str {
        "CUME_DIST"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "CUME_DIST",
            FunctionType::Window,
            "Returns the cumulative distribution of a value: (rows <= current) / total_rows",
            FunctionSignature::new(FunctionDataType::Float, vec![], 0, 0),
        )
    }

    fn process(
        &self,
        partition: &[Value],
        _order_by: &[Value],
        current_row: usize,
    ) -> Result<Value> {
        let n = partition.len();

        if n == 0 {
            return Ok(Value::Float(1.0));
        }

        if current_row >= n {
            return Ok(Value::Float(1.0));
        }

        let current_value = &partition[current_row];

        // Count rows that are <= current value (including ties after current row)
        let mut count = 0i64;
        for value in partition.iter() {
            if is_less_than(value, current_value) || values_equal(value, current_value) {
                count += 1;
            }
        }

        // CUME_DIST = count / n
        let cume_dist = count as f64 / n as f64;
        Ok(Value::Float(cume_dist))
    }

    fn clone_box(&self) -> Box<dyn WindowFunction> {
        Box::new(CumeDistFunction)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rank_unique_values() {
        let f = RankFunction;
        // Partition with unique values, sorted
        let partition = vec![Value::Integer(10), Value::Integer(20), Value::Integer(30)];
        let order_by = vec![];

        assert_eq!(
            f.process(&partition, &order_by, 0).unwrap(),
            Value::Integer(1)
        );
        assert_eq!(
            f.process(&partition, &order_by, 1).unwrap(),
            Value::Integer(2)
        );
        assert_eq!(
            f.process(&partition, &order_by, 2).unwrap(),
            Value::Integer(3)
        );
    }

    #[test]
    fn test_rank_with_ties() {
        let f = RankFunction;
        // Partition with ties
        let partition = vec![Value::Integer(10), Value::Integer(10), Value::Integer(30)];
        let order_by = vec![];

        // Both first rows have value 10, so they get rank 1
        assert_eq!(
            f.process(&partition, &order_by, 0).unwrap(),
            Value::Integer(1)
        );
        assert_eq!(
            f.process(&partition, &order_by, 1).unwrap(),
            Value::Integer(1)
        );
        // Third row with value 30 gets rank 3 (gap after the tie)
        assert_eq!(
            f.process(&partition, &order_by, 2).unwrap(),
            Value::Integer(3)
        );
    }

    #[test]
    fn test_dense_rank_unique_values() {
        let f = DenseRankFunction;
        let partition = vec![Value::Integer(10), Value::Integer(20), Value::Integer(30)];
        let order_by = vec![];

        assert_eq!(
            f.process(&partition, &order_by, 0).unwrap(),
            Value::Integer(1)
        );
        assert_eq!(
            f.process(&partition, &order_by, 1).unwrap(),
            Value::Integer(2)
        );
        assert_eq!(
            f.process(&partition, &order_by, 2).unwrap(),
            Value::Integer(3)
        );
    }

    #[test]
    fn test_dense_rank_with_ties() {
        let f = DenseRankFunction;
        // Partition with ties
        let partition = vec![Value::Integer(10), Value::Integer(10), Value::Integer(30)];
        let order_by = vec![];

        // Both first rows have value 10, so they get rank 1
        assert_eq!(
            f.process(&partition, &order_by, 0).unwrap(),
            Value::Integer(1)
        );
        assert_eq!(
            f.process(&partition, &order_by, 1).unwrap(),
            Value::Integer(1)
        );
        // Third row with value 30 gets rank 2 (no gap)
        assert_eq!(
            f.process(&partition, &order_by, 2).unwrap(),
            Value::Integer(2)
        );
    }

    #[test]
    fn test_rank_empty_partition() {
        let f = RankFunction;
        let partition = vec![];
        let order_by = vec![];

        assert_eq!(
            f.process(&partition, &order_by, 0).unwrap(),
            Value::Integer(1)
        );
    }

    #[test]
    fn test_dense_rank_empty_partition() {
        let f = DenseRankFunction;
        let partition = vec![];
        let order_by = vec![];

        assert_eq!(
            f.process(&partition, &order_by, 0).unwrap(),
            Value::Integer(1)
        );
    }

    #[test]
    fn test_rank_strings() {
        let f = RankFunction;
        let partition = vec![
            Value::text("apple"),
            Value::text("banana"),
            Value::text("cherry"),
        ];
        let order_by = vec![];

        assert_eq!(
            f.process(&partition, &order_by, 0).unwrap(),
            Value::Integer(1)
        );
        assert_eq!(
            f.process(&partition, &order_by, 1).unwrap(),
            Value::Integer(2)
        );
        assert_eq!(
            f.process(&partition, &order_by, 2).unwrap(),
            Value::Integer(3)
        );
    }

    #[test]
    fn test_percent_rank_unique_values() {
        let f = PercentRankFunction;
        // 4 unique values
        let partition = vec![
            Value::Integer(10),
            Value::Integer(20),
            Value::Integer(30),
            Value::Integer(40),
        ];
        let order_by = vec![];

        // PERCENT_RANK = (rank - 1) / (n - 1)
        // Row 0: (1-1)/(4-1) = 0/3 = 0.0
        assert_eq!(
            f.process(&partition, &order_by, 0).unwrap(),
            Value::Float(0.0)
        );
        // Row 1: (2-1)/(4-1) = 1/3 ≈ 0.333
        if let Value::Float(v) = f.process(&partition, &order_by, 1).unwrap() {
            assert!((v - 0.3333333333333333).abs() < 0.0001);
        } else {
            panic!("Expected float");
        }
        // Row 2: (3-1)/(4-1) = 2/3 ≈ 0.667
        if let Value::Float(v) = f.process(&partition, &order_by, 2).unwrap() {
            assert!((v - 0.6666666666666666).abs() < 0.0001);
        } else {
            panic!("Expected float");
        }
        // Row 3: (4-1)/(4-1) = 3/3 = 1.0
        assert_eq!(
            f.process(&partition, &order_by, 3).unwrap(),
            Value::Float(1.0)
        );
    }

    #[test]
    fn test_percent_rank_with_ties() {
        let f = PercentRankFunction;
        // Values: 10, 10, 30 (two ties at rank 1)
        let partition = vec![Value::Integer(10), Value::Integer(10), Value::Integer(30)];
        let order_by = vec![];

        // Both first rows have rank 1: (1-1)/(3-1) = 0
        assert_eq!(
            f.process(&partition, &order_by, 0).unwrap(),
            Value::Float(0.0)
        );
        assert_eq!(
            f.process(&partition, &order_by, 1).unwrap(),
            Value::Float(0.0)
        );
        // Third row has rank 3: (3-1)/(3-1) = 1.0
        assert_eq!(
            f.process(&partition, &order_by, 2).unwrap(),
            Value::Float(1.0)
        );
    }

    #[test]
    fn test_percent_rank_single_row() {
        let f = PercentRankFunction;
        let partition = vec![Value::Integer(10)];
        let order_by = vec![];

        // Single row always has PERCENT_RANK of 0
        assert_eq!(
            f.process(&partition, &order_by, 0).unwrap(),
            Value::Float(0.0)
        );
    }

    #[test]
    fn test_cume_dist_unique_values() {
        let f = CumeDistFunction;
        // 4 unique values
        let partition = vec![
            Value::Integer(10),
            Value::Integer(20),
            Value::Integer(30),
            Value::Integer(40),
        ];
        let order_by = vec![];

        // CUME_DIST = (rows <= current) / total
        // Row 0 (value 10): 1/4 = 0.25
        assert_eq!(
            f.process(&partition, &order_by, 0).unwrap(),
            Value::Float(0.25)
        );
        // Row 1 (value 20): 2/4 = 0.5
        assert_eq!(
            f.process(&partition, &order_by, 1).unwrap(),
            Value::Float(0.5)
        );
        // Row 2 (value 30): 3/4 = 0.75
        assert_eq!(
            f.process(&partition, &order_by, 2).unwrap(),
            Value::Float(0.75)
        );
        // Row 3 (value 40): 4/4 = 1.0
        assert_eq!(
            f.process(&partition, &order_by, 3).unwrap(),
            Value::Float(1.0)
        );
    }

    #[test]
    fn test_cume_dist_with_ties() {
        let f = CumeDistFunction;
        // Values: 10, 10, 30 (two ties)
        let partition = vec![Value::Integer(10), Value::Integer(10), Value::Integer(30)];
        let order_by = vec![];

        // Both rows with value 10: 2/3 ≈ 0.667 (2 rows have value <= 10)
        if let Value::Float(v) = f.process(&partition, &order_by, 0).unwrap() {
            assert!((v - 0.6666666666666666).abs() < 0.0001);
        } else {
            panic!("Expected float");
        }
        if let Value::Float(v) = f.process(&partition, &order_by, 1).unwrap() {
            assert!((v - 0.6666666666666666).abs() < 0.0001);
        } else {
            panic!("Expected float");
        }
        // Row with value 30: 3/3 = 1.0
        assert_eq!(
            f.process(&partition, &order_by, 2).unwrap(),
            Value::Float(1.0)
        );
    }

    #[test]
    fn test_cume_dist_single_row() {
        let f = CumeDistFunction;
        let partition = vec![Value::Integer(10)];
        let order_by = vec![];

        // Single row always has CUME_DIST of 1.0
        assert_eq!(
            f.process(&partition, &order_by, 0).unwrap(),
            Value::Float(1.0)
        );
    }
}
