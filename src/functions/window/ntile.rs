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

//! NTILE window function

use crate::core::{Result, Value};
use crate::functions::{
    FunctionDataType, FunctionInfo, FunctionSignature, FunctionType, WindowFunction,
};

/// NTILE window function
///
/// Divides the partition into n roughly equal groups and returns the group number
/// (1 to n) that each row belongs to.
///
/// If the partition doesn't divide evenly, the first groups get one extra row.
#[derive(Default)]
pub struct NtileFunction {
    num_buckets: i64,
}

impl NtileFunction {
    /// Create a new NTILE function with the specified number of buckets
    pub fn new(num_buckets: i64) -> Self {
        Self { num_buckets }
    }
}

impl WindowFunction for NtileFunction {
    fn name(&self) -> &str {
        "NTILE"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "NTILE",
            FunctionType::Window,
            "Divides the partition into n buckets and returns the bucket number",
            FunctionSignature::new(
                FunctionDataType::Integer,
                vec![FunctionDataType::Integer],
                1,
                1,
            ),
        )
    }

    fn process(
        &self,
        partition: &[Value],
        _order_by: &[Value],
        current_row: usize,
    ) -> Result<Value> {
        let n = self.num_buckets;
        if n <= 0 {
            return Ok(Value::Integer(1));
        }

        let total_rows = partition.len().max(1) as i64;
        let row_num = (current_row + 1) as i64; // 1-based

        // Calculate which bucket this row belongs to
        // Rows are distributed as evenly as possible
        // If total_rows = 10 and n = 3:
        //   - Bucket 1 gets rows 1-4 (4 rows)
        //   - Bucket 2 gets rows 5-7 (3 rows)
        //   - Bucket 3 gets rows 8-10 (3 rows)

        let base_size = total_rows / n;
        let remainder = total_rows % n;

        // First 'remainder' buckets get (base_size + 1) rows
        // Remaining buckets get base_size rows

        let bucket = if row_num <= remainder * (base_size + 1) {
            // In one of the first 'remainder' buckets (they have base_size + 1 rows)
            (row_num - 1) / (base_size + 1) + 1
        } else {
            // In one of the remaining buckets (they have base_size rows)
            let rows_in_larger_buckets = remainder * (base_size + 1);
            let remaining_row = row_num - rows_in_larger_buckets;
            if base_size > 0 {
                remainder + (remaining_row - 1) / base_size + 1
            } else {
                n // All rows go in the last bucket if base_size is 0
            }
        };

        Ok(Value::Integer(bucket.min(n)))
    }

    fn clone_box(&self) -> Box<dyn WindowFunction> {
        Box::new(NtileFunction::new(self.num_buckets))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ntile_even_distribution() {
        // 6 rows into 3 buckets = 2 rows per bucket
        let f = NtileFunction::new(3);
        let partition = vec![
            Value::Integer(1),
            Value::Integer(2),
            Value::Integer(3),
            Value::Integer(4),
            Value::Integer(5),
            Value::Integer(6),
        ];
        let order_by = vec![];

        assert_eq!(
            f.process(&partition, &order_by, 0).unwrap(),
            Value::Integer(1)
        );
        assert_eq!(
            f.process(&partition, &order_by, 1).unwrap(),
            Value::Integer(1)
        );
        assert_eq!(
            f.process(&partition, &order_by, 2).unwrap(),
            Value::Integer(2)
        );
        assert_eq!(
            f.process(&partition, &order_by, 3).unwrap(),
            Value::Integer(2)
        );
        assert_eq!(
            f.process(&partition, &order_by, 4).unwrap(),
            Value::Integer(3)
        );
        assert_eq!(
            f.process(&partition, &order_by, 5).unwrap(),
            Value::Integer(3)
        );
    }

    #[test]
    fn test_ntile_uneven_distribution() {
        // 5 rows into 3 buckets = 2, 2, 1
        let f = NtileFunction::new(3);
        let partition = vec![
            Value::Integer(1),
            Value::Integer(2),
            Value::Integer(3),
            Value::Integer(4),
            Value::Integer(5),
        ];
        let order_by = vec![];

        // First bucket gets 2 rows (row 1-2)
        assert_eq!(
            f.process(&partition, &order_by, 0).unwrap(),
            Value::Integer(1)
        );
        assert_eq!(
            f.process(&partition, &order_by, 1).unwrap(),
            Value::Integer(1)
        );
        // Second bucket gets 2 rows (row 3-4)
        assert_eq!(
            f.process(&partition, &order_by, 2).unwrap(),
            Value::Integer(2)
        );
        assert_eq!(
            f.process(&partition, &order_by, 3).unwrap(),
            Value::Integer(2)
        );
        // Third bucket gets 1 row (row 5)
        assert_eq!(
            f.process(&partition, &order_by, 4).unwrap(),
            Value::Integer(3)
        );
    }

    #[test]
    fn test_ntile_more_buckets_than_rows() {
        // 3 rows into 5 buckets
        let f = NtileFunction::new(5);
        let partition = vec![Value::Integer(1), Value::Integer(2), Value::Integer(3)];
        let order_by = vec![];

        // Each row gets its own bucket (1, 2, 3)
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
    fn test_ntile_single_bucket() {
        let f = NtileFunction::new(1);
        let partition = vec![Value::Integer(1), Value::Integer(2), Value::Integer(3)];
        let order_by = vec![];

        // All rows in bucket 1
        assert_eq!(
            f.process(&partition, &order_by, 0).unwrap(),
            Value::Integer(1)
        );
        assert_eq!(
            f.process(&partition, &order_by, 1).unwrap(),
            Value::Integer(1)
        );
        assert_eq!(
            f.process(&partition, &order_by, 2).unwrap(),
            Value::Integer(1)
        );
    }

    #[test]
    fn test_ntile_zero_buckets() {
        let f = NtileFunction::new(0);
        let partition = vec![Value::Integer(1)];
        let order_by = vec![];

        // Should return 1 for invalid bucket count
        assert_eq!(
            f.process(&partition, &order_by, 0).unwrap(),
            Value::Integer(1)
        );
    }

    #[test]
    fn test_ntile_negative_buckets() {
        let f = NtileFunction::new(-5);
        let partition = vec![Value::Integer(1)];
        let order_by = vec![];

        // Should return 1 for invalid bucket count
        assert_eq!(
            f.process(&partition, &order_by, 0).unwrap(),
            Value::Integer(1)
        );
    }
}
