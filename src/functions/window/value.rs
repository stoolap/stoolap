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

//! FIRST_VALUE, LAST_VALUE, and NTH_VALUE window functions

use crate::core::{Result, Value};
use crate::functions::{
    FunctionDataType, FunctionInfo, FunctionSignature, FunctionType, WindowFunction,
};

/// FIRST_VALUE window function
///
/// Returns the first value in the window frame.
/// If the frame is empty or starts with NULL, returns NULL.
#[derive(Default)]
pub struct FirstValueFunction;

impl WindowFunction for FirstValueFunction {
    fn name(&self) -> &str {
        "FIRST_VALUE"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "FIRST_VALUE",
            FunctionType::Window,
            "Returns the first value in the window frame",
            FunctionSignature::new(FunctionDataType::Any, vec![FunctionDataType::Any], 1, 1),
        )
    }

    fn process(
        &self,
        partition: &[Value],
        _order_by: &[Value],
        _current_row: usize,
    ) -> Result<Value> {
        // Return the first value in the partition
        if partition.is_empty() {
            return Ok(Value::null_unknown());
        }
        Ok(partition[0].clone())
    }

    fn clone_box(&self) -> Box<dyn WindowFunction> {
        Box::new(FirstValueFunction)
    }
}

/// LAST_VALUE window function
///
/// Returns the last value in the window frame.
/// Note: With default frame (RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),
/// this returns the current row's value. To get the true last value,
/// use ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING.
#[derive(Default)]
pub struct LastValueFunction;

impl WindowFunction for LastValueFunction {
    fn name(&self) -> &str {
        "LAST_VALUE"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "LAST_VALUE",
            FunctionType::Window,
            "Returns the last value in the window frame",
            FunctionSignature::new(FunctionDataType::Any, vec![FunctionDataType::Any], 1, 1),
        )
    }

    fn process(
        &self,
        partition: &[Value],
        _order_by: &[Value],
        _current_row: usize,
    ) -> Result<Value> {
        // Return the last value in the partition
        // Note: In a full implementation with frame support, this would respect the frame
        // For now, we return the last value in the entire partition
        if partition.is_empty() {
            return Ok(Value::null_unknown());
        }
        Ok(partition[partition.len() - 1].clone())
    }

    fn clone_box(&self) -> Box<dyn WindowFunction> {
        Box::new(LastValueFunction)
    }
}

/// NTH_VALUE window function
///
/// Returns the value at the specified position (1-indexed) in the window frame.
/// Returns NULL if the position is out of bounds or the partition is empty.
#[derive(Default)]
pub struct NthValueFunction {
    n: usize,
}

impl NthValueFunction {
    /// Create a new NTH_VALUE function with the specified position (1-indexed)
    pub fn new(n: usize) -> Self {
        Self { n }
    }
}

impl WindowFunction for NthValueFunction {
    fn name(&self) -> &str {
        "NTH_VALUE"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "NTH_VALUE",
            FunctionType::Window,
            "Returns the value at the specified position in the window frame",
            FunctionSignature::new(
                FunctionDataType::Any,
                vec![FunctionDataType::Any, FunctionDataType::Integer],
                2,
                2,
            ),
        )
    }

    fn process(
        &self,
        partition: &[Value],
        _order_by: &[Value],
        _current_row: usize,
    ) -> Result<Value> {
        // n is 1-indexed, so we need to subtract 1
        if self.n == 0 || partition.is_empty() {
            return Ok(Value::null_unknown());
        }

        let index = self.n - 1;
        if index >= partition.len() {
            return Ok(Value::null_unknown());
        }

        Ok(partition[index].clone())
    }

    fn clone_box(&self) -> Box<dyn WindowFunction> {
        Box::new(NthValueFunction::new(self.n))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_first_value_basic() {
        let f = FirstValueFunction;
        let partition = vec![
            Value::Integer(10),
            Value::Integer(20),
            Value::Integer(30),
            Value::Integer(40),
        ];
        let order_by = vec![];

        // FIRST_VALUE always returns the first value regardless of current row
        assert_eq!(
            f.process(&partition, &order_by, 0).unwrap(),
            Value::Integer(10)
        );
        assert_eq!(
            f.process(&partition, &order_by, 1).unwrap(),
            Value::Integer(10)
        );
        assert_eq!(
            f.process(&partition, &order_by, 2).unwrap(),
            Value::Integer(10)
        );
        assert_eq!(
            f.process(&partition, &order_by, 3).unwrap(),
            Value::Integer(10)
        );
    }

    #[test]
    fn test_first_value_empty() {
        let f = FirstValueFunction;
        let partition = vec![];
        let order_by = vec![];

        assert!(f.process(&partition, &order_by, 0).unwrap().is_null());
    }

    #[test]
    fn test_first_value_strings() {
        let f = FirstValueFunction;
        let partition = vec![
            Value::text("apple"),
            Value::text("banana"),
            Value::text("cherry"),
        ];
        let order_by = vec![];

        assert_eq!(
            f.process(&partition, &order_by, 2).unwrap(),
            Value::text("apple")
        );
    }

    #[test]
    fn test_last_value_basic() {
        let f = LastValueFunction;
        let partition = vec![
            Value::Integer(10),
            Value::Integer(20),
            Value::Integer(30),
            Value::Integer(40),
        ];
        let order_by = vec![];

        // LAST_VALUE returns the last value in the partition
        assert_eq!(
            f.process(&partition, &order_by, 0).unwrap(),
            Value::Integer(40)
        );
        assert_eq!(
            f.process(&partition, &order_by, 1).unwrap(),
            Value::Integer(40)
        );
        assert_eq!(
            f.process(&partition, &order_by, 2).unwrap(),
            Value::Integer(40)
        );
        assert_eq!(
            f.process(&partition, &order_by, 3).unwrap(),
            Value::Integer(40)
        );
    }

    #[test]
    fn test_last_value_empty() {
        let f = LastValueFunction;
        let partition = vec![];
        let order_by = vec![];

        assert!(f.process(&partition, &order_by, 0).unwrap().is_null());
    }

    #[test]
    fn test_last_value_strings() {
        let f = LastValueFunction;
        let partition = vec![
            Value::text("apple"),
            Value::text("banana"),
            Value::text("cherry"),
        ];
        let order_by = vec![];

        assert_eq!(
            f.process(&partition, &order_by, 0).unwrap(),
            Value::text("cherry")
        );
    }

    #[test]
    fn test_nth_value_basic() {
        let partition = vec![
            Value::Integer(10),
            Value::Integer(20),
            Value::Integer(30),
            Value::Integer(40),
        ];
        let order_by = vec![];

        // NTH_VALUE(1) returns first value
        let f1 = NthValueFunction::new(1);
        assert_eq!(
            f1.process(&partition, &order_by, 0).unwrap(),
            Value::Integer(10)
        );

        // NTH_VALUE(2) returns second value
        let f2 = NthValueFunction::new(2);
        assert_eq!(
            f2.process(&partition, &order_by, 0).unwrap(),
            Value::Integer(20)
        );

        // NTH_VALUE(3) returns third value
        let f3 = NthValueFunction::new(3);
        assert_eq!(
            f3.process(&partition, &order_by, 0).unwrap(),
            Value::Integer(30)
        );

        // NTH_VALUE(4) returns fourth value
        let f4 = NthValueFunction::new(4);
        assert_eq!(
            f4.process(&partition, &order_by, 0).unwrap(),
            Value::Integer(40)
        );
    }

    #[test]
    fn test_nth_value_out_of_bounds() {
        let f = NthValueFunction::new(5);
        let partition = vec![
            Value::Integer(10),
            Value::Integer(20),
            Value::Integer(30),
            Value::Integer(40),
        ];
        let order_by = vec![];

        // NTH_VALUE(5) with only 4 values returns NULL
        assert!(f.process(&partition, &order_by, 0).unwrap().is_null());
    }

    #[test]
    fn test_nth_value_zero() {
        let f = NthValueFunction::new(0);
        let partition = vec![Value::Integer(10), Value::Integer(20)];
        let order_by = vec![];

        // NTH_VALUE(0) is invalid, returns NULL
        assert!(f.process(&partition, &order_by, 0).unwrap().is_null());
    }

    #[test]
    fn test_nth_value_empty() {
        let f = NthValueFunction::new(1);
        let partition = vec![];
        let order_by = vec![];

        assert!(f.process(&partition, &order_by, 0).unwrap().is_null());
    }

    #[test]
    fn test_nth_value_strings() {
        let f = NthValueFunction::new(2);
        let partition = vec![
            Value::text("apple"),
            Value::text("banana"),
            Value::text("cherry"),
        ];
        let order_by = vec![];

        assert_eq!(
            f.process(&partition, &order_by, 0).unwrap(),
            Value::text("banana")
        );
    }
}
