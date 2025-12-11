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

//! LEAD and LAG window functions

use crate::core::{Result, Value};
use crate::functions::{
    FunctionDataType, FunctionInfo, FunctionSignature, FunctionType, WindowFunction,
};

/// LEAD window function
///
/// Returns the value from a row that is `offset` rows after the current row
/// within the partition. If there is no such row, returns the default value.
#[derive(Default)]
pub struct LeadFunction {
    offset: usize,
    default_value: Value,
}

impl LeadFunction {
    /// Create a new LEAD function with the specified offset and default value
    pub fn new(offset: usize, default_value: Value) -> Self {
        Self {
            offset,
            default_value,
        }
    }

    /// Create a new LEAD function with offset 1 and NULL default
    pub fn with_offset(offset: usize) -> Self {
        Self {
            offset,
            default_value: Value::null_unknown(),
        }
    }
}

impl WindowFunction for LeadFunction {
    fn name(&self) -> &str {
        "LEAD"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "LEAD",
            FunctionType::Window,
            "Returns the value from a row that is offset rows after the current row",
            FunctionSignature::new(
                FunctionDataType::Any,
                vec![
                    FunctionDataType::Any,     // column
                    FunctionDataType::Integer, // offset
                    FunctionDataType::Any,     // default
                ],
                1,
                3,
            ),
        )
    }

    fn process(
        &self,
        partition: &[Value],
        _order_by: &[Value],
        current_row: usize,
    ) -> Result<Value> {
        let target_row = current_row + self.offset;

        if target_row < partition.len() {
            Ok(partition[target_row].clone())
        } else {
            Ok(self.default_value.clone())
        }
    }

    fn clone_box(&self) -> Box<dyn WindowFunction> {
        Box::new(LeadFunction::new(self.offset, self.default_value.clone()))
    }
}

/// LAG window function
///
/// Returns the value from a row that is `offset` rows before the current row
/// within the partition. If there is no such row, returns the default value.
#[derive(Default)]
pub struct LagFunction {
    offset: usize,
    default_value: Value,
}

impl LagFunction {
    /// Create a new LAG function with the specified offset and default value
    pub fn new(offset: usize, default_value: Value) -> Self {
        Self {
            offset,
            default_value,
        }
    }

    /// Create a new LAG function with offset 1 and NULL default
    pub fn with_offset(offset: usize) -> Self {
        Self {
            offset,
            default_value: Value::null_unknown(),
        }
    }
}

impl WindowFunction for LagFunction {
    fn name(&self) -> &str {
        "LAG"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "LAG",
            FunctionType::Window,
            "Returns the value from a row that is offset rows before the current row",
            FunctionSignature::new(
                FunctionDataType::Any,
                vec![
                    FunctionDataType::Any,     // column
                    FunctionDataType::Integer, // offset
                    FunctionDataType::Any,     // default
                ],
                1,
                3,
            ),
        )
    }

    fn process(
        &self,
        partition: &[Value],
        _order_by: &[Value],
        current_row: usize,
    ) -> Result<Value> {
        if current_row >= self.offset {
            let target_row = current_row - self.offset;
            if target_row < partition.len() {
                return Ok(partition[target_row].clone());
            }
        }

        Ok(self.default_value.clone())
    }

    fn clone_box(&self) -> Box<dyn WindowFunction> {
        Box::new(LagFunction::new(self.offset, self.default_value.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lead_basic() {
        let f = LeadFunction::with_offset(1);
        let partition = vec![
            Value::Integer(10),
            Value::Integer(20),
            Value::Integer(30),
            Value::Integer(40),
        ];
        let order_by = vec![];

        assert_eq!(
            f.process(&partition, &order_by, 0).unwrap(),
            Value::Integer(20)
        );
        assert_eq!(
            f.process(&partition, &order_by, 1).unwrap(),
            Value::Integer(30)
        );
        assert_eq!(
            f.process(&partition, &order_by, 2).unwrap(),
            Value::Integer(40)
        );
        // Last row has no next row, returns NULL
        assert!(f.process(&partition, &order_by, 3).unwrap().is_null());
    }

    #[test]
    fn test_lead_offset_2() {
        let f = LeadFunction::with_offset(2);
        let partition = vec![
            Value::Integer(10),
            Value::Integer(20),
            Value::Integer(30),
            Value::Integer(40),
        ];
        let order_by = vec![];

        assert_eq!(
            f.process(&partition, &order_by, 0).unwrap(),
            Value::Integer(30)
        );
        assert_eq!(
            f.process(&partition, &order_by, 1).unwrap(),
            Value::Integer(40)
        );
        // No value 2 rows ahead
        assert!(f.process(&partition, &order_by, 2).unwrap().is_null());
        assert!(f.process(&partition, &order_by, 3).unwrap().is_null());
    }

    #[test]
    fn test_lead_with_default() {
        let f = LeadFunction::new(1, Value::Integer(-1));
        let partition = vec![Value::Integer(10), Value::Integer(20)];
        let order_by = vec![];

        assert_eq!(
            f.process(&partition, &order_by, 0).unwrap(),
            Value::Integer(20)
        );
        // Last row returns default value
        assert_eq!(
            f.process(&partition, &order_by, 1).unwrap(),
            Value::Integer(-1)
        );
    }

    #[test]
    fn test_lag_basic() {
        let f = LagFunction::with_offset(1);
        let partition = vec![
            Value::Integer(10),
            Value::Integer(20),
            Value::Integer(30),
            Value::Integer(40),
        ];
        let order_by = vec![];

        // First row has no previous row, returns NULL
        assert!(f.process(&partition, &order_by, 0).unwrap().is_null());
        assert_eq!(
            f.process(&partition, &order_by, 1).unwrap(),
            Value::Integer(10)
        );
        assert_eq!(
            f.process(&partition, &order_by, 2).unwrap(),
            Value::Integer(20)
        );
        assert_eq!(
            f.process(&partition, &order_by, 3).unwrap(),
            Value::Integer(30)
        );
    }

    #[test]
    fn test_lag_offset_2() {
        let f = LagFunction::with_offset(2);
        let partition = vec![
            Value::Integer(10),
            Value::Integer(20),
            Value::Integer(30),
            Value::Integer(40),
        ];
        let order_by = vec![];

        // No value 2 rows behind
        assert!(f.process(&partition, &order_by, 0).unwrap().is_null());
        assert!(f.process(&partition, &order_by, 1).unwrap().is_null());
        assert_eq!(
            f.process(&partition, &order_by, 2).unwrap(),
            Value::Integer(10)
        );
        assert_eq!(
            f.process(&partition, &order_by, 3).unwrap(),
            Value::Integer(20)
        );
    }

    #[test]
    fn test_lag_with_default() {
        let f = LagFunction::new(1, Value::Integer(0));
        let partition = vec![Value::Integer(10), Value::Integer(20)];
        let order_by = vec![];

        // First row returns default value
        assert_eq!(
            f.process(&partition, &order_by, 0).unwrap(),
            Value::Integer(0)
        );
        assert_eq!(
            f.process(&partition, &order_by, 1).unwrap(),
            Value::Integer(10)
        );
    }

    #[test]
    fn test_lead_empty_partition() {
        let f = LeadFunction::with_offset(1);
        let partition = vec![];
        let order_by = vec![];

        // Returns NULL for empty partition
        assert!(f.process(&partition, &order_by, 0).unwrap().is_null());
    }

    #[test]
    fn test_lag_empty_partition() {
        let f = LagFunction::with_offset(1);
        let partition = vec![];
        let order_by = vec![];

        // Returns NULL for empty partition
        assert!(f.process(&partition, &order_by, 0).unwrap().is_null());
    }

    #[test]
    fn test_lead_strings() {
        let f = LeadFunction::with_offset(1);
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
        assert_eq!(
            f.process(&partition, &order_by, 1).unwrap(),
            Value::text("cherry")
        );
        assert!(f.process(&partition, &order_by, 2).unwrap().is_null());
    }

    #[test]
    fn test_lag_strings() {
        let f = LagFunction::with_offset(1);
        let partition = vec![
            Value::text("apple"),
            Value::text("banana"),
            Value::text("cherry"),
        ];
        let order_by = vec![];

        assert!(f.process(&partition, &order_by, 0).unwrap().is_null());
        assert_eq!(
            f.process(&partition, &order_by, 1).unwrap(),
            Value::text("apple")
        );
        assert_eq!(
            f.process(&partition, &order_by, 2).unwrap(),
            Value::text("banana")
        );
    }
}
