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

//! SQL Function System
//!
//! This module provides the function system for Stoolap SQL, including:
//!
//! - [`AggregateFunction`] - Aggregate functions (COUNT, SUM, AVG, MIN, MAX, etc.)
//! - [`ScalarFunction`] - Scalar functions (UPPER, LOWER, ABS, ROUND, etc.)
//! - [`WindowFunction`] - Window functions (ROW_NUMBER, RANK, etc.)
//! - [`FunctionRegistry`] - Registry for function lookup and validation

pub mod aggregate;
pub mod registry;
pub mod scalar;
pub mod window;

use crate::core::{Error, Result, Value};

/// Function type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FunctionType {
    /// Aggregate function (operates on multiple rows)
    Aggregate,
    /// Scalar function (operates on a single row)
    Scalar,
    /// Window function (operates over a window of rows)
    Window,
}

/// Data type for function signatures
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FunctionDataType {
    /// Any type
    Any,
    /// Integer type
    Integer,
    /// Float type
    Float,
    /// String type
    String,
    /// Boolean type
    Boolean,
    /// Timestamp type
    Timestamp,
    /// Date type
    Date,
    /// Time type
    Time,
    /// DateTime type (alias for Timestamp)
    DateTime,
    /// JSON type
    Json,
    /// Unknown type
    Unknown,
}

/// Function signature information
#[derive(Debug, Clone)]
pub struct FunctionSignature {
    /// Return type
    pub return_type: FunctionDataType,
    /// Argument types
    pub argument_types: Vec<FunctionDataType>,
    /// Minimum number of arguments
    pub min_args: usize,
    /// Maximum number of arguments
    pub max_args: usize,
    /// Whether the function is variadic
    pub is_variadic: bool,
}

impl FunctionSignature {
    /// Create a new function signature
    pub fn new(
        return_type: FunctionDataType,
        argument_types: Vec<FunctionDataType>,
        min_args: usize,
        max_args: usize,
    ) -> Self {
        Self {
            return_type,
            argument_types,
            min_args,
            max_args,
            is_variadic: false,
        }
    }

    /// Create a variadic function signature
    pub fn variadic(return_type: FunctionDataType, arg_type: FunctionDataType) -> Self {
        Self {
            return_type,
            argument_types: vec![arg_type],
            min_args: 1,
            max_args: usize::MAX,
            is_variadic: true,
        }
    }

    /// Validate argument count
    pub fn validate_arg_count(&self, count: usize) -> Result<()> {
        if count < self.min_args {
            return Err(Error::invalid_argument(format!(
                "expected at least {} arguments, got {}",
                self.min_args, count
            )));
        }
        if count > self.max_args {
            return Err(Error::invalid_argument(format!(
                "expected at most {} arguments, got {}",
                self.max_args, count
            )));
        }
        Ok(())
    }
}

/// Function information
#[derive(Debug, Clone)]
pub struct FunctionInfo {
    /// Function name
    pub name: String,
    /// Function type
    pub function_type: FunctionType,
    /// Description
    pub description: String,
    /// Signature
    pub signature: FunctionSignature,
}

impl FunctionInfo {
    /// Create a new function info
    pub fn new(
        name: impl Into<String>,
        function_type: FunctionType,
        description: impl Into<String>,
        signature: FunctionSignature,
    ) -> Self {
        Self {
            name: name.into(),
            function_type,
            description: description.into(),
            signature,
        }
    }

    /// Get the function name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the function type
    pub fn function_type(&self) -> FunctionType {
        self.function_type
    }

    /// Get the description
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Get the signature
    pub fn signature(&self) -> &FunctionSignature {
        &self.signature
    }
}

/// Trait for aggregate functions
pub trait AggregateFunction: Send + Sync {
    /// Get the function name
    fn name(&self) -> &str;

    /// Get function information
    fn info(&self) -> FunctionInfo;

    /// Configure the function with additional arguments
    ///
    /// This is called once before accumulation with any extra arguments
    /// beyond the main column. For example, STRING_AGG(name, ', ') would
    /// receive &[Value::Text(", ")] as the options.
    ///
    /// Default implementation ignores options.
    fn configure(&mut self, _options: &[Value]) {
        // Default: ignore options
    }

    /// Configure ORDER BY for ordered-set aggregates like ARRAY_AGG, STRING_AGG
    ///
    /// The `directions` parameter contains true for ASC, false for DESC for each sort key.
    /// Default implementation ignores ordering.
    fn set_order_by(&mut self, _directions: Vec<bool>) {
        // Default: ignore ordering
    }

    /// Accumulate a value into the aggregate
    fn accumulate(&mut self, value: &Value, distinct: bool);

    /// Accumulate a value with sort keys for ordered aggregates
    ///
    /// The `sort_keys` are the evaluated ORDER BY expressions for this row.
    /// Default implementation falls back to regular accumulate, ignoring sort keys.
    fn accumulate_with_sort_key(&mut self, value: &Value, sort_keys: Vec<Value>, distinct: bool) {
        // Default: ignore sort keys, use regular accumulate
        let _ = sort_keys;
        self.accumulate(value, distinct);
    }

    /// Check if this aggregate supports ORDER BY clause
    ///
    /// If true, the executor will call accumulate_with_sort_key instead of accumulate.
    fn supports_order_by(&self) -> bool {
        false
    }

    /// Get the final result
    fn result(&self) -> Value;

    /// Reset the aggregate state
    fn reset(&mut self);

    /// Clone the function into a new instance
    fn clone_box(&self) -> Box<dyn AggregateFunction>;
}

/// Trait for scalar functions
pub trait ScalarFunction: Send + Sync {
    /// Get the function name
    fn name(&self) -> &str;

    /// Get function information
    fn info(&self) -> FunctionInfo;

    /// Evaluate the function with the given arguments
    fn evaluate(&self, args: &[Value]) -> Result<Value>;

    /// Clone the function into a new instance
    fn clone_box(&self) -> Box<dyn ScalarFunction>;
}

/// Trait for window functions
pub trait WindowFunction: Send + Sync {
    /// Get the function name
    fn name(&self) -> &str;

    /// Get function information
    fn info(&self) -> FunctionInfo;

    /// Process the window function
    ///
    /// # Arguments
    /// * `partition` - The partition of values
    /// * `order_by` - The ordering keys
    /// * `current_row` - The current row index (0-based)
    fn process(&self, partition: &[Value], order_by: &[Value], current_row: usize)
        -> Result<Value>;

    /// Clone the function into a new instance
    fn clone_box(&self) -> Box<dyn WindowFunction>;
}

// Re-export main types
pub use aggregate::{
    AvgFunction, CountFunction, FirstFunction, LastFunction, MaxFunction, MinFunction, SumFunction,
};
pub use registry::{global_registry, FunctionRegistry};
pub use scalar::{
    AbsFunction, CastFunction, CeilingFunction, CoalesceFunction, CollateFunction, ConcatFunction,
    DateTruncFunction, FloorFunction, IfNullFunction, LengthFunction, LowerFunction, NowFunction,
    NullIfFunction, RoundFunction, SubstringFunction, TimeTruncFunction, UpperFunction,
    VersionFunction,
};
pub use window::{
    DenseRankFunction, LagFunction, LeadFunction, NtileFunction, RankFunction, RowNumberFunction,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_function_signature_validation() {
        let sig =
            FunctionSignature::new(FunctionDataType::Integer, vec![FunctionDataType::Any], 1, 1);
        assert!(sig.validate_arg_count(1).is_ok());
        assert!(sig.validate_arg_count(0).is_err());
        assert!(sig.validate_arg_count(2).is_err());
    }

    #[test]
    fn test_variadic_signature() {
        let sig = FunctionSignature::variadic(FunctionDataType::String, FunctionDataType::Any);
        assert!(sig.is_variadic);
        assert!(sig.validate_arg_count(1).is_ok());
        assert!(sig.validate_arg_count(10).is_ok());
        assert!(sig.validate_arg_count(0).is_err());
    }

    #[test]
    fn test_function_info() {
        let info = FunctionInfo::new(
            "TEST",
            FunctionType::Scalar,
            "Test function",
            FunctionSignature::new(FunctionDataType::Integer, vec![], 0, 0),
        );
        assert_eq!(info.name, "TEST");
        assert_eq!(info.function_type, FunctionType::Scalar);
    }
}
