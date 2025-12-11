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

//! Function Registry
//!
//! This module provides the function registry for looking up and managing
//! SQL functions (aggregate, scalar, and window functions).

use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

/// Global function registry instance
static GLOBAL_REGISTRY: OnceLock<FunctionRegistry> = OnceLock::new();

/// Get the global function registry
#[inline]
pub fn global_registry() -> &'static FunctionRegistry {
    GLOBAL_REGISTRY.get_or_init(FunctionRegistry::new)
}

use super::aggregate::{
    ArrayAggFunction, AvgFunction, CountFunction, FirstFunction, GroupConcatFunction, LastFunction,
    MaxFunction, MedianFunction, MinFunction, StddevFunction, StddevPopFunction,
    StddevSampFunction, StringAggFunction, SumFunction, VarPopFunction, VarSampFunction,
    VarianceFunction,
};
use super::scalar::{
    AbsFunction, CastFunction, CeilFunction, CeilingFunction, CharFunction, CharLengthFunction,
    CoalesceFunction, CollateFunction, ConcatFunction, ConcatWsFunction, CosFunction,
    CurrentDateFunction, CurrentTimestampFunction, DateAddFunction, DateDiffAliasFunction,
    DateDiffFunction, DateSubFunction, DateTruncFunction, DayFunction, ExpFunction,
    ExtractFunction, FloorFunction, GreatestFunction, HourFunction, IfNullFunction, IifFunction,
    InstrFunction, JsonArrayFunction, JsonArrayLengthFunction, JsonExtractFunction,
    JsonKeysFunction, JsonObjectFunction, JsonTypeFunction, JsonTypeOfFunction, JsonValidFunction,
    LeastFunction, LeftFunction, LengthFunction, LnFunction, LocateFunction, Log10Function,
    Log2Function, LogFunction, LowerFunction, LpadFunction, LtrimFunction, MinuteFunction,
    ModFunction, MonthFunction, NowFunction, NullIfFunction, PiFunction, PositionFunction,
    PowFunction, PowerFunction, RandomFunction, RepeatFunction, ReplaceFunction, ReverseFunction,
    RightFunction, RoundFunction, RpadFunction, RtrimFunction, SecondFunction, SignFunction,
    SinFunction, SleepFunction, SplitPartFunction, SqrtFunction, StrposFunction, SubstrFunction,
    SubstringFunction, TanFunction, TimeTruncFunction, ToCharFunction, TrimFunction, TruncFunction,
    TruncateFunction, TypeOfFunction, UpperFunction, VersionFunction, YearFunction,
};
use super::window::{
    CumeDistFunction, DenseRankFunction, FirstValueFunction, LagFunction, LastValueFunction,
    LeadFunction, NthValueFunction, NtileFunction, PercentRankFunction, RankFunction,
    RowNumberFunction,
};
use super::{AggregateFunction, FunctionInfo, ScalarFunction, WindowFunction};

/// Type alias for aggregate function factory
type AggregateFnFactory = Arc<dyn Fn() -> Box<dyn AggregateFunction> + Send + Sync>;
/// Type alias for scalar function factory
type ScalarFnFactory = Arc<dyn Fn() -> Box<dyn ScalarFunction> + Send + Sync>;
/// Type alias for window function factory
type WindowFnFactory = Arc<dyn Fn() -> Box<dyn WindowFunction> + Send + Sync>;

/// Function registry for SQL functions
pub struct FunctionRegistry {
    /// Aggregate functions
    aggregate_functions: RwLock<HashMap<String, AggregateFnFactory>>,
    /// Scalar functions
    scalar_functions: RwLock<HashMap<String, ScalarFnFactory>>,
    /// Window functions
    window_functions: RwLock<HashMap<String, WindowFnFactory>>,
    /// Function info cache
    function_info: RwLock<HashMap<String, FunctionInfo>>,
}

impl Default for FunctionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl FunctionRegistry {
    /// Create a new function registry with all built-in functions registered
    pub fn new() -> Self {
        let registry = Self {
            aggregate_functions: RwLock::new(HashMap::new()),
            scalar_functions: RwLock::new(HashMap::new()),
            window_functions: RwLock::new(HashMap::new()),
            function_info: RwLock::new(HashMap::new()),
        };

        // Register built-in aggregate functions
        registry.register_aggregate::<CountFunction>();
        registry.register_aggregate::<SumFunction>();
        registry.register_aggregate::<AvgFunction>();
        registry.register_aggregate::<MinFunction>();
        registry.register_aggregate::<MaxFunction>();
        registry.register_aggregate::<FirstFunction>();
        registry.register_aggregate::<LastFunction>();
        registry.register_aggregate::<StringAggFunction>();
        registry.register_aggregate::<GroupConcatFunction>();
        registry.register_aggregate::<ArrayAggFunction>();
        registry.register_aggregate::<StddevPopFunction>();
        registry.register_aggregate::<StddevFunction>();
        registry.register_aggregate::<StddevSampFunction>();
        registry.register_aggregate::<VarPopFunction>();
        registry.register_aggregate::<VarianceFunction>();
        registry.register_aggregate::<VarSampFunction>();
        registry.register_aggregate::<MedianFunction>();

        // Register built-in scalar functions
        // String functions
        registry.register_scalar::<UpperFunction>();
        registry.register_scalar::<LowerFunction>();
        registry.register_scalar::<LengthFunction>();
        registry.register_scalar::<CharLengthFunction>();
        registry.register_scalar::<CharFunction>();
        registry.register_scalar::<ConcatFunction>();
        registry.register_scalar::<ConcatWsFunction>();
        registry.register_scalar::<SubstringFunction>();
        registry.register_scalar::<SubstrFunction>();
        registry.register_scalar::<TrimFunction>();
        registry.register_scalar::<LtrimFunction>();
        registry.register_scalar::<RtrimFunction>();
        registry.register_scalar::<ReplaceFunction>();
        registry.register_scalar::<ReverseFunction>();
        registry.register_scalar::<LeftFunction>();
        registry.register_scalar::<RightFunction>();
        registry.register_scalar::<RepeatFunction>();
        registry.register_scalar::<SplitPartFunction>();
        registry.register_scalar::<PositionFunction>();
        registry.register_scalar::<StrposFunction>();
        registry.register_scalar::<InstrFunction>();
        registry.register_scalar::<LocateFunction>();
        registry.register_scalar::<LpadFunction>();
        registry.register_scalar::<RpadFunction>();

        // Math functions
        registry.register_scalar::<AbsFunction>();
        registry.register_scalar::<RoundFunction>();
        registry.register_scalar::<FloorFunction>();
        registry.register_scalar::<CeilingFunction>();
        registry.register_scalar::<CeilFunction>();
        registry.register_scalar::<ModFunction>();
        registry.register_scalar::<PowerFunction>();
        registry.register_scalar::<PowFunction>();
        registry.register_scalar::<SqrtFunction>();
        registry.register_scalar::<LogFunction>();
        registry.register_scalar::<Log10Function>();
        registry.register_scalar::<Log2Function>();
        registry.register_scalar::<LnFunction>();
        registry.register_scalar::<ExpFunction>();
        registry.register_scalar::<SignFunction>();
        registry.register_scalar::<TruncateFunction>();
        registry.register_scalar::<TruncFunction>();
        registry.register_scalar::<PiFunction>();
        registry.register_scalar::<RandomFunction>();
        registry.register_scalar::<SinFunction>();
        registry.register_scalar::<CosFunction>();
        registry.register_scalar::<TanFunction>();

        // Date/Time functions
        registry.register_scalar::<NowFunction>();
        registry.register_scalar::<CurrentDateFunction>();
        registry.register_scalar::<CurrentTimestampFunction>();
        registry.register_scalar::<DateTruncFunction>();
        registry.register_scalar::<TimeTruncFunction>();
        registry.register_scalar::<ExtractFunction>();
        registry.register_scalar::<YearFunction>();
        registry.register_scalar::<MonthFunction>();
        registry.register_scalar::<DayFunction>();
        registry.register_scalar::<HourFunction>();
        registry.register_scalar::<MinuteFunction>();
        registry.register_scalar::<SecondFunction>();
        registry.register_scalar::<DateAddFunction>();
        registry.register_scalar::<DateSubFunction>();
        registry.register_scalar::<DateDiffFunction>();
        registry.register_scalar::<DateDiffAliasFunction>(); // DATE_DIFF alias for DATEDIFF
        registry.register_scalar::<VersionFunction>();
        registry.register_scalar::<ToCharFunction>();

        // Utility functions
        registry.register_scalar::<CoalesceFunction>();
        registry.register_scalar::<NullIfFunction>();
        registry.register_scalar::<IfNullFunction>();
        registry.register_scalar::<CastFunction>();
        registry.register_scalar::<CollateFunction>();
        registry.register_scalar::<GreatestFunction>();
        registry.register_scalar::<LeastFunction>();
        registry.register_scalar::<IifFunction>();
        registry.register_scalar::<JsonExtractFunction>();
        registry.register_scalar::<JsonArrayLengthFunction>();
        registry.register_scalar::<JsonArrayFunction>();
        registry.register_scalar::<JsonObjectFunction>();
        registry.register_scalar::<JsonTypeFunction>();
        registry.register_scalar::<JsonTypeOfFunction>();
        registry.register_scalar::<JsonValidFunction>();
        registry.register_scalar::<JsonKeysFunction>();
        registry.register_scalar::<TypeOfFunction>();
        registry.register_scalar::<SleepFunction>();

        // Register built-in window functions
        registry.register_window::<RowNumberFunction>();
        registry.register_window::<RankFunction>();
        registry.register_window::<DenseRankFunction>();
        registry.register_window::<NtileFunction>();
        registry.register_window::<LeadFunction>();
        registry.register_window::<LagFunction>();
        registry.register_window::<FirstValueFunction>();
        registry.register_window::<LastValueFunction>();
        registry.register_window::<NthValueFunction>();
        registry.register_window::<PercentRankFunction>();
        registry.register_window::<CumeDistFunction>();

        registry
    }

    /// Register an aggregate function
    pub fn register_aggregate<F: AggregateFunction + Default + 'static>(&self) {
        let instance = F::default();
        let name = instance.name().to_uppercase();
        let info = instance.info();

        let mut funcs = self.aggregate_functions.write().unwrap();
        funcs.insert(name.clone(), Arc::new(|| Box::new(F::default())));

        let mut infos = self.function_info.write().unwrap();
        infos.insert(name, info);
    }

    /// Register a scalar function
    pub fn register_scalar<F: ScalarFunction + Default + 'static>(&self) {
        let instance = F::default();
        let name = instance.name().to_uppercase();
        let info = instance.info();

        let mut funcs = self.scalar_functions.write().unwrap();
        funcs.insert(name.clone(), Arc::new(|| Box::new(F::default())));

        let mut infos = self.function_info.write().unwrap();
        infos.insert(name, info);
    }

    /// Register a window function
    pub fn register_window<F: WindowFunction + Default + 'static>(&self) {
        let instance = F::default();
        let name = instance.name().to_uppercase();
        let info = instance.info();

        let mut funcs = self.window_functions.write().unwrap();
        funcs.insert(name.clone(), Arc::new(|| Box::new(F::default())));

        let mut infos = self.function_info.write().unwrap();
        infos.insert(name, info);
    }

    /// Get a new instance of an aggregate function by name
    pub fn get_aggregate(&self, name: &str) -> Option<Box<dyn AggregateFunction>> {
        // OPTIMIZATION: Fast path - if name is already uppercase, avoid allocation
        let funcs = self.aggregate_functions.read().unwrap();
        if let Some(f) = funcs.get(name) {
            return Some(f());
        }
        // Slow path - try uppercase
        let upper = name.to_uppercase();
        funcs.get(&upper).map(|f| f())
    }

    /// Get a new instance of a scalar function by name
    pub fn get_scalar(&self, name: &str) -> Option<Box<dyn ScalarFunction>> {
        // OPTIMIZATION: Fast path - if name is already uppercase, avoid allocation
        let funcs = self.scalar_functions.read().unwrap();
        if let Some(f) = funcs.get(name) {
            return Some(f());
        }
        // Slow path - try uppercase
        let upper = name.to_uppercase();
        funcs.get(&upper).map(|f| f())
    }

    /// Get a new instance of a window function by name
    pub fn get_window(&self, name: &str) -> Option<Box<dyn WindowFunction>> {
        // OPTIMIZATION: Fast path - if name is already uppercase, avoid allocation
        let funcs = self.window_functions.read().unwrap();
        if let Some(f) = funcs.get(name) {
            return Some(f());
        }
        // Slow path - try uppercase
        let upper = name.to_uppercase();
        funcs.get(&upper).map(|f| f())
    }

    /// Check if a function name is an aggregate function
    pub fn is_aggregate(&self, name: &str) -> bool {
        // OPTIMIZATION: Fast path - if name is already uppercase, avoid allocation
        let funcs = self.aggregate_functions.read().unwrap();
        if funcs.contains_key(name) {
            return true;
        }
        // Slow path - try uppercase
        let upper = name.to_uppercase();
        funcs.contains_key(&upper)
    }

    /// Check if a function name is a scalar function
    pub fn is_scalar(&self, name: &str) -> bool {
        // OPTIMIZATION: Fast path - if name is already uppercase, avoid allocation
        let funcs = self.scalar_functions.read().unwrap();
        if funcs.contains_key(name) {
            return true;
        }
        // Slow path - try uppercase
        let upper = name.to_uppercase();
        funcs.contains_key(&upper)
    }

    /// Check if a function name is a window function
    pub fn is_window(&self, name: &str) -> bool {
        // OPTIMIZATION: Fast path - if name is already uppercase, avoid allocation
        let funcs = self.window_functions.read().unwrap();
        if funcs.contains_key(name) {
            return true;
        }
        // Slow path - try uppercase
        let upper = name.to_uppercase();
        funcs.contains_key(&upper)
    }

    /// Check if a function exists
    pub fn exists(&self, name: &str) -> bool {
        self.is_aggregate(name) || self.is_scalar(name) || self.is_window(name)
    }

    /// Get function info by name
    pub fn get_info(&self, name: &str) -> Option<FunctionInfo> {
        let name = name.to_uppercase();
        let infos = self.function_info.read().unwrap();
        infos.get(&name).cloned()
    }

    /// List all aggregate function names
    pub fn list_aggregates(&self) -> Vec<String> {
        let funcs = self.aggregate_functions.read().unwrap();
        funcs.keys().cloned().collect()
    }

    /// List all scalar function names
    pub fn list_scalars(&self) -> Vec<String> {
        let funcs = self.scalar_functions.read().unwrap();
        funcs.keys().cloned().collect()
    }

    /// List all window function names
    pub fn list_windows(&self) -> Vec<String> {
        let funcs = self.window_functions.read().unwrap();
        funcs.keys().cloned().collect()
    }

    /// List all function names
    pub fn list_all(&self) -> Vec<String> {
        let mut names = Vec::new();
        names.extend(self.list_aggregates());
        names.extend(self.list_scalars());
        names.extend(self.list_windows());
        names.sort();
        names
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_new() {
        let registry = FunctionRegistry::new();
        assert!(registry.is_aggregate("COUNT"));
        assert!(registry.is_aggregate("SUM"));
        assert!(registry.is_aggregate("AVG"));
        assert!(registry.is_aggregate("MIN"));
        assert!(registry.is_aggregate("MAX"));
    }

    #[test]
    fn test_registry_case_insensitive() {
        let registry = FunctionRegistry::new();
        assert!(registry.is_aggregate("count"));
        assert!(registry.is_aggregate("COUNT"));
        assert!(registry.is_aggregate("Count"));
    }

    #[test]
    fn test_get_aggregate() {
        let registry = FunctionRegistry::new();
        let count = registry.get_aggregate("COUNT");
        assert!(count.is_some());
        assert_eq!(count.unwrap().name(), "COUNT");
    }

    #[test]
    fn test_get_scalar() {
        let registry = FunctionRegistry::new();
        let upper = registry.get_scalar("UPPER");
        assert!(upper.is_some());
        assert_eq!(upper.unwrap().name(), "UPPER");
    }

    #[test]
    fn test_get_window() {
        let registry = FunctionRegistry::new();
        let row_number = registry.get_window("ROW_NUMBER");
        assert!(row_number.is_some());
        assert_eq!(row_number.unwrap().name(), "ROW_NUMBER");
    }

    #[test]
    fn test_function_info() {
        let registry = FunctionRegistry::new();
        let info = registry.get_info("COUNT");
        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.name, "COUNT");
    }

    #[test]
    fn test_list_functions() {
        let registry = FunctionRegistry::new();
        let aggregates = registry.list_aggregates();
        assert!(aggregates.contains(&"COUNT".to_string()));
        assert!(aggregates.contains(&"SUM".to_string()));

        let scalars = registry.list_scalars();
        assert!(scalars.contains(&"UPPER".to_string()));
        assert!(scalars.contains(&"LOWER".to_string()));

        let windows = registry.list_windows();
        assert!(windows.contains(&"ROW_NUMBER".to_string()));
    }

    #[test]
    fn test_global_registry() {
        let registry = global_registry();
        assert!(registry.is_aggregate("COUNT"));
        assert!(registry.is_scalar("UPPER"));
        assert!(registry.is_window("ROW_NUMBER"));
    }

    #[test]
    fn test_exists() {
        let registry = FunctionRegistry::new();
        assert!(registry.exists("COUNT"));
        assert!(registry.exists("UPPER"));
        assert!(registry.exists("ROW_NUMBER"));
        assert!(!registry.exists("NONEXISTENT"));
    }
}
