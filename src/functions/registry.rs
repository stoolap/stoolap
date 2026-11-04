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

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock, RwLock};

use crate::common::{StringMap, StringSet};
use crate::core::{Error, Result};

/// Global function registry instance
static GLOBAL_REGISTRY: OnceLock<FunctionRegistry> = OnceLock::new();

/// The functions the engine computes itself, which the registry holds no
/// entry for: the compiler loads the transaction id, grouping reads its flag
const ENGINE_FUNCTIONS: [&str; 2] = ["CURRENT_TRANSACTION_ID", "GROUPING"];

/// The kinds a name may hold a function of, as indexes of its volatilities
const SCALAR: usize = 0;
const AGGREGATE: usize = 1;
const WINDOW: usize = 2;

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
#[cfg(feature = "semantic")]
use super::scalar::semantic::EmbedFunction;
use super::scalar::vector::{
    VecDimsFunction, VecDistanceCosineFunction, VecDistanceIpFunction, VecDistanceL2Function,
    VecNormFunction, VecToTextFunction,
};
use super::scalar::{
    AbsFunction, CastFunction, CeilFunction, CeilingFunction, CharFunction, CharLengthFunction,
    CoalesceFunction, CollateFunction, ConcatFunction, ConcatWsFunction, ContainsFunction,
    CosFunction, Crc32Function, CurrentDateFunction, CurrentTimeFunction, CurrentTimestampFunction,
    DateAddFunction, DateDiffAliasFunction, DateDiffFunction, DateSubFunction, DateTruncFunction,
    DayFunction, EndsWithFunction, ExpFunction, ExtractFunction, FloorFunction, GreatestFunction,
    HourFunction, IfNullFunction, IifFunction, InstrFunction, JsonArrayFunction,
    JsonArrayLengthFunction, JsonExtractFunction, JsonKeysFunction, JsonObjectFunction,
    JsonTypeFunction, JsonTypeOfFunction, JsonValidFunction, LeastFunction, LeftFunction,
    LengthFunction, LnFunction, LocateFunction, Log10Function, Log2Function, LogFunction,
    LowerFunction, LpadFunction, LtrimFunction, Md5Function, MinuteFunction, ModFunction,
    MonthFunction, NowFunction, NullIfFunction, PiFunction, PositionFunction, PowFunction,
    PowerFunction, RandomFunction, RepeatFunction, ReplaceFunction, ReverseFunction, RightFunction,
    RoundFunction, RpadFunction, RtrimFunction, SecondFunction, Sha1Function, Sha256Function,
    Sha384Function, Sha512Function, SignFunction, SinFunction, SleepFunction, SplitPartFunction,
    SqrtFunction, StartsWithFunction, StrposFunction, SubstrFunction, SubstringFunction,
    TanFunction, TimeTruncFunction, ToCharFunction, TrimFunction, TruncFunction, TruncateFunction,
    TypeOfFunction, UpperFunction, VersionFunction, YearFunction,
};
use super::tvf::{GenerateSeriesFunction, GenerateSeriesScalarFunction, TableValuedFunction};
use super::window::{
    CumeDistFunction, DenseRankFunction, FirstValueFunction, LagFunction, LastValueFunction,
    LeadFunction, NthValueFunction, NtileFunction, PercentRankFunction, RankFunction,
    RowNumberFunction,
};
use super::{AggregateFunction, FunctionInfo, ScalarFunction, Volatility, WindowFunction};

/// Type alias for aggregate function factory
type AggregateFnFactory = Arc<dyn Fn() -> Box<dyn AggregateFunction> + Send + Sync>;
/// Type alias for scalar function factory
type ScalarFnFactory = Arc<dyn Fn() -> Box<dyn ScalarFunction> + Send + Sync>;
/// Type alias for window function factory
type WindowFnFactory = Arc<dyn Fn() -> Box<dyn WindowFunction> + Send + Sync>;
/// Type alias for table-valued function factory
type TvfFactory = Arc<dyn Fn() -> Box<dyn TableValuedFunction> + Send + Sync>;

/// Function registry for SQL functions
pub struct FunctionRegistry {
    /// Aggregate functions
    aggregate_functions: RwLock<StringMap<AggregateFnFactory>>,
    /// Scalar functions
    scalar_functions: RwLock<StringMap<ScalarFnFactory>>,
    /// Window functions
    window_functions: RwLock<StringMap<WindowFnFactory>>,
    /// Table-valued functions
    tvf_functions: RwLock<StringMap<TvfFactory>>,
    /// Function info cache
    function_info: RwLock<StringMap<FunctionInfo>>,
    /// The volatility of each kind registered under a name: one name may
    /// hold a scalar, an aggregate and a window function, each its own
    volatilities: RwLock<StringMap<[Option<Volatility>; 3]>>,
    /// Set once the built-in functions are in, and once a function is
    /// registered after them: a statement calling one may not run twice safely
    built: AtomicBool,
    extended: AtomicBool,
    /// The names the built-in and the engine's own functions hold, of any
    /// kind: no registration takes one, as paths read them by name
    built_in_names: StringSet,
}

impl Default for FunctionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl FunctionRegistry {
    /// Create a new function registry with all built-in functions registered
    pub fn new() -> Self {
        let mut registry = Self {
            aggregate_functions: RwLock::new(StringMap::new()),
            scalar_functions: RwLock::new(StringMap::new()),
            window_functions: RwLock::new(StringMap::new()),
            tvf_functions: RwLock::new(StringMap::new()),
            function_info: RwLock::new(StringMap::new()),
            volatilities: RwLock::new(StringMap::new()),
            built: AtomicBool::new(false),
            extended: AtomicBool::new(false),
            built_in_names: StringSet::default(),
        };

        // Register built-in aggregate functions
        registry.add_aggregate::<CountFunction>();
        registry.add_aggregate::<SumFunction>();
        registry.add_aggregate::<AvgFunction>();
        registry.add_aggregate::<MinFunction>();
        registry.add_aggregate::<MaxFunction>();
        registry.add_aggregate::<FirstFunction>();
        registry.add_aggregate::<LastFunction>();
        registry.add_aggregate::<StringAggFunction>();
        registry.add_aggregate::<GroupConcatFunction>();
        registry.add_aggregate::<ArrayAggFunction>();
        registry.add_aggregate::<StddevPopFunction>();
        registry.add_aggregate::<StddevFunction>();
        registry.add_aggregate::<StddevSampFunction>();
        registry.add_aggregate::<VarPopFunction>();
        registry.add_aggregate::<VarianceFunction>();
        registry.add_aggregate::<VarSampFunction>();
        registry.add_aggregate::<MedianFunction>();

        // Register built-in scalar functions
        // String functions
        registry.add_scalar::<UpperFunction>();
        registry.add_scalar::<LowerFunction>();
        registry.add_scalar::<LengthFunction>();
        registry.add_scalar::<CharLengthFunction>();
        registry.add_scalar::<CharFunction>();
        registry.add_scalar::<ConcatFunction>();
        registry.add_scalar::<ConcatWsFunction>();
        registry.add_scalar::<SubstringFunction>();
        registry.add_scalar::<SubstrFunction>();
        registry.add_scalar::<TrimFunction>();
        registry.add_scalar::<LtrimFunction>();
        registry.add_scalar::<RtrimFunction>();
        registry.add_scalar::<ReplaceFunction>();
        registry.add_scalar::<ReverseFunction>();
        registry.add_scalar::<LeftFunction>();
        registry.add_scalar::<RightFunction>();
        registry.add_scalar::<RepeatFunction>();
        registry.add_scalar::<SplitPartFunction>();
        registry.add_scalar::<StartsWithFunction>();
        registry.add_scalar::<EndsWithFunction>();
        registry.add_scalar::<ContainsFunction>();
        registry.add_scalar::<PositionFunction>();
        registry.add_scalar::<StrposFunction>();
        registry.add_scalar::<InstrFunction>();
        registry.add_scalar::<LocateFunction>();
        registry.add_scalar::<LpadFunction>();
        registry.add_scalar::<RpadFunction>();

        // Math functions
        registry.add_scalar::<AbsFunction>();
        registry.add_scalar::<RoundFunction>();
        registry.add_scalar::<FloorFunction>();
        registry.add_scalar::<CeilingFunction>();
        registry.add_scalar::<CeilFunction>();
        registry.add_scalar::<ModFunction>();
        registry.add_scalar::<PowerFunction>();
        registry.add_scalar::<PowFunction>();
        registry.add_scalar::<SqrtFunction>();
        registry.add_scalar::<LogFunction>();
        registry.add_scalar::<Log10Function>();
        registry.add_scalar::<Log2Function>();
        registry.add_scalar::<LnFunction>();
        registry.add_scalar::<ExpFunction>();
        registry.add_scalar::<SignFunction>();
        registry.add_scalar::<TruncateFunction>();
        registry.add_scalar::<TruncFunction>();
        registry.add_scalar::<PiFunction>();
        registry.add_scalar::<RandomFunction>();
        registry.add_scalar::<SinFunction>();
        registry.add_scalar::<CosFunction>();
        registry.add_scalar::<TanFunction>();

        // Date/Time functions
        registry.add_scalar::<NowFunction>();
        registry.add_scalar::<CurrentDateFunction>();
        registry.add_scalar::<CurrentTimeFunction>();
        registry.add_scalar::<CurrentTimestampFunction>();
        registry.add_scalar::<DateTruncFunction>();
        registry.add_scalar::<TimeTruncFunction>();
        registry.add_scalar::<ExtractFunction>();
        registry.add_scalar::<YearFunction>();
        registry.add_scalar::<MonthFunction>();
        registry.add_scalar::<DayFunction>();
        registry.add_scalar::<HourFunction>();
        registry.add_scalar::<MinuteFunction>();
        registry.add_scalar::<SecondFunction>();
        registry.add_scalar::<DateAddFunction>();
        registry.add_scalar::<DateSubFunction>();
        registry.add_scalar::<DateDiffFunction>();
        registry.add_scalar::<DateDiffAliasFunction>(); // DATE_DIFF alias for DATEDIFF
        registry.add_scalar::<ToCharFunction>();

        // Utility/System functions
        registry.add_scalar::<VersionFunction>();
        registry.add_scalar::<CoalesceFunction>();
        registry.add_scalar::<NullIfFunction>();
        registry.add_scalar::<IfNullFunction>();
        registry.add_scalar::<CastFunction>();
        registry.add_scalar::<CollateFunction>();
        registry.add_scalar::<GreatestFunction>();
        registry.add_scalar::<LeastFunction>();
        registry.add_scalar::<IifFunction>();
        registry.add_scalar::<JsonExtractFunction>();
        registry.add_scalar::<JsonArrayLengthFunction>();
        registry.add_scalar::<JsonArrayFunction>();
        registry.add_scalar::<JsonObjectFunction>();
        registry.add_scalar::<JsonTypeFunction>();
        registry.add_scalar::<JsonTypeOfFunction>();
        registry.add_scalar::<JsonValidFunction>();
        registry.add_scalar::<JsonKeysFunction>();
        registry.add_scalar::<TypeOfFunction>();
        registry.add_scalar::<SleepFunction>();

        // Hash functions
        registry.add_scalar::<Md5Function>();
        registry.add_scalar::<Sha1Function>();
        registry.add_scalar::<Sha256Function>();
        registry.add_scalar::<Sha384Function>();
        registry.add_scalar::<Sha512Function>();
        registry.add_scalar::<Crc32Function>();

        // Register vector functions
        registry.add_scalar::<VecDistanceL2Function>();
        registry.add_scalar::<VecDistanceCosineFunction>();
        registry.add_scalar::<VecDistanceIpFunction>();
        registry.add_scalar::<VecDimsFunction>();
        registry.add_scalar::<VecNormFunction>();
        registry.add_scalar::<VecToTextFunction>();

        // Register semantic embedding function (requires --features semantic)
        #[cfg(feature = "semantic")]
        registry.add_scalar::<EmbedFunction>();

        // Register generate_series as scalar (returns JSON array for SELECT usage)
        registry.add_scalar::<GenerateSeriesScalarFunction>();

        // Register built-in window functions
        registry.add_window::<RowNumberFunction>();
        registry.add_window::<RankFunction>();
        registry.add_window::<DenseRankFunction>();
        registry.add_window::<NtileFunction>();
        registry.add_window::<LeadFunction>();
        registry.add_window::<LagFunction>();
        registry.add_window::<FirstValueFunction>();
        registry.add_window::<LastValueFunction>();
        registry.add_window::<NthValueFunction>();
        registry.add_window::<PercentRankFunction>();
        registry.add_window::<CumeDistFunction>();

        // Register built-in table-valued functions
        registry.add_tvf(
            "GENERATE_SERIES",
            Arc::new(|| Box::new(GenerateSeriesFunction)),
        );

        registry.built_in_names = registry.list_all().into_iter().collect();
        registry
            .built_in_names
            .extend(registry.tvf_functions.read().unwrap().keys().cloned());
        registry
            .built_in_names
            .extend(ENGINE_FUNCTIONS.map(String::from));
        registry.built.store(true, Ordering::Release);
        registry
    }

    /// Whether a function was registered after the built-in ones
    pub fn has_registered_functions(&self) -> bool {
        self.extended.load(Ordering::Acquire)
    }

    fn note_registration(&self) {
        if self.built.load(Ordering::Acquire) {
            self.extended.store(true, Ordering::Release);
        }
    }

    /// An error when `name` is a built-in function's, of any kind
    fn refuse_built_in(&self, name: &str) -> Result<()> {
        match self.built_in_names.contains(&name.to_uppercase()) {
            true => Err(Error::invalid_argument(format!(
                "{} is a built-in function name and cannot be registered",
                name.to_uppercase()
            ))),
            false => Ok(()),
        }
    }

    /// Register an aggregate function under a name no built-in holds
    pub fn register_aggregate<F: AggregateFunction + Default + 'static>(&self) -> Result<()> {
        self.refuse_built_in(F::default().name())?;
        self.add_aggregate::<F>();
        Ok(())
    }

    #[inline]
    fn add_aggregate<F: AggregateFunction + Default + 'static>(&self) {
        let instance = F::default();
        self.register_aggregate_inner(
            instance.name().to_uppercase(),
            instance.info(),
            Arc::new(|| Box::new(F::default())),
        );
    }

    fn register_aggregate_inner(
        &self,
        name: String,
        info: FunctionInfo,
        factory: AggregateFnFactory,
    ) {
        self.note_registration();
        self.aggregate_functions
            .write()
            .unwrap()
            .insert(name.clone(), factory);
        self.note_volatility(&name, AGGREGATE, info.volatility);
        self.function_info.write().unwrap().insert(name, info);
    }

    fn note_volatility(&self, name: &str, kind: usize, volatility: Volatility) {
        let mut volatilities = self.volatilities.write().unwrap();
        match volatilities.get_mut(name) {
            Some(kinds) => kinds[kind] = Some(volatility),
            None => {
                let mut kinds = [None; 3];
                kinds[kind] = Some(volatility);
                volatilities.insert(name.to_string(), kinds);
            }
        }
    }

    /// Register a scalar function under a name no built-in holds
    pub fn register_scalar<F: ScalarFunction + Default + 'static>(&self) -> Result<()> {
        self.refuse_built_in(F::default().name())?;
        self.add_scalar::<F>();
        Ok(())
    }

    #[inline]
    fn add_scalar<F: ScalarFunction + Default + 'static>(&self) {
        let instance = F::default();
        self.register_scalar_inner(
            instance.name().to_uppercase(),
            instance.info(),
            Arc::new(|| Box::new(F::default())),
        );
    }

    fn register_scalar_inner(&self, name: String, info: FunctionInfo, factory: ScalarFnFactory) {
        self.note_registration();
        self.scalar_functions
            .write()
            .unwrap()
            .insert(name.clone(), factory);
        self.note_volatility(&name, SCALAR, info.volatility);
        self.function_info.write().unwrap().insert(name, info);
    }

    /// Register a window function under a name no built-in holds
    pub fn register_window<F: WindowFunction + Default + 'static>(&self) -> Result<()> {
        self.refuse_built_in(F::default().name())?;
        self.add_window::<F>();
        Ok(())
    }

    #[inline]
    fn add_window<F: WindowFunction + Default + 'static>(&self) {
        let instance = F::default();
        self.register_window_inner(
            instance.name().to_uppercase(),
            instance.info(),
            Arc::new(|| Box::new(F::default())),
        );
    }

    fn register_window_inner(&self, name: String, info: FunctionInfo, factory: WindowFnFactory) {
        self.note_registration();
        self.window_functions
            .write()
            .unwrap()
            .insert(name.clone(), factory);
        self.note_volatility(&name, WINDOW, info.volatility);
        self.function_info.write().unwrap().insert(name, info);
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

    /// Register a table-valued function under a name no built-in holds
    pub fn register_tvf(&self, name: &str, factory: TvfFactory) -> Result<()> {
        self.refuse_built_in(name)?;
        self.add_tvf(name, factory);
        Ok(())
    }

    fn add_tvf(&self, name: &str, factory: TvfFactory) {
        self.note_registration();
        let mut funcs = self.tvf_functions.write().unwrap();
        funcs.insert(name.to_string(), factory);
    }

    /// Get a new instance of a table-valued function by name
    pub fn get_tvf(&self, name: &str) -> Option<Box<dyn TableValuedFunction>> {
        let funcs = self.tvf_functions.read().unwrap();
        if let Some(f) = funcs.get(name) {
            return Some(f());
        }
        let upper = name.to_uppercase();
        funcs.get(&upper).map(|f| f())
    }

    /// Check if a function name is a table-valued function
    pub fn is_tvf(&self, name: &str) -> bool {
        let funcs = self.tvf_functions.read().unwrap();
        if funcs.contains_key(name) {
            return true;
        }
        let upper = name.to_uppercase();
        funcs.contains_key(&upper)
    }

    /// Check if a function exists
    pub fn exists(&self, name: &str) -> bool {
        self.is_aggregate(name) || self.is_scalar(name) || self.is_window(name) || self.is_tvf(name)
    }

    /// Get function info by name
    pub fn get_info(&self, name: &str) -> Option<FunctionInfo> {
        let name = name.to_uppercase();
        let infos = self.function_info.read().unwrap();
        infos.get(&name).cloned()
    }

    /// How often a call of `name` may answer differently: the most volatile
    /// of the functions the name can run, whichever kind a path resolves it
    /// to. The transaction id the compiler reads itself is one per statement;
    /// a name this registry does not hold answers per call
    pub fn volatility(&self, name: &str) -> Volatility {
        if name.eq_ignore_ascii_case("CURRENT_TRANSACTION_ID") {
            return Volatility::PerStatement;
        }
        let volatilities = self.volatilities.read().unwrap();
        let kinds = match volatilities.get(name) {
            Some(kinds) => Some(kinds),
            None => volatilities.get(&name.to_uppercase()),
        };
        kinds
            .and_then(|kinds| kinds.iter().flatten().max().copied())
            .unwrap_or(Volatility::PerCall)
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
