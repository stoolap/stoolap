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

//! Parameter binding for SQL queries
//!
//! This module provides ergonomic parameter passing for SQL queries,
//! similar to rusqlite's `params!` macro.
//!
//! # Examples
//!
//! ```ignore
//! use stoolap::{Database, params, named_params};
//!
//! let db = Database::open("memory://")?;
//!
//! // Using params! macro (positional)
//! db.execute("INSERT INTO users VALUES ($1, $2, $3)", params![1, "Alice", 30])?;
//!
//! // Using tuple syntax (positional)
//! db.execute("INSERT INTO users VALUES ($1, $2)", (1, "Alice"))?;
//!
//! // Using named_params! macro
//! db.execute_named(
//!     "INSERT INTO users VALUES (:id, :name, :age)",
//!     named_params!{ id: 1, name: "Alice", age: 30 }
//! )?;
//! ```

use rustc_hash::FxHashMap;
use std::collections::HashMap;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use smallvec::SmallVec;

use crate::common::SmartString;
use crate::core::Value;

/// Type alias for parameter vectors.
/// Uses SmallVec to avoid heap allocation for queries with ≤8 parameters (the common case).
pub type ParamVec = SmallVec<[Value; 8]>;

/// Trait for types that can be converted to SQL parameters
///
/// This trait is automatically implemented for common Rust types.
/// It enables the `params!` macro and tuple parameter syntax.
pub trait ToParam {
    /// Convert self into a Value for SQL parameter binding
    fn to_param(&self) -> Value;
}

// Implement ToParam for common types

impl ToParam for i64 {
    fn to_param(&self) -> Value {
        Value::Integer(*self)
    }
}

impl ToParam for i32 {
    fn to_param(&self) -> Value {
        Value::Integer(*self as i64)
    }
}

impl ToParam for i16 {
    fn to_param(&self) -> Value {
        Value::Integer(*self as i64)
    }
}

impl ToParam for i8 {
    fn to_param(&self) -> Value {
        Value::Integer(*self as i64)
    }
}

impl ToParam for u32 {
    fn to_param(&self) -> Value {
        Value::Integer(*self as i64)
    }
}

impl ToParam for u16 {
    fn to_param(&self) -> Value {
        Value::Integer(*self as i64)
    }
}

impl ToParam for u8 {
    fn to_param(&self) -> Value {
        Value::Integer(*self as i64)
    }
}

impl ToParam for usize {
    fn to_param(&self) -> Value {
        Value::Integer(*self as i64)
    }
}

impl ToParam for f64 {
    fn to_param(&self) -> Value {
        Value::Float(*self)
    }
}

impl ToParam for f32 {
    fn to_param(&self) -> Value {
        Value::Float(*self as f64)
    }
}

impl ToParam for bool {
    fn to_param(&self) -> Value {
        Value::Boolean(*self)
    }
}

impl ToParam for String {
    fn to_param(&self) -> Value {
        Value::Text(SmartString::new(self))
    }
}

impl ToParam for &str {
    fn to_param(&self) -> Value {
        Value::Text(SmartString::from(*self))
    }
}

impl ToParam for Arc<str> {
    fn to_param(&self) -> Value {
        Value::Text(SmartString::from(self.as_ref()))
    }
}

impl ToParam for DateTime<Utc> {
    fn to_param(&self) -> Value {
        Value::Timestamp(*self)
    }
}

impl ToParam for Value {
    fn to_param(&self) -> Value {
        self.clone()
    }
}

impl<T: ToParam> ToParam for Option<T> {
    fn to_param(&self) -> Value {
        match self {
            Some(v) => v.to_param(),
            None => Value::null_unknown(),
        }
    }
}

impl<T: ToParam> ToParam for &T {
    fn to_param(&self) -> Value {
        (*self).to_param()
    }
}

/// Trait for collections of parameters
///
/// This enables passing tuples, arrays, and slices as parameters.
pub trait Params {
    /// Convert into a ParamVec of Values.
    /// Uses SmallVec to avoid heap allocation for ≤4 parameters.
    fn into_params(self) -> ParamVec;
}

// Empty params
impl Params for () {
    fn into_params(self) -> ParamVec {
        ParamVec::new()
    }
}

// Slice of Value
impl Params for &[Value] {
    fn into_params(self) -> ParamVec {
        self.iter().cloned().collect()
    }
}

// Vec of Value
impl Params for Vec<Value> {
    fn into_params(self) -> ParamVec {
        self.into_iter().collect()
    }
}

// ParamVec (SmallVec) — zero-cost passthrough, no conversion needed
impl Params for ParamVec {
    fn into_params(self) -> ParamVec {
        self
    }
}

// Array of Value
impl<const N: usize> Params for [Value; N] {
    fn into_params(self) -> ParamVec {
        self.into_iter().collect()
    }
}

// Tuple implementations for 1-12 elements
macro_rules! impl_params_for_tuple {
    ($($idx:tt: $T:ident),+) => {
        impl<$($T: ToParam),+> Params for ($($T,)+) {
            fn into_params(self) -> ParamVec {
                smallvec::smallvec![$(self.$idx.to_param()),+]
            }
        }
    };
}

impl_params_for_tuple!(0: T0);
impl_params_for_tuple!(0: T0, 1: T1);
impl_params_for_tuple!(0: T0, 1: T1, 2: T2);
impl_params_for_tuple!(0: T0, 1: T1, 2: T2, 3: T3);
impl_params_for_tuple!(0: T0, 1: T1, 2: T2, 3: T3, 4: T4);
impl_params_for_tuple!(0: T0, 1: T1, 2: T2, 3: T3, 4: T4, 5: T5);
impl_params_for_tuple!(0: T0, 1: T1, 2: T2, 3: T3, 4: T4, 5: T5, 6: T6);
impl_params_for_tuple!(0: T0, 1: T1, 2: T2, 3: T3, 4: T4, 5: T5, 6: T6, 7: T7);
impl_params_for_tuple!(0: T0, 1: T1, 2: T2, 3: T3, 4: T4, 5: T5, 6: T6, 7: T7, 8: T8);
impl_params_for_tuple!(0: T0, 1: T1, 2: T2, 3: T3, 4: T4, 5: T5, 6: T6, 7: T7, 8: T8, 9: T9);
impl_params_for_tuple!(0: T0, 1: T1, 2: T2, 3: T3, 4: T4, 5: T5, 6: T6, 7: T7, 8: T8, 9: T9, 10: T10);
impl_params_for_tuple!(0: T0, 1: T1, 2: T2, 3: T3, 4: T4, 5: T5, 6: T6, 7: T7, 8: T8, 9: T9, 10: T10, 11: T11);

/// Create a parameter list for SQL queries
///
/// This macro provides a convenient way to create parameter lists without
/// manually wrapping each value in `Value::from()`.
///
/// # Examples
///
/// ```ignore
/// use stoolap::{Database, params};
///
/// let db = Database::open("memory://")?;
/// db.execute("CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)", ())?;
///
/// // Insert with params
/// db.execute(
///     "INSERT INTO users VALUES ($1, $2, $3)",
///     params![1, "Alice", 30]
/// )?;
///
/// // Query with params
/// let rows = db.query(
///     "SELECT * FROM users WHERE age > $1",
///     params![25]
/// )?;
///
/// // Mixed types work seamlessly
/// db.execute(
///     "INSERT INTO users VALUES ($1, $2, $3)",
///     params![2, String::from("Bob"), 25]
/// )?;
/// ```
#[macro_export]
macro_rules! params {
    () => {
        ()
    };
    ($($param:expr),+ $(,)?) => {
        ($($param,)+)
    };
}

/// Named parameters for SQL queries
///
/// This struct holds named parameter bindings that can be used with
/// the `:name` syntax in SQL queries.
///
/// # Examples
///
/// ```ignore
/// use stoolap::{Database, NamedParams, named_params};
///
/// let db = Database::open("memory://")?;
/// db.execute("CREATE TABLE users (id INTEGER, name TEXT)", ())?;
///
/// // Using the named_params! macro
/// db.execute_named(
///     "INSERT INTO users VALUES (:id, :name)",
///     named_params!{ id: 1, name: "Alice" }
/// )?;
///
/// // Building NamedParams manually
/// let params = NamedParams::new()
///     .add("id", 2)
///     .add("name", "Bob");
/// db.execute_named("INSERT INTO users VALUES (:id, :name)", params)?;
/// ```
#[derive(Debug, Clone, Default)]
pub struct NamedParams {
    params: FxHashMap<String, Value>,
}

impl NamedParams {
    /// Create empty named params
    pub fn new() -> Self {
        Self {
            params: FxHashMap::default(),
        }
    }

    /// Add a named parameter (builder style)
    pub fn add<T: ToParam>(mut self, name: impl Into<String>, value: T) -> Self {
        self.params.insert(name.into(), value.to_param());
        self
    }

    /// Insert a named parameter
    pub fn insert<T: ToParam>(&mut self, name: impl Into<String>, value: T) {
        self.params.insert(name.into(), value.to_param());
    }

    /// Get the underlying FxHashMap
    pub fn into_inner(self) -> FxHashMap<String, Value> {
        self.params
    }

    /// Get a reference to the underlying FxHashMap
    pub fn as_map(&self) -> &FxHashMap<String, Value> {
        &self.params
    }
}

impl From<HashMap<String, Value>> for NamedParams {
    fn from(params: HashMap<String, Value>) -> Self {
        Self {
            params: params.into_iter().collect(),
        }
    }
}

/// Create named parameters for SQL queries
///
/// This macro provides a convenient way to create named parameter bindings
/// for use with the `:name` syntax in SQL queries.
///
/// # Examples
///
/// ```ignore
/// use stoolap::{Database, named_params};
///
/// let db = Database::open("memory://")?;
/// db.execute("CREATE TABLE users (id INTEGER, name TEXT, active BOOLEAN)", ())?;
///
/// // Insert with named params
/// db.execute_named(
///     "INSERT INTO users VALUES (:id, :name, :active)",
///     named_params!{ id: 1, name: "Alice", active: true }
/// )?;
///
/// // Query with named params
/// let rows = db.query_named(
///     "SELECT * FROM users WHERE name = :name",
///     named_params!{ name: "Alice" }
/// )?;
/// ```
#[macro_export]
macro_rules! named_params {
    () => {
        $crate::NamedParams::new()
    };
    ($($name:ident : $value:expr),+ $(,)?) => {
        {
            let mut params = $crate::NamedParams::new();
            $(
                params.insert(stringify!($name), $value);
            )+
            params
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_param_integers() {
        assert_eq!(42i64.to_param(), Value::Integer(42));
        assert_eq!(42i32.to_param(), Value::Integer(42));
        assert_eq!(42i16.to_param(), Value::Integer(42));
        assert_eq!(42i8.to_param(), Value::Integer(42));
        assert_eq!(42u32.to_param(), Value::Integer(42));
        assert_eq!(42u16.to_param(), Value::Integer(42));
        assert_eq!(42u8.to_param(), Value::Integer(42));
    }

    #[test]
    fn test_to_param_floats() {
        assert_eq!(3.5f64.to_param(), Value::Float(3.5));
        assert_eq!(3.5f32.to_param(), Value::Float(3.5f32 as f64));
    }

    #[test]
    fn test_to_param_strings() {
        assert_eq!("hello".to_param(), Value::text("hello"));
        assert_eq!(String::from("world").to_param(), Value::text("world"));
    }

    #[test]
    fn test_to_param_bool() {
        assert_eq!(true.to_param(), Value::Boolean(true));
        assert_eq!(false.to_param(), Value::Boolean(false));
    }

    #[test]
    fn test_to_param_option() {
        assert_eq!(Some(42i64).to_param(), Value::Integer(42));
        assert!(Option::<i64>::None.to_param().is_null());
    }

    #[test]
    fn test_params_empty() {
        let params: ParamVec = ().into_params();
        assert!(params.is_empty());
    }

    #[test]
    fn test_params_tuple() {
        let params = (1i64, "hello", 3.5f64).into_params();
        assert_eq!(params.len(), 3);
        assert_eq!(params[0], Value::Integer(1));
        assert_eq!(params[1], Value::text("hello"));
        assert_eq!(params[2], Value::Float(3.5));
    }

    #[test]
    fn test_params_macro() {
        let p = params![1, "hello", 3.5];
        let params = p.into_params();
        assert_eq!(params.len(), 3);
        assert_eq!(params[0], Value::Integer(1));
        assert_eq!(params[1], Value::text("hello"));
        assert_eq!(params[2], Value::Float(3.5));
    }

    #[test]
    fn test_params_macro_empty() {
        let p = params![];
        let params: ParamVec = p.into_params();
        assert!(params.is_empty());
    }

    #[test]
    fn test_params_with_option() {
        let name: Option<&str> = Some("Alice");
        let age: Option<i32> = None;
        let params = (1i64, name, age).into_params();

        assert_eq!(params.len(), 3);
        assert_eq!(params[0], Value::Integer(1));
        assert_eq!(params[1], Value::text("Alice"));
        assert!(params[2].is_null());
    }

    #[test]
    fn test_params_from_param_vec() {
        // ParamVec -> ParamVec should be zero-cost identity
        let mut pv = ParamVec::new();
        pv.push(Value::Integer(1));
        pv.push(Value::text("hello"));
        pv.push(Value::Float(3.5));

        let result = pv.into_params();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], Value::Integer(1));
        assert_eq!(result[1], Value::text("hello"));
        assert_eq!(result[2], Value::Float(3.5));
    }

    #[test]
    fn test_params_from_empty_param_vec() {
        let pv = ParamVec::new();
        let result = pv.into_params();
        assert!(result.is_empty());
    }
}
