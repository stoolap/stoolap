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

//! Value Interning for Memory Efficiency
//!
//! This module provides interned `CompactArc<Value>` instances for common values,
//! reducing memory allocation overhead for frequently used values.
//!
//! ## Interned Values
//!
//! - **NULL**: All DataType variants (7 types)
//! - **Booleans**: `true` and `false`
//! - **Small Integers**: 0 to 1000
//!
//! ## Memory Savings
//!
//! In real-world databases, these values are extremely common:
//! - NULL values in nullable columns (10-30% of cells)
//! - Boolean flags (is_active, is_deleted, etc.)
//! - Small integers (status codes, counts, foreign keys)
//!
//! Without interning: Each value allocates 32 bytes on heap
//! With interning: One allocation shared by all references

use std::sync::OnceLock;

use crate::common::CompactArc;
use crate::core::types::DataType;
use crate::core::value::Value;

/// Maximum integer value to intern (0..=MAX_INTERNED_INT)
const MAX_INTERNED_INT: i64 = 1000;

/// Interned NULL values for each DataType
struct InternedNulls {
    null: CompactArc<Value>,
    integer: CompactArc<Value>,
    float: CompactArc<Value>,
    text: CompactArc<Value>,
    boolean: CompactArc<Value>,
    timestamp: CompactArc<Value>,
    json: CompactArc<Value>,
}

impl InternedNulls {
    fn new() -> Self {
        Self {
            null: CompactArc::new(Value::Null(DataType::Null)),
            integer: CompactArc::new(Value::Null(DataType::Integer)),
            float: CompactArc::new(Value::Null(DataType::Float)),
            text: CompactArc::new(Value::Null(DataType::Text)),
            boolean: CompactArc::new(Value::Null(DataType::Boolean)),
            timestamp: CompactArc::new(Value::Null(DataType::Timestamp)),
            json: CompactArc::new(Value::Null(DataType::Json)),
        }
    }

    #[inline]
    fn get(&self, dt: DataType) -> CompactArc<Value> {
        match dt {
            DataType::Null => self.null.clone(),
            DataType::Integer => self.integer.clone(),
            DataType::Float => self.float.clone(),
            DataType::Text => self.text.clone(),
            DataType::Boolean => self.boolean.clone(),
            DataType::Timestamp => self.timestamp.clone(),
            DataType::Json => self.json.clone(),
        }
    }
}

/// Interned boolean values
struct InternedBooleans {
    true_val: CompactArc<Value>,
    false_val: CompactArc<Value>,
}

impl InternedBooleans {
    fn new() -> Self {
        Self {
            true_val: CompactArc::new(Value::Boolean(true)),
            false_val: CompactArc::new(Value::Boolean(false)),
        }
    }

    #[inline]
    fn get(&self, b: bool) -> CompactArc<Value> {
        if b {
            self.true_val.clone()
        } else {
            self.false_val.clone()
        }
    }
}

/// Interned small integers (0..=1000)
struct InternedIntegers {
    values: Box<[CompactArc<Value>]>,
}

impl InternedIntegers {
    fn new() -> Self {
        let values: Vec<CompactArc<Value>> = (0..=MAX_INTERNED_INT)
            .map(|i| CompactArc::new(Value::Integer(i)))
            .collect();
        Self {
            values: values.into_boxed_slice(),
        }
    }

    #[inline]
    fn get(&self, n: i64) -> Option<CompactArc<Value>> {
        if (0..=MAX_INTERNED_INT).contains(&n) {
            Some(self.values[n as usize].clone())
        } else {
            None
        }
    }
}

// Global interned value stores
static INTERNED_NULLS: OnceLock<InternedNulls> = OnceLock::new();
static INTERNED_BOOLEANS: OnceLock<InternedBooleans> = OnceLock::new();
static INTERNED_INTEGERS: OnceLock<InternedIntegers> = OnceLock::new();

/// Get interned NULL values
#[inline]
fn nulls() -> &'static InternedNulls {
    INTERNED_NULLS.get_or_init(InternedNulls::new)
}

/// Get interned boolean values
#[inline]
fn booleans() -> &'static InternedBooleans {
    INTERNED_BOOLEANS.get_or_init(InternedBooleans::new)
}

/// Get interned integer values
#[inline]
fn integers() -> &'static InternedIntegers {
    INTERNED_INTEGERS.get_or_init(InternedIntegers::new)
}

/// Intern a Value, returning a CompactArc<Value>.
///
/// For common values (NULL, booleans, integers 0-1000), returns a shared
/// interned instance. For other values, creates a new CompactArc.
#[inline]
pub fn intern_value(value: Value) -> CompactArc<Value> {
    match &value {
        Value::Null(dt) => nulls().get(*dt),
        Value::Boolean(b) => booleans().get(*b),
        Value::Integer(n) => {
            if let Some(arc) = integers().get(*n) {
                arc
            } else {
                CompactArc::new(value)
            }
        }
        // Non-interned values: create new allocation
        Value::Float(_) | Value::Text(_) | Value::Timestamp(_) | Value::Json(_) => {
            CompactArc::new(value)
        }
    }
}

/// Get an interned NULL value for the given DataType.
#[inline]
pub fn interned_null(dt: DataType) -> CompactArc<Value> {
    nulls().get(dt)
}

/// Get an interned boolean value.
#[inline]
pub fn interned_bool(b: bool) -> CompactArc<Value> {
    booleans().get(b)
}

/// Get an interned integer value if in range 0..=1000.
#[inline]
pub fn interned_int(n: i64) -> Option<CompactArc<Value>> {
    integers().get(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_null_interning() {
        let null1 = intern_value(Value::Null(DataType::Integer));
        let null2 = intern_value(Value::Null(DataType::Integer));
        assert!(CompactArc::ptr_eq(&null1, &null2));
    }

    #[test]
    fn test_boolean_interning() {
        let true1 = intern_value(Value::Boolean(true));
        let true2 = intern_value(Value::Boolean(true));
        let false1 = intern_value(Value::Boolean(false));

        assert!(CompactArc::ptr_eq(&true1, &true2));
        assert!(!CompactArc::ptr_eq(&true1, &false1));
    }

    #[test]
    fn test_integer_interning() {
        let zero1 = intern_value(Value::Integer(0));
        let zero2 = intern_value(Value::Integer(0));
        assert!(CompactArc::ptr_eq(&zero1, &zero2));

        let thousand1 = intern_value(Value::Integer(1000));
        let thousand2 = intern_value(Value::Integer(1000));
        assert!(CompactArc::ptr_eq(&thousand1, &thousand2));

        // Large integers should NOT be interned
        let big1 = intern_value(Value::Integer(1001));
        let big2 = intern_value(Value::Integer(1001));
        assert!(!CompactArc::ptr_eq(&big1, &big2));
    }

    #[test]
    fn test_non_interned_values() {
        let float1 = intern_value(Value::Float(1.0));
        let float2 = intern_value(Value::Float(1.0));
        assert!(!CompactArc::ptr_eq(&float1, &float2));
    }
}
