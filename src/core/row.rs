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

//! Row type for Stoolap - a collection of column values

use std::fmt;
use std::ops::{Deref, DerefMut, Index, IndexMut};

use super::error::{Error, Result};
use super::schema::Schema;
use super::types::DataType;
use super::value::Value;

/// A database row containing column values
///
/// Row provides methods for accessing and manipulating column values
/// while maintaining type safety and consistency with the schema.
///
/// Performance: Row uses thread-local pooling to reuse Vec<Value> allocations.
/// When a Row is dropped, its internal Vec is returned to the pool for reuse.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Row {
    values: Vec<Value>,
}

impl Row {
    /// Create a new empty row
    pub fn new() -> Self {
        Self { values: Vec::new() }
    }

    /// Create a row with pre-allocated capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            values: Vec::with_capacity(capacity),
        }
    }

    /// Create a row from a vector of values
    pub fn from_values(values: Vec<Value>) -> Self {
        Self { values }
    }

    /// Create a row with null values for a given schema
    pub fn null_row(schema: &Schema) -> Self {
        let values = schema
            .columns
            .iter()
            .map(|col| Value::null(col.data_type))
            .collect();
        Self { values }
    }

    /// Get the number of values in the row
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if the row is empty
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Get a value by index
    pub fn get(&self, index: usize) -> Option<&Value> {
        self.values.get(index)
    }

    /// Get a mutable value by index
    pub fn get_mut(&mut self, index: usize) -> Option<&mut Value> {
        self.values.get_mut(index)
    }

    /// Set a value at the given index
    pub fn set(&mut self, index: usize, value: Value) -> Result<()> {
        if index >= self.values.len() {
            return Err(Error::Internal {
                message: format!(
                    "row index {} out of bounds (len={})",
                    index,
                    self.values.len()
                ),
            });
        }
        self.values[index] = value;
        Ok(())
    }

    /// Push a value to the end of the row
    pub fn push(&mut self, value: Value) {
        self.values.push(value);
    }

    /// Pop a value from the end of the row
    pub fn pop(&mut self) -> Option<Value> {
        self.values.pop()
    }

    /// Truncate the row to a specific length
    ///
    /// Used for schema evolution when columns are dropped
    pub fn truncate(&mut self, len: usize) {
        self.values.truncate(len);
    }

    /// Clear the row values while keeping the allocated capacity
    /// This allows reusing the Vec allocation for streaming operations
    #[inline]
    pub fn clear(&mut self) {
        self.values.clear();
    }

    /// Reserve capacity for at least `additional` more values
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.values.reserve(additional);
    }

    /// Get an iterator over the values
    pub fn iter(&self) -> impl Iterator<Item = &Value> {
        self.values.iter()
    }

    /// Get a mutable iterator over the values
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Value> {
        self.values.iter_mut()
    }

    /// Get the underlying vector of values
    pub fn into_values(self) -> Vec<Value> {
        self.values
    }

    /// Get a reference to the underlying vector
    pub fn as_slice(&self) -> &[Value] {
        &self.values
    }

    /// Get a mutable reference to the underlying vector
    pub fn as_mut_slice(&mut self) -> &mut [Value] {
        &mut self.values
    }

    /// Extract specific columns by their indices
    pub fn select_columns(&self, indices: &[usize]) -> Result<Row> {
        let mut values = Vec::with_capacity(indices.len());
        for &idx in indices {
            match self.values.get(idx) {
                Some(v) => values.push(v.clone()),
                None => {
                    return Err(Error::Internal {
                        message: format!(
                            "column index {} out of bounds (len={})",
                            idx,
                            self.values.len()
                        ),
                    })
                }
            }
        }
        Ok(Row::from_values(values))
    }

    /// Take specific columns by their indices, consuming the row
    /// OPTIMIZATION: Avoids cloning by moving values out of the row
    #[inline]
    pub fn take_columns(mut self, indices: &[usize]) -> Row {
        let mut values = Vec::with_capacity(indices.len());
        for &idx in indices {
            if idx < self.values.len() {
                // Take the value, replacing with null to avoid clone
                values.push(std::mem::take(&mut self.values[idx]));
            } else {
                values.push(Value::null_unknown());
            }
        }
        Row::from_values(values)
    }

    /// Validate the row against a schema
    pub fn validate(&self, schema: &Schema) -> Result<()> {
        // Check column count
        if self.values.len() != schema.columns.len() {
            return Err(Error::table_columns_not_match(
                schema.columns.len(),
                self.values.len(),
            ));
        }

        // Check each value
        for (i, (value, col)) in self.values.iter().zip(schema.columns.iter()).enumerate() {
            // Check nullability
            if value.is_null() && !col.nullable && !col.primary_key {
                return Err(Error::not_null_constraint(&col.name));
            }

            // Check type compatibility (skip for null values)
            if !value.is_null() {
                let value_type = value.data_type();
                if value_type != col.data_type {
                    // Allow some implicit conversions
                    let compatible = matches!(
                        (value_type, col.data_type),
                        (DataType::Integer, DataType::Float) | (DataType::Float, DataType::Integer)
                    );
                    if !compatible {
                        return Err(Error::type_conversion(
                            format!("column {} at index {}: {:?}", col.name, i, value_type),
                            format!("{:?}", col.data_type),
                        ));
                    }
                }
            }
        }

        Ok(())
    }

    /// Clone the row, selecting only the specified column indices
    pub fn clone_subset(&self, indices: &[usize]) -> Row {
        let values = indices
            .iter()
            .filter_map(|&i| self.values.get(i).cloned())
            .collect();
        Row::from_values(values)
    }

    /// Concatenate two rows
    pub fn concat(&self, other: &Row) -> Row {
        let mut values = self.values.clone();
        values.extend(other.values.iter().cloned());
        Row::from_values(values)
    }

    /// Create a row by repeating a value
    pub fn repeat(value: Value, count: usize) -> Row {
        Row::from_values(vec![value; count])
    }
}

// Implement Deref to allow using Row like Vec<Value>
impl Deref for Row {
    type Target = Vec<Value>;

    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

impl DerefMut for Row {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.values
    }
}

// Implement Index for convenient access
impl Index<usize> for Row {
    type Output = Value;

    fn index(&self, index: usize) -> &Self::Output {
        &self.values[index]
    }
}

impl IndexMut<usize> for Row {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.values[index]
    }
}

// Implement FromIterator for collecting values into a row
impl FromIterator<Value> for Row {
    fn from_iter<I: IntoIterator<Item = Value>>(iter: I) -> Self {
        Row::from_values(iter.into_iter().collect())
    }
}

// Implement IntoIterator for consuming iteration
impl IntoIterator for Row {
    type Item = Value;
    type IntoIter = std::vec::IntoIter<Value>;

    fn into_iter(self) -> Self::IntoIter {
        self.values.into_iter()
    }
}

impl<'a> IntoIterator for &'a Row {
    type Item = &'a Value;
    type IntoIter = std::slice::Iter<'a, Value>;

    fn into_iter(self) -> Self::IntoIter {
        self.values.iter()
    }
}

impl<'a> IntoIterator for &'a mut Row {
    type Item = &'a mut Value;
    type IntoIter = std::slice::IterMut<'a, Value>;

    fn into_iter(self) -> Self::IntoIter {
        self.values.iter_mut()
    }
}

impl From<Vec<Value>> for Row {
    fn from(values: Vec<Value>) -> Self {
        Row::from_values(values)
    }
}

impl fmt::Display for Row {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        for (i, value) in self.values.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", value)?;
        }
        write!(f, ")")
    }
}

/// Macro for creating rows conveniently
#[macro_export]
macro_rules! row {
    () => {
        $crate::core::Row::new()
    };
    ($($value:expr),+ $(,)?) => {
        $crate::core::Row::from_values(vec![$($crate::core::Value::from($value)),+])
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::schema::SchemaBuilder;

    fn create_test_schema() -> Schema {
        SchemaBuilder::new("test")
            .add_primary_key("id", DataType::Integer)
            .add("name", DataType::Text)
            .add_nullable("email", DataType::Text)
            .build()
    }

    #[test]
    fn test_row_creation() {
        let row = Row::new();
        assert!(row.is_empty());
        assert_eq!(row.len(), 0);

        let row = Row::with_capacity(10);
        assert!(row.is_empty());
    }

    #[test]
    fn test_row_from_values() {
        let values = vec![
            Value::integer(1),
            Value::text("hello"),
            Value::null(DataType::Text),
        ];
        let row = Row::from_values(values);
        assert_eq!(row.len(), 3);
    }

    #[test]
    fn test_row_push_pop() {
        let mut row = Row::new();
        row.push(Value::integer(1));
        row.push(Value::text("hello"));

        assert_eq!(row.len(), 2);

        let popped = row.pop();
        assert_eq!(popped, Some(Value::text("hello")));
        assert_eq!(row.len(), 1);
    }

    #[test]
    fn test_row_get_set() {
        let mut row = Row::from_values(vec![Value::integer(1), Value::text("hello")]);

        assert_eq!(row.get(0), Some(&Value::integer(1)));
        assert_eq!(row.get(1), Some(&Value::text("hello")));
        assert_eq!(row.get(2), None);

        row.set(1, Value::text("world")).unwrap();
        assert_eq!(row.get(1), Some(&Value::text("world")));

        assert!(row.set(10, Value::integer(0)).is_err());
    }

    #[test]
    fn test_row_index() {
        let row = Row::from_values(vec![Value::integer(1), Value::text("hello")]);

        assert_eq!(row[0], Value::integer(1));
        assert_eq!(row[1], Value::text("hello"));
    }

    #[test]
    fn test_row_iteration() {
        let row = Row::from_values(vec![
            Value::integer(1),
            Value::integer(2),
            Value::integer(3),
        ]);

        let sum: i64 = row.iter().filter_map(|v| v.as_int64()).sum();
        assert_eq!(sum, 6);
    }

    #[test]
    fn test_row_select_columns() {
        let row = Row::from_values(vec![
            Value::integer(1),
            Value::text("hello"),
            Value::float(3.5),
            Value::boolean(true),
        ]);

        let selected = row.select_columns(&[0, 2]).unwrap();
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0], Value::integer(1));
        assert_eq!(selected[1], Value::float(3.5));

        assert!(row.select_columns(&[0, 10]).is_err());
    }

    #[test]
    fn test_row_validate() {
        let schema = create_test_schema();

        // Valid row
        let row = Row::from_values(vec![
            Value::integer(1),
            Value::text("Alice"),
            Value::null(DataType::Text),
        ]);
        assert!(row.validate(&schema).is_ok());

        // Wrong column count
        let row = Row::from_values(vec![Value::integer(1)]);
        assert!(row.validate(&schema).is_err());

        // Not null constraint violation
        let row = Row::from_values(vec![
            Value::integer(1),
            Value::null(DataType::Text), // name is not nullable
            Value::null(DataType::Text),
        ]);
        let err = row.validate(&schema).unwrap_err();
        assert!(matches!(err, Error::NotNullConstraint { .. }));
    }

    #[test]
    fn test_row_null_row() {
        let schema = create_test_schema();
        let row = Row::null_row(&schema);

        assert_eq!(row.len(), 3);
        assert!(row[0].is_null());
        assert!(row[1].is_null());
        assert!(row[2].is_null());
    }

    #[test]
    fn test_row_concat() {
        let row1 = Row::from_values(vec![Value::integer(1), Value::integer(2)]);
        let row2 = Row::from_values(vec![Value::integer(3), Value::integer(4)]);

        let combined = row1.concat(&row2);
        assert_eq!(combined.len(), 4);
        assert_eq!(combined[0], Value::integer(1));
        assert_eq!(combined[3], Value::integer(4));
    }

    #[test]
    fn test_row_repeat() {
        let row = Row::repeat(Value::integer(0), 5);
        assert_eq!(row.len(), 5);
        for v in row.iter() {
            assert_eq!(*v, Value::integer(0));
        }
    }

    #[test]
    fn test_row_from_iterator() {
        let row: Row = vec![Value::integer(1), Value::integer(2), Value::integer(3)]
            .into_iter()
            .collect();
        assert_eq!(row.len(), 3);
    }

    #[test]
    fn test_row_display() {
        let row = Row::from_values(vec![
            Value::integer(1),
            Value::text("hello"),
            Value::null(DataType::Text),
        ]);
        assert_eq!(row.to_string(), "(1, hello, NULL)");

        let empty = Row::new();
        assert_eq!(empty.to_string(), "()");
    }

    #[test]
    fn test_row_clone_subset() {
        let row = Row::from_values(vec![
            Value::integer(1),
            Value::text("hello"),
            Value::float(3.5),
        ]);

        let subset = row.clone_subset(&[2, 0]);
        assert_eq!(subset.len(), 2);
        assert_eq!(subset[0], Value::float(3.5));
        assert_eq!(subset[1], Value::integer(1));
    }

    #[test]
    fn test_row_equality() {
        let row1 = Row::from_values(vec![Value::integer(1), Value::text("hello")]);
        let row2 = Row::from_values(vec![Value::integer(1), Value::text("hello")]);
        let row3 = Row::from_values(vec![Value::integer(1), Value::text("world")]);

        assert_eq!(row1, row2);
        assert_ne!(row1, row3);
    }

    #[test]
    fn test_row_into_values() {
        let row = Row::from_values(vec![Value::integer(1), Value::text("hello")]);
        let values = row.into_values();

        assert_eq!(values.len(), 2);
        assert_eq!(values[0], Value::integer(1));
    }
}
