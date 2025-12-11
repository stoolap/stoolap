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

//! Core types and definitions for Stoolap
//!
//! This module contains the fundamental types used throughout the database:
//!
//! - [`DataType`] - SQL data types (INTEGER, TEXT, FLOAT, etc.)
//! - [`Operator`] - Comparison operators (=, !=, >, <, etc.)
//! - [`Value`] - Runtime values with type information
//! - [`Row`] - A database row (collection of values)
//! - [`Schema`] - Table schema definition
//! - [`SchemaColumn`] - Column definition
//! - [`Error`] - Error types for all storage operations

pub mod error;
pub mod row;
pub mod schema;
pub mod types;
pub mod value;

// Re-export main types for convenience
pub use error::{Error, Result};
pub use row::Row;
pub use schema::{Schema, SchemaBuilder, SchemaColumn};
pub use types::{DataType, IndexEntry, IndexType, IsolationLevel, Operator};
pub use value::{parse_timestamp, Value};

#[cfg(test)]
mod integration_tests {
    use super::*;
    use chrono::Datelike;

    /// Integration test: Create a schema, build rows, validate
    #[test]
    fn test_schema_row_integration() {
        // Create a schema
        let schema = SchemaBuilder::new("users")
            .add_primary_key("id", DataType::Integer)
            .add("name", DataType::Text)
            .add_nullable("email", DataType::Text)
            .add("active", DataType::Boolean)
            .build();

        // Create a valid row
        let row = Row::from_values(vec![
            Value::integer(1),
            Value::text("Alice"),
            Value::text("alice@example.com"),
            Value::boolean(true),
        ]);

        // Validate the row
        assert!(row.validate(&schema).is_ok());

        // Create a row with NULL in nullable column
        let row = Row::from_values(vec![
            Value::integer(2),
            Value::text("Bob"),
            Value::null(DataType::Text),
            Value::boolean(false),
        ]);
        assert!(row.validate(&schema).is_ok());

        // Create a row with NULL in non-nullable column (should fail)
        let row = Row::from_values(vec![
            Value::integer(3),
            Value::null(DataType::Text), // name is not nullable
            Value::null(DataType::Text),
            Value::boolean(true),
        ]);
        assert!(row.validate(&schema).is_err());
    }

    /// Integration test: Value comparisons across types
    #[test]
    fn test_value_comparison_integration() {
        // Same type comparisons
        assert!(Value::integer(1).compare(&Value::integer(2)).unwrap() == std::cmp::Ordering::Less);
        assert!(Value::text("a").compare(&Value::text("b")).unwrap() == std::cmp::Ordering::Less);

        // Cross-type numeric comparisons
        assert!(Value::integer(1).compare(&Value::float(1.5)).unwrap() == std::cmp::Ordering::Less);
        assert!(
            Value::float(2.5).compare(&Value::integer(2)).unwrap() == std::cmp::Ordering::Greater
        );

        // NULL comparisons
        assert!(Value::null(DataType::Integer)
            .compare(&Value::integer(0))
            .is_err());
    }

    /// Integration test: Type conversions
    #[test]
    fn test_value_conversion_integration() {
        // Integer conversions
        let v = Value::integer(42);
        assert_eq!(v.as_int64(), Some(42));
        assert_eq!(v.as_float64(), Some(42.0));
        assert_eq!(v.as_string(), Some("42".to_string()));
        assert_eq!(v.as_boolean(), Some(true));

        // Float conversions
        let v = Value::float(3.5);
        assert_eq!(v.as_int64(), Some(3)); // truncates
        assert_eq!(v.as_float64(), Some(3.5));

        // Boolean conversions
        let v = Value::boolean(true);
        assert_eq!(v.as_int64(), Some(1));
        assert_eq!(v.as_float64(), Some(1.0));
        assert_eq!(v.as_string(), Some("true".to_string()));

        // String parsing
        let v = Value::text("42");
        assert_eq!(v.as_int64(), Some(42));
        assert_eq!(v.as_float64(), Some(42.0));

        // Timestamp parsing
        let v = Value::text("2024-01-15T10:30:00Z");
        assert!(v.as_timestamp().is_some());
    }

    /// Integration test: Timestamp handling
    #[test]
    fn test_timestamp_integration() {
        use chrono::TimeZone;

        // Parse various formats
        let ts1 = parse_timestamp("2024-01-15T10:30:00Z").unwrap();
        let ts2 = parse_timestamp("2024-01-15 10:30:00").unwrap();
        let ts3 = parse_timestamp("2024-01-15").unwrap();

        assert_eq!(ts1.year(), 2024);
        assert_eq!(ts2.year(), 2024);
        assert_eq!(ts3.year(), 2024);

        // Create timestamp value
        let v = Value::timestamp(ts1);
        assert_eq!(v.data_type(), DataType::Timestamp);
        assert!(v.as_timestamp().is_some());

        // Timestamp comparison
        let v1 = Value::timestamp(ts1);
        let v2 = Value::timestamp(chrono::Utc.with_ymd_and_hms(2024, 1, 15, 11, 0, 0).unwrap());
        assert!(v1.compare(&v2).unwrap() == std::cmp::Ordering::Less);
    }

    /// Integration test: Error handling
    #[test]
    fn test_error_integration() {
        // Constraint errors
        let err = Error::not_null_constraint("email");
        assert!(err.is_constraint_violation());
        assert!(!err.is_not_found());

        // Not found errors
        let err = Error::TableNotFound;
        assert!(err.is_not_found());
        assert!(!err.is_constraint_violation());

        // Transaction errors
        let err = Error::TransactionAborted;
        assert!(err.is_transaction_error());
    }

    /// Integration test: Schema operations
    #[test]
    fn test_schema_operations_integration() {
        let mut schema = SchemaBuilder::new("products")
            .add_primary_key("id", DataType::Integer)
            .add("name", DataType::Text)
            .build();

        // Add column
        schema
            .add_column(SchemaColumn::nullable(2, "description", DataType::Text))
            .unwrap();
        assert_eq!(schema.column_count(), 3);

        // Rename column
        schema.rename_column("name", "product_name").unwrap();
        assert!(schema.has_column("product_name"));
        assert!(!schema.has_column("name"));

        // Modify column
        schema
            .modify_column("description", None, Some(false))
            .unwrap();
        assert!(!schema.get_column_by_name("description").unwrap().nullable);

        // Remove column
        schema.remove_column("description").unwrap();
        assert_eq!(schema.column_count(), 2);
    }

    /// Integration test: Row creation
    #[test]
    fn test_row_creation_integration() {
        // Empty row
        let empty = Row::new();
        assert!(empty.is_empty());

        // Row with values using from_values
        let row = Row::from_values(vec![
            Value::integer(1),
            Value::text("hello"),
            Value::boolean(true),
            Value::float(3.5),
        ]);
        assert_eq!(row.len(), 4);
        assert_eq!(row[0], Value::integer(1));
        assert_eq!(row[1], Value::text("hello"));
        assert_eq!(row[2], Value::boolean(true));
        assert_eq!(row[3], Value::float(3.5));
    }
}
