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

//! Schema types for Stoolap - table and column definitions
//!
//! This module defines SchemaColumn and Schema types for table structure.

use std::fmt;
use std::sync::OnceLock;

use chrono::{DateTime, Utc};
use rustc_hash::FxHashMap;

use super::error::{Error, Result};
use super::types::DataType;

/// A column definition in a table schema
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchemaColumn {
    /// Unique identifier for the column (0-based index)
    pub id: usize,

    /// Column name
    pub name: String,

    /// Data type of the column
    pub data_type: DataType,

    /// Whether the column can contain NULL values
    pub nullable: bool,

    /// Whether this column is part of the primary key
    pub primary_key: bool,

    /// Whether this column auto-increments (generates sequential IDs for NULL values)
    pub auto_increment: bool,

    /// Default value expression as a string (to be parsed and evaluated during INSERT)
    pub default_expr: Option<String>,

    /// Pre-computed default value for schema evolution (used when adding column to existing rows)
    pub default_value: Option<super::Value>,

    /// CHECK constraint expression as a string (to be parsed and evaluated during INSERT)
    pub check_expr: Option<String>,
}

impl SchemaColumn {
    /// Create a new column definition
    pub fn new(
        id: usize,
        name: impl Into<String>,
        data_type: DataType,
        nullable: bool,
        primary_key: bool,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            data_type,
            nullable,
            primary_key,
            auto_increment: false,
            default_expr: None,
            default_value: None,
            check_expr: None,
        }
    }

    /// Create a new column definition with all options
    #[allow(clippy::too_many_arguments)]
    pub fn with_constraints(
        id: usize,
        name: impl Into<String>,
        data_type: DataType,
        nullable: bool,
        primary_key: bool,
        auto_increment: bool,
        default_expr: Option<String>,
        check_expr: Option<String>,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            data_type,
            nullable,
            primary_key,
            auto_increment,
            default_expr,
            default_value: None,
            check_expr,
        }
    }

    /// Create a new column definition with pre-computed default value
    #[allow(clippy::too_many_arguments)]
    pub fn with_default_value(
        id: usize,
        name: impl Into<String>,
        data_type: DataType,
        nullable: bool,
        primary_key: bool,
        auto_increment: bool,
        default_expr: Option<String>,
        default_value: Option<super::Value>,
        check_expr: Option<String>,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            data_type,
            nullable,
            primary_key,
            auto_increment,
            default_expr,
            default_value,
            check_expr,
        }
    }

    /// Create a simple non-nullable, non-primary-key column
    pub fn simple(id: usize, name: impl Into<String>, data_type: DataType) -> Self {
        Self::new(id, name, data_type, false, false)
    }

    /// Create a nullable column
    pub fn nullable(id: usize, name: impl Into<String>, data_type: DataType) -> Self {
        Self::new(id, name, data_type, true, false)
    }

    /// Create a primary key column
    pub fn primary_key(id: usize, name: impl Into<String>, data_type: DataType) -> Self {
        Self::new(id, name, data_type, false, true)
    }
}

impl fmt::Display for SchemaColumn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.name, self.data_type)?;
        if self.primary_key {
            write!(f, " PRIMARY KEY")?;
        }
        if !self.nullable && !self.primary_key {
            write!(f, " NOT NULL")?;
        }
        Ok(())
    }
}

/// Table schema definition
///

#[derive(Debug)]
pub struct Schema {
    /// Name of the table
    pub table_name: String,

    /// Pre-computed lowercase table name for case-insensitive lookups
    pub table_name_lower: String,

    /// Column definitions
    pub columns: Vec<SchemaColumn>,

    /// Creation timestamp
    pub created_at: DateTime<Utc>,

    /// Last update timestamp
    pub updated_at: DateTime<Utc>,

    /// Cached column names (computed lazily on first access)
    column_names_cache: OnceLock<Vec<String>>,

    /// Cached primary key column index (computed lazily on first access)
    /// None means not computed yet, Some(None) means no PK, Some(Some(idx)) means PK at idx
    pk_column_index_cache: OnceLock<Option<usize>>,

    /// Cached column index map (lowercase name -> index) for O(1) column lookup
    column_index_map_cache: OnceLock<FxHashMap<String, usize>>,
}

impl Clone for Schema {
    fn clone(&self) -> Self {
        Self {
            table_name: self.table_name.clone(),
            table_name_lower: self.table_name_lower.clone(),
            columns: self.columns.clone(),
            created_at: self.created_at,
            updated_at: self.updated_at,
            column_names_cache: OnceLock::new(), // Don't clone cache, it's recomputed lazily
            pk_column_index_cache: OnceLock::new(), // Don't clone cache, it's recomputed lazily
            column_index_map_cache: OnceLock::new(), // Don't clone cache, it's recomputed lazily
        }
    }
}

impl PartialEq for Schema {
    fn eq(&self, other: &Self) -> bool {
        self.table_name == other.table_name
            && self.columns == other.columns
            && self.created_at == other.created_at
            && self.updated_at == other.updated_at
    }
}

impl Eq for Schema {}

impl Schema {
    /// Create a new schema with the given table name and columns
    pub fn new(table_name: impl Into<String>, columns: Vec<SchemaColumn>) -> Self {
        let now = Utc::now();
        let name = table_name.into();
        let name_lower = name.to_lowercase();
        Self {
            table_name: name,
            table_name_lower: name_lower,
            columns,
            created_at: now,
            updated_at: now,
            column_names_cache: OnceLock::new(),
            pk_column_index_cache: OnceLock::new(),
            column_index_map_cache: OnceLock::new(),
        }
    }

    /// Create a new schema with explicit timestamps
    pub fn with_timestamps(
        table_name: impl Into<String>,
        columns: Vec<SchemaColumn>,
        created_at: DateTime<Utc>,
        updated_at: DateTime<Utc>,
    ) -> Self {
        let name = table_name.into();
        let name_lower = name.to_lowercase();
        Self {
            table_name: name,
            table_name_lower: name_lower,
            columns,
            created_at,
            updated_at,
            column_names_cache: OnceLock::new(),
            pk_column_index_cache: OnceLock::new(),
            column_index_map_cache: OnceLock::new(),
        }
    }

    /// Get the number of columns
    pub fn column_count(&self) -> usize {
        self.columns.len()
    }

    /// Check if the schema has any columns
    pub fn is_empty(&self) -> bool {
        self.columns.is_empty()
    }

    /// Find a column by name (case-insensitive)
    /// Returns the column index and reference
    pub fn find_column(&self, name: &str) -> Option<(usize, &SchemaColumn)> {
        let name_lower = name.to_lowercase();
        self.columns
            .iter()
            .enumerate()
            .find(|(_, col)| col.name.to_lowercase() == name_lower)
    }

    /// Get a column by index
    pub fn get_column(&self, index: usize) -> Option<&SchemaColumn> {
        self.columns.get(index)
    }

    /// Get a column by name (case-insensitive)
    pub fn get_column_by_name(&self, name: &str) -> Option<&SchemaColumn> {
        self.find_column(name).map(|(_, col)| col)
    }

    /// Get the column index by name (case-insensitive)
    pub fn get_column_index(&self, name: &str) -> Option<usize> {
        self.find_column(name).map(|(idx, _)| idx)
    }

    /// Get the data type of a column by name
    pub fn get_column_type(&self, name: &str) -> Option<DataType> {
        self.get_column_by_name(name).map(|col| col.data_type)
    }

    /// Check if a column exists by name
    pub fn has_column(&self, name: &str) -> bool {
        self.find_column(name).is_some()
    }

    /// Get all column names as borrowed strings (allocates Vec but not strings)
    pub fn column_names(&self) -> Vec<&str> {
        self.columns.iter().map(|c| c.name.as_str()).collect()
    }

    /// Get all column names as owned strings (cached - only clones once)
    ///
    /// This is more efficient than calling `.columns.iter().map(|c| c.name.clone()).collect()`
    /// repeatedly, as the result is computed once and cached.
    #[inline]
    pub fn column_names_owned(&self) -> &[String] {
        self.column_names_cache
            .get_or_init(|| self.columns.iter().map(|c| c.name.clone()).collect())
    }

    /// Get a cached map of lowercase column names to their indices
    /// OPTIMIZATION: Cached to avoid creating this map on every query
    #[inline]
    pub fn column_index_map(&self) -> &FxHashMap<String, usize> {
        self.column_index_map_cache.get_or_init(|| {
            self.columns
                .iter()
                .enumerate()
                .map(|(i, c)| (c.name.to_lowercase(), i))
                .collect()
        })
    }

    /// Get the primary key columns
    pub fn primary_key_columns(&self) -> Vec<&SchemaColumn> {
        self.columns.iter().filter(|c| c.primary_key).collect()
    }

    /// Check if the schema has a primary key
    pub fn has_primary_key(&self) -> bool {
        self.columns.iter().any(|c| c.primary_key)
    }

    /// Get the primary key column indices
    pub fn primary_key_indices(&self) -> Vec<usize> {
        self.columns
            .iter()
            .enumerate()
            .filter(|(_, c)| c.primary_key)
            .map(|(i, _)| i)
            .collect()
    }

    /// Get the single primary key column index (cached for performance)
    /// Returns None if there's no PK or if PK is not an integer type
    /// OPTIMIZATION: Cached to avoid iteration on every INSERT
    #[inline]
    pub fn pk_column_index(&self) -> Option<usize> {
        *self.pk_column_index_cache.get_or_init(|| {
            for (i, col) in self.columns.iter().enumerate() {
                if col.primary_key && col.data_type == DataType::Integer {
                    return Some(i);
                }
            }
            None
        })
    }

    /// Validate column count matches expected value
    pub fn validate_column_count(&self, expected: usize) -> Result<()> {
        if self.columns.len() != expected {
            return Err(Error::table_columns_not_match(expected, self.columns.len()));
        }
        Ok(())
    }

    /// Mark the schema as updated (sets updated_at to now)
    pub fn mark_updated(&mut self) {
        self.updated_at = Utc::now();
    }

    /// Add a column to the schema
    pub fn add_column(&mut self, column: SchemaColumn) -> Result<()> {
        // Check for duplicate column name
        if self.has_column(&column.name) {
            return Err(Error::DuplicateColumn);
        }
        self.columns.push(column);
        self.mark_updated();
        Ok(())
    }

    /// Remove a column by name
    pub fn remove_column(&mut self, name: &str) -> Result<SchemaColumn> {
        let idx = self.get_column_index(name).ok_or(Error::ColumnNotFound)?;
        let column = self.columns.remove(idx);

        // Re-index remaining columns
        for (i, col) in self.columns.iter_mut().enumerate() {
            col.id = i;
        }

        self.mark_updated();
        Ok(column)
    }

    /// Rename a column
    pub fn rename_column(&mut self, old_name: &str, new_name: impl Into<String>) -> Result<()> {
        let new_name = new_name.into();

        // Check new name doesn't exist
        if self.has_column(&new_name) {
            return Err(Error::DuplicateColumn);
        }

        let idx = self
            .get_column_index(old_name)
            .ok_or(Error::ColumnNotFound)?;

        self.columns[idx].name = new_name;
        self.mark_updated();
        Ok(())
    }

    /// Modify a column's properties (except name)
    pub fn modify_column(
        &mut self,
        name: &str,
        data_type: Option<DataType>,
        nullable: Option<bool>,
    ) -> Result<()> {
        let idx = self.get_column_index(name).ok_or(Error::ColumnNotFound)?;

        if let Some(dt) = data_type {
            self.columns[idx].data_type = dt;
        }
        if let Some(n) = nullable {
            self.columns[idx].nullable = n;
        }

        self.mark_updated();
        Ok(())
    }
}

impl Default for Schema {
    fn default() -> Self {
        Self::new("", Vec::new())
    }
}

impl fmt::Display for Schema {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CREATE TABLE {} (", self.table_name)?;
        for (i, col) in self.columns.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", col)?;
        }
        write!(f, ")")
    }
}

/// Builder for creating schemas more ergonomically
pub struct SchemaBuilder {
    table_name: String,
    columns: Vec<SchemaColumn>,
}

impl SchemaBuilder {
    /// Create a new schema builder
    pub fn new(table_name: impl Into<String>) -> Self {
        Self {
            table_name: table_name.into(),
            columns: Vec::new(),
        }
    }

    /// Add a column
    pub fn column(
        mut self,
        name: impl Into<String>,
        data_type: DataType,
        nullable: bool,
        primary_key: bool,
    ) -> Self {
        let id = self.columns.len();
        self.columns.push(SchemaColumn::new(
            id,
            name,
            data_type,
            nullable,
            primary_key,
        ));
        self
    }

    /// Add a simple non-nullable column
    pub fn add(self, name: impl Into<String>, data_type: DataType) -> Self {
        self.column(name, data_type, false, false)
    }

    /// Add a nullable column
    pub fn add_nullable(self, name: impl Into<String>, data_type: DataType) -> Self {
        self.column(name, data_type, true, false)
    }

    /// Add a primary key column
    pub fn add_primary_key(self, name: impl Into<String>, data_type: DataType) -> Self {
        self.column(name, data_type, false, true)
    }

    /// Add a column with full constraints (default, check)
    #[allow(clippy::too_many_arguments)]
    pub fn add_with_constraints(
        mut self,
        name: impl Into<String>,
        data_type: DataType,
        nullable: bool,
        primary_key: bool,
        auto_increment: bool,
        default_expr: Option<String>,
        check_expr: Option<String>,
    ) -> Self {
        let id = self.columns.len();
        self.columns.push(SchemaColumn::with_constraints(
            id,
            name,
            data_type,
            nullable,
            primary_key,
            auto_increment,
            default_expr,
            check_expr,
        ));
        self
    }

    /// Build the schema
    pub fn build(self) -> Schema {
        Schema::new(self.table_name, self.columns)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_schema() -> Schema {
        SchemaBuilder::new("users")
            .add_primary_key("id", DataType::Integer)
            .add("name", DataType::Text)
            .add_nullable("email", DataType::Text)
            .add("active", DataType::Boolean)
            .build()
    }

    #[test]
    fn test_schema_column_creation() {
        let col = SchemaColumn::new(0, "id", DataType::Integer, false, true);
        assert_eq!(col.id, 0);
        assert_eq!(col.name, "id");
        assert_eq!(col.data_type, DataType::Integer);
        assert!(!col.nullable);
        assert!(col.primary_key);
    }

    #[test]
    fn test_schema_column_helpers() {
        let simple = SchemaColumn::simple(0, "name", DataType::Text);
        assert!(!simple.nullable);
        assert!(!simple.primary_key);

        let nullable = SchemaColumn::nullable(1, "email", DataType::Text);
        assert!(nullable.nullable);
        assert!(!nullable.primary_key);

        let pk = SchemaColumn::primary_key(2, "id", DataType::Integer);
        assert!(!pk.nullable);
        assert!(pk.primary_key);
    }

    #[test]
    fn test_schema_creation() {
        let schema = create_test_schema();
        assert_eq!(schema.table_name, "users");
        assert_eq!(schema.column_count(), 4);
        assert!(!schema.is_empty());
    }

    #[test]
    fn test_schema_find_column() {
        let schema = create_test_schema();

        // Find by exact name
        let (idx, col) = schema.find_column("name").unwrap();
        assert_eq!(idx, 1);
        assert_eq!(col.name, "name");

        // Case-insensitive
        let (idx, _) = schema.find_column("NAME").unwrap();
        assert_eq!(idx, 1);

        // Not found
        assert!(schema.find_column("nonexistent").is_none());
    }

    #[test]
    fn test_schema_get_column() {
        let schema = create_test_schema();

        let col = schema.get_column(0).unwrap();
        assert_eq!(col.name, "id");

        let col = schema.get_column_by_name("email").unwrap();
        assert_eq!(col.data_type, DataType::Text);
        assert!(col.nullable);

        assert!(schema.get_column(100).is_none());
    }

    #[test]
    fn test_schema_column_names() {
        let schema = create_test_schema();
        let names = schema.column_names();
        assert_eq!(names, vec!["id", "name", "email", "active"]);
    }

    #[test]
    fn test_schema_primary_key() {
        let schema = create_test_schema();

        assert!(schema.has_primary_key());

        let pk_cols = schema.primary_key_columns();
        assert_eq!(pk_cols.len(), 1);
        assert_eq!(pk_cols[0].name, "id");

        let pk_indices = schema.primary_key_indices();
        assert_eq!(pk_indices, vec![0]);
    }

    #[test]
    fn test_schema_validate_column_count() {
        let schema = create_test_schema();

        assert!(schema.validate_column_count(4).is_ok());

        let err = schema.validate_column_count(3).unwrap_err();
        assert!(matches!(
            err,
            Error::TableColumnsNotMatch {
                expected: 3,
                got: 4
            }
        ));
    }

    #[test]
    fn test_schema_add_column() {
        let mut schema = create_test_schema();
        let original_count = schema.column_count();

        schema
            .add_column(SchemaColumn::simple(
                original_count,
                "age",
                DataType::Integer,
            ))
            .unwrap();

        assert_eq!(schema.column_count(), original_count + 1);
        assert!(schema.has_column("age"));

        // Duplicate column should fail
        let err = schema
            .add_column(SchemaColumn::simple(0, "age", DataType::Integer))
            .unwrap_err();
        assert!(matches!(err, Error::DuplicateColumn));
    }

    #[test]
    fn test_schema_remove_column() {
        let mut schema = create_test_schema();

        let removed = schema.remove_column("email").unwrap();
        assert_eq!(removed.name, "email");
        assert_eq!(schema.column_count(), 3);
        assert!(!schema.has_column("email"));

        // Column IDs should be re-indexed
        assert_eq!(schema.columns[2].id, 2);

        // Removing non-existent column should fail
        assert!(schema.remove_column("nonexistent").is_err());
    }

    #[test]
    fn test_schema_rename_column() {
        let mut schema = create_test_schema();

        schema.rename_column("name", "full_name").unwrap();
        assert!(schema.has_column("full_name"));
        assert!(!schema.has_column("name"));

        // Renaming to existing name should fail
        let err = schema.rename_column("full_name", "id").unwrap_err();
        assert!(matches!(err, Error::DuplicateColumn));

        // Renaming non-existent column should fail
        assert!(schema.rename_column("nonexistent", "new_name").is_err());
    }

    #[test]
    fn test_schema_modify_column() {
        let mut schema = create_test_schema();

        schema
            .modify_column("name", Some(DataType::Json), Some(true))
            .unwrap();

        let col = schema.get_column_by_name("name").unwrap();
        assert_eq!(col.data_type, DataType::Json);
        assert!(col.nullable);

        // Modifying non-existent column should fail
        assert!(schema
            .modify_column("nonexistent", None, Some(true))
            .is_err());
    }

    #[test]
    fn test_schema_column_display() {
        let col = SchemaColumn::new(0, "id", DataType::Integer, false, true);
        assert_eq!(col.to_string(), "id INTEGER PRIMARY KEY");

        let col = SchemaColumn::new(1, "name", DataType::Text, false, false);
        assert_eq!(col.to_string(), "name TEXT NOT NULL");

        let col = SchemaColumn::new(2, "email", DataType::Text, true, false);
        assert_eq!(col.to_string(), "email TEXT");
    }

    #[test]
    fn test_schema_display() {
        let schema = SchemaBuilder::new("users")
            .add_primary_key("id", DataType::Integer)
            .add("name", DataType::Text)
            .build();

        let expected = "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL)";
        assert_eq!(schema.to_string(), expected);
    }

    #[test]
    fn test_schema_builder() {
        let schema = SchemaBuilder::new("products")
            .add_primary_key("id", DataType::Integer)
            .add("name", DataType::Text)
            .add_nullable("description", DataType::Text)
            .add("price", DataType::Float)
            .build();

        assert_eq!(schema.table_name, "products");
        assert_eq!(schema.column_count(), 4);
        assert!(schema.get_column_by_name("id").unwrap().primary_key);
        assert!(schema.get_column_by_name("description").unwrap().nullable);
    }

    #[test]
    fn test_schema_timestamps() {
        let schema1 = Schema::new("test", vec![]);
        std::thread::sleep(std::time::Duration::from_millis(10));
        let schema2 = Schema::new("test", vec![]);

        // Different creation times
        assert!(schema2.created_at >= schema1.created_at);
    }

    #[test]
    fn test_schema_get_column_type() {
        let schema = create_test_schema();

        assert_eq!(schema.get_column_type("id"), Some(DataType::Integer));
        assert_eq!(schema.get_column_type("name"), Some(DataType::Text));
        assert_eq!(schema.get_column_type("active"), Some(DataType::Boolean));
        assert_eq!(schema.get_column_type("nonexistent"), None);
    }
}
