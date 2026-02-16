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

//! SHOW and DESCRIBE statement execution
//!
//! This module handles metadata query commands:
//! - SHOW TABLES
//! - SHOW VIEWS
//! - SHOW CREATE TABLE
//! - SHOW CREATE VIEW
//! - SHOW INDEXES
//! - DESCRIBE

use rustc_hash::FxHashSet;

use crate::common::SmartString;
use crate::core::row_vec::RowVec;
use crate::core::{Error, Result, Row, Value};
use crate::parser::ast::*;
use crate::storage::traits::{Engine, QueryResult};

use super::context::ExecutionContext;
use super::result::ExecutorResult;
use super::Executor;

impl Executor {
    /// Execute SHOW TABLES statement
    pub(crate) fn execute_show_tables(
        &self,
        _stmt: &ShowTablesStatement,
        _ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        let tx = self.engine.begin_transaction()?;
        let tables = tx.list_tables()?;

        let columns = vec!["table_name".to_string()];
        let mut rows = RowVec::with_capacity(tables.len());
        for (i, name) in tables.into_iter().enumerate() {
            rows.push((
                i as i64,
                Row::from_values(vec![Value::Text(SmartString::from_string(name))]),
            ));
        }

        Ok(Box::new(ExecutorResult::new(columns, rows)))
    }

    /// Execute SHOW VIEWS statement
    pub(crate) fn execute_show_views(
        &self,
        _stmt: &ShowViewsStatement,
        _ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        let views = self.engine.list_views()?;

        let columns = vec!["view_name".to_string()];
        let mut rows = RowVec::with_capacity(views.len());
        for (i, name) in views.into_iter().enumerate() {
            rows.push((
                i as i64,
                Row::from_values(vec![Value::Text(SmartString::from_string(name))]),
            ));
        }

        Ok(Box::new(ExecutorResult::new(columns, rows)))
    }

    /// Execute SHOW CREATE TABLE statement
    pub(crate) fn execute_show_create_table(
        &self,
        stmt: &ShowCreateTableStatement,
        _ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        let table_name = &stmt.table_name.value;
        let tx = self.engine.begin_transaction()?;
        let table = tx.get_table(table_name)?;
        let schema = table.schema();

        // Get unique column names from indexes
        let mut unique_columns: FxHashSet<String> = FxHashSet::default();
        if let Ok(indexes) = self.engine.list_table_indexes(table_name) {
            for index_name in indexes.keys() {
                if let Some(index) = table.get_index(index_name) {
                    // Only single-column unique indexes should be shown as UNIQUE constraint on column
                    let col_names = index.column_names();
                    if index.is_unique() && col_names.len() == 1 {
                        unique_columns.insert(col_names[0].to_lowercase());
                    }
                }
            }
        }

        // Build CREATE TABLE statement
        let mut create_sql = format!("CREATE TABLE {} (", table_name);
        let col_defs: Vec<String> = schema
            .columns
            .iter()
            .map(|col| {
                let mut def = format!("{} {:?}", col.name, col.data_type);
                if col.primary_key {
                    def.push_str(" PRIMARY KEY");
                    if col.auto_increment {
                        def.push_str(" AUTO_INCREMENT");
                    }
                } else {
                    // Check if this column has a UNIQUE constraint
                    if unique_columns.contains(&col.name_lower) {
                        def.push_str(" UNIQUE");
                    }
                    if !col.nullable {
                        def.push_str(" NOT NULL");
                    }
                }
                // Add DEFAULT if present
                if let Some(default_expr) = &col.default_expr {
                    def.push_str(&format!(" DEFAULT {}", default_expr));
                }
                // Add CHECK constraint if present
                if let Some(check) = &col.check_expr {
                    def.push_str(&format!(" CHECK ({})", check));
                }
                def
            })
            .collect();
        create_sql.push_str(&col_defs.join(", "));

        // Add foreign key constraints
        for fk in &schema.foreign_keys {
            use std::fmt::Write;
            write!(
                create_sql,
                ", FOREIGN KEY ({}) REFERENCES {}({}) ON DELETE {} ON UPDATE {}",
                fk.column_name,
                fk.referenced_table,
                fk.referenced_column,
                fk.on_delete,
                fk.on_update
            )
            .unwrap();
        }

        create_sql.push(')');

        let columns = vec!["Table".to_string(), "Create Table".to_string()];
        let mut rows = RowVec::with_capacity(1);
        rows.push((
            0,
            Row::from_values(vec![
                Value::Text(SmartString::new(table_name)),
                Value::Text(SmartString::from_string(create_sql)),
            ]),
        ));

        Ok(Box::new(ExecutorResult::new(columns, rows)))
    }

    /// Execute SHOW CREATE VIEW statement
    pub(crate) fn execute_show_create_view(
        &self,
        stmt: &ShowCreateViewStatement,
        _ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        let view_name = &stmt.view_name.value;

        // Get the view definition
        let view_def = self
            .engine
            .get_view(view_name)?
            .ok_or_else(|| Error::ViewNotFound(view_name.to_string()))?;

        // Build CREATE VIEW statement
        let create_sql = format!(
            "CREATE VIEW {} AS {}",
            view_def.original_name, view_def.query
        );

        let columns = vec!["View".to_string(), "Create View".to_string()];
        let mut rows = RowVec::with_capacity(1);
        rows.push((
            0,
            Row::from_values(vec![
                Value::Text(SmartString::new(&view_def.original_name)),
                Value::Text(SmartString::from_string(create_sql)),
            ]),
        ));

        Ok(Box::new(ExecutorResult::new(columns, rows)))
    }

    /// Execute SHOW INDEXES statement
    pub(crate) fn execute_show_indexes(
        &self,
        stmt: &ShowIndexesStatement,
        _ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        let table_name = &stmt.table_name.value;

        // Get a table reference to access indexes
        let tx = self.engine.begin_transaction()?;
        let table = tx.get_table(table_name)?;

        // Get index info from version store through the table
        let index_names = {
            let indexes = self.engine.list_table_indexes(table_name)?;
            indexes.keys().cloned().collect::<Vec<_>>()
        };

        // Build rows with: table_name, index_name, column_name, index_type, is_unique
        let columns = vec![
            "table_name".to_string(),
            "index_name".to_string(),
            "column_name".to_string(),
            "index_type".to_string(),
            "is_unique".to_string(),
        ];

        let mut rows = RowVec::with_capacity(index_names.len());
        let mut row_id = 0i64;
        for index_name in index_names {
            // Try to get index details from the table's underlying storage
            if let Some(index) = table.get_index(&index_name) {
                let column_names = index.column_names();
                // Show all columns for multi-column indexes
                let column_name = if column_names.len() > 1 {
                    format!("({})", column_names.join(", "))
                } else {
                    column_names
                        .first()
                        .map(|s| s.to_string())
                        .unwrap_or_default()
                };
                let is_unique = index.is_unique();
                let index_type = index.index_type().as_str().to_uppercase();

                rows.push((
                    row_id,
                    Row::from_values(vec![
                        Value::Text(SmartString::new(table_name)),
                        Value::Text(SmartString::new(&index_name)),
                        Value::Text(SmartString::from_string(column_name)),
                        Value::Text(SmartString::from_string(index_type)),
                        Value::Boolean(is_unique),
                    ]),
                ));
                row_id += 1;
            }
        }

        Ok(Box::new(ExecutorResult::new(columns, rows)))
    }

    /// Execute DESCRIBE statement - shows table structure
    pub(crate) fn execute_describe(
        &self,
        stmt: &DescribeStatement,
        _ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        let table_name = &stmt.table_name.value;
        let tx = self.engine.begin_transaction()?;
        let table = tx.get_table(table_name)?;
        let schema = table.schema();

        // Column headers: Field, Type, Null, Key, Default, Extra
        let columns = vec![
            "Field".to_string(),
            "Type".to_string(),
            "Null".to_string(),
            "Key".to_string(),
            "Default".to_string(),
            "Extra".to_string(),
        ];

        let mut rows = RowVec::with_capacity(schema.columns.len());
        for (i, col) in schema.columns.iter().enumerate() {
            // Determine type string
            let type_str = format!("{:?}", col.data_type);

            // Determine nullability
            let null_str = if col.nullable { "YES" } else { "NO" };

            // Determine key type
            let key_str = if col.primary_key {
                "PRI"
            } else if schema.foreign_keys.iter().any(|fk| fk.column_index == i) {
                "MUL"
            } else {
                ""
            };

            // Get default value if any
            let default_str = col
                .default_expr
                .as_ref()
                .map(|v| v.to_string())
                .unwrap_or_default();

            // Extra info (e.g., auto_increment equivalent)
            let extra_str = if col.auto_increment {
                "auto_increment"
            } else {
                ""
            };

            rows.push((
                i as i64,
                Row::from_values(vec![
                    Value::Text(SmartString::new(&col.name)),
                    Value::Text(SmartString::from_string(type_str)),
                    Value::Text(SmartString::new(null_str)),
                    Value::Text(SmartString::new(key_str)),
                    Value::Text(SmartString::from_string(default_str)),
                    Value::Text(SmartString::new(extra_str)),
                ]),
            ));
        }

        Ok(Box::new(ExecutorResult::new(columns, rows)))
    }
}
