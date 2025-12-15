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

//! ANALYZE command executor for statistics collection
//!
//! This module implements the ANALYZE command which collects statistics
//! about tables and stores them in system tables for query optimization.

use std::time::SystemTime;

use rand::Rng;
use rustc_hash::FxHashSet;

use crate::core::{Error, Result, Row, Value};
use crate::parser::ast::AnalyzeStatement;
use crate::storage::mvcc::zonemap::{ZoneMapBuilder, DEFAULT_SEGMENT_SIZE};
use crate::storage::statistics::{
    is_stats_table, Histogram, DEFAULT_HISTOGRAM_BUCKETS, DEFAULT_SAMPLE_SIZE, SYS_COLUMN_STATS,
    SYS_TABLE_STATS,
};
use crate::storage::traits::{Engine, QueryResult, Transaction};

use super::context::ExecutionContext;
use super::result::ExecutorMemoryResult;
use super::Executor;

impl Executor {
    /// Execute ANALYZE statement
    ///
    /// Collects statistics for the specified table (or all tables if none specified)
    /// and stores them in the system statistics tables.
    pub(crate) fn execute_analyze(
        &self,
        stmt: &AnalyzeStatement,
        _ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        // Ensure system tables exist
        self.ensure_stats_tables_exist()?;

        // Get list of tables to analyze
        let tables_to_analyze: Vec<String> = if let Some(ref table_name) = stmt.table_name {
            // Analyze specific table
            vec![table_name.clone()]
        } else {
            // Analyze all tables - need a transaction to list tables
            let tx = self.engine.begin_transaction()?;
            let all_tables = tx.list_tables()?;
            all_tables
                .into_iter()
                .filter(|name| !is_stats_table(name))
                .collect()
        };

        let mut analyzed_count = 0;

        for table_name in &tables_to_analyze {
            // Skip system tables
            if is_stats_table(table_name) {
                continue;
            }

            // Begin a transaction for this table's analysis
            let mut tx = self.engine.begin_transaction()?;

            let success = match self.analyze_table(&mut *tx, table_name) {
                Ok(_) => {
                    tx.commit()?;
                    analyzed_count += 1;
                    true
                }
                Err(e) => {
                    let _ = tx.rollback();
                    // Log warning but continue with other tables
                    eprintln!("Warning: Failed to analyze table '{}': {}", table_name, e);
                    false
                }
            };

            // Invalidate cached statistics AFTER transaction is dropped
            // This avoids potential lock ordering issues between transaction locks
            // and stats_cache write lock
            if success {
                self.get_query_planner().invalidate_stats_cache(table_name);
            }
        }

        // Return result showing how many tables were analyzed
        let columns = vec!["tables_analyzed".to_string()];
        let rows = vec![Row::from_values(vec![Value::Integer(analyzed_count)])];

        Ok(Box::new(ExecutorMemoryResult::new(columns, rows)))
    }

    /// Ensure the system statistics tables exist
    fn ensure_stats_tables_exist(&self) -> Result<()> {
        use crate::storage::statistics::{CREATE_COLUMN_STATS_SQL, CREATE_TABLE_STATS_SQL};

        // Check if tables exist first - need a transaction
        let tx = self.engine.begin_transaction()?;
        let tables = tx.list_tables()?;
        let has_table_stats = tables
            .iter()
            .any(|t| t.eq_ignore_ascii_case(SYS_TABLE_STATS));
        let has_column_stats = tables
            .iter()
            .any(|t| t.eq_ignore_ascii_case(SYS_COLUMN_STATS));
        drop(tx); // Drop transaction before creating tables

        if !has_table_stats {
            // Parse and execute CREATE TABLE for table stats
            self.execute_stats_sql(CREATE_TABLE_STATS_SQL)?;
        }

        if !has_column_stats {
            // Parse and execute CREATE TABLE for column stats
            self.execute_stats_sql(CREATE_COLUMN_STATS_SQL)?;
        }

        Ok(())
    }

    /// Execute SQL statement (helper for system table creation and stats updates)
    fn execute_stats_sql(&self, sql: &str) -> Result<()> {
        let mut parser = crate::parser::Parser::new(sql);
        let program = parser
            .parse_program()
            .map_err(|e| Error::parse(e.to_string()))?;

        for stmt in &program.statements {
            let ctx = ExecutionContext::default();
            self.execute_statement(stmt, &ctx)?;
        }

        Ok(())
    }

    /// Analyze a single table and update statistics
    fn analyze_table(&self, tx: &mut dyn Transaction, table_name: &str) -> Result<()> {
        let table = tx.get_table(table_name)?;
        let schema = table.schema().clone();

        // Get row count
        let row_count = table.row_count();

        // Collect all rows for zone map building (zone maps need complete data)
        let all_rows = table.collect_all_rows(None)?;

        // Build zone maps from all rows
        let zone_maps = self.build_zone_maps(&all_rows, &schema);

        // Store zone maps in the table
        table.set_zone_maps(zone_maps);

        // For statistics, use sampling for large tables
        let rows = if all_rows.len() > DEFAULT_SAMPLE_SIZE {
            // Sample from collected rows for statistics
            self.sample_from_rows(all_rows.clone(), DEFAULT_SAMPLE_SIZE)
        } else {
            all_rows.clone()
        };

        let actual_row_count = rows.len();

        // Calculate average row size
        let total_size: usize = rows.iter().map(|r| self.estimate_row_size(r)).sum();
        let avg_row_size = if actual_row_count > 0 {
            total_size / actual_row_count
        } else {
            0
        };

        // Estimate page count (assuming 8KB pages)
        let page_count = (total_size / 8192).max(1);

        // Collect column statistics before dropping table
        let mut column_stats_list = Vec::new();
        for (col_idx, col_def) in schema.columns.iter().enumerate() {
            let col_stats = self.collect_column_stats(&rows, col_idx);
            column_stats_list.push((col_def.name.clone(), col_stats));
        }

        // Drop the table reference before using tx again
        drop(table);

        // Update table stats
        self.upsert_table_stats(
            table_name,
            row_count as i64,
            page_count as i64,
            avg_row_size as i64,
        )?;

        // Update column statistics
        for (col_name, col_stats) in column_stats_list {
            self.upsert_column_stats(
                table_name,
                &col_name,
                col_stats.0,          // null_count
                col_stats.1,          // distinct_count
                col_stats.2.as_ref(), // min_value
                col_stats.3.as_ref(), // max_value
                col_stats.4,          // avg_width
                col_stats.5.as_ref(), // histogram
            )?;
        }

        Ok(())
    }

    /// Build zone maps from rows
    ///
    /// Creates zone maps with min/max statistics per segment for each column.
    /// This enables micro-partition pruning during scans.
    fn build_zone_maps(
        &self,
        rows: &[Row],
        schema: &crate::core::Schema,
    ) -> crate::storage::mvcc::zonemap::TableZoneMap {
        let mut builder = ZoneMapBuilder::new(DEFAULT_SEGMENT_SIZE);

        for row in rows {
            let mut col_values = Vec::with_capacity(schema.columns.len());
            for (col_idx, col_def) in schema.columns.iter().enumerate() {
                if let Some(value) = row.get(col_idx) {
                    col_values.push((col_def.name.clone(), value.clone()));
                }
            }
            builder.add_row(&col_values);
        }

        builder.build()
    }

    /// Sample from already-collected rows using reservoir sampling
    fn sample_from_rows(&self, rows: Vec<Row>, sample_size: usize) -> Vec<Row> {
        if rows.len() <= sample_size {
            return rows;
        }

        let mut rng = rand::rng();
        let mut reservoir: Vec<Row> = rows[..sample_size].to_vec();

        // Reservoir sampling
        for (i, row) in rows.into_iter().enumerate().skip(sample_size) {
            let j = rng.random_range(0..=i);
            if j < sample_size {
                reservoir[j] = row;
            }
        }

        reservoir
    }

    /// Estimate size of a row in bytes
    fn estimate_row_size(&self, row: &Row) -> usize {
        row.iter().map(|v| self.estimate_value_size(v)).sum()
    }

    /// Estimate size of a value in bytes
    fn estimate_value_size(&self, value: &Value) -> usize {
        match value {
            Value::Null(_) => 1,
            Value::Boolean(_) => 1,
            Value::Integer(_) => 8,
            Value::Float(_) => 8,
            Value::Text(s) => s.len() + 4, // string + length prefix
            Value::Timestamp(_) => 8,
            Value::Json(s) => s.len() + 4,
        }
    }

    /// Collect statistics for a single column
    /// Returns (null_count, distinct_count, min_value, max_value, avg_width, histogram)
    fn collect_column_stats(
        &self,
        rows: &[Row],
        col_idx: usize,
    ) -> (
        i64,
        i64,
        Option<Value>,
        Option<Value>,
        i64,
        Option<Histogram>,
    ) {
        let mut null_count = 0i64;
        let mut distinct_values: FxHashSet<String> = FxHashSet::default();
        let mut min_value: Option<Value> = None;
        let mut max_value: Option<Value> = None;
        let mut total_width = 0usize;
        let mut values_for_histogram: Vec<Value> = Vec::new();

        for row in rows {
            if let Some(value) = row.get(col_idx) {
                if value.is_null() {
                    null_count += 1;
                } else {
                    // Track distinct values using string representation with FxHash for speed
                    distinct_values.insert(format!("{}", value));

                    // Track min/max
                    let should_update_min = match &min_value {
                        None => true,
                        Some(min) => value < min,
                    };
                    if should_update_min {
                        min_value = Some(value.clone());
                    }

                    let should_update_max = match &max_value {
                        None => true,
                        Some(max) => value > max,
                    };
                    if should_update_max {
                        max_value = Some(value.clone());
                    }

                    // Collect values for histogram (only numeric types)
                    if matches!(value, Value::Integer(_) | Value::Float(_)) {
                        values_for_histogram.push(value.clone());
                    }
                }

                total_width += self.estimate_value_size(value);
            } else {
                null_count += 1;
            }
        }

        let distinct_count = distinct_values.len() as i64;
        let avg_width = if !rows.is_empty() {
            (total_width / rows.len()) as i64
        } else {
            0
        };

        // Build histogram for numeric columns with enough data
        let histogram = if values_for_histogram.len() >= DEFAULT_HISTOGRAM_BUCKETS * 2 {
            // Sort values for equi-depth histogram construction
            values_for_histogram
                .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            Histogram::from_sorted_values(&values_for_histogram, DEFAULT_HISTOGRAM_BUCKETS)
        } else {
            None
        };

        (
            null_count,
            distinct_count,
            min_value,
            max_value,
            avg_width,
            histogram,
        )
    }

    /// Escape a string for safe use in SQL string literals
    ///
    /// This prevents SQL injection by escaping single quotes and backslashes.
    fn escape_sql_string(s: &str) -> String {
        s.replace('\\', "\\\\").replace('\'', "''")
    }

    /// Insert or update table statistics
    fn upsert_table_stats(
        &self,
        table_name: &str,
        row_count: i64,
        page_count: i64,
        avg_row_size: i64,
    ) -> Result<()> {
        let escaped_table = Self::escape_sql_string(table_name);

        // Delete existing stats for this table
        let delete_sql = format!(
            "DELETE FROM {} WHERE table_name = '{}'",
            SYS_TABLE_STATS, escaped_table
        );
        self.execute_stats_sql(&delete_sql)?;

        // Insert new stats
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);

        let insert_sql = format!(
            "INSERT INTO {} (table_name, row_count, page_count, avg_row_size, last_analyzed) VALUES ('{}', {}, {}, {}, {})",
            SYS_TABLE_STATS, escaped_table, row_count, page_count, avg_row_size, now
        );
        self.execute_stats_sql(&insert_sql)?;

        Ok(())
    }

    /// Insert or update column statistics
    #[allow(clippy::too_many_arguments)]
    fn upsert_column_stats(
        &self,
        table_name: &str,
        column_name: &str,
        null_count: i64,
        distinct_count: i64,
        min_value: Option<&Value>,
        max_value: Option<&Value>,
        avg_width: i64,
        histogram: Option<&Histogram>,
    ) -> Result<()> {
        let escaped_table = Self::escape_sql_string(table_name);
        let escaped_column = Self::escape_sql_string(column_name);

        // Delete existing stats for this column
        let delete_sql = format!(
            "DELETE FROM {} WHERE table_name = '{}' AND column_name = '{}'",
            SYS_COLUMN_STATS, escaped_table, escaped_column
        );
        self.execute_stats_sql(&delete_sql)?;

        // Format min/max values as strings (also escaped)
        let min_str = min_value
            .map(|v| format!("'{}'", Self::escape_sql_string(&v.to_string())))
            .unwrap_or_else(|| "NULL".to_string());
        let max_str = max_value
            .map(|v| format!("'{}'", Self::escape_sql_string(&v.to_string())))
            .unwrap_or_else(|| "NULL".to_string());

        // Format histogram as JSON string
        let histogram_str = histogram
            .map(|h| format!("'{}'", Self::escape_sql_string(&h.to_json())))
            .unwrap_or_else(|| "NULL".to_string());

        // Insert new stats
        let insert_sql = format!(
            "INSERT INTO {} (table_name, column_name, null_count, distinct_count, min_value, max_value, avg_width, histogram) VALUES ('{}', '{}', {}, {}, {}, {}, {}, {})",
            SYS_COLUMN_STATS, escaped_table, escaped_column, null_count, distinct_count, min_str, max_str, avg_width, histogram_str
        );
        self.execute_stats_sql(&insert_sql)?;

        Ok(())
    }
}
