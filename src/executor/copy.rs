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

//! COPY FROM Statement Execution
//!
//! Bulk imports data from CSV or JSON files, bypassing per-row SQL parsing
//! for significantly faster loading compared to individual INSERT statements.

use crate::common::{CompactArc, SmartString};
use crate::core::{DataType, Error, Result, Row, Schema, Value};
use crate::parser::ast::{CopyFormat, CopyStatement};
use crate::storage::traits::{Engine, QueryResult, WriteTable};

use super::context::{
    invalidate_in_subquery_cache_for_table, invalidate_scalar_subquery_cache_for_table,
    invalidate_semi_join_cache_for_table, ExecutionContext,
};
use super::result::ExecResult;
use super::Executor;

/// Parse a CSV field directly into a Value for the target type.
/// Avoids the intermediate Value::text() + coerce_to_type() allocation path.
#[inline]
fn parse_field(field: &str, target_type: DataType, col_name: &str) -> Result<Value> {
    match target_type {
        DataType::Integer => field.parse::<i64>().map(Value::Integer).map_err(|_| {
            Error::Type(format!(
                "cannot convert value '{}' to INTEGER for column '{}'",
                field, col_name
            ))
        }),
        DataType::Float => field.parse::<f64>().map(Value::Float).map_err(|_| {
            Error::Type(format!(
                "cannot convert value '{}' to FLOAT for column '{}'",
                field, col_name
            ))
        }),
        DataType::Boolean => {
            if field.eq_ignore_ascii_case("true")
                || field.eq_ignore_ascii_case("t")
                || field.eq_ignore_ascii_case("yes")
                || field.eq_ignore_ascii_case("y")
                || field == "1"
            {
                Ok(Value::Boolean(true))
            } else if field.eq_ignore_ascii_case("false")
                || field.eq_ignore_ascii_case("f")
                || field.eq_ignore_ascii_case("no")
                || field.eq_ignore_ascii_case("n")
                || field == "0"
            {
                Ok(Value::Boolean(false))
            } else {
                Err(Error::Type(format!(
                    "cannot convert value '{}' to BOOLEAN for column '{}'",
                    field, col_name
                )))
            }
        }
        DataType::Timestamp => crate::core::parse_timestamp(field)
            .map(Value::Timestamp)
            .map_err(|_| {
                Error::Type(format!(
                    "cannot convert value '{}' to TIMESTAMP for column '{}'",
                    field, col_name
                ))
            }),
        DataType::Text => {
            // SmartString::new takes &str: inlines <=15 bytes (0 allocs),
            // heap-allocates only for longer strings (1 alloc for Arc)
            Ok(Value::Text(SmartString::new(field)))
        }
        DataType::Json => {
            // Validate JSON before storing
            if serde_json::from_str::<serde_json::Value>(field).is_ok() {
                Ok(Value::json(field))
            } else {
                Err(Error::Type(format!(
                    "cannot convert value '{}' to JSON for column '{}'",
                    field, col_name
                )))
            }
        }
        _ => {
            // Fallback: go through Value::text + coerce for uncommon types (Vector, etc.)
            let text_val = Value::text(field);
            let coerced = text_val.coerce_to_type(target_type);
            if !text_val.is_null() && coerced.is_null() {
                return Err(Error::Type(format!(
                    "cannot convert value '{}' to {:?} for column '{}'",
                    field, target_type, col_name
                )));
            }
            Ok(coerced)
        }
    }
}

impl Executor {
    /// Execute a COPY FROM statement
    pub(crate) fn execute_copy(
        &self,
        stmt: &CopyStatement,
        _ctx: &ExecutionContext,
    ) -> Result<Box<dyn QueryResult>> {
        let table_name = &stmt.table_name.value_lower;

        // COPY is not allowed inside explicit transactions (like PRAGMA CHECKPOINT)
        {
            let active_tx = self.active_transaction.lock().unwrap();
            if active_tx.is_some() {
                return Err(Error::InvalidArgument(
                    "COPY FROM cannot be used inside an explicit transaction".to_string(),
                ));
            }
        }

        // Create a standalone auto-commit transaction
        let mut tx = self.engine.begin_writable_transaction_internal()?;
        let mut table = tx.get_table(table_name)?;

        // Pre-compute schema information
        let schema = table.schema();
        let schema_column_count = schema.columns.len();

        let column_indices: Vec<usize>;
        let column_types: Vec<DataType>;
        let column_names: Vec<String>;
        let all_column_types: Vec<DataType> = schema.columns.iter().map(|c| c.data_type).collect();
        let all_vector_dims: Vec<u16> =
            schema.columns.iter().map(|c| c.vector_dimensions).collect();
        let default_exprs: Vec<Option<String>> = schema
            .columns
            .iter()
            .map(|c| c.default_expr.clone())
            .collect();
        let check_exprs: Vec<(usize, String, String)> = schema
            .columns
            .iter()
            .enumerate()
            .filter_map(|(idx, c)| {
                c.check_expr
                    .as_ref()
                    .map(|expr| (idx, c.name.clone(), expr.clone()))
            })
            .collect();

        if stmt.columns.is_empty() {
            column_indices = (0..schema_column_count).collect();
            column_types = all_column_types.clone();
            column_names = schema.column_names_owned().to_vec();
        } else {
            let col_map = schema.column_index_map();
            column_indices = stmt
                .columns
                .iter()
                .map(|id| {
                    col_map
                        .get(id.value_lower.as_str())
                        .copied()
                        .ok_or_else(|| Error::ColumnNotFound(id.value.to_string()))
                })
                .collect::<Result<Vec<_>>>()?;
            column_types = column_indices
                .iter()
                .map(|&idx| schema.columns[idx].data_type)
                .collect();
            column_names = column_indices
                .iter()
                .map(|&idx| schema.columns[idx].name.clone())
                .collect();
        }

        // Pre-compute FK info
        let fk_schema: Option<CompactArc<Schema>> = if !table.schema().foreign_keys.is_empty() {
            Some(self.engine.get_table_schema(table_name)?)
        } else {
            None
        };

        let rows_affected = match stmt.format {
            CopyFormat::Csv => self.copy_from_csv(
                stmt,
                &mut table,
                &column_indices,
                &column_names,
                &all_column_types,
                &all_vector_dims,
                &default_exprs,
                &check_exprs,
                &fk_schema,
                schema_column_count,
            )?,
            CopyFormat::Json => self.copy_from_json(
                stmt,
                &mut table,
                &column_indices,
                &column_types,
                &column_names,
                &all_column_types,
                &all_vector_dims,
                &default_exprs,
                &check_exprs,
                &fk_schema,
                schema_column_count,
            )?,
        };

        // Invalidate caches and commit
        if rows_affected > 0 {
            self.semantic_cache.invalidate_table(table_name);
            invalidate_semi_join_cache_for_table(table_name);
            invalidate_scalar_subquery_cache_for_table(table_name);
            invalidate_in_subquery_cache_for_table(table_name);
        }

        tx.commit()?;

        Ok(Box::new(ExecResult::with_rows_affected(rows_affected)))
    }

    /// Import rows from a CSV file
    #[allow(clippy::too_many_arguments)]
    fn copy_from_csv(
        &self,
        stmt: &CopyStatement,
        table: &mut Box<dyn WriteTable>,
        column_indices: &[usize],
        column_names: &[String],
        all_column_types: &[DataType],
        all_vector_dims: &[u16],
        default_exprs: &[Option<String>],
        check_exprs: &[(usize, String, String)],
        fk_schema: &Option<CompactArc<Schema>>,
        schema_column_count: usize,
    ) -> Result<i64> {
        let file = std::fs::File::open(&stmt.file_path).map_err(|e| {
            Error::InvalidArgument(format!("cannot open file '{}': {}", stmt.file_path, e))
        })?;

        let mut reader = csv::ReaderBuilder::new()
            .has_headers(stmt.header)
            .delimiter(stmt.delimiter)
            .from_reader(std::io::BufReader::new(file));

        // If header is present and columns are not specified, try to map header names to columns
        let field_to_col: Option<Vec<usize>> = if stmt.header && stmt.columns.is_empty() {
            let headers = reader
                .headers()
                .map_err(|e| Error::InvalidArgument(format!("cannot read CSV headers: {}", e)))?;
            let col_map = {
                // Re-fetch schema column_index_map
                let schema = table.schema();
                let m = schema.column_index_map();
                m.clone()
            };
            let mut mapping = Vec::with_capacity(headers.len());
            for h in headers.iter() {
                let lower = h.to_lowercase();
                if let Some(&idx) = col_map.get(lower.as_str()) {
                    mapping.push(idx);
                } else {
                    return Err(Error::ColumnNotFound(h.to_string()));
                }
            }
            Some(mapping)
        } else {
            None
        };

        let null_str = stmt.null_string.as_deref().unwrap_or("");
        let mut rows_affected = 0i64;

        let default_row =
            build_default_row(self, default_exprs, all_column_types, schema_column_count);

        for result in reader.records() {
            let record = result.map_err(|e| {
                Error::InvalidArgument(format!(
                    "CSV parse error at row {}: {}",
                    rows_affected + 1,
                    e
                ))
            })?;

            let effective_indices = field_to_col.as_deref().unwrap_or(column_indices);

            if record.len() != effective_indices.len() {
                return Err(Error::InvalidArgument(format!(
                    "CSV row {} has {} fields but expected {}",
                    rows_affected + 1,
                    record.len(),
                    effective_indices.len()
                )));
            }

            // Clone defaults template (cheap: Null/Integer/Bool are stack-only)
            let mut row_values = default_row.clone();

            // Parse CSV fields directly into target types (no intermediate Value::text allocation)
            for (i, field) in record.iter().enumerate() {
                let col_idx = effective_indices[i];

                if field == null_str {
                    row_values[col_idx] = Value::null_unknown();
                    continue;
                }

                let target_type = all_column_types[col_idx];
                let col_name = column_names.get(i).map(|s| s.as_str()).unwrap_or("?");
                let value = parse_field(field, target_type, col_name)?;
                validate_vector_dims(&value, target_type, all_vector_dims[col_idx])?;
                row_values[col_idx] = value;
            }

            // Validate CHECK constraints
            for (col_idx, col_name, check_expr) in check_exprs {
                let col_type = all_column_types[*col_idx];
                self.validate_check_constraint(
                    check_expr,
                    col_name,
                    &row_values[*col_idx],
                    col_type,
                )?;
            }

            let row = Row::from_values(row_values);

            // FK parent validation
            if let Some(ref fks) = fk_schema {
                super::foreign_key::check_parent_exists(&self.engine, table.txn_id(), fks, &row)?;
            }

            table.insert_discard(row)?;
            rows_affected += 1;
        }

        Ok(rows_affected)
    }

    /// Import rows from a JSON file (JSON Lines or JSON array)
    #[allow(clippy::too_many_arguments)]
    fn copy_from_json(
        &self,
        stmt: &CopyStatement,
        table: &mut Box<dyn WriteTable>,
        column_indices: &[usize],
        column_types: &[DataType],
        column_names: &[String],
        all_column_types: &[DataType],
        all_vector_dims: &[u16],
        default_exprs: &[Option<String>],
        check_exprs: &[(usize, String, String)],
        fk_schema: &Option<CompactArc<Schema>>,
        schema_column_count: usize,
    ) -> Result<i64> {
        let null_str = stmt.null_string.as_deref();
        let use_columns = !stmt.columns.is_empty();

        // Pre-compute default row template once
        let default_row =
            build_default_row(self, default_exprs, all_column_types, schema_column_count);

        // Pre-build lowercase column name map for case-insensitive JSON key matching
        let col_name_lower_map: Vec<(String, usize)> = if use_columns {
            stmt.columns
                .iter()
                .enumerate()
                .map(|(i, _)| (column_names[i].to_lowercase(), column_indices[i]))
                .collect()
        } else {
            let schema = table.schema();
            schema
                .columns
                .iter()
                .enumerate()
                .map(|(idx, c)| (c.name.to_lowercase(), idx))
                .collect()
        };

        // Stream JSON objects one at a time with O(object) memory.
        // For JSON arrays, we strip `[`, `]`, and `,` between objects so
        // StreamDeserializer sees a sequence of top-level values.
        // For JSON Lines, objects are already top-level.
        let file = std::fs::File::open(&stmt.file_path).map_err(|e| {
            Error::InvalidArgument(format!("cannot open file '{}': {}", stmt.file_path, e))
        })?;
        let reader = JsonArrayStripper::new(std::io::BufReader::new(file));
        let stream = serde_json::Deserializer::from_reader(reader).into_iter::<serde_json::Value>();

        let mut rows_affected = 0i64;

        for (idx, result) in stream.enumerate() {
            let item = result.map_err(|e| {
                Error::InvalidArgument(format!("JSON parse error at object {}: {}", idx + 1, e))
            })?;

            let obj = item.as_object().ok_or_else(|| {
                Error::InvalidArgument(format!("JSON item {} is not an object", idx + 1))
            })?;
            self.insert_json_row(
                obj,
                table,
                &default_row,
                &col_name_lower_map,
                use_columns,
                column_types,
                all_column_types,
                all_vector_dims,
                null_str,
                check_exprs,
                fk_schema,
            )?;
            rows_affected += 1;
        }

        Ok(rows_affected)
    }

    /// Insert a single JSON object as a row. Shared by array and lines paths.
    #[allow(clippy::too_many_arguments)]
    fn insert_json_row(
        &self,
        obj: &serde_json::Map<String, serde_json::Value>,
        table: &mut Box<dyn WriteTable>,
        default_row: &[Value],
        col_name_lower_map: &[(String, usize)],
        use_columns: bool,
        column_types: &[DataType],
        all_column_types: &[DataType],
        all_vector_dims: &[u16],
        null_str: Option<&str>,
        check_exprs: &[(usize, String, String)],
        fk_schema: &Option<CompactArc<Schema>>,
    ) -> Result<()> {
        let mut row_values = default_row.to_vec();

        if use_columns {
            // Only import specified columns
            for (i, (lower_name, col_idx)) in col_name_lower_map.iter().enumerate() {
                let target_type = column_types[i];

                // Case-insensitive key lookup
                let json_val = find_json_key_ci(obj, lower_name);

                if let Some(v) = json_val {
                    let value = json_value_to_stoolap(v, target_type, lower_name, null_str)?;
                    validate_vector_dims(&value, target_type, all_vector_dims[*col_idx])?;
                    row_values[*col_idx] = value;
                }
                // Missing key: keep default/null
            }
        } else {
            // Import all columns by matching JSON keys case-insensitively
            for (key, json_val) in obj {
                let lower_key = key.to_lowercase();
                // Find matching column
                if let Some(&(_, col_idx)) = col_name_lower_map
                    .iter()
                    .find(|(name, _)| *name == lower_key)
                {
                    let target_type = all_column_types[col_idx];
                    let value = json_value_to_stoolap(json_val, target_type, &lower_key, null_str)?;
                    validate_vector_dims(&value, target_type, all_vector_dims[col_idx])?;
                    row_values[col_idx] = value;
                }
                // Unknown keys silently ignored
            }
        }

        for (col_idx, col_name, check_expr) in check_exprs {
            let col_type = all_column_types[*col_idx];
            self.validate_check_constraint(check_expr, col_name, &row_values[*col_idx], col_type)?;
        }

        let row = Row::from_values(row_values);

        if let Some(ref fks) = fk_schema {
            super::foreign_key::check_parent_exists(&self.engine, table.txn_id(), fks, &row)?;
        }

        table.insert_discard(row)?;
        Ok(())
    }
}

/// Build the default row template from schema default expressions.
fn build_default_row(
    executor: &Executor,
    default_exprs: &[Option<String>],
    all_column_types: &[DataType],
    schema_column_count: usize,
) -> Vec<Value> {
    let mut row = Vec::with_capacity(schema_column_count);
    for i in 0..schema_column_count {
        if let Some(ref default_expr) = default_exprs[i] {
            let default_type = all_column_types[i];
            match executor.evaluate_default_expr(default_expr, default_type) {
                Ok(val) => row.push(val),
                Err(_) => row.push(Value::null_unknown()),
            }
        } else {
            row.push(Value::null_unknown());
        }
    }
    row
}

/// Validate vector dimensions if the column is a VECTOR type.
#[inline]
fn validate_vector_dims(value: &Value, target_type: DataType, expected_dims: u16) -> Result<()> {
    if target_type == DataType::Vector && expected_dims > 0 {
        if let Value::Extension(data) = value {
            if data.first() == Some(&(DataType::Vector as u8)) {
                let got_dim = u16::try_from((data.len() - 1) / 4).unwrap_or(u16::MAX);
                if got_dim != expected_dims {
                    return Err(Error::VectorDimensionMismatch {
                        expected: expected_dims,
                        got: got_dim,
                    });
                }
            }
        }
    }
    Ok(())
}

/// Case-insensitive JSON key lookup using Unicode-aware lowercasing.
#[inline]
fn find_json_key_ci<'a>(
    obj: &'a serde_json::Map<String, serde_json::Value>,
    lower_name: &str,
) -> Option<&'a serde_json::Value> {
    // Fast path: try exact match first
    if let Some(v) = obj.get(lower_name) {
        return Some(v);
    }
    // Slow path: Unicode-aware case-insensitive scan
    for (k, v) in obj {
        if k.to_lowercase() == lower_name {
            return Some(v);
        }
    }
    None
}

/// Convert a serde_json::Value to a stoolap Value with type coercion.
/// Returns an error if a non-null value silently becomes null during coercion.
fn json_value_to_stoolap(
    v: &serde_json::Value,
    target_type: DataType,
    col_name: &str,
    null_str: Option<&str>,
) -> Result<Value> {
    let val = match v {
        serde_json::Value::Null => return Ok(Value::null_unknown()),
        serde_json::Value::Bool(b) => Value::Boolean(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Integer(i)
            } else if let Some(f) = n.as_f64() {
                Value::Float(f)
            } else {
                Value::text(n.to_string())
            }
        }
        serde_json::Value::String(s) => {
            if let Some(ns) = null_str {
                if s == ns {
                    return Ok(Value::null_unknown());
                }
            }
            Value::text(s)
        }
        serde_json::Value::Object(_) | serde_json::Value::Array(_) => Value::text(v.to_string()),
    };

    let coerced = val.coerce_to_type(target_type);
    // P1 fix: detect silent coercion failure (non-null -> null)
    if !val.is_null() && coerced.is_null() {
        return Err(Error::Type(format!(
            "cannot convert value '{}' to {:?} for column '{}'",
            val, target_type, col_name
        )));
    }
    Ok(coerced)
}

/// A Read adapter that transforms a JSON array `[{...},{...}]` into a stream
/// of top-level objects `{...} {...}` by replacing `[`, `]`, and inter-element
/// commas with whitespace. For JSON Lines input (no leading `[`), bytes pass
/// through unchanged. This lets `serde_json::StreamDeserializer` yield one
/// object at a time with O(object) memory for both formats.
struct JsonArrayStripper<R> {
    inner: R,
    is_array: bool,
    /// Nesting depth inside JSON values. 0 = between top-level values.
    depth: u32,
    /// True while inside a JSON string literal (skip structural chars).
    in_string: bool,
    /// Previous byte was `\` inside a string (skip escaped quotes).
    escape: bool,
    /// Saved first non-whitespace byte for non-array input (needs replay).
    pending: Option<u8>,
}

impl<R: std::io::Read> JsonArrayStripper<R> {
    fn new(mut inner: R) -> Self {
        // Peek at first non-whitespace byte to detect array format
        let mut first = [0u8; 1];
        let (is_array, pending) = loop {
            match inner.read(&mut first) {
                Ok(1) if first[0].is_ascii_whitespace() => continue,
                Ok(1) if first[0] == b'[' => break (true, None), // `[` consumed, don't replay
                Ok(1) => break (false, Some(first[0])),          // save for replay
                _ => break (false, None),                        // empty file
            }
        };

        JsonArrayStripper {
            inner,
            is_array,
            depth: 0,
            in_string: false,
            escape: false,
            pending,
        }
    }
}

impl<R: std::io::Read> std::io::Read for JsonArrayStripper<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        // Replay the saved first byte if present
        if let Some(b) = self.pending.take() {
            buf[0] = b;
            if buf.len() == 1 {
                return Ok(1);
            }
            let n = self.inner.read(&mut buf[1..])?;
            let total = 1 + n;
            if self.is_array {
                self.strip_array_syntax(buf, total);
            }
            return Ok(total);
        }

        let n = self.inner.read(buf)?;
        if self.is_array {
            self.strip_array_syntax(buf, n);
        }
        Ok(n)
    }
}

impl<R> JsonArrayStripper<R> {
    /// Replace outer-array `[`, `]`, and inter-element `,` with spaces.
    /// Tracks nesting depth and string literals to avoid touching structural
    /// characters inside JSON values.
    fn strip_array_syntax(&mut self, buf: &mut [u8], len: usize) {
        for b in &mut buf[..len] {
            if self.in_string {
                if self.escape {
                    self.escape = false;
                } else if *b == b'\\' {
                    self.escape = true;
                } else if *b == b'"' {
                    self.in_string = false;
                }
                continue;
            }

            match *b {
                b'"' => {
                    self.in_string = true;
                }
                b'{' | b'[' => {
                    self.depth += 1;
                }
                b'}' | b']' => {
                    if self.depth > 0 {
                        self.depth -= 1;
                    } else {
                        // Closing `]` of the outer array
                        *b = b' ';
                    }
                }
                b',' if self.depth == 0 => {
                    // Comma between top-level array elements
                    *b = b' ';
                }
                _ => {}
            }
        }
    }
}
