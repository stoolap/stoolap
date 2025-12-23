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

//! Streaming GROUP BY operator for lazy CTE evaluation.
//!
//! This operator produces aggregated groups one at a time, enabling
//! early termination when used with LIMIT or in a JOIN context.
//!
//! Key features:
//! - Uses B-tree index for ordered group access
//! - Produces one group per `next()` call (lazy evaluation)
//! - Applies HAVING filter inline
//! - Enables early termination in CTE + JOIN scenarios

use crate::core::{Result, Row, Value};
use crate::storage::expression::logical::ConstBoolExpr;
use crate::storage::traits::Table;

use super::super::operator::{ColumnInfo, Operator, RowRef};

/// Streaming aggregate type
#[derive(Clone, Copy)]
pub enum StreamingAgg {
    /// COUNT(*) or COUNT(column)
    Count,
    /// SUM(column) with column index
    Sum(usize),
    /// AVG(column) with column index
    Avg(usize),
    /// MIN(column) with column index
    Min(usize),
    /// MAX(column) with column index
    Max(usize),
}

/// HAVING filter specification
#[derive(Clone)]
pub struct HavingFilter {
    /// Index into the aggregates list
    pub agg_idx: usize,
    /// Threshold value
    pub threshold: f64,
    /// If true, use >= comparison; if false, use > comparison
    pub inclusive: bool,
}

/// Streaming GROUP BY operator for lazy aggregation.
///
/// This operator wraps B-tree grouped row IDs and produces aggregated
/// groups one at a time via the Volcano iterator pattern.
pub struct StreamingGroupByOperator {
    /// Table for fetching rows
    table: Box<dyn Table>,
    /// Grouped row IDs from B-tree index (value -> row_ids)
    grouped_row_ids: Vec<(Value, Vec<i64>)>,
    /// Current position in grouped_row_ids
    current_idx: usize,
    /// Aggregate functions to compute
    aggregates: Vec<StreamingAgg>,
    /// Optional HAVING filter
    having: Option<HavingFilter>,
    /// Output schema
    schema: Vec<ColumnInfo>,
    /// Whether the operator is open
    is_open: bool,
}

impl StreamingGroupByOperator {
    /// Create a new streaming GROUP BY operator.
    ///
    /// # Arguments
    /// * `table` - Table to fetch rows from
    /// * `grouped_row_ids` - Pre-grouped row IDs from B-tree index (in sorted order)
    /// * `aggregates` - List of aggregates to compute
    /// * `having` - Optional HAVING filter
    /// * `output_columns` - Names for output columns
    pub fn new(
        table: Box<dyn Table>,
        grouped_row_ids: Vec<(Value, Vec<i64>)>,
        aggregates: Vec<StreamingAgg>,
        having: Option<HavingFilter>,
        output_columns: Vec<String>,
    ) -> Self {
        let schema = output_columns.into_iter().map(ColumnInfo::new).collect();

        Self {
            table,
            grouped_row_ids,
            current_idx: 0,
            aggregates,
            having,
            schema,
            is_open: false,
        }
    }

    /// Check if we need to fetch rows (SUM, AVG, MIN, MAX require row data)
    fn needs_row_fetch(&self) -> bool {
        self.aggregates.iter().any(|a| {
            matches!(
                a,
                StreamingAgg::Sum(_)
                    | StreamingAgg::Avg(_)
                    | StreamingAgg::Min(_)
                    | StreamingAgg::Max(_)
            )
        })
    }

    /// Process a single group and return the aggregated row if it passes HAVING
    fn process_group(&self, group_value: &Value, row_ids: &[i64]) -> Result<Option<Row>> {
        let num_aggs = self.aggregates.len();
        let mut agg_values = vec![0.0f64; num_aggs]; // For SUM/AVG accumulation
        let mut agg_has_value = vec![false; num_aggs];
        let mut counts = vec![0i64; num_aggs]; // For COUNT and AVG divisor
        let mut min_values: Vec<Option<Value>> = vec![None; num_aggs]; // For MIN
        let mut max_values: Vec<Option<Value>> = vec![None; num_aggs]; // For MAX
        let row_count = row_ids.len() as i64;

        if self.needs_row_fetch() {
            // Need to fetch rows for SUM, AVG, MIN, MAX
            let true_expr = ConstBoolExpr::true_expr();
            let fetched_rows = self.table.fetch_rows_by_ids(row_ids, &true_expr);

            for (_row_id, row) in &fetched_rows {
                for (i, agg) in self.aggregates.iter().enumerate() {
                    match agg {
                        StreamingAgg::Count => {
                            counts[i] += 1;
                        }
                        StreamingAgg::Sum(col_idx) | StreamingAgg::Avg(col_idx) => {
                            if let Some(value) = row.get(*col_idx) {
                                match value {
                                    Value::Integer(v) => {
                                        agg_values[i] += *v as f64;
                                        agg_has_value[i] = true;
                                        counts[i] += 1; // Track count for AVG
                                    }
                                    Value::Float(v) => {
                                        agg_values[i] += v;
                                        agg_has_value[i] = true;
                                        counts[i] += 1;
                                    }
                                    _ => {}
                                }
                            }
                        }
                        StreamingAgg::Min(col_idx) => {
                            if let Some(value) = row.get(*col_idx) {
                                if !value.is_null() {
                                    match &min_values[i] {
                                        None => {
                                            min_values[i] = Some(value.clone());
                                        }
                                        Some(current_min) => {
                                            if value < current_min {
                                                min_values[i] = Some(value.clone());
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        StreamingAgg::Max(col_idx) => {
                            if let Some(value) = row.get(*col_idx) {
                                if !value.is_null() {
                                    match &max_values[i] {
                                        None => {
                                            max_values[i] = Some(value.clone());
                                        }
                                        Some(current_max) => {
                                            if value > current_max {
                                                max_values[i] = Some(value.clone());
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else {
            // Fast path: All aggregates are COUNT, no row fetch needed
            for (i, agg) in self.aggregates.iter().enumerate() {
                if matches!(agg, StreamingAgg::Count) {
                    counts[i] = row_count;
                }
            }
        }

        // Apply HAVING filter
        if let Some(ref having) = self.having {
            let agg_val = match self.aggregates[having.agg_idx] {
                StreamingAgg::Count => Some(counts[having.agg_idx] as f64),
                StreamingAgg::Sum(_) => {
                    if agg_has_value[having.agg_idx] {
                        Some(agg_values[having.agg_idx])
                    } else {
                        None // NULL doesn't pass HAVING
                    }
                }
                StreamingAgg::Avg(_) => {
                    if counts[having.agg_idx] > 0 {
                        Some(agg_values[having.agg_idx] / counts[having.agg_idx] as f64)
                    } else {
                        None
                    }
                }
                StreamingAgg::Min(_) => min_values[having.agg_idx]
                    .as_ref()
                    .and_then(|v| v.as_float64()),
                StreamingAgg::Max(_) => max_values[having.agg_idx]
                    .as_ref()
                    .and_then(|v| v.as_float64()),
            };

            let Some(val) = agg_val else {
                return Ok(None); // NULL doesn't pass HAVING
            };

            let passes = if having.inclusive {
                val >= having.threshold
            } else {
                val > having.threshold
            };

            if !passes {
                return Ok(None);
            }
        }

        // Build result row: [group_value, agg1, agg2, ...]
        let mut values = Vec::with_capacity(1 + num_aggs);
        values.push(group_value.clone());

        for (i, agg) in self.aggregates.iter().enumerate() {
            let value = match agg {
                StreamingAgg::Count => Value::Integer(counts[i]),
                StreamingAgg::Sum(_) => {
                    if agg_has_value[i] {
                        Value::Float(agg_values[i])
                    } else {
                        Value::null_unknown()
                    }
                }
                StreamingAgg::Avg(_) => {
                    if counts[i] > 0 {
                        Value::Float(agg_values[i] / counts[i] as f64)
                    } else {
                        Value::null_unknown()
                    }
                }
                StreamingAgg::Min(_) => min_values[i].take().unwrap_or_else(Value::null_unknown),
                StreamingAgg::Max(_) => max_values[i].take().unwrap_or_else(Value::null_unknown),
            };
            values.push(value);
        }

        Ok(Some(Row::from_values(values)))
    }
}

impl Operator for StreamingGroupByOperator {
    fn open(&mut self) -> Result<()> {
        self.current_idx = 0;
        self.is_open = true;
        Ok(())
    }

    fn next(&mut self) -> Result<Option<RowRef>> {
        if !self.is_open {
            return Ok(None);
        }

        // Process groups until we find one that passes HAVING
        while self.current_idx < self.grouped_row_ids.len() {
            let (group_value, row_ids) = &self.grouped_row_ids[self.current_idx];
            self.current_idx += 1;

            if let Some(row) = self.process_group(group_value, row_ids)? {
                return Ok(Some(RowRef::owned(row)));
            }
            // Group didn't pass HAVING, try next
        }

        Ok(None) // Exhausted all groups
    }

    fn close(&mut self) -> Result<()> {
        self.is_open = false;
        Ok(())
    }

    fn schema(&self) -> &[ColumnInfo] {
        &self.schema
    }

    fn estimated_rows(&self) -> Option<usize> {
        Some(self.grouped_row_ids.len())
    }

    fn name(&self) -> &str {
        "StreamingGroupBy"
    }
}

#[cfg(test)]
mod tests {
    // Tests would go here once we have a test harness for Table mock
}
