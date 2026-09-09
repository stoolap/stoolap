// Copyright 2026 Stoolap Contributors
// Licensed under the Apache License, Version 2.0.

//! Bounded row materialization over an immutable statement-owned hot view.

use std::ops::Bound;
use std::sync::Arc;

use crate::common::CompactArc;
use crate::core::{Result, Row, Schema, Value};
use crate::storage::expression::Expression;
use crate::storage::traits::Scanner;

use super::version_store::CapturedHotView;

struct OrderedKey {
    key: Value,
    id: i64,
    ascending: bool,
}

impl PartialEq for OrderedKey {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}
impl Eq for OrderedKey {}
impl PartialOrd for OrderedKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for OrderedKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let order = self.key.cmp(&other.key);
        if self.ascending {
            order.then(self.id.cmp(&other.id))
        } else {
            order.reverse().then(other.id.cmp(&self.id))
        }
    }
}

/// Keep only offset+limit keys; materialize the selected immutable rows once.
pub(crate) fn sorted_hot_rows(
    view: &CapturedHotView,
    schema: &Schema,
    column: usize,
    ascending: bool,
    limit: usize,
    offset: usize,
    filter: Option<&dyn Expression>,
) -> Result<crate::core::RowVec> {
    let mut rows = crate::core::RowVec::new();
    if limit == 0 {
        return Ok(rows);
    }
    let needed = limit.saturating_add(offset);
    let default = schema
        .columns
        .get(column)
        .map(|column| {
            column
                .default_value
                .clone()
                .unwrap_or_else(|| Value::null(column.data_type))
        })
        .unwrap_or_else(Value::null_unknown);
    let mut heap = std::collections::BinaryHeap::<OrderedKey>::with_capacity(needed.min(256));
    let mut error = None;
    view.for_each_visible_until(|id, row| {
        let key = row.get(column).unwrap_or(&default);
        if heap.len() == needed {
            let worst = heap.peek().unwrap();
            let order = key.cmp(&worst.key);
            let order = if ascending {
                order.then(id.cmp(&worst.id))
            } else {
                order.reverse().then(worst.id.cmp(&id))
            };
            if !order.is_lt() {
                return true;
            }
        }
        if let Some(filter) = filter {
            let mut normalized;
            let row = if row.len() < schema.columns.len() {
                normalized = row.clone();
                for column in &schema.columns[row.len()..] {
                    normalized.push(
                        column
                            .default_value
                            .clone()
                            .unwrap_or_else(|| Value::null(column.data_type)),
                    );
                }
                &normalized
            } else {
                row
            };
            match filter.evaluate(row) {
                Ok(true) => (),
                Ok(false) => return true,
                Err(failure) => {
                    error = Some(failure);
                    return false;
                }
            }
        }
        if heap.len() == needed {
            heap.pop();
        }
        heap.push(OrderedKey {
            key: key.clone(),
            id,
            ascending,
        });
        true
    });
    if let Some(error) = error {
        return Err(error);
    }
    for selected in heap.into_sorted_vec().into_iter().skip(offset) {
        if let super::version_store::CapturedHotRow::Value(source) = view.row_state(selected.id) {
            let mut row = source.clone();
            if row.len() < schema.columns.len() {
                for column in &schema.columns[row.len()..] {
                    row.push(
                        column
                            .default_value
                            .clone()
                            .unwrap_or_else(|| Value::null(column.data_type)),
                    );
                }
            } else if row.len() > schema.columns.len() {
                row.truncate(schema.columns.len());
            }
            rows.push((selected.id, row));
        }
    }
    Ok(rows)
}

pub(crate) struct CapturedHotScanner {
    view: Option<Arc<CapturedHotView>>,
    schema: CompactArc<Schema>,
    filter: Option<Box<dyn Expression>>,
    projection: Vec<usize>,
    batch: Vec<(i64, Row)>,
    position: usize,
    last_id: Option<i64>,
    exhausted: bool,
    error: Option<crate::core::Error>,
    empty: Row,
}

impl CapturedHotScanner {
    const BATCH_ROWS: usize = 128;

    pub(crate) fn new(
        view: Arc<CapturedHotView>,
        schema: CompactArc<Schema>,
        columns: &[usize],
        filter: Option<&dyn Expression>,
    ) -> Self {
        let needs_projection = !columns.is_empty()
            && (columns.len() != schema.columns.len()
                || !columns.iter().enumerate().all(|(i, &column)| i == column));
        let batch_capacity = if view.is_empty() { 0 } else { Self::BATCH_ROWS };
        Self {
            view: Some(view),
            schema,
            filter: filter.map(Expression::clone_box),
            projection: if needs_projection {
                columns.to_vec()
            } else {
                Vec::new()
            },
            batch: Vec::with_capacity(batch_capacity),
            position: 0,
            last_id: None,
            exhausted: false,
            error: None,
            empty: Row::new(),
        }
    }

    fn refill(&mut self) {
        self.batch.clear();
        self.position = 0;
        let Some(view) = &self.view else {
            self.exhausted = true;
            return;
        };
        let start = self.last_id.map_or(Bound::Unbounded, Bound::Excluded);
        self.exhausted =
            view.for_each_visible_range_until((start, Bound::Unbounded), |id, source| {
                self.last_id = Some(id);
                let mut row = source.clone();
                if row.len() < self.schema.columns.len() {
                    for column in &self.schema.columns[row.len()..] {
                        row.push(
                            column
                                .default_value
                                .clone()
                                .unwrap_or_else(|| Value::null(column.data_type)),
                        );
                    }
                } else if row.len() > self.schema.columns.len() {
                    row.truncate(self.schema.columns.len());
                }
                let matched = match self
                    .filter
                    .as_ref()
                    .map(|filter| filter.evaluate(&row))
                    .transpose()
                {
                    Ok(matched) => matched.unwrap_or(true),
                    Err(error) => {
                        self.error = Some(error);
                        return false;
                    }
                };
                if matched {
                    if !self.projection.is_empty() {
                        row = Row::from_values(
                            self.projection
                                .iter()
                                .map(|&column| {
                                    row.get(column).cloned().unwrap_or_else(Value::null_unknown)
                                })
                                .collect(),
                        );
                    }
                    self.batch.push((id, row));
                }
                self.batch.len() < Self::BATCH_ROWS
            });
        if self.error.is_some() {
            self.batch.clear();
            self.exhausted = true;
        }
    }
}

impl Scanner for CapturedHotScanner {
    fn next(&mut self) -> bool {
        if self.position == self.batch.len() {
            if self.exhausted {
                let _ = self.close();
                return false;
            }
            self.refill();
            if self.batch.is_empty() {
                let _ = self.close();
                return false;
            }
        }
        self.position += 1;
        true
    }
    fn row(&self) -> &Row {
        self.position
            .checked_sub(1)
            .and_then(|index| self.batch.get(index))
            .map_or(&self.empty, |(_, row)| row)
    }
    fn take_row(&mut self) -> Row {
        self.position
            .checked_sub(1)
            .and_then(|index| self.batch.get_mut(index))
            .map_or_else(Row::new, |(_, row)| std::mem::take(row))
    }
    fn current_row_id(&self) -> i64 {
        self.position
            .checked_sub(1)
            .and_then(|index| self.batch.get(index))
            .map_or(0, |(id, _)| *id)
    }
    fn take_row_with_id(&mut self) -> (i64, Row) {
        (self.current_row_id(), self.take_row())
    }
    fn err(&self) -> Option<&crate::core::Error> {
        self.error.as_ref()
    }
    fn close(&mut self) -> Result<()> {
        self.view = None;
        self.filter = None;
        self.projection = Vec::new();
        self.batch = Vec::new();
        self.position = 0;
        self.exhausted = true;
        Ok(())
    }
}
