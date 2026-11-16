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

//! B-tree Index implementation for range queries and ordered access
//!
//! Provides a B-tree based index optimized for:
//! - Range queries (O(log n + k) with a [`ValueTree`])
//! - Ordered access (ORDER BY, MIN, MAX)
//! - Efficient column scans
//! - Aggregation operations (min/max from the tree's ends)
//!
//! B-tree index implementation for single-column indexing

use parking_lot::RwLock;
use std::cmp::Ordering;
use std::ops::Bound;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::Arc;

use super::value_tree::{Ids, ValueTree};
use crate::common::deadline::{Deadline, DeadlineSite, CHARGE_BEAT};
use crate::core::{DataType, Error, IndexEntry, IndexType, Operator, Result, Value};
use crate::storage::index::id_list::GroupIds;
use crate::storage::traits::Index;

/// The one NULL key: every NULL compares equal to it
const NULL: Value = Value::Null(DataType::Null);

/// B-tree index for efficient range queries and ordered access
///
/// This index stores column values in sorted order in a [`ValueTree`] for:
/// - O(log n + k) range queries (k = number of matching values)
/// - O(log n) point lookups
/// - O(log n) MIN/MAX queries from the tree's ends
/// - Efficient IN list queries
/// - NULL handling
///
/// B-tree index for single-column ordered access and range queries
pub struct BTreeIndex {
    /// Index name
    name: String,
    /// Table name
    table_name: String,
    /// Column ID
    column_id: i32,
    /// Column name
    column_name: String,
    /// Data type
    data_type: DataType,
    /// Whether this is a unique index
    unique: bool,
    /// What the index holds, shared with every object bound to its column
    data: Arc<BTreeData>,
}

/// A B-tree index's data and the state that changes with it
pub struct BTreeData {
    /// Whether the index is closed
    closed: AtomicBool,

    /// Sorted value to row IDs mapping (main index for range and equality queries)
    sorted_values: RwLock<ValueTree>,
}

impl std::ops::Deref for BTreeIndex {
    type Target = BTreeData;

    fn deref(&self) -> &BTreeData {
        &self.data
    }
}

impl BTreeIndex {
    /// Removes each entry's row from the list of its key, a key at a time:
    /// the entries are sorted by key and id before the lock is taken, and
    /// each key's ids leave its list in one pass
    fn remove_entries(&self, entries: &[(i64, &[Value])]) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }
        self.check_closed()?;

        let mut removed: Vec<(&Value, i64)> = entries
            .iter()
            .filter_map(|&(row_id, values)| values.first().map(|value| (value, row_id)))
            .collect();
        removed.sort_unstable_by(|a, b| a.0.cmp(b.0).then(a.1.cmp(&b.1)));
        let mut ids: Vec<i64> = Vec::with_capacity(removed.len());

        let mut sorted_values = self.sorted_values.write();
        for run in removed.chunk_by(|a, b| a.0 == b.0) {
            ids.clear();
            ids.extend(run.iter().map(|(_, row_id)| *row_id));
            sorted_values.remove_sorted(run[0].0, &ids);
        }
        Ok(())
    }

    /// Creates a new B-tree index
    pub fn new(
        name: String,
        table_name: String,
        column_id: i32,
        column_name: String,
        data_type: DataType,
        unique: bool,
    ) -> Self {
        Self {
            name,
            table_name,
            column_id,
            column_name,
            data_type,
            unique,
            data: Arc::new(BTreeData {
                closed: AtomicBool::new(false),
                sorted_values: RwLock::new(ValueTree::new()),
            }),
        }
    }

    /// Check if the index is closed
    fn check_closed(&self) -> Result<()> {
        if self.closed.load(AtomicOrdering::Acquire) {
            return Err(Error::IndexClosed);
        }
        Ok(())
    }

    /// The number of distinct values in the index
    #[cfg(test)]
    fn unique_value_count(&self) -> usize {
        self.sorted_values.read().len()
    }

    /// Every value with its row ids, in value order
    #[cfg(test)]
    fn grouped_row_ids(&self) -> Vec<(Value, Vec<i64>)> {
        let mut result = Vec::new();
        self.sorted_values
            .read()
            .walk(Bound::Unbounded, true, &mut |value, row_ids| {
                result.push((value.clone(), row_ids.iter().collect()));
                true
            });
        result
    }

    /// Returns the total number of entries (row IDs) in the index
    pub fn entry_count(&self) -> usize {
        let sorted_values = self.sorted_values.read();
        let mut count = 0;
        sorted_values.walk(Bound::Unbounded, true, &mut |_, rows| {
            count += rows.len();
            true
        });
        count
    }

    /// Gets all values in key order, each run of them charged to `deadline`
    pub fn get_all_values(&self, deadline: Deadline) -> Result<Vec<Value>> {
        let sorted_values = self.sorted_values.read();
        let mut values = Vec::with_capacity(sorted_values.len());
        if deadline.is_none() {
            sorted_values.walk(Bound::Unbounded, true, &mut |key, _| {
                values.push(key.clone());
                true
            });
            return Ok(values);
        }
        let mut stopped = None;
        sorted_values.walk(Bound::Unbounded, true, &mut |key, _| {
            values.push(key.clone());
            if values.len() % CHARGE_BEAT == 0 {
                if let Err(error) = deadline.charge(DeadlineSite::StorageRead, CHARGE_BEAT) {
                    stopped = Some(error);
                    return false;
                }
            }
            true
        });
        if let Some(error) = stopped {
            return Err(error);
        }
        deadline.charge(DeadlineSite::StorageRead, values.len() % CHARGE_BEAT)?;
        Ok(values)
    }

    /// Internal method to find row IDs matching an operator
    fn find_with_op(&self, op: Operator, value: &Value) -> Option<Vec<i64>> {
        let sorted_values = self.sorted_values.read();
        let mut results = Vec::new();
        // The rows of every key from `from` up while `keep` holds for it
        let mut collect = |from: Bound<&Value>, keep: &dyn Fn(&Value) -> bool| {
            sorted_values.walk(from, true, &mut |key, rows| {
                if !keep(key) {
                    return false;
                }
                rows.copy_into(&mut results);
                true
            })
        };
        match op {
            Operator::Eq | Operator::In => {
                if let Some(rows) = sorted_values.get(value) {
                    rows.copy_into(&mut results);
                }
            }
            Operator::Lt => collect(Bound::Unbounded, &|key| key.cmp(value) == Ordering::Less),
            Operator::Lte => collect(Bound::Unbounded, &|key| key.cmp(value) != Ordering::Greater),
            Operator::Gt => collect(Bound::Excluded(value), &|_| true),
            Operator::Gte => collect(Bound::Included(value), &|_| true),
            Operator::IsNull => {
                if let Some(rows) = sorted_values.get(&NULL) {
                    rows.copy_into(&mut results);
                }
            }
            Operator::IsNotNull => collect(Bound::Excluded(&NULL), &|_| true),
            _ => return None,
        }
        Some(results)
    }
}

impl Index for BTreeIndex {
    fn name(&self) -> &str {
        &self.name
    }

    fn add(&self, values: &[Value], row_id: i64, _ref_id: i64) -> Result<()> {
        self.check_closed()?;

        if values.is_empty() {
            return Ok(());
        }

        // BTree index only uses the first value (single column)
        let value = &values[0];

        // Hold the write lock for the entire operation to prevent race conditions
        // (read-then-write pattern is unsafe with concurrent modifications)
        let mut sorted_values = self.sorted_values.write();
        let unique = self.unique && !value.is_null();
        match sorted_values.insert(value, row_id, unique) {
            Ok(_) => Ok(()),
            Err(_) => Err(Error::unique_constraint(
                &self.name,
                &self.column_name,
                vec![value.clone()],
            )),
        }
    }

    fn remove(&self, values: &[Value], row_id: i64, _ref_id: i64) -> Result<()> {
        self.check_closed()?;

        let Some(value) = values.first() else {
            return Ok(());
        };
        self.sorted_values.write().remove(value, row_id);
        Ok(())
    }

    /// Optimized batch add with single lock acquisition
    ///
    /// Performance: O(1) lock acquisitions instead of O(N)
    fn add_batch_slice(&self, entries: &[(i64, &[Value])]) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }

        self.check_closed()?;

        // Acquire the write lock ONCE for entire batch
        let mut sorted_values = self.sorted_values.write();

        // For unique indexes: pre-check all entries before modifying
        if self.unique {
            for &(row_id, values) in entries {
                if values.is_empty() {
                    continue;
                }
                let value = &values[0];
                if value.is_null() {
                    continue;
                }

                // Check uniqueness against existing index
                if let Some(rows) = sorted_values.get(value) {
                    for existing_row_id in rows.iter() {
                        if existing_row_id != row_id {
                            return Err(Error::unique_constraint(
                                &self.name,
                                &self.column_name,
                                vec![value.clone()],
                            ));
                        }
                    }
                }
            }

            // Check for intra-batch duplicates
            // Use AHashMap per CLAUDE.md guidelines for Value keys (HashDoS resistance)
            let mut seen: ahash::AHashMap<&Value, i64> =
                ahash::AHashMap::with_capacity(entries.len());
            for &(row_id, values) in entries {
                if values.is_empty() {
                    continue;
                }
                let value = &values[0];
                if value.is_null() {
                    continue;
                }
                if let Some(&existing_row_id) = seen.get(value) {
                    if existing_row_id != row_id {
                        return Err(Error::unique_constraint(
                            &self.name,
                            &self.column_name,
                            vec![value.clone()],
                        ));
                    }
                }
                seen.insert(value, row_id);
            }
        }

        // All checks passed. An empty index is built from the entries
        if sorted_values.len() == 0 {
            let pairs = entries
                .iter()
                .filter_map(|&(row_id, values)| Some((values.first()?, row_id)));
            *sorted_values = ValueTree::from_entries(pairs.collect());
            return Ok(());
        }
        // An insert that does not check uniqueness refuses nothing
        for &(row_id, values) in entries {
            if let Some(value) = values.first() {
                let _ = sorted_values.insert(value, row_id, false);
            }
        }
        Ok(())
    }

    /// Optimized batch remove with single lock acquisition
    ///
    /// Performance: O(1) lock acquisitions instead of O(N)
    fn remove_batch_slice(&self, entries: &[(i64, &[Value])]) -> Result<()> {
        self.remove_entries(entries)
    }

    fn column_ids(&self) -> &[i32] {
        std::slice::from_ref(&self.column_id)
    }

    fn column_names(&self) -> &[String] {
        std::slice::from_ref(&self.column_name)
    }

    fn data_types(&self) -> &[DataType] {
        std::slice::from_ref(&self.data_type)
    }

    fn rebound(&self, column_names: &[String], column_ids: &[i32]) -> Arc<dyn Index> {
        Arc::new(Self {
            name: self.name.clone(),
            table_name: self.table_name.clone(),
            column_id: column_ids[0],
            column_name: column_names[0].clone(),
            data_type: self.data_type,
            unique: self.unique,
            data: Arc::clone(&self.data),
        })
    }

    fn index_type(&self) -> IndexType {
        IndexType::BTree
    }

    fn is_unique(&self) -> bool {
        self.unique
    }

    fn find(&self, values: &[Value]) -> Result<Vec<IndexEntry>> {
        self.check_closed()?;

        if values.is_empty() {
            return Ok(Vec::new());
        }

        let value = &values[0];
        let sorted_values = self.sorted_values.read();
        let entries = sorted_values
            .get(value)
            .map(|rows| {
                rows.iter()
                    .map(|row_id| IndexEntry {
                        row_id,
                        ref_id: row_id, // Use row_id as ref_id for simplicity
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(entries)
    }

    fn find_range(
        &self,
        min: &[Value],
        max: &[Value],
        min_inclusive: bool,
        max_inclusive: bool,
    ) -> Result<Vec<IndexEntry>> {
        self.check_closed()?;

        if min.is_empty() || max.is_empty() {
            return Ok(Vec::new());
        }

        let min_val = &min[0];
        let max_val = &max[0];

        let sorted_values = self.sorted_values.read();

        let min_bound = if min_inclusive {
            Bound::Included(min_val)
        } else {
            Bound::Excluded(min_val)
        };

        let max_bound = if max_inclusive {
            Bound::Included(max_val)
        } else {
            Bound::Excluded(max_val)
        };

        let mut entries = Vec::new();
        sorted_values.walk(min_bound, true, &mut |key, rows| {
            if past_end(&key, &max_bound) {
                return false;
            }
            entries.extend(rows.iter().map(|row_id| IndexEntry {
                row_id,
                ref_id: row_id,
            }));
            true
        });

        Ok(entries)
    }

    fn find_with_operator(&self, op: Operator, values: &[Value]) -> Result<Vec<IndexEntry>> {
        self.check_closed()?;

        if values.is_empty() {
            return Ok(Vec::new());
        }

        let value = &values[0];
        let row_ids = self.find_with_op(op, value).ok_or_else(|| {
            Error::internal(format!("btree index does not support operator {:?}", op))
        })?;

        Ok(row_ids
            .into_iter()
            .map(|row_id| IndexEntry {
                row_id,
                ref_id: row_id,
            })
            .collect())
    }

    fn get_row_ids_equal_into(&self, values: &[Value], buffer: &mut Vec<i64>) {
        if values.is_empty() {
            return;
        }

        let value = &values[0];
        let sorted_values = self.sorted_values.read();
        if let Some(rows) = sorted_values.get(value) {
            rows.copy_into(buffer);
        }
    }

    fn get_row_ids_in_into(
        &self,
        members: &mut dyn Iterator<Item = &Value>,
        buffer: &mut Vec<i64>,
    ) {
        let sorted_values = self.sorted_values.read();
        let mut found: smallvec::SmallVec<[(*const Ids, &Ids); 16]> = smallvec::SmallVec::new();
        for value in members {
            if let Some(rows) = sorted_values.get(value) {
                found.push((rows, rows));
            }
            #[cfg(feature = "test-failpoints")]
            crate::test_failpoints::in_member_probed();
        }
        super::each_entry_once(&found, |rows| rows.copy_into(buffer));
    }

    fn get_row_ids_equal_capped_into(
        &self,
        values: &[Value],
        max: usize,
        buffer: &mut Vec<i64>,
    ) -> Option<crate::storage::traits::CappedEqual> {
        use crate::storage::traits::CappedEqual;
        if values.is_empty() {
            return Some(CappedEqual::Copied);
        }

        let value = &values[0];
        let sorted_values = self.sorted_values.read();
        // The length is read before anything is appended, so an over-cap
        // bucket costs one lookup and no copy
        let Some(rows) = sorted_values.get(value) else {
            return Some(CappedEqual::Copied);
        };
        if rows.len() > max {
            return Some(CappedEqual::OverCap);
        }
        rows.copy_into(buffer);
        Some(CappedEqual::Copied)
    }

    fn count_capped(
        &self,
        lower: Bound<&Value>,
        upper: Bound<&Value>,
        max: usize,
    ) -> Option<usize> {
        if self.closed.load(AtomicOrdering::Acquire) {
            return None;
        }
        let sorted_values = self.sorted_values.read();
        if let (Bound::Included(low), Bound::Included(high)) = (lower, upper) {
            if low == high {
                let count = sorted_values.get(low).map_or(0, Ids::len);
                return Some(count.min(max.saturating_add(1)));
            }
        }
        let mut count = 0usize;
        sorted_values.walk(lower, true, &mut |key, rows| {
            if past_end(&key, &upper) {
                return false;
            }
            count += rows.len();
            count <= max
        });
        Some(count.min(max.saturating_add(1)))
    }

    fn get_row_ids_in_range_into(
        &self,
        min_value: &[Value],
        max_value: &[Value],
        include_min: bool,
        include_max: bool,
        buffer: &mut Vec<i64>,
    ) {
        if min_value.is_empty() || max_value.is_empty() {
            return;
        }

        // OPTIMIZATION: Collect row_ids directly without intermediate IndexEntry allocation
        if self.closed.load(AtomicOrdering::Acquire) {
            return;
        }

        let min_val = &min_value[0];
        let max_val = &max_value[0];

        let sorted_values = self.sorted_values.read();

        let min_bound = if include_min {
            Bound::Included(min_val)
        } else {
            Bound::Excluded(min_val)
        };

        let max_bound = if include_max {
            Bound::Included(max_val)
        } else {
            Bound::Excluded(max_val)
        };

        sorted_values.walk(min_bound, true, &mut |key, rows| {
            if past_end(&key, &max_bound) {
                return false;
            }
            rows.copy_into(buffer);
            true
        });
    }

    /// Returns the minimum value in the index; NULL, one key, sorts first
    fn get_min_value(&self) -> Option<Value> {
        if self.closed.load(AtomicOrdering::Acquire) {
            return None;
        }
        let sorted_values = self.sorted_values.read();
        let mut min = None;
        sorted_values.walk(Bound::Excluded(&NULL), true, &mut |key, _| {
            min = Some(key.clone());
            false
        });
        min
    }

    /// Returns the maximum value in the index
    fn get_max_value(&self) -> Option<Value> {
        if self.closed.load(AtomicOrdering::Acquire) {
            return None;
        }
        let sorted_values = self.sorted_values.read();
        let mut max = None;
        sorted_values.walk(Bound::Unbounded, false, &mut |key, _| {
            max = (!key.is_null()).then(|| key.clone());
            false
        });
        max
    }

    fn get_all_values(&self, deadline: Deadline) -> Result<Option<Vec<Value>>> {
        if self.closed.load(AtomicOrdering::Acquire) {
            return Ok(None);
        }
        BTreeIndex::get_all_values(self, deadline).map(Some)
    }

    fn get_distinct_count_excluding_null(&self) -> Option<usize> {
        if self.closed.load(AtomicOrdering::Acquire) {
            return None;
        }
        let sorted_values = self.sorted_values.read();
        let nulls = usize::from(sorted_values.get(&NULL).is_some());
        Some(sorted_values.len() - nulls)
    }

    fn get_row_ids_ordered(
        &self,
        ascending: bool,
        nulls_first: bool,
        limit: usize,
        offset: usize,
    ) -> Option<Vec<i64>> {
        if self.closed.load(AtomicOrdering::Acquire) {
            return None;
        }

        let sorted_values = self.sorted_values.read();
        let mut ordered = OrderedIds::new(limit, offset);

        // NULL is the first key of the tree: in its order when the caller
        // wants it there, otherwise moved to the other end
        let nulls = sorted_values.get(&NULL);
        match (ascending, nulls_first) {
            (true, false) => {
                sorted_values.walk(Bound::Excluded(&NULL), true, &mut |_, rows| {
                    ordered.take(rows)
                });
                if let Some(nulls) = nulls {
                    ordered.take(nulls);
                }
            }
            (false, true) => {
                if nulls.is_none_or(|nulls| ordered.take(nulls)) {
                    sorted_values.walk(Bound::Unbounded, false, &mut |key, rows| {
                        !key.is_null() && ordered.take(rows)
                    });
                }
            }
            (_, _) => sorted_values.walk(Bound::Unbounded, ascending, &mut |_, rows| {
                ordered.take(rows)
            }),
        }
        Some(ordered.ids)
    }

    /// Walks the whole key, so only an empty prefix. Each chunk of row ids is
    /// copied under the lock into room made before it, and visited after the
    /// lock is released; a chunk may stop inside a key's ids.
    fn walk_prefix_ordered(
        &self,
        prefix: &[Value],
        lower: Option<(&Value, bool)>,
        upper: Option<(&Value, bool)>,
        ascending: bool,
        visit: &mut dyn FnMut(i64, &Value) -> bool,
    ) -> bool {
        if !prefix.is_empty() || self.closed.load(AtomicOrdering::Acquire) {
            return false;
        }
        let bound = |side: Option<(&Value, bool)>| match side {
            Some((value, true)) => Bound::Included(value.clone()),
            Some((value, false)) => Bound::Excluded(value.clone()),
            None => Bound::Unbounded,
        };
        let (mut start, mut end) = (bound(lower), bound(upper));
        // The key the last chunk stopped in, and the last id it copied there
        let mut inside: Option<(Value, i64)> = None;
        let mut ids: Vec<i64> = Vec::new();
        let mut keys: Vec<(Value, usize)> = Vec::new();
        let mut chunk = 64usize;
        loop {
            ids.clear();
            keys.clear();
            ids.reserve_exact(chunk);
            keys.reserve_exact(chunk);
            let room = (ids.capacity(), keys.capacity());
            let mut full = false;
            {
                let sorted_values = self.sorted_values.read();
                let mut take = |key: &Value, rows: &Ids, after: Option<i64>| {
                    let before = ids.len();
                    rows.copy_past_into(after, ascending, chunk - before, &mut ids);
                    if ids.len() > before {
                        keys.push((key.clone(), ids.len()));
                    }
                    ids.len() == chunk
                };
                if let Some((key, last)) = &inside {
                    if let Some(rows) = sorted_values.get(key) {
                        full = take(key, rows, Some(*last));
                    }
                }
                if !full && ascending {
                    sorted_values.walk(start.as_ref(), true, &mut |key, rows| {
                        if past_end(key, &end) {
                            return false;
                        }
                        full = take(key, rows, None);
                        !full
                    });
                } else if !full {
                    sorted_values.walk(end.as_ref(), false, &mut |key, rows| {
                        if before_start(key, &start) {
                            return false;
                        }
                        full = take(key, rows, None);
                        !full
                    });
                }
            }
            debug_assert_eq!(room, (ids.capacity(), keys.capacity()));
            #[cfg(feature = "test-failpoints")]
            crate::test_failpoints::btree_walk_chunk(ids.len(), ids.capacity());
            let mut from = 0;
            for (key, upto) in &keys {
                for &row_id in &ids[from..*upto] {
                    if !visit(row_id, key) {
                        return true;
                    }
                }
                from = *upto;
            }
            let (Some((last, _)), Some(&last_id), true) = (keys.pop(), ids.last(), full) else {
                return true;
            };
            if ascending {
                start = Bound::Excluded(last.clone());
            } else {
                end = Bound::Excluded(last.clone());
            }
            inside = Some((last, last_id));
            chunk = (chunk * 2).min(4096);
        }
    }

    fn for_each_group(
        &self,
        callback: &mut dyn FnMut(&Value, GroupIds<'_>) -> Result<bool>,
    ) -> Option<Result<()>> {
        if self.closed.load(AtomicOrdering::Acquire) {
            return None;
        }

        let sorted_values = self.sorted_values.read();

        // One call per whole group in key order, borrowed: no allocation
        let mut outcome = Ok(());
        sorted_values.walk(
            Bound::Unbounded,
            true,
            &mut |value, row_ids| match callback(value, row_ids.as_group()) {
                Ok(more) => more,
                Err(e) => {
                    outcome = Err(e);
                    false
                }
            },
        );
        Some(outcome)
    }

    fn clear(&self) -> Result<()> {
        self.sorted_values.write().clear();
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// The row ids of keys taken in order, past the first `offset`, up to
/// `limit`
struct OrderedIds {
    ids: Vec<i64>,
    limit: usize,
    skip: usize,
}

impl OrderedIds {
    fn new(limit: usize, offset: usize) -> Self {
        OrderedIds {
            ids: Vec::with_capacity(limit.min(128)),
            limit,
            skip: offset,
        }
    }

    /// Takes a key's rows: whether more are wanted. A full walk takes no
    /// more, whichever key comes next
    fn take(&mut self, rows: &Ids) -> bool {
        for row_id in rows.iter() {
            if self.ids.len() >= self.limit {
                return false;
            }
            if self.skip > 0 {
                self.skip -= 1;
                continue;
            }
            self.ids.push(row_id);
        }
        self.ids.len() < self.limit
    }
}

/// Whether `key` lies past `end` in the map's own `Ord`. A range is walked
/// from its start alone and stopped here: `BTreeMap::range` checks its two
/// bounds with `PartialOrd`, which for `Value` can disagree with `Ord` and panic
pub(crate) fn past_end<K: Ord>(key: &K, end: &Bound<K>) -> bool {
    match end {
        Bound::Included(end) => Ord::cmp(key, end) == std::cmp::Ordering::Greater,
        Bound::Excluded(end) => Ord::cmp(key, end) != std::cmp::Ordering::Less,
        Bound::Unbounded => false,
    }
}

/// Whether `key` lies before `start` in the map's own `Ord`, for a range
/// walked down from its end alone
pub(crate) fn before_start<K: Ord>(key: &K, start: &Bound<K>) -> bool {
    match start {
        Bound::Included(start) => Ord::cmp(key, start) == std::cmp::Ordering::Less,
        Bound::Excluded(start) => Ord::cmp(key, start) != std::cmp::Ordering::Greater,
        Bound::Unbounded => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_index() -> BTreeIndex {
        BTreeIndex::new(
            "idx_test".to_string(),
            "test_table".to_string(),
            0,
            "id".to_string(),
            DataType::Integer,
            false,
        )
    }

    /// The cap is checked against the bucket length before anything is
    /// copied, so an over-cap key costs one lookup and leaves the buffer
    /// untouched. The buffer's length is the evidence that nothing was copied.
    #[test]
    fn a_capped_equal_probe_copies_only_under_the_cap() {
        use crate::storage::traits::CappedEqual;

        let index = create_test_index();
        for i in 1..=4096i64 {
            index.add(&[Value::Integer(1)], i, i).unwrap();
        }
        index.add(&[Value::Integer(2)], 9001, 9001).unwrap();
        index.add(&[Value::Integer(3)], 9002, 9002).unwrap();

        let at_cap = [Value::Integer(1)];
        let mut buffer = Vec::new();
        assert_eq!(
            index.get_row_ids_equal_capped_into(&at_cap, 4095, &mut buffer),
            Some(CappedEqual::OverCap)
        );
        assert!(buffer.is_empty(), "one over the cap copies nothing");

        assert_eq!(
            index.get_row_ids_equal_capped_into(&at_cap, 4096, &mut buffer),
            Some(CappedEqual::Copied)
        );
        assert_eq!(buffer.len(), 4096, "the cap is inclusive");

        // A key the index does not hold is a complete answer, not a refusal
        buffer.clear();
        assert_eq!(
            index.get_row_ids_equal_capped_into(&[Value::Integer(99)], 4096, &mut buffer),
            Some(CappedEqual::Copied)
        );
        assert!(buffer.is_empty());

        // One under the cap copies all of it
        buffer.clear();
        assert_eq!(
            index.get_row_ids_equal_capped_into(&[Value::Integer(2)], 1, &mut buffer),
            Some(CappedEqual::Copied)
        );
        assert_eq!(buffer, vec![9001]);
    }

    #[test]
    fn test_btree_index_creation() {
        let index = create_test_index();
        assert_eq!(index.name(), "idx_test");
        assert_eq!(index.column_names()[0], "id");
        assert_eq!(index.index_type(), IndexType::BTree);
        assert!(!index.is_unique());
    }

    #[test]
    fn test_btree_index_add_and_find() {
        let index = create_test_index();

        // Add some values
        index.add(&[Value::Integer(100)], 1, 1).unwrap();
        index.add(&[Value::Integer(200)], 2, 2).unwrap();
        index.add(&[Value::Integer(100)], 3, 3).unwrap(); // Duplicate value

        assert_eq!(index.unique_value_count(), 2);
        assert_eq!(index.entry_count(), 3);

        // Find exact match
        let entries = index.find(&[Value::Integer(100)]).unwrap();
        assert_eq!(entries.len(), 2);

        // Find non-existent
        let entries = index.find(&[Value::Integer(999)]).unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn test_btree_index_unique_constraint() {
        let index = BTreeIndex::new(
            "idx_unique".to_string(),
            "test".to_string(),
            0,
            "id".to_string(),
            DataType::Integer,
            true, // unique
        );

        // First insert should succeed
        assert!(index.add(&[Value::Integer(1)], 1, 1).is_ok());

        // Duplicate should fail
        let result = index.add(&[Value::Integer(1)], 2, 2);
        assert!(result.is_err());

        // NULL values should be allowed multiple times
        assert!(index.add(&[Value::null_unknown()], 3, 3).is_ok());
        assert!(index.add(&[Value::null_unknown()], 4, 4).is_ok());
    }

    #[test]
    fn test_btree_index_remove() {
        let index = create_test_index();

        index.add(&[Value::Integer(100)], 1, 1).unwrap();
        index.add(&[Value::Integer(100)], 2, 2).unwrap();
        index.add(&[Value::Integer(200)], 3, 3).unwrap();

        assert_eq!(index.entry_count(), 3);

        // Remove one entry
        index.remove(&[Value::Integer(100)], 1, 1).unwrap();
        assert_eq!(index.entry_count(), 2);

        // Value 100 should still exist (for row 2)
        let entries = index.find(&[Value::Integer(100)]).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].row_id, 2);

        // Remove the last entry with value 100
        index.remove(&[Value::Integer(100)], 2, 2).unwrap();
        assert!(index.find(&[Value::Integer(100)]).unwrap().is_empty());
    }

    #[test]
    fn test_btree_index_range_query() {
        let index = create_test_index();

        for i in 1..=10 {
            index.add(&[Value::Integer(i * 10)], i, i).unwrap();
        }

        // Range [30, 70] inclusive
        let entries = index
            .find_range(&[Value::Integer(30)], &[Value::Integer(70)], true, true)
            .unwrap();
        assert_eq!(entries.len(), 5); // 30, 40, 50, 60, 70

        // Range (30, 70) exclusive
        let entries = index
            .find_range(&[Value::Integer(30)], &[Value::Integer(70)], false, false)
            .unwrap();
        assert_eq!(entries.len(), 3); // 40, 50, 60
    }

    #[test]
    fn test_btree_index_operators() {
        let index = create_test_index();

        for i in 1..=5 {
            index.add(&[Value::Integer(i * 10)], i, i).unwrap();
        }

        // Less than 30
        let entries = index
            .find_with_operator(Operator::Lt, &[Value::Integer(30)])
            .unwrap();
        assert_eq!(entries.len(), 2); // 10, 20

        // Greater than or equal to 40
        let entries = index
            .find_with_operator(Operator::Gte, &[Value::Integer(40)])
            .unwrap();
        assert_eq!(entries.len(), 2); // 40, 50

        for op in [Operator::Ne, Operator::NotIn, Operator::Like] {
            assert!(
                index.find_with_operator(op, &[Value::Integer(30)]).is_err(),
                "{op:?}"
            );
        }
    }

    #[test]
    fn a_batch_built_into_an_empty_index_holds_what_one_by_one_adds_hold() {
        // NULLs, repeated keys, INTEGER and FLOAT keys that compare equal,
        // and words FLOAT values share
        let values: Vec<Value> = (0..5_000i64)
            .map(|i| match i % 5 {
                0 => Value::Null(DataType::Integer),
                1 => Value::Integer((i * 7_919) % 600),
                2 => Value::Float(((i * 104_729) % 600) as f64),
                3 => Value::Float((i % 997) as f64 * 0.37),
                _ => Value::Integer(-(i % 31)),
            })
            .collect();
        let entries: Vec<(i64, &[Value])> = values
            .iter()
            .enumerate()
            .map(|(id, v)| (id as i64, std::slice::from_ref(v)))
            .collect();
        let batched = create_test_index();
        batched.add_batch_slice(&entries).unwrap();
        let one_by_one = create_test_index();
        for &(id, v) in &entries {
            one_by_one.add(v, id, id).unwrap();
        }
        assert_eq!(batched.grouped_row_ids(), one_by_one.grouped_row_ids());
        assert_eq!(batched.entry_count(), values.len());
    }

    #[test]
    fn an_ordered_walk_stops_at_its_limit_on_either_side_of_the_nulls() {
        let index = create_test_index();
        index.add(&[Value::Integer(1)], 1, 1).unwrap();
        index.add(&[Value::Integer(2)], 3, 3).unwrap();
        index.add(&[Value::Null(DataType::Integer)], 2, 2).unwrap();
        // (ascending, nulls_first, limit, offset) and the ids in order
        let cases = [
            ((true, false, 1, 0), vec![1]),
            ((true, false, 2, 0), vec![1, 3]),
            ((true, false, 1, 2), vec![2]),
            ((true, false, 3, 1), vec![3, 2]),
            ((true, true, 1, 0), vec![2]),
            ((false, true, 1, 0), vec![2]),
            ((false, true, 2, 1), vec![3, 1]),
            ((false, false, 2, 0), vec![3, 1]),
            ((false, false, 1, 2), vec![2]),
            ((true, false, 0, 0), vec![]),
        ];
        for ((ascending, nulls_first, limit, offset), want) in cases {
            assert_eq!(
                index
                    .get_row_ids_ordered(ascending, nulls_first, limit, offset)
                    .unwrap(),
                want,
                "ascending {ascending}, nulls first {nulls_first}, limit {limit}, offset {offset}"
            );
        }
    }

    #[test]
    fn test_btree_index_null_handling() {
        let index = create_test_index();

        index.add(&[Value::null_unknown()], 1, 1).unwrap();
        index.add(&[Value::Integer(100)], 2, 2).unwrap();
        index.add(&[Value::null_unknown()], 3, 3).unwrap();

        // IS NULL - value parameter is ignored for IsNull operator
        let entries = index
            .find_with_operator(Operator::IsNull, &[Value::null_unknown()])
            .unwrap();
        assert_eq!(entries.len(), 2);

        // IS NOT NULL - value parameter is ignored for IsNotNull operator
        let entries = index
            .find_with_operator(Operator::IsNotNull, &[Value::null_unknown()])
            .unwrap();
        assert_eq!(entries.len(), 1);
    }

    #[test]
    fn test_btree_index_text_values() {
        let index = BTreeIndex::new(
            "idx_name".to_string(),
            "test".to_string(),
            0,
            "name".to_string(),
            DataType::Text,
            false,
        );

        index.add(&[Value::text("Alice")], 1, 1).unwrap();
        index.add(&[Value::text("Bob")], 2, 2).unwrap();
        index.add(&[Value::text("Charlie")], 3, 3).unwrap();

        // Range query on text (alphabetical)
        let entries = index
            .find_range(&[Value::text("Alice")], &[Value::text("Bob")], true, true)
            .unwrap();
        assert_eq!(entries.len(), 2); // Alice, Bob
    }

    #[test]
    fn test_btree_index_get_all_values() {
        let index = create_test_index();

        index.add(&[Value::Integer(30)], 1, 1).unwrap();
        index.add(&[Value::Integer(10)], 2, 2).unwrap();
        index.add(&[Value::Integer(20)], 3, 3).unwrap();

        let values = index.get_all_values(Deadline::NONE).unwrap();
        assert_eq!(values.len(), 3);
        // Values may not be sorted (HashMap)
        assert!(values.contains(&Value::Integer(10)));
        assert!(values.contains(&Value::Integer(20)));
        assert!(values.contains(&Value::Integer(30)));
    }

    #[test]
    fn test_btree_index_row_id_helpers() {
        let index = create_test_index();

        index.add(&[Value::Integer(100)], 1, 1).unwrap();
        index.add(&[Value::Integer(100)], 2, 2).unwrap();
        index.add(&[Value::Integer(200)], 3, 3).unwrap();

        // get_row_ids_equal
        let row_ids = index.get_row_ids_equal(&[Value::Integer(100)]);
        assert_eq!(row_ids.len(), 2);

        // get_row_ids_in_range
        let row_ids =
            index.get_row_ids_in_range(&[Value::Integer(100)], &[Value::Integer(200)], true, true);
        assert_eq!(row_ids.len(), 3);
    }

    #[test]
    fn test_btree_index_lte_gte_operators() {
        let index = create_test_index();

        for i in 1..=5 {
            index.add(&[Value::Integer(i * 10)], i, i).unwrap();
        }

        // Less than or equal to 30
        let entries = index
            .find_with_operator(Operator::Lte, &[Value::Integer(30)])
            .unwrap();
        assert_eq!(entries.len(), 3); // 10, 20, 30

        // Greater than 20
        let entries = index
            .find_with_operator(Operator::Gt, &[Value::Integer(20)])
            .unwrap();
        assert_eq!(entries.len(), 3); // 30, 40, 50
    }

    #[test]
    fn test_mixed_integer_float_ordering() {
        // Test that Integer and Float values are properly ordered together
        // This validates the Ord implementation consistency with PartialEq
        let index = BTreeIndex::new(
            "idx_mixed".to_string(),
            "test".to_string(),
            0,
            "value".to_string(),
            DataType::Float, // Use Float type to allow mixed values
            false,
        );

        // Add mixed Integer and Float values
        index.add(&[Value::Integer(1)], 1, 1).unwrap();
        index.add(&[Value::Float(2.5)], 2, 2).unwrap();
        index.add(&[Value::Integer(3)], 3, 3).unwrap();
        index.add(&[Value::Float(1.5)], 4, 4).unwrap();
        index.add(&[Value::Float(3.0)], 5, 5).unwrap(); // Same as Integer(3)

        // Integer(3) == Float(3.0), so they should be in the same bucket
        // The add() for Float(3.0) should be a no-op since row 5 doesn't exist yet
        // Actually, it should add normally since row 5 is new
        assert_eq!(index.entry_count(), 5);

        // Range query: all values >= 2
        let entries = index
            .find_with_operator(Operator::Gte, &[Value::Integer(2)])
            .unwrap();
        // Should find: 2.5, 3, 3.0 (row_ids: 2, 3, 5)
        assert_eq!(entries.len(), 3);

        // Range query: all values < 2
        let entries = index
            .find_with_operator(Operator::Lt, &[Value::Float(2.0)])
            .unwrap();
        // Should find: 1, 1.5 (row_ids: 1, 4)
        assert_eq!(entries.len(), 2);

        // Equality lookup with cross-type comparison
        // Integer(3) should find Float(3.0) and vice versa if they're in the same bucket
        let entries = index.find(&[Value::Integer(3)]).unwrap();
        // Due to how BTreeMap works with Ord, Integer(3) and Float(3.0) are Equal
        // so they should be in the same entry
        assert!(!entries.is_empty()); // At least row 3
    }

    #[test]
    fn test_nan_handling_in_range_queries() {
        // Test that NaN values are handled correctly (ordered last)
        let index = BTreeIndex::new(
            "idx_nan".to_string(),
            "test".to_string(),
            0,
            "value".to_string(),
            DataType::Float,
            false,
        );

        index.add(&[Value::Float(1.0)], 1, 1).unwrap();
        index.add(&[Value::Float(f64::NAN)], 2, 2).unwrap();
        index.add(&[Value::Float(2.0)], 3, 3).unwrap();
        index.add(&[Value::Float(f64::INFINITY)], 4, 4).unwrap();
        index.add(&[Value::Float(f64::NEG_INFINITY)], 5, 5).unwrap();

        assert_eq!(index.entry_count(), 5);

        // NaN should be ordered last, so Gt(any number) should NOT include NaN
        // Actually, per our Ord impl, NaN > any number, so Gt(100) would include NaN
        let entries = index
            .find_with_operator(Operator::Gt, &[Value::Float(100.0)])
            .unwrap();
        // Should find: INFINITY, NAN (row_ids: 4, 2)
        assert_eq!(entries.len(), 2);

        // Lt(NaN) - everything except NaN should be less than NaN
        // But NaN comparisons are tricky - partial_cmp returns None for NaN
        // Our Ord impl makes NaN > everything, so Lt(NaN) should return all non-NaN values
        let entries = index
            .find_with_operator(Operator::Lt, &[Value::Float(f64::NAN)])
            .unwrap();
        // Should find: NEG_INFINITY, 1.0, 2.0, INFINITY (row_ids: 5, 1, 3, 4)
        assert_eq!(entries.len(), 4);
    }

    #[test]
    fn test_high_cardinality_duplicates() {
        // Test performance and correctness with many row_ids per value
        let index = BTreeIndex::new(
            "idx_highcard".to_string(),
            "test".to_string(),
            0,
            "status".to_string(),
            DataType::Text,
            false,
        );

        // Add 100 rows with just 3 distinct values (low cardinality column)
        for i in 0..100 {
            let status = match i % 3 {
                0 => "active",
                1 => "pending",
                _ => "completed",
            };
            index
                .add(&[Value::text(status)], i as i64, i as i64)
                .unwrap();
        }

        assert_eq!(index.unique_value_count(), 3);
        assert_eq!(index.entry_count(), 100);

        // Find all "active" rows (should be ~33)
        let entries = index.find(&[Value::text("active")]).unwrap();
        assert_eq!(entries.len(), 34); // 0, 3, 6, 9, ... 99

        // Find all "pending" rows
        let entries = index.find(&[Value::text("pending")]).unwrap();
        assert_eq!(entries.len(), 33); // 1, 4, 7, ... 97

        // Update a row's value: the old key leaves, the new one comes
        index.remove(&[Value::text("active")], 0, 0).unwrap();
        index.add(&[Value::text("archived")], 0, 0).unwrap();
        assert_eq!(index.unique_value_count(), 4);

        // "active" should now have one less row
        let entries = index.find(&[Value::text("active")]).unwrap();
        assert_eq!(entries.len(), 33);

        // "archived" should have the moved row
        let entries = index.find(&[Value::text("archived")]).unwrap();
        assert_eq!(entries.len(), 1);
    }

    #[test]
    fn test_update_same_value_is_noop() {
        // Test that updating a row with the same value is a no-op
        let index = create_test_index();

        index.add(&[Value::Integer(100)], 1, 1).unwrap();
        assert_eq!(index.entry_count(), 1);

        // Add same row_id with same value - should be no-op
        index.add(&[Value::Integer(100)], 1, 1).unwrap();
        assert_eq!(index.entry_count(), 1);

        // Verify the value is still there
        let entries = index.find(&[Value::Integer(100)]).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].row_id, 1);
    }

    #[test]
    fn test_update_different_value_moves_row() {
        // Test that updating a row with a different value moves it
        let index = create_test_index();

        index.add(&[Value::Integer(100)], 1, 1).unwrap();
        index.add(&[Value::Integer(100)], 2, 2).unwrap();
        assert_eq!(index.entry_count(), 2);

        // Update row 1 to have value 200
        index.remove(&[Value::Integer(100)], 1, 1).unwrap();
        index.add(&[Value::Integer(200)], 1, 1).unwrap();
        assert_eq!(index.entry_count(), 2);
        assert_eq!(index.unique_value_count(), 2);

        // Value 100 should only have row 2 now
        let entries = index.find(&[Value::Integer(100)]).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].row_id, 2);

        // Value 200 should have row 1
        let entries = index.find(&[Value::Integer(200)]).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].row_id, 1);
    }
}
