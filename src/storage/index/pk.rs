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

//! Primary Key Index backed by a hybrid bitset + I64Set for O(1) lookups.
//!
//! For INTEGER PRIMARY KEY columns where row_id == pk_value, PkIndex uses:
//! - A `Vec<u64>` bitset for row_ids in `[0, BITSET_MAX_BITS)` — O(1) lookups at
//!   ~0.125 bytes/row.
//! - An `I64Set` overflow for row_ids outside that range — O(1) amortized lookups
//!   at ~10.7 bytes/entry.
//! - A `has_i64_min` flag because I64Set uses `i64::MIN` as its empty sentinel.
//!
//! All mutable state lives behind a **single `RwLock<PkIndexInner>`** to prevent
//! deadlocks (no lock-ordering concerns) and keep `count` always consistent.
//!
//! PkIndex participates in normal commit add/remove calls like any other index.
//! It reports `IndexType::PrimaryKey` which is used to filter it from
//! `list_indexes` display and to block redundant CREATE INDEX on the PK column.

use parking_lot::RwLock;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::common::{I64Map, I64Set};
use crate::core::{DataType, IndexEntry, IndexType, Operator, Result, RowIdVec, Value};
use crate::storage::expression::Expression;
use crate::storage::Index;

/// Row IDs in `[0, BITSET_MAX_BITS)` use the fast bitset path (~156 KB max).
/// Row IDs outside this range go into the overflow `I64Set`.
const BITSET_MAX_BITS: usize = 10_000_000;

/// Number of bits per u64 word.
const BITS: usize = 64;

// ---------------------------------------------------------------------------
// Inner data protected by a single RwLock
// ---------------------------------------------------------------------------

/// All mutable state for the PkIndex, guarded by one RwLock.
struct PkIndexInner {
    /// Bitset for row_ids in [0, BITSET_MAX_BITS). Grows lazily.
    words: Vec<u64>,
    /// Overflow set for row_ids >= BITSET_MAX_BITS or negative (except i64::MIN).
    overflow: I64Set,
    /// Separate flag for i64::MIN which I64Set cannot store (used as sentinel).
    has_i64_min: bool,
    /// Exact count of present entries.
    count: usize,
}

impl PkIndexInner {
    fn new() -> Self {
        Self {
            words: Vec::new(),
            overflow: I64Set::new(),
            has_i64_min: false,
            count: 0,
        }
    }

    // -- Primitive helpers --------------------------------------------------

    /// Check whether `id` is currently present.
    #[inline]
    fn contains(&self, id: i64) -> bool {
        if let Some((word_idx, mask)) = to_word_bit(id) {
            word_idx < self.words.len() && (self.words[word_idx] & mask) != 0
        } else if id == i64::MIN {
            self.has_i64_min
        } else {
            self.overflow.contains(id)
        }
    }

    /// Insert `id`. Returns `true` if newly inserted (was absent).
    #[inline]
    fn insert(&mut self, id: i64) -> bool {
        if let Some((word_idx, mask)) = to_word_bit(id) {
            ensure_capacity(&mut self.words, word_idx);
            if (self.words[word_idx] & mask) == 0 {
                self.words[word_idx] |= mask;
                self.count += 1;
                true
            } else {
                false
            }
        } else if id == i64::MIN {
            if self.has_i64_min {
                false
            } else {
                self.has_i64_min = true;
                self.count += 1;
                true
            }
        } else if self.overflow.insert(id) {
            self.count += 1;
            true
        } else {
            false
        }
    }

    /// Remove `id`. Returns `true` if it was present and removed.
    #[inline]
    fn remove(&mut self, id: i64) -> bool {
        if let Some((word_idx, mask)) = to_word_bit(id) {
            if word_idx < self.words.len() && (self.words[word_idx] & mask) != 0 {
                self.words[word_idx] &= !mask;
                self.count -= 1;
                true
            } else {
                false
            }
        } else if id == i64::MIN {
            if self.has_i64_min {
                self.has_i64_min = false;
                self.count -= 1;
                true
            } else {
                false
            }
        } else if self.overflow.remove(id) {
            self.count -= 1;
            true
        } else {
            false
        }
    }

    /// True when there are no entries in the overflow region.
    #[inline]
    fn overflow_empty(&self) -> bool {
        self.overflow.is_empty() && !self.has_i64_min
    }

    /// Reset everything.
    fn clear(&mut self) {
        self.words.clear();
        self.overflow = I64Set::new();
        self.has_i64_min = false;
        self.count = 0;
    }
}

// ---------------------------------------------------------------------------
// Free-standing bitset helpers (no &self)
// ---------------------------------------------------------------------------

/// Map a non-negative row_id below `BITSET_MAX_BITS` to (word_index, bitmask).
#[inline]
fn to_word_bit(row_id: i64) -> Option<(usize, u64)> {
    if row_id >= 0 {
        let idx = row_id as usize;
        if idx < BITSET_MAX_BITS {
            return Some((idx / BITS, 1u64 << (idx % BITS)));
        }
    }
    None
}

/// Grow `words` so that `word_idx` is valid.
#[inline]
fn ensure_capacity(words: &mut Vec<u64>, word_idx: usize) {
    if word_idx >= words.len() {
        words.resize(word_idx + 1, 0);
    }
}

/// Total bit capacity of a word slice.
#[inline]
fn bit_capacity(words: &[u64]) -> usize {
    words.len() * BITS
}

/// Iterate set bits in ascending order. Returns early if `f` returns false.
#[inline]
fn for_each_set_bit(words: &[u64], mut f: impl FnMut(i64) -> bool) {
    for (word_idx, &word) in words.iter().enumerate() {
        if word == 0 {
            continue;
        }
        let base = (word_idx * BITS) as i64;
        let mut w = word;
        while w != 0 {
            let bit = w.trailing_zeros() as i64;
            if !f(base + bit) {
                return;
            }
            w &= w - 1; // clear lowest set bit
        }
    }
}

/// Iterate set bits in descending order. Returns early if `f` returns false.
#[inline]
fn for_each_set_bit_rev(words: &[u64], mut f: impl FnMut(i64) -> bool) {
    for word_idx in (0..words.len()).rev() {
        let word = words[word_idx];
        if word == 0 {
            continue;
        }
        let base = (word_idx * BITS) as i64;
        let mut w = word;
        while w != 0 {
            let bit = (BITS - 1 - w.leading_zeros() as usize) as i64;
            if !f(base + bit) {
                return;
            }
            w &= !(1u64 << bit); // clear highest set bit
        }
    }
}

/// Iterate set bits in the inclusive range `[lo, hi]`.
fn for_each_set_bit_in_range(words: &[u64], lo: usize, hi: usize, mut f: impl FnMut(i64) -> bool) {
    let lo_word = lo / BITS;
    let hi_word = hi / BITS;
    let max_word = words.len().saturating_sub(1);

    if lo_word > max_word {
        return;
    }

    let hi_word = hi_word.min(max_word);

    for (word_idx, &raw_word) in words.iter().enumerate().take(hi_word + 1).skip(lo_word) {
        let mut word = raw_word;
        if word == 0 {
            continue;
        }
        // Mask out bits below lo in first word
        if word_idx == lo_word {
            let lo_bit = lo % BITS;
            word &= !((1u64 << lo_bit) - 1);
        }
        // Mask out bits above hi in last word
        if word_idx == hi_word {
            let hi_bit = hi % BITS;
            if hi_bit < BITS - 1 {
                word &= (1u64 << (hi_bit + 1)) - 1;
            }
        }
        if word == 0 {
            continue;
        }
        let base = (word_idx * BITS) as i64;
        while word != 0 {
            let bit = word.trailing_zeros() as i64;
            if !f(base + bit) {
                return;
            }
            word &= word - 1;
        }
    }
}

/// Collect set bits in `[lo, hi]` as `IndexEntry` values.
fn collect_range_entries(words: &[u64], lo: usize, hi: usize) -> Vec<IndexEntry> {
    let mut result = Vec::new();
    for_each_set_bit_in_range(words, lo, hi, |rid| {
        result.push(IndexEntry {
            row_id: rid,
            ref_id: rid,
        });
        true
    });
    result
}

/// Smallest set bit, or `None` if all-zero.
fn bitset_min(words: &[u64]) -> Option<i64> {
    for (word_idx, &word) in words.iter().enumerate() {
        if word != 0 {
            return Some((word_idx * BITS) as i64 + word.trailing_zeros() as i64);
        }
    }
    None
}

/// Largest set bit, or `None` if all-zero.
fn bitset_max(words: &[u64]) -> Option<i64> {
    for word_idx in (0..words.len()).rev() {
        let word = words[word_idx];
        if word != 0 {
            let bit = (BITS - 1 - word.leading_zeros() as usize) as i64;
            return Some((word_idx * BITS) as i64 + bit);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// PkIndex — the public struct
// ---------------------------------------------------------------------------

/// A fast primary key index backed by a hybrid bitset + I64Set.
pub struct PkIndex {
    name: String,
    table_name: String,
    column_id: i32,
    column_name: String,
    data: RwLock<PkIndexInner>,
    closed: AtomicBool,
}

impl PkIndex {
    pub fn new(name: String, table_name: String, column_id: i32, column_name: String) -> Self {
        Self {
            name,
            table_name,
            column_id,
            column_name,
            data: RwLock::new(PkIndexInner::new()),
            closed: AtomicBool::new(false),
        }
    }

    /// Extract an i64 from the first `Value` in a slice.
    #[inline]
    fn extract_i64(values: &[Value]) -> Option<i64> {
        values.first().and_then(|v| match v {
            Value::Integer(i) => Some(*i),
            _ => None,
        })
    }

    /// Compute the overflow min, considering both I64Set and the i64::MIN flag.
    #[inline]
    fn overflow_min(inner: &PkIndexInner) -> Option<i64> {
        if inner.has_i64_min {
            // i64::MIN is the absolute minimum — no need to compare with set entries.
            return Some(i64::MIN);
        }
        if !inner.overflow.is_empty() {
            inner.overflow.iter().min()
        } else {
            None
        }
    }

    /// Compute the overflow max, considering both I64Set and the i64::MIN flag.
    #[inline]
    fn overflow_max(inner: &PkIndexInner) -> Option<i64> {
        let set_max = if !inner.overflow.is_empty() {
            inner.overflow.iter().max()
        } else {
            None
        };
        if inner.has_i64_min {
            // i64::MIN is always <= any other value, so it only matters when alone.
            Some(set_max.unwrap_or(i64::MIN))
        } else {
            set_max
        }
    }
}

// ---------------------------------------------------------------------------
// Index trait implementation
// ---------------------------------------------------------------------------

impl Index for PkIndex {
    fn name(&self) -> &str {
        &self.name
    }

    fn table_name(&self) -> &str {
        &self.table_name
    }

    fn build(&mut self) -> Result<()> {
        Ok(())
    }

    fn add(&self, values: &[Value], row_id: i64, _ref_id: i64) -> Result<()> {
        let _ = values;
        let mut inner = self.data.write();
        inner.insert(row_id);
        Ok(())
    }

    fn add_batch(&self, entries: &I64Map<Vec<Value>>) -> Result<()> {
        let mut inner = self.data.write();
        for (row_id, _) in entries.iter() {
            inner.insert(row_id);
        }
        Ok(())
    }

    fn add_batch_slice(&self, entries: &[(i64, &[Value])]) -> Result<()> {
        let mut inner = self.data.write();
        for &(row_id, _) in entries {
            inner.insert(row_id);
        }
        Ok(())
    }

    fn remove(&self, _values: &[Value], row_id: i64, _ref_id: i64) -> Result<()> {
        let mut inner = self.data.write();
        inner.remove(row_id);
        Ok(())
    }

    fn remove_batch(&self, entries: &I64Map<Vec<Value>>) -> Result<()> {
        let mut inner = self.data.write();
        for (row_id, _) in entries.iter() {
            inner.remove(row_id);
        }
        Ok(())
    }

    fn remove_batch_slice(&self, entries: &[(i64, &[Value])]) -> Result<()> {
        let mut inner = self.data.write();
        for &(row_id, _) in entries {
            inner.remove(row_id);
        }
        Ok(())
    }

    fn column_ids(&self) -> &[i32] {
        std::slice::from_ref(&self.column_id)
    }

    fn column_names(&self) -> &[String] {
        std::slice::from_ref(&self.column_name)
    }

    fn data_types(&self) -> &[DataType] {
        &[DataType::Integer]
    }

    fn index_type(&self) -> IndexType {
        IndexType::PrimaryKey
    }

    fn is_unique(&self) -> bool {
        true
    }

    fn find(&self, values: &[Value]) -> Result<Vec<IndexEntry>> {
        let Some(row_id) = Self::extract_i64(values) else {
            return Ok(Vec::new());
        };
        let inner = self.data.read();
        if inner.contains(row_id) {
            Ok(vec![IndexEntry {
                row_id,
                ref_id: row_id,
            }])
        } else {
            Ok(Vec::new())
        }
    }

    fn find_range(
        &self,
        min: &[Value],
        max: &[Value],
        min_inclusive: bool,
        max_inclusive: bool,
    ) -> Result<Vec<IndexEntry>> {
        let min_val = if min.is_empty() {
            None
        } else {
            Self::extract_i64(min)
        };
        let max_val = if max.is_empty() {
            None
        } else {
            Self::extract_i64(max)
        };

        // Compute effective range bounds (use checked arithmetic to detect overflow)
        let lo = match min_val {
            Some(v) if min_inclusive => v,
            Some(v) => match v.checked_add(1) {
                Some(next) => next,
                None => return Ok(Vec::new()), // > i64::MAX is empty
            },
            None => i64::MIN,
        };
        let hi = match max_val {
            Some(v) if max_inclusive => v,
            Some(v) => match v.checked_sub(1) {
                Some(prev) => prev,
                None => return Ok(Vec::new()), // < i64::MIN is empty
            },
            None => i64::MAX,
        };

        if lo > hi {
            return Ok(Vec::new());
        }

        let inner = self.data.read();
        let total_bits = bit_capacity(&inner.words) as i64;

        // Collect from bitset portion
        let bitset_lo = lo.max(0);
        let bitset_hi = hi.min(total_bits - 1);
        let mut result = if total_bits > 0 && bitset_lo <= bitset_hi && bitset_hi >= 0 {
            collect_range_entries(&inner.words, bitset_lo as usize, bitset_hi as usize)
        } else {
            Vec::new()
        };

        // Collect from overflow (I64Set + i64::MIN flag)
        if !inner.overflow_empty() {
            if inner.has_i64_min && lo == i64::MIN {
                result.push(IndexEntry {
                    row_id: i64::MIN,
                    ref_id: i64::MIN,
                });
            }
            for id in inner.overflow.iter() {
                if id >= lo && id <= hi {
                    result.push(IndexEntry {
                        row_id: id,
                        ref_id: id,
                    });
                }
            }
            // Bitset entries are already sorted; overflow entries are not.
            result.sort_unstable_by_key(|e| e.row_id);
        }

        Ok(result)
    }

    fn find_with_operator(&self, op: Operator, values: &[Value]) -> Result<Vec<IndexEntry>> {
        match op {
            Operator::Eq => self.find(values),
            Operator::Gt => self.find_range(values, &[], false, false),
            Operator::Gte => self.find_range(values, &[], true, false),
            Operator::Lt => self.find_range(&[], values, false, false),
            Operator::Lte => self.find_range(&[], values, false, true),
            Operator::Ne => {
                let Some(exclude_id) = Self::extract_i64(values) else {
                    return Ok(Vec::new());
                };
                let inner = self.data.read();
                let mut result = Vec::new();
                for_each_set_bit(&inner.words, |rid| {
                    if rid != exclude_id {
                        result.push(IndexEntry {
                            row_id: rid,
                            ref_id: rid,
                        });
                    }
                    true
                });
                if inner.has_i64_min && i64::MIN != exclude_id {
                    result.push(IndexEntry {
                        row_id: i64::MIN,
                        ref_id: i64::MIN,
                    });
                }
                for id in inner.overflow.iter() {
                    if id != exclude_id {
                        result.push(IndexEntry {
                            row_id: id,
                            ref_id: id,
                        });
                    }
                }
                Ok(result)
            }
            _ => Ok(Vec::new()),
        }
    }

    fn get_row_ids_equal_into(&self, values: &[Value], buffer: &mut Vec<i64>) {
        let Some(row_id) = Self::extract_i64(values) else {
            return;
        };
        let inner = self.data.read();
        if inner.contains(row_id) {
            buffer.push(row_id);
        }
    }

    fn get_row_ids_in_range_into(
        &self,
        min_value: &[Value],
        max_value: &[Value],
        include_min: bool,
        include_max: bool,
        buffer: &mut Vec<i64>,
    ) {
        if let Ok(entries) = self.find_range(min_value, max_value, include_min, include_max) {
            buffer.extend(entries.into_iter().map(|e| e.row_id));
        }
    }

    fn get_row_ids_in_into(&self, value_list: &[Value], buffer: &mut Vec<i64>) {
        let inner = self.data.read();
        for value in value_list {
            if let Value::Integer(row_id) = value {
                if inner.contains(*row_id) {
                    buffer.push(*row_id);
                }
            }
        }
    }

    fn get_filtered_row_ids(&self, expr: &dyn Expression) -> RowIdVec {
        // Try to get comparison info from expression
        if let Some((col_name, operator, value)) = expr.get_comparison_info() {
            if col_name == self.column_name {
                match operator {
                    Operator::Eq => {
                        return self.get_row_ids_equal(std::slice::from_ref(value));
                    }
                    Operator::Gt | Operator::Gte | Operator::Lt | Operator::Lte => {
                        let value_slice = std::slice::from_ref(value);
                        let empty_slice: &[Value] = &[];
                        let (min_vals, max_vals, include_min, include_max) = match operator {
                            Operator::Gt => (value_slice, empty_slice, false, false),
                            Operator::Gte => (value_slice, empty_slice, true, false),
                            Operator::Lt => (empty_slice, value_slice, false, false),
                            Operator::Lte => (empty_slice, value_slice, false, true),
                            _ => return RowIdVec::new(),
                        };
                        return self.get_row_ids_in_range(
                            min_vals,
                            max_vals,
                            include_min,
                            include_max,
                        );
                    }
                    _ => {}
                }
            }
        }

        // Try to extract range from collect_comparisons (for AND expressions)
        let comparisons = expr.collect_comparisons();
        if !comparisons.is_empty() {
            let mut min_val: Option<&Value> = None;
            let mut max_val: Option<&Value> = None;
            let mut include_min = false;
            let mut include_max = false;
            let mut eq_val: Option<&Value> = None;

            for (col_name, op, val) in &comparisons {
                if *col_name != self.column_name {
                    continue;
                }
                match op {
                    Operator::Eq => {
                        eq_val = Some(val);
                    }
                    Operator::Gt => {
                        min_val = Some(val);
                        include_min = false;
                    }
                    Operator::Gte => {
                        min_val = Some(val);
                        include_min = true;
                    }
                    Operator::Lt => {
                        max_val = Some(val);
                        include_max = false;
                    }
                    Operator::Lte => {
                        max_val = Some(val);
                        include_max = true;
                    }
                    _ => {}
                }
            }

            if let Some(eq) = eq_val {
                return self.get_row_ids_equal(std::slice::from_ref(eq));
            }

            if min_val.is_some() || max_val.is_some() {
                let min_slice: &[Value] = min_val.map(std::slice::from_ref).unwrap_or(&[]);
                let max_slice: &[Value] = max_val.map(std::slice::from_ref).unwrap_or(&[]);
                return self.get_row_ids_in_range(min_slice, max_slice, include_min, include_max);
            }
        }

        RowIdVec::new()
    }

    fn get_min_value(&self) -> Option<Value> {
        let inner = self.data.read();
        let b_min = bitset_min(&inner.words);
        let o_min = Self::overflow_min(&inner);
        match (b_min, o_min) {
            (Some(a), Some(b)) => Some(Value::Integer(a.min(b))),
            (Some(a), None) => Some(Value::Integer(a)),
            (None, Some(b)) => Some(Value::Integer(b)),
            (None, None) => None,
        }
    }

    fn get_max_value(&self) -> Option<Value> {
        let inner = self.data.read();
        let b_max = bitset_max(&inner.words);
        let o_max = Self::overflow_max(&inner);
        match (b_max, o_max) {
            (Some(a), Some(b)) => Some(Value::Integer(a.max(b))),
            (Some(a), None) => Some(Value::Integer(a)),
            (None, Some(b)) => Some(Value::Integer(b)),
            (None, None) => None,
        }
    }

    fn get_all_values(&self) -> Vec<Value> {
        let inner = self.data.read();
        let mut result = Vec::with_capacity(inner.count);
        for_each_set_bit(&inner.words, |rid| {
            result.push(Value::Integer(rid));
            true
        });
        if !inner.overflow_empty() {
            if inner.has_i64_min {
                result.push(Value::Integer(i64::MIN));
            }
            let mut overflow_ids: Vec<i64> = inner.overflow.iter().collect();
            overflow_ids.sort_unstable();
            for id in overflow_ids {
                result.push(Value::Integer(id));
            }
        }
        result
    }

    fn get_distinct_count_excluding_null(&self) -> Option<usize> {
        let inner = self.data.read();
        Some(inner.count)
    }

    fn get_row_ids_ordered(
        &self,
        ascending: bool,
        limit: usize,
        offset: usize,
    ) -> Option<Vec<i64>> {
        let inner = self.data.read();

        if inner.overflow_empty() {
            // Fast path: bitset only — iterate in order with offset/limit
            let mut result = Vec::with_capacity(limit.min(inner.count));
            let mut skipped = 0usize;
            if ascending {
                for_each_set_bit(&inner.words, |rid| {
                    if skipped < offset {
                        skipped += 1;
                        return true;
                    }
                    result.push(rid);
                    result.len() < limit
                });
            } else {
                for_each_set_bit_rev(&inner.words, |rid| {
                    if skipped < offset {
                        skipped += 1;
                        return true;
                    }
                    result.push(rid);
                    result.len() < limit
                });
            }
            return Some(result);
        }

        // Collect and sort the (typically tiny) overflow set
        let mut overflow_ids =
            Vec::with_capacity(inner.overflow.len() + inner.has_i64_min as usize);
        if inner.has_i64_min {
            overflow_ids.push(i64::MIN);
        }
        for id in inner.overflow.iter() {
            overflow_ids.push(id);
        }
        overflow_ids.sort_unstable();

        // Streaming merge of sorted bitset + sorted overflow with early termination
        let mut result = Vec::with_capacity(limit.min(inner.count));
        let mut skipped = 0usize;

        if ascending {
            // Merge two ascending iterators
            let mut overflow_pos = 0;
            for_each_set_bit(&inner.words, |rid| {
                // Drain overflow IDs that come before this bitset ID
                while overflow_pos < overflow_ids.len() && overflow_ids[overflow_pos] < rid {
                    if skipped < offset {
                        skipped += 1;
                    } else {
                        result.push(overflow_ids[overflow_pos]);
                        if result.len() >= limit {
                            overflow_pos += 1;
                            return false;
                        }
                    }
                    overflow_pos += 1;
                }
                // Emit this bitset ID
                if skipped < offset {
                    skipped += 1;
                } else {
                    result.push(rid);
                    if result.len() >= limit {
                        return false;
                    }
                }
                true
            });
            // Drain remaining overflow IDs if limit not yet reached
            while overflow_pos < overflow_ids.len() && result.len() < limit {
                if skipped < offset {
                    skipped += 1;
                } else {
                    result.push(overflow_ids[overflow_pos]);
                }
                overflow_pos += 1;
            }
        } else {
            // Descending: merge from the high end
            let mut overflow_pos = overflow_ids.len();
            for_each_set_bit_rev(&inner.words, |rid| {
                // Drain overflow IDs that come after this bitset ID (descending)
                while overflow_pos > 0 && overflow_ids[overflow_pos - 1] > rid {
                    overflow_pos -= 1;
                    if skipped < offset {
                        skipped += 1;
                    } else {
                        result.push(overflow_ids[overflow_pos]);
                        if result.len() >= limit {
                            return false;
                        }
                    }
                }
                // Emit this bitset ID
                if skipped < offset {
                    skipped += 1;
                } else {
                    result.push(rid);
                    if result.len() >= limit {
                        return false;
                    }
                }
                true
            });
            // Drain remaining overflow IDs if limit not yet reached
            while overflow_pos > 0 && result.len() < limit {
                overflow_pos -= 1;
                if skipped < offset {
                    skipped += 1;
                } else {
                    result.push(overflow_ids[overflow_pos]);
                }
            }
        }
        Some(result)
    }

    fn get_grouped_row_ids(&self) -> Option<Vec<(Value, Vec<i64>)>> {
        let inner = self.data.read();
        let mut result = Vec::with_capacity(inner.count);
        for_each_set_bit(&inner.words, |rid| {
            result.push((Value::Integer(rid), vec![rid]));
            true
        });
        if inner.has_i64_min {
            result.push((Value::Integer(i64::MIN), vec![i64::MIN]));
        }
        for id in inner.overflow.iter() {
            result.push((Value::Integer(id), vec![id]));
        }
        Some(result)
    }

    fn for_each_group(
        &self,
        callback: &mut dyn FnMut(&Value, &[i64]) -> Result<bool>,
    ) -> Option<Result<()>> {
        let inner = self.data.read();

        if inner.overflow_empty() {
            // Fast path: bitset only, already in ascending order.
            let mut err: Option<crate::core::Error> = None;
            for_each_set_bit(&inner.words, |rid| {
                let val = Value::Integer(rid);
                match callback(&val, &[rid]) {
                    Ok(true) => true,
                    Ok(false) => false,
                    Err(e) => {
                        err = Some(e);
                        false
                    }
                }
            });
            return Some(match err {
                Some(e) => Err(e),
                None => Ok(()),
            });
        }

        // Slow path: collect and sort overflow, then merge in sorted order.
        let overflow_cap = inner.overflow.len() + if inner.has_i64_min { 1 } else { 0 };
        let mut overflow_sorted: Vec<i64> = Vec::with_capacity(overflow_cap);
        if inner.has_i64_min {
            overflow_sorted.push(i64::MIN);
        }
        overflow_sorted.extend(inner.overflow.iter());
        overflow_sorted.sort_unstable();

        // Phase 1: negative overflow IDs (all < 0, before bitset range)
        for &id in &overflow_sorted {
            if id >= 0 {
                break;
            }
            let val = Value::Integer(id);
            match callback(&val, &[id]) {
                Ok(true) => continue,
                Ok(false) => return Some(Ok(())),
                Err(e) => return Some(Err(e)),
            }
        }

        // Phase 2: bitset IDs [0, BITSET_MAX_BITS) — already ascending
        let mut early_exit = false;
        let mut err: Option<crate::core::Error> = None;
        for_each_set_bit(&inner.words, |rid| {
            let val = Value::Integer(rid);
            match callback(&val, &[rid]) {
                Ok(true) => true,
                Ok(false) => {
                    early_exit = true;
                    false
                }
                Err(e) => {
                    err = Some(e);
                    false
                }
            }
        });
        if let Some(e) = err {
            return Some(Err(e));
        }
        if early_exit {
            return Some(Ok(()));
        }

        // Phase 3: positive overflow IDs (all >= BITSET_MAX_BITS, after bitset)
        for &id in &overflow_sorted {
            if id < 0 {
                continue;
            }
            let val = Value::Integer(id);
            match callback(&val, &[id]) {
                Ok(true) => continue,
                Ok(false) => return Some(Ok(())),
                Err(e) => return Some(Err(e)),
            }
        }

        Some(Ok(()))
    }

    fn clear(&self) -> Result<()> {
        let mut inner = self.data.write();
        inner.clear();
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn close(&mut self) -> Result<()> {
        self.closed.store(true, Ordering::Release);
        self.clear()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_idx() -> PkIndex {
        PkIndex::new(
            "__pk_test_id".to_string(),
            "test".to_string(),
            0,
            "id".to_string(),
        )
    }

    fn count(idx: &PkIndex) -> usize {
        idx.data.read().count
    }

    #[test]
    fn test_pk_index_basic() {
        let idx = make_idx();
        assert!(idx.is_unique());
        assert_eq!(idx.index_type(), IndexType::PrimaryKey);

        // Empty index
        assert!(idx.find(&[Value::Integer(1)]).unwrap().is_empty());

        // Add entries
        idx.add(&[Value::Integer(1)], 1, 1).unwrap();
        idx.add(&[Value::Integer(5)], 5, 5).unwrap();
        idx.add(&[Value::Integer(3)], 3, 3).unwrap();
        assert_eq!(count(&idx), 3);

        // Point lookup
        let results = idx.find(&[Value::Integer(5)]).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].row_id, 5);

        // Missing key
        assert!(idx.find(&[Value::Integer(2)]).unwrap().is_empty());

        // Remove
        idx.remove(&[Value::Integer(5)], 5, 5).unwrap();
        assert!(idx.find(&[Value::Integer(5)]).unwrap().is_empty());
        assert_eq!(count(&idx), 2);
    }

    #[test]
    fn test_pk_index_range() {
        let idx = make_idx();
        for i in 1..=10 {
            idx.add(&[Value::Integer(i)], i, i).unwrap();
        }

        // Range 3..=7
        let results = idx
            .find_range(&[Value::Integer(3)], &[Value::Integer(7)], true, true)
            .unwrap();
        assert_eq!(results.len(), 5);
        assert_eq!(results[0].row_id, 3);
        assert_eq!(results[4].row_id, 7);

        // Range 3 < x < 7 (exclusive)
        let results = idx
            .find_range(&[Value::Integer(3)], &[Value::Integer(7)], false, false)
            .unwrap();
        assert_eq!(results.len(), 3); // 4, 5, 6
    }

    #[test]
    fn test_pk_index_min_max() {
        let idx = make_idx();
        assert!(idx.get_min_value().is_none());
        assert!(idx.get_max_value().is_none());

        idx.add(&[Value::Integer(5)], 5, 5).unwrap();
        idx.add(&[Value::Integer(10)], 10, 10).unwrap();
        idx.add(&[Value::Integer(2)], 2, 2).unwrap();

        assert_eq!(idx.get_min_value(), Some(Value::Integer(2)));
        assert_eq!(idx.get_max_value(), Some(Value::Integer(10)));
    }

    #[test]
    fn test_pk_index_noop_on_duplicate_add() {
        let idx = make_idx();
        idx.add(&[Value::Integer(1)], 1, 1).unwrap();
        idx.add(&[Value::Integer(1)], 1, 1).unwrap(); // duplicate
        assert_eq!(count(&idx), 1);
    }

    #[test]
    fn test_pk_index_batch() {
        let idx = make_idx();
        let entries: Vec<(i64, &[Value])> = vec![
            (1, &[Value::Integer(1)]),
            (2, &[Value::Integer(2)]),
            (3, &[Value::Integer(3)]),
        ];
        idx.add_batch_slice(&entries).unwrap();
        assert_eq!(count(&idx), 3);

        let results = idx.find(&[Value::Integer(2)]).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_pk_index_large_ids() {
        let idx = make_idx();
        // IDs that span multiple words
        for id in [0, 63, 64, 127, 1000] {
            idx.add(&[Value::Integer(id)], id, id).unwrap();
        }
        assert_eq!(count(&idx), 5);

        for id in [0, 63, 64, 127, 1000] {
            assert_eq!(idx.find(&[Value::Integer(id)]).unwrap().len(), 1);
        }
        assert!(idx.find(&[Value::Integer(1)]).unwrap().is_empty());
        assert!(idx.find(&[Value::Integer(65)]).unwrap().is_empty());

        assert_eq!(idx.get_min_value(), Some(Value::Integer(0)));
        assert_eq!(idx.get_max_value(), Some(Value::Integer(1000)));
    }

    #[test]
    fn test_pk_index_ordered() {
        let idx = make_idx();
        for i in [5, 2, 8, 1, 9] {
            idx.add(&[Value::Integer(i)], i, i).unwrap();
        }

        assert_eq!(
            idx.get_row_ids_ordered(true, 100, 0).unwrap(),
            vec![1, 2, 5, 8, 9]
        );
        assert_eq!(
            idx.get_row_ids_ordered(false, 100, 0).unwrap(),
            vec![9, 8, 5, 2, 1]
        );
        assert_eq!(idx.get_row_ids_ordered(true, 2, 1).unwrap(), vec![2, 5]);
    }

    #[test]
    fn test_pk_index_bitset_memory() {
        let idx = make_idx();
        for i in 0..1000 {
            idx.add(&[Value::Integer(i)], i, i).unwrap();
        }
        let inner = idx.data.read();
        // 1000 bits needs ceil(1000/64) = 16 words = 128 bytes
        assert!(inner.words.len() <= 16);
        assert_eq!(inner.count, 1000);
    }

    #[test]
    fn test_pk_index_large_row_ids_overflow() {
        let idx = make_idx();

        // IDs that exceed BITSET_MAX_BITS (10M) go into overflow
        let large_ids = [50_000_000i64, 2_000_000_000, i64::MAX - 1];
        for &id in &large_ids {
            idx.add(&[Value::Integer(id)], id, id).unwrap();
        }
        assert_eq!(count(&idx), 3);

        // Verify find works for overflow entries
        for &id in &large_ids {
            let results = idx.find(&[Value::Integer(id)]).unwrap();
            assert_eq!(results.len(), 1, "find failed for id={id}");
            assert_eq!(results[0].row_id, id);
        }

        // Verify missing ID returns empty
        assert!(idx.find(&[Value::Integer(99_999_999)]).unwrap().is_empty());

        // Verify min/max across overflow
        assert_eq!(idx.get_min_value(), Some(Value::Integer(50_000_000)));
        assert_eq!(idx.get_max_value(), Some(Value::Integer(i64::MAX - 1)));

        // Remove an overflow entry
        idx.remove(
            &[Value::Integer(2_000_000_000)],
            2_000_000_000,
            2_000_000_000,
        )
        .unwrap();
        assert_eq!(count(&idx), 2);
        assert!(idx
            .find(&[Value::Integer(2_000_000_000)])
            .unwrap()
            .is_empty());

        // get_row_ids_equal_into for overflow
        let mut buf = Vec::new();
        idx.get_row_ids_equal_into(&[Value::Integer(50_000_000)], &mut buf);
        assert_eq!(buf, vec![50_000_000]);

        // get_row_ids_in_into for overflow
        buf.clear();
        idx.get_row_ids_in_into(
            &[
                Value::Integer(50_000_000),
                Value::Integer(i64::MAX - 1),
                Value::Integer(999),
            ],
            &mut buf,
        );
        assert_eq!(buf.len(), 2);
        assert!(buf.contains(&50_000_000));
        assert!(buf.contains(&(i64::MAX - 1)));
    }

    #[test]
    fn test_pk_index_hybrid_range() {
        let idx = make_idx();

        // Add IDs in both bitset range and overflow range
        idx.add(&[Value::Integer(5)], 5, 5).unwrap();
        idx.add(&[Value::Integer(100)], 100, 100).unwrap();
        idx.add(&[Value::Integer(20_000_000)], 20_000_000, 20_000_000)
            .unwrap();
        idx.add(&[Value::Integer(50_000_000)], 50_000_000, 50_000_000)
            .unwrap();

        // Range that spans bitset and overflow
        let results = idx
            .find_range(
                &[Value::Integer(0)],
                &[Value::Integer(100_000_000)],
                true,
                true,
            )
            .unwrap();
        let row_ids: Vec<i64> = results.iter().map(|e| e.row_id).collect();
        assert_eq!(row_ids, vec![5, 100, 20_000_000, 50_000_000]);

        // Range only in bitset
        let results = idx
            .find_range(&[Value::Integer(0)], &[Value::Integer(200)], true, true)
            .unwrap();
        assert_eq!(results.len(), 2);

        // Range only in overflow
        let results = idx
            .find_range(
                &[Value::Integer(15_000_000)],
                &[Value::Integer(60_000_000)],
                true,
                true,
            )
            .unwrap();
        let row_ids: Vec<i64> = results.iter().map(|e| e.row_id).collect();
        assert_eq!(row_ids, vec![20_000_000, 50_000_000]);

        // Min/max across hybrid
        assert_eq!(idx.get_min_value(), Some(Value::Integer(5)));
        assert_eq!(idx.get_max_value(), Some(Value::Integer(50_000_000)));

        // Ordered traversal across hybrid
        assert_eq!(
            idx.get_row_ids_ordered(true, 100, 0).unwrap(),
            vec![5, 100, 20_000_000, 50_000_000]
        );
        assert_eq!(
            idx.get_row_ids_ordered(false, 100, 0).unwrap(),
            vec![50_000_000, 20_000_000, 100, 5]
        );
        assert_eq!(
            idx.get_row_ids_ordered(true, 2, 1).unwrap(),
            vec![100, 20_000_000]
        );
    }

    #[test]
    fn test_pk_index_negative_ids_overflow() {
        let idx = make_idx();

        // Negative IDs go to overflow
        idx.add(&[Value::Integer(-1)], -1, -1).unwrap();
        idx.add(&[Value::Integer(-100)], -100, -100).unwrap();
        idx.add(&[Value::Integer(5)], 5, 5).unwrap();

        assert_eq!(count(&idx), 3);

        assert_eq!(idx.find(&[Value::Integer(-1)]).unwrap().len(), 1);
        assert_eq!(idx.find(&[Value::Integer(-100)]).unwrap().len(), 1);
        assert!(idx.find(&[Value::Integer(-50)]).unwrap().is_empty());

        assert_eq!(idx.get_min_value(), Some(Value::Integer(-100)));
        assert_eq!(idx.get_max_value(), Some(Value::Integer(5)));

        // Remove negative
        idx.remove(&[Value::Integer(-1)], -1, -1).unwrap();
        assert_eq!(count(&idx), 2);
        assert!(idx.find(&[Value::Integer(-1)]).unwrap().is_empty());
    }

    #[test]
    fn test_pk_index_i64_min() {
        let idx = make_idx();

        // i64::MIN is the I64Set sentinel — must be handled separately
        idx.add(&[Value::Integer(i64::MIN)], i64::MIN, i64::MIN)
            .unwrap();
        idx.add(&[Value::Integer(0)], 0, 0).unwrap();
        idx.add(&[Value::Integer(100)], 100, 100).unwrap();
        assert_eq!(count(&idx), 3);

        // Find i64::MIN
        assert_eq!(idx.find(&[Value::Integer(i64::MIN)]).unwrap().len(), 1);
        assert_eq!(
            idx.find(&[Value::Integer(i64::MIN)]).unwrap()[0].row_id,
            i64::MIN
        );

        // Duplicate add is idempotent
        idx.add(&[Value::Integer(i64::MIN)], i64::MIN, i64::MIN)
            .unwrap();
        assert_eq!(count(&idx), 3);

        // Min/max
        assert_eq!(idx.get_min_value(), Some(Value::Integer(i64::MIN)));
        assert_eq!(idx.get_max_value(), Some(Value::Integer(100)));

        // get_row_ids_in_into
        let mut buf = Vec::new();
        idx.get_row_ids_in_into(
            &[
                Value::Integer(i64::MIN),
                Value::Integer(0),
                Value::Integer(999),
            ],
            &mut buf,
        );
        assert_eq!(buf.len(), 2);
        assert!(buf.contains(&i64::MIN));
        assert!(buf.contains(&0));

        // Range including i64::MIN
        let results = idx
            .find_range(&[], &[Value::Integer(0)], false, true)
            .unwrap();
        let row_ids: Vec<i64> = results.iter().map(|e| e.row_id).collect();
        assert!(row_ids.contains(&i64::MIN));
        assert!(row_ids.contains(&0));

        // Remove i64::MIN
        idx.remove(&[Value::Integer(i64::MIN)], i64::MIN, i64::MIN)
            .unwrap();
        assert_eq!(count(&idx), 2);
        assert!(idx.find(&[Value::Integer(i64::MIN)]).unwrap().is_empty());

        // Double remove is no-op
        idx.remove(&[Value::Integer(i64::MIN)], i64::MIN, i64::MIN)
            .unwrap();
        assert_eq!(count(&idx), 2);
    }

    #[test]
    fn test_pk_index_overflow_batch_ops() {
        let idx = make_idx();

        // Batch add with mixed bitset and overflow entries
        let entries: Vec<(i64, &[Value])> = vec![
            (1, &[Value::Integer(1)]),
            (100_000_000, &[Value::Integer(100_000_000)]),
            (5, &[Value::Integer(5)]),
        ];
        idx.add_batch_slice(&entries).unwrap();
        assert_eq!(count(&idx), 3);

        assert_eq!(idx.find(&[Value::Integer(1)]).unwrap().len(), 1);
        assert_eq!(idx.find(&[Value::Integer(100_000_000)]).unwrap().len(), 1);
        assert_eq!(idx.find(&[Value::Integer(5)]).unwrap().len(), 1);

        // Batch remove
        let remove_entries: Vec<(i64, &[Value])> = vec![
            (100_000_000, &[Value::Integer(100_000_000)]),
            (5, &[Value::Integer(5)]),
        ];
        idx.remove_batch_slice(&remove_entries).unwrap();
        assert_eq!(count(&idx), 1);
        assert!(idx.find(&[Value::Integer(100_000_000)]).unwrap().is_empty());
        assert!(idx.find(&[Value::Integer(5)]).unwrap().is_empty());
        assert_eq!(idx.find(&[Value::Integer(1)]).unwrap().len(), 1);
    }

    #[test]
    fn test_pk_index_clear_with_overflow() {
        let idx = make_idx();

        idx.add(&[Value::Integer(1)], 1, 1).unwrap();
        idx.add(&[Value::Integer(50_000_000)], 50_000_000, 50_000_000)
            .unwrap();
        idx.add(&[Value::Integer(i64::MIN)], i64::MIN, i64::MIN)
            .unwrap();
        assert_eq!(count(&idx), 3);

        idx.clear().unwrap();
        assert_eq!(count(&idx), 0);
        assert!(idx.find(&[Value::Integer(1)]).unwrap().is_empty());
        assert!(idx.find(&[Value::Integer(50_000_000)]).unwrap().is_empty());
        assert!(idx.find(&[Value::Integer(i64::MIN)]).unwrap().is_empty());
        assert!(idx.get_min_value().is_none());
        assert!(idx.get_max_value().is_none());
    }

    #[test]
    fn test_pk_index_count_always_accurate() {
        let idx = make_idx();

        // Add, duplicate add, remove, double remove — count must stay exact
        idx.add(&[Value::Integer(1)], 1, 1).unwrap();
        assert_eq!(count(&idx), 1);

        idx.add(&[Value::Integer(1)], 1, 1).unwrap(); // dup
        assert_eq!(count(&idx), 1);

        idx.add(&[Value::Integer(50_000_000)], 50_000_000, 50_000_000)
            .unwrap();
        assert_eq!(count(&idx), 2);

        idx.add(&[Value::Integer(50_000_000)], 50_000_000, 50_000_000)
            .unwrap(); // dup
        assert_eq!(count(&idx), 2);

        idx.remove(&[Value::Integer(1)], 1, 1).unwrap();
        assert_eq!(count(&idx), 1);

        idx.remove(&[Value::Integer(1)], 1, 1).unwrap(); // already gone
        assert_eq!(count(&idx), 1);

        idx.remove(&[Value::Integer(50_000_000)], 50_000_000, 50_000_000)
            .unwrap();
        assert_eq!(count(&idx), 0);

        idx.remove(&[Value::Integer(50_000_000)], 50_000_000, 50_000_000)
            .unwrap(); // already gone
        assert_eq!(count(&idx), 0);
    }

    #[test]
    fn test_pk_index_for_each_group_sorted_order() {
        let idx = make_idx();

        // Add IDs spanning negative overflow, bitset, and positive overflow
        idx.add(&[Value::Integer(i64::MIN)], i64::MIN, i64::MIN)
            .unwrap();
        idx.add(&[Value::Integer(-50)], -50, -50).unwrap();
        idx.add(&[Value::Integer(-1)], -1, -1).unwrap();
        idx.add(&[Value::Integer(0)], 0, 0).unwrap();
        idx.add(&[Value::Integer(100)], 100, 100).unwrap();
        idx.add(&[Value::Integer(9_999_999)], 9_999_999, 9_999_999)
            .unwrap();
        idx.add(&[Value::Integer(20_000_000)], 20_000_000, 20_000_000)
            .unwrap();
        idx.add(&[Value::Integer(100_000_000)], 100_000_000, 100_000_000)
            .unwrap();

        // for_each_group must yield values in ascending sorted order
        let mut collected = Vec::new();
        idx.for_each_group(&mut |val, row_ids| {
            if let Value::Integer(id) = val {
                collected.push(*id);
                assert_eq!(row_ids.len(), 1);
                assert_eq!(row_ids[0], *id);
            }
            Ok(true)
        })
        .unwrap()
        .unwrap();

        assert_eq!(
            collected,
            vec![
                i64::MIN,
                -50,
                -1,
                0,
                100,
                9_999_999,
                20_000_000,
                100_000_000
            ]
        );
    }
}
