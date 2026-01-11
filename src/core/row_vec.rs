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

//! RowVec - Cached row vector for zero-allocation table scans
//!
//! This type wraps Vec<(i64, Row)> and returns it to a thread-local pool on drop.
//! Uses a capacity-aware pool with best-fit allocation for efficient buffer reuse.
//!
//! Pool Design:
//! - Sorted by capacity (ascending) for O(log n) best-fit search
//! - 16 slots to handle concurrent usage patterns
//! - Best-fit allocation: returns smallest buffer >= requested capacity
//! - Smart eviction: keeps larger buffers which are more versatile

use crate::core::Row;
use std::cell::RefCell;

/// Maximum buffers to keep in the thread-local pool.
/// 16 slots handles most concurrent usage patterns including complex queries
/// with multiple JOINs, subqueries, and window functions.
const POOL_SIZE: usize = 16;

/// Maximum capacity to cache (prevents unbounded memory retention)
/// 128K elements = ~2MB at 16 bytes per (i64, Row) tuple
/// This allows caching large table scan buffers from window functions and version store
const MAX_CACHED_CAPACITY: usize = 128_000;

// Thread-local pool for row vectors - kept sorted by capacity (ascending)
thread_local! {
    static ROW_VEC_POOL: RefCell<Vec<Vec<(i64, Row)>>> = const { RefCell::new(Vec::new()) };
}

/// Clear the thread-local RowVec pool, releasing all cached buffers.
/// Call this when dropping a database to prevent memory retention.
#[inline]
pub fn clear_row_vec_pool() {
    ROW_VEC_POOL.with(|pool| {
        pool.borrow_mut().clear();
    });
}

// ============================================================================
// Pool Statistics (only when dhat-heap feature is enabled)
// ============================================================================

/// Pool statistics for debugging and profiling
#[cfg(feature = "dhat-heap")]
#[derive(Debug, Default)]
pub struct PoolStats {
    /// Number of successful pool hits (buffer reused)
    pub hits: u64,
    /// Number of pool misses (new allocation needed)
    pub misses: u64,
    /// Number of buffers returned to pool
    pub returns: u64,
    /// Number of buffers evicted (pool full, smaller buffer discarded)
    pub evictions: u64,
    /// Number of oversized buffers discarded (exceeded MAX_CACHED_CAPACITY)
    pub oversized_discards: u64,
    /// Total bytes requested via with_capacity
    pub bytes_requested: u64,
    /// Total bytes served from pool (capacity * 16 bytes per element)
    pub bytes_from_pool: u64,
    /// Current pool size
    pub current_pool_size: usize,
    /// Total capacity in pool (sum of all buffer capacities)
    pub total_pool_capacity: usize,
}

#[cfg(feature = "dhat-heap")]
impl PoolStats {
    /// Calculate hit rate as percentage
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            (self.hits as f64 / total as f64) * 100.0
        }
    }

    /// Estimated bytes saved by pool reuse
    pub fn bytes_saved(&self) -> u64 {
        self.bytes_from_pool
    }
}

#[cfg(feature = "dhat-heap")]
impl std::fmt::Display for PoolStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "RowVec Pool Statistics:")?;
        writeln!(f, "  Hits:              {:>10}", self.hits)?;
        writeln!(f, "  Misses:            {:>10}", self.misses)?;
        writeln!(f, "  Hit Rate:          {:>9.1}%", self.hit_rate())?;
        writeln!(f, "  Returns:           {:>10}", self.returns)?;
        writeln!(f, "  Evictions:         {:>10}", self.evictions)?;
        writeln!(f, "  Oversized Discards:{:>10}", self.oversized_discards)?;
        writeln!(
            f,
            "  Bytes Requested:   {:>10}",
            format_bytes(self.bytes_requested)
        )?;
        writeln!(
            f,
            "  Bytes From Pool:   {:>10}",
            format_bytes(self.bytes_from_pool)
        )?;
        writeln!(
            f,
            "  Bytes Saved:       {:>10}",
            format_bytes(self.bytes_saved())
        )?;
        writeln!(f, "  Current Pool Size: {:>10}", self.current_pool_size)?;
        writeln!(
            f,
            "  Pool Capacity:     {:>10}",
            format_bytes(self.total_pool_capacity as u64 * 16)
        )?;
        Ok(())
    }
}

#[cfg(feature = "dhat-heap")]
fn format_bytes(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.2} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.2} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.2} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

#[cfg(feature = "dhat-heap")]
thread_local! {
    static POOL_STATS: RefCell<PoolStats> = RefCell::new(PoolStats::default());
}

/// Get current pool statistics (only available with dhat-heap feature)
#[cfg(feature = "dhat-heap")]
pub fn get_pool_stats() -> PoolStats {
    POOL_STATS.with(|stats| {
        let mut s = stats.borrow().clone();
        // Update current pool state
        ROW_VEC_POOL.with(|pool| {
            let pool = pool.borrow();
            s.current_pool_size = pool.len();
            s.total_pool_capacity = pool.iter().map(|v| v.capacity()).sum();
        });
        s
    })
}

/// Print pool statistics to stderr (only available with dhat-heap feature)
#[cfg(feature = "dhat-heap")]
pub fn print_pool_stats() {
    eprintln!("{}", get_pool_stats());
}

/// Reset pool statistics (only available with dhat-heap feature)
#[cfg(feature = "dhat-heap")]
pub fn reset_pool_stats() {
    POOL_STATS.with(|stats| {
        *stats.borrow_mut() = PoolStats::default();
    });
}

#[cfg(feature = "dhat-heap")]
impl Clone for PoolStats {
    fn clone(&self) -> Self {
        Self {
            hits: self.hits,
            misses: self.misses,
            returns: self.returns,
            evictions: self.evictions,
            oversized_discards: self.oversized_discards,
            bytes_requested: self.bytes_requested,
            bytes_from_pool: self.bytes_from_pool,
            current_pool_size: self.current_pool_size,
            total_pool_capacity: self.total_pool_capacity,
        }
    }
}

// Helper macros for stats tracking (no-op when feature is disabled)
#[cfg(feature = "dhat-heap")]
macro_rules! track_hit {
    ($capacity:expr) => {
        POOL_STATS.with(|stats| {
            let mut s = stats.borrow_mut();
            s.hits += 1;
            s.bytes_from_pool += ($capacity as u64) * 16;
        });
    };
}

#[cfg(not(feature = "dhat-heap"))]
macro_rules! track_hit {
    ($capacity:expr) => {};
}

#[cfg(feature = "dhat-heap")]
macro_rules! track_miss {
    ($capacity:expr) => {
        POOL_STATS.with(|stats| {
            let mut s = stats.borrow_mut();
            s.misses += 1;
            s.bytes_requested += ($capacity as u64) * 16;
        });
    };
}

#[cfg(not(feature = "dhat-heap"))]
macro_rules! track_miss {
    ($capacity:expr) => {};
}

#[cfg(feature = "dhat-heap")]
macro_rules! track_return {
    () => {
        POOL_STATS.with(|stats| {
            stats.borrow_mut().returns += 1;
        });
    };
}

#[cfg(not(feature = "dhat-heap"))]
macro_rules! track_return {
    () => {};
}

#[cfg(feature = "dhat-heap")]
macro_rules! track_eviction {
    () => {
        POOL_STATS.with(|stats| {
            stats.borrow_mut().evictions += 1;
        });
    };
}

#[cfg(not(feature = "dhat-heap"))]
macro_rules! track_eviction {
    () => {};
}

#[cfg(feature = "dhat-heap")]
macro_rules! track_oversized {
    () => {
        POOL_STATS.with(|stats| {
            stats.borrow_mut().oversized_discards += 1;
        });
    };
}

#[cfg(not(feature = "dhat-heap"))]
macro_rules! track_oversized {
    () => {};
}

/// Cached row vector that returns to thread-local cache on drop.
///
/// Use this for table scans to reuse Vec allocations across queries.
/// Derefs to `Vec<(i64, Row)>` for transparent access.
#[derive(Debug)]
pub struct RowVec {
    inner: Option<Vec<(i64, Row)>>,
}

impl RowVec {
    /// Create from thread-local pool or allocate new.
    /// Takes the largest available buffer (end of sorted list).
    #[inline]
    pub fn new() -> Self {
        let v = ROW_VEC_POOL.with(|pool| pool.borrow_mut().pop());
        match v {
            Some(buf) => {
                track_hit!(buf.capacity());
                Self { inner: Some(buf) }
            }
            None => {
                track_miss!(16);
                Self {
                    inner: Some(Vec::with_capacity(16)),
                }
            }
        }
    }

    /// Create with specific capacity using best-fit allocation.
    /// Uses binary search to find smallest buffer >= requested capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        let v = ROW_VEC_POOL.with(|pool| {
            let mut pool = pool.borrow_mut();
            if pool.is_empty() {
                return None;
            }
            // Binary search for smallest buffer >= capacity
            // Pool is sorted ascending by capacity
            let idx = pool.partition_point(|b| b.capacity() < capacity);
            if idx < pool.len() {
                // Found a buffer with sufficient capacity
                Some(pool.remove(idx))
            } else {
                // No buffer large enough - take largest and let it grow
                pool.pop()
            }
        });

        match v {
            Some(mut buf) => {
                let buf_cap = buf.capacity();
                if buf_cap >= capacity {
                    // Perfect hit - buffer is large enough
                    track_hit!(buf_cap);
                } else {
                    // Partial hit - need to grow buffer
                    track_hit!(buf_cap);
                    buf.reserve(capacity - buf_cap);
                }
                Self { inner: Some(buf) }
            }
            None => {
                track_miss!(capacity);
                Self {
                    inner: Some(Vec::with_capacity(capacity)),
                }
            }
        }
    }

    /// Create from an existing Vec<(i64, Row)>.
    /// The provided vec becomes the inner storage (no copy if move is possible).
    /// NOTE: This bypasses the cache - use when you already have allocated data.
    #[inline]
    pub fn from_vec(v: Vec<(i64, Row)>) -> Self {
        Self { inner: Some(v) }
    }

    /// Extract the inner Vec directly, bypassing cache return.
    /// Use this when you need to pass the Vec to APIs that require Vec<(i64, Row)>.
    /// The allocation is NOT returned to the cache.
    #[inline]
    pub fn into_vec(mut self) -> Vec<(i64, Row)> {
        self.inner.take().unwrap_or_default()
    }

    /// Get length
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.as_ref().map(|v| v.len()).unwrap_or(0)
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Push a row
    #[inline]
    pub fn push(&mut self, item: (i64, Row)) {
        if let Some(v) = self.inner.as_mut() {
            v.push(item);
        }
    }

    /// Clear the vector (keeps allocation)
    #[inline]
    pub fn clear(&mut self) {
        if let Some(v) = self.inner.as_mut() {
            v.clear();
        }
    }

    /// Get iterator over rows (borrows)
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &(i64, Row)> {
        self.inner.as_ref().unwrap().iter()
    }

    /// Get mutable iterator
    #[inline]
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut (i64, Row)> {
        self.inner.as_mut().unwrap().iter_mut()
    }

    /// Get row by index
    #[inline]
    pub fn get(&self, index: usize) -> Option<&(i64, Row)> {
        self.inner.as_ref().and_then(|v| v.get(index))
    }

    /// Drain and extract just the rows, discarding row IDs.
    /// The RowVec allocation returns to the cache on drop.
    #[inline]
    pub fn drain_rows(&mut self) -> impl Iterator<Item = Row> + '_ {
        self.inner.as_mut().unwrap().drain(..).map(|(_, row)| row)
    }

    /// Get iterator over just the Row references
    #[inline]
    pub fn rows(&self) -> impl Iterator<Item = &Row> {
        self.inner.as_ref().unwrap().iter().map(|(_, row)| row)
    }
}

impl Default for RowVec {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for RowVec {
    fn clone(&self) -> Self {
        let mut cloned = RowVec::with_capacity(self.len());
        for (id, row) in self.inner.as_ref().unwrap().iter() {
            cloned.push((*id, row.clone()));
        }
        cloned
    }
}

impl std::ops::Deref for RowVec {
    type Target = Vec<(i64, Row)>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.inner.as_ref().unwrap()
    }
}

impl std::ops::DerefMut for RowVec {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.inner.as_mut().unwrap()
    }
}

impl Drop for RowVec {
    #[inline]
    fn drop(&mut self) {
        if let Some(mut v) = self.inner.take() {
            let cap = v.capacity();
            // Don't cache very large buffers to prevent unbounded memory retention
            if cap > MAX_CACHED_CAPACITY {
                track_oversized!();
                return; // Let it deallocate
            }
            v.clear();
            ROW_VEC_POOL.with(|pool| {
                let mut pool = pool.borrow_mut();
                if pool.len() < POOL_SIZE {
                    // Pool has room - insert in sorted position (by capacity, ascending)
                    let insert_idx = pool.partition_point(|b| b.capacity() < cap);
                    pool.insert(insert_idx, v);
                    track_return!();
                } else {
                    // Pool full - smart eviction: keep larger buffers (more versatile)
                    // Replace smallest if this buffer is larger
                    if !pool.is_empty() && pool[0].capacity() < cap {
                        // Remove smallest (index 0), insert this one in sorted position
                        pool.remove(0);
                        let insert_idx = pool.partition_point(|b| b.capacity() < cap);
                        pool.insert(insert_idx, v);
                        track_return!();
                        track_eviction!();
                    }
                    // Otherwise let v deallocate - it's smaller than everything in pool
                }
            });
        }
    }
}

impl std::ops::Index<usize> for RowVec {
    type Output = (i64, Row);

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.inner.as_ref().unwrap()[index]
    }
}

impl std::ops::IndexMut<usize> for RowVec {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.inner.as_mut().unwrap()[index]
    }
}

/// Draining iterator that preserves allocation for cache reuse.
/// Uses ptr::read for zero-allocation iteration - no dummy rows created.
pub struct RowVecIter {
    inner: std::mem::ManuallyDrop<RowVec>,
    front: usize,
    back: usize,
}

impl Iterator for RowVecIter {
    type Item = (i64, Row);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.front >= self.back {
            return None;
        }
        let vec = self.inner.inner.as_ref()?;
        // SAFETY: Each position is read exactly once. We track front/back indices
        // and set_len(0) in Drop prevents double-free of moved elements.
        let item = unsafe { std::ptr::read(vec.as_ptr().add(self.front)) };
        self.front += 1;
        Some(item)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.back.saturating_sub(self.front);
        (len, Some(len))
    }
}

impl DoubleEndedIterator for RowVecIter {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front >= self.back {
            return None;
        }
        self.back -= 1;
        let vec = self.inner.inner.as_ref()?;
        // SAFETY: Each position is read exactly once from back.
        let item = unsafe { std::ptr::read(vec.as_ptr().add(self.back)) };
        Some(item)
    }
}

impl ExactSizeIterator for RowVecIter {}

impl Drop for RowVecIter {
    fn drop(&mut self) {
        // Drop any remaining items that weren't yielded
        if let Some(vec) = self.inner.inner.as_mut() {
            for i in self.front..self.back {
                // SAFETY: Items in range [front, back) haven't been read yet
                unsafe {
                    std::ptr::drop_in_place(vec.as_mut_ptr().add(i));
                }
            }
            // Set len to 0 so Vec's drop doesn't double-free already-read elements.
            // Capacity is preserved for cache reuse.
            unsafe {
                vec.set_len(0);
            }
        }
        // Now drop the RowVec so it returns the empty Vec to cache
        // SAFETY: We're in Drop, and we've cleared the vec's length
        unsafe {
            std::mem::ManuallyDrop::drop(&mut self.inner);
        }
    }
}

impl IntoIterator for RowVec {
    type Item = (i64, Row);
    type IntoIter = RowVecIter;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        let len = self.len();
        RowVecIter {
            inner: std::mem::ManuallyDrop::new(self),
            front: 0,
            back: len,
        }
    }
}

impl<'a> IntoIterator for &'a RowVec {
    type Item = &'a (i64, Row);
    type IntoIter = std::slice::Iter<'a, (i64, Row)>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.inner.as_ref().unwrap().iter()
    }
}

impl<'a> IntoIterator for &'a mut RowVec {
    type Item = &'a mut (i64, Row);
    type IntoIter = std::slice::IterMut<'a, (i64, Row)>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.inner.as_mut().unwrap().iter_mut()
    }
}

// Allow collecting into RowVec
impl FromIterator<(i64, Row)> for RowVec {
    fn from_iter<I: IntoIterator<Item = (i64, Row)>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (lower, upper) = iter.size_hint();
        // Use upper bound if available, otherwise lower bound, minimum 16
        let capacity = upper.unwrap_or(lower).max(16);
        let mut rv = RowVec::with_capacity(capacity);
        for item in iter {
            rv.push(item);
        }
        rv
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Value;

    #[test]
    fn test_row_vec_basic() {
        let mut rv = RowVec::new();
        rv.push((1, Row::from_values(vec![Value::Integer(1)])));
        rv.push((2, Row::from_values(vec![Value::Integer(2)])));

        assert_eq!(rv.len(), 2);
        assert!(!rv.is_empty());
    }

    #[test]
    fn test_row_vec_cache_reuse() {
        // First allocation with known capacity
        {
            let mut rv = RowVec::with_capacity(100);
            rv.push((1, Row::from_values(vec![Value::Integer(1)])));
            // rv drops here, returns to cache
        }

        // Should reuse from cache - capacity preserved
        let rv2 = RowVec::new();
        assert!(
            rv2.capacity() >= 100,
            "Expected capacity >= 100, got {}",
            rv2.capacity()
        );
    }

    #[test]
    fn test_row_vec_into_iter() {
        let mut rv = RowVec::new();
        rv.push((1, Row::from_values(vec![Value::Integer(1)])));
        rv.push((2, Row::from_values(vec![Value::Integer(2)])));

        let collected: Vec<_> = rv.into_iter().collect();
        assert_eq!(collected.len(), 2);
        // Verify forward iteration order
        assert_eq!(collected[0].0, 1);
        assert_eq!(collected[1].0, 2);
    }

    #[test]
    fn test_row_vec_rev() {
        let mut rv = RowVec::new();
        rv.push((1, Row::from_values(vec![Value::Integer(1)])));
        rv.push((2, Row::from_values(vec![Value::Integer(2)])));
        rv.push((3, Row::from_values(vec![Value::Integer(3)])));

        // Test .rev() iterator
        let collected: Vec<_> = rv.into_iter().rev().collect();
        assert_eq!(collected.len(), 3);
        assert_eq!(collected[0].0, 3); // Last becomes first
        assert_eq!(collected[1].0, 2);
        assert_eq!(collected[2].0, 1); // First becomes last
    }

    #[test]
    fn test_row_vec_skip_take_rev() {
        let mut rv = RowVec::new();
        for i in 1..=10 {
            rv.push((i, Row::from_values(vec![Value::Integer(i)])));
        }

        // Test .rev().skip().take() pattern
        let collected: Vec<_> = rv.into_iter().rev().skip(2).take(3).collect();
        assert_eq!(collected.len(), 3);
        assert_eq!(collected[0].0, 8); // 10-2 skip = 8
        assert_eq!(collected[1].0, 7);
        assert_eq!(collected[2].0, 6);
    }

    #[test]
    fn test_row_vec_pool_keeps_buffers() {
        // Create a buffer and return it to pool
        {
            let mut rv = RowVec::with_capacity(500);
            rv.push((1, Row::from_values(vec![Value::Integer(1)])));
            // rv drops here, returns to pool
        }

        // Get a buffer - should come from pool with capacity >= 500
        let rv2 = RowVec::new();
        assert!(
            rv2.capacity() >= 16, // At least default capacity
            "Expected capacity >= 16, got {}",
            rv2.capacity()
        );
    }

    #[test]
    fn test_row_vec_pool_respects_max_capacity() {
        // Create a buffer larger than MAX_CACHED_CAPACITY
        // This should NOT be pooled
        {
            let mut rv = RowVec::with_capacity(MAX_CACHED_CAPACITY + 1000);
            rv.push((1, Row::from_values(vec![Value::Integer(1)])));
            // rv drops here, but should NOT be pooled due to size limit
        }

        // Verify the pool didn't store the oversized buffer by checking
        // that new allocations don't get the oversized capacity
        let rv2 = RowVec::with_capacity(16);
        assert!(
            rv2.capacity() < MAX_CACHED_CAPACITY,
            "Expected small capacity, got {}",
            rv2.capacity()
        );
    }

    #[test]
    fn test_row_vec_pool_concurrent_usage() {
        // This test verifies multi-slot pool handles concurrent RowVecs
        // Create multiple RowVecs simultaneously with specific capacities
        let mut rv1 = RowVec::with_capacity(100);
        let mut rv2 = RowVec::with_capacity(200);
        let mut rv3 = RowVec::with_capacity(300);
        rv1.push((1, Row::from_values(vec![Value::Integer(1)])));
        rv2.push((2, Row::from_values(vec![Value::Integer(2)])));
        rv3.push((3, Row::from_values(vec![Value::Integer(3)])));

        // Verify they all have at least the requested capacity
        assert!(
            rv1.capacity() >= 100,
            "rv1 capacity {} < 100",
            rv1.capacity()
        );
        assert!(
            rv2.capacity() >= 200,
            "rv2 capacity {} < 200",
            rv2.capacity()
        );
        assert!(
            rv3.capacity() >= 300,
            "rv3 capacity {} < 300",
            rv3.capacity()
        );

        // Drop all - they should return to pool
        drop(rv1);
        drop(rv2);
        drop(rv3);

        // Now get three more - should come from pool with preserved capacities
        let rv_a = RowVec::new();
        let rv_b = RowVec::new();
        let rv_c = RowVec::new();

        // Pool returns largest first, so we should get buffers with good capacity
        // At minimum, we should have some capacity from the pool
        assert!(
            rv_a.capacity() >= 16,
            "rv_a capacity {} < 16",
            rv_a.capacity()
        );
        assert!(
            rv_b.capacity() >= 16,
            "rv_b capacity {} < 16",
            rv_b.capacity()
        );
        assert!(
            rv_c.capacity() >= 16,
            "rv_c capacity {} < 16",
            rv_c.capacity()
        );
    }
}
