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

//! Self-tuning buffer pool for efficient memory reuse
//!
//!
//! This module provides a buffer pool that automatically tunes its buffer sizes
//! based on observed usage patterns. This reduces memory allocation overhead
//! and improves performance for I/O operations.

use crossbeam::queue::ArrayQueue;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Default number of buffers in the pool
const DEFAULT_POOL_SIZE: usize = 64;

/// Default calibration samples to collect before tuning
const DEFAULT_MAX_SAMPLES: usize = 100;

/// Calibration threshold - tune after this many samples
const CALIBRATION_THRESHOLD: usize = 50;

/// Self-tuning buffer pool for efficient memory reuse
///
/// The pool automatically adjusts buffer sizes based on observed usage patterns
/// via the PoolCalibrator. Buffers are reused to reduce allocation overhead.
///
/// # Example
/// ```
/// use stoolap::common::BufferPool;
///
/// let pool = BufferPool::new(4096, 1024 * 1024, "test");
///
/// // Get a buffer from the pool
/// let mut buf = pool.get();
/// buf.extend_from_slice(b"hello world");
///
/// // Return the buffer to the pool for reuse
/// pool.put(buf);
/// ```
pub struct BufferPool {
    /// Queue of available buffers
    pool: ArrayQueue<Vec<u8>>,
    /// Calibrator for auto-tuning
    calibrator: Arc<Mutex<PoolCalibrator>>,
    /// Default size for new buffers
    default_size: AtomicUsize,
    /// Maximum allowed buffer size
    max_size: usize,
    /// Name of the pool (for debugging)
    pool_name: String,
    /// Total buffers created (for statistics)
    buffers_created: AtomicUsize,
    /// Total get operations
    get_count: AtomicUsize,
    /// Total put operations
    put_count: AtomicUsize,
}

impl BufferPool {
    /// Create a new buffer pool
    ///
    /// # Arguments
    /// * `default_size` - Initial size for allocated buffers
    /// * `max_size` - Maximum buffer size (buffers larger than this are discarded)
    /// * `name` - Name of the pool (for debugging/logging)
    pub fn new(default_size: usize, max_size: usize, name: &str) -> Self {
        Self::with_pool_size(default_size, max_size, name, DEFAULT_POOL_SIZE)
    }

    /// Create a new buffer pool with specified pool size
    pub fn with_pool_size(
        default_size: usize,
        max_size: usize,
        name: &str,
        pool_size: usize,
    ) -> Self {
        Self {
            pool: ArrayQueue::new(pool_size),
            calibrator: Arc::new(Mutex::new(PoolCalibrator::new(DEFAULT_MAX_SAMPLES))),
            default_size: AtomicUsize::new(default_size),
            max_size,
            pool_name: name.to_string(),
            buffers_created: AtomicUsize::new(0),
            get_count: AtomicUsize::new(0),
            put_count: AtomicUsize::new(0),
        }
    }

    /// Get a buffer from the pool
    ///
    /// Returns a buffer from the pool if available, otherwise creates a new one.
    /// The buffer is cleared before returning.
    pub fn get(&self) -> Vec<u8> {
        self.get_count.fetch_add(1, Ordering::Relaxed);

        match self.pool.pop() {
            Some(mut buf) => {
                buf.clear();
                buf
            }
            None => {
                self.buffers_created.fetch_add(1, Ordering::Relaxed);
                Vec::with_capacity(self.default_size.load(Ordering::Relaxed))
            }
        }
    }

    /// Get a buffer with at least the specified capacity
    pub fn get_with_capacity(&self, capacity: usize) -> Vec<u8> {
        self.get_count.fetch_add(1, Ordering::Relaxed);

        match self.pool.pop() {
            Some(mut buf) => {
                buf.clear();
                // After clear(), len=0, so reserve(n) ensures capacity >= n
                buf.reserve(capacity);
                buf
            }
            None => {
                self.buffers_created.fetch_add(1, Ordering::Relaxed);
                let size = capacity.max(self.default_size.load(Ordering::Relaxed));
                Vec::with_capacity(size)
            }
        }
    }

    /// Return a buffer to the pool
    ///
    /// The buffer is cleared and returned to the pool for reuse.
    /// Buffers larger than max_size are discarded to prevent memory bloat.
    pub fn put(&self, mut buf: Vec<u8>) {
        self.put_count.fetch_add(1, Ordering::Relaxed);

        // Record the size for calibration
        self.record_size(buf.capacity());

        // Don't keep oversized buffers
        if buf.capacity() > self.max_size {
            return;
        }

        // Clear the buffer before returning
        buf.clear();

        // Try to return to pool (ignore if full)
        let _ = self.pool.push(buf);
    }

    /// Record a buffer size for calibration
    ///
    /// This is called automatically during put(), but can also be called
    /// manually to record sizes without returning a buffer.
    pub fn record_size(&self, size: usize) {
        let mut calibrator = self.calibrator.lock();
        calibrator.record_sample(size);

        // Check if we should recalibrate
        if calibrator.should_calibrate() {
            let optimal = calibrator.calculate_optimal_size();
            if optimal > 0 {
                self.default_size.store(optimal, Ordering::Relaxed);
            }
            calibrator.mark_calibrated();
        }
    }

    /// Get the current optimal size based on calibration
    pub fn get_optimal_size(&self) -> usize {
        let calibrator = self.calibrator.lock();
        let optimal = calibrator.calculate_optimal_size();
        if optimal > 0 {
            optimal
        } else {
            self.default_size.load(Ordering::Relaxed)
        }
    }

    /// Get the current default buffer size
    pub fn default_size(&self) -> usize {
        self.default_size.load(Ordering::Relaxed)
    }

    /// Get the maximum allowed buffer size
    pub fn max_size(&self) -> usize {
        self.max_size
    }

    /// Get the pool name
    pub fn name(&self) -> &str {
        &self.pool_name
    }

    /// Get the number of buffers currently in the pool
    pub fn available(&self) -> usize {
        self.pool.len()
    }

    /// Get statistics about the pool
    pub fn stats(&self) -> PoolStats {
        PoolStats {
            name: self.pool_name.clone(),
            default_size: self.default_size.load(Ordering::Relaxed),
            max_size: self.max_size,
            available: self.pool.len(),
            buffers_created: self.buffers_created.load(Ordering::Relaxed),
            get_count: self.get_count.load(Ordering::Relaxed),
            put_count: self.put_count.load(Ordering::Relaxed),
            optimal_size: self.get_optimal_size(),
        }
    }
}

impl Default for BufferPool {
    fn default() -> Self {
        Self::new(4096, 1024 * 1024, "default")
    }
}

impl std::fmt::Debug for BufferPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BufferPool")
            .field("name", &self.pool_name)
            .field("default_size", &self.default_size.load(Ordering::Relaxed))
            .field("max_size", &self.max_size)
            .field("available", &self.pool.len())
            .finish()
    }
}

/// Statistics about a buffer pool
#[derive(Debug, Clone)]
pub struct PoolStats {
    /// Pool name
    pub name: String,
    /// Current default buffer size
    pub default_size: usize,
    /// Maximum buffer size
    pub max_size: usize,
    /// Number of buffers available in pool
    pub available: usize,
    /// Total buffers created
    pub buffers_created: usize,
    /// Total get operations
    pub get_count: usize,
    /// Total put operations
    pub put_count: usize,
    /// Current optimal size from calibrator
    pub optimal_size: usize,
}

impl std::fmt::Display for PoolStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "BufferPool '{}': size={}/{} available={} created={} get={} put={} optimal={}",
            self.name,
            self.default_size,
            self.max_size,
            self.available,
            self.buffers_created,
            self.get_count,
            self.put_count,
            self.optimal_size
        )
    }
}

/// Calibrator for auto-tuning buffer sizes
///
/// Collects samples of buffer sizes used and calculates an optimal size
/// based on the 75th percentile of observed sizes.
struct PoolCalibrator {
    /// Size samples collected
    samples: Vec<usize>,
    /// Maximum samples to keep
    max_samples: usize,
    /// Cached average size
    avg_size: f64,
    /// Whether calibration has been performed
    calibrated: bool,
    /// Number of samples since last calibration
    samples_since_calibration: usize,
}

impl PoolCalibrator {
    /// Create a new calibrator
    fn new(max_samples: usize) -> Self {
        Self {
            samples: Vec::with_capacity(max_samples),
            max_samples,
            avg_size: 0.0,
            calibrated: false,
            samples_since_calibration: 0,
        }
    }

    /// Record a size sample
    fn record_sample(&mut self, size: usize) {
        if self.samples.len() >= self.max_samples {
            // Remove oldest sample (rotate buffer)
            self.samples.remove(0);
        }
        self.samples.push(size);
        self.samples_since_calibration += 1;

        // Update running average
        let sum: usize = self.samples.iter().sum();
        self.avg_size = sum as f64 / self.samples.len() as f64;
    }

    /// Check if we should recalibrate
    fn should_calibrate(&self) -> bool {
        self.samples_since_calibration >= CALIBRATION_THRESHOLD
    }

    /// Mark that calibration has been performed
    fn mark_calibrated(&mut self) {
        self.calibrated = true;
        self.samples_since_calibration = 0;
    }

    /// Calculate the optimal buffer size
    ///
    /// Returns the 75th percentile of observed sizes, rounded up to the
    /// nearest power of 2 for efficient allocation.
    fn calculate_optimal_size(&self) -> usize {
        if self.samples.is_empty() {
            return 0;
        }

        // Sort samples to find percentiles
        let mut sorted: Vec<usize> = self.samples.clone();
        sorted.sort_unstable();

        // Use 75th percentile
        let p75_index = (sorted.len() * 75) / 100;
        let p75 = sorted[p75_index.min(sorted.len() - 1)];

        // Round up to next power of 2 for efficient allocation
        round_up_to_power_of_2(p75)
    }

    /// Get the average observed size
    #[allow(dead_code)]
    fn average_size(&self) -> f64 {
        self.avg_size
    }
}

/// Round up to the next power of 2
fn round_up_to_power_of_2(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut v = n - 1;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    #[cfg(target_pointer_width = "64")]
    {
        v |= v >> 32;
    }
    v + 1
}

/// Global buffer pools for common use cases
pub mod global {
    use super::BufferPool;
    use std::sync::OnceLock;

    static SMALL_POOL: OnceLock<BufferPool> = OnceLock::new();
    static MEDIUM_POOL: OnceLock<BufferPool> = OnceLock::new();
    static LARGE_POOL: OnceLock<BufferPool> = OnceLock::new();

    /// Small buffer pool (4KB default, 64KB max)
    pub fn small() -> &'static BufferPool {
        SMALL_POOL.get_or_init(|| BufferPool::new(4096, 64 * 1024, "small"))
    }

    /// Medium buffer pool (64KB default, 1MB max)
    pub fn medium() -> &'static BufferPool {
        MEDIUM_POOL.get_or_init(|| BufferPool::new(64 * 1024, 1024 * 1024, "medium"))
    }

    /// Large buffer pool (1MB default, 16MB max)
    pub fn large() -> &'static BufferPool {
        LARGE_POOL.get_or_init(|| BufferPool::new(1024 * 1024, 16 * 1024 * 1024, "large"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_buffer_pool_new() {
        let pool = BufferPool::new(1024, 4096, "test");
        assert_eq!(pool.default_size(), 1024);
        assert_eq!(pool.max_size(), 4096);
        assert_eq!(pool.name(), "test");
    }

    #[test]
    fn test_buffer_pool_get() {
        let pool = BufferPool::new(1024, 4096, "test");

        let buf = pool.get();
        assert!(buf.capacity() >= 1024);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_buffer_pool_get_with_capacity() {
        let pool = BufferPool::new(1024, 4096, "test");

        let buf = pool.get_with_capacity(2048);
        assert!(buf.capacity() >= 2048);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_buffer_pool_put() {
        let pool = BufferPool::new(1024, 4096, "test");

        // Get a buffer, use it, put it back
        let mut buf = pool.get();
        buf.extend_from_slice(b"hello world");
        pool.put(buf);

        assert_eq!(pool.available(), 1);

        // Get it back - should be empty
        let buf = pool.get();
        assert!(buf.is_empty());
    }

    #[test]
    fn test_buffer_pool_reuse() {
        let pool = BufferPool::new(1024, 4096, "test");

        // Put some buffers
        for _ in 0..5 {
            let buf = pool.get();
            pool.put(buf);
        }

        // Check stats
        let stats = pool.stats();
        assert!(stats.available <= 5);
        assert!(stats.buffers_created <= 5);
    }

    #[test]
    fn test_buffer_pool_oversized_discard() {
        let pool = BufferPool::new(1024, 4096, "test");

        // Create an oversized buffer
        let buf = vec![0u8; 8192];
        pool.put(buf);

        // Oversized buffer should be discarded
        assert_eq!(pool.available(), 0);
    }

    #[test]
    fn test_buffer_pool_stats() {
        let pool = BufferPool::new(1024, 4096, "test");

        for _ in 0..10 {
            let buf = pool.get();
            pool.put(buf);
        }

        let stats = pool.stats();
        assert_eq!(stats.get_count, 10);
        assert_eq!(stats.put_count, 10);
    }

    #[test]
    fn test_buffer_pool_concurrent() {
        let pool = Arc::new(BufferPool::new(1024, 4096, "test"));
        let num_threads = 4;
        let ops_per_thread = 100;

        let handles: Vec<_> = (0..num_threads)
            .map(|_| {
                let pool = Arc::clone(&pool);
                thread::spawn(move || {
                    for _ in 0..ops_per_thread {
                        let mut buf = pool.get();
                        buf.extend_from_slice(b"test data");
                        pool.put(buf);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        let stats = pool.stats();
        assert_eq!(stats.get_count, num_threads * ops_per_thread);
        assert_eq!(stats.put_count, num_threads * ops_per_thread);
    }

    #[test]
    fn test_calibrator_basic() {
        let mut calibrator = PoolCalibrator::new(100);

        // Record some samples
        for i in 1..=10 {
            calibrator.record_sample(i * 100);
        }

        assert!(calibrator.average_size() > 0.0);
    }

    #[test]
    fn test_calibrator_optimal_size() {
        let mut calibrator = PoolCalibrator::new(100);

        // Record samples clustered around 1000
        for _ in 0..50 {
            calibrator.record_sample(900);
            calibrator.record_sample(1000);
            calibrator.record_sample(1100);
        }

        let optimal = calibrator.calculate_optimal_size();
        // Should be around 1024 (next power of 2 after ~1000)
        assert!((1024..=2048).contains(&optimal));
    }

    #[test]
    fn test_calibrator_auto_tune() {
        let pool = BufferPool::new(1024, 1024 * 1024, "test");

        // Record many samples of larger sizes
        for _ in 0..100 {
            pool.record_size(4000);
        }

        // Pool should have auto-tuned
        let optimal = pool.get_optimal_size();
        assert!(optimal >= 4096); // Should be at least 4096 (next power of 2)
    }

    #[test]
    fn test_round_up_to_power_of_2() {
        assert_eq!(round_up_to_power_of_2(0), 1);
        assert_eq!(round_up_to_power_of_2(1), 1);
        assert_eq!(round_up_to_power_of_2(2), 2);
        assert_eq!(round_up_to_power_of_2(3), 4);
        assert_eq!(round_up_to_power_of_2(4), 4);
        assert_eq!(round_up_to_power_of_2(5), 8);
        assert_eq!(round_up_to_power_of_2(1000), 1024);
        assert_eq!(round_up_to_power_of_2(1024), 1024);
        assert_eq!(round_up_to_power_of_2(1025), 2048);
    }

    #[test]
    fn test_default_pool() {
        let pool = BufferPool::default();
        assert_eq!(pool.default_size(), 4096);
        assert_eq!(pool.max_size(), 1024 * 1024);
    }

    #[test]
    fn test_global_pools() {
        // Just ensure global pools are accessible
        let _small = global::small();
        let _medium = global::medium();
        let _large = global::large();
    }

    #[test]
    fn test_pool_debug() {
        let pool = BufferPool::new(1024, 4096, "debug_test");
        let debug = format!("{:?}", pool);
        assert!(debug.contains("debug_test"));
        assert!(debug.contains("1024"));
    }

    #[test]
    fn test_stats_display() {
        let pool = BufferPool::new(1024, 4096, "display_test");
        let _ = pool.get();
        let stats = pool.stats();
        let display = stats.to_string();
        assert!(display.contains("display_test"));
    }
}
