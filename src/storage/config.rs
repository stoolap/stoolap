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

//! Storage engine configuration
//!

/// WAL sync mode for controlling durability vs performance tradeoff
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SyncMode {
    /// Fastest but least durable - doesn't force syncs
    None = 0,
    /// Syncs on transaction commits - good balance of performance and durability
    #[default]
    Normal = 1,
    /// Forces syncs on every WAL write - slowest but most durable
    Full = 2,
}

impl From<i32> for SyncMode {
    fn from(value: i32) -> Self {
        match value {
            0 => SyncMode::None,
            2 => SyncMode::Full,
            _ => SyncMode::Normal,
        }
    }
}

impl From<SyncMode> for i32 {
    fn from(mode: SyncMode) -> Self {
        mode as i32
    }
}

/// Configuration options for the persistence layer
#[derive(Debug, Clone)]
pub struct PersistenceConfig {
    /// Whether persistence is enabled
    /// Default: true if Path is not empty
    pub enabled: bool,

    /// WAL sync strategy
    /// Default: Normal
    pub sync_mode: SyncMode,

    /// Time between snapshots in seconds
    /// Default: 300 (5 minutes)
    pub snapshot_interval: u32,

    /// Number of snapshots to keep
    /// Default: 5
    pub keep_snapshots: u32,

    /// Size in bytes that triggers a WAL flush
    /// Default: 32768 (32KB)
    pub wal_flush_trigger: usize,

    /// Initial WAL buffer size in bytes
    /// Default: 65536 (64KB)
    pub wal_buffer_size: usize,

    /// Maximum size of a WAL file before rotation in bytes
    /// Default: 67108864 (64MB)
    pub wal_max_size: usize,

    /// Number of commits to batch before syncing in SyncNormal mode
    /// Default: 100
    pub commit_batch_size: u32,

    /// Minimum time between syncs in milliseconds in SyncNormal mode
    /// Default: 10
    pub sync_interval_ms: u32,

    /// Enable LZ4 compression for WAL entries
    /// Default: true
    pub wal_compression: bool,

    /// Enable LZ4 compression for snapshot rows
    /// Default: true
    pub snapshot_compression: bool,

    /// Minimum data size (bytes) before attempting compression
    /// Default: 64
    pub compression_threshold: usize,
}

impl Default for PersistenceConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            sync_mode: SyncMode::Normal,
            snapshot_interval: 300,         // 5 minutes
            keep_snapshots: 5,              // Keep 5 snapshots
            wal_flush_trigger: 32 * 1024,   // 32KB
            wal_buffer_size: 64 * 1024,     // 64KB
            wal_max_size: 64 * 1024 * 1024, // 64MB
            commit_batch_size: 100,         // Batch 100 commits
            sync_interval_ms: 10,           // 10ms minimum interval
            wal_compression: true,          // Enable WAL compression
            snapshot_compression: true,     // Enable snapshot compression
            compression_threshold: 64,      // Compress entries >= 64 bytes
        }
    }
}

impl PersistenceConfig {
    /// Creates a new PersistenceConfig with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a PersistenceConfig optimized for maximum durability
    pub fn durable() -> Self {
        Self {
            enabled: true,
            sync_mode: SyncMode::Full,
            snapshot_interval: 60, // 1 minute
            keep_snapshots: 10,
            wal_flush_trigger: 8 * 1024,    // 8KB - flush more often
            wal_buffer_size: 32 * 1024,     // 32KB
            wal_max_size: 32 * 1024 * 1024, // 32MB - smaller files
            commit_batch_size: 1,           // No batching
            sync_interval_ms: 0,            // Immediate sync
            wal_compression: true,
            snapshot_compression: true,
            compression_threshold: 64,
        }
    }

    /// Creates a PersistenceConfig optimized for maximum performance
    pub fn fast() -> Self {
        Self {
            enabled: true,
            sync_mode: SyncMode::None,
            snapshot_interval: 600, // 10 minutes
            keep_snapshots: 3,
            wal_flush_trigger: 64 * 1024,    // 64KB
            wal_buffer_size: 128 * 1024,     // 128KB
            wal_max_size: 128 * 1024 * 1024, // 128MB
            commit_batch_size: 500,          // Batch more commits
            sync_interval_ms: 100,           // Less frequent sync
            wal_compression: true,
            snapshot_compression: true,
            compression_threshold: 64,
        }
    }

    /// Builder method to set sync mode
    pub fn with_sync_mode(mut self, mode: SyncMode) -> Self {
        self.sync_mode = mode;
        self
    }

    /// Builder method to set snapshot interval
    pub fn with_snapshot_interval(mut self, seconds: u32) -> Self {
        self.snapshot_interval = seconds;
        self
    }

    /// Builder method to set number of snapshots to keep
    pub fn with_keep_snapshots(mut self, count: u32) -> Self {
        self.keep_snapshots = count;
        self
    }

    /// Builder method to enable/disable WAL compression
    pub fn with_wal_compression(mut self, enabled: bool) -> Self {
        self.wal_compression = enabled;
        self
    }

    /// Builder method to enable/disable snapshot compression
    pub fn with_snapshot_compression(mut self, enabled: bool) -> Self {
        self.snapshot_compression = enabled;
        self
    }

    /// Builder method to enable/disable all compression
    pub fn with_compression(mut self, enabled: bool) -> Self {
        self.wal_compression = enabled;
        self.snapshot_compression = enabled;
        self
    }

    /// Builder method to set compression threshold
    pub fn with_compression_threshold(mut self, bytes: usize) -> Self {
        self.compression_threshold = bytes;
        self
    }
}

/// Configuration for the storage engine
#[derive(Debug, Clone, Default)]
pub struct Config {
    /// Path to the database directory
    /// If empty, database operates in memory-only mode
    pub path: Option<String>,

    /// Configuration options for disk persistence
    /// Only used if path is Some
    pub persistence: PersistenceConfig,
}

impl Config {
    /// Creates a new in-memory configuration (no persistence)
    pub fn in_memory() -> Self {
        Self {
            path: None,
            persistence: PersistenceConfig {
                enabled: false,
                ..Default::default()
            },
        }
    }

    /// Creates a new configuration with persistence at the given path
    pub fn with_path<P: Into<String>>(path: P) -> Self {
        Self {
            path: Some(path.into()),
            persistence: PersistenceConfig::default(),
        }
    }

    /// Returns true if persistence is enabled
    pub fn is_persistent(&self) -> bool {
        self.path.is_some() && self.persistence.enabled
    }

    /// Builder method to set persistence config
    pub fn with_persistence(mut self, config: PersistenceConfig) -> Self {
        self.persistence = config;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sync_mode_default() {
        assert_eq!(SyncMode::default(), SyncMode::Normal);
    }

    #[test]
    fn test_sync_mode_from_i32() {
        assert_eq!(SyncMode::from(0), SyncMode::None);
        assert_eq!(SyncMode::from(1), SyncMode::Normal);
        assert_eq!(SyncMode::from(2), SyncMode::Full);
        assert_eq!(SyncMode::from(99), SyncMode::Normal); // Invalid defaults to Normal
    }

    #[test]
    fn test_persistence_config_default() {
        let config = PersistenceConfig::default();
        assert!(config.enabled);
        assert_eq!(config.sync_mode, SyncMode::Normal);
        assert_eq!(config.snapshot_interval, 300);
        assert_eq!(config.keep_snapshots, 5);
        assert_eq!(config.wal_flush_trigger, 32 * 1024);
        assert_eq!(config.wal_buffer_size, 64 * 1024);
        assert_eq!(config.wal_max_size, 64 * 1024 * 1024);
        assert_eq!(config.commit_batch_size, 100);
        assert_eq!(config.sync_interval_ms, 10);
        assert!(config.wal_compression);
        assert!(config.snapshot_compression);
        assert_eq!(config.compression_threshold, 64);
    }

    #[test]
    fn test_persistence_config_durable() {
        let config = PersistenceConfig::durable();
        assert_eq!(config.sync_mode, SyncMode::Full);
        assert_eq!(config.commit_batch_size, 1);
        assert_eq!(config.sync_interval_ms, 0);
    }

    #[test]
    fn test_persistence_config_fast() {
        let config = PersistenceConfig::fast();
        assert_eq!(config.sync_mode, SyncMode::None);
        assert_eq!(config.commit_batch_size, 500);
    }

    #[test]
    fn test_persistence_config_builder() {
        let config = PersistenceConfig::new()
            .with_sync_mode(SyncMode::Full)
            .with_snapshot_interval(120)
            .with_keep_snapshots(10);

        assert_eq!(config.sync_mode, SyncMode::Full);
        assert_eq!(config.snapshot_interval, 120);
        assert_eq!(config.keep_snapshots, 10);
    }

    #[test]
    fn test_persistence_config_compression() {
        // Test disabling all compression
        let config = PersistenceConfig::new().with_compression(false);
        assert!(!config.wal_compression);
        assert!(!config.snapshot_compression);

        // Test individual compression settings
        let config = PersistenceConfig::new()
            .with_wal_compression(false)
            .with_snapshot_compression(true);
        assert!(!config.wal_compression);
        assert!(config.snapshot_compression);

        // Test compression threshold
        let config = PersistenceConfig::new().with_compression_threshold(128);
        assert_eq!(config.compression_threshold, 128);
    }

    #[test]
    fn test_config_in_memory() {
        let config = Config::in_memory();
        assert!(config.path.is_none());
        assert!(!config.persistence.enabled);
        assert!(!config.is_persistent());
    }

    #[test]
    fn test_config_with_path() {
        let config = Config::with_path("/tmp/test.db");
        assert_eq!(config.path, Some("/tmp/test.db".to_string()));
        assert!(config.persistence.enabled);
        assert!(config.is_persistent());
    }

    #[test]
    fn test_config_builder() {
        let config =
            Config::with_path("/tmp/test.db").with_persistence(PersistenceConfig::durable());

        assert!(config.is_persistent());
        assert_eq!(config.persistence.sync_mode, SyncMode::Full);
    }
}
