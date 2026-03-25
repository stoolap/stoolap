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

//! Table manifest: the source of truth for a table's segment state.
//!
//! The manifest tracks which immutable segments (frozen volumes) exist for a
//! table, along with a tombstone set of cold row_ids that have been deleted
//! or superseded by hot buffer versions.
//!
//! # Persistence
//!
//! The manifest is written atomically (tmp + rename) to a single file
//! per table. On recovery, the manifest is loaded first, then segments
//! are loaded from the paths recorded in the manifest.

use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::RwLock;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::common::SmartString;
use crate::core::{Result, Value};

use super::writer::FrozenVolume;

// Manifest file magic: "STMF" (SToolap ManiFest)
const MANIFEST_MAGIC: [u8; 4] = *b"STMF";
const MANIFEST_VERSION: u32 = 4;

/// Metadata for a single immutable segment (frozen volume).
#[derive(Debug, Clone)]
pub struct SegmentMeta {
    /// Unique segment identifier (monotonically increasing).
    pub segment_id: u64,
    /// Path to the segment file on disk (relative to volume dir).
    pub file_path: PathBuf,
    /// Number of rows in this segment.
    pub row_count: usize,
    /// Minimum row_id in this segment.
    pub min_row_id: i64,
    /// Maximum row_id in this segment.
    pub max_row_id: i64,
    /// WAL LSN at which this segment was created (for recovery).
    pub creation_lsn: u64,
    /// Compaction epoch when this segment was last written/merged.
    /// Used to force-include stale volumes after max_age cycles.
    pub compaction_epoch: u64,
}

/// The manifest for a single table: tracks all live segments and tombstones.
///
/// This is the source of truth for what segments exist and which cold
/// row_ids have been deleted or superseded.
#[derive(Debug, Clone)]
pub struct TableManifest {
    /// Table name.
    pub table_name: SmartString,
    /// Live segments (ordered by segment_id, oldest first).
    pub segments: Vec<SegmentMeta>,
    /// Next segment ID to assign.
    pub next_segment_id: u64,
    /// WAL LSN of the last checkpoint that included this manifest.
    pub checkpoint_lsn: u64,
    /// Tombstone entries: (row_id, commit_seq) pairs for cold rows that have
    /// been deleted or superseded. The commit_seq records when the tombstone
    /// was created, enabling snapshot isolation: a snapshot transaction at
    /// begin_seq=N only sees tombstones with commit_seq <= N.
    /// Cleared after compaction processes them.
    pub tombstones: Vec<(i64, u64)>,
}

impl TableManifest {
    /// Create an empty manifest for a table.
    pub fn new(table_name: &str) -> Self {
        Self {
            table_name: SmartString::from(table_name),
            segments: Vec::new(),
            next_segment_id: 1,
            checkpoint_lsn: 0,
            tombstones: Vec::new(),
        }
    }

    /// Allocate a new segment ID.
    pub fn allocate_segment_id(&mut self) -> u64 {
        let id = self.next_segment_id;
        self.next_segment_id += 1;
        id
    }

    /// Add a segment to the manifest.
    pub fn add_segment(&mut self, meta: SegmentMeta) {
        self.segments.push(meta);
    }

    /// Remove segments by ID (after compaction).
    pub fn remove_segments(&mut self, ids: &[u64]) {
        let id_set: FxHashSet<u64> = ids.iter().copied().collect();
        self.segments.retain(|s| !id_set.contains(&s.segment_id));
    }

    /// Find which segment contains a given row_id.
    ///
    /// Uses min/max row_id metadata for fast rejection.
    /// Returns (segment_index, segment_meta) if found.
    pub fn find_segment_for_row_id(&self, row_id: i64) -> Option<(usize, &SegmentMeta)> {
        for (i, seg) in self.segments.iter().enumerate() {
            if row_id >= seg.min_row_id && row_id <= seg.max_row_id {
                return Some((i, seg));
            }
        }
        None
    }

    /// Serialize the manifest to bytes (V2 format with tombstones).
    pub fn serialize(&self) -> io::Result<Vec<u8>> {
        let mut buf = Vec::with_capacity(256);

        // Header
        buf.write_all(&MANIFEST_MAGIC)?;
        buf.write_all(&MANIFEST_VERSION.to_le_bytes())?;
        buf.write_all(&self.next_segment_id.to_le_bytes())?;
        buf.write_all(&self.checkpoint_lsn.to_le_bytes())?;

        // Table name
        let name_bytes = self.table_name.as_bytes();
        buf.write_all(&(name_bytes.len() as u32).to_le_bytes())?;
        buf.write_all(name_bytes)?;

        // Segments
        buf.write_all(&(self.segments.len() as u32).to_le_bytes())?;
        for seg in &self.segments {
            buf.write_all(&seg.segment_id.to_le_bytes())?;
            buf.write_all(&(seg.row_count as u64).to_le_bytes())?;
            buf.write_all(&seg.min_row_id.to_le_bytes())?;
            buf.write_all(&seg.max_row_id.to_le_bytes())?;
            buf.write_all(&seg.creation_lsn.to_le_bytes())?;

            // File path as UTF-8 string
            let path_str = seg.file_path.to_string_lossy();
            let path_bytes = path_str.as_bytes();
            buf.write_all(&(path_bytes.len() as u32).to_le_bytes())?;
            buf.write_all(path_bytes)?;
        }

        // Tombstones (V4: row_id + commit_seq pairs; V2-V3: row_id only)
        buf.write_all(&(self.tombstones.len() as u64).to_le_bytes())?;
        for &(row_id, commit_seq) in &self.tombstones {
            buf.write_all(&row_id.to_le_bytes())?;
            buf.write_all(&commit_seq.to_le_bytes())?;
        }

        // Trailing CRC32 over the entire payload (V3+)
        let crc = crc32fast::hash(&buf);
        buf.write_all(&crc.to_le_bytes())?;

        Ok(buf)
    }

    /// Deserialize a manifest from bytes.
    pub fn deserialize(data: &[u8]) -> io::Result<Self> {
        if data.len() < 24 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "manifest too small",
            ));
        }
        if data[0..4] != MANIFEST_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid manifest magic",
            ));
        }
        let mut pos = 4;

        let version = read_u32(data, &mut pos)?;
        if version > MANIFEST_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported manifest version {}", version),
            ));
        }

        // V3+: verify trailing CRC32 before parsing the rest.
        if version >= 3 {
            if data.len() < 28 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "manifest too small for CRC",
                ));
            }
            let payload = &data[..data.len() - 4];
            let stored_crc = u32::from_le_bytes(
                data[data.len() - 4..]
                    .try_into()
                    .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "bad CRC bytes"))?,
            );
            let computed_crc = crc32fast::hash(payload);
            if stored_crc != computed_crc {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "manifest CRC mismatch: stored={:#010x}, computed={:#010x}",
                        stored_crc, computed_crc
                    ),
                ));
            }
        }

        let next_segment_id = read_u64(data, &mut pos)?;
        let checkpoint_lsn = read_u64(data, &mut pos)?;

        // Table name
        let name_len = read_u32(data, &mut pos)? as usize;
        if pos + name_len > data.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "manifest truncated at table name",
            ));
        }
        let table_name = std::str::from_utf8(&data[pos..pos + name_len])
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        pos += name_len;

        // Segments
        let seg_count = read_u32(data, &mut pos)? as usize;
        let mut segments = Vec::with_capacity(seg_count);
        for _ in 0..seg_count {
            if pos + 40 > data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "manifest truncated at segment",
                ));
            }
            let segment_id = read_u64(data, &mut pos)?;
            let row_count = read_u64(data, &mut pos)? as usize;
            let min_row_id = read_i64(data, &mut pos)?;
            let max_row_id = read_i64(data, &mut pos)?;
            let creation_lsn = read_u64(data, &mut pos)?;

            let path_len = read_u32(data, &mut pos)? as usize;
            if pos + path_len > data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "manifest truncated at path",
                ));
            }
            let path_str = std::str::from_utf8(&data[pos..pos + path_len])
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            pos += path_len;

            segments.push(SegmentMeta {
                segment_id,
                file_path: PathBuf::from(path_str),
                row_count,
                min_row_id,
                max_row_id,
                creation_lsn,
                compaction_epoch: 0,
            });
        }

        // Tombstones (V2+, optional for V1 backward compat)
        // V4: (row_id, commit_seq) pairs. V2-V3: row_id only (commit_seq=0).
        // For V3+, the last 4 bytes are CRC, so stop before them.
        let tombstone_data_end = if version >= 3 {
            data.len() - 4
        } else {
            data.len()
        };
        let mut tombstones = Vec::new();
        if version >= 2 && pos + 8 <= tombstone_data_end {
            let tombstone_count = read_u64(data, &mut pos)? as usize;
            tombstones.reserve(tombstone_count);
            if version >= 4 {
                // V4: each tombstone is (row_id: i64, commit_seq: u64) = 16 bytes
                for _ in 0..tombstone_count {
                    if pos + 16 > tombstone_data_end {
                        break;
                    }
                    let row_id = read_i64(data, &mut pos)?;
                    let commit_seq = read_u64(data, &mut pos)?;
                    tombstones.push((row_id, commit_seq));
                }
            } else {
                // V2-V3: each tombstone is just row_id, assign commit_seq=0
                // (always visible to all snapshots — pre-restart tombstones)
                for _ in 0..tombstone_count {
                    if pos + 8 > tombstone_data_end {
                        break;
                    }
                    tombstones.push((read_i64(data, &mut pos)?, 0));
                }
            }
        }

        Ok(Self {
            table_name: SmartString::from(table_name),
            segments,
            next_segment_id,
            checkpoint_lsn,
            tombstones,
        })
    }

    /// Write manifest to disk atomically.
    pub fn write_to_disk(&self, path: &Path) -> Result<()> {
        let data = self.serialize().map_err(|e| {
            crate::core::Error::internal(format!("failed to serialize manifest: {}", e))
        })?;

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                crate::core::Error::internal(format!("failed to create manifest dir: {}", e))
            })?;
        }

        // Write to tmp file and fsync BEFORE rename for crash safety.
        let tmp_path = path.with_extension("manifest.tmp");
        {
            use std::io::Write;
            let mut f = std::fs::File::create(&tmp_path).map_err(|e| {
                crate::core::Error::internal(format!("failed to create manifest tmp file: {}", e))
            })?;
            f.write_all(&data).map_err(|e| {
                crate::core::Error::internal(format!("failed to write manifest: {}", e))
            })?;
            f.sync_all().map_err(|e| {
                crate::core::Error::internal(format!("failed to fsync manifest: {}", e))
            })?;
        }
        std::fs::rename(&tmp_path, path).map_err(|e| {
            crate::core::Error::internal(format!("failed to rename manifest: {}", e))
        })?;
        // Fsync parent directory to ensure the rename is durable.
        // Windows does not support opening directories for fsync;
        // NTFS metadata is flushed with the file's sync_all().
        #[cfg(not(windows))]
        if let Some(parent) = path.parent() {
            let d = std::fs::File::open(parent)
                .map_err(|e| io::Error::other(format!("failed to open dir for fsync: {}", e)))?;
            d.sync_all()
                .map_err(|e| io::Error::other(format!("failed to fsync dir: {}", e)))?;
        }
        Ok(())
    }

    /// Read manifest from disk.
    pub fn read_from_disk(path: &Path) -> Result<Self> {
        let data = std::fs::read(path)
            .map_err(|e| crate::core::Error::internal(format!("failed to read manifest: {}", e)))?;
        Self::deserialize(&data).map_err(|e| {
            crate::core::Error::internal(format!("failed to deserialize manifest: {}", e))
        })
    }
}

/// Per-table segment manager.
///
/// Owns the manifest, loaded segments, and tombstone set for one table.
/// Tombstones track cold row_ids that have been deleted or superseded
/// by hot buffer versions. They are persisted in the manifest and used
/// during scans to skip stale cold rows.
///
/// Thread safety: the manager uses interior mutability via RwLock for
/// concurrent read access (queries) and exclusive write access (seal, compaction).
pub struct SegmentManager {
    /// Table name.
    table_name: SmartString,
    /// The manifest (source of truth for segment state).
    manifest: RwLock<TableManifest>,
    /// Loaded segments, keyed by segment_id.
    /// CoW via Arc: readers clone the Arc (O(1) atomic increment, ~5ns),
    /// writers clone the inner map, modify, and swap the Arc.
    /// This eliminates write-lock starvation on register_segment.
    segments: RwLock<Arc<FxHashMap<u64, Arc<FrozenVolume>>>>,
    /// Base directory for volume files (None for memory-only databases).
    volume_dir: Option<PathBuf>,
    /// Fast atomic flag: true if any segments are loaded.
    has_segments_flag: std::sync::atomic::AtomicBool,
    /// Committed tombstone map: cold row_id → commit_seq (when the tombstone was created).
    /// Built from manifest tombstones on startup, updated at commit time.
    /// Wrapped in Arc for cheap O(1) reads — most callers only need to check
    /// membership, not mutate. Writers swap the Arc on mutation.
    /// The commit_seq enables snapshot isolation: a snapshot at begin_seq=N
    /// only sees tombstones with commit_seq <= N.
    tombstones: RwLock<Arc<FxHashMap<i64, u64>>>,
    /// Per-transaction pending tombstones: txn_id → list of cold row_ids to tombstone.
    /// Applied to the shared tombstone set on commit, discarded on rollback.
    /// This lives on the SegmentManager (not SegmentedTable) because the commit
    /// path in engine.rs creates fresh MVCCTable instances that don't have
    /// access to SegmentedTable state.
    pending_txn_tombstones: RwLock<FxHashMap<i64, FxHashSet<i64>>>,
    // Unique constraint checks use per-volume hash indices (on FrozenVolume).
    // No global cache needed. Each volume builds its index lazily on first
    // unique check and never invalidates (volumes are immutable).
    // Zone maps + bloom filters prune volumes before hash lookup.
    /// Cached deduplicated row count. Invalidated (set to u64::MAX) on
    /// segment or tombstone changes. Recomputed lazily on next read.
    cached_deduped_count: std::sync::atomic::AtomicU64,
    /// Per-table fence that serializes seal with cold-check + hot insert.
    /// INSERTs take a shared guard while checking cold constraints and
    /// publishing into hot; seal takes the exclusive guard while moving rows.
    seal_fence: RwLock<()>,
    /// Monotonic counter incremented on every register_segment. Used at
    /// commit time to detect whether a seal happened since statement time.
    /// If unchanged, the commit-time cold recheck is skipped (fast path).
    seal_generation: std::sync::atomic::AtomicU64,
    /// Per-txn seal generation at INSERT time. Small map — only active
    /// transactions with pending inserts on this table.
    txn_seal_gens: parking_lot::Mutex<rustc_hash::FxHashMap<i64, u64>>,
    /// Number of rows currently being sealed (exist in both hot and cold).
    /// Set to N before register_segment, cleared after remove_sealed_rows.
    /// Subtracted from row_count() to prevent double-counting during the seal window.
    seal_overlap_count: std::sync::atomic::AtomicUsize,
}

impl SegmentManager {
    /// Create a new segment manager for a table.
    pub fn new(table_name: &str, volume_dir: Option<PathBuf>) -> Self {
        Self {
            table_name: SmartString::from(table_name),
            manifest: RwLock::new(TableManifest::new(table_name)),
            segments: RwLock::new(Arc::new(FxHashMap::default())),
            volume_dir,
            has_segments_flag: std::sync::atomic::AtomicBool::new(false),
            tombstones: RwLock::new(Arc::new(FxHashMap::default())),
            pending_txn_tombstones: RwLock::new(FxHashMap::default()),
            cached_deduped_count: std::sync::atomic::AtomicU64::new(u64::MAX),
            seal_fence: RwLock::new(()),
            seal_generation: std::sync::atomic::AtomicU64::new(0),
            txn_seal_gens: parking_lot::Mutex::new(rustc_hash::FxHashMap::default()),
            seal_overlap_count: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Create from an existing manifest loaded from disk.
    pub fn from_manifest(manifest: TableManifest, volume_dir: Option<PathBuf>) -> Self {
        let table_name = manifest.table_name.clone();
        let tombstone_map: FxHashMap<i64, u64> = manifest.tombstones.iter().copied().collect();
        Self {
            table_name,
            manifest: RwLock::new(manifest),
            segments: RwLock::new(Arc::new(FxHashMap::default())),
            volume_dir,
            has_segments_flag: std::sync::atomic::AtomicBool::new(false),
            tombstones: RwLock::new(Arc::new(tombstone_map)),
            pending_txn_tombstones: RwLock::new(FxHashMap::default()),
            cached_deduped_count: std::sync::atomic::AtomicU64::new(u64::MAX),
            seal_fence: RwLock::new(()),
            seal_generation: std::sync::atomic::AtomicU64::new(0),
            txn_seal_gens: parking_lot::Mutex::new(rustc_hash::FxHashMap::default()),
            seal_overlap_count: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Get the table name.
    pub fn table_name(&self) -> &str {
        &self.table_name
    }

    /// Get all loaded segments (for scanning).
    pub fn get_segments(&self) -> Vec<Arc<FrozenVolume>> {
        let segs = Arc::clone(&*self.segments.read());
        segs.values().cloned().collect()
    }

    /// Get segments in order (by segment_id, oldest first).
    pub fn get_segments_ordered(&self) -> Vec<Arc<FrozenVolume>> {
        // Hold manifest lock while cloning segments Arc to prevent
        // compaction from swapping the map between the two reads.
        let (seg_ids, segments) = {
            let manifest = self.manifest.read();
            let seg_ids: Vec<u64> = manifest.segments.iter().map(|m| m.segment_id).collect();
            let segments = Arc::clone(&*self.segments.read());
            (seg_ids, segments)
        };
        seg_ids
            .iter()
            .filter_map(|id| segments.get(id).cloned())
            .collect()
    }

    /// Get volumes in newest-first order (by segment_id, descending).
    /// Used for building per-volume skip sets in SegmentedTable.scan().
    /// Segments are always appended in ascending order, so reverse gives
    /// newest-first in O(n) instead of O(n log n) sort.
    pub fn get_volumes_newest_first(&self) -> Arc<Vec<(u64, Arc<FrozenVolume>)>> {
        // Hold manifest lock while cloning segments Arc to prevent
        // compaction's replace_segments_atomic from swapping the map
        // between the two reads (TOCTOU → missing rows in scan).
        let (seg_ids, segs) = {
            let manifest = self.manifest.read();
            let seg_ids: Vec<u64> = manifest.segments.iter().map(|m| m.segment_id).collect();
            let segs = Arc::clone(&*self.segments.read());
            (seg_ids, segs)
        };
        let mut result: Vec<(u64, Arc<FrozenVolume>)> = seg_ids
            .iter()
            .filter_map(|&id| segs.get(&id).map(|v| (id, Arc::clone(v))))
            .collect();
        result.reverse();
        Arc::new(result)
    }

    /// Check if there are any segments. O(1) atomic read, no lock.
    pub fn has_segments(&self) -> bool {
        self.has_segments_flag
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Check if a value exists in any segment's column (for PK/UNIQUE constraint checks).
    ///
    /// Uses zone maps + binary search on sorted integer columns. No cloning.
    /// Returns the row_id if found and not tombstoned, None otherwise.
    /// Check if a value exists in cold segments for a given column.
    /// Used for PK constraint checking (Integer/Timestamp columns only).
    ///
    /// Iterates newest-first with row_id dedup so that when the same row_id
    /// exists in multiple overlapping volumes, only the newest version is checked.
    pub fn check_value_exists_in_segments(
        &self,
        col_idx: usize,
        value: &crate::core::Value,
    ) -> Option<i64> {
        // Atomic snapshot: read seg_ids and segments map under the manifest
        // lock so compaction's replace_segments_atomic (which holds manifest
        // write → segments write) cannot swap them between the two reads.
        let (seg_ids, segs, ts) = {
            let manifest = self.manifest.read();
            let seg_ids: Vec<u64> = manifest
                .segments
                .iter()
                .rev()
                .map(|m| m.segment_id)
                .collect();
            let segs = Arc::clone(&*self.segments.read());
            let ts = Arc::clone(&*self.tombstones.read());
            (seg_ids, segs, ts)
        };

        // Collect row_ids we've already seen across volumes (newest-first dedup).
        // For PK checks this is typically small because PK values are unique per volume.
        let mut seen = FxHashSet::default();

        for &seg_id in &seg_ids {
            let Some(vol) = segs.get(&seg_id) else {
                continue;
            };
            if col_idx >= vol.columns.len() || col_idx >= vol.zone_maps.len() {
                continue;
            }
            if !vol.zone_maps[col_idx].may_contain_eq(value) {
                continue;
            }
            let target = match value {
                crate::core::Value::Integer(int_val) => Some(*int_val),
                crate::core::Value::Timestamp(ts_val) => Some(
                    ts_val
                        .timestamp_nanos_opt()
                        .unwrap_or(ts_val.timestamp() * 1_000_000_000),
                ),
                _ => None,
            };
            if let Some(target) = target {
                if vol.is_sorted(col_idx) {
                    let start = vol.columns[col_idx].binary_search_ge(target);
                    let mut i = start;
                    while i < vol.row_count && vol.columns[col_idx].get_i64(i) == target {
                        let rid = vol.row_ids[i];
                        if seen.insert(rid) && !ts.contains_key(&rid) {
                            // Verify this is the authoritative version when overlapping
                            // volumes exist. After UPDATE changes PK + seal, the old PK
                            // value exists in an older volume without a tombstone.
                            if seg_ids.len() > 1 {
                                if let Some(current_row) = self.get_cold_row(rid) {
                                    if let Some(current_val) = current_row.get(col_idx) {
                                        if current_val != value {
                                            i += 1;
                                            continue; // stale PK in older volume
                                        }
                                    }
                                }
                            }
                            return Some(rid);
                        }
                        i += 1;
                    }
                } else {
                    for i in 0..vol.row_count {
                        let rid = vol.row_ids[i];
                        if !seen.insert(rid) {
                            continue;
                        }
                        if !vol.columns[col_idx].is_null(i)
                            && vol.columns[col_idx].get_i64(i) == target
                            && !ts.contains_key(&rid)
                        {
                            if seg_ids.len() > 1 {
                                if let Some(current_row) = self.get_cold_row(rid) {
                                    if let Some(current_val) = current_row.get(col_idx) {
                                        if current_val != value {
                                            continue; // stale PK in older volume
                                        }
                                    }
                                }
                            }
                            return Some(rid);
                        }
                    }
                }
            }
        }
        None
    }

    /// Find a visible cold row ID matching the given column values.
    /// Uses a three-tier pruning strategy per volume (newest first):
    ///   1. Zone map: skip if value outside [min, max] for any column
    ///   2. Bloom filter: skip if any column says "definitely not"
    ///   3. Per-volume hash index: O(1) lookup (lazily built, never invalidated)
    ///
    /// No global cache. Each volume's hash index is built once on first use
    /// and lives on the immutable FrozenVolume. Zero invalidation cost.
    pub fn find_row_id_by_values(
        &self,
        col_indices: &[usize],
        values: &[&Value],
        column_defaults: &[Value],
        schema_col_count: usize,
    ) -> Option<i64> {
        if col_indices.is_empty() || col_indices.len() != values.len() {
            return None;
        }

        // Atomic snapshot: read seg_ids and segments map under the manifest
        // lock so compaction's replace_segments_atomic cannot swap them
        // between the two reads (TOCTOU race → false unique-miss → dupes).
        let (seg_ids, segs, ts) = {
            let manifest = self.manifest.read();
            let seg_ids: Vec<u64> = manifest
                .segments
                .iter()
                .rev()
                .map(|m| m.segment_id)
                .collect();
            let segs = Arc::clone(&*self.segments.read());
            let ts = Arc::clone(&*self.tombstones.read());
            (seg_ids, segs, ts)
        };

        if seg_ids.is_empty() {
            return None;
        }

        // Pre-compute bloom hashes once (reused across all volumes).
        let bloom_hashes: Vec<u64> = values
            .iter()
            .map(|v| super::column::ColumnBloomFilter::hash_value_static(v))
            .collect();

        // Track seen row_ids for newest-first dedup across overlapping volumes.
        let mut seen = FxHashSet::default();

        for &seg_id in &seg_ids {
            let Some(vol) = segs.get(&seg_id) else {
                continue;
            };
            // For schema-evolved volumes missing columns: check if the search
            // value matches the column default. If not → no row can match → skip.
            // If yes → check only the columns that exist in this volume.
            let vol_cols = vol.columns.len().min(schema_col_count);
            if col_indices
                .iter()
                .enumerate()
                .any(|(i, &idx)| idx >= vol_cols && *values[i] != column_defaults[i])
            {
                continue;
            }

            // Tier 1: Zone map pruning — skip volume if value outside [min, max]
            let mut zone_skip = false;
            for (&ci, &val) in col_indices.iter().zip(values.iter()) {
                if ci < vol.zone_maps.len() && !vol.zone_maps[ci].may_contain_eq(val) {
                    zone_skip = true;
                    break;
                }
            }
            if zone_skip {
                continue;
            }

            // Tier 2: Bloom filter pruning — skip if any column says "definitely not"
            let mut bloom_skip = false;
            for (i, &ci) in col_indices.iter().enumerate() {
                if ci < vol.bloom_filters.len()
                    && !vol.bloom_filters[ci].might_contain_hash(bloom_hashes[i])
                {
                    bloom_skip = true;
                    break;
                }
            }
            if bloom_skip {
                continue;
            }

            // Tier 3: Per-volume hash index — O(1) lookup, supports duplicate rows
            let mut vol_result: Option<i64> = None;
            let vol_missing_cols = col_indices.iter().any(|&idx| idx >= vol_cols);
            if !vol_missing_cols {
                // All columns present: standard hash index lookup
                vol.unique_lookup_all(col_indices, values, |row_idx| {
                    let rid = vol.row_ids[row_idx as usize];
                    if ts.contains_key(&rid) {
                        false
                    } else if seen.insert(rid) {
                        vol_result = Some(rid);
                        true
                    } else {
                        false
                    }
                });
            } else {
                // Schema-evolved volume: some columns missing (default matches).
                // Check only the columns that exist in the volume.
                let present_cols: Vec<(usize, usize)> = col_indices
                    .iter()
                    .enumerate()
                    .filter(|(_, &ci)| ci < vol_cols)
                    .map(|(i, &ci)| (i, ci))
                    .collect();
                for i in 0..vol.row_count {
                    let rid = vol.row_ids[i];
                    if ts.contains_key(&rid) || !seen.insert(rid) {
                        continue;
                    }
                    let matches = present_cols.iter().all(|&(val_idx, ci)| {
                        let v = vol.columns[ci].get_value(i);
                        !v.is_null() && v == *values[val_idx]
                    });
                    if matches {
                        vol_result = Some(rid);
                        break;
                    }
                }
            }
            if let Some(rid) = vol_result {
                // Verify this is the authoritative version. After UPDATE old→new
                // + seal, overlapping volumes can have the same row_id with
                // different values. The older volume's stale value is not
                // tombstoned (tombstone cleared when row_id appeared in the newer
                // volume). get_cold_row returns the newest version (newest-first).
                // Use column_defaults for columns missing from schema-evolved volumes.
                if seg_ids.len() > 1 {
                    if let Some(current_row) = self.get_cold_row(rid) {
                        let still_matches = col_indices.iter().enumerate().all(|(i, &ci)| {
                            let v = current_row
                                .get(ci)
                                .cloned()
                                .unwrap_or_else(|| column_defaults[i].clone());
                            !v.is_null() && v == *values[i]
                        });
                        if !still_matches {
                            continue; // stale value in older volume, skip
                        }
                    }
                }
                return Some(rid);
            }
        }
        None
    }

    /// Get the number of segments.
    pub fn segment_count(&self) -> usize {
        self.manifest.read().segments.len()
    }

    /// Get the oldest compaction_epoch across all segments.
    pub fn oldest_segment_epoch(&self) -> Option<u64> {
        let manifest = self.manifest.read();
        manifest.segments.iter().map(|s| s.compaction_epoch).min()
    }

    /// Check if a segment with the given ID is already registered (loaded in memory).
    pub fn has_segment(&self, segment_id: u64) -> bool {
        self.segments.read().contains_key(&segment_id)
    }

    /// Check if a segment exists in the manifest (source of truth for what should be loaded).
    /// Cheaper than `load_volume_for_existing_segment` — no volume data needed.
    pub fn manifest_has_segment(&self, segment_id: u64) -> bool {
        self.manifest
            .read()
            .segments
            .iter()
            .any(|s| s.segment_id == segment_id)
    }

    /// Register a new segment (after seal, compaction, or load).
    /// When `invalidate_cache` is false (compaction), the unique lookup cache is
    /// preserved. Compaction doesn't change row_ids or values, just which volume
    /// they live in. The cache's `row_exists()` filter handles stale entries.
    pub fn register_segment(&self, segment_id: u64, volume: Arc<FrozenVolume>, meta: SegmentMeta) {
        self.register_segment_inner(segment_id, volume, meta, true);
    }

    /// Register without invalidating the unique lookup cache (for compaction).
    pub fn register_compacted_segment(
        &self,
        segment_id: u64,
        volume: Arc<FrozenVolume>,
        meta: SegmentMeta,
    ) {
        self.register_segment_inner(segment_id, volume, meta, false);
    }

    fn register_segment_inner(
        &self,
        segment_id: u64,
        volume: Arc<FrozenVolume>,
        meta: SegmentMeta,
        _invalidate_cache: bool,
    ) {
        {
            let mut manifest = self.manifest.write();
            if segment_id >= manifest.next_segment_id {
                manifest.next_segment_id = segment_id + 1;
            }
            manifest.add_segment(meta);
            let mut segments = self.segments.write();
            let mut new_map = (**segments).clone();
            new_map.insert(segment_id, volume);
            *segments = Arc::new(new_map);
        }
        self.cached_deduped_count
            .store(u64::MAX, std::sync::atomic::Ordering::Relaxed);
        self.has_segments_flag
            .store(true, std::sync::atomic::Ordering::Relaxed);
        self.seal_generation
            .fetch_add(1, std::sync::atomic::Ordering::Release);
    }

    /// Load a volume into the segments map for an existing manifest entry.
    /// Returns true if the segment_id was found in the manifest and loaded,
    /// false if the segment_id is not in the manifest.
    /// This avoids adding duplicate segment metadata when the manifest was
    /// pre-loaded from disk and volumes are loaded separately.
    pub fn load_volume_for_existing_segment(
        &self,
        segment_id: u64,
        volume: Arc<FrozenVolume>,
    ) -> bool {
        let manifest = self.manifest.read();
        let exists_in_manifest = manifest.segments.iter().any(|s| s.segment_id == segment_id);
        drop(manifest);

        if exists_in_manifest {
            let mut segments = self.segments.write();
            let mut new_map = (**segments).clone();
            new_map.insert(segment_id, volume);
            *segments = Arc::new(new_map);
            self.has_segments_flag
                .store(true, std::sync::atomic::Ordering::Relaxed);

            true
        } else {
            false
        }
    }

    /// Rename this segment manager's table (for ALTER TABLE RENAME).
    pub fn rename(&mut self, new_name: &str) {
        self.table_name = SmartString::from(new_name);
        self.manifest.write().table_name = SmartString::from(new_name);
    }

    /// Add tombstone row_ids with their commit_seq (when the tombstone was created).
    /// Lock order: manifest FIRST, then tombstones (matches read paths like
    /// deduped_row_count, total_row_count, check_value_exists_in_segments).
    /// The commit_seq enables snapshot isolation: older snapshots don't see
    /// newer tombstones, so the original cold row remains visible to them.
    pub fn add_tombstones(&self, row_ids: &[i64], commit_seq: u64) {
        if row_ids.is_empty() {
            return;
        }
        let mut manifest = self.manifest.write();
        let mut ts_guard = self.tombstones.write();
        let ts = Arc::make_mut(&mut *ts_guard);
        let mut changed = false;
        for &rid in row_ids {
            use std::collections::hash_map::Entry;
            match ts.entry(rid) {
                Entry::Vacant(e) => {
                    e.insert(commit_seq);
                    manifest.tombstones.push((rid, commit_seq));
                    changed = true;
                }
                Entry::Occupied(mut e) => {
                    // Update existing tombstone if the new commit_seq is
                    // different. This ensures repeated seal-skip tombstones
                    // get a fresh sequence that won't match an older
                    // compaction snapshot.
                    if *e.get() != commit_seq {
                        let old_seq = *e.get();
                        e.insert(commit_seq);
                        // Update the manifest entry in-place.
                        if let Some(entry) = manifest
                            .tombstones
                            .iter_mut()
                            .find(|(r, s)| *r == rid && *s == old_seq)
                        {
                            entry.1 = commit_seq;
                        }
                        changed = true;
                    }
                }
            }
        }
        if changed {
            self.cached_deduped_count
                .store(u64::MAX, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Clear all tombstones (after compaction has resolved them).
    /// Lock order: manifest FIRST, then tombstones.
    pub fn clear_tombstones(&self) {
        self.manifest.write().tombstones.clear();
        *self.tombstones.write() = Arc::new(FxHashMap::default());
        self.cached_deduped_count
            .store(u64::MAX, std::sync::atomic::Ordering::Relaxed);
    }

    /// Remove tombstones only for row_ids that were in the compacted volumes.
    /// Used by partial compaction (merging a subset of volumes) where tombstones
    /// for unmerged volumes must remain.
    pub fn remove_tombstones_for_rows(&self, row_ids: &FxHashSet<i64>) {
        if row_ids.is_empty() {
            return;
        }
        let mut manifest = self.manifest.write();
        let mut ts_guard = self.tombstones.write();
        let ts = Arc::make_mut(&mut *ts_guard);
        let before = ts.len();
        ts.retain(|rid, _| !row_ids.contains(rid));
        if ts.len() != before {
            manifest
                .tombstones
                .retain(|&(rid, _)| !row_ids.contains(&rid));
            self.cached_deduped_count
                .store(u64::MAX, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Remove only tombstones that match both row_id AND commit_seq from a
    /// prior snapshot. Tombstones added after the snapshot (e.g., by a
    /// concurrent seal) are preserved.
    pub fn remove_tombstones_matching_snapshot(
        &self,
        snapshot: &FxHashMap<i64, u64>,
        row_ids: &FxHashSet<i64>,
    ) {
        if row_ids.is_empty() || snapshot.is_empty() {
            return;
        }
        let mut manifest = self.manifest.write();
        let mut ts_guard = self.tombstones.write();
        let ts = Arc::make_mut(&mut *ts_guard);
        let before = ts.len();
        ts.retain(|rid, seq| {
            if !row_ids.contains(rid) {
                return true; // not in merged volumes, keep
            }
            // Only remove if the commit_seq matches the snapshot.
            // If a newer tombstone was added (different seq), keep it.
            !matches!(snapshot.get(rid), Some(snap_seq) if *snap_seq == *seq)
        });
        if ts.len() != before {
            manifest.tombstones.retain(|&(rid, seq)| {
                if !row_ids.contains(&rid) {
                    return true;
                }
                !matches!(snapshot.get(&rid), Some(snap_seq) if *snap_seq == seq)
            });
            self.cached_deduped_count
                .store(u64::MAX, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Get an Arc reference to the tombstone map. O(1) — no data clone.
    /// Use this for read-only access (membership checks, iteration).
    /// Keys are row_ids, values are commit_seq (for snapshot filtering).
    pub fn tombstone_set_arc(&self) -> Arc<FxHashMap<i64, u64>> {
        Arc::clone(&*self.tombstones.read())
    }

    /// Check if the tombstone set is empty without cloning.
    pub fn is_tombstone_set_empty(&self) -> bool {
        self.tombstones.read().is_empty()
    }

    /// Get write access to the tombstone map (for seal cleanup).
    pub fn tombstones_write(&self) -> parking_lot::RwLockWriteGuard<'_, Arc<FxHashMap<i64, u64>>> {
        self.tombstones.write()
    }

    // ---- Per-transaction pending tombstones ----

    /// Track a cold row_id as pending tombstone for a transaction.
    /// Called during DML (UPDATE/DELETE of cold rows).
    pub fn add_pending_tombstone(&self, txn_id: i64, row_id: i64) {
        self.pending_txn_tombstones
            .write()
            .entry(txn_id)
            .or_default()
            .insert(row_id);
    }

    /// Get pending tombstone row_ids for a transaction (for WAL recording).
    pub fn get_pending_tombstones(&self, txn_id: i64) -> Vec<i64> {
        self.pending_txn_tombstones
            .read()
            .get(&txn_id)
            .map(|set| set.iter().copied().collect())
            .unwrap_or_default()
    }

    /// Insert pending tombstones for a transaction directly into a set (no Vec clone).
    pub fn insert_pending_tombstones_into(
        &self,
        txn_id: i64,
        dest: &mut rustc_hash::FxHashSet<i64>,
    ) {
        if let Some(ids) = self.pending_txn_tombstones.read().get(&txn_id) {
            for &id in ids {
                dest.insert(id);
            }
        }
    }

    /// Get the count of pending tombstones for a transaction without cloning.
    pub fn pending_tombstone_count(&self, txn_id: i64) -> usize {
        self.pending_txn_tombstones
            .read()
            .get(&txn_id)
            .map_or(0, |v| v.len())
    }

    /// Check if a specific row_id is a pending tombstone for a transaction.
    /// O(1) with FxHashSet (was O(n) with Vec).
    pub fn is_pending_tombstone(&self, txn_id: i64, row_id: i64) -> bool {
        self.pending_txn_tombstones
            .read()
            .get(&txn_id)
            .is_some_and(|set| set.contains(&row_id))
    }

    /// Commit pending tombstones: move from per-txn pending to shared tombstone set.
    /// The commit_seq is the transaction's commit sequence, used for snapshot
    /// isolation: older snapshots won't see these tombstones.
    pub fn commit_pending_tombstones(&self, txn_id: i64, commit_seq: u64) {
        let pending = self.pending_txn_tombstones.write().remove(&txn_id);
        if let Some(ids) = pending {
            if !ids.is_empty() {
                let id_vec: Vec<i64> = ids.into_iter().collect();
                self.add_tombstones(&id_vec, commit_seq);
            }
        }
    }

    /// Rollback pending tombstones: discard without applying.
    pub fn rollback_pending_tombstones(&self, txn_id: i64) {
        self.pending_txn_tombstones.write().remove(&txn_id);
    }

    /// Check if a txn has any pending tombstones (for has_local_changes).
    pub fn has_pending_tombstones(&self, txn_id: i64) -> bool {
        self.pending_txn_tombstones
            .read()
            .get(&txn_id)
            .is_some_and(|v| !v.is_empty())
    }

    /// Check if a row_id is tombstoned (any commit_seq).
    pub fn is_tombstoned(&self, row_id: i64) -> bool {
        self.tombstones.read().contains_key(&row_id)
    }

    /// Check if a row_id exists in any segment (not tombstoned).
    ///
    /// Used for constraint checking (PK/UNIQUE).
    pub fn row_exists(&self, row_id: i64) -> bool {
        let ts = Arc::clone(&*self.tombstones.read());
        if ts.contains_key(&row_id) {
            return false;
        }
        let (seg_ids, segments) = {
            let manifest = self.manifest.read();
            let seg_ids: Vec<(u64, i64, i64)> = manifest
                .segments
                .iter()
                .map(|m| (m.segment_id, m.min_row_id, m.max_row_id))
                .collect();
            let segments = Arc::clone(&*self.segments.read());
            (seg_ids, segments)
        };
        for (seg_id, min_id, max_id) in &seg_ids {
            if row_id < *min_id || row_id > *max_id {
                continue;
            }
            if let Some(vol) = segments.get(seg_id) {
                if vol.row_ids.binary_search(&row_id).is_ok() {
                    return true;
                }
            }
        }
        false
    }

    /// Get a cold row by row_id. Returns the Row if found and not tombstoned.
    /// Iterates newest-first so overlapping row_ids return the newest version.
    pub fn get_cold_row(&self, row_id: i64) -> Option<crate::core::Row> {
        let ts = Arc::clone(&*self.tombstones.read());
        if ts.contains_key(&row_id) {
            return None;
        }
        // Hold manifest lock while cloning segments Arc to prevent
        // compaction from swapping the map between the two reads.
        let (seg_ids, segments) = {
            let manifest = self.manifest.read();
            let seg_ids: Vec<(u64, i64, i64)> = manifest
                .segments
                .iter()
                .rev()
                .map(|m| (m.segment_id, m.min_row_id, m.max_row_id))
                .collect();
            let segments = Arc::clone(&*self.segments.read());
            (seg_ids, segments)
        };
        for (seg_id, min_id, max_id) in &seg_ids {
            if row_id < *min_id || row_id > *max_id {
                continue;
            }
            if let Some(vol) = segments.get(seg_id) {
                if let Ok(idx) = vol.row_ids.binary_search(&row_id) {
                    return Some(vol.get_row(idx));
                }
            }
        }
        None
    }

    /// Get a cold row by row_id, normalized to the current schema.
    /// After ALTER TABLE ADD COLUMN, cold volumes may have fewer columns.
    /// This variant fills in defaults for missing columns.
    /// Iterates newest-first so overlapping row_ids return the newest version.
    pub fn get_cold_row_normalized(
        &self,
        row_id: i64,
        schema: &crate::core::Schema,
    ) -> Option<crate::core::Row> {
        let ts = Arc::clone(&*self.tombstones.read());
        if ts.contains_key(&row_id) {
            return None;
        }
        // Hold manifest lock while cloning segments Arc to prevent
        // compaction from swapping the map between the two reads.
        let (seg_ids, segments) = {
            let manifest = self.manifest.read();
            let seg_ids: Vec<(u64, i64, i64)> = manifest
                .segments
                .iter()
                .rev()
                .map(|m| (m.segment_id, m.min_row_id, m.max_row_id))
                .collect();
            let segments = Arc::clone(&*self.segments.read());
            (seg_ids, segments)
        };
        for (seg_id, min_id, max_id) in &seg_ids {
            if row_id < *min_id || row_id > *max_id {
                continue;
            }
            if let Some(vol) = segments.get(seg_id) {
                if let Ok(idx) = vol.row_ids.binary_search(&row_id) {
                    let mapping = super::writer::compute_column_mapping(schema, vol);
                    if mapping.is_identity {
                        return Some(vol.get_row(idx));
                    }
                    return Some(vol.get_row_mapped(idx, &mapping));
                }
            }
        }
        None
    }

    /// Check if a row_id actually exists in any loaded volume.
    /// Does NOT check tombstones. Used for idempotent WAL replay.
    /// Uses binary search on the volume's row_ids for O(log n) per segment.
    pub fn is_row_id_in_volume(&self, row_id: i64) -> bool {
        let (seg_ids, segments) = {
            let manifest = self.manifest.read();
            let seg_ids: Vec<(u64, i64, i64)> = manifest
                .segments
                .iter()
                .map(|m| (m.segment_id, m.min_row_id, m.max_row_id))
                .collect();
            let segments = Arc::clone(&*self.segments.read());
            (seg_ids, segments)
        };
        for (seg_id, min_id, max_id) in &seg_ids {
            if row_id < *min_id || row_id > *max_id {
                continue;
            }
            if let Some(vol) = segments.get(seg_id) {
                if vol.row_ids.binary_search(&row_id).is_ok() {
                    return true;
                }
            }
        }
        false
    }

    /// Get the total live row count across all segments (minus tombstones).
    /// NOTE: This is a fast estimate that does not deduplicate overlapping row_ids.
    /// Use `deduped_row_count()` for an exact count.
    pub fn total_row_count(&self) -> usize {
        let manifest = self.manifest.read();
        let ts_count = self.tombstones.read().len();
        let total: usize = manifest.segments.iter().map(|s| s.row_count).sum();
        total.saturating_sub(ts_count)
    }

    /// Get the exact deduplicated row count across all segments.
    /// Uses a cached value that is invalidated on segment/tombstone changes.
    pub fn deduped_row_count(&self) -> usize {
        let cached = self
            .cached_deduped_count
            .load(std::sync::atomic::Ordering::Relaxed);
        if cached != u64::MAX {
            return cached as usize;
        }
        let count = self.compute_deduped_row_count();
        self.cached_deduped_count
            .store(count as u64, std::sync::atomic::Ordering::Relaxed);
        count
    }

    /// Acquire a shared guard while performing a cold check + hot insert.
    /// Seal takes the exclusive guard so it cannot move rows between the
    /// cold visibility check and hot publication.
    #[inline]
    pub fn acquire_seal_read(&self) -> parking_lot::RwLockReadGuard<'_, ()> {
        self.seal_fence.read()
    }

    /// Acquire the exclusive guard for the seal critical section.
    #[inline]
    pub fn acquire_seal_write(&self) -> parking_lot::RwLockWriteGuard<'_, ()> {
        self.seal_fence.write()
    }

    /// Current seal generation. Incremented on every register_segment.
    #[inline]
    pub fn seal_generation(&self) -> u64 {
        self.seal_generation
            .load(std::sync::atomic::Ordering::Acquire)
    }

    /// Record the current seal generation for a transaction. Called under
    /// the seal read fence during INSERT so the value is consistent.
    /// Stores the minimum (earliest) generation seen by this txn, so that
    /// a later INSERT within the same txn cannot hide an earlier seal.
    #[inline]
    pub fn record_txn_seal_generation(&self, txn_id: i64) {
        let gen = self.seal_generation();
        let mut map = self.txn_seal_gens.lock();
        map.entry(txn_id)
            .and_modify(|existing| {
                if gen < *existing {
                    *existing = gen;
                }
            })
            .or_insert(gen);
    }

    /// Get the seal generation recorded for a transaction.
    #[inline]
    pub fn get_txn_seal_generation(&self, txn_id: i64) -> Option<u64> {
        self.txn_seal_gens.lock().get(&txn_id).copied()
    }

    /// Remove the seal generation record for a transaction (on commit/rollback).
    #[inline]
    pub fn clear_txn_seal_generation(&self, txn_id: i64) {
        self.txn_seal_gens.lock().remove(&txn_id);
    }

    /// Scan all row_ids, deduplicate (newest-first wins), and exclude tombstones.
    fn compute_deduped_row_count(&self) -> usize {
        let (seg_ids, segments) = {
            let manifest = self.manifest.read();
            let seg_ids: Vec<u64> = manifest
                .segments
                .iter()
                .rev()
                .map(|m| m.segment_id)
                .collect();
            let segments = Arc::clone(&*self.segments.read());
            (seg_ids, segments)
        };
        if segments.is_empty() {
            return 0;
        }
        let tombstones = Arc::clone(&*self.tombstones.read());
        if segments.len() == 1 {
            let total: usize = segments.values().map(|v| v.row_count).sum();
            return total.saturating_sub(tombstones.len());
        }
        let mut seen = FxHashSet::default();

        let mut count = 0usize;
        for seg_id in &seg_ids {
            let Some(vol) = segments.get(seg_id) else {
                continue;
            };
            for &rid in &vol.row_ids {
                if tombstones.contains_key(&rid) {
                    continue;
                }
                if seen.insert(rid) {
                    count += 1;
                }
            }
        }
        count
    }

    /// Set the seal overlap count (rows that exist in both hot and cold during seal).
    /// Called BEFORE register_segment with the number of rows being sealed.
    pub fn set_seal_overlap(&self, count: usize) {
        self.seal_overlap_count
            .store(count, std::sync::atomic::Ordering::Release);
    }

    /// Clear the seal overlap count. Called AFTER remove_sealed_rows completes.
    pub fn clear_seal_overlap(&self) {
        self.seal_overlap_count
            .store(0, std::sync::atomic::Ordering::Release);
    }

    /// Get the current seal overlap count (for row_count correction).
    pub fn seal_overlap(&self) -> usize {
        self.seal_overlap_count
            .load(std::sync::atomic::Ordering::Acquire)
    }

    /// Persist the manifest to disk (includes tombstones).
    pub fn persist(&self) -> Result<()> {
        self.persist_manifest_only()
    }

    /// Persist only the manifest.
    pub fn persist_manifest_only(&self) -> Result<()> {
        let Some(ref vol_dir) = self.volume_dir else {
            return Ok(());
        };

        let persist_name = self.manifest.read().table_name.clone();
        let table_dir = vol_dir.join(persist_name.as_str());
        std::fs::create_dir_all(&table_dir).map_err(|e| {
            crate::core::Error::internal(format!("failed to create table dir: {}", e))
        })?;

        let manifest_path = table_dir.join("manifest.bin");
        self.manifest.read().write_to_disk(&manifest_path)?;

        Ok(())
    }

    /// Load manifest from disk.
    pub fn load_from_disk(table_name: &str, volume_dir: &Path) -> Result<Option<Self>> {
        let table_dir = volume_dir.join(table_name);
        let manifest_path = table_dir.join("manifest.bin");

        if !manifest_path.exists() {
            return Ok(None);
        }

        let manifest = TableManifest::read_from_disk(&manifest_path)?;
        let manager = Self::from_manifest(manifest, Some(volume_dir.to_path_buf()));

        Ok(Some(manager))
    }

    /// Remove all segments and tombstones (for DROP TABLE / TRUNCATE).
    pub fn clear(&self) {
        {
            let mut manifest = self.manifest.write();
            manifest.segments.clear();
            manifest.tombstones.clear();
        }
        *self.segments.write() = Arc::new(FxHashMap::default());
        *self.tombstones.write() = Arc::new(FxHashMap::default());
        self.cached_deduped_count
            .store(u64::MAX, std::sync::atomic::Ordering::Relaxed);
        self.has_segments_flag
            .store(false, std::sync::atomic::Ordering::Relaxed);
    }

    /// Remove specific segments after compaction.
    pub fn remove_segments(&self, segment_ids: &[u64]) {
        self.remove_segments_inner(segment_ids, true);
    }

    /// Remove segments without invalidating the unique lookup cache (for compaction).
    pub fn remove_segments_compacted(&self, segment_ids: &[u64]) {
        self.remove_segments_inner(segment_ids, false);
    }

    fn remove_segments_inner(&self, segment_ids: &[u64], _invalidate_cache: bool) {
        self.manifest.write().remove_segments(segment_ids);
        {
            let mut segments = self.segments.write();
            let mut new_map = (**segments).clone();
            for &id in segment_ids {
                new_map.remove(&id);
            }
            *segments = Arc::new(new_map);
        }
        self.cached_deduped_count
            .store(u64::MAX, std::sync::atomic::Ordering::Relaxed);
    }

    /// Atomically replace old segments with a new compacted segment.
    /// Both operations happen under a single manifest+segments write lock,
    /// so concurrent queries never see the intermediate state where both
    /// old and new segments are registered (which causes duplicate scanning).
    pub fn replace_segments_atomic(
        &self,
        new_segment_id: u64,
        new_volume: Arc<FrozenVolume>,
        new_meta: SegmentMeta,
        old_segment_ids: &[u64],
    ) {
        {
            let mut manifest = self.manifest.write();
            // Find insertion point: the compacted volume replaces the oldest
            // merged segments, so it must go at the position of the first old
            // segment. If placed at the end (like add_segment does), the
            // newest-first scan would treat it as newer than unmerged volumes,
            // returning stale data instead of the latest version.
            let insert_pos = manifest
                .segments
                .iter()
                .position(|s| old_segment_ids.contains(&s.segment_id))
                .unwrap_or(manifest.segments.len());
            manifest.remove_segments(old_segment_ids);
            if new_segment_id >= manifest.next_segment_id {
                manifest.next_segment_id = new_segment_id + 1;
            }
            // Insert at the position of the old segments, not at the end.
            let insert_pos = insert_pos.min(manifest.segments.len());
            manifest.segments.insert(insert_pos, new_meta);

            // Swap segments map under the same logical operation
            let mut segments = self.segments.write();
            let mut new_map = (**segments).clone();
            for &id in old_segment_ids {
                new_map.remove(&id);
            }
            new_map.insert(new_segment_id, new_volume);
            *segments = Arc::new(new_map);
        }
        self.cached_deduped_count
            .store(u64::MAX, std::sync::atomic::Ordering::Relaxed);
        // has_segments_flag remains true (we just added a segment)
    }

    /// Atomically remove old segments without adding a replacement.
    /// Used when partial compaction finds all rows in merged volumes are tombstoned.
    pub fn replace_segments_atomic_remove_only(&self, old_segment_ids: &[u64]) {
        {
            let mut manifest = self.manifest.write();
            manifest.remove_segments(old_segment_ids);

            let mut segments = self.segments.write();
            let mut new_map = (**segments).clone();
            for &id in old_segment_ids {
                new_map.remove(&id);
            }
            let has_any = !new_map.is_empty();
            *segments = Arc::new(new_map);
            self.has_segments_flag
                .store(has_any, std::sync::atomic::Ordering::Relaxed);
        }
        self.cached_deduped_count
            .store(u64::MAX, std::sync::atomic::Ordering::Relaxed);
    }

    /// Get the manifest for reading (e.g., to iterate segment metadata).
    pub fn manifest(&self) -> parking_lot::RwLockReadGuard<'_, TableManifest> {
        self.manifest.read()
    }

    /// CoW snapshot of the loaded segments map.
    pub fn segments_snapshot(&self) -> Arc<FxHashMap<u64, Arc<FrozenVolume>>> {
        Arc::clone(&*self.segments.read())
    }

    /// Get the manifest for writing (e.g., to allocate segment IDs).
    pub fn manifest_mut(&self) -> parking_lot::RwLockWriteGuard<'_, TableManifest> {
        self.manifest.write()
    }

    /// Get the volume directory path.
    pub fn volume_dir(&self) -> Option<&Path> {
        self.volume_dir.as_deref()
    }
}

impl std::fmt::Debug for SegmentManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let manifest = self.manifest.read();
        f.debug_struct("SegmentManager")
            .field("table", &self.table_name)
            .field("segments", &manifest.segments.len())
            .field("next_id", &manifest.next_segment_id)
            .field("tombstones", &self.tombstones.read().len())
            .finish()
    }
}

// Helper functions for binary deserialization — return errors on truncation
#[inline]
fn read_u32(data: &[u8], pos: &mut usize) -> std::io::Result<u32> {
    let end = *pos + 4;
    if end > data.len() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "truncated manifest: expected u32",
        ));
    }
    let v = u32::from_le_bytes([data[*pos], data[*pos + 1], data[*pos + 2], data[*pos + 3]]);
    *pos = end;
    Ok(v)
}

#[inline]
fn read_u64(data: &[u8], pos: &mut usize) -> std::io::Result<u64> {
    let end = *pos + 8;
    if end > data.len() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "truncated manifest: expected u64",
        ));
    }
    let v = u64::from_le_bytes([
        data[*pos],
        data[*pos + 1],
        data[*pos + 2],
        data[*pos + 3],
        data[*pos + 4],
        data[*pos + 5],
        data[*pos + 6],
        data[*pos + 7],
    ]);
    *pos = end;
    Ok(v)
}

#[inline]
fn read_i64(data: &[u8], pos: &mut usize) -> std::io::Result<i64> {
    let end = *pos + 8;
    if end > data.len() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "truncated manifest: expected i64",
        ));
    }
    let v = i64::from_le_bytes([
        data[*pos],
        data[*pos + 1],
        data[*pos + 2],
        data[*pos + 3],
        data[*pos + 4],
        data[*pos + 5],
        data[*pos + 6],
        data[*pos + 7],
    ]);
    *pos = end;
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manifest_new() {
        let m = TableManifest::new("test_table");
        assert_eq!(m.table_name.as_str(), "test_table");
        assert!(m.segments.is_empty());
        assert_eq!(m.next_segment_id, 1);
        assert_eq!(m.checkpoint_lsn, 0);
        assert!(m.tombstones.is_empty());
    }

    #[test]
    fn test_manifest_allocate_id() {
        let mut m = TableManifest::new("t");
        assert_eq!(m.allocate_segment_id(), 1);
        assert_eq!(m.allocate_segment_id(), 2);
        assert_eq!(m.allocate_segment_id(), 3);
        assert_eq!(m.next_segment_id, 4);
    }

    #[test]
    fn test_manifest_add_remove_segment() {
        let mut m = TableManifest::new("t");
        m.add_segment(SegmentMeta {
            segment_id: 1,
            file_path: PathBuf::from("vol_001.vol"),
            row_count: 1000,
            min_row_id: 1,
            max_row_id: 1000,
            creation_lsn: 100,
            compaction_epoch: 0,
        });
        m.add_segment(SegmentMeta {
            segment_id: 2,
            file_path: PathBuf::from("vol_002.vol"),
            row_count: 500,
            min_row_id: 1001,
            max_row_id: 1500,
            creation_lsn: 200,
            compaction_epoch: 0,
        });
        assert_eq!(m.segments.len(), 2);

        m.remove_segments(&[1]);
        assert_eq!(m.segments.len(), 1);
        assert_eq!(m.segments[0].segment_id, 2);
    }

    #[test]
    fn test_manifest_find_segment() {
        let mut m = TableManifest::new("t");
        m.add_segment(SegmentMeta {
            segment_id: 1,
            file_path: PathBuf::from("a.vol"),
            row_count: 100,
            min_row_id: 1,
            max_row_id: 100,
            creation_lsn: 0,
            compaction_epoch: 0,
        });
        m.add_segment(SegmentMeta {
            segment_id: 2,
            file_path: PathBuf::from("b.vol"),
            row_count: 100,
            min_row_id: 101,
            max_row_id: 200,
            creation_lsn: 0,
            compaction_epoch: 0,
        });

        assert_eq!(m.find_segment_for_row_id(50).unwrap().1.segment_id, 1);
        assert_eq!(m.find_segment_for_row_id(150).unwrap().1.segment_id, 2);
        assert!(m.find_segment_for_row_id(300).is_none());
    }

    #[test]
    fn test_manifest_serialize_roundtrip() {
        let mut m = TableManifest::new("my_table");
        m.next_segment_id = 5;
        m.checkpoint_lsn = 42;
        m.tombstones = vec![(10, 0), (20, 0), (30, 0)];
        m.add_segment(SegmentMeta {
            segment_id: 1,
            file_path: PathBuf::from("seg_0001.vol"),
            row_count: 10000,
            min_row_id: 1,
            max_row_id: 10000,
            creation_lsn: 10,
            compaction_epoch: 0,
        });
        m.add_segment(SegmentMeta {
            segment_id: 3,
            file_path: PathBuf::from("seg_0003.vol"),
            row_count: 5000,
            min_row_id: 10001,
            max_row_id: 15000,
            creation_lsn: 30,
            compaction_epoch: 0,
        });

        let data = m.serialize().unwrap();
        let loaded = TableManifest::deserialize(&data).unwrap();

        assert_eq!(loaded.table_name.as_str(), "my_table");
        assert_eq!(loaded.next_segment_id, 5);
        assert_eq!(loaded.checkpoint_lsn, 42);
        assert_eq!(loaded.segments.len(), 2);
        assert_eq!(loaded.segments[0].segment_id, 1);
        assert_eq!(loaded.segments[0].row_count, 10000);
        assert_eq!(loaded.segments[0].min_row_id, 1);
        assert_eq!(loaded.segments[0].max_row_id, 10000);
        assert_eq!(loaded.segments[0].file_path, PathBuf::from("seg_0001.vol"));
        assert_eq!(loaded.segments[1].segment_id, 3);
        assert_eq!(loaded.segments[1].creation_lsn, 30);
        assert_eq!(loaded.tombstones, vec![(10, 0), (20, 0), (30, 0)]);
    }

    #[test]
    fn test_manifest_disk_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("manifest.bin");

        let mut m = TableManifest::new("disk_test");
        m.tombstones = vec![(5, 0), (10, 0)];
        m.add_segment(SegmentMeta {
            segment_id: 1,
            file_path: PathBuf::from("vol.vol"),
            row_count: 100,
            min_row_id: 1,
            max_row_id: 100,
            creation_lsn: 0,
            compaction_epoch: 0,
        });

        m.write_to_disk(&path).unwrap();
        let loaded = TableManifest::read_from_disk(&path).unwrap();

        assert_eq!(loaded.table_name.as_str(), "disk_test");
        assert_eq!(loaded.segments.len(), 1);
        assert_eq!(loaded.tombstones, vec![(5, 0), (10, 0)]);
    }

    #[test]
    fn test_segment_manager_register_and_query() {
        use crate::core::{DataType, Row, SchemaBuilder, Value};

        let schema = SchemaBuilder::new("test")
            .column("id", DataType::Integer, false, true)
            .build();

        let mut builder = super::super::writer::VolumeBuilder::new(&schema);
        for i in 1..=10i64 {
            builder.add_row(i, &Row::from_values(vec![Value::Integer(i)]));
        }
        let volume = Arc::new(builder.finish());

        let mgr = SegmentManager::new("test", None);
        let meta = SegmentMeta {
            segment_id: 1,
            file_path: PathBuf::from("test.vol"),
            row_count: 10,
            min_row_id: 1,
            max_row_id: 10,
            creation_lsn: 0,
            compaction_epoch: 0,
        };
        mgr.register_segment(1, volume, meta);

        assert_eq!(mgr.segment_count(), 1);
        assert_eq!(mgr.total_row_count(), 10);
        assert!(mgr.row_exists(5));
        assert!(!mgr.row_exists(11));
    }

    #[test]
    fn test_segment_manager_tombstones() {
        use crate::core::{DataType, Row, SchemaBuilder, Value};

        let schema = SchemaBuilder::new("test")
            .column("id", DataType::Integer, false, true)
            .build();

        let mut builder = super::super::writer::VolumeBuilder::new(&schema);
        for i in 1..=10i64 {
            builder.add_row(i, &Row::from_values(vec![Value::Integer(i)]));
        }
        let volume = Arc::new(builder.finish());

        let mgr = SegmentManager::new("test", None);
        mgr.register_segment(
            1,
            volume,
            SegmentMeta {
                segment_id: 1,
                file_path: PathBuf::from("test.vol"),
                row_count: 10,
                min_row_id: 1,
                max_row_id: 10,
                creation_lsn: 0,
                compaction_epoch: 0,
            },
        );

        // Tombstone row_id=5 (commit_seq=1)
        mgr.add_tombstones(&[5], 1);
        assert!(!mgr.row_exists(5));
        assert!(mgr.row_exists(4));
        assert!(mgr.row_exists(6));
        assert_eq!(mgr.total_row_count(), 9);
        assert!(mgr.is_tombstoned(5));
        assert!(!mgr.is_tombstoned(4));

        // Clear tombstones
        mgr.clear_tombstones();
        assert!(mgr.row_exists(5));
        assert_eq!(mgr.total_row_count(), 10);
    }

    #[test]
    fn test_segment_manager_volumes_newest_first() {
        use crate::core::{DataType, Row, SchemaBuilder, Value};

        let schema = SchemaBuilder::new("test")
            .column("id", DataType::Integer, false, true)
            .build();

        let mgr = SegmentManager::new("test", None);

        for seg_id in [1u64, 3, 2] {
            let mut builder = super::super::writer::VolumeBuilder::new(&schema);
            builder.add_row(
                seg_id as i64,
                &Row::from_values(vec![Value::Integer(seg_id as i64)]),
            );
            let vol = Arc::new(builder.finish());
            mgr.register_segment(
                seg_id,
                vol,
                SegmentMeta {
                    segment_id: seg_id,
                    file_path: PathBuf::from(format!("vol_{}.vol", seg_id)),
                    row_count: 1,
                    min_row_id: seg_id as i64,
                    max_row_id: seg_id as i64,
                    creation_lsn: 0,
                    compaction_epoch: 0,
                },
            );
        }

        let newest_first = mgr.get_volumes_newest_first();
        assert_eq!(newest_first.len(), 3);
        // get_volumes_newest_first reverses manifest insertion order.
        // Segments were registered as [1, 3, 2], so reversed = [2, 3, 1].
        assert_eq!(newest_first[0].0, 2);
        assert_eq!(newest_first[1].0, 3);
        assert_eq!(newest_first[2].0, 1);
    }

    #[test]
    fn test_segment_manager_clear() {
        let mgr = SegmentManager::new("test", None);
        mgr.manifest.write().add_segment(SegmentMeta {
            segment_id: 1,
            file_path: PathBuf::from("x.vol"),
            row_count: 10,
            min_row_id: 1,
            max_row_id: 10,
            creation_lsn: 0,
            compaction_epoch: 0,
        });
        mgr.add_tombstones(&[5], 1);

        assert_eq!(mgr.segment_count(), 1);
        mgr.clear();
        assert_eq!(mgr.segment_count(), 0);
        assert!(mgr.tombstone_set_arc().is_empty());
    }
}
