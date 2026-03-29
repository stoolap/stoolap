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

//! Seal operation: freezes hot buffer rows into a frozen volume.
//!
//! When the hot buffer reaches a size threshold, its committed rows are
//! frozen into a column-major FrozenVolume, written to disk as a `.vol`
//! file, and the corresponding arena entries are cleared.
//!
//! This runs as a periodic background operation, similar to snapshot creation.

use std::sync::Arc;

use crate::core::{Result, Row, Schema};

use super::io;
use super::writer::{FrozenVolume, VolumeBuilder};

/// Seal hot buffer rows into a frozen volume.
///
/// Takes committed rows from a VersionStore and creates a FrozenVolume.
/// The caller is responsible for clearing the sealed rows from the arena
/// after the volume is safely written to disk.
///
/// # Arguments
/// * `schema` - Table schema
/// * `rows` - Committed rows to freeze (row_id, row_data)
///
/// # Returns
/// A FrozenVolume ready to be queried and/or written to disk.
pub fn seal_rows(schema: &Schema, rows: &[(i64, Row)]) -> FrozenVolume {
    let mut builder = VolumeBuilder::with_capacity(schema, rows.len());
    for (row_id, row) in rows {
        builder.add_row(*row_id, row);
    }
    builder.finish()
}

/// Seal hot buffer rows and write the volume to disk.
///
/// This is the full seal operation:
/// 1. Build a FrozenVolume from the rows
/// 2. Write it to disk as a `.vol` file
/// 3. Return the volume for in-memory use
///
/// # Arguments
/// * `schema` - Table schema
/// * `rows` - Committed rows to freeze
/// * `volume_dir` - Directory for volume files
/// * `table_name` - Table name (for the subdirectory)
///
/// # Returns
/// (FrozenVolume, volume_file_path) on success
pub fn seal_and_persist(
    schema: &Schema,
    rows: &[(i64, Row)],
    volume_dir: &std::path::Path,
    table_name: &str,
) -> Result<(Arc<FrozenVolume>, std::path::PathBuf, u64)> {
    seal_and_persist_opts(schema, rows, volume_dir, table_name, true)
}

/// Seal rows into a frozen volume with optional LZ4 compression.
/// When V4 (compress=true), the returned volume keeps eager columns pre-loaded
/// in OnceLock slots with a compressed backing store for per-group scanner access.
pub fn seal_and_persist_opts(
    schema: &Schema,
    rows: &[(i64, Row)],
    volume_dir: &std::path::Path,
    table_name: &str,
    compress: bool,
) -> Result<(Arc<FrozenVolume>, std::path::PathBuf, u64)> {
    let volume = seal_rows(schema, rows);
    let volume_id = io::next_volume_id();
    let (path, _store) =
        io::write_volume_to_disk_opts(volume_dir, table_name, volume_id, &volume, compress)?;
    // V4 writes compressed blocks to disk but we drop the CompressedBlockStore.
    // Eager columns from seal_rows are already in RAM via OnceLock — no need to
    // keep ~30MB of compressed data alongside them.
    Ok((Arc::new(volume), path, volume_id))
}

/// Seal rows into multiple target-sized volumes.
/// Split is row-group aligned (multiples of 64K rows) so every volume
/// has complete row groups. Optimal for LZ4 compression and zone maps.
/// The last chunk may be smaller (partial row groups at the tail).
///
/// Returns a Vec of (volume, path, volume_id) tuples.
pub fn seal_and_persist_multi(
    schema: &Schema,
    rows: &[(i64, Row)],
    volume_dir: &std::path::Path,
    table_name: &str,
    compress: bool,
    target_rows: usize,
) -> Result<Vec<(Arc<FrozenVolume>, std::path::PathBuf, u64)>> {
    // Row-group aligned chunk size
    let row_group_size = 65_536usize;
    let chunk_size = (target_rows / row_group_size).max(1) * row_group_size;

    if rows.len() <= chunk_size || target_rows == 0 {
        let (vol, path, id) =
            seal_and_persist_opts(schema, rows, volume_dir, table_name, compress)?;
        return Ok(vec![(vol, path, id)]);
    }

    let mut results = Vec::new();
    for chunk in rows.chunks(chunk_size) {
        let volume = seal_rows(schema, chunk);
        let volume_id = io::next_volume_id();
        match io::write_volume_to_disk_opts(volume_dir, table_name, volume_id, &volume, compress) {
            Ok((path, _store)) => {
                results.push((Arc::new(volume), path, volume_id));
            }
            Err(e) => {
                // Clean up already-written files before propagating the error.
                for (_, path, _) in &results {
                    let _ = std::fs::remove_file(path);
                }
                return Err(e);
            }
        }
    }
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{DataType, SchemaBuilder, Value};

    #[test]
    fn test_seal_basic() {
        let schema = SchemaBuilder::new("test")
            .column("id", DataType::Integer, false, true)
            .column("name", DataType::Text, false, false)
            .build();

        let rows = vec![
            (
                1,
                Row::from_values(vec![Value::Integer(1), Value::text("alice")]),
            ),
            (
                2,
                Row::from_values(vec![Value::Integer(2), Value::text("bob")]),
            ),
            (
                3,
                Row::from_values(vec![Value::Integer(3), Value::text("carol")]),
            ),
        ];

        let volume = seal_rows(&schema, &rows);
        assert_eq!(volume.row_count, 3);
        assert_eq!(volume.columns[0].get_i64(0), 1);
        assert_eq!(volume.columns[1].get_str(2), "carol");
        assert!(volume.is_sorted(0)); // id is sorted
    }

    #[test]
    fn test_seal_and_persist() {
        let dir = tempfile::tempdir().unwrap();
        let schema = SchemaBuilder::new("test")
            .column("id", DataType::Integer, false, true)
            .column("val", DataType::Float, false, false)
            .build();

        let rows = vec![
            (
                1,
                Row::from_values(vec![Value::Integer(1), Value::Float(10.0)]),
            ),
            (
                2,
                Row::from_values(vec![Value::Integer(2), Value::Float(20.0)]),
            ),
        ];

        let vol_dir = dir.path().join("volumes");
        let (volume, path, _vol_id) =
            seal_and_persist(&schema, &rows, &vol_dir, "test_table").unwrap();

        assert_eq!(volume.row_count, 2);
        assert!(path.exists());

        // Read it back
        let loaded = io::read_volume_from_disk(&path).unwrap();
        assert_eq!(loaded.row_count, 2);
        assert_eq!(loaded.columns[0].get_i64(0), 1);
        assert_eq!(loaded.stats.sum(1), 30.0);
    }
}
