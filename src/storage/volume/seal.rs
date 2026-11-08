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

use std::cmp::Ordering;
use std::sync::Arc;

use crate::core::{Result, Row, Schema, Value};

use super::column::value_kind;
use super::io;
use super::output::VolumeFileWriter;
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
pub fn seal_rows(schema: &Schema, rows: &[(i64, Row)]) -> Result<FrozenVolume> {
    let mut builder = VolumeBuilder::with_capacity(schema, rows.len());
    // A clustered table's rows arrive in key order, which the caller decided
    if !schema.cluster_key.is_empty() {
        builder.allow_any_row_order();
    }
    for (row_id, row) in rows {
        builder.add_row(*row_id, row);
    }
    builder.finish().map_err(|e| {
        crate::core::Error::internal(format!("sealing table {}: {e}", schema.table_name))
    })
}

/// Orders two values of a clustering key column: by kind, then by value
fn compare_key_value(a: &Value, b: &Value) -> Ordering {
    let kinds = value_kind(a).cmp(&value_kind(b));
    if kinds != Ordering::Equal || a.is_null() {
        return kinds;
    }
    a.compare(b).unwrap_or(Ordering::Equal)
}

/// Orders two values of one clustering key column
pub fn compare_key_values(a: &Value, b: &Value) -> Ordering {
    compare_key_value(a, b)
}

/// Orders two rows of a clustered table by its key, then by row id
pub fn cluster_order(schema: &Schema, a: &(i64, Row), b: &(i64, Row)) -> Ordering {
    let null = Value::null_unknown();
    schema
        .cluster_key
        .iter()
        .map(|&column| {
            compare_key_value(
                a.1.get(column).unwrap_or(&null),
                b.1.get(column).unwrap_or(&null),
            )
        })
        .find(|o| *o != Ordering::Equal)
        .unwrap_or_else(|| a.0.cmp(&b.0))
}

/// Writes one volume of `rows` through the streaming writer, its unique
/// column sets indexed as the rows pass; the volume's columns are backed
/// by the file
fn seal_chunk(
    schema: &Schema,
    rows: &[(i64, Row)],
    volume_dir: &std::path::Path,
    table_name: &str,
    compress: bool,
    unique_sets: &[Vec<usize>],
) -> Result<(Arc<FrozenVolume>, std::path::PathBuf, u64)> {
    let volume_id = io::next_volume_id();
    let mut writer = VolumeFileWriter::new(
        volume_dir,
        table_name,
        volume_id,
        schema,
        rows.len(),
        compress,
    )?;
    // A clustered table's rows arrive in key order, which the caller decided
    if !schema.cluster_key.is_empty() {
        writer.allow_any_row_order();
    }
    writer.index_unique_sets(unique_sets.to_vec());
    let fed = rows
        .iter()
        .try_for_each(|(row_id, row)| writer.add_row(*row_id, row));
    let (volume, path) = fed.and_then(|()| writer.finish()).map_err(|e| {
        crate::core::Error::internal(format!("sealing table {}: {e}", schema.table_name))
    })?;
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
    unique_sets: &[Vec<usize>],
) -> Result<Vec<(Arc<FrozenVolume>, std::path::PathBuf, u64)>> {
    // Row-group aligned chunk size
    let row_group_size = 65_536usize;
    let chunk_size = if target_rows == 0 {
        rows.len().max(1)
    } else {
        (target_rows / row_group_size).max(1) * row_group_size
    };

    let mut results = Vec::new();
    for chunk in rows.chunks(chunk_size) {
        let sealed = seal_chunk(schema, chunk, volume_dir, table_name, compress, unique_sets);
        match sealed {
            Ok(entry) => results.push(entry),
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
    fn a_row_with_a_timestamp_past_the_range_fails_the_seal() {
        let schema = SchemaBuilder::new("events")
            .column("id", DataType::Integer, false, true)
            .column("ts", DataType::Timestamp, true, false)
            .build();
        let at = |s: &str| Value::Timestamp(crate::core::parse_timestamp(s).unwrap());
        let row = |id: i64, ts: Value| (id, Row::from_values(vec![Value::Integer(id), ts]));
        let rows = vec![
            row(1, at("2024-01-01")),
            row(2, at("9999-12-31")),
            row(3, at("2024-01-02")),
        ];
        let err = match seal_rows(&schema, &rows) {
            Ok(_) => panic!("a TIMESTAMP past the range was sealed"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("sealing table events") && err.contains("row 2"),
            "{err}"
        );
        assert!(err.contains("outside the range"), "{err}");

        let mut builder = VolumeBuilder::new(&schema);
        builder.add_row(1, &rows[1].1);
        assert!(builder.flush_group(|_, _| Ok(())).is_err());
        assert_eq!(builder.group_len(), 0, "the refused row took no column");
    }

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

        let volume = seal_rows(&schema, &rows).unwrap();
        assert_eq!(volume.meta.row_count, 3);
        assert_eq!(volume.columns.get(0).unwrap().get_i64(0), 1);
        assert_eq!(volume.columns.get(1).unwrap().get_str(2), "carol");
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
        let mut sealed =
            seal_and_persist_multi(&schema, &rows, &vol_dir, "test_table", true, 0, &[]).unwrap();
        let (volume, path, _vol_id) = sealed.pop().unwrap();

        assert_eq!(volume.meta.row_count, 2);
        assert!(path.exists());

        // Read it back
        let loaded = io::read_volume_from_disk(&path).unwrap();
        assert_eq!(loaded.meta.row_count, 2);
        assert_eq!(loaded.columns.get(0).unwrap().get_i64(0), 1);
        assert_eq!(loaded.meta.stats.sum(1), 30.0);
    }
}
