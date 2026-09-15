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

//! Writes a V4 volume file from typed batches as they arrive.
//!
//! The rows go into one row group's typed accumulators; when the group is
//! full, each column's block is encoded from the accumulators, compressed
//! when that shrinks it, and appended to a temporary block file, and the
//! accumulators are cleared with their capacity kept. Neither the output's
//! decoded columns nor its compressed blocks accumulate in memory. When
//! every row is in, the metadata is serialized from its parts and the
//! final file is assembled in V4's order (header, metadata, length index,
//! blocks column-major): each block is copied from the block file through
//! a bounded buffer, the whole-file CRC hashed as it goes. The compressed
//! data is therefore written and read once more than a single-pass layout
//! would need. The finished volume has its metadata resident and its
//! columns backed by the file, read a group at a time.

use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::core::{Error, Result, Schema};

use super::column::ROW_GROUP_SIZE;
use super::format::{serialize_typed_block, serialize_volume_metadata_parts};
use super::writer::{CompressedBlockStore, FrozenVolume, LazyColumns, TypedCells, VolumeBuilder};

/// Bytes copied at a time when the final file is assembled
const COPY_BUFFER: usize = 1 << 20;

/// The block file's name next to the final file
fn block_file_path(final_path: &Path) -> PathBuf {
    final_path.with_extension("blocks.tmp")
}

fn io_error(what: &str, error: std::io::Error) -> Error {
    Error::internal(format!("{what}: {error}"))
}

/// Writes one volume file from typed batches; see the module
pub struct VolumeFileWriter {
    /// None once `finish` has taken it
    builder: Option<VolumeBuilder>,
    compress: bool,
    final_path: PathBuf,
    blocks_path: PathBuf,
    blocks: Option<BufWriter<File>>,
    /// Bytes written to the block file so far, the next block's position
    block_pos: u64,
    /// By column, by group: (position in the block file, compressed
    /// length, decompressed length)
    index: Vec<Vec<(u64, usize, usize)>>,
    packed: Vec<u8>,
    finished: bool,
}

impl VolumeFileWriter {
    /// A writer for `dir/table_name/vol_<id>.vol`, sized for
    /// `expected_rows` (the bloom filters take it)
    pub fn new(
        dir: &Path,
        table_name: &str,
        volume_id: u64,
        schema: &Schema,
        expected_rows: usize,
        compress: bool,
    ) -> Result<Self> {
        let table_dir = dir.join(table_name);
        std::fs::create_dir_all(&table_dir)
            .map_err(|e| io_error("failed to create volume dir", e))?;
        let final_path =
            table_dir.join(format!("vol_{:016x}.{}", volume_id, super::io::VOLUME_EXT));
        let blocks_path = block_file_path(&final_path);
        let blocks = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&blocks_path)
            .map_err(|e| io_error("failed to create volume block file", e))?;
        let mut builder = VolumeBuilder::new(schema);
        builder.feed_bloom_filters(expected_rows);
        Ok(Self {
            index: vec![Vec::new(); schema.columns.len()],
            builder: Some(builder),
            compress,
            final_path,
            blocks_path,
            blocks: Some(BufWriter::with_capacity(COPY_BUFFER, blocks)),
            block_pos: 0,
            packed: Vec::new(),
            finished: false,
        })
    }

    /// Accept rows in the order the producer chose; see
    /// `VolumeBuilder::allow_any_row_order`
    pub fn allow_any_row_order(&mut self) {
        if let Some(builder) = &mut self.builder {
            builder.allow_any_row_order();
        }
    }

    /// The final file's path
    pub fn path(&self) -> &Path {
        &self.final_path
    }

    /// Rows written so far
    pub fn rows(&self) -> usize {
        self.builder.as_ref().map_or(0, VolumeBuilder::row_count)
    }

    fn builder(&mut self) -> Result<&mut VolumeBuilder> {
        self.builder
            .as_mut()
            .ok_or_else(|| Error::internal("volume writer already finished"))
    }

    /// Appends a batch, see `VolumeBuilder::append_typed`; a batch that
    /// crosses a group boundary is split there, so every group but the
    /// last holds exactly `ROW_GROUP_SIZE` rows
    pub fn append_typed(&mut self, row_ids: &[i64], columns: &[TypedCells<'_>]) -> Result<()> {
        let mut start = 0;
        while start < row_ids.len() {
            let builder = self.builder()?;
            let room = ROW_GROUP_SIZE - builder.group_len();
            let end = (start + room).min(row_ids.len());
            if start == 0 && end == row_ids.len() {
                builder.append_typed(row_ids, columns)?;
            } else {
                let part: Vec<TypedCells<'_>> = columns
                    .iter()
                    .map(|cells| cells.slice(start..end))
                    .collect();
                builder.append_typed(&row_ids[start..end], &part)?;
            }
            if builder.group_len() == ROW_GROUP_SIZE {
                self.flush_group()?;
            }
            start = end;
        }
        Ok(())
    }

    /// Encodes the accumulated group's blocks into the block file
    fn flush_group(&mut self) -> Result<()> {
        let blocks = self
            .blocks
            .as_mut()
            .ok_or_else(|| Error::internal("volume writer already finished"))?;
        let compress = self.compress;
        let (packed, index, block_pos) = (&mut self.packed, &mut self.index, &mut self.block_pos);
        let builder = self
            .builder
            .as_mut()
            .ok_or_else(|| Error::internal("volume writer already finished"))?;
        builder.flush_group(|col_idx, cells| {
            let encoded = serialize_typed_block(&cells);
            let decompressed_len = encoded.len();
            let bytes: &[u8] = if compress {
                packed.clear();
                packed.resize(lz4_flex::block::get_maximum_output_size(encoded.len()), 0);
                let n = lz4_flex::block::compress_into(&encoded, packed)
                    .map_err(|e| Error::internal(format!("LZ4: {e}")))?;
                if n < encoded.len() {
                    &packed[..n]
                } else {
                    &encoded
                }
            } else {
                &encoded
            };
            blocks
                .write_all(bytes)
                .map_err(|e| io_error("failed to write volume block", e))?;
            index[col_idx].push((*block_pos, bytes.len(), decompressed_len));
            *block_pos += bytes.len() as u64;
            Ok(())
        })
    }

    /// Writes the last group, the metadata and the final file, syncs it
    /// and renames it into place; returns the volume, its columns backed
    /// by the file, and the path
    pub fn finish(mut self) -> Result<(FrozenVolume, PathBuf)> {
        if self.builder()?.group_len() > 0 {
            self.flush_group()?;
        }
        let mut blocks = self
            .blocks
            .take()
            .ok_or_else(|| Error::internal("volume writer already finished"))?;
        blocks
            .flush()
            .map_err(|e| io_error("failed to flush volume block file", e))?;
        let mut block_file = blocks
            .into_inner()
            .map_err(|e| io_error("failed to flush volume block file", e.into_error()))?;

        let builder = self
            .builder
            .take()
            .ok_or_else(|| Error::internal("volume writer already finished"))?;
        let kinds: Vec<(u8, u8)> = (0..self.index.len())
            .map(|col| builder.column_kind(col))
            .collect();
        let (meta, dict_tables) = builder.finish_streamed()?;
        let dictionaries: Vec<&[crate::common::SmartString]> =
            dict_tables.iter().map(Vec::as_slice).collect();
        let meta_raw = serialize_volume_metadata_parts(&meta, &kinds, &dictionaries)
            .map_err(|e| io_error("V4 metadata", e))?;
        let meta_compressed = lz4_flex::compress_prepend_size(&meta_raw);
        drop(meta_raw);

        let col_count = self.index.len();
        let num_groups = self.index.first().map(Vec::len).unwrap_or(0);
        if self.index.iter().any(|col| col.len() != num_groups) {
            return Err(Error::internal(
                "volume columns have different group counts",
            ));
        }

        // The final file in V4's order, hashed as it is written
        let tmp_path = self.final_path.with_extension("vol.tmp");
        let mut hasher = crc32fast::Hasher::new();
        {
            let file = File::create(&tmp_path)
                .map_err(|e| io_error("failed to create volume tmp file", e))?;
            let mut out = BufWriter::with_capacity(COPY_BUFFER, file);
            let mut put = |out: &mut BufWriter<File>, bytes: &[u8]| -> Result<()> {
                hasher.update(bytes);
                out.write_all(bytes)
                    .map_err(|e| io_error("failed to write volume file", e))
            };
            put(&mut out, &super::io::V4_MAGIC)?;
            put(&mut out, &super::io::V4_VERSION.to_le_bytes())?;
            put(&mut out, &(col_count as u32).to_le_bytes())?;
            put(&mut out, &(num_groups as u32).to_le_bytes())?;
            put(&mut out, &(meta_compressed.len() as u32).to_le_bytes())?;
            put(&mut out, &meta_compressed)?;
            drop(meta_compressed);
            for col in &self.index {
                for &(_, compressed_len, decompressed_len) in col {
                    put(&mut out, &(compressed_len as u64).to_le_bytes())?;
                    put(&mut out, &(decompressed_len as u64).to_le_bytes())?;
                }
            }
            let mut copy = vec![0u8; COPY_BUFFER];
            for col in &self.index {
                for &(position, compressed_len, _) in col {
                    block_file
                        .seek(SeekFrom::Start(position))
                        .map_err(|e| io_error("failed to seek volume block file", e))?;
                    let mut left = compressed_len;
                    while left > 0 {
                        let take = copy.len().min(left);
                        block_file
                            .read_exact(&mut copy[..take])
                            .map_err(|e| io_error("failed to read volume block file", e))?;
                        put(&mut out, &copy[..take])?;
                        left -= take;
                    }
                }
            }
            let crc = hasher.finalize();
            out.write_all(&crc.to_le_bytes())
                .map_err(|e| io_error("failed to write volume file", e))?;
            out.flush()
                .map_err(|e| io_error("failed to flush volume file", e))?;
            out.get_ref()
                .sync_all()
                .map_err(|e| io_error("failed to fsync volume tmp file", e))?;
        }
        drop(block_file);
        let _ = std::fs::remove_file(&self.blocks_path);
        std::fs::rename(&tmp_path, &self.final_path)
            .map_err(|e| io_error("failed to rename volume file", e))?;
        #[cfg(not(windows))]
        if let Some(dir) = self.final_path.parent() {
            if let Ok(d) = File::open(dir) {
                d.sync_all()
                    .map_err(|e| io_error("failed to fsync volume directory", e))?;
            }
        }
        self.finished = true;

        // The volume over the file just written: the block offsets follow
        // from the lengths as the reader computes them
        let file = Arc::new(
            File::open(&self.final_path).map_err(|e| io_error("failed to open volume file", e))?,
        );
        let blocks_start = 20 + lz4_meta_len(&self.final_path)? + col_count * num_groups * 16;
        let mut offsets = Vec::with_capacity(col_count);
        let mut comp_lens = Vec::with_capacity(col_count);
        let mut decomp_lens = Vec::with_capacity(col_count);
        let mut position = blocks_start as u64;
        for col in &self.index {
            let mut col_offsets = Vec::with_capacity(num_groups);
            let mut col_comp = Vec::with_capacity(num_groups);
            let mut col_decomp = Vec::with_capacity(num_groups);
            for &(_, compressed_len, decompressed_len) in col {
                col_offsets.push(position);
                col_comp.push(compressed_len);
                col_decomp.push(decompressed_len);
                position += compressed_len as u64;
            }
            offsets.push(col_offsets);
            comp_lens.push(col_comp);
            decomp_lens.push(col_decomp);
        }
        let mut shared_dict = Vec::new();
        let mut dict_ranges = Vec::new();
        let mut dict_col = 0;
        for (col_idx, &(type_tag, _)) in kinds.iter().enumerate() {
            if type_tag == super::format::COL_DICTIONARY {
                let start = shared_dict.len();
                shared_dict.extend(dict_tables[dict_col].iter().cloned());
                dict_ranges.push((col_idx, start, shared_dict.len()));
                dict_col += 1;
            }
        }
        let column_types = meta.column_types.clone();
        let store = CompressedBlockStore::from_file(
            file,
            offsets,
            comp_lens,
            decomp_lens,
            kinds.iter().map(|k| k.0).collect(),
            column_types.clone(),
            kinds.iter().map(|k| k.1).collect(),
            shared_dict,
            dict_ranges,
            ROW_GROUP_SIZE,
            meta.row_count,
        );
        let volume = FrozenVolume {
            columns: LazyColumns::deferred(store, column_types),
            meta: Arc::new(meta),
            unique_indices: Arc::new(parking_lot::RwLock::new(rustc_hash::FxHashMap::default())),
            last_access_epoch: std::sync::atomic::AtomicU64::new(
                super::writer::GLOBAL_EVICTION_EPOCH.load(std::sync::atomic::Ordering::Relaxed),
            ),
        };
        Ok((volume, self.final_path.clone()))
    }

    /// Drops the writer and removes what it wrote
    pub fn abort(self) {
        drop(self);
    }
}

impl Drop for VolumeFileWriter {
    fn drop(&mut self) {
        if !self.finished {
            self.blocks = None;
            let _ = std::fs::remove_file(&self.blocks_path);
            let _ = std::fs::remove_file(self.final_path.with_extension("vol.tmp"));
        }
    }
}

/// The compressed metadata length in a written V4 file's header
fn lz4_meta_len(path: &Path) -> Result<usize> {
    let mut file = File::open(path).map_err(|e| io_error("failed to open volume file", e))?;
    let mut header = [0u8; 20];
    file.read_exact(&mut header)
        .map_err(|e| io_error("failed to read volume header", e))?;
    Ok(u32::from_le_bytes([header[16], header[17], header[18], header[19]]) as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{DataType, Row, SchemaBuilder, Value};
    use crate::storage::volume::column::ColumnData;
    use crate::storage::volume::io::{read_volume_from_disk, serialize_v4_public};

    fn schema() -> Schema {
        SchemaBuilder::new("t")
            .column("id", DataType::Integer, false, true)
            .column("n", DataType::Integer, true, false)
            .column("x", DataType::Float, true, false)
            .column("at", DataType::Timestamp, true, false)
            .column("ok", DataType::Boolean, true, false)
            .column("name", DataType::Text, true, false)
            .column("doc", DataType::Json, true, false)
            .build()
    }

    fn row(i: i64) -> Row {
        let null_here = i % 7 == 3;
        Row::from_values(vec![
            Value::Integer(i),
            if null_here {
                Value::Null(DataType::Integer)
            } else {
                Value::Integer(1000 - i)
            },
            if i % 11 == 5 {
                Value::Float(f64::NAN)
            } else if null_here {
                Value::Null(DataType::Float)
            } else {
                Value::Float(i as f64 * 0.5)
            },
            if i % 5 == 4 {
                Value::Null(DataType::Timestamp)
            } else {
                Value::Timestamp(
                    chrono::DateTime::from_timestamp(1_700_000_000 + i * 60, 0).unwrap(),
                )
            },
            if null_here {
                Value::Null(DataType::Boolean)
            } else {
                Value::Boolean(i % 3 == 0)
            },
            if i % 13 == 6 {
                Value::Null(DataType::Text)
            } else {
                Value::text(format!("name-{}", i % 17))
            },
            if null_here {
                Value::Null(DataType::Json)
            } else {
                Value::json(format!("{{\"k\":{i}}}"))
            },
        ])
    }

    /// The typed cells of a decoded column
    fn cells(col: &ColumnData) -> TypedCells<'_> {
        match col {
            ColumnData::Int64 { values, nulls } => TypedCells::Int64 { values, nulls },
            ColumnData::Float64 { values, nulls } => TypedCells::Float64 { values, nulls },
            ColumnData::TimestampNanos { values, nulls } => {
                TypedCells::TimestampNanos { values, nulls }
            }
            ColumnData::Boolean { values, nulls } => TypedCells::Boolean { values, nulls },
            ColumnData::Dictionary { ids, nulls, .. } => TypedCells::Dictionary { ids, nulls },
            ColumnData::Bytes {
                data,
                offsets,
                nulls,
                ..
            } => TypedCells::Bytes {
                data,
                offsets,
                nulls,
            },
        }
    }

    /// The same rows through `add_row` and through the writer in batches
    /// of `batch`, text interned in the writer from the source strings
    fn written(
        schema: &Schema,
        rows: i64,
        batch: usize,
    ) -> (Vec<u8>, FrozenVolume, PathBuf, tempfile::TempDir) {
        let mut builder = VolumeBuilder::new(schema);
        for i in 0..rows {
            builder.add_row(i, &row(i));
        }
        let by_rows = builder.finish().unwrap();
        let (bytes, _) = serialize_v4_public(&by_rows).unwrap();

        let dir = tempfile::tempdir().unwrap();
        let mut writer =
            VolumeFileWriter::new(dir.path(), "t", 7, schema, rows as usize, true).unwrap();
        let columns: Vec<&ColumnData> = (0..schema.columns.len())
            .map(|c| by_rows.columns.get(c).unwrap())
            .collect();
        let ids = by_rows.row_ids().unwrap();
        let mut start = 0;
        while start < rows as usize {
            let end = (start + batch).min(rows as usize);
            let mut typed = Vec::new();
            let mut interned: Vec<Vec<u32>> = Vec::new();
            for (c, col) in columns.iter().enumerate() {
                let part = cells(col).slice(start..end);
                // The writer's dictionary is its own: intern the strings
                if let ColumnData::Dictionary {
                    ids,
                    dictionary,
                    nulls,
                } = col
                {
                    let mut out = Vec::with_capacity(end - start);
                    for i in start..end {
                        out.push(if nulls[i] {
                            0
                        } else {
                            writer
                                .builder()
                                .unwrap()
                                .intern_text(c, dictionary[ids[i] as usize].as_str())
                                .unwrap()
                        });
                    }
                    interned.push(out);
                    typed.push(None);
                } else {
                    typed.push(Some(part));
                }
            }
            let mut interned_iter = interned.iter();
            let batch_cells: Vec<TypedCells<'_>> = typed
                .into_iter()
                .enumerate()
                .map(|(c, part)| match part {
                    Some(part) => part,
                    None => {
                        let ids = interned_iter.next().unwrap();
                        let nulls = match columns[c] {
                            ColumnData::Dictionary { nulls, .. } => &nulls[start..end],
                            _ => unreachable!(),
                        };
                        TypedCells::Dictionary { ids, nulls }
                    }
                })
                .collect();
            writer.append_typed(&ids[start..end], &batch_cells).unwrap();
            start = end;
        }
        let (volume, path) = writer.finish().unwrap();
        (bytes, volume, path, dir)
    }

    #[test]
    fn the_file_is_the_one_serialize_v4_writes_for_the_same_rows() {
        let schema = schema();
        for (rows, batch) in [(500, 64), (500, 500), (3, 1)] {
            let (want, volume, path, _dir) = written(&schema, rows, batch);
            let got = std::fs::read(&path).unwrap();
            assert_eq!(got.len(), want.len(), "{rows} rows in batches of {batch}");
            assert!(
                got == want,
                "{rows} rows in batches of {batch}: file differs"
            );
            assert!(volume.is_warm());
            assert!(volume.columns.compressed_store().unwrap().is_file_backed());
            assert!(!path.with_extension("blocks.tmp").exists());
            for c in 0..schema.columns.len() {
                let col = volume.columns.get(c).unwrap();
                for i in 0..rows as usize {
                    let v = col.get_value(i);
                    let w = row(i as i64).get(c).unwrap().clone();
                    let same = match (&v, &w) {
                        (Value::Float(a), Value::Float(b)) => a.to_bits() == b.to_bits(),
                        _ => v == w,
                    };
                    assert!(same, "column {c} row {i}: {v:?} vs {w:?}");
                }
            }
        }
    }

    #[test]
    fn groups_close_at_their_size_and_a_batch_that_crosses_one_is_split() {
        let schema = SchemaBuilder::new("t")
            .column("id", DataType::Integer, false, true)
            .column("name", DataType::Text, true, false)
            .build();
        let rows = 2 * ROW_GROUP_SIZE as i64 + 4_000;
        let mut builder = VolumeBuilder::new(&schema);
        for i in 0..rows {
            builder.add_row(
                i,
                &Row::from_values(vec![Value::Integer(i), Value::text(format!("n{}", i % 5))]),
            );
        }
        let by_rows = builder.finish().unwrap();
        let (want, _) = serialize_v4_public(&by_rows).unwrap();

        let dir = tempfile::tempdir().unwrap();
        let mut writer =
            VolumeFileWriter::new(dir.path(), "t", 9, &schema, rows as usize, true).unwrap();
        let ids = by_rows.row_ids().unwrap();
        let (id_col, name_col) = (
            by_rows.columns.get(0).unwrap(),
            by_rows.columns.get(1).unwrap(),
        );
        let (name_ids, dictionary, name_nulls) = match name_col {
            ColumnData::Dictionary {
                ids,
                dictionary,
                nulls,
            } => (ids, dictionary, nulls),
            _ => unreachable!(),
        };
        // 4,096 a batch: the boundaries fall inside batches
        let mut start = 0;
        while start < rows as usize {
            let end = (start + 4_096).min(rows as usize);
            let interned: Vec<u32> = (start..end)
                .map(|i| {
                    writer
                        .builder()
                        .unwrap()
                        .intern_text(1, dictionary[name_ids[i] as usize].as_str())
                        .unwrap()
                })
                .collect();
            writer
                .append_typed(
                    &ids[start..end],
                    &[
                        cells(id_col).slice(start..end),
                        TypedCells::Dictionary {
                            ids: &interned,
                            nulls: &name_nulls[start..end],
                        },
                    ],
                )
                .unwrap();
            start = end;
        }
        let (volume, path) = writer.finish().unwrap();
        assert!(std::fs::read(&path).unwrap() == want);
        assert_eq!(volume.meta.row_groups.len(), 3);
        assert_eq!(
            volume
                .meta
                .row_groups
                .iter()
                .map(|g| (g.start_idx, g.end_idx))
                .collect::<Vec<_>>(),
            vec![
                (0, ROW_GROUP_SIZE as u32),
                (ROW_GROUP_SIZE as u32, 2 * ROW_GROUP_SIZE as u32),
                (2 * ROW_GROUP_SIZE as u32, rows as u32)
            ]
        );
        // Read back from the file: one group at a time through the store
        let reread = read_volume_from_disk(&path).unwrap();
        assert!(reread.columns.compressed_store().unwrap().is_file_backed());
        let store = reread.columns.compressed_store().unwrap();
        let group = store.group_column(1, 2).unwrap();
        assert_eq!(group.len(), 4_000);
        assert_eq!(group.get_str(0), format!("n{}", (2 * ROW_GROUP_SIZE) % 5));
        assert_eq!(
            reread.columns.get(0).unwrap().get_i64(rows as usize - 1),
            rows - 1
        );
    }

    #[test]
    fn a_corrupted_file_is_refused_at_open_and_an_aborted_writer_leaves_nothing() {
        let schema = schema();
        let (_, volume, path, dir) = written(&schema, 200, 50);
        drop(volume);
        let mut bytes = std::fs::read(&path).unwrap();
        let middle = bytes.len() / 2;
        bytes[middle] ^= 0xff;
        std::fs::write(&path, &bytes).unwrap();
        assert!(read_volume_from_disk(&path).is_err());

        let mut writer = VolumeFileWriter::new(dir.path(), "t", 11, &schema, 10, true).unwrap();
        let ids = [1i64, 2];
        let nulls = [false, false];
        let n = [5i64, 6];
        let x = [0.5, 1.5];
        let at = [1i64, 2];
        let ok = [true, false];
        let names: Vec<u32> = ["a", "b"]
            .iter()
            .map(|s| writer.builder().unwrap().intern_text(5, s).unwrap())
            .collect();
        let doc = TypedCells::Bytes {
            data: b"{}{}",
            offsets: &[(0, 2), (2, 2)],
            nulls: &nulls,
        };
        writer
            .append_typed(
                &ids,
                &[
                    TypedCells::Int64 {
                        values: &ids,
                        nulls: &nulls,
                    },
                    TypedCells::Int64 {
                        values: &n,
                        nulls: &nulls,
                    },
                    TypedCells::Float64 {
                        values: &x,
                        nulls: &nulls,
                    },
                    TypedCells::TimestampNanos {
                        values: &at,
                        nulls: &nulls,
                    },
                    TypedCells::Boolean {
                        values: &ok,
                        nulls: &nulls,
                    },
                    TypedCells::Dictionary {
                        ids: &names,
                        nulls: &nulls,
                    },
                    doc,
                ],
            )
            .unwrap();
        let blocks = writer.blocks_path.clone();
        let final_path = writer.path().to_path_buf();
        assert!(blocks.exists());
        writer.abort();
        assert!(!blocks.exists());
        assert!(!final_path.exists());
        assert!(!final_path.with_extension("vol.tmp").exists());
    }
}
