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

//! Writes a V5 volume file from typed batches as they arrive.
//!
//! The rows go into one row group's typed accumulators; when the group is
//! full, each column's block is encoded from the accumulators, compressed
//! when that shrinks it, and appended to the file being written, and the
//! accumulators are cleared with their capacity kept. Neither the output's
//! decoded columns nor its compressed blocks accumulate in memory. When
//! every row is in, the metadata sections, the section table and the
//! footer follow the blocks, and the file is synced and renamed into
//! place. The finished volume has its metadata resident and its columns
//! backed by the file, read a group at a time.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::core::{Error, Result, Row, Schema};

use super::column::ROW_GROUP_SIZE;
use super::format::serialize_typed_block;
use super::transfer::TypedSink;
use super::writer::{
    Accumulators, CompressedBlockStore, FrozenVolume, LazyColumns, TypedCells, VolumeBuilder,
};

/// The write buffer of the file being written
const COPY_BUFFER: usize = 1 << 20;

/// The decoded size of a volume's widest column block past which its
/// groups stop doubling
const TARGET_BLOCK_BYTES: usize = 8 << 20;

/// The fewest rows a group of a volume holds, but its last
const MIN_GROUP_ROWS: usize = 1024;

/// The name the file has until it is complete, next to the final file
fn block_file_path(final_path: &Path) -> PathBuf {
    final_path.with_extension("blocks.tmp")
}

fn io_error(what: &str, error: std::io::Error) -> Error {
    Error::internal(format!("{what}: {error}"))
}

/// Encodes one column's cells into `block`, LZ4-compressed when `compress`
/// and that shrinks it; returns the encoded length
fn encode_block(cells: &TypedCells<'_>, compress: bool, block: &mut Vec<u8>) -> Result<usize> {
    let encoded = serialize_typed_block(cells);
    let decoded = encoded.len();
    if compress {
        block.clear();
        block.resize(lz4_flex::block::get_maximum_output_size(decoded), 0);
        let n = lz4_flex::block::compress_into(&encoded, block)
            .map_err(|e| Error::internal(format!("LZ4: {e}")))?;
        if n < decoded {
            block.truncate(n);
            return Ok(decoded);
        }
    }
    *block = encoded;
    Ok(decoded)
}

/// One row group's cells and, once encoded, its blocks by column with
/// their decoded lengths
struct EncodeJob {
    group: Accumulators,
    blocks: Vec<(Vec<u8>, usize)>,
    result: Result<()>,
}

impl EncodeJob {
    fn run(&mut self, compress: bool) {
        let group = &self.group;
        self.result =
            self.blocks
                .iter_mut()
                .enumerate()
                .try_for_each(|(col_idx, (block, decoded))| {
                    *decoded = encode_block(&group.cells(col_idx), compress, block)?;
                    Ok(())
                });
    }
}

/// Encodes the groups handed to it in order, one at a time, on a thread
/// of its own when the build has threads: a group is encoded while the
/// next one accumulates
struct Encoder {
    compress: bool,
    /// Whether a job was handed over and not taken back
    out: bool,
    #[cfg(feature = "parallel")]
    channel: Option<(
        std::sync::mpsc::SyncSender<EncodeJob>,
        std::sync::mpsc::Receiver<EncodeJob>,
    )>,
    #[cfg(not(feature = "parallel"))]
    done: Option<EncodeJob>,
}

impl Encoder {
    fn new(compress: bool) -> Self {
        Self {
            compress,
            out: false,
            #[cfg(feature = "parallel")]
            channel: None,
            #[cfg(not(feature = "parallel"))]
            done: None,
        }
    }

    /// Hands `job` over; the previous one must have been taken back
    fn submit(&mut self, job: EncodeJob) -> Result<()> {
        debug_assert!(!self.out, "an encoded group was not taken back");
        #[cfg(feature = "parallel")]
        {
            if self.channel.is_none() {
                let (jobs, inbox) = std::sync::mpsc::sync_channel::<EncodeJob>(1);
                let (outbox, done) = std::sync::mpsc::sync_channel::<EncodeJob>(1);
                let compress = self.compress;
                std::thread::Builder::new()
                    .name("stoolap-volume-encode".into())
                    .spawn(move || {
                        for mut job in inbox {
                            job.run(compress);
                            if outbox.send(job).is_err() {
                                break;
                            }
                        }
                    })
                    .map_err(|e| io_error("failed to start the volume encoder", e))?;
                self.channel = Some((jobs, done));
            }
            let Some((jobs, _)) = &self.channel else {
                return Err(Error::internal("volume encoder not started"));
            };
            jobs.send(job)
                .map_err(|_| Error::internal("volume encoder stopped"))?;
        }
        #[cfg(not(feature = "parallel"))]
        {
            let mut job = job;
            job.run(self.compress);
            self.done = Some(job);
        }
        self.out = true;
        Ok(())
    }

    /// The job handed over last, once encoded; None when none is out
    fn take(&mut self) -> Result<Option<EncodeJob>> {
        if !self.out {
            return Ok(None);
        }
        self.out = false;
        #[cfg(feature = "parallel")]
        {
            let Some((_, done)) = &self.channel else {
                return Err(Error::internal("volume encoder not started"));
            };
            done.recv()
                .map(Some)
                .map_err(|_| Error::internal("volume encoder stopped"))
        }
        #[cfg(not(feature = "parallel"))]
        Ok(self.done.take())
    }
}

/// Writes one volume file from typed batches; see the module
pub struct VolumeFileWriter {
    /// None once `finish` has taken it
    builder: Option<VolumeBuilder>,
    /// The volume's rows per group, decided as its first group fills
    group_rows: Option<usize>,
    /// The row count at which the group is next measured or flushed:
    /// powers of two from `MIN_GROUP_ROWS` until the size is decided
    cut: usize,
    encoder: Encoder,
    /// The accumulators and block buffers of the last group written,
    /// taken by the next group handed to the encoder
    spare: Option<Accumulators>,
    spare_blocks: Option<Vec<(Vec<u8>, usize)>>,
    final_path: PathBuf,
    blocks_path: PathBuf,
    blocks: Option<BufWriter<File>>,
    /// Bytes written to the file so far, the next block's position
    block_pos: u64,
    /// By column, by group: (position in the file, compressed length,
    /// decompressed length, CRC-32C of the stored bytes)
    index: Vec<Vec<(u64, usize, usize, u32)>>,
    /// Every block written, in file order, for the block table
    entries: Vec<super::v5::BlockEntry>,
    finished: bool,
    /// Rows the writer was sized for; caps a unique set's capacity
    expected_rows: usize,
    volume_id: u64,
    /// The unique column sets indexed as the rows pass
    unique: Vec<UniqueIndexing>,
}

/// One unique column set and its (hash, row) entries, hashed as the
/// lookup hashes its values
struct UniqueIndexing {
    columns: Vec<usize>,
    entries: Vec<(u64, u32)>,
}

impl TypedSink for VolumeFileWriter {
    fn intern_text(&mut self, col_idx: usize, text: &str) -> Result<u32> {
        self.builder()?.intern_text(col_idx, text)
    }

    fn append_typed(&mut self, row_ids: &[i64], columns: &[TypedCells<'_>]) -> Result<()> {
        VolumeFileWriter::append_typed(self, row_ids, columns)
    }
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
        Self::at(final_path, volume_id, schema, expected_rows, compress)
    }

    /// A writer for the file at `final_path`, whose directory exists
    pub fn at(
        final_path: PathBuf,
        volume_id: u64,
        schema: &Schema,
        expected_rows: usize,
        compress: bool,
    ) -> Result<Self> {
        let blocks_path = block_file_path(&final_path);
        let blocks = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&blocks_path)
            .map_err(|e| io_error("failed to create volume block file", e))?;
        let mut builder = VolumeBuilder::new(schema);
        builder.feed_bloom_filters(expected_rows);
        builder.reserve_rows(expected_rows);
        let mut blocks = BufWriter::with_capacity(COPY_BUFFER, blocks);
        blocks
            .write_all(&super::v5::header())
            .map_err(|e| io_error("failed to write volume header", e))?;
        Ok(Self {
            index: vec![Vec::new(); schema.columns.len()],
            entries: Vec::new(),
            builder: Some(builder),
            group_rows: None,
            cut: MIN_GROUP_ROWS,
            encoder: Encoder::new(compress),
            spare: None,
            spare_blocks: None,
            final_path,
            blocks_path,
            blocks: Some(blocks),
            block_pos: super::v5::HEADER_LEN,
            finished: false,
            expected_rows,
            volume_id,
            unique: Vec::new(),
        })
    }

    /// Index these column sets for unique lookups as the rows pass, so
    /// the volume is published with its indexes built and the first
    /// INSERT after it reads no column whole
    pub fn index_unique_sets(&mut self, sets: Vec<Vec<usize>>) {
        self.unique = sets
            .into_iter()
            .map(|columns| UniqueIndexing {
                columns,
                entries: Vec::new(),
            })
            .collect();
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
    /// crosses a group boundary, or a row count where the group size is
    /// decided, is split there, so every group but the last holds exactly
    /// the volume's group size
    pub fn append_typed(&mut self, row_ids: &[i64], columns: &[TypedCells<'_>]) -> Result<()> {
        let mut start = 0;
        while start < row_ids.len() {
            let cut = self.cut;
            let builder = self.builder()?;
            let room = cut - builder.group_len();
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
            self.reached_cut()?;
            start = end;
        }
        Ok(())
    }

    /// Adds one row, see `VolumeBuilder::add_row`; a full group is
    /// written out
    pub fn add_row(&mut self, row_id: i64, row: &Row) -> Result<()> {
        let cut = self.cut;
        let builder = self.builder()?;
        builder.add_row(row_id, row);
        if builder.group_len() == cut {
            self.reached_cut()?;
        }
        Ok(())
    }

    /// At `cut` rows: flushes the group when the group size is, or now
    /// becomes, `cut`, at the ceiling or when the widest column's block
    /// would pass `TARGET_BLOCK_BYTES` on the next doubling; else doubles
    /// `cut` for the next measure
    fn reached_cut(&mut self) -> Result<()> {
        let builder = self
            .builder
            .as_ref()
            .ok_or_else(|| Error::internal("volume writer already finished"))?;
        if builder.group_len() < self.cut {
            return Ok(());
        }
        if self.group_rows.is_none() {
            let widest = builder.widest_group_block();
            if self.cut < ROW_GROUP_SIZE && widest.saturating_mul(2) <= TARGET_BLOCK_BYTES {
                self.cut *= 2;
                return Ok(());
            }
            self.group_rows = Some(self.cut);
        }
        self.flush_group()
    }

    /// Hashes every row's cells of the accumulated group for each unique
    /// column set the way `unique_lookup_all` hashes its values; a row
    /// with a null in the set is left out, as the index built over
    /// columns leaves it out
    fn hash_unique(&mut self) -> Result<()> {
        use std::hash::{Hash, Hasher};
        let builder = self
            .builder
            .as_ref()
            .ok_or_else(|| Error::internal("volume writer already finished"))?;
        let first = (builder.row_count() - builder.group_len()) as u32;
        let columns: Vec<TypedCells<'_>> = (0..self.index.len())
            .map(|col| builder.group_cells(col))
            .collect();
        let columns = columns.as_slice();
        let rows = builder.group_len();
        for UniqueIndexing {
            columns: set,
            entries,
        } in &mut self.unique
        {
            if entries.capacity() - entries.len() < rows {
                let nulls_of = |col: usize| {
                    columns
                        .get(col)
                        .map(TypedCells::nulls)
                        .ok_or_else(|| Error::internal("unique column beyond the batch"))
                };
                let indexed = match set.as_slice() {
                    &[one] => nulls_of(one)?.iter().filter(|&&null| !null).count(),
                    &[a, b] => {
                        let (a, b) = (nulls_of(a)?, nulls_of(b)?);
                        a.iter().zip(b).filter(|&(&a, &b)| !(a | b)).count()
                    }
                    wide => {
                        // Nulls merged a stack chunk at a time, column by column
                        let mut any = [false; 256];
                        let mut indexed = 0;
                        for start in (0..rows).step_by(any.len()) {
                            let end = (start + any.len()).min(rows);
                            let any = &mut any[..end - start];
                            any.fill(false);
                            for &col in wide {
                                for (any, &null) in any.iter_mut().zip(&nulls_of(col)?[start..end])
                                {
                                    *any |= null;
                                }
                            }
                            indexed += any.iter().filter(|&&any| !any).count();
                        }
                        indexed
                    }
                };
                let needed = entries.len() + indexed;
                if needed > entries.capacity() {
                    // Doubling from 4 as push would, never past the rows expected
                    let target = needed.next_power_of_two().max(4);
                    let target = if needed <= self.expected_rows {
                        target.min(self.expected_rows)
                    } else {
                        target
                    };
                    entries.reserve_exact(target - entries.len());
                }
            }
            'rows: for row in 0..rows {
                let mut hasher = ahash::AHasher::default();
                for &col in set.iter() {
                    let cells = columns
                        .get(col)
                        .ok_or_else(|| Error::internal("unique column beyond the batch"))?;
                    let value = match cells {
                        TypedCells::Int64 { values, nulls } => {
                            if nulls[row] {
                                continue 'rows;
                            }
                            crate::core::Value::Integer(values[row])
                        }
                        TypedCells::Float64 { values, nulls } => {
                            if nulls[row] {
                                continue 'rows;
                            }
                            crate::core::Value::Float(values[row])
                        }
                        TypedCells::TimestampNanos { values, nulls } => {
                            if nulls[row] {
                                continue 'rows;
                            }
                            super::writer::timestamp_value(values[row])
                        }
                        TypedCells::Boolean { values, nulls } => {
                            if nulls[row] {
                                continue 'rows;
                            }
                            crate::core::Value::Boolean(values[row])
                        }
                        TypedCells::Dictionary { ids, nulls } => {
                            if nulls[row] {
                                continue 'rows;
                            }
                            let text = builder
                                .dictionary(col)
                                .and_then(|dict| dict.get(ids[row] as usize))
                                .ok_or_else(|| {
                                    Error::internal("dictionary id beyond the dictionary")
                                })?;
                            crate::core::Value::Text(text.clone())
                        }
                        TypedCells::Bytes {
                            data,
                            offsets,
                            nulls,
                        } => {
                            if nulls[row] {
                                continue 'rows;
                            }
                            let (offset, length) = offsets[row];
                            let (_, ext_tag) = builder.column_kind(col);
                            let mut tagged = Vec::with_capacity(1 + length as usize);
                            tagged.push(ext_tag);
                            tagged.extend_from_slice(
                                &data[offset as usize..(offset + length) as usize],
                            );
                            crate::core::Value::Extension(crate::common::CompactArc::from(tagged))
                        }
                    };
                    value.hash(&mut hasher);
                }
                entries.push((hasher.finish(), first + row as u32));
            }
        }
        Ok(())
    }

    /// Encodes the accumulated group's blocks into the block file
    fn flush_group(&mut self) -> Result<()> {
        if !self.unique.is_empty() {
            self.hash_unique()?;
        }
        // The previous group, encoded while this one accumulated, is
        // written first, and its accumulators take this group's place
        self.write_encoded()?;
        let builder = self
            .builder
            .as_mut()
            .ok_or_else(|| Error::internal("volume writer already finished"))?;
        let mut group = match self.spare.take() {
            Some(group) => group,
            None => builder.spare_accumulators(),
        };
        if builder.flush_group_into(&mut group)? == 0 {
            self.spare = Some(group);
            return Ok(());
        }
        let blocks = match self.spare_blocks.take() {
            Some(blocks) => blocks,
            None => vec![(Vec::new(), 0); group.columns()],
        };
        self.encoder.submit(EncodeJob {
            group,
            blocks,
            result: Ok(()),
        })
    }

    /// Waits for the group handed to the encoder last and appends its
    /// blocks to the file, column by column
    fn write_encoded(&mut self) -> Result<()> {
        let Some(EncodeJob {
            group,
            blocks,
            result,
        }) = self.encoder.take()?
        else {
            return Ok(());
        };
        self.spare = Some(group);
        result?;
        let file = self
            .blocks
            .as_mut()
            .ok_or_else(|| Error::internal("volume writer already finished"))?;
        for (col_idx, (bytes, decompressed_len)) in blocks.iter().enumerate() {
            file.write_all(bytes)
                .map_err(|e| io_error("failed to write volume block", e))?;
            let group = self.index[col_idx].len();
            let entry = super::v5::BlockEntry::new(
                self.block_pos,
                bytes,
                *decompressed_len,
                col_idx,
                group,
            )
            .map_err(|e| io_error("V5 block", e))?;
            self.index[col_idx].push((self.block_pos, bytes.len(), *decompressed_len, entry.crc()));
            self.entries.push(entry);
            self.block_pos += bytes.len() as u64;
        }
        self.spare_blocks = Some(blocks);
        Ok(())
    }

    /// Writes the last group, the metadata and the final file, syncs it
    /// and renames it into place; returns the volume, its columns backed
    /// by the file, and the path
    pub fn finish(mut self) -> Result<(FrozenVolume, PathBuf)> {
        if self.builder()?.group_len() > 0 {
            self.flush_group()?;
        }
        self.write_encoded()?;
        let mut blocks = self
            .blocks
            .take()
            .ok_or_else(|| Error::internal("volume writer already finished"))?;

        let builder = self
            .builder
            .take()
            .ok_or_else(|| Error::internal("volume writer already finished"))?;
        let kinds: Vec<(u8, u8)> = (0..self.index.len())
            .map(|col| builder.column_kind(col))
            .collect();
        // A volume still deciding at its end is one group
        let (meta, dict_tables) =
            builder.finish_streamed(self.group_rows.unwrap_or(ROW_GROUP_SIZE))?;
        let dictionaries: Vec<&[crate::common::SmartString]> =
            dict_tables.iter().map(Vec::as_slice).collect();
        let sections = super::format::v5_metadata_sections(&meta, &kinds, &dictionaries)
            .map_err(|e| io_error("V5 metadata", e))?;

        let col_count = self.index.len();
        let num_groups = self.index.first().map(Vec::len).unwrap_or(0);
        if self.index.iter().any(|col| col.len() != num_groups) {
            return Err(Error::internal(
                "volume columns have different group counts",
            ));
        }

        // The blocks are in the file already, group by group; the
        // sections, their table and the footer follow them
        let root = super::v5::Root {
            cols: col_count as u32,
            groups: num_groups as u32,
            group_rows: meta.group_rows as u32,
            rows: meta.row_count as u64,
            volume_id: self.volume_id,
            nonce: super::v5::new_nonce(),
        };
        let identity =
            super::v5::write_tail(&mut blocks, self.block_pos, &root, sections, &self.entries)
                .map_err(|e| io_error("failed to write volume file", e))?;
        let _ = meta.file_identity.set(identity);
        let file = blocks
            .into_inner()
            .map_err(|e| io_error("failed to flush volume file", e.into_error()))?;
        super::io::sync_ordered(&file)
            .map_err(|e| io_error("failed to fsync volume tmp file", e))?;
        drop(file);
        std::fs::rename(&self.blocks_path, &self.final_path)
            .map_err(|e| io_error("failed to rename volume file", e))?;
        // From here the final file exists; whatever fails until the volume
        // is handed over removes it, since nothing has published it
        let published = self.publish(&kinds, &dict_tables, meta, col_count, num_groups);
        match published {
            Ok(volume) => {
                self.finished = true;
                Ok((volume, self.final_path.clone()))
            }
            Err(error) => {
                let _ = std::fs::remove_file(&self.final_path);
                Err(error)
            }
        }
    }

    /// Syncs the directory and builds the volume over the file just
    /// written, its blocks at the positions they were written to
    fn publish(
        &mut self,
        kinds: &[(u8, u8)],
        dict_tables: &[Vec<crate::common::SmartString>],
        meta: super::writer::VolumeMeta,
        col_count: usize,
        num_groups: usize,
    ) -> Result<FrozenVolume> {
        #[cfg(not(windows))]
        if let Some(dir) = self.final_path.parent() {
            let d = File::open(dir).map_err(|e| io_error("failed to open volume directory", e))?;
            super::io::sync_ordered(&d)
                .map_err(|e| io_error("failed to fsync volume directory", e))?;
        }
        // The volume is handed over only over a file its readers can open
        File::open(&self.final_path).map_err(|e| io_error("failed to open volume file", e))?;
        let mut offsets = Vec::with_capacity(col_count);
        let mut comp_lens = Vec::with_capacity(col_count);
        let mut decomp_lens = Vec::with_capacity(col_count);
        let mut crcs = Vec::with_capacity(col_count);
        for col in &self.index {
            let mut col_offsets = Vec::with_capacity(num_groups);
            let mut col_comp = Vec::with_capacity(num_groups);
            let mut col_decomp = Vec::with_capacity(num_groups);
            let mut col_crcs = Vec::with_capacity(num_groups);
            for &(position, compressed_len, decompressed_len, crc) in col {
                col_offsets.push(position);
                col_comp.push(compressed_len);
                col_decomp.push(decompressed_len);
                col_crcs.push(crc);
            }
            offsets.push(col_offsets);
            comp_lens.push(col_comp);
            decomp_lens.push(col_decomp);
            crcs.push(col_crcs);
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
        let mut store = CompressedBlockStore::from_file(
            self.final_path.clone(),
            offsets,
            comp_lens,
            decomp_lens,
            kinds.iter().map(|k| k.0).collect(),
            kinds.iter().map(|k| k.1).collect(),
            shared_dict,
            dict_ranges,
            meta.group_rows,
            meta.row_count,
        );
        store.check_blocks(crcs);
        let mut unique_indices = rustc_hash::FxHashMap::default();
        for UniqueIndexing {
            columns,
            mut entries,
        } in std::mem::take(&mut self.unique)
        {
            entries.sort_unstable_by_key(|&(h, _)| h);
            unique_indices.insert(columns, Arc::new(entries));
        }
        Ok(FrozenVolume {
            columns: LazyColumns::deferred(store, column_types),
            meta: Arc::new(meta),
            unique_indices: Arc::new(super::writer::UniqueIndexes::new(unique_indices)),
            last_access_epoch: std::sync::atomic::AtomicU64::new(
                super::writer::GLOBAL_EVICTION_EPOCH.load(std::sync::atomic::Ordering::Relaxed),
            ),
        })
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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{DataType, Row, SchemaBuilder, Value};
    use crate::storage::volume::column::ColumnData;
    use crate::storage::volume::format::{
        V5_BLOOMS, V5_COLUMNS, V5_DICT, V5_GROUPS, V5_ROW_IDS, V5_STATS, V5_ZONES,
    };
    use crate::storage::volume::io::{read_volume_from_disk, write_volume_to_disk};

    /// The file the writer makes from `volume`'s rows fed one at a time
    fn row_by_row(volume: &FrozenVolume, volume_id: u64) -> Vec<u8> {
        let dir = tempfile::tempdir().unwrap();
        let path = write_volume_to_disk(dir.path(), "t", volume_id, volume).unwrap();
        std::fs::read(path).unwrap()
    }

    type Layout = (
        Vec<u8>,
        Vec<(u32, u8, Vec<u8>)>,
        Vec<Vec<usize>>,
        Vec<Vec<u32>>,
        [u64; 5],
    );

    /// A V5 file's blocks, sections and root, all but its random nonce
    fn layout(bytes: &[u8]) -> Layout {
        let read_at = |buf: &mut [u8], at: u64| {
            buf.copy_from_slice(&bytes[at as usize..at as usize + buf.len()]);
            Ok(())
        };
        let known = [
            V5_COLUMNS, V5_GROUPS, V5_DICT, V5_ROW_IDS, V5_ZONES, V5_BLOOMS, V5_STATS,
        ];
        let o =
            crate::storage::volume::v5::open(bytes.len() as u64, &known, &[], &read_at).unwrap();
        let p = o
            .places(o.root.cols as usize, o.root.groups as usize)
            .unwrap();
        let end = p
            .offsets
            .iter()
            .zip(&p.stored)
            .flat_map(|(at, len)| at.iter().zip(len).map(|(at, len)| at + *len as u64))
            .max()
            .unwrap_or(crate::storage::volume::v5::HEADER_LEN) as usize;
        let r = &o.root;
        (
            bytes[..end].to_vec(),
            o.meta,
            p.decoded,
            p.crcs,
            [
                r.cols as u64,
                r.groups as u64,
                r.group_rows as u64,
                r.rows,
                r.volume_id,
            ],
        )
    }

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
        let bytes = row_by_row(&by_rows, 7);

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

    /// CRC32 of the metadata the serializer wrote for this fixture before
    /// its buffer was counted; the bytes must not change
    const METADATA_CRC: u32 = 0x2502_8372;

    #[test]
    fn the_metadata_buffer_is_counted_exactly_and_the_bytes_are_unchanged() {
        let schema = schema();
        // Bloom filters, a text dictionary, JSON and text min and max of
        // varying length, three row groups
        let (_, volume, _, _dir) = written(&schema, 2 * ROW_GROUP_SIZE as i64 + 5_000, 10_000);
        let buf = super::super::format::serialize_volume_metadata(&volume).unwrap();
        assert_eq!(
            crc32fast::hash(&buf),
            METADATA_CRC,
            "the metadata bytes changed"
        );
        assert_eq!(
            buf.capacity(),
            buf.len(),
            "the buffer was grown past its counted length"
        );
    }

    #[test]
    fn typed_batches_write_the_file_rows_fed_one_at_a_time_write() {
        let schema = schema();
        for (rows, batch) in [(500, 64), (500, 500), (3, 1)] {
            let (want, volume, path, _dir) = written(&schema, rows, batch);
            let got = std::fs::read(&path).unwrap();
            assert_eq!(got.len(), want.len(), "{rows} rows in batches of {batch}");
            assert!(
                layout(&got) == layout(&want),
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
        let want = row_by_row(&by_rows, 9);

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
        // 10,000 a batch, which does not divide the group size: the
        // boundaries fall inside batches
        let mut start = 0;
        while start < rows as usize {
            let end = (start + 10_000).min(rows as usize);
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
        assert!(layout(&std::fs::read(&path).unwrap()) == layout(&want));
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

    /// The final file is renamed into place before the volume is built
    /// over it; a failure after the rename must not leave it behind
    #[cfg(unix)]
    #[test]
    fn a_failure_after_the_rename_removes_the_final_file() {
        use std::os::unix::fs::PermissionsExt;
        let schema = schema();
        let dir = tempfile::tempdir().unwrap();
        let mut writer = VolumeFileWriter::new(dir.path(), "t", 13, &schema, 10, true).unwrap();
        // The file being written is writable but not readable: the writer
        // writes it, renames it, and cannot open it to read
        let tmp = writer.path().with_extension("blocks.tmp");
        std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o200)).unwrap();
        let ids = [1i64];
        let nulls = [false];
        let name = writer.builder().unwrap().intern_text(5, "a").unwrap();
        writer
            .append_typed(
                &ids,
                &[
                    TypedCells::Int64 {
                        values: &ids,
                        nulls: &nulls,
                    },
                    TypedCells::Int64 {
                        values: &[5],
                        nulls: &nulls,
                    },
                    TypedCells::Float64 {
                        values: &[0.5],
                        nulls: &nulls,
                    },
                    TypedCells::TimestampNanos {
                        values: &[1],
                        nulls: &nulls,
                    },
                    TypedCells::Boolean {
                        values: &[true],
                        nulls: &nulls,
                    },
                    TypedCells::Dictionary {
                        ids: &[name],
                        nulls: &nulls,
                    },
                    TypedCells::Bytes {
                        data: b"{}",
                        offsets: &[(0, 2)],
                        nulls: &nulls,
                    },
                ],
            )
            .unwrap();
        let final_path = writer.path().to_path_buf();
        let error = match writer.finish() {
            Ok(_) => panic!("finish succeeded on an unreadable file"),
            Err(error) => error,
        };
        assert!(
            error.to_string().contains("failed to open volume file"),
            "{error}"
        );
        assert!(!final_path.exists(), "the final file was left behind");
        assert!(!final_path.with_extension("blocks.tmp").exists());
    }

    #[test]
    fn a_retired_file_stays_until_its_last_reader_lets_go() {
        let schema = schema();
        let (_, volume, path, _dir) = written(&schema, 100, 50);
        drop(volume);
        // Two stores over the same file, as a reload beside a reader
        // still holding the old volume leaves them: they share the file
        let first = Arc::new(read_volume_from_disk(&path).unwrap());
        let second = Arc::new(read_volume_from_disk(&path).unwrap());
        assert!(second.retire_file());
        drop(second);
        assert!(path.exists(), "retired under a reader");
        // The reader still holding the old store reads the file
        assert_eq!(first.get_row(7).unwrap()[0], Value::Integer(7));
        drop(first);
        assert!(!path.exists(), "the file outlived its last holder");
        // A volume without a file-backed store has nothing to retire
        let mut builder = VolumeBuilder::new(&schema);
        builder.add_row(1, &row(1));
        assert!(!builder.finish().unwrap().retire_file());
    }

    #[test]
    fn a_written_volume_holds_its_row_ids_without_spare_capacity() {
        let (_, volume, _, _dir) = written(&schema(), 3000, 128);
        assert_eq!(
            volume.meta.row_ids.capacity(),
            volume.meta.row_ids.len(),
            "the row ids kept the growth of their batches"
        );
    }

    #[test]
    fn a_damaged_file_is_refused_where_it_is_read_and_an_aborted_writer_leaves_nothing() {
        let schema = schema();
        let (_, volume, path, dir) = written(&schema, 200, 50);
        drop(volume);
        let intact = std::fs::read(&path).unwrap();
        let mut bytes = intact.clone();
        bytes[crate::storage::volume::v5::HEADER_LEN as usize] ^= 0xff;
        std::fs::write(&path, &bytes).unwrap();
        let damaged = read_volume_from_disk(&path).unwrap();
        let Err(error) = damaged
            .columns
            .compressed_store()
            .unwrap()
            .group_column(0, 0)
        else {
            panic!("a damaged block was decoded");
        };
        assert!(error.to_string().contains("CRC mismatch"), "{error}");
        drop(damaged);
        let mut bytes = intact;
        let footer = bytes.len() - 8;
        bytes[footer] ^= 0xff;
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

    const UNIQUE_ROWS: usize = 100_000;

    /// The published (len, capacity) of the unique set `set` over columns
    /// `a` and `b`, each null where its rule says, and every capacity the
    /// entries grew to, batch by batch
    fn unique_entries(
        a_null: fn(usize) -> bool,
        b_null: fn(usize) -> bool,
        set: Vec<usize>,
    ) -> (usize, usize, Vec<usize>) {
        let schema = SchemaBuilder::new("t")
            .column("id", DataType::Integer, false, true)
            .column("a", DataType::Integer, true, false)
            .column("b", DataType::Integer, true, false)
            .build();
        let ids: Vec<i64> = (0..UNIQUE_ROWS as i64).collect();
        let values: Vec<i64> = ids.iter().map(|i| i * 3).collect();
        let no_nulls = vec![false; UNIQUE_ROWS];
        let a_nulls: Vec<bool> = (0..UNIQUE_ROWS).map(a_null).collect();
        let b_nulls: Vec<bool> = (0..UNIQUE_ROWS).map(b_null).collect();
        let dir = tempfile::tempdir().unwrap();
        let mut writer =
            VolumeFileWriter::new(dir.path(), "t", 15, &schema, UNIQUE_ROWS, true).unwrap();
        writer.index_unique_sets(vec![set.clone()]);
        let mut grown = Vec::new();
        for start in (0..UNIQUE_ROWS).step_by(4_096) {
            let end = (start + 4_096).min(UNIQUE_ROWS);
            let columns = [
                TypedCells::Int64 {
                    values: &ids[start..end],
                    nulls: &no_nulls[start..end],
                },
                TypedCells::Int64 {
                    values: &values[start..end],
                    nulls: &a_nulls[start..end],
                },
                TypedCells::Int64 {
                    values: &values[start..end],
                    nulls: &b_nulls[start..end],
                },
            ];
            writer.append_typed(&ids[start..end], &columns).unwrap();
            let capacity = writer.unique[0].entries.capacity();
            if grown.last().copied().unwrap_or(0) != capacity {
                grown.push(capacity);
            }
        }
        let (volume, _) = writer.finish().unwrap();
        let indices = volume.unique_indices.read();
        let entries = indices.get(&set).unwrap();
        (entries.len(), entries.capacity(), grown)
    }

    #[test]
    fn a_full_unique_index_is_published_without_growth_slack() {
        let (len, capacity, grown) = unique_entries(|_| false, |_| false, vec![1]);
        assert_eq!(len, UNIQUE_ROWS);
        assert_eq!(capacity, UNIQUE_ROWS, "entries hold no growth slack");
        assert_eq!(grown, [65_536], "the first group's rows, a group at a time");
    }

    #[test]
    fn an_all_null_unique_index_reserves_nothing() {
        let (len, capacity, grown) = unique_entries(|_| true, |_| false, vec![1]);
        assert_eq!((len, capacity), (0, 0));
        assert!(grown.is_empty());
    }

    #[test]
    fn a_mostly_null_unique_index_grows_as_push_would() {
        let (len, capacity, grown) = unique_entries(|row| row % 10_000 != 0, |_| false, vec![1]);
        assert_eq!((len, capacity), (10, 16));
        assert_eq!(
            grown,
            [8],
            "the first group's 7 entries, rounded up as push would"
        );
    }

    #[test]
    fn a_composite_set_with_a_null_column_reserves_nothing() {
        let (len, capacity, grown) = unique_entries(|_| false, |_| true, vec![1, 2]);
        assert_eq!((len, capacity), (0, 0));
        assert!(grown.is_empty());
    }

    #[test]
    fn a_three_column_set_counts_only_rows_without_nulls() {
        let (len, capacity, grown) =
            unique_entries(|_| false, |row| row % 10_000 != 0, vec![0, 1, 2]);
        assert_eq!((len, capacity), (10, 16));
        assert_eq!(grown, [8]);
    }

    /// `rows` rows of an id and a JSON document of about `bytes` bytes,
    /// written row by row, or in typed batches of `batch` rows
    fn wide(
        rows: i64,
        bytes: usize,
        batch: Option<usize>,
    ) -> (FrozenVolume, Vec<u8>, PathBuf, tempfile::TempDir) {
        let schema = SchemaBuilder::new("t")
            .column("id", DataType::Integer, false, true)
            .column("doc", DataType::Json, true, false)
            .build();
        let doc = |i: i64| format!("{{\"i\":{i},\"k\":\"{}\"}}", "x".repeat(bytes));
        let dir = tempfile::tempdir().unwrap();
        let mut writer =
            VolumeFileWriter::new(dir.path(), "t", 21, &schema, rows as usize, true).unwrap();
        match batch {
            None => {
                for i in 0..rows {
                    let row = Row::from_values(vec![Value::Integer(i), Value::json(doc(i))]);
                    writer.add_row(i, &row).unwrap();
                }
            }
            Some(batch) => {
                let ids: Vec<i64> = (0..rows).collect();
                let nulls = vec![false; rows as usize];
                for start in (0..rows as usize).step_by(batch) {
                    let end = (start + batch).min(rows as usize);
                    let mut data = Vec::new();
                    let mut offsets = Vec::new();
                    for &i in &ids[start..end] {
                        let text = doc(i);
                        offsets.push((data.len() as u64, text.len() as u64));
                        data.extend_from_slice(text.as_bytes());
                    }
                    let cells = [
                        TypedCells::Int64 {
                            values: &ids[start..end],
                            nulls: &nulls[start..end],
                        },
                        TypedCells::Bytes {
                            data: &data,
                            offsets: &offsets,
                            nulls: &nulls[start..end],
                        },
                    ];
                    writer.append_typed(&ids[start..end], &cells).unwrap();
                }
            }
        }
        let (volume, path) = writer.finish().unwrap();
        let bytes = std::fs::read(&path).unwrap();
        (volume, bytes, path, dir)
    }

    /// The volume read back from `path`, checked row by row against what
    /// was written at the rows on either side of every group boundary
    fn read_back(path: &Path, rows: i64, bytes: usize) -> FrozenVolume {
        let read = read_volume_from_disk(path).unwrap();
        let size = read.meta.group_rows as i64;
        let mut at: Vec<i64> = (1..=rows / size)
            .flat_map(|g| [g * size - 1, g * size])
            .collect();
        at.extend([0, rows - 1]);
        for i in at.into_iter().filter(|&i| i < rows) {
            let row = read.get_row(i as usize).unwrap();
            assert_eq!(row.get(0), Some(&Value::Integer(i)), "row {i}");
            let want = format!("{{\"i\":{i},\"k\":\"{}\"}}", "x".repeat(bytes));
            assert_eq!(row.get(1), Some(&Value::json(want)), "row {i}");
        }
        read
    }

    #[test]
    fn a_volume_of_wide_rows_takes_the_group_size_its_first_group_measures() {
        // 3,000 bytes a row: about 3 MiB at 1,024 rows, so a doubling stays
        // under 8 MiB; about 6 MiB at 2,048, so the next one would not
        let (volume, by_rows, path, _dir) = wide(10_000, 3_000, None);
        assert_eq!(volume.meta.group_rows, 2_048);
        assert_eq!(volume.meta.row_groups.len(), 5);
        assert_eq!(read_back(&path, 10_000, 3_000).meta.group_rows, 2_048);
        // Batches of 1,500 rows cross both the measures and the cuts
        let (_, by_batches, _, _dir2) = wide(10_000, 3_000, Some(1_500));
        assert!(
            layout(&by_batches) == layout(&by_rows),
            "batches and rows wrote different files"
        );
    }

    #[test]
    fn rows_past_the_target_at_the_floor_take_the_floor() {
        let (volume, _, path, _dir) = wide(3_000, 20_000, None);
        assert_eq!(volume.meta.group_rows, 1_024);
        assert_eq!(volume.meta.row_groups.len(), 3);
        read_back(&path, 3_000, 20_000);
    }

    #[test]
    fn a_wide_volume_that_ends_before_its_size_is_decided_is_one_group() {
        let (volume, _, path, _dir) = wide(1_500, 3_000, None);
        assert_eq!(volume.meta.group_rows, ROW_GROUP_SIZE);
        assert!(volume.meta.row_groups.is_empty());
        read_back(&path, 1_500, 3_000);
    }

    #[test]
    fn the_measured_block_length_is_the_length_of_the_block_written() {
        let values = [3i64, -7, 0, 9];
        let floats = [1.5f64, f64::NAN, 0.0, -2.0];
        let flags = [true, false, true, true];
        let ids = [0u32, 4, 1, 2];
        let nulls = [false, true, false, false];
        let data = b"{}{\"a\":1}xyz";
        // A null's cell, an empty one and one past the data hold no bytes
        let offsets = [(0u64, 2u64), (0, 0), (2, 7), (9, 9)];
        for cells in [
            TypedCells::Int64 {
                values: &values,
                nulls: &nulls,
            },
            TypedCells::TimestampNanos {
                values: &values,
                nulls: &nulls,
            },
            TypedCells::Float64 {
                values: &floats,
                nulls: &nulls,
            },
            TypedCells::Boolean {
                values: &flags,
                nulls: &nulls,
            },
            TypedCells::Dictionary {
                ids: &ids,
                nulls: &nulls,
            },
            TypedCells::Bytes {
                data,
                offsets: &offsets,
                nulls: &nulls,
            },
        ] {
            assert_eq!(
                super::super::format::typed_block_len(&cells),
                serialize_typed_block(&cells).len()
            );
        }
    }
}
