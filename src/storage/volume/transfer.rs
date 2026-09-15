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

//! Moves rows from sealed volumes into a `VolumeBuilder` column by column.
//!
//! Compaction names its output rows as references (row id, input, row
//! index) in output order. A transfer takes them in bounded batches:
//! for each batch it holds one handle per (input, row group) the batch
//! touches, gathers every output column from the handles' typed columns
//! into a scratch buffer of the column's storage form, and appends the
//! batch to the builder in one call. No `Row` or `Value` is built for a
//! cell whose source has the output's type; a text cell moves as a
//! dictionary id through a per-input remap that copies a string into the
//! output's dictionary once per distinct entry.
//!
//! A handle borrows a column the input already holds decoded and reads
//! any other column one row group at a time through the decoded group
//! cache, so an input is never decoded whole for the transfer and never
//! promoted to resident by it. The handles held at once are those of the
//! current batch; their decoded bytes are counted apart from the cache's
//! budget, since a group the cache lets go lives on while a handle holds
//! it.

use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::core::{DataType, Error, Result, Schema, Value};

use super::column::{ColumnData, ROW_GROUP_SIZE};
use super::writer::{ColSource, ColumnMapping, FrozenVolume, TypedCells, VolumeBuilder};

/// Rows a transfer moves per batch: the bound on the handles held at once
/// and on the scratch buffers
pub const TRANSFER_BATCH_ROWS: usize = 4096;

/// A column of an input as one batch reads it
enum Cell<'a> {
    /// Decoded in the input already; indexed by the input's row index
    Resident(&'a ColumnData),
    /// One row group, shared with the decoded group cache; indexed by the
    /// row index less the group's first row
    Group(Arc<ColumnData>, usize),
}

impl Cell<'_> {
    fn column_and_local(&self, row: usize) -> (&ColumnData, usize) {
        match self {
            Cell::Resident(column) => (column, row),
            Cell::Group(column, start) => (column, row - start),
        }
    }
}

/// The columns one batch reads from one (input, row group)
struct Handle<'a> {
    input: usize,
    group: usize,
    /// By the input's physical column; None for a column the plan does
    /// not read
    columns: Vec<Option<Cell<'a>>>,
    /// The decoded bytes the handle holds on its own, apart from the cache
    held_bytes: usize,
    used: bool,
}

/// Where an output column comes from in one input
enum Source {
    Column(usize),
    Default(Value),
}

/// A scratch buffer for one output column, in its storage form
enum Scratch {
    Int64(Vec<i64>, Vec<bool>),
    Float64(Vec<f64>, Vec<bool>),
    TimestampNanos(Vec<i64>, Vec<bool>),
    Boolean(Vec<bool>, Vec<bool>),
    Dictionary(Vec<u32>, Vec<bool>),
    Bytes(Vec<u8>, Vec<(u64, u64)>, Vec<bool>),
}

impl Scratch {
    fn for_type(data_type: DataType) -> Self {
        match data_type {
            DataType::Integer => Scratch::Int64(Vec::new(), Vec::new()),
            DataType::Float => Scratch::Float64(Vec::new(), Vec::new()),
            DataType::Timestamp => Scratch::TimestampNanos(Vec::new(), Vec::new()),
            DataType::Boolean => Scratch::Boolean(Vec::new(), Vec::new()),
            DataType::Text => Scratch::Dictionary(Vec::new(), Vec::new()),
            _ => Scratch::Bytes(Vec::new(), Vec::new(), Vec::new()),
        }
    }

    fn clear(&mut self) {
        match self {
            Scratch::Int64(values, nulls) | Scratch::TimestampNanos(values, nulls) => {
                values.clear();
                nulls.clear();
            }
            Scratch::Float64(values, nulls) => {
                values.clear();
                nulls.clear();
            }
            Scratch::Boolean(values, nulls) => {
                values.clear();
                nulls.clear();
            }
            Scratch::Dictionary(ids, nulls) => {
                ids.clear();
                nulls.clear();
            }
            Scratch::Bytes(data, offsets, nulls) => {
                data.clear();
                offsets.clear();
                nulls.clear();
            }
        }
    }

    fn cells(&self) -> TypedCells<'_> {
        match self {
            Scratch::Int64(values, nulls) => TypedCells::Int64 { values, nulls },
            Scratch::Float64(values, nulls) => TypedCells::Float64 { values, nulls },
            Scratch::TimestampNanos(values, nulls) => TypedCells::TimestampNanos { values, nulls },
            Scratch::Boolean(values, nulls) => TypedCells::Boolean { values, nulls },
            Scratch::Dictionary(ids, nulls) => TypedCells::Dictionary { ids, nulls },
            Scratch::Bytes(data, offsets, nulls) => TypedCells::Bytes {
                data,
                offsets,
                nulls,
            },
        }
    }

    fn push_null(&mut self) {
        match self {
            Scratch::Int64(values, nulls) | Scratch::TimestampNanos(values, nulls) => {
                values.push(0);
                nulls.push(true);
            }
            Scratch::Float64(values, nulls) => {
                values.push(0.0);
                nulls.push(true);
            }
            Scratch::Boolean(values, nulls) => {
                values.push(false);
                nulls.push(true);
            }
            Scratch::Dictionary(ids, nulls) => {
                ids.push(0);
                nulls.push(true);
            }
            Scratch::Bytes(_, offsets, nulls) => {
                offsets.push((0, 0));
                nulls.push(true);
            }
        }
    }

    /// Pushes a value the way `add_row` stores it: a value of another
    /// type becomes the column's zero, an extension payload loses its tag.
    /// Text goes through `text` for its id in the output's dictionary
    fn push_value(
        &mut self,
        value: &Value,
        text: &mut dyn FnMut(&str) -> Result<u32>,
    ) -> Result<()> {
        if value.is_null() {
            self.push_null();
            return Ok(());
        }
        match self {
            Scratch::Int64(values, nulls) => {
                values.push(match value {
                    Value::Integer(i) => *i,
                    _ => 0,
                });
                nulls.push(false);
            }
            Scratch::Float64(values, nulls) => {
                values.push(match value {
                    Value::Float(f) => *f,
                    _ => 0.0,
                });
                nulls.push(false);
            }
            Scratch::TimestampNanos(values, nulls) => {
                values.push(match value {
                    Value::Timestamp(ts) => ts.timestamp_nanos_opt().unwrap_or_else(|| {
                        ts.timestamp()
                            .wrapping_mul(1_000_000_000)
                            .wrapping_add(ts.timestamp_subsec_nanos() as i64)
                    }),
                    _ => 0,
                });
                nulls.push(false);
            }
            Scratch::Boolean(values, nulls) => {
                values.push(matches!(value, Value::Boolean(true)));
                nulls.push(false);
            }
            Scratch::Dictionary(ids, nulls) => {
                let s = match value {
                    Value::Text(s) => s.as_str(),
                    _ => "",
                };
                ids.push(text(s)?);
                nulls.push(false);
            }
            Scratch::Bytes(data, offsets, nulls) => {
                let payload: &[u8] = match value {
                    Value::Extension(bytes) if bytes.len() > 1 => &bytes[1..],
                    _ => &[],
                };
                offsets.push((data.len() as u64, payload.len() as u64));
                data.extend_from_slice(payload);
                nulls.push(false);
            }
        }
        Ok(())
    }
}

/// Moves compaction's output rows from its inputs into builders, one
/// bounded batch at a time. One transfer serves every output of a
/// compaction; `begin_output` resets what belongs to one builder
pub struct Transfer<'a> {
    inputs: &'a [(u64, Arc<FrozenVolume>)],
    /// By input, by output column
    plans: Vec<Vec<Source>>,
    /// By input: the physical columns some plan entry reads
    needed: Vec<Vec<bool>>,
    handles: Vec<Handle<'a>>,
    handle_of: FxHashMap<(u32, u32), usize>,
    /// By reference in the batch: its handle
    ref_handle: Vec<u32>,
    /// By input, by output column: the input's dictionary id to the
    /// output's; u32::MAX until the entry is seen. Reset per output
    remaps: Vec<Vec<Vec<u32>>>,
    /// By input, by output column: the id of a text default in the
    /// output's dictionary once interned. Reset per output
    default_ids: Vec<Vec<Option<u32>>>,
    scratch: Vec<Scratch>,
    row_ids: Vec<i64>,
    held_bytes: usize,
    peak_held_bytes: usize,
}

impl<'a> Transfer<'a> {
    /// A transfer from `inputs` into volumes of `schema`, each input read
    /// through its mapping
    pub fn new(
        schema: &Schema,
        inputs: &'a [(u64, Arc<FrozenVolume>)],
        mappings: &[ColumnMapping],
    ) -> Result<Self> {
        if mappings.len() != inputs.len() {
            return Err(Error::internal("one column mapping per compaction input"));
        }
        let columns = schema.columns.len();
        let mut plans = Vec::with_capacity(inputs.len());
        let mut needed = Vec::with_capacity(inputs.len());
        for ((_, volume), mapping) in inputs.iter().zip(mappings) {
            let physical = volume.columns.len();
            let plan: Vec<Source> = if mapping.is_identity {
                (0..columns).map(Source::Column).collect()
            } else {
                if mapping.sources.len() != columns {
                    return Err(Error::internal("column mapping does not cover the schema"));
                }
                mapping
                    .sources
                    .iter()
                    .map(|source| match source {
                        ColSource::Volume(index) => Source::Column(*index),
                        ColSource::Default(value) => Source::Default(value.clone()),
                    })
                    .collect()
            };
            let mut reads = vec![false; physical];
            for source in &plan {
                if let Source::Column(index) = source {
                    match reads.get_mut(*index) {
                        Some(read) => *read = true,
                        None => {
                            return Err(Error::internal(
                                "column mapping names a column the volume lacks",
                            ))
                        }
                    }
                }
            }
            plans.push(plan);
            needed.push(reads);
        }
        Ok(Self {
            inputs,
            plans,
            needed,
            handles: Vec::new(),
            handle_of: FxHashMap::default(),
            ref_handle: Vec::new(),
            remaps: vec![vec![Vec::new(); columns]; inputs.len()],
            default_ids: vec![vec![None; columns]; inputs.len()],
            scratch: schema
                .columns
                .iter()
                .map(|column| Scratch::for_type(column.data_type))
                .collect(),
            row_ids: Vec::new(),
            held_bytes: 0,
            peak_held_bytes: 0,
        })
    }

    /// The next output starts: the dictionary remaps belong to the
    /// builder that is done, so they are cleared, capacity kept
    pub fn begin_output(&mut self) {
        for remap in self.remaps.iter_mut().flatten() {
            remap.clear();
        }
        for ids in self.default_ids.iter_mut().flatten() {
            *ids = None;
        }
    }

    /// The most decoded bytes the transfer's handles held at once
    pub fn peak_held_bytes(&self) -> usize {
        self.peak_held_bytes
    }

    /// Appends `refs` (row id, input, row index), in that order, to
    /// `builder`; at most `TRANSFER_BATCH_ROWS` of them per call
    pub fn append(
        &mut self,
        refs: &[(i64, usize, usize)],
        builder: &mut VolumeBuilder,
    ) -> Result<()> {
        if refs.len() > TRANSFER_BATCH_ROWS {
            return Err(Error::internal("transfer batch exceeds its bound"));
        }
        self.take_handles(refs)?;
        for scratch in &mut self.scratch {
            scratch.clear();
        }
        self.row_ids.clear();
        self.row_ids.extend(refs.iter().map(|r| r.0));
        for column in 0..self.scratch.len() {
            gather(
                column,
                refs,
                &self.ref_handle,
                &self.handles,
                &self.plans,
                &mut self.remaps,
                &mut self.default_ids,
                &mut self.scratch[column],
                builder,
            )?;
        }
        let cells: Vec<TypedCells<'_>> = self.scratch.iter().map(Scratch::cells).collect();
        builder.append_typed(&self.row_ids, &cells)
    }

    /// Holds a handle for every (input, group) the batch reads and lets
    /// go of the ones it does not, before the new ones are taken
    fn take_handles(&mut self, refs: &[(i64, usize, usize)]) -> Result<()> {
        for handle in &mut self.handles {
            handle.used = false;
        }
        self.ref_handle.clear();
        self.ref_handle.reserve(refs.len());
        let mut missing = false;
        for &(_, input, row) in refs {
            let key = (input as u32, (row / ROW_GROUP_SIZE) as u32);
            match self.handle_of.get(&key) {
                Some(&position) => {
                    self.handles[position].used = true;
                    self.ref_handle.push(position as u32);
                }
                None => {
                    missing = true;
                    self.ref_handle.push(u32::MAX);
                }
            }
        }
        if self.handles.iter().any(|handle| !handle.used) {
            let mut kept = Vec::with_capacity(self.handles.len());
            let mut renumber: FxHashMap<usize, usize> = FxHashMap::default();
            for (position, handle) in self.handles.drain(..).enumerate() {
                if handle.used {
                    renumber.insert(position, kept.len());
                    kept.push(handle);
                } else {
                    self.held_bytes -= handle.held_bytes;
                }
            }
            self.handles = kept;
            self.handle_of.clear();
            for (position, handle) in self.handles.iter().enumerate() {
                self.handle_of
                    .insert((handle.input as u32, handle.group as u32), position);
            }
            for slot in &mut self.ref_handle {
                if *slot != u32::MAX {
                    *slot = renumber[&(*slot as usize)] as u32;
                }
            }
        }
        if !missing {
            return Ok(());
        }
        for (index, &(_, input, row)) in refs.iter().enumerate() {
            if self.ref_handle[index] != u32::MAX {
                continue;
            }
            let group = row / ROW_GROUP_SIZE;
            let key = (input as u32, group as u32);
            let position = match self.handle_of.get(&key) {
                Some(&position) => position,
                None => {
                    let handle = self.open_handle(input, group)?;
                    self.held_bytes += handle.held_bytes;
                    self.peak_held_bytes = self.peak_held_bytes.max(self.held_bytes);
                    self.handles.push(handle);
                    let position = self.handles.len() - 1;
                    self.handle_of.insert(key, position);
                    position
                }
            };
            self.ref_handle[index] = position as u32;
        }
        Ok(())
    }

    fn open_handle(&self, input: usize, group: usize) -> Result<Handle<'a>> {
        let volume = &self.inputs[input].1;
        let mut columns = Vec::with_capacity(self.needed[input].len());
        let mut held_bytes = 0;
        for (physical, &needed) in self.needed[input].iter().enumerate() {
            if !needed {
                columns.push(None);
                continue;
            }
            if let Some(column) = volume.columns.resident(physical) {
                columns.push(Some(Cell::Resident(column)));
                continue;
            }
            let store = volume
                .columns
                .compressed_store()
                .ok_or_else(|| Error::internal("compaction input has no column data to read"))?;
            let column = store.group_column(physical, group)?;
            held_bytes += column.cache_size();
            columns.push(Some(Cell::Group(column, group * ROW_GROUP_SIZE)));
        }
        Ok(Handle {
            input,
            group,
            columns,
            held_bytes,
            used: true,
        })
    }
}

/// Fills `scratch` with output column `column` of every reference
#[allow(clippy::too_many_arguments)]
fn gather(
    column: usize,
    refs: &[(i64, usize, usize)],
    ref_handle: &[u32],
    handles: &[Handle<'_>],
    plans: &[Vec<Source>],
    remaps: &mut [Vec<Vec<u32>>],
    default_ids: &mut [Vec<Option<u32>>],
    scratch: &mut Scratch,
    builder: &mut VolumeBuilder,
) -> Result<()> {
    for (&(_, input, row), &handle) in refs.iter().zip(ref_handle) {
        let handle = &handles[handle as usize];
        let physical = match &plans[input][column] {
            Source::Column(physical) => *physical,
            Source::Default(value) => {
                if let Scratch::Dictionary(ids, nulls) = scratch {
                    match value {
                        Value::Text(text) => {
                            let id = match default_ids[input][column] {
                                Some(id) => id,
                                None => {
                                    let id = builder.intern_text(column, text)?;
                                    default_ids[input][column] = Some(id);
                                    id
                                }
                            };
                            ids.push(id);
                            nulls.push(false);
                        }
                        _ => scratch
                            .push_value(value, &mut |text| builder.intern_text(column, text))?,
                    }
                } else {
                    scratch.push_value(value, &mut |text| builder.intern_text(column, text))?;
                }
                continue;
            }
        };
        let cell = handle.columns[physical]
            .as_ref()
            .ok_or_else(|| Error::internal("transfer handle lacks a planned column"))?;
        let (source, local) = cell.column_and_local(row);
        if local >= source.len() {
            return Err(Error::internal(
                "compaction reference beyond its input's rows",
            ));
        }
        match (&mut *scratch, source) {
            (
                Scratch::Int64(values, nulls),
                ColumnData::Int64 {
                    values: v,
                    nulls: n,
                },
            )
            | (
                Scratch::TimestampNanos(values, nulls),
                ColumnData::TimestampNanos {
                    values: v,
                    nulls: n,
                },
            ) => {
                values.push(v[local]);
                nulls.push(n[local]);
            }
            (
                Scratch::Float64(values, nulls),
                ColumnData::Float64 {
                    values: v,
                    nulls: n,
                },
            ) => {
                values.push(v[local]);
                nulls.push(n[local]);
            }
            (
                Scratch::Boolean(values, nulls),
                ColumnData::Boolean {
                    values: v,
                    nulls: n,
                },
            ) => {
                values.push(v[local]);
                nulls.push(n[local]);
            }
            (
                Scratch::Dictionary(ids, nulls),
                ColumnData::Dictionary {
                    ids: source_ids,
                    dictionary,
                    nulls: n,
                },
            ) => {
                if n[local] {
                    ids.push(0);
                    nulls.push(true);
                    continue;
                }
                let source_id = source_ids[local] as usize;
                let remap = &mut remaps[input][column];
                if remap.len() < dictionary.len() {
                    remap.resize(dictionary.len(), u32::MAX);
                }
                let id = match remap.get(source_id) {
                    Some(&id) if id != u32::MAX => id,
                    Some(_) => {
                        let id = builder.intern_text(column, dictionary[source_id].as_str())?;
                        remap[source_id] = id;
                        id
                    }
                    None => return Err(Error::internal("dictionary id beyond its dictionary")),
                };
                ids.push(id);
                nulls.push(false);
            }
            (
                Scratch::Bytes(data, offsets, nulls),
                ColumnData::Bytes {
                    data: source_data,
                    offsets: source_offsets,
                    nulls: n,
                    ..
                },
            ) => {
                if n[local] {
                    offsets.push((0, 0));
                    nulls.push(true);
                    continue;
                }
                let (offset, length) = source_offsets[local];
                let payload = &source_data[offset as usize..(offset + length) as usize];
                offsets.push((data.len() as u64, length));
                data.extend_from_slice(payload);
                nulls.push(false);
            }
            // The input's column has another type than the output's (the
            // column's type changed after the input was sealed): the cell
            // goes through a value, as a row would
            (scratch, source) => {
                let value = source.get_value(local);
                scratch.push_value(&value, &mut |text| builder.intern_text(column, text))?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Row, SchemaBuilder};
    use crate::storage::volume::io::serialize_v4_public;
    use crate::storage::volume::stats::ColumnAggregateStats;

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

    fn timestamp(seconds: i64) -> Value {
        Value::Timestamp(chrono::DateTime::from_timestamp(seconds, 0).unwrap())
    }

    /// Row `i` of input `salt`, with nulls, a NaN and repeated names
    fn row(salt: i64, i: i64) -> Row {
        let null_here = i % 7 == 3;
        Row::from_values(vec![
            Value::Integer(i),
            if null_here {
                Value::Null(DataType::Integer)
            } else {
                Value::Integer(salt * 1000 - i)
            },
            if i % 11 == 5 {
                Value::Float(f64::NAN)
            } else if null_here {
                Value::Null(DataType::Float)
            } else {
                Value::Float(i as f64 * 0.5 - salt as f64)
            },
            if i % 5 == 4 {
                Value::Null(DataType::Timestamp)
            } else {
                timestamp(1_700_000_000 + salt * 100_000 - i * 60)
            },
            if null_here {
                Value::Null(DataType::Boolean)
            } else {
                Value::Boolean(i % 3 == 0)
            },
            if i % 13 == 6 {
                Value::Null(DataType::Text)
            } else {
                Value::text(format!("name{}-{}", salt, i % 17))
            },
            if null_here {
                Value::Null(DataType::Json)
            } else {
                Value::json(format!("{{\"k\":{}}}", i + salt))
            },
        ])
    }

    fn eager(schema: &Schema, salt: i64, rows: i64) -> FrozenVolume {
        let mut builder = VolumeBuilder::new(schema);
        for i in 0..rows {
            builder.add_row(salt * 1_000_000 + i, &row(salt, i));
        }
        builder.finish().unwrap()
    }

    /// The same rows in their compressed form, every column deferred
    fn warm(schema: &Schema, salt: i64, rows: i64) -> FrozenVolume {
        let mut volume = eager(schema, salt, rows);
        let (_, store) = serialize_v4_public(&volume).unwrap();
        volume.columns.attach_compressed_store(store);
        volume.to_warm().unwrap()
    }

    fn identity() -> ColumnMapping {
        ColumnMapping {
            sources: Vec::new(),
            names: Vec::new(),
            is_identity: true,
        }
    }

    /// The volume `add_row` builds from the same references
    fn reference(
        schema: &Schema,
        inputs: &[(u64, Arc<FrozenVolume>)],
        mappings: &[ColumnMapping],
        refs: &[(i64, usize, usize)],
    ) -> FrozenVolume {
        let mut builder = VolumeBuilder::new(schema);
        builder.allow_any_row_order();
        for &(row_id, input, row) in refs {
            let volume = &inputs[input].1;
            let row = if mappings[input].is_identity {
                volume.get_row(row).unwrap()
            } else {
                volume.get_row_mapped(row, &mappings[input]).unwrap()
            };
            builder.add_row(row_id, &row);
        }
        builder.finish().unwrap()
    }

    fn transferred(
        schema: &Schema,
        inputs: &[(u64, Arc<FrozenVolume>)],
        mappings: &[ColumnMapping],
        refs: &[(i64, usize, usize)],
        batch: usize,
    ) -> (FrozenVolume, usize) {
        let mut transfer = Transfer::new(schema, inputs, mappings).unwrap();
        let mut builder = VolumeBuilder::new(schema);
        builder.allow_any_row_order();
        transfer.begin_output();
        for chunk in refs.chunks(batch) {
            transfer.append(chunk, &mut builder).unwrap();
        }
        (builder.finish().unwrap(), transfer.peak_held_bytes())
    }

    fn assert_same_stats(a: &ColumnAggregateStats, b: &ColumnAggregateStats) {
        assert_eq!(a.sum_int, b.sum_int);
        assert_eq!(a.sum_float.to_bits(), b.sum_float.to_bits());
        assert_eq!(a.numeric_count, b.numeric_count);
        assert_eq!(a.non_null_count, b.non_null_count);
        assert_eq!(a.min, b.min);
        assert_eq!(a.max, b.max);
    }

    /// Rows, extents, stats, sortedness and row group maps agree
    fn assert_same_volume(got: &FrozenVolume, want: &FrozenVolume) {
        assert_eq!(got.meta.row_ids, want.meta.row_ids);
        assert_eq!(got.meta.row_count, want.meta.row_count);
        assert_eq!(got.meta.sorted_columns, want.meta.sorted_columns);
        assert_eq!(got.meta.stats.total_rows, want.meta.stats.total_rows);
        assert_eq!(got.meta.stats.live_rows, want.meta.stats.live_rows);
        for column in 0..want.columns.len() {
            let (g, w) = (&got.meta.zone_maps[column], &want.meta.zone_maps[column]);
            assert_eq!(
                (&g.min, &g.max, g.null_count, g.row_count),
                (&w.min, &w.max, w.null_count, w.row_count),
                "zone map of column {column}"
            );
            assert_same_stats(
                &got.meta.stats.columns[column],
                &want.meta.stats.columns[column],
            );
            let (g, w) = (
                got.columns.get(column).unwrap(),
                want.columns.get(column).unwrap(),
            );
            for row in 0..want.meta.row_count {
                let (gv, wv) = (g.get_value(row), w.get_value(row));
                let same = match (&gv, &wv) {
                    (Value::Float(a), Value::Float(b)) => a.to_bits() == b.to_bits(),
                    _ => gv == wv,
                };
                assert!(same, "column {column} row {row}: {gv:?} vs {wv:?}");
            }
        }
        assert_eq!(got.meta.row_groups.len(), want.meta.row_groups.len());
        for (g, w) in got.meta.row_groups.iter().zip(&want.meta.row_groups) {
            assert_eq!((g.start_idx, g.end_idx), (w.start_idx, w.end_idx));
            for (gz, wz) in g.zone_maps.iter().zip(&w.zone_maps) {
                assert_eq!(
                    (&gz.min, &gz.max, gz.null_count, gz.row_count),
                    (&wz.min, &wz.max, wz.null_count, wz.row_count)
                );
            }
        }
    }

    #[test]
    fn rows_from_a_warm_and_a_resident_input_move_as_add_row_would_build_them() {
        let schema = schema();
        let inputs: Vec<(u64, Arc<FrozenVolume>)> = vec![
            (1, Arc::new(warm(&schema, 1, 300))),
            (2, Arc::new(eager(&schema, 2, 200))),
        ];
        let mappings = vec![identity(), identity()];
        // Sources alternate row by row, some rows left out as dead
        let mut refs = Vec::new();
        for i in 0..300usize {
            if i % 9 != 8 {
                refs.push((1_000_000 + i as i64, 0, i));
            }
            if i < 200 && i % 10 != 0 {
                refs.push((2_000_000 + i as i64, 1, i));
            }
        }
        let mut outputs = Vec::new();
        for batch in [7, 64, TRANSFER_BATCH_ROWS] {
            outputs.push(transferred(&schema, &inputs, &mappings, &refs, batch).0);
        }
        // The warm input was not decoded whole by the transfer; the
        // reference build below decodes it
        assert!(inputs[0].1.columns.resident(5).is_none());
        let want = reference(&schema, &inputs, &mappings, &refs);
        for got in &outputs {
            assert_same_volume(got, &want);
        }
    }

    #[test]
    fn a_column_the_input_predates_takes_its_default_and_a_typed_default_its_zero() {
        let schema = schema();
        let old = SchemaBuilder::new("t")
            .column("id", DataType::Integer, false, true)
            .column("n", DataType::Integer, true, false)
            .column("x", DataType::Float, true, false)
            .column("at", DataType::Timestamp, true, false)
            .column("ok", DataType::Boolean, true, false)
            .build();
        let mut builder = VolumeBuilder::new(&old);
        for i in 0..50 {
            let full = row(3, i);
            builder.add_row(
                3_000_000 + i,
                &Row::from_values((0..5).map(|c| full.get(c).unwrap().clone()).collect()),
            );
        }
        let mut volume = builder.finish().unwrap();
        let (_, store) = serialize_v4_public(&volume).unwrap();
        volume.columns.attach_compressed_store(store);
        let inputs: Vec<(u64, Arc<FrozenVolume>)> = vec![
            (3, Arc::new(volume.to_warm().unwrap())),
            (1, Arc::new(eager(&schema, 1, 40))),
        ];
        let mappings = vec![
            ColumnMapping {
                sources: vec![
                    ColSource::Volume(0),
                    ColSource::Volume(1),
                    ColSource::Volume(2),
                    ColSource::Volume(3),
                    ColSource::Volume(4),
                    ColSource::Default(Value::text("unnamed")),
                    ColSource::Default(Value::Null(DataType::Json)),
                ],
                names: Vec::new(),
                is_identity: false,
            },
            identity(),
        ];
        let refs: Vec<(i64, usize, usize)> = (0..40)
            .flat_map(|i| [(3_000_000 + i as i64, 0, i), (1_000_000 + i as i64, 1, i)])
            .chain((40..50).map(|i| (3_000_000 + i as i64, 0, i)))
            .collect();
        let want = reference(&schema, &inputs, &mappings, &refs);
        let (got, _) = transferred(&schema, &inputs, &mappings, &refs, 16);
        assert_same_volume(&got, &want);
        assert_eq!(
            got.columns.get(5).unwrap().get_value(0),
            Value::text("unnamed")
        );
        assert!(got.columns.get(6).unwrap().is_null(0));
    }

    #[test]
    fn every_output_starts_its_own_dictionary() {
        let schema = schema();
        let inputs: Vec<(u64, Arc<FrozenVolume>)> = vec![
            (1, Arc::new(warm(&schema, 1, 120))),
            (2, Arc::new(warm(&schema, 2, 120))),
        ];
        let mappings = vec![identity(), identity()];
        let refs: Vec<(i64, usize, usize)> = (0..120usize)
            .flat_map(|i| [(1_000_000 + i as i64, 0, i), (2_000_000 + i as i64, 1, i)])
            .collect();
        let mut transfer = Transfer::new(&schema, &inputs, &mappings).unwrap();
        let mut outputs = Vec::new();
        // The second output begins with the names of the second input, so
        // its ids differ from the first output's for the same strings
        for (chunk, order) in refs.chunks(120).zip([false, true]) {
            let mut builder = VolumeBuilder::new(&schema);
            builder.allow_any_row_order();
            transfer.begin_output();
            let chunk: Vec<(i64, usize, usize)> = if order {
                chunk.iter().rev().copied().collect()
            } else {
                chunk.to_vec()
            };
            for batch in chunk.chunks(50) {
                transfer.append(batch, &mut builder).unwrap();
            }
            outputs.push((chunk, builder.finish().unwrap()));
        }
        for (chunk, output) in &outputs {
            let want = reference(&schema, &inputs, &mappings, chunk);
            assert_same_volume(output, &want);
        }
        // The first input's first name leads the first output's dictionary
        // and comes second in the second output's
        let first = outputs[0].1.columns.get(5).unwrap();
        let second = outputs[1].1.columns.get(5).unwrap();
        assert_eq!(first.get_str(0), "name1-0");
        assert_eq!(second.get_str(1), "name1-0");
        assert_eq!(first.get_dict_id(0), 0);
        assert_eq!(second.get_dict_id(1), 1);
    }

    #[test]
    fn a_batch_holds_the_groups_it_reads_and_lets_the_rest_go() {
        let schema = SchemaBuilder::new("t")
            .column("id", DataType::Integer, false, true)
            .column("name", DataType::Text, true, false)
            .build();
        let rows = 2 * ROW_GROUP_SIZE as i64 + 5_000;
        let mut builder = VolumeBuilder::new(&schema);
        for i in 0..rows {
            builder.add_row(
                i,
                &Row::from_values(vec![Value::Integer(i), Value::text(format!("n{}", i % 3))]),
            );
        }
        let mut volume = builder.finish().unwrap();
        let (_, store) = serialize_v4_public(&volume).unwrap();
        volume.columns.attach_compressed_store(store);
        let inputs: Vec<(u64, Arc<FrozenVolume>)> = vec![(1, Arc::new(volume.to_warm().unwrap()))];
        let mappings = vec![identity()];
        let refs: Vec<(i64, usize, usize)> = (0..rows as usize)
            .step_by(3)
            .map(|i| (i as i64, 0, i))
            .collect();
        let (got, peak) = transferred(&schema, &inputs, &mappings, &refs, TRANSFER_BATCH_ROWS);
        // A batch spans two groups at most, so at most two of the three
        // are held at once: a group's two columns are 8 and 4 bytes a
        // row plus a null flag each
        let one_group = ROW_GROUP_SIZE * (8 + 1) + ROW_GROUP_SIZE * (4 + 1);
        assert!(peak > 0 && peak <= 2 * one_group, "held {peak} bytes");
        assert!(inputs[0].1.columns.resident(0).is_none());
        let want = reference(&schema, &inputs, &mappings, &refs);
        assert_same_volume(&got, &want);
    }

    #[test]
    fn an_input_without_column_data_fails_the_transfer() {
        let schema = schema();
        let inputs: Vec<(u64, Arc<FrozenVolume>)> =
            vec![(1, Arc::new(eager(&schema, 1, 10).to_cold()))];
        let mappings = vec![identity()];
        let mut transfer = Transfer::new(&schema, &inputs, &mappings).unwrap();
        let mut builder = VolumeBuilder::new(&schema);
        let error = transfer.append(&[(1, 0, 0)], &mut builder).unwrap_err();
        assert!(error.to_string().contains("no column data"), "{error}");
    }
}
