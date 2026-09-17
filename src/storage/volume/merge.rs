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

//! Orders a clustered compaction's live rows by merging its inputs' own
//! key orders, one row group of key columns per input at a time

use std::cmp::Ordering;
use std::sync::Arc;

use crate::core::{Error, Result, Schema, Value};

use super::column::{ColumnData, ROW_GROUP_SIZE};
use super::writer::{ColSource, ColumnMapping, FrozenVolume};

/// One key column of one input as the merge reads it
enum KeyCell<'a> {
    Resident(&'a ColumnData),
    /// One row group and its first row
    Group(Arc<ColumnData>, usize),
    /// The input predates the column
    Default(Value),
}

/// Where an input's key column is
enum KeySource {
    Column(usize),
    Default(Value),
}

/// An input's live rows in their physical order, with the key columns of
/// the group its head row is in
struct Cursor<'a> {
    volume: &'a FrozenVolume,
    sources: Vec<KeySource>,
    refs: &'a [(i64, usize, usize)],
    next: usize,
    group: Option<usize>,
    cells: Vec<KeyCell<'a>>,
    held_bytes: usize,
    /// The key of the last row of the previous group, to check the order
    /// across the boundary
    last_key: Vec<Value>,
}

impl<'a> Cursor<'a> {
    fn head(&self) -> Option<(i64, usize, usize)> {
        self.refs.get(self.next).copied()
    }

    /// The cells of the group `row` is in; the previous group's go
    fn load(&mut self, row: usize) -> Result<()> {
        let group = row / ROW_GROUP_SIZE;
        if self.group == Some(group) {
            return Ok(());
        }
        if self.group.is_some() {
            self.last_key = self.key_values(self.refs[self.next - 1].2);
        }
        self.cells.clear();
        self.held_bytes = 0;
        for source in &self.sources {
            let cell = match source {
                KeySource::Default(value) => KeyCell::Default(value.clone()),
                KeySource::Column(physical) => {
                    if let Some(column) = self.volume.columns.resident(*physical) {
                        KeyCell::Resident(column)
                    } else {
                        let store = self.volume.columns.compressed_store().ok_or_else(|| {
                            Error::internal("compaction input has no column data to read")
                        })?;
                        let column = store.group_column(*physical, group)?;
                        self.held_bytes += column.cache_size();
                        KeyCell::Group(column, group * ROW_GROUP_SIZE)
                    }
                }
            };
            self.cells.push(cell);
        }
        self.group = Some(group);
        Ok(())
    }

    fn key_values(&self, row: usize) -> Vec<Value> {
        self.cells
            .iter()
            .map(|cell| match cell {
                KeyCell::Resident(column) => column.get_value(row),
                KeyCell::Group(column, start) => column.get_value(row - start),
                KeyCell::Default(value) => value.clone(),
            })
            .collect()
    }
}

/// A cell as a column and a local row, or the default it stands for
fn locate<'c>(
    cell: &'c KeyCell<'_>,
    row: usize,
) -> std::result::Result<(&'c ColumnData, usize), &'c Value> {
    match cell {
        KeyCell::Resident(column) => Ok((column, row)),
        KeyCell::Group(column, start) => Ok((column, row - start)),
        KeyCell::Default(value) => Err(value),
    }
}

fn compare_cell(a: &KeyCell<'_>, row_a: usize, b: &KeyCell<'_>, row_b: usize) -> Ordering {
    match (locate(a, row_a), locate(b, row_b)) {
        (Ok((ca, i)), Ok((cb, j))) => ca.compare_cells(i, cb, j),
        (Ok((ca, i)), Err(vb)) => ca.compare_cell_with_value(i, vb),
        (Err(va), Ok((cb, j))) => cb.compare_cell_with_value(j, va).reverse(),
        (Err(va), Err(vb)) => super::seal::compare_key_values(va, vb),
    }
}

/// Orders head row `a` against head row `b` by the key, then by row id
fn compare_heads(a: &Cursor<'_>, b: &Cursor<'_>) -> Ordering {
    let (id_a, _, row_a) = a.refs[a.next];
    let (id_b, _, row_b) = b.refs[b.next];
    a.cells
        .iter()
        .zip(&b.cells)
        .map(|(ca, cb)| compare_cell(ca, row_a, cb, row_b))
        .find(|o| *o != Ordering::Equal)
        .unwrap_or_else(|| id_a.cmp(&id_b))
}

/// Whether the row at `refs[next]` follows the row before it in the
/// cursor's own run, by the key then by row id
fn in_order(cursor: &Cursor<'_>) -> bool {
    let (id, _, row) = cursor.refs[cursor.next];
    let (prev_id, _, prev_row) = cursor.refs[cursor.next - 1];
    let order = if prev_row / ROW_GROUP_SIZE == row / ROW_GROUP_SIZE {
        cursor
            .cells
            .iter()
            .map(|cell| compare_cell(cell, prev_row, cell, row))
            .find(|o| *o != Ordering::Equal)
    } else {
        cursor
            .cells
            .iter()
            .zip(&cursor.last_key)
            .map(|(cell, last)| compare_cell(&KeyCell::Default(last.clone()), 0, cell, row))
            .find(|o| *o != Ordering::Equal)
    };
    order.unwrap_or_else(|| prev_id.cmp(&id)) != Ordering::Greater
}

/// The merged order and the most key bytes the cursors held at once
pub struct Merged {
    pub refs: Vec<(i64, usize, usize)>,
    pub peak_held_bytes: usize,
}

/// The live rows of every input, each input's run in `live_refs` in its
/// physical order, merged by the schema's key then by row id; None when a
/// run turns out not to be in that order, before any of it is used
pub fn merge_key_order<'a>(
    schema: &Schema,
    inputs: &'a [(u64, Arc<FrozenVolume>)],
    mappings: &[ColumnMapping],
    live_refs: &'a [(i64, usize, usize)],
) -> Result<Option<Merged>> {
    if schema.cluster_key.is_empty() || mappings.len() != inputs.len() {
        return Ok(None);
    }
    // One contiguous run per input, in input order
    let mut runs: Vec<(usize, usize)> = vec![(0, 0); inputs.len()];
    let mut seen_input = vec![false; inputs.len()];
    let mut at = 0;
    while at < live_refs.len() {
        let input = live_refs[at].1;
        let start = at;
        while at < live_refs.len() && live_refs[at].1 == input {
            at += 1;
        }
        if input >= inputs.len() || seen_input[input] {
            return Ok(None);
        }
        seen_input[input] = true;
        runs[input] = (start, at);
    }
    let mut cursors: Vec<Cursor<'a>> = Vec::with_capacity(inputs.len());
    for (input, (_, volume)) in inputs.iter().enumerate() {
        let mapping = &mappings[input];
        let sources = schema
            .cluster_key
            .iter()
            .map(|&column| {
                if mapping.is_identity {
                    KeySource::Column(column)
                } else {
                    match mapping.sources.get(column) {
                        Some(ColSource::Volume(v)) => KeySource::Column(*v),
                        Some(ColSource::Default(value)) => KeySource::Default(value.clone()),
                        None => KeySource::Default(Value::null_unknown()),
                    }
                }
            })
            .collect();
        let (start, end) = runs[input];
        let mut cursor = Cursor {
            volume,
            sources,
            refs: &live_refs[start..end],
            next: 0,
            group: None,
            cells: Vec::with_capacity(schema.cluster_key.len()),
            held_bytes: 0,
            last_key: Vec::new(),
        };
        if let Some((_, _, row)) = cursor.head() {
            cursor.load(row)?;
        }
        cursors.push(cursor);
    }
    let mut merged = Vec::with_capacity(live_refs.len());
    let mut held: usize = cursors.iter().map(|c| c.held_bytes).sum();
    let mut peak_held_bytes = held;
    // A heap of the cursors with a head, the least head on top
    let mut heap: Vec<usize> = (0..cursors.len())
        .filter(|&i| cursors[i].head().is_some())
        .collect();
    for i in (0..heap.len() / 2).rev() {
        sift_down(&mut heap, i, &cursors);
    }
    while let Some(&best) = heap.first() {
        let cursor = &mut cursors[best];
        merged.push(cursor.refs[cursor.next]);
        cursor.next += 1;
        let before = cursor.held_bytes;
        if let Some((_, _, row)) = cursor.head() {
            cursor.load(row)?;
            if !in_order(cursor) {
                return Ok(None);
            }
            held = held - before + cursor.held_bytes;
            peak_held_bytes = peak_held_bytes.max(held);
        } else {
            cursor.cells.clear();
            cursor.held_bytes = 0;
            held -= before;
            heap.swap_remove(0);
            if heap.is_empty() {
                break;
            }
        }
        sift_down(&mut heap, 0, &cursors);
    }
    Ok(Some(Merged {
        refs: merged,
        peak_held_bytes,
    }))
}

/// Restores the heap below `at` after its cursor's head changed
fn sift_down(heap: &mut [usize], mut at: usize, cursors: &[Cursor<'_>]) {
    loop {
        let left = 2 * at + 1;
        if left >= heap.len() {
            return;
        }
        let right = left + 1;
        let mut least = left;
        if right < heap.len()
            && compare_heads(&cursors[heap[right]], &cursors[heap[left]]) == Ordering::Less
        {
            least = right;
        }
        if compare_heads(&cursors[heap[least]], &cursors[heap[at]]) != Ordering::Less {
            return;
        }
        heap.swap(at, least);
        at = least;
    }
}

/// Whether `volume` holds its rows in the order of its own `key` columns,
/// read one row group at a time and left as they are
pub fn volume_in_key_order(volume: &FrozenVolume, key: &[usize]) -> Result<bool> {
    let rows = volume.meta.row_count;
    if rows < 2 || key.is_empty() {
        return Ok(true);
    }
    let mut cells: Vec<KeyCell<'_>> = Vec::with_capacity(key.len());
    let mut last_key: Vec<Value> = Vec::new();
    let mut group_start = 0;
    while group_start < rows {
        let group = group_start / ROW_GROUP_SIZE;
        cells.clear();
        for &column in key {
            cells.push(if let Some(resident) = volume.columns.resident(column) {
                KeyCell::Resident(resident)
            } else {
                let store = volume
                    .columns
                    .compressed_store()
                    .ok_or_else(|| Error::internal("volume has no column data to read"))?;
                KeyCell::Group(store.group_column(column, group)?, group_start)
            });
        }
        let group_end = (group_start + ROW_GROUP_SIZE).min(rows);
        if group_start > 0 {
            let across = cells
                .iter()
                .zip(&last_key)
                .map(|(cell, last)| {
                    compare_cell(&KeyCell::Default(last.clone()), 0, cell, group_start)
                })
                .find(|o| *o != Ordering::Equal);
            if across == Some(Ordering::Greater) {
                return Ok(false);
            }
        }
        for row in group_start + 1..group_end {
            let order = cells
                .iter()
                .map(|cell| compare_cell(cell, row - 1, cell, row))
                .find(|o| *o != Ordering::Equal);
            if order == Some(Ordering::Greater) {
                return Ok(false);
            }
        }
        last_key = cells
            .iter()
            .map(|cell| match cell {
                KeyCell::Resident(column) => column.get_value(group_end - 1),
                KeyCell::Group(column, start) => column.get_value(group_end - 1 - start),
                KeyCell::Default(value) => value.clone(),
            })
            .collect();
        group_start = group_end;
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{DataType, Row, SchemaBuilder};
    use crate::storage::volume::io::serialize_v4_public;
    use crate::storage::volume::writer::VolumeBuilder;

    fn schema() -> Schema {
        let mut schema = SchemaBuilder::new("t")
            .column("id", DataType::Integer, false, true)
            .column("k", DataType::Text, true, false)
            .column("n", DataType::Integer, true, false)
            .build();
        schema.set_cluster_key(vec![1, 2]).unwrap();
        schema
    }

    /// A warm input of `keys` in physical order, row ids from `first_id`
    fn warm(schema: &Schema, first_id: i64, keys: &[(&str, i64)]) -> Arc<FrozenVolume> {
        let mut builder = VolumeBuilder::new(schema);
        for (i, (k, n)) in keys.iter().enumerate() {
            builder.add_row(
                first_id + i as i64,
                &Row::from_values(vec![
                    Value::Integer(first_id + i as i64),
                    Value::text(*k),
                    Value::Integer(*n),
                ]),
            );
        }
        let mut volume = builder.finish().unwrap();
        let (_, store) = serialize_v4_public(&volume).unwrap();
        volume.columns.attach_compressed_store(store);
        Arc::new(volume.to_warm().unwrap())
    }

    fn identity() -> ColumnMapping {
        ColumnMapping {
            sources: Vec::new(),
            names: Vec::new(),
            is_identity: true,
        }
    }

    fn refs(inputs: &[(u64, Arc<FrozenVolume>)]) -> Vec<(i64, usize, usize)> {
        let mut refs = Vec::new();
        for (input, (_, volume)) in inputs.iter().enumerate() {
            for (row, &id) in volume.row_ids().unwrap().iter().enumerate() {
                refs.push((id, input, row));
            }
        }
        refs
    }

    /// The order the general sort gives: key values, then row id
    fn sorted(inputs: &[(u64, Arc<FrozenVolume>)], refs: &[(i64, usize, usize)]) -> Vec<i64> {
        let mut rows: Vec<(i64, Row)> = refs
            .iter()
            .map(|&(id, input, row)| (id, inputs[input].1.get_row(row).unwrap()))
            .collect();
        rows.sort_by(|a, b| super::super::seal::cluster_order(&schema(), a, b));
        rows.into_iter().map(|(id, _)| id).collect()
    }

    #[test]
    fn key_ordered_inputs_merge_by_key_then_row_id() {
        let schema = schema();
        let a = warm(
            &schema,
            100,
            &[("a", 1), ("a", 3), ("b", 1), ("c", 5), ("c", 5)],
        );
        let b = warm(
            &schema,
            10,
            &[("a", 2), ("a", 3), ("b", 0), ("c", 5), ("d", 1)],
        );
        let inputs = vec![(1u64, a), (2u64, b)];
        let refs = refs(&inputs);
        let merged = merge_key_order(&schema, &inputs, &[identity(), identity()], &refs)
            .unwrap()
            .expect("in order");
        let ids: Vec<i64> = merged.refs.iter().map(|r| r.0).collect();
        assert_eq!(ids, sorted(&inputs, &refs));
        // Equal keys across the inputs: (c, 5) at ids 13, 103, 104
        assert_eq!(&ids[6..9], &[13, 103, 104]);
        assert!(merged.peak_held_bytes > 0);
        // The key columns stay unloaded in the inputs
        for (_, volume) in &inputs {
            assert!(volume.columns.resident(1).is_none());
            assert!(volume.columns.resident(2).is_none());
        }
    }

    #[test]
    fn an_input_out_of_key_order_is_no_merge() {
        let schema = schema();
        let a = warm(&schema, 100, &[("a", 1), ("b", 1)]);
        let b = warm(&schema, 10, &[("b", 2), ("a", 9)]);
        let inputs = vec![(1u64, a), (2u64, b)];
        let refs = refs(&inputs);
        assert!(
            merge_key_order(&schema, &inputs, &[identity(), identity()], &refs)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn equal_keys_with_row_ids_descending_is_no_merge() {
        let schema = schema();
        let a = warm(&schema, 100, &[("a", 1), ("a", 1)]);
        let mut builder = VolumeBuilder::new(&schema);
        builder.allow_any_row_order();
        for id in [20i64, 19] {
            builder.add_row(
                id,
                &Row::from_values(vec![
                    Value::Integer(id),
                    Value::text("b"),
                    Value::Integer(2),
                ]),
            );
        }
        let b = Arc::new(builder.finish().unwrap());
        let inputs = vec![(1u64, a), (2u64, b)];
        let refs = refs(&inputs);
        assert!(
            merge_key_order(&schema, &inputs, &[identity(), identity()], &refs)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn a_column_an_input_predates_orders_by_its_default() {
        let schema = schema();
        let old_schema = SchemaBuilder::new("t")
            .column("id", DataType::Integer, false, true)
            .column("k", DataType::Text, true, false)
            .build();
        let mut builder = VolumeBuilder::new(&old_schema);
        for (i, k) in ["a", "c"].iter().enumerate() {
            builder.add_row(
                10 + i as i64,
                &Row::from_values(vec![Value::Integer(10 + i as i64), Value::text(*k)]),
            );
        }
        let old = Arc::new(builder.finish().unwrap());
        let new = warm(&schema, 100, &[("a", 1), ("c", 4), ("c", 9)]);
        let inputs = vec![(1u64, old), (2u64, new)];
        let refs = refs(&inputs);
        let mapping = ColumnMapping {
            sources: vec![
                ColSource::Volume(0),
                ColSource::Volume(1),
                ColSource::Default(Value::Integer(5)),
            ],
            names: Vec::new(),
            is_identity: false,
        };
        let merged = merge_key_order(&schema, &inputs, &[mapping, identity()], &refs)
            .unwrap()
            .expect("in order");
        let ids: Vec<i64> = merged.refs.iter().map(|r| r.0).collect();
        // (a,1)=100, (a,5)=10, (c,4)=101, (c,5)=11, (c,9)=102
        assert_eq!(ids, vec![100, 10, 101, 11, 102]);
    }

    #[test]
    fn many_inputs_merge_in_order_with_one_heap_step_per_row() {
        let schema = schema();
        let mut inputs = Vec::new();
        for input in 0..128i64 {
            let keys: Vec<(String, i64)> = (0..64)
                .map(|i| (format!("k{:03}", (i * 128 + input) % 200), input))
                .collect();
            let mut sorted = keys.clone();
            sorted.sort();
            let borrowed: Vec<(&str, i64)> = sorted.iter().map(|(k, n)| (k.as_str(), *n)).collect();
            inputs.push((input as u64 + 1, warm(&schema, input * 1000, &borrowed)));
        }
        let refs = refs(&inputs);
        let mappings: Vec<ColumnMapping> = (0..128).map(|_| identity()).collect();
        let merged = merge_key_order(&schema, &inputs, &mappings, &refs)
            .unwrap()
            .expect("in order");
        let ids: Vec<i64> = merged.refs.iter().map(|r| r.0).collect();
        assert_eq!(ids, sorted(&inputs, &refs));
    }

    #[test]
    fn the_peak_counts_the_groups_loaded_before_the_first_row() {
        let schema = schema();
        let a = warm(&schema, 1, &[("a", 1)]);
        let inputs = vec![(1u64, a)];
        let refs = refs(&inputs);
        let merged = merge_key_order(&schema, &inputs, &[identity()], &refs)
            .unwrap()
            .expect("in order");
        assert_eq!(merged.refs.len(), 1);
        assert!(merged.peak_held_bytes > 0);
    }

    #[test]
    fn the_order_check_reads_a_group_at_a_time_and_leaves_the_columns_unloaded() {
        let schema = schema();
        let rows = ROW_GROUP_SIZE + 5;
        let keys: Vec<(String, i64)> = (0..rows).map(|i| (format!("k{:07}", i / 2), 0)).collect();
        let borrowed: Vec<(&str, i64)> = keys.iter().map(|(k, n)| (k.as_str(), *n)).collect();
        let ordered = warm(&schema, 0, &borrowed);
        assert!(ordered.in_key_order(&[1, 2]).unwrap());
        assert!(ordered.columns.resident(1).is_none());
        assert!(ordered.columns.resident(2).is_none());
        // An inversion inside a group, and one across the group boundary
        let mut inside = keys.clone();
        inside.swap(10, 11);
        inside[10].0 = "k9999999".to_string();
        let borrowed: Vec<(&str, i64)> = inside.iter().map(|(k, n)| (k.as_str(), *n)).collect();
        assert!(!warm(&schema, 0, &borrowed).in_key_order(&[1, 2]).unwrap());
        let mut across = keys.clone();
        across[ROW_GROUP_SIZE].0 = "k0000000".to_string();
        let borrowed: Vec<(&str, i64)> = across.iter().map(|(k, n)| (k.as_str(), *n)).collect();
        assert!(!warm(&schema, 0, &borrowed).in_key_order(&[1, 2]).unwrap());
    }

    #[test]
    fn a_run_crossing_a_group_boundary_keeps_its_order() {
        let schema = schema();
        let rows = ROW_GROUP_SIZE + 10;
        let keys: Vec<(String, i64)> = (0..rows).map(|i| (format!("k{:07}", i / 3), 0)).collect();
        let borrowed: Vec<(&str, i64)> = keys.iter().map(|(k, n)| (k.as_str(), *n)).collect();
        let a = warm(&schema, 0, &borrowed);
        let b = warm(&schema, 1_000_000, &[("k0000000", 0), ("k0021850", 0)]);
        let inputs = vec![(1u64, a), (2u64, b)];
        let refs = refs(&inputs);
        let merged = merge_key_order(&schema, &inputs, &[identity(), identity()], &refs)
            .unwrap()
            .expect("in order");
        assert_eq!(merged.refs.len(), rows + 2);
        assert_eq!(merged.refs[0].0, 0);
        assert_eq!(merged.refs[3].0, 1_000_000);
        assert_eq!(merged.refs[merged.refs.len() - 1].0, 1_000_001);
        // One group of the two key columns of the large input at a time
        let one_group = inputs[0]
            .1
            .columns
            .compressed_store()
            .unwrap()
            .group_column(1, 0)
            .unwrap()
            .cache_size()
            + inputs[0]
                .1
                .columns
                .compressed_store()
                .unwrap()
                .group_column(2, 0)
                .unwrap()
                .cache_size();
        assert!(
            merged.peak_held_bytes <= one_group * 2,
            "{} held",
            merged.peak_held_bytes
        );
    }
}
