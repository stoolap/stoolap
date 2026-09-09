// Copyright 2026 Stoolap Contributors
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

//! Lazy cold iteration over one immutable table generation.

use std::sync::Arc;

use rustc_hash::FxHashSet;

use crate::core::{DataType, Error, Operator, Result, Row, Schema, Value};
use crate::storage::expression::Expression;
use crate::storage::mvcc::version_store::{CapturedHotRow, CapturedHotView};
use crate::storage::traits::Scanner;

use super::manifest::{ColdGeneration, ColdSegment};
use super::scanner::VolumeScanner;

type MetadataComparisons = smallvec::SmallVec<[(usize, Operator, Value, Option<u64>); 4]>;

/// Only the current volume is loaded. Metadata owners survive compaction and
/// eviction; advancing never consults the mutable segment manager.
pub(crate) struct CapturedColdScanner {
    generation: Option<Arc<ColdGeneration>>,
    view: Option<Arc<CapturedHotView>>,
    pending: Option<Arc<FxHashSet<i64>>>,
    columns: Vec<usize>,
    filter: Option<Box<dyn Expression>>,
    comparisons: MetadataComparisons,
    // Only a single exact equality on the current INTEGER primary key.
    pk_lookup: Option<(usize, i64)>,
    remaining: usize,
    active: Option<VolumeScanner>,
    error: Option<Error>,
    empty: Row,
}

impl CapturedColdScanner {
    pub(crate) fn new(
        generation: Arc<ColdGeneration>,
        view: Arc<CapturedHotView>,
        pending: Arc<FxHashSet<i64>>,
        columns: Vec<usize>,
        filter: Option<Box<dyn Expression>>,
        schema: &Schema,
    ) -> Self {
        let remaining = generation.segment_ids_newest_first.len();
        let pk_lookup = filter.as_ref().and_then(|filter| {
            let (name, Operator::Eq, Value::Integer(id)) = filter.get_comparison_info()? else {
                return None;
            };
            let mut primary = schema
                .columns
                .iter()
                .enumerate()
                .filter(|(_, c)| c.primary_key);
            let (index, column) = primary.next()?;
            (primary.next().is_none()
                && column.data_type == DataType::Integer
                && name.eq_ignore_ascii_case(&column.name))
            .then_some((index, *id))
        });
        let comparisons = filter
            .as_ref()
            .map(|filter| {
                filter
                    .collect_comparisons()
                    .into_iter()
                    .filter_map(|(column, operator, value)| {
                        let column = *schema.column_index_map().get(&column.to_lowercase())?;
                        let hash = (operator == Operator::Eq)
                            .then(|| super::column::ColumnBloomFilter::hash_value_static(value));
                        Some((column, operator, value.clone(), hash))
                    })
                    .collect()
            })
            .unwrap_or_default();
        Self {
            generation: Some(generation),
            view: Some(view),
            pending: Some(pending),
            columns,
            filter,
            comparisons,
            pk_lookup,
            remaining,
            active: None,
            error: None,
            empty: Row::new(),
        }
    }

    fn pruned(&self, segment: &ColdSegment) -> bool {
        self.comparisons
            .iter()
            .any(|(logical, operator, value, hash)| {
                let Some(super::writer::ColSource::Volume(physical)) =
                    segment.mapping.sources.get(*logical)
                else {
                    return false;
                };
                let Some(zone) = segment.volume.meta.zone_maps.get(*physical) else {
                    return false;
                };
                match operator {
                    Operator::Eq => {
                        !zone.may_contain_eq(value)
                            || (segment.volume.meta.column_types.get(*physical).is_some_and(
                                |&data_type| {
                                    super::column::ColumnBloomFilter::equality_hash_compatible(
                                        data_type, value,
                                    )
                                },
                            ) && hash.is_some_and(|hash| {
                                segment
                                    .volume
                                    .meta
                                    .bloom_filters
                                    .get(*physical)
                                    .is_some_and(|bloom| !bloom.might_contain_hash(hash))
                            }))
                    }
                    Operator::Gt | Operator::Gte => !zone.may_contain_gte(value),
                    Operator::Lt | Operator::Lte => !zone.may_contain_lte(value),
                    _ => false,
                }
            })
    }

    /// A metadata-only pass prevents reloading a completely hidden cold file.
    /// One group of authority bits bounds scratch independently of table size.
    pub(crate) fn visible_count(
        segment: &ColdSegment,
        generation: &ColdGeneration,
        view: &CapturedHotView,
        pending: &FxHashSet<i64>,
        stop_after_first: bool,
    ) -> usize {
        if view.is_empty()
            && pending.is_empty()
            && generation.tombstones.is_empty()
            && segment.visible.is_none()
        {
            return if stop_after_first {
                usize::from(segment.volume.meta.row_count != 0)
            } else {
                segment.volume.meta.row_count
            };
        }
        let mut count = 0;
        Self::for_each_visible(segment, generation, view, pending, |_, _| {
            count += 1;
            !stop_after_first
        });
        count
    }

    pub(crate) fn for_each_visible(
        segment: &ColdSegment,
        generation: &ColdGeneration,
        view: &CapturedHotView,
        pending: &FxHashSet<i64>,
        mut visit: impl FnMut(usize, i64) -> bool,
    ) -> bool {
        const GROUP: usize = super::column::ROW_GROUP_SIZE;
        let mut bits = [0u64; GROUP.div_ceil(64)];
        for (group, ids) in segment.volume.meta.row_ids.chunks(GROUP).enumerate() {
            view.mark_authoritative(ids, &mut bits);
            for (local, &id) in ids.iter().enumerate() {
                if bits[local / 64] & (1 << (local % 64)) != 0
                    || !segment.is_visible(group * GROUP + local)
                    || pending.contains(&id)
                    || generation.tombstones.get(&id).is_some_and(|&sequence| {
                        i64::try_from(sequence)
                            .is_ok_and(|sequence| view.epoch().admits_commit_sequence(sequence))
                    })
                {
                    continue;
                }
                if !visit(group * GROUP + local, id) {
                    return false;
                }
            }
        }
        true
    }

    fn advance(&mut self) -> Result<bool> {
        let (Some(generation), Some(view), Some(pending)) =
            (&self.generation, &self.view, &self.pending)
        else {
            return Ok(false);
        };
        while self.remaining != 0 {
            self.remaining -= 1;
            let id = generation.segment_ids_newest_first[self.remaining];
            let segment = generation.segments.get(&id).ok_or_else(|| {
                Error::internal(format!("captured generation has no segment {id}"))
            })?;
            // Row IDs are the physical identity of a single INTEGER PK.
            // A default or type-changed PK mapping cannot use that proof.
            let point = self.pk_lookup.and_then(|(logical, row_id)| {
                let super::writer::ColSource::Volume(physical) =
                    segment.mapping.sources.get(logical)?
                else {
                    return None;
                };
                (segment.volume.meta.column_types.get(*physical) == Some(&DataType::Integer))
                    .then(|| segment.volume.meta.row_ids.binary_search(&row_id))
            });
            let range = match point {
                Some(Ok(index)) => {
                    let row_id = segment.volume.meta.row_ids[index];
                    // Resolve authority before opening a captured cold file,
                    // even when unrelated rows in that file remain visible.
                    if !matches!(view.row_state(row_id), CapturedHotRow::NoVisibleVersion)
                        || !segment.is_visible(index)
                        || pending.contains(&row_id)
                        || generation.tombstones.get(&row_id).is_some_and(|&sequence| {
                            i64::try_from(sequence)
                                .is_ok_and(|sequence| view.epoch().admits_commit_sequence(sequence))
                        })
                    {
                        continue;
                    }
                    Some((index, index + 1))
                }
                Some(Err(_)) => continue,
                None => None,
            };
            if self.pruned(segment) {
                continue;
            }
            if range.is_none()
                && segment.volume.is_cold()
                && Self::visible_count(segment, generation, view, pending, true) == 0
            {
                continue;
            }
            let loaded = generation.load_segment(id)?;
            let mut scanner = if let Some((start, end)) = range {
                VolumeScanner::with_range(loaded.volume, self.columns.clone(), start, end, None)
            } else {
                VolumeScanner::new(loaded.volume, self.columns.clone(), None)
            };
            scanner.set_skip_sets(generation.tombstones.clone(), pending.clone());
            scanner.set_visibility_bitmap(loaded.visible);
            scanner.set_column_mapping(loaded.mapping);
            scanner.set_captured_visibility(view.clone(), pending.clone());
            if let Some(filter) = &self.filter {
                scanner.set_filter(filter.clone_box());
            }
            self.active = Some(scanner);
            return Ok(true);
        }
        Ok(false)
    }
}

impl Scanner for CapturedColdScanner {
    fn next(&mut self) -> bool {
        if self.error.is_some() {
            return false;
        }
        loop {
            if let Some(active) = &mut self.active {
                if active.next() {
                    return true;
                }
                if active.err().is_some() {
                    return false;
                }
            }
            self.active = None;
            match self.advance() {
                Ok(true) => (),
                Ok(false) => {
                    let _ = self.close();
                    return false;
                }
                Err(error) => {
                    self.error = Some(error);
                    return false;
                }
            }
        }
    }
    fn row(&self) -> &Row {
        self.active.as_ref().map_or(&self.empty, Scanner::row)
    }
    fn current_row_id(&self) -> i64 {
        self.active.as_ref().map_or(0, Scanner::current_row_id)
    }
    fn take_row(&mut self) -> Row {
        self.active
            .as_mut()
            .map_or_else(Row::new, Scanner::take_row)
    }
    fn take_row_with_id(&mut self) -> (i64, Row) {
        (self.current_row_id(), self.take_row())
    }
    fn err(&self) -> Option<&Error> {
        self.error
            .as_ref()
            .or_else(|| self.active.as_ref().and_then(Scanner::err))
    }
    fn close(&mut self) -> Result<()> {
        if let Some(active) = &mut self.active {
            if self.error.is_none() {
                self.error = active.err().cloned();
            }
            active.close()?;
        }
        self.active = None;
        self.generation = None;
        self.view = None;
        self.pending = None;
        self.filter = None;
        self.comparisons = smallvec::SmallVec::new();
        self.pk_lookup = None;
        self.columns = Vec::new();
        self.remaining = 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::SchemaBuilder;
    use crate::storage::expression::ComparisonExpr;
    use crate::storage::mvcc::version_store::{RowVersion, TransactionVersionStore, VersionStore};
    use crate::storage::mvcc::{MVCCTable, TransactionRegistry};
    use crate::storage::traits::Table;
    use crate::storage::volume::manifest::{SegmentManager, SegmentMeta};
    use crate::storage::volume::table::SegmentedTable;
    use crate::storage::volume::writer::{ColSource, VolumeBuilder};

    fn row(id: i64, n: i64) -> Row {
        Row::from_values(vec![Value::Integer(id), Value::Integer(n)])
    }
    fn fixture() -> (
        Schema,
        Arc<TransactionRegistry>,
        Arc<VersionStore>,
        Arc<SegmentManager>,
    ) {
        let schema = SchemaBuilder::new("point")
            .add_primary_key("id", DataType::Integer)
            .add("n", DataType::Integer)
            .build();
        let registry = Arc::new(TransactionRegistry::new());
        let store = Arc::new(VersionStore::with_visibility_checker(
            "point",
            schema.clone(),
            registry.clone(),
        ));
        let manager = Arc::new(SegmentManager::new("point", None));
        let mut builder = VolumeBuilder::new(&schema);
        for id in 1..=6 {
            builder.add_row(id, &row(id, id * 10));
        }
        manager.register_segment(
            1,
            Arc::new(builder.finish()),
            SegmentMeta {
                segment_id: 1,
                file_path: "unopened-point.vol".into(),
                row_count: 6,
                min_row_id: 1,
                max_row_id: 6,
                schema_version: 0,
                creation_lsn: 0,
                seal_seq: 0,
            },
            Some(&schema),
        );
        (schema, registry, store, manager)
    }
    fn collect(scanner: &mut dyn Scanner) -> Result<Vec<(i64, Row)>> {
        let mut rows = Vec::new();
        while scanner.next() {
            rows.push(scanner.take_row_with_id());
        }
        if let Some(error) = scanner.err() {
            return Err(error.clone());
        }
        scanner.close()?;
        Ok(rows)
    }

    #[test]
    fn captured_pk_range_keeps_own_authority_and_frozen_sources() {
        let (_, registry, store, manager) = fixture();
        let (inflight, _) = registry.begin_transaction();
        registry.start_commit(inflight);
        store
            .add_version(6, RowVersion::new(inflight, row(6, 600)))
            .unwrap();
        let (reader, _) = registry.begin_transaction();
        let mut local = TransactionVersionStore::new(store.clone(), reader);
        local.put(2, row(2, 200), false).unwrap();
        local.put(3, row(3, 30), true).unwrap();
        manager.add_pending_tombstone(reader, 4);
        let hot = MVCCTable::new(reader, store.clone(), local);
        let mut table = SegmentedTable::new(Box::new(hot), manager.clone());
        table.set_read_epoch(registry.capture_read_epoch()).unwrap();
        registry.complete_commit(inflight);
        for (id, expected) in [
            (1, Some(10)),
            (2, Some(200)),
            (3, None),
            (4, None),
            (6, Some(60)),
            (99, None),
        ] {
            let filter = ComparisonExpr::eq("id", Value::Integer(id));
            let mut scanner = table.scan(&[1], Some(&filter)).unwrap();
            assert_eq!(
                collect(&mut *scanner).unwrap(),
                expected
                    .into_iter()
                    .map(|n| (id, Row::from_values(vec![Value::Integer(n)])))
                    .collect::<Vec<_>>()
            );
        }
        let filter = ComparisonExpr::eq("id", Value::Integer(5));
        let mut scanner = table.scan(&[1], Some(&filter)).unwrap();
        let (newer, _) = registry.begin_transaction();
        registry.start_commit(newer);
        store
            .add_version(5, RowVersion::new(newer, row(5, 500)))
            .unwrap();
        registry.complete_commit(newer);
        manager.rollback_pending_tombstones(reader);
        manager.clear();
        assert_eq!(
            collect(&mut *scanner).unwrap(),
            vec![(5, Row::from_values(vec![Value::Integer(50)]))]
        );
    }

    #[test]
    fn captured_pk_range_resolves_hidden_or_absent_before_cold_loading() {
        let (schema, registry, store, manager) = fixture();
        let (writer, _) = registry.begin_transaction();
        registry.start_commit(writer);
        store
            .add_version(1, RowVersion::new(writer, row(1, 100)))
            .unwrap();
        let mut deleted = RowVersion::new(writer, row(2, 20));
        deleted.deleted_at_txn_id = writer;
        store.add_version(2, deleted).unwrap();
        registry.complete_commit(writer);
        manager.add_tombstones(&[4], registry.get_commit_sequence(writer).unwrap() as u64);
        let (_, mut generation) = manager.capture_with_hot(|| ()).unwrap();
        drop(manager);
        let generation_mut = Arc::get_mut(&mut generation).unwrap();
        let segment = Arc::make_mut(&mut generation_mut.segments)
            .get_mut(&1)
            .unwrap();
        segment.volume = Arc::new(segment.volume.to_cold());
        let view = Arc::new(CapturedHotView::new(
            store.capture_hot_root(),
            registry.capture_read_epoch(),
            None,
        ));
        let pending: Arc<FxHashSet<i64>> = Arc::new([3].into_iter().collect());
        for id in [1, 2, 3, 4, 99, 6] {
            let mut filter = ComparisonExpr::eq("id", Value::Integer(id));
            filter.prepare_for_schema(&schema);
            let mut scanner = CapturedColdScanner::new(
                generation.clone(),
                view.clone(),
                pending.clone(),
                vec![1],
                Some(Box::new(filter)),
                &schema,
            );
            assert_eq!(scanner.pk_lookup, Some((0, id)));
            let result = collect(&mut scanner);
            if id == 6 {
                assert!(result.is_err(), "visible captured file failure propagates");
            } else {
                assert!(
                    result.unwrap().is_empty(),
                    "id {id} loaded a hidden or absent source"
                );
            }
        }
    }

    #[test]
    fn captured_pk_range_rejects_coercions_and_preserves_mapped_defaults() {
        let (schema, registry, store, manager) = fixture();
        let (_, mut generation) = manager.capture_with_hot(|| ()).unwrap();
        let view = Arc::new(CapturedHotView::new(
            store.capture_hot_root(),
            registry.capture_read_epoch(),
            None,
        ));
        for (column, value, expected_id) in
            [("n", Value::Integer(20), 2), ("id", Value::Float(2.0), 2)]
        {
            let mut filter = ComparisonExpr::eq(column, value);
            filter.prepare_for_schema(&schema);
            let mut scanner = CapturedColdScanner::new(
                generation.clone(),
                view.clone(),
                Arc::default(),
                vec![1],
                Some(Box::new(filter)),
                &schema,
            );
            assert!(scanner.pk_lookup.is_none());
            assert_eq!(
                collect(&mut scanner).unwrap(),
                vec![(expected_id, Row::from_values(vec![Value::Integer(20)]))]
            );
        }
        drop(manager);
        let segment = Arc::make_mut(&mut Arc::get_mut(&mut generation).unwrap().segments)
            .get_mut(&1)
            .unwrap();
        segment.mapping.is_identity = false;
        segment.mapping.sources[0] = ColSource::Default(Value::Integer(7));
        segment.mapping.sources[1] = ColSource::Default(Value::Integer(42));
        let mut filter = ComparisonExpr::eq("id", Value::Integer(7));
        filter.prepare_for_schema(&schema);
        let mut scanner = CapturedColdScanner::new(
            generation,
            view,
            Arc::default(),
            vec![1],
            Some(Box::new(filter)),
            &schema,
        );
        assert_eq!(
            collect(&mut scanner).unwrap(),
            (1..=6)
                .map(|id| (id, Row::from_values(vec![Value::Integer(42)])))
                .collect::<Vec<_>>()
        );
    }
}
