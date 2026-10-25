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

//! A change of a column's type or nullability: with the table closed, every
//! layer takes the new definition, or the change is refused with nothing
//! changed

use super::{
    begin_column_change, close_and_settle, populate_hnsw_from_volumes, refresh_schema,
    to_lowercase_cow, DdlShared, PersistenceManager, TransactionRegistry, VersionStore,
    WALOperationType, ALTER_MODIFY_COLUMN,
};
use crate::common::{CompactArc, SmartString};
use crate::core::{DataType, Error, Result, Schema, Value};
use crate::storage::mvcc::version_store::HotCast;
use crate::storage::traits::Index;
use crate::storage::volume::manifest::{ColdSnapshot, SegmentManager, SegmentMeta};
use crate::storage::volume::secondary::IndexFile;
use crate::storage::volume::side_index::{kind_for, Component, Domain, SideTarget};
use crate::storage::volume::writer::{self, ColumnMapping, FrozenVolume};
use rustc_hash::{FxHashMap, FxHashSet};
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::{Arc, RwLock};

/// The definition a column takes
pub struct ColumnTypeChange<'a> {
    pub table: &'a str,
    pub column: &'a str,
    pub data_type: DataType,
    pub nullable: bool,
    pub vector_dimensions: u16,
}

/// What a change reads and writes of the engine, shared by the engine and
/// a transaction's engine operations
pub(super) struct Shared<'a> {
    pub version_stores: &'a RwLock<FxHashMap<String, Arc<VersionStore>>>,
    pub segment_managers: &'a RwLock<FxHashMap<String, Arc<SegmentManager>>>,
    pub persistence: &'a Option<PersistenceManager>,
    pub registry: &'a TransactionRegistry,
    pub schemas: &'a RwLock<FxHashMap<String, CompactArc<Schema>>>,
    pub ddl: &'a DdlShared,
    /// WAL replay applies the change to the hot rows only, with no record:
    /// the volumes are repaired once the log is replayed
    pub replaying: bool,
}

/// Runs `change` with the DDL guard held: the policy, the close, the cast
/// built and validated, the change's log record, then the publication. A
/// refusal before the record leaves the table as it was
pub(super) fn change_column_type(shared: &Shared<'_>, change: &ColumnTypeChange<'_>) -> Result<()> {
    debug_assert!(shared.ddl.serial.is_locked());
    let name = to_lowercase_cow(change.table);
    let store = shared
        .version_stores
        .read()
        .map_err(|_| Error::internal("version stores lock poisoned"))?
        .get(name.as_ref())
        .cloned()
        .ok_or_else(|| Error::TableNotFound(name.to_string()))?;
    let current = store.schema();
    let column = current
        .get_column_index(change.column)
        .ok_or_else(|| Error::ColumnNotFound(change.column.to_string()))?;
    let mut next = Schema::clone(&current);
    next.modify_column(change.column, Some(change.data_type), Some(change.nullable))?;
    next.columns[column].vector_dimensions = change.vector_dimensions;
    // A volume sealed before the column was added reads its default
    let default = next.columns[column].default_value.take();
    next.columns[column].default_value =
        default.map(|value| value.coerce_to_type(change.data_type));
    // A change the log replays was taken when it was made
    if let (false, Some(Value::Timestamp(t))) =
        (shared.replaying, &next.columns[column].default_value)
    {
        crate::core::value::timestamp_in_range(*t)?;
    }
    let next = CompactArc::new(next);
    let retyped = current.columns[column].data_type != change.data_type
        || current.columns[column].vector_dimensions != change.vector_dimensions;
    if retyped {
        if !shared.replaying
            && segment_manager(shared, name.as_ref())?
                .is_some_and(|mgr| mgr.has_unversioned_renames())
        {
            return Err(unversioned_renames(change.table));
        }
        check_policy(shared, &store, &current, column, change)?;
    }

    let mut schema_change =
        begin_column_change(shared.segment_managers, shared.persistence, name.as_ref())?;
    let closed = close_and_settle(
        shared.version_stores,
        shared.segment_managers,
        shared.persistence,
        shared.registry,
        name.as_ref(),
    )?;

    let cast = store.prepare_cast(column, CompactArc::clone(&next), retyped)?;
    let mgr = if shared.replaying {
        None
    } else {
        segment_manager(shared, name.as_ref())?
    };
    let cold = match (&mgr, retyped) {
        (Some(mgr), true) => prepare_cold(shared, &store, mgr, name.as_ref(), &next)?,
        _ => ColdCast::default(),
    };
    #[cfg(feature = "test-failpoints")]
    if !cold.rewrites.is_empty() {
        crate::test_failpoints::retype_outputs_written();
    }
    let checked = validate(
        &store,
        &cast,
        mgr.as_deref(),
        &cold,
        &next,
        column,
        change,
        retyped || !change.nullable,
        !shared.replaying,
    );
    // The new HNSW objects take the sealed rows before anything is decided
    let hnsw: Vec<Arc<dyn Index>> = cast
        .indexes()
        .iter()
        .filter(|(_, index)| index.index_type() == crate::core::IndexType::Hnsw)
        .map(|(_, index)| Arc::clone(index))
        .collect();
    let checked = checked.and_then(|()| match &mgr {
        Some(mgr) if !hnsw.is_empty() => populate_hnsw_from_volumes(&store, mgr, &hnsw),
        _ => Ok(()),
    });
    if let Err(error) = checked {
        cold.discard();
        return Err(error);
    }
    if let Err(error) = record(shared, change) {
        cold.discard();
        if matches!(error, Error::WalOutcomeUnknown { .. }) {
            // The record may replay: the table and its maintenance stay
            // closed until the next open decides
            schema_change.mark_changed();
            closed.keep_closed();
        }
        return Err(error);
    }
    let rewritten = mgr.as_ref().filter(|_| !cold.rewrites.is_empty());
    if let Some(mgr) = rewritten {
        mgr.set_incomplete_change(true);
    }

    store.publish_cast(cast);
    schema_change.mark_changed();
    refresh_schema(
        shared.version_stores,
        shared.schemas,
        shared.ddl,
        name.as_ref(),
    )?;
    if let Some(mgr) = &mgr {
        if let Err(error) = cold.publish(mgr, &next) {
            closed.keep_closed();
            return Err(Error::internal(format!(
                "the change of column '{}' of table '{}' is durable but could not complete: {error}; \
                 the table stays closed until the database reopens",
                change.column, change.table
            )));
        }
        mgr.set_incomplete_change(false);
    }
    drop(closed);
    schema_change.finish();
    Ok(())
}

/// Rewrites, once the log is replayed, the volumes of `table` that hold a
/// column in another type than its schema's (left by a build before the
/// change rewrote them): the table validated in the schema's types, then
/// the manifest written, then the sources retired. A violation fails the
/// open with the table's source files kept
pub(super) fn repair_retyped_volumes(shared: &Shared<'_>, table: &str) -> Result<()> {
    let store = shared
        .version_stores
        .read()
        .map_err(|_| Error::internal("version stores lock poisoned"))?
        .get(table)
        .cloned();
    let (Some(store), Some(mgr)) = (store, segment_manager(shared, table)?) else {
        return Ok(());
    };
    if !mgr.has_segments() {
        return Ok(());
    }
    let schema = store.schema();
    let cold = prepare_cold(shared, &store, &mgr, table, &schema)?;
    if cold.rewrites.is_empty() {
        return Ok(());
    }
    let heads = store.current_heads();
    for &column in &cold.columns {
        let definition = &schema.columns[column];
        let change = ColumnTypeChange {
            table,
            column: &definition.name,
            data_type: definition.data_type,
            nullable: definition.nullable,
            vector_dimensions: definition.vector_dimensions,
        };
        let checked = validate(
            &store,
            &heads,
            Some(&mgr),
            &cold,
            &schema,
            column,
            &change,
            true,
            false,
        );
        if let Err(error) = checked {
            cold.discard();
            return Err(Error::internal(format!(
                "table '{table}' cannot open: its volumes, read in the schema's types, \
                 violate a constraint: {error}"
            )));
        }
    }
    cold.publish(&mgr, &schema)
}

/// The change's log record, synced whatever the sync mode: the change is
/// decided by its record being durable. Replay writes none
fn record(shared: &Shared<'_>, change: &ColumnTypeChange<'_>) -> Result<()> {
    if shared.replaying {
        return Ok(());
    }
    let Some(pm) = shared.persistence.as_ref().filter(|pm| pm.is_enabled()) else {
        return Ok(());
    };
    // operation(1), table name length(2) and name, column name length(2)
    // and name, type(1), the dimension(2) of a vector, nullable(1)
    let mut data = vec![ALTER_MODIFY_COLUMN];
    data.extend_from_slice(&(change.table.len() as u16).to_le_bytes());
    data.extend_from_slice(change.table.as_bytes());
    data.extend_from_slice(&(change.column.len() as u16).to_le_bytes());
    data.extend_from_slice(change.column.as_bytes());
    data.push(change.data_type.as_u8());
    if change.data_type == DataType::Vector {
        data.extend_from_slice(&change.vector_dimensions.to_le_bytes());
    }
    data.push(u8::from(change.nullable));
    pm.record_ddl_operation_synced(change.table, WALOperationType::AlterTable, &data)?;
    Ok(())
}

fn segment_manager(shared: &Shared<'_>, table: &str) -> Result<Option<Arc<SegmentManager>>> {
    Ok(shared
        .segment_managers
        .read()
        .map_err(|_| Error::internal("segment managers lock poisoned"))?
        .get(table)
        .cloned())
}

/// Volumes read through renames that carry no version are not rewritten:
/// which column a name stands for there is not known, and a rewrite would
/// keep a wrong reading for good
fn unversioned_renames(table: &str) -> Error {
    Error::NotSupported(format!(
        "table '{table}' has column renames recorded without a version by a build \
         before 0.4.2, so its volumes cannot be converted to a column's new type \
         safely; recreate the table to change the type"
    ))
}

fn name_dropped_and_reused(table: &str, column: &str) -> Error {
    Error::NotSupported(format!(
        "table '{table}' holds a volume whose column '{column}' turns on a dropped name \
         used again, and a build before 0.4.2 may have recorded that drop in no order \
         against the volume, so the volume cannot be converted to a column's new type \
         safely; recreate the table to change the type"
    ))
}

/// The rules a change of type is checked against before the table closes
fn check_policy(
    shared: &Shared<'_>,
    store: &VersionStore,
    schema: &Schema,
    column: usize,
    change: &ColumnTypeChange<'_>,
) -> Result<()> {
    let refuse = |why: &str| {
        Err(Error::InvalidArgument(format!(
            "cannot change the type of column '{}' of table '{}': {why}",
            change.column, schema.table_name
        )))
    };
    if schema.columns[column].primary_key || schema.pk_column_index() == Some(column) {
        return refuse("it is the primary key");
    }
    if schema
        .foreign_keys
        .iter()
        .any(|fk| fk.column_index == column)
    {
        return refuse("it references another table");
    }
    let table = schema.table_name.to_lowercase();
    let column_name = schema.columns[column].name.to_lowercase();
    let referenced = shared
        .schemas
        .read()
        .map_err(|_| Error::internal("schemas lock poisoned"))?
        .values()
        .any(|other| {
            other
                .foreign_keys
                .iter()
                .any(|fk| fk.referenced_table == table && fk.referenced_column == column_name)
        });
    if referenced {
        return refuse("a foreign key references it");
    }
    for index in store.get_all_indexes() {
        let on_column = index.column_ids().contains(&(column as i32));
        if on_column
            && index.index_type() == crate::core::IndexType::Hnsw
            && (change.data_type != DataType::Vector || change.vector_dimensions == 0)
        {
            return refuse(&format!(
                "HNSW index '{}' holds only vectors of a known dimension",
                index.name()
            ));
        }
    }
    Ok(())
}

/// A volume rewritten in the new type, written and synced under its final
/// name, not yet registered
struct Rewrite {
    source: u64,
    source_volume: Arc<FrozenVolume>,
    id: u64,
    volume: Arc<FrozenVolume>,
    meta: SegmentMeta,
    side: Option<Arc<IndexFile>>,
    path: PathBuf,
}

/// The volumes of a table that hold the changed column in another type,
/// each rewritten aside, with the history their mappings are read through
#[derive(Default)]
struct ColdCast {
    rewrites: Vec<Rewrite>,
    /// The schema positions some volume held in another type
    columns: Vec<usize>,
    dropped: Vec<(SmartString, u64)>,
    renames: Vec<(SmartString, SmartString)>,
    schema_version: u64,
}

impl ColdCast {
    /// Removes the outputs of a refused change; nothing registered them
    fn discard(self) {
        for rewrite in self.rewrites {
            drop(rewrite.side);
            let _ = std::fs::remove_file(&rewrite.path);
            crate::storage::volume::secondary::retire_side_of(&rewrite.path);
        }
    }

    /// Each output takes its source's place in the segment order, the
    /// manifest is written, and the sources go once their last holder lets
    /// go
    fn publish(self, mgr: &SegmentManager, schema: &Schema) -> Result<()> {
        if self.rewrites.is_empty() {
            return mgr.invalidate_mappings(schema);
        }
        let mut retired = Vec::with_capacity(self.rewrites.len());
        for rewrite in self.rewrites {
            mgr.replace_segments_atomic_multi(
                vec![(rewrite.id, rewrite.volume, rewrite.meta)],
                &[rewrite.source],
                Some(schema),
                vec![rewrite.side],
            )?;
            retired.push((rewrite.source, rewrite.source_volume, rewrite.path));
        }
        mgr.invalidate_mappings(schema)?;
        mgr.persist_manifest_only()?;
        #[cfg(feature = "test-failpoints")]
        crate::test_failpoints::retype_manifest_written();
        for (source, volume, output) in retired {
            let Some(dir) = output.parent() else {
                continue;
            };
            let path = dir.join(format!("vol_{:016x}.vol", source));
            if !volume.retire_file() {
                crate::storage::volume::writer::VolumeFile::shared(&path).retire();
            }
            crate::storage::volume::secondary::SideFiles::in_dir(dir).retire_for(&path);
        }
        Ok(())
    }

    /// The output that replaces `source`, if the change rewrote it
    fn output_of(&self, source: u64) -> Option<&Rewrite> {
        self.rewrites
            .iter()
            .find(|rewrite| rewrite.source == source)
    }
}

/// Rewrites, aside, every volume of `mgr` that holds a column of `schema` in
/// another type
fn prepare_cold(
    shared: &Shared<'_>,
    store: &VersionStore,
    mgr: &SegmentManager,
    table: &str,
    schema: &Schema,
) -> Result<ColdCast> {
    let Some(pm) = shared.persistence.as_ref() else {
        return Ok(ColdCast::default());
    };
    let dir = pm.path().join("volumes");
    let (dropped, renames, metas) = {
        let manifest = mgr.manifest();
        (
            manifest.dropped_columns.clone(),
            manifest.column_renames.clone(),
            manifest.segments.clone(),
        )
    };
    let mut cold = ColdCast {
        rewrites: Vec::new(),
        columns: Vec::new(),
        dropped,
        renames,
        schema_version: shared.ddl.schema_epoch.load(Ordering::Acquire),
    };
    let unique: Vec<Vec<usize>> = store
        .get_unique_non_pk_index_columns()
        .into_iter()
        .map(|(columns, _)| columns)
        .collect();
    // Taken under the old schema: an index whose column the change takes to
    // a type no side file holds covers no output
    let identities: Vec<SideTarget> = store
        .side_index_defs()
        .iter()
        .filter_map(|def| {
            // The rewritten volume holds each column in its new type
            let components: smallvec::SmallVec<[Component; 4]> = def
                .components
                .iter()
                .map(|c| {
                    let domain = Domain::of(schema.columns[c.column as usize].data_type)?;
                    Some(Component { domain, ..*c })
                })
                .collect::<Option<_>>()?;
            Some(SideTarget {
                identity: def.identity,
                kind: kind_for(def.index_type, &components)?,
                components: components
                    .iter()
                    .map(|c| (c.column as usize, c.domain))
                    .collect(),
            })
        })
        .collect();
    let unversioned = mgr.has_unversioned_renames();
    let snapshot = mgr.cold_snapshot();
    for &seg_id in &snapshot.seg_ids {
        let Some(segment) = snapshot.segs.get(&seg_id) else {
            continue;
        };
        let rewritten = (|| -> Result<Option<(Vec<usize>, Rewrite)>> {
            #[cfg(feature = "test-failpoints")]
            if crate::test_failpoints::retype_read_fails() {
                return Err(Error::internal("failpoint: volume read failed"));
            }
            let Some(source) = mgr.ensure_volume(seg_id)? else {
                return Ok(None);
            };
            let retyped = writer::retyped_columns(
                schema,
                &source,
                &cold.dropped,
                segment.schema_version,
                &cold.renames,
            )?;
            if retyped.is_empty() {
                return Ok(None);
            }
            if unversioned {
                return Err(unversioned_renames(table));
            }
            if let Some(column) = writer::reading_turns_on_a_drop(
                schema,
                &source,
                &cold.dropped,
                segment.schema_version,
                &cold.renames,
            ) {
                return Err(name_dropped_and_reused(table, column));
            }
            let rewrite = rewrite_volume(
                &dir,
                table,
                schema,
                (seg_id, &source),
                |i| segment.is_dead(i),
                &cold,
                segment.schema_version,
                metas.iter().find(|meta| meta.segment_id == seg_id),
                &unique,
                &identities,
                pm.volume_compression(),
            )?;
            Ok(Some((retyped, rewrite)))
        })();
        match rewritten {
            Ok(Some((retyped, rewrite))) => {
                for column in retyped {
                    if !cold.columns.contains(&column) {
                        cold.columns.push(column);
                    }
                }
                cold.rewrites.push(rewrite);
            }
            Ok(None) => {}
            Err(error) => {
                cold.discard();
                return Err(error);
            }
        }
    }
    Ok(cold)
}

/// One volume rewritten through compaction's transfer, every copy that has
/// authority kept in its physical order, the column cast to `schema`
#[allow(clippy::too_many_arguments)]
fn rewrite_volume(
    dir: &std::path::Path,
    table: &str,
    schema: &Schema,
    (seg_id, source): (u64, &Arc<FrozenVolume>),
    is_dead: impl Fn(usize) -> bool,
    cold: &ColdCast,
    source_schema_version: u64,
    source_meta: Option<&SegmentMeta>,
    unique: &[Vec<usize>],
    identities: &[SideTarget],
    compress: bool,
) -> Result<Rewrite> {
    let mapping = writer::compute_rewrite_mapping(
        schema,
        source,
        &cold.dropped,
        source_schema_version,
        &cold.renames,
    )?;
    let refs: Vec<(i64, usize, usize)> = source
        .row_ids()?
        .iter()
        .enumerate()
        .filter(|(i, _)| !is_dead(*i))
        .map(|(i, &row_id)| (row_id, 0, i))
        .collect();
    let inputs = [(seg_id, Arc::clone(source))];
    let mut transfer =
        crate::storage::volume::transfer::Transfer::new(schema, &inputs, &[mapping], &refs)?;
    let id = crate::storage::volume::io::next_volume_id();
    let mut output = crate::storage::volume::output::VolumeFileWriter::new(
        dir,
        table,
        id,
        schema,
        refs.len(),
        compress,
    )?;
    output.allow_any_row_order();
    output.index_unique_sets(unique.to_vec());
    transfer.begin_output();
    for batch in refs.chunks(crate::storage::volume::transfer::TRANSFER_BATCH_ROWS) {
        transfer.append(batch, &mut output)?;
    }
    let (volume, path) = output.finish()?;
    let side =
        match crate::storage::volume::secondary::build_side_for(&volume, &path, id, identities) {
            Ok(side) => side,
            Err(error) => {
                let _ = std::fs::remove_file(&path);
                return Err(error.into());
            }
        };
    let (min_row_id, max_row_id) = volume.id_bounds().unwrap_or((0, 0));
    let meta = SegmentMeta {
        segment_id: id,
        file_path: PathBuf::new(),
        row_count: refs.len(),
        min_row_id,
        max_row_id,
        creation_lsn: source_meta.map_or(0, |meta| meta.creation_lsn),
        seal_seq: source_meta.map_or(0, |meta| meta.seal_seq),
        schema_version: cold.schema_version,
    };
    Ok(Rewrite {
        source: seg_id,
        source_volume: Arc::clone(source),
        id,
        volume: Arc::new(volume),
        meta,
        side,
        path,
    })
}

/// NOT NULL, the vector dimension, the stored TIMESTAMP range when
/// `storable`, and every unique index over the column, over the current
/// committed view in the new type: each live row once, its newest copy (the
/// hot store first, then the volumes newest first, read through rewrites)
#[allow(clippy::too_many_arguments)]
fn validate(
    store: &VersionStore,
    cast: &HotCast,
    mgr: Option<&SegmentManager>,
    cold: &ColdCast,
    schema: &Schema,
    column: usize,
    change: &ColumnTypeChange<'_>,
    sealed_too: bool,
    storable: bool,
) -> Result<()> {
    let mut unique = UniqueKeys::new(store, column);
    for (row_id, row) in cast.live_rows() {
        validate_value(row.get(column), change, storable)?;
        unique.insert(row_id, |id| row.get(id).cloned())?;
    }
    let Some(mgr) = mgr.filter(|_| sealed_too) else {
        return Ok(());
    };
    let snapshot = mgr.cold_snapshot();
    let (dropped, renames) = {
        let manifest = mgr.manifest();
        (
            manifest.dropped_columns.clone(),
            manifest.column_renames.clone(),
        )
    };
    let mut wanted = unique.columns();
    if !wanted.contains(&column) {
        wanted.push(column);
    }
    let mut seen = FxHashSet::default();
    for &seg_id in snapshot.seg_ids.iter().rev() {
        let Some((volume, mapping, dead)) =
            sealed_input(mgr, &snapshot, cold, seg_id, schema, &dropped, &renames)?
        else {
            continue;
        };
        for (i, &row_id) in volume.row_ids()?.iter().enumerate() {
            if dead.binary_search(&(i as u32)).is_ok()
                || cast.holds(row_id)
                || snapshot.ts.contains_key(&row_id)
                || !seen.insert(row_id)
            {
                continue;
            }
            let row = volume.get_row_mapped_projected(i, &mapping, &wanted)?;
            let value = |id: usize| {
                wanted
                    .iter()
                    .position(|&w| w == id)
                    .and_then(|at| row.get(at).cloned())
            };
            validate_value(value(column).as_ref(), change, storable)?;
            unique.insert(row_id, value)?;
        }
    }
    Ok(())
}

/// A volume as the change will leave it: its rewrite, read whole, or the
/// volume itself read through its mapping, with its copies without authority
type SealedInput = (Arc<FrozenVolume>, ColumnMapping, Arc<[u32]>);

fn sealed_input(
    mgr: &SegmentManager,
    snapshot: &ColdSnapshot,
    cold: &ColdCast,
    seg_id: u64,
    schema: &Schema,
    dropped: &[(SmartString, u64)],
    renames: &[(SmartString, SmartString)],
) -> Result<Option<SealedInput>> {
    if let Some(rewrite) = cold.output_of(seg_id) {
        let mapping = writer::compute_column_mapping_with_drops(
            schema,
            &rewrite.volume,
            dropped,
            rewrite.meta.schema_version,
            renames,
        )?;
        return Ok(Some((Arc::clone(&rewrite.volume), mapping, Arc::from([]))));
    }
    let Some(segment) = snapshot.segs.get(&seg_id) else {
        return Ok(None);
    };
    let Some(volume) = mgr.ensure_volume(seg_id)? else {
        return Ok(None);
    };
    let mapping = writer::compute_column_mapping_with_drops(
        schema,
        &volume,
        dropped,
        segment.schema_version,
        renames,
    )?;
    let dead = segment.dead.clone().unwrap_or_else(|| Arc::from([]));
    Ok(Some((volume, mapping, dead)))
}

/// A unique index's name, its columns, and each key seen with the row
/// holding it
type SeenKeys = (String, Vec<usize>, FxHashMap<Vec<Value>, i64>);

/// The keys of each unique index over the changed column, a key seen twice
/// refusing the change
struct UniqueKeys {
    indexes: Vec<SeenKeys>,
}

impl UniqueKeys {
    fn new(store: &VersionStore, column: usize) -> Self {
        let indexes = store
            .get_all_indexes()
            .into_iter()
            .filter(|index| {
                index.is_unique()
                    && index.index_type() != crate::core::IndexType::PrimaryKey
                    && index.column_ids().contains(&(column as i32))
            })
            .map(|index| {
                let columns = index.column_ids().iter().map(|&id| id as usize).collect();
                (index.name().to_string(), columns, FxHashMap::default())
            })
            .collect();
        Self { indexes }
    }

    fn columns(&self) -> Vec<usize> {
        let mut columns: Vec<usize> = self
            .indexes
            .iter()
            .flat_map(|(_, columns, _)| columns.iter().copied())
            .collect();
        columns.sort_unstable();
        columns.dedup();
        columns
    }

    /// Takes row `row_id`'s key of every index, `value` reading a column; a
    /// key with a NULL holds nothing
    fn insert(&mut self, row_id: i64, value: impl Fn(usize) -> Option<Value>) -> Result<()> {
        for (name, columns, keys) in &mut self.indexes {
            let key: Option<Vec<Value>> = columns
                .iter()
                .map(|&id| value(id).filter(|value| !value.is_null()))
                .collect();
            let Some(key) = key else {
                continue;
            };
            if let Some(&holder) = keys.get(&key) {
                return Err(Error::UniqueConstraint {
                    index: name.clone(),
                    column: String::new(),
                    value: key.into(),
                    row_id: holder,
                });
            }
            keys.insert(key, row_id);
        }
        Ok(())
    }
}

/// A value of the column in the new type, checked against its definition
/// and, when `storable`, the range a TIMESTAMP is stored in
fn validate_value(
    value: Option<&Value>,
    change: &ColumnTypeChange<'_>,
    storable: bool,
) -> Result<()> {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        if change.nullable {
            return Ok(());
        }
        return Err(Error::not_null_constraint(change.column));
    };
    if let (true, Value::Timestamp(t)) = (storable, value) {
        crate::core::value::timestamp_in_range(*t)?;
    }
    if change.data_type == DataType::Vector && change.vector_dimensions > 0 {
        let got = value.vector_f32_iter().map_or(0, |values| values.len());
        if got != change.vector_dimensions as usize {
            return Err(Error::VectorDimensionMismatch {
                expected: change.vector_dimensions,
                got: u16::try_from(got).unwrap_or(u16::MAX),
            });
        }
    }
    Ok(())
}
