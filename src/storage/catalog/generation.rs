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

//! Immutable catalog contents. Validation proves internal consistency, not
//! provenance, persistence, or permission to release WAL retention ownership.

use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroU64;
use std::sync::Arc;

use crate::core::{DataType, Error, ForeignKeyAction, ForeignKeyConstraint, IndexType, Result};

use super::{ColumnId, Incarnation, SchemaRevision, TableId, TableIdentity, TableSchemaHistory};

fn invalid(message: &'static str) -> Error {
    Error::internal(message)
}

/// Actual identity of a DDL record. Epochs and source LSNs are distinct orders;
/// a reserved epoch may be appended after another epoch. Neither is a txn ID.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DdlStamp {
    epoch: NonZeroU64,
    source_lsn: NonZeroU64,
}

impl DdlStamp {
    pub fn new(epoch: u64, source_lsn: u64) -> Result<Self> {
        Ok(Self {
            epoch: NonZeroU64::new(epoch).ok_or_else(|| invalid("zero catalog DDL epoch"))?,
            source_lsn: NonZeroU64::new(source_lsn)
                .ok_or_else(|| invalid("zero catalog DDL source LSN"))?,
        })
    }

    pub fn epoch(self) -> u64 {
        self.epoch.get()
    }

    pub fn source_lsn(self) -> u64 {
        self.source_lsn.get()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DdlKind {
    CreateTable,
    AlterTable,
    RenameTable,
    TruncateTable,
    DropTable,
    CreateIndex,
    DropIndex,
    CreateView,
    DropView,
}

/// A revision key, not a replay command or a substitute for its original WAL
/// payload. The installer will acknowledge only the captured stamps.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum CatalogEffect {
    Table {
        identity: TableIdentity,
        schema_version: u64,
    },
    Index {
        identity: TableIdentity,
        name: String,
        definition_epoch: u64,
    },
    View {
        name: String,
        definition_epoch: u64,
    },
}

#[derive(Clone, Debug)]
pub struct CatalogMutation {
    pub stamp: DdlStamp,
    pub kind: DdlKind,
    pub effects: Vec<CatalogEffect>,
}

/// A declaration captured with these immutable contents. In particular,
/// `through_lsn` cannot be used as a durable receipt after encode/decode alone.
/// Captured mutations above this conservative prefix can still be represented.
#[derive(Clone, Debug, Default)]
pub struct CatalogCoverage {
    pub through_lsn: u64,
    pub ddl_epoch_cut: u64,
    pub captured_mutations: Vec<CatalogMutation>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IncarnationEndKind {
    Truncate,
    Drop,
}

#[derive(Clone, Copy, Debug)]
pub struct IncarnationEnd {
    pub stamp: DdlStamp,
    pub kind: IncarnationEndKind,
}

#[derive(Clone, Debug)]
pub struct CatalogIncarnation {
    pub incarnation: Incarnation,
    pub started_at: DdlStamp,
    pub first_schema_version: u64,
    pub last_schema_version: u64,
    pub ended: Option<IncarnationEnd>,
}

#[derive(Clone, Debug)]
pub struct TableNameEvent {
    pub stamp: DdlStamp,
    pub name: Option<String>,
}

#[derive(Clone, Copy, Debug)]
pub struct SchemaEvent {
    pub stamp: DdlStamp,
    pub identity: TableIdentity,
    pub version: u64,
}

/// The spelling/actions live in Schema.foreign_keys; these bindings in the same
/// order carry identity. Parent TRUNCATE does not change this relationship.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ForeignKeyBinding {
    /// The child schema event that introduced this constraint. A carried
    /// constraint keeps this stamp, including through parent/name changes.
    pub defined_at: DdlStamp,
    pub local_column: ColumnId,
    pub parent_table: TableId,
    pub parent_column: ColumnId,
    pub parent_schema_version: u64,
}

#[derive(Clone, Debug)]
pub struct SchemaForeignKeys {
    pub schema_version: u64,
    pub bindings: Vec<ForeignKeyBinding>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CatalogHnswOptions {
    pub m: Option<u16>,
    pub ef_construction: Option<u16>,
    pub ef_search: Option<u16>,
    pub distance_metric: Option<u8>,
}

#[derive(Clone, Debug)]
pub struct CatalogIndex {
    pub name: String,
    pub defined_at: DdlStamp,
    pub dropped_at: Option<DdlStamp>,
    pub schema_version: u64,
    pub columns: Vec<ColumnId>,
    pub index_type: IndexType,
    pub is_unique: bool,
    pub hnsw: CatalogHnswOptions,
}

#[derive(Clone, Debug)]
pub struct CatalogView {
    pub original_name: String,
    pub query: String,
    pub defined_at: DdlStamp,
    pub dropped_at: Option<DdlStamp>,
}

#[derive(Clone, Debug)]
pub struct CatalogTable {
    pub history: TableSchemaHistory,
    pub incarnations: Vec<CatalogIncarnation>,
    pub names: Vec<TableNameEvent>,
    pub schema_events: Vec<SchemaEvent>,
    pub foreign_keys: Vec<SchemaForeignKeys>,
    pub indexes: Vec<CatalogIndex>,
}

impl CatalogTable {
    pub fn table_id(&self) -> TableId {
        self.history.identity().table_id
    }

    pub fn current_name(&self) -> Option<&str> {
        self.names.last().and_then(|event| event.name.as_deref())
    }

    pub fn is_dropped(&self) -> bool {
        self.current_name().is_none()
    }

    fn name_at(&self, epoch: u64) -> Option<&str> {
        let end = self
            .names
            .partition_point(|event| event.stamp.epoch() <= epoch);
        end.checked_sub(1)
            .and_then(|index| self.names[index].name.as_deref())
    }

    fn incarnation_at(&self, epoch: u64) -> Option<&CatalogIncarnation> {
        let end = self
            .incarnations
            .partition_point(|item| item.started_at.epoch() <= epoch);
        end.checked_sub(1)
            .map(|index| &self.incarnations[index])
            .filter(|item| item.ended.is_none_or(|end| epoch <= end.stamp.epoch()))
    }

    fn schema_event(&self, version: u64) -> Result<&SchemaEvent> {
        self.schema_events
            .binary_search_by_key(&version, |event| event.version)
            .map(|index| &self.schema_events[index])
            .map_err(|_| invalid("catalog schema event is unavailable"))
    }

    fn schema_version_at(&self, epoch: u64) -> Option<u64> {
        let end = self
            .schema_events
            .partition_point(|event| event.stamp.epoch() <= epoch);
        end.checked_sub(1)
            .map(|index| self.schema_events[index].version)
    }

    fn revision(&self, version: u64) -> Result<&SchemaRevision> {
        self.history.lookup_version(version)
    }

    fn resolve(&self, identity: TableIdentity, version: u64) -> Result<&SchemaRevision> {
        if identity.table_id != self.table_id() {
            return Err(invalid("catalog table identity mismatch"));
        }
        let item = self
            .incarnations
            .binary_search_by_key(&identity.incarnation, |item| item.incarnation)
            .map(|index| &self.incarnations[index])
            .map_err(|_| invalid("catalog incarnation is unavailable"))?;
        if version < item.first_schema_version || version > item.last_schema_version {
            return Err(invalid("schema version is outside its catalog incarnation"));
        }
        self.revision(version)
    }
}

/// Owned construction input. A successful generation exposes only immutable
/// borrows of these parts; cloning a generation shares all of its metadata.
#[derive(Clone, Debug)]
pub struct CatalogParts {
    pub generation: NonZeroU64,
    pub table_id_high_water_mark: u64,
    pub ddl_epoch_high_water_mark: u64,
    pub wal_observation_ceiling: u64,
    pub coverage: CatalogCoverage,
    pub tables: Vec<CatalogTable>,
    pub views: Vec<CatalogView>,
}

#[derive(Clone, Debug)]
pub struct CatalogGeneration {
    parts: Arc<CatalogParts>,
}

impl CatalogGeneration {
    pub fn try_new(parts: CatalogParts) -> Result<Self> {
        validate_parts(&parts)?;
        Ok(Self {
            parts: Arc::new(parts),
        })
    }

    pub fn validate(&self) -> Result<()> {
        validate_parts(&self.parts)
    }

    pub fn generation(&self) -> NonZeroU64 {
        self.parts.generation
    }

    pub fn parts(&self) -> &CatalogParts {
        &self.parts
    }

    pub fn tables(&self) -> &[CatalogTable] {
        &self.parts.tables
    }

    pub fn coverage(&self) -> &CatalogCoverage {
        &self.parts.coverage
    }

    pub fn resolve_schema(&self, identity: TableIdentity, version: u64) -> Result<&SchemaRevision> {
        find_table(&self.parts, identity.table_id)?.resolve(identity, version)
    }
}

fn find_table(parts: &CatalogParts, id: TableId) -> Result<&CatalogTable> {
    parts
        .tables
        .binary_search_by_key(&id, CatalogTable::table_id)
        .map(|index| &parts.tables[index])
        .map_err(|_| invalid("catalog table ID is unavailable"))
}

fn mutation(parts: &CatalogParts, stamp: DdlStamp) -> Result<&CatalogMutation> {
    let item = parts
        .coverage
        .captured_mutations
        .binary_search_by_key(&stamp.epoch(), |item| item.stamp.epoch())
        .map(|index| &parts.coverage.captured_mutations[index])
        .ok()
        .filter(|item| item.stamp == stamp)
        .ok_or_else(|| invalid("catalog DDL stamp is absent or has a different source LSN"))?;
    Ok(item)
}

fn table_effect(parts: &CatalogParts, event: SchemaEvent) -> Result<()> {
    let expected = CatalogEffect::Table {
        identity: event.identity,
        schema_version: event.version,
    };
    if mutation(parts, event.stamp)?
        .effects
        .binary_search(&expected)
        .is_err()
    {
        return Err(invalid(
            "catalog DDL does not name its table schema revision",
        ));
    }
    Ok(())
}

fn table_at_stamp_effect(
    parts: &CatalogParts,
    table: &CatalogTable,
    stamp: DdlStamp,
) -> Result<()> {
    let incarnation = table
        .incarnation_at(stamp.epoch())
        .ok_or_else(|| invalid("catalog DDL is outside the table lifetime"))?;
    let version = table
        .schema_version_at(stamp.epoch())
        .ok_or_else(|| invalid("catalog DDL precedes the table schema"))?;
    table_effect(
        parts,
        SchemaEvent {
            stamp,
            identity: TableIdentity::new(table.table_id(), incarnation.incarnation),
            version,
        },
    )
}

fn validate_parts(parts: &CatalogParts) -> Result<()> {
    let coverage = &parts.coverage;
    if coverage.through_lsn > parts.wal_observation_ceiling
        || coverage.ddl_epoch_cut > parts.ddl_epoch_high_water_mark
    {
        return Err(invalid("catalog coverage exceeds its capture boundary"));
    }
    let mut previous_epoch = 0;
    let mut source_lsns = BTreeSet::new();
    for item in &coverage.captured_mutations {
        if item.stamp.epoch() <= previous_epoch
            || item.stamp.epoch() > coverage.ddl_epoch_cut
            || item.stamp.source_lsn() > parts.wal_observation_ceiling
            || !source_lsns.insert(item.stamp.source_lsn())
            || item.effects.is_empty()
            || item.effects.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(invalid("invalid or duplicate catalog DDL mutation"));
        }
        previous_epoch = item.stamp.epoch();
    }
    let mut previous_table = 0;
    for table in &parts.tables {
        let id = table.table_id().get();
        if id <= previous_table || id > parts.table_id_high_water_mark {
            return Err(invalid(
                "catalog table IDs are unordered or exceed allocation high-water mark",
            ));
        }
        previous_table = id;
        validate_table(parts, table)?;
    }
    validate_names(parts)?;
    for table in &parts.tables {
        validate_foreign_keys(parts, table)?;
    }
    validate_views(parts)?;
    for item in &coverage.captured_mutations {
        for effect in &item.effects {
            validate_effect(parts, item, effect)?;
        }
    }
    Ok(())
}

fn validate_table(parts: &CatalogParts, table: &CatalogTable) -> Result<()> {
    let first = table
        .incarnations
        .first()
        .ok_or_else(|| invalid("catalog table has no incarnation"))?;
    let last = table.incarnations.last().unwrap();
    if first.incarnation != Incarnation::FIRST
        || last.incarnation != table.history.identity().incarnation
        || last.last_schema_version != table.history.current_version()
    {
        return Err(invalid(
            "catalog incarnation history does not cover its table",
        ));
    }
    for (index, item) in table.incarnations.iter().enumerate() {
        table.revision(item.first_schema_version)?;
        table.revision(item.last_schema_version)?;
        if item.first_schema_version > item.last_schema_version {
            return Err(invalid("catalog incarnation schema interval is reversed"));
        }
        let start_kind = mutation(parts, item.started_at)?.kind;
        if (index == 0 && start_kind != DdlKind::CreateTable)
            || (index != 0 && start_kind != DdlKind::TruncateTable)
        {
            return Err(invalid("catalog incarnation has an invalid creation event"));
        }
        table_effect(
            parts,
            SchemaEvent {
                stamp: item.started_at,
                identity: TableIdentity::new(table.table_id(), item.incarnation),
                version: item.first_schema_version,
            },
        )?;
        if let Some(end) = item.ended {
            if end.stamp.epoch() <= item.started_at.epoch()
                || mutation(parts, end.stamp)?.kind
                    != match end.kind {
                        IncarnationEndKind::Truncate => DdlKind::TruncateTable,
                        IncarnationEndKind::Drop => DdlKind::DropTable,
                    }
            {
                return Err(invalid("catalog incarnation has an invalid terminal event"));
            }
            table_effect(
                parts,
                SchemaEvent {
                    stamp: end.stamp,
                    identity: TableIdentity::new(table.table_id(), item.incarnation),
                    version: item.last_schema_version,
                },
            )?;
        }
        if let Some(next) = table.incarnations.get(index + 1) {
            if !item.ended.is_some_and(|end| {
                end.kind == IncarnationEndKind::Truncate && end.stamp == next.started_at
            }) || item.incarnation.checked_next()? != next.incarnation
                || item.last_schema_version != next.first_schema_version
            {
                return Err(invalid(
                    "catalog incarnation transition has a gap or reused identity",
                ));
            }
        } else if item
            .ended
            .is_some_and(|end| end.kind == IncarnationEndKind::Truncate)
        {
            return Err(invalid("catalog TRUNCATE has no replacement incarnation"));
        }
    }
    if table.schema_events.len() != table.history.revisions().count() {
        return Err(invalid("catalog schema event count does not match history"));
    }
    let mut previous_epoch = 0;
    for (event, revision) in table.schema_events.iter().zip(table.history.revisions()) {
        if event.version != revision.version() || event.stamp.epoch() <= previous_epoch {
            return Err(invalid("catalog schema events are not in revision order"));
        }
        previous_epoch = event.stamp.epoch();
        table.resolve(event.identity, event.version)?;
        let incarnation = table
            .incarnation_at(event.stamp.epoch())
            .ok_or_else(|| invalid("catalog schema event is outside the table lifetime"))?;
        if incarnation.incarnation != event.identity.incarnation {
            return Err(invalid(
                "catalog schema event belongs to another incarnation",
            ));
        }
        if !matches!(
            mutation(parts, event.stamp)?.kind,
            DdlKind::CreateTable
                | DdlKind::AlterTable
                | DdlKind::RenameTable
                | DdlKind::TruncateTable
                | DdlKind::DropTable
        ) {
            return Err(invalid("catalog schema revision is not a table DDL event"));
        }
        table_effect(parts, *event)?;
    }
    if table.schema_events.first().is_none_or(|event| {
        event.stamp != first.started_at || event.version != first.first_schema_version
    }) {
        return Err(invalid(
            "catalog first schema does not match table creation",
        ));
    }
    for item in &table.incarnations {
        if table.schema_version_at(item.started_at.epoch()) != Some(item.first_schema_version)
            || item.ended.is_some_and(|end| {
                table.schema_version_at(end.stamp.epoch()) != Some(item.last_schema_version)
            })
        {
            return Err(invalid(
                "catalog incarnation schema bounds do not match DDL history",
            ));
        }
    }
    validate_table_names(parts, table)?;
    validate_indexes(parts, table)
}

fn validate_table_names(parts: &CatalogParts, table: &CatalogTable) -> Result<()> {
    let first = &table.incarnations[0];
    if table
        .names
        .first()
        .is_none_or(|event| event.stamp != first.started_at || event.name.is_none())
    {
        return Err(invalid(
            "catalog table name history does not begin at CREATE",
        ));
    }
    let mut previous_epoch = 0;
    for (index, event) in table.names.iter().enumerate() {
        if event.stamp.epoch() <= previous_epoch
            || event.name.as_ref().is_some_and(String::is_empty)
        {
            return Err(invalid("invalid catalog table name event"));
        }
        previous_epoch = event.stamp.epoch();
        let kind = mutation(parts, event.stamp)?.kind;
        if kind
            != if index == 0 {
                DdlKind::CreateTable
            } else if event.name.is_some() {
                DdlKind::RenameTable
            } else {
                DdlKind::DropTable
            }
            || (event.name.is_none() && index + 1 != table.names.len())
        {
            return Err(invalid("catalog name event is not CREATE/RENAME/DROP"));
        }
        table_at_stamp_effect(parts, table, event.stamp)?;
    }
    let end = table.incarnations.last().unwrap().ended;
    let name_end = table.names.last().unwrap();
    if table.is_dropped() != end.is_some_and(|end| end.kind == IncarnationEndKind::Drop)
        || (table.is_dropped() && end.unwrap().stamp != name_end.stamp)
    {
        return Err(invalid("catalog DROP tombstone and incarnation disagree"));
    }
    Ok(())
}

fn validate_indexes(parts: &CatalogParts, table: &CatalogTable) -> Result<()> {
    let mut names: BTreeMap<String, Option<u64>> = BTreeMap::new();
    let mut previous_key = None;
    for index in &table.indexes {
        let key = (index.name.to_lowercase(), index.defined_at.epoch());
        if index.name.is_empty()
            || previous_key
                .as_ref()
                .is_some_and(|previous| previous >= &key)
        {
            return Err(invalid(
                "catalog index revisions are not in canonical order",
            ));
        }
        if let Some(end) = names.get(&key.0) {
            if end.is_none_or(|end| end > index.defined_at.epoch()) {
                return Err(invalid("catalog index name lifetimes overlap"));
            }
        }
        names.insert(key.0.clone(), index.dropped_at.map(DdlStamp::epoch));
        previous_key = Some(key);
        let revision = table.revision(index.schema_version)?;
        let event = table.schema_event(index.schema_version)?;
        if table.schema_version_at(index.defined_at.epoch()) != Some(index.schema_version) {
            return Err(invalid(
                "catalog index definition does not use its exact active schema version",
            ));
        }
        if event.stamp.epoch() > index.defined_at.epoch()
            || table.name_at(index.defined_at.epoch()).is_none()
        {
            return Err(invalid("catalog index precedes its schema or table"));
        }
        let mut columns = BTreeSet::new();
        if index.columns.is_empty() {
            return Err(invalid("catalog index has no columns"));
        }
        for &column in &index.columns {
            revision.column_position(column)?;
            if !columns.insert(column) {
                return Err(invalid("catalog index repeats a column identity"));
            }
            if index.dropped_at.is_none() {
                table
                    .revision(table.history.current_version())?
                    .column_position(column)?;
            }
        }
        validate_column_lifetime(
            table,
            &index.columns,
            index.defined_at.epoch(),
            index.dropped_at.map(DdlStamp::epoch),
        )?;
        if index.index_type == IndexType::PrimaryKey {
            return Err(invalid(
                "implicit primary-key index must be derived from its schema",
            ));
        }
        if index.index_type == IndexType::Hnsw {
            if index.columns.len() != 1
                || revision.schema().columns[revision.column_position(index.columns[0])?].data_type
                    != DataType::Vector
                || index.hnsw.m.is_some_and(|m| m < 2)
                || index.hnsw.distance_metric.is_some_and(|metric| metric > 2)
            {
                return Err(invalid("invalid catalog HNSW definition"));
            }
            // Existing APIs permit zero ef values; retain their runtime meaning.
            // An installed graph has a fixed vector shape. ALTER only changes
            // schema metadata today, so a shape change needs a separate index
            // revision; it cannot silently keep this graph alive.
            let dimensions = revision.schema().columns
                [revision.column_position(index.columns[0])?]
            .vector_dimensions;
            let begin = table
                .schema_events
                .partition_point(|event| event.stamp.epoch() <= index.defined_at.epoch());
            let finish = index.dropped_at.map_or(table.schema_events.len(), |end| {
                table
                    .schema_events
                    .partition_point(|event| event.stamp.epoch() < end.epoch())
            });
            for event in &table.schema_events[begin..finish.max(begin)] {
                let schema = table.revision(event.version)?;
                let column = &schema.schema().columns[schema.column_position(index.columns[0])?];
                if column.data_type != DataType::Vector || column.vector_dimensions != dimensions {
                    return Err(invalid(
                        "catalog HNSW vector shape changed without an index replacement",
                    ));
                }
            }
        } else if index.hnsw != CatalogHnswOptions::default() {
            return Err(invalid("non-HNSW catalog index has HNSW options"));
        }
        if index
            .dropped_at
            .is_some_and(|end| end.epoch() < index.defined_at.epoch())
            || (table.is_dropped() && index.dropped_at.is_none())
        {
            return Err(invalid("catalog index has an invalid terminal event"));
        }
        for (stamp, creating) in [(Some(index.defined_at), true), (index.dropped_at, false)] {
            if let Some(stamp) = stamp {
                let item = mutation(parts, stamp)?;
                let explicit = if creating {
                    DdlKind::CreateIndex
                } else {
                    DdlKind::DropIndex
                };
                if item.kind == explicit {
                    let incarnation = table
                        .incarnation_at(stamp.epoch())
                        .ok_or_else(|| invalid("catalog index DDL is outside table lifetime"))?;
                    if item
                        .effects
                        .binary_search(&CatalogEffect::Index {
                            identity: TableIdentity::new(table.table_id(), incarnation.incarnation),
                            name: index.name.to_lowercase(),
                            definition_epoch: index.defined_at.epoch(),
                        })
                        .is_err()
                    {
                        return Err(invalid("catalog DDL does not name its index revision"));
                    }
                } else if (creating
                    && matches!(item.kind, DdlKind::CreateTable | DdlKind::AlterTable))
                    || (!creating && matches!(item.kind, DdlKind::AlterTable | DdlKind::DropTable))
                {
                    table_at_stamp_effect(parts, table, stamp)?;
                } else {
                    return Err(invalid("catalog index event has an invalid DDL kind"));
                }
            }
        }
    }
    Ok(())
}

/// Half-open lifetime: removing an index/FK in the same DDL event that drops
/// its column/table is valid. A later removal cannot conceal an earlier gap.
fn validate_column_lifetime(
    table: &CatalogTable,
    columns: &[ColumnId],
    start: u64,
    end: Option<u64>,
) -> Result<()> {
    if end.is_some_and(|end| end < start) || table.name_at(start).is_none() {
        return Err(invalid(
            "catalog column reference is outside its table lifetime",
        ));
    }
    let start_version = table
        .schema_version_at(start)
        .ok_or_else(|| invalid("catalog column reference precedes its schema"))?;
    for &column in columns {
        table.revision(start_version)?.column_position(column)?;
    }
    let begin = table
        .schema_events
        .partition_point(|event| event.stamp.epoch() <= start);
    let finish = end.map_or(table.schema_events.len(), |end| {
        table
            .schema_events
            .partition_point(|event| event.stamp.epoch() < end)
    });
    for event in &table.schema_events[begin..finish.max(begin)] {
        for &column in columns {
            table.revision(event.version)?.column_position(column)?;
        }
    }
    if table
        .incarnations
        .last()
        .and_then(|item| item.ended)
        .is_some_and(|dropped| end.is_none_or(|end| dropped.stamp.epoch() < end))
    {
        return Err(invalid("catalog column reference outlives its table"));
    }
    Ok(())
}

fn validate_foreign_keys(parts: &CatalogParts, table: &CatalogTable) -> Result<()> {
    if table.foreign_keys.len() != table.schema_events.len() {
        return Err(invalid(
            "catalog FK binding history does not cover every schema",
        ));
    }
    type Signature = (ForeignKeyBinding, u8, u8);
    // One previous schema only. Consuming an entry preserves multiplicity and
    // prevents a removed constraint from borrowing an older surviving origin.
    let mut previous: BTreeMap<Signature, Vec<&ForeignKeyConstraint>> = BTreeMap::new();
    for (ordinal, (bound, event)) in table
        .foreign_keys
        .iter()
        .zip(&table.schema_events)
        .enumerate()
    {
        let mut next: BTreeMap<Signature, Vec<&ForeignKeyConstraint>> = BTreeMap::new();
        let revision = table.revision(event.version)?;
        if bound.schema_version != event.version
            || bound.bindings.len() != revision.schema().foreign_keys.len()
        {
            return Err(invalid(
                "catalog FK binding count/version differs from its schema",
            ));
        }
        for (binding, foreign_key) in bound.bindings.iter().zip(&revision.schema().foreign_keys) {
            if revision.column_id_at(foreign_key.column_index)? != binding.local_column {
                return Err(invalid(
                    "catalog FK local identity does not match its schema",
                ));
            }
            let local = &revision.schema().columns[foreign_key.column_index];
            if !local.nullable
                && (foreign_key.on_delete == ForeignKeyAction::SetNull
                    || foreign_key.on_update == ForeignKeyAction::SetNull)
            {
                return Err(invalid(
                    "catalog SET NULL foreign key has a nonnullable column",
                ));
            }
            let parent = find_table(parts, binding.parent_table)?;
            let parent_version = parent
                .schema_version_at(event.stamp.epoch())
                .ok_or_else(|| invalid("catalog FK precedes its parent schema"))?;
            let parent_revision = parent.revision(parent_version)?;
            let parent_column = &parent_revision.schema().columns
                [parent_revision.column_position(binding.parent_column)?];
            let parent_name = parent
                .name_at(event.stamp.epoch())
                .ok_or_else(|| invalid("catalog FK is outside the parent table lifetime"))?;
            let signature = (
                *binding,
                foreign_key.on_delete.as_u8(),
                foreign_key.on_update.as_u8(),
            );
            let refers_to_current_name = parent_name.to_lowercase()
                == foreign_key.referenced_table.to_lowercase()
                && parent_column.name_lower == foreign_key.referenced_column.to_lowercase();
            if binding.defined_at == event.stamp {
                if !matches!(
                    mutation(parts, event.stamp)?.kind,
                    DdlKind::CreateTable | DdlKind::AlterTable
                ) || binding.parent_schema_version != parent_version
                    || !refers_to_current_name
                {
                    return Err(invalid(
                        "new catalog FK origin does not match the parent's exact schema and name",
                    ));
                }
            } else {
                if binding.defined_at.epoch() >= event.stamp.epoch() {
                    return Err(invalid(
                        "catalog FK origin does not match an introducing schema event",
                    ));
                }
                let prior = previous.get_mut(&signature).ok_or_else(|| {
                    invalid(
                        "catalog FK reuses an origin after its definition changed or was removed",
                    )
                })?;
                let match_at = prior.iter().position(|old| refers_to_current_name
                    || (old.referenced_table.to_lowercase() == foreign_key.referenced_table.to_lowercase()
                        && old.referenced_column.to_lowercase() == foreign_key.referenced_column.to_lowercase()))
                    .ok_or_else(|| invalid("carried catalog FK changes its target spelling without a matching rename"))?;
                prior.swap_remove(match_at);
            }
            next.entry(signature).or_default().push(foreign_key);
            let lifetime_end = table
                .schema_events
                .get(ordinal + 1)
                .map(|next| next.stamp.epoch())
                .or_else(|| {
                    table
                        .incarnations
                        .last()
                        .and_then(|item| item.ended)
                        .map(|end| end.stamp.epoch())
                });
            validate_column_lifetime(
                parent,
                &[binding.parent_column],
                event.stamp.epoch(),
                lifetime_end,
            )?;
        }
        previous = next;
    }
    Ok(())
}

fn validate_names(parts: &CatalogParts) -> Result<()> {
    // Half-open intervals allow DROP/RENAME and reuse by one atomic DDL epoch.
    let mut names: BTreeMap<String, Vec<(u64, Option<u64>)>> = BTreeMap::new();
    for table in &parts.tables {
        for (index, event) in table.names.iter().enumerate() {
            if let Some(name) = &event.name {
                names.entry(name.to_lowercase()).or_default().push((
                    event.stamp.epoch(),
                    table.names.get(index + 1).map(|next| next.stamp.epoch()),
                ));
            }
        }
    }
    for view in &parts.views {
        names
            .entry(view.original_name.to_lowercase())
            .or_default()
            .push((
                view.defined_at.epoch(),
                view.dropped_at.map(DdlStamp::epoch),
            ));
    }
    for intervals in names.values_mut() {
        intervals.sort_unstable_by_key(|interval| interval.0);
        if intervals
            .windows(2)
            .any(|pair| pair[0].1.is_none_or(|end| end > pair[1].0))
        {
            return Err(invalid("catalog table/view name lifetimes overlap"));
        }
    }
    Ok(())
}

fn validate_views(parts: &CatalogParts) -> Result<()> {
    let mut previous = None;
    for view in &parts.views {
        let key = (view.original_name.to_lowercase(), view.defined_at.epoch());
        if view.original_name.is_empty()
            || previous.as_ref().is_some_and(|previous| previous >= &key)
            || view
                .dropped_at
                .is_some_and(|end| end.epoch() < view.defined_at.epoch())
        {
            return Err(invalid("invalid or unordered catalog view history"));
        }
        previous = Some(key.clone());
        for (stamp, kind) in [
            (Some(view.defined_at), DdlKind::CreateView),
            (view.dropped_at, DdlKind::DropView),
        ] {
            if let Some(stamp) = stamp {
                let item = mutation(parts, stamp)?;
                if item.kind != kind
                    || item
                        .effects
                        .binary_search(&CatalogEffect::View {
                            name: key.0.clone(),
                            definition_epoch: view.defined_at.epoch(),
                        })
                        .is_err()
                {
                    return Err(invalid("catalog DDL does not name its view revision"));
                }
            }
        }
    }
    Ok(())
}

fn validate_effect(
    parts: &CatalogParts,
    item: &CatalogMutation,
    effect: &CatalogEffect,
) -> Result<()> {
    let represented = match effect {
        CatalogEffect::Table {
            identity,
            schema_version,
        } => {
            let table = find_table(parts, identity.table_id)?;
            table.resolve(*identity, *schema_version)?;
            let incarnation = &table.incarnations[table
                .incarnations
                .binary_search_by_key(&identity.incarnation, |record| record.incarnation)
                .unwrap()];
            let event = table.schema_event(*schema_version)?;
            (event.stamp == item.stamp && event.identity == *identity)
                || (incarnation.started_at == item.stamp
                    && incarnation.first_schema_version == *schema_version)
                || incarnation.ended.is_some_and(|end| {
                    end.stamp == item.stamp && incarnation.last_schema_version == *schema_version
                })
                || (table
                    .incarnation_at(item.stamp.epoch())
                    .is_some_and(|record| record.incarnation == identity.incarnation)
                    && table.schema_version_at(item.stamp.epoch()) == Some(*schema_version)
                    && table
                        .names
                        .binary_search_by_key(&item.stamp.epoch(), |event| event.stamp.epoch())
                        .is_ok())
        }
        CatalogEffect::Index {
            identity,
            name,
            definition_epoch,
        } => {
            let table = find_table(parts, identity.table_id)?;
            if name != &name.to_lowercase()
                || table
                    .incarnation_at(item.stamp.epoch())
                    .is_none_or(|record| record.incarnation != identity.incarnation)
            {
                return Err(invalid(
                    "catalog index effect has a different identity or noncanonical name",
                ));
            }
            table
                .indexes
                .binary_search_by(|index| {
                    index
                        .name
                        .to_lowercase()
                        .cmp(name)
                        .then_with(|| index.defined_at.epoch().cmp(definition_epoch))
                })
                .ok()
                .map(|position| &table.indexes[position])
                .is_some_and(|index| {
                    index.defined_at == item.stamp || index.dropped_at == Some(item.stamp)
                })
        }
        CatalogEffect::View {
            name,
            definition_epoch,
        } => {
            if name != &name.to_lowercase() {
                return Err(invalid("catalog view effect name is not canonical"));
            }
            parts
                .views
                .binary_search_by(|view| {
                    view.original_name
                        .to_lowercase()
                        .cmp(name)
                        .then_with(|| view.defined_at.epoch().cmp(definition_epoch))
                })
                .ok()
                .map(|position| &parts.views[position])
                .is_some_and(|view| {
                    view.defined_at == item.stamp || view.dropped_at == Some(item.stamp)
                })
        }
    };
    if !represented {
        return Err(invalid(
            "captured catalog DDL effect is not represented by the generation",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{ForeignKeyConstraint, Schema, SchemaColumn, Value};

    fn stamp(epoch: u64) -> DdlStamp {
        DdlStamp::new(epoch, epoch * 100).unwrap()
    }

    fn identity(table: u64, incarnation: u64) -> TableIdentity {
        TableIdentity::new(
            TableId::new(table).unwrap(),
            Incarnation::new(incarnation).unwrap(),
        )
    }

    fn column(id: u64) -> ColumnId {
        ColumnId::new(id).unwrap()
    }

    fn table_effect_for(table: u64, incarnation: u64, version: u64) -> CatalogEffect {
        CatalogEffect::Table {
            identity: identity(table, incarnation),
            schema_version: version,
        }
    }

    fn index_effect(incarnation: u64, definition_epoch: u64) -> CatalogEffect {
        CatalogEffect::Index {
            identity: identity(1, incarnation),
            name: "search".to_string(),
            definition_epoch,
        }
    }

    fn view_effect(definition_epoch: u64) -> CatalogEffect {
        CatalogEffect::View {
            name: "v".to_string(),
            definition_epoch,
        }
    }

    fn simple_table(
        id: u64,
        name: &str,
        epoch: u64,
        schema: Schema,
        columns: Vec<ColumnId>,
    ) -> CatalogTable {
        let identity = identity(id, 1);
        CatalogTable {
            history: TableSchemaHistory::new(identity, 0, schema, columns).unwrap(),
            incarnations: vec![CatalogIncarnation {
                incarnation: Incarnation::FIRST,
                started_at: stamp(epoch),
                first_schema_version: 0,
                last_schema_version: 0,
                ended: None,
            }],
            names: vec![TableNameEvent {
                stamp: stamp(epoch),
                name: Some(name.to_string()),
            }],
            schema_events: vec![SchemaEvent {
                stamp: stamp(epoch),
                identity,
                version: 0,
            }],
            foreign_keys: vec![SchemaForeignKeys {
                schema_version: 0,
                bindings: Vec::new(),
            }],
            indexes: Vec::new(),
        }
    }

    fn parts() -> CatalogParts {
        let mut parent_schema = Schema::new(
            "Parent",
            vec![
                SchemaColumn::primary_key(0, "id", DataType::Integer),
                SchemaColumn::nullable(1, "label", DataType::Text),
                SchemaColumn::nullable(2, "embedding", DataType::Vector).with_vector_dimensions(3),
            ],
        );
        let mut parent = simple_table(
            1,
            "Parent",
            1,
            parent_schema.clone(),
            vec![column(1), column(2), column(3)],
        );
        parent.names.push(TableNameEvent {
            stamp: stamp(4),
            name: Some("Renamed".to_string()),
        });
        parent_schema.table_name = "Renamed".to_string();
        parent_schema.table_name_lower = "renamed".to_string();
        let mut added = SchemaColumn::new(3, "introduced", DataType::Text, false, false);
        added.default_expr = Some("'recorded'".to_string());
        added.default_value = Some(Value::text("recorded"));
        added.check_expr = Some("introduced <> ''".to_string());
        parent_schema.add_column(added).unwrap();
        parent.history = parent
            .history
            .with_revision(
                identity(1, 1),
                10,
                parent_schema,
                vec![column(1), column(2), column(3), column(9)],
            )
            .unwrap()
            .with_column_high_water_mark(20)
            .unwrap()
            .checked_next_incarnation()
            .unwrap();
        parent.incarnations[0].last_schema_version = 10;
        parent.incarnations[0].ended = Some(IncarnationEnd {
            stamp: stamp(6),
            kind: IncarnationEndKind::Truncate,
        });
        parent.incarnations.push(CatalogIncarnation {
            incarnation: Incarnation::new(2).unwrap(),
            started_at: stamp(6),
            first_schema_version: 10,
            last_schema_version: 10,
            ended: None,
        });
        parent.schema_events.push(SchemaEvent {
            stamp: stamp(5),
            identity: identity(1, 1),
            version: 10,
        });
        parent.foreign_keys.push(SchemaForeignKeys {
            schema_version: 10,
            bindings: Vec::new(),
        });
        parent.indexes = vec![
            CatalogIndex {
                name: "Search".to_string(),
                defined_at: stamp(2),
                dropped_at: Some(stamp(7)),
                schema_version: 0,
                columns: vec![column(2)],
                index_type: IndexType::Hash,
                is_unique: true,
                hnsw: CatalogHnswOptions::default(),
            },
            CatalogIndex {
                name: "Search".to_string(),
                defined_at: stamp(8),
                dropped_at: None,
                schema_version: 10,
                columns: vec![column(3)],
                index_type: IndexType::Hnsw,
                is_unique: false,
                hnsw: CatalogHnswOptions {
                    m: Some(8),
                    ef_construction: Some(0),
                    ef_search: Some(0),
                    distance_metric: Some(2),
                },
            },
        ];
        let child_schema = Schema::with_foreign_keys(
            "Child",
            vec![
                SchemaColumn::primary_key(0, "id", DataType::Integer),
                SchemaColumn::nullable(1, "parent", DataType::Integer),
            ],
            vec![ForeignKeyConstraint {
                column_index: 1,
                column_name: "parent".to_string(),
                referenced_table: "parent".to_string(),
                referenced_column: "id".to_string(),
                on_delete: ForeignKeyAction::SetNull,
                on_update: ForeignKeyAction::Cascade,
            }],
        );
        let mut child = simple_table(5, "Child", 3, child_schema, vec![column(1), column(8)]);
        child.foreign_keys[0].bindings.push(ForeignKeyBinding {
            defined_at: stamp(3),
            local_column: column(8),
            parent_table: TableId::new(1).unwrap(),
            parent_column: column(1),
            parent_schema_version: 0,
        });
        child.incarnations[0].ended = Some(IncarnationEnd {
            stamp: stamp(12),
            kind: IncarnationEndKind::Drop,
        });
        child.names.push(TableNameEvent {
            stamp: stamp(12),
            name: None,
        });
        let new_child = simple_table(
            8,
            "Child",
            13,
            Schema::new(
                "Child",
                vec![SchemaColumn::primary_key(0, "id", DataType::Integer)],
            ),
            vec![column(1)],
        );
        let operations = [
            (1, DdlKind::CreateTable, vec![table_effect_for(1, 1, 0)]),
            (2, DdlKind::CreateIndex, vec![index_effect(1, 2)]),
            (3, DdlKind::CreateTable, vec![table_effect_for(5, 1, 0)]),
            (4, DdlKind::RenameTable, vec![table_effect_for(1, 1, 0)]),
            (5, DdlKind::AlterTable, vec![table_effect_for(1, 1, 10)]),
            (
                6,
                DdlKind::TruncateTable,
                vec![table_effect_for(1, 1, 10), table_effect_for(1, 2, 10)],
            ),
            (7, DdlKind::DropIndex, vec![index_effect(2, 2)]),
            (8, DdlKind::CreateIndex, vec![index_effect(2, 8)]),
            (9, DdlKind::CreateView, vec![view_effect(9)]),
            (10, DdlKind::DropView, vec![view_effect(9)]),
            (11, DdlKind::CreateView, vec![view_effect(11)]),
            (12, DdlKind::DropTable, vec![table_effect_for(5, 1, 0)]),
            (13, DdlKind::CreateTable, vec![table_effect_for(8, 1, 0)]),
        ];
        CatalogParts {
            generation: NonZeroU64::new(1).unwrap(),
            table_id_high_water_mark: 32,
            ddl_epoch_high_water_mark: 30,
            wal_observation_ceiling: 2000,
            coverage: CatalogCoverage {
                through_lsn: 650,
                ddl_epoch_cut: 13,
                captured_mutations: operations
                    .into_iter()
                    .map(|(epoch, kind, effects)| CatalogMutation {
                        stamp: stamp(epoch),
                        kind,
                        effects,
                    })
                    .collect(),
            },
            tables: vec![parent, child, new_child],
            views: vec![
                CatalogView {
                    original_name: "V".to_string(),
                    query: "SELECT id FROM Renamed".to_string(),
                    defined_at: stamp(9),
                    dropped_at: Some(stamp(10)),
                },
                CatalogView {
                    original_name: "V".to_string(),
                    query: "SELECT label FROM Renamed".to_string(),
                    defined_at: stamp(11),
                    dropped_at: None,
                },
            ],
        }
    }

    #[test]
    fn complete_generation_retains_exact_schema_incarnations_and_tombstones() {
        let generation = CatalogGeneration::try_new(parts()).unwrap();
        generation.validate().unwrap();
        assert_eq!(
            generation
                .resolve_schema(identity(1, 1), 0)
                .unwrap()
                .schema()
                .table_name,
            "Parent"
        );
        let current = generation.resolve_schema(identity(1, 2), 10).unwrap();
        assert_eq!(
            current.column_ids(),
            &[column(1), column(2), column(3), column(9)]
        );
        assert_eq!(
            current.schema().columns[3].default_value,
            Some(Value::text("recorded"))
        );
        assert_eq!(
            current.schema().columns[3].check_expr.as_deref(),
            Some("introduced <> ''")
        );
        assert!(generation.resolve_schema(identity(1, 2), 0).is_err());
        assert!(generation.resolve_schema(identity(1, 3), 10).is_err());
        assert!(generation.resolve_schema(identity(1, 1), 9).is_err());
        assert!(generation.resolve_schema(identity(2, 1), 0).is_err());
        assert!(generation.tables()[1].is_dropped());
        assert_eq!(generation.tables()[2].current_name(), Some("Child"));
        assert_eq!(generation.tables()[0].history.column_high_water_mark(), 20);
        assert_eq!(generation.tables()[0].indexes[1].hnsw.ef_search, Some(0));
        assert_eq!(generation.parts().views[0].query, "SELECT id FROM Renamed");
        let clone = generation.clone();
        assert!(std::ptr::eq(generation.parts(), clone.parts()));
    }

    #[test]
    fn captured_coverage_and_reserved_ids_do_not_follow_later_parts() {
        let mut next = parts();
        let old = CatalogGeneration::try_new(next.clone()).unwrap();
        next.generation = NonZeroU64::new(2).unwrap();
        next.table_id_high_water_mark = u64::MAX;
        next.ddl_epoch_high_water_mark = u64::MAX;
        next.coverage.through_lsn = 1900;
        let new = CatalogGeneration::try_new(next).unwrap();
        assert_eq!(old.coverage().through_lsn, 650);
        assert_eq!(old.parts().table_id_high_water_mark, 32);
        assert_eq!(new.coverage().through_lsn, 1900);
        assert_eq!(new.parts().table_id_high_water_mark, u64::MAX);
        assert!(old
            .coverage()
            .captured_mutations
            .iter()
            .any(|item| item.stamp.source_lsn() > old.coverage().through_lsn));
    }

    #[test]
    fn mutation_coverage_rejects_missing_conflicting_or_unrepresented_effects() {
        for case in 0..8 {
            let mut candidate = parts();
            match case {
                0 => candidate.coverage.captured_mutations[3].stamp = stamp(3),
                1 => {
                    candidate.coverage.captured_mutations[3].stamp = DdlStamp::new(4, 300).unwrap()
                }
                2 => {
                    candidate.coverage.captured_mutations.remove(3);
                }
                3 => candidate.tables[0].names[1].stamp = DdlStamp::new(4, 401).unwrap(),
                4 => candidate.coverage.captured_mutations[0]
                    .effects
                    .push(table_effect_for(8, 1, 0)),
                5 => candidate.coverage.captured_mutations[5].effects.reverse(),
                6 => candidate.coverage.through_lsn = 2001,
                7 => candidate.ddl_epoch_high_water_mark = 12,
                _ => unreachable!(),
            }
            assert!(
                CatalogGeneration::try_new(candidate).is_err(),
                "case {case}"
            );
        }
        assert!(DdlStamp::new(0, 1).is_err());
        assert!(DdlStamp::new(1, 0).is_err());
        let mut candidate = parts();
        // Epoch reservation and WAL append order are independent. Every stamp
        // reference must still agree with the actual source LSN.
        candidate.coverage.captured_mutations[8].stamp = DdlStamp::new(9, 1101).unwrap();
        candidate.views[0].defined_at = DdlStamp::new(9, 1101).unwrap();
        assert!(CatalogGeneration::try_new(candidate).is_ok());
    }

    #[test]
    fn incarnation_and_name_history_reject_stale_work_and_reused_names() {
        for case in 0..7 {
            let mut candidate = parts();
            match case {
                0 => candidate.tables[0].incarnations[0].ended = None,
                1 => candidate.tables[0].incarnations[1].first_schema_version = 0,
                2 => {
                    candidate.tables[0].incarnations[0]
                        .ended
                        .as_mut()
                        .unwrap()
                        .kind = IncarnationEndKind::Drop
                }
                3 => candidate.tables[1].names.pop().map(|_| ()).unwrap(),
                4 => candidate.tables[2].names[0].name = Some("renamed".to_string()),
                5 => candidate.table_id_high_water_mark = 7,
                6 => candidate.tables[0].schema_events[1].identity = identity(1, 2),
                _ => unreachable!(),
            }
            assert!(
                CatalogGeneration::try_new(candidate).is_err(),
                "case {case}"
            );
        }
        let mut stale = parts();
        stale.coverage.captured_mutations[7].effects[0] = index_effect(1, 8);
        assert!(CatalogGeneration::try_new(stale).is_err());
    }

    #[test]
    fn stable_fk_bindings_survive_parent_rename_and_truncate_but_not_target_drop() {
        let mut active = parts();
        active.tables.pop();
        active.tables[1].names.pop();
        active.tables[1].incarnations[0].ended = None;
        active.coverage.captured_mutations.truncate(11);
        assert!(CatalogGeneration::try_new(active.clone()).is_ok());
        let parent = &mut active.tables[0];
        parent.incarnations[1].ended = Some(IncarnationEnd {
            stamp: stamp(14),
            kind: IncarnationEndKind::Drop,
        });
        parent.names.push(TableNameEvent {
            stamp: stamp(14),
            name: None,
        });
        parent.indexes[1].dropped_at = Some(stamp(14));
        active.coverage.ddl_epoch_cut = 14;
        active.coverage.captured_mutations.push(CatalogMutation {
            stamp: stamp(14),
            kind: DdlKind::DropTable,
            effects: vec![table_effect_for(1, 2, 10)],
        });
        let error = CatalogGeneration::try_new(active.clone())
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("catalog column reference outlives its table"),
            "{error}"
        );
        // The same target may remain in historical FK definitions after child DROP.
        active.tables[1].names.push(TableNameEvent {
            stamp: stamp(12),
            name: None,
        });
        active.tables[1].incarnations[0].ended = Some(IncarnationEnd {
            stamp: stamp(12),
            kind: IncarnationEndKind::Drop,
        });
        active.coverage.captured_mutations.insert(
            11,
            CatalogMutation {
                stamp: stamp(12),
                kind: DdlKind::DropTable,
                effects: vec![table_effect_for(5, 1, 0)],
            },
        );
        assert!(CatalogGeneration::try_new(active).is_ok());
        for case in 0..3 {
            let mut candidate = parts();
            let binding = &mut candidate.tables[1].foreign_keys[0].bindings[0];
            match case {
                0 => binding.parent_table = TableId::new(8).unwrap(),
                1 => binding.parent_column = column(9),
                2 => binding.local_column = column(1),
                _ => unreachable!(),
            }
            assert!(CatalogGeneration::try_new(candidate).is_err());
        }
    }

    #[test]
    fn index_and_view_revisions_preserve_options_and_reject_invalid_lifetimes() {
        for case in 0..7 {
            let mut candidate = parts();
            match case {
                0 => candidate.tables[0].indexes[0].dropped_at = None,
                1 => candidate.tables[0].indexes[1].hnsw.m = Some(1),
                2 => candidate.tables[0].indexes[1].hnsw.distance_metric = Some(3),
                3 => candidate.tables[0].indexes[1].columns[0] = column(2),
                4 => candidate.tables[0].indexes[0].hnsw.ef_search = Some(7),
                5 => candidate.views[0].dropped_at = None,
                6 => candidate.views[1].original_name = "Renamed".to_string(),
                _ => unreachable!(),
            }
            assert!(
                CatalogGeneration::try_new(candidate).is_err(),
                "case {case}"
            );
        }
    }

    #[test]
    fn maximum_epoch_is_not_an_open_lifetime_sentinel() {
        let mut candidate = parts();
        let maximum = DdlStamp::new(u64::MAX, u64::MAX).unwrap();
        candidate.coverage.ddl_epoch_cut = u64::MAX;
        candidate.ddl_epoch_high_water_mark = u64::MAX;
        candidate.wal_observation_ceiling = u64::MAX;
        candidate.coverage.captured_mutations.push(CatalogMutation {
            stamp: maximum,
            kind: DdlKind::CreateView,
            effects: vec![CatalogEffect::View {
                name: "z".to_string(),
                definition_epoch: u64::MAX,
            }],
        });
        candidate.views.push(CatalogView {
            original_name: "Z".to_string(),
            query: String::new(),
            defined_at: maximum,
            dropped_at: None,
        });
        assert!(CatalogGeneration::try_new(candidate.clone()).is_ok());
        candidate.views.last_mut().unwrap().original_name = "V".to_string();
        candidate
            .coverage
            .captured_mutations
            .last_mut()
            .unwrap()
            .effects[0] = CatalogEffect::View {
            name: "v".to_string(),
            definition_epoch: u64::MAX,
        };
        assert!(CatalogGeneration::try_new(candidate).is_err());
    }

    #[test]
    fn empty_catalog_preserves_exhausted_reservations_without_fabricated_lsns() {
        let candidate = CatalogParts {
            generation: NonZeroU64::new(u64::MAX).unwrap(),
            table_id_high_water_mark: u64::MAX,
            ddl_epoch_high_water_mark: u64::MAX,
            wal_observation_ceiling: 0,
            coverage: CatalogCoverage::default(),
            tables: Vec::new(),
            views: Vec::new(),
        };
        assert!(CatalogGeneration::try_new(candidate).is_ok());
    }

    fn remove_parent_column(candidate: &mut CatalogParts, name: &str, columns: Vec<ColumnId>) {
        let table = &mut candidate.tables[0];
        let first = table
            .history
            .lookup(identity(1, 2), 0)
            .unwrap()
            .schema()
            .clone();
        let mut current = table
            .history
            .lookup(identity(1, 2), 10)
            .unwrap()
            .schema()
            .clone();
        current.remove_column(name).unwrap();
        table.history = TableSchemaHistory::new(
            identity(1, 1),
            0,
            first,
            vec![column(1), column(2), column(3)],
        )
        .unwrap()
        .with_revision(identity(1, 1), 10, current, columns)
        .unwrap()
        .with_column_high_water_mark(20)
        .unwrap()
        .checked_next_incarnation()
        .unwrap();
    }

    #[test]
    fn historical_index_cannot_bridge_column_removal_or_use_a_stale_definition_schema() {
        let mut candidate = parts();
        remove_parent_column(
            &mut candidate,
            "label",
            vec![column(1), column(3), column(9)],
        );
        assert!(
            CatalogGeneration::try_new(candidate.clone()).is_err(),
            "index remained active after its column was dropped"
        );
        // Removing the index with that same ALTER is a valid half-open lifetime.
        candidate.tables[0].indexes[0].dropped_at = Some(stamp(5));
        candidate
            .coverage
            .captured_mutations
            .retain(|item| item.stamp.epoch() != 7);
        assert!(CatalogGeneration::try_new(candidate.clone()).is_ok());
        candidate.tables[0].indexes.push(CatalogIndex {
            name: "Zombie".to_string(),
            defined_at: stamp(14),
            dropped_at: Some(stamp(15)),
            schema_version: 0,
            columns: vec![column(2)],
            index_type: IndexType::Hash,
            is_unique: false,
            hnsw: CatalogHnswOptions::default(),
        });
        candidate.coverage.ddl_epoch_cut = 15;
        for (epoch, kind) in [(14, DdlKind::CreateIndex), (15, DdlKind::DropIndex)] {
            candidate.coverage.captured_mutations.push(CatalogMutation {
                stamp: stamp(epoch),
                kind,
                effects: vec![CatalogEffect::Index {
                    identity: identity(1, 2),
                    name: "zombie".to_string(),
                    definition_epoch: 14,
                }],
            });
        }
        let error = CatalogGeneration::try_new(candidate.clone())
            .unwrap_err()
            .to_string();
        assert!(error.contains("exact active schema"), "{error}");
        candidate.tables[0].indexes[2].schema_version = 10;
        assert!(
            CatalogGeneration::try_new(candidate).is_err(),
            "current schema has no such stable column"
        );
    }

    #[test]
    fn table_drop_cannot_introduce_an_implicit_index() {
        let mut candidate = parts();
        let child = &mut candidate.tables[1];
        child.indexes.push(CatalogIndex {
            name: "Late".to_string(),
            defined_at: stamp(12),
            dropped_at: Some(stamp(12)),
            schema_version: 0,
            columns: vec![column(1)],
            index_type: IndexType::Hash,
            is_unique: false,
            hnsw: CatalogHnswOptions::default(),
        });
        assert!(CatalogGeneration::try_new(candidate).is_err());
        let mut candidate = parts();
        candidate.tables[0].indexes[1].dropped_at = Some(stamp(13));
        candidate.coverage.captured_mutations[12]
            .effects
            .insert(0, table_effect_for(1, 2, 10));
        assert!(
            CatalogGeneration::try_new(candidate).is_err(),
            "CREATE TABLE cannot remove another table's index"
        );
    }

    #[test]
    fn new_fk_must_bind_the_current_owner_of_a_reused_parent_name() {
        let mut candidate = parts();
        let replacement = simple_table(
            32,
            "Parent",
            14,
            Schema::new(
                "Parent",
                vec![SchemaColumn::primary_key(0, "id", DataType::Integer)],
            ),
            vec![column(1)],
        );
        let schema = candidate.tables[1]
            .history
            .lookup(identity(5, 1), 0)
            .unwrap()
            .schema();
        let mut new_schema = schema.clone();
        new_schema.table_name = "NewChild".to_string();
        new_schema.table_name_lower = "newchild".to_string();
        let mut child = simple_table(40, "NewChild", 15, new_schema, vec![column(1), column(8)]);
        child.foreign_keys[0].bindings.push(ForeignKeyBinding {
            defined_at: stamp(15),
            local_column: column(8),
            parent_table: TableId::new(1).unwrap(),
            parent_column: column(1),
            parent_schema_version: 10,
        });
        candidate.tables.extend([replacement, child]);
        candidate.table_id_high_water_mark = 40;
        candidate.coverage.ddl_epoch_cut = 15;
        candidate.coverage.captured_mutations.extend([
            CatalogMutation {
                stamp: stamp(14),
                kind: DdlKind::CreateTable,
                effects: vec![table_effect_for(32, 1, 0)],
            },
            CatalogMutation {
                stamp: stamp(15),
                kind: DdlKind::CreateTable,
                effects: vec![table_effect_for(40, 1, 0)],
            },
        ]);
        let error = CatalogGeneration::try_new(candidate.clone())
            .unwrap_err()
            .to_string();
        assert!(error.contains("new catalog FK origin"), "{error}");
        candidate.tables[4].foreign_keys[0].bindings[0].parent_table = TableId::new(32).unwrap();
        candidate.tables[4].foreign_keys[0].bindings[0].parent_schema_version = 0;
        assert!(CatalogGeneration::try_new(candidate).is_ok());
    }

    fn child_revision(
        candidate: &mut CatalogParts,
        epoch: u64,
        version: u64,
        foreign_keys: Vec<ForeignKeyConstraint>,
        bindings: Vec<ForeignKeyBinding>,
    ) {
        let child = &mut candidate.tables[1];
        let mut schema = child
            .history
            .lookup(identity(5, 1), child.history.current_version())
            .unwrap()
            .schema()
            .clone();
        schema.foreign_keys = foreign_keys;
        child.history = child
            .history
            .with_revision(identity(5, 1), version, schema, vec![column(1), column(8)])
            .unwrap();
        child.incarnations[0].last_schema_version = version;
        child.schema_events.push(SchemaEvent {
            stamp: stamp(epoch),
            identity: identity(5, 1),
            version,
        });
        child.foreign_keys.push(SchemaForeignKeys {
            schema_version: version,
            bindings,
        });
        candidate
            .coverage
            .captured_mutations
            .iter_mut()
            .find(|item| item.stamp.epoch() == epoch)
            .unwrap()
            .effects
            .push(table_effect_for(5, 1, version));
        candidate
            .coverage
            .captured_mutations
            .iter_mut()
            .find(|item| item.stamp.epoch() == 12)
            .unwrap()
            .effects[0] = table_effect_for(5, 1, version);
    }

    #[test]
    fn carried_fk_origins_allow_rename_but_preserve_actions_and_multiplicity() {
        let base = parts();
        let mut renamed = base.tables[1]
            .history
            .lookup(identity(5, 1), 0)
            .unwrap()
            .schema()
            .foreign_keys[0]
            .clone();
        renamed.referenced_table = "renamed".to_string();
        let binding = base.tables[1].foreign_keys[0].bindings[0];
        let mut carried = base.clone();
        child_revision(&mut carried, 4, 10, vec![renamed.clone()], vec![binding]);
        assert!(CatalogGeneration::try_new(carried).is_ok());
        let mut changed = base.clone();
        let mut changed_action = renamed.clone();
        changed_action.on_update = ForeignKeyAction::Restrict;
        child_revision(&mut changed, 5, 10, vec![changed_action], vec![binding]);
        assert!(CatalogGeneration::try_new(changed.clone()).is_err());
        changed.tables[1].foreign_keys[1].bindings[0].defined_at = stamp(5);
        changed.tables[1].foreign_keys[1].bindings[0].parent_schema_version = 10;
        assert!(CatalogGeneration::try_new(changed).is_ok());
        let mut duplicated = base.clone();
        child_revision(
            &mut duplicated,
            5,
            10,
            vec![renamed.clone(), renamed],
            vec![binding, binding],
        );
        assert!(CatalogGeneration::try_new(duplicated.clone()).is_err());
        duplicated.tables[1].foreign_keys[1].bindings[1].defined_at = stamp(5);
        duplicated.tables[1].foreign_keys[1].bindings[1].parent_schema_version = 10;
        assert!(CatalogGeneration::try_new(duplicated).is_ok());
        let mut reused = base;
        child_revision(&mut reused, 4, 10, Vec::new(), Vec::new());
        let definition = reused.tables[1]
            .history
            .lookup(identity(5, 1), 0)
            .unwrap()
            .schema()
            .foreign_keys[0]
            .clone();
        child_revision(&mut reused, 5, 20, vec![definition], vec![binding]);
        assert!(CatalogGeneration::try_new(reused.clone()).is_err());
        // A real reintroduction needs a new origin and the parent's then-active metadata.
        let mut schema = reused.tables[1]
            .history
            .lookup(identity(5, 1), 20)
            .unwrap()
            .schema()
            .clone();
        schema.foreign_keys[0].referenced_table = "renamed".to_string();
        let previous = &reused.tables[1];
        let mut history = TableSchemaHistory::new(
            identity(5, 1),
            0,
            previous
                .history
                .lookup(identity(5, 1), 0)
                .unwrap()
                .schema()
                .clone(),
            vec![column(1), column(8)],
        )
        .unwrap();
        history = history
            .with_revision(
                identity(5, 1),
                10,
                previous
                    .history
                    .lookup(identity(5, 1), 10)
                    .unwrap()
                    .schema()
                    .clone(),
                vec![column(1), column(8)],
            )
            .unwrap();
        reused.tables[1].history = history
            .with_revision(identity(5, 1), 20, schema, vec![column(1), column(8)])
            .unwrap();
        reused.tables[1].foreign_keys[2].bindings[0].defined_at = stamp(5);
        reused.tables[1].foreign_keys[2].bindings[0].parent_schema_version = 10;
        assert!(CatalogGeneration::try_new(reused).is_ok());
    }

    #[test]
    fn historical_fk_cannot_bridge_a_parent_column_removal() {
        let mut candidate = parts();
        remove_parent_column(&mut candidate, "id", vec![column(2), column(3), column(9)]);
        assert!(
            CatalogGeneration::try_new(candidate).is_err(),
            "child FK stayed active until epoch12 but parent column disappeared at5"
        );
    }

    fn parent_shape_revision(candidate: &mut CatalogParts, data_type: DataType, dimensions: u16) {
        let parent = &mut candidate.tables[0];
        let mut schema = parent
            .history
            .lookup(identity(1, 2), 10)
            .unwrap()
            .schema()
            .clone();
        schema.columns[2].data_type = data_type;
        schema.columns[2].vector_dimensions = dimensions;
        schema.columns[2].nullable = false;
        schema
            .rename_column("embedding", "renamed_embedding")
            .unwrap();
        parent.history = parent
            .history
            .with_revision(
                identity(1, 2),
                20,
                schema,
                vec![column(1), column(2), column(3), column(9)],
            )
            .unwrap();
        parent.incarnations[1].last_schema_version = 20;
        parent.schema_events.push(SchemaEvent {
            stamp: stamp(14),
            identity: identity(1, 2),
            version: 20,
        });
        parent.foreign_keys.push(SchemaForeignKeys {
            schema_version: 20,
            bindings: Vec::new(),
        });
        candidate.coverage.ddl_epoch_cut = 14;
        candidate.coverage.captured_mutations.push(CatalogMutation {
            stamp: stamp(14),
            kind: DdlKind::AlterTable,
            effects: vec![table_effect_for(1, 2, 20)],
        });
    }

    #[test]
    fn hnsw_shape_changes_need_an_explicit_index_lifetime_transition() {
        let mut renamed = parts();
        parent_shape_revision(&mut renamed, DataType::Vector, 3);
        assert!(
            CatalogGeneration::try_new(renamed).is_ok(),
            "rename and nullability do not change a graph's vector shape"
        );
        for (data_type, dimensions) in [(DataType::Integer, 0), (DataType::Vector, 4)] {
            let mut changed = parts();
            parent_shape_revision(&mut changed, data_type, dimensions);
            let error = CatalogGeneration::try_new(changed.clone())
                .unwrap_err()
                .to_string();
            assert!(error.contains("HNSW vector shape changed"), "{error}");
            changed.tables[0].indexes[1].dropped_at = Some(stamp(14));
            assert!(
                CatalogGeneration::try_new(changed.clone()).is_ok(),
                "the old graph ends before the new schema"
            );
            if data_type == DataType::Vector {
                let mut replacement = changed.tables[0].indexes[1].clone();
                replacement.defined_at = stamp(14);
                replacement.dropped_at = None;
                replacement.schema_version = 20;
                changed.tables[0].indexes.push(replacement);
                assert!(
                    CatalogGeneration::try_new(changed).is_ok(),
                    "a separately represented replacement binds the new dimension"
                );
            }
        }
    }
}
