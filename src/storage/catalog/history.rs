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

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::core::{DataType, Error, Result, Schema, SchemaColumn, Value};

use super::identity::{ColumnId, TableIdentity};

/// One immutable schema and its ordered stable column identities.
/// Version zero is valid: legacy volume schemas may start at zero.
#[derive(Debug)]
pub struct SchemaRevision {
    version: u64,
    schema: Arc<Schema>,
    columns: Box<[ColumnId]>,
}

impl SchemaRevision {
    fn new(version: u64, schema: Schema, columns: Vec<ColumnId>) -> Result<Self> {
        if columns.len() != schema.columns.len() {
            return Err(Error::internal(
                "catalog column identity count does not match schema",
            ));
        }
        if schema.table_name_lower != schema.table_name.to_lowercase() {
            return Err(Error::internal("catalog table name cache is inconsistent"));
        }
        let mut ids = BTreeSet::new();
        let mut names = BTreeSet::new();
        for (position, (column, id)) in schema.columns.iter().zip(&columns).enumerate() {
            if column.id != position {
                return Err(Error::internal(
                    "catalog schema column ID must remain positional",
                ));
            }
            if column.name_lower != column.name.to_lowercase() {
                return Err(Error::internal("catalog column name cache is inconsistent"));
            }
            if !ids.insert(*id) || !names.insert(column.name_lower.as_str()) {
                return Err(Error::internal("duplicate catalog column identity or name"));
            }
            if column.data_type != DataType::Vector && column.vector_dimensions != 0 {
                return Err(Error::internal(
                    "non-vector catalog column has vector dimensions",
                ));
            }
        }
        for foreign_key in &schema.foreign_keys {
            if schema
                .columns
                .get(foreign_key.column_index)
                .is_none_or(|column| column.name_lower != foreign_key.column_name.to_lowercase())
            {
                return Err(Error::internal(
                    "catalog foreign key has an invalid local column",
                ));
            }
        }
        Ok(Self {
            version,
            schema: Arc::new(schema),
            columns: columns.into_boxed_slice(),
        })
    }

    pub fn version(&self) -> u64 {
        self.version
    }
    pub fn schema(&self) -> &Schema {
        &self.schema
    }
    pub fn column_ids(&self) -> &[ColumnId] {
        &self.columns
    }

    /// Translate an existing schema/index/FK position without changing its meaning.
    pub fn column_id_at(&self, position: usize) -> Result<ColumnId> {
        self.columns
            .get(position)
            .copied()
            .ok_or_else(|| Error::internal("catalog column position is out of range"))
    }

    pub fn column_position(&self, id: ColumnId) -> Result<usize> {
        self.columns
            .iter()
            .position(|candidate| *candidate == id)
            .ok_or_else(|| Error::internal("column identity is absent from catalog schema version"))
    }
}

#[derive(Clone, Copy, Debug)]
struct ColumnOrigin {
    version: u64,
    position: usize,
}

/// Immutable history for exactly one table incarnation. Clones and incarnation
/// transitions share schema history; adding a revision creates a new snapshot.
/// There is deliberately no history-pruning API before WAL/volume references
/// can prove that a schema version is no longer needed.
#[derive(Clone, Debug)]
pub struct TableSchemaHistory {
    identity: TableIdentity,
    current_version: u64,
    revisions: Arc<BTreeMap<u64, Arc<SchemaRevision>>>,
    origins: Arc<BTreeMap<ColumnId, ColumnOrigin>>,
    column_high_water_mark: u64,
}

impl TableSchemaHistory {
    pub fn new(
        identity: TableIdentity,
        version: u64,
        schema: Schema,
        columns: Vec<ColumnId>,
    ) -> Result<Self> {
        let revision = Arc::new(SchemaRevision::new(version, schema, columns)?);
        let origins = revision
            .columns
            .iter()
            .enumerate()
            .map(|(position, &id)| (id, ColumnOrigin { version, position }))
            .collect();
        let column_high_water_mark = revision
            .columns
            .iter()
            .map(|id| id.get())
            .max()
            .unwrap_or(0);
        Ok(Self {
            identity,
            current_version: version,
            revisions: Arc::new(BTreeMap::from([(version, revision)])),
            origins: Arc::new(origins),
            column_high_water_mark,
        })
    }

    pub fn identity(&self) -> TableIdentity {
        self.identity
    }
    pub fn current_version(&self) -> u64 {
        self.current_version
    }
    pub fn column_high_water_mark(&self) -> u64 {
        self.column_high_water_mark
    }

    /// Retain IDs reserved by failed DDL as well as currently represented IDs.
    pub fn with_column_high_water_mark(&self, high_water_mark: u64) -> Result<Self> {
        if high_water_mark < self.column_high_water_mark {
            return Err(Error::internal(
                "catalog column allocation high-water mark cannot decrease",
            ));
        }
        let mut next = self.clone();
        next.column_high_water_mark = high_water_mark;
        Ok(next)
    }

    /// A metadata transition only: it neither executes TRUNCATE nor publishes
    /// its outcome. Column identities and their allocation space are preserved.
    pub fn checked_next_incarnation(&self) -> Result<Self> {
        let mut next = self.clone();
        next.identity = self.identity.checked_next_incarnation()?;
        Ok(next)
    }

    fn check_identity(&self, identity: TableIdentity) -> Result<()> {
        if identity != self.identity {
            return Err(Error::internal(
                "catalog table identity or incarnation mismatch",
            ));
        }
        Ok(())
    }

    pub fn lookup(&self, identity: TableIdentity, version: u64) -> Result<&SchemaRevision> {
        self.check_identity(identity)?;
        self.lookup_version(version)
    }

    /// Ordered, borrowed schema metadata for catalog validation and encoding.
    pub fn revisions(
        &self,
    ) -> impl ExactSizeIterator<Item = &SchemaRevision> + DoubleEndedIterator {
        self.revisions.values().map(Arc::as_ref)
    }

    // Only the catalog may authorize a historical incarnation before looking
    // up its exact version in this shared TableId history.
    pub(super) fn lookup_version(&self, version: u64) -> Result<&SchemaRevision> {
        self.revisions
            .get(&version)
            .map(Arc::as_ref)
            .ok_or_else(|| Error::internal("catalog schema version is unavailable"))
    }

    pub fn with_revision(
        &self,
        identity: TableIdentity,
        version: u64,
        schema: Schema,
        columns: Vec<ColumnId>,
    ) -> Result<Self> {
        self.check_identity(identity)?;
        if version <= self.current_version {
            return Err(Error::internal("catalog schema version must increase"));
        }
        let revision = Arc::new(SchemaRevision::new(version, schema, columns)?);
        let current = self.lookup(identity, self.current_version)?;
        let live: BTreeSet<_> = current.columns.iter().copied().collect();
        for &id in &revision.columns {
            if !live.contains(&id) && id.get() <= self.column_high_water_mark {
                return Err(Error::internal(
                    "catalog column identity was retired or already reserved",
                ));
            }
        }
        let mut next = self.clone();
        for (position, &id) in revision.columns.iter().enumerate() {
            if !live.contains(&id) {
                Arc::make_mut(&mut next.origins).insert(id, ColumnOrigin { version, position });
                next.column_high_water_mark = next.column_high_water_mark.max(id.get());
            }
        }
        Arc::make_mut(&mut next.revisions).insert(version, revision);
        next.current_version = version;
        Ok(next)
    }

    /// Compile an old-to-current projection once per source schema. Names are
    /// never used to match columns, and this method never evaluates default SQL.
    pub fn projection(
        &self,
        identity: TableIdentity,
        source_version: u64,
    ) -> Result<ProjectionPlan> {
        self.lookup(identity, source_version)?;
        let source = Arc::clone(&self.revisions[&source_version]);
        let target = Arc::clone(&self.revisions[&self.current_version]);
        let positions: BTreeMap<_, _> = source
            .columns
            .iter()
            .enumerate()
            .map(|(position, &id)| (id, position))
            .collect();
        let mut slots = Vec::with_capacity(target.columns.len());
        let mut not_null = Vec::new();
        for (target_position, &id) in target.columns.iter().enumerate() {
            let target_column = &target.schema.columns[target_position];
            if let Some(&position) = positions.get(&id) {
                let source_column = &source.schema.columns[position];
                compatible_types(source_column, target_column)?;
                if source_column.nullable && (!target_column.nullable || target_column.primary_key)
                {
                    not_null.push(position);
                }
                slots.push(ProjectionSource::Input(position));
            } else {
                let origin = self
                    .origins
                    .get(&id)
                    .ok_or_else(|| Error::internal("catalog column introduction is unavailable"))?;
                let definition = &self.revisions[&origin.version].schema.columns[origin.position];
                compatible_types(definition, target_column)?;
                let value = match &definition.default_value {
                    Some(value) => value.clone(),
                    None if definition.default_expr.is_none()
                        && definition.nullable
                        && !definition.primary_key =>
                    {
                        Value::Null(definition.data_type)
                    }
                    None => {
                        return Err(Error::internal(
                            "historical column requires a recorded introduction default",
                        ))
                    }
                };
                validate_value(&value, target_column)?;
                slots.push(ProjectionSource::Default(value));
            }
        }
        Ok(ProjectionPlan {
            identity,
            source,
            target,
            slots: slots.into_boxed_slice(),
            not_null: not_null.into_boxed_slice(),
        })
    }
}

fn compatible_types(source: &SchemaColumn, target: &SchemaColumn) -> Result<()> {
    if source.data_type != target.data_type || source.vector_dimensions != target.vector_dimensions
    {
        return Err(Error::internal(
            "catalog projection requires an explicit type migration",
        ));
    }
    Ok(())
}

fn validate_value(value: &Value, column: &SchemaColumn) -> Result<()> {
    if value.is_null() {
        if matches!(value, Value::Null(data_type) if *data_type != DataType::Null && *data_type != column.data_type)
        {
            return Err(Error::internal(
                "typed NULL does not match catalog schema version",
            ));
        }
        if column.nullable && !column.primary_key {
            return Ok(());
        }
        return Err(Error::internal("NULL in a nonnullable catalog column"));
    }
    if value.data_type() != column.data_type {
        return Err(Error::internal(
            "row value type does not match catalog schema version",
        ));
    }
    if column.data_type == DataType::Vector {
        let Value::Extension(bytes) = value else {
            return Err(Error::internal(
                "invalid vector representation in catalog row",
            ));
        };
        let Some(payload) = bytes.len().checked_sub(1) else {
            return Err(Error::internal(
                "invalid vector representation in catalog row",
            ));
        };
        if payload % 4 != 0
            || (column.vector_dimensions != 0
                && payload / 4 != usize::from(column.vector_dimensions))
        {
            return Err(Error::internal(
                "row vector dimensions do not match catalog schema version",
            ));
        }
    }
    Ok(())
}

#[derive(Debug)]
enum ProjectionSource {
    Input(usize),
    Default(Value),
}

/// Reusable mapping. Preparing it may allocate metadata; applying it and
/// iterating its borrowed values do not allocate or clone row/default payloads.
#[derive(Debug)]
pub struct ProjectionPlan {
    identity: TableIdentity,
    source: Arc<SchemaRevision>,
    target: Arc<SchemaRevision>,
    slots: Box<[ProjectionSource]>,
    not_null: Box<[usize]>,
}

impl ProjectionPlan {
    pub fn identity(&self) -> TableIdentity {
        self.identity
    }
    pub fn source_version(&self) -> u64 {
        self.source.version
    }
    pub fn target_version(&self) -> u64 {
        self.target.version
    }

    pub fn project<'a>(
        &'a self,
        identity: TableIdentity,
        source_version: u64,
        values: &'a [Value],
    ) -> Result<ProjectedRow<'a>> {
        if identity != self.identity || source_version != self.source.version {
            return Err(Error::internal(
                "row identity or schema version does not match projection",
            ));
        }
        if values.len() != self.source.columns.len() {
            return Err(Error::internal(
                "row width does not match catalog schema version",
            ));
        }
        for (value, column) in values.iter().zip(&self.source.schema.columns) {
            validate_value(value, column)?;
        }
        for &position in &self.not_null {
            if values[position].is_null() {
                return Err(Error::internal(
                    "NULL cannot satisfy the target catalog schema",
                ));
            }
        }
        Ok(ProjectedRow {
            values,
            slots: &self.slots,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ProjectedRow<'a> {
    values: &'a [Value],
    slots: &'a [ProjectionSource],
}

impl<'a> ProjectedRow<'a> {
    pub fn len(&self) -> usize {
        self.slots.len()
    }
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    pub fn get(&self, position: usize) -> Option<&'a Value> {
        self.slots.get(position).map(|slot| match slot {
            ProjectionSource::Input(position) => &self.values[*position],
            ProjectionSource::Default(value) => value,
        })
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = &'a Value> + DoubleEndedIterator + 'a {
        let values = self.values;
        self.slots.iter().map(move |slot| match slot {
            ProjectionSource::Input(position) => &values[*position],
            ProjectionSource::Default(value) => value,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::catalog::identity::{Incarnation, TableId};

    fn identity() -> TableIdentity {
        TableIdentity::new(TableId::new(1).unwrap(), Incarnation::FIRST)
    }

    fn ids(values: &[u64]) -> Vec<ColumnId> {
        values
            .iter()
            .map(|&value| ColumnId::new(value).unwrap())
            .collect()
    }

    fn schema(name: &str, columns: &[(&str, DataType, Option<Value>)]) -> Schema {
        Schema::new(
            name,
            columns
                .iter()
                .enumerate()
                .map(|(index, (name, data_type, value))| {
                    SchemaColumn::with_default_value(
                        index,
                        *name,
                        *data_type,
                        true,
                        false,
                        false,
                        None,
                        value.clone(),
                        None,
                    )
                })
                .collect(),
        )
    }

    #[test]
    fn rename_reorder_drop_and_readd_use_stable_ids() {
        let first = TableSchemaHistory::new(
            identity(),
            0,
            schema(
                "t",
                &[
                    ("id", DataType::Integer, None),
                    ("name", DataType::Text, None),
                ],
            ),
            ids(&[1, 2]),
        )
        .unwrap();
        let renamed = first
            .with_revision(
                identity(),
                1,
                schema(
                    "renamed",
                    &[
                        ("label", DataType::Text, None),
                        ("id", DataType::Integer, None),
                    ],
                ),
                ids(&[2, 1]),
            )
            .unwrap();
        let input = [Value::Integer(7), Value::text("original")];
        let plan = renamed.projection(identity(), 0).unwrap();
        let projected = plan.project(identity(), 0, &input).unwrap();
        assert!(std::ptr::eq(projected.get(0).unwrap(), &input[1]));
        assert!(std::ptr::eq(projected.get(1).unwrap(), &input[0]));
        assert_eq!(projected.iter().len(), 2);
        assert!(std::ptr::eq(
            projected.iter().next_back().unwrap(),
            &input[0]
        ));
        assert_eq!(
            first.lookup(identity(), 0).unwrap().schema().table_name,
            "t"
        );
        assert!(
            first.lookup(identity(), 1).is_err(),
            "old snapshots remain immutable"
        );
        let dropped = renamed
            .with_revision(
                identity(),
                2,
                schema("renamed", &[("id", DataType::Integer, None)]),
                ids(&[1]),
            )
            .unwrap();
        assert!(dropped
            .with_revision(
                identity(),
                3,
                schema(
                    "renamed",
                    &[
                        ("id", DataType::Integer, None),
                        ("label", DataType::Text, Some(Value::text("new"))),
                    ]
                ),
                ids(&[1, 2])
            )
            .is_err());
        let readded = dropped
            .with_revision(
                identity(),
                3,
                schema(
                    "renamed",
                    &[
                        ("id", DataType::Integer, None),
                        ("label", DataType::Text, Some(Value::text("new"))),
                    ],
                ),
                ids(&[1, 3]),
            )
            .unwrap();
        let plan = readded.projection(identity(), 0).unwrap();
        let projected = plan.project(identity(), 0, &input).unwrap();
        assert_eq!(projected.get(1), Some(&Value::text("new")));
        assert!(!std::ptr::eq(projected.get(1).unwrap(), &input[1]));
        assert_eq!(readded.column_high_water_mark(), 3);
        assert_eq!(
            readded
                .lookup(identity(), 3)
                .unwrap()
                .column_id_at(1)
                .unwrap()
                .get(),
            3
        );
    }

    #[test]
    fn projection_uses_introduction_default_not_later_insert_default() {
        let first = TableSchemaHistory::new(
            identity(),
            5,
            schema("t", &[("id", DataType::Integer, None)]),
            ids(&[1]),
        )
        .unwrap();
        let added = first
            .with_revision(
                identity(),
                8,
                schema(
                    "t",
                    &[
                        ("id", DataType::Integer, None),
                        ("n", DataType::Integer, Some(Value::Integer(7))),
                    ],
                ),
                ids(&[1, 2]),
            )
            .unwrap();
        let changed = added
            .with_revision(
                identity(),
                12,
                schema(
                    "t",
                    &[
                        ("id", DataType::Integer, None),
                        ("n", DataType::Integer, Some(Value::Integer(99))),
                    ],
                ),
                ids(&[1, 2]),
            )
            .unwrap();
        let plan = changed.projection(identity(), 5).unwrap();
        let input = [Value::Integer(1)];
        let first_row = plan.project(identity(), 5, &input).unwrap();
        let second_row = plan.project(identity(), 5, &input).unwrap();
        assert_eq!(first_row.get(1), Some(&Value::Integer(7)));
        assert!(std::ptr::eq(
            first_row.get(1).unwrap(),
            second_row.get(1).unwrap()
        ));
        let stored = [Value::Integer(1), Value::Integer(13)];
        let newer_plan = changed.projection(identity(), 8).unwrap();
        assert_eq!(
            newer_plan.project(identity(), 8, &stored).unwrap().get(1),
            Some(&stored[1])
        );
        assert!(
            changed.lookup(identity(), 9).is_err(),
            "lookup must not choose a nearby version"
        );
    }

    #[test]
    fn incarnation_transition_shares_history_without_resetting_column_space() {
        let history = TableSchemaHistory::new(
            identity(),
            0,
            schema("t", &[("id", DataType::Integer, None)]),
            ids(&[10]),
        )
        .unwrap()
        .with_column_high_water_mark(20)
        .unwrap();
        let next = history.checked_next_incarnation().unwrap();
        assert_eq!(next.identity().table_id, identity().table_id);
        assert_eq!(next.identity().incarnation.get(), 2);
        assert!(Arc::ptr_eq(&history.revisions, &next.revisions));
        assert!(Arc::ptr_eq(&history.origins, &next.origins));
        assert_eq!(next.column_high_water_mark(), 20);
        assert!(next.lookup(identity(), 0).is_err());
        assert!(history.lookup(next.identity(), 0).is_err());
        assert!(next.with_column_high_water_mark(19).is_err());
        assert!(next
            .with_revision(
                next.identity(),
                1,
                schema(
                    "t",
                    &[
                        ("id", DataType::Integer, None),
                        ("n", DataType::Integer, None),
                    ]
                ),
                ids(&[10, 20])
            )
            .is_err());
        assert!(next
            .with_revision(
                next.identity(),
                1,
                schema(
                    "t",
                    &[
                        ("id", DataType::Integer, None),
                        ("n", DataType::Integer, None),
                    ]
                ),
                ids(&[10, 21])
            )
            .is_ok());
        let recreated = TableIdentity::new(TableId::new(2).unwrap(), Incarnation::FIRST);
        assert!(history.lookup(recreated, 0).is_err());
        let max = TableSchemaHistory::new(
            TableIdentity::new(
                TableId::new(1).unwrap(),
                Incarnation::new(u64::MAX).unwrap(),
            ),
            0,
            schema("t", &[]),
            vec![],
        )
        .unwrap();
        assert!(max.checked_next_incarnation().is_err());
    }

    #[test]
    fn projection_rejects_wrong_identity_width_type_and_tightened_nullability() {
        let history = TableSchemaHistory::new(
            identity(),
            0,
            schema("t", &[("n", DataType::Integer, None)]),
            ids(&[1]),
        )
        .unwrap();
        let plan = history.projection(identity(), 0).unwrap();
        assert!(plan.project(identity(), 0, &[]).is_err());
        assert!(plan
            .project(identity(), 0, &[Value::Integer(1), Value::Integer(2)])
            .is_err());
        assert!(plan.project(identity(), 1, &[Value::Integer(1)]).is_err());
        assert!(plan
            .project(
                identity().checked_next_incarnation().unwrap(),
                0,
                &[Value::Integer(1)]
            )
            .is_err());
        assert!(plan.project(identity(), 0, &[Value::text("1")]).is_err());
        assert!(plan
            .project(identity(), 0, &[Value::Null(DataType::Text)])
            .is_err());
        let mut tightened = schema("t", &[("n", DataType::Integer, None)]);
        tightened.columns[0].nullable = false;
        let next = history
            .with_revision(identity(), 1, tightened, ids(&[1]))
            .unwrap();
        let plan = next.projection(identity(), 0).unwrap();
        assert!(plan
            .project(identity(), 0, &[Value::Null(DataType::Integer)])
            .is_err());
        assert!(plan.project(identity(), 0, &[Value::Integer(1)]).is_ok());
        let changed = history
            .with_revision(
                identity(),
                1,
                schema("t", &[("n", DataType::Float, None)]),
                ids(&[1]),
            )
            .unwrap();
        assert!(changed.projection(identity(), 0).is_err());
    }

    #[test]
    fn missing_default_is_never_evaluated_or_invented() {
        let first = TableSchemaHistory::new(identity(), 0, schema("t", &[]), vec![]).unwrap();
        let mut definition = schema("t", &[("n", DataType::Integer, None)]);
        definition.columns[0].nullable = false;
        let next = first
            .with_revision(identity(), 1, definition.clone(), ids(&[1]))
            .unwrap();
        assert!(next.projection(identity(), 0).is_err());
        definition.columns[0].nullable = true;
        definition.columns[0].default_expr = Some("RANDOM()".into());
        let next = first
            .with_revision(identity(), 1, definition.clone(), ids(&[1]))
            .unwrap();
        assert!(next.projection(identity(), 0).is_err());
        definition.columns[0].default_value = Some(Value::Integer(17));
        let next = first
            .with_revision(identity(), 1, definition, ids(&[1]))
            .unwrap();
        assert_eq!(
            next.projection(identity(), 0)
                .unwrap()
                .project(identity(), 0, &[])
                .unwrap()
                .get(0),
            Some(&Value::Integer(17))
        );
        let nullable = first
            .with_revision(
                identity(),
                1,
                schema("t", &[("n", DataType::Integer, None)]),
                ids(&[1]),
            )
            .unwrap();
        assert_eq!(
            nullable
                .projection(identity(), 0)
                .unwrap()
                .project(identity(), 0, &[])
                .unwrap()
                .get(0),
            Some(&Value::Null(DataType::Integer))
        );
    }

    #[test]
    fn vector_shape_and_schema_metadata_fail_closed() {
        let mut vector = schema("t", &[("v", DataType::Vector, None)]);
        vector.columns[0].vector_dimensions = 2;
        let history = TableSchemaHistory::new(identity(), 0, vector.clone(), ids(&[1])).unwrap();
        let plan = history.projection(identity(), 0).unwrap();
        assert!(plan
            .project(identity(), 0, &[Value::vector(vec![1.0, 2.0])])
            .is_ok());
        assert!(plan
            .project(identity(), 0, &[Value::vector(vec![1.0])])
            .is_err());
        vector.columns[0].vector_dimensions = 3;
        assert!(history
            .with_revision(identity(), 1, vector, ids(&[1]))
            .unwrap()
            .projection(identity(), 0)
            .is_err());
        let two = schema(
            "t",
            &[
                ("a", DataType::Integer, None),
                ("b", DataType::Integer, None),
            ],
        );
        assert!(TableSchemaHistory::new(identity(), 0, two.clone(), ids(&[1])).is_err());
        assert!(TableSchemaHistory::new(identity(), 0, two.clone(), ids(&[1, 1])).is_err());
        let mut invalid = two;
        invalid.columns[1].id = 10;
        assert!(TableSchemaHistory::new(identity(), 0, invalid, ids(&[1, 2])).is_err());
        assert!(history
            .with_revision(identity(), 0, schema("t", &[]), vec![])
            .is_err());
        let empty =
            TableSchemaHistory::new(identity(), u64::MAX, schema("t", &[]), vec![]).unwrap();
        assert!(empty
            .with_revision(identity(), 0, schema("t", &[]), vec![])
            .is_err());
    }
}
