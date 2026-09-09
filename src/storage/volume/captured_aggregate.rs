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

//! Typed aggregation over one captured hot/cold view. Visibility is resolved
//! before predicates or column reads, and only one cold volume is active.

use std::cell::OnceCell;
use std::cmp::Ordering;
use std::sync::Arc;

use ahash::AHashMap;
use chrono::{DateTime, Utc};
use rustc_hash::FxHashSet;
use smallvec::SmallVec;

use crate::common::SmartString;
use crate::core::{DataType, Error, Operator, Result, Row, Schema, Value};
use crate::storage::expression::Expression;
use crate::storage::mvcc::version_store::{AggregateOp, CapturedHotView, GroupedAggregateResult};

use super::column::{ColumnData, DictFilter, ROW_GROUP_SIZE};
use super::manifest::{ColdGeneration, ColdSegment};
use super::writer::ColSource;

type ColdInput<'a> = Option<(&'a ColdGeneration, &'a FxHashSet<i64>)>;

// Reuse bounded scratch across groups without growing with input cardinality.
const DICTIONARY_WINDOW: usize = 1024;
const DICTIONARY_PREFIX: usize = 4;

/// A borrowed scalar, never a reconstructed Row or a cloned hot payload.
#[derive(Clone, Copy)]
enum Cell<'a> {
    Null,
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Timestamp(DateTime<Utc>),
    Text(&'a SmartString),
    Other(DataType),
}

impl<'a> Cell<'a> {
    fn from_value(value: &'a Value) -> Self {
        match value {
            Value::Null(_) => Self::Null,
            Value::Integer(v) => Self::Integer(*v),
            Value::Float(v) => Self::Float(*v),
            Value::Boolean(v) => Self::Boolean(*v),
            Value::Timestamp(v) => Self::Timestamp(*v),
            Value::Text(v) => Self::Text(v),
            Value::Extension(_) => Self::Other(value.data_type()),
        }
    }

    #[inline]
    fn from_column(column: &'a ColumnData, index: usize) -> Self {
        if column.is_null(index) {
            return Self::Null;
        }
        match column {
            ColumnData::Int64 { values, .. } => Self::Integer(values[index]),
            ColumnData::Float64 { values, .. } => Self::Float(values[index]),
            ColumnData::Boolean { values, .. } => Self::Boolean(values[index]),
            ColumnData::TimestampNanos { values, .. } => {
                Self::Timestamp(DateTime::from_timestamp_nanos(values[index]))
            }
            ColumnData::Dictionary {
                ids, dictionary, ..
            } => Self::Text(&dictionary[ids[index] as usize]),
            ColumnData::Bytes { ext_type, .. } => Self::Other(*ext_type),
        }
    }

    fn compare(self, other: Self) -> Option<Ordering> {
        match (self, other) {
            (Self::Integer(a), Self::Integer(b)) => Some(a.cmp(&b)),
            (Self::Float(a), Self::Float(b)) => a.partial_cmp(&b),
            (Self::Integer(a), Self::Float(b)) => crate::core::value::cmp_i64_f64(a, b),
            (Self::Float(a), Self::Integer(b)) => {
                crate::core::value::cmp_i64_f64(b, a).map(Ordering::reverse)
            }
            (Self::Boolean(a), Self::Boolean(b)) => Some(a.cmp(&b)),
            (Self::Timestamp(a), Self::Timestamp(b)) => Some(a.cmp(&b)),
            (Self::Text(a), Self::Text(b)) => Some(a.cmp(b)),
            _ => None,
        }
    }

    fn data_type(self) -> DataType {
        match self {
            Self::Null => DataType::Null,
            Self::Integer(_) => DataType::Integer,
            Self::Float(_) => DataType::Float,
            Self::Boolean(_) => DataType::Boolean,
            Self::Timestamp(_) => DataType::Timestamp,
            Self::Text(_) => DataType::Text,
            Self::Other(data_type) => data_type,
        }
    }
}

struct Predicate<'a> {
    column: usize,
    operator: Operator,
    target: Cell<'a>,
}

struct DictionaryPredicate {
    physical: usize,
    target: Option<u32>,
}

impl Predicate<'_> {
    fn bind_dictionary(&self, columns: &GroupColumns<'_>) -> Option<DictionaryPredicate> {
        if !matches!(self.operator, Operator::Eq | Operator::Ne) {
            return None;
        }
        let Cell::Text(target) = self.target else {
            return None;
        };
        let ColSource::Volume(physical) = columns.segment.mapping.sources[self.column] else {
            return None;
        };
        let (column, _) = columns.columns[physical].get()?.data();
        let ColumnData::Dictionary { dictionary, .. } = column else {
            return None;
        };
        Some(DictionaryPredicate {
            physical,
            target: dictionary
                .iter()
                .position(|value| value == target)
                .map(|index| index as u32),
        })
    }

    #[inline]
    fn matches_cold(
        &self,
        columns: &GroupColumns<'_>,
        local: usize,
        dictionary: Option<&DictionaryPredicate>,
    ) -> Result<bool> {
        if let Some(predicate) = dictionary {
            let (column, offset) = columns.bound_column(predicate.physical);
            if let ColumnData::Dictionary { ids, nulls, .. } = column {
                if nulls[offset + local] {
                    return Ok(false);
                }
                let equal = Some(ids[offset + local]) == predicate.target;
                return Ok(if self.operator == Operator::Eq {
                    equal
                } else {
                    !equal
                });
            }
        }
        self.matches(columns.get(self.column, local))
    }

    #[inline]
    fn matches(&self, value: Cell<'_>) -> Result<bool> {
        if matches!(self.operator, Operator::IsNull) {
            return Ok(matches!(value, Cell::Null));
        }
        if matches!(self.operator, Operator::IsNotNull) {
            return Ok(!matches!(value, Cell::Null));
        }
        if matches!(value, Cell::Null) || matches!(self.target, Cell::Null) {
            return Ok(false);
        }
        // Equality can reject different string lengths without ordering or
        // scanning their shared prefix. Keep ordered and mixed-type semantics
        // in the common comparison path below.
        if let (Cell::Text(value), Cell::Text(target)) = (value, self.target) {
            match self.operator {
                Operator::Eq => return Ok(value == target),
                Operator::Ne => return Ok(value != target),
                _ => (),
            }
        }
        let order = value.compare(self.target);
        if order.is_none()
            && !matches!(
                (value, self.target),
                (
                    Cell::Integer(_) | Cell::Float(_),
                    Cell::Integer(_) | Cell::Float(_)
                )
            )
        {
            return Err(Error::type_conversion(
                format!("{:?}", value.data_type()),
                format!("{:?}", self.target.data_type()),
            ));
        }
        Ok(match self.operator {
            Operator::Eq => order == Some(Ordering::Equal),
            Operator::Ne => order != Some(Ordering::Equal),
            Operator::Lt => order == Some(Ordering::Less),
            Operator::Lte => matches!(order, Some(Ordering::Less | Ordering::Equal)),
            Operator::Gt => order == Some(Ordering::Greater),
            Operator::Gte => matches!(order, Some(Ordering::Greater | Ordering::Equal)),
            _ => unreachable!("preflight rejects unsupported operators"),
        })
    }
}

fn scalar_type(data_type: DataType) -> bool {
    matches!(
        data_type,
        DataType::Integer
            | DataType::Float
            | DataType::Boolean
            | DataType::Timestamp
            | DataType::Text
    )
}

/// Inspect every leaf. collect_comparisons alone can silently omit an IN or
/// other conjunct and must not be used as proof that the whole filter is bound.
fn predicates<'a>(
    expression: &'a dyn Expression,
    schema: &Schema,
    output: &mut Vec<Predicate<'a>>,
) -> bool {
    if let Some(children) = expression.get_and_operands() {
        return children
            .iter()
            .all(|child| predicates(&**child, schema, output));
    }
    let Some((name, operator, target)) = expression.get_comparison_info() else {
        return false;
    };
    if !matches!(
        operator,
        Operator::Eq
            | Operator::Ne
            | Operator::Lt
            | Operator::Lte
            | Operator::Gt
            | Operator::Gte
            | Operator::IsNull
            | Operator::IsNotNull
    ) {
        return false;
    }
    let Some(&column) = schema.column_index_map().get(&name.to_lowercase()) else {
        return false;
    };
    let data_type = schema.columns[column].data_type;
    if data_type == DataType::Boolean
        && !matches!(
            operator,
            Operator::Eq | Operator::Ne | Operator::IsNull | Operator::IsNotNull
        )
    {
        return false;
    }
    if !matches!(operator, Operator::IsNull | Operator::IsNotNull)
        && !target.is_null()
        && !(scalar_type(data_type)
            && (data_type == target.data_type()
                || matches!(
                    (data_type, target.data_type()),
                    (DataType::Integer, DataType::Float) | (DataType::Float, DataType::Integer)
                )))
    {
        return false;
    }
    output.push(Predicate {
        column,
        operator,
        target: Cell::from_value(target),
    });
    true
}

#[derive(Clone)]
enum Extremum {
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Timestamp(DateTime<Utc>),
    Text(SmartString),
}

impl Extremum {
    fn cell(&self) -> Cell<'_> {
        match self {
            Self::Integer(v) => Cell::Integer(*v),
            Self::Float(v) => Cell::Float(*v),
            Self::Boolean(v) => Cell::Boolean(*v),
            Self::Timestamp(v) => Cell::Timestamp(*v),
            Self::Text(v) => Cell::Text(v),
        }
    }
    fn from_cell(cell: Cell<'_>) -> Option<Self> {
        Some(match cell {
            Cell::Integer(v) => Self::Integer(v),
            Cell::Float(v) if !v.is_nan() => Self::Float(v),
            Cell::Boolean(v) => Self::Boolean(v),
            Cell::Timestamp(v) => Self::Timestamp(v),
            Cell::Text(v) => Self::Text(v.clone()),
            _ => return None,
        })
    }
    fn into_value(self) -> Value {
        match self {
            Self::Integer(v) => Value::Integer(v),
            Self::Float(v) => Value::Float(v),
            Self::Boolean(v) => Value::Boolean(v),
            Self::Timestamp(v) => Value::Timestamp(v),
            Self::Text(v) => Value::Text(v),
        }
    }
}

/// Widened integer sum, separate floating sum, and NULL/NaN numeric exclusion.
/// Grouped SUM preserves the SQL grouped path's Float result across hot and
/// cold layouts. Filtered global SUM retains its integer narrowing behavior.
#[derive(Default)]
struct Accumulator {
    count: i64,
    integer: i128,
    float: f64,
    saw_float: bool,
    extremum: Option<Extremum>,
}

impl Accumulator {
    #[inline(always)]
    fn add(&mut self, operation: AggregateOp, cell: Cell<'_>) {
        match operation {
            AggregateOp::CountStar => self.count += 1,
            AggregateOp::Count => self.count += i64::from(!matches!(cell, Cell::Null)),
            AggregateOp::Sum | AggregateOp::Avg => match cell {
                Cell::Integer(v) => {
                    self.integer += i128::from(v);
                    self.count += 1;
                }
                Cell::Boolean(v) => {
                    self.integer += i128::from(v);
                    self.count += 1;
                }
                Cell::Float(v) if !v.is_nan() => {
                    self.float += v;
                    self.count += 1;
                    self.saw_float = true;
                }
                _ => (),
            },
            AggregateOp::Min | AggregateOp::Max => self.add_extremum(operation, cell),
        }
    }

    // Keep the larger comparison/owned-text path out of numeric row loops.
    #[inline(never)]
    fn add_extremum(&mut self, operation: AggregateOp, cell: Cell<'_>) {
        let replace = self.extremum.as_ref().is_none_or(|current| {
            cell.compare(current.cell()).is_some_and(|order| {
                if operation == AggregateOp::Min {
                    order.is_lt()
                } else {
                    order.is_gt()
                }
            })
        });
        if replace {
            if let Some(value) = Extremum::from_cell(cell) {
                self.extremum = Some(value);
            }
        }
    }
    fn finish(self, operation: AggregateOp, data_type: DataType, grouped: bool) -> Value {
        match operation {
            AggregateOp::Count | AggregateOp::CountStar => Value::Integer(self.count),
            AggregateOp::Sum if self.count != 0 => {
                if !grouped && !self.saw_float {
                    if let Ok(integer) = i64::try_from(self.integer) {
                        return Value::Integer(integer);
                    }
                }
                Value::Float(self.integer as f64 + self.float)
            }
            AggregateOp::Avg if self.count != 0 => {
                Value::Float((self.integer as f64 + self.float) / self.count as f64)
            }
            AggregateOp::Sum | AggregateOp::Avg => Value::Null(DataType::Float),
            AggregateOp::Min | AggregateOp::Max => self
                .extremum
                .map(Extremum::into_value)
                .unwrap_or(Value::Null(data_type)),
        }
    }
}

#[derive(Clone, Hash, PartialEq, Eq)]
enum KeyPart {
    Null,
    Integer(i64),
    Float(u64),
    Boolean(bool),
    Timestamp(DateTime<Utc>),
    Text(usize),
}
type Key = SmallVec<[KeyPart; 4]>;

#[derive(Default)]
struct Groups {
    index: AHashMap<Key, usize>,
    // Text IDs belong to this aggregation, never to a volume dictionary.
    text_index: AHashMap<SmartString, usize>,
    texts: Vec<SmartString>,
    keys: Vec<Key>,
    accumulators: Vec<Vec<Accumulator>>,
    scratch: Key,
    unsupported: bool,
    // The legacy one-column primitive path groups floats by exact bits, unlike
    // multi-column Value equality. Off-schema keys use the existing fallback;
    // reproducing its mixed-type, first-arena-row policy here would be unstable.
    single_type: Option<DataType>,
}

impl Groups {
    fn part(&mut self, cell: Cell<'_>) -> KeyPart {
        match cell {
            Cell::Null => KeyPart::Null,
            Cell::Integer(v) => KeyPart::Integer(v),
            Cell::Float(v) if self.single_type.is_some() => KeyPart::Float(v.to_bits()),
            Cell::Float(v) => {
                let integer = v as i64;
                if crate::core::value::cmp_i64_f64(integer, v) == Some(Ordering::Equal) {
                    KeyPart::Integer(integer)
                } else {
                    KeyPart::Float(if v.is_nan() {
                        f64::NAN.to_bits()
                    } else {
                        v.to_bits()
                    })
                }
            }
            Cell::Boolean(v) => KeyPart::Boolean(v),
            Cell::Timestamp(v) => KeyPart::Timestamp(v),
            Cell::Text(v) => {
                let id = if let Some(&id) = self.text_index.get(v.as_str()) {
                    id
                } else {
                    let id = self.texts.len();
                    let text = v.clone();
                    self.texts.push(text.clone());
                    self.text_index.insert(text, id);
                    id
                };
                KeyPart::Text(id)
            }
            Cell::Other(_) => {
                self.unsupported = true;
                KeyPart::Null
            }
        }
    }
    fn resolve(
        &mut self,
        cells: &Cells<'_, '_>,
        columns: &[usize],
        aggregate_count: usize,
    ) -> usize {
        if columns.is_empty() {
            return 0;
        }
        self.scratch.clear();
        for &column in columns {
            let cell = cells.get(column);
            if self
                .single_type
                .is_some_and(|expected| !matches!(cell, Cell::Null) && cell.data_type() != expected)
            {
                self.unsupported = true;
                return 0;
            }
            let part = self.part(cell);
            if self.unsupported {
                return 0;
            }
            self.scratch.push(part);
        }
        if let Some(&index) = self.index.get(self.scratch.as_slice()) {
            return index;
        }
        let index = self.keys.len();
        self.keys.push(self.scratch.clone());
        self.index.insert(self.scratch.clone(), index);
        self.accumulators.push(
            (0..aggregate_count)
                .map(|_| Accumulator::default())
                .collect(),
        );
        index
    }
}

enum Column<'a> {
    Borrowed(&'a ColumnData, usize),
    Group(Arc<ColumnData>),
}
impl Column<'_> {
    fn data(&self) -> (&ColumnData, usize) {
        match self {
            Self::Borrowed(column, offset) => (column, *offset),
            Self::Group(column) => (column, 0),
        }
    }
}

struct GroupColumns<'a> {
    segment: &'a ColdSegment,
    group: usize,
    count: usize,
    columns: Vec<OnceCell<Column<'a>>>,
}
impl<'a> GroupColumns<'a> {
    fn new(segment: &'a ColdSegment, group: usize, count: usize) -> Self {
        Self {
            segment,
            group,
            count,
            columns: (0..segment.volume.columns.len())
                .map(|_| OnceCell::new())
                .collect(),
        }
    }
    fn bind(&self, logical: usize) -> Result<()> {
        let ColSource::Volume(physical) = self.segment.mapping.sources[logical] else {
            return Ok(());
        };
        if self.columns[physical].get().is_some() {
            return Ok(());
        }
        let source = &self.segment.volume.columns;
        let column = if source.should_use_group_cache() {
            Column::Group(
                source
                    .compressed_store()
                    .ok_or_else(|| Error::internal("missing compressed group source"))?
                    .group_column(physical, self.group)?,
            )
        } else {
            Column::Borrowed(source.get(physical)?, self.group * ROW_GROUP_SIZE)
        };
        let (data, offset) = column.data();
        if offset
            .checked_add(self.count)
            .is_none_or(|end| end > data.len())
        {
            return Err(Error::internal(
                "captured aggregate column is shorter than row group",
            ));
        }
        // OnceCell keeps existing references valid while binding more columns.
        let _ = self.columns[physical].set(column);
        Ok(())
    }
    fn dictionary_selection(
        &self,
        predicates: &[Option<DictionaryPredicate>],
    ) -> Result<DictionarySelection<'_>> {
        let mut filters: SmallVec<[DictFilter<'_>; DICTIONARY_PREFIX]> = SmallVec::new();
        for predicate in predicates {
            let predicate = predicate
                .as_ref()
                .ok_or_else(|| Error::internal("invalid dictionary equality prefix"))?;
            // Missing targets prove the conjunction empty, including NULL rows.
            let Some(target) = predicate.target else {
                return Ok(DictionarySelection { filters: None });
            };
            let (column, offset) = self.bound_column(predicate.physical);
            filters.push((column, offset, target));
        }
        // Sample the infallible equality prefix once per group without moving scalar predicates.
        if filters.len() > 1 && self.count >= 128 {
            let mut leading = 0;
            let mut fewest = usize::MAX;
            for (index, &(column, offset, target)) in filters.iter().enumerate() {
                let (ids, nulls) = column
                    .dict_ids()
                    .ok_or_else(|| Error::internal("bound predicate has no dictionary"))?;
                let end = offset
                    .checked_add(64)
                    .ok_or_else(|| Error::internal("captured dictionary sample offset overflow"))?;
                let ids = ids
                    .get(offset..end)
                    .ok_or_else(|| Error::internal("captured dictionary sample exceeds column"))?;
                let nulls = nulls.get(offset..end).ok_or_else(|| {
                    Error::internal("captured dictionary sample exceeds null flags")
                })?;
                let matches = ids
                    .iter()
                    .zip(nulls)
                    .filter(|&(id, null)| *id == target && !null)
                    .count();
                if matches < fewest {
                    leading = index;
                    fewest = matches;
                }
            }
            filters.swap(0, leading);
        }
        Ok(DictionarySelection {
            filters: Some(filters),
        })
    }

    fn bound_column(&self, physical: usize) -> (&ColumnData, usize) {
        let Some(column) = self.columns[physical].get() else {
            unreachable!("required aggregate column is bound");
        };
        column.data()
    }

    fn get(&self, logical: usize, local: usize) -> Cell<'_> {
        match &self.segment.mapping.sources[logical] {
            ColSource::Default(value) => Cell::from_value(value),
            ColSource::Volume(physical) => {
                let (column, offset) = self.bound_column(*physical);
                Cell::from_column(column, offset + local)
            }
        }
    }

    fn aggregate(&self, operation: AggregateOp, logical: usize) -> BoundAggregate<'_> {
        let source = if operation == AggregateOp::CountStar {
            AggregateSource::Constant(Cell::Null)
        } else {
            match &self.segment.mapping.sources[logical] {
                ColSource::Default(value) => AggregateSource::Constant(Cell::from_value(value)),
                ColSource::Volume(physical) => {
                    let (column, offset) = self.bound_column(*physical);
                    AggregateSource::Column(column, offset)
                }
            }
        };
        BoundAggregate { operation, source }
    }
}

struct DictionarySelection<'a> {
    // None proves the conjunction empty; Some([]) is the unfiltered case.
    filters: Option<SmallVec<[DictFilter<'a>; DICTIONARY_PREFIX]>>,
}
impl DictionarySelection<'_> {
    fn candidates(&self, start: usize, count: usize, out: &mut Vec<usize>) -> Result<()> {
        out.clear();
        let Some(filters) = &self.filters else {
            return Ok(());
        };
        let window: SmallVec<[DictFilter<'_>; DICTIONARY_PREFIX]> = filters
            .iter()
            .map(|&(column, offset, target)| (column, offset + start, target))
            .collect();
        ColumnData::dict_matching_offsets(&window, count, out)
            .ok_or_else(|| Error::internal("invalid captured dictionary predicate window"))
    }
}

enum AggregateSource<'a> {
    Column(&'a ColumnData, usize),
    Constant(Cell<'a>),
}

struct BoundAggregate<'a> {
    operation: AggregateOp,
    source: AggregateSource<'a>,
}

impl BoundAggregate<'_> {
    #[inline]
    fn cell(&self, local: usize) -> Cell<'_> {
        match self.source {
            AggregateSource::Column(column, offset) => Cell::from_column(column, offset + local),
            AggregateSource::Constant(cell) => cell,
        }
    }
}

enum Cells<'a, 'b> {
    Hot(&'a Row, &'a [Value]),
    Cold(&'a GroupColumns<'b>, usize),
}
impl Cells<'_, '_> {
    fn get(&self, column: usize) -> Cell<'_> {
        match self {
            Self::Hot(row, defaults) => {
                Cell::from_value(row.get(column).unwrap_or(&defaults[column]))
            }
            Self::Cold(columns, local) => columns.get(column, *local),
        }
    }
}

pub(crate) fn grouped(
    schema: &Schema,
    view: &CapturedHotView,
    cold: ColdInput<'_>,
    group_by: &[usize],
    aggregates: &[(AggregateOp, usize)],
) -> Result<Option<Vec<GroupedAggregateResult>>> {
    if group_by.is_empty() {
        return Ok(None);
    }
    run(schema, view, cold, group_by, aggregates, None)
}

pub(crate) fn filtered(
    schema: &Schema,
    view: &CapturedHotView,
    cold: ColdInput<'_>,
    aggregates: &[(AggregateOp, usize)],
    expression: &dyn Expression,
) -> Result<Option<Vec<Value>>> {
    run(schema, view, cold, &[], aggregates, Some(expression))?
        .map(|mut results| {
            results
                .pop()
                .map(|result| result.aggregate_values)
                .ok_or_else(|| Error::internal("global aggregate returned no group"))
        })
        .transpose()
}

fn run(
    schema: &Schema,
    view: &CapturedHotView,
    cold: ColdInput<'_>,
    group_by: &[usize],
    aggregates: &[(AggregateOp, usize)],
    expression: Option<&dyn Expression>,
) -> Result<Option<Vec<GroupedAggregateResult>>> {
    let mut predicates_bound = Vec::new();
    if expression.is_some_and(|expression| !predicates(expression, schema, &mut predicates_bound)) {
        return Ok(None);
    }
    if group_by.iter().any(|&column| {
        schema
            .columns
            .get(column)
            .is_none_or(|c| !scalar_type(c.data_type))
    }) {
        return Ok(None);
    }
    for &(operation, column) in aggregates {
        if operation == AggregateOp::CountStar {
            continue;
        }
        let Some(column) = schema.columns.get(column) else {
            return Ok(None);
        };
        if match operation {
            AggregateOp::Sum | AggregateOp::Avg => !matches!(
                column.data_type,
                DataType::Integer | DataType::Float | DataType::Boolean
            ),
            AggregateOp::Min | AggregateOp::Max => !scalar_type(column.data_type),
            _ => false,
        } {
            return Ok(None);
        }
    }
    let mut needed: SmallVec<[usize; 12]> = group_by.iter().copied().collect();
    needed.extend(
        aggregates
            .iter()
            .filter(|(operation, _)| *operation != AggregateOp::CountStar)
            .map(|(_, column)| *column),
    );
    needed.extend(predicates_bound.iter().map(|p| p.column));
    needed.sort_unstable();
    needed.dedup();
    // Schema evolution and unsupported physical representations are decided
    // from metadata before any hot/cold accumulation has happened.
    if let Some((generation, _)) = cold {
        for segment in generation.segments.values() {
            for &logical in &needed {
                let Some(source) = segment.mapping.sources.get(logical) else {
                    return Ok(None);
                };
                if let ColSource::Volume(physical) = source {
                    if segment.volume.meta.column_types.get(*physical)
                        != Some(&schema.columns[logical].data_type)
                    {
                        return Ok(None);
                    }
                }
            }
        }
    }
    let defaults: Vec<_> = schema
        .columns
        .iter()
        .map(|column| {
            column
                .default_value
                .clone()
                .unwrap_or(Value::Null(column.data_type))
        })
        .collect();
    let mut groups = Groups {
        single_type: (group_by.len() == 1).then(|| schema.columns[group_by[0]].data_type),
        ..Groups::default()
    };
    if group_by.is_empty() {
        groups.keys.push(Key::new());
        groups.index.insert(Key::new(), 0);
        groups.accumulators.push(
            (0..aggregates.len())
                .map(|_| Accumulator::default())
                .collect(),
        );
    }
    let add = |groups: &mut Groups, index: usize, cells: &Cells<'_, '_>| {
        for (accumulator, &(operation, column)) in
            groups.accumulators[index].iter_mut().zip(aggregates)
        {
            let cell = if operation == AggregateOp::CountStar {
                Cell::Null
            } else {
                cells.get(column)
            };
            if matches!(operation, AggregateOp::Min | AggregateOp::Max)
                && matches!(cell, Cell::Other(_))
            {
                groups.unsupported = true;
                return;
            }
            accumulator.add(operation, cell);
        }
    };
    let mut predicate_error = None;
    view.for_each_visible_until(|_, row| {
        let cells = Cells::Hot(row, &defaults);
        for predicate in &predicates_bound {
            match predicate.matches(cells.get(predicate.column)) {
                Ok(true) => (),
                Ok(false) => return true,
                Err(error) => {
                    predicate_error = Some(error);
                    return false;
                }
            }
        }
        let index = if group_by.is_empty() {
            0
        } else {
            groups.resolve(&cells, group_by, aggregates.len())
        };
        if !groups.unsupported {
            add(&mut groups, index, &cells);
        }
        !groups.unsupported
    });
    if let Some(error) = predicate_error {
        return Err(error);
    }
    if groups.unsupported {
        return Ok(None);
    }
    if let Some((generation, pending)) = cold {
        let mut hidden = [0u64; ROW_GROUP_SIZE.div_ceil(64)];
        let mut candidates = Vec::new();
        for &id in &generation.segment_ids_newest_first {
            let segment = generation.segments.get(&id).ok_or_else(|| {
                Error::internal("captured aggregate generation is missing a segment")
            })?;
            let mut loaded = None;
            for (group, ids) in segment
                .volume
                .meta
                .row_ids
                .chunks(ROW_GROUP_SIZE)
                .enumerate()
            {
                view.mark_authoritative(ids, &mut hidden);
                if segment.visible.is_some()
                    || !pending.is_empty()
                    || !generation.tombstones.is_empty()
                {
                    for (local, row_id) in ids.iter().enumerate() {
                        if !segment.is_visible(group * ROW_GROUP_SIZE + local)
                            || pending.contains(row_id)
                            || generation.tombstones.get(row_id).is_some_and(|&sequence| {
                                i64::try_from(sequence).is_ok_and(|sequence| {
                                    view.epoch().admits_commit_sequence(sequence)
                                })
                            })
                        {
                            hidden[local / 64] |= 1 << (local % 64);
                        }
                    }
                }
                // mark_authoritative zeros unused tail bits. Inspect words,
                // rather than every row, when no cold exclusions need merging.
                if hidden[..ids.len().div_ceil(64)]
                    .iter()
                    .map(|bits| bits.count_ones() as usize)
                    .sum::<usize>()
                    == ids.len()
                {
                    continue;
                }
                let segment = match &mut loaded {
                    Some(segment) => segment,
                    slot @ None => slot.insert(generation.load_segment(id)?),
                };
                let columns = GroupColumns::new(segment, group, ids.len());
                for predicate in &predicates_bound {
                    columns.bind(predicate.column)?;
                }
                let dictionary_predicates: SmallVec<[Option<DictionaryPredicate>; 4]> =
                    predicates_bound
                        .iter()
                        .map(|predicate| predicate.bind_dictionary(&columns))
                        .collect();
                // Only move a leading, infallible equality prefix into the
                // batched selector. This preserves errors from earlier scalar
                // predicates and keeps both selector scratch buffers bounded.
                let dictionary_prefix = predicates_bound
                    .iter()
                    .zip(&dictionary_predicates)
                    .take(DICTIONARY_PREFIX)
                    .take_while(|(predicate, dictionary)| {
                        predicate.operator == Operator::Eq && dictionary.is_some()
                    })
                    .count();
                if dictionary_prefix != 0 && candidates.capacity() == 0 {
                    candidates.reserve_exact(DICTIONARY_WINDOW);
                }
                let dictionary_selection =
                    columns.dictionary_selection(&dictionary_predicates[..dictionary_prefix])?;
                candidates.clear();
                let mut candidate_position = 0;
                let mut window_start = 0;
                let mut window_end = 0;
                let mut plain = 0;
                let mut bound_aggregates: Option<SmallVec<[BoundAggregate<'_>; 4]>> = None;
                // Dictionary identity is local to this bound group. Cache only
                // its translation to globally canonical group indices.
                let mut dictionary_groups: Vec<usize> = Vec::new();
                let mut null_group = None;
                'rows: loop {
                    let local = if dictionary_prefix != 0 {
                        while candidate_position == candidates.len() {
                            if window_end == ids.len() {
                                break 'rows;
                            }
                            window_start = window_end;
                            window_end = (window_start + DICTIONARY_WINDOW).min(ids.len());
                            dictionary_selection.candidates(
                                window_start,
                                window_end - window_start,
                                &mut candidates,
                            )?;
                            candidate_position = 0;
                        }
                        let local = window_start + candidates[candidate_position];
                        candidate_position += 1;
                        local
                    } else {
                        if plain == ids.len() {
                            break;
                        }
                        plain += 1;
                        plain - 1
                    };
                    if hidden[local / 64] & (1 << (local % 64)) != 0 {
                        continue;
                    }
                    for (predicate, dictionary) in predicates_bound
                        .iter()
                        .zip(&dictionary_predicates)
                        .skip(dictionary_prefix)
                    {
                        if !predicate.matches_cold(&columns, local, dictionary.as_ref())? {
                            continue 'rows;
                        }
                    }
                    let bound = match &mut bound_aggregates {
                        Some(bound) => bound,
                        slot @ None => {
                            for &column in &needed {
                                columns.bind(column)?;
                            }
                            slot.insert(
                                aggregates
                                    .iter()
                                    .map(|&(operation, column)| {
                                        columns.aggregate(operation, column)
                                    })
                                    .collect(),
                            )
                        }
                    };
                    let cells = Cells::Cold(&columns, local);
                    let mut cached_dictionary = None;
                    if group_by.len() == 1 {
                        if let ColSource::Volume(physical) =
                            columns.segment.mapping.sources[group_by[0]]
                        {
                            let (data, offset) = columns.bound_column(physical);
                            if let ColumnData::Dictionary {
                                ids,
                                dictionary,
                                nulls,
                            } = data
                            {
                                if dictionary.len() <= ROW_GROUP_SIZE {
                                    if dictionary_groups.is_empty() {
                                        dictionary_groups.resize(dictionary.len(), usize::MAX);
                                    }
                                    cached_dictionary = Some(if nulls[offset + local] {
                                        None
                                    } else {
                                        Some(ids[offset + local] as usize)
                                    });
                                }
                            }
                        }
                    }
                    let index = match cached_dictionary {
                        Some(Some(dictionary_id)) => {
                            let cached = dictionary_groups[dictionary_id];
                            if cached != usize::MAX {
                                cached
                            } else {
                                let index = groups.resolve(&cells, group_by, aggregates.len());
                                dictionary_groups[dictionary_id] = index;
                                index
                            }
                        }
                        Some(None) => *null_group.get_or_insert_with(|| {
                            groups.resolve(&cells, group_by, aggregates.len())
                        }),
                        None if group_by.is_empty() => 0,
                        None => groups.resolve(&cells, group_by, aggregates.len()),
                    };
                    if groups.unsupported {
                        return Ok(None);
                    }
                    for (accumulator, aggregate) in
                        groups.accumulators[index].iter_mut().zip(bound.iter())
                    {
                        let cell = aggregate.cell(local);
                        if matches!(aggregate.operation, AggregateOp::Min | AggregateOp::Max)
                            && matches!(cell, Cell::Other(_))
                        {
                            return Ok(None);
                        }
                        accumulator.add(aggregate.operation, cell);
                    }
                    if groups.unsupported {
                        return Ok(None);
                    }
                }
            }
        }
    }
    Ok(Some(
        groups
            .keys
            .into_iter()
            .zip(groups.accumulators)
            .map(|(key, accumulators)| {
                let group_values = key
                    .into_iter()
                    .zip(group_by)
                    .map(|(part, &column)| match part {
                        KeyPart::Null => Value::Null(schema.columns[column].data_type),
                        KeyPart::Integer(v)
                            if group_by.len() != 1
                                && schema.columns[column].data_type == DataType::Float =>
                        {
                            Value::Float(v as f64)
                        }
                        KeyPart::Integer(v) => Value::Integer(v),
                        KeyPart::Float(v) => Value::Float(f64::from_bits(v)),
                        KeyPart::Boolean(v) => Value::Boolean(v),
                        KeyPart::Timestamp(v) => Value::Timestamp(v),
                        KeyPart::Text(v) => Value::Text(groups.texts[v].clone()),
                    })
                    .collect();
                let aggregate_values = accumulators
                    .into_iter()
                    .zip(aggregates)
                    .map(|(accumulator, &(operation, column))| {
                        accumulator.finish(
                            operation,
                            schema
                                .columns
                                .get(column)
                                .map_or(DataType::Null, |c| c.data_type),
                            !group_by.is_empty(),
                        )
                    })
                    .collect();
                GroupedAggregateResult {
                    group_values,
                    aggregate_values,
                }
            })
            .collect(),
    ))
}

#[cfg(test)]
mod tests {
    use super::super::manifest::{SegmentManager, SegmentMeta};
    use super::super::table::SegmentedTable;
    use super::super::writer::{ColumnMapping, VolumeBuilder};
    use super::*;
    use crate::core::{IsolationLevel, SchemaBuilder};
    use crate::functions::aggregate::CompiledAggregate;
    use crate::storage::expression::{AndExpr, ComparisonExpr, InListExpr};
    use crate::storage::mvcc::registry::TransactionRegistry;
    use crate::storage::mvcc::version_store::{RowVersion, TransactionVersionStore, VersionStore};
    use crate::storage::mvcc::MVCCTable;
    use crate::storage::traits::Table;

    fn schema() -> Schema {
        SchemaBuilder::new("aggregate")
            .column("id", DataType::Integer, false, true)
            .column("a", DataType::Text, true, false)
            .column("b", DataType::Text, true, false)
            .column("n", DataType::Integer, true, false)
            .column("f", DataType::Float, true, false)
            .column("flag", DataType::Boolean, true, false)
            .column("time", DataType::Timestamp, true, false)
            .build()
    }

    fn row(id: i64, a: &str, b: &str, n: i64) -> Row {
        Row::from_values(vec![
            Value::Integer(id),
            Value::text(a),
            Value::text(b),
            Value::Integer(n),
            Value::Float(n as f64 / 2.0),
            Value::Boolean(n % 2 == 0),
            Value::Timestamp(DateTime::from_timestamp(n % 10_000, 123).unwrap()),
        ])
    }

    fn add_segment(
        manager: &SegmentManager,
        schema: &Schema,
        id: u64,
        rows: &[(i64, Row)],
        warm: bool,
    ) {
        let mut builder = VolumeBuilder::new(schema);
        for (id, row) in rows {
            builder.add_row(*id, row);
        }
        let mut volume = builder.finish();
        if warm {
            let (_, compressed) = super::super::io::serialize_v4_public(&volume).unwrap();
            volume.columns.attach_compressed_store(compressed);
            volume = volume.to_warm().unwrap();
        }
        manager.register_segment(
            id,
            Arc::new(volume),
            SegmentMeta {
                segment_id: id,
                file_path: format!("aggregate-{id}.vol").into(),
                row_count: rows.len(),
                min_row_id: rows.first().map_or(0, |(id, _)| *id),
                max_row_id: rows.last().map_or(0, |(id, _)| *id),
                schema_version: 0,
                creation_lsn: 0,
                seal_seq: 0,
            },
            Some(schema),
        );
    }

    fn store(schema: &Schema) -> (Arc<TransactionRegistry>, Arc<VersionStore>) {
        let registry = Arc::new(TransactionRegistry::new());
        let store = Arc::new(VersionStore::with_visibility_checker(
            "aggregate",
            schema.clone(),
            registry.clone(),
        ));
        (registry, store)
    }

    fn result_map(results: Vec<GroupedAggregateResult>) -> AHashMap<Vec<Value>, Vec<Value>> {
        results
            .into_iter()
            .map(|result| (result.group_values, result.aggregate_values))
            .collect()
    }

    fn reference(
        rows: &[Row],
        groups: &[usize],
        operations: &[(AggregateOp, usize)],
    ) -> AHashMap<Vec<Value>, Vec<Value>> {
        let mut accumulators: AHashMap<Vec<Value>, Vec<CompiledAggregate>> = AHashMap::default();
        for row in rows {
            let key = groups.iter().map(|&column| row[column].clone()).collect();
            let group = accumulators.entry(key).or_insert_with(|| {
                operations
                    .iter()
                    .map(|&(operation, _)| match operation {
                        AggregateOp::CountStar => CompiledAggregate::count_star(),
                        AggregateOp::Count => CompiledAggregate::count(false),
                        AggregateOp::Sum => CompiledAggregate::sum(false),
                        AggregateOp::Avg => CompiledAggregate::avg(false),
                        AggregateOp::Min => CompiledAggregate::min(),
                        AggregateOp::Max => CompiledAggregate::max(),
                    })
                    .collect()
            });
            for (accumulator, &(_, column)) in group.iter_mut().zip(operations) {
                accumulator.accumulate(&row[column]);
            }
        }
        accumulators
            .into_iter()
            .map(|(key, group)| (key, group.iter().map(CompiledAggregate::result).collect()))
            .collect()
    }

    #[test]
    fn captured_aggregate_snapshot_overlap_own_overlay_and_dictionary_identity() {
        let schema = schema();
        let operations = [
            (AggregateOp::CountStar, 0),
            (AggregateOp::Count, 3),
            (AggregateOp::Sum, 3),
            (AggregateOp::Avg, 4),
            (AggregateOp::Min, 3),
            (AggregateOp::Max, 3),
            (AggregateOp::Min, 1),
            (AggregateOp::Max, 6),
            (AggregateOp::Sum, 5),
            (AggregateOp::Min, 5),
        ];
        for warm in [false, true] {
            let manager = Arc::new(SegmentManager::new("aggregate", None));
            // Dictionary ID zero names different strings in these volumes;
            // repeated strings receive different IDs and must still merge.
            add_segment(
                &manager,
                &schema,
                1,
                &[
                    (1, row(1, "a", "x", 1)),
                    (2, row(2, "b", "y", 2)),
                    (3, row(3, "a", "y", 3)),
                    (4, row(4, "b", "x", 4)),
                    (5, row(5, "c", "x", 5)),
                ],
                warm,
            );
            let mut null = row(8, "a", "x", 8);
            null.set(1, Value::Null(DataType::Text)).unwrap();
            null.set(3, Value::Null(DataType::Integer)).unwrap();
            add_segment(
                &manager,
                &schema,
                2,
                &[
                    (5, row(5, "b", "y", 50)),
                    (6, row(6, "a", "x", 6)),
                    (7, row(7, "b", "x", 7)),
                    (8, null.clone()),
                ],
                warm,
            );
            manager.set_seal_overlap(3);
            let (registry, store) = store(&schema);
            let (writer, _) = registry.begin_transaction();
            registry.start_commit(writer);
            store
                .add_version(1, RowVersion::new(writer, row(1, "changed", "x", 10)))
                .unwrap();
            let mut deleted = RowVersion::new(writer, row(2, "b", "y", 2));
            deleted.deleted_at_txn_id = writer;
            store.add_version(2, deleted).unwrap();
            registry.complete_commit(writer);
            let (inflight, _) = registry.begin_transaction();
            let inflight_sequence = registry.start_commit(inflight);
            store
                .add_version(3, RowVersion::new(inflight, row(3, "wrong", "wrong", 300)))
                .unwrap();
            let (reader, _) =
                registry.begin_transaction_with_isolation(IsolationLevel::SnapshotIsolation);
            manager.add_tombstones(&[3], inflight_sequence as u64);
            manager.add_tombstones(&[7], registry.get_commit_sequence(writer).unwrap() as u64);
            manager.add_pending_tombstone(reader, 4);
            let mut local = TransactionVersionStore::new(store.clone(), reader);
            local.put(6, row(6, "b", "y", 60), false).unwrap();
            local.put(9, row(9, "a", "x", 9), false).unwrap();
            let hot = MVCCTable::new(reader, store.clone(), local);
            let mut table = SegmentedTable::new(Box::new(hot), manager.clone());
            table
                .set_read_epoch(registry.read_epoch_for_transaction(reader).unwrap())
                .unwrap();
            registry.complete_commit(inflight);
            let expected = vec![
                row(1, "changed", "x", 10),
                row(3, "a", "y", 3),
                row(5, "b", "y", 50),
                row(6, "b", "y", 60),
                null,
                row(9, "a", "x", 9),
            ];
            for groups in [vec![1], vec![1, 2], vec![1, 2, 4, 5, 6], vec![5], vec![4]] {
                let actual = table
                    .compute_grouped_aggregates(&groups, &operations)
                    .unwrap()
                    .unwrap();
                assert_eq!(
                    result_map(actual),
                    reference(&expected, &groups, &operations),
                    "warm={warm}, group={groups:?}"
                );
            }
            let filter = AndExpr::new(vec![
                Box::new(ComparisonExpr::eq("a", Value::text("a"))),
                Box::new(ComparisonExpr::new("n", Operator::Gt, Value::Float(2.5))),
            ]);
            let actual = table
                .compute_filtered_aggregates(&operations, &filter)
                .unwrap()
                .unwrap();
            let expected_rows = [row(3, "a", "y", 3), row(9, "a", "x", 9)];
            assert_eq!(
                actual,
                reference(&expected_rows, &[], &operations)
                    .remove(&Vec::new())
                    .unwrap()
            );
            // Frozen pending and generation remain authoritative after manager changes.
            manager.rollback_pending_tombstones(reader);
            manager.clear();
            assert_eq!(
                table
                    .compute_filtered_aggregates(&operations, &filter)
                    .unwrap()
                    .unwrap(),
                actual
            );
        }
    }

    #[test]
    fn captured_aggregate_hot_only_numeric_boundaries_and_complete_filter_preflight() {
        let schema = schema();
        let (registry, store) = store(&schema);
        let (writer, _) = registry.begin_transaction();
        registry.start_commit(writer);
        let mut first = row(1, "a", "x", i64::MAX);
        first.set(4, Value::Float(f64::NAN)).unwrap();
        let mut second = row(2, "a", "x", 2);
        second.set(4, Value::Float(0.0)).unwrap();
        let mut third = row(3, "a", "x", 3);
        third.set(4, Value::Float(-0.0)).unwrap();
        store
            .add_version(1, RowVersion::new(writer, first))
            .unwrap();
        store
            .add_version(2, RowVersion::new(writer, second))
            .unwrap();
        store
            .add_version(3, RowVersion::new(writer, third))
            .unwrap();
        registry.complete_commit(writer);
        let (reader, _) = registry.begin_transaction();
        let mut hot = MVCCTable::new(
            reader,
            store.clone(),
            TransactionVersionStore::new(store.clone(), reader),
        );
        hot.set_read_epoch(registry.capture_read_epoch()).unwrap();
        let operations = [
            (AggregateOp::Sum, 3),
            (AggregateOp::Sum, 4),
            (AggregateOp::Avg, 4),
            (AggregateOp::Min, 4),
            (AggregateOp::Max, 4),
            (AggregateOp::Count, 4),
        ];
        let filter = ComparisonExpr::new("n", Operator::Gt, Value::Float(i64::MAX as f64));
        assert_eq!(
            hot.compute_filtered_aggregates(&[(AggregateOp::CountStar, 0)], &filter)
                .unwrap(),
            Some(vec![Value::Integer(0)]),
            "exact integer/float comparison at 2^63"
        );
        let all = ComparisonExpr::new("id", Operator::Gt, Value::Integer(0));
        assert_eq!(
            hot.compute_filtered_aggregates(&operations, &all)
                .unwrap()
                .unwrap(),
            vec![
                Value::Float((i64::MAX as i128 + 5) as f64),
                Value::Float(0.0),
                Value::Float(0.0),
                Value::Float(0.0),
                Value::Float(0.0),
                Value::Integer(3)
            ]
        );
        let groups = hot
            .compute_grouped_aggregates(&[4], &[(AggregateOp::CountStar, 0)])
            .unwrap()
            .unwrap();
        assert_eq!(
            groups.len(),
            3,
            "single-column float grouping preserves the legacy bit-pattern keys"
        );
        let unsupported = AndExpr::new(vec![
            Box::new(all),
            Box::new(InListExpr::new("id", vec![Value::Integer(1)])),
        ]);
        assert!(hot
            .compute_filtered_aggregates(&operations, &unsupported)
            .unwrap()
            .is_none());
        let unsupported_operator = ComparisonExpr::new("a", Operator::Like, Value::text("a%"));
        assert!(hot
            .compute_filtered_aggregates(&operations, &unsupported_operator)
            .unwrap()
            .is_none());
    }

    #[test]
    fn captured_sum_result_types_survive_layout_and_own_overlay_changes() {
        fn assert_sum(actual: &Value, expected: &Value) {
            match (actual, expected) {
                (Value::Integer(a), Value::Integer(b)) => assert_eq!(a, b),
                (Value::Float(a), Value::Float(b)) => assert_eq!(a, b),
                (Value::Null(a), Value::Null(b)) => assert_eq!(a, b),
                _ => panic!("SUM changed its result type: {actual:?}, expected {expected:?}"),
            }
        }
        for (cold, warm) in [(false, false), (true, false), (true, true)] {
            for own in [false, true] {
                let schema = schema();
                let (registry, store) = store(&schema);
                let manager = Arc::new(SegmentManager::new("aggregate", None));
                let mut null = row(5, "null", "same", 0);
                null.set(3, Value::Null(DataType::Integer)).unwrap();
                let seed = vec![
                    (1, row(1, "plain", "same", 10)),
                    (2, row(2, "plain", "same", 20)),
                    (3, row(3, "overflow", "same", i64::MAX)),
                    (4, row(4, "overflow", "same", 1)),
                    (5, null),
                ];
                if cold {
                    add_segment(&manager, &schema, 1, &seed[..2], warm);
                    add_segment(&manager, &schema, 2, &seed[2..], warm);
                } else {
                    let (writer, _) = registry.begin_transaction();
                    registry.start_commit(writer);
                    for (id, value) in &seed {
                        store
                            .add_version(*id, RowVersion::new(writer, value.clone()))
                            .unwrap();
                    }
                    registry.complete_commit(writer);
                }
                let (reader, _) = registry.begin_transaction();
                let mut local = TransactionVersionStore::new(store.clone(), reader);
                if own {
                    local.put(1, row(1, "plain", "same", 11), false).unwrap();
                    local.put(2, seed[1].1.clone(), true).unwrap();
                    let mut mixed = row(6, "plain", "same", 0);
                    mixed.set(3, Value::Float(0.5)).unwrap();
                    local.put(6, mixed, false).unwrap();
                }
                let hot = MVCCTable::new(reader, store, local);
                let mut table = SegmentedTable::new(Box::new(hot), manager);
                table.set_read_epoch(registry.capture_read_epoch()).unwrap();
                let groups = table
                    .compute_grouped_aggregates(&[1], &[(AggregateOp::Sum, 3)])
                    .unwrap()
                    .unwrap();
                assert_eq!(groups.len(), 3);
                for (name, global, grouped) in [
                    (
                        "plain",
                        if own {
                            Value::Float(11.5)
                        } else {
                            Value::Integer(30)
                        },
                        Value::Float(if own { 11.5 } else { 30.0 }),
                    ),
                    (
                        "overflow",
                        Value::Float(9_223_372_036_854_775_808.0),
                        Value::Float(9_223_372_036_854_775_808.0),
                    ),
                    (
                        "null",
                        Value::Null(DataType::Float),
                        Value::Null(DataType::Float),
                    ),
                ] {
                    let result = groups
                        .iter()
                        .find(|g| g.group_values == [Value::text(name)])
                        .unwrap();
                    assert_sum(&result.aggregate_values[0], &grouped);
                    let filter = ComparisonExpr::new("a", Operator::Eq, Value::text(name));
                    let result = table
                        .compute_filtered_aggregates(&[(AggregateOp::Sum, 3)], &filter)
                        .unwrap()
                        .unwrap();
                    assert_sum(&result[0], &global);
                }
                let empty = ComparisonExpr::new("id", Operator::Lt, Value::Integer(0));
                let result = table
                    .compute_filtered_aggregates(&[(AggregateOp::Sum, 3)], &empty)
                    .unwrap()
                    .unwrap();
                assert_sum(&result[0], &Value::Null(DataType::Float));
            }
        }
    }

    #[test]
    fn captured_group_single_float_preserves_bits_across_hot_cold_overlap_and_own_rows() {
        const NAN_A: u64 = 0x7ff8_0000_0000_0001;
        const NAN_B: u64 = 0xfff8_0000_0000_0002;
        fn floating(id: i64, bits: Option<u64>, weight: i64) -> Row {
            let mut value = row(id, "same", "same", weight);
            value
                .set(
                    4,
                    bits.map_or(Value::Null(DataType::Float), |b| {
                        Value::Float(f64::from_bits(b))
                    }),
                )
                .unwrap();
            value
        }
        fn bit_groups(
            results: Vec<GroupedAggregateResult>,
        ) -> std::collections::BTreeMap<Option<u64>, (i64, f64)> {
            results
                .into_iter()
                .map(|result| {
                    let key = match &result.group_values[..] {
                        [Value::Float(v)] => Some(v.to_bits()),
                        [Value::Null(DataType::Float)] => None,
                        other => panic!("float group retained the wrong value type: {other:?}"),
                    };
                    let values = match result.aggregate_values[..] {
                        [Value::Integer(count), Value::Float(sum)] => (count, sum),
                        ref other => panic!("unexpected aggregates: {other:?}"),
                    };
                    (key, values)
                })
                .collect()
        }
        for (cold, warm, overlap) in [
            (false, false, false),
            (true, false, false),
            (true, true, false),
            (true, true, true),
        ] {
            let schema = schema();
            let (registry, store) = store(&schema);
            let manager = Arc::new(SegmentManager::new("aggregate", None));
            let mut expected = vec![
                floating(1, Some(0.0f64.to_bits()), 10),
                floating(2, Some((-0.0f64).to_bits()), 20),
                floating(3, Some(NAN_A), 30),
                floating(4, Some(NAN_B), 40),
                floating(5, Some(1.0f64.to_bits()), 50),
                floating(6, Some(0.0f64.to_bits()), 60),
                floating(7, Some(NAN_A), 70),
                floating(8, None, 80),
            ];
            if cold {
                let rows: Vec<_> = expected
                    .iter()
                    .enumerate()
                    .map(|(i, row)| (i as i64 + 1, row.clone()))
                    .collect();
                add_segment(&manager, &schema, 1, &rows[..4], warm);
                add_segment(&manager, &schema, 2, &rows[4..], warm);
            } else {
                let (writer, _) = registry.begin_transaction();
                registry.start_commit(writer);
                for (i, row) in expected.iter().enumerate() {
                    store
                        .add_version(i as i64 + 1, RowVersion::new(writer, row.clone()))
                        .unwrap();
                }
                registry.complete_commit(writer);
            }
            if overlap {
                let (writer, _) = registry.begin_transaction();
                registry.start_commit(writer);
                expected[0] = floating(1, Some((-0.0f64).to_bits()), 11);
                store
                    .add_version(1, RowVersion::new(writer, expected[0].clone()))
                    .unwrap();
                let mut deletion = RowVersion::new(writer, expected[3].clone());
                deletion.deleted_at_txn_id = writer;
                store.add_version(4, deletion).unwrap();
                registry.complete_commit(writer);
                expected.remove(3);
                manager.set_seal_overlap(2);
            }
            let (reader, _) = registry.begin_transaction();
            let mut local = TransactionVersionStore::new(store.clone(), reader);
            if overlap {
                let own = floating(9, Some(NAN_B), 90);
                local.put(9, own.clone(), false).unwrap();
                expected.push(own);
                local
                    .put(6, floating(6, Some(0.0f64.to_bits()), 60), true)
                    .unwrap();
                expected.retain(|row| row[0] != Value::Integer(6));
            }
            let hot = MVCCTable::new(reader, store.clone(), local);
            let mut table = SegmentedTable::new(Box::new(hot), manager);
            table.set_read_epoch(registry.capture_read_epoch()).unwrap();
            // A later committed change must not rewrite the captured group key.
            let (later, _) = registry.begin_transaction();
            registry.start_commit(later);
            store
                .add_version(2, RowVersion::new(later, floating(2, Some(NAN_A), 2000)))
                .unwrap();
            registry.complete_commit(later);
            let operations = [(AggregateOp::CountStar, 0), (AggregateOp::Sum, 3)];
            let mut expected_bits = std::collections::BTreeMap::<Option<u64>, (i64, f64)>::new();
            for row in &expected {
                let key = match row[4] {
                    Value::Float(v) => Some(v.to_bits()),
                    Value::Null(_) => None,
                    _ => unreachable!(),
                };
                let Value::Integer(weight) = row[3] else {
                    unreachable!()
                };
                let group = expected_bits.entry(key).or_default();
                group.0 += 1;
                group.1 += weight as f64;
            }
            let actual = table
                .compute_grouped_aggregates(&[4], &operations)
                .unwrap()
                .unwrap();
            assert_eq!(
                actual.len(),
                expected_bits.len(),
                "cold={cold} warm={warm} overlap={overlap}"
            );
            assert_eq!(bit_groups(actual), expected_bits);
            // Existing multi-key Value equality still merges signed zeros and
            // NaN payloads. Comparing by Value is intentional only in this case.
            let multi = table
                .compute_grouped_aggregates(&[4, 1], &operations)
                .unwrap()
                .unwrap();
            assert_eq!(
                result_map(multi),
                reference(&expected, &[4, 1], &operations)
            );
        }
    }

    #[test]
    fn captured_group_single_off_schema_keys_fall_back_but_multi_numeric_equality_remains() {
        let schema = schema();
        let (registry, store) = store(&schema);
        let (writer, _) = registry.begin_transaction();
        registry.start_commit(writer);
        for (id, key) in [
            Value::Integer(1),
            Value::Float(1.0),
            Value::Float(-0.0),
            Value::Integer(0),
        ]
        .into_iter()
        .enumerate()
        {
            let mut data = row(id as i64 + 1, "same", "same", 1);
            data.set(4, key).unwrap();
            store
                .add_version(id as i64 + 1, RowVersion::new(writer, data))
                .unwrap();
        }
        registry.complete_commit(writer);
        let view = CapturedHotView::new(
            store.capture_hot_root(),
            registry.capture_read_epoch(),
            None,
        );
        assert!(
            grouped(&schema, &view, None, &[4], &[(AggregateOp::CountStar, 0)])
                .unwrap()
                .is_none()
        );
        let multi = grouped(
            &schema,
            &view,
            None,
            &[4, 1],
            &[(AggregateOp::CountStar, 0)],
        )
        .unwrap()
        .unwrap();
        assert_eq!(multi.len(), 2);
        assert!(multi
            .iter()
            .all(|group| group.aggregate_values == [Value::Integer(2)]));
        // Homogeneous primitive grouping emits the actual schema-compatible
        // value type; Float(1.0) must not become Integer(1).
        for (column, wanted) in [
            (3, DataType::Integer),
            (5, DataType::Boolean),
            (6, DataType::Timestamp),
        ] {
            let groups = grouped(
                &schema,
                &view,
                None,
                &[column],
                &[(AggregateOp::CountStar, 0)],
            )
            .unwrap()
            .unwrap();
            assert!(groups
                .iter()
                .all(|group| group.group_values[0].data_type() == wanted));
        }
    }

    #[cfg(feature = "test-failpoints")]
    #[test]
    fn captured_aggregate_visibility_precedes_io_and_late_binding_propagates_error() {
        let _guard = crate::test_failpoints::FailpointGuard::new();
        let schema = schema();
        let (registry, store) = store(&schema);
        let manager = SegmentManager::new("aggregate", None);
        add_segment(&manager, &schema, 1, &[(1, row(1, "a", "x", 1))], true);
        add_segment(&manager, &schema, 2, &[(2, row(2, "b", "y", 2))], true);
        let (_, mut generation) = manager.capture_with_hot(|| ()).unwrap();
        let view = CapturedHotView::new(
            store.capture_hot_root(),
            registry.capture_read_epoch(),
            None,
        );
        let pending = FxHashSet::default();
        let filter = ComparisonExpr::new("id", Operator::Gt, Value::Integer(0));
        let operations = [(AggregateOp::Sum, 3)];
        // First volume predicate + aggregate succeeds. The later volume's
        // aggregate read must fail the entire call, never return the prefix.
        crate::test_failpoints::fail_cold_read_on(4);
        let error = filtered(
            &schema,
            &view,
            Some((&generation, &pending)),
            &operations,
            &filter,
        )
        .unwrap_err();
        assert!(error.to_string().contains("injected cold read failure"));
        assert_eq!(
            filtered(
                &schema,
                &view,
                Some((&generation, &pending)),
                &operations,
                &filter
            )
            .unwrap(),
            Some(vec![Value::Integer(3)])
        );
        // Five aggregates borrow the same physical input. Each volume still
        // loads only its predicate and aggregate columns, once each.
        crate::test_failpoints::fail_cold_read_on(5);
        assert_eq!(
            filtered(
                &schema,
                &view,
                Some((&generation, &pending)),
                &[
                    (AggregateOp::Count, 3),
                    (AggregateOp::Sum, 3),
                    (AggregateOp::Avg, 3),
                    (AggregateOp::Min, 3),
                    (AggregateOp::Max, 3),
                ],
                &filter,
            )
            .unwrap(),
            Some(vec![
                Value::Integer(2),
                Value::Integer(3),
                Value::Float(1.5),
                Value::Integer(1),
                Value::Integer(2),
            ])
        );
        assert!(crate::test_failpoints::check_cold_read().is_err());
        let reject = ComparisonExpr::new("id", Operator::Lt, Value::Integer(0));
        crate::test_failpoints::fail_cold_read_on(3);
        assert_eq!(
            filtered(
                &schema,
                &view,
                Some((&generation, &pending)),
                &operations,
                &reject
            )
            .unwrap(),
            Some(vec![Value::Null(DataType::Float)])
        );
        assert!(
            generation.segments[&1].volume.columns.get(3).is_err(),
            "rejected rows must not bind aggregate columns"
        );
        // A missing dictionary target also skips the aggregate read, even
        // after both predicate groups have been decoded successfully.
        crate::test_failpoints::fail_cold_read_on(3);
        assert_eq!(
            filtered(
                &schema,
                &view,
                Some((&generation, &pending)),
                &operations,
                &ComparisonExpr::eq("a", Value::text("absent"))
            )
            .unwrap(),
            Some(vec![Value::Null(DataType::Float)])
        );
        assert!(crate::test_failpoints::check_cold_read().is_err());
        // Authority alone hides both metadata-only files. No backing exists,
        // so any eager promotion here would fail.
        drop(manager);
        let hidden = Arc::get_mut(&mut generation).unwrap();
        for segment in Arc::make_mut(&mut hidden.segments).values_mut() {
            segment.volume = Arc::new(segment.volume.to_cold());
        }
        let pending: FxHashSet<_> = [1, 2].into_iter().collect();
        crate::test_failpoints::fail_cold_read_on(1);
        assert_eq!(
            filtered(
                &schema,
                &view,
                Some((&generation, &pending)),
                &operations,
                &filter
            )
            .unwrap(),
            Some(vec![Value::Null(DataType::Float)])
        );
        assert!(crate::test_failpoints::check_cold_read().is_err());
    }

    #[cfg(feature = "test-failpoints")]
    #[test]
    fn captured_aggregate_mapped_defaults_omit_physical_columns() {
        let _guard = crate::test_failpoints::FailpointGuard::new();
        let schema = schema();
        let (registry, store) = store(&schema);
        let manager = SegmentManager::new("aggregate", None);
        add_segment(
            &manager,
            &schema,
            1,
            &[(1, row(1, "old", "x", 1)), (2, row(2, "old", "y", 2))],
            true,
        );
        let (_, mut generation) = manager.capture_with_hot(|| ()).unwrap();
        drop(manager);
        let segments = Arc::get_mut(&mut generation).unwrap();
        let segment = Arc::make_mut(&mut segments.segments).get_mut(&1).unwrap();
        segment.mapping = ColumnMapping {
            is_identity: false,
            sources: (0..schema.columns.len())
                .map(|column| {
                    if column == 1 {
                        ColSource::Default(Value::text("new"))
                    } else if column == 3 {
                        ColSource::Default(Value::Integer(7))
                    } else {
                        ColSource::Volume(column)
                    }
                })
                .collect(),
        };
        let view = CapturedHotView::new(
            store.capture_hot_root(),
            registry.capture_read_epoch(),
            None,
        );
        let pending = FxHashSet::default();
        crate::test_failpoints::fail_cold_read_on(1);
        assert_eq!(
            filtered(
                &schema,
                &view,
                Some((&generation, &pending)),
                &[(AggregateOp::Sum, 3)],
                &ComparisonExpr::eq("a", Value::text("new"))
            )
            .unwrap(),
            Some(vec![Value::Integer(14)])
        );
        let grouped = grouped(
            &schema,
            &view,
            Some((&generation, &pending)),
            &[1, 3],
            &[(AggregateOp::CountStar, 0)],
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            grouped[0].group_values,
            vec![Value::text("new"), Value::Integer(7)]
        );
        assert_eq!(grouped[0].aggregate_values, vec![Value::Integer(2)]);
        assert!(crate::test_failpoints::check_cold_read().is_err());
    }

    #[test]
    fn captured_aggregate_dictionary_windows_match_scalar_filters() {
        let schema = schema();
        let operations = [(AggregateOp::CountStar, 0), (AggregateOp::Sum, 3)];
        for warm in [false, true] {
            for length in [
                7,
                8,
                9,
                DICTIONARY_WINDOW - 1,
                DICTIONARY_WINDOW,
                DICTIONARY_WINDOW + 1,
                DICTIONARY_WINDOW + 9,
            ] {
                let (registry, store) = store(&schema);
                let manager = SegmentManager::new("aggregate", None);
                // Dense leading blocks, a sparse tail, nulls and mismatching
                // second keys all cross the selector's vector/window boundary.
                let rows: Vec<_> = (1..=length)
                    .map(|id| {
                        let mut value = row(
                            id as i64,
                            if id < 512 || id % 31 == 0 { "a" } else { "b" },
                            if id % 3 == 0 { "y" } else { "x" },
                            id as i64,
                        );
                        if id % 19 == 0 {
                            value.set(1, Value::Null(DataType::Text)).unwrap();
                        }
                        if id % 23 == 0 {
                            value.set(2, Value::Null(DataType::Text)).unwrap();
                        }
                        (id as i64, value)
                    })
                    .collect();
                add_segment(&manager, &schema, 1, &rows, warm);
                let (_, generation) = manager.capture_with_hot(|| ()).unwrap();
                let view = CapturedHotView::new(
                    store.capture_hot_root(),
                    registry.capture_read_epoch(),
                    None,
                );
                let pending: FxHashSet<_> = rows
                    .iter()
                    .filter(|(id, _)| id % 17 == 0)
                    .map(|(id, _)| *id)
                    .collect();
                for prefix in 0..=6 {
                    for scalar_first in [false, true] {
                        for target in ["a", "missing"] {
                            let scalar = || {
                                Box::new(ComparisonExpr::new("n", Operator::Gt, Value::Integer(4)))
                                    as Box<dyn Expression>
                            };
                            let mut children: Vec<Box<dyn Expression>> = Vec::new();
                            if scalar_first {
                                children.push(scalar());
                            }
                            for index in 0..prefix {
                                children.push(Box::new(ComparisonExpr::eq(
                                    if index % 2 == 0 { "a" } else { "b" },
                                    Value::text(if index % 2 == 0 { target } else { "x" }),
                                )));
                            }
                            if !scalar_first {
                                children.push(scalar());
                            }
                            children.push(Box::new(ComparisonExpr::new(
                                "b",
                                Operator::Ne,
                                Value::text("y"),
                            )));
                            let mut expression = AndExpr::new(children);
                            expression.prepare_for_schema(&schema);
                            let matches: Vec<_> = rows
                                .iter()
                                .filter(|(id, row)| {
                                    !pending.contains(id) && expression.evaluate(row).unwrap()
                                })
                                .collect();
                            let expected = vec![
                                Value::Integer(matches.len() as i64),
                                if matches.is_empty() {
                                    Value::Null(DataType::Float)
                                } else {
                                    Value::Integer(matches.iter().map(|(id, _)| *id).sum())
                                },
                            ];
                            assert_eq!(filtered(&schema, &view, Some((&generation, &pending)),
                                &operations, &expression).unwrap(), Some(expected),
                                "warm={warm}, length={length}, prefix={prefix}, first={scalar_first}, target={target}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn dictionary_sample_is_only_a_hint_and_keeps_later_window_matches() {
        let schema = schema();
        for warm in [false, true] {
            let (registry, store) = store(&schema);
            let manager = SegmentManager::new("aggregate", None);
            let rows: Vec<_> = (1..=DICTIONARY_WINDOW + 65)
                .map(|id| {
                    // The second target has zero sample hits but matches in
                    // later windows. NULL has the target's dictionary ID on
                    // some rows and must still be excluded.
                    let mut value = row(
                        id as i64,
                        if id <= 64 || id % 3 == 0 { "a" } else { "b" },
                        if id <= 64 { "y" } else { "x" },
                        id as i64,
                    );
                    if id % 17 == 0 {
                        value.set(2, Value::Null(DataType::Text)).unwrap();
                    }
                    (id as i64, value)
                })
                .collect();
            add_segment(&manager, &schema, 1, &rows, warm);
            let (_, generation) = manager.capture_with_hot(|| ()).unwrap();
            let view = CapturedHotView::new(
                store.capture_hot_root(),
                registry.capture_read_epoch(),
                None,
            );
            let pending = FxHashSet::from_iter([69, 1026]);
            let mut expression = AndExpr::new(vec![
                Box::new(ComparisonExpr::eq("a", Value::text("a"))),
                Box::new(ComparisonExpr::eq("b", Value::text("x"))),
            ]);
            expression.prepare_for_schema(&schema);
            let expected: Vec<_> = rows
                .iter()
                .filter(|(id, row)| !pending.contains(id) && expression.evaluate(row).unwrap())
                .collect();
            assert!(!expected.is_empty());
            assert!(expected
                .iter()
                .any(|(id, _)| *id > DICTIONARY_WINDOW as i64));
            assert_eq!(
                filtered(
                    &schema,
                    &view,
                    Some((&generation, &pending)),
                    &[(AggregateOp::CountStar, 0), (AggregateOp::Sum, 3)],
                    &expression,
                )
                .unwrap(),
                Some(vec![
                    Value::Integer(expected.len() as i64),
                    Value::Integer(expected.iter().map(|(id, _)| *id).sum()),
                ]),
            );
        }
    }

    #[test]
    fn captured_aggregate_dictionary_selection_preserves_scalar_error_order() {
        let schema = schema();
        let (registry, store) = store(&schema);
        let manager = SegmentManager::new("aggregate", None);
        add_segment(&manager, &schema, 1, &[(1, row(1, "a", "x", 1))], true);
        let (_, mut generation) = manager.capture_with_hot(|| ()).unwrap();
        drop(manager);
        let segment = Arc::make_mut(&mut Arc::get_mut(&mut generation).unwrap().segments)
            .get_mut(&1)
            .unwrap();
        // A schema/default change can leave an incompatible runtime scalar.
        // A rejecting later dictionary equality cannot conceal its error.
        segment.mapping.sources[2] = ColSource::Default(Value::Integer(10));
        let view = CapturedHotView::new(
            store.capture_hot_root(),
            registry.capture_read_epoch(),
            None,
        );
        let pending = FxHashSet::default();
        for scalar_first in [true, false] {
            let invalid = Box::new(ComparisonExpr::new("b", Operator::Ne, Value::text("10")))
                as Box<dyn Expression>;
            let reject =
                Box::new(ComparisonExpr::eq("a", Value::text("absent"))) as Box<dyn Expression>;
            let expression = AndExpr::new(if scalar_first {
                vec![invalid, reject]
            } else {
                vec![reject, invalid]
            });
            let result = filtered(
                &schema,
                &view,
                Some((&generation, &pending)),
                &[(AggregateOp::CountStar, 0)],
                &expression,
            );
            if scalar_first {
                assert!(result.is_err());
            } else {
                assert_eq!(result.unwrap(), Some(vec![Value::Integer(0)]));
            }
        }
    }

    #[test]
    fn captured_aggregate_reuses_authority_and_dictionary_state_across_groups() {
        let schema = schema();
        let (registry, store) = store(&schema);
        let manager = SegmentManager::new("aggregate", None);
        let rows: Vec<_> = (1..=ROW_GROUP_SIZE + 3)
            .map(|id| {
                (
                    id as i64,
                    row(id as i64, if id % 2 == 0 { "a" } else { "b" }, "x", 1),
                )
            })
            .collect();
        add_segment(&manager, &schema, 1, &rows, true);
        let (_, generation) = manager.capture_with_hot(|| ()).unwrap();
        let view = CapturedHotView::new(
            store.capture_hot_root(),
            registry.capture_read_epoch(),
            None,
        );
        let pending: FxHashSet<_> = (1..=ROW_GROUP_SIZE as i64 + 1).collect();
        let result = grouped(
            &schema,
            &view,
            Some((&generation, &pending)),
            &[1],
            &[(AggregateOp::CountStar, 0), (AggregateOp::Sum, 3)],
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            result_map(result),
            [
                (
                    vec![Value::text("a")],
                    vec![Value::Integer(1), Value::Integer(1)]
                ),
                (
                    vec![Value::text("b")],
                    vec![Value::Integer(1), Value::Integer(1)]
                ),
            ]
            .into_iter()
            .collect()
        );
        let empty_pending = FxHashSet::default();
        let result = grouped(
            &schema,
            &view,
            Some((&generation, &empty_pending)),
            &[1, 2],
            &[(AggregateOp::CountStar, 0)],
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            result_map(result),
            [
                (
                    vec![Value::text("a"), Value::text("x")],
                    vec![Value::Integer((ROW_GROUP_SIZE / 2 + 1) as i64)]
                ),
                (
                    vec![Value::text("b"), Value::text("x")],
                    vec![Value::Integer((ROW_GROUP_SIZE / 2 + 2) as i64)]
                ),
            ]
            .into_iter()
            .collect()
        );
    }

    #[test]
    fn captured_aggregate_changed_schema_preserves_hot_type_errors_and_fallback() {
        let mut schema = schema();
        schema
            .modify_column("b", Some(DataType::Json), Some(true))
            .unwrap();
        let (registry, store) = store(&schema);
        let (writer, _) = registry.begin_transaction();
        registry.start_commit(writer);
        let mut original = row(1, "a", "x", 10);
        original.set(2, Value::json("{}")).unwrap();
        store
            .add_version(1, RowVersion::new(writer, original.clone()))
            .unwrap();
        registry.complete_commit(writer);
        let (reader, _) = registry.begin_transaction();
        let mut table = MVCCTable::new(
            reader,
            store.clone(),
            TransactionVersionStore::new(store, reader),
        );
        table.modify_column("n", DataType::Text, true).unwrap();
        table.modify_column("b", DataType::Text, true).unwrap();
        table.set_read_epoch(registry.capture_read_epoch()).unwrap();
        let mut predicate = ComparisonExpr::new("n", Operator::Ne, Value::text("10"));
        predicate.prepare_for_schema(table.schema());
        let expected = predicate.evaluate(&original).unwrap_err().to_string();
        assert_eq!(
            table
                .compute_filtered_aggregates(&[(AggregateOp::CountStar, 0)], &predicate)
                .unwrap_err()
                .to_string(),
            expected
        );
        assert!(
            table
                .compute_grouped_aggregates(&[2], &[(AggregateOp::CountStar, 0)])
                .unwrap()
                .is_none(),
            "old extension payloads under a scalar schema require the generic path"
        );
        let all = ComparisonExpr::new("id", Operator::Gt, Value::Integer(0));
        assert!(table
            .compute_filtered_aggregates(&[(AggregateOp::Min, 2)], &all)
            .unwrap()
            .is_none());
    }
}
