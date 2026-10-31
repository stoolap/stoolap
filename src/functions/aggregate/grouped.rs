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

//! The groups of an aggregation. Two rows share a group exactly when their
//! keys are equal as DISTINCT and UNION take them, every row lands in a
//! group, and each group runs its calls by the one compiled implementation

use std::hash::{Hash, Hasher};

use hashbrown::HashTable;
use rustc_hash::FxHasher;

use crate::common::{CompactVec, I64Map};
use crate::core::value::NULL_VALUE;
use crate::core::{Result, Row, RowVec, Value};
use crate::functions::{AggregateFunction, AggregateOrder};

use super::{AggregateState, CompiledAggregate};

/// One group's row, or its key values and the error a call's result gave
pub type GroupResult = std::result::Result<Row, (CompactVec<Value>, crate::core::Error)>;

static NULL_KEY: Value = NULL_VALUE;

/// How every group runs one aggregate call
pub enum CallTemplate {
    /// A call with a compiled state of its own
    Compiled(fn() -> CompiledAggregate),
    /// Any other call, by the registry's function with the call's modifiers
    Dynamic {
        function: Box<dyn AggregateFunction>,
        distinct: bool,
        extra_args: Vec<Value>,
        order_by: Vec<AggregateOrder>,
        /// The error an extra argument gave, the result of the call once
        /// there is a group to read
        failed: Option<crate::core::Error>,
    },
}

impl CallTemplate {
    /// The template of a call to `function`, compiled when its name has a
    /// compiled state. A call whose extra argument gave `failed` keeps it
    /// for `into_rows` to report, and no result of it is read
    pub fn new(
        function: Box<dyn AggregateFunction>,
        is_count_star: bool,
        distinct: bool,
        extra_args: &[Value],
        order_by: Vec<AggregateOrder>,
        failed: Option<crate::core::Error>,
    ) -> Self {
        let compiled = failed
            .is_none()
            .then(|| CompiledAggregate::constructor(function.name(), is_count_star, distinct))
            .flatten();
        match compiled {
            Some(make) => CallTemplate::Compiled(make),
            None => CallTemplate::Dynamic {
                function,
                distinct,
                extra_args: extra_args.to_vec(),
                order_by,
                failed,
            },
        }
    }

    /// A new, empty run of the call
    #[inline]
    pub fn instance(&self) -> CompiledAggregate {
        match self {
            CallTemplate::Compiled(make) => make(),
            CallTemplate::Dynamic {
                function,
                distinct,
                extra_args,
                order_by,
                ..
            } => CompiledAggregate::Dynamic(AggregateState::new(
                function.clone_box(),
                *distinct,
                extra_args,
                order_by.clone(),
            )),
        }
    }

    /// The error the call's extra argument gave, if it gave one
    fn failure(&self) -> Option<&crate::core::Error> {
        match self {
            CallTemplate::Dynamic { failed, .. } => failed.as_ref(),
            CallTemplate::Compiled(_) => None,
        }
    }

    pub(crate) fn duplicate(&self) -> CallTemplate {
        match self {
            CallTemplate::Compiled(make) => CallTemplate::Compiled(*make),
            CallTemplate::Dynamic {
                function,
                distinct,
                extra_args,
                order_by,
                failed,
            } => CallTemplate::Dynamic {
                function: function.clone_box(),
                distinct: *distinct,
                extra_args: extra_args.clone(),
                order_by: order_by.clone(),
                failed: failed.clone(),
            },
        }
    }

    /// This call with `extra_args` for its extra arguments, and `failed`
    /// the error one of them gave; a compiled call takes none
    pub(crate) fn with_extra_args(
        &self,
        extra_args: Vec<Value>,
        failed: Option<crate::core::Error>,
    ) -> CallTemplate {
        match self {
            CallTemplate::Compiled(make) => CallTemplate::Compiled(*make),
            CallTemplate::Dynamic {
                function,
                distinct,
                order_by,
                ..
            } => CallTemplate::Dynamic {
                function: function.clone_box(),
                distinct: *distinct,
                extra_args,
                order_by: order_by.clone(),
                failed,
            },
        }
    }
}

/// A row's group key, read where the row holds it
pub trait GroupKey {
    fn width(&self) -> usize;
    fn at(&self, i: usize) -> &Value;
}

impl GroupKey for [Value] {
    #[inline(always)]
    fn width(&self) -> usize {
        self.len()
    }
    #[inline(always)]
    fn at(&self, i: usize) -> &Value {
        &self[i]
    }
}

/// The key some columns of a row form; a column past the row's end is NULL
pub struct ColumnsKey<'a> {
    pub values: &'a [Value],
    pub columns: &'a [usize],
}

impl GroupKey for ColumnsKey<'_> {
    #[inline(always)]
    fn width(&self) -> usize {
        self.columns.len()
    }
    #[inline(always)]
    fn at(&self, i: usize) -> &Value {
        self.values.get(self.columns[i]).unwrap_or(&NULL_KEY)
    }
}

/// A key's hash, equal for equal keys. A text equals only a text, so it
/// hashes its bytes; every other value hashes as `Value` does
#[inline(always)]
fn hash_key<K: GroupKey + ?Sized>(key: &K) -> u64 {
    let mut hasher = FxHasher::default();
    for i in 0..key.width() {
        match key.at(i) {
            Value::Text(text) => text.as_str().hash(&mut hasher),
            value => value.hash(&mut hasher),
        }
    }
    hasher.finish()
}

/// The groups of one aggregation, in the order their first rows came, each
/// with its key and one run of every call
pub struct GroupedAggregates {
    calls: Vec<CallTemplate>,
    width: usize,
    table: HashTable<u32>,
    /// `width` values per group
    keys: Vec<Value>,
    /// One run of every call per group
    aggregates: Vec<CompiledAggregate>,
    /// The group of each INTEGER key of one value seen so far, a cache of
    /// the table: equal integers are equal keys
    integers: I64Map<u32>,
    groups: usize,
    cap: usize,
}

impl GroupedAggregates {
    /// No groups yet, for keys of `width` values
    pub fn new(calls: Vec<CallTemplate>, width: usize) -> Self {
        Self {
            calls,
            width,
            table: HashTable::new(),
            keys: Vec::new(),
            aggregates: Vec::new(),
            integers: I64Map::new(),
            groups: 0,
            cap: usize::MAX,
        }
    }

    /// The one group of an aggregation without GROUP BY, there with no rows
    pub fn global(calls: Vec<CallTemplate>) -> Self {
        let mut groups = Self::new(calls, 0);
        groups.group(&[] as &[Value]);
        groups
    }

    /// No group past the first `cap`; their rows are not aggregated
    pub fn with_group_cap(mut self, cap: usize) -> Self {
        self.cap = cap;
        self
    }

    /// No groups and no group cap, for the same calls and keys
    pub fn empty_like(&self) -> Self {
        Self::new(
            self.calls.iter().map(CallTemplate::duplicate).collect(),
            self.width,
        )
    }

    pub fn is_empty(&self) -> bool {
        self.groups == 0
    }

    /// Whether runs over disjoint rows merge into the run over all of them;
    /// under a group cap a part would drop rows of a group another part keeps
    pub fn can_merge(&self) -> bool {
        self.cap == usize::MAX
            && self.calls.iter().all(|call| match call {
                CallTemplate::Compiled(make) => make().can_merge(),
                CallTemplate::Dynamic { .. } => false,
            })
    }

    /// The group `key` belongs to, made at its first row; None for a new
    /// key past the group cap
    #[inline(always)]
    pub fn group<K: GroupKey + ?Sized>(&mut self, key: &K) -> Option<usize> {
        debug_assert_eq!(key.width(), self.width, "group key width");
        if self.width == 0 && self.groups == 1 {
            return Some(0);
        }
        if self.width == 1 {
            if let Value::Integer(i) = key.at(0) {
                if let Some(&group) = self.integers.get(*i) {
                    return Some(group as usize);
                }
                return self.integer_group(key, *i);
            }
        }
        self.find_or_add(key)
    }

    /// `group` for an INTEGER key not cached yet
    #[cold]
    #[inline(never)]
    fn integer_group<K: GroupKey + ?Sized>(&mut self, key: &K, i: i64) -> Option<usize> {
        let group = self.find_or_add(key)?;
        self.integers.insert(i, group as u32);
        Some(group)
    }

    #[inline]
    fn find_or_add<K: GroupKey + ?Sized>(&mut self, key: &K) -> Option<usize> {
        let hash = hash_key(key);
        let width = self.width;
        let keys = &self.keys;
        let found = self.table.find(hash, |&group| {
            let stored = &keys[group as usize * width..][..width];
            (0..width).all(|i| key.at(i) == &stored[i])
        });
        if let Some(&group) = found {
            return Some(group as usize);
        }
        if self.groups >= self.cap {
            return None;
        }
        Some(self.add_group(hash, key))
    }

    #[cold]
    fn add_group<K: GroupKey + ?Sized>(&mut self, hash: u64, key: &K) -> usize {
        let group = self.groups;
        self.groups += 1;
        for i in 0..self.width {
            self.keys.push(key.at(i).clone());
        }
        self.aggregates
            .extend(self.calls.iter().map(CallTemplate::instance));
        let (keys, width) = (&self.keys, self.width);
        self.table.insert_unique(hash, group as u32, |&g| {
            hash_key(&keys[g as usize * width..][..width])
        });
        group
    }

    /// The runs of every call in `group`
    #[inline(always)]
    pub fn aggregates(&mut self, group: usize) -> &mut [CompiledAggregate] {
        let calls = self.calls.len();
        &mut self.aggregates[group * calls..][..calls]
    }

    /// Folds the groups of a run over other rows into these, see `can_merge`
    pub fn merge(&mut self, other: GroupedAggregates) {
        debug_assert!(self.can_merge(), "merge needs runs that merge");
        let calls = self.calls.len();
        for group in 0..other.groups {
            let key = &other.keys[group * other.width..][..other.width];
            if let Some(into) = self.group(key) {
                let runs = &other.aggregates[group * calls..][..calls];
                for (run, more) in self.aggregates(into).iter_mut().zip(runs) {
                    run.merge(more);
                }
            }
        }
    }

    /// One row per group: its key values, then the result of every call
    pub fn into_rows(self) -> Result<RowVec> {
        let mut rows = RowVec::with_capacity(self.groups);
        self.finish_each(|group, values| {
            let values = values.map_err(|(_, error)| error)?;
            rows.push((group as i64, Row::from_compact_vec(values)));
            Ok(())
        })?;
        Ok(rows)
    }

    /// `into_rows` with each group's own error in its row's place, beside
    /// the group's key values
    pub fn into_group_results(self) -> Result<Vec<GroupResult>> {
        let mut rows = Vec::with_capacity(self.groups);
        self.finish_each(|_, values| {
            rows.push(values.map(Row::from_compact_vec));
            Ok(())
        })?;
        Ok(rows)
    }

    /// Hands `each` every group's key values and call results in order;
    /// a call whose result fails leaves the key values and its error
    fn finish_each(
        self,
        mut each: impl FnMut(
            usize,
            std::result::Result<CompactVec<Value>, (CompactVec<Value>, crate::core::Error)>,
        ) -> Result<()>,
    ) -> Result<()> {
        if self.groups > 0 {
            if let Some(error) = self.calls.iter().find_map(CallTemplate::failure) {
                return Err(error.clone());
            }
        }
        let calls = self.calls.len();
        let mut keys = self.keys.into_iter();
        'groups: for group in 0..self.groups {
            let mut values = CompactVec::with_capacity(self.width + calls);
            values.extend(keys.by_ref().take(self.width));
            for run in &self.aggregates[group * calls..][..calls] {
                match run.result() {
                    Ok(value) => values.push(value),
                    Err(error) => {
                        each(group, Err((values, error)))?;
                        continue 'groups;
                    }
                }
            }
            each(group, Ok(values))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::functions::aggregate::CountFunction;

    fn count_star() -> Vec<CallTemplate> {
        vec![CallTemplate::new(
            Box::new(CountFunction::default()),
            true,
            false,
            &[],
            Vec::new(),
            None,
        )]
    }

    fn counts(keys: &[Value]) -> Vec<String> {
        let mut groups = GroupedAggregates::new(count_star(), 1);
        for key in keys {
            let group = groups.group(std::slice::from_ref(key)).unwrap();
            groups.aggregates(group)[0].accumulate(key);
        }
        groups
            .into_rows()
            .unwrap()
            .iter()
            .map(|(_, row)| format!("{:?}|{:?}", row.get(0), row.get(1)))
            .collect()
    }

    #[test]
    fn equal_keys_share_a_group_and_keep_the_first_value() {
        let got = counts(&[
            Value::Integer(1),
            Value::Float(1.0),
            Value::Float(-0.0),
            Value::Integer(0),
            Value::Float(f64::NAN),
            Value::Float(f64::NAN),
            Value::null_unknown(),
            Value::Null(crate::core::DataType::Integer),
            Value::text("1"),
            Value::Boolean(true),
        ]);
        assert_eq!(got.len(), 6, "{got:?}");
        assert!(got[0].starts_with("Some(Integer(1))") && got[0].ends_with("Integer(2))"));
        assert!(got[1].starts_with("Some(Float(-0.0))") && got[1].ends_with("Integer(2))"));
    }

    #[test]
    fn group_cap_stops_new_groups_only() {
        let mut groups = GroupedAggregates::new(count_star(), 1).with_group_cap(1);
        assert_eq!(groups.group(&[Value::Integer(1)][..]), Some(0));
        assert_eq!(groups.group(&[Value::Integer(2)][..]), None);
        assert_eq!(groups.group(&[Value::Float(1.0)][..]), Some(0));
    }

    #[test]
    fn global_has_its_group_without_rows() {
        let rows = GroupedAggregates::global(count_star()).into_rows().unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].1.get(0), Some(&Value::Integer(0)));
    }
}
