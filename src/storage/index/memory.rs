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

//! Physical ownership helpers shared by the hot index implementations.

use crate::common::{MemoryAccount, MemoryCharge};
use crate::core::{Error, Result};

/// One immutable origin per index. Keep this field after all owned collections
/// so their buffers and key aliases are dropped before the metadata charge.
#[derive(Default)]
pub(super) struct IndexMemory {
    account: Option<MemoryAccount>,
    _metadata: Option<MemoryCharge>,
    _object: Option<MemoryCharge>,
}

impl IndexMemory {
    pub fn account(&self) -> Option<&MemoryAccount> {
        self.account.as_ref()
    }

    pub fn value_arc(
        &self,
        value: &crate::core::Value,
    ) -> crate::common::CompactArc<crate::core::Value> {
        match self.account() {
            Some(account) => {
                crate::common::CompactArc::new_in(value.clone().into_hot(account), account)
            }
            None => crate::common::CompactArc::new(value.clone()),
        }
    }

    pub fn value(&self, value: &crate::core::Value) -> crate::core::Value {
        match self.account() {
            Some(account) => value.clone().into_hot(account),
            None => value.clone(),
        }
    }

    /// Startup-only adoption preserves key identity and rejects foreign backing.
    pub fn adopt_value_arc(
        value: &crate::common::CompactArc<crate::core::Value>,
        account: &MemoryAccount,
    ) -> Result<()> {
        use crate::common::MemoryAdoption;
        if !value.try_adopt_hot(account)
            || value.try_adopt_shallow(account) == MemoryAdoption::ForeignEngine
        {
            return Err(Error::internal(
                "prebuilt index key belongs to another engine",
            ));
        }
        Ok(())
    }

    /// Returns false for idempotent same-engine attachment. An index is never
    /// reparented after a first account has taken physical ownership.
    pub fn needs_attachment(&self, parent: &MemoryAccount) -> Result<bool> {
        match &self.account {
            None => Ok(true),
            Some(origin) if origin.same_engine(parent) => Ok(false),
            Some(_) => Err(Error::internal("index belongs to another memory account")),
        }
    }

    pub fn attach<T>(&mut self, parent: &MemoryAccount, metadata_bytes: usize) {
        debug_assert!(self.account.is_none());
        let account = parent.clone();
        self._metadata = Some(MemoryCharge::new(&account, metadata_bytes));
        // Bound the index object, its inline handles and std::Arc's opaque control block.
        self._object = Some(MemoryCharge::conservative(
            &account,
            std::mem::size_of::<T>() + 4 * std::mem::size_of::<usize>(),
        ));
        self.account = Some(account);
    }
}

/// Grow a known vector allocation with explicit old/new ownership overlap.
/// Existing-capacity mutations do not touch counters or allocate a charge.
pub(super) fn reserve_vec<T>(
    values: &mut Vec<T>,
    needed: usize,
    charge: &mut Option<MemoryCharge>,
) {
    if needed <= values.capacity() {
        return;
    }
    let capacity = needed.max(values.capacity().saturating_mul(2)).max(4);
    let mut replacement = Vec::with_capacity(capacity);
    let replacement_charge = charge.as_ref().map(|charge| {
        MemoryCharge::new(
            charge.account(),
            replacement.capacity() * std::mem::size_of::<T>(),
        )
    });
    replacement.append(values);
    let old = std::mem::replace(values, replacement);
    let old_charge = std::mem::replace(charge, replacement_charge);
    drop(old);
    drop(old_charge);
}

#[cfg(test)]
mod ownership_tests {
    use super::*;
    use crate::core::{DataType, Value};
    use crate::storage::index::{BTreeIndex, BitmapIndex, HashIndex, MultiColumnIndex};
    use crate::storage::traits::Index;
    use std::sync::Arc;

    fn keyed_indexes() -> Vec<Box<dyn Index>> {
        vec![
            Box::new(BTreeIndex::new(
                "b".into(),
                "t".into(),
                0,
                "v".into(),
                DataType::Text,
                false,
                0,
            )),
            Box::new(BitmapIndex::new(
                "m".into(),
                "t".into(),
                vec!["v".into()],
                vec![0],
                vec![DataType::Text],
                false,
                0,
            )),
            Box::new(HashIndex::new(
                "h".into(),
                "t".into(),
                vec!["v".into()],
                vec![0],
                vec![DataType::Text],
                false,
                0,
            )),
            Box::new(MultiColumnIndex::new(
                "c".into(),
                "t".into(),
                vec!["v".into()],
                vec![0],
                vec![DataType::Text],
                false,
                0,
            )),
        ]
    }

    #[test]
    fn prebuilt_index_attachment_and_last_owner_cover_every_keyed_family() {
        for mut index in keyed_indexes() {
            let root = MemoryAccount::new();
            let origin = root.child();
            let baseline = root.snapshot().accounted_bytes;
            let key = Value::text("prebuilt index text with independently shared heap backing");
            index.add(std::slice::from_ref(&key), 1, 0).unwrap();
            index.attach_memory_account(&origin).unwrap();
            let attached = root.snapshot();
            index.attach_memory_account(&origin).unwrap();
            assert_eq!(root.snapshot(), attached);
            assert!(index.attach_memory_account(&MemoryAccount::new()).is_err());
            for row in 2..100 {
                index.add(std::slice::from_ref(&key), row, 0).unwrap();
            }
            assert_eq!(
                index.get_row_ids_equal(std::slice::from_ref(&key)).len(),
                99
            );
            assert!(
                origin.snapshot().retained_bytes > 0,
                "index attribution must remain on its table origin"
            );
            drop(key);
            let owner: Arc<dyn Index> = Arc::from(index);
            let alias = owner.clone();
            let retained = root.snapshot().accounted_bytes;
            drop(owner);
            assert_eq!(root.snapshot().accounted_bytes, retained);
            alias
                .remove_batch_ids(&(1..50).collect::<Vec<_>>())
                .unwrap()
                .unwrap();
            drop(alias);
            assert_eq!(root.snapshot().retained_bytes, 0);
            assert_eq!(root.snapshot().accounted_bytes, baseline);
        }
    }

    #[test]
    fn returned_text_alias_outlives_btree_and_foreign_insert_copies_origin() {
        let first = MemoryAccount::new();
        let second = MemoryAccount::new();
        let input = Value::text("a foreign engine text backing that must never be reparented")
            .into_hot(&first);
        let mut index = BTreeIndex::new(
            "b".into(),
            "t".into(),
            0,
            "v".into(),
            DataType::Text,
            false,
            0,
        );
        index.attach_memory_account(&second).unwrap();
        index.add(std::slice::from_ref(&input), 1, 0).unwrap();
        let alias = index.get_min_value().unwrap();
        let Value::Text(text) = &alias else {
            unreachable!()
        };
        assert!(text.memory_account().unwrap().same_engine(&second));
        drop(index);
        assert!(second.snapshot().retained_bytes > 0);
        drop(alias);
        assert_eq!(second.snapshot().retained_bytes, 0);
        assert!(first.snapshot().retained_bytes > 0);
        drop(input);
        assert_eq!(first.snapshot().retained_bytes, 0);
    }

    #[test]
    fn lookup_bounds_do_not_adopt_external_text_into_hot_storage() {
        for mut index in keyed_indexes() {
            let account = MemoryAccount::new();
            index.attach_memory_account(&account).unwrap();
            index
                .add(
                    &[Value::text("stored long text that belongs to the index")],
                    1,
                    0,
                )
                .unwrap();
            let query = Value::text("external query text with a separate heap allocation");
            let before = account.snapshot();
            index.find(std::slice::from_ref(&query)).unwrap();
            if index.as_any().is::<BTreeIndex>() {
                for op in [
                    crate::core::Operator::Lt,
                    crate::core::Operator::Lte,
                    crate::core::Operator::Gt,
                    crate::core::Operator::Gte,
                ] {
                    index
                        .find_with_operator(op, std::slice::from_ref(&query))
                        .unwrap();
                }
            }
            assert_eq!(account.snapshot(), before);
            let Value::Text(text) = &query else {
                unreachable!()
            };
            assert!(text.memory_account().is_none());
        }
    }

    #[test]
    fn prebuilt_foreign_payload_attachment_is_rejected() {
        let first = MemoryAccount::new();
        let second = MemoryAccount::new();
        let key = Value::text("foreign-owned backing inside an otherwise untracked index")
            .into_hot(&first);
        for mut index in keyed_indexes() {
            index.add(std::slice::from_ref(&key), 1, 0).unwrap();
            assert!(index.attach_memory_account(&second).is_err());
            assert!(index.memory_account().is_none());
            drop(index);
            assert_eq!(second.snapshot().retained_bytes, 0);
        }
    }
}
