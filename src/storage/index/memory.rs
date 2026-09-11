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

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;

use crate::common::{CompactArc, I64Map};
use crate::core::Value;
pub(super) use crate::storage::mvcc::memory::hash_table_bytes;
use crate::storage::mvcc::memory::{
    arc_allocation_bytes, ChargedSmallVec, HotObjectCharge, RetainedBytes, TableMemory,
};

pub(super) fn value_bytes(value: &CompactArc<Value>) -> u128 {
    (2 * std::mem::size_of::<usize>() + std::mem::size_of::<Value>()) as u128
        + value.heap_bytes() as u128
}

pub(super) fn btree_node_bytes<K, V>(entries: usize) -> u128 {
    // Standard-library nodes hold 11 keys, with at least five per non-root.
    let node_bound =
        16 * (std::mem::size_of::<K>() + std::mem::size_of::<V>() + std::mem::size_of::<usize>());
    (1 + entries / 5) as u128 * node_bound as u128
}

pub(super) struct IndexValueMap {
    pub map: I64Map<CompactArc<Value>>,
    pub payload_bytes: u128,
}

impl IndexValueMap {
    pub fn insert(&mut self, row_id: i64, value: CompactArc<Value>) {
        self.payload_bytes += value_bytes(&value);
        if let Some(previous) = self.map.insert(row_id, value) {
            self.payload_bytes -= value_bytes(&previous);
        }
    }

    pub fn remove(&mut self, row_id: i64) -> Option<CompactArc<Value>> {
        let value = self.map.remove(row_id)?;
        self.payload_bytes -= value_bytes(&value);
        Some(value)
    }

    pub fn requested_bytes(&self) -> u128 {
        self.map.allocation_bytes() as u128 + self.payload_bytes
    }
}

impl std::ops::Deref for IndexValueMap {
    type Target = I64Map<CompactArc<Value>>;

    fn deref(&self) -> &Self::Target {
        &self.map
    }
}

/// Retained-allocation account supplied by a built-in index.
/// The engine registers this handle independently of catalog membership.
pub struct IndexMemory {
    requested: RetainedBytes,
    estimated: RetainedBytes,
    active: AtomicBool,
    owners: Mutex<ChargedSmallVec<[Arc<TableMemory>; 1]>>,
    _object: HotObjectCharge<Self>,
}

impl IndexMemory {
    pub(crate) fn register(self: &Arc<Self>, table: &Arc<TableMemory>) {
        let mut owners = self.owners.lock();
        if owners.iter().any(|owner| Arc::ptr_eq(owner, table)) {
            return;
        }
        owners.push(Arc::clone(table));
        table.register_index(self);
    }

    pub(crate) fn requested_bytes(&self) -> usize {
        if self.active.load(Ordering::Acquire) {
            self.requested.get()
        } else {
            0
        }
    }

    pub(crate) fn estimated_bytes(&self) -> usize {
        if self.active.load(Ordering::Acquire) {
            self.estimated.get()
        } else {
            0
        }
    }

    pub(crate) fn resize(&self, before: u128, after: u128) {
        self.requested.resize(before, after);
    }

    pub(crate) fn resize_estimate(&self, before: u128, after: u128) {
        self.estimated.resize(before, after);
    }
}

// Place after index containers so their allocations die before the charge.
pub(crate) struct IndexMemoryOwner {
    pub account: Arc<IndexMemory>,
}

impl IndexMemoryOwner {
    pub fn new<T>(requested: u128, estimated: u128) -> Self {
        let account = Arc::new(IndexMemory {
            requested: RetainedBytes::new(),
            estimated: RetainedBytes::new(),
            active: AtomicBool::new(true),
            owners: Mutex::new(ChargedSmallVec::default()),
            _object: HotObjectCharge::new(),
        });
        account
            .requested
            .add(requested + arc_allocation_bytes::<T>() as u128);
        account.estimated.add(estimated);
        Self { account }
    }
}

impl Drop for IndexMemoryOwner {
    fn drop(&mut self) {
        self.account.active.store(false, Ordering::Release);
    }
}
