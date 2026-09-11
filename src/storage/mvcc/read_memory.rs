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

use std::cell::RefCell;
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::Arc;

use super::memory::{ChargedSmallVec, ChargedWeak, HotObjectCharge, RetainedBytes};
use crate::core::{Row, Value};
use parking_lot::Mutex;

static EXPORTED_PAYLOAD_BYTES: RetainedBytes = RetainedBytes::new();

thread_local! {
    static ACTIVE_SCOPE: RefCell<Option<Arc<ReadScope>>> = const { RefCell::new(None) };
}

pub(crate) fn exported_payload_bytes() -> usize {
    EXPORTED_PAYLOAD_BYTES.get()
}

pub(crate) fn current_scope() -> Option<Arc<ReadScope>> {
    ACTIVE_SCOPE.with(|scope| scope.borrow().clone())
}

pub(crate) fn charge_export(row: &Row) {
    ACTIVE_SCOPE.with(|scope| {
        if let Some(scope) = scope.borrow().as_ref() {
            scope.add(row.heap_bytes());
        }
    });
}

pub(crate) fn charge_value_export(value: &Value) {
    charge_bytes_export(value.heap_bytes() as u128);
}

pub(crate) fn charge_bytes_export(bytes: u128) {
    if bytes == 0 {
        return;
    }
    ACTIVE_SCOPE.with(|scope| {
        if let Some(scope) = scope.borrow().as_ref() {
            scope.add(bytes);
        }
    });
}

#[derive(Default)]
pub(crate) struct ReadScope {
    bytes: RetainedBytes,
    imports: Mutex<ChargedSmallVec<[ChargedWeak<SharedPayloadCharge>; 2]>>,
    last_import: AtomicPtr<SharedPayloadCharge>,
    _object: HotObjectCharge<Self>,
}

impl ReadScope {
    #[cfg(test)]
    pub fn exported_bytes(&self) -> u128 {
        self.bytes.get_wide()
    }

    fn add(&self, bytes: u128) {
        EXPORTED_PAYLOAD_BYTES.add(bytes);
        self.bytes.add(bytes);
    }

    pub fn enter(self: &Arc<Self>) -> ReadScopeGuard {
        let previous = ACTIVE_SCOPE.with(|scope| scope.replace(Some(Arc::clone(self))));
        ReadScopeGuard { previous }
    }

    fn import(&self, source: &Arc<SharedPayloadCharge>) {
        let pointer = Arc::as_ptr(source).cast_mut();
        if self.last_import.load(Ordering::Acquire) == pointer {
            return;
        }
        let mut imports = self.imports.lock();
        if !imports.iter().any(|owner| owner.as_ptr() == pointer) {
            self.add(source.payload.bytes);
            imports.push(ChargedWeak::new(source));
        }
        self.last_import.store(pointer, Ordering::Release);
    }
}

impl Drop for ReadScope {
    fn drop(&mut self) {
        EXPORTED_PAYLOAD_BYTES.remove(self.bytes.get_wide());
    }
}

pub(crate) struct ReadScopeGuard {
    previous: Option<Arc<ReadScope>>,
}

impl Drop for ReadScopeGuard {
    fn drop(&mut self) {
        ACTIVE_SCOPE.with(|scope| scope.replace(self.previous.take()));
    }
}

pub(crate) struct ExportBatch {
    scope: Option<Arc<ReadScope>>,
    bytes: u128,
}

impl ExportBatch {
    pub fn new() -> Self {
        Self {
            scope: current_scope(),
            bytes: 0,
        }
    }

    pub fn record(&mut self, row: &Row) {
        if self.scope.is_some() {
            self.bytes += row.heap_bytes();
        }
    }

    pub fn capture(&mut self, row: &Row) -> Row {
        self.record(row);
        row.clone()
    }

    pub fn record_value(&mut self, value: &Value) {
        if self.scope.is_some() {
            self.bytes += value.heap_bytes() as u128;
        }
    }

    pub fn capture_value(&mut self, value: &Value) -> Value {
        self.record_value(value);
        value.clone()
    }
}

impl Drop for ExportBatch {
    fn drop(&mut self) {
        if let Some(scope) = &self.scope {
            scope.add(self.bytes);
        }
    }
}

#[derive(Debug)]
pub(crate) struct PayloadCharge {
    bytes: u128,
}

impl PayloadCharge {
    #[cfg(test)]
    pub(crate) fn bytes(&self) -> u128 {
        self.bytes
    }

    pub fn unshared(bytes: u128) -> Self {
        EXPORTED_PAYLOAD_BYTES.add(bytes);
        Self { bytes }
    }

    pub fn add(&mut self, bytes: u128) {
        EXPORTED_PAYLOAD_BYTES.add(bytes);
        self.bytes += bytes;
    }
}

pub(crate) struct SharedPayloadCharge {
    payload: PayloadCharge,
    _object: HotObjectCharge<Self>,
}

impl std::fmt::Debug for SharedPayloadCharge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.payload.fmt(f)
    }
}

impl SharedPayloadCharge {
    pub fn new(bytes: u128) -> Arc<Self> {
        Arc::new(Self {
            payload: PayloadCharge::unshared(bytes),
            _object: HotObjectCharge::new(),
        })
    }

    pub fn import(self: &Arc<Self>) {
        ACTIVE_SCOPE.with(|scope| {
            if let Some(scope) = scope.borrow().as_ref() {
                scope.import(self);
            }
        });
    }
}

impl Drop for PayloadCharge {
    fn drop(&mut self) {
        EXPORTED_PAYLOAD_BYTES.remove(self.bytes);
    }
}

impl Clone for PayloadCharge {
    fn clone(&self) -> Self {
        Self::unshared(self.bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scope_restores_nested_activation_and_batches_exports() {
        let outer = Arc::new(ReadScope::default());
        let inner = Arc::new(ReadScope::default());
        let row = Row::from_values(vec![Value::text(
            "a retained string longer than inline storage",
        )]);
        assert!(current_scope().is_none());
        {
            let _outer = outer.enter();
            {
                let _inner = inner.enter();
                let mut batch = ExportBatch::new();
                batch.record(&row);
                batch.record(&row);
                assert_eq!(inner.bytes.get(), 0);
            }
            assert_eq!(inner.bytes.get_wide(), row.heap_bytes() * 2);
            charge_export(&row);
            assert_eq!(outer.bytes.get_wide(), row.heap_bytes());
        }
        assert!(current_scope().is_none());
    }

    #[test]
    fn cache_import_deduplicates_and_survives_cache_eviction() {
        let scope = Arc::new(ReadScope::default());
        let _active = scope.enter();
        let first = SharedPayloadCharge::new(512);
        let second = SharedPayloadCharge::new(1024);
        first.import();
        second.import();
        first.import();
        assert_eq!(scope.bytes.get(), 1536);
        let identity = Arc::downgrade(&first);
        drop(first);
        assert!(identity.upgrade().is_none());
        assert_eq!(scope.bytes.get(), 1536);
        let replacement = SharedPayloadCharge::new(256);
        replacement.import();
        assert_eq!(scope.bytes.get(), 1792);
    }

    #[test]
    fn scopes_restore_after_unwind_and_release_wide_totals() {
        let scope = Arc::new(ReadScope::default());
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _active = scope.enter();
            scope.add(usize::MAX as u128 + 1024);
            panic!("leave active scope");
        }));
        assert!(outcome.is_err());
        assert!(current_scope().is_none());
        assert_eq!(scope.bytes.get_wide(), usize::MAX as u128 + 1024);
        drop(scope);
    }
}
