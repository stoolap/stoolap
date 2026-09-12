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
    static ACTIVE_SCOPE: RefCell<ActiveScope> = const { RefCell::new(ActiveScope::Inactive) };
    static RECYCLED_SCOPE: RefCell<Option<Arc<ReadScope>>> = const { RefCell::new(None) };
}

fn new_scope() -> Arc<ReadScope> {
    RECYCLED_SCOPE
        .try_with(|scope| scope.take())
        .ok()
        .flatten()
        .unwrap_or_default()
}

enum ActiveScope {
    Inactive,
    Active(Option<Arc<ReadScope>>),
}

impl ActiveScope {
    fn activate(&mut self) -> Option<Self> {
        match self {
            Self::Inactive => Some(std::mem::replace(self, Self::Active(None))),
            Self::Active(_) => None,
        }
    }
}

pub(crate) fn exported_payload_bytes() -> usize {
    EXPORTED_PAYLOAD_BYTES.get()
}

pub(crate) fn current_scope() -> Option<Arc<ReadScope>> {
    ACTIVE_SCOPE.with(|scope| match &*scope.borrow() {
        ActiveScope::Active(owner) => owner.clone(),
        ActiveScope::Inactive => None,
    })
}

pub(crate) fn charge_export(row: &Row) {
    ACTIVE_SCOPE.with(|scope| {
        if let ActiveScope::Active(owner) = &mut *scope.borrow_mut() {
            let bytes = row.heap_bytes();
            if bytes != 0 {
                owner.get_or_insert_with(new_scope).add(bytes);
            }
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
        if let ActiveScope::Active(owner) = &mut *scope.borrow_mut() {
            owner.get_or_insert_with(new_scope).add(bytes);
        }
    });
}

/// Retained storage payload ownership managed by the executor.
#[derive(Default)]
pub struct ReadScope {
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

    pub(crate) fn enter(self: &Arc<Self>) -> ReadScopeGuard<'static> {
        let previous =
            ACTIVE_SCOPE.with(|scope| scope.replace(ActiveScope::Active(Some(Arc::clone(self)))));
        ReadScopeGuard {
            previous: Some(previous),
            retain: None,
            state: None,
        }
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

#[derive(Default)]
pub(crate) struct ResultReadScope(Option<Arc<ReadScope>>);

impl ResultReadScope {
    pub fn retain(&mut self, scope: &mut Option<Arc<ReadScope>>) {
        match (&self.0, scope.as_ref()) {
            (None, _) => self.0 = scope.take(),
            (Some(retained), Some(incoming)) if Arc::ptr_eq(retained, incoming) => {
                scope.take();
            }
            _ => {}
        }
    }
}

impl Drop for ResultReadScope {
    fn drop(&mut self) {
        let Some(mut scope) = self.0.take() else {
            return;
        };
        let Some(owner) = Arc::get_mut(&mut scope) else {
            return;
        };
        EXPORTED_PAYLOAD_BYTES.remove(owner.bytes.get_wide());
        owner.bytes = RetainedBytes::new();
        *owner.imports.get_mut() = ChargedSmallVec::default();
        *owner.last_import.get_mut() = std::ptr::null_mut();
        // One idle scope stays charged until reuse or thread destruction.
        let _ = RECYCLED_SCOPE.try_with(|spare| spare.replace(Some(scope)));
    }
}

#[derive(Clone, Default)]
pub(crate) struct DeferredExports {
    bound: bool,
    owner: std::sync::OnceLock<Arc<ReadScope>>,
}

impl DeferredExports {
    #[cfg(test)]
    pub fn scope(&self) -> Option<&Arc<ReadScope>> {
        self.owner.get()
    }

    pub fn bind(&mut self, scope: Option<&Arc<ReadScope>>) {
        self.bound = true;
        if let Some(scope) = scope {
            self.owner.get_or_init(|| Arc::clone(scope));
        }
    }

    pub fn add(&self, bytes: u128) {
        if bytes == 0 {
            return;
        }
        if !self.bound {
            charge_bytes_export(bytes);
            return;
        }
        let scope = self.owner.get_or_init(|| {
            ACTIVE_SCOPE.with(|active| match &mut *active.borrow_mut() {
                ActiveScope::Active(scope) => Arc::clone(scope.get_or_insert_with(new_scope)),
                ActiveScope::Inactive => new_scope(),
            })
        });
        scope.add(bytes);
    }
}

impl Drop for DeferredExports {
    fn drop(&mut self) {
        drop(ResultReadScope(self.owner.take()));
    }
}

pub(crate) struct ReadScopeGuard<'a> {
    previous: Option<ActiveScope>,
    retain: Option<&'a RefCell<Option<Arc<ReadScope>>>>,
    state: Option<&'a RefCell<ActiveScope>>,
}

impl ReadScopeGuard<'static> {
    pub fn lazy() -> Self {
        let previous = ACTIVE_SCOPE.with(|scope| scope.borrow_mut().activate());
        Self {
            previous,
            retain: None,
            state: None,
        }
    }

    pub fn with_lazy<T>(execute: impl FnOnce() -> T) -> (T, Option<Arc<ReadScope>>) {
        ACTIVE_SCOPE.with(|state| {
            let _active = ReadScopeGuard {
                previous: state.borrow_mut().activate(),
                retain: None,
                state: Some(state),
            };
            let result = execute();
            let owner = match &*state.borrow() {
                ActiveScope::Active(owner) => owner.clone(),
                ActiveScope::Inactive => None,
            };
            (result, owner)
        })
    }

    pub fn fresh() -> Self {
        let previous = ACTIVE_SCOPE.with(|scope| scope.replace(ActiveScope::Active(None)));
        Self {
            previous: Some(previous),
            retain: None,
            state: None,
        }
    }
}

impl<'a> ReadScopeGuard<'a> {
    pub fn for_result(owner: &'a RefCell<Option<Arc<ReadScope>>>) -> Self {
        let mut active = match owner.borrow().as_ref() {
            Some(scope) => scope.enter(),
            None => ReadScopeGuard::lazy(),
        };
        Self {
            previous: active.previous.take(),
            retain: Some(owner),
            state: None,
        }
    }
}

impl Drop for ReadScopeGuard<'_> {
    fn drop(&mut self) {
        if let Some(owner) = self.retain {
            if owner.borrow().is_none() {
                *owner.borrow_mut() = current_scope();
            }
        }
        if let Some(previous) = self.previous.take() {
            match self.state {
                Some(state) => {
                    state.replace(previous);
                }
                None => {
                    ACTIVE_SCOPE.with(|scope| scope.replace(previous));
                }
            }
        }
    }
}

#[cfg(feature = "parallel")]
pub(crate) struct ParallelReadScope {
    enabled: bool,
    shared: Option<Arc<ReadScope>>,
    owner: Mutex<Option<Arc<ReadScope>>>,
}

#[cfg(feature = "parallel")]
impl ParallelReadScope {
    pub fn new() -> Self {
        Self {
            enabled: ACTIVE_SCOPE.with(|scope| matches!(*scope.borrow(), ActiveScope::Active(_))),
            shared: current_scope(),
            owner: Mutex::new(None),
        }
    }

    pub fn run<T>(&self, read: impl FnOnce() -> T) -> T {
        if !self.enabled {
            return read();
        }
        if let Some(scope) = &self.shared {
            let _active = scope.enter();
            return read();
        }
        let _active = ReadScopeGuard::fresh();
        let result = read();
        if let Some(scope) = current_scope() {
            let mut owner = self.owner.lock();
            match owner.as_ref() {
                Some(owner) => owner.add(scope.bytes.get_wide()),
                None => *owner = Some(scope),
            }
        }
        result
    }
}

#[cfg(feature = "parallel")]
impl Drop for ParallelReadScope {
    fn drop(&mut self) {
        let Some(scope) = self.owner.get_mut().take() else {
            return;
        };
        ACTIVE_SCOPE.with(|state| {
            if let ActiveScope::Active(owner) = &mut *state.borrow_mut() {
                match owner.as_ref() {
                    Some(owner) => owner.add(scope.bytes.get_wide()),
                    None => *owner = Some(scope),
                }
            }
        });
    }
}

pub(crate) struct ExportBatch {
    scope: Option<Arc<ReadScope>>,
    enabled: bool,
    bytes: u128,
}

impl ExportBatch {
    pub fn new() -> Self {
        Self {
            scope: None,
            enabled: ACTIVE_SCOPE.with(|scope| matches!(*scope.borrow(), ActiveScope::Active(_))),
            bytes: 0,
        }
    }

    pub fn record(&mut self, row: &Row) {
        if self.enabled {
            self.record_bytes(row.heap_bytes());
        }
    }

    pub fn record_rows(
        &mut self,
        rows: &[(i64, Row)],
        inline_row_bytes: Option<std::num::NonZeroUsize>,
    ) {
        if self.enabled {
            let bytes = match inline_row_bytes {
                Some(bytes) => rows.len() as u128 * bytes.get() as u128,
                None => rows.iter().map(|(_, row)| row.heap_bytes()).sum(),
            };
            self.record_bytes(bytes);
        }
    }

    pub fn capture(&mut self, row: &Row) -> Row {
        self.record(row);
        row.clone()
    }

    pub fn record_value(&mut self, value: &Value) {
        if self.enabled {
            self.record_bytes(value.heap_bytes() as u128);
        }
    }

    #[inline]
    fn record_bytes(&mut self, bytes: u128) {
        if bytes == 0 {
            return;
        }
        if self.scope.is_none() {
            self.bind_scope();
        }
        self.bytes += bytes;
    }

    #[cold]
    fn bind_scope(&mut self) {
        self.scope = ACTIVE_SCOPE.with(|scope| match &mut *scope.borrow_mut() {
            ActiveScope::Active(owner) => Some(Arc::clone(owner.get_or_insert_with(new_scope))),
            ActiveScope::Inactive => None,
        });
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
        if self.payload.bytes == 0 {
            return;
        }
        ACTIVE_SCOPE.with(|scope| {
            if let ActiveScope::Active(owner) = &mut *scope.borrow_mut() {
                owner.get_or_insert_with(new_scope).import(self);
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
    fn deferred_exports_stay_lazy_and_follow_their_owner_across_threads() {
        let mut first = DeferredExports::default();
        first.bind(None);
        first.add(0);
        assert!(first.scope().is_none());
        first.add(512);
        let first_identity = Arc::downgrade(first.scope().unwrap());
        let mut second = DeferredExports::default();
        second.bind(None);
        second.add(1024);
        assert!(!Arc::ptr_eq(
            first.scope().unwrap(),
            second.scope().unwrap()
        ));
        let first = std::thread::spawn(move || {
            let unrelated = Arc::new(ReadScope::default());
            let _active = unrelated.enter();
            first.add(256);
            assert_eq!(unrelated.exported_bytes(), 0);
            assert_eq!(first.scope().unwrap().exported_bytes(), 768);
            first
        })
        .join()
        .unwrap();
        second.add(128);
        assert_eq!(second.scope().unwrap().exported_bytes(), 1152);
        assert!(current_scope().is_none());
        drop(first);
        assert!(first_identity.upgrade().is_none());
        assert_eq!(second.scope().unwrap().exported_bytes(), 1152);
    }

    #[test]
    fn recycled_scope_clears_exports_and_imports_before_reuse() {
        RECYCLED_SCOPE.with(|scope| scope.take());
        let sources = [
            SharedPayloadCharge::new(512),
            SharedPayloadCharge::new(128),
            SharedPayloadCharge::new(256),
        ];
        let scope = new_scope();
        let pointer = Arc::as_ptr(&scope);
        for source in &sources {
            scope.import(source);
            assert_eq!(Arc::weak_count(source), 1);
        }
        assert!(scope.imports.lock().spilled());
        scope.add(64);
        assert_eq!(scope.exported_bytes(), 960);
        drop(ResultReadScope(Some(scope)));
        for source in &sources {
            assert_eq!(Arc::weak_count(source), 0);
        }
        RECYCLED_SCOPE.with(|spare| {
            let spare = spare.borrow();
            let scope = spare.as_ref().unwrap();
            assert_eq!(scope.exported_bytes(), 0);
            assert!(scope.imports.lock().is_empty());
            assert!(!scope.imports.lock().spilled());
            assert!(scope.last_import.load(Ordering::Relaxed).is_null());
        });
        let reused = new_scope();
        assert_eq!(Arc::as_ptr(&reused), pointer);
        reused.import(&sources[0]);
        assert_eq!(reused.exported_bytes(), 512);
        assert_eq!(Arc::weak_count(&sources[0]), 1);
    }

    #[test]
    fn shared_scope_is_not_reset_or_recycled() {
        RECYCLED_SCOPE.with(|scope| scope.take());
        let scope = new_scope();
        scope.add(512);
        drop(ResultReadScope(Some(Arc::clone(&scope))));
        assert_eq!(scope.exported_bytes(), 512);
        RECYCLED_SCOPE.with(|spare| assert!(spare.borrow().is_none()));
        let weak = Arc::downgrade(&scope);
        drop(ResultReadScope(Some(scope)));
        assert!(weak.upgrade().is_none());
        RECYCLED_SCOPE.with(|spare| assert!(spare.borrow().is_none()));
    }

    #[test]
    fn lazy_scope_materializes_only_for_nonzero_exports() {
        let _active = ReadScopeGuard::lazy();
        {
            let _nested = ReadScopeGuard::lazy();
            charge_bytes_export(0);
            charge_value_export(&Value::Integer(7));
            SharedPayloadCharge::new(0).import();
            let mut batch = ExportBatch::new();
            batch.record_value(&Value::Integer(7));
        }
        assert!(current_scope().is_none());
        let source = SharedPayloadCharge::new(512);
        source.import();
        let scope = current_scope().unwrap();
        assert_eq!(scope.exported_bytes(), 512);
        source.import();
        assert_eq!(scope.exported_bytes(), 512);
        let _nested = ReadScopeGuard::lazy();
        charge_bytes_export(64);
        assert!(Arc::ptr_eq(&scope, &current_scope().unwrap()));
        assert_eq!(scope.exported_bytes(), 576);
    }

    #[test]
    fn lazy_result_owner_retains_first_export_on_unwind() {
        let owner = RefCell::new(None);
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _active = ReadScopeGuard::for_result(&owner);
            charge_bytes_export(512);
            panic!("leave result iteration");
        }));
        assert!(outcome.is_err());
        assert!(current_scope().is_none());
        assert_eq!(owner.borrow().as_ref().unwrap().exported_bytes(), 512);
        let _active = ReadScopeGuard::for_result(&owner);
        charge_bytes_export(64);
        assert_eq!(owner.borrow().as_ref().unwrap().exported_bytes(), 576);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn lazy_workers_transfer_exports_to_unmaterialized_parent() {
        let _active = ReadScopeGuard::lazy();
        {
            let workers = ParallelReadScope::new();
            workers.run(|| charge_value_export(&Value::Integer(7)));
            assert!(workers.owner.lock().is_none());
        }
        assert!(current_scope().is_none());
        {
            let workers = ParallelReadScope::new();
            std::thread::scope(|threads| {
                for _ in 0..4 {
                    let workers = &workers;
                    threads.spawn(move || workers.run(|| charge_bytes_export(512)));
                }
            });
            assert!(current_scope().is_none());
            assert_eq!(
                workers.owner.lock().as_ref().unwrap().exported_bytes(),
                2048
            );
        }
        assert_eq!(current_scope().unwrap().exported_bytes(), 2048);
        let parent = current_scope().unwrap();
        {
            let workers = ParallelReadScope::new();
            workers.run(|| {
                assert!(Arc::ptr_eq(&parent, &current_scope().unwrap()));
                charge_bytes_export(64);
            });
            assert!(workers.owner.lock().is_none());
        }
        assert_eq!(parent.exported_bytes(), 2112);
    }

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
