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

//! Allocation-lifetime accounting for retained hot storage.
//!
//! Accounts observe requested allocation bytes, not RSS. They do not enforce a
//! limit. Child accounts retain their engine root so dropped-table allocations
//! remain in the root total until their final owner releases them.

use std::mem::ManuallyDrop;
use std::ptr;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemorySnapshot {
    pub retained_bytes: usize,
    pub conservative_bytes: usize,
    pub pending_bytes: usize,
    pub accounted_bytes: usize,
    pub peak_accounted_bytes: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Retained,
    Conservative,
    Pending,
}

#[derive(Default)]
struct Counters {
    conservative: AtomicUsize,
    pending: AtomicUsize,
    total: AtomicUsize,
    peak: AtomicUsize,
}

impl Counters {
    fn separate_counter(&self, kind: Kind) -> Option<&AtomicUsize> {
        match kind {
            Kind::Retained => None,
            Kind::Conservative => Some(&self.conservative),
            Kind::Pending => Some(&self.pending),
        }
    }

    fn add(&self, bytes: usize, kind: Kind) {
        if bytes == 0 {
            return;
        }
        // Overflow indicates an accounting bug: real retained allocations cannot
        // consume more than the process address space. Check rather than wrap.
        let previous = self
            .total
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
                n.checked_add(bytes)
            })
            .expect("memory accounting overflow");
        if let Some(counter) = self.separate_counter(kind) {
            counter.fetch_add(bytes, Ordering::Relaxed);
        }
        let total = previous + bytes;
        if total > self.peak.load(Ordering::Relaxed) {
            self.peak.fetch_max(total, Ordering::Relaxed);
        }
    }

    fn remove(&self, bytes: usize, kind: Kind) {
        if bytes == 0 {
            return;
        }
        if let Some(counter) = self.separate_counter(kind) {
            let previous = counter.fetch_sub(bytes, Ordering::Relaxed);
            assert!(previous >= bytes, "memory accounting underflow");
        }
        let previous = self.total.fetch_sub(bytes, Ordering::Relaxed);
        assert!(previous >= bytes, "memory accounting total underflow");
    }

    fn retain_pending(&self, bytes: usize) {
        if bytes == 0 {
            return;
        }
        let previous = self.pending.fetch_sub(bytes, Ordering::Relaxed);
        assert!(previous >= bytes, "memory accounting pending underflow");
        // Total and peak do not change: pending ownership was already charged.
    }

    fn snapshot(&self) -> MemorySnapshot {
        let accounted_bytes = self.total.load(Ordering::Relaxed);
        let conservative_bytes = self.conservative.load(Ordering::Relaxed);
        let pending_bytes = self.pending.load(Ordering::Relaxed);
        // Retained ownership is the residual of the single total. Avoid a
        // redundant atomic update for every hot allocation and final release.
        // These loads are observational, as before: a concurrent conversion
        // may straddle them, so use saturating arithmetic for the derived field.
        let retained_bytes =
            accounted_bytes.saturating_sub(conservative_bytes.saturating_add(pending_bytes));
        MemorySnapshot {
            retained_bytes,
            conservative_bytes,
            pending_bytes,
            accounted_bytes,
            peak_accounted_bytes: self.peak.load(Ordering::Relaxed),
        }
    }
}

// Conservative allowance for the opaque std::Arc control block and padding.
// The account object itself is retained memory, even when no payload is live.
const ACCOUNT_BYTES: usize = std::mem::size_of::<AccountInner>() + 4 * std::mem::size_of::<usize>();

struct AccountInner {
    counters: Counters,
    // Root accounts have no parent. Children point directly to the root, never
    // to a mutable table/engine object or to a pool that owns memory charges.
    root: Option<Arc<AccountInner>>,
}

impl Drop for AccountInner {
    fn drop(&mut self) {
        if let Some(root) = &self.root {
            root.counters.remove(ACCOUNT_BYTES, Kind::Conservative);
        }
    }
}

/// Cheap shared handle to an engine or origin account. Clone never allocates.
#[derive(Clone)]
pub struct MemoryAccount(Arc<AccountInner>);

impl Default for MemoryAccount {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryAccount {
    pub fn new() -> Self {
        let account = Self(Arc::new(AccountInner {
            counters: Counters::default(),
            root: None,
        }));
        account.0.counters.add(ACCOUNT_BYTES, Kind::Conservative);
        account
    }

    pub fn child(&self) -> Self {
        let root = self.0.root.as_ref().unwrap_or(&self.0).clone();
        let account = Self(Arc::new(AccountInner {
            counters: Counters::default(),
            root: Some(root),
        }));
        account.add(ACCOUNT_BYTES, Kind::Conservative);
        account
    }

    pub fn same_engine(&self, other: &Self) -> bool {
        Arc::ptr_eq(
            self.0.root.as_ref().unwrap_or(&self.0),
            other.0.root.as_ref().unwrap_or(&other.0),
        )
    }

    pub fn same_origin(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    /// Counters are read independently; concurrent snapshots are observational,
    /// not an atomic transaction across all fields. The total is one atomic.
    pub fn snapshot(&self) -> MemorySnapshot {
        self.0.counters.snapshot()
    }

    fn add(&self, bytes: usize, kind: Kind) {
        if let Some(root) = &self.0.root {
            root.counters.add(bytes, kind);
        }
        self.0.counters.add(bytes, kind);
    }

    fn remove(&self, bytes: usize, kind: Kind) {
        self.0.counters.remove(bytes, kind);
        if let Some(root) = &self.0.root {
            root.counters.remove(bytes, kind);
        }
    }

    fn retain_pending(&self, bytes: usize) {
        if let Some(root) = &self.0.root {
            root.counters.retain_pending(bytes);
        }
        self.0.counters.retain_pending(bytes);
    }
}

/// One unique allocation/capacity owner. Move this token with its buffer.
/// Aliased allocations store equivalent ownership in their shared header.
pub struct MemoryCharge {
    account: MemoryAccount,
    bytes: usize,
    kind: Kind,
}

impl MemoryCharge {
    pub fn new(account: &MemoryAccount, bytes: usize) -> Self {
        Self::with_kind(account, bytes, Kind::Retained)
    }

    pub fn conservative(account: &MemoryAccount, bytes: usize) -> Self {
        Self::with_kind(account, bytes, Kind::Conservative)
    }

    fn with_kind(account: &MemoryAccount, bytes: usize, kind: Kind) -> Self {
        account.add(bytes, kind);
        Self {
            account: account.clone(),
            bytes,
            kind,
        }
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }
    pub fn account(&self) -> &MemoryAccount {
        &self.account
    }

    /// Transfer a replacement allocation into one part of this combined owner.
    /// Both allocations are already charged. The returned token owns the old
    /// part and must stay alive until that backing is freed. No counter changes
    /// occur here, so ownership transfer cannot create a false double-charge
    /// peak or a gap between freeing the old buffer and retaining its replacement.
    pub(crate) fn replace_part(&mut self, old_bytes: usize, mut replacement: Self) -> Self {
        assert!(self.account.same_origin(&replacement.account));
        assert!(self.kind == replacement.kind);
        let bytes = self
            .bytes
            .checked_sub(old_bytes)
            .and_then(|rest| rest.checked_add(replacement.bytes))
            .expect("invalid memory charge replacement");
        self.bytes = bytes;
        replacement.bytes = old_bytes;
        replacement
    }

    /// Update observed retained capacity. This does not claim that realloc held
    /// both buffers simultaneously; a future preallocation reservation records
    /// that separate bound before an allocation is attempted.
    pub fn resize(&mut self, bytes: usize) {
        if bytes == self.bytes {
            return;
        }
        if bytes > self.bytes {
            self.account.add(bytes - self.bytes, self.kind);
        } else {
            self.account.remove(self.bytes - bytes, self.kind);
        }
        self.bytes = bytes;
    }

    fn retain_for_header(self) {
        let this = ManuallyDrop::new(self);
        this.account.retain_pending(this.bytes);
        // SAFETY: the header already owns a distinct raw Arc reference. Release
        // the pending token's account handle without releasing its byte charge.
        unsafe {
            drop(ptr::read(&this.account));
        }
    }
}

impl Drop for MemoryCharge {
    fn drop(&mut self) {
        self.account.remove(self.bytes, self.kind);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryAdoption {
    Adopted,
    AlreadyOwned,
    ForeignEngine,
}

/// Exactly one optional pointer word in a shared allocation header.
/// No Drop impl: the allocation owner must take the charge with its known size.
#[repr(transparent)]
pub(crate) struct AccountSlot(AtomicPtr<AccountInner>);

impl AccountSlot {
    pub(crate) const fn new() -> Self {
        Self(AtomicPtr::new(ptr::null_mut()))
    }

    /// Install ownership in a freshly constructed, unexposed allocation.
    pub(crate) fn install_new(&mut self, account: &MemoryAccount, bytes: usize) {
        debug_assert!(self.0.get_mut().is_null());
        account.add(bytes, Kind::Retained);
        *self.0.get_mut() = Arc::into_raw(account.0.clone()) as *mut AccountInner;
    }

    pub(crate) fn get(&self) -> Option<MemoryAccount> {
        let account = self
            .0
            .load(Ordering::Acquire)
            .map_addr(|address| address & !1);
        if account.is_null() {
            return None;
        }
        // SAFETY: callers hold an allocation owner, so its account reference
        // cannot be released during this borrow. Once installed it never changes
        // until the allocation's exclusive final destruction.
        unsafe {
            Arc::increment_strong_count(account);
            Some(MemoryAccount(Arc::from_raw(account)))
        }
    }

    pub(crate) fn belongs_to(&self, account: &MemoryAccount) -> bool {
        self.ownership(account) == Some(MemoryAdoption::AlreadyOwned)
    }

    fn ownership(&self, account: &MemoryAccount) -> Option<MemoryAdoption> {
        let pointer = self
            .0
            .load(Ordering::Acquire)
            .map_addr(|address| address & !1);
        if pointer.is_null() {
            return None;
        }
        // SAFETY: a caller holds this allocation alive. The installed account
        // stays alive and its root pointer is immutable throughout this borrow.
        let current = unsafe { &*pointer };
        let current_root = current
            .root
            .as_ref()
            .map_or(pointer as *const _, Arc::as_ptr);
        let requested_root = Arc::as_ptr(account.0.root.as_ref().unwrap_or(&account.0));
        Some(if ptr::eq(current_root, requested_root) {
            MemoryAdoption::AlreadyOwned
        } else {
            MemoryAdoption::ForeignEngine
        })
    }

    pub(crate) fn adopt(&self, account: &MemoryAccount, bytes: usize) -> MemoryAdoption {
        if let Some(ownership) = self.ownership(account) {
            return ownership;
        }
        // Pending ownership covers the interval between account publication and
        // retained finalization, including a concurrent observer of that pointer.
        let pending = MemoryCharge::with_kind(account, bytes, Kind::Pending);
        let raw = Arc::into_raw(account.0.clone()) as *mut AccountInner;
        match self
            .0
            .compare_exchange(ptr::null_mut(), raw, Ordering::AcqRel, Ordering::Acquire)
        {
            Ok(_) => {
                pending.retain_for_header();
                MemoryAdoption::Adopted
            }
            Err(_) => {
                // SAFETY: the failed CAS did not transfer the new raw reference.
                unsafe {
                    drop(Arc::from_raw(raw));
                }
                drop(pending);
                // A winning adoption cannot disappear while this caller holds
                // the allocation alive, so this read must observe an owner.
                self.ownership(account).expect("installed memory account")
            }
        }
    }

    pub(crate) fn clear_fully_accounted(&mut self) {
        let pointer = self.0.get_mut();
        *pointer = pointer.map_addr(|address| address & !1);
    }

    pub(crate) fn is_fully_accounted(&self) -> bool {
        self.0.load(Ordering::Acquire).addr() & 1 != 0
    }

    /// Only a container's audited ownership ingress may certify its children.
    /// Generic shallow adoption deliberately leaves this marker clear.
    pub(crate) fn mark_fully_accounted(&self) {
        let mut pointer = self.0.load(Ordering::Acquire);
        loop {
            assert!(
                !pointer.is_null(),
                "cannot certify an unaccounted allocation"
            );
            let marked = pointer.map_addr(|address| address | 1);
            match self.0.compare_exchange_weak(
                pointer,
                marked,
                Ordering::Release,
                Ordering::Acquire,
            ) {
                Ok(_) => return,
                Err(actual) => pointer = actual,
            }
        }
    }

    /// Transfer the header's charge to a guard that outlives deallocation.
    /// Requires exclusive final ownership; no other account accesses may race.
    pub(crate) fn take(&mut self, bytes: usize) -> Option<MemoryCharge> {
        let account = self.0.get_mut().map_addr(|address| address & !1);
        *self.0.get_mut() = ptr::null_mut();
        if account.is_null() {
            return None;
        }
        // SAFETY: successful adoption transferred one Arc reference into this
        // slot; exclusive take transfers it once, without a counter change.
        Some(MemoryCharge {
            account: MemoryAccount(unsafe { Arc::from_raw(account) }),
            bytes,
            kind: Kind::Retained,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retained_residual_tracks_mixed_classes_and_pending_conversion() {
        let counters = Counters::default();
        counters.add(80, Kind::Retained);
        counters.add(120, Kind::Conservative);
        counters.add(64, Kind::Pending);
        assert_eq!(
            counters.snapshot(),
            MemorySnapshot {
                retained_bytes: 80,
                conservative_bytes: 120,
                pending_bytes: 64,
                accounted_bytes: 264,
                peak_accounted_bytes: 264,
            }
        );
        counters.retain_pending(64);
        assert_eq!(counters.snapshot().retained_bytes, 144);
        assert_eq!(counters.snapshot().pending_bytes, 0);
        assert_eq!(counters.snapshot().accounted_bytes, 264);
        counters.remove(144, Kind::Retained);
        counters.remove(120, Kind::Conservative);
        assert_eq!(
            counters.snapshot(),
            MemorySnapshot {
                peak_accounted_bytes: 264,
                ..MemorySnapshot::default()
            }
        );
    }

    #[test]
    fn concurrent_mixed_classes_leave_exact_quiescent_totals() {
        let counters = Counters::default();
        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| {
                    for _ in 0..1000 {
                        counters.add(80, Kind::Retained);
                        counters.add(120, Kind::Conservative);
                        counters.add(64, Kind::Pending);
                        counters.retain_pending(64);
                        let sample = counters.snapshot();
                        assert!(sample.retained_bytes <= sample.accounted_bytes);
                        counters.remove(144, Kind::Retained);
                        counters.remove(120, Kind::Conservative);
                    }
                });
            }
        });
        let snapshot = counters.snapshot();
        assert_eq!(snapshot.retained_bytes, 0);
        assert_eq!(snapshot.conservative_bytes, 0);
        assert_eq!(snapshot.pending_bytes, 0);
        assert_eq!(snapshot.accounted_bytes, 0);
        assert!(snapshot.peak_accounted_bytes >= 264);
        assert!(snapshot.peak_accounted_bytes <= 8 * 264);
    }

    #[test]
    fn replacement_transfers_ownership_without_counter_gap_or_duplicate_peak() {
        let account = MemoryAccount::new();
        let mut combined = MemoryCharge::new(&account, 100);
        let replacement = MemoryCharge::new(&account, 80);
        let before = account.snapshot();
        let retired = combined.replace_part(40, replacement);
        assert_eq!(account.snapshot(), before);
        assert_eq!(combined.bytes(), 140);
        assert_eq!(retired.bytes(), 40);
        drop(retired);
        assert_eq!(account.snapshot().retained_bytes, 140);
        drop(combined);
        assert_eq!(account.snapshot().retained_bytes, 0);
        assert_eq!(account.snapshot().peak_accounted_bytes, ACCOUNT_BYTES + 180);
    }

    #[test]
    fn child_charge_survives_origin_handle_and_moves_without_churn() {
        let root = MemoryAccount::new();
        let child = root.child();
        let mut charge = MemoryCharge::new(&child, 100);
        drop(child);
        assert_eq!(root.snapshot().retained_bytes, 100);
        charge.resize(200);
        let moved = charge;
        assert_eq!(root.snapshot().retained_bytes, 200);
        drop(moved);
        assert_eq!(root.snapshot().accounted_bytes, ACCOUNT_BYTES);
        assert_eq!(
            root.snapshot().peak_accounted_bytes,
            200 + 2 * ACCOUNT_BYTES
        );
    }

    #[test]
    fn exact_and_conservative_capacity_are_separate() {
        let root = MemoryAccount::new();
        let exact = MemoryCharge::new(&root, 80);
        let estimate = MemoryCharge::conservative(&root, 120);
        let snapshot = root.snapshot();
        assert_eq!(snapshot.retained_bytes, 80);
        assert_eq!(snapshot.conservative_bytes, 120 + ACCOUNT_BYTES);
        assert_eq!(snapshot.accounted_bytes, 200 + ACCOUNT_BYTES);
        drop((exact, estimate));
        assert_eq!(root.snapshot().accounted_bytes, ACCOUNT_BYTES);
    }

    #[test]
    fn concurrent_adoption_installs_one_final_charge() {
        let root = MemoryAccount::new();
        let slot = Arc::new(AccountSlot::new());
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let slot = slot.clone();
                let account = root.child();
                scope.spawn(move || {
                    assert_ne!(slot.adopt(&account, 512), MemoryAdoption::ForeignEngine);
                });
            }
        });
        assert_eq!(root.snapshot().retained_bytes, 512);
        assert_eq!(root.snapshot().pending_bytes, 0);
        let mut slot = Arc::try_unwrap(slot).ok().unwrap();
        drop(slot.take(512));
        assert_eq!(root.snapshot().accounted_bytes, ACCOUNT_BYTES);
    }
}
