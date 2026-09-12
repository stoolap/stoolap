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

//! Chunked row payloads with stable addresses and prepared write capacity.

use parking_lot::{Mutex, RwLock};
use smallvec::SmallVec;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::common::CompactArc;
use crate::core::{Error, Result, Value};
use crate::storage::mvcc::memory::{HotObjectCharge, TableMemory};

pub const ARENA_CHUNK_ROWS: usize = 1 << 18;
const CHUNK_BITS: u32 = 18;
const CHUNK_ID_LIMIT: u64 = (1 << (64 - CHUNK_BITS)) - 1;

/// Bytes requested by a payload, conservatively counting shared children again.
pub fn row_bytes(values: &[Value]) -> u128 {
    (2 * std::mem::size_of::<usize>() + std::mem::size_of_val(values)) as u128
        + values
            .iter()
            .map(|value| value.heap_bytes() as u128)
            .sum::<u128>()
}

struct ArenaBuffer<T> {
    values: Vec<T>,
    // Fields drop in order: the allocation is freed before its charge.
    _charge: ArenaBufferCharge,
}

#[derive(Default)]
struct ArenaBufferCharge {
    account: Option<Arc<TableMemory>>,
    bytes: usize,
}

impl Drop for ArenaBufferCharge {
    fn drop(&mut self) {
        if let Some(account) = &self.account {
            account.arena_capacity.remove(self.bytes as u128);
        }
    }
}

struct RetiredPayloadCharge<'a> {
    account: &'a TableMemory,
    bytes: u128,
}

impl<'a> RetiredPayloadCharge<'a> {
    fn new(account: &'a TableMemory, bytes: u128) -> Self {
        *account.retired_arena_payloads.lock() += bytes;
        Self { account, bytes }
    }
}

impl Drop for RetiredPayloadCharge<'_> {
    fn drop(&mut self) {
        *self.account.retired_arena_payloads.lock() -= self.bytes;
    }
}

impl<T> Default for ArenaBuffer<T> {
    fn default() -> Self {
        Self {
            values: Vec::new(),
            _charge: ArenaBufferCharge::default(),
        }
    }
}

impl<T> ArenaBuffer<T> {
    fn new(values: Vec<T>, account: &Arc<TableMemory>) -> Self {
        let bytes = values.capacity() * std::mem::size_of::<T>();
        let owner = (bytes != 0).then(|| {
            account.arena_capacity.add(bytes as u128);
            Arc::clone(account)
        });
        Self {
            values,
            _charge: ArenaBufferCharge {
                account: owner,
                bytes,
            },
        }
    }

    fn try_with_capacity(capacity: usize, account: &Arc<TableMemory>) -> Result<Self> {
        let mut values = Vec::new();
        values
            .try_reserve_exact(capacity)
            .map_err(allocation_error)?;
        Ok(Self::new(values, account))
    }
}

impl<T> std::ops::Deref for ArenaBuffer<T> {
    type Target = Vec<T>;
    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

impl<T> std::ops::DerefMut for ArenaBuffer<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.values
    }
}

impl<'a, T> IntoIterator for &'a ArenaBuffer<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.values.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut ArenaBuffer<T> {
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.values.iter_mut()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct ArenaSlot(NonZeroU64);

impl ArenaSlot {
    fn new(chunk: u64, offset: usize) -> Self {
        debug_assert!(chunk < CHUNK_ID_LIMIT && offset < ARENA_CHUNK_ROWS);
        Self(NonZeroU64::new((chunk << CHUNK_BITS) + offset as u64 + 1).unwrap_or(NonZeroU64::MIN))
    }

    fn chunk(self) -> u64 {
        (self.0.get() - 1) >> CHUNK_BITS
    }

    fn offset(self) -> usize {
        ((self.0.get() - 1) & (ARENA_CHUNK_ROWS as u64 - 1)) as usize
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ArenaRowMeta {
    pub row_id: i64,
    pub txn_id: i64,
    pub deleted_at_txn_id: i64,
}

#[derive(Default)]
struct Chunk {
    id: u64,
    data: ArenaBuffer<Option<CompactArc<[Value]>>>,
    meta: ArenaBuffer<ArenaRowMeta>,
    len: usize,
    occupied: usize,
    reserved: usize,
    base_row_id: Option<i64>,
}

impl Chunk {
    fn prepare(id: u64, capacity: usize, account: &Arc<TableMemory>) -> Result<Self> {
        #[cfg(any(test, feature = "test-failpoints"))]
        if crate::test_failpoints::arena_growth_fails() {
            return Err(Error::internal("injected hot arena growth failure"));
        }
        let mut data = ArenaBuffer::try_with_capacity(capacity, account)?;
        data.resize_with(capacity, || None);
        let mut meta = ArenaBuffer::try_with_capacity(capacity, account)?;
        meta.resize(capacity, ArenaRowMeta::default());
        Ok(Self {
            id,
            data,
            meta,
            len: 0,
            occupied: 0,
            reserved: 0,
            base_row_id: None,
        })
    }

    fn get(&self, offset: usize, row_id: i64) -> Option<(&ArenaRowMeta, &CompactArc<[Value]>)> {
        let meta = self.meta.get(offset)?;
        if meta.txn_id == 0 || meta.row_id != row_id {
            return None;
        }
        Some((meta, self.data.get(offset)?.as_ref()?))
    }

    fn probe(&self, row_id: i64) -> Option<(&ArenaRowMeta, &CompactArc<[Value]>)> {
        let offset = row_id.checked_sub(self.base_row_id?)?;
        self.get(usize::try_from(offset).ok()?, row_id)
    }
}

fn allocation_error(error: impl std::fmt::Display) -> Error {
    Error::internal(format!("cannot reserve hot arena capacity: {error}"))
}

struct ArenaInner {
    active: Option<Chunk>,
    frozen: ArenaBuffer<Chunk>,
    free: ArenaBuffer<u32>,
    free_len: usize,
    next_chunk_id: u64,
    reserved: usize,
    ordered_bases: bool,
    payload_bytes: u128,
    row_bound: u128,
}

impl ArenaInner {
    fn refresh_probe_order(&mut self) {
        let mut end = None;
        self.ordered_bases = self.frozen.iter().all(|chunk| {
            let Some(base) = chunk.base_row_id else {
                return false;
            };
            let Some(next) = base.checked_add(chunk.len as i64) else {
                return false;
            };
            let ordered = end.is_none_or(|previous| previous <= base);
            end = Some(next);
            ordered
        });
    }

    #[cfg(test)]
    fn chunk(&self, id: u64) -> Option<&Chunk> {
        if let Some(active) = &self.active {
            if active.id == id {
                return Some(active);
            }
        }
        self.frozen
            .binary_search_by_key(&id, |c| c.id)
            .ok()
            .map(|i| &self.frozen[i])
    }

    fn chunk_mut(&mut self, id: u64) -> Option<&mut Chunk> {
        if self.active.as_ref().is_some_and(|c| c.id == id) {
            return self.active.as_mut();
        }
        self.frozen
            .binary_search_by_key(&id, |c| c.id)
            .ok()
            .map(|i| &mut self.frozen[i])
    }

    fn retire_empty(&mut self, id: u64) -> Option<Chunk> {
        if let Some(active) = &self.active {
            if active.id == id {
                if active.occupied == 0 && active.reserved == 0 {
                    self.free_len = 0;
                    return self.active.take();
                }
                return None;
            }
        }
        let i = self.frozen.binary_search_by_key(&id, |c| c.id).ok()?;
        if self.frozen[i].occupied == 0 && self.frozen[i].reserved == 0 {
            let retired = self.frozen.remove(i);
            self.refresh_probe_order();
            Some(retired)
        } else {
            None
        }
    }
}

struct ArenaState {
    inner: RwLock<ArenaInner>,
    growth: Mutex<()>,
    reuse_holds: AtomicUsize,
    account: Arc<TableMemory>,
    initial_capacity: usize,
    _object: HotObjectCharge<Self>,
}

impl Drop for ArenaState {
    fn drop(&mut self) {
        #[cfg(any(test, feature = "test-failpoints"))]
        crate::test_failpoints::hot_owner_dropping();
        let inner = self.inner.get_mut();
        drop(inner.active.take());
        drop(std::mem::take(&mut inner.frozen));
        drop(std::mem::take(&mut inner.free));
        self.account.arena_payloads.store(0, Ordering::Release);
    }
}

pub struct RowArena {
    state: Arc<ArenaState>,
}

/// Exclusive slots remain valid across freeze until consumed or released.
pub struct ArenaReservation {
    state: Arc<ArenaState>,
    slots: SmallVec<[Option<ArenaSlot>; 1]>,
    cursor: usize,
    reuse_heads: bool,
}

/// Owns detached buffers until the caller has released its publication guards.
#[derive(Default)]
pub struct ArenaRetirement {
    chunks: ArenaBuffer<Chunk>,
    directory: ArenaBuffer<Chunk>,
    active: Option<Chunk>,
    payloads: ArenaBuffer<Option<CompactArc<[Value]>>>,
    payload_count: usize,
    free: ArenaBuffer<u32>,
    account: Option<Arc<TableMemory>>,
    payload_bytes: u128,
}

impl ArenaRetirement {
    fn retain_payloads(&mut self, account: &Arc<TableMemory>, bytes: u128) {
        if let Some(owner) = &self.account {
            debug_assert!(Arc::ptr_eq(owner, account));
        } else {
            self.account = Some(Arc::clone(account));
        }
        *account.retired_arena_payloads.lock() += bytes;
        self.payload_bytes += bytes;
    }
}

impl Drop for ArenaRetirement {
    fn drop(&mut self) {
        drop(std::mem::take(&mut self.chunks));
        drop(std::mem::take(&mut self.directory));
        self.active.take();
        drop(std::mem::take(&mut self.payloads));
        drop(std::mem::take(&mut self.free));
        if let Some(account) = &self.account {
            *account.retired_arena_payloads.lock() -= self.payload_bytes;
        }
    }
}

impl RowArena {
    pub fn with_capacity(expected_rows: usize) -> Self {
        Self::with_account(expected_rows, Arc::new(TableMemory::default()))
    }

    pub(crate) fn with_account(expected_rows: usize, account: Arc<TableMemory>) -> Self {
        Self {
            state: Arc::new(ArenaState {
                inner: RwLock::new(ArenaInner {
                    active: None,
                    frozen: ArenaBuffer::default(),
                    free: ArenaBuffer::default(),
                    free_len: 0,
                    next_chunk_id: 0,
                    reserved: 0,
                    ordered_bases: true,
                    payload_bytes: 0,
                    row_bound: 0,
                }),
                growth: Mutex::new(()),
                reuse_holds: AtomicUsize::new(0),
                account,
                initial_capacity: expected_rows.clamp(4, ARENA_CHUNK_ROWS),
                _object: HotObjectCharge::new(),
            }),
        }
    }

    pub fn reserve(&self, count: usize) -> Result<ArenaReservation> {
        let mut reservation = ArenaReservation {
            state: Arc::clone(&self.state),
            slots: SmallVec::new(),
            cursor: 0,
            reuse_heads: false,
        };
        reservation
            .slots
            .try_reserve(count)
            .map_err(allocation_error)?;
        if reservation.slots.spilled() {
            self.state.account.arena_capacity.add(
                (reservation.slots.capacity() * std::mem::size_of::<Option<ArenaSlot>>()) as u128,
            );
        }
        reservation.slots.resize(count, None);
        let mut filled = 0;
        while filled < count {
            let mut inner = self.state.inner.write();
            while filled < count {
                let free_len = inner.free_len;
                let offset = if free_len > 0 {
                    inner.free_len -= 1;
                    inner.free[free_len - 1] as usize
                } else if let Some(active) = &mut inner.active {
                    if active.len == active.data.len() {
                        break;
                    }
                    let offset = active.len;
                    active.len += 1;
                    offset
                } else {
                    break;
                };
                if let Some(active) = &mut inner.active {
                    active.reserved += 1;
                    reservation.slots[filled] = Some(ArenaSlot::new(active.id, offset));
                    filled += 1;
                    inner.reserved += 1;
                }
            }
            drop(inner);
            if filled < count {
                self.grow(count - filled)?;
            }
        }
        Ok(reservation)
    }

    /// The caller holds versions.read until the existing-head reservation is installed.
    pub fn reserve_existing(&self) -> ArenaReservation {
        self.state.reuse_holds.fetch_add(1, Ordering::Relaxed);
        ArenaReservation {
            state: Arc::clone(&self.state),
            slots: SmallVec::new(),
            cursor: 0,
            reuse_heads: true,
        }
    }

    /// Removal checks this under versions.write, paired with reserve_existing.
    pub fn has_reserved_heads(&self) -> bool {
        self.state.reuse_holds.load(Ordering::Relaxed) != 0
    }

    fn grow(&self, remaining: usize) -> Result<()> {
        let _growth = self.state.growth.lock();
        let (id, capacity, directory_capacity) = {
            let inner = self.state.inner.read();
            if inner.free_len > 0 || inner.active.as_ref().is_some_and(|c| c.len < c.data.len()) {
                return Ok(());
            }
            if let Some(active) = &inner.active {
                if active.data.len() < ARENA_CHUNK_ROWS {
                    (
                        active.id,
                        (active
                            .data
                            .len()
                            .saturating_add(remaining)
                            .max(active.data.len() * 2))
                        .min(ARENA_CHUNK_ROWS),
                        inner.frozen.len() + 1,
                    )
                } else {
                    (
                        inner.next_chunk_id,
                        remaining.clamp(self.state.initial_capacity, ARENA_CHUNK_ROWS),
                        inner.frozen.len() + 2,
                    )
                }
            } else {
                (
                    inner.next_chunk_id,
                    remaining.clamp(self.state.initial_capacity, ARENA_CHUNK_ROWS),
                    inner.frozen.len() + 1,
                )
            }
        };
        if id >= CHUNK_ID_LIMIT {
            return Err(Error::internal("hot arena chunk address space exhausted"));
        }
        let mut prepared = Chunk::prepare(id, capacity, &self.state.account)?;
        let mut free = ArenaBuffer::try_with_capacity(capacity, &self.state.account)?;
        free.resize(capacity, 0);
        let mut directory =
            ArenaBuffer::try_with_capacity(directory_capacity, &self.state.account)?;
        directory.resize_with(directory_capacity, Chunk::default);
        directory.clear();
        let mut inner = self.state.inner.write();
        if inner
            .active
            .as_ref()
            .is_some_and(|c| c.id != id && c.data.len() < ARENA_CHUNK_ROWS)
            || (inner.active.is_none() && inner.next_chunk_id != id)
        {
            return Ok(());
        }
        if let Some(active) = &mut inner.active {
            if active.id == id {
                prepared.len = active.len;
                prepared.occupied = active.occupied;
                prepared.reserved = active.reserved;
                prepared.base_row_id = active.base_row_id;
                for (dst, src) in prepared.data.iter_mut().zip(&mut active.data) {
                    *dst = src.take();
                }
                prepared.meta[..active.len].copy_from_slice(&active.meta[..active.len]);
                std::mem::swap(active, &mut prepared);
            } else {
                directory.append(&mut inner.frozen);
                if let Some(old) = inner.active.take() {
                    directory.push(old);
                }
                std::mem::swap(&mut inner.frozen, &mut directory);
                inner.refresh_probe_order();
                inner.active = Some(prepared);
                inner.next_chunk_id += 1;
                // Free offsets belong to the old chunk, so rotation starts empty.
                inner.free_len = 0;
                std::mem::swap(&mut inner.free, &mut free);
                drop(inner);
                return Ok(());
            }
        } else {
            inner.active = Some(prepared);
            inner.next_chunk_id += 1;
            std::mem::swap(&mut inner.free, &mut free);
            drop(inner);
            return Ok(());
        }
        free[..inner.free_len].copy_from_slice(&inner.free[..inner.free_len]);
        std::mem::swap(&mut inner.free, &mut free);
        drop(inner);
        Ok(())
    }

    /// VersionStore retains the canonical nondeleted payload until this returns.
    pub(crate) fn install(
        &self,
        reservation: &mut ArenaReservation,
        existing: Option<ArenaSlot>,
        row_id: i64,
        txn_id: i64,
        data: CompactArc<[Value]>,
    ) -> (ArenaSlot, u128) {
        debug_assert!(Arc::ptr_eq(&self.state, &reservation.state));
        let reserved = if reservation.reuse_heads {
            None
        } else {
            let reserved = reservation.slots[reservation.cursor];
            if existing.is_none() {
                reservation.slots[reservation.cursor] = None;
            }
            reservation.cursor += 1;
            reserved
        };
        let slot = existing
            .or(reserved)
            .unwrap_or_else(|| unreachable!("write capacity was reserved before publication"));
        let bytes = row_bytes(&data);
        let mut inner = self.state.inner.write();
        let chunk = inner
            .chunk_mut(slot.chunk())
            .unwrap_or_else(|| unreachable!("published head or reservation owns its chunk"));
        let offset = slot.offset();
        let was_deleted = chunk.meta[offset].deleted_at_txn_id != 0;
        let mut old_charge = None;
        let old = chunk.data[offset].replace(data);
        if old.is_none() {
            chunk.occupied += 1;
        }
        chunk.meta[offset] = ArenaRowMeta {
            row_id,
            txn_id,
            deleted_at_txn_id: 0,
        };
        let new_base = chunk.base_row_id.is_none();
        if new_base {
            chunk.base_row_id = row_id.checked_sub(offset as i64);
        }
        if Some(slot) == reserved {
            chunk.reserved -= 1;
            inner.reserved -= 1;
        }
        if new_base {
            inner.refresh_probe_order();
        }
        inner.payload_bytes += bytes;
        inner.row_bound = inner.row_bound.max(bytes);
        if let Some(old) = &old {
            let old_bytes = row_bytes(old);
            if was_deleted {
                old_charge = Some(RetiredPayloadCharge::new(&self.state.account, old_bytes));
            }
            inner.payload_bytes -= old_bytes;
        }
        self.state.account.arena_payloads.store(
            inner.payload_bytes.min(usize::MAX as u128) as usize,
            Ordering::Release,
        );
        drop(inner);
        #[cfg(any(test, feature = "test-failpoints"))]
        if old_charge.is_some() {
            crate::test_failpoints::hot_owner_dropping();
        }
        drop(old);
        drop(old_charge);
        (slot, bytes)
    }

    pub fn mark_deleted(&self, slot: ArenaSlot, txn_id: i64) {
        let mut inner = self.state.inner.write();
        if let Some(chunk) = inner.chunk_mut(slot.chunk()) {
            chunk.meta[slot.offset()].deleted_at_txn_id = txn_id;
        }
    }

    pub fn prepare_clear(&self, rows: usize) -> ArenaRetirement {
        let chunks = rows.min(self.state.inner.read().frozen.len() + 1);
        let mut retirement = ArenaRetirement {
            payloads: ArenaBuffer::new((0..rows).map(|_| None).collect(), &self.state.account),
            chunks: ArenaBuffer::new(Vec::with_capacity(chunks), &self.state.account),
            directory: ArenaBuffer::default(),
            active: None,
            payload_count: 0,
            free: ArenaBuffer::default(),
            account: None,
            payload_bytes: 0,
        };
        retirement.chunks.resize_with(chunks, Chunk::default);
        retirement.chunks.clear();
        retirement
    }

    pub fn clear_batch(&self, slots: &[ArenaSlot], retired: &mut ArenaRetirement) -> usize {
        let mut inner = self.state.inner.write();
        let mut cleared = 0;
        let mut bytes = 0;
        for &slot in slots {
            if let Some(chunk) = inner.chunk_mut(slot.chunk()) {
                let offset = slot.offset();
                if let Some(data) = &chunk.data[offset] {
                    let payload_bytes = row_bytes(data);
                    let destination = &mut retired.payloads[retired.payload_count];
                    let data = chunk.data[offset].take();
                    chunk.meta[offset] = ArenaRowMeta::default();
                    *destination = data;
                    bytes += payload_bytes;
                    retired.payload_count += 1;
                    chunk.occupied -= 1;
                    cleared += 1;
                    if inner.active.as_ref().is_some_and(|c| c.id == slot.chunk()) {
                        let n = inner.free_len;
                        inner.free[n] = slot.offset() as u32;
                        inner.free_len += 1;
                    }
                    if let Some(chunk) = inner.retire_empty(slot.chunk()) {
                        debug_assert!(retired.chunks.len() < retired.chunks.capacity());
                        retired.chunks.push(chunk);
                        if inner.active.is_none() && !inner.free.is_empty() {
                            debug_assert!(retired.free.is_empty());
                            retired.free = std::mem::take(&mut inner.free);
                        }
                    }
                }
            }
        }
        retired.retain_payloads(&self.state.account, bytes);
        inner.payload_bytes -= bytes;
        if inner.payload_bytes == 0 {
            inner.row_bound = 0;
        }
        self.state.account.arena_payloads.store(
            inner.payload_bytes.min(usize::MAX as u128) as usize,
            Ordering::Release,
        );
        drop(inner);
        cleared
    }

    pub fn finish_clear(&self, retired: &mut ArenaRetirement) {
        let mut inner = self.state.inner.write();
        if inner.frozen.is_empty() && inner.frozen.capacity() != 0 {
            debug_assert_eq!(retired.directory.capacity(), 0);
            retired.directory = std::mem::take(&mut inner.frozen);
        }
    }

    pub fn clear_all(&self) -> Result<ArenaRetirement> {
        let mut inner = self.state.inner.write();
        if inner.reserved != 0 || self.has_reserved_heads() {
            return Err(Error::TableHasActiveTransactions);
        }
        let active = inner.active.take();
        let frozen = std::mem::take(&mut inner.frozen);
        let free = std::mem::take(&mut inner.free);
        inner.free_len = 0;
        let payload_bytes = std::mem::take(&mut inner.payload_bytes);
        inner.row_bound = 0;
        let mut retired = ArenaRetirement {
            active,
            chunks: frozen,
            free,
            directory: ArenaBuffer::default(),
            payloads: ArenaBuffer::default(),
            payload_count: 0,
            account: None,
            payload_bytes: 0,
        };
        retired.retain_payloads(&self.state.account, payload_bytes);
        self.state
            .account
            .arena_payloads
            .store(0, Ordering::Release);
        drop(inner);
        Ok(retired)
    }

    pub fn slot_count(&self) -> usize {
        let inner = self.state.inner.read();
        inner
            .active
            .iter()
            .chain(&inner.frozen)
            .map(|c| c.len)
            .sum()
    }

    pub fn bytes(&self) -> usize {
        self.state.account.arena_payloads.load(Ordering::Acquire)
    }

    pub fn capacity_bytes(&self) -> usize {
        self.state.account.arena_capacity.get()
    }

    pub fn read_guard(&self) -> ArenaReadGuard<'_> {
        ArenaReadGuard {
            inner: self.state.inner.read(),
        }
    }
}

impl Drop for ArenaReservation {
    fn drop(&mut self) {
        if self.reuse_heads {
            self.state.reuse_holds.fetch_sub(1, Ordering::Relaxed);
        }
        let mut slots = self.slots.iter().flatten().copied().peekable();
        while let Some(first) = slots.peek().copied() {
            let mut inner = self.state.inner.write();
            for _ in 0..1024 {
                let Some(slot) = slots.next_if(|slot| slot.chunk() == first.chunk()) else {
                    break;
                };
                if let Some(chunk) = inner.chunk_mut(slot.chunk()) {
                    debug_assert!(chunk.reserved > 0);
                    chunk.reserved -= 1;
                    inner.reserved -= 1;
                    if inner.active.as_ref().is_some_and(|c| c.id == slot.chunk()) {
                        let n = inner.free_len;
                        inner.free[n] = slot.offset() as u32;
                        inner.free_len += 1;
                    }
                }
            }
            let retired = inner.retire_empty(first.chunk());
            let free = if inner.active.is_none() {
                std::mem::take(&mut inner.free)
            } else {
                ArenaBuffer::default()
            };
            let directory = if inner.frozen.is_empty() {
                std::mem::take(&mut inner.frozen)
            } else {
                ArenaBuffer::default()
            };
            drop(inner);
            drop((retired, free, directory));
        }
        if self.slots.spilled() {
            let bytes = self.slots.capacity() * std::mem::size_of::<Option<ArenaSlot>>();
            drop(std::mem::take(&mut self.slots));
            self.state.account.arena_capacity.remove(bytes as u128);
        }
    }
}

pub struct ArenaReadGuard<'a> {
    inner: parking_lot::RwLockReadGuard<'a, ArenaInner>,
}

pub(crate) struct ArenaChunkSlices<'a> {
    meta: &'a [ArenaRowMeta],
    data: &'a [Option<CompactArc<[Value]>>],
}

pub(crate) struct ArenaLiveRow<'a> {
    meta: &'a ArenaRowMeta,
    data: &'a Option<CompactArc<[Value]>>,
}

impl<'a> ArenaLiveRow<'a> {
    #[inline]
    pub fn metadata(&self) -> &'a ArenaRowMeta {
        self.meta
    }

    #[inline]
    pub fn payload(&self) -> &'a CompactArc<[Value]> {
        match self.data.as_ref() {
            Some(data) => data,
            None => panic!("live arena slot has no payload"),
        }
    }
}

impl ArenaChunkSlices<'_> {
    #[inline]
    pub fn slot_count(&self) -> usize {
        self.meta.len().min(self.data.len())
    }

    #[inline]
    pub fn metadata(&self) -> &[ArenaRowMeta] {
        self.meta
    }

    #[inline]
    pub fn live_row(&self, offset: usize) -> Option<ArenaLiveRow<'_>> {
        let meta = self.meta.get(offset)?;
        if meta.txn_id == 0 {
            return None;
        }
        Some(ArenaLiveRow {
            meta,
            data: &self.data[offset],
        })
    }
}

impl ArenaReadGuard<'_> {
    pub(crate) fn row_bound(&self) -> u128 {
        self.inner.row_bound
    }

    #[cfg(test)]
    fn get(&self, slot: ArenaSlot, row_id: i64) -> Option<(&ArenaRowMeta, &CompactArc<[Value]>)> {
        self.inner.chunk(slot.chunk())?.get(slot.offset(), row_id)
    }

    #[inline]
    pub fn probe(&self, row_id: i64) -> Option<(&ArenaRowMeta, &CompactArc<[Value]>)> {
        if let Some(hit) = self.inner.active.as_ref().and_then(|c| c.probe(row_id)) {
            return Some(hit);
        }
        self.probe_frozen(row_id)
    }

    #[inline(never)]
    fn probe_frozen(&self, row_id: i64) -> Option<(&ArenaRowMeta, &CompactArc<[Value]>)> {
        if !self.inner.ordered_bases {
            return None;
        }
        let i = self
            .inner
            .frozen
            .partition_point(|chunk| chunk.base_row_id.is_some_and(|base| base <= row_id));
        self.inner.frozen.get(i.checked_sub(1)?)?.probe(row_id)
    }

    pub(crate) fn chunks(&self) -> impl Iterator<Item = ArenaChunkSlices<'_>> {
        self.inner
            .frozen
            .iter()
            .chain(&self.inner.active)
            .map(|chunk| ArenaChunkSlices {
                meta: &chunk.meta[..chunk.len],
                data: &chunk.data[..chunk.len],
            })
    }

    pub fn is_empty(&self) -> bool {
        self.inner
            .active
            .iter()
            .chain(&self.inner.frozen)
            .all(|c| c.occupied == 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn insert(arena: &RowArena, id: i64) -> ArenaSlot {
        let mut reservation = arena.reserve(1).unwrap();
        arena
            .install(
                &mut reservation,
                None,
                id,
                1,
                CompactArc::from(vec![Value::Integer(id)]),
            )
            .0
    }

    #[test]
    fn address_and_slot_layouts_stay_compact() {
        assert_eq!(std::mem::size_of::<Option<ArenaSlot>>(), 8);
        assert_eq!(std::mem::size_of::<Option<CompactArc<[Value]>>>(), 8);
        assert_eq!(std::mem::size_of::<ArenaRowMeta>(), 24);
        let slot = ArenaSlot::new(CHUNK_ID_LIMIT - 1, ARENA_CHUNK_ROWS - 1);
        assert_eq!(slot.chunk(), CHUNK_ID_LIMIT - 1);
        assert_eq!(slot.offset(), ARENA_CHUNK_ROWS - 1);
    }

    #[test]
    fn retirement_capacity_panic_keeps_published_payload_installed() {
        let arena = RowArena::with_capacity(0);
        let slot = insert(&arena, 7);
        let mut retired = ArenaRetirement::default();
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            arena.clear_batch(&[slot], &mut retired);
        }));
        assert!(panic.is_err());
        let guard = arena.read_guard();
        assert_eq!(
            guard.get(slot, 7).map(|(_, row)| &row[0]),
            Some(&Value::Integer(7))
        );
    }

    #[test]
    fn chunk_payload_checks_reserved_cleared_and_out_of_range_slots() {
        let arena = RowArena::with_capacity(0);
        let first = insert(&arena, 7);
        let _reservation = arena.reserve(1).unwrap();
        let last = insert(&arena, 9);
        {
            let guard = arena.read_guard();
            let chunk = guard.chunks().next().unwrap();
            assert_eq!(
                chunk.live_row(first.offset()).map(|row| &row.payload()[0]),
                Some(&Value::Integer(7))
            );
            assert!(chunk.live_row(1).is_none());
            assert_eq!(
                chunk.live_row(last.offset()).map(|row| &row.payload()[0]),
                Some(&Value::Integer(9))
            );
            assert!(chunk.live_row(usize::MAX).is_none());
        }
        let mut retired = arena.prepare_clear(1);
        arena.clear_batch(&[first], &mut retired);
        let guard = arena.read_guard();
        let chunk = guard.chunks().next().unwrap();
        assert!(chunk.live_row(first.offset()).is_none());
        assert_eq!(
            chunk.live_row(last.offset()).map(|row| &row.payload()[0]),
            Some(&Value::Integer(9))
        );
    }

    #[test]
    fn retained_account_outlives_arena_and_prepared_reservations() {
        let _guard = crate::test_failpoints::FailpointGuard::new();
        let registry = Arc::new(crate::storage::mvcc::memory::HotMemoryRegistry::default());
        let arena = RowArena::with_capacity(0);
        registry.register(&arena.state.account);
        insert(&arena, 1);
        let reservation = arena.reserve(2).unwrap();
        let capacity = arena.capacity_bytes();
        let reservation_bytes =
            reservation.slots.capacity() * std::mem::size_of::<Option<ArenaSlot>>();
        let weak = Arc::downgrade(&arena.state);
        drop(arena);
        assert_eq!(registry.total().arena_capacity, capacity);
        let observer = Arc::clone(&registry);
        let observed = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let observed_drop = Arc::clone(&observed);
        crate::test_failpoints::before_hot_owner_drop(move || {
            assert!(weak.upgrade().is_none());
            let usage = observer.total();
            assert_eq!(usage.arena_payloads, 32);
            assert_eq!(usage.arena_capacity, capacity - reservation_bytes);
            observed_drop.store(true, Ordering::Relaxed);
        });
        drop(reservation);
        assert!(observed.load(Ordering::Relaxed));
        let usage = registry.total();
        assert_eq!(usage.arena_payloads, 0);
        assert_eq!(usage.arena_capacity, 0);
    }

    #[test]
    fn detached_buffers_outlive_the_arena_without_releasing_their_charge() {
        let registry = crate::storage::mvcc::memory::HotMemoryRegistry::default();
        let arena = RowArena::with_capacity(0);
        registry.register(&arena.state.account);
        insert(&arena, 1);
        let retired = arena.clear_all().unwrap();
        let capacity = arena.capacity_bytes();
        drop(arena);
        let usage = registry.total();
        assert_eq!(usage.arena_payloads, 0);
        assert_eq!(usage.retired_arena_payloads, 32);
        assert_eq!(usage.arena_capacity, capacity);
        drop(retired);
        let usage = registry.total();
        assert_eq!(usage.retired_arena_payloads, 0);
        assert_eq!(usage.arena_capacity, 0);
    }

    #[test]
    fn empty_chunks_release_capacity_without_reusing_identity() {
        let arena = RowArena::with_capacity(0);
        let old = insert(&arena, 100);
        let before = arena.capacity_bytes();
        let mut retirement = arena.prepare_clear(1);
        assert_eq!(arena.clear_batch(&[old], &mut retirement), 1);
        assert!(arena.capacity_bytes() >= before);
        drop(retirement);
        assert!(arena.capacity_bytes() < before);
        let next = insert(&arena, -10);
        assert!(next.chunk() > old.chunk());
        assert!(arena.read_guard().get(old, 100).is_none());
        assert!(arena.read_guard().get(next, -10).is_some());
    }

    #[test]
    fn reservation_survives_freeze_and_prevents_truncate() {
        let arena = RowArena::with_capacity(0);
        let mut first = arena.reserve(ARENA_CHUNK_ROWS).unwrap();
        let later = insert(&arena, 900_000);
        assert!(matches!(
            arena.clear_all(),
            Err(Error::TableHasActiveTransactions)
        ));
        let (slot, _) = arena.install(
            &mut first,
            None,
            0,
            1,
            CompactArc::from(vec![Value::Integer(7)]),
        );
        assert!(slot.chunk() < later.chunk());
        drop(first);
        assert!(arena.read_guard().get(slot, 0).is_some());
        arena.clear_all().unwrap();
        assert!(insert(&arena, 1).chunk() > later.chunk());
    }

    #[test]
    fn exhausted_addresses_fail_before_installing_rows() {
        let arena = RowArena::with_capacity(0);
        arena.state.inner.write().next_chunk_id = CHUNK_ID_LIMIT;
        assert!(arena.reserve(2).is_err());
        assert!(arena.read_guard().is_empty());
        assert_eq!(arena.state.inner.read().reserved, 0);
        assert_eq!(arena.capacity_bytes(), 0);
    }

    #[test]
    fn middle_chunk_removal_preserves_neighbors_and_releases_the_directory() {
        let arena = RowArena::with_capacity(0);
        let mut slots = Vec::new();
        let mut reservations = Vec::new();
        for id in [0, ARENA_CHUNK_ROWS as i64, 2 * ARENA_CHUNK_ROWS as i64] {
            let mut reserved = arena.reserve(ARENA_CHUNK_ROWS).unwrap();
            let (slot, _) = arena.install(
                &mut reserved,
                None,
                id,
                1,
                CompactArc::from(vec![Value::Integer(id)]),
            );
            slots.push(slot);
            reservations.push(reserved);
        }
        drop(reservations);
        let before = arena.capacity_bytes();
        let mut retired = arena.prepare_clear(3);
        assert_eq!(arena.clear_batch(&[slots[1]], &mut retired), 1);
        assert!(arena
            .read_guard()
            .get(slots[1], ARENA_CHUNK_ROWS as i64)
            .is_none());
        assert!(arena.read_guard().get(slots[0], 0).is_some());
        assert!(arena
            .read_guard()
            .get(slots[2], 2 * ARENA_CHUNK_ROWS as i64)
            .is_some());
        drop(retired);
        assert!(arena.capacity_bytes() < before);
        let mut retired = arena.prepare_clear(2);
        assert_eq!(arena.clear_batch(&[slots[0], slots[2]], &mut retired), 2);
        arena.finish_clear(&mut retired);
        drop(retired);
        assert_eq!(arena.capacity_bytes(), 0);
        assert!(insert(&arena, 50).chunk() > slots[2].chunk());
    }

    #[test]
    fn existing_head_reservation_does_not_grow_a_full_chunk() {
        let arena = RowArena::with_capacity(0);
        let mut reserved = arena.reserve(ARENA_CHUNK_ROWS).unwrap();
        let (slot, _) = arena.install(
            &mut reserved,
            None,
            5,
            1,
            CompactArc::from(vec![Value::Integer(5)]),
        );
        let capacity = arena.capacity_bytes();
        let next_id = arena.state.inner.read().next_chunk_id;
        let mut update = arena.reserve_existing();
        assert!(arena.has_reserved_heads());
        assert!(matches!(
            arena.clear_all(),
            Err(Error::TableHasActiveTransactions)
        ));
        arena.install(
            &mut update,
            Some(slot),
            5,
            2,
            CompactArc::from(vec![Value::Integer(9)]),
        );
        drop(update);
        assert!(!arena.has_reserved_heads());
        assert_eq!(arena.capacity_bytes(), capacity);
        assert_eq!(arena.state.inner.read().next_chunk_id, next_id);
    }

    #[test]
    fn payload_bytes_include_spare_text_capacity() {
        let mut text = String::with_capacity(4096);
        text.push_str("a text value that lives on the heap");
        let values = [Value::Text(crate::common::SmartString::from_string(text))];
        assert!(row_bytes(&values) >= 16 + 16 + 40 + 4096);
    }

    #[test]
    fn shared_payload_amplification_survives_replacement_and_retirement() {
        let value = Value::Extension(CompactArc::from(vec![0u8; 65_536]));
        let data: CompactArc<[Value]> = CompactArc::from(vec![value; 65_536]);
        let expected = (2 * std::mem::size_of::<usize>() + std::mem::size_of_val(data.as_ref()))
            as u128
            + 65_536 * (2 * std::mem::size_of::<usize>() as u128 + 65_536);
        assert!(expected > u32::MAX as u128);
        let arena = RowArena::with_capacity(0);
        let mut reservation = arena.reserve(1).unwrap();
        let (slot, charge) = arena.install(&mut reservation, None, 1, 1, CompactArc::clone(&data));
        assert_eq!(charge, expected);
        assert_eq!(arena.state.inner.read().payload_bytes, expected);
        drop(reservation);
        let current = (2 * std::mem::size_of::<usize>() + std::mem::size_of::<Value>()) as u128;
        let mut update = arena.reserve_existing();
        arena.install(
            &mut update,
            Some(slot),
            1,
            2,
            CompactArc::from(vec![Value::Integer(2)]),
        );
        assert_eq!(arena.state.inner.read().payload_bytes, current);
        arena.install(&mut update, Some(slot), 1, 3, data);
        assert_eq!(arena.state.inner.read().payload_bytes, expected);
        drop(update);
        let mut retired = arena.prepare_clear(1);
        arena.clear_batch(&[slot], &mut retired);
        assert_eq!(arena.bytes(), 0);
        assert_eq!(*arena.state.account.retired_arena_payloads.lock(), expected);
        let replacement = insert(&arena, 2);
        assert_eq!(arena.state.inner.read().payload_bytes, current);
        drop(retired);
        assert_eq!(*arena.state.account.retired_arena_payloads.lock(), 0);
        assert!(arena.read_guard().get(replacement, 2).is_some());
        let retired = arena.clear_all().unwrap();
        assert_eq!(*arena.state.account.retired_arena_payloads.lock(), current);
        drop(retired);
        assert_eq!(*arena.state.account.retired_arena_payloads.lock(), 0);
    }

    #[test]
    fn preparation_between_removal_batches_keeps_both_directories_owned() {
        let arena = RowArena::with_capacity(0);
        let mut first = arena.reserve(ARENA_CHUNK_ROWS).unwrap();
        let (a, _) = arena.install(
            &mut first,
            None,
            1,
            1,
            CompactArc::from(vec![Value::Integer(1)]),
        );
        let mut second = arena.reserve(ARENA_CHUNK_ROWS).unwrap();
        let (b, _) = arena.install(
            &mut second,
            None,
            2,
            1,
            CompactArc::from(vec![Value::Integer(2)]),
        );
        drop(first);
        let mut retired = arena.prepare_clear(2);
        assert_eq!(arena.clear_batch(&[a], &mut retired), 1);
        let third = arena.reserve(1).unwrap();
        drop(second);
        assert_eq!(arena.clear_batch(&[b], &mut retired), 1);
        arena.finish_clear(&mut retired);
        assert!(retired.directory.capacity() > 0);
        assert_eq!(arena.state.inner.read().frozen.capacity(), 0);
        drop(third);
        drop(retired);
        assert_eq!(arena.capacity_bytes(), 0);
    }

    #[test]
    fn update_reuses_the_payload_slot_and_releases_its_reservation() {
        let arena = RowArena::with_capacity(0);
        let slot = insert(&arena, 10);
        let data = CompactArc::from(vec![
            Value::Integer(10),
            Value::text("a retained heap text payload"),
        ]);
        let expected = row_bytes(&data);
        let mut reservation = arena.reserve(1).unwrap();
        assert_eq!(
            arena.install(&mut reservation, Some(slot), 10, 2, data),
            (slot, expected)
        );
        drop(reservation);
        assert_eq!(arena.bytes() as u128, expected);
        assert_eq!(arena.read_guard().row_bound(), expected);
        let mut reservation = arena.reserve(1).unwrap();
        arena.install(
            &mut reservation,
            Some(slot),
            10,
            3,
            CompactArc::from(vec![Value::Integer(10)]),
        );
        assert!(arena.clear_all().is_err());
        assert_eq!(arena.read_guard().row_bound(), expected);
        drop(reservation);
        let mut retirement = arena.prepare_clear(2);
        assert_eq!(arena.clear_batch(&[slot, slot], &mut retirement), 1);
        assert_eq!(arena.bytes(), 0);
        assert_eq!(arena.read_guard().row_bound(), 0);
        insert(&arena, 11);
        assert!(arena.read_guard().row_bound() < expected);
        drop(arena.clear_all().unwrap());
        assert_eq!(arena.read_guard().row_bound(), 0);
    }
}
