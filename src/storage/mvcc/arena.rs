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

//! Chunked row storage with stable, thin handles and shared row payloads.
//!
//! One lock protects the directory, payloads, metadata and active free list.
//! A chunk starts with no slot allocation and grows only as rows arrive.
//! Frozen chunks remain mutable, but their vacant slots are never reused.

use hashbrown::HashMap;
use parking_lot::{Mutex, RwLock};
use smallvec::SmallVec;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::common::{CompactArc, MemoryAccount, MemoryCharge};
use crate::core::{Error, Result, Row, Value};

pub const ARENA_CHUNK_SHIFT: u32 = 18;
pub const ARENA_CHUNK_ROWS: usize = 1 << ARENA_CHUNK_SHIFT;
pub const ARENA_CHUNK_MASK: u64 = (ARENA_CHUNK_ROWS - 1) as u64;
const MAX_CHUNK_ID: u64 = u64::MAX >> ARENA_CHUNK_SHIFT;
const LSN_LEAF_SHIFT: usize = 8;
const LSN_LEAF_ROWS: usize = 1 << LSN_LEAF_SHIFT;

/// Monotonically allocated chunk identity. It is independent of SQL row IDs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct ChunkId(u64);

impl ChunkId {
    pub fn new(value: u64) -> Result<Self> {
        if value > MAX_CHUNK_ID {
            return Err(Error::internal("arena chunk address space exhausted"));
        }
        Ok(Self(value))
    }

    #[inline]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// An encoded chunk/slot address plus one. `Option<ArenaId>` is also 8 bytes.
/// A handle must be checked against row/transaction identity before mutation;
/// vacant slots in the active chunk can be reused.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct ArenaId(NonZeroU64);

impl ArenaId {
    pub fn from_parts(chunk_id: ChunkId, slot: usize) -> Result<Self> {
        if slot >= ARENA_CHUNK_ROWS {
            return Err(Error::internal("arena slot exceeds chunk capacity"));
        }
        let encoded = chunk_id
            .0
            .checked_mul(ARENA_CHUNK_ROWS as u64)
            .and_then(|base| base.checked_add(slot as u64))
            .and_then(|address| address.checked_add(1))
            .and_then(NonZeroU64::new)
            .ok_or_else(|| Error::internal("arena address space exhausted"))?;
        Ok(Self(encoded))
    }

    #[inline]
    pub const fn from_nonzero(encoded: NonZeroU64) -> Self {
        Self(encoded)
    }

    #[inline]
    pub const fn as_nonzero(self) -> NonZeroU64 {
        self.0
    }

    #[inline]
    pub const fn chunk_id(self) -> ChunkId {
        ChunkId((self.0.get() - 1) >> ARENA_CHUNK_SHIFT)
    }

    #[inline]
    pub const fn slot(self) -> usize {
        ((self.0.get() - 1) & ARENA_CHUNK_MASK) as usize
    }
}

const _: () = {
    assert!(std::mem::size_of::<ArenaId>() == 8);
    assert!(std::mem::size_of::<Option<ArenaId>>() == 8);
    assert!(
        std::mem::size_of::<Option<CompactArc<[Value]>>>()
            == std::mem::size_of::<CompactArc<[Value]>>()
    );
};

/// Bytes referenced by a committed slot, distinct from structural capacity.
/// Shared payload allocation accounting follows the final payload owner.
pub fn row_bytes(values: &[Value]) -> usize {
    let mut bytes = 16 + std::mem::size_of_val(values);
    for value in values {
        match value {
            Value::Text(s) if s.is_heap() => bytes += 40 + s.heap_capacity(),
            Value::Extension(bytes_ref) => bytes += 16 + bytes_ref.len(),
            _ => {}
        }
    }
    bytes
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ArenaRowMeta {
    pub row_id: i64,
    /// Transaction that created the payload. Zero denotes a vacant slot.
    pub txn_id: i64,
    pub deleted_at_txn_id: i64,
    /// WAL record needed to reconstruct the latest mutation, including DELETE.
    /// None denotes an unlogged row or the separately retained legacy baseline.
    pub source_lsn: Option<NonZeroU64>,
}

impl ArenaRowMeta {
    #[inline]
    pub fn is_deleted(&self) -> bool {
        self.deleted_at_txn_id != 0
    }

    #[inline]
    pub fn mutation_txn_id(&self) -> i64 {
        if self.is_deleted() {
            self.deleted_at_txn_id
        } else {
            self.txn_id
        }
    }
}

const _: () = assert!(std::mem::size_of::<ArenaRowMeta>() == 32);

#[inline]
fn min_lsn(a: Option<NonZeroU64>, b: Option<NonZeroU64>) -> Option<NonZeroU64> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, None) => a,
        (None, b) => b,
    }
}

fn reserve_error(error: std::collections::TryReserveError) -> Error {
    Error::internal(format!("arena allocation failed: {error}"))
}

/// Grow accounted buffers explicitly so the old and new allocation coexist
/// under separate charges. An opaque realloc cannot expose that overlap.
fn reserve_charged_vec<T>(
    buffer: &mut Vec<T>,
    desired: usize,
    charge: &mut Option<MemoryCharge>,
) -> Result<()> {
    if desired <= buffer.capacity() {
        return Ok(());
    }
    let Some(charge) = charge else {
        return buffer
            .try_reserve_exact(desired - buffer.len())
            .map_err(reserve_error);
    };
    let mut replacement = Vec::new();
    replacement
        .try_reserve_exact(desired)
        .map_err(reserve_error)?;
    let replacement_charge = MemoryCharge::new(
        charge.account(),
        replacement.capacity() * std::mem::size_of::<T>(),
    );
    let old_bytes = buffer.capacity() * std::mem::size_of::<T>();
    replacement.append(buffer);
    let retired = std::mem::replace(buffer, replacement);
    let retired_charge = charge.replace_part(old_bytes, replacement_charge);
    drop(retired);
    drop(retired_charge);
    Ok(())
}

/// A binary tree over 256-slot leaf minima. It is absent for unlogged chunks.
/// Growth occurs once per leaf-capacity doubling, never on an existing-row update.
#[derive(Default)]
struct LsnMinTree {
    nodes: Vec<Option<NonZeroU64>>,
    leaf_capacity: usize,
    // Own the buffer independently so growth can retain old and new charges.
    charge: Option<MemoryCharge>,
}

impl LsnMinTree {
    fn new(account: Option<&MemoryAccount>) -> Self {
        Self {
            charge: account.map(|account| MemoryCharge::new(account, 0)),
            ..Self::default()
        }
    }

    fn reserve_slot(&mut self, slot: usize) -> Result<()> {
        let needed = (slot >> LSN_LEAF_SHIFT) + 1;
        if needed <= self.leaf_capacity {
            return Ok(());
        }
        let new_capacity = needed.next_power_of_two();
        let mut nodes = Vec::new();
        nodes
            .try_reserve_exact(new_capacity * 2)
            .map_err(reserve_error)?;
        let new_charge = self.charge.as_ref().map(|charge| {
            MemoryCharge::new(
                charge.account(),
                nodes.capacity() * std::mem::size_of::<Option<NonZeroU64>>(),
            )
        });
        nodes.resize(new_capacity * 2, None);
        if self.leaf_capacity != 0 {
            nodes[new_capacity..new_capacity + self.leaf_capacity]
                .copy_from_slice(&self.nodes[self.leaf_capacity..self.leaf_capacity * 2]);
        }
        for parent in (1..new_capacity).rev() {
            nodes[parent] = min_lsn(nodes[parent * 2], nodes[parent * 2 + 1]);
        }
        let old_nodes = std::mem::replace(&mut self.nodes, nodes);
        let old_charge = std::mem::replace(&mut self.charge, new_charge);
        drop(old_nodes);
        drop(old_charge);
        self.leaf_capacity = new_capacity;
        Ok(())
    }

    #[inline]
    fn first(&self) -> Option<NonZeroU64> {
        self.nodes.get(1).copied().flatten()
    }

    fn changed(
        &mut self,
        slot: usize,
        old: Option<NonZeroU64>,
        new: Option<NonZeroU64>,
        meta: &[ArenaRowMeta],
    ) {
        if old == new || self.leaf_capacity == 0 {
            return;
        }
        let leaf = slot >> LSN_LEAF_SHIFT;
        if leaf >= self.leaf_capacity {
            debug_assert!(new.is_none());
            return;
        }
        let mut node = self.leaf_capacity + leaf;
        let previous = self.nodes[node];
        let minimum = if new.is_some() && (previous.is_none() || new < previous) {
            new
        } else if old.is_some() && old == previous {
            let begin = leaf * LSN_LEAF_ROWS;
            let end = (begin + LSN_LEAF_ROWS).min(meta.len());
            meta[begin..end]
                .iter()
                .filter_map(|row| row.source_lsn)
                .min()
        } else {
            return;
        };
        if minimum == previous {
            return;
        }
        self.nodes[node] = minimum;
        while node > 1 {
            node /= 2;
            let value = min_lsn(self.nodes[node * 2], self.nodes[node * 2 + 1]);
            if self.nodes[node] == value {
                break;
            }
            self.nodes[node] = value;
        }
    }
}

struct Chunk {
    id: ChunkId,
    data: Vec<Option<CompactArc<[Value]>>>,
    meta: Vec<ArenaRowMeta>,
    lsn_minima: LsnMinTree,
    occupied: usize,
    frozen: bool,
    // Last field: vector allocations are freed before their charge is released.
    charge: Option<MemoryCharge>,
}

impl Chunk {
    fn new(id: ChunkId, account: Option<&MemoryAccount>) -> Self {
        Self {
            id,
            data: Vec::new(),
            meta: Vec::new(),
            lsn_minima: LsnMinTree::new(account),
            occupied: 0,
            frozen: false,
            charge: account.map(|account| MemoryCharge::new(account, 0)),
        }
    }

    fn capacity_bytes(&self) -> usize {
        // The LSN tree owns a separate charge for its explicit replacement buffer.
        self.data.capacity() * std::mem::size_of::<Option<CompactArc<[Value]>>>()
            + self.meta.capacity() * std::mem::size_of::<ArenaRowMeta>()
    }

    fn refresh_charge(&mut self) {
        let bytes = self.capacity_bytes();
        if let Some(charge) = &mut self.charge {
            if charge.bytes() != bytes {
                charge.resize(bytes);
            }
        }
    }

    fn reserve_lsn(&mut self, slot: usize) -> Result<()> {
        let result = self.lsn_minima.reserve_slot(slot);
        self.refresh_charge();
        result
    }

    fn reserve_slot(&mut self, slot: usize, hint: usize, logged: bool) -> Result<()> {
        // One vector may grow before the next reservation fails. Account that
        // retained capacity on both the success and error paths.
        let result = self.reserve_slot_inner(slot, hint, logged);
        self.refresh_charge();
        result
    }

    fn reserve_slot_inner(&mut self, slot: usize, hint: usize, logged: bool) -> Result<()> {
        debug_assert!(slot < ARENA_CHUNK_ROWS);
        let capacity = self.data.capacity().min(self.meta.capacity());
        if slot >= capacity {
            let desired = (slot + 1)
                .max(capacity.saturating_mul(2))
                .max(hint)
                .clamp(4, ARENA_CHUNK_ROWS);
            reserve_charged_vec(&mut self.data, desired, &mut self.charge)?;
            reserve_charged_vec(&mut self.meta, desired, &mut self.charge)?;
        }
        if logged {
            self.lsn_minima.reserve_slot(slot)?;
        }
        Ok(())
    }

    #[inline]
    fn get(&self, slot: usize) -> Option<(&ArenaRowMeta, &CompactArc<[Value]>)> {
        Some((self.meta.get(slot)?, self.data.get(slot)?.as_ref()?))
    }
}

/// Detached chunk ownership, released only after a truncate transfer unlocks.
pub(crate) struct RetiredArena {
    _inner: ArenaInner,
}

struct ArenaInner {
    chunks: Vec<Chunk>,
    active: Option<ChunkId>,
    next_chunk_id: u64,
    /// Intrusive links live in row_id only while metadata.txn_id == 0.
    /// Frozen chunks never consult their retired links.
    free_head: Option<u32>,
    occupied: usize,
    slot_count: usize,
    initial_capacity: usize,
    // Kept after chunks so the directory buffer is released first on Drop.
    directory_charge: Option<MemoryCharge>,
}

impl ArenaInner {
    fn refresh_directory_charge(&mut self) {
        if let Some(charge) = &mut self.directory_charge {
            let bytes = self.chunks.capacity() * std::mem::size_of::<Chunk>();
            if charge.bytes() != bytes {
                charge.resize(bytes);
            }
        }
    }

    fn reserve_directory_slot(&mut self) -> Result<()> {
        if self.chunks.len() < self.chunks.capacity() {
            return Ok(());
        }
        let desired = self.chunks.capacity().saturating_mul(2).max(4);
        let mut chunks = Vec::new();
        chunks.try_reserve_exact(desired).map_err(reserve_error)?;
        let new_charge = self.directory_charge.as_ref().map(|charge| {
            MemoryCharge::new(
                charge.account(),
                chunks.capacity() * std::mem::size_of::<Chunk>(),
            )
        });
        // The replacement allocation exists beside the original until every
        // chunk has moved. Neither moving the entries nor swapping can allocate.
        chunks.append(&mut self.chunks);
        let old_chunks = std::mem::replace(&mut self.chunks, chunks);
        let old_charge = std::mem::replace(&mut self.directory_charge, new_charge);
        drop(old_chunks);
        drop(old_charge);
        Ok(())
    }

    #[inline]
    fn position(&self, id: ChunkId) -> Option<usize> {
        self.chunks.binary_search_by_key(&id, |chunk| chunk.id).ok()
    }

    fn get(&self, id: ArenaId) -> Option<(&ArenaRowMeta, &CompactArc<[Value]>)> {
        self.chunks
            .get(self.position(id.chunk_id())?)?
            .get(id.slot())
    }

    fn insert(
        &mut self,
        row_id: i64,
        txn_id: i64,
        data: CompactArc<[Value]>,
        source_lsn: Option<NonZeroU64>,
    ) -> Result<ArenaId> {
        if txn_id == 0 {
            return Err(Error::internal("arena transaction identity cannot be zero"));
        }
        if let Some(active) = self.active {
            let last = self.chunks.len() - 1;
            debug_assert_eq!(self.chunks[last].id, active);
            let slot = self
                .free_head
                .map_or(self.chunks[last].meta.len(), |slot| slot as usize);
            if slot < ARENA_CHUNK_ROWS {
                let id = ArenaId::from_parts(active, slot)?;
                let chunk = &mut self.chunks[last];
                chunk.reserve_slot(slot, self.initial_capacity, source_lsn.is_some())?;
                let meta = ArenaRowMeta {
                    row_id,
                    txn_id,
                    deleted_at_txn_id: 0,
                    source_lsn,
                };
                if self.free_head.is_some() {
                    debug_assert!(chunk.data[slot].is_none());
                    debug_assert_eq!(chunk.meta[slot].txn_id, 0);
                    self.free_head = match chunk.meta[slot].row_id {
                        0 => None,
                        next => Some((next - 1) as u32),
                    };
                    chunk.data[slot] = Some(data);
                    chunk.meta[slot] = meta;
                } else {
                    chunk.data.push(Some(data));
                    chunk.meta.push(meta);
                    self.slot_count += 1;
                }
                chunk.occupied += 1;
                self.occupied += 1;
                chunk
                    .lsn_minima
                    .changed(slot, None, source_lsn, &chunk.meta);
                return Ok(id);
            }
        }
        // All address and structural allocation checks precede publication of
        // the new chunk or freezing of its predecessor.
        let chunk_id = ChunkId::new(self.next_chunk_id)?;
        let id = ArenaId::from_parts(chunk_id, 0)?;
        let account = self.directory_charge.as_ref().map(MemoryCharge::account);
        let mut chunk = Chunk::new(chunk_id, account);
        chunk.reserve_slot(0, self.initial_capacity, source_lsn.is_some())?;
        self.reserve_directory_slot()?;
        chunk.data.push(Some(data));
        chunk.meta.push(ArenaRowMeta {
            row_id,
            txn_id,
            deleted_at_txn_id: 0,
            source_lsn,
        });
        chunk.occupied = 1;
        chunk.lsn_minima.changed(0, None, source_lsn, &chunk.meta);
        if let Some(previous) = self.chunks.last_mut() {
            previous.frozen = true;
        }
        self.free_head = None;
        self.chunks.push(chunk);
        self.active = Some(chunk_id);
        self.next_chunk_id += 1;
        self.occupied += 1;
        self.slot_count += 1;
        Ok(id)
    }

    fn clear(&mut self, id: ArenaId, row_id: i64, txn_id: i64) -> usize {
        let Some(position) = self.position(id.chunk_id()) else {
            return 0;
        };
        let chunk = &mut self.chunks[position];
        let slot = id.slot();
        let Some((meta, payload)) = chunk.get(slot) else {
            return 0;
        };
        if meta.row_id != row_id || meta.mutation_txn_id() != txn_id {
            return 0;
        }
        let bytes = row_bytes(payload);
        let old_lsn = meta.source_lsn;
        chunk.data[slot] = None;
        chunk.meta[slot] = ArenaRowMeta::default();
        chunk.occupied -= 1;
        chunk.lsn_minima.changed(slot, old_lsn, None, &chunk.meta);
        self.occupied -= 1;
        if chunk.occupied == 0 {
            self.slot_count -= chunk.meta.len();
            if self.active == Some(chunk.id) {
                self.active = None;
                self.free_head = None;
            }
            self.chunks.remove(position);
            if self.chunks.is_empty() {
                self.chunks = Vec::new();
                self.refresh_directory_charge();
            }
        } else if self.active == Some(chunk.id) {
            // Vacant metadata carries the next slot plus one. Occupancy is
            // identified by txn_id, so row_id == 0 remains a valid SQL key.
            // This path allocates nothing and repeated clear cannot link twice.
            chunk.meta[slot].row_id = self.free_head.map_or(0, |next| i64::from(next) + 1);
            self.free_head = Some(slot as u32);
        }
        bytes
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Receipt {
    lsn: NonZeroU64,
    token: u64,
}

type ReceiptHeap = SmallVec<[Receipt; 1]>;
type ReceiptPositions = HashMap<u64, (ChunkId, usize)>;

/// Receipt maps use primitive, non-panicking keys. Allocate and charge a
/// replacement before moving entries so rehash retains the old bucket owner.
fn reserve_receipt_map<K: Eq + std::hash::Hash, V>(
    map: &mut HashMap<K, V>,
    additional: usize,
    charge: &mut Option<MemoryCharge>,
) -> Result<()> {
    let needed = map
        .len()
        .checked_add(additional)
        .ok_or_else(|| Error::internal("arena receipt capacity overflow"))?;
    if needed <= map.capacity() {
        return Ok(());
    }
    let Some(charge) = charge else {
        return map.try_reserve(additional).map_err(|error| {
            Error::internal(format!("arena receipt allocation failed: {error:?}"))
        });
    };
    let desired = needed.max(map.capacity().saturating_mul(2));
    let mut replacement = HashMap::with_hasher(map.hasher().clone());
    replacement
        .try_reserve(desired)
        .map_err(|error| Error::internal(format!("arena receipt allocation failed: {error:?}")))?;
    let replacement_charge = MemoryCharge::new(charge.account(), replacement.allocation_size());
    let old_bytes = map.allocation_size();
    // The replacement has room for every existing entry and the requested
    // additions. Rehashing the fixed receipt keys cannot allocate or unwind.
    replacement.extend(map.drain());
    let retired = std::mem::replace(map, replacement);
    let retired_charge = charge.replace_part(old_bytes, replacement_charge);
    drop(retired);
    drop(retired_charge);
    Ok(())
}

/// Indexed per-chunk min-heaps. A pin or release touches only its chunk's
/// heap, and token lookup is independent of how many other chunks are pinned.
#[derive(Default)]
struct ReceiptRegistry {
    /// Ordinary one-chunk publication needs no receipt-table allocations.
    /// Inline and indexed states are mutually exclusive.
    inline: Option<(ChunkId, Receipt)>,
    chunks: HashMap<ChunkId, ReceiptHeap>,
    positions: ReceiptPositions,
    next_token: u64,
    spilled_heap_bytes: usize,
    #[cfg(test)]
    fail_promotion_after_positions: bool,
    // Hash tables/heaps above are destroyed before releasing their charge.
    charge: Option<MemoryCharge>,
}

impl ReceiptRegistry {
    fn new(account: Option<&MemoryAccount>) -> Self {
        Self {
            charge: account.map(|account| MemoryCharge::new(account, 0)),
            ..Self::default()
        }
    }

    fn heap_bytes(heap: &ReceiptHeap) -> usize {
        if heap.spilled() {
            heap.capacity() * std::mem::size_of::<Receipt>()
        } else {
            0
        }
    }

    fn reserve_heap(
        heap: &mut ReceiptHeap,
        additional: usize,
        charge: &mut Option<MemoryCharge>,
    ) -> Result<()> {
        let needed = heap
            .len()
            .checked_add(additional)
            .ok_or_else(|| Error::internal("arena receipt capacity overflow"))?;
        if needed <= heap.capacity() {
            return Ok(());
        }
        let Some(charge) = charge else {
            return heap.try_reserve(additional).map_err(|error| {
                Error::internal(format!("arena receipt allocation failed: {error:?}"))
            });
        };
        let desired = needed.max(heap.capacity().saturating_mul(2));
        let mut replacement = ReceiptHeap::new();
        replacement.try_reserve_exact(desired).map_err(|error| {
            Error::internal(format!("arena receipt allocation failed: {error:?}"))
        })?;
        let replacement_charge =
            MemoryCharge::new(charge.account(), Self::heap_bytes(&replacement));
        let old_bytes = Self::heap_bytes(heap);
        // Receipt is Copy and the full destination capacity is already present.
        replacement.extend_from_slice(heap.as_slice());
        let retired = std::mem::replace(heap, replacement);
        let retired_charge = charge.replace_part(old_bytes, replacement_charge);
        drop(retired);
        drop(retired_charge);
        Ok(())
    }

    fn refresh_charge(&mut self) {
        let bytes = self.capacity_bytes();
        if let Some(charge) = &mut self.charge {
            if charge.bytes() != bytes {
                charge.resize(bytes);
            }
        }
    }

    fn first(&self, chunk: Option<ChunkId>) -> Option<NonZeroU64> {
        if let Some((id, receipt)) = self.inline {
            return chunk
                .is_none_or(|requested| requested == id)
                .then_some(receipt.lsn);
        }
        if let Some(id) = chunk {
            self.chunks
                .get(&id)
                .and_then(|heap| heap.first())
                .map(|receipt| receipt.lsn)
        } else {
            self.chunks
                .values()
                .filter_map(|heap| heap.first().map(|receipt| receipt.lsn))
                .min()
        }
    }

    fn swap(heap: &mut ReceiptHeap, positions: &mut ReceiptPositions, a: usize, b: usize) {
        heap.swap(a, b);
        positions.get_mut(&heap[a].token).unwrap().1 = a;
        positions.get_mut(&heap[b].token).unwrap().1 = b;
    }

    fn sift_up(heap: &mut ReceiptHeap, positions: &mut ReceiptPositions, mut slot: usize) {
        while slot > 0 {
            let parent = (slot - 1) / 2;
            if heap[parent] <= heap[slot] {
                break;
            }
            Self::swap(heap, positions, parent, slot);
            slot = parent;
        }
    }

    fn sift_down(heap: &mut ReceiptHeap, positions: &mut ReceiptPositions, mut slot: usize) {
        loop {
            let left = slot * 2 + 1;
            if left >= heap.len() {
                break;
            }
            let right = left + 1;
            let child = if right < heap.len() && heap[right] < heap[left] {
                right
            } else {
                left
            };
            if heap[slot] <= heap[child] {
                break;
            }
            Self::swap(heap, positions, slot, child);
            slot = child;
        }
    }

    fn insert(&mut self, chunk_id: ChunkId, lsn: NonZeroU64) -> Result<u64> {
        let result = self.insert_inner(chunk_id, lsn);
        self.refresh_charge();
        result
    }

    fn insert_inner(&mut self, chunk_id: ChunkId, lsn: NonZeroU64) -> Result<u64> {
        let token = self
            .next_token
            .checked_add(1)
            .ok_or_else(|| Error::internal("arena LSN receipt identities exhausted"))?;
        if self.positions.is_empty() {
            if let Some((old_chunk, old_receipt)) = self.inline {
                // Keep the original receipt inline until every allocation for
                // promotion succeeds. A failure retains its token and minimum.
                reserve_receipt_map(&mut self.positions, 2, &mut self.charge)?;
                #[cfg(test)]
                if std::mem::take(&mut self.fail_promotion_after_positions) {
                    return Err(Error::internal(
                        "injected arena receipt promotion allocation failure",
                    ));
                }
                reserve_receipt_map(&mut self.chunks, 2, &mut self.charge)?;
                let mut heap = ReceiptHeap::new();
                Self::reserve_heap(
                    &mut heap,
                    if old_chunk == chunk_id { 2 } else { 1 },
                    &mut self.charge,
                )?;
                heap.push(old_receipt);
                self.positions.insert(old_receipt.token, (old_chunk, 0));
                if old_chunk == chunk_id {
                    heap.push(Receipt { lsn, token });
                    self.positions.insert(token, (chunk_id, 1));
                    Self::sift_up(&mut heap, &mut self.positions, 1);
                } else {
                    let mut other = ReceiptHeap::new();
                    other.push(Receipt { lsn, token });
                    self.chunks.insert(chunk_id, other);
                    self.positions.insert(token, (chunk_id, 0));
                }
                self.spilled_heap_bytes += Self::heap_bytes(&heap);
                self.chunks.insert(old_chunk, heap);
                self.inline = None;
            } else {
                self.inline = Some((chunk_id, Receipt { lsn, token }));
            }
            self.next_token = token;
            return Ok(token);
        }
        reserve_receipt_map(&mut self.positions, 1, &mut self.charge)?;
        if let Some(heap) = self.chunks.get_mut(&chunk_id) {
            let previous_bytes = Self::heap_bytes(heap);
            Self::reserve_heap(heap, 1, &mut self.charge)?;
            self.spilled_heap_bytes += Self::heap_bytes(heap) - previous_bytes;
            let slot = heap.len();
            heap.push(Receipt { lsn, token });
            self.positions.insert(token, (chunk_id, slot));
            Self::sift_up(heap, &mut self.positions, slot);
        } else {
            reserve_receipt_map(&mut self.chunks, 1, &mut self.charge)?;
            // The first receipt is inline; no separate allocation per handle.
            let mut heap = ReceiptHeap::new();
            heap.push(Receipt { lsn, token });
            self.chunks.insert(chunk_id, heap);
            self.positions.insert(token, (chunk_id, 0));
        }
        self.next_token = token;
        Ok(token)
    }

    fn remove(&mut self, token: u64) {
        if let Some((_, receipt)) = self.inline {
            if receipt.token == token {
                self.inline = None;
                // A failed promotion may have reserved empty hash tables.
                self.positions = HashMap::new();
                self.chunks = HashMap::new();
                self.refresh_charge();
            }
            return;
        }
        let Some((chunk_id, slot)) = self.positions.remove(&token) else {
            return;
        };
        let heap = self.chunks.get_mut(&chunk_id).unwrap();
        heap.swap_remove(slot);
        if slot < heap.len() {
            self.positions.get_mut(&heap[slot].token).unwrap().1 = slot;
            if slot > 0 && heap[slot] < heap[(slot - 1) / 2] {
                Self::sift_up(heap, &mut self.positions, slot);
            } else {
                Self::sift_down(heap, &mut self.positions, slot);
            }
        }
        if heap.is_empty() {
            self.spilled_heap_bytes -= Self::heap_bytes(heap);
            self.chunks.remove(&chunk_id);
        } else if heap.len() == 1 && heap.spilled() {
            // Return to inline storage without allocating in Drop.
            let remaining = heap.pop().unwrap();
            self.spilled_heap_bytes -= Self::heap_bytes(heap);
            *heap = ReceiptHeap::new();
            heap.push(remaining);
        }
        if self.positions.len() == 1 {
            let (&remaining_token, &(remaining_chunk, remaining_slot)) =
                self.positions.iter().next().unwrap();
            let receipt = self.chunks[&remaining_chunk][remaining_slot];
            debug_assert_eq!(receipt.token, remaining_token);
            self.inline = Some((remaining_chunk, receipt));
            self.positions = HashMap::new();
            self.chunks = HashMap::new();
            self.spilled_heap_bytes = 0;
        } else if self.positions.is_empty() {
            self.positions = HashMap::new();
            self.chunks = HashMap::new();
        }
        self.refresh_charge();
    }

    fn capacity_bytes(&self) -> usize {
        self.chunks.allocation_size() + self.positions.allocation_size() + self.spilled_heap_bytes
    }
}

/// A durability pin independent of the chunk's allocation. Held through commit
/// or undo, it prevents a replaced/deleted row from advancing the WAL floor.
/// A later durable manifest receipt can take ownership with `pin_lsn`.
pub struct ArenaLsnPin {
    registry: CompactArc<Mutex<ReceiptRegistry>>,
    token: u64,
}

impl Drop for ArenaLsnPin {
    fn drop(&mut self) {
        self.registry.lock().remove(self.token);
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ArenaCapacity {
    pub directory_bytes: usize,
    pub payload_slots_bytes: usize,
    pub metadata_bytes: usize,
    pub free_list_bytes: usize,
    pub lsn_tree_bytes: usize,
    pub receipt_bytes: usize,
}

impl ArenaCapacity {
    pub fn total(self) -> usize {
        self.directory_bytes
            + self.payload_slots_bytes
            + self.metadata_bytes
            + self.free_list_bytes
            + self.lsn_tree_bytes
            + self.receipt_bytes
    }
}

pub struct RowArena {
    inner: RwLock<ArenaInner>,
    receipts: CompactArc<Mutex<ReceiptRegistry>>,
    /// Committed-slot payload estimate, separate from capacity/last-owner charge.
    bytes: AtomicUsize,
}

impl RowArena {
    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    /// A lazy hint, capped to one chunk. No slot vectors are allocated yet.
    pub fn with_capacity(row_capacity: usize) -> Self {
        Self::with_optional_account(row_capacity, None)
    }

    /// Account arena capacity and receipt ownership. Payloads retain their own
    /// allocation accounts; VersionStore certifies them through Row::into_hot
    /// before insertion. Standalone callers must account payloads separately.
    pub fn with_account(row_capacity: usize, account: &MemoryAccount) -> Self {
        Self::with_optional_account(row_capacity, Some(account))
    }

    fn with_optional_account(row_capacity: usize, account: Option<&MemoryAccount>) -> Self {
        let registry = Mutex::new(ReceiptRegistry::new(account));
        let receipts = match account {
            Some(account) => CompactArc::new_in(registry, account),
            None => CompactArc::new(registry),
        };
        Self {
            inner: RwLock::new(ArenaInner {
                chunks: Vec::new(),
                active: None,
                next_chunk_id: 0,
                free_head: None,
                occupied: 0,
                slot_count: 0,
                initial_capacity: row_capacity.min(ARENA_CHUNK_ROWS),
                directory_charge: account.map(|account| MemoryCharge::new(account, 0)),
            }),
            receipts,
            bytes: AtomicUsize::new(0),
        }
    }

    /// Convenience constructors preserve CompactArc's allocation behavior;
    /// Result covers arena address and structural capacity failures. Payload
    /// construction occurs before any arena state changes.
    pub fn insert(
        &self,
        row_id: i64,
        txn_id: i64,
        values: &[Value],
        source_lsn: Option<NonZeroU64>,
    ) -> Result<ArenaId> {
        self.insert_arc(row_id, txn_id, CompactArc::from_slice(values), source_lsn)
    }

    pub fn insert_row(
        &self,
        row_id: i64,
        txn_id: i64,
        row: &Row,
        source_lsn: Option<NonZeroU64>,
    ) -> Result<ArenaId> {
        let data = row
            .as_arc()
            .cloned()
            .unwrap_or_else(|| CompactArc::from(row.iter().cloned().collect::<Vec<_>>()));
        self.insert_arc(row_id, txn_id, data, source_lsn)
    }

    pub fn insert_row_get_arc(
        &self,
        row_id: i64,
        txn_id: i64,
        row: &Row,
        source_lsn: Option<NonZeroU64>,
    ) -> Result<(ArenaId, CompactArc<[Value]>)> {
        let data = row
            .as_arc()
            .cloned()
            .unwrap_or_else(|| CompactArc::from(row.iter().cloned().collect::<Vec<_>>()));
        let id = self.insert_arc(row_id, txn_id, data.clone(), source_lsn)?;
        Ok((id, data))
    }

    /// Insert an existing payload without changing its allocation account.
    pub fn insert_arc(
        &self,
        row_id: i64,
        txn_id: i64,
        data: CompactArc<[Value]>,
        source_lsn: Option<NonZeroU64>,
    ) -> Result<ArenaId> {
        let bytes = row_bytes(&data);
        let mut inner = self.inner.write();
        let id = inner.insert(row_id, txn_id, data, source_lsn)?;
        self.bytes.fetch_add(bytes, Ordering::Relaxed);
        Ok(id)
    }

    /// Replace an occupied slot belonging to this row, including a frozen slot.
    /// LSN tree growth can fail before a formerly unlogged slot is changed.
    pub fn update_at(
        &self,
        id: ArenaId,
        row_id: i64,
        txn_id: i64,
        data: CompactArc<[Value]>,
        source_lsn: Option<NonZeroU64>,
    ) -> Result<bool> {
        if txn_id == 0 {
            return Err(Error::internal("arena transaction identity cannot be zero"));
        }
        let mut inner = self.inner.write();
        let Some(position) = inner.position(id.chunk_id()) else {
            return Ok(false);
        };
        let chunk = &mut inner.chunks[position];
        let Some((meta, old_data)) = chunk.get(id.slot()) else {
            return Ok(false);
        };
        if meta.row_id != row_id {
            return Ok(false);
        }
        let old_lsn = meta.source_lsn;
        let old_bytes = row_bytes(old_data);
        if source_lsn.is_some() {
            chunk.reserve_lsn(id.slot())?;
        }
        let new_bytes = row_bytes(&data);
        chunk.data[id.slot()] = Some(data);
        chunk.meta[id.slot()] = ArenaRowMeta {
            row_id,
            txn_id,
            deleted_at_txn_id: 0,
            source_lsn,
        };
        chunk
            .lsn_minima
            .changed(id.slot(), old_lsn, source_lsn, &chunk.meta);
        self.bytes.fetch_add(new_bytes, Ordering::Relaxed);
        self.bytes.fetch_sub(old_bytes, Ordering::Relaxed);
        Ok(true)
    }

    pub fn mark_deleted(
        &self,
        id: ArenaId,
        row_id: i64,
        deleted_at_txn_id: i64,
        source_lsn: Option<NonZeroU64>,
    ) -> Result<bool> {
        if deleted_at_txn_id == 0 {
            return Err(Error::internal(
                "arena deletion transaction identity cannot be zero",
            ));
        }
        let mut inner = self.inner.write();
        let Some(position) = inner.position(id.chunk_id()) else {
            return Ok(false);
        };
        let chunk = &mut inner.chunks[position];
        let Some((meta, _)) = chunk.get(id.slot()) else {
            return Ok(false);
        };
        if meta.row_id != row_id {
            return Ok(false);
        }
        let old_lsn = meta.source_lsn;
        if source_lsn.is_some() {
            chunk.reserve_lsn(id.slot())?;
        }
        chunk.meta[id.slot()].deleted_at_txn_id = deleted_at_txn_id;
        chunk.meta[id.slot()].source_lsn = source_lsn;
        chunk
            .lsn_minima
            .changed(id.slot(), old_lsn, source_lsn, &chunk.meta);
        Ok(true)
    }

    /// Clear only the expected latest mutation. Duplicate/stale clears do not
    /// release another row's bytes or add duplicate free-list entries.
    pub fn clear_at(&self, id: ArenaId, row_id: i64, txn_id: i64) -> bool {
        let mut inner = self.inner.write();
        let freed = inner.clear(id, row_id, txn_id);
        self.bytes.fetch_sub(freed, Ordering::Relaxed);
        freed != 0
    }

    pub fn clear_batch(&self, identities: &[(ArenaId, i64, i64)]) -> usize {
        let mut inner = self.inner.write();
        let mut count = 0;
        let mut bytes = 0;
        for &(id, row_id, txn_id) in identities {
            let freed = inner.clear(id, row_id, txn_id);
            count += usize::from(freed != 0);
            bytes += freed;
        }
        self.bytes.fetch_sub(bytes, Ordering::Relaxed);
        count
    }

    /// Retire the current free list. The next active chunk starts unallocated.
    pub fn freeze_active(&self) -> Result<Option<ChunkId>> {
        let mut inner = self.inner.write();
        let Some(active) = inner.active else {
            return Ok(None);
        };
        // Do not freeze successfully if opening a successor can never succeed.
        ChunkId::new(inner.next_chunk_id)?;
        inner.chunks.last_mut().unwrap().frozen = true;
        inner.active = None;
        inner.free_head = None;
        Ok(Some(active))
    }

    /// Release every slot allocation without recycling chunk identities or
    /// releasing independently owned durability receipts.
    pub fn clear_all(&self) {
        let mut inner = self.inner.write();
        inner.chunks = Vec::new();
        inner.refresh_directory_charge();
        inner.active = None;
        inner.free_head = None;
        inner.occupied = 0;
        inner.slot_count = 0;
        self.bytes.store(0, Ordering::Relaxed);
    }

    /// Detach ownership without dropping row payloads inside a transfer fence.
    /// The caller drops the returned owner after publication unlocks. Chunk
    /// identities and independent WAL receipts are never reset by TRUNCATE.
    pub(crate) fn take_all_for_truncate(&self) -> RetiredArena {
        let mut inner = self.inner.write();
        let replacement = ArenaInner {
            chunks: Vec::new(),
            active: None,
            next_chunk_id: inner.next_chunk_id,
            free_head: None,
            occupied: 0,
            slot_count: 0,
            initial_capacity: inner.initial_capacity,
            directory_charge: inner
                .directory_charge
                .as_ref()
                .map(|charge| MemoryCharge::new(charge.account(), 0)),
        };
        let retired = std::mem::replace(&mut *inner, replacement);
        self.bytes.store(0, Ordering::Relaxed);
        RetiredArena { _inner: retired }
    }

    pub fn active_chunk_id(&self) -> Option<ChunkId> {
        self.inner.read().active
    }
    pub fn len(&self) -> usize {
        self.inner.read().occupied
    }
    pub fn slot_count(&self) -> usize {
        self.inner.read().slot_count
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn bytes(&self) -> usize {
        self.bytes.load(Ordering::Relaxed)
    }

    pub fn capacity(&self) -> ArenaCapacity {
        let inner = self.inner.read();
        let receipts = self.receipts.lock();
        let mut result = ArenaCapacity {
            directory_bytes: inner.chunks.capacity() * std::mem::size_of::<Chunk>(),
            // Free links occupy otherwise vacant metadata, counted below.
            free_list_bytes: 0,
            receipt_bytes: self.receipts.allocation_size() + receipts.capacity_bytes(),
            ..ArenaCapacity::default()
        };
        for chunk in &inner.chunks {
            result.payload_slots_bytes +=
                chunk.data.capacity() * std::mem::size_of::<Option<CompactArc<[Value]>>>();
            result.metadata_bytes += chunk.meta.capacity() * std::mem::size_of::<ArenaRowMeta>();
            result.lsn_tree_bytes +=
                chunk.lsn_minima.nodes.capacity() * std::mem::size_of::<Option<NonZeroU64>>();
        }
        result
    }

    pub fn capacity_bytes(&self) -> usize {
        self.capacity().total()
    }

    pub fn read_guard(&self) -> ArenaReadGuard<'_> {
        ArenaReadGuard {
            inner: self.inner.read(),
        }
    }

    pub fn get_arc(&self, id: ArenaId) -> Option<CompactArc<[Value]>> {
        self.inner.read().get(id).map(|(_, data)| data.clone())
    }

    pub fn get_meta_and_arc(&self, id: ArenaId) -> Option<(ArenaRowMeta, CompactArc<[Value]>)> {
        self.inner
            .read()
            .get(id)
            .map(|(meta, data)| (*meta, data.clone()))
    }

    pub fn chunk_first_lsn(&self, id: ChunkId) -> Option<NonZeroU64> {
        let inner = self.inner.read();
        let rows = inner
            .position(id)
            .and_then(|position| inner.chunks[position].lsn_minima.first());
        min_lsn(rows, self.receipts.lock().first(Some(id)))
    }

    pub fn first_lsn(&self) -> Option<NonZeroU64> {
        let inner = self.inner.read();
        let rows = inner
            .chunks
            .iter()
            .filter_map(|chunk| chunk.lsn_minima.first())
            .min();
        min_lsn(rows, self.receipts.lock().first(None))
    }

    /// Pin the current row minimum before publication. Existing receipts are
    /// deliberately not inherited: overlapping writers must let completed
    /// predecessors release their older floor.
    pub fn pin_chunk_lsn(&self, id: ChunkId) -> Result<Option<ArenaLsnPin>> {
        let inner = self.inner.read();
        let rows = inner
            .position(id)
            .and_then(|position| inner.chunks[position].lsn_minima.first());
        let Some(lsn) = rows else {
            return Ok(None);
        };
        let mut receipts = self.receipts.lock();
        let token = receipts.insert(id, lsn)?;
        Ok(Some(ArenaLsnPin {
            registry: CompactArc::clone(&self.receipts),
            token,
        }))
    }

    /// Acquire an explicit receipt before relinquishing the row or old receipt
    /// whose durability ownership is being transferred.
    pub fn pin_lsn(&self, id: ChunkId, lsn: NonZeroU64) -> Result<ArenaLsnPin> {
        let token = self.receipts.lock().insert(id, lsn)?;
        Ok(ArenaLsnPin {
            registry: CompactArc::clone(&self.receipts),
            token,
        })
    }
}

impl Default for RowArena {
    fn default() -> Self {
        Self::new()
    }
}

/// Read-only chunk slices. Payload and metadata positions are parallel;
/// vacant payloads are None and metadata.txn_id is zero.
#[derive(Clone, Copy)]
pub struct ArenaChunkRef<'a> {
    chunk: &'a Chunk,
}

impl<'a> ArenaChunkRef<'a> {
    pub fn id(self) -> ChunkId {
        self.chunk.id
    }
    pub fn is_frozen(self) -> bool {
        self.chunk.frozen
    }
    pub fn len(self) -> usize {
        self.chunk.occupied
    }
    pub fn is_empty(self) -> bool {
        self.chunk.occupied == 0
    }
    pub fn slot_count(self) -> usize {
        self.chunk.meta.len()
    }
    pub fn data(self) -> &'a [Option<CompactArc<[Value]>>] {
        &self.chunk.data
    }
    pub fn meta(self) -> &'a [ArenaRowMeta] {
        &self.chunk.meta
    }
    pub fn row_first_lsn(self) -> Option<NonZeroU64> {
        self.chunk.lsn_minima.first()
    }
    pub fn iter(
        self,
    ) -> impl Iterator<Item = (ArenaId, &'a ArenaRowMeta, &'a CompactArc<[Value]>)> + 'a {
        self.chunk
            .meta
            .iter()
            .zip(&self.chunk.data)
            .enumerate()
            .filter_map(move |(slot, (meta, data))| {
                data.as_ref().map(|data| {
                    (
                        ArenaId::from_parts(self.chunk.id, slot)
                            .expect("allocated arena address is valid"),
                        meta,
                        data,
                    )
                })
            })
    }
}

pub struct ArenaReadGuard<'a> {
    inner: parking_lot::RwLockReadGuard<'a, ArenaInner>,
}

impl ArenaReadGuard<'_> {
    pub fn get(&self, id: ArenaId) -> Option<&CompactArc<[Value]>> {
        self.inner.get(id).map(|(_, data)| data)
    }
    pub fn get_meta(&self, id: ArenaId) -> Option<&ArenaRowMeta> {
        self.inner.get(id).map(|(meta, _)| meta)
    }
    pub fn get_entry(&self, id: ArenaId) -> Option<(&ArenaRowMeta, &CompactArc<[Value]>)> {
        self.inner.get(id)
    }
    /// Reuse the directory position while reading nearby rows. The guard
    /// prevents directory mutation; usize::MAX is an empty cache.
    pub fn get_entry_cached(
        &self,
        id: ArenaId,
        cache: &mut usize,
    ) -> Option<(&ArenaRowMeta, &CompactArc<[Value]>)> {
        if self
            .inner
            .chunks
            .get(*cache)
            .is_none_or(|chunk| chunk.id != id.chunk_id())
        {
            *cache = self.inner.position(id.chunk_id())?;
        }
        self.inner.chunks[*cache].get(id.slot())
    }
    pub fn chunks(&self) -> impl DoubleEndedIterator<Item = ArenaChunkRef<'_>> + ExactSizeIterator {
        self.inner
            .chunks
            .iter()
            .map(|chunk| ArenaChunkRef { chunk })
    }
    pub fn iter(&self) -> impl Iterator<Item = (ArenaId, &ArenaRowMeta, &CompactArc<[Value]>)> {
        self.chunks().flat_map(ArenaChunkRef::iter)
    }
    pub fn len(&self) -> usize {
        self.inner.occupied
    }
    pub fn slot_count(&self) -> usize {
        self.inner.slot_count
    }
    pub fn is_empty(&self) -> bool {
        self.inner.occupied == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lsn(value: u64) -> Option<NonZeroU64> {
        NonZeroU64::new(value)
    }

    fn payload(value: i64) -> CompactArc<[Value]> {
        CompactArc::from(vec![Value::Integer(value)])
    }

    #[test]
    fn structural_charges_follow_capacity_and_the_last_receipt_owner() {
        let root = MemoryAccount::new();
        let baseline = root.snapshot().accounted_bytes;
        let origin = root.child();
        let arena = RowArena::with_account(0, &origin);
        assert_eq!(root.snapshot().retained_bytes, arena.capacity_bytes());
        let data = payload(7); // Payload accounting belongs to hot ingress.
        let mut ids = Vec::new();
        for row in 0..=LSN_LEAF_ROWS {
            ids.push(arena.insert_arc(row as i64, 1, data.clone(), None).unwrap());
        }
        assert_eq!(root.snapshot().retained_bytes, arena.capacity_bytes());
        arena
            .update_at(ids[0], 0, 2, data.clone(), lsn(10))
            .unwrap();
        arena
            .update_at(
                ids[LSN_LEAF_ROWS],
                LSN_LEAF_ROWS as i64,
                2,
                data.clone(),
                lsn(20),
            )
            .unwrap();
        assert_eq!(root.snapshot().retained_bytes, arena.capacity_bytes());
        let pin = arena.pin_chunk_lsn(ids[0].chunk_id()).unwrap().unwrap();
        let more = arena
            .pin_lsn(ids[0].chunk_id(), NonZeroU64::new(30).unwrap())
            .unwrap();
        assert_eq!(root.snapshot().retained_bytes, arena.capacity_bytes());
        drop(more);
        assert_eq!(root.snapshot().retained_bytes, arena.capacity_bytes());
        arena.freeze_active().unwrap();
        arena.clear_all();
        assert_eq!(root.snapshot().retained_bytes, arena.capacity_bytes());
        let receipt_capacity = arena.capacity().receipt_bytes;
        drop(origin);
        drop(arena);
        assert_eq!(root.snapshot().retained_bytes, receipt_capacity);
        drop(pin);
        assert_eq!(root.snapshot().retained_bytes, 0);
        assert_eq!(root.snapshot().accounted_bytes, baseline);
    }

    #[test]
    fn lsn_tree_growth_accounts_the_simultaneously_live_buffers() {
        let root = MemoryAccount::new();
        // Preallocate the parallel slot arrays so only the LSN tree grows.
        let arena = RowArena::with_account(LSN_LEAF_ROWS + 1, &root);
        let data = payload(1);
        let ids: Vec<_> = (0..=LSN_LEAF_ROWS)
            .map(|slot| {
                arena
                    .insert_arc(slot as i64, 1, data.clone(), None)
                    .unwrap()
            })
            .collect();
        arena
            .update_at(ids[0], 0, 2, data.clone(), lsn(10))
            .unwrap();
        let before = root.snapshot().accounted_bytes;
        let old_tree_bytes = arena.capacity().lsn_tree_bytes;
        assert_eq!(old_tree_bytes, 16);
        arena
            .update_at(ids[LSN_LEAF_ROWS], LSN_LEAF_ROWS as i64, 2, data, lsn(20))
            .unwrap();
        let new_tree_bytes = arena.capacity().lsn_tree_bytes;
        assert_eq!(new_tree_bytes, 32);
        assert_eq!(
            root.snapshot().peak_accounted_bytes,
            before + new_tree_bytes
        );
        assert_eq!(root.snapshot().retained_bytes, arena.capacity_bytes());
    }

    #[test]
    fn directory_growth_accounts_the_simultaneously_live_buffers() {
        let root = MemoryAccount::new();
        let arena = RowArena::with_account(0, &root);
        let mut inner = arena.inner.write();
        inner.reserve_directory_slot().unwrap();
        let count = inner.chunks.capacity();
        // Empty test chunks isolate the directory's physical allocations.
        for value in 0..count {
            inner
                .chunks
                .push(Chunk::new(ChunkId::new(value as u64).unwrap(), Some(&root)));
        }
        let before = root.snapshot().accounted_bytes;
        inner.reserve_directory_slot().unwrap();
        let new_directory_bytes = inner.chunks.capacity() * std::mem::size_of::<Chunk>();
        assert_eq!(
            root.snapshot().peak_accounted_bytes,
            before + new_directory_bytes
        );
        drop(inner);
        assert_eq!(root.snapshot().retained_bytes, arena.capacity_bytes());
    }

    #[test]
    fn one_receipt_promotes_and_demotes_without_losing_identity_or_charge() {
        let root = MemoryAccount::new();
        let arena = RowArena::with_account(0, &root);
        let baseline = root.snapshot().retained_bytes;
        let first_chunk = ChunkId::new(3).unwrap();
        let other_chunk = ChunkId::new(9).unwrap();
        for _ in 0..32 {
            let single = arena
                .pin_lsn(first_chunk, NonZeroU64::new(20).unwrap())
                .unwrap();
            assert_eq!(root.snapshot().retained_bytes, baseline);
            assert_eq!(arena.chunk_first_lsn(first_chunk), lsn(20));
            assert_eq!(arena.chunk_first_lsn(other_chunk), None);
            drop(single);
            assert_eq!(arena.first_lsn(), None);
            assert_eq!(root.snapshot().retained_bytes, baseline);
        }
        let first = arena
            .pin_lsn(first_chunk, NonZeroU64::new(20).unwrap())
            .unwrap();
        let second = arena
            .pin_lsn(first_chunk, NonZeroU64::new(10).unwrap())
            .unwrap();
        assert!(root.snapshot().retained_bytes > baseline);
        assert_eq!(root.snapshot().retained_bytes, arena.capacity_bytes());
        assert_eq!(arena.first_lsn(), lsn(10));
        drop(first);
        assert_eq!(root.snapshot().retained_bytes, baseline);
        assert_eq!(arena.first_lsn(), lsn(10));
        let third = arena
            .pin_lsn(other_chunk, NonZeroU64::new(5).unwrap())
            .unwrap();
        assert_eq!(arena.chunk_first_lsn(first_chunk), lsn(10));
        assert_eq!(arena.chunk_first_lsn(other_chunk), lsn(5));
        drop(second);
        assert_eq!(root.snapshot().retained_bytes, baseline);
        assert_eq!(arena.chunk_first_lsn(first_chunk), None);
        assert_eq!(arena.first_lsn(), lsn(5));
        // Token exhaustion is checked before changing the existing inline pin.
        arena.receipts.lock().next_token = u64::MAX;
        assert!(arena
            .pin_lsn(first_chunk, NonZeroU64::new(1).unwrap())
            .is_err());
        assert_eq!(arena.first_lsn(), lsn(5));
        assert_eq!(root.snapshot().retained_bytes, baseline);
        drop(third);
        assert_eq!(arena.first_lsn(), None);
        assert_eq!(root.snapshot().retained_bytes, baseline);
    }

    #[test]
    fn partial_receipt_promotion_failure_preserves_the_original_pin() {
        let root = MemoryAccount::new();
        let arena = RowArena::with_account(0, &root);
        let baseline = root.snapshot().retained_bytes;
        let chunk = ChunkId::new(7).unwrap();
        let first = arena.pin_lsn(chunk, NonZeroU64::new(10).unwrap()).unwrap();
        let token = first.token;
        arena.receipts.lock().fail_promotion_after_positions = true;
        assert!(arena.pin_lsn(chunk, NonZeroU64::new(5).unwrap()).is_err());
        assert_eq!(arena.first_lsn(), lsn(10));
        assert_eq!(arena.receipts.lock().next_token, token);
        // The first hash allocation succeeded before the injected second
        // allocation failure. Its retained bytes must not disappear from the ledger.
        assert!(root.snapshot().retained_bytes > baseline);
        assert_eq!(root.snapshot().retained_bytes, arena.capacity_bytes());
        let second = arena.pin_lsn(chunk, NonZeroU64::new(5).unwrap()).unwrap();
        assert_eq!(second.token, token + 1);
        drop(second);
        assert_eq!(arena.first_lsn(), lsn(10));
        assert_eq!(root.snapshot().retained_bytes, baseline);
        // Failure followed by release (instead of retry) frees partial capacity too.
        arena.receipts.lock().fail_promotion_after_positions = true;
        assert!(arena.pin_lsn(chunk, NonZeroU64::new(5).unwrap()).is_err());
        drop(first);
        assert_eq!(arena.first_lsn(), None);
        assert_eq!(root.snapshot().retained_bytes, baseline);
    }

    #[test]
    fn receipt_map_replacement_charges_exact_old_and_new_buckets() {
        let root = MemoryAccount::new();
        let mut charge = Some(MemoryCharge::new(&root, 0));
        let mut map = ReceiptPositions::new();
        reserve_receipt_map(&mut map, 2, &mut charge).unwrap();
        for token in 0..map.capacity() {
            map.insert(token as u64, (ChunkId::new(7).unwrap(), token));
        }
        let old_bytes = map.allocation_size();
        let before = root.snapshot();
        reserve_receipt_map(&mut map, 1, &mut charge).unwrap();
        let new_bytes = map.allocation_size();
        assert!(new_bytes > old_bytes);
        assert_eq!(root.snapshot().retained_bytes, new_bytes);
        assert_eq!(
            root.snapshot().peak_accounted_bytes,
            before.accounted_bytes + new_bytes
        );
        for (token, &(chunk, position)) in &map {
            assert_eq!(chunk, ChunkId::new(7).unwrap());
            assert_eq!(*token, position as u64);
        }
        let before_failure = root.snapshot();
        assert!(reserve_receipt_map(&mut map, usize::MAX, &mut charge).is_err());
        assert_eq!(root.snapshot(), before_failure);
        drop(map);
        drop(charge);
        assert_eq!(root.snapshot().retained_bytes, 0);
    }

    #[test]
    fn receipt_heap_replacement_charges_exact_old_and_new_buffers() {
        let root = MemoryAccount::new();
        let mut charge = Some(MemoryCharge::new(&root, 0));
        let mut heap = ReceiptHeap::new();
        heap.push(Receipt {
            token: 1,
            lsn: lsn(10).unwrap(),
        });
        ReceiptRegistry::reserve_heap(&mut heap, 1, &mut charge).unwrap();
        heap.push(Receipt {
            token: 2,
            lsn: lsn(20).unwrap(),
        });
        let before = root.snapshot();
        ReceiptRegistry::reserve_heap(&mut heap, 1, &mut charge).unwrap();
        let new_bytes = ReceiptRegistry::heap_bytes(&heap);
        assert_eq!(root.snapshot().retained_bytes, new_bytes);
        assert_eq!(
            root.snapshot().peak_accounted_bytes,
            before.accounted_bytes + new_bytes
        );
        assert_eq!(heap[0].token, 1);
        assert_eq!(heap[1].token, 2);
        let before_failure = root.snapshot();
        assert!(ReceiptRegistry::reserve_heap(&mut heap, usize::MAX, &mut charge).is_err());
        assert_eq!(root.snapshot(), before_failure);
        drop(heap);
        drop(charge);
        assert_eq!(root.snapshot().retained_bytes, 0);
    }

    #[test]
    fn addresses_are_checked_and_keep_the_nonzero_niche() {
        let first = ArenaId::from_parts(ChunkId::new(0).unwrap(), 0).unwrap();
        assert_eq!(first.as_nonzero().get(), 1);
        assert_eq!(first.chunk_id().get(), 0);
        assert_eq!(first.slot(), 0);
        assert!(ChunkId::new(MAX_CHUNK_ID + 1).is_err());
        let last_chunk = ChunkId::new(MAX_CHUNK_ID).unwrap();
        assert!(ArenaId::from_parts(last_chunk, ARENA_CHUNK_ROWS - 1).is_err());
        assert!(ArenaId::from_parts(last_chunk, ARENA_CHUNK_ROWS).is_err());
        let last = ArenaId::from_parts(last_chunk, ARENA_CHUNK_ROWS - 2).unwrap();
        assert_eq!(last.as_nonzero().get(), u64::MAX);
        assert_eq!(ArenaId::from_nonzero(last.as_nonzero()), last);

        let arena = RowArena::new();
        arena.inner.write().next_chunk_id = MAX_CHUNK_ID + 1;
        let capacity = arena.capacity();
        assert!(arena.insert_arc(0, 1, payload(0), lsn(1)).is_err());
        assert!(arena.is_empty());
        assert_eq!(arena.bytes(), 0);
        assert_eq!(arena.capacity(), capacity);
    }

    #[test]
    fn real_chunk_boundary_releases_the_old_allocation() {
        let arena = RowArena::new();
        let data = payload(7);
        let first = arena.insert_arc(0, 1, data.clone(), None).unwrap();
        for row_id in 1..=ARENA_CHUNK_ROWS {
            let id = arena
                .insert_arc(row_id as i64, 1, data.clone(), None)
                .unwrap();
            assert_eq!(id.chunk_id().get(), (row_id / ARENA_CHUNK_ROWS) as u64);
            assert_eq!(id.slot(), row_id % ARENA_CHUNK_ROWS);
        }
        let next = ArenaId::from_parts(ChunkId::new(1).unwrap(), 0).unwrap();
        assert_eq!(arena.len(), ARENA_CHUNK_ROWS + 1);
        let before = arena.capacity();
        {
            let guard = arena.read_guard();
            let chunks: Vec<_> = guard.chunks().collect();
            assert_eq!(chunks.len(), 2);
            assert!(chunks[0].is_frozen());
            assert!(!chunks[1].is_frozen());
            assert_eq!(chunks[0].slot_count(), ARENA_CHUNK_ROWS);
            assert_eq!(guard.get_meta(first).unwrap().row_id, 0);
            assert_eq!(
                guard.get_meta(next).unwrap().row_id,
                ARENA_CHUNK_ROWS as i64
            );
        }
        for begin in (0..ARENA_CHUNK_ROWS).step_by(2000) {
            let identities: Vec<_> = (begin..(begin + 2000).min(ARENA_CHUNK_ROWS))
                .map(|slot| {
                    (
                        ArenaId::from_parts(first.chunk_id(), slot).unwrap(),
                        slot as i64,
                        1,
                    )
                })
                .collect();
            assert_eq!(arena.clear_batch(&identities), identities.len());
        }
        let after = arena.capacity();
        assert!(after.metadata_bytes < before.metadata_bytes / 100);
        assert!(after.payload_slots_bytes < before.payload_slots_bytes / 100);
        assert_eq!(arena.len(), 1);
        assert!(arena.get_arc(first).is_none());
        assert!(arena.get_arc(next).is_some());
        assert_eq!(arena.bytes(), row_bytes(&data));
    }

    #[test]
    fn active_reuse_and_stale_clears_preserve_identity_and_bytes() {
        let arena = RowArena::new();
        let zero = arena.insert_arc(0, 1, payload(0), None).unwrap();
        let one = arena.insert_arc(1, 1, payload(1), None).unwrap();
        let two = arena.insert_arc(2, 1, payload(2), None).unwrap();
        assert!(!arena.clear_at(one, 7, 1));
        assert!(!arena.clear_at(one, 1, 2));
        assert!(arena.clear_at(one, 1, 1));
        assert!(!arena.clear_at(one, 1, 1));
        assert_eq!(arena.clear_batch(&[(two, 2, 1), (two, 2, 1)]), 1);
        let replacement = arena.insert_arc(9, 2, payload(9), None).unwrap();
        assert_eq!(replacement, two);
        assert!(!arena.clear_at(two, 2, 1));
        assert_eq!(arena.len(), 2);
        assert_eq!(arena.bytes(), row_bytes(&payload(0)) * 2);
        assert!(arena
            .update_at(replacement, 9, 3, payload(99), lsn(30))
            .unwrap());
        assert!(!arena.clear_at(replacement, 9, 2));
        assert!(arena.mark_deleted(replacement, 9, 4, lsn(40)).unwrap());
        assert!(!arena.clear_at(replacement, 9, 3));
        assert_eq!(
            arena
                .get_meta_and_arc(replacement)
                .unwrap()
                .0
                .mutation_txn_id(),
            4
        );
        assert!(arena.clear_at(replacement, 9, 4));
        assert!(arena.clear_at(zero, 0, 1));
        assert_eq!(arena.bytes(), 0);
        assert_eq!(arena.slot_count(), 0);
        assert_eq!(arena.capacity().metadata_bytes, 0);
        let later = arena.insert_arc(0, 5, payload(0), None).unwrap();
        assert_ne!(later.chunk_id(), zero.chunk_id());
        assert!(!arena.clear_at(zero, 0, 1));
        assert!(arena.insert_arc(1, 0, payload(1), None).is_err());
        assert_eq!(arena.len(), 1);
    }

    #[test]
    fn freezing_is_lazy_and_never_reuses_captured_slots() {
        let arena = RowArena::with_capacity(usize::MAX);
        let empty_capacity = arena.capacity();
        assert_eq!(empty_capacity.directory_bytes, 0);
        assert_eq!(empty_capacity.metadata_bytes, 0);
        assert_eq!(arena.freeze_active().unwrap(), None);
        assert_eq!(arena.capacity(), empty_capacity);
        let first = arena.insert_arc(99, 1, payload(99), None).unwrap();
        let hole = arena.insert_arc(-2, 1, payload(-2), None).unwrap();
        assert!(arena.clear_at(hole, -2, 1));
        assert_eq!(arena.inner.read().free_head, Some(hole.slot() as u32));
        assert_eq!(arena.capacity().free_list_bytes, 0);
        let row_capacity = arena.capacity().metadata_bytes;
        assert_eq!(
            row_capacity,
            ARENA_CHUNK_ROWS * std::mem::size_of::<ArenaRowMeta>()
        );
        assert_eq!(arena.freeze_active().unwrap(), Some(first.chunk_id()));
        assert_eq!(arena.capacity().free_list_bytes, 0);
        assert_eq!(arena.inner.read().free_head, None);
        assert_eq!(arena.capacity().metadata_bytes, row_capacity);
        assert!(arena.update_at(first, 99, 2, payload(100), None).unwrap());
        let next = arena.insert_arc(0, 2, payload(0), None).unwrap();
        assert_ne!(next.chunk_id(), first.chunk_id());
        assert_eq!(next.slot(), 0);
        assert!(arena.get_arc(hole).is_none());
        assert!(arena.clear_at(first, 99, 2));
        arena.clear_all();
        assert_eq!(arena.capacity().metadata_bytes, 0);
        let after_clear = arena.insert_arc(0, 3, payload(0), None).unwrap();
        assert!(after_clear.chunk_id() > next.chunk_id());
    }

    #[test]
    fn lsn_minima_follow_updates_delete_and_undo_across_leaves() {
        let arena = RowArena::new();
        let mut ids = Vec::new();
        let data = payload(1);
        for slot in 0..=LSN_LEAF_ROWS {
            let source = match slot {
                0 => lsn(10),
                1 => lsn(20),
                LSN_LEAF_ROWS => lsn(30),
                _ => None,
            };
            ids.push(
                arena
                    .insert_arc(slot as i64, 1, data.clone(), source)
                    .unwrap(),
            );
        }
        let chunk = ids[0].chunk_id();
        arena.freeze_active().unwrap();
        let pin = arena.pin_chunk_lsn(chunk).unwrap().unwrap();
        assert_eq!(arena.first_lsn(), lsn(10));
        arena
            .update_at(ids[0], 0, 2, data.clone(), lsn(100))
            .unwrap();
        arena.mark_deleted(ids[1], 1, 2, lsn(200)).unwrap();
        // Published changes raise the row minimum, but cannot release old WAL.
        assert_eq!(
            arena.read_guard().chunks().next().unwrap().row_first_lsn(),
            lsn(30)
        );
        assert_eq!(arena.first_lsn(), lsn(10));
        arena
            .update_at(ids[0], 0, 1, data.clone(), lsn(10))
            .unwrap();
        arena
            .update_at(ids[1], 1, 1, data.clone(), lsn(20))
            .unwrap();
        drop(pin);
        assert_eq!(arena.first_lsn(), lsn(10));
        assert!(arena.clear_at(ids[0], 0, 1));
        assert_eq!(arena.first_lsn(), lsn(20));
        assert!(arena.clear_at(ids[1], 1, 1));
        assert_eq!(arena.first_lsn(), lsn(30));
        // A lower value in an already allocated leaf updates the path directly.
        arena.update_at(ids[5], 5, 3, data, lsn(5)).unwrap();
        assert_eq!(arena.first_lsn(), lsn(5));
        assert_eq!(
            arena.capacity().lsn_tree_bytes,
            4 * std::mem::size_of::<Option<NonZeroU64>>()
        );
    }

    #[test]
    fn truncate_detaches_charges_without_reusing_chunk_ids_or_dropping_receipts() {
        let account = MemoryAccount::new();
        let baseline = account.snapshot().accounted_bytes;
        let arena = RowArena::with_account(0, &account);
        let first = arena.insert_arc(1, 1, payload(1), lsn(10)).unwrap();
        let pin = arena.pin_chunk_lsn(first.chunk_id()).unwrap().unwrap();
        let retained = account.snapshot().retained_bytes;
        let retired = arena.take_all_for_truncate();
        assert!(arena.is_empty());
        assert!(arena.read_guard().get(first).is_none());
        assert_eq!(arena.capacity().metadata_bytes, 0);
        assert_eq!(account.snapshot().retained_bytes, retained);
        assert_eq!(arena.first_lsn(), lsn(10));

        let second = arena.insert_arc(1, 2, payload(2), lsn(20)).unwrap();
        assert!(second.chunk_id() > first.chunk_id());
        assert!(arena.read_guard().get(first).is_none());
        let both = account.snapshot().retained_bytes;
        drop(retired);
        assert!(account.snapshot().retained_bytes < both);
        assert_eq!(arena.first_lsn(), lsn(10));
        drop(pin);
        assert_eq!(arena.first_lsn(), lsn(20));
        drop(arena);
        assert_eq!(account.snapshot().accounted_bytes, baseline);
    }

    #[test]
    fn receipts_outlive_empty_chunks_and_release_only_their_own_pin() {
        let arena = RowArena::new();
        let id = arena.insert_arc(1, 1, payload(1), lsn(10)).unwrap();
        let pin = arena.pin_chunk_lsn(id.chunk_id()).unwrap().unwrap();
        let later = arena
            .pin_lsn(id.chunk_id(), NonZeroU64::new(20).unwrap())
            .unwrap();
        assert!(arena.clear_at(id, 1, 1));
        assert_eq!(arena.capacity().metadata_bytes, 0);
        assert_eq!(arena.capacity().lsn_tree_bytes, 0);
        assert_eq!(arena.chunk_first_lsn(id.chunk_id()), lsn(10));
        arena.clear_all();
        assert_eq!(arena.first_lsn(), lsn(10));
        drop(pin);
        assert_eq!(arena.first_lsn(), lsn(20));
        assert!(arena.pin_chunk_lsn(id.chunk_id()).unwrap().is_none());
        let copied = arena
            .pin_lsn(id.chunk_id(), NonZeroU64::new(20).unwrap())
            .unwrap();
        drop(later);
        assert_eq!(arena.first_lsn(), lsn(20));
        drop(copied);
        assert_eq!(arena.first_lsn(), None);
        assert!(arena.pin_chunk_lsn(id.chunk_id()).unwrap().is_none());
    }

    #[test]
    fn overlapping_publications_do_not_inherit_predecessor_receipts() {
        let arena = RowArena::new();
        let id = arena.insert_arc(1, 1, payload(1), lsn(10)).unwrap();
        let first = arena.pin_chunk_lsn(id.chunk_id()).unwrap().unwrap();
        arena.update_at(id, 1, 2, payload(2), lsn(20)).unwrap();
        let second = arena.pin_chunk_lsn(id.chunk_id()).unwrap().unwrap();
        arena.update_at(id, 1, 3, payload(3), lsn(30)).unwrap();
        assert_eq!(arena.first_lsn(), lsn(10));
        drop(first);
        assert_eq!(arena.first_lsn(), lsn(20));
        let third = arena.pin_chunk_lsn(id.chunk_id()).unwrap().unwrap();
        drop(second);
        assert_eq!(arena.first_lsn(), lsn(30));
        drop(third);
        assert_eq!(arena.first_lsn(), lsn(30));
    }

    #[test]
    fn indexed_receipt_heaps_release_arbitrary_tokens_across_chunks() {
        let arena = RowArena::new();
        let initial_capacity = arena.capacity().receipt_bytes;
        let mut pins = Vec::new();
        for value in 0..512 {
            // Exercise both a large same-chunk heap and one inline pin per chunk.
            let chunk = ChunkId::new(if value < 256 { 0 } else { value - 255 }).unwrap();
            let lsn = NonZeroU64::new((value * 137) % 512 + 1).unwrap();
            pins.push((chunk, lsn, Some(arena.pin_lsn(chunk, lsn).unwrap())));
        }
        for ordinal in 0..512 {
            let index = (ordinal * 37) % 512;
            let chunk = pins[index].0;
            drop(pins[index].2.take());
            let global = pins
                .iter()
                .filter(|(_, _, pin)| pin.is_some())
                .map(|(_, lsn, _)| *lsn)
                .min();
            let local = pins
                .iter()
                .filter(|(id, _, pin)| *id == chunk && pin.is_some())
                .map(|(_, lsn, _)| *lsn)
                .min();
            assert_eq!(arena.first_lsn(), global);
            assert_eq!(arena.chunk_first_lsn(chunk), local);
        }
        assert_eq!(arena.capacity().receipt_bytes, initial_capacity);
        let next_token = arena.receipts.lock().next_token;
        let id = ChunkId::new(0).unwrap();
        let pin = arena.pin_lsn(id, NonZeroU64::new(1).unwrap()).unwrap();
        assert!(pin.token > next_token);
        arena.receipts.lock().next_token = u64::MAX;
        let capacity = arena.capacity();
        assert!(arena.pin_lsn(id, NonZeroU64::new(2).unwrap()).is_err());
        assert_eq!(arena.first_lsn(), lsn(1));
        assert_eq!(arena.capacity(), capacity);
    }

    #[test]
    fn sparse_read_cache_and_iteration_validate_real_chunk_addresses() {
        let arena = RowArena::new();
        let first = arena.insert_arc(99, 1, payload(99), None).unwrap();
        arena.freeze_active().unwrap();
        let removed = arena.insert_arc(-2, 1, payload(-2), None).unwrap();
        arena.freeze_active().unwrap();
        let last = arena.insert_arc(0, 1, payload(0), None).unwrap();
        assert!(arena.clear_at(removed, -2, 1));
        let guard = arena.read_guard();
        let mut cache = usize::MAX;
        assert_eq!(
            guard.get_entry_cached(first, &mut cache).unwrap().0.row_id,
            99
        );
        assert_eq!(cache, 0);
        assert_eq!(
            guard.get_entry_cached(last, &mut cache).unwrap().0.row_id,
            0
        );
        assert_eq!(cache, 1);
        assert!(guard.get_entry_cached(removed, &mut cache).is_none());
        assert_eq!(
            guard.get_entry_cached(first, &mut cache).unwrap().0.row_id,
            99
        );
        assert_eq!(
            guard
                .iter()
                .map(|(id, meta, _)| (id, meta.row_id))
                .collect::<Vec<_>>(),
            vec![(first, 99), (last, 0)]
        );
        let shared = guard.get(first).unwrap().clone();
        drop(guard);
        arena.clear_all();
        assert_eq!(shared.as_ref(), &[Value::Integer(99)]);
    }
}
