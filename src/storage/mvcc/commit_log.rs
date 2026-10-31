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

//! The order commits complete in, with their times: the points a read AS OF
//! a time or a transaction reads at. One writer appends under the registry's
//! transactions lock; readers search it without a lock.

use crate::common::time_compat::{SystemTime, UNIX_EPOCH};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::OnceLock;

use crate::core::{Error, Result};

/// Slots the log keeps at most: about 10 MiB once that many commits ran
pub const COMMIT_LOG_CAPACITY: usize = 1 << 18;

/// Slots allocated together as the log grows: 40 KiB
const CHUNK_SLOTS: usize = 1 << 10;

/// One completed commit; every field is an atomic so a reader racing an
/// overwrite reads values, never a torn field
#[derive(Default)]
struct Slot {
    time: AtomicI64,
    txn_id: AtomicI64,
    commit_seq: AtomicI64,
    completed_through: AtomicI64,
    assigned_through: AtomicI64,
}

/// What a completion records, read under the transactions lock
pub(crate) struct Completion {
    pub txn_id: i64,
    pub commit_seq: i64,
    pub completed_through: i64,
    pub assigned_through: i64,
}

/// A statement's bounds for every point it reads: the time it treats as
/// now and the next ordinal, taken once at its start
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HistoricalCut {
    now: i64,
    head: u64,
    /// The open point as it stood then: it moves past history events
    /// until the first commit
    open_seq: i64,
    open_time: i64,
    /// What had completed at the cut: the newest point sees as much, since
    /// no commit completed between it and the cut
    completed_through: i64,
    /// The ordinal a completion last went unrecorded at: the newest point
    /// is then not extended to the cut
    skipped_at: u64,
}

/// A point K: what had completed when the K-th commit did
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReadPoint {
    k: u64,
    completed_through: i64,
    /// Sequences above `completed_through` that had completed, sorted
    later: Vec<i64>,
}

impl ReadPoint {
    /// Whether a version stamped `commit_seq` had completed at this point
    #[inline]
    pub fn sees(&self, commit_seq: i64) -> bool {
        commit_seq <= self.completed_through || self.later.binary_search(&commit_seq).is_ok()
    }

    /// Every commit at or below this had completed at the point
    pub fn completed_through(&self) -> i64 {
        self.completed_through
    }
}

pub(crate) struct CommitLog {
    /// Allocated one chunk at a time, outside every lock, ahead of use;
    /// reused in turn once all are allocated
    chunks: Box<[OnceLock<Box<[Slot]>>]>,
    chunk_slots: usize,
    capacity: usize,
    /// Set by the first commit that writes; before it only history events
    /// complete, and they move the open point
    prepared: AtomicBool,
    /// The ordinal the next completion takes; the open point is 0
    head: AtomicU64,
    /// The oldest ordinal still held
    tail: AtomicU64,
    /// Every point must have completed through this: raised by eviction
    bound: AtomicI64,
    /// The largest time handed out or promised; written under the lock
    time_floor: AtomicI64,
    open_seq: AtomicI64,
    open_time: AtomicI64,
    /// The ordinal a completion last found no prepared slot at, 0 if none
    skipped_at: AtomicU64,
}

/// Nanoseconds since the epoch, as AS OF TIMESTAMP reads them
pub(crate) fn wall_clock() -> i64 {
    #[cfg(feature = "test-failpoints")]
    let shift = crate::test_failpoints::commit_clock_shift();
    #[cfg(not(feature = "test-failpoints"))]
    let shift = 0;
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as i64)
        + shift
}

impl CommitLog {
    pub fn new(capacity: usize) -> Self {
        let now = wall_clock();
        let chunk_slots = CHUNK_SLOTS.min(capacity).max(1);
        let chunks = capacity.div_ceil(chunk_slots);
        Self {
            chunks: (0..chunks).map(|_| OnceLock::new()).collect(),
            chunk_slots,
            capacity: chunks * chunk_slots,
            prepared: AtomicBool::new(false),
            head: AtomicU64::new(1),
            tail: AtomicU64::new(1),
            bound: AtomicI64::new(0),
            time_floor: AtomicI64::new(now),
            open_seq: AtomicI64::new(0),
            open_time: AtomicI64::new(now),
            skipped_at: AtomicU64::new(0),
        }
    }

    /// Makes sure the chunk the next completion lands in exists, and the
    /// following one once this one is half used; called outside every lock
    /// before a commit that writes takes its sequence
    pub fn prepare(&self) {
        if !self.prepared.load(Ordering::Relaxed) {
            self.prepared.store(true, Ordering::Release);
        }
        let head = self.head.load(Ordering::Acquire);
        self.ensure_chunk(head);
        if head % self.chunk_slots as u64 >= self.chunk_slots as u64 / 2 {
            self.ensure_chunk(head + self.chunk_slots as u64);
        }
    }

    /// Allocates and writes the chunk ordinal `k` falls in, once
    fn ensure_chunk(&self, k: u64) {
        self.chunks[self.chunk_of(k)].get_or_init(|| {
            (0..self.chunk_slots)
                .map(|_| Slot::default())
                .collect::<Vec<_>>()
                .into_boxed_slice()
        });
    }

    fn chunk_of(&self, k: u64) -> usize {
        ((k / self.chunk_slots as u64) % self.chunks.len() as u64) as usize
    }

    /// The point a recovered database opens at: every sequence at or below
    /// `seq` is the recovered state. Called before any commit
    pub fn open_at(&self, seq: i64, time: i64) {
        self.open_seq.store(seq, Ordering::Release);
        self.open_time.store(time, Ordering::Release);
        self.time_floor.store(time, Ordering::Release);
        self.bound.fetch_max(seq, Ordering::Release);
    }

    /// Appends a completion. Called under the transactions lock
    pub fn append(&self, completion: Completion, clock: i64) {
        let time = clock.max(self.time_floor.load(Ordering::Relaxed) + 1);
        self.time_floor.store(time, Ordering::Release);
        if !self.prepared.load(Ordering::Acquire) {
            // Before the first commit only history events complete: the open
            // point moves past them
            self.open_seq
                .store(completion.completed_through, Ordering::Release);
            self.open_time.store(time, Ordering::Release);
            return;
        }
        let k = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Relaxed);
        let Some(slot) = self.slot(k) else {
            // Not prepared ahead: no point that needs this entry is kept, and
            // the entry before it no longer reaches to the next cut
            self.bound
                .fetch_max(completion.commit_seq, Ordering::Release);
            self.skipped_at.store(k, Ordering::Release);
            return;
        };
        if k - tail >= self.capacity as u64 {
            let evicted = slot.commit_seq.load(Ordering::Relaxed);
            self.bound.fetch_max(evicted, Ordering::Release);
            self.tail.store(tail + 1, Ordering::Release);
        }
        slot.time.store(time, Ordering::Release);
        slot.txn_id.store(completion.txn_id, Ordering::Release);
        slot.commit_seq
            .store(completion.commit_seq, Ordering::Release);
        slot.completed_through
            .store(completion.completed_through, Ordering::Release);
        slot.assigned_through
            .store(completion.assigned_through, Ordering::Release);
        self.head.store(k + 1, Ordering::Release);
    }

    /// A statement's cut, `completed_through` read with it. Called under
    /// the transactions lock
    pub fn cut(&self, clock: i64, completed_through: i64) -> HistoricalCut {
        let now = clock.max(self.time_floor.load(Ordering::Relaxed));
        self.time_floor.store(now, Ordering::Release);
        HistoricalCut {
            now,
            head: self.head.load(Ordering::Acquire),
            open_seq: self.open_seq.load(Ordering::Acquire),
            open_time: self.open_time.load(Ordering::Acquire),
            completed_through,
            skipped_at: self.skipped_at.load(Ordering::Acquire),
        }
    }

    /// Every point must have completed through this
    pub fn bound(&self) -> i64 {
        self.bound.load(Ordering::Acquire)
    }

    /// The time of the oldest point that has completed through `seq` and
    /// the bound, None when only points still to come will
    pub fn oldest_time_through(&self, seq: i64) -> Option<i64> {
        let needed = seq.max(self.bound());
        let (tail, head) = (
            self.tail.load(Ordering::Acquire),
            self.head.load(Ordering::Acquire),
        );
        if tail == 1 && self.open_seq.load(Ordering::Acquire) >= needed {
            return Some(self.open_time.load(Ordering::Acquire));
        }
        let found = (tail..head).find_map(|k| {
            let slot = self.slot(k)?;
            (slot.completed_through.load(Ordering::Acquire) >= needed)
                .then(|| slot.time.load(Ordering::Acquire))
        });
        // A slot overwritten meanwhile only makes the answer later than true
        found.filter(|_| self.tail.load(Ordering::Acquire) == tail)
    }

    /// A point that sees exactly what no AS OF can still read
    pub fn unreadable_point(&self) -> ReadPoint {
        ReadPoint {
            k: 0,
            completed_through: self.bound().max(self.open_seq.load(Ordering::Acquire)),
            later: Vec::new(),
        }
    }

    /// The point AS OF TIMESTAMP `time` reads under `cut`
    pub fn point_at_time(&self, time: i64, cut: HistoricalCut) -> Result<ReadPoint> {
        if time > cut.now {
            return self.point(cut.head - 1, u64::MAX, &cut);
        }
        if time < cut.open_time {
            return Err(self.unavailable("before the oldest point the database keeps"));
        }
        let tail = self.tail.load(Ordering::Acquire);
        let (mut low, mut high) = (tail, cut.head);
        let mut touched = u64::MAX;
        while low < high {
            let mid = low + (high - low) / 2;
            touched = touched.min(mid);
            match self.kept(mid)?.time.load(Ordering::Acquire) <= time {
                true => low = mid + 1,
                false => high = mid,
            }
        }
        match low {
            // Every kept entry is later: the open point, unless entries
            // between it and the oldest kept one were evicted
            k if k == tail && tail > 1 => {
                Err(self.unavailable("older than the commit log reaches"))
            }
            k if k == tail => self.point(0, touched, &cut),
            k => self.point(k - 1, touched, &cut),
        }
    }

    /// The point right after transaction `txn_id` completed, None when no
    /// kept entry is its
    pub fn point_of_transaction(
        &self,
        txn_id: i64,
        cut: HistoricalCut,
    ) -> Result<Option<ReadPoint>> {
        let tail = self.tail.load(Ordering::Acquire);
        for k in (tail..cut.head).rev() {
            if self.kept(k)?.txn_id.load(Ordering::Acquire) == txn_id {
                return self.point(k, tail, &cut).map(Some);
            }
        }
        self.validate(tail, i64::MAX)?;
        Ok(None)
    }

    /// The point at ordinal `k`; `touched` is the smallest ordinal the
    /// search already read
    fn point(&self, k: u64, touched: u64, cut: &HistoricalCut) -> Result<ReadPoint> {
        let newest = k + 1 == cut.head && cut.skipped_at != cut.head;
        if k == 0 {
            let completed_through = match newest {
                true => cut.open_seq.max(cut.completed_through),
                false => cut.open_seq,
            };
            self.validate(touched, completed_through)?;
            return Ok(ReadPoint {
                k,
                completed_through,
                later: Vec::new(),
            });
        }
        let tail = self.tail.load(Ordering::Acquire);
        #[cfg(feature = "test-failpoints")]
        crate::test_failpoints::commit_log_slots_read();
        let slot = self.kept(k)?;
        let completed_through = slot.completed_through.load(Ordering::Acquire);
        let mut later = Vec::new();
        let mut j = k;
        loop {
            let entry = self.kept(j)?;
            if entry.assigned_through.load(Ordering::Acquire) <= completed_through {
                break;
            }
            let seq = entry.commit_seq.load(Ordering::Acquire);
            if seq > completed_through {
                later.push(seq);
            }
            if j == tail {
                break;
            }
            j -= 1;
        }
        // The newest point sees what had completed at the cut
        let completed_through = match newest {
            true => completed_through.max(cut.completed_through),
            false => completed_through,
        };
        later.retain(|&seq| seq > completed_through);
        later.sort_unstable();
        self.validate(touched.min(j), completed_through)?;
        Ok(ReadPoint {
            k,
            completed_through,
            later,
        })
    }

    /// The entries read from `touched` up are still the ones written for
    /// them, and a point that completed through `completed_through` needs
    /// no evicted entry
    fn validate(&self, touched: u64, completed_through: i64) -> Result<()> {
        if touched < self.tail.load(Ordering::Acquire) {
            return Err(self.unavailable("older than the commit log reaches"));
        }
        if completed_through < self.bound.load(Ordering::Acquire) {
            return Err(self.unavailable("older than the commit log reaches"));
        }
        Ok(())
    }

    /// The slot of ordinal `k`, None while its chunk is not allocated
    fn slot(&self, k: u64) -> Option<&Slot> {
        let chunk = self.chunks[self.chunk_of(k)].get()?;
        Some(&chunk[(k % self.chunk_slots as u64) as usize])
    }

    /// The slot of an ordinal a completion was appended at
    fn kept(&self, k: u64) -> Result<&Slot> {
        self.slot(k)
            .ok_or_else(|| self.unavailable("older than the commit log reaches"))
    }

    fn unavailable(&self, why: &str) -> Error {
        let oldest = match self.oldest_time_through(i64::MIN) {
            Some(nanos) => {
                let time = chrono::DateTime::from_timestamp_nanos(nanos);
                format!("the oldest point it can read is {time}")
            }
            None => "it can read no earlier point than the next commit".to_string(),
        };
        Error::invalid_argument(format!("AS OF point is {why}: {oldest}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn allocated(log: &CommitLog) -> usize {
        log.chunks
            .iter()
            .filter(|chunk| chunk.get().is_some())
            .count()
    }

    fn complete(log: &CommitLog, seq: i64) {
        let completion = Completion {
            txn_id: seq,
            commit_seq: seq,
            completed_through: seq,
            assigned_through: seq,
        };
        log.append(completion, wall_clock());
    }

    #[test]
    fn the_log_grows_a_chunk_at_a_time_ahead_of_use() {
        let log = CommitLog::new(COMMIT_LOG_CAPACITY);
        assert_eq!(allocated(&log), 0, "nothing before the first commit");
        // Ordinals start at 1: after these the next one is half way in
        let mut seq = 0;
        for _ in 1..CHUNK_SLOTS / 2 {
            log.prepare();
            seq += 1;
            complete(&log, seq);
        }
        assert_eq!(allocated(&log), 1, "a chunk not half used");
        log.prepare();
        assert_eq!(allocated(&log), 2, "the next one ready before it is needed");
        for _ in 0..CHUNK_SLOTS {
            log.prepare();
            seq += 1;
            complete(&log, seq);
        }
        assert_eq!(allocated(&log), 2, "the second chunk not half used");
        log.prepare();
        assert_eq!(allocated(&log), 3);
        assert_eq!(log.bound(), 0, "every completion found its slot");
    }

    #[test]
    fn a_completion_without_its_chunk_only_raises_the_bound() {
        let log = CommitLog::new(COMMIT_LOG_CAPACITY);
        log.prepare();
        for seq in 1..=CHUNK_SLOTS as i64 + 1 {
            complete(&log, seq);
        }
        assert_eq!(allocated(&log), 1);
        assert_eq!(log.bound(), CHUNK_SLOTS as i64 + 1);
        assert_eq!(log.head.load(Ordering::Acquire), CHUNK_SLOTS as u64);
    }

    #[test]
    fn an_unrecorded_completion_keeps_the_entry_before_it_from_reaching_the_cut() {
        let log = CommitLog::new(COMMIT_LOG_CAPACITY);
        log.prepare();
        let last = CHUNK_SLOTS as i64 + 1;
        for seq in 1..=last {
            complete(&log, seq);
        }
        let cut = log.cut(wall_clock(), last);
        assert!(
            log.point_at_time(i64::MAX, cut).is_err(),
            "the newest kept entry predates the unrecorded commits"
        );
    }
}
