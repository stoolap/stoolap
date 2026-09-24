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

//! The sorted row ids of one index key: inline up to `P` ids, then pages of
//! at most `P` ids under a directory keyed by each page's lower bound, so an
//! insert or a removal moves at most a page of ids.

use std::collections::btree_map;
use std::collections::BTreeMap;
use std::ops::Bound;

use crate::common::CompactVec;

/// Ids per page
pub const PAGE_IDS: usize = 512;

#[derive(Clone, Debug)]
pub enum IdList<const P: usize = PAGE_IDS> {
    Inline(CompactVec<i64>),
    Paged(Box<Paged>),
}

// The paged variant lives in the inline variant's niche: no wider than the
// list it replaces
const _: () = assert!(std::mem::size_of::<IdList>() == std::mem::size_of::<CompactVec<i64>>());

/// Pages are non-empty with room for `P` ids. A page holds the ids from its
/// key up to the next key, and the first page also those below its key, so
/// a removal never changes a key. A page under half full has no neighbour
/// it fits with.
#[derive(Debug)]
pub struct Paged {
    len: usize,
    pages: BTreeMap<i64, CompactVec<i64>>,
}

/// Keeps every page's capacity, which a later merge relies on to not grow
impl Clone for Paged {
    fn clone(&self) -> Self {
        let pages = self
            .pages
            .iter()
            .map(|(&key, page)| {
                let mut copy = CompactVec::with_capacity(page.capacity());
                copy.extend_copy(page);
                (key, copy)
            })
            .collect();
        Paged {
            len: self.len,
            pages,
        }
    }
}

impl<const P: usize> Default for IdList<P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const P: usize> IdList<P> {
    pub fn new() -> Self {
        IdList::Inline(CompactVec::new())
    }

    /// From strictly increasing ids
    pub fn from_sorted(ids: Vec<i64>) -> Self {
        debug_assert!(ids.windows(2).all(|w| w[0] < w[1]));
        if ids.len() <= P {
            return IdList::Inline(CompactVec::from_vec(ids));
        }
        let mut pages = BTreeMap::new();
        for chunk in ids.chunks(P) {
            let mut page = CompactVec::with_capacity(P);
            page.extend_copy(chunk);
            pages.insert(chunk[0], page);
            work::add(chunk.len(), 1, chunk.len());
        }
        IdList::Paged(Box::new(Paged {
            len: ids.len(),
            pages,
        }))
    }

    pub fn len(&self) -> usize {
        match self {
            IdList::Inline(ids) => ids.len(),
            IdList::Paged(paged) => paged.len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn contains(&self, id: i64) -> bool {
        match self {
            IdList::Inline(ids) => ids.binary_search(&id).is_ok(),
            IdList::Paged(paged) => paged
                .page_of(id)
                .and_then(|key| paged.pages.get(&key))
                .is_some_and(|page| page.binary_search(&id).is_ok()),
        }
    }

    /// The ids in order, one slice per page
    pub fn pages(&self) -> Pages<'_> {
        match self {
            IdList::Inline(ids) => Pages {
                inline: Some(ids.as_slice()),
                paged: None,
            },
            IdList::Paged(paged) => Pages {
                inline: None,
                paged: Some(paged.pages.values()),
            },
        }
    }

    pub fn iter(&self) -> impl DoubleEndedIterator<Item = i64> + '_ {
        self.pages().flat_map(|page| page.iter().copied())
    }

    /// Appends the ids in order, a page at a time into room reserved once
    pub fn copy_into(&self, out: &mut Vec<i64>) {
        match self {
            IdList::Inline(ids) => out.extend_from_slice(ids),
            IdList::Paged(paged) => {
                out.reserve(paged.len);
                for page in paged.pages.values() {
                    out.extend_from_slice(page);
                }
            }
        }
    }

    /// Whether `id` was not there
    pub fn insert(&mut self, id: i64) -> bool {
        match self {
            IdList::Inline(ids) => {
                let Err(pos) = ids.binary_search(&id) else {
                    return false;
                };
                if ids.len() < P {
                    work::add(ids.len() - pos, 0, ids.len() - pos);
                    ids.insert(pos, id);
                    return true;
                }
                let first = std::mem::take(ids);
                let mut paged = Paged {
                    len: first.len(),
                    pages: BTreeMap::new(),
                };
                paged.pages.insert(first[0], first);
                work::add(0, 1, 0);
                paged.insert::<P>(id);
                *self = IdList::Paged(Box::new(paged));
                true
            }
            IdList::Paged(paged) => paged.insert::<P>(id),
        }
    }

    /// Whether `id` was there
    pub fn remove(&mut self, id: i64) -> bool {
        self.remove_sorted(std::slice::from_ref(&id)) == 1
    }

    /// Removes the strictly increasing `ids`, a page at a time; returns how
    /// many were there
    pub fn remove_sorted(&mut self, ids: &[i64]) -> usize {
        debug_assert!(ids.windows(2).all(|w| w[0] < w[1]));
        let removed = match self {
            IdList::Inline(list) => subtract(list, ids),
            IdList::Paged(paged) => paged.remove_sorted::<P>(ids),
        };
        if let IdList::Paged(paged) = self {
            if paged.len <= P / 2 {
                *self = IdList::Inline(paged.take_one_page());
            }
        }
        removed
    }
}

impl Paged {
    /// Key of the page `id` belongs to
    fn page_of(&self, id: i64) -> Option<i64> {
        work::add(0, 1, 0);
        self.pages
            .range(..=id)
            .next_back()
            .or_else(|| self.pages.first_key_value())
            .map(|(&key, _)| key)
    }

    fn next_key(&self, key: i64) -> Option<i64> {
        work::add(0, 1, 0);
        self.pages
            .range((Bound::Excluded(key), Bound::Unbounded))
            .next()
            .map(|(&k, _)| k)
    }

    fn insert<const P: usize>(&mut self, id: i64) -> bool {
        let inserted = self.insert_into_page::<P>(id);
        if let Some(Some((lower_key, upper_key))) = inserted {
            self.rebalance::<P>(upper_key);
            self.rebalance::<P>(lower_key);
        }
        inserted.is_some()
    }

    /// Some when inserted, with the keys of the two halves when it split
    fn insert_into_page<const P: usize>(&mut self, id: i64) -> Option<Option<(i64, i64)>> {
        work::add(0, 1, 0);
        // Appends go to the last page without a key search
        let tail = self.pages.last_key_value().is_some_and(|(&k, _)| k <= id);
        let found = match tail {
            true => self.pages.iter_mut().next_back(),
            false => match self.pages.range_mut(..=id).next_back() {
                Some(entry) => Some(entry),
                None => self.pages.iter_mut().next(),
            },
        }
        .map(|(&key, page)| (key, page));
        let (key, page) = found?;
        let Err(pos) = page.binary_search(&id) else {
            return None;
        };
        self.len += 1;
        if page.len() < P {
            work::add(page.len() - pos, 0, page.len() - pos);
            page.insert(pos, id);
            return Some(None);
        }
        let next = self.next_key(key);
        let last_page = next.is_none();
        let last = pos == P && last_page;
        let first = pos == 0 && self.pages.first_key_value().is_some_and(|(&k, _)| k == key);
        // Full, with a neighbour that has room: it takes the page's end id,
        // or the new one at that end, rekeyed so the ranges stay in order.
        // Only a page between full neighbours splits
        if let Some(next_key) = next.filter(|k| self.pages.get(k).is_some_and(|p| p.len() < P)) {
            let page = self.pages.get_mut(&key)?;
            let moved = if pos == P {
                id
            } else {
                let moved = page.pop()?;
                work::add(page.len() - pos, 0, page.len() - pos);
                page.insert(pos, id);
                moved
            };
            work::add(0, 2, 0);
            let mut next_page = self.pages.remove(&next_key)?;
            work::add(next_page.len(), 0, next_page.len());
            next_page.insert(0, moved);
            // The first page may hold ids below its key: it takes its first
            // id as key when the moved id would not sort after it
            if moved <= key {
                work::add(0, 2, 0);
                let this = self.pages.remove(&key)?;
                self.pages.insert(this[0], this);
            }
            self.pages.insert(moved, next_page);
            return Some(None);
        }
        let prev = self
            .pages
            .range(..key)
            .next_back()
            .map(|(&k, p)| (k, p.len()));
        if let Some((prev_key, _)) = prev.filter(|(_, len)| *len < P) {
            let mut this = self.pages.remove(&key)?;
            let moved = if pos == 0 {
                id
            } else {
                let moved = this.remove(0);
                work::add(pos, 0, pos);
                this.insert(pos - 1, id);
                moved
            };
            work::add(0, 3, 0);
            self.pages.insert(this[0], this);
            self.pages.get_mut(&prev_key)?.push(moved);
            return Some(None);
        }
        let page = self.pages.get_mut(&key)?;
        // Full, and the id goes past either end of the list: a page of its
        // own, so ids that arrive in order leave full pages behind
        if last || first {
            let mut alone = CompactVec::with_capacity(P);
            alone.push(id);
            work::add(0, 1, 0);
            if pos == 0 && page[0] != key {
                work::add(0, 2, 0);
                let old = self.pages.remove(&key)?;
                self.pages.insert(old[0], old);
            }
            self.pages.insert(id, alone);
            return Some(None);
        }
        // Full: the upper half becomes a page of its own. The last page splits
        // where the id goes when that is in its upper half, so a batch landing
        // near the end of the list leaves the lower page full
        let mid = if last_page && pos > P / 2 { pos } else { P / 2 };
        let mut upper = CompactVec::with_capacity(P);
        upper.extend_copy(&page[mid..]);
        page.truncate(mid);
        let upper_key = upper[0];
        work::add(P - mid, 1, P - mid);
        if id < upper_key {
            work::add(mid - pos, 0, mid - pos);
            page.insert(pos, id);
        } else {
            let at = pos - mid;
            work::add(upper.len() - at, 0, upper.len() - at);
            upper.insert(at, id);
        }
        // The first page may hold ids below its key: it takes its first id
        // as key when the split-off half would sort before it
        let mut lower_key = key;
        if upper_key <= key {
            work::add(0, 2, 0);
            let lower = self.pages.remove(&key)?;
            lower_key = lower[0];
            self.pages.insert(lower_key, lower);
        }
        self.pages.insert(upper_key, upper);
        Some(Some((lower_key, upper_key)))
    }

    fn remove_sorted<const P: usize>(&mut self, ids: &[i64]) -> usize {
        let mut removed = 0;
        let mut i = 0;
        while i < ids.len() {
            let Some(key) = self.page_of(ids[i]) else {
                break;
            };
            let end = self.next_key(key);
            let run = i + ids[i..].partition_point(|&id| end.is_none_or(|end| id < end));
            if let Some(page) = self.pages.get_mut(&key) {
                let gone = subtract(page, &ids[i..run]);
                removed += gone;
                self.len -= gone;
                if gone > 0 {
                    self.rebalance::<P>(key);
                }
            }
            i = run;
        }
        removed
    }

    /// After the page at `key` changed size: drops it when empty, then joins
    /// it, or the pages that met where it was, with each neighbour it fits
    /// with while either is under half full
    fn rebalance<const P: usize>(&mut self, mut key: i64) {
        let under = |len: usize| len < P / 2;
        let len_of = |pages: &BTreeMap<i64, CompactVec<i64>>, key| pages.get(&key).map(|p| p.len());
        if len_of(&self.pages, key) == Some(0) {
            work::add(0, 2, 0);
            self.pages.remove(&key);
            let met = self.pages.range(..key).next_back().map(|(&k, _)| k);
            let Some(met) = met.or_else(|| self.pages.first_key_value().map(|(&k, _)| k)) else {
                return;
            };
            key = met;
        }
        loop {
            let Some(len) = len_of(&self.pages, key) else {
                return;
            };
            work::add(0, 1, 0);
            let prev = self
                .pages
                .range(..key)
                .next_back()
                .map(|(&k, page)| (k, page.len()));
            if let Some((prev_key, prev_len)) = prev {
                if (under(len) || under(prev_len)) && prev_len + len <= P {
                    self.merge(prev_key, key);
                    key = prev_key;
                    continue;
                }
            }
            if let Some(next_key) = self.next_key(key) {
                let next_len = len_of(&self.pages, next_key).unwrap_or(P);
                if (under(len) || under(next_len)) && len + next_len <= P {
                    self.merge(key, next_key);
                    continue;
                }
            }
            return;
        }
    }

    /// Moves the page at `upper` onto the end of the page at `lower`
    fn merge(&mut self, lower: i64, upper: i64) {
        work::add(0, 1, 0);
        let Some(moved) = self.pages.remove(&upper) else {
            return;
        };
        if let Some(page) = self.pages.get_mut(&lower) {
            work::add(moved.len(), 0, moved.len());
            page.extend_copy(&moved);
        }
    }

    /// All ids in the first page, which has room for them
    fn take_one_page(&mut self) -> CompactVec<i64> {
        let mut pages = std::mem::take(&mut self.pages).into_values();
        let mut first = pages.next().unwrap_or_default();
        for page in pages {
            work::add(page.len(), 0, page.len());
            first.extend_copy(&page);
        }
        first
    }
}

/// Removes the strictly increasing `ids` from `list` in one pass from the
/// first one's position; returns how many were there
fn subtract(list: &mut CompactVec<i64>, ids: &[i64]) -> usize {
    if let [id] = ids {
        let Ok(pos) = list.binary_search(id) else {
            return 0;
        };
        work::add(list.len() - pos, 0, list.len() - pos - 1);
        list.remove(pos);
        return 1;
    }
    let Some(&first) = ids.first() else {
        return 0;
    };
    let start = list.binary_search(&first).unwrap_or_else(|pos| pos);
    let slice = list.as_mut_slice();
    let mut keep = start;
    let mut next = 0;
    let mut read = start;
    let mut shifted = 0;
    while read < slice.len() && next < ids.len() {
        let id = slice[read];
        while next < ids.len() && ids[next] < id {
            next += 1;
        }
        if next < ids.len() && ids[next] == id {
            next += 1;
        } else {
            shifted += usize::from(keep != read);
            slice[keep] = id;
            keep += 1;
        }
        read += 1;
    }
    let gone = read - keep;
    let len = slice.len();
    if gone > 0 {
        slice.copy_within(read.., keep);
        shifted += len - read;
    }
    work::add(len - start, 0, shifted);
    list.truncate(len - gone);
    gone
}

/// The whole of one index group's row ids, in order
#[derive(Clone, Copy)]
pub enum GroupIds<'a> {
    List(&'a IdList),
    Slice(&'a [i64]),
}

impl<'a> GroupIds<'a> {
    pub fn len(&self) -> usize {
        match self {
            GroupIds::List(list) => list.len(),
            GroupIds::Slice(ids) => ids.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn pages(&self) -> Pages<'a> {
        match *self {
            GroupIds::List(list) => list.pages(),
            GroupIds::Slice(ids) => Pages {
                inline: Some(ids),
                paged: None,
            },
        }
    }

    pub fn iter(&self) -> impl DoubleEndedIterator<Item = i64> + 'a {
        self.pages().flat_map(|page| page.iter().copied())
    }
}

pub struct Pages<'a> {
    inline: Option<&'a [i64]>,
    paged: Option<btree_map::Values<'a, i64, CompactVec<i64>>>,
}

impl<'a> Iterator for Pages<'a> {
    type Item = &'a [i64];

    fn next(&mut self) -> Option<&'a [i64]> {
        if let Some(ids) = self.inline.take() {
            return Some(ids);
        }
        self.paged.as_mut()?.next().map(|page| page.as_slice())
    }
}

impl DoubleEndedIterator for Pages<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if let Some(ids) = self.inline.take() {
            return Some(ids);
        }
        self.paged.as_mut()?.next_back().map(|page| page.as_slice())
    }
}

/// Work done by id lists on this thread: ids visited, directory operations
/// and ids moved
#[cfg(any(test, feature = "test-failpoints"))]
pub mod work {
    use std::cell::Cell;

    thread_local! {
        static WORK: Cell<(u64, u64, u64)> = const { Cell::new((0, 0, 0)) };
    }

    pub(super) fn add(visited: usize, directory: usize, moved: usize) {
        WORK.with(|w| {
            let (v, d, m) = w.get();
            w.set((v + visited as u64, d + directory as u64, m + moved as u64));
        });
    }

    /// The work since the last call
    pub fn take() -> (u64, u64, u64) {
        WORK.with(|w| w.replace((0, 0, 0)))
    }
}

#[cfg(not(any(test, feature = "test-failpoints")))]
mod work {
    #[inline(always)]
    pub(super) fn add(_visited: usize, _directory: usize, _moved: usize) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn check<const P: usize>(list: &IdList<P>, model: &BTreeSet<i64>) {
        assert_eq!(list.len(), model.len());
        let ids: Vec<i64> = list.iter().collect();
        let expected: Vec<i64> = model.iter().copied().collect();
        assert_eq!(ids, expected);
        let back: Vec<i64> = list.iter().rev().collect();
        assert_eq!(back, expected.iter().rev().copied().collect::<Vec<_>>());
        let IdList::Paged(paged) = list else {
            return;
        };
        assert!(
            paged.len > P / 2,
            "a paged list holds more than half a page"
        );
        let pages: Vec<(i64, &CompactVec<i64>)> =
            paged.pages.iter().map(|(&k, p)| (k, p)).collect();
        for (i, &(key, page)) in pages.iter().enumerate() {
            assert!(!page.is_empty() && page.len() <= P && page.capacity() >= P);
            if i > 0 {
                assert!(page[0] >= key, "a later page starts at its key");
            }
            if let Some(&(next_key, _)) = pages.get(i + 1) {
                assert!(*page.last().unwrap() < next_key);
            }
            if page.len() < P / 2 {
                for j in [i.wrapping_sub(1), i + 1] {
                    if let Some(&(_, other)) = pages.get(j) {
                        assert!(
                            page.len() + other.len() > P,
                            "an under half page fits a neighbour"
                        );
                    }
                }
            }
        }
    }

    /// xorshift, so the test needs no dependency
    fn next(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    #[test]
    fn random_operations_track_a_set() {
        let mut state = 0x9e37_79b9_7f4a_7c15;
        for round in 0..40 {
            let span = [64i64, 1_000, i64::MAX][round % 3];
            let mut list = IdList::<8>::new();
            let mut model = BTreeSet::new();
            for _ in 0..600 {
                let r = next(&mut state);
                let id = (r as i64) % span;
                match r % 7 {
                    0..=3 => assert_eq!(list.insert(id), model.insert(id)),
                    4 => assert_eq!(list.remove(id), model.remove(&id)),
                    _ => {
                        let batch: Vec<i64> = model
                            .range(id..)
                            .copied()
                            .filter(|_| next(&mut state).is_multiple_of(2))
                            .take(12)
                            .collect();
                        for id in &batch {
                            model.remove(id);
                        }
                        assert_eq!(list.remove_sorted(&batch), batch.len());
                    }
                }
                if r.is_multiple_of(13) {
                    list = list.clone();
                }
                check(&list, &model);
                assert_eq!(list.contains(id), model.contains(&id));
            }
        }
    }

    #[test]
    fn extreme_ids_and_inserts_below_the_first_key() {
        let mut list = IdList::<4>::new();
        let mut model = BTreeSet::new();
        for id in (0..40).rev().map(|i| i64::MAX - i * 3) {
            list.insert(id);
            model.insert(id);
        }
        for id in [i64::MIN + 1, -1, 0, 1, i64::MIN + 2] {
            list.insert(id);
            model.insert(id);
        }
        check(&list, &model);
        let low: Vec<i64> = model.iter().copied().take(3).collect();
        assert_eq!(list.remove_sorted(&low), 3);
        for id in &low {
            model.remove(id);
        }
        check(&list, &model);
    }

    #[test]
    fn ids_inserted_in_order_leave_full_pages() {
        for ids in [(0..1_000).collect::<Vec<i64>>(), (0..1_000).rev().collect()] {
            let mut list = IdList::<8>::new();
            let mut model = BTreeSet::new();
            for id in ids {
                list.insert(id);
                model.insert(id);
            }
            check(&list, &model);
            let IdList::Paged(paged) = &list else {
                panic!("1,000 ids are paged");
            };
            assert_eq!(paged.pages.len(), 125, "every page full");
        }
    }

    #[test]
    fn batches_added_out_of_order_past_the_end_fill_their_pages() {
        let mut state = 0x2545_f491_4f6c_dd1d;
        let mut list = IdList::<64>::new();
        let mut model = BTreeSet::new();
        for batch in 0..100i64 {
            let mut ids: Vec<i64> = (batch * 20..batch * 20 + 20).collect();
            for i in (1..ids.len()).rev() {
                ids.swap(i, (next(&mut state) % (i as u64 + 1)) as usize);
            }
            for id in ids {
                list.insert(id);
                model.insert(id);
            }
        }
        check(&list, &model);
        let IdList::Paged(paged) = &list else {
            panic!("2,000 ids are paged");
        };
        assert!(
            paged.pages.len() <= 33,
            "{} pages for 2,000 ids of 64 per page",
            paged.pages.len()
        );
    }

    #[test]
    fn the_pages_around_an_emptied_one_join_when_they_fit() {
        let mut list = IdList::<8>::from_sorted((0..24).collect());
        let mut model: BTreeSet<i64> = (0..24).collect();
        let mut drop = |list: &mut IdList<8>, ids: Vec<i64>| {
            for id in &ids {
                model.remove(id);
            }
            list.remove_sorted(&ids);
            check(list, &model);
        };
        drop(&mut list, (3..8).collect());
        drop(&mut list, (19..24).collect());
        let IdList::Paged(paged) = &list else {
            panic!("14 ids are paged");
        };
        let lens: Vec<usize> = paged.pages.values().map(|p| p.len()).collect();
        assert_eq!(lens, vec![3, 8, 3]);
        drop(&mut list, (8..16).collect());
        let IdList::Paged(paged) = &list else {
            panic!("6 ids stay paged above half a page");
        };
        let lens: Vec<usize> = paged.pages.values().map(|p| p.len()).collect();
        assert_eq!(lens, vec![6], "the two small pages joined");
    }

    /// The first page holds ids below its key; the id it hands its full
    /// neighbour can equal that key or sort below it
    fn first_page_below_its_key_hands_over<const P: usize>(overflow: i64) {
        let n = 2 * P as i64;
        let mut list = IdList::<P>::from_sorted((0..n).collect());
        let mut model: BTreeSet<i64> = (0..n).collect();
        list.remove(n - 1);
        model.remove(&(n - 1));
        for id in 1..P as i64 {
            list.remove(id);
            list.insert(-id);
            model.remove(&id);
            model.insert(-id);
        }
        list.insert(-overflow);
        model.insert(-overflow);
        check(&list, &model);
    }

    #[test]
    fn random_operations_drifting_below_the_first_key_track_a_set() {
        let mut state = 0x51_7cc1_b727_220a;
        for round in 0..20 {
            let mut list = IdList::<8>::from_sorted((0..64).collect());
            let mut model: BTreeSet<i64> = (0..64).collect();
            let mut low = 0i64;
            for _ in 0..800 {
                let r = next(&mut state);
                match r % 10 {
                    0..=4 => {
                        low -= 1 + (r >> 8) as i64 % 3;
                        let id = low + (r >> 16) as i64 % (4 + round as i64);
                        assert_eq!(list.insert(id), model.insert(id));
                    }
                    5..=7 => {
                        let nth = (r >> 8) as usize % model.len().max(1);
                        if let Some(&id) = model.iter().nth(nth) {
                            assert!(list.remove(id));
                            model.remove(&id);
                        }
                    }
                    _ => {
                        let id = low + (r >> 8) as i64 % 200;
                        assert_eq!(list.insert(id), model.insert(id));
                    }
                }
                check(&list, &model);
            }
        }
    }

    #[test]
    fn a_first_page_hands_its_neighbour_an_id_at_or_below_its_key() {
        first_page_below_its_key_hands_over::<512>(512);
        first_page_below_its_key_hands_over::<8>(8);
        first_page_below_its_key_hands_over::<8>(100);
    }

    #[test]
    fn a_list_shrinks_back_inline_into_its_first_page() {
        let mut list = IdList::<8>::from_sorted((0..100).collect());
        let doomed: Vec<i64> = (0..97).collect();
        assert_eq!(list.remove_sorted(&doomed), 97);
        let IdList::Inline(ids) = &list else {
            panic!("three ids are inline");
        };
        assert_eq!(ids.as_slice(), &[97, 98, 99]);
    }

    #[test]
    fn removing_a_batch_from_a_large_list_does_the_work_of_the_batch() {
        let mut list = IdList::<256>::from_sorted((0..1_000_000).collect());
        work::take();
        let front: Vec<i64> = (0..2_000).collect();
        assert_eq!(list.remove_sorted(&front), 2_000);
        let (visited, directory, moved) = work::take();
        assert!(visited <= 2_000 + 2 * 256, "visited {visited}");
        assert!(directory <= 6 * (2_000 / 256 + 2), "directory {directory}");
        assert!(moved <= 2 * 256, "moved {moved}");

        let spread: Vec<i64> = (0..2_000).map(|i| 2_000 + i * 499).collect();
        assert_eq!(list.remove_sorted(&spread), 2_000);
        let (visited, directory, moved) = work::take();
        assert!(visited <= 2_000 * 256, "visited {visited}");
        assert!(directory <= 6 * 2_000, "directory {directory}");
        assert!(moved <= 2_000 * 256, "moved {moved}");
    }
}
