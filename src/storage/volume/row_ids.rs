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

//! The row id of each position of a volume, read through accessors only, so
//! the form the ids are held in stays this module's choice

/// The row ids of a volume by position
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowIds(Repr);

#[derive(Clone, Debug, PartialEq, Eq)]
enum Repr {
    /// `start`, `start + 1`, ... for `len` positions
    Range { start: i64, len: usize },
    /// Ascending ids within 2^32 of the smallest, packed by blocks of 64
    Blocks(Box<Blocks>),
    /// `base + offsets[pos]`: ids within 2^32 of the smallest, four bytes each
    Offsets { base: i64, offsets: Vec<u32> },
    /// Any other ids, eight bytes each
    Full(Vec<i64>),
}

impl From<Vec<i64>> for RowIds {
    /// Ids that run without a gap are kept as their range; ids that fit in
    /// 32 bits above the smallest are packed by blocks when they ascend and
    /// that takes less, else kept as offsets from the smallest
    fn from(ids: Vec<i64>) -> Self {
        let Some(&start) = ids.first() else {
            return Self(Repr::Full(ids));
        };
        if is_run(&ids) {
            return Self::range(start, ids.len());
        }
        let (min, max) = ids
            .iter()
            .fold((start, start), |(min, max), &id| (min.min(id), max.max(id)));
        if u32::try_from(i128::from(max) - i128::from(min)).is_err() {
            return Self(Repr::Full(ids));
        }
        let blocks = ids
            .windows(2)
            .all(|w| w[0] < w[1])
            .then(|| Blocks::new(&ids, min, max));
        match blocks.flatten() {
            Some(blocks) if blocks.memory_size() < ids.len() * 4 => Self(Repr::Blocks(blocks)),
            _ => Self(Repr::Offsets {
                base: min,
                offsets: ids.iter().map(|&id| (id - min) as u32).collect(),
            }),
        }
    }
}

/// Positions per block of `Blocks`
const BLOCK: usize = 64;
/// Bits of a block header that hold where its offsets start in `data`
const START_BITS: u32 = 26;

/// Ascending ids packed by blocks of 64 positions: each id as its offset
/// from the first id of its block, at the block's width
#[derive(Clone, Debug, PartialEq, Eq)]
struct Blocks {
    min: i64,
    max: i64,
    len: usize,
    /// Per block, one word: the offset of its smallest id from `min` (bits
    /// 0..32), its first bit in `data` (32..58) and its width (58..64)
    heads: Box<[u64]>,
    /// The offsets, then 8 spare bytes so that any read takes 8 bytes
    data: Box<[u8]>,
}

impl Blocks {
    /// None when the packed offsets would not fit the header's bit start
    fn new(ids: &[i64], min: i64, max: i64) -> Option<Box<Blocks>> {
        let mut heads = Vec::with_capacity(ids.len().div_ceil(BLOCK));
        let mut bit = 0u64;
        for block in ids.chunks(BLOCK) {
            let (lo, hi) = (block[0], block[block.len() - 1]);
            let width = u64::from(64 - ((hi - lo) as u64).leading_zeros());
            if bit >= 1 << START_BITS {
                return None;
            }
            heads.push((lo - min) as u64 | bit << 32 | width << (32 + START_BITS));
            bit += block.len() as u64 * width;
        }
        let mut data = vec![0u8; bit.div_ceil(8) as usize + 8];
        for (block, &head) in ids.chunks(BLOCK).zip(&heads) {
            let (lo, start, width) = Self::parts(head, min);
            for (k, &id) in block.iter().enumerate() {
                let at = start + k as u64 * u64::from(width);
                let bytes = &mut data[(at >> 3) as usize..][..8];
                let mut word = [0u8; 8];
                word.copy_from_slice(bytes);
                let packed = u64::from_le_bytes(word) | ((id - lo) as u64) << (at & 7);
                bytes.copy_from_slice(&packed.to_le_bytes());
            }
        }
        Some(Box::new(Blocks {
            min,
            max,
            len: ids.len(),
            heads: heads.into_boxed_slice(),
            data: data.into_boxed_slice(),
        }))
    }

    /// A block's first id, first bit and width
    #[inline(always)]
    fn parts(head: u64, min: i64) -> (i64, u64, u32) {
        let lo = min + i64::from(head as u32);
        let start = (head >> 32) & ((1 << START_BITS) - 1);
        (lo, start, (head >> (32 + START_BITS)) as u32)
    }

    /// The bits from bit `at` of `data` on, at least 57 of them, to be masked
    /// to a width of at most 32
    #[inline(always)]
    fn read(&self, at: u64) -> u64 {
        let mut word = [0u8; 8];
        word.copy_from_slice(&self.data[(at >> 3) as usize..][..8]);
        u64::from_le_bytes(word) >> (at & 7)
    }

    #[inline(always)]
    fn at(&self, pos: usize) -> i64 {
        match self.get(pos) {
            Some(id) => id,
            None => past_end(pos, self.len),
        }
    }

    #[inline(always)]
    fn get(&self, pos: usize) -> Option<i64> {
        if pos >= self.len {
            return None;
        }
        let (lo, start, width) = Self::parts(self.heads[pos / BLOCK], self.min);
        let offset = self.read(start + (pos % BLOCK) as u64 * u64::from(width));
        Some(lo + (offset & !(u64::MAX << width)) as i64)
    }

    /// Unpacks `positions` a block at a time, the block's parts read once
    fn copy_into(&self, positions: std::ops::Range<usize>, out: &mut Vec<i64>) {
        assert!(positions.start <= positions.end && positions.end <= self.len);
        out.reserve(positions.len());
        let mut pos = positions.start;
        while pos < positions.end {
            let (lo, start, width) = Self::parts(self.heads[pos / BLOCK], self.min);
            let end = (pos / BLOCK * BLOCK + BLOCK).min(positions.end);
            let (mask, step) = (!(u64::MAX << width), u64::from(width));
            let first = start + (pos % BLOCK) as u64 * step;
            out.extend(
                (0..(end - pos) as u64).map(|k| lo + (self.read(first + k * step) & mask) as i64),
            );
            pos = end;
        }
    }

    /// The first of `positions` whose id is `id` or larger, in ids that ascend
    fn first_at_least(&self, positions: std::ops::Range<usize>, id: i64) -> usize {
        assert!(positions.start <= positions.end && positions.end <= self.len);
        let (mut lo, mut hi) = (positions.start, positions.end);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.at(mid) < id {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }

    /// The position of `id` in ids that ascend: the block by its first id,
    /// then within it, each from where the id would lie if ids were spread
    /// evenly
    fn search_ascending(&self, id: i64) -> Option<usize> {
        if id > self.max {
            return None;
        }
        let offset = u64::try_from(id.checked_sub(self.min)?).ok()?;
        let span = (self.max - self.min) as u64 + 1;
        let first_of = |b: usize| u64::from(self.heads[b] as u32);
        let blocks = self.heads.len();
        let block = last_at_most(blocks, offset * blocks as u64 / span, offset, first_of)?;
        let (_, start, width) = Self::parts(self.heads[block], self.min);
        let len = (self.len - block * BLOCK).min(BLOCK);
        let end = match block + 1 < blocks {
            true => first_of(block + 1),
            false => span,
        };
        let (target, mask) = (offset - first_of(block), !(u64::MAX << width));
        let offset_at = |k: usize| self.read(start + k as u64 * u64::from(width)) & mask;
        let guess = target * len as u64 / (end - first_of(block));
        let k = last_at_most(len, guess, target, offset_at)?;
        (offset_at(k) == target).then_some(block * BLOCK + k)
    }

    /// Heap bytes held, the boxed struct included
    fn memory_size(&self) -> usize {
        size_of::<Blocks>() + self.heads.len() * 8 + self.data.len()
    }
}

/// The last of `0..len` whose ascending `value` is `target` or less, None
/// when none is: galloping from `guess` away from the side it misses on,
/// then halving the bracket found
#[inline(always)]
fn last_at_most(
    len: usize,
    guess: u64,
    target: u64,
    value: impl Fn(usize) -> u64,
) -> Option<usize> {
    let guess = (guess as usize).min(len - 1);
    let mut step = 1;
    let (mut below, mut above) = if value(guess) <= target {
        let mut below = guess;
        while below + step < len && value(below + step) <= target {
            below += step;
            step *= 2;
        }
        (below, (below + step).min(len))
    } else {
        let mut above = guess;
        loop {
            let next = above.saturating_sub(step);
            if value(next) <= target {
                break (next, above);
            }
            if next == 0 {
                return None;
            }
            above = next;
            step *= 2;
        }
    };
    while above - below > 1 {
        let mid = below + (above - below) / 2;
        if value(mid) <= target {
            below = mid;
        } else {
            above = mid;
        }
    }
    Some(below)
}

#[cold]
#[inline(never)]
fn past_end(pos: usize, len: usize) -> ! {
    panic!("row id position {pos} past {len}")
}

/// Whether each id is one more than the one before it
fn is_run(ids: &[i64]) -> bool {
    ids.windows(2).all(|w| w[0].checked_add(1) == Some(w[1]))
}

static NONE: RowIds = RowIds(Repr::Range { start: 0, len: 0 });

impl RowIds {
    /// No ids, for a lookup that finds no volume
    pub fn none() -> &'static RowIds {
        &NONE
    }

    /// `len` ids from `start` on; the caller checked that the last one fits
    pub fn range(start: i64, len: usize) -> Self {
        debug_assert!(len == 0 || start.checked_add(len as i64 - 1).is_some());
        Self(Repr::Range { start, len })
    }

    #[inline]
    pub fn len(&self) -> usize {
        match &self.0 {
            Repr::Range { len, .. } => *len,
            Repr::Offsets { offsets, .. } => offsets.len(),
            Repr::Blocks(blocks) => blocks.len,
            Repr::Full(ids) => ids.len(),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The id at `pos`; panics past the end, as an index does
    #[inline(always)]
    pub fn at(&self, pos: usize) -> i64 {
        match &self.0 {
            Repr::Offsets { base, offsets } => base + i64::from(offsets[pos]),
            Repr::Blocks(blocks) => blocks.at(pos),
            Repr::Full(ids) => ids[pos],
            Repr::Range { start, len } => {
                if pos >= *len {
                    past_end(pos, *len);
                }
                start + pos as i64
            }
        }
    }

    /// The id at `pos`, None past the end
    #[inline]
    pub fn get(&self, pos: usize) -> Option<i64> {
        match &self.0 {
            Repr::Range { start, len } => (pos < *len).then(|| start + pos as i64),
            Repr::Offsets { base, offsets } => {
                offsets.get(pos).map(|&offset| base + i64::from(offset))
            }
            Repr::Blocks(blocks) => blocks.get(pos),
            Repr::Full(ids) => ids.get(pos).copied(),
        }
    }

    pub fn first(&self) -> Option<i64> {
        self.get(0)
    }

    pub fn last(&self) -> Option<i64> {
        self.len().checked_sub(1).and_then(|pos| self.get(pos))
    }

    /// The largest id held
    pub fn max(&self) -> Option<i64> {
        match &self.0 {
            Repr::Range { .. } => self.last(),
            Repr::Offsets { base, offsets } => {
                offsets.iter().max().map(|&offset| base + i64::from(offset))
            }
            Repr::Blocks(blocks) => Some(blocks.max),
            Repr::Full(ids) => ids.iter().max().copied(),
        }
    }

    /// The ids in position order
    #[inline]
    pub fn iter(&self) -> Iter<'_> {
        self.iter_range(0..self.len())
    }

    /// The ids of `positions`, in order; panics past the end, as a slice does
    pub fn iter_range(&self, positions: std::ops::Range<usize>) -> Iter<'_> {
        Iter(match &self.0 {
            Repr::Range { start, len } => {
                assert!(positions.start <= positions.end && positions.end <= *len);
                // Wrapping only where nothing is left: the empty range past i64::MAX
                IterRepr::Range {
                    next: start.wrapping_add(positions.start as i64),
                    left: positions.len(),
                }
            }
            Repr::Offsets { base, offsets } => IterRepr::Offsets {
                base: *base,
                offsets: offsets[positions].iter(),
            },
            Repr::Blocks(blocks) => {
                assert!(positions.start <= positions.end && positions.end <= blocks.len);
                IterRepr::Blocks { blocks, positions }
            }
            Repr::Full(ids) => IterRepr::Full(ids[positions].iter()),
        })
    }

    /// The ids of `positions` as a new array; panics past the end, as a
    /// slice does
    pub fn to_vec(&self, positions: std::ops::Range<usize>) -> Vec<i64> {
        let mut ids = Vec::new();
        self.copy_into(positions, &mut ids);
        ids
    }

    /// Replaces `out` with the ids of `positions`, for a loop over a row
    /// group to index a plain array; panics past the end, as a slice does
    pub fn copy_into(&self, positions: std::ops::Range<usize>, out: &mut Vec<i64>) {
        out.clear();
        match &self.0 {
            Repr::Range { .. } => out.extend(self.iter_range(positions)),
            Repr::Offsets { base, offsets } => out.extend(
                offsets[positions]
                    .iter()
                    .map(|&offset| base + i64::from(offset)),
            ),
            Repr::Blocks(blocks) => blocks.copy_into(positions, out),
            Repr::Full(ids) => out.extend_from_slice(&ids[positions]),
        }
    }

    /// Whether every id is larger than the one before it
    pub fn ascends(&self) -> bool {
        match &self.0 {
            Repr::Range { .. } | Repr::Blocks(_) => true,
            Repr::Offsets { offsets, .. } => offsets.windows(2).all(|w| w[0] < w[1]),
            Repr::Full(ids) => ids.windows(2).all(|w| w[0] < w[1]),
        }
    }

    /// `(start, len)` when the ids run without a gap
    pub fn as_range(&self) -> Option<(i64, usize)> {
        match &self.0 {
            Repr::Range { start, len } => Some((*start, *len)),
            Repr::Offsets { .. } | Repr::Blocks(_) | Repr::Full(_) => None,
        }
    }

    /// Positions sorted by id, None when the ids ascend already
    pub fn sorted_order(&self) -> Option<Box<[u32]>> {
        if self.ascends() {
            return None;
        }
        match &self.0 {
            Repr::Range { .. } | Repr::Blocks(_) => None,
            Repr::Offsets { offsets, .. } => {
                let mut order: Vec<u32> = (0..offsets.len() as u32).collect();
                order.sort_unstable_by_key(|&i| offsets[i as usize]);
                Some(order.into_boxed_slice())
            }
            Repr::Full(ids) => Some(super::writer::row_order_of(ids)),
        }
    }

    /// The position of `id` in ids that ascend
    pub fn search_ascending(&self, id: i64) -> Option<usize> {
        match &self.0 {
            Repr::Range { start, len } => {
                let pos = usize::try_from(id.checked_sub(*start)?).ok()?;
                (pos < *len).then_some(pos)
            }
            Repr::Offsets { base, offsets } => {
                let offset = u32::try_from(id.checked_sub(*base)?).ok()?;
                if offset > *offsets.last()? {
                    return None;
                }
                offsets.binary_search(&offset).ok()
            }
            Repr::Blocks(blocks) => blocks.search_ascending(id),
            Repr::Full(ids) => {
                if id < *ids.first()? || id > *ids.last()? {
                    return None;
                }
                ids.binary_search(&id).ok()
            }
        }
    }

    /// The first of `positions` whose id is `id` or larger, `positions.end`
    /// when none is, in ids that ascend; panics past the end, as a slice does
    pub fn first_at_least(&self, positions: std::ops::Range<usize>, id: i64) -> usize {
        match &self.0 {
            Repr::Range { start, len } => {
                assert!(positions.start <= positions.end && positions.end <= *len);
                let from = i128::from(id) - i128::from(*start);
                (from.clamp(positions.start as i128, positions.end as i128)) as usize
            }
            Repr::Offsets { base, offsets } => {
                let below = &offsets[positions.clone()];
                let found = match u32::try_from(i128::from(id) - i128::from(*base)) {
                    Ok(offset) => below.partition_point(|&o| o < offset),
                    Err(_) if id < *base => 0,
                    Err(_) => below.len(),
                };
                positions.start + found
            }
            Repr::Blocks(blocks) => blocks.first_at_least(positions, id),
            Repr::Full(ids) => {
                positions.start + ids[positions].partition_point(|&other| other < id)
            }
        }
    }

    /// Writes every id as 8 little-endian bytes
    pub fn write_le<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        if let Repr::Full(ids) = &self.0 {
            return super::format::write_i64_le(w, ids);
        }
        let mut bytes = Vec::with_capacity(8 * 1024);
        for start in (0..self.len()).step_by(1024) {
            bytes.clear();
            for id in self.iter_range(start..(start + 1024).min(self.len())) {
                bytes.extend_from_slice(&id.to_le_bytes());
            }
            w.write_all(&bytes)?;
        }
        Ok(())
    }

    /// Bytes the ids hold on the heap
    pub fn memory_size(&self) -> usize {
        match &self.0 {
            Repr::Range { .. } => 0,
            Repr::Offsets { offsets, .. } => offsets.capacity() * 4,
            Repr::Blocks(blocks) => blocks.memory_size(),
            Repr::Full(ids) => ids.capacity() * 8,
        }
    }
}

/// The ids of one row group as a plain array, loaded once per group so a
/// loop over its rows reads them without matching on the form they are held in
#[derive(Default)]
pub struct GroupRowIds {
    start: usize,
    ids: Vec<i64>,
}

impl GroupRowIds {
    /// Holds the ids of `positions` of `ids`, reusing the array
    pub fn load(&mut self, ids: &RowIds, positions: std::ops::Range<usize>) {
        self.start = positions.start;
        ids.copy_into(positions, &mut self.ids);
    }

    /// The id at volume position `pos`, one of the loaded positions
    #[inline(always)]
    pub fn at(&self, pos: usize) -> i64 {
        self.ids[pos - self.start]
    }

    /// Whether `pos` is one of the loaded positions
    #[inline(always)]
    pub fn holds(&self, pos: usize) -> bool {
        pos.wrapping_sub(self.start) < self.ids.len()
    }
}

/// The ids of a range of positions, in order
pub struct Iter<'a>(IterRepr<'a>);

enum IterRepr<'a> {
    /// `left` ids from `next` on, counted so the last id may be `i64::MAX`
    Range {
        next: i64,
        left: usize,
    },
    Offsets {
        base: i64,
        offsets: std::slice::Iter<'a, u32>,
    },
    Blocks {
        blocks: &'a Blocks,
        positions: std::ops::Range<usize>,
    },
    Full(std::slice::Iter<'a, i64>),
}

impl Iterator for Iter<'_> {
    type Item = i64;

    #[inline]
    fn next(&mut self) -> Option<i64> {
        match &mut self.0 {
            IterRepr::Range { next, left } => {
                *left = left.checked_sub(1)?;
                let id = *next;
                *next = next.wrapping_add(1);
                Some(id)
            }
            IterRepr::Offsets { base, offsets } => {
                offsets.next().map(|&offset| *base + i64::from(offset))
            }
            IterRepr::Blocks { blocks, positions } => positions.next().map(|pos| blocks.at(pos)),
            IterRepr::Full(ids) => ids.next().copied(),
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<i64> {
        match &mut self.0 {
            IterRepr::Range { next, left } => {
                if n >= *left {
                    *left = 0;
                    return None;
                }
                *left -= n;
                *next += n as i64;
                self.next()
            }
            IterRepr::Offsets { base, offsets } => {
                offsets.nth(n).map(|&offset| *base + i64::from(offset))
            }
            IterRepr::Blocks { blocks, positions } => positions.nth(n).map(|pos| blocks.at(pos)),
            IterRepr::Full(ids) => ids.nth(n).copied(),
        }
    }
}

impl ExactSizeIterator for Iter<'_> {
    #[inline]
    fn len(&self) -> usize {
        match &self.0 {
            IterRepr::Range { left, .. } => *left,
            IterRepr::Offsets { offsets, .. } => offsets.len(),
            IterRepr::Blocks { positions, .. } => positions.len(),
            IterRepr::Full(ids) => ids.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all(ids: &RowIds) -> Vec<i64> {
        ids.iter().collect()
    }

    #[test]
    fn ids_without_a_gap_are_held_as_their_range() {
        let ids = RowIds::from(vec![5, 6, 7]);
        assert_eq!(ids.as_range(), Some((5, 3)));
        assert_eq!(ids.memory_size(), 0);
        assert_eq!(all(&ids), [5, 6, 7]);
        assert_eq!(ids.iter_range(1..3).collect::<Vec<_>>(), [6, 7]);
        assert_eq!((ids.at(2), ids.get(3), ids.max()), (7, None, Some(7)));
        assert_eq!(ids.search_ascending(6), Some(1));
        assert_eq!(ids.search_ascending(8), None);
        assert_eq!(ids.search_ascending(i64::MIN), None);
        assert!(ids.ascends() && ids.sorted_order().is_none());
        let top = RowIds::from(vec![i64::MAX - 1, i64::MAX]);
        assert_eq!(top.as_range(), Some((i64::MAX - 1, 2)));
        assert_eq!(all(&top), [i64::MAX - 1, i64::MAX]);
        assert_eq!(top.iter().nth(1), Some(i64::MAX));
        let last = RowIds::from(vec![i64::MAX]);
        assert_eq!(
            last.iter_range(1..1).count(),
            0,
            "the empty range past i64::MAX"
        );
        assert!(last.to_vec(1..1).is_empty());
    }

    #[test]
    fn ids_within_32_bits_of_the_smallest_take_four_bytes_each() {
        let gap = RowIds::from(vec![-3, 10, 4_000_000_000]);
        assert_eq!(gap.memory_size(), 12);
        let few = RowIds::from(vec![0, 2, 4, 6, 8, 10]);
        assert_eq!(few.memory_size(), 24, "blocks cost more than offsets here");
        assert_eq!(all(&gap), [-3, 10, 4_000_000_000]);
        assert_eq!(
            (gap.at(1), gap.get(3), gap.max()),
            (10, None, Some(4_000_000_000))
        );
        assert_eq!(gap.to_vec(1..3), [10, 4_000_000_000]);
        assert_eq!(
            (gap.search_ascending(10), gap.search_ascending(11)),
            (Some(1), None)
        );
        assert_eq!(gap.search_ascending(i64::MIN), None);
        assert!(gap.ascends() && gap.sorted_order().is_none());
        let unordered = RowIds::from(vec![9, 2, 5]);
        assert_eq!(unordered.memory_size(), 12);
        assert!(!unordered.ascends());
        assert_eq!(unordered.sorted_order().as_deref(), Some(&[1u32, 2, 0][..]));
        assert_eq!(
            (unordered.first(), unordered.last(), unordered.max()),
            (Some(9), Some(5), Some(9))
        );
        assert_eq!(unordered.iter().nth(1), Some(2));
        let mut bytes = Vec::new();
        unordered.write_le(&mut bytes).unwrap();
        assert_eq!(bytes[8..16], 2i64.to_le_bytes());
    }

    #[test]
    fn the_first_id_at_least_a_bound_is_found_within_the_positions() {
        let forms = [
            RowIds::from(vec![10, 11, 12, 13]),
            RowIds::from(vec![10, 12, 13, 20]),
            RowIds::from(vec![i64::MIN, 10, 12, i64::MAX]),
        ];
        for ids in &forms {
            assert_eq!(ids.first_at_least(0..4, i64::MIN), 0, "{ids:?}");
            assert_eq!(ids.first_at_least(1..3, 0), 1, "{ids:?}");
            assert_eq!(ids.first_at_least(1..3, 14), 3, "{ids:?}");
            assert_eq!(ids.first_at_least(2..2, 0), 2, "{ids:?}");
        }
        let at_least = |ids: &RowIds, id| ids.first_at_least(0..4, id);
        assert_eq!([12, 14].map(|id| at_least(&forms[0], id)), [2, 4]);
        assert_eq!(
            [11, 13, i64::MAX].map(|id| at_least(&forms[1], id)),
            [1, 2, 4]
        );
        assert_eq!([11, i64::MAX].map(|id| at_least(&forms[2], id)), [2, 3]);
    }

    #[test]
    fn ascending_ids_with_gaps_are_packed_by_blocks_of_64() {
        let mut expected = Vec::new();
        let mut id = -5_000i64;
        for pos in 0..10_000u64 {
            id += 1 + (pos * 7919 % 97) as i64 + if pos % 3000 == 2999 { 1 << 30 } else { 0 };
            expected.push(id);
        }
        let ids = RowIds::from(expected.clone());
        assert!(
            ids.memory_size() < expected.len() * 4,
            "{}",
            ids.memory_size()
        );
        assert_eq!(all(&ids), expected);
        assert_eq!(
            ids.iter_range(63..130).collect::<Vec<_>>(),
            expected[63..130]
        );
        assert_eq!(ids.iter_range(5..9).nth(2), Some(expected[7]));
        assert_eq!(ids.to_vec(100..9_999), expected[100..9_999]);
        assert_eq!(
            (ids.get(10_000), ids.max()),
            (None, expected.last().copied())
        );
        assert!(ids.ascends() && ids.sorted_order().is_none() && ids.as_range().is_none());
        for (pos, &id) in expected.iter().enumerate() {
            assert_eq!(ids.at(pos), id);
            assert_eq!(ids.search_ascending(id), Some(pos), "{id}");
            assert_eq!(
                ids.search_ascending(id + 1),
                (expected.get(pos + 1) == Some(&(id + 1))).then_some(pos + 1)
            );
            assert_eq!(ids.first_at_least(0..10_000, id + 1), pos + 1);
        }
        assert_eq!(ids.search_ascending(i64::MIN), None);
        assert_eq!(ids.search_ascending(i64::MAX), None);
        assert_eq!(ids.first_at_least(200..300, i64::MIN), 200);
        let mut bytes = Vec::new();
        ids.write_le(&mut bytes).unwrap();
        assert_eq!(bytes[8 * 77..8 * 78], expected[77].to_le_bytes());
        let mut group = GroupRowIds::default();
        group.load(&ids, 65..4_000);
        assert_eq!(group.at(3_999), expected[3_999]);
    }

    #[test]
    fn packed_ids_are_found_where_they_are_spread_unevenly() {
        let dense = (0..5_000i64).map(|k| 10 + 3 * k + k % 3);
        let sparse = (1..5_000i64).map(|k| 20_000 + k * 60_000 + (k * 7919) % 1_000);
        let expected: Vec<i64> = dense.chain(sparse).chain([4_000_000_000]).collect();
        let ids = RowIds::from(expected.clone());
        assert!(
            ids.memory_size() < expected.len() * 4,
            "{}",
            ids.memory_size()
        );
        for (pos, &id) in expected.iter().enumerate() {
            assert_eq!(ids.search_ascending(id), Some(pos), "{id}");
            if expected.get(pos + 1) != Some(&(id + 1)) {
                assert_eq!(ids.search_ascending(id + 1), None, "{}", id + 1);
            }
        }
        assert_eq!(ids.search_ascending(9), None);
    }

    #[test]
    #[should_panic(expected = "row id position 4096 past 4096")]
    fn packed_ids_panic_past_the_end_inside_the_last_block() {
        RowIds::from((0..4096).map(|pos| pos * 3).collect::<Vec<i64>>()).at(4096);
    }

    #[test]
    fn ids_wider_than_32_bits_keep_eight_bytes_each() {
        let wide = RowIds::from(vec![i64::MIN, 0, i64::MAX]);
        assert_eq!(wide.memory_size(), 24);
        assert_eq!(all(&wide), [i64::MIN, 0, i64::MAX]);
        assert_eq!(wide.search_ascending(0), Some(1));
        let unordered = RowIds::from(vec![i64::MAX, i64::MIN]);
        assert_eq!(unordered.sorted_order().as_deref(), Some(&[1u32, 0][..]));
        assert_eq!(RowIds::from(Vec::new()).first(), None);
    }
}
