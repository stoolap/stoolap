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
    /// `base + offsets[pos]`: ids within 2^32 of the smallest, four bytes each
    Offsets { base: i64, offsets: Vec<u32> },
    /// Any other ids, eight bytes each
    Full(Vec<i64>),
}

impl From<Vec<i64>> for RowIds {
    /// Ids that run without a gap are kept as their range, ids that fit in
    /// 32 bits above the smallest as offsets from it
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
        match u32::try_from(i128::from(max) - i128::from(min)) {
            Ok(_) => Self(Repr::Offsets {
                base: min,
                offsets: ids.iter().map(|&id| (id - min) as u32).collect(),
            }),
            Err(_) => Self(Repr::Full(ids)),
        }
    }
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
            Repr::Full(ids) => out.extend_from_slice(&ids[positions]),
        }
    }

    /// Whether every id is larger than the one before it
    pub fn ascends(&self) -> bool {
        match &self.0 {
            Repr::Range { .. } => true,
            Repr::Offsets { offsets, .. } => offsets.windows(2).all(|w| w[0] < w[1]),
            Repr::Full(ids) => ids.windows(2).all(|w| w[0] < w[1]),
        }
    }

    /// `(start, len)` when the ids run without a gap
    pub fn as_range(&self) -> Option<(i64, usize)> {
        match &self.0 {
            Repr::Range { start, len } => Some((*start, *len)),
            Repr::Offsets { .. } | Repr::Full(_) => None,
        }
    }

    /// Positions sorted by id, None when the ids ascend already
    pub fn sorted_order(&self) -> Option<Box<[u32]>> {
        if self.ascends() {
            return None;
        }
        match &self.0 {
            Repr::Range { .. } => None,
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
            Repr::Full(ids) => {
                if id < *ids.first()? || id > *ids.last()? {
                    return None;
                }
                ids.binary_search(&id).ok()
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
